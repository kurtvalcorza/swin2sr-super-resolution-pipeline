"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): evaluation with the
per-category breakdown, a one-epoch adaptation of the last stage on a dozen drawn scenes, the artifact round trip,
the loader's scope check, the transactional guarantee and — where CUDA is visible — the same path on the
accelerator. Skipped when the weights are absent."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw

from swin2sr_super_resolution_pipeline import DEFAULT_WEIGHTS_DIR, WEIGHT_FILE, Swin2SRPipeline, degrade

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

COLOURS = ["red", "green", "blue", "yellow", "white", "black"]


def _scene(i, size=64):
    image = Image.new("RGB", (size, size), (135, 206, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, size * 2 // 3, size, size], fill=(60, 179, 75))
    draw.ellipse([size // 4, size // 6, size * 3 // 4, size * 2 // 3], fill=COLOURS[i % 6])
    draw.line([(0, i % size), (size, (i * 7) % size)], fill=(200, 30, 30), width=2)
    image.putpixel((i % size, 0), (i % 256, 0, 0))
    return image


@pytest.fixture(scope="module")
def records():
    return [{"id": f"s{i:02d}", "hr": _scene(i), "lr": degrade(_scene(i)), "category": COLOURS[i % 6]} for i in range(16)]


@pytest.fixture(scope="module")
def pipe():
    return Swin2SRPipeline.from_pretrained(device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)


def test_evaluate_scores_the_frozen_model_with_the_breakdown(pipe, records):
    metrics = pipe.evaluate(records[:8])
    assert metrics["n"] == 8 and 10.0 < metrics["psnr"] < 60.0 and 0.0 < metrics["ssim"] <= 1.0 and metrics["adapted"] is False
    assert set(metrics["per_category"]) == set(COLOURS) and metrics["verdict"] == "measured-small-sample"
    assert pipe.weight_sha256 is not None and len(pipe.weight_sha256) == 64
    out = pipe.upscale(records[0]["lr"])["image"]
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, records, tmp_path):
    result = pipe.adapt(records[:12], records[12:], epochs=1, trainable_stages=1, batch_size=4)
    assert result["n_trainable"] == 2_463_011  # last stage + conv_after_body + upsample + final_convolution
    assert result["history"][0]["note"] == "frozen model" and result["history"][1]["train_loss"] > 0.0
    assert set(result["history"][1]["val"]) == {"psnr", "ssim", "n"}
    assert all(n.startswith(("swin2sr.encoder.stages.5.", "swin2sr.conv_after_body", "upsample", "final_convolution")) for n in result["trainable_names"])
    assert not any(n.startswith(("swin2sr.embeddings", "swin2sr.first_convolution", "swin2sr.encoder.stages.0.")) for n in result["trainable_names"])
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"]) and manifest["base"]["weight_sha256"] == pipe.weight_sha256
    reloaded = Swin2SRPipeline.from_artifact(artifact, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    a = [pipe.upscale(r["lr"])["image"] for r in records[:4]]
    b = [reloaded.upscale(r["lr"])["image"] for r in records[:4]]
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert reloaded.adapter["best_epoch"] == result["best_epoch"]
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, records, tmp_path):
    result = pipe.adapt(records[:12], None, epochs=2, trainable_stages=1, batch_size=4)
    assert result["best_epoch"] == 2 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 3
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = Swin2SRPipeline.from_artifact(artifact, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])
    assert reloaded.adapter["trainable_stages"] == 1


def test_adapt_refuses_mixed_lr_sizes(pipe, records):
    mixed = [*records[:11], {"id": "big", "hr": _scene(99, 96), "lr": degrade(_scene(99, 96)), "category": "red"}]
    with pytest.raises(ValueError, match="share one LR size"):
        pipe.adapt(mixed, None, epochs=1, trainable_stages=1, batch_size=4)


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(pipe, records, tmp_path):
    from safetensors.torch import load_file, save_file

    pipe.adapt(records[:12], None, epochs=1, trainable_stages=1, batch_size=4)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        Swin2SRPipeline.from_artifact(fewer, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["zz.extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    files = [{**manifest["files"][0], "bytes": (extra / "adapter.safetensors").stat().st_size, "sha256": digest}]
    (extra / "manifest.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        Swin2SRPipeline.from_artifact(extra, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    other = tmp_path / "other_stages"
    shutil.copytree(artifact, other)
    (other / "manifest.json").write_text(json.dumps({**manifest, "adapter": {**manifest["adapter"], "trainable_stages": 2}}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        Swin2SRPipeline.from_artifact(other, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe, records):
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(records[:12], None, epochs=2, trainable_stages=1, batch_size=4, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert not any(p.requires_grad for p in pipe._model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_evaluate_adapt_and_reload_run_on_a_cuda_device(records, tmp_path):
    cuda = Swin2SRPipeline.from_pretrained(device="cuda:0", weights_dir=DEFAULT_WEIGHTS_DIR)
    assert cuda.device == "cuda:0"
    metrics = cuda.evaluate(records[:8])
    assert 10.0 < metrics["psnr"] < 60.0
    result = cuda.adapt(records[:12], records[12:], epochs=1, trainable_stages=1, batch_size=4)
    assert result["best_epoch"] in (0, 1) and result["history"][1]["train_loss"] > 0.0
    artifact = cuda.save_artifact(tmp_path / "cuda")
    reloaded = Swin2SRPipeline.from_artifact(artifact, device="cuda:0", weights_dir=DEFAULT_WEIGHTS_DIR)
    a = [cuda.upscale(r["lr"])["image"] for r in records[:4]]
    b = [reloaded.upscale(r["lr"])["image"] for r in records[:4]]
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
