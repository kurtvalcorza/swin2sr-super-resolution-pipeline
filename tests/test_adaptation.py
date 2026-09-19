"""Offline checks of the adaptation contract: the paired-record contract and its refusals, the degradation,
the pinned corpus (fetch refusals, the draw), splitting, the BYOD loader, the metrics and baselines, `evaluate`
through the injected runner, the artifact-manifest checks, and the model-free refusals of `adapt` / artifacts."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image, ImageDraw

from swin2sr_super_resolution_pipeline import (
    ARTIFACT_FORMAT,
    CORPUS_BYTES,
    ENCODER_STAGES,
    HR_CROP,
    JPEG_QUALITY,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RECORDS,
    SAMPLE_SPLIT,
    SPECIES,
    UPSCALE,
    Swin2SRPipeline,
    bicubic_baseline,
    build_sample_dataset,
    check_split_disjoint,
    dataset_digest,
    degrade,
    fetch_corpus,
    image_digest,
    load_byod_dataset,
    nearest_baseline,
    observer_overlap,
    read_corpus,
    split_dataset,
    sr_metrics,
    ssim,
    validate_dataset,
    write_dataset_csv,
)
from swin2sr_super_resolution_pipeline import pipeline as pl
from swin2sr_super_resolution_pipeline import samples as sm

COLOURS = ["red", "green", "blue", "yellow", "white", "black"]


def _scene(i, size=64):
    image = Image.new("RGB", (size, size), (135, 206, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, size * 2 // 3, size, size], fill=(60, 179, 75))
    draw.ellipse([size // 4, size // 6, size * 3 // 4, size * 2 // 3], fill=COLOURS[i % 6])
    draw.line([(0, i % size), (size, (i * 7) % size)], fill=(200, 30, 30), width=2)
    image.putpixel((i % size, 0), (i % 256, 0, 0))  # distinct per record
    return image


def _records(n=12, size=64):
    return [{"id": f"r{i:03d}", "hr": _scene(i, size), "lr": degrade(_scene(i, size)), "category": COLOURS[i % 6]} for i in range(n)]


class _ScriptedRunner:
    """Upscales by nearest-neighbour replication (records the calls)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        return np.asarray(image.resize((image.width * UPSCALE, image.height * UPSCALE), Image.NEAREST), dtype=np.uint8)


# --- degradation and record contract ---------------------------------------------------------------------------


def test_degrade_halves_the_image_and_is_deterministic():
    hr = _scene(3, 96)
    lr = degrade(hr)
    assert lr.size == (48, 48) and lr.mode == "RGB"
    assert np.array_equal(np.asarray(lr), np.asarray(degrade(hr)))
    assert not np.array_equal(np.asarray(lr), np.asarray(hr.resize((48, 48), Image.BICUBIC)))  # JPEG changed it
    with pytest.raises(ValueError, match="multiples"):
        degrade(Image.new("RGB", (33, 32)))
    with pytest.raises(ValueError, match="quality"):
        degrade(hr, quality=0)
    with pytest.raises(TypeError):
        degrade("not an image")


def test_validate_dataset_accepts_records_and_reports_counts_and_digest():
    manifest = validate_dataset(_records())
    assert manifest["n_records"] == 12 and manifest["category_counts"] == {c: 2 for c in COLOURS}
    assert manifest["hr_side"] == {"min": 64, "max": 64} and len(manifest["digest"]) == 64 and manifest["model_id"] == MODEL_ID


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.__setitem__("id", "bad id"), "id must match"),
        (lambda r: r.__setitem__("lr", r["hr"]), "downscaled by"),
        (lambda r: r.__setitem__("hr", Image.new("RGB", (72, 72))), "multiples of 16"),
        (lambda r: r.__setitem__("hr", Image.new("RGB", (8, 8))), "within"),
        (lambda r: r.__setitem__("hr", "missing.png"), "not found"),
        (lambda r: r.__setitem__("category", "x" * 40), "category must be"),
        (lambda r: r.pop("lr"), "missing 'lr'"),
    ],
)
def test_validate_dataset_refuses_malformed_records(mutate, message):
    records = _records()
    mutate(records[0])
    with pytest.raises(ValueError, match=message):
        validate_dataset(records)


def test_validate_dataset_enforces_bounds_and_unique_ids():
    with pytest.raises(ValueError, match="8..5000"):
        validate_dataset(_records(4))
    records = _records()
    records[1]["id"] = records[0]["id"]
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset(records)
    with pytest.raises(ValueError, match="list of"):
        validate_dataset({"id": "x"})


def test_validate_dataset_refuses_before_importing_model_libraries(forbid_model_imports):
    with pytest.raises(ValueError):
        validate_dataset(_records(3))
    validate_dataset(_records())


def test_digests_and_split_disjointness():
    records = _records(24)
    assert dataset_digest(records) == dataset_digest(list(reversed(records)))
    assert image_digest(records[0]["hr"]) != image_digest(records[1]["hr"])
    splits = split_dataset(records, seed=1)
    assert sum(len(v) for v in splits.values()) == 24 and all(splits.values())
    assert check_split_disjoint(splits) == {k: len(v) for k, v in splits.items()}
    leaked = {**splits, "test": [*splits["test"], splits["train"][0]]}
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint(leaked)
    duplicated = [*records, {**records[0], "id": "copy"}]
    assert sum(len(v) for v in split_dataset(duplicated, seed=1).values()) == 24  # pixel-identical copy dropped
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.9)


def test_observer_overlap_counts_observers_in_more_than_one_split():
    splits = {"train": [{"observer": "a"}, {"observer": "b"}], "test": [{"observer": "a"}]}
    assert observer_overlap(splits) == {"observers": 2, "in_more_than_one_split": 1}


# --- pinned corpus ----------------------------------------------------------------------------------------------


def test_pin_table_is_360_photographs_over_six_species():
    assert len(SAMPLE_RECORDS) == 360 and len(SPECIES) == 6
    ids = {r[0] for r in SAMPLE_RECORDS}
    assert len(ids) == 360 and all(r[1] in SPECIES for r in SAMPLE_RECORDS)
    assert sum(r[5] for r in SAMPLE_RECORDS) == CORPUS_BYTES
    assert all(len(r[6]) == 64 and r[7].lower() in ("jpg", "jpeg", "png") for r in SAMPLE_RECORDS)


def test_fetch_corpus_refuses_a_photo_that_does_not_match_its_pin(tmp_path):
    with pytest.raises(ValueError, match="bytes, pinned"):
        fetch_corpus(cache_dir=tmp_path, fetcher=lambda url: b"nope")
    rid, _label, _photo, _obs, _user, size, _digest, _ext = SAMPLE_RECORDS[0]
    with pytest.raises(ValueError, match="sha256"):
        fetch_corpus(cache_dir=tmp_path, fetcher=lambda url: b"x" * size)
    assert not any(tmp_path.iterdir())


def test_read_corpus_crops_and_degrades_every_pinned_photo():
    files = {}
    for rid, *_rest in SAMPLE_RECORDS:
        buffer = io.BytesIO()
        _scene(len(files), 256).save(buffer, format="JPEG")
        files[rid] = buffer.getvalue()
    records = read_corpus(files)
    assert len(records) == 360 and records[0]["hr"].size == (HR_CROP, HR_CROP) and records[0]["lr"].size == (HR_CROP // 2, HR_CROP // 2)
    assert records[0]["category"] == SAMPLE_RECORDS[0][1] and records[0]["inat_observation_url"].startswith("https://www.inaturalist.org/observations/")
    with pytest.raises(ValueError, match="missing"):
        read_corpus({})
    splits = build_sample_dataset(records)
    assert {k: len(v) for k, v in splits.items()} == {k: v * 6 for k, v in SAMPLE_SPLIT.items()}
    assert check_split_disjoint(splits)
    assert build_sample_dataset(records) == splits  # seeded


def test_default_draw_matches_the_pinned_digest_when_the_photographs_are_cached():
    cached = sm.DEFAULT_CACHE_DIR
    if not (cached / f"{SAMPLE_RECORDS[0][2]}.jpg").is_file():
        pytest.skip("iNaturalist photographs not cached")
    splits = build_sample_dataset(read_corpus(fetch_corpus(cache_dir=cached)))
    assert dataset_digest([r for part in splits.values() for r in part]) == sm.SAMPLE_DIGEST


# --- BYOD ---------------------------------------------------------------------------------------------------------


def test_load_byod_dataset_reads_a_zip_or_directory_and_degrades_the_crops(tmp_path):
    zip_path = tmp_path / "photos.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for i in range(10):
            buffer = io.BytesIO()
            _scene(i, 200).save(buffer, format="PNG")
            archive.writestr(f"photo{i}.png", buffer.getvalue())
        small = io.BytesIO()
        _scene(99, 64).save(small, format="PNG")
        archive.writestr("tiny.png", small.getvalue())
        archive.writestr("labels.csv", "id,file,category\n" + "".join(f"p{i},photo{i}.png,{COLOURS[i % 6]}\n" for i in range(10)))
    records = load_byod_dataset(zip_path)
    assert len(records) == 10 and {r["id"] for r in records} == {f"p{i}" for i in range(10)}  # the 64-px image is skipped
    assert all(r["hr"].size == (HR_CROP, HR_CROP) and r["lr"].size == (HR_CROP // 2, HR_CROP // 2) for r in records)
    assert validate_dataset(records)["category_counts"] == {c: 2 if i < 4 else 1 for i, c in enumerate(COLOURS)}
    folder = tmp_path / "folder"
    folder.mkdir()
    for i in range(8):
        _scene(i, 200).save(folder / f"img{i}.png")
    assert len(load_byod_dataset(folder)) == 8
    (folder / "bad.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="not a decodable image"):
        load_byod_dataset(folder)
    with pytest.raises(ValueError, match="neither"):
        load_byod_dataset(tmp_path / "nothing")
    csv_path = write_dataset_csv(records, tmp_path / "train.csv")
    assert csv_path.read_text(encoding="utf-8").startswith("id,category,hr_width")


# --- metrics, baselines, evaluate ----------------------------------------------------------------------------------


def test_ssim_and_sr_metrics_score_reconstructions_per_category():
    records = _records()
    perfect = sr_metrics([np.asarray(r["hr"]) for r in records], records)
    assert perfect["psnr"] == float("inf") and perfect["ssim"] == pytest.approx(1.0)
    assert set(perfect["per_category"]) == set(COLOURS) and perfect["definitions"]
    noisy = [np.clip(np.asarray(r["hr"]).astype(int) + 20, 0, 255).astype(np.uint8) for r in records]
    scored = sr_metrics(noisy, records)
    assert 20.0 < scored["psnr"] < 30.0 and 0.0 < scored["ssim"] < 1.0
    assert ssim(noisy[0], np.asarray(records[0]["hr"])) < 1.0
    with pytest.raises(ValueError, match="same length"):
        sr_metrics(noisy[:3], records)
    with pytest.raises(ValueError, match="shape mismatch"):
        ssim(noisy[0][:32], np.asarray(records[0]["hr"]))
    with pytest.raises(TypeError):
        ssim(noisy[0].astype(np.float32), np.asarray(records[0]["hr"]))


def test_baselines_score_through_the_same_metrics():
    records = _records()
    bicubic, nearest = bicubic_baseline(records), nearest_baseline(records)
    assert bicubic["baseline"].startswith("bicubic") and nearest["baseline"].startswith("nearest")
    assert bicubic["n"] == 12 and 15.0 < bicubic["psnr"] < 60.0 and 15.0 < nearest["psnr"] < 60.0


def test_evaluate_runs_every_record_through_upscale_and_scores_it():
    runner = _ScriptedRunner()
    pipe = Swin2SRPipeline(runner, "cpu")
    result = pipe.evaluate(_records())
    assert runner.calls == 12 and result["n"] == 12 and result["adapted"] is False
    assert result["verdict"] == "measured-small-sample" and result["psnr"] == pytest.approx(nearest_baseline(_records())["psnr"])
    with pytest.raises(ValueError):
        pipe.evaluate([{"id": "x"}] * 8)


def test_adapt_and_artifacts_require_a_loaded_model(forbid_model_imports):
    pipe = Swin2SRPipeline(_ScriptedRunner(), "cpu")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.adapt(_records())
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.save_artifact("x")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.load_artifact("x")


# --- artifact manifest checks --------------------------------------------------------------------------------------


def _manifest(tmp_path, **overrides):
    weights = tmp_path / "adapter.safetensors"
    weights.write_bytes(b"tensor-bytes")
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": "base-digest"},
        "adapter": {"trainable_stages": 1},
        "tensors": ["swin2sr.encoder.stages.5.layers.0.attention.self.query.weight", "final_convolution.weight"],
        "files": [{"path": "adapter.safetensors", "bytes": weights.stat().st_size, "sha256": hashlib.sha256(b"tensor-bytes").hexdigest()}],
    }
    manifest.update(overrides)
    return manifest


def test_check_artifact_manifest_accepts_a_consistent_manifest_and_refuses_each_deviation(tmp_path):
    pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="format"):
        pl._check_artifact_manifest(_manifest(tmp_path, format="other"), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="trained on"):
        pl._check_artifact_manifest(_manifest(tmp_path, base={"model_id": "x", "revision": MODEL_REVISION, "weight_sha256": "base-digest"}), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="base weight digest"):
        pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "another-digest")
    bad = _manifest(tmp_path)
    bad["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        pl._check_artifact_manifest(bad, tmp_path, "base-digest")
    with pytest.raises(ValueError, match="encoder stages or the reconstruction tail"):
        pl._check_artifact_manifest(_manifest(tmp_path, tensors=["swin2sr.embeddings.patch_embeddings.projection.weight"]), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="trainable_stages"):
        pl._check_artifact_manifest(_manifest(tmp_path, adapter={"trainable_stages": ENCODER_STAGES + 1}), tmp_path, "base-digest")


def test_trainable_names_selects_the_last_stages_and_the_tail():
    class _Param:
        def numel(self):
            return 1

    class _Model:
        class swin2sr:  # noqa: N801
            class encoder:  # noqa: N801
                stages = list(range(6))

            conv_after_body = None

        def named_parameters(self):
            names = [f"swin2sr.encoder.stages.{i}.w" for i in range(6)] + ["swin2sr.embeddings.w", "swin2sr.first_convolution.w", "swin2sr.conv_after_body.w", "upsample.w", "final_convolution.w"]
            return [(n, _Param()) for n in names]

    assert pl._trainable_names(_Model(), 1) == ["swin2sr.encoder.stages.5.w", "swin2sr.conv_after_body.w", "upsample.w", "final_convolution.w"]
    assert len(pl._trainable_names(_Model(), 0)) == 3
    with pytest.raises(ValueError, match="trainable_stages"):
        pl._trainable_names(_Model(), 7)


def test_json_round_trip_of_the_manifest_keeps_the_history_shape(tmp_path):
    payload = {"epoch": 1, "train_loss": 0.03, "val": {"psnr": 26.0, "ssim": 0.8, "n": 48}}
    (tmp_path / "h.json").write_text(json.dumps([payload]), encoding="utf-8")
    assert json.loads((tmp_path / "h.json").read_text(encoding="utf-8"))[0]["val"]["psnr"] == 26.0
    assert JPEG_QUALITY == 40
