import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from swin2sr_super_resolution_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    MAX_INPUT_SIDE,
    MIN_INPUT_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    UPSCALE,
    Swin2SRPipeline,
    psnr,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "caidas/swin2SR-classical-sr-x2-64"
    assert UPSCALE == 2
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    manifest = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["modelId"] == MODEL_ID
        assert data["revision"] == MODEL_REVISION


def _write_snapshot(root: Path, content: bytes, sha: str | None = None, size: int | None = None) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_manifest(tmp_path):
    _write_snapshot(tmp_path, b'{"model_type": "swin2sr"}')
    info = verify_snapshot(tmp_path)
    assert info["revision"] == MODEL_REVISION and info["files"] == 1


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"model_type": "swin2sr"}'
    good = hashlib.sha256(content).hexdigest()
    flipped = ("0" if good[0] != "0" else "1") + good[1:]
    _write_snapshot(tmp_path, content, sha=flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_size_missing_file_and_revision(tmp_path):
    _write_snapshot(tmp_path, b"abc", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["revision"] = "0" * 40
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def _fake_pipeline(calls: list | None = None) -> Swin2SRPipeline:
    def runner(image: Image.Image) -> np.ndarray:
        if calls is not None:
            calls.append(image.mode)
        big = image.resize((image.width * UPSCALE, image.height * UPSCALE), Image.NEAREST)
        return np.asarray(big, dtype=np.uint8)

    return Swin2SRPipeline(runner, "cpu")


def test_upscale_output_fields():
    calls: list = []
    pipe = _fake_pipeline(calls)
    result = pipe.upscale(Image.new("L", (40, 30), color=7))
    assert result["image"].shape == (60, 80, 3) and result["image"].dtype == np.uint8
    assert result["scale"] == 2
    assert result["input_size"] == (40, 30) and result["output_size"] == (80, 60)
    assert result["model_id"] == MODEL_ID and result["model_revision"] == MODEL_REVISION
    assert calls == ["RGB"]


def test_upscale_rejects_bad_inputs():
    pipe = _fake_pipeline()
    with pytest.raises(TypeError):
        pipe.upscale(np.zeros((30, 40, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="MIN_INPUT_SIDE"):
        pipe.upscale(Image.new("RGB", (MIN_INPUT_SIDE - 1, 64)))
    with pytest.raises(ValueError, match="MAX_INPUT_SIDE"):
        pipe.upscale(Image.new("RGB", (MAX_INPUT_SIDE + 1, 64)))


def test_upscale_rejects_backend_shape_mismatch():
    pipe = Swin2SRPipeline(lambda image: np.zeros((3, 3, 3), dtype=np.uint8), "cpu")
    with pytest.raises(RuntimeError):
        pipe.upscale(Image.new("RGB", (40, 30)))


def test_psnr():
    ref = np.full((8, 8, 3), 100, dtype=np.uint8)
    assert psnr(ref, ref) == float("inf")
    pred = ref.copy()
    pred[..., 0] += 3  # mse = 9 on one of three channels -> 3.0
    assert psnr(pred, ref) == pytest.approx(10 * np.log10(255**2 / 3.0))
    with pytest.raises(ValueError):
        psnr(ref[:4], ref)
    with pytest.raises(TypeError):
        psnr(ref.astype(np.float32), ref)
