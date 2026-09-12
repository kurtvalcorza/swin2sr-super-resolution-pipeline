from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_ID = "caidas/swin2SR-classical-sr-x2-64"
MODEL_REVISION = "cee1c923c6a37361c6e5650b65dcf4be821e5d52"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "swin2sr-x2-64"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

UPSCALE = 2  # config.json "upscale": 2
# Input ceilings. Swin2SR runs windowed attention over every input pixel (patch_size 1), so activation
# memory and time grow with input area; MAX_INPUT_SIDE is the largest square side actually executed
# on CPU during the card pass (1024 px in 160 s; see MODEL_CARD.md, Runtime).
MAX_INPUT_SIDE = 1024
MIN_INPUT_SIDE = 8  # config.json "window_size": 8; the processor pads to a multiple of 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB between two uint8 RGB arrays of identical shape.

    ``10 * log10(255**2 / MSE)`` over all channels; returns ``inf`` when the arrays are identical.
    The caller supplies the high-resolution reference; PSNR on one image is a check, not a benchmark.
    """
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs ref {ref.shape}")
    if pred.dtype != np.uint8 or ref.dtype != np.uint8:
        raise TypeError("psnr expects uint8 arrays")
    mse = float(np.mean((pred.astype(np.float64) - ref.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def validate_image(image: Any) -> Image.Image:
    """Type- and size-check a caller image and return it as RGB."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_INPUT_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_INPUT_SIDE {MIN_INPUT_SIDE}")
    if max(width, height) > MAX_INPUT_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_INPUT_SIDE {MAX_INPUT_SIDE}")
    return image.convert("RGB")


@dataclass
class Swin2SRPipeline:
    """2x single-image super-resolution over the pinned Swin2SR classical-SR checkpoint."""

    _runner: Callable[[Image.Image], np.ndarray]
    device: str

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Swin2SRPipeline:
        import torch
        from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), {"local_files_only": True}
        elif allow_download:
            source, kwargs = MODEL_ID, {}
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage {MODEL_ID}@{MODEL_REVISION} under weights/{MODEL_KEY}"
            )
        processor = Swin2SRImageProcessor.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = Swin2SRForImageSuperResolution.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = model.to(resolved_device).eval()

        def runner(image: Image.Image) -> np.ndarray:
            inputs = processor(images=image, return_tensors="pt").to(resolved_device)
            with torch.inference_mode():
                reconstruction = model(**inputs).reconstruction
            # the processor pads to a multiple of 8; crop the padding back off at output scale
            out = reconstruction[0, :, : image.height * UPSCALE, : image.width * UPSCALE]
            out = out.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
            return out.permute(1, 2, 0).cpu().numpy()

        return cls(runner, resolved_device)

    def upscale(self, image: Image.Image) -> dict[str, Any]:
        """Return the 2x-upscaled RGB image as a uint8 array of shape (2H, 2W, 3)."""
        rgb = validate_image(image)
        expected = (rgb.height * UPSCALE, rgb.width * UPSCALE, 3)
        result = np.asarray(self._runner(rgb))
        if result.shape != expected or result.dtype != np.uint8:
            raise RuntimeError(f"backend returned {result.shape} {result.dtype}, expected {expected} uint8")
        return {
            "image": result,
            "scale": UPSCALE,
            "input_size": (rgb.width, rgb.height),
            "output_size": (rgb.width * UPSCALE, rgb.height * UPSCALE),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
