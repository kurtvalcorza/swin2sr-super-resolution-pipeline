from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
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
# memory and time grow with input area. The card pass executed up to 1024 px on CPU (160 s); the ceiling
# was set to 512 px (34 s on the reference CPU) on 2026-09-12 by the repository owner so DIMER validators
# stay responsive. See MODEL_CARD.md, Runtime.
MAX_INPUT_SIDE = 512
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


INPUT_SCHEMA: dict[str, Any] = {
    "input": "PIL.Image.Image, or a sequence of them for the validation stage; any mode, converted to RGB",
    "image_side_px": [MIN_INPUT_SIDE, MAX_INPUT_SIDE],
    "scale": UPSCALE,
    "output": f"uint8 RGB array of shape ({UPSCALE}H, {UPSCALE}W, 3)",
    "preprocessing": (
        f"convert to RGB and pad to a multiple of the {MIN_INPUT_SIDE} px attention window; the padding "
        "is cropped back off at output scale"
    ),
}


def validate_inputs(images: Any, *, names: Sequence[str] | None = None) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, verdict).

    Each image is routed through the public ``validate_image`` that ``upscale`` itself calls, so a
    rejection here raises exactly what ``upscale`` would; a caller that wants the finding recorded
    catches the exception and stores ``str(exc)`` under ``findings``.
    """
    batch = [images] if isinstance(images, Image.Image) else images
    if not isinstance(batch, Sequence) or isinstance(batch, str | bytes):
        raise TypeError("images must be a PIL.Image.Image or a sequence of them")
    if len(batch) < 1:
        raise ValueError("at least one image is required")
    if names is not None and len(names) != len(batch):
        raise ValueError("names must have one entry per image")
    inputs = []
    for index, candidate in enumerate(batch):
        rgb = validate_image(candidate)
        inputs.append(
            {
                "id": names[index] if names else f"image-{index}",
                "mode": getattr(candidate, "mode", rgb.mode),
                "size": [rgb.width, rgb.height],
                "output_size": [rgb.width * UPSCALE, rgb.height * UPSCALE],
            }
        )
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": inputs,
        "scale": UPSCALE,
        "n_images": len(inputs),
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def _as_uint8_rgb(image: Any) -> np.ndarray:
    """Coerce a PIL image or an array to a uint8 RGB array; raise on anything else."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise TypeError("reference must be a PIL image or a uint8 RGB array of shape (H, W, 3)")
    return array


def evaluation_report(
    result: Mapping[str, Any],
    reference: Any | None = None,
    *,
    low_resolution: Any | None = None,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``reference`` (the high-resolution image the output should match, same shape as
    ``result["image"]``) the report carries ``psnr`` in dB as sample-sanity evidence; pass
    ``low_resolution`` as well and the same ``psnr`` is computed for a plain bicubic upscale of the
    input, which is the only baseline worth comparing against. Without a reference the verdict is
    ``not-measurable``: a reconstruction has no intrinsic score.
    """
    output = np.asarray(result["image"])
    base = {
        "task": f"{UPSCALE}x single-image super-resolution (classical SR)",
        "score_semantics": (
            "PSNR in dB between two uint8 RGB arrays (10*log10(255**2 / MSE)); higher is closer to the "
            "reference, it is not a perceptual quality score, and the pipeline ships no threshold"
        ),
        "sample_kind": sample_kind,
        "n_images": 1,
        "scale": result.get("scale", UPSCALE),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if reference is None:
        return {
            **base,
            "metrics": [],
            "baselines": [],
            "verdict": "not-measurable",
            "reason": "no high-resolution reference was supplied for the evaluated image",
            "needs": (
                "high-resolution/low-resolution pairs produced by a stated degradation — a public "
                "benchmark such as Set5, Set14 or DIV2K with its own downscaling kernel — scored with "
                "psnr against a plain bicubic upscale of the same input as the baseline"
            ),
        }
    ref_array = _as_uint8_rgb(reference)
    baselines = []
    if low_resolution is not None:
        height, width = ref_array.shape[0], ref_array.shape[1]
        bicubic = _as_uint8_rgb(
            _as_pil(low_resolution).convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        )
        baselines.append(
            {
                "id": "psnr",
                "name": f"bicubic {UPSCALE}x resize of the same input",
                "value": psnr(bicubic, ref_array),
                "unit": "dB",
            }
        )
    return {
        **base,
        "metrics": [
            {
                "id": "psnr",
                "value": psnr(output, ref_array),
                "unit": "dB",
                "estimation": "single image against a caller-supplied reference, no dispersion estimate",
            }
        ],
        "baselines": baselines,
        "verdict": "sample-sanity",
        "reason": (
            "one image scored against a reference the caller supplied, under the caller's own "
            "downscaling kernel; not a benchmark"
        ),
        "needs": (
            "a public benchmark set with its stated degradation kernel for any comparable PSNR claim"
        ),
    }


def _as_pil(image: Any) -> Image.Image:
    """Coerce a PIL image or a uint8 RGB array to a PIL image."""
    if isinstance(image, Image.Image):
        return image
    return Image.fromarray(_as_uint8_rgb(image))


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
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
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
