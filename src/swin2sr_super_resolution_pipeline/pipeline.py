from __future__ import annotations

# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width
import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
PARAMETER_COUNT = 12_091_571
ENCODER_STAGES = 6  # config.json "depths": [6, 6, 6, 6, 6, 6]
DEFAULT_TRAINABLE_STAGES = 1  # last residual Swin transformer blocks; plus conv_after_body, upsample, final_convolution
TAIL_PREFIXES = ("swin2sr.conv_after_body", "upsample", "final_convolution")
ARTIFACT_FORMAT = f"org.valcorza.{MODEL_KEY}.adapter.v1"
ARTIFACT_VERSION = "1.0"
ADAPTER_WEIGHTS = "adapter.safetensors"
ADAPTER_MANIFEST = "manifest.json"
WEIGHT_FILE = "model.safetensors"
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
MAX_EVAL_RECORDS = 5_000


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


def _trainable_names(model: Any, trainable_stages: int) -> list[str]:
    """The last `trainable_stages` residual Swin transformer stages of the encoder plus the convolution after the
    body, the pixel-shuffle upsampler and the final convolution. The patch embedding, the earlier stages and the
    first convolution stay frozen."""
    if isinstance(trainable_stages, bool) or not isinstance(trainable_stages, int) or not 0 <= trainable_stages <= ENCODER_STAGES:
        raise ValueError(f"trainable_stages must be an int in 0..{ENCODER_STAGES}")
    n_stages = len(model.swin2sr.encoder.stages)
    stages = tuple(f"swin2sr.encoder.stages.{i}." for i in range(n_stages - trainable_stages, n_stages))
    return [name for name, _ in model.named_parameters() if name.startswith(stages) or name.startswith(TAIL_PREFIXES)]


def _check_artifact_manifest(manifest: Mapping[str, Any], artifact_dir: Path, base_sha256: str) -> None:
    """Refuse an adapter that names another base, another format or a file that does not match its digest."""
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
    base = manifest.get("base", {})
    if base.get("model_id") != MODEL_ID or base.get("revision") != MODEL_REVISION:
        raise ValueError(f"artifact was trained on {base.get('model_id')}@{base.get('revision')}, not {MODEL_ID}@{MODEL_REVISION}")
    if base.get("weight_sha256") != base_sha256:
        raise ValueError("artifact base weight digest does not match the verified snapshot")
    files = manifest.get("files") or []
    if len(files) != 1 or files[0].get("path") != ADAPTER_WEIGHTS:
        raise ValueError(f"artifact manifest must list exactly {ADAPTER_WEIGHTS}")
    weights = artifact_dir / ADAPTER_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"artifact weights missing: {weights}")
    size = weights.stat().st_size
    if size != files[0].get("bytes"):
        raise ValueError(f"{ADAPTER_WEIGHTS}: size {size} != manifest {files[0].get('bytes')}")
    digest = _sha256(weights)
    if digest != files[0].get("sha256"):
        raise ValueError(f"{ADAPTER_WEIGHTS}: sha256 {digest} != manifest {files[0].get('sha256')}")
    adapter = manifest.get("adapter") or {}
    names = manifest.get("tensors") or []
    if not names or any(not str(n).startswith(("swin2sr.encoder.stages.", *TAIL_PREFIXES)) for n in names):
        raise ValueError("artifact tensors must all belong to the encoder stages or the reconstruction tail")
    stages = adapter.get("trainable_stages")
    if isinstance(stages, bool) or not isinstance(stages, int) or not 0 <= stages <= ENCODER_STAGES:
        raise ValueError("artifact adapter.trainable_stages must be an int in 0..ENCODER_STAGES")


@dataclass
class Swin2SRPipeline:
    """2x single-image super-resolution over the pinned Swin2SR classical-SR checkpoint."""

    _runner: Callable[[Image.Image], np.ndarray]
    device: str
    _model: Any = field(default=None, repr=False)
    weight_sha256: str | None = None
    adapter: dict[str, Any] | None = None

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Swin2SRPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        weight_sha256 = None
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            with open(root / MANIFEST_NAME, encoding="utf-8") as handle:
                entries = json.load(handle).get("files", [])
            weight_sha256 = next((e["sha256"] for e in entries if e["path"] == WEIGHT_FILE), None)
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
        for param in model.parameters():
            param.requires_grad_(False)

        def runner(image: Image.Image) -> np.ndarray:
            inputs = processor(images=image, return_tensors="pt").to(resolved_device)
            with torch.inference_mode():
                reconstruction = model(**inputs).reconstruction
            # the processor pads to a multiple of 8; crop the padding back off at output scale
            out = reconstruction[0, :, : image.height * UPSCALE, : image.width * UPSCALE]
            out = out.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
            return out.permute(1, 2, 0).cpu().numpy()

        return cls(runner, resolved_device, model, weight_sha256)

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

    def _require_model(self) -> Any:
        if self._model is None:
            raise RuntimeError("this pipeline has no loaded model (injected runner); use from_pretrained for evaluate/adapt")
        return self._model


    def evaluate(self, records: Sequence[Mapping[str, Any]], *, progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        """Upscale every validated record's `lr` through `upscale` and score the reconstructions against `hr` with
        `metrics.sr_metrics` (PSNR and SSIM, overall and per category)."""
        from .metrics import sr_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        outputs = []
        for i, record in enumerate(checked):
            outputs.append(self.upscale(record["lr"])["image"])
            if progress is not None:
                progress(i + 1, len(checked))
        metrics = sr_metrics(outputs, checked)
        metrics.update(
            {
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics


    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 5,
        lr: float = 1e-4,
        batch_size: int = 8,
        trainable_stages: int = DEFAULT_TRAINABLE_STAGES,
        seed: int = 0,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning with the L1 reconstruction loss the checkpoint was trained with: every batch of
        low-resolution inputs (scaled to 0..1 exactly as the processor scales them; the pair contract keeps their
        sides multiples of the attention window, so no padding is involved) is reconstructed and compared with its
        high-resolution reference. Only the last `trainable_stages` encoder stages, the convolution after the body,
        the upsampler and the final convolution receive gradients; AdamW (no weight decay), gradient clipping at
        1.0, seeded shuffling, no scheduler. Epoch 0 records the frozen model's validation metrics; the epoch with
        the highest validation PSNR is kept (the final one without a validation split). On any exception the frozen
        weights are restored."""
        model = self._require_model()  # refuse before importing torch
        import torch

        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be an int in 1..50")
        if not isinstance(lr, int | float) or not 0.0 < float(lr) <= 1e-2:
            raise ValueError("lr must be in (0, 1e-2]")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be an int in 1..64")
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val is not None else None
        sizes = {r["lr"].size for r in train_checked}
        if len(sizes) != 1:
            raise ValueError(f"training records must share one LR size to batch; got {sorted(sizes)[:4]}")
        names = _trainable_names(model, trainable_stages)
        device = torch.device(self.device)
        name_set = set(names)
        frozen_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
        previous_adapter = self.adapter
        history: list[dict[str, Any]] = []
        started = time.perf_counter()

        def _tensor(images: Sequence[Image.Image]) -> Any:
            arrays = [np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0 for im in images]
            return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device)

        def _val() -> dict[str, Any] | None:
            if val_checked is None:
                return None
            result = self.evaluate(val_checked)
            return {k: result[k] for k in ("psnr", "ssim", "n")}

        try:
            for param in model.parameters():
                param.requires_grad_(False)
            params = []
            for name, param in model.named_parameters():
                if name in name_set:
                    param.requires_grad_(True)
                    params.append(param)
            n_trainable = sum(p.numel() for p in params)
            entry = {"epoch": 0, "train_loss": None, "val": _val(), "note": "frozen model"}
            history.append(entry)
            if progress is not None:
                progress(entry)
            best_epoch, best_score = 0, (history[0]["val"] or {}).get("psnr", -1.0)
            best_state = frozen_state
            optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
            rng = random.Random(seed)
            torch.manual_seed(seed)
            for epoch in range(1, epochs + 1):
                model.train()
                order = list(train_checked)
                rng.shuffle(order)
                losses = []
                for start in range(0, len(order), batch_size):
                    batch = order[start : start + batch_size]
                    x = _tensor([r["lr"] for r in batch])
                    y = _tensor([r["hr"] for r in batch])
                    out = model(pixel_values=x).reconstruction[:, :, : y.shape[2], : y.shape[3]]
                    loss = torch.nn.functional.l1_loss(out, y)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": _val()}
                history.append(entry)
                if progress is not None:
                    progress(entry)
                if val_checked is None or entry["val"]["psnr"] > best_score:
                    best_epoch, best_score = epoch, (entry["val"] or {}).get("psnr", -1.0)
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
            model.load_state_dict(best_state, strict=False)
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
        except BaseException:
            model.load_state_dict(frozen_state, strict=False)
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
            self.adapter = previous_adapter
            raise
        self.adapter = {
            "trainable_stages": trainable_stages,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "highest validation PSNR" if val_checked is not None else "final epoch (no validation split)",
            "loss": "L1 between the reconstruction and the high-resolution reference (0..1 scale)",
            "lr": float(lr),
            "batch_size": batch_size,
            "seed": seed,
            "n_train": len(train_checked),
            "n_val": len(val_checked) if val_checked is not None else 0,
            "lr_size": list(next(iter(sizes))),
            "history": history,
            "seconds": round(time.perf_counter() - started, 3),
        }
        return dict(self.adapter)


    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the trained tensors as safetensors plus a manifest naming the base, the digests and the training
        configuration. Requires a prior `adapt`."""
        model = self._require_model()  # refuse before importing torch
        import torch
        from safetensors.torch import save_file

        if self.adapter is None:
            raise RuntimeError("nothing to save: call adapt() first")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = list(self.adapter["trainable_names"])
        state = model.state_dict()
        tensors = {name: state[name].detach().cpu().contiguous() for name in names}
        weights = out / ADAPTER_WEIGHTS
        save_file(tensors, str(weights), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "version": ARTIFACT_VERSION,
            "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_file": WEIGHT_FILE, "weight_sha256": self.weight_sha256},
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": names,
            "files": [{"path": ADAPTER_WEIGHTS, "bytes": weights.stat().st_size, "sha256": _sha256(weights)}],
            "torch": torch.__version__,
            "metadata": dict(metadata or {}),
        }
        with open(out / ADAPTER_MANIFEST, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return out


    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Overlay a saved adapter onto this (freshly loaded) pipeline after checking its manifest, digest and exact
        tensor set. Refuses tensors outside the encoder stages and the reconstruction tail."""
        model = self._require_model()  # refuse before importing safetensors
        from safetensors.torch import load_file

        artifact = Path(artifact_dir)
        manifest_path = artifact / ADAPTER_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        _check_artifact_manifest(manifest, artifact, self.weight_sha256 or "")
        expected = _trainable_names(model, int(manifest["adapter"]["trainable_stages"]))
        if sorted(manifest["tensors"]) != sorted(expected):
            raise ValueError("artifact tensor set does not match its recorded configuration")
        tensors = load_file(str(artifact / ADAPTER_WEIGHTS))
        if sorted(tensors) != sorted(expected):
            raise ValueError("artifact tensor names differ from the manifest")
        state = model.state_dict()
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != tuple(state[name].shape):
                raise ValueError(f"artifact tensor {name} has shape {tuple(tensor.shape)}, base has {tuple(state[name].shape)}")
        model.load_state_dict({k: v.to(state[k].device, state[k].dtype) for k, v in tensors.items()}, strict=False)
        model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": expected, "history": manifest.get("history", [])}
        return dict(self.adapter)


    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Swin2SRPipeline:
        """Load the verified base snapshot, then overlay the adapter (verified before deserialising)."""
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe
