"""Corpus-level super-resolution measures and two non-neural baselines, in numpy.

``sr_metrics`` scores one reconstruction per record against its high-resolution reference: PSNR (``pipeline.psnr``,
RGB, dB) and SSIM (this module; the luma channel, an 11-tap Gaussian window of sigma 1.5, a 5-pixel border
excluded — the classical-SR convention), each averaged over the records and per ``category``. The baselines
upscale the low-resolution input with bicubic or nearest-neighbour interpolation and are scored by the same
function.
"""
# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image

from .pipeline import UPSCALE, psnr

METRIC_DEFINITIONS = {
    "psnr": "mean peak signal-to-noise ratio in dB over the RGB channels of the reconstruction against the reference (255 peak); higher is better, unbounded",
    "ssim": "mean structural similarity on the luma channel (11-tap Gaussian window, sigma 1.5, 5-px border excluded); in -1..1, higher is better",
}
_GAUSS = np.exp(-((np.arange(11) - 5) ** 2) / (2 * 1.5**2))
_WINDOW = np.outer(_GAUSS, _GAUSS) / np.outer(_GAUSS, _GAUSS).sum()


def luma(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma (16..235) of a uint8 RGB array, as float64."""
    arr = np.asarray(rgb, dtype=np.float64)
    return 16.0 + (65.481 * arr[..., 0] + 128.553 * arr[..., 1] + 24.966 * arr[..., 2]) / 255.0


def _filter(z: np.ndarray) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    return np.einsum("ijkl,kl->ij", sliding_window_view(z, (11, 11)), _WINDOW)


def ssim(pred: np.ndarray, ref: np.ndarray) -> float:
    """Structural similarity between two uint8 RGB arrays of identical shape (see METRIC_DEFINITIONS)."""
    pred, ref = np.asarray(pred), np.asarray(ref)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs ref {ref.shape}")
    if pred.dtype != np.uint8 or ref.dtype != np.uint8:
        raise TypeError("ssim expects uint8 arrays")
    if min(pred.shape[:2]) < 11:
        raise ValueError("ssim needs images of at least 11 px on each side")
    x, y = luma(pred), luma(ref)
    mu_x, mu_y = _filter(x), _filter(y)
    sxx, syy, sxy = _filter(x * x) - mu_x**2, _filter(y * y) - mu_y**2, _filter(x * y) - mu_x * mu_y
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return float(s.mean())


def sr_metrics(outputs: Sequence[np.ndarray], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score one uint8 RGB reconstruction per record against `record['hr']`; per-record rows, means overall and
    per category. Raises when the lengths differ or nothing is scored."""
    if len(outputs) != len(records) or not outputs:
        raise ValueError("outputs and records must be non-empty and the same length")
    rows = []
    for output, record in zip(outputs, records, strict=True):
        ref = np.asarray(record["hr"].convert("RGB"))
        rows.append({"id": record["id"], "category": record.get("category"), "psnr": psnr(output, ref), "ssim": ssim(output, ref)})

    def _mean(items: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        finite = [r["psnr"] for r in items if np.isfinite(r["psnr"])]
        return {
            "n": len(items),
            "psnr": float(np.mean(finite)) if finite else float("inf"),
            "ssim": float(np.mean([r["ssim"] for r in items])),
        }

    categories = sorted({r["category"] for r in rows if r["category"] is not None})
    return {
        **_mean(rows),
        "per_category": {c: _mean([r for r in rows if r["category"] == c]) for c in categories},
        "per_record": rows,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def _interpolate(records: Sequence[Mapping[str, Any]], resample: int) -> list[np.ndarray]:
    return [np.asarray(r["lr"].convert("RGB").resize((r["lr"].width * UPSCALE, r["lr"].height * UPSCALE), resample), dtype=np.uint8) for r in records]


def bicubic_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The classical reference: the low-resolution input upscaled 2x with bicubic interpolation."""
    return {**sr_metrics(_interpolate(records, Image.BICUBIC), records), "baseline": "bicubic 2x interpolation of the LR input"}


def nearest_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The floor: the low-resolution input upscaled 2x by pixel replication."""
    return {**sr_metrics(_interpolate(records, Image.NEAREST), records), "baseline": "nearest-neighbour 2x interpolation of the LR input"}
