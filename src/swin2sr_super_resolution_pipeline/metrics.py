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


# Pre-registration amendment 1 (2026-10-18, docs/swin2sr-preregistration-amendment-1.md): how a test split is read.
READ_SEED = 20261018
READ_N_BOOT = 10_000
READ_MARGIN_DB = 0.10


def paired_bootstrap(
    diffs: Sequence[float], strata: Sequence[Any], *, seed: int = READ_SEED, n_boot: int = READ_N_BOOT
) -> tuple[float, float]:
    """95 % percentile bootstrap interval of the mean paired difference, resampling with replacement within each
    stratum (species) so the stratum counts are kept."""
    d = np.asarray(diffs, dtype=float)
    if d.size == 0:
        raise ValueError("no paired differences")
    labels = np.asarray(["" if s is None else str(s) for s in strata])
    rng = np.random.default_rng(seed)
    totals = np.zeros(n_boot)
    for label in sorted(set(labels.tolist())):
        values = d[labels == label]
        totals += values[rng.integers(0, values.size, size=(n_boot, values.size))].sum(axis=1)
    lo, hi = np.percentile(totals / d.size, [2.5, 97.5])
    return float(lo), float(hi)


def read_comparison(
    selected: Mapping[str, Any], reference: Mapping[str, Any], *, margin: float = READ_MARGIN_DB,
    seed: int = READ_SEED, n_boot: int = READ_N_BOOT,
) -> dict[str, Any]:
    """The amendment-1 reading rule for `selected` against `reference` (two `sr_metrics` results on the same crops):
    paired per-crop PSNR and SSIM differences, species-stratified bootstrap 95 % intervals, and a verdict that is
    'improved' only if the PSNR interval's lower bound is above 0 AND the mean gain is at least `margin` dB ('worse'
    mirrors it; anything else is 'within tolerance'). SSIM is reported, not used for the verdict."""
    ref = {r["id"]: r for r in reference.get("per_record", [])}
    pairs = [(r, ref[r["id"]]) for r in selected.get("per_record", []) if r["id"] in ref]
    finite = [(a, b) for a, b in pairs if np.isfinite(a["psnr"]) and np.isfinite(b["psnr"])]
    rule = f"improved if the 95 % CI lower bound > 0 and the mean PSNR gain >= {margin:g} dB (bootstrap {n_boot}, seed {seed}, stratified by species)"
    if not finite:
        return {"n": 0, "dropped": len(pairs), "verdict": "not computed (no paired crops)", "rule": rule}
    strata = [a.get("category") for a, _ in finite]
    dp = [a["psnr"] - b["psnr"] for a, b in finite]
    ds = [a["ssim"] - b["ssim"] for a, b in finite]
    lo, hi = paired_bootstrap(dp, strata, seed=seed, n_boot=n_boot)
    slo, shi = paired_bootstrap(ds, strata, seed=seed, n_boot=n_boot)
    mean = float(np.mean(dp))
    tol = 1e-9
    verdict = "improved" if lo > 0 and mean >= margin - tol else ("worse" if hi < 0 and mean <= -margin + tol else "within tolerance")
    return {
        "n": len(finite),
        "dropped": len(pairs) - len(finite),
        "psnr": {"mean_diff": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "wins": int(sum(x > 0 for x in dp)), "losses": int(sum(x < 0 for x in dp))},
        "ssim": {"mean_diff": round(float(np.mean(ds)), 5), "ci95": [round(slo, 5), round(shi, 5)]},
        "verdict": verdict,
        "rule": rule,
    }
