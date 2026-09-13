"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from swin2sr_super_resolution_pipeline import (
    INPUT_SCHEMA,
    MAX_INPUT_SIDE,
    MIN_INPUT_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    UPSCALE,
    evaluation_report,
    validate_inputs,
)


def _image(width: int = 32, height: int = 16, mode: str = "RGB") -> Image.Image:
    return Image.new(mode, (width, height), 90 if mode == "L" else (10, 20, 30))


def _reference(width: int = 32, height: int = 16) -> Image.Image:
    """A high-resolution reference with a sharp edge, so bicubic and the model can differ."""
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, width // 2 :] = 255
    return Image.fromarray(array)


def _result(image: np.ndarray) -> dict:
    height, width = image.shape[0], image.shape[1]
    return {
        "image": image,
        "scale": UPSCALE,
        "input_size": (width // UPSCALE, height // UPSCALE),
        "output_size": (width, height),
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs([_image(), _image(64, 64)], names=["a", "b"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_INPUT_SIDE, MAX_INPUT_SIDE]
    assert manifest["schema"]["scale"] == UPSCALE
    assert manifest["scale"] == UPSCALE
    assert manifest["n_images"] == 2
    assert manifest["inputs"] == [
        {"id": "a", "mode": "RGB", "size": [32, 16], "output_size": [64, 32]},
        {"id": "b", "mode": "RGB", "size": [64, 64], "output_size": [128, 128]},
    ]
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_single_image_default_ids_and_reports_source_mode() -> None:
    manifest = validate_inputs(_image(mode="L"))
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["inputs"][0]["mode"] == "L"  # the input's own mode, before the RGB conversion


def test_validate_inputs_rejects_like_upscale() -> None:
    with pytest.raises(ValueError, match="MIN_INPUT_SIDE"):
        validate_inputs(_image(MIN_INPUT_SIDE - 1, 32))
    with pytest.raises(ValueError, match="MAX_INPUT_SIDE"):
        validate_inputs(_image(MAX_INPUT_SIDE + 1, 32))
    with pytest.raises(TypeError):
        validate_inputs("not an image")
    with pytest.raises(ValueError, match="names must have one entry per image"):
        validate_inputs([_image()], names=["a", "b"])
    with pytest.raises(ValueError, match="at least one image"):
        validate_inputs([])


def test_evaluation_report_not_measurable_without_reference() -> None:
    report = evaluation_report(_result(np.zeros((32, 64, 3), dtype=np.uint8)))
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert "Set5" in report["needs"]
    assert "PSNR in dB" in report["score_semantics"]
    assert report["scale"] == UPSCALE
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_sample_sanity_with_reference() -> None:
    reference = _reference()
    report = evaluation_report(_result(np.asarray(reference)), reference, sample_kind="BYOD")
    assert report["verdict"] == "sample-sanity"
    assert report["sample_kind"] == "BYOD"
    (metric,) = report["metrics"]
    assert metric["id"] == "psnr"  # the repository's own helper name (EVAL2)
    assert metric["unit"] == "dB"
    assert metric["value"] == float("inf")  # identical arrays
    assert report["baselines"] == []  # no low_resolution supplied, so no baseline is claimed


def test_evaluation_report_adds_the_bicubic_baseline_when_the_input_is_supplied() -> None:
    reference = _reference()
    low_resolution = reference.resize(
        (reference.width // UPSCALE, reference.height // UPSCALE), Image.Resampling.BICUBIC
    )
    report = evaluation_report(_result(np.asarray(reference)), reference, low_resolution=low_resolution)
    (baseline,) = report["baselines"]
    assert baseline["id"] == "psnr"
    assert "bicubic" in baseline["name"]
    assert baseline["value"] < report["metrics"][0]["value"]  # the perfect output beats interpolation


def test_evaluation_report_rejects_a_non_uint8_reference() -> None:
    with pytest.raises(TypeError, match="uint8 RGB array"):
        evaluation_report(
            _result(np.zeros((32, 64, 3), dtype=np.uint8)), np.zeros((32, 64, 3), dtype=np.float32)
        )
