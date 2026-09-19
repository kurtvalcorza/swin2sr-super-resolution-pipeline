# Swin2SR 2x super-resolution pipeline

DIMER inference wrapper for **Swin2SR classical-sr-x2-64** (`caidas/swin2SR-classical-sr-x2-64`), pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The pipeline upscales one RGB image by exactly 2x and returns a uint8 array; it is the classical (bicubic-degradation) variant, not a compression-artefact or real-world restorer.

## Upstream alignment

- Model: `caidas/swin2SR-classical-sr-x2-64`
- Revision: `cee1c923c6a37361c6e5650b65dcf4be821e5d52`
- Upstream weight license: Apache-2.0
- Upstream task: single-image super-resolution, x2
- Repository adaptation: a bounded fine-tuning contract (`evaluate`, `adapt`, `save_artifact`, `from_artifact`) over the last encoder stage and the reconstruction tail with the L1 loss; the base weights are never modified on disk

## Quick start

```python
from PIL import Image
from swin2sr_super_resolution_pipeline import Swin2SRPipeline, psnr

pipe = Swin2SRPipeline.from_pretrained()            # stages + verifies weights/swin2sr-x2-64 first
result = pipe.upscale(Image.open("small.png"))      # sides 8–512 px, one image per call
Image.fromarray(result["image"]).save("large.png")  # uint8 (2H, 2W, 3)
print(result["input_size"], "->", result["output_size"])

# optional: score against a caller-supplied high-resolution original of the same shape
# print(psnr(result["image"], reference_uint8))
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the offline test suite (no weights needed). On a fresh clone the manifest is committed but the weights are not: `Swin2SRPipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

## Weights layout

```
weights/swin2sr-x2-64/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256
  config.json
  preprocessor_config.json
  model.safetensors          # git-ignored, 48,460,660 bytes
  README.md
```

## Adaptation contract

```python
from swin2sr_super_resolution_pipeline import (
    Swin2SRPipeline, bicubic_baseline, build_sample_dataset, fetch_corpus, load_byod_dataset, read_corpus, split_dataset,
)

splits = build_sample_dataset(read_corpus(fetch_corpus()), seed=42)   # 360 CC0 iNaturalist crops, 216 / 48 / 96
# or: splits = split_dataset(load_byod_dataset("my_photos.zip"), seed=42)  # your HR images, cropped + degraded the same way

pipe = Swin2SRPipeline.from_pretrained()                              # cuda:0 if available, else cpu
bicubic = bicubic_baseline(splits["test"])                            # psnr, ssim, per_category
frozen = pipe.evaluate(splits["test"])
result = pipe.adapt(splits["train"], splits["validation"], epochs=5, lr=1e-4, trainable_stages=1)
adapted = pipe.evaluate(splits["test"])
pipe.save_artifact("outputs/adapter")                                 # adapter.safetensors + manifest.json
again = Swin2SRPipeline.from_artifact("outputs/adapter")
```

- Records are `{id, lr, hr, category?}`: `hr` a PIL image with sides that are multiples of 16 px within 16..1024 px, `lr` exactly half its size; `validate_dataset` checks the shape (8..5,000 records, unique ids). `degrade(hr)` is the stated degradation — bicubic ÷2 then JPEG at `JPEG_QUALITY` (40) — and `split_dataset` de-duplicates by decoded high-resolution pixels; `check_split_disjoint` asserts no image is shared.
- The default sample (`samples.py`) is the 360 CC0 iNaturalist bird photographs also pinned by the fleet's SigLIP 2 tutorial (photo id, byte size and SHA-256 per file; fetched from the open-data bucket; cached git-ignored under `weights/inat-birds/`), each cut to a centred 192-px reference and its 96-px input; `build_sample_dataset` draws 36 / 8 / 16 per species.
- `evaluate(records)` upscales every record's input through `upscale` and returns `sr_metrics` (`metrics.py`): PSNR (dB, RGB) and SSIM (luma, 11-tap Gaussian window, 5-px border excluded), overall and per category, plus `per_record`, `verdict` (`measured` / `measured-small-sample`) and `adapted`. `nearest_baseline` and `bicubic_baseline` are the two non-neural references the tutorial scores beside the model.
- `adapt(train, val=None, *, epochs=5, lr=1e-4, batch_size=8, trainable_stages=1, seed=0, progress=None)` trains only the last `trainable_stages` residual Swin stages, `conv_after_body`, the pixel-shuffle upsampler and `final_convolution` (2,463,011 of 12,091,571 parameters by default) with the L1 loss the checkpoint was trained with; AdamW without weight decay, gradient clipping at 1.0, seeded shuffling; epoch 0 records the frozen model and the epoch with the highest validation PSNR is kept. Training records must share one input size. The update is transactional: an exception restores the frozen weights.
- `save_artifact(dir)` writes the trained tensors as `adapter.safetensors` plus a `manifest.json` (format `org.valcorza.swin2sr-x2-64.adapter.v1`: base id, revision and weight digest, tensor names, file size and SHA-256, training configuration, epoch history); `from_artifact(dir)` re-verifies the base snapshot, checks the manifest, the digest and the exact tensor set before deserialising, refuses any tensor outside the encoder stages and the reconstruction tail, and overlays the tensors onto a freshly loaded base.

## Input ceilings

`MIN_INPUT_SIDE = 8`, `MAX_INPUT_SIDE = 512` (512x512 takes 34 s on the reference CPU; the 1024 px measured during the card pass took 160 s and was ruled out for DIMER); one image per call. See `MODEL_CARD.md` for the measured timings behind those numbers.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/swin2sr-super-resolution-pipeline/blob/main/tutorials/swin2sr_super_resolution_colab.ipynb)

`tutorials/swin2sr_super_resolution_colab.ipynb` is declared `E2E` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the package's three modules (`pipeline.py`, `metrics.py`, `samples.py`), the model identity, the 4-file manifest digests and the runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py` and `tools/validate_release_assets.py`; see `tutorials/README.md`). It stages and digest-verifies the snapshot, fetches 360 digest-pinned CC0 iNaturalist photographs and pairs them under a stated degradation (192-px reference crop; input = bicubic ÷2 + JPEG 40), splits them per species without leakage, upscales a synthetic scene through the inference contract with an input manifest and a rejection probe, scores the frozen model's PSNR and SSIM on the 96 held-out crops beside the nearest-neighbour and bicubic baselines (the frozen model is below bicubic on this input), runs a bounded fine-tuning of the last encoder stage and the reconstruction tail with the L1 loss and validation-PSNR epoch selection, re-scores the held-out split per species, re-upscales the scene and four crops as side-by-side panels, exports the adapter and reloads it with verified pixel parity, and writes `swin2sr_super_resolution_train.csv`, `swin2sr_super_resolution_input_manifest.json`, `swin2sr_super_resolution_output_frozen.png`, `swin2sr_super_resolution_output_adapted.png`, `swin2sr_super_resolution_evaluation_report.json`, `swin2sr_super_resolution_examples/`, `swin2sr_super_resolution_adapter/` and `swin2sr_super_resolution_result.json` under `outputs/`.

The default path runs on CPU and uses CUDA automatically when present (about 25 minutes on the build workstation's CPU after the downloads, longer on a 2-vCPU hosted runtime; about 7 minutes on an RTX 5070 Ti). The metrics it prints are one seeded split of one 360-crop sample under one degradation — evidence that the adaptation contract works, not a restoration benchmark or production-fitness evidence.

## Release status

**Release-grade** — the `E2E` notebook blob `41a0b631` (committed at `e703ad1`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-20 (11/11 ok (1 restart after install cell), 596.8 s); the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted run is. A later change to the carried modules or the notebook returns the status to Candidate until re-verified.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests, input/output contract, measured runtime.
- `docs/WEIGHTS.md` — weight provenance and hosting notes.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
