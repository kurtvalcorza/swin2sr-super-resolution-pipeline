---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: image-to-image
task: "Others - Image Super-Resolution"
base_model: caidas/swin2SR-classical-sr-x2-64
date_published: "2022-12-16"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/caidas/swin2SR-classical-sr-x2-64)"
---

# Swin2SR classical-sr-x2-64 — Image Super-Resolution (Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-caidas%2Fswin2SR--classical--sr--x2--64-ffcc4d?style=flat)](https://huggingface.co/caidas/swin2SR-classical-sr-x2-64)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-mv--lab%2Fswin2sr-181717?style=flat&logo=github&logoColor=white)](https://github.com/mv-lab/swin2sr)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2209.11345-b31b1b.svg)](https://arxiv.org/abs/2209.11345)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, stage and verify the pinned upstream revision, fetch and validate a digest-pinned paired photograph set, measure the frozen model against two non-neural baselines, run a bounded fine-tuning, evaluate on an image-disjoint split, and export and reload the adapter:

- **E2E Fine-tuning Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/swin2sr-super-resolution-pipeline/blob/main/tutorials/swin2sr_super_resolution_colab.ipynb) [`swin2sr_super_resolution_colab.ipynb`](https://github.com/kurtvalcorza/swin2sr-super-resolution-pipeline/blob/main/tutorials/swin2sr_super_resolution_colab.ipynb)  
  *2× single-image super-resolution with the pinned `caidas/swin2SR-classical-sr-x2-64` weights, then bounded supervised fine-tuning on 360 CC0 iNaturalist crops paired under a stated degradation (bicubic ÷2 + JPEG 40): the synthetic scene through the inference contract, the frozen model's PSNR and SSIM beside the nearest-neighbour and bicubic baselines (the classical checkpoint scores below bicubic on JPEG input), the L1 loss over the last encoder stage and the reconstruction tail with validation-PSNR epoch selection, held-out evaluation per species, side-by-side panels, and a safetensors adapter that reloads with verified pixel parity.*

---

#### Description

`caidas/swin2SR-classical-sr-x2-64` is the Transformers-format release of the Swin2SR classical 2x super-resolution model (Conde et al., arXiv:2209.11345), pinned here to revision `cee1c923c6a37361c6e5650b65dcf4be821e5d52`. The snapshot `config.json` declares `Swin2SRForImageSuperResolution`: a SwinV2 transformer with `patch_size` 1 (every pixel is a token), `embed_dim` 180, six residual Swin groups of six layers each (`depths` and `num_heads` all 6), `window_size` 8, `mlp_ratio` 2.0, a `1conv` residual connection, and a `pixelshuffle` upsampler with `upscale` 2; `image_size` 64 records the training patch size. At inference the model maps a padded RGB tensor in `[0, 1]` to a 2x-larger RGB reconstruction in one forward pass; nothing is trained, fine-tuned, or conditioned here. What this repository adds is packaging: `verify_snapshot` and `stage_missing_files` (manifest digest checking and fresh-clone staging), `Swin2SRPipeline.from_pretrained` (verified local loading, `trust_remote_code=False`), `upscale` (input validation, padding crop, uint8 conversion), and a `psnr` helper for caller-supplied high-resolution references.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is 2x single-image super-resolution: input one RGB still (`PIL.Image.Image`, any mode, converted to RGB) of side 8–512 px; output a uint8 array of shape `(2H, 2W, 3)` plus `scale`, `input_size`, `output_size`. Envisioned applications are enlarging small photographs or thumbnails for display, preparing low-resolution archival or product images for print or web layouts, and upsampling frames before a downstream detector or OCR stage that expects more pixels than the source provides. The checkpoint is the *classical* SR variant trained on bicubic-downsampled images, so the intended input is a clean image that was downscaled, not a compressed, noisy, or blurred one (upstream ships separate compressed-input and real-world variants that are not packaged here). The pipeline is an inference component and a baseline, not an image-authentication tool.

###### Primary Intended Users

Intended users are machine-learning engineers, imaging researchers, and application developers integrating a fixed-factor upscaler into research prototypes or in-house tooling. A user is expected to understand that super-resolution hallucinates plausible detail rather than recovering true detail, that PSNR against a single reference is a sanity check and not a benchmark, that the output is only 2x (chaining calls compounds artefacts), and that the CPU cost grows with input area (34 s for 512x512, the ceiling, on the reference machine). Users who need arbitrary scale factors, denoising, JPEG-artefact removal, or face-specific restoration are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** only x2 upscaling; no x3/x4, no denoising, no blind restoration of unknown degradations (the upstream `compressed-sr` and `realworld-sr` checkpoints are not packaged), no face restoration, no video with temporal consistency. The frozen checkpoint expects clean, bicubic-degraded input; the adaptation contract can teach it one stated degradation (the tutorial's bicubic ÷2 + JPEG 40) from paired crops, and only that one. **Adaptation boundary:** `adapt` trains the last `trainable_stages` encoder stages, `conv_after_body`, the upsampler and `final_convolution` only; the patch embedding, the first convolution and the earlier stages stay frozen; training records must share one input size; a pair is validated for shape (reference sides multiples of 16 within 16..1024 px, input exactly half), never for alignment or for matching your deployment's degradation.
2. **Input boundary:** `upscale` rejects anything that is not a `PIL.Image.Image` (`TypeError`), any side below `MIN_INPUT_SIDE = 8` px or above `MAX_INPUT_SIDE = 512` px (`ValueError`); one image per call, no batching. Grayscale, palette, and RGBA inputs are converted to RGB and alpha is discarded.
3. **Input boundary:** inputs that were not produced by downscaling a sharp image — screenshots, scanned text, medical or scientific imagery, heavily compressed social-media photos — fall outside the training degradation; the pipeline does not detect them and the output on them is undefined.
4. **Decision boundary:** not for forensic enhancement (licence plates, faces, documents) presented as recovered evidence, and not for any diagnostic or measurement use where invented pixels could change a decision.

#### Factors

###### Groups

This pipeline is not human-centric: it regresses pixels and neither classifies nor identifies people. Photographs given to it may nonetheless contain faces and bodies, and the classical-SR training sets the upstream paper names (DIV2K and Flickr2K; the snapshot README itself carries no training-data disclosure) are general photo collections that neither the upstream authors nor this repository have audited for skin tone, age, gender, or other group balance. Any difference in reconstruction fidelity across such groups is therefore unknown, not known to be absent. A downstream operator who upscales images of people is responsible for a fairness audit on their own data: score `psnr` (or a perceptual metric of their choosing) per group on held-out high-resolution originals and compare strata before relying on the output.

###### Instrumentation

The upstream training pairs are synthetic: high-resolution photographs from DIV2K/Flickr2K (per the Swin2SR paper) downscaled by bicubic interpolation to produce the low-resolution inputs, so the "instrument" is a resampling kernel rather than a camera, and the model learns to invert that specific kernel. Inference inputs come from whatever produced the caller's small image — a phone camera, a web thumbnail generator, a scanner, a video frame — and each differs from bicubic downscaling in blur, noise, sharpening, and compression; those differences reach the model as unmodelled degradation and can appear in the output as ringing or invented texture. The pipeline pads each input to a multiple of 8 px (`preprocessor_config.json`: `pad_size` 8) and crops the padding back off; it validates type and size only and cannot detect the capture chain.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, float32 on CPU; CUDA is used automatically when visible but was not exercised for this card. Measured on the reference machine with the GPU hidden (`CUDA_VISIBLE_DEVICES=""`): load 4.0–16.6 s (cold file cache versus warm), 64x48 input 0.36 s, 256x256 7.3 s, 512x512 33.9 s, 1024x1024 160.0 s; time scales roughly with input area, which is why `MAX_INPUT_SIDE` was set to 512 (the 1024 px run is retained above as the measurement that motivated the ceiling). Data environment: the model assumes a sharp photograph that was bicubically downscaled by exactly 2x; sharper-than-expected inputs produce over-sharpening and halos, blurrier inputs are not deblurred, and compressed inputs have their block artefacts enlarged rather than removed. Lighting, colour, and scene content are unrestricted within ordinary photographs.

#### Metrics

###### Performance Measures

Over a paired dataset the repository reports `sr_metrics` (`metrics.py`): **PSNR** (dB over the RGB channels against the high-resolution reference) and **SSIM** (luma channel, 11-tap Gaussian window of sigma 1.5, 5-px border excluded — the classical-SR convention), averaged over the records and per `category`; `nearest_baseline` and `bicubic_baseline` (2× interpolation of the same input) answer through the same result shape. On the tutorial's 96-crop test split (RTX 5070 Ti build record, seed 42, bicubic ÷2 + JPEG 40): nearest 25.29 dB / 0.764, bicubic 26.31 / 0.797, frozen 25.92 / 0.785 (below bicubic on every species — the classical checkpoint sharpens block artefacts), adapted (one stage + tail, lr 1e-4, five epochs, epoch 5 kept) 26.91 / 0.816 — every species gained about a decibel. These are observations on one seeded split under one degradation with no dispersion estimate, not a benchmark. The single-image measure is `psnr(pred, ref)`: peak signal-to-noise ratio in dB, `10 * log10(255**2 / MSE)` over all RGB channels of two uint8 arrays of identical shape, returning `inf` for identical arrays. PSNR captures pixel-wise reconstruction error and is the standard first metric for classical SR, which is why it was chosen over perceptual metrics that need an extra model; its known blind spot is that it rewards blur over texture, so a reader who sees only PSNR cannot tell a sharp result from a smooth one. It requires a caller-supplied high-resolution reference; on an image without one the pipeline reports nothing. The smoke run's `psnr(model, bicubic)` = 35.21 dB compares the model to a bicubic upscale of the same synthetic input and is a consistency check, not a quality score. The upstream paper's Set5/Set14/Urban100 numbers are not reproduced or claimed. The public `evaluation_report(result, reference, low_resolution=...)` helper is the only reporting path: it emits a machine-readable report whose verdict is `sample-sanity` with the `psnr` metric and a bicubic-2× baseline when a high-resolution reference is supplied, and `not-measurable` otherwise, stating in that case which benchmark HR/LR pairs would make the task measurable.

###### Decision thresholds

No decision threshold is applied. `upscale` returns the reconstruction clamped to `[0, 1]`, scaled to 0–255, rounded to uint8; that clamp-and-round is the only value-level rule in the code path and is not a threshold on any score. `psnr` returns a number without judging it. A deployment that wants an acceptance rule — for example a minimum PSNR on a held-out set of its own originals before a model update ships — must set it against its own data, weighing the cost of invented detail that misleads a viewer against the cost of rejecting usable enlargements, and owns revisiting it when the source-image chain changes.

###### Approaches to uncertainty and variability

Adaptation is seeded (`seed=0`: shuffling order) but not bit-reproducible across devices; every corpus metric the tutorial reports is one value on one seeded split (`build_sample_dataset(seed=42)`) of one 360-crop sample under one degradation, with a 48-crop validation split selecting the epoch and the training loss still falling at the last epoch (the recipe is bounded by budget, not convergence). For the single-image path this repository reports no central metric value and therefore no dispersion: the smoke run records timings and one PSNR against a bicubic baseline, not accuracy against ground truth. Run-to-run variability comes from floating-point kernel selection across CPU builds and accelerators and from the uint8 rounding at the output; there is no sampling, no dropout at inference, and no seed to set, so a fixed input on fixed hardware is repeatable but not guaranteed bitwise-identical across machines. The model emits no confidence map and no per-pixel uncertainty. A caller who needs an uncertainty estimate must supply high-resolution references and compute `psnr` over many images or bootstrap resamples themselves.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The snapshot README discloses no training data; the Swin2SR paper reports training the classical-SR models on DIV2K and Flickr2K, public photo collections whose per-image consent and licence status the paper does not enumerate, so whether personal data (faces, private property) is present is unknown, not ruled out. This repository distributes code, tests, and documentation; it does not distribute the 48,460,660-byte `model.safetensors`, which is staged locally under `weights/swin2sr-x2-64/` and git-ignored, and it ships no sample images. The operator must audit the images they submit for personal, proprietary, or otherwise restricted content and for consent to enhancement; the pipeline performs no such check.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Foreseeable but unintended sensitive uses — enlarging medical images before a read, enhancing surveillance stills for identification, restoring documents for legal proceedings — would be admissible only with the original low-resolution image retained and shown alongside, human review that treats added detail as synthetic, domain validation on the deployment's own data, and whatever regulatory or evidentiary clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks every listed file's byte size and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True` and always passes `trust_remote_code=False`. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused.
- **Input integrity:** `validate_image` rejects non-PIL inputs and sides outside 8–512 px before the model runs; `upscale` raises if the backend returns anything other than a `(2H, 2W, 3)` uint8 array. The public `validate_inputs(images, names=...)` stage routes every image through that same `validate_image`, so it raises exactly what `upscale` raises while returning a machine-readable input manifest of the schema, ceilings, per-input observations and verdict.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; every result carries `model_id` and `model_revision`.
- **Refusals:** no x4, no compressed/real-world variants, no download without the explicit flag.
- **Adaptation integrity:** `adapt` validates the dataset before any tensor is built, trains only the named tensors with every other parameter's `requires_grad` false, restores the frozen weights on any exception, and records the configuration and epoch history in the artifact; `from_artifact` re-verifies the base snapshot and checks the manifest's format, base identity and weight digest, the file size and SHA-256 and the exact tensor set **before** deserialising, refuses any tensor outside the encoder stages and the reconstruction tail, and overlays onto a freshly loaded base.
- No statistical mitigation (class balancing, subsampling) applies: the model is a dense regressor with no classes.

###### Risks and harms

- **Invented detail:** the model synthesises texture that was never in the source; a viewer, the data subject, or a third party bears the harm when that detail is read as real (a face, a digit, a lesion); likely on any input with fine structure; magnitude ranges from cosmetic to a wrong identification.
- **Degradation mismatch:** inputs that are compressed, noisy, or sharpened rather than bicubically downscaled produce halos and enlarged artefacts; the operator bears the harm; likely for web-sourced images.
- **Automation bias:** a crisp output invites more trust than a blurry input, so reviewers may stop questioning it.
- **Privacy exposure:** images of people or private spaces are processed without any content check; the data subject bears the harm.
- **Bias amplification:** any under-representation in DIV2K/Flickr2K is reproduced as lower fidelity for those inputs, undetected because no per-group evaluation exists.
- **Resource failure:** a 512x512 input (the ceiling) took 34 s and a 1024x1024 input 160 s on the reference CPU; memory grows with area, so a request stream near the ceiling can still exhaust a shared host.

###### Use cases

Prohibited even where the model would work: enhancing images for covert surveillance, biometric identification, or demographic profiling; presenting upscaled output as authentic evidence or as a faithful record of a document, face, or scene; social scoring; and any use that discriminates unlawfully in employment, housing, credit, insurance, education, or healthcare access. Also prohibited are deceptive or non-consensual uses — enhancing intimate or private images, or restoring content whose subject has not consented — and any use that violates the upstream Apache-2.0 licence terms, the terms of the deployment that runs the pipeline, or the data-protection obligations attached to the images processed.

## Immutable provenance

- Model: `caidas/swin2SR-classical-sr-x2-64`
- Revision: `cee1c923c6a37361c6e5650b65dcf4be821e5d52`
- Snapshot manifest: `weights/swin2sr-x2-64/dimer-base-manifest.json`, 4 files, `totalBytes` 48462226
- `model.safetensors` SHA-256: `a515955815ab6dba3a9331d8c75db13a5166a5ffbcebf1a95ab676a5656ac4fb` (48,460,660 bytes)
- `config.json` SHA-256: `11179bdfd48394977f7fe6177b3a87d9c2c6fa2a67a4b3a7376229ca407f0376` (772 bytes)
- Weight format: SafeTensors; loader `Swin2SRForImageSuperResolution.from_pretrained(<dir>, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False)`
- Adapter artifact format: `org.valcorza.swin2sr-x2-64.adapter.v1` — `adapter.safetensors` (the trained tensors only; 126 tensors, 9,868,140 bytes for the default one stage + tail) plus `manifest.json` naming the base id, revision and `model.safetensors` digest, the tensor names, the file size and SHA-256, the training configuration and the epoch history.
- Tutorial corpus: 360 iNaturalist photographs (CC0 1.0; six bird species, 60 each, one per observer), `CORPUS_BYTES = 39,223,447`, each pinned by photo id, byte size and SHA-256 in `samples.py` and fetched from `https://inaturalist-open-data.s3.amazonaws.com/photos/<id>/medium.<ext>`; paired as a centred 192-px reference and a 96-px input (`degrade`: bicubic ÷2, JPEG quality 40); split 216 / 48 / 96 by `build_sample_dataset(seed=42)`; default draw digest `827eb4a31619d7073a825ce5e5a46b58f26e07b3c920d6cf75b6f11a1241ca83` (`SAMPLE_DIGEST`).

## Input/output contract

- `Swin2SRPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — stages missing manifest files (only with `allow_download=True`), verifies digests, loads; `device` defaults to `cuda:0` when visible, else `cpu`.
- `upscale(image: PIL.Image.Image) -> dict` with keys `image` (uint8 `(2H, 2W, 3)` RGB), `scale` (2), `input_size` (`(W, H)`), `output_size` (`(2W, 2H)`), `model_id`, `model_revision`.
- Ceilings: `MIN_INPUT_SIDE = 8`, `MAX_INPUT_SIDE = 512` (lowered from the 1024 px executed during the card pass, by owner decision on 2026-09-12: 512² costs 34 s on the reference CPU, 1024² 160 s); `UPSCALE = 2`.
- `psnr(pred, ref) -> float` — two uint8 arrays of identical shape; `inf` when identical; `ValueError`/`TypeError` otherwise.
- `verify_snapshot(path=None) -> dict`, `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.
- `evaluate(records, *, progress=None) -> dict` — upscales every validated record's `lr` and returns `sr_metrics` plus `verdict`, `adapted`, `seconds`; `adapt(train, val=None, *, epochs=5, lr=1e-4, batch_size=8, trainable_stages=1, seed=0, progress=None) -> dict` — bounded L1 fine-tuning with validation-PSNR epoch selection, transactional; `save_artifact(dir, metadata=None) -> Path`; `load_artifact(dir) -> dict`; `Swin2SRPipeline.from_artifact(dir, *, device=None, weights_dir=None, allow_download=False)`.
- Dataset contract (`samples.py`): records `{id, lr, hr, category?}`; `degrade(hr, *, quality=40)`; `centre_crop(image, size=192)`; `validate_dataset(records, *, min_records=8, max_records=5000) -> dict`; `split_dataset(records, *, val_fraction=0.15, test_fraction=0.2, seed=0)`; `check_split_disjoint(splits)`; `observer_overlap(splits)`; `image_digest(image)`; `dataset_digest(records)`; `load_byod_dataset(path, *, crop=192, quality=40)`; `write_dataset_csv(records, path)`; `export_pairs(records, directory)`; `fetch_corpus(cache_dir=None, fetcher=None)`, `read_corpus(files, *, crop=192, quality=40)`, `build_sample_dataset(records, *, seed=42, sizes=SAMPLE_SPLIT)`, `fetch_sample_dataset(cache_dir=None, seed=42)`.
- Metrics (`metrics.py`): `sr_metrics(outputs, records)`, `ssim(pred, ref)`, `luma(rgb)`, `nearest_baseline(records)`, `bicubic_baseline(records)`, `METRIC_DEFINITIONS`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`; Python 3.12.
- Precision: float32; preprocessing is rescale to `[0, 1]` and pad to a multiple of 8 (`Swin2SRImageProcessor` from the snapshot); the padding is cropped off at 2x on output.
- Measured 2026-09-12 in the Windows venv (`torch 2.14.0+cu130`) with `CUDA_VISIBLE_DEVICES=""`, device `cpu`: `verify_snapshot` 0.03–0.05 s; load 16.62 s cold / 4.01 s warm; `upscale` on a synthetic 64x48 gradient-and-square image 0.356 s cold / 0.315 s warm, output `(96, 128, 3)` uint8, `psnr` vs bicubic 35.21 dB; 256x256 → 7.29 s; 512x512 → 33.90 s; 1024x1024 → 160.04 s (process wall 169 s including load).
- Tests: `pytest -q -o addopts= tests` — offline tests (no weights required) plus `tests/test_model_backed.py` (7 tests, skipped without `torch` or the snapshot; the CUDA test skipped without a device), run 2026-09-20 on CPU in the Windows venv and on the RTX 5070 Ti in WSL (7/7); `ruff check src tests tools` clean.
- Tutorial execution: the `E2E` `tutorials/swin2sr_super_resolution_colab.ipynb` ran top-to-bottom in a fresh local CPU kernel on 2026-09-20 (all 11 code cells, 1,523.2 s with the snapshot and photographs pre-staged; frozen test 25.92 dB / 0.785 in 70.8 s, five epochs 1,314.5 s, epoch 5 kept, adapted 26.91 / 0.816, reload parity 8/8; the clean synthetic scene fell from 37.1 to 33.5 dB after adaptation) and on the RTX 5070 Ti in WSL (419.4 s; identical metrics to three decimals); recorded in `docs/release-verification.md` as pre-flight, not supported-runtime evidence.
- Executed 2026-09-20: the **committed notebook blob** (`e703ad1` / `41a0b631`) run top-to-bottom on a clean Kaggle Tesla T4 kernel (`kurtvalcorza/dimer-nb2-swin2sr-super-resolution` v2, `torch 2.14.0+cu130`, Python 3.12.13, `cuda:0`, empty Hugging Face cache, no repository checkout, blob SHA-1 verified against GitHub before execution): 11/11 ok (1 restart after install cell), 596.8 s, 370 files, 88 MB fetched from the Hub and digest-verified inside the notebook; comparison {psnr: {nearest: 25.29, bicubic: 26.31, frozen: 25.92, adapted: 26.91}, ssim: {nearest: 0.764, bicubic: 0.797, frozen: 0.785, adapted: 0.816}, delta_vs_frozen: {psnr: 0.991, ssim: 0.032}, delta_vs_bicubic: {psnr: 0.6, ssim: 0.019}, by_species: {american_goldfinch: {n: 16, frozen_psnr: 26.73, adapted_psnr: 27.87, frozen_ssim: 0.832, adapted_ssim: 0.871}, chipping_sparrow: {n: 16, frozen_psnr: 25.81, adapted_psnr: 26.61, frozen_ssim: 0.764, adapted_ssim: 0.788}, dark_eyed_junco: {n: 16, frozen_psnr: 25.76, adapted_psnr: 26.97, frozen_ssim: 0.766, adapted_ssim: 0.801}, house_finch: {n: 16, frozen_psnr: 26.17, adapted_psnr: 27.35, frozen_ssim: 0.796, adapted_ssim: 0.827}, song_sparrow: {n: 16, frozen_psnr: 25.57, adapted_psnr: 26.36, frozen_ssim: 0.781, adapted_ssim: 0.811}, white_throated_sparrow: {n: 16, frozen_psnr: 25.48, adapted_psnr: 26.31, frozen_ssim: 0.77, adapted_ssim: 0.8}}}; reload parity {identical_images: 8, of: 8, max_abs_difference: 0}. Recorded in `docs/release-verification.md`.
- Not executed: half precision, any degradation other than bicubic ÷2 + JPEG 40, any corpus other than the one iNaturalist sample, repeated seeds or splits (no dispersion), BYOD, two or more trainable stages (an 8-epoch two-stage probe was stopped after 37 minutes on the GPU without finishing), and the adapted model on any images but that test split.

## References

- Conde, Choi, Burchi, Timofte. Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration. ECCV 2022 Workshops. https://arxiv.org/abs/2209.11345
- Liu et al. Swin Transformer V2: Scaling Up Capacity and Resolution. CVPR 2022. https://arxiv.org/abs/2111.09883
- Upstream code: https://github.com/mv-lab/swin2sr
- Upstream card: https://huggingface.co/caidas/swin2SR-classical-sr-x2-64
- Transformers `Swin2SR` documentation: https://huggingface.co/docs/transformers/model_doc/swin2sr
