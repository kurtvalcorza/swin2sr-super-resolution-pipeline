---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: image-to-image
base_model: caidas/swin2SR-classical-sr-x2-64
---

# Swin2SR classical-sr-x2-64 (DIMER package v0.1.0) — Image Super-Resolution (Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-caidas%2Fswin2SR--classical--sr--x2--64-ffcc4d?style=flat)](https://huggingface.co/caidas/swin2SR-classical-sr-x2-64)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-mv--lab%2Fswin2sr-181717?style=flat&logo=github&logoColor=white)](https://github.com/mv-lab/swin2sr)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2209.11345-b31b1b.svg)](https://arxiv.org/abs/2209.11345)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pipeline](https://img.shields.io/badge/Pipeline-swin2sr--super--resolution--pipeline-2ea44f?style=flat&logo=github)](https://github.com/kurtvalcorza/swin2sr-super-resolution-pipeline)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This release ships no tutorial notebook (`tutorials/` is absent). The package is exercised through its test suite (`tests/`) and the run instructions in the README; a `NOTEBOOK_SPEC` 1.0 `TASK-INFERENCE` notebook is a follow-up, not a claim this card makes.

---

###### Description

`caidas/swin2SR-classical-sr-x2-64` is the Transformers-format release of the Swin2SR classical 2x super-resolution model (Conde et al., arXiv:2209.11345), pinned here to revision `cee1c923c6a37361c6e5650b65dcf4be821e5d52`. The snapshot `config.json` declares `Swin2SRForImageSuperResolution`: a SwinV2 transformer with `patch_size` 1 (every pixel is a token), `embed_dim` 180, six residual Swin groups of six layers each (`depths` and `num_heads` all 6), `window_size` 8, `mlp_ratio` 2.0, a `1conv` residual connection, and a `pixelshuffle` upsampler with `upscale` 2; `image_size` 64 records the training patch size. At inference the model maps a padded RGB tensor in `[0, 1]` to a 2x-larger RGB reconstruction in one forward pass; nothing is trained, fine-tuned, or conditioned here. What this repository adds is packaging: `verify_snapshot` and `stage_missing_files` (manifest digest checking and fresh-clone staging), `Swin2SRPipeline.from_pretrained` (verified local loading, `trust_remote_code=False`), `upscale` (input validation, padding crop, uint8 conversion), and a `psnr` helper for caller-supplied high-resolution references.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is 2x single-image super-resolution: input one RGB still (`PIL.Image.Image`, any mode, converted to RGB) of side 8–512 px; output a uint8 array of shape `(2H, 2W, 3)` plus `scale`, `input_size`, `output_size`. Envisioned applications are enlarging small photographs or thumbnails for display, preparing low-resolution archival or product images for print or web layouts, and upsampling frames before a downstream detector or OCR stage that expects more pixels than the source provides. The checkpoint is the *classical* SR variant trained on bicubic-downsampled images, so the intended input is a clean image that was downscaled, not a compressed, noisy, or blurred one (upstream ships separate compressed-input and real-world variants that are not packaged here). Within DIMER the pipeline is an inference component and a baseline, not an image-authentication tool.

###### Primary Intended Users

Intended users are machine-learning engineers, imaging researchers, and application developers integrating a fixed-factor upscaler into research prototypes, internal enterprise tooling, or the DIMER workbench. A user is expected to understand that super-resolution hallucinates plausible detail rather than recovering true detail, that PSNR against a single reference is a sanity check and not a benchmark, that the output is only 2x (chaining calls compounds artefacts), and that the CPU cost grows with input area (34 s for 512x512, the ceiling, on the reference machine). Users who need arbitrary scale factors, denoising, JPEG-artefact removal, or face-specific restoration are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** only x2 upscaling of clean, bicubic-degraded RGB input; no x3/x4, no denoising, no JPEG or compression-artefact removal (the upstream `compressed-sr` and `realworld-sr` checkpoints are not packaged), no face restoration, no video with temporal consistency.
2. **Input boundary:** `upscale` rejects anything that is not a `PIL.Image.Image` (`TypeError`), any side below `MIN_INPUT_SIDE = 8` px or above `MAX_INPUT_SIDE = 512` px (`ValueError`); one image per call, no batching. Grayscale, palette, and RGBA inputs are converted to RGB and alpha is discarded.
3. **Input boundary:** inputs that were not produced by downscaling a sharp image — screenshots, scanned text, medical or scientific imagery, heavily compressed social-media photos — fall outside the training degradation; the pipeline does not detect them and the output on them is undefined.
4. **Decision boundary:** not for forensic enhancement (licence plates, faces, documents) presented as recovered evidence, and not for any diagnostic or measurement use where invented pixels could change a decision.

#### Factors

###### Groups

This pipeline is not human-centric: it regresses pixels and neither classifies nor identifies people. Photographs given to it may nonetheless contain faces and bodies, and the classical-SR training sets the upstream paper names (DIV2K and Flickr2K; the snapshot README itself carries no training-data disclosure) are general photo collections that neither the upstream authors nor this repository have audited for skin tone, age, gender, or other group balance. Any difference in reconstruction fidelity across such groups is therefore unknown, not known to be absent. A downstream operator who upscales images of people is responsible for a fairness audit on their own data: score `psnr` (or a perceptual metric of their choosing) per group on held-out high-resolution originals and compare strata before relying on the output.

###### Instrumentation

The upstream training pairs are synthetic: high-resolution photographs from DIV2K/Flickr2K (per the Swin2SR paper) downscaled by bicubic interpolation to produce the low-resolution inputs, so the "instrument" is a resampling kernel rather than a camera, and the model learns to invert that specific kernel. Inference inputs come from whatever produced the caller's small image — a phone camera, a web thumbnail generator, a scanner, a video frame — and each differs from bicubic downscaling in blur, noise, sharpening, and compression; those differences reach the model as unmodelled degradation and can appear in the output as ringing or invented texture. The pipeline pads each input to a multiple of 8 px (`preprocessor_config.json`: `pad_size` 8) and crops the padding back off; it validates type and size only and cannot detect the capture chain.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, float32 on CPU; CUDA is used automatically when visible but was not exercised for this card. Measured on the reference machine with the GPU hidden (`CUDA_VISIBLE_DEVICES=""`): load 4.0–16.6 s (cold file cache versus warm), 64x48 input 0.36 s, 256x256 7.3 s, 512x512 33.9 s, 1024x1024 160.0 s; time scales roughly with input area, which is why `MAX_INPUT_SIDE` was set to 512 (the 1024 px run is retained above as the measurement that motivated the ceiling). Data environment: the model assumes a sharp photograph that was bicubically downscaled by exactly 2x; sharper-than-expected inputs produce over-sharpening and halos, blurrier inputs are not deblurred, and compressed inputs have their block artefacts enlarged rather than removed. Lighting, colour, and scene content are unrestricted within ordinary photographs.

#### Metrics

###### Performance Measures

The only measure the repository reports is `psnr(pred, ref)`: peak signal-to-noise ratio in dB, `10 * log10(255**2 / MSE)` over all RGB channels of two uint8 arrays of identical shape, returning `inf` for identical arrays. PSNR captures pixel-wise reconstruction error and is the standard first metric for classical SR, which is why it was chosen over perceptual metrics that need an extra model; its known blind spot is that it rewards blur over texture, so a reader who sees only PSNR cannot tell a sharp result from a smooth one. It requires a caller-supplied high-resolution reference; on an image without one the pipeline reports nothing. The smoke run's `psnr(model, bicubic)` = 35.21 dB compares the model to a bicubic upscale of the same synthetic input and is a consistency check, not a quality score. The upstream paper's Set5/Set14/Urban100 numbers are not reproduced or claimed.

###### Decision thresholds

No decision threshold is applied. `upscale` returns the reconstruction clamped to `[0, 1]`, scaled to 0–255, rounded to uint8; that clamp-and-round is the only value-level rule in the code path and is not a threshold on any score. `psnr` returns a number without judging it. A deployment that wants an acceptance rule — for example a minimum PSNR on a held-out set of its own originals before a model update ships — must set it against its own data, weighing the cost of invented detail that misleads a viewer against the cost of rejecting usable enlargements, and owns revisiting it when the source-image chain changes.

###### Approaches to uncertainty and variability

This repository reports no central metric value and therefore no dispersion: the smoke run records timings and one PSNR against a bicubic baseline, not accuracy against ground truth. Run-to-run variability comes from floating-point kernel selection across CPU builds and accelerators and from the uint8 rounding at the output; there is no sampling, no dropout at inference, and no seed to set, so a fixed input on fixed hardware is repeatable but not guaranteed bitwise-identical across machines. The model emits no confidence map and no per-pixel uncertainty. A caller who needs an uncertainty estimate must supply high-resolution references and compute `psnr` over many images or bootstrap resamples themselves.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The snapshot README discloses no training data; the Swin2SR paper reports training the classical-SR models on DIV2K and Flickr2K, public photo collections whose per-image consent and licence status the paper does not enumerate, so whether personal data (faces, private property) is present is unknown, not ruled out. This repository distributes code, tests, and documentation; it does not distribute the 48,460,660-byte `model.safetensors`, which is staged locally under `weights/swin2sr-x2-64/` and git-ignored, and it ships no sample images. The operator must audit the images they submit for personal, proprietary, or otherwise restricted content and for consent to enhancement; the pipeline performs no such check.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Foreseeable but unintended sensitive uses — enlarging medical images before a read, enhancing surveillance stills for identification, restoring documents for legal proceedings — would be admissible only with the original low-resolution image retained and shown alongside, human review that treats added detail as synthetic, domain validation on the deployment's own data, and whatever regulatory or evidentiary clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks every listed file's byte size and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True` and always passes `trust_remote_code=False`. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused.
- **Input integrity:** `validate_image` rejects non-PIL inputs and sides outside 8–512 px before the model runs; `upscale` raises if the backend returns anything other than a `(2H, 2W, 3)` uint8 array.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; every result carries `model_id` and `model_revision`.
- **Refusals:** no x4, no compressed/real-world variants, no download without the explicit flag.
- No statistical mitigation (class balancing, subsampling) applies: the model is a dense regressor with no classes.

###### Risks and harms

- **Invented detail:** the model synthesises texture that was never in the source; a viewer, the data subject, or a third party bears the harm when that detail is read as real (a face, a digit, a lesion); likely on any input with fine structure; magnitude ranges from cosmetic to a wrong identification.
- **Degradation mismatch:** inputs that are compressed, noisy, or sharpened rather than bicubically downscaled produce halos and enlarged artefacts; the operator bears the harm; likely for web-sourced images.
- **Automation bias:** a crisp output invites more trust than a blurry input, so reviewers may stop questioning it.
- **Privacy exposure:** images of people or private spaces are processed without any content check; the data subject bears the harm.
- **Bias amplification:** any under-representation in DIV2K/Flickr2K is reproduced as lower fidelity for those inputs, undetected because no per-group evaluation exists.
- **Resource failure:** a 512x512 input (the ceiling) took 34 s and a 1024x1024 input 160 s on the reference CPU; memory grows with area, so a request stream near the ceiling can still exhaust a shared host.

###### Use cases

Prohibited even where the model would work: enhancing images for covert surveillance, biometric identification, or demographic profiling; presenting upscaled output as authentic evidence or as a faithful record of a document, face, or scene; social scoring; and any use that discriminates unlawfully in employment, housing, credit, insurance, education, or healthcare access. Also prohibited are deceptive or non-consensual uses — enhancing intimate or private images, or restoring content whose subject has not consented — and any use that violates the upstream Apache-2.0 licence terms, the DIMER deployment terms, or the data-protection obligations attached to the images processed.

## Immutable provenance

- Model: `caidas/swin2SR-classical-sr-x2-64`
- Revision: `cee1c923c6a37361c6e5650b65dcf4be821e5d52`
- Snapshot manifest: `weights/swin2sr-x2-64/dimer-base-manifest.json`, 4 files, `totalBytes` 48462226
- `model.safetensors` SHA-256: `a515955815ab6dba3a9331d8c75db13a5166a5ffbcebf1a95ab676a5656ac4fb` (48,460,660 bytes)
- `config.json` SHA-256: `11179bdfd48394977f7fe6177b3a87d9c2c6fa2a67a4b3a7376229ca407f0376` (772 bytes)
- Weight format: SafeTensors; loader `Swin2SRForImageSuperResolution.from_pretrained(<dir>, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False)`

## Input/output contract

- `Swin2SRPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — stages missing manifest files (only with `allow_download=True`), verifies digests, loads; `device` defaults to `cuda:0` when visible, else `cpu`.
- `upscale(image: PIL.Image.Image) -> dict` with keys `image` (uint8 `(2H, 2W, 3)` RGB), `scale` (2), `input_size` (`(W, H)`), `output_size` (`(2W, 2H)`), `model_id`, `model_revision`.
- Ceilings: `MIN_INPUT_SIDE = 8`, `MAX_INPUT_SIDE = 512` (lowered from the 1024 px executed during the card pass, by owner decision on 2026-09-12: 512² costs 34 s on the reference CPU, 1024² 160 s); `UPSCALE = 2`.
- `psnr(pred, ref) -> float` — two uint8 arrays of identical shape; `inf` when identical; `ValueError`/`TypeError` otherwise.
- `verify_snapshot(path=None) -> dict`, `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.

## Runtime

- Pins: `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`; Python 3.12.
- Precision: float32; preprocessing is rescale to `[0, 1]` and pad to a multiple of 8 (`Swin2SRImageProcessor` from the snapshot); the padding is cropped off at 2x on output.
- Measured 2026-09-12 in the Windows venv (`torch 2.14.0+cu130`) with `CUDA_VISIBLE_DEVICES=""`, device `cpu`: `verify_snapshot` 0.03–0.05 s; load 16.62 s cold / 4.01 s warm; `upscale` on a synthetic 64x48 gradient-and-square image 0.356 s cold / 0.315 s warm, output `(96, 128, 3)` uint8, `psnr` vs bicubic 35.21 dB; 256x256 → 7.29 s; 512x512 → 33.90 s; 1024x1024 → 160.04 s (process wall 169 s including load).
- Tests: `pytest -q -o addopts= tests` — 10 passed, offline, no weights required; `ruff check src tests` clean.
- Not executed: CUDA path, half precision, any accuracy measurement against a true high-resolution reference.

## References

- Conde, Choi, Burchi, Timofte. Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration. ECCV 2022 Workshops. https://arxiv.org/abs/2209.11345
- Liu et al. Swin Transformer V2: Scaling Up Capacity and Resolution. CVPR 2022. https://arxiv.org/abs/2111.09883
- Upstream code: https://github.com/mv-lab/swin2sr
- Upstream card: https://huggingface.co/caidas/swin2SR-classical-sr-x2-64
- Transformers `Swin2SR` documentation: https://huggingface.co/docs/transformers/model_doc/swin2sr
