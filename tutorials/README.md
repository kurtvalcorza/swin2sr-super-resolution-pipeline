# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/swin2sr-super-resolution-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/swin2sr-super-resolution-pipeline/blob/main/tutorials/swin2sr_super_resolution_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-caidas%2Fswin2SR--classical--sr--x2--64-ffcc4d?style=flat)](https://huggingface.co/caidas/swin2SR-classical-sr-x2-64)
[![Upstream](https://img.shields.io/badge/Upstream-mv--lab%2Fswin2sr-181717?style=flat&logo=github&logoColor=white)](https://github.com/mv-lab/swin2sr)
[![arXiv](https://img.shields.io/badge/arXiv-2209.11345-b31b1b.svg)](https://arxiv.org/abs/2209.11345)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `swin2sr_super_resolution_colab.ipynb` | `TASK-INFERENCE` | 2× single-image super-resolution (classical SR) with `caidas/swin2SR-classical-sr-x2-64`; uint8 `(2H, 2W, 3)` output; `psnr` against a self-made reference plus a bicubic baseline on the synthetic path only | CPU (CUDA used automatically when available) | single image file (8–512 px per side), gated off by default; no reference, so no PSNR | **Candidate** — static checks pass; the clean-runtime execution run is pending and will be recorded in `../docs/release-verification.md`, which must be reviewed for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `Swin2SRPipeline` from the repository public API rather than reimplementing model loading; model acquisition goes through the package: `stage_missing_files(WEIGHTS_DIR, allow_download=True)` fetches only the manifest entries a fresh clone lacks, at the pinned revision, `verify_snapshot` re-hashes every entry, and `from_pretrained(weights_dir=WEIGHTS_DIR)` loads the verified files (`local_files_only=True`, `trust_remote_code=False`; the notebook never calls `huggingface_hub`).
- The default sample is a synthetic 128×96 RGB image generated in code (gradient, square, stripe), bicubic-downscaled to 64×48 as the input; the PSNR of the model output and of a bicubic 2× baseline against that self-made reference is sanity evidence for the input contract and forward pass, not a benchmark (Set5/Set14/DIV2K figures are upstream-reported and not measured).
- Ceilings `UPSCALE` (2), `MIN_INPUT_SIDE` (8 px) and `MAX_INPUT_SIDE` (512 px) are printed before the model runs; the notebook states why the ceiling was lowered from 1024 px on 2026-09-12 (160 s vs 34 s on the reference CPU) and that larger inputs must be tiled.
- CPU is documented as adequate for the default input (card-measured 0.36 s for 64×48, 34 s for 512×512).
- `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
