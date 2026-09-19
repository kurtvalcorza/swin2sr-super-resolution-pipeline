# Release verification

`tutorials/swin2sr_super_resolution_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 4-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `Swin2SRPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path, `read_corpus`,
  `build_sample_dataset(corpus, seed=SPLIT_SEED)` / `load_byod_dataset` + `split_dataset`, `validate_dataset` per
  split, `check_split_disjoint`, `observer_overlap`, `write_dataset_csv`, the four dataset refusal probes, the
  ceiling print, `validate_inputs` with the oversize refusal probe, `upscale` with the sanity checks and the
  per-image `evaluation_report` on the synthetic scene, `nearest_baseline`, `bicubic_baseline`, `pipe.evaluate` on
  the frozen model with the floor assertion, `pipe.adapt` with its explicit hyperparameters, `pipe.evaluate` on the
  validation and test splits after adaptation with the PSNR assertion, `upscale` + `evaluation_report` on the scene
  after adaptation, the example panels against a freshly loaded frozen base, `pipe.save_artifact`,
  `Swin2SRPipeline.from_artifact` and the pixel-parity assertion, and the result fields `weight_file` /
  `weight_format` / `weight_sha256` and the `corpus` block), the eight expected `outputs/` paths, the learner-facing
  statements (Apache-2.0 weights, synthesised detail, adaptation with paired crops, the CC0 corpus, the JPEG-40
  degradation, the frozen model below bicubic, PSNR and SSIM, the two non-neural baselines, the L1 loss, no
  dispersion estimate, the degradation and leakage guidance, the excluded tasks, the snapshot note) and the
  gated-off BYOD default; forbidden patterns (credential-in-URL, any `git clone` / `github.com/kurtvalcorza` /
  repository import on the primary path, a mutable `revision='main'`, direct `from transformers import` /
  `Swin2SRForImageSuperResolution` / `Swin2SRImageProcessor` / `from torchvision import` / `from huggingface_hub
  import` / `urllib.request` / `safetensors` imports / `torch.optim` / `.backward(` / `requires_grad` /
  `pipe._model` / `extractall(` use **outside the carried module cells**, `trust_remote_code=True`, `pickle.load`,
  `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs `pytest`, `ruff`, `numpy` and `pillow`, the package with `--no-deps`, runs `ruff check src tests
tools`, `tools/build_notebook.py --check`, and the offline unit suite (`tests/`, including `test_adaptation.py`,
`test_role_helpers.py`, `test_import_boundary.py`, `test_notebook_parity.py`; injected runner and photo fetcher, no
weights, no `torch` — `tests/test_model_backed.py` is skipped without `torch` or the snapshot). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/swin2sr-x2-64/` or the photograph cache `weights/inat-birds/` (the standalone path writes the
   manifest itself, stages the missing files from the Hub and fetches the pinned photographs from the iNaturalist
   open-data bucket, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `EPOCHS = 5`, `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 8`,
   `TRAINABLE_STAGES = 1`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`,
   `pillow==11.3.0`, `huggingface-hub==0.36.2` (an interpreter restart after the install is expected where the
   runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `Swin2SRPipeline`, `verify_snapshot`, `stage_missing_files`,
     `validate_inputs`, `evaluation_report`, `psnr`, `ssim`, `sr_metrics`, `nearest_baseline`, `bicubic_baseline`,
     `fetch_corpus`, `read_corpus`, `degrade`, `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`,
     `observer_overlap`, `split_dataset`, `load_byod_dataset`, `write_dataset_csv` and the ceilings) with no import
     of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 4 manifest entries fetched from `caidas/swin2SR-classical-sr-x2-64` at the
     immutable revision on a clean runtime, `verify_snapshot` returning its dict (4 files, the 48 MB
     `model.safetensors` re-hashed), and `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified
     directory;
   - Section 4: `fetch_corpus` fetching the 360 pinned photographs with every byte count and SHA-256 matching; 192-px
     references and 96-px JPEG-40 inputs; the seeded split into 216 / 48 / 96 (36 / 8 / 16 per species) with
     `check_split_disjoint` reporting no shared image, the observer overlap and the three dataset digests printed;
     `outputs/…_train.csv` written; the four dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`MIN_INPUT_SIDE` 8, `MAX_INPUT_SIDE` 512, `UPSCALE` 2, `HR_CROP` 192, `MIN_RECORDS` 8,
     `MAX_RECORDS` 5000) surfaced; the synthetic scene built; `validate_inputs` writing `outputs/…_input_manifest.json`
     (verdict `accepted`, one recorded rejection finding from the oversize probe); `upscale` on the 64×48 input with
     every sanity check `True`, `outputs/…_output_frozen.png` written and the per-image `evaluation_report` verdict
     `sample-sanity`;
   - Section 6: the nearest-neighbour floor (≈ 25.3 dB / 0.764), the bicubic baseline (≈ 26.3 dB / 0.797) and the
     frozen model's test score (≈ 25.9 dB / 0.785 in the RTX 5070 Ti build record — below bicubic on this input) with
     the per-species breakdown, and the cell's assertion that the frozen model is above the floor;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 2,463,011 trainable of 12,091,571 parameters, and
     a five-epoch history with the validation PSNR rising every epoch (25.90 → 26.40 → 26.56 → 26.68 → 26.74 →
     26.78 dB in the build record; `best_epoch` 5);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison on both measures, the
     per-species breakdown and `outputs/…_evaluation_report.json` written (the cell asserts the adapted test PSNR
     exceeds the frozen one — 26.91 versus 25.92 dB in the build record, with SSIM 0.785 → 0.816 and every species
     gaining about a decibel; the adapted model also clears bicubic, reported, not asserted);
   - Section 9: the scene re-upscaled by the adapted model with the `sample-sanity` report,
     `outputs/…_output_adapted.png` and four example panels under `outputs/…_examples/` written; `pipe.save_artifact`
     writing `outputs/…_adapter/{adapter.safetensors,manifest.json}` (126 tensors, 9,868,140 bytes) and
     `Swin2SRPipeline.from_artifact` reloading it with 8/8 pixel-identical reconstructions on eight test crops (the
     cell asserts it); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity and licence, the
     snapshot block (`weight_file`, `weight_format`, `weight_sha256`), the `corpus` block with the degradation, the
     inference-contract reports, the comparison, the artifact digest, the reload parity, the runtime versions and
     device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the photograph cache were
   clean, outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or
   applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `swin2sr_super_resolution_colab.ipynb` (`E2E`) | pending | — | — | **PENDING** — a clean-runtime execution of the `E2E` blob has not been recorded yet |
| `swin2sr_super_resolution_colab.ipynb` (`TASK-INFERENCE`, superseded) | `8e25aa7` / `472a93920103` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-swin2sr-super-resolution` v1) | PASS — 8/8 code cells, 267.5 s, 10 files, 48 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/swin2sr_super_resolution_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/swin2sr_super_resolution_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-20 | generated at `ae8eef9` / blob `4af52954ed42` | Local Windows-venv harness (`run_nb_local.py`: nbclient, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6, snapshot and the 360 photographs pre-staged | Default sample path, all 11 code cells: pinned install skipped (pre-installed), `stage_missing_files` reported nothing to fetch, `verify_snapshot` PASS (4 files), 360 photographs re-hashed from the pre-staged cache, paired and split 216 / 48 / 96, four refusal probes raised, the synthetic scene upscaled (37.1 dB against its reference), baselines 25.29 / 26.31 dB, frozen test 25.92 dB / 0.785 in 70.8 s, five epochs 1,314.5 s (validation PSNR 25.90 → 26.40 → 26.57 → 26.68 → 26.74 → 26.79, epoch 5 kept), adapted test 26.91 dB / 0.816 (every species +0.8..1.2 dB; above bicubic on both measures), the scene re-upscaled at 33.5 dB (−3.6 dB on clean input — the adapted tail expects JPEG artefacts), four panels written, adapter 9,868,140 B / 126 tensors, reload parity 8/8 with max abs difference 0, 8 outputs written; the committed blob differs from the executed one in markdown prose and the recorded generating revision only (timing figures filled in after this run) | 1523.2 s | PASS — pre-flight only; not promotion evidence |
| 2026-09-20 | generated at `ae8eef9` / blob `4af52954ed42` | Local WSL harness (same `run_nb_local.py`, `CUDA_VISIBLE_DEVICES=0`), Python 3.12.3, torch 2.14.0+cu130, RTX 5070 Ti (`cuda:0`), snapshot and photographs pre-staged | Default sample path, all 11 code cells; frozen 25.92 / 0.785 in 12.1 s, five epochs 312.9 s, epoch 5 kept, adapted 26.91 / 0.816, scene 37.1 → 33.5 dB, reload parity 8/8 — identical to the CPU row to three decimals | 419.4 s | PASS — pre-flight only; not promotion evidence |

## Current status

**Candidate.** The `E2E` notebook's source passes static validation, the generator parity checks and the offline unit
suite; the model-backed regressions (`tests/test_model_backed.py`, 7 tests) pass on CPU and on the RTX 5070 Ti in
WSL; the local pre-flight above executed the default path top-to-bottom. None of that is REL1/REL10 supported-runtime
evidence: the registry stays **Candidate** until a clean-runtime run of the committed blob — no repository checkout,
an empty Hugging Face cache, no pre-staged snapshot or photographs — is recorded in the tables above and an integrator
promotes it.

Facts a reviewer should still weigh: the frozen classical-SR checkpoint scores *below* bicubic interpolation on
JPEG-40 input (25.9 versus 26.3 dB in the build record) because it sharpens block artefacts as if they were detail,
so the adaptation's gain (to 26.9 dB / 0.816 SSIM, past bicubic on both measures) is a repair of that specific
mismatch and says nothing about other degradations; both measures are reference-based signal fidelity, not
perceptual quality; the 48-crop validation split selects the epoch and the training loss was still falling at epoch
five, so the recipe is bounded by budget rather than convergence; and the synthetic scene re-upscaled after
adaptation — clean bicubic-only input — lost 3.6 dB (37.1 → 33.5) in both pre-flights: the adapted tail now expects
JPEG artefacts and smooths a clean image, which is what "one stated degradation" costs.
