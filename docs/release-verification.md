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
| `swin2sr_super_resolution_colab.ipynb` (`E2E`) | `7ada094` / `abbfe78e` | 2026-10-18 | Colab CLI 0.7.4, fresh Colab Tesla T4 (session `suite-swin2sr-7ada094-1c23`; isolated `uv` CPython 3.12.12, torch 2.14.0+cu130, transformers 4.57.6, `cuda:0`) | **PASSED — reproduction of the pre-registered evaluation (REL13)**: 13/13 code cells, 0 errors; `test_look` "reproduction"; grid kept C0; fresh test selected 25.398 dB / 0.773 vs frozen 24.508 / 0.744, paired +0.89 dB (CI 0.8168–0.967, 144/144), equal to the record; evidence `docs/execution-evidence/2026-10-18-7ada094/`; see *Hosted T4 reproduction 1 (2026-10-18)* |
| `swin2sr_super_resolution_colab.ipynb` (`E2E`) | `e6a7fa2` / `eb13bd63` | 2026-10-18 | Colab CLI 0.7.4, fresh Colab Tesla T4 (session `suite-swin2sr-e6a7fa2-1c23`; isolated `uv` CPython 3.12.12, torch 2.14.0+cu130, transformers 4.57.6, `cuda:0`) | **PASSED — pre-registered evaluation of record** (first hosted run of the pre-registered settings; first scoring of the fresh test): 13/13 code cells, 0 errors; grid kept C0; fresh test selected 25.398 dB / 0.773 vs frozen 24.508 / 0.744, paired +0.89 dB (CI 0.8168–0.967, 144/144); evidence `docs/execution-evidence/2026-10-18-e6a7fa2/`; see *Hosted T4 evaluation of record (2026-10-18)* |
| `swin2sr_super_resolution_colab.ipynb` (`E2E`) | `e703ad1` / `41a0b631` | 2026-09-20 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-swin2sr-super-resolution` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 370 files, 88 MB staged from the Hub into a clean cache; comparison {psnr: {nearest: 25.29, bicubic: 26.31, frozen: 25.92, adapted: 26.91}, ssim: {nearest: 0.764, bicubic: 0.797, frozen: 0.785, adapted: 0.816}, delta_vs_frozen: {psnr: 0.991, ssim: 0.032}, delta_vs_bicubic: {psnr: 0.6, ssim: 0.019}, by_species: {american_goldfinch: {n: 16, frozen_psnr: 26.73, adapted_psnr: 27.87, frozen_ssim: 0.832, adapted_ssim: 0.871}, chipping_sparrow: {n: 16, frozen_psnr: 25.81, adapted_psnr: 26.61, frozen_ssim: 0.764, adapted_ssim: 0.788}, dark_eyed_junco: {n: 16, frozen_psnr: 25.76, adapted_psnr: 26.97, frozen_ssim: 0.766, adapted_ssim: 0.801}, house_finch: {n: 16, frozen_psnr: 26.17, adapted_psnr: 27.35, frozen_ssim: 0.796, adapted_ssim: 0.827}, song_sparrow: {n: 16, frozen_psnr: 25.57, adapted_psnr: 26.36, frozen_ssim: 0.781, adapted_ssim: 0.811}, white_throated_sparrow: {n: 16, frozen_psnr: 25.48, adapted_psnr: 26.31, frozen_ssim: 0.77, adapted_ssim: 0.8}}}; reload parity {identical_images: 8, of: 8, max_abs_difference: 0}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-swin2sr-super-resolution/v2/evidence/` in the workspace |
| `swin2sr_super_resolution_colab.ipynb` (`TASK-INFERENCE`, superseded) | `8e25aa7` / `472a93920103` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-swin2sr-super-resolution` v1) | PASS — 8/8 code cells, 267.5 s, 10 files, 48 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/swin2sr_super_resolution_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/swin2sr_super_resolution_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-29 | `f373b9c5ce905360851deae01cd8a59c584c2ae7` (PR #10 head) / `swin2sr_super_resolution_colab.ipynb` blob `5c9364c48f2639480ea14b7e7e84a66d0b12b483` (current blob) | Google Colab CLI sequential execution (`colab exec -f`), fresh Colab Tesla T4 (session `suite-swin2sr-f373b9c-1c23`): every code cell run in order in one kernel; not a browser `Run all`, and the CLI records no execution counts, so order is evidenced by `exec.log` (cells 1–13 in order); kernel Python 3.13.16, isolated CPython 3.12.12 (47 locked packages, 56 s); the committed blob fetched at the 40-character SHA and Git-blob verified, input unchanged | Default path only, form fields at their defaults, no overrides | 2013.2 s | **One pass, no restart, 0 errors**; 13/13 code cells; VM released (`stop_verified`); the executed notebook's SHA-256 equals `run_summary.json` `executed_sha256`. 4 of 4 snapshot files verified at `cee1c923`, `cuda:0`; fresh test 144 crops disjoint from the 360 pinned sample crops; four refusal probes rejected. Grid kept C0 (`best_epoch` 5; 2,463,011 of 12,091,571 parameters trainable); `test_look` `reproduction`. Exposed development test (n = 96) PSNR nearest 25.289, bicubic 26.314, frozen 25.923, selected 26.914; fresh test (n = 144) PSNR 23.83 / 24.868 / 24.508 / 25.398 (`delta_vs_frozen` 0.89 dB, 95% CI 0.8168 to 0.967, 144 wins, 0 losses; SSIM 0.744 → 0.773), verdicts `selected_vs_frozen_psnr` and `selected_vs_bicubic_psnr` `improved`. Adapter 126 tensors, 9,868,140 bytes; reload parity 8 of 8 identical images, `max_abs_difference` 0; `RUN_EXPERIMENT` off (`skipped`). Evidence `docs/execution-evidence/2026-10-29-f373b9c/`: SHA-256 captured → committed: `swin2sr_super_resolution_colab_f373b9c_colab-cli-t4_default.ipynb` `a7c8f93892a36674c8bee97ff616bf493daa1fda69d7fd8fb65c776a3e3a5e1a` → unchanged (no local path); `swin2sr_super_resolution_colab_f373b9c_colab-cli-t4_default_exec.log` `76097480f719ffa509315776c0ffa7daea0efb77b2ff65e191cae814a27a03c4` → `65976e08daf282ef5729e04a7c5d53577829baad44fb78ffd84c78a58a2eadb6` (2 local-path replacements); `swin2sr_super_resolution_colab_f373b9c_colab-cli-t4_default_run_summary.json` `c39cdabc4397dd6cdcd6aee425e132467f711e761fa36410ced787274f804846` → unchanged (no local path). Not exercised: an accepted BYOD input, optional switches off their defaults, and the form widgets |
| 2026-10-29 | same | same, fresh VM (session `suite-swin2sr-f373b9c-1c23`) | Scripted DAT20 refusal: `USE_BYOD = True`; `BYOD_PATH = '/etc/hostname'` (set by `--set` on a copy; the run copy differs from the blob in exactly those 2 source lines), and the run copy cut right after the refusing cell `swin2sr_super_resolution-16`, dropping 20 cells (6 code cells), which reproduces `Run all` stopping at the first error | 91.5 s | **Refused as designed** — 7/7 code cells of the cut copy. The 6 earlier code cells completed without an error; the refusing cell printed exactly one `Refused:` line, with no traceback: `Refused: hostname: /etc/hostname is neither a directory nor a zip file. Nothing was trained or scored: fix the input and choose Run after from Section 4; until then this run is invalid, and files already in outputs/ are from the earlier run, not from your photographs.`. The suite recorded the halting error as `IsolatedCellStop`. `verdict` `PASS`, input unchanged, VM released. Cut copy `run_copy.ipynb` SHA-256 `feed7f7cc5e2454770f85a91c5bd163f46ddc42fd0cc9643d37530e53a6da833` (Git blob `35a51506`), equal to the summary's `run_copy_sha256`. This run says nothing about the 20 cut cells, which did not run in it. Evidence `docs/execution-evidence/2026-10-29-f373b9c/`: SHA-256 captured → committed: `swin2sr_super_resolution_colab_f373b9c_colab-cli-t4_refusal.ipynb` `2741488ec7ce60328e9723ebd8841dd84e98d50391ac6239786727c97e2067d4` → unchanged (no local path); `swin2sr_super_resolution_colab_f373b9c_colab-cli-t4_refusal_exec.log` `56fa638c36a80e988b2521a5a33b0463359c686832dc6c1db8981f28d5cf1449` → `86f4610505c56f9a065dc9c396799ccc8de6c2d2d2afd988a30ea929b13e7744` (2 local-path replacements); `swin2sr_super_resolution_colab_f373b9c_colab-cli-t4_refusal_run_summary.json` `5f8a478298fdc236cd20b0399e07e8de181ea1b56b840c3205fd98fcb9b95c16` → unchanged (no local path); `swin2sr_super_resolution_colab_f373b9c_colab-cli-t4_refusal_run_copy.ipynb` `feed7f7cc5e2454770f85a91c5bd163f46ddc42fd0cc9643d37530e53a6da833` → unchanged (no local path) |
| 2026-10-18 (07:32:20 UTC start) | `7ada094` / `abbfe78ee763cfe4b688b2c10e12be424e34c851` (`NOTEBOOK_SOURCE.repository_revision` `5a2bad7`; `5a2bad7..7ada094` adds the regeneration and docs only; embedded `module_sha256` `88c39c33f8bd…`, generator `build_notebook.py/2.3`, `notebook_spec` 2.5) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4 VM (session `suite-swin2sr-7ada094-1c23`), committed blob fetched at the 40-char SHA and Git-blob verified (`source.ipynb` byte-equal to the committed notebook), no repository checkout; kernel Python 3.13.15; isolated `uv` environment CPython 3.12.12, 47 locked packages, built in 51 s; torch 2.14.0+cu130, transformers 4.57.6; device `cuda:0` | Default path, every form field at its default (`RUN_GRID = 'auto'` → grid on, BYOD off, `EPOCHS = 5`, `BATCH_SIZE = 8`, `RUN_EXPERIMENT = False`); same snapshot, splits and fresh test (digest `57a6730d…`); forms and uploads not exercised | 1952.9 s (session wall) | **PASSED — reproduction of the pre-registered evaluation (REL13)**: 13/13 code cells in order (`exec.log`), 0 error outputs; `test_look` "reproduction", fresh-test gate open, outputs suffix `_reproduction_1`; grid C0 / C1 / C2 completed, C0 kept; every grid, exposed and fresh-test figure equal to the evaluation of record; reload parity 8/8, max difference 0. Evidence: `docs/execution-evidence/2026-10-18-7ada094/` (executed notebook SHA-256 `f677036ccf02848e69933968108d75e9e21ac3e06fe8259fb1e25ea453e4fa86`, `run_summary.json` `b9b0e2512f2b857bb824f2ac29e460fd4a53aa0ae02d200dabce526a6dce780a`, `exec.log` redacted, see the reproduction section). Not exercised: REL12 BYOD, Section 10 experiment |
| 2026-10-18 (01:12:15 UTC start) | `e6a7fa2` / `eb13bd63145944fadbfff38642186197c40ff943` (`NOTEBOOK_SOURCE.repository_revision` `8041569`, equal to `metadata.dimer.generated_from.revision`; `8041569..e6a7fa2` adds tests, validator markers and the regeneration only; embedded `module_sha256` `7ebac86fb03f…`, generator `build_notebook.py/2.3`, `notebook_spec` 2.5) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4 VM (session `suite-swin2sr-e6a7fa2-1c23`), committed blob fetched at the 40-char SHA and Git-blob verified (`source.ipynb` byte-equal to the committed notebook), no repository checkout; kernel Python 3.13.15; isolated `uv` environment CPython 3.12.12, 47 locked packages, built in 48 s; torch 2.14.0+cu130, transformers 4.57.6; device `cuda:0` | Default path, every form field at its default (`RUN_GRID = 'auto'` → grid on, BYOD off, `EPOCHS = 5`, `BATCH_SIZE = 8`, `RUN_EXPERIMENT = False`); snapshot 4 files digest-verified; 360 photographs split 216 / 48 / 96; fresh test 144 photographs, digest `57a6730d…`; forms and uploads not exercised | 1615.7 s (session wall) | **PASSED — pre-registered evaluation of record (REL13)**: 13/13 code cells in order (`exec.log`), 0 error outputs; code cells 3–5 print nothing; `test_look` "pre-registered evaluation (this runtime)", fresh-test gate open, no output suffix; grid C0 / C1 / C2 completed, C0 kept, frozen not kept; fresh test selected vs frozen +0.89 dB (CI 0.8168–0.967), vs bicubic +0.5302 dB (CI 0.4614–0.6042), both *improved*; reload parity 8/8, max difference 0. Evidence: `docs/execution-evidence/2026-10-18-e6a7fa2/` (executed notebook SHA-256 `c6512ab1b24869e92a6b80b5fd3c225eaea9044db31b35b600e9573a9b8db121`, `run_summary.json` `cba22f1ed90ecc51f7052add41786837184e4095a69b5765161d5d1b42f0549a`, `exec.log` redacted, see the record section). Not exercised: REL12 BYOD, Section 10 experiment |
| 2026-09-20 | `e703ad1` / `41a0b631` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-swin2sr-super-resolution` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 596.8 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 370 files, 88 MB staged from the Hub into a clean cache; comparison {psnr: {nearest: 25.29, bicubic: 26.31, frozen: 25.92, adapted: 26.91}, ssim: {nearest: 0.764, bicubic: 0.797, frozen: 0.785, adapted: 0.816}, delta_vs_frozen: {psnr: 0.991, ssim: 0.032}, delta_vs_bicubic: {psnr: 0.6, ssim: 0.019}, by_species: {american_goldfinch: {n: 16, frozen_psnr: 26.73, adapted_psnr: 27.87, frozen_ssim: 0.832, adapted_ssim: 0.871}, chipping_sparrow: {n: 16, frozen_psnr: 25.81, adapted_psnr: 26.61, frozen_ssim: 0.764, adapted_ssim: 0.788}, dark_eyed_junco: {n: 16, frozen_psnr: 25.76, adapted_psnr: 26.97, frozen_ssim: 0.766, adapted_ssim: 0.801}, house_finch: {n: 16, frozen_psnr: 26.17, adapted_psnr: 27.35, frozen_ssim: 0.796, adapted_ssim: 0.827}, song_sparrow: {n: 16, frozen_psnr: 25.57, adapted_psnr: 26.36, frozen_ssim: 0.781, adapted_ssim: 0.811}, white_throated_sparrow: {n: 16, frozen_psnr: 25.48, adapted_psnr: 26.31, frozen_ssim: 0.77, adapted_ssim: 0.8}}}; reload parity {identical_images: 8, of: 8, max_abs_difference: 0}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-swin2sr-super-resolution/v2/evidence/` in the workspace |
| 2026-09-20 | generated at `ae8eef9` / blob `4af52954ed42` | Local Windows-venv harness (`run_nb_local.py`: nbclient, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6, snapshot and the 360 photographs pre-staged | Default sample path, all 11 code cells: pinned install skipped (pre-installed), `stage_missing_files` reported nothing to fetch, `verify_snapshot` PASS (4 files), 360 photographs re-hashed from the pre-staged cache, paired and split 216 / 48 / 96, four refusal probes raised, the synthetic scene upscaled (37.1 dB against its reference), baselines 25.29 / 26.31 dB, frozen test 25.92 dB / 0.785 in 70.8 s, five epochs 1,314.5 s (validation PSNR 25.90 → 26.40 → 26.57 → 26.68 → 26.74 → 26.79, epoch 5 kept), adapted test 26.91 dB / 0.816 (every species +0.8..1.2 dB; above bicubic on both measures), the scene re-upscaled at 33.5 dB (−3.6 dB on clean input — the adapted tail expects JPEG artefacts), four panels written, adapter 9,868,140 B / 126 tensors, reload parity 8/8 with max abs difference 0, 8 outputs written; the committed blob differs from the executed one in markdown prose and the recorded generating revision only (timing figures filled in after this run) | 1523.2 s | PASS — pre-flight only; not promotion evidence |
| 2026-09-20 | generated at `ae8eef9` / blob `4af52954ed42` | Local WSL harness (same `run_nb_local.py`, `CUDA_VISIBLE_DEVICES=0`), Python 3.12.3, torch 2.14.0+cu130, RTX 5070 Ti (`cuda:0`), snapshot and photographs pre-staged | Default sample path, all 11 code cells; frozen 25.92 / 0.785 in 12.1 s, five epochs 312.9 s, epoch 5 kept, adapted 26.91 / 0.816, scene 37.1 → 33.5 dB, reload parity 8/8 — identical to the CPU row to three decimals | 419.4 s | PASS — pre-flight only; not promotion evidence |
| 2026-10-09 (23:22:11 UTC start) | `2be467b` / `c3ddc101ec53` (`NOTEBOOK_SOURCE.repository_revision` `a1e659a`, equal to `metadata.dimer.generated_from.revision`; `a1e659a..2be467b` adds the worker's google.colab ModuleSpec stubs, the honest `restore_base` count and the spec 2.2 declaration; embedded `module_sha256` `46e67bf53dc5…`, generator `build_notebook.py/2.2`, `notebook_spec` 2.2) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4 VM (session `suite-swin2sr-2be467b-1c23`), committed blob fetched at the 40-char SHA and Git-blob verified, no repository checkout; kernel Python 3.13.15; isolated `uv` environment CPython 3.12.12, 47 locked packages, built in 55 s; torch 2.14.0+cu130, transformers 4.57.6; device `cuda:0` | Default sample path, every form field at its default (BYOD off, `EPOCHS = 5`, `RUN_EXPERIMENT = False`); snapshot 4 files digest-verified; 360 photographs fetched and digest-verified, split 216 / 48 / 96 | 913.5 s (session wall) | **PASSED — one pass, no restart, 0 errors**: 13/13 code cells in order (`exec.log`); code cells 3–5 (the carried modules) print nothing; comparison {psnr: {nearest: 25.289, bicubic: 26.314, frozen: 25.923, adapted: 26.914}, ssim: {nearest: 0.764, bicubic: 0.797, frozen: 0.785, adapted: 0.816}}; verdicts frozen above nearest / below bicubic, adapted improved and above bicubic; validation PSNR 25.899 (epoch 0, frozen) → 26.40 → 26.569 → 26.684 → 26.745 → 26.784, epoch 5 kept, `started_from` pinned base; every species gained; clean scene 37.1 → 33.47 dB; reload parity 8/8 identical, max difference 0. Evidence: `docs/execution-evidence/2026-10-09-2be467b/` (executed notebook SHA-256 `73e0f49adf875dac8146bfe9512b6ff2b6cde2ab159ddc25645a6a240c8ba6d1`, `run_summary.json`, `exec.log`). Not exercised: REL12 BYOD, Section 10 experiment |

## DAT20 refusal display (2026-10-29, row 88)

- **Change:** code commit `4d8662e`; the notebook is regenerated in the next commit (blob `5c9364c4`). A refused BYOD input in Section 4 (a missing path, a path that is neither a folder nor a zip, a cancelled, empty or multi-file upload, a non-zip upload, a corrupt zip, no upload dialog, and every rule `load_byod_dataset` and `split_dataset` enforce) stops with one line, `Refused: <file>: <reason>. Nothing was trained or scored: …`, through `RunStopped` (`dimer_stop`), which the router shows without a traceback; `RunStopped._render_traceback_` gives the same line with routing disabled. The run stays invalid (`RUN_VALID = False`) as before. A FRESH_TEST_DIGEST mismatch of the pinned fresh test is a genuine pinned-asset failure: it now raises `RuntimeError` with its traceback instead of `RunStopped`.
- **Status:** blob `5c9364c4` is **Candidate** (REL14). Its default path (a reproduction under the pre-registered settings) and its scripted refusal (`USE_BYOD=True;BYOD_PATH='/etc/hostname'`, a file that is neither a folder nor a zip) passed on a hosted Colab Tesla T4 on 2026-10-29: 13/13 code cells in one pass with no restart and 0 errors, and exactly one `Refused:` line at cell `swin2sr_super_resolution-16` with no traceback (Colab CLI sequential execution, not a browser `Run all`; see "Recorded executions"). The evaluation of record and the reproduction by `7ada094` / `abbfe78e` below stay as history; they are not evidence for `5c9364c4`.

## Fix cycle 2026-10-18 (row 88 review: R88-M1..M4, m1..m4, s1)

- **Pre-registration:** `docs/swin2sr-preregistration.md`, committed alone (`ad1498f`) after the reduced CPU smoke below (whose validation numbers were in view when the 0.10 dB margin was set) and before the full grid or any fresh-test score. Amendment 1 (`docs/swin2sr-preregistration-amendment-1.md`, committed alone as `de077f5`) adds the fresh-test reading rule (paired species-stratified bootstrap, 'improved' only if the 95 % CI lower bound > 0 and the gain >= 0.10 dB) and the incomplete-grid rule. Frames for replaying the seeded draw: `docs/fresh-test-frames.json.gz`. Whether C2 fits a T4 at batch 8 is unproven; the hosted run decides and logs each config's peak CUDA memory. Fresh test: 144 photographs, `FRESH_TEST_DIGEST` `57a6730d6a3b…`, draw record `docs/fresh-test-draw.json`.
- **Development looks (SCORE_ONCE_RULE item 8):** none. The fixer's CPU checks ran with the fresh test masked (not loaded); no fresh-test number exists before the hosted run.
- **Reduced CPU smoke (2026-10-18, main `136092f` package, build workstation, 48 training / 16 validation pairs, 2 epochs, batch 8; validation only):** C0 139.5 s (val PSNR 26.574 → 26.943), C1 138.3 s (→ 27.059), C2 256.8 s (→ 27.042); peak private memory C0 8.5 GiB, C2 15.8 GiB. Three earlier C2 attempts ended in an access violation while another process competed for memory; run alone, C2 completed at 1 and at 4 threads. These are timing and memory figures for a reduced split, not selection evidence: the grid is decided on the hosted run.
- **Evaluation of record:** the 2026-10-18 Colab T4 run of `e6a7fa2` / `eb13bd63` (section *Hosted T4 evaluation of record (2026-10-18)* below). `RECORDED_SELECTION` / `RECORDED_EVALUATION` in `samples.py` carry it from commit `5a2bad7` (notebook regenerated in `9a649d2`, blob `abbfe78e`); with them set, a later run under the pre-registered settings is labelled "reproduction" and writes `_reproduction_<n>` files. Blob `abbfe78e` differs from the evaluated `eb13bd63` in prose, the two records, the selection `reason` text with `margins_db`, and the numbering of `_reproduction_<n>` when a record exists; it was executed on a hosted runtime on 2026-10-18 as reproduction 1 (below).
- **Reproductions:** 1 — the 2026-10-18 Colab T4 run of `7ada094` / `abbfe78e` (section *Hosted T4 reproduction 1 (2026-10-18)* below): every grid and fresh-test figure equal to the record. Blob `abbfe78e` is therefore hosted-executed.
- **Worked answers** (requoted 2026-10-18 in `5a2bad7`, generator only) quote the 2026-10-18 evaluation of record, re-derived from its executed notebook: timings (1615.7 s session, Section 7 1181.7 s, C0 369.6 s), the grid answer (C2 fails the clause against C1 by < 0.10 dB unrounded), the C2 fit (11.94 GiB vs 6.32 GiB), the fresh-test answer (cell 29) and the conclusion (cell 36). Two C0 curve values the earlier answer gave as 26.56 and 26.74 are 26.569 and 26.745 in both hosted runs; the answer now quotes three decimals. The selection `reason` now names each rule clause with a margin printed precisely enough not to round across 0.10 dB, and `margins_db` keeps the margins in the grid file.

- **Evidence redaction (R88-s1, 2026-10-18):** `docs/execution-evidence/2026-10-09-2be467b/exec.log` carried the executor's local run directory (`/home/<user>/colab-runs/...`, lines 1 and 71). That prefix was replaced with `<run-dir>/` in a separate commit; nothing else changed. SHA-256 before `b46e0db9b10463e913e0562184fc9b59cbd7b5f17bf9cadcb704a2c7b46bf3e8`, after `07dd1e16df4b3a3c0e6c7943c8b18211d3c0a658682d2586b3b609d830e93daa`. The executed notebook and `run_summary.json` beside it are untouched.

## Hosted T4 evaluation of record (2026-10-18)

**REL13 label: pre-registered evaluation of record** — the first hosted run of the pre-registered settings (`docs/swin2sr-preregistration.md` + amendment 1) and the first scoring of the 144-photograph fresh test by any run. The fixer's CPU checks ran with the fresh test masked (no development look). Later identical hosted runs are *reproductions*. Executor: Colab CLI 0.7.4, fresh Colab Tesla T4; evidence boundary: CLI sequential execution, not a browser `Run all`; forms and uploads not exercised. Status stays **Candidate**.

- **Identity.** PR head `e6a7fa2aa4ca3f28fbc0409332e66686bb15ad35`, notebook blob `eb13bd63145944fadbfff38642186197c40ff943` (verified before execution; `source.ipynb` byte-equal to the committed file). `inspect_run.py`: 37 / 37 cells, same ids and order, sources identical (0 parameter toggles; the executed file stores sources as line lists). Embedded revision `8041569`, `module_sha256` `7ebac86fb03fa039…`, generator 2.3, spec 2.5.
- **Pre-registration check.** The notebook printed `fresh_test_gate: open`, `test_look: pre-registered evaluation (this runtime)`, `outputs_suffix: ''`; Section 7 printed no `EXPLORATORY` line, so no setting differed. The grid completed all three configs in one Section 7 pass.
- **Evidence files** (`docs/execution-evidence/2026-10-18-e6a7fa2/`, `-text`): executed notebook `c6512ab1b24869e92a6b80b5fd3c225eaea9044db31b35b600e9573a9b8db121` and `run_summary.json` `cba22f1ed90ecc51f7052add41786837184e4095a69b5765161d5d1b42f0549a`, byte-equal to the run directory. `exec.log`: captured `82c0197ea838f01b720b2f9348d8da12c4874ea9c24e4d9392f4e1d35a1d10ff`, committed `5fe1efec6467254b6c387a7d83ac054cbfcdf9b5bf047916de698eb0b6646e08` after the R88-s1 convention (`229469c`): the executor's local run-directory prefix replaced with `<run-dir>/` on lines 1 and 108, nothing else changed.

### Selection grid (validation, 48 pairs)

| Config | Stages | LR | Validation PSNR, epochs 0→5 (dB) | Kept epoch | Score | Seconds | Peak CUDA memory (`max_memory_allocated`) |
|---|---|---|---|---|---|---|---|
| C0 | 1 | 1e-4 | 25.899 → 26.400 → 26.569 → 26.684 → 26.745 → 26.784 | 5 | 26.784 | 369.6 | 6,783,353,856 B (6.32 GiB) |
| C1 | 1 | 2e-4 | 25.899 → 26.481 → 26.686 → 26.755 → 26.798 → 26.833 | 5 | 26.833 | 372.1 | 6,783,229,440 B (6.32 GiB) |
| C2 | 2 | 1e-4 | 25.899 → 26.499 → 26.737 → 26.820 → 26.873 → 26.933 | 5 | 26.933 | 439.7 | 12,817,431,040 B (11.94 GiB) |

- **C2 fitted the T4 at batch 8**: 11.94 GiB peak, 1.89× C0; no out-of-memory error. C2 took 1.19× C0's time (the reduced CPU smoke predicted 1.84×).
- **Selection re-applied:** C1 − C0 = +0.049 dB (< 0.10, C1 not selected). C2 − C0 = +0.149 dB, but C2 − max(C0, C1) = 26.933 − 26.833 = 0.100 dB at printed precision; `select_config` compares unrounded scores and found it below 0.10 dB, so rule 3 failed and **C0 was kept**. C2 missed the threshold by less than 0.001 dB; the unrounded scores were in the grid file on the released VM and are not in the evidence. The printed reason ("no challenger beats C0 (26.784 dB) by at least 0.1 dB") is inaccurate for C2, which beat C0 by 0.149 dB and failed the second clause (wording finding; code not changed here).
- **Keep-frozen rule:** C0 26.784 − frozen 25.899 = +0.885 dB ≥ 0.10, so the adapted C0 is the selected system. Every config was still improving at epoch 5.
- Section 7 total 1181.7 s; the C0 curve and training losses equal the 2026-10-09 run at every epoch.

### Fresh test (144 photographs, 24 per species) — the claim

| System | PSNR (dB) | SSIM |
|---|---|---|
| Nearest | 23.830 | 0.715 |
| Bicubic | 24.868 | 0.753 |
| Frozen | 24.508 | 0.744 |
| Selected (C0) | 25.398 | 0.773 |

| Paired reading (amendment 1; species-stratified bootstrap, 10,000 resamples, seed 20261018) | Mean PSNR difference | 95 % CI | Wins / losses | SSIM difference (95 % CI) | Verdict |
|---|---|---|---|---|---|
| Selected − frozen | +0.890 dB | 0.8168 to 0.967 | 144 / 0 | +0.02915 (0.02543 to 0.03303) | improved |
| Selected − bicubic | +0.5302 dB | 0.4614 to 0.6042 | 142 / 2 | +0.02013 (0.01731 to 0.02313) | improved |

Per species (frozen → selected PSNR / SSIM): American goldfinch 26.00 → 26.94 / 0.771 → 0.801; chipping sparrow 24.47 → 25.30 / 0.749 → 0.774; dark-eyed junco 24.67 → 25.64 / 0.718 → 0.749; house finch 24.64 → 25.66 / 0.763 → 0.800; song sparrow 24.69 → 25.59 / 0.772 → 0.802; white-throated sparrow 22.57 → 23.27 / 0.690 → 0.711. Every species gained (+0.70 to +1.02 dB). The frozen model is below bicubic on the fresh test too (−0.36 dB), as on the exposed split.

### Exposed development test (96 crops; shown for continuity, decides nothing)

Nearest 25.289 / 0.764, bicubic 26.314 / 0.797, frozen 25.923 / 0.785, selected 26.914 / 0.816. Selected − frozen +0.9907 dB (CI 0.8824 to 1.105, 96 / 0), SSIM +0.03151 (0.02716 to 0.03579); selected − bicubic +0.5998 dB (CI 0.5109 to 0.6916, 95 / 1), SSIM +0.01887 (0.01591 to 0.02189). Per-species figures equal the 2026-10-09 run. The fresh-test gains are 0.10 dB (vs frozen) and 0.07 dB (vs bicubic) smaller than on this exposed split.

### Re-derivation (by the recorder, from the printed output)

Printed deltas equal the differences of printed means (fresh 25.398 − 24.508 = 0.890, 25.398 − 24.868 = 0.530; exposed 0.991, 0.600); the mean of the six equal-n species means reproduces each overall mean within rounding (fresh 24.507 / 25.400, exposed 25.920 / 26.912); both fresh verdicts satisfy the rule (CI lower bound > 0 and mean gain ≥ 0.10 dB). The bootstrap itself was not re-run: per-crop scores stayed on the released VM.

### Comparison with earlier figures

| Item | Earlier | This record |
|---|---|---|
| Reduced CPU smoke (2026-10-18; 48 / 16 pairs, 2 epochs; **not comparable**: different split sizes and budget) | C0 26.943, C1 27.059, C2 27.042 (C1 > C2 > C0) | C0 26.784, C1 26.833, C2 26.933 (C2 > C1 > C0) |
| C2 memory | 15.8 GiB CPU process peak (not CUDA memory) | 11.94 GiB CUDA peak |
| 2026-10-09 Colab T4 (`2be467b` / `c3ddc101`, C0 only) | exposed 25.289 / 26.314 / 25.923 / 26.914; C0 curve 25.899 … 26.784; scene 37.1 → 33.47; parity 8/8; adapter 9,868,140 B | equal at printed precision |
| C0 seconds / frozen-test seconds / environment build | 316.7 / 24.8 / 55 | 369.6 / 22.6 / 48 |
| Session wall | 913.5 s (no grid, no fresh test); estimate ≈ 32 min with the grid | 1615.7 s (26.9 min) |
| C1 / C2 seconds (estimate) | ≈ 317 / ≈ 580 | 372.1 / 439.7 |

## Hosted T4 reproduction 1 (2026-10-18)

**REL13 label: reproduction of the pre-registered evaluation** — a later hosted run with the identical pre-registered settings, of blob `abbfe78e`, which carries the evaluation of record in `RECORDED_SELECTION` / `RECORDED_EVALUATION`. Nothing was changed in response to the test number: `e6a7fa2..7ada094` changes prose, the two records, the selection `reason` text with `margins_db`, the `_reproduction_<n>` numbering and dated notes on the pre-registration documents. Executor: Colab CLI 0.7.4, fresh Colab Tesla T4; evidence boundary: CLI sequential execution, not a browser `Run all`; forms and uploads not exercised. Status stays **Candidate**.

- **Identity.** PR head `7ada0946c708d11a8c3a3bf16661d321a3e01fc7`, notebook blob `abbfe78ee763cfe4b688b2c10e12be424e34c851` (verified before execution; `source.ipynb` byte-equal to the committed file). `inspect_run.py`: 37 / 37 cells, same ids and order, source identity OK (0 parameter toggles). Embedded revision `5a2bad7`, `module_sha256` `88c39c33f8bdbc59…`, generator 2.3, spec 2.5.
- **Pre-registration check.** The notebook printed `fresh_test_gate: open`, `test_look: reproduction`, `outputs_suffix: _reproduction_1`, and cell 25 printed the evaluation of record it reproduces; no `EXPLORATORY` line, so no setting differed. The grid completed all three configs.
- **Evidence files** (`docs/execution-evidence/2026-10-18-7ada094/`, `-text`): executed notebook `f677036ccf02848e69933968108d75e9e21ac3e06fe8259fb1e25ea453e4fa86` and `run_summary.json` `b9b0e2512f2b857bb824f2ac29e460fd4a53aa0ae02d200dabce526a6dce780a`, byte-equal to the run directory. `exec.log`: captured `152a6cba87dc889b42ab0ada0a5ec308bd4274883e27ac79e44b2ec0ee8b4783`, committed `ac2271ecf2bfd040c60082da22ff89fb22f1dcace3c85eb9ecbfe7da05dec0fd`: the executor's local run-directory prefix replaced with `<local-run-dir>/` on lines 1 and 108, nothing else changed.
- **Match with the record.** Every printed result line equals the 2026-10-18 evaluation of record (diff of the two `exec.log` files with the commit ids normalised): fresh test nearest 23.830 / 0.715, bicubic 24.868 / 0.753, frozen 24.508 / 0.744, selected 25.398 / 0.773; selected − frozen +0.890 dB (CI 0.8168 to 0.967, 144 / 0), SSIM +0.02915 (0.02543 to 0.03303); selected − bicubic +0.5302 dB (CI 0.4614 to 0.6042, 142 / 2), SSIM +0.02013 (0.01731 to 0.02313); all six fresh per-species pairs; the exposed split and its readings; grid curves and training losses for C0, C1 and C2; peak CUDA memory per config; scene 37.1 → 33.47 dB; adapter 126 tensors, 9,868,140 B, SHA-256 `e96f1c2b8d470716…`; reload parity 8/8.
- **Selection.** C0 kept again. The new `margins_db` gives the unrounded margins the record could not show: C1 − C0 = 0.049240 dB, C2 − C0 = 0.148584 dB, C2 − C1 = 0.099344 dB, so C2 missed the 0.10 dB clause against C1 by 0.00066 dB. The new reason reads: C1 +0.049 over C0 (< 0.10); C2 +0.149 over C0 (>= 0.10) and +0.099 over the better one-stage config C1 (< 0.10), "so the more-stages rule fails". This names the failed clause correctly. The grid table still prints the scores at three decimals (26.933 − 26.833 = 0.100), so a reader comparing the table with the reason sees 0.100 vs 0.099; the reason and `margins_db` are the authority.
- **Differences (none in a result):** session wall 1952.9 vs 1615.7 s; Section 7 1051.8 vs 1181.7 s (C0 331.4 / C1 330.3 / C2 389.8 vs 369.6 / 372.1 / 439.7 s); environment build 51 vs 48 s; frozen exposed test 22.9 vs 22.6 s; scene timings; worker pid; embedded revision and `module_sha256`; `evaluation_of_record` printed (was `None`); `test_look`, output-file suffixes and the selection `reason` / `margins_db` (the intended changes); order of one interleaved Pillow warning.

## Current status

**Candidate.** The first hosted pre-registered run, of `e6a7fa2` / `eb13bd63`, is the evaluation of record (fresh Colab Tesla T4, 2026-10-18, Colab CLI 0.7.4 sequential execution, 13/13 code cells, 0 errors, 1615.7 s session wall): the grid kept C0 (validation PSNR C0 26.784, C1 26.833, C2 26.933 dB; C2 fitted at 11.94 GiB peak CUDA memory, no out-of-memory error), and on the 144-photograph fresh test the selected model scored 25.398 dB / 0.773 against frozen 24.508 / 0.744 and bicubic 24.868 / 0.753 (paired gain over frozen +0.89 dB, 95 % CI 0.817 to 0.967, 144 of 144 crops; over bicubic +0.53 dB, CI 0.461 to 0.604, 142 of 144; both *improved* under amendment 1). A reproduction of the pre-registered evaluation (REL13), the 2026-10-18 Colab T4 run of `7ada094` / `abbfe78e` (13/13 code cells, 0 errors, 1952.9 s session wall), printed every grid and fresh-test figure equal to the record and kept C0 (unrounded margin of C2 over C1: 0.0993 dB). The 2026-09-20 Kaggle T4 run of blob `41a0b631` (commit `e703ad1`) needed a manual restart after the install cell (11/11 ok (1 restart after install cell)), so it is not a one-pass `Run all` and not promotion evidence (review finding S2R-M1, 2026-10-05). The review-fix blob `c3ddc101ec53` (commit `2be467b`; uv isolated environment, guided layer, S2R-M1..M4 / S2R-m1 fixes, worker google.colab ModuleSpec stubs) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-09 (Colab CLI 0.7.4 sequential execution, 13/13 code cells, 913.5 s session wall; nearest 25.29 / 0.764, bicubic 26.31 / 0.797, frozen 25.92 / 0.785, adapted 26.91 / 0.816, epoch 5 kept; clean scene 37.1 → 33.5 dB; reload parity 8/8, max difference 0). REL12 BYOD exercise (path-based and upload) and the optional Section 10 experiment are pending on a hosted runtime, so the status stays Candidate (REL14). The 2026-09-20 rows above remain history. Any later change to the carried modules or to the notebook produces a new blob that needs its own clean run.

Facts a reviewer should still weigh: the frozen classical-SR checkpoint scores *below* bicubic interpolation on
JPEG-40 input (25.9 versus 26.3 dB in the build record) because it sharpens block artefacts as if they were detail,
so the adaptation's gain (to 26.9 dB / 0.816 SSIM, past bicubic on both measures) is a repair of that specific
mismatch and says nothing about other degradations; both measures are reference-based signal fidelity, not
perceptual quality; the 48-crop validation split selects the epoch and the training loss was still falling at epoch
five, so the recipe is bounded by budget rather than convergence; and the synthetic scene re-upscaled after
adaptation — clean bicubic-only input — lost 3.6 dB (37.1 → 33.5) in both pre-flights and on the T4: the adapted tail now expects
JPEG artefacts and smooths a clean image, which is what "one stated degradation" costs. These figures are on the 96-crop split that is now the exposed development test: the old default was accepted on it, so it cannot confirm that default.

**Configuration and test sets (row 88 review, 2026-10-18).** The earlier default (one trainable stage at 1e-4) was accepted because it beat bicubic on the 96-crop test split, and no other configuration had completed. The configuration is now chosen by a validation grid pre-registered in `docs/swin2sr-preregistration.md` before any run (C0 one stage at 1e-4, C1 one stage at 2e-4, C2 two stages at 1e-4; 0.10 dB tolerance on best-epoch validation PSNR; the frozen model kept unless beaten by 0.10 dB). The grid runs by default on a GPU; on CPU it is off and the run is exploratory. The 96-crop split is now the **exposed development test**, and a **fresh test** of 144 iNaturalist photographs (24 per species, CC0, one per observer, observers disjoint from the sample and the fleet's other pins; `FRESH_TEST_RECORDS`, `docs/fresh-test-draw.json`) is scored only under the pre-registered settings. The first hosted run with those settings, on 2026-10-18, is the evaluation of record (see *Hosted T4 evaluation of record (2026-10-18)*). Upstream trained the checkpoint on DIV2K and Flickr2K (paper); an exact overlap with the iNaturalist photographs is not checked and cannot be ruled out, though it is unlikely.
