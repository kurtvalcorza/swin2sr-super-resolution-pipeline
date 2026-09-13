# Release verification

`tutorials/swin2sr_super_resolution_colab.ipynb` (`TASK-INFERENCE`) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests,
JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 1.1. This file is
the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec
  `1.1`, `standalone: true` and `generated_from` (repository, generating revision, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the primary
  path; exactly one cell tagged `embedded_module` equal to `src/swin2sr_super_resolution_pipeline/pipeline.py`
  after the generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and
  the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical to
  `tools/build_notebook.py` output; the pinned-install cell with its restart-on-stale-import guard;
  `NOTEBOOK_SOURCE` recorded in the exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is
  a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `Swin2SRPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `upscale(lr_image)`,
  `evaluation_report`), the ceiling print
  (`UPSCALE`, `MIN_INPUT_SIDE`, `MAX_INPUT_SIDE`), the four exports, the learner-facing statements (self-made
  reference, bicubic baseline, ceiling rationale, the `sample-sanity`/`not-measurable` verdicts, no benchmark
  claim) and the gated-off BYOD default listed in the validator; forbidden patterns (credential-in-URL, any
  `git clone` / `github.com` / repository import on the primary path, a mutable `revision='main'`, direct
  `from transformers import` / `Swin2SRForImageSuperResolution` / `Swin2SRImageProcessor` /
  `from huggingface_hub import` use **outside the carried module cell**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also installs the pinned CPU-only torch wheels plus `transformers`, `safetensors`, `numpy` and `pillow`, runs
`ruff`, `tools/build_notebook.py --check`, and the offline unit suite (`tests/test_pipeline.py`,
`tests/test_role_helpers.py`, `tests/test_notebook_parity.py`; injected runner, no weights). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebook is standalone) |
| Local Windows-venv harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines the pipeline class and both role helpers) with no import of the
     repository package;
   - synthetic 128×96 reference generated in code, bicubic-downscaled to a 64×48 input, both RGB SHA-256 digests printed, and the ceilings (`UPSCALE` 2, `MIN_INPUT_SIDE` 8, `MAX_INPUT_SIDE` 512) surfaced;
   - pinned `caidas/swin2SR-classical-sr-x2-64` acquisition at the immutable revision through the package:
     the inline `MANIFEST` is asserted against the module identity and written to `weights/swin2sr-x2-64/`,
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports all four manifest entries
     (`README.md`, `config.json`, `model.safetensors`, `preprocessor_config.json`) on a clean runtime,
     `verify_snapshot` returns its dict, and `from_pretrained(weights_dir=WEIGHTS_DIR)`
     loads from the verified directory;
   - `validate_inputs` writes `outputs/swin2sr_super_resolution_input_manifest.json` (verdict `accepted`, the
     64×48 input with its 128×96 output size, one recorded rejection finding from the oversized probe);
   - `upscale(lr_image)` returning `output_size == (128, 96)` and an array of shape `(96, 128, 3)` uint8;
   - `evaluation_report` writes `outputs/swin2sr_super_resolution_evaluation_report.json` with verdict
     `sample-sanity`, one `psnr` metric and one bicubic-2× baseline against the self-made reference, labelled
     sanity evidence rather than a benchmark;
   - `outputs/swin2sr_super_resolution_result.json` and `outputs/swin2sr_super_resolution_output.png` written with
     `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/swin2sr_super_resolution_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/swin2sr_super_resolution_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| | | | Default sample path | | pending — queued to the GPU lane |

## Current status

No clean-runtime execution of the notebook has been recorded yet; the run is **pending** and queued
to the GPU lane. Static validation (`tools/validate_release_assets.py`), nbformat validation, a
`compile()` sweep over every code cell, and the offline unit suite passed on the tutorial source at
the candidate revision, which is necessary but not sufficient. The registry status remains
**Candidate** until a reviewer confirms a recorded run against the notebook blob under review and
an integrator promotes it; promotion is not performed by the builder. Facts a reviewer should weigh:
`stage_missing_files` was exercised only with an injected downloader in the unit suite (the real
`hf_hub_download` fetch into the snapshot directory has not been executed), and the card pass executed `upscale` only on CPU in the Windows venv (returns/L2: CUDA path not executed), so the clean run will be the first real execution of the staging path; the CPU inference path against real weights was executed once locally (card Runtime: 0.36 s for 64×48).

A further fact a reviewer should weigh from the standalone pass: the carrier itself — executing the
carried module cell in a runtime that has no repository checkout — has been validated statically only
(parity PASS plus an offline carrier probe that stopped before any fetch), never run.
