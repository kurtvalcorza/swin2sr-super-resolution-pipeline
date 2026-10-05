"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (three modules,
carried verbatim in dependency order), and the model pin/stage/verify cells are produced by the generator from
repository sources so they cannot drift from the package.

This template configures an E2E super-resolution fine-tuning workflow: the pinned
caidas/swin2SR-classical-sr-x2-64 snapshot is digest-verified and loaded, 360 CC0 iNaturalist photographs are fetched
with per-file digests and turned into paired crops (a 192-px high-resolution crop and its bicubic-downsampled,
JPEG-compressed 96-px input), the pairs are validated and split by photograph, a synthetic scene is upscaled through
the inference contract, the frozen model's PSNR and SSIM over the held-out crops are measured beside the bicubic and
nearest-neighbour baselines, a bounded fine-tuning of the last encoder stage and the reconstruction tail runs with
the L1 loss, the held-out split is scored again per species, the scene and four held-out crops are re-upscaled with
the adapted model, and the adapter is exported and reloaded.

2026-10-05 review fix cycle (S2R-M1..M4, S2R-m1): generator /2.2 keys ``isolated_runtime`` (nothing installed into the
kernel; no restart), ``infrastructure_labels`` and ``guided``; quality outcomes are reported verdicts;
``Swin2SRPipeline.adapt`` restarts from the pinned base and Section 4 restores it before a BYOD re-run; Section 10 runs a
change-one-thing experiment in its own pipeline; BYOD takes a path or one uploaded zip and reports every skipped file.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "swin2sr_super_resolution_pipeline",
    "repo_name": "swin2sr-super-resolution-pipeline",
    "stem": "swin2sr_super_resolution",
    "notebook_name": "swin2sr_super_resolution_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (bioclip2-biodiversity-pipeline): managed CPython, a size- and
    # SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "Swin2SRPipeline",
    "weights_key": "swin2sr-x2-64",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "runtime_imports": ["torch", "transformers"],
    "title": "Swin2SR classical-SR x2 — DIMER E2E super-resolution fine-tuning tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/swin2sr-super-resolution-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/swin2sr-super-resolution-pipeline/blob/main/tutorials/swin2sr_super_resolution_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-caidas%2Fswin2SR--classical--sr--x2--64-ffcc4d?style=flat",
            "https://huggingface.co/caidas/swin2SR-classical-sr-x2-64",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-mv--lab%2Fswin2sr-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/mv-lab/swin2sr",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2209.11345-b31b1b.svg", "https://arxiv.org/abs/2209.11345"),
    ],
    "capability": "2× single-image super-resolution (classical SR) and bounded supervised fine-tuning of the last encoder stage and the reconstruction tail on paired low-/high-resolution crops, using the pinned `caidas/swin2SR-classical-sr-x2-64` weights",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the "
        "pinned `caidas/swin2SR-classical-sr-x2-64` snapshot (a 48 MB `model.safetensors`; no pickle is opened anywhere), fetches "
        "the 360 pinned iNaturalist photographs from the project's open-data bucket (about 39 MB, each refused on any byte-size "
        "or SHA-256 mismatch), cuts each into a 192-px high-resolution crop and its 96-px input (bicubic 2× downsampling, then "
        "JPEG at quality 40), splits them per species into 216 / 48 / 96 pairs, upscales a synthetic scene through the "
        "inference contract with an input manifest and a rejection probe, measures the frozen model's PSNR and SSIM over the "
        "96 held-out crops beside the bicubic and nearest-neighbour baselines, runs a bounded fine-tuning of the last encoder "
        "stage and the reconstruction tail with the L1 loss and validation-PSNR epoch selection, scores the held-out crops "
        "again per species, re-upscales the scene and four held-out crops with the adapted model, exports the adapter as "
        "safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify pixel parity. The default "
        "path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration "
        "edit (NOTEBOOK_SPEC 2.0 §5). On the build workstation's CPU the whole path took about 25 minutes after the "
        "downloads (expect longer on a 2-vCPU hosted runtime); a CUDA runtime is used automatically when present and finishes "
        "in a few minutes."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and either set `BYOD_PATH` to a zip or folder "
        "in the runtime (Colab, Kaggle or Jupyter) or leave it empty to upload one zip in Colab, then choose **Run after** from "
        "that cell. The zip or folder holds high-resolution photographs (optionally with a `labels.csv` of `id`, `file`, "
        "`category`, naming files by their path relative to `labels.csv`). After the split the training split needs eight "
        "photographs and the validation split one, so the **effective minimum is 12 distinct photographs with both sides of "
        "at least 192 px** (split 8 / 2 / 2); smaller photographs, `__MACOSX` entries, hidden and non-image files are skipped "
        "and listed by name, and Section 4 first puts the model back to the pinned base so Sections 5 and 6 score the frozen "
        "model on your photographs. Each is centre-cropped to 192 px and degraded exactly as the sample (bicubic 2× "
        "down, JPEG 40), then passes through the same validation, image-disjoint split, baselines, fine-tuning, held-out "
        "evaluation, artifact export and reload-parity cells as the iNaturalist sample. Uploaded files stay inside this "
        "runtime. BYOD is optional and never part of the default path."
    ),
    "intro": (
        "`caidas/swin2SR-classical-sr-x2-64` is the Swin2SR model of Conde et al. (2022) — a SwinV2 image restoration "
        "transformer (six residual Swin stages of six blocks over 8-px attention windows, a convolution after the body, a "
        "pixel-shuffle upsampler and a final convolution; 12,091,571 parameters, published under the **Apache-2.0** "
        "licence) trained for **classical** 2× super-resolution: the input it expects is a clean image downsampled with a "
        "bicubic filter, and plausible detail is **synthesised** from what the input contains, not recovered from anywhere "
        "else. The output is a uint8 RGB array of shape `(2H, 2W, 3)`; there is no confidence and no abstention.\n\n"
        "What this notebook adds to inference is **adaptation with paired crops** under a degradation the checkpoint was not "
        "trained for. The photographs are real: 360 CC0-licensed, research-grade iNaturalist photographs of six North "
        "American bird species (**CC0 1.0**; 60 per species, one per observer within each species), pinned by photo id, byte size and SHA-256 "
        "and fetched from the open-data bucket at run time. Each becomes a 192-px centre crop (the high-resolution "
        "reference) and a 96-px input made by bicubic 2× downsampling **and then JPEG compression at quality 40** — the "
        "kind of input a phone gallery or a web page actually hands to a super-resolver. On that input the frozen model is "
        "*worse than bicubic interpolation* (the build record measured 25.9 dB against bicubic's 26.3 dB on the 96 test "
        "crops: it sharpens the block artefacts along with the edges), so the honest question is narrow: does a bounded "
        "fine-tuning of the last encoder stage and the reconstruction tail on 216 pairs move **PSNR** and **SSIM** on an "
        "image-disjoint test split, per species, past the two **non-neural baselines** (**bicubic** and **nearest-neighbour** "
        "interpolation of the same input)? Nothing here is a quality claim about your photographs: it is one seeded split "
        "of one sample under one degradation.\n\n"
        "**Snapshot note:** the pinned revision ships `model.safetensors` (a 4-file manifest) — no pickle is opened anywhere "
        "in this notebook. Section 3 stages and digest-verifies those files before the processor or the model is constructed."
    ),
    "guided": {
        "opening": [
            (
                "**Who this notebook is for.** A learner who knows basic Python and NumPy, has used Colab or Jupyter, and wants to see what a pretrained super-resolution model does with a degradation it was not built for, how to measure that honestly against simple interpolation, and how to adapt it to a small paired set without fooling themselves — including what the adaptation costs on clean input. No prior experience with super-resolution or fine-tuning is assumed; each term is explained where it first matters and again in the **Glossary** at the end. A T4 GPU runtime finishes in about ten minutes; CPU works but takes about half an hour.\n\n**Input → Model → Output.**\n\n| | 2× upscaling | Scored reconstruction | Bounded fine-tuning |\n|---|---|---|---|\n| Input | one RGB image, sides 8..512 px | a 96-px input crop and its 192-px reference | paired crops (216 training and 48 validation in the sample) |\n| Model | Swin2SR: six residual Swin stages over 8-px windows, then a pixel-shuffle upsampler | the same model | the last encoder stage, the convolution after the body, the upsampler and the final convolution, trained with the L1 loss; everything else frozen |\n| Output | a uint8 RGB array twice the size — no confidence | PSNR and luma SSIM against the reference | a 10 MB safetensors adapter, and held-out PSNR / SSIM per species |\n\n**How to use this notebook.** Choose a runtime (**Runtime → Change runtime type → T4 GPU** is faster; CPU works), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried package (including the 4,000-line pinned sample table) and the model snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked answer from the recorded run (the Kaggle T4 run of 20 September 2026 and the build record). Section 10 is a **change-one-thing experiment**, off by default. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 photographs, paired crops and a leakage-free split *(evaluation practice)* → 5 the inference contract on a synthetic scene *(core concept: what a super-resolver returns)* → 6 two baselines and the frozen model *(evaluation practice)* → 7 bounded fine-tuning *(core concept: what is trained)* → 8 held-out evaluation, per species → 9 reconstructions by eye, export and reload *(engineering)* → 10 change one thing (optional) → conclude."
            )
        ]
    },
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot; fetch a digest-pinned photograph set, turn it into paired crops under a stated degradation, "
        "validate it and split it by image (no image in two splits, observer overlap reported); upscale a synthetic scene through the public API and read the "
        "output contract correctly (shape, dtype, no confidence, PSNR against a self-made reference is not a benchmark); "
        "measure the frozen model's PSNR and SSIM beside two non-neural baselines and read the per-species breakdown; run a "
        "bounded fine-tuning with the L1 loss, explicit hyperparameters and validation-based epoch selection; evaluate on an "
        "image-disjoint test split; look at the adapted reconstructions next to the frozen ones and the references; and "
        "export a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "scales other than 2×, blind restoration of unknown degradations (only the one stated degradation is trained and "
        "measured), denoising, video, face restoration, perceptual or no-reference quality scores, fine-tuning of the patch "
        "embedding or of any stage but the last ones, evaluation on Set5 / Set14 / DIV2K or any benchmark proper (only one "
        "seeded 360-crop sample is scored here), and any claim that six bird species stand in for your images. Inputs above "
        "512 px per side are refused — tile them yourself. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Learner:** basic Python, NumPy and image-handling familiarity, and Colab or Jupyter; no prior experience with super-resolution or fine-tuning. The notebook explains classical super-resolution, pixel shuffle, PSNR, luma SSIM and the border crop, the L1 loss and the adapter where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter and nothing is installed into it. The default path runs on CPU (float32) and uses CUDA automatically when available. Swin2SR runs windowed attention over every input pixel, so time grows with input area: the build record measured about 71 s to upscale and score the 96 test crops and 1,315 s for the five epochs of fine-tuning over 216 pairs with per-epoch validation scoring; the whole default path took 1,523 s on the build workstation's CPU with the snapshot and photographs already cached, and 419 s on an RTX 5070 Ti. A 2-vCPU hosted runtime will take longer. The pinned `torch==2.14.0` install is the largest download of the run; the checkpoint is 48 MB and the photographs about 39 MB.",
        "- **Knowledge:** basic Python, NumPy and PIL image handling; what PSNR and SSIM measure and why neither is a perceptual quality score; why a self-made reference is a plumbing check and a held-out split under a stated degradation is a measurement of that degradation only.",
        "- **Data contract:** records are `{id, lr, hr}` — `hr` a PIL image (or a file decodable by Pillow) with sides that are multiples of 16 px within 16..1024 px, `lr` exactly half its size (the pair the model is asked to reverse), an optional `category` of at most 32 characters. Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..5,000 records; splitting de-duplicates by decoded high-resolution pixels so no image lands in two splits; training records share one input size so they can be batched. BYOD accepts one zip (or directory) of high-resolution images plus an optional `labels.csv`; the notebook crops and degrades them itself. After the split the training split needs eight photographs and the validation split one, so the effective BYOD minimum is **12 distinct photographs of at least 192 px a side** (`min_byod_records()` computes it). Nothing is dropped silently: smaller photographs, `__MACOSX` entries, hidden and non-image files are skipped and listed with the reason; two images with the same file name in different folders are refused.",
        "- **Validation is structural, not semantic:** every image is decoded and every pair's sizes checked, but nothing checks that a pair is aligned or that the degradation matches your deployment — a mismatched set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path fetches 360 JPEG/PNG files from `https://inaturalist-open-data.s3.amazonaws.com/photos/<id>/medium.<ext>` (about 39 MB in total), each pinned by byte size and SHA-256 in the carried `samples.py` and refused on any mismatch; every photograph's iNaturalist observation page and observer login are kept in its record. Each photograph carries the CC0 1.0 licence its observer chose; nothing is redistributed by this repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. iNaturalist photographs, paired crops and split\n\n"
                "`fetch_corpus` returns the 360 pinned photographs from the cache under `weights/inat-birds/` or the "
                "iNaturalist open-data bucket — every cached file is re-hashed and every fetched file refused on any byte-size or "
                "SHA-256 mismatch — and `read_corpus` turns each into a record: the **high-resolution reference** is the "
                "centred `HR_CROP`-pixel crop of the served photo, the **input** is `degrade(hr)` — bicubic 2× downsampling "
                "followed by JPEG compression at `JPEG_QUALITY` — and the species is the record's `category`. "
                "`build_sample_dataset` draws a seeded stratified split per species (36 / 8 / 16 → 216 / 48 / 96). "
                "`validate_dataset` then checks every record against the contract, `check_split_disjoint` asserts no "
                "high-resolution image (by decoded-pixel digest) is shared, `observer_overlap` reports how many observers "
                "contributed to more than one split (an observation about the draw, not an assertion), and the training "
                "split's summary table is written to `outputs/{stem}_train.csv`.\n\n"
                "Look for: 360 photographs, the six species with 36 / 8 / 16 each, 192-px references and 96-px inputs, three "
                "digests, and four refusal probes — a duplicate id, an input that is not half its reference, a reference whose "
                "sides are not multiples of 16, and a dataset too small to use — each rejected before the model does anything.\n\n"
                "*Evaluation practice.* **Bring your own data (optional):** set `USE_BYOD = True` and either `BYOD_PATH` (a zip or "
                "a folder of photographs, as a path in this runtime — this works on Colab, Kaggle and Jupyter) or leave "
                "`BYOD_PATH` empty to upload exactly one zip through the Colab dialog; then choose **Run after** from this cell. "
                "This cell first puts the model back to the pinned base, so Sections 5 and 6 score the frozen model on your "
                "photographs, and it lists every file it skipped and why. The effective minimum is 12 distinct photographs of at "
                "least 192 px a side.\n\n"
                "**Predict before running:** each of the 360 photographs is a different iNaturalist observation, and within a "
                "species each comes from a different observer. Will any observer still contribute to more than one split?"
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if pipe.adapter is not None:\n"
                "    # A re-run after Section 7 (BYOD, or a new split): Sections 5 and 6 must score the frozen model, not the adapted one.\n"
                "    print({{'restored_pinned_base': len(pipe.restore_base()), 'note': 'the adapted weights were removed; Sections 5-6 score the frozen model again'}})\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_zip = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_zip.exists():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} does not exist (relative paths start at {{Path.cwd()}}): give a .zip or a folder holding high-resolution photographs (and optionally labels.csv).')\n"
                "        file_name = byod_zip.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the zip (or folder) in the runtime and set BYOD_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one .zip file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        if not file_name.lower().endswith('.zip'):\n"
                "            raise ValueError(f'{{file_name}}: upload one .zip holding high-resolution photographs (and optionally labels.csv).')\n"
                "        byod_zip = Path('work') / 'byod.zip'\n"
                "        byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_zip.write_bytes(payload)\n"
                "    byod_report = {{}}\n"
                "    records = load_byod_dataset(byod_zip, report=byod_report)\n"
                "    for item in byod_report['skipped']:\n"
                "        print({{'skipped': item['file'], 'reason': item['reason']}})\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod_images': byod_report['images'], 'used': len(records), 'skipped': len(byod_report['skipped']), 'labelled': byod_report['labelled'], 'duplicate_images_dropped': len(records) - sum(len(part) for part in splits.values()), 'effective_minimum': min_byod_records()['total']}}\n"
                "    if len(splits['test']) < 20:\n"
                "        print({{'caution': f\"only {{len(splits['test'])}} held-out test crops: PSNR and SSIM carry no dispersion estimate at this size; add photographs before reading them\"}})\n"
                "else:\n"
                "    corpus_files = fetch_corpus(cache_dir='weights/inat-birds')\n"
                "    corpus = read_corpus(corpus_files)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}}: {{CORPUS_RELEASE}} ({{CORPUS_LICENSE}})'\n"
                "    raw_rows = {{'photographs': len(corpus), 'bytes': sum(len(v) for v in corpus_files.values()), 'observers': len({{r['observer'] for r in corpus}})}}\n"
                "# The training split must hold MIN_RECORDS; validation and test only need one pair each (split_dataset checks that).\n"
                "dataset_manifests = {{name: validate_dataset(part, min_records=MIN_RECORDS if name == 'train' else 1) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'observer_overlap': observer_overlap(splits), 'degradation': {{'hr_crop': HR_CROP, 'downsample': f'bicubic x{{UPSCALE}}', 'jpeg_quality': JPEG_QUALITY}}}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'category_counts': manifest['category_counts'], 'hr_side': manifest['hr_side'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "print({{'example': {{'id': example['id'], 'category': example.get('category'), 'hr': list(example['hr'].size), 'lr': list(example['lr'].size), 'observation': example.get('inat_observation_url')}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'input is not half its reference': [{{**train_records[0], 'lr': train_records[0]['hr']}}, *train_records[1:8]],\n"
                "    'reference sides not multiples of 16': [{{**train_records[0], 'hr': train_records[0]['hr'].resize((200, 200)), 'lr': train_records[0]['lr'].resize((100, 100))}}, *train_records[1:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the three split sizes, the 192-px references and 96-px inputs, the `observer_overlap` line and the four refusals.\n\n<details><summary>Check your reasoning</summary>Yes: in the recorded run 53 of the 222 observers contributed to more than one split. Within a species every photograph has its own observer, but some observers photographed several of the six species, and the split is stratified per species. No image is shared (the pixel-digest check proves that); the same photographer's camera and processing can appear on both sides, which matters more for super-resolution than for classification because the model learns the camera's texture. The notebook reports the overlap rather than hiding it.</details>"
            ),
        },
        {
            "md": (
                "## 5. Upscale a synthetic scene through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: a deterministic 128×96 "
                "RGB scene built in code (a colour gradient, a filled square, a diagonal stripe — sharp edges are what a "
                "super-resolver has to reconstruct), kept as its own high-resolution reference and bicubic-downscaled to "
                "64×48 to become the input; a different image family from the photographs, and an image the adapted model "
                "will upscale again in Section 9. `validate_inputs` applies exactly the checks `upscale` applies "
                "(`MIN_INPUT_SIDE` 8, `MAX_INPUT_SIDE` 512) and returns an input manifest; an oversize image is validated too "
                "and its rejection recorded as a finding. `upscale` returns a uint8 RGB array of shape `(2H, 2W, 3)` with the "
                "input and output sizes and the model identity — no confidence, no abstention. The per-image "
                "`evaluation_report` against the self-made reference is `sample-sanity` (PSNR beside a bicubic upscale of the "
                "same input) — plumbing evidence, not a measurement; whether the model is *better* is what Section 6 measures "
                "on 96 photographs.\n\n"
                "*Core concept.* A super-resolver returns pixels and nothing else: no confidence, no flag where it invented "
                "detail. **PSNR** compares those pixels with a reference you already have, which is why every number here needs "
                "a reference.\n\n"
                "**Predict before running:** the scene's input is a clean bicubic downscale — exactly the degradation this "
                "checkpoint was trained on. Will the model beat a bicubic upscale of the same input by a little, or by a lot?"
            ),
            "code": (
                "HR_WIDTH, HR_HEIGHT = 128, 96\n"
                "ramp = np.linspace(0.0, 255.0, HR_WIDTH)\n"
                "red = np.tile(ramp, (HR_HEIGHT, 1))\n"
                "green = np.tile(np.linspace(255.0, 0.0, HR_HEIGHT), (HR_WIDTH, 1)).T\n"
                "blue = np.full((HR_HEIGHT, HR_WIDTH), 96.0)\n"
                "scene_reference = Image.fromarray(np.rint(np.stack([red, green, blue], axis=-1)).astype(np.uint8), mode='RGB')\n"
                "draw = ImageDraw.Draw(scene_reference)\n"
                "draw.rectangle([24, 20, 60, 56], fill=(20, 20, 20))\n"
                "draw.line([(70, 84), (118, 12)], fill=(250, 250, 250), width=5)\n"
                "scene_input = scene_reference.resize((HR_WIDTH // UPSCALE, HR_HEIGHT // UPSCALE), Image.Resampling.BICUBIC)\n"
                "scene_name = f'synthetic_shapes_{{HR_WIDTH // UPSCALE}}x{{HR_HEIGHT // UPSCALE}}.png'\n"
                "scene_sha256 = {{'input': hashlib.sha256(np.asarray(scene_input).tobytes()).hexdigest(), 'reference': hashlib.sha256(np.asarray(scene_reference).tobytes()).hexdigest()}}\n"
                "print({{'ceilings': {{'MIN_INPUT_SIDE': MIN_INPUT_SIDE, 'MAX_INPUT_SIDE': MAX_INPUT_SIDE, 'UPSCALE': UPSCALE, 'HR_CROP': HR_CROP, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'device': pipe.device}}}})\n"
                "input_manifest = validate_inputs(scene_input, names=[scene_name])\n"
                "try:\n"
                "    validate_inputs(Image.new('RGB', (MAX_INPUT_SIDE + 1, MIN_INPUT_SIDE)))\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'oversize-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'scene': scene_name, 'sha256': {{k: v[:16] + '...' for k, v in scene_sha256.items()}}, 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n\n\n"
                "def upscale_scene(pipeline, label):\n"
                "    started = time.perf_counter()\n"
                "    result = pipeline.upscale(scene_input)\n"
                "    elapsed = time.perf_counter() - started\n"
                "    array = np.asarray(result['image'])\n"
                "    checks = {{\n"
                "        'shape': array.shape == (scene_input.height * UPSCALE, scene_input.width * UPSCALE, 3),\n"
                "        'dtype_uint8': array.dtype == np.uint8,\n"
                "        'output_size_reported': tuple(result['output_size']) == (scene_input.width * UPSCALE, scene_input.height * UPSCALE),\n"
                "        'identity_reported': result['model_id'] == MODEL_ID and result['model_revision'] == MODEL_REVISION,\n"
                "    }}\n"
                "    if not all(checks.values()):\n"
                "        raise RuntimeError(f'upscale output failed a sanity check: {{checks}}')\n"
                "    report = evaluation_report(result, scene_reference, low_resolution=scene_input, sample_kind='synthetic (authored in this notebook)')\n"
                "    Image.fromarray(array, mode='RGB').save(f'outputs/{stem}_output_{{label}}.png')\n"
                "    print({{label: {{'seconds': round(elapsed, 3), 'checks': checks, 'psnr': {{m['id']: round(m['value'], 2) for m in report['metrics']}}, 'bicubic_baseline': {{b['id']: round(b['value'], 2) for b in report['baselines']}}, 'verdict': report['verdict']}}}})\n"
                "    return result, report\n\n\n"
                "frozen_scene_result, frozen_scene = upscale_scene(pipe, 'frozen')"
            ),
        },
        {
            "md": (
                "**What to notice:** the four structural checks, the model's PSNR beside the bicubic baseline, and the verdict.\n\n<details><summary>Check your reasoning</summary>By a lot: the build record measured 37.1 dB for the frozen model on this scene, several decibels above bicubic, because sharp edges on a clean bicubic downscale are exactly what the checkpoint learned to restore. The verdict is `sample-sanity` — one scene you authored, plumbing evidence, not a measurement. Keep the number: Section 9 upscales the same scene with the adapted model.</details>"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model on the test crops\n\n"
                "Three systems frame the adaptation, each read two ways by `sr_metrics` (carried in `metrics.py`): **PSNR** "
                "in dB over the RGB channels against the high-resolution reference and **SSIM** on the luma channel (11-tap "
                "Gaussian window, sigma 1.5, 5-px border excluded — the classical-SR convention), averaged over the crops and "
                "per species. The **nearest-neighbour** baseline replicates each input pixel 2×2 — the floor. The **bicubic** "
                "baseline is the interpolation every image viewer applies and the reference point of the super-resolution "
                "literature. The **frozen model** is scored by `pipe.evaluate`, which upscales every record's input through "
                "`upscale` and scores the reconstruction. Expect the frozen model **below bicubic** on this input — the build "
                "record measured 25.9 dB / 0.785 against bicubic's 26.3 dB / 0.797 — because a classical-SR model sharpens "
                "JPEG block edges as if they were detail; read the per-species rows to see that it holds for every species.\n\n"
                "*Evaluation practice.* The cell reports a **verdict** — whether the frozen model is above the nearest-neighbour "
                "floor — and records it; it does not stop the notebook when the answer is no.\n\n"
                "**Predict before running:** on photographs that were downscaled *and* JPEG-compressed, will the frozen model "
                "beat bicubic interpolation?"
            ),
            "code": (
                "METRICS = ('psnr', 'ssim')\n"
                "baseline_nearest = nearest_baseline(test_records)\n"
                "baseline_bicubic = bicubic_baseline(test_records)\n"
                "print({{'nearest_baseline': {{k: round(baseline_nearest[k], 3) for k in METRICS}}, 'n': baseline_nearest['n'], 'note': baseline_nearest['baseline']}})\n"
                "print({{'bicubic_baseline': {{k: round(baseline_bicubic[k], 3) for k in METRICS}}, 'note': baseline_bicubic['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "frozen_fields = {{c: {{'n': v['n'], 'psnr': round(v['psnr'], 2), 'ssim': round(v['ssim'], 3)}} for c, v in frozen_test['per_category'].items()}}\n"
                "print({{'by_species_frozen': frozen_fields}})\n"
                "# A reported verdict, not an assertion: on other photographs the frozen model may lose even to nearest-neighbour, and that is a finding.\n"
                "frozen_verdict = {{'above_nearest': bool(frozen_test['psnr'] > baseline_nearest['psnr']), 'above_bicubic': bool(frozen_test['psnr'] > baseline_bicubic['psnr'])}}\n"
                "print({{'frozen_vs_baselines': frozen_verdict}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the three systems on PSNR and SSIM, the per-species rows, and the verdict line.\n\n<details><summary>Check your reasoning</summary>No. In the recorded run nearest-neighbour scored 25.29 dB / 0.764, bicubic 26.31 dB / 0.797 and the frozen model 25.92 dB / 0.785: above the floor, below bicubic, for every species. The model sharpens the JPEG block edges as if they were detail — a pretrained model on the wrong degradation can lose to interpolation, which is why bicubic is the baseline to beat.</details>'
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the last stage and the reconstruction tail\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_STAGES` residual Swin stages of the encoder, the convolution "
                "after the body, the pixel-shuffle upsampler and the final convolution — one stage by default, 2,463,011 of "
                "12,091,571 parameters — while the patch embedding, the first convolution and the earlier stages stay "
                "frozen. Every batch of inputs (scaled to 0..1 exactly as the processor scales them; the pair contract keeps "
                "their sides multiples of the attention window, so no padding is involved) is reconstructed and compared "
                "with its reference under the **L1 loss** the checkpoint was trained with. AdamW without weight decay at a "
                "fixed learning rate, gradient clipping at 1.0, seeded shuffling, no scheduler. Epoch 0 records the frozen "
                "model's validation metrics; every epoch is scored on the 48 validation pairs, and the epoch with the highest "
                "validation PSNR is kept.\n\n"
                "Watch the training loss fall from about 0.035 while the validation PSNR climbs by roughly a decibel over "
                "five epochs and SSIM by about 0.03: the model is learning to *not* sharpen the block artefacts, which a "
                "handful of pairs is enough to teach the tail. The build record's counter-examples — two stages at twice the "
                "rate, or more epochs — are in the model card; the default is the smallest configuration that beat bicubic "
                "on the held-out split.\n\n"
                "*Core concept.* Every call to `pipe.adapt` starts from the **pinned base**: tensors an earlier call changed — "
                "whichever stages it trained — are restored first, so epoch 0 is always the frozen model. Re-running this cell "
                "with a changed field is a fresh run, not continued training, but it replaces the default results that "
                "Sections 8 and 9 export; to compare a change side by side, use Section 10.\n\n"
                "**Predict before running:** will the validation PSNR rise every epoch, or peak and fall?"
            ),
            "code": (
                "EPOCHS = 5  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_STAGES = 1  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 5)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "settings = {{'epochs': EPOCHS, 'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'trainable_stages': TRAINABLE_STAGES}}\n"
                "if settings != {{'epochs': 5, 'lr': 1e-4, 'batch_size': 8, 'trainable_stages': 1}}:\n"
                "    print({{'note': 'changed settings: this run starts again from the pinned base and replaces the default results of Sections 8-9; Section 10 compares a change side by side instead', 'settings': settings}})\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_stages=TRAINABLE_STAGES, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'loss': adapt_result['loss'], 'started_from': adapt_result['started_from'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                '**What to notice:** epoch 0 (the frozen model), the loss column, the validation PSNR and SSIM columns, and `best_epoch`.\n\n<details><summary>Check your reasoning</summary>It rose every epoch in the build record: validation PSNR 25.90 (epoch 0, the frozen model) → 26.40 → 26.56 → 26.68 → 26.74 → 26.78 dB, so the last epoch, 5, was kept, and the training loss was still falling. Five epochs are therefore a budget, not a converged point — but the held-out test, not the validation curve, decides whether it was enough.</details>'
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test crops were never used for training or epoch selection, and no image appears in two splits. The "
                "adapted model is scored exactly as the frozen model was in Section 6, the four systems are put side by side "
                "on both measures, and the per-species breakdown is repeated. Read it in this order: **PSNR** first (the "
                "measure the epoch was selected on — the build record measured 25.9 → 26.9 dB, past bicubic's 26.3), then "
                "**SSIM** (0.785 → 0.816, past bicubic's 0.797), then the per-species rows, where every species gained. The "
                "cell reports **verdicts** — whether the adapted PSNR `improved`, showed `no gain` or got `worse` against the "
                "frozen one, and whether it is above bicubic — and records them in the evaluation report; a fine-tune that does "
                "not help is a finding, not an error, and Section 9 still exports, reloads and writes the result. Ninety-six "
                "crops from one seeded split under one degradation give **no dispersion estimate**; the deltas are "
                "sample-sanity evidence that the adaptation contract works, not a benchmark, and a gain under JPEG-40 "
                "bicubic downsampling says nothing about a camera's own blur or noise until you measure them.\n\n"
                "**Predict before running:** will the adapted model beat bicubic — the baseline the frozen model lost to?"
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "adapted_fields = {{c: {{'n': v['n'], 'psnr': round(v['psnr'], 2), 'ssim': round(v['ssim'], 3)}} for c, v in adapted_test['per_category'].items()}}\n"
                "comparison = {{metric: {{'nearest': round(baseline_nearest[metric], 3), 'bicubic': round(baseline_bicubic[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['delta_vs_bicubic'] = {{metric: round(adapted_test[metric] - baseline_bicubic[metric], 3) for metric in METRICS}}\n"
                "comparison['by_species'] = {{c: {{'n': frozen_fields[c]['n'], 'frozen_psnr': frozen_fields[c]['psnr'], 'adapted_psnr': adapted_fields[c]['psnr'], 'frozen_ssim': frozen_fields[c]['ssim'], 'adapted_ssim': adapted_fields[c]['ssim']}} for c in frozen_fields}}\n"
                "# Reported verdicts, not assertions: a non-improving fine-tune is a result to record, and export and reload still run.\n"
                "delta_psnr = adapted_test['psnr'] - frozen_test['psnr']\n"
                "adaptation_verdict = 'improved' if delta_psnr > 0 else ('no gain' if delta_psnr == 0 else 'worse')\n"
                "comparison['verdicts'] = {{'frozen_vs_baselines': frozen_verdict, 'adapted_vs_frozen_psnr': adaptation_verdict, 'adapted_above_bicubic': bool(adapted_test['psnr'] > baseline_bicubic['psnr'])}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'degradation': {{'hr_crop': HR_CROP, 'downsample': f'bicubic x{{UPSCALE}}', 'jpeg_quality': JPEG_QUALITY}},\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'baselines': {{'nearest': {{k: v for k, v in baseline_nearest.items() if k != 'per_record'}}, 'bicubic': {{k: v for k, v in baseline_bicubic.items() if k != 'per_record'}}}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'verdicts': comparison['verdicts']}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the four-way table, the `verdicts` row, and the per-species deltas.\n\n<details><summary>Check your reasoning</summary>Yes. In the recorded run the verdict was *improved*: PSNR 25.92 → 26.91 dB and SSIM 0.785 → 0.816, past bicubic (26.31 dB / 0.797), and every species gained. The tail learned not to sharpen block artefacts — a small, consistent gain under one stated degradation, with no dispersion estimate from one seeded split of 96 crops.</details>'
            ),
        },
        {
            "md": (
                "## 9. Look at the reconstructions, export the adapter and reload it\n\n"
                "The synthetic scene from Section 5 is upscaled again by the adapted model — an image family the adaptation "
                "never saw, so this is a small look at what the adaptation did *outside* its corpus: the build record measured "
                "37.1 dB frozen and 33.5 dB adapted on this clean, bicubic-only input — the adapted tail now expects JPEG "
                "artefacts and smooths a clean image, a real cost of the adaptation to record, not a failure — and four held-out "
                "crops are written as side-by-side panels (`outputs/{stem}_examples/`: input replicated 2×, bicubic, frozen, "
                "adapted, reference) so the numbers can be checked by eye: the adapted panels should show the block edges "
                "smoothed and the feather detail kept.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the last stage, the convolution after the body, the "
                "upsampler and the final convolution, about 10 MB — as `adapter.safetensors`, with a `manifest.json` recording "
                "the artifact format, the base model id and revision, the digest of the base `model.safetensors`, the tensor "
                "names, the file size and SHA-256, the training configuration and the epoch history (OUT8). "
                "`Swin2SRPipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest, its digest and "
                "its exact tensor set **before** deserialising, refuses any tensor outside the encoder stages and the "
                "reconstruction tail, and overlays the tensors onto a freshly loaded base — a new object from files, not the "
                "in-memory model (VER2). The cell asserts pixel-identical reconstructions on eight test crops (VER4) — a "
                "contract check, so it stays a hard check.\n\n"
                "**Predict before running:** the adapted model learned to smooth JPEG artefacts. Will it score higher or lower "
                "than the frozen model on the clean synthetic scene from Section 5?"
            ),
            "code": (
                "import shutil\n\n"
                "adapted_scene_result, adapted_scene = upscale_scene(pipe, 'adapted')\n"
                "examples_dir = Path('outputs/{stem}_examples')\n"
                "shutil.rmtree(examples_dir, ignore_errors=True)\n"
                "examples_dir.mkdir(parents=True)\n"
                "frozen_base = Swin2SRPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "for record in test_records[:4]:\n"
                "    hr = record['hr']\n"
                "    panels = [\n"
                "        record['lr'].resize(hr.size, Image.Resampling.NEAREST),\n"
                "        record['lr'].resize(hr.size, Image.Resampling.BICUBIC),\n"
                "        Image.fromarray(frozen_base.upscale(record['lr'])['image'], mode='RGB'),\n"
                "        Image.fromarray(pipe.upscale(record['lr'])['image'], mode='RGB'),\n"
                "        hr,\n"
                "    ]\n"
                "    sheet = Image.new('RGB', (hr.width * 5 + 40, hr.height), (255, 255, 255))\n"
                "    for i, panel in enumerate(panels):\n"
                "        sheet.paste(panel, (i * (hr.width + 10), 0))\n"
                "    sheet.save(examples_dir / f\"{{record['id']}}.png\")\n"
                "print({{'examples': sorted(p.name for p in examples_dir.iterdir()), 'panel_order': ['input x2 nearest', 'bicubic', 'frozen', 'adapted', 'reference']}})\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = Swin2SRPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [pipe.upscale(r['lr'])['image'] for r in test_records[:8]]\n"
                "after = [reloaded.upscale(r['lr'])['image'] for r in test_records[:8]]\n"
                "parity = {{'identical_images': sum(np.array_equal(a, b) for a, b in zip(before, after, strict=True)), 'of': len(before), 'max_abs_difference': int(max(np.abs(a.astype(int) - b.astype(int)).max() for a, b in zip(before, after, strict=True)))}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_images'] == parity['of']\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': pipe.weight_sha256}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'base_url': CORPUS_BASE_URL, 'bytes': CORPUS_BYTES, 'pinned_photographs': len(SAMPLE_RECORDS), 'species': {{k: list(v) for k, v in SPECIES.items()}}, 'degradation': {{'hr_crop': HR_CROP, 'downsample': f'bicubic x{{UPSCALE}}', 'jpeg_quality': JPEG_QUALITY}}}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'scene': {{'name': scene_name, 'sha256': scene_sha256}}, 'frozen_report': frozen_scene, 'adapted_report': adapted_scene, 'output_files': ['outputs/{stem}_output_frozen.png', 'outputs/{stem}_output_adapted.png']}},\n"
                "    'comparison': comparison,\n"
                "    'examples': 'outputs/{stem}_examples',\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32'}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                "**What to notice:** the scene's PSNR before and after, the panels in `outputs/{stem}_examples/`, and `identical_images` against `of`.\n\n<details><summary>Check your reasoning</summary>Lower. The build record measured 37.1 dB frozen and 33.5 dB adapted on the clean scene: the tail now expects JPEG artefacts and smooths an image that has none — a real cost of adapting to one degradation, recorded rather than hidden. Reload parity held: 8 of 8 crops pixel-identical (maximum difference 0).</details>"
            ),
        },
        {
            "md": (
                "## 10. Change one thing: how many stages to train (optional)\n\n"
                "*Evaluation practice.* A **Predict → Change one thing → Run → Observe → Explain** activity, off by default so "
                "Run all is unaffected. Set `RUN_EXPERIMENT = True`, change **one** field — by default two encoder stages train "
                "instead of one — and run this cell after Sections 4–9. The experiment loads its **own** pipeline from the "
                "verified snapshot, so it starts from the frozen model and never touches the default `pipe`; it writes only to "
                "`outputs/{stem}_experiment/`, prints the default and the changed run side by side, and checks that the default "
                "exports (adapter, evaluation report, result) are byte-identical afterwards. Expect a few minutes on a T4 and "
                "about twenty on CPU.\n\n"
                "**Predict:** with twice the trainable stages, will the held-out PSNR rise further, stay, or fall? And what "
                "should epoch 0 of the experiment show?"
            ),
            "code": (
                "RUN_EXPERIMENT = False  # @param {{type:\"boolean\"}}\n"
                "EXPERIMENT_TRAINABLE_STAGES = 2  # @param {{type:\"integer\"}}\n"
                "EXPERIMENT_EPOCHS = 5  # @param {{type:\"integer\"}}\n"
                "EXPERIMENT_LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n\n"
                "if not RUN_EXPERIMENT:\n"
                "    print({{'experiment': 'skipped (RUN_EXPERIMENT = False); the default path above is complete'}})\n"
                "else:\n"
                "    canonical_files = {{'adapter': artifact_dir / 'adapter.safetensors', 'evaluation_report': Path('outputs/{stem}_evaluation_report.json'), 'result': Path('outputs/{stem}_result.json')}}\n"
                "    canonical = {{name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in canonical_files.items()}}\n"
                "    experiment_dir = Path('outputs/{stem}_experiment')\n"
                "    shutil.rmtree(experiment_dir, ignore_errors=True)\n"
                "    experiment_dir.mkdir(parents=True)\n"
                "    # Its own pipeline from the verified snapshot: the experiment starts from the frozen model and the default pipe is untouched.\n"
                "    experiment_pipe = Swin2SRPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "    experiment_result = experiment_pipe.adapt(train_records, val_records, epochs=EXPERIMENT_EPOCHS, lr=EXPERIMENT_LEARNING_RATE, batch_size=BATCH_SIZE, trainable_stages=EXPERIMENT_TRAINABLE_STAGES, progress=report)\n"
                "    experiment_test = experiment_pipe.evaluate(test_records)\n"
                "    experiment_pipe.save_artifact(experiment_dir / 'adapter', metadata={{'tutorial': '{stem}', 'experiment': True, 'data_source': data_source}})\n"
                "    side_by_side = {{\n"
                "        'settings': {{'default': {{'trainable_stages': adapt_result['trainable_stages'], 'epochs': adapt_result['epochs'], 'lr': adapt_result['lr']}}, 'experiment': {{'trainable_stages': EXPERIMENT_TRAINABLE_STAGES, 'epochs': EXPERIMENT_EPOCHS, 'lr': EXPERIMENT_LEARNING_RATE}}}},\n"
                "        'trainable_parameters': {{'default': adapt_result['n_trainable'], 'experiment': experiment_result['n_trainable']}},\n"
                "        'epoch0_validation_psnr (frozen model)': {{'default': round(adapt_result['history'][0]['val']['psnr'], 3), 'experiment': round(experiment_result['history'][0]['val']['psnr'], 3)}},\n"
                "        'best_epoch': {{'default': adapt_result['best_epoch'], 'experiment': experiment_result['best_epoch']}},\n"
                "        'test': {{metric: {{'bicubic': round(baseline_bicubic[metric], 3), 'frozen': round(frozen_test[metric], 3), 'default': round(adapted_test[metric], 3), 'experiment': round(experiment_test[metric], 3)}} for metric in METRICS}},\n"
                "        'adapter_bytes': {{'default': artifact_manifest['files'][0]['bytes'], 'experiment': (experiment_dir / 'adapter' / 'adapter.safetensors').stat().st_size}},\n"
                "    }}\n"
                "    for key, row in side_by_side.items():\n"
                "        print({{key: row}})\n"
                "    with open(experiment_dir / 'experiment_report.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'side_by_side': side_by_side, 'history': experiment_result['history'], 'test_metrics': experiment_test}}, handle, indent=2, ensure_ascii=False)\n"
                "    unchanged = {{name: hashlib.sha256(path.read_bytes()).hexdigest() == canonical[name] for name, path in canonical_files.items()}}\n"
                "    if not all(unchanged.values()):\n"
                "        raise RuntimeError(f'the experiment changed a default export: {{unchanged}}')\n"
                "    print({{'default_exports_unchanged': unchanged, 'experiment_outputs': str(experiment_dir)}})\n"
                "    del experiment_pipe"
            ),
        },
        {
            "md": (
                "**Observe → Explain.** Compare the `test` columns with bicubic and with the default run, and the adapter sizes.\n\n"
                "<details><summary>Check your reasoning</summary>Epoch 0 of the experiment must equal epoch 0 of the default run "
                "(validation PSNR 25.90 in the build record): both are the frozen model, which the restart-from-base rule "
                "guarantees. No experiment run is recorded in this notebook's record yet; the model card summarises the build "
                "sweep, where two stages at twice the rate did not beat the default. More trainable stages give the model more "
                "capacity to fit 216 pairs — which can raise the held-out PSNR or overfit the training crops — and a larger "
                "adapter; with 48 validation and 96 test crops, a difference of a few hundredths of a decibel is within what a "
                "different seed could produce. Say which it was on your run.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A classical-SR checkpoint given JPEG-compressed input does worse than bicubic interpolation — it sharpens the block "
        "artefacts as if they were detail — and a bounded fine-tuning of one encoder stage and the reconstruction tail on "
        "216 pairs turns that around (PSNR 25.9 → 26.9 dB in the build record, past bicubic's 26.3; SSIM 0.785 → 0.816, past "
        "0.797), consistently across the six species, with a 10 MB adapter that reloads pixel-for-pixel. That is the claim: the "
        "adaptation contract works end to end on a real paired set under a stated degradation, and the numbers it produces are "
        "read on two measures, per species, against two non-neural baselines and the frozen model rather than in isolation.\n\n"
        "The test split is 96 crops from one seeded draw of one sample under one degradation, the validation split that picks "
        "the epoch is 48, and both measures are reference-based signal fidelity, not perceptual quality — a smoother output "
        "scores higher on PSNR even when a viewer prefers the sharper one. So a gain here says the contract works under "
        "JPEG-40 bicubic downsampling, not that the adapted model is better on your images, that it handles a camera's blur "
        "or noise, or that it is safe to trust synthesised detail: the model still invents plausible texture, and it has no "
        "way to say so. Fine-tuning on a narrow set also erodes the model elsewhere: the synthetic scene re-upscaled in "
        "Section 9 — clean bicubic-only input, the degradation the checkpoint was built for — lost 3.6 dB in the build record "
        "(37.1 → 33.5), one image of evidence that the adapted model is now a JPEG-input model, not a measurement.\n\n"
        "Three things to carry to real data. **Baselines first:** bicubic and nearest-neighbour upscales of *your* inputs, "
        "scored against *your* references, are the numbers to read before any adapted one, per subset. **Degradation:** the "
        "pairs the model learns from define what it learns to undo; make your LR inputs the way your deployment makes them "
        "(the contract's `degrade` is one stated recipe, and BYOD applies it to your photographs). **Leakage:** keep every "
        "image in one split (the contract de-duplicates by decoded pixels) and split by photographer or session when your "
        "images come from few sources — the sample's observer overlap is printed for exactly that reason.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real photograph set and pair it under a "
        "stated degradation, validate the demonstrated dataset contract without leakage, execute the inference contract and a "
        "bounded fine-tuning, evaluate against two trivial baselines and the frozen model on an image-disjoint split, and emit "
        "the shown machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark "
        "superiority, restoration quality under any other degradation, perceptual quality, or production fitness.\n\n"
        "**Optional experiments (off by default; each names its field and what to run):** Section 10 trains two stages in its "
        "own pipeline and prints it beside the default run — change `EXPERIMENT_TRAINABLE_STAGES`, `EXPERIMENT_EPOCHS` or "
        "`EXPERIMENT_LEARNING_RATE` (for example `2e-4`, a faster, less stable climb) there, one at a time, and run that cell "
        "again; every run starts from the frozen model and writes only to `outputs/{stem}_experiment/`. Changing the fields "
        "in Section 7 and choosing **Run after** also starts from the pinned base, but replaces the default results and "
        "exports. BYOD: `USE_BYOD` and `BYOD_PATH` in Section 4, then **Run after** from Section 4, and read the two "
        "baselines before the adapted number.\n\n"
        "## Troubleshooting\n\n"
'- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused, an incomplete one is finished. If it repeats, the network is blocking or altering `files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and choose **Run all**; leave the optional experiment off on a small runtime.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again; the snapshot comes from `huggingface.co`.\n- **CUDA out of memory in Section 7 or 10** — set `BATCH_SIZE = 4` in Section 7 and choose **Run after** from Section 7 (the numbers will differ a little from the recorded run).\n- **Section 4 cannot fetch a photograph, or one fails its digest** — `fetch_corpus` names the photo; the open-data bucket may be briefly unreachable. Run Section 4 again (cached photographs are re-hashed, not re-downloaded); delete `weights/inat-birds/` if a cached file is corrupt.\n- **BYOD: "BYOD_PATH … does not exist"** — the path is relative to the working directory printed in the message.\n- **BYOD: "the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the zip in the runtime (or attach it as a dataset) and set `BYOD_PATH`.\n- **BYOD: "Upload exactly one .zip file"** — the dialog was cancelled or several files were chosen; run the cell again.\n- **BYOD: a `skipped` line names your photograph** — the reason is printed: smaller than the 192-px crop, not an image, or an operating-system file.\n- **BYOD: "labels.csv line N (file …): names an image that is not in the dataset"** — name images by their path relative to the folder holding `labels.csv`.\n- **BYOD: "two images share the file name …"** — rename one of the two.\n- **BYOD: "… leave the training split with … supply at least 12 distinct usable photographs"** — add photographs of at least 192 px a side.\n'
        "## Glossary\n\n"
        "- **Super-resolution (2×)** — producing an image twice as wide and tall from a smaller one; the added detail is "
        "inferred, not recovered.\n"
        "- **Classical SR** — trained on bicubic downscales only; a different degradation (JPEG, blur, noise) is outside "
        "what it learned.\n"
        "- **Degradation** — the recipe that makes the low-resolution input from the reference (here bicubic ÷2, then JPEG "
        "at quality 40).\n"
        "- **Swin stage / attention window** — a group of transformer blocks that attend within 8-px windows.\n"
        "- **Pixel shuffle** — the upsampler that rearranges channels into a twice-larger image.\n"
        "- **PSNR** — peak signal-to-noise ratio in dB: higher means pixel values closer to the reference; it rewards "
        "smoothness.\n"
        "- **SSIM (luma, border crop)** — structural similarity on the brightness channel, ignoring a 5-px border, the "
        "classical-SR convention.\n"
        "- **Nearest-neighbour / bicubic baseline** — replicate each pixel 2×2; the interpolation every image viewer "
        "uses.\n"
        "- **L1 loss** — the mean absolute pixel difference, the loss the checkpoint was trained with.\n"
        "- **Epoch / learning rate / validation selection** — one pass over the training pairs; the update step size; "
        "choosing the epoch on pairs that are neither trained on nor used for the final score.\n"
        "- **Held-out test split / leakage** — crops never used for training or selection; the same image or source on "
        "both sides of a split.\n"
        "- **Adapter / reload parity** — the trained tensors only (safetensors) overlaid on the pinned base; the reloaded "
        "pipeline produces identical pixels.\n"
        "- **BYOD** — bring your own data: your photographs, cropped and degraded the same way, through the same cells.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional — fill in from **your** run, not the recorded one:\n\n"
        "- The task was ___ under the degradation ___ on ___ test crops.\n"
        "- Nearest-neighbour scored ___ dB, bicubic ___ dB and the frozen model ___ dB; the frozen verdict was ___.\n"
        "- After fine-tuning ___ stage(s), PSNR moved from ___ to ___ dB and SSIM from ___ to ___; the verdict was ___.\n"
        "- On the clean scene the adapted model scored ___ against ___ before, because ___.\n"
        "- One reason not to trust this number on my own images yet: ___ (for example a different degradation, observer "
        "overlap, or the size of the test split).\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/swin2sr-super-resolution-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/swin2sr-super-resolution-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/swin2sr-super-resolution-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/mv-lab/swin2sr\n"
        "- Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration (Conde, Choi, Burchi, Timofte, 2022): https://arxiv.org/abs/2209.11345\n"
        "- iNaturalist open data (CC0 photographs, each observer's own licence): https://www.inaturalist.org/pages/developers — bucket https://inaturalist-open-data.s3.amazonaws.com/\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
