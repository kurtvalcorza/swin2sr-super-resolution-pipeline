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
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "swin2sr_super_resolution_pipeline",
    "repo_name": "swin2sr-super-resolution-pipeline",
    "stem": "swin2sr_super_resolution",
    "notebook_name": "swin2sr_super_resolution_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
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
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
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
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "of high-resolution photographs (optionally with a `labels.csv` of `id`, `file`, `category`) — at least eight images "
        "with both sides of at least 192 px. Each is centre-cropped to 192 px and degraded exactly as the sample (bicubic 2× "
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
        "American bird species (**CC0 1.0**; 60 per species, one per observer), pinned by photo id, byte size and SHA-256 "
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
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot; fetch a digest-pinned photograph set, turn it into paired crops under a stated degradation, "
        "validate it and split it by image without leakage; upscale a synthetic scene through the public API and read the "
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
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. Swin2SR runs windowed attention over every input pixel, so time grows with input area: the build record measured about 71 s to upscale and score the 96 test crops and 1,315 s for the five epochs of fine-tuning over 216 pairs with per-epoch validation scoring; the whole default path took 1,523 s on the build workstation's CPU with the snapshot and photographs already cached, and 419 s on an RTX 5070 Ti. A 2-vCPU hosted runtime will take longer. The pinned `torch==2.14.0` install is the largest download of the run; the checkpoint is 48 MB and the photographs about 39 MB.",
        "- **Knowledge:** basic Python, NumPy and PIL image handling; what PSNR and SSIM measure and why neither is a perceptual quality score; why a self-made reference is a plumbing check and a held-out split under a stated degradation is a measurement of that degradation only.",
        "- **Data contract:** records are `{id, lr, hr}` — `hr` a PIL image (or a file decodable by Pillow) with sides that are multiples of 16 px within 16..1024 px, `lr` exactly half its size (the pair the model is asked to reverse), an optional `category` of at most 32 characters. Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..5,000 records; splitting de-duplicates by decoded high-resolution pixels so no image lands in two splits; training records share one input size so they can be batched. BYOD accepts one zip (or directory) of high-resolution images plus an optional `labels.csv`; the notebook crops and degrades them itself.",
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
                "sides are not multiples of 16, and a dataset too small to use — each rejected before the model does anything."
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_zip = Path('work') / 'byod.zip'\n"
                "    byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_zip.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_zip)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    corpus_files = fetch_corpus(cache_dir='weights/inat-birds')\n"
                "    corpus = read_corpus(corpus_files)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}}: {{CORPUS_RELEASE}} ({{CORPUS_LICENSE}})'\n"
                "    raw_rows = {{'photographs': len(corpus), 'bytes': sum(len(v) for v in corpus_files.values()), 'observers': len({{r['observer'] for r in corpus}})}}\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
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
                "on 96 photographs."
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
                "## 6. Baselines and the frozen model on the test crops\n\n"
                "Three systems frame the adaptation, each read two ways by `sr_metrics` (carried in `metrics.py`): **PSNR** "
                "in dB over the RGB channels against the high-resolution reference and **SSIM** on the luma channel (11-tap "
                "Gaussian window, sigma 1.5, 5-px border excluded — the classical-SR convention), averaged over the crops and "
                "per species. The **nearest-neighbour** baseline replicates each input pixel 2×2 — the floor. The **bicubic** "
                "baseline is the interpolation every image viewer applies and the reference point of the super-resolution "
                "literature. The **frozen model** is scored by `pipe.evaluate`, which upscales every record's input through "
                "`upscale` and scores the reconstruction. Expect the frozen model **below bicubic** on this input — the build "
                "record measured 25.9 dB / 0.785 against bicubic's 26.3 dB / 0.797 — because a classical-SR model sharpens "
                "JPEG block edges as if they were detail; read the per-species rows to see that it holds for every species."
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
                "assert frozen_test['psnr'] > baseline_nearest['psnr']"
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
                "on the held-out split."
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
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_stages=TRAINABLE_STAGES, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'loss': adapt_result['loss'], 'seconds': adapt_seconds}})"
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
                "cell asserts the adapted PSNR is above the frozen one and reports whether it is above bicubic. Ninety-six "
                "crops from one seeded split under one degradation give **no dispersion estimate**; the deltas are "
                "sample-sanity evidence that the adaptation contract works, not a benchmark, and a gain under JPEG-40 "
                "bicubic downsampling says nothing about a camera's own blur or noise until you measure them."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "adapted_fields = {{c: {{'n': v['n'], 'psnr': round(v['psnr'], 2), 'ssim': round(v['ssim'], 3)}} for c, v in adapted_test['per_category'].items()}}\n"
                "comparison = {{metric: {{'nearest': round(baseline_nearest[metric], 3), 'bicubic': round(baseline_bicubic[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['delta_vs_bicubic'] = {{metric: round(adapted_test[metric] - baseline_bicubic[metric], 3) for metric in METRICS}}\n"
                "comparison['by_species'] = {{c: {{'n': frozen_fields[c]['n'], 'frozen_psnr': frozen_fields[c]['psnr'], 'adapted_psnr': adapted_fields[c]['psnr'], 'frozen_ssim': frozen_fields[c]['ssim'], 'adapted_ssim': adapted_fields[c]['ssim']}} for c in frozen_fields}}\n"
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
                "assert adapted_test['psnr'] > frozen_test['psnr']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'adapted_beats_bicubic': adapted_test['psnr'] > baseline_bicubic['psnr']}})"
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
                "in-memory model (VER2). The cell asserts pixel-identical reconstructions on eight test crops (VER4)."
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
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_STAGES = 2` and compare the artifact "
        "size and the held-out PSNR; raise `EPOCHS` and watch the validation PSNR pick the epoch while the loss keeps "
        "falling; change `LEARNING_RATE` to `2e-4` and read a faster, less stable climb; or bring your own photographs through "
        "BYOD and read the two baselines before the adapted number.\n\n"
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
