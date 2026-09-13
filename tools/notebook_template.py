"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "swin2sr_super_resolution_pipeline",
    "repo_name": "swin2sr-super-resolution-pipeline",
    "stem": "swin2sr_super_resolution",
    "notebook_name": "swin2sr_super_resolution_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "Swin2SRPipeline",
    "weights_key": "swin2sr-x2-64",
    "runtime_imports": ["torch", "transformers"],
    "title": "Swin2SR classical-SR x2 — DIMER super-resolution tutorial (standalone)",
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
    "capability": "2× single-image super-resolution (classical SR) using the pinned `caidas/swin2SR-classical-sr-x2-64` weights",
    "intro": (
        "The model reconstructs a 2× larger image from one low-resolution RGB input: plausible detail is **synthesised** "
        "from what the input contains, not recovered from anywhere else. **No adaptation occurs:** no training, "
        "fine-tuning, in-context conditioning, or preprocessing fitting happens in this notebook — the pinned "
        "checkpoint is used as published, and the carried pipeline module adds snapshot verification, input validation, "
        "the output contract (a uint8 RGB array of shape `(2H, 2W, 3)`), and the `psnr`, `validate_inputs` and "
        "`evaluation_report` helpers. The default sample is a synthetic image built in code together with its own "
        "high-resolution reference; the numbers it produces are demonstration (plumbing) evidence, not a "
        "production-quality or benchmark claim."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, build a synthetic high-resolution reference and its low-resolution input, "
        "validate the input into an input manifest, run the supported task, read PSNR correctly against a plain bicubic "
        "baseline and understand why a self-made reference is not a benchmark, exercise an optional BYOD path where no "
        "reference exists and the report is `not-measurable`, and export the upscaled PNG plus machine-readable "
        "outputs and provenance."
    ),
    "exclusions": (
        "scales other than 2×, compressed-image or real-world (blind) restoration, denoising, artefact removal, "
        "video, face restoration, or any training. Inputs above 512 px per side are refused — tile them yourself."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. Swin2SR runs windowed attention over every input pixel, so time and memory grow with input **area**: the model card's CPU pass measured 160 s for a 1024×1024 input against 34 s for 512×512, which is why the ceiling is 512 px per side. The default 64×48 input is a fraction of a second either way. The pinned `torch==2.14.0` install and the 48 MB checkpoint are the largest downloads of the run.",
        "- **Knowledge:** basic Python, NumPy and PIL image handling; what PSNR measures and why it is not a perceptual quality score.",
        "- **Data:** the default sample is a deterministic 128×96 RGB image built in code and bicubic-downscaled to 64×48, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image file decodable by Pillow (PNG/JPEG/WebP and similar), any colour mode, both sides between 8 px and 512 px. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Build the synthetic sample and its reference, or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own reference: a deterministic 128×96 RGB image is "
                "built in code (horizontal colour gradient, one filled square, one diagonal stripe — sharp edges are "
                "what a super-resolver has to reconstruct), kept as the **high-resolution reference**, and "
                "bicubic-downscaled to 64×48 to become the model input. Both digests are printed. This is the same kind "
                "of gradient-and-square input the repository's smoke run used. Because you made the reference yourself, "
                "PSNR against it later is a self-consistency check of the input contract and forward pass, **not a "
                "benchmark**: real benchmarks use fixed public HR/LR pairs and a specified degradation kernel. BYOD is "
                "optional and disabled by default; when enabled, upload one image and it is upscaled as-is with no "
                "reference, so the evaluation report is `not-measurable` for it."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "HR_WIDTH, HR_HEIGHT = 128, 96\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    lr_image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    lr_image.load()\n"
                "    hr_reference = None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic synthetic high-resolution reference: no randomness, so no seed is needed.\n"
                "    ramp = np.linspace(0.0, 255.0, HR_WIDTH)\n"
                "    red = np.tile(ramp, (HR_HEIGHT, 1))\n"
                "    green = np.tile(np.linspace(255.0, 0.0, HR_HEIGHT), (HR_WIDTH, 1)).T\n"
                "    blue = np.full((HR_HEIGHT, HR_WIDTH), 96.0)\n"
                "    hr_reference = Image.fromarray(np.rint(np.stack([red, green, blue], axis=-1)).astype(np.uint8), mode='RGB')\n"
                "    draw = ImageDraw.Draw(hr_reference)\n"
                "    draw.rectangle([24, 20, 60, 56], fill=(20, 20, 20))\n"
                "    draw.line([(70, 84), (118, 12)], fill=(250, 250, 250), width=5)\n"
                "    # The model input is the reference bicubic-downscaled by exactly UPSCALE; the kernel choice is part of the sample.\n"
                "    lr_image = hr_reference.resize((HR_WIDTH // UPSCALE, HR_HEIGHT // UPSCALE), Image.Resampling.BICUBIC)\n"
                "    image_name = f'synthetic_shapes_{{HR_WIDTH // UPSCALE}}x{{HR_HEIGHT // UPSCALE}}.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "lr_sha256 = hashlib.sha256(np.asarray(lr_image.convert('RGB')).tobytes()).hexdigest()\n"
                "hr_sha256 = None if hr_reference is None else hashlib.sha256(np.asarray(hr_reference).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'input_size': lr_image.size, 'input_rgb_sha256': lr_sha256, 'reference_size': None if hr_reference is None else hr_reference.size, 'reference_rgb_sha256': hr_sha256}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the input → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: each image is routed through the same "
                "`validate_image` that `upscale` itself calls, so the checks — PIL type, both sides within "
                "`MIN_INPUT_SIDE`..`MAX_INPUT_SIDE` — cannot diverge between the two. It returns an **input manifest** "
                "naming the schema and ceilings, each input's identifier, observed mode, size and the output size it "
                "will produce, and the verdict, written to `outputs/{stem}_input_manifest.json`. The ceilings are "
                "printed first: `UPSCALE` (2), `MIN_INPUT_SIDE` (8 px; the processor pads to a multiple of the 8-px "
                "attention window) and `MAX_INPUT_SIDE` (512 px). The 512 px ceiling was lowered from 1024 px by the "
                "repository owner on 2026-09-12 because the card-pass smoke measured 160 s for a 1024×1024 input on CPU "
                "against 34 s for 512×512, and DIMER validators must stay responsive; larger images must be tiled by "
                "the caller. To show what rejection looks like, the cell also validates a deliberately oversized image "
                "and records the pipeline's own error message as a finding. Inside the pipeline the image is converted "
                "to RGB and padded to a multiple of 8, and the padding is cropped off at output scale; nothing else is "
                "dropped or altered."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'UPSCALE': UPSCALE, 'MIN_INPUT_SIDE': MIN_INPUT_SIDE, 'MAX_INPUT_SIDE': MAX_INPUT_SIDE}}}})\n"
                "input_manifest = validate_inputs(lr_image, names=[image_name])\n"
                "# Demonstrate rejection on an input that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(Image.new('RGB', (MAX_INPUT_SIDE + 1, MIN_INPUT_SIDE)))\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'oversized-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Upscale\n\n"
                "`upscale` returns a dict with `image` (uint8 array of shape `(2H, 2W, 3)`), `scale` (2), `input_size` "
                "and `output_size` as `(width, height)`, and the model identity. The output is a deterministic "
                "reconstruction — no sampling, no seed, `torch.inference_mode` — so repeated runs on the same device and "
                "dtype give the same bytes; CPU versus CUDA kernels can differ in the last rounding step. Look for "
                "`output_size` equal to twice `input_size`."
            ),
            "code": (
                "import time\n\n"
                "started = time.perf_counter()\n"
                "result = pipe.upscale(lr_image)\n"
                "elapsed = time.perf_counter() - started\n"
                "sr_array = result['image']\n"
                "print({{'scale': result['scale'], 'input_size': result['input_size'], 'output_size': result['output_size'], 'array_shape': sr_array.shape, 'dtype': str(sr_array.dtype), 'seconds': round(elapsed, 3), 'device': pipe.device}})"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. The "
                "repository's only metric helper is `psnr(pred, ref)`: peak signal-to-noise ratio in dB between two "
                "uint8 RGB arrays of identical shape (`10·log10(255² / MSE)`, `inf` when identical). It applies only "
                "when a high-resolution reference exists — on the synthetic default path, against the reference you "
                "generated in Section 4, so the verdict is `sample-sanity`; on BYOD none exists and the verdict is "
                "`not-measurable`, with the report naming what would make the task measurable. As a **meaningful "
                "baseline** the report also carries the same PSNR for a plain bicubic 2× resize of the input, so you can "
                "see whether the model beats interpolation on this one image. Both numbers are single-image tutorial "
                "evidence with no dispersion estimate; they depend on the downscaling kernel you chose and say nothing "
                "about photographs, compression artefacts, or the Set5/Set14/DIV2K figures reported by the upstream "
                "paper (not measured here). The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, hr_reference, low_resolution=lr_image, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No high-resolution reference exists for this input, so psnr is not computed; inspect the exported PNG instead.')\n"
                "else:\n"
                "    print({{'model_psnr_db': report['metrics'][0]['value'], 'bicubic_baseline_db': report['baselines'][0]['value'], 'note': 'single synthetic image, self-made reference; sanity evidence, not a benchmark'}})"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "The upscaled image is written as PNG (`outputs/{stem}_output.png`) — the actual artifact a downstream "
                "consumer wants — and machine-readable JSON preserves the sizes and timing, the sample identity and "
                "digests, the input manifest, the evaluation report, the notebook's source (repository, revision, "
                "embedded module digest, generator), the model identifier, the immutable model revision, the model "
                "licence, and the runtime identity (Python, `torch`, `transformers`, device). No credentials are "
                "recorded."
            ),
            "code": (
                "Image.fromarray(sr_array, mode='RGB').save('outputs/{stem}_output.png')\n"
                "payload = {{\n"
                "    'prediction': {{key: value for key, value in result.items() if key != 'image'}},\n"
                "    'output_file': 'outputs/{stem}_output.png',\n"
                "    'output_rgb_sha256': hashlib.sha256(sr_array.tobytes()).hexdigest(),\n"
                "    'seconds': round(elapsed, 3),\n"
                "    'input_manifest': input_manifest,\n"
                "    'evaluation_report': report,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'input_size': list(lr_image.size), 'input_rgb_sha256': lr_sha256, 'reference_rgb_sha256': hr_sha256}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The upscaled image is a learned reconstruction: plausible detail synthesised from the low-resolution input, "
        "not recovered ground truth. The PSNR values shown on the synthetic path compare the model and a bicubic "
        "baseline against a reference you generated and downscaled yourself, so they measure how well each inverts "
        "*that* bicubic downscaling on one synthetic image; they are not comparable to published Set5/Set14/DIV2K "
        "numbers and must not be generalised to photographs, other kernels, compressed inputs, or other scales. On a "
        "BYOD image no reference exists and the evaluation report says `not-measurable`. Inputs above 512 px per side "
        "are refused (tile them), the model handles only 2×, and content that was never in the input cannot be "
        "recovered. The pipeline provides no denoising, artefact removal, blind restoration, video, or training "
        "capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model, validate the demonstrated input, execute the public pipeline "
        "path, and emit the shown machine-readable outputs in the tested runtime — without the repository being "
        "reachable. It does **not** establish benchmark superiority, deployment calibration, safety for "
        "high-consequence decisions, or production fitness on an unseen domain.\n\n"
        "**Next experiments:** enable `USE_BYOD` with a small photograph (≤ 512 px per side) and inspect the exported "
        "PNG next to a bicubic resize; regenerate the synthetic reference with `Image.Resampling.NEAREST` or `BOX` "
        "downscaling and watch how much the model PSNR moves with the kernel alone; time `upscale` on 128×128, 256×256 "
        "and 512×512 inputs on your runtime to see the area scaling that motivated the ceiling.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/swin2sr-super-resolution-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/swin2sr-super-resolution-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/swin2sr-super-resolution-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/mv-lab/swin2sr\n"
        "- Swin2SR paper: https://arxiv.org/abs/2209.11345\n"
        "- Transformers `Swin2SR` documentation: https://huggingface.co/docs/transformers/model_doc/swin2sr"
    ),
}
