"""Row 88 tutorial review (2026-10-17): pre-registered grid, fresh test, score-once rule, provenance, spec 2.5.

Stand-in evidence only: no model is loaded and no photograph is fetched. The notebook cells are executed with
stand-in pipelines so the score-once wiring is checked as the notebook runs it (SCORE_ONCE_RULE item 7)."""

# ruff: noqa: E501  -- stand-in records and expected strings are kept on single lines
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from swin2sr_super_resolution_pipeline import pipeline as pl
from swin2sr_super_resolution_pipeline import samples as sm

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "swin2sr_super_resolution_colab.ipynb"
DRAW = ROOT / "docs" / "fresh-test-draw.json"
STEM = "swin2sr_super_resolution"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code" and marker in c["source"]]
    assert len(found) == 1, (marker, len(found))
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


# --- R88-M4: the fresh-test pin ------------------------------------------------------------------------------


def test_fresh_test_pin_is_balanced_one_per_observer_and_disjoint_from_the_fleet_sample():
    fresh = sm.FRESH_TEST_RECORDS
    assert len(fresh) == 144
    assert {label: sum(r[1] == label for r in fresh) for label in sm.SPECIES} == dict.fromkeys(sm.SPECIES, 24)
    assert len({r[4] for r in fresh}) == 144, "one photograph per observer"
    assert len({r[2] for r in fresh}) == len({r[3] for r in fresh}) == len({r[6] for r in fresh}) == 144
    for i, key in ((2, "photo id"), (3, "observation id"), (4, "observer"), (6, "sha256")):
        assert not {r[i] for r in fresh} & {r[i] for r in sm.SAMPLE_RECORDS}, key
    assert all(len(r[6]) == 64 and r[5] > 0 and r[7].lower() in ("jpg", "jpeg", "png") for r in fresh)


def test_fresh_test_draw_record_matches_the_pin():
    draw = json.loads(DRAW.read_text(encoding="utf-8"))
    assert [tuple(r) for r in draw["records"]] == list(sm.FRESH_TEST_RECORDS)
    assert draw["per_species_drawn"] == dict.fromkeys(sm.SPECIES, 24)
    assert draw["user_agent"] == "dimer-fresh-test-draw/1.0"
    screen = draw["near_duplicate_screen"]
    assert screen["hamming_threshold"] == screen["synthetic_copy_hamming"]["max"], "the threshold comes from synthetic copies"


def test_check_fresh_disjoint_refuses_a_shared_observer():
    from PIL import Image

    a = {"id": "a", "hr": Image.new("RGB", (16, 16), (1, 2, 3)), "observer": "x", "inat_photo_id": 1, "inat_observation_url": "u1"}
    b = {"id": "b", "hr": Image.new("RGB", (16, 16), (4, 5, 6)), "observer": "x", "inat_photo_id": 2, "inat_observation_url": "u2"}
    with pytest.raises(ValueError, match="observer"):
        sm.check_fresh_disjoint([a], [b])
    assert sm.check_fresh_disjoint([a], [{**b, "observer": "y"}])["images_shared"] == 0


# --- R88-M1 / D2: the pre-registered grid rules --------------------------------------------------------------


def _result(name: str, curve: list[float]) -> dict:
    best = max(range(len(curve)), key=lambda i: curve[i])
    return {**sm.GRID[name], "best_epoch": best, "n_trainable": 1, "history": [{"epoch": i, "val": {"psnr": v}} for i, v in enumerate(curve)]}


@pytest.mark.parametrize(
    ("c0", "c1", "c2", "chosen", "model"),
    [
        (26.80, 26.89, 26.95, "C0", "C0"),  # within tolerance everywhere: keep C0
        (26.80, 26.90, 26.95, "C1", "C1"),  # C1 beats C0 by exactly 0.10
        (26.80, 26.95, 27.00, "C1", "C1"),  # C2 beats C0 but not the better one-stage config by 0.10
        (26.80, 26.85, 26.95, "C2", "C2"),  # C2 beats C0 and the better one-stage config (26.85) by 0.10
        (25.95, 25.96, 25.97, "C0", "frozen"),  # nothing beats the frozen model (25.90) by 0.10
    ],
)
def test_select_config_applies_the_preregistered_tolerances(c0, c1, c2, chosen, model):
    results = {"C0": _result("C0", [25.90, c0]), "C1": _result("C1", [25.90, c1]), "C2": _result("C2", [25.90, c2])}
    selection = sm.select_config(results)
    assert (selection["chosen"], selection["model"]) == (chosen, model), selection["reason"]
    assert selection["selection_split"].startswith("validation")


def test_select_config_refuses_a_config_without_a_validation_curve():
    bad = {"C0": {**sm.GRID["C0"], "best_epoch": 0, "history": [{"epoch": 0, "val": None}]}}
    with pytest.raises(ValueError, match="validation only"):
        sm.select_config(bad)


def test_preregistration_names_every_fixed_setting():
    keys = set(sm.PREREGISTERED_SETTINGS)
    assert {"data", "sample_digest", "split_seed", "split_per_species", "fresh_test_digest", "hr_crop", "jpeg_quality", "model_revision", "epochs", "batch_size", "adapt_seed", "grid", "grid_configs", "grid_margin_db", "selection_metric"} <= keys
    assert sm.PREREGISTERED_SETTINGS["grid_configs"] == {"C0": {"trainable_stages": 1, "lr": 1e-4}, "C1": {"trainable_stages": 1, "lr": 2e-4}, "C2": {"trainable_stages": 2, "lr": 1e-4}}
    assert sm.preregistration_differences(dict(sm.PREREGISTERED_SETTINGS)) == []
    assert sm.RECORDED_EVALUATION is None and sm.RECORDED_SELECTION is None, "set only from the first hosted pre-registered run"


def test_model_state_check():
    adapter = {"trainable_stages": 1, "lr": 1e-4, "epochs": 5, "batch_size": 8, "seed": 0, "n_train": 216, "n_val": 48}
    assert sm.model_state_check(adapter, "C0", sm.GRID["C0"], epochs=5, batch_size=8) == []
    assert sm.model_state_check({**adapter, "n_train": 200}, "C0", sm.GRID["C0"], epochs=5, batch_size=8)
    assert sm.model_state_check(adapter, "C1", sm.GRID["C1"], epochs=5, batch_size=8)
    assert sm.model_state_check(None, "frozen", None, epochs=5, batch_size=8) == []
    assert sm.model_state_check(adapter, "frozen", None, epochs=5, batch_size=8)


# --- R88-M3: score-once wiring, executed in the notebook cells with stand-ins -----------------------------------


def _scores(psnr: float) -> dict:
    rows = [{"id": f"r{i}", "category": "ab"[i % 2], "psnr": psnr + 0.01 * (i % 3), "ssim": 0.8} for i in range(8)]
    return {"psnr": psnr, "ssim": 0.8, "n": 8, "verdict": "measured-small-sample", "definitions": {}, "per_category": {"a": {"n": 4, "psnr": psnr, "ssim": 0.8}}, "per_record": rows}


class _Pipe:
    def __init__(self, device: str, calls: list) -> None:
        self.device, self.calls, self.adapter, self.saved = device, calls, None, {}
        self.fail_stages, self.gain = None, 0.2  # stand-in OOM for a config; validation gain per epoch

    def evaluate(self, records):
        self.calls.append(records[0]["split"])
        return _scores(26.0 if self.adapter is None else 27.0)

    def adapt(self, train, val, *, epochs, lr, batch_size, trainable_stages, seed=0, progress=None):
        if trainable_stages == self.fail_stages:
            raise MemoryError("stand-in CUDA out of memory")
        history = [{"epoch": e, "train_loss": None if e == 0 else 0.03, "val": {"psnr": 25.9 + self.gain * e * (1 + trainable_stages / 10), "ssim": 0.8}} for e in range(epochs + 1)]
        self.adapter = {"trainable_stages": trainable_stages, "lr": float(lr), "epochs": epochs, "batch_size": batch_size, "seed": seed, "n_train": len(train), "n_val": len(val), "best_epoch": epochs, "n_trainable": 10 * trainable_stages, "n_total": 100, "selection": "highest validation PSNR", "loss": "L1", "started_from": "pinned base", "history": history}
        return dict(self.adapter)

    def save_artifact(self, path, metadata=None):
        self.saved[str(path)] = dict(self.adapter)

    def load_artifact(self, path):
        self.adapter = dict(self.saved[str(path)])

    def restore_base(self):
        self.adapter = None
        return []


def _namespace(tmp_path: Path, device: str = "cuda:0") -> dict:
    ns = {k: getattr(pl, k) for k in dir(pl) if not k.startswith("__")}
    ns.update({k: getattr(sm, k) for k in dir(sm) if not k.startswith("__")})
    calls: list = []
    pipe = _Pipe(device, calls)

    class _Stub:
        @staticmethod
        def from_pretrained(weights_dir=None, device=None):
            return _Pipe(device, calls)

    split = {"train": 216, "validation": 48, "test": 96}
    splits = {name: [{"split": name}] * n for name, n in split.items()}
    ns.update({
        "__name__": "__main__", "json": json, "time": time, "shutil": shutil, "Path": Path, "os": __import__("os"), "pipe": pipe, "calls": calls,
        "Swin2SRPipeline": _Stub, "WEIGHTS_DIR": "w", "NOTEBOOK_SOURCE": {}, "require_valid_run": lambda where: None,
        "splits": splits, "train_records": splits["train"], "val_records": splits["validation"], "dev_test_records": splits["test"],
        "fresh_test_records": [{"split": "fresh"}] * 144, "data_label": "sample", "SPLIT_SEED": 42, "data_source": "stand-in", "disjoint": {},
        "dataset_digest": lambda records: sm.SAMPLE_DIGEST, "dataset_manifests": {"fresh_test": {"digest": sm.FRESH_TEST_DIGEST}, "train": {"digest": "t"}},
        "nearest_baseline": lambda r: {**_scores(25.0), "baseline": "nearest"}, "bicubic_baseline": lambda r: {**_scores(26.3), "baseline": "bicubic"},
        "run_history": [], "SECTION4_PASS": 1, "data_fingerprint": "fp-1", "hashlib": hashlib,
    })
    (tmp_path / "outputs").mkdir(exist_ok=True)
    return ns


def _run(notebook: dict, ns: dict, **fields) -> None:
    s7 = _cell(notebook, "RUN_GRID = 'auto'  # @param")
    for name, value in fields.items():
        default = {"EPOCHS": "EPOCHS = 5", "BATCH_SIZE": "BATCH_SIZE = 8", "RUN_GRID": "RUN_GRID = 'auto'"}[name]
        assert default in s7
        s7 = s7.replace(default, f"{name} = {value!r}", 1)
    exec(_cell(notebook, "baseline_nearest = nearest_baseline("), ns)
    ns["calls"].clear()
    exec(s7, ns)
    exec(_cell(notebook, "adapted_test = pipe.evaluate(dev_test_records)"), ns)


def test_preregistered_run_scores_the_fresh_test_then_an_identical_run_is_a_reproduction(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = _namespace(tmp_path)
    _run(notebook, ns)
    assert ns["preregistered"] and ns["grid_selection"] is not None
    assert "fresh" in ns["calls"], "the pre-registered run scores the fresh test"
    first = json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    assert first["test_look"] == "pre-registered evaluation (this runtime)" and first["settings_differing"] == []
    _run(notebook, ns)
    assert ns["test_look"] == "reproduction"
    again = json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report_reproduction_1.json").read_text(encoding="utf-8"))
    assert again["test_look"] == "reproduction"
    assert json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8")) == first, "never overwritten"


@pytest.mark.parametrize(
    ("fields", "extra"),
    [
        ({"EPOCHS": 3}, {}),
        ({"BATCH_SIZE": 4}, {}),
        ({"RUN_GRID": "off"}, {}),
        ({}, {"SPLIT_SEED": 7}),
        ({}, {"data_label": "byod", "fresh_test_records": None}),
    ],
)
def test_a_changed_setting_never_scores_the_fresh_test(notebook, tmp_path, monkeypatch, fields, extra):
    monkeypatch.chdir(tmp_path)
    ns = _namespace(tmp_path)
    ns.update(extra)
    _run(notebook, ns, **fields)
    assert not ns["preregistered"] and ns["setting_differences"]
    assert "fresh" not in ns["calls"]
    report = json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report_exploratory_1.json").read_text(encoding="utf-8"))
    assert report["test_look"] == "not scored (exploratory)" and report["fresh_test"] is None and report["settings_differing"]
    assert not (tmp_path / "outputs" / f"{STEM}_evaluation_report.json").exists()


def test_a_changed_grid_learning_rate_is_exploratory(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = _namespace(tmp_path)
    ns["GRID"] = {**sm.GRID, "C1": {"trainable_stages": 1, "lr": 3e-4}}
    _run(notebook, ns)
    assert not ns["preregistered"] and "fresh" not in ns["calls"]


def test_cpu_default_runs_without_the_grid_and_is_exploratory(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = _namespace(tmp_path, device="cpu")
    _run(notebook, ns)
    assert ns["grid_on"] is False and ns["chosen_config"] == "C0" and not ns["preregistered"]
    assert "fresh" not in ns["calls"]


def test_a_wrong_model_state_makes_the_run_exploratory(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = _namespace(tmp_path)
    exec(_cell(notebook, "baseline_nearest = nearest_baseline("), ns)
    exec(_cell(notebook, "RUN_GRID = 'auto'  # @param"), ns)
    ns["pipe"].adapter["lr"] = 5e-4  # something changed the loaded adapter after selection
    ns["calls"].clear()
    exec(_cell(notebook, "adapted_test = pipe.evaluate(dev_test_records)"), ns)
    assert ns["state_problems"] and ns["test_look"] == "not scored (exploratory)" and "fresh" not in ns["calls"]


def test_section_10_is_exploratory_and_never_touches_the_fresh_test(notebook):
    s10 = _cell(notebook, "RUN_EXPERIMENT = False  # @param")
    assert "fresh_test_records" not in s10 and "evaluate(test_records)" not in s10
    assert "experiment_exploratory_" in s10 and "TEST_LOOKS[2]" in s10


# --- R88-M1/M2/m2/m3/m4: prose, spec 2.5 and DAT9 -------------------------------------------------------------


def test_prose_no_longer_claims_test_selection_or_a_sweep(notebook):
    text = json.dumps(notebook)
    for banned in ("smallest configuration that beat bicubic", "the held-out test, not the validation curve, decides", "never used for training or", "build sweep", "did not beat the default", "counter-examples", "Kaggle T4 run of 20 September", "about ten minutes", "scored once"):
        assert banned not in text, banned
    md = _markdown(notebook)
    assert "exposed development test" in md and "Flickr2K" in md and "DIV2K" in md and "913.5 s" in md


def test_spec_2_5_metadata_and_use_notice(notebook):
    dimer = notebook["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "code")
    assert any("**⚠️ Notice.**" in c["source"] for c in notebook["cells"][:first_code])
    assert "`DIMER_ISOLATED_ENV`" in _markdown(notebook)


def test_byod_refusal_stops_with_a_message_and_invalidates_the_run(notebook):
    s4 = _cell(notebook, "USE_BYOD = False  # @param")
    assert "raise RunStopped(f'BYOD refused, nothing was trained or scored: {exc}" in s4
    assert s4.rstrip().endswith("RUN_VALID = True") and "RUN_VALID = False" in s4
    assert getattr(sm.RunStopped("x"), "dimer_stop", False) is True


# --- R88-m1: the provenance stamp names a commit that contains the carried modules ----------------------------


def test_stamped_revision_contains_the_carried_modules(notebook):
    generated = notebook["metadata"]["dimer"]["generated_from"]
    revision = generated["revision"]
    try:
        texts = [subprocess.check_output(["git", "-C", str(ROOT), "show", f"{revision}:{rel}"]).decode("utf-8").replace("\r\n", "\n") for rel in generated["modules"]]
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git history with {revision[:12]} is not available in this checkout")
    assert hashlib.sha256("".join(texts).encode("utf-8")).hexdigest() == generated["module_sha256"]
    embedded = {c["metadata"]["dimer"]["embedded_module"]: c["metadata"]["dimer"]["module_sha256"] for c in notebook["cells"] if c["metadata"].get("dimer", {}).get("embedded_module")}
    for rel, text in zip(generated["modules"], texts, strict=True):
        key = next(k for k in embedded if k.endswith(rel.split("/")[-1]))
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == embedded[key], rel


# --- Round 2 (peer review F1, F3, F6) -------------------------------------------------------------------------


def _s8(notebook: dict, ns: dict) -> None:
    ns["calls"].clear()
    exec(_cell(notebook, "adapted_test = pipe.evaluate(dev_test_records)"), ns)


def test_f1_a_section_8_rerun_alone_is_a_reproduction_and_never_overwrites(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = _namespace(tmp_path)
    _run(notebook, ns)
    main = tmp_path / "outputs" / f"{STEM}_evaluation_report.json"
    first = main.read_bytes()
    assert ns["test_look"] == "pre-registered evaluation (this runtime)"
    _s8(notebook, ns)
    assert ns["test_look"] == "reproduction" and ns["suffix"] == "_reproduction_1"
    assert json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report_reproduction_1.json").read_text(encoding="utf-8"))["test_look"] == "reproduction"
    assert main.read_bytes() == first
    assert [h["test_look"] for h in ns["run_history"]] == ["pre-registered evaluation (this runtime)", "reproduction"]


def test_f1_b_c2_oom_on_a_rerun_leaves_an_incomplete_grid_that_never_scores_the_fresh_test(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = _namespace(tmp_path)
    _run(notebook, ns)
    ns["pipe"].fail_stages = 2
    with pytest.raises(sm.RunStopped, match="Grid incomplete: C2 ran out of memory"):
        exec(_cell(notebook, "RUN_GRID = 'auto'  # @param"), ns)
    grid = json.loads((tmp_path / "outputs" / f"{STEM}_grid_selection_reproduction_1.json").read_text(encoding="utf-8"))
    assert grid["complete"] is False and grid["status"]["C2"]["status"] == "out of memory"
    assert grid["status"]["C0"]["status"] == grid["status"]["C1"]["status"] == "completed", "write-immediately keeps finished configs"
    _s8(notebook, ns)
    assert not ns["preregistered"] and ns["test_look"] == "not scored (exploratory)" and "fresh" not in ns["calls"]
    assert ns["gate_problems"]


def test_f1_c_c2_oom_in_a_fresh_runtime_stops_cleanly_and_section_8_is_exploratory(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = _namespace(tmp_path)
    ns["pipe"].fail_stages = 2
    exec(_cell(notebook, "baseline_nearest = nearest_baseline("), ns)
    with pytest.raises(sm.RunStopped, match="Grid incomplete"):
        exec(_cell(notebook, "RUN_GRID = 'auto'  # @param"), ns)
    _s8(notebook, ns)
    assert ns["test_look"] == "not scored (exploratory)" and "fresh" not in ns["calls"]
    assert (tmp_path / "outputs" / f"{STEM}_evaluation_report_exploratory_1.json").exists()


def test_f1_d_section_4_rerun_with_another_split_then_section_8_is_exploratory(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = _namespace(tmp_path)
    ns["pipe"].gain = 0.001  # nothing beats the frozen model by 0.10 dB: the grid keeps the frozen model
    _run(notebook, ns)
    assert ns["selected_model"] == "frozen" and ns["preregistered"]
    first = (tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_bytes()
    # What Section 4 does on a re-run with SPLIT_SEED = 7: a new pass id, a new fingerprint, the base restored.
    ns.update({"SECTION4_PASS": 2, "SPLIT_SEED": 7, "data_fingerprint": "fp-2"})
    ns["pipe"].restore_base()
    _s8(notebook, ns)
    assert not ns["preregistered"] and "fresh" not in ns["calls"] and ns["test_look"] == "not scored (exploratory)"
    assert any("latest Section 4" in g for g in ns["gate_problems"])
    assert (tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_bytes() == first


def test_f1_section_4_resets_the_grid_state(notebook):
    s4 = _cell(notebook, "USE_BYOD = False  # @param")
    assert "SECTION4_PASS = globals().get('SECTION4_PASS', 0) + 1" in s4 and "data_fingerprint = hashlib.sha256(" in s4
    assert "'grid_completed'" in s4 and "globals().pop(_name, None)" in s4


def test_f3_reading_rule_needs_a_ci_above_zero_and_a_tenth_of_a_db():
    from swin2sr_super_resolution_pipeline import metrics as mt

    def rows(values):
        return {"per_record": [{"id": f"r{i}", "category": "ab"[i % 2], "psnr": v, "ssim": 0.8 + v / 1000} for i, v in enumerate(values)]}

    base = rows([26.0 + 0.01 * (i % 5) for i in range(48)])
    big = rows([26.3 + 0.01 * (i % 5) for i in range(48)])
    tiny = rows([26.05 + 0.01 * (i % 5) for i in range(48)])
    noisy = rows([26.0 + (1.0 if i % 2 else -0.8) for i in range(48)])
    assert mt.read_comparison(big, base)["verdict"] == "improved"
    assert mt.read_comparison(tiny, base)["verdict"] == "within tolerance", "a significant but < 0.10 dB gain is not 'improved'"
    assert mt.read_comparison(base, big)["verdict"] == "worse"
    r = mt.read_comparison(noisy, base)
    assert r["psnr"]["ci95"][0] <= r["psnr"]["mean_diff"] <= r["psnr"]["ci95"][1] and "ssim" in r
    assert mt.read_comparison(big, base) == mt.read_comparison(big, base), "seeded"
    assert mt.READ_SEED == 20261018 and mt.READ_N_BOOT == 10_000 and mt.READ_MARGIN_DB == 0.10


def test_f3_verdicts_use_the_reading_rule(notebook):
    s8 = _cell(notebook, "adapted_test = pipe.evaluate(dev_test_records)")
    assert "read_comparison(fresh_scores['selected'], fresh_scores['frozen'])" in s8 and "read_comparison(adapted_test, frozen_test)" in s8
    assert "> 0 else" not in s8, "no zero-threshold verdicts remain"


def test_frame_lists_replay_the_draw_record():
    import gzip

    frames = json.loads(gzip.decompress((ROOT / "docs" / "fresh-test-frames.json.gz").read_bytes()))["frames"]
    draw = json.loads(DRAW.read_text(encoding="utf-8"))
    for label, meta in draw["frame"].items():
        ids = ",".join(str(r["obs"]) for r in frames[label])
        assert hashlib.sha256(ids.encode()).hexdigest() == meta["sha256_of_ids"], label
