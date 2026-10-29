"""DAT20 (row 88): a refused BYOD input is one line, ``Refused: <file>: <reason>``; a pinned-asset failure is not a refusal.

Section 4 of the generated notebook runs verbatim with the carried module's namespace (the review-fix harness). A
refused input must raise ``RunStopped`` (``dimer_stop``: the router shows only its message, and its
``_render_traceback_`` does the same when routing is disabled), as exactly one line that starts with
``Refused: <file>: `` and names the fix. A FRESH_TEST_DIGEST mismatch of the pinned fresh test is a genuine
failure: it raises an ordinary error that keeps its traceback.
"""
# ruff: noqa: E501

from __future__ import annotations

import sys
import types

import pytest

import test_notebook_review_fixes as rf
from swin2sr_super_resolution_pipeline import samples as sm

notebook = rf.notebook  # the generated notebook, as a module-scoped fixture


def _refused(ns: dict, source: str) -> str:
    with pytest.raises(sm.RunStopped) as caught:
        exec(source, ns)  # noqa: S102
    exc = caught.value
    assert getattr(exc, "dimer_stop", False) is True
    lines = exc._render_traceback_()
    assert lines == [str(exc)] and "\n" not in lines[0] and "Traceback" not in lines[0]
    assert lines[0].startswith("Refused: ") and "Nothing was trained or scored" in lines[0]
    assert ns.get("RUN_VALID") is False  # Sections 6-10 refuse this run
    return lines[0]


def test_scripted_refusal_input_is_one_line_naming_the_file(notebook, tmp_path, monkeypatch):
    """The hosted refusal run sets BYOD_PATH='/etc/hostname': a text file that is neither a folder nor a zip."""
    monkeypatch.chdir(tmp_path)
    hostname = tmp_path / "hostname"
    hostname.write_text("colab-vm\n", encoding="utf-8")
    line = _refused(rf._section_4_namespace([]), rf._section_4(notebook, str(hostname)))
    assert line.startswith(f"Refused: hostname: {hostname} is neither a directory nor a zip file. "), line


@pytest.mark.parametrize(
    ("path", "uploaded", "prefix"),
    [
        ("nowhere.zip", None, "Refused: nowhere.zip: BYOD_PATH 'nowhere.zip' does not exist"),
        ("", {}, "Refused: (no file): Upload exactly one .zip file (received 0"),
        ("", {"a.zip": b"", "b.zip": b""}, "Refused: a.zip, b.zip: Upload exactly one .zip file (received 2"),
        ("", {"notes.txt": b"x"}, "Refused: notes.txt: upload one .zip"),
        ("", "no-colab", "Refused: (no file): USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab"),
    ],
    ids=["missing-path", "cancelled-upload", "two-uploads", "not-a-zip", "no-upload-dialog"],
)
def test_byod_refusals_are_one_line(notebook, tmp_path, monkeypatch, path, uploaded, prefix):
    monkeypatch.chdir(tmp_path)
    if uploaded == "no-colab":
        monkeypatch.setitem(sys.modules, "google", None)
    elif uploaded is not None:
        google, colab, files = (types.ModuleType(n) for n in ("google", "google.colab", "google.colab.files"))
        files.upload = lambda: uploaded
        colab.files, google.colab = files, colab
        for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
            monkeypatch.setitem(sys.modules, name, module)
    line = _refused(rf._section_4_namespace([]), rf._section_4(notebook, path))
    assert line.startswith(prefix), line


def test_a_corrupt_zip_is_refused_by_name(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    archive = rf._zip(tmp_path / "mine.zip", 1)
    data = bytearray(archive.read_bytes())
    data[60:90] = b"\x00" * 30  # damage the first member, keep the central directory
    archive.write_bytes(bytes(data))
    line = _refused(rf._section_4_namespace([]), rf._section_4(notebook, "mine.zip"))
    assert line.startswith("Refused: mine.zip: "), line


def test_fresh_test_digest_mismatch_is_a_genuine_error_with_its_traceback(notebook, tmp_path, monkeypatch):
    """The pinned fresh test failing FRESH_TEST_DIGEST is not the learner's input: it must not be shown as a refusal."""
    monkeypatch.chdir(tmp_path)
    records = sm.load_byod_dataset(rf._zip(tmp_path / "stand_in.zip", 14))
    ns = rf._section_4_namespace([])
    ns.update(
        fetch_corpus=lambda cache_dir: {},
        read_corpus=lambda files: [],
        build_sample_dataset=lambda corpus, seed: sm.split_dataset(records[:12], seed=seed),
        fetch_fresh_test=lambda cache_dir: records[12:],
    )
    source = rf._cell(notebook, "USE_BYOD = False")
    with pytest.raises(RuntimeError, match="do not match FRESH_TEST_DIGEST") as caught:
        exec(source, ns)  # noqa: S102
    assert not isinstance(caught.value, sm.RunStopped) and not getattr(caught.value, "dimer_stop", False)
    assert ns.get("RUN_VALID") is False


def test_run_stopped_renders_without_a_traceback_when_routing_is_disabled():
    assert sm.RunStopped("Refused: a.zip: reason")._render_traceback_() == ["Refused: a.zip: reason"]
