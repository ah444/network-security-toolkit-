"""v5.3 F2f: ``toolkit ids --runs`` and ``--compare A B``. They read the saved TrainingRun
records only (no model is opened) and print what the web page shows: Δ = B − A only when both
runs scored the same test rows, otherwise the numbers side by side and why."""

import copy
import json

import pytest

from toolkit.cli.ui import runs as runs_screen
from toolkit.cli.ui.runs import REASONS
from toolkit.core.data.ids.run_files import runs_dir
from web_jobs import train_and_wait

CIC = "cic_ddos_sample.csv"
TARGET = "Label"
OTHER_ID = "20261030T120000Z-0000000a"


@pytest.fixture
def exported(client, upload, fixtures_dir):
    """A run trained and exported through the web routes: its record."""
    columns = upload(fixtures_dir / CIC).get_json()["columns"]
    trained = train_and_wait(client, {
        "features": [c for c in columns if c != TARGET], "target": TARGET, "balancing": "none",
        "test_size": 0.3, "models": ["decision_tree", "logistic_regression"]})
    assert trained.status_code == 200, trained.get_json()
    answer = client.post("/api/v1/run/export", json={"name": "cli"}).get_json()
    return json.loads((runs_dir() / f"{answer['run_id']}.json").read_text(encoding="utf-8"))


@pytest.fixture
def no_model(cli_module, monkeypatch):
    """Opening a bundle fails the test: the runs are records only."""
    def refuse(*_args, **_kwargs):
        raise AssertionError("a model was opened")
    monkeypatch.setattr(cli_module.MODEL_STORE, "load", refuse)
    monkeypatch.setattr(cli_module.MODEL_STORE, "list_bundles", refuse)


def _save(record, run_id):
    record = copy.deepcopy(record)
    record["run_id"] = run_id
    (runs_dir() / f"{run_id}.json").write_text(json.dumps(record), encoding="utf-8")
    return record


def _run(cli_module, capsys, *argv):
    code = cli_module.main(list(argv))
    return code, capsys.readouterr().out


def test_no_runs_yet(cli_module, capsys, no_model):
    code, out = _run(cli_module, capsys, "--runs")
    assert code == 0 and out.strip() == runs_screen.NO_RUNS


def test_a_run_is_listed(cli_module, capsys, exported, no_model):
    code, out = _run(cli_module, capsys, "--runs")
    assert code == 0
    line = out.splitlines()[0]
    assert line.startswith(f"1. {exported['run_id']}  ")
    assert f"{CIC} (200 rows)  label {TARGET}, stratified" in line
    assert " of 2: CV " in line and "Older runs" not in out


def test_the_same_run_twice_is_comparable(cli_module, capsys, exported, no_model):
    code, out = _run(cli_module, capsys, "--compare", exported["run_id"], exported["run_id"])
    assert code == 0
    assert out.startswith(f"A: {exported['run_id']}") and f"\nB: {exported['run_id']}" in out
    assert "✅ Both runs scored the same test rows" in out and "Δ (B − A)" in out
    assert "\nDecision Tree" in out and "\nLogistic Regression" in out
    for line in out.splitlines():
        if line.startswith("  Test macro-F1"):
            assert line.split()[-1] == "0.000"
    assert "Settings: the same features, models, balancing and feature selection." in out


def test_other_data_is_side_by_side_with_the_reasons(cli_module, capsys, exported, no_model):
    other = copy.deepcopy(exported)
    other["dataset"]["sha256"] = "cd" * 32
    other["evaluation"]["leaderboard"] = other["evaluation"]["leaderboard"][:1]
    other["settings"]["models"] = other["settings"]["models"][:1]
    _save(other, OTHER_ID)
    code, out = _run(cli_module, capsys, "--compare", exported["run_id"], OTHER_ID)
    assert code == 0
    assert "⚠️ These runs did not score the same test rows" in out
    assert "   - They were trained on different datasets (the files' SHA-256 differ)." in out
    assert "Δ" not in out.replace("Δ = B", "")
    dropped = exported["evaluation"]["leaderboard"][1]["title"]
    assert f"Only in A: {dropped}" in out, "by its title"
    assert "Settings that differ:\n  Models\n    A: " in out


def test_older_runs_page_by_page(cli_module, capsys, exported, no_model):
    ids = [f"20200101T00{n // 60:02d}{n % 60:02d}Z-{n:08x}" for n in range(1, 52)]  # older
    for run_id in ids:
        _save(exported, run_id)
    code, out = _run(cli_module, capsys, "--runs")
    lines = out.splitlines()
    assert code == 0 and len(lines) == 51 and lines[0].startswith(f"1. {exported['run_id']}")
    last = ids[::-1][48]  # the export and 49 older ones
    assert lines[-1] == f"Older runs: add --before {last}"
    code, out = _run(cli_module, capsys, "--runs", "--before", last)
    assert [line.split()[1] for line in out.splitlines()] == [ids[1], ids[0]]


def test_a_file_that_is_not_a_run_is_left_out(cli_module, capsys, exported, no_model):
    (runs_dir() / f"{OTHER_ID}.json").write_text("{not json", encoding="utf-8")
    code, out = _run(cli_module, capsys, "--runs")
    assert code == 0 and out.startswith(f"1. {exported['run_id']}")
    assert "⚠️ 1 file(s) in instance/runs are not runs this toolkit can show" in out
    code, out = _run(cli_module, capsys, "--compare", OTHER_ID, exported["run_id"])
    assert code == 2 and "not a TrainingRun record" in out and str(runs_dir()) not in out


@pytest.mark.parametrize("argv", [["--compare", "x", OTHER_ID],
                                  ["--compare", OTHER_ID, "../signing"],
                                  ["--runs", "--before", "..\\x"]])
def test_a_bad_run_id_is_refused(cli_module, capsys, argv, no_model):
    code, out = _run(cli_module, capsys, *argv)
    assert code == 2 and out.strip() == "❌ That is not a run id."


def test_a_missing_run(cli_module, capsys, no_model):
    code, out = _run(cli_module, capsys, "--compare", OTHER_ID, OTHER_ID)
    assert code == 2 and out.strip() == f"❌ There is no saved run {OTHER_ID}."


@pytest.mark.parametrize("argv", [["--before", OTHER_ID], ["--runs", "--list"],
                                  ["--runs", "--compare", OTHER_ID, OTHER_ID],
                                  ["--compare", OTHER_ID]])
def test_flags_that_dont_go_together(cli_module, capsys, argv):
    with pytest.raises(SystemExit) as stopped:
        cli_module.main(argv)
    assert stopped.value.code == 2


def test_numbers_as_the_page_writes_them():
    """The same text as the web page's charts.js and runsview.js (tests/web/test_charts.py,
    tests/web/test_runs_page.py)."""
    assert [runs_screen.seconds(v) for v in (0.0042, 0.5, 1, 9.99, 12.5, 4414.07)] == \
        ["0.0042 s", "0.5 s", "1 s", "10 s", "13 s", "4,414 s"]
    assert [runs_screen.difference(v, k) for v, k in ((0.012, "x"), (-0.004, "x"), (0, "x"),
                                                     (0.0002, "x"), (-0.0004, "x"),
                                                     (1.2, "fit_seconds"),
                                                     (-4414, "fit_seconds"))] == \
        ["+0.012", "−0.004", "0.000", "0.000", "0.000", "+1.2 s", "−4,414 s"]
    assert runs_screen.when("2026-10-30T12:05:09Z") == "2026-10-30 12:05 UTC"
    assert runs_screen.when("not a time") == "not a time"


def test_the_reasons_are_the_pages():
    """Each reason code has the same words in the CLI and on the page."""
    from web_js import js_sources
    page = js_sources()["runsview.js"]
    for code, words in REASONS.items():
        assert f'{code}: "{words}"' in page, code


# ---- v5.3 step HR4a: --delete RUN_ID [--with-bundle] ----

def _answer(monkeypatch, text):
    """What the user types at the question (None: the end of the input)."""
    def ask(_prompt=""):
        if text is None:
            raise EOFError
        return text
    monkeypatch.setattr("builtins.input", ask)


def _run_files(run_id):
    from toolkit.core.data.ids.bundle_files import bundles_dir
    return [runs_dir() / f"{run_id}.json", bundles_dir() / f"{run_id}.bundle"]


def test_delete_names_the_files_and_asks(cli_module, capsys, exported, no_model, monkeypatch):
    run_id = exported["run_id"]
    _answer(monkeypatch, "y")
    code, out = _run(cli_module, capsys, "--delete", run_id)
    assert code == 0
    assert f"{run_id}.json" in out and f"{run_id}.bundle" in out
    assert "kept: it becomes an unregistered bundle" in out
    record, bundle = _run_files(run_id)
    assert not record.exists() and bundle.exists()
    assert f"Deleted run {run_id}" in out


def test_delete_with_its_bundle(cli_module, capsys, exported, no_model, monkeypatch):
    run_id = exported["run_id"]
    _answer(monkeypatch, "yes")
    code, out = _run(cli_module, capsys, "--delete", run_id, "--with-bundle")
    assert code == 0 and not any(path.exists() for path in _run_files(run_id))


@pytest.mark.parametrize("typed", ["", "n", "no", None])
def test_anything_but_yes_deletes_nothing(cli_module, capsys, exported, no_model, monkeypatch,
                                          typed):
    _answer(monkeypatch, typed)
    code, out = _run(cli_module, capsys, "--delete", exported["run_id"], "--with-bundle")
    assert code == 0 and "Nothing deleted." in out
    assert all(path.exists() for path in _run_files(exported["run_id"]))


def test_delete_a_bad_or_missing_run(cli_module, capsys, no_model):
    code, out = _run(cli_module, capsys, "--delete", "../x")
    assert code == 2 and "That is not a run id." in capsys.readouterr().err + out
    code, out = _run(cli_module, capsys, "--delete", OTHER_ID)
    assert code == 2


def test_a_file_that_cant_be_removed(cli_module, capsys, exported, no_model, monkeypatch):
    from toolkit.core.services.ids import run_removal
    real = run_removal.remove_file

    def remove(path):
        if path.suffix == ".bundle":
            raise PermissionError(13, "used by another process", str(path))
        real(path)
    monkeypatch.setattr(run_removal, "remove_file", remove)
    _answer(monkeypatch, "y")
    code, _out = _run(cli_module, capsys, "--delete", exported["run_id"], "--with-bundle")
    assert code == 2
    assert all(path.exists() for path in _run_files(exported["run_id"]))


def test_with_bundle_goes_with_delete(cli_module):
    with pytest.raises(SystemExit) as stopped:
        cli_module.main(["--with-bundle"])
    assert stopped.value.code == 2


# ---- v5.3 step HR4b: --storage and --register RUN_ID ----

def test_storage_counts_and_names_the_leftovers(cli_module, capsys, exported, no_model,
                                                 monkeypatch):
    run_id = exported["run_id"]
    _answer(monkeypatch, "y")
    _run(cli_module, capsys, "--delete", run_id)
    code, out = _run(cli_module, capsys, "--storage")
    assert code == 0
    assert "Unregistered bundles" in out and f"  - {run_id}.bundle" in out
    assert "Total:" in out and "free on this disk" in out
    assert f"--register {run_id}" in out, "how to bring it back"


def test_register_brings_a_deleted_run_back(cli_module, capsys, exported, no_model,
                                            monkeypatch):
    run_id = exported["run_id"]
    record = (runs_dir() / f"{run_id}.json").read_bytes()
    _answer(monkeypatch, "y")
    _run(cli_module, capsys, "--delete", run_id)
    code, out = _run(cli_module, capsys, "--register", run_id)
    assert code == 0 and f"Run {run_id} is a saved run again" in out
    assert (runs_dir() / f"{run_id}.json").read_bytes() == record
    code, out = _run(cli_module, capsys, "--register", run_id)
    assert code == 2 and "already saved" in out


def test_register_a_bad_or_missing_bundle(cli_module, capsys, no_model):
    assert _run(cli_module, capsys, "--register", "../x")[0] == 2
    assert _run(cli_module, capsys, "--register", OTHER_ID)[0] == 2
