"""v5.3 step HR4c: the storage's limits. Nothing is ever deleted on their account.

- ``TOOLKIT_MIN_FREE_MB`` (default 1024): a new file (a record, an importance result, a
  bundle, a scan report) is refused when the disk would keep less free space than that; the
  refusal says so, and nothing is left behind.
- ``TOOLKIT_STORAGE_WARN_MB`` (default 20480): past that size, the storage report warns (the
  page, ``toolkit ids --storage`` and the web server's log at its start)."""

import json

import numpy as np
import pandas as pd
import pytest

from toolkit.cli.controllers import ids as cli_module
from toolkit.core.data import disk, publish
from toolkit.core.data.ids.run_files import runs_dir
from toolkit.core.data.recon import scan_files
from toolkit.core.data.settings import check_settings, instance_dir, min_free_mb, storage_warn_mb
from toolkit.core.services import storage
from toolkit.core.services.ids import saved_importance
from toolkit.core.services.ids.exports import write_run
from toolkit.core.services.ids.importance import permutation_importances
from toolkit.core.services.ids.runs import DatasetRef, TrainSettings, train
from toolkit.core.services.storage import NoSpace

MB = 2**20


@pytest.fixture
def free(monkeypatch):
    """The disk's free space, set by the test (in bytes)."""
    space = {"bytes": 100 * 1024 * MB}
    monkeypatch.setattr(disk, "free_bytes", lambda _path: space["bytes"])
    return space


@pytest.fixture(scope="module")
def run():
    rng = np.random.default_rng(0)
    label = rng.choice(["BENIGN", "DoS"], size=200)
    frame = pd.DataFrame({"a": (label == "DoS") * 2.0 + rng.normal(size=200),
                          "noise": rng.normal(size=200), "Label": label})
    settings = TrainSettings(label="Label", strategy="stratified", models=("decision_tree",),
                             workers=1)
    return train(frame, settings=settings,
                 source=DatasetRef(name="synthetic.csv", sha256="cd" * 32, rows=200, columns=3))


def test_the_defaults_and_the_settings(monkeypatch):
    assert (min_free_mb(), storage_warn_mb()) == (1024, 20480)
    monkeypatch.setenv("TOOLKIT_MIN_FREE_MB", "10")
    monkeypatch.setenv("TOOLKIT_STORAGE_WARN_MB", "500")
    assert (min_free_mb(), storage_warn_mb()) == (10, 500)
    for command in ("scan", "ids", "web"):
        monkeypatch.setenv("TOOLKIT_MIN_FREE_MB", "0")
        monkeypatch.setenv("TOOLKIT_STORAGE_WARN_MB", "lots")
        assert check_settings(command) == [
            "TOOLKIT_STORAGE_WARN_MB must be a whole number >= 1, not 'lots'",
            "TOOLKIT_MIN_FREE_MB must be a whole number >= 1, not '0'"], command


def test_a_new_file_under_the_floor_is_refused_and_leaves_nothing(free, tmp_path):
    data = b"x" * 1000
    free["bytes"] = 1024 * MB + len(data) - 1  # it would leave one byte less than the floor
    with pytest.raises(NoSpace) as refused:
        publish.publish_bytes(tmp_path / "runs" / "a.json", data)
    message = str(refused.value)
    assert message.startswith("Not saved: only 1,024 MB free on this disk")
    assert "TOOLKIT_MIN_FREE_MB" in message and "Nothing was deleted" in message
    assert list((tmp_path / "runs").iterdir()) == [], "no file, no temporary file"
    free["bytes"] += 1  # exactly the floor left: saved
    assert publish.publish_bytes(tmp_path / "runs" / "a.json", data).read_bytes() == data


def test_the_floor_is_the_setting(free, tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLKIT_MIN_FREE_MB", "5")
    free["bytes"] = 6 * MB
    publish.publish_bytes(tmp_path / "a.json", b"x")
    free["bytes"] = 4 * MB
    with pytest.raises(NoSpace, match="leave 5 MB free"):
        publish.publish_bytes(tmp_path / "b.json", b"x")


def test_a_record_and_an_importance_result_say_why_they_are_not_saved(free, run):
    write_run(run)
    found = permutation_importances(run.evaluation.training.models["decision_tree"],
                                    run.split.test, "Label", seed=run.evaluation.seed)
    free["bytes"] = 10 * MB
    reason = saved_importance.publish(run.run_id, "decision_tree", found)
    assert reason is not None and reason.startswith("Not saved: only 10 MB free")
    assert importance_path_absent(run)


def importance_path_absent(run):
    return not (runs_dir() / f"{run.run_id}.importance.decision_tree.json").exists()


def test_a_scan_report_is_refused_too(free, tmp_path):
    free["bytes"] = 10 * MB
    with pytest.raises(NoSpace):
        scan_files.export_results([], "json", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_the_report_warns_past_the_size_and_under_the_floor(free, monkeypatch):
    (instance_dir() / "logs").mkdir(parents=True)
    (instance_dir() / "logs" / "web.log").write_bytes(b"x" * (2 * MB))
    assert storage.storage_report()["warnings"] == []
    monkeypatch.setenv("TOOLKIT_STORAGE_WARN_MB", "1")
    free["bytes"] = 512 * MB
    report = storage.storage_report()
    assert (report["warn_bytes"], report["floor_bytes"]) == (MB, 1024 * MB)
    over, low = report["warnings"]
    assert over.startswith("The toolkit's files take 2.0 MB, more than the 1.0 MB set in "
                           "TOOLKIT_STORAGE_WARN_MB.")
    assert "Nothing is deleted on its own" in over
    assert low.startswith("Only 512.0 MB is free on this disk, under the 1.0 GB a new file "
                          "must leave (TOOLKIT_MIN_FREE_MB)")
    assert (instance_dir() / "logs" / "web.log").stat().st_size == 2 * MB, "nothing deleted"


def test_the_web_export_and_registering_say_no_space(free, client, upload, fixtures_dir):
    from web_jobs import train_and_wait
    columns = upload(fixtures_dir / "cic_ddos_sample.csv").get_json()["columns"]
    trained = train_and_wait(client, {
        "features": [c for c in columns if c != "Label"], "target": "Label",
        "balancing": "none", "test_size": 0.3, "models": ["decision_tree"]})
    run_id = trained.get_json()["run_id"]
    free["bytes"] = 10 * MB
    answer = client.post("/api/v1/run/export", json={"name": "x"})
    assert answer.status_code == 507
    error = answer.get_json()["error"]
    assert error["code"] == "NO_SPACE" and error["message"].startswith("Not saved: only 10 MB")
    assert not (runs_dir() / f"{run_id}.json").exists()
    registered = client.post(f"/api/v1/bundles/{run_id}/register")
    assert registered.status_code in (404, 507), "no bundle was saved, or no room for its record"


def test_the_cli_register_says_no_space(free, run, capsys):
    from toolkit.core.services.ids.exports import save_bundle
    save_bundle(run)
    free["bytes"] = 10 * MB
    assert cli_module.main(["--register", run.run_id]) == 2
    assert "Not saved: only 10 MB free" in capsys.readouterr().out
    assert not (runs_dir() / f"{run.run_id}.json").exists()


def test_storage_prints_the_warnings(free, capsys, monkeypatch):
    monkeypatch.setenv("TOOLKIT_STORAGE_WARN_MB", "1")
    (instance_dir() / "logs").mkdir(parents=True)
    (instance_dir() / "logs" / "web.log").write_bytes(b"x" * (2 * MB))
    assert cli_module.main(["--storage"]) == 0
    assert "⚠️ The toolkit's files take 2.0 MB" in capsys.readouterr().out


def test_json_numbers(free):
    """The report is JSON (the page reads it)."""
    json.dumps(storage.storage_report())


def test_the_scan_says_its_report_is_not_saved(monkeypatch, capsys):
    """The scan command: the refusal is a message and exit code 2, not a traceback."""
    from toolkit.cli.controllers import scan as scan_cli

    def full(*_args, **_kwargs):
        raise NoSpace("Not saved: only 10 MB free on this disk, ...")
    monkeypatch.setattr(scan_cli, "run_scan", full)
    assert scan_cli.main(["127.0.0.1", "--i-am-authorized", "-p", "80", "--no-nvd", "--no-ai",
                          "--no-abuseipdb", "--export", "json"]) == 2
    assert "Not saved: only 10 MB free" in capsys.readouterr().out
