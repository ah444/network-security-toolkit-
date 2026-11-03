"""F6a: a finished web scan is saved in the scan history like the CLI's, and its answer names
the record, or says why it was not saved. The network is faked
(``web_scan.fake_network``): nothing leaves this computer."""

import json
import time

import pytest

from toolkit.core.data.recon import scan_records
from toolkit.core.data.settings import ScanConfig
from toolkit.web.services import jobs, scan
from toolkit.web.services.workspace import Workspace
from web_scan import fake_network

LIMITS = ScanConfig()


@pytest.fixture
def net(monkeypatch):
    return fake_network(monkeypatch)


def _scan(origin=jobs.HERE):
    job = scan.start_scan(Workspace(), scan.ScanChoices(target="8.8.8.8", ports="22,8080",
                                                        authorized=True),
                          jobs.JobRunner(jobs.SCAN), LIMITS, origin)
    deadline = time.monotonic() + 30
    while job.state not in jobs.FINISHED:
        assert time.monotonic() < deadline, "the scan job didn't end"
        time.sleep(0.01)
    return job


def test_a_finished_web_scan_is_saved_and_its_answer_names_it(net):
    job = _scan()
    assert job.state == "done", job.error
    scan_id = job.result["saved"]["scan_id"]
    record = json.loads((scan_records.history_dir() / f"{scan_id}.json").read_text(
        encoding="utf-8"))
    assert (record["target"], record["ports"], record["source"]) == (
        "8.8.8.8", "22,8080", "this computer")
    assert record["entries"] == job.result["entries"]
    assert record["hosts"][0]["state"] == "scanned"


def test_a_scan_from_another_computer_keeps_its_address(net):
    job = _scan(jobs.Origin("192.168.56.20", "unlock-1"))
    scan_id = job.result["saved"]["scan_id"]
    record = json.loads((scan_records.history_dir() / f"{scan_id}.json").read_text(
        encoding="utf-8"))
    assert record["source"] == "192.168.56.20"


def test_a_full_disk_keeps_the_answer_and_says_why(net, monkeypatch):
    monkeypatch.setenv("TOOLKIT_MIN_FREE_MB", str(10 ** 12))
    job = _scan()
    assert job.state == "done" and job.result["entries"]
    assert set(job.result["saved"]) == {"error"} and job.result["saved"]["error"]
    assert list(scan_records.history_dir().glob("*.json")) == []
