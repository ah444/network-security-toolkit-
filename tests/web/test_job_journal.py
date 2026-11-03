"""v5.3 step HR3c (RD4: IA4): the jobs journal. Each job's ask, start and end are a line in
``instance/jobs.journal``; at the next start the web server tells, in the terminal and once
on that kind's page in each session, every job the last one cut off, and what was lost. A
line cut off by a crash counts for nothing, and a journal that can't be written never stops
a job."""

import json
import threading
import time

import pytest

from toolkit.core.data.settings import WebConfig
from toolkit.web.controllers import server
from toolkit.web.data import instance_lock, journal
from toolkit.web.services import jobs
from toolkit.web.ui import console


@pytest.fixture
def path(tmp_path):
    return tmp_path / "jobs.journal"


def _job(kind=jobs.TRAINING, work=lambda job: {}, origin=jobs.HERE):
    return jobs.Job(kind=kind, work=work, on_end=lambda: None, max_seconds=60,
                    steps=lambda done, total: "", origin=origin)


def _wait(job):
    deadline = time.monotonic() + 30
    while job.state not in jobs.FINISHED:
        assert time.monotonic() < deadline
        time.sleep(0.01)


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_job_that_ends_leaves_nothing_to_tell(path):
    job = _job()
    jobs.JobRunner(jobs.TRAINING, journal=journal.JobJournal(path)).submit(job)
    _wait(job)
    assert [line["event"] for line in _lines(path)] == ["asked", "started", "ended"]
    assert _lines(path)[0]["source"] == "this computer" and _lines(path)[2]["state"] == "done"
    assert journal.recover(path) == []


def test_a_running_job_and_a_waiting_one_are_told(path):
    release = threading.Event()
    runner = jobs.JobRunner(jobs.SCAN, journal=journal.JobJournal(path))
    running = _job(jobs.SCAN, work=lambda job: release.wait(30) and {})
    waiting = _job(jobs.SCAN, origin=jobs.Origin("192.168.56.5", "unlock-1"))
    runner.submit(running)
    runner.submit(waiting)
    while running.state != jobs.RUNNING:
        time.sleep(0.01)
    try:  # as the next server reads it after this one stopped
        told = journal.recover(path)
    finally:
        release.set()
    assert [(job.kind, job.source, job.ran) for job in told] == [
        ("scan", "this computer", True), ("scan", "192.168.56.5", False)]
    assert told[0].asked <= told[1].asked <= time.time()
    assert path.read_bytes() == b"", "emptied: the next start tells nothing again"


def test_a_job_cancelled_while_waiting_has_ended(path):
    release = threading.Event()
    runner = jobs.JobRunner(jobs.TRAINING, journal=journal.JobJournal(path))
    first, second = _job(work=lambda job: release.wait(30) and {}), _job()
    runner.submit(first)
    runner.submit(second)
    runner.cancel(second)
    release.set()
    _wait(first)
    assert journal.recover(path) == []


def test_a_line_cut_off_by_a_crash_counts_for_nothing(path):
    path.write_text('{"job": "a", "event": "asked", "kind": "training", "source": "x", '
                    '"at": 1}\n{"job": "a", "event": "end', encoding="utf-8")
    assert [job.kind for job in journal.recover(path)] == ["training"], "a never ended"
    path.write_text('not json\n{"event": "asked"}\n[1, 2]\n', encoding="utf-8")
    assert journal.recover(path) == []


def test_a_journal_that_cant_be_written_never_stops_a_job(tmp_path, caplog):
    folder = tmp_path / "jobs.journal"
    folder.mkdir()  # a folder where the file should be
    job = _job()
    jobs.JobRunner(jobs.TRAINING, journal=journal.JobJournal(folder)).submit(job)
    _wait(job)
    assert job.state == jobs.DONE and "Could not write the jobs journal" in caplog.text


def test_the_lock_holder_reads_it_at_its_start():
    journal.JobJournal(journal.journal_path()).asked("job-1", "training", "this computer")
    lock = instance_lock.take("http://127.0.0.1:5000/")
    try:
        assert [(job.kind, job.ran) for job in lock.interrupted] == [("training", False)]
    finally:
        lock.release()


def test_the_terminal_says_it_as_the_server_starts(monkeypatch, capsys):
    journal.JobJournal(journal.journal_path()).asked("job-1", "training", "this computer")

    def in_use(hosts, port):
        raise OSError("in use")  # stops the start right after the journal is told

    monkeypatch.setattr(server.net, "bind_ports", in_use)
    assert server.run_server(WebConfig(port=5099, open_browser=False)) == 2
    out = capsys.readouterr().out
    assert "INTERRUPTED: a training asked for at " in out
    assert "from this computer was cut off before it started" in out
    assert "The data uploaded to its workspace is gone: upload it again." in out


def test_what_each_kind_says_was_lost():
    assert console.interrupted_line("training", "2026-11-03 14:02", "this computer", True) == (
        "INTERRUPTED: a training asked for at 2026-11-03 14:02 from this computer was cut off "
        "while it ran, when the web server stopped. The data uploaded to its workspace is "
        "gone: upload it again.")
    assert console.interrupted_line("scan", "2026-11-03 14:02", "192.168.56.5", False).endswith(
        "from 192.168.56.5 was cut off before it started, when the web server stopped. "
        "Nothing of it was kept: start it again.")


def test_each_page_tells_its_kind_once_per_session(web_app, client):
    web_app.extensions["toolkit"].interrupted = {
        "training": ["INTERRUPTED: a training <b>x</b>"], "scan": ["INTERRUPTED: a scan y"]}
    first = client.get("/dynamic_ids").get_data(as_text=True)
    assert "INTERRUPTED: a training &lt;b&gt;x&lt;/b&gt;" in first, "as text"
    assert "a scan y" not in first
    assert "INTERRUPTED" not in client.get("/dynamic_ids").get_data(as_text=True), "once"
    assert "INTERRUPTED: a scan y" in client.get("/scan").get_data(as_text=True)
    assert "INTERRUPTED" not in client.get("/scan").get_data(as_text=True)
    other = web_app.test_client()  # another session is told too
    assert "INTERRUPTED: a training" in other.get("/dynamic_ids").get_data(as_text=True)
