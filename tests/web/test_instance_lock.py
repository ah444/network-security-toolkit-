"""v5.3 step HR3b (RD4: IA4): one web server per instance. Its lock (``instance/web.lock``,
``msvcrt`` or ``fcntl``) refuses a second server with who holds it, is released by the system
when the process ends however it ends, and its holder alone clears the temporary files a
stopped export left. The command line takes no lock."""

import logging
import subprocess
import sys
import textwrap
import time

import pytest

from toolkit.core.data import publish
from toolkit.core.data.settings import WebConfig
from toolkit.web.controllers import server
from toolkit.web.data import instance_lock

URL = "http://127.0.0.1:5000/"


@pytest.fixture
def held():
    """The instance's lock, held by this test; released at its end."""
    lock = instance_lock.take(URL)
    yield lock
    lock.release()


def test_a_second_holder_is_refused_with_who_holds_it(held, tmp_path):
    with pytest.raises(instance_lock.InstanceBusy) as busy:
        instance_lock.take("http://127.0.0.1:5001/")
    assert busy.value.holder is not None and busy.value.holder.url == URL
    held.release()
    instance_lock.take("http://127.0.0.1:5001/").release()  # free again once released


def test_a_second_web_server_stops_with_the_message(held, monkeypatch, capsys):
    def never(*args):
        raise AssertionError("the port was bound by a second server")

    monkeypatch.setattr(server.net, "bind_ports", never)
    assert server.run_server(WebConfig(port=5099, open_browser=False)) == 2
    out = capsys.readouterr().out
    assert "Another toolkit web server is already using the data folder" in out
    assert f"at {URL})" in out and "TOOLKIT_INSTANCE_DIR" in out


HOLD = textwrap.dedent("""
    import os, sys, time
    from toolkit.web.data import instance_lock
    lock = instance_lock.take("http://127.0.0.1:5077/")  # kept: closing the file releases it
    print(os.getpid(), flush=True)  # the interpreter's own (a venv launcher may start it)
    if sys.argv[1] == "crash":
        os._exit(3)  # dies at once: nothing released by the program
    time.sleep(60)
""")


def _child(how):
    child = subprocess.Popen([sys.executable, "-c", HOLD, how], stdout=subprocess.PIPE,
                             text=True)
    pid = int(child.stdout.readline())
    return child, pid


def test_another_process_holds_it_until_it_is_killed():
    child, pid = _child("hold")
    try:
        with pytest.raises(instance_lock.InstanceBusy) as busy:
            instance_lock.take(URL)
        assert busy.value.holder == instance_lock.Holder(pid, "http://127.0.0.1:5077/")
    finally:
        child.kill()  # as the launcher menu's stop or a crash: no clean-up in the child
        child.wait(30)
    deadline = time.monotonic() + 10
    while True:
        try:
            instance_lock.take(URL).release()
            break
        except instance_lock.InstanceBusy:
            assert time.monotonic() < deadline, "the system never released the lock"
            time.sleep(0.05)


def test_a_process_that_dies_releases_it():
    child, _ = _child("crash")
    assert child.wait(30) == 3
    instance_lock.take(URL).release()


def test_an_unreadable_holder_still_refuses(held, monkeypatch, capsys):
    monkeypatch.setattr(instance_lock, "holder", lambda path: None)
    assert server.run_server(WebConfig(port=5099, open_browser=False)) == 2
    out = capsys.readouterr().out
    assert "already using the data folder" in out and "(process" not in out


def test_the_holder_clears_what_a_stopped_export_left(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("TOOLKIT_INSTANCE_DIR", str(tmp_path))
    for name in ("runs/a.json.1a2b3c4d.tmp", "bundles/b.bundle.deadbeef.tmp",
                 "runs/a.json", "bundles/notes.tmp", "uploads/c.csv.1a2b3c4d.tmp",
                 "scan_history/s.json.0a0b0c0d.tmp", "scan_history/s.json"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    caplog.set_level(logging.INFO, "toolkit.web.data.instance_lock")
    instance_lock.take(URL).release()
    left = sorted(str(p.relative_to(tmp_path)).replace("\\", "/") for p in tmp_path.rglob("*")
                  if p.is_file())
    assert left == ["bundles/notes.tmp", "runs/a.json", "scan_history/s.json",
                    "uploads/c.csv.1a2b3c4d.tmp", "web.lock"], \
        "only publish's own temporary files, in runs/, bundles/ and scan_history/ (F6e)"
    assert "Removed 3 temporary file(s)" in caplog.text


def test_only_publishs_temporary_names_are_leftovers():
    names = ("a.json.1a2b3c4d.tmp", "a.json.1A2B3C4D.tmp", "a.json.1a2b.tmp", "a.tmp",
             "a.json", "x.bundle.00000000.tmp")
    assert [n for n in names if publish.LEFTOVER.fullmatch(n)] == [
        "a.json.1a2b3c4d.tmp", "x.bundle.00000000.tmp"]
