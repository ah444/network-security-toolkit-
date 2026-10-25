"""Step N3: running nmap. A small Python program stands in for nmap: the XML is read as it
comes, with its progress; a stop, Ctrl+C, the time limit and too much XML end the program;
nmap's failures are told in plain words."""

import io
import subprocess
import sys
import threading
import time

import pytest

from toolkit.core.data.recon import nmap_run
from toolkit.core.data.recon.nmap import NEEDS_PRIVILEGES
from toolkit.core.data.recon.nmap_run import NmapFailed
from toolkit.core.data.recon.scanner import ScanStopped

FAKE_NMAP = r'''
import sys, time
mode = sys.argv[1]
out = sys.stdout.buffer
def write(text):
    out.write(text.encode()); out.flush()
if mode == "ok":
    write('<?xml version="1.0"?>\n<nmaprun>\n')
    write('<taskprogress task="Connect Scan" time="1" percent="50.00" remaining="1"/>\n')
    time.sleep(0.3)
    write('<taskprogress task="Service scan" time="2" percent="33.33" remaining="1"/>\n')
    write('<taskprogress task="Service scan" time="3" percent="')  # a line in two pieces
    time.sleep(0.3)
    write('100.00" remaining="0"/>\n</nmaprun>\n')
elif mode == "sleep":
    write('<taskprogress task="Connect Scan" time="1" percent="1.00" remaining="9"/>\n')
    time.sleep(60)
elif mode == "flood":
    while True:
        write("x" * 65536)
elif mode == "deaf":  # ends only when killed
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    write("ready\n")
    time.sleep(60)
elif mode == "root":
    sys.stderr.write("You requested a scan type which requires root privileges.\nQUITTING!\n")
    sys.exit(1)
elif mode == "odd":
    sys.stderr.write("Starting\n\x1b[31mbad things\x1b[0m happened\nQUITTING!\n")
    sys.exit(2)
'''


@pytest.fixture
def fake(tmp_path):
    """The command of the stand-in nmap in ``mode``."""
    program = tmp_path / "fake_nmap.py"
    program.write_text(FAKE_NMAP, encoding="utf-8")
    return lambda mode: [sys.executable, str(program), mode]


@pytest.fixture
def started(monkeypatch):
    """The processes ``run_nmap`` starts, and how."""
    processes = []

    class Recorded(subprocess.Popen):
        def __init__(self, args, **kwargs):
            super().__init__(args, **kwargs)
            processes.append((self, kwargs))

    monkeypatch.setattr(nmap_run.subprocess, "Popen", Recorded)
    return processes


def test_the_xml_and_its_progress(fake, started):
    seen = []
    xml = nmap_run.run_nmap(fake("ok"), lambda task, percent: seen.append((task, percent)))
    assert xml.startswith(b'<?xml version="1.0"?>') and xml.rstrip().endswith(b"</nmaprun>")
    assert seen == [("Connect Scan", 50.0), ("Service scan", 33.33), ("Service scan", 100.0)]
    ((process, kwargs),) = started
    assert process.args[0] == sys.executable, "the program by its full path"
    assert kwargs["cwd"] == nmap_run.Path(sys.executable).parent, "from its own folder"
    assert kwargs["stdin"] == subprocess.DEVNULL and not kwargs.get("shell")


def test_a_stop_ends_nmap(fake, started):
    stop = threading.Event()
    threading.Timer(0.5, stop.set).start()
    began = time.monotonic()
    with pytest.raises(ScanStopped):
        nmap_run.run_nmap(fake("sleep"), stop=stop)
    assert time.monotonic() - began < nmap_run.KILL_AFTER + 3
    assert started[0][0].poll() is not None, "nmap is not left running"


def test_ctrl_c_ends_nmap(fake, started):
    class CtrlC:
        def wait(self, seconds):
            time.sleep(0.3)
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        nmap_run.run_nmap(fake("sleep"), stop=CtrlC())
    assert started[0][0].poll() is not None


def test_the_time_limit_ends_nmap(fake, started):
    with pytest.raises(NmapFailed, match=r"still running after 1 s \(TOOLKIT_SCAN_MAX_MIN\)"):
        nmap_run.run_nmap(fake("sleep"), time_limit=1)
    assert started[0][0].poll() is not None


def test_too_much_xml_ends_nmap(fake, started, monkeypatch):
    monkeypatch.setattr(nmap_run, "MAX_XML_BYTES", 2 * 2**20)
    with pytest.raises(NmapFailed, match="more than 2 MB of XML"):
        nmap_run.run_nmap(fake("flood"))
    assert started[0][0].poll() is not None


@pytest.mark.skipif(sys.platform == "win32", reason="Windows' terminate is already a kill")
def test_nmap_that_ignores_the_request_is_killed(fake, started, monkeypatch):
    monkeypatch.setattr(nmap_run, "KILL_AFTER", 0.5)
    stop = threading.Event()
    threading.Timer(1, stop.set).start()
    with pytest.raises(ScanStopped):
        nmap_run.run_nmap(fake("deaf"), stop=stop)
    assert started[0][0].poll() is not None


def test_privileges_are_told_in_plain_words(fake):
    with pytest.raises(NmapFailed) as raised:
        nmap_run.run_nmap(fake("root"))
    assert str(raised.value) == NEEDS_PRIVILEGES


def test_another_failure_gives_its_code_and_cleaned_lines(fake):
    with pytest.raises(NmapFailed) as raised:
        nmap_run.run_nmap(fake("odd"))
    assert str(raised.value) == "nmap stopped with exit code 2: Starting; bad things happened"


def test_nmap_that_does_not_start(tmp_path):
    with pytest.raises(NmapFailed, match=r"nmap did not start \(FileNotFoundError\)"):
        nmap_run.run_nmap([str(tmp_path / "nmap.exe"), "-V"])


@pytest.mark.parametrize(("stderr", "says"), [
    (b"dnet: Failed to open device eth0\nQUITTING!\n", "could not use Npcap"),
    (b"WARNING: Could not import all necessary Npcap functions.\n", "could not use Npcap"),
    (b"Sorry, but Nmap was unable to create a raw socket\nrequires raw socket access\n",
     "send raw packets"),
    (b"", "exit code 1: no message"),
])
def test_explain(stderr, says):
    assert says in nmap_run.explain(1, stderr)


def test_progress_lines_split_across_reads():
    class Pieces(io.RawIOBase):
        def __init__(self, pieces):
            self.pieces = list(pieces)

        def read1(self, size=-1):
            return self.pieces.pop(0) if self.pieces else b""

    seen = []
    pieces = [b'<taskprogress task="UDP Sc', b'an" percent="12.5', b'0" remaining="3"/>\n<tas',
              b'kprogress task="NSE" percent="99.01"/>\n', b"tail without a new line"]
    reader = nmap_run._Reader(Pieces(pieces), 10**6,
                              nmap_run._progress_lines(lambda *event: seen.append(event)))
    reader.run()
    assert seen == [("UDP Scan", 12.5), ("NSE", 99.01)]
    assert bytes(reader.data) == b"".join(pieces)


def test_the_kept_output_is_capped():
    reader = nmap_run._Reader(io.BytesIO(b"x" * 5000), 1000)
    reader.run()
    assert reader.overflow and len(reader.data) <= 1000
