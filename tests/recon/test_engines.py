"""Step N1: the scan goes through a ``ScanEngine``; the built-in one is the toolkit's own
scanner, unchanged, and it can now be stopped (for the web scan's Cancel, step 2.4)."""

import threading
import time
from pathlib import Path

import pytest

from toolkit.core.data.recon import scanner as recon_scanner
from toolkit.core.data.recon.scanner import PortResult, ScanStopped
from toolkit.core.services import ports
from toolkit.core.services.recon import engines
from toolkit.core.services.recon import scan as scan_service
from toolkit.core.services.recon.intel import Intel


def test_the_builtin_engine_is_a_scan_engine():
    engine = engines.engine_named("builtin")
    assert isinstance(engine, engines.BuiltinEngine) and isinstance(engine, ports.ScanEngine)
    assert (engine.name, engine.available()) == ("builtin", None)
    assert engines.DEFAULT_ENGINE == "builtin" == scan_service.ScanRequest("x", [], [], []).engine


def test_an_unknown_engine_is_refused():
    with pytest.raises(ValueError, match="Unknown scan engine .nmapp.; use one of builtin, nmap"):
        engines.engine_named("nmapp")


def test_the_builtin_engine_is_the_scanner(tcp_server, free_port):
    port = tcp_server(b"SSH-2.0-OpenSSH_9.6\r\n")
    by_engine = engines.BuiltinEngine().scan("127.0.0.1", [port, free_port], ["tcp"])
    assert by_engine == recon_scanner.scan_ports("127.0.0.1", [port, free_port], ["tcp"])
    opened = next(r for r in by_engine if r.state == "open")
    assert (opened.service, opened.banner) == ("ssh", "SSH-2.0-OpenSSH_9.6")
    assert (opened.product, opened.version, opened.cpe, opened.reason, opened.scripts) == (
        "", "", (), "", ()), "the fields of an engine that names software stay empty"


def test_a_set_stop_runs_no_check(monkeypatch):
    checked = []
    monkeypatch.setattr(recon_scanner, "scan_tcp_port",
                        lambda host, port: checked.append(port) or PortResult(port, "tcp", "closed"))
    stop = threading.Event()
    stop.set()
    with pytest.raises(ScanStopped):
        engines.BuiltinEngine().scan("127.0.0.1", range(1, 101), ["tcp"], stop=stop)
    assert checked == []


def test_stop_ends_a_running_scan(monkeypatch):
    """Like Ctrl+C: the checks already running finish, the queued ones don't start."""
    started = []
    lock = threading.Lock()
    stop = threading.Event()

    def check(host, port):
        with lock:
            started.append(port)
            if len(started) == 20:
                stop.set()
        time.sleep(0.01)
        return PortResult(port, "tcp", "closed")

    class Sink:
        finished = False

        def start(self, total):
            pass

        def advance(self, done=1):
            pass

        def finish(self):
            Sink.finished = True

    monkeypatch.setattr(recon_scanner, "scan_tcp_port", check)
    with pytest.raises(ScanStopped):
        engines.BuiltinEngine().scan("127.0.0.1", range(1, 2001), ["tcp"], Sink(), stop)
    assert len(started) < 500 and Sink.finished, "the progress is finished after a stop too"


def test_no_stop_changes_nothing(monkeypatch):
    monkeypatch.setattr(recon_scanner, "scan_tcp_port",
                        lambda host, port: PortResult(port, "tcp", "closed"))
    assert len(engines.BuiltinEngine().scan("127.0.0.1", range(1, 51), ["tcp"],
                                            stop=threading.Event())) == 50


def test_the_scan_uses_the_engine_it_is_asked_for(monkeypatch):
    asked = []

    class Recording(engines.BuiltinEngine):
        name = "recording"

        def scan(self, host, port_list, protocols, progress=None, stop=None):
            asked.append((host, list(port_list), list(protocols)))
            return [PortResult(80, "tcp", "open", "http", "nginx")]

    monkeypatch.setitem(engines.ENGINES, "recording", Recording)
    monkeypatch.setattr(scan_service, "ping_host", lambda host: (True, 64))

    class View:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    request = scan_service.ScanRequest("10.0.0.5", ["10.0.0.5"], [80], ["tcp"],
                                       engine="recording")
    entries = scan_service.scan_target("10.0.0.5", request, Intel({}), View())
    assert asked == [("10.0.0.5", [80], ["tcp"])]
    assert [(e["port"], e["banner"]) for e in entries] == [(80, "nginx")]


# Step N3: the nmap engine

class Recorder:
    """A ProgressSink that records its calls."""

    def __init__(self):
        self.calls = []

    def start(self, total):
        self.calls.append(("start", total))

    def advance(self, done=1):
        self.calls.append(("advance", done))

    def finish(self):
        self.calls.append(("finish",))


@pytest.fixture
def fake_nmap(monkeypatch, fixtures_dir):
    """nmap found and run without running it: the commands, and the real XML of this PC."""
    from toolkit.core.data.recon.nmap import Nmap

    runs = []
    monkeypatch.setattr(engines, "find_nmap", lambda: Nmap(Path("/opt/nmap/nmap"), (7, 992)))
    monkeypatch.setattr(engines, "has_privileges", lambda: False)

    def run(command, progress, stop, time_limit):
        runs.append({"command": command, "time_limit": time_limit, "stop": stop})
        for task, percent in (("Connect Scan", 50.0), ("Unknown task", 90.0),
                              ("Service scan", 40.0), ("Connect Scan", 10.0)):
            progress(task, percent)
        return (fixtures_dir / "nmap_connect_scripts.xml").read_bytes()

    monkeypatch.setattr(engines, "run_nmap", run)
    return runs


def test_the_nmap_engine_is_named():
    assert isinstance(engines.engine_named("nmap"), engines.NmapEngine)
    assert isinstance(engines.NmapEngine(), ports.ScanEngine)


def test_the_nmap_engine_says_why_it_cant_run(monkeypatch):
    from toolkit.core.data.recon.nmap import NmapMissing

    def missing():
        raise NmapMissing("nmap was not found (...). To use it, install it")

    monkeypatch.setattr(engines, "find_nmap", missing)
    assert engines.NmapEngine().available().startswith("nmap was not found")
    monkeypatch.setattr(engines, "find_nmap", lambda: object())
    assert engines.NmapEngine().available() is None


def test_the_nmap_engine_scans(fake_nmap):
    sink = Recorder()
    engine = engines.NmapEngine()
    found = engine.scan("127.0.0.1", [2222, 3306, 1], ["tcp"], sink)
    assert [(r.port, r.state, r.service) for r in found] == [
        (1, "closed", ""), (2222, "open", "ssh"), (3306, "open", "mysql")]
    (run,) = fake_nmap
    assert run["command"][-4:] == ["60m", "-p", "T:1,2222,3306", "127.0.0.1"]
    assert "-sT" in run["command"] and "-sS" not in run["command"], "no privileges here"
    assert run["time_limit"] == 60 * 60 + engines.NMAP_GRACE
    assert sink.calls == [("start", 300), ("advance", 50), ("advance", 90), ("advance", 160),
                          ("finish",)], "one bar over 3 phases, never back, full at the end"
    assert dict(engine.host_scripts)["smb2-time"]


def test_the_nmap_engine_with_privileges_and_options(fake_nmap, monkeypatch):
    from toolkit.core.data.recon.nmap import NmapOptions

    monkeypatch.setattr(engines, "has_privileges", lambda: True)
    monkeypatch.setenv("TOOLKIT_SCAN_MAX_MIN", "5")
    engine = engines.NmapEngine(NmapOptions(service_detection="light", os_detect=True))
    engine.scan("127.0.0.1", [22], ["tcp", "udp"])
    command = fake_nmap[0]["command"]
    assert {"-sS", "-sU", "-O"} <= set(command) and "--unprivileged" not in command
    assert command[command.index("--version-intensity") + 1] == "2"
    assert command[command.index("--host-timeout") + 1] == "5m"
    assert fake_nmap[0]["time_limit"] == 5 * 60 + engines.NMAP_GRACE


def test_a_stopped_nmap_scan_finishes_the_progress(monkeypatch, fake_nmap):
    def stopped(command, progress, stop, time_limit):
        raise ScanStopped()

    monkeypatch.setattr(engines, "run_nmap", stopped)
    sink = Recorder()
    with pytest.raises(ScanStopped):
        engines.NmapEngine().scan("127.0.0.1", [22], ["tcp"], sink, threading.Event())
    assert sink.calls == [("start", 300), ("finish",)]


def test_udp_without_privileges_is_refused_before_nmap_runs(fake_nmap):
    with pytest.raises(ValueError, match="send raw packets"):
        engines.NmapEngine().scan("127.0.0.1", [53], ["udp"])
    assert fake_nmap == []


def test_nmap_progress_phases():
    sink = Recorder()
    bar = engines.NmapProgress(sink, ["tcp", "udp"])
    assert bar.total == 400
    for task, percent in (("UDP Scan", 30), ("SYN Stealth Scan", 100), ("NSE", 250),
                          ("Service scan", 1)):
        bar.report(task, percent)
    bar.complete()
    assert sink.calls == [("advance", 130), ("advance", 270)], "NSE capped at its 100"
