"""Step N1: the scan goes through a ``ScanEngine``; the built-in one is the toolkit's own
scanner, unchanged, and it can now be stopped (for the web scan's Cancel, step 2.4)."""

import threading
import time

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
    with pytest.raises(ValueError, match="Unknown scan engine 'nmapp'; use one of builtin"):
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
