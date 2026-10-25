"""Step 1.4d: the web trainer starts properly (B16): the browser opens once the port
answers, a port in use is a message (and never shared, even on Windows), --port and
--no-browser, Ctrl+C stops cleanly. No test opens a real browser (see conftest.py)."""

import socket
import webbrowser

import pytest

from toolkit.controllers.cli import web as web_command
from toolkit.controllers.web import server as web
from toolkit.data.web import net
from toolkit.core.data.settings import WebConfig, port_number, web_config

HOST = "127.0.0.1"


def _free_port():
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


@pytest.fixture
def started(monkeypatch):
    """run_server with fake waitress and browser opener; records what they were given."""
    calls = {"serve": [], "browser": []}

    def fake_serve(app, **kw):
        calls["serve"].append({**kw, "address": kw["sockets"][0].getsockname()})

    import waitress
    monkeypatch.setattr(waitress, "serve", fake_serve)
    monkeypatch.setattr(web, "open_browser_when_ready",
                        lambda url, host, port: calls["browser"].append((url, host, port)))
    return calls


def test_port_number():
    assert port_number("80") == 80 and port_number(" 65535 ") == 65535
    for text in ("0", "65536", "-1", "abc", ""):
        with pytest.raises(ValueError, match="TOOLKIT_PORT must be a port number from 1 to "
                                             "65535"):
            port_number(text)
    with pytest.raises(ValueError, match="--port must be"):
        port_number("x", "--port")
    with pytest.raises(ValueError, match="TOOLKIT_PORT"):
        web_config({"TOOLKIT_PORT": "70000"})


def test_no_browser_setting():
    assert WebConfig().open_browser and web_config({}).open_browser
    assert not web_config({"TOOLKIT_NO_BROWSER": "1"}).open_browser
    assert web_config({"TOOLKIT_NO_BROWSER": "0"}).open_browser
    with pytest.raises(ValueError, match="TOOLKIT_NO_BROWSER"):  # only 0 or 1 (step 1.5e)
        web_config({"TOOLKIT_NO_BROWSER": "yes"})


def test_browser_opens_once_the_port_answers(monkeypatch):
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url) or True)
    with socket.socket() as server:
        server.bind((HOST, 0))
        server.listen()
        port = server.getsockname()[1]
        url = f"http://{HOST}:{port}/"
        web.open_browser_when_ready(url, HOST, port).join(10)
    assert opened == [url]


def test_browser_not_opened_when_the_port_never_answers(monkeypatch, capsys):
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url) or True)
    port = _free_port()
    web.open_browser_when_ready(f"http://{HOST}:{port}/", HOST, port, timeout=0.5).join(10)
    assert opened == [] and capsys.readouterr().out == ""


@pytest.mark.parametrize("failure", ["returns False", "raises"])
def test_no_browser_says_where_to_go(monkeypatch, capsys, failure):
    def no_browser(url):
        if failure == "raises":
            raise webbrowser.Error("no runnable browser")
        return False

    monkeypatch.setattr(webbrowser, "open", no_browser)
    with socket.socket() as server:
        server.bind((HOST, 0))
        server.listen()
        port = server.getsockname()[1]
        web.open_browser_when_ready(f"http://{HOST}:{port}/", HOST, port).join(10)
    assert capsys.readouterr().out == f"Open http://{HOST}:{port}/ in your browser.\n"


def test_run_server_opens_the_browser_unless_turned_off(started, capsys):
    port = _free_port()
    assert web.run_server(WebConfig(port=port)) == 0
    url = f"http://{HOST}:{port}/"
    assert started["browser"] == [(url, HOST, port)]
    assert f"IDS web app running at {url}" in capsys.readouterr().out
    assert web.run_server(WebConfig(port=port, open_browser=False)) == 0
    assert len(started["browser"]) == 1


def test_waitress_gets_the_socket_bound_here(started):
    port = _free_port()
    web.run_server(WebConfig(port=port, open_browser=False))
    [kw] = started["serve"]
    assert kw["address"] == (HOST, port)
    assert "host" not in kw and "port" not in kw


def test_port_in_use_is_a_message(started, capsys):
    with net.bind_port(HOST, 0) as taken:
        port = taken.getsockname()[1]
        assert web.run_server(WebConfig(port=port)) == 2
    out = capsys.readouterr().out
    assert f"Port {port} is already in use on this computer" in out
    assert f"python -m toolkit web --port {port + 1}" in out
    assert "running at" not in out
    assert started["serve"] == [] and started["browser"] == []


def test_the_port_is_never_shared():
    """A second bind fails while the trainer's socket is open, even with SO_REUSEADDR (on
    Windows waitress's own socket would have let it through, step 1.4d)."""
    with net.bind_port(HOST, 0) as ours:
        port = ours.getsockname()[1]
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            assert ours.getsockopt(socket.SOL_SOCKET, exclusive)
        for reuse in (False, True):
            with socket.socket() as other:
                if reuse:
                    other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                with pytest.raises(OSError):
                    other.bind((HOST, port))
                    other.listen()
        with pytest.raises(OSError):
            net.bind_port(HOST, port)
    net.bind_port(HOST, port).close()  # free again once closed (a quick restart works)


def test_ctrl_c_stops_cleanly(monkeypatch, capsys):
    def interrupted(app, **kw):
        raise KeyboardInterrupt

    import waitress
    monkeypatch.setattr(waitress, "serve", interrupted)
    port = _free_port()
    assert web.run_server(WebConfig(port=port, open_browser=False)) == 0
    assert capsys.readouterr().out.endswith("Web trainer stopped.\n")
    net.bind_port(HOST, port).close()  # the port was released


def test_ctrl_c_in_debug_mode(monkeypatch, capsys):
    import flask

    def interrupted(self, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(flask.Flask, "run", interrupted)
    assert web.run_server(WebConfig(port=_free_port(), debug=True, open_browser=False)) == 0
    assert "Web trainer stopped." in capsys.readouterr().out


def test_web_flags(monkeypatch):
    received = []
    monkeypatch.setattr(web_command, "run_server", lambda config: received.append(config) or 0)
    monkeypatch.setenv("TOOLKIT_PORT", "5055")
    monkeypatch.delenv("TOOLKIT_NO_BROWSER")
    web_command.run([])
    assert (received[-1].port, received[-1].open_browser) == (5055, True)
    web_command.run(["--port", "5099", "--no-browser"])
    assert (received[-1].port, received[-1].open_browser) == (5099, False)


@pytest.mark.parametrize("argv, env", [(["--port", "0"], None), (["--port", "web"], None),
                                       ([], "abc"), ([], "70000")])
def test_bad_port_exits_2(monkeypatch, capsys, argv, env):
    monkeypatch.setattr(web_command, "run_server", lambda config: pytest.fail("started"))
    if env:
        monkeypatch.setenv("TOOLKIT_PORT", env)
    with pytest.raises(SystemExit) as stop:
        web_command.run(argv)
    assert stop.value.code == 2
    captured = capsys.readouterr()
    assert "must be a port number from 1 to 65535" in captured.out + captured.err


def test_port_in_use_exits_2(monkeypatch):
    monkeypatch.setattr(web_command, "run_server", lambda config: 2)
    with pytest.raises(SystemExit) as stop:
        web_command.run([])
    assert stop.value.code == 2
