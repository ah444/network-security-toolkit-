"""v5.2 step 2.4f: --host opens the toolkit to the network. Other computers type an access
token (new at every start, never logged, five wrong tries a minute per address); this
computer's own browser never needs it; every page shows a banner; TOOLKIT_DEBUG=1 is refused;
the warning and the token are printed. No test binds a network address: the binding is faked.
"""

import errno
import logging
import re

import pytest

from toolkit.core.data import logs
from toolkit.core.data.settings import WebConfig
from toolkit.web.controllers import command as web_command
from toolkit.web.controllers import server as web
from toolkit.web.data import net
from toolkit.web.services import access as access_service

OTHER = "192.168.56.20"  # another computer on the host-only network
TOKEN_FORMAT = re.compile(r"[A-HJ-NP-Z2-9]{4}(-[A-HJ-NP-Z2-9]{4}){3}")


@pytest.fixture
def open_app(make_app):
    return make_app(WebConfig(host="0.0.0.0"))


@pytest.fixture
def token(open_app):
    return open_app.extensions["toolkit"].access.token


def _other(app, address=OTHER):
    test_client = app.test_client()
    test_client.environ_base["REMOTE_ADDR"] = address
    return test_client


def _unlock(test_client, typed):
    test_client.get("/unlock")  # the unlock page gives the session its CSRF token
    return test_client.post("/api/v1/unlock", json={"token": typed})


def _code(response):
    return response.get_json()["error"]["code"]


# --- The token ---------------------------------------------------------------------------

def test_a_token_is_16_unambiguous_characters_new_each_time():
    tokens = {access_service.new_token() for _ in range(200)}
    assert len(tokens) == 200 and all(TOKEN_FORMAT.fullmatch(t) for t in tokens)
    assert not set("".join(tokens)) & set("01OI")
    assert len(access_service.ALPHABET) ** access_service.TOKEN_LENGTH == 2 ** 80


@pytest.mark.parametrize("address, here", [
    ("127.0.0.1", True), ("127.8.8.8", True), ("::1", True), ("::ffff:127.0.0.1", True),
    (OTHER, False), ("::ffff:192.168.56.20", False), ("", False), (None, False), ("x", False)])
def test_this_computer_is_a_loopback_address(address, here):
    assert access_service.is_this_computer(address) is here


def test_the_default_app_has_no_token_and_no_banner(web_app):
    assert web_app.extensions["toolkit"].access is None
    other = _other(web_app)
    assert other.get("/").status_code == 200, "nothing to lock: it listens on 127.0.0.1 only"
    assert "network-banner" not in other.get("/").get_data(as_text=True)


# --- Another computer --------------------------------------------------------------------

@pytest.mark.parametrize("url", ["/", "/dynamic_ids", "/scan"])
def test_another_computers_pages_go_to_the_unlock_page(open_app, url):
    response = _other(open_app).get(url)
    assert response.status_code == 302 and response.headers["Location"] == "/unlock"


@pytest.mark.parametrize("method, url", [("GET", "/api/v1/scan/options"),
                                         ("GET", "/api/v1/workspace"),
                                         ("POST", "/api/v1/jobs/scan"),
                                         ("DELETE", "/api/v1/workspace")])
def test_another_computers_api_is_locked(open_app, method, url):
    response = _other(open_app).open(url, method=method, json={},
                                     headers={"X-CSRF-Token": "x"})
    assert response.status_code == 401 and _code(response) == "LOCKED"
    assert "/unlock" in response.get_json()["error"]["message"]


def test_the_unlock_page_and_the_static_files_stay_open(open_app, token):
    other = _other(open_app)
    page = other.get("/unlock")
    html = page.get_data(as_text=True)
    assert page.status_code == 200 and page.headers["Cache-Control"] == "no-store"
    assert 'id="accessToken"' in html and token not in html
    assert '<p class="network-banner" role="note">This toolkit is open to your network' in html
    assert other.get("/static/css/app.css").status_code == 200


def test_the_token_unlocks_the_session(open_app, token):
    other = _other(open_app)
    wrong = _unlock(other, "AAAA-BBBB-CCCC-DDDD")
    assert wrong.status_code == 401 and _code(wrong) == "WRONG_TOKEN"
    typed = token.lower().replace("-", " ")  # case, spaces and dashes don't matter
    unlocked = _unlock(other, typed)
    assert unlocked.status_code == 200 and token not in unlocked.get_data(as_text=True)
    assert other.get("/scan").status_code == 200
    assert other.get("/api/v1/scan/options").status_code == 200
    assert other.get("/unlock").headers["Location"] == "/", "nothing more to type"
    assert _other(open_app).get("/").status_code == 302, "another browser types it too"


def test_five_wrong_tokens_a_minute_lock_the_address_out(open_app, token, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(access_service.time, "monotonic", lambda: clock[0])
    other = _other(open_app)
    for _ in range(access_service.MAX_TRIES):
        assert _code(_unlock(other, "wrong")) == "WRONG_TOKEN"
    refused = _unlock(other, token)
    assert refused.status_code == 429 and _code(refused) == "TOO_MANY_TRIES"
    assert _unlock(_other(open_app, "192.168.56.21"), token).status_code == 200, \
        "another address isn't locked out"
    clock[0] += access_service.TRY_WINDOW + 1
    assert _unlock(other, token).status_code == 200, "a minute later it may try again"


def test_a_forwarded_for_header_is_never_trusted(open_app):
    response = _other(open_app).get("/api/v1/scan/options",
                                    headers={"X-Forwarded-For": "127.0.0.1"})
    assert response.status_code == 401


def test_a_new_start_needs_the_new_token(open_app, token):
    other = _other(open_app)
    assert _unlock(other, token).status_code == 200
    state = open_app.extensions["toolkit"]
    state.access = access_service.Access()  # what a restart makes
    assert other.get("/scan").status_code == 302, "the old unlock is gone"
    assert _code(_unlock(other, token)) == "WRONG_TOKEN", "the old token is gone"
    assert _unlock(other, state.access.token).status_code == 200


def test_the_token_is_never_logged(open_app, token, caplog):
    caplog.set_level(logging.INFO)
    other = _other(open_app)
    _unlock(other, "wrong-token")
    _unlock(other, token)
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert f"A wrong access token from {OTHER}" in text and f"Unlocked for {OTHER}" in text
    assert token not in text
    assert logs.redact(f"token {token}") == "token [REDACTED]", "a registered secret"


# --- This computer -----------------------------------------------------------------------

def test_this_computer_needs_no_token(open_app):
    test_client = open_app.test_client()  # REMOTE_ADDR 127.0.0.1
    page = test_client.get("/scan")
    assert page.status_code == 200
    assert "open to your network (all of this computer&#39;s addresses, port 5000)" in \
        page.get_data(as_text=True)
    assert test_client.get("/unlock").headers["Location"] == "/"
    answer = _unlock(test_client, "anything")
    assert answer.status_code == 200 and "No access token is needed" in answer.get_json()[
        "message"]


@pytest.mark.parametrize("host, where", [("192.168.56.1", "192.168.56.1, port 5000"),
                                         ("::", "all of this computer&#39;s addresses")])
def test_every_page_says_where_it_is_open(make_app, host, where):
    test_client = make_app(WebConfig(host=host)).test_client()
    for url in ("/", "/dynamic_ids", "/scan"):
        assert f"This toolkit is open to your network ({where}" in test_client.get(url).get_data(
            as_text=True), url


# --- The command and the server ----------------------------------------------------------

@pytest.mark.parametrize("text, address", [("0.0.0.0", "0.0.0.0"), ("192.168.56.1", "192.168.56.1"),
                                           (" [::1] ", "::1"), ("::", "::")])
def test_host_takes_an_ip_address(text, address):
    assert web_command.listen_address(text) == address


@pytest.mark.parametrize("text", ["my-laptop", "localhost", "192.168.56", ""])
def test_host_refuses_names(monkeypatch, capsys, text):
    monkeypatch.setattr(web_command, "run_server", lambda config: pytest.fail("started"))
    with pytest.raises(SystemExit) as stop:
        web_command.run(["--host", text])
    assert stop.value.code == 2
    assert "--host takes an IP address of this computer" in capsys.readouterr().err


def test_host_reaches_the_server(monkeypatch):
    received = []
    monkeypatch.setattr(web_command, "run_server", lambda config: received.append(config) or 0)
    web_command.run(["--host", "192.168.56.1", "--no-browser"])
    web_command.run([])
    assert [config.host for config in received] == ["192.168.56.1", "127.0.0.1"]


def test_there_is_no_environment_variable_for_the_host(monkeypatch):
    monkeypatch.setenv("TOOLKIT_HOST", "0.0.0.0")
    from toolkit.core.data.settings import web_config
    assert web_config().host == "127.0.0.1"


class FakeSocket:
    def __init__(self, host, port):
        self.address = (host, port)
        self.closed = False

    def getsockname(self):
        return self.address

    def close(self):
        self.closed = True


@pytest.fixture
def server(monkeypatch):
    """run_server with the binding, waitress and the browser faked; what they were given."""
    calls = {"bound": [], "served": [], "browser": [], "refuse": None}

    def bind_ports(hosts, port):
        calls["bound"].append(list(hosts))
        if calls["refuse"] is not None:
            raise calls["refuse"]
        return [FakeSocket(host, port) for host in hosts]

    import waitress
    monkeypatch.setattr(net, "bind_ports", bind_ports)
    monkeypatch.setattr(net, "local_addresses", lambda: ["192.168.1.11", "192.168.56.1"])
    monkeypatch.setattr(waitress, "serve", lambda app, sockets: calls["served"].append(
        [sock.getsockname()[0] for sock in sockets]))
    monkeypatch.setattr(web, "open_browser_when_ready",
                        lambda url, host, port: calls["browser"].append(url))
    return calls


@pytest.mark.parametrize("host, bound", [("127.0.0.1", ["127.0.0.1"]),
                                         ("192.168.56.1", ["192.168.56.1", "127.0.0.1"]),
                                         ("0.0.0.0", ["0.0.0.0"]),
                                         ("::", ["::", "127.0.0.1"])])
def test_the_addresses_bound(server, host, bound):
    assert web.run_server(WebConfig(host=host, port=5099)) == 0
    assert server["bound"] == [bound] and server["served"] == [bound]
    assert server["browser"] == ["http://127.0.0.1:5099/"], "this computer's browser: no token"


def test_the_warning_and_the_token_are_printed(server, capsys):
    web.run_server(WebConfig(host="0.0.0.0", port=5099, open_browser=False))
    out = capsys.readouterr().out
    assert "WARNING: the toolkit is open to your network" in out
    assert "port scans FROM THIS COMPUTER" in out and "plain HTTP" in out
    assert "  http://192.168.1.11:5099/\n  http://192.168.56.1:5099/\n" in out
    assert TOKEN_FORMAT.search(out.split("Access token (typed once in each of their browsers): ")[1])
    assert "This computer's own browser, at 127.0.0.1, needs no token." in out


def test_this_computer_only_prints_no_warning(server, capsys):
    web.run_server(WebConfig(port=5099, open_browser=False))
    out = capsys.readouterr().out
    assert "WARNING" not in out and "Access token" not in out


def test_debug_is_refused_on_the_network(server, capsys):
    assert web.run_server(WebConfig(host="0.0.0.0", debug=True)) == 2
    assert "TOOLKIT_DEBUG=1 can't be used with --host 0.0.0.0" in capsys.readouterr().out
    assert server["bound"] == [], "refused before anything listens"


def test_an_address_of_another_computer_is_a_message(server, capsys):
    server["refuse"] = OSError(errno.EADDRNOTAVAIL, "cannot assign requested address")
    assert web.run_server(WebConfig(host="10.9.9.9")) == 2
    out = capsys.readouterr().out
    assert "10.9.9.9 is not an address of this computer" in out and "Port" not in out


def test_the_page_addresses():
    assert web.url_of("::1", 5000) == "http://[::1]:5000/"
    assert web.network_urls("192.168.56.1", 5000) == ["http://192.168.56.1:5000/"]
    assert web.network_urls("::", 5000) == []


def test_ipv6_loopback_can_be_bound():
    with net.bind_port("::1", 0) as sock:
        assert sock.getsockname()[0] == "::1"
