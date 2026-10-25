"""v5.2 step 2.4a: the Host check against DNS rebinding. A page on another site whose name is
made to point at 127.0.0.1 is the same site as the app to the browser, so it could read the
CSRF token and send everything; its requests carry its own name in the Host header, and only
an IP address or localhost on the app's port is answered."""

import logging

import pytest

from toolkit.web.controllers.security import MAX_HOSTS_LOGGED, is_own_host

PORT = 5000


@pytest.mark.parametrize(("host", "scheme"), [
    ("127.0.0.1:5000", "http"),
    ("localhost:5000", "http"),
    ("LocalHost:5000", "http"),
    ("[::1]:5000", "http"),
    ("192.168.1.5:5000", "http"),  # this computer's LAN address (with --host, step 2.4f)
    ("[2001:db8::1:2]:5000", "http"),
])
def test_an_address_or_localhost_on_the_port_is_answered(host, scheme):
    assert is_own_host(host, PORT, scheme)


@pytest.mark.parametrize(("host", "port", "scheme"), [
    ("127.0.0.1", 80, "http"),  # a browser leaves the scheme's own port out
    ("localhost", 443, "https"),
    ("[::1]", 80, "http"),
])
def test_no_port_is_the_schemes_own(host, port, scheme):
    assert is_own_host(host, port, scheme)


@pytest.mark.parametrize("host", [
    "evil.example:5000",  # the rebound name
    "127.0.0.1.nip.io:5000",
    "localhost.evil.example:5000",
    "my-pc:5000",
    "127.0.0.1:5001",  # another port
    "127.0.0.1",  # port 80, not the app's
    "127.0.0.1:5000:5000",
    "::1:5000",  # IPv6 without brackets
    "[::1]5000",
    "[::1",
    "[fe80::1%eth0]:5000",  # a zone
    "[127.0.0.1]:5000",  # IPv4 in brackets
    "[localhost]:5000",
    "2130706433:5000",  # 127.0.0.1 as a number
    "0x7f.0.0.1:5000",
    "127.1:5000",
    ":5000",
    "",
])
def test_any_other_host_is_refused(host):
    assert not is_own_host(host, PORT)


@pytest.fixture
def evil(client):
    """The test client, its requests naming the rebound host."""
    port = client.application.extensions["toolkit"].config.port
    return {"Host": f"evil.example:{port}"}


@pytest.mark.parametrize("path", ["/", "/dynamic_ids", "/static/js/main.js"])
def test_pages_are_refused_with_the_address_to_open(client, evil, path):
    response = client.get(path, headers=evil)
    assert response.status_code == 400 and response.mimetype == "text/plain"
    port = client.application.extensions["toolkit"].config.port
    assert response.get_data(as_text=True) == (
        f"This address isn't the toolkit's: open http://127.0.0.1:{port}/ instead.\n")
    assert response.headers["Content-Security-Policy"].startswith("default-src 'self'")
    assert "csrf-token" not in response.get_data(as_text=True)


def test_the_api_is_refused_as_json(client, evil):
    response = client.get("/api/v1/models", headers=evil)
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "BAD_HOST"
    assert response.headers["Cache-Control"] == "no-store"


def test_a_change_is_refused_before_the_token_is_checked(client, evil, fixtures_dir):
    client.get("/dynamic_ids")  # the session has a token, which the client sends

    def upload(**kwargs):
        with (fixtures_dir / "cic_ddos_sample.csv").open("rb") as handle:
            return client.post("/api/v1/dataset", data={"dataset": (handle, "d.csv")},
                               content_type="multipart/form-data", **kwargs)

    response = upload(headers=evil)
    assert response.status_code == 400 and response.get_json()["error"]["code"] == "BAD_HOST"
    assert client.get("/api/v1/dataset/preview").get_json()["error"]["code"] == "NO_DATASET"
    assert upload().status_code == 200  # the same upload on the app's own address


def test_the_apps_own_address_is_answered(client):
    port = client.application.extensions["toolkit"].config.port
    for host in (f"127.0.0.1:{port}", f"localhost:{port}"):
        assert client.get("/api/v1/models", headers={"Host": host}).status_code == 200


def test_each_refused_host_is_logged_once(client, evil, caplog):
    with caplog.at_level(logging.WARNING, logger="toolkit.web.controllers.security"):
        for _ in range(3):
            client.get("/", headers=evil)
        for number in range(MAX_HOSTS_LOGGED + 5):
            client.get("/", headers={"Host": f"name{number}.example:5000"})
    refused = [r.getMessage() for r in caplog.records if "Refused a request" in r.getMessage()]
    assert refused.count(f"Refused a request for the host '{evil['Host']}': not this app's "
                         "address") == 1
    assert len(refused) == MAX_HOSTS_LOGGED
