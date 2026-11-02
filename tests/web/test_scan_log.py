"""HS3 (RD1): a scan started on the web page is recorded like the CLI's, with where its request
came from: this computer, or the other computer's address (from HS1's ``Origin``); the job's
own line of v5.2 step 2.4d is gone. The network is faked (``web_scan.fake_network``)."""

import json

import pytest

from toolkit.core.data.settings import WebConfig
from toolkit.web.controllers import scan as scan_routes
from web_jobs import wait_for
from web_scan import fake_network

OTHER = "192.168.56.20"
SCAN_URL = "/api/v1/jobs/scan"
BODY = {"target": "10.0.0.5", "ports": "22,8080", "authorized": True}
RECORD = "toolkit.core.services.recon.record"


@pytest.fixture
def net(monkeypatch, caplog):
    caplog.set_level("INFO")
    rec = fake_network(monkeypatch)
    monkeypatch.setattr(scan_routes, "load_api_keys", lambda: dict(rec.keys))
    return rec


def _lines(caplog):
    return [(item.getMessage().split(" ", 2)[1], json.loads(item.getMessage().split(" ", 2)[2]))
            for item in caplog.records if item.name == RECORD]


def test_a_scan_from_this_computer(client, net, caplog):
    job = client.post(SCAN_URL, json=BODY).get_json()
    assert wait_for(client, job["url"])["state"] == "done"
    [(first, started), (last, ended)] = _lines(caplog)
    assert (first, last) == ("started", "ended")
    assert (started["source"], started["target"], started["ports"]) == (
        "this computer", "10.0.0.5", "22,8080")
    assert (ended["scan_id"], ended["outcome"], ended["open_ports"]) == (
        started["scan_id"], "done", 2)
    assert not [item for item in caplog.records if item.getMessage().startswith("Scan done:")], \
        "the job's own line of 2.4d is gone"


def test_a_scan_from_another_computer(make_app, net, caplog):
    app = make_app(WebConfig(host="0.0.0.0"))
    token = app.extensions["toolkit"].access.token
    other = app.test_client()
    other.environ_base["REMOTE_ADDR"] = OTHER
    other.get("/unlock")
    assert other.post("/api/v1/unlock", json={"token": token}).status_code == 200
    job = other.post(SCAN_URL, json=BODY).get_json()
    assert wait_for(other, job["url"])["state"] == "done"
    assert _lines(caplog)[0][1]["source"] == OTHER
    text = "\n".join(item.getMessage() for item in caplog.records)
    assert token not in text


def test_a_cancelled_scan_is_recorded_as_stopped(client, net, caplog):
    net.release.clear()  # the fake engine holds the scan until released
    job = client.post(SCAN_URL, json=BODY).get_json()
    assert net.entered.wait(10)
    client.delete(job["url"])
    net.release.set()
    assert wait_for(client, job["url"])["state"] == "cancelled"
    assert [event for event, _ in _lines(caplog)] == ["started", "ended"]
    assert _lines(caplog)[1][1]["outcome"] == "stopped"
