"""v5.2 step 2.4d: the scan's routes over HTTP - the scan contract as the page meets it,
the job routes for a scan (its log), the session's scan, the report download, and the other
checks of every request (the CSRF token, the cross-site check, the Host check). The network
is faked (``web_scan.fake_network``): nothing leaves this computer.
"""

import json

import pytest

from toolkit.core.services.recon import scan as core_scan
from toolkit.web.controllers import scan as scan_routes
from web_jobs import wait_for
from web_scan import fake_network

SCAN_URL = "/api/v1/jobs/scan"
BODY = {"target": "10.0.0.5", "ports": "22,8080", "authorized": True}


@pytest.fixture
def net(monkeypatch):
    rec = fake_network(monkeypatch)
    monkeypatch.setattr(scan_routes, "load_api_keys", lambda: dict(rec.keys))
    return rec


def _code(response):
    return response.get_json()["error"]["code"]


def test_the_options(client, net):
    net.keys.update(openrouter="or-secret-value-1")
    response = client.get("/api/v1/scan/options")
    assert response.status_code == 200 and response.headers["Cache-Control"] == "no-store"
    options = response.get_json()
    assert options["keys"] == {"openrouter": True, "abuseipdb": False, "nvd": False}
    assert "or-secret-value-1" not in response.get_data(as_text=True)
    assert [engine["id"] for engine in options["engines"]] == ["builtin", "nmap"]
    assert options["engines"][1]["available"] is False


@pytest.mark.parametrize("body", [{**BODY, "authorized": False}, {"target": "scanme.example"},
                                  {**BODY, "target": "10.0.0.0/24", "authorized": False,
                                   "engine": "nmap", "abuseipdb": True}])
def test_nothing_is_sent_before_the_authorization(client, net, workspaces, body):
    response = client.post(SCAN_URL, json=body)
    assert response.status_code == 400 and _code(response) == "NOT_AUTHORIZED"
    assert net.touched() == {} and len(workspaces) == 0, "no DNS, no workspace even"


@pytest.mark.parametrize("value", ["yes", 1, "true", None])
def test_the_authorization_is_true_or_nothing(client, net, value):
    response = client.post(SCAN_URL, json={**BODY, "authorized": value})
    assert response.status_code == 400 and net.touched() == {}
    assert _code(response) == ("NOT_AUTHORIZED" if value is None else "BAD_REQUEST")


def test_a_scan_through_the_routes(client, net):
    submitted = client.post(SCAN_URL, json={**BODY, "target": "scanme.example"})
    assert submitted.status_code == 202
    view = submitted.get_json()
    assert view["kind"] == "scan" and view["url"] == f"/api/v1/jobs/{view['id']}"
    done = wait_for(client, view["url"])
    assert done["state"] == "done" and done["percent"] == 100, done
    assert done["result"]["message"] == "Scanned 1 host(s): 2 open port(s)."
    assert done["log"][0] == "scanme.example resolved to 10.0.0.9"
    assert done["log_size"] == len(done["log"])
    tail = client.get(view["url"] + f"?since={done['log_size'] - 1}").get_json()
    assert tail["log"] == done["log"][-1:] and tail["log_size"] == done["log_size"]
    session = client.get("/api/v1/scan").get_json()
    assert session == {"job": None, "last": done["result"]}
    assert client.get("/api/v1/workspace").get_json()["job"] is None, "the trainer's job only"


def test_the_report_download(client, net, web_app):
    assert _code(client.get("/api/v1/scan/report?format=csv")) == "NO_SCAN"
    assert client.get("/api/v1/scan/report").status_code == 404
    wait_for(client, client.post(SCAN_URL, json=BODY).get_json()["url"])
    for fmt, media in (("txt", "text/plain"), ("json", "application/json"),
                       ("csv", "text/csv")):
        response = client.get(f"/api/v1/scan/report?format={fmt}")
        assert response.status_code == 200
        assert response.headers["Content-Type"] == f"{media}; charset=utf-8"
        disposition = response.headers["Content-Disposition"]
        assert disposition.startswith('attachment; filename="scan-') and \
            disposition.endswith(f'.{fmt}"')
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        (ws,) = web_app.extensions["toolkit"].workspaces.values()
        assert response.get_data(as_text=True) == core_scan.report_text(ws.scan.entries, fmt)
    assert json.loads(client.get("/api/v1/scan/report?format=json").get_data())[0]["port"] == 22
    bad = client.get("/api/v1/scan/report?format=exe")
    assert bad.status_code == 400 and "'format' must be one of" in bad.get_json()["error"][
        "message"]
    stranger = web_app.test_client()
    assert stranger.get("/api/v1/scan/report").status_code == 404, "the session's own scan"


def test_one_scan_per_session_cancel_and_clearing(client, net):
    net.release.clear()
    first = client.post(SCAN_URL, json=BODY).get_json()
    assert net.entered.wait(30)
    busy = client.post(SCAN_URL, json=BODY)
    assert busy.status_code == 409 and _code(busy) == "BUSY"
    clearing = client.delete("/api/v1/workspace")
    assert clearing.status_code == 409 and "A scan is running" in clearing.get_json()[
        "error"]["message"]
    running = client.get("/api/v1/scan").get_json()["job"]
    assert running["id"] == first["id"] and running["state"] == "running"
    assert client.delete(first["url"]).get_json()["state"] == "cancelling"
    ended = wait_for(client, first["url"])
    assert ended["error"]["code"] == "CANCELLED" and "result" not in ended
    assert client.get("/api/v1/scan").get_json() == {"job": None, "last": None}
    assert client.delete("/api/v1/workspace").status_code == 200


def test_a_session_can_scan_again_as_soon_as_it_sees_the_end(client, net):
    for _ in range(5):
        submitted = client.post(SCAN_URL, json=BODY)
        assert submitted.status_code == 202, submitted.get_json()
        assert wait_for(client, submitted.get_json()["url"])["state"] == "done"


@pytest.mark.parametrize("body, code", [
    ({**BODY, "engine": "nmap"}, "ENGINE_UNAVAILABLE"),
    ({**BODY, "target": "10.0.0.0/16"}, "SCAN_LIMIT"),
    ({**BODY, "ports": "1-99999"}, "BAD_REQUEST"),
])
def test_refusals_answer_at_once(client, net, body, code):
    response = client.post(SCAN_URL, json=body)
    assert response.status_code == 400 and _code(response) == code
    assert net.sent["ping"] == net.sent["scan"] == []


def test_every_other_check_comes_first(client, net, web_app):
    no_token = client.post(SCAN_URL, json=BODY, headers={"X-CSRF-Token": "wrong"})
    assert no_token.status_code == 403 and _code(no_token) == "CSRF_TOKEN"
    cross_site = client.post(SCAN_URL, json=BODY, headers={"Origin": "http://evil.example"})
    assert cross_site.status_code == 403 and _code(cross_site) == "CROSS_SITE"
    rebound = client.post(SCAN_URL, json=BODY, headers={"Host": "evil.example:5000"})
    assert rebound.status_code == 400 and _code(rebound) == "BAD_HOST"
    form = client.post(SCAN_URL, data={"target": "10.0.0.5", "authorized": "true"})
    assert form.status_code == 400 and _code(form) == "BAD_REQUEST", "JSON only"
    assert net.touched() == {}
