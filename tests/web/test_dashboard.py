"""F10a: the home page's dashboard, ``GET /api/v1/dashboard``. Two parts side by side and
never linked: the scans and their risks (the newest scan of each target, the riskiest first;
the newest scans), and the models (the newest saved runs; what this session holds). No IDS
model is suggested from ports or CVEs. A session without a workspace gets nulls and no
cookie; a part that can't be read says so and the other still shows."""

from types import SimpleNamespace

import pandas as pd
from test_scans_page import _record
from toolkit.core.data.recon import scan_records
from toolkit.web.services import dashboard
from toolkit.web.services.jobs import SCAN, TRAINING
from web_js import JS_DIR
from web_jobs import train_and_wait

URL = "/api/v1/dashboard"


def _save(scan_id, target, severity="NO KNOWN CVE", **over):
    entry = {**_record(scan_id)["entries"][0], "software": "OpenSSH 8.9", "severity": severity}
    scan_records.write_scan(scan_id, _record(scan_id, target=target, addresses=[target],
                                             entries=[entry], **over))


def test_an_empty_install_and_session(client):
    answer = client.get(URL)
    assert answer.status_code == 200 and "Set-Cookie" not in answer.headers
    body = answer.get_json()
    assert set(body) == {"scans", "runs", "session"}, "two parts and the session, unlinked"
    assert body["scans"] == {"targets": [], "by_risk": [], "recent": [], "read": 0,
                             "more": False, "left_out": 0}
    assert body["runs"] == {"recent": [], "more": False, "left_out": 0}
    assert set(body["session"].values()) == {None}


def test_the_scans_and_their_risks(client):
    _save("20261101T100000Z-00000001", "10.0.0.9", "CRITICAL")
    _save("20261102T100000Z-00000002", "10.0.0.9", "LOW")  # the newest of 10.0.0.9
    for day in range(3, 9):
        _save(f"202611{day:02d}T100000Z-0000000{day}", f"10.0.1.{day}", "HIGH")
    scans = client.get(URL).get_json()["scans"]
    assert len(scans["recent"]) == 5 and scans["recent"][0]["scan_id"].startswith("20261108")
    assert [t["risk"] for t in scans["targets"]] == ["HIGH"] * 6 + ["LOW"]
    assert scans["by_risk"] == [{"risk": "HIGH", "count": 6}, {"risk": "LOW", "count": 1}]
    assert scans["read"] == 8 and scans["more"] is False


def test_a_file_that_isnt_a_scan_is_counted_not_an_error(client):
    _save("20261101T100000Z-00000001", "10.0.0.9")
    (scan_records.history_dir() / "20261102T100000Z-00000002.json").write_text("{}", "utf-8")
    scans = client.get(URL).get_json()["scans"]
    assert scans["left_out"] == 1 and len(scans["targets"]) == 1


def test_a_part_that_cant_be_read_says_so(client, monkeypatch):
    def broken(*_args, **_kwargs):
        raise PermissionError(13, "Access is denied")
    monkeypatch.setattr(dashboard, "saved_scans", broken)
    body = client.get(URL).get_json()
    assert body["scans"] == {"error": "The saved scans couldn't be read: Access is denied."}
    assert body["runs"]["recent"] == [], "the other part still shows"


def test_the_runs_and_the_session_after_a_training(client, upload, fixtures_dir):
    columns = upload(fixtures_dir / "cic_ddos_sample.csv").get_json()["columns"]
    trained = train_and_wait(client, {"features": [c for c in columns if c != "Label"],
                                      "target": "Label", "balancing": "none",
                                      "test_size": 0.3, "models": ["decision_tree"]})
    run_id = trained.get_json()["run_id"]
    body = client.get(URL).get_json()
    assert [r["run_id"] for r in body["runs"]["recent"]] == [run_id]
    assert body["runs"]["recent"][0]["selected"] == "decision_tree"
    session = body["session"]
    assert session["dataset"]["files"] == ["cic_ddos_sample.csv"] and session["dataset"]["rows"]
    assert session["run"] == {"run_id": run_id, "saved": True}
    assert session["training"] is None and session["scan"] is None


def test_the_home_page_shows_the_two_parts(client):
    page = client.get("/").get_data(as_text=True)
    for part in ('id="scansPart"', 'id="runsPart"', 'id="sessionPart"', 'id="dashboardRefresh"',
                 "js/dashboard.js", 'href="/dynamic_ids"', 'href="/scan"'):
        assert part in page, part
    assert "feature-card" not in page, "the old cards are gone"
    script = (JS_DIR / "dashboard.js").read_text(encoding="utf-8")
    assert 'getJSON("/api/v1/dashboard")' in script and "innerHTML" not in script
    assert "kept in this session only" in script, "the predictions aren't said to be saved"


def test_the_session_part_names_what_the_session_holds():
    job = SimpleNamespace(state="running", phase=lambda: "Training 2 of 3")
    ws = SimpleNamespace(
        data=SimpleNamespace(files=["a.csv"], frame=pd.DataFrame({"x": [1, 2, 3]})),
        run=SimpleNamespace(run_id="20261103T100000Z-00000001"), not_saved="disk full",
        loaded=SimpleNamespace(run_id="20261102T100000Z-00000002", models={"m1": 0, "m2": 0}),
        predictions=SimpleNamespace(answer={"source": "bundle", "model": "m1", "file": "b.csv",
                                            "rows": 9, "counts": [{"label": "DoS", "count": 9}],
                                            "download_url": "/api/v1/predictions"}),
        scan=SimpleNamespace(answer={"target": "10.0.0.5", "entries": [{}, {}],
                                     "saved": {"scan_id": "20261103T110000Z-00000003"}}),
        active_job=lambda kind: job if kind in (TRAINING, SCAN) else None)
    assert dashboard.session_part(ws) == {
        "dataset": {"files": ["a.csv"], "rows": 3},
        "training": {"state": "running", "phase": "Training 2 of 3"},
        "run": {"run_id": "20261103T100000Z-00000001", "saved": False},
        "bundle": {"run_id": "20261102T100000Z-00000002", "models": 2},
        "predictions": {"source": "bundle", "model": "m1", "file": "b.csv", "rows": 9,
                        "counts": [{"label": "DoS", "count": 9}]},
        "scan": {"state": "running", "phase": "Training 2 of 3"},
        "last_scan": {"target": "10.0.0.5", "open_ports": 2,
                      "scan_id": "20261103T110000Z-00000003"},
    }
