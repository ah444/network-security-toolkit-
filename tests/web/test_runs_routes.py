"""v5.3 F2c: the saved runs' routes. They read the TrainingRun records only and need no
session; a run id is checked before a file name is made from it."""

import copy
import json

import pytest

from toolkit.core.data.ids.run_files import runs_dir
from web_jobs import train_and_wait

CIC = "cic_ddos_sample.csv"
TARGET = "Label"
OTHER_ID = "20261030T120000Z-0000000a"


@pytest.fixture
def exported(client, upload, fixtures_dir):
    """A run trained and exported through the routes, as the page does: its record."""
    columns = upload(fixtures_dir / CIC).get_json()["columns"]
    trained = train_and_wait(client, {
        "features": [c for c in columns if c != TARGET], "target": TARGET, "balancing": "none",
        "test_size": 0.3, "models": ["decision_tree", "logistic_regression"]})
    assert trained.status_code == 200, trained.get_json()
    result = trained.get_json()
    assert set(result["folds"]) == {"decision_tree", "logistic_regression"}
    assert all(len(scores) == result["cv"]["folds"] for scores in result["folds"].values())
    answer = client.post("/api/v1/run/export", json={"name": "first"}).get_json()
    path = runs_dir() / f"{answer['run_id']}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _save(record, run_id):
    record = copy.deepcopy(record)
    record["run_id"] = run_id
    (runs_dir() / f"{run_id}.json").write_text(json.dumps(record), encoding="utf-8")
    return record


def _error(response):
    return response.get_json()["error"]


def test_no_runs_yet(client):
    response = client.get("/api/v1/runs")
    assert response.status_code == 200
    assert response.get_json() == {"runs": [], "more": False, "left_out": 0}


def test_an_exported_run_is_listed_shown_and_compared(client, exported):
    listed = client.get("/api/v1/runs").get_json()
    assert [r["run_id"] for r in listed["runs"]] == [exported["run_id"]]
    summary = listed["runs"][0]
    assert summary["dataset"] == {"name": CIC, "rows": 200} and summary["label"] == TARGET

    one = client.get(f"/api/v1/runs/{exported['run_id']}")
    assert one.status_code == 200
    assert set(one.get_json()) == {"summary", "leaderboard", "folds"}
    assert one.get_json()["summary"] == summary

    same = client.get("/api/v1/runs/compare",
                      query_string={"a": exported["run_id"], "b": exported["run_id"]}).get_json()
    assert same["comparable"] is True and same["reasons"] == []
    assert {m["model"] for m in same["models"]} == {"decision_tree", "logistic_regression"}
    assert all(set(m["delta"].values()) == {0} for m in same["models"])


def test_other_data_is_compared_without_differences(client, exported):
    other = copy.deepcopy(exported)
    other["dataset"]["sha256"] = "cd" * 32
    _save(other, OTHER_ID)
    answer = client.get("/api/v1/runs/compare",
                        query_string={"a": exported["run_id"], "b": OTHER_ID}).get_json()
    assert answer["comparable"] is False and answer["reasons"] == ["DIFFERENT_DATASET"]
    assert all(m["delta"] is None for m in answer["models"])


def test_the_list_pages(client, exported):
    ids = [f"20200101T0000{n:02d}Z-0000000{n}" for n in range(1, 4)]  # older than the export
    for run_id in ids:
        _save(exported, run_id)
    newest = client.get("/api/v1/runs").get_json()
    assert [r["run_id"] for r in newest["runs"]] == [exported["run_id"], *ids[::-1]]
    older = client.get("/api/v1/runs", query_string={"before": ids[1]}).get_json()
    assert [r["run_id"] for r in older["runs"]] == [ids[0]] and older["more"] is False


@pytest.mark.parametrize("bad", ["x", "20261030T120001Z-0000000G",
                                 "..%5C20261030T120001Z-00000001",
                                 "20261030T120001Z-00000001.json", "%2e%2e"])
def test_a_bad_run_id_is_refused(client, bad):
    response = client.get(f"/api/v1/runs/{bad}")
    assert response.status_code == 400 and _error(response)["code"] == "BAD_REQUEST"
    assert _error(response)["message"] == "That is not a run id."
    for query in ({"a": bad, "b": OTHER_ID}, {"a": OTHER_ID, "b": bad}):
        response = client.get("/api/v1/runs/compare", query_string=query)
        assert response.status_code == 400
    assert client.get("/api/v1/runs", query_string={"before": bad}).status_code == 400


def test_a_path_in_the_url_is_not_a_route(client):
    assert client.get("/api/v1/runs/../signing.key").status_code == 404


def test_a_missing_run(client):
    response = client.get(f"/api/v1/runs/{OTHER_ID}")
    assert response.status_code == 404 and _error(response)["code"] == "NOT_FOUND"
    response = client.get("/api/v1/runs/compare", query_string={"a": OTHER_ID, "b": OTHER_ID})
    assert response.status_code == 404 and _error(response)["code"] == "NOT_FOUND"


def test_compare_needs_both(client):
    response = client.get("/api/v1/runs/compare", query_string={"a": OTHER_ID})
    assert response.status_code == 400 and "'b' is required" in _error(response)["message"]


def test_a_bad_record_is_left_out_and_never_a_500(client, exported):
    runs_dir().joinpath(f"{OTHER_ID}.json").write_text("{not json", encoding="utf-8")
    listed = client.get("/api/v1/runs").get_json()
    assert [r["run_id"] for r in listed["runs"]] == [exported["run_id"]]
    assert listed["left_out"] == 1
    response = client.get(f"/api/v1/runs/{OTHER_ID}")
    assert response.status_code == 400
    message = _error(response)["message"]
    assert "not a TrainingRun record" in message
    assert str(runs_dir()) not in message and "\\" not in message, "no server paths"


def test_reading_runs_makes_no_workspace(client, workspaces, exported):
    """The routes need no session: reading the runs doesn't create a workspace."""
    client.delete("/api/v1/workspace")
    before = len(workspaces)
    client.get("/api/v1/runs")
    client.get(f"/api/v1/runs/{exported['run_id']}")
    assert len(workspaces) == before
