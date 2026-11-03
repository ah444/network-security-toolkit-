"""v5.3 step F3d: the explanations' routes.

- ``POST /api/v1/jobs/importance``: the permutation importance of one of the workspace's
  models, a job on the training's queue (RD2), bound to its run and model, its result a file
  of its own (HR2); followed and cancelled through ``/api/v1/jobs/<id>``.
- ``GET /api/v1/runs/<run id>/importance``: a saved run's results, read only (a saved run has
  no test split in memory: nothing is computed).
- ``why`` in a single prediction's answer (``/run/predict``, ``/bundle/predict``): the local
  explanation (F3c); a model without probabilities says so, and still predicts."""

import io
import threading

import pytest

from toolkit.core.services.ids.datasets import load_dataset
from toolkit.web.services import importance
from web_jobs import train_and_wait, wait_for

CIC = "cic_ddos_sample.csv"
TARGET = "Label"
URL = "/api/v1/jobs/importance"


@pytest.fixture
def columns(upload, fixtures_dir):
    return upload(fixtures_dir / CIC).get_json()["columns"]


def _train(client, columns, models=("decision_tree", "logistic_regression")):
    trained = train_and_wait(client, {
        "features": [c for c in columns if c != TARGET], "target": TARGET, "balancing": "none",
        "test_size": 0.3, "models": list(models)})
    assert trained.status_code == 200, trained.get_json()
    return trained.get_json()


def _row(fixtures_dir):
    """A real row of the fixture as the prediction form sends it (text values)."""
    row = load_dataset(fixtures_dir / CIC).frame.iloc[-1]
    return {c: str(v) for c, v in row.items() if c != TARGET}


def test_the_job_saves_its_result_and_the_run_shows_it(client, columns):
    trained = _train(client, columns)
    started = client.post(URL, json={})
    assert started.status_code == 202, started.get_json()
    assert started.get_json()["kind"] == "importance"
    job = wait_for(client, started.get_json()["url"])
    assert job["state"] == "done", job
    result = job["result"]
    assert (result["run_id"], result["model"]) == (trained["run_id"], trained["selected"])
    assert result["saved"] is True and result["not_saved"] is None
    assert len(result["features"]) == len(columns) - 1
    saved = client.get(f"/api/v1/runs/{trained['run_id']}/importance")
    assert saved.status_code == 200
    body = saved.get_json()
    assert body["left_out"] == 0 and [r["model"] for r in body["importances"]] == [result["model"]]
    stored = {f["feature"]: f["mean"] for f in body["importances"][0]["features"]}
    assert stored == {f["feature"]: f["mean"] for f in result["features"]}


def test_asked_twice_it_is_the_same_job(client, columns):
    _train(client, columns)
    first = client.post(URL, json={"model": "logistic_regression"}).get_json()
    second = client.post(URL, json={"model": "logistic_regression"}).get_json()
    assert first["id"] == second["id"]
    assert wait_for(client, first["url"])["result"]["model"] == "logistic_regression"


def test_it_can_be_cancelled(client, columns, monkeypatch):
    _train(client, columns)
    entered, release = threading.Event(), threading.Event()
    real = importance.permutation_importances

    def held(*args, progress, **kwargs):
        entered.set()
        assert release.wait(30), "the test never released the job"
        progress.advance()  # where a cancel stops it
        return real(*args, progress=progress, **kwargs)

    monkeypatch.setattr(importance, "permutation_importances", held)
    job = client.post(URL, json={}).get_json()
    assert entered.wait(30)
    assert client.delete(job["url"]).status_code == 202
    release.set()
    assert wait_for(client, job["url"])["state"] == "cancelled"


def test_it_needs_a_training_and_one_of_its_models(client, columns):
    none = client.post(URL, json={})
    assert none.status_code == 400 and none.get_json()["error"]["code"] == "NOT_TRAINED"
    _train(client, columns)
    other = client.post(URL, json={"model": "random_forest"})
    assert other.status_code == 400 and "no model 'random_forest'" in \
        other.get_json()["error"]["message"]


def test_a_saved_runs_results_are_read_only(client, columns):
    run_id = _train(client, columns)["run_id"]
    assert client.get(f"/api/v1/runs/{run_id}/importance").get_json() == {
        "run_id": run_id, "importances": [], "left_out": 0}, "none computed, none made"
    missing = client.get("/api/v1/runs/20261030T120000Z-0000000a/importance")
    assert missing.status_code == 404 and missing.get_json()["error"]["code"] == "NOT_FOUND"
    bad = client.get("/api/v1/runs/not-a-run/importance")
    assert bad.status_code == 400 and bad.get_json()["error"]["message"] == "That is not a run id."


def _why_is_whole(answer):
    why = answer["why"]
    assert why["prediction"] == answer["prediction"] and why["problem"] is None
    assert 0 <= why["probability"] <= 1 and why["source"] == "training"
    assert len(why["toward"]) <= 5 and len(why["away"]) <= 5
    assert all(r["effect"] > 0 for r in why["toward"]) and all(r["effect"] < 0 for r in why["away"])
    assert set(why["toward"][0]) == {"feature", "value", "typical", "effect"}


def test_a_prediction_says_why(client, columns, fixtures_dir):
    _train(client, columns)
    answer = client.post("/api/v1/run/predict",
                         json={"values": _row(fixtures_dir), "model": "decision_tree"})
    assert answer.status_code == 200, answer.get_json()
    _why_is_whole(answer.get_json())


def test_a_loaded_bundle_says_why(client, columns, fixtures_dir):
    _train(client, columns)
    assert client.post("/api/v1/run/export", json={"name": "why"}).status_code == 200
    data = client.get("/api/v1/run/bundle").data
    loaded = client.post("/api/v1/bundle", data={"file": (io.BytesIO(data), "why.bundle")},
                         content_type="multipart/form-data")
    assert loaded.status_code == 200, loaded.get_json()
    answer = client.post("/api/v1/bundle/predict",
                         json={"values": _row(fixtures_dir), "model": "decision_tree"})
    assert answer.status_code == 200, answer.get_json()
    _why_is_whole(answer.get_json())


def test_a_model_without_probabilities_predicts_and_says_why_not(client, columns, fixtures_dir):
    _train(client, columns, models=("svm",))
    answer = client.post("/api/v1/run/predict", json={"values": _row(fixtures_dir)})
    assert answer.status_code == 200, answer.get_json()
    why = answer.get_json()["why"]
    assert why["problem"] == "NO_PROBABILITIES" and why["probability"] is None
    assert why["toward"] == [] and why["away"] == []


def test_a_missing_number_is_no_reason(client, columns, fixtures_dir):
    """"nan" is a number the pipeline imputes with its typical value, so it changes nothing
    and is no reason; the answer stays JSON (no NaN)."""
    _train(client, columns)
    values = _row(fixtures_dir)
    empty = next(iter(values))
    values[empty] = "nan"
    answer = client.post("/api/v1/run/predict", json={"values": values, "model": "decision_tree"})
    assert answer.status_code == 200, answer.get_json()
    assert "NaN" not in answer.get_data(as_text=True)
    reasons = answer.get_json()["why"]["toward"] + answer.get_json()["why"]["away"]
    assert empty not in [r["feature"] for r in reasons]
