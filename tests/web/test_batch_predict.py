"""v5.1 step 1.4f (plan F1): "Predict a CSV file" with the session's run or a loaded bundle;
the labelled CSV is the CLI's (``toolkit.ids_core.batch``)."""

import io

import pandas as pd
import pytest

from toolkit.data.ids import bundle_files
from toolkit.ids_core.batch import prediction_summary
from toolkit.ids_core.datasets import load_dataset
from web_jobs import train_and_wait

CIC = "cic_ddos_sample.csv"
TARGET = "Label"
MODELS = ("decision_tree", "logistic_regression")


@pytest.fixture
def trained(client, upload, fixtures_dir):
    columns = upload(fixtures_dir / CIC).get_json()["columns"]
    response = train_and_wait(client, {
        "features": [c for c in columns if c != TARGET], "target": TARGET, "balancing": "none",
        "test_size": 0.3, "models": list(MODELS)})
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def _send(client, url, data, name, model=None):
    form = {"file": (io.BytesIO(data), name)}
    if model:
        form["model"] = model
    return client.post(url, data=form, content_type="multipart/form-data")


def _fixture(fixtures_dir):
    return (fixtures_dir / CIC).read_bytes()


def _csv(response):
    return pd.read_csv(io.BytesIO(response.data))


def _uploads_left(web_app):
    return list(web_app.extensions["toolkit"].upload_dir.iterdir())


def test_run_predict_file_counts_and_download(client, web_app, trained, fixtures_dir):
    response = _send(client, "/api/v1/run/predict-file", _fixture(fixtures_dir), CIC)
    assert response.status_code == 200, response.get_json()
    answer = response.get_json()
    assert (answer["source"], answer["model"], answer["rows"]) == ("run", trained["selected"], 200)
    assert answer["message"].startswith("200 rows predicted with ")
    counts = answer["counts"]
    assert sum(c["count"] for c in counts) == 200
    assert counts == sorted(counts, key=lambda c: (-c["count"], c["label"]))
    assert abs(sum(c["share"] for c in counts) - 1) < 1e-9
    assert answer["label_column"] == TARGET and answer["matches"]["rows"] == 200
    assert (answer["download_url"], answer["download_name"]) == (
        "/api/v1/predictions", "cic_ddos_sample.predictions.csv")

    download = client.get("/api/v1/predictions")
    assert download.status_code == 200 and download.mimetype == "text/csv"
    assert "cic_ddos_sample.predictions.csv" in download.headers["Content-Disposition"]
    table = _csv(download)
    assert list(table.columns) == ["row", "prediction", "confidence", "true_label"]
    assert table["row"].tolist() == list(range(1, 201))
    labels = load_dataset(fixtures_dir / CIC).frame[TARGET].astype(str).tolist()
    assert table["true_label"].tolist() == labels
    assert dict(table["prediction"].value_counts()) == {c["label"]: c["count"] for c in counts}
    assert int((table["prediction"] == table["true_label"]).sum()) == \
        answer["matches"]["matching"]
    assert _uploads_left(web_app) == [], "the uploaded file is deleted once read (S11)"


def test_chosen_model_and_same_as_one_row_predictions(client, trained, fixtures_dir):
    response = _send(client, "/api/v1/run/predict-file", _fixture(fixtures_dir), CIC,
                     model="logistic_regression")
    assert response.get_json()["model"] == "logistic_regression"
    table = _csv(client.get("/api/v1/predictions"))
    frame = load_dataset(fixtures_dir / CIC).frame
    for index in (0, 1, 100, 199):
        values = {c: str(v) for c, v in frame.iloc[index].items() if c != TARGET}
        one = client.post("/api/v1/run/predict",
                          json={"values": values, "model": "logistic_regression"}).get_json()
        assert one["prediction"] == table["prediction"][index]


def test_file_without_the_label_column(client, trained, fixtures_dir):
    frame = load_dataset(fixtures_dir / CIC).frame.drop(columns=[TARGET])
    answer = _send(client, "/api/v1/run/predict-file", frame.to_csv(index=False).encode(),
                   "flows.csv").get_json()
    assert answer["matches"] is None and answer["label_column"] is None
    assert answer["download_name"] == "flows.predictions.csv"
    assert list(_csv(client.get("/api/v1/predictions")).columns) == ["row", "prediction",
                                                                     "confidence"]


def test_missing_feature_columns_are_named(client, web_app, trained, fixtures_dir):
    frame = load_dataset(fixtures_dir / CIC).frame
    dropped = [c for c in frame.columns if c != TARGET][:2]
    response = _send(client, "/api/v1/run/predict-file",
                     frame.drop(columns=dropped).to_csv(index=False).encode(), "short.csv")
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert error["code"] == "UNKNOWN_COLUMN"
    assert all(name in error["message"] for name in dropped)
    assert _uploads_left(web_app) == []


def test_bundle_predict_file_same_as_run(client, trained, fixtures_dir):
    run = _send(client, "/api/v1/run/predict-file", _fixture(fixtures_dir), CIC)
    run_csv = client.get("/api/v1/predictions").data
    assert client.post("/api/v1/run/export", json={"name": "m"}).status_code == 200
    bundle = client.get("/api/v1/run/bundle").data
    loaded = client.post("/api/v1/bundle", data={"file": (io.BytesIO(bundle), "m.bundle")},
                         content_type="multipart/form-data")
    assert loaded.status_code == 200
    answer = _send(client, "/api/v1/bundle/predict-file", _fixture(fixtures_dir), CIC).get_json()
    assert answer["source"] == "bundle"
    assert {k: v for k, v in answer.items() if k != "source"} == \
        {k: v for k, v in run.get_json().items() if k != "source"}
    assert client.get("/api/v1/predictions").data == run_csv


def test_web_file_is_the_cli_file(client, trained, fixtures_dir, tmp_path, capsys):
    """The same bundle and rows give the same bytes from the web and from --csv."""
    from toolkit.ids_cli import cli

    exported = client.post("/api/v1/run/export", json={"name": "m"}).get_json()
    bundle_path = bundle_files.bundles_dir() / exported["file"]
    bundle = cli.load_model(str(bundle_path))
    loaded = client.post("/api/v1/bundle",
                         data={"file": (io.BytesIO(bundle_path.read_bytes()), "m.bundle")},
                         content_type="multipart/form-data")
    assert loaded.status_code == 200
    for model in MODELS:
        output = tmp_path / f"{model}.csv"
        assert cli.run_csv(bundle, fixtures_dir / CIC, output, model) == 0
        _send(client, "/api/v1/bundle/predict-file", _fixture(fixtures_dir), CIC, model=model)
        assert client.get("/api/v1/predictions").data == output.read_bytes(), model
    short = tmp_path / "short.csv"
    frame = load_dataset(fixtures_dir / CIC).frame
    frame.drop(columns=[c for c in frame.columns if c != TARGET][:1]).to_csv(short, index=False)
    with pytest.raises(ValueError, match="Missing feature column"):
        cli.run_csv(bundle, short, tmp_path / "short.out.csv")
    capsys.readouterr()


def test_errors(client, web_app, upload, fixtures_dir, trained):
    def code(response):
        return response.status_code, response.get_json()["error"]["code"]

    data = _fixture(fixtures_dir)
    assert code(_send(client, "/api/v1/bundle/predict-file", data, CIC)) == (400,
                                                                              "NO_MODEL_LOADED")
    assert code(_send(client, "/api/v1/run/predict-file", data, CIC, model="svm")) == \
        (400, "BAD_REQUEST")
    assert code(_send(client, "/api/v1/run/predict-file", data, "flows.json")) == \
        (400, "BAD_REQUEST")
    assert code(client.post("/api/v1/run/predict-file", data={},
                            content_type="multipart/form-data")) == (400, "BAD_REQUEST")
    unreadable = _send(client, "/api/v1/run/predict-file", b"\x00\xff\xfe" * 100, "bad.csv")
    assert code(unreadable) == (400, "UNREADABLE_FILE")
    assert "bad.csv" in unreadable.get_json()["error"]["message"]
    assert _uploads_left(web_app) == []


def test_not_trained(client, fixtures_dir):
    response = _send(client, "/api/v1/run/predict-file", _fixture(fixtures_dir), CIC)
    assert (response.status_code, response.get_json()["error"]["code"]) == (400, "NOT_TRAINED")


def test_too_large(make_app, monkeypatch, fixtures_dir):
    monkeypatch.setenv("TOOLKIT_MAX_PREDICT_MB", "0")
    app = make_app()
    with app.test_client() as client:
        columns = client.post("/api/v1/dataset",
                              data={"dataset": (io.BytesIO(_fixture(fixtures_dir)), CIC)},
                              content_type="multipart/form-data").get_json()["columns"]
        assert train_and_wait(client, {"features": [c for c in columns if c != TARGET],
                                       "target": TARGET, "balancing": "none", "test_size": 0.3,
                                       "models": ["decision_tree"]}).status_code == 200
        response = _send(client, "/api/v1/run/predict-file", _fixture(fixtures_dir), CIC)
    assert response.status_code == 413
    error = response.get_json()["error"]
    assert error["code"] == "TOO_LARGE" and "TOOLKIT_MAX_PREDICT_MB" in error["message"]
    assert list(app.extensions["toolkit"].upload_dir.iterdir()) == []


def test_download_only_after_a_batch_and_kept_until_cleared(client, trained, fixtures_dir):
    before = client.get("/api/v1/predictions")
    assert (before.status_code, before.get_json()["error"]["code"]) == (404, "NOT_FOUND")
    first = _send(client, "/api/v1/run/predict-file", _fixture(fixtures_dir), CIC).get_json()
    assert client.get("/api/v1/workspace").get_json()["predictions"] == first
    frame = load_dataset(fixtures_dir / CIC).frame.head(10)
    second = _send(client, "/api/v1/run/predict-file", frame.to_csv(index=False).encode(),
                   "ten.csv").get_json()
    assert second["rows"] == 10 and len(_csv(client.get("/api/v1/predictions"))) == 10
    assert client.get("/api/v1/workspace").get_json()["predictions"] == second
    assert client.delete("/api/v1/workspace").status_code == 200
    assert client.get("/api/v1/predictions").status_code == 404


def test_summary_order_and_matches():
    table = pd.DataFrame({"row": [1, 2, 3, 4], "prediction": ["b", "a", "b", "a"],
                          "confidence": [1.0] * 4, "true_label": ["b", "a", "a", "a"]})
    summary = prediction_summary(table)
    assert summary["counts"] == [{"label": "a", "count": 2, "share": 0.5},
                                 {"label": "b", "count": 2, "share": 0.5}]  # ties by name
    assert summary["matches"] == {"matching": 3, "rows": 4, "share": 0.75}
    assert prediction_summary(table.drop(columns=["true_label"]))["matches"] is None
    uneven = prediction_summary(table.assign(prediction=["a", "b", "b", "b"]))
    assert [c["label"] for c in uneven["counts"]] == ["b", "a"], "largest first"


def test_page_has_the_forms(repo_root):
    page = (repo_root / "toolkit/web/templates/dynamic_ids.html").read_text(encoding="utf-8")
    for prefix in ("predict", "loadedPredict"):
        assert f'id="{prefix}FileForm"' in page and f'for="{prefix}File"' in page
        assert f'id="{prefix}FileResult" class="message" role="status"' in page
        assert f'id="{prefix}FileOutput"' in page
