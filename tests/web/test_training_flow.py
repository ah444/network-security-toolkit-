"""Step 0.4g: the web trainer runs on the v5 core (0.4a–f) [L2, L3, L8, B2, B3, B4, S1].

These go through the real Flask routes with real fixture files, as the browser does.
"""

import io
import json
import pickle

import joblib
import pytest

from toolkit import bundle_files
from toolkit.ids_core.bundle import pack
from toolkit.ids_core.datasets import load_dataset
from toolkit.web import datasets, training
from web_jobs import train_and_wait
from web_js import all_js

CIC = "cic_ddos_sample.csv"
TARGET = "Label"  # the v5 loader strips the header's leading space (B17)


@pytest.fixture
def cic(upload, fixtures_dir):
    response = upload(fixtures_dir / CIC)
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def _features(columns):
    return [c for c in columns if c != TARGET]


def _train(client, columns, models=("decision_tree",), balancing="none", test_size=0.3):
    return train_and_wait(client, {
        "features": _features(columns), "target": TARGET, "balancing": balancing,
        "test_size": test_size, "models": list(models)})


def _rows(fixtures_dir, count=6):
    """Real rows of the fixture as the prediction form sends them (text values)."""
    frame = load_dataset(fixtures_dir / CIC).frame
    picked = frame.iloc[[0, 1, 2, -3, -2, -1][:count]]
    return [({c: str(v) for c, v in row.items() if c != TARGET}, row[TARGET])
            for _, row in picked.iterrows()]


# --- upload and training --------------------------------------------------------------------

def test_upload_uses_v5_loader(cic):
    assert TARGET in cic["columns"] and " Label" not in cic["columns"]
    profile = cic["dataset"]["profile"]
    assert (profile["name"], profile["label_column"]) == ("cic_ids2017", TARGET)
    assert cic["dataset"]["rows"] == 200 and cic["dataset"]["test_file"] is None


def test_end_to_end_training_uses_pipeline(client, cic, monkeypatch):
    calls = []
    real_train = training.train_run

    def spy(frame, **kwargs):
        calls.append((len(frame), kwargs))
        return real_train(frame, **kwargs)

    monkeypatch.setattr(training, "train_run", spy)
    response = _train(client, cic["columns"], models=("decision_tree", "logistic_regression"),
                      balancing="recommended")
    assert response.status_code == 200, response.get_json()
    result = response.get_json()
    assert len(calls) == 1 and calls[0][0] == 200
    settings = calls[0][1]["settings"]
    assert settings.label == TARGET and settings.balancer == "recommended"
    assert settings.split.test_size == 0.3

    assert {row["model"] for row in result["leaderboard"]} == {"decision_tree",
                                                                "logistic_regression"}
    assert result["selected"] in {"decision_tree", "logistic_regression"}
    split = result["split"]
    assert split["strategy"] == "stratified"
    assert "STRATIFIED_SCORES_MAY_BE_OPTIMISTIC" in split["warnings"]
    assert split["train_rows"] + split["test_rows"] == split["dedupe"]["rows_after"]
    assert result["cv"]["folds"] == 3
    assert result["balancing"]["method"] in {"none", "class_weight", "smote"}
    assert set(result["confusion_matrices"]) == {"decision_tree", "logistic_regression"}
    assert [f["name"] for f in result["features_info"]] == _features(cic["columns"])
    assert "validation" not in json.dumps(result).lower(), "no v4 validation split any more"


def test_old_preprocessing_code_deleted(repo_root):
    for path in sorted((repo_root / "toolkit" / "web").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for name in ("train_test_split", "RobustScaler", "LabelEncoder", "SMOTE(", "SelectKBest",
                     "recommend_balancing_method", "import dask", "MODEL_FILES_PAUSED",
                     "unique_export_path", "df_balanced", "smote_applied", "label_encoders",
                     "pickle"):
            assert name not in source, f"old code still in {path.name}: {name}"


def test_training_errors_are_clear(client, cic):
    unknown = train_and_wait(client, {
        "features": _features(cic["columns"]), "target": TARGET,
        "models": ["quantum_forest"]})
    assert unknown.status_code == 400 and "quantum_forest" in unknown.get_json()["error"]["message"]
    label_as_feature = train_and_wait(client, {
        "features": cic["columns"], "target": TARGET, "models": ["decision_tree"]})
    assert label_as_feature.status_code == 400


def test_training_without_dataset(client):
    response = train_and_wait(client, {"features": ["a"], "target": TARGET})
    assert response.status_code == 400 and response.get_json()["error"]["code"] == "NO_DATASET"


def test_nslkdd_official_pair_uses_official_split(client, fixtures_dir):
    train_path = fixtures_dir / "nslkdd_train_sample.txt"
    test_path = fixtures_dir / "nslkdd_test_sample_noheader.txt"
    with train_path.open("rb") as train_file, test_path.open("rb") as test_file:
        uploaded = client.post("/api/v1/dataset", data={"dataset": [(train_file, train_path.name),
                                                            (test_file, test_path.name)]},
                               content_type="multipart/form-data")
    assert uploaded.status_code == 200, uploaded.get_json()
    dataset = uploaded.get_json()["dataset"]
    assert dataset["test_file"] == test_path.name and dataset["test_rows"] == 100
    assert dataset["profile"]["split_strategy"] == "official"
    columns = uploaded.get_json()["columns"]
    response = train_and_wait(client, {
        "features": [c for c in columns if c != "attack"], "target": "attack",
        "models": ["decision_tree"], "balancing": "none"})
    assert response.status_code == 200, response.get_json()
    result = response.get_json()
    assert (result["split"]["strategy"], result["split"]["official_mode"]) == (
        "official", "as_published")
    assert result["split"]["test_rows"] == 100
    assert "level" in {b["column"] for b in result["leakage"]["blocked"]}


# --- train-only analysis ---------------------------------------------------------------------

def test_class_analysis_recommendation_from_train_split(client, cic, monkeypatch):
    seen = []
    real = datasets.recommend_balancing

    def spy(labels, num_features):
        seen.append(len(labels))
        return real(labels, num_features)

    monkeypatch.setattr(datasets, "recommend_balancing", spy)
    result = client.get("/api/v1/dataset/distribution",
                         query_string={"target": TARGET, "test_size": 0.3}).get_json()
    assert sum(result["class_counts"].values()) == result["total_samples"] == 200
    assert seen == [result["split"]["train_rows"]] and seen[0] < 200
    assert result["recommendation"]["train_rows"] == seen[0]
    assert result["recommendation"]["recommended"]


def test_apply_smote_is_train_only_preview(client, cic):
    before = client.get("/api/v1/dataset/preview", query_string={"rows": 5}).get_json()["stats"]
    response = client.post("/api/v1/dataset/balancing-preview", json={
        "target": TARGET, "balancing": "smote", "test_size": 0.3,
        "features": _features(cic["columns"])})
    assert response.status_code == 200, response.get_json()
    preview = response.get_json()
    assert sum(preview["before"].values()) == preview["train_rows"] == preview["rows"] < 200
    assert len(set(preview["after"].values())) == 1  # SMOTE grows every class to the majority
    after = client.get("/api/v1/dataset/preview", query_string={"rows": 5}).get_json()["stats"]
    assert after == before, "the preview must not change the dataset"
    assert client.post("/reset_smote", json={}).status_code == 404


def test_recommend_features_train_only_and_guarded(client, upload, fixtures_dir):
    upload(fixtures_dir / "label_leak.csv")
    result = client.post("/api/v1/dataset/feature-scores",
                         json={"target": "Label", "k": 3, "test_size": 0.3}).get_json()
    blocked = {b["column"] for b in result["blocked"]}
    assert {"row_id", "xyz_col", "enc_col", "pre_col"} <= blocked
    names = [name for name, _ in result["recommended_features"]]
    assert len(names) == 3 and not blocked & set(names)
    assert names[0] == "f_strong"
    assert result["rows"] == result["train_rows"] < 300


# --- predicting ---------------------------------------------------------------------------------

def test_session_predict_uses_saved_pipeline(client, cic, fixtures_dir):
    assert _train(client, cic["columns"]).status_code == 200
    for values, _ in _rows(fixtures_dir):
        result = client.post("/api/v1/run/predict", json={"values": values, "model": "decision_tree"})
        assert result.status_code == 200, result.get_json()
        assert result.get_json()["prediction"] in {"BENIGN", "DDoS"}
    values = dict(_rows(fixtures_dir, 1)[0][0])
    values[_features(cic["columns"])[0]] = "fast"
    bad = client.post("/api/v1/run/predict", json={"values": values, "model": "decision_tree"})
    assert bad.status_code == 400 and "must be a number" in bad.get_json()["error"]["message"]
    unknown = client.post("/api/v1/run/predict", json={"values": values, "model": "svm"})
    assert unknown.status_code == 400


def _export(client, name="web_model"):
    response = client.post("/api/v1/run/export", json={"name": name})
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def _load(client, data, name="web_model.bundle"):
    return client.post("/api/v1/bundle", data={"file": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_loaded_bundle_predicts_same_as_session_model(client, cic, fixtures_dir):
    trained = _train(client, cic["columns"], models=("decision_tree", "logistic_regression"))
    assert trained.status_code == 200
    exported = _export(client)
    download = client.get("/api/v1/run/bundle")
    assert download.status_code == 200
    assert "web_model.bundle" in download.headers["Content-Disposition"]
    loaded = _load(client, download.data)
    assert loaded.status_code == 200, loaded.get_json()
    info = loaded.get_json()
    assert info["run_id"] == exported["run_id"] and info["selected"] == trained.get_json()[
        "selected"]
    assert [f["name"] for f in info["features_info"]] == _features(cic["columns"])
    for algorithm in ("decision_tree", "logistic_regression"):
        for values, _ in _rows(fixtures_dir):
            session = client.post("/api/v1/run/predict", json={"values": values, "model": algorithm})
            from_file = client.post("/api/v1/bundle/predict",
                                    json={"values": values, "model": algorithm})
            assert from_file.status_code == 200, from_file.get_json()
            assert from_file.get_json()["prediction"] == session.get_json()["prediction"]


def test_export_writes_bundle_and_run_record(client, cic):
    assert _train(client, cic["columns"]).status_code == 200
    exported = _export(client)
    bundle_path = bundle_files.bundles_dir() / exported["file"]
    assert bundle_path.is_file()
    record_path = bundle_files.instance_dir() / "runs" / f"{exported['run_id']}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["bundle_sha256"] == exported["sha256"]
    assert str(bundle_files.instance_dir()) not in json.dumps(exported), "no server paths"


def test_web_load_accepts_signed_rejects_other(workspaces, client, cic, monkeypatch):
    assert _train(client, cic["columns"]).status_code == 200
    _export(client)
    good = client.get("/api/v1/run/bundle").data

    def forbidden(*args, **kwargs):
        raise AssertionError("an untrusted file was deserialized")

    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    real_joblib_load = joblib.load
    monkeypatch.setattr(joblib, "load", forbidden)
    tampered = bytearray(good)
    tampered[len(tampered) // 2] ^= 1
    foreign = pack(_session_run(workspaces), bytes(range(32)))
    cases = {"pickle": (pickle.dumps({"model": "anything"}), "NOT_A_BUNDLE"),
             "tampered": (bytes(tampered), "BAD_SIGNATURE"),
             "foreign": (foreign, "BAD_SIGNATURE"),
             "empty": (b"", "EMPTY")}
    for name, (data, code) in cases.items():
        response = _load(client, data, f"{name}.bundle")
        assert response.status_code == 400, name
        assert response.get_json()["error"]["code"] == code, name
    monkeypatch.setattr(joblib, "load", real_joblib_load)
    assert _load(client, good).status_code == 200


def _session_run(workspaces):
    """The run trained in the (only) workspace of this test."""
    (workspace,) = workspaces.values()
    return workspace.run


def test_predict_loaded_route(client, cic, fixtures_dir):
    values = _rows(fixtures_dir, 1)[0][0]
    empty = client.post("/api/v1/bundle/predict", json={"values": values})
    assert empty.status_code == 400 and empty.get_json()["error"]["code"] == "NO_MODEL_LOADED"
    assert client.post("/api/v1/bundle/predict/", json={"values": values}).status_code == 404
    assert _train(client, cic["columns"]).status_code == 200
    _export(client)
    assert _load(client, client.get("/api/v1/run/bundle").data).status_code == 200
    response = client.post("/api/v1/bundle/predict", json={"values": values})  # default: selected
    assert response.status_code == 200
    assert response.get_json()["prediction"] in {"BENIGN", "DDoS"}


def test_js_has_no_dead_loaded_model_config():
    js = all_js(code_only=False)
    assert "loaded_model_config" not in js
    assert '"/api/v1/bundle/predict"' in js and '"/api/v1/bundle/predict/"' not in js


def test_download_needs_an_export(client):
    assert client.get("/api/v1/run/bundle").status_code == 404
