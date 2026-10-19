"""Step 0.3 interim trust rule: web model load/export are paused until 0.4g (S1, S4)."""

import io
import pickle
from pathlib import Path


def test_uploaded_pickle_never_deserialized(app_module, client, monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("pickle was deserialized")

    monkeypatch.setattr(pickle, "load", _forbidden)
    monkeypatch.setattr(pickle, "loads", _forbidden)
    payload = pickle.dumps({"model": "anything"})
    response = client.post("/load_model", data={"model_file": (io.BytesIO(payload), "m.pkl")},
                           content_type="multipart/form-data")
    assert response.status_code == 503
    assert "paused" in response.get_json()["error"].lower()


def test_export_and_load_paused_with_message(app_module, client):
    models_dir = app_module.models_dir
    before = {p.name for p in Path(models_dir).glob("*")}
    response = client.post("/export_model", json={"model_name": "x"})
    assert response.status_code == 503
    assert "paused" in response.get_json()["error"].lower()
    after = {p.name for p in Path(models_dir).glob("*")}
    assert after == before, "export must not write any file while paused"


def test_predict_loaded_without_model_is_a_clear_error(client):
    response = client.post("/predict_loaded/", json={"values": {"a": 1}})
    assert response.status_code == 400
    assert "no model" in response.get_json()["error"].lower()


def test_in_session_train_predict_still_work(client, upload, fixtures_dir):
    columns = upload(fixtures_dir / "cic_ddos_sample.csv").get_json()["columns"]
    target = " Label"
    features = [c for c in columns if c != target]
    trained = client.post("/configure", json={
        "input_features": features, "target": target, "balancing_method": "none",
        "test_size": 0.3, "selected_models": ["decision_tree"],
    })
    assert trained.status_code == 200, trained.get_json()
    values = {feature: "0" for feature in features}
    predicted = client.post("/predict", json={"values": values, "algorithm": "decision_tree"})
    assert predicted.status_code == 200, predicted.get_json()
    assert predicted.get_json()["prediction"] in {"BENIGN", "DDoS"}
