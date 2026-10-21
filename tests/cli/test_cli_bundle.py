"""Step 0.5: the CLI reads any bundle; the default model is a pinned web-trainer export.

[D5, D12, B14, L1, B21]
"""

import builtins
import json
import os
import pickle
import subprocess
import sys
import types

import joblib
import pandas as pd
import pytest

from toolkit import bundle_files
from toolkit.ids_core import bundle as core_bundle
from toolkit.ids_core.datasets import load_dataset

MODELS = "tools/IDS/IDS_CLI - CIC-2017/models"
CLI_FILE = "tools/IDS/IDS_CLI - CIC-2017/IDS_with_Machine_Learning.py"
CIC = "cic_ddos_sample.csv"
TARGET = "Label"


@pytest.fixture
def exported(client, upload, fixtures_dir):
    """A bundle trained and exported through the real web routes (this install's key)."""
    columns = upload(fixtures_dir / CIC).get_json()["columns"]
    trained = client.post("/api/v1/run", json={
        "features": [c for c in columns if c != TARGET], "target": TARGET,
        "balancing": "none", "models": ["decision_tree", "logistic_regression"]})
    assert trained.status_code == 200, trained.get_json()
    export = client.post("/api/v1/run/export", json={"name": "web"}).get_json()
    return bundle_files.bundles_dir() / export["file"]


@pytest.fixture
def frame(fixtures_dir):
    return load_dataset(fixtures_dir / CIC).frame


# --- any bundle -------------------------------------------------------------------------------

def test_cli_loads_web_export(cli_module, exported):
    bundle = cli_module.load_model(exported)
    assert bundle.trust == "signature"
    assert set(bundle.models) == {"decision_tree", "logistic_regression"}
    assert len(bundle.features) == 14 and set(bundle.classes) == {"BENIGN", "DDoS"}
    card = cli_module.model_card(bundle)
    assert bundle.run_id in card and "cic_ddos_sample.csv" in card
    assert "macro-F1" in card


def test_cli_and_web_same_prediction(cli_module, client, exported, frame):
    bundle = cli_module.load_model(exported)
    rows = frame.iloc[[0, 1, 2, 100, 150, 199]]
    for model in ("decision_tree", "logistic_regression"):
        cli = cli_module.predict_rows(bundle, rows, model=model)
        for (_, row), predicted in zip(rows.iterrows(), cli["prediction"], strict=True):
            values = {c: str(v) for c, v in row.items() if c != TARGET}
            web = client.post("/api/v1/run/predict", json={"values": values, "model": model})
            assert web.get_json()["prediction"] == predicted, model


def test_cli_csv_batch(cli_module, exported, fixtures_dir, tmp_path, capsys):
    output = tmp_path / "out.csv"
    code = cli_module.main(["--bundle", str(exported), "--csv", str(fixtures_dir / CIC),
                            "--output", str(output)])
    assert code == 0
    result = pd.read_csv(output)
    assert list(result.columns) == ["row", "prediction", "confidence", "true_label"]
    assert len(result) == 200 and set(result["prediction"]) <= {"BENIGN", "DDoS"}
    out = capsys.readouterr().out
    assert "200 rows" in out and "match the file's Label column" in out
    before = output.read_bytes()
    again = cli_module.main(["--bundle", str(exported), "--csv", str(fixtures_dir / CIC),
                             "--output", str(output)])
    assert again == 2 and "already exists" in capsys.readouterr().out
    assert output.read_bytes() == before, "a CSV result is never overwritten"


def test_cli_csv_missing_features(cli_module, exported, frame, tmp_path, capsys):
    partial = tmp_path / "partial.csv"
    frame.drop(columns=["Idle Max"]).to_csv(partial, index=False)
    code = cli_module.main(["--bundle", str(exported), "--csv", str(partial)])
    assert code == 2 and "Idle Max" in capsys.readouterr().out


def test_cli_json(cli_module, exported, frame, capsys):
    row = {c: v.item() if hasattr(v, "item") else v
           for c, v in frame.iloc[0].items() if c != TARGET}
    assert cli_module.main(["--bundle", str(exported), "--json", json.dumps(row)]) == 0
    (result,) = json.loads(capsys.readouterr().out)
    assert result["prediction"] in {"BENIGN", "DDoS"} and result["model"]
    assert 0.0 <= result["confidence"] <= 1.0
    many = json.dumps([row, row])
    assert cli_module.main(["--bundle", str(exported), "--json", many,
                            "--model", "logistic_regression"]) == 0
    results = json.loads(capsys.readouterr().out)
    assert len(results) == 2 and results[0]["model"] == "logistic_regression"
    bad = dict(row, **{"Idle Max": "fast"})
    assert cli_module.main(["--bundle", str(exported), "--json", json.dumps(bad)]) == 2
    assert "'Idle Max' must be a number" in capsys.readouterr().out


def test_cli_refuses_untrusted_files(cli_module, exported, tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("an untrusted file was deserialized")

    tampered = tmp_path / "tampered.bundle"
    data = bytearray(exported.read_bytes())
    data[len(data) // 2] ^= 1
    tampered.write_bytes(bytes(data))
    pickled = tmp_path / "model.pkl"
    pickled.write_bytes(pickle.dumps({"model": 1}))
    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    for path, words in ((tampered, "signature"), (pickled, "not a toolkit bundle")):
        assert cli_module.main(["--bundle", str(path), "--json", "{}"]) == 2
        assert words in capsys.readouterr().out


def test_cli_output_redirected(exported, repo_root):
    """Redirected output is cp1252 on Windows by default: the emoji must not crash it."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    result = subprocess.run([sys.executable, str(repo_root / CLI_FILE), "--bundle", str(exported),
                             "--info"], capture_output=True, env=env, timeout=120)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert "📦 Model bundle" in result.stdout.decode("utf-8")


def _answers(monkeypatch, answers):
    remaining = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(remaining))
    return remaining


def test_interactive_examples_manual_and_exit(cli_module, exported, monkeypatch, capsys):
    bundle = cli_module.load_model(exported)
    examples = bundle.manifest["examples"]
    # default model; 2 = real example #1; 1 = manual input from example #1, keep every
    # value; 3 = switch to model 2; 5 = exit
    answers = ["", "2", "1", "1", "", *[""] * len(bundle.features), "3", "2", "5"]
    remaining = _answers(monkeypatch, answers)
    assert cli_module.main(["--bundle", str(exported)]) == 0
    assert list(remaining) == []
    out = capsys.readouterr().out
    assert out.count("PREDICTION RESULT") == 2
    assert f"True label: {examples[0]['label']}" in out
    assert "Goodbye" in out


# --- choosing among this install's exports (no model is shipped; plan D12 revised) -------------

@pytest.fixture
def two_exports(client, exported):
    """``exported`` plus a newer export of a second run (decision tree only)."""
    columns = list(cli_frame_columns(exported))
    trained = client.post("/api/v1/run", json={
        "features": columns, "target": TARGET, "balancing": "none",
        "models": ["decision_tree"]})
    assert trained.status_code == 200, trained.get_json()
    newer = client.post("/api/v1/run/export", json={"name": "newer"}).get_json()
    return exported, bundle_files.bundles_dir() / newer["file"]


def cli_frame_columns(path):
    return core_bundle.peek_manifest(path.read_bytes())["features"]


def test_exports_listed_newest_first_from_headers(two_exports, monkeypatch):
    older, newer = two_exports
    (bundle_files.bundles_dir() / "notes.bundle").write_bytes(b"not a bundle")

    def forbidden(*args, **kwargs):
        raise AssertionError("listing must not deserialize models")

    monkeypatch.setattr(joblib, "load", forbidden)
    exports = bundle_files.list_bundles()
    assert [e.path for e in exports] == [newer, older], "newest first; junk files skipped"
    assert exports[0].manifest["models"] == ["decision_tree"]
    assert exports[0].size == newer.stat().st_size
    # only the header is read: a bundle cut off after its manifest is still listed
    data = older.read_bytes()
    body = int.from_bytes(data[len(core_bundle.MAGIC):len(core_bundle.MAGIC) + 8], "big")
    cut = bundle_files.bundles_dir() / "cut.bundle"
    cut.write_bytes(data[:len(core_bundle.MAGIC) + 8 + body + 10])
    assert bundle_files.read_manifest_header(cut) == core_bundle.peek_manifest(data)


def test_cli_without_exports(cli_module, capsys):
    assert cli_module.main(["--info"]) == 2
    assert "No exported model yet" in capsys.readouterr().out
    assert cli_module.main(["--list"]) == 0
    assert "No exported model yet" in capsys.readouterr().out


def test_cli_uses_newest_export_by_default(cli_module, two_exports, capsys):
    older, newer = two_exports
    assert cli_module.main(["--info"]) == 0
    assert newer.stem in capsys.readouterr().out
    assert cli_module.main(["--list"]) == 0
    listing = capsys.readouterr().out
    assert listing.index(newer.stem) < listing.index(older.stem)
    assert "Decision Tree" in listing and "cic_ddos_sample.csv" in listing


def test_interactive_chooses_an_export(cli_module, two_exports, monkeypatch, capsys):
    older, newer = two_exports
    # 2 = the older export; Enter = its best model; 5 = exit
    remaining = _answers(monkeypatch, ["2", "", "5"])
    assert cli_module.main([]) == 0
    assert list(remaining) == []
    out = capsys.readouterr().out
    assert newer.stem in out and f"run {older.stem}" in out and "Goodbye" in out


def test_nothing_is_shipped_with_the_cli(repo_root):
    models = repo_root / MODELS
    assert sorted(p.name for p in models.iterdir()) == ["README.md"]
    assert not (repo_root / "scripts" / "pin_default_model.py").exists()
    card = (models / "README.md").read_text(encoding="utf-8")
    assert "Export" in card and "instance/bundles" in card


def test_no_v4_artifacts_remain(repo_root):
    models = repo_root / MODELS
    assert not list(models.glob("*.pkl")), "v4 pickles must be deleted"
    assert not (models / "config.json").exists(), "v4 config.json lists the leaky feature"
    source = (repo_root / CLI_FILE).read_text(encoding="utf-8")
    for leftover in ("Attempted Category", "pickle", "scaler.pkl", "label_encoder"):
        assert leftover not in source, leftover


def test_prediction_status_names_normal_traffic(cli_module, capsys):
    """CIC-IDS2017 calls normal flows BENIGN, NSL-KDD calls them normal (real labels)."""
    def status(label, classes):
        cli_module.display_prediction({"prediction": label, "confidence": None, "top": []},
                                      classes=classes)
        return next(line for line in capsys.readouterr().out.splitlines() if "Status" in line)

    assert "✅" in status("normal", ("neptune", "normal"))
    assert "✅" in status("BENIGN", ("BENIGN", "DDoS"))
    assert "ATTACK DETECTED - neptune" in status("neptune", ("neptune", "normal"))
    # a dataset without a normal class: say what was predicted, not "attack"
    generic = status("setosa", ("setosa", "virginica"))
    assert "ATTACK" not in generic and "setosa" in generic


def test_examples_of_unlearned_classes_are_marked(cli_module):
    bundle = types.SimpleNamespace(classes=("neptune", "normal"))
    assert cli_module.example_name(bundle, {"label": "normal"}) == "normal"
    assert "not learned" in cli_module.example_name(bundle, {"label": "apache2"})
