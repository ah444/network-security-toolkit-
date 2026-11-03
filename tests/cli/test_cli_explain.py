"""v5.3 step F3f: the explanations in the CLI.

- ``ids --info`` (and the model card at the start of the menu): what each model relies on,
  when it was calculated in the web trainer (read from its saved file; nothing is computed).
- A prediction in the menu: its top 3 reasons toward the predicted class, with the limits.
- ``--json``: each result's ``why``, as the web API gives it (one function in the core)."""

import builtins
import json

import pytest

from toolkit.core.data.ids import bundle_files
from toolkit.core.services.ids.datasets import load_dataset
from web_jobs import train_and_wait, wait_for

CIC = "cic_ddos_sample.csv"
TARGET = "Label"
NOT_CALCULATED = ("   What the models rely on: not calculated. Calculate it in the web trainer "
                  "(the Train step) right after training.")


@pytest.fixture
def trained(client, upload, fixtures_dir):
    columns = upload(fixtures_dir / CIC).get_json()["columns"]
    answer = train_and_wait(client, {
        "features": [c for c in columns if c != TARGET], "target": TARGET,
        "balancing": "none", "models": ["decision_tree", "svm"]})
    assert answer.status_code == 200, answer.get_json()
    return answer.get_json()


def _export(client):
    export = client.post("/api/v1/run/export", json={"name": "cli"}).get_json()
    return str(bundle_files.bundles_dir() / export["file"])


@pytest.fixture
def row(fixtures_dir):
    frame = load_dataset(fixtures_dir / CIC).frame
    return {c: v.item() if hasattr(v, "item") else v
            for c, v in frame.iloc[-1].items() if c != TARGET}


def test_json_gives_why(cli_module, client, trained, row, capsys):
    bundle = _export(client)
    assert cli_module.main(["--bundle", bundle, "--json", json.dumps(row),
                            "--model", "decision_tree"]) == 0
    (result,) = json.loads(capsys.readouterr().out)
    why = result["why"]
    assert set(why) == {"prediction", "probability", "toward", "away", "without_typical",
                        "source", "problem"}
    assert why["prediction"] == result["prediction"] and why["source"] == "training"
    web = client.post("/api/v1/run/predict", json={"values": row, "model": "decision_tree"})
    assert web.get_json()["why"] == why, "the web's and the CLI's are the same"


def test_json_of_a_model_without_probabilities(cli_module, client, trained, row, capsys):
    bundle = _export(client)
    assert cli_module.main(["--bundle", bundle, "--json", json.dumps(row), "--model", "svm"]) == 0
    (result,) = json.loads(capsys.readouterr().out)
    assert result["why"]["problem"] == "NO_PROBABILITIES" and result["prediction"]


def test_a_prediction_shows_its_top_3_reasons(ids_screen, capsys):
    reason = {"feature": "Fwd IAT Std", "value": 3128082.202, "typical": 7698.2715,
              "effect": 0.38}
    why = {"prediction": "DDoS", "probability": 1.0, "toward": [reason] * 5, "away": [],
           "without_typical": [], "source": "training", "problem": None}
    ids_screen.display_prediction({"prediction": "DDoS", "confidence": 1.0, "top": [],
                                   "why": why}, classes=("BENIGN", "DDoS"))
    out = capsys.readouterr().out
    assert ("   💡 Why: what pushed it toward DDoS (an approximation, not causes; "
            "--json gives every reason):") in out
    assert out.count("      - Fwd IAT Std = 3128082.202 (typical 7698.2715): +0.380") == 3


@pytest.mark.parametrize("why, line", [
    ({"prediction": "DDoS", "probability": None, "toward": [], "away": [],
      "without_typical": [], "source": "training", "problem": "NO_PROBABILITIES"},
     "   💡 Why: this model gives no probabilities, so it can't say why."),
    (None, "   💡 Why: no explanation for this prediction (details in the log)."),
    ({"prediction": "DDoS", "probability": 0.6, "toward": [], "away": [],
      "without_typical": [], "source": "training", "problem": None},
     "      no value pushed it toward DDoS more than its typical value would"),
])
def test_a_prediction_without_reasons_says_so(ids_screen, capsys, why, line):
    ids_screen.display_prediction({"prediction": "DDoS", "confidence": None, "top": [],
                                   "why": why}, classes=("BENIGN", "DDoS"))
    assert line in capsys.readouterr().out


def test_the_menu_shows_why(cli_module, client, trained, monkeypatch, capsys):
    bundle = _export(client)
    answers = iter(["", "2", "1", "5"])  # the default model, real example 1, exit
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    assert cli_module.main(["--bundle", bundle]) == 0
    out = capsys.readouterr().out
    assert "💡 Why:" in out and NOT_CALCULATED in out


def test_info_says_when_it_was_not_calculated(cli_module, client, trained, capsys):
    bundle = _export(client)
    assert cli_module.main(["--bundle", bundle, "--info"]) == 0
    assert NOT_CALCULATED in capsys.readouterr().out.splitlines()


def test_info_shows_what_each_model_relies_on(cli_module, client, trained, capsys):
    job = client.post("/api/v1/jobs/importance", json={"model": "decision_tree"}).get_json()
    result = wait_for(client, job["url"])["result"]
    assert result["saved"] is True
    bundle = _export(client)
    assert cli_module.main(["--bundle", bundle, "--info"]) == 0
    lines = capsys.readouterr().out.splitlines()
    start = lines.index("   What the models rely on (the drop in test macro-F1 when a "
                        "feature's values are shuffled; a diagnosis, not causes):")
    assert lines[start + 1] == "   Decision Tree:"
    top = result["features"][:5]
    assert lines[start + 2:start + 2 + len(top)] == [
        f"      {f['feature'][:28]:<28} {f['mean']:.4f} ± {f['std']:.4f}" for f in top]
    if result["concentrated"]:
        assert (f"      ⚠️  {result['concentrated']} carries more than half of the effect: "
                "check that it is not a copy of the label.") in lines
    assert "      Support Vector Machine (linear): not calculated" in lines
