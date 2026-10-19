"""The 0.1b baseline must exist and be complete (plan §7, §8 "Baseline exists")."""

import json
import re

import pytest

DATASETS = ["cic_ddos", "nslkdd_train"]
SCENARIOS = ["S1_all_features_recommended_balancing", "S2_apply_smote_then_recommended_features"]
RUNS = 3


@pytest.fixture
def datasets_doc(repo_root):
    return (repo_root / "docs" / "datasets.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_baseline_complete(repo_root, datasets_doc, dataset, scenario):
    path = repo_root / "docs" / "baseline" / f"{dataset}__{scenario}.json"
    assert path.is_file(), f"missing baseline file {path.name}"
    record = json.loads(path.read_text(encoding="utf-8"))

    assert re.fullmatch(r"[0-9a-f]{40}", record["app_code_commit"])
    assert record["environment"]["libraries"]["scikit-learn"]
    assert record["environment"]["single_core_env"]["LOKY_MAX_CPU_COUNT"] == "1"
    assert record["click_sequence"], "the scripted click sequence must be recorded"
    assert len(record["runs"]) == RUNS
    assert record["spread"], "run-to-run spread must be recorded"
    # The dataset fingerprint must match the one documented in docs/datasets.md.
    assert record["dataset"]["sha256"] in datasets_doc
    for run in record["runs"]:
        for metrics in run["models"].values():
            assert {"accuracy", "macro_f1", "balanced_accuracy", "confusion_matrix"} <= set(
                metrics["test"]
            )
