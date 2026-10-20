"""5.0 final: the ML report script runs end to end (on the fixtures, one run)."""

import importlib.util
import json
import sys


def test_ml_report_quick(repo_root, tmp_path):
    spec = importlib.util.spec_from_file_location("ml_report", repo_root / "scripts/ml_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ml_report"] = module
    spec.loader.exec_module(module)
    assert module.main(["--quick", "--out", str(tmp_path)]) == 0
    result = json.loads((tmp_path / "nslkdd_official__S2.json").read_text(encoding="utf-8"))
    run = result["runs"][0]
    assert run["leakage"]["blocked"] == ["level"], "the label-derived column never trains"
    assert run["split"]["strategy"] == "official"
    assert run["selection"] == {"selector": "f_classif", "k_features": 10}
    assert set(result["spread"]) == set(module.MODELS)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "| nslkdd_official | S1 | random_forest |" in readme
    assert result["baseline"] is None, "fixtures are never compared with the real baseline"
