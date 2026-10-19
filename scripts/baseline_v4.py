"""Baseline snapshot of the ORIGINAL web trainer (plan step 0.1b).

Drives the unchanged ``tools/IDS/IDS_Web_CIC_Dynamic/app.py`` through Flask's
test client with a fixed, scripted click sequence, runs every scenario several
times, and writes the results to ``docs/baseline/``. Nothing in the app is
modified; the app module is re-imported fresh for every run so no state leaks
between runs.

These numbers are the "before" side of the v5.0 ML report. They are expected
to be inflated by the leakage documented in the plan (L2, L3, L4, L6).

Usage (project root, inside .venv):
    python scripts/baseline_v4.py              # full baseline (slow)
    python scripts/baseline_v4.py --quick      # tiny fixtures, 1 run (sanity check)
"""

# Single-core execution for the snapshot, set before numpy/sklearn are imported.
# The app hard-codes n_jobs=-1; LOKY_MAX_CPU_COUNT makes "-1" mean 1 core.
import os

SINGLE_CORE_ENV = {
    "LOKY_MAX_CPU_COUNT": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
os.environ.update(SINGLE_CORE_ENV)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "tools" / "IDS" / "IDS_Web_CIC_Dynamic" / "app.py"
UPLOADS = APP_PATH.parent / "uploads"
OUT_DIR = ROOT / "docs" / "baseline"

MODELS = ["random_forest", "logistic_regression", "decision_tree"]
RECOMMEND_K = 10
TEST_SIZE = 0.30
LIBRARIES = ["flask", "numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost",
             "lightgbm", "dask", "joblib"]

DATASETS = {
    "cic_ddos": {"file": ROOT / "data" / "raw" / "DDOS(CIC-2017)_FILTERED_TEST.csv",
                 "target": " Label"},
    "nslkdd_train": {"file": ROOT / "data" / "raw" / "KDDTrain+.txt", "target": "attack"},
}
QUICK_DATASETS = {
    "cic_ddos": {"file": ROOT / "tests" / "fixtures" / "cic_ddos_sample.csv",
                 "target": " Label"},
    "nslkdd_train": {"file": ROOT / "tests" / "fixtures" / "nslkdd_train_sample.txt",
                     "target": "attack"},
}

# Each scenario is the exact sequence of UI actions a user would click.
SCENARIOS = {
    "S1_all_features_recommended_balancing": [
        "upload",
        "analyze_class_distribution (take the app's recommended balancing method)",
        "configure: ALL columns except the target as features, recommended balancing, "
        f"models {MODELS}, test_size {TEST_SIZE}",
    ],
    "S2_apply_smote_then_recommended_features": [
        "upload",
        "apply_smote on the FULL dataset (the 'Apply SMOTE' preview button)",
        f"recommend_features k={RECOMMEND_K} (computed on the SMOTE-balanced full data)",
        f"configure: the {RECOMMEND_K} recommended features, balancing 'none', "
        f"models {MODELS}, test_size {TEST_SIZE}",
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", check=True)
    return result.stdout.strip()


def _environment() -> dict:
    versions = {}
    for lib in LIBRARIES:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = None
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "single_core_env": SINGLE_CORE_ENV,
        "libraries": versions,
    }


def _fresh_client():
    """Import the app module from scratch, so every run starts with clean globals."""
    spec = importlib.util.spec_from_file_location("ids_web_app_baseline", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ids_web_app_baseline"] = module
    spec.loader.exec_module(module)
    module.app.config["TESTING"] = True
    return module.app.test_client()


def _post_json(client, url: str, payload: dict) -> dict:
    response = client.post(url, json=payload)
    body = response.get_json()
    if response.status_code != 200 or body is None or "error" in body:
        raise RuntimeError(f"{url} failed ({response.status_code}): {body}")
    return body


def _upload(client, path: Path) -> list:
    with path.open("rb") as handle:
        response = client.post("/upload", data={"dataset": (handle, path.name)},
                               content_type="multipart/form-data")
    body = response.get_json()
    if response.status_code != 200 or "columns" not in body:
        raise RuntimeError(f"/upload failed ({response.status_code}): {body}")
    return body["columns"]


def _macro_from_confusion(matrix: list) -> dict:
    """Macro-F1 and balanced accuracy from a confusion matrix (rows = true class)."""
    n = len(matrix)
    recalls, f1s = [], []
    for i in range(n):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(matrix[r][i] for r in range(n))
        if support == 0:
            continue
        recall = tp / support
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {
        "macro_f1": sum(f1s) / len(f1s),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "classes_in_split": len(recalls),
    }


def _run_once(dataset: dict, scenario: str) -> dict:
    client = _fresh_client()
    target = dataset["target"]
    columns = _upload(client, dataset["file"])
    steps = {"columns_after_upload": columns}

    if scenario == "S1_all_features_recommended_balancing":
        analysis = _post_json(client, "/analyze_class_distribution", {"target": target})
        balancing = analysis["recommendation"]["recommended"]
        steps["recommendation"] = analysis["recommendation"]
        features = [c for c in columns if c != target]
    else:
        smote = _post_json(client, "/apply_smote", {"target": target})
        steps["apply_smote"] = {k: smote.get(k) for k in
                                ("original_shape", "balanced_shape", "class_counts")}
        rec = _post_json(client, "/recommend_features", {"target": target, "k": RECOMMEND_K})
        features = [name for name, _score in rec["recommended_features"]]
        steps["recommended_features"] = rec["recommended_features"]
        balancing = "none"

    started = time.perf_counter()
    result = _post_json(client, "/configure", {
        "input_features": features,
        "target": target,
        "balancing_method": balancing,
        "test_size": TEST_SIZE,
        "selected_models": MODELS,
    })
    elapsed = time.perf_counter() - started

    models = {}
    for split in ("validation", "test"):
        for name, accuracy in result["accuracies"][split].items():
            matrix = result["confusion_matrices"][split][name]
            entry = models.setdefault(name, {})
            entry[split] = {
                "accuracy": accuracy,
                "weighted_f1": result["f1_scores"][split].get(name),
                "auc": result["auc_scores"][split].get(name),
                **_macro_from_confusion(matrix),
                "confusion_matrix": matrix,
            }

    # Remove the uploaded copy (the app saves every upload into uploads/).
    for leftover in UPLOADS.glob("*"):
        if leftover.name != ".gitkeep":
            leftover.unlink()

    return {
        "features_used": features,
        "balancing_method": balancing,
        "split_info": result.get("split_info"),
        "class_labels": result.get("class_labels"),
        "train_seconds": round(elapsed, 2),
        "models": models,
        "steps": steps,
    }


def _spread(runs: list) -> dict:
    summary = {}
    for name in runs[0]["models"]:
        summary[name] = {}
        for split in ("validation", "test"):
            summary[name][split] = {}
            for metric in ("accuracy", "weighted_f1", "macro_f1", "balanced_accuracy"):
                values = [run["models"][name][split][metric] for run in runs]
                summary[name][split][metric] = {
                    "mean": statistics.fmean(values),
                    "min": min(values),
                    "max": max(values),
                    "spread": max(values) - min(values),
                }
    return summary


def _pct(value: float) -> str:
    return f"{value * 100:.2f}"


def write_report(out_dir: Path = OUT_DIR) -> Path:
    """Build docs/baseline/README.md from the JSON results."""
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*.json"))]
    if not records:
        raise SystemExit("no baseline JSON files found")
    first = records[0]
    env = first["environment"]
    lines = [
        "# Baseline — original web trainer (v4), before any v5.0 change",
        "",
        "These are the **\"before\"** numbers for the v5.0 ML report (plan step 0.1b). They were "
        "produced by `scripts/baseline_v4.py`, which drives the **unchanged** web app through "
        "Flask's test client with a fixed click sequence. They are **expected to be inflated** "
        "by the leakage the plan documents (L2 full-data SMOTE, L3 full-data feature selection, "
        "L4 duplicates across the split, L6 label-derived columns such as NSL-KDD `level`).",
        "",
        "**Read the observations in [FINDINGS.md](FINDINGS.md)** (hand-written; this README is "
        "generated by `python scripts/baseline_v4.py --report`).",
        "",
        "## How they were produced",
        "",
        f"- App code: `{first['app_file']}` at commit `{first['app_code_commit'][:7]}` "
        "(identical to the baseline import)",
        f"- Generated: {first['generated_at']} (repo commit `{first['repo_commit'][:7]}`)",
        f"- Python {env['python']} on {env['os']}; single core "
        "(`LOKY_MAX_CPU_COUNT=1`, `OMP_NUM_THREADS=1`), so `n_jobs=-1` in the app means 1 core",
        "- Libraries: " + ", ".join(f"{k} {v}" for k, v in env["libraries"].items() if v),
        f"- Models: {', '.join(first['models'])}; test_size {first['test_size']} "
        "(70 / 15 / 15 split, `random_state=42` as hard-coded in the app)",
        f"- Each scenario was run **{len(first['runs'])} times**; the spread column is max − min "
        "across runs",
        "- Macro-F1 and balanced accuracy are computed from the app's own confusion matrices "
        "(the app itself only reports accuracy and weighted F1)",
        "",
        "Re-run: `python scripts/baseline_v4.py` (needs the datasets in `data/raw/`, see "
        "`docs/datasets.md`).",
        "",
        "## Scenarios (exact click sequences)",
        "",
    ]
    for scenario, sequence in SCENARIOS.items():
        lines.append(f"**{scenario}**")
        lines.extend(f"{i}. {step}" for i, step in enumerate(sequence, 1))
        lines.append("")
    lines += ["## Results (test split, mean over runs, %)", "",
              "| Dataset | Scenario | Model | Accuracy | Weighted F1 | Macro-F1 | "
              "Balanced acc. | Spread (macro-F1) | Features | Balancing |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for rec in records:
        run0 = rec["runs"][0]
        for model, splits in rec["spread"].items():
            test = splits["test"]
            lines.append(
                f"| {rec['dataset']['name']} | {rec['scenario'].split('_')[0]} | {model} | "
                f"{_pct(test['accuracy']['mean'])} | {_pct(test['weighted_f1']['mean'])} | "
                f"{_pct(test['macro_f1']['mean'])} | {_pct(test['balanced_accuracy']['mean'])} | "
                f"{_pct(test['macro_f1']['spread'])} | {len(run0['features_used'])} | "
                f"{run0['balancing_method']} |"
            )
    lines += ["", "## Datasets", ""]
    seen = set()
    for rec in records:
        ds = rec["dataset"]
        if ds["name"] in seen:
            continue
        seen.add(ds["name"])
        lines.append(f"- **{ds['name']}**: `{ds['file']}`, target `{ds['target']!s}`, "
                     f"SHA-256 `{ds['sha256']}`")
    lines.append("")
    report = out_dir / "README.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--quick", action="store_true",
                        help="use the small fixtures and 1 run (sanity check only)")
    parser.add_argument("--report", action="store_true",
                        help="only rebuild docs/baseline/README.md from existing JSON files")
    args = parser.parse_args()
    if args.report:
        print("wrote", write_report().relative_to(ROOT).as_posix())
        return
    datasets = QUICK_DATASETS if args.quick else DATASETS
    runs_wanted = 1 if args.quick else args.runs
    out_dir = OUT_DIR / "quick" if args.quick else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    app_commit = _git("log", "-1", "--format=%H", "--", APP_PATH.relative_to(ROOT).as_posix())
    common = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": _git("rev-parse", "HEAD"),
        "app_code_commit": app_commit,
        "app_file": APP_PATH.relative_to(ROOT).as_posix(),
        "environment": _environment(),
        "models": MODELS,
        "test_size": TEST_SIZE,
    }

    for ds_name, dataset in datasets.items():
        for scenario, sequence in SCENARIOS.items():
            runs = []
            for index in range(runs_wanted):
                print(f"[{ds_name} / {scenario}] run {index + 1}/{runs_wanted} ...", flush=True)
                runs.append(_run_once(dataset, scenario))
                print(f"    done in {runs[-1]['train_seconds']} s (training)", flush=True)
            record = {
                **common,
                "dataset": {
                    "name": ds_name,
                    "file": dataset["file"].relative_to(ROOT).as_posix(),
                    "sha256": _sha256(dataset["file"]),
                    "target": dataset["target"],
                },
                "scenario": scenario,
                "click_sequence": sequence,
                "runs": runs,
                "spread": _spread(runs),
            }
            out_file = out_dir / f"{ds_name}__{scenario}.json"
            out_file.write_text(json.dumps(record, indent=1), encoding="utf-8")
            print(f"    wrote {out_file.relative_to(ROOT).as_posix()}", flush=True)
    print("wrote", write_report(out_dir).relative_to(ROOT).as_posix(), flush=True)


if __name__ == "__main__":
    main()
