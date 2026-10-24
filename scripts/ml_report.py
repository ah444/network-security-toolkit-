"""ML report for v5.0: the baseline click paths, re-run on the fixed ML core.

The "before" numbers are the v4 web trainer's, saved by ``scripts/baseline_v4.py`` in
``docs/baseline/``. This script runs the same datasets, models and scenarios through the
v5 core (``toolkit.services.ids.runs.train``) three times each, on one core like the
baseline, and writes the results to ``docs/report/`` (one JSON file per dataset and
scenario plus a generated README comparing them with the baseline).

Scenarios (same intent as the baseline's click paths, now leakage-free):
    S1  every column except the label as a candidate feature, recommended balancing
    S2  SMOTE + the 10 best features (ANOVA F-test), both fitted on the training split
        only (v4 ran both on the whole dataset before splitting)

Usage (project root, inside .venv):
    python scripts/ml_report.py            # full report (needs data/raw, see docs/datasets.md)
    python scripts/ml_report.py --quick    # tiny fixtures, 1 run, into a temporary folder
    python scripts/ml_report.py --report   # only regenerate docs/report/README.md
    python scripts/ml_report.py --case nslkdd_official:S2 --jobs 3
                                           # one case, its 3 runs side by side (one core each)
"""

# Single-core execution like the baseline, set before numpy/sklearn are imported.
import os

SINGLE_CORE_ENV = {
    "LOKY_MAX_CPU_COUNT": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
os.environ.update(SINGLE_CORE_ENV)

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolkit.services.ids.datasets import load_dataset  # noqa: E402
from toolkit.services.ids.runs import (  # noqa: E402
    DatasetRef,
    TrainSettings,
    package_versions,
    train,
)

REPORT_DIR = REPO_ROOT / "docs" / "report"
BASELINE_DIR = REPO_ROOT / "docs" / "baseline"
RAW = REPO_ROOT / "data" / "raw"
FIXTURES = REPO_ROOT / "tests" / "fixtures"
MODELS = ("random_forest", "logistic_regression", "decision_tree")
RUNS = 3
METRICS = ("accuracy", "weighted_f1", "macro_f1", "balanced_accuracy")

# name -> (train file, official test file or None, baseline dataset name or None)
DATASETS = {
    "cic_ddos": (RAW / "DDOS(CIC-2017)_FILTERED_TEST.csv", None, "cic_ddos"),
    "nslkdd_train": (RAW / "KDDTrain+.txt", None, "nslkdd_train"),
    "nslkdd_official": (RAW / "KDDTrain+.txt", RAW / "KDDTest+.txt", None),
}
QUICK_DATASETS = {
    "cic_ddos": (FIXTURES / "cic_ddos_sample.csv", None, None),  # no baseline for fixtures
    "nslkdd_official": (FIXTURES / "nslkdd_train_sample.txt",
                        FIXTURES / "nslkdd_test_sample_noheader.txt", None),
}
SCENARIOS = {
    "S1": {"title": "every column as a candidate, recommended balancing",
           "baseline": "S1_all_features_recommended_balancing",
           "settings": {"balancer": "recommended"}},
    "S2": {"title": "SMOTE + 10 best features (F-test), fitted on the training split only",
           "baseline": "S2_apply_smote_then_recommended_features",
           "settings": {"balancer": "smote", "selector": "f_classif", "k_features": 10}},
}


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _summary(record: dict, seconds: float) -> dict:
    """What the report needs from one TrainingRun (the full record is not kept)."""
    split = record["split"]
    dedupe = split.get("dedupe") or {}
    evaluation = record["evaluation"]
    models = {}
    for model in evaluation["models"]:
        test = model.get("test") or {}
        cv = model.get("cv") or {}
        models[model["name"]] = {
            "test": {m: test.get(m) for m in (*METRICS, "rows")},
            "test_recall_per_class": {c["label"]: c["recall"] for c in test.get("per_class", [])},
            "test_support_per_class": {c["label"]: c["support"]
                                       for c in test.get("per_class", [])},
            "cv_macro_f1_mean": cv.get("macro_f1_mean"),
            "cv_macro_f1_std": cv.get("macro_f1_std"),
        }
    return {
        "run_id": record["run_id"],
        "seconds": round(seconds, 1),
        "split": {"strategy": split["strategy"], "warnings": split.get("warnings", []),
                  "rows_before_dedupe": dedupe.get("rows_before"),
                  "duplicates_removed": dedupe.get("removed"),
                  "test_rows": next(iter(models.values()))["test"]["rows"] if models else None},
        "leakage": {"blocked": record["leakage"].get("blocked", []),
                    "warned": record["leakage"].get("warned", [])},
        "features": record["features"],
        "balancing": record["balancing"],
        "selection": record["selection"],
        "selected_model": evaluation.get("selected"),
        "classes_only_in_test": evaluation.get("classes_only_in_test", []),
        "failures": evaluation.get("failures", []),
        "models": models,
    }


def _spread(runs: list[dict]) -> dict:
    out = {}
    for name in runs[0]["models"]:
        out[name] = {}
        for metric in METRICS:
            values = [r["models"][name]["test"][metric] for r in runs
                      if r["models"].get(name, {}).get("test", {}).get(metric) is not None]
            if values:
                out[name][metric] = {"mean": statistics.fmean(values), "min": min(values),
                                     "max": max(values), "spread": max(values) - min(values)}
    return out


def _one_run(scenario: str, paths) -> dict:
    """One training run on one core; top level so it can run in a worker process."""
    train_path, test_path, _ = paths
    loaded = load_dataset(train_path)
    test = load_dataset(test_path) if test_path else None
    settings = TrainSettings(models=MODELS, workers=1, **SCENARIOS[scenario]["settings"])
    start = time.time()
    record = train(loaded.frame, profile=loaded.profile, settings=settings,
                   test_frame=test.frame if test else None,
                   source=DatasetRef.from_loaded(train_path.name, loaded),
                   test_source=DatasetRef.from_loaded(test_path.name, test) if test else None,
                   ).to_dict()
    return _summary(record, time.time() - start)


def run_case(dataset: str, scenario: str, paths, runs: int, jobs: int = 1) -> dict:
    """``runs`` runs, ``jobs`` of them at a time in separate processes (each on one core)."""
    train_path, test_path, baseline_name = paths
    loaded = load_dataset(train_path)
    test = load_dataset(test_path) if test_path else None
    summaries = []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, runs)) as pool:
            futures = [pool.submit(_one_run, scenario, paths) for _ in range(runs)]
            for number, future in enumerate(futures, 1):
                summaries.append(future.result())
                print(f"  {dataset} {scenario} run {number}/{runs}: "
                      f"{summaries[-1]['seconds']} s", flush=True)
    else:
        for number in range(1, runs + 1):
            summaries.append(_one_run(scenario, paths))
            print(f"  {dataset} {scenario} run {number}/{runs}: {summaries[-1]['seconds']} s",
                  flush=True)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": _git_commit(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "single_core": SINGLE_CORE_ENV, "versions": package_versions()},
        "dataset": {"name": dataset, "file": train_path.name, "sha256": loaded.report.sha256,
                    "rows": loaded.report.rows, "profile": f"{loaded.profile.profile.name} ({loaded.profile.variant})"
                    if loaded.profile else None,
                    "test_file": test_path.name if test_path else None,
                    "test_sha256": test.report.sha256 if test else None},
        "baseline": f"{baseline_name}__{SCENARIOS[scenario]['baseline']}.json"
        if baseline_name else None,
        "scenario": scenario,
        "scenario_title": SCENARIOS[scenario]["title"],
        "models": list(MODELS),
        "runs": summaries,
        "spread": _spread(summaries),
    }


# --- README -----------------------------------------------------------------------------------

def _pct(value) -> str:
    return "–" if value is None else f"{value * 100:.2f}"


def _baseline_test(file_name: str | None) -> dict:
    if not file_name or not (BASELINE_DIR / file_name).is_file():
        return {}
    data = json.loads((BASELINE_DIR / file_name).read_text(encoding="utf-8"))
    return {model: {m: v["mean"] for m, v in values["test"].items()}
            for model, values in data["spread"].items()}


def _feature_count(run: dict) -> int:
    """Features the models were trained on: the selector's k, else every guarded feature."""
    k = run["selection"].get("k_features")
    if run["selection"].get("selector", "none") != "none" and k:
        return min(k, len(run["features"]))
    return len(run["features"])


def write_readme(report_dir: Path) -> Path:
    results = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(report_dir.glob("*__S*.json"))]
    runs = max((len(r["runs"]) for r in results), default=0)
    lines = [
        "# ML report — v4 baseline vs the v5.0 ML core",
        "",
        "Generated by `python scripts/ml_report.py --report` from the JSON files in this "
        "folder. **Read the explanation in [FINDINGS.md](FINDINGS.md).** The \"before\" "
        "numbers are the v4 web trainer's, from [`docs/baseline/`](../baseline/README.md).",
        "",
        f"All numbers are on the **test split**, mean of {runs} run(s), in %. v4 split "
        "70/15/15 with no deduplication; v5 removes duplicate rows, then splits before "
        "anything is fitted (80/20 stratified, or the official test file), and scores the "
        "test split once (the model is chosen by cross-validation on the training split).",
        "",
        "| Dataset | Scenario | Model | v4 macro-F1 | v5 macro-F1 | v4 bal. acc. | "
        "v5 bal. acc. | v4 accuracy | v5 accuracy | v5 spread (macro-F1) | v5 features |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        before = _baseline_test(result["baseline"])
        first = result["runs"][0]
        for model in result["models"]:
            after = result["spread"].get(model, {})
            if not after:
                continue
            b = before.get(model, {})
            lines.append(
                f"| {result['dataset']['name']} | {result['scenario']} | {model} "
                f"| {_pct(b.get('macro_f1'))} | {_pct(after['macro_f1']['mean'])} "
                f"| {_pct(b.get('balanced_accuracy'))} "
                f"| {_pct(after['balanced_accuracy']['mean'])} "
                f"| {_pct(b.get('accuracy'))} | {_pct(after['accuracy']['mean'])} "
                f"| {after['macro_f1']['spread'] * 100:.2f} | {_feature_count(first)} |")
    lines += ["", "## Runs", ""]
    for result in results:
        first = result["runs"][0]
        split = first["split"]
        lines.append(
            f"- **{result['dataset']['name']} {result['scenario']}** "
            f"({result['scenario_title']}): `{result['dataset']['file']}`"
            + (f" + `{result['dataset']['test_file']}`" if result['dataset']['test_file']
               else "")
            + f", split `{split['strategy']}`, {split['duplicates_removed'] or 0:,} duplicate "
            f"rows removed, {split['test_rows']:,} test rows; leakage guard blocked "
            f"{', '.join(f'`{b}`' for b in first['leakage']['blocked']) or 'nothing'}; "
            f"model chosen by CV: `{first['selected_model']}`; "
            f"{sum(r['seconds'] for r in result['runs']):.0f} s for {len(result['runs'])} runs "
            f"(commit `{result['repo_commit']}`).")
    env = results[0]["environment"] if results else {}
    versions = {k: v for k, v in (env.get("versions") or {}).items()
                if v and k not in ("python", "platform")}
    lines += ["", f"Python {env.get('python')} on {env.get('platform')}, one core; "
              + ", ".join(f"{k} {v}" for k, v in versions.items()), ""]
    path = report_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="fixtures, 1 run, temp folder")
    parser.add_argument("--report", action="store_true", help="only rewrite README.md")
    parser.add_argument("--out", type=Path, help="output folder (default docs/report)")
    parser.add_argument("--case", action="append", metavar="DATASET:SCENARIO",
                        help="run only this case (repeatable), e.g. nslkdd_official:S2")
    parser.add_argument("--jobs", type=int, default=1,
                        help="runs of one case in parallel processes, each on one core")
    args = parser.parse_args(argv)
    out = args.out or (Path(tempfile.mkdtemp(prefix="ml_report_")) if args.quick
                       else REPORT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    if not args.report:
        datasets = QUICK_DATASETS if args.quick else DATASETS
        cases = [(d, s) for d in datasets for s in SCENARIOS]
        if args.case:
            unknown = [c for c in args.case if tuple(c.split(":", 1)) not in cases]
            if unknown:
                parser.error(f"unknown case(s): {', '.join(unknown)}")
            cases = [c for c in cases if ":".join(c) in args.case]
        for dataset, scenario in cases:
            result = run_case(dataset, scenario, datasets[dataset],
                              1 if args.quick else RUNS, jobs=args.jobs)
            target = out / f"{dataset}__{scenario}.json"
            target.write_text(json.dumps(result, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    print(f"Wrote {write_readme(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
