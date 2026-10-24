"""
IDS with Machine Learning - CLI Tool (v5)
=========================================
Predicts with the models you export from the web trainer: the features, classes and best
model come from the bundle itself (plan D5, step 0.5). No model is shipped with the
project: train one in the web trainer and click Export (saved in instance/bundles).

    python -m toolkit ids                     interactive: choose an export
    python -m toolkit ids --list              list the exports
    python -m toolkit ids --csv flows.csv     label every row (newest export)
    python -m toolkit ids --bundle my.bundle --json '{"Flow Duration": 3, ...}'

(from the project folder; ``python -m toolkit.ids_cli`` also works; moved from
tools/IDS/IDS_CLI - CIC-2017/ in v5.1 step 1.4a)

Trust (D11): a bundle is opened only if it is signed by this install's key
(instance/signing.key). Nothing else is ever opened.
"""

import argparse
import json
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ..services.ids.exports import BundleInfo, BundleStore
from ..services.ids.batch import prediction_csv, prediction_summary, prediction_table
from ..services.ids.bundle import Bundle, BundleVersionWarning
from ..data.ids.bundle_files import BundleError
from ..data.ids.dataset_files import DatasetError
from ..services.ids.datasets import load_dataset
from ..services.ids.registry import MODELS
from ..data.files import read_text, write_new_text
from ..data.logs import setup_logging
from ..services.ports import ModelStore

MODEL_STORE: ModelStore = BundleStore()  # this install's exports (instance/bundles)
LARGE_BUNDLE = 500 * 2**20  # bytes: say that loading takes a while
# Normal-traffic labels of the supported datasets (real labels): CIC-IDS2017 "BENIGN",
# NSL-KDD "normal"
NORMAL_LABELS = {"benign", "normal"}
NO_EXPORT = ("No exported model yet: train one in the web trainer and click Export (it is "
             "saved in instance/bundles), or pass --bundle.")

# Plain-language notes for the split codes a model card can show
SPLIT_NOTES = {
    "STRATIFIED_SCORES_MAY_BE_OPTIMISTIC": "random (stratified) split: scores may be optimistic",
    "OFFICIAL_TEST_OVERLAP_KEPT": "official test file kept as published (some rows also in train)",
    "CLASS_MISSING_FROM_TRAIN": "some classes appear only in the test split",
    "CLASS_MISSING_FROM_TEST": "some classes have no test rows",
}


# ============================================================
# Loading and predicting
# ============================================================

def size_text(size: int) -> str:
    return f"{size / 2**30:.2f} GB" if size >= 2**30 else f"{size / 2**20:.1f} MB"


def export_line(info: BundleInfo) -> str:
    """One export, from its manifest header (read without loading the models)."""
    manifest = info.manifest
    dataset = ((manifest.get("run") or {}).get("dataset") or {}).get("name", "unknown data")
    selected = manifest.get("selected", "")
    title = MODELS[selected].title if selected in MODELS else selected
    others = len(manifest.get("models", [])) - 1
    return (f"{info.path.stem}  {dataset}  {title}" + (f" +{others}" if others > 0 else "")
            + f"  ({size_text(info.size)})")


def load_model(bundle_path: str | Path | None = None) -> Bundle:
    """A bundle signed by this install: ``bundle_path``, else the newest export.

    Raises BundleError (with a message for the user) for anything else."""
    if bundle_path is None:
        exports = MODEL_STORE.list_bundles()
        if not exports:
            raise BundleError("NO_EXPORT", NO_EXPORT)
        bundle_path = exports[0].path
    path = Path(bundle_path)
    if path.is_file() and path.stat().st_size > LARGE_BUNDLE:
        print(f"⏳ Loading {path.name} ({size_text(path.stat().st_size)}): large models can "
              "take minutes and need several times that size in memory...", file=sys.stderr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BundleVersionWarning)  # shown in the model card
        return MODEL_STORE.load(path)


def choose_export() -> Path:
    """The export the user picks (newest first; Enter = the newest)."""
    exports = MODEL_STORE.list_bundles()
    if not exports:
        raise BundleError("NO_EXPORT", NO_EXPORT)
    print("\n📂 Models exported from the web trainer (newest first):")
    for i, info in enumerate(exports, 1):
        print(f"  {i}. {export_line(info)}")
    picked = choose(f"\nChoose an export (1-{len(exports)}, Enter = 1): ", len(exports),
                    default=1)
    return exports[picked - 1].path


def list_exports() -> int:
    exports = MODEL_STORE.list_bundles()
    if not exports:
        print(NO_EXPORT)
    for i, info in enumerate(exports, 1):
        print(f"{i}. {export_line(info)}")
    return 0


def label_column(bundle: Bundle) -> str | None:
    record = bundle.run
    return cast(str | None, record["settings"].get("label")
                or (record.get("profile") or {}).get("label_column"))


def model_order(bundle: Bundle) -> list[str]:
    """Model names, best first (the run's leaderboard order)."""
    ranked = [row["model"] for row in bundle.run["evaluation"]["leaderboard"]]
    return [name for name in ranked if name in bundle.models] + \
        [name for name in bundle.models if name not in ranked]


def predict_rows(bundle: Bundle, frame: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    """Predictions (and confidence, when the model has probabilities) for every row."""
    missing = [f for f in bundle.features if f not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature column(s): {', '.join(missing)}")
    fitted = bundle.model(model)
    result = pd.DataFrame({"prediction": fitted.predict(frame)}, index=frame.index)
    proba = fitted.predict_proba(frame)
    result["confidence"] = proba.max(axis=1) if proba is not None else np.nan
    return result


def predict_values(bundle: Bundle, values: Mapping[str, Any],
                   model: str | None = None) -> dict[str, Any]:
    """One row of feature values (text or numbers) -> {prediction, confidence, top}."""
    fitted = bundle.model(model)
    row = fitted.input_frame(values)
    proba = fitted.predict_proba(row)
    result: dict[str, Any] = {"prediction": str(fitted.predict(row)[0]), "model": fitted.name,
                              "confidence": None, "top": []}
    if proba is not None:
        order = np.argsort(proba[0])[::-1][:5]
        result["confidence"] = float(proba[0].max())
        result["top"] = [[fitted.classes[i], float(proba[0][i])] for i in order]
    return result


def model_card(bundle: Bundle) -> str:
    """What the model is and how good it is, from the bundle's TrainingRun record."""
    run = bundle.run
    dataset = run.get("dataset") or {}
    split = run["split"]
    lines = [
        f"📦 Model bundle: run {bundle.run_id} (exported on this install)",
        f"   Trained {run['created_at']} on {dataset.get('name', 'unknown data')}"
        + (f" ({dataset['rows']:,} rows, SHA-256 {dataset['sha256'][:12]}…)" if dataset else ""),
        f"   Split: {split['strategy']}, {split['train_rows']:,} train / {split['test_rows']:,} test rows",
    ]
    for code in split.get("warnings", []):
        if code in SPLIT_NOTES:
            lines.append(f"   ⚠️  {SPLIT_NOTES[code]}")
    lines.append("   Models (CV macro-F1 on the training split | test macro-F1 | test balanced accuracy):")
    for row in run["evaluation"]["leaderboard"]:
        if row["model"] in bundle.models:
            star = "⭐" if row["model"] == bundle.selected else "  "
            lines.append(f"   {star} {row['title']:<26} {row['cv_macro_f1_mean']:.3f} ± "
                         f"{row['cv_macro_f1_std']:.3f} | {row['test_macro_f1']:.3f} | "
                         f"{row['test_balanced_accuracy']:.3f}")
    lines.append(f"   Classes ({len(bundle.classes)}): {', '.join(bundle.classes)}")
    lines.append(f"   Features: {len(bundle.features)}")
    for warning in bundle.warnings:
        lines.append(f"   ⚠️  {warning}")
    return "\n".join(lines)


# ============================================================
# Batch modes: --csv and --json
# ============================================================

def run_csv(bundle: Bundle, csv_path: str | Path, output: str | Path | None = None,
            model: str | None = None) -> int:
    """Label every row of a CSV file; the result file is never overwritten."""
    csv_path = Path(csv_path)
    output = Path(output) if output else csv_path.with_name(csv_path.stem + ".predictions.csv")
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose another --output")
    frame = load_dataset(csv_path).frame
    label = label_column(bundle)
    # the same table as the web trainer's "Predict a CSV file" (services.ids.batch, step 1.4f)
    table = prediction_table(bundle.model(model), frame, label)
    write_new_text(output, prediction_csv(table))
    summary = prediction_summary(table)
    print(f"✅ {len(table)} rows predicted with {bundle.model(model).name}; saved to {output}")
    for count in summary["counts"]:
        print(f"   {count['label']:<30} {count['count']:>8,}")
    if summary["matches"]:
        same = summary["matches"]["matching"]
        print(f"   {same:,} of {len(table):,} predictions ({summary['matches']['share']:.1%}) "
              f"match the file's {label} column")
    return 0


def run_json(bundle: Bundle, text: str, model: str | None = None) -> int:
    """Rows given as JSON (an object or a list, inline or a .json file) -> JSON results."""
    source = Path(text)
    if text.strip()[:1] not in "[{" and source.is_file():
        text = read_text(source)
    rows = json.loads(text)
    rows = [rows] if isinstance(rows, dict) else rows
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError("JSON input must be an object of feature values, or a list of them")
    print(json.dumps([predict_values(bundle, row, model) for row in rows], indent=1))
    return 0


# ============================================================
# Interactive CLI
# ============================================================

def display_prediction(result: Mapping[str, Any], true_label: str | None = None,
                       classes: Sequence[str] = ()) -> None:
    """Display a prediction result (``classes``: the model's classes)."""
    print("\n" + "=" * 50)
    print("🎯 PREDICTION RESULT")
    print("=" * 50)
    label = result["prediction"]
    if label.lower() in NORMAL_LABELS:
        print(f"   Status: ✅ Normal traffic - {label}")
    elif any(c.lower() in NORMAL_LABELS for c in classes):
        print(f"   Status: 🚨 ATTACK DETECTED - {label}")
    else:  # a dataset without a normal class
        print(f"   Status: 🔎 Predicted class - {label}")
    if true_label is not None:
        print(f"   True label: {true_label}")
    if result["confidence"] is not None:
        print(f"   Confidence: {result['confidence']:.2%}")
    if result["top"]:
        print("\n   📊 Top Probabilities:")
        for cls, prob in result["top"]:
            bar = "█" * int(prob * 20)
            print(f"      {cls[:25]:25} {prob:6.2%} {bar}")
    print("=" * 50)


def example_name(bundle: Bundle, example: Mapping[str, Any]) -> str:
    """An example's label, marked when the model never learned that class."""
    label = cast(str, example["label"])
    if label in bundle.classes:
        return label
    return f"{label}  (not learned: this class is only in the test split)"


def choose(prompt: str, count: int, default: int | None = None) -> int:
    """A number 1..count from the user (Enter gives ``default``)."""
    while True:
        answer = input(prompt).strip()
        if answer == "" and default is not None:
            return default
        if answer.isdigit() and 1 <= int(answer) <= count:
            return int(answer)
        print("⚠️ Invalid selection, try again")


def choose_model(bundle: Bundle, current: str) -> str:
    names = model_order(bundle)
    for i, name in enumerate(names, 1):
        marker = "⭐" if name == bundle.selected else "  "
        print(f"  {marker} {i}. {name}")
    picked = choose(f"\nEnter model number (or press Enter for {current}): ", len(names),
                    default=names.index(current) + 1)
    return names[picked - 1]


def manual_input(bundle: Bundle, model: str | None) -> dict[str, Any]:
    """Every feature, prefilled from a real example (Enter keeps a value)."""
    examples = bundle.manifest.get("examples", [])
    start: Mapping[str, Any] = {}
    if examples:
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example_name(bundle, example)}")
        number = input("Start from example (Enter = 1, 0 = empty form): ").strip()
        if number != "0":
            index = int(number) - 1 if number.isdigit() and 1 <= int(number) <= len(examples) else 0
            start = examples[index]["values"]
    print("\n📝 Enter feature values (Enter keeps the value in brackets):")
    values: dict[str, Any] = {}
    for i, info in enumerate(bundle.model(model).feature_schema(), 1):
        default = start.get(info.name)
        hint = f" ({'/'.join(info.categories[:6])})" if info.categories else ""
        shown = f" [{default}]" if default is not None else ""
        while True:
            answer = input(f"  {i}. {info.name}{hint}{shown}: ").strip()
            if answer or default is not None:
                values[info.name] = answer or default
                break
            print("     ⚠️ Please enter a value")
    return values


def interactive(bundle: Bundle, model: str | None = None) -> int:
    print(model_card(bundle))
    model = choose_model(bundle, model or bundle.selected)
    print(f"\n🎯 Ready for predictions with {model}!")
    while True:
        print("\nOptions:")
        print("  1. Enter features manually")
        print("  2. Test with a real example (from the model's test split)")
        print("  3. Switch model")
        print("  4. Predict a CSV file")
        print("  5. Exit")
        choice = input("\nSelect option: ").strip()
        try:
            if choice == "1":
                display_prediction(predict_values(bundle, manual_input(bundle, model), model),
                                   classes=bundle.classes)
            elif choice == "2":
                examples = bundle.manifest.get("examples", [])
                if not examples:
                    print("⚠️ This bundle has no examples; use option 1 or 4.")
                    continue
                for i, example in enumerate(examples, 1):
                    print(f"  {i}. {example_name(bundle, example)}")
                example = examples[choose(f"\nChoose example (1-{len(examples)}): ",
                                          len(examples)) - 1]
                display_prediction(predict_values(bundle, example["values"], model),
                                   true_label=example["label"], classes=bundle.classes)
            elif choice == "3":
                model = choose_model(bundle, model)
                print(f"✅ Using {model}")
            elif choice == "4":
                path = input("CSV file path: ").strip().strip('"')
                run_csv(bundle, path, model=model)
            elif choice == "5":
                print("\n👋 Goodbye!")
                return 0
            else:
                print("⚠️ Invalid option")
        except (ValueError, FileExistsError, FileNotFoundError, DatasetError) as error:
            print(f"❌ {error}")


# ============================================================
# Main
# ============================================================

def parse_args(argv: Sequence[str] | None, prog: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=prog, description="Intrusion detection with a model bundle exported by the web trainer.")
    parser.add_argument("--bundle", help="a .bundle exported on this computer "
                        "(default: choose one; without a prompt, the newest export)")
    parser.add_argument("--list", action="store_true", help="list the exports and exit")
    parser.add_argument("--model", help="which model of the bundle (default: its best)")
    parser.add_argument("--csv", help="label every row of this CSV file")
    parser.add_argument("--output", help="where --csv writes its result "
                        "(default: <file>.predictions.csv; never overwritten)")
    parser.add_argument("--json", help="rows as JSON (object or list, inline or a .json file); "
                        "prints JSON results")
    parser.add_argument("--info", action="store_true", help="show the model card and exit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    args = parse_args(argv, prog)
    if args.list:
        return list_exports()
    interactive_mode = not (args.csv or args.json or args.info)
    if interactive_mode:
        print("\n" + "=" * 60)
        print("🛡️  IDS with Machine Learning - CLI Tool")
        print("=" * 60)
    try:
        if interactive_mode and args.bundle is None:
            bundle = load_model(choose_export())
        else:
            bundle = load_model(args.bundle)
        if args.model and args.model not in bundle.models:
            raise ValueError(f"Unknown model {args.model!r}; this bundle has "
                             f"{', '.join(bundle.models)}")
        if args.info:
            print(model_card(bundle))
            return 0
        if args.csv:
            return run_csv(bundle, args.csv, args.output, args.model)
        if args.json:
            return run_json(bundle, args.json, args.model)
        return interactive(bundle, args.model)
    except (BundleError, ValueError, FileExistsError, FileNotFoundError, DatasetError) as error:
        print(f"❌ {error}")
        return 2
    except (EOFError, KeyboardInterrupt):
        print("\n👋 Goodbye!")
        return 0


def utf8_output() -> None:
    """UTF-8 output even when redirected: Windows then defaults to cp1252, which can't
    print the emoji (or the JSON of non-ASCII class names) and would crash."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and \
                (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8")


def run(argv: Sequence[str] | None = None, prog: str | None = None) -> None:
    """The command: UTF-8 output, the tool's log, then ``main``; exits with its code.
    ``prog`` is the command its help shows (``python -m toolkit ids``)."""
    utf8_output()
    setup_logging("ids_cli")  # instance/logs/ids_cli.log; warnings also on the console
    sys.exit(main(argv, prog))
