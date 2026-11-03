"""What the IDS command shows and asks: the exports, the model card, predictions, the
interactive menu and its prompts (``toolkit.cli.controllers.ids`` runs them)."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ...core.services.ids.bundle import Bundle
from ...core.services.ids.exports import EXPORTED, INCOMPLETE, VERIFICATION_FAILED, BundleInfo
from ...core.services.ids.prediction import model_order
from ...core.services.ids.registry import MODELS

# Normal-traffic labels of the supported datasets (real labels): CIC-IDS2017 "BENIGN",
# NSL-KDD "normal"
NORMAL_LABELS = {"benign", "normal"}
NO_EXPORT = ("No exported model yet: train one in the web trainer and click Export (it is "
             "saved in instance/bundles), or pass --bundle.")
# What a listed bundle that isn't a whole export says (HR3d); it is never removed on its own
STATE_NOTES = {
    INCOMPLETE: "  [INCOMPLETE: the export was interrupted; remove the file to export again]",
    VERIFICATION_FAILED: "  [VERIFICATION FAILED: its signature does not match this install's "
                         "key]",
}

# Plain-language notes for the split codes a model card can show
SPLIT_NOTES = {
    "STRATIFIED_SCORES_MAY_BE_OPTIMISTIC": "random (stratified) split: scores may be optimistic",
    "OFFICIAL_TEST_OVERLAP_KEPT": "official test file kept as published (some rows also in train)",
    "CLASS_MISSING_FROM_TRAIN": "some classes appear only in the test split",
    "CLASS_MISSING_FROM_TEST": "some classes have no test rows",
}



def size_text(size: int) -> str:
    """A file size in MB, or in GB from 1 GB."""
    return f"{size / 2**30:.2f} GB" if size >= 2**30 else f"{size / 2**20:.1f} MB"


def export_line(info: BundleInfo) -> str:
    """One export, from its manifest header (read without loading the models), and its state
    when it is not a whole export (HR3d)."""
    manifest = info.manifest
    if not manifest:  # cut off before its manifest ended
        return f"{info.path.stem}  ({size_text(info.size)})" + STATE_NOTES.get(info.state, "")
    dataset = ((manifest.get("run") or {}).get("dataset") or {}).get("name", "unknown data")
    selected = manifest.get("selected", "")
    title = MODELS[selected].title if selected in MODELS else selected
    others = len(manifest.get("models", [])) - 1
    return (f"{info.path.stem}  {dataset}  {title}" + (f" +{others}" if others > 0 else "")
            + f"  ({size_text(info.size)})" + STATE_NOTES.get(info.state, ""))


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



def display_prediction(result: Mapping[str, Any], true_label: str | None = None,
                       classes: Sequence[str] = ()) -> None:
    """The prediction: normal traffic, an attack (when the model has a normal class among
    ``classes``) or just the class; then the true label, the confidence and the top
    probabilities when known."""
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
    """Lists the bundle's models (the best one starred) and returns the one picked; Enter keeps
    ``current``."""
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


def header() -> None:
    """The tool's title."""
    print("\n" + "=" * 60)
    print("🛡️  IDS with Machine Learning - CLI Tool")
    print("=" * 60)


def exports_list(exports: Sequence[BundleInfo]) -> None:
    """``--list``: one line per export, newest first."""
    if not exports:
        print(NO_EXPORT)
    for i, info in enumerate(exports, 1):
        print(f"{i}. {export_line(info)}")


def choose_export(exports: Sequence[BundleInfo]) -> Path:
    """The export the user picks (newest first; Enter = the newest whole one)."""
    print("\n📂 Models exported from the web trainer (newest first):")
    for i, info in enumerate(exports, 1):
        print(f"  {i}. {export_line(info)}")
    default = next((i for i, info in enumerate(exports, 1) if info.state == EXPORTED), 1)
    picked = choose(f"\nChoose an export (1-{len(exports)}, Enter = {default}): ", len(exports),
                    default=default)
    return exports[picked - 1].path


def loading_large(path: Path, size: int) -> None:
    """Warns (on stderr) that a large bundle takes a while and needs memory."""
    print(f"⏳ Loading {path.name} ({size_text(size)}): large models can "
          "take minutes and need several times that size in memory...", file=sys.stderr)


def labelled(rows: int, model: str, output: Path, summary: Mapping[str, Any],
             label: str | None) -> None:
    """What ``--csv`` (or menu option 4) did."""
    print(f"✅ {rows} rows predicted with {model}; saved to {output}")
    for count in summary["counts"]:
        print(f"   {count['label']:<30} {count['count']:>8,}")
    if summary["matches"]:
        same = summary["matches"]["matching"]
        print(f"   {same:,} of {rows:,} predictions ({summary['matches']['share']:.1%}) "
              f"match the file's {label} column")


def json_results(results: Sequence[Mapping[str, Any]]) -> None:
    """``--json``: the results as JSON."""
    print(json.dumps(list(results), indent=1))


def ready(model: str) -> None:
    """Says which model the predictions use."""
    print(f"\n🎯 Ready for predictions with {model}!")


def ask_option() -> str:
    """Shows the menu's options and returns the answer as typed."""
    print("\nOptions:")
    print("  1. Enter features manually")
    print("  2. Test with a real example (from the model's test split)")
    print("  3. Switch model")
    print("  4. Predict a CSV file")
    print("  5. Exit")
    return input("\nSelect option: ").strip()


def choose_example(bundle: Bundle, examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Lists the bundle's examples and returns the one picked."""
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example_name(bundle, example)}")
    return examples[choose(f"\nChoose example (1-{len(examples)}): ", len(examples)) - 1]


def ask_csv_path() -> str:
    """A CSV file's path, without the quotes a pasted path may have."""
    return input("CSV file path: ").strip().strip('"')


def no_examples() -> None:
    """Says the bundle has no examples, and which options remain."""
    print("⚠️ This bundle has no examples; use option 1 or 4.")


def using(model: str) -> None:
    """Says which model is now used."""
    print(f"✅ Using {model}")


def invalid_option() -> None:
    """Says the option is not on the menu."""
    print("⚠️ Invalid option")


def error(problem: Exception) -> None:
    """Shows a problem's message (a refused bundle, a bad value, a file)."""
    print(f"❌ {problem}")


def card(bundle: Bundle) -> None:
    """Shows the model card."""
    print(model_card(bundle))


def goodbye() -> None:
    """The last line."""
    print("\n👋 Goodbye!")


def utf8_output() -> None:
    """UTF-8 output even when redirected: Windows then defaults to cp1252, which can't
    print the emoji (or the JSON of non-ASCII class names) and would crash."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and \
                (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8")
