"""What ``toolkit ids --runs`` and ``--compare`` print (v5.3 F2): the saved runs' TrainingRun
records, never a model, in the words and numbers of the web page's saved runs (runsview.js).
``--delete`` (HR4a) names a run's files and asks before anything is removed."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


NO_RUNS = ("No saved runs yet: train a model in the web trainer and click Export (each export "
           "is saved in instance/runs).")
METRICS = (("cv_macro_f1_mean", "CV macro-F1"), ("cv_balanced_accuracy_mean", "CV balanced acc."),
           ("test_macro_f1", "Test macro-F1"), ("test_balanced_accuracy", "Test balanced acc."),
           ("test_accuracy", "Test accuracy"), ("test_min_recall", "Weakest class recall"),
           ("fit_seconds", "Fit time"))
# Why two runs did not score the same test rows (toolkit.core.services.ids.comparison)
REASONS = {
    "DIFFERENT_DATASET": "They were trained on different datasets (the files' SHA-256 differ).",
    "DIFFERENT_TEST_FILE": "Their test rows came from different test files, or only one had a "
                           "test file.",
    "DIFFERENT_LABEL": "They predict different label columns.",
    "DIFFERENT_SPLIT": "Their splits differ (the strategy, the seed, the test size or the "
                       "official mode).",
    "DIFFERENT_ROWS": "They had different numbers of training or test rows.",
}
SETTING_NAMES = {"features": "Features", "models": "Models", "balancing": "Balancing",
                 "selection": "Feature selection"}
NAME_WIDTH = 28  # the first column of a model's numbers
KINDS = {"importance": "importance result", "bundle": "bundle", "record": "record"}
YES = ("y", "yes")


def when(created_at: str) -> str:
    """``2026-10-30T12:05:09Z`` as ``2026-10-30 12:05 UTC``."""
    if len(created_at) >= 16 and created_at[10] == "T":
        return f"{created_at[:10]} {created_at[11:16]} UTC"
    return created_at


def seconds(value: float) -> str:
    """A time: whole seconds from 10 s (a half up, as the web page's Math.round, where
    Python's round() would round 12.5 to 12), else two significant digits."""
    return f"{math.floor(value + 0.5):,} s" if value >= 10 else f"{float(f'{value:.2g}'):g} s"


def number(value: float, key: str) -> str:
    """A score with 3 decimals, or a time."""
    return seconds(value) if key == "fit_seconds" else f"{value:.3f}"


def difference(value: float, key: str) -> str:
    """B − A with its sign: ``+0.012``, ``−0.004``, ``0.000``; one that rounds to zero has no
    sign (not ``+0.000``)."""
    size = number(abs(value), key)
    if not any(digit in size for digit in "123456789"):
        return size
    return f"+{size}" if value > 0 else f"−{size}"


def run_line(run: Mapping[str, Any]) -> str:
    """One saved run: its id, when, on what, how, and the selected model's scores."""
    dataset = run["dataset"]
    data = f"{dataset['name'] or 'unknown data'} ({dataset['rows']:,} rows)"
    if run["test_dataset"]:
        data += f", test file {run['test_dataset']}"
    models = len(run["models"])
    chosen = (f"★ {run['selected_title']} of {models}: CV {run['cv_macro_f1_mean']:.3f}, "
              f"test {run['test_macro_f1']:.3f}") if run["selected_title"] else f"{models} models"
    return (f"{run['run_id']}  {when(run['created_at'])}  {data}  label {run['label']}, "
            f"{run['split']} {run['train_rows']:,}/{run['test_rows']:,}  {chosen}")


def runs_list(answer: Mapping[str, Any]) -> None:
    """``--runs``: one line per saved run, newest first; how to see the older ones."""
    runs = answer["runs"]
    if not runs:
        print(NO_RUNS)
    for i, run in enumerate(runs, 1):
        print(f"{i}. {run_line(run)}")
    if answer["left_out"]:
        print(f"⚠️ {answer['left_out']} file(s) in instance/runs are not runs this toolkit can "
              "show (edited, or from a newer version): left out.")
    if answer["more"] and runs:
        print(f"Older runs: add --before {runs[-1]['run_id']}")


def file_size(size: int) -> str:
    """A file's size as the page writes it (runsview.js): "812 bytes", "2.1 KB", "1.8 GB"."""
    for unit, bytes_ in (("GB", 2**30), ("MB", 2**20), ("KB", 2**10)):
        if size >= bytes_:
            return f"{size / bytes_:.1f} {unit}"
    return f"{size} bytes"


def delete_files(run_id: str, files: Sequence[Any], with_bundle: bool) -> None:
    """``--delete``: the files that go, in order, and the bundle kept unless asked."""
    gone = [item for item in files if with_bundle or item.kind != "bundle"]
    print(f"Deleting run {run_id} removes these files from this computer, in this order; "
          "it can't be undone:")
    for i, item in enumerate(gone, 1):
        print(f"  {i}. {item.name}: {KINDS.get(item.kind, item.kind)}, {file_size(item.size)}")
    for item in files:
        if item.kind == "bundle" and not with_bundle:
            print(f"Its bundle {item.name} ({file_size(item.size)}) is kept: it becomes an "
                  "unregistered bundle in instance/bundles (add --with-bundle to delete it too).")


def confirm_delete() -> bool:
    """The user's yes; anything else, or the end of the input, is no."""
    try:
        return input("Delete these files? [y/N] ").strip().lower() in YES
    except EOFError:
        return False


def deleted(run_id: str, count: int | None) -> None:
    """After ``--delete``: how many files went, or that nothing did (``count`` None)."""
    if count is None:
        print("Nothing deleted.")
    else:
        print(f"✅ Deleted run {run_id}: {count} file(s) removed.")


def _setting(item: Mapping[str, Any], side: str, titles: Mapping[str, str]) -> str:
    """A setting's value in one run, as the comparison prints it."""
    value = item[side]
    if item["setting"] == "features":
        other = set(item["b" if side == "a" else "a"])
        only = [name for name in value if name not in other]
        return f"{len(value)} features" + (f"; only here: {', '.join(only)}" if only else "")
    if item["setting"] == "models":
        return ", ".join(titles.get(name, name) for name in value)
    if item["setting"] == "selection":
        selector, k = value
        return "none" if selector == "none" else selector if k is None else f"{selector}, {k} features"
    return str(value)


def _model_numbers(model: Mapping[str, Any], comparable: bool) -> None:
    """One model of both runs: every number of A and B (and B − A when comparable), then the
    test recall of the classes in both."""
    a_selected, b_selected = model["selected"]
    star = (" (★ selected in both)" if a_selected and b_selected else
            " (★ selected in A)" if a_selected else " (★ selected in B)" if b_selected else "")
    print(f"\n{model['title']}{star}")
    print(f"  {'':<{NAME_WIDTH}}{'A':>10}{'B':>10}" + (f"{'Δ (B − A)':>12}" if comparable else ""))

    def line(name: str, a: str, b: str, d: str) -> None:
        print(f"  {name:<{NAME_WIDTH}}{a:>10}{b:>10}" + (f"{d:>12}" if comparable else ""))

    for key, name in METRICS:
        line(name, number(model["a"][key], key), number(model["b"][key], key),
             difference(model["delta"][key], key) if comparable else "")
    if model["recall"]:
        print("  Test recall per class")
        for item in model["recall"]:
            line(f"  {item['label']}", f"{item['a']:.3f}", f"{item['b']:.3f}",
                 difference(item["delta"], "recall") if comparable else "")


def comparison(answer: Mapping[str, Any]) -> None:
    """``--compare A B``: the two runs, whether they scored the same test rows (Δ = B − A) or
    why not, each model of both, the models of one run only and the settings that differ."""
    titles = answer["titles"]
    print(f"A: {run_line(answer['a'])}")
    print(f"B: {run_line(answer['b'])}")
    comparable = answer["comparable"]
    if comparable:
        print("\n✅ Both runs scored the same test rows, so their numbers can be compared: "
              "Δ = B − A.")
    else:
        print("\n⚠️ These runs did not score the same test rows, so the numbers are side by side, "
              "without a difference:")
        for reason in answer["reasons"]:
            print(f"   - {REASONS.get(reason, reason)}")
    if not answer["models"]:
        print("\nThe two runs have no model in common.")
    for model in answer["models"]:
        _model_numbers(model, comparable)
    for side, names in (("A", answer["only_in_a"]), ("B", answer["only_in_b"])):
        if names:
            print(f"\nOnly in {side}: {', '.join(titles.get(name, name) for name in names)}")
    if not answer["settings"]:
        print("\nSettings: the same features, models, balancing and feature selection.")
        return
    print("\nSettings that differ:")
    for item in answer["settings"]:
        print(f"  {SETTING_NAMES.get(item['setting'], item['setting'])}")
        print(f"    A: {_setting(item, 'a', titles)}")
        print(f"    B: {_setting(item, 'b', titles)}")
