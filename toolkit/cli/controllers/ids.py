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
    python -m toolkit ids --runs              list the saved runs (v5.3 F2)
    python -m toolkit ids --compare A B       two saved runs side by side

(from the project folder; ``toolkit ids`` once installed; moved from
tools/IDS/IDS_CLI - CIC-2017/ in v5.1 step 1.4a; the predictions are
``toolkit.core.services.ids.prediction``, what the user sees ``toolkit.cli.ui.ids``)

Trust (D11): a bundle is opened only if it is signed by this install's key
(instance/signing.key). Nothing else is ever opened.
"""

import argparse
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

from ...core.services.ids.bundle import Bundle, BundleError, BundleVersionWarning
from ...core.services.ids.comparison import NotARun, compare_saved, saved_runs
from ...core.services.ids.datasets import DatasetError
from ...core.services.ids.exports import EXPORTED, BundleStore
from ...core.services.ids.prediction import (
    json_rows,
    label_column,
    label_file,
    output_path,
    predict_values,
)
from ...core.data.logs import setup_logging
from ...core.services.ports import ModelStore
from ..ui import ids as screen
from ..ui import runs as runs_screen

MODEL_STORE: ModelStore = BundleStore()  # this install's exports (instance/bundles)
LARGE_BUNDLE = 500 * 2**20  # bytes: say that loading takes a while


def load_model(bundle_path: str | Path | None = None) -> Bundle:
    """A bundle signed by this install: ``bundle_path``, else the newest whole export.

    Raises BundleError (with a message for the user) for anything else."""
    if bundle_path is None:
        exports = [info for info in MODEL_STORE.list_bundles() if info.state == EXPORTED]
        if not exports:
            raise BundleError("NO_EXPORT", screen.NO_EXPORT)
        bundle_path = exports[0].path
    path = Path(bundle_path)
    if path.is_file() and path.stat().st_size > LARGE_BUNDLE:
        screen.loading_large(path, path.stat().st_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BundleVersionWarning)  # shown in the model card
        return MODEL_STORE.load(path)


def choose_export() -> Path:
    """The export the user picks (newest first; Enter = the newest)."""
    exports = MODEL_STORE.list_bundles()
    if not exports:
        raise BundleError("NO_EXPORT", screen.NO_EXPORT)
    return screen.choose_export(exports)


def list_exports() -> int:
    """``--list``: this install's exports, newest first; exit code 0."""
    screen.exports_list(MODEL_STORE.list_bundles())
    return 0


def list_runs(before: str | None = None) -> int:
    """``--runs``: the saved runs, newest first, 50 at a time (``--before``: the older ones);
    their records only, no model is opened. Exit code 0."""
    runs_screen.runs_list(saved_runs(before))
    return 0


def compare_runs(a: str, b: str) -> int:
    """``--compare A B``: two saved runs side by side, Δ = B − A when they scored the same
    test rows. Exit code 0."""
    runs_screen.comparison(compare_saved(a, b))
    return 0


def listing(args: argparse.Namespace) -> int:
    """``--list``, ``--runs`` or ``--compare``, which open no model; exit code 2 with the message
    for a bad run id, a file that is not a run, or a run that isn't saved."""
    if args.list:
        return list_exports()
    try:
        return list_runs(args.before) if args.runs else compare_runs(*args.compare)
    except NotARun as error:
        screen.error(error)
    except FileNotFoundError as error:
        screen.error(FileNotFoundError(f"There is no saved run {error}."))
    return 2


def run_csv(bundle: Bundle, csv_path: str | Path, output: str | Path | None = None,
            model: str | None = None) -> int:
    """Label every row of a CSV file; the result file is never overwritten."""
    target = output_path(csv_path, output)
    summary = label_file(bundle, csv_path, target, model)
    screen.labelled(summary["rows"], bundle.model(model).name, target, summary,
                    label_column(bundle))
    return 0


def run_json(bundle: Bundle, text: str, model: str | None = None) -> int:
    """Rows given as JSON (an object or a list, inline or a .json file) -> JSON results."""
    screen.json_results([predict_values(bundle, row, model) for row in json_rows(text)])
    return 0


def interactive(bundle: Bundle, model: str | None = None) -> int:
    """The menu: the model card, the model to use, then predictions until Exit. A bad value, a
    missing file or a result file that already exists is shown and the menu goes on."""
    screen.card(bundle)
    model = screen.choose_model(bundle, model or bundle.selected)
    screen.ready(model)
    while True:
        choice = screen.ask_option()
        try:
            if choice == "1":
                screen.display_prediction(
                    predict_values(bundle, screen.manual_input(bundle, model), model),
                    classes=bundle.classes)
            elif choice == "2":
                examples = bundle.manifest.get("examples", [])
                if not examples:
                    screen.no_examples()
                    continue
                example = screen.choose_example(bundle, examples)
                screen.display_prediction(predict_values(bundle, example["values"], model),
                                          true_label=example["label"], classes=bundle.classes)
            elif choice == "3":
                model = screen.choose_model(bundle, model)
                screen.using(model)
            elif choice == "4":
                run_csv(bundle, screen.ask_csv_path(), model=model)
            elif choice == "5":
                screen.goodbye()
                return 0
            else:
                screen.invalid_option()
        except (ValueError, FileExistsError, FileNotFoundError, DatasetError) as error:
            screen.error(error)


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    """The command's flags (each one is in docs/reference.md)."""
    parser = argparse.ArgumentParser(
        prog=prog, description="Intrusion detection with a model bundle exported by the web trainer.")
    parser.add_argument("--bundle", help="a .bundle exported on this computer "
                        "(default: choose one; without a prompt, the newest export)")
    listing = parser.add_mutually_exclusive_group()
    listing.add_argument("--list", action="store_true", help="list the exports and exit")
    listing.add_argument("--runs", action="store_true",
                         help="list the saved runs (newest first, 50 at a time) and exit")
    listing.add_argument("--compare", nargs=2, metavar="RUN_ID",
                         help="compare two saved runs (A then B; differences are B - A) and exit")
    parser.add_argument("--before", metavar="RUN_ID",
                        help="with --runs: the runs saved before this one")
    parser.add_argument("--model", help="which model of the bundle (default: its best)")
    parser.add_argument("--csv", help="label every row of this CSV file")
    parser.add_argument("--output", help="where --csv writes its result "
                        "(default: <file>.predictions.csv; never overwritten)")
    parser.add_argument("--json", help="rows as JSON (object or list, inline or a .json file); "
                        "prints JSON results")
    parser.add_argument("--info", action="store_true", help="show the model card and exit")
    return parser


def parse_args(argv: Sequence[str] | None, prog: str | None = None) -> argparse.Namespace:
    """The flags of ``argv`` (``prog``: the command the help shows)."""
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    if args.before is not None and not args.runs:
        parser.error("--before goes with --runs")
    return args


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    """The command, without exiting: ``--list``, ``--runs``, ``--compare``, the model card
    (``--info``), ``--csv``, ``--json``, or the menu. Only a bundle signed by this install is
    opened. Exit code 0, or 2 with the message for a bundle that is refused, a bad value or a
    file problem; Ctrl+C or the end of input says goodbye (0)."""
    args = parse_args(argv, prog)
    if args.list or args.runs or args.compare:
        return listing(args)
    interactive_mode = not (args.csv or args.json or args.info)
    if interactive_mode:
        screen.header()
    try:
        if interactive_mode and args.bundle is None:
            bundle = load_model(choose_export())
        else:
            bundle = load_model(args.bundle)
        if args.model and args.model not in bundle.models:
            raise ValueError(f"Unknown model {args.model!r}; this bundle has "
                             f"{', '.join(bundle.models)}")
        if args.info:
            screen.card(bundle)
            return 0
        if args.csv:
            return run_csv(bundle, args.csv, args.output, args.model)
        if args.json:
            return run_json(bundle, args.json, args.model)
        return interactive(bundle, args.model)
    except (BundleError, ValueError, FileExistsError, FileNotFoundError, DatasetError) as error:
        screen.error(error)
        return 2
    except (EOFError, KeyboardInterrupt):
        screen.goodbye()
        return 0


def run(argv: Sequence[str] | None = None, prog: str | None = None) -> None:
    """The command: UTF-8 output, the tool's log, then ``main``; exits with its code.
    ``prog`` is the command its help shows (``python -m toolkit ids``)."""
    screen.utf8_output()
    setup_logging("ids_cli")  # instance/logs/ids_cli.log; warnings also on the console
    sys.exit(main(argv, prog))
