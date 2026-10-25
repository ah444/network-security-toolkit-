# `toolkit/cli/`: the command line

The menu (`python -m toolkit`), the port scanner (`scan`) and the IDS command line (`ids`).
It uses [`core/`](../core/README.md) and never imports [`web/`](../web/README.md).

```
cli/
  ui/
    menu.py         the banner, the menu and its prompt
    scan.py         the scanner's questions, the authorization question, progress, tables
    ids.py          the IDS command's export list, model card, predictions and prompts
  controllers/
    main.py         the menu, or scan / ids with the rest of the line; the toolkit's --help
    scan.py         the scanner's flags or questions, then the scan; the exit codes
    ids.py          choose an export, predict one row, a CSV or a JSON file; the exit codes
  data/
    processes.py    the menu starts each tool as a child process
```

How a command runs: `toolkit/start.py` gives it everything but `web`; `main.py` checks the
command's `TOOLKIT_*` settings (a bad one: one line and exit code 2, before its log starts),
then hands the rest of the line to that command's own parser. The menu runs each tool as
`python -m toolkit <command>` in a child process, so a large bundle's memory is freed when
it ends and Ctrl+C in it brings the menu back; choice 3 starts the web trainer that way,
without importing it.

Rules that matter here:

- Only `ui/` prints or asks; only `controllers/` uses argparse.
- The scan asks "are you authorised" before any packet reaches the target (or takes
  `--i-am-authorized`); the questions and their order are pinned by
  `tests/recon/test_scan_flags.py` and `tests/recon/test_scan_contract.py`.
- Every flag is in [docs/reference.md](../../docs/reference.md) (`tests/test_docs_reference.py`).

Where a change goes: [docs/architecture.md, section 11](../../docs/architecture.md#11-where-to-make-a-change).
