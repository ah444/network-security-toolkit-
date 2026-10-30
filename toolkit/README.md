# `toolkit/`: the package

The Network Security Toolkit is this one Python package. Since v5.2 step 2.3 it is arranged
by interface:

```
toolkit/
  start.py      the one entry point: "web" goes to web/, anything else to cli/
  __main__.py   python -m toolkit (main.py and the installed toolkit command start the same)
  core/         what both tools share: the ML core, the scanner, settings, files, network
  cli/          the command line: the menu, scan, ids
  web/          the web trainer: the page, its routes and jobs, the web command
```

- [`core/`](core/README.md) is used by both interfaces and imports neither.
- [`cli/`](cli/README.md) and [`web/`](web/README.md) use `core/` and never each other.
  The menu starts the web trainer as a child process (`python -m toolkit web`), and
  `start.py` imports only the interface it runs, so `scan` never loads Flask.
- Inside each interface the layers are the same: `ui/` (the only code that prints, asks or
  draws the page), `controllers/` (flags and requests in, answers and exit codes out; the
  only code that uses argparse or Flask), `services/` (what the tools do, with no I/O),
  `data/` (the only code that opens files, sockets, HTTP or child processes). Each layer
  imports only the ones below it.

`tests/test_architecture.py` checks all of this on the real imports, and fails on an old
module name (`toolkit.ids_core`, the v5.1 folders `toolkit/ui`, `controllers`, `services`,
`data`) or a file left in those folders.

Where a usual change goes (a route, a flag, a setting, a model, a dataset, a probe, a page
step): [docs/architecture.md, section 11](../docs/architecture.md#11-where-to-make-a-change).
Every module and what it is for: [section 2](../docs/architecture.md#2-module-map).
