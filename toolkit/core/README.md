# `toolkit/core/`: what both tools share

The ML core and the scanner, and the files, settings and network access they need. Both
interfaces use it ([`cli/`](../cli/README.md), [`web/`](../web/README.md)); it imports
neither, and has no user interface: nothing here prints, asks, or uses Flask or argparse.

```
core/
  services/        what the tools do, with no I/O
    ids/           the ML core: load, split, leakage guard, train, evaluate, bundles, predict,
                   saved runs compared, what a model relies on
    recon/         the scan, start to end: targets and limits, the scan, lookups, report
    ports.py       the three interfaces between layers: ModelStore, IntelProvider, ProgressSink
  data/            files, the network and the environment
    settings.py    every tunable setting and TOOLKIT_* variable, the data folder, .env
    keys.py        API keys and the AI model setting
    logs.py        one rotating log per tool, secrets redacted
    files.py       the text files the commands read and write
    ids/           bundle files and the signing key, run records, dataset files
    recon/         port checks, banner probes, ping, names to addresses, NVD, OpenRouter,
                   AbuseIPDB, scan report files
```

Rules that matter here:

- `services/` never opens a file or a socket: it goes through `data/` or a port.
- `ids` and `recon` never import each other.
- The ML core sees the test split only once, after the model is chosen by cross-validation
  on the training split; the leakage guard and the balancing recommendation read the
  training split only.
- Nothing leaves the computer except through `data/recon/`: NVD gets product names and
  versions, OpenRouter (`--ai` or a yes) the same, AbuseIPDB (`--abuseipdb` or a yes) a
  public address; a private address never. `tests/recon/test_scan_contract.py` holds the
  order: validation, limits and the authorization question first (only the name lookup may
  come before it), then AbuseIPDB, ping, the scan, the lookups.
- A bundle stores its models with the module names of their classes. Moving or renaming a
  module of `services/ids/` breaks every bundle saved before, unless the old name is added
  to `services/ids/legacy_names.py` (and a fixture made by the old code proves it:
  `tests/ml/test_legacy_bundle.py`).
- `data/settings.py` finds the project folder from its own place and module name; it must
  stay `toolkit.core.data.settings` or the test of the paths fails (`tests/test_paths.py`).

Every module: [docs/architecture.md, section 2](../../docs/architecture.md#2-module-map).
Where a change goes: [section 11](../../docs/architecture.md#11-where-to-make-a-change).
