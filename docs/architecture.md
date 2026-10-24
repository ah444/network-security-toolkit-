# Architecture

How the code of the Network Security Toolkit is organised **today** (v5.1). The reasons
behind each choice are in [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) §2 (architecture styles)
and §3 (decisions D1–D23); this page describes what was built, and section 10 lists where it
differs from the plan's target structure.

Diagrams are plain ASCII so they read the same in any editor, font or terminal.
`tests/test_docs_architecture.py` fails when a module is missing from the map in section 2.

## 1. Overview

One installable Python package, `toolkit/`, with three tools that share its code:

| Command | Tool | What it does |
|---|---|---|
| `python -m toolkit scan` | port scanner | open TCP/UDP ports, banners, an OS guess, known CVEs |
| `python -m toolkit web` | web trainer | upload a dataset, train and compare models, export a signed bundle |
| `python -m toolkit ids` | IDS command line | predict with a bundle exported by the web trainer |

`python -m toolkit` alone opens a menu that starts each tool as a child process. After
`pip install .` the same commands are `toolkit`, `toolkit scan`, and so on.

The style is a hybrid, as plan §2 recommends:

- **Modular monolith:** one package, one process per tool, no network between modules.
- **Hexagonal core:** the ML core (`ids_core`) and the scanner core (`recon`) talk to the
  outside only through three small interfaces, the ports (section 4).
- **Plugin registries:** models, balancers, feature selectors, dataset profiles, banner
  probes and export formats are table entries, not `if/elif` chains (section 5).
- **Pipe-and-filter ML:** every preprocessing step is inside one scikit-learn/imblearn
  pipeline fitted on the training split only (section 6).
- **Jobs for long tasks:** a training runs in a background job with progress and cancel
  (section 7).
- **Layered web app:** HTML pages plus a versioned JSON API (`/api/v1`, see
  [web_api.md](web_api.md)), with native ES modules in the browser and no build step.

## 2. Module map

Core (no user interface; tested with plain data in, results out):

| Module | Role |
|---|---|
| `toolkit/services/__init__.py` | the services layer: what the tools do, no UI, no I/O |
| `toolkit/services/ids/__init__.py` | the ML core package: no web framework, no `print()`, no network |
| `toolkit/services/ids/datasets.py` | reads a CSV/TXT dataset: encoding, separator, header, dtypes, a load report |
| `toolkit/services/ids/profiles.py` | recognises CIC-IDS2017, NSL-KDD or a generic file from its content |
| `toolkit/services/ids/split.py` | removes exact copies, then splits train/test (official, temporal, group, stratified) |
| `toolkit/services/ids/leakage.py` | the leakage guard: columns that must not be features (profile, name, ID-like, label copies) |
| `toolkit/services/ids/sampling.py` | stratified row samples that keep every class |
| `toolkit/services/ids/balancing.py` | the balancing recommendation, from the training labels only |
| `toolkit/services/ids/registry.py` | the model, balancer and feature-selector registries |
| `toolkit/services/ids/pipeline.py` | builds and fits one pipeline per model; `FittedModel` predicts |
| `toolkit/services/ids/evaluation.py` | cross-validation on train, model choice, one score on test, the leaderboard |
| `toolkit/services/ids/preview.py` | train-only previews for the web trainer (feature scores, class counts) |
| `toolkit/services/ids/runs.py` | `train()` and the `TrainingRun` record of how a model was made |
| `toolkit/services/ids/bundle.py` | packs and checks signed bundles (bytes only) |
| `toolkit/services/ids/legacy_names.py` | the old `toolkit.ids_core` names, only while a bundle saved before step 1.5d loads |
| `toolkit/services/ids/exports.py` | saves, lists and loads runs and bundles; `BundleStore` is the `ModelStore` |
| `toolkit/services/ids/batch.py` | labels every row of a table: the CLI's `--csv` and the web's "Predict a CSV file" |
| `toolkit/recon/__init__.py` | the scanner's command package (until the controller layer) |
| `toolkit/services/recon/__init__.py` | the scanner's logic package |
| `toolkit/services/recon/targets.py` | targets, port lists and scan limits |
| `toolkit/services/recon/os_guess.py` | the OS guess from a ping reply's TTL |
| `toolkit/services/recon/report.py` | one report entry per open port, its network text cleaned |
| `toolkit/services/recon/intel.py` | `Intel`: the lookups for open ports, in parallel, each product once |

Ports, and the data layer (`toolkit/data/`: files, settings, logging, the network):

| Module | Role |
|---|---|
| `toolkit/__init__.py` | the package |
| `toolkit/services/ports.py` | the three ports: `ModelStore`, `IntelProvider`, `ProgressSink` |
| `toolkit/data/__init__.py` | the data layer: files, the network, the environment |
| `toolkit/data/settings.py` | every tunable setting, the `TOOLKIT_*` variables, the data folder, `.env` reading |
| `toolkit/data/logs.py` | one rotating log file per tool, secrets redacted |
| `toolkit/data/keys.py` | API keys and the AI model setting, from the environment or `.env` |
| `toolkit/data/files.py` | the text files the commands read and write |
| `toolkit/data/processes.py` | the menu's child processes (`python -m toolkit <command>`) |
| `toolkit/data/ids/__init__.py` | the IDS's files |
| `toolkit/data/ids/bundle_files.py` | the signing key and the bundle files, as bytes |
| `toolkit/data/ids/run_files.py` | `TrainingRun` records on disk, as dicts (`instance/runs/`) |
| `toolkit/data/ids/dataset_files.py` | a dataset file's first bytes, SHA-256 and row chunks |
| `toolkit/data/recon/__init__.py` | the scanner's network access and report files |
| `toolkit/data/recon/scanner.py` | TCP and UDP port checks of one host |
| `toolkit/data/recon/probes.py` | the banner-probe registry (one entry per port) and product/version parsing |
| `toolkit/data/recon/ping.py` | ping (per-OS flags) and the TTL of the reply |
| `toolkit/data/recon/resolve.py` | host names to addresses |
| `toolkit/data/recon/text.py` | makes text from the network safe to show and export |
| `toolkit/data/recon/scan_files.py` | scan report files as JSON, CSV or TXT in `instance/scans/` |
| `toolkit/data/recon/findings.py` | what the intel gateways return (`CveEntry`, `IntelFinding`) |
| `toolkit/data/recon/nvd.py` | CVEs and CVSS scores from the NVD API (an `IntelProvider`) |
| `toolkit/data/recon/openrouter.py` | the optional AI explanation through OpenRouter (an `IntelProvider`) |
| `toolkit/data/recon/abuseipdb.py` | the optional IP reputation from AbuseIPDB (an `IntelProvider`) |
| `toolkit/data/recon/lookup.py` | the HTTP session, lookup time limits, expected failures and their names |
| `toolkit/data/web/__init__.py` | the web trainer's files and sockets |
| `toolkit/data/web/uploads.py` | uploads on disk, only while they are read |
| `toolkit/data/web/net.py` | the listening socket, and opening the page in the browser |

Presentation (the only code that prints, asks questions or uses Flask):

| Module | Role |
|---|---|
| `toolkit/__main__.py` | `python -m toolkit` |
| `toolkit/cli.py` | the entry point: the menu, or `scan`, `ids`, `web` with the rest of the line |
| `toolkit/recon/cli.py` | the scanner's questions, flags, consent prompts, tables and export |
| `toolkit/ids_cli/__init__.py` | the IDS command-line package |
| `toolkit/ids_cli/__main__.py` | `python -m toolkit.ids_cli` |
| `toolkit/ids_cli/cli.py` | the IDS command line: choose an export, predict, `--csv`, `--json`, `--info` |
| `toolkit/web/__init__.py` | `create_app()` and `run_server()` (waitress on 127.0.0.1) |
| `toolkit/web/__main__.py` | `python -m toolkit.web` |
| `toolkit/web/pages.py` | the two HTML pages |
| `toolkit/web/datasets.py` | upload, preview, class filter and the analyses before training |
| `toolkit/web/training.py` | the model list, training jobs and their results |
| `toolkit/web/bundles.py` | export, download and load bundles; predictions, one row or a CSV file |
| `toolkit/web/resume.py` | `GET /api/v1/workspace`, so the page resumes after a reload |
| `toolkit/web/workspace.py` | the calling session's workspace, locked for the request |
| `toolkit/services/web/__init__.py` | the web trainer's services |
| `toolkit/services/web/workspace.py` | one in-memory workspace per browser session, with memory caps |
| `toolkit/services/web/jobs.py` | the background job runner: one training at a time, progress, cancel |
| `toolkit/services/web/errors.py` | the error codes and `ServiceError` |
| `toolkit/services/web/datasets.py` | uploads read, files combined, the class filter, the training split |
| `toolkit/services/web/training.py` | training settings checked, the job queued, the run kept |
| `toolkit/services/web/prediction.py` | one row or a whole file predicted |
| `toolkit/web/state.py` | what one app keeps between requests (no module globals) |
| `toolkit/web/security.py` | security headers, the CSRF token, the cross-site check, safe file names |
| `toolkit/web/schemas.py` | one small request schema per route |
| `toolkit/web/api.py` | request and error helpers shared by the routes |

The page itself is `toolkit/web/templates/` and `toolkit/web/static/` (one CSS file, ES
modules; `api.js` holds the only `fetch`).

## 3. Dependency rules

Arrows point from a module to what it imports (the real imports, as of this page).

```
 +--------------------------------------------------------------------+
 |  presentation: cli, __main__, ids_cli, web, recon/cli              |
 |  (print, input, Flask, waitress only here; logging_setup.py too)   |
 +------+-------------------------+-------------------------+---------+
        |                         |                         |
        |                         v                         |
        |      +-------------------------------------+      |
        |      | edge code: bundle_files.py,         |      |
        |      | run_files.py (files on disk)        |      |
        |      +------------------+------------------+      |
        |                         |                         |
        v                         v                         v
 +-----------------------------------------+    +------------------------+
 |  ids_core  (ML core)                    |    |  recon (scanner core)  |
 |  no network, no print, no web,          |    |  no print, no web      |
 |  writes no files                        |    |  framework             |
 +--------------------+--------------------+    +-----------+------------+
                      |   ids_core and recon never          |
                      |   import each other                 |
                      v                                     v
 +--------------------------------------------------------------------+
 |  ports.py (interfaces), config.py (settings, data folder)          |
 +--------------------------------------------------------------------+
```

`ports.py` names `Bundle`, `TrainingRun` and the bundle-file types for type checking only
(`if TYPE_CHECKING`), so at run time the ports import no layer.

- No module outside presentation imports a presentation module, prints, or imports
  `flask`, `werkzeug`, `jinja2` or `waitress`: `tests/test_architecture.py`.
- `ids_core` and `recon` never import each other: `tests/test_architecture.py`.
- `ids_core` imports no network library (`requests`, `socket`, `urllib`, `http`) and never
  prints: `tests/ml/test_core_rules.py`.
- `ids_core` writes no files. It reads datasets (`toolkit/services/ids/datasets.py`) and returns
  results; saving runs and bundles is edge code (`toolkit/data/ids/run_files.py`,
  `toolkit/data/ids/bundle_files.py`).
- `mypy --strict` checks the whole package (`tests/test_type_check.py`).

## 4. The three ports

Interfaces exist only where an implementation really is swapped or faked in tests
(`toolkit/services/ports.py`, all `typing.Protocol`):

| Port | Used by | Real implementation | In tests |
|---|---|---|---|
| `ModelStore` | web trainer, IDS command line | `BundleStore` (`toolkit/data/ids/bundle_files.py`): signed bundles in `instance/bundles/` | a scratch instance folder |
| `IntelProvider` | `Intel` in the scanner | `NvdProvider`, `OpenRouterProvider`, `AbuseIpdbProvider` | fake providers; recorded NVD answers in `tests/fixtures/` |
| `ProgressSink` | `train()`, `evaluate()`, `scan_ports()` | a web job (`toolkit/services/web/jobs.py`), the scanner's verbose line | `NullProgress` or a recorder |

Everything else is plain functions and dict registries.

## 5. Registries

| Registry | Where | Entries | To add one |
|---|---|---|---|
| models | `MODELS` in `toolkit/services/ids/registry.py` | random forest, logistic regression, decision tree, KNN, naive Bayes, linear SVM, gradient boosting, AdaBoost, extra trees, XGBoost and LightGBM (`[boost]` extra) | a `ModelSpec`: build function, scaling, encoding, how class weights apply, optional package |
| balancers | `BALANCERS`, same file | none, class weights, SMOTE, SMOTE-ENN, ADASYN, borderline SMOTE, random undersampling, Tomek links, SMOTE-Tomek | a `BalancerSpec` |
| feature selectors | `SELECTORS`, same file | all chosen features, ANOVA F-test, mutual information | a `SelectorSpec` |
| dataset profiles | `PROFILES` in `toolkit/services/ids/profiles.py` | CIC-IDS2017, NSL-KDD, generic | a `DatasetProfile` with the real column names (never guessed) |
| banner probes | `PROBES` and `READERS` in `toolkit/data/recon/probes.py` | 30 ports, 6 ways to read a banner | a `Probe` for the port |
| scan export formats | `WRITERS` in `toolkit/data/recon/scan_files.py` | JSON, CSV, TXT | a writer function |

A model whose optional package is missing (XGBoost, LightGBM without `[boost]`) is simply
not offered (`registry.available_models()`).

## 6. Training flow

`train()` in `toolkit/services/ids/runs.py`; the web trainer runs it in a job. The upload route
loads the file; `train()` starts from the loaded table.

```
 dataset file
     |
     v
 load_dataset ----------> encoding, separator, header, dtypes   (datasets.py)
     |                    + the profile: label, blocked columns, split strategy
     v
 split_dataset ---------> drop rows without a label, remove exact copies,
     |                    then official / temporal / group / stratified split   (split.py)
     |
     +--> train split ------------------------+          test split
     |                                        |              |  (not looked at
     v                                        |              |   until step 5)
 check_leakage (train only) --> features      |              |
     |                                        |              |
     v                                        v              |
 evaluate:                                                   |
   1. balancing recommendation (train labels)                |
   2. K-fold CV on train  --> mean macro-F1 per model        |
   3. fit every model on the whole train split               |
   4. choose the model by CV (never by test score)           |
   5. score each model on the test split once  <-------------+
     |
     v
 TrainingRun (settings, profile, split, leakage, CV and test scores, versions)
     |   kept in the workspace; nothing is written until Export:
     +--> bundle.pack + sign     -> instance/bundles/<run id>.bundle
     +--> run_files.write_run    -> instance/runs/<run id>.json
```

One pipeline per model, all fitted on the training rows only (`toolkit/services/ids/pipeline.py`):

```
 numeric columns:      inf -> missing -> median imputation [-> signed log -> scaling]
 categorical columns:  one-hot (linear, distance models) or ordinal (trees)
          |
          v
 feature selection (optional) -> sampler, e.g. SMOTE (optional) -> model
```

A bundle is `MAGIC | manifest | joblib payload | HMAC-SHA256` (`toolkit/services/ids/bundle.py`).
The manifest embeds the TrainingRun. Nothing is unpickled until the size, layout, signature
(this install's key in `instance/signing.key`), manifest and library versions have passed.

## 7. Web trainer flow

```
 browser (static/js, ES modules)
     |  GET /, /dynamic_ids               HTML; CSP, no inline script
     |  /api/v1/... with X-CSRF-Token     JSON in, JSON out
     v
 security.py:  CSRF token and cross-site check on every state-changing request
     v
 routes:  datasets.py | training.py | bundles.py | resume.py | workspace.py
     |                         |
     v                         | POST /api/v1/jobs/training
 workspace.py:                 v
   one workspace per       jobs.py: one worker thread, one training at a time
   session cookie            for the whole server; progress and cancel
   (in memory: dataset,      through ProgressSink.advance(); a time limit
   run, bundle, last         |
   predictions)              v
     ^                    ids_core.runs.train(...)
     +------ result ---------+
```

- Workspaces live in memory, are freed after `TOOLKIT_WORKSPACE_IDLE_MIN` minutes, and are
  capped by `TOOLKIT_MAX_WORKSPACES` and `TOOLKIT_MAX_DATA_MB`.
- A cancelled or failed job leaves nothing behind: no run record, no bundle.
- `GET /api/v1/workspace` repeats the earlier answers, so a reload resumes where it was.
- The server listens on 127.0.0.1 only (waitress), and the page opens in the browser once
  the port answers.

## 8. Scan flow

`scan_target()` in `toolkit/recon/cli.py`, for each target:

```
 target (IP, name or range) + ports + protocols      targets.py: limits checked first
     |
     +--> public IP, a key, and --abuseipdb (or yes)?  --> AbuseIPDB reputation
     v
 ping_host --> no answer: skipped unless --scan-unresponsive     os_detect.py
     |
     v
 OS guess from the TTL                                           os_detect.py
     |
     v
 scan_ports: TCP connect / UDP probe per port                    scanner.py
     |       banner read with the port's probe                   probes.py
     v
 products from the banners, each one once
     |
     v
 Intel.lookup_many: up to INTEL_WORKERS at a time                intel/
     |   NVD (always; spaced as NVD asks), OpenRouter (--ai or yes, and a key)
     v
 report entries --> table on screen --> export (--export json|csv|txt)   export.py
```

A failed lookup is recorded in the report and the scan goes on. Ctrl+C stops the scan with
exit code 130.

## 9. Data on disk and what leaves the computer

```
 instance/                    (gitignored; TOOLKIT_INSTANCE_DIR moves it)
   signing.key                this install's bundle key: back it up
   bundles/<run id>.bundle    exports of the web trainer
   runs/<run id>.json         TrainingRun records (retrain from these)
   scans/scan-<time>.<fmt>    scanner exports
   uploads/                   web uploads, deleted once read
   logs/<tool>.log            one rotating log per tool, secrets redacted
 .env                         optional API keys
```

- From the project folder (a clone, or `pip install -e .`), `instance/` and `.env` are
  in the project folder.
- An installed copy (`pip install .`) uses a per-user folder: `%LOCALAPPDATA%` on Windows,
  `~/Library/Application Support` on macOS, `$XDG_DATA_HOME` or `~/.local/share` on Linux;
  then `network-security-toolkit`. `toolkit --help` prints the folder in use
  (`toolkit/data/settings.py`).

```
 this computer                                  the internet
 +------------------------------------+
 | web trainer, IDS, bundles, datasets| ---- nothing ----
 |                                    |
 | scanner ---- product names --------|------> NVD (CVE lookups; always)
 |         ---- product names --------|------> OpenRouter (--ai, or yes; and a key)
 |         ---- the public IP --------|------> AbuseIPDB (--abuseipdb, or yes; and a key)
 +------------------------------------+
```

"or yes" is the scanner's question when it runs without flags. Each optional lookup prints
one line saying what is sent, or that its key is missing.

## 10. Where this differs from the plan's target structure

The plan's §2 "Target structure" was written before the code. What was built differs here:

| Plan | Built | Why |
|---|---|---|
| `services/` (TrainingService, ScanService, PredictionService) | no services layer: the web routes and CLIs call `ids_core` and `recon` directly | each service would have had one caller; `train()` and `scan_target()` already are the use cases |
| `ids_core/ports.py` | `toolkit/services/ports.py` | the ports serve the scanner too, not only the ML core |
| `ids_core/explain.py` (feature importance, SHAP) | not built | explainability (F3) is planned for v5.2 |
| `recon/probes/` with one file per service | one `toolkit/data/recon/probes.py` with a registry | each probe is a table entry plus a few shared readers |
| `recon/report.py` with HTML/PDF | `toolkit/data/recon/scan_files.py` with JSON, CSV, TXT | HTML/PDF reports (F5) are later features |
| `.github/workflows/ci.yml` | none | checks run locally (`python -m pytest`, ruff, mypy, the pre-commit hook), as plan D19 decided |
| a `train` command in the CLI | none | models are trained in the web trainer (plan D12) |
