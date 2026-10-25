# Architecture

How the code of the Network Security Toolkit is organised **today** (v5.2: arranged by
interface since step 2.3, in layers since v5.1 step 1.5d). The reasons
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

- **Interfaces and layers:** `core/` (what both tools share), `cli/` (the command line)
  and `web/` (the web trainer); `cli` and `web` use `core` and never each other. Inside
  each, the same layers `ui/` → `controllers/` → `services/` → `data/` (section 3). There
  is no database: the data layer is files, the network and the environment.
- **Modular monolith:** one package, one process per tool, no network between modules.
- **Hexagonal core:** the ML core (`core/services/ids`) and the scanner's logic
  (`core/services/recon`) reach the outside only through the data layer and three small
  interfaces, the ports (section 4).
- **Plugin registries:** models, balancers, feature selectors, dataset profiles, banner
  probes and export formats are table entries, not `if/elif` chains (section 5).
- **Pipe-and-filter ML:** every preprocessing step is inside one scikit-learn/imblearn
  pipeline fitted on the training split only (section 6).
- **Jobs for long tasks:** a training runs in a background job with progress and cancel
  (section 7).
- **Layered web app:** HTML pages plus a versioned JSON API (`/api/v1`, see
  [web_api.md](web_api.md)), with native ES modules in the browser and no build step.

## 2. Module map

The package is arranged by interface (v5.2 step 2.3,
[toolkit/README.md](../toolkit/README.md)): `core/` what both tools share, `cli/` the
command line, `web/` the web trainer. Inside each are the same layers (v5.1 step 1.5d):
`ui/` what the user sees, `controllers/` requests and flags in and answers out, `services/`
what the tools do, `data/` files, the network and the environment. There is no database.
Section 11 says where a usual change goes.

### The package

What starts the toolkit.

| Module | Layer | Role |
|---|---|---|
| `toolkit/__init__.py` | - | the package |
| `toolkit/__main__.py` | - | `python -m toolkit` |
| `toolkit/start.py` | - | the one entry point (`python -m toolkit`, `main.py`, the `toolkit` command): `web` to the web trainer's command, the rest to the command line |

### `core/`: what both tools share

The ML core and the scanner's logic, and the files and network they use; no user interface,
no Flask, no argparse. Read by `cli/` and `web/`, it imports neither
([toolkit/core/README.md](../toolkit/core/README.md)).

| Module | Layer | Role |
|---|---|---|
| `toolkit/core/__init__.py` | - | what both tools share (v5.2 step 2.3b): `core/services/` and `core/data/` |
| `toolkit/core/services/__init__.py` | services | the shared services |
| `toolkit/core/services/ports.py` | services | the four ports: `ModelStore`, `IntelProvider`, `ProgressSink`, `ScanEngine` |
| `toolkit/core/services/ids/__init__.py` | services | the ML core: no web framework, no `print()`, no network |
| `toolkit/core/services/ids/datasets.py` | services | a CSV/TXT dataset understood: encoding, separator, profile, header, dtypes, a load report |
| `toolkit/core/services/ids/profiles.py` | services | recognises CIC-IDS2017, NSL-KDD or a generic file from its content |
| `toolkit/core/services/ids/split.py` | services | removes exact copies, then splits train/test (official, temporal, group, stratified) |
| `toolkit/core/services/ids/split_strategies.py` | services | the strategies: each returns the train and test rows, or the reason code its prerequisite failed with |
| `toolkit/core/services/ids/split_time.py` | services | the time column of a temporal split: the empty and unparseable shares, `d/m/Y` or `m/d/Y`, capture files that overlap in time |
| `toolkit/core/services/ids/leakage.py` | services | the leakage guard: columns that must not be features (profile, name, ID-like, label copies) |
| `toolkit/core/services/ids/sampling.py` | services | stratified row samples that keep every class |
| `toolkit/core/services/ids/balancing.py` | services | the balancing recommendation, from the training labels only |
| `toolkit/core/services/ids/registry.py` | services | the model, balancer and feature-selector registries |
| `toolkit/core/services/ids/pipeline.py` | services | builds and fits one pipeline per model; `FittedModel` predicts |
| `toolkit/core/services/ids/evaluation.py` | services | cross-validation on train, model choice, one score on test, the leaderboard |
| `toolkit/core/services/ids/preview.py` | services | train-only previews for the web trainer (feature scores, class counts) |
| `toolkit/core/services/ids/runs.py` | services | `train()` and the `TrainingRun` record of how a model was made |
| `toolkit/core/services/ids/run_json.py` | services | the record's plain-JSON form (`TrainingRun.to_dict()`): any value as JSON types, what is kept of the profile, split, training and pipelines |
| `toolkit/core/services/ids/bundle.py` | services | packs and checks signed bundles (bytes only) |
| `toolkit/core/services/ids/legacy_names.py` | services | the old names (`toolkit.ids_core`, before step 1.5d; `toolkit.services.ids`, until v5.2 step 2.3a), only while a bundle saved with them loads |
| `toolkit/core/services/ids/exports.py` | services | saves, lists and loads runs and bundles; `BundleStore` is the `ModelStore` |
| `toolkit/core/services/ids/batch.py` | services | labels every row of a table: the CLI's `--csv` and the web's "Predict a CSV file" |
| `toolkit/core/services/ids/prediction.py` | services | the IDS command's predictions: one row, a table, a CSV file into a new file |
| `toolkit/core/services/recon/__init__.py` | services | the scanner's logic |
| `toolkit/core/services/recon/targets.py` | services | targets, port lists and scan limits |
| `toolkit/core/services/recon/scan.py` | services | a scan, start to end, through a `ScanView` (what it shows and asks); a set `stop` event ends it (`ScanStopped`), nothing more sent |
| `toolkit/core/services/recon/intel.py` | services | `Intel`: the lookups for open ports, in parallel, each product once; none asked once the scan is stopped |
| `toolkit/core/services/recon/engines.py` | services | the scan engines by name: the built-in one (the toolkit's own scanner) and nmap (`NmapEngine`; `NmapProgress`: nmap's phases as one bar); `engine_problem` (why an engine can't do a scan here, asked before the authorization) and `describe` (the opening notice's lines) |
| `toolkit/core/services/recon/os_guess.py` | services | the OS guess from the TTL, the banners, the open ports and nmap's OS match, with its evidence; this computer said |
| `toolkit/core/services/recon/report.py` | services | one report entry per open port, its network text cleaned |
| `toolkit/core/data/__init__.py` | data | the shared data: files, the network, the environment |
| `toolkit/core/data/settings.py` | data | every tunable setting, the `TOOLKIT_*` variables, the data folder, `.env` reading |
| `toolkit/core/data/logs.py` | data | one rotating log file per tool, secrets redacted |
| `toolkit/core/data/keys.py` | data | API keys and the AI model setting, from the environment or `.env` |
| `toolkit/core/data/files.py` | data | the text files the commands read and write |
| `toolkit/core/data/ids/__init__.py` | data | the IDS's files |
| `toolkit/core/data/ids/bundle_files.py` | data | the signing key and the bundle files, as bytes |
| `toolkit/core/data/ids/run_files.py` | data | `TrainingRun` records on disk, as dicts (`instance/runs/`) |
| `toolkit/core/data/ids/dataset_files.py` | data | a dataset file's first bytes, SHA-256 and row chunks |
| `toolkit/core/data/recon/__init__.py` | data | the scanner's network access and report files |
| `toolkit/core/data/recon/scanner.py` | data | TCP and UDP port checks of one host |
| `toolkit/core/data/recon/probes.py` | data | the banner-probe registry (one entry per port) |
| `toolkit/core/data/recon/services.py` | data | which service answered: from its answer, else the port's usual name with a "?" |
| `toolkit/core/data/recon/software.py` | data | the product and version in a banner (the only part of it looked up online), and nmap's product and version as the same names (`nmap_software`) |
| `toolkit/core/data/recon/ping.py` | data | ping (per-OS flags) and the TTL of the reply |
| `toolkit/core/data/recon/resolve.py` | data | host names to addresses |
| `toolkit/core/data/recon/nmap.py` | data | the optional nmap engine's program: found by its full path (`TOOLKIT_NMAP_PATH`, `PATH`, its install folder; never the current folder), its version (7.80 or newer), its command from an allow-list only; whether it can send raw packets (`has_privileges`: Npcap usable on Windows, root elsewhere) |
| `toolkit/core/data/recon/nmap_run.py` | data | running nmap: its XML read as it comes, the progress from `<taskprogress>`, a stop, Ctrl+C or a limit ends it (asked, killed 5 s later), its failures in plain words |
| `toolkit/core/data/recon/nmap_xml.py` | data | nmap's XML as the scanner's `PortResult`s (the built-in scanner's states, nmap's own in `reason`), the OS matches and host scripts; every text from the network cleaned and cut |
| `toolkit/core/data/recon/local.py` | data | whether an address is this computer's own (nothing sent), and its OS |
| `toolkit/core/data/recon/text.py` | data | makes text from the network safe to show and export |
| `toolkit/core/data/recon/scan_files.py` | data | scan reports as JSON, CSV or TXT: as text (`report_text`, for a download) or a file in `instance/scans/` (with the engine and nmap's CPE names) |
| `toolkit/core/data/recon/findings.py` | data | what the intel gateways return (`CveEntry`, `IntelFinding`) |
| `toolkit/core/data/recon/nvd.py` | data | CVEs and CVSS scores from the NVD API, by version (CPE names) or by keyword (an `IntelProvider`) |
| `toolkit/core/data/recon/openrouter.py` | data | the optional AI explanation through OpenRouter (an `IntelProvider`) |
| `toolkit/core/data/recon/abuseipdb.py` | data | the optional IP reputation from AbuseIPDB (an `IntelProvider`) |
| `toolkit/core/data/recon/lookup.py` | data | the HTTP session (it counts the requests to each host), lookup time limits, expected failures and their names |

### `cli/`: the command line

The menu, `scan` and `ids`; it never imports the web trainer
([toolkit/cli/README.md](../toolkit/cli/README.md)).

| Module | Layer | Role |
|---|---|---|
| `toolkit/cli/__init__.py` | - | the command line (v5.2 step 2.3c): the menu, `scan`, `ids`; it never imports the web trainer |
| `toolkit/cli/ui/__init__.py` | ui | the command line's prompts and printing |
| `toolkit/cli/ui/menu.py` | ui | the banner, the menu and its prompt |
| `toolkit/cli/ui/scan.py` | ui | the scanner's questions, authorization, progress line, tables and report; `ConsoleScan` is the scan's `ScanView` |
| `toolkit/cli/ui/ids.py` | ui | the IDS command's export list, model card, predictions, interactive menu and prompts |
| `toolkit/cli/controllers/__init__.py` | controllers | the command line's commands |
| `toolkit/cli/controllers/main.py` | controllers | the command line: the menu, or `scan`, `ids` with the rest of the line; the toolkit's `--help` (it lists `web` too) |
| `toolkit/cli/controllers/scan.py` | controllers | the scanner's flags or questions, a `ScanRequest`, the exit codes |
| `toolkit/cli/controllers/ids.py` | controllers | the IDS command: choose an export, predict, `--csv`, `--json`, `--info`, the exit codes |
| `toolkit/cli/data/__init__.py` | data | the command line's own data |
| `toolkit/cli/data/processes.py` | data | the menu's child processes (`python -m toolkit <command>`) |

### `web/`: the web trainer

The page, its routes and jobs, and the `web` command; it never imports the command line
([toolkit/web/README.md](../toolkit/web/README.md)).

| Module | Layer | Role |
|---|---|---|
| `toolkit/web/__init__.py` | - | the web trainer (v5.2 step 2.3d): `ui/`, `controllers/`, `services/`, `data/`; it never imports the command line |
| `toolkit/web/ui/__init__.py` | ui | the web trainer's page: `templates/` and `static/` (one CSS file, ES modules; `api.js` holds the only `fetch`) |
| `toolkit/web/ui/console.py` | ui | what the web trainer's command prints |
| `toolkit/web/controllers/command.py` | controllers | the web trainer's command (from `toolkit.start`): its settings checked, `--port` and `--no-browser`, then the server |
| `toolkit/web/controllers/__init__.py` | controllers | `create_app()`: the Flask app with its own state, serving `toolkit/web/ui` |
| `toolkit/web/controllers/server.py` | controllers | `run_server()`: waitress on 127.0.0.1, the port bound exclusively, the browser opened |
| `toolkit/web/controllers/pages.py` | controllers | the two HTML pages |
| `toolkit/web/controllers/datasets.py` | controllers | upload, preview, class filter and the analyses before training |
| `toolkit/web/controllers/training.py` | controllers | the model list, a training job started, its results |
| `toolkit/web/controllers/jobs.py` | controllers | any job's state, progress and cancel (`GET`/`DELETE /api/v1/jobs/<id>`) |
| `toolkit/web/controllers/bundles.py` | controllers | export, download and load bundles; predictions, one row or a CSV file |
| `toolkit/web/controllers/resume.py` | controllers | `GET /api/v1/workspace`, so the page resumes after a reload |
| `toolkit/web/controllers/workspace.py` | controllers | the calling session's workspace (the cookie holds its ID), locked for the request |
| `toolkit/web/controllers/state.py` | controllers | what one app keeps between requests (no module globals) |
| `toolkit/web/controllers/security.py` | controllers | security headers, the Host check (against DNS rebinding), the CSRF token, the cross-site check, safe file names |
| `toolkit/web/controllers/schemas.py` | controllers | one small request schema per route |
| `toolkit/web/controllers/api.py` | controllers | request and error helpers; the HTTP status of each `ServiceError` |
| `toolkit/web/services/__init__.py` | services | the web trainer's services |
| `toolkit/web/services/workspace.py` | services | one in-memory workspace per browser session, with memory caps |
| `toolkit/web/services/jobs.py` | services | the background jobs: a queue per kind (training, scan), one job of a kind at a time, progress, cancel |
| `toolkit/web/services/errors.py` | services | the error codes and `ServiceError` |
| `toolkit/web/services/datasets.py` | services | uploads read, files combined, the class filter, the training split |
| `toolkit/web/services/training.py` | services | training settings checked, the job queued, the run kept |
| `toolkit/web/services/scan.py` | services | the web port scan: the choices checked (the authorization first, nothing sent before it), the scan queued as a job, the last scan kept for the download |
| `toolkit/web/services/scan_view.py` | services | `WebScan`, the page's `ScanView`: a running scan's log, its progress host by host and its answer, no path of this computer shown |
| `toolkit/web/services/prediction.py` | services | one row or a whole file predicted |
| `toolkit/web/data/__init__.py` | data | the web trainer's files and sockets |
| `toolkit/web/data/uploads.py` | data | uploads on disk, only while they are read |
| `toolkit/web/data/net.py` | data | the listening socket, and opening the page in the browser |

## 3. Dependency rules

Two rules, checked on the real imports by `tests/test_architecture.py`. First the
interfaces; arrows point from a package to what it imports:

```
                 toolkit/start.py  (python -m toolkit, main.py, the toolkit command)
                   |            |
         "web"     |            |   anything else
                   v            v
 +--------------------+      +--------------------+
 |  web/              |      |  cli/              |
 |  the web trainer:  |  X   |  the command line: |
 |  page, routes,     |<---->|  menu, scan, ids   |
 |  jobs, web command |      |                    |
 +---------+----------+      +----------+---------+
           |                            |
           v                            v
 +---------------------------------------------------------+
 |  core/   services/: ids (ML core, bundles, predictions), |
 |                     recon (targets, the scan, lookups)   |
 |          data/:     settings, keys, logs, files,         |
 |                     ids (bundle, run, dataset files),    |
 |                     recon (sockets, ping, banners, APIs) |
 |  imports neither web/ nor cli/                           |
 +---------------------------------------------------------+
```

`web/` and `cli/` never import each other (X): the menu starts the web trainer as a child
process (`python -m toolkit web`), and `toolkit/start.py` imports only the interface it runs.

Then the layers, the same inside every interface; each imports only the layers below it:

```
 +---------------------------------------------------------------------+
 |  ui/            web/ui (the page: templates, CSS, ES modules;        |
 |                 the web command's console lines)                     |
 |                 cli/ui (menu, scan, ids: prompts and printing)       |
 |                 the only code that prints or asks                    |
 +----------------------+----------------------------------------------+
        ^ uses          | may import services' types, to show them
        |               v
 +------+--------------------------------------------------------------+
 |  controllers/   web/controllers (Flask app, routes, the web command) |
 |                 cli/controllers (the menu, scan, ids)                |
 |                 the only code that uses Flask (web) or argparse      |
 +----------------------+----------------------------------------------+
                        | calls services; from data only settings, keys,
                        | logs, and what starts things (socket, processes)
                        v
 +---------------------------------------------------------------------+
 |  services/      core/services (ids, recon, the ports)                |
 |                 web/services (workspaces, jobs, the work behind the  |
 |                 routes)                                              |
 |                 no print, no Flask, no files, no network             |
 +----------------------+----------------------------------------------+
                        |
                        v
 +---------------------------------------------------------------------+
 |  data/          core/data (settings, keys, logs, files, ids, recon)  |
 |                 cli/data (the menu's child processes)                |
 |                 web/data (uploads, the listening socket, the browser)|
 |                 the only code that opens files, sockets, HTTP or     |
 |                 processes; imports nothing above it (no database)    |
 +---------------------------------------------------------------------+
```

In the web trainer the page (`web/ui`) calls the controllers (`web/controllers`) over HTTP. A
terminal tool starts in its command (`cli/controllers`), which asks `cli/ui` to show each
screen; the scan reports what it does through a `ScanView`
(`toolkit/core/services/recon/scan.py`), which `ConsoleScan` (`toolkit/cli/ui/scan.py`)
prints.

What `tests/test_architecture.py` checks, for every module of `toolkit/`:

- `core` imports only `core`; `cli` and `web` import themselves and `core`; the package
  itself (`toolkit/start.py`, `__main__.py`) only the interfaces' controllers.
- Imports go only downward (the second diagram); `data/` imports nothing from the layers
  above.
- Controllers use from `data/` only the settings, keys and logs, the web server's socket
  and the menu's child processes; everything else goes through a service.
- Only `ui/` calls `print()` or `input()`; only `web/controllers` imports Flask, Werkzeug,
  Jinja2 or waitress; only `cli/controllers` and the web command
  (`toolkit/web/controllers/command.py`) import argparse.
- Only `data/` imports `socket`, `ssl`, `requests`, `urllib`, `http`, `subprocess` or
  `webbrowser`, or opens, reads, writes or lists files.
- Inside `core` the tools stay apart: `ids` and `recon` never import each other.
- No module imports an old name (`toolkit.ids_core`, `toolkit.recon`, the v5.1 layers
  `toolkit.ui`, `toolkit.controllers`, `toolkit.services`, `toolkit.data`, ...), and those
  four folders hold no file. Bundles name their classes' modules inside them: bundles saved
  before v5.1 step 1.5d name `toolkit.ids_core`, those saved until v5.2 step 2.3a
  `toolkit.services.ids`; while one is loaded, `toolkit/core/services/ids/legacy_names.py`
  points those names to the modules of today (`tests/ml/test_legacy_bundle.py`).

`toolkit/core/services/ports.py` names `Bundle`, `TrainingRun` and the bundle-file types for
type checking only (`if TYPE_CHECKING`), so at run time the ports import no other module. The
services rules again, from the ML core's side: `tests/ml/test_core_rules.py`. `mypy
--strict` checks the whole package (`tests/test_type_check.py`).

## 4. The four ports

Interfaces exist only where an implementation really is swapped or faked in tests
(`toolkit/core/services/ports.py`, all `typing.Protocol`):

| Port | Used by | Real implementation | In tests |
|---|---|---|---|
| `ModelStore` | web trainer, IDS command line | `BundleStore` (`toolkit/core/services/ids/exports.py`, over `toolkit/core/data/ids/bundle_files.py`): signed bundles in `instance/bundles/` | a scratch instance folder |
| `IntelProvider` | `Intel` in the scanner | `NvdProvider`, `OpenRouterProvider`, `AbuseIpdbProvider` (`toolkit/core/data/recon/`) | fake providers; recorded NVD answers in `tests/fixtures/` |
| `ProgressSink` | `train()`, `evaluate()`, `scan_ports()` | a web job (`toolkit/web/services/jobs.py`), the scanner's verbose line (`toolkit/cli/ui/scan.py`) | `NullProgress` or a recorder |
| `ScanEngine` | `scan_target` in the scanner, by the request's `engine` name | `BuiltinEngine` (`toolkit/core/services/recon/engines.py`, over `scan_ports` in `toolkit/core/data/recon/scanner.py`); `NmapEngine` (the same module, over `toolkit/core/data/recon/nmap.py`, `nmap_run.py`, `nmap_xml.py`) when nmap is installed | the `fake_ports` fixture (`tests/recon/conftest.py`), an engine put in `ENGINES` |

Everything else is plain functions and dict registries.

## 5. Registries

| Registry | Where | Entries | To add one |
|---|---|---|---|
| models | `MODELS` in `toolkit/core/services/ids/registry.py` | random forest, logistic regression, decision tree, KNN, naive Bayes, linear SVM, gradient boosting, AdaBoost, extra trees, XGBoost and LightGBM (`[boost]` extra) | a `ModelSpec`: build function, scaling, encoding, how class weights apply, optional package |
| balancers | `BALANCERS`, same file | none, class weights, SMOTE, SMOTE-ENN, ADASYN, borderline SMOTE, random undersampling, Tomek links, SMOTE-Tomek | a `BalancerSpec` |
| feature selectors | `SELECTORS`, same file | all chosen features, ANOVA F-test, mutual information | a `SelectorSpec` |
| dataset profiles | `PROFILES` in `toolkit/core/services/ids/profiles.py` | CIC-IDS2017, NSL-KDD, generic | a `DatasetProfile` with the real column names (never guessed) |
| banner probes | `PROBES` and `READERS` in `toolkit/core/data/recon/probes.py` | 34 ports (8 of them printers, which get nothing sent), 7 ways to read a banner | a `Probe` for the port |
| signs of an OS | `BANNER_WORDS` and `PORT_SIGNS` in `toolkit/core/services/recon/os_guess.py` | 11 banner words (Windows, 7 Linux distributions, 3 BSDs), 1 port (135, Microsoft RPC) | a pattern or a port, its family and what it says |
| NVD product names | `CPE_PRODUCTS` in `toolkit/core/data/recon/nvd.py` | 17 products, 18 CPE names (MySQL has two), each checked against NVD | a product name and its NVD `vendor:product` |
| services an answer shows | `RULES` in `toolkit/core/data/recon/services.py` | SSH, HTTP, VNC, POP3, IMAP, FTP, SMTP, Redis, Memcached, MySQL, MongoDB, AMQP | a pattern and the service's name |
| software in a banner | `RULES` in `toolkit/core/data/recon/software.py` | SSH, MariaDB, Redis, Memcached; then the generic "Name/1.2" | a `Rule`: a pattern and how its match is named |
| scan export formats | `WRITERS` in `toolkit/core/data/recon/scan_files.py` | JSON, CSV, TXT | a writer function |

A model whose optional package is missing (XGBoost, LightGBM without `[boost]`) is simply
not offered (`registry.available_models()`).

## 6. Training flow

`train()` in `toolkit/core/services/ids/runs.py`; the web trainer runs it in a job. The
upload route loads the file; `train()` starts from the loaded table.

```
 dataset file
     |
     v
 load_dataset ----------> encoding, separator, header, dtypes   (datasets.py)
     |                    + the profile: label, blocked columns, split strategy
     v
 split_dataset ---------> drop rows without a label, remove exact copies,
     |                    then official / temporal / group / stratified split   (split.py,
     |                    the strategies in split_strategies.py)
     |                    (a temporal split checks the time column first: split_time.py)
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
     +--> exports.write_run      -> instance/runs/<run id>.json
```

One pipeline per model, all fitted on the training rows only
(`toolkit/core/services/ids/pipeline.py`):

```
 numeric columns:      inf -> missing -> median imputation [-> signed log -> scaling]
 categorical columns:  one-hot (linear, distance models) or ordinal (trees)
          |
          v
 feature selection (optional) -> sampler, e.g. SMOTE (optional) -> model
```

A bundle is `MAGIC | manifest | joblib payload | HMAC-SHA256`
(`toolkit/core/services/ids/bundle.py`).
The manifest embeds the TrainingRun. Nothing is unpickled until the size, layout, signature
(this install's key in `instance/signing.key`), manifest and library versions have passed.

## 7. Web trainer flow

```
 web/ui: the page (static/js, ES modules)
     |  GET /, /dynamic_ids               HTML; CSP, no inline script
     |  /api/v1/... with X-CSRF-Token     JSON in, JSON out
     v
 web/controllers/security.py:  the Host check on every request; CSRF token and cross-site
                               check on every change
     v
 web/controllers routes:  datasets | training | bundles | resume | workspace
     |   request checked (schemas.py), the session's workspace found by its
     |   cookie (workspace.py), JSON answers shaped here
     v
 web/services:  datasets.py (uploads read, files combined, the training split)
     |          training.py (settings checked, the job queued) ----+
     |          prediction.py (one row, a whole file)               |
     |          workspace.py: one workspace per session, in memory  v
     |            (dataset, run, bundle, last predictions)   jobs.py: a worker
     |                                                       thread per kind, one
     |                                                       training and one scan
     |                                                       at a time; progress
     |                                                       and cancel through
     v                                                       ProgressSink.advance()
 core/services/ids:  runs.train(...), bundles, batch predictions <-+
     v
 web/data, core/data:  uploads on disk only while read, bundle and run files,
                       the signing key
```

- Workspaces live in memory, are freed after `TOOLKIT_WORKSPACE_IDLE_MIN` minutes, and are
  capped by `TOOLKIT_MAX_WORKSPACES` and `TOOLKIT_MAX_DATA_MB`.
- A refusal is a `ServiceError` with a code (`toolkit/web/services/errors.py`); the routes
  answer it with the HTTP status it has always had (`toolkit/web/controllers/api.py`).
- A cancelled or failed job leaves nothing behind: no run record, no bundle.
- `GET /api/v1/workspace` repeats the earlier answers, so a reload resumes where it was.
- The server listens on 127.0.0.1 only (waitress, `toolkit/web/controllers/server.py`), and
  the page opens in the browser once the port answers.

## 8. Scan flow

`scan_target()` in `toolkit/core/services/recon/scan.py`, for each target; every line on the
screen and every question goes through the `ScanView` (`ConsoleScan` in
`toolkit/cli/ui/scan.py`):

```
 target (IP, name or range) + ports + protocols   core/services/recon/targets.py: limits first
     |
     +--> public IP, a key, --abuseipdb (or yes)? --> AbuseIPDB    core/data/recon/abuseipdb.py
     v
 ping --> no answer: skipped unless --scan-unresponsive        core/data/recon/ping.py
     |
     v
 OS guess from the TTL                                          core/services/recon/os_guess.py
     |
     v
 the engine (ScanRequest.engine, --engine or the question)      core/services/recon/engines.py
 builtin: TCP connect / UDP probe per port                      core/data/recon/scanner.py
 nmap: its command, its run, its XML              core/data/recon/nmap.py, nmap_run.py, nmap_xml.py
     |       banner read with the port's probe                  core/data/recon/probes.py
     |       the service from the answer, else "usual?"         core/data/recon/services.py
     v
 OS hint: TTL, banners, open ports, nmap's match; this PC?      core/services/recon/os_guess.py
     |
     v
 products from nmap or the banners, each one once               core/data/recon/software.py
     |
     v
 Intel.lookup_many: up to INTEL_WORKERS at a time               core/services/recon/intel.py
     |   NVD (always; spaced as NVD asks): by version, else by keyword
     |   OpenRouter (--ai or yes, and a key)
     v
 report entries (network text cleaned)                          core/services/recon/report.py
     --> table on screen --> export (--export json|csv|txt)     core/data/recon/scan_files.py
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
  (`toolkit/core/data/settings.py`).

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

"or yes" is the scanner's question when it runs without flags. Each scan prints, before the
first target, what it may send to each service or why it sends nothing (`online_plan` in
`toolkit/core/services/recon/scan.py`: no key, not asked for, the answer no, no public
address), and in its summary how many requests went to each (`Intel.sent`, counted by the
scan's `Session` in `toolkit/core/data/recon/lookup.py`).

## 10. Where this differs from the plan's target structure

The plan's §2 "Target structure" was written before the code. What was built differs here:

| Plan | Built | Why |
|---|---|---|
| `services/` with TrainingService, ScanService, PredictionService classes | `services/` (step 1.5d) with one folder per tool and plain functions: `toolkit/web/services/training.py`, `toolkit/core/services/recon/scan.py`, `toolkit/core/services/ids/prediction.py` | a class with one method per use case would add nothing; the layer rules are what keep it clean |
| presentation and core only | four layers (`ui/`, `controllers/`, `services/`, `data/`, step 1.5d) | each kind of code has one place: a new route, flag, rule or file format has an obvious home |
| one tree of layers, with one folder per tool in each (v5.1) | three interfaces, `core/`, `cli/`, `web/`, each with the same layers (v5.2 step 2.3) | a change to one interface stays in its folder; the command line and the web trainer can't depend on each other by accident |
| `ids_core/ports.py` | `toolkit/core/services/ports.py` | the ports serve the scanner too, not only the ML core |
| `ids_core/explain.py` (feature importance, SHAP) | not built | explainability (F3) is planned for v5.3 |
| `recon/probes/` with one file per service | one `toolkit/core/data/recon/probes.py` with a registry | each probe is a table entry plus a few shared readers |
| `recon/report.py` with HTML/PDF | `toolkit/core/services/recon/report.py` (the entries) and `toolkit/core/data/recon/scan_files.py` (JSON, CSV, TXT) | HTML/PDF reports (F5) are later features |
| `.github/workflows/ci.yml` | none | checks run locally (`python -m pytest`, ruff, mypy, the pre-commit hook), as plan D19 decided |
| a `train` command in the CLI | none | models are trained in the web trainer (plan D12) |

## 11. Where to make a change

The usual changes, where they go, and what else must change with them (a test fails until
it does). Something both tools need goes in `core/`; something only one interface needs
stays in its folder.

| To | Change | Also | Guarded by |
|---|---|---|---|
| add a web API route | a function on a blueprint in `toolkit/web/controllers/` (`datasets.py`, `training.py`, `bundles.py`, `resume.py`, `workspace.py`; a new blueprint goes in `BLUEPRINTS` in `toolkit/web/controllers/__init__.py`), its request schema in `toolkit/web/controllers/schemas.py`, the work in `toolkit/web/services/`; the page calls it through `getJSON` or `send` (`toolkit/web/ui/static/js/api.js`) | [web_api.md](web_api.md) | `tests/web/test_api_routes.py` (the routes are exactly the documented ones) |
| add a flag | `build_parser()` of the command: `toolkit/cli/controllers/scan.py`, `toolkit/cli/controllers/ids.py`, or `toolkit/web/controllers/command.py` | [reference.md](reference.md) | `tests/test_docs_reference.py` |
| add a command | a module in `toolkit/cli/controllers/` and an entry in `COMMANDS` (`toolkit/cli/controllers/main.py`), and in `MENU_COMMANDS` if the menu offers it; a new interface is picked in `toolkit/start.py` | [reference.md](reference.md), [user_guide.md](user_guide.md) | `tests/cli/test_main_menu.py`, `tests/cli/test_start.py`, `tests/test_docs_user_guide.py` |
| add a `TOOLKIT_*` setting | `toolkit/core/data/settings.py`: its config and reader, and `COMMAND_SETTINGS` for the commands that read it | the settings table of [reference.md](reference.md) | `tests/test_settings_check.py`, `tests/test_docs_reference.py` |
| add a model, balancer or feature selector | a `ModelSpec`, `BalancerSpec` or `SelectorSpec` in `MODELS`, `BALANCERS` or `SELECTORS` (`toolkit/core/services/ids/registry.py`); the page and `/api/v1/models` list it from there | [model_card.md](model_card.md) if it changes what the card says | `tests/ml/test_pipeline.py`, `tests/ml/test_evaluation.py` |
| recognise a new dataset | a `DatasetProfile` in `PROFILES` (`toolkit/core/services/ids/profiles.py`) with the real column names, never guessed; a sample cut from the real file by `tests/fixtures/make_fixtures.py` | [datasets.md](datasets.md), [tests/fixtures/README.md](../tests/fixtures/README.md) | `tests/ml/test_profiles.py`, `tests/ml/test_loader.py` |
| read another port's banner | a `Probe` in `PROBES` (and a reader in `READERS` if none fits) in `toolkit/core/data/recon/probes.py` | - | `tests/recon/test_probes.py` |
| add a scan engine | a class with `name`, `os_matches` (its OS matches after a scan, else `()`), `available()` and `scan(host, ports, protocols, progress, stop)` (the `ScanEngine` port, `toolkit/core/services/ports.py`), named in `ENGINES` (`toolkit/core/services/recon/engines.py`); it fills `PortResult`'s `product`, `version`, `cpe`, `reason`, `scripts` if it knows them, and raises `ScanStopped` once `stop` is set | - | `tests/recon/test_engines.py` |
| recognise another service from its answer | a pattern in `RULES` (`toolkit/core/data/recon/services.py`); the port's usual name, when no list has it, in `PORT_NAMES` | - | `tests/recon/test_services.py` |
| recognise another sign of an OS | a pattern in `BANNER_WORDS` or a port in `PORT_SIGNS` (`toolkit/core/services/recon/os_guess.py`); a new family also needs `agree()` to know which families it fits | - | `tests/recon/test_os_detect.py` |
| match another product's CVEs by version | its NVD `vendor:product` in `CPE_PRODUCTS` (`toolkit/core/data/recon/nvd.py`), under the name `software.product_version` gives, lower case; check it against NVD with a version known to have CVEs | the list in [README.md](../README.md) | `tests/recon/test_nvd.py` |
| recognise another product in a banner | a `Rule` in `RULES` (`toolkit/core/data/recon/software.py`), before the generic pattern; what it returns is looked up online, so never an address or free text | - | `tests/recon/test_software.py` |
| add a scan export format | a writer in `WRITERS` and its name in `EXPORT_FORMATS` (`toolkit/core/data/recon/scan_files.py`; the `--export` choices and the question come from there) | [reference.md](reference.md) | `tests/recon/test_export.py` |
| add an online lookup | an `IntelProvider` in `toolkit/core/data/recon/`, added to `Intel`'s providers (`toolkit/core/services/recon/intel.py`) only after consent or a flag | what leaves the computer: section 9 here and the README's privacy notes | `tests/recon/test_scan_contract.py`, `tests/recon/test_intel_providers.py` |
| add a step to the trainer page | a `<section class="step" data-step="...">` and its step button in `toolkit/web/ui/templates/dynamic_ids.html`, its name in `STEPS` (`toolkit/web/ui/static/js/steps.js`), a module with `setup()` started from `toolkit/web/ui/static/js/main.js`; styles in `toolkit/web/ui/static/css/app.css` (colours as tokens) | [user_guide.md](user_guide.md) | `tests/web/test_steps.py`, `tests/web/test_js_modules.py`, `tests/web/test_css.py`, `tests/web/test_responsive.py` |
| change what a terminal tool prints or asks | `toolkit/cli/ui/` (`menu.py`, `scan.py`, `ids.py`); the web command's console lines are `toolkit/web/ui/console.py` | [user_guide.md](user_guide.md) if a documented screen changes | `tests/cli/`, `tests/recon/` |
| allow a lint or type exception | the comment with its rule and a reason, and its entry in `tests/exceptions_inventory.py` | - | `tests/test_exceptions_inventory.py` |
| keep a function above complexity 10, or a file of `toolkit/` above 400 lines | first see whether splitting it helps; if not, `# noqa: C901` with its reason (an entry in `SUPPRESSIONS`), or the file's exact length and a reason in `LONG_FILES` (`tests/exceptions_inventory.py`) | - | ruff (C901), `tests/test_exceptions_inventory.py`, `tests/test_file_sizes.py` |
| show a result on a page as a panel of label/value tiles | `summaryPanel`, `statTile`, `code`, `quiet` and `chips` in `toolkit/web/ui/static/js/summary.js` (every value set as text); the styles are `.summary-panel`, `.stat-grid`, `.stat-tile` and `.chip` in `toolkit/web/ui/static/css/app.css`; render from the answer's fields, never by parsing its `message` | the answer's fields in [web_api.md](web_api.md) | `tests/web/test_upload_card.py`, `tests/security/test_xss.py` |
| add a class, function or method to `toolkit/` | a docstring: what it is or returns, and what is not obvious (what may leave the computer, a check that must come first, a limit, why); dunder methods and functions defined inside a function need none, their parent's docstring says what matters about them | - | `tests/test_docstrings.py` |
