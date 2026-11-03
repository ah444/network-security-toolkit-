# Project status: what exists today

What the Network Security Toolkit is and does **now**, as built and tested. What is planned is in [ROADMAP.md](ROADMAP.md); why it is built this way is in [DECISIONS.md](DECISIONS.md) and [REVIEW_DECISIONS.md](REVIEW_DECISIONS.md); the tasks in progress are in [V5.4_CHECKLIST.md](../docs/V5.4_CHECKLIST.md).

**As of:** 2026-11-03, branch `main`. Last release: **5.4** (2026-11-03, [gate record](../docs/release/5.4-gate.md)), the plan's last: the scan history (F6), the scan reports (F5) and the home page's dashboard (F10 with F7). Next: the review's condition (7), HV1, then HV3, and last HV2 (CI) with the push ([ROADMAP.md](ROADMAP.md), section 4). Published on GitHub on 2026-11-04, the whole history and the tags: https://github.com/ah444/network-security-toolkit-

Every claim below names how it is known. Where something is only implemented and not verified, it says so.

---

## 1. In one paragraph

A local-first network security toolkit with optional external intelligence services. It has three tools: a **port scanner** (a terminal command and a web page), a **web IDS model trainer** (upload a labelled network-flow dataset, train and compare up to 11 models, predict, export a signed model bundle), and an **IDS command line** that predicts with an exported bundle. It is an IDS **model trainer and predictor**, not a live network IDS: it does not capture traffic or read PCAP files. It runs on Windows with Python 3.14, where it is tested; Linux and macOS are written for but not tested on real machines.

## 2. Numbers

| | Value | How it was measured |
|---|---|---|
| Commits | 225 | `git rev-list --count HEAD`, the release commit included |
| Tags | 5.0-alpha, 5.0-beta, 5.0-rc1, 5.0-rc2, 5.0, 5.0.1, 5.1, 5.2, 5.3, 5.4 | `git tag` |
| Automated tests | 1,981 passed, 2 skipped on purpose | `python -m pytest` (the pre-commit hook), on the release commit |
| Python in `toolkit/` | 17,077 lines | line count of every `*.py` |
| JavaScript, HTML and CSS in `toolkit/` | 6,289 lines | line count |
| Test code | 21,805 lines in 165 Python files | line count of `tests/**/*.py` |
| Routes | 54, of which 46 under `/api/v1` | the app's URL map; `tests/web/test_api_routes.py` |
| ruff, mypy `--strict`, pip-audit | no findings | the hook; `pip-audit` at the end of each step |

The Python lines by part (they add up to the total):

| Part | Lines |
|---|---|
| `toolkit/core/services/ids` (the ML core) | 4,940 |
| `toolkit/core/services/recon` (the scanner's logic) | 1,881 |
| `toolkit/core/services` (`ports.py`, `storage.py`, `__init__.py`) | 273 |
| `toolkit/core/data/recon` (sockets, nmap, NVD, AbuseIPDB, OpenRouter, report files, the scan history) | 2,024 |
| `toolkit/core/data` (settings, logs, keys, publishing, disk space, permissions, the ML files) | 1,125 |
| `toolkit/core/__init__.py` | 4 |
| `toolkit/web` | 4,711 |
| `toolkit/cli` | 2,081 |
| `toolkit/__init__.py`, `__main__.py`, `start.py` | 38 |
| **Total** | **17,077** |

The skipped tests are `tests/recon/test_nmap_run.py::test_nmap_that_ignores_the_request_is_killed` (on Windows, asking nmap to stop already kills it, so the case it tests can't happen there) and `tests/test_protect.py`'s check of the POSIX modes (on Windows the permissions are set with `icacls`, tested beside it).

---

## 3. The tools

### Port scanner (`python -m toolkit scan`, and the `/scan` page)
- Scans a host, a network (up to 256 hosts) or a host name, on chosen TCP and UDP ports (up to 65,536 probes), IPv4 and IPv6
- One connection per port; reads and cleans the banner; finds the product and its version from it; guesses the OS from the TTL and other evidence
- Looks up CVEs and their CVSS scores at NVD (by version for about 20 products, by keyword otherwise; each CVE says which: version match, or name match, unconfirmed); an AI explanation (OpenRouter) and AbuseIPDB for public addresses when their key is in `.env`. Each of the three is on by default and has its own switch; a service that fails in a scan is not asked again in it, and its ports say `NOT CHECKED`
- Every scan is recorded in the log when it starts and ends: the targets, the ports, where it was asked from, what may be sent outside
- Optional nmap engine; the toolkit's own scanner is the default
- Reports as TXT, JSON and CSV, never written over an existing file; a CSV cell that starts like a formula is written as text
- On the web: a job with a live log, progress and Cancel

### Web IDS model trainer (`python -m toolkit web`)
Six steps: **Data** (upload one or more files; the dataset is recognised from its content), **Explore**, **Clean**, **Features**, **Train**, **Use**.
- Up to 11 models (random forest, logistic regression, decision tree, KNN, naive Bayes, linear SVM, gradient boosting, AdaBoost, extra trees, XGBoost, LightGBM) and 8 balancing methods or none
- Training is a job on a queue with progress and Cancel; the page comes back after a reload; a job cut off by a stop is shown once as `INTERRUPTED`
- Every finished training is saved on this computer (its record: scores, class and column names, typical values, not the rows); **Save again** when it couldn't be
- A leaderboard with charts (CV per fold, CV against test, recall per class, fit time) and a confusion matrix
- **What the model relies on** (permutation importance on the test rows, on **Calculate** only, on the training's queue), saved with the run
- Predict one row (typed, or a real example from the test split), with **Why?** (an approximation, not causes), or a whole CSV; export a signed bundle
- **Saved Runs** (`/runs`): every saved training, newest first, with its bundle's state and what its models rely on; show one, or compare two (differences only when both were scored on the same test rows); **Delete** (names every file, asks first), **Storage** (only shows), **Register again** for a kept bundle
- Opened to other computers with `--host ADDRESS` behind an access token; this computer's home page has **Other computers** (the holder's rights, **New token**, their jobs with **Stop**)
- One web server per data folder (an OS lock); records, importance results and bundles published whole or not at all; a new file refused under `TOOLKIT_MIN_FREE_MB` free

### IDS command line (`python -m toolkit ids`)
- Lists the bundles exported on this install and predicts with the chosen one: typed values, an example, `--csv`, `--json` (each result with its `why`); a prediction in the menu shows its top 3 reasons; `--info` (with what the models rely on, when calculated in the trainer), `--list`, `--runs`, `--compare A B`, `--delete RUN_ID`, `--storage`
- A bundle is checked (format, signature, library versions) before it is opened

### Launcher (`python -m toolkit`)
A flat menu (scanner, IDS, web trainer); every tool can also be started directly with its flags. Settings come from `TOOLKIT_*` environment variables and `.env`; a bad setting stops the start with a message and exit code 2.

---

## 4. Architecture

**Style:** a modular monolith (one installable package, one process per tool) whose core follows hexagonal rules, with plugin registries for models, balancers, selectors and dataset profiles, pipe-and-filter for the ML, and background jobs for long work.

```
  toolkit/cli                         toolkit/web
  (menu, scan, ids, web)              (5 pages, the /api/v1 routes, jobs, workspaces)
          \                                 /
           \  both use the core; neither   /
            \ imports the other           /
             v                           v
        toolkit/core/services      ids/ (the ML), recon/ (the scanner), ports.py
             |
             v
        toolkit/core/data          files, settings, logs, keys, sockets, nmap, NVD, AbuseIPDB, OpenRouter
             |
             v
        instance/                  bundles/  runs/  scans/  logs/  signing.key
```

- Inside each of `core`, `cli` and `web` the layers are `ui → controllers → services → data`, and an import only goes down. **Enforced** by `tests/test_architecture.py`.
- The network libraries (`socket`, `requests`, `urllib`, `http`, `ssl`, `subprocess`, `webbrowser`) are imported only in the data layers. **Enforced** by `tests/test_core_rules.py`. This proves where the code that can reach the network lives; it does not prove which calls are made.
- **Four ports**, each with more than one implementation: `ModelStore` (signed bundles), `IntelProvider` (NVD, AbuseIPDB, OpenRouter, fakes in tests), `ProgressSink` (terminal, web job, tests), `ScanEngine` (the toolkit's scanner or nmap). Everything else is plain functions and registries.
- The ML core takes data in and returns results (`train(frame, settings) → TrainingRun`); reading and writing files happens at the edges.

### The ML core, `toolkit/core/services/ids/`
| Module | Does |
|---|---|
| `datasets.py`, `profiles.py` | Read CSV and TXT (encoding, separator, chunks, dtypes, SHA-256); recognise CIC-IDS2017, NSL-KDD or a generic dataset from the content |
| `split.py`, `split_strategies.py`, `split_time.py` | Remove duplicates, then split: official, temporal, temporal per class, group, stratified, with reason codes for every fallback |
| `leakage.py` | The leakage guard |
| `sampling.py`, `balancing.py` | Stratified samples with every class; the balancing recommendation from the training labels only |
| `registry.py`, `pipeline.py` | The registries; one pipeline per model fitted on the training split only; `FittedModel`; the typical values of each feature |
| `evaluation.py` | Cross-validation on the training split, selection by CV macro-F1, the test split scored, the leaderboard |
| `runs.py`, `run_json.py` | `train()` and the `TrainingRun` record |
| `bundle.py`, `exports.py` | Signed bundles: write, list, open |
| `comparison.py` | Saved runs: summaries and comparisons |
| `importance.py` | Permutation importance (F3b), run as a job from the trainer (F3d, F3e) |
| `explanation.py` | Why one prediction came out as it did (F3c), in the web's and the CLI's answers (F3d–F3f) |
| `prediction.py`, `batch.py`, `preview.py`, `legacy_names.py` | Predictions; the trainer's previews; older module names, so older bundles still open |

---

## 5. How a training works

```
load + profile → dedupe → split → leakage guard → CV on the training split → fit
  → select by CV macro-F1 → score the test split → TrainingRun saved → (on Export) signed bundle
```

- **Dedupe before the split:** a row in both train and test inflates the score (23,416 duplicate rows in the CIC DDoS file)
- **The split comes from the profile:** NSL-KDD uses its official files, **as published by default**; the 610 test rows that also occur in the training file are counted and reported, and removed only in strict mode (`SplitConfig.official_dedupe`, "official split with the train/test overlap removed"), which the record notes. CIC-IDS2017 as available here has no time column, so it is split stratified, with a warning that scores may be optimistic
- **The leakage guard** detects and blocks a set of **known** leakage patterns: a single feature that predicts the label almost perfectly and behaves like a relabelling, label-like and ID-like names (whole words), ID-like integer or text columns. It cannot prove that no leakage is left; what a feature means and when it was collected is the user's to check
- **Every step learns from the training split only**, inside one pipeline saved with the model
- **The test split is not used to train or to choose anything.** It is scored for the leaderboard after the model is chosen. The permutation importance (F3b) reuses it afterwards, as a diagnosis
- **The record** holds the data's hash and profile, the split and its reasons, the leakage decisions, the features, the settings, the CV and test metrics, the library versions and, since F3c, each feature's typical value. Since HR5 it is written **when every successful training ends**; Export writes the bundle only

---

## 6. Security and privacy

| Risk | What is in place | How it is known |
|---|---|---|
| A model file running code | Bundles are signed with HMAC-SHA256 with this install's key (`instance/signing.key`). The format, the signature and the library versions are checked on the bytes in memory **before** any unpickle; the same bytes are then loaded | `tests/ml/test_bundle.py` changes one byte in several places and asserts the unpickle never runs |
| What the signature does not do | It rejects a bundle not signed with our key. It does not make unpickling safe, keeps nothing secret, and does not protect against someone who has the key or can write `instance/` | By design (plan D11) |
| The key on Windows | Created once with an exclusive open and never replaced. At every start, `instance/` and `.env` get the current account, SYSTEM and Administrators only, nothing inherited (`icacls`, HS5); the key inherits it | `tests/test_protect.py`; checked with `icacls` on this PC (HS5) |
| XSS | Every value is put in the page as text (no `innerHTML`); a strict CSP | Tests on the templates and the JS |
| Path traversal | Uploads get random names and are removed after reading; a run id is checked against its form before a path is made | Tests |
| Cross-site requests | A CSRF token per session, the `Origin` check, `SameSite=Strict` and `HttpOnly` cookies | Tests |
| DNS rebinding | The `Host` check: the server answers only to its own address or `localhost` | Tests |
| Opening the app to the network | `--host` with a warning; a new 80-bit token at every start, shown only in the terminal and never logged; five wrong tries per address per minute; `X-Forwarded-For` never trusted; the loopback browser needs no token | Tests |
| **What remote access means** | Whoever holds the token can do everything the owner can: train, scan from this computer (its internal network included), read and compare the saved runs. Said in the terminal and on the unlock page. **New token** on this computer locks every other computer out at once and cancels their waiting jobs; each job keeps where it came from. The traffic is plain HTTP (HTTPS, HS2, was dropped by the owner): a trusted network only | Tests; the gate 5.3 check from this PC's host-only address |
| What leaves the computer | The scan sends its probes to the targets the user chose. A host name is resolved by the system's DNS **before** the authorization question (documented in `resolve_targets`); nothing reaches the target before validation, the limits, the authorization and the opt-ins. NVD receives the product and version of every service found; OpenRouter the product and version only; AbuseIPDB a public address only, never a private one. Each is on by default (RD5; the AI and AbuseIPDB only with their key in `.env`), has its own switch, and the scan says first what it may send and at the end how many requests went to each | Tests of the scanner and the intel gateways; the gate 5.3 check (all three off: 0 requests each) |
| Error messages | No paths or exception details; details go to the log, where the API keys are hidden | Tests |
| Dependencies | Pinned in `requirements.lock`; `pip-audit` at every step and tag | `pip-audit` |

---

## 7. Code quality: what enforces it

| Rule | Enforced by |
|---|---|
| The layers and the network rule | `test_architecture.py`, `test_core_rules.py` |
| Complexity at most 10 per function | ruff `C901`; the two exceptions are listed with their reasons in `tests/exceptions_inventory.py`, which fails on an unlisted or stale suppression |
| At most 400 lines per Python or JavaScript file in `toolkit/` (not CSS) | `test_file_sizes.py`. A signal for review, not a target: files are split by responsibility |
| Types | `mypy --strict` on the package |
| Docstrings on every function | `test_docstrings.py` |
| Portability | Tests that refuse `shell=True`, text files opened without an encoding, Windows-only paths; ruff targets Python 3.11 |
| The pages | JS modules without globals, tested in Node; CSS tokens for both themes; contrast at least 4.5:1 for text and 3:1 for marks, tested |
| Where it runs | The local pre-commit hook (ruff, mypy, pytest). It can be skipped or not installed; there is no central CI yet (HV2) |

### What is verified, and how
| Level | What it covers here |
|---|---|
| Unit | Most of the 1,981 tests: one module with fakes |
| Integration | Flask test-client tests through routes, services and files; the CLI through its entry point |
| End-to-end | The browser checks of each gate (8 sizes, both themes, keyboard only) on Chromium; the fresh installs of each gate on Windows |
| Manual | The gate records in `docs/release/`, dated |

Every test is marked by level since HC4 (`tests/levels.py`; docs/architecture.md §12). Not verified: Linux and macOS on real machines, browsers other than Chromium, a screen reader, a user who is not the developer.

---

## 8. Results

From `docs/report/FINDINGS.md` (three runs per case on the v5 core; the 5.1 and 5.2 refactors reproduced all 1,246 values exactly).

| Case | v4 | v5 | What it means |
|---|---|---|---|
| CIC-IDS2017 DDoS | trees 99.9% | trees 99.9%; logistic regression from 95.1% to 98.1% macro-F1 | The data is easy: the score stays high after 23,416 duplicates are removed. It shows nothing about other data |
| NSL-KDD, random split | macro-F1 76.6% | accuracy 99.8%, macro-F1 68.8% | Without the leaky `level` column macro-F1 drops. Six rare classes (one or two test rows each) have a recall of 0, while `normal` and `neptune` are 86% of the test rows, so the accuracy hides them. The confusion matrices are in the JSON |
| NSL-KDD, official test file | – | accuracy 72.1%, macro-F1 27.0% | 17 attack types (16.6% of the test rows) never occur in the training file. That limits this split and this closed-set training set-up; it is not a limit for every model or method |
| v4's "SMOTE + 10 features" | ~81% | macro-F1 44.3% | The 81% was measured on mostly synthetic rows and used `level` |

**Generalization is not shown.** No test on data from another source or another time of collection has been made.

---

## 9. v5.3 and v5.4

### v5.3
- **F2, the full leaderboard: done** (F2a–F2g): Saved Runs, the comparison, the charts, `ids --runs` and `--compare`
- **F3, explainability: done** (F3a–F3g): what a model relies on in the trainer and on Saved Runs, **Why?** under a prediction, the CLI's `--info`, reasons and `--json`
- **The outside review's hardening (section H): done but HV** (HC1–HC4, HS1, HS3–HS5, HR1–HR5; HS2 dropped by the owner). The review's conditions (1)–(6) are met; **(7), real use shown (HV1–HV3), is open**, moved after 5.4 by the owner
- **Gate 5.3** (2026-11-03): [record](../docs/release/5.3-gate.md)

### v5.4
- **F6, the scan history: done** (F6a–F6e): every finished scan saved, Saved Scans, the comparison, `scan --history`, `--show`, `--compare`, `--delete`, counted by Storage
- **F5, scan reports: done** (F5a–F5d): a saved scan's HTML report, on the page and as one file, `scan --report`; printed to PDF by the browser
- **F10 with F7, the dashboard: done** (F10a–F10c): the scans and their risks beside the runs and the session, never linked
- **Gate 5.4** (2026-11-03): [record](../docs/release/5.4-gate.md)

---

## 10. Known limits
- Linux and macOS: the OS-specific code is unit-tested with a patched platform only; not called supported until HV3
- No user who is not the developer has used it (HV1); no CI (HV2)
- A bundle opens only on the install that made it; if `instance/signing.key` is lost, its bundles can't be opened. Back up `instance/`
- `--host` serves plain HTTP: use it only on a trusted network
- Workspaces and running jobs are lost when the server stops; the next start says so once (`INTERRUPTED`), but they are not resumed
- Files are published whole after a crash of the process; after a power cut it is best effort
- What a model relies on, and **Why?**, are a diagnosis and an approximation, not causes
- Large models (a random forest on all of CIC-IDS2017 is about 2 GB) take minutes to open and need several times their size in memory; a training on all of CIC-IDS2017 needs about 6 GB
- `max_data_mb` (4,096 MB) is the memory of the loaded datasets, not disk space, and doesn't cover the training's own memory
- CVEs found by keyword may be wrong or missing; each CVE says whether its version was confirmed
