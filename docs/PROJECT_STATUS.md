# Project status: what exists today

What the Network Security Toolkit is and does **now**, as built and tested. What is planned is in [ROADMAP.md](ROADMAP.md); why it is built this way is in [DECISIONS.md](DECISIONS.md) and [REVIEW_DECISIONS.md](REVIEW_DECISIONS.md); the tasks in progress are in [V5.3_CHECKLIST.md](V5.3_CHECKLIST.md).

**As of:** 2026-11-02, branch `main`. Last release tag: **5.2** (2026-10-30). Work in progress: **v5.3**. Nothing has been pushed to a remote.

Every claim below names how it is known. Where something is only implemented and not verified, it says so.

---

## 1. In one paragraph

A local-first network security toolkit with optional external intelligence services. It has three tools: a **port scanner** (a terminal command and a web page), a **web IDS model trainer** (upload a labelled network-flow dataset, train and compare up to 11 models, predict, export a signed model bundle), and an **IDS command line** that predicts with an exported bundle. It is an IDS **model trainer and predictor**, not a live network IDS: it does not capture traffic or read PCAP files. It runs on Windows with Python 3.14, where it is tested; Linux and macOS are written for but not tested on real machines.

## 2. Numbers

| | Value | How it was measured |
|---|---|---|
| Commits | 170 | `git rev-list --count HEAD` |
| Tags | 5.0-alpha, 5.0-beta, 5.0-rc1, 5.0-rc2, 5.0, 5.0.1, 5.1, 5.2 | `git tag` |
| Automated tests | 1,611 passed, 1 skipped on purpose | `python -m pytest` (the pre-commit hook), last run on this commit's parent |
| Python in `toolkit/` | 13,354 lines | line count of every `*.py` |
| JavaScript, HTML and CSS in `toolkit/` | 4,851 lines | line count |
| Test code | 16,385 lines in 131 Python files | line count of `tests/**/*.py` |
| Routes | 35, of which 30 under `/api/v1` | route decorators; `tests/web/test_api_routes.py` |
| ruff, mypy `--strict`, pip-audit | no findings | the hook; `pip-audit` at the end of each step |

The Python lines by part (they add up to the total):

| Part | Lines |
|---|---|
| `toolkit/core/services/ids` (the ML core) | 4,451 |
| `toolkit/core/services/recon` (the scanner's logic) | 990 |
| `toolkit/core/services` (`ports.py`, `__init__.py`) | 121 |
| `toolkit/core/data/recon` (sockets, nmap, NVD, AbuseIPDB, OpenRouter, report files) | 1,868 |
| `toolkit/core/data` (settings, logs, keys, files, the ML files) | 745 |
| `toolkit/core/__init__.py` | 4 |
| `toolkit/web` | 3,651 |
| `toolkit/cli` | 1,486 |
| `toolkit/__init__.py`, `__main__.py`, `start.py` | 38 |
| **Total** | **13,354** |

The skipped test is `tests/recon/test_nmap_run.py::test_nmap_that_ignores_the_request_is_killed`: on Windows, asking nmap to stop already kills it, so the case it tests can't happen there.

---

## 3. The tools

### Port scanner (`python -m toolkit scan`, and the `/scan` page)
- Scans a host, a network (up to 256 hosts) or a host name, on chosen TCP and UDP ports (up to 65,536 probes), IPv4 and IPv6
- One connection per port; reads and cleans the banner; finds the product and its version from it; guesses the OS from the TTL and other evidence
- Looks up CVEs and their CVSS scores at NVD (by version for about 20 products, by keyword otherwise; the report says which); an optional AI explanation (OpenRouter) after the user's consent; optional AbuseIPDB for public addresses when a key is set
- Optional nmap engine; the toolkit's own scanner is the default
- Reports as TXT, JSON and CSV, never written over an existing file; a CSV cell that starts like a formula is written as text
- On the web: a job with a live log, progress and Cancel

### Web IDS model trainer (`python -m toolkit web`)
Six steps: **Data** (upload one or more files; the dataset is recognised from its content), **Explore**, **Clean**, **Features**, **Train**, **Use**.
- Up to 11 models (random forest, logistic regression, decision tree, KNN, naive Bayes, linear SVM, gradient boosting, AdaBoost, extra trees, XGBoost, LightGBM) and 8 balancing methods or none
- Training is a job on a queue with progress and Cancel; the page comes back after a reload
- A leaderboard with charts (CV per fold, CV against test, recall per class, fit time) and a confusion matrix
- Predict one row (typed, or a real example from the test split) or a whole CSV; export a signed bundle
- **Saved Runs** (`/runs`): the exported runs, newest first; show one, or compare two (differences only when both were scored on the same test rows)
- Opened to other computers with `--host ADDRESS` behind an access token

### IDS command line (`python -m toolkit ids`)
- Lists the bundles exported on this install and predicts with the chosen one: typed values, an example, `--csv`, `--json`; `--info`, `--list`, `--runs`, `--compare A B`
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
| `importance.py` | Permutation importance (F3b; not reachable from the pages or the CLI yet) |
| `explanation.py` | Why one prediction came out as it did (F3c; not reachable yet) |
| `prediction.py`, `batch.py`, `preview.py`, `legacy_names.py` | Predictions; the trainer's previews; older module names, so older bundles still open |

---

## 5. How a training works

```
load + profile → dedupe → split → leakage guard → CV on the training split → fit
  → select by CV macro-F1 → score the test split → TrainingRun → (on Export) signed bundle
```

- **Dedupe before the split:** a row in both train and test inflates the score (23,416 duplicate rows in the CIC DDoS file)
- **The split comes from the profile:** NSL-KDD uses its official files, **as published by default**; the 610 test rows that also occur in the training file are counted and reported, and removed only in strict mode (`SplitConfig.official_dedupe`, "official split with the train/test overlap removed"), which the record notes. CIC-IDS2017 as available here has no time column, so it is split stratified, with a warning that scores may be optimistic
- **The leakage guard** detects and blocks a set of **known** leakage patterns: a single feature that predicts the label almost perfectly and behaves like a relabelling, label-like and ID-like names (whole words), ID-like integer or text columns. It cannot prove that no leakage is left; what a feature means and when it was collected is the user's to check
- **Every step learns from the training split only**, inside one pipeline saved with the model
- **The test split is not used to train or to choose anything.** It is scored for the leaderboard after the model is chosen. The permutation importance (F3b) reuses it afterwards, as a diagnosis
- **The record** holds the data's hash and profile, the split and its reasons, the leakage decisions, the features, the settings, the CV and test metrics, the library versions and, since F3c, each feature's typical value. Today it is written **on Export only**

---

## 6. Security and privacy

| Risk | What is in place | How it is known |
|---|---|---|
| A model file running code | Bundles are signed with HMAC-SHA256 with this install's key (`instance/signing.key`). The format, the signature and the library versions are checked on the bytes in memory **before** any unpickle; the same bytes are then loaded | `tests/ml/test_bundle.py` changes one byte in several places and asserts the unpickle never runs |
| What the signature does not do | It rejects a bundle not signed with our key. It does not make unpickling safe, keeps nothing secret, and does not protect against someone who has the key or can write `instance/` | By design (plan D11) |
| The key on Windows | Created once with an exclusive open and never replaced; `chmod 0o600`, which sets no real permission on Windows. `instance/` inherits its folder's permissions, not checked yet | HS5 is open |
| XSS | Every value is put in the page as text (no `innerHTML`); a strict CSP | Tests on the templates and the JS |
| Path traversal | Uploads get random names and are removed after reading; a run id is checked against its form before a path is made | Tests |
| Cross-site requests | A CSRF token per session, the `Origin` check, `SameSite=Strict` and `HttpOnly` cookies | Tests |
| DNS rebinding | The `Host` check: the server answers only to its own address or `localhost` | Tests |
| Opening the app to the network | `--host` with a warning; a new 80-bit token at every start, shown only in the terminal and never logged; five wrong tries per address per minute; `X-Forwarded-For` never trusted; the loopback browser needs no token | Tests |
| **What remote access means** | Whoever holds the token can do everything the owner can: train, scan from this computer (its internal network included), read and compare the saved runs. The traffic is plain HTTP. A session's workspace is not a security boundary: the saved runs, bundles and scan reports are shared | Not yet stated on the pages (HS1, HS2) |
| What leaves the computer | The scan sends its probes to the targets the user chose. A host name is resolved by the system's DNS **before** the authorization question (documented in `resolve_targets`); nothing reaches the target before validation, the limits, the authorization and the opt-ins. NVD receives the product and version of every service found, on every scan, with no switch to turn it off. OpenRouter receives the product and version only after consent. AbuseIPDB receives a public address only when a key is set; a private address never | Tests of the scanner and the intel gateways; the NVD switch is HS4 |
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
| Unit | Most of the 1,611 tests: one module with fakes |
| Integration | Flask test-client tests through routes, services and files; the CLI through its entry point |
| End-to-end | The browser checks of each gate (8 sizes, both themes, keyboard only) on Chromium; the fresh installs of each gate on Windows |
| Manual | The gate records in `docs/release/`, dated |

The tests are not yet marked by level (HC4). Not verified: Linux and macOS on real machines, browsers other than Chromium, a screen reader, a user who is not the developer.

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

## 9. v5.3 so far
- **F2, the full leaderboard: done** (F2a–F2g): Saved Runs, the comparison, the charts, `ids --runs` and `--compare`
- **F3, explainability: 3 of 7 steps done, in the core only** (F3a decisions, F3b permutation importance, F3c the explanation of one prediction). Nothing new is visible to the user yet
- **The outside review (2026-11-02):** hardening tasks (section H of the checklist) and four design decisions (RD1–RD4 in [REVIEW_DECISIONS.md](REVIEW_DECISIONS.md)), written down, **not implemented**

---

## 10. Known limits
- Linux and macOS: the OS-specific code is unit-tested with a patched platform only
- A bundle opens only on the install that made it; if `instance/signing.key` is lost, its bundles can't be opened. Back up `instance/`
- `--host` serves plain HTTP: use it only on a trusted network
- NVD is asked on every scan, with no switch
- Workspaces and running jobs are lost when the server stops, and nothing tells the user afterwards
- A crash while a bundle is written leaves an incomplete file under its final name (the signature check refuses it, but its name blocks a new export of that run)
- Large models (a random forest on all of CIC-IDS2017 is about 2 GB) take minutes to open and need several times their size in memory; a training on all of CIC-IDS2017 needs about 6 GB
- `max_data_mb` (4,096 MB) is the memory of the loaded datasets, not disk space, and doesn't cover the training's own memory
- CVEs found by keyword may be wrong or missing; the report says per product how they were found, not yet per CVE
