# Changelog

All notable changes to this project. Versions follow the roadmap in
[docs/IMPROVEMENT_PLAN.md](docs/IMPROVEMENT_PLAN.md); day-to-day progress is in
[docs/V5.0_CHECKLIST.md](docs/V5.0_CHECKLIST.md).

## [5.0] — 2026-10-21

Release of v5.0 "correct & safe": the checks of the plan's §8 pass, the ML report
measures v4 against the fixed ML core, and a fresh install was walked through end to end.
No code behaviour changed since 5.0-rc2. Project version: `5.0.0`.

**Test result:** 419 automated tests passed (`python -m pytest`), ruff clean. Fresh clone
of 5.0-rc2 with a new venv from `requirements.lock` (`pip check` clean), a model exported
through the web routes and a walk-through of every menu path; web trainer at 375 px in light
and dark mode with 0 texts below WCAG AA ([record](docs/release/5.0-final-ui-and-install.md)).
Every §8 check is mapped to its tests ([audit](docs/release/5.0-section8-audit.md)).
Windows 10 Enterprise LTSC 2021, Python 3.14.3, scikit-learn 1.8.0.

### Added
- **ML report** (`scripts/ml_report.py` → [`docs/report/`](docs/report/FINDINGS.md)): the
  baseline datasets and scenarios re-run on the v5 core, 3 runs each, with macro-F1,
  balanced accuracy, per-class recall and CV mean ± std, plus NSL-KDD on its official
  test file. Main results:
  - CIC-IDS2017 DDoS: trees stay at 99.9 % after removing 23,416 duplicate rows; logistic
    regression rises from 95.1 % to 98.1 % macro-F1 (S1).
  - NSL-KDD, random split of `KDDTrain+`: random forest accuracy is still 99.8 %, but its
    macro-F1 is 68.8 % without the label-derived `level` (v4: 76.6 %).
  - NSL-KDD, official `KDDTest+`: **72.1 % accuracy, 27.0 % macro-F1**. 17 attack types
    (16.6 % of the test rows) never appear in the training file.
  - v4's ~81 % for "SMOTE + 10 recommended features" was measured on mostly synthetic test
    rows with `level` as a feature; on real test rows v5 gets 44.3 % macro-F1.
  `--case` and `--jobs` run single cases, with their runs side by side on one core each.
- `tests/test_docs_claims.py` checks that every test named in the §8 audit exists;
  `tests/test_portability.py::test_python_policy` checks `requires-python >= 3.11` and
  ruff's `py311` target.

### Changed
- README: Linux and macOS are not tested on real machines (OS-specific logic is
  unit-tested only); links to the ML report.

## [5.0-rc2] — 2026-10-20

Fourth checkpoint of v5.0: **the CLI predicts with the models you train and export in the
web trainer** (no model is shipped any more), dataset licences are recorded, and the
**port scanner is fixed for correctness and privacy** (steps 0.5, 0.5b, 0.6). Project
version: `5.0.0rc2`.

**Test result:** 416 automated tests passed (`python -m pytest`), ruff clean, `pip-audit`
clean (installed environment and `requirements.lock`). Real-data check: CLI and web gave
the same prediction for every compared row of the official NSL-KDD test file and a 5 %
sample of CIC-IDS2017, for all 7 models of two real exports; real scans of `127.0.0.1`
and `::1` with local services and real NVD lookups; launcher walk-through of every menu
path ([record](docs/release/5.0-rc2-cli-and-scanner.md)).
Windows 10 Enterprise LTSC 2021, Python 3.14.3, scikit-learn 1.8.0.

### Added
- **CLI on exported models** (D5, D12 revised, B14): the CLI lists the bundles exported on
  this install (`instance/bundles`, newest first, reading only each manifest header) and
  you pick one; `--bundle`, `--list`, `--info`, `--csv` (labels every row; the result file
  is never overwritten) and `--json`. Features, classes, the best model and the model card
  come from the bundle and its TrainingRun; real test-split examples are embedded in each
  bundle. The status line says *Normal traffic* for `BENIGN`/`normal`, and examples of
  classes the model never learned are marked.
- **Scanner:** authorization question before every scan (`--i-am-authorized` for
  scripts) and limits of 256 hosts / 65,536 host×port checks (`TOOLKIT_SCAN_MAX_HOSTS`,
  `TOOLKIT_SCAN_MAX_PROBES`) (S16); IPv6 targets (B13); CVEs with CVSS scores from the NVD
  CVE API 2.0 (B10); keys from environment variables or `.env`, with `api_keys.txt` as a
  fallback, plus an optional `NVD_API_KEY` (S15).
- **Dataset licences** in `docs/datasets.md`: terms, required citations and
  redistribution rules for CIC-IDS2017, NSL-KDD and UNSW-NB15, checked on the official
  pages; the committed fixtures carry their citations (0.5b).

### Changed
- **Scanner privacy** (S12, S13): the AI explanation runs only after you agree, receives
  only the detected product and version (e.g. `OpenSSH 8.9`), never the raw banner, and
  its reply is plain text marked "may be wrong" that can't add a CVE or change a severity
  (severity comes from CVSS only).
- **Scanner correctness:** one connection per port, always closed, with the banner read
  over it (R1, R2); OS guess from TTL ranges (B6); ping flags and timeouts per OS (B7);
  silent UDP ports are `open|filtered` (B8; the v4 DNS probe was invalid and never ran);
  TLS banners from self-signed services (B9); port and target validation (B11); each
  lookup has a 15 s timeout and failures are listed (R3).
- **Scanner output** (B12, S14): reports are UTF-8 files in `instance/scans/` that never
  overwrite each other; terminal escape codes are stripped from banners and replies; CSV
  cells that start like a formula are written as text.

### Removed
- The v4 CLI model files (`models/*.pkl`, `config.json`, `manifest.json`): the model used
  the label-derived `Attempted Category` feature (L1, B21). The planned pinned default
  model was dropped: a full-file Random Forest export is about 2 GB, far over GitHub's
  100 MB file limit.
- The MITRE CVE page scraper (retired service).

### Known limitations
See **Known limitations** in the [README](README.md). New in this release:
- A fresh install has no CLI model until one is trained and exported in the web trainer.
  Large exports (about 2 GB for Random Forest on the full CIC file) take minutes to load
  and need several times their size in memory.
- CVE lists come from an NVD keyword search on the software name and version, so they can
  include CVEs that don't affect the exact build, or miss some. Without an NVD key, NVD
  allows about one request every 6 seconds.

## [5.0-rc1] — 2026-10-20

Third checkpoint of v5.0: **the web trainer now runs on the leakage-free ML core**, and
trained models can be exported and loaded again as signed bundles (steps 0.4e–0.4g).
Web trainer results are no longer the optimistic v4 ones. The CLI still uses the v4 model
until 5.0-rc2 (step 0.5). Project version: `5.0.0rc1`.

**Test result:** 353 automated tests passed (`python -m pytest`), ruff clean, `pip-audit`
clean (installed environment and `requirements.lock`). End-user check in the browser at
phone width on real CIC-IDS2017 and NSL-KDD samples, from upload through export, a cleared
workspace, loading the bundle and predicting, plus refused fake and tampered bundles
([record](docs/release/5.0-rc1-web-trainer.md)).
Windows 10 Enterprise LTSC 2021, Python 3.14.3, scikit-learn 1.8.0.

### Added
- **TrainingRun** (`runs.py`, `run_files.py`): `train(...)` returns one record of how a
  model was made (dataset SHA-256, profile, split and reason codes, leakage decisions,
  features, balancing, model parameters, the evaluation, library versions);
  `rerun(...)` repeats it on the same data and `compare_metrics(...)` checks the result
  (exact with one worker). Records go to `instance/runs/` and are never overwritten (D21).
- **Signed bundles** (`bundle.py`, `bundle_files.py`): the fitted models of a run plus a
  manifest embedding its TrainingRun, signed with HMAC-SHA256. Everything is checked
  before anything is deserialized: format, signature (this install's key in
  `instance/signing.key`) or a pinned SHA-256, and library versions (estimator libraries:
  minor difference refused, patch warned; numpy/pandas: major refused) (D11, S1, S17, B18).
- **Train-only previews** (`preview.py`): feature scores after the leakage guard and the
  balancing preview, both on the training split.

### Changed
- **Web trainer on the v5 core** (L2, L3, L8, B2): the v5 loader reads uploads (dataset
  recognised, label preselected, an NSL-KDD train file plus a `*test*` file uses the
  official split); training calls `train(...)`; class analysis, feature recommendation and
  "Apply SMOTE" (now *Preview on training split*) look at the training split only and
  change nothing; results are a leaderboard of CV macro-F1 (mean ± std) with the test
  split scored once, in place of the v4 validation split; prediction goes through the
  saved pipeline. The v4 preprocessing code is deleted (`app.py` 1,857 → 866 lines).
- **Export / Load model are back** (S1, S4): export saves a signed `.bundle` and its run
  record under `instance/` and offers a download; load accepts only bundles signed by this
  install. `/predict_loaded` lost its stray slash and the dead `loaded_model_config` JS is
  gone (B3, B4).
- Checkbox lists replace the Ctrl+Click multi-selects, so the trainer works on touch
  screens (U5); split warnings are shown in plain language and long words wrap at phone
  width; uploaded files keep their own names in messages and records.

### Removed
- `dask` and the packages only it needed (`pyarrow`, `fsspec`, `partd`, `locket`,
  `toolz`, `PyYAML`): nothing uses them since the v5 loader replaced the v4 upload code. A
  test now checks that every runtime requirement is imported somewhere.

### Known limitations
See **Known limitations** in the [README](README.md). New in this release:
- Bundles load only on the install that made them; back up `instance/`. Cross-device
  import comes in v5.1.
- Training runs inside the request, with no progress bar or cancel button yet (v5.1).
- `scripts/baseline_v4.py` replays the v4 click sequences; it only works on the v4 app
  (any commit before `1cc1068`, step 0.4g), as recorded in `docs/baseline/`.

## [5.0-beta] — 2026-10-20

Second checkpoint of v5.0: the **leakage-free ML core** (steps 0.4a–0.4d) is finished
and tested. It is a library (`toolkit/ids_core`) for now; the web trainer and CLI switch
to it in 5.0-rc1 (step 0.4g), so **their results are still the old, optimistic ones**.
Project version: `5.0.0b1`.

**Test result:** 295 automated tests passed (`python -m pytest`), ruff clean, `pip-audit`
clean (installed environment and `requirements.lock`). The old web flow was replayed with
the recorded baseline click sequences on the full files, with **identical** results, and
used in the browser from upload to prediction
([record](docs/release/5.0-beta-old-web-flow.md)).
Windows 10 Enterprise LTSC 2021, Python 3.14.3, scikit-learn 1.8.0.

### Added
- **Loader** (`datasets.py`): sniffed encoding and separator, chunked C-engine read,
  stripped headers (B17), identical `.1` columns dropped, exact dtype downcast, SHA-256,
  `LoadReport`.
- **Content-based dataset profiles** (`profiles.py`): CIC-IDS2017 (original, subset,
  improved names), NSL-KDD (with or without header / `level`), generic; each gives the
  label, blocked columns and split strategy, with reason-code warnings (F4, L10).
- **Leakage guard** (`leakage.py`): single-feature rule (label copies blocked, strong
  features only warned), whole-word name patterns, ID-like columns, recorded overrides (L6).
- **Dedupe and split** (`split.py`): exact copies removed before the split (L4); official,
  temporal, temporal-per-class, group and stratified strategies with time-column
  validation, fallbacks and enumerated reason codes; class coverage (L5). NSL-KDD's
  official files are kept as published by default; strict mode removes the 610 test rows
  that also occur in train.
- **Training pipeline** (`pipeline.py`, `registry.py`): one imblearn pipeline per model,
  fitted on training rows only (L2, L3); one-hot or ordinal encoding by model with safe
  unseen categories (L7); registries for the 11 models, 9 balancing choices and feature
  selectors; every seed fixed, nested SMOTE included (B20); failed models left out (B1);
  SVM limit on training rows (B5); `TOOLKIT_MAX_WORKERS` caps CPU use (D8).
- **Evaluation** (`evaluation.py`, `balancing.py`): CV on the training split, model chosen
  by CV macro-F1 before the test split is used, test scored once (L8); macro-F1, balanced
  accuracy, per-class precision/recall/F1, confusion matrices (L9); the v4 balancing
  rules fed with training labels only; a basic leaderboard (F2).

### Changed
- Linear and distance models (logistic regression, KNN, linear SVM, Naive Bayes) use
  signed log + standard scaling instead of RobustScaler: on CIC, logistic regression went
  from 0.144 to 0.691 balanced accuracy and converges.
- Prepared features are float32, so 2 million CIC training rows fit in an 8 GB laptop
  (peak 8.35 → 5.98 GB).

### Decided
- **Dataset for the default CLI model (0.5): the original CIC-IDS2017 file**, with a
  stratified split and the "scores may be optimistic" warning; details and accepted
  caveats in [docs/datasets.md](docs/datasets.md).
- Real-data results (0.4d, recommended balancing): CIC XGBoost test macro-F1 0.886
  (CV 0.854 ± 0.043); NSL-KDD XGBoost 0.262 on the official KDDTest+, where 17 attack
  types exist only in the test file.

### Known limitations
See **Known limitations** in the [README](README.md). New in this release:
- The web trainer still uses the old training code (until rc1); the touch-unfriendly
  Ctrl+Click lists and the sideways-scrolling preview table at phone width are still there.
- Training on the full CIC file is slow on a laptop (random forest about 9 min, logistic
  regression with class weights about 22 min) and needs about 6 GB of memory.

## [5.0-alpha] — 2026-10-20

First checkpoint of v5.0 ("correct & safe"). Its focus is **honesty and safety**:
it doesn't fix the ML leakage itself (v5.0 step 0.4), but it states it, and closes
the critical security holes. Project version: `5.0.0a1`.

**Test result:** 122 automated tests passed (`python -m pytest`), ruff clean,
`pip-audit` clean (installed environment, `requirements.txt`, `requirements.lock`),
plus a scripted walk-through of every launcher menu path
([transcript](docs/release/5.0-alpha-menu-walkthrough.md)).
Windows 10 Enterprise LTSC 2021, Python 3.14.3, scikit-learn 1.8.0, Flask 3.1.3.

### Added
- Test suite (pytest) with small fixtures cut from real CIC-IDS2017 / NSL-KDD files, ruff
  linting, pip-audit, a local pre-commit hook, and portability guards (encoding,
  `shell=True`, Windows-only paths) so the code stays runnable on Linux and macOS.
- Measured baseline of the original web trainer and CLI model in `docs/baseline/`
  (repeated runs gave identical results), with findings in `docs/baseline/FINDINGS.md`.
- Dataset fingerprints and profiles in `docs/datasets.md`.
- Web trainer: one **workspace per browser session**, a **Clear workspace** button
  (`DELETE /workspace`), idle workspaces dropped after `TOOLKIT_WORKSPACE_IDLE_MIN`
  (default 240) minutes, and `409` if a second training starts in the same workspace.
- `scripts/menu_walkthrough.py`: drives every menu path of `main.py`, local only.
- `scripts/contrast_audit.js`: WCAG contrast check used for the UI fixes.
- Environment settings: `TOOLKIT_PORT`, `TOOLKIT_DEBUG`, `TOOLKIT_MAX_UPLOAD_MB`,
  `TOOLKIT_HTTPS`, `TOOLKIT_SECRET_KEY`, `TOOLKIT_WORKSPACE_IDLE_MIN`.

### Changed
- **Accuracy claims withdrawn** (README, `models/README.md`): the v4 CLI model uses the
  label-derived `Attempted Category` feature. The CLI now prints a warning when it loads.
- Web server: `waitress` on `127.0.0.1` by default; Flask's debug server only with
  `TOOLKIT_DEBUG=1`.
- UI: readable light and dark themes, system fonts (nothing downloaded), viewport and
  `lang` set, labels linked to inputs, visible keyboard focus; 0 contrast failures in
  the audit. Before/after screenshots in `docs/ui-screenshots/`.
- Dependencies pinned (`requirements.txt`, full `requirements.lock`); Flask 3.1.3 and
  requests 2.33.0 to fix published advisories.

### Fixed
- **Security:**
  - Uploaded model pickles were unpickled, which could run code (S1). Web model
    load/export is **paused** until signed bundles arrive in v5.0-rc1.
  - The CLI only unpickles model files whose SHA-256 is pinned in
    `models/manifest.json` (S17, B18).
  - Upload and export path traversal (S2, S4); uploads get unique names and are
    deleted once read (S3, S11).
  - Stored XSS from dataset text (S5), with a CSP and security headers as a second
    layer.
  - Error messages no longer reveal paths or exception text (S8).
  - JSON-only request bodies, upload size limit and clamped inputs (S9).
  - Cross-site requests blocked by an `Origin` check (S10). The session cookie is
    `HttpOnly`, `SameSite=Strict`, holds only the workspace ID, and is `Secure`
    with `TOOLKIT_HTTPS=1`.
  - All users shared one global dataset and model; each browser session now has
    its own workspace (S7).
- Library warnings are no longer silenced globally (B19).
- Launcher: answering "n" after a recon run exited the whole app (B15); the menus
  called each other recursively (R8).
- Light mode showed white text on white (U1); the dark-mode CSS block was never
  closed (U2); the dark-mode Data Preview values were invisible.

### Known limitations
See **Known limitations** at the top of the [README](README.md). In short:
- The web trainer's results are still inflated by leakage until step 0.4.
- Web model export/load is paused until rc1.
- The port scanner still sends banners to a third-party AI service when a key is
  set, and uses a retired MITRE CVE page with lookups run one at a time.
- Linux and macOS haven't been tested on real machines.

## [4.0] — before 2026-10

Original project as imported (`da41472`): menu launcher, port scanner, CLI IDS
(v4 model), Flask web trainer.
