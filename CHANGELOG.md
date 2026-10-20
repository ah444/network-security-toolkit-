# Changelog

All notable changes to this project. Versions follow the roadmap in
[docs/IMPROVEMENT_PLAN.md](docs/IMPROVEMENT_PLAN.md); day-to-day progress is in
[docs/V5.0_CHECKLIST.md](docs/V5.0_CHECKLIST.md).

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
