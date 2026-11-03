# Changelog

All notable changes to this project. Versions follow the roadmap in
[docs/IMPROVEMENT_PLAN.md](docs/IMPROVEMENT_PLAN.md); day-to-day progress is in
[docs/V5.4_CHECKLIST.md](docs/V5.4_CHECKLIST.md) (v5.4),
[docs/V5.3_CHECKLIST.md](docs/V5.3_CHECKLIST.md) (v5.3),
[docs/V5.2_CHECKLIST.md](docs/V5.2_CHECKLIST.md) (v5.2),
[docs/V5.1_CHECKLIST.md](docs/V5.1_CHECKLIST.md) (v5.1) and
[docs/V5.0_CHECKLIST.md](docs/V5.0_CHECKLIST.md) (v5.0).

## [Unreleased]

v5.4, so far: the scan history (F6) and the scan reports (F5)
([checklist](docs/V5.4_CHECKLIST.md)).

### Added

- Every finished scan, from the command line or the web, is saved once in
  `instance/scan_history/<scan id>.json`, under the id of its lines in the log: the request,
  each host and how it ended, the ports with their CVEs, AbuseIPDB's answers and the totals.
  A stopped scan saves nothing, and nothing is deleted on its own.
- The **Saved Scans** page (`/scans`): the saved scans, newest first, one shown again with
  **Scan again**, two compared port by port (a port the other scan didn't check is "not
  scanned there", not closed), and a delete that asks first.
- The same from the terminal: `python -m toolkit scan --history`, `--show`, `--compare`,
  `--delete`.
- Storage counts the saved scans as a kind of its own; the web server clears a stopped
  save's temporary files at its start.
- A saved scan's **report** in HTML: the ports to look at first, each CVE with how it was
  found (by version: a possible match to check; by name: unconfirmed), the advice for each
  port, the AI's text marked "may be wrong", and what the report can't say. **Open the
  report** on Saved Scans (and **Open its report** on the Port Scanner) to read or print it;
  Print, then Save as PDF, makes the PDF. **HTML** downloads it as one file that opens in
  any browser, with no script and nothing loaded; `python -m toolkit scan --report SCAN_ID`
  saves the same file in `instance/scans/`, never over another.

### Changed

- `jinja2` and `markupsafe` are pinned in `requirements.txt` and `pyproject.toml` (they
  already came with Flask): the scan report uses them, in one module only.

## [5.3] — 2026-11-03

v5.3 "impressive features", the first part: the full leaderboard and Saved Runs (F2), what a
model relies on and why it predicts what it does (F3), and the hardening an outside review
asked for (section H of the [checklist](docs/V5.3_CHECKLIST.md)): what the evaluation and the
explanations can and can't say, remote access, the outside services, files published whole,
one web server per data folder, deleting and storage. Project version: `5.3.0`. Scan reports,
the history and the dashboard (F5, F6, F10) come in 5.4.

**Test result:** 1,864 automated tests pass; `ruff`, `mypy` and `pip-audit` are clean. The
gate ([record](docs/release/5.3-gate.md)): the review's conditions (1), (2) and (6) checked
again on the real server; the full NSL-KDD files through the trainer with the keyboard only,
with a cancel, what the model relies on and **Why?**, and the web and the CLI label `KDDTest+`
the same, byte for byte; the ML report on the v5.3 code gives the 5.0 numbers (six cases, all
1,246 values equal); a fresh install on Windows passes every menu path; every page, Saved Runs
included, at 8 sizes from 320 × 568 to 1920 × 1080 in both themes, with no contrast failure.
Windows 10 Enterprise LTSC 2021, Python 3.14.3, scikit-learn 1.8.0.

**Not done:** the review's condition (7), real use shown (a user who is not the developer,
Linux and macOS, CI; HV1–HV3), was moved after 5.4 by the owner. Until then the toolkit is
tested on Windows only, and Linux and macOS are not called supported.

### Added
- **Charts of a run** (F2d) under the trainer's leaderboard: each model's CV macro-F1 fold by
  fold, CV against test macro-F1, the test recall of each class, and the fit time. Drawn by
  the page itself (no library), every number also written out, in both themes.
- **Saved Runs** (F2e), a page (`/runs`) and a link in every navbar and after an export: the
  saved runs, newest first, 50 at a time; one shown with its leaderboard and charts,
  or two compared. A difference (Δ = B − A) only when both runs scored the same test rows;
  otherwise the numbers side by side and why. It reads the runs' records only: nothing is
  uploaded or trained, no model is loaded ([user guide 4.8](docs/user_guide.md#48-compare-saved-runs)).
- **`toolkit ids --runs` and `--compare A B`** (F2f): the same in the terminal; `--before`
  pages through older runs ([reference](docs/reference.md#4-ids-intrusion-detection-with-an-export)).
- **Web API** (F2c): `GET /api/v1/runs`, `/api/v1/runs/<run id>` and
  `/api/v1/runs/compare` ([web_api.md](docs/web_api.md#saved-runs-v53-f2)); a finished
  training's result carries each model's CV scores fold by fold (`folds`).
- **What the model relies on** (F3) in the trainer's Train step: choose a model and click
  **Calculate**; each feature's values are shuffled in turn on the test rows (5,000 at most,
  5 repeats, a fixed seed) and the drop in macro-F1 is drawn for the 15 features with the
  biggest drop, with a progress bar and **Cancel**. Nothing starts until you click. A yellow
  note when one feature carries more than half of the effect (check that it is not a copy of
  the label). The result is saved with the run and shown again on **Saved Runs**
  ([user guide 4.5](docs/user_guide.md#45-step-5-train)).
- **Why?** (F3) under a one-row prediction, for the session's model and a loaded bundle: the
  values that pushed the model toward the predicted class and away from it, each swapped
  alone for its typical value in the training rows. An approximation, not causes, and the
  page says so; a model without probabilities (the linear SVM) says it can't say why
  ([user guide 4.6](docs/user_guide.md#46-step-6-use)). A run's record now keeps each
  column's typical value (not the rows); an older bundle explains its numbers only.
- **`toolkit ids --info`** (F3) ends with what the bundle's models rely on, when it was
  calculated in the trainer; a prediction in the menu shows its top 3 reasons, and `--json`
  gives every result's `why` ([reference](docs/reference.md#4-ids-intrusion-detection-with-an-export)).
- **Web API** (F3): `POST /api/v1/jobs/importance`, `GET /api/v1/runs/<run id>/importance`,
  and `why` in each one-row prediction's answer ([web_api.md](docs/web_api.md#explanations-v53-f3)).
- **Delete a run** (HR4) on Saved Runs and with `toolkit ids --delete RUN_ID`: it first names
  every file it removes and its size, and removes nothing before you say yes. The bundle is
  kept unless you tick it; kept, it is listed as an unregistered bundle and **Register again**
  brings the run back from the signed copy of the record it carries.
- **Storage** (HR4) on Saved Runs and `toolkit ids --storage`: what the toolkit keeps on this
  computer, by kind, the free space, and any leftovers by name. It only shows: nothing is ever
  deleted on its own.
- **Other computers** (HS1) on the home page of this computer while the toolkit is open to
  the network: what a holder of the token can do, **New token** (every other computer is
  locked out at once, and their waiting jobs are cancelled), and their jobs with **Stop**.
- **A switch for each outside service** (HS4) on the scan page and in `toolkit scan`
  (`--no-nvd`, `--no-ai`, `--no-abuseipdb`): NVD, the AI explanation (OpenRouter) and
  AbuseIPDB.
- **Save again** (HR5) when a finished training couldn't save its record (a full disk): the
  same run, under the same run id.

### Changed
- **Every finished training is saved** (HR5) on this computer when it ends: its record (the
  scores, the class names, the column names and each column's typical value, not your rows).
  Saved Runs, `--runs` and `--compare` list every training, not only the exported ones; each
  run says whether it was exported. **Export** writes the bundle only.
- **The outside services are on by default** (HS4, RD5): NVD always; the AI and AbuseIPDB
  when their key is in `.env` (putting it there is the choice to use them). A service that
  fails in a scan is not asked again in it: the summary says "Unavailable" and why, and its
  ports say `NOT CHECKED`, never "no known CVE".
- **Each CVE says how it was matched** (HC3): **version match**, or **name match,
  unconfirmed**, in the CLI, on the page, in the CSV and in the reports.
- **One web server per data folder** (HR3): a second `toolkit web` on the same `instance/`
  stops and says which one holds it. Records, importance results and bundles are written to a
  temporary file and published whole, or not at all. A job cut off by a stop is shown once as
  `INTERRUPTED` at the next start, in the terminal and on its page.
- **A disk floor and a size warning** (HR4): a new file is refused when the disk would keep
  less than `TOOLKIT_MIN_FREE_MB` free (1 GB), and the storage report warns past
  `TOOLKIT_STORAGE_WARN_MB` (20 GB). Nothing is deleted.
- The docs say what the test split is used for, what the explanations can't say, what the
  toolkit is (a local IDS model trainer and port scanner with optional outside services, not
  a live network IDS), and each test's level (HC1, HC2, HC4).

### Security
- **Remote access** (HS1): a new access token on this computer locks every other computer out
  at once; each job keeps where it came from; the terminal and the unlock page say that a
  holder of the token has your rights over the runs, bundles and scans.
- **A record of every scan** (HS3) in the log: when it starts and ends, the targets, the
  ports, where it was asked from and what may be sent outside, each value escaped.
- **`instance/` and `.env` limited to your account** (HS5): on Windows, at every start, the
  folder and the file get your account, SYSTEM and Administrators only, nothing inherited. The
  signing key is kept, so every bundle made here still loads.
- HTTPS for `--host` (HS2) was dropped by the owner: the toolkit is a local tool; `--host` is
  for a trusted network only, and says so.

### Fixed
- The trainer's leaderboard table had no style of its own (its cells had no padding and
  touched); it now looks like the data preview's table (F2g).
- **Why?** under a prediction was a 26 px target; it is 44 px, like the others (F3g).
- On Saved Runs, a long file name (an importance result's) ran past a phone's screen in the
  Delete panel; it wraps, there and in Storage (gate 5.3).

### Upgrading from 5.2
- Nothing to migrate. Bundles and records made with 5.2 still load and list. A bundle saved
  before 5.3 has no typical values for its categories: **Why?** explains its numbers only.
- A run you trained with 5.2 and never exported has no record; it can't be listed.
- With an AI or AbuseIPDB key in `.env`, those services are now on by default for a scan:
  turn each off with its switch or `--no-` flag, or remove the key.
- At its first start, 5.3 limits `instance/` and `.env` to your account (Windows).

### Known limitations
See **Known limitations** in the [README](README.md). New in this release:
- The review's condition (7) is open (above): no user who is not the developer has tried it,
  and there is no CI.
- What a model relies on, and **Why?**, are a diagnosis of one run and an approximation, not
  causes: features that go together share their effect, and the probabilities are not
  calibrated.
- "Published whole or not at all" holds after a crash of the process; after a power cut it is
  best effort.
- The full CIC-IDS2017 file was not part of the gate (see its record).

## [5.2] — 2026-10-30

v5.2 "the user's improvements": a theme button, pages for every screen, the code arranged by
interface, a port scanner that reads what it finds (with nmap as an optional engine), and the
port scan in the web app, also from your own virtual machines behind an access token (steps
2.1–2.4 of the [checklist](docs/V5.2_CHECKLIST.md)). Project version: `5.2.0`.

**Test result:** 1,484 automated tests pass; `ruff`, `mypy` and `pip-audit` are clean. The
gate ([record](docs/release/5.2-gate.md)): the full NSL-KDD files through the trainer with the
keyboard only, and the web and the CLI label `KDDTest+` the same, byte for byte; the ML report
on the v5.2 code gives the 5.0 numbers (six cases, all 1,246 values equal); a fresh install on
Windows passes every menu path; the home page, the trainer, the port scanner and the unlock
page at 8 sizes from 320 × 568 to 1920 × 1080 in both themes, with no contrast failure (text
and, new in this gate, field borders). Windows 10 Enterprise LTSC 2021, Python 3.14.3,
scikit-learn 1.8.0.

### Added
- **A theme button** (2.1) in the navbar of every page: Auto (follows the system), Light,
  Dark, kept in the browser.
- **The port scan in the web app** (2.4): a **Port Scanner** page (`/scan`) with the same
  choices as `toolkit scan`, the "I own this target or have written permission" box, a live
  log, progress and **Cancel scan**, a card per open port with its software, CVEs and CVSS
  level, and the report as TXT, JSON or CSV. One scan at a time, beside a training; nothing is
  sent, not even a DNS query, before the box is ticked; the API keys stay on the server;
  `TOOLKIT_SCAN_MAX_MIN` (default 60). See [docs/web_api.md](docs/web_api.md) and the
  [user guide](docs/user_guide.md) 3.6.
- **`toolkit web --host ADDRESS`** (2.4f): opens the web app to your network (for example
  your virtual machines' host-only network) behind a warning; another computer types an
  access token, new at every start and shown in the terminal, once in its browser; five wrong
  tries a minute; this computer's own browser needs none; a banner on every page while it is
  open. User guide 3.7.
- **nmap as an optional engine** (N1–N4): `toolkit scan --engine nmap` (and the page's
  choice) when nmap is installed, with `--service-detection`, `--timing`, `--scripts safe`
  and `--os-detect`; the toolkit's own scanner stays the default and needs nothing
  installed. `TOOLKIT_NMAP_PATH`.
- The report files get the columns `os_hint`, `cve_source`, `engine` and `cpe` (A, N).

### Changed
- **Pages for every screen** (2.2): a Menu button on phones (the sticky navbar from 55 % of a
  320 px screen to 11 %), less padding, the step bar on one line; the trainer as wide as the
  home page on large screens, lists in columns, a long form folded.
- **The scanner reads what it finds** (A1–A5): the product and version from a banner's own
  words (`OpenSSH_for_Windows_8.1` is OpenSSH 8.1); the service from its answer, else the
  port's usual name with a `?`; the OS guess from the TTL, the banners and the open ports,
  each with where it was seen, and saying when they disagree or the target is this
  computer; CVEs matched by version (CPE names) for about 20 products, else by keyword, and
  each report says which; every scan says first what it may send to each online service.
- **The Upload Dataset card** redesigned from the user's mockup: a dropzone, the chosen files,
  the result as tiles.
- **The code arranged by interface** (2.3): `toolkit/core` (shared by both tools),
  `toolkit/cli` and `toolkit/web`, the same layers inside each; `cli` and `web` never import
  each other; a complexity limit of 10 and 400 lines a file, checked on every commit.

### Security
- **The Host check** (2.4a): the web app answers only a request named with its own IP
  address or `localhost` and its port, against DNS rebinding.
- **`ping` by its full path** (found in N2): on Windows a `ping.exe` in the current folder
  could run instead of the system's.
- A field's border is at least 3:1 in both themes (WCAG 1.4.11, 5.2 gate).

### Fixed
- `scripts/menu_walkthrough.py` failed on the Host check and on the scan engine question
  asked when nmap is installed (5.2 gate).

### Upgrading from 5.1
- Code that imports the toolkit's modules: they moved to `toolkit.core`, `toolkit.cli` and
  `toolkit.web` ([architecture](docs/architecture.md)). The commands are unchanged.
- Bundles exported with 5.1 still load on the install that made them (2.3: a test bundle with
  the old module names, and real exports made before 2.3).
- Scripts that read the scan's CSV get four more columns at the end.

### Known limitations
See **Known limitations** in the [README](README.md). New in this release:
- `--host` is plain HTTP: someone who can read the network's traffic could take over a
  session; use it only on a network you trust.
- The full CIC-IDS2017 file, a real second computer, Linux and macOS were not part of the
  gate (see its record).

## [5.1] — 2026-10-25

v5.1 "clean architecture": the same three tools as one package with one command, a
versioned web API, training jobs you can follow and cancel, and the documentation
(steps 1.1–1.5 of the [checklist](docs/V5.1_CHECKLIST.md)). Project version: `5.1.0`.

**Test result:** 803 automated tests pass; `ruff`, `mypy` and `pip-audit` are clean. The v5.1
checks of the plan's §8 pass: a cancelled training on the full NSL-KDD files stops, frees its
memory and keeps nothing; the progress bar, Cancel and a reload during training work; the whole
flow works with the keyboard only; a fresh install on Windows passes every menu path ([gate
record](docs/release/5.1-gate.md), every §8 check mapped to its tests in the
[audit](docs/release/5.1-section8-audit.md)). The web and the CLI label the full `KDDTest+`
file the same, byte for byte. The ML report on the v5.1 code gives the 5.0 numbers: one run of
each of the six cases, all 1,246 values equal.
Windows 10 Enterprise LTSC 2021, Python 3.14.3 (an install on Python 3.11.0 checked too),
scikit-learn 1.8.0.

### Added
- **One command** (1.4b): `python -m toolkit` opens a flat menu (the banner once, the
  menu again after every tool, `q` to quit); `python -m toolkit scan`, `ids` and `web`
  start a tool directly, with its own flags, help and exit codes. Installed with
  `pip install ".[boost]"`, the same is the `toolkit` command (1.5b).
- **Scanner flags** (1.4c): `scan TARGET` asks only "are you authorised?"
  (`--i-am-authorized` skips it); `-p/--ports`, `--protocol`, `-v`, `--ai`, `--abuseipdb`,
  `--scan-unresponsive`, `--export`. Nothing but the CVE lookups leaves the computer
  unless a flag says so. Ctrl+C or the end of input cancels a scan ("Scan cancelled.",
  exit 130, no traceback).
- **Parallel CVE lookups** (1.4e): up to 4 products at a time, each product once per
  scan; NVD's request spacing is kept.
- **Training jobs** (1.2d): training runs in the background with a progress bar and a
  **Cancel training** button; one training at a time per server, others wait in a queue;
  a cancelled or failed job keeps nothing; `TOOLKIT_JOB_MAX_MIN` (default 180).
- **Resume after a reload** (1.3c): the page shows the session's dataset, results, a
  running training and the export link again (`GET /api/v1/workspace`).
- **Predict a CSV file** in the web trainer (1.4f): rows per class, matches with the
  file's label column, and the labelled CSV, the same file as `python -m toolkit ids --csv`;
  `TOOLKIT_MAX_PREDICT_MB` (default 512).
- **The web trainer opens the browser** once it is ready (1.4d); `--port`, `--no-browser`
  (`TOOLKIT_NO_BROWSER=1`).
- **Log files** (1.1b): one per tool in `instance/logs/`, rotated at 1 MB; API keys and
  `Authorization` / `Bearer` values redacted, no dataset values; `TOOLKIT_LOG_LEVEL`.
- **Packaging** (1.5b): `pyproject.toml` with the pins of `requirements.txt`, the extras
  `[boost]` (XGBoost, LightGBM) and `[dev]`; an installed copy keeps its data in a
  per-user folder.
- **Documentation**: the [user guide](docs/user_guide.md) with screenshots (1.5f), the
  [reference](docs/reference.md) of every flag, exit code and setting (1.5e), the
  [architecture](docs/architecture.md) with text diagrams and the
  [model card](docs/model_card.md) (1.5c); [`LICENSE`](LICENSE) (MIT, 1.5g).

### Changed
- **The web trainer in six steps** (1.3c): Data, Explore, Clean, Features, Train, Use,
  one at a time (a loaded bundle: Data → Use); 44 px touch targets and a usable 375 px
  layout; the page script as ES modules (1.3a); one stylesheet with light and dark tokens,
  contrast AA in both themes (1.3b); status messages announced to screen readers (1.3e);
  an infinite feature score shown as ∞ (1.3d).
- **Web API under `/api/v1`** (1.2b) with one error format,
  `{"error": {"code", "message"}}`; see [docs/web_api.md](docs/web_api.md).
- **Workspaces** (1.2c): a lock per workspace; at most 4 workspaces
  (`TOOLKIT_MAX_WORKSPACES`) and 4,096 MB of loaded datasets (`TOOLKIT_MAX_DATA_MB`), the
  least recently used other session's workspace dropped first; idle workspaces swept every
  5 minutes; leftover uploads removed at startup.
- **Layered package** (1.5d): `toolkit/ui`, `controllers`, `services` and `data`, one
  folder per tool in each; tests keep every import pointing down a layer.
- **Type checks** (1.5a): `mypy --strict` on the whole package, in the pre-commit hook.

### Security
- A CSRF token on every change (1.2e), with the `Origin` check and `SameSite=Strict` kept.
- `Cross-Origin-Opener-Policy`, `Cross-Origin-Resource-Policy: same-origin` and a
  `Permissions-Policy` on every answer; `Cache-Control: no-store` on API answers and the
  trainer page (1.2e).
- CSP `style-src 'self'`: no inline styles in the page (1.3b).
- The web trainer's port is never shared (1.4d): on Windows a second server could bind a
  port already in use without an error; it is now bound exclusively, and a port in use is
  a message and exit code 2.
- A bad `TOOLKIT_*` setting is refused before the tool starts (1.5e): a one-line message
  and exit code 2. Before, a limit of `0` or less was accepted (`TOOLKIT_MAX_WORKSPACES=0`
  refused every workspace) and on/off settings silently meant off for anything but `1`.

### Fixed
- When every chosen model failed in cross-validation, the final fit trained the default
  models that nobody chose (1.2d).
- `OPENROUTER_MODEL` was not read from `.env` as documented (1.1b).
- A bad `TOOLKIT_LOG_LEVEL` crashed `scan`, `ids` and `web` with a traceback (1.5e).
- Keyboard focus fell to the page when Train and then Cancel were disabled during a
  training; it now moves to the status, Cancel, the progress and the result (5.1 gate).
- `scripts/menu_walkthrough.py` counted a page from another server when its own web trainer
  couldn't take the port, and crashed when no page answered (5.1 gate).
- The README's install steps now say what to do when PowerShell refuses `Activate.ps1`, and
  to upgrade setuptools on Python 3.11 (5.1 gate).

### Removed
- The `tools/` folder (1.4a): the tools live in `toolkit/`.
- `tools/Port_scan/api_keys.txt` is no longer read (1.4a): keys come from environment
  variables or `.env`; the scanner says so once while the old file is on disk.
- The web routes outside `/api/v1` (1.2b) and `POST /api/v1/run` (1.2d, now a job).
- The module entry forms `python -m toolkit.web`, `toolkit.ids_cli` and
  `toolkit.recon.cli` (1.5d), and the menu's "Proceed to Intrusion Detection?" and
  "another detection?" questions (1.4b).

### Upgrading from 5.0
- Start the toolkit with `python -m toolkit` (`python main.py` does the same).
- Move the keys of `tools/Port_scan/api_keys.txt` into `.env` (copy `.env.example`).
- Scripts that call the web trainer use the `/api/v1` routes and send the page's token in
  `X-CSRF-Token` with every change ([docs/web_api.md](docs/web_api.md#security-token)).
- On/off settings (`TOOLKIT_HTTPS`, `TOOLKIT_DEBUG`, `TOOLKIT_NO_BROWSER`) take only `1`
  (on) or `0` (off), and every number setting must be 1 or more
  ([docs/reference.md](docs/reference.md#71-toolkit_-settings)).
- Bundles exported with 5.0 still load on the install that made them (1.5d: a test bundle
  with the old module names, and two real exports made before 5.0).

### Known limitations
See **Known limitations** in the [README](README.md). New in this release:
- Bundles still load only on the install that made them; importing them on another
  computer is not part of 5.1.
- A bundle is held in memory two to three times while it loads, so the 1.98 GB random
  forest export of 5.0 doesn't load on a PC with 8 GB of memory.
- Open, recorded in the [checklist](docs/V5.1_CHECKLIST.md): long checkbox lists take one
  Tab stop per item; a native `confirm()` before Clear workspace; after a reload a session
  with a dataset and a loaded bundle shows the dataset path; the navbar takes about a
  quarter of the screen at 375 px; `--json` prints every digit of the confidence; `ids` with a missing file
  shows Python's own `[Errno 2]` message.
- Linux and macOS are not tested on real machines.

## [5.0.1] — 2026-10-21

Security patch: dependency update only, no code change. Project version: `5.0.1`.

**Test result:** 419 automated tests passed (`python -m pytest`), ruff clean, `pip-audit`
clean (installed environment and `requirements.lock`), `pip check` clean; a real HTTPS
request to the NVD API with the new CA bundle succeeded.

### Security
- `certifi` 2026.10.18 → **2026.10.19** in `requirements.lock` (PYSEC-2026-4320 /
  GHSA-6q48-6c7w-9gw2: revoked eMudhra intermediates under the `emSign ECC Root CA - G3`
  root). `requests` uses this CA bundle for the scanner's NVD, OpenRouter and AbuseIPDB
  calls. The advisory was published after 5.0-rc2, and the 5.0 gate did not re-run
  `pip-audit`; it is run again for this patch.
- Update an existing install with `python -m pip install -r requirements.lock`.

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
