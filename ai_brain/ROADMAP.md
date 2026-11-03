# Roadmap: from v4 to now, and what comes next

How the project was changed, release by release, and what is planned. What exists today is in [PROJECT_STATUS.md](PROJECT_STATUS.md); the reasons are in [DECISIONS.md](DECISIONS.md) and [REVIEW_DECISIONS.md](REVIEW_DECISIONS.md); the original plan with every option weighed is [IMPROVEMENT_PLAN.md](../docs/IMPROVEMENT_PLAN.md); the day-to-day tasks are in the checklists (`V5.0_CHECKLIST.md` to `V5.3_CHECKLIST.md`).

**As of:** 2026-11-03. The plan ends with 5.4 (section 5).

---

## 1. The starting point: v4 (2026-10-19)

The project arrived as three large scripts launched by `main.py`: `Port_Scan.py` (input, network, outside APIs and export in one script), an IDS command line with its own preprocessing and pickled models, and a 1,724-line Flask `app.py` with routes, ML, state and files in one file. It was measured before anything changed (`docs/baseline/`).

| Problem | Why it mattered |
|---|---|
| Data leakage | The CLI model used `Attempted Category`, derived from the label; the NSL-KDD flow used `level`. The documented ~81% was 44.3% macro-F1 on real rows |
| Preprocessing copied four times | Training and serving drifted apart |
| Pickle uploads | Uploading a model file could run any code |
| XSS and path traversal | Dataset text went into the page as HTML; file names were not cleaned |
| Global state | Every user shared one dataset and one model |
| Privacy | Full banners went to an outside AI; a retired MITRE page was scraped |
| Quality | No tests, no pinned versions, no packaging; light mode was white text on white |

Root causes: no shared core; global mutable state; outside input trusted; presentation, logic and I/O mixed; no tests or packaging.

## 2. How every step is done
1. A test for the correct behaviour first (it fails, which proves the problem), or a characterization test for behaviour that must not change
2. The smallest change that makes it pass
3. The fixed logic moved into the shared core; the callers moved one at a time; the old code deleted
4. The checks: the full tests, ruff, mypy, pip-audit, and the browser when a page changed
5. One commit per step. No step starts before the user approves its proposal; a tag only after the user approves it; nothing is pushed

Every tag is shippable: the app starts, every menu path works, the README lists the limits, the CHANGELOG has the entry. Each release ends with a gate recorded in `docs/release/`.

---

## 3. Done

| Release | Date | Goal | The main work | Tests |
|---|---|---|---|---|
| 5.0-alpha | 2026-10-20 | Honest and safe | False accuracy claims removed; pickle uploads stopped; XSS and path traversal closed; CSP and security headers; a workspace per session and Clear workspace; `waitress` on 127.0.0.1; both themes readable | 122 |
| 5.0-beta | 2026-10-20 | A clean ML core | The loader and content-based profiles; the leakage guard; dedupe before the split; the split strategies with reason codes; one pipeline per model; registries; every seed fixed; CV-based selection | 295 |
| 5.0-rc1 | 2026-10-20 | The web app on the new core | The TrainingRun record; signed bundles; previews from the training split only; the old preprocessing deleted (`app.py` 1,857 → 866 lines) | 353 |
| 5.0-rc2 | 2026-10-20 | The CLI and the scanner | The CLI on the user's own bundles; the scanner's authorization question and limits; IPv6; CVEs from NVD 2.0; the AI after consent with product and version only; dataset licences documented; the leaky v4 models deleted | 416 |
| 5.0, 5.0.1 | 2026-10-21 | The first release | The ML report comparing v4 and v5 (three runs per case); a fresh install walked through every menu path; 5.0.1 updated `certifi` for an advisory published after rc2 | 419 |
| 5.1 | 2026-10-25 | Clean architecture | `python -m toolkit`; the `/api/v1` API with one error shape; training as a job with progress and Cancel; the page restored after a reload; Predict a CSV; CSRF; `mypy --strict`; packaging with `pyproject.toml`; the full docs | 803 |
| 5.2 | 2026-10-30 | The user's improvements | The theme button; every page at every size; the code arranged as `core`, `cli`, `web` with the same layers inside; the scanner reading banners, versions and OS evidence; optional nmap; the scan on the web; `--host` with a token | 1,484 |
| 5.3 | 2026-11-03 | Impressive features, the first part, and the review's hardening | Saved Runs and the comparison; what a model relies on and **Why?**; a new token ends every remote session; a switch for each outside service; files published whole; one web server per data folder; `INTERRUPTED` after a stop; delete and storage; a record for every training | 1,864 |

### v5.3 in detail
| Item | State |
|---|---|
| **F2** The full leaderboard: Saved Runs, the comparison, the charts, the CLI | Done, F2a–F2g (2026-10-30 to 2026-11-02) |
| **F3** Explainability: what a model relies on, **Why?**, the CLI | Done, F3a–F3g (2026-11-02 to 2026-11-03) |
| **The outside review** | 2026-11-02: three reviews of the documents and a committee of five reviewers (security, architecture, reliability, ML, product). Result: the hardening section H of the checklist, and four design decisions (RD1–RD4); RD5 followed on 2026-11-03 |
| **H Phase 1** Correctness (HC1–HC4) | Done |
| **H Phase 2** Security | Done: HS1 (a new token ends every remote session), HS3 (a record of every scan), HS4 (each outside service on by default with its own switch, RD5), HS5 (`instance/` and `.env` limited to this account). HS2 (HTTPS) dropped by the owner on 2026-11-02: the toolkit is a local tool |
| **H Phase 3** Reliability (HR1–HR5) | Done: the importance on the training's queue and in a file of its own; files published whole; one web server per data folder; `INTERRUPTED` after a stop; delete, storage, register again, disk limits; a record for every training |
| **Gate 5.3** | 2026-11-03, [record](../docs/release/5.3-gate.md): the review's conditions (1)–(6) met; (7) open |

---

## 4. Left from v5.3: the review's condition (7), after 5.4

Decided by the owner on 2026-11-03: these depend on things outside the project (a tester, a
virtual machine that downloads from the internet, a push), so they no longer hold the
features back. Until HV3, Linux and macOS are never called supported.

| Task | What |
|---|---|
| HV1 | A user who is not the developer installs and uses every tool on a clean machine; each problem written as action, expected, actual, severity, fix, retest |
| HV3 | A real run on a Linux VM, and on a Mac when one is available |
| HV2 (last) | CI on Windows, Ubuntu and macOS. **Needs a push, the user's decision**, made with it |
| HV4 | Later, not a gate condition: training on some days of CIC-IDS2017 and testing on others; leaving one attack out of training |

The order: 5.4 (section 5) and its gate, then HV1, then HV3, and last HV2 with the push.

---

## 5. Next: 5.4, the last release

Decided by the owner on 2026-11-03: 5.4 is the project's last planned release. The three
features complete one use of the toolkit: a scan or a training, its result kept in the
history, opened again, a scan's report printed, and the home page showing where things
stand. They come in this order, each proposed and approved as usual.

| Item | Goal | What it must respect |
|---|---|---|
| F6 History | Every training and scan kept, opened again and compared | The results, the settings and the metadata, never copies of a dataset. The trainings rest on HR4 and HR5. **Done 2026-11-03** ([V5.4_CHECKLIST.md](../docs/V5.4_CHECKLIST.md), F6a–F6e): every finished scan, the CLI's and the web's, is saved once in `instance/scan_history/` (DECISIONS F6) and deleted only by the user; the Saved Scans page and `scan --history` show, compare and delete them; Storage counts them |
| F5 Scan reports | An HTML report of a scan (open ports, CVEs with CVSS, advice), printable | A CVE matched by a version is a possible match to check, not a confirmed vulnerability: each shows how it was matched (version, or name only and unconfirmed), its source and the date of the lookup; the AI's text marked "may be wrong"; advice general and tied to its CVE. The PDF comes from the browser's print with a print stylesheet; every banner and CVE text put in as text (XSS); a file never overwritten |
| F10 Dashboard, with F7 | A home page with the recent jobs, models and scans | Built on F6. The scans and their risks in one part, the models, the trainings and the predictions in another, side by side: no link made between them, and no IDS model suggested from open ports or CVEs (the review's objection to F7 as first planned) |

Then the gate and the tag `5.4`, after the owner approves it.

## 6. Not planned
- Accounts, roles or per-user ownership (RD1: one project trust domain)
- Microservices, a task queue service or a database server (plan §2)
- Shipping a trained model in the repository (plan D12, revised in 0.5)

Set aside by the owner on 2026-11-03, outside this project's plan (they may come back after
5.4, as work of its own):

| Item | Goal | Why not now |
|---|---|---|
| F9 Tuning | Hyperparameter search with a time budget | Of real value but not needed for a complete toolkit: it needs cross-validation on the training split only (the test split never choosing), limits on time, memory and trials, and a real Cancel |
| F12 Arabic and RTL | The pages in Arabic | A change to the interface, not a new capability; it touches every page and the stylesheet |
| F8 PCAP and live capture | Predictions from captured traffic | A project of its own: the flows and features must be built exactly as the training data's (CIC-IDS2017), or the predictions mislead without anyone seeing it; capture rights differ on each OS |
| F11 Scheduled scans | Scans on a schedule, alerts on a new open port | A service running while the app is closed, missed times, overlaps: not a local tool the user opens and closes |
| F13 Offline CVE data | CVEs without the internet | A local copy needs refreshing and grows old; an old copy says nothing about new CVEs. Asking NVD when needed, and saying when it can't be reached (RD5), stays |
| An asyncio scanner (plan D13-B) | Faster scans of large networks | Speed only, not a new capability, and the limits (256 hosts) fit the use; only if a measurement shows a real bottleneck |
