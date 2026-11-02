# Roadmap: from v4 to now, and what comes next

How the project was changed, release by release, and what is planned. What exists today is in [PROJECT_STATUS.md](PROJECT_STATUS.md); the reasons are in [DECISIONS.md](DECISIONS.md) and [REVIEW_DECISIONS.md](REVIEW_DECISIONS.md); the original plan with every option weighed is [IMPROVEMENT_PLAN.md](../docs/IMPROVEMENT_PLAN.md); the day-to-day tasks are in the checklists (`V5.0_CHECKLIST.md` to `V5.3_CHECKLIST.md`).

**As of:** 2026-11-02.

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

### v5.3 so far
| Item | State |
|---|---|
| **F2** The full leaderboard: Saved Runs, the comparison, the charts, the CLI | Done, F2a–F2g (2026-10-30 to 2026-11-02) |
| **F3** Explainability | F3a–F3c done in the core (permutation importance; the explanation of one prediction); F3d–F3g open |
| **The outside review** | 2026-11-02: three reviews of the documents and a committee of five reviewers (security, architecture, reliability, ML, product). Result: the hardening section H of the checklist, and four design decisions (RD1–RD4). Written down; none implemented |

---

## 4. Next: the rest of v5.3, in this order

The order is the review's: **correctness → security → reliability → usability → features.** No new feature before the risks below are closed.

### Phase 1. Correctness (mostly documentation)
| Task | What |
|---|---|
| HC1 | Say how the test split is used: never to train or choose; reused afterwards by the permutation importance as a diagnosis; a validation split for any feature or setting chosen from it |
| HC2 | The limits of both explanations shown beside them; "importance concentration warning" instead of a hint of leakage |
| HC3 | How each CVE was matched: by version, or by name only (unconfirmed) |
| HC4 | One description everywhere: local-first with optional outside services; an IDS model trainer and predictor; no claim of generalization; Linux and macOS not tested; the tests marked by level |

### Phase 2. Security
| Task | What |
|---|---|
| HS1 | A new token on demand (from this computer only) that ends every remote session; what happens to each job; the statement that a token holder has the owner's rights |
| HS2 | HTTPS with the user's own certificate (Python's `ssl`), the SSH-tunnel way documented; HTTP stays with a clear warning |
| HS3 | A record of every scan with the source of its request |
| HS4 | A switch for each outside service (NVD, AI, AbuseIPDB) and one for all of them; "CVEs not checked" when NVD is off |
| HS5 | The permissions of `instance/` and the key on Windows; backup and restore, tested on a clean install |

### Phase 3. Reliability
| Task | What |
|---|---|
| HR3 (first) | One module that publishes every saved file safely (temporary file, then a publish that replaces nothing); one web server per instance (an OS lock); a journal so a cut-off job shows as `INTERRUPTED`; a bundle's state from its header |
| HR1 | The permutation importance on the training queue, bound to the run it was asked for |
| HR2 | Each importance result a file of its own per run and model |
| HR4 | Deleting a run and its files in a safe order; a storage report; registering an orphaned bundle again; disk limits |
| HR5 | A record for every successful training; Saved Runs list every training, exported or not |

### Phase 4. Usability and validation
| Task | What |
|---|---|
| HV1 | A user who is not the developer installs and uses every tool on a clean machine; each problem written as action, expected, actual, severity, fix, retest |
| HV2 | CI on Windows, Ubuntu and macOS. **Needs a push, the user's decision** |
| HV3 | A real run on a Linux VM, and on a Mac when one is available |
| HV4 | Later, not a gate condition: training on some days of CIC-IDS2017 and testing on others; leaving one attack out of training |

### Phase 5. Features
| Item | What |
|---|---|
| F3d | The routes: the importance job, `why` with a single prediction |
| F3e | The page: the top features chart, "Why?" with its limits, Calculate only for the workspace's run |
| F3f | The CLI: `ids --info`, the top reasons, `why` in `--json` |
| F3g | The docs, and the browser check at 8 sizes, both themes, keyboard only |

### Gate 5.3, then the tag
The review's conditions: remote access kept and its risks closed; real control of the outside connections; the limits of the evaluation and the explanations written; the importance kept with its run; one heavy job at a time; a defined behaviour on a sudden stop and a managed history; real use shown by a user who is not the developer, and no claim of Linux or macOS support before a real run. Then the usual gate (full files, fresh install, browser audits, the same numbers) and tag `5.3`, after the user approves it.

---

## 5. Later (after 5.3)

The plan's groups (§5 and §7 of the plan), with what the review asked to settle before each starts. None of these is approved yet.

| Item | Group | Goal | To settle first |
|---|---|---|---|
| F5 Scan reports | First | An HTML report of a scan (open ports, CVEs with CVSS, advice), printable | The PDF comes from the browser's print with a print stylesheet, not a hand-made PDF; long tables across pages; every banner and CVE text put in as text (XSS); never overwrite a file |
| F6 History | First | Every training and scan listed, reopened and compared | Built on HR4 and HR5 (records already saved, deletion and limits already defined): its own part is listing, paging and retention, not saving |
| F7 Recon and IDS | Next | One dashboard for both tools | The review objects to suggesting an IDS model from scan results: open ports and CVEs say nothing about which model reads traffic best. A risk dashboard from the scans, kept apart from any model advice |
| F9 Tuning | Next | Hyperparameter search with a time budget | Only on the training split (CV); the test split stays out; limits on time, memory and trials; a real Cancel |
| F10 Dashboard | Next | A home page with the recent jobs, models and scans | — |
| F8 PCAP and live capture | Stretch | Predictions from captured traffic | A project of its own: packet parsing, flow rebuilding, the same features as the training data, timeouts, capture rights on each OS, tests on real captures. Reading a PCAP alone does not make a live IDS |
| F11 Scheduled scans | Stretch | Scans on a schedule, alerts on a new open port | What happens when the app is closed, missed times, overlaps; never dependent on an open browser |
| F12 Arabic and RTL | Stretch | The pages in Arabic | — |
| F13 Offline CVE data | Stretch | CVEs without the internet | The copy's date, how it is refreshed, its size, how stale it is; an old copy proves nothing about new CVEs |
| An asyncio scanner (plan D13-B) | Stretch | Faster scans of large networks | — |

## 6. Not planned
- Accounts, roles or per-user ownership (RD1: one project trust domain)
- Microservices, a task queue service or a database server (plan §2)
- Shipping a trained model in the repository (plan D12, revised in 0.5)
