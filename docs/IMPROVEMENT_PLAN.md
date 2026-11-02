# Network Security Toolkit — Whole‑Project Audit, Architecture & Improvement Plan

## Context
You asked for a complete review of the **whole project**, followed by a plan that covers **everything**:
1. a comparison of **architecture styles**, applied to this project,
2. improvements for **every layer** (UI, API, logic, ML, data, performance, security, ops, tests, docs),
3. **new features**,
4. **code quality standards**,
5. every leak and bug.

Each problem is written as **problem → options → pros/cons → recommendation**. The toolkit runs locally but must work on **any device/OS**, for web training and the CLI. This is a plan only; no code has been changed yet.

**Ground rules:** there is no deadline, so the full scope stays. Work runs **release by release** (§7). Nothing is implemented until you have reviewed the plan and said go.

### Revision 2 — changes after review
| Change | Where |
|---|---|
| The CLI model is now **trained in the (fixed) web trainer** and exported as a bundle the CLI loads. No separate notebook or script is needed | D12, D5 |
| The roadmap is split into **three releases**: v5.0 correct & safe → v5.1 clean architecture → v5.2+ features *(2026-10-25: v5.2 became the user's improvements, the features moved to v5.3+)*. No feature work starts until the v5.0 checks pass | §7 |
| **Tests first, then fix, then migrate callers one at a time.** The new folder layout is the end state, not the first step | §7 |
| Abstractions (ports) only at **real boundaries**: model storage, dataset loading, intel providers, job progress. Everything else stays plain code *(dataset loading later demoted, Rev 4)* | §2 |
| Signing rule made precise: **never deserialize an unsigned or untrusted bundle**. joblib is not "safe" by itself. Added a pinned hash for bundled models and a cross-device import path | D11 |
| **`TrainingRun` record** added as a first-class concept for reproducibility | D21 |
| Cheap, highly visible fixes moved into v5.0: README accuracy claims, CSS white-on-white and the unclosed dark-mode block | §7 |
| Session isolation (S7) goes in v5.0 **only if** other devices reach the web app over the network; otherwise v5.1 *(superseded by Rev 4: unconditional)* | §7 |
| The verification section (§8) is now the **definition of done** for each release | §8 |

### Revision 3 — changes after second review
| Change | Where |
|---|---|
| Step 0.4 split into 7 milestones (0.4a–g), each its own commit with tests | §7 |
| Trust policy narrowed: v5.0 loads **only** locally signed bundles and the repo-pinned default. The "trust this file" option is removed. Cross-device/third-party import moves to v5.1+ (own-key import, `skops`/ONNX) | D11 |
| `split_strategy` (official / temporal / group / stratified) is declared **per dataset profile** and recorded in the TrainingRun; generic stratified is only a fallback with a warning | D9, D16, D21 |
| New invariant: the test set never influences selection, balancing (incl. the recommender), tuning, thresholds or model choice; scored once | D9, §8 |
| Basic leaderboard (F2) moved forward into 0.4d as evidence that the ML redesign works; full UI stays in v5.2 *(now v5.3)* | §7 |
| Wording: `ids_core` is the "ML core layer", not "domain" in the strict DDD sense | §2, §3.5 |
| Tests: correct-behaviour tests for bugs (fail first) plus characterization tests for behaviour to keep | §7 |

### Revision 4 — changes after third review
| Change | Where |
|---|---|
| Internal v5.0 checkpoints `5.0-alpha → beta → rc1 → rc2 → final`, each a tagged known-good commit with an exit criterion | §7 |
| Minimal session isolation is **unconditional** in v5.0 (two tabs on localhost already collide); the full TTL workspace stays in v5.1 | D7, §7 0.8 |
| Python policy: 3.11–3.13 supported (CI required), 3.14 experimental; develop in a 3.13 venv *(replaced: develop and test on your current 3.14, `requires-python >=3.11`)* | D19, D18 |
| CPU limit: `TOOLKIT_MAX_WORKERS`, default `max(1, cpu_count − 1)` | D8 |
| Temporal split is a **validated recommendation** with fallbacks; added `temporal_per_class`, because each CIC-IDS2017 attack type occurs on a single day | D9 |
| Dataset acquisition is a formal **gate** for 0.5 (acquired, hashed, profiled, documented) | §7 |
| `DatasetSource` demoted from port to plain loaders + registry → only 3 ports | §2 |
| Extra tests: profile determinism, split fallback, unknown-name label leak, malformed bundles, cancellation, resource limits | §8 |
| Scope discipline: no v5.1 restructuring during v5.0 | §7 |

### Revision 5 — changes after fourth review
| Change | Where |
|---|---|
| **Every checkpoint is shippable** (works, README limitations, CHANGELOG, tests green), as protection against abandonment | §7 |
| **Baseline snapshot (0.1b)** of current metrics before any change | §7, §8 |
| **Dataset fallback:** if CIC-IDS2017 isn't available by rc2, ship an NSL-KDD multi-class bundle (on disk, official split) with an honest note, instead of blocking v5.0 *(decision point moved to beta exit, Rev 6)* | D12, §7 |
| **XSS boundary:** v5.0 escapes `innerHTML` in place + CSP; the JS module rewrite stays in v5.1 | §7 0.3 |
| **Exact label-leak rule:** single-feature depth-3 tree, balanced accuracy ≥ 0.99 block / 0.95–0.99 warn, override recorded | D9, §8 |
| **Version-mismatch policy:** major/minor differences refused before loading, patch differences warn | D11, §8 |
| **Signing-key backup/loss policy** + test that `instance/` is gitignored | D11, §8 |
| **Prompt injection upgraded:** LLM never drives decisions; severity from CVSS; output shown as untrusted plain text | D15, §8 |
| **`pip-audit`** in CI (Dependabot once on GitHub) | D19 |
| **Dataset licensing** per dataset in `docs/datasets.md` (0.5b) | §7, D20 |
| **Runaway protection:** one training job per workspace (`409`) instead of a generic rate limiter | D8 |
| **Cookie hygiene:** `HttpOnly`, `SameSite=Strict`, ID-only cookie | D7, S10 |
| **"Clear workspace"** button in v5.0 | D7, §7 0.8 |
| **Gradual mypy:** strict only on touched modules in v5.0, full in v5.1 | §6.1 |
| **Accessibility:** contrast, labels, focus, `lang` in v5.0; keyboard flow + `aria-live` in v5.1 | D2, §8 |
| **`docs/V5.0_CHECKLIST.md`** drives day-to-day work | new file |
| Ordering fix found while writing the checklist: **interim trust rule in 0.3** (web model upload disabled, CLI loads only hash-pinned v4 files) until signed bundles arrive in 0.4f; the `409` rule moved to 0.8 because it needs workspaces *(window extended to 0.4g and export paused too, Rev 6)* | §7 |

### Revision 6 — fifth review + full self-review
**From the review (verified against the code first):**
| Change | Where |
|---|---|
| **0.4f/0.4g ordering:** 0.4f is the bundle *library* only; web load/export come back in 0.4g, after the routes use the new pipeline | §7 |
| **Dataset decision moved to `5.0-beta` exit** (not rc2), so 0.5 can start in rc1 on NSL-KDD if needed | D12, §7 |
| **Baseline made trustworthy:** scripted click sequence, same pinned environment as v5.0, `n_jobs=1`, 3 runs with spread recorded | §7 0.1b, §8 |
| **Web export also paused** during the trust window (no orphan files); in-session train/predict keep working; documented as a known limitation | §7 0.3 |
| `409` test made meaningful with a blocking hook | D8, §8 |
| v4 version check in 0.3 via a pinned manifest (versions read from the pickles; `config.json` has no version field); B19 `filterwarnings` removal scheduled | D11, §7 |
| Content-based profile detection; CIC must match ≥ 2 published variants | D9, §8 |
| Exact reason codes; fake-CVE injection test; formula-injection test separate from ANSI; effective-worker test | D9, §8 |
| Delete v4 `config.json` with the pickles; R3 disposition (one-at-a-time + timeouts in v5.0); S16 raised to Medium with a required authorization confirmation | D12, §7, §4.2 |
| D22 (running job keeps its workspace alive) and D23 (`Secure` via `TOOLKIT_HTTPS`) decided | D7 |
| System font stack instead of loading `Inter`; before/after screenshots for the CSS fix | D2, U2, U3 |

**Review points corrected or rejected:**
| Claim / suggestion | Outcome |
|---|---|
| "`train_test_split` has no `random_state`" | Wrong: both use 42. The real non-determinism is the unseeded inner SMOTE (new **B20**) |
| "Unclosed `@media` is ignored in some browsers" | Wrong: the CSS spec auto-closes it everywhere → dark-mode-only (U2 as stated) |
| "B21 two-sided bug" | Harmless dead JS (server already converts values); noted under B4 |
| "Name patterns `id`, `row`, `index`" | Adopted as **whole-word** matches only (a substring `id` would block `Idle Mean`) |
| "> 99 % unique = ID" | Adopted only for integer/text columns (real continuous features are also almost all unique) |
| "Flag columns constant in train but not test" | **Rejected:** uses the test set to choose features, which breaks test isolation |
| "Promote `DatasetRepository` + `TrainingRunStore` to ports (5 total)" | **Rejected** in favour of a functional core (`train(df, …) → TrainingRun`); tests need no fakes |

**Found in my own full re-read of the plan:**
| Problem | Fix |
|---|---|
| The label-leak rule (≥ 0.99 → block) would wrongly block genuinely strong features on easy binary data such as the CIC DDoS file | Block only if ≥ 0.99 **and** the feature acts like a relabelling (about one value per class); otherwise warn |
| Cookie hygiene was scheduled in 0.3, but the session cookie only exists once workspaces arrive in 0.8 | Moved to 0.8 |
| CSP in 0.3 would break the page: templates and `main.js` use inline `style=` attributes | v5.0 CSP: strict `script-src`, `style-src 'self' 'unsafe-inline'`; tightened in v5.1 |
| "Green in CI on Win/Linux/macOS", but there is no GitHub remote | **CI host** decided: everything local on Windows with plain `pytest` and your current Python 3.14; Linux/macOS supported through portability rules + patched-platform tests |
| Baseline and v5.0 could run in different environments → differences could come from library versions | Baseline and v5.0 run in the **same** pinned project environment (your Python 3.14 `.venv`) |
| Shipped model's TrainingRun lived only in gitignored `instance/` → provenance not in the repo | Commit `default.run.json` next to the default bundle |
| Version policy refused any numpy minor difference (needlessly strict) | numpy: refuse only on major; strict major/minor for estimator libraries |
| CSRF had no v5.0 answer (token is v5.1) | v5.0: `SameSite=Strict` + JSON-only state-changing routes + `Origin` check on uploads |
| Original CIC "MachineLearningCSV" files (and the local DDoS file) have **no time column**, so `temporal_per_class` can't apply, and grouping by capture file would leave classes on one side only | CIC profile detects the variant; without time → `stratified` + visible "optimistic" warning recorded in the TrainingRun/model card; group fallback only if every class appears on both sides |
| Hexagonal row still listed the old 4 ports; §6.1 still said "Repository for datasets"; "D1–D20" heading; misplaced dataset paragraph; "all cores" in §3.9; cancellation and `python -m toolkit` listed as v5.0 checks | All made consistent (3 ports, functional core, D1–D23, `TOOLKIT_MAX_WORKERS`, v5.1 labels) |

---

## 0. Coverage — every file was inspected
| File | Lines | Role |
|---|---|---|
| `main.py` | 123 | Menu launcher; runs each tool as a subprocess |
| `tools/Port_scan/Port_Scan.py` | 784 | TCP/UDP scan, banner probes, TTL OS guess, MITRE CVE scrape, DeepSeek LLM, AbuseIPDB, export |
| `tools/Port_scan/api_keys.example.txt` | 11 | Key template |
| `tools/IDS/IDS_CLI - CIC-2017/IDS_with_Machine_Learning.py` | 436 | CLI predictor (RF / XGBoost) |
| `…/models/config.json`, `README.md`, 4× `.pkl` | — | v4 model: 8 features incl. `Attempted Category`, 16 classes. Pickles saved with **sklearn 1.6.1 + numpy 2.x** (read with `pickletools`, not executed) |
| `tools/IDS/IDS_Web_CIC_Dynamic/app.py` | 1,724 | Flask trainer: upload, preview, filter, SMOTE, SelectKBest, 11 models, export/load, predict |
| `…/static/js/main.js` | 1,061 | All front-end logic |
| `…/templates/home.html`, `dynamic_ids.html` | 48 / 226 | Pages |
| `…/static/css/style.css`, `style2.css` | 444 / 801 | Styles |
| `…/uploads/*` | 42 MB | CIC DDoS CSV (225,745 rows, **23,416 duplicates**), `KDDTrain+.txt` (header + `level` column), `KDDTest+.txt` (**no header**) |
| `…/dataset/desktop.ini` | — | Folder is empty (refers to a missing UNSW-NB15 file) |
| `README.md`, `requirements.txt`, `.gitignore` | — | Docs and config |

Environment: Python 3.14.3, sklearn **1.8.0** (models saved with 1.6.1), numpy 2.3.5, Flask 3.1.2. No git and no tests. `Project.zip` has already been removed.

---

## 1. Current architecture & root causes
```
main.py ──subprocess──► Port_Scan.py   (one script: UI + network + 3rd-party APIs + export, module globals)
        ──subprocess──► IDS CLI         (own loader + own preprocessing, pickle)
        ──subprocess──► Flask app.py    (routes + ML + state + file IO in one file)
                             ├─ global dicts: data_store / model_config / loaded_model_config
                             └─ main.js: HTML built as strings → innerHTML
```
Current style: **"big-ball-of-mud" scripts**, three separate monoliths with no shared layer.

**Root causes:**
1. No shared domain core. Preprocessing is copied 4× (`app.py:610-665`, `:904-922`, `:1032-1062`, CLI), so the copies drift. This causes the leakage and the training/serving skew.
2. Global mutable state, so there is no isolation, data mixes between users, there are races, and memory is never freed.
3. Untrusted input is trusted (filenames, pickles, CSV text → HTML, banners → LLM).
4. Presentation, logic and IO are mixed in every function (e.g. `Port_Scan.main()` does input, scanning, printing, API calls and export).
5. No tests, no pinned versions, no packaging, so the project breaks on other devices.

---

## 2. Architecture styles — compared for THIS project

| Style | How it would look here | Pros | Cons | Fit |
|---|---|---|---|---|
| **Layered (N-tier) / MVC** | `presentation` (Flask views, CLI) → `services` → `domain` (ML, recon) → `infrastructure` (files, APIs) | Simple, well known, easy to explain in a thesis; Flask matches MVC | Layers can leak into each other; domain still depends on infrastructure | Good |
| **Clean / Hexagonal (ports & adapters)** | Core `ids_core`/`recon` depend only on interfaces (`ModelStore`, `IntelProvider`, `ProgressSink`); adapters for joblib bundles, NVD/AbuseIPDB/OpenRouter, Flask, CLI | Core is fully testable with fakes; swap NVD↔offline DB or OpenRouter↔Ollama easily; CLI and web share the same core | More files and interfaces; can be too much ceremony for small parts | Very good for the core |
| **Modular monolith** | One installable package `toolkit/` with clear modules (`recon`, `ids_core`, `web`, `cli`) and one process per tool | One deployable; no network between modules; works offline on any device; clean boundaries | Must enforce module boundaries yourself | **Best overall** |
| **Microservices** | Separate scanner, training and prediction services over HTTP + a queue | Independent scaling and deployment | Huge ops overhead (Docker, networking, auth) for a local tool; slower; poor on weak devices | Poor |
| **Plugin / microkernel** | Core + registries: models, balancers, banner probes, intel sources, dataset profiles are plugins | Add a model or probe without editing core code; replaces long `if/elif` chains (`app.py:1101-1141`, `:1183-1276`, `Port_Scan.py:200-445`) | Needs a registry and conventions | Very good for extensible parts |
| **Event-driven** | Training/scan jobs publish progress events; UI subscribes (SSE/polling) | Live progress; decouples long jobs from requests | Event plumbing; debugging is harder | Good for jobs only |
| **Pipe-and-filter** | ML as `load → clean → dedupe → split → impute → encode → scale → select → sample → model` (sklearn/imblearn Pipeline) | Leakage-proof ordering; one object to save; reproducible | Custom steps need a transformer API | **Required for ML** |
| **Client–server SPA** | Static front end calls a JSON API (`/api/v1`) | Clear contract; the CLI could use the same API | Needs API design + a JS app | Good (lightweight, no framework) |

**Recommendation — a hybrid:** a **Modular monolith** whose core follows **Hexagonal** rules, with **Plugin registries** for models, balancers, probes, intel sources and dataset profiles, **Pipe-and-filter** for ML, **Event-driven jobs** for long tasks, and **Layered MVC** in the web layer with a versioned JSON API.
Why: one install that runs on any OS, a testable core, honest ML, and easy to extend. Microservices are rejected because they add network and ops cost with no benefit for a local tool.

**Rule against over-engineering:** a "port" (interface) is created only where an implementation really gets swapped or faked in tests:
- `ModelStore` (signed bundles on disk)
- `IntelProvider` (NVD, AbuseIPDB, LLM: online / offline / mock)
- `ProgressSink` (job progress to the web, the CLI, or tests)

Dataset loading (CSV/TXT/Parquet + profiles) starts as **ordinary loader functions behind a registry**, not a port. It becomes a formal `DatasetSource` port only if a second kind of source appears (SQLite history, remote datasets).

Everything else (probes, TTL table, metrics, the balancing heuristic) is plain functions plus simple dict registries. No interface for a single implementation.

**Functional core, imperative shell (why 3 ports are enough).** The ML core never reads or writes files itself. It takes data in and returns results: `train(df, profile, config) → TrainingRun` (which contains the fitted pipeline and an `EvaluationResult`). Loading datasets, saving runs and writing bundles happen at the edges (web routes, CLI). Tests pass a small in-memory DataFrame straight in and check the returned object, so there is no `FakeDatasetRepository` or `FakeTrainingRunStore` to write and nothing to monkeypatch. The edge code (loaders, run writer) is tested on its own with pytest's `tmp_path`. *(Considered and rejected: adding `DatasetRepository` and `TrainingRunStore` as ports 4 and 5. Passing data in makes them unnecessary.)*

### Target structure
```
pyproject.toml  requirements.lock  .env.example  .gitignore  .github/workflows/ci.yml
toolkit/
  __main__.py / cli.py          # argparse: menu | scan | ids | web | train
  config.py                     # pathlib paths, env/.env, per-OS defaults (ping flags, cpu count)
  logging_setup.py              # structured + rotating file logs, secret redaction
  ids_core/                     # ML CORE layer (no Flask, no print), not "domain" in the strict DDD sense
    ports.py                    # only 3 interfaces: ModelStore, IntelProvider, ProgressSink
    runs.py                     # TrainingRun record (D21)
    datasets.py profiles.py     # load, sniff sep, fix headers, known schemas (CIC/NSL-KDD/UNSW)
    leakage.py                  # dedupe, ID/label-column guard
    pipeline.py registry.py     # Pipeline builder; MODEL_REGISTRY, BALANCER_REGISTRY (plugins)
    evaluation.py explain.py    # CV, metrics, confusion, feature importance / SHAP
    bundle.py                   # signed joblib bundle + version manifest
  recon/
    scanner.py probes/          # one plugin per service (http.py, ssh.py, redis.py…)
    os_detect.py                # TTL ranges table
    intel/ nvd.py abuseipdb.py llm.py (redaction + consent)
    report.py export.py         # JSON/CSV/HTML/PDF, sanitised
  services/                     # APPLICATION layer: TrainingService, ScanService, PredictionService
  web/                          # PRESENTATION: create_app(), blueprints, jobs.py, workspace.py, security.py
    templates/ static/css/app.css static/js/{api,dom,upload,train,predict,charts}.js
  ids_cli/                      # PRESENTATION: CLI predictor using ids_core.bundle
tests/ unit/ integration/ fixtures/
(no separate training script: the CLI model is trained in the web trainer, see D12)
```

---

## 3. Layer-by-layer improvements (decisions D1–D23)

### 3.1 Presentation — Web UI
**D1. Rendering (XSS, maintainability).** ~15 `innerHTML` sites inject dataset text (`main.js:59,71,79,199,210,285,295,557,609,666,1003,1049`). The 1,061-line file uses `var` globals.
| Option | Pros | Cons |
|---|---|---|
| A. Escape before `innerHTML` | Quick | Easy to miss one spot |
| **B. DOM helper `el()` + `textContent`; ES modules (`api.js`, `dom.js`, `train.js`…)** | Safe by default; no build step; readable | Medium JS rewrite |
| C. React/Vue SPA | Rich components | Build toolchain; overkill |
| D. htmx + Jinja partials | Server-rendered, auto-escaped | Restructure every endpoint |
→ **B**, plus a CSP header.

**D2. Styling & devices.** `*:not(h1):not(h2){color:white!important}` (both CSS `:7-9`) makes light-mode text **white on white**. The dark-mode `@media` in `style2.css:375` is **never closed**, so lines 400–802 only apply in dark mode. Rules are duplicated and conflict. No viewport meta, so phones break. The 'Inter' font is never loaded; the fix is to **remove it and use the system font stack**, not to load a web font (that would break offline use and send a request to a third party). Multi-selects need Ctrl+Click, which doesn't work on touch screens.
| Option | Pros | Cons |
|---|---|---|
| A. Patch both files | Quick | Conflicts remain |
| **B. One `app.css` with CSS variables (light/dark tokens), viewport meta, checkbox lists instead of Ctrl-multi-select** | Readable in both themes, touch-friendly, maintainable | Re-test all sections |
| C. Pico/Bootstrap | Responsive for free | Restyle; extra asset |
→ **B.**

**Accessibility (applies to D2 + D3):**
- Text/background contrast ≥ 4.5:1 (WCAG AA) in both themes
- Every control reachable and usable by keyboard (Tab order, Enter/Space)
- **Visible focus outline** (never `outline: none` without a replacement)
- Every input has a `<label for>`
- `lang` on every page
- Status/error messages in an `aria-live` region, so screen readers announce progress and errors

v5.0 (0.7) covers contrast, `lang`, labels and focus outlines, because they are cheap. Keyboard flow and `aria-live` come with the stepper in v5.1.

**D3. UX flow.** Today it is one long page of hidden sections, with no busy state, no progress and no step indicator.
| Option | Pros | Cons |
|---|---|---|
| **A. Stepper wizard (Upload → Explore → Clean → Features → Train → Evaluate → Export) with disabled buttons + progress bar** | Guides the user; mirrors the pipeline | Template restructure |
| B. Keep a single page, add spinners | Small | Still confusing |
→ **A.**

### 3.2 Presentation — CLIs
**D4. Launcher.** `main.py:108-120` calls itself recursively; "n" exits the whole app (`:80-82`); a 5 s animation plays on every loop; the web IDS starts with no URL.
| Option | Pros | Cons |
|---|---|---|
| A. Patch | Small | Still interactive-only |
| **B. Flat loop menu + `argparse` subcommands (`toolkit scan 10.0.0.1 -p 1-1024`)** | Scriptable, testable, no recursion; auto-opens the browser | Small rewrite |
| C. Typer + Rich TUI | Pretty tables, colours | Extra dependencies |
→ **B** (Rich optional for tables).

**D5. CLI IDS input.** You have to type 8 numbers by hand, one of which is the label leak. The features and the 4 "sample traffic" presets are hard-coded for the v4 model (CLI `:353-394`).
→ The CLI becomes **model-agnostic**: it reads features, types, classes and the best model from the bundle manifest, so any bundle exported by the web trainer works. Presets become rows of the training data's test split stored in the bundle (real examples, not invented numbers). Add `--csv file` batch prediction, `--json` input, and `--bundle path`.

### 3.3 API layer (Flask)
**D6. API design.** RPC-style POST for reads (`/get_target_values`, `/get_confusion_matrix`). `req = request.get_json()` followed by `req.get(...)` crashes with a 500 on a non-JSON body (every endpoint). No validation. Errors leak `str(e)` and server paths. `/predict_loaded/` vs `/predict_loaded` mismatch.
| Option | Pros | Cons |
|---|---|---|
| A. Patch each route | Small | Inconsistent |
| **B. Versioned `/api/v1` REST blueprints, request schemas (pydantic), one error handler `{error:{code,message}}`, `MAX_CONTENT_LENGTH`, clamps** | Consistent, safe, self-documenting | Front end must follow the new routes |
| C. Move to FastAPI | Async + OpenAPI docs | Framework migration |
→ **B** (FastAPI later if needed).

### 3.4 Application layer (services, state, jobs)
**D7. State.** Global `data_store` (`app.py:241-247`) is shared by every tab/device and is never freed.
| Option | Pros | Cons |
|---|---|---|
| A. Globals + lock | Trivial | No isolation |
| **B. Per-session in-memory Workspace (lock + TTL cleanup)** | Isolation; simple | Lost on restart; RAM-bound |
| C. Disk workspace (Parquet + SQLite) | Survives restart; big data | More IO and code |
→ **B**, delivered in two steps:
- **v5.0 (unconditional):** a minimal workspace ID per browser session replaces the globals. Even on localhost, two tabs or two browsers otherwise overwrite each other's dataset and model.
- **v5.0 also adds a "Clear workspace" button (`DELETE /workspace`).** Isolation alone does not stop one long session from piling up DataFrames and models, so this lets a user free memory without restarting, which matters on weak devices.
- **v5.1:** the full Workspace with lock, TTL cleanup and memory limits.

**Session cookie hygiene (with S10, arrives with the workspace in 0.8):** `HttpOnly`, `SameSite=Strict`, signed with `SECRET_KEY` from `instance/`, no data in the cookie (only the workspace ID).

**D23. `Secure` cookie flag — explicit mechanism.** Flask does **not** set `Secure` by itself.
| Option | Pros | Cons |
|---|---|---|
| **A. Config flag: `SESSION_COOKIE_SECURE = TOOLKIT_HTTPS` (env, default off); if served behind a TLS proxy, also enable Werkzeug `ProxyFix`** | Explicit, testable, no magic | You must set the flag when enabling HTTPS |
| B. Decide per request from `request.is_secure` | Automatic | Needs a custom session interface; easy to get wrong behind proxies |
→ **A.** v5.0 runs on plain-http localhost, so `Secure` is off and this is documented. Tests cover both: flag off → no `Secure`; `TOOLKIT_HTTPS=1` → `Secure` present.

**CSRF in v5.0:** the full CSRF token arrives in v5.1 (1.2). Until then, `SameSite=Strict` cookies plus JSON-only state-changing routes (requests must send `Content-Type: application/json`, which a plain HTML form on another site can't do without a CORS preflight) cover the realistic local risk. Multipart upload routes additionally check the `Origin` header.

**D22. Workspace TTL vs running jobs (decided now, built in v5.1).** If a workspace expired while a job was still training, the job would write into a dead workspace.
| Option | Pros | Cons |
|---|---|---|
| **A. A running job holds a lease on its workspace; TTL cleanup skips leased workspaces; the lease ends when the job finishes or is cancelled** | Simple, no lost work | A stuck job could keep memory; mitigated by the job's own timeout |
| B. Cancel jobs when their workspace expires | Frees memory fast | Loses long trainings if the user steps away |
→ **A**, plus a maximum job duration (configurable) after which the job is cancelled and the lease released.

Move to **C** when history/persistence features arrive (§5).

**D8. Long tasks.** Training blocks the request; threads × `n_jobs=-1` overload the CPU (`app.py:1293-1295,1363-1374`); progress only goes to the log.
| Option | Pros | Cons |
|---|---|---|
| A. Synchronous | — | Timeouts, frozen UI |
| **B. In-process JobRunner + `/api/v1/jobs/<id>` polling (progress events)** | No extra services; any OS | One process |
| C. Celery/RQ + Redis | Scales | Extra services; weak on Windows |
| D. SSE push | Live | Needs B anyway |
→ **B** (SSE optional). Train models one after another, each with `n_jobs = TOOLKIT_MAX_WORKERS` (default `max(1, os.cpu_count() - 1)`, so one core stays free and the laptop stays usable); add a cancel button. The same setting caps scanner threads and CV parallelism.

**Runaway-request protection** (instead of a generic rate limiter, which adds little for a local single-user app):
| Option | Pros | Cons |
|---|---|---|
| A. Generic rate limiter (flask-limiter) on every route | Standard | Needs tuning; doesn't stop the real problem (several heavy jobs at once) |
| **B. One active training job per workspace (second request → `409 Busy`) + existing size/row limits + light limit on `/predict` only** | Targets the actual risk: a runaway JS loop or double-click starting many trainings | Not a defence against a determined attacker (out of scope locally) |
→ **B.** In v5.0 the training route is still synchronous, so a per-workspace "training in progress" flag (set before training, cleared in `finally`) gives the same `409` until the job runner arrives in v5.1. In real use, waitress runs 4 threads by default, so two requests really can overlap and the flag matters.
**How it is tested:** the Flask test client sends requests one after another, so a naive "send two requests" test passes even if the flag is broken. The test instead injects a hook into training that blocks on a `threading.Event`. While the first request is held inside training (in a background thread), a second request must get `409`; then the event is released and the first finishes normally.

### 3.5 ML core layer
**D9. Pipeline & leakage.** See L1–L10 in §4.1.
| Option | Pros | Cons |
|---|---|---|
| A. Reorder steps manually | Small | Leakage comes back easily |
| **B. `imblearn` Pipeline fitted on train only; dedupe + profile-defined split first; leakage guard** | Leakage-proof; one object to save | The UI "preview SMOTE" becomes train-only |
| C. AutoML | Tuning for free | Black box, heavy |
→ **B.**

**Split strategy is set per dataset profile, not a generic default.** Removing exact duplicates is not enough: near-identical flows from the same attack session can still land on both sides of a random row split and inflate scores.
| `split_strategy` | When | Example |
|---|---|---|
| `official` | The dataset ships its own train/test files | NSL-KDD (`KDDTrain+` / `KDDTest+`), UNSW-NB15 training/testing sets |
| `temporal` | A **validated** time column exists | CIC-IDS2017/2018 `Timestamp`: train on earlier traffic, test on later |
| `temporal_per_class` | Time is valid, but each attack type happens in only one time window | CIC-IDS2017: within each class, the earliest part goes to train and the latest to test |
| `group` | Rows belong to sessions/hosts | Group by flow/session, capture file or host id so one group never spans the split |
| `stratified` | Fallback for unknown datasets only | Custom CSV with no time/group/official split, shown with a warning that scores may be optimistic |
Each profile declares `split_strategy` plus its `time_column` / `group_column`. The user can override it, and the override is recorded in the TrainingRun. Temporal/group splits can leave a rare class out of the test set, so the split step checks class coverage and warns about it.

**A profile's strategy is a recommendation with prerequisites, not an unquestioned rule.** Before a temporal strategy is used, the split step validates the time column:
1. It parses reliably; the unparseable share stays under a threshold, and day-first vs month-first ambiguity is resolved.
2. It is complete enough (few missing values).
3. It is consistent across the uploaded files: no clock resets, overlaps or time zone jumps between capture files.
4. It gives class coverage: every class appears in both train and test. **Important for CIC-IDS2017:** each attack type was captured on a single day (e.g. DDoS only on Friday), so a plain global time cut puts some classes **only** in test. The profile therefore defaults to `temporal_per_class`.

**Not every CIC file has a time column.** The original CIC-IDS2017 "MachineLearningCSV" files (and the DDoS file in `uploads/`, whose header was checked) contain only flow features + `Label`, with no `Timestamp`, IPs or `Flow ID`. For those files a temporal split is impossible. `group` by capture file is also unusable, because each file is one day and holds whole attack classes, so a class would end up only in train or only in test. The CIC profile therefore detects which variant it has. With time → `temporal_per_class`; without time → `stratified` after dedupe, **with a visible warning** that scores may be optimistic and that this limitation is written into the TrainingRun and the model card. This is one more reason the improved CIC-IDS2017 version (which has timestamps) is the preferred dataset.

If any check fails, the profile falls back to its documented next strategy (`group` by capture file **only if every class still appears on both sides**, then `stratified` with a warning). It never silently produces a strange split. The checks, the result and the fallback taken are recorded in the TrainingRun as **enumerated reason codes** (not free text), e.g. `TIME_PARSE_FAILURE_RATIO_EXCEEDED`, `TIME_MISSING_RATIO_EXCEEDED`, `TIME_NON_MONOTONIC_ACROSS_FILES`, `CLASS_MISSING_FROM_TEST`. Tests assert the exact code.

*Implemented in step 0.4b (`toolkit/ids_core/split.py`; details in `docs/V5.0_CHECKLIST.md`). Decisions made there: dedupe ignores identifier columns (IDs, time, IPs), so copies that differ only in their ID are removed; a temporal cut never separates rows with equal timestamps; an ambiguous `d/m` vs `m/d` date is resolved from the values or the file order, and accepted when both readings give the same order; an `official` split keeps both published files unchanged by default, so scores stay comparable with other work; test rows that also occur in train (610 on NSL-KDD) are counted and reported, and removed only in strict mode (`SplitConfig.official_dedupe`), which the TrainingRun records. Clock order inside one file isn't checked yet (only overlaps between files), because no CIC file with timestamps is available locally.*

**Profile detection is content-based:** a profile is chosen from the header set and value patterns, never from the filename. The same content gives the same profile even after renaming the file, shuffling rows, or changing line endings/encoding. CIC-IDS2017 has several published variants with different headers (the original "MachineLearningCSV" files with leading spaces and `Fwd Header Length.1`, the "improved" re-labelled version, Kaggle repacks with or without `Flow ID`). The CIC profile must match **at least two** of them, each covered by a small header fixture.

**Leakage guard — exact single-feature rule** (catches a label hidden under an unfamiliar name):
For each candidate feature, on the **training split only**, fit a depth-limited decision tree (`max_depth=3`) on that one feature with 3-fold stratified CV and compute balanced accuracy. Also measure how many distinct values the feature takes **within each class**.
| Result | Action |
|---|---|
| ≥ **0.99** **and** the feature behaves like a relabelling (≤ 2 distinct values per class on average, so each class maps to almost one value) | **Block**: very likely the label or a trivial transform of it (renamed copy, encoded, prefix/suffix) |
| ≥ 0.95 otherwise (including a near-perfect *continuous* feature) | **Warn**: may be a legitimately strong feature. On easy binary data such as the CIC DDoS file, single packet-size features can genuinely separate classes almost perfectly, so these must not be blocked |
| < 0.95 | OK |

*(Self-review fix: an accuracy threshold alone would wrongly block real, strong features on easy datasets. The "one value per class" condition is what separates a disguised label from a good feature.)*

**Name patterns** are checked independently, as **whole words / tokens only**, case-insensitive, after normalising separators. Substring matching would block real features: `id` would hit **`Idle Mean`/`Idle Max`** in CIC. Patterns: `label`, `attack`, `attack_cat`, `level`, `attempted`, `class`, `ip`, `src ip`, `dst ip`, `timestamp`, `flow id`, `flowid`, `id`, `row`, `index`, `unnamed: N`. Duplicate columns such as `Fwd Header Length.1` are removed as duplicates (the current loader already drops `.1` columns, `app.py:328`; the new loader keeps this and records it).

**ID-like column check:** flag a column only if it is **integer or text** and is (a) ≥ 99% unique **or** (b) strictly increasing with row order. Continuous floats are never flagged by uniqueness alone, because real features like `Flow Duration` and `Flow Bytes/s` are also almost all unique.

*Implemented in step 0.4a with three refinements found by tests and the real-data check (details in `docs/V5.0_CHECKLIST.md`): the tree is limited by **leaves** (8, the capacity of a depth-3 tree, or 2 per class when there are more classes), because a greedy depth-3 tree misses an exact copy of a label with more than 8 classes; the name pattern is `attempted category`, not `attempted`, which would block the legitimate NSL-KDD feature `su_attempted`; the "≥ 99 % unique" rule needs at least 1,000 rows, since on small samples most integer features look unique.*

**Rejected suggestion:** "flag columns constant in train but not in test." Choosing features by looking at the test set would break the test-isolation invariant below. Drift between train and test may only be **reported** after the test set has been scored, never used to select features.

The user can **override** a block or warning via an allowlist, and every override (feature, score, reason) is recorded in the TrainingRun. The user can **override** a block or warning via an allowlist, and every override (feature, score, reason) is recorded in the TrainingRun. Thresholds live in `config.py` so tests can rely on them. On very large datasets the check runs on a stratified sample of the training split (size recorded).

**Test-set isolation invariant:** the test set must never influence feature selection, balancing choice (including the recommender, which today uses full-data class counts at `app.py:486-537`), hyperparameter tuning, threshold/calibration choice, or model selection. All of these use train/CV/validation data only. The test set is scored **once**, at the end.

**D10. ML quality (beyond leakage).**
| Gap today | Improvement | Pros | Cons |
|---|---|---|---|
| One split, no CV | Stratified K-fold CV on train, mean ± std | Stable estimates | Slower → limit folds on big data |
| Default hyperparameters | Optional `RandomizedSearchCV`/Optuna with time budget | Better models | Training time |
| `LabelEncoder` on nominal features fed to LR/KNN/SVM (treated as an ordering) | `OneHotEncoder` for linear/distance models, `OrdinalEncoder` for trees, `handle_unknown` | Correct maths; no predict crash | Wider matrices |
| `f_classif` only (linear assumption) | Add mutual information + model-based importance | Catches non-linear features | Slower |
| Accuracy / weighted F1 | Macro-F1, balanced accuracy, per-class P/R, PR-AUC, FPR for BENIGN | Honest for IDS | More UI |
| Model chosen by test score (`main.js:870`) | Choose on CV/validation, test reported once | No selection bias | — |
| No explainability | Feature importance + SHAP (optional) per model | Trust; good for the thesis | SHAP dependency |
| Raw `predict` | Probability + confidence + calibrated threshold | Tune false alarms | Calibration cost |
| No model card | Bundle stores dataset hash, features, metrics, versions, date | Reproducible | — |
| Hard-coded thresholds in `recommend_balancing_method` | Keep the heuristic but validate it by quick CV of 2–3 balancers | Evidence-based | Time |

**D11. Serialization.** `pickle.load` of uploads (`app.py:1645`) can run any code. The CLI unpickles with warnings hidden. The models are tied to sklearn 1.6.1/numpy 2.
| Option | Pros | Cons |
|---|---|---|
| A. Pickle | — | Code execution; hidden version breakage |
| **B. joblib + HMAC-SHA256 signature + version manifest** | Rejects tampered/foreign files; works with XGBoost/LightGBM | Bundles are tied to the install key (exportable) |
| C. skops.io | Safe for files from anyone | Limited XGB/LGBM support |
| D. ONNX | Safe, fast, portable | Conversion gaps |
→ **B** (C as an optional import path).

**Security rule:** signing does not make deserialization safe. It only proves who made the file. The rule is: **never deserialize a bundle unless its signature or pinned hash checks out first.** Everything else is rejected before `joblib.load` runs.
| Case | How it is trusted |
|---|---|
| Bundle made on this machine | HMAC-SHA256 with the per-install key in `instance/` (never in git) |
| Model shipped with the project (CLI default) | Can't use a per-install key → **SHA-256 hash pinned** in the repo manifest, checked before loading. *Not used since 0.5: no model is shipped (D12 revision); the library keeps the option* |
| Anything else (unsigned, tampered, wrong hash, made on another device) | Rejected with a clear message; never loaded |

**v5.0 policy (narrow):** only the two trusted sources above are loaded, nothing else. There is deliberately **no "trust this file" button**: confirming a hash only proves you picked that exact file, not that the pickle inside is safe.

**Later (v5.1+), cross-device import:**
| Option | Pros | Cons |
|---|---|---|
| **Opt-in export/import of your own signing key** between your devices | Same trust level as a local bundle (you are the signer) | Key handling; a leaked key means forged bundles |
| `skops` safe format for files from anyone | Built for untrusted files: never runs code while loading | Limited XGBoost/LightGBM support |
| ONNX for inference-only sharing | Safe, portable | Conversion gaps |
→ Key import for your own devices; `skops`/ONNX for third-party models.

Threat model note: this protects against files from **other people** (uploads, downloads). Someone who can already write to your disk can also replace the key or the code, and that is out of scope.

**Signing-key loss & backup policy:**
- The key is created once in `instance/signing.key`. **If it is lost, every locally exported bundle becomes unloadable.** The UI and README say this plainly.
- v5.0: bundles are documented as tied to this install; the README explains how to back up `instance/` (copy the folder somewhere safe). The TrainingRun is also kept as plain JSON, so a lost bundle can be **retrained** from its record.
- v5.1: the opt-in key export/import above doubles as the backup tool.
- `.gitignore` must contain `instance/`; a test asserts `git check-ignore instance/signing.key` succeeds, so the key can never be committed by accident.

**Library-version mismatch policy (B18):** sklearn often loads an old pickle with only a warning and can then predict wrongly, which is the dangerous case. The bundle manifest stores exact versions, and the loader compares them **before** deserializing:
| Difference vs manifest | Action |
|---|---|
| Same version | Load |
| **Estimator libraries** (sklearn, imblearn, xgboost, lightgbm): patch only (e.g. 1.6.1 → 1.6.3) | Load + visible warning |
| **Estimator libraries**: major or minor (e.g. 1.6 → 1.8) | **Refuse**, with a clear message: "made with scikit-learn 1.6.1, you have 1.8.0. Retrain from its TrainingRun or install 1.6.x" |
| **numpy**: minor/patch | Load + warning (numpy minor releases rarely break pickles; refusing would be needlessly strict) |
| **numpy**: major (1.x ↔ 2.x) | **Refuse** (pickle layout changed: `numpy._core` vs `numpy.core`) |

**v4 interim check (0.3):** the v4 pickles have no manifest and `config.json` has no version field. The versions were read from inside the pickles without running them (sklearn 1.6.1, numpy 2.x), so 0.3 writes them into a small pinned manifest next to the files (with their SHA-256). The CLI compares those against the installed versions before loading v4 and warns loudly (v4 is being replaced anyway). Removing the global `filterwarnings("ignore")` (B19) is a prerequisite: today it hides exactly the warning this policy relies on.
No bundle is shipped (D12 revision, 0.5): every bundle is made on the install that loads it, so versions only differ after a library upgrade.

**D12. CLI model** (v4 has the label leak `Attempted Category`).
| Option | Pros | Cons |
|---|---|---|
| A. Keep v4 with a warning | No work | Invalid accuracy claims |
| B. Separate training script/notebook | Headless, scriptable | A second training code path that can drift from the web trainer; the referenced notebook doesn't exist |
| **C. Train in the fixed web trainer → export a signed bundle → CLI loads it** | **One** training path (the leakage-proof pipeline, D9); the CLI and web give identical results; TrainingRun (D21) records how the model was made; you can retrain any time from the UI | Depends on the web trainer being fixed first (v5.0); the dataset must be provided by you |
→ **C (your choice).** Steps:
1. Fix the web trainer pipeline (D9, D10) and the bundle format (D11).
2. Make the CLI read any bundle (D5).
3. Train with the leakage guard **on**, so `Attempted Category`, IDs and timestamps are blocked.
4. Export the bundle and pin its hash as the CLI default.
5. Delete the v4 `.pkl` files **and the v4 `config.json`** (it still lists `Attempted Category`, so stale metadata can't be read by mistake), and update `models/README.md` with the real metrics.
6. Commit the default bundle's **TrainingRun JSON** next to it in the repo (`default.run.json`). `instance/runs/` is gitignored, and without this the shipped model's provenance would exist only on your machine.

**Revised in 0.5 (your decision, 2026-10-20): no model is shipped; steps 4 and 6 are dropped.** A Random Forest trained on the full CIC file exports to about 2 GB, far over GitHub's 100 MB file limit, and a size cap would decide which models are allowed. The CLI lists the bundles exported on this install (newest first, reading only each manifest header) and loads the one you pick; `--list`, and `--csv` / `--json` / `--info` use the newest unless `--bundle` is given. Each bundle embeds its TrainingRun, so provenance travels with the model. Cost: a fresh install has no CLI model until one is trained and exported in the web trainer.

**Dataset note:** the CSV in `uploads/` only covers BENIGN vs DDoS, so a model trained on it detects DDoS only. For the 16-class CLI model you need the improved CIC-IDS2017 files (Kaggle). The web trainer must handle large uploads (D16 chunked loading). Optional later: `toolkit train --from-run run.json` re-runs a saved TrainingRun headlessly through the **same** pipeline code, which keeps option B's benefit without a second code path.

**Dataset status (2026-10-19, details in `docs/datasets.md`):** `data/raw/CIC-2017.csv` is the **original** CIC-IDS2017 "MachineLearningCSV" variant: 2,830,743 rows, **15 classes**, 10.9 % exact duplicates (PortScan 43 %, SSH-Patator 45 %), `inf`/empty cells, and **no time, IP or Flow ID columns**. So:
- a CIC CLI model is possible now, with 15 classes rather than v4's 16 "improved" classes, and the leaky `Attempted Category` doesn't exist in this file;
- its split must be **dedupe → stratified with the optimistic-score warning** (no temporal or group split is possible on this file);
- Heartbleed (11), Sql Injection (21) and Infiltration (36) are too small to score reliably.

The beta-exit decision is now about accepting that caveat, not about availability.

**Decision point at `5.0-beta` exit (not rc2):** at the end of beta, check whether CIC-IDS2017 is downloaded, hashed and passes profile detection. If not, **commit to the NSL-KDD fallback right then**, so 0.5 can start during rc1 instead of hitting a wall at rc2.

**Fallback if CIC-IDS2017 isn't available at the beta decision point:** ship the default CLI bundle trained on **NSL-KDD** (`KDDTrain+` / `KDDTest+`, already on disk). It is multi-class (normal + DoS / Probe / R2L / U2R and their sub-types) and comes with an **official train/test split**, which is much better than a two-class DDoS model. It ships with an honest note: NSL-KDD comes from 1999-era traffic (KDD'99 lineage), so it shows the pipeline works but is not representative of modern traffic. The CIC model replaces it in a later release once the data is available. This way one unavailable download never blocks v5.0.

**D21. Reproducibility — `TrainingRun` record (new).**
Problem: today nothing records how a model was produced, so a result like "94%" can't be proven or repeated.
Every training job writes one immutable `TrainingRun` JSON:
`run id → dataset SHA-256 + profile → dedupe stats → split strategy (official/temporal/temporal_per_class/group/stratified) + validation reason codes + time/group column + seed + sizes + class coverage → preprocessing steps → leakage-guard decisions → selected features → sampler + params → model + hyperparameters → CV mean±std → validation/test metrics → package versions (python, sklearn, numpy, xgboost, imblearn) → bundle SHA-256 → timestamp`.
| Option | Pros | Cons |
|---|---|---|
| A. Only metrics inside the bundle | Simple | Can't reproduce or compare runs |
| **B. `TrainingRun` JSON per run, embedded in the bundle and saved under `instance/runs/`** | One source of truth: model card, bundle manifest, history list (F6) and the thesis tables are all built from it; "re-run" becomes possible | A small schema to maintain |
| C. MLflow tracking | Full UI, comparisons | Heavy dependency/server for a local tool |
→ **B** (MLflow-compatible export later if wanted). All random seeds are fixed, **including nested estimators**. Today SMOTE-ENN and SMOTE+Tomek pass an inner `SMOTE(k_neighbors=…)` with no seed (`app.py:1108`, `:1138`), so those runs most likely can't be repeated exactly (B20). Note that `n_jobs>1` can still cause tiny differences, so the record states its tolerance.

**Schema evolution:** 0.4d produces an `EvaluationResult` object (CV mean ± std, per-class metrics, timings, the leaderboard rows). 0.4e's `TrainingRun` contains it unchanged as a field, so the leaderboard doesn't need a throwaway structure. The TrainingRun carries a `schema_version`, and readers accept older versions.

### 3.6 Recon / scanner core
**D13. Engine.** Sockets leak on errors (`Port_Scan.py:200-448`), double connect (`:450-460`), lookups run one at a time (`:733-737`), IPv4 only, a 245-line `if/elif` of probes.
| Option | Pros | Cons |
|---|---|---|
| **A. Threads kept; `with` sockets; reuse the connection; probe plugins (dict port→handler); parallel intel** | Safe, small, extensible | Thread cap |
| B. asyncio rewrite | Thousands of concurrent probes | Larger rewrite |
| C. Wrap nmap/scapy | Mature | External binary / admin rights |
→ **A**, B as a stretch goal.

**D14. Correctness.** TTL exact match (`:505-532`) → use a range table; ping timeout per OS (`:486-497`); UDP open\|filtered (`:464-480`); TLS without verification for banners only (`:206-209`); guard against an empty range (`:715,586`); validate ports and cap host count.

**D15. Intel & privacy.** The MITRE scraper is retired (`:104`). Raw banners go to a **free third-party LLM** (`:61-98`) and can carry prompt injection.
| Option | Pros | Cons |
|---|---|---|
| **A. NVD CVE API 2.0 + LLM only after consent, redacted `product/version`, configurable model, LLM output treated as untrusted** | Accurate; minimal data leaves the device | NVD key for rate limits |
| B. Local LLM (Ollama) | Zero leak | Needs a strong device |
| C. Drop the LLM | Zero leak | Loses a feature |
→ **A** (B optional).

**Prompt-injection policy (S13).** Delimiting the prompt reduces the risk but is **not** a mitigation on its own. The real rule: **the LLM never drives a decision.**
1. **Facts come from deterministic sources only.** CVE IDs, CVSS scores and severity come from NVD, computed in code. The LLM is never asked for a severity, and if it gives one, it is ignored.
2. **Input:** send only the parsed, redacted `product` + `version` (allowlisted characters, length-capped), never the raw banner.
3. **Output is untrusted data.** It is shown only as plain text in a clearly labelled "AI explanation (may be wrong)" section, with control characters stripped, length-capped and never rendered as HTML. Nothing is parsed out of it that changes program behaviour (no commands, URLs to fetch, file paths or scores).
4. If structured fields are ever needed, the output must validate against a strict schema with an allowlist of fields. Anything else is discarded.

### 3.7 Data / infrastructure layer
**D16. Dataset handling.** `sep=None` python engine is slow (`app.py:296-299`); dask parses differently (`:294`); headers keep leading spaces; NSL-KDD test has no header; same-name uploads reuse the old file (`:284-287`); unsafe filenames (`:283`); uploads are never deleted.
→ Dataset **profiles** plugin (auto-detect CIC / NSL-KDD / UNSW and apply known headers; each profile also declares the label column, columns to block as IDs/label-derived, and its `split_strategy` + time/group column, see D9); `csv.Sniffer` + C/pyarrow engine; strip headers; dtype downcast; UUID filenames; Parquet cache; TTL cleanup. Dask becomes optional.

**D17. Config, secrets, logging.**
| Concern | Options | Recommendation |
|---|---|---|
| Secrets | txt file / **env + .env** / OS keyring | env + `.env`, txt fallback |
| Config | scattered constants / **`config.py` + env overrides** | `config.py` |
| Logging | `print` + `basicConfig(INFO)` / **structured logs, rotating file in `instance/logs`, secret redaction, no dataset values** | structured + rotating |
| Warnings | global `filterwarnings("ignore")` (`app.py:50`, CLI `:14-15`) / **show version warnings once** | remove the global ignore |

### 3.8 Security (cross-cutting) — see §4.2
Signed bundles (S1), safe filenames (S2–S4), safe DOM (S5), debug off + `waitress` bound to 127.0.0.1 (S6), workspaces (S7), error hygiene (S8), limits (S9), CSRF header token + CSP/X-Frame/nosniff (S10), upload TTL (S11), LLM privacy (S12–S13), output sanitising (S14), env secrets (S15), scan limits + authorization notice (S16).

### 3.9 Performance (cross-cutting) — see §4.3
float32/category dtypes, Parquet cache, one model at a time each using `TOOLKIT_MAX_WORKERS` (all cores but one), connection reuse, parallel intel lookups, lazy loading of the 18 MB RF model, preview limits.

### 3.10 DevOps, packaging, cross-device
**D18. Packaging.**
| Option | Pros | Cons |
|---|---|---|
| A. Pin `requirements.txt` | Simple | Partial |
| **B. `pyproject.toml` + lock + extras (`[boost]`, `[explain]`, `[dev]`) + `waitress`** | `pip install .` on Win/Linux/macOS | Small setup |
| C. Docker | Identical everywhere | Scanning/ping awkward in containers |
| D. PyInstaller | No Python needed | Large; antivirus flags for scanners |
→ **B** (Docker optional for the web IDS).
Also: `pathlib`, UTF-8 everywhere, per-OS ping flags, no spaces in folder names, numpy ≥2 / sklearn pinned to match the bundles, Python ≥ 3.11 with development on your current 3.14 (policy under D19), `TOOLKIT_MAX_WORKERS` for weak devices, and the portability rules in §7 (CI host).

**D19. Quality gates.** `git` + `pytest` (unit, Flask client, leakage, scanner with a local server) + `ruff` + `mypy` (gradual, see §6.1) + `pre-commit` + **`pip-audit`** (fails the run on known-vulnerable dependencies; a network-security tool must not ship with them) , all run **locally on Windows with plain `python -m pytest`** (no nox, no GitHub). Linux and macOS are supported by following the portability rules (§7 CI host), with OS-specific logic unit-tested by patching the platform. Run `pip-audit` before every checkpoint tag.

**Python version policy (resolves the 3.14 vs CI mismatch).** Your machine runs Python 3.14.3 while the ML ecosystem often lags new Python releases.
| Option | Pros | Cons |
|---|---|---|
| A. Supported = 3.11–3.13, develop in a 3.13 venv, 3.14 experimental | Matches what most other devices have | You install extra Python versions + a multi-version test tool |
| **B. Develop and test on your current Python 3.14; declare `requires-python = ">=3.11"`; write code that avoids 3.12+/3.14-only features** | No extra installs; everything already runs on this machine; older Pythons should still work | Only 3.14 is actually tested; an optional future dependency without 3.14 wheels would need a check before adding it |
| C. 3.14 only (`>=3.14`) | Simplest | Needlessly blocks other devices with older Python |
→ **B (your choice: "as we are now").** `ruff` is configured with `target-version = "py311"` so it flags syntax newer than 3.11, which keeps the code runnable on older Pythons without installing them. New dependencies are checked for Windows/Linux/macOS wheels on 3.11+ before being added. *(Replaces the earlier 3.13-venv recommendation.)*

**D20. Documentation.** README rewrite (honest metrics, what data leaves the device, setup per OS, signing-key backup), architecture doc with diagrams, model card, API reference, user guide with screenshots, `CHANGELOG`, add a `LICENSE`, and `docs/datasets.md` with per-dataset licence and redistribution terms (0.5b). Day-to-day work runs from **`docs/V5.0_CHECKLIST.md`**; this plan stays the reference for *why*.

---

## 4. Problem catalogue (every finding)

### 4.1 ML data leakage
| ID | Sev | Location | Problem |
|---|---|---|---|
| L1 | Crit | `config.json:23`, CLI `:355` | `Attempted Category` comes from the label; `notebook_v5_fixed.ipynb` doesn't exist |
| L2 | Crit | `app.py:550-877`, `:1016-1018` | Full-data SMOTE is used for training → synthetic test rows in train/test |
| L3 | High | `app.py:879-948` | SelectKBest on the full/SMOTE data |
| L4 | High | data | 23,416 duplicate rows end up on both sides of the split |
| L5 | High | `app.py:1072-1073` | No stratify; random split of time-ordered flows |
| L6 | High | datasets | `level`, `attack_cat`, IP/Timestamp columns are not blocked |
| L7 | Med | `app.py:1058-1062` | Encoders fitted before the split; unseen categories crash |
| L8 | Med | `main.js:870` | Model chosen by test accuracy |
| L9 | Med | `app.py:1304-1311` | Accuracy/weighted F1 hide minority classes |
| L10 | Med | `KDDTest+.txt` | No header; merging the official train/test files |

### 4.2 Security & privacy
| ID | Sev | Location | Problem |
|---|---|---|---|
| S1 | Crit | `app.py:1645` | Uploaded pickle → arbitrary code execution |
| S2 | Crit | `app.py:283` | Upload path traversal |
| S3 | High | `app.py:284-287` | Same name → stale file used |
| S4 | Crit | `app.py:1489-1495` | Export path traversal + **silent overwrite** (fix: refuse names outside the allowlist; never overwrite, append `-1`, `-2` …) |
| S5 | High | `main.js` (~15 sites) | Stored XSS from CSV |
| S6 | High | `app.py:1724` | `debug=True` |
| S7 | High | `app.py:241-247` | Shared global state across devices |
| S8 | Med | `str(e)`, `app.py:1513-1515` | Error/path disclosure |
| S9 | Med | `app.py:353,883` | Unbounded inputs; no upload limit; non-JSON body → 500 |
| S10 | Med | app-wide | No CSRF check, security headers, or cookie hygiene (`HttpOnly`, `SameSite`) |
| S11 | Med | `uploads/` | Data kept forever |
| S12 | High | `Port_Scan.py:61-98,737` | Raw banners sent to a 3rd-party LLM |
| S13 | High | `Port_Scan.py:67-73` | Prompt injection |
| S14 | Med | `Port_Scan.py:742-747,603-616` | Terminal escape-sequence injection; CSV formula injection |
| S15 | Med | `Port_Scan.py:30-48` | Plaintext keys |
| S16 | **Med** | `Port_Scan.py:641-672` | No scan limits (host/port caps) and no authorization notice. Raised from Low: the tool will be demoed, so scanning without consent is a legal risk. v5.0 requires an explicit confirmation (interactive prompt, or an `--i-am-authorized` flag in scripts) before any scan |
| S17 | Med | CLI `:14-15,82-104` | Unpickling with warnings hidden |

### 4.3 Resource leaks & performance
| ID | Location | Problem |
|---|---|---|
| R1 | `Port_Scan.py:200-448` | Sockets not closed on exceptions |
| R2 | `Port_Scan.py:450-460` | Double connect per open port |
| R3 | `Port_Scan.py:733-737` | Lookups run one at a time |
| R4 | `app.py:241-247,847` | DataFrames and models stay in memory for good |
| R5 | `app.py:1293-1295,1363-1374` | CPU overload |
| R6 | `app.py:976-1450` | Training blocks the request |
| R7 | `app.py:294-299` | Slow parser; inconsistent dask path |
| R8 | `main.py:108-120` | Menu recursion |
| R9 | `main.py:28-34` | Repeated animation |

### 4.4 Functional bugs
| ID | Location | Problem |
|---|---|---|
| B1 | `app.py:1358-1359,1381` | Failed model still saved/used/exported unfitted |
| B2 | `app.py:1617-1626` | Loaded models not scaled at predict time (skew) |
| B3 | `main.js:1037` / `app.py:1571` | Trailing-slash mismatch |
| B4 | `main.js:2,1030` | `loaded_model_config` never filled. Harmless dead code: the server already converts values to numbers itself (`app.py:1590-1594`); remove it in 0.4g |
| B5 | `app.py:1201` | SVM size check uses `len(df)` |
| B6 | `Port_Scan.py:505-532` | Exact TTL match |
| B7 | `Port_Scan.py:486-497` | Ping timeouts wrong on Windows / missing |
| B8 | `Port_Scan.py:464-480` | UDP "closed" should be open\|filtered |
| B9 | `Port_Scan.py:206-209` | Self-signed TLS → banner lost |
| B10 | `Port_Scan.py:104` | Retired MITRE endpoint |
| B11 | `Port_Scan.py:715,586` | Empty range → crash / division by zero |
| B12 | `Port_Scan.py:597,605,623` | Non-UTF-8 export crash; written to the current folder; overwrite |
| B13 | `Port_Scan.py:183-189,258` | IPv4 only |
| B14 | CLI `:299-301,425` | Ignores `best_model`; bare `except`; no shape check |
| B15 | `main.py:80-82` | "n" exits the whole app |
| B16 | `main.py:65` | No URL/browser; reloader starts a second process |
| B17 | data | Header names keep leading spaces |
| B18 | models | Pickles need sklearn 1.6.1/numpy ≥2 (this machine has 1.8.0; numpy 1.x devices can't load them) |
| B19 | `app.py:50`, CLI `:14-15` | Global `warnings.filterwarnings("ignore")` hides sklearn's version-mismatch warning, the exact signal B18's policy depends on. **Removed in 0.3** |
| B20 | `app.py:1108`, `:1138` | SMOTE-ENN / SMOTE+Tomek create an inner `SMOTE` without `random_state`, so most likely their results can't be reproduced exactly |
| B21 | `…/models/config.json` | v4 metadata still lists `Attempted Category`; must be deleted together with the v4 pickles (0.5) |

### 4.5 UI
| ID | Location | Problem |
|---|---|---|
| U1 | both CSS `:7-9` | White-on-white text in light mode |
| U2 | `style2.css:375-802` | Unclosed `@media`. Per the CSS spec the block is auto-closed at the end of the file in every browser, so lines 400–802 apply **only in dark mode**. Fixing it changes the light-mode layout of ~400 lines → take **before/after screenshots of every section** (manual visual check) |
| U3 | both CSS | Duplicate rules; `Inter` font named but never loaded → replace with the system font stack (no web-font download) |
| U4 | templates | No viewport meta / `lang`; `home.html:47` loads unused `main.js` |
| U5 | `dynamic_ids.html` | No progress/busy states; Ctrl-click multi-selects |

### 4.6 Hygiene & docs
`__pycache__` folders present; 42 MB in `uploads/`; `.gitignore` gaps (`uploads/*.txt`, exported models, `instance/`); no git; jupyter listed in runtime dependencies; README duplicate lines (`:458-459`), placeholder URL, missing LICENSE, inflated metrics, no third-party data disclosure; `models/README.md` refers to a missing notebook; `dataset/` is empty.

---

## 5. New features (options with pros/cons)
| # | Feature | Pros | Cons / cost | Priority |
|---|---|---|---|---|
| F1 | **Batch prediction**: upload a CSV → labelled CSV + summary chart (web + `toolkit ids --csv`) | Realistic IDS use; easy | Needs the bundle schema check | High |
| F2 | **Model leaderboard & comparison** (CV mean±std, macro-F1, per-class recall, training time) | Shows ML rigour | UI work | High |
| F3 | **Explainability panel** (feature importance, SHAP for a single prediction) | Trust; strong thesis material | SHAP dependency/time | High |
| F4 | **Dataset profiles** (auto-detect CIC-2017/2018, NSL-KDD, UNSW-NB15; apply headers, drop ID/label columns) | Fixes L6/L10/B17; one-click use | Maintain schemas | High |
| F5 | **Scan reports**: HTML/PDF report with open ports, CVEs (CVSS), recommendations | Professional output | PDF library | Medium |
| F6 | **Scan & training history** (SQLite): list, re-open, compare runs | Persistence; trends; training history mostly comes free from TrainingRun (D21) | Storage layer (D7-C) | Medium |
| F7 | **Recon → IDS integration**: scan results feed a "risk dashboard"; suggest which IDS model to deploy | Ties the tools into one product | Design effort | Medium |
| F8 | **Live / PCAP detection**: read a `.pcap` or live interface → flow features (`nfstream`/CICFlowMeter-style) → real-time predictions | Turns it into a real IDS | Admin rights, Npcap on Windows, feature parity with the training data | Medium (stretch) |
| F9 | **Hyperparameter tuning UI** with time budget | Better models | Long jobs (needs D8) | Medium |
| F10 | **Dashboard home page** (recent jobs, models, scans, charts) | Polished UX | Front-end work | Low |
| F11 | **Scheduled scans** + diff alerts ("new port opened") | Monitoring value | Scheduler; OS differences | Low |
| F12 | **Localisation (English/Arabic, RTL)** | Wider audience | Translation upkeep | Low |
| F13 | **Offline CVE mode** (cached NVD feed) | Works without internet | Disk space; refresh job | Low |

---

## 6. Code-quality standards

### 6.1 Rules to adopt
| Area | Standard |
|---|---|
| Style | PEP 8 via `ruff` (format + lint), line length 100 |
| Typing | **Gradual:** v5.0 = type hints on all new code, `mypy --strict` enforced only on modules touched in that commit (existing untyped code is not rewritten just for types); v5.1 = full `mypy --strict` on `ids_core`/`recon` |
| Docs | Google-style docstrings; module headers saying the layer and responsibility |
| Errors | No bare `except` (CLI `:425`, `Port_Scan.py:264`…); domain exceptions (`DatasetError`, `LeakageError`, `BundleError`); one error mapper per presentation layer |
| Logging | `logging.getLogger(__name__)`; no `print` in the core; never log keys or dataset values |
| Config | No magic numbers (thresholds, timeouts, worker counts → `config.py`) |
| Naming | Fix typos/misleading names (`search_cves_cash` → `search_cves_cached`, `smote_applied` used for every balancer, `log_progress` that only logs) |
| Dead code | Remove the unused `SVC` import, the scapy logger, legacy `apply_smote` flags, the unused CSS (`.workflow-steps`, `.model-checkbox-*`) |
| Design patterns | **Strategy/Registry** for balancers (`app.py:1101-1141`) and models (`:1183-1276`); **Strategy** per banner probe (`Port_Scan.py:200-445`); **lookup table** for TTL (`:505-532`); **Repository** for models only (`ModelStore`); **functional core** for ML (data in, results out); **Factory** `create_app()` |
| SOLID | SRP: split `Port_Scan.main()` (input/scan/print/export); DIP: core depends on ports, not on Flask/requests |
| JS | ES modules, `const/let`, no globals, ESLint, a single `api()` fetch wrapper with error handling |
| CSS | Design tokens (CSS vars), no `!important`, BEM-ish class names |
| Tests | Arrange/Act/Assert; fixtures under `tests/fixtures` (small CSV samples); ≥80% coverage of the core |
| Git | Small commits, conventional messages, `pre-commit` hooks |

### 6.2 Per-file refactor map
| Current file | Becomes |
|---|---|
| `main.py` | `toolkit/cli.py` (flat menu + subcommands) |
| `Port_Scan.py` | `recon/scanner.py`, `recon/probes/*`, `recon/os_detect.py`, `recon/intel/{nvd,abuseipdb,llm}.py`, `recon/export.py`, `recon/report.py` |
| `IDS_with_Machine_Learning.py` | `ids_cli/app.py` + `ids_core/bundle.py` |
| `app.py` (1,724 lines) | `web/__init__.py`, `web/routes/{datasets,training,models,predict,pages}.py`, `web/jobs.py`, `web/workspace.py`, `web/security.py`, `services/*`, `ids_core/*` (incl. `recommend_balancing_method` → `ids_core/balancing.py`) |
| `main.js` | `static/js/{api,dom,upload,explore,train,evaluate,predict,charts}.js` |
| `style.css` + `style2.css` | `static/css/app.css` |
| `models/*.pkl` (v4, leaky) | Deleted; the CLI uses bundles exported from the web trainer on the same install (nothing shipped, D12 revision) |

---

## 7. Roadmap — three releases

**How every step is done (strangler approach):**
1. Write a test for the **correct** behaviour. It fails at first, which proves the bug exists. (For behaviour that is already right and must not change, write a characterization test that passes from day one.)
2. Make the smallest fix that makes it pass.
3. Pull the fixed logic into the shared core.
4. Move callers (web, CLI) over one at a time.
5. Delete the old code.

The target folder layout (§2) is where we end up, not where we start. No mass file moves before the tests exist. Each release ends only when its checks in §8 pass in CI.

### v5.0 — Correct & safe
| Step | Content | Fixes |
|---|---|---|
| 0.1 Safety net | git init, `.gitignore` (incl. `instance/`), datasets moved from `uploads/` to a gitignored `data/raw/`, project `.venv` on your current Python 3.14 + pinned requirements, pytest + ruff (`target-version = "py311"`) + `pip-audit`, pre-commit hook, patched-platform test helper (see **CI host** below), small test fixtures; **start acquiring the CIC-IDS2017 dataset now** (0.5 gate) | D18 (partial), D19 |
| 0.1b **Baseline snapshot** | **Before any code changes** (right after the `git init` commit), run the *current* web trainer through a **scripted** Flask test-client session (no manual clicking) in the **same pinned project environment** (Python 3.14 `.venv`) that v5.0 will use, so before/after differences can't come from library versions. Fixed: the exact click sequence (upload → with/without "Apply SMOTE" → feature list → balancer → models), seeds, `n_jobs=1`. Run **3 times** and record the spread (the unseeded inner SMOTE, B20, can make runs differ). Save accuracy, F1, per-class metrics and confusion matrices to `docs/baseline/` (JSON + README with the sequence, settings, library versions and commit hash). The script uploads from `data/raw/`, so moving the datasets in 0.1 doesn't break it. The v4 CLI's original test data isn't on disk, so its baseline is its documented claims plus predictions on the built-in samples | §8 ML report |
| 0.2 Honest claims now | Correct the README/model README accuracy claims; fix README duplicate lines + placeholder clone URL; mark the v4 model as leaky in the CLI | L1 (communication), §4.6 |
| 0.3 Critical security | **Interim trust rule until 0.4g:** web "Load model" upload **and** model export disabled with a visible "returns in v5.0-rc1" message (option A: no orphan files that nothing can load); CLI loads only the v4 files whose SHA-256 + versions are pinned in a small manifest, with a loud version warning; global `filterwarnings("ignore")` removed (B19); safe upload/export names (no overwrite); **XSS fixed in place**; debug off + waitress on 127.0.0.1; size/input limits; non-JSON → 400; generic errors. Training and predicting **inside the session keep working** | S1–S6, S8, S9, S17, B19 |
| 0.3 **boundary** | **XSS in v5.0 = escape/replace the ~15 `innerHTML` sites inside the existing `main.js` + a CSP header. The JS module rewrite and DOM-helper architecture (D1-B) are v5.1 (1.3) and must not start during 0.3.** CSP detail: v5.0 uses `script-src 'self'` (strict) but must allow `style-src 'self' 'unsafe-inline'`, because the current templates and `main.js` set inline `style=` attributes (e.g. `display:none`, confusion-matrix colours); a strict style policy would break the page. Inline styles are removed in v5.1 and the policy tightened | D1 |
| 0.4 Shared ML pipeline | Split into milestones below; each one is its own commit with its own tests | — |
| 0.4a Load & guard | Dataset loader (sniffed separator, header fix, chunked), **content-based** profiles (CIC matching ≥ 2 published variants, NSL-KDD, UNSW), leakage guard (single-feature rule + relabelling check, whole-word name patterns, integer/text ID check) | D16, F4, L6, L10, B17 |
| 0.4b Split | Dedupe + profile-defined split (official / temporal / temporal_per_class / group / stratified) with time-column validation and documented fallbacks, fixed seeds, class-coverage check | L4, L5, D9 |
| 0.4c Pipeline | Train-only Pipeline: impute → encode (one-hot / ordinal by model) → scale → select → sample → model; model and balancer registries; **every seed set, including nested samplers** | L2, L3, L7, B1, B5, B20 — *done 2026-10-20 (`toolkit/ids_core/pipeline.py`, `registry.py`)* |
| 0.4d Evaluate | CV on train, validation-based model selection, macro-F1 / balanced accuracy / per-class recall, test scored once; returns an **`EvaluationResult`** (becomes a TrainingRun field in 0.4e); **basic leaderboard table** built from it | L8, L9, F2 (basic) |
| 0.4e TrainingRun | Run record schema (`schema_version`, contains `EvaluationResult`), returned by `train(...)`; written to `instance/runs/` by the caller | D21 |
| 0.4f Bundle **library** | Signed bundle + manifest embedding the TrainingRun; local-key / pinned-hash loader; version-mismatch policy. **Library + its own tests only: no web route uses it yet**, so the app keeps working exactly as in 0.4e | D11, S1, S17, B18 |
| 0.4g Web integration | The web trainer and predict routes switch to 0.4a–f; `/apply_smote` becomes a train-only preview; recommender uses train data only; old preprocessing code deleted; **then** web "Load model" and export are re-enabled (signed bundles only). *(Ordering fix: re-enabling loading before the routes use the new pipeline would let a new bundle be served through old preprocessing code)* | L2, D9, B2, B3, B4 |
| 0.5 New CLI model | CLI reads any bundle and lets you choose among this install's exports (no shipped model, D12 revision); delete v4 pickles | L1, D5, D12, B14 |
| 0.5 **dependency** | **Gate:** 0.5 is not complete until the intended training dataset (improved CIC-IDS2017 for the multi-class model) is **acquired, SHA-256 hashed, profiled (0.4a), split-validated (0.4b) and documented** (source, licence, version, file list, hashes) in `docs/datasets.md`. Start acquiring it during 0.1 so it never blocks the end. **Decision at `5.0-beta` exit:** if CIC isn't downloaded, hashed and passing profile detection by then, commit to the **NSL-KDD fallback (D12)** with an honest note, so 0.5 can start during rc1; the CIC model moves to a later release | — |
| 0.5b Dataset licensing | `docs/datasets.md` records, **per dataset** (CIC-IDS2017, NSL-KDD, UNSW-NB15, any Kaggle derivative): source URL, licence/terms of use, citation requirement, and whether redistributing the raw data **and derived models** is allowed. Datasets are never committed; no derived model is shipped (0.5), and if that changes, it follows these terms | D20 |
| 0.6 Scanner correctness | Socket handling, no double connect, TTL ranges, ping per OS, UDP open\|filtered, TLS banners, empty-range guards, UTF-8 sanitised export, NVD instead of MITRE, LLM consent + redaction, **LLM output treated as untrusted (severity from CVSS only)**, env-based keys, **authorization confirmation + host/port caps**. **R3 disposition:** intel lookups stay one-at-a-time in v5.0 (parallel lookups are v5.1 step 1.4). Every lookup gets a timeout so a scan always finishes; listed under Known limitations | R1, R2, B6–B13, S12–S16 |
| 0.7 Visible quick fixes | CSS white-on-white + unclosed dark-mode block (with **before/after screenshots** of every section), system font stack instead of `Inter`, viewport meta, **contrast ≥ 4.5:1, `lang`, `<label for>`, visible focus outlines**; `main.py` "n" exit bug + recursion | U1, U2, U3 (font), U4, B15, R8 |
| 0.8 Minimal isolation | Workspace ID per browser session replaces `data_store` / `model_config` / `loaded_model_config` globals (unconditional; full TTL workspace stays in v5.1) + **"Clear workspace" button (`DELETE /workspace`)** + one-training-at-a-time `409` per workspace (tested with a blocking hook, D8) + **session cookie hygiene** (`HttpOnly`, `SameSite=Strict`, `Secure` via `TOOLKIT_HTTPS`, D23). *(Ordering fix: the cookie only exists once workspaces exist, so cookie hygiene moved here from 0.3)* | S7 (minimal), S10 (cookies), R4 (partial), D8, D23 |

**CI host (self-review finding).** The plan says "passes in CI on Windows/Ubuntu/macOS", but the project has **no GitHub remote**, and GitHub Actions needs one.
| Option | Pros | Cons |
|---|---|---|
| **A. Create a private GitHub repo and use GitHub Actions** | Real Win/Ubuntu/macOS matrix for free; Dependabot becomes possible | Your decision and your account; code leaves the machine (private repo) |
| B. Local only: `nox` sessions on Windows + WSL (Ubuntu) | Nothing leaves the machine | No macOS coverage; you must remember to run it |
| C. A + B | Fast local feedback, full matrix on push | Both to maintain (small) |
→ **Decided (final): everything local, on Windows, with your current setup.** No GitHub, no WSL, no nox, no extra Python versions. In this plan, **"CI" / "green" simply means: `pytest` passes on your Windows PC.**

How it works:
- **One command runs the tests:** `python -m pytest`. Lint is `python -m ruff check .` and the dependency check is `python -m pip_audit`. All three are ordinary pip packages installed in the project's virtual environment; nothing runs outside your machine.
- **Python:** your current **Python 3.14** in a project virtual environment (`.venv`). All libraries the project needs already run on it (sklearn 1.8.0, numpy 2.3.5, pandas, Flask, imbalanced-learn, XGBoost, LightGBM were checked on this machine).
- **Enforcement:** a git **pre-commit hook** runs ruff + the fast tests on every commit, and **no checkpoint is tagged** until the full `pytest` run is green. Each tag's CHANGELOG entry records the result (date, Python version).
- **Linux and macOS: written to work, not tested on real machines.** The code follows the portability rules below so it should run there unchanged. The tests "pretend" to be Linux or macOS (by patching `platform.system()` / `sys.platform`) to check the OS-specific parts from Windows. Only real network/ping behaviour on those systems stays unverified, and the README says so.

**Portability rules (so the code also runs on Linux and macOS, with no extra tools):**
| Rule | Why |
|---|---|
| Build every path with `pathlib.Path` relative to the project folder; no hard-coded `\` or `C:\` | Windows uses `\`, Linux/macOS use `/` |
| Open every text file with `encoding="utf-8"` and CSV files with `newline=""` | Default encodings differ per OS (B12 crash today) |
| OS-specific commands go through one small function per command, e.g. `ping_args(os_name)` → `-n 1 -w ms` on Windows, `-c 1 -W s` on Linux, `-c 1 -t s` on macOS | Flags differ per OS (B7) |
| Never call shell-only commands (`cls`, `dir`); use Python equivalents (`os.system("cls" if os.name == "nt" else "clear")` stays in one helper) | Commands don't exist on other OSes |
| Never pass `shell=True` to `subprocess`; always a list of arguments | Different shells per OS + injection risk |
| Use only libraries that publish wheels for Windows, Linux and macOS (all current ones do); new dependencies are checked for this before being added | Avoids "can't install on Mac" surprises |
| No spaces in new folder names (`IDS_CLI - CIC-2017` → `ids_cli` in v5.1) | Spaces break scripts and shells on every OS |
| File names compared case-sensitively and written in lowercase | Linux is case-sensitive, Windows isn't |
| CPU count from `os.cpu_count()`, temp files from `tempfile` | No OS-specific assumptions |
| Each OS-specific function gets a test for `windows`, `linux` and `darwin` (patched platform) | Catches most cross-OS bugs from Windows |

**Later, no rework needed:** if you ever want real Linux/macOS testing, the same `pytest` command runs unchanged on those systems (or in WSL / GitHub Actions).

**Internal v5.0 checkpoints:** each one is a tagged, known-good commit you can always go back to. If a later step breaks the web trainer, the last checkpoint still works.

**Every checkpoint is shippable (anti-abandonment rule).** If work stops at any checkpoint, the project must still be usable and honestly documented at that point:
- the app starts and every menu path works
- the README describes what this checkpoint does and doesn't do (known limitations listed)
- `CHANGELOG.md` has an entry for the tag
- the checkpoint's tests pass in CI

A checkpoint that fails any of these isn't tagged.
| Checkpoint | Contains | Exit criterion |
|---|---|---|
| `5.0-alpha` | 0.1 safety net, 0.1b baseline snapshot, 0.2 honest claims, 0.3 critical security, 0.7 quick fixes, 0.8 minimal isolation | Baseline saved; security + isolation tests green; `pip-audit` clean; app still runs end to end; README "Known limitations" says plainly that **web model export/load is paused until rc1** |
| `5.0-beta` | 0.4a–0.4d (load/guard, split, pipeline, evaluate) | Leakage, split-strategy and test-isolation tests green; the old web flow still works; **CIC-or-NSL-KDD decision recorded** in `docs/datasets.md` |
| `5.0-rc1` | 0.4e–0.4g (TrainingRun, bundle library, web integration, export/load re-enabled); 0.5 may start here if the NSL-KDD fallback was chosen | Reproducibility + bundle tests green; old preprocessing code deleted; export → load round trip works in the web app |
| `5.0-rc2` | 0.5 CLI model (dataset gate met by CIC **or** the NSL-KDD fallback), 0.5b licensing, 0.6 scanner correctness | CLI/web agreement + scanner tests green; `docs/datasets.md` complete |
| `5.0` | — | The full §8 suite passes with `pytest` on Windows (Python 3.14) + manual smoke run on Windows; README states Linux/macOS are supported by design but not tested on real machines |

**Scope discipline:** v5.1 architecture work must not leak back into v5.0. During v5.0, ugly-but-tested code inside `app.py` is acceptable. Folders are not reorganised mid-fix because a file "feels too big"; that waits for 1.1.

### v5.1 — Clean architecture
| Step | Content | Fixes |
|---|---|---|
| 1.1 Package | `toolkit/` modular monolith, `config.py`, logging, the 3 ports, registries (models, balancers, probes, profiles) | §2, D17 |
| 1.2 Web structure | `create_app` + blueprints + `/api/v1` schemas, full TTL workspaces, job runner + progress + cancel, CSRF + security headers, upload TTL | D6, D7, D8, S7, S10, S11, R4–R7 |
| 1.3 Front end | JS modules + DOM helper, one `app.css` with tokens, stepper wizard, touch-friendly inputs | D1–D3, B3, B4, U3, U5 |
| 1.4 CLI & launcher | flat menu + subcommands, browser auto-open, batch/JSON prediction, parallel intel lookups | D4, R3, R9, B16, F1 |
| 1.5 Packaging & docs | pyproject + extras, architecture doc, API reference, user guide, LICENSE, CHANGELOG | D18, D20, §4.6 |

### v5.2 — The user's improvements (asked by the user, 2026-10-25)
| Step | Scope |
|---|---|
| 2.1 Theme button | Auto / Light / Dark in the navbar of both pages, saved in the browser |
| 2.2 Responsive pages | 2.2a phones and tablets, 2.2b large screens and long content (D2) |
| 2.3 The code arranged by interface | `cli/`, `web/`, `core/`, the same layers inside each |
| 2.4 The port scan in the web app | a job with progress and cancel; 127.0.0.1 by default, `--host` behind a warning |
| Gate 5.2 | full files, fresh install, layout and contrast audits, keyboard only, `pip-audit`; see [V5.2_CHECKLIST.md](V5.2_CHECKLIST.md) |

### v5.3+ — Impressive features (moved from v5.2 on 2026-10-25)
| Order | Features |
|---|---|
| First | F2 full leaderboard UI (charts, run-to-run comparison; the basic table already ships in 0.4d), F3 explainability, F5 scan reports, F6 history (built on TrainingRun) |
| Next | F7 recon→IDS dashboard, F9 tuning UI, F10 dashboard home |
| Stretch | F8 PCAP/live detection, F11 scheduled scans, F12 Arabic/RTL, F13 offline CVE, D13-B asyncio scanner |

v5.3 started 2026-10-30 with F2 (done 2026-11-02); F3 approved 2026-11-02 with permutation importance and a typical-value explanation instead of SHAP: see [V5.3_CHECKLIST.md](V5.3_CHECKLIST.md).

---

## 8. Verification = definition of done
A release is finished only when all of its checks pass automatically in CI (manual checks are marked).

- **Leakage tests:** no duplicate row appears in both train and test; sampler/selector fitted only on train indices; the guard flags `level`, `attack_cat`, `Attempted Category`, IP/Timestamp, and label-copy columns.
- **ML report:** before/after macro-F1, balanced accuracy and per-class recall on CIC + NSL-KDD (scores are expected to drop to realistic levels); CV mean±std; model card generated from the TrainingRun.
- **Test-set isolation:** with a spy on the test split, assert it is never passed to the feature selector, balancer, balancing recommender, tuner, threshold/calibration step or model-selection step; it is scored exactly once.
- **Split strategy:** each profile produces its declared split (NSL-KDD uses the official files; CIC `temporal_per_class` has, within each class, all test timestamps after train timestamps; group split has no group on both sides), and the strategy appears in the TrainingRun.
- **Split fallback:** a dataset with a broken/unparseable/reset time column falls back to the documented strategy and records the **exact reason code** (e.g. `TIME_PARSE_FAILURE_RATIO_EXCEEDED`); the test asserts the code, not just that a reason key exists. It never silently produces a temporal split.
- **Profile detection (content-based):** shuffling rows, renaming the file and changing line endings/encoding give the same profile; the CIC profile matches **≥ 2 published CIC-IDS2017 header variants**; an unknown CSV gets the generic profile with a warning.
- **Unknown-name label leak (exact rule, D9):** a renamed copy of the label (`xyz_col`), a label-encoded copy and a prefix/suffix variant are **blocked** (≥ 0.99 and about one value per class); a near-perfect **continuous** feature only **warns**; `Idle Mean` is **not** flagged by the `id` pattern; an integer row counter is flagged as ID-like; a continuous, almost-all-unique feature (`Flow Duration`-like) is **not**; an allowlist override is accepted and recorded in the TrainingRun.
- **Baseline exists:** `docs/baseline/` holds the pre-change metrics from the scripted sequence, 3-run spread, library versions and commit hash; the v5.0 ML report compares against it.
- **Seeds:** two identical training runs (same data, same settings, `n_jobs=1`) with SMOTE-ENN and SMOTE+Tomek give identical results.
- **Bundle robustness:** truncated, malformed, empty, or wrong-format files are rejected with a clear error before deserialization, and nothing crashes.
- **Version mismatch:** a bundle whose manifest says sklearn 1.6.1, loaded in a 1.8 environment, is **refused with a clear message** before deserialization (never a silent prediction); a patch-only difference loads with a warning.
- **Signing key safety:** `git check-ignore instance/signing.key` succeeds; deleting the key makes old bundles fail with the documented "key lost → retrain from TrainingRun" message.
- **Dependencies:** `pip-audit` reports no known vulnerabilities in the locked dependencies (CI fails otherwise).
- **Runaway protection:** while a first training request is held inside training by a blocking test hook (`threading.Event`), a second request in the same workspace gets `409 Busy`; after release the first finishes normally. "Clear workspace" frees the dataset and models and the workspace is usable again.
- **Cookies:** the session cookie has `HttpOnly` and `SameSite=Strict` and contains only the workspace ID; `Secure` is absent by default and present with `TOOLKIT_HTTPS=1`.
- **Warnings not hidden (B19):** no module installs a global `filterwarnings("ignore")`; loading a mismatched model shows the warning.
- **Export:** exporting twice with the same name creates `name`, then `name-1`, and never overwrites; during alpha/beta the export and load routes return a clear "paused until rc1" message.
- **Parallelism:** the **effective** number of workers (checked by patching `joblib.Parallel` / using `threadpoolctl`) never exceeds `TOOLKIT_MAX_WORKERS`, not just the configured value.
- **(v5.1) Cancellation:** cancelling a training job stops it, frees its memory, leaves no partial bundle/TrainingRun behind, and the workspace stays usable.
- **Resource limits:** an upload above the size limit → 413; a dataset above the row/memory limit is refused (or sampled with explicit consent) instead of exhausting RAM; training respects `TOOLKIT_MAX_WORKERS`.
- **Session isolation (v5.0 minimal):** two browser sessions on localhost never see or overwrite each other's dataset or model.
- **Python policy:** `pytest` passes on your Python 3.14; `ruff` (target py311) reports no syntax newer than Python 3.11.
- **Portability:** no hard-coded `\` paths, every text file opened with `encoding="utf-8"`, no `shell=True` (checked by a small test that scans the source, plus ruff rules).
- **Reproducibility:** re-running a saved TrainingRun on the same dataset gives the same metrics within the stated tolerance, and the bundle hash in the record matches the file.
- **CLI model:** the default bundle's pinned hash matches; the CLI loads a bundle exported by the web trainer and gives the same prediction as the web app for the same input.
- **Security tests (Flask client):** unsigned/tampered/foreign-device bundle rejected **before** deserialization (assert `joblib.load` is never called); `../` upload contained; `<img onerror>` header shown as text; bad export name refused; two sessions isolated; oversized upload → 413; non-JSON body → 400.
- **Scanner:** against `127.0.0.1` + `python -m http.server` + a closed port: stable socket count, TTL classified, UTF-8 sanitised export, mocked LLM receives only the redacted version.
- **LLM output untrusted:** a mocked LLM reply containing HTML, ANSI escapes, a fake "CRITICAL" severity and a command is shown as stripped plain text only; severity still comes from the CVSS score; nothing in the reply changes program behaviour. **Second case:** a perfectly plain-text reply that mentions a fake `CVE-9999-00000`: the report's CVE list comes **only** from NVD and the fake ID never appears in it.
- **CSV formula injection (separate test from ANSI):** exported cells starting with `=`, `+`, `-`, `@` (and tab/CR) are written as text (prefixed with `'`).
- **Authorization:** a scan doesn't start without the confirmation prompt or the `--i-am-authorized` flag; host and port counts above the caps are refused with a clear message.
- **Slow intel:** with mocked NVD/AbuseIPDB/LLM that time out, a scan still **finishes** and reports which lookups failed.
- **UI:** light and dark modes readable; 375 px phone width usable; progress bar and cancel during training (v5.1).
- **Accessibility:** contrast ≥ 4.5:1 in both themes (automated check, e.g. axe/pa11y in CI or a manual checklist); every input has a label; visible focus outline; `lang` set; (v5.1) the whole flow works with keyboard only and status messages are in `aria-live`.
- **Checkpoint shippable:** for every tag, the app starts, every menu path works, README limitations + CHANGELOG entry exist (manual checklist).
- **UI regression (manual):** before/after screenshots of every section in light and dark mode for the 0.7 CSS fixes.
- **Cross-platform logic:** OS-specific functions (ping flags/timeouts, paths, line endings, UTF-8 IO) pass their unit tests for `windows`, `linux` and `darwin` via a patched platform, run on Windows.
- **Cross-device:** `pytest` on Windows; manual fresh install on Windows → `python main.py` (v5.0) / `python -m toolkit` (v5.1) → every menu path end to end. Real Linux/macOS runs aren't done (no extra tools/machines); the code follows the portability rules so they should work, and the README says it is untested there.
