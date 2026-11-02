# Decisions and rules of work

The decisions the project is built on, why they were taken, and the rules every change follows. The decisions taken with the outside review committee in v5.3 (RD1–RD4) are in [REVIEW_DECISIONS.md](REVIEW_DECISIONS.md). The full reasoning behind D1–D23, with every option weighed, is in [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) §2–§3. What exists is in [PROJECT_STATUS.md](PROJECT_STATUS.md); what comes next is in [ROADMAP.md](ROADMAP.md).

A decision that was replaced stays listed, marked with what replaced it (section 9).

---

## 1. Priorities

**For every change:** security → correctness → maintainability and testability → performance → extensibility → style.

**For v5.3 (from the review):** correctness → security → reliability → usability → features. No new feature is added just to have more features before the known risks are closed.

## 2. How the work is done

- **Propose first.** Every step is proposed (what changes, where, why, the risks) and starts only after the user approves it. When the user asks for an opinion, the answer is an opinion, and nothing is changed
- **A test first, then the smallest change, then one commit.** For behaviour that must not change, a characterization test that passes from the start
- **The checks after every step:** the full tests, ruff, mypy, pip-audit (the environment and `requirements.lock`), and the browser when a page changed
- **Git:** commits in the user's name; a tag only after the user approves it; **nothing is pushed** without the user's decision; each file keeps its own line endings; new files use LF
- **Dependencies:** none added without approval. F3 uses no SHAP; F5 uses no PDF library
- **Data:** the datasets are never committed. Real-data checks have three tiers: fixtures for every change, stratified samples for each step, the full files only at a gate or for a reported metric. A number from a sample is called a sample number, not a result
- **Platforms:** works on Windows now (Python 3.14, the user's machine); written so it can run on Linux and macOS with nothing extra, all of it local
- **Every tag is shippable:** the app starts, every menu path works, the README lists the limits, the CHANGELOG has the entry
- **Claims need evidence.** A document says how each claim is known (a test, a capture, a dated manual check), and says what is only implemented and not verified. Failures are reported as they are
- **User files are left alone.** Files the user made are not changed; the real `instance/` is never touched by checks (a scratch `TOOLKIT_INSTANCE_DIR` is used)

## 3. Engineering rules

**Metrics are signals, not orders** (agreed in v5.2 step 2.3):
- Complexity at most 10 (ruff `C901`) and at most 400 lines per Python or JavaScript file in `toolkit/` (not CSS, not `scripts/`) are review signals. A file or a function is split only for a real benefit (a separate responsibility, testability, fewer bugs), never with helpers or dispatch tables made just to lower a number. A justified exception gets `# noqa: C901` with its reason
- No limit on function length
- Every suppression (`noqa`, `type: ignore`, `pragma: no cover`, `per-file-ignores`) is listed with its file, rule and reason in one inventory; a test fails on an unlisted one and on a stale entry; the list only shrinks
- A docstring on every function, saying what isn't obvious: a security assumption, an order that must be kept, a leakage or correctness point, a memory cost. No filler
- A refactor keeps behaviour, measured: memory at most 5% more and time at most 10% more (fresh process, median of three runs), or it stops and is reported

**Structure:**
- Layers `ui → controllers → services → data` inside `core`, `cli` and `web`; imports only go down; `cli` and `web` never import each other (tested)
- Network, processes and the browser only in the data layers (tested)
- An interface (port) only where an implementation is really swapped or faked in tests; everything else is plain functions and registries
- A functional core: the ML takes data in and returns results; files are read and written at the edges
- Settings in typed dataclasses in `settings.py`, overridable from the environment; no magic numbers
- Domain errors with codes; one error mapper per interface; no bare `except`; no `print` in the core; keys and dataset values never logged
- **Reuse before adding:** look for the existing mechanism (a queue, a helper, a registry) before writing a new one

**Portability:** `pathlib` paths; every text file opened with `encoding="utf-8"` and CSV with `newline=""`; one function per OS-specific command, tested for Windows, Linux and macOS with a patched platform; never `shell=True`; only libraries with wheels for all three systems; ruff targets Python 3.11.

## 4. ML rules

- **Test isolation:** the test split never influences feature selection, balancing (the recommender included), tuning, thresholds or model selection. They use the training split, CV or a validation split only
- **The test split is scored once per model for the leaderboard**, after the model is chosen by CV. Reusing it afterwards (the permutation importance) is a diagnosis; anything chosen from that diagnosis needs a validation split cut from the training split, and a final claim after such choices needs a holdout never used before (HC1)
- **Dedupe before the split**; the split strategy comes from the dataset's profile, its checks and fallbacks recorded as reason codes, never a silent strange split
- **The leakage guard** catches a set of known patterns; it cannot prove there is no leakage left
- **Every seed is fixed**, nested samplers included
- **Honest metrics:** macro-F1, balanced accuracy and recall per class, not accuracy alone; the numbers reported as they are, even when worse than before
- **Explanations are approximate, not causal**, and say so where they are shown (HC2)
- **No generalization claim** before a test on data from another source or time

## 5. Security rules

- **Nothing is deserialized before its signature (or a pinned hash) and its library versions are checked**, on the same bytes that are then loaded. Signing proves who made a file; it does not make unpickling safe
- **Outside input is data, never instructions:** dataset text, file names, banners, AI replies, saved files read back (size limit, schema, ids checked: RD4)
- **The AI never drives a decision:** CVE IDs, CVSS and severity come from NVD only; the AI gets the product and version only, after consent; its reply is plain text, labelled, length-capped
- **The scan contract:** no packet to the target and nothing to NVD, AbuseIPDB or the AI before validation, the limits, the authorization and the opt-ins. A host name may be resolved by the system's DNS before the authorization question (documented; whether to move it after the question is not decided yet). A private address never goes to AbuseIPDB. The lookups send the product and version only
- **One project trust domain** (RD1): the local machine and every holder of the remote token; no accounts or roles

## 6. Terms

**Severity:**
- **Critical:** a way to break in, lose or damage data, or ML results that mislead in a way that matters
- **High:** a real practical risk, or a core function that isn't safe
- **Medium:** a real problem with a workaround or a limited effect
- **Low:** an improvement that doesn't block use

**Test levels:**
- **Unit:** one module, with fakes for what it uses
- **Integration:** several modules or layers together, such as a route through its service to the files
- **End-to-end:** the app as a user runs it: the CLI from its entry point, or the web app in a browser
- **Manual verification:** a check made by a person, written down with its date

**Done** means evidence that can be checked: a test, a browser capture, or a dated manual test record. The word alone is not evidence.

**Support levels for an OS or a browser:** implemented (the code is meant to work); verified by automated CI; verified by hand on a real machine; supported (verified, and its problems are fixed). Today Windows with Python 3.14 is verified by hand and by the local tests; Linux and macOS are implemented only; the pages are verified on Chromium only.

---

## 7. The plan's decisions, D1–D23

From [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) §3 (2026-10-19, revised through six reviews). The table gives the choice and the main reason; the plan has every option.

| ID | Topic | Choice | Why | State |
|---|---|---|---|---|
| D1 | Rendering | A DOM helper with `textContent` and ES modules, plus a CSP | Safe by default, no build step | Done (5.0 escaped in place; 5.1 the modules) |
| D2 | Styling and devices | One `app.css` with light and dark tokens, the viewport meta, checkbox lists instead of Ctrl+Click; the system font stack (a web font would send a request outside) | Readable in both themes, touch-friendly, nothing fetched from outside | Done; every page at 8 sizes in 5.2 |
| D3 | The trainer's flow | A stepper with disabled buttons and progress | Guides the user and mirrors the pipeline | Done (6 steps) |
| D4 | The launcher | A flat menu and `argparse` subcommands | Scriptable, testable, no recursion | Done (5.1) |
| D5 | CLI input | Model-agnostic: features, types and classes read from the bundle; real examples from the test split; `--csv`, `--json`, `--bundle` | Any exported bundle works | Done (5.0-rc2) |
| D6 | The API | `/api/v1` with request schemas, one error shape, size limits | Consistent, safe, self-documenting | Done (5.1) |
| D7 | State | A per-session in-memory workspace with a lock, TTL clean-up and limits | Separates sessions; simple | Done. The workspace separates work and bounds memory; it is not a security boundary (RD1) |
| D8 | Long tasks | An in-process job runner with polling, progress and Cancel; one training at a time; `TOOLKIT_MAX_WORKERS` leaves a core free | No extra services; works on any OS | Done (5.1). RD2 puts the permutation importance on the same queue |
| D9 | Pipeline and leakage | An `imblearn` pipeline fitted on the training split only; dedupe and the profile's split first; the leakage guard; the test-isolation invariant | Leakage-proof; one object to save | Done (5.0-beta) |
| D10 | ML quality | CV on the training split; one-hot or ordinal encoding by model; macro-F1, balanced accuracy, recall per class; selection on CV; explainability | Honest numbers for an IDS | Done, except explainability (F3, in progress) and tuning (F9) |
| D11 | Serialization | joblib plus an HMAC-SHA256 signature and a version manifest; nothing deserialized before it is checked; a version policy (patch: warn; minor or major of the estimator libraries: refuse) | Rejects tampered and foreign files | Done (5.0-rc1). Its threat model excludes someone who can write `instance/` (see HS5) |
| D12 | The CLI model | Trained in the web trainer, exported as a signed bundle, loaded by the CLI | One training path | Done. **Revised in 0.5:** no model is shipped (a random forest on CIC is about 2 GB); the CLI lists this install's bundles |
| D13 | The scanner engine | Threads kept; sockets closed with `with`; one connection per port; probe plugins; parallel lookups | Safe, small, extensible | Done. An asyncio rewrite (D13-B) is a stretch item |
| D14 | Scanner correctness | TTL ranges, ping per OS, UDP open or filtered, TLS for banners only, guards, validated ports, host limits | Correct results on every OS | Done (5.0-rc2) |
| D15 | Outside lookups and privacy | NVD CVE API 2.0; the AI only after consent with the product and version; its reply untrusted | Accurate; little data leaves the computer | Done. An off switch for NVD is HS4 |
| D16 | Dataset handling | Content-based profiles; sniffed separator; stripped headers; random file names; uploads removed | Correct loading of every known dataset | Done |
| D17 | Config, secrets, logs | `config` (now `settings.py`) with environment overrides; keys from the environment or `.env`; rotating logs with secrets hidden; no global warning filter | One place for settings; no secret in a log | Done |
| D18 | Packaging | `pyproject.toml`, a lock file, extras (`[boost]`, `[dev]`), `waitress` | `pip install .` on any OS | Done (5.1) |
| D19 | Quality gates | pytest, ruff, mypy, pip-audit, a pre-commit hook, all local on Windows; Python 3.14 for development, `requires-python >= 3.11` | No extra installs; works on the user's machine | Done. CI on a remote (HV2) waits for the user's decision to push |
| D20 | Documentation | An honest README, the architecture document, the model card, the API reference, the user guide, the CHANGELOG, a LICENSE, the dataset licences | Anyone can install, use and check it | Done; HC4 corrects the descriptions |
| D21 | Reproducibility | One immutable TrainingRun record per run, embedded in the bundle and saved in `instance/runs/` | One source of truth for the model card, the history and the thesis tables | Done; written on Export only until HR5 (RD3) |
| D22 | Workspace TTL and jobs | A running job holds a lease on its workspace; a maximum job duration | No lost work, no stuck memory | Done (5.1) |
| D23 | The `Secure` cookie | `SESSION_COOKIE_SECURE = TOOLKIT_HTTPS`, off by default | Explicit and testable | Done. HTTPS itself is HS2 |

## 8. Decisions taken during the releases

The ones that shape later work. The checklists have every step's details.

| When | Decision | Why |
|---|---|---|
| 0.4a | The leakage guard's tree is limited by leaves, not depth; the name pattern is `attempted category`, not `attempted`; the 99%-unique ID rule needs at least 1,000 rows | Found by tests and real data: an exact label copy with many classes was missed; `su_attempted` is a real NSL-KDD feature; small samples look unique |
| 0.4b | Dedupe ignores identifier columns; a temporal cut never separates equal timestamps; the official split keeps both published files by default, the overlap counted, and removed only in strict mode | Copies that differ only by an ID are still copies; scores stay comparable with other published work |
| 0.5 | No model is shipped (the D12 revision) | Bundles are too large for GitHub, and a size cap would decide which models are allowed |
| CI host | Everything local on Windows: "green" means `python -m pytest` passes on the user's PC | No remote, no extra tools; the user's choice |
| 5.2, 2.3 | The code arranged by interface (`core`, `cli`, `web`) with the same layers inside; metrics are signals (section 3) | One structure everywhere; refactors done for real benefits, not scores |
| 5.2, 2.4 | The scan on the web as a job; the app on 127.0.0.1 by default; `--host` behind a warning and a token per start | The scanner in the browser without exposing it by default |
| F2 | Saved Runs read only the records, never a bundle; 50 per page; differences only when both runs scored the same test rows; a record is data, its id checked before a path is made | Fast, safe, and no misleading comparison |
| F3 | Permutation importance (not SHAP), on the test split (a stratified sample of 5,000 rows, 5 repeats, the run's seed), one feature at a time so a job can show progress and stop; the explanation of one row by the typical value of each feature | No new dependency; the same method for all 11 models; immediate explanations |
| v5.3 review | The hardening section H before the rest of F3; the committee's RD1–RD4 | The review's conditions for calling v5.3 complete |

## 9. Superseded

| Decision | Replaced by |
|---|---|
| Plan F6: the history in SQLite (D7-C, "a disk workspace when history arrives") | RD3 and RD4: every training's record and every result are files written once in `instance/`; F6 builds the history, its retention and its deletion on them (HR4, HR5) |
| D12 steps 4 and 6: ship a default CLI model and its record | The 0.5 revision: no model is shipped |
| v5.0's plain description "works offline" | Local-first with optional outside services (HC4) |
| "The test split is scored once" as a literal statement | Section 4: scored once per model for the leaderboard; reused afterwards only as a diagnosis (HC1) |
| The F2 and F3 decisions replaced in the review | Listed in [REVIEW_DECISIONS.md](REVIEW_DECISIONS.md), "Superseded" |
