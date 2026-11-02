# Decisions of the review committee

The design decisions taken with the outside review committee in v5.3, with why. The project's own decisions (D1–D23 of the plan and the later ones) and the rules of work are in [DECISIONS.md](DECISIONS.md), which also defines severity, test levels and "done". The checklist ([V5.3_CHECKLIST.md](../docs/V5.3_CHECKLIST.md)) holds the tasks that implement these; [CHANGELOG.md](../CHANGELOG.md) holds what shipped. The IDs start with **RD** so they never clash with the plan's D1–D23. When a decision replaces an earlier one, the earlier one is listed under **Superseded** with what replaced it, and the checklist marks it there too.

**Status:** RD1–RD4 were approved on 2026-11-02 by the outside review committee (security, architecture, reliability, ML, product) and by the user. RD1 is implemented: the new token and what happens to each job (HS1, 2026-11-02) and the scan record's source (HS3); the permissions of `instance/` (HS5) are not yet. RD5 (2026-11-03) is implemented on the command line (HS4a) and on the web page (HS4b). **RD2–RD4 are not implemented or tested yet.** The tasks that implement them are in the checklist's section H.

---

## RD1 One project trust domain
Approved 2026-11-02.

The toolkit has one owner. It trusts **the local machine and every holder of the remote access token as one project trust domain.** Remote access is project-level access: there is no user identity, no ownership of files and no roles. A session's workspace keeps one person's work from mixing with another's and bounds the memory; it is **not** a security boundary: saved runs, bundles and scan reports are shared by everyone in the domain.

Why: the toolkit is a local tool its owner can open to other computers, not a service with independent users; per-file ownership would be RBAC under another name.

Rules:
- The token is a bearer credential: whoever holds it has the owner's rights over the shared resources, not only a view of the pages. The terminal and the unlock page say so before remote access is used
- A new token (made on this computer only) ends every remote session at once
- After a new token: a remote job still **waiting** that was asked for under the old token is cancelled; a job already **running** goes on, and the owner can stop it on this computer; the old token's holder can neither follow nor cancel anything (every request answers 401). Jobs asked for on this computer are not touched. Each job records where it came from (this computer or another address) and the unlock it was asked under
- The scan record names the **source of the request** (this computer, or an address); an address is not a person's identity
- The local machine is inside the domain (loopback needs no token). On a computer shared by OS accounts that don't trust each other, RD1 alone does not separate them: `instance/` must be protected by the OS permissions (HS5)

## RD2 The heavy jobs share the training queue
Approved 2026-11-02.

The permutation importance runs on the queue the training already has: one FIFO queue, one worker, so one heavy job at a time. The scan keeps its own queue and the limits of `ScanConfig`.

Why: the queue already has cancel while waiting, cancel while running, progress and clean-up after a failure, all tested; a second mechanism for the same purpose would drift from it.

Rules:
- The jobs share the queue, the worker, cancel and progress; each kind keeps its own inputs and outputs
- An importance job is bound when it is **asked for** to a snapshot of one run: its `run_id`, the model's id, references to that fitted model and to that run's test rows and labels, the settings and the seed. References, not copies. It never reads "the current run" when it starts
- One job per `(run_id, model id)`: asked again, it answers the job that exists, or the result if it was saved
- The limit is one worker, fixed; it does not change while the app runs
- No other path starts a training beside the queue

## RD3 A record for every successful training
Approved 2026-11-02.

Every training that succeeds gets its `run_id` and a TrainingRun record in `instance/runs/`, written at its end. Export is a separate step that writes the bundle; it is no longer what saves the run.

Why: the user can keep and compare results without exporting a model; the importance has a run to belong to.

Rules:
- **Saved Runs are every successful training** with a record, exported or not; nothing assumes a run has a bundle
- A training is shown as done for good only once its record is saved. If the save fails while the app runs, it shows "Completed, not saved" with a retry that saves the result kept by its job (same `run_id`), without training again. If the app stops before the record is saved, the result is lost: nothing promises a retry after a crash
- The importance can be **calculated** only for a run whose models and test split are in memory (the workspace's run). An older run shows the results saved for it, and no Calculate button. A bundle holds the models but not the test split, so it cannot replace that
- The page says, before the first training: "Training results are saved locally after each successful training. The saved record contains aggregated metrics, class names, feature metadata and typical feature values; it does not contain the original dataset rows."
- Deleting a run is a deliberate, confirmed act that lists every file it removes (RD4)

## RD4 Artifacts are never changed once published
Approved 2026-11-02.

A record, an importance result and a bundle are each written once, to a file of their own, and never changed: "exported" is the bundle file being there, not a field of the record. The rule already held for records (`open("x")`) and bundles (`"xb"`); RD4 makes it the rule for everything the toolkit saves.

Why: with no file updated, there is no lost update and no read-merge-write; the importance and the export never write the same file.

Rules (IA1–IA10 of the committee, and what was added to them):
- **IA1** A record is published once, after a training succeeds
- **IA2** An importance result is its own file per run and model, `<run id>.importance.<model id>.json`, and never replaces a published one. It holds the `run_id`, the model id, its schema version, the method's version, the settings (rows, repeats, seed, per class), and the toolkit's version. A change of method later gets a new versioned file and a stated migration, never a silent replacement
- **IA3** A bundle is its own file, `<run id>.bundle`; the export never touches the record
- **IA4** All three are published the same way, by **one module of the data layer** that records, importance results and bundles all use, tested once: write a temporary file in the destination's folder, flush and `fsync` it, check it (and sign a bundle), then publish it without replacing anything; it is published only if that succeeds. Planned mechanism (to be tested on each OS): `os.rename` on Windows (it fails if the name exists and is atomic on one volume); `os.link` then removing the temporary file on Linux and macOS (it fails if the name exists); a file system without hard links is refused with a message, never written with a replace. **The guarantee:** a stopped or crashed process leaves either the whole file or none. A power cut is best effort: Windows has no folder `fsync`, and that is written in the docs
- **IA5** File names come from checked ids only: a run id in the form the toolkit writes, a model id from the registry and among the record's own models, never a display name. A file is read as untrusted data: a size limit before reading, its schema, a newer schema refused, its ids matching its name; a bad file is left out and counted, never a 500
- **IA6** One importance job per run and model, bound to its snapshot (RD2)
- **IA7** One lock inside the process orders publishing and deleting (no read-merge-write). A result computed outside the lock is published under it only if its record still exists; a run deleted meanwhile gets nothing published. An OS lock on the instance is a separate thing (HR3)
- **IA8** Deleting a run removes its importance files, then its bundle when the user chose to remove it, then the record last; if any of them can't be removed, the record stays, the run is still shown and the delete can be tried again. The confirmation names every file, and lets the user keep the bundle: kept, it becomes an unregistered bundle on purpose
- **IA9** A storage report finds what has no owner or is damaged, and deletes nothing on its own: importance files without a record or for a model not in it, temporary files, unregistered bundles, incomplete bundles. Listing a bundle never unpickles it
- **IA10** A bundle's state comes from its header: the format records its lengths (`MAGIC | manifest length | manifest | payload length | payload | signature`), so a few bytes and the file's size show it is complete, with no HMAC and no unpickle. The states: **Not exported** (no file), **Incomplete: the export was interrupted** (the lengths don't match the size, such as a file left by the old direct write), **Exported** (complete; its signature is checked when it is used), **Verification failed** (shown after a use or a Verify found a bad signature). An incomplete bundle is never deleted on its own; the user removes it, then may export again
- **Unregistered bundles can be registered again:** a bundle carries the signed copy of its run's record, and records never change, so after its signature is checked that copy can be published as the record (to be confirmed: the copy is complete)

---

## RD5 The outside services are on by default; the key is the consent
Decided by the user on 2026-11-03, while HS4 was designed.

NVD, the AI explanation (OpenRouter) and the AbuseIPDB check are on by default in every scan,
on the command line and on the web page. A service needs its key in `.env` (NVD needs none);
without it the service is off and the scan says so ("off: no OPENROUTER_API_KEY"); on the web
page its switch says "No API key: add it to .env, then refresh this page". Each service can be
turned off for a scan: `--no-nvd`, `--no-ai`, `--no-abuseipdb`, the answer no to its question,
or its switch on the page. A service that fails during a scan (no connection, a timeout, a key
refused, too many requests) is not asked again in that scan; the scan goes on, its summary
names the service and why ("Unavailable: NVD (no connection)"), and its ports say "NOT
CHECKED", never "NO KNOWN CVE". The next scan tries again.

Why: the user wants the toolkit to use what it was given; putting a key in `.env` is the
choice to use that service. Asking again in every scan added nothing a key doesn't say.

Rules:
- A private address never goes to AbuseIPDB, and NVD and the AI get the software name and
  version only: unchanged
- The scan still says first, under "Online in this scan", what it may send to each service
  or why not, and at the end how many requests went to each
- There is no setting that turns every service off at once: each has its own switch
- A port's CVEs not looked up (NVD off, failed or unavailable) are "NOT CHECKED"; a port with
  no software found keeps "UNKNOWN"
- `--ai` and `--abuseipdb` are still accepted (they are the default now), so older commands
  work; each with its `--no-` form is refused

## Not decided yet
- **DNS before consent:** today a host name is resolved before the authorization question (`toolkit/core/services/recon/scan.py`, `resolve_targets`). Proposed: ask first, then resolve

---

## Superseded
| Decision | Where | Replaced by |
|---|---|---|
| "Which runs: those saved with Export (a record is written then)" | F2 | RD3: every successful training has a record |
| "Kept with the run: in the session until the next training; computed before Export, it is written in the TrainingRun record (and so in the bundle's copy of it)" | F3 | RD3 and RD4: a file per run and model, kept until the run is deleted; the bundle's copy of the record does not hold it |
| "A warning when one feature carries most of the effect: check that it is not a copy of the label" | F3 | HC2: an importance concentration warning, a label copy one possible cause |
| A heavy-job slot with a limit in the config | H, HR1 as first written | RD2: the training queue |
| A separate queue for the importance | F3d as first proposed | RD2 |
| "Nothing to NVD, AbuseIPDB or the AI before the validation, the limits, the authorization and the opt-in": the AI and AbuseIPDB only with `--ai` / `--abuseipdb` or the answer yes | the scan contract (v5.2 step 2.3a, `tests/recon/test_scan_contract.py`), steps 1.4c and A5 | RD5: on when the key is set, off with `--no-ai` / `--no-abuseipdb` or the answer no; still nothing before the validation, the limits and the authorization |
| "A failed lookup is recorded and tried again on the next host" | step 1.4e (`Intel.lookup`) | RD5: the service is not asked again in that scan; the next scan tries again |
| One setting that turns all the outside services off ("not offline") | HS4 as first written | RD5: a switch for each service; no setting for all |
