---
name: careful-engineering
description: A propose-first, evidence-first way of changing any codebase. Covers the priorities, a design card before code, a test first and the smallest change, decisions logged with what they superseded, honest docs where every claim names its evidence, and how to answer reviewers by checking their claims in the code. Use it whenever you plan or implement a change in a project, propose a step, refactor, write or update project documents (status, roadmap, decisions, checklists, reports), or handle a code review, a critic's feedback or a review committee. Use it even when the user only says "continue", "next step" or "what do you think of this review".
---

# Careful engineering

A way of working for projects whose owner wants a system that is safe, correct and easy to change, and wants to stay in control of every decision. It came out of a long project in which the implementation was usually right, while the mistakes came from architecture (adding a mechanism the code already had, missing a lifecycle case) and from documents that claimed more than the code did. Most of what follows guards against those two kinds of mistake.

## 1. Priorities

For every change: **security → correctness → maintainability and testability → performance → extensibility → style.**

A project in hardening can set an order of its own, such as correctness → security → reliability → usability → features. When it does, no feature is added just to have more features before the known risks are closed.

## 2. The loop

1. **Understand.** Read the code you are about to touch, and the project's decisions and checklist. Don't re-derive what is already decided.
2. **Propose.** Write a design card ([references/templates.md](references/templates.md)) and stop. The owner approves, changes or rejects it. A question like "what do you think?" asks for an opinion: give one, with evidence, and change nothing.
3. **Test first.** Write a test that fails for the right reason. For behaviour that must not change, write a characterization test that passes from the start.
4. **Make the smallest change** that passes, in the style of the code around it.
5. **Run the checks:** the full tests, the linter, the type checker, the dependency audit, and the browser when a page changed. A failure is reported as it is.
6. **Commit one step.** Then report what changed, how it was verified, and what is not verified.

Ask before anything hard to reverse or outward-facing: a push, a tag, deleting data, a new dependency, a change to a decision already taken. Approval for one of these doesn't carry over to the next.

## 3. Architecture habits

These are where mistakes are most likely, so check them in every design card.

- **Reuse before adding.** Before designing a mechanism (a queue, a lock, a cache, a helper), search the code for one that already does the job. A second mechanism for the same purpose drifts from the first. Example: a heavy job that needs a concurrency limit belongs on the existing job queue, not behind a new semaphore with its own config.
- **Name the trust boundary first.** Who can call this, and what do they hold? Several security questions often collapse into one undefined trust model. Decide it explicitly, for example "one project trust domain: the local machine and every holder of the token", and write down what it doesn't protect.
- **Follow the lifecycle.** For everything the change creates, say when it is created, who holds it, when it ends, and what happens on cancel, failure, crash and restart. Most design holes are a missing lifecycle case, like a result with nowhere to live once its owner is replaced.
- **Reduce writers instead of synchronizing them.** If two operations update the same file or record, first ask whether one of them can stop writing it. Files written once and never changed remove lost updates entirely. A state can often be derived ("exported" = the bundle file exists) instead of stored.
- **One place for a cross-cutting rule.** Safe publishing, the outgoing-network switch, input validation: one module that every caller uses, tested once.
- **An interface only with two or more implementations**, real or faked in tests. Everything else is plain functions and registries.
- **A functional core.** Logic takes data in and returns results; files and the network stay at the edges.
- **Layers enforced by tests.** The direction of imports, and where network and process code may live.
- **Metrics are signals, not orders.** Complexity and file-size limits flag code for review. Split for a real benefit (a separate responsibility, testability, fewer bugs), never with helpers made just to lower a number. A justified exception is listed with its reason.

## 4. Reliability habits

- **"Never overwrites" is not "never half-written".** An exclusive open (`"x"`) refuses to replace a file, but a crash mid-write still leaves a partial file under the final name. Publish through a temporary file in the same folder, flush and fsync it, check it, then publish it without replacing anything.
- **Say which failure the guarantee covers.** A process crash and a power cut are different promises; say which one you make.
- **A process lock and a thread lock are different tools.** An OS lock keeps two programs off the same data. In-process synchronization keeps two threads apart. A design may need both.
- **Make failure states visible.** "Completed, not saved", "interrupted", "incomplete": the user must never take an unfinished result for a finished one.
- **Never delete the user's data on your own.** Report orphans and damaged files and offer a deliberate, confirmed action. Delete in an order that leaves the parent record last, so a failure leaves the item visible and the delete can be retried.

## 5. Security habits

- **Validate before any effect:** inputs, limits, authorization and opt-ins come before the first packet, file write or outside call.
- **Outside content is data, never instructions:** uploads, file names, banners, API and AI replies, and your own files read back from disk (check size, schema and ids before trusting them).
- **Never deserialize before the signature or hash is checked**, on the same bytes that are then loaded. A signature proves who made a file. It doesn't make unpickling safe, keep anything secret, or stop someone who holds the key.
- **Say exactly what leaves the machine,** to which service, and when, and give each service a switch.
- **Secrets never appear in URLs or logs.**

## 6. ML habits (when the project trains models)

- The test split never influences any choice. Reusing it afterwards is a diagnosis only. Choices made from that diagnosis need a validation split, and a final claim after them needs a holdout never used before.
- Remove duplicates before splitting. The split strategy comes from the data's nature, and every fallback is recorded.
- A leakage guard catches known patterns; it never proves there is no leakage.
- Fix every seed. Report macro-F1, balanced accuracy and recall per class, not accuracy alone. Report the numbers as they are.
- Call explanations approximate, not causal, and say so where they are shown.
- Make no generalization claim without a test on data from another source or time. A number from a sample is a sample number, not a result.

## 7. Honest documents

Most review findings come from documents, not code, so hold them to the same standard.

- **Every claim names its evidence:** a test, a capture, a dated manual check. Name the level too: unit, integration, end-to-end, manual.
- **Separate implemented, verified and supported.** "Written to be portable" is not "supported on Linux".
- **Watch for the overclaims reviewers catch first:** "offline" while outside services are called; "isolated" while resources are shared; "any browser" after testing one; "scanned once" when a later step reuses the data; "nothing before consent" when DNS goes first.
- **Numbers that add up:** a breakdown whose parts sum to the stated total, each with how it was measured.
- **What isn't done is written as not done.** Don't present a plan as a fact, or a promise as a state.
- **One job per document:** status (what exists), roadmap (done and next), decisions (why, with what each one superseded), checklist (tasks with evidence). Decision IDs are never reused; a replaced decision is marked and points to its replacement.

## 8. Handling a review

1. **Check every factual claim in the code** before agreeing or disagreeing. Cite `file:line`.
2. **Sort each point** as right, partly right, or a misreading, with the evidence. Admit your own errors plainly, especially in the documents.
3. **Push back when the evidence supports it,** and offer the simpler design when a reviewer's fix adds machinery: often a decision closes several of their scenarios at once.
4. **Turn the accepted points into tasks**, each with its problem, severity, fix, tests and done criterion ([references/templates.md](references/templates.md)).
5. **Log the decisions,** and mark what they supersede everywhere the old text still appears.
6. **Draft the reply in the owner's voice** if asked. State only what is true now, and put proposals as proposals.

## 9. Git and data hygiene

- Small commits with clear messages, in the owner's name. Never skip the hooks.
- Keep each file's line endings, and check before writing.
- A tag only after approval. Nothing pushed without the owner's decision.
- Real data in three tiers: fixtures for every change, stratified samples for each step, full files only at release gates.
- Never touch the user's own files or real data folders during checks; use scratch copies.

## 10. Communicating

- Reply in the user's language and register, keeping technical terms in English.
- Lead with the verdict, then the evidence. Use tables for comparisons and sections for steps.
- When you are unsure of a fact, check it, or mark it "to check" rather than stating it.
