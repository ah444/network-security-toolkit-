# Templates

Copy and fill these in. Leave out a field only when it truly doesn't apply, and say so.

## Design card (before any code)

```markdown
### <step id>: <what it does, in a few words>
- **Assumptions:** <what you take as given>; <any other reading of the request, asked about>
- **Where:** <file(s), layer>; why there and not elsewhere
- **Reuses:** <existing functions, queues, helpers, by name>; what is new and why nothing existing fits
- **Lifecycle:** created when <…>; held by <…>; ends when <…>; on cancel <…>; on failure <…>; on crash or restart <…>
- **Concurrency:** what can run at the same time; what protects it (or why nothing needs to)
- **Trust and data:** what crosses a boundary; what is validated and where
- **User sees:** who uses it, through which interface; the steps; the states; the messages' wording; the confirmations; what changes for existing users
- **Failure modes:** <case> → <what the user sees>
- **Tests:** unit <…>; integration <…>; end-to-end or manual <…>
- **Out of scope:** <…>
- **Risks / open questions for the owner:** <…>
```

## Suggestion outside the scope

```markdown
**Suggestion (not needed for this request):** <what you noticed>
- **Evidence:** <file:line, version, advisory id>
- **Why it matters:** <the risk or the cost of leaving it>
- **Severity:** critical | high | medium | low
- **Options:** <option A and what it changes>; <option B>; leave it as it is
```

## Task (from a review or a gap)

```markdown
- [ ] **<ID>** <title>
  - **Problem:** <what is wrong today, with evidence>
  - **Severity:** critical | high | medium | low (as defined in the decisions file)
  - **Fix:** <the simplest change that closes the risk>
  - **To check first:** <facts not yet read in the code>
  - **Tests:** <by level>
  - **Done when:** <evidence that can be checked>
```

## Decision entry

```markdown
## <ID> <title>
Approved <date> by <who>. State: not implemented | implemented in <commit> | superseded by <ID>.

<the decision in two or three sentences>

Why: <the reason, and the alternative rejected>

Rules:
- <the conditions any implementation must keep>
```

## Severity

- **Critical:** a way to break in, lose or damage data, or results that mislead in a way that matters
- **High:** a real practical risk, or a core function that isn't safe
- **Medium:** a real problem with a workaround or a limited effect
- **Low:** an improvement that doesn't block use

## Reply to a review (in the owner's voice)

```markdown
Thanks for the review. I checked every claim against the code; this is the current state.

**What was wrong in our documents (admitted):** <point> — <evidence>
**What already exists and is tested:** <point> — <file / test>
**What was misread:** <point> — <evidence>
**What is right and not done yet:** <point> — stated as missing, no promise
**Where we see it differently:** <point> — <reason>
**Proposed (not implemented):** <design decisions for their opinion>
**Current state:** <what changed; what didn't>
```

## Status document outline

1. One paragraph: what it is and isn't
2. Numbers, each with how it was measured; breakdowns that add up
3. The features as they are today
4. The architecture, and which rules are enforced by tests
5. Security and privacy: each risk, its protection, and how that is known
6. What is verified, by level, and what is not
7. Results with their caveats
8. Known limits
