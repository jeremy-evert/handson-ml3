# 010 Run Prompt Candidate Review

## Short Summary

`tools/codex/run_prompt.py` appears to satisfy the intended V1 bridge-runner role: it executes one prompt, writes one durable execution record in `notes/`, separates execution from review, and stops at `review_status: UNREVIEWED`. The sample record is structurally close to the V1 artifact shape and preserves the failure evidence that matters. The most important next issue is environment-related, not runner design or broader implementation.

## Runner Assessment

### Does `tools/codex/run_prompt.py` successfully implement the V1 design intent?

Mostly yes.

It implements the core V1 intent described across the architecture and bridge-runner spec:

- one prompt execution
- one durable record in `notes/`
- stable run-oriented naming
- explicit execution facts
- explicit review facts initialized to `UNREVIEWED`
- preserved stderr and return code
- no queue automation or review automation

### What is strong about it?

- It stays thin and preserves the bounded behavior of `baby_run_prompt.py` instead of turning into a larger system.
- It cleanly separates `execution_status` from `review_status`, which is the main V1 workflow requirement.
- It records enough automatic evidence to support later review without introducing extra persistence layers or analysis logic.
- It keeps the runner output inspectable and deterministic.
- It preserves failure-path behavior well: even when Codex execution fails, the run still yields a reviewable artifact.

### Small correctness or clarity issues before further build-out

There are a few small issues, but none look like the highest-priority next step:

- `build_record_path(...)` appends `__2`, `__3`, etc. on collisions. That is practical, but it weakens the strict "`run_id` is `<prompt_stem>__<started_at_utc>`" identity rule described in the V1 artifact doc. This is a small spec/implementation mismatch, not a blocker.
- The runner identity now correctly points at `tools/codex/run_prompt.py`, but `tools/codex/V1_Bridge_Runner_Change_Spec.md` still describes the prior target as `baby_run_prompt.py`. That is a doc drift issue more than a runner issue.
- The record uses a `## Stderr` section rather than a literal `stderr_text` field. That still preserves the evidence and is consistent with the practical record shape, so this is more a clarity point than a correctness problem.

## Artifact Assessment

### Does `notes/001_smoke_test_pipeline__20260415_233343.md` match the intended V1 execution-record shape?

Yes, substantially.

It contains the expected V1 sections in the intended order:

1. header / identity
2. execution facts
3. review facts
4. failure analysis
5. resource / cost facts
6. prompt text
7. Codex final output
8. stderr

It also includes the minimum required fields that matter for review:

- `run_id`
- prompt identity
- `execution_status`
- `return_code`
- `review_status`
- metrics
- full prompt text
- captured stderr

### Is the separation between execution and review clear?

Yes.

The record makes the separation very clear:

- execution is marked `EXECUTION_FAILED`
- review remains `UNREVIEWED`

That is exactly the distinction V1 is supposed to preserve.

### Did the record preserve useful failure evidence?

Yes.

The stderr content is the most useful evidence in this run, and it was preserved intact enough to support diagnosis:

- PATH update warning
- session creation failure
- read-only filesystem error

That is enough to conclude the failure happened during Codex session initialization rather than during prompt-specific task execution.

## Operational Assessment

### Is the most important next issue design-related, implementation-related, or environment-related?

Environment-related.

The V1 design boundary is already demonstrated well enough for this stage, and the runner implementation is good enough to produce the intended artifact on both success and failure paths. What is blocking meaningful validation now is the Codex execution environment failing before the prompt can actually run.

### Does the read-only filesystem/session error appear to be mainly a runner problem or a Codex execution environment problem?

It appears mainly to be a Codex execution environment problem.

Reasons:

- The same runner flow successfully produced earlier smoke-test notes in prior runs.
- The new runner completed its own responsibilities: prompt resolution, subprocess invocation, artifact creation, stderr capture, and exit-code propagation.
- The recorded failure happens inside Codex session startup: `Failed to create session: Read-only file system (os error 30)`.
- Nothing in `run_prompt.py` suggests it is forcing a read-only mode or attempting forbidden writes beyond the existing thin subprocess path.

## Recommended Next Move

### Chosen next move: environment diagnosis prompt

This is the smallest and safest next move.

Why this should happen next:

- The runner already proves the V1 execution-record pattern well enough to justify holding off on further polish.
- A review write-back helper would optimize a manual step before the environment can reliably produce successful reviewed runs again.
- The observed failure is upstream of prompt execution and is currently the main obstacle to validating normal V1 operation.
- A focused diagnosis prompt can stay narrow: identify what Codex session initialization is trying to write, where the read-only path is, and whether the failure reproduces outside this runner.

What this next step should try to produce:

- a short, evidence-based diagnosis note
- the likely writable vs read-only path involved in session startup
- a conclusion about whether the issue is runner-independent
- one bounded follow-up action after diagnosis

What should wait:

- runner refactors
- workflow helpers for manual review write-back
- broader V1 build-out beyond diagnosis
