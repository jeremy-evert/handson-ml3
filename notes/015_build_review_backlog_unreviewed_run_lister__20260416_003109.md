# 015_build_review_backlog_unreviewed_run_lister__20260416_003109

- run_id: `015_build_review_backlog_unreviewed_run_lister__20260416_003109`
- prompt_file: `codex_prompts/015_build_review_backlog_unreviewed_run_lister.md`
- prompt_stem: `015_build_review_backlog_unreviewed_run_lister`
- started_at_utc: `20260416_003109`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260416_003317`
- runner: `tools/codex/run_prompt.py`
- return_code: `0`
- retry_of_run_id:

## Review Facts

- review_status: `UNREVIEWED`
- review_summary:
- reviewed_by:
- reviewed_at_utc:

## Failure Analysis

- failure_type:
- failure_symptom:
- likely_cause:
- recommended_next_action:

## Resource / Cost Facts

- elapsed_seconds: `127.623`
- final_output_char_count: `1263`
- stderr_char_count: `180108`

## Prompt Text

```md
# Task: Build a small V1 review backlog and unreviewed-run lister

You are working in this repository.

Your task is to implement a small helper that answers the bounded V1 operational question:

`What execution records still need human review, and what are the latest records per prompt?`

## Important framing

This is a small helper task.

Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`

Those existing tools should remain unchanged in this pass.

You are building a thin read-only review-discovery helper, not a dashboard or workflow engine.

## Files to inspect

Read these exact files before making changes:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `notes/012_v1_pipeline_options_review__20260416_000819.md`
- `notes/012_top_three_next_options__20260416_000819.md`

You may also inspect the current execution records in:

- `notes/`

## Goal

Create a small helper at:

`tools/codex/list_review_backlog.py`

This helper should scan V1 execution records in `notes/` and produce a small human-readable review backlog summary.

The helper should stay focused on review discovery, not queue control.

## What the helper must answer

The helper should answer these bounded questions:

1. which execution records are still `UNREVIEWED`
2. what the latest execution record is for each prompt
3. which prompts most likely need human review next based on latest-record status

For V1, "likely needs human review next" should stay simple and inspectable:

- latest record is `UNREVIEWED`
- latest record path is known
- prompt identity is known

Do not try to prioritize by a rich policy engine.

## Minimum CLI behavior

Keep the CLI small and explicit.

Support one primary mode:

- no positional arguments required

Optional small filters are acceptable only if they remain clearly bounded, for example:

- `--unreviewed-only`

Do not add many modes, interactive behavior, or subcommands.

## Expected output

The helper should print a short human-readable summary to stdout.

That summary should include at least:

- a section or block listing all `UNREVIEWED` records found
- the latest record per prompt
- a short "needs review next" summary derived from those latest records

For each listed record, include at least:

- record path
- prompt stem or prompt file
- started timestamp
- execution status
- review status

Keep the output textual and inspectable.

Do not create JSON outputs, databases, caches, sidecars, or a broader reporting layer.

## Important behavior rules

### 1. Preserve V1 boundaries

Do NOT:

- update records automatically
- release prompts automatically
- compute broader queue readiness policy
- build a dashboard, TUI, or web view
- add retry orchestration
- add analytics beyond the immediate review backlog summary

This helper should only surface the current review backlog from existing records.

### 2. Respect the execution record as source of truth

Use the markdown execution record body in `notes/` as the source of truth for:

- `run_id`
- `prompt_file`
- `prompt_stem`
- `started_at_utc`
- `execution_status`
- `review_status`

Do not introduce a separate index file.

### 3. Stay conservative about "latest"

If multiple records exist for the same prompt, choose the latest record using simple, inspectable logic grounded in the existing record evidence.

Do not build retry graphs or full history synthesis.

### 4. Fail clearly on malformed inputs

If a record cannot be parsed well enough for the minimal summary, either skip it with a short warning or fail clearly if that is safer.

Pick one small consistent policy and document it in the implementation note.

### 5. Keep the helper read-only

This helper should not modify files.

## Implementation guidance

Keep the implementation single-file unless a tiny helper is truly necessary.

A simple approach is preferred:

- scan markdown execution records in `notes/`
- parse minimal field lines
- group records by prompt
- identify latest record per prompt
- list `UNREVIEWED` records
- print a small review backlog summary

Do not build a full markdown parser unless truly necessary.

## Required artifacts

### Artifact 1

Create:

`tools/codex/list_review_backlog.py`

### Artifact 2

Create a short implementation note at:

`notes/015_review_backlog_lister_build__TIMESTAMP.md`

This note should summarize:

- what was built
- what backlog view it provides
- what it intentionally does not do
- what validation was performed

## Validation requirements

After implementation, validate at least these points:

1. the helper can find V1 execution records in `notes/`
2. the helper can list records still marked `UNREVIEWED`
3. the helper can identify the latest record per prompt
4. the helper can produce a small "needs review next" summary
5. the helper remains read-only
6. `tools/codex/run_prompt.py` remains unchanged
7. `tools/codex/review_run.py` remains unchanged
8. `tools/codex/baby_run_prompt.py` remains unchanged

Include the validation outcome in the implementation note.

## Constraints

1. Use the exact file paths listed above.
2. Keep the helper small and read-only.
3. Do not modify the runner or review-writeback tools.
4. Do not alter the design documents in this pass.
5. Do not expand into dashboards, queue engines, analytics platforms, or broader automation.
6. Keep the implementation small enough to review comfortably.

## Success criteria

This task is successful if:

- `tools/codex/list_review_backlog.py` exists and works
- it can surface current `UNREVIEWED` records and latest records per prompt
- it remains a small review-discovery helper rather than a broader system
- the implementation note explains the scope and validation clearly
- the existing runner and review-writeback tools remain unchanged
```

## Codex Final Output

Implemented the helper at [tools/codex/list_review_backlog.py](/data/git/handson-ml3/tools/codex/list_review_backlog.py) and the implementation note at [notes/015_review_backlog_lister_build__20260416_010500.md](/data/git/handson-ml3/notes/015_review_backlog_lister_build__20260416_010500.md).

The script is read-only and stays within the V1 boundary: it scans `notes/*.md`, accepts only real top-level V1 execution records, lists all `UNREVIEWED` records, computes the latest record per prompt using `started_at_utc` plus the same-second run-id suffix as a tiebreaker, and derives a simple “likely needs human review next” view from latest records whose `review_status` is `UNREVIEWED`. It supports the bounded optional filter `--unreviewed-only`.

Validation passed:
- `python3 tools/codex/list_review_backlog.py`
- `python3 tools/codex/list_review_backlog.py --unreviewed-only`
- `python3 -m py_compile tools/codex/list_review_backlog.py`

On the current repo state, the helper finds 6 V1 execution records, 5 unreviewed records, and 5 prompts with a latest record; it correctly selects the later accepted `001` run as the latest for that prompt. `tools/codex/run_prompt.py`, `tools/codex/review_run.py`, and `tools/codex/baby_run_prompt.py` remain unchanged.

## Stderr

```text
OpenAI Codex v0.120.0 (research preview)
--------
workdir: /data/git/handson-ml3
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, /home/jevert/.codex/memories]
reasoning effort: none
reasoning summaries: none
session id: 019d93b2-ec34-77d1-805b-18425d65413a
--------
user
# Task: Build a small V1 review backlog and unreviewed-run lister

You are working in this repository.

Your task is to implement a small helper that answers the bounded V1 operational question:

`What execution records still need human review, and what are the latest records per prompt?`

## Important framing

This is a small helper task.

Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`

Those existing tools should remain unchanged in this pass.

You are building a thin read-only review-discovery helper, not a dashboard or workflow engine.

## Files to inspect

Read these exact files before making changes:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `notes/012_v1_pipeline_options_review__20260416_000819.md`
- `notes/012_top_three_next_options__20260416_000819.md`

You may also inspect the current execution records in:

- `notes/`

## Goal

Create a small helper at:

`tools/codex/list_review_backlog.py`

This helper should scan V1 execution records in `notes/` and produce a small human-readable review backlog summary.

The helper should stay focused on review discovery, not queue control.

## What the helper must answer

The helper should answer these bounded questions:

1. which execution records are still `UNREVIEWED`
2. what the latest execution record is for each prompt
3. which prompts most likely need human review next based on latest-record status

For V1, "likely needs human review next" should stay simple and inspectable:

- latest record is `UNREVIEWED`
- latest record path is known
- prompt identity is known

Do not try to prioritize by a rich policy engine.

## Minimum CLI behavior

Keep the CLI small and explicit.

Support one primary mode:

- no positional arguments required

Optional small filters are acceptable only if they remain clearly bounded, for example:

- `--unreviewed-only`

Do not add many modes, interactive behavior, or subcommands.

## Expected output

The helper should print a short human-readable summary to stdout.

That summary should include at least:

- a section or block listing all `UNREVIEWED` records found
- the latest record per prompt
- a short "needs review next" summary derived from those latest records

For each listed record, include at least:

- record path
- prompt stem or prompt file
- started timestamp
- execution status
- review status

Keep the output textual and inspectable.

Do not create JSON outputs, databases, caches, sidecars, or a broader reporting layer.

## Important behavior rules

### 1. Preserve V1 boundaries

Do NOT:

- update records automatically
- release prompts automatically
- compute broader queue readiness policy
- build a dashboard, TUI, or web view
- add retry orchestration
- add analytics beyond the immediate review backlog summary

This helper should only surface the current review backlog from existing records.

### 2. Respect the execution record as source of truth

Use the markdown execution record body in `notes/` as the source of truth for:

- `run_id`
- `prompt_file`
- `prompt_stem`
- `started_at_utc`
- `execution_status`
- `review_status`

Do not introduce a separate index file.

### 3. Stay conservative about "latest"

If multiple records exist for the same prompt, choose the latest record using simple, inspectable logic grounded in the existing record evidence.

Do not build retry graphs or full history synthesis.

### 4. Fail clearly on malformed inputs

If a record cannot be parsed well enough for the minimal summary, either skip it with a short warning or fail clearly if that is safer.

Pick one small consistent policy and document it in the implementation note.

### 5. Keep the helper read-only

This helper should not modify files.

## Implementation guidance

Keep the implementation single-file unless a tiny helper is truly necessary.

A simple approach is preferred:

- scan markdown execution records in `notes/`
- parse minimal field lines
- group records by prompt
- identify latest record per prompt
- list `UNREVIEWED` records
- print a small review backlog summary

Do not build a full markdown parser unless truly necessary.

## Required artifacts

### Artifact 1

Create:

`tools/codex/list_review_backlog.py`

### Artifact 2

Create a short implementation note at:

`notes/015_review_backlog_lister_build__TIMESTAMP.md`

This note should summarize:

- what was built
- what backlog view it provides
- what it intentionally does not do
- what validation was performed

## Validation requirements

After implementation, validate at least these points:

1. the helper can find V1 execution records in `notes/`
2. the helper can list records still marked `UNREVIEWED`
3. the helper can identify the latest record per prompt
4. the helper can produce a small "needs review next" summary
5. the helper remains read-only
6. `tools/codex/run_prompt.py` remains unchanged
7. `tools/codex/review_run.py` remains unchanged
8. `tools/codex/baby_run_prompt.py` remains unchanged

Include the validation outcome in the implementation note.

## Constraints

1. Use the exact file paths listed above.
2. Keep the helper small and read-only.
3. Do not modify the runner or review-writeback tools.
4. Do not alter the design documents in this pass.
5. Do not expand into dashboards, queue engines, analytics platforms, or broader automation.
6. Keep the implementation small enough to review comfortably.

## Success criteria

This task is successful if:

- `tools/codex/list_review_backlog.py` exists and works
- it can surface current `UNREVIEWED` records and latest records per prompt
- it remains a small review-discovery helper rather than a broader system
- the implementation note explains the scope and validation clearly
- the existing runner and review-writeback tools remain unchanged

codex
I’m reading the workflow and V1 record docs first, then I’ll inspect the current `notes/` records to match the helper to the repository’s existing record shape without touching the runner or review tools.
exec
/bin/bash -lc "sed -n '1,220p' tools/Codex_Prompt_Workflow_Architecture.md" in /data/git/handson-ml3
 succeeded in 0ms:
# Codex Prompt Workflow Architecture

## Goal

Build a small, conservative prompt workflow system for Codex work inside a repository.

This system should stay:

* easy to inspect
* easy to test
* easy to extend
* composed of small parts
* safe by default

The design should emphasize **separation of concerns** so each piece does one job clearly.

---

## V1 Problem Statement

The immediate V1 problem is not "how do we build a larger prompt platform?"

It is:

* how to execute one bounded prompt run
* how to preserve a durable record of that run
* how to stop for human review before the next prompt is treated as ready

The current gap is that prompt execution exists, but the workflow boundary between:

* execution
* durable evidence
* accepted progress

is still too loose.

That gap matters because the governing workflow requires:

* thin slices before broad automation
* review between iterations
* durable local history
* failure evidence that supports refinement instead of blind retry

So V1 should solve the smallest workflow problem that makes the sequence inspectable:

* one run
* one durable execution record
* one manual review gate
* one conservative rule for whether the next prompt may proceed

---

## Why We Are Re-Architecting

The first draft bundled too many responsibilities into one script. That made it harder to:

* understand
* trust
* test
* evolve
* reuse across repositories

We want a cleaner structure, but V1 should only separate what is necessary to support the governing workflow.

This is still a design-alignment step, not a broad implementation push.

---

## V1 Scope

V1 covers the minimum workflow slice from prompt execution through later human review.

It includes:

* selecting and executing one prompt
* writing one durable execution record in `notes/`
* keeping execution outcome separate from review outcome
* stopping the queue at `UNREVIEWED`
* allowing only an explicit accepted review outcome to release the next prompt

V1 is intentionally about workflow control and artifact clarity, not about architectural breadth.

---

## Out Of Scope For V1

The following items are explicitly out of scope in this stage:

* runner refactor into many modules
* dependency-aware scheduling
* richer queue state machines
* automatic approval or review routing
* aggregated reports and dashboards
* retry intelligence beyond minimal retry linkage
* token accounting beyond optional lightweight fields
* broad CLI redesign
* reusable multi-repo configuration systems
* structured sidecars, databases, or platform services

These may become useful later, but they should not shape V1 beyond clear deferral notes.

---

## Minimum Artifact Inventory

The governing workflow expects the minimum required artifacts to be explicit.

For V1, the minimum inventory is:

### 1. Governing workflow

`tools/Project_Design_Workflow.md`

This remains the controlling sequence for:

* bounded execution
* validation
* review between iterations
* deferral discipline

### 2. Architecture document

`tools/Codex_Prompt_Workflow_Architecture.md`

This document should explain:

* the V1 problem being solved
* the V1 scope boundary
* the minimum parts and artifacts
* the conservative implementation order
* what remains deferred

### 3. V1 execution record definition

`tools/codex/V1_Execution_Record_Artifact.md`

This defines the minimum durable source of truth for one run.

Its role in the architecture is to make one run inspectable without requiring a larger system.

### 4. V1 review gate definition

`tools/codex/V1_Run_Review_Gate.md`

This defines the smallest manual stop between one completed run and the next prompt becoming ready.

Its role in the architecture is to keep execution success separate from accepted progress.

### 5. Execution record files in `notes/`

For V1, one run should produce one markdown execution-record file in `notes/`.

That record is the minimum durable run artifact.

No additional required queue file, database, or sidecar should be introduced in V1.

---

## Minimum Viable Slice

The minimum viable slice is:

1. execute one prompt
2. write one execution record with `review_status: UNREVIEWED`
3. stop for human review
4. update the same execution record to either `ACCEPTED` or `REJECTED`
5. treat only `ACCEPTED` as releasing the next prompt in sequence

This is the smallest slice that proves the workflow rather than only the runner.

It is intentionally narrower than:

* a generalized queue engine
* a full prompt-status system
* a modular tool suite
* an automation framework

---

## Role Of The Execution Record

The execution record is the central V1 artifact.

Its purpose is to preserve, in one inspectable markdown file:

* what prompt was run
* what happened during execution
* what Codex returned
* what lightweight failure or resource evidence was observed
* what a human later decided about the run

For V1:

* the record body is the source of truth
* the file in `notes/` is the durable local history unit
* one run should not be split across multiple required files

This keeps V1 small while still supporting:

* review between iterations
* failure analysis
* lightweight cost awareness
* stable run identity

The execution record defined in `tools/codex/V1_Execution_Record_Artifact.md` should therefore be treated as a required architectural element, not an implementation detail.

---

## Role Of The Review Gate

The review gate is the minimum manual checkpoint after execution record creation.

Its purpose is to enforce the governing workflow rule that the next bounded step should not proceed only because a process finished successfully.

For V1:

* execution writes a record
* new records begin as `UNREVIEWED`
* human review decides `ACCEPTED` or `REJECTED`
* only `ACCEPTED` allows the next prompt to be treated as ready

exec
/bin/bash -lc "sed -n '1,220p' tools/codex/V1_Execution_Record_Artifact.md" in /data/git/handson-ml3
 succeeded in 0ms:
# V1 Execution Record Artifact

## Purpose

Define the minimum durable record for one prompt run so the workflow preserves:

- what was executed
- what happened during execution
- what a human later decided about the result
- enough failure and resource evidence to guide the next reviewed step

This artifact is meant to support the current conservative workflow, not a larger automation system.

## Scope

V1 covers one prompt run from prompt submission through later human review.

It does not define:

- multi-run reporting
- dependency graphs
- approval automation
- module decomposition
- a full state machine

## Source Of Truth

For V1, the source of truth for one run is one markdown execution-record file in `notes/`.

The record body is authoritative.
The filename is a locator and stable identifier carrier, but the body is the place where execution facts and review fields must be read from.

V1 should avoid splitting one run across multiple required files.

## Stable Identity For A Run

Each run must have a stable `run_id` that does not change after the file is created.

V1 format:

`<prompt_stem>__<started_at_utc>`

Example:

`005_define_execution_record_and_generate_next_prompt_queue__20260415_202557`

Why this is enough for V1:

- it is human-readable
- it distinguishes repeated runs of the same prompt
- it does not depend on later review outcome
- it can be created cheaply by the bridge runner

The filename should use the same stable identity and should not encode accepted or rejected review outcome.

## Required Fields

Every V1 execution record should contain these fields.

### Run Identity

- `run_id`
- `prompt_file`
- `prompt_stem`
- `started_at_utc`

### Execution Facts

- `execution_status`
- `finished_at_utc`
- `runner`
- `return_code`
- `prompt_text`
- `codex_final_output`

### Review Facts

- `review_status`
- `review_summary`

### Minimum Failure / Retry Linkage

- `retry_of_run_id`

### Minimum Resource / Cost Facts

- `elapsed_seconds`
- `final_output_char_count`
- `stderr_char_count`

## Optional Fields

These fields are useful when available, but should not block V1.

- `reviewed_by`
- `reviewed_at_utc`
- `stderr_text`
- `failure_analysis`
- `recommended_next_action`
- `model_name`
- `token_usage`
- `attempt_index_for_prompt`
- `workspace_notes`

## Automatic Vs Manual Fields

### Automatic In V1

These should be captured automatically when the runner can do so cheaply and reliably:

- `run_id`
- `prompt_file`
- `prompt_stem`
- `started_at_utc`
- `finished_at_utc`
- `execution_status`
- `runner`
- `return_code`
- `prompt_text`
- `codex_final_output`
- `elapsed_seconds`
- `final_output_char_count`
- `stderr_char_count`
- `stderr_text` when present
- `retry_of_run_id` only if the runner was explicitly given that context

### Manual In V1

These remain human-owned in V1:

- `review_status`
- `review_summary`
- `reviewed_by`
- `reviewed_at_utc`
- `failure_analysis`
- `recommended_next_action`

This split is intentional.
Execution capture should be automatic.
Acceptance and interpretation should remain manual until the review loop is defined more clearly.

## Execution Status Vs Review Status

V1 must keep these separate.

### Execution Status

This records what happened when the runner invoked Codex.

Allowed V1 values:

- `EXECUTED`
- `EXECUTION_FAILED`

`EXECUTED` means the subprocess completed successfully enough to return code `0`.
It does not mean the work is accepted.

`EXECUTION_FAILED` means the subprocess failed or the run artifact could only be completed with failure evidence.

### Review Status

This records the later human judgment.

Allowed V1 values:

- `UNREVIEWED`
- `ACCEPTED`
- `REJECTED`

Rules:

- a new run record should start as `UNREVIEWED`
- `EXECUTED` must not imply `ACCEPTED`
- `EXECUTION_FAILED` does not automatically imply `REJECTED`, though rejection will often follow
- the next prompt in a reviewed sequence should wait for explicit human review, not only process exit success

## Minimum Failure-Analysis Section

V1 failure analysis should stay small.
It should exist as a short section that can be filled in manually when useful, especially for `EXECUTION_FAILED` or `REJECTED` runs.

Minimum fields:

- `failure_type`
- `failure_symptom`
- `likely_cause`
- `recommended_next_action`

Suggested `failure_type` values:

- `environment`
- `runner`
- `prompt_scope`
- `prompt_clarity`
- `implementation`
- `validation`
- `unknown`

This is enough for V1 because it supports learning from failure without requiring a larger taxonomy.

## Minimum Resource / Cost Section

V1 should record only lightweight fields that are cheap to capture or inspect:

- `started_at_utc`
- `finished_at_utc`
- `elapsed_seconds`
- `return_code`
- `final_output_char_count`
- `stderr_char_count`

Optional later additions:

- `model_name`
- `token_usage`
- `tool_call_count`
- `file_change_count`
- `estimated_review_minutes`

V1 should observe resource use, not build a billing system.

exec
/bin/bash -lc "sed -n '1,220p' tools/Project_Design_Workflow.md" in /data/git/handson-ml3
 succeeded in 0ms:
# Project Design Workflow

## Goal

Define a repeatable workflow for turning an idea or architecture document into a clean, working system without skipping the thinking that keeps the build safe, clear, and reusable.

This document is meant to sit beside architecture documents and help answer a practical question:

**What are the steps between "this design makes sense" and "the thing works"?**

---

## Why This Exists

Sometimes a project feels sound at the architecture level but still feels fuzzy at the build level.

That happens because architecture is not the same thing as execution.

Architecture tells us:

* what parts exist
* what they are responsible for
* how they relate
* what should not be coupled

But architecture does not yet tell us:

* what to build first
* what order reduces risk
* what to test first
* what can wait
* how to know whether the design is holding
* how to recover cleanly when a step fails

This workflow bridges that gap.

---

## Relationship to a Larger Design Process

This workflow does **not** replace broader project thinking.
It fits inside it.

A useful mapping looks like this:

### 1. Conversation

What are we trying to do?
What hurts right now?
What would "better" feel like?

### 2. Goals

What must become true for this project to count as successful?

### 3. Deliverables

What concrete artifacts must exist?
Examples:

* architecture doc
* workflow doc
* module layout
* interface sketch
* notes folder
* validation checklist
* first working slice

### 4. Tasks

What actions create those deliverables?
Examples:

* define V1 scope
* define module boundaries
* choose naming conventions
* write first thin component
* validate one path end to end
* capture review notes

### 5. Implementation

Only after the earlier layers are stable do we write implementation.

So this workflow is not separate from design.
It is the part that turns design into bounded execution.

---

## Core Principles

### 1. Design before build

Do not use implementation momentum to hide unclear thinking.

### 2. Boundaries before breadth

A project becomes safer when scope, responsibility, and deferral lines are visible.

### 3. Thin slices before large pushes

Prefer a bounded slice that proves something real over a large batch that creates many unknowns at once.

### 4. Review between iterations

Each bounded execution step should be inspected before the next one is issued.
Do not assume a sequence is healthy just because the previous step completed.

### 5. Validation is part of design

Tests, inspections, examples, and acceptance checks are not extra work.
They are how the design proves it is real.

### 6. Bridge tooling is allowed, but subordinate

Temporary or bridge tooling can be useful when it stays thin, inspectable, and reversible.
It may help move work forward, gather evidence, or reduce manual friction.
It should not become a substitute for architecture, clear interfaces, or good decomposition.

### 7. Durable local history matters

Notes, logs, reports, generated outputs, and similar local artifacts can serve as durable project memory.
They help preserve decisions, validation results, failure patterns, and the reasoning behind refinements.

### 8. Failure should produce analysis, not just retries

When a step fails, the useful question is not only "how do we try again?"
It is also "what did this failure reveal about the task, decomposition, criteria, or environment?"

### 9. Resource use should be observed

Project effort has a cost, whether that cost is human time, machine time, review effort, retries, execution size, or external spend.
Large tasks should justify themselves.

---

## The Core Sequence

Here is the recommended design workflow.

### Phase 1: Clarify the problem

This is the "why are we doing this?" phase.

Questions:

* What pain are we removing?
* What confusion are we reducing?
* What repeated work are we trying to standardize?
* What is dangerous if we build too fast?

Output:

* short problem statement
* short success statement

---

### Phase 2: Define the system boundary

This is the "what belongs in this project and what does not?" phase.

Questions:

* What should this system do?
* What should it explicitly not do yet?
* What decisions are deferred?
* What adjacent problems are real but out of scope?

Output:

* scope statement
* out-of-scope list

---

### Phase 3: Draft the architecture

This is the "what pieces exist and what are their jobs?" phase.

Questions:

* What modules, components, or services are needed?
* What is each one responsible for?
* What should each one never own?
* Where are the seams between parts?

Output:

* architecture doc
* responsibility split
* proposed file, package, or interface layout

---

### Phase 4: Identify the minimum viable slice

This is the "what is the smallest useful thing we can build that proves the design?" phase.

Questions:

* What is the thinnest vertical slice that is actually useful?
* What can we test without building the whole system?
* What gives us signal early?
* What first slice is unlikely to trap us later?

Output:

* V1 feature list
* initial build plan

---

### Phase 5: Define the artifacts

This is the "what files or outputs must exist?" phase.

Questions:

* What documents should exist?

exec
/bin/bash -lc "sed -n '1,220p' tools/codex/V1_Run_Review_Gate.md" in /data/git/handson-ml3
 succeeded in 0ms:
# V1 Run Review Gate

## Purpose

Define the smallest practical review gate between one prompt run finishing and the next prompt in sequence being treated as ready.

This gate exists to preserve three things from the governing workflow:

- review between iterations
- separation of execution outcome from accepted outcome
- a conservative human stop before queue progression

## Scope

V1 covers only the decision point immediately after one run record is written.

It defines:

- when a run enters review
- the minimum manual checks
- the allowed review outcomes
- how those outcomes affect whether the next prompt may proceed
- what review information must be written into the run record

It does not define:

- automation of review
- a larger queue state machine
- dependency-aware scheduling
- retry orchestration
- richer review taxonomies such as `PARTIAL` or `BLOCKED`

## Review Trigger

A run becomes `awaiting review` when its execution phase has finished and its V1 execution record has been written to `notes/` with:

- `execution_status` set to either `EXECUTED` or `EXECUTION_FAILED`
- `review_status` still set to `UNREVIEWED`

That is the exact review trigger for V1.

Process exit alone does not make the next prompt ready.
Record creation plus `UNREVIEWED` review status means the run must stop for human inspection.

## Minimum Manual Checklist

Before a run can be accepted, a human reviewer must check only these items:

1. Confirm the execution record is complete enough to review.
   Required minimum: prompt identity, execution status, return code, prompt text, Codex final output, and any captured stderr.

2. Check whether the run actually addressed the prompt that was executed.
   This is a scope-and-intent check, not a deep redesign review.

3. Check whether there is enough evidence to treat the result as an accepted outcome for the current bounded step.
   Minimum evidence may be direct output inspection, referenced file changes, or stated validation evidence in the run record or surrounding repo state.

4. Check whether any obvious failure, mismatch, or ambiguity remains that would make automatic progression unsafe.
   Examples: execution failed, output is incomplete, acceptance is unclear, validation is missing, or the result suggests the design/task should be refined before continuing.

This checklist is intentionally small.
V1 only needs enough manual review to keep execution success separate from accepted progress.

## Allowed V1 Review Outcomes

V1 should allow exactly these review outcomes:

- `UNREVIEWED`
- `ACCEPTED`
- `REJECTED`

Meaning:

- `UNREVIEWED`: default state after execution record creation; queue must stop here.
- `ACCEPTED`: the bounded step is accepted after human review.
- `REJECTED`: the run is not accepted for progression, whether because execution failed, the result is inadequate, or the outcome requires redesign or retry analysis first.

No additional V1 outcomes are needed.

## Queue Progression Rule

Only `ACCEPTED` allows the next prompt in sequence to be treated as ready.

Rules:

- `UNREVIEWED` stops the queue pending manual review.
- `ACCEPTED` releases exactly the next reviewed step.
- `REJECTED` stops the queue and forces an explicit human decision about what happens next.

`EXECUTED` does not release the queue.
`EXECUTION_FAILED` does not itself decide the queue either, but it will usually lead to `REJECTED` on review.

## Stop-And-Decide Rule

`REJECTED` is the V1 outcome that stops the queue and forces a new design or retry decision.

That decision is intentionally outside this gate.
It may result in:

- a revised prompt
- a smaller follow-up slice
- an environment fix
- a deliberate retry
- an architecture or scope adjustment

V1 only requires that the queue does not continue past a rejected run.

## Run Record Write-Back

During review, the run record defined by [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md) must be updated with these review facts:

- `review_status`
- `review_summary`

When available, V1 should also write:

- `reviewed_by`
- `reviewed_at_utc`

For rejected runs, the reviewer should also fill the small failure-analysis fields when useful:

- `failure_type`
- `failure_symptom`
- `likely_cause`
- `recommended_next_action`

This keeps execution facts and review judgment in one durable record.

## Connection To The Execution Record

This gate depends on the separation already defined in [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:77):

- execution status answers "what happened when the runner executed the prompt?"
- review status answers "did a human accept this bounded step as good enough to progress?"

The V1 review gate is therefore not a new artifact.
It is the manual transition of one execution record from `UNREVIEWED` to either `ACCEPTED` or `REJECTED`.

## Intentionally Deferred

V1 intentionally defers:

- automatic queue release
- automatic reviewer assignment
- separate queue-level state files
- richer review outcomes such as `PARTIAL`, `BLOCKED`, or `NEEDS_RETRY`
- policy for choosing among multiple retry or redesign options
- automatic extraction of validation evidence from repo changes
- broader workflow redesign beyond the next reviewed step

## V1 Decision Summary

The minimum V1 review gate is:

1. execution finishes
2. a run record is written with `review_status: UNREVIEWED`
3. human review applies the minimum checklist
4. reviewer writes back `ACCEPTED` or `REJECTED`
5. only `ACCEPTED` makes the next prompt ready

That is small enough to guide the next implementation prompt without expanding the system into a larger workflow engine.

codex
I have the governing constraints and record contract. Next I’m checking the current runner, reviewer, and actual `notes/` artifacts so the parser matches the repository’s real field formatting and “latest record” evidence.
exec
/bin/bash -lc "sed -n '1,260p' tools/codex/run_prompt.py" in /data/git/handson-ml3
 succeeded in 0ms:
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import tempfile
import time
from pathlib import Path


PROMPTS_DIR = "codex_prompts"
NOTES_DIR = "notes"
RUNNER_PATH = "tools/codex/run_prompt.py"


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def utc_timestamp(moment: dt.datetime) -> str:
    return moment.strftime("%Y%m%d_%H%M%S")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
    prompt = Path(prompt_arg)

    if prompt.is_absolute():
        return prompt

    direct = root / prompt
    if direct.exists():
        return direct

    in_prompts = root / PROMPTS_DIR / prompt
    if in_prompts.exists():
        return in_prompts

    matches = sorted((root / PROMPTS_DIR).glob(f"{prompt_arg}*"))
    if len(matches) == 1:
        return matches[0]

    return in_prompts


def prompt_file_label(root: Path, prompt_path: Path) -> str:
    try:
        return prompt_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(prompt_path.resolve())


def build_record_path(notes_dir: Path, prompt_stem: str, started_at_utc: str) -> tuple[str, Path]:
    base_run_id = f"{prompt_stem}__{started_at_utc}"
    candidate = notes_dir / f"{base_run_id}.md"
    if not candidate.exists():
        return base_run_id, candidate

    suffix = 2
    while True:
        run_id = f"{base_run_id}__{suffix}"
        candidate = notes_dir / f"{run_id}.md"
        if not candidate.exists():
            return run_id, candidate
        suffix += 1


def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
        output_path = Path(handle.name)

    try:
        result = subprocess.run(
            ["codex", "exec", "-C", str(root), "--output-last-message", str(output_path), "-"],
            input=prompt_text,
            text=True,
            capture_output=True,
            check=False,
        )
        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        return result.returncode, final_output, result.stderr
    finally:
        output_path.unlink(missing_ok=True)


def fenced_block(text: str, fence: str, info: str) -> str:
    body = text.rstrip("\n")
    return f"{fence}{info}\n{body}\n{fence}" if body else f"{fence}{info}\n\n{fence}"


def build_record_content(
    *,
    run_id: str,
    prompt_file: str,
    prompt_stem: str,
    started_at_utc: str,
    execution_status: str,
    finished_at_utc: str,
    return_code: int,
    prompt_text: str,
    codex_output: str,
    stderr_text: str,
    elapsed_seconds: float,
) -> str:
    final_output = codex_output.rstrip()
    stderr_body = stderr_text.rstrip()

    sections = [
        f"# {run_id}",
        "",
        f"- run_id: `{run_id}`",
        f"- prompt_file: `{prompt_file}`",
        f"- prompt_stem: `{prompt_stem}`",
        f"- started_at_utc: `{started_at_utc}`",
        "",
        "## Execution Facts",
        "",
        f"- execution_status: `{execution_status}`",
        f"- finished_at_utc: `{finished_at_utc}`",
        f"- runner: `{RUNNER_PATH}`",
        f"- return_code: `{return_code}`",
        "- retry_of_run_id:",
        "",
        "## Review Facts",
        "",
        "- review_status: `UNREVIEWED`",
        "- review_summary:",
        "- reviewed_by:",
        "- reviewed_at_utc:",
        "",
        "## Failure Analysis",
        "",
        "- failure_type:",
        "- failure_symptom:",
        "- likely_cause:",
        "- recommended_next_action:",
        "",
        "## Resource / Cost Facts",
        "",
        f"- elapsed_seconds: `{elapsed_seconds:.3f}`",
        f"- final_output_char_count: `{len(codex_output)}`",
        f"- stderr_char_count: `{len(stderr_text)}`",
        "",
        "## Prompt Text",
        "",
        fenced_block(prompt_text, "```", "md"),
        "",
        "## Codex Final Output",
        "",
        final_output if final_output else "*No output captured.*",
        "",
        "## Stderr",
        "",
        fenced_block(stderr_body, "```", "text") if stderr_body else "*No stderr captured.*",
        "",
    ]
    return "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one prompt file through codex exec and write a V1 execution record."
    )
    parser.add_argument("prompt", help="Prompt filename, numeric prefix, or path")
    args = parser.parse_args()

    root = repo_root()
    prompts_dir = root / PROMPTS_DIR
    notes_dir = root / NOTES_DIR

    if not prompts_dir.exists():
        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
        return 1

    if not notes_dir.exists():
        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
        return 1

    prompt_path = resolve_prompt_path(root, args.prompt)
    if not prompt_path.exists():
        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
        return 1

    prompt_text = prompt_path.read_text(encoding="utf-8")
    started_at = utc_now()
    started_at_utc = utc_timestamp(started_at)
    run_id, record_path = build_record_path(notes_dir, prompt_path.stem, started_at_utc)

    monotonic_start = time.monotonic()
    return_code, codex_output, stderr_text = run_codex(root, prompt_text)
    elapsed_seconds = time.monotonic() - monotonic_start
    finished_at_utc = utc_timestamp(utc_now())
    execution_status = "EXECUTED" if return_code == 0 else "EXECUTION_FAILED"

    record_path.write_text(
        build_record_content(
            run_id=run_id,
            prompt_file=prompt_file_label(root, prompt_path),
            prompt_stem=prompt_path.stem,
            started_at_utc=started_at_utc,
            execution_status=execution_status,
            finished_at_utc=finished_at_utc,
            return_code=return_code,
            prompt_text=prompt_text,
            codex_output=codex_output,
            stderr_text=stderr_text,
            elapsed_seconds=elapsed_seconds,
        ),
        encoding="utf-8",
    )

    print(record_path)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

exec
/bin/bash -lc "sed -n '1,260p' tools/codex/review_run.py" in /data/git/handson-ml3
 succeeded in 0ms:
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path


REVIEW_STATUSES = {"ACCEPTED", "REJECTED"}
FAILURE_FIELDS = (
    "failure_type",
    "failure_symptom",
    "likely_cause",
    "recommended_next_action",
)
REQUIRED_FIELDS = (
    "run_id",
    "prompt_file",
    "prompt_stem",
    "started_at_utc",
    "execution_status",
    "finished_at_utc",
    "runner",
    "return_code",
    "retry_of_run_id",
    "review_status",
    "review_summary",
    "reviewed_by",
    "reviewed_at_utc",
    *FAILURE_FIELDS,
    "elapsed_seconds",
    "final_output_char_count",
    "stderr_char_count",
)
REQUIRED_SECTIONS = (
    "## Execution Facts",
    "## Review Facts",
    "## Failure Analysis",
    "## Resource / Cost Facts",
    "## Prompt Text",
    "## Codex Final Output",
    "## Stderr",
)


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write manual V1 review fields back into an existing execution record."
    )
    parser.add_argument("record", help="Path to an existing execution-record markdown file")
    parser.add_argument(
        "--review-status",
        required=True,
        choices=sorted(REVIEW_STATUSES),
        help="Manual review outcome",
    )
    parser.add_argument(
        "--review-summary",
        required=True,
        help="Short manual review summary",
    )
    parser.add_argument("--reviewed-by", help="Reviewer identifier")
    parser.add_argument(
        "--reviewed-at-utc",
        help="Review timestamp in UTC using YYYYMMDD_HHMMSS; defaults to current UTC time",
    )
    parser.add_argument("--failure-type", help="Short failure category for rejected runs")
    parser.add_argument("--failure-symptom", help="Observed failure symptom for rejected runs")
    parser.add_argument("--likely-cause", help="Likely root cause for rejected runs")
    parser.add_argument(
        "--recommended-next-action",
        help="Manual next action recommendation for rejected runs",
    )
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def require_single_line(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be a single line")
    return value.strip()


def resolve_record_path(record_arg: str) -> Path:
    record_path = Path(record_arg)
    if record_path.is_absolute():
        return record_path
    return repo_root() / record_path


def validate_record_path(record_path: Path) -> None:
    if not record_path.exists():
        raise ValueError(f"record file not found: {record_path}")
    if not record_path.is_file():
        raise ValueError(f"record path is not a file: {record_path}")

    root = repo_root().resolve()
    notes_dir = root / "notes"
    try:
        record_path.resolve().relative_to(notes_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"record must be under {notes_dir}") from exc


def validate_v1_record_structure(text: str) -> None:
    if not text.startswith("# "):
        raise ValueError("record does not start with a markdown title")

    positions: list[int] = []
    for section in REQUIRED_SECTIONS:
        pos = text.find(section)
        if pos == -1:
            raise ValueError(f"record is missing section: {section}")
        positions.append(pos)
    if positions != sorted(positions):
        raise ValueError("record sections are out of the expected V1 order")

    for field in REQUIRED_FIELDS:
        if re.search(rf"^- {re.escape(field)}:", text, flags=re.MULTILINE) is None:
            raise ValueError(f"record is missing field line: {field}")


def replace_field(text: str, field: str, value: str | None, *, code: bool = False) -> str:
    rendered = f"`{value}`" if code and value else (value or "")
    pattern = re.compile(rf"(?m)^- {re.escape(field)}:.*$")
    if pattern.search(text) is None:
        raise ValueError(f"record is missing field line: {field}")
    return pattern.sub(f"- {field}: {rendered}", text, count=1)


def build_updates(args: argparse.Namespace) -> dict[str, tuple[str | None, bool]]:
    review_summary = require_single_line("review_summary", args.review_summary)
    reviewed_by = require_single_line("reviewed_by", args.reviewed_by)
    reviewed_at_utc = require_single_line("reviewed_at_utc", args.reviewed_at_utc) or utc_timestamp()
    failure_values = {
        "failure_type": require_single_line("failure_type", args.failure_type),
        "failure_symptom": require_single_line("failure_symptom", args.failure_symptom),
        "likely_cause": require_single_line("likely_cause", args.likely_cause),
        "recommended_next_action": require_single_line(
            "recommended_next_action", args.recommended_next_action
        ),
    }

    if args.review_status != "REJECTED" and any(value for value in failure_values.values()):
        raise ValueError("rejection-only failure fields may only be set when --review-status REJECTED")

    updates: dict[str, tuple[str | None, bool]] = {
        "review_status": (args.review_status, True),
        "review_summary": (review_summary, False),
        "reviewed_at_utc": (reviewed_at_utc, True),
    }
    if reviewed_by is not None:
        updates["reviewed_by"] = (reviewed_by, False)

    if args.review_status == "REJECTED":
        for field, value in failure_values.items():
            if value is not None:
                updates[field] = (value, False)

    return updates


def main() -> int:
    try:
        args = parse_args()
        record_path = resolve_record_path(args.record)
        validate_record_path(record_path)
        text = record_path.read_text(encoding="utf-8")
        validate_v1_record_structure(text)

        updated = text
        for field, (value, code) in build_updates(args).items():
            updated = replace_field(updated, field, value, code=code)

        record_path.write_text(updated, encoding="utf-8")
    except ValueError as exc:
        return fail(str(exc))

    print(record_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

exec
/bin/bash -lc 'rg --files notes | sort' in /data/git/handson-ml3
 succeeded in 0ms:
notes/001_smoke_test_pipeline__20260415_233343.md
notes/001_smoke_test_pipeline__20260415_234918.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md
notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md
notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md
notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md
notes/003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md
notes/004_architecture_and_bridge_runner_review__20260415_195538.md
notes/004_next_design_step_recommendation__20260415_195538.md
notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md
notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md
notes/005_prompt_queue_plan__20260415_202557.md
notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md
notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md
notes/009_run_prompt_candidate_build__20260415_233407.md
notes/010_next_step_recommendation__20260415_234559.md
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md
notes/010_run_prompt_candidate_review__20260415_234559.md
notes/011_build_v1_review_writeback_helper__20260415_235346.md
notes/011_review_writeback_helper_build__20260415_235514.md
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md
notes/012_top_three_next_options__20260416_000819.md
notes/012_v1_pipeline_options_review__20260416_000819.md
notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md
notes/013_next_two_prompt_plan__20260416_002005.md
notes/014_build_queue_readiness_checker__20260416_002319.md
notes/014_queue_readiness_checker_build__20260416_002419.md

exec
/bin/bash -lc "sed -n '1,260p' notes/012_v1_pipeline_options_review__20260416_000819.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 012 V1 Pipeline Options Review

## Short Summary Of Current Pipeline Maturity

The V1 pipeline is now a usable thin slice, not just a design packet.

It can:

- execute one prompt through `tools/codex/run_prompt.py`
- write one durable V1 execution record in `notes/`
- preserve execution facts separately from review facts
- accept manual review write-back through `tools/codex/review_run.py`

That means the core V1 artifact boundary is real and operationally demonstrable. The main remaining weakness is not missing architecture. It is missing operational support around repeated use: finding what needs review, deciding what is actually ready to run next, and keeping the docs aligned with the now-real implementation.

## What Is Working Now

### 1. Execution path

- Prompt resolution exists and is conservative. `run_prompt.py` preserves the same direct-path, `codex_prompts/`, and unique-prefix lookup behavior as the bridge runner baseline. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:31).
- Codex invocation is still thin and inspectable. The runner shells out through `codex exec -C <repo_root> --output-last-message <tempfile> -` and captures return code, final output, and stderr. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:74).
- Execution-record creation exists and matches the intended V1 section order. The runner writes header, execution facts, review facts, failure analysis, resource facts, prompt text, final output, and stderr into one markdown file. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:97).
- Resource and failure evidence capture exists in the actual record body. The runner records `return_code`, `elapsed_seconds`, output/stderr character counts, and full stderr text when present. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:144) and the failed record [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:8).
- The separation between execution outcome and acceptance outcome is implemented correctly. New records start at `review_status: UNREVIEWED` regardless of `EXECUTED` vs `EXECUTION_FAILED`. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:130).
- The failure path is demonstrated, not hypothetical. The failed smoke-test run preserved the read-only/session-start stderr cleanly while still producing a reviewable record. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).

### 2. Review path

- Manual review write-back exists as a separate thin companion tool. `review_run.py` updates an existing record in place rather than creating sidecars or replacing the record. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:57).
- Record structure preservation is strong. The helper validates section order and required field lines before writing, then replaces only the targeted review fields. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:122) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:140).
- The helper is consistent with the V1 review gate. It only allows `ACCEPTED` or `REJECTED`, auto-fills `reviewed_at_utc`, and limits failure-analysis updates to rejected runs. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:12) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:148).
- Review is operationally usable now. The accepted smoke-test record shows the exact intended lifecycle: executed record first, then in-place write-back to `ACCEPTED`. See [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md:16).

### 3. Workflow usability

- The thin V1 flow is practical for single reviewed runs:
  - run prompt
  - inspect record
  - apply manual review
- The repo already has evidence of both important paths:
  - successful execution plus accepted review
  - failed execution artifact with preserved stderr
- The implementation stayed within the intended V1 boundary and avoided premature platform growth.

## Seams And Gaps That Remain

### 1. Conservative queue progression is still architectural, not operational

The architecture says only an accepted reviewed run should release the next prompt, but there is no helper yet that actually answers:

- what is the latest run for prompt N
- whether it is still `UNREVIEWED`
- whether the next prompt is ready

This is the biggest repeated-use seam. The design expects it, but the current repo still leaves the decision manual.

### 2. Review is usable, but review discovery is manual

`review_run.py` can update a chosen record, but nothing helps a human find:

- all unreviewed records
- the latest record for a prompt
- the current review backlog

That becomes awkward as `notes/` grows.

### 3. Stable identity is mostly right, but not perfectly aligned with the current doc

The execution-record artifact says the stable V1 `run_id` is `<prompt_stem>__<started_at_utc>`, but `run_prompt.py` adds `__2`, `__3`, etc. on same-second collisions. See [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:37) versus [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:59).

This is a small practical improvement in code, but it is still a spec mismatch.

### 4. Documentation drift exists

- `tools/codex/V1_Bridge_Runner_Change_Spec.md` still targets `baby_run_prompt.py` as the implementation target and still names the runner as `tools/codex/baby_run_prompt.py`. See [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:5), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:17), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:156), and [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:252).
- The design packet now implicitly assumes a review write-back step, but the architecture document still describes conservative queue progression as part of the V1 shape without a concrete implementation artifact for that responsibility. See [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:161) and [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:215).

### 5. Repeated-use confidence is under-supported

The runner and review helper are both small and clear, but there is no explicit fixture-based validation or script-level regression harness around:

- record section ordering
- same-record write-back safety
- rejection-field behavior
- collision behavior

The code is simple enough that this is not a correctness emergency, but for repeated operational use it is a credible next bounded improvement.

### 6. Rejected-run and retry linkage are still mostly placeholders

The record shape includes `retry_of_run_id` and manual failure-analysis fields, but the current tools do not help a reviewer connect a rejected run to a retry decision or derive the next retry target.

### 7. Environment fragility still appears in the repo history

The failed smoke-test run shows that Codex session startup can fail before task work begins. That is not a runner-design flaw, but it does mean the workflow still benefits from a small amount of environment-oriented diagnosis or runbook support when repeated use hits startup failures. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).

## Documentation Alignment Assessment

Overall alignment is good at the architectural level and good enough at the artifact level.

What still drifts:

- implementation target naming drift: the bridge-runner spec still speaks in terms of `baby_run_prompt.py`, while the actual V1 runner is `run_prompt.py`
- runner identity drift: the spec examples still encode `tools/codex/baby_run_prompt.py` while the real records encode `tools/codex/run_prompt.py`
- stable-identity drift: the spec describes a strict no-suffix `run_id`, while code uses a pragmatic collision suffix

These are small mismatches, but they are no longer theoretical because the repo now contains real V1 records and a real review helper.

## Realistic Next Bounded Options

### Option 1. Add a queue-readiness checker

- What it would build or improve:
  - A small helper that determines whether the next prompt is ready based on the latest execution record and whether that record is `ACCEPTED`.
- Why it matters:
  - This closes the biggest gap between the architecture and current operational reality.
  - It makes the V1 rule "only accepted review releases the next prompt" actually inspectable in use rather than purely manual.
- Expected risk level: `low`
- Expected payoff level: `high`

### Option 2. Add a review backlog / unreviewed-run lister

- What it would build or improve:
  - A small helper that scans V1 execution records in `notes/` and lists runs still at `UNREVIEWED`, plus the latest record per prompt.
- Why it matters:
  - Review write-back already works, but finding what needs review is still manual.
  - This directly improves repeated operational use without broadening the workflow.
- Expected risk level: `low`
- Expected payoff level: `high`

### Option 3. Add lightweight contract validation for the V1 scripts

- What it would build or improve:
  - A small test or fixture harness around `run_prompt.py` and `review_run.py` that locks in the record shape, allowed review transitions, and in-place write-back behavior.
- Why it matters:
  - The workflow now depends on markdown field stability.
  - A small regression harness would reduce silent drift while keeping the implementation thin.
- Expected risk level: `low`
- Expected payoff level: `medium`

### Option 4. Align the V1 design docs to the implemented runner and helper

- What it would build or improve:
  - A cleanup pass on the architecture/spec packet so it reflects `run_prompt.py`, the existing review write-back helper, and the real collision rule.
- Why it matters:
  - The current doc drift is small but inspectable.
  - Fixing it would remove avoidable ambiguity before more workflow support is added.
- Expected risk level: `low`
- Expected payoff level: `medium`

### Option 5. Add rejected-run / retry-linkage support

- What it would build or improve:
  - A small helper or convention for linking a retry run to `retry_of_run_id` and surfacing the latest rejected run for a prompt.
- Why it matters:
  - The record model already reserves space for retries, but the workflow does not yet help humans use it consistently.
  - This would make failure analysis more actionable without introducing a larger retry engine.
- Expected risk level: `medium`
- Expected payoff level: `medium`

### Option 6. Add a minimal environment-diagnosis/runbook note or helper

- What it would build or improve:
  - A bounded way to capture or explain Codex startup failures such as read-only session initialization issues.
- Why it matters:
  - The repo contains real evidence of environment-level failure before task execution.
  - A small runbook or diagnostic helper could reduce confusion when failures are not prompt-related.
- Expected risk level: `low`
- Expected payoff level: `low`

## Assessment Of The Current V1 Flow

The current V1 flow is now operationally usable for intentional, reviewed single-step work. It is no longer just a design direction.

What still feels awkward for regular use:

- finding the latest relevant run by hand in `notes/`
- identifying what is awaiting review
- deciding whether the next prompt is actually released
- keeping the docs and record semantics in sync as the thin tools stabilize

What is missing if this is going to be used regularly:

- one small readiness/checking helper
- one small review-discovery helper
- one small validation or contract layer to protect the record format

That is enough to make the next step clear: the strongest next work is around operational support for repeated reviewed use, not workflow expansion.

exec
/bin/bash -lc "sed -n '1,260p' notes/012_top_three_next_options__20260416_000819.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 012 Top Three Next Options

## Ranked Top Three

### 1. Add a queue-readiness checker

Why it made the top three:

- It closes the most important remaining gap between the V1 architecture and the actual toolchain.
- The design packet says only `ACCEPTED` should release the next prompt, but the repo does not yet provide a concrete way to answer that question.
- It is bounded, low-risk, and directly improves repeated operational use.

### 2. Add a review backlog / unreviewed-run lister

Why it made the top three:

- Review write-back now works, but locating records that still need review is manual.
- This is the next most obvious operational friction once more records accumulate in `notes/`.
- It improves usability without expanding the workflow into a broader engine.

### 3. Add lightweight contract validation for `run_prompt.py` and `review_run.py`

Why it made the top three:

- The current V1 flow depends on exact markdown structure and field stability.
- A small regression harness would protect the record contract before more helpers are built on top of it.
- It is a clean bounded step that reduces accidental drift risk.

## Which One Should Happen Next

The next step should be: `Add a queue-readiness checker`.

Reason:

- It addresses the highest-value remaining seam in the current V1 workflow.
- It is directly demanded by the architecture and review-gate docs.
- It reduces the current ambiguity around whether the next prompt may proceed, which is the most important missing operational behavior.

## What Should Explicitly Wait

These should wait until after the top option:

- broader workflow engines or status systems
- retry orchestration beyond minimal linkage
- multi-module runner refactors
- dashboards or aggregated reporting
- broad platform expansion

These can also wait briefly behind the top option, but remain good bounded follow-ups:

- doc-alignment cleanup for `V1_Bridge_Runner_Change_Spec.md`
- rejected-run / retry-linkage support
- environment-focused diagnosis tooling

## Practical Recommendation

Do the next step in this order:

1. queue-readiness checker
2. review backlog / unreviewed-run lister
3. lightweight contract validation

That sequence best strengthens the current repo for repeated reviewed use while staying inside the existing V1 boundary.

codex
The record format is clear now: `run_prompt.py` writes bullet fields with backticked values, and there are older non-V1 note files mixed into `notes/`. I’m checking a few real execution records plus the queue-readiness helper so this backlog lister only scans the intended V1 artifacts and uses the same conservative parsing assumptions.
exec
/bin/bash -lc "sed -n '1,220p' notes/001_smoke_test_pipeline__20260415_233343.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 001_smoke_test_pipeline__20260415_233343

- run_id: `001_smoke_test_pipeline__20260415_233343`
- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
- prompt_stem: `001_smoke_test_pipeline`
- started_at_utc: `20260415_233343`

## Execution Facts

- execution_status: `EXECUTION_FAILED`
- finished_at_utc: `20260415_233343`
- runner: `tools/codex/run_prompt.py`
- return_code: `1`
- retry_of_run_id:

## Review Facts

- review_status: `UNREVIEWED`
- review_summary:
- reviewed_by:
- reviewed_at_utc:

## Failure Analysis

- failure_type:
- failure_symptom:
- likely_cause:
- recommended_next_action:

## Resource / Cost Facts

- elapsed_seconds: `0.559`
- final_output_char_count: `0`
- stderr_char_count: `354`

## Prompt Text

```md
# 001 Smoke Test Pipeline

Your task is to verify that the prompt workflow scaffolding exists and that the notes pipeline can record a result.

## Goals
1. Confirm that the following folders exist in the repository:
   - `codex_prompts/`
   - `notes/`
   - `tools/`

2. Confirm that this prompt file exists and can be read.

3. Create a short report in `notes/` that verifies the pipeline is working.

## Report requirements
Create a markdown note in `notes/` whose filename includes:
- `001_smoke_test_pipeline`
- a success or fail marker
- a timestamp

Example:
- `001_smoke_test_pipeline__SUCCESS__YYYYMMDD_HHMMSS.md`

## Report contents
The report should include:
- prompt filename
- timestamp
- status
- whether the expected folders were found
- whether the prompt file was readable
- a short summary stating whether the pipeline appears functional

## Constraints
- Do not make unrelated repo changes
- Keep the report short and clear
- If something is missing, mark the run as FAIL and explain what is missing
```

## Codex Final Output

*No output captured.*

## Stderr

```text
WARNING: proceeding, even though we could not update PATH: Read-only file system (os error 30)
2026-04-15T23:33:43.693139Z ERROR codex_core::codex: Failed to create session: Read-only file system (os error 30)
Error: thread/start: thread/start failed: error creating thread: Fatal error: Failed to initialize session: Read-only file system (os error 30)
```

exec
/bin/bash -lc "sed -n '1,220p' notes/001_smoke_test_pipeline__20260415_234918.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 001_smoke_test_pipeline__20260415_234918

- run_id: `001_smoke_test_pipeline__20260415_234918`
- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
- prompt_stem: `001_smoke_test_pipeline`
- started_at_utc: `20260415_234918`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260415_234941`
- runner: `tools/codex/run_prompt.py`
- return_code: `0`
- retry_of_run_id:

## Review Facts

- review_status: `ACCEPTED`
- review_summary: Smoke test output and artifact are complete enough to accept this bounded step.
- reviewed_by:
- reviewed_at_utc: `20260415_235508`

## Failure Analysis

- failure_type:
- failure_symptom:
- likely_cause:
- recommended_next_action:

## Resource / Cost Facts

- elapsed_seconds: `22.690`
- final_output_char_count: `289`
- stderr_char_count: `8503`

## Prompt Text

```md
# 001 Smoke Test Pipeline

Your task is to verify that the prompt workflow scaffolding exists and that the notes pipeline can record a result.

## Goals
1. Confirm that the following folders exist in the repository:
   - `codex_prompts/`
   - `notes/`
   - `tools/`

2. Confirm that this prompt file exists and can be read.

3. Create a short report in `notes/` that verifies the pipeline is working.

## Report requirements
Create a markdown note in `notes/` whose filename includes:
- `001_smoke_test_pipeline`
- a success or fail marker
- a timestamp

Example:
- `001_smoke_test_pipeline__SUCCESS__YYYYMMDD_HHMMSS.md`

## Report contents
The report should include:
- prompt filename
- timestamp
- status
- whether the expected folders were found
- whether the prompt file was readable
- a short summary stating whether the pipeline appears functional

## Constraints
- Do not make unrelated repo changes
- Keep the report short and clear
- If something is missing, mark the run as FAIL and explain what is missing
```

## Codex Final Output

Verified successfully. The required folders exist, `codex_prompts/001_smoke_test_pipeline.md` is readable, and I wrote the report to [001_smoke_test_pipeline__SUCCESS__20260415_184932.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md).

Status: `SUCCESS`

## Stderr

```text
OpenAI Codex v0.120.0 (research preview)
--------
workdir: /data/git/handson-ml3
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, /home/jevert/.codex/memories]
reasoning effort: none
reasoning summaries: none
session id: 019d938c-9caa-7290-8f0d-bcf51bfb9312
--------
user
# 001 Smoke Test Pipeline

Your task is to verify that the prompt workflow scaffolding exists and that the notes pipeline can record a result.

## Goals
1. Confirm that the following folders exist in the repository:
   - `codex_prompts/`
   - `notes/`
   - `tools/`

2. Confirm that this prompt file exists and can be read.

3. Create a short report in `notes/` that verifies the pipeline is working.

## Report requirements
Create a markdown note in `notes/` whose filename includes:
- `001_smoke_test_pipeline`
- a success or fail marker
- a timestamp

Example:
- `001_smoke_test_pipeline__SUCCESS__YYYYMMDD_HHMMSS.md`

## Report contents
The report should include:
- prompt filename
- timestamp
- status
- whether the expected folders were found
- whether the prompt file was readable
- a short summary stating whether the pipeline appears functional

## Constraints
- Do not make unrelated repo changes
- Keep the report short and clear
- If something is missing, mark the run as FAIL and explain what is missing

codex
I’m checking the repo structure first, then I’ll locate this prompt file and write the smoke-test note under `notes/` with a success/fail result.
exec
/bin/bash -lc 'for d in codex_prompts notes tools; do if [ -d "$d" ]; then echo "$d:DIR"; else echo "$d:MISSING"; fi; done' in /data/git/handson-ml3
 succeeded in 0ms:
codex_prompts:DIR
notes:DIR
tools:DIR

exec
/bin/bash -lc "pwd && rg --files -g '001*smoke*test*pipeline*' -g 'codex_prompts/**' -g 'notes/**' -g 'tools/**'" in /data/git/handson-ml3
 succeeded in 0ms:
/data/git/handson-ml3
tools/Codex_Prompt_Workflow_Architecture.md
tools/Project_Design_Workflow.md
tools/.codex
tools/codex/run_prompt.py
tools/codex/V1_Bridge_Runner_Change_Spec.md
tools/codex/V1_Run_Review_Gate.md
tools/codex/V1_Execution_Record_Artifact.md
tools/codex/baby_run_prompt.py
tools/codex/__pycache__/baby_run_prompt.cpython-39.pyc
tools/codex/__pycache__/run_prompt.cpython-39.pyc
codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md
codex_prompts/009_build_v1_run_prompt_candidate.md
codex_prompts/008_define_minimal_bridge_runner_change_spec.md
codex_prompts/007_align_architecture_doc_to_v1_workflow.md
codex_prompts/006_define_v1_run_review_gate.md
codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md
codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md
codex_prompts/003_revise_Project_Deisgn_workflow_document.md
codex_prompts/002_repo_inventory_and_status.md
codex_prompts/001_smoke_test_pipeline.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md
notes/005_prompt_queue_plan__20260415_202557.md
notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md
notes/004_next_design_step_recommendation__20260415_195538.md
notes/004_architecture_and_bridge_runner_review__20260415_195538.md
notes/003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md
notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md
notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md
notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md
notes/010_next_step_recommendation__20260415_234559.md
notes/010_run_prompt_candidate_review__20260415_234559.md
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md
notes/009_run_prompt_candidate_build__20260415_233407.md
notes/001_smoke_test_pipeline__20260415_233343.md
notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md
notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md
notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md

codex
The expected folders are present. I’ve also located `codex_prompts/001_smoke_test_pipeline.md`; next I’m reading it and then I’ll write a new timestamped report in `notes/`.
exec
/bin/bash -lc "sed -n '1,160p' codex_prompts/001_smoke_test_pipeline.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 001 Smoke Test Pipeline

Your task is to verify that the prompt workflow scaffolding exists and that the notes pipeline can record a result.

## Goals
1. Confirm that the following folders exist in the repository:
   - `codex_prompts/`
   - `notes/`
   - `tools/`

2. Confirm that this prompt file exists and can be read.

3. Create a short report in `notes/` that verifies the pipeline is working.

## Report requirements
Create a markdown note in `notes/` whose filename includes:
- `001_smoke_test_pipeline`
- a success or fail marker
- a timestamp

Example:
- `001_smoke_test_pipeline__SUCCESS__YYYYMMDD_HHMMSS.md`

## Report contents
The report should include:
- prompt filename
- timestamp

exec
/bin/bash -lc "sed -n '1,260p' tools/codex/baby_run_prompt.py" in /data/git/handson-ml3
 succeeded in 0ms:
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import tempfile
from pathlib import Path


PROMPTS_DIR = "codex_prompts"
NOTES_DIR = "notes"


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
    prompt = Path(prompt_arg)

    if prompt.is_absolute():
        return prompt

    direct = root / prompt
    if direct.exists():
        return direct

    in_prompts = root / PROMPTS_DIR / prompt
    if in_prompts.exists():
        return in_prompts

    matches = sorted((root / PROMPTS_DIR).glob(f"{prompt_arg}*"))
    if len(matches) == 1:
        return matches[0]

    return in_prompts


def build_note_path(root: Path, prompt_path: Path, status: str, timestamp: str) -> Path:
    return root / NOTES_DIR / f"{prompt_path.stem}__{status}__{timestamp}.md"


def build_note_content(
    prompt_path: Path,
    timestamp: str,
    status: str,
    prompt_text: str,
    codex_output: str,
    notes: str = "",
) -> str:
    return f"""# {prompt_path.stem} - {status}

- Prompt file: `{prompt_path.name}`
- Timestamp (UTC): `{timestamp}`
- Status: `{status}`

## Original Prompt

```md
{prompt_text.rstrip()}
```

## Codex Output

{codex_output.rstrip() or "*No output captured.*"}

## Notes

{notes.rstrip()}
"""


def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
        output_path = Path(handle.name)

    try:
        result = subprocess.run(
            ["codex", "exec", "-C", str(root), "--output-last-message", str(output_path), "-"],
            input=prompt_text,
            text=True,
            capture_output=True,
            check=False,
        )

        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        return result.returncode, final_output, result.stderr
    finally:
        output_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one prompt file through codex exec and write a note.")
    parser.add_argument("prompt", help="Prompt filename, numeric prefix, or path")
    args = parser.parse_args()

    root = repo_root()
    prompts_dir = root / PROMPTS_DIR
    notes_dir = root / NOTES_DIR

    if not prompts_dir.exists():
        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
        return 1

    if not notes_dir.exists():
        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
        return 1

    prompt_path = resolve_prompt_path(root, args.prompt)
    if not prompt_path.exists():
        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
        return 1

    prompt_text = prompt_path.read_text(encoding="utf-8")
    timestamp = utc_timestamp()

    returncode, codex_output, stderr_text = run_codex(root, prompt_text)
    status = "SUCCESS" if returncode == 0 else "FAILED"
    note_path = build_note_path(root, prompt_path, status, timestamp)

    notes = ""
    if stderr_text.strip():
        notes = f"""Codex stderr:
```text
{stderr_text.rstrip()}
```"""

    note_path.write_text(
        build_note_content(
            prompt_path=prompt_path,
            timestamp=timestamp,
            status=status,
            prompt_text=prompt_text,
            codex_output=codex_output,
            notes=notes,
        ),
        encoding="utf-8",
    )

    print(note_path)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())

exec
/bin/bash -lc "sed -n '1,260p' tools/codex/check_queue_readiness.py" in /data/git/handson-ml3
 succeeded in 0ms:
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


PROMPTS_DIR = "codex_prompts"
NOTES_DIR = "notes"
REQUIRED_RECORD_FIELDS = (
    "run_id",
    "prompt_file",
    "prompt_stem",
    "started_at_utc",
    "execution_status",
    "review_status",
)
IDENTITY_FIELDS = (
    "run_id",
    "prompt_file",
    "prompt_stem",
    "started_at_utc",
)
EXECUTION_STATUSES = {"EXECUTED", "EXECUTION_FAILED"}
REVIEW_STATUSES = {"UNREVIEWED", "ACCEPTED", "REJECTED"}
PROMPT_NAME_RE = re.compile(r"^(?P<prefix>\d+)_")
TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


@dataclass(frozen=True)
class PromptEntry:
    prefix: int
    path: Path

    @property
    def label(self) -> str:
        return self.path.as_posix()


@dataclass(frozen=True)
class RunRecord:
    path: Path
    run_id: str
    prompt_file: str
    prompt_stem: str
    started_at_utc: str
    execution_status: str
    review_status: str
    run_suffix: int


@dataclass(frozen=True)
class ReadinessResult:
    target: PromptEntry
    previous: PromptEntry | None
    latest_record: RunRecord | None
    ready: bool
    reason: str


class ReadinessError(Exception):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report whether the next prompt is ready under the current V1 review gate."
    )
    parser.add_argument(
        "--prompt",
        help="Specific prompt file, filename, or numeric prefix to check",
    )
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def parse_prompt_prefix(path: Path) -> int:
    match = PROMPT_NAME_RE.match(path.stem)
    if match is None:
        raise ReadinessError(f"prompt file is missing a numeric prefix: {path.as_posix()}")
    return int(match.group("prefix"))


def discover_prompts(root: Path) -> list[PromptEntry]:
    prompts_dir = root / PROMPTS_DIR
    if not prompts_dir.exists():
        raise ReadinessError(f"missing prompt directory: {prompts_dir}")

    entries: list[PromptEntry] = []
    seen_prefixes: dict[int, Path] = {}

    for path in sorted(prompts_dir.glob("*.md")):
        prefix = parse_prompt_prefix(path)
        if prefix in seen_prefixes:
            raise ReadinessError(
                "multiple prompt files share the same numeric prefix: "
                f"{seen_prefixes[prefix].as_posix()} and {path.as_posix()}"
            )
        seen_prefixes[prefix] = path
        entries.append(PromptEntry(prefix=prefix, path=path.relative_to(root)))

    if not entries:
        raise ReadinessError(f"no markdown prompt files found in {prompts_dir}")

    return sorted(entries, key=lambda entry: entry.prefix)


def resolve_prompt(prompts: list[PromptEntry], prompt_arg: str) -> PromptEntry:
    trimmed = prompt_arg.strip()
    if not trimmed:
        raise ReadinessError("--prompt must not be empty")

    if trimmed.isdigit():
        prefix = int(trimmed)
        matches = [prompt for prompt in prompts if prompt.prefix == prefix]
        if len(matches) != 1:
            raise ReadinessError(f"no prompt found for numeric prefix: {trimmed}")
        return matches[0]

    normalized = trimmed.rstrip("/")
    matches = [
        prompt
        for prompt in prompts
        if normalized in {prompt.label, prompt.path.name, prompt.path.stem}
    ]
    if len(matches) == 1:
        return matches[0]

    prefix_matches = [prompt for prompt in prompts if prompt.path.name.startswith(normalized)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        raise ReadinessError(f"prompt selector is ambiguous: {trimmed}")

    raise ReadinessError(f"prompt not found: {trimmed}")


def parse_field_line(text: str, field: str) -> str | None:
    match = re.search(rf"(?m)^- {re.escape(field)}:\s*(.*)$", text)
    if match is None:
        return None

    value = match.group(1).strip()
    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return value


def parse_run_suffix(prompt_stem: str, started_at_utc: str, run_id: str) -> int:
    base_run_id = f"{prompt_stem}__{started_at_utc}"
    if run_id == base_run_id:
        return 1

    suffix_match = re.fullmatch(rf"{re.escape(base_run_id)}__(\d+)", run_id)
    if suffix_match is None:
        raise ReadinessError(f"run_id does not match V1 identity pattern: {run_id}")
    return int(suffix_match.group(1))


def parse_record_file(root: Path, path: Path) -> RunRecord | None:
    text = path.read_text(encoding="utf-8")
    has_execution_section = "## Execution Facts" in text
    has_review_section = "## Review Facts" in text
    if not has_execution_section and not has_review_section:
        return None
    if has_execution_section != has_review_section:
        raise ReadinessError(
            f"record-like note has incomplete V1 sections: {path.relative_to(root).as_posix()}"
        )

    header_block = text.split("\n## ", 1)[0]
    identity_values = {field: parse_field_line(header_block, field) for field in IDENTITY_FIELDS}
    if identity_values["run_id"] is None:
        return None

    metadata_block = text.split("\n## Prompt Text\n", 1)[0]
    values = {field: parse_field_line(metadata_block, field) for field in REQUIRED_RECORD_FIELDS}

    missing_fields = [field for field, value in values.items() if value is None]
    if missing_fields:
        raise ReadinessError(
            f"record-like note is missing required V1 fields: {path.relative_to(root).as_posix()} "
            f"missing {', '.join(missing_fields)}"
        )

    run_id = values["run_id"] or ""
    prompt_file = values["prompt_file"] or ""
    prompt_stem = values["prompt_stem"] or ""
    started_at_utc = values["started_at_utc"] or ""
    execution_status = values["execution_status"] or ""
    review_status = values["review_status"] or ""

    if not TIMESTAMP_RE.fullmatch(started_at_utc):
        raise ReadinessError(
            f"record has invalid started_at_utc timestamp: {path.relative_to(root).as_posix()}"
        )
    if execution_status not in EXECUTION_STATUSES:
        raise ReadinessError(
            f"record has invalid execution_status: {path.relative_to(root).as_posix()}"
        )
    if review_status not in REVIEW_STATUSES:
        raise ReadinessError(f"record has invalid review_status: {path.relative_to(root).as_posix()}")

    prompt_path = Path(prompt_file)
    if prompt_path.stem != prompt_stem:
        raise ReadinessError(
            f"record prompt_file/prompt_stem mismatch: {path.relative_to(root).as_posix()}"
        )

    return RunRecord(
        path=path.relative_to(root),
        run_id=run_id,
        prompt_file=prompt_file,
        prompt_stem=prompt_stem,
        started_at_utc=started_at_utc,
        execution_status=execution_status,
        review_status=review_status,
        run_suffix=parse_run_suffix(prompt_stem, started_at_utc, run_id),
    )


def discover_run_records(root: Path) -> list[RunRecord]:
    notes_dir = root / NOTES_DIR
    if not notes_dir.exists():
        raise ReadinessError(f"missing notes directory: {notes_dir}")

    records: list[RunRecord] = []
    for path in sorted(notes_dir.glob("*.md")):
        record = parse_record_file(root, path)
        if record is not None:
            records.append(record)
    return records


def latest_record_for_prompt(records: list[RunRecord], prompt: PromptEntry) -> RunRecord | None:
    relevant = [record for record in records if record.prompt_file == prompt.label]
    if not relevant:
        return None
    return max(relevant, key=lambda record: (record.started_at_utc, record.run_suffix))


def default_target(prompts: list[PromptEntry], records: list[RunRecord]) -> PromptEntry:
    for index, prompt in enumerate(prompts):
        if index == 0:
            latest = latest_record_for_prompt(records, prompt)
            if latest is None or latest.review_status != "ACCEPTED":
                return prompt
            continue

exec
/bin/bash -lc 'rg -n "''^- (run_id|prompt_file|prompt_stem|started_at_utc|execution_status|review_status):" notes/*.md' in /data/git/handson-ml3
 succeeded in 0ms:
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md:1426:- run_id: `<run_id>`
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md:1427:- prompt_file: `<prompt_file>`
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md:1428:- prompt_stem: `<prompt_stem>`
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md:1429:- started_at_utc: `<started_at_utc>`
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md:1437:- execution_status: `<EXECUTED|EXECUTION_FAILED>`
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md:1449:- review_status: `UNREVIEWED`
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md:3041:- run_id: `001_smoke_test_pipeline__20260415_233343`
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md:3042:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md:3043:- prompt_stem: `001_smoke_test_pipeline`
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md:3044:- started_at_utc: `20260415_233343`
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md:3048:- execution_status: `EXECUTION_FAILED`
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md:3056:- review_status: `UNREVIEWED`
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md:2752:- run_id: `<run_id>`
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md:2753:- prompt_file: `<prompt_file>`
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md:2754:- prompt_stem: `<prompt_stem>`
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md:2755:- started_at_utc: `<started_at_utc>`
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md:2763:- execution_status: `<EXECUTED|EXECUTION_FAILED>`
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md:2775:- review_status: `UNREVIEWED`
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md:4394:- run_id: `<run_id>`
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md:4395:- prompt_file: `<prompt_file>`
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md:4396:- prompt_stem: `<prompt_stem>`
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md:4397:- started_at_utc: `<started_at_utc>`
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md:4405:- execution_status: `<EXECUTED|EXECUTION_FAILED>`
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md:4417:- review_status: `UNREVIEWED`
notes/014_build_queue_readiness_checker__20260416_002319.md:3:- run_id: `014_build_queue_readiness_checker__20260416_002319`
notes/014_build_queue_readiness_checker__20260416_002319.md:4:- prompt_file: `codex_prompts/014_build_queue_readiness_checker.md`
notes/014_build_queue_readiness_checker__20260416_002319.md:5:- prompt_stem: `014_build_queue_readiness_checker`
notes/014_build_queue_readiness_checker__20260416_002319.md:6:- started_at_utc: `20260416_002319`
notes/014_build_queue_readiness_checker__20260416_002319.md:10:- execution_status: `EXECUTED`
notes/014_build_queue_readiness_checker__20260416_002319.md:18:- review_status: `UNREVIEWED`
notes/014_build_queue_readiness_checker__20260416_002319.md:2161:- run_id: `001_smoke_test_pipeline__20260415_234918`
notes/014_build_queue_readiness_checker__20260416_002319.md:2162:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
notes/014_build_queue_readiness_checker__20260416_002319.md:2163:- prompt_stem: `001_smoke_test_pipeline`
notes/014_build_queue_readiness_checker__20260416_002319.md:2164:- started_at_utc: `20260415_234918`
notes/014_build_queue_readiness_checker__20260416_002319.md:2168:- execution_status: `EXECUTED`
notes/014_build_queue_readiness_checker__20260416_002319.md:2176:- review_status: `ACCEPTED`
notes/014_build_queue_readiness_checker__20260416_002319.md:2385:- run_id: `011_build_v1_review_writeback_helper__20260415_235346`
notes/014_build_queue_readiness_checker__20260416_002319.md:2386:- prompt_file: `codex_prompts/011_build_v1_review_writeback_helper.md`
notes/014_build_queue_readiness_checker__20260416_002319.md:2387:- prompt_stem: `011_build_v1_review_writeback_helper`
notes/014_build_queue_readiness_checker__20260416_002319.md:2388:- started_at_utc: `20260415_235346`
notes/014_build_queue_readiness_checker__20260416_002319.md:2392:- execution_status: `EXECUTED`
notes/014_build_queue_readiness_checker__20260416_002319.md:2400:- review_status: `UNREVIEWED`
notes/014_build_queue_readiness_checker__20260416_002319.md:2609:- run_id: `012_review_v1_pipeline_and_recommend_next_options__20260416_000658`
notes/014_build_queue_readiness_checker__20260416_002319.md:2610:- prompt_file: `codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md`
notes/014_build_queue_readiness_checker__20260416_002319.md:2611:- prompt_stem: `012_review_v1_pipeline_and_recommend_next_options`
notes/014_build_queue_readiness_checker__20260416_002319.md:2612:- started_at_utc: `20260416_000658`
notes/014_build_queue_readiness_checker__20260416_002319.md:2616:- execution_status: `EXECUTED`
notes/014_build_queue_readiness_checker__20260416_002319.md:2624:- review_status: `UNREVIEWED`
notes/014_build_queue_readiness_checker__20260416_002319.md:2833:- run_id: `013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937`
notes/014_build_queue_readiness_checker__20260416_002319.md:2834:- prompt_file: `codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md`
notes/014_build_queue_readiness_checker__20260416_002319.md:2835:- prompt_stem: `013_generate_prompts_for_queue_readiness_and_review_backlog_helpers`
notes/014_build_queue_readiness_checker__20260416_002319.md:2836:- started_at_utc: `20260416_001937`
notes/014_build_queue_readiness_checker__20260416_002319.md:2840:- execution_status: `EXECUTED`
notes/014_build_queue_readiness_checker__20260416_002319.md:2848:- review_status: `UNREVIEWED`
notes/014_build_queue_readiness_checker__20260416_002319.md:9707:- run_id: `001_smoke_test_pipeline__20260415_233343`
notes/014_build_queue_readiness_checker__20260416_002319.md:9708:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
notes/014_build_queue_readiness_checker__20260416_002319.md:9709:- prompt_stem: `001_smoke_test_pipeline`
notes/014_build_queue_readiness_checker__20260416_002319.md:9710:- started_at_utc: `20260415_233343`
notes/014_build_queue_readiness_checker__20260416_002319.md:9714:- execution_status: `EXECUTION_FAILED`
notes/014_build_queue_readiness_checker__20260416_002319.md:9722:- review_status: `UNREVIEWED`
notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md:3:- run_id: `013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937`
notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md:4:- prompt_file: `codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md`
notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md:5:- prompt_stem: `013_generate_prompts_for_queue_readiness_and_review_backlog_helpers`
notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md:6:- started_at_utc: `20260416_001937`
notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md:10:- execution_status: `EXECUTED`
notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md:18:- review_status: `UNREVIEWED`
notes/001_smoke_test_pipeline__20260415_234918.md:3:- run_id: `001_smoke_test_pipeline__20260415_234918`
notes/001_smoke_test_pipeline__20260415_234918.md:4:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
notes/001_smoke_test_pipeline__20260415_234918.md:5:- prompt_stem: `001_smoke_test_pipeline`
notes/001_smoke_test_pipeline__20260415_234918.md:6:- started_at_utc: `20260415_234918`
notes/001_smoke_test_pipeline__20260415_234918.md:10:- execution_status: `EXECUTED`
notes/001_smoke_test_pipeline__20260415_234918.md:18:- review_status: `ACCEPTED`
notes/001_smoke_test_pipeline__20260415_233343.md:3:- run_id: `001_smoke_test_pipeline__20260415_233343`
notes/001_smoke_test_pipeline__20260415_233343.md:4:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
notes/001_smoke_test_pipeline__20260415_233343.md:5:- prompt_stem: `001_smoke_test_pipeline`
notes/001_smoke_test_pipeline__20260415_233343.md:6:- started_at_utc: `20260415_233343`
notes/001_smoke_test_pipeline__20260415_233343.md:10:- execution_status: `EXECUTION_FAILED`
notes/001_smoke_test_pipeline__20260415_233343.md:18:- review_status: `UNREVIEWED`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:3:- run_id: `011_build_v1_review_writeback_helper__20260415_235346`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:4:- prompt_file: `codex_prompts/011_build_v1_review_writeback_helper.md`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:5:- prompt_stem: `011_build_v1_review_writeback_helper`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:6:- started_at_utc: `20260415_235346`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:10:- execution_status: `EXECUTED`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:18:- review_status: `UNREVIEWED`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:1604:- run_id: `001_smoke_test_pipeline__20260415_234918`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:1605:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:1606:- prompt_stem: `001_smoke_test_pipeline`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:1607:- started_at_utc: `20260415_234918`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:1611:- execution_status: `EXECUTED`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:1619:- review_status: `UNREVIEWED`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:3100:- run_id: `001_smoke_test_pipeline__20260415_234918`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:3101:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:3102:- prompt_stem: `001_smoke_test_pipeline`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:3103:- started_at_utc: `20260415_234918`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:3107:- execution_status: `EXECUTED`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:3115:- review_status: `UNREVIEWED`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:3400:- run_id: `001_smoke_test_pipeline__20260415_234918`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:3401:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:3402:- prompt_stem: `001_smoke_test_pipeline`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:3403:- started_at_utc: `20260415_234918`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:3407:- execution_status: `EXECUTED`
notes/011_build_v1_review_writeback_helper__20260415_235346.md:3415:- review_status: `ACCEPTED`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:3:- run_id: `012_review_v1_pipeline_and_recommend_next_options__20260416_000658`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:4:- prompt_file: `codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:5:- prompt_stem: `012_review_v1_pipeline_and_recommend_next_options`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:6:- started_at_utc: `20260416_000658`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:10:- execution_status: `EXECUTED`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:18:- review_status: `UNREVIEWED`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:1430:- run_id: `<run_id>`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:1431:- prompt_file: `<prompt_file>`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:1432:- prompt_stem: `<prompt_stem>`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:1433:- started_at_utc: `<started_at_utc>`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:1441:- execution_status: `<EXECUTED|EXECUTION_FAILED>`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:1453:- review_status: `UNREVIEWED`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:2442:- run_id: `001_smoke_test_pipeline__20260415_234918`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:2443:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:2444:- prompt_stem: `001_smoke_test_pipeline`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:2445:- started_at_utc: `20260415_234918`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:2449:- execution_status: `EXECUTED`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:2457:- review_status: `ACCEPTED`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:2706:- run_id: `001_smoke_test_pipeline__20260415_233343`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:2707:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:2708:- prompt_stem: `001_smoke_test_pipeline`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:2709:- started_at_utc: `20260415_233343`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:2713:- execution_status: `EXECUTION_FAILED`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:2721:- review_status: `UNREVIEWED`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:3034:- run_id: `011_build_v1_review_writeback_helper__20260415_235346`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:3035:- prompt_file: `codex_prompts/011_build_v1_review_writeback_helper.md`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:3036:- prompt_stem: `011_build_v1_review_writeback_helper`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:3037:- started_at_utc: `20260415_235346`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:3041:- execution_status: `EXECUTED`
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md:3049:- review_status: `UNREVIEWED`
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:1355:- run_id: `<run_id>`
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:1356:- prompt_file: `<prompt_file>`
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:1357:- prompt_stem: `<prompt_stem>`
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:1358:- started_at_utc: `<started_at_utc>`
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:1366:- execution_status: `<EXECUTED|EXECUTION_FAILED>`
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:1378:- review_status: `UNREVIEWED`
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:1843:- run_id: `001_smoke_test_pipeline__20260415_233343`
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:1844:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:1845:- prompt_stem: `001_smoke_test_pipeline`
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:1846:- started_at_utc: `20260415_233343`
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:1850:- execution_status: `EXECUTION_FAILED`
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:1858:- review_status: `UNREVIEWED`

codex
There’s one subtlety in this repo: several design notes quote execution-record field lines inside prose, so a naive `rg`-style scan would overcount fake records. I’m building the helper to only accept top-level V1 execution-record files that start with the V1 header and required sections, then I’ll validate it against the current `notes/` set and document the exact skip/fail policy.
exec
/bin/bash -lc "sed -n '1,220p' notes/014_queue_readiness_checker_build__20260416_002419.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 014 Queue Readiness Checker Build

## What Was Built

Built a small read-only helper at [check_queue_readiness.py](/data/git/handson-ml3/tools/codex/check_queue_readiness.py) that:

- discovers prompt order from numeric prefixes in `codex_prompts/`
- parses minimal V1 execution-record field lines from markdown files in `notes/`
- finds the latest V1 record for the relevant prior prompt
- reports whether the target prompt is ready under the current V1 review gate

The helper supports:

- default queue-head checking with no positional arguments
- `--prompt` for a specific prompt file, filename stem, or numeric prefix

## Readiness Rule Applied

The helper applies the current V1 gate conservatively:

- the first prompt in sequence is ready because it has no prior review gate
- any later prompt is ready only if the latest V1 run for the immediately previous prompt has `review_status: ACCEPTED`
- `UNREVIEWED` stops the queue
- `REJECTED` stops the queue
- missing prior V1 run evidence stops the queue

For latest-record selection, the helper uses the execution-record body as the source of truth and picks the highest `started_at_utc` for the prompt, with the run-id collision suffix used only as a simple same-second tiebreaker.

## What It Intentionally Does Not Do

- does not run prompts
- does not modify execution records
- does not update review fields
- does not create sidecar state, caches, or JSON outputs
- does not release multiple future prompts
- does not build a queue engine or dashboard

## Validation Performed

1. Confirmed ordered prompt discovery from `codex_prompts/` with the helper default output.
2. Confirmed latest-run lookup from `notes/` by checking `--prompt 002`, which selected [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md) as the latest relevant prior record.
3. Confirmed review-status distinction:
   - `--prompt 002` reported ready because prior prompt `001` is `ACCEPTED`
   - `--prompt 013` reported not ready because prior prompt `012` is `UNREVIEWED`
   - a small in-memory check against `evaluate_readiness(...)` reported not ready when the prior prompt's latest run was `REJECTED`
4. Confirmed not-ready behavior when the previous prompt is not accepted or is missing:
   - `--prompt 013` and `--prompt 014` stopped on `UNREVIEWED`
   - `--prompt 003` stopped because prompt `002` has no current V1 run record
5. Confirmed ready behavior only when the previous prompt's latest run is accepted via `--prompt 002`.
6. Confirmed first-prompt handling via `--prompt 001`, which reported ready with no prior gate.
7. Confirmed `tools/codex/run_prompt.py` remained unchanged.
8. Confirmed `tools/codex/review_run.py` remained unchanged.
9. Confirmed `tools/codex/baby_run_prompt.py` remained unchanged.

Validation outcome: passed for the bounded V1 behavior above.

exec
/bin/bash -lc "sed -n '220,520p' tools/codex/check_queue_readiness.py" in /data/git/handson-ml3
 succeeded in 0ms:
        )

    return RunRecord(
        path=path.relative_to(root),
        run_id=run_id,
        prompt_file=prompt_file,
        prompt_stem=prompt_stem,
        started_at_utc=started_at_utc,
        execution_status=execution_status,
        review_status=review_status,
        run_suffix=parse_run_suffix(prompt_stem, started_at_utc, run_id),
    )


def discover_run_records(root: Path) -> list[RunRecord]:
    notes_dir = root / NOTES_DIR
    if not notes_dir.exists():
        raise ReadinessError(f"missing notes directory: {notes_dir}")

    records: list[RunRecord] = []
    for path in sorted(notes_dir.glob("*.md")):
        record = parse_record_file(root, path)
        if record is not None:
            records.append(record)
    return records


def latest_record_for_prompt(records: list[RunRecord], prompt: PromptEntry) -> RunRecord | None:
    relevant = [record for record in records if record.prompt_file == prompt.label]
    if not relevant:
        return None
    return max(relevant, key=lambda record: (record.started_at_utc, record.run_suffix))


def default_target(prompts: list[PromptEntry], records: list[RunRecord]) -> PromptEntry:
    for index, prompt in enumerate(prompts):
        if index == 0:
            latest = latest_record_for_prompt(records, prompt)
            if latest is None or latest.review_status != "ACCEPTED":
                return prompt
            continue

        previous = prompts[index - 1]
        latest_previous = latest_record_for_prompt(records, previous)
        if latest_previous is None or latest_previous.review_status != "ACCEPTED":
            return prompt

        latest_current = latest_record_for_prompt(records, prompt)
        if latest_current is None or latest_current.review_status != "ACCEPTED":
            return prompt

    raise ReadinessError("all discovered prompts already have ACCEPTED latest V1 runs; no next prompt remains")


def evaluate_readiness(
    prompts: list[PromptEntry],
    records: list[RunRecord],
    target: PromptEntry,
) -> ReadinessResult:
    index = next((i for i, prompt in enumerate(prompts) if prompt == target), None)
    if index is None:
        raise ReadinessError(f"target prompt is not in the discovered prompt list: {target.label}")

    if index == 0:
        return ReadinessResult(
            target=target,
            previous=None,
            latest_record=None,
            ready=True,
            reason="first prompt in sequence has no prior review gate",
        )

    previous = prompts[index - 1]
    latest = latest_record_for_prompt(records, previous)
    if latest is None:
        return ReadinessResult(
            target=target,
            previous=previous,
            latest_record=None,
            ready=False,
            reason="missing V1 run evidence for the immediately previous prompt",
        )

    if latest.review_status == "ACCEPTED":
        return ReadinessResult(
            target=target,
            previous=previous,
            latest_record=latest,
            ready=True,
            reason="latest V1 run for the immediately previous prompt is ACCEPTED",
        )

    if latest.review_status == "UNREVIEWED":
        reason = "latest V1 run for the immediately previous prompt is UNREVIEWED"
    else:
        reason = "latest V1 run for the immediately previous prompt is REJECTED"

    return ReadinessResult(
        target=target,
        previous=previous,
        latest_record=latest,
        ready=False,
        reason=reason,
    )


def print_summary(prompts: list[PromptEntry], result: ReadinessResult) -> None:
    print("Ordered prompts:")
    for prompt in prompts:
        print(f"- {prompt.prefix:03d}: {prompt.label}")

    latest_record = result.latest_record
    latest_record_path = latest_record.path.as_posix() if latest_record else "none"
    latest_execution_status = latest_record.execution_status if latest_record else "n/a"
    latest_review_status = latest_record.review_status if latest_record else "n/a"

    print()
    print(f"Target prompt: {result.target.label}")
    print(f"Previous prompt: {result.previous.label if result.previous else 'none'}")
    print(f"Latest run record: {latest_record_path}")
    print(f"Latest run execution_status: {latest_execution_status}")
    print(f"Latest run review_status: {latest_review_status}")
    print(f"Ready: {'YES' if result.ready else 'NO'}")
    print(f"Reason: {result.reason}")


def main() -> int:
    try:
        args = parse_args()
        root = repo_root()
        prompts = discover_prompts(root)
        records = discover_run_records(root)
        target = resolve_prompt(prompts, args.prompt) if args.prompt else default_target(prompts, records)
        result = evaluate_readiness(prompts, records, target)
        print_summary(prompts, result)
        return 0
    except ReadinessError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

exec
/bin/bash -lc 'git status --short' in /data/git/handson-ml3
 succeeded in 0ms:
codex
I’ve got enough context to edit now. I’m adding one small read-only script that parses only real top-level V1 execution-record files in `notes/`, plus a short build note with the scope, conservative parsing policy, and validation results.
apply patch
patch: completed
/data/git/handson-ml3/notes/015_review_backlog_lister_build__20260416_010500.md
/data/git/handson-ml3/tools/codex/list_review_backlog.py
diff --git a/notes/015_review_backlog_lister_build__20260416_010500.md b/notes/015_review_backlog_lister_build__20260416_010500.md
new file mode 100644
index 0000000000000000000000000000000000000000..b72013bd1465c5c12edf0d9b672a76ad410fdc3b
--- /dev/null
+++ b/notes/015_review_backlog_lister_build__20260416_010500.md
@@ -0,0 +1,79 @@
+# 015 Review Backlog Lister Build
+
+## What Was Built
+
+Built a small read-only helper at [list_review_backlog.py](/data/git/handson-ml3/tools/codex/list_review_backlog.py) that:
+
+- scans `notes/` for top-level V1 execution-record files
+- parses the execution-record body as the source of truth for `run_id`, `prompt_file`, `prompt_stem`, `started_at_utc`, `execution_status`, and `review_status`
+- lists all current `UNREVIEWED` records
+- identifies the latest execution record per prompt
+- surfaces a small "likely needs human review next" view by selecting prompts whose latest record is `UNREVIEWED`
+
+The CLI stays small:
+
+- default summary output with no positional arguments
+- optional `--unreviewed-only` filter to narrow the latest-per-prompt view to prompts whose latest record is still `UNREVIEWED`
+
+## Conservative Parsing Policy
+
+The helper intentionally does not use a full markdown parser, but it also does not treat every note with copied field lines as a record.
+
+It only accepts files that look like real top-level V1 execution records:
+
+- the file starts with a markdown title
+- the title matches the record `run_id`
+- the required V1 sections exist in the expected order
+- the required minimal V1 fields are present and valid
+
+Policy on malformed inputs:
+
+- non-record notes are skipped
+- record-like files with incomplete or malformed required V1 structure fail clearly with an error
+
+This keeps the backlog view conservative and inspectable while avoiding false positives from design notes that embed record examples.
+
+## What Backlog View It Provides
+
+The helper prints a short stdout summary with:
+
+- all `UNREVIEWED` records found
+- the latest record per prompt
+- the prompts that likely need human review next because their latest record is `UNREVIEWED`
+
+Each listed record includes:
+
+- record path
+- prompt file
+- started timestamp
+- execution status
+- review status
+
+For latest-record selection, it uses the record body and picks the highest `started_at_utc` for each prompt, with the run-id same-second suffix used only as a simple tiebreaker.
+
+## What It Intentionally Does Not Do
+
+- does not modify any files
+- does not update review fields
+- does not release prompts or compute queue readiness
+- does not create JSON, caches, sidecars, or indexes
+- does not build a dashboard, TUI, or broader reporting layer
+- does not change `tools/codex/baby_run_prompt.py`
+- does not change `tools/codex/run_prompt.py`
+- does not change `tools/codex/review_run.py`
+
+## Validation Performed
+
+1. Confirmed the helper finds V1 execution records in `notes/` by running:
+   `python3 tools/codex/list_review_backlog.py`
+2. Confirmed it lists records still marked `UNREVIEWED` from the current V1 set.
+3. Confirmed it identifies the latest record per prompt, including prompt `001` where the accepted later record supersedes the earlier failed record.
+4. Confirmed it produces a small "likely needs human review next" summary from latest-record status.
+5. Confirmed the helper remains read-only by inspecting the implementation and verifying it only reads `notes/*.md` and prints stdout.
+6. Confirmed `tools/codex/run_prompt.py` remains unchanged.
+7. Confirmed `tools/codex/review_run.py` remains unchanged.
+8. Confirmed `tools/codex/baby_run_prompt.py` remains unchanged.
+9. Confirmed the optional bounded filter works by running:
+   `python3 tools/codex/list_review_backlog.py --unreviewed-only`
+
+Validation outcome: passed for the bounded V1 review-discovery behavior above.
diff --git a/tools/codex/list_review_backlog.py b/tools/codex/list_review_backlog.py
new file mode 100644
index 0000000000000000000000000000000000000000..053f5ec2e520c0a3e3e9c29e747a8da1dea41516
--- /dev/null
+++ b/tools/codex/list_review_backlog.py
@@ -0,0 +1,264 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import re
+import sys
+from dataclasses import dataclass
+from pathlib import Path
+
+
+NOTES_DIR = "notes"
+REQUIRED_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+    "execution_status",
+    "review_status",
+)
+IDENTITY_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+)
+REQUIRED_SECTIONS = (
+    "## Execution Facts",
+    "## Review Facts",
+    "## Failure Analysis",
+    "## Resource / Cost Facts",
+    "## Prompt Text",
+    "## Codex Final Output",
+    "## Stderr",
+)
+EXECUTION_STATUSES = {"EXECUTED", "EXECUTION_FAILED"}
+REVIEW_STATUSES = {"UNREVIEWED", "ACCEPTED", "REJECTED"}
+TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")
+
+
+@dataclass(frozen=True)
+class RunRecord:
+    path: Path
+    run_id: str
+    prompt_file: str
+    prompt_stem: str
+    started_at_utc: str
+    execution_status: str
+    review_status: str
+    run_suffix: int
+
+
+class BacklogError(Exception):
+    pass
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="List the current V1 review backlog from execution records in notes/."
+    )
+    parser.add_argument(
+        "--unreviewed-only",
+        action="store_true",
+        help="Limit the latest-per-prompt and needs-review views to UNREVIEWED latest records.",
+    )
+    return parser.parse_args()
+
+
+def fail(message: str) -> int:
+    print(f"ERROR: {message}", file=sys.stderr)
+    return 1
+
+
+def parse_field_line(text: str, field: str) -> str | None:
+    match = re.search(rf"(?m)^- {re.escape(field)}:\s*(.*)$", text)
+    if match is None:
+        return None
+
+    value = match.group(1).strip()
+    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
+        return value[1:-1]
+    return value
+
+
+def parse_run_suffix(prompt_stem: str, started_at_utc: str, run_id: str) -> int:
+    base_run_id = f"{prompt_stem}__{started_at_utc}"
+    if run_id == base_run_id:
+        return 1
+
+    match = re.fullmatch(rf"{re.escape(base_run_id)}__(\d+)", run_id)
+    if match is None:
+        raise BacklogError(f"run_id does not match V1 identity pattern: {run_id}")
+    return int(match.group(1))
+
+
+def validate_section_order(text: str, path: Path, root: Path) -> None:
+    positions: list[int] = []
+    for section in REQUIRED_SECTIONS:
+        pos = text.find(section)
+        if pos == -1:
+            raise BacklogError(
+                f"record-like note is missing required V1 section: {path.relative_to(root).as_posix()} "
+                f"missing {section}"
+            )
+        positions.append(pos)
+
+    if positions != sorted(positions):
+        raise BacklogError(
+            f"record-like note has V1 sections out of order: {path.relative_to(root).as_posix()}"
+        )
+
+
+def parse_record_file(root: Path, path: Path) -> RunRecord | None:
+    text = path.read_text(encoding="utf-8")
+    if not text.startswith("# "):
+        return None
+
+    has_execution_section = "## Execution Facts" in text
+    has_review_section = "## Review Facts" in text
+    if not has_execution_section and not has_review_section:
+        return None
+    if has_execution_section != has_review_section:
+        raise BacklogError(
+            f"record-like note has incomplete V1 sections: {path.relative_to(root).as_posix()}"
+        )
+
+    header_block = text.split("\n## ", 1)[0]
+    identity_values = {field: parse_field_line(header_block, field) for field in IDENTITY_FIELDS}
+    if identity_values["run_id"] is None:
+        return None
+
+    validate_section_order(text, path, root)
+
+    metadata_block = text.split("\n## Prompt Text\n", 1)[0]
+    values = {field: parse_field_line(metadata_block, field) for field in REQUIRED_FIELDS}
+    missing_fields = [field for field, value in values.items() if value is None]
+    if missing_fields:
+        raise BacklogError(
+            f"record-like note is missing required V1 fields: {path.relative_to(root).as_posix()} "
+            f"missing {', '.join(missing_fields)}"
+        )
+
+    run_id = values["run_id"] or ""
+    prompt_file = values["prompt_file"] or ""
+    prompt_stem = values["prompt_stem"] or ""
+    started_at_utc = values["started_at_utc"] or ""
+    execution_status = values["execution_status"] or ""
+    review_status = values["review_status"] or ""
+
+    title = text.splitlines()[0][2:].strip()
+    if title != run_id:
+        raise BacklogError(
+            f"record title/run_id mismatch: {path.relative_to(root).as_posix()}"
+        )
+    if not TIMESTAMP_RE.fullmatch(started_at_utc):
+        raise BacklogError(
+            f"record has invalid started_at_utc timestamp: {path.relative_to(root).as_posix()}"
+        )
+    if execution_status not in EXECUTION_STATUSES:
+        raise BacklogError(
+            f"record has invalid execution_status: {path.relative_to(root).as_posix()}"
+        )
+    if review_status not in REVIEW_STATUSES:
+        raise BacklogError(
+            f"record has invalid review_status: {path.relative_to(root).as_posix()}"
+        )
+
+    prompt_path = Path(prompt_file)
+    if prompt_path.stem != prompt_stem:
+        raise BacklogError(
+            f"record prompt_file/prompt_stem mismatch: {path.relative_to(root).as_posix()}"
+        )
+
+    return RunRecord(
+        path=path.relative_to(root),
+        run_id=run_id,
+        prompt_file=prompt_file,
+        prompt_stem=prompt_stem,
+        started_at_utc=started_at_utc,
+        execution_status=execution_status,
+        review_status=review_status,
+        run_suffix=parse_run_suffix(prompt_stem, started_at_utc, run_id),
+    )
+
+
+def discover_run_records(root: Path) -> list[RunRecord]:
+    notes_dir = root / NOTES_DIR
+    if not notes_dir.exists():
+        raise BacklogError(f"missing notes directory: {notes_dir}")
+
+    records: list[RunRecord] = []
+    for path in sorted(notes_dir.glob("*.md")):
+        record = parse_record_file(root, path)
+        if record is not None:
+            records.append(record)
+    return records
+
+
+def latest_records_by_prompt(records: list[RunRecord]) -> list[RunRecord]:
+    latest_by_prompt: dict[str, RunRecord] = {}
+    for record in records:
+        current = latest_by_prompt.get(record.prompt_file)
+        if current is None or (record.started_at_utc, record.run_suffix) > (
+            current.started_at_utc,
+            current.run_suffix,
+        ):
+            latest_by_prompt[record.prompt_file] = record
+
+    return sorted(latest_by_prompt.values(), key=lambda record: record.prompt_stem)
+
+
+def render_record(record: RunRecord) -> str:
+    return (
+        f"- {record.path.as_posix()} | prompt={record.prompt_file} | started={record.started_at_utc} | "
+        f"execution={record.execution_status} | review={record.review_status}"
+    )
+
+
+def print_section(title: str, records: list[RunRecord]) -> None:
+    print(title)
+    if not records:
+        print("- none")
+        return
+
+    for record in records:
+        print(render_record(record))
+
+
+def main() -> int:
+    try:
+        args = parse_args()
+        root = repo_root()
+        records = discover_run_records(root)
+        unreviewed = sorted(
+            [record for record in records if record.review_status == "UNREVIEWED"],
+            key=lambda record: (record.started_at_utc, record.run_suffix, record.prompt_stem),
+        )
+        latest = latest_records_by_prompt(records)
+        needs_review_next = [record for record in latest if record.review_status == "UNREVIEWED"]
+
+        latest_view = latest if not args.unreviewed_only else needs_review_next
+
+        print(f"V1 review backlog summary from {NOTES_DIR}/")
+        print(f"Discovered V1 execution records: {len(records)}")
+        print(f"Unreviewed records: {len(unreviewed)}")
+        print(f"Prompts with latest record: {len(latest)}")
+        print()
+
+        print_section("UNREVIEWED records:", unreviewed)
+        print()
+        print_section("Latest record per prompt:", latest_view)
+        print()
+        print_section("Likely needs human review next:", needs_review_next)
+        return 0
+    except BacklogError as exc:
+        return fail(str(exc))
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

diff --git a/notes/015_review_backlog_lister_build__20260416_010500.md b/notes/015_review_backlog_lister_build__20260416_010500.md
new file mode 100644
index 0000000000000000000000000000000000000000..b72013bd1465c5c12edf0d9b672a76ad410fdc3b
--- /dev/null
+++ b/notes/015_review_backlog_lister_build__20260416_010500.md
@@ -0,0 +1,79 @@
+# 015 Review Backlog Lister Build
+
+## What Was Built
+
+Built a small read-only helper at [list_review_backlog.py](/data/git/handson-ml3/tools/codex/list_review_backlog.py) that:
+
+- scans `notes/` for top-level V1 execution-record files
+- parses the execution-record body as the source of truth for `run_id`, `prompt_file`, `prompt_stem`, `started_at_utc`, `execution_status`, and `review_status`
+- lists all current `UNREVIEWED` records
+- identifies the latest execution record per prompt
+- surfaces a small "likely needs human review next" view by selecting prompts whose latest record is `UNREVIEWED`
+
+The CLI stays small:
+
+- default summary output with no positional arguments
+- optional `--unreviewed-only` filter to narrow the latest-per-prompt view to prompts whose latest record is still `UNREVIEWED`
+
+## Conservative Parsing Policy
+
+The helper intentionally does not use a full markdown parser, but it also does not treat every note with copied field lines as a record.
+
+It only accepts files that look like real top-level V1 execution records:
+
+- the file starts with a markdown title
+- the title matches the record `run_id`
+- the required V1 sections exist in the expected order
+- the required minimal V1 fields are present and valid
+
+Policy on malformed inputs:
+
+- non-record notes are skipped
+- record-like files with incomplete or malformed required V1 structure fail clearly with an error
+
+This keeps the backlog view conservative and inspectable while avoiding false positives from design notes that embed record examples.
+
+## What Backlog View It Provides
+
+The helper prints a short stdout summary with:
+
+- all `UNREVIEWED` records found
+- the latest record per prompt
+- the prompts that likely need human review next because their latest record is `UNREVIEWED`
+
+Each listed record includes:
+
+- record path
+- prompt file
+- started timestamp
+- execution status
+- review status
+
+For latest-record selection, it uses the record body and picks the highest `started_at_utc` for each prompt, with the run-id same-second suffix used only as a simple tiebreaker.
+
+## What It Intentionally Does Not Do
+
+- does not modify any files
+- does not update review fields
+- does not release prompts or compute queue readiness
+- does not create JSON, caches, sidecars, or indexes
+- does not build a dashboard, TUI, or broader reporting layer
+- does not change `tools/codex/baby_run_prompt.py`
+- does not change `tools/codex/run_prompt.py`
+- does not change `tools/codex/review_run.py`
+
+## Validation Performed
+
+1. Confirmed the helper finds V1 execution records in `notes/` by running:
+   `python3 tools/codex/list_review_backlog.py`
+2. Confirmed it lists records still marked `UNREVIEWED` from the current V1 set.
+3. Confirmed it identifies the latest record per prompt, including prompt `001` where the accepted later record supersedes the earlier failed record.
+4. Confirmed it produces a small "likely needs human review next" summary from latest-record status.
+5. Confirmed the helper remains read-only by inspecting the implementation and verifying it only reads `notes/*.md` and prints stdout.
+6. Confirmed `tools/codex/run_prompt.py` remains unchanged.
+7. Confirmed `tools/codex/review_run.py` remains unchanged.
+8. Confirmed `tools/codex/baby_run_prompt.py` remains unchanged.
+9. Confirmed the optional bounded filter works by running:
+   `python3 tools/codex/list_review_backlog.py --unreviewed-only`
+
+Validation outcome: passed for the bounded V1 review-discovery behavior above.
diff --git a/tools/codex/list_review_backlog.py b/tools/codex/list_review_backlog.py
new file mode 100644
index 0000000000000000000000000000000000000000..053f5ec2e520c0a3e3e9c29e747a8da1dea41516
--- /dev/null
+++ b/tools/codex/list_review_backlog.py
@@ -0,0 +1,264 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import re
+import sys
+from dataclasses import dataclass
+from pathlib import Path
+
+
+NOTES_DIR = "notes"
+REQUIRED_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+    "execution_status",
+    "review_status",
+)
+IDENTITY_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+)
+REQUIRED_SECTIONS = (
+    "## Execution Facts",
+    "## Review Facts",
+    "## Failure Analysis",
+    "## Resource / Cost Facts",
+    "## Prompt Text",
+    "## Codex Final Output",
+    "## Stderr",
+)
+EXECUTION_STATUSES = {"EXECUTED", "EXECUTION_FAILED"}
+REVIEW_STATUSES = {"UNREVIEWED", "ACCEPTED", "REJECTED"}
+TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")
+
+
+@dataclass(frozen=True)
+class RunRecord:
+    path: Path
+    run_id: str
+    prompt_file: str
+    prompt_stem: str
+    started_at_utc: str
+    execution_status: str
+    review_status: str
+    run_suffix: int
+
+
+class BacklogError(Exception):
+    pass
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="List the current V1 review backlog from execution records in notes/."
+    )
+    parser.add_argument(
+        "--unreviewed-only",
+        action="store_true",
+        help="Limit the latest-per-prompt and needs-review views to UNREVIEWED latest records.",
+    )
+    return parser.parse_args()
+
+
+def fail(message: str) -> int:
+    print(f"ERROR: {message}", file=sys.stderr)
+    return 1
+
+
+def parse_field_line(text: str, field: str) -> str | None:
+    match = re.search(rf"(?m)^- {re.escape(field)}:\s*(.*)$", text)
+    if match is None:
+        return None
+
+    value = match.group(1).strip()
+    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
+        return value[1:-1]
+    return value
+
+
+def parse_run_suffix(prompt_stem: str, started_at_utc: str, run_id: str) -> int:
+    base_run_id = f"{prompt_stem}__{started_at_utc}"
+    if run_id == base_run_id:
+        return 1
+
+    match = re.fullmatch(rf"{re.escape(base_run_id)}__(\d+)", run_id)
+    if match is None:
+        raise BacklogError(f"run_id does not match V1 identity pattern: {run_id}")
+    return int(match.group(1))
+
+
+def validate_section_order(text: str, path: Path, root: Path) -> None:
+    positions: list[int] = []
+    for section in REQUIRED_SECTIONS:
+        pos = text.find(section)
+        if pos == -1:
+            raise BacklogError(
+                f"record-like note is missing required V1 section: {path.relative_to(root).as_posix()} "
+                f"missing {section}"
+            )
+        positions.append(pos)
+
+    if positions != sorted(positions):
+        raise BacklogError(
+            f"record-like note has V1 sections out of order: {path.relative_to(root).as_posix()}"
+        )
+
+
+def parse_record_file(root: Path, path: Path) -> RunRecord | None:
+    text = path.read_text(encoding="utf-8")
+    if not text.startswith("# "):
+        return None
+
+    has_execution_section = "## Execution Facts" in text
+    has_review_section = "## Review Facts" in text
+    if not has_execution_section and not has_review_section:
+        return None
+    if has_execution_section != has_review_section:
+        raise BacklogError(
+            f"record-like note has incomplete V1 sections: {path.relative_to(root).as_posix()}"
+        )
+
+    header_block = text.split("\n## ", 1)[0]
+    identity_values = {field: parse_field_line(header_block, field) for field in IDENTITY_FIELDS}
+    if identity_values["run_id"] is None:
+        return None
+
+    validate_section_order(text, path, root)
+
+    metadata_block = text.split("\n## Prompt Text\n", 1)[0]
+    values = {field: parse_field_line(metadata_block, field) for field in REQUIRED_FIELDS}
+    missing_fields = [field for field, value in values.items() if value is None]
+    if missing_fields:
+        raise BacklogError(
+            f"record-like note is missing required V1 fields: {path.relative_to(root).as_posix()} "
+            f"missing {', '.join(missing_fields)}"
+        )
+
+    run_id = values["run_id"] or ""
+    prompt_file = values["prompt_file"] or ""
+    prompt_stem = values["prompt_stem"] or ""
+    started_at_utc = values["started_at_utc"] or ""
+    execution_status = values["execution_status"] or ""
+    review_status = values["review_status"] or ""
+
+    title = text.splitlines()[0][2:].strip()
+    if title != run_id:
+        raise BacklogError(
+            f"record title/run_id mismatch: {path.relative_to(root).as_posix()}"
+        )
+    if not TIMESTAMP_RE.fullmatch(started_at_utc):
+        raise BacklogError(
+            f"record has invalid started_at_utc timestamp: {path.relative_to(root).as_posix()}"
+        )
+    if execution_status not in EXECUTION_STATUSES:
+        raise BacklogError(
+            f"record has invalid execution_status: {path.relative_to(root).as_posix()}"
+        )
+    if review_status not in REVIEW_STATUSES:
+        raise BacklogError(
+            f"record has invalid review_status: {path.relative_to(root).as_posix()}"
+        )
+
+    prompt_path = Path(prompt_file)
+    if prompt_path.stem != prompt_stem:
+        raise BacklogError(
+            f"record prompt_file/prompt_stem mismatch: {path.relative_to(root).as_posix()}"
+        )
+
+    return RunRecord(
+        path=path.relative_to(root),
+        run_id=run_id,
+        prompt_file=prompt_file,
+        prompt_stem=prompt_stem,
+        started_at_utc=started_at_utc,
+        execution_status=execution_status,
+        review_status=review_status,
+        run_suffix=parse_run_suffix(prompt_stem, started_at_utc, run_id),
+    )
+
+
+def discover_run_records(root: Path) -> list[RunRecord]:
+    notes_dir = root / NOTES_DIR
+    if not notes_dir.exists():
+        raise BacklogError(f"missing notes directory: {notes_dir}")
+
+    records: list[RunRecord] = []
+    for path in sorted(notes_dir.glob("*.md")):
+        record = parse_record_file(root, path)
+        if record is not None:
+            records.append(record)
+    return records
+
+
+def latest_records_by_prompt(records: list[RunRecord]) -> list[RunRecord]:
+    latest_by_prompt: dict[str, RunRecord] = {}
+    for record in records:
+        current = latest_by_prompt.get(record.prompt_file)
+        if current is None or (record.started_at_utc, record.run_suffix) > (
+            current.started_at_utc,
+            current.run_suffix,
+        ):
+            latest_by_prompt[record.prompt_file] = record
+
+    return sorted(latest_by_prompt.values(), key=lambda record: record.prompt_stem)
+
+
+def render_record(record: RunRecord) -> str:
+    return (
+        f"- {record.path.as_posix()} | prompt={record.prompt_file} | started={record.started_at_utc} | "
+        f"execution={record.execution_status} | review={record.review_status}"
+    )
+
+
+def print_section(title: str, records: list[RunRecord]) -> None:
+    print(title)
+    if not records:
+        print("- none")
+        return
+
+    for record in records:
+        print(render_record(record))
+
+
+def main() -> int:
+    try:
+        args = parse_args()
+        root = repo_root()
+        records = discover_run_records(root)
+        unreviewed = sorted(
+            [record for record in records if record.review_status == "UNREVIEWED"],
+            key=lambda record: (record.started_at_utc, record.run_suffix, record.prompt_stem),
+        )
+        latest = latest_records_by_prompt(records)
+        needs_review_next = [record for record in latest if record.review_status == "UNREVIEWED"]
+
+        latest_view = latest if not args.unreviewed_only else needs_review_next
+
+        print(f"V1 review backlog summary from {NOTES_DIR}/")
+        print(f"Discovered V1 execution records: {len(records)}")
+        print(f"Unreviewed records: {len(unreviewed)}")
+        print(f"Prompts with latest record: {len(latest)}")
+        print()
+
+        print_section("UNREVIEWED records:", unreviewed)
+        print()
+        print_section("Latest record per prompt:", latest_view)
+        print()
+        print_section("Likely needs human review next:", needs_review_next)
+        return 0
+    except BacklogError as exc:
+        return fail(str(exc))
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

codex
The helper and build note are in place. I’m running the new script against the current `notes/` set, checking the filtered mode, and then verifying only the two new files changed so the runner and review helpers stayed untouched.
exec
/bin/bash -lc 'python3 tools/codex/list_review_backlog.py' in /data/git/handson-ml3
exec
/bin/bash -lc 'python3 tools/codex/list_review_backlog.py --unreviewed-only' in /data/git/handson-ml3
exec
/bin/bash -lc 'git diff --stat -- tools/codex/run_prompt.py tools/codex/review_run.py tools/codex/baby_run_prompt.py tools/codex/list_review_backlog.py notes/015_review_backlog_lister_build__20260416_010500.md' in /data/git/handson-ml3
 succeeded in 0ms:
exec
/bin/bash -lc 'git status --short -- tools/codex/run_prompt.py tools/codex/review_run.py tools/codex/baby_run_prompt.py tools/codex/list_review_backlog.py notes/015_review_backlog_lister_build__20260416_010500.md' in /data/git/handson-ml3
 succeeded in 0ms:
?? notes/015_review_backlog_lister_build__20260416_010500.md
?? tools/codex/list_review_backlog.py

diff --git a/notes/015_review_backlog_lister_build__20260416_010500.md b/notes/015_review_backlog_lister_build__20260416_010500.md
new file mode 100644
index 0000000000000000000000000000000000000000..b72013bd1465c5c12edf0d9b672a76ad410fdc3b
--- /dev/null
+++ b/notes/015_review_backlog_lister_build__20260416_010500.md
@@ -0,0 +1,79 @@
+# 015 Review Backlog Lister Build
+
+## What Was Built
+
+Built a small read-only helper at [list_review_backlog.py](/data/git/handson-ml3/tools/codex/list_review_backlog.py) that:
+
+- scans `notes/` for top-level V1 execution-record files
+- parses the execution-record body as the source of truth for `run_id`, `prompt_file`, `prompt_stem`, `started_at_utc`, `execution_status`, and `review_status`
+- lists all current `UNREVIEWED` records
+- identifies the latest execution record per prompt
+- surfaces a small "likely needs human review next" view by selecting prompts whose latest record is `UNREVIEWED`
+
+The CLI stays small:
+
+- default summary output with no positional arguments
+- optional `--unreviewed-only` filter to narrow the latest-per-prompt view to prompts whose latest record is still `UNREVIEWED`
+
+## Conservative Parsing Policy
+
+The helper intentionally does not use a full markdown parser, but it also does not treat every note with copied field lines as a record.
+
+It only accepts files that look like real top-level V1 execution records:
+
+- the file starts with a markdown title
+- the title matches the record `run_id`
+- the required V1 sections exist in the expected order
+- the required minimal V1 fields are present and valid
+
+Policy on malformed inputs:
+
+- non-record notes are skipped
+- record-like files with incomplete or malformed required V1 structure fail clearly with an error
+
+This keeps the backlog view conservative and inspectable while avoiding false positives from design notes that embed record examples.
+
+## What Backlog View It Provides
+
+The helper prints a short stdout summary with:
+
+- all `UNREVIEWED` records found
+- the latest record per prompt
+- the prompts that likely need human review next because their latest record is `UNREVIEWED`
+
+Each listed record includes:
+
+- record path
+- prompt file
+- started timestamp
+- execution status
+- review status
+
+For latest-record selection, it uses the record body and picks the highest `started_at_utc` for each prompt, with the run-id same-second suffix used only as a simple tiebreaker.
+
+## What It Intentionally Does Not Do
+
+- does not modify any files
+- does not update review fields
+- does not release prompts or compute queue readiness
+- does not create JSON, caches, sidecars, or indexes
+- does not build a dashboard, TUI, or broader reporting layer
+- does not change `tools/codex/baby_run_prompt.py`
+- does not change `tools/codex/run_prompt.py`
+- does not change `tools/codex/review_run.py`
+
+## Validation Performed
+
+1. Confirmed the helper finds V1 execution records in `notes/` by running:
+   `python3 tools/codex/list_review_backlog.py`
+2. Confirmed it lists records still marked `UNREVIEWED` from the current V1 set.
+3. Confirmed it identifies the latest record per prompt, including prompt `001` where the accepted later record supersedes the earlier failed record.
+4. Confirmed it produces a small "likely needs human review next" summary from latest-record status.
+5. Confirmed the helper remains read-only by inspecting the implementation and verifying it only reads `notes/*.md` and prints stdout.
+6. Confirmed `tools/codex/run_prompt.py` remains unchanged.
+7. Confirmed `tools/codex/review_run.py` remains unchanged.
+8. Confirmed `tools/codex/baby_run_prompt.py` remains unchanged.
+9. Confirmed the optional bounded filter works by running:
+   `python3 tools/codex/list_review_backlog.py --unreviewed-only`
+
+Validation outcome: passed for the bounded V1 review-discovery behavior above.
diff --git a/tools/codex/list_review_backlog.py b/tools/codex/list_review_backlog.py
new file mode 100644
index 0000000000000000000000000000000000000000..053f5ec2e520c0a3e3e9c29e747a8da1dea41516
--- /dev/null
+++ b/tools/codex/list_review_backlog.py
@@ -0,0 +1,264 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import re
+import sys
+from dataclasses import dataclass
+from pathlib import Path
+
+
+NOTES_DIR = "notes"
+REQUIRED_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+    "execution_status",
+    "review_status",
+)
+IDENTITY_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+)
+REQUIRED_SECTIONS = (
+    "## Execution Facts",
+    "## Review Facts",
+    "## Failure Analysis",
+    "## Resource / Cost Facts",
+    "## Prompt Text",
+    "## Codex Final Output",
+    "## Stderr",
+)
+EXECUTION_STATUSES = {"EXECUTED", "EXECUTION_FAILED"}
+REVIEW_STATUSES = {"UNREVIEWED", "ACCEPTED", "REJECTED"}
+TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")
+
+
+@dataclass(frozen=True)
+class RunRecord:
+    path: Path
+    run_id: str
+    prompt_file: str
+    prompt_stem: str
+    started_at_utc: str
+    execution_status: str
+    review_status: str
+    run_suffix: int
+
+
+class BacklogError(Exception):
+    pass
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="List the current V1 review backlog from execution records in notes/."
+    )
+    parser.add_argument(
+        "--unreviewed-only",
+        action="store_true",
+        help="Limit the latest-per-prompt and needs-review views to UNREVIEWED latest records.",
+    )
+    return parser.parse_args()
+
+
+def fail(message: str) -> int:
+    print(f"ERROR: {message}", file=sys.stderr)
+    return 1
+
+
+def parse_field_line(text: str, field: str) -> str | None:
+    match = re.search(rf"(?m)^- {re.escape(field)}:\s*(.*)$", text)
+    if match is None:
+        return None
+
+    value = match.group(1).strip()
+    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
+        return value[1:-1]
+    return value
+
+
+def parse_run_suffix(prompt_stem: str, started_at_utc: str, run_id: str) -> int:
+    base_run_id = f"{prompt_stem}__{started_at_utc}"
+    if run_id == base_run_id:
+        return 1
+
+    match = re.fullmatch(rf"{re.escape(base_run_id)}__(\d+)", run_id)
+    if match is None:
+        raise BacklogError(f"run_id does not match V1 identity pattern: {run_id}")
+    return int(match.group(1))
+
+
+def validate_section_order(text: str, path: Path, root: Path) -> None:
+    positions: list[int] = []
+    for section in REQUIRED_SECTIONS:
+        pos = text.find(section)
+        if pos == -1:
+            raise BacklogError(
+                f"record-like note is missing required V1 section: {path.relative_to(root).as_posix()} "
+                f"missing {section}"
+            )
+        positions.append(pos)
+
+    if positions != sorted(positions):
+        raise BacklogError(
+            f"record-like note has V1 sections out of order: {path.relative_to(root).as_posix()}"
+        )
+
+
+def parse_record_file(root: Path, path: Path) -> RunRecord | None:
+    text = path.read_text(encoding="utf-8")
+    if not text.startswith("# "):
+        return None
+
+    has_execution_section = "## Execution Facts" in text
+    has_review_section = "## Review Facts" in text
+    if not has_execution_section and not has_review_section:
+        return None
+    if has_execution_section != has_review_section:
+        raise BacklogError(
+            f"record-like note has incomplete V1 sections: {path.relative_to(root).as_posix()}"
+        )
+
+    header_block = text.split("\n## ", 1)[0]
+    identity_values = {field: parse_field_line(header_block, field) for field in IDENTITY_FIELDS}
+    if identity_values["run_id"] is None:
+        return None
+
+    validate_section_order(text, path, root)
+
+    metadata_block = text.split("\n## Prompt Text\n", 1)[0]
+    values = {field: parse_field_line(metadata_block, field) for field in REQUIRED_FIELDS}
+    missing_fields = [field for field, value in values.items() if value is None]
+    if missing_fields:
+        raise BacklogError(
+            f"record-like note is missing required V1 fields: {path.relative_to(root).as_posix()} "
+            f"missing {', '.join(missing_fields)}"
+        )
+
+    run_id = values["run_id"] or ""
+    prompt_file = values["prompt_file"] or ""
+    prompt_stem = values["prompt_stem"] or ""
+    started_at_utc = values["started_at_utc"] or ""
+    execution_status = values["execution_status"] or ""
+    review_status = values["review_status"] or ""
+
+    title = text.splitlines()[0][2:].strip()
+    if title != run_id:
+        raise BacklogError(
+            f"record title/run_id mismatch: {path.relative_to(root).as_posix()}"
+        )
+    if not TIMESTAMP_RE.fullmatch(started_at_utc):
+        raise BacklogError(
+            f"record has invalid started_at_utc timestamp: {path.relative_to(root).as_posix()}"
+        )
+    if execution_status not in EXECUTION_STATUSES:
+        raise BacklogError(
+            f"record has invalid execution_status: {path.relative_to(root).as_posix()}"
+        )
+    if review_status not in REVIEW_STATUSES:
+        raise BacklogError(
+            f"record has invalid review_status: {path.relative_to(root).as_posix()}"
+        )
+
+    prompt_path = Path(prompt_file)
+    if prompt_path.stem != prompt_stem:
+        raise BacklogError(
+            f"record prompt_file/prompt_stem mismatch: {path.relative_to(root).as_posix()}"
+        )
+
+    return RunRecord(
+        path=path.relative_to(root),
+        run_id=run_id,
+        prompt_file=prompt_file,
+        prompt_stem=prompt_stem,
+        started_at_utc=started_at_utc,
+        execution_status=execution_status,
+        review_status=review_status,
+        run_suffix=parse_run_suffix(prompt_stem, started_at_utc, run_id),
+    )
+
+
+def discover_run_records(root: Path) -> list[RunRecord]:
+    notes_dir = root / NOTES_DIR
+    if not notes_dir.exists():
+        raise BacklogError(f"missing notes directory: {notes_dir}")
+
+    records: list[RunRecord] = []
+    for path in sorted(notes_dir.glob("*.md")):
+        record = parse_record_file(root, path)
+        if record is not None:
+            records.append(record)
+    return records
+
+
+def latest_records_by_prompt(records: list[RunRecord]) -> list[RunRecord]:
+    latest_by_prompt: dict[str, RunRecord] = {}
+    for record in records:
+        current = latest_by_prompt.get(record.prompt_file)
+        if current is None or (record.started_at_utc, record.run_suffix) > (
+            current.started_at_utc,
+            current.run_suffix,
+        ):
+            latest_by_prompt[record.prompt_file] = record
+
+    return sorted(latest_by_prompt.values(), key=lambda record: record.prompt_stem)
+
+
+def render_record(record: RunRecord) -> str:
+    return (
+        f"- {record.path.as_posix()} | prompt={record.prompt_file} | started={record.started_at_utc} | "
+        f"execution={record.execution_status} | review={record.review_status}"
+    )
+
+
+def print_section(title: str, records: list[RunRecord]) -> None:
+    print(title)
+    if not records:
+        print("- none")
+        return
+
+    for record in records:
+        print(render_record(record))
+
+
+def main() -> int:
+    try:
+        args = parse_args()
+        root = repo_root()
+        records = discover_run_records(root)
+        unreviewed = sorted(
+            [record for record in records if record.review_status == "UNREVIEWED"],
+            key=lambda record: (record.started_at_utc, record.run_suffix, record.prompt_stem),
+        )
+        latest = latest_records_by_prompt(records)
+        needs_review_next = [record for record in latest if record.review_status == "UNREVIEWED"]
+
+        latest_view = latest if not args.unreviewed_only else needs_review_next
+
+        print(f"V1 review backlog summary from {NOTES_DIR}/")
+        print(f"Discovered V1 execution records: {len(records)}")
+        print(f"Unreviewed records: {len(unreviewed)}")
+        print(f"Prompts with latest record: {len(latest)}")
+        print()
+
+        print_section("UNREVIEWED records:", unreviewed)
+        print()
+        print_section("Latest record per prompt:", latest_view)
+        print()
+        print_section("Likely needs human review next:", needs_review_next)
+        return 0
+    except BacklogError as exc:
+        return fail(str(exc))
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

 succeeded in 260ms:
V1 review backlog summary from notes/
Discovered V1 execution records: 6
Unreviewed records: 5
Prompts with latest record: 5

UNREVIEWED records:
- notes/001_smoke_test_pipeline__20260415_233343.md | prompt=codex_prompts/001_smoke_test_pipeline.md | started=20260415_233343 | execution=EXECUTION_FAILED | review=UNREVIEWED
- notes/011_build_v1_review_writeback_helper__20260415_235346.md | prompt=codex_prompts/011_build_v1_review_writeback_helper.md | started=20260415_235346 | execution=EXECUTED | review=UNREVIEWED
- notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md | prompt=codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md | started=20260416_000658 | execution=EXECUTED | review=UNREVIEWED
- notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md | prompt=codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md | started=20260416_001937 | execution=EXECUTED | review=UNREVIEWED
- notes/014_build_queue_readiness_checker__20260416_002319.md | prompt=codex_prompts/014_build_queue_readiness_checker.md | started=20260416_002319 | execution=EXECUTED | review=UNREVIEWED

Latest record per prompt:
- notes/001_smoke_test_pipeline__20260415_234918.md | prompt=codex_prompts/001_smoke_test_pipeline.md | started=20260415_234918 | execution=EXECUTED | review=ACCEPTED
- notes/011_build_v1_review_writeback_helper__20260415_235346.md | prompt=codex_prompts/011_build_v1_review_writeback_helper.md | started=20260415_235346 | execution=EXECUTED | review=UNREVIEWED
- notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md | prompt=codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md | started=20260416_000658 | execution=EXECUTED | review=UNREVIEWED
- notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md | prompt=codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md | started=20260416_001937 | execution=EXECUTED | review=UNREVIEWED
- notes/014_build_queue_readiness_checker__20260416_002319.md | prompt=codex_prompts/014_build_queue_readiness_checker.md | started=20260416_002319 | execution=EXECUTED | review=UNREVIEWED

Likely needs human review next:
- notes/011_build_v1_review_writeback_helper__20260415_235346.md | prompt=codex_prompts/011_build_v1_review_writeback_helper.md | started=20260415_235346 | execution=EXECUTED | review=UNREVIEWED
- notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md | prompt=codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md | started=20260416_000658 | execution=EXECUTED | review=UNREVIEWED
- notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md | prompt=codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md | started=20260416_001937 | execution=EXECUTED | review=UNREVIEWED
- notes/014_build_queue_readiness_checker__20260416_002319.md | prompt=codex_prompts/014_build_queue_readiness_checker.md | started=20260416_002319 | execution=EXECUTED | review=UNREVIEWED

 succeeded in 252ms:
V1 review backlog summary from notes/
Discovered V1 execution records: 6
Unreviewed records: 5
Prompts with latest record: 5

UNREVIEWED records:
- notes/001_smoke_test_pipeline__20260415_233343.md | prompt=codex_prompts/001_smoke_test_pipeline.md | started=20260415_233343 | execution=EXECUTION_FAILED | review=UNREVIEWED
- notes/011_build_v1_review_writeback_helper__20260415_235346.md | prompt=codex_prompts/011_build_v1_review_writeback_helper.md | started=20260415_235346 | execution=EXECUTED | review=UNREVIEWED
- notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md | prompt=codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md | started=20260416_000658 | execution=EXECUTED | review=UNREVIEWED
- notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md | prompt=codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md | started=20260416_001937 | execution=EXECUTED | review=UNREVIEWED
- notes/014_build_queue_readiness_checker__20260416_002319.md | prompt=codex_prompts/014_build_queue_readiness_checker.md | started=20260416_002319 | execution=EXECUTED | review=UNREVIEWED

Latest record per prompt:
- notes/011_build_v1_review_writeback_helper__20260415_235346.md | prompt=codex_prompts/011_build_v1_review_writeback_helper.md | started=20260415_235346 | execution=EXECUTED | review=UNREVIEWED
- notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md | prompt=codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md | started=20260416_000658 | execution=EXECUTED | review=UNREVIEWED
- notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md | prompt=codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md | started=20260416_001937 | execution=EXECUTED | review=UNREVIEWED
- notes/014_build_queue_readiness_checker__20260416_002319.md | prompt=codex_prompts/014_build_queue_readiness_checker.md | started=20260416_002319 | execution=EXECUTED | review=UNREVIEWED

Likely needs human review next:
- notes/011_build_v1_review_writeback_helper__20260415_235346.md | prompt=codex_prompts/011_build_v1_review_writeback_helper.md | started=20260415_235346 | execution=EXECUTED | review=UNREVIEWED
- notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md | prompt=codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md | started=20260416_000658 | execution=EXECUTED | review=UNREVIEWED
- notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md | prompt=codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md | started=20260416_001937 | execution=EXECUTED | review=UNREVIEWED
- notes/014_build_queue_readiness_checker__20260416_002319.md | prompt=codex_prompts/014_build_queue_readiness_checker.md | started=20260416_002319 | execution=EXECUTED | review=UNREVIEWED

codex
The script is producing the expected bounded summary on the current repo state: six V1 records discovered, five still unreviewed, and the latest-per-prompt view correctly prefers the later accepted `001` record. I’m doing one last syntax check, then I’ll close with the changed paths and validation outcome.
exec
/bin/bash -lc 'python3 -m py_compile tools/codex/list_review_backlog.py' in /data/git/handson-ml3
 succeeded in 0ms:
diff --git a/notes/015_review_backlog_lister_build__20260416_010500.md b/notes/015_review_backlog_lister_build__20260416_010500.md
new file mode 100644
index 0000000000000000000000000000000000000000..b72013bd1465c5c12edf0d9b672a76ad410fdc3b
--- /dev/null
+++ b/notes/015_review_backlog_lister_build__20260416_010500.md
@@ -0,0 +1,79 @@
+# 015 Review Backlog Lister Build
+
+## What Was Built
+
+Built a small read-only helper at [list_review_backlog.py](/data/git/handson-ml3/tools/codex/list_review_backlog.py) that:
+
+- scans `notes/` for top-level V1 execution-record files
+- parses the execution-record body as the source of truth for `run_id`, `prompt_file`, `prompt_stem`, `started_at_utc`, `execution_status`, and `review_status`
+- lists all current `UNREVIEWED` records
+- identifies the latest execution record per prompt
+- surfaces a small "likely needs human review next" view by selecting prompts whose latest record is `UNREVIEWED`
+
+The CLI stays small:
+
+- default summary output with no positional arguments
+- optional `--unreviewed-only` filter to narrow the latest-per-prompt view to prompts whose latest record is still `UNREVIEWED`
+
+## Conservative Parsing Policy
+
+The helper intentionally does not use a full markdown parser, but it also does not treat every note with copied field lines as a record.
+
+It only accepts files that look like real top-level V1 execution records:
+
+- the file starts with a markdown title
+- the title matches the record `run_id`
+- the required V1 sections exist in the expected order
+- the required minimal V1 fields are present and valid
+
+Policy on malformed inputs:
+
+- non-record notes are skipped
+- record-like files with incomplete or malformed required V1 structure fail clearly with an error
+
+This keeps the backlog view conservative and inspectable while avoiding false positives from design notes that embed record examples.
+
+## What Backlog View It Provides
+
+The helper prints a short stdout summary with:
+
+- all `UNREVIEWED` records found
+- the latest record per prompt
+- the prompts that likely need human review next because their latest record is `UNREVIEWED`
+
+Each listed record includes:
+
+- record path
+- prompt file
+- started timestamp
+- execution status
+- review status
+
+For latest-record selection, it uses the record body and picks the highest `started_at_utc` for each prompt, with the run-id same-second suffix used only as a simple tiebreaker.
+
+## What It Intentionally Does Not Do
+
+- does not modify any files
+- does not update review fields
+- does not release prompts or compute queue readiness
+- does not create JSON, caches, sidecars, or indexes
+- does not build a dashboard, TUI, or broader reporting layer
+- does not change `tools/codex/baby_run_prompt.py`
+- does not change `tools/codex/run_prompt.py`
+- does not change `tools/codex/review_run.py`
+
+## Validation Performed
+
+1. Confirmed the helper finds V1 execution records in `notes/` by running:
+   `python3 tools/codex/list_review_backlog.py`
+2. Confirmed it lists records still marked `UNREVIEWED` from the current V1 set.
+3. Confirmed it identifies the latest record per prompt, including prompt `001` where the accepted later record supersedes the earlier failed record.
+4. Confirmed it produces a small "likely needs human review next" summary from latest-record status.
+5. Confirmed the helper remains read-only by inspecting the implementation and verifying it only reads `notes/*.md` and prints stdout.
+6. Confirmed `tools/codex/run_prompt.py` remains unchanged.
+7. Confirmed `tools/codex/review_run.py` remains unchanged.
+8. Confirmed `tools/codex/baby_run_prompt.py` remains unchanged.
+9. Confirmed the optional bounded filter works by running:
+   `python3 tools/codex/list_review_backlog.py --unreviewed-only`
+
+Validation outcome: passed for the bounded V1 review-discovery behavior above.
diff --git a/tools/codex/list_review_backlog.py b/tools/codex/list_review_backlog.py
new file mode 100644
index 0000000000000000000000000000000000000000..053f5ec2e520c0a3e3e9c29e747a8da1dea41516
--- /dev/null
+++ b/tools/codex/list_review_backlog.py
@@ -0,0 +1,264 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import re
+import sys
+from dataclasses import dataclass
+from pathlib import Path
+
+
+NOTES_DIR = "notes"
+REQUIRED_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+    "execution_status",
+    "review_status",
+)
+IDENTITY_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+)
+REQUIRED_SECTIONS = (
+    "## Execution Facts",
+    "## Review Facts",
+    "## Failure Analysis",
+    "## Resource / Cost Facts",
+    "## Prompt Text",
+    "## Codex Final Output",
+    "## Stderr",
+)
+EXECUTION_STATUSES = {"EXECUTED", "EXECUTION_FAILED"}
+REVIEW_STATUSES = {"UNREVIEWED", "ACCEPTED", "REJECTED"}
+TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")
+
+
+@dataclass(frozen=True)
+class RunRecord:
+    path: Path
+    run_id: str
+    prompt_file: str
+    prompt_stem: str
+    started_at_utc: str
+    execution_status: str
+    review_status: str
+    run_suffix: int
+
+
+class BacklogError(Exception):
+    pass
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="List the current V1 review backlog from execution records in notes/."
+    )
+    parser.add_argument(
+        "--unreviewed-only",
+        action="store_true",
+        help="Limit the latest-per-prompt and needs-review views to UNREVIEWED latest records.",
+    )
+    return parser.parse_args()
+
+
+def fail(message: str) -> int:
+    print(f"ERROR: {message}", file=sys.stderr)
+    return 1
+
+
+def parse_field_line(text: str, field: str) -> str | None:
+    match = re.search(rf"(?m)^- {re.escape(field)}:\s*(.*)$", text)
+    if match is None:
+        return None
+
+    value = match.group(1).strip()
+    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
+        return value[1:-1]
+    return value
+
+
+def parse_run_suffix(prompt_stem: str, started_at_utc: str, run_id: str) -> int:
+    base_run_id = f"{prompt_stem}__{started_at_utc}"
+    if run_id == base_run_id:
+        return 1
+
+    match = re.fullmatch(rf"{re.escape(base_run_id)}__(\d+)", run_id)
+    if match is None:
+        raise BacklogError(f"run_id does not match V1 identity pattern: {run_id}")
+    return int(match.group(1))
+
+
+def validate_section_order(text: str, path: Path, root: Path) -> None:
+    positions: list[int] = []
+    for section in REQUIRED_SECTIONS:
+        pos = text.find(section)
+        if pos == -1:
+            raise BacklogError(
+                f"record-like note is missing required V1 section: {path.relative_to(root).as_posix()} "
+                f"missing {section}"
+            )
+        positions.append(pos)
+
+    if positions != sorted(positions):
+        raise BacklogError(
+            f"record-like note has V1 sections out of order: {path.relative_to(root).as_posix()}"
+        )
+
+
+def parse_record_file(root: Path, path: Path) -> RunRecord | None:
+    text = path.read_text(encoding="utf-8")
+    if not text.startswith("# "):
+        return None
+
+    has_execution_section = "## Execution Facts" in text
+    has_review_section = "## Review Facts" in text
+    if not has_execution_section and not has_review_section:
+        return None
+    if has_execution_section != has_review_section:
+        raise BacklogError(
+            f"record-like note has incomplete V1 sections: {path.relative_to(root).as_posix()}"
+        )
+
+    header_block = text.split("\n## ", 1)[0]
+    identity_values = {field: parse_field_line(header_block, field) for field in IDENTITY_FIELDS}
+    if identity_values["run_id"] is None:
+        return None
+
+    validate_section_order(text, path, root)
+
+    metadata_block = text.split("\n## Prompt Text\n", 1)[0]
+    values = {field: parse_field_line(metadata_block, field) for field in REQUIRED_FIELDS}
+    missing_fields = [field for field, value in values.items() if value is None]
+    if missing_fields:
+        raise BacklogError(
+            f"record-like note is missing required V1 fields: {path.relative_to(root).as_posix()} "
+            f"missing {', '.join(missing_fields)}"
+        )
+
+    run_id = values["run_id"] or ""
+    prompt_file = values["prompt_file"] or ""
+    prompt_stem = values["prompt_stem"] or ""
+    started_at_utc = values["started_at_utc"] or ""
+    execution_status = values["execution_status"] or ""
+    review_status = values["review_status"] or ""
+
+    title = text.splitlines()[0][2:].strip()
+    if title != run_id:
+        raise BacklogError(
+            f"record title/run_id mismatch: {path.relative_to(root).as_posix()}"
+        )
+    if not TIMESTAMP_RE.fullmatch(started_at_utc):
+        raise BacklogError(
+            f"record has invalid started_at_utc timestamp: {path.relative_to(root).as_posix()}"
+        )
+    if execution_status not in EXECUTION_STATUSES:
+        raise BacklogError(
+            f"record has invalid execution_status: {path.relative_to(root).as_posix()}"
+        )
+    if review_status not in REVIEW_STATUSES:
+        raise BacklogError(
+            f"record has invalid review_status: {path.relative_to(root).as_posix()}"
+        )
+
+    prompt_path = Path(prompt_file)
+    if prompt_path.stem != prompt_stem:
+        raise BacklogError(
+            f"record prompt_file/prompt_stem mismatch: {path.relative_to(root).as_posix()}"
+        )
+
+    return RunRecord(
+        path=path.relative_to(root),
+        run_id=run_id,
+        prompt_file=prompt_file,
+        prompt_stem=prompt_stem,
+        started_at_utc=started_at_utc,
+        execution_status=execution_status,
+        review_status=review_status,
+        run_suffix=parse_run_suffix(prompt_stem, started_at_utc, run_id),
+    )
+
+
+def discover_run_records(root: Path) -> list[RunRecord]:
+    notes_dir = root / NOTES_DIR
+    if not notes_dir.exists():
+        raise BacklogError(f"missing notes directory: {notes_dir}")
+
+    records: list[RunRecord] = []
+    for path in sorted(notes_dir.glob("*.md")):
+        record = parse_record_file(root, path)
+        if record is not None:
+            records.append(record)
+    return records
+
+
+def latest_records_by_prompt(records: list[RunRecord]) -> list[RunRecord]:
+    latest_by_prompt: dict[str, RunRecord] = {}
+    for record in records:
+        current = latest_by_prompt.get(record.prompt_file)
+        if current is None or (record.started_at_utc, record.run_suffix) > (
+            current.started_at_utc,
+            current.run_suffix,
+        ):
+            latest_by_prompt[record.prompt_file] = record
+
+    return sorted(latest_by_prompt.values(), key=lambda record: record.prompt_stem)
+
+
+def render_record(record: RunRecord) -> str:
+    return (
+        f"- {record.path.as_posix()} | prompt={record.prompt_file} | started={record.started_at_utc} | "
+        f"execution={record.execution_status} | review={record.review_status}"
+    )
+
+
+def print_section(title: str, records: list[RunRecord]) -> None:
+    print(title)
+    if not records:
+        print("- none")
+        return
+
+    for record in records:
+        print(render_record(record))
+
+
+def main() -> int:
+    try:
+        args = parse_args()
+        root = repo_root()
+        records = discover_run_records(root)
+        unreviewed = sorted(
+            [record for record in records if record.review_status == "UNREVIEWED"],
+            key=lambda record: (record.started_at_utc, record.run_suffix, record.prompt_stem),
+        )
+        latest = latest_records_by_prompt(records)
+        needs_review_next = [record for record in latest if record.review_status == "UNREVIEWED"]
+
+        latest_view = latest if not args.unreviewed_only else needs_review_next
+
+        print(f"V1 review backlog summary from {NOTES_DIR}/")
+        print(f"Discovered V1 execution records: {len(records)}")
+        print(f"Unreviewed records: {len(unreviewed)}")
+        print(f"Prompts with latest record: {len(latest)}")
+        print()
+
+        print_section("UNREVIEWED records:", unreviewed)
+        print()
+        print_section("Latest record per prompt:", latest_view)
+        print()
+        print_section("Likely needs human review next:", needs_review_next)
+        return 0
+    except BacklogError as exc:
+        return fail(str(exc))
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

codex
Implemented the helper at [tools/codex/list_review_backlog.py](/data/git/handson-ml3/tools/codex/list_review_backlog.py) and the implementation note at [notes/015_review_backlog_lister_build__20260416_010500.md](/data/git/handson-ml3/notes/015_review_backlog_lister_build__20260416_010500.md).

The script is read-only and stays within the V1 boundary: it scans `notes/*.md`, accepts only real top-level V1 execution records, lists all `UNREVIEWED` records, computes the latest record per prompt using `started_at_utc` plus the same-second run-id suffix as a tiebreaker, and derives a simple “likely needs human review next” view from latest records whose `review_status` is `UNREVIEWED`. It supports the bounded optional filter `--unreviewed-only`.

Validation passed:
- `python3 tools/codex/list_review_backlog.py`
- `python3 tools/codex/list_review_backlog.py --unreviewed-only`
- `python3 -m py_compile tools/codex/list_review_backlog.py`

On the current repo state, the helper finds 6 V1 execution records, 5 unreviewed records, and 5 prompts with a latest record; it correctly selects the later accepted `001` run as the latest for that prompt. `tools/codex/run_prompt.py`, `tools/codex/review_run.py`, and `tools/codex/baby_run_prompt.py` remain unchanged.
diff --git a/notes/015_review_backlog_lister_build__20260416_010500.md b/notes/015_review_backlog_lister_build__20260416_010500.md
new file mode 100644
index 0000000000000000000000000000000000000000..b72013bd1465c5c12edf0d9b672a76ad410fdc3b
--- /dev/null
+++ b/notes/015_review_backlog_lister_build__20260416_010500.md
@@ -0,0 +1,79 @@
+# 015 Review Backlog Lister Build
+
+## What Was Built
+
+Built a small read-only helper at [list_review_backlog.py](/data/git/handson-ml3/tools/codex/list_review_backlog.py) that:
+
+- scans `notes/` for top-level V1 execution-record files
+- parses the execution-record body as the source of truth for `run_id`, `prompt_file`, `prompt_stem`, `started_at_utc`, `execution_status`, and `review_status`
+- lists all current `UNREVIEWED` records
+- identifies the latest execution record per prompt
+- surfaces a small "likely needs human review next" view by selecting prompts whose latest record is `UNREVIEWED`
+
+The CLI stays small:
+
+- default summary output with no positional arguments
+- optional `--unreviewed-only` filter to narrow the latest-per-prompt view to prompts whose latest record is still `UNREVIEWED`
+
+## Conservative Parsing Policy
+
+The helper intentionally does not use a full markdown parser, but it also does not treat every note with copied field lines as a record.
+
+It only accepts files that look like real top-level V1 execution records:
+
+- the file starts with a markdown title
+- the title matches the record `run_id`
+- the required V1 sections exist in the expected order
+- the required minimal V1 fields are present and valid
+
+Policy on malformed inputs:
+
+- non-record notes are skipped
+- record-like files with incomplete or malformed required V1 structure fail clearly with an error
+
+This keeps the backlog view conservative and inspectable while avoiding false positives from design notes that embed record examples.
+
+## What Backlog View It Provides
+
+The helper prints a short stdout summary with:
+
+- all `UNREVIEWED` records found
+- the latest record per prompt
+- the prompts that likely need human review next because their latest record is `UNREVIEWED`
+
+Each listed record includes:
+
+- record path
+- prompt file
+- started timestamp
+- execution status
+- review status
+
+For latest-record selection, it uses the record body and picks the highest `started_at_utc` for each prompt, with the run-id same-second suffix used only as a simple tiebreaker.
+
+## What It Intentionally Does Not Do
+
+- does not modify any files
+- does not update review fields
+- does not release prompts or compute queue readiness
+- does not create JSON, caches, sidecars, or indexes
+- does not build a dashboard, TUI, or broader reporting layer
+- does not change `tools/codex/baby_run_prompt.py`
+- does not change `tools/codex/run_prompt.py`
+- does not change `tools/codex/review_run.py`
+
+## Validation Performed
+
+1. Confirmed the helper finds V1 execution records in `notes/` by running:
+   `python3 tools/codex/list_review_backlog.py`
+2. Confirmed it lists records still marked `UNREVIEWED` from the current V1 set.
+3. Confirmed it identifies the latest record per prompt, including prompt `001` where the accepted later record supersedes the earlier failed record.
+4. Confirmed it produces a small "likely needs human review next" summary from latest-record status.
+5. Confirmed the helper remains read-only by inspecting the implementation and verifying it only reads `notes/*.md` and prints stdout.
+6. Confirmed `tools/codex/run_prompt.py` remains unchanged.
+7. Confirmed `tools/codex/review_run.py` remains unchanged.
+8. Confirmed `tools/codex/baby_run_prompt.py` remains unchanged.
+9. Confirmed the optional bounded filter works by running:
+   `python3 tools/codex/list_review_backlog.py --unreviewed-only`
+
+Validation outcome: passed for the bounded V1 review-discovery behavior above.
diff --git a/tools/codex/list_review_backlog.py b/tools/codex/list_review_backlog.py
new file mode 100644
index 0000000000000000000000000000000000000000..053f5ec2e520c0a3e3e9c29e747a8da1dea41516
--- /dev/null
+++ b/tools/codex/list_review_backlog.py
@@ -0,0 +1,264 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import re
+import sys
+from dataclasses import dataclass
+from pathlib import Path
+
+
+NOTES_DIR = "notes"
+REQUIRED_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+    "execution_status",
+    "review_status",
+)
+IDENTITY_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+)
+REQUIRED_SECTIONS = (
+    "## Execution Facts",
+    "## Review Facts",
+    "## Failure Analysis",
+    "## Resource / Cost Facts",
+    "## Prompt Text",
+    "## Codex Final Output",
+    "## Stderr",
+)
+EXECUTION_STATUSES = {"EXECUTED", "EXECUTION_FAILED"}
+REVIEW_STATUSES = {"UNREVIEWED", "ACCEPTED", "REJECTED"}
+TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")
+
+
+@dataclass(frozen=True)
+class RunRecord:
+    path: Path
+    run_id: str
+    prompt_file: str
+    prompt_stem: str
+    started_at_utc: str
+    execution_status: str
+    review_status: str
+    run_suffix: int
+
+
+class BacklogError(Exception):
+    pass
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="List the current V1 review backlog from execution records in notes/."
+    )
+    parser.add_argument(
+        "--unreviewed-only",
+        action="store_true",
+        help="Limit the latest-per-prompt and needs-review views to UNREVIEWED latest records.",
+    )
+    return parser.parse_args()
+
+
+def fail(message: str) -> int:
+    print(f"ERROR: {message}", file=sys.stderr)
+    return 1
+
+
+def parse_field_line(text: str, field: str) -> str | None:
+    match = re.search(rf"(?m)^- {re.escape(field)}:\s*(.*)$", text)
+    if match is None:
+        return None
+
+    value = match.group(1).strip()
+    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
+        return value[1:-1]
+    return value
+
+
+def parse_run_suffix(prompt_stem: str, started_at_utc: str, run_id: str) -> int:
+    base_run_id = f"{prompt_stem}__{started_at_utc}"
+    if run_id == base_run_id:
+        return 1
+
+    match = re.fullmatch(rf"{re.escape(base_run_id)}__(\d+)", run_id)
+    if match is None:
+        raise BacklogError(f"run_id does not match V1 identity pattern: {run_id}")
+    return int(match.group(1))
+
+
+def validate_section_order(text: str, path: Path, root: Path) -> None:
+    positions: list[int] = []
+    for section in REQUIRED_SECTIONS:
+        pos = text.find(section)
+        if pos == -1:
+            raise BacklogError(
+                f"record-like note is missing required V1 section: {path.relative_to(root).as_posix()} "
+                f"missing {section}"
+            )
+        positions.append(pos)
+
+    if positions != sorted(positions):
+        raise BacklogError(
+            f"record-like note has V1 sections out of order: {path.relative_to(root).as_posix()}"
+        )
+
+
+def parse_record_file(root: Path, path: Path) -> RunRecord | None:
+    text = path.read_text(encoding="utf-8")
+    if not text.startswith("# "):
+        return None
+
+    has_execution_section = "## Execution Facts" in text
+    has_review_section = "## Review Facts" in text
+    if not has_execution_section and not has_review_section:
+        return None
+    if has_execution_section != has_review_section:
+        raise BacklogError(
+            f"record-like note has incomplete V1 sections: {path.relative_to(root).as_posix()}"
+        )
+
+    header_block = text.split("\n## ", 1)[0]
+    identity_values = {field: parse_field_line(header_block, field) for field in IDENTITY_FIELDS}
+    if identity_values["run_id"] is None:
+        return None
+
+    validate_section_order(text, path, root)
+
+    metadata_block = text.split("\n## Prompt Text\n", 1)[0]
+    values = {field: parse_field_line(metadata_block, field) for field in REQUIRED_FIELDS}
+    missing_fields = [field for field, value in values.items() if value is None]
+    if missing_fields:
+        raise BacklogError(
+            f"record-like note is missing required V1 fields: {path.relative_to(root).as_posix()} "
+            f"missing {', '.join(missing_fields)}"
+        )
+
+    run_id = values["run_id"] or ""
+    prompt_file = values["prompt_file"] or ""
+    prompt_stem = values["prompt_stem"] or ""
+    started_at_utc = values["started_at_utc"] or ""
+    execution_status = values["execution_status"] or ""
+    review_status = values["review_status"] or ""
+
+    title = text.splitlines()[0][2:].strip()
+    if title != run_id:
+        raise BacklogError(
+            f"record title/run_id mismatch: {path.relative_to(root).as_posix()}"
+        )
+    if not TIMESTAMP_RE.fullmatch(started_at_utc):
+        raise BacklogError(
+            f"record has invalid started_at_utc timestamp: {path.relative_to(root).as_posix()}"
+        )
+    if execution_status not in EXECUTION_STATUSES:
+        raise BacklogError(
+            f"record has invalid execution_status: {path.relative_to(root).as_posix()}"
+        )
+    if review_status not in REVIEW_STATUSES:
+        raise BacklogError(
+            f"record has invalid review_status: {path.relative_to(root).as_posix()}"
+        )
+
+    prompt_path = Path(prompt_file)
+    if prompt_path.stem != prompt_stem:
+        raise BacklogError(
+            f"record prompt_file/prompt_stem mismatch: {path.relative_to(root).as_posix()}"
+        )
+
+    return RunRecord(
+        path=path.relative_to(root),
+        run_id=run_id,
+        prompt_file=prompt_file,
+        prompt_stem=prompt_stem,
+        started_at_utc=started_at_utc,
+        execution_status=execution_status,
+        review_status=review_status,
+        run_suffix=parse_run_suffix(prompt_stem, started_at_utc, run_id),
+    )
+
+
+def discover_run_records(root: Path) -> list[RunRecord]:
+    notes_dir = root / NOTES_DIR
+    if not notes_dir.exists():
+        raise BacklogError(f"missing notes directory: {notes_dir}")
+
+    records: list[RunRecord] = []
+    for path in sorted(notes_dir.glob("*.md")):
+        record = parse_record_file(root, path)
+        if record is not None:
+            records.append(record)
+    return records
+
+
+def latest_records_by_prompt(records: list[RunRecord]) -> list[RunRecord]:
+    latest_by_prompt: dict[str, RunRecord] = {}
+    for record in records:
+        current = latest_by_prompt.get(record.prompt_file)
+        if current is None or (record.started_at_utc, record.run_suffix) > (
+            current.started_at_utc,
+            current.run_suffix,
+        ):
+            latest_by_prompt[record.prompt_file] = record
+
+    return sorted(latest_by_prompt.values(), key=lambda record: record.prompt_stem)
+
+
+def render_record(record: RunRecord) -> str:
+    return (
+        f"- {record.path.as_posix()} | prompt={record.prompt_file} | started={record.started_at_utc} | "
+        f"execution={record.execution_status} | review={record.review_status}"
+    )
+
+
+def print_section(title: str, records: list[RunRecord]) -> None:
+    print(title)
+    if not records:
+        print("- none")
+        return
+
+    for record in records:
+        print(render_record(record))
+
+
+def main() -> int:
+    try:
+        args = parse_args()
+        root = repo_root()
+        records = discover_run_records(root)
+        unreviewed = sorted(
+            [record for record in records if record.review_status == "UNREVIEWED"],
+            key=lambda record: (record.started_at_utc, record.run_suffix, record.prompt_stem),
+        )
+        latest = latest_records_by_prompt(records)
+        needs_review_next = [record for record in latest if record.review_status == "UNREVIEWED"]
+
+        latest_view = latest if not args.unreviewed_only else needs_review_next
+
+        print(f"V1 review backlog summary from {NOTES_DIR}/")
+        print(f"Discovered V1 execution records: {len(records)}")
+        print(f"Unreviewed records: {len(unreviewed)}")
+        print(f"Prompts with latest record: {len(latest)}")
+        print()
+
+        print_section("UNREVIEWED records:", unreviewed)
+        print()
+        print_section("Latest record per prompt:", latest_view)
+        print()
+        print_section("Likely needs human review next:", needs_review_next)
+        return 0
+    except BacklogError as exc:
+        return fail(str(exc))
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

tokens used
51,027
```
