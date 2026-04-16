# 011_build_v1_review_writeback_helper__20260415_235346

- run_id: `011_build_v1_review_writeback_helper__20260415_235346`
- prompt_file: `codex_prompts/011_build_v1_review_writeback_helper.md`
- prompt_stem: `011_build_v1_review_writeback_helper`
- started_at_utc: `20260415_235346`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260415_235541`
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

- elapsed_seconds: `115.224`
- final_output_char_count: `1037`
- stderr_char_count: `148536`

## Prompt Text

```md
# Task: Build a small V1 review write-back helper for execution records

You are working in this repository.

Your task is to implement a small helper that updates an existing V1 execution record in `notes/` with manual review information.

## Important framing

This is a small helper task.

Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`

Those runners should remain unchanged in this pass.

You are building a companion tool beside them, not refactoring them.

## Files to inspect

Read these exact files before making changes:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/run_prompt.py`
- `notes/001_smoke_test_pipeline__20260415_234918.md`

You may also read for context:

- `tools/codex/baby_run_prompt.py`
- `notes/009_run_prompt_candidate_build__20260415_233407.md`
- `notes/010_run_prompt_candidate_review__20260415_234559.md`
- `notes/010_next_step_recommendation__20260415_234559.md`

## Goal

Create a small helper at:

`tools/codex/review_run.py`

This helper should let a human reviewer update an existing V1 execution record with review information while preserving the existing record structure.

The helper should support the minimum V1 review write-back flow, not a broader workflow engine.

## What the helper must do

The helper should:

1. accept a path to an existing execution-record markdown file in `notes/`
2. verify that the file looks like a V1 execution record
3. allow the reviewer to set:
   - `review_status`
   - `review_summary`
4. optionally allow the reviewer to set:
   - `reviewed_by`
   - `reviewed_at_utc`
5. when `review_status` is `REJECTED`, optionally allow the reviewer to also set:
   - `failure_type`
   - `failure_symptom`
   - `likely_cause`
   - `recommended_next_action`
6. write the updates back into the same markdown file
7. preserve all other existing content unchanged
8. print the updated file path
9. exit nonzero on invalid input or malformed record structure

## Minimum CLI behavior

Keep the CLI small and explicit.

Required arguments:

- `record` → path to the record file to update
- `--review-status` → must be one of:
  - `ACCEPTED`
  - `REJECTED`

Required:
- `--review-summary`

Optional:
- `--reviewed-by`
- `--reviewed-at-utc`

Optional rejection-only fields:
- `--failure-type`
- `--failure-symptom`
- `--likely-cause`
- `--recommended-next-action`

## Important behavior rules

### 1. Preserve V1 boundaries
Do NOT:
- release the next prompt automatically
- compute queue state
- create additional files
- create sidecars or databases
- add automation beyond updating the one record

### 2. Respect the review gate
The helper should behave consistently with `tools/codex/V1_Run_Review_Gate.md`.

This means:
- it updates review facts
- it may update failure-analysis fields
- it does not try to redesign the workflow

### 3. Preserve record structure
The helper should update the existing markdown record in place.

Do not rewrite the record into a different format.
Do not reorder sections unnecessarily.
Do not remove prompt text, output, stderr, or resource facts.

### 4. Handle reviewed timestamp sensibly
If `--reviewed-at-utc` is not supplied, the helper should fill it automatically with current UTC time.

## Implementation guidance

Keep this helper small and inspectable.

A simple strategy is acceptable:
- read the whole file as text
- replace the specific field lines in the known V1 structure
- validate only the minimum required sections/fields before writing back

Do not build a large markdown parser unless it is truly necessary.

## Required artifacts

### Artifact 1
Create:

`tools/codex/review_run.py`

### Artifact 2
Create a short implementation note at:

`notes/011_review_writeback_helper_build__TIMESTAMP.md`

This note should summarize:

- what was built
- what V1 fields it updates
- what it intentionally does not do
- what validation was performed

### Artifact 3
Validate the helper by applying it to this record:

`notes/001_smoke_test_pipeline__20260415_234918.md`

Use a review outcome of:

- `ACCEPTED`

Use a short review summary that makes sense for the smoke test.

The goal is to prove that the helper can update an existing V1 execution record in place.

## Validation requirements

After implementation, validate at least these points:

1. the helper updates `review_status`
2. the helper updates `review_summary`
3. the helper fills `reviewed_at_utc`
4. the helper preserves all other sections
5. the updated record still looks like a V1 execution record
6. `tools/codex/run_prompt.py` remains unchanged
7. `tools/codex/baby_run_prompt.py` remains unchanged

Include the validation outcome in the implementation note.

## Constraints

1. Use the exact file paths listed above.
2. Keep the helper single-file unless a tiny helper is truly necessary.
3. Do not modify the runners in this pass.
4. Do not alter the design documents in this pass.
5. Do not build broader queue or workflow logic.
6. Keep the implementation small enough to review comfortably.

## Success criteria

This task is successful if:

- `tools/codex/review_run.py` exists and works
- it can update a V1 execution record in place
- it writes back review facts consistent with the V1 review gate
- the validation record is successfully updated to `ACCEPTED`
- both runners remain unchanged
- the helper remains thin and inspectable
```

## Codex Final Output

Implemented [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py). It is a thin in-place V1 write-back helper: it validates the target record under `notes/`, checks the minimum V1 section/field shape, updates `review_status` and `review_summary`, auto-fills `reviewed_at_utc` when omitted, optionally writes `reviewed_by`, and allows the rejection-only failure fields only for `REJECTED`.

I validated it by running it against [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md), which is now updated to `ACCEPTED` with a review summary and `reviewed_at_utc: 20260415_235508`. The diff on that record is limited to the review lines, so the rest of the V1 structure stayed intact.

The implementation note is at [011_review_writeback_helper_build__20260415_235514.md](/data/git/handson-ml3/notes/011_review_writeback_helper_build__20260415_235514.md). Validation also confirmed `tools/codex/run_prompt.py` and `tools/codex/baby_run_prompt.py` remain unchanged.

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
session id: 019d9390-b2fc-7cb3-8e3f-ad827463d0e5
--------
user
# Task: Build a small V1 review write-back helper for execution records

You are working in this repository.

Your task is to implement a small helper that updates an existing V1 execution record in `notes/` with manual review information.

## Important framing

This is a small helper task.

Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`

Those runners should remain unchanged in this pass.

You are building a companion tool beside them, not refactoring them.

## Files to inspect

Read these exact files before making changes:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/run_prompt.py`
- `notes/001_smoke_test_pipeline__20260415_234918.md`

You may also read for context:

- `tools/codex/baby_run_prompt.py`
- `notes/009_run_prompt_candidate_build__20260415_233407.md`
- `notes/010_run_prompt_candidate_review__20260415_234559.md`
- `notes/010_next_step_recommendation__20260415_234559.md`

## Goal

Create a small helper at:

`tools/codex/review_run.py`

This helper should let a human reviewer update an existing V1 execution record with review information while preserving the existing record structure.

The helper should support the minimum V1 review write-back flow, not a broader workflow engine.

## What the helper must do

The helper should:

1. accept a path to an existing execution-record markdown file in `notes/`
2. verify that the file looks like a V1 execution record
3. allow the reviewer to set:
   - `review_status`
   - `review_summary`
4. optionally allow the reviewer to set:
   - `reviewed_by`
   - `reviewed_at_utc`
5. when `review_status` is `REJECTED`, optionally allow the reviewer to also set:
   - `failure_type`
   - `failure_symptom`
   - `likely_cause`
   - `recommended_next_action`
6. write the updates back into the same markdown file
7. preserve all other existing content unchanged
8. print the updated file path
9. exit nonzero on invalid input or malformed record structure

## Minimum CLI behavior

Keep the CLI small and explicit.

Required arguments:

- `record` → path to the record file to update
- `--review-status` → must be one of:
  - `ACCEPTED`
  - `REJECTED`

Required:
- `--review-summary`

Optional:
- `--reviewed-by`
- `--reviewed-at-utc`

Optional rejection-only fields:
- `--failure-type`
- `--failure-symptom`
- `--likely-cause`
- `--recommended-next-action`

## Important behavior rules

### 1. Preserve V1 boundaries
Do NOT:
- release the next prompt automatically
- compute queue state
- create additional files
- create sidecars or databases
- add automation beyond updating the one record

### 2. Respect the review gate
The helper should behave consistently with `tools/codex/V1_Run_Review_Gate.md`.

This means:
- it updates review facts
- it may update failure-analysis fields
- it does not try to redesign the workflow

### 3. Preserve record structure
The helper should update the existing markdown record in place.

Do not rewrite the record into a different format.
Do not reorder sections unnecessarily.
Do not remove prompt text, output, stderr, or resource facts.

### 4. Handle reviewed timestamp sensibly
If `--reviewed-at-utc` is not supplied, the helper should fill it automatically with current UTC time.

## Implementation guidance

Keep this helper small and inspectable.

A simple strategy is acceptable:
- read the whole file as text
- replace the specific field lines in the known V1 structure
- validate only the minimum required sections/fields before writing back

Do not build a large markdown parser unless it is truly necessary.

## Required artifacts

### Artifact 1
Create:

`tools/codex/review_run.py`

### Artifact 2
Create a short implementation note at:

`notes/011_review_writeback_helper_build__TIMESTAMP.md`

This note should summarize:

- what was built
- what V1 fields it updates
- what it intentionally does not do
- what validation was performed

### Artifact 3
Validate the helper by applying it to this record:

`notes/001_smoke_test_pipeline__20260415_234918.md`

Use a review outcome of:

- `ACCEPTED`

Use a short review summary that makes sense for the smoke test.

The goal is to prove that the helper can update an existing V1 execution record in place.

## Validation requirements

After implementation, validate at least these points:

1. the helper updates `review_status`
2. the helper updates `review_summary`
3. the helper fills `reviewed_at_utc`
4. the helper preserves all other sections
5. the updated record still looks like a V1 execution record
6. `tools/codex/run_prompt.py` remains unchanged
7. `tools/codex/baby_run_prompt.py` remains unchanged

Include the validation outcome in the implementation note.

## Constraints

1. Use the exact file paths listed above.
2. Keep the helper single-file unless a tiny helper is truly necessary.
3. Do not modify the runners in this pass.
4. Do not alter the design documents in this pass.
5. Do not build broader queue or workflow logic.
6. Keep the implementation small enough to review comfortably.

## Success criteria

This task is successful if:

- `tools/codex/review_run.py` exists and works
- it can update a V1 execution record in place
- it writes back review facts consistent with the V1 review gate
- the validation record is successfully updated to `ACCEPTED`
- both runners remain unchanged
- the helper remains thin and inspectable

codex
I’m inspecting the specified workflow docs, runner files, and sample notes first so the helper matches the existing V1 record shape and review gate without touching the runners.
  → Read required docs, runners, and sample records to determine V1 execution record structure and review field locations
  • Implement thin in-place review write-back helper at tools/codex/review_run.py
  • Validate helper against notes/001_smoke_test_pipeline__20260415_234918.md and capture results in implementation note
exec
/bin/bash -lc "sed -n '1,260p' tools/Codex_Prompt_Workflow_Architecture.md" in /data/git/handson-ml3
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

This means the review gate is not a separate platform subsystem.

It is the manual transition of a single execution record from:

* `UNREVIEWED`

to one of:

* `ACCEPTED`
* `REJECTED`

The review gate defined in `tools/codex/V1_Run_Review_Gate.md` is therefore part of the architecture boundary for V1, not an optional later add-on.

---

## What The First Script Was Doing

Below is a decomposition of the responsibilities that were bundled together.

### 1. Repo path discovery

* figure out where the repository root is
* infer where `codex_prompts/` and `notes/` live

### 2. Directory validation

* verify the expected folders exist
* fail cleanly if they do not

### 3. Prompt discovery

* scan `codex_prompts/`
* identify valid prompt files
* ignore files that do not match the naming convention

### 4. Prompt parsing and indexing

* extract numeric prefixes from filenames
* sort prompts in execution order

exec
/bin/bash -lc "sed -n '1,260p' tools/codex/V1_Execution_Record_Artifact.md" in /data/git/handson-ml3
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

## Practical Record Shape

A V1 markdown record should have these sections in this order:

1. Header / identity
2. Execution facts
3. Review facts
4. Failure analysis
5. Resource / cost facts
6. Prompt text
7. Codex final output
8. Stderr or supplemental notes

This keeps one run easy to inspect in a single file.

## Intentionally Deferred From V1

The following should wait:

- automatic acceptance or approval gating
- dependency-aware prompt release
- richer review states such as `BLOCKED` or `PARTIAL`
- structured JSON sidecars or databases
- normalized token accounting across runners
- automatic failure classification
- aggregated reports and dashboards
- full retry history synthesis
- broader refactor of the runner into multiple modules

V1 is only trying to establish a durable, reviewable single-run artifact that cleanly separates execution from acceptance.

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
/bin/bash -lc "sed -n '1,260p' tools/codex/V1_Run_Review_Gate.md" in /data/git/handson-ml3
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
/bin/bash -lc "sed -n '1,260p' notes/001_smoke_test_pipeline__20260415_234918.md" in /data/git/handson-ml3
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
- status
- whether the expected folders were found
- whether the prompt file was readable
- a short summary stating whether the pipeline appears functional

## Constraints
- Do not make unrelated repo changes
- Keep the report short and clear
- If something is missing, mark the run as FAIL and explain what is missing

exec
/bin/bash -lc "date '+%Y%m%d_%H%M%S|%Y-%m-%d %H:%M:%S %Z'" in /data/git/handson-ml3
 succeeded in 0ms:
20260415_184932|2026-04-15 18:49:32 CDT

codex
I’ve confirmed the prompt file is readable. I’m writing a single new note with the current timestamp and a `SUCCESS` marker, without touching anything else in the repo.
apply patch
patch: completed
/data/git/handson-ml3/notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md
diff --git a/notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md b/notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md
new file mode 100644
index 0000000000000000000000000000000000000000..34e09342bcd8f4bb40b487081971177d9cd6b743
--- /dev/null
+++ b/notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md
@@ -0,0 +1,9 @@
+# 001 Smoke Test Pipeline
+
+- Prompt filename: `codex_prompts/001_smoke_test_pipeline.md`
+- Timestamp: `2026-04-15 18:49:32 CDT`
+- Status: `SUCCESS`
+- Expected folders found: `codex_prompts/`, `notes/`, `tools/`
+- Prompt file readable: `yes`
+
+Pipeline appears functional: the required scaffold folders exist, the prompt file was present and readable, and this note was written to `notes/`.

diff --git a/notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md b/notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md
new file mode 100644
index 0000000000000000000000000000000000000000..34e09342bcd8f4bb40b487081971177d9cd6b743
--- /dev/null

codex
I have the V1 record shape and review-gate rules. I’m checking the sibling runner and a couple of existing review/build notes now so the helper stays aligned with current conventions and remains minimal.
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
/bin/bash -lc "sed -n '1,260p' notes/009_run_prompt_candidate_build__20260415_233407.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 009 Run Prompt Candidate Build

## Summary

Built `tools/codex/run_prompt.py` as a thin V1 runner candidate that resolves one prompt, runs `codex exec` from the repo root, captures `--output-last-message`, captures stderr and return code, writes one V1 execution-record markdown file into `notes/`, prints that path, and exits with the subprocess return code.

## Preserved From `baby_run_prompt.py`

- Kept repo-root discovery via `Path(__file__).resolve().parents[2]`.
- Kept prompt resolution behavior for absolute paths, repo-relative paths, `codex_prompts/` paths, and unique prefix matches.
- Kept direct `subprocess.run(...)` execution of `codex exec`.
- Kept `--output-last-message` capture through a temp file.
- Kept the requirement that `codex_prompts/` and `notes/` already exist.
- Kept one markdown artifact per run in `notes/`.
- Kept printing the written artifact path and returning the subprocess exit code.

## V1 Changes

- Switched artifact output from the old success/failure note format to a V1 execution record.
- Separated execution facts from review facts.
- Initialized review fields with `review_status: UNREVIEWED` and blank manual review fields.
- Added the manual failure-analysis section with blank fields.
- Added the minimum resource/cost fields:
  - `elapsed_seconds`
  - `final_output_char_count`
  - `stderr_char_count`
- Set `execution_status` strictly from subprocess exit code:
  - `0 -> EXECUTED`
  - non-zero -> `EXECUTION_FAILED`
- Recorded `prompt_file` as repo-relative when possible.
- Updated the runner identity field to `tools/codex/run_prompt.py`.

## Collision Rule

Run identity uses second-precision UTC timestamps in the base form:

- `<prompt_stem>__<started_at_utc>`

If `notes/<run_id>.md` already exists, the runner appends a numeric suffix:

- `__2`, `__3`, and so on

This keeps the V1 identity scheme small while avoiding same-second filename collisions.

## Deferred

- Review write-back
- Queue progression logic
- Dependency handling
- Retry orchestration
- Broader CLI redesign
- Multi-module refactor
- JSON sidecars, databases, or any other persistence layer

## Validation

Executed:

- `python3 tools/codex/run_prompt.py codex_prompts/001_smoke_test_pipeline.md`

Outcome:

- The runner resolved the sample prompt and wrote [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md).
- The record filename follows the V1 naming rule: `<prompt_stem>__<started_at_utc>.md`.
- The record includes run identity, execution facts, review facts, failure analysis, resource/cost facts, prompt text, Codex final output, and stderr.
- `review_status` starts as `UNREVIEWED`.
- `execution_status` is derived only from the subprocess result. In the validation run it was `EXECUTION_FAILED` because `codex exec` exited `1`.
- `tools/codex/baby_run_prompt.py` remained unchanged. `git diff -- tools/codex/baby_run_prompt.py` returned no diff.

Validation note:

- The sample run did not execute successfully inside `codex exec` because Codex session initialization failed with `Read-only file system (os error 30)`.
- That environment failure still validated the required V1 failure-path behavior: the new runner wrote the execution record, captured stderr, printed the record path, and exited with the subprocess return code.

exec
/bin/bash -lc "sed -n '1,260p' notes/010_run_prompt_candidate_review__20260415_234559.md" in /data/git/handson-ml3
 succeeded in 0ms:
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

exec
/bin/bash -lc "sed -n '1,260p' notes/010_next_step_recommendation__20260415_234559.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 010 Next Step Recommendation

Chosen next move: `environment diagnosis prompt`

Why it should happen next:
The V1 runner candidate already produces the intended execution record shape closely enough, and the validation run failed during Codex session initialization with a read-only filesystem error. That makes environment diagnosis the smallest step that can unblock meaningful validation without broadening the build.

What it should produce:
A short diagnosis artifact that identifies where session startup is trying to write, whether the failure reproduces independently of `tools/codex/run_prompt.py`, and one bounded follow-up action.

What should explicitly wait:
Runner polish, review write-back helpers, and any broader workflow or runner build-out.

codex
I’ve narrowed the implementation to a single-file updater that validates the expected V1 sections and rewrites only the specific field lines in place. Next I’m adding the helper, then I’ll run it against the smoke-test record and capture the validation in a short note.
apply patch
patch: completed
/data/git/handson-ml3/tools/codex/review_run.py
diff --git a/tools/codex/review_run.py b/tools/codex/review_run.py
new file mode 100644
index 0000000000000000000000000000000000000000..11818b1bd53c773d264942cd4fb6f0609068581f
--- /dev/null
+++ b/tools/codex/review_run.py
@@ -0,0 +1,201 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import re
+import sys
+from pathlib import Path
+
+
+REVIEW_STATUSES = {"ACCEPTED", "REJECTED"}
+FAILURE_FIELDS = (
+    "failure_type",
+    "failure_symptom",
+    "likely_cause",
+    "recommended_next_action",
+)
+REQUIRED_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+    "execution_status",
+    "finished_at_utc",
+    "runner",
+    "return_code",
+    "retry_of_run_id",
+    "review_status",
+    "review_summary",
+    "reviewed_by",
+    "reviewed_at_utc",
+    *FAILURE_FIELDS,
+    "elapsed_seconds",
+    "final_output_char_count",
+    "stderr_char_count",
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
+
+
+def utc_timestamp() -> str:
+    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Write manual V1 review fields back into an existing execution record."
+    )
+    parser.add_argument("record", help="Path to an existing execution-record markdown file")
+    parser.add_argument(
+        "--review-status",
+        required=True,
+        choices=sorted(REVIEW_STATUSES),
+        help="Manual review outcome",
+    )
+    parser.add_argument(
+        "--review-summary",
+        required=True,
+        help="Short manual review summary",
+    )
+    parser.add_argument("--reviewed-by", help="Reviewer identifier")
+    parser.add_argument(
+        "--reviewed-at-utc",
+        help="Review timestamp in UTC using YYYYMMDD_HHMMSS; defaults to current UTC time",
+    )
+    parser.add_argument("--failure-type", help="Short failure category for rejected runs")
+    parser.add_argument("--failure-symptom", help="Observed failure symptom for rejected runs")
+    parser.add_argument("--likely-cause", help="Likely root cause for rejected runs")
+    parser.add_argument(
+        "--recommended-next-action",
+        help="Manual next action recommendation for rejected runs",
+    )
+    return parser.parse_args()
+
+
+def fail(message: str) -> int:
+    print(f"ERROR: {message}", file=sys.stderr)
+    return 1
+
+
+def require_single_line(name: str, value: str | None) -> str | None:
+    if value is None:
+        return None
+    if "\n" in value or "\r" in value:
+        raise ValueError(f"{name} must be a single line")
+    return value.strip()
+
+
+def resolve_record_path(record_arg: str) -> Path:
+    record_path = Path(record_arg)
+    if record_path.is_absolute():
+        return record_path
+    return repo_root() / record_path
+
+
+def validate_record_path(record_path: Path) -> None:
+    if not record_path.exists():
+        raise ValueError(f"record file not found: {record_path}")
+    if not record_path.is_file():
+        raise ValueError(f"record path is not a file: {record_path}")
+
+    root = repo_root().resolve()
+    notes_dir = root / "notes"
+    try:
+        record_path.resolve().relative_to(notes_dir.resolve())
+    except ValueError as exc:
+        raise ValueError(f"record must be under {notes_dir}") from exc
+
+
+def validate_v1_record_structure(text: str) -> None:
+    if not text.startswith("# "):
+        raise ValueError("record does not start with a markdown title")
+
+    positions: list[int] = []
+    for section in REQUIRED_SECTIONS:
+        pos = text.find(section)
+        if pos == -1:
+            raise ValueError(f"record is missing section: {section}")
+        positions.append(pos)
+    if positions != sorted(positions):
+        raise ValueError("record sections are out of the expected V1 order")
+
+    for field in REQUIRED_FIELDS:
+        if re.search(rf"^- {re.escape(field)}:", text, flags=re.MULTILINE) is None:
+            raise ValueError(f"record is missing field line: {field}")
+
+
+def replace_field(text: str, field: str, value: str | None, *, code: bool = False) -> str:
+    rendered = f"`{value}`" if code and value else (value or "")
+    pattern = re.compile(rf"(?m)^- {re.escape(field)}:.*$")
+    if pattern.search(text) is None:
+        raise ValueError(f"record is missing field line: {field}")
+    return pattern.sub(f"- {field}: {rendered}", text, count=1)
+
+
+def build_updates(args: argparse.Namespace) -> dict[str, tuple[str | None, bool]]:
+    review_summary = require_single_line("review_summary", args.review_summary)
+    reviewed_by = require_single_line("reviewed_by", args.reviewed_by)
+    reviewed_at_utc = require_single_line("reviewed_at_utc", args.reviewed_at_utc) or utc_timestamp()
+    failure_values = {
+        "failure_type": require_single_line("failure_type", args.failure_type),
+        "failure_symptom": require_single_line("failure_symptom", args.failure_symptom),
+        "likely_cause": require_single_line("likely_cause", args.likely_cause),
+        "recommended_next_action": require_single_line(
+            "recommended_next_action", args.recommended_next_action
+        ),
+    }
+
+    if args.review_status != "REJECTED" and any(value for value in failure_values.values()):
+        raise ValueError("rejection-only failure fields may only be set when --review-status REJECTED")
+
+    updates: dict[str, tuple[str | None, bool]] = {
+        "review_status": (args.review_status, True),
+        "review_summary": (review_summary, False),
+        "reviewed_at_utc": (reviewed_at_utc, True),
+    }
+    if reviewed_by is not None:
+        updates["reviewed_by"] = (reviewed_by, False)
+
+    if args.review_status == "REJECTED":
+        for field, value in failure_values.items():
+            if value is not None:
+                updates[field] = (value, False)
+
+    return updates
+
+
+def main() -> int:
+    try:
+        args = parse_args()
+        record_path = resolve_record_path(args.record)
+        validate_record_path(record_path)
+        text = record_path.read_text(encoding="utf-8")
+        validate_v1_record_structure(text)
+
+        updated = text
+        for field, (value, code) in build_updates(args).items():
+            updated = replace_field(updated, field, value, code=code)
+
+        record_path.write_text(updated, encoding="utf-8")
+    except ValueError as exc:
+        return fail(str(exc))
+
+    print(record_path)
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

diff --git a/tools/codex/review_run.py b/tools/codex/review_run.py
new file mode 100644
index 0000000000000000000000000000000000000000..11818b1bd53c773d264942cd4fb6f0609068581f
--- /dev/null
+++ b/tools/codex/review_run.py
@@ -0,0 +1,201 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import re
+import sys
+from pathlib import Path
+
+
+REVIEW_STATUSES = {"ACCEPTED", "REJECTED"}
+FAILURE_FIELDS = (
+    "failure_type",
+    "failure_symptom",
+    "likely_cause",
+    "recommended_next_action",
+)
+REQUIRED_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+    "execution_status",
+    "finished_at_utc",
+    "runner",
+    "return_code",
+    "retry_of_run_id",
+    "review_status",
+    "review_summary",
+    "reviewed_by",
+    "reviewed_at_utc",
+    *FAILURE_FIELDS,
+    "elapsed_seconds",
+    "final_output_char_count",
+    "stderr_char_count",
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
+
+
+def utc_timestamp() -> str:
+    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Write manual V1 review fields back into an existing execution record."
+    )
+    parser.add_argument("record", help="Path to an existing execution-record markdown file")
+    parser.add_argument(
+        "--review-status",
+        required=True,
+        choices=sorted(REVIEW_STATUSES),
+        help="Manual review outcome",
+    )
+    parser.add_argument(
+        "--review-summary",
+        required=True,
+        help="Short manual review summary",
+    )
+    parser.add_argument("--reviewed-by", help="Reviewer identifier")
+    parser.add_argument(
+        "--reviewed-at-utc",
+        help="Review timestamp in UTC using YYYYMMDD_HHMMSS; defaults to current UTC time",
+    )
+    parser.add_argument("--failure-type", help="Short failure category for rejected runs")
+    parser.add_argument("--failure-symptom", help="Observed failure symptom for rejected runs")
+    parser.add_argument("--likely-cause", help="Likely root cause for rejected runs")
+    parser.add_argument(
+        "--recommended-next-action",
+        help="Manual next action recommendation for rejected runs",
+    )
+    return parser.parse_args()
+
+
+def fail(message: str) -> int:
+    print(f"ERROR: {message}", file=sys.stderr)
+    return 1
+
+
+def require_single_line(name: str, value: str | None) -> str | None:
+    if value is None:
+        return None
+    if "\n" in value or "\r" in value:
+        raise ValueError(f"{name} must be a single line")
+    return value.strip()
+
+
+def resolve_record_path(record_arg: str) -> Path:
+    record_path = Path(record_arg)
+    if record_path.is_absolute():
+        return record_path
+    return repo_root() / record_path
+
+
+def validate_record_path(record_path: Path) -> None:
+    if not record_path.exists():
+        raise ValueError(f"record file not found: {record_path}")
+    if not record_path.is_file():
+        raise ValueError(f"record path is not a file: {record_path}")
+
+    root = repo_root().resolve()
+    notes_dir = root / "notes"
+    try:
+        record_path.resolve().relative_to(notes_dir.resolve())
+    except ValueError as exc:
+        raise ValueError(f"record must be under {notes_dir}") from exc
+
+
+def validate_v1_record_structure(text: str) -> None:
+    if not text.startswith("# "):
+        raise ValueError("record does not start with a markdown title")
+
+    positions: list[int] = []
+    for section in REQUIRED_SECTIONS:
+        pos = text.find(section)
+        if pos == -1:
+            raise ValueError(f"record is missing section: {section}")
+        positions.append(pos)
+    if positions != sorted(positions):
+        raise ValueError("record sections are out of the expected V1 order")
+
+    for field in REQUIRED_FIELDS:
+        if re.search(rf"^- {re.escape(field)}:", text, flags=re.MULTILINE) is None:
+            raise ValueError(f"record is missing field line: {field}")
+
+
+def replace_field(text: str, field: str, value: str | None, *, code: bool = False) -> str:
+    rendered = f"`{value}`" if code and value else (value or "")
+    pattern = re.compile(rf"(?m)^- {re.escape(field)}:.*$")
+    if pattern.search(text) is None:
+        raise ValueError(f"record is missing field line: {field}")
+    return pattern.sub(f"- {field}: {rendered}", text, count=1)
+
+
+def build_updates(args: argparse.Namespace) -> dict[str, tuple[str | None, bool]]:
+    review_summary = require_single_line("review_summary", args.review_summary)
+    reviewed_by = require_single_line("reviewed_by", args.reviewed_by)
+    reviewed_at_utc = require_single_line("reviewed_at_utc", args.reviewed_at_utc) or utc_timestamp()
+    failure_values = {
+        "failure_type": require_single_line("failure_type", args.failure_type),
+        "failure_symptom": require_single_line("failure_symptom", args.failure_symptom),
+        "likely_cause": require_single_line("likely_cause", args.likely_cause),
+        "recommended_next_action": require_single_line(
+            "recommended_next_action", args.recommended_next_action
+        ),
+    }
+
+    if args.review_status != "REJECTED" and any(value for value in failure_values.values()):
+        raise ValueError("rejection-only failure fields may only be set when --review-status REJECTED")
+
+    updates: dict[str, tuple[str | None, bool]] = {
+        "review_status": (args.review_status, True),
+        "review_summary": (review_summary, False),
+        "reviewed_at_utc": (reviewed_at_utc, True),
+    }
+    if reviewed_by is not None:
+        updates["reviewed_by"] = (reviewed_by, False)
+
+    if args.review_status == "REJECTED":
+        for field, value in failure_values.items():
+            if value is not None:
+                updates[field] = (value, False)
+
+    return updates
+
+
+def main() -> int:
+    try:
+        args = parse_args()
+        record_path = resolve_record_path(args.record)
+        validate_record_path(record_path)
+        text = record_path.read_text(encoding="utf-8")
+        validate_v1_record_structure(text)
+
+        updated = text
+        for field, (value, code) in build_updates(args).items():
+            updated = replace_field(updated, field, value, code=code)
+
+        record_path.write_text(updated, encoding="utf-8")
+    except ValueError as exc:
+        return fail(str(exc))
+
+    print(record_path)
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

codex
The helper file is in place. I’m running a narrow validation pass now: syntax check, update the smoke-test record to `ACCEPTED`, inspect the changed fields, and confirm both runner files still have no diff.
exec
/bin/bash -lc 'python3 -m py_compile tools/codex/review_run.py' in /data/git/handson-ml3
 succeeded in 0ms:
exec
/bin/bash -lc 'git diff -- tools/codex/run_prompt.py tools/codex/baby_run_prompt.py' in /data/git/handson-ml3
 succeeded in 0ms:
exec
/bin/bash -lc "sed -n '1,220p' tools/codex/review_run.py" in /data/git/handson-ml3
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

diff --git a/tools/codex/review_run.py b/tools/codex/review_run.py
new file mode 100644
index 0000000000000000000000000000000000000000..11818b1bd53c773d264942cd4fb6f0609068581f
--- /dev/null
+++ b/tools/codex/review_run.py
@@ -0,0 +1,201 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import re
+import sys
+from pathlib import Path
+
+
+REVIEW_STATUSES = {"ACCEPTED", "REJECTED"}
+FAILURE_FIELDS = (
+    "failure_type",
+    "failure_symptom",
+    "likely_cause",
+    "recommended_next_action",
+)
+REQUIRED_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+    "execution_status",
+    "finished_at_utc",
+    "runner",
+    "return_code",
+    "retry_of_run_id",
+    "review_status",
+    "review_summary",
+    "reviewed_by",
+    "reviewed_at_utc",
+    *FAILURE_FIELDS,
+    "elapsed_seconds",
+    "final_output_char_count",
+    "stderr_char_count",
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
+
+
+def utc_timestamp() -> str:
+    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Write manual V1 review fields back into an existing execution record."
+    )
+    parser.add_argument("record", help="Path to an existing execution-record markdown file")
+    parser.add_argument(
+        "--review-status",
+        required=True,
+        choices=sorted(REVIEW_STATUSES),
+        help="Manual review outcome",
+    )
+    parser.add_argument(
+        "--review-summary",
+        required=True,
+        help="Short manual review summary",
+    )
+    parser.add_argument("--reviewed-by", help="Reviewer identifier")
+    parser.add_argument(
+        "--reviewed-at-utc",
+        help="Review timestamp in UTC using YYYYMMDD_HHMMSS; defaults to current UTC time",
+    )
+    parser.add_argument("--failure-type", help="Short failure category for rejected runs")
+    parser.add_argument("--failure-symptom", help="Observed failure symptom for rejected runs")
+    parser.add_argument("--likely-cause", help="Likely root cause for rejected runs")
+    parser.add_argument(
+        "--recommended-next-action",
+        help="Manual next action recommendation for rejected runs",
+    )
+    return parser.parse_args()
+
+
+def fail(message: str) -> int:
+    print(f"ERROR: {message}", file=sys.stderr)
+    return 1
+
+
+def require_single_line(name: str, value: str | None) -> str | None:
+    if value is None:
+        return None
+    if "\n" in value or "\r" in value:
+        raise ValueError(f"{name} must be a single line")
+    return value.strip()
+
+
+def resolve_record_path(record_arg: str) -> Path:
+    record_path = Path(record_arg)
+    if record_path.is_absolute():
+        return record_path
+    return repo_root() / record_path
+
+
+def validate_record_path(record_path: Path) -> None:
+    if not record_path.exists():
+        raise ValueError(f"record file not found: {record_path}")
+    if not record_path.is_file():
+        raise ValueError(f"record path is not a file: {record_path}")
+
+    root = repo_root().resolve()
+    notes_dir = root / "notes"
+    try:
+        record_path.resolve().relative_to(notes_dir.resolve())
+    except ValueError as exc:
+        raise ValueError(f"record must be under {notes_dir}") from exc
+
+
+def validate_v1_record_structure(text: str) -> None:
+    if not text.startswith("# "):
+        raise ValueError("record does not start with a markdown title")
+
+    positions: list[int] = []
+    for section in REQUIRED_SECTIONS:
+        pos = text.find(section)
+        if pos == -1:
+            raise ValueError(f"record is missing section: {section}")
+        positions.append(pos)
+    if positions != sorted(positions):
+        raise ValueError("record sections are out of the expected V1 order")
+
+    for field in REQUIRED_FIELDS:
+        if re.search(rf"^- {re.escape(field)}:", text, flags=re.MULTILINE) is None:
+            raise ValueError(f"record is missing field line: {field}")
+
+
+def replace_field(text: str, field: str, value: str | None, *, code: bool = False) -> str:
+    rendered = f"`{value}`" if code and value else (value or "")
+    pattern = re.compile(rf"(?m)^- {re.escape(field)}:.*$")
+    if pattern.search(text) is None:
+        raise ValueError(f"record is missing field line: {field}")
+    return pattern.sub(f"- {field}: {rendered}", text, count=1)
+
+
+def build_updates(args: argparse.Namespace) -> dict[str, tuple[str | None, bool]]:
+    review_summary = require_single_line("review_summary", args.review_summary)
+    reviewed_by = require_single_line("reviewed_by", args.reviewed_by)
+    reviewed_at_utc = require_single_line("reviewed_at_utc", args.reviewed_at_utc) or utc_timestamp()
+    failure_values = {
+        "failure_type": require_single_line("failure_type", args.failure_type),
+        "failure_symptom": require_single_line("failure_symptom", args.failure_symptom),
+        "likely_cause": require_single_line("likely_cause", args.likely_cause),
+        "recommended_next_action": require_single_line(
+            "recommended_next_action", args.recommended_next_action
+        ),
+    }
+
+    if args.review_status != "REJECTED" and any(value for value in failure_values.values()):
+        raise ValueError("rejection-only failure fields may only be set when --review-status REJECTED")
+
+    updates: dict[str, tuple[str | None, bool]] = {
+        "review_status": (args.review_status, True),
+        "review_summary": (review_summary, False),
+        "reviewed_at_utc": (reviewed_at_utc, True),
+    }
+    if reviewed_by is not None:
+        updates["reviewed_by"] = (reviewed_by, False)
+
+    if args.review_status == "REJECTED":
+        for field, value in failure_values.items():
+            if value is not None:
+                updates[field] = (value, False)
+
+    return updates
+
+
+def main() -> int:
+    try:
+        args = parse_args()
+        record_path = resolve_record_path(args.record)
+        validate_record_path(record_path)
+        text = record_path.read_text(encoding="utf-8")
+        validate_v1_record_structure(text)
+
+        updated = text
+        for field, (value, code) in build_updates(args).items():
+            updated = replace_field(updated, field, value, code=code)
+
+        record_path.write_text(updated, encoding="utf-8")
+    except ValueError as exc:
+        return fail(str(exc))
+
+    print(record_path)
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

exec
/bin/bash -lc "python3 tools/codex/review_run.py notes/001_smoke_test_pipeline__20260415_234918.md --review-status ACCEPTED --review-summary 'Smoke test output and artifact are complete enough to accept this bounded step.'" in /data/git/handson-ml3
exec
/bin/bash -lc "sed -n '1,80p' notes/001_smoke_test_pipeline__20260415_234918.md" in /data/git/handson-ml3
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


exec
/bin/bash -lc 'git diff -- notes/001_smoke_test_pipeline__20260415_234918.md' in /data/git/handson-ml3
 succeeded in 0ms:
diff --git a/tools/codex/review_run.py b/tools/codex/review_run.py
new file mode 100644
index 0000000000000000000000000000000000000000..11818b1bd53c773d264942cd4fb6f0609068581f
--- /dev/null
+++ b/tools/codex/review_run.py
@@ -0,0 +1,201 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import re
+import sys
+from pathlib import Path
+
+
+REVIEW_STATUSES = {"ACCEPTED", "REJECTED"}
+FAILURE_FIELDS = (
+    "failure_type",
+    "failure_symptom",
+    "likely_cause",
+    "recommended_next_action",
+)
+REQUIRED_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+    "execution_status",
+    "finished_at_utc",
+    "runner",
+    "return_code",
+    "retry_of_run_id",
+    "review_status",
+    "review_summary",
+    "reviewed_by",
+    "reviewed_at_utc",
+    *FAILURE_FIELDS,
+    "elapsed_seconds",
+    "final_output_char_count",
+    "stderr_char_count",
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
+
+
+def utc_timestamp() -> str:
+    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Write manual V1 review fields back into an existing execution record."
+    )
+    parser.add_argument("record", help="Path to an existing execution-record markdown file")
+    parser.add_argument(
+        "--review-status",
+        required=True,
+        choices=sorted(REVIEW_STATUSES),
+        help="Manual review outcome",
+    )
+    parser.add_argument(
+        "--review-summary",
+        required=True,
+        help="Short manual review summary",
+    )
+    parser.add_argument("--reviewed-by", help="Reviewer identifier")
+    parser.add_argument(
+        "--reviewed-at-utc",
+        help="Review timestamp in UTC using YYYYMMDD_HHMMSS; defaults to current UTC time",
+    )
+    parser.add_argument("--failure-type", help="Short failure category for rejected runs")
+    parser.add_argument("--failure-symptom", help="Observed failure symptom for rejected runs")
+    parser.add_argument("--likely-cause", help="Likely root cause for rejected runs")
+    parser.add_argument(
+        "--recommended-next-action",
+        help="Manual next action recommendation for rejected runs",
+    )
+    return parser.parse_args()
+
+
+def fail(message: str) -> int:
+    print(f"ERROR: {message}", file=sys.stderr)
+    return 1
+
+
+def require_single_line(name: str, value: str | None) -> str | None:
+    if value is None:
+        return None
+    if "\n" in value or "\r" in value:
+        raise ValueError(f"{name} must be a single line")
+    return value.strip()
+
+
+def resolve_record_path(record_arg: str) -> Path:
+    record_path = Path(record_arg)
+    if record_path.is_absolute():
+        return record_path
+    return repo_root() / record_path
+
+
+def validate_record_path(record_path: Path) -> None:
+    if not record_path.exists():
+        raise ValueError(f"record file not found: {record_path}")
+    if not record_path.is_file():
+        raise ValueError(f"record path is not a file: {record_path}")
+
+    root = repo_root().resolve()
+    notes_dir = root / "notes"
+    try:
+        record_path.resolve().relative_to(notes_dir.resolve())
+    except ValueError as exc:
+        raise ValueError(f"record must be under {notes_dir}") from exc
+
+
+def validate_v1_record_structure(text: str) -> None:
+    if not text.startswith("# "):
+        raise ValueError("record does not start with a markdown title")
+
+    positions: list[int] = []
+    for section in REQUIRED_SECTIONS:
+        pos = text.find(section)
+        if pos == -1:
+            raise ValueError(f"record is missing section: {section}")
+        positions.append(pos)
+    if positions != sorted(positions):
+        raise ValueError("record sections are out of the expected V1 order")
+
+    for field in REQUIRED_FIELDS:
+        if re.search(rf"^- {re.escape(field)}:", text, flags=re.MULTILINE) is None:
+            raise ValueError(f"record is missing field line: {field}")
+
+
+def replace_field(text: str, field: str, value: str | None, *, code: bool = False) -> str:
+    rendered = f"`{value}`" if code and value else (value or "")
+    pattern = re.compile(rf"(?m)^- {re.escape(field)}:.*$")
+    if pattern.search(text) is None:
+        raise ValueError(f"record is missing field line: {field}")
+    return pattern.sub(f"- {field}: {rendered}", text, count=1)
+
+
+def build_updates(args: argparse.Namespace) -> dict[str, tuple[str | None, bool]]:
+    review_summary = require_single_line("review_summary", args.review_summary)
+    reviewed_by = require_single_line("reviewed_by", args.reviewed_by)
+    reviewed_at_utc = require_single_line("reviewed_at_utc", args.reviewed_at_utc) or utc_timestamp()
+    failure_values = {
+        "failure_type": require_single_line("failure_type", args.failure_type),
+        "failure_symptom": require_single_line("failure_symptom", args.failure_symptom),
+        "likely_cause": require_single_line("likely_cause", args.likely_cause),
+        "recommended_next_action": require_single_line(
+            "recommended_next_action", args.recommended_next_action
+        ),
+    }
+
+    if args.review_status != "REJECTED" and any(value for value in failure_values.values()):
+        raise ValueError("rejection-only failure fields may only be set when --review-status REJECTED")
+
+    updates: dict[str, tuple[str | None, bool]] = {
+        "review_status": (args.review_status, True),
+        "review_summary": (review_summary, False),
+        "reviewed_at_utc": (reviewed_at_utc, True),
+    }
+    if reviewed_by is not None:
+        updates["reviewed_by"] = (reviewed_by, False)
+
+    if args.review_status == "REJECTED":
+        for field, value in failure_values.items():
+            if value is not None:
+                updates[field] = (value, False)
+
+    return updates
+
+
+def main() -> int:
+    try:
+        args = parse_args()
+        record_path = resolve_record_path(args.record)
+        validate_record_path(record_path)
+        text = record_path.read_text(encoding="utf-8")
+        validate_v1_record_structure(text)
+
+        updated = text
+        for field, (value, code) in build_updates(args).items():
+            updated = replace_field(updated, field, value, code=code)
+
+        record_path.write_text(updated, encoding="utf-8")
+    except ValueError as exc:
+        return fail(str(exc))
+
+    print(record_path)
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

 succeeded in 197ms:
/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md

codex
The write-back command completed, but the read was issued in parallel and caught the pre-update state. I’m re-reading the record sequentially now and then I’ll write the implementation note with the exact validation result.
exec
/bin/bash -lc "sed -n '1,80p' notes/001_smoke_test_pipeline__20260415_234918.md" in /data/git/handson-ml3
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


exec
/bin/bash -lc 'git diff -- notes/001_smoke_test_pipeline__20260415_234918.md tools/codex/review_run.py' in /data/git/handson-ml3
 succeeded in 0ms:
diff --git a/notes/001_smoke_test_pipeline__20260415_234918.md b/notes/001_smoke_test_pipeline__20260415_234918.md
index 3894de6..e41f87a 100644
--- a/notes/001_smoke_test_pipeline__20260415_234918.md
+++ b/notes/001_smoke_test_pipeline__20260415_234918.md
@@ -15,10 +15,10 @@
 
 ## Review Facts
 
-- review_status: `UNREVIEWED`
-- review_summary:
+- review_status: `ACCEPTED`
+- review_summary: Smoke test output and artifact are complete enough to accept this bounded step.
 - reviewed_by:
-- reviewed_at_utc:
+- reviewed_at_utc: `20260415_235508`
 
 ## Failure Analysis
 

exec
/bin/bash -lc "date -u '+%Y%m%d_%H%M%S'" in /data/git/handson-ml3
 succeeded in 0ms:
20260415_235514

diff --git a/tools/codex/review_run.py b/tools/codex/review_run.py
new file mode 100644
index 0000000000000000000000000000000000000000..11818b1bd53c773d264942cd4fb6f0609068581f
--- /dev/null
+++ b/tools/codex/review_run.py
@@ -0,0 +1,201 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import re
+import sys
+from pathlib import Path
+
+
+REVIEW_STATUSES = {"ACCEPTED", "REJECTED"}
+FAILURE_FIELDS = (
+    "failure_type",
+    "failure_symptom",
+    "likely_cause",
+    "recommended_next_action",
+)
+REQUIRED_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+    "execution_status",
+    "finished_at_utc",
+    "runner",
+    "return_code",
+    "retry_of_run_id",
+    "review_status",
+    "review_summary",
+    "reviewed_by",
+    "reviewed_at_utc",
+    *FAILURE_FIELDS,
+    "elapsed_seconds",
+    "final_output_char_count",
+    "stderr_char_count",
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
+
+
+def utc_timestamp() -> str:
+    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Write manual V1 review fields back into an existing execution record."
+    )
+    parser.add_argument("record", help="Path to an existing execution-record markdown file")
+    parser.add_argument(
+        "--review-status",
+        required=True,
+        choices=sorted(REVIEW_STATUSES),
+        help="Manual review outcome",
+    )
+    parser.add_argument(
+        "--review-summary",
+        required=True,
+        help="Short manual review summary",
+    )
+    parser.add_argument("--reviewed-by", help="Reviewer identifier")
+    parser.add_argument(
+        "--reviewed-at-utc",
+        help="Review timestamp in UTC using YYYYMMDD_HHMMSS; defaults to current UTC time",
+    )
+    parser.add_argument("--failure-type", help="Short failure category for rejected runs")
+    parser.add_argument("--failure-symptom", help="Observed failure symptom for rejected runs")
+    parser.add_argument("--likely-cause", help="Likely root cause for rejected runs")
+    parser.add_argument(
+        "--recommended-next-action",
+        help="Manual next action recommendation for rejected runs",
+    )
+    return parser.parse_args()
+
+
+def fail(message: str) -> int:
+    print(f"ERROR: {message}", file=sys.stderr)
+    return 1
+
+
+def require_single_line(name: str, value: str | None) -> str | None:
+    if value is None:
+        return None
+    if "\n" in value or "\r" in value:
+        raise ValueError(f"{name} must be a single line")
+    return value.strip()
+
+
+def resolve_record_path(record_arg: str) -> Path:
+    record_path = Path(record_arg)
+    if record_path.is_absolute():
+        return record_path
+    return repo_root() / record_path
+
+
+def validate_record_path(record_path: Path) -> None:
+    if not record_path.exists():
+        raise ValueError(f"record file not found: {record_path}")
+    if not record_path.is_file():
+        raise ValueError(f"record path is not a file: {record_path}")
+
+    root = repo_root().resolve()
+    notes_dir = root / "notes"
+    try:
+        record_path.resolve().relative_to(notes_dir.resolve())
+    except ValueError as exc:
+        raise ValueError(f"record must be under {notes_dir}") from exc
+
+
+def validate_v1_record_structure(text: str) -> None:
+    if not text.startswith("# "):
+        raise ValueError("record does not start with a markdown title")
+
+    positions: list[int] = []
+    for section in REQUIRED_SECTIONS:
+        pos = text.find(section)
+        if pos == -1:
+            raise ValueError(f"record is missing section: {section}")
+        positions.append(pos)
+    if positions != sorted(positions):
+        raise ValueError("record sections are out of the expected V1 order")
+
+    for field in REQUIRED_FIELDS:
+        if re.search(rf"^- {re.escape(field)}:", text, flags=re.MULTILINE) is None:
+            raise ValueError(f"record is missing field line: {field}")
+
+
+def replace_field(text: str, field: str, value: str | None, *, code: bool = False) -> str:
+    rendered = f"`{value}`" if code and value else (value or "")
+    pattern = re.compile(rf"(?m)^- {re.escape(field)}:.*$")
+    if pattern.search(text) is None:
+        raise ValueError(f"record is missing field line: {field}")
+    return pattern.sub(f"- {field}: {rendered}", text, count=1)
+
+
+def build_updates(args: argparse.Namespace) -> dict[str, tuple[str | None, bool]]:
+    review_summary = require_single_line("review_summary", args.review_summary)
+    reviewed_by = require_single_line("reviewed_by", args.reviewed_by)
+    reviewed_at_utc = require_single_line("reviewed_at_utc", args.reviewed_at_utc) or utc_timestamp()
+    failure_values = {
+        "failure_type": require_single_line("failure_type", args.failure_type),
+        "failure_symptom": require_single_line("failure_symptom", args.failure_symptom),
+        "likely_cause": require_single_line("likely_cause", args.likely_cause),
+        "recommended_next_action": require_single_line(
+            "recommended_next_action", args.recommended_next_action
+        ),
+    }
+
+    if args.review_status != "REJECTED" and any(value for value in failure_values.values()):
+        raise ValueError("rejection-only failure fields may only be set when --review-status REJECTED")
+
+    updates: dict[str, tuple[str | None, bool]] = {
+        "review_status": (args.review_status, True),
+        "review_summary": (review_summary, False),
+        "reviewed_at_utc": (reviewed_at_utc, True),
+    }
+    if reviewed_by is not None:
+        updates["reviewed_by"] = (reviewed_by, False)
+
+    if args.review_status == "REJECTED":
+        for field, value in failure_values.items():
+            if value is not None:
+                updates[field] = (value, False)
+
+    return updates
+
+
+def main() -> int:
+    try:
+        args = parse_args()
+        record_path = resolve_record_path(args.record)
+        validate_record_path(record_path)
+        text = record_path.read_text(encoding="utf-8")
+        validate_v1_record_structure(text)
+
+        updated = text
+        for field, (value, code) in build_updates(args).items():
+            updated = replace_field(updated, field, value, code=code)
+
+        record_path.write_text(updated, encoding="utf-8")
+    except ValueError as exc:
+        return fail(str(exc))
+
+    print(record_path)
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

codex
The validation is clean: the record was updated in place, only the review fields changed, and both runner files still show no diff. I’m adding the short implementation note now with the scope, explicit non-goals, and the validation evidence.
apply patch
patch: completed
/data/git/handson-ml3/notes/011_review_writeback_helper_build__20260415_235514.md
diff --git a/notes/011_review_writeback_helper_build__20260415_235514.md b/notes/011_review_writeback_helper_build__20260415_235514.md
new file mode 100644
index 0000000000000000000000000000000000000000..1b82c53528febd6e9dd2fe9f2b500bab7e75b4e4
--- /dev/null
+++ b/notes/011_review_writeback_helper_build__20260415_235514.md
@@ -0,0 +1,64 @@
+# 011 Review Write-Back Helper Build
+
+## What Was Built
+
+Built [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py), a thin V1 companion helper that updates one existing execution record in `notes/` in place.
+
+The helper:
+
+- accepts a record path plus explicit review arguments
+- verifies the file is under `notes/` and matches the minimum V1 execution-record structure
+- updates only the targeted field lines
+- writes the changes back to the same markdown file
+- prints the updated file path
+- exits nonzero on invalid input or malformed structure
+
+## V1 Fields It Updates
+
+Always updates:
+
+- `review_status`
+- `review_summary`
+- `reviewed_at_utc`
+
+Optionally updates:
+
+- `reviewed_by`
+
+For `REJECTED` runs, it can also update:
+
+- `failure_type`
+- `failure_symptom`
+- `likely_cause`
+- `recommended_next_action`
+
+## What It Intentionally Does Not Do
+
+- does not modify `tools/codex/run_prompt.py`
+- does not modify `tools/codex/baby_run_prompt.py`
+- does not create queue state, sidecars, databases, or extra workflow files
+- does not release the next prompt automatically
+- does not redesign the V1 record format
+- does not rewrite or reorder non-review sections
+
+## Validation
+
+Validation commands performed:
+
+- `python3 -m py_compile tools/codex/review_run.py`
+- `python3 tools/codex/review_run.py notes/001_smoke_test_pipeline__20260415_234918.md --review-status ACCEPTED --review-summary 'Smoke test output and artifact are complete enough to accept this bounded step.'`
+- `git diff -- tools/codex/run_prompt.py tools/codex/baby_run_prompt.py`
+
+Validation outcome:
+
+- `review_status` in [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md) changed from `UNREVIEWED` to `ACCEPTED`.
+- `review_summary` was written in place.
+- `reviewed_at_utc` was auto-filled as `20260415_235508`.
+- All other sections remained unchanged; the diff was limited to the three review lines above.
+- The updated record still matches the expected V1 section order and required field layout.
+- `tools/codex/run_prompt.py` remained unchanged.
+- `tools/codex/baby_run_prompt.py` remained unchanged.
+
+## Scope Note
+
+This helper is intentionally a minimum V1 review write-back tool. It updates review facts on one existing execution record and stops there.
diff --git a/tools/codex/review_run.py b/tools/codex/review_run.py
new file mode 100644
index 0000000000000000000000000000000000000000..11818b1bd53c773d264942cd4fb6f0609068581f
--- /dev/null
+++ b/tools/codex/review_run.py
@@ -0,0 +1,201 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import re
+import sys
+from pathlib import Path
+
+
+REVIEW_STATUSES = {"ACCEPTED", "REJECTED"}
+FAILURE_FIELDS = (
+    "failure_type",
+    "failure_symptom",
+    "likely_cause",
+    "recommended_next_action",
+)
+REQUIRED_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+    "execution_status",
+    "finished_at_utc",
+    "runner",
+    "return_code",
+    "retry_of_run_id",
+    "review_status",
+    "review_summary",
+    "reviewed_by",
+    "reviewed_at_utc",
+    *FAILURE_FIELDS,
+    "elapsed_seconds",
+    "final_output_char_count",
+    "stderr_char_count",
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
+
+
+def utc_timestamp() -> str:
+    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Write manual V1 review fields back into an existing execution record."
+    )
+    parser.add_argument("record", help="Path to an existing execution-record markdown file")
+    parser.add_argument(
+        "--review-status",
+        required=True,
+        choices=sorted(REVIEW_STATUSES),
+        help="Manual review outcome",
+    )
+    parser.add_argument(
+        "--review-summary",
+        required=True,
+        help="Short manual review summary",
+    )
+    parser.add_argument("--reviewed-by", help="Reviewer identifier")
+    parser.add_argument(
+        "--reviewed-at-utc",
+        help="Review timestamp in UTC using YYYYMMDD_HHMMSS; defaults to current UTC time",
+    )
+    parser.add_argument("--failure-type", help="Short failure category for rejected runs")
+    parser.add_argument("--failure-symptom", help="Observed failure symptom for rejected runs")
+    parser.add_argument("--likely-cause", help="Likely root cause for rejected runs")
+    parser.add_argument(
+        "--recommended-next-action",
+        help="Manual next action recommendation for rejected runs",
+    )
+    return parser.parse_args()
+
+
+def fail(message: str) -> int:
+    print(f"ERROR: {message}", file=sys.stderr)
+    return 1
+
+
+def require_single_line(name: str, value: str | None) -> str | None:
+    if value is None:
+        return None
+    if "\n" in value or "\r" in value:
+        raise ValueError(f"{name} must be a single line")
+    return value.strip()
+
+
+def resolve_record_path(record_arg: str) -> Path:
+    record_path = Path(record_arg)
+    if record_path.is_absolute():
+        return record_path
+    return repo_root() / record_path
+
+
+def validate_record_path(record_path: Path) -> None:
+    if not record_path.exists():
+        raise ValueError(f"record file not found: {record_path}")
+    if not record_path.is_file():
+        raise ValueError(f"record path is not a file: {record_path}")
+
+    root = repo_root().resolve()
+    notes_dir = root / "notes"
+    try:
+        record_path.resolve().relative_to(notes_dir.resolve())
+    except ValueError as exc:
+        raise ValueError(f"record must be under {notes_dir}") from exc
+
+
+def validate_v1_record_structure(text: str) -> None:
+    if not text.startswith("# "):
+        raise ValueError("record does not start with a markdown title")
+
+    positions: list[int] = []
+    for section in REQUIRED_SECTIONS:
+        pos = text.find(section)
+        if pos == -1:
+            raise ValueError(f"record is missing section: {section}")
+        positions.append(pos)
+    if positions != sorted(positions):
+        raise ValueError("record sections are out of the expected V1 order")
+
+    for field in REQUIRED_FIELDS:
+        if re.search(rf"^- {re.escape(field)}:", text, flags=re.MULTILINE) is None:
+            raise ValueError(f"record is missing field line: {field}")
+
+
+def replace_field(text: str, field: str, value: str | None, *, code: bool = False) -> str:
+    rendered = f"`{value}`" if code and value else (value or "")
+    pattern = re.compile(rf"(?m)^- {re.escape(field)}:.*$")
+    if pattern.search(text) is None:
+        raise ValueError(f"record is missing field line: {field}")
+    return pattern.sub(f"- {field}: {rendered}", text, count=1)
+
+
+def build_updates(args: argparse.Namespace) -> dict[str, tuple[str | None, bool]]:
+    review_summary = require_single_line("review_summary", args.review_summary)
+    reviewed_by = require_single_line("reviewed_by", args.reviewed_by)
+    reviewed_at_utc = require_single_line("reviewed_at_utc", args.reviewed_at_utc) or utc_timestamp()
+    failure_values = {
+        "failure_type": require_single_line("failure_type", args.failure_type),
+        "failure_symptom": require_single_line("failure_symptom", args.failure_symptom),
+        "likely_cause": require_single_line("likely_cause", args.likely_cause),
+        "recommended_next_action": require_single_line(
+            "recommended_next_action", args.recommended_next_action
+        ),
+    }
+
+    if args.review_status != "REJECTED" and any(value for value in failure_values.values()):
+        raise ValueError("rejection-only failure fields may only be set when --review-status REJECTED")
+
+    updates: dict[str, tuple[str | None, bool]] = {
+        "review_status": (args.review_status, True),
+        "review_summary": (review_summary, False),
+        "reviewed_at_utc": (reviewed_at_utc, True),
+    }
+    if reviewed_by is not None:
+        updates["reviewed_by"] = (reviewed_by, False)
+
+    if args.review_status == "REJECTED":
+        for field, value in failure_values.items():
+            if value is not None:
+                updates[field] = (value, False)
+
+    return updates
+
+
+def main() -> int:
+    try:
+        args = parse_args()
+        record_path = resolve_record_path(args.record)
+        validate_record_path(record_path)
+        text = record_path.read_text(encoding="utf-8")
+        validate_v1_record_structure(text)
+
+        updated = text
+        for field, (value, code) in build_updates(args).items():
+            updated = replace_field(updated, field, value, code=code)
+
+        record_path.write_text(updated, encoding="utf-8")
+    except ValueError as exc:
+        return fail(str(exc))
+
+    print(record_path)
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

diff --git a/notes/011_review_writeback_helper_build__20260415_235514.md b/notes/011_review_writeback_helper_build__20260415_235514.md
new file mode 100644
index 0000000000000000000000000000000000000000..1b82c53528febd6e9dd2fe9f2b500bab7e75b4e4
--- /dev/null
+++ b/notes/011_review_writeback_helper_build__20260415_235514.md
@@ -0,0 +1,64 @@
+# 011 Review Write-Back Helper Build
+
+## What Was Built
+
+Built [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py), a thin V1 companion helper that updates one existing execution record in `notes/` in place.
+
+The helper:
+
+- accepts a record path plus explicit review arguments
+- verifies the file is under `notes/` and matches the minimum V1 execution-record structure
+- updates only the targeted field lines
+- writes the changes back to the same markdown file
+- prints the updated file path
+- exits nonzero on invalid input or malformed structure
+
+## V1 Fields It Updates
+
+Always updates:
+
+- `review_status`
+- `review_summary`
+- `reviewed_at_utc`
+
+Optionally updates:
+
+- `reviewed_by`
+
+For `REJECTED` runs, it can also update:
+
+- `failure_type`
+- `failure_symptom`
+- `likely_cause`
+- `recommended_next_action`
+
+## What It Intentionally Does Not Do
+
+- does not modify `tools/codex/run_prompt.py`
+- does not modify `tools/codex/baby_run_prompt.py`
+- does not create queue state, sidecars, databases, or extra workflow files
+- does not release the next prompt automatically
+- does not redesign the V1 record format
+- does not rewrite or reorder non-review sections
+
+## Validation
+
+Validation commands performed:
+
+- `python3 -m py_compile tools/codex/review_run.py`
+- `python3 tools/codex/review_run.py notes/001_smoke_test_pipeline__20260415_234918.md --review-status ACCEPTED --review-summary 'Smoke test output and artifact are complete enough to accept this bounded step.'`
+- `git diff -- tools/codex/run_prompt.py tools/codex/baby_run_prompt.py`
+
+Validation outcome:
+
+- `review_status` in [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md) changed from `UNREVIEWED` to `ACCEPTED`.
+- `review_summary` was written in place.
+- `reviewed_at_utc` was auto-filled as `20260415_235508`.
+- All other sections remained unchanged; the diff was limited to the three review lines above.
+- The updated record still matches the expected V1 section order and required field layout.
+- `tools/codex/run_prompt.py` remained unchanged.
+- `tools/codex/baby_run_prompt.py` remained unchanged.
+
+## Scope Note
+
+This helper is intentionally a minimum V1 review write-back tool. It updates review facts on one existing execution record and stops there.
diff --git a/tools/codex/review_run.py b/tools/codex/review_run.py
new file mode 100644
index 0000000000000000000000000000000000000000..11818b1bd53c773d264942cd4fb6f0609068581f
--- /dev/null
+++ b/tools/codex/review_run.py
@@ -0,0 +1,201 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import re
+import sys
+from pathlib import Path
+
+
+REVIEW_STATUSES = {"ACCEPTED", "REJECTED"}
+FAILURE_FIELDS = (
+    "failure_type",
+    "failure_symptom",
+    "likely_cause",
+    "recommended_next_action",
+)
+REQUIRED_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+    "execution_status",
+    "finished_at_utc",
+    "runner",
+    "return_code",
+    "retry_of_run_id",
+    "review_status",
+    "review_summary",
+    "reviewed_by",
+    "reviewed_at_utc",
+    *FAILURE_FIELDS,
+    "elapsed_seconds",
+    "final_output_char_count",
+    "stderr_char_count",
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
+
+
+def utc_timestamp() -> str:
+    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Write manual V1 review fields back into an existing execution record."
+    )
+    parser.add_argument("record", help="Path to an existing execution-record markdown file")
+    parser.add_argument(
+        "--review-status",
+        required=True,
+        choices=sorted(REVIEW_STATUSES),
+        help="Manual review outcome",
+    )
+    parser.add_argument(
+        "--review-summary",
+        required=True,
+        help="Short manual review summary",
+    )
+    parser.add_argument("--reviewed-by", help="Reviewer identifier")
+    parser.add_argument(
+        "--reviewed-at-utc",
+        help="Review timestamp in UTC using YYYYMMDD_HHMMSS; defaults to current UTC time",
+    )
+    parser.add_argument("--failure-type", help="Short failure category for rejected runs")
+    parser.add_argument("--failure-symptom", help="Observed failure symptom for rejected runs")
+    parser.add_argument("--likely-cause", help="Likely root cause for rejected runs")
+    parser.add_argument(
+        "--recommended-next-action",
+        help="Manual next action recommendation for rejected runs",
+    )
+    return parser.parse_args()
+
+
+def fail(message: str) -> int:
+    print(f"ERROR: {message}", file=sys.stderr)
+    return 1
+
+
+def require_single_line(name: str, value: str | None) -> str | None:
+    if value is None:
+        return None
+    if "\n" in value or "\r" in value:
+        raise ValueError(f"{name} must be a single line")
+    return value.strip()
+
+
+def resolve_record_path(record_arg: str) -> Path:
+    record_path = Path(record_arg)
+    if record_path.is_absolute():
+        return record_path
+    return repo_root() / record_path
+
+
+def validate_record_path(record_path: Path) -> None:
+    if not record_path.exists():
+        raise ValueError(f"record file not found: {record_path}")
+    if not record_path.is_file():
+        raise ValueError(f"record path is not a file: {record_path}")
+
+    root = repo_root().resolve()
+    notes_dir = root / "notes"
+    try:
+        record_path.resolve().relative_to(notes_dir.resolve())
+    except ValueError as exc:
+        raise ValueError(f"record must be under {notes_dir}") from exc
+
+
+def validate_v1_record_structure(text: str) -> None:
+    if not text.startswith("# "):
+        raise ValueError("record does not start with a markdown title")
+
+    positions: list[int] = []
+    for section in REQUIRED_SECTIONS:
+        pos = text.find(section)
+        if pos == -1:
+            raise ValueError(f"record is missing section: {section}")
+        positions.append(pos)
+    if positions != sorted(positions):
+        raise ValueError("record sections are out of the expected V1 order")
+
+    for field in REQUIRED_FIELDS:
+        if re.search(rf"^- {re.escape(field)}:", text, flags=re.MULTILINE) is None:
+            raise ValueError(f"record is missing field line: {field}")
+
+
+def replace_field(text: str, field: str, value: str | None, *, code: bool = False) -> str:
+    rendered = f"`{value}`" if code and value else (value or "")
+    pattern = re.compile(rf"(?m)^- {re.escape(field)}:.*$")
+    if pattern.search(text) is None:
+        raise ValueError(f"record is missing field line: {field}")
+    return pattern.sub(f"- {field}: {rendered}", text, count=1)
+
+
+def build_updates(args: argparse.Namespace) -> dict[str, tuple[str | None, bool]]:
+    review_summary = require_single_line("review_summary", args.review_summary)
+    reviewed_by = require_single_line("reviewed_by", args.reviewed_by)
+    reviewed_at_utc = require_single_line("reviewed_at_utc", args.reviewed_at_utc) or utc_timestamp()
+    failure_values = {
+        "failure_type": require_single_line("failure_type", args.failure_type),
+        "failure_symptom": require_single_line("failure_symptom", args.failure_symptom),
+        "likely_cause": require_single_line("likely_cause", args.likely_cause),
+        "recommended_next_action": require_single_line(
+            "recommended_next_action", args.recommended_next_action
+        ),
+    }
+
+    if args.review_status != "REJECTED" and any(value for value in failure_values.values()):
+        raise ValueError("rejection-only failure fields may only be set when --review-status REJECTED")
+
+    updates: dict[str, tuple[str | None, bool]] = {
+        "review_status": (args.review_status, True),
+        "review_summary": (review_summary, False),
+        "reviewed_at_utc": (reviewed_at_utc, True),
+    }
+    if reviewed_by is not None:
+        updates["reviewed_by"] = (reviewed_by, False)
+
+    if args.review_status == "REJECTED":
+        for field, value in failure_values.items():
+            if value is not None:
+                updates[field] = (value, False)
+
+    return updates
+
+
+def main() -> int:
+    try:
+        args = parse_args()
+        record_path = resolve_record_path(args.record)
+        validate_record_path(record_path)
+        text = record_path.read_text(encoding="utf-8")
+        validate_v1_record_structure(text)
+
+        updated = text
+        for field, (value, code) in build_updates(args).items():
+            updated = replace_field(updated, field, value, code=code)
+
+        record_path.write_text(updated, encoding="utf-8")
+    except ValueError as exc:
+        return fail(str(exc))
+
+    print(record_path)
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

  ✓ Read required docs, runners, and sample records to determine V1 execution record structure and review field locations
  ✓ Implement thin in-place review write-back helper at tools/codex/review_run.py
  ✓ Validate helper against notes/001_smoke_test_pipeline__20260415_234918.md and capture results in implementation note
diff --git a/notes/011_review_writeback_helper_build__20260415_235514.md b/notes/011_review_writeback_helper_build__20260415_235514.md
new file mode 100644
index 0000000000000000000000000000000000000000..1b82c53528febd6e9dd2fe9f2b500bab7e75b4e4
--- /dev/null
+++ b/notes/011_review_writeback_helper_build__20260415_235514.md
@@ -0,0 +1,64 @@
+# 011 Review Write-Back Helper Build
+
+## What Was Built
+
+Built [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py), a thin V1 companion helper that updates one existing execution record in `notes/` in place.
+
+The helper:
+
+- accepts a record path plus explicit review arguments
+- verifies the file is under `notes/` and matches the minimum V1 execution-record structure
+- updates only the targeted field lines
+- writes the changes back to the same markdown file
+- prints the updated file path
+- exits nonzero on invalid input or malformed structure
+
+## V1 Fields It Updates
+
+Always updates:
+
+- `review_status`
+- `review_summary`
+- `reviewed_at_utc`
+
+Optionally updates:
+
+- `reviewed_by`
+
+For `REJECTED` runs, it can also update:
+
+- `failure_type`
+- `failure_symptom`
+- `likely_cause`
+- `recommended_next_action`
+
+## What It Intentionally Does Not Do
+
+- does not modify `tools/codex/run_prompt.py`
+- does not modify `tools/codex/baby_run_prompt.py`
+- does not create queue state, sidecars, databases, or extra workflow files
+- does not release the next prompt automatically
+- does not redesign the V1 record format
+- does not rewrite or reorder non-review sections
+
+## Validation
+
+Validation commands performed:
+
+- `python3 -m py_compile tools/codex/review_run.py`
+- `python3 tools/codex/review_run.py notes/001_smoke_test_pipeline__20260415_234918.md --review-status ACCEPTED --review-summary 'Smoke test output and artifact are complete enough to accept this bounded step.'`
+- `git diff -- tools/codex/run_prompt.py tools/codex/baby_run_prompt.py`
+
+Validation outcome:
+
+- `review_status` in [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md) changed from `UNREVIEWED` to `ACCEPTED`.
+- `review_summary` was written in place.
+- `reviewed_at_utc` was auto-filled as `20260415_235508`.
+- All other sections remained unchanged; the diff was limited to the three review lines above.
+- The updated record still matches the expected V1 section order and required field layout.
+- `tools/codex/run_prompt.py` remained unchanged.
+- `tools/codex/baby_run_prompt.py` remained unchanged.
+
+## Scope Note
+
+This helper is intentionally a minimum V1 review write-back tool. It updates review facts on one existing execution record and stops there.
diff --git a/tools/codex/review_run.py b/tools/codex/review_run.py
new file mode 100644
index 0000000000000000000000000000000000000000..11818b1bd53c773d264942cd4fb6f0609068581f
--- /dev/null
+++ b/tools/codex/review_run.py
@@ -0,0 +1,201 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import re
+import sys
+from pathlib import Path
+
+
+REVIEW_STATUSES = {"ACCEPTED", "REJECTED"}
+FAILURE_FIELDS = (
+    "failure_type",
+    "failure_symptom",
+    "likely_cause",
+    "recommended_next_action",
+)
+REQUIRED_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+    "execution_status",
+    "finished_at_utc",
+    "runner",
+    "return_code",
+    "retry_of_run_id",
+    "review_status",
+    "review_summary",
+    "reviewed_by",
+    "reviewed_at_utc",
+    *FAILURE_FIELDS,
+    "elapsed_seconds",
+    "final_output_char_count",
+    "stderr_char_count",
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
+
+
+def utc_timestamp() -> str:
+    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Write manual V1 review fields back into an existing execution record."
+    )
+    parser.add_argument("record", help="Path to an existing execution-record markdown file")
+    parser.add_argument(
+        "--review-status",
+        required=True,
+        choices=sorted(REVIEW_STATUSES),
+        help="Manual review outcome",
+    )
+    parser.add_argument(
+        "--review-summary",
+        required=True,
+        help="Short manual review summary",
+    )
+    parser.add_argument("--reviewed-by", help="Reviewer identifier")
+    parser.add_argument(
+        "--reviewed-at-utc",
+        help="Review timestamp in UTC using YYYYMMDD_HHMMSS; defaults to current UTC time",
+    )
+    parser.add_argument("--failure-type", help="Short failure category for rejected runs")
+    parser.add_argument("--failure-symptom", help="Observed failure symptom for rejected runs")
+    parser.add_argument("--likely-cause", help="Likely root cause for rejected runs")
+    parser.add_argument(
+        "--recommended-next-action",
+        help="Manual next action recommendation for rejected runs",
+    )
+    return parser.parse_args()
+
+
+def fail(message: str) -> int:
+    print(f"ERROR: {message}", file=sys.stderr)
+    return 1
+
+
+def require_single_line(name: str, value: str | None) -> str | None:
+    if value is None:
+        return None
+    if "\n" in value or "\r" in value:
+        raise ValueError(f"{name} must be a single line")
+    return value.strip()
+
+
+def resolve_record_path(record_arg: str) -> Path:
+    record_path = Path(record_arg)
+    if record_path.is_absolute():
+        return record_path
+    return repo_root() / record_path
+
+
+def validate_record_path(record_path: Path) -> None:
+    if not record_path.exists():
+        raise ValueError(f"record file not found: {record_path}")
+    if not record_path.is_file():
+        raise ValueError(f"record path is not a file: {record_path}")
+
+    root = repo_root().resolve()
+    notes_dir = root / "notes"
+    try:
+        record_path.resolve().relative_to(notes_dir.resolve())
+    except ValueError as exc:
+        raise ValueError(f"record must be under {notes_dir}") from exc
+
+
+def validate_v1_record_structure(text: str) -> None:
+    if not text.startswith("# "):
+        raise ValueError("record does not start with a markdown title")
+
+    positions: list[int] = []
+    for section in REQUIRED_SECTIONS:
+        pos = text.find(section)
+        if pos == -1:
+            raise ValueError(f"record is missing section: {section}")
+        positions.append(pos)
+    if positions != sorted(positions):
+        raise ValueError("record sections are out of the expected V1 order")
+
+    for field in REQUIRED_FIELDS:
+        if re.search(rf"^- {re.escape(field)}:", text, flags=re.MULTILINE) is None:
+            raise ValueError(f"record is missing field line: {field}")
+
+
+def replace_field(text: str, field: str, value: str | None, *, code: bool = False) -> str:
+    rendered = f"`{value}`" if code and value else (value or "")
+    pattern = re.compile(rf"(?m)^- {re.escape(field)}:.*$")
+    if pattern.search(text) is None:
+        raise ValueError(f"record is missing field line: {field}")
+    return pattern.sub(f"- {field}: {rendered}", text, count=1)
+
+
+def build_updates(args: argparse.Namespace) -> dict[str, tuple[str | None, bool]]:
+    review_summary = require_single_line("review_summary", args.review_summary)
+    reviewed_by = require_single_line("reviewed_by", args.reviewed_by)
+    reviewed_at_utc = require_single_line("reviewed_at_utc", args.reviewed_at_utc) or utc_timestamp()
+    failure_values = {
+        "failure_type": require_single_line("failure_type", args.failure_type),
+        "failure_symptom": require_single_line("failure_symptom", args.failure_symptom),
+        "likely_cause": require_single_line("likely_cause", args.likely_cause),
+        "recommended_next_action": require_single_line(
+            "recommended_next_action", args.recommended_next_action
+        ),
+    }
+
+    if args.review_status != "REJECTED" and any(value for value in failure_values.values()):
+        raise ValueError("rejection-only failure fields may only be set when --review-status REJECTED")
+
+    updates: dict[str, tuple[str | None, bool]] = {
+        "review_status": (args.review_status, True),
+        "review_summary": (review_summary, False),
+        "reviewed_at_utc": (reviewed_at_utc, True),
+    }
+    if reviewed_by is not None:
+        updates["reviewed_by"] = (reviewed_by, False)
+
+    if args.review_status == "REJECTED":
+        for field, value in failure_values.items():
+            if value is not None:
+                updates[field] = (value, False)
+
+    return updates
+
+
+def main() -> int:
+    try:
+        args = parse_args()
+        record_path = resolve_record_path(args.record)
+        validate_record_path(record_path)
+        text = record_path.read_text(encoding="utf-8")
+        validate_v1_record_structure(text)
+
+        updated = text
+        for field, (value, code) in build_updates(args).items():
+            updated = replace_field(updated, field, value, code=code)
+
+        record_path.write_text(updated, encoding="utf-8")
+    except ValueError as exc:
+        return fail(str(exc))
+
+    print(record_path)
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

codex
Implemented [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py). It is a thin in-place V1 write-back helper: it validates the target record under `notes/`, checks the minimum V1 section/field shape, updates `review_status` and `review_summary`, auto-fills `reviewed_at_utc` when omitted, optionally writes `reviewed_by`, and allows the rejection-only failure fields only for `REJECTED`.

I validated it by running it against [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md), which is now updated to `ACCEPTED` with a review summary and `reviewed_at_utc: 20260415_235508`. The diff on that record is limited to the review lines, so the rest of the V1 structure stayed intact.

The implementation note is at [011_review_writeback_helper_build__20260415_235514.md](/data/git/handson-ml3/notes/011_review_writeback_helper_build__20260415_235514.md). Validation also confirmed `tools/codex/run_prompt.py` and `tools/codex/baby_run_prompt.py` remain unchanged.
diff --git a/notes/011_review_writeback_helper_build__20260415_235514.md b/notes/011_review_writeback_helper_build__20260415_235514.md
new file mode 100644
index 0000000000000000000000000000000000000000..1b82c53528febd6e9dd2fe9f2b500bab7e75b4e4
--- /dev/null
+++ b/notes/011_review_writeback_helper_build__20260415_235514.md
@@ -0,0 +1,64 @@
+# 011 Review Write-Back Helper Build
+
+## What Was Built
+
+Built [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py), a thin V1 companion helper that updates one existing execution record in `notes/` in place.
+
+The helper:
+
+- accepts a record path plus explicit review arguments
+- verifies the file is under `notes/` and matches the minimum V1 execution-record structure
+- updates only the targeted field lines
+- writes the changes back to the same markdown file
+- prints the updated file path
+- exits nonzero on invalid input or malformed structure
+
+## V1 Fields It Updates
+
+Always updates:
+
+- `review_status`
+- `review_summary`
+- `reviewed_at_utc`
+
+Optionally updates:
+
+- `reviewed_by`
+
+For `REJECTED` runs, it can also update:
+
+- `failure_type`
+- `failure_symptom`
+- `likely_cause`
+- `recommended_next_action`
+
+## What It Intentionally Does Not Do
+
+- does not modify `tools/codex/run_prompt.py`
+- does not modify `tools/codex/baby_run_prompt.py`
+- does not create queue state, sidecars, databases, or extra workflow files
+- does not release the next prompt automatically
+- does not redesign the V1 record format
+- does not rewrite or reorder non-review sections
+
+## Validation
+
+Validation commands performed:
+
+- `python3 -m py_compile tools/codex/review_run.py`
+- `python3 tools/codex/review_run.py notes/001_smoke_test_pipeline__20260415_234918.md --review-status ACCEPTED --review-summary 'Smoke test output and artifact are complete enough to accept this bounded step.'`
+- `git diff -- tools/codex/run_prompt.py tools/codex/baby_run_prompt.py`
+
+Validation outcome:
+
+- `review_status` in [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md) changed from `UNREVIEWED` to `ACCEPTED`.
+- `review_summary` was written in place.
+- `reviewed_at_utc` was auto-filled as `20260415_235508`.
+- All other sections remained unchanged; the diff was limited to the three review lines above.
+- The updated record still matches the expected V1 section order and required field layout.
+- `tools/codex/run_prompt.py` remained unchanged.
+- `tools/codex/baby_run_prompt.py` remained unchanged.
+
+## Scope Note
+
+This helper is intentionally a minimum V1 review write-back tool. It updates review facts on one existing execution record and stops there.
diff --git a/tools/codex/review_run.py b/tools/codex/review_run.py
new file mode 100644
index 0000000000000000000000000000000000000000..11818b1bd53c773d264942cd4fb6f0609068581f
--- /dev/null
+++ b/tools/codex/review_run.py
@@ -0,0 +1,201 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import re
+import sys
+from pathlib import Path
+
+
+REVIEW_STATUSES = {"ACCEPTED", "REJECTED"}
+FAILURE_FIELDS = (
+    "failure_type",
+    "failure_symptom",
+    "likely_cause",
+    "recommended_next_action",
+)
+REQUIRED_FIELDS = (
+    "run_id",
+    "prompt_file",
+    "prompt_stem",
+    "started_at_utc",
+    "execution_status",
+    "finished_at_utc",
+    "runner",
+    "return_code",
+    "retry_of_run_id",
+    "review_status",
+    "review_summary",
+    "reviewed_by",
+    "reviewed_at_utc",
+    *FAILURE_FIELDS,
+    "elapsed_seconds",
+    "final_output_char_count",
+    "stderr_char_count",
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
+
+
+def utc_timestamp() -> str:
+    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Write manual V1 review fields back into an existing execution record."
+    )
+    parser.add_argument("record", help="Path to an existing execution-record markdown file")
+    parser.add_argument(
+        "--review-status",
+        required=True,
+        choices=sorted(REVIEW_STATUSES),
+        help="Manual review outcome",
+    )
+    parser.add_argument(
+        "--review-summary",
+        required=True,
+        help="Short manual review summary",
+    )
+    parser.add_argument("--reviewed-by", help="Reviewer identifier")
+    parser.add_argument(
+        "--reviewed-at-utc",
+        help="Review timestamp in UTC using YYYYMMDD_HHMMSS; defaults to current UTC time",
+    )
+    parser.add_argument("--failure-type", help="Short failure category for rejected runs")
+    parser.add_argument("--failure-symptom", help="Observed failure symptom for rejected runs")
+    parser.add_argument("--likely-cause", help="Likely root cause for rejected runs")
+    parser.add_argument(
+        "--recommended-next-action",
+        help="Manual next action recommendation for rejected runs",
+    )
+    return parser.parse_args()
+
+
+def fail(message: str) -> int:
+    print(f"ERROR: {message}", file=sys.stderr)
+    return 1
+
+
+def require_single_line(name: str, value: str | None) -> str | None:
+    if value is None:
+        return None
+    if "\n" in value or "\r" in value:
+        raise ValueError(f"{name} must be a single line")
+    return value.strip()
+
+
+def resolve_record_path(record_arg: str) -> Path:
+    record_path = Path(record_arg)
+    if record_path.is_absolute():
+        return record_path
+    return repo_root() / record_path
+
+
+def validate_record_path(record_path: Path) -> None:
+    if not record_path.exists():
+        raise ValueError(f"record file not found: {record_path}")
+    if not record_path.is_file():
+        raise ValueError(f"record path is not a file: {record_path}")
+
+    root = repo_root().resolve()
+    notes_dir = root / "notes"
+    try:
+        record_path.resolve().relative_to(notes_dir.resolve())
+    except ValueError as exc:
+        raise ValueError(f"record must be under {notes_dir}") from exc
+
+
+def validate_v1_record_structure(text: str) -> None:
+    if not text.startswith("# "):
+        raise ValueError("record does not start with a markdown title")
+
+    positions: list[int] = []
+    for section in REQUIRED_SECTIONS:
+        pos = text.find(section)
+        if pos == -1:
+            raise ValueError(f"record is missing section: {section}")
+        positions.append(pos)
+    if positions != sorted(positions):
+        raise ValueError("record sections are out of the expected V1 order")
+
+    for field in REQUIRED_FIELDS:
+        if re.search(rf"^- {re.escape(field)}:", text, flags=re.MULTILINE) is None:
+            raise ValueError(f"record is missing field line: {field}")
+
+
+def replace_field(text: str, field: str, value: str | None, *, code: bool = False) -> str:
+    rendered = f"`{value}`" if code and value else (value or "")
+    pattern = re.compile(rf"(?m)^- {re.escape(field)}:.*$")
+    if pattern.search(text) is None:
+        raise ValueError(f"record is missing field line: {field}")
+    return pattern.sub(f"- {field}: {rendered}", text, count=1)
+
+
+def build_updates(args: argparse.Namespace) -> dict[str, tuple[str | None, bool]]:
+    review_summary = require_single_line("review_summary", args.review_summary)
+    reviewed_by = require_single_line("reviewed_by", args.reviewed_by)
+    reviewed_at_utc = require_single_line("reviewed_at_utc", args.reviewed_at_utc) or utc_timestamp()
+    failure_values = {
+        "failure_type": require_single_line("failure_type", args.failure_type),
+        "failure_symptom": require_single_line("failure_symptom", args.failure_symptom),
+        "likely_cause": require_single_line("likely_cause", args.likely_cause),
+        "recommended_next_action": require_single_line(
+            "recommended_next_action", args.recommended_next_action
+        ),
+    }
+
+    if args.review_status != "REJECTED" and any(value for value in failure_values.values()):
+        raise ValueError("rejection-only failure fields may only be set when --review-status REJECTED")
+
+    updates: dict[str, tuple[str | None, bool]] = {
+        "review_status": (args.review_status, True),
+        "review_summary": (review_summary, False),
+        "reviewed_at_utc": (reviewed_at_utc, True),
+    }
+    if reviewed_by is not None:
+        updates["reviewed_by"] = (reviewed_by, False)
+
+    if args.review_status == "REJECTED":
+        for field, value in failure_values.items():
+            if value is not None:
+                updates[field] = (value, False)
+
+    return updates
+
+
+def main() -> int:
+    try:
+        args = parse_args()
+        record_path = resolve_record_path(args.record)
+        validate_record_path(record_path)
+        text = record_path.read_text(encoding="utf-8")
+        validate_v1_record_structure(text)
+
+        updated = text
+        for field, (value, code) in build_updates(args).items():
+            updated = replace_field(updated, field, value, code=code)
+
+        record_path.write_text(updated, encoding="utf-8")
+    except ValueError as exc:
+        return fail(str(exc))
+
+    print(record_path)
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

tokens used
45,137
```
