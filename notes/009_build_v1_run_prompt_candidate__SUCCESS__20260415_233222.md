# 009_build_v1_run_prompt_candidate - SUCCESS

- Prompt file: `009_build_v1_run_prompt_candidate.md`
- Timestamp (UTC): `20260415_233222`
- Status: `SUCCESS`

## Original Prompt

```md
# Task: Build the first proper V1 runner candidate beside baby_run_prompt.py

You are working in this repository.

Your task is to implement a new runner candidate at:

`tools/codex/run_prompt.py`

This new runner should be the first proper V1 implementation candidate for prompt execution records.

## Important framing

Do NOT modify:

- `tools/codex/baby_run_prompt.py`

That file is a working bootstrap artifact and should remain untouched in this pass.

You may read it for reference or inspiration, but you must build the new runner as a separate implementation.

## Files to inspect

Read these exact files before making changes:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/baby_run_prompt.py`

## Goal

Create a new runner at:

`tools/codex/run_prompt.py`

The new runner should implement the V1 execution-record behavior described in the design documents while preserving a thin, inspectable scope.

This is a replacement candidate, not a broad framework.

## What the new runner must do

The new runner should:

1. resolve one prompt file from the provided argument
2. run `codex exec` in the repository root
3. capture the final Codex message via `--output-last-message`
4. capture stderr and subprocess return code
5. write one V1 execution-record markdown file into `notes/`
6. use stable run identity naming based on:
   - `<prompt_stem>__<started_at_utc>`
7. initialize:
   - `review_status: UNREVIEWED`
   - blank review fields
   - blank failure-analysis fields
8. separate:
   - execution outcome
   - review outcome
9. capture the minimum automatic resource/cost metrics defined by the spec
10. print the written execution-record path
11. exit with the subprocess return code

## What the new runner must NOT do

Do NOT:

- modify `baby_run_prompt.py`
- implement review write-back
- implement queue progression logic
- implement dependency handling
- implement retry orchestration
- redesign the broader CLI
- split the system into multiple modules unless one tiny helper is absolutely necessary
- introduce JSON sidecars, databases, or other new persistence layers

## Collision rule

The spec left one open question about same-second collisions for `run_id`.

Settle it in the smallest practical way for V1:

- default to second-precision UTC timestamps
- if the target run-record path already exists, append a short numeric suffix such as `__2`, `__3`, etc.
- document this briefly in the implementation note you create below

Do not invent a larger identity system.

## Implementation guidance

Preserve the spirit of the current bootstrap runner where it is still useful:

- simple CLI with one required prompt argument
- repo-root execution
- prompt resolution from path / repo-relative path / `codex_prompts/` / unique prefix match
- direct subprocess use for `codex exec`

But the output artifact must follow the V1 execution-record design rather than the old success/failure note model.

## Required artifacts

### Artifact 1
Create or update:

`tools/codex/run_prompt.py`

### Artifact 2
Create a short implementation note at:

`notes/009_run_prompt_candidate_build__TIMESTAMP.md`

This note should summarize:

- what was built
- what behavior was intentionally preserved from `baby_run_prompt.py`
- what changed to support the V1 execution record
- how the collision rule was handled
- what was intentionally deferred

### Artifact 3
Create one sample execution-record file by running the new runner against this prompt:

`codex_prompts/001_smoke_test_pipeline.md`

Use the new runner itself for this validation run.

The goal is to prove that the new runner can emit a correctly shaped V1 execution record in `notes/`.

## Validation requirements

After implementation, validate at least these points:

1. the new runner can resolve and execute the sample prompt
2. the written record filename follows the V1 naming rule
3. the record includes:
   - run identity
   - execution facts
   - review facts
   - failure-analysis section
   - resource/cost section
   - prompt text
   - Codex final output
   - stderr section
4. `review_status` starts as `UNREVIEWED`
5. `execution_status` is derived only from subprocess result
6. `baby_run_prompt.py` remains unchanged

Include the validation outcome in the implementation note.

## Constraints

1. Use the exact file paths listed above.
2. Keep the new runner thin and inspectable.
3. Prefer one file unless a tiny helper is truly necessary.
4. Do not perform a broader refactor in this pass.
5. Do not alter the design documents in this pass.
6. Do not change the bootstrap runner in this pass.

## Success criteria

This task is successful if:

- `tools/codex/run_prompt.py` exists and runs
- it emits a V1 execution record rather than the old success/failure note format
- execution and review are clearly separated in the written record
- the validation run produces one correctly shaped record in `notes/`
- `baby_run_prompt.py` is unchanged
- the implementation remains small enough to review comfortably
```

## Codex Output

Implemented the V1 runner candidate in [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py). It keeps the bootstrap runner’s prompt resolution and thin `codex exec` subprocess flow, but now writes a single V1 execution-record markdown artifact with separate execution and review sections, blank review/failure-analysis fields, lightweight resource metrics, stable run-id naming, and numeric suffix collision handling.

I also wrote the implementation note at [009_run_prompt_candidate_build__20260415_233407.md](/data/git/handson-ml3/notes/009_run_prompt_candidate_build__20260415_233407.md) and validated the runner by invoking it against the smoke-test prompt. That produced [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md), which has the required V1 shape. The `codex exec` subprocess itself exited `1` because session initialization hit `Read-only file system (os error 30)`, so the sample record correctly shows `execution_status: EXECUTION_FAILED`, `review_status: UNREVIEWED`, captured stderr, and the printed record path. `tools/codex/baby_run_prompt.py` was not modified.

## Notes

Codex stderr:
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
session id: 019d937d-21af-70b2-9d20-2cd44cc2b8fa
--------
user
# Task: Build the first proper V1 runner candidate beside baby_run_prompt.py

You are working in this repository.

Your task is to implement a new runner candidate at:

`tools/codex/run_prompt.py`

This new runner should be the first proper V1 implementation candidate for prompt execution records.

## Important framing

Do NOT modify:

- `tools/codex/baby_run_prompt.py`

That file is a working bootstrap artifact and should remain untouched in this pass.

You may read it for reference or inspiration, but you must build the new runner as a separate implementation.

## Files to inspect

Read these exact files before making changes:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/baby_run_prompt.py`

## Goal

Create a new runner at:

`tools/codex/run_prompt.py`

The new runner should implement the V1 execution-record behavior described in the design documents while preserving a thin, inspectable scope.

This is a replacement candidate, not a broad framework.

## What the new runner must do

The new runner should:

1. resolve one prompt file from the provided argument
2. run `codex exec` in the repository root
3. capture the final Codex message via `--output-last-message`
4. capture stderr and subprocess return code
5. write one V1 execution-record markdown file into `notes/`
6. use stable run identity naming based on:
   - `<prompt_stem>__<started_at_utc>`
7. initialize:
   - `review_status: UNREVIEWED`
   - blank review fields
   - blank failure-analysis fields
8. separate:
   - execution outcome
   - review outcome
9. capture the minimum automatic resource/cost metrics defined by the spec
10. print the written execution-record path
11. exit with the subprocess return code

## What the new runner must NOT do

Do NOT:

- modify `baby_run_prompt.py`
- implement review write-back
- implement queue progression logic
- implement dependency handling
- implement retry orchestration
- redesign the broader CLI
- split the system into multiple modules unless one tiny helper is absolutely necessary
- introduce JSON sidecars, databases, or other new persistence layers

## Collision rule

The spec left one open question about same-second collisions for `run_id`.

Settle it in the smallest practical way for V1:

- default to second-precision UTC timestamps
- if the target run-record path already exists, append a short numeric suffix such as `__2`, `__3`, etc.
- document this briefly in the implementation note you create below

Do not invent a larger identity system.

## Implementation guidance

Preserve the spirit of the current bootstrap runner where it is still useful:

- simple CLI with one required prompt argument
- repo-root execution
- prompt resolution from path / repo-relative path / `codex_prompts/` / unique prefix match
- direct subprocess use for `codex exec`

But the output artifact must follow the V1 execution-record design rather than the old success/failure note model.

## Required artifacts

### Artifact 1
Create or update:

`tools/codex/run_prompt.py`

### Artifact 2
Create a short implementation note at:

`notes/009_run_prompt_candidate_build__TIMESTAMP.md`

This note should summarize:

- what was built
- what behavior was intentionally preserved from `baby_run_prompt.py`
- what changed to support the V1 execution record
- how the collision rule was handled
- what was intentionally deferred

### Artifact 3
Create one sample execution-record file by running the new runner against this prompt:

`codex_prompts/001_smoke_test_pipeline.md`

Use the new runner itself for this validation run.

The goal is to prove that the new runner can emit a correctly shaped V1 execution record in `notes/`.

## Validation requirements

After implementation, validate at least these points:

1. the new runner can resolve and execute the sample prompt
2. the written record filename follows the V1 naming rule
3. the record includes:
   - run identity
   - execution facts
   - review facts
   - failure-analysis section
   - resource/cost section
   - prompt text
   - Codex final output
   - stderr section
4. `review_status` starts as `UNREVIEWED`
5. `execution_status` is derived only from subprocess result
6. `baby_run_prompt.py` remains unchanged

Include the validation outcome in the implementation note.

## Constraints

1. Use the exact file paths listed above.
2. Keep the new runner thin and inspectable.
3. Prefer one file unless a tiny helper is truly necessary.
4. Do not perform a broader refactor in this pass.
5. Do not alter the design documents in this pass.
6. Do not change the bootstrap runner in this pass.

## Success criteria

This task is successful if:

- `tools/codex/run_prompt.py` exists and runs
- it emits a V1 execution record rather than the old success/failure note format
- execution and review are clearly separated in the written record
- the validation run produces one correctly shaped record in `notes/`
- `baby_run_prompt.py` is unchanged
- the implementation remains small enough to review comfortably

codex
I’m reading the design docs and the bootstrap runner first, then I’ll implement `tools/codex/run_prompt.py` as a thin V1 candidate and validate it by generating a sample execution record.
exec
/bin/bash -lc "sed -n '1,220p' tools/Project_Design_Workflow.md" in /data/git/handson-ml3
exec
/bin/bash -lc "sed -n '1,260p' tools/codex/V1_Execution_Record_Artifact.md" in /data/git/handson-ml3
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
/bin/bash -lc "sed -n '1,240p' tools/codex/V1_Run_Review_Gate.md" in /data/git/handson-ml3
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
/bin/bash -lc "sed -n '1,260p' tools/codex/V1_Bridge_Runner_Change_Spec.md" in /data/git/handson-ml3
 succeeded in 0ms:
# V1 Bridge Runner Change Spec

## Purpose

Define the smallest change set for [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py) so the current bridge runner writes a V1 execution record in `notes/` without a broader refactor.

The target outcome is:

- execution produces one durable V1 execution record
- execution outcome and review outcome are kept separate
- run identity is stable
- the runner captures only the minimum automatic fields
- human review fields remain manual

## Scope

This spec covers only the current bridge runner at [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py).

It defines:

- the record filename pattern
- the markdown section order
- the exact fields the runner should populate automatically
- the manual review fields the runner should initialize for later completion
- the minimal mapping from subprocess result to execution status
- the minimal metrics to capture now

## Current Behavior Summary

The current runner:

- resolves one prompt path from an argument using the existing lookup rules
- validates that `codex_prompts/` and `notes/` exist
- reads the prompt text from the selected file
- runs `codex exec -C <repo_root> --output-last-message <tempfile> -`
- captures subprocess return code, last-message output, and stderr
- writes one markdown note into `notes/`
- encodes `SUCCESS` or `FAILED` into the filename and note body based only on return code
- prints the written note path
- exits with the subprocess return code

The current gap is that the note conflates execution completion with outcome labeling and does not initialize the V1 review gate.

## Required V1 Changes

### 1. Replace success/failure note naming with stable run identity naming

The runner should stop writing filenames in the form:

`<prompt_stem>__<status>__<timestamp>.md`

It should instead write:

`notes/<prompt_stem>__<started_at_utc>.md`

This filename is the run's stable identity carrier and must not encode:

- `SUCCESS`
- `FAILED`
- `ACCEPTED`
- `REJECTED`

### 2. Emit a V1 execution record instead of a success-implies-acceptance note

The runner should replace the current freeform note body with a V1 execution record shaped around:

1. Header / identity
2. Execution facts
3. Review facts
4. Failure analysis
5. Resource / cost facts
6. Prompt text
7. Codex final output
8. Stderr or supplemental notes

### 3. Initialize review fields but leave review work manual

The runner should create a new record with:

- `review_status: UNREVIEWED`
- `review_summary:` blank placeholder

It should not attempt to:

- infer acceptance
- mark `ACCEPTED`
- mark `REJECTED`
- fill reviewer identity
- fill review timestamp
- classify the failure automatically

### 4. Derive execution status only from subprocess execution result

The minimal V1 mapping is:

- subprocess `return_code == 0` -> `execution_status: EXECUTED`
- subprocess `return_code != 0` -> `execution_status: EXECUTION_FAILED`

This mapping controls only execution facts.
It must not affect `review_status` beyond leaving it at `UNREVIEWED`.

### 5. Capture only the minimum automatic metrics now

The runner should capture and write:

- `started_at_utc`
- `finished_at_utc`
- `elapsed_seconds`
- `return_code`
- `final_output_char_count`
- `stderr_char_count`

It should also write:

- `stderr_text` when present

No broader token accounting, file-diff analysis, validation harvesting, or queue-state calculation should be added in this pass.

## Non-Goals

This spec does not authorize:

- splitting the runner into multiple modules
- a larger CLI redesign
- queue progression logic beyond initializing `review_status: UNREVIEWED`
- automatic review write-back
- retry orchestration
- dependency-aware scheduling
- broader note/history discovery logic
- structured sidecars, JSON records, or databases

## Exact Record Shape

The record should remain plain markdown and use these sections in this order.

### 1. Header / identity

Suggested format:

```md
# <run_id>

- run_id: `<run_id>`
- prompt_file: `<prompt_file>`
- prompt_stem: `<prompt_stem>`
- started_at_utc: `<started_at_utc>`
```

### 2. Execution facts

```md
## Execution Facts

- execution_status: `<EXECUTED|EXECUTION_FAILED>`
- finished_at_utc: `<finished_at_utc>`
- runner: `tools/codex/baby_run_prompt.py`
- return_code: `<int>`
- retry_of_run_id: ``
```

### 3. Review facts

```md
## Review Facts

- review_status: `UNREVIEWED`
- review_summary:
- reviewed_by:
- reviewed_at_utc:
```

### 4. Failure analysis

```md
## Failure Analysis

- failure_type:
- failure_symptom:
- likely_cause:
- recommended_next_action:
```

These fields are initialized only.
They remain manual in V1.

### 5. Resource / cost facts

```md
## Resource / Cost Facts

- elapsed_seconds: `<float_or_decimal_seconds>`
- final_output_char_count: `<int>`
- stderr_char_count: `<int>`
```

### 6. Prompt text

````md
## Prompt Text

```md
<full prompt text>
```
````

### 7. Codex final output

```md
## Codex Final Output

<captured last message, or `*No output captured.*`>
```

### 8. Stderr or supplemental notes

If stderr exists:

````md
## Stderr

```text
<stderr text>
```
````

If stderr is empty:

```md
## Stderr

*No stderr captured.*
```

## Exact Data / Field Mapping

### Run identity mapping

- `run_id` -> `f"{prompt_path.stem}__{started_at_utc}"`
- filename -> `notes/{run_id}.md`
- `prompt_stem` -> `prompt_path.stem`
- `started_at_utc` -> timestamp captured immediately before `run_codex(...)`

### Prompt file mapping

- `prompt_file` -> repo-relative POSIX path when the resolved prompt is inside the repo root
- fallback `prompt_file` -> absolute path string only if the resolved prompt is outside the repo root

This is the smallest precise mapping that preserves current prompt resolution behavior without introducing a new prompt identity system.

### Execution mapping

- `runner` -> literal `tools/codex/baby_run_prompt.py`
- `return_code` -> subprocess return code from `subprocess.run(...)`
- `execution_status` -> `EXECUTED` when return code is `0`, else `EXECUTION_FAILED`
- `finished_at_utc` -> timestamp captured immediately after `run_codex(...)` returns
- `prompt_text` section -> full prompt file contents as currently read
- `codex_final_output` section -> current `--output-last-message` file contents

### Review mapping


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

codex
The docs are aligned around a single-record markdown artifact with execution and review clearly separated. I’m checking the rest of the runner spec, the sample prompt, and the current worktree before I edit anything.
exec
/bin/bash -lc "sed -n '261,520p' tools/codex/V1_Bridge_Runner_Change_Spec.md" in /data/git/handson-ml3
 succeeded in 0ms:
- `review_status` -> literal `UNREVIEWED`
- `review_summary` -> blank
- `reviewed_by` -> blank
- `reviewed_at_utc` -> blank

### Failure / retry mapping

- `retry_of_run_id` -> blank in this pass
- `failure_type` -> blank
- `failure_symptom` -> blank
- `likely_cause` -> blank
- `recommended_next_action` -> blank

This preserves the manual review boundary from [V1_Run_Review_Gate.md](/data/git/handson-ml3/tools/codex/V1_Run_Review_Gate.md).

### Metrics mapping

- `elapsed_seconds` -> `finished_at_utc - started_at_utc` in seconds
- `final_output_char_count` -> `len(codex_output)`
- `stderr_char_count` -> `len(stderr_text)`
- `stderr_text` section -> full stderr text when non-empty

These are the only required automatic metrics for V1 because they are cheap, inspectable, and already available from the current subprocess flow.

## Review Fields Left Manual

The runner should initialize but not populate these beyond blanks:

- `review_summary`
- `reviewed_by`
- `reviewed_at_utc`
- `failure_type`
- `failure_symptom`
- `likely_cause`
- `recommended_next_action`

The runner should set only one review field automatically:

- `review_status: UNREVIEWED`

## Behaviors That Should Remain Unchanged For V1

The following current behaviors should stay as they are:

- repository-root discovery via `Path(__file__).resolve().parents[2]`
- requirement that `codex_prompts/` and `notes/` already exist
- prompt resolution behavior for absolute paths, repo-relative paths, `codex_prompts/` paths, and unique prefix matches
- reading the entire prompt file as UTF-8 text
- invoking `codex exec` through the existing thin subprocess path
- using `--output-last-message` and preserving stderr capture
- writing exactly one markdown artifact per run into `notes/`
- printing the written record path to stdout
- returning the subprocess return code as the script exit code

These preserved behaviors keep the implementation bridge-sized and avoid a premature runner redesign.

## Open Questions To Resolve Before Implementation

### 1. Timestamp format precision

The current runner uses second precision: `YYYYMMDD_HHMMSS`.
This matches the documented V1 examples and is likely sufficient, but repeated same-second runs of the same prompt would collide.

Decision needed:

- keep second precision for strict alignment with the current V1 record doc, or
- add a small collision-avoidance suffix if a file already exists

The narrower V1 choice is to keep the current format and fail fast on collision unless this becomes a real issue.

### 2. Blank-value formatting convention

This spec uses empty field values for manual fields.
Implementation should keep that representation consistent across:

- empty list-style fields
- empty sections
- review write-back later

Decision needed:

- use blank bullet values exactly as shown in this spec

No other open questions need to block implementation.

exec
/bin/bash -lc "sed -n '1,220p' codex_prompts/001_smoke_test_pipeline.md" in /data/git/handson-ml3
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
/bin/bash -lc 'git status --short' in /data/git/handson-ml3
 succeeded in 0ms:
 M tools/Codex_Prompt_Workflow_Architecture.md
?? codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md
?? codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md
?? codex_prompts/006_define_v1_run_review_gate.md
?? codex_prompts/007_align_architecture_doc_to_v1_workflow.md
?? codex_prompts/008_define_minimal_bridge_runner_change_spec.md
?? codex_prompts/009_build_v1_run_prompt_candidate.md
?? notes/004_architecture_and_bridge_runner_review__20260415_195538.md
?? notes/004_next_design_step_recommendation__20260415_195538.md
?? notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md
?? notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md
?? notes/005_prompt_queue_plan__20260415_202557.md
?? notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md
?? notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md
?? notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md
?? tools/codex/V1_Bridge_Runner_Change_Spec.md
?? tools/codex/V1_Execution_Record_Artifact.md
?? tools/codex/V1_Run_Review_Gate.md

codex
I have the spec details now. Next step is to inspect whether `tools/codex/run_prompt.py` already exists and then patch in a single-file implementation that preserves the bootstrap lookup and subprocess behavior.
exec
/bin/bash -lc "test -f tools/codex/run_prompt.py && sed -n '1,260p' tools/codex/run_prompt.py || true" in /data/git/handson-ml3
 succeeded in 0ms:
exec
/bin/bash -lc 'rg --files tools/codex | sort' in /data/git/handson-ml3
 succeeded in 0ms:
tools/codex/V1_Bridge_Runner_Change_Spec.md
tools/codex/V1_Execution_Record_Artifact.md
tools/codex/V1_Run_Review_Gate.md
tools/codex/baby_run_prompt.py

Implementing the V1 runner in one file, then validating it by producing a sample execution record and an implementation note.
  → Add tools/codex/run_prompt.py with V1 execution-record behavior and collision-safe run-id naming
  • Run the new runner against codex_prompts/001_smoke_test_pipeline.md to generate a sample execution record
  • Write notes/009_run_prompt_candidate_build__TIMESTAMP.md with implementation and validation results
codex
I’m writing the new runner now. It will keep the bootstrap resolution and subprocess flow, but switch the artifact to the V1 record shape and add the small same-second collision suffix rule.
apply patch
patch: completed
/data/git/handson-ml3/tools/codex/run_prompt.py
diff --git a/tools/codex/run_prompt.py b/tools/codex/run_prompt.py
new file mode 100644
index 0000000000000000000000000000000000000000..3b9ea148301aefed31de2705205b247bbe223699
--- /dev/null
+++ b/tools/codex/run_prompt.py
@@ -0,0 +1,223 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import subprocess
+import sys
+import tempfile
+import time
+from pathlib import Path
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+RUNNER_PATH = "tools/codex/run_prompt.py"
+
+
+def utc_now() -> dt.datetime:
+    return dt.datetime.now(dt.timezone.utc)
+
+
+def utc_timestamp(moment: dt.datetime) -> str:
+    return moment.strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
+    prompt = Path(prompt_arg)
+
+    if prompt.is_absolute():
+        return prompt
+
+    direct = root / prompt
+    if direct.exists():
+        return direct
+
+    in_prompts = root / PROMPTS_DIR / prompt
+    if in_prompts.exists():
+        return in_prompts
+
+    matches = sorted((root / PROMPTS_DIR).glob(f"{prompt_arg}*"))
+    if len(matches) == 1:
+        return matches[0]
+
+    return in_prompts
+
+
+def prompt_file_label(root: Path, prompt_path: Path) -> str:
+    try:
+        return prompt_path.resolve().relative_to(root.resolve()).as_posix()
+    except ValueError:
+        return str(prompt_path.resolve())
+
+
+def build_record_path(notes_dir: Path, prompt_stem: str, started_at_utc: str) -> tuple[str, Path]:
+    base_run_id = f"{prompt_stem}__{started_at_utc}"
+    candidate = notes_dir / f"{base_run_id}.md"
+    if not candidate.exists():
+        return base_run_id, candidate
+
+    suffix = 2
+    while True:
+        run_id = f"{base_run_id}__{suffix}"
+        candidate = notes_dir / f"{run_id}.md"
+        if not candidate.exists():
+            return run_id, candidate
+        suffix += 1
+
+
+def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
+    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
+        output_path = Path(handle.name)
+
+    try:
+        result = subprocess.run(
+            ["codex", "exec", "-C", str(root), "--output-last-message", str(output_path), "-"],
+            input=prompt_text,
+            text=True,
+            capture_output=True,
+            check=False,
+        )
+        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
+        return result.returncode, final_output, result.stderr
+    finally:
+        output_path.unlink(missing_ok=True)
+
+
+def fenced_block(text: str, fence: str, info: str) -> str:
+    body = text.rstrip("\n")
+    return f"{fence}{info}\n{body}\n{fence}" if body else f"{fence}{info}\n\n{fence}"
+
+
+def build_record_content(
+    *,
+    run_id: str,
+    prompt_file: str,
+    prompt_stem: str,
+    started_at_utc: str,
+    execution_status: str,
+    finished_at_utc: str,
+    return_code: int,
+    prompt_text: str,
+    codex_output: str,
+    stderr_text: str,
+    elapsed_seconds: float,
+) -> str:
+    final_output = codex_output.rstrip()
+    stderr_body = stderr_text.rstrip()
+
+    sections = [
+        f"# {run_id}",
+        "",
+        f"- run_id: `{run_id}`",
+        f"- prompt_file: `{prompt_file}`",
+        f"- prompt_stem: `{prompt_stem}`",
+        f"- started_at_utc: `{started_at_utc}`",
+        "",
+        "## Execution Facts",
+        "",
+        f"- execution_status: `{execution_status}`",
+        f"- finished_at_utc: `{finished_at_utc}`",
+        f"- runner: `{RUNNER_PATH}`",
+        f"- return_code: `{return_code}`",
+        "- retry_of_run_id:",
+        "",
+        "## Review Facts",
+        "",
+        "- review_status: `UNREVIEWED`",
+        "- review_summary:",
+        "- reviewed_by:",
+        "- reviewed_at_utc:",
+        "",
+        "## Failure Analysis",
+        "",
+        "- failure_type:",
+        "- failure_symptom:",
+        "- likely_cause:",
+        "- recommended_next_action:",
+        "",
+        "## Resource / Cost Facts",
+        "",
+        f"- elapsed_seconds: `{elapsed_seconds:.3f}`",
+        f"- final_output_char_count: `{len(codex_output)}`",
+        f"- stderr_char_count: `{len(stderr_text)}`",
+        "",
+        "## Prompt Text",
+        "",
+        fenced_block(prompt_text, "```", "md"),
+        "",
+        "## Codex Final Output",
+        "",
+        final_output if final_output else "*No output captured.*",
+        "",
+        "## Stderr",
+        "",
+        fenced_block(stderr_body, "```", "text") if stderr_body else "*No stderr captured.*",
+        "",
+    ]
+    return "\n".join(sections)
+
+
+def main() -> int:
+    parser = argparse.ArgumentParser(
+        description="Run one prompt file through codex exec and write a V1 execution record."
+    )
+    parser.add_argument("prompt", help="Prompt filename, numeric prefix, or path")
+    args = parser.parse_args()
+
+    root = repo_root()
+    prompts_dir = root / PROMPTS_DIR
+    notes_dir = root / NOTES_DIR
+
+    if not prompts_dir.exists():
+        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
+        return 1
+
+    if not notes_dir.exists():
+        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
+        return 1
+
+    prompt_path = resolve_prompt_path(root, args.prompt)
+    if not prompt_path.exists():
+        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
+        return 1
+
+    prompt_text = prompt_path.read_text(encoding="utf-8")
+    started_at = utc_now()
+    started_at_utc = utc_timestamp(started_at)
+    run_id, record_path = build_record_path(notes_dir, prompt_path.stem, started_at_utc)
+
+    monotonic_start = time.monotonic()
+    return_code, codex_output, stderr_text = run_codex(root, prompt_text)
+    elapsed_seconds = time.monotonic() - monotonic_start
+    finished_at_utc = utc_timestamp(utc_now())
+    execution_status = "EXECUTED" if return_code == 0 else "EXECUTION_FAILED"
+
+    record_path.write_text(
+        build_record_content(
+            run_id=run_id,
+            prompt_file=prompt_file_label(root, prompt_path),
+            prompt_stem=prompt_path.stem,
+            started_at_utc=started_at_utc,
+            execution_status=execution_status,
+            finished_at_utc=finished_at_utc,
+            return_code=return_code,
+            prompt_text=prompt_text,
+            codex_output=codex_output,
+            stderr_text=stderr_text,
+            elapsed_seconds=elapsed_seconds,
+        ),
+        encoding="utf-8",
+    )
+
+    print(record_path)
+    return return_code
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

diff --git a/tools/codex/run_prompt.py b/tools/codex/run_prompt.py
new file mode 100644
index 0000000000000000000000000000000000000000..3b9ea148301aefed31de2705205b247bbe223699
--- /dev/null
+++ b/tools/codex/run_prompt.py
@@ -0,0 +1,223 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import subprocess
+import sys
+import tempfile
+import time
+from pathlib import Path
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+RUNNER_PATH = "tools/codex/run_prompt.py"
+
+
+def utc_now() -> dt.datetime:
+    return dt.datetime.now(dt.timezone.utc)
+
+
+def utc_timestamp(moment: dt.datetime) -> str:
+    return moment.strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
+    prompt = Path(prompt_arg)
+
+    if prompt.is_absolute():
+        return prompt
+
+    direct = root / prompt
+    if direct.exists():
+        return direct
+
+    in_prompts = root / PROMPTS_DIR / prompt
+    if in_prompts.exists():
+        return in_prompts
+
+    matches = sorted((root / PROMPTS_DIR).glob(f"{prompt_arg}*"))
+    if len(matches) == 1:
+        return matches[0]
+
+    return in_prompts
+
+
+def prompt_file_label(root: Path, prompt_path: Path) -> str:
+    try:
+        return prompt_path.resolve().relative_to(root.resolve()).as_posix()
+    except ValueError:
+        return str(prompt_path.resolve())
+
+
+def build_record_path(notes_dir: Path, prompt_stem: str, started_at_utc: str) -> tuple[str, Path]:
+    base_run_id = f"{prompt_stem}__{started_at_utc}"
+    candidate = notes_dir / f"{base_run_id}.md"
+    if not candidate.exists():
+        return base_run_id, candidate
+
+    suffix = 2
+    while True:
+        run_id = f"{base_run_id}__{suffix}"
+        candidate = notes_dir / f"{run_id}.md"
+        if not candidate.exists():
+            return run_id, candidate
+        suffix += 1
+
+
+def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
+    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
+        output_path = Path(handle.name)
+
+    try:
+        result = subprocess.run(
+            ["codex", "exec", "-C", str(root), "--output-last-message", str(output_path), "-"],
+            input=prompt_text,
+            text=True,
+            capture_output=True,
+            check=False,
+        )
+        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
+        return result.returncode, final_output, result.stderr
+    finally:
+        output_path.unlink(missing_ok=True)
+
+
+def fenced_block(text: str, fence: str, info: str) -> str:
+    body = text.rstrip("\n")
+    return f"{fence}{info}\n{body}\n{fence}" if body else f"{fence}{info}\n\n{fence}"
+
+
+def build_record_content(
+    *,
+    run_id: str,
+    prompt_file: str,
+    prompt_stem: str,
+    started_at_utc: str,
+    execution_status: str,
+    finished_at_utc: str,
+    return_code: int,
+    prompt_text: str,
+    codex_output: str,
+    stderr_text: str,
+    elapsed_seconds: float,
+) -> str:
+    final_output = codex_output.rstrip()
+    stderr_body = stderr_text.rstrip()
+
+    sections = [
+        f"# {run_id}",
+        "",
+        f"- run_id: `{run_id}`",
+        f"- prompt_file: `{prompt_file}`",
+        f"- prompt_stem: `{prompt_stem}`",
+        f"- started_at_utc: `{started_at_utc}`",
+        "",
+        "## Execution Facts",
+        "",
+        f"- execution_status: `{execution_status}`",
+        f"- finished_at_utc: `{finished_at_utc}`",
+        f"- runner: `{RUNNER_PATH}`",
+        f"- return_code: `{return_code}`",
+        "- retry_of_run_id:",
+        "",
+        "## Review Facts",
+        "",
+        "- review_status: `UNREVIEWED`",
+        "- review_summary:",
+        "- reviewed_by:",
+        "- reviewed_at_utc:",
+        "",
+        "## Failure Analysis",
+        "",
+        "- failure_type:",
+        "- failure_symptom:",
+        "- likely_cause:",
+        "- recommended_next_action:",
+        "",
+        "## Resource / Cost Facts",
+        "",
+        f"- elapsed_seconds: `{elapsed_seconds:.3f}`",
+        f"- final_output_char_count: `{len(codex_output)}`",
+        f"- stderr_char_count: `{len(stderr_text)}`",
+        "",
+        "## Prompt Text",
+        "",
+        fenced_block(prompt_text, "```", "md"),
+        "",
+        "## Codex Final Output",
+        "",
+        final_output if final_output else "*No output captured.*",
+        "",
+        "## Stderr",
+        "",
+        fenced_block(stderr_body, "```", "text") if stderr_body else "*No stderr captured.*",
+        "",
+    ]
+    return "\n".join(sections)
+
+
+def main() -> int:
+    parser = argparse.ArgumentParser(
+        description="Run one prompt file through codex exec and write a V1 execution record."
+    )
+    parser.add_argument("prompt", help="Prompt filename, numeric prefix, or path")
+    args = parser.parse_args()
+
+    root = repo_root()
+    prompts_dir = root / PROMPTS_DIR
+    notes_dir = root / NOTES_DIR
+
+    if not prompts_dir.exists():
+        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
+        return 1
+
+    if not notes_dir.exists():
+        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
+        return 1
+
+    prompt_path = resolve_prompt_path(root, args.prompt)
+    if not prompt_path.exists():
+        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
+        return 1
+
+    prompt_text = prompt_path.read_text(encoding="utf-8")
+    started_at = utc_now()
+    started_at_utc = utc_timestamp(started_at)
+    run_id, record_path = build_record_path(notes_dir, prompt_path.stem, started_at_utc)
+
+    monotonic_start = time.monotonic()
+    return_code, codex_output, stderr_text = run_codex(root, prompt_text)
+    elapsed_seconds = time.monotonic() - monotonic_start
+    finished_at_utc = utc_timestamp(utc_now())
+    execution_status = "EXECUTED" if return_code == 0 else "EXECUTION_FAILED"
+
+    record_path.write_text(
+        build_record_content(
+            run_id=run_id,
+            prompt_file=prompt_file_label(root, prompt_path),
+            prompt_stem=prompt_path.stem,
+            started_at_utc=started_at_utc,
+            execution_status=execution_status,
+            finished_at_utc=finished_at_utc,
+            return_code=return_code,
+            prompt_text=prompt_text,
+            codex_output=codex_output,
+            stderr_text=stderr_text,
+            elapsed_seconds=elapsed_seconds,
+        ),
+        encoding="utf-8",
+    )
+
+    print(record_path)
+    return return_code
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

codex
The runner file is in place. I’m doing a quick readback and syntax check before I use it to produce the sample execution record.
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
/bin/bash -lc 'python3 -m py_compile tools/codex/run_prompt.py' in /data/git/handson-ml3
 succeeded in 0ms:
diff --git a/tools/codex/run_prompt.py b/tools/codex/run_prompt.py
new file mode 100644
index 0000000000000000000000000000000000000000..3b9ea148301aefed31de2705205b247bbe223699
--- /dev/null
+++ b/tools/codex/run_prompt.py
@@ -0,0 +1,223 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import subprocess
+import sys
+import tempfile
+import time
+from pathlib import Path
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+RUNNER_PATH = "tools/codex/run_prompt.py"
+
+
+def utc_now() -> dt.datetime:
+    return dt.datetime.now(dt.timezone.utc)
+
+
+def utc_timestamp(moment: dt.datetime) -> str:
+    return moment.strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
+    prompt = Path(prompt_arg)
+
+    if prompt.is_absolute():
+        return prompt
+
+    direct = root / prompt
+    if direct.exists():
+        return direct
+
+    in_prompts = root / PROMPTS_DIR / prompt
+    if in_prompts.exists():
+        return in_prompts
+
+    matches = sorted((root / PROMPTS_DIR).glob(f"{prompt_arg}*"))
+    if len(matches) == 1:
+        return matches[0]
+
+    return in_prompts
+
+
+def prompt_file_label(root: Path, prompt_path: Path) -> str:
+    try:
+        return prompt_path.resolve().relative_to(root.resolve()).as_posix()
+    except ValueError:
+        return str(prompt_path.resolve())
+
+
+def build_record_path(notes_dir: Path, prompt_stem: str, started_at_utc: str) -> tuple[str, Path]:
+    base_run_id = f"{prompt_stem}__{started_at_utc}"
+    candidate = notes_dir / f"{base_run_id}.md"
+    if not candidate.exists():
+        return base_run_id, candidate
+
+    suffix = 2
+    while True:
+        run_id = f"{base_run_id}__{suffix}"
+        candidate = notes_dir / f"{run_id}.md"
+        if not candidate.exists():
+            return run_id, candidate
+        suffix += 1
+
+
+def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
+    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
+        output_path = Path(handle.name)
+
+    try:
+        result = subprocess.run(
+            ["codex", "exec", "-C", str(root), "--output-last-message", str(output_path), "-"],
+            input=prompt_text,
+            text=True,
+            capture_output=True,
+            check=False,
+        )
+        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
+        return result.returncode, final_output, result.stderr
+    finally:
+        output_path.unlink(missing_ok=True)
+
+
+def fenced_block(text: str, fence: str, info: str) -> str:
+    body = text.rstrip("\n")
+    return f"{fence}{info}\n{body}\n{fence}" if body else f"{fence}{info}\n\n{fence}"
+
+
+def build_record_content(
+    *,
+    run_id: str,
+    prompt_file: str,
+    prompt_stem: str,
+    started_at_utc: str,
+    execution_status: str,
+    finished_at_utc: str,
+    return_code: int,
+    prompt_text: str,
+    codex_output: str,
+    stderr_text: str,
+    elapsed_seconds: float,
+) -> str:
+    final_output = codex_output.rstrip()
+    stderr_body = stderr_text.rstrip()
+
+    sections = [
+        f"# {run_id}",
+        "",
+        f"- run_id: `{run_id}`",
+        f"- prompt_file: `{prompt_file}`",
+        f"- prompt_stem: `{prompt_stem}`",
+        f"- started_at_utc: `{started_at_utc}`",
+        "",
+        "## Execution Facts",
+        "",
+        f"- execution_status: `{execution_status}`",
+        f"- finished_at_utc: `{finished_at_utc}`",
+        f"- runner: `{RUNNER_PATH}`",
+        f"- return_code: `{return_code}`",
+        "- retry_of_run_id:",
+        "",
+        "## Review Facts",
+        "",
+        "- review_status: `UNREVIEWED`",
+        "- review_summary:",
+        "- reviewed_by:",
+        "- reviewed_at_utc:",
+        "",
+        "## Failure Analysis",
+        "",
+        "- failure_type:",
+        "- failure_symptom:",
+        "- likely_cause:",
+        "- recommended_next_action:",
+        "",
+        "## Resource / Cost Facts",
+        "",
+        f"- elapsed_seconds: `{elapsed_seconds:.3f}`",
+        f"- final_output_char_count: `{len(codex_output)}`",
+        f"- stderr_char_count: `{len(stderr_text)}`",
+        "",
+        "## Prompt Text",
+        "",
+        fenced_block(prompt_text, "```", "md"),
+        "",
+        "## Codex Final Output",
+        "",
+        final_output if final_output else "*No output captured.*",
+        "",
+        "## Stderr",
+        "",
+        fenced_block(stderr_body, "```", "text") if stderr_body else "*No stderr captured.*",
+        "",
+    ]
+    return "\n".join(sections)
+
+
+def main() -> int:
+    parser = argparse.ArgumentParser(
+        description="Run one prompt file through codex exec and write a V1 execution record."
+    )
+    parser.add_argument("prompt", help="Prompt filename, numeric prefix, or path")
+    args = parser.parse_args()
+
+    root = repo_root()
+    prompts_dir = root / PROMPTS_DIR
+    notes_dir = root / NOTES_DIR
+
+    if not prompts_dir.exists():
+        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
+        return 1
+
+    if not notes_dir.exists():
+        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
+        return 1
+
+    prompt_path = resolve_prompt_path(root, args.prompt)
+    if not prompt_path.exists():
+        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
+        return 1
+
+    prompt_text = prompt_path.read_text(encoding="utf-8")
+    started_at = utc_now()
+    started_at_utc = utc_timestamp(started_at)
+    run_id, record_path = build_record_path(notes_dir, prompt_path.stem, started_at_utc)
+
+    monotonic_start = time.monotonic()
+    return_code, codex_output, stderr_text = run_codex(root, prompt_text)
+    elapsed_seconds = time.monotonic() - monotonic_start
+    finished_at_utc = utc_timestamp(utc_now())
+    execution_status = "EXECUTED" if return_code == 0 else "EXECUTION_FAILED"
+
+    record_path.write_text(
+        build_record_content(
+            run_id=run_id,
+            prompt_file=prompt_file_label(root, prompt_path),
+            prompt_stem=prompt_path.stem,
+            started_at_utc=started_at_utc,
+            execution_status=execution_status,
+            finished_at_utc=finished_at_utc,
+            return_code=return_code,
+            prompt_text=prompt_text,
+            codex_output=codex_output,
+            stderr_text=stderr_text,
+            elapsed_seconds=elapsed_seconds,
+        ),
+        encoding="utf-8",
+    )
+
+    print(record_path)
+    return return_code
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

codex
I’ve verified the script parses. Next I’m running `tools/codex/run_prompt.py` against the smoke-test prompt to generate the sample V1 execution record required for validation.
exec
/bin/bash -lc 'python3 tools/codex/run_prompt.py codex_prompts/001_smoke_test_pipeline.md' in /data/git/handson-ml3
diff --git a/tools/codex/run_prompt.py b/tools/codex/run_prompt.py
new file mode 100644
index 0000000000000000000000000000000000000000..3b9ea148301aefed31de2705205b247bbe223699
--- /dev/null
+++ b/tools/codex/run_prompt.py
@@ -0,0 +1,223 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import subprocess
+import sys
+import tempfile
+import time
+from pathlib import Path
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+RUNNER_PATH = "tools/codex/run_prompt.py"
+
+
+def utc_now() -> dt.datetime:
+    return dt.datetime.now(dt.timezone.utc)
+
+
+def utc_timestamp(moment: dt.datetime) -> str:
+    return moment.strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
+    prompt = Path(prompt_arg)
+
+    if prompt.is_absolute():
+        return prompt
+
+    direct = root / prompt
+    if direct.exists():
+        return direct
+
+    in_prompts = root / PROMPTS_DIR / prompt
+    if in_prompts.exists():
+        return in_prompts
+
+    matches = sorted((root / PROMPTS_DIR).glob(f"{prompt_arg}*"))
+    if len(matches) == 1:
+        return matches[0]
+
+    return in_prompts
+
+
+def prompt_file_label(root: Path, prompt_path: Path) -> str:
+    try:
+        return prompt_path.resolve().relative_to(root.resolve()).as_posix()
+    except ValueError:
+        return str(prompt_path.resolve())
+
+
+def build_record_path(notes_dir: Path, prompt_stem: str, started_at_utc: str) -> tuple[str, Path]:
+    base_run_id = f"{prompt_stem}__{started_at_utc}"
+    candidate = notes_dir / f"{base_run_id}.md"
+    if not candidate.exists():
+        return base_run_id, candidate
+
+    suffix = 2
+    while True:
+        run_id = f"{base_run_id}__{suffix}"
+        candidate = notes_dir / f"{run_id}.md"
+        if not candidate.exists():
+            return run_id, candidate
+        suffix += 1
+
+
+def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
+    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
+        output_path = Path(handle.name)
+
+    try:
+        result = subprocess.run(
+            ["codex", "exec", "-C", str(root), "--output-last-message", str(output_path), "-"],
+            input=prompt_text,
+            text=True,
+            capture_output=True,
+            check=False,
+        )
+        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
+        return result.returncode, final_output, result.stderr
+    finally:
+        output_path.unlink(missing_ok=True)
+
+
+def fenced_block(text: str, fence: str, info: str) -> str:
+    body = text.rstrip("\n")
+    return f"{fence}{info}\n{body}\n{fence}" if body else f"{fence}{info}\n\n{fence}"
+
+
+def build_record_content(
+    *,
+    run_id: str,
+    prompt_file: str,
+    prompt_stem: str,
+    started_at_utc: str,
+    execution_status: str,
+    finished_at_utc: str,
+    return_code: int,
+    prompt_text: str,
+    codex_output: str,
+    stderr_text: str,
+    elapsed_seconds: float,
+) -> str:
+    final_output = codex_output.rstrip()
+    stderr_body = stderr_text.rstrip()
+
+    sections = [
+        f"# {run_id}",
+        "",
+        f"- run_id: `{run_id}`",
+        f"- prompt_file: `{prompt_file}`",
+        f"- prompt_stem: `{prompt_stem}`",
+        f"- started_at_utc: `{started_at_utc}`",
+        "",
+        "## Execution Facts",
+        "",
+        f"- execution_status: `{execution_status}`",
+        f"- finished_at_utc: `{finished_at_utc}`",
+        f"- runner: `{RUNNER_PATH}`",
+        f"- return_code: `{return_code}`",
+        "- retry_of_run_id:",
+        "",
+        "## Review Facts",
+        "",
+        "- review_status: `UNREVIEWED`",
+        "- review_summary:",
+        "- reviewed_by:",
+        "- reviewed_at_utc:",
+        "",
+        "## Failure Analysis",
+        "",
+        "- failure_type:",
+        "- failure_symptom:",
+        "- likely_cause:",
+        "- recommended_next_action:",
+        "",
+        "## Resource / Cost Facts",
+        "",
+        f"- elapsed_seconds: `{elapsed_seconds:.3f}`",
+        f"- final_output_char_count: `{len(codex_output)}`",
+        f"- stderr_char_count: `{len(stderr_text)}`",
+        "",
+        "## Prompt Text",
+        "",
+        fenced_block(prompt_text, "```", "md"),
+        "",
+        "## Codex Final Output",
+        "",
+        final_output if final_output else "*No output captured.*",
+        "",
+        "## Stderr",
+        "",
+        fenced_block(stderr_body, "```", "text") if stderr_body else "*No stderr captured.*",
+        "",
+    ]
+    return "\n".join(sections)
+
+
+def main() -> int:
+    parser = argparse.ArgumentParser(
+        description="Run one prompt file through codex exec and write a V1 execution record."
+    )
+    parser.add_argument("prompt", help="Prompt filename, numeric prefix, or path")
+    args = parser.parse_args()
+
+    root = repo_root()
+    prompts_dir = root / PROMPTS_DIR
+    notes_dir = root / NOTES_DIR
+
+    if not prompts_dir.exists():
+        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
+        return 1
+
+    if not notes_dir.exists():
+        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
+        return 1
+
+    prompt_path = resolve_prompt_path(root, args.prompt)
+    if not prompt_path.exists():
+        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
+        return 1
+
+    prompt_text = prompt_path.read_text(encoding="utf-8")
+    started_at = utc_now()
+    started_at_utc = utc_timestamp(started_at)
+    run_id, record_path = build_record_path(notes_dir, prompt_path.stem, started_at_utc)
+
+    monotonic_start = time.monotonic()
+    return_code, codex_output, stderr_text = run_codex(root, prompt_text)
+    elapsed_seconds = time.monotonic() - monotonic_start
+    finished_at_utc = utc_timestamp(utc_now())
+    execution_status = "EXECUTED" if return_code == 0 else "EXECUTION_FAILED"
+
+    record_path.write_text(
+        build_record_content(
+            run_id=run_id,
+            prompt_file=prompt_file_label(root, prompt_path),
+            prompt_stem=prompt_path.stem,
+            started_at_utc=started_at_utc,
+            execution_status=execution_status,
+            finished_at_utc=finished_at_utc,
+            return_code=return_code,
+            prompt_text=prompt_text,
+            codex_output=codex_output,
+            stderr_text=stderr_text,
+            elapsed_seconds=elapsed_seconds,
+        ),
+        encoding="utf-8",
+    )
+
+    print(record_path)
+    return return_code
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

 exited 1 in 782ms:
/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md

codex
The validation run completed and wrote a record, but the subprocess exit code was `1`, so I’m inspecting the generated artifact and confirming the bootstrap runner stayed untouched before I write the implementation note.
exec
/bin/bash -lc "sed -n '1,260p' notes/001_smoke_test_pipeline__20260415_233343.md" in /data/git/handson-ml3
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
/bin/bash -lc 'git diff -- tools/codex/baby_run_prompt.py' in /data/git/handson-ml3
 succeeded in 0ms:
exec
/bin/bash -lc "python3 - <<'PY'
from pathlib import Path
path = Path('notes/001_smoke_test_pipeline__20260415_233343.md')
text = path.read_text(encoding='utf-8')
checks = {
    'run_id': '- run_id:' in text,
    'execution_facts': '## Execution Facts' in text,
    'review_facts': '## Review Facts' in text,
    'failure_analysis': '## Failure Analysis' in text,
    'resource_cost': '## Resource / Cost Facts' in text,
    'prompt_text': '## Prompt Text' in text,
    'final_output': '## Codex Final Output' in text,
    'stderr': '## Stderr' in text,
    'review_unreviewed': '- review_status: "'`UNREVIEWED`'"' in text,
    'execution_failed': '- execution_status: "'`EXECUTION_FAILED`'"' in text,
}
for k,v in checks.items():
    print(f'{k}={v}')
PY" in /data/git/handson-ml3
diff --git a/tools/codex/run_prompt.py b/tools/codex/run_prompt.py
new file mode 100644
index 0000000000000000000000000000000000000000..3b9ea148301aefed31de2705205b247bbe223699
--- /dev/null
+++ b/tools/codex/run_prompt.py
@@ -0,0 +1,223 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import subprocess
+import sys
+import tempfile
+import time
+from pathlib import Path
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+RUNNER_PATH = "tools/codex/run_prompt.py"
+
+
+def utc_now() -> dt.datetime:
+    return dt.datetime.now(dt.timezone.utc)
+
+
+def utc_timestamp(moment: dt.datetime) -> str:
+    return moment.strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
+    prompt = Path(prompt_arg)
+
+    if prompt.is_absolute():
+        return prompt
+
+    direct = root / prompt
+    if direct.exists():
+        return direct
+
+    in_prompts = root / PROMPTS_DIR / prompt
+    if in_prompts.exists():
+        return in_prompts
+
+    matches = sorted((root / PROMPTS_DIR).glob(f"{prompt_arg}*"))
+    if len(matches) == 1:
+        return matches[0]
+
+    return in_prompts
+
+
+def prompt_file_label(root: Path, prompt_path: Path) -> str:
+    try:
+        return prompt_path.resolve().relative_to(root.resolve()).as_posix()
+    except ValueError:
+        return str(prompt_path.resolve())
+
+
+def build_record_path(notes_dir: Path, prompt_stem: str, started_at_utc: str) -> tuple[str, Path]:
+    base_run_id = f"{prompt_stem}__{started_at_utc}"
+    candidate = notes_dir / f"{base_run_id}.md"
+    if not candidate.exists():
+        return base_run_id, candidate
+
+    suffix = 2
+    while True:
+        run_id = f"{base_run_id}__{suffix}"
+        candidate = notes_dir / f"{run_id}.md"
+        if not candidate.exists():
+            return run_id, candidate
+        suffix += 1
+
+
+def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
+    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
+        output_path = Path(handle.name)
+
+    try:
+        result = subprocess.run(
+            ["codex", "exec", "-C", str(root), "--output-last-message", str(output_path), "-"],
+            input=prompt_text,
+            text=True,
+            capture_output=True,
+            check=False,
+        )
+        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
+        return result.returncode, final_output, result.stderr
+    finally:
+        output_path.unlink(missing_ok=True)
+
+
+def fenced_block(text: str, fence: str, info: str) -> str:
+    body = text.rstrip("\n")
+    return f"{fence}{info}\n{body}\n{fence}" if body else f"{fence}{info}\n\n{fence}"
+
+
+def build_record_content(
+    *,
+    run_id: str,
+    prompt_file: str,
+    prompt_stem: str,
+    started_at_utc: str,
+    execution_status: str,
+    finished_at_utc: str,
+    return_code: int,
+    prompt_text: str,
+    codex_output: str,
+    stderr_text: str,
+    elapsed_seconds: float,
+) -> str:
+    final_output = codex_output.rstrip()
+    stderr_body = stderr_text.rstrip()
+
+    sections = [
+        f"# {run_id}",
+        "",
+        f"- run_id: `{run_id}`",
+        f"- prompt_file: `{prompt_file}`",
+        f"- prompt_stem: `{prompt_stem}`",
+        f"- started_at_utc: `{started_at_utc}`",
+        "",
+        "## Execution Facts",
+        "",
+        f"- execution_status: `{execution_status}`",
+        f"- finished_at_utc: `{finished_at_utc}`",
+        f"- runner: `{RUNNER_PATH}`",
+        f"- return_code: `{return_code}`",
+        "- retry_of_run_id:",
+        "",
+        "## Review Facts",
+        "",
+        "- review_status: `UNREVIEWED`",
+        "- review_summary:",
+        "- reviewed_by:",
+        "- reviewed_at_utc:",
+        "",
+        "## Failure Analysis",
+        "",
+        "- failure_type:",
+        "- failure_symptom:",
+        "- likely_cause:",
+        "- recommended_next_action:",
+        "",
+        "## Resource / Cost Facts",
+        "",
+        f"- elapsed_seconds: `{elapsed_seconds:.3f}`",
+        f"- final_output_char_count: `{len(codex_output)}`",
+        f"- stderr_char_count: `{len(stderr_text)}`",
+        "",
+        "## Prompt Text",
+        "",
+        fenced_block(prompt_text, "```", "md"),
+        "",
+        "## Codex Final Output",
+        "",
+        final_output if final_output else "*No output captured.*",
+        "",
+        "## Stderr",
+        "",
+        fenced_block(stderr_body, "```", "text") if stderr_body else "*No stderr captured.*",
+        "",
+    ]
+    return "\n".join(sections)
+
+
+def main() -> int:
+    parser = argparse.ArgumentParser(
+        description="Run one prompt file through codex exec and write a V1 execution record."
+    )
+    parser.add_argument("prompt", help="Prompt filename, numeric prefix, or path")
+    args = parser.parse_args()
+
+    root = repo_root()
+    prompts_dir = root / PROMPTS_DIR
+    notes_dir = root / NOTES_DIR
+
+    if not prompts_dir.exists():
+        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
+        return 1
+
+    if not notes_dir.exists():
+        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
+        return 1
+
+    prompt_path = resolve_prompt_path(root, args.prompt)
+    if not prompt_path.exists():
+        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
+        return 1
+
+    prompt_text = prompt_path.read_text(encoding="utf-8")
+    started_at = utc_now()
+    started_at_utc = utc_timestamp(started_at)
+    run_id, record_path = build_record_path(notes_dir, prompt_path.stem, started_at_utc)
+
+    monotonic_start = time.monotonic()
+    return_code, codex_output, stderr_text = run_codex(root, prompt_text)
+    elapsed_seconds = time.monotonic() - monotonic_start
+    finished_at_utc = utc_timestamp(utc_now())
+    execution_status = "EXECUTED" if return_code == 0 else "EXECUTION_FAILED"
+
+    record_path.write_text(
+        build_record_content(
+            run_id=run_id,
+            prompt_file=prompt_file_label(root, prompt_path),
+            prompt_stem=prompt_path.stem,
+            started_at_utc=started_at_utc,
+            execution_status=execution_status,
+            finished_at_utc=finished_at_utc,
+            return_code=return_code,
+            prompt_text=prompt_text,
+            codex_output=codex_output,
+            stderr_text=stderr_text,
+            elapsed_seconds=elapsed_seconds,
+        ),
+        encoding="utf-8",
+    )
+
+    print(record_path)
+    return return_code
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

 succeeded in 158ms:
run_id=True
execution_facts=True
review_facts=True
failure_analysis=True
resource_cost=True
prompt_text=True
final_output=True
stderr=True
review_unreviewed=True
execution_failed=True

codex
The sample record has the required shape. The validation run itself failed inside `codex exec` because session initialization hit a read-only filesystem error, but that still proved the runner’s V1 failure-path recording and exit-code behavior.
exec
/bin/bash -lc 'date -u +%Y%m%d_%H%M%S' in /data/git/handson-ml3
 succeeded in 0ms:
20260415_233407

exec
/bin/bash -lc 'git status --short tools/codex/run_prompt.py tools/codex/baby_run_prompt.py notes/001_smoke_test_pipeline__20260415_233343.md' in /data/git/handson-ml3
 succeeded in 0ms:
?? notes/001_smoke_test_pipeline__20260415_233343.md
?? tools/codex/run_prompt.py

diff --git a/tools/codex/run_prompt.py b/tools/codex/run_prompt.py
new file mode 100644
index 0000000000000000000000000000000000000000..3b9ea148301aefed31de2705205b247bbe223699
--- /dev/null
+++ b/tools/codex/run_prompt.py
@@ -0,0 +1,223 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import subprocess
+import sys
+import tempfile
+import time
+from pathlib import Path
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+RUNNER_PATH = "tools/codex/run_prompt.py"
+
+
+def utc_now() -> dt.datetime:
+    return dt.datetime.now(dt.timezone.utc)
+
+
+def utc_timestamp(moment: dt.datetime) -> str:
+    return moment.strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
+    prompt = Path(prompt_arg)
+
+    if prompt.is_absolute():
+        return prompt
+
+    direct = root / prompt
+    if direct.exists():
+        return direct
+
+    in_prompts = root / PROMPTS_DIR / prompt
+    if in_prompts.exists():
+        return in_prompts
+
+    matches = sorted((root / PROMPTS_DIR).glob(f"{prompt_arg}*"))
+    if len(matches) == 1:
+        return matches[0]
+
+    return in_prompts
+
+
+def prompt_file_label(root: Path, prompt_path: Path) -> str:
+    try:
+        return prompt_path.resolve().relative_to(root.resolve()).as_posix()
+    except ValueError:
+        return str(prompt_path.resolve())
+
+
+def build_record_path(notes_dir: Path, prompt_stem: str, started_at_utc: str) -> tuple[str, Path]:
+    base_run_id = f"{prompt_stem}__{started_at_utc}"
+    candidate = notes_dir / f"{base_run_id}.md"
+    if not candidate.exists():
+        return base_run_id, candidate
+
+    suffix = 2
+    while True:
+        run_id = f"{base_run_id}__{suffix}"
+        candidate = notes_dir / f"{run_id}.md"
+        if not candidate.exists():
+            return run_id, candidate
+        suffix += 1
+
+
+def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
+    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
+        output_path = Path(handle.name)
+
+    try:
+        result = subprocess.run(
+            ["codex", "exec", "-C", str(root), "--output-last-message", str(output_path), "-"],
+            input=prompt_text,
+            text=True,
+            capture_output=True,
+            check=False,
+        )
+        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
+        return result.returncode, final_output, result.stderr
+    finally:
+        output_path.unlink(missing_ok=True)
+
+
+def fenced_block(text: str, fence: str, info: str) -> str:
+    body = text.rstrip("\n")
+    return f"{fence}{info}\n{body}\n{fence}" if body else f"{fence}{info}\n\n{fence}"
+
+
+def build_record_content(
+    *,
+    run_id: str,
+    prompt_file: str,
+    prompt_stem: str,
+    started_at_utc: str,
+    execution_status: str,
+    finished_at_utc: str,
+    return_code: int,
+    prompt_text: str,
+    codex_output: str,
+    stderr_text: str,
+    elapsed_seconds: float,
+) -> str:
+    final_output = codex_output.rstrip()
+    stderr_body = stderr_text.rstrip()
+
+    sections = [
+        f"# {run_id}",
+        "",
+        f"- run_id: `{run_id}`",
+        f"- prompt_file: `{prompt_file}`",
+        f"- prompt_stem: `{prompt_stem}`",
+        f"- started_at_utc: `{started_at_utc}`",
+        "",
+        "## Execution Facts",
+        "",
+        f"- execution_status: `{execution_status}`",
+        f"- finished_at_utc: `{finished_at_utc}`",
+        f"- runner: `{RUNNER_PATH}`",
+        f"- return_code: `{return_code}`",
+        "- retry_of_run_id:",
+        "",
+        "## Review Facts",
+        "",
+        "- review_status: `UNREVIEWED`",
+        "- review_summary:",
+        "- reviewed_by:",
+        "- reviewed_at_utc:",
+        "",
+        "## Failure Analysis",
+        "",
+        "- failure_type:",
+        "- failure_symptom:",
+        "- likely_cause:",
+        "- recommended_next_action:",
+        "",
+        "## Resource / Cost Facts",
+        "",
+        f"- elapsed_seconds: `{elapsed_seconds:.3f}`",
+        f"- final_output_char_count: `{len(codex_output)}`",
+        f"- stderr_char_count: `{len(stderr_text)}`",
+        "",
+        "## Prompt Text",
+        "",
+        fenced_block(prompt_text, "```", "md"),
+        "",
+        "## Codex Final Output",
+        "",
+        final_output if final_output else "*No output captured.*",
+        "",
+        "## Stderr",
+        "",
+        fenced_block(stderr_body, "```", "text") if stderr_body else "*No stderr captured.*",
+        "",
+    ]
+    return "\n".join(sections)
+
+
+def main() -> int:
+    parser = argparse.ArgumentParser(
+        description="Run one prompt file through codex exec and write a V1 execution record."
+    )
+    parser.add_argument("prompt", help="Prompt filename, numeric prefix, or path")
+    args = parser.parse_args()
+
+    root = repo_root()
+    prompts_dir = root / PROMPTS_DIR
+    notes_dir = root / NOTES_DIR
+
+    if not prompts_dir.exists():
+        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
+        return 1
+
+    if not notes_dir.exists():
+        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
+        return 1
+
+    prompt_path = resolve_prompt_path(root, args.prompt)
+    if not prompt_path.exists():
+        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
+        return 1
+
+    prompt_text = prompt_path.read_text(encoding="utf-8")
+    started_at = utc_now()
+    started_at_utc = utc_timestamp(started_at)
+    run_id, record_path = build_record_path(notes_dir, prompt_path.stem, started_at_utc)
+
+    monotonic_start = time.monotonic()
+    return_code, codex_output, stderr_text = run_codex(root, prompt_text)
+    elapsed_seconds = time.monotonic() - monotonic_start
+    finished_at_utc = utc_timestamp(utc_now())
+    execution_status = "EXECUTED" if return_code == 0 else "EXECUTION_FAILED"
+
+    record_path.write_text(
+        build_record_content(
+            run_id=run_id,
+            prompt_file=prompt_file_label(root, prompt_path),
+            prompt_stem=prompt_path.stem,
+            started_at_utc=started_at_utc,
+            execution_status=execution_status,
+            finished_at_utc=finished_at_utc,
+            return_code=return_code,
+            prompt_text=prompt_text,
+            codex_output=codex_output,
+            stderr_text=stderr_text,
+            elapsed_seconds=elapsed_seconds,
+        ),
+        encoding="utf-8",
+    )
+
+    print(record_path)
+    return return_code
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

codex
I’m writing the implementation note now. It will capture what was built, what was preserved, what changed for V1, the collision rule, what was deferred, and the validation result including the environment failure from `codex exec`.
apply patch
patch: completed
/data/git/handson-ml3/notes/009_run_prompt_candidate_build__20260415_233407.md
diff --git a/notes/009_run_prompt_candidate_build__20260415_233407.md b/notes/009_run_prompt_candidate_build__20260415_233407.md
new file mode 100644
index 0000000000000000000000000000000000000000..d25109549b5dd2f75546347cda782e9522027676
--- /dev/null
+++ b/notes/009_run_prompt_candidate_build__20260415_233407.md
@@ -0,0 +1,73 @@
+# 009 Run Prompt Candidate Build
+
+## Summary
+
+Built `tools/codex/run_prompt.py` as a thin V1 runner candidate that resolves one prompt, runs `codex exec` from the repo root, captures `--output-last-message`, captures stderr and return code, writes one V1 execution-record markdown file into `notes/`, prints that path, and exits with the subprocess return code.
+
+## Preserved From `baby_run_prompt.py`
+
+- Kept repo-root discovery via `Path(__file__).resolve().parents[2]`.
+- Kept prompt resolution behavior for absolute paths, repo-relative paths, `codex_prompts/` paths, and unique prefix matches.
+- Kept direct `subprocess.run(...)` execution of `codex exec`.
+- Kept `--output-last-message` capture through a temp file.
+- Kept the requirement that `codex_prompts/` and `notes/` already exist.
+- Kept one markdown artifact per run in `notes/`.
+- Kept printing the written artifact path and returning the subprocess exit code.
+
+## V1 Changes
+
+- Switched artifact output from the old success/failure note format to a V1 execution record.
+- Separated execution facts from review facts.
+- Initialized review fields with `review_status: UNREVIEWED` and blank manual review fields.
+- Added the manual failure-analysis section with blank fields.
+- Added the minimum resource/cost fields:
+  - `elapsed_seconds`
+  - `final_output_char_count`
+  - `stderr_char_count`
+- Set `execution_status` strictly from subprocess exit code:
+  - `0 -> EXECUTED`
+  - non-zero -> `EXECUTION_FAILED`
+- Recorded `prompt_file` as repo-relative when possible.
+- Updated the runner identity field to `tools/codex/run_prompt.py`.
+
+## Collision Rule
+
+Run identity uses second-precision UTC timestamps in the base form:
+
+- `<prompt_stem>__<started_at_utc>`
+
+If `notes/<run_id>.md` already exists, the runner appends a numeric suffix:
+
+- `__2`, `__3`, and so on
+
+This keeps the V1 identity scheme small while avoiding same-second filename collisions.
+
+## Deferred
+
+- Review write-back
+- Queue progression logic
+- Dependency handling
+- Retry orchestration
+- Broader CLI redesign
+- Multi-module refactor
+- JSON sidecars, databases, or any other persistence layer
+
+## Validation
+
+Executed:
+
+- `python3 tools/codex/run_prompt.py codex_prompts/001_smoke_test_pipeline.md`
+
+Outcome:
+
+- The runner resolved the sample prompt and wrote [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md).
+- The record filename follows the V1 naming rule: `<prompt_stem>__<started_at_utc>.md`.
+- The record includes run identity, execution facts, review facts, failure analysis, resource/cost facts, prompt text, Codex final output, and stderr.
+- `review_status` starts as `UNREVIEWED`.
+- `execution_status` is derived only from the subprocess result. In the validation run it was `EXECUTION_FAILED` because `codex exec` exited `1`.
+- `tools/codex/baby_run_prompt.py` remained unchanged. `git diff -- tools/codex/baby_run_prompt.py` returned no diff.
+
+Validation note:
+
+- The sample run did not execute successfully inside `codex exec` because Codex session initialization failed with `Read-only file system (os error 30)`.
+- That environment failure still validated the required V1 failure-path behavior: the new runner wrote the execution record, captured stderr, printed the record path, and exited with the subprocess return code.
diff --git a/tools/codex/run_prompt.py b/tools/codex/run_prompt.py
new file mode 100644
index 0000000000000000000000000000000000000000..3b9ea148301aefed31de2705205b247bbe223699
--- /dev/null
+++ b/tools/codex/run_prompt.py
@@ -0,0 +1,223 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import subprocess
+import sys
+import tempfile
+import time
+from pathlib import Path
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+RUNNER_PATH = "tools/codex/run_prompt.py"
+
+
+def utc_now() -> dt.datetime:
+    return dt.datetime.now(dt.timezone.utc)
+
+
+def utc_timestamp(moment: dt.datetime) -> str:
+    return moment.strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
+    prompt = Path(prompt_arg)
+
+    if prompt.is_absolute():
+        return prompt
+
+    direct = root / prompt
+    if direct.exists():
+        return direct
+
+    in_prompts = root / PROMPTS_DIR / prompt
+    if in_prompts.exists():
+        return in_prompts
+
+    matches = sorted((root / PROMPTS_DIR).glob(f"{prompt_arg}*"))
+    if len(matches) == 1:
+        return matches[0]
+
+    return in_prompts
+
+
+def prompt_file_label(root: Path, prompt_path: Path) -> str:
+    try:
+        return prompt_path.resolve().relative_to(root.resolve()).as_posix()
+    except ValueError:
+        return str(prompt_path.resolve())
+
+
+def build_record_path(notes_dir: Path, prompt_stem: str, started_at_utc: str) -> tuple[str, Path]:
+    base_run_id = f"{prompt_stem}__{started_at_utc}"
+    candidate = notes_dir / f"{base_run_id}.md"
+    if not candidate.exists():
+        return base_run_id, candidate
+
+    suffix = 2
+    while True:
+        run_id = f"{base_run_id}__{suffix}"
+        candidate = notes_dir / f"{run_id}.md"
+        if not candidate.exists():
+            return run_id, candidate
+        suffix += 1
+
+
+def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
+    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
+        output_path = Path(handle.name)
+
+    try:
+        result = subprocess.run(
+            ["codex", "exec", "-C", str(root), "--output-last-message", str(output_path), "-"],
+            input=prompt_text,
+            text=True,
+            capture_output=True,
+            check=False,
+        )
+        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
+        return result.returncode, final_output, result.stderr
+    finally:
+        output_path.unlink(missing_ok=True)
+
+
+def fenced_block(text: str, fence: str, info: str) -> str:
+    body = text.rstrip("\n")
+    return f"{fence}{info}\n{body}\n{fence}" if body else f"{fence}{info}\n\n{fence}"
+
+
+def build_record_content(
+    *,
+    run_id: str,
+    prompt_file: str,
+    prompt_stem: str,
+    started_at_utc: str,
+    execution_status: str,
+    finished_at_utc: str,
+    return_code: int,
+    prompt_text: str,
+    codex_output: str,
+    stderr_text: str,
+    elapsed_seconds: float,
+) -> str:
+    final_output = codex_output.rstrip()
+    stderr_body = stderr_text.rstrip()
+
+    sections = [
+        f"# {run_id}",
+        "",
+        f"- run_id: `{run_id}`",
+        f"- prompt_file: `{prompt_file}`",
+        f"- prompt_stem: `{prompt_stem}`",
+        f"- started_at_utc: `{started_at_utc}`",
+        "",
+        "## Execution Facts",
+        "",
+        f"- execution_status: `{execution_status}`",
+        f"- finished_at_utc: `{finished_at_utc}`",
+        f"- runner: `{RUNNER_PATH}`",
+        f"- return_code: `{return_code}`",
+        "- retry_of_run_id:",
+        "",
+        "## Review Facts",
+        "",
+        "- review_status: `UNREVIEWED`",
+        "- review_summary:",
+        "- reviewed_by:",
+        "- reviewed_at_utc:",
+        "",
+        "## Failure Analysis",
+        "",
+        "- failure_type:",
+        "- failure_symptom:",
+        "- likely_cause:",
+        "- recommended_next_action:",
+        "",
+        "## Resource / Cost Facts",
+        "",
+        f"- elapsed_seconds: `{elapsed_seconds:.3f}`",
+        f"- final_output_char_count: `{len(codex_output)}`",
+        f"- stderr_char_count: `{len(stderr_text)}`",
+        "",
+        "## Prompt Text",
+        "",
+        fenced_block(prompt_text, "```", "md"),
+        "",
+        "## Codex Final Output",
+        "",
+        final_output if final_output else "*No output captured.*",
+        "",
+        "## Stderr",
+        "",
+        fenced_block(stderr_body, "```", "text") if stderr_body else "*No stderr captured.*",
+        "",
+    ]
+    return "\n".join(sections)
+
+
+def main() -> int:
+    parser = argparse.ArgumentParser(
+        description="Run one prompt file through codex exec and write a V1 execution record."
+    )
+    parser.add_argument("prompt", help="Prompt filename, numeric prefix, or path")
+    args = parser.parse_args()
+
+    root = repo_root()
+    prompts_dir = root / PROMPTS_DIR
+    notes_dir = root / NOTES_DIR
+
+    if not prompts_dir.exists():
+        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
+        return 1
+
+    if not notes_dir.exists():
+        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
+        return 1
+
+    prompt_path = resolve_prompt_path(root, args.prompt)
+    if not prompt_path.exists():
+        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
+        return 1
+
+    prompt_text = prompt_path.read_text(encoding="utf-8")
+    started_at = utc_now()
+    started_at_utc = utc_timestamp(started_at)
+    run_id, record_path = build_record_path(notes_dir, prompt_path.stem, started_at_utc)
+
+    monotonic_start = time.monotonic()
+    return_code, codex_output, stderr_text = run_codex(root, prompt_text)
+    elapsed_seconds = time.monotonic() - monotonic_start
+    finished_at_utc = utc_timestamp(utc_now())
+    execution_status = "EXECUTED" if return_code == 0 else "EXECUTION_FAILED"
+
+    record_path.write_text(
+        build_record_content(
+            run_id=run_id,
+            prompt_file=prompt_file_label(root, prompt_path),
+            prompt_stem=prompt_path.stem,
+            started_at_utc=started_at_utc,
+            execution_status=execution_status,
+            finished_at_utc=finished_at_utc,
+            return_code=return_code,
+            prompt_text=prompt_text,
+            codex_output=codex_output,
+            stderr_text=stderr_text,
+            elapsed_seconds=elapsed_seconds,
+        ),
+        encoding="utf-8",
+    )
+
+    print(record_path)
+    return return_code
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

diff --git a/notes/009_run_prompt_candidate_build__20260415_233407.md b/notes/009_run_prompt_candidate_build__20260415_233407.md
new file mode 100644
index 0000000000000000000000000000000000000000..d25109549b5dd2f75546347cda782e9522027676
--- /dev/null
+++ b/notes/009_run_prompt_candidate_build__20260415_233407.md
@@ -0,0 +1,73 @@
+# 009 Run Prompt Candidate Build
+
+## Summary
+
+Built `tools/codex/run_prompt.py` as a thin V1 runner candidate that resolves one prompt, runs `codex exec` from the repo root, captures `--output-last-message`, captures stderr and return code, writes one V1 execution-record markdown file into `notes/`, prints that path, and exits with the subprocess return code.
+
+## Preserved From `baby_run_prompt.py`
+
+- Kept repo-root discovery via `Path(__file__).resolve().parents[2]`.
+- Kept prompt resolution behavior for absolute paths, repo-relative paths, `codex_prompts/` paths, and unique prefix matches.
+- Kept direct `subprocess.run(...)` execution of `codex exec`.
+- Kept `--output-last-message` capture through a temp file.
+- Kept the requirement that `codex_prompts/` and `notes/` already exist.
+- Kept one markdown artifact per run in `notes/`.
+- Kept printing the written artifact path and returning the subprocess exit code.
+
+## V1 Changes
+
+- Switched artifact output from the old success/failure note format to a V1 execution record.
+- Separated execution facts from review facts.
+- Initialized review fields with `review_status: UNREVIEWED` and blank manual review fields.
+- Added the manual failure-analysis section with blank fields.
+- Added the minimum resource/cost fields:
+  - `elapsed_seconds`
+  - `final_output_char_count`
+  - `stderr_char_count`
+- Set `execution_status` strictly from subprocess exit code:
+  - `0 -> EXECUTED`
+  - non-zero -> `EXECUTION_FAILED`
+- Recorded `prompt_file` as repo-relative when possible.
+- Updated the runner identity field to `tools/codex/run_prompt.py`.
+
+## Collision Rule
+
+Run identity uses second-precision UTC timestamps in the base form:
+
+- `<prompt_stem>__<started_at_utc>`
+
+If `notes/<run_id>.md` already exists, the runner appends a numeric suffix:
+
+- `__2`, `__3`, and so on
+
+This keeps the V1 identity scheme small while avoiding same-second filename collisions.
+
+## Deferred
+
+- Review write-back
+- Queue progression logic
+- Dependency handling
+- Retry orchestration
+- Broader CLI redesign
+- Multi-module refactor
+- JSON sidecars, databases, or any other persistence layer
+
+## Validation
+
+Executed:
+
+- `python3 tools/codex/run_prompt.py codex_prompts/001_smoke_test_pipeline.md`
+
+Outcome:
+
+- The runner resolved the sample prompt and wrote [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md).
+- The record filename follows the V1 naming rule: `<prompt_stem>__<started_at_utc>.md`.
+- The record includes run identity, execution facts, review facts, failure analysis, resource/cost facts, prompt text, Codex final output, and stderr.
+- `review_status` starts as `UNREVIEWED`.
+- `execution_status` is derived only from the subprocess result. In the validation run it was `EXECUTION_FAILED` because `codex exec` exited `1`.
+- `tools/codex/baby_run_prompt.py` remained unchanged. `git diff -- tools/codex/baby_run_prompt.py` returned no diff.
+
+Validation note:
+
+- The sample run did not execute successfully inside `codex exec` because Codex session initialization failed with `Read-only file system (os error 30)`.
+- That environment failure still validated the required V1 failure-path behavior: the new runner wrote the execution record, captured stderr, printed the record path, and exited with the subprocess return code.
diff --git a/tools/codex/run_prompt.py b/tools/codex/run_prompt.py
new file mode 100644
index 0000000000000000000000000000000000000000..3b9ea148301aefed31de2705205b247bbe223699
--- /dev/null
+++ b/tools/codex/run_prompt.py
@@ -0,0 +1,223 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import subprocess
+import sys
+import tempfile
+import time
+from pathlib import Path
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+RUNNER_PATH = "tools/codex/run_prompt.py"
+
+
+def utc_now() -> dt.datetime:
+    return dt.datetime.now(dt.timezone.utc)
+
+
+def utc_timestamp(moment: dt.datetime) -> str:
+    return moment.strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
+    prompt = Path(prompt_arg)
+
+    if prompt.is_absolute():
+        return prompt
+
+    direct = root / prompt
+    if direct.exists():
+        return direct
+
+    in_prompts = root / PROMPTS_DIR / prompt
+    if in_prompts.exists():
+        return in_prompts
+
+    matches = sorted((root / PROMPTS_DIR).glob(f"{prompt_arg}*"))
+    if len(matches) == 1:
+        return matches[0]
+
+    return in_prompts
+
+
+def prompt_file_label(root: Path, prompt_path: Path) -> str:
+    try:
+        return prompt_path.resolve().relative_to(root.resolve()).as_posix()
+    except ValueError:
+        return str(prompt_path.resolve())
+
+
+def build_record_path(notes_dir: Path, prompt_stem: str, started_at_utc: str) -> tuple[str, Path]:
+    base_run_id = f"{prompt_stem}__{started_at_utc}"
+    candidate = notes_dir / f"{base_run_id}.md"
+    if not candidate.exists():
+        return base_run_id, candidate
+
+    suffix = 2
+    while True:
+        run_id = f"{base_run_id}__{suffix}"
+        candidate = notes_dir / f"{run_id}.md"
+        if not candidate.exists():
+            return run_id, candidate
+        suffix += 1
+
+
+def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
+    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
+        output_path = Path(handle.name)
+
+    try:
+        result = subprocess.run(
+            ["codex", "exec", "-C", str(root), "--output-last-message", str(output_path), "-"],
+            input=prompt_text,
+            text=True,
+            capture_output=True,
+            check=False,
+        )
+        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
+        return result.returncode, final_output, result.stderr
+    finally:
+        output_path.unlink(missing_ok=True)
+
+
+def fenced_block(text: str, fence: str, info: str) -> str:
+    body = text.rstrip("\n")
+    return f"{fence}{info}\n{body}\n{fence}" if body else f"{fence}{info}\n\n{fence}"
+
+
+def build_record_content(
+    *,
+    run_id: str,
+    prompt_file: str,
+    prompt_stem: str,
+    started_at_utc: str,
+    execution_status: str,
+    finished_at_utc: str,
+    return_code: int,
+    prompt_text: str,
+    codex_output: str,
+    stderr_text: str,
+    elapsed_seconds: float,
+) -> str:
+    final_output = codex_output.rstrip()
+    stderr_body = stderr_text.rstrip()
+
+    sections = [
+        f"# {run_id}",
+        "",
+        f"- run_id: `{run_id}`",
+        f"- prompt_file: `{prompt_file}`",
+        f"- prompt_stem: `{prompt_stem}`",
+        f"- started_at_utc: `{started_at_utc}`",
+        "",
+        "## Execution Facts",
+        "",
+        f"- execution_status: `{execution_status}`",
+        f"- finished_at_utc: `{finished_at_utc}`",
+        f"- runner: `{RUNNER_PATH}`",
+        f"- return_code: `{return_code}`",
+        "- retry_of_run_id:",
+        "",
+        "## Review Facts",
+        "",
+        "- review_status: `UNREVIEWED`",
+        "- review_summary:",
+        "- reviewed_by:",
+        "- reviewed_at_utc:",
+        "",
+        "## Failure Analysis",
+        "",
+        "- failure_type:",
+        "- failure_symptom:",
+        "- likely_cause:",
+        "- recommended_next_action:",
+        "",
+        "## Resource / Cost Facts",
+        "",
+        f"- elapsed_seconds: `{elapsed_seconds:.3f}`",
+        f"- final_output_char_count: `{len(codex_output)}`",
+        f"- stderr_char_count: `{len(stderr_text)}`",
+        "",
+        "## Prompt Text",
+        "",
+        fenced_block(prompt_text, "```", "md"),
+        "",
+        "## Codex Final Output",
+        "",
+        final_output if final_output else "*No output captured.*",
+        "",
+        "## Stderr",
+        "",
+        fenced_block(stderr_body, "```", "text") if stderr_body else "*No stderr captured.*",
+        "",
+    ]
+    return "\n".join(sections)
+
+
+def main() -> int:
+    parser = argparse.ArgumentParser(
+        description="Run one prompt file through codex exec and write a V1 execution record."
+    )
+    parser.add_argument("prompt", help="Prompt filename, numeric prefix, or path")
+    args = parser.parse_args()
+
+    root = repo_root()
+    prompts_dir = root / PROMPTS_DIR
+    notes_dir = root / NOTES_DIR
+
+    if not prompts_dir.exists():
+        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
+        return 1
+
+    if not notes_dir.exists():
+        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
+        return 1
+
+    prompt_path = resolve_prompt_path(root, args.prompt)
+    if not prompt_path.exists():
+        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
+        return 1
+
+    prompt_text = prompt_path.read_text(encoding="utf-8")
+    started_at = utc_now()
+    started_at_utc = utc_timestamp(started_at)
+    run_id, record_path = build_record_path(notes_dir, prompt_path.stem, started_at_utc)
+
+    monotonic_start = time.monotonic()
+    return_code, codex_output, stderr_text = run_codex(root, prompt_text)
+    elapsed_seconds = time.monotonic() - monotonic_start
+    finished_at_utc = utc_timestamp(utc_now())
+    execution_status = "EXECUTED" if return_code == 0 else "EXECUTION_FAILED"
+
+    record_path.write_text(
+        build_record_content(
+            run_id=run_id,
+            prompt_file=prompt_file_label(root, prompt_path),
+            prompt_stem=prompt_path.stem,
+            started_at_utc=started_at_utc,
+            execution_status=execution_status,
+            finished_at_utc=finished_at_utc,
+            return_code=return_code,
+            prompt_text=prompt_text,
+            codex_output=codex_output,
+            stderr_text=stderr_text,
+            elapsed_seconds=elapsed_seconds,
+        ),
+        encoding="utf-8",
+    )
+
+    print(record_path)
+    return return_code
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

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
/bin/bash -lc 'git diff -- tools/codex/run_prompt.py notes/009_run_prompt_candidate_build__20260415_233407.md' in /data/git/handson-ml3
 succeeded in 0ms:
diff --git a/notes/009_run_prompt_candidate_build__20260415_233407.md b/notes/009_run_prompt_candidate_build__20260415_233407.md
new file mode 100644
index 0000000000000000000000000000000000000000..d25109549b5dd2f75546347cda782e9522027676
--- /dev/null
+++ b/notes/009_run_prompt_candidate_build__20260415_233407.md
@@ -0,0 +1,73 @@
+# 009 Run Prompt Candidate Build
+
+## Summary
+
+Built `tools/codex/run_prompt.py` as a thin V1 runner candidate that resolves one prompt, runs `codex exec` from the repo root, captures `--output-last-message`, captures stderr and return code, writes one V1 execution-record markdown file into `notes/`, prints that path, and exits with the subprocess return code.
+
+## Preserved From `baby_run_prompt.py`
+
+- Kept repo-root discovery via `Path(__file__).resolve().parents[2]`.
+- Kept prompt resolution behavior for absolute paths, repo-relative paths, `codex_prompts/` paths, and unique prefix matches.
+- Kept direct `subprocess.run(...)` execution of `codex exec`.
+- Kept `--output-last-message` capture through a temp file.
+- Kept the requirement that `codex_prompts/` and `notes/` already exist.
+- Kept one markdown artifact per run in `notes/`.
+- Kept printing the written artifact path and returning the subprocess exit code.
+
+## V1 Changes
+
+- Switched artifact output from the old success/failure note format to a V1 execution record.
+- Separated execution facts from review facts.
+- Initialized review fields with `review_status: UNREVIEWED` and blank manual review fields.
+- Added the manual failure-analysis section with blank fields.
+- Added the minimum resource/cost fields:
+  - `elapsed_seconds`
+  - `final_output_char_count`
+  - `stderr_char_count`
+- Set `execution_status` strictly from subprocess exit code:
+  - `0 -> EXECUTED`
+  - non-zero -> `EXECUTION_FAILED`
+- Recorded `prompt_file` as repo-relative when possible.
+- Updated the runner identity field to `tools/codex/run_prompt.py`.
+
+## Collision Rule
+
+Run identity uses second-precision UTC timestamps in the base form:
+
+- `<prompt_stem>__<started_at_utc>`
+
+If `notes/<run_id>.md` already exists, the runner appends a numeric suffix:
+
+- `__2`, `__3`, and so on
+
+This keeps the V1 identity scheme small while avoiding same-second filename collisions.
+
+## Deferred
+
+- Review write-back
+- Queue progression logic
+- Dependency handling
+- Retry orchestration
+- Broader CLI redesign
+- Multi-module refactor
+- JSON sidecars, databases, or any other persistence layer
+
+## Validation
+
+Executed:
+
+- `python3 tools/codex/run_prompt.py codex_prompts/001_smoke_test_pipeline.md`
+
+Outcome:
+
+- The runner resolved the sample prompt and wrote [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md).
+- The record filename follows the V1 naming rule: `<prompt_stem>__<started_at_utc>.md`.
+- The record includes run identity, execution facts, review facts, failure analysis, resource/cost facts, prompt text, Codex final output, and stderr.
+- `review_status` starts as `UNREVIEWED`.
+- `execution_status` is derived only from the subprocess result. In the validation run it was `EXECUTION_FAILED` because `codex exec` exited `1`.
+- `tools/codex/baby_run_prompt.py` remained unchanged. `git diff -- tools/codex/baby_run_prompt.py` returned no diff.
+
+Validation note:
+
+- The sample run did not execute successfully inside `codex exec` because Codex session initialization failed with `Read-only file system (os error 30)`.
+- That environment failure still validated the required V1 failure-path behavior: the new runner wrote the execution record, captured stderr, printed the record path, and exited with the subprocess return code.
diff --git a/tools/codex/run_prompt.py b/tools/codex/run_prompt.py
new file mode 100644
index 0000000000000000000000000000000000000000..3b9ea148301aefed31de2705205b247bbe223699
--- /dev/null
+++ b/tools/codex/run_prompt.py
@@ -0,0 +1,223 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import subprocess
+import sys
+import tempfile
+import time
+from pathlib import Path
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+RUNNER_PATH = "tools/codex/run_prompt.py"
+
+
+def utc_now() -> dt.datetime:
+    return dt.datetime.now(dt.timezone.utc)
+
+
+def utc_timestamp(moment: dt.datetime) -> str:
+    return moment.strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
+    prompt = Path(prompt_arg)
+
+    if prompt.is_absolute():
+        return prompt
+
+    direct = root / prompt
+    if direct.exists():
+        return direct
+
+    in_prompts = root / PROMPTS_DIR / prompt
+    if in_prompts.exists():
+        return in_prompts
+
+    matches = sorted((root / PROMPTS_DIR).glob(f"{prompt_arg}*"))
+    if len(matches) == 1:
+        return matches[0]
+
+    return in_prompts
+
+
+def prompt_file_label(root: Path, prompt_path: Path) -> str:
+    try:
+        return prompt_path.resolve().relative_to(root.resolve()).as_posix()
+    except ValueError:
+        return str(prompt_path.resolve())
+
+
+def build_record_path(notes_dir: Path, prompt_stem: str, started_at_utc: str) -> tuple[str, Path]:
+    base_run_id = f"{prompt_stem}__{started_at_utc}"
+    candidate = notes_dir / f"{base_run_id}.md"
+    if not candidate.exists():
+        return base_run_id, candidate
+
+    suffix = 2
+    while True:
+        run_id = f"{base_run_id}__{suffix}"
+        candidate = notes_dir / f"{run_id}.md"
+        if not candidate.exists():
+            return run_id, candidate
+        suffix += 1
+
+
+def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
+    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
+        output_path = Path(handle.name)
+
+    try:
+        result = subprocess.run(
+            ["codex", "exec", "-C", str(root), "--output-last-message", str(output_path), "-"],
+            input=prompt_text,
+            text=True,
+            capture_output=True,
+            check=False,
+        )
+        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
+        return result.returncode, final_output, result.stderr
+    finally:
+        output_path.unlink(missing_ok=True)
+
+
+def fenced_block(text: str, fence: str, info: str) -> str:
+    body = text.rstrip("\n")
+    return f"{fence}{info}\n{body}\n{fence}" if body else f"{fence}{info}\n\n{fence}"
+
+
+def build_record_content(
+    *,
+    run_id: str,
+    prompt_file: str,
+    prompt_stem: str,
+    started_at_utc: str,
+    execution_status: str,
+    finished_at_utc: str,
+    return_code: int,
+    prompt_text: str,
+    codex_output: str,
+    stderr_text: str,
+    elapsed_seconds: float,
+) -> str:
+    final_output = codex_output.rstrip()
+    stderr_body = stderr_text.rstrip()
+
+    sections = [
+        f"# {run_id}",
+        "",
+        f"- run_id: `{run_id}`",
+        f"- prompt_file: `{prompt_file}`",
+        f"- prompt_stem: `{prompt_stem}`",
+        f"- started_at_utc: `{started_at_utc}`",
+        "",
+        "## Execution Facts",
+        "",
+        f"- execution_status: `{execution_status}`",
+        f"- finished_at_utc: `{finished_at_utc}`",
+        f"- runner: `{RUNNER_PATH}`",
+        f"- return_code: `{return_code}`",
+        "- retry_of_run_id:",
+        "",
+        "## Review Facts",
+        "",
+        "- review_status: `UNREVIEWED`",
+        "- review_summary:",
+        "- reviewed_by:",
+        "- reviewed_at_utc:",
+        "",
+        "## Failure Analysis",
+        "",
+        "- failure_type:",
+        "- failure_symptom:",
+        "- likely_cause:",
+        "- recommended_next_action:",
+        "",
+        "## Resource / Cost Facts",
+        "",
+        f"- elapsed_seconds: `{elapsed_seconds:.3f}`",
+        f"- final_output_char_count: `{len(codex_output)}`",
+        f"- stderr_char_count: `{len(stderr_text)}`",
+        "",
+        "## Prompt Text",
+        "",
+        fenced_block(prompt_text, "```", "md"),
+        "",
+        "## Codex Final Output",
+        "",
+        final_output if final_output else "*No output captured.*",
+        "",
+        "## Stderr",
+        "",
+        fenced_block(stderr_body, "```", "text") if stderr_body else "*No stderr captured.*",
+        "",
+    ]
+    return "\n".join(sections)
+
+
+def main() -> int:
+    parser = argparse.ArgumentParser(
+        description="Run one prompt file through codex exec and write a V1 execution record."
+    )
+    parser.add_argument("prompt", help="Prompt filename, numeric prefix, or path")
+    args = parser.parse_args()
+
+    root = repo_root()
+    prompts_dir = root / PROMPTS_DIR
+    notes_dir = root / NOTES_DIR
+
+    if not prompts_dir.exists():
+        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
+        return 1
+
+    if not notes_dir.exists():
+        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
+        return 1
+
+    prompt_path = resolve_prompt_path(root, args.prompt)
+    if not prompt_path.exists():
+        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
+        return 1
+
+    prompt_text = prompt_path.read_text(encoding="utf-8")
+    started_at = utc_now()
+    started_at_utc = utc_timestamp(started_at)
+    run_id, record_path = build_record_path(notes_dir, prompt_path.stem, started_at_utc)
+
+    monotonic_start = time.monotonic()
+    return_code, codex_output, stderr_text = run_codex(root, prompt_text)
+    elapsed_seconds = time.monotonic() - monotonic_start
+    finished_at_utc = utc_timestamp(utc_now())
+    execution_status = "EXECUTED" if return_code == 0 else "EXECUTION_FAILED"
+
+    record_path.write_text(
+        build_record_content(
+            run_id=run_id,
+            prompt_file=prompt_file_label(root, prompt_path),
+            prompt_stem=prompt_path.stem,
+            started_at_utc=started_at_utc,
+            execution_status=execution_status,
+            finished_at_utc=finished_at_utc,
+            return_code=return_code,
+            prompt_text=prompt_text,
+            codex_output=codex_output,
+            stderr_text=stderr_text,
+            elapsed_seconds=elapsed_seconds,
+        ),
+        encoding="utf-8",
+    )
+
+    print(record_path)
+    return return_code
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

  ✓ Add tools/codex/run_prompt.py with V1 execution-record behavior and collision-safe run-id naming
  ✓ Run the new runner against codex_prompts/001_smoke_test_pipeline.md to generate a sample execution record
  ✓ Write notes/009_run_prompt_candidate_build__TIMESTAMP.md with implementation and validation results
diff --git a/notes/009_run_prompt_candidate_build__20260415_233407.md b/notes/009_run_prompt_candidate_build__20260415_233407.md
new file mode 100644
index 0000000000000000000000000000000000000000..d25109549b5dd2f75546347cda782e9522027676
--- /dev/null
+++ b/notes/009_run_prompt_candidate_build__20260415_233407.md
@@ -0,0 +1,73 @@
+# 009 Run Prompt Candidate Build
+
+## Summary
+
+Built `tools/codex/run_prompt.py` as a thin V1 runner candidate that resolves one prompt, runs `codex exec` from the repo root, captures `--output-last-message`, captures stderr and return code, writes one V1 execution-record markdown file into `notes/`, prints that path, and exits with the subprocess return code.
+
+## Preserved From `baby_run_prompt.py`
+
+- Kept repo-root discovery via `Path(__file__).resolve().parents[2]`.
+- Kept prompt resolution behavior for absolute paths, repo-relative paths, `codex_prompts/` paths, and unique prefix matches.
+- Kept direct `subprocess.run(...)` execution of `codex exec`.
+- Kept `--output-last-message` capture through a temp file.
+- Kept the requirement that `codex_prompts/` and `notes/` already exist.
+- Kept one markdown artifact per run in `notes/`.
+- Kept printing the written artifact path and returning the subprocess exit code.
+
+## V1 Changes
+
+- Switched artifact output from the old success/failure note format to a V1 execution record.
+- Separated execution facts from review facts.
+- Initialized review fields with `review_status: UNREVIEWED` and blank manual review fields.
+- Added the manual failure-analysis section with blank fields.
+- Added the minimum resource/cost fields:
+  - `elapsed_seconds`
+  - `final_output_char_count`
+  - `stderr_char_count`
+- Set `execution_status` strictly from subprocess exit code:
+  - `0 -> EXECUTED`
+  - non-zero -> `EXECUTION_FAILED`
+- Recorded `prompt_file` as repo-relative when possible.
+- Updated the runner identity field to `tools/codex/run_prompt.py`.
+
+## Collision Rule
+
+Run identity uses second-precision UTC timestamps in the base form:
+
+- `<prompt_stem>__<started_at_utc>`
+
+If `notes/<run_id>.md` already exists, the runner appends a numeric suffix:
+
+- `__2`, `__3`, and so on
+
+This keeps the V1 identity scheme small while avoiding same-second filename collisions.
+
+## Deferred
+
+- Review write-back
+- Queue progression logic
+- Dependency handling
+- Retry orchestration
+- Broader CLI redesign
+- Multi-module refactor
+- JSON sidecars, databases, or any other persistence layer
+
+## Validation
+
+Executed:
+
+- `python3 tools/codex/run_prompt.py codex_prompts/001_smoke_test_pipeline.md`
+
+Outcome:
+
+- The runner resolved the sample prompt and wrote [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md).
+- The record filename follows the V1 naming rule: `<prompt_stem>__<started_at_utc>.md`.
+- The record includes run identity, execution facts, review facts, failure analysis, resource/cost facts, prompt text, Codex final output, and stderr.
+- `review_status` starts as `UNREVIEWED`.
+- `execution_status` is derived only from the subprocess result. In the validation run it was `EXECUTION_FAILED` because `codex exec` exited `1`.
+- `tools/codex/baby_run_prompt.py` remained unchanged. `git diff -- tools/codex/baby_run_prompt.py` returned no diff.
+
+Validation note:
+
+- The sample run did not execute successfully inside `codex exec` because Codex session initialization failed with `Read-only file system (os error 30)`.
+- That environment failure still validated the required V1 failure-path behavior: the new runner wrote the execution record, captured stderr, printed the record path, and exited with the subprocess return code.
diff --git a/tools/codex/run_prompt.py b/tools/codex/run_prompt.py
new file mode 100644
index 0000000000000000000000000000000000000000..3b9ea148301aefed31de2705205b247bbe223699
--- /dev/null
+++ b/tools/codex/run_prompt.py
@@ -0,0 +1,223 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import subprocess
+import sys
+import tempfile
+import time
+from pathlib import Path
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+RUNNER_PATH = "tools/codex/run_prompt.py"
+
+
+def utc_now() -> dt.datetime:
+    return dt.datetime.now(dt.timezone.utc)
+
+
+def utc_timestamp(moment: dt.datetime) -> str:
+    return moment.strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
+    prompt = Path(prompt_arg)
+
+    if prompt.is_absolute():
+        return prompt
+
+    direct = root / prompt
+    if direct.exists():
+        return direct
+
+    in_prompts = root / PROMPTS_DIR / prompt
+    if in_prompts.exists():
+        return in_prompts
+
+    matches = sorted((root / PROMPTS_DIR).glob(f"{prompt_arg}*"))
+    if len(matches) == 1:
+        return matches[0]
+
+    return in_prompts
+
+
+def prompt_file_label(root: Path, prompt_path: Path) -> str:
+    try:
+        return prompt_path.resolve().relative_to(root.resolve()).as_posix()
+    except ValueError:
+        return str(prompt_path.resolve())
+
+
+def build_record_path(notes_dir: Path, prompt_stem: str, started_at_utc: str) -> tuple[str, Path]:
+    base_run_id = f"{prompt_stem}__{started_at_utc}"
+    candidate = notes_dir / f"{base_run_id}.md"
+    if not candidate.exists():
+        return base_run_id, candidate
+
+    suffix = 2
+    while True:
+        run_id = f"{base_run_id}__{suffix}"
+        candidate = notes_dir / f"{run_id}.md"
+        if not candidate.exists():
+            return run_id, candidate
+        suffix += 1
+
+
+def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
+    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
+        output_path = Path(handle.name)
+
+    try:
+        result = subprocess.run(
+            ["codex", "exec", "-C", str(root), "--output-last-message", str(output_path), "-"],
+            input=prompt_text,
+            text=True,
+            capture_output=True,
+            check=False,
+        )
+        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
+        return result.returncode, final_output, result.stderr
+    finally:
+        output_path.unlink(missing_ok=True)
+
+
+def fenced_block(text: str, fence: str, info: str) -> str:
+    body = text.rstrip("\n")
+    return f"{fence}{info}\n{body}\n{fence}" if body else f"{fence}{info}\n\n{fence}"
+
+
+def build_record_content(
+    *,
+    run_id: str,
+    prompt_file: str,
+    prompt_stem: str,
+    started_at_utc: str,
+    execution_status: str,
+    finished_at_utc: str,
+    return_code: int,
+    prompt_text: str,
+    codex_output: str,
+    stderr_text: str,
+    elapsed_seconds: float,
+) -> str:
+    final_output = codex_output.rstrip()
+    stderr_body = stderr_text.rstrip()
+
+    sections = [
+        f"# {run_id}",
+        "",
+        f"- run_id: `{run_id}`",
+        f"- prompt_file: `{prompt_file}`",
+        f"- prompt_stem: `{prompt_stem}`",
+        f"- started_at_utc: `{started_at_utc}`",
+        "",
+        "## Execution Facts",
+        "",
+        f"- execution_status: `{execution_status}`",
+        f"- finished_at_utc: `{finished_at_utc}`",
+        f"- runner: `{RUNNER_PATH}`",
+        f"- return_code: `{return_code}`",
+        "- retry_of_run_id:",
+        "",
+        "## Review Facts",
+        "",
+        "- review_status: `UNREVIEWED`",
+        "- review_summary:",
+        "- reviewed_by:",
+        "- reviewed_at_utc:",
+        "",
+        "## Failure Analysis",
+        "",
+        "- failure_type:",
+        "- failure_symptom:",
+        "- likely_cause:",
+        "- recommended_next_action:",
+        "",
+        "## Resource / Cost Facts",
+        "",
+        f"- elapsed_seconds: `{elapsed_seconds:.3f}`",
+        f"- final_output_char_count: `{len(codex_output)}`",
+        f"- stderr_char_count: `{len(stderr_text)}`",
+        "",
+        "## Prompt Text",
+        "",
+        fenced_block(prompt_text, "```", "md"),
+        "",
+        "## Codex Final Output",
+        "",
+        final_output if final_output else "*No output captured.*",
+        "",
+        "## Stderr",
+        "",
+        fenced_block(stderr_body, "```", "text") if stderr_body else "*No stderr captured.*",
+        "",
+    ]
+    return "\n".join(sections)
+
+
+def main() -> int:
+    parser = argparse.ArgumentParser(
+        description="Run one prompt file through codex exec and write a V1 execution record."
+    )
+    parser.add_argument("prompt", help="Prompt filename, numeric prefix, or path")
+    args = parser.parse_args()
+
+    root = repo_root()
+    prompts_dir = root / PROMPTS_DIR
+    notes_dir = root / NOTES_DIR
+
+    if not prompts_dir.exists():
+        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
+        return 1
+
+    if not notes_dir.exists():
+        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
+        return 1
+
+    prompt_path = resolve_prompt_path(root, args.prompt)
+    if not prompt_path.exists():
+        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
+        return 1
+
+    prompt_text = prompt_path.read_text(encoding="utf-8")
+    started_at = utc_now()
+    started_at_utc = utc_timestamp(started_at)
+    run_id, record_path = build_record_path(notes_dir, prompt_path.stem, started_at_utc)
+
+    monotonic_start = time.monotonic()
+    return_code, codex_output, stderr_text = run_codex(root, prompt_text)
+    elapsed_seconds = time.monotonic() - monotonic_start
+    finished_at_utc = utc_timestamp(utc_now())
+    execution_status = "EXECUTED" if return_code == 0 else "EXECUTION_FAILED"
+
+    record_path.write_text(
+        build_record_content(
+            run_id=run_id,
+            prompt_file=prompt_file_label(root, prompt_path),
+            prompt_stem=prompt_path.stem,
+            started_at_utc=started_at_utc,
+            execution_status=execution_status,
+            finished_at_utc=finished_at_utc,
+            return_code=return_code,
+            prompt_text=prompt_text,
+            codex_output=codex_output,
+            stderr_text=stderr_text,
+            elapsed_seconds=elapsed_seconds,
+        ),
+        encoding="utf-8",
+    )
+
+    print(record_path)
+    return return_code
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

codex
Implemented the V1 runner candidate in [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py). It keeps the bootstrap runner’s prompt resolution and thin `codex exec` subprocess flow, but now writes a single V1 execution-record markdown artifact with separate execution and review sections, blank review/failure-analysis fields, lightweight resource metrics, stable run-id naming, and numeric suffix collision handling.

I also wrote the implementation note at [009_run_prompt_candidate_build__20260415_233407.md](/data/git/handson-ml3/notes/009_run_prompt_candidate_build__20260415_233407.md) and validated the runner by invoking it against the smoke-test prompt. That produced [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md), which has the required V1 shape. The `codex exec` subprocess itself exited `1` because session initialization hit `Read-only file system (os error 30)`, so the sample record correctly shows `execution_status: EXECUTION_FAILED`, `review_status: UNREVIEWED`, captured stderr, and the printed record path. `tools/codex/baby_run_prompt.py` was not modified.
diff --git a/notes/009_run_prompt_candidate_build__20260415_233407.md b/notes/009_run_prompt_candidate_build__20260415_233407.md
new file mode 100644
index 0000000000000000000000000000000000000000..d25109549b5dd2f75546347cda782e9522027676
--- /dev/null
+++ b/notes/009_run_prompt_candidate_build__20260415_233407.md
@@ -0,0 +1,73 @@
+# 009 Run Prompt Candidate Build
+
+## Summary
+
+Built `tools/codex/run_prompt.py` as a thin V1 runner candidate that resolves one prompt, runs `codex exec` from the repo root, captures `--output-last-message`, captures stderr and return code, writes one V1 execution-record markdown file into `notes/`, prints that path, and exits with the subprocess return code.
+
+## Preserved From `baby_run_prompt.py`
+
+- Kept repo-root discovery via `Path(__file__).resolve().parents[2]`.
+- Kept prompt resolution behavior for absolute paths, repo-relative paths, `codex_prompts/` paths, and unique prefix matches.
+- Kept direct `subprocess.run(...)` execution of `codex exec`.
+- Kept `--output-last-message` capture through a temp file.
+- Kept the requirement that `codex_prompts/` and `notes/` already exist.
+- Kept one markdown artifact per run in `notes/`.
+- Kept printing the written artifact path and returning the subprocess exit code.
+
+## V1 Changes
+
+- Switched artifact output from the old success/failure note format to a V1 execution record.
+- Separated execution facts from review facts.
+- Initialized review fields with `review_status: UNREVIEWED` and blank manual review fields.
+- Added the manual failure-analysis section with blank fields.
+- Added the minimum resource/cost fields:
+  - `elapsed_seconds`
+  - `final_output_char_count`
+  - `stderr_char_count`
+- Set `execution_status` strictly from subprocess exit code:
+  - `0 -> EXECUTED`
+  - non-zero -> `EXECUTION_FAILED`
+- Recorded `prompt_file` as repo-relative when possible.
+- Updated the runner identity field to `tools/codex/run_prompt.py`.
+
+## Collision Rule
+
+Run identity uses second-precision UTC timestamps in the base form:
+
+- `<prompt_stem>__<started_at_utc>`
+
+If `notes/<run_id>.md` already exists, the runner appends a numeric suffix:
+
+- `__2`, `__3`, and so on
+
+This keeps the V1 identity scheme small while avoiding same-second filename collisions.
+
+## Deferred
+
+- Review write-back
+- Queue progression logic
+- Dependency handling
+- Retry orchestration
+- Broader CLI redesign
+- Multi-module refactor
+- JSON sidecars, databases, or any other persistence layer
+
+## Validation
+
+Executed:
+
+- `python3 tools/codex/run_prompt.py codex_prompts/001_smoke_test_pipeline.md`
+
+Outcome:
+
+- The runner resolved the sample prompt and wrote [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md).
+- The record filename follows the V1 naming rule: `<prompt_stem>__<started_at_utc>.md`.
+- The record includes run identity, execution facts, review facts, failure analysis, resource/cost facts, prompt text, Codex final output, and stderr.
+- `review_status` starts as `UNREVIEWED`.
+- `execution_status` is derived only from the subprocess result. In the validation run it was `EXECUTION_FAILED` because `codex exec` exited `1`.
+- `tools/codex/baby_run_prompt.py` remained unchanged. `git diff -- tools/codex/baby_run_prompt.py` returned no diff.
+
+Validation note:
+
+- The sample run did not execute successfully inside `codex exec` because Codex session initialization failed with `Read-only file system (os error 30)`.
+- That environment failure still validated the required V1 failure-path behavior: the new runner wrote the execution record, captured stderr, printed the record path, and exited with the subprocess return code.
diff --git a/tools/codex/run_prompt.py b/tools/codex/run_prompt.py
new file mode 100644
index 0000000000000000000000000000000000000000..3b9ea148301aefed31de2705205b247bbe223699
--- /dev/null
+++ b/tools/codex/run_prompt.py
@@ -0,0 +1,223 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import subprocess
+import sys
+import tempfile
+import time
+from pathlib import Path
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+RUNNER_PATH = "tools/codex/run_prompt.py"
+
+
+def utc_now() -> dt.datetime:
+    return dt.datetime.now(dt.timezone.utc)
+
+
+def utc_timestamp(moment: dt.datetime) -> str:
+    return moment.strftime("%Y%m%d_%H%M%S")
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
+    prompt = Path(prompt_arg)
+
+    if prompt.is_absolute():
+        return prompt
+
+    direct = root / prompt
+    if direct.exists():
+        return direct
+
+    in_prompts = root / PROMPTS_DIR / prompt
+    if in_prompts.exists():
+        return in_prompts
+
+    matches = sorted((root / PROMPTS_DIR).glob(f"{prompt_arg}*"))
+    if len(matches) == 1:
+        return matches[0]
+
+    return in_prompts
+
+
+def prompt_file_label(root: Path, prompt_path: Path) -> str:
+    try:
+        return prompt_path.resolve().relative_to(root.resolve()).as_posix()
+    except ValueError:
+        return str(prompt_path.resolve())
+
+
+def build_record_path(notes_dir: Path, prompt_stem: str, started_at_utc: str) -> tuple[str, Path]:
+    base_run_id = f"{prompt_stem}__{started_at_utc}"
+    candidate = notes_dir / f"{base_run_id}.md"
+    if not candidate.exists():
+        return base_run_id, candidate
+
+    suffix = 2
+    while True:
+        run_id = f"{base_run_id}__{suffix}"
+        candidate = notes_dir / f"{run_id}.md"
+        if not candidate.exists():
+            return run_id, candidate
+        suffix += 1
+
+
+def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
+    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
+        output_path = Path(handle.name)
+
+    try:
+        result = subprocess.run(
+            ["codex", "exec", "-C", str(root), "--output-last-message", str(output_path), "-"],
+            input=prompt_text,
+            text=True,
+            capture_output=True,
+            check=False,
+        )
+        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
+        return result.returncode, final_output, result.stderr
+    finally:
+        output_path.unlink(missing_ok=True)
+
+
+def fenced_block(text: str, fence: str, info: str) -> str:
+    body = text.rstrip("\n")
+    return f"{fence}{info}\n{body}\n{fence}" if body else f"{fence}{info}\n\n{fence}"
+
+
+def build_record_content(
+    *,
+    run_id: str,
+    prompt_file: str,
+    prompt_stem: str,
+    started_at_utc: str,
+    execution_status: str,
+    finished_at_utc: str,
+    return_code: int,
+    prompt_text: str,
+    codex_output: str,
+    stderr_text: str,
+    elapsed_seconds: float,
+) -> str:
+    final_output = codex_output.rstrip()
+    stderr_body = stderr_text.rstrip()
+
+    sections = [
+        f"# {run_id}",
+        "",
+        f"- run_id: `{run_id}`",
+        f"- prompt_file: `{prompt_file}`",
+        f"- prompt_stem: `{prompt_stem}`",
+        f"- started_at_utc: `{started_at_utc}`",
+        "",
+        "## Execution Facts",
+        "",
+        f"- execution_status: `{execution_status}`",
+        f"- finished_at_utc: `{finished_at_utc}`",
+        f"- runner: `{RUNNER_PATH}`",
+        f"- return_code: `{return_code}`",
+        "- retry_of_run_id:",
+        "",
+        "## Review Facts",
+        "",
+        "- review_status: `UNREVIEWED`",
+        "- review_summary:",
+        "- reviewed_by:",
+        "- reviewed_at_utc:",
+        "",
+        "## Failure Analysis",
+        "",
+        "- failure_type:",
+        "- failure_symptom:",
+        "- likely_cause:",
+        "- recommended_next_action:",
+        "",
+        "## Resource / Cost Facts",
+        "",
+        f"- elapsed_seconds: `{elapsed_seconds:.3f}`",
+        f"- final_output_char_count: `{len(codex_output)}`",
+        f"- stderr_char_count: `{len(stderr_text)}`",
+        "",
+        "## Prompt Text",
+        "",
+        fenced_block(prompt_text, "```", "md"),
+        "",
+        "## Codex Final Output",
+        "",
+        final_output if final_output else "*No output captured.*",
+        "",
+        "## Stderr",
+        "",
+        fenced_block(stderr_body, "```", "text") if stderr_body else "*No stderr captured.*",
+        "",
+    ]
+    return "\n".join(sections)
+
+
+def main() -> int:
+    parser = argparse.ArgumentParser(
+        description="Run one prompt file through codex exec and write a V1 execution record."
+    )
+    parser.add_argument("prompt", help="Prompt filename, numeric prefix, or path")
+    args = parser.parse_args()
+
+    root = repo_root()
+    prompts_dir = root / PROMPTS_DIR
+    notes_dir = root / NOTES_DIR
+
+    if not prompts_dir.exists():
+        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
+        return 1
+
+    if not notes_dir.exists():
+        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
+        return 1
+
+    prompt_path = resolve_prompt_path(root, args.prompt)
+    if not prompt_path.exists():
+        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
+        return 1
+
+    prompt_text = prompt_path.read_text(encoding="utf-8")
+    started_at = utc_now()
+    started_at_utc = utc_timestamp(started_at)
+    run_id, record_path = build_record_path(notes_dir, prompt_path.stem, started_at_utc)
+
+    monotonic_start = time.monotonic()
+    return_code, codex_output, stderr_text = run_codex(root, prompt_text)
+    elapsed_seconds = time.monotonic() - monotonic_start
+    finished_at_utc = utc_timestamp(utc_now())
+    execution_status = "EXECUTED" if return_code == 0 else "EXECUTION_FAILED"
+
+    record_path.write_text(
+        build_record_content(
+            run_id=run_id,
+            prompt_file=prompt_file_label(root, prompt_path),
+            prompt_stem=prompt_path.stem,
+            started_at_utc=started_at_utc,
+            execution_status=execution_status,
+            finished_at_utc=finished_at_utc,
+            return_code=return_code,
+            prompt_text=prompt_text,
+            codex_output=codex_output,
+            stderr_text=stderr_text,
+            elapsed_seconds=elapsed_seconds,
+        ),
+        encoding="utf-8",
+    )
+
+    print(record_path)
+    return return_code
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

tokens used
42,845
```
