# 020_align_v1_doc_and_spec_packet_to_actual_toolset__20260416_010453

- run_id: `020_align_v1_doc_and_spec_packet_to_actual_toolset__20260416_010453`
- prompt_file: `codex_prompts/020_align_v1_doc_and_spec_packet_to_actual_toolset.md`
- prompt_stem: `020_align_v1_doc_and_spec_packet_to_actual_toolset`
- started_at_utc: `20260416_010453`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260416_010748`
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

- elapsed_seconds: `174.920`
- final_output_char_count: `1226`
- stderr_char_count: `209043`

## Prompt Text

```md
# Task: Align the V1 doc/spec packet to the actual implemented toolset

You are working in this repository.

Your task is to perform a bounded doc/spec alignment cleanup so the current V1 design packet matches the implemented workflow.

This is a cleanup pass, not a new architecture pass.

## Primary goal

Make the current V1 design packet accurately describe the actual implemented toolset and current behavior without expanding scope or redesigning the workflow.

## Files to inspect

Read these exact files before editing:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
- `notes/018_prioritized_remaining_work__20260416_005130.md`

## Required cleanup scope

Keep the work bounded to doc/spec alignment for the current V1 slice.

The cleanup must address at least these items:

1. `tools/codex/V1_Bridge_Runner_Change_Spec.md`
   - remove stale references to `tools/codex/baby_run_prompt.py`
   - align the document to the real runner at `tools/codex/run_prompt.py`
   - stop describing review write-back and queue support as unimplemented when helper scripts now exist

2. Any stale references in the inspected design packet that still imply:
   - the runner is `baby_run_prompt.py`
   - readiness and backlog helpers do not yet exist
   - the V1 run id is always unsuffixed

3. The current run-id collision behavior where relevant
   - document that the base V1 identity is `<prompt_stem>__<started_at_utc>`
   - document that `run_prompt.py` adds a numeric suffix such as `__2` only when needed to avoid same-second collisions
   - keep that explanation small and implementation-accurate

## Output artifacts to create

Create exactly one note:

- `notes/020_doc_spec_alignment_cleanup__TIMESTAMP.md`

Update only the minimum necessary design/spec files from the inspected list.

## Constraints

- Do not modify `tools/codex/run_prompt.py`
- Do not modify `tools/codex/review_run.py`
- Do not modify `tools/codex/check_queue_readiness.py`
- Do not modify `tools/codex/list_review_backlog.py`
- Do not rewrite the overall architecture
- Do not introduce new workflow states, new tools, or new platform layers
- Do not expand into retry tooling, scheduling, queue engines, dashboards, or orchestration
- Keep this as a V1-sized doc/spec cleanup only

## Validation requirements

Validate the cleanup by doing all of the following:

1. Confirm the updated design/spec files consistently name `tools/codex/run_prompt.py` as the active V1 runner where applicable.
2. Confirm the updated packet no longer implies that review write-back, readiness checking, or backlog listing are still future work if those helpers are already present.
3. Confirm the run-id wording is consistent with current code behavior, including same-second collision suffixes where relevant.
4. Use a search pass to verify there are no remaining stale `baby_run_prompt.py` references inside the edited design/spec files unless a historical reference is explicitly intentional and clearly marked as historical.

## Success criteria

This task is successful if:

1. The V1 design/spec packet matches the current implemented toolset closely enough that a new prompt author would not be misled.
2. `tools/codex/V1_Bridge_Runner_Change_Spec.md` no longer reads like a pre-implementation bridge spec for a different runner.
3. The packet reflects that the V1 workflow now includes:
   - `run_prompt.py`
   - `review_run.py`
   - `check_queue_readiness.py`
   - `list_review_backlog.py`
4. The run-id collision suffix behavior is described accurately but briefly.
5. The cleanup stays comfortably reviewable and does not turn into a broader rewrite.
```

## Codex Final Output

Aligned the V1 packet to the implemented toolset and added the required cleanup note at [020_doc_spec_alignment_cleanup__20260416_010534.md](/data/git/handson-ml3/notes/020_doc_spec_alignment_cleanup__20260416_010534.md).

I updated [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md), [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md), and [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md). The packet now names `tools/codex/run_prompt.py` as the active V1 runner where applicable, reflects that `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py` are implemented parts of the current workflow, and documents the run-id behavior accurately: base identity `<prompt_stem>__<started_at_utc>`, with a numeric suffix like `__2` only when needed for same-second collisions.

Validation was done with search passes and diff review. There are no remaining stale `baby_run_prompt.py` references in the edited design/spec files; the only remaining mentions are the intentional historical references inside the new cleanup note. No code files were modified.

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
session id: 019d93d1-cd96-78d3-810a-d591f32c3c6a
--------
user
# Task: Align the V1 doc/spec packet to the actual implemented toolset

You are working in this repository.

Your task is to perform a bounded doc/spec alignment cleanup so the current V1 design packet matches the implemented workflow.

This is a cleanup pass, not a new architecture pass.

## Primary goal

Make the current V1 design packet accurately describe the actual implemented toolset and current behavior without expanding scope or redesigning the workflow.

## Files to inspect

Read these exact files before editing:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
- `notes/018_prioritized_remaining_work__20260416_005130.md`

## Required cleanup scope

Keep the work bounded to doc/spec alignment for the current V1 slice.

The cleanup must address at least these items:

1. `tools/codex/V1_Bridge_Runner_Change_Spec.md`
   - remove stale references to `tools/codex/baby_run_prompt.py`
   - align the document to the real runner at `tools/codex/run_prompt.py`
   - stop describing review write-back and queue support as unimplemented when helper scripts now exist

2. Any stale references in the inspected design packet that still imply:
   - the runner is `baby_run_prompt.py`
   - readiness and backlog helpers do not yet exist
   - the V1 run id is always unsuffixed

3. The current run-id collision behavior where relevant
   - document that the base V1 identity is `<prompt_stem>__<started_at_utc>`
   - document that `run_prompt.py` adds a numeric suffix such as `__2` only when needed to avoid same-second collisions
   - keep that explanation small and implementation-accurate

## Output artifacts to create

Create exactly one note:

- `notes/020_doc_spec_alignment_cleanup__TIMESTAMP.md`

Update only the minimum necessary design/spec files from the inspected list.

## Constraints

- Do not modify `tools/codex/run_prompt.py`
- Do not modify `tools/codex/review_run.py`
- Do not modify `tools/codex/check_queue_readiness.py`
- Do not modify `tools/codex/list_review_backlog.py`
- Do not rewrite the overall architecture
- Do not introduce new workflow states, new tools, or new platform layers
- Do not expand into retry tooling, scheduling, queue engines, dashboards, or orchestration
- Keep this as a V1-sized doc/spec cleanup only

## Validation requirements

Validate the cleanup by doing all of the following:

1. Confirm the updated design/spec files consistently name `tools/codex/run_prompt.py` as the active V1 runner where applicable.
2. Confirm the updated packet no longer implies that review write-back, readiness checking, or backlog listing are still future work if those helpers are already present.
3. Confirm the run-id wording is consistent with current code behavior, including same-second collision suffixes where relevant.
4. Use a search pass to verify there are no remaining stale `baby_run_prompt.py` references inside the edited design/spec files unless a historical reference is explicitly intentional and clearly marked as historical.

## Success criteria

This task is successful if:

1. The V1 design/spec packet matches the current implemented toolset closely enough that a new prompt author would not be misled.
2. `tools/codex/V1_Bridge_Runner_Change_Spec.md` no longer reads like a pre-implementation bridge spec for a different runner.
3. The packet reflects that the V1 workflow now includes:
   - `run_prompt.py`
   - `review_run.py`
   - `check_queue_readiness.py`
   - `list_review_backlog.py`
4. The run-id collision suffix behavior is described accurately but briefly.
5. The cleanup stays comfortably reviewable and does not turn into a broader rewrite.

codex
I’m reviewing the current V1 design packet against the implemented scripts first, then I’ll make the smallest doc/spec edits needed and add the single cleanup note.
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
/bin/bash -lc "sed -n '1,320p' tools/codex/V1_Bridge_Runner_Change_Spec.md" in /data/git/handson-ml3
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


def format_prefixes(prefixes: list[int]) -> str:
    return ", ".join(f"{prefix:03d}" for prefix in prefixes)


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


def discover_legacy_success_prefixes(root: Path) -> set[int]:
    notes_dir = root / NOTES_DIR
    prefixes: set[int] = set()
    for path in sorted(notes_dir.glob("*__SUCCESS__*.md")):
        match = PROMPT_NAME_RE.match(path.stem)
        if match is not None:
            prefixes.add(int(match.group("prefix")))
    return prefixes



exec
/bin/bash -lc "sed -n '1,260p' tools/codex/list_review_backlog.py" in /data/git/handson-ml3
 succeeded in 0ms:
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


NOTES_DIR = "notes"
REQUIRED_FIELDS = (
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
REQUIRED_SECTIONS = (
    "## Execution Facts",
    "## Review Facts",
    "## Failure Analysis",
    "## Resource / Cost Facts",
    "## Prompt Text",
    "## Codex Final Output",
    "## Stderr",
)
EXECUTION_STATUSES = {"EXECUTED", "EXECUTION_FAILED"}
REVIEW_STATUSES = {"UNREVIEWED", "ACCEPTED", "REJECTED"}
TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


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


class BacklogError(Exception):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List the current V1 review backlog from execution records in notes/."
    )
    parser.add_argument(
        "--unreviewed-only",
        action="store_true",
        help="Limit the latest-per-prompt and needs-review views to UNREVIEWED latest records.",
    )
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


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

    match = re.fullmatch(rf"{re.escape(base_run_id)}__(\d+)", run_id)
    if match is None:
        raise BacklogError(f"run_id does not match V1 identity pattern: {run_id}")
    return int(match.group(1))


def validate_section_order(text: str, path: Path, root: Path) -> None:
    positions: list[int] = []
    for section in REQUIRED_SECTIONS:
        pos = text.find(section)
        if pos == -1:
            raise BacklogError(
                f"record-like note is missing required V1 section: {path.relative_to(root).as_posix()} "
                f"missing {section}"
            )
        positions.append(pos)

    if positions != sorted(positions):
        raise BacklogError(
            f"record-like note has V1 sections out of order: {path.relative_to(root).as_posix()}"
        )


def parse_record_file(root: Path, path: Path) -> RunRecord | None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("# "):
        return None

    has_execution_section = "## Execution Facts" in text
    has_review_section = "## Review Facts" in text
    if not has_execution_section and not has_review_section:
        return None
    if has_execution_section != has_review_section:
        raise BacklogError(
            f"record-like note has incomplete V1 sections: {path.relative_to(root).as_posix()}"
        )

    header_block = text.split("\n## ", 1)[0]
    identity_values = {field: parse_field_line(header_block, field) for field in IDENTITY_FIELDS}
    if identity_values["run_id"] is None:
        return None

    validate_section_order(text, path, root)

    metadata_block = text.split("\n## Prompt Text\n", 1)[0]
    values = {field: parse_field_line(metadata_block, field) for field in REQUIRED_FIELDS}
    missing_fields = [field for field, value in values.items() if value is None]
    if missing_fields:
        raise BacklogError(
            f"record-like note is missing required V1 fields: {path.relative_to(root).as_posix()} "
            f"missing {', '.join(missing_fields)}"
        )

    run_id = values["run_id"] or ""
    prompt_file = values["prompt_file"] or ""
    prompt_stem = values["prompt_stem"] or ""
    started_at_utc = values["started_at_utc"] or ""
    execution_status = values["execution_status"] or ""
    review_status = values["review_status"] or ""

    title = text.splitlines()[0][2:].strip()
    if title != run_id:
        raise BacklogError(
            f"record title/run_id mismatch: {path.relative_to(root).as_posix()}"
        )
    if not TIMESTAMP_RE.fullmatch(started_at_utc):
        raise BacklogError(
            f"record has invalid started_at_utc timestamp: {path.relative_to(root).as_posix()}"
        )
    if execution_status not in EXECUTION_STATUSES:
        raise BacklogError(
            f"record has invalid execution_status: {path.relative_to(root).as_posix()}"
        )
    if review_status not in REVIEW_STATUSES:
        raise BacklogError(
            f"record has invalid review_status: {path.relative_to(root).as_posix()}"
        )

    prompt_path = Path(prompt_file)
    if prompt_path.stem != prompt_stem:
        raise BacklogError(
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
        raise BacklogError(f"missing notes directory: {notes_dir}")

    records: list[RunRecord] = []
    for path in sorted(notes_dir.glob("*.md")):
        record = parse_record_file(root, path)
        if record is not None:
            records.append(record)
    return records


def latest_records_by_prompt(records: list[RunRecord]) -> list[RunRecord]:
    latest_by_prompt: dict[str, RunRecord] = {}
    for record in records:
        current = latest_by_prompt.get(record.prompt_file)
        if current is None or (record.started_at_utc, record.run_suffix) > (
            current.started_at_utc,
            current.run_suffix,
        ):
            latest_by_prompt[record.prompt_file] = record

    return sorted(latest_by_prompt.values(), key=lambda record: record.prompt_stem)


def render_record(record: RunRecord) -> str:
    return (
        f"- {record.path.as_posix()} | prompt={record.prompt_file} | started={record.started_at_utc} | "
        f"execution={record.execution_status} | review={record.review_status}"
    )


def print_section(title: str, records: list[RunRecord]) -> None:
    print(title)
    if not records:
        print("- none")
        return

    for record in records:
        print(render_record(record))


def main() -> int:
    try:
        args = parse_args()
        root = repo_root()
        records = discover_run_records(root)
        unreviewed = sorted(
            [record for record in records if record.review_status == "UNREVIEWED"],
            key=lambda record: (record.started_at_utc, record.run_suffix, record.prompt_stem),
        )
        latest = latest_records_by_prompt(records)
        needs_review_next = [record for record in latest if record.review_status == "UNREVIEWED"]

        latest_view = latest if not args.unreviewed_only else needs_review_next

        print(f"V1 review backlog summary from {NOTES_DIR}/")
        print(f"Discovered V1 execution records: {len(records)}")
        print(f"Unreviewed records: {len(unreviewed)}")
        print(f"Prompts with latest record: {len(latest)}")
        print()

        print_section("UNREVIEWED records:", unreviewed)
        print()
        print_section("Latest record per prompt:", latest_view)
        print()
        print_section("Likely needs human review next:", needs_review_next)
        return 0
    except BacklogError as exc:
        return fail(str(exc))

exec
/bin/bash -lc "sed -n '1,260p' notes/018_architecture_vs_actual_sweep__20260416_005130.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 018 Architecture Vs Actual Sweep

## Short Summary

The V1 prompt workflow is now real, not just specified. The repo has a working runner at `tools/codex/run_prompt.py`, an in-place review write-back helper at `tools/codex/review_run.py`, a queue-readiness checker at `tools/codex/check_queue_readiness.py`, and a review-backlog lister at `tools/codex/list_review_backlog.py`.

The main remaining work is no longer core architecture implementation. It is cleanup around doc/spec drift, lightweight contract protection for the markdown record format, and a small amount of operational tightening before heavier repeated use.

## Implemented Vs Intended

### Implemented and operational

- One bounded prompt run can be executed through `tools/codex/run_prompt.py`.
- One durable V1 execution record is written into `notes/` with execution facts, review facts, failure-analysis placeholders, resource facts, prompt text, final output, and stderr.
- Execution status and review status are cleanly separated. New records start at `review_status: UNREVIEWED`.
- Manual review write-back exists through `tools/codex/review_run.py` and updates the same record in place.
- Conservative queue progression is operational through `tools/codex/check_queue_readiness.py`.
  - Only `ACCEPTED` on the immediately previous prompt releases the next prompt.
- Review discovery is operational through `tools/codex/list_review_backlog.py`.
  - The repo currently shows 9 V1 execution records, 8 unreviewed records, and latest unreviewed records for prompts `011` through `017`.

### Partially implemented or narrower than the docs imply

- Retry linkage exists only as a placeholder field. `retry_of_run_id` is present in the artifact but there is no thin helper or established operating path for using it consistently.
- Validation exists as ad hoc note-based evidence, not as a durable contract test layer. The system currently depends on markdown shape stability across four scripts.
- The architecture packet describes the core V1 shape well, but it does not reflect the current operational helper set as clearly as the repo now warrants.

### Stale or lagging parts of the design packet

- `tools/codex/V1_Bridge_Runner_Change_Spec.md` is the clearest stale document.
  - It still targets `tools/codex/baby_run_prompt.py`.
  - It still names the runner field as `tools/codex/baby_run_prompt.py`.
  - It still frames review write-back and queue logic as not yet implemented, while the repo now has separate helpers for both.
- The execution-record spec says the stable V1 `run_id` is `<prompt_stem>__<started_at_utc>`, but `tools/codex/run_prompt.py` now adds same-second suffixes like `__2` when needed. The code is pragmatic; the spec lags.
- `tools/Codex_Prompt_Workflow_Architecture.md` is broadly right, but it now underspecifies actual repo reality. The practical V1 toolset is no longer just runner plus manual review conceptually; it includes explicit readiness and backlog helpers.

## Remaining Work

### 1. Align the design packet to the actual V1 toolset

Why it matters:
The repo has outgrown parts of the design packet. The next prompt or reviewer could still be steered by stale references to `baby_run_prompt.py`, by the old strict run-id rule, or by the older assumption that readiness/backlog support does not exist.

Evidence:
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- actual scripts in `tools/codex/`

### 2. Add lightweight contract checks for the V1 markdown record

Why it matters:
`run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py` all rely on the same markdown field and section conventions. That contract is now central to the workflow, but it is only protected by careful coding and prior notes, not by a small repeatable test layer.

Evidence:
- repeated required-section and required-field parsing logic in `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py`
- current validation is documented in notes rather than locked in as an executable check

### 3. Tighten operational guidance around legacy notes and the existing review backlog

Why it matters:
The tooling is correct, but current repo state is mixed:
- legacy `__SUCCESS__` notes still exist for older prompts
- latest V1 reviews for `011` through `017` are still `UNREVIEWED`

That is not a V1 architecture failure, but it does create friction when interpreting readiness and when deciding whether to trust the queue as a day-to-day operational surface.

Evidence:
- `tools/codex/check_queue_readiness.py` needed a special explanatory `Queue note:`
- `tools/codex/list_review_backlog.py` currently reports 8 unreviewed records

### 4. Defer retry-linkage tooling and richer queue semantics

Why it matters:
These are plausible future improvements, but they are not required to use the current V1 slice for real work. Building them now would expand the system before the present workflow has been used enough to justify it.

Evidence:
- `retry_of_run_id` remains unused in normal flow
- current V1 already covers run, record, review, backlog, and readiness

## Prioritized Ranking

1. `Doc/spec alignment cleanup`
Why: removes the most immediate source of incorrect future work and makes the architecture packet trustworthy again.
Expected risk: `low`
Expected payoff: `high`
Recommended timing: `next`

2. `Record-contract validation`
Why: protects the central markdown artifact contract shared by all current V1 tools.
Expected risk: `low`
Expected payoff: `high`
Recommended timing: `soon`

3. `Operational guidance cleanup for legacy notes and open review backlog`
Why: reduces day-to-day confusion without adding new platform surface.
Expected risk: `low`
Expected payoff: `medium`
Recommended timing: `soon`

4. `Retry-linkage helper or richer queue semantics`
Why: useful later, but not justified before the current V1 flow sees more real use.
Expected risk: `medium`
Expected payoff: `medium`
Recommended timing: `explicitly defer`

## Judgment

Yes. The V1 prompt workflow system is good enough to use for real work now, provided it is used in the intended conservative way:

- run one prompt
- inspect the record
- write back manual review
- use readiness/backlog helpers to decide what is next

The smallest remaining thing to clean up before using it heavily is doc/spec alignment, especially:

- update the bridge-runner spec to reflect `run_prompt.py`
- reflect that review write-back, readiness, and backlog helpers now exist
- align the run-id collision rule with actual behavior

That is the shortest path to making the current V1 implementation and the architecture packet agree on what the system actually is.

exec
/bin/bash -lc "sed -n '1,260p' notes/018_prioritized_remaining_work__20260416_005130.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 018 Prioritized Remaining Work

Top remaining items:

1. `Doc/spec alignment cleanup`
- Do this next.
- The repo now has `run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py`, but parts of the design packet still describe the older bridge-runner state and strict unsuffixed run ids.

2. `Lightweight record-contract validation`
- Do this soon.
- The V1 workflow now depends on one markdown record shape shared across four scripts. A small repeatable check is the main technical safeguard still missing.

3. `Operational guidance cleanup for legacy notes and the current unreviewed backlog`
- Do this soon.
- The system works, but mixed legacy notes plus unreviewed latest records still make regular use noisier than it needs to be.

What should explicitly wait:

- retry-linkage tooling
- richer queue states or scheduling
- broader runner refactors
- any platform-style expansion beyond the current V1 slice

codex
I have the current code and packet context. Next I’m scanning the inspected docs for stale runner/helper/run-id wording so I only touch the files that are actually drifting.
exec
/bin/bash -lc 'ls notes | sort | tail -n 5' in /data/git/handson-ml3
 succeeded in 0ms:
018_architecture_vs_actual_sweep__20260416_005130.md
018_prioritized_remaining_work__20260416_005130.md
018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work__20260416_005034.md
019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209.md
019_next_two_cleanup_prompts_plan__20260416_010251.md

exec
/bin/bash -lc "rg -n \"baby_run_prompt\\.py|run_id|review write-back|readiness|backlog|future work|unimplemented|UNREVIEWED|run_prompt\\.py\" tools/Project_Design_Workflow.md tools/Codex_Prompt_Workflow_Architecture.md tools/codex/V1_Execution_Record_Artifact.md tools/codex/V1_Run_Review_Gate.md tools/codex/V1_Bridge_Runner_Change_Spec.md" in /data/git/handson-ml3
 succeeded in 0ms:
tools/codex/V1_Run_Review_Gate.md:38:- `review_status` still set to `UNREVIEWED`
tools/codex/V1_Run_Review_Gate.md:43:Record creation plus `UNREVIEWED` review status means the run must stop for human inspection.
tools/codex/V1_Run_Review_Gate.md:68:- `UNREVIEWED`
tools/codex/V1_Run_Review_Gate.md:74:- `UNREVIEWED`: default state after execution record creation; queue must stop here.
tools/codex/V1_Run_Review_Gate.md:86:- `UNREVIEWED` stops the queue pending manual review.
tools/codex/V1_Run_Review_Gate.md:137:It is the manual transition of one execution record from `UNREVIEWED` to either `ACCEPTED` or `REJECTED`.
tools/codex/V1_Run_Review_Gate.md:156:2. a run record is written with `review_status: UNREVIEWED`
tools/codex/V1_Execution_Record_Artifact.md:37:Each run must have a stable `run_id` that does not change after the file is created.
tools/codex/V1_Execution_Record_Artifact.md:62:- `run_id`
tools/codex/V1_Execution_Record_Artifact.md:83:- `retry_of_run_id`
tools/codex/V1_Execution_Record_Artifact.md:111:- `run_id`
tools/codex/V1_Execution_Record_Artifact.md:125:- `retry_of_run_id` only if the runner was explicitly given that context
tools/codex/V1_Execution_Record_Artifact.md:166:- `UNREVIEWED`
tools/codex/V1_Execution_Record_Artifact.md:172:- a new run record should start as `UNREVIEWED`
tools/codex/V1_Bridge_Runner_Change_Spec.md:5:Define the smallest change set for [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py) so the current bridge runner writes a V1 execution record in `notes/` without a broader refactor.
tools/codex/V1_Bridge_Runner_Change_Spec.md:17:This spec covers only the current bridge runner at [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py).
tools/codex/V1_Bridge_Runner_Change_Spec.md:80:- `review_status: UNREVIEWED`
tools/codex/V1_Bridge_Runner_Change_Spec.md:100:It must not affect `review_status` beyond leaving it at `UNREVIEWED`.
tools/codex/V1_Bridge_Runner_Change_Spec.md:125:- queue progression logic beyond initializing `review_status: UNREVIEWED`
tools/codex/V1_Bridge_Runner_Change_Spec.md:126:- automatic review write-back
tools/codex/V1_Bridge_Runner_Change_Spec.md:141:# <run_id>
tools/codex/V1_Bridge_Runner_Change_Spec.md:143:- run_id: `<run_id>`
tools/codex/V1_Bridge_Runner_Change_Spec.md:156:- runner: `tools/codex/baby_run_prompt.py`
tools/codex/V1_Bridge_Runner_Change_Spec.md:158:- retry_of_run_id: ``
tools/codex/V1_Bridge_Runner_Change_Spec.md:166:- review_status: `UNREVIEWED`
tools/codex/V1_Bridge_Runner_Change_Spec.md:238:- `run_id` -> `f"{prompt_path.stem}__{started_at_utc}"`
tools/codex/V1_Bridge_Runner_Change_Spec.md:239:- filename -> `notes/{run_id}.md`
tools/codex/V1_Bridge_Runner_Change_Spec.md:252:- `runner` -> literal `tools/codex/baby_run_prompt.py`
tools/codex/V1_Bridge_Runner_Change_Spec.md:261:- `review_status` -> literal `UNREVIEWED`
tools/codex/V1_Bridge_Runner_Change_Spec.md:268:- `retry_of_run_id` -> blank in this pass
tools/codex/V1_Bridge_Runner_Change_Spec.md:299:- `review_status: UNREVIEWED`
tools/codex/V1_Bridge_Runner_Change_Spec.md:338:- review write-back later
tools/Codex_Prompt_Workflow_Architecture.md:78:* stopping the queue at `UNREVIEWED`
tools/Codex_Prompt_Workflow_Architecture.md:164:2. write one execution record with `review_status: UNREVIEWED`
tools/Codex_Prompt_Workflow_Architecture.md:218:* new records begin as `UNREVIEWED`
tools/Codex_Prompt_Workflow_Architecture.md:226:* `UNREVIEWED`
tools/Codex_Prompt_Workflow_Architecture.md:386:* confirm a new record stops at `UNREVIEWED`

exec
/bin/bash -lc "sed -n '150,280p' tools/Codex_Prompt_Workflow_Architecture.md" in /data/git/handson-ml3
 succeeded in 0ms:

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
* establish a stable prompt identity from filename/stem

### 5. Note discovery

* scan `notes/`
* find note files that match the naming convention
* ignore unrelated markdown files

### 6. Note parsing

* extract prompt name, success/fail state, and timestamp from note filenames
* turn note filenames into structured metadata

### 7. Status reconstruction

* map prompts to matching notes
* determine whether each prompt is:
  * UNRUN
  * SUCCESS
  * FAIL

exec
/bin/bash -lc "sed -n '280,460p' tools/Codex_Prompt_Workflow_Architecture.md" in /data/git/handson-ml3
 succeeded in 0ms:
  * FAIL
* pick the latest note when there are multiple notes for one prompt

### 8. Prompt selection

* find the next unrun prompt
* find the first failed prompt
* find a prompt by numeric prefix, base name, or full filename

### 9. Prompt display

* print a selected prompt to the terminal

### 10. Retry context assembly

* find the latest failed note for a prompt
* display the original prompt plus the latest failed note
* present instructions for retrying

### 11. Note writing

* create timestamped note filenames
* write notes in markdown
* optionally include the previous note as context

### 12. Manual status marking

* let the user record a success or failure
* attach summary/details text to the note

### 13. Command-line interface

* parse subcommands like `list`, `next`, `show`, `mark`, `retry-failed`
* route commands to the right behaviors

### 14. Terminal presentation

* format output for human readability
* present status tables and retry blocks

This decomposition is still useful, but V1 should not treat every decomposed responsibility as something that must be built now.

---

## V1 Architectural Shape

The conservative V1 shape is smaller than a full prompt-workflow toolkit.

It needs only these practical responsibilities:

### 1. Prompt resolution

Enough logic to identify the intended prompt file safely and reproducibly.

### 2. Bounded execution

Enough runner behavior to execute one prompt and capture the immediate execution facts.

### 3. Execution-record writing

Enough note policy to write one execution record in the V1 format defined by `tools/codex/V1_Execution_Record_Artifact.md`.

### 4. Review write-back

Enough manual workflow support to update that same record with the review facts defined by `tools/codex/V1_Run_Review_Gate.md`.

### 5. Conservative queue progression

Enough status logic to preserve one rule:

* only an accepted reviewed run releases the next prompt

Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.

---

## Implementation Order

The implementation order should follow the governing workflow and the queued design decisions:

1. define the V1 review gate
2. align the architecture document to the workflow, the execution record, and the review gate
3. define the smallest bridge-runner change spec needed to emit the V1 execution record
4. only then implement the bounded runner changes for that slice
5. validate the slice and inspect the resulting run record before any broader refactor

This order is conservative on purpose.

It reduces the risk of:

* hard-coding the wrong note model
* coupling queue progression to process exit
* building convenience layers before the review loop is stable
* refactoring modules before the actual V1 artifact boundary is settled

---

## Validation And Review Posture

Validation in V1 should stay small and inspectable.

The minimum posture is:

* confirm one prompt can be executed intentionally
* confirm one execution record is written in the expected V1 shape
* confirm execution status and review status remain separate
* confirm a new record stops at `UNREVIEWED`
* confirm only `ACCEPTED` would release the next prompt
* confirm `REJECTED` stops progression and preserves enough evidence for the next human decision

This is not a deep automation plan.

It is the minimum validation needed to show that the workflow sequence is being honored:

* execute
* preserve evidence
* review
* decide whether to continue

---

## Explicit Deferred Items

The following remain intentionally deferred until after the V1 slice is proven:

### Better architecture and reuse

* per-repo configuration layers
* reusable templates across many repositories

### Richer metadata and content models

* prompt frontmatter
* sidecar metadata
* machine-readable structured note bodies beyond the V1 markdown record

### Richer status and queue models

* `READY`
* `RUNNING`
* `BLOCKED`
* `PARTIAL`
* `SKIPPED`
* `ARCHIVED`
* dependency-aware release rules

### Retry intelligence

* multi-run retry synthesis
* recurring failure summaries
* automatic retry context assembly beyond minimal linkage

### Broader execution preparation

* richer execution bundles
* automated repo-context assembly
* broader prompt packaging

### Reporting and diagnostics

* aggregated queue reports
* dashboards
* cross-run audit summaries
* separate diagnostic subsystems

### Larger implementation restructuring

* broad module decomposition
* larger CLI surface
* extensive folder expansion

These are extension paths, not V1 requirements.

---

## Extension Path After V1

If V1 proves clean, the next steps should still remain bounded.

A sensible post-V1 path would be:


exec
/bin/bash -lc "sed -n '220,380p' tools/codex/V1_Bridge_Runner_Change_Spec.md" in /data/git/handson-ml3
 succeeded in 0ms:

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

codex
I’ve isolated the drift. The edits will stay narrow: update the bridge spec to describe the real runner and current helper set, then tighten the architecture/artifact wording where run-id and helper behavior lag the code.
exec
/bin/bash -lc 'date -u +%Y%m%d_%H%M%S' in /data/git/handson-ml3
 succeeded in 0ms:
20260416_010534

codex
I’m updating the docs now. I’m keeping the edits constrained to the files that actually misstate the current V1 implementation, plus the required cleanup note.
apply patch
patch: completed
/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md
diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index 6e565317400cbb364e78db8a66ac79150f29140a..a3a3738661f0b6ff88e6fa4d1e58a3689292af61
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -154,6 +154,18 @@
 
 No additional required queue file, database, or sidecar should be introduced in V1.
 
+### 6. Current V1 helper scripts
+
+The current implemented V1 toolset is:
+
+* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
+* `tools/codex/review_run.py` for manual review write-back into the same record
+* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
+* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`
+
+These helpers operate on the same markdown execution-record contract.
+They are part of the current V1 workflow surface, not future placeholders.
+
 ---
 
 ## Minimum Viable Slice
@@ -351,6 +363,13 @@
 
 Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
+The current implementation keeps those responsibilities in a small script set rather than a larger platform layer:
+
+* `run_prompt.py`
+* `review_run.py`
+* `check_queue_readiness.py`
+* `list_review_backlog.py`
+
 ---
 
 ## Implementation Order

diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index 6e565317400cbb364e78db8a66ac79150f29140a..a3a3738661f0b6ff88e6fa4d1e58a3689292af61
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -154,6 +154,18 @@
 
 No additional required queue file, database, or sidecar should be introduced in V1.
 
+### 6. Current V1 helper scripts
+
+The current implemented V1 toolset is:
+
+* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
+* `tools/codex/review_run.py` for manual review write-back into the same record
+* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
+* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`
+
+These helpers operate on the same markdown execution-record contract.
+They are part of the current V1 workflow surface, not future placeholders.
+
 ---
 
 ## Minimum Viable Slice
@@ -351,6 +363,13 @@
 
 Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
+The current implementation keeps those responsibilities in a small script set rather than a larger platform layer:
+
+* `run_prompt.py`
+* `review_run.py`
+* `check_queue_readiness.py`
+* `list_review_backlog.py`
+
 ---
 
 ## Implementation Order

apply patch
patch: completed
/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md
diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index 6e565317400cbb364e78db8a66ac79150f29140a..a3a3738661f0b6ff88e6fa4d1e58a3689292af61
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -154,6 +154,18 @@
 
 No additional required queue file, database, or sidecar should be introduced in V1.
 
+### 6. Current V1 helper scripts
+
+The current implemented V1 toolset is:
+
+* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
+* `tools/codex/review_run.py` for manual review write-back into the same record
+* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
+* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`
+
+These helpers operate on the same markdown execution-record contract.
+They are part of the current V1 workflow surface, not future placeholders.
+
 ---
 
 ## Minimum Viable Slice
@@ -351,6 +363,13 @@
 
 Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
+The current implementation keeps those responsibilities in a small script set rather than a larger platform layer:
+
+* `run_prompt.py`
+* `review_run.py`
+* `check_queue_readiness.py`
+* `list_review_backlog.py`
+
 ---
 
 ## Implementation Order
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
index 2b3a9bd724ef6c395b9312859e80c60691a7a668..3ffac97ced47aef7e50318254c8ca44aec145a96
--- a/tools/codex/V1_Execution_Record_Artifact.md
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -51,6 +51,8 @@
 - it does not depend on later review outcome
 - it can be created cheaply by the bridge runner
 
+`tools/codex/run_prompt.py` uses that base identity and adds a numeric suffix such as `__2` only when needed to avoid a same-second filename collision for the same prompt.
+
 The filename should use the same stable identity and should not encode accepted or rejected review outcome.
 
 ## Required Fields

diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index 6e565317400cbb364e78db8a66ac79150f29140a..a3a3738661f0b6ff88e6fa4d1e58a3689292af61
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -154,6 +154,18 @@
 
 No additional required queue file, database, or sidecar should be introduced in V1.
 
+### 6. Current V1 helper scripts
+
+The current implemented V1 toolset is:
+
+* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
+* `tools/codex/review_run.py` for manual review write-back into the same record
+* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
+* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`
+
+These helpers operate on the same markdown execution-record contract.
+They are part of the current V1 workflow surface, not future placeholders.
+
 ---
 
 ## Minimum Viable Slice
@@ -351,6 +363,13 @@
 
 Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
+The current implementation keeps those responsibilities in a small script set rather than a larger platform layer:
+
+* `run_prompt.py`
+* `review_run.py`
+* `check_queue_readiness.py`
+* `list_review_backlog.py`
+
 ---
 
 ## Implementation Order
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
index 2b3a9bd724ef6c395b9312859e80c60691a7a668..3ffac97ced47aef7e50318254c8ca44aec145a96
--- a/tools/codex/V1_Execution_Record_Artifact.md
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -51,6 +51,8 @@
 - it does not depend on later review outcome
 - it can be created cheaply by the bridge runner
 
+`tools/codex/run_prompt.py` uses that base identity and adds a numeric suffix such as `__2` only when needed to avoid a same-second filename collision for the same prompt.
+
 The filename should use the same stable identity and should not encode accepted or rejected review outcome.
 
 ## Required Fields

2026-04-16T01:06:30.465507Z ERROR codex_core::tools::router: error=apply_patch verification failed: Failed to find expected lines in /data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:
### Run identity mapping

- `run_id` -> `f"{prompt_path.stem}__{started_at_utc}"`
- filename -> `notes/{run_id}.md`
- `prompt_stem` -> `prompt_path.stem`
- `started_at_utc` -> timestamp captured immediately before `run_codex(...)`

This is the smallest precise mapping that preserves current prompt resolution behavior without introducing a new prompt identity system.
diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index 6e565317400cbb364e78db8a66ac79150f29140a..a3a3738661f0b6ff88e6fa4d1e58a3689292af61
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -154,6 +154,18 @@
 
 No additional required queue file, database, or sidecar should be introduced in V1.
 
+### 6. Current V1 helper scripts
+
+The current implemented V1 toolset is:
+
+* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
+* `tools/codex/review_run.py` for manual review write-back into the same record
+* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
+* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`
+
+These helpers operate on the same markdown execution-record contract.
+They are part of the current V1 workflow surface, not future placeholders.
+
 ---
 
 ## Minimum Viable Slice
@@ -351,6 +363,13 @@
 
 Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
+The current implementation keeps those responsibilities in a small script set rather than a larger platform layer:
+
+* `run_prompt.py`
+* `review_run.py`
+* `check_queue_readiness.py`
+* `list_review_backlog.py`
+
 ---
 
 ## Implementation Order
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
index 2b3a9bd724ef6c395b9312859e80c60691a7a668..3ffac97ced47aef7e50318254c8ca44aec145a96
--- a/tools/codex/V1_Execution_Record_Artifact.md
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -51,6 +51,8 @@
 - it does not depend on later review outcome
 - it can be created cheaply by the bridge runner
 
+`tools/codex/run_prompt.py` uses that base identity and adds a numeric suffix such as `__2` only when needed to avoid a same-second filename collision for the same prompt.
+
 The filename should use the same stable identity and should not encode accepted or rejected review outcome.
 
 ## Required Fields

codex
The bridge spec needs a more careful patch because it still carries a pre-implementation structure in several places. I’m re-reading the exact sections and then applying a tighter diff rather than doing a broad rewrite.
exec
/bin/bash -lc "nl -ba tools/codex/V1_Bridge_Runner_Change_Spec.md | sed -n '1,220p'" in /data/git/handson-ml3
 succeeded in 0ms:
     1	# V1 Bridge Runner Change Spec
     2	
     3	## Purpose
     4	
     5	Define the smallest change set for [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py) so the current bridge runner writes a V1 execution record in `notes/` without a broader refactor.
     6	
     7	The target outcome is:
     8	
     9	- execution produces one durable V1 execution record
    10	- execution outcome and review outcome are kept separate
    11	- run identity is stable
    12	- the runner captures only the minimum automatic fields
    13	- human review fields remain manual
    14	
    15	## Scope
    16	
    17	This spec covers only the current bridge runner at [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py).
    18	
    19	It defines:
    20	
    21	- the record filename pattern
    22	- the markdown section order
    23	- the exact fields the runner should populate automatically
    24	- the manual review fields the runner should initialize for later completion
    25	- the minimal mapping from subprocess result to execution status
    26	- the minimal metrics to capture now
    27	
    28	## Current Behavior Summary
    29	
    30	The current runner:
    31	
    32	- resolves one prompt path from an argument using the existing lookup rules
    33	- validates that `codex_prompts/` and `notes/` exist
    34	- reads the prompt text from the selected file
    35	- runs `codex exec -C <repo_root> --output-last-message <tempfile> -`
    36	- captures subprocess return code, last-message output, and stderr
    37	- writes one markdown note into `notes/`
    38	- encodes `SUCCESS` or `FAILED` into the filename and note body based only on return code
    39	- prints the written note path
    40	- exits with the subprocess return code
    41	
    42	The current gap is that the note conflates execution completion with outcome labeling and does not initialize the V1 review gate.
    43	
    44	## Required V1 Changes
    45	
    46	### 1. Replace success/failure note naming with stable run identity naming
    47	
    48	The runner should stop writing filenames in the form:
    49	
    50	`<prompt_stem>__<status>__<timestamp>.md`
    51	
    52	It should instead write:
    53	
    54	`notes/<prompt_stem>__<started_at_utc>.md`
    55	
    56	This filename is the run's stable identity carrier and must not encode:
    57	
    58	- `SUCCESS`
    59	- `FAILED`
    60	- `ACCEPTED`
    61	- `REJECTED`
    62	
    63	### 2. Emit a V1 execution record instead of a success-implies-acceptance note
    64	
    65	The runner should replace the current freeform note body with a V1 execution record shaped around:
    66	
    67	1. Header / identity
    68	2. Execution facts
    69	3. Review facts
    70	4. Failure analysis
    71	5. Resource / cost facts
    72	6. Prompt text
    73	7. Codex final output
    74	8. Stderr or supplemental notes
    75	
    76	### 3. Initialize review fields but leave review work manual
    77	
    78	The runner should create a new record with:
    79	
    80	- `review_status: UNREVIEWED`
    81	- `review_summary:` blank placeholder
    82	
    83	It should not attempt to:
    84	
    85	- infer acceptance
    86	- mark `ACCEPTED`
    87	- mark `REJECTED`
    88	- fill reviewer identity
    89	- fill review timestamp
    90	- classify the failure automatically
    91	
    92	### 4. Derive execution status only from subprocess execution result
    93	
    94	The minimal V1 mapping is:
    95	
    96	- subprocess `return_code == 0` -> `execution_status: EXECUTED`
    97	- subprocess `return_code != 0` -> `execution_status: EXECUTION_FAILED`
    98	
    99	This mapping controls only execution facts.
   100	It must not affect `review_status` beyond leaving it at `UNREVIEWED`.
   101	
   102	### 5. Capture only the minimum automatic metrics now
   103	
   104	The runner should capture and write:
   105	
   106	- `started_at_utc`
   107	- `finished_at_utc`
   108	- `elapsed_seconds`
   109	- `return_code`
   110	- `final_output_char_count`
   111	- `stderr_char_count`
   112	
   113	It should also write:
   114	
   115	- `stderr_text` when present
   116	
   117	No broader token accounting, file-diff analysis, validation harvesting, or queue-state calculation should be added in this pass.
   118	
   119	## Non-Goals
   120	
   121	This spec does not authorize:
   122	
   123	- splitting the runner into multiple modules
   124	- a larger CLI redesign
   125	- queue progression logic beyond initializing `review_status: UNREVIEWED`
   126	- automatic review write-back
   127	- retry orchestration
   128	- dependency-aware scheduling
   129	- broader note/history discovery logic
   130	- structured sidecars, JSON records, or databases
   131	
   132	## Exact Record Shape
   133	
   134	The record should remain plain markdown and use these sections in this order.
   135	
   136	### 1. Header / identity
   137	
   138	Suggested format:
   139	
   140	```md
   141	# <run_id>
   142	
   143	- run_id: `<run_id>`
   144	- prompt_file: `<prompt_file>`
   145	- prompt_stem: `<prompt_stem>`
   146	- started_at_utc: `<started_at_utc>`
   147	```
   148	
   149	### 2. Execution facts
   150	
   151	```md
   152	## Execution Facts
   153	
   154	- execution_status: `<EXECUTED|EXECUTION_FAILED>`
   155	- finished_at_utc: `<finished_at_utc>`
   156	- runner: `tools/codex/baby_run_prompt.py`
   157	- return_code: `<int>`
   158	- retry_of_run_id: ``
   159	```
   160	
   161	### 3. Review facts
   162	
   163	```md
   164	## Review Facts
   165	
   166	- review_status: `UNREVIEWED`
   167	- review_summary:
   168	- reviewed_by:
   169	- reviewed_at_utc:
   170	```
   171	
   172	### 4. Failure analysis
   173	
   174	```md
   175	## Failure Analysis
   176	
   177	- failure_type:
   178	- failure_symptom:
   179	- likely_cause:
   180	- recommended_next_action:
   181	```
   182	
   183	These fields are initialized only.
   184	They remain manual in V1.
   185	
   186	### 5. Resource / cost facts
   187	
   188	```md
   189	## Resource / Cost Facts
   190	
   191	- elapsed_seconds: `<float_or_decimal_seconds>`
   192	- final_output_char_count: `<int>`
   193	- stderr_char_count: `<int>`
   194	```
   195	
   196	### 6. Prompt text
   197	
   198	````md
   199	## Prompt Text
   200	
   201	```md
   202	<full prompt text>
   203	```
   204	````
   205	
   206	### 7. Codex final output
   207	
   208	```md
   209	## Codex Final Output
   210	
   211	<captured last message, or `*No output captured.*`>
   212	```
   213	
   214	### 8. Stderr or supplemental notes
   215	
   216	If stderr exists:
   217	
   218	````md
   219	## Stderr
   220	

exec
/bin/bash -lc "nl -ba tools/codex/V1_Bridge_Runner_Change_Spec.md | sed -n '220,380p'" in /data/git/handson-ml3
 succeeded in 0ms:
   220	
   221	```text
   222	<stderr text>
   223	```
   224	````
   225	
   226	If stderr is empty:
   227	
   228	```md
   229	## Stderr
   230	
   231	*No stderr captured.*
   232	```
   233	
   234	## Exact Data / Field Mapping
   235	
   236	### Run identity mapping
   237	
   238	- `run_id` -> `f"{prompt_path.stem}__{started_at_utc}"`
   239	- filename -> `notes/{run_id}.md`
   240	- `prompt_stem` -> `prompt_path.stem`
   241	- `started_at_utc` -> timestamp captured immediately before `run_codex(...)`
   242	
   243	### Prompt file mapping
   244	
   245	- `prompt_file` -> repo-relative POSIX path when the resolved prompt is inside the repo root
   246	- fallback `prompt_file` -> absolute path string only if the resolved prompt is outside the repo root
   247	
   248	This is the smallest precise mapping that preserves current prompt resolution behavior without introducing a new prompt identity system.
   249	
   250	### Execution mapping
   251	
   252	- `runner` -> literal `tools/codex/baby_run_prompt.py`
   253	- `return_code` -> subprocess return code from `subprocess.run(...)`
   254	- `execution_status` -> `EXECUTED` when return code is `0`, else `EXECUTION_FAILED`
   255	- `finished_at_utc` -> timestamp captured immediately after `run_codex(...)` returns
   256	- `prompt_text` section -> full prompt file contents as currently read
   257	- `codex_final_output` section -> current `--output-last-message` file contents
   258	
   259	### Review mapping
   260	
   261	- `review_status` -> literal `UNREVIEWED`
   262	- `review_summary` -> blank
   263	- `reviewed_by` -> blank
   264	- `reviewed_at_utc` -> blank
   265	
   266	### Failure / retry mapping
   267	
   268	- `retry_of_run_id` -> blank in this pass
   269	- `failure_type` -> blank
   270	- `failure_symptom` -> blank
   271	- `likely_cause` -> blank
   272	- `recommended_next_action` -> blank
   273	
   274	This preserves the manual review boundary from [V1_Run_Review_Gate.md](/data/git/handson-ml3/tools/codex/V1_Run_Review_Gate.md).
   275	
   276	### Metrics mapping
   277	
   278	- `elapsed_seconds` -> `finished_at_utc - started_at_utc` in seconds
   279	- `final_output_char_count` -> `len(codex_output)`
   280	- `stderr_char_count` -> `len(stderr_text)`
   281	- `stderr_text` section -> full stderr text when non-empty
   282	
   283	These are the only required automatic metrics for V1 because they are cheap, inspectable, and already available from the current subprocess flow.
   284	
   285	## Review Fields Left Manual
   286	
   287	The runner should initialize but not populate these beyond blanks:
   288	
   289	- `review_summary`
   290	- `reviewed_by`
   291	- `reviewed_at_utc`
   292	- `failure_type`
   293	- `failure_symptom`
   294	- `likely_cause`
   295	- `recommended_next_action`
   296	
   297	The runner should set only one review field automatically:
   298	
   299	- `review_status: UNREVIEWED`
   300	
   301	## Behaviors That Should Remain Unchanged For V1
   302	
   303	The following current behaviors should stay as they are:
   304	
   305	- repository-root discovery via `Path(__file__).resolve().parents[2]`
   306	- requirement that `codex_prompts/` and `notes/` already exist
   307	- prompt resolution behavior for absolute paths, repo-relative paths, `codex_prompts/` paths, and unique prefix matches
   308	- reading the entire prompt file as UTF-8 text
   309	- invoking `codex exec` through the existing thin subprocess path
   310	- using `--output-last-message` and preserving stderr capture
   311	- writing exactly one markdown artifact per run into `notes/`
   312	- printing the written record path to stdout
   313	- returning the subprocess return code as the script exit code
   314	
   315	These preserved behaviors keep the implementation bridge-sized and avoid a premature runner redesign.
   316	
   317	## Open Questions To Resolve Before Implementation
   318	
   319	### 1. Timestamp format precision
   320	
   321	The current runner uses second precision: `YYYYMMDD_HHMMSS`.
   322	This matches the documented V1 examples and is likely sufficient, but repeated same-second runs of the same prompt would collide.
   323	
   324	Decision needed:
   325	
   326	- keep second precision for strict alignment with the current V1 record doc, or
   327	- add a small collision-avoidance suffix if a file already exists
   328	
   329	The narrower V1 choice is to keep the current format and fail fast on collision unless this becomes a real issue.
   330	
   331	### 2. Blank-value formatting convention
   332	
   333	This spec uses empty field values for manual fields.
   334	Implementation should keep that representation consistent across:
   335	
   336	- empty list-style fields
   337	- empty sections
   338	- review write-back later
   339	
   340	Decision needed:
   341	
   342	- use blank bullet values exactly as shown in this spec
   343	
   344	No other open questions need to block implementation.

diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index 6e565317400cbb364e78db8a66ac79150f29140a..a3a3738661f0b6ff88e6fa4d1e58a3689292af61
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -154,6 +154,18 @@
 
 No additional required queue file, database, or sidecar should be introduced in V1.
 
+### 6. Current V1 helper scripts
+
+The current implemented V1 toolset is:
+
+* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
+* `tools/codex/review_run.py` for manual review write-back into the same record
+* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
+* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`
+
+These helpers operate on the same markdown execution-record contract.
+They are part of the current V1 workflow surface, not future placeholders.
+
 ---
 
 ## Minimum Viable Slice
@@ -351,6 +363,13 @@
 
 Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
+The current implementation keeps those responsibilities in a small script set rather than a larger platform layer:
+
+* `run_prompt.py`
+* `review_run.py`
+* `check_queue_readiness.py`
+* `list_review_backlog.py`
+
 ---
 
 ## Implementation Order
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
index 2b3a9bd724ef6c395b9312859e80c60691a7a668..3ffac97ced47aef7e50318254c8ca44aec145a96
--- a/tools/codex/V1_Execution_Record_Artifact.md
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -51,6 +51,8 @@
 - it does not depend on later review outcome
 - it can be created cheaply by the bridge runner
 
+`tools/codex/run_prompt.py` uses that base identity and adds a numeric suffix such as `__2` only when needed to avoid a same-second filename collision for the same prompt.
+
 The filename should use the same stable identity and should not encode accepted or rejected review outcome.
 
 ## Required Fields

apply patch
patch: completed
/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md
diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index 6e565317400cbb364e78db8a66ac79150f29140a..a3a3738661f0b6ff88e6fa4d1e58a3689292af61
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -154,6 +154,18 @@
 
 No additional required queue file, database, or sidecar should be introduced in V1.
 
+### 6. Current V1 helper scripts
+
+The current implemented V1 toolset is:
+
+* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
+* `tools/codex/review_run.py` for manual review write-back into the same record
+* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
+* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`
+
+These helpers operate on the same markdown execution-record contract.
+They are part of the current V1 workflow surface, not future placeholders.
+
 ---
 
 ## Minimum Viable Slice
@@ -351,6 +363,13 @@
 
 Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
+The current implementation keeps those responsibilities in a small script set rather than a larger platform layer:
+
+* `run_prompt.py`
+* `review_run.py`
+* `check_queue_readiness.py`
+* `list_review_backlog.py`
+
 ---
 
 ## Implementation Order
diff --git a/tools/codex/V1_Bridge_Runner_Change_Spec.md b/tools/codex/V1_Bridge_Runner_Change_Spec.md
index 0df75382d12837baedd42a6f5d6235f4314f149f..104f20bd25c287ba1cf46b0b3ee4051236d532f6
--- a/tools/codex/V1_Bridge_Runner_Change_Spec.md
+++ b/tools/codex/V1_Bridge_Runner_Change_Spec.md
@@ -2,7 +2,7 @@
 
 ## Purpose
 
-Define the smallest change set for [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py) so the current bridge runner writes a V1 execution record in `notes/` without a broader refactor.
+Describe the current V1 bridge runner and adjacent helpers so this spec matches the implemented workflow without a broader refactor.
 
 The target outcome is:
 
@@ -14,7 +14,7 @@
 
 ## Scope
 
-This spec covers only the current bridge runner at [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py).
+This spec covers the current V1 runner at [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py) and the small helper scripts that operate on the same execution-record contract.
 
 It defines:
 
@@ -24,22 +24,32 @@
 - the manual review fields the runner should initialize for later completion
 - the minimal mapping from subprocess result to execution status
 - the minimal metrics to capture now
+- the current helper support around review write-back, queue readiness, and review backlog inspection
 
 ## Current Behavior Summary
 
-The current runner:
+The current runner at `tools/codex/run_prompt.py`:
 
 - resolves one prompt path from an argument using the existing lookup rules
 - validates that `codex_prompts/` and `notes/` exist
 - reads the prompt text from the selected file
 - runs `codex exec -C <repo_root> --output-last-message <tempfile> -`
 - captures subprocess return code, last-message output, and stderr
-- writes one markdown note into `notes/`
-- encodes `SUCCESS` or `FAILED` into the filename and note body based only on return code
+- writes one markdown execution record into `notes/`
+- uses base run identity `<prompt_stem>__<started_at_utc>`
+- adds a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt
+- initializes `review_status: UNREVIEWED`
 - prints the written note path
 - exits with the subprocess return code
 
-The current gap is that the note conflates execution completion with outcome labeling and does not initialize the V1 review gate.
+The current helper set also includes:
+
+- `tools/codex/review_run.py` for manual review write-back into the same record
+- `tools/codex/check_queue_readiness.py` for conservative readiness checks
+- `tools/codex/list_review_backlog.py` for backlog inspection from `notes/`
+
+The current gap is therefore no longer missing helper support.
+The remaining need is only to keep this document aligned to the implemented V1 behavior.
 
 ## Required V1 Changes
 
@@ -53,6 +63,11 @@
 
 `notes/<prompt_stem>__<started_at_utc>.md`
 
+This is the base V1 identity.
+If that path already exists for the same prompt and second, `run_prompt.py` appends a numeric suffix:
+
+`notes/<prompt_stem>__<started_at_utc>__2.md`
+
 This filename is the run's stable identity carrier and must not encode:
 
 - `SUCCESS`
@@ -122,13 +137,15 @@
 
 - splitting the runner into multiple modules
 - a larger CLI redesign
-- queue progression logic beyond initializing `review_status: UNREVIEWED`
-- automatic review write-back
+- new queue states or queue engines beyond the current V1 readiness rule
 - retry orchestration
 - dependency-aware scheduling
 - broader note/history discovery logic
 - structured sidecars, JSON records, or databases
 
+This document is an alignment spec for the current V1 slice.
+It is not a request to redesign the already-implemented helper scripts.
+
 ## Exact Record Shape
 
 The record should remain plain markdown and use these sections in this order.
@@ -153,7 +170,7 @@
 
 - execution_status: `<EXECUTED|EXECUTION_FAILED>`
 - finished_at_utc: `<finished_at_utc>`
-- runner: `tools/codex/baby_run_prompt.py`
+- runner: `tools/codex/run_prompt.py`
 - return_code: `<int>`
 - retry_of_run_id: ``
 ```
@@ -235,7 +252,8 @@
 
 ### Run identity mapping
 
-- `run_id` -> `f"{prompt_path.stem}__{started_at_utc}"`
+- base V1 identity -> `f"{prompt_path.stem}__{started_at_utc}"`
+- `run_id` -> base identity unless a same-second collision exists for the same prompt, then `f"{base_run_id}__{n}"`
 - filename -> `notes/{run_id}.md`
 - `prompt_stem` -> `prompt_path.stem`
 - `started_at_utc` -> timestamp captured immediately before `run_codex(...)`
@@ -249,7 +267,7 @@
 
 ### Execution mapping
 
-- `runner` -> literal `tools/codex/baby_run_prompt.py`
+- `runner` -> literal `tools/codex/run_prompt.py`
 - `return_code` -> subprocess return code from `subprocess.run(...)`
 - `execution_status` -> `EXECUTED` when return code is `0`, else `EXECUTION_FAILED`
 - `finished_at_utc` -> timestamp captured immediately after `run_codex(...)` returns
@@ -314,31 +332,14 @@
 
 These preserved behaviors keep the implementation bridge-sized and avoid a premature runner redesign.
 
-## Open Questions To Resolve Before Implementation
+## Current Helper Support Around The Runner
 
-### 1. Timestamp format precision
+The current V1 workflow around the runner is:
 
-The current runner uses second precision: `YYYYMMDD_HHMMSS`.
-This matches the documented V1 examples and is likely sufficient, but repeated same-second runs of the same prompt would collide.
-
-Decision needed:
+1. `tools/codex/run_prompt.py` writes the execution record with `review_status: UNREVIEWED`
+2. `tools/codex/review_run.py` writes back `ACCEPTED` or `REJECTED` review fields into that same file
+3. `tools/codex/check_queue_readiness.py` evaluates whether the next prompt is ready under the V1 review gate
+4. `tools/codex/list_review_backlog.py` reports the current review backlog from `notes/`
 
-- keep second precision for strict alignment with the current V1 record doc, or
-- add a small collision-avoidance suffix if a file already exists
-
-The narrower V1 choice is to keep the current format and fail fast on collision unless this becomes a real issue.
-
-### 2. Blank-value formatting convention
-
-This spec uses empty field values for manual fields.
-Implementation should keep that representation consistent across:
-
-- empty list-style fields
-- empty sections
-- review write-back later
-
-Decision needed:
-
-- use blank bullet values exactly as shown in this spec
-
-No other open questions need to block implementation.
+These helpers are intentionally small.
+They do not introduce a larger orchestration layer, but they do mean review write-back, readiness checking, and backlog listing are implemented parts of the current V1 slice.
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
index 2b3a9bd724ef6c395b9312859e80c60691a7a668..3ffac97ced47aef7e50318254c8ca44aec145a96
--- a/tools/codex/V1_Execution_Record_Artifact.md
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -51,6 +51,8 @@
 - it does not depend on later review outcome
 - it can be created cheaply by the bridge runner
 
+`tools/codex/run_prompt.py` uses that base identity and adds a numeric suffix such as `__2` only when needed to avoid a same-second filename collision for the same prompt.
+
 The filename should use the same stable identity and should not encode accepted or rejected review outcome.
 
 ## Required Fields

diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index 6e565317400cbb364e78db8a66ac79150f29140a..a3a3738661f0b6ff88e6fa4d1e58a3689292af61
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -154,6 +154,18 @@
 
 No additional required queue file, database, or sidecar should be introduced in V1.
 
+### 6. Current V1 helper scripts
+
+The current implemented V1 toolset is:
+
+* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
+* `tools/codex/review_run.py` for manual review write-back into the same record
+* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
+* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`
+
+These helpers operate on the same markdown execution-record contract.
+They are part of the current V1 workflow surface, not future placeholders.
+
 ---
 
 ## Minimum Viable Slice
@@ -351,6 +363,13 @@
 
 Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
+The current implementation keeps those responsibilities in a small script set rather than a larger platform layer:
+
+* `run_prompt.py`
+* `review_run.py`
+* `check_queue_readiness.py`
+* `list_review_backlog.py`
+
 ---
 
 ## Implementation Order
diff --git a/tools/codex/V1_Bridge_Runner_Change_Spec.md b/tools/codex/V1_Bridge_Runner_Change_Spec.md
index 0df75382d12837baedd42a6f5d6235f4314f149f..104f20bd25c287ba1cf46b0b3ee4051236d532f6
--- a/tools/codex/V1_Bridge_Runner_Change_Spec.md
+++ b/tools/codex/V1_Bridge_Runner_Change_Spec.md
@@ -2,7 +2,7 @@
 
 ## Purpose
 
-Define the smallest change set for [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py) so the current bridge runner writes a V1 execution record in `notes/` without a broader refactor.
+Describe the current V1 bridge runner and adjacent helpers so this spec matches the implemented workflow without a broader refactor.
 
 The target outcome is:
 
@@ -14,7 +14,7 @@
 
 ## Scope
 
-This spec covers only the current bridge runner at [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py).
+This spec covers the current V1 runner at [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py) and the small helper scripts that operate on the same execution-record contract.
 
 It defines:
 
@@ -24,22 +24,32 @@
 - the manual review fields the runner should initialize for later completion
 - the minimal mapping from subprocess result to execution status
 - the minimal metrics to capture now
+- the current helper support around review write-back, queue readiness, and review backlog inspection
 
 ## Current Behavior Summary
 
-The current runner:
+The current runner at `tools/codex/run_prompt.py`:
 
 - resolves one prompt path from an argument using the existing lookup rules
 - validates that `codex_prompts/` and `notes/` exist
 - reads the prompt text from the selected file
 - runs `codex exec -C <repo_root> --output-last-message <tempfile> -`
 - captures subprocess return code, last-message output, and stderr
-- writes one markdown note into `notes/`
-- encodes `SUCCESS` or `FAILED` into the filename and note body based only on return code
+- writes one markdown execution record into `notes/`
+- uses base run identity `<prompt_stem>__<started_at_utc>`
+- adds a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt
+- initializes `review_status: UNREVIEWED`
 - prints the written note path
 - exits with the subprocess return code
 
-The current gap is that the note conflates execution completion with outcome labeling and does not initialize the V1 review gate.
+The current helper set also includes:
+
+- `tools/codex/review_run.py` for manual review write-back into the same record
+- `tools/codex/check_queue_readiness.py` for conservative readiness checks
+- `tools/codex/list_review_backlog.py` for backlog inspection from `notes/`
+
+The current gap is therefore no longer missing helper support.
+The remaining need is only to keep this document aligned to the implemented V1 behavior.
 
 ## Required V1 Changes
 
@@ -53,6 +63,11 @@
 
 `notes/<prompt_stem>__<started_at_utc>.md`
 
+This is the base V1 identity.
+If that path already exists for the same prompt and second, `run_prompt.py` appends a numeric suffix:
+
+`notes/<prompt_stem>__<started_at_utc>__2.md`
+
 This filename is the run's stable identity carrier and must not encode:
 
 - `SUCCESS`
@@ -122,13 +137,15 @@
 
 - splitting the runner into multiple modules
 - a larger CLI redesign
-- queue progression logic beyond initializing `review_status: UNREVIEWED`
-- automatic review write-back
+- new queue states or queue engines beyond the current V1 readiness rule
 - retry orchestration
 - dependency-aware scheduling
 - broader note/history discovery logic
 - structured sidecars, JSON records, or databases
 
+This document is an alignment spec for the current V1 slice.
+It is not a request to redesign the already-implemented helper scripts.
+
 ## Exact Record Shape
 
 The record should remain plain markdown and use these sections in this order.
@@ -153,7 +170,7 @@
 
 - execution_status: `<EXECUTED|EXECUTION_FAILED>`
 - finished_at_utc: `<finished_at_utc>`
-- runner: `tools/codex/baby_run_prompt.py`
+- runner: `tools/codex/run_prompt.py`
 - return_code: `<int>`
 - retry_of_run_id: ``
 ```
@@ -235,7 +252,8 @@
 
 ### Run identity mapping
 
-- `run_id` -> `f"{prompt_path.stem}__{started_at_utc}"`
+- base V1 identity -> `f"{prompt_path.stem}__{started_at_utc}"`
+- `run_id` -> base identity unless a same-second collision exists for the same prompt, then `f"{base_run_id}__{n}"`
 - filename -> `notes/{run_id}.md`
 - `prompt_stem` -> `prompt_path.stem`
 - `started_at_utc` -> timestamp captured immediately before `run_codex(...)`
@@ -249,7 +267,7 @@
 
 ### Execution mapping
 
-- `runner` -> literal `tools/codex/baby_run_prompt.py`
+- `runner` -> literal `tools/codex/run_prompt.py`
 - `return_code` -> subprocess return code from `subprocess.run(...)`
 - `execution_status` -> `EXECUTED` when return code is `0`, else `EXECUTION_FAILED`
 - `finished_at_utc` -> timestamp captured immediately after `run_codex(...)` returns
@@ -314,31 +332,14 @@
 
 These preserved behaviors keep the implementation bridge-sized and avoid a premature runner redesign.
 
-## Open Questions To Resolve Before Implementation
+## Current Helper Support Around The Runner
 
-### 1. Timestamp format precision
+The current V1 workflow around the runner is:
 
-The current runner uses second precision: `YYYYMMDD_HHMMSS`.
-This matches the documented V1 examples and is likely sufficient, but repeated same-second runs of the same prompt would collide.
-
-Decision needed:
+1. `tools/codex/run_prompt.py` writes the execution record with `review_status: UNREVIEWED`
+2. `tools/codex/review_run.py` writes back `ACCEPTED` or `REJECTED` review fields into that same file
+3. `tools/codex/check_queue_readiness.py` evaluates whether the next prompt is ready under the V1 review gate
+4. `tools/codex/list_review_backlog.py` reports the current review backlog from `notes/`
 
-- keep second precision for strict alignment with the current V1 record doc, or
-- add a small collision-avoidance suffix if a file already exists
-
-The narrower V1 choice is to keep the current format and fail fast on collision unless this becomes a real issue.
-
-### 2. Blank-value formatting convention
-
-This spec uses empty field values for manual fields.
-Implementation should keep that representation consistent across:
-
-- empty list-style fields
-- empty sections
-- review write-back later
-
-Decision needed:
-
-- use blank bullet values exactly as shown in this spec
-
-No other open questions need to block implementation.
+These helpers are intentionally small.
+They do not introduce a larger orchestration layer, but they do mean review write-back, readiness checking, and backlog listing are implemented parts of the current V1 slice.
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
index 2b3a9bd724ef6c395b9312859e80c60691a7a668..3ffac97ced47aef7e50318254c8ca44aec145a96
--- a/tools/codex/V1_Execution_Record_Artifact.md
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -51,6 +51,8 @@
 - it does not depend on later review outcome
 - it can be created cheaply by the bridge runner
 
+`tools/codex/run_prompt.py` uses that base identity and adds a numeric suffix such as `__2` only when needed to avoid a same-second filename collision for the same prompt.
+
 The filename should use the same stable identity and should not encode accepted or rejected review outcome.
 
 ## Required Fields

codex
The main spec edits are in place. I’m doing the consistency pass now: runner naming, helper existence, and the small run-id suffix rule across the edited packet before I write the cleanup note.
exec
/bin/bash -lc "rg -n \"baby_run_prompt\\.py|review write-back later|automatic review write-back|do not yet exist|not yet implemented|future work|__2|<prompt_stem>__<started_at_utc>|run_prompt\\.py|check_queue_readiness\\.py|list_review_backlog\\.py|review_run\\.py\" tools/Codex_Prompt_Workflow_Architecture.md tools/codex/V1_Execution_Record_Artifact.md tools/codex/V1_Run_Review_Gate.md tools/codex/V1_Bridge_Runner_Change_Spec.md" in /data/git/handson-ml3
 succeeded in 0ms:
tools/codex/V1_Bridge_Runner_Change_Spec.md:17:This spec covers the current V1 runner at [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py) and the small helper scripts that operate on the same execution-record contract.
tools/codex/V1_Bridge_Runner_Change_Spec.md:31:The current runner at `tools/codex/run_prompt.py`:
tools/codex/V1_Bridge_Runner_Change_Spec.md:39:- uses base run identity `<prompt_stem>__<started_at_utc>`
tools/codex/V1_Bridge_Runner_Change_Spec.md:40:- adds a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt
tools/codex/V1_Bridge_Runner_Change_Spec.md:47:- `tools/codex/review_run.py` for manual review write-back into the same record
tools/codex/V1_Bridge_Runner_Change_Spec.md:48:- `tools/codex/check_queue_readiness.py` for conservative readiness checks
tools/codex/V1_Bridge_Runner_Change_Spec.md:49:- `tools/codex/list_review_backlog.py` for backlog inspection from `notes/`
tools/codex/V1_Bridge_Runner_Change_Spec.md:64:`notes/<prompt_stem>__<started_at_utc>.md`
tools/codex/V1_Bridge_Runner_Change_Spec.md:67:If that path already exists for the same prompt and second, `run_prompt.py` appends a numeric suffix:
tools/codex/V1_Bridge_Runner_Change_Spec.md:69:`notes/<prompt_stem>__<started_at_utc>__2.md`
tools/codex/V1_Bridge_Runner_Change_Spec.md:173:- runner: `tools/codex/run_prompt.py`
tools/codex/V1_Bridge_Runner_Change_Spec.md:270:- `runner` -> literal `tools/codex/run_prompt.py`
tools/codex/V1_Bridge_Runner_Change_Spec.md:339:1. `tools/codex/run_prompt.py` writes the execution record with `review_status: UNREVIEWED`
tools/codex/V1_Bridge_Runner_Change_Spec.md:340:2. `tools/codex/review_run.py` writes back `ACCEPTED` or `REJECTED` review fields into that same file
tools/codex/V1_Bridge_Runner_Change_Spec.md:341:3. `tools/codex/check_queue_readiness.py` evaluates whether the next prompt is ready under the V1 review gate
tools/codex/V1_Bridge_Runner_Change_Spec.md:342:4. `tools/codex/list_review_backlog.py` reports the current review backlog from `notes/`
tools/codex/V1_Execution_Record_Artifact.md:41:`<prompt_stem>__<started_at_utc>`
tools/codex/V1_Execution_Record_Artifact.md:45:`005_define_execution_record_and_generate_next_prompt_queue__20260415_202557`
tools/codex/V1_Execution_Record_Artifact.md:54:`tools/codex/run_prompt.py` uses that base identity and adds a numeric suffix such as `__2` only when needed to avoid a same-second filename collision for the same prompt.
tools/Codex_Prompt_Workflow_Architecture.md:161:* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
tools/Codex_Prompt_Workflow_Architecture.md:162:* `tools/codex/review_run.py` for manual review write-back into the same record
tools/Codex_Prompt_Workflow_Architecture.md:163:* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
tools/Codex_Prompt_Workflow_Architecture.md:164:* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`
tools/Codex_Prompt_Workflow_Architecture.md:368:* `run_prompt.py`
tools/Codex_Prompt_Workflow_Architecture.md:369:* `review_run.py`
tools/Codex_Prompt_Workflow_Architecture.md:370:* `check_queue_readiness.py`
tools/Codex_Prompt_Workflow_Architecture.md:371:* `list_review_backlog.py`

exec
/bin/bash -lc 'git diff -- tools/Codex_Prompt_Workflow_Architecture.md tools/codex/V1_Execution_Record_Artifact.md tools/codex/V1_Bridge_Runner_Change_Spec.md' in /data/git/handson-ml3
 succeeded in 0ms:
diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index 6e56531..a3a3738 100644
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -154,6 +154,18 @@ That record is the minimum durable run artifact.
 
 No additional required queue file, database, or sidecar should be introduced in V1.
 
+### 6. Current V1 helper scripts
+
+The current implemented V1 toolset is:
+
+* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
+* `tools/codex/review_run.py` for manual review write-back into the same record
+* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
+* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`
+
+These helpers operate on the same markdown execution-record contract.
+They are part of the current V1 workflow surface, not future placeholders.
+
 ---
 
 ## Minimum Viable Slice
@@ -351,6 +363,13 @@ Enough status logic to preserve one rule:
 
 Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
+The current implementation keeps those responsibilities in a small script set rather than a larger platform layer:
+
+* `run_prompt.py`
+* `review_run.py`
+* `check_queue_readiness.py`
+* `list_review_backlog.py`
+
 ---
 
 ## Implementation Order
diff --git a/tools/codex/V1_Bridge_Runner_Change_Spec.md b/tools/codex/V1_Bridge_Runner_Change_Spec.md
index 0df7538..104f20b 100644
--- a/tools/codex/V1_Bridge_Runner_Change_Spec.md
+++ b/tools/codex/V1_Bridge_Runner_Change_Spec.md
@@ -2,7 +2,7 @@
 
 ## Purpose
 
-Define the smallest change set for [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py) so the current bridge runner writes a V1 execution record in `notes/` without a broader refactor.
+Describe the current V1 bridge runner and adjacent helpers so this spec matches the implemented workflow without a broader refactor.
 
 The target outcome is:
 
@@ -14,7 +14,7 @@ The target outcome is:
 
 ## Scope
 
-This spec covers only the current bridge runner at [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py).
+This spec covers the current V1 runner at [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py) and the small helper scripts that operate on the same execution-record contract.
 
 It defines:
 
@@ -24,22 +24,32 @@ It defines:
 - the manual review fields the runner should initialize for later completion
 - the minimal mapping from subprocess result to execution status
 - the minimal metrics to capture now
+- the current helper support around review write-back, queue readiness, and review backlog inspection
 
 ## Current Behavior Summary
 
-The current runner:
+The current runner at `tools/codex/run_prompt.py`:
 
 - resolves one prompt path from an argument using the existing lookup rules
 - validates that `codex_prompts/` and `notes/` exist
 - reads the prompt text from the selected file
 - runs `codex exec -C <repo_root> --output-last-message <tempfile> -`
 - captures subprocess return code, last-message output, and stderr
-- writes one markdown note into `notes/`
-- encodes `SUCCESS` or `FAILED` into the filename and note body based only on return code
+- writes one markdown execution record into `notes/`
+- uses base run identity `<prompt_stem>__<started_at_utc>`
+- adds a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt
+- initializes `review_status: UNREVIEWED`
 - prints the written note path
 - exits with the subprocess return code
 
-The current gap is that the note conflates execution completion with outcome labeling and does not initialize the V1 review gate.
+The current helper set also includes:
+
+- `tools/codex/review_run.py` for manual review write-back into the same record
+- `tools/codex/check_queue_readiness.py` for conservative readiness checks
+- `tools/codex/list_review_backlog.py` for backlog inspection from `notes/`
+
+The current gap is therefore no longer missing helper support.
+The remaining need is only to keep this document aligned to the implemented V1 behavior.
 
 ## Required V1 Changes
 
@@ -53,6 +63,11 @@ It should instead write:
 
 `notes/<prompt_stem>__<started_at_utc>.md`
 
+This is the base V1 identity.
+If that path already exists for the same prompt and second, `run_prompt.py` appends a numeric suffix:
+
+`notes/<prompt_stem>__<started_at_utc>__2.md`
+
 This filename is the run's stable identity carrier and must not encode:
 
 - `SUCCESS`
@@ -122,13 +137,15 @@ This spec does not authorize:
 
 - splitting the runner into multiple modules
 - a larger CLI redesign
-- queue progression logic beyond initializing `review_status: UNREVIEWED`
-- automatic review write-back
+- new queue states or queue engines beyond the current V1 readiness rule
 - retry orchestration
 - dependency-aware scheduling
 - broader note/history discovery logic
 - structured sidecars, JSON records, or databases
 
+This document is an alignment spec for the current V1 slice.
+It is not a request to redesign the already-implemented helper scripts.
+
 ## Exact Record Shape
 
 The record should remain plain markdown and use these sections in this order.
@@ -153,7 +170,7 @@ Suggested format:
 
 - execution_status: `<EXECUTED|EXECUTION_FAILED>`
 - finished_at_utc: `<finished_at_utc>`
-- runner: `tools/codex/baby_run_prompt.py`
+- runner: `tools/codex/run_prompt.py`
 - return_code: `<int>`
 - retry_of_run_id: ``
 ```
@@ -235,7 +252,8 @@ If stderr is empty:
 
 ### Run identity mapping
 
-- `run_id` -> `f"{prompt_path.stem}__{started_at_utc}"`
+- base V1 identity -> `f"{prompt_path.stem}__{started_at_utc}"`
+- `run_id` -> base identity unless a same-second collision exists for the same prompt, then `f"{base_run_id}__{n}"`
 - filename -> `notes/{run_id}.md`
 - `prompt_stem` -> `prompt_path.stem`
 - `started_at_utc` -> timestamp captured immediately before `run_codex(...)`
@@ -249,7 +267,7 @@ This is the smallest precise mapping that preserves current prompt resolution be
 
 ### Execution mapping
 
-- `runner` -> literal `tools/codex/baby_run_prompt.py`
+- `runner` -> literal `tools/codex/run_prompt.py`
 - `return_code` -> subprocess return code from `subprocess.run(...)`
 - `execution_status` -> `EXECUTED` when return code is `0`, else `EXECUTION_FAILED`
 - `finished_at_utc` -> timestamp captured immediately after `run_codex(...)` returns
@@ -314,31 +332,14 @@ The following current behaviors should stay as they are:
 
 These preserved behaviors keep the implementation bridge-sized and avoid a premature runner redesign.
 
-## Open Questions To Resolve Before Implementation
-
-### 1. Timestamp format precision
-
-The current runner uses second precision: `YYYYMMDD_HHMMSS`.
-This matches the documented V1 examples and is likely sufficient, but repeated same-second runs of the same prompt would collide.
-
-Decision needed:
-
-- keep second precision for strict alignment with the current V1 record doc, or
-- add a small collision-avoidance suffix if a file already exists
-
-The narrower V1 choice is to keep the current format and fail fast on collision unless this becomes a real issue.
-
-### 2. Blank-value formatting convention
-
-This spec uses empty field values for manual fields.
-Implementation should keep that representation consistent across:
-
-- empty list-style fields
-- empty sections
-- review write-back later
+## Current Helper Support Around The Runner
 
-Decision needed:
+The current V1 workflow around the runner is:
 
-- use blank bullet values exactly as shown in this spec
+1. `tools/codex/run_prompt.py` writes the execution record with `review_status: UNREVIEWED`
+2. `tools/codex/review_run.py` writes back `ACCEPTED` or `REJECTED` review fields into that same file
+3. `tools/codex/check_queue_readiness.py` evaluates whether the next prompt is ready under the V1 review gate
+4. `tools/codex/list_review_backlog.py` reports the current review backlog from `notes/`
 
-No other open questions need to block implementation.
+These helpers are intentionally small.
+They do not introduce a larger orchestration layer, but they do mean review write-back, readiness checking, and backlog listing are implemented parts of the current V1 slice.
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
index 2b3a9bd..3ffac97 100644
--- a/tools/codex/V1_Execution_Record_Artifact.md
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -51,6 +51,8 @@ Why this is enough for V1:
 - it does not depend on later review outcome
 - it can be created cheaply by the bridge runner
 
+`tools/codex/run_prompt.py` uses that base identity and adds a numeric suffix such as `__2` only when needed to avoid a same-second filename collision for the same prompt.
+
 The filename should use the same stable identity and should not encode accepted or rejected review outcome.
 
 ## Required Fields

diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index 6e565317400cbb364e78db8a66ac79150f29140a..a3a3738661f0b6ff88e6fa4d1e58a3689292af61
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -154,6 +154,18 @@
 
 No additional required queue file, database, or sidecar should be introduced in V1.
 
+### 6. Current V1 helper scripts
+
+The current implemented V1 toolset is:
+
+* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
+* `tools/codex/review_run.py` for manual review write-back into the same record
+* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
+* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`
+
+These helpers operate on the same markdown execution-record contract.
+They are part of the current V1 workflow surface, not future placeholders.
+
 ---
 
 ## Minimum Viable Slice
@@ -351,6 +363,13 @@
 
 Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
+The current implementation keeps those responsibilities in a small script set rather than a larger platform layer:
+
+* `run_prompt.py`
+* `review_run.py`
+* `check_queue_readiness.py`
+* `list_review_backlog.py`
+
 ---
 
 ## Implementation Order
diff --git a/tools/codex/V1_Bridge_Runner_Change_Spec.md b/tools/codex/V1_Bridge_Runner_Change_Spec.md
index 0df75382d12837baedd42a6f5d6235f4314f149f..104f20bd25c287ba1cf46b0b3ee4051236d532f6
--- a/tools/codex/V1_Bridge_Runner_Change_Spec.md
+++ b/tools/codex/V1_Bridge_Runner_Change_Spec.md
@@ -2,7 +2,7 @@
 
 ## Purpose
 
-Define the smallest change set for [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py) so the current bridge runner writes a V1 execution record in `notes/` without a broader refactor.
+Describe the current V1 bridge runner and adjacent helpers so this spec matches the implemented workflow without a broader refactor.
 
 The target outcome is:
 
@@ -14,7 +14,7 @@
 
 ## Scope
 
-This spec covers only the current bridge runner at [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py).
+This spec covers the current V1 runner at [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py) and the small helper scripts that operate on the same execution-record contract.
 
 It defines:
 
@@ -24,22 +24,32 @@
 - the manual review fields the runner should initialize for later completion
 - the minimal mapping from subprocess result to execution status
 - the minimal metrics to capture now
+- the current helper support around review write-back, queue readiness, and review backlog inspection
 
 ## Current Behavior Summary
 
-The current runner:
+The current runner at `tools/codex/run_prompt.py`:
 
 - resolves one prompt path from an argument using the existing lookup rules
 - validates that `codex_prompts/` and `notes/` exist
 - reads the prompt text from the selected file
 - runs `codex exec -C <repo_root> --output-last-message <tempfile> -`
 - captures subprocess return code, last-message output, and stderr
-- writes one markdown note into `notes/`
-- encodes `SUCCESS` or `FAILED` into the filename and note body based only on return code
+- writes one markdown execution record into `notes/`
+- uses base run identity `<prompt_stem>__<started_at_utc>`
+- adds a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt
+- initializes `review_status: UNREVIEWED`
 - prints the written note path
 - exits with the subprocess return code
 
-The current gap is that the note conflates execution completion with outcome labeling and does not initialize the V1 review gate.
+The current helper set also includes:
+
+- `tools/codex/review_run.py` for manual review write-back into the same record
+- `tools/codex/check_queue_readiness.py` for conservative readiness checks
+- `tools/codex/list_review_backlog.py` for backlog inspection from `notes/`
+
+The current gap is therefore no longer missing helper support.
+The remaining need is only to keep this document aligned to the implemented V1 behavior.
 
 ## Required V1 Changes
 
@@ -53,6 +63,11 @@
 
 `notes/<prompt_stem>__<started_at_utc>.md`
 
+This is the base V1 identity.
+If that path already exists for the same prompt and second, `run_prompt.py` appends a numeric suffix:
+
+`notes/<prompt_stem>__<started_at_utc>__2.md`
+
 This filename is the run's stable identity carrier and must not encode:
 
 - `SUCCESS`
@@ -122,13 +137,15 @@
 
 - splitting the runner into multiple modules
 - a larger CLI redesign
-- queue progression logic beyond initializing `review_status: UNREVIEWED`
-- automatic review write-back
+- new queue states or queue engines beyond the current V1 readiness rule
 - retry orchestration
 - dependency-aware scheduling
 - broader note/history discovery logic
 - structured sidecars, JSON records, or databases
 
+This document is an alignment spec for the current V1 slice.
+It is not a request to redesign the already-implemented helper scripts.
+
 ## Exact Record Shape
 
 The record should remain plain markdown and use these sections in this order.
@@ -153,7 +170,7 @@
 
 - execution_status: `<EXECUTED|EXECUTION_FAILED>`
 - finished_at_utc: `<finished_at_utc>`
-- runner: `tools/codex/baby_run_prompt.py`
+- runner: `tools/codex/run_prompt.py`
 - return_code: `<int>`
 - retry_of_run_id: ``
 ```
@@ -235,7 +252,8 @@
 
 ### Run identity mapping
 
-- `run_id` -> `f"{prompt_path.stem}__{started_at_utc}"`
+- base V1 identity -> `f"{prompt_path.stem}__{started_at_utc}"`
+- `run_id` -> base identity unless a same-second collision exists for the same prompt, then `f"{base_run_id}__{n}"`
 - filename -> `notes/{run_id}.md`
 - `prompt_stem` -> `prompt_path.stem`
 - `started_at_utc` -> timestamp captured immediately before `run_codex(...)`
@@ -249,7 +267,7 @@
 
 ### Execution mapping
 
-- `runner` -> literal `tools/codex/baby_run_prompt.py`
+- `runner` -> literal `tools/codex/run_prompt.py`
 - `return_code` -> subprocess return code from `subprocess.run(...)`
 - `execution_status` -> `EXECUTED` when return code is `0`, else `EXECUTION_FAILED`
 - `finished_at_utc` -> timestamp captured immediately after `run_codex(...)` returns
@@ -314,31 +332,14 @@
 
 These preserved behaviors keep the implementation bridge-sized and avoid a premature runner redesign.
 
-## Open Questions To Resolve Before Implementation
+## Current Helper Support Around The Runner
 
-### 1. Timestamp format precision
+The current V1 workflow around the runner is:
 
-The current runner uses second precision: `YYYYMMDD_HHMMSS`.
-This matches the documented V1 examples and is likely sufficient, but repeated same-second runs of the same prompt would collide.
-
-Decision needed:
+1. `tools/codex/run_prompt.py` writes the execution record with `review_status: UNREVIEWED`
+2. `tools/codex/review_run.py` writes back `ACCEPTED` or `REJECTED` review fields into that same file
+3. `tools/codex/check_queue_readiness.py` evaluates whether the next prompt is ready under the V1 review gate
+4. `tools/codex/list_review_backlog.py` reports the current review backlog from `notes/`
 
-- keep second precision for strict alignment with the current V1 record doc, or
-- add a small collision-avoidance suffix if a file already exists
-
-The narrower V1 choice is to keep the current format and fail fast on collision unless this becomes a real issue.
-
-### 2. Blank-value formatting convention
-
-This spec uses empty field values for manual fields.
-Implementation should keep that representation consistent across:
-
-- empty list-style fields
-- empty sections
-- review write-back later
-
-Decision needed:
-
-- use blank bullet values exactly as shown in this spec
-
-No other open questions need to block implementation.
+These helpers are intentionally small.
+They do not introduce a larger orchestration layer, but they do mean review write-back, readiness checking, and backlog listing are implemented parts of the current V1 slice.
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
index 2b3a9bd724ef6c395b9312859e80c60691a7a668..3ffac97ced47aef7e50318254c8ca44aec145a96
--- a/tools/codex/V1_Execution_Record_Artifact.md
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -51,6 +51,8 @@
 - it does not depend on later review outcome
 - it can be created cheaply by the bridge runner
 
+`tools/codex/run_prompt.py` uses that base identity and adds a numeric suffix such as `__2` only when needed to avoid a same-second filename collision for the same prompt.
+
 The filename should use the same stable identity and should not encode accepted or rejected review outcome.
 
 ## Required Fields

apply patch
patch: completed
/data/git/handson-ml3/notes/020_doc_spec_alignment_cleanup__20260416_010534.md
diff --git a/notes/020_doc_spec_alignment_cleanup__20260416_010534.md b/notes/020_doc_spec_alignment_cleanup__20260416_010534.md
new file mode 100644
index 0000000000000000000000000000000000000000..c06a30e8c8da7424145f29cfcb8d11b0bf8200f7
--- /dev/null
+++ b/notes/020_doc_spec_alignment_cleanup__20260416_010534.md
@@ -0,0 +1,46 @@
+# 020 Doc Spec Alignment Cleanup
+
+## Scope
+
+This pass stayed bounded to doc/spec alignment for the current V1 slice.
+
+Updated files:
+
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
+
+No runner or helper code was changed.
+
+## What Was Aligned
+
+### Active V1 runner naming
+
+- Replaced stale bridge-runner references to `tools/codex/baby_run_prompt.py` with `tools/codex/run_prompt.py` in the edited packet.
+- Updated the bridge-runner spec so it reads as an alignment document for the implemented runner instead of a pre-implementation change request for a different file.
+
+### Implemented helper set
+
+- The packet now reflects that the current V1 workflow includes:
+  - `tools/codex/run_prompt.py`
+  - `tools/codex/review_run.py`
+  - `tools/codex/check_queue_readiness.py`
+  - `tools/codex/list_review_backlog.py`
+- Removed stale framing in the bridge spec that treated review write-back or review/backlog helper support as still pending.
+
+### Run-id wording
+
+- Kept the base V1 identity as `<prompt_stem>__<started_at_utc>`.
+- Added the implementation-accurate note that `run_prompt.py` appends a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt.
+
+## Validation
+
+1. Confirmed the edited design/spec files now name `tools/codex/run_prompt.py` as the active V1 runner where applicable.
+2. Confirmed the edited packet no longer implies review write-back, readiness checking, or backlog listing are future work.
+3. Confirmed run-id wording in the edited packet now matches current code behavior: base identity first, collision suffix only when needed.
+4. Ran a search pass across the edited design/spec files and found no remaining `baby_run_prompt.py` references.
+
+## Notes
+
+This was intentionally not a broader architecture rewrite.
+It only brought the current V1 packet back into alignment with the implemented toolset and record behavior.
diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index 6e565317400cbb364e78db8a66ac79150f29140a..a3a3738661f0b6ff88e6fa4d1e58a3689292af61
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -154,6 +154,18 @@
 
 No additional required queue file, database, or sidecar should be introduced in V1.
 
+### 6. Current V1 helper scripts
+
+The current implemented V1 toolset is:
+
+* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
+* `tools/codex/review_run.py` for manual review write-back into the same record
+* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
+* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`
+
+These helpers operate on the same markdown execution-record contract.
+They are part of the current V1 workflow surface, not future placeholders.
+
 ---
 
 ## Minimum Viable Slice
@@ -351,6 +363,13 @@
 
 Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
+The current implementation keeps those responsibilities in a small script set rather than a larger platform layer:
+
+* `run_prompt.py`
+* `review_run.py`
+* `check_queue_readiness.py`
+* `list_review_backlog.py`
+
 ---
 
 ## Implementation Order
diff --git a/tools/codex/V1_Bridge_Runner_Change_Spec.md b/tools/codex/V1_Bridge_Runner_Change_Spec.md
index 0df75382d12837baedd42a6f5d6235f4314f149f..104f20bd25c287ba1cf46b0b3ee4051236d532f6
--- a/tools/codex/V1_Bridge_Runner_Change_Spec.md
+++ b/tools/codex/V1_Bridge_Runner_Change_Spec.md
@@ -2,7 +2,7 @@
 
 ## Purpose
 
-Define the smallest change set for [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py) so the current bridge runner writes a V1 execution record in `notes/` without a broader refactor.
+Describe the current V1 bridge runner and adjacent helpers so this spec matches the implemented workflow without a broader refactor.
 
 The target outcome is:
 
@@ -14,7 +14,7 @@
 
 ## Scope
 
-This spec covers only the current bridge runner at [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py).
+This spec covers the current V1 runner at [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py) and the small helper scripts that operate on the same execution-record contract.
 
 It defines:
 
@@ -24,22 +24,32 @@
 - the manual review fields the runner should initialize for later completion
 - the minimal mapping from subprocess result to execution status
 - the minimal metrics to capture now
+- the current helper support around review write-back, queue readiness, and review backlog inspection
 
 ## Current Behavior Summary
 
-The current runner:
+The current runner at `tools/codex/run_prompt.py`:
 
 - resolves one prompt path from an argument using the existing lookup rules
 - validates that `codex_prompts/` and `notes/` exist
 - reads the prompt text from the selected file
 - runs `codex exec -C <repo_root> --output-last-message <tempfile> -`
 - captures subprocess return code, last-message output, and stderr
-- writes one markdown note into `notes/`
-- encodes `SUCCESS` or `FAILED` into the filename and note body based only on return code
+- writes one markdown execution record into `notes/`
+- uses base run identity `<prompt_stem>__<started_at_utc>`
+- adds a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt
+- initializes `review_status: UNREVIEWED`
 - prints the written note path
 - exits with the subprocess return code
 
-The current gap is that the note conflates execution completion with outcome labeling and does not initialize the V1 review gate.
+The current helper set also includes:
+
+- `tools/codex/review_run.py` for manual review write-back into the same record
+- `tools/codex/check_queue_readiness.py` for conservative readiness checks
+- `tools/codex/list_review_backlog.py` for backlog inspection from `notes/`
+
+The current gap is therefore no longer missing helper support.
+The remaining need is only to keep this document aligned to the implemented V1 behavior.
 
 ## Required V1 Changes
 
@@ -53,6 +63,11 @@
 
 `notes/<prompt_stem>__<started_at_utc>.md`
 
+This is the base V1 identity.
+If that path already exists for the same prompt and second, `run_prompt.py` appends a numeric suffix:
+
+`notes/<prompt_stem>__<started_at_utc>__2.md`
+
 This filename is the run's stable identity carrier and must not encode:
 
 - `SUCCESS`
@@ -122,13 +137,15 @@
 
 - splitting the runner into multiple modules
 - a larger CLI redesign
-- queue progression logic beyond initializing `review_status: UNREVIEWED`
-- automatic review write-back
+- new queue states or queue engines beyond the current V1 readiness rule
 - retry orchestration
 - dependency-aware scheduling
 - broader note/history discovery logic
 - structured sidecars, JSON records, or databases
 
+This document is an alignment spec for the current V1 slice.
+It is not a request to redesign the already-implemented helper scripts.
+
 ## Exact Record Shape
 
 The record should remain plain markdown and use these sections in this order.
@@ -153,7 +170,7 @@
 
 - execution_status: `<EXECUTED|EXECUTION_FAILED>`
 - finished_at_utc: `<finished_at_utc>`
-- runner: `tools/codex/baby_run_prompt.py`
+- runner: `tools/codex/run_prompt.py`
 - return_code: `<int>`
 - retry_of_run_id: ``
 ```
@@ -235,7 +252,8 @@
 
 ### Run identity mapping
 
-- `run_id` -> `f"{prompt_path.stem}__{started_at_utc}"`
+- base V1 identity -> `f"{prompt_path.stem}__{started_at_utc}"`
+- `run_id` -> base identity unless a same-second collision exists for the same prompt, then `f"{base_run_id}__{n}"`
 - filename -> `notes/{run_id}.md`
 - `prompt_stem` -> `prompt_path.stem`
 - `started_at_utc` -> timestamp captured immediately before `run_codex(...)`
@@ -249,7 +267,7 @@
 
 ### Execution mapping
 
-- `runner` -> literal `tools/codex/baby_run_prompt.py`
+- `runner` -> literal `tools/codex/run_prompt.py`
 - `return_code` -> subprocess return code from `subprocess.run(...)`
 - `execution_status` -> `EXECUTED` when return code is `0`, else `EXECUTION_FAILED`
 - `finished_at_utc` -> timestamp captured immediately after `run_codex(...)` returns
@@ -314,31 +332,14 @@
 
 These preserved behaviors keep the implementation bridge-sized and avoid a premature runner redesign.
 
-## Open Questions To Resolve Before Implementation
+## Current Helper Support Around The Runner
 
-### 1. Timestamp format precision
+The current V1 workflow around the runner is:
 
-The current runner uses second precision: `YYYYMMDD_HHMMSS`.
-This matches the documented V1 examples and is likely sufficient, but repeated same-second runs of the same prompt would collide.
-
-Decision needed:
+1. `tools/codex/run_prompt.py` writes the execution record with `review_status: UNREVIEWED`
+2. `tools/codex/review_run.py` writes back `ACCEPTED` or `REJECTED` review fields into that same file
+3. `tools/codex/check_queue_readiness.py` evaluates whether the next prompt is ready under the V1 review gate
+4. `tools/codex/list_review_backlog.py` reports the current review backlog from `notes/`
 
-- keep second precision for strict alignment with the current V1 record doc, or
-- add a small collision-avoidance suffix if a file already exists
-
-The narrower V1 choice is to keep the current format and fail fast on collision unless this becomes a real issue.
-
-### 2. Blank-value formatting convention
-
-This spec uses empty field values for manual fields.
-Implementation should keep that representation consistent across:
-
-- empty list-style fields
-- empty sections
-- review write-back later
-
-Decision needed:
-
-- use blank bullet values exactly as shown in this spec
-
-No other open questions need to block implementation.
+These helpers are intentionally small.
+They do not introduce a larger orchestration layer, but they do mean review write-back, readiness checking, and backlog listing are implemented parts of the current V1 slice.
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
index 2b3a9bd724ef6c395b9312859e80c60691a7a668..3ffac97ced47aef7e50318254c8ca44aec145a96
--- a/tools/codex/V1_Execution_Record_Artifact.md
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -51,6 +51,8 @@
 - it does not depend on later review outcome
 - it can be created cheaply by the bridge runner
 
+`tools/codex/run_prompt.py` uses that base identity and adds a numeric suffix such as `__2` only when needed to avoid a same-second filename collision for the same prompt.
+
 The filename should use the same stable identity and should not encode accepted or rejected review outcome.
 
 ## Required Fields

diff --git a/notes/020_doc_spec_alignment_cleanup__20260416_010534.md b/notes/020_doc_spec_alignment_cleanup__20260416_010534.md
new file mode 100644
index 0000000000000000000000000000000000000000..c06a30e8c8da7424145f29cfcb8d11b0bf8200f7
--- /dev/null
+++ b/notes/020_doc_spec_alignment_cleanup__20260416_010534.md
@@ -0,0 +1,46 @@
+# 020 Doc Spec Alignment Cleanup
+
+## Scope
+
+This pass stayed bounded to doc/spec alignment for the current V1 slice.
+
+Updated files:
+
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
+
+No runner or helper code was changed.
+
+## What Was Aligned
+
+### Active V1 runner naming
+
+- Replaced stale bridge-runner references to `tools/codex/baby_run_prompt.py` with `tools/codex/run_prompt.py` in the edited packet.
+- Updated the bridge-runner spec so it reads as an alignment document for the implemented runner instead of a pre-implementation change request for a different file.
+
+### Implemented helper set
+
+- The packet now reflects that the current V1 workflow includes:
+  - `tools/codex/run_prompt.py`
+  - `tools/codex/review_run.py`
+  - `tools/codex/check_queue_readiness.py`
+  - `tools/codex/list_review_backlog.py`
+- Removed stale framing in the bridge spec that treated review write-back or review/backlog helper support as still pending.
+
+### Run-id wording
+
+- Kept the base V1 identity as `<prompt_stem>__<started_at_utc>`.
+- Added the implementation-accurate note that `run_prompt.py` appends a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt.
+
+## Validation
+
+1. Confirmed the edited design/spec files now name `tools/codex/run_prompt.py` as the active V1 runner where applicable.
+2. Confirmed the edited packet no longer implies review write-back, readiness checking, or backlog listing are future work.
+3. Confirmed run-id wording in the edited packet now matches current code behavior: base identity first, collision suffix only when needed.
+4. Ran a search pass across the edited design/spec files and found no remaining `baby_run_prompt.py` references.
+
+## Notes
+
+This was intentionally not a broader architecture rewrite.
+It only brought the current V1 packet back into alignment with the implemented toolset and record behavior.
diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index 6e565317400cbb364e78db8a66ac79150f29140a..a3a3738661f0b6ff88e6fa4d1e58a3689292af61
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -154,6 +154,18 @@
 
 No additional required queue file, database, or sidecar should be introduced in V1.
 
+### 6. Current V1 helper scripts
+
+The current implemented V1 toolset is:
+
+* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
+* `tools/codex/review_run.py` for manual review write-back into the same record
+* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
+* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`
+
+These helpers operate on the same markdown execution-record contract.
+They are part of the current V1 workflow surface, not future placeholders.
+
 ---
 
 ## Minimum Viable Slice
@@ -351,6 +363,13 @@
 
 Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
+The current implementation keeps those responsibilities in a small script set rather than a larger platform layer:
+
+* `run_prompt.py`
+* `review_run.py`
+* `check_queue_readiness.py`
+* `list_review_backlog.py`
+
 ---
 
 ## Implementation Order
diff --git a/tools/codex/V1_Bridge_Runner_Change_Spec.md b/tools/codex/V1_Bridge_Runner_Change_Spec.md
index 0df75382d12837baedd42a6f5d6235f4314f149f..104f20bd25c287ba1cf46b0b3ee4051236d532f6
--- a/tools/codex/V1_Bridge_Runner_Change_Spec.md
+++ b/tools/codex/V1_Bridge_Runner_Change_Spec.md
@@ -2,7 +2,7 @@
 
 ## Purpose
 
-Define the smallest change set for [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py) so the current bridge runner writes a V1 execution record in `notes/` without a broader refactor.
+Describe the current V1 bridge runner and adjacent helpers so this spec matches the implemented workflow without a broader refactor.
 
 The target outcome is:
 
@@ -14,7 +14,7 @@
 
 ## Scope
 
-This spec covers only the current bridge runner at [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py).
+This spec covers the current V1 runner at [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py) and the small helper scripts that operate on the same execution-record contract.
 
 It defines:
 
@@ -24,22 +24,32 @@
 - the manual review fields the runner should initialize for later completion
 - the minimal mapping from subprocess result to execution status
 - the minimal metrics to capture now
+- the current helper support around review write-back, queue readiness, and review backlog inspection
 
 ## Current Behavior Summary
 
-The current runner:
+The current runner at `tools/codex/run_prompt.py`:
 
 - resolves one prompt path from an argument using the existing lookup rules
 - validates that `codex_prompts/` and `notes/` exist
 - reads the prompt text from the selected file
 - runs `codex exec -C <repo_root> --output-last-message <tempfile> -`
 - captures subprocess return code, last-message output, and stderr
-- writes one markdown note into `notes/`
-- encodes `SUCCESS` or `FAILED` into the filename and note body based only on return code
+- writes one markdown execution record into `notes/`
+- uses base run identity `<prompt_stem>__<started_at_utc>`
+- adds a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt
+- initializes `review_status: UNREVIEWED`
 - prints the written note path
 - exits with the subprocess return code
 
-The current gap is that the note conflates execution completion with outcome labeling and does not initialize the V1 review gate.
+The current helper set also includes:
+
+- `tools/codex/review_run.py` for manual review write-back into the same record
+- `tools/codex/check_queue_readiness.py` for conservative readiness checks
+- `tools/codex/list_review_backlog.py` for backlog inspection from `notes/`
+
+The current gap is therefore no longer missing helper support.
+The remaining need is only to keep this document aligned to the implemented V1 behavior.
 
 ## Required V1 Changes
 
@@ -53,6 +63,11 @@
 
 `notes/<prompt_stem>__<started_at_utc>.md`
 
+This is the base V1 identity.
+If that path already exists for the same prompt and second, `run_prompt.py` appends a numeric suffix:
+
+`notes/<prompt_stem>__<started_at_utc>__2.md`
+
 This filename is the run's stable identity carrier and must not encode:
 
 - `SUCCESS`
@@ -122,13 +137,15 @@
 
 - splitting the runner into multiple modules
 - a larger CLI redesign
-- queue progression logic beyond initializing `review_status: UNREVIEWED`
-- automatic review write-back
+- new queue states or queue engines beyond the current V1 readiness rule
 - retry orchestration
 - dependency-aware scheduling
 - broader note/history discovery logic
 - structured sidecars, JSON records, or databases
 
+This document is an alignment spec for the current V1 slice.
+It is not a request to redesign the already-implemented helper scripts.
+
 ## Exact Record Shape
 
 The record should remain plain markdown and use these sections in this order.
@@ -153,7 +170,7 @@
 
 - execution_status: `<EXECUTED|EXECUTION_FAILED>`
 - finished_at_utc: `<finished_at_utc>`
-- runner: `tools/codex/baby_run_prompt.py`
+- runner: `tools/codex/run_prompt.py`
 - return_code: `<int>`
 - retry_of_run_id: ``
 ```
@@ -235,7 +252,8 @@
 
 ### Run identity mapping
 
-- `run_id` -> `f"{prompt_path.stem}__{started_at_utc}"`
+- base V1 identity -> `f"{prompt_path.stem}__{started_at_utc}"`
+- `run_id` -> base identity unless a same-second collision exists for the same prompt, then `f"{base_run_id}__{n}"`
 - filename -> `notes/{run_id}.md`
 - `prompt_stem` -> `prompt_path.stem`
 - `started_at_utc` -> timestamp captured immediately before `run_codex(...)`
@@ -249,7 +267,7 @@
 
 ### Execution mapping
 
-- `runner` -> literal `tools/codex/baby_run_prompt.py`
+- `runner` -> literal `tools/codex/run_prompt.py`
 - `return_code` -> subprocess return code from `subprocess.run(...)`
 - `execution_status` -> `EXECUTED` when return code is `0`, else `EXECUTION_FAILED`
 - `finished_at_utc` -> timestamp captured immediately after `run_codex(...)` returns
@@ -314,31 +332,14 @@
 
 These preserved behaviors keep the implementation bridge-sized and avoid a premature runner redesign.
 
-## Open Questions To Resolve Before Implementation
+## Current Helper Support Around The Runner
 
-### 1. Timestamp format precision
+The current V1 workflow around the runner is:
 
-The current runner uses second precision: `YYYYMMDD_HHMMSS`.
-This matches the documented V1 examples and is likely sufficient, but repeated same-second runs of the same prompt would collide.
-
-Decision needed:
+1. `tools/codex/run_prompt.py` writes the execution record with `review_status: UNREVIEWED`
+2. `tools/codex/review_run.py` writes back `ACCEPTED` or `REJECTED` review fields into that same file
+3. `tools/codex/check_queue_readiness.py` evaluates whether the next prompt is ready under the V1 review gate
+4. `tools/codex/list_review_backlog.py` reports the current review backlog from `notes/`
 
-- keep second precision for strict alignment with the current V1 record doc, or
-- add a small collision-avoidance suffix if a file already exists
-
-The narrower V1 choice is to keep the current format and fail fast on collision unless this becomes a real issue.
-
-### 2. Blank-value formatting convention
-
-This spec uses empty field values for manual fields.
-Implementation should keep that representation consistent across:
-
-- empty list-style fields
-- empty sections
-- review write-back later
-
-Decision needed:
-
-- use blank bullet values exactly as shown in this spec
-
-No other open questions need to block implementation.
+These helpers are intentionally small.
+They do not introduce a larger orchestration layer, but they do mean review write-back, readiness checking, and backlog listing are implemented parts of the current V1 slice.
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
index 2b3a9bd724ef6c395b9312859e80c60691a7a668..3ffac97ced47aef7e50318254c8ca44aec145a96
--- a/tools/codex/V1_Execution_Record_Artifact.md
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -51,6 +51,8 @@
 - it does not depend on later review outcome
 - it can be created cheaply by the bridge runner
 
+`tools/codex/run_prompt.py` uses that base identity and adds a numeric suffix such as `__2` only when needed to avoid a same-second filename collision for the same prompt.
+
 The filename should use the same stable identity and should not encode accepted or rejected review outcome.
 
 ## Required Fields

codex
The required note is added. I’m running the final verification pass now: stale-reference search, runner/helper consistency, and a quick diff review to keep the cleanup comfortably bounded.
exec
/bin/bash -lc "rg -n \"baby_run_prompt\\.py\" tools/Codex_Prompt_Workflow_Architecture.md tools/codex/V1_Execution_Record_Artifact.md tools/codex/V1_Bridge_Runner_Change_Spec.md notes/020_doc_spec_alignment_cleanup__20260416_010534.md" in /data/git/handson-ml3
 succeeded in 0ms:
notes/020_doc_spec_alignment_cleanup__20260416_010534.md:19:- Replaced stale bridge-runner references to `tools/codex/baby_run_prompt.py` with `tools/codex/run_prompt.py` in the edited packet.
notes/020_doc_spec_alignment_cleanup__20260416_010534.md:41:4. Ran a search pass across the edited design/spec files and found no remaining `baby_run_prompt.py` references.

exec
/bin/bash -lc "rg -n \"run_prompt\\.py|review_run\\.py|check_queue_readiness\\.py|list_review_backlog\\.py|<prompt_stem>__<started_at_utc>|__2|same-second|collision\" tools/Codex_Prompt_Workflow_Architecture.md tools/codex/V1_Execution_Record_Artifact.md tools/codex/V1_Bridge_Runner_Change_Spec.md notes/020_doc_spec_alignment_cleanup__20260416_010534.md" in /data/git/handson-ml3
 succeeded in 0ms:
tools/codex/V1_Bridge_Runner_Change_Spec.md:17:This spec covers the current V1 runner at [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py) and the small helper scripts that operate on the same execution-record contract.
tools/codex/V1_Bridge_Runner_Change_Spec.md:31:The current runner at `tools/codex/run_prompt.py`:
tools/codex/V1_Bridge_Runner_Change_Spec.md:39:- uses base run identity `<prompt_stem>__<started_at_utc>`
tools/codex/V1_Bridge_Runner_Change_Spec.md:40:- adds a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt
tools/codex/V1_Bridge_Runner_Change_Spec.md:47:- `tools/codex/review_run.py` for manual review write-back into the same record
tools/codex/V1_Bridge_Runner_Change_Spec.md:48:- `tools/codex/check_queue_readiness.py` for conservative readiness checks
tools/codex/V1_Bridge_Runner_Change_Spec.md:49:- `tools/codex/list_review_backlog.py` for backlog inspection from `notes/`
tools/codex/V1_Bridge_Runner_Change_Spec.md:64:`notes/<prompt_stem>__<started_at_utc>.md`
tools/codex/V1_Bridge_Runner_Change_Spec.md:67:If that path already exists for the same prompt and second, `run_prompt.py` appends a numeric suffix:
tools/codex/V1_Bridge_Runner_Change_Spec.md:69:`notes/<prompt_stem>__<started_at_utc>__2.md`
tools/codex/V1_Bridge_Runner_Change_Spec.md:173:- runner: `tools/codex/run_prompt.py`
tools/codex/V1_Bridge_Runner_Change_Spec.md:256:- `run_id` -> base identity unless a same-second collision exists for the same prompt, then `f"{base_run_id}__{n}"`
tools/codex/V1_Bridge_Runner_Change_Spec.md:270:- `runner` -> literal `tools/codex/run_prompt.py`
tools/codex/V1_Bridge_Runner_Change_Spec.md:339:1. `tools/codex/run_prompt.py` writes the execution record with `review_status: UNREVIEWED`
tools/codex/V1_Bridge_Runner_Change_Spec.md:340:2. `tools/codex/review_run.py` writes back `ACCEPTED` or `REJECTED` review fields into that same file
tools/codex/V1_Bridge_Runner_Change_Spec.md:341:3. `tools/codex/check_queue_readiness.py` evaluates whether the next prompt is ready under the V1 review gate
tools/codex/V1_Bridge_Runner_Change_Spec.md:342:4. `tools/codex/list_review_backlog.py` reports the current review backlog from `notes/`
notes/020_doc_spec_alignment_cleanup__20260416_010534.md:19:- Replaced stale bridge-runner references to `tools/codex/baby_run_prompt.py` with `tools/codex/run_prompt.py` in the edited packet.
notes/020_doc_spec_alignment_cleanup__20260416_010534.md:25:  - `tools/codex/run_prompt.py`
notes/020_doc_spec_alignment_cleanup__20260416_010534.md:26:  - `tools/codex/review_run.py`
notes/020_doc_spec_alignment_cleanup__20260416_010534.md:27:  - `tools/codex/check_queue_readiness.py`
notes/020_doc_spec_alignment_cleanup__20260416_010534.md:28:  - `tools/codex/list_review_backlog.py`
notes/020_doc_spec_alignment_cleanup__20260416_010534.md:33:- Kept the base V1 identity as `<prompt_stem>__<started_at_utc>`.
notes/020_doc_spec_alignment_cleanup__20260416_010534.md:34:- Added the implementation-accurate note that `run_prompt.py` appends a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt.
notes/020_doc_spec_alignment_cleanup__20260416_010534.md:38:1. Confirmed the edited design/spec files now name `tools/codex/run_prompt.py` as the active V1 runner where applicable.
notes/020_doc_spec_alignment_cleanup__20260416_010534.md:40:3. Confirmed run-id wording in the edited packet now matches current code behavior: base identity first, collision suffix only when needed.
notes/020_doc_spec_alignment_cleanup__20260416_010534.md:41:4. Ran a search pass across the edited design/spec files and found no remaining `baby_run_prompt.py` references.
tools/codex/V1_Execution_Record_Artifact.md:41:`<prompt_stem>__<started_at_utc>`
tools/codex/V1_Execution_Record_Artifact.md:45:`005_define_execution_record_and_generate_next_prompt_queue__20260415_202557`
tools/codex/V1_Execution_Record_Artifact.md:54:`tools/codex/run_prompt.py` uses that base identity and adds a numeric suffix such as `__2` only when needed to avoid a same-second filename collision for the same prompt.
tools/Codex_Prompt_Workflow_Architecture.md:161:* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
tools/Codex_Prompt_Workflow_Architecture.md:162:* `tools/codex/review_run.py` for manual review write-back into the same record
tools/Codex_Prompt_Workflow_Architecture.md:163:* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
tools/Codex_Prompt_Workflow_Architecture.md:164:* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`
tools/Codex_Prompt_Workflow_Architecture.md:368:* `run_prompt.py`
tools/Codex_Prompt_Workflow_Architecture.md:369:* `review_run.py`
tools/Codex_Prompt_Workflow_Architecture.md:370:* `check_queue_readiness.py`
tools/Codex_Prompt_Workflow_Architecture.md:371:* `list_review_backlog.py`

exec
/bin/bash -lc 'git diff --stat -- tools/Codex_Prompt_Workflow_Architecture.md tools/codex/V1_Execution_Record_Artifact.md tools/codex/V1_Bridge_Runner_Change_Spec.md notes/020_doc_spec_alignment_cleanup__20260416_010534.md' in /data/git/handson-ml3
 succeeded in 0ms:
 tools/Codex_Prompt_Workflow_Architecture.md | 19 ++++++++
 tools/codex/V1_Bridge_Runner_Change_Spec.md | 73 +++++++++++++++--------------
 tools/codex/V1_Execution_Record_Artifact.md |  2 +
 3 files changed, 58 insertions(+), 36 deletions(-)

diff --git a/notes/020_doc_spec_alignment_cleanup__20260416_010534.md b/notes/020_doc_spec_alignment_cleanup__20260416_010534.md
new file mode 100644
index 0000000000000000000000000000000000000000..c06a30e8c8da7424145f29cfcb8d11b0bf8200f7
--- /dev/null
+++ b/notes/020_doc_spec_alignment_cleanup__20260416_010534.md
@@ -0,0 +1,46 @@
+# 020 Doc Spec Alignment Cleanup
+
+## Scope
+
+This pass stayed bounded to doc/spec alignment for the current V1 slice.
+
+Updated files:
+
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
+
+No runner or helper code was changed.
+
+## What Was Aligned
+
+### Active V1 runner naming
+
+- Replaced stale bridge-runner references to `tools/codex/baby_run_prompt.py` with `tools/codex/run_prompt.py` in the edited packet.
+- Updated the bridge-runner spec so it reads as an alignment document for the implemented runner instead of a pre-implementation change request for a different file.
+
+### Implemented helper set
+
+- The packet now reflects that the current V1 workflow includes:
+  - `tools/codex/run_prompt.py`
+  - `tools/codex/review_run.py`
+  - `tools/codex/check_queue_readiness.py`
+  - `tools/codex/list_review_backlog.py`
+- Removed stale framing in the bridge spec that treated review write-back or review/backlog helper support as still pending.
+
+### Run-id wording
+
+- Kept the base V1 identity as `<prompt_stem>__<started_at_utc>`.
+- Added the implementation-accurate note that `run_prompt.py` appends a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt.
+
+## Validation
+
+1. Confirmed the edited design/spec files now name `tools/codex/run_prompt.py` as the active V1 runner where applicable.
+2. Confirmed the edited packet no longer implies review write-back, readiness checking, or backlog listing are future work.
+3. Confirmed run-id wording in the edited packet now matches current code behavior: base identity first, collision suffix only when needed.
+4. Ran a search pass across the edited design/spec files and found no remaining `baby_run_prompt.py` references.
+
+## Notes
+
+This was intentionally not a broader architecture rewrite.
+It only brought the current V1 packet back into alignment with the implemented toolset and record behavior.
diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index 6e565317400cbb364e78db8a66ac79150f29140a..a3a3738661f0b6ff88e6fa4d1e58a3689292af61
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -154,6 +154,18 @@
 
 No additional required queue file, database, or sidecar should be introduced in V1.
 
+### 6. Current V1 helper scripts
+
+The current implemented V1 toolset is:
+
+* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
+* `tools/codex/review_run.py` for manual review write-back into the same record
+* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
+* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`
+
+These helpers operate on the same markdown execution-record contract.
+They are part of the current V1 workflow surface, not future placeholders.
+
 ---
 
 ## Minimum Viable Slice
@@ -351,6 +363,13 @@
 
 Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
+The current implementation keeps those responsibilities in a small script set rather than a larger platform layer:
+
+* `run_prompt.py`
+* `review_run.py`
+* `check_queue_readiness.py`
+* `list_review_backlog.py`
+
 ---
 
 ## Implementation Order
diff --git a/tools/codex/V1_Bridge_Runner_Change_Spec.md b/tools/codex/V1_Bridge_Runner_Change_Spec.md
index 0df75382d12837baedd42a6f5d6235f4314f149f..104f20bd25c287ba1cf46b0b3ee4051236d532f6
--- a/tools/codex/V1_Bridge_Runner_Change_Spec.md
+++ b/tools/codex/V1_Bridge_Runner_Change_Spec.md
@@ -2,7 +2,7 @@
 
 ## Purpose
 
-Define the smallest change set for [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py) so the current bridge runner writes a V1 execution record in `notes/` without a broader refactor.
+Describe the current V1 bridge runner and adjacent helpers so this spec matches the implemented workflow without a broader refactor.
 
 The target outcome is:
 
@@ -14,7 +14,7 @@
 
 ## Scope
 
-This spec covers only the current bridge runner at [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py).
+This spec covers the current V1 runner at [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py) and the small helper scripts that operate on the same execution-record contract.
 
 It defines:
 
@@ -24,22 +24,32 @@
 - the manual review fields the runner should initialize for later completion
 - the minimal mapping from subprocess result to execution status
 - the minimal metrics to capture now
+- the current helper support around review write-back, queue readiness, and review backlog inspection
 
 ## Current Behavior Summary
 
-The current runner:
+The current runner at `tools/codex/run_prompt.py`:
 
 - resolves one prompt path from an argument using the existing lookup rules
 - validates that `codex_prompts/` and `notes/` exist
 - reads the prompt text from the selected file
 - runs `codex exec -C <repo_root> --output-last-message <tempfile> -`
 - captures subprocess return code, last-message output, and stderr
-- writes one markdown note into `notes/`
-- encodes `SUCCESS` or `FAILED` into the filename and note body based only on return code
+- writes one markdown execution record into `notes/`
+- uses base run identity `<prompt_stem>__<started_at_utc>`
+- adds a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt
+- initializes `review_status: UNREVIEWED`
 - prints the written note path
 - exits with the subprocess return code
 
-The current gap is that the note conflates execution completion with outcome labeling and does not initialize the V1 review gate.
+The current helper set also includes:
+
+- `tools/codex/review_run.py` for manual review write-back into the same record
+- `tools/codex/check_queue_readiness.py` for conservative readiness checks
+- `tools/codex/list_review_backlog.py` for backlog inspection from `notes/`
+
+The current gap is therefore no longer missing helper support.
+The remaining need is only to keep this document aligned to the implemented V1 behavior.
 
 ## Required V1 Changes
 
@@ -53,6 +63,11 @@
 
 `notes/<prompt_stem>__<started_at_utc>.md`
 
+This is the base V1 identity.
+If that path already exists for the same prompt and second, `run_prompt.py` appends a numeric suffix:
+
+`notes/<prompt_stem>__<started_at_utc>__2.md`
+
 This filename is the run's stable identity carrier and must not encode:
 
 - `SUCCESS`
@@ -122,13 +137,15 @@
 
 - splitting the runner into multiple modules
 - a larger CLI redesign
-- queue progression logic beyond initializing `review_status: UNREVIEWED`
-- automatic review write-back
+- new queue states or queue engines beyond the current V1 readiness rule
 - retry orchestration
 - dependency-aware scheduling
 - broader note/history discovery logic
 - structured sidecars, JSON records, or databases
 
+This document is an alignment spec for the current V1 slice.
+It is not a request to redesign the already-implemented helper scripts.
+
 ## Exact Record Shape
 
 The record should remain plain markdown and use these sections in this order.
@@ -153,7 +170,7 @@
 
 - execution_status: `<EXECUTED|EXECUTION_FAILED>`
 - finished_at_utc: `<finished_at_utc>`
-- runner: `tools/codex/baby_run_prompt.py`
+- runner: `tools/codex/run_prompt.py`
 - return_code: `<int>`
 - retry_of_run_id: ``
 ```
@@ -235,7 +252,8 @@
 
 ### Run identity mapping
 
-- `run_id` -> `f"{prompt_path.stem}__{started_at_utc}"`
+- base V1 identity -> `f"{prompt_path.stem}__{started_at_utc}"`
+- `run_id` -> base identity unless a same-second collision exists for the same prompt, then `f"{base_run_id}__{n}"`
 - filename -> `notes/{run_id}.md`
 - `prompt_stem` -> `prompt_path.stem`
 - `started_at_utc` -> timestamp captured immediately before `run_codex(...)`
@@ -249,7 +267,7 @@
 
 ### Execution mapping
 
-- `runner` -> literal `tools/codex/baby_run_prompt.py`
+- `runner` -> literal `tools/codex/run_prompt.py`
 - `return_code` -> subprocess return code from `subprocess.run(...)`
 - `execution_status` -> `EXECUTED` when return code is `0`, else `EXECUTION_FAILED`
 - `finished_at_utc` -> timestamp captured immediately after `run_codex(...)` returns
@@ -314,31 +332,14 @@
 
 These preserved behaviors keep the implementation bridge-sized and avoid a premature runner redesign.
 
-## Open Questions To Resolve Before Implementation
+## Current Helper Support Around The Runner
 
-### 1. Timestamp format precision
+The current V1 workflow around the runner is:
 
-The current runner uses second precision: `YYYYMMDD_HHMMSS`.
-This matches the documented V1 examples and is likely sufficient, but repeated same-second runs of the same prompt would collide.
-
-Decision needed:
+1. `tools/codex/run_prompt.py` writes the execution record with `review_status: UNREVIEWED`
+2. `tools/codex/review_run.py` writes back `ACCEPTED` or `REJECTED` review fields into that same file
+3. `tools/codex/check_queue_readiness.py` evaluates whether the next prompt is ready under the V1 review gate
+4. `tools/codex/list_review_backlog.py` reports the current review backlog from `notes/`
 
-- keep second precision for strict alignment with the current V1 record doc, or
-- add a small collision-avoidance suffix if a file already exists
-
-The narrower V1 choice is to keep the current format and fail fast on collision unless this becomes a real issue.
-
-### 2. Blank-value formatting convention
-
-This spec uses empty field values for manual fields.
-Implementation should keep that representation consistent across:
-
-- empty list-style fields
-- empty sections
-- review write-back later
-
-Decision needed:
-
-- use blank bullet values exactly as shown in this spec
-
-No other open questions need to block implementation.
+These helpers are intentionally small.
+They do not introduce a larger orchestration layer, but they do mean review write-back, readiness checking, and backlog listing are implemented parts of the current V1 slice.
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
index 2b3a9bd724ef6c395b9312859e80c60691a7a668..3ffac97ced47aef7e50318254c8ca44aec145a96
--- a/tools/codex/V1_Execution_Record_Artifact.md
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -51,6 +51,8 @@
 - it does not depend on later review outcome
 - it can be created cheaply by the bridge runner
 
+`tools/codex/run_prompt.py` uses that base identity and adds a numeric suffix such as `__2` only when needed to avoid a same-second filename collision for the same prompt.
+
 The filename should use the same stable identity and should not encode accepted or rejected review outcome.
 
 ## Required Fields

codex
Aligned the V1 packet to the implemented toolset and added the required cleanup note at [020_doc_spec_alignment_cleanup__20260416_010534.md](/data/git/handson-ml3/notes/020_doc_spec_alignment_cleanup__20260416_010534.md).

I updated [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md), [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md), and [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md). The packet now names `tools/codex/run_prompt.py` as the active V1 runner where applicable, reflects that `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py` are implemented parts of the current workflow, and documents the run-id behavior accurately: base identity `<prompt_stem>__<started_at_utc>`, with a numeric suffix like `__2` only when needed for same-second collisions.

Validation was done with search passes and diff review. There are no remaining stale `baby_run_prompt.py` references in the edited design/spec files; the only remaining mentions are the intentional historical references inside the new cleanup note. No code files were modified.
diff --git a/notes/020_doc_spec_alignment_cleanup__20260416_010534.md b/notes/020_doc_spec_alignment_cleanup__20260416_010534.md
new file mode 100644
index 0000000000000000000000000000000000000000..c06a30e8c8da7424145f29cfcb8d11b0bf8200f7
--- /dev/null
+++ b/notes/020_doc_spec_alignment_cleanup__20260416_010534.md
@@ -0,0 +1,46 @@
+# 020 Doc Spec Alignment Cleanup
+
+## Scope
+
+This pass stayed bounded to doc/spec alignment for the current V1 slice.
+
+Updated files:
+
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
+
+No runner or helper code was changed.
+
+## What Was Aligned
+
+### Active V1 runner naming
+
+- Replaced stale bridge-runner references to `tools/codex/baby_run_prompt.py` with `tools/codex/run_prompt.py` in the edited packet.
+- Updated the bridge-runner spec so it reads as an alignment document for the implemented runner instead of a pre-implementation change request for a different file.
+
+### Implemented helper set
+
+- The packet now reflects that the current V1 workflow includes:
+  - `tools/codex/run_prompt.py`
+  - `tools/codex/review_run.py`
+  - `tools/codex/check_queue_readiness.py`
+  - `tools/codex/list_review_backlog.py`
+- Removed stale framing in the bridge spec that treated review write-back or review/backlog helper support as still pending.
+
+### Run-id wording
+
+- Kept the base V1 identity as `<prompt_stem>__<started_at_utc>`.
+- Added the implementation-accurate note that `run_prompt.py` appends a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt.
+
+## Validation
+
+1. Confirmed the edited design/spec files now name `tools/codex/run_prompt.py` as the active V1 runner where applicable.
+2. Confirmed the edited packet no longer implies review write-back, readiness checking, or backlog listing are future work.
+3. Confirmed run-id wording in the edited packet now matches current code behavior: base identity first, collision suffix only when needed.
+4. Ran a search pass across the edited design/spec files and found no remaining `baby_run_prompt.py` references.
+
+## Notes
+
+This was intentionally not a broader architecture rewrite.
+It only brought the current V1 packet back into alignment with the implemented toolset and record behavior.
diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index 6e565317400cbb364e78db8a66ac79150f29140a..a3a3738661f0b6ff88e6fa4d1e58a3689292af61
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -154,6 +154,18 @@
 
 No additional required queue file, database, or sidecar should be introduced in V1.
 
+### 6. Current V1 helper scripts
+
+The current implemented V1 toolset is:
+
+* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
+* `tools/codex/review_run.py` for manual review write-back into the same record
+* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
+* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`
+
+These helpers operate on the same markdown execution-record contract.
+They are part of the current V1 workflow surface, not future placeholders.
+
 ---
 
 ## Minimum Viable Slice
@@ -351,6 +363,13 @@
 
 Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
+The current implementation keeps those responsibilities in a small script set rather than a larger platform layer:
+
+* `run_prompt.py`
+* `review_run.py`
+* `check_queue_readiness.py`
+* `list_review_backlog.py`
+
 ---
 
 ## Implementation Order
diff --git a/tools/codex/V1_Bridge_Runner_Change_Spec.md b/tools/codex/V1_Bridge_Runner_Change_Spec.md
index 0df75382d12837baedd42a6f5d6235f4314f149f..104f20bd25c287ba1cf46b0b3ee4051236d532f6
--- a/tools/codex/V1_Bridge_Runner_Change_Spec.md
+++ b/tools/codex/V1_Bridge_Runner_Change_Spec.md
@@ -2,7 +2,7 @@
 
 ## Purpose
 
-Define the smallest change set for [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py) so the current bridge runner writes a V1 execution record in `notes/` without a broader refactor.
+Describe the current V1 bridge runner and adjacent helpers so this spec matches the implemented workflow without a broader refactor.
 
 The target outcome is:
 
@@ -14,7 +14,7 @@
 
 ## Scope
 
-This spec covers only the current bridge runner at [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py).
+This spec covers the current V1 runner at [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py) and the small helper scripts that operate on the same execution-record contract.
 
 It defines:
 
@@ -24,22 +24,32 @@
 - the manual review fields the runner should initialize for later completion
 - the minimal mapping from subprocess result to execution status
 - the minimal metrics to capture now
+- the current helper support around review write-back, queue readiness, and review backlog inspection
 
 ## Current Behavior Summary
 
-The current runner:
+The current runner at `tools/codex/run_prompt.py`:
 
 - resolves one prompt path from an argument using the existing lookup rules
 - validates that `codex_prompts/` and `notes/` exist
 - reads the prompt text from the selected file
 - runs `codex exec -C <repo_root> --output-last-message <tempfile> -`
 - captures subprocess return code, last-message output, and stderr
-- writes one markdown note into `notes/`
-- encodes `SUCCESS` or `FAILED` into the filename and note body based only on return code
+- writes one markdown execution record into `notes/`
+- uses base run identity `<prompt_stem>__<started_at_utc>`
+- adds a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt
+- initializes `review_status: UNREVIEWED`
 - prints the written note path
 - exits with the subprocess return code
 
-The current gap is that the note conflates execution completion with outcome labeling and does not initialize the V1 review gate.
+The current helper set also includes:
+
+- `tools/codex/review_run.py` for manual review write-back into the same record
+- `tools/codex/check_queue_readiness.py` for conservative readiness checks
+- `tools/codex/list_review_backlog.py` for backlog inspection from `notes/`
+
+The current gap is therefore no longer missing helper support.
+The remaining need is only to keep this document aligned to the implemented V1 behavior.
 
 ## Required V1 Changes
 
@@ -53,6 +63,11 @@
 
 `notes/<prompt_stem>__<started_at_utc>.md`
 
+This is the base V1 identity.
+If that path already exists for the same prompt and second, `run_prompt.py` appends a numeric suffix:
+
+`notes/<prompt_stem>__<started_at_utc>__2.md`
+
 This filename is the run's stable identity carrier and must not encode:
 
 - `SUCCESS`
@@ -122,13 +137,15 @@
 
 - splitting the runner into multiple modules
 - a larger CLI redesign
-- queue progression logic beyond initializing `review_status: UNREVIEWED`
-- automatic review write-back
+- new queue states or queue engines beyond the current V1 readiness rule
 - retry orchestration
 - dependency-aware scheduling
 - broader note/history discovery logic
 - structured sidecars, JSON records, or databases
 
+This document is an alignment spec for the current V1 slice.
+It is not a request to redesign the already-implemented helper scripts.
+
 ## Exact Record Shape
 
 The record should remain plain markdown and use these sections in this order.
@@ -153,7 +170,7 @@
 
 - execution_status: `<EXECUTED|EXECUTION_FAILED>`
 - finished_at_utc: `<finished_at_utc>`
-- runner: `tools/codex/baby_run_prompt.py`
+- runner: `tools/codex/run_prompt.py`
 - return_code: `<int>`
 - retry_of_run_id: ``
 ```
@@ -235,7 +252,8 @@
 
 ### Run identity mapping
 
-- `run_id` -> `f"{prompt_path.stem}__{started_at_utc}"`
+- base V1 identity -> `f"{prompt_path.stem}__{started_at_utc}"`
+- `run_id` -> base identity unless a same-second collision exists for the same prompt, then `f"{base_run_id}__{n}"`
 - filename -> `notes/{run_id}.md`
 - `prompt_stem` -> `prompt_path.stem`
 - `started_at_utc` -> timestamp captured immediately before `run_codex(...)`
@@ -249,7 +267,7 @@
 
 ### Execution mapping
 
-- `runner` -> literal `tools/codex/baby_run_prompt.py`
+- `runner` -> literal `tools/codex/run_prompt.py`
 - `return_code` -> subprocess return code from `subprocess.run(...)`
 - `execution_status` -> `EXECUTED` when return code is `0`, else `EXECUTION_FAILED`
 - `finished_at_utc` -> timestamp captured immediately after `run_codex(...)` returns
@@ -314,31 +332,14 @@
 
 These preserved behaviors keep the implementation bridge-sized and avoid a premature runner redesign.
 
-## Open Questions To Resolve Before Implementation
+## Current Helper Support Around The Runner
 
-### 1. Timestamp format precision
+The current V1 workflow around the runner is:
 
-The current runner uses second precision: `YYYYMMDD_HHMMSS`.
-This matches the documented V1 examples and is likely sufficient, but repeated same-second runs of the same prompt would collide.
-
-Decision needed:
+1. `tools/codex/run_prompt.py` writes the execution record with `review_status: UNREVIEWED`
+2. `tools/codex/review_run.py` writes back `ACCEPTED` or `REJECTED` review fields into that same file
+3. `tools/codex/check_queue_readiness.py` evaluates whether the next prompt is ready under the V1 review gate
+4. `tools/codex/list_review_backlog.py` reports the current review backlog from `notes/`
 
-- keep second precision for strict alignment with the current V1 record doc, or
-- add a small collision-avoidance suffix if a file already exists
-
-The narrower V1 choice is to keep the current format and fail fast on collision unless this becomes a real issue.
-
-### 2. Blank-value formatting convention
-
-This spec uses empty field values for manual fields.
-Implementation should keep that representation consistent across:
-
-- empty list-style fields
-- empty sections
-- review write-back later
-
-Decision needed:
-
-- use blank bullet values exactly as shown in this spec
-
-No other open questions need to block implementation.
+These helpers are intentionally small.
+They do not introduce a larger orchestration layer, but they do mean review write-back, readiness checking, and backlog listing are implemented parts of the current V1 slice.
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
index 2b3a9bd724ef6c395b9312859e80c60691a7a668..3ffac97ced47aef7e50318254c8ca44aec145a96
--- a/tools/codex/V1_Execution_Record_Artifact.md
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -51,6 +51,8 @@
 - it does not depend on later review outcome
 - it can be created cheaply by the bridge runner
 
+`tools/codex/run_prompt.py` uses that base identity and adds a numeric suffix such as `__2` only when needed to avoid a same-second filename collision for the same prompt.
+
 The filename should use the same stable identity and should not encode accepted or rejected review outcome.
 
 ## Required Fields

tokens used
59,723
```
