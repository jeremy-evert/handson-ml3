# 016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601

- run_id: `016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601`
- prompt_file: `codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md`
- prompt_stem: `016_validate_queue_and_backlog_helpers_against_current_repo`
- started_at_utc: `20260416_003601`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260416_003825`
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

- elapsed_seconds: `143.919`
- final_output_char_count: `1057`
- stderr_char_count: `189025`

## Prompt Text

```md
# Task: Validate the new queue-readiness and review-backlog helpers against the current repo state

You are working in this repository.

Your task is to exercise the two newly built read-only helpers against the current prompt and note history, then write a short validation report.

## Important framing

This is a validation and review task.

Do NOT modify existing code in this pass.
Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

Do NOT rewrite design documents in this pass.

Your job is to run the current helpers, inspect their outputs against the current repo state, and summarize whether they behave correctly and conservatively.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`

Also inspect current repo contents in:

- `codex_prompts/`
- `notes/`

## Goal

Validate whether the two helpers behave correctly against the repo’s current V1 artifacts.

The two helpers are:

- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

## What to do

### 1. Run the queue-readiness helper
Run at least these checks:

- `python3 tools/codex/check_queue_readiness.py`
- `python3 tools/codex/check_queue_readiness.py --prompt 001`
- `python3 tools/codex/check_queue_readiness.py --prompt 002`
- `python3 tools/codex/check_queue_readiness.py --prompt 013`
- `python3 tools/codex/check_queue_readiness.py --prompt 014`
- `python3 tools/codex/check_queue_readiness.py --prompt 015`

### 2. Run the review-backlog helper
Run at least these checks:

- `python3 tools/codex/list_review_backlog.py`
- `python3 tools/codex/list_review_backlog.py --unreviewed-only`

### 3. Compare outputs against actual repo evidence
Check whether the helper outputs agree with:

- actual prompt ordering in `codex_prompts/`
- actual review statuses in the current execution records in `notes/`
- the intended V1 rules in the design documents

## Questions to answer

Please answer these questions in the report:

### A. Queue-readiness correctness
- Does `check_queue_readiness.py` appear to choose the correct target and prior prompt?
- Do its readiness decisions match the current record evidence?
- Are there any edge cases or confusing behaviors visible from the current repo state?

### B. Review-backlog correctness
- Does `list_review_backlog.py` appear to identify `UNREVIEWED` records correctly?
- Does it pick the latest record per prompt correctly?
- Does its “likely needs human review next” view match the repo evidence?

### C. Consistency between the two helpers
- Do the two helpers tell a coherent story about the current repo state?
- Is there any mismatch between what the readiness checker says and what the backlog lister says?

### D. Smallest next improvement
- Based on this validation pass, what is the single smallest next improvement or cleanup step?
- Prefer one bounded suggestion only.

## Required output artifacts

Create exactly two artifacts.

### Artifact 1
Create a validation report at:

`notes/016_queue_and_backlog_helper_validation__TIMESTAMP.md`

This report should include:

- short summary
- queue-readiness validation findings
- review-backlog validation findings
- consistency findings
- one recommended next improvement

### Artifact 2
Create a short recommendation note at:

`notes/016_next_improvement_recommendation__TIMESTAMP.md`

This should contain only:

- the recommended next bounded improvement
- why it should happen next
- what should explicitly wait

## Constraints

1. Use the exact file paths listed above.
2. Do not modify the helpers in this pass.
3. Do not create a broad roadmap.
4. Recommend only one bounded next improvement.
5. Keep the validation practical and evidence-based.

## Success criteria

This task is successful if:

- both helpers are exercised against the current repo state
- the report compares their outputs to actual prompt and note evidence
- any mismatches or edge cases are clearly identified
- one small next improvement is recommended
- the result helps decide the next deliberate step with confidence
```

## Codex Final Output

Created the two requested artifacts:

- [016_queue_and_backlog_helper_validation__20260416_003710.md](/data/git/handson-ml3/notes/016_queue_and_backlog_helper_validation__20260416_003710.md)
- [016_next_improvement_recommendation__20260416_003710.md](/data/git/handson-ml3/notes/016_next_improvement_recommendation__20260416_003710.md)

Validation result: both helpers behave correctly and conservatively against the current repo state. `check_queue_readiness.py` correctly treats `002` as the next ready prompt because only `001` has a latest `ACCEPTED` V1 record, and `list_review_backlog.py` correctly distinguishes the stale unreviewed failed `001` run from the newer accepted `001` run by keeping it in the all-unreviewed list but excluding it from latest-per-prompt and likely-needs-review-next.

The single recommended next step is a small output clarification in `tools/codex/check_queue_readiness.py` so it explains missing V1 history gaps for earlier prompts like `002` through `010`, which currently have legacy notes but no V1 execution records.

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
session id: 019d93b7-5fd2-7240-b9b5-76a7c6043c1f
--------
user
# Task: Validate the new queue-readiness and review-backlog helpers against the current repo state

You are working in this repository.

Your task is to exercise the two newly built read-only helpers against the current prompt and note history, then write a short validation report.

## Important framing

This is a validation and review task.

Do NOT modify existing code in this pass.
Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

Do NOT rewrite design documents in this pass.

Your job is to run the current helpers, inspect their outputs against the current repo state, and summarize whether they behave correctly and conservatively.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`

Also inspect current repo contents in:

- `codex_prompts/`
- `notes/`

## Goal

Validate whether the two helpers behave correctly against the repo’s current V1 artifacts.

The two helpers are:

- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

## What to do

### 1. Run the queue-readiness helper
Run at least these checks:

- `python3 tools/codex/check_queue_readiness.py`
- `python3 tools/codex/check_queue_readiness.py --prompt 001`
- `python3 tools/codex/check_queue_readiness.py --prompt 002`
- `python3 tools/codex/check_queue_readiness.py --prompt 013`
- `python3 tools/codex/check_queue_readiness.py --prompt 014`
- `python3 tools/codex/check_queue_readiness.py --prompt 015`

### 2. Run the review-backlog helper
Run at least these checks:

- `python3 tools/codex/list_review_backlog.py`
- `python3 tools/codex/list_review_backlog.py --unreviewed-only`

### 3. Compare outputs against actual repo evidence
Check whether the helper outputs agree with:

- actual prompt ordering in `codex_prompts/`
- actual review statuses in the current execution records in `notes/`
- the intended V1 rules in the design documents

## Questions to answer

Please answer these questions in the report:

### A. Queue-readiness correctness
- Does `check_queue_readiness.py` appear to choose the correct target and prior prompt?
- Do its readiness decisions match the current record evidence?
- Are there any edge cases or confusing behaviors visible from the current repo state?

### B. Review-backlog correctness
- Does `list_review_backlog.py` appear to identify `UNREVIEWED` records correctly?
- Does it pick the latest record per prompt correctly?
- Does its “likely needs human review next” view match the repo evidence?

### C. Consistency between the two helpers
- Do the two helpers tell a coherent story about the current repo state?
- Is there any mismatch between what the readiness checker says and what the backlog lister says?

### D. Smallest next improvement
- Based on this validation pass, what is the single smallest next improvement or cleanup step?
- Prefer one bounded suggestion only.

## Required output artifacts

Create exactly two artifacts.

### Artifact 1
Create a validation report at:

`notes/016_queue_and_backlog_helper_validation__TIMESTAMP.md`

This report should include:

- short summary
- queue-readiness validation findings
- review-backlog validation findings
- consistency findings
- one recommended next improvement

### Artifact 2
Create a short recommendation note at:

`notes/016_next_improvement_recommendation__TIMESTAMP.md`

This should contain only:

- the recommended next bounded improvement
- why it should happen next
- what should explicitly wait

## Constraints

1. Use the exact file paths listed above.
2. Do not modify the helpers in this pass.
3. Do not create a broad roadmap.
4. Recommend only one bounded next improvement.
5. Keep the validation practical and evidence-based.

## Success criteria

This task is successful if:

- both helpers are exercised against the current repo state
- the report compares their outputs to actual prompt and note evidence
- any mismatches or edge cases are clearly identified
- one small next improvement is recommended
- the result helps decide the next deliberate step with confidence

codex
I’m validating the two read-only helpers against the current prompts and notes, then I’ll write the two requested notes under `notes/` without touching the helper code.
exec
/bin/bash -lc "sed -n '1,260p' tools/Codex_Prompt_Workflow_Architecture.md" in /data/git/handson-ml3
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
/bin/bash -lc "pwd && rg --files codex_prompts notes tools/codex tools | sed 's#"'^#/#'"'" in /data/git/handson-ml3
 succeeded in 0ms:
/data/git/handson-ml3
/tools/Codex_Prompt_Workflow_Architecture.md
/tools/Project_Design_Workflow.md
/tools/codex/V1_Bridge_Runner_Change_Spec.md
/tools/codex/V1_Run_Review_Gate.md
/tools/codex/V1_Execution_Record_Artifact.md
/tools/codex/baby_run_prompt.py
/tools/codex/review_run.py
/tools/codex/run_prompt.py
/tools/codex/check_queue_readiness.py
/tools/codex/list_review_backlog.py
/tools/codex/V1_Bridge_Runner_Change_Spec.md
/tools/codex/V1_Run_Review_Gate.md
/tools/codex/V1_Execution_Record_Artifact.md
/tools/codex/baby_run_prompt.py
/tools/codex/review_run.py
/tools/codex/run_prompt.py
/tools/codex/check_queue_readiness.py
/tools/codex/list_review_backlog.py
/notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md
/notes/009_run_prompt_candidate_build__20260415_233407.md
/notes/001_smoke_test_pipeline__20260415_233343.md
/notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md
/notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md
/notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md
/notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md
/notes/005_prompt_queue_plan__20260415_202557.md
/notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md
/notes/004_next_design_step_recommendation__20260415_195538.md
/notes/004_architecture_and_bridge_runner_review__20260415_195538.md
/notes/003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md
/notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md
/notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md
/notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md
/notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md
/notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md
/notes/012_v1_pipeline_options_review__20260416_000819.md
/notes/011_build_v1_review_writeback_helper__20260415_235346.md
/notes/011_review_writeback_helper_build__20260415_235514.md
/notes/001_smoke_test_pipeline__20260415_234918.md
/notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md
/notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md
/notes/010_next_step_recommendation__20260415_234559.md
/notes/010_run_prompt_candidate_review__20260415_234559.md
/notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md
/notes/013_next_two_prompt_plan__20260416_002005.md
/notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md
/notes/012_top_three_next_options__20260416_000819.md
/notes/014_build_queue_readiness_checker__20260416_002319.md
/notes/014_queue_readiness_checker_build__20260416_002419.md
/notes/015_review_backlog_lister_build__20260416_010500.md
/notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md
/codex_prompts/008_define_minimal_bridge_runner_change_spec.md
/codex_prompts/007_align_architecture_doc_to_v1_workflow.md
/codex_prompts/006_define_v1_run_review_gate.md
/codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md
/codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md
/codex_prompts/003_revise_Project_Deisgn_workflow_document.md
/codex_prompts/002_repo_inventory_and_status.md
/codex_prompts/001_smoke_test_pipeline.md
/codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md
/codex_prompts/011_build_v1_review_writeback_helper.md
/codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md
/codex_prompts/009_build_v1_run_prompt_candidate.md
/codex_prompts/014_build_queue_readiness_checker.md
/codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md
/codex_prompts/015_build_review_backlog_unreviewed_run_lister.md
/codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md

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

codex
I’ve loaded the governing docs. Next I’m reading the helper implementations, the runner/review scripts they depend on, and the current `codex_prompts/` and `notes/` evidence so I can compare tool output to actual state.
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
/bin/bash -lc "printf 'PROMPTS\\n'; ls -1 codex_prompts | sort; printf '\\nNOTES\\n'; ls -1 notes | sort" in /data/git/handson-ml3
 succeeded in 0ms:
PROMPTS
001_smoke_test_pipeline.md
002_repo_inventory_and_status.md
003_revise_Project_Deisgn_workflow_document.md
004_review_architecture_and_bridge_runner_against_workflow.md
005_define_execution_record_and_generate_next_prompt_queue.md
006_define_v1_run_review_gate.md
007_align_architecture_doc_to_v1_workflow.md
008_define_minimal_bridge_runner_change_spec.md
009_build_v1_run_prompt_candidate.md
010_review_run_prompt_candidate_and_recommend_next_step.md
011_build_v1_review_writeback_helper.md
012_review_v1_pipeline_and_recommend_next_options.md
013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md
014_build_queue_readiness_checker.md
015_build_review_backlog_unreviewed_run_lister.md
016_validate_queue_and_backlog_helpers_against_current_repo.md

NOTES
001_smoke_test_pipeline__20260415_233343.md
001_smoke_test_pipeline__20260415_234918.md
001_smoke_test_pipeline__SUCCESS__20260415_133235.md
001_smoke_test_pipeline__SUCCESS__20260415_183223.md
001_smoke_test_pipeline__SUCCESS__20260415_184932.md
002_repo_inventory_and_status__SUCCESS__20260415_133347.md
002_repo_inventory_and_status__SUCCESS__20260415_183259.md
003_project_design_workflow_revision__SUCCESS__20260415_144244.md
003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md
004_architecture_and_bridge_runner_review__20260415_195538.md
004_next_design_step_recommendation__20260415_195538.md
004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md
005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md
005_prompt_queue_plan__20260415_202557.md
006_define_v1_run_review_gate__SUCCESS__20260415_203019.md
007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md
008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md
009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md
009_run_prompt_candidate_build__20260415_233407.md
010_next_step_recommendation__20260415_234559.md
010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md
010_run_prompt_candidate_review__20260415_234559.md
011_build_v1_review_writeback_helper__20260415_235346.md
011_review_writeback_helper_build__20260415_235514.md
012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md
012_top_three_next_options__20260416_000819.md
012_v1_pipeline_options_review__20260416_000819.md
013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md
013_next_two_prompt_plan__20260416_002005.md
014_build_queue_readiness_checker__20260416_002319.md
014_queue_readiness_checker_build__20260416_002419.md
015_build_review_backlog_unreviewed_run_lister__20260416_003109.md
015_review_backlog_lister_build__20260416_010500.md

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

codex
The helper implementations are small enough to inspect directly. I’m pulling the remaining logic plus the current V1 record headers so I can verify target selection and latest-record behavior against the actual notes.
exec
/bin/bash -lc "sed -n '261,420p' tools/codex/check_queue_readiness.py" in /data/git/handson-ml3
 succeeded in 0ms:

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
/bin/bash -lc "for f in notes/*.md; do if rg -q '"'^## Execution Facts$'"' \""'$f" && rg -q '"'"'^- run_id:'"' \""'$f"; then printf '"'\\nFILE %s\\n' \""'$f"; sed -n '"'1,80p' \""'$f"; fi; done' in /data/git/handson-ml3
exec
/bin/bash -lc "for f in codex_prompts/013* codex_prompts/014* codex_prompts/015* codex_prompts/016*; do printf '\\nFILE %s\\n' \""'$f"; sed -n '"'1,220p' \""'$f"; done' in /data/git/handson-ml3
 succeeded in 0ms:

FILE codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md
# Task: Generate the next two implementation prompts for the V1 workflow pipeline

You are working in this repository.

Your task is to write the next two bounded Codex prompt files for the V1 pipeline based on the current repo state and the recent options review.

## Important framing

This is a prompt-generation task only.

Do NOT implement either helper in this pass.
Do NOT modify existing code or design documents in this pass.
Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`

Your job is to generate the next two implementation prompts as files in `codex_prompts/`.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `notes/012_v1_pipeline_options_review__20260416_000819.md`
- `notes/012_top_three_next_options__20260416_000819.md`

## Goal

Generate exactly two prompt files:

1. a prompt to build the recommended **queue-readiness checker**
2. a prompt to build the recommended **review backlog / unreviewed-run lister**

These prompts should be ready for later execution through the current V1 pipeline.

## Prompt 1 requirements

Create a prompt file for the queue-readiness checker.

This future helper should answer a bounded operational question such as:

- given the current prompt set and current records in `notes/`, is the next prompt ready to run?
- what is the latest run for the current or previous prompt?
- is that latest run still `UNREVIEWED`, `ACCEPTED`, or `REJECTED`?
- should the next prompt be treated as ready under the current V1 rules?

The prompt should keep the helper small, inspectable, and consistent with the current architecture and review gate.

## Prompt 2 requirements

Create a prompt file for the review backlog / unreviewed-run lister.

This future helper should answer bounded operational questions such as:

- which execution records are still `UNREVIEWED`?
- what are the latest records per prompt?
- what likely needs human review next?

The prompt should keep the helper small, inspectable, and avoid turning into a broader dashboard or queue engine.

## What each generated prompt must include

Each prompt file must:

- have a clear numeric filename prefix
- define one primary goal only
- name exact files to inspect
- define exact output artifacts to create
- include constraints that prevent broad workflow-engine growth
- define success criteria
- stay small enough to review comfortably before execution

## Also create one planning note

Create a short note at:

`notes/013_next_two_prompt_plan__TIMESTAMP.md`

This note should summarize:

- why these two prompts were chosen
- why they are ordered this way
- which one should be executed first
- what should explicitly wait until later

## Constraints

1. Generate exactly two prompt files, no more.
2. Do not implement the helpers in this pass.
3. Keep both generated prompts bounded and V1-sized.
4. Avoid speculative expansion into dashboards, dependency engines, or broader automation systems.
5. Use the current repo evidence and the options review to shape the prompts.

## Success criteria

This task is successful if:

- exactly two prompt files are created
- one is for the queue-readiness checker
- one is for the review backlog / unreviewed-run lister
- both prompts are implementation-ready and bounded
- the planning note clearly explains the sequence

FILE codex_prompts/014_build_queue_readiness_checker.md
# Task: Build a small V1 queue-readiness checker for the next prompt

You are working in this repository.

Your task is to implement a small helper that answers the bounded V1 operational question:

`Is the next prompt in sequence ready to run under the current review-gate rules?`

## Important framing

This is a small helper task.

Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`

Those existing tools should remain unchanged in this pass.

You are building a thin read-only companion helper beside them, not refactoring the pipeline.

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

You may also inspect the current prompt set and current execution records in:

- `codex_prompts/`
- `notes/`

## Goal

Create a small helper at:

`tools/codex/check_queue_readiness.py`

This helper should determine, from the current prompt files and current V1 execution records in `notes/`, whether the next prompt should be treated as ready under the current V1 review-gate rules.

The helper should stay small, inspectable, and conservative.

## What the helper must answer

The helper should answer these bounded questions:

1. what is the ordered prompt list based on numeric filename prefixes in `codex_prompts/`
2. what is the latest execution record for the relevant prompt
3. what is that record's `execution_status`
4. what is that record's `review_status`
5. based on the current V1 rule, is the next prompt ready

For V1, the key release rule is:

- only an `ACCEPTED` latest run for the immediately previous prompt releases the next prompt
- `UNREVIEWED` stops the queue
- `REJECTED` stops the queue
- missing prior run evidence stops the queue

## Minimum CLI behavior

Keep the CLI small and explicit.

Support one primary mode:

- no positional arguments required

Optional arguments are acceptable only if they remain small and directly useful, for example:

- `--prompt` to check readiness for a specific prompt file or numeric prefix

Do not add many modes or subcommands.

## Expected output

The helper should print a short human-readable readiness summary to stdout.

That summary should include at least:

- target prompt
- previous prompt, if one exists
- latest run record path considered
- latest run `execution_status`
- latest run `review_status`
- final readiness decision
- short reason

Keep the output textual and inspectable.

Do not create JSON outputs, databases, caches, or sidecar state.

## Important behavior rules

### 1. Preserve V1 boundaries

Do NOT:

- run prompts automatically
- update review fields automatically
- create or modify execution records
- release multiple future prompts
- build dependency-aware scheduling
- build a queue engine or status platform

This helper should only inspect and report the current readiness decision.

### 2. Respect the execution record as source of truth

Use the markdown execution record body in `notes/` as the source of truth for:

- `run_id`
- `prompt_file`
- `prompt_stem`
- `started_at_utc`
- `execution_status`
- `review_status`

Do not invent a separate state file.

### 3. Stay conservative about "latest"

If multiple records exist for the same prompt, pick the latest V1 run record using the execution-record evidence that is already present and inspectable.

Keep this logic simple and explainable.
Do not build a complex retry or history synthesizer.

### 4. Fail clearly on ambiguous or malformed inputs

If prompt ordering or record parsing is too malformed to make a safe decision, exit nonzero with a short error.

### 5. Keep the helper read-only

This helper should not modify files.

## Implementation guidance

Keep the implementation single-file unless a tiny helper is truly necessary.

A simple approach is preferred:

- discover prompt files with numeric prefixes
- sort them in sequence
- parse minimal field lines from candidate execution records in `notes/`
- identify the latest relevant record
- apply the V1 readiness rule
- print a short decision summary

Do not build a full markdown parser unless truly necessary.

## Required artifacts

### Artifact 1

Create:

`tools/codex/check_queue_readiness.py`

### Artifact 2

Create a short implementation note at:

`notes/014_queue_readiness_checker_build__TIMESTAMP.md`

This note should summarize:

- what was built
- what rule it applies for readiness
- what it intentionally does not do
- what validation was performed

## Validation requirements

After implementation, validate at least these points:

1. the helper can identify the ordered prompt sequence from `codex_prompts/`
2. the helper can find the latest run record for a prompt from `notes/`
3. the helper distinguishes `UNREVIEWED`, `ACCEPTED`, and `REJECTED`
4. the helper reports not-ready when the previous prompt is not accepted
5. the helper reports ready only when the previous prompt's latest run is accepted
6. `tools/codex/run_prompt.py` remains unchanged
7. `tools/codex/review_run.py` remains unchanged
8. `tools/codex/baby_run_prompt.py` remains unchanged

Include the validation outcome in the implementation note.

## Constraints

1. Use the exact file paths listed above.
2. Keep the helper small and read-only.
3. Do not modify the runner or review-writeback tools.
4. Do not alter the design documents in this pass.
5. Do not expand into dashboards, queue engines, retry orchestration, or broader automation.
6. Keep the implementation small enough to review comfortably.

## Success criteria

This task is successful if:

- `tools/codex/check_queue_readiness.py` exists and works
- it gives a conservative readiness decision grounded in current prompt order and latest execution-record review status
- it remains read-only and inspectable
- the implementation note explains the rule and validation clearly
- the existing runner and review-writeback tools remain unchanged

FILE codex_prompts/015_build_review_backlog_unreviewed_run_lister.md
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

FILE codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md
# Task: Validate the new queue-readiness and review-backlog helpers against the current repo state

You are working in this repository.

Your task is to exercise the two newly built read-only helpers against the current prompt and note history, then write a short validation report.

## Important framing

This is a validation and review task.

Do NOT modify existing code in this pass.
Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

Do NOT rewrite design documents in this pass.

Your job is to run the current helpers, inspect their outputs against the current repo state, and summarize whether they behave correctly and conservatively.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`

Also inspect current repo contents in:

- `codex_prompts/`
- `notes/`

## Goal

Validate whether the two helpers behave correctly against the repo’s current V1 artifacts.

The two helpers are:

- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

## What to do

### 1. Run the queue-readiness helper
Run at least these checks:

- `python3 tools/codex/check_queue_readiness.py`
- `python3 tools/codex/check_queue_readiness.py --prompt 001`
- `python3 tools/codex/check_queue_readiness.py --prompt 002`
- `python3 tools/codex/check_queue_readiness.py --prompt 013`
- `python3 tools/codex/check_queue_readiness.py --prompt 014`
- `python3 tools/codex/check_queue_readiness.py --prompt 015`

### 2. Run the review-backlog helper
Run at least these checks:

- `python3 tools/codex/list_review_backlog.py`
- `python3 tools/codex/list_review_backlog.py --unreviewed-only`

### 3. Compare outputs against actual repo evidence
Check whether the helper outputs agree with:

- actual prompt ordering in `codex_prompts/`
- actual review statuses in the current execution records in `notes/`
- the intended V1 rules in the design documents

## Questions to answer

Please answer these questions in the report:

### A. Queue-readiness correctness
- Does `check_queue_readiness.py` appear to choose the correct target and prior prompt?
- Do its readiness decisions match the current record evidence?
- Are there any edge cases or confusing behaviors visible from the current repo state?

### B. Review-backlog correctness
- Does `list_review_backlog.py` appear to identify `UNREVIEWED` records correctly?
- Does it pick the latest record per prompt correctly?
- Does its “likely needs human review next” view match the repo evidence?

### C. Consistency between the two helpers
- Do the two helpers tell a coherent story about the current repo state?
- Is there any mismatch between what the readiness checker says and what the backlog lister says?

### D. Smallest next improvement
- Based on this validation pass, what is the single smallest next improvement or cleanup step?
- Prefer one bounded suggestion only.

## Required output artifacts

Create exactly two artifacts.

### Artifact 1
Create a validation report at:

`notes/016_queue_and_backlog_helper_validation__TIMESTAMP.md`

This report should include:

- short summary
- queue-readiness validation findings
- review-backlog validation findings
- consistency findings
- one recommended next improvement

### Artifact 2
Create a short recommendation note at:

`notes/016_next_improvement_recommendation__TIMESTAMP.md`

This should contain only:

- the recommended next bounded improvement
- why it should happen next
- what should explicitly wait

## Constraints

1. Use the exact file paths listed above.
2. Do not modify the helpers in this pass.
3. Do not create a broad roadmap.
4. Recommend only one bounded next improvement.
5. Keep the validation practical and evidence-based.

## Success criteria

This task is successful if:

- both helpers are exercised against the current repo state
- the report compares their outputs to actual prompt and note evidence
- any mismatches or edge cases are clearly identified
- one small next improvement is recommended
- the result helps decide the next deliberate step with confidence

 succeeded in 421ms:

FILE notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md
# 008_define_minimal_bridge_runner_change_spec - SUCCESS

- Prompt file: `008_define_minimal_bridge_runner_change_spec.md`
- Timestamp (UTC): `20260415_203548`
- Status: `SUCCESS`

## Original Prompt

```md
# Task: Define the minimal bridge-runner change spec for V1 execution records

You are working in this repository.

Your task is to define the smallest change spec for `tools/codex/baby_run_prompt.py` so it can support the V1 execution record model without a larger refactor.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/baby_run_prompt.py`
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
- `notes/005_prompt_queue_plan__20260415_202557.md`

## Goal

Produce a narrow implementation spec for the current bridge runner.

The spec should define only the smallest changes needed to make the runner:

- emit a V1 execution record instead of a success-implies-acceptance note
- preserve stable run identity
- capture the minimum automatic fields
- leave manual review fields untouched or initialized for later completion

## Important framing

This is a design-spec task.

Do NOT implement the changes in this pass.
Do NOT split the runner into multiple modules in this pass.
Do NOT introduce a large CLI redesign.

## Questions to settle

Please settle these points:

1. What exact filename pattern should the runner write for a run record?
2. What exact markdown sections and fields should it populate automatically?
3. What review fields should be initialized but left manual?
4. How should execution status be derived from the subprocess result?
5. What minimal runtime and output metrics should be captured now?
6. What current behaviors in `baby_run_prompt.py` should remain unchanged for V1?

## Required output artifact

Create one markdown spec at:

`tools/codex/V1_Bridge_Runner_Change_Spec.md`

The spec should include:

- purpose
- scope
- current behavior summary
- required V1 changes
- non-goals
- exact data/field mapping from runner output to execution record
- open questions, if any, that must be resolved before implementation

## Constraints

1. Use the exact file paths listed above.
2. Keep the spec single-file and bridge-sized.
3. Preserve the current runner's thin role where possible.
4. Do not design a large future module layout here.
5. Keep the change set small enough that it could be reviewed and implemented in one later prompt.

FILE notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md
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

FILE notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md
# 010_review_run_prompt_candidate_and_recommend_next_step - SUCCESS

- Prompt file: `010_review_run_prompt_candidate_and_recommend_next_step.md`
- Timestamp (UTC): `20260415_234523`
- Status: `SUCCESS`

## Original Prompt

```md
# Task: Review the new V1 runner candidate and recommend the smallest next step

You are working in this repository.

Your task is to review the newly built V1 runner candidate, inspect the sample execution record it produced, and recommend the smallest sensible next move.

## Important framing

This is a review-and-recommendation task.

Do NOT implement code changes in this pass.
Do NOT rewrite major documents.
Do NOT refactor the runner.
Do NOT modify `tools/codex/baby_run_prompt.py`.

Your job is to inspect what now exists and decide the smallest next improvement or validation step.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/run_prompt.py`
- `notes/009_run_prompt_candidate_build__20260415_233407.md`
- `notes/001_smoke_test_pipeline__20260415_233343.md`

You may also read, for comparison only:

- `tools/codex/baby_run_prompt.py`
- `notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md`
- `notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md`

## Goal

Determine whether the next bounded step should be one of these:

1. a tiny polish to `tools/codex/run_prompt.py`
2. a small helper or workflow aid for manual review write-back
3. a focused environment-diagnosis prompt for the read-only session issue seen in the validation run

You must recommend only **one** next step.

## Questions to answer

Please answer these questions in your review:

### 1. Runner assessment
- Does `tools/codex/run_prompt.py` successfully implement the V1 design intent?
- What is strong about it?
- Are there any small correctness or clarity issues that should be fixed before further build-out?

### 2. Artifact assessment
- Does `notes/001_smoke_test_pipeline__20260415_233343.md` match the intended V1 execution-record shape?
- Is the separation between execution and review clear?
- Did the record preserve useful failure evidence?

### 3. Operational assessment
- Is the most important next issue now design-related, implementation-related, or environment-related?
- Does the read-only filesystem/session error appear to be mainly a runner problem or a Codex execution environment problem?

### 4. Smallest next move
Choose exactly one of the following as the recommended next move:

- tiny runner polish
- review write-back helper/workflow aid
- environment diagnosis prompt


FILE notes/011_build_v1_review_writeback_helper__20260415_235346.md
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

FILE notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md
# 012_review_v1_pipeline_and_recommend_next_options__20260416_000658

- run_id: `012_review_v1_pipeline_and_recommend_next_options__20260416_000658`
- prompt_file: `codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md`
- prompt_stem: `012_review_v1_pipeline_and_recommend_next_options`
- started_at_utc: `20260416_000658`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260416_001003`
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

- elapsed_seconds: `184.446`
- final_output_char_count: `869`
- stderr_char_count: `260421`

## Prompt Text

```md
# Task: Review the current V1 prompt workflow pipeline and recommend the best next options

You are working in this repository.

Your task is to review the current V1 pipeline as it now exists in the repository, identify the realistic next-step options, and recommend the top three.

## Important framing

This is a review and recommendation task.

Do NOT implement code in this pass.
Do NOT modify existing code or design documents in this pass.
Do NOT rewrite the workflow.
Do NOT modify `tools/codex/baby_run_prompt.py`, `tools/codex/run_prompt.py`, or `tools/codex/review_run.py`.

Your job is to inspect what now exists, enumerate the most plausible next bounded options, and recommend the strongest next three.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`

Read these notes and records:

- `notes/009_run_prompt_candidate_build__20260415_233407.md`
- `notes/010_run_prompt_candidate_review__20260415_234559.md`
- `notes/010_next_step_recommendation__20260415_234559.md`
- `notes/011_review_writeback_helper_build__20260415_235514.md`
- `notes/001_smoke_test_pipeline__20260415_234918.md`
- `notes/001_smoke_test_pipeline__20260415_233343.md`

You may also read for context only:

- `tools/codex/baby_run_prompt.py`
- `notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md`

FILE notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md
# 013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937

- run_id: `013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937`
- prompt_file: `codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md`
- prompt_stem: `013_generate_prompts_for_queue_readiness_and_review_backlog_helpers`
- started_at_utc: `20260416_001937`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260416_002126`
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

- elapsed_seconds: `109.295`
- final_output_char_count: `756`
- stderr_char_count: `149589`

## Prompt Text

```md
# Task: Generate the next two implementation prompts for the V1 workflow pipeline

You are working in this repository.

Your task is to write the next two bounded Codex prompt files for the V1 pipeline based on the current repo state and the recent options review.

## Important framing

This is a prompt-generation task only.

Do NOT implement either helper in this pass.
Do NOT modify existing code or design documents in this pass.
Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`

Your job is to generate the next two implementation prompts as files in `codex_prompts/`.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `notes/012_v1_pipeline_options_review__20260416_000819.md`
- `notes/012_top_three_next_options__20260416_000819.md`

## Goal

Generate exactly two prompt files:

1. a prompt to build the recommended **queue-readiness checker**
2. a prompt to build the recommended **review backlog / unreviewed-run lister**

These prompts should be ready for later execution through the current V1 pipeline.


FILE notes/014_build_queue_readiness_checker__20260416_002319.md
# 014_build_queue_readiness_checker__20260416_002319

- run_id: `014_build_queue_readiness_checker__20260416_002319`
- prompt_file: `codex_prompts/014_build_queue_readiness_checker.md`
- prompt_stem: `014_build_queue_readiness_checker`
- started_at_utc: `20260416_002319`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260416_002833`
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

- elapsed_seconds: `314.096`
- final_output_char_count: `1153`
- stderr_char_count: `487601`

## Prompt Text

```md
# Task: Build a small V1 queue-readiness checker for the next prompt

You are working in this repository.

Your task is to implement a small helper that answers the bounded V1 operational question:

`Is the next prompt in sequence ready to run under the current review-gate rules?`

## Important framing

This is a small helper task.

Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`

Those existing tools should remain unchanged in this pass.

You are building a thin read-only companion helper beside them, not refactoring the pipeline.

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

You may also inspect the current prompt set and current execution records in:

- `codex_prompts/`
- `notes/`

## Goal


FILE notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md
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

codex
I have enough context to validate behavior against the live repo state.

Plan:
1. Run the required helper commands exactly as requested and capture their outputs.
2. Cross-check those outputs against actual prompt order in `codex_prompts/` and actual V1 records in `notes/`, especially the latest record per prompt and each latest `review_status`.
3. Write the two requested notes with one bounded improvement recommendation only.
exec
/bin/bash -lc 'python3 tools/codex/check_queue_readiness.py' in /data/git/handson-ml3
exec
/bin/bash -lc 'python3 tools/codex/check_queue_readiness.py --prompt 001' in /data/git/handson-ml3
exec
/bin/bash -lc 'python3 tools/codex/check_queue_readiness.py --prompt 002' in /data/git/handson-ml3
exec
/bin/bash -lc 'python3 tools/codex/check_queue_readiness.py --prompt 013' in /data/git/handson-ml3
exec
/bin/bash -lc 'python3 tools/codex/check_queue_readiness.py --prompt 014' in /data/git/handson-ml3
exec
/bin/bash -lc 'python3 tools/codex/check_queue_readiness.py --prompt 015' in /data/git/handson-ml3
exec
/bin/bash -lc 'python3 tools/codex/list_review_backlog.py' in /data/git/handson-ml3
exec
/bin/bash -lc 'python3 tools/codex/list_review_backlog.py --unreviewed-only' in /data/git/handson-ml3
 succeeded in 293ms:
Ordered prompts:
- 001: codex_prompts/001_smoke_test_pipeline.md
- 002: codex_prompts/002_repo_inventory_and_status.md
- 003: codex_prompts/003_revise_Project_Deisgn_workflow_document.md
- 004: codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md
- 005: codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md
- 006: codex_prompts/006_define_v1_run_review_gate.md
- 007: codex_prompts/007_align_architecture_doc_to_v1_workflow.md
- 008: codex_prompts/008_define_minimal_bridge_runner_change_spec.md
- 009: codex_prompts/009_build_v1_run_prompt_candidate.md
- 010: codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md
- 011: codex_prompts/011_build_v1_review_writeback_helper.md
- 012: codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md
- 013: codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md
- 014: codex_prompts/014_build_queue_readiness_checker.md
- 015: codex_prompts/015_build_review_backlog_unreviewed_run_lister.md
- 016: codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md

Target prompt: codex_prompts/002_repo_inventory_and_status.md
Previous prompt: codex_prompts/001_smoke_test_pipeline.md
Latest run record: notes/001_smoke_test_pipeline__20260415_234918.md
Latest run execution_status: EXECUTED
Latest run review_status: ACCEPTED
Ready: YES
Reason: latest V1 run for the immediately previous prompt is ACCEPTED

 succeeded in 255ms:
Ordered prompts:
- 001: codex_prompts/001_smoke_test_pipeline.md
- 002: codex_prompts/002_repo_inventory_and_status.md
- 003: codex_prompts/003_revise_Project_Deisgn_workflow_document.md
- 004: codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md
- 005: codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md
- 006: codex_prompts/006_define_v1_run_review_gate.md
- 007: codex_prompts/007_align_architecture_doc_to_v1_workflow.md
- 008: codex_prompts/008_define_minimal_bridge_runner_change_spec.md
- 009: codex_prompts/009_build_v1_run_prompt_candidate.md
- 010: codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md
- 011: codex_prompts/011_build_v1_review_writeback_helper.md
- 012: codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md
- 013: codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md
- 014: codex_prompts/014_build_queue_readiness_checker.md
- 015: codex_prompts/015_build_review_backlog_unreviewed_run_lister.md
- 016: codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md

Target prompt: codex_prompts/001_smoke_test_pipeline.md
Previous prompt: none
Latest run record: none
Latest run execution_status: n/a
Latest run review_status: n/a
Ready: YES
Reason: first prompt in sequence has no prior review gate

 succeeded in 244ms:
Ordered prompts:
- 001: codex_prompts/001_smoke_test_pipeline.md
- 002: codex_prompts/002_repo_inventory_and_status.md
- 003: codex_prompts/003_revise_Project_Deisgn_workflow_document.md
- 004: codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md
- 005: codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md
- 006: codex_prompts/006_define_v1_run_review_gate.md
- 007: codex_prompts/007_align_architecture_doc_to_v1_workflow.md
- 008: codex_prompts/008_define_minimal_bridge_runner_change_spec.md
- 009: codex_prompts/009_build_v1_run_prompt_candidate.md
- 010: codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md
- 011: codex_prompts/011_build_v1_review_writeback_helper.md
- 012: codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md
- 013: codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md
- 014: codex_prompts/014_build_queue_readiness_checker.md
- 015: codex_prompts/015_build_review_backlog_unreviewed_run_lister.md
- 016: codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md

Target prompt: codex_prompts/002_repo_inventory_and_status.md
Previous prompt: codex_prompts/001_smoke_test_pipeline.md
Latest run record: notes/001_smoke_test_pipeline__20260415_234918.md
Latest run execution_status: EXECUTED
Latest run review_status: ACCEPTED
Ready: YES
Reason: latest V1 run for the immediately previous prompt is ACCEPTED

 succeeded in 206ms:
Ordered prompts:
- 001: codex_prompts/001_smoke_test_pipeline.md
- 002: codex_prompts/002_repo_inventory_and_status.md
- 003: codex_prompts/003_revise_Project_Deisgn_workflow_document.md
- 004: codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md
- 005: codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md
- 006: codex_prompts/006_define_v1_run_review_gate.md
- 007: codex_prompts/007_align_architecture_doc_to_v1_workflow.md
- 008: codex_prompts/008_define_minimal_bridge_runner_change_spec.md
- 009: codex_prompts/009_build_v1_run_prompt_candidate.md
- 010: codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md
- 011: codex_prompts/011_build_v1_review_writeback_helper.md
- 012: codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md
- 013: codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md
- 014: codex_prompts/014_build_queue_readiness_checker.md
- 015: codex_prompts/015_build_review_backlog_unreviewed_run_lister.md
- 016: codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md

Target prompt: codex_prompts/014_build_queue_readiness_checker.md
Previous prompt: codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md
Latest run record: notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md
Latest run execution_status: EXECUTED
Latest run review_status: UNREVIEWED
Ready: NO
Reason: latest V1 run for the immediately previous prompt is UNREVIEWED

 succeeded in 149ms:
V1 review backlog summary from notes/
Discovered V1 execution records: 7
Unreviewed records: 6
Prompts with latest record: 6

UNREVIEWED records:
- notes/001_smoke_test_pipeline__20260415_233343.md | prompt=codex_prompts/001_smoke_test_pipeline.md | started=20260415_233343 | execution=EXECUTION_FAILED | review=UNREVIEWED
- notes/011_build_v1_review_writeback_helper__20260415_235346.md | prompt=codex_prompts/011_build_v1_review_writeback_helper.md | started=20260415_235346 | execution=EXECUTED | review=UNREVIEWED
- notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md | prompt=codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md | started=20260416_000658 | execution=EXECUTED | review=UNREVIEWED
- notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md | prompt=codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md | started=20260416_001937 | execution=EXECUTED | review=UNREVIEWED
- notes/014_build_queue_readiness_checker__20260416_002319.md | prompt=codex_prompts/014_build_queue_readiness_checker.md | started=20260416_002319 | execution=EXECUTED | review=UNREVIEWED
- notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md | prompt=codex_prompts/015_build_review_backlog_unreviewed_run_lister.md | started=20260416_003109 | execution=EXECUTED | review=UNREVIEWED

Latest record per prompt:
- notes/011_build_v1_review_writeback_helper__20260415_235346.md | prompt=codex_prompts/011_build_v1_review_writeback_helper.md | started=20260415_235346 | execution=EXECUTED | review=UNREVIEWED
- notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md | prompt=codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md | started=20260416_000658 | execution=EXECUTED | review=UNREVIEWED
- notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md | prompt=codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md | started=20260416_001937 | execution=EXECUTED | review=UNREVIEWED
- notes/014_build_queue_readiness_checker__20260416_002319.md | prompt=codex_prompts/014_build_queue_readiness_checker.md | started=20260416_002319 | execution=EXECUTED | review=UNREVIEWED
- notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md | prompt=codex_prompts/015_build_review_backlog_unreviewed_run_lister.md | started=20260416_003109 | execution=EXECUTED | review=UNREVIEWED

Likely needs human review next:
- notes/011_build_v1_review_writeback_helper__20260415_235346.md | prompt=codex_prompts/011_build_v1_review_writeback_helper.md | started=20260415_235346 | execution=EXECUTED | review=UNREVIEWED
- notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md | prompt=codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md | started=20260416_000658 | execution=EXECUTED | review=UNREVIEWED
- notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md | prompt=codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md | started=20260416_001937 | execution=EXECUTED | review=UNREVIEWED
- notes/014_build_queue_readiness_checker__20260416_002319.md | prompt=codex_prompts/014_build_queue_readiness_checker.md | started=20260416_002319 | execution=EXECUTED | review=UNREVIEWED
- notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md | prompt=codex_prompts/015_build_review_backlog_unreviewed_run_lister.md | started=20260416_003109 | execution=EXECUTED | review=UNREVIEWED

 succeeded in 255ms:
Ordered prompts:
- 001: codex_prompts/001_smoke_test_pipeline.md
- 002: codex_prompts/002_repo_inventory_and_status.md
- 003: codex_prompts/003_revise_Project_Deisgn_workflow_document.md
- 004: codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md
- 005: codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md
- 006: codex_prompts/006_define_v1_run_review_gate.md
- 007: codex_prompts/007_align_architecture_doc_to_v1_workflow.md
- 008: codex_prompts/008_define_minimal_bridge_runner_change_spec.md
- 009: codex_prompts/009_build_v1_run_prompt_candidate.md
- 010: codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md
- 011: codex_prompts/011_build_v1_review_writeback_helper.md
- 012: codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md
- 013: codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md
- 014: codex_prompts/014_build_queue_readiness_checker.md
- 015: codex_prompts/015_build_review_backlog_unreviewed_run_lister.md
- 016: codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md

Target prompt: codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md
Previous prompt: codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md
Latest run record: notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md
Latest run execution_status: EXECUTED
Latest run review_status: UNREVIEWED
Ready: NO
Reason: latest V1 run for the immediately previous prompt is UNREVIEWED

 succeeded in 162ms:
V1 review backlog summary from notes/
Discovered V1 execution records: 7
Unreviewed records: 6
Prompts with latest record: 6

UNREVIEWED records:
- notes/001_smoke_test_pipeline__20260415_233343.md | prompt=codex_prompts/001_smoke_test_pipeline.md | started=20260415_233343 | execution=EXECUTION_FAILED | review=UNREVIEWED
- notes/011_build_v1_review_writeback_helper__20260415_235346.md | prompt=codex_prompts/011_build_v1_review_writeback_helper.md | started=20260415_235346 | execution=EXECUTED | review=UNREVIEWED
- notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md | prompt=codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md | started=20260416_000658 | execution=EXECUTED | review=UNREVIEWED
- notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md | prompt=codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md | started=20260416_001937 | execution=EXECUTED | review=UNREVIEWED
- notes/014_build_queue_readiness_checker__20260416_002319.md | prompt=codex_prompts/014_build_queue_readiness_checker.md | started=20260416_002319 | execution=EXECUTED | review=UNREVIEWED
- notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md | prompt=codex_prompts/015_build_review_backlog_unreviewed_run_lister.md | started=20260416_003109 | execution=EXECUTED | review=UNREVIEWED

Latest record per prompt:
- notes/001_smoke_test_pipeline__20260415_234918.md | prompt=codex_prompts/001_smoke_test_pipeline.md | started=20260415_234918 | execution=EXECUTED | review=ACCEPTED
- notes/011_build_v1_review_writeback_helper__20260415_235346.md | prompt=codex_prompts/011_build_v1_review_writeback_helper.md | started=20260415_235346 | execution=EXECUTED | review=UNREVIEWED
- notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md | prompt=codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md | started=20260416_000658 | execution=EXECUTED | review=UNREVIEWED
- notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md | prompt=codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md | started=20260416_001937 | execution=EXECUTED | review=UNREVIEWED
- notes/014_build_queue_readiness_checker__20260416_002319.md | prompt=codex_prompts/014_build_queue_readiness_checker.md | started=20260416_002319 | execution=EXECUTED | review=UNREVIEWED
- notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md | prompt=codex_prompts/015_build_review_backlog_unreviewed_run_lister.md | started=20260416_003109 | execution=EXECUTED | review=UNREVIEWED

Likely needs human review next:
- notes/011_build_v1_review_writeback_helper__20260415_235346.md | prompt=codex_prompts/011_build_v1_review_writeback_helper.md | started=20260415_235346 | execution=EXECUTED | review=UNREVIEWED
- notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md | prompt=codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md | started=20260416_000658 | execution=EXECUTED | review=UNREVIEWED
- notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md | prompt=codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md | started=20260416_001937 | execution=EXECUTED | review=UNREVIEWED
- notes/014_build_queue_readiness_checker__20260416_002319.md | prompt=codex_prompts/014_build_queue_readiness_checker.md | started=20260416_002319 | execution=EXECUTED | review=UNREVIEWED
- notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md | prompt=codex_prompts/015_build_review_backlog_unreviewed_run_lister.md | started=20260416_003109 | execution=EXECUTED | review=UNREVIEWED

 succeeded in 211ms:
Ordered prompts:
- 001: codex_prompts/001_smoke_test_pipeline.md
- 002: codex_prompts/002_repo_inventory_and_status.md
- 003: codex_prompts/003_revise_Project_Deisgn_workflow_document.md
- 004: codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md
- 005: codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md
- 006: codex_prompts/006_define_v1_run_review_gate.md
- 007: codex_prompts/007_align_architecture_doc_to_v1_workflow.md
- 008: codex_prompts/008_define_minimal_bridge_runner_change_spec.md
- 009: codex_prompts/009_build_v1_run_prompt_candidate.md
- 010: codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md
- 011: codex_prompts/011_build_v1_review_writeback_helper.md
- 012: codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md
- 013: codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md
- 014: codex_prompts/014_build_queue_readiness_checker.md
- 015: codex_prompts/015_build_review_backlog_unreviewed_run_lister.md
- 016: codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md

Target prompt: codex_prompts/015_build_review_backlog_unreviewed_run_lister.md
Previous prompt: codex_prompts/014_build_queue_readiness_checker.md
Latest run record: notes/014_build_queue_readiness_checker__20260416_002319.md
Latest run execution_status: EXECUTED
Latest run review_status: UNREVIEWED
Ready: NO
Reason: latest V1 run for the immediately previous prompt is UNREVIEWED

codex
The helper outputs are in hand. I’m doing one more evidence pass to pin down the latest-record state cleanly before I write the validation notes, because the main subtlety is how stale unreviewed records relate to a newer accepted run for the same prompt.
exec
/bin/bash -lc "for f in notes/*.md; do if rg -q '"'^## Execution Facts$'"' \""'$f" && rg -q '"'"'^- run_id:'"' \""'$f"; then printf '"'%s\\n' \""'$f"; rg -n '"'"'^- (run_id|prompt_file|started_at_utc|execution_status|review_status):'"' \""'$f"; printf '"'\\n'; fi; done" in /data/git/handson-ml3
exec
/bin/bash -lc 'date -u +%Y%m%d_%H%M%S' in /data/git/handson-ml3
 succeeded in 0ms:
20260416_003710

 succeeded in 430ms:
notes/001_smoke_test_pipeline__20260415_234918.md
3:- run_id: `001_smoke_test_pipeline__20260415_234918`
4:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
6:- started_at_utc: `20260415_234918`
10:- execution_status: `EXECUTED`
18:- review_status: `ACCEPTED`

notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md
2752:- run_id: `<run_id>`
2753:- prompt_file: `<prompt_file>`
2755:- started_at_utc: `<started_at_utc>`
2763:- execution_status: `<EXECUTED|EXECUTION_FAILED>`
2775:- review_status: `UNREVIEWED`
4394:- run_id: `<run_id>`
4395:- prompt_file: `<prompt_file>`
4397:- started_at_utc: `<started_at_utc>`
4405:- execution_status: `<EXECUTED|EXECUTION_FAILED>`
4417:- review_status: `UNREVIEWED`

notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md
1426:- run_id: `<run_id>`
1427:- prompt_file: `<prompt_file>`
1429:- started_at_utc: `<started_at_utc>`
1437:- execution_status: `<EXECUTED|EXECUTION_FAILED>`
1449:- review_status: `UNREVIEWED`
3041:- run_id: `001_smoke_test_pipeline__20260415_233343`
3042:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
3044:- started_at_utc: `20260415_233343`
3048:- execution_status: `EXECUTION_FAILED`
3056:- review_status: `UNREVIEWED`

notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md
1355:- run_id: `<run_id>`
1356:- prompt_file: `<prompt_file>`
1358:- started_at_utc: `<started_at_utc>`
1366:- execution_status: `<EXECUTED|EXECUTION_FAILED>`
1378:- review_status: `UNREVIEWED`
1843:- run_id: `001_smoke_test_pipeline__20260415_233343`
1844:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
1846:- started_at_utc: `20260415_233343`
1850:- execution_status: `EXECUTION_FAILED`
1858:- review_status: `UNREVIEWED`

notes/011_build_v1_review_writeback_helper__20260415_235346.md
3:- run_id: `011_build_v1_review_writeback_helper__20260415_235346`
4:- prompt_file: `codex_prompts/011_build_v1_review_writeback_helper.md`
6:- started_at_utc: `20260415_235346`
10:- execution_status: `EXECUTED`
18:- review_status: `UNREVIEWED`
1604:- run_id: `001_smoke_test_pipeline__20260415_234918`
1605:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
1607:- started_at_utc: `20260415_234918`
1611:- execution_status: `EXECUTED`
1619:- review_status: `UNREVIEWED`
3100:- run_id: `001_smoke_test_pipeline__20260415_234918`
3101:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
3103:- started_at_utc: `20260415_234918`
3107:- execution_status: `EXECUTED`
3115:- review_status: `UNREVIEWED`
3400:- run_id: `001_smoke_test_pipeline__20260415_234918`
3401:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
3403:- started_at_utc: `20260415_234918`
3407:- execution_status: `EXECUTED`
3415:- review_status: `ACCEPTED`

notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md
3:- run_id: `012_review_v1_pipeline_and_recommend_next_options__20260416_000658`
4:- prompt_file: `codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md`
6:- started_at_utc: `20260416_000658`
10:- execution_status: `EXECUTED`
18:- review_status: `UNREVIEWED`
1430:- run_id: `<run_id>`
1431:- prompt_file: `<prompt_file>`
1433:- started_at_utc: `<started_at_utc>`
1441:- execution_status: `<EXECUTED|EXECUTION_FAILED>`
1453:- review_status: `UNREVIEWED`
2442:- run_id: `001_smoke_test_pipeline__20260415_234918`
2443:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
2445:- started_at_utc: `20260415_234918`
2449:- execution_status: `EXECUTED`
2457:- review_status: `ACCEPTED`
2706:- run_id: `001_smoke_test_pipeline__20260415_233343`
2707:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
2709:- started_at_utc: `20260415_233343`
2713:- execution_status: `EXECUTION_FAILED`
2721:- review_status: `UNREVIEWED`
3034:- run_id: `011_build_v1_review_writeback_helper__20260415_235346`
3035:- prompt_file: `codex_prompts/011_build_v1_review_writeback_helper.md`
3037:- started_at_utc: `20260415_235346`
3041:- execution_status: `EXECUTED`
3049:- review_status: `UNREVIEWED`

notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md
3:- run_id: `013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937`
4:- prompt_file: `codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md`
6:- started_at_utc: `20260416_001937`
10:- execution_status: `EXECUTED`
18:- review_status: `UNREVIEWED`

notes/014_build_queue_readiness_checker__20260416_002319.md
3:- run_id: `014_build_queue_readiness_checker__20260416_002319`
4:- prompt_file: `codex_prompts/014_build_queue_readiness_checker.md`
6:- started_at_utc: `20260416_002319`
10:- execution_status: `EXECUTED`
18:- review_status: `UNREVIEWED`
2161:- run_id: `001_smoke_test_pipeline__20260415_234918`
2162:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
2164:- started_at_utc: `20260415_234918`
2168:- execution_status: `EXECUTED`
2176:- review_status: `ACCEPTED`
2385:- run_id: `011_build_v1_review_writeback_helper__20260415_235346`
2386:- prompt_file: `codex_prompts/011_build_v1_review_writeback_helper.md`
2388:- started_at_utc: `20260415_235346`
2392:- execution_status: `EXECUTED`
2400:- review_status: `UNREVIEWED`
2609:- run_id: `012_review_v1_pipeline_and_recommend_next_options__20260416_000658`
2610:- prompt_file: `codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md`
2612:- started_at_utc: `20260416_000658`
2616:- execution_status: `EXECUTED`
2624:- review_status: `UNREVIEWED`
2833:- run_id: `013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937`
2834:- prompt_file: `codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md`
2836:- started_at_utc: `20260416_001937`
2840:- execution_status: `EXECUTED`
2848:- review_status: `UNREVIEWED`
9707:- run_id: `001_smoke_test_pipeline__20260415_233343`
9708:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
9710:- started_at_utc: `20260415_233343`
9714:- execution_status: `EXECUTION_FAILED`
9722:- review_status: `UNREVIEWED`

notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md
3:- run_id: `015_build_review_backlog_unreviewed_run_lister__20260416_003109`
4:- prompt_file: `codex_prompts/015_build_review_backlog_unreviewed_run_lister.md`
6:- started_at_utc: `20260416_003109`
10:- execution_status: `EXECUTED`
18:- review_status: `UNREVIEWED`
2072:- run_id: `001_smoke_test_pipeline__20260415_233343`
2073:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
2075:- started_at_utc: `20260415_233343`
2079:- execution_status: `EXECUTION_FAILED`
2087:- review_status: `UNREVIEWED`
2163:- run_id: `001_smoke_test_pipeline__20260415_234918`
2164:- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
2166:- started_at_utc: `20260415_234918`
2170:- execution_status: `EXECUTED`
2178:- review_status: `ACCEPTED`


codex
I’m at the write-up step. The validation result is coherent: both helpers are conservative and line up with the current V1 records, with the main usability issue being that the default readiness target falls back to `002` because prompts `002` through `010` have no V1 records even though later prompts do.
apply patch
patch: completed
/data/git/handson-ml3/notes/016_queue_and_backlog_helper_validation__20260416_003710.md
diff --git a/notes/016_queue_and_backlog_helper_validation__20260416_003710.md b/notes/016_queue_and_backlog_helper_validation__20260416_003710.md
new file mode 100644
index 0000000000000000000000000000000000000000..903db8f1f88e70d819fc10528076fa3e866759fa
--- /dev/null
+++ b/notes/016_queue_and_backlog_helper_validation__20260416_003710.md
@@ -0,0 +1,101 @@
+# 016 Queue And Backlog Helper Validation
+
+## Short Summary
+
+I validated `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py` against the current prompt order in `codex_prompts/` and the current V1 execution records in `notes/`.
+
+Both helpers behaved conservatively and consistently with the V1 design documents. The main visible usability issue is that the default queue-readiness result points back to prompt `002`, because prompts `002` through `010` do not have V1 execution records even though later prompts `011` through `015` do.
+
+Current V1 record evidence in `notes/`:
+
+- `notes/001_smoke_test_pipeline__20260415_233343.md`: `EXECUTION_FAILED`, `UNREVIEWED`
+- `notes/001_smoke_test_pipeline__20260415_234918.md`: `EXECUTED`, `ACCEPTED`
+- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/014_build_queue_readiness_checker__20260416_002319.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md`: `EXECUTED`, `UNREVIEWED`
+
+## Queue-Readiness Validation Findings
+
+Prompt ordering was correct. The helper discovered the current sequence `001` through `016` from `codex_prompts/` in numeric order.
+
+The default run of `python3 tools/codex/check_queue_readiness.py` selected `codex_prompts/002_repo_inventory_and_status.md` as the target, with `001` as the previous prompt and `notes/001_smoke_test_pipeline__20260415_234918.md` as the latest prior V1 record. That is correct under the V1 rule set: the latest V1 run for `001` is `ACCEPTED`, and there is no V1 run evidence for `002`, so `002` is the first unreleased prompt in sequence.
+
+The prompt-specific checks also matched the repo evidence:
+
+- `--prompt 001`: `Ready: YES`, because the first prompt has no prior gate.
+- `--prompt 002`: `Ready: YES`, because latest `001` is `ACCEPTED`.
+- `--prompt 013`: `Ready: NO`, because latest `012` is `UNREVIEWED` in `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`.
+- `--prompt 014`: `Ready: NO`, because latest `013` is `UNREVIEWED` in `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`.
+- `--prompt 015`: `Ready: NO`, because latest `014` is `UNREVIEWED` in `notes/014_build_queue_readiness_checker__20260416_002319.md`.
+
+This matches the V1 rule in `tools/codex/V1_Run_Review_Gate.md`: only an `ACCEPTED` latest run for the immediately previous prompt releases the next prompt.
+
+Visible edge case from the current repo state:
+
+- The default target being `002` is correct but potentially surprising, because later prompts `011` through `015` already have V1 execution records. The helper is correctly ignoring older non-V1 `__SUCCESS__` notes for prompts `002` through `010`, but that gap is not explained in the output.
+
+## Review-Backlog Validation Findings
+
+`python3 tools/codex/list_review_backlog.py` reported:
+
+- `Discovered V1 execution records: 7`
+- `Unreviewed records: 6`
+- `Prompts with latest record: 6`
+
+Those counts match the current repo evidence exactly: there are 7 V1 records total, 6 of them are still `UNREVIEWED`, and those 7 records cover 6 prompts because prompt `001` has two V1 records.
+
+Its latest-per-prompt selection was also correct:
+
+- For `001`, it chose `notes/001_smoke_test_pipeline__20260415_234918.md`, which is newer than the failed `001` run and has `review_status: ACCEPTED`.
+- For `011` through `015`, each prompt has one V1 record and each latest record is `UNREVIEWED`.
+
+Its `UNREVIEWED records` section also behaved correctly. It includes the older failed `001` run because that record is still literally `UNREVIEWED`, even though it is no longer the latest record for prompt `001`.
+
+Its `Likely needs human review next` view is the right conservative reduction of the backlog for the current repo state. It lists only prompts whose latest record is `UNREVIEWED`, which means:
+
+- `011`
+- `012`
+- `013`
+- `014`
+- `015`
+
+That matches the actual latest-record evidence.
+
+The `--unreviewed-only` mode also behaved consistently. It filtered the `Latest record per prompt` section down to latest records that are themselves `UNREVIEWED`, while leaving the full `UNREVIEWED records` section intact.
+
+## Consistency Findings
+
+The two helpers tell a coherent story.
+
+- `check_queue_readiness.py` says the queue is only V1-ready up to prompt `001`, so the next ready prompt is `002`.
+- `list_review_backlog.py` says the current latest records needing review are `011` through `015`.
+
+Those outputs are not contradictory. They are answering different bounded questions:
+
+- readiness asks whether the reviewed V1 queue may advance from the start of the prompt sequence
+- backlog asks which existing V1 run records still await human review
+
+The stale failed `001` record is the main point that could look inconsistent at first glance, but the helpers remain coherent:
+
+- backlog includes it in the all-unreviewed list because it is still unreviewed
+- backlog excludes it from "likely needs human review next" because latest `001` is already accepted
+- readiness also ignores it for queue release because latest `001` is accepted
+
+## One Recommended Next Improvement
+
+The single smallest next improvement is to make `tools/codex/check_queue_readiness.py` print a short explanation when the default target is blocked by a missing V1 history gap, especially in a repo like this one where prompts `002` through `010` have legacy notes but no V1 execution records.
+
+Why this should happen next:
+
+- the helper is already making the correct decision
+- the current repo state makes that correct decision easy to misread
+- a one-line explanation would reduce confusion without changing queue policy or adding workflow complexity
+
+What should explicitly wait:
+
+- any change to queue rules
+- any backlog prioritization policy
+- any automatic bridging from legacy `__SUCCESS__` notes into V1 state
+- any broader dashboard or workflow-engine work

diff --git a/notes/016_queue_and_backlog_helper_validation__20260416_003710.md b/notes/016_queue_and_backlog_helper_validation__20260416_003710.md
new file mode 100644
index 0000000000000000000000000000000000000000..903db8f1f88e70d819fc10528076fa3e866759fa
--- /dev/null
+++ b/notes/016_queue_and_backlog_helper_validation__20260416_003710.md
@@ -0,0 +1,101 @@
+# 016 Queue And Backlog Helper Validation
+
+## Short Summary
+
+I validated `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py` against the current prompt order in `codex_prompts/` and the current V1 execution records in `notes/`.
+
+Both helpers behaved conservatively and consistently with the V1 design documents. The main visible usability issue is that the default queue-readiness result points back to prompt `002`, because prompts `002` through `010` do not have V1 execution records even though later prompts `011` through `015` do.
+
+Current V1 record evidence in `notes/`:
+
+- `notes/001_smoke_test_pipeline__20260415_233343.md`: `EXECUTION_FAILED`, `UNREVIEWED`
+- `notes/001_smoke_test_pipeline__20260415_234918.md`: `EXECUTED`, `ACCEPTED`
+- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/014_build_queue_readiness_checker__20260416_002319.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md`: `EXECUTED`, `UNREVIEWED`
+
+## Queue-Readiness Validation Findings
+
+Prompt ordering was correct. The helper discovered the current sequence `001` through `016` from `codex_prompts/` in numeric order.
+
+The default run of `python3 tools/codex/check_queue_readiness.py` selected `codex_prompts/002_repo_inventory_and_status.md` as the target, with `001` as the previous prompt and `notes/001_smoke_test_pipeline__20260415_234918.md` as the latest prior V1 record. That is correct under the V1 rule set: the latest V1 run for `001` is `ACCEPTED`, and there is no V1 run evidence for `002`, so `002` is the first unreleased prompt in sequence.
+
+The prompt-specific checks also matched the repo evidence:
+
+- `--prompt 001`: `Ready: YES`, because the first prompt has no prior gate.
+- `--prompt 002`: `Ready: YES`, because latest `001` is `ACCEPTED`.
+- `--prompt 013`: `Ready: NO`, because latest `012` is `UNREVIEWED` in `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`.
+- `--prompt 014`: `Ready: NO`, because latest `013` is `UNREVIEWED` in `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`.
+- `--prompt 015`: `Ready: NO`, because latest `014` is `UNREVIEWED` in `notes/014_build_queue_readiness_checker__20260416_002319.md`.
+
+This matches the V1 rule in `tools/codex/V1_Run_Review_Gate.md`: only an `ACCEPTED` latest run for the immediately previous prompt releases the next prompt.
+
+Visible edge case from the current repo state:
+
+- The default target being `002` is correct but potentially surprising, because later prompts `011` through `015` already have V1 execution records. The helper is correctly ignoring older non-V1 `__SUCCESS__` notes for prompts `002` through `010`, but that gap is not explained in the output.
+
+## Review-Backlog Validation Findings
+
+`python3 tools/codex/list_review_backlog.py` reported:
+
+- `Discovered V1 execution records: 7`
+- `Unreviewed records: 6`
+- `Prompts with latest record: 6`
+
+Those counts match the current repo evidence exactly: there are 7 V1 records total, 6 of them are still `UNREVIEWED`, and those 7 records cover 6 prompts because prompt `001` has two V1 records.
+
+Its latest-per-prompt selection was also correct:
+
+- For `001`, it chose `notes/001_smoke_test_pipeline__20260415_234918.md`, which is newer than the failed `001` run and has `review_status: ACCEPTED`.
+- For `011` through `015`, each prompt has one V1 record and each latest record is `UNREVIEWED`.
+
+Its `UNREVIEWED records` section also behaved correctly. It includes the older failed `001` run because that record is still literally `UNREVIEWED`, even though it is no longer the latest record for prompt `001`.
+
+Its `Likely needs human review next` view is the right conservative reduction of the backlog for the current repo state. It lists only prompts whose latest record is `UNREVIEWED`, which means:
+
+- `011`
+- `012`
+- `013`
+- `014`
+- `015`
+
+That matches the actual latest-record evidence.
+
+The `--unreviewed-only` mode also behaved consistently. It filtered the `Latest record per prompt` section down to latest records that are themselves `UNREVIEWED`, while leaving the full `UNREVIEWED records` section intact.
+
+## Consistency Findings
+
+The two helpers tell a coherent story.
+
+- `check_queue_readiness.py` says the queue is only V1-ready up to prompt `001`, so the next ready prompt is `002`.
+- `list_review_backlog.py` says the current latest records needing review are `011` through `015`.
+
+Those outputs are not contradictory. They are answering different bounded questions:
+
+- readiness asks whether the reviewed V1 queue may advance from the start of the prompt sequence
+- backlog asks which existing V1 run records still await human review
+
+The stale failed `001` record is the main point that could look inconsistent at first glance, but the helpers remain coherent:
+
+- backlog includes it in the all-unreviewed list because it is still unreviewed
+- backlog excludes it from "likely needs human review next" because latest `001` is already accepted
+- readiness also ignores it for queue release because latest `001` is accepted
+
+## One Recommended Next Improvement
+
+The single smallest next improvement is to make `tools/codex/check_queue_readiness.py` print a short explanation when the default target is blocked by a missing V1 history gap, especially in a repo like this one where prompts `002` through `010` have legacy notes but no V1 execution records.
+
+Why this should happen next:
+
+- the helper is already making the correct decision
+- the current repo state makes that correct decision easy to misread
+- a one-line explanation would reduce confusion without changing queue policy or adding workflow complexity
+
+What should explicitly wait:
+
+- any change to queue rules
+- any backlog prioritization policy
+- any automatic bridging from legacy `__SUCCESS__` notes into V1 state
+- any broader dashboard or workflow-engine work

codex
I’ve written the main validation report. I’m adding the short follow-on recommendation note now, keeping it strictly limited to the single bounded improvement and what should wait.
apply patch
patch: completed
/data/git/handson-ml3/notes/016_next_improvement_recommendation__20260416_003710.md
diff --git a/notes/016_next_improvement_recommendation__20260416_003710.md b/notes/016_next_improvement_recommendation__20260416_003710.md
new file mode 100644
index 0000000000000000000000000000000000000000..d740ec7dc33bce389afedfcd7f17f70f838ed245
--- /dev/null
+++ b/notes/016_next_improvement_recommendation__20260416_003710.md
@@ -0,0 +1,17 @@
+# 016 Next Improvement Recommendation
+
+Recommended next bounded improvement:
+
+- Add a short gap-explanation line to `tools/codex/check_queue_readiness.py` when the default target is blocked by missing V1 execution-record history for earlier prompts.
+
+Why it should happen next:
+
+- The helper is making the correct conservative decision now, but in the current repo state that decision is easy to misread because prompts `002` through `010` have legacy notes but no V1 execution records.
+- A small output clarification would improve operator understanding without changing any queue logic.
+
+What should explicitly wait:
+
+- any change to the V1 release rule
+- any migration or inference from legacy `__SUCCESS__` notes
+- any backlog prioritization feature
+- any broader queue engine or dashboard work
diff --git a/notes/016_queue_and_backlog_helper_validation__20260416_003710.md b/notes/016_queue_and_backlog_helper_validation__20260416_003710.md
new file mode 100644
index 0000000000000000000000000000000000000000..903db8f1f88e70d819fc10528076fa3e866759fa
--- /dev/null
+++ b/notes/016_queue_and_backlog_helper_validation__20260416_003710.md
@@ -0,0 +1,101 @@
+# 016 Queue And Backlog Helper Validation
+
+## Short Summary
+
+I validated `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py` against the current prompt order in `codex_prompts/` and the current V1 execution records in `notes/`.
+
+Both helpers behaved conservatively and consistently with the V1 design documents. The main visible usability issue is that the default queue-readiness result points back to prompt `002`, because prompts `002` through `010` do not have V1 execution records even though later prompts `011` through `015` do.
+
+Current V1 record evidence in `notes/`:
+
+- `notes/001_smoke_test_pipeline__20260415_233343.md`: `EXECUTION_FAILED`, `UNREVIEWED`
+- `notes/001_smoke_test_pipeline__20260415_234918.md`: `EXECUTED`, `ACCEPTED`
+- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/014_build_queue_readiness_checker__20260416_002319.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md`: `EXECUTED`, `UNREVIEWED`
+
+## Queue-Readiness Validation Findings
+
+Prompt ordering was correct. The helper discovered the current sequence `001` through `016` from `codex_prompts/` in numeric order.
+
+The default run of `python3 tools/codex/check_queue_readiness.py` selected `codex_prompts/002_repo_inventory_and_status.md` as the target, with `001` as the previous prompt and `notes/001_smoke_test_pipeline__20260415_234918.md` as the latest prior V1 record. That is correct under the V1 rule set: the latest V1 run for `001` is `ACCEPTED`, and there is no V1 run evidence for `002`, so `002` is the first unreleased prompt in sequence.
+
+The prompt-specific checks also matched the repo evidence:
+
+- `--prompt 001`: `Ready: YES`, because the first prompt has no prior gate.
+- `--prompt 002`: `Ready: YES`, because latest `001` is `ACCEPTED`.
+- `--prompt 013`: `Ready: NO`, because latest `012` is `UNREVIEWED` in `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`.
+- `--prompt 014`: `Ready: NO`, because latest `013` is `UNREVIEWED` in `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`.
+- `--prompt 015`: `Ready: NO`, because latest `014` is `UNREVIEWED` in `notes/014_build_queue_readiness_checker__20260416_002319.md`.
+
+This matches the V1 rule in `tools/codex/V1_Run_Review_Gate.md`: only an `ACCEPTED` latest run for the immediately previous prompt releases the next prompt.
+
+Visible edge case from the current repo state:
+
+- The default target being `002` is correct but potentially surprising, because later prompts `011` through `015` already have V1 execution records. The helper is correctly ignoring older non-V1 `__SUCCESS__` notes for prompts `002` through `010`, but that gap is not explained in the output.
+
+## Review-Backlog Validation Findings
+
+`python3 tools/codex/list_review_backlog.py` reported:
+
+- `Discovered V1 execution records: 7`
+- `Unreviewed records: 6`
+- `Prompts with latest record: 6`
+
+Those counts match the current repo evidence exactly: there are 7 V1 records total, 6 of them are still `UNREVIEWED`, and those 7 records cover 6 prompts because prompt `001` has two V1 records.
+
+Its latest-per-prompt selection was also correct:
+
+- For `001`, it chose `notes/001_smoke_test_pipeline__20260415_234918.md`, which is newer than the failed `001` run and has `review_status: ACCEPTED`.
+- For `011` through `015`, each prompt has one V1 record and each latest record is `UNREVIEWED`.
+
+Its `UNREVIEWED records` section also behaved correctly. It includes the older failed `001` run because that record is still literally `UNREVIEWED`, even though it is no longer the latest record for prompt `001`.
+
+Its `Likely needs human review next` view is the right conservative reduction of the backlog for the current repo state. It lists only prompts whose latest record is `UNREVIEWED`, which means:
+
+- `011`
+- `012`
+- `013`
+- `014`
+- `015`
+
+That matches the actual latest-record evidence.
+
+The `--unreviewed-only` mode also behaved consistently. It filtered the `Latest record per prompt` section down to latest records that are themselves `UNREVIEWED`, while leaving the full `UNREVIEWED records` section intact.
+
+## Consistency Findings
+
+The two helpers tell a coherent story.
+
+- `check_queue_readiness.py` says the queue is only V1-ready up to prompt `001`, so the next ready prompt is `002`.
+- `list_review_backlog.py` says the current latest records needing review are `011` through `015`.
+
+Those outputs are not contradictory. They are answering different bounded questions:
+
+- readiness asks whether the reviewed V1 queue may advance from the start of the prompt sequence
+- backlog asks which existing V1 run records still await human review
+
+The stale failed `001` record is the main point that could look inconsistent at first glance, but the helpers remain coherent:
+
+- backlog includes it in the all-unreviewed list because it is still unreviewed
+- backlog excludes it from "likely needs human review next" because latest `001` is already accepted
+- readiness also ignores it for queue release because latest `001` is accepted
+
+## One Recommended Next Improvement
+
+The single smallest next improvement is to make `tools/codex/check_queue_readiness.py` print a short explanation when the default target is blocked by a missing V1 history gap, especially in a repo like this one where prompts `002` through `010` have legacy notes but no V1 execution records.
+
+Why this should happen next:
+
+- the helper is already making the correct decision
+- the current repo state makes that correct decision easy to misread
+- a one-line explanation would reduce confusion without changing queue policy or adding workflow complexity
+
+What should explicitly wait:
+
+- any change to queue rules
+- any backlog prioritization policy
+- any automatic bridging from legacy `__SUCCESS__` notes into V1 state
+- any broader dashboard or workflow-engine work

diff --git a/notes/016_next_improvement_recommendation__20260416_003710.md b/notes/016_next_improvement_recommendation__20260416_003710.md
new file mode 100644
index 0000000000000000000000000000000000000000..d740ec7dc33bce389afedfcd7f17f70f838ed245
--- /dev/null
+++ b/notes/016_next_improvement_recommendation__20260416_003710.md
@@ -0,0 +1,17 @@
+# 016 Next Improvement Recommendation
+
+Recommended next bounded improvement:
+
+- Add a short gap-explanation line to `tools/codex/check_queue_readiness.py` when the default target is blocked by missing V1 execution-record history for earlier prompts.
+
+Why it should happen next:
+
+- The helper is making the correct conservative decision now, but in the current repo state that decision is easy to misread because prompts `002` through `010` have legacy notes but no V1 execution records.
+- A small output clarification would improve operator understanding without changing any queue logic.
+
+What should explicitly wait:
+
+- any change to the V1 release rule
+- any migration or inference from legacy `__SUCCESS__` notes
+- any backlog prioritization feature
+- any broader queue engine or dashboard work
diff --git a/notes/016_queue_and_backlog_helper_validation__20260416_003710.md b/notes/016_queue_and_backlog_helper_validation__20260416_003710.md
new file mode 100644
index 0000000000000000000000000000000000000000..903db8f1f88e70d819fc10528076fa3e866759fa
--- /dev/null
+++ b/notes/016_queue_and_backlog_helper_validation__20260416_003710.md
@@ -0,0 +1,101 @@
+# 016 Queue And Backlog Helper Validation
+
+## Short Summary
+
+I validated `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py` against the current prompt order in `codex_prompts/` and the current V1 execution records in `notes/`.
+
+Both helpers behaved conservatively and consistently with the V1 design documents. The main visible usability issue is that the default queue-readiness result points back to prompt `002`, because prompts `002` through `010` do not have V1 execution records even though later prompts `011` through `015` do.
+
+Current V1 record evidence in `notes/`:
+
+- `notes/001_smoke_test_pipeline__20260415_233343.md`: `EXECUTION_FAILED`, `UNREVIEWED`
+- `notes/001_smoke_test_pipeline__20260415_234918.md`: `EXECUTED`, `ACCEPTED`
+- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/014_build_queue_readiness_checker__20260416_002319.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md`: `EXECUTED`, `UNREVIEWED`
+
+## Queue-Readiness Validation Findings
+
+Prompt ordering was correct. The helper discovered the current sequence `001` through `016` from `codex_prompts/` in numeric order.
+
+The default run of `python3 tools/codex/check_queue_readiness.py` selected `codex_prompts/002_repo_inventory_and_status.md` as the target, with `001` as the previous prompt and `notes/001_smoke_test_pipeline__20260415_234918.md` as the latest prior V1 record. That is correct under the V1 rule set: the latest V1 run for `001` is `ACCEPTED`, and there is no V1 run evidence for `002`, so `002` is the first unreleased prompt in sequence.
+
+The prompt-specific checks also matched the repo evidence:
+
+- `--prompt 001`: `Ready: YES`, because the first prompt has no prior gate.
+- `--prompt 002`: `Ready: YES`, because latest `001` is `ACCEPTED`.
+- `--prompt 013`: `Ready: NO`, because latest `012` is `UNREVIEWED` in `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`.
+- `--prompt 014`: `Ready: NO`, because latest `013` is `UNREVIEWED` in `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`.
+- `--prompt 015`: `Ready: NO`, because latest `014` is `UNREVIEWED` in `notes/014_build_queue_readiness_checker__20260416_002319.md`.
+
+This matches the V1 rule in `tools/codex/V1_Run_Review_Gate.md`: only an `ACCEPTED` latest run for the immediately previous prompt releases the next prompt.
+
+Visible edge case from the current repo state:
+
+- The default target being `002` is correct but potentially surprising, because later prompts `011` through `015` already have V1 execution records. The helper is correctly ignoring older non-V1 `__SUCCESS__` notes for prompts `002` through `010`, but that gap is not explained in the output.
+
+## Review-Backlog Validation Findings
+
+`python3 tools/codex/list_review_backlog.py` reported:
+
+- `Discovered V1 execution records: 7`
+- `Unreviewed records: 6`
+- `Prompts with latest record: 6`
+
+Those counts match the current repo evidence exactly: there are 7 V1 records total, 6 of them are still `UNREVIEWED`, and those 7 records cover 6 prompts because prompt `001` has two V1 records.
+
+Its latest-per-prompt selection was also correct:
+
+- For `001`, it chose `notes/001_smoke_test_pipeline__20260415_234918.md`, which is newer than the failed `001` run and has `review_status: ACCEPTED`.
+- For `011` through `015`, each prompt has one V1 record and each latest record is `UNREVIEWED`.
+
+Its `UNREVIEWED records` section also behaved correctly. It includes the older failed `001` run because that record is still literally `UNREVIEWED`, even though it is no longer the latest record for prompt `001`.
+
+Its `Likely needs human review next` view is the right conservative reduction of the backlog for the current repo state. It lists only prompts whose latest record is `UNREVIEWED`, which means:
+
+- `011`
+- `012`
+- `013`
+- `014`
+- `015`
+
+That matches the actual latest-record evidence.
+
+The `--unreviewed-only` mode also behaved consistently. It filtered the `Latest record per prompt` section down to latest records that are themselves `UNREVIEWED`, while leaving the full `UNREVIEWED records` section intact.
+
+## Consistency Findings
+
+The two helpers tell a coherent story.
+
+- `check_queue_readiness.py` says the queue is only V1-ready up to prompt `001`, so the next ready prompt is `002`.
+- `list_review_backlog.py` says the current latest records needing review are `011` through `015`.
+
+Those outputs are not contradictory. They are answering different bounded questions:
+
+- readiness asks whether the reviewed V1 queue may advance from the start of the prompt sequence
+- backlog asks which existing V1 run records still await human review
+
+The stale failed `001` record is the main point that could look inconsistent at first glance, but the helpers remain coherent:
+
+- backlog includes it in the all-unreviewed list because it is still unreviewed
+- backlog excludes it from "likely needs human review next" because latest `001` is already accepted
+- readiness also ignores it for queue release because latest `001` is accepted
+
+## One Recommended Next Improvement
+
+The single smallest next improvement is to make `tools/codex/check_queue_readiness.py` print a short explanation when the default target is blocked by a missing V1 history gap, especially in a repo like this one where prompts `002` through `010` have legacy notes but no V1 execution records.
+
+Why this should happen next:
+
+- the helper is already making the correct decision
+- the current repo state makes that correct decision easy to misread
+- a one-line explanation would reduce confusion without changing queue policy or adding workflow complexity
+
+What should explicitly wait:
+
+- any change to queue rules
+- any backlog prioritization policy
+- any automatic bridging from legacy `__SUCCESS__` notes into V1 state
+- any broader dashboard or workflow-engine work

codex
The two requested artifacts are in place. I’m doing a quick final check on the new note paths and then I’ll close with the concise validation outcome.
exec
/bin/bash -lc 'ls -1 notes/016_queue_and_backlog_helper_validation__20260416_003710.md notes/016_next_improvement_recommendation__20260416_003710.md' in /data/git/handson-ml3
 succeeded in 0ms:
notes/016_next_improvement_recommendation__20260416_003710.md
notes/016_queue_and_backlog_helper_validation__20260416_003710.md

exec
/bin/bash -lc "sed -n '1,220p' notes/016_queue_and_backlog_helper_validation__20260416_003710.md && printf '\\n---\\n' && sed -n '1,200p' notes/016_next_improvement_recommendation__20260416_003710.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 016 Queue And Backlog Helper Validation

## Short Summary

I validated `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py` against the current prompt order in `codex_prompts/` and the current V1 execution records in `notes/`.

Both helpers behaved conservatively and consistently with the V1 design documents. The main visible usability issue is that the default queue-readiness result points back to prompt `002`, because prompts `002` through `010` do not have V1 execution records even though later prompts `011` through `015` do.

Current V1 record evidence in `notes/`:

- `notes/001_smoke_test_pipeline__20260415_233343.md`: `EXECUTION_FAILED`, `UNREVIEWED`
- `notes/001_smoke_test_pipeline__20260415_234918.md`: `EXECUTED`, `ACCEPTED`
- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`: `EXECUTED`, `UNREVIEWED`
- `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`: `EXECUTED`, `UNREVIEWED`
- `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`: `EXECUTED`, `UNREVIEWED`
- `notes/014_build_queue_readiness_checker__20260416_002319.md`: `EXECUTED`, `UNREVIEWED`
- `notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md`: `EXECUTED`, `UNREVIEWED`

## Queue-Readiness Validation Findings

Prompt ordering was correct. The helper discovered the current sequence `001` through `016` from `codex_prompts/` in numeric order.

The default run of `python3 tools/codex/check_queue_readiness.py` selected `codex_prompts/002_repo_inventory_and_status.md` as the target, with `001` as the previous prompt and `notes/001_smoke_test_pipeline__20260415_234918.md` as the latest prior V1 record. That is correct under the V1 rule set: the latest V1 run for `001` is `ACCEPTED`, and there is no V1 run evidence for `002`, so `002` is the first unreleased prompt in sequence.

The prompt-specific checks also matched the repo evidence:

- `--prompt 001`: `Ready: YES`, because the first prompt has no prior gate.
- `--prompt 002`: `Ready: YES`, because latest `001` is `ACCEPTED`.
- `--prompt 013`: `Ready: NO`, because latest `012` is `UNREVIEWED` in `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`.
- `--prompt 014`: `Ready: NO`, because latest `013` is `UNREVIEWED` in `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`.
- `--prompt 015`: `Ready: NO`, because latest `014` is `UNREVIEWED` in `notes/014_build_queue_readiness_checker__20260416_002319.md`.

This matches the V1 rule in `tools/codex/V1_Run_Review_Gate.md`: only an `ACCEPTED` latest run for the immediately previous prompt releases the next prompt.

Visible edge case from the current repo state:

- The default target being `002` is correct but potentially surprising, because later prompts `011` through `015` already have V1 execution records. The helper is correctly ignoring older non-V1 `__SUCCESS__` notes for prompts `002` through `010`, but that gap is not explained in the output.

## Review-Backlog Validation Findings

`python3 tools/codex/list_review_backlog.py` reported:

- `Discovered V1 execution records: 7`
- `Unreviewed records: 6`
- `Prompts with latest record: 6`

Those counts match the current repo evidence exactly: there are 7 V1 records total, 6 of them are still `UNREVIEWED`, and those 7 records cover 6 prompts because prompt `001` has two V1 records.

Its latest-per-prompt selection was also correct:

- For `001`, it chose `notes/001_smoke_test_pipeline__20260415_234918.md`, which is newer than the failed `001` run and has `review_status: ACCEPTED`.
- For `011` through `015`, each prompt has one V1 record and each latest record is `UNREVIEWED`.

Its `UNREVIEWED records` section also behaved correctly. It includes the older failed `001` run because that record is still literally `UNREVIEWED`, even though it is no longer the latest record for prompt `001`.

Its `Likely needs human review next` view is the right conservative reduction of the backlog for the current repo state. It lists only prompts whose latest record is `UNREVIEWED`, which means:

- `011`
- `012`
- `013`
- `014`
- `015`

That matches the actual latest-record evidence.

The `--unreviewed-only` mode also behaved consistently. It filtered the `Latest record per prompt` section down to latest records that are themselves `UNREVIEWED`, while leaving the full `UNREVIEWED records` section intact.

## Consistency Findings

The two helpers tell a coherent story.

- `check_queue_readiness.py` says the queue is only V1-ready up to prompt `001`, so the next ready prompt is `002`.
- `list_review_backlog.py` says the current latest records needing review are `011` through `015`.

Those outputs are not contradictory. They are answering different bounded questions:

- readiness asks whether the reviewed V1 queue may advance from the start of the prompt sequence
- backlog asks which existing V1 run records still await human review

The stale failed `001` record is the main point that could look inconsistent at first glance, but the helpers remain coherent:

- backlog includes it in the all-unreviewed list because it is still unreviewed
- backlog excludes it from "likely needs human review next" because latest `001` is already accepted
- readiness also ignores it for queue release because latest `001` is accepted

## One Recommended Next Improvement

The single smallest next improvement is to make `tools/codex/check_queue_readiness.py` print a short explanation when the default target is blocked by a missing V1 history gap, especially in a repo like this one where prompts `002` through `010` have legacy notes but no V1 execution records.

Why this should happen next:

- the helper is already making the correct decision
- the current repo state makes that correct decision easy to misread
- a one-line explanation would reduce confusion without changing queue policy or adding workflow complexity

What should explicitly wait:

- any change to queue rules
- any backlog prioritization policy
- any automatic bridging from legacy `__SUCCESS__` notes into V1 state
- any broader dashboard or workflow-engine work

---
# 016 Next Improvement Recommendation

Recommended next bounded improvement:

- Add a short gap-explanation line to `tools/codex/check_queue_readiness.py` when the default target is blocked by missing V1 execution-record history for earlier prompts.

Why it should happen next:

- The helper is making the correct conservative decision now, but in the current repo state that decision is easy to misread because prompts `002` through `010` have legacy notes but no V1 execution records.
- A small output clarification would improve operator understanding without changing any queue logic.

What should explicitly wait:

- any change to the V1 release rule
- any migration or inference from legacy `__SUCCESS__` notes
- any backlog prioritization feature
- any broader queue engine or dashboard work

diff --git a/notes/016_next_improvement_recommendation__20260416_003710.md b/notes/016_next_improvement_recommendation__20260416_003710.md
new file mode 100644
index 0000000000000000000000000000000000000000..d740ec7dc33bce389afedfcd7f17f70f838ed245
--- /dev/null
+++ b/notes/016_next_improvement_recommendation__20260416_003710.md
@@ -0,0 +1,17 @@
+# 016 Next Improvement Recommendation
+
+Recommended next bounded improvement:
+
+- Add a short gap-explanation line to `tools/codex/check_queue_readiness.py` when the default target is blocked by missing V1 execution-record history for earlier prompts.
+
+Why it should happen next:
+
+- The helper is making the correct conservative decision now, but in the current repo state that decision is easy to misread because prompts `002` through `010` have legacy notes but no V1 execution records.
+- A small output clarification would improve operator understanding without changing any queue logic.
+
+What should explicitly wait:
+
+- any change to the V1 release rule
+- any migration or inference from legacy `__SUCCESS__` notes
+- any backlog prioritization feature
+- any broader queue engine or dashboard work
diff --git a/notes/016_queue_and_backlog_helper_validation__20260416_003710.md b/notes/016_queue_and_backlog_helper_validation__20260416_003710.md
new file mode 100644
index 0000000000000000000000000000000000000000..903db8f1f88e70d819fc10528076fa3e866759fa
--- /dev/null
+++ b/notes/016_queue_and_backlog_helper_validation__20260416_003710.md
@@ -0,0 +1,101 @@
+# 016 Queue And Backlog Helper Validation
+
+## Short Summary
+
+I validated `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py` against the current prompt order in `codex_prompts/` and the current V1 execution records in `notes/`.
+
+Both helpers behaved conservatively and consistently with the V1 design documents. The main visible usability issue is that the default queue-readiness result points back to prompt `002`, because prompts `002` through `010` do not have V1 execution records even though later prompts `011` through `015` do.
+
+Current V1 record evidence in `notes/`:
+
+- `notes/001_smoke_test_pipeline__20260415_233343.md`: `EXECUTION_FAILED`, `UNREVIEWED`
+- `notes/001_smoke_test_pipeline__20260415_234918.md`: `EXECUTED`, `ACCEPTED`
+- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/014_build_queue_readiness_checker__20260416_002319.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md`: `EXECUTED`, `UNREVIEWED`
+
+## Queue-Readiness Validation Findings
+
+Prompt ordering was correct. The helper discovered the current sequence `001` through `016` from `codex_prompts/` in numeric order.
+
+The default run of `python3 tools/codex/check_queue_readiness.py` selected `codex_prompts/002_repo_inventory_and_status.md` as the target, with `001` as the previous prompt and `notes/001_smoke_test_pipeline__20260415_234918.md` as the latest prior V1 record. That is correct under the V1 rule set: the latest V1 run for `001` is `ACCEPTED`, and there is no V1 run evidence for `002`, so `002` is the first unreleased prompt in sequence.
+
+The prompt-specific checks also matched the repo evidence:
+
+- `--prompt 001`: `Ready: YES`, because the first prompt has no prior gate.
+- `--prompt 002`: `Ready: YES`, because latest `001` is `ACCEPTED`.
+- `--prompt 013`: `Ready: NO`, because latest `012` is `UNREVIEWED` in `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`.
+- `--prompt 014`: `Ready: NO`, because latest `013` is `UNREVIEWED` in `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`.
+- `--prompt 015`: `Ready: NO`, because latest `014` is `UNREVIEWED` in `notes/014_build_queue_readiness_checker__20260416_002319.md`.
+
+This matches the V1 rule in `tools/codex/V1_Run_Review_Gate.md`: only an `ACCEPTED` latest run for the immediately previous prompt releases the next prompt.
+
+Visible edge case from the current repo state:
+
+- The default target being `002` is correct but potentially surprising, because later prompts `011` through `015` already have V1 execution records. The helper is correctly ignoring older non-V1 `__SUCCESS__` notes for prompts `002` through `010`, but that gap is not explained in the output.
+
+## Review-Backlog Validation Findings
+
+`python3 tools/codex/list_review_backlog.py` reported:
+
+- `Discovered V1 execution records: 7`
+- `Unreviewed records: 6`
+- `Prompts with latest record: 6`
+
+Those counts match the current repo evidence exactly: there are 7 V1 records total, 6 of them are still `UNREVIEWED`, and those 7 records cover 6 prompts because prompt `001` has two V1 records.
+
+Its latest-per-prompt selection was also correct:
+
+- For `001`, it chose `notes/001_smoke_test_pipeline__20260415_234918.md`, which is newer than the failed `001` run and has `review_status: ACCEPTED`.
+- For `011` through `015`, each prompt has one V1 record and each latest record is `UNREVIEWED`.
+
+Its `UNREVIEWED records` section also behaved correctly. It includes the older failed `001` run because that record is still literally `UNREVIEWED`, even though it is no longer the latest record for prompt `001`.
+
+Its `Likely needs human review next` view is the right conservative reduction of the backlog for the current repo state. It lists only prompts whose latest record is `UNREVIEWED`, which means:
+
+- `011`
+- `012`
+- `013`
+- `014`
+- `015`
+
+That matches the actual latest-record evidence.
+
+The `--unreviewed-only` mode also behaved consistently. It filtered the `Latest record per prompt` section down to latest records that are themselves `UNREVIEWED`, while leaving the full `UNREVIEWED records` section intact.
+
+## Consistency Findings
+
+The two helpers tell a coherent story.
+
+- `check_queue_readiness.py` says the queue is only V1-ready up to prompt `001`, so the next ready prompt is `002`.
+- `list_review_backlog.py` says the current latest records needing review are `011` through `015`.
+
+Those outputs are not contradictory. They are answering different bounded questions:
+
+- readiness asks whether the reviewed V1 queue may advance from the start of the prompt sequence
+- backlog asks which existing V1 run records still await human review
+
+The stale failed `001` record is the main point that could look inconsistent at first glance, but the helpers remain coherent:
+
+- backlog includes it in the all-unreviewed list because it is still unreviewed
+- backlog excludes it from "likely needs human review next" because latest `001` is already accepted
+- readiness also ignores it for queue release because latest `001` is accepted
+
+## One Recommended Next Improvement
+
+The single smallest next improvement is to make `tools/codex/check_queue_readiness.py` print a short explanation when the default target is blocked by a missing V1 history gap, especially in a repo like this one where prompts `002` through `010` have legacy notes but no V1 execution records.
+
+Why this should happen next:
+
+- the helper is already making the correct decision
+- the current repo state makes that correct decision easy to misread
+- a one-line explanation would reduce confusion without changing queue policy or adding workflow complexity
+
+What should explicitly wait:
+
+- any change to queue rules
+- any backlog prioritization policy
+- any automatic bridging from legacy `__SUCCESS__` notes into V1 state
+- any broader dashboard or workflow-engine work

codex
Created the two requested artifacts:

- [016_queue_and_backlog_helper_validation__20260416_003710.md](/data/git/handson-ml3/notes/016_queue_and_backlog_helper_validation__20260416_003710.md)
- [016_next_improvement_recommendation__20260416_003710.md](/data/git/handson-ml3/notes/016_next_improvement_recommendation__20260416_003710.md)

Validation result: both helpers behave correctly and conservatively against the current repo state. `check_queue_readiness.py` correctly treats `002` as the next ready prompt because only `001` has a latest `ACCEPTED` V1 record, and `list_review_backlog.py` correctly distinguishes the stale unreviewed failed `001` run from the newer accepted `001` run by keeping it in the all-unreviewed list but excluding it from latest-per-prompt and likely-needs-review-next.

The single recommended next step is a small output clarification in `tools/codex/check_queue_readiness.py` so it explains missing V1 history gaps for earlier prompts like `002` through `010`, which currently have legacy notes but no V1 execution records.
diff --git a/notes/016_next_improvement_recommendation__20260416_003710.md b/notes/016_next_improvement_recommendation__20260416_003710.md
new file mode 100644
index 0000000000000000000000000000000000000000..d740ec7dc33bce389afedfcd7f17f70f838ed245
--- /dev/null
+++ b/notes/016_next_improvement_recommendation__20260416_003710.md
@@ -0,0 +1,17 @@
+# 016 Next Improvement Recommendation
+
+Recommended next bounded improvement:
+
+- Add a short gap-explanation line to `tools/codex/check_queue_readiness.py` when the default target is blocked by missing V1 execution-record history for earlier prompts.
+
+Why it should happen next:
+
+- The helper is making the correct conservative decision now, but in the current repo state that decision is easy to misread because prompts `002` through `010` have legacy notes but no V1 execution records.
+- A small output clarification would improve operator understanding without changing any queue logic.
+
+What should explicitly wait:
+
+- any change to the V1 release rule
+- any migration or inference from legacy `__SUCCESS__` notes
+- any backlog prioritization feature
+- any broader queue engine or dashboard work
diff --git a/notes/016_queue_and_backlog_helper_validation__20260416_003710.md b/notes/016_queue_and_backlog_helper_validation__20260416_003710.md
new file mode 100644
index 0000000000000000000000000000000000000000..903db8f1f88e70d819fc10528076fa3e866759fa
--- /dev/null
+++ b/notes/016_queue_and_backlog_helper_validation__20260416_003710.md
@@ -0,0 +1,101 @@
+# 016 Queue And Backlog Helper Validation
+
+## Short Summary
+
+I validated `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py` against the current prompt order in `codex_prompts/` and the current V1 execution records in `notes/`.
+
+Both helpers behaved conservatively and consistently with the V1 design documents. The main visible usability issue is that the default queue-readiness result points back to prompt `002`, because prompts `002` through `010` do not have V1 execution records even though later prompts `011` through `015` do.
+
+Current V1 record evidence in `notes/`:
+
+- `notes/001_smoke_test_pipeline__20260415_233343.md`: `EXECUTION_FAILED`, `UNREVIEWED`
+- `notes/001_smoke_test_pipeline__20260415_234918.md`: `EXECUTED`, `ACCEPTED`
+- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/014_build_queue_readiness_checker__20260416_002319.md`: `EXECUTED`, `UNREVIEWED`
+- `notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md`: `EXECUTED`, `UNREVIEWED`
+
+## Queue-Readiness Validation Findings
+
+Prompt ordering was correct. The helper discovered the current sequence `001` through `016` from `codex_prompts/` in numeric order.
+
+The default run of `python3 tools/codex/check_queue_readiness.py` selected `codex_prompts/002_repo_inventory_and_status.md` as the target, with `001` as the previous prompt and `notes/001_smoke_test_pipeline__20260415_234918.md` as the latest prior V1 record. That is correct under the V1 rule set: the latest V1 run for `001` is `ACCEPTED`, and there is no V1 run evidence for `002`, so `002` is the first unreleased prompt in sequence.
+
+The prompt-specific checks also matched the repo evidence:
+
+- `--prompt 001`: `Ready: YES`, because the first prompt has no prior gate.
+- `--prompt 002`: `Ready: YES`, because latest `001` is `ACCEPTED`.
+- `--prompt 013`: `Ready: NO`, because latest `012` is `UNREVIEWED` in `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`.
+- `--prompt 014`: `Ready: NO`, because latest `013` is `UNREVIEWED` in `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`.
+- `--prompt 015`: `Ready: NO`, because latest `014` is `UNREVIEWED` in `notes/014_build_queue_readiness_checker__20260416_002319.md`.
+
+This matches the V1 rule in `tools/codex/V1_Run_Review_Gate.md`: only an `ACCEPTED` latest run for the immediately previous prompt releases the next prompt.
+
+Visible edge case from the current repo state:
+
+- The default target being `002` is correct but potentially surprising, because later prompts `011` through `015` already have V1 execution records. The helper is correctly ignoring older non-V1 `__SUCCESS__` notes for prompts `002` through `010`, but that gap is not explained in the output.
+
+## Review-Backlog Validation Findings
+
+`python3 tools/codex/list_review_backlog.py` reported:
+
+- `Discovered V1 execution records: 7`
+- `Unreviewed records: 6`
+- `Prompts with latest record: 6`
+
+Those counts match the current repo evidence exactly: there are 7 V1 records total, 6 of them are still `UNREVIEWED`, and those 7 records cover 6 prompts because prompt `001` has two V1 records.
+
+Its latest-per-prompt selection was also correct:
+
+- For `001`, it chose `notes/001_smoke_test_pipeline__20260415_234918.md`, which is newer than the failed `001` run and has `review_status: ACCEPTED`.
+- For `011` through `015`, each prompt has one V1 record and each latest record is `UNREVIEWED`.
+
+Its `UNREVIEWED records` section also behaved correctly. It includes the older failed `001` run because that record is still literally `UNREVIEWED`, even though it is no longer the latest record for prompt `001`.
+
+Its `Likely needs human review next` view is the right conservative reduction of the backlog for the current repo state. It lists only prompts whose latest record is `UNREVIEWED`, which means:
+
+- `011`
+- `012`
+- `013`
+- `014`
+- `015`
+
+That matches the actual latest-record evidence.
+
+The `--unreviewed-only` mode also behaved consistently. It filtered the `Latest record per prompt` section down to latest records that are themselves `UNREVIEWED`, while leaving the full `UNREVIEWED records` section intact.
+
+## Consistency Findings
+
+The two helpers tell a coherent story.
+
+- `check_queue_readiness.py` says the queue is only V1-ready up to prompt `001`, so the next ready prompt is `002`.
+- `list_review_backlog.py` says the current latest records needing review are `011` through `015`.
+
+Those outputs are not contradictory. They are answering different bounded questions:
+
+- readiness asks whether the reviewed V1 queue may advance from the start of the prompt sequence
+- backlog asks which existing V1 run records still await human review
+
+The stale failed `001` record is the main point that could look inconsistent at first glance, but the helpers remain coherent:
+
+- backlog includes it in the all-unreviewed list because it is still unreviewed
+- backlog excludes it from "likely needs human review next" because latest `001` is already accepted
+- readiness also ignores it for queue release because latest `001` is accepted
+
+## One Recommended Next Improvement
+
+The single smallest next improvement is to make `tools/codex/check_queue_readiness.py` print a short explanation when the default target is blocked by a missing V1 history gap, especially in a repo like this one where prompts `002` through `010` have legacy notes but no V1 execution records.
+
+Why this should happen next:
+
+- the helper is already making the correct decision
+- the current repo state makes that correct decision easy to misread
+- a one-line explanation would reduce confusion without changing queue policy or adding workflow complexity
+
+What should explicitly wait:
+
+- any change to queue rules
+- any backlog prioritization policy
+- any automatic bridging from legacy `__SUCCESS__` notes into V1 state
+- any broader dashboard or workflow-engine work

tokens used
64,501
```
