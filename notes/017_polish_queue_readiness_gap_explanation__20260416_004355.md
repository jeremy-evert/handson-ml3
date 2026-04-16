# 017_polish_queue_readiness_gap_explanation__20260416_004355

- run_id: `017_polish_queue_readiness_gap_explanation__20260416_004355`
- prompt_file: `codex_prompts/017_polish_queue_readiness_gap_explanation.md`
- prompt_stem: `017_polish_queue_readiness_gap_explanation`
- started_at_utc: `20260416_004355`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260416_004523`
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

- elapsed_seconds: `87.766`
- final_output_char_count: `1060`
- stderr_char_count: `96412`

## Prompt Text

```md
# Task: Polish the queue-readiness checker output for missing V1 history gaps

You are working in this repository.

Your task is to make one small usability improvement to the queue-readiness helper.

## Important framing

This is a small polish task.

Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/list_review_backlog.py`

Do NOT rewrite design documents in this pass.

You may modify only:

- `tools/codex/check_queue_readiness.py`

and create the required implementation note.

## Files to inspect

Read these exact files before making changes:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/check_queue_readiness.py`
- `notes/016_queue_and_backlog_helper_validation__20260416_003710.md`
- `notes/016_next_improvement_recommendation__20260416_003710.md`

Also inspect current repo contents in:

- `codex_prompts/`
- `notes/`

## Goal

Improve the stdout output of `tools/codex/check_queue_readiness.py` so that when the default target is chosen conservatively because earlier prompts have no V1 execution-record history, the output includes a short explanation that makes this easier for a human to understand.

This is a usability clarification, not a policy change.

## Problem to address

The current helper is making the correct conservative decision, but in the current repo state the default result can be confusing because:

- prompts `002` through `010` have legacy `__SUCCESS__` notes
- those legacy notes are not V1 execution records
- the helper correctly ignores them
- the default output does not currently explain that gap clearly enough

## Required behavior change

Keep the current readiness logic unchanged.

Add a short explanatory line or small explanatory block to the human-readable output when all of the following are true:

1. the helper is using default target selection
2. the chosen target is blocked or selected because prior prompts lack V1 execution-record history
3. legacy-looking success notes or otherwise non-V1 history may make the result surprising to a human

The explanation should stay small, practical, and inspectable.

It should clarify that:

- the helper is using only V1 execution records in `notes/`
- legacy `__SUCCESS__` notes do not count as V1 queue history
- this is why the default target may point back to an earlier prompt than a human might first expect

## Important constraints

### 1. Do not change the queue rule
Do NOT change:
- how prompt order is computed
- how latest records are selected
- the `ACCEPTED` / `UNREVIEWED` / `REJECTED` logic
- the meaning of missing prior V1 evidence

This pass is only about clearer output.

### 2. Keep output conservative and small
Do not add:
- dashboards
- verbose debug dumps
- broad repo scans in the output
- speculative migration behavior
- any automatic inference from legacy notes

### 3. Keep the helper read-only
Do not modify any files other than the helper source itself and the implementation note.

## Required artifacts

### Artifact 1
Update:

`tools/codex/check_queue_readiness.py`

### Artifact 2
Create a short implementation note at:

`notes/017_queue_readiness_gap_explanation_polish__TIMESTAMP.md`

This note should summarize:

- what output was changed
- why the change was made
- what logic was intentionally left unchanged
- what validation was performed

## Validation requirements

After the change, validate at least these points:

1. `python3 tools/codex/check_queue_readiness.py` still succeeds
2. the default output now includes a clearer explanation when the repo state includes missing V1 history for earlier prompts
3. `python3 tools/codex/check_queue_readiness.py --prompt 002` still reports readiness correctly
4. `python3 tools/codex/check_queue_readiness.py --prompt 013` still reports not-ready correctly
5. `tools/codex/list_review_backlog.py` remains unchanged
6. `tools/codex/run_prompt.py` remains unchanged
7. `tools/codex/review_run.py` remains unchanged
8. `tools/codex/baby_run_prompt.py` remains unchanged

Include the validation outcome in the implementation note.

## Success criteria

This task is successful if:

- `tools/codex/check_queue_readiness.py` keeps the same readiness logic
- the default output is easier for a human to understand in the current repo state
- the helper stays small, read-only, and conservative
- the implementation note clearly explains the polish and validation
```

## Codex Final Output

Updated [tools/codex/check_queue_readiness.py](/data/git/handson-ml3/tools/codex/check_queue_readiness.py) to add a small `Queue note:` line in the default-target path when missing earlier V1 history is likely to be confusing because legacy `__SUCCESS__` notes exist. The readiness logic itself is unchanged.

The new output now explains that only V1 execution records in `notes/` count for queue history, that legacy `__SUCCESS__` notes do not, and that this is why the default target can point back to an earlier prompt such as `002`. I also added the implementation note at [notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md](/data/git/handson-ml3/notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md).

Validation passed: default invocation still succeeds and now shows the clarification, `--prompt 002` still reports ready, `--prompt 013` still reports not ready, and `tools/codex/baby_run_prompt.py`, `tools/codex/run_prompt.py`, `tools/codex/review_run.py`, and `tools/codex/list_review_backlog.py` remained unchanged.

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
session id: 019d93be-9e5f-75e2-a182-92eceb630882
--------
user
# Task: Polish the queue-readiness checker output for missing V1 history gaps

You are working in this repository.

Your task is to make one small usability improvement to the queue-readiness helper.

## Important framing

This is a small polish task.

Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/list_review_backlog.py`

Do NOT rewrite design documents in this pass.

You may modify only:

- `tools/codex/check_queue_readiness.py`

and create the required implementation note.

## Files to inspect

Read these exact files before making changes:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/check_queue_readiness.py`
- `notes/016_queue_and_backlog_helper_validation__20260416_003710.md`
- `notes/016_next_improvement_recommendation__20260416_003710.md`

Also inspect current repo contents in:

- `codex_prompts/`
- `notes/`

## Goal

Improve the stdout output of `tools/codex/check_queue_readiness.py` so that when the default target is chosen conservatively because earlier prompts have no V1 execution-record history, the output includes a short explanation that makes this easier for a human to understand.

This is a usability clarification, not a policy change.

## Problem to address

The current helper is making the correct conservative decision, but in the current repo state the default result can be confusing because:

- prompts `002` through `010` have legacy `__SUCCESS__` notes
- those legacy notes are not V1 execution records
- the helper correctly ignores them
- the default output does not currently explain that gap clearly enough

## Required behavior change

Keep the current readiness logic unchanged.

Add a short explanatory line or small explanatory block to the human-readable output when all of the following are true:

1. the helper is using default target selection
2. the chosen target is blocked or selected because prior prompts lack V1 execution-record history
3. legacy-looking success notes or otherwise non-V1 history may make the result surprising to a human

The explanation should stay small, practical, and inspectable.

It should clarify that:

- the helper is using only V1 execution records in `notes/`
- legacy `__SUCCESS__` notes do not count as V1 queue history
- this is why the default target may point back to an earlier prompt than a human might first expect

## Important constraints

### 1. Do not change the queue rule
Do NOT change:
- how prompt order is computed
- how latest records are selected
- the `ACCEPTED` / `UNREVIEWED` / `REJECTED` logic
- the meaning of missing prior V1 evidence

This pass is only about clearer output.

### 2. Keep output conservative and small
Do not add:
- dashboards
- verbose debug dumps
- broad repo scans in the output
- speculative migration behavior
- any automatic inference from legacy notes

### 3. Keep the helper read-only
Do not modify any files other than the helper source itself and the implementation note.

## Required artifacts

### Artifact 1
Update:

`tools/codex/check_queue_readiness.py`

### Artifact 2
Create a short implementation note at:

`notes/017_queue_readiness_gap_explanation_polish__TIMESTAMP.md`

This note should summarize:

- what output was changed
- why the change was made
- what logic was intentionally left unchanged
- what validation was performed

## Validation requirements

After the change, validate at least these points:

1. `python3 tools/codex/check_queue_readiness.py` still succeeds
2. the default output now includes a clearer explanation when the repo state includes missing V1 history for earlier prompts
3. `python3 tools/codex/check_queue_readiness.py --prompt 002` still reports readiness correctly
4. `python3 tools/codex/check_queue_readiness.py --prompt 013` still reports not-ready correctly
5. `tools/codex/list_review_backlog.py` remains unchanged
6. `tools/codex/run_prompt.py` remains unchanged
7. `tools/codex/review_run.py` remains unchanged
8. `tools/codex/baby_run_prompt.py` remains unchanged

Include the validation outcome in the implementation note.

## Success criteria

This task is successful if:

- `tools/codex/check_queue_readiness.py` keeps the same readiness logic
- the default output is easier for a human to understand in the current repo state
- the helper stays small, read-only, and conservative
- the implementation note clearly explains the polish and validation

codex
I’m inspecting the specified workflow docs, the helper, and the current `codex_prompts/` and `notes/` state first so I can keep the logic untouched and limit this to output polish.
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
/bin/bash -lc "pwd && rg --files codex_prompts notes tools | sed -n '1,240p'" in /data/git/handson-ml3
 succeeded in 0ms:
/data/git/handson-ml3
tools/Codex_Prompt_Workflow_Architecture.md
tools/Project_Design_Workflow.md
codex_prompts/017_polish_queue_readiness_gap_explanation.md
codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md
codex_prompts/015_build_review_backlog_unreviewed_run_lister.md
codex_prompts/014_build_queue_readiness_checker.md
codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md
codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md
codex_prompts/011_build_v1_review_writeback_helper.md
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
notes/005_prompt_queue_plan__20260415_202557.md
notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md
notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md
notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md
notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md
notes/004_next_design_step_recommendation__20260415_195538.md
notes/003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md
notes/004_architecture_and_bridge_runner_review__20260415_195538.md
notes/001_smoke_test_pipeline__20260415_233343.md
notes/010_run_prompt_candidate_review__20260415_234559.md
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md
notes/009_run_prompt_candidate_build__20260415_233407.md
notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md
notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md
notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md
notes/010_next_step_recommendation__20260415_234559.md
notes/011_review_writeback_helper_build__20260415_235514.md
notes/001_smoke_test_pipeline__20260415_234918.md
notes/011_build_v1_review_writeback_helper__20260415_235346.md
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md
tools/codex/baby_run_prompt.py
tools/codex/list_review_backlog.py
tools/codex/check_queue_readiness.py
tools/codex/review_run.py
tools/codex/run_prompt.py
tools/codex/V1_Bridge_Runner_Change_Spec.md
tools/codex/V1_Run_Review_Gate.md
tools/codex/V1_Execution_Record_Artifact.md
notes/014_queue_readiness_checker_build__20260416_002419.md
notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md
notes/013_next_two_prompt_plan__20260416_002005.md
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md
notes/012_top_three_next_options__20260416_000819.md
notes/012_v1_pipeline_options_review__20260416_000819.md
notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md
notes/015_review_backlog_lister_build__20260416_010500.md
notes/014_build_queue_readiness_checker__20260416_002319.md
notes/016_next_improvement_recommendation__20260416_003710.md
notes/016_queue_and_backlog_helper_validation__20260416_003710.md
notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md

exec
/bin/bash -lc "sed -n '1,240p' tools/codex/V1_Execution_Record_Artifact.md" in /data/git/handson-ml3
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
/bin/bash -lc "sed -n '1,240p' notes/016_queue_and_backlog_helper_validation__20260416_003710.md" in /data/git/handson-ml3
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

exec
/bin/bash -lc "sed -n '1,240p' notes/016_next_improvement_recommendation__20260416_003710.md" in /data/git/handson-ml3
 succeeded in 0ms:
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

codex
I’ve confirmed the repo state matches the task framing: the helper is already conservative, and the gap is specifically about humans seeing old `__SUCCESS__` notes for `002` to `010`. I’m reading the rest of the helper now and then I’ll patch only its stdout path plus the required note.
exec
/bin/bash -lc "find codex_prompts -maxdepth 1 -type f | sort && printf '\\n---\\n' && find notes -maxdepth 1 -type f | sort" in /data/git/handson-ml3
 succeeded in 0ms:
codex_prompts/001_smoke_test_pipeline.md
codex_prompts/002_repo_inventory_and_status.md
codex_prompts/003_revise_Project_Deisgn_workflow_document.md
codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md
codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md
codex_prompts/006_define_v1_run_review_gate.md
codex_prompts/007_align_architecture_doc_to_v1_workflow.md
codex_prompts/008_define_minimal_bridge_runner_change_spec.md
codex_prompts/009_build_v1_run_prompt_candidate.md
codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md
codex_prompts/011_build_v1_review_writeback_helper.md
codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md
codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md
codex_prompts/014_build_queue_readiness_checker.md
codex_prompts/015_build_review_backlog_unreviewed_run_lister.md
codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md
codex_prompts/017_polish_queue_readiness_gap_explanation.md

---
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
notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md
notes/015_review_backlog_lister_build__20260416_010500.md
notes/016_next_improvement_recommendation__20260416_003710.md
notes/016_queue_and_backlog_helper_validation__20260416_003710.md
notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md

exec
/bin/bash -lc "sed -n '261,520p' tools/codex/check_queue_readiness.py" in /data/git/handson-ml3
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

codex
I have the helper’s current stdout path. I’m patching only that file now to add a small default-mode explanation that triggers when legacy `__SUCCESS__` notes could make a missing V1 history gap surprising, without touching readiness logic.
apply patch
patch: completed
/data/git/handson-ml3/tools/codex/check_queue_readiness.py
diff --git a/tools/codex/check_queue_readiness.py b/tools/codex/check_queue_readiness.py
index 1f183b056023e4f1158e440aad4aa8102b7cf816..c6eca9f62311a8daeaf744e2ceb8196ac317a597
--- a/tools/codex/check_queue_readiness.py
+++ b/tools/codex/check_queue_readiness.py
@@ -66,6 +66,10 @@
     pass
 
 
+def format_prefixes(prefixes: list[int]) -> str:
+    return ", ".join(f"{prefix:03d}" for prefix in prefixes)
+
+
 def repo_root() -> Path:
     return Path(__file__).resolve().parents[2]
 
@@ -244,6 +248,16 @@
     return records
 
 
+def discover_legacy_success_prefixes(root: Path) -> set[int]:
+    notes_dir = root / NOTES_DIR
+    prefixes: set[int] = set()
+    for path in sorted(notes_dir.glob("*__SUCCESS__*.md")):
+        match = PROMPT_NAME_RE.match(path.stem)
+        if match is not None:
+            prefixes.add(int(match.group("prefix")))
+    return prefixes
+
+
 def latest_record_for_prompt(records: list[RunRecord], prompt: PromptEntry) -> RunRecord | None:
     relevant = [record for record in records if record.prompt_file == prompt.label]
     if not relevant:
@@ -323,7 +337,42 @@
     )
 
 
-def print_summary(prompts: list[PromptEntry], result: ReadinessResult) -> None:
+def build_default_gap_explanation(
+    prompts: list[PromptEntry],
+    records: list[RunRecord],
+    legacy_success_prefixes: set[int],
+    target: PromptEntry,
+) -> str | None:
+    later_v1_prompts = [prompt for prompt in prompts if prompt.prefix > target.prefix]
+    if not any(latest_record_for_prompt(records, prompt) is not None for prompt in later_v1_prompts):
+        return None
+
+    gap_prefixes = [
+        prompt.prefix
+        for prompt in prompts
+        if prompt.prefix >= target.prefix and latest_record_for_prompt(records, prompt) is None
+    ]
+    if not gap_prefixes:
+        return None
+
+    surprising_prefixes = [prefix for prefix in gap_prefixes if prefix in legacy_success_prefixes]
+    if not surprising_prefixes:
+        return None
+
+    return (
+        "Queue note: default selection uses only V1 execution records in notes/. "
+        f"Legacy __SUCCESS__ notes do not count as V1 queue history, so prompts "
+        f"{format_prefixes(surprising_prefixes)} still count as missing V1 evidence "
+        "and can pull the default target earlier than older notes suggest."
+    )
+
+
+def print_summary(
+    prompts: list[PromptEntry],
+    result: ReadinessResult,
+    *,
+    default_gap_explanation: str | None = None,
+) -> None:
     print("Ordered prompts:")
     for prompt in prompts:
         print(f"- {prompt.prefix:03d}: {prompt.label}")
@@ -341,6 +390,8 @@
     print(f"Latest run review_status: {latest_review_status}")
     print(f"Ready: {'YES' if result.ready else 'NO'}")
     print(f"Reason: {result.reason}")
+    if default_gap_explanation:
+        print(default_gap_explanation)
 
 
 def main() -> int:
@@ -349,9 +400,23 @@
         root = repo_root()
         prompts = discover_prompts(root)
         records = discover_run_records(root)
+        using_default_target = args.prompt is None
         target = resolve_prompt(prompts, args.prompt) if args.prompt else default_target(prompts, records)
         result = evaluate_readiness(prompts, records, target)
-        print_summary(prompts, result)
+        default_gap_explanation = None
+        if using_default_target:
+            legacy_success_prefixes = discover_legacy_success_prefixes(root)
+            default_gap_explanation = build_default_gap_explanation(
+                prompts,
+                records,
+                legacy_success_prefixes,
+                target,
+            )
+        print_summary(
+            prompts,
+            result,
+            default_gap_explanation=default_gap_explanation,
+        )
         return 0
     except ReadinessError as exc:
         return fail(str(exc))

diff --git a/tools/codex/check_queue_readiness.py b/tools/codex/check_queue_readiness.py
index 1f183b056023e4f1158e440aad4aa8102b7cf816..c6eca9f62311a8daeaf744e2ceb8196ac317a597
--- a/tools/codex/check_queue_readiness.py
+++ b/tools/codex/check_queue_readiness.py
@@ -66,6 +66,10 @@
     pass
 
 
+def format_prefixes(prefixes: list[int]) -> str:
+    return ", ".join(f"{prefix:03d}" for prefix in prefixes)
+
+
 def repo_root() -> Path:
     return Path(__file__).resolve().parents[2]
 
@@ -244,6 +248,16 @@
     return records
 
 
+def discover_legacy_success_prefixes(root: Path) -> set[int]:
+    notes_dir = root / NOTES_DIR
+    prefixes: set[int] = set()
+    for path in sorted(notes_dir.glob("*__SUCCESS__*.md")):
+        match = PROMPT_NAME_RE.match(path.stem)
+        if match is not None:
+            prefixes.add(int(match.group("prefix")))
+    return prefixes
+
+
 def latest_record_for_prompt(records: list[RunRecord], prompt: PromptEntry) -> RunRecord | None:
     relevant = [record for record in records if record.prompt_file == prompt.label]
     if not relevant:
@@ -323,7 +337,42 @@
     )
 
 
-def print_summary(prompts: list[PromptEntry], result: ReadinessResult) -> None:
+def build_default_gap_explanation(
+    prompts: list[PromptEntry],
+    records: list[RunRecord],
+    legacy_success_prefixes: set[int],
+    target: PromptEntry,
+) -> str | None:
+    later_v1_prompts = [prompt for prompt in prompts if prompt.prefix > target.prefix]
+    if not any(latest_record_for_prompt(records, prompt) is not None for prompt in later_v1_prompts):
+        return None
+
+    gap_prefixes = [
+        prompt.prefix
+        for prompt in prompts
+        if prompt.prefix >= target.prefix and latest_record_for_prompt(records, prompt) is None
+    ]
+    if not gap_prefixes:
+        return None
+
+    surprising_prefixes = [prefix for prefix in gap_prefixes if prefix in legacy_success_prefixes]
+    if not surprising_prefixes:
+        return None
+
+    return (
+        "Queue note: default selection uses only V1 execution records in notes/. "
+        f"Legacy __SUCCESS__ notes do not count as V1 queue history, so prompts "
+        f"{format_prefixes(surprising_prefixes)} still count as missing V1 evidence "
+        "and can pull the default target earlier than older notes suggest."
+    )
+
+
+def print_summary(
+    prompts: list[PromptEntry],
+    result: ReadinessResult,
+    *,
+    default_gap_explanation: str | None = None,
+) -> None:
     print("Ordered prompts:")
     for prompt in prompts:
         print(f"- {prompt.prefix:03d}: {prompt.label}")
@@ -341,6 +390,8 @@
     print(f"Latest run review_status: {latest_review_status}")
     print(f"Ready: {'YES' if result.ready else 'NO'}")
     print(f"Reason: {result.reason}")
+    if default_gap_explanation:
+        print(default_gap_explanation)
 
 
 def main() -> int:
@@ -349,9 +400,23 @@
         root = repo_root()
         prompts = discover_prompts(root)
         records = discover_run_records(root)
+        using_default_target = args.prompt is None
         target = resolve_prompt(prompts, args.prompt) if args.prompt else default_target(prompts, records)
         result = evaluate_readiness(prompts, records, target)
-        print_summary(prompts, result)
+        default_gap_explanation = None
+        if using_default_target:
+            legacy_success_prefixes = discover_legacy_success_prefixes(root)
+            default_gap_explanation = build_default_gap_explanation(
+                prompts,
+                records,
+                legacy_success_prefixes,
+                target,
+            )
+        print_summary(
+            prompts,
+            result,
+            default_gap_explanation=default_gap_explanation,
+        )
         return 0
     except ReadinessError as exc:
         return fail(str(exc))

codex
The code change is in place. I’m running the required checks now, including the default output, the two prompt-specific cases, and verifying the protected files stayed untouched.
exec
/bin/bash -lc 'python3 tools/codex/check_queue_readiness.py' in /data/git/handson-ml3
exec
/bin/bash -lc 'python3 tools/codex/check_queue_readiness.py --prompt 002' in /data/git/handson-ml3
exec
/bin/bash -lc 'python3 tools/codex/check_queue_readiness.py --prompt 013' in /data/git/handson-ml3
exec
/bin/bash -lc 'git diff --name-only -- tools/codex/baby_run_prompt.py tools/codex/run_prompt.py tools/codex/review_run.py tools/codex/list_review_backlog.py' in /data/git/handson-ml3
 succeeded in 0ms:
exec
/bin/bash -lc 'date -u +%Y%m%d_%H%M%S' in /data/git/handson-ml3
 succeeded in 0ms:
20260416_004458

diff --git a/tools/codex/check_queue_readiness.py b/tools/codex/check_queue_readiness.py
index 1f183b056023e4f1158e440aad4aa8102b7cf816..c6eca9f62311a8daeaf744e2ceb8196ac317a597
--- a/tools/codex/check_queue_readiness.py
+++ b/tools/codex/check_queue_readiness.py
@@ -66,6 +66,10 @@
     pass
 
 
+def format_prefixes(prefixes: list[int]) -> str:
+    return ", ".join(f"{prefix:03d}" for prefix in prefixes)
+
+
 def repo_root() -> Path:
     return Path(__file__).resolve().parents[2]
 
@@ -244,6 +248,16 @@
     return records
 
 
+def discover_legacy_success_prefixes(root: Path) -> set[int]:
+    notes_dir = root / NOTES_DIR
+    prefixes: set[int] = set()
+    for path in sorted(notes_dir.glob("*__SUCCESS__*.md")):
+        match = PROMPT_NAME_RE.match(path.stem)
+        if match is not None:
+            prefixes.add(int(match.group("prefix")))
+    return prefixes
+
+
 def latest_record_for_prompt(records: list[RunRecord], prompt: PromptEntry) -> RunRecord | None:
     relevant = [record for record in records if record.prompt_file == prompt.label]
     if not relevant:
@@ -323,7 +337,42 @@
     )
 
 
-def print_summary(prompts: list[PromptEntry], result: ReadinessResult) -> None:
+def build_default_gap_explanation(
+    prompts: list[PromptEntry],
+    records: list[RunRecord],
+    legacy_success_prefixes: set[int],
+    target: PromptEntry,
+) -> str | None:
+    later_v1_prompts = [prompt for prompt in prompts if prompt.prefix > target.prefix]
+    if not any(latest_record_for_prompt(records, prompt) is not None for prompt in later_v1_prompts):
+        return None
+
+    gap_prefixes = [
+        prompt.prefix
+        for prompt in prompts
+        if prompt.prefix >= target.prefix and latest_record_for_prompt(records, prompt) is None
+    ]
+    if not gap_prefixes:
+        return None
+
+    surprising_prefixes = [prefix for prefix in gap_prefixes if prefix in legacy_success_prefixes]
+    if not surprising_prefixes:
+        return None
+
+    return (
+        "Queue note: default selection uses only V1 execution records in notes/. "
+        f"Legacy __SUCCESS__ notes do not count as V1 queue history, so prompts "
+        f"{format_prefixes(surprising_prefixes)} still count as missing V1 evidence "
+        "and can pull the default target earlier than older notes suggest."
+    )
+
+
+def print_summary(
+    prompts: list[PromptEntry],
+    result: ReadinessResult,
+    *,
+    default_gap_explanation: str | None = None,
+) -> None:
     print("Ordered prompts:")
     for prompt in prompts:
         print(f"- {prompt.prefix:03d}: {prompt.label}")
@@ -341,6 +390,8 @@
     print(f"Latest run review_status: {latest_review_status}")
     print(f"Ready: {'YES' if result.ready else 'NO'}")
     print(f"Reason: {result.reason}")
+    if default_gap_explanation:
+        print(default_gap_explanation)
 
 
 def main() -> int:
@@ -349,9 +400,23 @@
         root = repo_root()
         prompts = discover_prompts(root)
         records = discover_run_records(root)
+        using_default_target = args.prompt is None
         target = resolve_prompt(prompts, args.prompt) if args.prompt else default_target(prompts, records)
         result = evaluate_readiness(prompts, records, target)
-        print_summary(prompts, result)
+        default_gap_explanation = None
+        if using_default_target:
+            legacy_success_prefixes = discover_legacy_success_prefixes(root)
+            default_gap_explanation = build_default_gap_explanation(
+                prompts,
+                records,
+                legacy_success_prefixes,
+                target,
+            )
+        print_summary(
+            prompts,
+            result,
+            default_gap_explanation=default_gap_explanation,
+        )
         return 0
     except ReadinessError as exc:
         return fail(str(exc))

 succeeded in 266ms:
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
- 017: codex_prompts/017_polish_queue_readiness_gap_explanation.md

Target prompt: codex_prompts/002_repo_inventory_and_status.md
Previous prompt: codex_prompts/001_smoke_test_pipeline.md
Latest run record: notes/001_smoke_test_pipeline__20260415_234918.md
Latest run execution_status: EXECUTED
Latest run review_status: ACCEPTED
Ready: YES
Reason: latest V1 run for the immediately previous prompt is ACCEPTED
Queue note: default selection uses only V1 execution records in notes/. Legacy __SUCCESS__ notes do not count as V1 queue history, so prompts 002, 003, 004, 005, 006, 007, 008, 009, 010 still count as missing V1 evidence and can pull the default target earlier than older notes suggest.

 succeeded in 256ms:
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
- 017: codex_prompts/017_polish_queue_readiness_gap_explanation.md

Target prompt: codex_prompts/002_repo_inventory_and_status.md
Previous prompt: codex_prompts/001_smoke_test_pipeline.md
Latest run record: notes/001_smoke_test_pipeline__20260415_234918.md
Latest run execution_status: EXECUTED
Latest run review_status: ACCEPTED
Ready: YES
Reason: latest V1 run for the immediately previous prompt is ACCEPTED

 succeeded in 242ms:
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
- 017: codex_prompts/017_polish_queue_readiness_gap_explanation.md

Target prompt: codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md
Previous prompt: codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md
Latest run record: notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md
Latest run execution_status: EXECUTED
Latest run review_status: UNREVIEWED
Ready: NO
Reason: latest V1 run for the immediately previous prompt is UNREVIEWED

codex
The behavior is validated. I’m writing the short implementation note now with the concrete outputs and the explicit statement that the queue rule stayed unchanged.
apply patch
patch: completed
/data/git/handson-ml3/notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md
diff --git a/notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md b/notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md
new file mode 100644
index 0000000000000000000000000000000000000000..d75ad237a42a6a0b371af0049d3aa8bfe775f452
--- /dev/null
+++ b/notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md
@@ -0,0 +1,60 @@
+# 017 Queue Readiness Gap Explanation Polish
+
+## Summary
+
+Updated the human-readable stdout of `tools/codex/check_queue_readiness.py` for the default-target case where missing earlier V1 execution-record history can make the result look surprising.
+
+The helper now prints one small `Queue note:` line when:
+
+- default target selection is being used
+- later prompts already have V1 execution records
+- one or more earlier prompts still have no V1 record
+- those missing-V1 prompts also have legacy `__SUCCESS__` notes that a human might otherwise mistake for queue history
+
+## Why This Change Was Made
+
+The current repo state includes legacy `__SUCCESS__` notes for prompts `002` through `010`, but those notes are not V1 execution records.
+
+The helper was already making the correct conservative decision by defaulting back to prompt `002`. The usability issue was that the output did not clearly explain why older-looking success notes were being ignored.
+
+The added note clarifies that:
+
+- only V1 execution records in `notes/` count
+- legacy `__SUCCESS__` notes do not count as V1 queue history
+- missing V1 evidence for earlier prompts can pull the default target earlier than a human might first expect
+
+## Logic Intentionally Left Unchanged
+
+This pass did not change queue policy or readiness logic.
+
+Specifically left unchanged:
+
+- prompt discovery and ordering
+- default target selection behavior
+- latest-record selection behavior
+- `ACCEPTED` / `UNREVIEWED` / `REJECTED` handling
+- the meaning of missing prior V1 evidence
+- treatment of legacy notes as non-authoritative for V1 queue readiness
+
+The only behavior change is an additional explanatory output line in the narrow default-mode gap case.
+
+## Validation
+
+Validation performed in the current repo state:
+
+1. `python3 tools/codex/check_queue_readiness.py`
+   Outcome: succeeded. Default output still selected `codex_prompts/002_repo_inventory_and_status.md` and now included:
+   `Queue note: default selection uses only V1 execution records in notes/. Legacy __SUCCESS__ notes do not count as V1 queue history, so prompts 002, 003, 004, 005, 006, 007, 008, 009, 010 still count as missing V1 evidence and can pull the default target earlier than older notes suggest.`
+
+2. `python3 tools/codex/check_queue_readiness.py --prompt 002`
+   Outcome: succeeded and still reported readiness correctly:
+   `Ready: YES`
+
+3. `python3 tools/codex/check_queue_readiness.py --prompt 013`
+   Outcome: succeeded and still reported not-ready correctly:
+   `Ready: NO`
+   Reason remained `latest V1 run for the immediately previous prompt is UNREVIEWED`.
+
+4. Protected files unchanged check:
+   `git diff --name-only -- tools/codex/baby_run_prompt.py tools/codex/run_prompt.py tools/codex/review_run.py tools/codex/list_review_backlog.py`
+   Outcome: no output, confirming those files remained unchanged.
diff --git a/tools/codex/check_queue_readiness.py b/tools/codex/check_queue_readiness.py
index 1f183b056023e4f1158e440aad4aa8102b7cf816..c6eca9f62311a8daeaf744e2ceb8196ac317a597
--- a/tools/codex/check_queue_readiness.py
+++ b/tools/codex/check_queue_readiness.py
@@ -66,6 +66,10 @@
     pass
 
 
+def format_prefixes(prefixes: list[int]) -> str:
+    return ", ".join(f"{prefix:03d}" for prefix in prefixes)
+
+
 def repo_root() -> Path:
     return Path(__file__).resolve().parents[2]
 
@@ -244,6 +248,16 @@
     return records
 
 
+def discover_legacy_success_prefixes(root: Path) -> set[int]:
+    notes_dir = root / NOTES_DIR
+    prefixes: set[int] = set()
+    for path in sorted(notes_dir.glob("*__SUCCESS__*.md")):
+        match = PROMPT_NAME_RE.match(path.stem)
+        if match is not None:
+            prefixes.add(int(match.group("prefix")))
+    return prefixes
+
+
 def latest_record_for_prompt(records: list[RunRecord], prompt: PromptEntry) -> RunRecord | None:
     relevant = [record for record in records if record.prompt_file == prompt.label]
     if not relevant:
@@ -323,7 +337,42 @@
     )
 
 
-def print_summary(prompts: list[PromptEntry], result: ReadinessResult) -> None:
+def build_default_gap_explanation(
+    prompts: list[PromptEntry],
+    records: list[RunRecord],
+    legacy_success_prefixes: set[int],
+    target: PromptEntry,
+) -> str | None:
+    later_v1_prompts = [prompt for prompt in prompts if prompt.prefix > target.prefix]
+    if not any(latest_record_for_prompt(records, prompt) is not None for prompt in later_v1_prompts):
+        return None
+
+    gap_prefixes = [
+        prompt.prefix
+        for prompt in prompts
+        if prompt.prefix >= target.prefix and latest_record_for_prompt(records, prompt) is None
+    ]
+    if not gap_prefixes:
+        return None
+
+    surprising_prefixes = [prefix for prefix in gap_prefixes if prefix in legacy_success_prefixes]
+    if not surprising_prefixes:
+        return None
+
+    return (
+        "Queue note: default selection uses only V1 execution records in notes/. "
+        f"Legacy __SUCCESS__ notes do not count as V1 queue history, so prompts "
+        f"{format_prefixes(surprising_prefixes)} still count as missing V1 evidence "
+        "and can pull the default target earlier than older notes suggest."
+    )
+
+
+def print_summary(
+    prompts: list[PromptEntry],
+    result: ReadinessResult,
+    *,
+    default_gap_explanation: str | None = None,
+) -> None:
     print("Ordered prompts:")
     for prompt in prompts:
         print(f"- {prompt.prefix:03d}: {prompt.label}")
@@ -341,6 +390,8 @@
     print(f"Latest run review_status: {latest_review_status}")
     print(f"Ready: {'YES' if result.ready else 'NO'}")
     print(f"Reason: {result.reason}")
+    if default_gap_explanation:
+        print(default_gap_explanation)
 
 
 def main() -> int:
@@ -349,9 +400,23 @@
         root = repo_root()
         prompts = discover_prompts(root)
         records = discover_run_records(root)
+        using_default_target = args.prompt is None
         target = resolve_prompt(prompts, args.prompt) if args.prompt else default_target(prompts, records)
         result = evaluate_readiness(prompts, records, target)
-        print_summary(prompts, result)
+        default_gap_explanation = None
+        if using_default_target:
+            legacy_success_prefixes = discover_legacy_success_prefixes(root)
+            default_gap_explanation = build_default_gap_explanation(
+                prompts,
+                records,
+                legacy_success_prefixes,
+                target,
+            )
+        print_summary(
+            prompts,
+            result,
+            default_gap_explanation=default_gap_explanation,
+        )
         return 0
     except ReadinessError as exc:
         return fail(str(exc))

diff --git a/notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md b/notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md
new file mode 100644
index 0000000000000000000000000000000000000000..d75ad237a42a6a0b371af0049d3aa8bfe775f452
--- /dev/null
+++ b/notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md
@@ -0,0 +1,60 @@
+# 017 Queue Readiness Gap Explanation Polish
+
+## Summary
+
+Updated the human-readable stdout of `tools/codex/check_queue_readiness.py` for the default-target case where missing earlier V1 execution-record history can make the result look surprising.
+
+The helper now prints one small `Queue note:` line when:
+
+- default target selection is being used
+- later prompts already have V1 execution records
+- one or more earlier prompts still have no V1 record
+- those missing-V1 prompts also have legacy `__SUCCESS__` notes that a human might otherwise mistake for queue history
+
+## Why This Change Was Made
+
+The current repo state includes legacy `__SUCCESS__` notes for prompts `002` through `010`, but those notes are not V1 execution records.
+
+The helper was already making the correct conservative decision by defaulting back to prompt `002`. The usability issue was that the output did not clearly explain why older-looking success notes were being ignored.
+
+The added note clarifies that:
+
+- only V1 execution records in `notes/` count
+- legacy `__SUCCESS__` notes do not count as V1 queue history
+- missing V1 evidence for earlier prompts can pull the default target earlier than a human might first expect
+
+## Logic Intentionally Left Unchanged
+
+This pass did not change queue policy or readiness logic.
+
+Specifically left unchanged:
+
+- prompt discovery and ordering
+- default target selection behavior
+- latest-record selection behavior
+- `ACCEPTED` / `UNREVIEWED` / `REJECTED` handling
+- the meaning of missing prior V1 evidence
+- treatment of legacy notes as non-authoritative for V1 queue readiness
+
+The only behavior change is an additional explanatory output line in the narrow default-mode gap case.
+
+## Validation
+
+Validation performed in the current repo state:
+
+1. `python3 tools/codex/check_queue_readiness.py`
+   Outcome: succeeded. Default output still selected `codex_prompts/002_repo_inventory_and_status.md` and now included:
+   `Queue note: default selection uses only V1 execution records in notes/. Legacy __SUCCESS__ notes do not count as V1 queue history, so prompts 002, 003, 004, 005, 006, 007, 008, 009, 010 still count as missing V1 evidence and can pull the default target earlier than older notes suggest.`
+
+2. `python3 tools/codex/check_queue_readiness.py --prompt 002`
+   Outcome: succeeded and still reported readiness correctly:
+   `Ready: YES`
+
+3. `python3 tools/codex/check_queue_readiness.py --prompt 013`
+   Outcome: succeeded and still reported not-ready correctly:
+   `Ready: NO`
+   Reason remained `latest V1 run for the immediately previous prompt is UNREVIEWED`.
+
+4. Protected files unchanged check:
+   `git diff --name-only -- tools/codex/baby_run_prompt.py tools/codex/run_prompt.py tools/codex/review_run.py tools/codex/list_review_backlog.py`
+   Outcome: no output, confirming those files remained unchanged.
diff --git a/tools/codex/check_queue_readiness.py b/tools/codex/check_queue_readiness.py
index 1f183b056023e4f1158e440aad4aa8102b7cf816..c6eca9f62311a8daeaf744e2ceb8196ac317a597
--- a/tools/codex/check_queue_readiness.py
+++ b/tools/codex/check_queue_readiness.py
@@ -66,6 +66,10 @@
     pass
 
 
+def format_prefixes(prefixes: list[int]) -> str:
+    return ", ".join(f"{prefix:03d}" for prefix in prefixes)
+
+
 def repo_root() -> Path:
     return Path(__file__).resolve().parents[2]
 
@@ -244,6 +248,16 @@
     return records
 
 
+def discover_legacy_success_prefixes(root: Path) -> set[int]:
+    notes_dir = root / NOTES_DIR
+    prefixes: set[int] = set()
+    for path in sorted(notes_dir.glob("*__SUCCESS__*.md")):
+        match = PROMPT_NAME_RE.match(path.stem)
+        if match is not None:
+            prefixes.add(int(match.group("prefix")))
+    return prefixes
+
+
 def latest_record_for_prompt(records: list[RunRecord], prompt: PromptEntry) -> RunRecord | None:
     relevant = [record for record in records if record.prompt_file == prompt.label]
     if not relevant:
@@ -323,7 +337,42 @@
     )
 
 
-def print_summary(prompts: list[PromptEntry], result: ReadinessResult) -> None:
+def build_default_gap_explanation(
+    prompts: list[PromptEntry],
+    records: list[RunRecord],
+    legacy_success_prefixes: set[int],
+    target: PromptEntry,
+) -> str | None:
+    later_v1_prompts = [prompt for prompt in prompts if prompt.prefix > target.prefix]
+    if not any(latest_record_for_prompt(records, prompt) is not None for prompt in later_v1_prompts):
+        return None
+
+    gap_prefixes = [
+        prompt.prefix
+        for prompt in prompts
+        if prompt.prefix >= target.prefix and latest_record_for_prompt(records, prompt) is None
+    ]
+    if not gap_prefixes:
+        return None
+
+    surprising_prefixes = [prefix for prefix in gap_prefixes if prefix in legacy_success_prefixes]
+    if not surprising_prefixes:
+        return None
+
+    return (
+        "Queue note: default selection uses only V1 execution records in notes/. "
+        f"Legacy __SUCCESS__ notes do not count as V1 queue history, so prompts "
+        f"{format_prefixes(surprising_prefixes)} still count as missing V1 evidence "
+        "and can pull the default target earlier than older notes suggest."
+    )
+
+
+def print_summary(
+    prompts: list[PromptEntry],
+    result: ReadinessResult,
+    *,
+    default_gap_explanation: str | None = None,
+) -> None:
     print("Ordered prompts:")
     for prompt in prompts:
         print(f"- {prompt.prefix:03d}: {prompt.label}")
@@ -341,6 +390,8 @@
     print(f"Latest run review_status: {latest_review_status}")
     print(f"Ready: {'YES' if result.ready else 'NO'}")
     print(f"Reason: {result.reason}")
+    if default_gap_explanation:
+        print(default_gap_explanation)
 
 
 def main() -> int:
@@ -349,9 +400,23 @@
         root = repo_root()
         prompts = discover_prompts(root)
         records = discover_run_records(root)
+        using_default_target = args.prompt is None
         target = resolve_prompt(prompts, args.prompt) if args.prompt else default_target(prompts, records)
         result = evaluate_readiness(prompts, records, target)
-        print_summary(prompts, result)
+        default_gap_explanation = None
+        if using_default_target:
+            legacy_success_prefixes = discover_legacy_success_prefixes(root)
+            default_gap_explanation = build_default_gap_explanation(
+                prompts,
+                records,
+                legacy_success_prefixes,
+                target,
+            )
+        print_summary(
+            prompts,
+            result,
+            default_gap_explanation=default_gap_explanation,
+        )
         return 0
     except ReadinessError as exc:
         return fail(str(exc))

codex
Updated [tools/codex/check_queue_readiness.py](/data/git/handson-ml3/tools/codex/check_queue_readiness.py) to add a small `Queue note:` line in the default-target path when missing earlier V1 history is likely to be confusing because legacy `__SUCCESS__` notes exist. The readiness logic itself is unchanged.

The new output now explains that only V1 execution records in `notes/` count for queue history, that legacy `__SUCCESS__` notes do not, and that this is why the default target can point back to an earlier prompt such as `002`. I also added the implementation note at [notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md](/data/git/handson-ml3/notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md).

Validation passed: default invocation still succeeds and now shows the clarification, `--prompt 002` still reports ready, `--prompt 013` still reports not ready, and `tools/codex/baby_run_prompt.py`, `tools/codex/run_prompt.py`, `tools/codex/review_run.py`, and `tools/codex/list_review_backlog.py` remained unchanged.
diff --git a/notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md b/notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md
new file mode 100644
index 0000000000000000000000000000000000000000..d75ad237a42a6a0b371af0049d3aa8bfe775f452
--- /dev/null
+++ b/notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md
@@ -0,0 +1,60 @@
+# 017 Queue Readiness Gap Explanation Polish
+
+## Summary
+
+Updated the human-readable stdout of `tools/codex/check_queue_readiness.py` for the default-target case where missing earlier V1 execution-record history can make the result look surprising.
+
+The helper now prints one small `Queue note:` line when:
+
+- default target selection is being used
+- later prompts already have V1 execution records
+- one or more earlier prompts still have no V1 record
+- those missing-V1 prompts also have legacy `__SUCCESS__` notes that a human might otherwise mistake for queue history
+
+## Why This Change Was Made
+
+The current repo state includes legacy `__SUCCESS__` notes for prompts `002` through `010`, but those notes are not V1 execution records.
+
+The helper was already making the correct conservative decision by defaulting back to prompt `002`. The usability issue was that the output did not clearly explain why older-looking success notes were being ignored.
+
+The added note clarifies that:
+
+- only V1 execution records in `notes/` count
+- legacy `__SUCCESS__` notes do not count as V1 queue history
+- missing V1 evidence for earlier prompts can pull the default target earlier than a human might first expect
+
+## Logic Intentionally Left Unchanged
+
+This pass did not change queue policy or readiness logic.
+
+Specifically left unchanged:
+
+- prompt discovery and ordering
+- default target selection behavior
+- latest-record selection behavior
+- `ACCEPTED` / `UNREVIEWED` / `REJECTED` handling
+- the meaning of missing prior V1 evidence
+- treatment of legacy notes as non-authoritative for V1 queue readiness
+
+The only behavior change is an additional explanatory output line in the narrow default-mode gap case.
+
+## Validation
+
+Validation performed in the current repo state:
+
+1. `python3 tools/codex/check_queue_readiness.py`
+   Outcome: succeeded. Default output still selected `codex_prompts/002_repo_inventory_and_status.md` and now included:
+   `Queue note: default selection uses only V1 execution records in notes/. Legacy __SUCCESS__ notes do not count as V1 queue history, so prompts 002, 003, 004, 005, 006, 007, 008, 009, 010 still count as missing V1 evidence and can pull the default target earlier than older notes suggest.`
+
+2. `python3 tools/codex/check_queue_readiness.py --prompt 002`
+   Outcome: succeeded and still reported readiness correctly:
+   `Ready: YES`
+
+3. `python3 tools/codex/check_queue_readiness.py --prompt 013`
+   Outcome: succeeded and still reported not-ready correctly:
+   `Ready: NO`
+   Reason remained `latest V1 run for the immediately previous prompt is UNREVIEWED`.
+
+4. Protected files unchanged check:
+   `git diff --name-only -- tools/codex/baby_run_prompt.py tools/codex/run_prompt.py tools/codex/review_run.py tools/codex/list_review_backlog.py`
+   Outcome: no output, confirming those files remained unchanged.
diff --git a/tools/codex/check_queue_readiness.py b/tools/codex/check_queue_readiness.py
index 1f183b056023e4f1158e440aad4aa8102b7cf816..c6eca9f62311a8daeaf744e2ceb8196ac317a597
--- a/tools/codex/check_queue_readiness.py
+++ b/tools/codex/check_queue_readiness.py
@@ -66,6 +66,10 @@
     pass
 
 
+def format_prefixes(prefixes: list[int]) -> str:
+    return ", ".join(f"{prefix:03d}" for prefix in prefixes)
+
+
 def repo_root() -> Path:
     return Path(__file__).resolve().parents[2]
 
@@ -244,6 +248,16 @@
     return records
 
 
+def discover_legacy_success_prefixes(root: Path) -> set[int]:
+    notes_dir = root / NOTES_DIR
+    prefixes: set[int] = set()
+    for path in sorted(notes_dir.glob("*__SUCCESS__*.md")):
+        match = PROMPT_NAME_RE.match(path.stem)
+        if match is not None:
+            prefixes.add(int(match.group("prefix")))
+    return prefixes
+
+
 def latest_record_for_prompt(records: list[RunRecord], prompt: PromptEntry) -> RunRecord | None:
     relevant = [record for record in records if record.prompt_file == prompt.label]
     if not relevant:
@@ -323,7 +337,42 @@
     )
 
 
-def print_summary(prompts: list[PromptEntry], result: ReadinessResult) -> None:
+def build_default_gap_explanation(
+    prompts: list[PromptEntry],
+    records: list[RunRecord],
+    legacy_success_prefixes: set[int],
+    target: PromptEntry,
+) -> str | None:
+    later_v1_prompts = [prompt for prompt in prompts if prompt.prefix > target.prefix]
+    if not any(latest_record_for_prompt(records, prompt) is not None for prompt in later_v1_prompts):
+        return None
+
+    gap_prefixes = [
+        prompt.prefix
+        for prompt in prompts
+        if prompt.prefix >= target.prefix and latest_record_for_prompt(records, prompt) is None
+    ]
+    if not gap_prefixes:
+        return None
+
+    surprising_prefixes = [prefix for prefix in gap_prefixes if prefix in legacy_success_prefixes]
+    if not surprising_prefixes:
+        return None
+
+    return (
+        "Queue note: default selection uses only V1 execution records in notes/. "
+        f"Legacy __SUCCESS__ notes do not count as V1 queue history, so prompts "
+        f"{format_prefixes(surprising_prefixes)} still count as missing V1 evidence "
+        "and can pull the default target earlier than older notes suggest."
+    )
+
+
+def print_summary(
+    prompts: list[PromptEntry],
+    result: ReadinessResult,
+    *,
+    default_gap_explanation: str | None = None,
+) -> None:
     print("Ordered prompts:")
     for prompt in prompts:
         print(f"- {prompt.prefix:03d}: {prompt.label}")
@@ -341,6 +390,8 @@
     print(f"Latest run review_status: {latest_review_status}")
     print(f"Ready: {'YES' if result.ready else 'NO'}")
     print(f"Reason: {result.reason}")
+    if default_gap_explanation:
+        print(default_gap_explanation)
 
 
 def main() -> int:
@@ -349,9 +400,23 @@
         root = repo_root()
         prompts = discover_prompts(root)
         records = discover_run_records(root)
+        using_default_target = args.prompt is None
         target = resolve_prompt(prompts, args.prompt) if args.prompt else default_target(prompts, records)
         result = evaluate_readiness(prompts, records, target)
-        print_summary(prompts, result)
+        default_gap_explanation = None
+        if using_default_target:
+            legacy_success_prefixes = discover_legacy_success_prefixes(root)
+            default_gap_explanation = build_default_gap_explanation(
+                prompts,
+                records,
+                legacy_success_prefixes,
+                target,
+            )
+        print_summary(
+            prompts,
+            result,
+            default_gap_explanation=default_gap_explanation,
+        )
         return 0
     except ReadinessError as exc:
         return fail(str(exc))

tokens used
40,967
```
