# 018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work__20260416_005034

- run_id: `018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work__20260416_005034`
- prompt_file: `codex_prompts/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work.md`
- prompt_stem: `018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work`
- started_at_utc: `20260416_005034`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260416_005240`
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

- elapsed_seconds: `125.390`
- final_output_char_count: `720`
- stderr_char_count: `183460`

## Prompt Text

```md
# Task: Sweep the architecture document against the actual V1 implementation and prioritize remaining work

You are working in this repository.

Your task is to compare the current architecture/design documents against the actual V1 implementation and artifacts now present in the repo, then produce a short prioritized list of what remains to build or clean up.

## Important framing

This is an architecture-review and prioritization task.

Do NOT implement code in this pass.
Do NOT rewrite major documents in this pass.
Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

Do NOT build new helpers in this pass.

Your job is to compare the intended architecture to the repo as it actually exists and decide what is left, what matters most, and what should wait.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

Read these recent notes:

- `notes/012_v1_pipeline_options_review__20260416_000819.md`
- `notes/014_queue_readiness_checker_build__20260416_002419.md`
- `notes/015_review_backlog_lister_build__20260416_010500.md`
- `notes/016_queue_and_backlog_helper_validation__20260416_003710.md`
- `notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md`

Also inspect current repo contents in:

- `codex_prompts/`
- `notes/`
- `tools/codex/`

## Goal

Determine:

1. what the architecture/design packet says should exist
2. what actually exists now
3. what still appears missing, stale, drifting, or deferred
4. what remaining work should be prioritized next

## Questions to answer

### 1. Architecture alignment
- Which parts of the intended V1 architecture are now implemented and operational?
- Which parts are only partially implemented?
- Which parts of the architecture doc are stale, misleading, or lagging the actual repo state?

### 2. Remaining work
Identify the realistic remaining work items that are still justified by the current repo state.

Prefer items such as:
- small usability gaps
- doc/spec drift cleanup
- lightweight validation/contract checks
- missing thin operational helpers
- one or two small pieces needed before using the system regularly for real work

Avoid speculative platform expansion.

### 3. Prioritization
Rank the remaining items in a short prioritized list.

For each item include:
- short name
- why it matters
- expected risk: low / medium / high
- expected payoff: low / medium / high
- recommended timing:
  - next
  - soon
  - later
  - explicitly defer

### 4. Practical stopping point
Based on the current repo state, answer this clearly:

- Is the V1 prompt workflow system now good enough to use for real work?
- If yes, what is the smallest remaining thing to clean up before using it heavily?
- If no, what single missing piece still blocks that?

## Required output artifacts

Create exactly two artifacts.

### Artifact 1
Create a review report at:

`notes/018_architecture_vs_actual_sweep__TIMESTAMP.md`

This report should include:

- short summary
- implemented vs intended comparison
- remaining-work list
- prioritized ranking
- judgment about whether V1 is ready for real work

### Artifact 2
Create a short recommendation note at:

`notes/018_prioritized_remaining_work__TIMESTAMP.md`

This note should contain only:

- the top prioritized remaining items
- which one should happen next
- what should explicitly wait

Keep it brief and decisive.

## Constraints

1. Use the exact file paths listed above.
2. Do not implement code in this pass.
3. Do not rewrite the architecture doc in this pass.
4. Do not produce a giant roadmap.
5. Prioritize bounded, evidence-based remaining work only.
6. Keep the result practical enough to drive the next useful step.

## Success criteria

This task is successful if:

- the report accurately compares the architecture packet to the actual repo state
- the remaining work list is short and grounded
- the prioritization is practical
- the result helps decide whether to keep polishing the tool or start using it for the real job that motivated it
```

## Codex Final Output

Created the two requested artifacts:

- [018_architecture_vs_actual_sweep__20260416_005130.md](/data/git/handson-ml3/notes/018_architecture_vs_actual_sweep__20260416_005130.md)
- [018_prioritized_remaining_work__20260416_005130.md](/data/git/handson-ml3/notes/018_prioritized_remaining_work__20260416_005130.md)

Bottom line: the V1 workflow is good enough for real work now. The next cleanup should be doc/spec alignment so the architecture packet matches the actual toolset (`run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, `list_review_backlog.py`) and the current run-id behavior. After that, the most justified technical follow-up is a small contract-validation layer for the markdown execution record.

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
session id: 019d93c4-b63c-7e81-8c1a-be0aee0a02b2
--------
user
# Task: Sweep the architecture document against the actual V1 implementation and prioritize remaining work

You are working in this repository.

Your task is to compare the current architecture/design documents against the actual V1 implementation and artifacts now present in the repo, then produce a short prioritized list of what remains to build or clean up.

## Important framing

This is an architecture-review and prioritization task.

Do NOT implement code in this pass.
Do NOT rewrite major documents in this pass.
Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

Do NOT build new helpers in this pass.

Your job is to compare the intended architecture to the repo as it actually exists and decide what is left, what matters most, and what should wait.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

Read these recent notes:

- `notes/012_v1_pipeline_options_review__20260416_000819.md`
- `notes/014_queue_readiness_checker_build__20260416_002419.md`
- `notes/015_review_backlog_lister_build__20260416_010500.md`
- `notes/016_queue_and_backlog_helper_validation__20260416_003710.md`
- `notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md`

Also inspect current repo contents in:

- `codex_prompts/`
- `notes/`
- `tools/codex/`

## Goal

Determine:

1. what the architecture/design packet says should exist
2. what actually exists now
3. what still appears missing, stale, drifting, or deferred
4. what remaining work should be prioritized next

## Questions to answer

### 1. Architecture alignment
- Which parts of the intended V1 architecture are now implemented and operational?
- Which parts are only partially implemented?
- Which parts of the architecture doc are stale, misleading, or lagging the actual repo state?

### 2. Remaining work
Identify the realistic remaining work items that are still justified by the current repo state.

Prefer items such as:
- small usability gaps
- doc/spec drift cleanup
- lightweight validation/contract checks
- missing thin operational helpers
- one or two small pieces needed before using the system regularly for real work

Avoid speculative platform expansion.

### 3. Prioritization
Rank the remaining items in a short prioritized list.

For each item include:
- short name
- why it matters
- expected risk: low / medium / high
- expected payoff: low / medium / high
- recommended timing:
  - next
  - soon
  - later
  - explicitly defer

### 4. Practical stopping point
Based on the current repo state, answer this clearly:

- Is the V1 prompt workflow system now good enough to use for real work?
- If yes, what is the smallest remaining thing to clean up before using it heavily?
- If no, what single missing piece still blocks that?

## Required output artifacts

Create exactly two artifacts.

### Artifact 1
Create a review report at:

`notes/018_architecture_vs_actual_sweep__TIMESTAMP.md`

This report should include:

- short summary
- implemented vs intended comparison
- remaining-work list
- prioritized ranking
- judgment about whether V1 is ready for real work

### Artifact 2
Create a short recommendation note at:

`notes/018_prioritized_remaining_work__TIMESTAMP.md`

This note should contain only:

- the top prioritized remaining items
- which one should happen next
- what should explicitly wait

Keep it brief and decisive.

## Constraints

1. Use the exact file paths listed above.
2. Do not implement code in this pass.
3. Do not rewrite the architecture doc in this pass.
4. Do not produce a giant roadmap.
5. Prioritize bounded, evidence-based remaining work only.
6. Keep the result practical enough to drive the next useful step.

## Success criteria

This task is successful if:

- the report accurately compares the architecture packet to the actual repo state
- the remaining work list is short and grounded
- the prioritization is practical
- the result helps decide whether to keep polishing the tool or start using it for the real job that motivated it

codex
I’m comparing the architecture packet to the current V1 implementation and recent notes, then I’ll write two short notes with the remaining work prioritized from actual repo evidence.
  → Read the specified architecture, implementation, and note files plus inspect relevant directories
  • Assess alignment, identify drift/gaps, and decide practical remaining work
  • Create the two required notes artifacts with concise prioritized recommendations
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
/bin/bash -lc 'pwd && ls -1 tools && ls -1 tools/codex && ls -1 notes | tail -n 40 && ls -1 codex_prompts | head -n 200' in /data/git/handson-ml3
 succeeded in 0ms:
/data/git/handson-ml3
Codex_Prompt_Workflow_Architecture.md
Project_Design_Workflow.md
codex
V1_Bridge_Runner_Change_Spec.md
V1_Execution_Record_Artifact.md
V1_Run_Review_Gate.md
__pycache__
baby_run_prompt.py
check_queue_readiness.py
list_review_backlog.py
review_run.py
run_prompt.py
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
016_next_improvement_recommendation__20260416_003710.md
016_queue_and_backlog_helper_validation__20260416_003710.md
016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md
017_polish_queue_readiness_gap_explanation__20260416_004355.md
017_queue_readiness_gap_explanation_polish__20260416_004458.md
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
017_polish_queue_readiness_gap_explanation.md
018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work.md

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


codex
I’ve got the design packet loaded. Next I’m reading the actual V1 scripts and the recent notes to see where the docs and the repo diverge.
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
/bin/bash -lc "sed -n '1,240p' notes/012_v1_pipeline_options_review__20260416_000819.md" in /data/git/handson-ml3
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
/bin/bash -lc "sed -n '1,240p' notes/014_queue_readiness_checker_build__20260416_002419.md" in /data/git/handson-ml3
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
/bin/bash -lc "sed -n '1,240p' notes/015_review_backlog_lister_build__20260416_010500.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 015 Review Backlog Lister Build

## What Was Built

Built a small read-only helper at [list_review_backlog.py](/data/git/handson-ml3/tools/codex/list_review_backlog.py) that:

- scans `notes/` for top-level V1 execution-record files
- parses the execution-record body as the source of truth for `run_id`, `prompt_file`, `prompt_stem`, `started_at_utc`, `execution_status`, and `review_status`
- lists all current `UNREVIEWED` records
- identifies the latest execution record per prompt
- surfaces a small "likely needs human review next" view by selecting prompts whose latest record is `UNREVIEWED`

The CLI stays small:

- default summary output with no positional arguments
- optional `--unreviewed-only` filter to narrow the latest-per-prompt view to prompts whose latest record is still `UNREVIEWED`

## Conservative Parsing Policy

The helper intentionally does not use a full markdown parser, but it also does not treat every note with copied field lines as a record.

It only accepts files that look like real top-level V1 execution records:

- the file starts with a markdown title
- the title matches the record `run_id`
- the required V1 sections exist in the expected order
- the required minimal V1 fields are present and valid

Policy on malformed inputs:

- non-record notes are skipped
- record-like files with incomplete or malformed required V1 structure fail clearly with an error

This keeps the backlog view conservative and inspectable while avoiding false positives from design notes that embed record examples.

## What Backlog View It Provides

The helper prints a short stdout summary with:

- all `UNREVIEWED` records found
- the latest record per prompt
- the prompts that likely need human review next because their latest record is `UNREVIEWED`

Each listed record includes:

- record path
- prompt file
- started timestamp
- execution status
- review status

For latest-record selection, it uses the record body and picks the highest `started_at_utc` for each prompt, with the run-id same-second suffix used only as a simple tiebreaker.

## What It Intentionally Does Not Do

- does not modify any files
- does not update review fields
- does not release prompts or compute queue readiness
- does not create JSON, caches, sidecars, or indexes
- does not build a dashboard, TUI, or broader reporting layer
- does not change `tools/codex/baby_run_prompt.py`
- does not change `tools/codex/run_prompt.py`
- does not change `tools/codex/review_run.py`

## Validation Performed

1. Confirmed the helper finds V1 execution records in `notes/` by running:
   `python3 tools/codex/list_review_backlog.py`
2. Confirmed it lists records still marked `UNREVIEWED` from the current V1 set.
3. Confirmed it identifies the latest record per prompt, including prompt `001` where the accepted later record supersedes the earlier failed record.
4. Confirmed it produces a small "likely needs human review next" summary from latest-record status.
5. Confirmed the helper remains read-only by inspecting the implementation and verifying it only reads `notes/*.md` and prints stdout.
6. Confirmed `tools/codex/run_prompt.py` remains unchanged.
7. Confirmed `tools/codex/review_run.py` remains unchanged.
8. Confirmed `tools/codex/baby_run_prompt.py` remains unchanged.
9. Confirmed the optional bounded filter works by running:
   `python3 tools/codex/list_review_backlog.py --unreviewed-only`

Validation outcome: passed for the bounded V1 review-discovery behavior above.

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
/bin/bash -lc "sed -n '1,240p' notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 017 Queue Readiness Gap Explanation Polish

## Summary

Updated the human-readable stdout of `tools/codex/check_queue_readiness.py` for the default-target case where missing earlier V1 execution-record history can make the result look surprising.

The helper now prints one small `Queue note:` line when:

- default target selection is being used
- later prompts already have V1 execution records
- one or more earlier prompts still have no V1 record
- those missing-V1 prompts also have legacy `__SUCCESS__` notes that a human might otherwise mistake for queue history

## Why This Change Was Made

The current repo state includes legacy `__SUCCESS__` notes for prompts `002` through `010`, but those notes are not V1 execution records.

The helper was already making the correct conservative decision by defaulting back to prompt `002`. The usability issue was that the output did not clearly explain why older-looking success notes were being ignored.

The added note clarifies that:

- only V1 execution records in `notes/` count
- legacy `__SUCCESS__` notes do not count as V1 queue history
- missing V1 evidence for earlier prompts can pull the default target earlier than a human might first expect

## Logic Intentionally Left Unchanged

This pass did not change queue policy or readiness logic.

Specifically left unchanged:

- prompt discovery and ordering
- default target selection behavior
- latest-record selection behavior
- `ACCEPTED` / `UNREVIEWED` / `REJECTED` handling
- the meaning of missing prior V1 evidence
- treatment of legacy notes as non-authoritative for V1 queue readiness

The only behavior change is an additional explanatory output line in the narrow default-mode gap case.

## Validation

Validation performed in the current repo state:

1. `python3 tools/codex/check_queue_readiness.py`
   Outcome: succeeded. Default output still selected `codex_prompts/002_repo_inventory_and_status.md` and now included:
   `Queue note: default selection uses only V1 execution records in notes/. Legacy __SUCCESS__ notes do not count as V1 queue history, so prompts 002, 003, 004, 005, 006, 007, 008, 009, 010 still count as missing V1 evidence and can pull the default target earlier than older notes suggest.`

2. `python3 tools/codex/check_queue_readiness.py --prompt 002`
   Outcome: succeeded and still reported readiness correctly:
   `Ready: YES`

3. `python3 tools/codex/check_queue_readiness.py --prompt 013`
   Outcome: succeeded and still reported not-ready correctly:
   `Ready: NO`
   Reason remained `latest V1 run for the immediately previous prompt is UNREVIEWED`.

4. Protected files unchanged check:
   `git diff --name-only -- tools/codex/baby_run_prompt.py tools/codex/run_prompt.py tools/codex/review_run.py tools/codex/list_review_backlog.py`
   Outcome: no output, confirming those files remained unchanged.

codex
The recent notes confirm the core runner, review write-back, readiness checker, and backlog lister all exist now. I’m checking the remaining details in the scripts and current repo artifacts before I rank what still matters.
exec
/bin/bash -lc "sed -n '261,520p' tools/Codex_Prompt_Workflow_Architecture.md" in /data/git/handson-ml3
 succeeded in 0ms:
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

1. keep the execution record as the stable source artifact
2. keep the review gate explicit
3. add only the smallest runner changes needed to support that record reliably
4. review whether any module split is justified by actual friction
5. defer broader queue features until the single-run slice has been used enough to reveal real needs

This keeps the architecture reusable and inspectable without turning the design document into a platform roadmap.

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
/bin/bash -lc "sed -n '261,520p' tools/codex/check_queue_readiness.py" in /data/git/handson-ml3
 succeeded in 0ms:
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


def build_default_gap_explanation(
    prompts: list[PromptEntry],
    records: list[RunRecord],
    legacy_success_prefixes: set[int],
    target: PromptEntry,
) -> str | None:
    later_v1_prompts = [prompt for prompt in prompts if prompt.prefix > target.prefix]
    if not any(latest_record_for_prompt(records, prompt) is not None for prompt in later_v1_prompts):
        return None

    gap_prefixes = [
        prompt.prefix
        for prompt in prompts
        if prompt.prefix >= target.prefix and latest_record_for_prompt(records, prompt) is None
    ]
    if not gap_prefixes:
        return None

    surprising_prefixes = [prefix for prefix in gap_prefixes if prefix in legacy_success_prefixes]
    if not surprising_prefixes:
        return None

    return (
        "Queue note: default selection uses only V1 execution records in notes/. "
        f"Legacy __SUCCESS__ notes do not count as V1 queue history, so prompts "
        f"{format_prefixes(surprising_prefixes)} still count as missing V1 evidence "
        "and can pull the default target earlier than older notes suggest."
    )


def print_summary(
    prompts: list[PromptEntry],
    result: ReadinessResult,
    *,
    default_gap_explanation: str | None = None,
) -> None:
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
    if default_gap_explanation:
        print(default_gap_explanation)


def main() -> int:
    try:
        args = parse_args()
        root = repo_root()
        prompts = discover_prompts(root)
        records = discover_run_records(root)
        using_default_target = args.prompt is None
        target = resolve_prompt(prompts, args.prompt) if args.prompt else default_target(prompts, records)
        result = evaluate_readiness(prompts, records, target)
        default_gap_explanation = None
        if using_default_target:
            legacy_success_prefixes = discover_legacy_success_prefixes(root)
            default_gap_explanation = build_default_gap_explanation(
                prompts,
                records,
                legacy_success_prefixes,
                target,
            )
        print_summary(
            prompts,
            result,
            default_gap_explanation=default_gap_explanation,
        )
        return 0
    except ReadinessError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

exec
/bin/bash -lc 'python3 tools/codex/check_queue_readiness.py' in /data/git/handson-ml3
exec
/bin/bash -lc 'python3 tools/codex/list_review_backlog.py' in /data/git/handson-ml3
exec
/bin/bash -lc 'rg -n "review_status: `|execution_status: `|run_id: `|retry_of_run_id:|review_summary:" notes/*.md' in /data/git/handson-ml3
 exited 2 in 0ms:
/bin/bash: -c: line 1: unexpected EOF while looking for matching ``'
/bin/bash: -c: line 2: syntax error: unexpected end of file

exec
/bin/bash -lc 'ls -1 notes/*__20260415_23*.md notes/*__20260416_00*.md notes/*__20260416_01*.md 2>/dev/null' in /data/git/handson-ml3
 succeeded in 0ms:
notes/001_smoke_test_pipeline__20260415_233343.md
notes/001_smoke_test_pipeline__20260415_234918.md
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
notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md
notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md

 succeeded in 251ms:
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
- 018: codex_prompts/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work.md

Target prompt: codex_prompts/002_repo_inventory_and_status.md
Previous prompt: codex_prompts/001_smoke_test_pipeline.md
Latest run record: notes/001_smoke_test_pipeline__20260415_234918.md
Latest run execution_status: EXECUTED
Latest run review_status: ACCEPTED
Ready: YES
Reason: latest V1 run for the immediately previous prompt is ACCEPTED
Queue note: default selection uses only V1 execution records in notes/. Legacy __SUCCESS__ notes do not count as V1 queue history, so prompts 002, 003, 004, 005, 006, 007, 008, 009, 010 still count as missing V1 evidence and can pull the default target earlier than older notes suggest.

 succeeded in 231ms:
V1 review backlog summary from notes/
Discovered V1 execution records: 9
Unreviewed records: 8
Prompts with latest record: 8

UNREVIEWED records:
- notes/001_smoke_test_pipeline__20260415_233343.md | prompt=codex_prompts/001_smoke_test_pipeline.md | started=20260415_233343 | execution=EXECUTION_FAILED | review=UNREVIEWED
- notes/011_build_v1_review_writeback_helper__20260415_235346.md | prompt=codex_prompts/011_build_v1_review_writeback_helper.md | started=20260415_235346 | execution=EXECUTED | review=UNREVIEWED
- notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md | prompt=codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md | started=20260416_000658 | execution=EXECUTED | review=UNREVIEWED
- notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md | prompt=codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md | started=20260416_001937 | execution=EXECUTED | review=UNREVIEWED
- notes/014_build_queue_readiness_checker__20260416_002319.md | prompt=codex_prompts/014_build_queue_readiness_checker.md | started=20260416_002319 | execution=EXECUTED | review=UNREVIEWED
- notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md | prompt=codex_prompts/015_build_review_backlog_unreviewed_run_lister.md | started=20260416_003109 | execution=EXECUTED | review=UNREVIEWED
- notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md | prompt=codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md | started=20260416_003601 | execution=EXECUTED | review=UNREVIEWED
- notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md | prompt=codex_prompts/017_polish_queue_readiness_gap_explanation.md | started=20260416_004355 | execution=EXECUTED | review=UNREVIEWED

Latest record per prompt:
- notes/001_smoke_test_pipeline__20260415_234918.md | prompt=codex_prompts/001_smoke_test_pipeline.md | started=20260415_234918 | execution=EXECUTED | review=ACCEPTED
- notes/011_build_v1_review_writeback_helper__20260415_235346.md | prompt=codex_prompts/011_build_v1_review_writeback_helper.md | started=20260415_235346 | execution=EXECUTED | review=UNREVIEWED
- notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md | prompt=codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md | started=20260416_000658 | execution=EXECUTED | review=UNREVIEWED
- notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md | prompt=codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md | started=20260416_001937 | execution=EXECUTED | review=UNREVIEWED
- notes/014_build_queue_readiness_checker__20260416_002319.md | prompt=codex_prompts/014_build_queue_readiness_checker.md | started=20260416_002319 | execution=EXECUTED | review=UNREVIEWED
- notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md | prompt=codex_prompts/015_build_review_backlog_unreviewed_run_lister.md | started=20260416_003109 | execution=EXECUTED | review=UNREVIEWED
- notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md | prompt=codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md | started=20260416_003601 | execution=EXECUTED | review=UNREVIEWED
- notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md | prompt=codex_prompts/017_polish_queue_readiness_gap_explanation.md | started=20260416_004355 | execution=EXECUTED | review=UNREVIEWED

Likely needs human review next:
- notes/011_build_v1_review_writeback_helper__20260415_235346.md | prompt=codex_prompts/011_build_v1_review_writeback_helper.md | started=20260415_235346 | execution=EXECUTED | review=UNREVIEWED
- notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md | prompt=codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md | started=20260416_000658 | execution=EXECUTED | review=UNREVIEWED
- notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md | prompt=codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md | started=20260416_001937 | execution=EXECUTED | review=UNREVIEWED
- notes/014_build_queue_readiness_checker__20260416_002319.md | prompt=codex_prompts/014_build_queue_readiness_checker.md | started=20260416_002319 | execution=EXECUTED | review=UNREVIEWED
- notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md | prompt=codex_prompts/015_build_review_backlog_unreviewed_run_lister.md | started=20260416_003109 | execution=EXECUTED | review=UNREVIEWED
- notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md | prompt=codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md | started=20260416_003601 | execution=EXECUTED | review=UNREVIEWED
- notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md | prompt=codex_prompts/017_polish_queue_readiness_gap_explanation.md | started=20260416_004355 | execution=EXECUTED | review=UNREVIEWED

codex
The current repo state is clear now: the V1 flow is real, but there’s visible drift between the design packet and the now-expanded toolset. I’m pulling a few final references so the report can be precise about what’s implemented versus what’s merely described.
exec
/bin/bash -lc "sed -n '1,140p' notes/011_review_writeback_helper_build__20260415_235514.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 011 Review Write-Back Helper Build

## What Was Built

Built [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py), a thin V1 companion helper that updates one existing execution record in `notes/` in place.

The helper:

- accepts a record path plus explicit review arguments
- verifies the file is under `notes/` and matches the minimum V1 execution-record structure
- updates only the targeted field lines
- writes the changes back to the same markdown file
- prints the updated file path
- exits nonzero on invalid input or malformed structure

## V1 Fields It Updates

Always updates:

- `review_status`
- `review_summary`
- `reviewed_at_utc`

Optionally updates:

- `reviewed_by`

For `REJECTED` runs, it can also update:

- `failure_type`
- `failure_symptom`
- `likely_cause`
- `recommended_next_action`

## What It Intentionally Does Not Do

- does not modify `tools/codex/run_prompt.py`
- does not modify `tools/codex/baby_run_prompt.py`
- does not create queue state, sidecars, databases, or extra workflow files
- does not release the next prompt automatically
- does not redesign the V1 record format
- does not rewrite or reorder non-review sections

## Validation

Validation commands performed:

- `python3 -m py_compile tools/codex/review_run.py`
- `python3 tools/codex/review_run.py notes/001_smoke_test_pipeline__20260415_234918.md --review-status ACCEPTED --review-summary 'Smoke test output and artifact are complete enough to accept this bounded step.'`
- `git diff -- tools/codex/run_prompt.py tools/codex/baby_run_prompt.py`

Validation outcome:

- `review_status` in [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md) changed from `UNREVIEWED` to `ACCEPTED`.
- `review_summary` was written in place.
- `reviewed_at_utc` was auto-filled as `20260415_235508`.
- All other sections remained unchanged; the diff was limited to the three review lines above.
- The updated record still matches the expected V1 section order and required field layout.
- `tools/codex/run_prompt.py` remained unchanged.
- `tools/codex/baby_run_prompt.py` remained unchanged.

## Scope Note

This helper is intentionally a minimum V1 review write-back tool. It updates review facts on one existing execution record and stops there.

exec
/bin/bash -lc 'rg -n "baby_run_prompt|run_prompt.py|review_run.py|check_queue_readiness.py|list_review_backlog.py|Conservative queue progression|review write-back|UNREVIEWED|ACCEPTED|REJECTED" tools/Codex_Prompt_Workflow_Architecture.md tools/codex/V1_Bridge_Runner_Change_Spec.md tools/codex/V1_Execution_Record_Artifact.md tools/codex/V1_Run_Review_Gate.md' in /data/git/handson-ml3
 succeeded in 0ms:
tools/codex/V1_Run_Review_Gate.md:38:- `review_status` still set to `UNREVIEWED`
tools/codex/V1_Run_Review_Gate.md:43:Record creation plus `UNREVIEWED` review status means the run must stop for human inspection.
tools/codex/V1_Run_Review_Gate.md:68:- `UNREVIEWED`
tools/codex/V1_Run_Review_Gate.md:69:- `ACCEPTED`
tools/codex/V1_Run_Review_Gate.md:70:- `REJECTED`
tools/codex/V1_Run_Review_Gate.md:74:- `UNREVIEWED`: default state after execution record creation; queue must stop here.
tools/codex/V1_Run_Review_Gate.md:75:- `ACCEPTED`: the bounded step is accepted after human review.
tools/codex/V1_Run_Review_Gate.md:76:- `REJECTED`: the run is not accepted for progression, whether because execution failed, the result is inadequate, or the outcome requires redesign or retry analysis first.
tools/codex/V1_Run_Review_Gate.md:82:Only `ACCEPTED` allows the next prompt in sequence to be treated as ready.
tools/codex/V1_Run_Review_Gate.md:86:- `UNREVIEWED` stops the queue pending manual review.
tools/codex/V1_Run_Review_Gate.md:87:- `ACCEPTED` releases exactly the next reviewed step.
tools/codex/V1_Run_Review_Gate.md:88:- `REJECTED` stops the queue and forces an explicit human decision about what happens next.
tools/codex/V1_Run_Review_Gate.md:91:`EXECUTION_FAILED` does not itself decide the queue either, but it will usually lead to `REJECTED` on review.
tools/codex/V1_Run_Review_Gate.md:95:`REJECTED` is the V1 outcome that stops the queue and forces a new design or retry decision.
tools/codex/V1_Run_Review_Gate.md:137:It is the manual transition of one execution record from `UNREVIEWED` to either `ACCEPTED` or `REJECTED`.
tools/codex/V1_Run_Review_Gate.md:156:2. a run record is written with `review_status: UNREVIEWED`
tools/codex/V1_Run_Review_Gate.md:158:4. reviewer writes back `ACCEPTED` or `REJECTED`
tools/codex/V1_Run_Review_Gate.md:159:5. only `ACCEPTED` makes the next prompt ready
tools/codex/V1_Execution_Record_Artifact.md:166:- `UNREVIEWED`
tools/codex/V1_Execution_Record_Artifact.md:167:- `ACCEPTED`
tools/codex/V1_Execution_Record_Artifact.md:168:- `REJECTED`
tools/codex/V1_Execution_Record_Artifact.md:172:- a new run record should start as `UNREVIEWED`
tools/codex/V1_Execution_Record_Artifact.md:173:- `EXECUTED` must not imply `ACCEPTED`
tools/codex/V1_Execution_Record_Artifact.md:174:- `EXECUTION_FAILED` does not automatically imply `REJECTED`, though rejection will often follow
tools/codex/V1_Execution_Record_Artifact.md:180:It should exist as a short section that can be filled in manually when useful, especially for `EXECUTION_FAILED` or `REJECTED` runs.
tools/codex/V1_Bridge_Runner_Change_Spec.md:5:Define the smallest change set for [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py) so the current bridge runner writes a V1 execution record in `notes/` without a broader refactor.
tools/codex/V1_Bridge_Runner_Change_Spec.md:17:This spec covers only the current bridge runner at [baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py).
tools/codex/V1_Bridge_Runner_Change_Spec.md:60:- `ACCEPTED`
tools/codex/V1_Bridge_Runner_Change_Spec.md:61:- `REJECTED`
tools/codex/V1_Bridge_Runner_Change_Spec.md:80:- `review_status: UNREVIEWED`
tools/codex/V1_Bridge_Runner_Change_Spec.md:86:- mark `ACCEPTED`
tools/codex/V1_Bridge_Runner_Change_Spec.md:87:- mark `REJECTED`
tools/codex/V1_Bridge_Runner_Change_Spec.md:100:It must not affect `review_status` beyond leaving it at `UNREVIEWED`.
tools/codex/V1_Bridge_Runner_Change_Spec.md:125:- queue progression logic beyond initializing `review_status: UNREVIEWED`
tools/codex/V1_Bridge_Runner_Change_Spec.md:126:- automatic review write-back
tools/codex/V1_Bridge_Runner_Change_Spec.md:156:- runner: `tools/codex/baby_run_prompt.py`
tools/codex/V1_Bridge_Runner_Change_Spec.md:166:- review_status: `UNREVIEWED`
tools/codex/V1_Bridge_Runner_Change_Spec.md:252:- `runner` -> literal `tools/codex/baby_run_prompt.py`
tools/codex/V1_Bridge_Runner_Change_Spec.md:261:- `review_status` -> literal `UNREVIEWED`
tools/codex/V1_Bridge_Runner_Change_Spec.md:299:- `review_status: UNREVIEWED`
tools/codex/V1_Bridge_Runner_Change_Spec.md:338:- review write-back later
tools/Codex_Prompt_Workflow_Architecture.md:78:* stopping the queue at `UNREVIEWED`
tools/Codex_Prompt_Workflow_Architecture.md:164:2. write one execution record with `review_status: UNREVIEWED`
tools/Codex_Prompt_Workflow_Architecture.md:166:4. update the same execution record to either `ACCEPTED` or `REJECTED`
tools/Codex_Prompt_Workflow_Architecture.md:167:5. treat only `ACCEPTED` as releasing the next prompt in sequence
tools/Codex_Prompt_Workflow_Architecture.md:218:* new records begin as `UNREVIEWED`
tools/Codex_Prompt_Workflow_Architecture.md:219:* human review decides `ACCEPTED` or `REJECTED`
tools/Codex_Prompt_Workflow_Architecture.md:220:* only `ACCEPTED` allows the next prompt to be treated as ready
tools/Codex_Prompt_Workflow_Architecture.md:226:* `UNREVIEWED`
tools/Codex_Prompt_Workflow_Architecture.md:230:* `ACCEPTED`
tools/Codex_Prompt_Workflow_Architecture.md:231:* `REJECTED`
tools/Codex_Prompt_Workflow_Architecture.md:346:### 5. Conservative queue progression
tools/Codex_Prompt_Workflow_Architecture.md:386:* confirm a new record stops at `UNREVIEWED`
tools/Codex_Prompt_Workflow_Architecture.md:387:* confirm only `ACCEPTED` would release the next prompt
tools/Codex_Prompt_Workflow_Architecture.md:388:* confirm `REJECTED` stops progression and preserves enough evidence for the next human decision

exec
/bin/bash -lc "sed -n '1,160p' notes/010_next_step_recommendation__20260415_234559.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 010 Next Step Recommendation

Chosen next move: `environment diagnosis prompt`

Why it should happen next:
The V1 runner candidate already produces the intended execution record shape closely enough, and the validation run failed during Codex session initialization with a read-only filesystem error. That makes environment diagnosis the smallest step that can unblock meaningful validation without broadening the build.

What it should produce:
A short diagnosis artifact that identifies where session startup is trying to write, whether the failure reproduces independently of `tools/codex/run_prompt.py`, and one bounded follow-up action.

What should explicitly wait:
Runner polish, review write-back helpers, and any broader workflow or runner build-out.

exec
/bin/bash -lc 'date -u +%Y%m%d_%H%M%S' in /data/git/handson-ml3
 succeeded in 0ms:
20260416_005130

codex
I’ve finished the comparison. I’m writing the two requested notes now: one full sweep report and one short recommendation note with a tight priority order.
  ✓ Read the specified architecture, implementation, and note files plus inspect relevant directories
  ✓ Assess alignment, identify drift/gaps, and decide practical remaining work
  → Create the two required notes artifacts with concise prioritized recommendations
apply patch
patch: completed
/data/git/handson-ml3/notes/018_architecture_vs_actual_sweep__20260416_005130.md
/data/git/handson-ml3/notes/018_prioritized_remaining_work__20260416_005130.md
diff --git a/notes/018_architecture_vs_actual_sweep__20260416_005130.md b/notes/018_architecture_vs_actual_sweep__20260416_005130.md
new file mode 100644
index 0000000000000000000000000000000000000000..847fe6bf96010bc30748b8469fe1e00044586786
--- /dev/null
+++ b/notes/018_architecture_vs_actual_sweep__20260416_005130.md
@@ -0,0 +1,121 @@
+# 018 Architecture Vs Actual Sweep
+
+## Short Summary
+
+The V1 prompt workflow is now real, not just specified. The repo has a working runner at `tools/codex/run_prompt.py`, an in-place review write-back helper at `tools/codex/review_run.py`, a queue-readiness checker at `tools/codex/check_queue_readiness.py`, and a review-backlog lister at `tools/codex/list_review_backlog.py`.
+
+The main remaining work is no longer core architecture implementation. It is cleanup around doc/spec drift, lightweight contract protection for the markdown record format, and a small amount of operational tightening before heavier repeated use.
+
+## Implemented Vs Intended
+
+### Implemented and operational
+
+- One bounded prompt run can be executed through `tools/codex/run_prompt.py`.
+- One durable V1 execution record is written into `notes/` with execution facts, review facts, failure-analysis placeholders, resource facts, prompt text, final output, and stderr.
+- Execution status and review status are cleanly separated. New records start at `review_status: UNREVIEWED`.
+- Manual review write-back exists through `tools/codex/review_run.py` and updates the same record in place.
+- Conservative queue progression is operational through `tools/codex/check_queue_readiness.py`.
+  - Only `ACCEPTED` on the immediately previous prompt releases the next prompt.
+- Review discovery is operational through `tools/codex/list_review_backlog.py`.
+  - The repo currently shows 9 V1 execution records, 8 unreviewed records, and latest unreviewed records for prompts `011` through `017`.
+
+### Partially implemented or narrower than the docs imply
+
+- Retry linkage exists only as a placeholder field. `retry_of_run_id` is present in the artifact but there is no thin helper or established operating path for using it consistently.
+- Validation exists as ad hoc note-based evidence, not as a durable contract test layer. The system currently depends on markdown shape stability across four scripts.
+- The architecture packet describes the core V1 shape well, but it does not reflect the current operational helper set as clearly as the repo now warrants.
+
+### Stale or lagging parts of the design packet
+
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md` is the clearest stale document.
+  - It still targets `tools/codex/baby_run_prompt.py`.
+  - It still names the runner field as `tools/codex/baby_run_prompt.py`.
+  - It still frames review write-back and queue logic as not yet implemented, while the repo now has separate helpers for both.
+- The execution-record spec says the stable V1 `run_id` is `<prompt_stem>__<started_at_utc>`, but `tools/codex/run_prompt.py` now adds same-second suffixes like `__2` when needed. The code is pragmatic; the spec lags.
+- `tools/Codex_Prompt_Workflow_Architecture.md` is broadly right, but it now underspecifies actual repo reality. The practical V1 toolset is no longer just runner plus manual review conceptually; it includes explicit readiness and backlog helpers.
+
+## Remaining Work
+
+### 1. Align the design packet to the actual V1 toolset
+
+Why it matters:
+The repo has outgrown parts of the design packet. The next prompt or reviewer could still be steered by stale references to `baby_run_prompt.py`, by the old strict run-id rule, or by the older assumption that readiness/backlog support does not exist.
+
+Evidence:
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- actual scripts in `tools/codex/`
+
+### 2. Add lightweight contract checks for the V1 markdown record
+
+Why it matters:
+`run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py` all rely on the same markdown field and section conventions. That contract is now central to the workflow, but it is only protected by careful coding and prior notes, not by a small repeatable test layer.
+
+Evidence:
+- repeated required-section and required-field parsing logic in `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py`
+- current validation is documented in notes rather than locked in as an executable check
+
+### 3. Tighten operational guidance around legacy notes and the existing review backlog
+
+Why it matters:
+The tooling is correct, but current repo state is mixed:
+- legacy `__SUCCESS__` notes still exist for older prompts
+- latest V1 reviews for `011` through `017` are still `UNREVIEWED`
+
+That is not a V1 architecture failure, but it does create friction when interpreting readiness and when deciding whether to trust the queue as a day-to-day operational surface.
+
+Evidence:
+- `tools/codex/check_queue_readiness.py` needed a special explanatory `Queue note:`
+- `tools/codex/list_review_backlog.py` currently reports 8 unreviewed records
+
+### 4. Defer retry-linkage tooling and richer queue semantics
+
+Why it matters:
+These are plausible future improvements, but they are not required to use the current V1 slice for real work. Building them now would expand the system before the present workflow has been used enough to justify it.
+
+Evidence:
+- `retry_of_run_id` remains unused in normal flow
+- current V1 already covers run, record, review, backlog, and readiness
+
+## Prioritized Ranking
+
+1. `Doc/spec alignment cleanup`
+Why: removes the most immediate source of incorrect future work and makes the architecture packet trustworthy again.
+Expected risk: `low`
+Expected payoff: `high`
+Recommended timing: `next`
+
+2. `Record-contract validation`
+Why: protects the central markdown artifact contract shared by all current V1 tools.
+Expected risk: `low`
+Expected payoff: `high`
+Recommended timing: `soon`
+
+3. `Operational guidance cleanup for legacy notes and open review backlog`
+Why: reduces day-to-day confusion without adding new platform surface.
+Expected risk: `low`
+Expected payoff: `medium`
+Recommended timing: `soon`
+
+4. `Retry-linkage helper or richer queue semantics`
+Why: useful later, but not justified before the current V1 flow sees more real use.
+Expected risk: `medium`
+Expected payoff: `medium`
+Recommended timing: `explicitly defer`
+
+## Judgment
+
+Yes. The V1 prompt workflow system is good enough to use for real work now, provided it is used in the intended conservative way:
+
+- run one prompt
+- inspect the record
+- write back manual review
+- use readiness/backlog helpers to decide what is next
+
+The smallest remaining thing to clean up before using it heavily is doc/spec alignment, especially:
+
+- update the bridge-runner spec to reflect `run_prompt.py`
+- reflect that review write-back, readiness, and backlog helpers now exist
+- align the run-id collision rule with actual behavior
+
+That is the shortest path to making the current V1 implementation and the architecture packet agree on what the system actually is.
diff --git a/notes/018_prioritized_remaining_work__20260416_005130.md b/notes/018_prioritized_remaining_work__20260416_005130.md
new file mode 100644
index 0000000000000000000000000000000000000000..d46407a92f4f23dc93ef1692e27cd956bb572e9a
--- /dev/null
+++ b/notes/018_prioritized_remaining_work__20260416_005130.md
@@ -0,0 +1,22 @@
+# 018 Prioritized Remaining Work
+
+Top remaining items:
+
+1. `Doc/spec alignment cleanup`
+- Do this next.
+- The repo now has `run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py`, but parts of the design packet still describe the older bridge-runner state and strict unsuffixed run ids.
+
+2. `Lightweight record-contract validation`
+- Do this soon.
+- The V1 workflow now depends on one markdown record shape shared across four scripts. A small repeatable check is the main technical safeguard still missing.
+
+3. `Operational guidance cleanup for legacy notes and the current unreviewed backlog`
+- Do this soon.
+- The system works, but mixed legacy notes plus unreviewed latest records still make regular use noisier than it needs to be.
+
+What should explicitly wait:
+
+- retry-linkage tooling
+- richer queue states or scheduling
+- broader runner refactors
+- any platform-style expansion beyond the current V1 slice

diff --git a/notes/018_architecture_vs_actual_sweep__20260416_005130.md b/notes/018_architecture_vs_actual_sweep__20260416_005130.md
new file mode 100644
index 0000000000000000000000000000000000000000..847fe6bf96010bc30748b8469fe1e00044586786
--- /dev/null
+++ b/notes/018_architecture_vs_actual_sweep__20260416_005130.md
@@ -0,0 +1,121 @@
+# 018 Architecture Vs Actual Sweep
+
+## Short Summary
+
+The V1 prompt workflow is now real, not just specified. The repo has a working runner at `tools/codex/run_prompt.py`, an in-place review write-back helper at `tools/codex/review_run.py`, a queue-readiness checker at `tools/codex/check_queue_readiness.py`, and a review-backlog lister at `tools/codex/list_review_backlog.py`.
+
+The main remaining work is no longer core architecture implementation. It is cleanup around doc/spec drift, lightweight contract protection for the markdown record format, and a small amount of operational tightening before heavier repeated use.
+
+## Implemented Vs Intended
+
+### Implemented and operational
+
+- One bounded prompt run can be executed through `tools/codex/run_prompt.py`.
+- One durable V1 execution record is written into `notes/` with execution facts, review facts, failure-analysis placeholders, resource facts, prompt text, final output, and stderr.
+- Execution status and review status are cleanly separated. New records start at `review_status: UNREVIEWED`.
+- Manual review write-back exists through `tools/codex/review_run.py` and updates the same record in place.
+- Conservative queue progression is operational through `tools/codex/check_queue_readiness.py`.
+  - Only `ACCEPTED` on the immediately previous prompt releases the next prompt.
+- Review discovery is operational through `tools/codex/list_review_backlog.py`.
+  - The repo currently shows 9 V1 execution records, 8 unreviewed records, and latest unreviewed records for prompts `011` through `017`.
+
+### Partially implemented or narrower than the docs imply
+
+- Retry linkage exists only as a placeholder field. `retry_of_run_id` is present in the artifact but there is no thin helper or established operating path for using it consistently.
+- Validation exists as ad hoc note-based evidence, not as a durable contract test layer. The system currently depends on markdown shape stability across four scripts.
+- The architecture packet describes the core V1 shape well, but it does not reflect the current operational helper set as clearly as the repo now warrants.
+
+### Stale or lagging parts of the design packet
+
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md` is the clearest stale document.
+  - It still targets `tools/codex/baby_run_prompt.py`.
+  - It still names the runner field as `tools/codex/baby_run_prompt.py`.
+  - It still frames review write-back and queue logic as not yet implemented, while the repo now has separate helpers for both.
+- The execution-record spec says the stable V1 `run_id` is `<prompt_stem>__<started_at_utc>`, but `tools/codex/run_prompt.py` now adds same-second suffixes like `__2` when needed. The code is pragmatic; the spec lags.
+- `tools/Codex_Prompt_Workflow_Architecture.md` is broadly right, but it now underspecifies actual repo reality. The practical V1 toolset is no longer just runner plus manual review conceptually; it includes explicit readiness and backlog helpers.
+
+## Remaining Work
+
+### 1. Align the design packet to the actual V1 toolset
+
+Why it matters:
+The repo has outgrown parts of the design packet. The next prompt or reviewer could still be steered by stale references to `baby_run_prompt.py`, by the old strict run-id rule, or by the older assumption that readiness/backlog support does not exist.
+
+Evidence:
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- actual scripts in `tools/codex/`
+
+### 2. Add lightweight contract checks for the V1 markdown record
+
+Why it matters:
+`run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py` all rely on the same markdown field and section conventions. That contract is now central to the workflow, but it is only protected by careful coding and prior notes, not by a small repeatable test layer.
+
+Evidence:
+- repeated required-section and required-field parsing logic in `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py`
+- current validation is documented in notes rather than locked in as an executable check
+
+### 3. Tighten operational guidance around legacy notes and the existing review backlog
+
+Why it matters:
+The tooling is correct, but current repo state is mixed:
+- legacy `__SUCCESS__` notes still exist for older prompts
+- latest V1 reviews for `011` through `017` are still `UNREVIEWED`
+
+That is not a V1 architecture failure, but it does create friction when interpreting readiness and when deciding whether to trust the queue as a day-to-day operational surface.
+
+Evidence:
+- `tools/codex/check_queue_readiness.py` needed a special explanatory `Queue note:`
+- `tools/codex/list_review_backlog.py` currently reports 8 unreviewed records
+
+### 4. Defer retry-linkage tooling and richer queue semantics
+
+Why it matters:
+These are plausible future improvements, but they are not required to use the current V1 slice for real work. Building them now would expand the system before the present workflow has been used enough to justify it.
+
+Evidence:
+- `retry_of_run_id` remains unused in normal flow
+- current V1 already covers run, record, review, backlog, and readiness
+
+## Prioritized Ranking
+
+1. `Doc/spec alignment cleanup`
+Why: removes the most immediate source of incorrect future work and makes the architecture packet trustworthy again.
+Expected risk: `low`
+Expected payoff: `high`
+Recommended timing: `next`
+
+2. `Record-contract validation`
+Why: protects the central markdown artifact contract shared by all current V1 tools.
+Expected risk: `low`
+Expected payoff: `high`
+Recommended timing: `soon`
+
+3. `Operational guidance cleanup for legacy notes and open review backlog`
+Why: reduces day-to-day confusion without adding new platform surface.
+Expected risk: `low`
+Expected payoff: `medium`
+Recommended timing: `soon`
+
+4. `Retry-linkage helper or richer queue semantics`
+Why: useful later, but not justified before the current V1 flow sees more real use.
+Expected risk: `medium`
+Expected payoff: `medium`
+Recommended timing: `explicitly defer`
+
+## Judgment
+
+Yes. The V1 prompt workflow system is good enough to use for real work now, provided it is used in the intended conservative way:
+
+- run one prompt
+- inspect the record
+- write back manual review
+- use readiness/backlog helpers to decide what is next
+
+The smallest remaining thing to clean up before using it heavily is doc/spec alignment, especially:
+
+- update the bridge-runner spec to reflect `run_prompt.py`
+- reflect that review write-back, readiness, and backlog helpers now exist
+- align the run-id collision rule with actual behavior
+
+That is the shortest path to making the current V1 implementation and the architecture packet agree on what the system actually is.
diff --git a/notes/018_prioritized_remaining_work__20260416_005130.md b/notes/018_prioritized_remaining_work__20260416_005130.md
new file mode 100644
index 0000000000000000000000000000000000000000..d46407a92f4f23dc93ef1692e27cd956bb572e9a
--- /dev/null
+++ b/notes/018_prioritized_remaining_work__20260416_005130.md
@@ -0,0 +1,22 @@
+# 018 Prioritized Remaining Work
+
+Top remaining items:
+
+1. `Doc/spec alignment cleanup`
+- Do this next.
+- The repo now has `run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py`, but parts of the design packet still describe the older bridge-runner state and strict unsuffixed run ids.
+
+2. `Lightweight record-contract validation`
+- Do this soon.
+- The V1 workflow now depends on one markdown record shape shared across four scripts. A small repeatable check is the main technical safeguard still missing.
+
+3. `Operational guidance cleanup for legacy notes and the current unreviewed backlog`
+- Do this soon.
+- The system works, but mixed legacy notes plus unreviewed latest records still make regular use noisier than it needs to be.
+
+What should explicitly wait:
+
+- retry-linkage tooling
+- richer queue states or scheduling
+- broader runner refactors
+- any platform-style expansion beyond the current V1 slice

exec
/bin/bash -lc "sed -n '1,260p' notes/018_architecture_vs_actual_sweep__20260416_005130.md && printf '
---
' && sed -n '1,220p' notes/018_prioritized_remaining_work__20260416_005130.md" in /data/git/handson-ml3
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

---
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

diff --git a/notes/018_architecture_vs_actual_sweep__20260416_005130.md b/notes/018_architecture_vs_actual_sweep__20260416_005130.md
new file mode 100644
index 0000000000000000000000000000000000000000..847fe6bf96010bc30748b8469fe1e00044586786
--- /dev/null
+++ b/notes/018_architecture_vs_actual_sweep__20260416_005130.md
@@ -0,0 +1,121 @@
+# 018 Architecture Vs Actual Sweep
+
+## Short Summary
+
+The V1 prompt workflow is now real, not just specified. The repo has a working runner at `tools/codex/run_prompt.py`, an in-place review write-back helper at `tools/codex/review_run.py`, a queue-readiness checker at `tools/codex/check_queue_readiness.py`, and a review-backlog lister at `tools/codex/list_review_backlog.py`.
+
+The main remaining work is no longer core architecture implementation. It is cleanup around doc/spec drift, lightweight contract protection for the markdown record format, and a small amount of operational tightening before heavier repeated use.
+
+## Implemented Vs Intended
+
+### Implemented and operational
+
+- One bounded prompt run can be executed through `tools/codex/run_prompt.py`.
+- One durable V1 execution record is written into `notes/` with execution facts, review facts, failure-analysis placeholders, resource facts, prompt text, final output, and stderr.
+- Execution status and review status are cleanly separated. New records start at `review_status: UNREVIEWED`.
+- Manual review write-back exists through `tools/codex/review_run.py` and updates the same record in place.
+- Conservative queue progression is operational through `tools/codex/check_queue_readiness.py`.
+  - Only `ACCEPTED` on the immediately previous prompt releases the next prompt.
+- Review discovery is operational through `tools/codex/list_review_backlog.py`.
+  - The repo currently shows 9 V1 execution records, 8 unreviewed records, and latest unreviewed records for prompts `011` through `017`.
+
+### Partially implemented or narrower than the docs imply
+
+- Retry linkage exists only as a placeholder field. `retry_of_run_id` is present in the artifact but there is no thin helper or established operating path for using it consistently.
+- Validation exists as ad hoc note-based evidence, not as a durable contract test layer. The system currently depends on markdown shape stability across four scripts.
+- The architecture packet describes the core V1 shape well, but it does not reflect the current operational helper set as clearly as the repo now warrants.
+
+### Stale or lagging parts of the design packet
+
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md` is the clearest stale document.
+  - It still targets `tools/codex/baby_run_prompt.py`.
+  - It still names the runner field as `tools/codex/baby_run_prompt.py`.
+  - It still frames review write-back and queue logic as not yet implemented, while the repo now has separate helpers for both.
+- The execution-record spec says the stable V1 `run_id` is `<prompt_stem>__<started_at_utc>`, but `tools/codex/run_prompt.py` now adds same-second suffixes like `__2` when needed. The code is pragmatic; the spec lags.
+- `tools/Codex_Prompt_Workflow_Architecture.md` is broadly right, but it now underspecifies actual repo reality. The practical V1 toolset is no longer just runner plus manual review conceptually; it includes explicit readiness and backlog helpers.
+
+## Remaining Work
+
+### 1. Align the design packet to the actual V1 toolset
+
+Why it matters:
+The repo has outgrown parts of the design packet. The next prompt or reviewer could still be steered by stale references to `baby_run_prompt.py`, by the old strict run-id rule, or by the older assumption that readiness/backlog support does not exist.
+
+Evidence:
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- actual scripts in `tools/codex/`
+
+### 2. Add lightweight contract checks for the V1 markdown record
+
+Why it matters:
+`run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py` all rely on the same markdown field and section conventions. That contract is now central to the workflow, but it is only protected by careful coding and prior notes, not by a small repeatable test layer.
+
+Evidence:
+- repeated required-section and required-field parsing logic in `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py`
+- current validation is documented in notes rather than locked in as an executable check
+
+### 3. Tighten operational guidance around legacy notes and the existing review backlog
+
+Why it matters:
+The tooling is correct, but current repo state is mixed:
+- legacy `__SUCCESS__` notes still exist for older prompts
+- latest V1 reviews for `011` through `017` are still `UNREVIEWED`
+
+That is not a V1 architecture failure, but it does create friction when interpreting readiness and when deciding whether to trust the queue as a day-to-day operational surface.
+
+Evidence:
+- `tools/codex/check_queue_readiness.py` needed a special explanatory `Queue note:`
+- `tools/codex/list_review_backlog.py` currently reports 8 unreviewed records
+
+### 4. Defer retry-linkage tooling and richer queue semantics
+
+Why it matters:
+These are plausible future improvements, but they are not required to use the current V1 slice for real work. Building them now would expand the system before the present workflow has been used enough to justify it.
+
+Evidence:
+- `retry_of_run_id` remains unused in normal flow
+- current V1 already covers run, record, review, backlog, and readiness
+
+## Prioritized Ranking
+
+1. `Doc/spec alignment cleanup`
+Why: removes the most immediate source of incorrect future work and makes the architecture packet trustworthy again.
+Expected risk: `low`
+Expected payoff: `high`
+Recommended timing: `next`
+
+2. `Record-contract validation`
+Why: protects the central markdown artifact contract shared by all current V1 tools.
+Expected risk: `low`
+Expected payoff: `high`
+Recommended timing: `soon`
+
+3. `Operational guidance cleanup for legacy notes and open review backlog`
+Why: reduces day-to-day confusion without adding new platform surface.
+Expected risk: `low`
+Expected payoff: `medium`
+Recommended timing: `soon`
+
+4. `Retry-linkage helper or richer queue semantics`
+Why: useful later, but not justified before the current V1 flow sees more real use.
+Expected risk: `medium`
+Expected payoff: `medium`
+Recommended timing: `explicitly defer`
+
+## Judgment
+
+Yes. The V1 prompt workflow system is good enough to use for real work now, provided it is used in the intended conservative way:
+
+- run one prompt
+- inspect the record
+- write back manual review
+- use readiness/backlog helpers to decide what is next
+
+The smallest remaining thing to clean up before using it heavily is doc/spec alignment, especially:
+
+- update the bridge-runner spec to reflect `run_prompt.py`
+- reflect that review write-back, readiness, and backlog helpers now exist
+- align the run-id collision rule with actual behavior
+
+That is the shortest path to making the current V1 implementation and the architecture packet agree on what the system actually is.
diff --git a/notes/018_prioritized_remaining_work__20260416_005130.md b/notes/018_prioritized_remaining_work__20260416_005130.md
new file mode 100644
index 0000000000000000000000000000000000000000..d46407a92f4f23dc93ef1692e27cd956bb572e9a
--- /dev/null
+++ b/notes/018_prioritized_remaining_work__20260416_005130.md
@@ -0,0 +1,22 @@
+# 018 Prioritized Remaining Work
+
+Top remaining items:
+
+1. `Doc/spec alignment cleanup`
+- Do this next.
+- The repo now has `run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py`, but parts of the design packet still describe the older bridge-runner state and strict unsuffixed run ids.
+
+2. `Lightweight record-contract validation`
+- Do this soon.
+- The V1 workflow now depends on one markdown record shape shared across four scripts. A small repeatable check is the main technical safeguard still missing.
+
+3. `Operational guidance cleanup for legacy notes and the current unreviewed backlog`
+- Do this soon.
+- The system works, but mixed legacy notes plus unreviewed latest records still make regular use noisier than it needs to be.
+
+What should explicitly wait:
+
+- retry-linkage tooling
+- richer queue states or scheduling
+- broader runner refactors
+- any platform-style expansion beyond the current V1 slice

  ✓ Read the specified architecture, implementation, and note files plus inspect relevant directories
  ✓ Assess alignment, identify drift/gaps, and decide practical remaining work
  ✓ Create the two required notes artifacts with concise prioritized recommendations
diff --git a/notes/018_architecture_vs_actual_sweep__20260416_005130.md b/notes/018_architecture_vs_actual_sweep__20260416_005130.md
new file mode 100644
index 0000000000000000000000000000000000000000..847fe6bf96010bc30748b8469fe1e00044586786
--- /dev/null
+++ b/notes/018_architecture_vs_actual_sweep__20260416_005130.md
@@ -0,0 +1,121 @@
+# 018 Architecture Vs Actual Sweep
+
+## Short Summary
+
+The V1 prompt workflow is now real, not just specified. The repo has a working runner at `tools/codex/run_prompt.py`, an in-place review write-back helper at `tools/codex/review_run.py`, a queue-readiness checker at `tools/codex/check_queue_readiness.py`, and a review-backlog lister at `tools/codex/list_review_backlog.py`.
+
+The main remaining work is no longer core architecture implementation. It is cleanup around doc/spec drift, lightweight contract protection for the markdown record format, and a small amount of operational tightening before heavier repeated use.
+
+## Implemented Vs Intended
+
+### Implemented and operational
+
+- One bounded prompt run can be executed through `tools/codex/run_prompt.py`.
+- One durable V1 execution record is written into `notes/` with execution facts, review facts, failure-analysis placeholders, resource facts, prompt text, final output, and stderr.
+- Execution status and review status are cleanly separated. New records start at `review_status: UNREVIEWED`.
+- Manual review write-back exists through `tools/codex/review_run.py` and updates the same record in place.
+- Conservative queue progression is operational through `tools/codex/check_queue_readiness.py`.
+  - Only `ACCEPTED` on the immediately previous prompt releases the next prompt.
+- Review discovery is operational through `tools/codex/list_review_backlog.py`.
+  - The repo currently shows 9 V1 execution records, 8 unreviewed records, and latest unreviewed records for prompts `011` through `017`.
+
+### Partially implemented or narrower than the docs imply
+
+- Retry linkage exists only as a placeholder field. `retry_of_run_id` is present in the artifact but there is no thin helper or established operating path for using it consistently.
+- Validation exists as ad hoc note-based evidence, not as a durable contract test layer. The system currently depends on markdown shape stability across four scripts.
+- The architecture packet describes the core V1 shape well, but it does not reflect the current operational helper set as clearly as the repo now warrants.
+
+### Stale or lagging parts of the design packet
+
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md` is the clearest stale document.
+  - It still targets `tools/codex/baby_run_prompt.py`.
+  - It still names the runner field as `tools/codex/baby_run_prompt.py`.
+  - It still frames review write-back and queue logic as not yet implemented, while the repo now has separate helpers for both.
+- The execution-record spec says the stable V1 `run_id` is `<prompt_stem>__<started_at_utc>`, but `tools/codex/run_prompt.py` now adds same-second suffixes like `__2` when needed. The code is pragmatic; the spec lags.
+- `tools/Codex_Prompt_Workflow_Architecture.md` is broadly right, but it now underspecifies actual repo reality. The practical V1 toolset is no longer just runner plus manual review conceptually; it includes explicit readiness and backlog helpers.
+
+## Remaining Work
+
+### 1. Align the design packet to the actual V1 toolset
+
+Why it matters:
+The repo has outgrown parts of the design packet. The next prompt or reviewer could still be steered by stale references to `baby_run_prompt.py`, by the old strict run-id rule, or by the older assumption that readiness/backlog support does not exist.
+
+Evidence:
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- actual scripts in `tools/codex/`
+
+### 2. Add lightweight contract checks for the V1 markdown record
+
+Why it matters:
+`run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py` all rely on the same markdown field and section conventions. That contract is now central to the workflow, but it is only protected by careful coding and prior notes, not by a small repeatable test layer.
+
+Evidence:
+- repeated required-section and required-field parsing logic in `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py`
+- current validation is documented in notes rather than locked in as an executable check
+
+### 3. Tighten operational guidance around legacy notes and the existing review backlog
+
+Why it matters:
+The tooling is correct, but current repo state is mixed:
+- legacy `__SUCCESS__` notes still exist for older prompts
+- latest V1 reviews for `011` through `017` are still `UNREVIEWED`
+
+That is not a V1 architecture failure, but it does create friction when interpreting readiness and when deciding whether to trust the queue as a day-to-day operational surface.
+
+Evidence:
+- `tools/codex/check_queue_readiness.py` needed a special explanatory `Queue note:`
+- `tools/codex/list_review_backlog.py` currently reports 8 unreviewed records
+
+### 4. Defer retry-linkage tooling and richer queue semantics
+
+Why it matters:
+These are plausible future improvements, but they are not required to use the current V1 slice for real work. Building them now would expand the system before the present workflow has been used enough to justify it.
+
+Evidence:
+- `retry_of_run_id` remains unused in normal flow
+- current V1 already covers run, record, review, backlog, and readiness
+
+## Prioritized Ranking
+
+1. `Doc/spec alignment cleanup`
+Why: removes the most immediate source of incorrect future work and makes the architecture packet trustworthy again.
+Expected risk: `low`
+Expected payoff: `high`
+Recommended timing: `next`
+
+2. `Record-contract validation`
+Why: protects the central markdown artifact contract shared by all current V1 tools.
+Expected risk: `low`
+Expected payoff: `high`
+Recommended timing: `soon`
+
+3. `Operational guidance cleanup for legacy notes and open review backlog`
+Why: reduces day-to-day confusion without adding new platform surface.
+Expected risk: `low`
+Expected payoff: `medium`
+Recommended timing: `soon`
+
+4. `Retry-linkage helper or richer queue semantics`
+Why: useful later, but not justified before the current V1 flow sees more real use.
+Expected risk: `medium`
+Expected payoff: `medium`
+Recommended timing: `explicitly defer`
+
+## Judgment
+
+Yes. The V1 prompt workflow system is good enough to use for real work now, provided it is used in the intended conservative way:
+
+- run one prompt
+- inspect the record
+- write back manual review
+- use readiness/backlog helpers to decide what is next
+
+The smallest remaining thing to clean up before using it heavily is doc/spec alignment, especially:
+
+- update the bridge-runner spec to reflect `run_prompt.py`
+- reflect that review write-back, readiness, and backlog helpers now exist
+- align the run-id collision rule with actual behavior
+
+That is the shortest path to making the current V1 implementation and the architecture packet agree on what the system actually is.
diff --git a/notes/018_prioritized_remaining_work__20260416_005130.md b/notes/018_prioritized_remaining_work__20260416_005130.md
new file mode 100644
index 0000000000000000000000000000000000000000..d46407a92f4f23dc93ef1692e27cd956bb572e9a
--- /dev/null
+++ b/notes/018_prioritized_remaining_work__20260416_005130.md
@@ -0,0 +1,22 @@
+# 018 Prioritized Remaining Work
+
+Top remaining items:
+
+1. `Doc/spec alignment cleanup`
+- Do this next.
+- The repo now has `run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py`, but parts of the design packet still describe the older bridge-runner state and strict unsuffixed run ids.
+
+2. `Lightweight record-contract validation`
+- Do this soon.
+- The V1 workflow now depends on one markdown record shape shared across four scripts. A small repeatable check is the main technical safeguard still missing.
+
+3. `Operational guidance cleanup for legacy notes and the current unreviewed backlog`
+- Do this soon.
+- The system works, but mixed legacy notes plus unreviewed latest records still make regular use noisier than it needs to be.
+
+What should explicitly wait:
+
+- retry-linkage tooling
+- richer queue states or scheduling
+- broader runner refactors
+- any platform-style expansion beyond the current V1 slice

codex
Created the two requested artifacts:

- [018_architecture_vs_actual_sweep__20260416_005130.md](/data/git/handson-ml3/notes/018_architecture_vs_actual_sweep__20260416_005130.md)
- [018_prioritized_remaining_work__20260416_005130.md](/data/git/handson-ml3/notes/018_prioritized_remaining_work__20260416_005130.md)

Bottom line: the V1 workflow is good enough for real work now. The next cleanup should be doc/spec alignment so the architecture packet matches the actual toolset (`run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, `list_review_backlog.py`) and the current run-id behavior. After that, the most justified technical follow-up is a small contract-validation layer for the markdown execution record.
diff --git a/notes/018_architecture_vs_actual_sweep__20260416_005130.md b/notes/018_architecture_vs_actual_sweep__20260416_005130.md
new file mode 100644
index 0000000000000000000000000000000000000000..847fe6bf96010bc30748b8469fe1e00044586786
--- /dev/null
+++ b/notes/018_architecture_vs_actual_sweep__20260416_005130.md
@@ -0,0 +1,121 @@
+# 018 Architecture Vs Actual Sweep
+
+## Short Summary
+
+The V1 prompt workflow is now real, not just specified. The repo has a working runner at `tools/codex/run_prompt.py`, an in-place review write-back helper at `tools/codex/review_run.py`, a queue-readiness checker at `tools/codex/check_queue_readiness.py`, and a review-backlog lister at `tools/codex/list_review_backlog.py`.
+
+The main remaining work is no longer core architecture implementation. It is cleanup around doc/spec drift, lightweight contract protection for the markdown record format, and a small amount of operational tightening before heavier repeated use.
+
+## Implemented Vs Intended
+
+### Implemented and operational
+
+- One bounded prompt run can be executed through `tools/codex/run_prompt.py`.
+- One durable V1 execution record is written into `notes/` with execution facts, review facts, failure-analysis placeholders, resource facts, prompt text, final output, and stderr.
+- Execution status and review status are cleanly separated. New records start at `review_status: UNREVIEWED`.
+- Manual review write-back exists through `tools/codex/review_run.py` and updates the same record in place.
+- Conservative queue progression is operational through `tools/codex/check_queue_readiness.py`.
+  - Only `ACCEPTED` on the immediately previous prompt releases the next prompt.
+- Review discovery is operational through `tools/codex/list_review_backlog.py`.
+  - The repo currently shows 9 V1 execution records, 8 unreviewed records, and latest unreviewed records for prompts `011` through `017`.
+
+### Partially implemented or narrower than the docs imply
+
+- Retry linkage exists only as a placeholder field. `retry_of_run_id` is present in the artifact but there is no thin helper or established operating path for using it consistently.
+- Validation exists as ad hoc note-based evidence, not as a durable contract test layer. The system currently depends on markdown shape stability across four scripts.
+- The architecture packet describes the core V1 shape well, but it does not reflect the current operational helper set as clearly as the repo now warrants.
+
+### Stale or lagging parts of the design packet
+
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md` is the clearest stale document.
+  - It still targets `tools/codex/baby_run_prompt.py`.
+  - It still names the runner field as `tools/codex/baby_run_prompt.py`.
+  - It still frames review write-back and queue logic as not yet implemented, while the repo now has separate helpers for both.
+- The execution-record spec says the stable V1 `run_id` is `<prompt_stem>__<started_at_utc>`, but `tools/codex/run_prompt.py` now adds same-second suffixes like `__2` when needed. The code is pragmatic; the spec lags.
+- `tools/Codex_Prompt_Workflow_Architecture.md` is broadly right, but it now underspecifies actual repo reality. The practical V1 toolset is no longer just runner plus manual review conceptually; it includes explicit readiness and backlog helpers.
+
+## Remaining Work
+
+### 1. Align the design packet to the actual V1 toolset
+
+Why it matters:
+The repo has outgrown parts of the design packet. The next prompt or reviewer could still be steered by stale references to `baby_run_prompt.py`, by the old strict run-id rule, or by the older assumption that readiness/backlog support does not exist.
+
+Evidence:
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- actual scripts in `tools/codex/`
+
+### 2. Add lightweight contract checks for the V1 markdown record
+
+Why it matters:
+`run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py` all rely on the same markdown field and section conventions. That contract is now central to the workflow, but it is only protected by careful coding and prior notes, not by a small repeatable test layer.
+
+Evidence:
+- repeated required-section and required-field parsing logic in `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py`
+- current validation is documented in notes rather than locked in as an executable check
+
+### 3. Tighten operational guidance around legacy notes and the existing review backlog
+
+Why it matters:
+The tooling is correct, but current repo state is mixed:
+- legacy `__SUCCESS__` notes still exist for older prompts
+- latest V1 reviews for `011` through `017` are still `UNREVIEWED`
+
+That is not a V1 architecture failure, but it does create friction when interpreting readiness and when deciding whether to trust the queue as a day-to-day operational surface.
+
+Evidence:
+- `tools/codex/check_queue_readiness.py` needed a special explanatory `Queue note:`
+- `tools/codex/list_review_backlog.py` currently reports 8 unreviewed records
+
+### 4. Defer retry-linkage tooling and richer queue semantics
+
+Why it matters:
+These are plausible future improvements, but they are not required to use the current V1 slice for real work. Building them now would expand the system before the present workflow has been used enough to justify it.
+
+Evidence:
+- `retry_of_run_id` remains unused in normal flow
+- current V1 already covers run, record, review, backlog, and readiness
+
+## Prioritized Ranking
+
+1. `Doc/spec alignment cleanup`
+Why: removes the most immediate source of incorrect future work and makes the architecture packet trustworthy again.
+Expected risk: `low`
+Expected payoff: `high`
+Recommended timing: `next`
+
+2. `Record-contract validation`
+Why: protects the central markdown artifact contract shared by all current V1 tools.
+Expected risk: `low`
+Expected payoff: `high`
+Recommended timing: `soon`
+
+3. `Operational guidance cleanup for legacy notes and open review backlog`
+Why: reduces day-to-day confusion without adding new platform surface.
+Expected risk: `low`
+Expected payoff: `medium`
+Recommended timing: `soon`
+
+4. `Retry-linkage helper or richer queue semantics`
+Why: useful later, but not justified before the current V1 flow sees more real use.
+Expected risk: `medium`
+Expected payoff: `medium`
+Recommended timing: `explicitly defer`
+
+## Judgment
+
+Yes. The V1 prompt workflow system is good enough to use for real work now, provided it is used in the intended conservative way:
+
+- run one prompt
+- inspect the record
+- write back manual review
+- use readiness/backlog helpers to decide what is next
+
+The smallest remaining thing to clean up before using it heavily is doc/spec alignment, especially:
+
+- update the bridge-runner spec to reflect `run_prompt.py`
+- reflect that review write-back, readiness, and backlog helpers now exist
+- align the run-id collision rule with actual behavior
+
+That is the shortest path to making the current V1 implementation and the architecture packet agree on what the system actually is.
diff --git a/notes/018_prioritized_remaining_work__20260416_005130.md b/notes/018_prioritized_remaining_work__20260416_005130.md
new file mode 100644
index 0000000000000000000000000000000000000000..d46407a92f4f23dc93ef1692e27cd956bb572e9a
--- /dev/null
+++ b/notes/018_prioritized_remaining_work__20260416_005130.md
@@ -0,0 +1,22 @@
+# 018 Prioritized Remaining Work
+
+Top remaining items:
+
+1. `Doc/spec alignment cleanup`
+- Do this next.
+- The repo now has `run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py`, but parts of the design packet still describe the older bridge-runner state and strict unsuffixed run ids.
+
+2. `Lightweight record-contract validation`
+- Do this soon.
+- The V1 workflow now depends on one markdown record shape shared across four scripts. A small repeatable check is the main technical safeguard still missing.
+
+3. `Operational guidance cleanup for legacy notes and the current unreviewed backlog`
+- Do this soon.
+- The system works, but mixed legacy notes plus unreviewed latest records still make regular use noisier than it needs to be.
+
+What should explicitly wait:
+
+- retry-linkage tooling
+- richer queue states or scheduling
+- broader runner refactors
+- any platform-style expansion beyond the current V1 slice

tokens used
60,406
```
