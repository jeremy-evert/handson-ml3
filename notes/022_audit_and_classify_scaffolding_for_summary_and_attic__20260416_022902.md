# 022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902

- run_id: `022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902`
- prompt_file: `codex_prompts/022_audit_and_classify_scaffolding_for_summary_and_attic.md`
- prompt_stem: `022_audit_and_classify_scaffolding_for_summary_and_attic`
- started_at_utc: `20260416_022902`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260416_023257`
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

- elapsed_seconds: `234.721`
- final_output_char_count: `670`
- stderr_char_count: `571539`

## Prompt Text

```md
# Task: Audit notes and prompt scaffolding, then classify what should be kept, summarized, or moved to an attic

You are working in this repository.

Your task is to inspect the current `notes/` and `codex_prompts/` folders and produce a conservative classification plan for cleanup.

## Important framing

This is an audit and classification task only.

Do NOT move files.
Do NOT delete files.
Do NOT rewrite tools.
Do NOT rewrite major design documents.
Do NOT create the attic in this pass.

Your job is to classify the current scaffolding so later passes can safely summarize and move it without losing durable knowledge.

## Files and folders to inspect

Read and inspect:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `tools/codex/v1_record_validation.py`
- all files in `notes/`
- all files in `codex_prompts/`

You may also inspect the current repo tree to understand the broader context, but the classification target is specifically:

- `notes/`
- `codex_prompts/`

## Goal

Produce a conservative classification plan that distinguishes:

1. permanent residents
2. summarize, then move to attic
3. move to attic without summary
4. uncertain / needs human review

## Classification standard

A file should be treated as worth summarizing if it contains one or more of the following:

- a decision that still governs current behavior
- explanation of why the current design exists
- validation evidence for a still-live tool
- meaningful scope or priority decisions that still matter
- important failure findings that shaped the final design
- operational guidance that would still help a future maintainer

A file should be treated as likely scaffolding if it is mainly:

- a transient execution receipt
- an intermediate prompt-generation step
- a superseded planning or sequencing artifact
- a redundant success note from before the V1 system stabilized
- a local construction artifact whose main purpose was building the current toolset

## Required output artifacts

Create exactly two artifacts.

### Artifact 1

Create:

`notes/022_scaffolding_classification_report__TIMESTAMP.md`

This report should include:

- short summary
- classification criteria used
- permanent residents
- summarize-then-attic list
- attic-without-summary list
- uncertain list

For each listed file, include a short reason.

Be specific enough that a later prompt can act on this report without re-arguing every classification.

### Artifact 2

Create:

`notes/022_scaffolding_cleanup_plan__TIMESTAMP.md`

This note should contain only:

- what should happen in the next pass
- what should happen after that
- what should explicitly not happen yet

Keep this second note brief and operational.

## Constraints

1. Be conservative.
2. Prefer `uncertain` over aggressive classification when in doubt.
3. Do not move or delete anything in this pass.
4. Do not collapse the entire build history into one sentence.
5. Keep the plan practical enough to drive the next prompt.
6. Focus the classification on `notes/` and `codex_prompts/`, while using the tool and design docs only as context for deciding what is durable.

## Success criteria

This task is successful if:

- the current notes and prompts are classified into the four categories above
- the reasoning is concrete and repo-specific
- the result is safe enough to use as the basis for a summary-extraction pass
- no files are moved or deleted
```

## Codex Final Output

Created [022_scaffolding_classification_report__20260416_023016.md](/data/git/handson-ml3/notes/022_scaffolding_classification_report__20260416_023016.md) and [022_scaffolding_cleanup_plan__20260416_023016.md](/data/git/handson-ml3/notes/022_scaffolding_cleanup_plan__20260416_023016.md).

The report classifies `notes/` and `codex_prompts/` into the four requested buckets, with file-specific reasons. The main conservative call is that current V1 execution records remain `uncertain` because `check_queue_readiness.py` and `list_review_backlog.py` still treat them as live source-of-truth data, so moving them now would change behavior. No files were moved or deleted.

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
session id: 019d941e-ddbf-7dd0-bd88-ecbe75f1e3a6
--------
user
# Task: Audit notes and prompt scaffolding, then classify what should be kept, summarized, or moved to an attic

You are working in this repository.

Your task is to inspect the current `notes/` and `codex_prompts/` folders and produce a conservative classification plan for cleanup.

## Important framing

This is an audit and classification task only.

Do NOT move files.
Do NOT delete files.
Do NOT rewrite tools.
Do NOT rewrite major design documents.
Do NOT create the attic in this pass.

Your job is to classify the current scaffolding so later passes can safely summarize and move it without losing durable knowledge.

## Files and folders to inspect

Read and inspect:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `tools/codex/v1_record_validation.py`
- all files in `notes/`
- all files in `codex_prompts/`

You may also inspect the current repo tree to understand the broader context, but the classification target is specifically:

- `notes/`
- `codex_prompts/`

## Goal

Produce a conservative classification plan that distinguishes:

1. permanent residents
2. summarize, then move to attic
3. move to attic without summary
4. uncertain / needs human review

## Classification standard

A file should be treated as worth summarizing if it contains one or more of the following:

- a decision that still governs current behavior
- explanation of why the current design exists
- validation evidence for a still-live tool
- meaningful scope or priority decisions that still matter
- important failure findings that shaped the final design
- operational guidance that would still help a future maintainer

A file should be treated as likely scaffolding if it is mainly:

- a transient execution receipt
- an intermediate prompt-generation step
- a superseded planning or sequencing artifact
- a redundant success note from before the V1 system stabilized
- a local construction artifact whose main purpose was building the current toolset

## Required output artifacts

Create exactly two artifacts.

### Artifact 1

Create:

`notes/022_scaffolding_classification_report__TIMESTAMP.md`

This report should include:

- short summary
- classification criteria used
- permanent residents
- summarize-then-attic list
- attic-without-summary list
- uncertain list

For each listed file, include a short reason.

Be specific enough that a later prompt can act on this report without re-arguing every classification.

### Artifact 2

Create:

`notes/022_scaffolding_cleanup_plan__TIMESTAMP.md`

This note should contain only:

- what should happen in the next pass
- what should happen after that
- what should explicitly not happen yet

Keep this second note brief and operational.

## Constraints

1. Be conservative.
2. Prefer `uncertain` over aggressive classification when in doubt.
3. Do not move or delete anything in this pass.
4. Do not collapse the entire build history into one sentence.
5. Keep the plan practical enough to drive the next prompt.
6. Focus the classification on `notes/` and `codex_prompts/`, while using the tool and design docs only as context for deciding what is durable.

## Success criteria

This task is successful if:

- the current notes and prompts are classified into the four categories above
- the reasoning is concrete and repo-specific
- the result is safe enough to use as the basis for a summary-extraction pass
- no files are moved or deleted

codex
I’m auditing `notes/` and `codex_prompts/` against the current V1 workflow/tooling so I can classify what is durable versus scaffolding. I’ll read the architecture and tool docs first, then inventory the target folders and write the two requested notes without moving or deleting anything.
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

### 6. Current V1 helper scripts

The current implemented V1 toolset is:

* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
* `tools/codex/review_run.py` for manual review write-back into the same record
* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`

These helpers operate on the same markdown execution-record contract.
They are part of the current V1 workflow surface, not future placeholders.

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
/bin/bash -lc 'rg --files tools notes codex_prompts' in /data/git/handson-ml3
 succeeded in 0ms:
codex_prompts/001_smoke_test_pipeline.md
codex_prompts/003_revise_Project_Deisgn_workflow_document.md
codex_prompts/002_repo_inventory_and_status.md
codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md
codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md
codex_prompts/022_audit_and_classify_scaffolding_for_summary_and_attic.md
codex_prompts/021_add_lightweight_v1_record_contract_validation.md
codex_prompts/020_align_v1_doc_and_spec_packet_to_actual_toolset.md
codex_prompts/019_generate_prompts_for_doc_alignment_and_record_contract_validation.md
codex_prompts/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work.md
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
tools/Codex_Prompt_Workflow_Architecture.md
tools/codex/V1_Run_Review_Gate.md
tools/codex/v1_record_validation.py
tools/codex/review_run.py
tools/Project_Design_Workflow.md
tools/codex/baby_run_prompt.py
tools/codex/V1_Execution_Record_Artifact.md
tools/codex/V1_Bridge_Runner_Change_Spec.md
tools/codex/check_queue_readiness.py
tools/codex/run_prompt.py
tools/codex/list_review_backlog.py
notes/011_build_v1_review_writeback_helper__20260415_235346.md
notes/011_review_writeback_helper_build__20260415_235514.md
notes/001_smoke_test_pipeline__20260415_234918.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md
notes/010_next_step_recommendation__20260415_234559.md
notes/010_run_prompt_candidate_review__20260415_234559.md
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md
notes/009_run_prompt_candidate_build__20260415_233407.md
notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md
notes/001_smoke_test_pipeline__20260415_233343.md
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md
notes/016_next_improvement_recommendation__20260416_003710.md
notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md
notes/016_queue_and_backlog_helper_validation__20260416_003710.md
notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md
notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md
notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md
notes/005_prompt_queue_plan__20260415_202557.md
notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md
notes/015_review_backlog_lister_build__20260416_010500.md
notes/004_next_design_step_recommendation__20260415_195538.md
notes/014_build_queue_readiness_checker__20260416_002319.md
notes/004_architecture_and_bridge_runner_review__20260415_195538.md
notes/003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md
notes/014_queue_readiness_checker_build__20260416_002419.md
notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md
notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md
notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md
notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md
notes/013_next_two_prompt_plan__20260416_002005.md
notes/012_top_three_next_options__20260416_000819.md
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md
notes/012_v1_pipeline_options_review__20260416_000819.md
notes/019_next_two_cleanup_prompts_plan__20260416_010251.md
notes/020_align_v1_doc_and_spec_packet_to_actual_toolset__20260416_010453.md
notes/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work__20260416_005034.md
notes/020_doc_spec_alignment_cleanup__20260416_010534.md
notes/018_prioritized_remaining_work__20260416_005130.md
notes/019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209.md
notes/018_architecture_vs_actual_sweep__20260416_005130.md
notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md
notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md
notes/021_record_contract_validation__20260416_011314.md
notes/021_add_lightweight_v1_record_contract_validation__20260416_010810.md

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

`tools/codex/run_prompt.py` uses that base identity and adds a numeric suffix such as `__2` only when needed to avoid a same-second filename collision for the same prompt.

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

exec
/bin/bash -lc "sed -n '1,260p' tools/codex/V1_Bridge_Runner_Change_Spec.md" in /data/git/handson-ml3
 succeeded in 0ms:
# V1 Bridge Runner Change Spec

## Purpose

Describe the current V1 bridge runner and adjacent helpers so this spec matches the implemented workflow without a broader refactor.

The target outcome is:

- execution produces one durable V1 execution record
- execution outcome and review outcome are kept separate
- run identity is stable
- the runner captures only the minimum automatic fields
- human review fields remain manual

## Scope

This spec covers the current V1 runner at [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py) and the small helper scripts that operate on the same execution-record contract.

It defines:

- the record filename pattern
- the markdown section order
- the exact fields the runner should populate automatically
- the manual review fields the runner should initialize for later completion
- the minimal mapping from subprocess result to execution status
- the minimal metrics to capture now
- the current helper support around review write-back, queue readiness, and review backlog inspection

## Current Behavior Summary

The current runner at `tools/codex/run_prompt.py`:

- resolves one prompt path from an argument using the existing lookup rules
- validates that `codex_prompts/` and `notes/` exist
- reads the prompt text from the selected file
- runs `codex exec -C <repo_root> --output-last-message <tempfile> -`
- captures subprocess return code, last-message output, and stderr
- writes one markdown execution record into `notes/`
- uses base run identity `<prompt_stem>__<started_at_utc>`
- adds a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt
- initializes `review_status: UNREVIEWED`
- prints the written note path
- exits with the subprocess return code

The current helper set also includes:

- `tools/codex/review_run.py` for manual review write-back into the same record
- `tools/codex/check_queue_readiness.py` for conservative readiness checks
- `tools/codex/list_review_backlog.py` for backlog inspection from `notes/`

The current gap is therefore no longer missing helper support.
The remaining need is only to keep this document aligned to the implemented V1 behavior.

## Required V1 Changes

### 1. Replace success/failure note naming with stable run identity naming

The runner should stop writing filenames in the form:

`<prompt_stem>__<status>__<timestamp>.md`

It should instead write:

`notes/<prompt_stem>__<started_at_utc>.md`

This is the base V1 identity.
If that path already exists for the same prompt and second, `run_prompt.py` appends a numeric suffix:

`notes/<prompt_stem>__<started_at_utc>__2.md`

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
- new queue states or queue engines beyond the current V1 readiness rule
- retry orchestration
- dependency-aware scheduling
- broader note/history discovery logic
- structured sidecars, JSON records, or databases

This document is an alignment spec for the current V1 slice.
It is not a request to redesign the already-implemented helper scripts.

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
- runner: `tools/codex/run_prompt.py`
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

- base V1 identity -> `f"{prompt_path.stem}__{started_at_utc}"`
- `run_id` -> base identity unless a same-second collision exists for the same prompt, then `f"{base_run_id}__{n}"`
- filename -> `notes/{run_id}.md`
- `prompt_stem` -> `prompt_path.stem`
- `started_at_utc` -> timestamp captured immediately before `run_codex(...)`


codex
I’ve got the governing workflow and V1 architecture context. Next I’m reading the actual helper scripts and every file under `notes/` and `codex_prompts/` so the classification is based on content, not just filenames.
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

from v1_record_validation import validate_record_text


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
    validate_record_text(record_path.read_text(encoding="utf-8"), source=record_path)

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

from v1_record_validation import REVIEW_STATUS_VALUES, validate_record_text


REVIEW_STATUSES = REVIEW_STATUS_VALUES - {"UNREVIEWED"}
FAILURE_FIELDS = (
    "failure_type",
    "failure_symptom",
    "likely_cause",
    "recommended_next_action",
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
        validate_record_text(text, source=record_path)

        updated = text
        for field, (value, code) in build_updates(args).items():
            updated = replace_field(updated, field, value, code=code)

        validate_record_text(updated, source=record_path)
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

from v1_record_validation import V1Record, ValidationError, parse_record_file


PROMPTS_DIR = "codex_prompts"
NOTES_DIR = "notes"
PROMPT_NAME_RE = re.compile(r"^(?P<prefix>\d+)_")


@dataclass(frozen=True)
class PromptEntry:
    prefix: int
    path: Path

    @property
    def label(self) -> str:
        return self.path.as_posix()


@dataclass(frozen=True)
class ReadinessResult:
    target: PromptEntry
    previous: PromptEntry | None
    latest_record: V1Record | None
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


def discover_run_records(root: Path) -> list[V1Record]:
    notes_dir = root / NOTES_DIR
    if not notes_dir.exists():
        raise ReadinessError(f"missing notes directory: {notes_dir}")

    records: list[V1Record] = []
    for path in sorted(notes_dir.glob("*.md")):
        try:
            record = parse_record_file(path)
        except ValidationError as exc:
            raise ReadinessError(str(exc)) from exc
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


def latest_record_for_prompt(records: list[V1Record], prompt: PromptEntry) -> V1Record | None:
    relevant = [record for record in records if record.prompt_file == prompt.label]
    if not relevant:
        return None
    return max(relevant, key=lambda record: (record.started_at_utc, record.run_suffix))


def default_target(prompts: list[PromptEntry], records: list[V1Record]) -> PromptEntry:
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
    records: list[V1Record],
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
    records: list[V1Record],
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


exec
/bin/bash -lc "sed -n '1,260p' tools/codex/list_review_backlog.py" in /data/git/handson-ml3
 succeeded in 0ms:
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from v1_record_validation import V1Record, ValidationError, parse_record_file


NOTES_DIR = "notes"


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


def discover_run_records(root: Path) -> list[V1Record]:
    notes_dir = root / NOTES_DIR
    if not notes_dir.exists():
        raise BacklogError(f"missing notes directory: {notes_dir}")

    records: list[V1Record] = []
    for path in sorted(notes_dir.glob("*.md")):
        try:
            record = parse_record_file(path)
        except ValidationError as exc:
            raise BacklogError(str(exc)) from exc
        if record is not None:
            records.append(record)
    return records


def latest_records_by_prompt(records: list[V1Record]) -> list[V1Record]:
    latest_by_prompt: dict[str, V1Record] = {}
    for record in records:
        current = latest_by_prompt.get(record.prompt_file)
        if current is None or (record.started_at_utc, record.run_suffix) > (
            current.started_at_utc,
            current.run_suffix,
        ):
            latest_by_prompt[record.prompt_file] = record

    return sorted(latest_by_prompt.values(), key=lambda record: record.prompt_stem)


def render_record(record: V1Record) -> str:
    return (
        f"- {record.path.as_posix()} | prompt={record.prompt_file} | started={record.started_at_utc} | "
        f"execution={record.execution_status} | review={record.review_status}"
    )


def print_section(title: str, records: list[V1Record]) -> None:
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


if __name__ == "__main__":
    raise SystemExit(main())

exec
/bin/bash -lc "sed -n '1,260p' tools/codex/v1_record_validation.py" in /data/git/handson-ml3
 succeeded in 0ms:
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


EXECUTION_STATUS_VALUES = {"EXECUTED", "EXECUTION_FAILED"}
REVIEW_STATUS_VALUES = {"UNREVIEWED", "ACCEPTED", "REJECTED"}
REQUIRED_SECTIONS = (
    "## Execution Facts",
    "## Review Facts",
    "## Failure Analysis",
    "## Resource / Cost Facts",
    "## Prompt Text",
    "## Codex Final Output",
    "## Stderr",
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
    "failure_type",
    "failure_symptom",
    "likely_cause",
    "recommended_next_action",
    "elapsed_seconds",
    "final_output_char_count",
    "stderr_char_count",
)
TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class V1Record:
    path: Path | None
    title: str
    run_id: str
    prompt_file: str
    prompt_stem: str
    started_at_utc: str
    execution_status: str
    finished_at_utc: str
    runner: str
    return_code: str
    retry_of_run_id: str
    review_status: str
    review_summary: str
    reviewed_by: str
    reviewed_at_utc: str
    failure_type: str
    failure_symptom: str
    likely_cause: str
    recommended_next_action: str
    elapsed_seconds: str
    final_output_char_count: str
    stderr_char_count: str
    run_suffix: int


def _source_label(source: Path | str | None) -> str:
    if source is None:
        return "<text>"
    if isinstance(source, Path):
        return source.as_posix()
    return source


def parse_field_line(text: str, field: str) -> str | None:
    match = re.search(rf"(?m)^- {re.escape(field)}:\s*(.*)$", text)
    if match is None:
        return None

    value = match.group(1).strip()
    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return value


def looks_like_v1_record(text: str) -> bool:
    preamble = text.split("```", 1)[0]
    has_run_id = parse_field_line(preamble, "run_id") is not None
    has_execution_section = "## Execution Facts" in preamble
    has_review_section = "## Review Facts" in preamble
    return has_run_id or (has_execution_section and has_review_section)


def parse_run_suffix(prompt_stem: str, started_at_utc: str, run_id: str) -> int:
    base_run_id = f"{prompt_stem}__{started_at_utc}"
    if run_id == base_run_id:
        return 1

    match = re.fullmatch(rf"{re.escape(base_run_id)}__(\d+)", run_id)
    if match is None:
        raise ValidationError(f"run_id does not match V1 identity pattern: {run_id}")

    suffix = int(match.group(1))
    if suffix < 2:
        raise ValidationError(f"run_id suffix must be >= 2 when present: {run_id}")
    return suffix


def validate_record_text(text: str, *, source: Path | str | None = None) -> V1Record:
    label = _source_label(source)
    if not text.startswith("# "):
        raise ValidationError(f"record does not start with a markdown title: {label}")

    positions: list[int] = []
    for section in REQUIRED_SECTIONS:
        pos = text.find(section)
        if pos == -1:
            raise ValidationError(f"record is missing section: {section} ({label})")
        positions.append(pos)
    if positions != sorted(positions):
        raise ValidationError(f"record sections are out of the expected V1 order: {label}")

    values: dict[str, str] = {}
    for field in REQUIRED_FIELDS:
        value = parse_field_line(text, field)
        if value is None:
            raise ValidationError(f"record is missing field line: {field} ({label})")
        values[field] = value

    title = text.splitlines()[0][2:].strip()
    run_id = values["run_id"]
    prompt_file = values["prompt_file"]
    prompt_stem = values["prompt_stem"]
    started_at_utc = values["started_at_utc"]
    execution_status = values["execution_status"]
    review_status = values["review_status"]

    if title != run_id:
        raise ValidationError(f"record title/run_id mismatch: {label}")
    if not TIMESTAMP_RE.fullmatch(started_at_utc):
        raise ValidationError(f"record has invalid started_at_utc timestamp: {label}")
    if execution_status not in EXECUTION_STATUS_VALUES:
        raise ValidationError(f"record has invalid execution_status: {label}")
    if review_status not in REVIEW_STATUS_VALUES:
        raise ValidationError(f"record has invalid review_status: {label}")

    if Path(prompt_file).stem != prompt_stem:
        raise ValidationError(f"record prompt_file/prompt_stem mismatch: {label}")

    run_suffix = parse_run_suffix(prompt_stem, started_at_utc, run_id)

    if isinstance(source, Path) and source.suffix == ".md" and source.stem != run_id:
        raise ValidationError(f"record filename/run_id mismatch: {label}")

    return V1Record(
        path=source if isinstance(source, Path) else None,
        title=title,
        run_id=run_id,
        prompt_file=prompt_file,
        prompt_stem=prompt_stem,
        started_at_utc=started_at_utc,
        execution_status=execution_status,
        finished_at_utc=values["finished_at_utc"],
        runner=values["runner"],
        return_code=values["return_code"],
        retry_of_run_id=values["retry_of_run_id"],
        review_status=review_status,
        review_summary=values["review_summary"],
        reviewed_by=values["reviewed_by"],
        reviewed_at_utc=values["reviewed_at_utc"],
        failure_type=values["failure_type"],
        failure_symptom=values["failure_symptom"],
        likely_cause=values["likely_cause"],
        recommended_next_action=values["recommended_next_action"],
        elapsed_seconds=values["elapsed_seconds"],
        final_output_char_count=values["final_output_char_count"],
        stderr_char_count=values["stderr_char_count"],
        run_suffix=run_suffix,
    )


def parse_record_file(path: Path) -> V1Record | None:
    text = path.read_text(encoding="utf-8")
    if not looks_like_v1_record(text):
        return None
    return validate_record_text(text, source=path)

exec
/bin/bash -lc "for f in codex_prompts/*.md; do printf '\\n===== %s =====\\n' \""'$f"; sed -n '"'1,240p' \""'$f"; done' in /data/git/handson-ml3
exec
/bin/bash -lc "for f in notes/*.md; do printf '\\n===== %s =====\\n' \""'$f"; sed -n '"'1,260p' \""'$f"; done' in /data/git/handson-ml3
 succeeded in 176ms:
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

## Success criteria

This task is successful if:

- the runner change scope is small and explicit
- the spec is directly grounded in the V1 execution record and review gate
- execution and review are no longer conflated in the target behavior
- the spec avoids a premature refactor

===== codex_prompts/009_build_v1_run_prompt_candidate.md =====
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

===== codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md =====
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

Explain why that is the smallest and safest next move.

## Required output artifacts

Create exactly two artifacts.

### Artifact 1
Create a review note at:

`notes/010_run_prompt_candidate_review__TIMESTAMP.md`

This note should include:

- short summary
- runner assessment
- artifact assessment
- operational assessment
- exactly one recommended next move

### Artifact 2
Create a short recommendation note at:

`notes/010_next_step_recommendation__TIMESTAMP.md`

This should contain only:

- the chosen next move
- why it should happen next
- what it should produce
- what should explicitly wait

Keep it brief and decisive.

## Constraints

1. Use the exact file paths listed above.
2. Recommend only one next step.
3. Do not produce a broad roadmap.
4. Do not modify code or design documents in this pass.
5. Keep the recommendation small enough to become the next Codex prompt.

## Success criteria

This task is successful if:

- the review clearly assesses both `run_prompt.py` and the sample execution record
- the next move is grounded in what actually happened
- only one next step is recommended
- the result helps us continue in a slow, deliberate, low-risk way

===== codex_prompts/011_build_v1_review_writeback_helper.md =====
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

===== codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md =====
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
- `notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md`
- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`

## Goal

Review the full current V1 workflow pipe and answer:

1. what pieces now exist and are functioning
2. what important gaps or awkward seams remain
3. what realistic next bounded options are available
4. which three options are the strongest next candidates

## What to evaluate

Please evaluate the current state of the pipe across these areas:

### 1. Execution path
- prompt resolution
- Codex invocation
- execution-record creation
- stability of record identity
- resource/failure evidence capture

### 2. Review path
- manual review write-back
- preservation of record structure
- consistency with the V1 review gate
- whether review is now operationally usable

### 3. Workflow usability
- whether the V1 flow is now practical for repeated use
- what still feels awkward, manual, fragile, or under-supported
- what is missing if this is going to be used regularly

### 4. Documentation alignment
- whether the code and notes still match the design packet
- whether there is any doc drift or naming drift that should be addressed
- whether there are any small correctness mismatches between design and implementation

## Required outputs

Create exactly two artifacts.

### Artifact 1
Create a review report at:

`notes/012_v1_pipeline_options_review__TIMESTAMP.md`

This report should include:

- short summary of current pipeline maturity
- what is working now
- what seams or gaps remain
- a list of realistic next bounded options

For the options list:
- provide at least 5 options if there are 5 credible ones
- provide fewer only if the repo truly supports fewer
- each option should include:
  - short name
  - what it would build or improve
  - why it matters
  - expected risk level: low / medium / high
  - expected payoff level: low / medium / high

### Artifact 2
Create a short recommendation note at:

`notes/012_top_three_next_options__TIMESTAMP.md`

This note should contain:

- the top three next options in ranked order
- why each made the top three
- which one should happen next
- what should explicitly wait

## Ranking guidance

When ranking options, prefer:
- bounded steps
- low-risk progress
- improvements that strengthen repeated operational use
- improvements that reduce ambiguity or manual fragility
- improvements supported by the current repo state

Avoid preferring:
- broad platform expansion
- premature refactors
- speculative subsystems
- large workflow engines

## Constraints

1. Use the exact file paths listed above.
2. Do not implement anything in this pass.
3. Do not produce a giant roadmap.
4. Recommend only bounded next options that are plausible from the current state.
5. Keep the tone practical and inspectable.
6. Let the repo evidence drive the options.

## Success criteria

This task is successful if:

- the review clearly describes what the V1 pipe can do now
- the options are grounded in actual repo artifacts and recent notes
- the report gives a realistic menu of next moves
- the recommendation note ranks the top three clearly
- the result helps a human choose the next deliberate step with confidence

===== codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md =====
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

===== codex_prompts/014_build_queue_readiness_checker.md =====
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

===== codex_prompts/015_build_review_backlog_unreviewed_run_lister.md =====
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

===== codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md =====
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

===== codex_prompts/017_polish_queue_readiness_gap_explanation.md =====
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

===== codex_prompts/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work.md =====
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

===== codex_prompts/019_generate_prompts_for_doc_alignment_and_record_contract_validation.md =====
# Task: Generate the next two implementation prompts for the current V1 workflow cleanup

You are working in this repository.

Your task is to write the next two bounded implementation prompt files based on the current architecture sweep and prioritized remaining work.

## Important framing

This is a prompt-generation task only.

Do NOT implement code or doc edits in this pass.
Do NOT modify existing tools or design documents in this pass.

Your job is to create the next two implementation prompts as files in `codex_prompts/`.

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
- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
- `notes/018_prioritized_remaining_work__20260416_005130.md`

## Goal

Generate exactly two prompt files:

1. a prompt to perform the recommended **doc/spec alignment cleanup**
2. a prompt to implement the recommended **lightweight record-contract validation**

These prompts should be implementation-ready, bounded, and consistent with the current repo state.

## Prompt 1 requirements

Create a prompt file for doc/spec alignment cleanup.

This future task should focus on aligning the design packet with the actual V1 toolset and current behavior, including at least:

- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- any stale references to `baby_run_prompt.py` where the actual V1 runner is now `run_prompt.py`
- any stale implication that readiness/backlog helpers do not yet exist
- the current run-id collision suffix behavior where relevant

The prompt should keep the cleanup bounded and should not turn into a broad architecture rewrite.

## Prompt 2 requirements

Create a prompt file for lightweight record-contract validation.

This future task should focus on creating a small repeatable validation layer for the shared V1 markdown execution-record shape used by:

- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

The prompt should keep the validation lightweight and inspectable.

Avoid proposing a large test framework, platform service, or dashboard.

## What each generated prompt must include

Each prompt file must:

- have a clear numeric filename prefix
- define one primary goal only
- name exact files to inspect
- define exact output artifacts to create
- include constraints that prevent broad platform growth
- define validation requirements
- define success criteria
- stay small enough to review comfortably before execution

## Also create one planning note

Create a short note at:

`notes/019_next_two_cleanup_prompts_plan__TIMESTAMP.md`

This note should summarize:

- why these two prompts were chosen
- why they are ordered this way
- which one should be executed first
- what should explicitly wait until later

## Constraints

1. Generate exactly two prompt files, no more.
2. Do not implement the cleanup tasks in this pass.
3. Keep both generated prompts bounded and V1-sized.
4. Do not introduce a broader orchestration layer, queue engine, retry manager, or platform expansion.
5. Let the architecture sweep drive the prompts.

## Success criteria

This task is successful if:

- exactly two prompt files are created
- one is for doc/spec alignment cleanup
- one is for lightweight record-contract validation
- both prompts are implementation-ready and bounded
- the planning note clearly explains the sequence

===== codex_prompts/020_align_v1_doc_and_spec_packet_to_actual_toolset.md =====
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

===== codex_prompts/021_add_lightweight_v1_record_contract_validation.md =====
# Task: Add lightweight validation for the shared V1 execution-record contract

You are working in this repository.

Your task is to add a small, repeatable validation layer for the shared markdown execution-record shape used across the current V1 tools.

Keep this implementation lightweight and inspectable.

## Primary goal

Create one small validation path that protects the shared V1 markdown record contract without introducing a larger framework or platform surface.

## Files to inspect

Read these exact files before editing:

- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
- `notes/018_prioritized_remaining_work__20260416_005130.md`

## Required implementation scope

Create a lightweight validation layer for the shared V1 markdown execution-record shape used by:

- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

The implementation should cover only the minimum contract that these tools already depend on, such as:

- required section presence and order
- required field-line presence
- allowed execution-status values
- allowed review-status values
- the expected `run_id` pattern including the optional same-second numeric suffix
- basic consistency checks like title/run-id agreement and prompt-file/prompt-stem agreement where appropriate

## Output artifacts to create

Create exactly these artifacts:

1. one small shared validation module under `tools/codex/`
2. only the minimal script changes needed so the current V1 tools reuse that validation instead of carrying separate ad hoc contract checks
3. one short implementation note:
   - `notes/021_record_contract_validation__TIMESTAMP.md`

## Constraints

- Do not build a larger test framework
- Do not introduce a service, daemon, dashboard, or platform layer
- Do not add a database, JSON sidecar, or alternate record format
- Do not redesign the V1 markdown artifact
- Do not expand queue semantics, retry tooling, or orchestration behavior
- Do not add broad dependency or configuration systems
- Keep the validation readable enough that a reviewer can inspect it quickly in one sitting

## Validation requirements

Validate the work by doing all of the following:

1. Run a lightweight syntax check on the touched Python files.
2. Run at least one small direct validation path against existing repo records in `notes/` so the shared validator proves it can parse or reject records using the current contract.
3. Confirm `run_prompt.py` still writes records that the shared validator accepts.
4. Confirm `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py` still work with the shared validator in place.
5. Keep validation evidence in the implementation note concise and concrete.

## Success criteria

This task is successful if:

1. The shared V1 markdown record contract is enforced through one lightweight reusable validation path.
2. The four V1 scripts no longer each carry their own independent, partially duplicated contract assumptions where simple sharing would suffice.
3. The validator remains small, inspectable, and local to the current V1 workflow.
4. No broader platform growth is introduced.
5. The result is still easy to review before execution.

===== codex_prompts/022_audit_and_classify_scaffolding_for_summary_and_attic.md =====
# Task: Audit notes and prompt scaffolding, then classify what should be kept, summarized, or moved to an attic

You are working in this repository.

Your task is to inspect the current `notes/` and `codex_prompts/` folders and produce a conservative classification plan for cleanup.

## Important framing

This is an audit and classification task only.

Do NOT move files.
Do NOT delete files.
Do NOT rewrite tools.
Do NOT rewrite major design documents.
Do NOT create the attic in this pass.

Your job is to classify the current scaffolding so later passes can safely summarize and move it without losing durable knowledge.

## Files and folders to inspect

Read and inspect:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `tools/codex/v1_record_validation.py`
- all files in `notes/`
- all files in `codex_prompts/`

You may also inspect the current repo tree to understand the broader context, but the classification target is specifically:

- `notes/`
- `codex_prompts/`

## Goal

Produce a conservative classification plan that distinguishes:

1. permanent residents
2. summarize, then move to attic
3. move to attic without summary
4. uncertain / needs human review

## Classification standard

A file should be treated as worth summarizing if it contains one or more of the following:

- a decision that still governs current behavior
- explanation of why the current design exists
- validation evidence for a still-live tool
- meaningful scope or priority decisions that still matter
- important failure findings that shaped the final design
- operational guidance that would still help a future maintainer

A file should be treated as likely scaffolding if it is mainly:

- a transient execution receipt
- an intermediate prompt-generation step
- a superseded planning or sequencing artifact
- a redundant success note from before the V1 system stabilized
- a local construction artifact whose main purpose was building the current toolset

## Required output artifacts

Create exactly two artifacts.

### Artifact 1

Create:

`notes/022_scaffolding_classification_report__TIMESTAMP.md`

This report should include:

- short summary
- classification criteria used
- permanent residents
- summarize-then-attic list
- attic-without-summary list
- uncertain list

For each listed file, include a short reason.

Be specific enough that a later prompt can act on this report without re-arguing every classification.

### Artifact 2

Create:

`notes/022_scaffolding_cleanup_plan__TIMESTAMP.md`

This note should contain only:

- what should happen in the next pass
- what should happen after that
- what should explicitly not happen yet

Keep this second note brief and operational.

## Constraints

1. Be conservative.
2. Prefer `uncertain` over aggressive classification when in doubt.
3. Do not move or delete anything in this pass.
4. Do not collapse the entire build history into one sentence.
5. Keep the plan practical enough to drive the next prompt.
6. Focus the classification on `notes/` and `codex_prompts/`, while using the tool and design docs only as context for deciding what is durable.

## Success criteria

This task is successful if:

- the current notes and prompts are classified into the four categories above
- the reasoning is concrete and repo-specific
- the result is safe enough to use as the basis for a summary-extraction pass
- no files are moved or deleted

 succeeded in 321ms:
# Project Design Workflow Revision

## What Changed

Revised `tools/Project_Design_Workflow.md` into a more project-agnostic workflow document.
Removed the project-specific application section and replaced it with reusable guidance.
Added explicit sections for bounded iteration, failure analysis, resource and cost awareness, and durable local history.

## Why These Changes Were Made

The earlier version had a strong phased structure, but part of it was tied to one specific workflow and repo context.
The revision keeps the design-first, boundary-aware approach while making the document reusable across different projects, tools, and maturity levels.

## New Reusable Principles Added

* bounded execution in small reviewable slices
* review between iterations before issuing the next step
* bridge tooling allowed when thin, inspectable, and subordinate
* durable local notes, logs, outputs, and reports as project memory
* failure triggering analysis rather than reflexive retry
* lightweight observation of cost, time, retries, and failure patterns

## Tradeoffs and Open Questions

The document is now more broadly reusable, but less concrete for any one project.
That tradeoff seems correct for a shared workflow template.
If later needed, a separate project-specific companion doc could show how this workflow maps onto a particular repo without narrowing the core template.

===== notes/003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md =====
# 003_revise_Project_Deisgn_workflow_document - SUCCESS

- Prompt file: `003_revise_Project_Deisgn_workflow_document.md`
- Timestamp (UTC): `20260415_194216`
- Status: `SUCCESS`

## Original Prompt

```md
# Task: Revise Project_Design_Workflow.md to remain project-agnostic and include failure-analysis discipline

You are working in this repository.

Your task is to revise the document `Project_Design_Workflow.md`.

## Goal

Produce an improved version of `Project_Design_Workflow.md` that remains:

- project-agnostic
- reusable across many repositories and workflows
- grounded in careful design-before-build discipline
- compatible with a prompt-by-prompt execution style

The revised document must NOT become narrowly tailored to the current Codex prompt workflow project.
It must stay general enough to guide many different projects.

## Important design intent

The workflow document should describe a reusable way to move from:

- idea
- architecture
- design clarification
- bounded implementation
- validation
- refinement

It should not depend on the current maturity level of any one project.

In other words:

- do not make the workflow care whether a project is early, midstream, or late
- do not make it depend on specific current tools being present
- do not overfit it to this repo

## New ideas that MUST be incorporated

Please revise the workflow so it clearly supports these general principles:

### 1. Bounded iterative execution
A project may move forward one carefully scoped chunk at a time.
The workflow should support iterative cycles of:

- clarify next chunk
- define bounded task
- execute
- inspect results
- refine plan
- repeat

This should be expressed in a project-agnostic way.

### 2. Bridge tooling is allowed
The workflow should acknowledge that a project may use temporary or bridge tooling during execution, as long as that tooling remains:

- thin
- inspectable
- subordinate to the design
- not a substitute for good architecture

Do not mention any specific tool names unless needed as a light example.
Keep the principle general.

### 3. Durable local history matters
The workflow should acknowledge that local artifacts, notes, logs, outputs, and reports can serve as durable project memory.
This should be expressed as a reusable principle, not tied to this specific project.

### 4. Review between iterations
The workflow should reinforce that each bounded execution step should be reviewed before the next one is issued.

### 5. Failure should trigger analysis
Add a reusable principle and likely a dedicated section explaining that after a failed execution step, the workflow should encourage a second-pass review or report that asks questions such as:

- Was the task too large?
- Was the task poorly decomposed?
- Were success criteria unclear?
- Was the failure caused by tooling/infrastructure rather than task difficulty?
- Would a smaller or differently framed prompt/task have worked better?

This failure-analysis concept should be included as part of the workflow, not as an afterthought.

### 6. Resource/cost awareness
Add a reusable principle that project minutes, notes, or reports may record metrics such as:

- token usage
- execution size
- elapsed time
- repeated retries
- failure frequency

The document should encourage observing whether larger or more expensive tasks correlate with failure or lower-quality results.

This should remain general and not assume a single specific model or platform.

## Constraints

1. Keep the tone consistent with the current document:
   - clear
   - calm
   - practical
   - thoughtful
   - not flashy

2. Preserve the strengths of the current document:
   - phased workflow
   - emphasis on boundaries
   - emphasis on validation
   - emphasis on refinement
   - reusable decomposition pattern

3. Do NOT turn this into a tool-specific runbook.
4. Do NOT overfit it to Codex.
5. Do NOT make it depend on the current state of this project.
6. Do NOT remove useful existing structure unless replacement is clearly better.

## Preferred output

Please do the following:

1. Read the existing `Project_Design_Workflow.md`
2. Revise it directly into a stronger replacement
3. Preserve the filename
4. Make the revised document feel like a better version of the current one, not a completely different manifesto

## Also create a short companion note

After revising the workflow doc, create a short markdown note in `notes/` with a timestamped filename that summarizes:

- what changed
- why those changes were made
- what new reusable principles were added
- any tradeoffs or open questions that remain

The note should be concise and practical.

## Success criteria

The task is successful if:

- `Project_Design_Workflow.md` remains reusable across many projects
- it now clearly supports bounded iterative execution
- it includes a clear failure-analysis concept
- it includes resource/cost-awareness as a reusable practice
- it still feels grounded, conservative, and design-centered
- it does not become narrowly tailored to this specific repo
```

## Codex Output

Revised [Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md) into a stronger project-agnostic version. The update keeps the phased design-before-build structure, but removes the repo-specific application section and adds explicit guidance for bounded iterative execution, review between iterations, bridge tooling, durable local history, failure analysis, and resource/cost awareness.

I also added a short companion note at [003_project_design_workflow_revision__SUCCESS__20260415_144244.md](/data/git/handson-ml3/notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md) summarizing what changed, why, the new reusable principles, and the remaining tradeoff that broader reuse means less project-specific concreteness.

No code tests were needed for this task.

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
session id: 019d92aa-7447-7302-b4d3-ee30d88317dd
--------
user
# Task: Revise Project_Design_Workflow.md to remain project-agnostic and include failure-analysis discipline

You are working in this repository.

Your task is to revise the document `Project_Design_Workflow.md`.

## Goal

Produce an improved version of `Project_Design_Workflow.md` that remains:

- project-agnostic
- reusable across many repositories and workflows
- grounded in careful design-before-build discipline
- compatible with a prompt-by-prompt execution style

The revised document must NOT become narrowly tailored to the current Codex prompt workflow project.
It must stay general enough to guide many different projects.

## Important design intent

The workflow document should describe a reusable way to move from:

- idea
- architecture
- design clarification
- bounded implementation
- validation
- refinement

It should not depend on the current maturity level of any one project.

In other words:

- do not make the workflow care whether a project is early, midstream, or late
- do not make it depend on specific current tools being present
- do not overfit it to this repo

## New ideas that MUST be incorporated

Please revise the workflow so it clearly supports these general principles:

### 1. Bounded iterative execution
A project may move forward one carefully scoped chunk at a time.
The workflow should support iterative cycles of:

- clarify next chunk
- define bounded task
- execute
- inspect results
- refine plan
- repeat

This should be expressed in a project-agnostic way.

### 2. Bridge tooling is allowed
The workflow should acknowledge that a project may use temporary or bridge tooling during execution, as long as that tooling remains:

- thin
- inspectable
- subordinate to the design
- not a substitute for good architecture

Do not mention any specific tool names unless needed as a light example.
Keep the principle general.

### 3. Durable local history matters
The workflow should acknowledge that local artifacts, notes, logs, outputs, and reports can serve as durable project memory.
This should be expressed as a reusable principle, not tied to this specific project.

### 4. Review between iterations
The workflow should reinforce that each bounded execution step should be reviewed before the next one is issued.

### 5. Failure should trigger analysis
Add a reusable principle and likely a dedicated section explaining that after a failed execution step, the workflow should encourage a second-pass review or report that asks questions such as:

- Was the task too large?
- Was the task poorly decomposed?
- Were success criteria unclear?

===== notes/004_architecture_and_bridge_runner_review__20260415_195538.md =====
# Architecture And Bridge Runner Review

## Summary

`tools/Codex_Prompt_Workflow_Architecture.md` is directionally aligned with the governing workflow because it pushes toward smaller parts, clearer ownership, and conservative growth. `tools/codex/baby_run_prompt.py` is acceptable as thin bridge tooling for a single bounded slice, but the current architecture doc does not yet translate the workflow into a clear V1 boundary, validation plan, review loop, or failure/resource evidence model. The next step should stay at the design layer: define the minimum execution-record artifact that every run must produce before more tooling is added.

## Architecture Alignment Findings

### Strong alignment

- The architecture doc is explicitly conservative, small-part oriented, and separation-of-concerns driven, which matches design-before-build and boundaries-before-breadth in the workflow ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:92), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:5)).
- Its decomposition of the original script into repo paths, prompt parsing, note handling, status reconstruction, retry handling, and CLI concerns is a good responsibility split and a useful architecture starting point ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:177), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:33)).
- It repeatedly says not to build everything at once and proposes a smaller folder shape before a larger one, which is consistent with thin slices before large pushes ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:100), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:121), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:246)).

### Stale, incomplete, or misaligned

- The architecture doc does not yet reflect the workflow's full sequence. It jumps from decomposition into a wish list and folder layout, but it does not define a short problem statement, explicit scope fence, out-of-scope list, minimum viable slice, artifact inventory, implementation order, or validation checklist as the governing workflow expects ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:141), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:159), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:196), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:214), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:233), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:251)).
- The "eventually" section is useful as an extension path, but it is carrying too much conceptual breadth for a workflow that emphasizes bounded next steps. There is no explicit line saying which of those items are intentionally deferred from V1 ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:96), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:374), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:119)).
- The document mentions logging and diagnostics, richer notes, retry intelligence, approval gates, and dependencies, but it does not specify the source of truth or the stable identity of important records beyond prompt stems. That leaves the durable-history model underdesigned relative to the workflow's emphasis on preserved local memory and stable identities ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:120), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:374), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:187)).
- Review points between iterations are largely absent. The architecture doc talks about manual status marking and approval gates eventually, but not about the concrete review stop that should happen after each bounded run before the next run proceeds ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:262), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:208)).

## Runner Alignment Findings

### Useful and appropriately thin

- The runner does one bounded thing: resolve a prompt, execute `codex exec`, and write a local note. That fits the workflow's allowance for thin bridge tooling that gathers evidence while remaining inspectable and reversible ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:114), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:99)).
- It validates basic directory prerequisites and fails cleanly when expected folders or prompt files are missing, which is a reasonable minimal safeguard for a bridge script ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:109), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:108)).
- It writes timestamped markdown notes into `notes/`, which gives the system at least some durable local history of what was run and what Codex returned ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:120), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:46), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:135)).

### Responsibilities it is carrying that should move elsewhere later

- Prompt resolution policy is embedded in the runner. Selection by absolute path, repo-relative path, `codex_prompts/` path, or globbed prefix is helpful now, but prompt identity and lookup rules belong in a prompt-discovery/status layer if the system grows ([tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:47), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:25)).
- Note-path and note-content policy are also embedded here. That is acceptable for a bridge script, but note schema, status vocabulary, and execution-record structure should eventually be owned by a note/history module rather than the process launcher ([tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:96), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:187), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:50)).
- The runner currently decides final state as `SUCCESS` or `FAILED` based only on the subprocess exit code. That is an expedient bridge behavior, but review judgment, validation outcome, and richer state modeling should not stay inside the launcher if the workflow matures ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:141), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:124)).

### Current misalignment

- As bridge tooling it is still acceptable, but it is only partially subordinate to workflow. It captures execution, yet it does not encode the review point that should separate execution from acceptance. A zero exit code becomes `SUCCESS` immediately, which skips the workflow's "validate and inspect before the next step" posture ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:271), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:125)).
- It stores the original prompt text and final output, but not explicit success criteria, validation evidence, or reviewer judgment. That means the note is an execution artifact, not yet a full workflow artifact ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:251), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:58)).

## Failure-Analysis Findings

- The current setup provides some raw failure evidence because nonzero exit runs still write a note and stderr is preserved when present ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:343), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:128)).
- That said, it does not support useful post-failure analysis very well. The note structure has no fields for failure classification, whether the failure was task/design/environment related, whether the task was too large, what changed on retry, or what the recommended next action is. The workflow explicitly wants failure to produce analysis rather than mere retry cost ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:125), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:343)).
- The smallest improvement here is not a larger retry engine. It is a defined execution-note schema with a short "failure analysis" block for failed runs, even if the user fills some of it manually. That preserves inspectable local history without expanding the runner's responsibilities much.

## Resource-Awareness Findings

- The setup preserves only minimal cost evidence today: timestamp in the note filename/content and the captured final output/stderr. That gives some weak signal about artifact volume, but almost nothing about runtime, retries, or service usage ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:382), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:17), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:58)).
- There is no recorded elapsed time, no attempt count or retry linkage, no output-size metric, and no token usage capture if Codex exposes it. The workflow asks for lightweight observation, not a billing system, and the current setup falls short of even that lightweight bar ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:130), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:382)).
- The smallest improvement is to define a tiny metrics block in the execution record. Start with fields the runner can always capture cheaply: start time, end time, elapsed seconds, subprocess return code, stderr present yes/no, and output character count. Token usage can remain optional when available later.

## Single Recommended Next Design Step

Create a short design artifact that defines the V1 execution record for one prompt run.

That artifact should do four things only:

1. Declare the source of truth for a run record.
2. Define the minimum fields for execution, review, failure analysis, and lightweight metrics.
3. Separate execution outcome from reviewed outcome.
4. State what bridge tooling may capture now versus what remains manual for this stage.

This is the best next step because both the architecture doc and the runner are currently weak at the same seam: they can run prompts and save notes, but they do not yet define what a durable, reviewable run artifact must contain. Defining that record is smaller and safer than refactoring modules now, and it directly supports failure analysis, resource awareness, and review between iterations.

===== notes/004_next_design_step_recommendation__20260415_195538.md =====
# Next Design Step Recommendation

Define the V1 execution-record artifact for a single prompt run.

This should happen next because the current system can execute prompts and write notes, but it does not yet define the minimum durable record needed for review, failure analysis, and lightweight cost observation. Without that decision, more implementation risks hard-coding the wrong note model and the wrong success semantics.

The artifact should be a short design note that specifies:

- the source of truth for one run record
- the minimum fields for execution facts, reviewed outcome, failure-analysis notes, and lightweight metrics
- which fields are captured automatically now
- which fields remain manual until later

What should wait until later:

- module refactors and file splits
- richer status/state machines
- retry intelligence
- dependency handling
- approval-gate automation
- reporting layers beyond the single-run record

===== notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md =====
# 004_review_architecture_and_bridge_runner_against_workflow - SUCCESS

- Prompt file: `004_review_architecture_and_bridge_runner_against_workflow.md`
- Timestamp (UTC): `20260415_195505`
- Status: `SUCCESS`

## Original Prompt

```md
# Task: Review the Codex prompt workflow architecture and the baby bridge runner against the current project design workflow

You are working in this repository.

Your task is to review two files against the current workflow guidance and produce a small design assessment.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/baby_run_prompt.py`

## Goal

Use `tools/Project_Design_Workflow.md` as the governing workflow document.

Then inspect:

- the architecture we say we want
- the bridge runner we actually have

Your job is to identify:

1. where the architecture and runner align with the workflow
2. where they do not align
3. what the smallest next design step should be before more implementation happens

## Important framing

This is a design-review task, not a large implementation task.

Do NOT do a broad rewrite of the system.
Do NOT build multiple new modules.
Do NOT refactor the runner in this pass.

This pass should stay focused on assessment, mismatch detection, and next-step recommendation.

## What to look for

Please evaluate the architecture doc and the bridge runner using the workflow principles in `tools/Project_Design_Workflow.md`, especially:

- design before build
- boundaries before breadth
- thin slices before large pushes
- review between iterations
- bridge tooling is allowed, but subordinate
- durable local history matters
- failure should produce analysis, not just retries
- resource use should be observed

## Questions to answer

Please answer these questions in the report:

### 1. Architecture alignment
- Does `tools/Codex_Prompt_Workflow_Architecture.md` reflect the workflow in `tools/Project_Design_Workflow.md`?
- Where is it strong?
- Where is it stale, incomplete, or misaligned?

### 2. Runner alignment
- Does `tools/codex/baby_run_prompt.py` behave like acceptable bridge tooling under the workflow?
- What parts of it are useful and appropriately thin?
- What responsibilities is it currently carrying that should eventually move elsewhere?

### 3. Failure-analysis support
- Does the current runner/workflow setup support useful post-failure analysis?
- If not, what is the smallest next improvement that would help?

### 4. Resource-awareness support
- Does the current setup preserve any useful execution-cost evidence?
- What is the smallest next improvement that would let us track lightweight metrics such as runtime, retries, token usage if available, or output size?

### 5. Smallest next design step
- What is the single best next design artifact or design decision to create before more implementation?
- Prefer one bounded next step, not a large roadmap.

## Required output artifacts

Please create exactly two artifacts.

### Artifact 1
Create a markdown report at:

`notes/004_architecture_and_bridge_runner_review__TIMESTAMP.md`

This report should include:

- a short summary
- architecture alignment findings
- runner alignment findings
- failure-analysis findings
- resource-awareness findings
- the single recommended next design step

Keep it practical and inspectable.

### Artifact 2
Create a short markdown file at:

`notes/004_next_design_step_recommendation__TIMESTAMP.md`

This should contain only:

- the recommended next step
- why it should happen next
- what artifact it should produce
- what should explicitly wait until later

Keep this one brief and decisive.

## Constraints

1. Use the exact file paths listed above.
2. Do not edit the workflow doc in this pass.
3. Do not perform a major refactor in this pass.
4. Do not produce a giant implementation plan.
5. Prefer specific observations over vague advice.
6. Keep the recommendation bounded enough that it can become the next Codex prompt.

## Success criteria

This task is successful if:

- the report clearly compares desired architecture vs current runner
- the assessment is grounded in `tools/Project_Design_Workflow.md`
- the next step is small, concrete, and design-focused
- the outputs help us decide the next prompt with confidence
```

## Codex Output

Created the two requested artifacts:

- [004_architecture_and_bridge_runner_review__20260415_195538.md](/data/git/handson-ml3/notes/004_architecture_and_bridge_runner_review__20260415_195538.md)
- [004_next_design_step_recommendation__20260415_195538.md](/data/git/handson-ml3/notes/004_next_design_step_recommendation__20260415_195538.md)

The assessment’s main conclusion is that the architecture doc is directionally right on decomposition and conservative growth, and the runner is thin enough to count as acceptable bridge tooling, but both are weak at the same seam: they do not yet define a durable, reviewable execution record that separates raw execution from reviewed outcome and captures minimal failure/resource evidence.

I did not modify the workflow doc or refactor the runner. The recommendation is intentionally bounded: define the V1 execution-record artifact next, before more implementation.

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
session id: 019d92b6-2caf-7452-b496-1a1a9adc8574
--------
user
# Task: Review the Codex prompt workflow architecture and the baby bridge runner against the current project design workflow

You are working in this repository.

Your task is to review two files against the current workflow guidance and produce a small design assessment.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/baby_run_prompt.py`

## Goal

Use `tools/Project_Design_Workflow.md` as the governing workflow document.

Then inspect:

- the architecture we say we want
- the bridge runner we actually have

Your job is to identify:

1. where the architecture and runner align with the workflow
2. where they do not align
3. what the smallest next design step should be before more implementation happens

## Important framing

This is a design-review task, not a large implementation task.

Do NOT do a broad rewrite of the system.
Do NOT build multiple new modules.
Do NOT refactor the runner in this pass.

This pass should stay focused on assessment, mismatch detection, and next-step recommendation.

## What to look for

Please evaluate the architecture doc and the bridge runner using the workflow principles in `tools/Project_Design_Workflow.md`, especially:

- design before build
- boundaries before breadth
- thin slices before large pushes
- review between iterations
- bridge tooling is allowed, but subordinate
- durable local history matters
- failure should produce analysis, not just retries
- resource use should be observed

## Questions to answer

Please answer these questions in the report:

### 1. Architecture alignment
- Does `tools/Codex_Prompt_Workflow_Architecture.md` reflect the workflow in `tools/Project_Design_Workflow.md`?
- Where is it strong?
- Where is it stale, incomplete, or misaligned?

### 2. Runner alignment
- Does `tools/codex/baby_run_prompt.py` behave like acceptable bridge tooling under the workflow?
- What parts of it are useful and appropriately thin?
- What responsibilities is it currently carrying that should eventually move elsewhere?

### 3. Failure-analysis support
- Does the current runner/workflow setup support useful post-failure analysis?
- If not, what is the smallest next improvement that would help?

### 4. Resource-awareness support
- Does the current setup preserve any useful execution-cost evidence?
- What is the smallest next improvement that would let us track lightweight metrics such as runtime, retries, token usage if available, or output size?

### 5. Smallest next design step
- What is the single best next design artifact or design decision to create before more implementation?
- Prefer one bounded next step, not a large roadmap.

## Required output artifacts

Please create exactly two artifacts.

### Artifact 1
Create a markdown report at:

`notes/004_architecture_and_bridge_runner_review__TIMESTAMP.md`

This report should include:

- a short summary
- architecture alignment findings
- runner alignment findings
- failure-analysis findings

===== notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md =====
# 005_define_execution_record_and_generate_next_prompt_queue - SUCCESS

- Prompt file: `005_define_execution_record_and_generate_next_prompt_queue.md`
- Timestamp (UTC): `20260415_202522`
- Status: `SUCCESS`

## Original Prompt

```md
# Task: Define the V1 execution-record artifact and generate the next bounded prompt queue

You are working in this repository.

Your task has two tightly related parts:

1. define the V1 execution-record artifact for one prompt run
2. use that design decision to generate the next bounded sequence of prompt files needed to reduce the current workflow/architecture/runner misalignment

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/baby_run_prompt.py`
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
- `notes/004_next_design_step_recommendation__20260415_195538.md`

## Goal

Use the current workflow as the governing law.

Then:

### Part 1
Define the V1 execution-record artifact for one prompt run.

This artifact should clarify:

- what the source of truth is for one run
- what is captured automatically
- what remains manual for human review
- how execution outcome is separated from accepted/reviewed outcome
- what minimal failure-analysis fields exist
- what minimal cost/resource fields exist
- what should explicitly wait until later

### Part 2
Using that execution-record decision, generate the next small sequence of prompt files needed to reduce the remaining misalignment between:

- the workflow we now want
- the architecture we say we want
- the bridge runner we currently have

## Important framing

This is still a design task, not a broad implementation task.

Do NOT refactor the runner in this pass.
Do NOT implement multiple modules in this pass.
Do NOT produce a giant roadmap.

Instead, define the execution-record artifact and then produce the smallest useful queue of next prompts.

## Requirements for Part 1

Create one markdown design artifact at:

`tools/codex/V1_Execution_Record_Artifact.md`

This document should include:

- purpose
- scope
- source of truth
- stable identity for a run
- required fields
- optional fields
- automatic vs manual fields
- execution status vs review status
- minimum failure-analysis section
- minimum resource/cost section
- what is intentionally deferred from V1

Keep it practical, small, and reusable.

## Requirements for Part 2

Create a short sequence of prompt files in `codex_prompts/`.

Create between **3 and 5** prompt files total.
Do not create more than 5.

Each prompt must:

- have one primary goal
- produce one primary artifact or decision
- be small enough to review before the next prompt is run
- follow from the execution-record design
- reduce a specific misalignment already identified in the review notes

Each prompt file should have:

- a clear filename with numeric prefix
- a focused task
- exact file paths where relevant
- explicit constraints
- explicit success criteria

## Also create one companion sequence note

Create a short note at:

`notes/005_prompt_queue_plan__TIMESTAMP.md`

This note should summarize:

- why these prompts were chosen
- why this order reduces risk
- what each prompt is meant to settle
- what larger work is intentionally deferred

## Constraints

1. Use the exact file paths listed above.
2. Treat `tools/Project_Design_Workflow.md` as governing.
3. Keep the execution-record artifact conservative and V1-sized.
4. Do not let the prompt queue become a giant backlog.
5. Prefer the smallest sequence that meaningfully reduces misalignment.
6. Do not implement the later prompts in this pass. Only write them.
7. Do not rewrite the workflow doc in this pass.

## Success criteria

This task is successful if:

- `tools/codex/V1_Execution_Record_Artifact.md` is clear and usable
- the execution-record design separates execution from review/acceptance
- the prompt queue contains only 3 to 5 bounded next prompts
- the prompts are ordered sensibly
- the outputs help us continue one reviewed step at a time
```

## Codex Output

Created the V1 execution-record design at [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:1). It makes one `notes/` markdown record the V1 source of truth for a run, gives each run a stable `run_id`, separates `execution_status` from `review_status`, defines the minimum manual failure-analysis and lightweight resource fields, and explicitly defers broader state machines, reporting, and refactors.

I also added the next bounded prompt queue:
[006_define_v1_run_review_gate.md](/data/git/handson-ml3/codex_prompts/006_define_v1_run_review_gate.md:1),
[007_align_architecture_doc_to_v1_workflow.md](/data/git/handson-ml3/codex_prompts/007_align_architecture_doc_to_v1_workflow.md:1),
and [008_define_minimal_bridge_runner_change_spec.md](/data/git/handson-ml3/codex_prompts/008_define_minimal_bridge_runner_change_spec.md:1).
The companion sequencing note is [005_prompt_queue_plan__20260415_202557.md](/data/git/handson-ml3/notes/005_prompt_queue_plan__20260415_202557.md:1).

Verification was by direct file inspection; no implementation or runner refactor was performed.

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
session id: 019d92d1-ea45-71c3-8fbd-c83284522cdc
--------
user
# Task: Define the V1 execution-record artifact and generate the next bounded prompt queue

You are working in this repository.

Your task has two tightly related parts:

1. define the V1 execution-record artifact for one prompt run
2. use that design decision to generate the next bounded sequence of prompt files needed to reduce the current workflow/architecture/runner misalignment

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/baby_run_prompt.py`
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
- `notes/004_next_design_step_recommendation__20260415_195538.md`

## Goal

Use the current workflow as the governing law.

Then:

### Part 1
Define the V1 execution-record artifact for one prompt run.

This artifact should clarify:

- what the source of truth is for one run
- what is captured automatically
- what remains manual for human review
- how execution outcome is separated from accepted/reviewed outcome
- what minimal failure-analysis fields exist
- what minimal cost/resource fields exist
- what should explicitly wait until later

### Part 2
Using that execution-record decision, generate the next small sequence of prompt files needed to reduce the remaining misalignment between:

- the workflow we now want
- the architecture we say we want
- the bridge runner we currently have

## Important framing

This is still a design task, not a broad implementation task.

Do NOT refactor the runner in this pass.
Do NOT implement multiple modules in this pass.
Do NOT produce a giant roadmap.

Instead, define the execution-record artifact and then produce the smallest useful queue of next prompts.

## Requirements for Part 1

Create one markdown design artifact at:

`tools/codex/V1_Execution_Record_Artifact.md`

This document should include:

- purpose
- scope
- source of truth
- stable identity for a run
- required fields
- optional fields
- automatic vs manual fields
- execution status vs review status
- minimum failure-analysis section
- minimum resource/cost section
- what is intentionally deferred from V1

Keep it practical, small, and reusable.

## Requirements for Part 2

Create a short sequence of prompt files in `codex_prompts/`.

Create between **3 and 5** prompt files total.
Do not create more than 5.

Each prompt must:

- have one primary goal
- produce one primary artifact or decision
- be small enough to review before the next prompt is run

===== notes/005_prompt_queue_plan__20260415_202557.md =====
# Prompt Queue Plan

## Why These Prompts Were Chosen

These prompts were chosen because the current misalignment is concentrated at one seam:
the system can execute prompts, but it does not yet define the review checkpoint, the V1 architectural boundary around that checkpoint, or the smallest runner behavior needed to support it.

The queue stays small on purpose.
It settles the workflow gap before any implementation prompt is allowed to widen scope.

## Why This Order Reduces Risk

The order moves from policy to architecture to bridge-tooling:

1. define the review gate
2. align the architecture doc to that gate and the execution record
3. define the smallest runner change spec that fits those decisions

This reduces the risk of hard-coding the wrong note model or the wrong queue semantics into the runner before the manual review loop is settled.

## What Each Prompt Is Meant To Settle

`006_define_v1_run_review_gate.md`

- settles the missing review stop between execution and accepted progress

`007_align_architecture_doc_to_v1_workflow.md`

- settles the V1 boundary, artifact inventory, validation posture, and deferrals inside the architecture document

`008_define_minimal_bridge_runner_change_spec.md`

- settles the smallest bridge-runner behavior needed to emit the V1 execution record without a broader refactor

## What Is Intentionally Deferred

The following work is intentionally not in this queue:

- runner refactor into multiple modules
- richer state machines
- retry intelligence
- dependency-aware scheduling
- aggregated reporting
- token accounting beyond optional fields
- broad CLI redesign

===== notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md =====
# 006_define_v1_run_review_gate - SUCCESS

- Prompt file: `006_define_v1_run_review_gate.md`
- Timestamp (UTC): `20260415_203019`
- Status: `SUCCESS`

## Original Prompt

```md
# Task: Define the V1 run-review gate for prompt execution

You are working in this repository.

Your task is to define the minimum review gate that must sit between:

- a prompt run finishing
- the next prompt in sequence being treated as ready

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/baby_run_prompt.py`
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
- `notes/005_prompt_queue_plan__20260415_202557.md`

## Goal

Use `tools/Project_Design_Workflow.md` as governing.

Define the smallest practical V1 review gate that preserves:

- review between iterations
- separation of execution outcome from accepted outcome
- a conservative manual step before the next prompt is treated as ready

## Important framing

This is a design task.

Do NOT implement the gate in code in this pass.
Do NOT create a large state machine.
Do NOT redesign the whole queue system.

## Questions to settle

Please settle these points:

1. What exact event marks a run as awaiting review?
2. What minimum human checks must happen before a run can be accepted?
3. What review outcomes are needed in V1?
4. What outcome should let the next prompt proceed?
5. What outcome should stop the queue and force a new design or retry decision?
6. What information must be written back into the run record during review?

## Required output artifact

Create one markdown design note at:

`tools/codex/V1_Run_Review_Gate.md`

The note should include:

- purpose
- scope
- the review trigger
- the minimum manual checklist
- allowed V1 review outcomes
- how the outcome affects queue progression
- how this connects to `tools/codex/V1_Execution_Record_Artifact.md`
- what is intentionally deferred

## Constraints

1. Use the exact file paths listed above.
2. Keep the gate manual and conservative.
3. Do not implement CLI or runner changes in this pass.
4. Do not invent a large workflow beyond the next reviewed step.
5. Keep the decision small enough to review before any implementation prompt follows.

## Success criteria

This task is successful if:

- the review gate is explicit and easy to apply
- execution success is clearly separated from accepted outcome
- queue progression rules are clear for V1
- the result is small enough to guide the next prompt without expanding scope
```

## Codex Output

Created [V1_Run_Review_Gate.md](/data/git/handson-ml3/tools/codex/V1_Run_Review_Gate.md).

It defines the minimal V1 gate as: execution finishes, a run record is written with `review_status: UNREVIEWED`, a human performs a short manual checklist, and only `ACCEPTED` allows the next prompt to be treated as ready. `REJECTED` is the stop condition that forces an explicit redesign or retry decision, and the note ties that directly back to the fields in [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md).

No code or CLI changes were made.

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
session id: 019d92d6-6e64-7182-9253-261cc553a5fb
--------
user
# Task: Define the V1 run-review gate for prompt execution

You are working in this repository.

Your task is to define the minimum review gate that must sit between:

- a prompt run finishing
- the next prompt in sequence being treated as ready

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/baby_run_prompt.py`
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
- `notes/005_prompt_queue_plan__20260415_202557.md`

## Goal

Use `tools/Project_Design_Workflow.md` as governing.

Define the smallest practical V1 review gate that preserves:

- review between iterations
- separation of execution outcome from accepted outcome
- a conservative manual step before the next prompt is treated as ready

## Important framing

This is a design task.

Do NOT implement the gate in code in this pass.
Do NOT create a large state machine.
Do NOT redesign the whole queue system.

## Questions to settle

Please settle these points:

1. What exact event marks a run as awaiting review?
2. What minimum human checks must happen before a run can be accepted?
3. What review outcomes are needed in V1?
4. What outcome should let the next prompt proceed?
5. What outcome should stop the queue and force a new design or retry decision?
6. What information must be written back into the run record during review?

## Required output artifact

Create one markdown design note at:

`tools/codex/V1_Run_Review_Gate.md`

The note should include:

- purpose
- scope
- the review trigger
- the minimum manual checklist
- allowed V1 review outcomes
- how the outcome affects queue progression
- how this connects to `tools/codex/V1_Execution_Record_Artifact.md`
- what is intentionally deferred

## Constraints

1. Use the exact file paths listed above.
2. Keep the gate manual and conservative.
3. Do not implement CLI or runner changes in this pass.
4. Do not invent a large workflow beyond the next reviewed step.
5. Keep the decision small enough to review before any implementation prompt follows.

## Success criteria

This task is successful if:

- the review gate is explicit and easy to apply
- execution success is clearly separated from accepted outcome
- queue progression rules are clear for V1
- the result is small enough to guide the next prompt without expanding scope

codex
I’m reading the specified workflow and architecture files first, then I’ll draft a minimal V1 review-gate note that stays within the current design boundaries.
exec
/bin/bash -lc "sed -n '1,240p' tools/Project_Design_Workflow.md" in /data/git/handson-ml3
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


===== notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md =====
# 007_align_architecture_doc_to_v1_workflow - SUCCESS

- Prompt file: `007_align_architecture_doc_to_v1_workflow.md`
- Timestamp (UTC): `20260415_203257`
- Status: `SUCCESS`

## Original Prompt

```md
# Task: Align the Codex prompt architecture document to the governing workflow and V1 run artifact

You are working in this repository.

Your task is to bring the architecture document into explicit V1 alignment with:

- the governing workflow
- the V1 execution record
- the V1 review gate

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
- `notes/005_prompt_queue_plan__20260415_202557.md`

## Goal

Update the architecture document so it no longer skips the workflow steps that matter for V1.

The revised architecture should make clear:

- the V1 problem being solved
- the V1 scope boundary
- the minimum artifact inventory
- the role of the execution record
- the role of the review gate
- the conservative implementation order
- what remains deferred

## Important framing

This is still a design-alignment task.

Do NOT implement the architecture in code in this pass.
Do NOT expand the document into a giant platform roadmap.
Do NOT rewrite the governing workflow document.

## Required output artifact

Revise this file directly:

`tools/Codex_Prompt_Workflow_Architecture.md`

The revision should add or clarify:

- short problem statement
- V1 scope
- out-of-scope items
- minimum artifact inventory
- minimum viable slice
- implementation order
- validation and review posture
- explicit deferred items

## Constraints

1. Use the exact file paths listed above.
2. Treat `tools/Project_Design_Workflow.md` as governing.
3. Preserve the conservative tone of the architecture doc.
4. Keep the document reusable and inspectable.
5. Do not add speculative subsystems that are not needed for V1.

## Success criteria

This task is successful if:

- the architecture doc clearly reflects the governing workflow sequence
- the V1 execution record and review gate are integrated into the architecture
- V1 boundaries and deferrals are explicit
- the result reduces the mismatch identified in the review note without becoming a large rewrite
```

## Codex Output

Revised [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md) to align it explicitly with the governing workflow and the V1 artifacts.

The document now states the V1 problem, scope boundary, out-of-scope items, minimum artifact inventory, minimum viable slice, implementation order, validation/review posture, and explicit deferrals. It also makes the roles of [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md) and [V1_Run_Review_Gate.md](/data/git/handson-ml3/tools/codex/V1_Run_Review_Gate.md) architectural rather than optional future ideas, and it keeps the conservative tone without expanding into a broader platform roadmap.

No code was changed and no tests were run, since this pass was a document alignment edit only.

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
session id: 019d92d8-d961-7322-b900-1014ababe6e7
--------
user
# Task: Align the Codex prompt architecture document to the governing workflow and V1 run artifact

You are working in this repository.

Your task is to bring the architecture document into explicit V1 alignment with:

- the governing workflow
- the V1 execution record
- the V1 review gate

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
- `notes/005_prompt_queue_plan__20260415_202557.md`

## Goal

Update the architecture document so it no longer skips the workflow steps that matter for V1.

The revised architecture should make clear:

- the V1 problem being solved
- the V1 scope boundary
- the minimum artifact inventory
- the role of the execution record
- the role of the review gate
- the conservative implementation order
- what remains deferred

## Important framing

This is still a design-alignment task.

Do NOT implement the architecture in code in this pass.
Do NOT expand the document into a giant platform roadmap.
Do NOT rewrite the governing workflow document.

## Required output artifact

Revise this file directly:

`tools/Codex_Prompt_Workflow_Architecture.md`

The revision should add or clarify:

- short problem statement
- V1 scope
- out-of-scope items
- minimum artifact inventory
- minimum viable slice
- implementation order
- validation and review posture
- explicit deferred items

## Constraints

1. Use the exact file paths listed above.
2. Treat `tools/Project_Design_Workflow.md` as governing.
3. Preserve the conservative tone of the architecture doc.
4. Keep the document reusable and inspectable.
5. Do not add speculative subsystems that are not needed for V1.

## Success criteria

This task is successful if:

- the architecture doc clearly reflects the governing workflow sequence
- the V1 execution record and review gate are integrated into the architecture
- V1 boundaries and deferrals are explicit
- the result reduces the mismatch identified in the review note without becoming a large rewrite

codex
I’m reviewing the governing workflow, the current architecture doc, and the V1 artifacts first so I can make a targeted alignment edit rather than broaden the design.
exec
/bin/bash -lc "sed -n '1,220p' tools/Project_Design_Workflow.md" in /data/git/handson-ml3
exec
/bin/bash -lc "sed -n '1,260p' tools/Codex_Prompt_Workflow_Architecture.md" in /data/git/handson-ml3
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

===== notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md =====
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

## Success criteria

This task is successful if:

- the runner change scope is small and explicit
- the spec is directly grounded in the V1 execution record and review gate
- execution and review are no longer conflated in the target behavior
- the spec avoids a premature refactor
```

## Codex Output

Created [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md).

The spec keeps the change set narrow: the runner keeps its current prompt-resolution and `codex exec` launch behavior, but switches from `SUCCESS`/`FAILED` notes to a stable `notes/<prompt_stem>__<started_at_utc>.md` V1 execution record. It specifies the exact markdown sections, sets `execution_status` from the subprocess return code (`0 -> EXECUTED`, nonzero -> `EXECUTION_FAILED`), captures only the minimum automatic fields and metrics, and initializes review/failure-analysis fields for later manual completion with `review_status: UNREVIEWED`.

One implementation question is left explicit: whether to keep second-precision timestamps as-is and fail on same-second collisions, or add a minimal collision suffix. Everything else is specified tightly enough for a single later implementation prompt.

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
session id: 019d92db-7473-7ce0-8808-40bf265b5df0
--------
user
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

## Success criteria

This task is successful if:

- the runner change scope is small and explicit
- the spec is directly grounded in the V1 execution record and review gate
- execution and review are no longer conflated in the target behavior
- the spec avoids a premature refactor

codex
I’m reading the architecture, workflow, V1 record docs, and the current runner to extract the smallest viable change surface before drafting the spec file.
  → Read the specified workflow, architecture, V1 record, review gate, runner, and note files
  • Derive the minimal runner change spec grounded in current behavior and V1 requirements
  • Write tools/codex/V1_Bridge_Runner_Change_Spec.md
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

===== notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md =====
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

===== notes/009_run_prompt_candidate_build__20260415_233407.md =====
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

===== notes/010_next_step_recommendation__20260415_234559.md =====
# 010 Next Step Recommendation

Chosen next move: `environment diagnosis prompt`

Why it should happen next:
The V1 runner candidate already produces the intended execution record shape closely enough, and the validation run failed during Codex session initialization with a read-only filesystem error. That makes environment diagnosis the smallest step that can unblock meaningful validation without broadening the build.

What it should produce:
A short diagnosis artifact that identifies where session startup is trying to write, whether the failure reproduces independently of `tools/codex/run_prompt.py`, and one bounded follow-up action.

What should explicitly wait:
Runner polish, review write-back helpers, and any broader workflow or runner build-out.

===== notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md =====
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

Explain why that is the smallest and safest next move.

## Required output artifacts

Create exactly two artifacts.

### Artifact 1
Create a review note at:

`notes/010_run_prompt_candidate_review__TIMESTAMP.md`

This note should include:

- short summary
- runner assessment
- artifact assessment
- operational assessment
- exactly one recommended next move

### Artifact 2
Create a short recommendation note at:

`notes/010_next_step_recommendation__TIMESTAMP.md`

This should contain only:

- the chosen next move
- why it should happen next
- what it should produce
- what should explicitly wait

Keep it brief and decisive.

## Constraints

1. Use the exact file paths listed above.
2. Recommend only one next step.
3. Do not produce a broad roadmap.
4. Do not modify code or design documents in this pass.
5. Keep the recommendation small enough to become the next Codex prompt.

## Success criteria

This task is successful if:

- the review clearly assesses both `run_prompt.py` and the sample execution record
- the next move is grounded in what actually happened
- only one next step is recommended
- the result helps us continue in a slow, deliberate, low-risk way
```

## Codex Output

Created the two required artifacts:

- [010_run_prompt_candidate_review__20260415_234559.md](/data/git/handson-ml3/notes/010_run_prompt_candidate_review__20260415_234559.md)
- [010_next_step_recommendation__20260415_234559.md](/data/git/handson-ml3/notes/010_next_step_recommendation__20260415_234559.md)

The recommendation is `environment diagnosis prompt`. The review concludes `tools/codex/run_prompt.py` is close enough to the V1 intent, the sample execution record has the right separation between execution and review, and the read-only failure appears to be primarily a Codex session/environment problem rather than a runner defect.

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
session id: 019d9389-06fa-7f91-9395-77ccc8c3598c
--------
user
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

Explain why that is the smallest and safest next move.

## Required output artifacts

Create exactly two artifacts.

### Artifact 1
Create a review note at:

`notes/010_run_prompt_candidate_review__TIMESTAMP.md`

This note should include:

- short summary
- runner assessment
- artifact assessment
- operational assessment
- exactly one recommended next move

### Artifact 2
Create a short recommendation note at:

`notes/010_next_step_recommendation__TIMESTAMP.md`

This should contain only:

- the chosen next move
- why it should happen next
- what it should produce
- what should explicitly wait

Keep it brief and decisive.


===== notes/010_run_prompt_candidate_review__20260415_234559.md =====
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

===== notes/011_build_v1_review_writeback_helper__20260415_235346.md =====
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

===== notes/011_review_writeback_helper_build__20260415_235514.md =====
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

===== notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md =====
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
- `notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md`
- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`

## Goal

Review the full current V1 workflow pipe and answer:

1. what pieces now exist and are functioning
2. what important gaps or awkward seams remain
3. what realistic next bounded options are available
4. which three options are the strongest next candidates

## What to evaluate

Please evaluate the current state of the pipe across these areas:

### 1. Execution path
- prompt resolution
- Codex invocation
- execution-record creation
- stability of record identity
- resource/failure evidence capture

### 2. Review path
- manual review write-back
- preservation of record structure
- consistency with the V1 review gate
- whether review is now operationally usable

### 3. Workflow usability
- whether the V1 flow is now practical for repeated use
- what still feels awkward, manual, fragile, or under-supported
- what is missing if this is going to be used regularly

### 4. Documentation alignment
- whether the code and notes still match the design packet
- whether there is any doc drift or naming drift that should be addressed
- whether there are any small correctness mismatches between design and implementation

## Required outputs

Create exactly two artifacts.

### Artifact 1
Create a review report at:

`notes/012_v1_pipeline_options_review__TIMESTAMP.md`

This report should include:

- short summary of current pipeline maturity
- what is working now
- what seams or gaps remain
- a list of realistic next bounded options

For the options list:
- provide at least 5 options if there are 5 credible ones
- provide fewer only if the repo truly supports fewer
- each option should include:
  - short name
  - what it would build or improve
  - why it matters
  - expected risk level: low / medium / high
  - expected payoff level: low / medium / high

### Artifact 2
Create a short recommendation note at:

`notes/012_top_three_next_options__TIMESTAMP.md`

This note should contain:

- the top three next options in ranked order
- why each made the top three
- which one should happen next
- what should explicitly wait

## Ranking guidance

When ranking options, prefer:
- bounded steps
- low-risk progress
- improvements that strengthen repeated operational use
- improvements that reduce ambiguity or manual fragility
- improvements supported by the current repo state

Avoid preferring:
- broad platform expansion
- premature refactors
- speculative subsystems
- large workflow engines

## Constraints

1. Use the exact file paths listed above.
2. Do not implement anything in this pass.
3. Do not produce a giant roadmap.
4. Recommend only bounded next options that are plausible from the current state.
5. Keep the tone practical and inspectable.
6. Let the repo evidence drive the options.

## Success criteria

This task is successful if:

- the review clearly describes what the V1 pipe can do now
- the options are grounded in actual repo artifacts and recent notes
- the report gives a realistic menu of next moves
- the recommendation note ranks the top three clearly
- the result helps a human choose the next deliberate step with confidence
```

## Codex Final Output

Created the two requested artifacts:

- [012_v1_pipeline_options_review__20260416_000819.md](/data/git/handson-ml3/notes/012_v1_pipeline_options_review__20260416_000819.md)
- [012_top_three_next_options__20260416_000819.md](/data/git/handson-ml3/notes/012_top_three_next_options__20260416_000819.md)

The review concludes the V1 pipe is now a real usable thin slice: `run_prompt.py` creates durable execution records, `review_run.py` performs in-place manual review write-back, and the main remaining weakness is operational support for repeated use rather than missing architecture. The top recommendation is to add a small queue-readiness checker next, followed by an unreviewed-run lister and then lightweight contract validation for the two V1 scripts.

I did not modify `tools/codex/baby_run_prompt.py`, `tools/codex/run_prompt.py`, or `tools/codex/review_run.py`.

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
session id: 019d939c-cbad-7e11-9d95-7794a03d6150
--------
user
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

===== notes/012_top_three_next_options__20260416_000819.md =====
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

===== notes/012_v1_pipeline_options_review__20260416_000819.md =====
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

===== notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md =====
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
```

## Codex Final Output

Created exactly two prompt files and one planning note:

- [014_build_queue_readiness_checker.md](/data/git/handson-ml3/codex_prompts/014_build_queue_readiness_checker.md)
- [015_build_review_backlog_unreviewed_run_lister.md](/data/git/handson-ml3/codex_prompts/015_build_review_backlog_unreviewed_run_lister.md)
- [013_next_two_prompt_plan__20260416_002005.md](/data/git/handson-ml3/notes/013_next_two_prompt_plan__20260416_002005.md)

The prompts are bounded, implementation-ready, and explicitly constrained against broader queue-engine or dashboard growth. The plan note explains why these two were chosen, why readiness checking comes first, and what should wait. I did not implement either helper and did not modify the protected runner/review files.

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
session id: 019d93a8-5d4b-76a1-92c4-602e989e9ef9
--------
user
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

===== notes/013_next_two_prompt_plan__20260416_002005.md =====
# 013 Next Two Prompt Plan

The next two prompts were chosen because the recent options review identified the main remaining V1 friction as operational support around repeated reviewed use, not missing architecture. The strongest bounded additions are the queue-readiness checker and the review backlog / unreviewed-run lister because both rely on the now-real execution-record and review-writeback artifacts without expanding the workflow into a larger engine.

They are ordered this way because readiness checking closes the most important architectural seam first: the V1 rule that only an accepted reviewed run should release the next prompt. The review backlog lister should follow immediately after because it improves day-to-day review discovery, but it is slightly less central than the readiness decision itself.

The first prompt to execute should be `codex_prompts/014_build_queue_readiness_checker.md`.

What should wait until later:

- lightweight contract validation for `tools/codex/run_prompt.py` and `tools/codex/review_run.py`
- doc-alignment cleanup for remaining naming and run-id drift
- rejected-run / retry-linkage support beyond the current placeholder fields
- any broader dashboard, queue engine, dependency system, or workflow automation layer

===== notes/014_build_queue_readiness_checker__20260416_002319.md =====
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
```

## Codex Final Output

Implemented the read-only V1 checker at [check_queue_readiness.py](/data/git/handson-ml3/tools/codex/check_queue_readiness.py) and documented it in [014_queue_readiness_checker_build__20260416_002419.md](/data/git/handson-ml3/notes/014_queue_readiness_checker_build__20260416_002419.md).

The helper:
- discovers prompt order from numeric prefixes in `codex_prompts/`
- parses V1 execution-record metadata from `notes/`
- finds the latest record for the immediately previous prompt
- applies the current gate conservatively: only prior `ACCEPTED` releases the target prompt

===== notes/014_queue_readiness_checker_build__20260416_002419.md =====
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

===== notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md =====
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


===== notes/015_review_backlog_lister_build__20260416_010500.md =====
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

===== notes/016_next_improvement_recommendation__20260416_003710.md =====
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

===== notes/016_queue_and_backlog_helper_validation__20260416_003710.md =====
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

===== notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md =====
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


===== notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md =====
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

===== notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md =====
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

===== notes/018_architecture_vs_actual_sweep__20260416_005130.md =====
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

===== notes/018_prioritized_remaining_work__20260416_005130.md =====
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

===== notes/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work__20260416_005034.md =====
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


===== notes/019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209.md =====
# 019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209

- run_id: `019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209`
- prompt_file: `codex_prompts/019_generate_prompts_for_doc_alignment_and_record_contract_validation.md`
- prompt_stem: `019_generate_prompts_for_doc_alignment_and_record_contract_validation`
- started_at_utc: `20260416_010209`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260416_010353`
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

- elapsed_seconds: `103.841`
- final_output_char_count: `775`
- stderr_char_count: `1010614`

## Prompt Text

```md
# Task: Generate the next two implementation prompts for the current V1 workflow cleanup

You are working in this repository.

Your task is to write the next two bounded implementation prompt files based on the current architecture sweep and prioritized remaining work.

## Important framing

This is a prompt-generation task only.

Do NOT implement code or doc edits in this pass.
Do NOT modify existing tools or design documents in this pass.

Your job is to create the next two implementation prompts as files in `codex_prompts/`.

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
- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
- `notes/018_prioritized_remaining_work__20260416_005130.md`

## Goal

Generate exactly two prompt files:

1. a prompt to perform the recommended **doc/spec alignment cleanup**
2. a prompt to implement the recommended **lightweight record-contract validation**

These prompts should be implementation-ready, bounded, and consistent with the current repo state.

## Prompt 1 requirements

Create a prompt file for doc/spec alignment cleanup.

This future task should focus on aligning the design packet with the actual V1 toolset and current behavior, including at least:

- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- any stale references to `baby_run_prompt.py` where the actual V1 runner is now `run_prompt.py`
- any stale implication that readiness/backlog helpers do not yet exist
- the current run-id collision suffix behavior where relevant

The prompt should keep the cleanup bounded and should not turn into a broad architecture rewrite.

## Prompt 2 requirements

Create a prompt file for lightweight record-contract validation.

This future task should focus on creating a small repeatable validation layer for the shared V1 markdown execution-record shape used by:

- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

The prompt should keep the validation lightweight and inspectable.

Avoid proposing a large test framework, platform service, or dashboard.

## What each generated prompt must include

Each prompt file must:

- have a clear numeric filename prefix
- define one primary goal only
- name exact files to inspect
- define exact output artifacts to create
- include constraints that prevent broad platform growth
- define validation requirements
- define success criteria
- stay small enough to review comfortably before execution

## Also create one planning note

Create a short note at:

`notes/019_next_two_cleanup_prompts_plan__TIMESTAMP.md`

This note should summarize:

- why these two prompts were chosen
- why they are ordered this way
- which one should be executed first
- what should explicitly wait until later

## Constraints

1. Generate exactly two prompt files, no more.
2. Do not implement the cleanup tasks in this pass.
3. Keep both generated prompts bounded and V1-sized.
4. Do not introduce a broader orchestration layer, queue engine, retry manager, or platform expansion.
5. Let the architecture sweep drive the prompts.

## Success criteria

This task is successful if:

- exactly two prompt files are created
- one is for doc/spec alignment cleanup
- one is for lightweight record-contract validation
- both prompts are implementation-ready and bounded
- the planning note clearly explains the sequence
```

## Codex Final Output

Created the two prompt files and the planning note:

- [020_align_v1_doc_and_spec_packet_to_actual_toolset.md](/data/git/handson-ml3/codex_prompts/020_align_v1_doc_and_spec_packet_to_actual_toolset.md)
- [021_add_lightweight_v1_record_contract_validation.md](/data/git/handson-ml3/codex_prompts/021_add_lightweight_v1_record_contract_validation.md)
- [019_next_two_cleanup_prompts_plan__20260416_010251.md](/data/git/handson-ml3/notes/019_next_two_cleanup_prompts_plan__20260416_010251.md)

The sequence is explicit in the note: execute `020` first for doc/spec alignment, then `021` for lightweight record-contract validation. I did not implement any cleanup or tool changes in this pass, and the generated prompts stay bounded to the V1 workflow without platform expansion.

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
session id: 019d93cf-4fd3-7420-8818-596d86c068fb
--------
user
# Task: Generate the next two implementation prompts for the current V1 workflow cleanup

You are working in this repository.

Your task is to write the next two bounded implementation prompt files based on the current architecture sweep and prioritized remaining work.

## Important framing

This is a prompt-generation task only.

Do NOT implement code or doc edits in this pass.
Do NOT modify existing tools or design documents in this pass.

Your job is to create the next two implementation prompts as files in `codex_prompts/`.

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
- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
- `notes/018_prioritized_remaining_work__20260416_005130.md`

## Goal

Generate exactly two prompt files:

1. a prompt to perform the recommended **doc/spec alignment cleanup**
2. a prompt to implement the recommended **lightweight record-contract validation**

These prompts should be implementation-ready, bounded, and consistent with the current repo state.

## Prompt 1 requirements

Create a prompt file for doc/spec alignment cleanup.

This future task should focus on aligning the design packet with the actual V1 toolset and current behavior, including at least:

- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- any stale references to `baby_run_prompt.py` where the actual V1 runner is now `run_prompt.py`
- any stale implication that readiness/backlog helpers do not yet exist
- the current run-id collision suffix behavior where relevant

The prompt should keep the cleanup bounded and should not turn into a broad architecture rewrite.

## Prompt 2 requirements

Create a prompt file for lightweight record-contract validation.

This future task should focus on creating a small repeatable validation layer for the shared V1 markdown execution-record shape used by:

- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

The prompt should keep the validation lightweight and inspectable.

Avoid proposing a large test framework, platform service, or dashboard.

## What each generated prompt must include

Each prompt file must:

- have a clear numeric filename prefix
- define one primary goal only
- name exact files to inspect
- define exact output artifacts to create
- include constraints that prevent broad platform growth
- define validation requirements
- define success criteria
- stay small enough to review comfortably before execution

## Also create one planning note

Create a short note at:

===== notes/019_next_two_cleanup_prompts_plan__20260416_010251.md =====
# 019 Next Two Cleanup Prompts Plan

These two prompts were chosen directly from the architecture sweep and prioritized remaining-work note because they are the highest-payoff low-risk items still left in the current V1 workflow.

They are ordered this way because doc/spec alignment should happen first. The design packet still contains stale references to `baby_run_prompt.py`, outdated assumptions about missing helpers, and unsuffixed run-id wording. Cleaning that up first makes the repo's written guidance match the actual V1 toolset before adding more implementation hardening.

The first prompt to execute should be `codex_prompts/020_align_v1_doc_and_spec_packet_to_actual_toolset.md`. After that, execute `codex_prompts/021_add_lightweight_v1_record_contract_validation.md` to protect the shared markdown execution-record shape that the current tools already depend on.

What should wait until later:

- operational guidance cleanup for legacy `__SUCCESS__` notes and the current review backlog
- retry-linkage tooling around `retry_of_run_id`
- richer queue states, scheduling, or orchestration
- broader runner refactors or any platform-style expansion beyond the current V1 slice

===== notes/020_align_v1_doc_and_spec_packet_to_actual_toolset__20260416_010453.md =====
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


===== notes/020_doc_spec_alignment_cleanup__20260416_010534.md =====
# 020 Doc Spec Alignment Cleanup

## Scope

This pass stayed bounded to doc/spec alignment for the current V1 slice.

Updated files:

- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`

No runner or helper code was changed.

## What Was Aligned

### Active V1 runner naming

- Replaced stale bridge-runner references to `tools/codex/baby_run_prompt.py` with `tools/codex/run_prompt.py` in the edited packet.
- Updated the bridge-runner spec so it reads as an alignment document for the implemented runner instead of a pre-implementation change request for a different file.

### Implemented helper set

- The packet now reflects that the current V1 workflow includes:
  - `tools/codex/run_prompt.py`
  - `tools/codex/review_run.py`
  - `tools/codex/check_queue_readiness.py`
  - `tools/codex/list_review_backlog.py`
- Removed stale framing in the bridge spec that treated review write-back or review/backlog helper support as still pending.

### Run-id wording

- Kept the base V1 identity as `<prompt_stem>__<started_at_utc>`.
- Added the implementation-accurate note that `run_prompt.py` appends a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt.

## Validation

1. Confirmed the edited design/spec files now name `tools/codex/run_prompt.py` as the active V1 runner where applicable.
2. Confirmed the edited packet no longer implies review write-back, readiness checking, or backlog listing are future work.
3. Confirmed run-id wording in the edited packet now matches current code behavior: base identity first, collision suffix only when needed.
4. Ran a search pass across the edited design/spec files and found no remaining `baby_run_prompt.py` references.

## Notes

This was intentionally not a broader architecture rewrite.
It only brought the current V1 packet back into alignment with the implemented toolset and record behavior.

===== notes/021_add_lightweight_v1_record_contract_validation__20260416_010810.md =====
# 021_add_lightweight_v1_record_contract_validation__20260416_010810

- run_id: `021_add_lightweight_v1_record_contract_validation__20260416_010810`
- prompt_file: `codex_prompts/021_add_lightweight_v1_record_contract_validation.md`
- prompt_stem: `021_add_lightweight_v1_record_contract_validation`
- started_at_utc: `20260416_010810`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260416_011435`
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

- elapsed_seconds: `385.422`
- final_output_char_count: `1541`
- stderr_char_count: `1020777`

## Prompt Text

```md
# Task: Add lightweight validation for the shared V1 execution-record contract

You are working in this repository.

Your task is to add a small, repeatable validation layer for the shared markdown execution-record shape used across the current V1 tools.

Keep this implementation lightweight and inspectable.

## Primary goal

Create one small validation path that protects the shared V1 markdown record contract without introducing a larger framework or platform surface.

## Files to inspect

Read these exact files before editing:

- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
- `notes/018_prioritized_remaining_work__20260416_005130.md`

## Required implementation scope

Create a lightweight validation layer for the shared V1 markdown execution-record shape used by:

- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

The implementation should cover only the minimum contract that these tools already depend on, such as:

- required section presence and order
- required field-line presence
- allowed execution-status values
- allowed review-status values
- the expected `run_id` pattern including the optional same-second numeric suffix
- basic consistency checks like title/run-id agreement and prompt-file/prompt-stem agreement where appropriate

## Output artifacts to create

Create exactly these artifacts:

1. one small shared validation module under `tools/codex/`
2. only the minimal script changes needed so the current V1 tools reuse that validation instead of carrying separate ad hoc contract checks
3. one short implementation note:
   - `notes/021_record_contract_validation__TIMESTAMP.md`

## Constraints

- Do not build a larger test framework
- Do not introduce a service, daemon, dashboard, or platform layer
- Do not add a database, JSON sidecar, or alternate record format
- Do not redesign the V1 markdown artifact
- Do not expand queue semantics, retry tooling, or orchestration behavior
- Do not add broad dependency or configuration systems
- Keep the validation readable enough that a reviewer can inspect it quickly in one sitting

## Validation requirements

Validate the work by doing all of the following:

1. Run a lightweight syntax check on the touched Python files.
2. Run at least one small direct validation path against existing repo records in `notes/` so the shared validator proves it can parse or reject records using the current contract.
3. Confirm `run_prompt.py` still writes records that the shared validator accepts.
4. Confirm `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py` still work with the shared validator in place.
5. Keep validation evidence in the implementation note concise and concrete.

## Success criteria

This task is successful if:

1. The shared V1 markdown record contract is enforced through one lightweight reusable validation path.
2. The four V1 scripts no longer each carry their own independent, partially duplicated contract assumptions where simple sharing would suffice.
3. The validator remains small, inspectable, and local to the current V1 workflow.
4. No broader platform growth is introduced.
5. The result is still easy to review before execution.
```

## Codex Final Output

Implemented a shared V1 record validator in [v1_record_validation.py](/data/git/handson-ml3/tools/codex/v1_record_validation.py) and rewired [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py), [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py), [check_queue_readiness.py](/data/git/handson-ml3/tools/codex/check_queue_readiness.py), and [list_review_backlog.py](/data/git/handson-ml3/tools/codex/list_review_backlog.py) to reuse that one path instead of carrying separate markdown-contract checks. The shared path now enforces required section order, required field lines, allowed status values, the `run_id` pattern with optional same-second suffix, and basic identity consistency such as title/run-id and prompt-file/prompt-stem agreement.

The implementation note is at [021_record_contract_validation__20260416_011314.md](/data/git/handson-ml3/notes/021_record_contract_validation__20260416_011314.md). Validation performed:
- `python -m py_compile` on all touched Python files
- direct validator checks against existing `notes/` content, including one valid V1 record parse and one mutated invalid-status rejection
- a stubbed `run_prompt.py` execution proving it still writes records the shared validator accepts
- a `review_run.py` write-back on that temporary record, followed by re-validation
- successful runs of `list_review_backlog.py --unreviewed-only` and `check_queue_readiness.py --prompt 021`

The worktree already had unrelated pre-existing changes in docs/prompts/notes; I left those untouched.

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
session id: 019d93d4-ce63-7401-9526-2ca58b9e1b2b
--------
user
# Task: Add lightweight validation for the shared V1 execution-record contract

You are working in this repository.

Your task is to add a small, repeatable validation layer for the shared markdown execution-record shape used across the current V1 tools.

Keep this implementation lightweight and inspectable.

## Primary goal

Create one small validation path that protects the shared V1 markdown record contract without introducing a larger framework or platform surface.

## Files to inspect

Read these exact files before editing:

- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
- `notes/018_prioritized_remaining_work__20260416_005130.md`

## Required implementation scope

Create a lightweight validation layer for the shared V1 markdown execution-record shape used by:

- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

The implementation should cover only the minimum contract that these tools already depend on, such as:

- required section presence and order
- required field-line presence
- allowed execution-status values
- allowed review-status values
- the expected `run_id` pattern including the optional same-second numeric suffix
- basic consistency checks like title/run-id agreement and prompt-file/prompt-stem agreement where appropriate

## Output artifacts to create

Create exactly these artifacts:

1. one small shared validation module under `tools/codex/`
2. only the minimal script changes needed so the current V1 tools reuse that validation instead of carrying separate ad hoc contract checks
3. one short implementation note:
   - `notes/021_record_contract_validation__TIMESTAMP.md`

## Constraints

- Do not build a larger test framework
- Do not introduce a service, daemon, dashboard, or platform layer
- Do not add a database, JSON sidecar, or alternate record format
- Do not redesign the V1 markdown artifact
- Do not expand queue semantics, retry tooling, or orchestration behavior
- Do not add broad dependency or configuration systems
- Keep the validation readable enough that a reviewer can inspect it quickly in one sitting

## Validation requirements

Validate the work by doing all of the following:

1. Run a lightweight syntax check on the touched Python files.
2. Run at least one small direct validation path against existing repo records in `notes/` so the shared validator proves it can parse or reject records using the current contract.
3. Confirm `run_prompt.py` still writes records that the shared validator accepts.
4. Confirm `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py` still work with the shared validator in place.
5. Keep validation evidence in the implementation note concise and concrete.

## Success criteria

This task is successful if:

1. The shared V1 markdown record contract is enforced through one lightweight reusable validation path.
2. The four V1 scripts no longer each carry their own independent, partially duplicated contract assumptions where simple sharing would suffice.
3. The validator remains small, inspectable, and local to the current V1 workflow.
4. No broader platform growth is introduced.
5. The result is still easy to review before execution.

codex
I’m starting by reading the requested docs and the four scripts to map the current shared record assumptions before I touch any code.
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


===== notes/021_record_contract_validation__20260416_011314.md =====
# 021 Record Contract Validation

## What Changed

- Added `tools/codex/v1_record_validation.py` as one small shared parser/validator for the V1 markdown execution-record contract.
- Moved the repeated section-order, field-line, status-value, `run_id`, and prompt-identity checks out of `run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py`.
- Kept the current V1 markdown artifact and tool boundaries unchanged.

## Contract Covered

- required section presence and order
- required field-line presence
- allowed `execution_status` values
- allowed `review_status` values
- `run_id` pattern: `<prompt_stem>__<started_at_utc>` with optional same-second `__<n>` suffix
- `title == run_id`
- `prompt_file` stem matches `prompt_stem`
- filename stem matches `run_id` when validating a record file

## Validation Evidence

- Syntax check: `python -m py_compile tools/codex/run_prompt.py tools/codex/review_run.py tools/codex/check_queue_readiness.py tools/codex/list_review_backlog.py tools/codex/v1_record_validation.py`
- Existing-record parse path: shared validator accepted `notes/011_build_v1_review_writeback_helper__20260415_235346.md`
- Existing-note non-record path: shared validator returned `None` for `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
- Existing-record reject path: mutating `review_status` from the accepted record text to ``BROKEN`` raised `record has invalid review_status`
- Runner write path: `run_prompt.py` was exercised with a stubbed `codex` executable, wrote a temporary V1 record, and that record was accepted by the shared validator before the temporary note was removed
- Review write-back path: `review_run.py` was exercised against that temporary record and the updated record still validated
- Read-only helpers: `python tools/codex/list_review_backlog.py --unreviewed-only` and `python tools/codex/check_queue_readiness.py --prompt 021` both ran successfully with the shared validator in place

## Intentional Limits

- no new framework, database, sidecar, or alternate record format
- no queue-semantics expansion
- no broader orchestration or retry tooling

codex
I’ve read the full prompt set and the current V1 scripts. The remaining work is narrowing the notes into durable versus procedural artifacts, especially separating V1 execution records from older construction notes and prompt-generation scaffolding.
exec
/bin/bash -lc "for f in notes/*.md; do printf '%s\\n' \"FILE: "'$f"; sed -n '"'1,24p' \""'$f" | sed '"'s/"'^/  /'"'; printf '\\n'; done" in /data/git/handson-ml3
exec
/bin/bash -lc "for f in notes/*.md; do printf '%s|' \""'$f"; rg -n "''^## (What Changed|Validation|Summary|Recommended Next Step|Top Prioritized Remaining Items|Current Verdict|Why These Prompts|What Was Built|What Was Aligned|Priority|Judgment|Queue note|What changed|What should happen)|''^- review_status:|''^- execution_status:|''^- Status:|''^- status:" "$f" -N || true; done' in /data/git/handson-ml3
exec
/bin/bash -lc 'git status --short notes codex_prompts tools/codex tools/*.md' in /data/git/handson-ml3
 succeeded in 0ms:
?? codex_prompts/022_audit_and_classify_scaffolding_for_summary_and_attic.md

 succeeded in 335ms:
  # Architecture And Bridge Runner Review
  
  ## Summary
  
  `tools/Codex_Prompt_Workflow_Architecture.md` is directionally aligned with the governing workflow because it pushes toward smaller parts, clearer ownership, and conservative growth. `tools/codex/baby_run_prompt.py` is acceptable as thin bridge tooling for a single bounded slice, but the current architecture doc does not yet translate the workflow into a clear V1 boundary, validation plan, review loop, or failure/resource evidence model. The next step should stay at the design layer: define the minimum execution-record artifact that every run must produce before more tooling is added.
  
  ## Architecture Alignment Findings
  
  ### Strong alignment
  
  - The architecture doc is explicitly conservative, small-part oriented, and separation-of-concerns driven, which matches design-before-build and boundaries-before-breadth in the workflow ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:92), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:5)).
  - Its decomposition of the original script into repo paths, prompt parsing, note handling, status reconstruction, retry handling, and CLI concerns is a good responsibility split and a useful architecture starting point ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:177), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:33)).
  - It repeatedly says not to build everything at once and proposes a smaller folder shape before a larger one, which is consistent with thin slices before large pushes ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:100), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:121), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:246)).
  
  ### Stale, incomplete, or misaligned
  
  - The architecture doc does not yet reflect the workflow's full sequence. It jumps from decomposition into a wish list and folder layout, but it does not define a short problem statement, explicit scope fence, out-of-scope list, minimum viable slice, artifact inventory, implementation order, or validation checklist as the governing workflow expects ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:141), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:159), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:196), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:214), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:233), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:251)).
  - The "eventually" section is useful as an extension path, but it is carrying too much conceptual breadth for a workflow that emphasizes bounded next steps. There is no explicit line saying which of those items are intentionally deferred from V1 ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:96), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:374), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:119)).
  - The document mentions logging and diagnostics, richer notes, retry intelligence, approval gates, and dependencies, but it does not specify the source of truth or the stable identity of important records beyond prompt stems. That leaves the durable-history model underdesigned relative to the workflow's emphasis on preserved local memory and stable identities ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:120), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:374), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:187)).
  - Review points between iterations are largely absent. The architecture doc talks about manual status marking and approval gates eventually, but not about the concrete review stop that should happen after each bounded run before the next run proceeds ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:262), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:208)).
  
  ## Runner Alignment Findings
  
  ### Useful and appropriately thin

FILE: notes/004_next_design_step_recommendation__20260415_195538.md
  # Next Design Step Recommendation
  
  Define the V1 execution-record artifact for a single prompt run.
  
  This should happen next because the current system can execute prompts and write notes, but it does not yet define the minimum durable record needed for review, failure analysis, and lightweight cost observation. Without that decision, more implementation risks hard-coding the wrong note model and the wrong success semantics.
  
  The artifact should be a short design note that specifies:
  
  - the source of truth for one run record
  - the minimum fields for execution facts, reviewed outcome, failure-analysis notes, and lightweight metrics
  - which fields are captured automatically now
  - which fields remain manual until later
  
  What should wait until later:
  
  - module refactors and file splits
  - richer status/state machines
  - retry intelligence
  - dependency handling
  - approval-gate automation
  - reporting layers beyond the single-run record

FILE: notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md
  # 004_review_architecture_and_bridge_runner_against_workflow - SUCCESS
  
  - Prompt file: `004_review_architecture_and_bridge_runner_against_workflow.md`
  - Timestamp (UTC): `20260415_195505`
  - Status: `SUCCESS`
  
  ## Original Prompt
  
  ```md
  # Task: Review the Codex prompt workflow architecture and the baby bridge runner against the current project design workflow
  
  You are working in this repository.
  
  Your task is to review two files against the current workflow guidance and produce a small design assessment.
  
  ## Files to inspect
  
  Read these exact files:
  
  - `tools/Project_Design_Workflow.md`
  - `tools/Codex_Prompt_Workflow_Architecture.md`
  - `tools/codex/baby_run_prompt.py`
  
  ## Goal

FILE: notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md
  # 005_define_execution_record_and_generate_next_prompt_queue - SUCCESS
  
  - Prompt file: `005_define_execution_record_and_generate_next_prompt_queue.md`
  - Timestamp (UTC): `20260415_202522`
  - Status: `SUCCESS`
  
  ## Original Prompt
  
  ```md
  # Task: Define the V1 execution-record artifact and generate the next bounded prompt queue
  
  You are working in this repository.
  
  Your task has two tightly related parts:
  
  1. define the V1 execution-record artifact for one prompt run
  2. use that design decision to generate the next bounded sequence of prompt files needed to reduce the current workflow/architecture/runner misalignment
  
  ## Files to inspect
  
  Read these exact files:
  
  - `tools/Project_Design_Workflow.md`
  - `tools/Codex_Prompt_Workflow_Architecture.md`

FILE: notes/005_prompt_queue_plan__20260415_202557.md
  # Prompt Queue Plan
  
  ## Why These Prompts Were Chosen
  
  These prompts were chosen because the current misalignment is concentrated at one seam:
  the system can execute prompts, but it does not yet define the review checkpoint, the V1 architectural boundary around that checkpoint, or the smallest runner behavior needed to support it.
  
  The queue stays small on purpose.
  It settles the workflow gap before any implementation prompt is allowed to widen scope.
  
  ## Why This Order Reduces Risk
  
  The order moves from policy to architecture to bridge-tooling:
  
  1. define the review gate
  2. align the architecture doc to that gate and the execution record
  3. define the smallest runner change spec that fits those decisions
  
  This reduces the risk of hard-coding the wrong note model or the wrong queue semantics into the runner before the manual review loop is settled.
  
  ## What Each Prompt Is Meant To Settle
  
  `006_define_v1_run_review_gate.md`
  

FILE: notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md
  # 006_define_v1_run_review_gate - SUCCESS
  
  - Prompt file: `006_define_v1_run_review_gate.md`
  - Timestamp (UTC): `20260415_203019`
  - Status: `SUCCESS`
  
  ## Original Prompt
  
  ```md
  # Task: Define the V1 run-review gate for prompt execution
  
  You are working in this repository.
  
  Your task is to define the minimum review gate that must sit between:
  
  - a prompt run finishing
  - the next prompt in sequence being treated as ready
  
  ## Files to inspect
  
  Read these exact files:
  
  - `tools/Project_Design_Workflow.md`
  - `tools/Codex_Prompt_Workflow_Architecture.md`

FILE: notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md
  # 007_align_architecture_doc_to_v1_workflow - SUCCESS
  
  - Prompt file: `007_align_architecture_doc_to_v1_workflow.md`
  - Timestamp (UTC): `20260415_203257`
  - Status: `SUCCESS`
  
  ## Original Prompt
  
  ```md
  # Task: Align the Codex prompt architecture document to the governing workflow and V1 run artifact
  
  You are working in this repository.
  
  Your task is to bring the architecture document into explicit V1 alignment with:
  
  - the governing workflow
  - the V1 execution record
  - the V1 review gate
  
  ## Files to inspect
  
  Read these exact files:
  
  - `tools/Project_Design_Workflow.md`

FILE: notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md
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

FILE: notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md
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

FILE: notes/009_run_prompt_candidate_build__20260415_233407.md
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

FILE: notes/010_next_step_recommendation__20260415_234559.md
  # 010 Next Step Recommendation
  
  Chosen next move: `environment diagnosis prompt`
  
  Why it should happen next:
  The V1 runner candidate already produces the intended execution record shape closely enough, and the validation run failed during Codex session initialization with a read-only filesystem error. That makes environment diagnosis the smallest step that can unblock meaningful validation without broadening the build.
  
  What it should produce:
  A short diagnosis artifact that identifies where session startup is trying to write, whether the failure reproduces independently of `tools/codex/run_prompt.py`, and one bounded follow-up action.
  
  What should explicitly wait:
  Runner polish, review write-back helpers, and any broader workflow or runner build-out.

FILE: notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md
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
  

FILE: notes/010_run_prompt_candidate_review__20260415_234559.md
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
  

FILE: notes/011_build_v1_review_writeback_helper__20260415_235346.md
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
  

FILE: notes/011_review_writeback_helper_build__20260415_235514.md
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

FILE: notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md
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
  

FILE: notes/012_top_three_next_options__20260416_000819.md
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
  

FILE: notes/012_v1_pipeline_options_review__20260416_000819.md
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

FILE: notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md
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
  

FILE: notes/013_next_two_prompt_plan__20260416_002005.md
  # 013 Next Two Prompt Plan
  
  The next two prompts were chosen because the recent options review identified the main remaining V1 friction as operational support around repeated reviewed use, not missing architecture. The strongest bounded additions are the queue-readiness checker and the review backlog / unreviewed-run lister because both rely on the now-real execution-record and review-writeback artifacts without expanding the workflow into a larger engine.
  
  They are ordered this way because readiness checking closes the most important architectural seam first: the V1 rule that only an accepted reviewed run should release the next prompt. The review backlog lister should follow immediately after because it improves day-to-day review discovery, but it is slightly less central than the readiness decision itself.
  
  The first prompt to execute should be `codex_prompts/014_build_queue_readiness_checker.md`.
  
  What should wait until later:
  
  - lightweight contract validation for `tools/codex/run_prompt.py` and `tools/codex/review_run.py`
  - doc-alignment cleanup for remaining naming and run-id drift
  - rejected-run / retry-linkage support beyond the current placeholder fields
  - any broader dashboard, queue engine, dependency system, or workflow automation layer

FILE: notes/014_build_queue_readiness_checker__20260416_002319.md
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
  

FILE: notes/014_queue_readiness_checker_build__20260416_002419.md
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

FILE: notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md
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
  

FILE: notes/015_review_backlog_lister_build__20260416_010500.md
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

FILE: notes/016_next_improvement_recommendation__20260416_003710.md
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

FILE: notes/016_queue_and_backlog_helper_validation__20260416_003710.md
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
  

FILE: notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md
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
  

FILE: notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md
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
  

FILE: notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md
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

FILE: notes/018_architecture_vs_actual_sweep__20260416_005130.md
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

FILE: notes/018_prioritized_remaining_work__20260416_005130.md
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

FILE: notes/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work__20260416_005034.md
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
  

FILE: notes/019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209.md
  # 019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209
  
  - run_id: `019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209`
  - prompt_file: `codex_prompts/019_generate_prompts_for_doc_alignment_and_record_contract_validation.md`
  - prompt_stem: `019_generate_prompts_for_doc_alignment_and_record_contract_validation`
  - started_at_utc: `20260416_010209`
  
  ## Execution Facts
  
  - execution_status: `EXECUTED`
  - finished_at_utc: `20260416_010353`
  - runner: `tools/codex/run_prompt.py`
  - return_code: `0`
  - retry_of_run_id:
  
  ## Review Facts
  
  - review_status: `UNREVIEWED`
  - review_summary:
  - reviewed_by:
  - reviewed_at_utc:
  
  ## Failure Analysis
  

FILE: notes/019_next_two_cleanup_prompts_plan__20260416_010251.md
  # 019 Next Two Cleanup Prompts Plan
  
  These two prompts were chosen directly from the architecture sweep and prioritized remaining-work note because they are the highest-payoff low-risk items still left in the current V1 workflow.
  
  They are ordered this way because doc/spec alignment should happen first. The design packet still contains stale references to `baby_run_prompt.py`, outdated assumptions about missing helpers, and unsuffixed run-id wording. Cleaning that up first makes the repo's written guidance match the actual V1 toolset before adding more implementation hardening.
  
  The first prompt to execute should be `codex_prompts/020_align_v1_doc_and_spec_packet_to_actual_toolset.md`. After that, execute `codex_prompts/021_add_lightweight_v1_record_contract_validation.md` to protect the shared markdown execution-record shape that the current tools already depend on.
  
  What should wait until later:
  
  - operational guidance cleanup for legacy `__SUCCESS__` notes and the current review backlog
  - retry-linkage tooling around `retry_of_run_id`
  - richer queue states, scheduling, or orchestration
  - broader runner refactors or any platform-style expansion beyond the current V1 slice

FILE: notes/020_align_v1_doc_and_spec_packet_to_actual_toolset__20260416_010453.md
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
  

FILE: notes/020_doc_spec_alignment_cleanup__20260416_010534.md
  # 020 Doc Spec Alignment Cleanup
  
  ## Scope
  
  This pass stayed bounded to doc/spec alignment for the current V1 slice.
  
  Updated files:
  
  - `tools/Codex_Prompt_Workflow_Architecture.md`
  - `tools/codex/V1_Execution_Record_Artifact.md`
  - `tools/codex/V1_Bridge_Runner_Change_Spec.md`
  
  No runner or helper code was changed.
  
  ## What Was Aligned
  
  ### Active V1 runner naming
  
  - Replaced stale bridge-runner references to `tools/codex/baby_run_prompt.py` with `tools/codex/run_prompt.py` in the edited packet.
  - Updated the bridge-runner spec so it reads as an alignment document for the implemented runner instead of a pre-implementation change request for a different file.
  
  ### Implemented helper set
  
  - The packet now reflects that the current V1 workflow includes:

FILE: notes/021_add_lightweight_v1_record_contract_validation__20260416_010810.md
  # 021_add_lightweight_v1_record_contract_validation__20260416_010810
  
  - run_id: `021_add_lightweight_v1_record_contract_validation__20260416_010810`
  - prompt_file: `codex_prompts/021_add_lightweight_v1_record_contract_validation.md`
  - prompt_stem: `021_add_lightweight_v1_record_contract_validation`
  - started_at_utc: `20260416_010810`
  
  ## Execution Facts
  
  - execution_status: `EXECUTED`
  - finished_at_utc: `20260416_011435`
  - runner: `tools/codex/run_prompt.py`
  - return_code: `0`
  - retry_of_run_id:
  
  ## Review Facts
  
  - review_status: `UNREVIEWED`
  - review_summary:
  - reviewed_by:
  - reviewed_at_utc:
  
  ## Failure Analysis
  

FILE: notes/021_record_contract_validation__20260416_011314.md
  # 021 Record Contract Validation
  
  ## What Changed
  
  - Added `tools/codex/v1_record_validation.py` as one small shared parser/validator for the V1 markdown execution-record contract.
  - Moved the repeated section-order, field-line, status-value, `run_id`, and prompt-identity checks out of `run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py`.
  - Kept the current V1 markdown artifact and tool boundaries unchanged.
  
  ## Contract Covered
  
  - required section presence and order
  - required field-line presence
  - allowed `execution_status` values
  - allowed `review_status` values
  - `run_id` pattern: `<prompt_stem>__<started_at_utc>` with optional same-second `__<n>` suffix
  - `title == run_id`
  - `prompt_file` stem matches `prompt_stem`
  - filename stem matches `run_id` when validating a record file
  
  ## Validation Evidence
  
  - Syntax check: `python -m py_compile tools/codex/run_prompt.py tools/codex/review_run.py tools/codex/check_queue_readiness.py tools/codex/list_review_backlog.py tools/codex/v1_record_validation.py`
  - Existing-record parse path: shared validator accepted `notes/011_build_v1_review_writeback_helper__20260415_235346.md`
  - Existing-note non-record path: shared validator returned `None` for `notes/018_architecture_vs_actual_sweep__20260416_005130.md`


 succeeded in 945ms:
- status: `SUCCESS`
notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md|- Status: `SUCCESS`
- status: `SUCCESS`
notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md|- Status: `SUCCESS`
notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md|notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md|- Status: `SUCCESS`
notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md|## What Changed
notes/003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md|- Status: `SUCCESS`
## What Changed
notes/004_architecture_and_bridge_runner_review__20260415_195538.md|## Summary
notes/004_next_design_step_recommendation__20260415_195538.md|notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md|- Status: `SUCCESS`
- Status: `{status}`
## Summary
notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md|- Status: `SUCCESS`
- Status: `{status}`
## Summary
## Why These Prompts Were Chosen
notes/005_prompt_queue_plan__20260415_202557.md|## Why These Prompts Were Chosen
notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md|- Status: `SUCCESS`
- Status: `{status}`
## Summary
## Why These Prompts Were Chosen
notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md|- Status: `SUCCESS`
## Summary
## Why These Prompts Were Chosen
## Validation And Review Posture
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md|- Status: `SUCCESS`
- Status: `{status}`
## Summary
## Why These Prompts Were Chosen
## Validation And Review Posture
- execution_status: `<EXECUTED|EXECUTION_FAILED>`
- review_status: `UNREVIEWED`
- execution_status: `<EXECUTED|EXECUTION_FAILED>`
- review_status: `UNREVIEWED`
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md|- Status: `SUCCESS`
## Validation requirements
## Validation requirements
- execution_status: `<EXECUTED|EXECUTION_FAILED>`
- review_status: `UNREVIEWED`
- Status: `{status}`
- execution_status: `EXECUTION_FAILED`
- review_status: `UNREVIEWED`
## Summary
## Validation
notes/009_run_prompt_candidate_build__20260415_233407.md|## Summary
## Validation
notes/010_next_step_recommendation__20260415_234559.md|notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md|- Status: `SUCCESS`
- execution_status: `<EXECUTED|EXECUTION_FAILED>`
- review_status: `UNREVIEWED`
## Summary
## Validation
- execution_status: `EXECUTION_FAILED`
- review_status: `UNREVIEWED`
- Status: `{status}`
- status: `SUCCESS`
- Status: `SUCCESS`
notes/010_run_prompt_candidate_review__20260415_234559.md|notes/011_build_v1_review_writeback_helper__20260415_235346.md|- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
## Validation requirements
## Validation requirements
- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
- Status: `{status}`
## Summary
## Validation
- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
- execution_status: `EXECUTED`
- review_status: `ACCEPTED`
notes/011_review_writeback_helper_build__20260415_235514.md|## What Was Built
## Validation
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md|- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
- execution_status: `<EXECUTED|EXECUTION_FAILED>`
- review_status: `UNREVIEWED`
- Status: `{status}`
## Summary
## Validation
## What Was Built
## Validation
- execution_status: `EXECUTED`
- review_status: `ACCEPTED`
- execution_status: `EXECUTION_FAILED`
- review_status: `UNREVIEWED`
- status: `SUCCESS`
- Status: `SUCCESS`
- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
## Validation requirements
## Validation And Review Posture
notes/012_top_three_next_options__20260416_000819.md|notes/012_v1_pipeline_options_review__20260416_000819.md|notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md|- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
## Validation requirements
## Validation requirements
## Validation requirements
notes/013_next_two_prompt_plan__20260416_002005.md|notes/014_build_queue_readiness_checker__20260416_002319.md|- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
## Validation requirements
## Validation requirements
- execution_status: `EXECUTED`
- review_status: `ACCEPTED`
- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
## Validation requirements
- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
- Status: `SUCCESS`
- Status: `SUCCESS`
- execution_status: `EXECUTION_FAILED`
- review_status: `UNREVIEWED`
notes/014_queue_readiness_checker_build__20260416_002419.md|## What Was Built
## Validation Performed
notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md|- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
## Validation requirements
## Validation requirements
- execution_status: `EXECUTION_FAILED`
- review_status: `UNREVIEWED`
- execution_status: `EXECUTED`
- review_status: `ACCEPTED`
- Status: `{status}`
## What Was Built
## Validation Performed
notes/015_review_backlog_lister_build__20260416_010500.md|## What Was Built
## Validation Performed
notes/016_next_improvement_recommendation__20260416_003710.md|notes/016_queue_and_backlog_helper_validation__20260416_003710.md|notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md|- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
## Validation requirements
## Validation requirements
- Status: `SUCCESS`
- Status: `SUCCESS`
- Status: `SUCCESS`
- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md|- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
## Validation requirements
## Validation requirements
notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md|## Summary
## Validation
notes/018_architecture_vs_actual_sweep__20260416_005130.md|## Judgment
notes/018_prioritized_remaining_work__20260416_005130.md|notes/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work__20260416_005034.md|- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
- execution_status: `<EXECUTED|EXECUTION_FAILED>`
- review_status: `UNREVIEWED`
## What Was Built
## Validation Performed
## What Was Built
## Validation Performed
## Summary
## Validation
## Validation And Review Posture
## What Was Built
## Validation
## Judgment
notes/019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209.md|- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
- execution_status: `<EXECUTED|EXECUTION_FAILED>`
- review_status: `UNREVIEWED`
## Judgment
## Validation requirements
## Validation requirements
notes/019_next_two_cleanup_prompts_plan__20260416_010251.md|notes/020_align_v1_doc_and_spec_packet_to_actual_toolset__20260416_010453.md|- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
## Validation requirements
## Validation requirements
- execution_status: `<EXECUTED|EXECUTION_FAILED>`
- review_status: `UNREVIEWED`
## Judgment
## Validation And Review Posture
notes/020_doc_spec_alignment_cleanup__20260416_010534.md|## What Was Aligned
## Validation
notes/021_add_lightweight_v1_record_contract_validation__20260416_010810.md|- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
## Validation requirements
## Validation requirements
## Judgment
- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
## Validation requirements
- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
## Validation requirements
- execution_status: `EXECUTED`
- review_status: `UNREVIEWED`
## Validation requirements
- Status: `SUCCESS`
## What Changed
## Validation Evidence
notes/021_record_contract_validation__20260416_011314.md|## What Changed
## Validation Evidence

codex
I’ve isolated the durable note types: design-review notes, implementation/validation notes, and a large set of raw execution receipts. I’m doing one more targeted pass on the mid-sequence prompts and notes so the written classification can stay specific instead of flattening everything into “old scaffolding.”
exec
/bin/bash -lc "for f in codex_prompts/009_build_v1_run_prompt_candidate.md codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md codex_prompts/011_build_v1_review_writeback_helper.md codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md codex_prompts/014_build_queue_readiness_checker.md codex_prompts/015_build_review_backlog_unreviewed_run_lister.md codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md codex_prompts/017_polish_queue_readiness_gap_explanation.md; do printf '\\n===== %s =====\\n' \""'$f"; sed -n '"'1,220p' \""'$f"; done' in /data/git/handson-ml3
exec
/bin/bash -lc "for f in notes/011_review_writeback_helper_build__20260415_235514.md notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md notes/012_top_three_next_options__20260416_000819.md notes/014_queue_readiness_checker_build__20260416_002419.md notes/015_review_backlog_lister_build__20260416_010500.md notes/018_architecture_vs_actual_sweep__20260416_005130.md notes/018_prioritized_remaining_work__20260416_005130.md; do printf '\\n===== %s =====\\n' \""'$f"; sed -n '"'1,220p' \""'$f"; done' in /data/git/handson-ml3
 succeeded in 0ms:

===== notes/011_review_writeback_helper_build__20260415_235514.md =====
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

===== notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md =====
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
- `notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md`
- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`

## Goal

Review the full current V1 workflow pipe and answer:

1. what pieces now exist and are functioning
2. what important gaps or awkward seams remain
3. what realistic next bounded options are available
4. which three options are the strongest next candidates

## What to evaluate

Please evaluate the current state of the pipe across these areas:

### 1. Execution path
- prompt resolution
- Codex invocation
- execution-record creation
- stability of record identity
- resource/failure evidence capture

### 2. Review path
- manual review write-back
- preservation of record structure
- consistency with the V1 review gate
- whether review is now operationally usable

### 3. Workflow usability
- whether the V1 flow is now practical for repeated use
- what still feels awkward, manual, fragile, or under-supported
- what is missing if this is going to be used regularly

### 4. Documentation alignment
- whether the code and notes still match the design packet
- whether there is any doc drift or naming drift that should be addressed
- whether there are any small correctness mismatches between design and implementation

## Required outputs

Create exactly two artifacts.

### Artifact 1
Create a review report at:

`notes/012_v1_pipeline_options_review__TIMESTAMP.md`

This report should include:

- short summary of current pipeline maturity
- what is working now
- what seams or gaps remain
- a list of realistic next bounded options

For the options list:
- provide at least 5 options if there are 5 credible ones
- provide fewer only if the repo truly supports fewer
- each option should include:
  - short name
  - what it would build or improve
  - why it matters
  - expected risk level: low / medium / high
  - expected payoff level: low / medium / high

### Artifact 2
Create a short recommendation note at:

`notes/012_top_three_next_options__TIMESTAMP.md`

This note should contain:

- the top three next options in ranked order
- why each made the top three
- which one should happen next
- what should explicitly wait

## Ranking guidance

When ranking options, prefer:
- bounded steps
- low-risk progress
- improvements that strengthen repeated operational use
- improvements that reduce ambiguity or manual fragility
- improvements supported by the current repo state

Avoid preferring:
- broad platform expansion
- premature refactors
- speculative subsystems
- large workflow engines

## Constraints

1. Use the exact file paths listed above.
2. Do not implement anything in this pass.
3. Do not produce a giant roadmap.
4. Recommend only bounded next options that are plausible from the current state.
5. Keep the tone practical and inspectable.
6. Let the repo evidence drive the options.

## Success criteria

This task is successful if:

- the review clearly describes what the V1 pipe can do now
- the options are grounded in actual repo artifacts and recent notes
- the report gives a realistic menu of next moves
- the recommendation note ranks the top three clearly
- the result helps a human choose the next deliberate step with confidence
```

## Codex Final Output

Created the two requested artifacts:

- [012_v1_pipeline_options_review__20260416_000819.md](/data/git/handson-ml3/notes/012_v1_pipeline_options_review__20260416_000819.md)
- [012_top_three_next_options__20260416_000819.md](/data/git/handson-ml3/notes/012_top_three_next_options__20260416_000819.md)

The review concludes the V1 pipe is now a real usable thin slice: `run_prompt.py` creates durable execution records, `review_run.py` performs in-place manual review write-back, and the main remaining weakness is operational support for repeated use rather than missing architecture. The top recommendation is to add a small queue-readiness checker next, followed by an unreviewed-run lister and then lightweight contract validation for the two V1 scripts.

I did not modify `tools/codex/baby_run_prompt.py`, `tools/codex/run_prompt.py`, or `tools/codex/review_run.py`.

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
session id: 019d939c-cbad-7e11-9d95-7794a03d6150
--------
user
# Task: Review the current V1 prompt workflow pipeline and recommend the best next options


===== notes/012_top_three_next_options__20260416_000819.md =====
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

===== notes/014_queue_readiness_checker_build__20260416_002419.md =====
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

===== notes/015_review_backlog_lister_build__20260416_010500.md =====
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

===== notes/018_architecture_vs_actual_sweep__20260416_005130.md =====
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

===== notes/018_prioritized_remaining_work__20260416_005130.md =====
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

 succeeded in 107ms:
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

===== codex_prompts/017_polish_queue_readiness_gap_explanation.md =====
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
I have enough context to write the classification. I’m keeping it conservative: durable design/validation notes stay or get summarized, raw execution receipts and superseded prompt-generation steps go to attic, and anything that still anchors current operational guidance is not being aggressively retired.
exec
/bin/bash -lc "date -u '+%Y%m%d_%H%M%S'" in /data/git/handson-ml3
 succeeded in 0ms:
20260416_023016

codex
I’m writing the two requested notes now. The report will make one crucial distinction explicit: current V1 execution records in `notes/` are not just historical clutter, they are still live inputs to the backlog/readiness helpers, so I’m not classifying them as safe attic material in this pass.
apply patch
patch: completed
/data/git/handson-ml3/notes/022_scaffolding_classification_report__20260416_023016.md
diff --git a/notes/022_scaffolding_classification_report__20260416_023016.md b/notes/022_scaffolding_classification_report__20260416_023016.md
new file mode 100644
index 0000000000000000000000000000000000000000..ff6b5ff2902c910c2b6d5b21196592666fef35c0
--- /dev/null
+++ b/notes/022_scaffolding_classification_report__20260416_023016.md
@@ -0,0 +1,189 @@
+# 022 Scaffolding Classification Report
+
+## Short Summary
+
+`notes/` currently mixes four different things:
+
+- legacy pre-V1 success receipts
+- durable design and implementation notes
+- current V1 execution records that the live helpers still read
+- small planning / recommendation notes that were useful during construction but are now mostly superseded
+
+`codex_prompts/` is almost entirely construction scaffolding for building the current V1 toolset. The durable knowledge from those prompts now lives in the tool/docs packet and in a smaller set of implementation and review notes.
+
+The main conservative rule in this report is:
+
+- current V1 execution-record files stay out of attic decisions for now, because moving them would change the behavior of `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py`
+
+## Classification Criteria Used
+
+- `permanent residents`: still useful as standing maintainer context in their current form
+- `summarize, then move to attic`: contains durable decisions, validation evidence, or design rationale, but does not need to stay in `notes/` once that knowledge is extracted
+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
+- `uncertain / needs human review`: not safe to move yet because it is still operationally active, still part of the current source-of-truth set, or still too close to the active cleanup pass
+
+## Permanent Residents
+
+- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
+  Reason: best compact repo-specific statement of what the V1 system currently is, what remains missing, and whether it is ready for real use.
+- `notes/020_doc_spec_alignment_cleanup__20260416_010534.md`
+  Reason: explains why the current design/spec packet changed and records the final doc-alignment decisions that still govern the written packet.
+- `notes/021_record_contract_validation__20260416_011314.md`
+  Reason: durable contract and validation evidence for the shared V1 execution-record parser used by live tools.
+
+## Summarize, Then Move To Attic
+
+- `notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md`
+  Reason: useful initial repo-context snapshot, but stale as a standing note.
+- `notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md`
+  Reason: explains why `tools/Project_Design_Workflow.md` was revised and what reusable principles were added.
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+  Reason: foundational mismatch review that drove the V1 execution-record and review-gate design.
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+  Reason: captures why the early V1 sequence was ordered policy first, architecture second, runner change last.
+- `notes/009_run_prompt_candidate_build__20260415_233407.md`
+  Reason: records what behavior `run_prompt.py` intentionally preserved and what changed for V1 records.
+- `notes/010_run_prompt_candidate_review__20260415_234559.md`
+  Reason: captures the first serious assessment of the new runner and the environment-vs-runner distinction.
+- `notes/011_review_writeback_helper_build__20260415_235514.md`
+  Reason: implementation and validation note for the still-live `review_run.py` helper.
+- `notes/012_v1_pipeline_options_review__20260416_000819.md`
+  Reason: useful maturity review and bounded option set before operational helpers were added.
+- `notes/014_queue_readiness_checker_build__20260416_002419.md`
+  Reason: implementation and validation note for the still-live readiness helper.
+- `notes/015_review_backlog_lister_build__20260416_010500.md`
+  Reason: implementation and validation note for the still-live review backlog helper.
+- `notes/016_queue_and_backlog_helper_validation__20260416_003710.md`
+  Reason: repo-specific validation evidence showing how readiness and backlog behavior interact with mixed legacy and V1 notes.
+- `notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md`
+  Reason: records the rationale for the current `Queue note:` behavior in the readiness helper.
+
+## Move To Attic Without Summary
+
+### notes/
+
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md`
+  Reason: legacy smoke-test receipt; superseded by later V1 records and duplicates.
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md`
+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md`
+  Reason: duplicate legacy smoke-test receipt with no new durable decision.
+- `notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md`
+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
+- `notes/003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md`
+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
+- `notes/004_next_design_step_recommendation__20260415_195538.md`
+  Reason: brief recommendation fully subsumed by the main review and later execution-record work.
+- `notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md`
+  Reason: success receipt; the main review note is the durable artifact.
+- `notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md`
+  Reason: success receipt; resulting doc and queue-plan note carry the lasting value.
+- `notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md`
+  Reason: success receipt; durable knowledge lives in `tools/codex/V1_Run_Review_Gate.md`.
+- `notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md`
+  Reason: success receipt; durable knowledge lives in the architecture doc.
+- `notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md`
+  Reason: success receipt; durable knowledge lives in the bridge-runner spec.
+- `notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md`
+  Reason: implementation receipt; the build note and the code itself carry the durable content.
+- `notes/010_next_step_recommendation__20260415_234559.md`
+  Reason: brief next-step recommendation that was quickly superseded by the actual sequence taken.
+- `notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md`
+  Reason: success receipt; the review note carries the useful content.
+- `notes/012_top_three_next_options__20260416_000819.md`
+  Reason: short ranking note largely duplicated by the fuller options review and later implementation history.
+- `notes/013_next_two_prompt_plan__20260416_002005.md`
+  Reason: prompt-generation plan already realized by `014` and `015`.
+- `notes/016_next_improvement_recommendation__20260416_003710.md`
+  Reason: one-line recommendation already realized by `017`.
+- `notes/018_prioritized_remaining_work__20260416_005130.md`
+  Reason: concise ranking duplicated by the fuller architecture sweep and the implemented follow-up work.
+- `notes/019_next_two_cleanup_prompts_plan__20260416_010251.md`
+  Reason: prompt-generation plan already realized by `020` and `021`.
+
+### codex_prompts/
+
+- `codex_prompts/001_smoke_test_pipeline.md`
+  Reason: one-use bootstrap prompt; durable outcome was only the existence of a working notes path.
+- `codex_prompts/002_repo_inventory_and_status.md`
+  Reason: one-use repo-inventory scaffold; durable value is in the resulting note, not the prompt text.
+- `codex_prompts/003_revise_Project_Deisgn_workflow_document.md`
+  Reason: implementation instruction for a completed doc revision; resulting document and revision note now carry the knowledge.
+- `codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md`
+  Reason: review scaffold whose durable output is the architecture/runner assessment note.
+- `codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md`
+  Reason: queue-generation scaffold; durable outputs are the execution-record doc and queue-plan note.
+- `codex_prompts/006_define_v1_run_review_gate.md`
+  Reason: design scaffold whose durable output is `tools/codex/V1_Run_Review_Gate.md`.
+- `codex_prompts/007_align_architecture_doc_to_v1_workflow.md`
+  Reason: alignment scaffold whose durable output is the revised architecture doc.
+- `codex_prompts/008_define_minimal_bridge_runner_change_spec.md`
+  Reason: spec-writing scaffold whose durable output is the bridge-runner spec.
+- `codex_prompts/009_build_v1_run_prompt_candidate.md`
+  Reason: implementation prompt for completed runner work; code and build note now carry the value.
+- `codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md`
+  Reason: review scaffold whose durable output is the runner-review note.
+- `codex_prompts/011_build_v1_review_writeback_helper.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md`
+  Reason: review scaffold whose durable output is the options review note.
+- `codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md`
+  Reason: prompt-generation scaffold already realized by later prompt files and notes.
+- `codex_prompts/014_build_queue_readiness_checker.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/015_build_review_backlog_unreviewed_run_lister.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md`
+  Reason: validation scaffold whose durable content is in the validation note.
+- `codex_prompts/017_polish_queue_readiness_gap_explanation.md`
+  Reason: small polish prompt for completed helper behavior; code and polish note now carry the value.
+- `codex_prompts/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work.md`
+  Reason: review scaffold whose durable output is the architecture sweep note.
+- `codex_prompts/019_generate_prompts_for_doc_alignment_and_record_contract_validation.md`
+  Reason: prompt-generation scaffold already realized by `020` and `021`.
+- `codex_prompts/020_align_v1_doc_and_spec_packet_to_actual_toolset.md`
+  Reason: implementation prompt for completed doc cleanup; the edited docs and cleanup note now carry the value.
+- `codex_prompts/021_add_lightweight_v1_record_contract_validation.md`
+  Reason: implementation prompt for completed validator work; code and validation note now carry the value.
+
+## Uncertain / Needs Human Review
+
+### notes/
+
+- `notes/001_smoke_test_pipeline__20260415_233343.md`
+  Reason: current V1 execution record; moving it changes live run history and helper output.
+- `notes/001_smoke_test_pipeline__20260415_234918.md`
+  Reason: current V1 execution record and current accepted prior evidence for prompt `001`.
+- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/014_build_queue_readiness_checker__20260416_002319.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work__20260416_005034.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/020_align_v1_doc_and_spec_packet_to_actual_toolset__20260416_010453.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/021_add_lightweight_v1_record_contract_validation__20260416_010810.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+
+### codex_prompts/
+
+- `codex_prompts/022_audit_and_classify_scaffolding_for_summary_and_attic.md`
+  Reason: current audit prompt for the cleanup sequence; do not classify as attic material until this report has been acted on and superseded.
+
+## Practical Read Of The Repo
+
+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
+- The second cleanup target is the durable-but-noisy implementation notes, but only after their key rationale is extracted into a smaller standing summary.
+- The current V1 execution records should not move in a cleanup pass unless the repo first decides how the readiness/backlog tools should treat archived records.

diff --git a/notes/022_scaffolding_classification_report__20260416_023016.md b/notes/022_scaffolding_classification_report__20260416_023016.md
new file mode 100644
index 0000000000000000000000000000000000000000..ff6b5ff2902c910c2b6d5b21196592666fef35c0
--- /dev/null
+++ b/notes/022_scaffolding_classification_report__20260416_023016.md
@@ -0,0 +1,189 @@
+# 022 Scaffolding Classification Report
+
+## Short Summary
+
+`notes/` currently mixes four different things:
+
+- legacy pre-V1 success receipts
+- durable design and implementation notes
+- current V1 execution records that the live helpers still read
+- small planning / recommendation notes that were useful during construction but are now mostly superseded
+
+`codex_prompts/` is almost entirely construction scaffolding for building the current V1 toolset. The durable knowledge from those prompts now lives in the tool/docs packet and in a smaller set of implementation and review notes.
+
+The main conservative rule in this report is:
+
+- current V1 execution-record files stay out of attic decisions for now, because moving them would change the behavior of `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py`
+
+## Classification Criteria Used
+
+- `permanent residents`: still useful as standing maintainer context in their current form
+- `summarize, then move to attic`: contains durable decisions, validation evidence, or design rationale, but does not need to stay in `notes/` once that knowledge is extracted
+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
+- `uncertain / needs human review`: not safe to move yet because it is still operationally active, still part of the current source-of-truth set, or still too close to the active cleanup pass
+
+## Permanent Residents
+
+- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
+  Reason: best compact repo-specific statement of what the V1 system currently is, what remains missing, and whether it is ready for real use.
+- `notes/020_doc_spec_alignment_cleanup__20260416_010534.md`
+  Reason: explains why the current design/spec packet changed and records the final doc-alignment decisions that still govern the written packet.
+- `notes/021_record_contract_validation__20260416_011314.md`
+  Reason: durable contract and validation evidence for the shared V1 execution-record parser used by live tools.
+
+## Summarize, Then Move To Attic
+
+- `notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md`
+  Reason: useful initial repo-context snapshot, but stale as a standing note.
+- `notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md`
+  Reason: explains why `tools/Project_Design_Workflow.md` was revised and what reusable principles were added.
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+  Reason: foundational mismatch review that drove the V1 execution-record and review-gate design.
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+  Reason: captures why the early V1 sequence was ordered policy first, architecture second, runner change last.
+- `notes/009_run_prompt_candidate_build__20260415_233407.md`
+  Reason: records what behavior `run_prompt.py` intentionally preserved and what changed for V1 records.
+- `notes/010_run_prompt_candidate_review__20260415_234559.md`
+  Reason: captures the first serious assessment of the new runner and the environment-vs-runner distinction.
+- `notes/011_review_writeback_helper_build__20260415_235514.md`
+  Reason: implementation and validation note for the still-live `review_run.py` helper.
+- `notes/012_v1_pipeline_options_review__20260416_000819.md`
+  Reason: useful maturity review and bounded option set before operational helpers were added.
+- `notes/014_queue_readiness_checker_build__20260416_002419.md`
+  Reason: implementation and validation note for the still-live readiness helper.
+- `notes/015_review_backlog_lister_build__20260416_010500.md`
+  Reason: implementation and validation note for the still-live review backlog helper.
+- `notes/016_queue_and_backlog_helper_validation__20260416_003710.md`
+  Reason: repo-specific validation evidence showing how readiness and backlog behavior interact with mixed legacy and V1 notes.
+- `notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md`
+  Reason: records the rationale for the current `Queue note:` behavior in the readiness helper.
+
+## Move To Attic Without Summary
+
+### notes/
+
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md`
+  Reason: legacy smoke-test receipt; superseded by later V1 records and duplicates.
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md`
+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md`
+  Reason: duplicate legacy smoke-test receipt with no new durable decision.
+- `notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md`
+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
+- `notes/003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md`
+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
+- `notes/004_next_design_step_recommendation__20260415_195538.md`
+  Reason: brief recommendation fully subsumed by the main review and later execution-record work.
+- `notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md`
+  Reason: success receipt; the main review note is the durable artifact.
+- `notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md`
+  Reason: success receipt; resulting doc and queue-plan note carry the lasting value.
+- `notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md`
+  Reason: success receipt; durable knowledge lives in `tools/codex/V1_Run_Review_Gate.md`.
+- `notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md`
+  Reason: success receipt; durable knowledge lives in the architecture doc.
+- `notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md`
+  Reason: success receipt; durable knowledge lives in the bridge-runner spec.
+- `notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md`
+  Reason: implementation receipt; the build note and the code itself carry the durable content.
+- `notes/010_next_step_recommendation__20260415_234559.md`
+  Reason: brief next-step recommendation that was quickly superseded by the actual sequence taken.
+- `notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md`
+  Reason: success receipt; the review note carries the useful content.
+- `notes/012_top_three_next_options__20260416_000819.md`
+  Reason: short ranking note largely duplicated by the fuller options review and later implementation history.
+- `notes/013_next_two_prompt_plan__20260416_002005.md`
+  Reason: prompt-generation plan already realized by `014` and `015`.
+- `notes/016_next_improvement_recommendation__20260416_003710.md`
+  Reason: one-line recommendation already realized by `017`.
+- `notes/018_prioritized_remaining_work__20260416_005130.md`
+  Reason: concise ranking duplicated by the fuller architecture sweep and the implemented follow-up work.
+- `notes/019_next_two_cleanup_prompts_plan__20260416_010251.md`
+  Reason: prompt-generation plan already realized by `020` and `021`.
+
+### codex_prompts/
+
+- `codex_prompts/001_smoke_test_pipeline.md`
+  Reason: one-use bootstrap prompt; durable outcome was only the existence of a working notes path.
+- `codex_prompts/002_repo_inventory_and_status.md`
+  Reason: one-use repo-inventory scaffold; durable value is in the resulting note, not the prompt text.
+- `codex_prompts/003_revise_Project_Deisgn_workflow_document.md`
+  Reason: implementation instruction for a completed doc revision; resulting document and revision note now carry the knowledge.
+- `codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md`
+  Reason: review scaffold whose durable output is the architecture/runner assessment note.
+- `codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md`
+  Reason: queue-generation scaffold; durable outputs are the execution-record doc and queue-plan note.
+- `codex_prompts/006_define_v1_run_review_gate.md`
+  Reason: design scaffold whose durable output is `tools/codex/V1_Run_Review_Gate.md`.
+- `codex_prompts/007_align_architecture_doc_to_v1_workflow.md`
+  Reason: alignment scaffold whose durable output is the revised architecture doc.
+- `codex_prompts/008_define_minimal_bridge_runner_change_spec.md`
+  Reason: spec-writing scaffold whose durable output is the bridge-runner spec.
+- `codex_prompts/009_build_v1_run_prompt_candidate.md`
+  Reason: implementation prompt for completed runner work; code and build note now carry the value.
+- `codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md`
+  Reason: review scaffold whose durable output is the runner-review note.
+- `codex_prompts/011_build_v1_review_writeback_helper.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md`
+  Reason: review scaffold whose durable output is the options review note.
+- `codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md`
+  Reason: prompt-generation scaffold already realized by later prompt files and notes.
+- `codex_prompts/014_build_queue_readiness_checker.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/015_build_review_backlog_unreviewed_run_lister.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md`
+  Reason: validation scaffold whose durable content is in the validation note.
+- `codex_prompts/017_polish_queue_readiness_gap_explanation.md`
+  Reason: small polish prompt for completed helper behavior; code and polish note now carry the value.
+- `codex_prompts/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work.md`
+  Reason: review scaffold whose durable output is the architecture sweep note.
+- `codex_prompts/019_generate_prompts_for_doc_alignment_and_record_contract_validation.md`
+  Reason: prompt-generation scaffold already realized by `020` and `021`.
+- `codex_prompts/020_align_v1_doc_and_spec_packet_to_actual_toolset.md`
+  Reason: implementation prompt for completed doc cleanup; the edited docs and cleanup note now carry the value.
+- `codex_prompts/021_add_lightweight_v1_record_contract_validation.md`
+  Reason: implementation prompt for completed validator work; code and validation note now carry the value.
+
+## Uncertain / Needs Human Review
+
+### notes/
+
+- `notes/001_smoke_test_pipeline__20260415_233343.md`
+  Reason: current V1 execution record; moving it changes live run history and helper output.
+- `notes/001_smoke_test_pipeline__20260415_234918.md`
+  Reason: current V1 execution record and current accepted prior evidence for prompt `001`.
+- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/014_build_queue_readiness_checker__20260416_002319.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work__20260416_005034.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/020_align_v1_doc_and_spec_packet_to_actual_toolset__20260416_010453.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/021_add_lightweight_v1_record_contract_validation__20260416_010810.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+
+### codex_prompts/
+
+- `codex_prompts/022_audit_and_classify_scaffolding_for_summary_and_attic.md`
+  Reason: current audit prompt for the cleanup sequence; do not classify as attic material until this report has been acted on and superseded.
+
+## Practical Read Of The Repo
+
+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
+- The second cleanup target is the durable-but-noisy implementation notes, but only after their key rationale is extracted into a smaller standing summary.
+- The current V1 execution records should not move in a cleanup pass unless the repo first decides how the readiness/backlog tools should treat archived records.

apply patch
patch: completed
/data/git/handson-ml3/notes/022_scaffolding_cleanup_plan__20260416_023016.md
diff --git a/notes/022_scaffolding_classification_report__20260416_023016.md b/notes/022_scaffolding_classification_report__20260416_023016.md
new file mode 100644
index 0000000000000000000000000000000000000000..ff6b5ff2902c910c2b6d5b21196592666fef35c0
--- /dev/null
+++ b/notes/022_scaffolding_classification_report__20260416_023016.md
@@ -0,0 +1,189 @@
+# 022 Scaffolding Classification Report
+
+## Short Summary
+
+`notes/` currently mixes four different things:
+
+- legacy pre-V1 success receipts
+- durable design and implementation notes
+- current V1 execution records that the live helpers still read
+- small planning / recommendation notes that were useful during construction but are now mostly superseded
+
+`codex_prompts/` is almost entirely construction scaffolding for building the current V1 toolset. The durable knowledge from those prompts now lives in the tool/docs packet and in a smaller set of implementation and review notes.
+
+The main conservative rule in this report is:
+
+- current V1 execution-record files stay out of attic decisions for now, because moving them would change the behavior of `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py`
+
+## Classification Criteria Used
+
+- `permanent residents`: still useful as standing maintainer context in their current form
+- `summarize, then move to attic`: contains durable decisions, validation evidence, or design rationale, but does not need to stay in `notes/` once that knowledge is extracted
+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
+- `uncertain / needs human review`: not safe to move yet because it is still operationally active, still part of the current source-of-truth set, or still too close to the active cleanup pass
+
+## Permanent Residents
+
+- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
+  Reason: best compact repo-specific statement of what the V1 system currently is, what remains missing, and whether it is ready for real use.
+- `notes/020_doc_spec_alignment_cleanup__20260416_010534.md`
+  Reason: explains why the current design/spec packet changed and records the final doc-alignment decisions that still govern the written packet.
+- `notes/021_record_contract_validation__20260416_011314.md`
+  Reason: durable contract and validation evidence for the shared V1 execution-record parser used by live tools.
+
+## Summarize, Then Move To Attic
+
+- `notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md`
+  Reason: useful initial repo-context snapshot, but stale as a standing note.
+- `notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md`
+  Reason: explains why `tools/Project_Design_Workflow.md` was revised and what reusable principles were added.
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+  Reason: foundational mismatch review that drove the V1 execution-record and review-gate design.
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+  Reason: captures why the early V1 sequence was ordered policy first, architecture second, runner change last.
+- `notes/009_run_prompt_candidate_build__20260415_233407.md`
+  Reason: records what behavior `run_prompt.py` intentionally preserved and what changed for V1 records.
+- `notes/010_run_prompt_candidate_review__20260415_234559.md`
+  Reason: captures the first serious assessment of the new runner and the environment-vs-runner distinction.
+- `notes/011_review_writeback_helper_build__20260415_235514.md`
+  Reason: implementation and validation note for the still-live `review_run.py` helper.
+- `notes/012_v1_pipeline_options_review__20260416_000819.md`
+  Reason: useful maturity review and bounded option set before operational helpers were added.
+- `notes/014_queue_readiness_checker_build__20260416_002419.md`
+  Reason: implementation and validation note for the still-live readiness helper.
+- `notes/015_review_backlog_lister_build__20260416_010500.md`
+  Reason: implementation and validation note for the still-live review backlog helper.
+- `notes/016_queue_and_backlog_helper_validation__20260416_003710.md`
+  Reason: repo-specific validation evidence showing how readiness and backlog behavior interact with mixed legacy and V1 notes.
+- `notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md`
+  Reason: records the rationale for the current `Queue note:` behavior in the readiness helper.
+
+## Move To Attic Without Summary
+
+### notes/
+
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md`
+  Reason: legacy smoke-test receipt; superseded by later V1 records and duplicates.
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md`
+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md`
+  Reason: duplicate legacy smoke-test receipt with no new durable decision.
+- `notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md`
+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
+- `notes/003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md`
+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
+- `notes/004_next_design_step_recommendation__20260415_195538.md`
+  Reason: brief recommendation fully subsumed by the main review and later execution-record work.
+- `notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md`
+  Reason: success receipt; the main review note is the durable artifact.
+- `notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md`
+  Reason: success receipt; resulting doc and queue-plan note carry the lasting value.
+- `notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md`
+  Reason: success receipt; durable knowledge lives in `tools/codex/V1_Run_Review_Gate.md`.
+- `notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md`
+  Reason: success receipt; durable knowledge lives in the architecture doc.
+- `notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md`
+  Reason: success receipt; durable knowledge lives in the bridge-runner spec.
+- `notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md`
+  Reason: implementation receipt; the build note and the code itself carry the durable content.
+- `notes/010_next_step_recommendation__20260415_234559.md`
+  Reason: brief next-step recommendation that was quickly superseded by the actual sequence taken.
+- `notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md`
+  Reason: success receipt; the review note carries the useful content.
+- `notes/012_top_three_next_options__20260416_000819.md`
+  Reason: short ranking note largely duplicated by the fuller options review and later implementation history.
+- `notes/013_next_two_prompt_plan__20260416_002005.md`
+  Reason: prompt-generation plan already realized by `014` and `015`.
+- `notes/016_next_improvement_recommendation__20260416_003710.md`
+  Reason: one-line recommendation already realized by `017`.
+- `notes/018_prioritized_remaining_work__20260416_005130.md`
+  Reason: concise ranking duplicated by the fuller architecture sweep and the implemented follow-up work.
+- `notes/019_next_two_cleanup_prompts_plan__20260416_010251.md`
+  Reason: prompt-generation plan already realized by `020` and `021`.
+
+### codex_prompts/
+
+- `codex_prompts/001_smoke_test_pipeline.md`
+  Reason: one-use bootstrap prompt; durable outcome was only the existence of a working notes path.
+- `codex_prompts/002_repo_inventory_and_status.md`
+  Reason: one-use repo-inventory scaffold; durable value is in the resulting note, not the prompt text.
+- `codex_prompts/003_revise_Project_Deisgn_workflow_document.md`
+  Reason: implementation instruction for a completed doc revision; resulting document and revision note now carry the knowledge.
+- `codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md`
+  Reason: review scaffold whose durable output is the architecture/runner assessment note.
+- `codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md`
+  Reason: queue-generation scaffold; durable outputs are the execution-record doc and queue-plan note.
+- `codex_prompts/006_define_v1_run_review_gate.md`
+  Reason: design scaffold whose durable output is `tools/codex/V1_Run_Review_Gate.md`.
+- `codex_prompts/007_align_architecture_doc_to_v1_workflow.md`
+  Reason: alignment scaffold whose durable output is the revised architecture doc.
+- `codex_prompts/008_define_minimal_bridge_runner_change_spec.md`
+  Reason: spec-writing scaffold whose durable output is the bridge-runner spec.
+- `codex_prompts/009_build_v1_run_prompt_candidate.md`
+  Reason: implementation prompt for completed runner work; code and build note now carry the value.
+- `codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md`
+  Reason: review scaffold whose durable output is the runner-review note.
+- `codex_prompts/011_build_v1_review_writeback_helper.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md`
+  Reason: review scaffold whose durable output is the options review note.
+- `codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md`
+  Reason: prompt-generation scaffold already realized by later prompt files and notes.
+- `codex_prompts/014_build_queue_readiness_checker.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/015_build_review_backlog_unreviewed_run_lister.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md`
+  Reason: validation scaffold whose durable content is in the validation note.
+- `codex_prompts/017_polish_queue_readiness_gap_explanation.md`
+  Reason: small polish prompt for completed helper behavior; code and polish note now carry the value.
+- `codex_prompts/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work.md`
+  Reason: review scaffold whose durable output is the architecture sweep note.
+- `codex_prompts/019_generate_prompts_for_doc_alignment_and_record_contract_validation.md`
+  Reason: prompt-generation scaffold already realized by `020` and `021`.
+- `codex_prompts/020_align_v1_doc_and_spec_packet_to_actual_toolset.md`
+  Reason: implementation prompt for completed doc cleanup; the edited docs and cleanup note now carry the value.
+- `codex_prompts/021_add_lightweight_v1_record_contract_validation.md`
+  Reason: implementation prompt for completed validator work; code and validation note now carry the value.
+
+## Uncertain / Needs Human Review
+
+### notes/
+
+- `notes/001_smoke_test_pipeline__20260415_233343.md`
+  Reason: current V1 execution record; moving it changes live run history and helper output.
+- `notes/001_smoke_test_pipeline__20260415_234918.md`
+  Reason: current V1 execution record and current accepted prior evidence for prompt `001`.
+- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/014_build_queue_readiness_checker__20260416_002319.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work__20260416_005034.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/020_align_v1_doc_and_spec_packet_to_actual_toolset__20260416_010453.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/021_add_lightweight_v1_record_contract_validation__20260416_010810.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+
+### codex_prompts/
+
+- `codex_prompts/022_audit_and_classify_scaffolding_for_summary_and_attic.md`
+  Reason: current audit prompt for the cleanup sequence; do not classify as attic material until this report has been acted on and superseded.
+
+## Practical Read Of The Repo
+
+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
+- The second cleanup target is the durable-but-noisy implementation notes, but only after their key rationale is extracted into a smaller standing summary.
+- The current V1 execution records should not move in a cleanup pass unless the repo first decides how the readiness/backlog tools should treat archived records.
diff --git a/notes/022_scaffolding_cleanup_plan__20260416_023016.md b/notes/022_scaffolding_cleanup_plan__20260416_023016.md
new file mode 100644
index 0000000000000000000000000000000000000000..f6140b3bc4f4031dae4d9e337672d4f3d2ecae99
--- /dev/null
+++ b/notes/022_scaffolding_cleanup_plan__20260416_023016.md
@@ -0,0 +1,20 @@
+# 022 Scaffolding Cleanup Plan
+
+## Next Pass
+
+- Extract a compact durable summary from the files marked `summarize, then move to attic`.
+- Keep that summary in a small standing note or note packet before any moves happen.
+- Do a separate policy check on current V1 execution-record files in `notes/`; do not move them unless the repo first decides how archived records should affect readiness and backlog helpers.
+
+## After That
+
+- Move the `attic without summary` set in small batches.
+- Move the `summarize, then move to attic` set only after the summary note exists and has been reviewed.
+- Re-run a quick repo sanity check so the remaining `notes/` surface is mostly current operational notes plus live V1 records.
+
+## Do Not Do Yet
+
+- Do not create the attic in this pass.
+- Do not move or delete current V1 execution-record files.
+- Do not rewrite tools or major design docs as part of cleanup.
+- Do not delete prompt files or notes before the summary-extraction step is reviewed.

diff --git a/notes/022_scaffolding_classification_report__20260416_023016.md b/notes/022_scaffolding_classification_report__20260416_023016.md
new file mode 100644
index 0000000000000000000000000000000000000000..ff6b5ff2902c910c2b6d5b21196592666fef35c0
--- /dev/null
+++ b/notes/022_scaffolding_classification_report__20260416_023016.md
@@ -0,0 +1,189 @@
+# 022 Scaffolding Classification Report
+
+## Short Summary
+
+`notes/` currently mixes four different things:
+
+- legacy pre-V1 success receipts
+- durable design and implementation notes
+- current V1 execution records that the live helpers still read
+- small planning / recommendation notes that were useful during construction but are now mostly superseded
+
+`codex_prompts/` is almost entirely construction scaffolding for building the current V1 toolset. The durable knowledge from those prompts now lives in the tool/docs packet and in a smaller set of implementation and review notes.
+
+The main conservative rule in this report is:
+
+- current V1 execution-record files stay out of attic decisions for now, because moving them would change the behavior of `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py`
+
+## Classification Criteria Used
+
+- `permanent residents`: still useful as standing maintainer context in their current form
+- `summarize, then move to attic`: contains durable decisions, validation evidence, or design rationale, but does not need to stay in `notes/` once that knowledge is extracted
+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
+- `uncertain / needs human review`: not safe to move yet because it is still operationally active, still part of the current source-of-truth set, or still too close to the active cleanup pass
+
+## Permanent Residents
+
+- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
+  Reason: best compact repo-specific statement of what the V1 system currently is, what remains missing, and whether it is ready for real use.
+- `notes/020_doc_spec_alignment_cleanup__20260416_010534.md`
+  Reason: explains why the current design/spec packet changed and records the final doc-alignment decisions that still govern the written packet.
+- `notes/021_record_contract_validation__20260416_011314.md`
+  Reason: durable contract and validation evidence for the shared V1 execution-record parser used by live tools.
+
+## Summarize, Then Move To Attic
+
+- `notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md`
+  Reason: useful initial repo-context snapshot, but stale as a standing note.
+- `notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md`
+  Reason: explains why `tools/Project_Design_Workflow.md` was revised and what reusable principles were added.
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+  Reason: foundational mismatch review that drove the V1 execution-record and review-gate design.
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+  Reason: captures why the early V1 sequence was ordered policy first, architecture second, runner change last.
+- `notes/009_run_prompt_candidate_build__20260415_233407.md`
+  Reason: records what behavior `run_prompt.py` intentionally preserved and what changed for V1 records.
+- `notes/010_run_prompt_candidate_review__20260415_234559.md`
+  Reason: captures the first serious assessment of the new runner and the environment-vs-runner distinction.
+- `notes/011_review_writeback_helper_build__20260415_235514.md`
+  Reason: implementation and validation note for the still-live `review_run.py` helper.
+- `notes/012_v1_pipeline_options_review__20260416_000819.md`
+  Reason: useful maturity review and bounded option set before operational helpers were added.
+- `notes/014_queue_readiness_checker_build__20260416_002419.md`
+  Reason: implementation and validation note for the still-live readiness helper.
+- `notes/015_review_backlog_lister_build__20260416_010500.md`
+  Reason: implementation and validation note for the still-live review backlog helper.
+- `notes/016_queue_and_backlog_helper_validation__20260416_003710.md`
+  Reason: repo-specific validation evidence showing how readiness and backlog behavior interact with mixed legacy and V1 notes.
+- `notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md`
+  Reason: records the rationale for the current `Queue note:` behavior in the readiness helper.
+
+## Move To Attic Without Summary
+
+### notes/
+
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md`
+  Reason: legacy smoke-test receipt; superseded by later V1 records and duplicates.
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md`
+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md`
+  Reason: duplicate legacy smoke-test receipt with no new durable decision.
+- `notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md`
+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
+- `notes/003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md`
+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
+- `notes/004_next_design_step_recommendation__20260415_195538.md`
+  Reason: brief recommendation fully subsumed by the main review and later execution-record work.
+- `notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md`
+  Reason: success receipt; the main review note is the durable artifact.
+- `notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md`
+  Reason: success receipt; resulting doc and queue-plan note carry the lasting value.
+- `notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md`
+  Reason: success receipt; durable knowledge lives in `tools/codex/V1_Run_Review_Gate.md`.
+- `notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md`
+  Reason: success receipt; durable knowledge lives in the architecture doc.
+- `notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md`
+  Reason: success receipt; durable knowledge lives in the bridge-runner spec.
+- `notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md`
+  Reason: implementation receipt; the build note and the code itself carry the durable content.
+- `notes/010_next_step_recommendation__20260415_234559.md`
+  Reason: brief next-step recommendation that was quickly superseded by the actual sequence taken.
+- `notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md`
+  Reason: success receipt; the review note carries the useful content.
+- `notes/012_top_three_next_options__20260416_000819.md`
+  Reason: short ranking note largely duplicated by the fuller options review and later implementation history.
+- `notes/013_next_two_prompt_plan__20260416_002005.md`
+  Reason: prompt-generation plan already realized by `014` and `015`.
+- `notes/016_next_improvement_recommendation__20260416_003710.md`
+  Reason: one-line recommendation already realized by `017`.
+- `notes/018_prioritized_remaining_work__20260416_005130.md`
+  Reason: concise ranking duplicated by the fuller architecture sweep and the implemented follow-up work.
+- `notes/019_next_two_cleanup_prompts_plan__20260416_010251.md`
+  Reason: prompt-generation plan already realized by `020` and `021`.
+
+### codex_prompts/
+
+- `codex_prompts/001_smoke_test_pipeline.md`
+  Reason: one-use bootstrap prompt; durable outcome was only the existence of a working notes path.
+- `codex_prompts/002_repo_inventory_and_status.md`
+  Reason: one-use repo-inventory scaffold; durable value is in the resulting note, not the prompt text.
+- `codex_prompts/003_revise_Project_Deisgn_workflow_document.md`
+  Reason: implementation instruction for a completed doc revision; resulting document and revision note now carry the knowledge.
+- `codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md`
+  Reason: review scaffold whose durable output is the architecture/runner assessment note.
+- `codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md`
+  Reason: queue-generation scaffold; durable outputs are the execution-record doc and queue-plan note.
+- `codex_prompts/006_define_v1_run_review_gate.md`
+  Reason: design scaffold whose durable output is `tools/codex/V1_Run_Review_Gate.md`.
+- `codex_prompts/007_align_architecture_doc_to_v1_workflow.md`
+  Reason: alignment scaffold whose durable output is the revised architecture doc.
+- `codex_prompts/008_define_minimal_bridge_runner_change_spec.md`
+  Reason: spec-writing scaffold whose durable output is the bridge-runner spec.
+- `codex_prompts/009_build_v1_run_prompt_candidate.md`
+  Reason: implementation prompt for completed runner work; code and build note now carry the value.
+- `codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md`
+  Reason: review scaffold whose durable output is the runner-review note.
+- `codex_prompts/011_build_v1_review_writeback_helper.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md`
+  Reason: review scaffold whose durable output is the options review note.
+- `codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md`
+  Reason: prompt-generation scaffold already realized by later prompt files and notes.
+- `codex_prompts/014_build_queue_readiness_checker.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/015_build_review_backlog_unreviewed_run_lister.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md`
+  Reason: validation scaffold whose durable content is in the validation note.
+- `codex_prompts/017_polish_queue_readiness_gap_explanation.md`
+  Reason: small polish prompt for completed helper behavior; code and polish note now carry the value.
+- `codex_prompts/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work.md`
+  Reason: review scaffold whose durable output is the architecture sweep note.
+- `codex_prompts/019_generate_prompts_for_doc_alignment_and_record_contract_validation.md`
+  Reason: prompt-generation scaffold already realized by `020` and `021`.
+- `codex_prompts/020_align_v1_doc_and_spec_packet_to_actual_toolset.md`
+  Reason: implementation prompt for completed doc cleanup; the edited docs and cleanup note now carry the value.
+- `codex_prompts/021_add_lightweight_v1_record_contract_validation.md`
+  Reason: implementation prompt for completed validator work; code and validation note now carry the value.
+
+## Uncertain / Needs Human Review
+
+### notes/
+
+- `notes/001_smoke_test_pipeline__20260415_233343.md`
+  Reason: current V1 execution record; moving it changes live run history and helper output.
+- `notes/001_smoke_test_pipeline__20260415_234918.md`
+  Reason: current V1 execution record and current accepted prior evidence for prompt `001`.
+- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/014_build_queue_readiness_checker__20260416_002319.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work__20260416_005034.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/020_align_v1_doc_and_spec_packet_to_actual_toolset__20260416_010453.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/021_add_lightweight_v1_record_contract_validation__20260416_010810.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+
+### codex_prompts/
+
+- `codex_prompts/022_audit_and_classify_scaffolding_for_summary_and_attic.md`
+  Reason: current audit prompt for the cleanup sequence; do not classify as attic material until this report has been acted on and superseded.
+
+## Practical Read Of The Repo
+
+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
+- The second cleanup target is the durable-but-noisy implementation notes, but only after their key rationale is extracted into a smaller standing summary.
+- The current V1 execution records should not move in a cleanup pass unless the repo first decides how the readiness/backlog tools should treat archived records.
diff --git a/notes/022_scaffolding_cleanup_plan__20260416_023016.md b/notes/022_scaffolding_cleanup_plan__20260416_023016.md
new file mode 100644
index 0000000000000000000000000000000000000000..f6140b3bc4f4031dae4d9e337672d4f3d2ecae99
--- /dev/null
+++ b/notes/022_scaffolding_cleanup_plan__20260416_023016.md
@@ -0,0 +1,20 @@
+# 022 Scaffolding Cleanup Plan
+
+## Next Pass
+
+- Extract a compact durable summary from the files marked `summarize, then move to attic`.
+- Keep that summary in a small standing note or note packet before any moves happen.
+- Do a separate policy check on current V1 execution-record files in `notes/`; do not move them unless the repo first decides how archived records should affect readiness and backlog helpers.
+
+## After That
+
+- Move the `attic without summary` set in small batches.
+- Move the `summarize, then move to attic` set only after the summary note exists and has been reviewed.
+- Re-run a quick repo sanity check so the remaining `notes/` surface is mostly current operational notes plus live V1 records.
+
+## Do Not Do Yet
+
+- Do not create the attic in this pass.
+- Do not move or delete current V1 execution-record files.
+- Do not rewrite tools or major design docs as part of cleanup.
+- Do not delete prompt files or notes before the summary-extraction step is reviewed.

codex
The two notes are in place. I’m doing one quick verification on the filenames and contents, then I’ll hand back the result.
exec
/bin/bash -lc 'ls -1 notes/022_scaffolding_classification_report__20260416_023016.md notes/022_scaffolding_cleanup_plan__20260416_023016.md' in /data/git/handson-ml3
 succeeded in 0ms:
notes/022_scaffolding_classification_report__20260416_023016.md
notes/022_scaffolding_cleanup_plan__20260416_023016.md

exec
/bin/bash -lc "sed -n '1,220p' notes/022_scaffolding_classification_report__20260416_023016.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 022 Scaffolding Classification Report

## Short Summary

`notes/` currently mixes four different things:

- legacy pre-V1 success receipts
- durable design and implementation notes
- current V1 execution records that the live helpers still read
- small planning / recommendation notes that were useful during construction but are now mostly superseded

`codex_prompts/` is almost entirely construction scaffolding for building the current V1 toolset. The durable knowledge from those prompts now lives in the tool/docs packet and in a smaller set of implementation and review notes.

The main conservative rule in this report is:

- current V1 execution-record files stay out of attic decisions for now, because moving them would change the behavior of `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py`

## Classification Criteria Used

- `permanent residents`: still useful as standing maintainer context in their current form
- `summarize, then move to attic`: contains durable decisions, validation evidence, or design rationale, but does not need to stay in `notes/` once that knowledge is extracted
- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
- `uncertain / needs human review`: not safe to move yet because it is still operationally active, still part of the current source-of-truth set, or still too close to the active cleanup pass

## Permanent Residents

- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
  Reason: best compact repo-specific statement of what the V1 system currently is, what remains missing, and whether it is ready for real use.
- `notes/020_doc_spec_alignment_cleanup__20260416_010534.md`
  Reason: explains why the current design/spec packet changed and records the final doc-alignment decisions that still govern the written packet.
- `notes/021_record_contract_validation__20260416_011314.md`
  Reason: durable contract and validation evidence for the shared V1 execution-record parser used by live tools.

## Summarize, Then Move To Attic

- `notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md`
  Reason: useful initial repo-context snapshot, but stale as a standing note.
- `notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md`
  Reason: explains why `tools/Project_Design_Workflow.md` was revised and what reusable principles were added.
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
  Reason: foundational mismatch review that drove the V1 execution-record and review-gate design.
- `notes/005_prompt_queue_plan__20260415_202557.md`
  Reason: captures why the early V1 sequence was ordered policy first, architecture second, runner change last.
- `notes/009_run_prompt_candidate_build__20260415_233407.md`
  Reason: records what behavior `run_prompt.py` intentionally preserved and what changed for V1 records.
- `notes/010_run_prompt_candidate_review__20260415_234559.md`
  Reason: captures the first serious assessment of the new runner and the environment-vs-runner distinction.
- `notes/011_review_writeback_helper_build__20260415_235514.md`
  Reason: implementation and validation note for the still-live `review_run.py` helper.
- `notes/012_v1_pipeline_options_review__20260416_000819.md`
  Reason: useful maturity review and bounded option set before operational helpers were added.
- `notes/014_queue_readiness_checker_build__20260416_002419.md`
  Reason: implementation and validation note for the still-live readiness helper.
- `notes/015_review_backlog_lister_build__20260416_010500.md`
  Reason: implementation and validation note for the still-live review backlog helper.
- `notes/016_queue_and_backlog_helper_validation__20260416_003710.md`
  Reason: repo-specific validation evidence showing how readiness and backlog behavior interact with mixed legacy and V1 notes.
- `notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md`
  Reason: records the rationale for the current `Queue note:` behavior in the readiness helper.

## Move To Attic Without Summary

### notes/

- `notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md`
  Reason: legacy smoke-test receipt; superseded by later V1 records and duplicates.
- `notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md`
  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
- `notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md`
  Reason: duplicate legacy smoke-test receipt with no new durable decision.
- `notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md`
  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
- `notes/003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md`
  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
- `notes/004_next_design_step_recommendation__20260415_195538.md`
  Reason: brief recommendation fully subsumed by the main review and later execution-record work.
- `notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md`
  Reason: success receipt; the main review note is the durable artifact.
- `notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md`
  Reason: success receipt; resulting doc and queue-plan note carry the lasting value.
- `notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md`
  Reason: success receipt; durable knowledge lives in `tools/codex/V1_Run_Review_Gate.md`.
- `notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md`
  Reason: success receipt; durable knowledge lives in the architecture doc.
- `notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md`
  Reason: success receipt; durable knowledge lives in the bridge-runner spec.
- `notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md`
  Reason: implementation receipt; the build note and the code itself carry the durable content.
- `notes/010_next_step_recommendation__20260415_234559.md`
  Reason: brief next-step recommendation that was quickly superseded by the actual sequence taken.
- `notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md`
  Reason: success receipt; the review note carries the useful content.
- `notes/012_top_three_next_options__20260416_000819.md`
  Reason: short ranking note largely duplicated by the fuller options review and later implementation history.
- `notes/013_next_two_prompt_plan__20260416_002005.md`
  Reason: prompt-generation plan already realized by `014` and `015`.
- `notes/016_next_improvement_recommendation__20260416_003710.md`
  Reason: one-line recommendation already realized by `017`.
- `notes/018_prioritized_remaining_work__20260416_005130.md`
  Reason: concise ranking duplicated by the fuller architecture sweep and the implemented follow-up work.
- `notes/019_next_two_cleanup_prompts_plan__20260416_010251.md`
  Reason: prompt-generation plan already realized by `020` and `021`.

### codex_prompts/

- `codex_prompts/001_smoke_test_pipeline.md`
  Reason: one-use bootstrap prompt; durable outcome was only the existence of a working notes path.
- `codex_prompts/002_repo_inventory_and_status.md`
  Reason: one-use repo-inventory scaffold; durable value is in the resulting note, not the prompt text.
- `codex_prompts/003_revise_Project_Deisgn_workflow_document.md`
  Reason: implementation instruction for a completed doc revision; resulting document and revision note now carry the knowledge.
- `codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md`
  Reason: review scaffold whose durable output is the architecture/runner assessment note.
- `codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md`
  Reason: queue-generation scaffold; durable outputs are the execution-record doc and queue-plan note.
- `codex_prompts/006_define_v1_run_review_gate.md`
  Reason: design scaffold whose durable output is `tools/codex/V1_Run_Review_Gate.md`.
- `codex_prompts/007_align_architecture_doc_to_v1_workflow.md`
  Reason: alignment scaffold whose durable output is the revised architecture doc.
- `codex_prompts/008_define_minimal_bridge_runner_change_spec.md`
  Reason: spec-writing scaffold whose durable output is the bridge-runner spec.
- `codex_prompts/009_build_v1_run_prompt_candidate.md`
  Reason: implementation prompt for completed runner work; code and build note now carry the value.
- `codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md`
  Reason: review scaffold whose durable output is the runner-review note.
- `codex_prompts/011_build_v1_review_writeback_helper.md`
  Reason: implementation prompt for completed helper work; code and build note now carry the value.
- `codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md`
  Reason: review scaffold whose durable output is the options review note.
- `codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md`
  Reason: prompt-generation scaffold already realized by later prompt files and notes.
- `codex_prompts/014_build_queue_readiness_checker.md`
  Reason: implementation prompt for completed helper work; code and build note now carry the value.
- `codex_prompts/015_build_review_backlog_unreviewed_run_lister.md`
  Reason: implementation prompt for completed helper work; code and build note now carry the value.
- `codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md`
  Reason: validation scaffold whose durable content is in the validation note.
- `codex_prompts/017_polish_queue_readiness_gap_explanation.md`
  Reason: small polish prompt for completed helper behavior; code and polish note now carry the value.
- `codex_prompts/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work.md`
  Reason: review scaffold whose durable output is the architecture sweep note.
- `codex_prompts/019_generate_prompts_for_doc_alignment_and_record_contract_validation.md`
  Reason: prompt-generation scaffold already realized by `020` and `021`.
- `codex_prompts/020_align_v1_doc_and_spec_packet_to_actual_toolset.md`
  Reason: implementation prompt for completed doc cleanup; the edited docs and cleanup note now carry the value.
- `codex_prompts/021_add_lightweight_v1_record_contract_validation.md`
  Reason: implementation prompt for completed validator work; code and validation note now carry the value.

## Uncertain / Needs Human Review

### notes/

- `notes/001_smoke_test_pipeline__20260415_233343.md`
  Reason: current V1 execution record; moving it changes live run history and helper output.
- `notes/001_smoke_test_pipeline__20260415_234918.md`
  Reason: current V1 execution record and current accepted prior evidence for prompt `001`.
- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/014_build_queue_readiness_checker__20260416_002319.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work__20260416_005034.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/020_align_v1_doc_and_spec_packet_to_actual_toolset__20260416_010453.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/021_add_lightweight_v1_record_contract_validation__20260416_010810.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.

### codex_prompts/

- `codex_prompts/022_audit_and_classify_scaffolding_for_summary_and_attic.md`
  Reason: current audit prompt for the cleanup sequence; do not classify as attic material until this report has been acted on and superseded.

## Practical Read Of The Repo

- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
- The second cleanup target is the durable-but-noisy implementation notes, but only after their key rationale is extracted into a smaller standing summary.
- The current V1 execution records should not move in a cleanup pass unless the repo first decides how the readiness/backlog tools should treat archived records.

exec
/bin/bash -lc "sed -n '1,120p' notes/022_scaffolding_cleanup_plan__20260416_023016.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 022 Scaffolding Cleanup Plan

## Next Pass

- Extract a compact durable summary from the files marked `summarize, then move to attic`.
- Keep that summary in a small standing note or note packet before any moves happen.
- Do a separate policy check on current V1 execution-record files in `notes/`; do not move them unless the repo first decides how archived records should affect readiness and backlog helpers.

## After That

- Move the `attic without summary` set in small batches.
- Move the `summarize, then move to attic` set only after the summary note exists and has been reviewed.
- Re-run a quick repo sanity check so the remaining `notes/` surface is mostly current operational notes plus live V1 records.

## Do Not Do Yet

- Do not create the attic in this pass.
- Do not move or delete current V1 execution-record files.
- Do not rewrite tools or major design docs as part of cleanup.
- Do not delete prompt files or notes before the summary-extraction step is reviewed.

diff --git a/notes/022_scaffolding_classification_report__20260416_023016.md b/notes/022_scaffolding_classification_report__20260416_023016.md
new file mode 100644
index 0000000000000000000000000000000000000000..ff6b5ff2902c910c2b6d5b21196592666fef35c0
--- /dev/null
+++ b/notes/022_scaffolding_classification_report__20260416_023016.md
@@ -0,0 +1,189 @@
+# 022 Scaffolding Classification Report
+
+## Short Summary
+
+`notes/` currently mixes four different things:
+
+- legacy pre-V1 success receipts
+- durable design and implementation notes
+- current V1 execution records that the live helpers still read
+- small planning / recommendation notes that were useful during construction but are now mostly superseded
+
+`codex_prompts/` is almost entirely construction scaffolding for building the current V1 toolset. The durable knowledge from those prompts now lives in the tool/docs packet and in a smaller set of implementation and review notes.
+
+The main conservative rule in this report is:
+
+- current V1 execution-record files stay out of attic decisions for now, because moving them would change the behavior of `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py`
+
+## Classification Criteria Used
+
+- `permanent residents`: still useful as standing maintainer context in their current form
+- `summarize, then move to attic`: contains durable decisions, validation evidence, or design rationale, but does not need to stay in `notes/` once that knowledge is extracted
+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
+- `uncertain / needs human review`: not safe to move yet because it is still operationally active, still part of the current source-of-truth set, or still too close to the active cleanup pass
+
+## Permanent Residents
+
+- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
+  Reason: best compact repo-specific statement of what the V1 system currently is, what remains missing, and whether it is ready for real use.
+- `notes/020_doc_spec_alignment_cleanup__20260416_010534.md`
+  Reason: explains why the current design/spec packet changed and records the final doc-alignment decisions that still govern the written packet.
+- `notes/021_record_contract_validation__20260416_011314.md`
+  Reason: durable contract and validation evidence for the shared V1 execution-record parser used by live tools.
+
+## Summarize, Then Move To Attic
+
+- `notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md`
+  Reason: useful initial repo-context snapshot, but stale as a standing note.
+- `notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md`
+  Reason: explains why `tools/Project_Design_Workflow.md` was revised and what reusable principles were added.
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+  Reason: foundational mismatch review that drove the V1 execution-record and review-gate design.
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+  Reason: captures why the early V1 sequence was ordered policy first, architecture second, runner change last.
+- `notes/009_run_prompt_candidate_build__20260415_233407.md`
+  Reason: records what behavior `run_prompt.py` intentionally preserved and what changed for V1 records.
+- `notes/010_run_prompt_candidate_review__20260415_234559.md`
+  Reason: captures the first serious assessment of the new runner and the environment-vs-runner distinction.
+- `notes/011_review_writeback_helper_build__20260415_235514.md`
+  Reason: implementation and validation note for the still-live `review_run.py` helper.
+- `notes/012_v1_pipeline_options_review__20260416_000819.md`
+  Reason: useful maturity review and bounded option set before operational helpers were added.
+- `notes/014_queue_readiness_checker_build__20260416_002419.md`
+  Reason: implementation and validation note for the still-live readiness helper.
+- `notes/015_review_backlog_lister_build__20260416_010500.md`
+  Reason: implementation and validation note for the still-live review backlog helper.
+- `notes/016_queue_and_backlog_helper_validation__20260416_003710.md`
+  Reason: repo-specific validation evidence showing how readiness and backlog behavior interact with mixed legacy and V1 notes.
+- `notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md`
+  Reason: records the rationale for the current `Queue note:` behavior in the readiness helper.
+
+## Move To Attic Without Summary
+
+### notes/
+
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md`
+  Reason: legacy smoke-test receipt; superseded by later V1 records and duplicates.
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md`
+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md`
+  Reason: duplicate legacy smoke-test receipt with no new durable decision.
+- `notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md`
+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
+- `notes/003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md`
+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
+- `notes/004_next_design_step_recommendation__20260415_195538.md`
+  Reason: brief recommendation fully subsumed by the main review and later execution-record work.
+- `notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md`
+  Reason: success receipt; the main review note is the durable artifact.
+- `notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md`
+  Reason: success receipt; resulting doc and queue-plan note carry the lasting value.
+- `notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md`
+  Reason: success receipt; durable knowledge lives in `tools/codex/V1_Run_Review_Gate.md`.
+- `notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md`
+  Reason: success receipt; durable knowledge lives in the architecture doc.
+- `notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md`
+  Reason: success receipt; durable knowledge lives in the bridge-runner spec.
+- `notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md`
+  Reason: implementation receipt; the build note and the code itself carry the durable content.
+- `notes/010_next_step_recommendation__20260415_234559.md`
+  Reason: brief next-step recommendation that was quickly superseded by the actual sequence taken.
+- `notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md`
+  Reason: success receipt; the review note carries the useful content.
+- `notes/012_top_three_next_options__20260416_000819.md`
+  Reason: short ranking note largely duplicated by the fuller options review and later implementation history.
+- `notes/013_next_two_prompt_plan__20260416_002005.md`
+  Reason: prompt-generation plan already realized by `014` and `015`.
+- `notes/016_next_improvement_recommendation__20260416_003710.md`
+  Reason: one-line recommendation already realized by `017`.
+- `notes/018_prioritized_remaining_work__20260416_005130.md`
+  Reason: concise ranking duplicated by the fuller architecture sweep and the implemented follow-up work.
+- `notes/019_next_two_cleanup_prompts_plan__20260416_010251.md`
+  Reason: prompt-generation plan already realized by `020` and `021`.
+
+### codex_prompts/
+
+- `codex_prompts/001_smoke_test_pipeline.md`
+  Reason: one-use bootstrap prompt; durable outcome was only the existence of a working notes path.
+- `codex_prompts/002_repo_inventory_and_status.md`
+  Reason: one-use repo-inventory scaffold; durable value is in the resulting note, not the prompt text.
+- `codex_prompts/003_revise_Project_Deisgn_workflow_document.md`
+  Reason: implementation instruction for a completed doc revision; resulting document and revision note now carry the knowledge.
+- `codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md`
+  Reason: review scaffold whose durable output is the architecture/runner assessment note.
+- `codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md`
+  Reason: queue-generation scaffold; durable outputs are the execution-record doc and queue-plan note.
+- `codex_prompts/006_define_v1_run_review_gate.md`
+  Reason: design scaffold whose durable output is `tools/codex/V1_Run_Review_Gate.md`.
+- `codex_prompts/007_align_architecture_doc_to_v1_workflow.md`
+  Reason: alignment scaffold whose durable output is the revised architecture doc.
+- `codex_prompts/008_define_minimal_bridge_runner_change_spec.md`
+  Reason: spec-writing scaffold whose durable output is the bridge-runner spec.
+- `codex_prompts/009_build_v1_run_prompt_candidate.md`
+  Reason: implementation prompt for completed runner work; code and build note now carry the value.
+- `codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md`
+  Reason: review scaffold whose durable output is the runner-review note.
+- `codex_prompts/011_build_v1_review_writeback_helper.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md`
+  Reason: review scaffold whose durable output is the options review note.
+- `codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md`
+  Reason: prompt-generation scaffold already realized by later prompt files and notes.
+- `codex_prompts/014_build_queue_readiness_checker.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/015_build_review_backlog_unreviewed_run_lister.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md`
+  Reason: validation scaffold whose durable content is in the validation note.
+- `codex_prompts/017_polish_queue_readiness_gap_explanation.md`
+  Reason: small polish prompt for completed helper behavior; code and polish note now carry the value.
+- `codex_prompts/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work.md`
+  Reason: review scaffold whose durable output is the architecture sweep note.
+- `codex_prompts/019_generate_prompts_for_doc_alignment_and_record_contract_validation.md`
+  Reason: prompt-generation scaffold already realized by `020` and `021`.
+- `codex_prompts/020_align_v1_doc_and_spec_packet_to_actual_toolset.md`
+  Reason: implementation prompt for completed doc cleanup; the edited docs and cleanup note now carry the value.
+- `codex_prompts/021_add_lightweight_v1_record_contract_validation.md`
+  Reason: implementation prompt for completed validator work; code and validation note now carry the value.
+
+## Uncertain / Needs Human Review
+
+### notes/
+
+- `notes/001_smoke_test_pipeline__20260415_233343.md`
+  Reason: current V1 execution record; moving it changes live run history and helper output.
+- `notes/001_smoke_test_pipeline__20260415_234918.md`
+  Reason: current V1 execution record and current accepted prior evidence for prompt `001`.
+- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/014_build_queue_readiness_checker__20260416_002319.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work__20260416_005034.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/020_align_v1_doc_and_spec_packet_to_actual_toolset__20260416_010453.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/021_add_lightweight_v1_record_contract_validation__20260416_010810.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+
+### codex_prompts/
+
+- `codex_prompts/022_audit_and_classify_scaffolding_for_summary_and_attic.md`
+  Reason: current audit prompt for the cleanup sequence; do not classify as attic material until this report has been acted on and superseded.
+
+## Practical Read Of The Repo
+
+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
+- The second cleanup target is the durable-but-noisy implementation notes, but only after their key rationale is extracted into a smaller standing summary.
+- The current V1 execution records should not move in a cleanup pass unless the repo first decides how the readiness/backlog tools should treat archived records.
diff --git a/notes/022_scaffolding_cleanup_plan__20260416_023016.md b/notes/022_scaffolding_cleanup_plan__20260416_023016.md
new file mode 100644
index 0000000000000000000000000000000000000000..f6140b3bc4f4031dae4d9e337672d4f3d2ecae99
--- /dev/null
+++ b/notes/022_scaffolding_cleanup_plan__20260416_023016.md
@@ -0,0 +1,20 @@
+# 022 Scaffolding Cleanup Plan
+
+## Next Pass
+
+- Extract a compact durable summary from the files marked `summarize, then move to attic`.
+- Keep that summary in a small standing note or note packet before any moves happen.
+- Do a separate policy check on current V1 execution-record files in `notes/`; do not move them unless the repo first decides how archived records should affect readiness and backlog helpers.
+
+## After That
+
+- Move the `attic without summary` set in small batches.
+- Move the `summarize, then move to attic` set only after the summary note exists and has been reviewed.
+- Re-run a quick repo sanity check so the remaining `notes/` surface is mostly current operational notes plus live V1 records.
+
+## Do Not Do Yet
+
+- Do not create the attic in this pass.
+- Do not move or delete current V1 execution-record files.
+- Do not rewrite tools or major design docs as part of cleanup.
+- Do not delete prompt files or notes before the summary-extraction step is reviewed.

codex
Created [022_scaffolding_classification_report__20260416_023016.md](/data/git/handson-ml3/notes/022_scaffolding_classification_report__20260416_023016.md) and [022_scaffolding_cleanup_plan__20260416_023016.md](/data/git/handson-ml3/notes/022_scaffolding_cleanup_plan__20260416_023016.md).

The report classifies `notes/` and `codex_prompts/` into the four requested buckets, with file-specific reasons. The main conservative call is that current V1 execution records remain `uncertain` because `check_queue_readiness.py` and `list_review_backlog.py` still treat them as live source-of-truth data, so moving them now would change behavior. No files were moved or deleted.
diff --git a/notes/022_scaffolding_classification_report__20260416_023016.md b/notes/022_scaffolding_classification_report__20260416_023016.md
new file mode 100644
index 0000000000000000000000000000000000000000..ff6b5ff2902c910c2b6d5b21196592666fef35c0
--- /dev/null
+++ b/notes/022_scaffolding_classification_report__20260416_023016.md
@@ -0,0 +1,189 @@
+# 022 Scaffolding Classification Report
+
+## Short Summary
+
+`notes/` currently mixes four different things:
+
+- legacy pre-V1 success receipts
+- durable design and implementation notes
+- current V1 execution records that the live helpers still read
+- small planning / recommendation notes that were useful during construction but are now mostly superseded
+
+`codex_prompts/` is almost entirely construction scaffolding for building the current V1 toolset. The durable knowledge from those prompts now lives in the tool/docs packet and in a smaller set of implementation and review notes.
+
+The main conservative rule in this report is:
+
+- current V1 execution-record files stay out of attic decisions for now, because moving them would change the behavior of `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py`
+
+## Classification Criteria Used
+
+- `permanent residents`: still useful as standing maintainer context in their current form
+- `summarize, then move to attic`: contains durable decisions, validation evidence, or design rationale, but does not need to stay in `notes/` once that knowledge is extracted
+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
+- `uncertain / needs human review`: not safe to move yet because it is still operationally active, still part of the current source-of-truth set, or still too close to the active cleanup pass
+
+## Permanent Residents
+
+- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
+  Reason: best compact repo-specific statement of what the V1 system currently is, what remains missing, and whether it is ready for real use.
+- `notes/020_doc_spec_alignment_cleanup__20260416_010534.md`
+  Reason: explains why the current design/spec packet changed and records the final doc-alignment decisions that still govern the written packet.
+- `notes/021_record_contract_validation__20260416_011314.md`
+  Reason: durable contract and validation evidence for the shared V1 execution-record parser used by live tools.
+
+## Summarize, Then Move To Attic
+
+- `notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md`
+  Reason: useful initial repo-context snapshot, but stale as a standing note.
+- `notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md`
+  Reason: explains why `tools/Project_Design_Workflow.md` was revised and what reusable principles were added.
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+  Reason: foundational mismatch review that drove the V1 execution-record and review-gate design.
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+  Reason: captures why the early V1 sequence was ordered policy first, architecture second, runner change last.
+- `notes/009_run_prompt_candidate_build__20260415_233407.md`
+  Reason: records what behavior `run_prompt.py` intentionally preserved and what changed for V1 records.
+- `notes/010_run_prompt_candidate_review__20260415_234559.md`
+  Reason: captures the first serious assessment of the new runner and the environment-vs-runner distinction.
+- `notes/011_review_writeback_helper_build__20260415_235514.md`
+  Reason: implementation and validation note for the still-live `review_run.py` helper.
+- `notes/012_v1_pipeline_options_review__20260416_000819.md`
+  Reason: useful maturity review and bounded option set before operational helpers were added.
+- `notes/014_queue_readiness_checker_build__20260416_002419.md`
+  Reason: implementation and validation note for the still-live readiness helper.
+- `notes/015_review_backlog_lister_build__20260416_010500.md`
+  Reason: implementation and validation note for the still-live review backlog helper.
+- `notes/016_queue_and_backlog_helper_validation__20260416_003710.md`
+  Reason: repo-specific validation evidence showing how readiness and backlog behavior interact with mixed legacy and V1 notes.
+- `notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md`
+  Reason: records the rationale for the current `Queue note:` behavior in the readiness helper.
+
+## Move To Attic Without Summary
+
+### notes/
+
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md`
+  Reason: legacy smoke-test receipt; superseded by later V1 records and duplicates.
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md`
+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
+- `notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md`
+  Reason: duplicate legacy smoke-test receipt with no new durable decision.
+- `notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md`
+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
+- `notes/003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md`
+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
+- `notes/004_next_design_step_recommendation__20260415_195538.md`
+  Reason: brief recommendation fully subsumed by the main review and later execution-record work.
+- `notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md`
+  Reason: success receipt; the main review note is the durable artifact.
+- `notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md`
+  Reason: success receipt; resulting doc and queue-plan note carry the lasting value.
+- `notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md`
+  Reason: success receipt; durable knowledge lives in `tools/codex/V1_Run_Review_Gate.md`.
+- `notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md`
+  Reason: success receipt; durable knowledge lives in the architecture doc.
+- `notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md`
+  Reason: success receipt; durable knowledge lives in the bridge-runner spec.
+- `notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md`
+  Reason: implementation receipt; the build note and the code itself carry the durable content.
+- `notes/010_next_step_recommendation__20260415_234559.md`
+  Reason: brief next-step recommendation that was quickly superseded by the actual sequence taken.
+- `notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md`
+  Reason: success receipt; the review note carries the useful content.
+- `notes/012_top_three_next_options__20260416_000819.md`
+  Reason: short ranking note largely duplicated by the fuller options review and later implementation history.
+- `notes/013_next_two_prompt_plan__20260416_002005.md`
+  Reason: prompt-generation plan already realized by `014` and `015`.
+- `notes/016_next_improvement_recommendation__20260416_003710.md`
+  Reason: one-line recommendation already realized by `017`.
+- `notes/018_prioritized_remaining_work__20260416_005130.md`
+  Reason: concise ranking duplicated by the fuller architecture sweep and the implemented follow-up work.
+- `notes/019_next_two_cleanup_prompts_plan__20260416_010251.md`
+  Reason: prompt-generation plan already realized by `020` and `021`.
+
+### codex_prompts/
+
+- `codex_prompts/001_smoke_test_pipeline.md`
+  Reason: one-use bootstrap prompt; durable outcome was only the existence of a working notes path.
+- `codex_prompts/002_repo_inventory_and_status.md`
+  Reason: one-use repo-inventory scaffold; durable value is in the resulting note, not the prompt text.
+- `codex_prompts/003_revise_Project_Deisgn_workflow_document.md`
+  Reason: implementation instruction for a completed doc revision; resulting document and revision note now carry the knowledge.
+- `codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md`
+  Reason: review scaffold whose durable output is the architecture/runner assessment note.
+- `codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md`
+  Reason: queue-generation scaffold; durable outputs are the execution-record doc and queue-plan note.
+- `codex_prompts/006_define_v1_run_review_gate.md`
+  Reason: design scaffold whose durable output is `tools/codex/V1_Run_Review_Gate.md`.
+- `codex_prompts/007_align_architecture_doc_to_v1_workflow.md`
+  Reason: alignment scaffold whose durable output is the revised architecture doc.
+- `codex_prompts/008_define_minimal_bridge_runner_change_spec.md`
+  Reason: spec-writing scaffold whose durable output is the bridge-runner spec.
+- `codex_prompts/009_build_v1_run_prompt_candidate.md`
+  Reason: implementation prompt for completed runner work; code and build note now carry the value.
+- `codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md`
+  Reason: review scaffold whose durable output is the runner-review note.
+- `codex_prompts/011_build_v1_review_writeback_helper.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md`
+  Reason: review scaffold whose durable output is the options review note.
+- `codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md`
+  Reason: prompt-generation scaffold already realized by later prompt files and notes.
+- `codex_prompts/014_build_queue_readiness_checker.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/015_build_review_backlog_unreviewed_run_lister.md`
+  Reason: implementation prompt for completed helper work; code and build note now carry the value.
+- `codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md`
+  Reason: validation scaffold whose durable content is in the validation note.
+- `codex_prompts/017_polish_queue_readiness_gap_explanation.md`
+  Reason: small polish prompt for completed helper behavior; code and polish note now carry the value.
+- `codex_prompts/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work.md`
+  Reason: review scaffold whose durable output is the architecture sweep note.
+- `codex_prompts/019_generate_prompts_for_doc_alignment_and_record_contract_validation.md`
+  Reason: prompt-generation scaffold already realized by `020` and `021`.
+- `codex_prompts/020_align_v1_doc_and_spec_packet_to_actual_toolset.md`
+  Reason: implementation prompt for completed doc cleanup; the edited docs and cleanup note now carry the value.
+- `codex_prompts/021_add_lightweight_v1_record_contract_validation.md`
+  Reason: implementation prompt for completed validator work; code and validation note now carry the value.
+
+## Uncertain / Needs Human Review
+
+### notes/
+
+- `notes/001_smoke_test_pipeline__20260415_233343.md`
+  Reason: current V1 execution record; moving it changes live run history and helper output.
+- `notes/001_smoke_test_pipeline__20260415_234918.md`
+  Reason: current V1 execution record and current accepted prior evidence for prompt `001`.
+- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/014_build_queue_readiness_checker__20260416_002319.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work__20260416_005034.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/020_align_v1_doc_and_spec_packet_to_actual_toolset__20260416_010453.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+- `notes/021_add_lightweight_v1_record_contract_validation__20260416_010810.md`
+  Reason: current V1 execution record; still part of backlog/readiness evidence.
+
+### codex_prompts/
+
+- `codex_prompts/022_audit_and_classify_scaffolding_for_summary_and_attic.md`
+  Reason: current audit prompt for the cleanup sequence; do not classify as attic material until this report has been acted on and superseded.
+
+## Practical Read Of The Repo
+
+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
+- The second cleanup target is the durable-but-noisy implementation notes, but only after their key rationale is extracted into a smaller standing summary.
+- The current V1 execution records should not move in a cleanup pass unless the repo first decides how the readiness/backlog tools should treat archived records.
diff --git a/notes/022_scaffolding_cleanup_plan__20260416_023016.md b/notes/022_scaffolding_cleanup_plan__20260416_023016.md
new file mode 100644
index 0000000000000000000000000000000000000000..f6140b3bc4f4031dae4d9e337672d4f3d2ecae99
--- /dev/null
+++ b/notes/022_scaffolding_cleanup_plan__20260416_023016.md
@@ -0,0 +1,20 @@
+# 022 Scaffolding Cleanup Plan
+
+## Next Pass
+
+- Extract a compact durable summary from the files marked `summarize, then move to attic`.
+- Keep that summary in a small standing note or note packet before any moves happen.
+- Do a separate policy check on current V1 execution-record files in `notes/`; do not move them unless the repo first decides how archived records should affect readiness and backlog helpers.
+
+## After That
+
+- Move the `attic without summary` set in small batches.
+- Move the `summarize, then move to attic` set only after the summary note exists and has been reviewed.
+- Re-run a quick repo sanity check so the remaining `notes/` surface is mostly current operational notes plus live V1 records.
+
+## Do Not Do Yet
+
+- Do not create the attic in this pass.
+- Do not move or delete current V1 execution-record files.
+- Do not rewrite tools or major design docs as part of cleanup.
+- Do not delete prompt files or notes before the summary-extraction step is reviewed.

tokens used
90,003
```
