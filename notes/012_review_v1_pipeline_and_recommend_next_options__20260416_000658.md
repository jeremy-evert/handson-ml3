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

codex
I’m reviewing the current V1 prompt workflow end to end from the design docs, code, and recent notes, then I’ll distill the credible next bounded options into two note artifacts.
  → Read the specified design docs, workflow code, and recent notes to map the current V1 pipeline state
  • Synthesize working pieces, seams, and bounded next-step options grounded in repo evidence
  • Write the two requested notes under notes/ with timestamped filenames
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
/bin/bash -lc "sed -n '1,260p' tools/codex/V1_Execution_Record_Artifact.md" in /data/git/handson-ml3
exec
/bin/bash -lc "sed -n '1,260p' tools/Codex_Prompt_Workflow_Architecture.md" in /data/git/handson-ml3
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
I’ve got the design packet loaded. Next I’m reading the actual runner and review code plus the recent execution/review notes to see where the implementation matches the spec and where it still has seams.
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
/bin/bash -lc "sed -n '1,240p' notes/009_run_prompt_candidate_build__20260415_233407.md" in /data/git/handson-ml3
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
/bin/bash -lc "sed -n '1,240p' notes/010_run_prompt_candidate_review__20260415_234559.md" in /data/git/handson-ml3
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
/bin/bash -lc "sed -n '1,240p' notes/010_next_step_recommendation__20260415_234559.md" in /data/git/handson-ml3
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
/bin/bash -lc "sed -n '1,260p' notes/011_review_writeback_helper_build__20260415_235514.md" in /data/git/handson-ml3
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
/bin/bash -lc "sed -n '1,220p' notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 001 Smoke Test Pipeline

- prompt filename: `codex_prompts/001_smoke_test_pipeline.md`
- timestamp: `20260415_133235`
- status: `SUCCESS`
- expected folders found: `codex_prompts/`, `notes/`, `tools/`
- prompt file readable: `yes`

Pipeline appears functional: the required repository scaffolding exists, the prompt file was readable, and this report was written to `notes/`.

exec
/bin/bash -lc "sed -n '1,220p' notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 001_smoke_test_pipeline - SUCCESS

- Prompt file: `001_smoke_test_pipeline.md`
- Timestamp (UTC): `20260415_183223`
- Status: `SUCCESS`

## Original Prompt

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

## Codex Output

Verified successfully. `codex_prompts/`, `notes/`, and `tools/` all exist, and `codex_prompts/001_smoke_test_pipeline.md` was readable.

I created the report at [001_smoke_test_pipeline__SUCCESS__20260415_133235.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md). No unrelated repo changes were made.

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
session id: 019d926a-73fc-72b1-8f58-f26cc7ac7320
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
I’m checking the repository layout first, then I’ll verify the prompt file context and write the smoke-test note into `notes/` with a pass/fail result.
exec
/bin/bash -lc pwd in /data/git/handson-ml3
 succeeded in 0ms:
/data/git/handson-ml3

exec
/bin/bash -lc 'for d in codex_prompts notes tools; do if [ -d "$d" ]; then echo "$d: present"; else echo "$d: missing"; fi; done' in /data/git/handson-ml3
 succeeded in 0ms:
codex_prompts: present
notes: present
tools: present

exec
/bin/bash -lc "rg --files -g '*001*smoke*test*pipeline*' -g '*001*Smoke*Test*Pipeline*'" in /data/git/handson-ml3
 succeeded in 0ms:
codex_prompts/001_smoke_test_pipeline.md

codex
The expected directories are present, and I found the prompt file at `codex_prompts/001_smoke_test_pipeline.md`. I’m reading it now so the report can explicitly record that it was accessible.
exec
/bin/bash -lc "date '+%Y%m%d_%H%M%S'" in /data/git/handson-ml3
 succeeded in 0ms:
20260415_133235

exec
/bin/bash -lc "sed -n '1,120p' codex_prompts/001_smoke_test_pipeline.md" in /data/git/handson-ml3
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

codex
I have enough context. I’m writing a single note under `notes/` and keeping the change limited to that file.
apply patch
patch: completed
/data/git/handson-ml3/notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md
diff --git a/notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md b/notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md
new file mode 100644
index 0000000000000000000000000000000000000000..3555cd0a504aac1d95a557b043423c8177eb1b22
--- /dev/null
+++ b/notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md
@@ -0,0 +1,9 @@
+# 001 Smoke Test Pipeline
+
+- prompt filename: `codex_prompts/001_smoke_test_pipeline.md`
+- timestamp: `20260415_133235`
+- status: `SUCCESS`
+- expected folders found: `codex_prompts/`, `notes/`, `tools/`
+- prompt file readable: `yes`
+
+Pipeline appears functional: the required repository scaffolding exists, the prompt file was readable, and this report was written to `notes/`.

diff --git a/notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md b/notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md
new file mode 100644
index 0000000000000000000000000000000000000000..3555cd0a504aac1d95a557b043423c8177eb1b22
--- /dev/null
+++ b/notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md
@@ -0,0 +1,9 @@
+# 001 Smoke Test Pipeline
+
+- prompt filename: `codex_prompts/001_smoke_test_pipeline.md`
+- timestamp: `20260415_133235`
+- status: `SUCCESS`
+- expected folders found: `codex_prompts/`, `notes/`, `tools/`
+- prompt file readable: `yes`
+
+Pipeline appears functional: the required repository scaffolding exists, the prompt file was readable, and this report was written to `notes/`.

codex
The report file is in place. I’m doing one quick verification pass on the note path, then I’ll close with the result.
exec
/bin/bash -lc 'ls -1 notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md' in /data/git/handson-ml3
 succeeded in 0ms:
notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md

exec
/bin/bash -lc "sed -n '1,80p' notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md" in /data/git/handson-ml3
 succeeded in 0ms:
# 001 Smoke Test Pipeline

exec
/bin/bash -lc "sed -n '1,220p' notes/011_build_v1_review_writeback_helper__20260415_235346.md" in /data/git/handson-ml3
 succeeded in 0ms:
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

codex
The implementation evidence is clearer now: the execution runner and the manual review write-back helper both exist, and there’s already one successful reviewed run plus one failure-path artifact. I’m doing a targeted pass for doc drift and awkward seams before I write the recommendation notes.
exec
/bin/bash -lc "rg -n \"baby_run_prompt|run_prompt.py|review_run.py|SUCCESS|FAILED|UNREVIEWED|ACCEPTED|REJECTED|run_id\" tools notes/codex_prompts -g '"'!**/__pycache__/**'"'" in /data/git/handson-ml3
exec
/bin/bash -lc "sed -n '260,520p' tools/Codex_Prompt_Workflow_Architecture.md" in /data/git/handson-ml3
 succeeded in 0ms:
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
/bin/bash -lc "sed -n '220,420p' tools/codex/V1_Bridge_Runner_Change_Spec.md" in /data/git/handson-ml3
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

exec
/bin/bash -lc 'git status --short' in /data/git/handson-ml3
 succeeded in 0ms:
?? codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md

 exited 2 in 101ms:
codex
I’ve identified the main implementation seams already: one is documentation drift around the runner target and identity rules, and another is that the review gate exists operationally but not yet as an explicit queue-release mechanism. I’m pulling exact line references now so the report can stay grounded in the repo rather than general commentary.
exec
/bin/bash -lc "nl -ba tools/codex/run_prompt.py | sed -n '1,240p'" in /data/git/handson-ml3
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	
     3	from __future__ import annotations
     4	
     5	import argparse
     6	import datetime as dt
     7	import subprocess
     8	import sys
     9	import tempfile
    10	import time
    11	from pathlib import Path
    12	
    13	
    14	PROMPTS_DIR = "codex_prompts"
    15	NOTES_DIR = "notes"
    16	RUNNER_PATH = "tools/codex/run_prompt.py"
    17	
    18	
    19	def utc_now() -> dt.datetime:
    20	    return dt.datetime.now(dt.timezone.utc)
    21	
    22	
    23	def utc_timestamp(moment: dt.datetime) -> str:
    24	    return moment.strftime("%Y%m%d_%H%M%S")
    25	
    26	
    27	def repo_root() -> Path:
    28	    return Path(__file__).resolve().parents[2]
    29	
    30	
    31	def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
    32	    prompt = Path(prompt_arg)
    33	
    34	    if prompt.is_absolute():
    35	        return prompt
    36	
    37	    direct = root / prompt
    38	    if direct.exists():
    39	        return direct
    40	
    41	    in_prompts = root / PROMPTS_DIR / prompt
    42	    if in_prompts.exists():
    43	        return in_prompts
    44	
    45	    matches = sorted((root / PROMPTS_DIR).glob(f"{prompt_arg}*"))
    46	    if len(matches) == 1:
    47	        return matches[0]
    48	
    49	    return in_prompts
    50	
    51	
    52	def prompt_file_label(root: Path, prompt_path: Path) -> str:
    53	    try:
    54	        return prompt_path.resolve().relative_to(root.resolve()).as_posix()
    55	    except ValueError:
    56	        return str(prompt_path.resolve())
    57	
    58	
    59	def build_record_path(notes_dir: Path, prompt_stem: str, started_at_utc: str) -> tuple[str, Path]:
    60	    base_run_id = f"{prompt_stem}__{started_at_utc}"
    61	    candidate = notes_dir / f"{base_run_id}.md"
    62	    if not candidate.exists():
    63	        return base_run_id, candidate
    64	
    65	    suffix = 2
    66	    while True:
    67	        run_id = f"{base_run_id}__{suffix}"
    68	        candidate = notes_dir / f"{run_id}.md"
    69	        if not candidate.exists():
    70	            return run_id, candidate
    71	        suffix += 1
    72	
    73	
    74	def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
    75	    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
    76	        output_path = Path(handle.name)
    77	
    78	    try:
    79	        result = subprocess.run(
    80	            ["codex", "exec", "-C", str(root), "--output-last-message", str(output_path), "-"],
    81	            input=prompt_text,
    82	            text=True,
    83	            capture_output=True,
    84	            check=False,
    85	        )
    86	        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    87	        return result.returncode, final_output, result.stderr
    88	    finally:
    89	        output_path.unlink(missing_ok=True)
    90	
    91	
    92	def fenced_block(text: str, fence: str, info: str) -> str:
    93	    body = text.rstrip("\n")
    94	    return f"{fence}{info}\n{body}\n{fence}" if body else f"{fence}{info}\n\n{fence}"
    95	
    96	
    97	def build_record_content(
    98	    *,
    99	    run_id: str,
   100	    prompt_file: str,
   101	    prompt_stem: str,
   102	    started_at_utc: str,
   103	    execution_status: str,
   104	    finished_at_utc: str,
   105	    return_code: int,
   106	    prompt_text: str,
   107	    codex_output: str,
   108	    stderr_text: str,
   109	    elapsed_seconds: float,
   110	) -> str:
   111	    final_output = codex_output.rstrip()
   112	    stderr_body = stderr_text.rstrip()
   113	
   114	    sections = [
   115	        f"# {run_id}",
   116	        "",
   117	        f"- run_id: `{run_id}`",
   118	        f"- prompt_file: `{prompt_file}`",
   119	        f"- prompt_stem: `{prompt_stem}`",
   120	        f"- started_at_utc: `{started_at_utc}`",
   121	        "",
   122	        "## Execution Facts",
   123	        "",
   124	        f"- execution_status: `{execution_status}`",
   125	        f"- finished_at_utc: `{finished_at_utc}`",
   126	        f"- runner: `{RUNNER_PATH}`",
   127	        f"- return_code: `{return_code}`",
   128	        "- retry_of_run_id:",
   129	        "",
   130	        "## Review Facts",
   131	        "",
   132	        "- review_status: `UNREVIEWED`",
   133	        "- review_summary:",
   134	        "- reviewed_by:",
   135	        "- reviewed_at_utc:",
   136	        "",
   137	        "## Failure Analysis",
   138	        "",
   139	        "- failure_type:",
   140	        "- failure_symptom:",
   141	        "- likely_cause:",
   142	        "- recommended_next_action:",
   143	        "",
   144	        "## Resource / Cost Facts",
   145	        "",
   146	        f"- elapsed_seconds: `{elapsed_seconds:.3f}`",
   147	        f"- final_output_char_count: `{len(codex_output)}`",
   148	        f"- stderr_char_count: `{len(stderr_text)}`",
   149	        "",
   150	        "## Prompt Text",
   151	        "",
   152	        fenced_block(prompt_text, "```", "md"),
   153	        "",
   154	        "## Codex Final Output",
   155	        "",
   156	        final_output if final_output else "*No output captured.*",
   157	        "",
   158	        "## Stderr",
   159	        "",
   160	        fenced_block(stderr_body, "```", "text") if stderr_body else "*No stderr captured.*",
   161	        "",
   162	    ]
   163	    return "\n".join(sections)
   164	
   165	
   166	def main() -> int:
   167	    parser = argparse.ArgumentParser(
   168	        description="Run one prompt file through codex exec and write a V1 execution record."
   169	    )
   170	    parser.add_argument("prompt", help="Prompt filename, numeric prefix, or path")
   171	    args = parser.parse_args()
   172	
   173	    root = repo_root()
   174	    prompts_dir = root / PROMPTS_DIR
   175	    notes_dir = root / NOTES_DIR
   176	
   177	    if not prompts_dir.exists():
   178	        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
   179	        return 1
   180	
   181	    if not notes_dir.exists():
   182	        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
   183	        return 1
   184	
   185	    prompt_path = resolve_prompt_path(root, args.prompt)
   186	    if not prompt_path.exists():
   187	        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
   188	        return 1
   189	
   190	    prompt_text = prompt_path.read_text(encoding="utf-8")
   191	    started_at = utc_now()
   192	    started_at_utc = utc_timestamp(started_at)
   193	    run_id, record_path = build_record_path(notes_dir, prompt_path.stem, started_at_utc)
   194	
   195	    monotonic_start = time.monotonic()
   196	    return_code, codex_output, stderr_text = run_codex(root, prompt_text)
   197	    elapsed_seconds = time.monotonic() - monotonic_start
   198	    finished_at_utc = utc_timestamp(utc_now())
   199	    execution_status = "EXECUTED" if return_code == 0 else "EXECUTION_FAILED"
   200	
   201	    record_path.write_text(
   202	        build_record_content(
   203	            run_id=run_id,
   204	            prompt_file=prompt_file_label(root, prompt_path),
   205	            prompt_stem=prompt_path.stem,
   206	            started_at_utc=started_at_utc,
   207	            execution_status=execution_status,
   208	            finished_at_utc=finished_at_utc,
   209	            return_code=return_code,
   210	            prompt_text=prompt_text,
   211	            codex_output=codex_output,
   212	            stderr_text=stderr_text,
   213	            elapsed_seconds=elapsed_seconds,
   214	        ),
   215	        encoding="utf-8",
   216	    )
   217	
   218	    print(record_path)
   219	    return return_code
   220	
   221	
   222	if __name__ == "__main__":
   223	    raise SystemExit(main())

exec
/bin/bash -lc "nl -ba tools/codex/review_run.py | sed -n '1,240p'" in /data/git/handson-ml3
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	
     3	from __future__ import annotations
     4	
     5	import argparse
     6	import datetime as dt
     7	import re
     8	import sys
     9	from pathlib import Path
    10	
    11	
    12	REVIEW_STATUSES = {"ACCEPTED", "REJECTED"}
    13	FAILURE_FIELDS = (
    14	    "failure_type",
    15	    "failure_symptom",
    16	    "likely_cause",
    17	    "recommended_next_action",
    18	)
    19	REQUIRED_FIELDS = (
    20	    "run_id",
    21	    "prompt_file",
    22	    "prompt_stem",
    23	    "started_at_utc",
    24	    "execution_status",
    25	    "finished_at_utc",
    26	    "runner",
    27	    "return_code",
    28	    "retry_of_run_id",
    29	    "review_status",
    30	    "review_summary",
    31	    "reviewed_by",
    32	    "reviewed_at_utc",
    33	    *FAILURE_FIELDS,
    34	    "elapsed_seconds",
    35	    "final_output_char_count",
    36	    "stderr_char_count",
    37	)
    38	REQUIRED_SECTIONS = (
    39	    "## Execution Facts",
    40	    "## Review Facts",
    41	    "## Failure Analysis",
    42	    "## Resource / Cost Facts",
    43	    "## Prompt Text",
    44	    "## Codex Final Output",
    45	    "## Stderr",
    46	)
    47	
    48	
    49	def utc_timestamp() -> str:
    50	    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    51	
    52	
    53	def repo_root() -> Path:
    54	    return Path(__file__).resolve().parents[2]
    55	
    56	
    57	def parse_args() -> argparse.Namespace:
    58	    parser = argparse.ArgumentParser(
    59	        description="Write manual V1 review fields back into an existing execution record."
    60	    )
    61	    parser.add_argument("record", help="Path to an existing execution-record markdown file")
    62	    parser.add_argument(
    63	        "--review-status",
    64	        required=True,
    65	        choices=sorted(REVIEW_STATUSES),
    66	        help="Manual review outcome",
    67	    )
    68	    parser.add_argument(
    69	        "--review-summary",
    70	        required=True,
    71	        help="Short manual review summary",
    72	    )
    73	    parser.add_argument("--reviewed-by", help="Reviewer identifier")
    74	    parser.add_argument(
    75	        "--reviewed-at-utc",
    76	        help="Review timestamp in UTC using YYYYMMDD_HHMMSS; defaults to current UTC time",
    77	    )
    78	    parser.add_argument("--failure-type", help="Short failure category for rejected runs")
    79	    parser.add_argument("--failure-symptom", help="Observed failure symptom for rejected runs")
    80	    parser.add_argument("--likely-cause", help="Likely root cause for rejected runs")
    81	    parser.add_argument(
    82	        "--recommended-next-action",
    83	        help="Manual next action recommendation for rejected runs",
    84	    )
    85	    return parser.parse_args()
    86	
    87	
    88	def fail(message: str) -> int:
    89	    print(f"ERROR: {message}", file=sys.stderr)
    90	    return 1
    91	
    92	
    93	def require_single_line(name: str, value: str | None) -> str | None:
    94	    if value is None:
    95	        return None
    96	    if "\n" in value or "\r" in value:
    97	        raise ValueError(f"{name} must be a single line")
    98	    return value.strip()
    99	
   100	
   101	def resolve_record_path(record_arg: str) -> Path:
   102	    record_path = Path(record_arg)
   103	    if record_path.is_absolute():
   104	        return record_path
   105	    return repo_root() / record_path
   106	
   107	
   108	def validate_record_path(record_path: Path) -> None:
   109	    if not record_path.exists():
   110	        raise ValueError(f"record file not found: {record_path}")
   111	    if not record_path.is_file():
   112	        raise ValueError(f"record path is not a file: {record_path}")
   113	
   114	    root = repo_root().resolve()
   115	    notes_dir = root / "notes"
   116	    try:
   117	        record_path.resolve().relative_to(notes_dir.resolve())
   118	    except ValueError as exc:
   119	        raise ValueError(f"record must be under {notes_dir}") from exc
   120	
   121	
   122	def validate_v1_record_structure(text: str) -> None:
   123	    if not text.startswith("# "):
   124	        raise ValueError("record does not start with a markdown title")
   125	
   126	    positions: list[int] = []
   127	    for section in REQUIRED_SECTIONS:
   128	        pos = text.find(section)
   129	        if pos == -1:
   130	            raise ValueError(f"record is missing section: {section}")
   131	        positions.append(pos)
   132	    if positions != sorted(positions):
   133	        raise ValueError("record sections are out of the expected V1 order")
   134	
   135	    for field in REQUIRED_FIELDS:
   136	        if re.search(rf"^- {re.escape(field)}:", text, flags=re.MULTILINE) is None:
   137	            raise ValueError(f"record is missing field line: {field}")
   138	
   139	
   140	def replace_field(text: str, field: str, value: str | None, *, code: bool = False) -> str:
   141	    rendered = f"`{value}`" if code and value else (value or "")
   142	    pattern = re.compile(rf"(?m)^- {re.escape(field)}:.*$")
   143	    if pattern.search(text) is None:
   144	        raise ValueError(f"record is missing field line: {field}")
   145	    return pattern.sub(f"- {field}: {rendered}", text, count=1)
   146	
   147	
   148	def build_updates(args: argparse.Namespace) -> dict[str, tuple[str | None, bool]]:
   149	    review_summary = require_single_line("review_summary", args.review_summary)
   150	    reviewed_by = require_single_line("reviewed_by", args.reviewed_by)
   151	    reviewed_at_utc = require_single_line("reviewed_at_utc", args.reviewed_at_utc) or utc_timestamp()
   152	    failure_values = {
   153	        "failure_type": require_single_line("failure_type", args.failure_type),
   154	        "failure_symptom": require_single_line("failure_symptom", args.failure_symptom),
   155	        "likely_cause": require_single_line("likely_cause", args.likely_cause),
   156	        "recommended_next_action": require_single_line(
   157	            "recommended_next_action", args.recommended_next_action
   158	        ),
   159	    }
   160	
   161	    if args.review_status != "REJECTED" and any(value for value in failure_values.values()):
   162	        raise ValueError("rejection-only failure fields may only be set when --review-status REJECTED")
   163	
   164	    updates: dict[str, tuple[str | None, bool]] = {
   165	        "review_status": (args.review_status, True),
   166	        "review_summary": (review_summary, False),
   167	        "reviewed_at_utc": (reviewed_at_utc, True),
   168	    }
   169	    if reviewed_by is not None:
   170	        updates["reviewed_by"] = (reviewed_by, False)
   171	
   172	    if args.review_status == "REJECTED":
   173	        for field, value in failure_values.items():
   174	            if value is not None:
   175	                updates[field] = (value, False)
   176	
   177	    return updates
   178	
   179	
   180	def main() -> int:
   181	    try:
   182	        args = parse_args()
   183	        record_path = resolve_record_path(args.record)
   184	        validate_record_path(record_path)
   185	        text = record_path.read_text(encoding="utf-8")
   186	        validate_v1_record_structure(text)
   187	
   188	        updated = text
   189	        for field, (value, code) in build_updates(args).items():
   190	            updated = replace_field(updated, field, value, code=code)
   191	
   192	        record_path.write_text(updated, encoding="utf-8")
   193	    except ValueError as exc:
   194	        return fail(str(exc))
   195	
   196	    print(record_path)
   197	    return 0
   198	
   199	
   200	if __name__ == "__main__":
   201	    raise SystemExit(main())

exec
/bin/bash -lc "nl -ba tools/codex/V1_Bridge_Runner_Change_Spec.md | sed -n '1,260p'" in /data/git/handson-ml3
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

exec
/bin/bash -lc "nl -ba tools/Codex_Prompt_Workflow_Architecture.md | sed -n '150,260p'" in /data/git/handson-ml3
 succeeded in 0ms:
   150	
   151	For V1, one run should produce one markdown execution-record file in `notes/`.
   152	
   153	That record is the minimum durable run artifact.
   154	
   155	No additional required queue file, database, or sidecar should be introduced in V1.
   156	
   157	---
   158	
   159	## Minimum Viable Slice
   160	
   161	The minimum viable slice is:
   162	
   163	1. execute one prompt
   164	2. write one execution record with `review_status: UNREVIEWED`
   165	3. stop for human review
   166	4. update the same execution record to either `ACCEPTED` or `REJECTED`
   167	5. treat only `ACCEPTED` as releasing the next prompt in sequence
   168	
   169	This is the smallest slice that proves the workflow rather than only the runner.
   170	
   171	It is intentionally narrower than:
   172	
   173	* a generalized queue engine
   174	* a full prompt-status system
   175	* a modular tool suite
   176	* an automation framework
   177	
   178	---
   179	
   180	## Role Of The Execution Record
   181	
   182	The execution record is the central V1 artifact.
   183	
   184	Its purpose is to preserve, in one inspectable markdown file:
   185	
   186	* what prompt was run
   187	* what happened during execution
   188	* what Codex returned
   189	* what lightweight failure or resource evidence was observed
   190	* what a human later decided about the run
   191	
   192	For V1:
   193	
   194	* the record body is the source of truth
   195	* the file in `notes/` is the durable local history unit
   196	* one run should not be split across multiple required files
   197	
   198	This keeps V1 small while still supporting:
   199	
   200	* review between iterations
   201	* failure analysis
   202	* lightweight cost awareness
   203	* stable run identity
   204	
   205	The execution record defined in `tools/codex/V1_Execution_Record_Artifact.md` should therefore be treated as a required architectural element, not an implementation detail.
   206	
   207	---
   208	
   209	## Role Of The Review Gate
   210	
   211	The review gate is the minimum manual checkpoint after execution record creation.
   212	
   213	Its purpose is to enforce the governing workflow rule that the next bounded step should not proceed only because a process finished successfully.
   214	
   215	For V1:
   216	
   217	* execution writes a record
   218	* new records begin as `UNREVIEWED`
   219	* human review decides `ACCEPTED` or `REJECTED`
   220	* only `ACCEPTED` allows the next prompt to be treated as ready
   221	
   222	This means the review gate is not a separate platform subsystem.
   223	
   224	It is the manual transition of a single execution record from:
   225	
   226	* `UNREVIEWED`
   227	
   228	to one of:
   229	
   230	* `ACCEPTED`
   231	* `REJECTED`
   232	
   233	The review gate defined in `tools/codex/V1_Run_Review_Gate.md` is therefore part of the architecture boundary for V1, not an optional later add-on.
   234	
   235	---
   236	
   237	## What The First Script Was Doing
   238	
   239	Below is a decomposition of the responsibilities that were bundled together.
   240	
   241	### 1. Repo path discovery
   242	
   243	* figure out where the repository root is
   244	* infer where `codex_prompts/` and `notes/` live
   245	
   246	### 2. Directory validation
   247	
   248	* verify the expected folders exist
   249	* fail cleanly if they do not
   250	
   251	### 3. Prompt discovery
   252	
   253	* scan `codex_prompts/`
   254	* identify valid prompt files
   255	* ignore files that do not match the naming convention
   256	
   257	### 4. Prompt parsing and indexing
   258	
   259	* extract numeric prefixes from filenames
   260	* sort prompts in execution order

exec
/bin/bash -lc "nl -ba notes/001_smoke_test_pipeline__20260415_234918.md | sed -n '1,120p'" in /data/git/handson-ml3
 succeeded in 0ms:
     1	# 001_smoke_test_pipeline__20260415_234918
     2	
     3	- run_id: `001_smoke_test_pipeline__20260415_234918`
     4	- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
     5	- prompt_stem: `001_smoke_test_pipeline`
     6	- started_at_utc: `20260415_234918`
     7	
     8	## Execution Facts
     9	
    10	- execution_status: `EXECUTED`
    11	- finished_at_utc: `20260415_234941`
    12	- runner: `tools/codex/run_prompt.py`
    13	- return_code: `0`
    14	- retry_of_run_id:
    15	
    16	## Review Facts
    17	
    18	- review_status: `ACCEPTED`
    19	- review_summary: Smoke test output and artifact are complete enough to accept this bounded step.
    20	- reviewed_by:
    21	- reviewed_at_utc: `20260415_235508`
    22	
    23	## Failure Analysis
    24	
    25	- failure_type:
    26	- failure_symptom:
    27	- likely_cause:
    28	- recommended_next_action:
    29	
    30	## Resource / Cost Facts
    31	
    32	- elapsed_seconds: `22.690`
    33	- final_output_char_count: `289`
    34	- stderr_char_count: `8503`
    35	
    36	## Prompt Text
    37	
    38	```md
    39	# 001 Smoke Test Pipeline
    40	
    41	Your task is to verify that the prompt workflow scaffolding exists and that the notes pipeline can record a result.
    42	
    43	## Goals
    44	1. Confirm that the following folders exist in the repository:
    45	   - `codex_prompts/`
    46	   - `notes/`
    47	   - `tools/`
    48	
    49	2. Confirm that this prompt file exists and can be read.
    50	
    51	3. Create a short report in `notes/` that verifies the pipeline is working.
    52	
    53	## Report requirements
    54	Create a markdown note in `notes/` whose filename includes:
    55	- `001_smoke_test_pipeline`
    56	- a success or fail marker
    57	- a timestamp
    58	
    59	Example:
    60	- `001_smoke_test_pipeline__SUCCESS__YYYYMMDD_HHMMSS.md`
    61	
    62	## Report contents
    63	The report should include:
    64	- prompt filename
    65	- timestamp
    66	- status
    67	- whether the expected folders were found
    68	- whether the prompt file was readable
    69	- a short summary stating whether the pipeline appears functional
    70	
    71	## Constraints
    72	- Do not make unrelated repo changes
    73	- Keep the report short and clear
    74	- If something is missing, mark the run as FAIL and explain what is missing
    75	```
    76	
    77	## Codex Final Output
    78	
    79	Verified successfully. The required folders exist, `codex_prompts/001_smoke_test_pipeline.md` is readable, and I wrote the report to [001_smoke_test_pipeline__SUCCESS__20260415_184932.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md).
    80	
    81	Status: `SUCCESS`
    82	
    83	## Stderr
    84	
    85	```text
    86	OpenAI Codex v0.120.0 (research preview)
    87	--------
    88	workdir: /data/git/handson-ml3
    89	model: gpt-5.4
    90	provider: openai
    91	approval: never
    92	sandbox: workspace-write [workdir, /tmp, $TMPDIR, /home/jevert/.codex/memories]
    93	reasoning effort: none
    94	reasoning summaries: none
    95	session id: 019d938c-9caa-7290-8f0d-bcf51bfb9312
    96	--------
    97	user
    98	# 001 Smoke Test Pipeline
    99	
   100	Your task is to verify that the prompt workflow scaffolding exists and that the notes pipeline can record a result.
   101	
   102	## Goals
   103	1. Confirm that the following folders exist in the repository:
   104	   - `codex_prompts/`
   105	   - `notes/`
   106	   - `tools/`
   107	
   108	2. Confirm that this prompt file exists and can be read.
   109	
   110	3. Create a short report in `notes/` that verifies the pipeline is working.
   111	
   112	## Report requirements
   113	Create a markdown note in `notes/` whose filename includes:
   114	- `001_smoke_test_pipeline`
   115	- a success or fail marker
   116	- a timestamp
   117	
   118	Example:
   119	- `001_smoke_test_pipeline__SUCCESS__YYYYMMDD_HHMMSS.md`
   120	

exec
/bin/bash -lc "nl -ba notes/001_smoke_test_pipeline__20260415_233343.md | sed -n '1,120p'" in /data/git/handson-ml3
 succeeded in 0ms:
     1	# 001_smoke_test_pipeline__20260415_233343
     2	
     3	- run_id: `001_smoke_test_pipeline__20260415_233343`
     4	- prompt_file: `codex_prompts/001_smoke_test_pipeline.md`
     5	- prompt_stem: `001_smoke_test_pipeline`
     6	- started_at_utc: `20260415_233343`
     7	
     8	## Execution Facts
     9	
    10	- execution_status: `EXECUTION_FAILED`
    11	- finished_at_utc: `20260415_233343`
    12	- runner: `tools/codex/run_prompt.py`
    13	- return_code: `1`
    14	- retry_of_run_id:
    15	
    16	## Review Facts
    17	
    18	- review_status: `UNREVIEWED`
    19	- review_summary:
    20	- reviewed_by:
    21	- reviewed_at_utc:
    22	
    23	## Failure Analysis
    24	
    25	- failure_type:
    26	- failure_symptom:
    27	- likely_cause:
    28	- recommended_next_action:
    29	
    30	## Resource / Cost Facts
    31	
    32	- elapsed_seconds: `0.559`
    33	- final_output_char_count: `0`
    34	- stderr_char_count: `354`
    35	
    36	## Prompt Text
    37	
    38	```md
    39	# 001 Smoke Test Pipeline
    40	
    41	Your task is to verify that the prompt workflow scaffolding exists and that the notes pipeline can record a result.
    42	
    43	## Goals
    44	1. Confirm that the following folders exist in the repository:
    45	   - `codex_prompts/`
    46	   - `notes/`
    47	   - `tools/`
    48	
    49	2. Confirm that this prompt file exists and can be read.
    50	
    51	3. Create a short report in `notes/` that verifies the pipeline is working.
    52	
    53	## Report requirements
    54	Create a markdown note in `notes/` whose filename includes:
    55	- `001_smoke_test_pipeline`
    56	- a success or fail marker
    57	- a timestamp
    58	
    59	Example:
    60	- `001_smoke_test_pipeline__SUCCESS__YYYYMMDD_HHMMSS.md`
    61	
    62	## Report contents
    63	The report should include:
    64	- prompt filename
    65	- timestamp
    66	- status
    67	- whether the expected folders were found
    68	- whether the prompt file was readable
    69	- a short summary stating whether the pipeline appears functional
    70	
    71	## Constraints
    72	- Do not make unrelated repo changes
    73	- Keep the report short and clear
    74	- If something is missing, mark the run as FAIL and explain what is missing
    75	```
    76	
    77	## Codex Final Output
    78	
    79	*No output captured.*
    80	
    81	## Stderr
    82	
    83	```text
    84	WARNING: proceeding, even though we could not update PATH: Read-only file system (os error 30)
    85	2026-04-15T23:33:43.693139Z ERROR codex_core::codex: Failed to create session: Read-only file system (os error 30)
    86	Error: thread/start: thread/start failed: error creating thread: Fatal error: Failed to initialize session: Read-only file system (os error 30)
    87	```

exec
/bin/bash -lc "nl -ba tools/Codex_Prompt_Workflow_Architecture.md | sed -n '268,320p'" in /data/git/handson-ml3
 succeeded in 0ms:
   268	
   269	### 6. Note parsing
   270	
   271	* extract prompt name, success/fail state, and timestamp from note filenames
   272	* turn note filenames into structured metadata
   273	
   274	### 7. Status reconstruction
   275	
   276	* map prompts to matching notes
   277	* determine whether each prompt is:
   278	  * UNRUN
   279	  * SUCCESS
   280	  * FAIL
   281	* pick the latest note when there are multiple notes for one prompt
   282	
   283	### 8. Prompt selection
   284	
   285	* find the next unrun prompt
   286	* find the first failed prompt
   287	* find a prompt by numeric prefix, base name, or full filename
   288	
   289	### 9. Prompt display
   290	
   291	* print a selected prompt to the terminal
   292	
   293	### 10. Retry context assembly
   294	
   295	* find the latest failed note for a prompt
   296	* display the original prompt plus the latest failed note
   297	* present instructions for retrying
   298	
   299	### 11. Note writing
   300	
   301	* create timestamped note filenames
   302	* write notes in markdown
   303	* optionally include the previous note as context
   304	
   305	### 12. Manual status marking
   306	
   307	* let the user record a success or failure
   308	* attach summary/details text to the note
   309	
   310	### 13. Command-line interface
   311	
   312	* parse subcommands like `list`, `next`, `show`, `mark`, `retry-failed`
   313	* route commands to the right behaviors
   314	
   315	### 14. Terminal presentation
   316	
   317	* format output for human readability
   318	* present status tables and retry blocks
   319	
   320	This decomposition is still useful, but V1 should not treat every decomposed responsibility as something that must be built now.

codex
I have enough to write. The recommendation is converging on operational support around the existing runner and review helper, not broader refactors: the biggest remaining friction is finding what to review next and deciding what is actually ready to run.
exec
/bin/bash -lc 'date -u +%Y%m%d_%H%M%S' in /data/git/handson-ml3
 succeeded in 0ms:
20260416_000819

The repo review is complete. I’m writing the two timestamped notes now, with options ranked around bounded operational improvements rather than refactors.
  ✓ Read the specified design docs, workflow code, and recent notes to map the current V1 pipeline state
  ✓ Synthesize working pieces, seams, and bounded next-step options grounded in repo evidence
  → Write the two requested notes under notes/ with timestamped filenames
codex
I’m creating the two requested artifacts under `notes/` now. The report will stay evidence-first and will keep the option set bounded to what the repo actually supports today.
apply patch
patch: completed
/data/git/handson-ml3/notes/012_v1_pipeline_options_review__20260416_000819.md
diff --git a/notes/012_v1_pipeline_options_review__20260416_000819.md b/notes/012_v1_pipeline_options_review__20260416_000819.md
new file mode 100644
index 0000000000000000000000000000000000000000..0f6cc92d9ee3b3f247d902d8d7eb7c676a696429
--- /dev/null
+++ b/notes/012_v1_pipeline_options_review__20260416_000819.md
@@ -0,0 +1,188 @@
+# 012 V1 Pipeline Options Review
+
+## Short Summary Of Current Pipeline Maturity
+
+The V1 pipeline is now a usable thin slice, not just a design packet.
+
+It can:
+
+- execute one prompt through `tools/codex/run_prompt.py`
+- write one durable V1 execution record in `notes/`
+- preserve execution facts separately from review facts
+- accept manual review write-back through `tools/codex/review_run.py`
+
+That means the core V1 artifact boundary is real and operationally demonstrable. The main remaining weakness is not missing architecture. It is missing operational support around repeated use: finding what needs review, deciding what is actually ready to run next, and keeping the docs aligned with the now-real implementation.
+
+## What Is Working Now
+
+### 1. Execution path
+
+- Prompt resolution exists and is conservative. `run_prompt.py` preserves the same direct-path, `codex_prompts/`, and unique-prefix lookup behavior as the bridge runner baseline. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:31).
+- Codex invocation is still thin and inspectable. The runner shells out through `codex exec -C <repo_root> --output-last-message <tempfile> -` and captures return code, final output, and stderr. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:74).
+- Execution-record creation exists and matches the intended V1 section order. The runner writes header, execution facts, review facts, failure analysis, resource facts, prompt text, final output, and stderr into one markdown file. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:97).
+- Resource and failure evidence capture exists in the actual record body. The runner records `return_code`, `elapsed_seconds`, output/stderr character counts, and full stderr text when present. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:144) and the failed record [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:8).
+- The separation between execution outcome and acceptance outcome is implemented correctly. New records start at `review_status: UNREVIEWED` regardless of `EXECUTED` vs `EXECUTION_FAILED`. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:130).
+- The failure path is demonstrated, not hypothetical. The failed smoke-test run preserved the read-only/session-start stderr cleanly while still producing a reviewable record. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).
+
+### 2. Review path
+
+- Manual review write-back exists as a separate thin companion tool. `review_run.py` updates an existing record in place rather than creating sidecars or replacing the record. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:57).
+- Record structure preservation is strong. The helper validates section order and required field lines before writing, then replaces only the targeted review fields. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:122) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:140).
+- The helper is consistent with the V1 review gate. It only allows `ACCEPTED` or `REJECTED`, auto-fills `reviewed_at_utc`, and limits failure-analysis updates to rejected runs. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:12) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:148).
+- Review is operationally usable now. The accepted smoke-test record shows the exact intended lifecycle: executed record first, then in-place write-back to `ACCEPTED`. See [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md:16).
+
+### 3. Workflow usability
+
+- The thin V1 flow is practical for single reviewed runs:
+  - run prompt
+  - inspect record
+  - apply manual review
+- The repo already has evidence of both important paths:
+  - successful execution plus accepted review
+  - failed execution artifact with preserved stderr
+- The implementation stayed within the intended V1 boundary and avoided premature platform growth.
+
+## Seams And Gaps That Remain
+
+### 1. Conservative queue progression is still architectural, not operational
+
+The architecture says only an accepted reviewed run should release the next prompt, but there is no helper yet that actually answers:
+
+- what is the latest run for prompt N
+- whether it is still `UNREVIEWED`
+- whether the next prompt is ready
+
+This is the biggest repeated-use seam. The design expects it, but the current repo still leaves the decision manual.
+
+### 2. Review is usable, but review discovery is manual
+
+`review_run.py` can update a chosen record, but nothing helps a human find:
+
+- all unreviewed records
+- the latest record for a prompt
+- the current review backlog
+
+That becomes awkward as `notes/` grows.
+
+### 3. Stable identity is mostly right, but not perfectly aligned with the current doc
+
+The execution-record artifact says the stable V1 `run_id` is `<prompt_stem>__<started_at_utc>`, but `run_prompt.py` adds `__2`, `__3`, etc. on same-second collisions. See [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:37) versus [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:59).
+
+This is a small practical improvement in code, but it is still a spec mismatch.
+
+### 4. Documentation drift exists
+
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md` still targets `baby_run_prompt.py` as the implementation target and still names the runner as `tools/codex/baby_run_prompt.py`. See [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:5), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:17), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:156), and [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:252).
+- The design packet now implicitly assumes a review write-back step, but the architecture document still describes conservative queue progression as part of the V1 shape without a concrete implementation artifact for that responsibility. See [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:161) and [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:215).
+
+### 5. Repeated-use confidence is under-supported
+
+The runner and review helper are both small and clear, but there is no explicit fixture-based validation or script-level regression harness around:
+
+- record section ordering
+- same-record write-back safety
+- rejection-field behavior
+- collision behavior
+
+The code is simple enough that this is not a correctness emergency, but for repeated operational use it is a credible next bounded improvement.
+
+### 6. Rejected-run and retry linkage are still mostly placeholders
+
+The record shape includes `retry_of_run_id` and manual failure-analysis fields, but the current tools do not help a reviewer connect a rejected run to a retry decision or derive the next retry target.
+
+### 7. Environment fragility still appears in the repo history
+
+The failed smoke-test run shows that Codex session startup can fail before task work begins. That is not a runner-design flaw, but it does mean the workflow still benefits from a small amount of environment-oriented diagnosis or runbook support when repeated use hits startup failures. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).
+
+## Documentation Alignment Assessment
+
+Overall alignment is good at the architectural level and good enough at the artifact level.
+
+What still drifts:
+
+- implementation target naming drift: the bridge-runner spec still speaks in terms of `baby_run_prompt.py`, while the actual V1 runner is `run_prompt.py`
+- runner identity drift: the spec examples still encode `tools/codex/baby_run_prompt.py` while the real records encode `tools/codex/run_prompt.py`
+- stable-identity drift: the spec describes a strict no-suffix `run_id`, while code uses a pragmatic collision suffix
+
+These are small mismatches, but they are no longer theoretical because the repo now contains real V1 records and a real review helper.
+
+## Realistic Next Bounded Options
+
+### Option 1. Add a queue-readiness checker
+
+- What it would build or improve:
+  - A small helper that determines whether the next prompt is ready based on the latest execution record and whether that record is `ACCEPTED`.
+- Why it matters:
+  - This closes the biggest gap between the architecture and current operational reality.
+  - It makes the V1 rule "only accepted review releases the next prompt" actually inspectable in use rather than purely manual.
+- Expected risk level: `low`
+- Expected payoff level: `high`
+
+### Option 2. Add a review backlog / unreviewed-run lister
+
+- What it would build or improve:
+  - A small helper that scans V1 execution records in `notes/` and lists runs still at `UNREVIEWED`, plus the latest record per prompt.
+- Why it matters:
+  - Review write-back already works, but finding what needs review is still manual.
+  - This directly improves repeated operational use without broadening the workflow.
+- Expected risk level: `low`
+- Expected payoff level: `high`
+
+### Option 3. Add lightweight contract validation for the V1 scripts
+
+- What it would build or improve:
+  - A small test or fixture harness around `run_prompt.py` and `review_run.py` that locks in the record shape, allowed review transitions, and in-place write-back behavior.
+- Why it matters:
+  - The workflow now depends on markdown field stability.
+  - A small regression harness would reduce silent drift while keeping the implementation thin.
+- Expected risk level: `low`
+- Expected payoff level: `medium`
+
+### Option 4. Align the V1 design docs to the implemented runner and helper
+
+- What it would build or improve:
+  - A cleanup pass on the architecture/spec packet so it reflects `run_prompt.py`, the existing review write-back helper, and the real collision rule.
+- Why it matters:
+  - The current doc drift is small but inspectable.
+  - Fixing it would remove avoidable ambiguity before more workflow support is added.
+- Expected risk level: `low`
+- Expected payoff level: `medium`
+
+### Option 5. Add rejected-run / retry-linkage support
+
+- What it would build or improve:
+  - A small helper or convention for linking a retry run to `retry_of_run_id` and surfacing the latest rejected run for a prompt.
+- Why it matters:
+  - The record model already reserves space for retries, but the workflow does not yet help humans use it consistently.
+  - This would make failure analysis more actionable without introducing a larger retry engine.
+- Expected risk level: `medium`
+- Expected payoff level: `medium`
+
+### Option 6. Add a minimal environment-diagnosis/runbook note or helper
+
+- What it would build or improve:
+  - A bounded way to capture or explain Codex startup failures such as read-only session initialization issues.
+- Why it matters:
+  - The repo contains real evidence of environment-level failure before task execution.
+  - A small runbook or diagnostic helper could reduce confusion when failures are not prompt-related.
+- Expected risk level: `low`
+- Expected payoff level: `low`
+
+## Assessment Of The Current V1 Flow
+
+The current V1 flow is now operationally usable for intentional, reviewed single-step work. It is no longer just a design direction.
+
+What still feels awkward for regular use:
+
+- finding the latest relevant run by hand in `notes/`
+- identifying what is awaiting review
+- deciding whether the next prompt is actually released
+- keeping the docs and record semantics in sync as the thin tools stabilize
+
+What is missing if this is going to be used regularly:
+
+- one small readiness/checking helper
+- one small review-discovery helper
+- one small validation or contract layer to protect the record format
+
+That is enough to make the next step clear: the strongest next work is around operational support for repeated reviewed use, not workflow expansion.

diff --git a/notes/012_v1_pipeline_options_review__20260416_000819.md b/notes/012_v1_pipeline_options_review__20260416_000819.md
new file mode 100644
index 0000000000000000000000000000000000000000..0f6cc92d9ee3b3f247d902d8d7eb7c676a696429
--- /dev/null
+++ b/notes/012_v1_pipeline_options_review__20260416_000819.md
@@ -0,0 +1,188 @@
+# 012 V1 Pipeline Options Review
+
+## Short Summary Of Current Pipeline Maturity
+
+The V1 pipeline is now a usable thin slice, not just a design packet.
+
+It can:
+
+- execute one prompt through `tools/codex/run_prompt.py`
+- write one durable V1 execution record in `notes/`
+- preserve execution facts separately from review facts
+- accept manual review write-back through `tools/codex/review_run.py`
+
+That means the core V1 artifact boundary is real and operationally demonstrable. The main remaining weakness is not missing architecture. It is missing operational support around repeated use: finding what needs review, deciding what is actually ready to run next, and keeping the docs aligned with the now-real implementation.
+
+## What Is Working Now
+
+### 1. Execution path
+
+- Prompt resolution exists and is conservative. `run_prompt.py` preserves the same direct-path, `codex_prompts/`, and unique-prefix lookup behavior as the bridge runner baseline. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:31).
+- Codex invocation is still thin and inspectable. The runner shells out through `codex exec -C <repo_root> --output-last-message <tempfile> -` and captures return code, final output, and stderr. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:74).
+- Execution-record creation exists and matches the intended V1 section order. The runner writes header, execution facts, review facts, failure analysis, resource facts, prompt text, final output, and stderr into one markdown file. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:97).
+- Resource and failure evidence capture exists in the actual record body. The runner records `return_code`, `elapsed_seconds`, output/stderr character counts, and full stderr text when present. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:144) and the failed record [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:8).
+- The separation between execution outcome and acceptance outcome is implemented correctly. New records start at `review_status: UNREVIEWED` regardless of `EXECUTED` vs `EXECUTION_FAILED`. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:130).
+- The failure path is demonstrated, not hypothetical. The failed smoke-test run preserved the read-only/session-start stderr cleanly while still producing a reviewable record. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).
+
+### 2. Review path
+
+- Manual review write-back exists as a separate thin companion tool. `review_run.py` updates an existing record in place rather than creating sidecars or replacing the record. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:57).
+- Record structure preservation is strong. The helper validates section order and required field lines before writing, then replaces only the targeted review fields. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:122) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:140).
+- The helper is consistent with the V1 review gate. It only allows `ACCEPTED` or `REJECTED`, auto-fills `reviewed_at_utc`, and limits failure-analysis updates to rejected runs. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:12) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:148).
+- Review is operationally usable now. The accepted smoke-test record shows the exact intended lifecycle: executed record first, then in-place write-back to `ACCEPTED`. See [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md:16).
+
+### 3. Workflow usability
+
+- The thin V1 flow is practical for single reviewed runs:
+  - run prompt
+  - inspect record
+  - apply manual review
+- The repo already has evidence of both important paths:
+  - successful execution plus accepted review
+  - failed execution artifact with preserved stderr
+- The implementation stayed within the intended V1 boundary and avoided premature platform growth.
+
+## Seams And Gaps That Remain
+
+### 1. Conservative queue progression is still architectural, not operational
+
+The architecture says only an accepted reviewed run should release the next prompt, but there is no helper yet that actually answers:
+
+- what is the latest run for prompt N
+- whether it is still `UNREVIEWED`
+- whether the next prompt is ready
+
+This is the biggest repeated-use seam. The design expects it, but the current repo still leaves the decision manual.
+
+### 2. Review is usable, but review discovery is manual
+
+`review_run.py` can update a chosen record, but nothing helps a human find:
+
+- all unreviewed records
+- the latest record for a prompt
+- the current review backlog
+
+That becomes awkward as `notes/` grows.
+
+### 3. Stable identity is mostly right, but not perfectly aligned with the current doc
+
+The execution-record artifact says the stable V1 `run_id` is `<prompt_stem>__<started_at_utc>`, but `run_prompt.py` adds `__2`, `__3`, etc. on same-second collisions. See [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:37) versus [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:59).
+
+This is a small practical improvement in code, but it is still a spec mismatch.
+
+### 4. Documentation drift exists
+
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md` still targets `baby_run_prompt.py` as the implementation target and still names the runner as `tools/codex/baby_run_prompt.py`. See [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:5), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:17), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:156), and [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:252).
+- The design packet now implicitly assumes a review write-back step, but the architecture document still describes conservative queue progression as part of the V1 shape without a concrete implementation artifact for that responsibility. See [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:161) and [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:215).
+
+### 5. Repeated-use confidence is under-supported
+
+The runner and review helper are both small and clear, but there is no explicit fixture-based validation or script-level regression harness around:
+
+- record section ordering
+- same-record write-back safety
+- rejection-field behavior
+- collision behavior
+
+The code is simple enough that this is not a correctness emergency, but for repeated operational use it is a credible next bounded improvement.
+
+### 6. Rejected-run and retry linkage are still mostly placeholders
+
+The record shape includes `retry_of_run_id` and manual failure-analysis fields, but the current tools do not help a reviewer connect a rejected run to a retry decision or derive the next retry target.
+
+### 7. Environment fragility still appears in the repo history
+
+The failed smoke-test run shows that Codex session startup can fail before task work begins. That is not a runner-design flaw, but it does mean the workflow still benefits from a small amount of environment-oriented diagnosis or runbook support when repeated use hits startup failures. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).
+
+## Documentation Alignment Assessment
+
+Overall alignment is good at the architectural level and good enough at the artifact level.
+
+What still drifts:
+
+- implementation target naming drift: the bridge-runner spec still speaks in terms of `baby_run_prompt.py`, while the actual V1 runner is `run_prompt.py`
+- runner identity drift: the spec examples still encode `tools/codex/baby_run_prompt.py` while the real records encode `tools/codex/run_prompt.py`
+- stable-identity drift: the spec describes a strict no-suffix `run_id`, while code uses a pragmatic collision suffix
+
+These are small mismatches, but they are no longer theoretical because the repo now contains real V1 records and a real review helper.
+
+## Realistic Next Bounded Options
+
+### Option 1. Add a queue-readiness checker
+
+- What it would build or improve:
+  - A small helper that determines whether the next prompt is ready based on the latest execution record and whether that record is `ACCEPTED`.
+- Why it matters:
+  - This closes the biggest gap between the architecture and current operational reality.
+  - It makes the V1 rule "only accepted review releases the next prompt" actually inspectable in use rather than purely manual.
+- Expected risk level: `low`
+- Expected payoff level: `high`
+
+### Option 2. Add a review backlog / unreviewed-run lister
+
+- What it would build or improve:
+  - A small helper that scans V1 execution records in `notes/` and lists runs still at `UNREVIEWED`, plus the latest record per prompt.
+- Why it matters:
+  - Review write-back already works, but finding what needs review is still manual.
+  - This directly improves repeated operational use without broadening the workflow.
+- Expected risk level: `low`
+- Expected payoff level: `high`
+
+### Option 3. Add lightweight contract validation for the V1 scripts
+
+- What it would build or improve:
+  - A small test or fixture harness around `run_prompt.py` and `review_run.py` that locks in the record shape, allowed review transitions, and in-place write-back behavior.
+- Why it matters:
+  - The workflow now depends on markdown field stability.
+  - A small regression harness would reduce silent drift while keeping the implementation thin.
+- Expected risk level: `low`
+- Expected payoff level: `medium`
+
+### Option 4. Align the V1 design docs to the implemented runner and helper
+
+- What it would build or improve:
+  - A cleanup pass on the architecture/spec packet so it reflects `run_prompt.py`, the existing review write-back helper, and the real collision rule.
+- Why it matters:
+  - The current doc drift is small but inspectable.
+  - Fixing it would remove avoidable ambiguity before more workflow support is added.
+- Expected risk level: `low`
+- Expected payoff level: `medium`
+
+### Option 5. Add rejected-run / retry-linkage support
+
+- What it would build or improve:
+  - A small helper or convention for linking a retry run to `retry_of_run_id` and surfacing the latest rejected run for a prompt.
+- Why it matters:
+  - The record model already reserves space for retries, but the workflow does not yet help humans use it consistently.
+  - This would make failure analysis more actionable without introducing a larger retry engine.
+- Expected risk level: `medium`
+- Expected payoff level: `medium`
+
+### Option 6. Add a minimal environment-diagnosis/runbook note or helper
+
+- What it would build or improve:
+  - A bounded way to capture or explain Codex startup failures such as read-only session initialization issues.
+- Why it matters:
+  - The repo contains real evidence of environment-level failure before task execution.
+  - A small runbook or diagnostic helper could reduce confusion when failures are not prompt-related.
+- Expected risk level: `low`
+- Expected payoff level: `low`
+
+## Assessment Of The Current V1 Flow
+
+The current V1 flow is now operationally usable for intentional, reviewed single-step work. It is no longer just a design direction.
+
+What still feels awkward for regular use:
+
+- finding the latest relevant run by hand in `notes/`
+- identifying what is awaiting review
+- deciding whether the next prompt is actually released
+- keeping the docs and record semantics in sync as the thin tools stabilize
+
+What is missing if this is going to be used regularly:
+
+- one small readiness/checking helper
+- one small review-discovery helper
+- one small validation or contract layer to protect the record format
+
+That is enough to make the next step clear: the strongest next work is around operational support for repeated reviewed use, not workflow expansion.

apply patch
patch: completed
/data/git/handson-ml3/notes/012_top_three_next_options__20260416_000819.md
diff --git a/notes/012_top_three_next_options__20260416_000819.md b/notes/012_top_three_next_options__20260416_000819.md
new file mode 100644
index 0000000000000000000000000000000000000000..c9c7fb98ff7eeabfb54bbcb6485b3b8729de70e4
--- /dev/null
+++ b/notes/012_top_three_next_options__20260416_000819.md
@@ -0,0 +1,63 @@
+# 012 Top Three Next Options
+
+## Ranked Top Three
+
+### 1. Add a queue-readiness checker
+
+Why it made the top three:
+
+- It closes the most important remaining gap between the V1 architecture and the actual toolchain.
+- The design packet says only `ACCEPTED` should release the next prompt, but the repo does not yet provide a concrete way to answer that question.
+- It is bounded, low-risk, and directly improves repeated operational use.
+
+### 2. Add a review backlog / unreviewed-run lister
+
+Why it made the top three:
+
+- Review write-back now works, but locating records that still need review is manual.
+- This is the next most obvious operational friction once more records accumulate in `notes/`.
+- It improves usability without expanding the workflow into a broader engine.
+
+### 3. Add lightweight contract validation for `run_prompt.py` and `review_run.py`
+
+Why it made the top three:
+
+- The current V1 flow depends on exact markdown structure and field stability.
+- A small regression harness would protect the record contract before more helpers are built on top of it.
+- It is a clean bounded step that reduces accidental drift risk.
+
+## Which One Should Happen Next
+
+The next step should be: `Add a queue-readiness checker`.
+
+Reason:
+
+- It addresses the highest-value remaining seam in the current V1 workflow.
+- It is directly demanded by the architecture and review-gate docs.
+- It reduces the current ambiguity around whether the next prompt may proceed, which is the most important missing operational behavior.
+
+## What Should Explicitly Wait
+
+These should wait until after the top option:
+
+- broader workflow engines or status systems
+- retry orchestration beyond minimal linkage
+- multi-module runner refactors
+- dashboards or aggregated reporting
+- broad platform expansion
+
+These can also wait briefly behind the top option, but remain good bounded follow-ups:
+
+- doc-alignment cleanup for `V1_Bridge_Runner_Change_Spec.md`
+- rejected-run / retry-linkage support
+- environment-focused diagnosis tooling
+
+## Practical Recommendation
+
+Do the next step in this order:
+
+1. queue-readiness checker
+2. review backlog / unreviewed-run lister
+3. lightweight contract validation
+
+That sequence best strengthens the current repo for repeated reviewed use while staying inside the existing V1 boundary.
diff --git a/notes/012_v1_pipeline_options_review__20260416_000819.md b/notes/012_v1_pipeline_options_review__20260416_000819.md
new file mode 100644
index 0000000000000000000000000000000000000000..0f6cc92d9ee3b3f247d902d8d7eb7c676a696429
--- /dev/null
+++ b/notes/012_v1_pipeline_options_review__20260416_000819.md
@@ -0,0 +1,188 @@
+# 012 V1 Pipeline Options Review
+
+## Short Summary Of Current Pipeline Maturity
+
+The V1 pipeline is now a usable thin slice, not just a design packet.
+
+It can:
+
+- execute one prompt through `tools/codex/run_prompt.py`
+- write one durable V1 execution record in `notes/`
+- preserve execution facts separately from review facts
+- accept manual review write-back through `tools/codex/review_run.py`
+
+That means the core V1 artifact boundary is real and operationally demonstrable. The main remaining weakness is not missing architecture. It is missing operational support around repeated use: finding what needs review, deciding what is actually ready to run next, and keeping the docs aligned with the now-real implementation.
+
+## What Is Working Now
+
+### 1. Execution path
+
+- Prompt resolution exists and is conservative. `run_prompt.py` preserves the same direct-path, `codex_prompts/`, and unique-prefix lookup behavior as the bridge runner baseline. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:31).
+- Codex invocation is still thin and inspectable. The runner shells out through `codex exec -C <repo_root> --output-last-message <tempfile> -` and captures return code, final output, and stderr. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:74).
+- Execution-record creation exists and matches the intended V1 section order. The runner writes header, execution facts, review facts, failure analysis, resource facts, prompt text, final output, and stderr into one markdown file. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:97).
+- Resource and failure evidence capture exists in the actual record body. The runner records `return_code`, `elapsed_seconds`, output/stderr character counts, and full stderr text when present. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:144) and the failed record [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:8).
+- The separation between execution outcome and acceptance outcome is implemented correctly. New records start at `review_status: UNREVIEWED` regardless of `EXECUTED` vs `EXECUTION_FAILED`. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:130).
+- The failure path is demonstrated, not hypothetical. The failed smoke-test run preserved the read-only/session-start stderr cleanly while still producing a reviewable record. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).
+
+### 2. Review path
+
+- Manual review write-back exists as a separate thin companion tool. `review_run.py` updates an existing record in place rather than creating sidecars or replacing the record. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:57).
+- Record structure preservation is strong. The helper validates section order and required field lines before writing, then replaces only the targeted review fields. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:122) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:140).
+- The helper is consistent with the V1 review gate. It only allows `ACCEPTED` or `REJECTED`, auto-fills `reviewed_at_utc`, and limits failure-analysis updates to rejected runs. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:12) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:148).
+- Review is operationally usable now. The accepted smoke-test record shows the exact intended lifecycle: executed record first, then in-place write-back to `ACCEPTED`. See [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md:16).
+
+### 3. Workflow usability
+
+- The thin V1 flow is practical for single reviewed runs:
+  - run prompt
+  - inspect record
+  - apply manual review
+- The repo already has evidence of both important paths:
+  - successful execution plus accepted review
+  - failed execution artifact with preserved stderr
+- The implementation stayed within the intended V1 boundary and avoided premature platform growth.
+
+## Seams And Gaps That Remain
+
+### 1. Conservative queue progression is still architectural, not operational
+
+The architecture says only an accepted reviewed run should release the next prompt, but there is no helper yet that actually answers:
+
+- what is the latest run for prompt N
+- whether it is still `UNREVIEWED`
+- whether the next prompt is ready
+
+This is the biggest repeated-use seam. The design expects it, but the current repo still leaves the decision manual.
+
+### 2. Review is usable, but review discovery is manual
+
+`review_run.py` can update a chosen record, but nothing helps a human find:
+
+- all unreviewed records
+- the latest record for a prompt
+- the current review backlog
+
+That becomes awkward as `notes/` grows.
+
+### 3. Stable identity is mostly right, but not perfectly aligned with the current doc
+
+The execution-record artifact says the stable V1 `run_id` is `<prompt_stem>__<started_at_utc>`, but `run_prompt.py` adds `__2`, `__3`, etc. on same-second collisions. See [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:37) versus [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:59).
+
+This is a small practical improvement in code, but it is still a spec mismatch.
+
+### 4. Documentation drift exists
+
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md` still targets `baby_run_prompt.py` as the implementation target and still names the runner as `tools/codex/baby_run_prompt.py`. See [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:5), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:17), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:156), and [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:252).
+- The design packet now implicitly assumes a review write-back step, but the architecture document still describes conservative queue progression as part of the V1 shape without a concrete implementation artifact for that responsibility. See [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:161) and [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:215).
+
+### 5. Repeated-use confidence is under-supported
+
+The runner and review helper are both small and clear, but there is no explicit fixture-based validation or script-level regression harness around:
+
+- record section ordering
+- same-record write-back safety
+- rejection-field behavior
+- collision behavior
+
+The code is simple enough that this is not a correctness emergency, but for repeated operational use it is a credible next bounded improvement.
+
+### 6. Rejected-run and retry linkage are still mostly placeholders
+
+The record shape includes `retry_of_run_id` and manual failure-analysis fields, but the current tools do not help a reviewer connect a rejected run to a retry decision or derive the next retry target.
+
+### 7. Environment fragility still appears in the repo history
+
+The failed smoke-test run shows that Codex session startup can fail before task work begins. That is not a runner-design flaw, but it does mean the workflow still benefits from a small amount of environment-oriented diagnosis or runbook support when repeated use hits startup failures. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).
+
+## Documentation Alignment Assessment
+
+Overall alignment is good at the architectural level and good enough at the artifact level.
+
+What still drifts:
+
+- implementation target naming drift: the bridge-runner spec still speaks in terms of `baby_run_prompt.py`, while the actual V1 runner is `run_prompt.py`
+- runner identity drift: the spec examples still encode `tools/codex/baby_run_prompt.py` while the real records encode `tools/codex/run_prompt.py`
+- stable-identity drift: the spec describes a strict no-suffix `run_id`, while code uses a pragmatic collision suffix
+
+These are small mismatches, but they are no longer theoretical because the repo now contains real V1 records and a real review helper.
+
+## Realistic Next Bounded Options
+
+### Option 1. Add a queue-readiness checker
+
+- What it would build or improve:
+  - A small helper that determines whether the next prompt is ready based on the latest execution record and whether that record is `ACCEPTED`.
+- Why it matters:
+  - This closes the biggest gap between the architecture and current operational reality.
+  - It makes the V1 rule "only accepted review releases the next prompt" actually inspectable in use rather than purely manual.
+- Expected risk level: `low`
+- Expected payoff level: `high`
+
+### Option 2. Add a review backlog / unreviewed-run lister
+
+- What it would build or improve:
+  - A small helper that scans V1 execution records in `notes/` and lists runs still at `UNREVIEWED`, plus the latest record per prompt.
+- Why it matters:
+  - Review write-back already works, but finding what needs review is still manual.
+  - This directly improves repeated operational use without broadening the workflow.
+- Expected risk level: `low`
+- Expected payoff level: `high`
+
+### Option 3. Add lightweight contract validation for the V1 scripts
+
+- What it would build or improve:
+  - A small test or fixture harness around `run_prompt.py` and `review_run.py` that locks in the record shape, allowed review transitions, and in-place write-back behavior.
+- Why it matters:
+  - The workflow now depends on markdown field stability.
+  - A small regression harness would reduce silent drift while keeping the implementation thin.
+- Expected risk level: `low`
+- Expected payoff level: `medium`
+
+### Option 4. Align the V1 design docs to the implemented runner and helper
+
+- What it would build or improve:
+  - A cleanup pass on the architecture/spec packet so it reflects `run_prompt.py`, the existing review write-back helper, and the real collision rule.
+- Why it matters:
+  - The current doc drift is small but inspectable.
+  - Fixing it would remove avoidable ambiguity before more workflow support is added.
+- Expected risk level: `low`
+- Expected payoff level: `medium`
+
+### Option 5. Add rejected-run / retry-linkage support
+
+- What it would build or improve:
+  - A small helper or convention for linking a retry run to `retry_of_run_id` and surfacing the latest rejected run for a prompt.
+- Why it matters:
+  - The record model already reserves space for retries, but the workflow does not yet help humans use it consistently.
+  - This would make failure analysis more actionable without introducing a larger retry engine.
+- Expected risk level: `medium`
+- Expected payoff level: `medium`
+
+### Option 6. Add a minimal environment-diagnosis/runbook note or helper
+
+- What it would build or improve:
+  - A bounded way to capture or explain Codex startup failures such as read-only session initialization issues.
+- Why it matters:
+  - The repo contains real evidence of environment-level failure before task execution.
+  - A small runbook or diagnostic helper could reduce confusion when failures are not prompt-related.
+- Expected risk level: `low`
+- Expected payoff level: `low`
+
+## Assessment Of The Current V1 Flow
+
+The current V1 flow is now operationally usable for intentional, reviewed single-step work. It is no longer just a design direction.
+
+What still feels awkward for regular use:
+
+- finding the latest relevant run by hand in `notes/`
+- identifying what is awaiting review
+- deciding whether the next prompt is actually released
+- keeping the docs and record semantics in sync as the thin tools stabilize
+
+What is missing if this is going to be used regularly:
+
+- one small readiness/checking helper
+- one small review-discovery helper
+- one small validation or contract layer to protect the record format
+
+That is enough to make the next step clear: the strongest next work is around operational support for repeated reviewed use, not workflow expansion.

diff --git a/notes/012_top_three_next_options__20260416_000819.md b/notes/012_top_three_next_options__20260416_000819.md
new file mode 100644
index 0000000000000000000000000000000000000000..c9c7fb98ff7eeabfb54bbcb6485b3b8729de70e4
--- /dev/null
+++ b/notes/012_top_three_next_options__20260416_000819.md
@@ -0,0 +1,63 @@
+# 012 Top Three Next Options
+
+## Ranked Top Three
+
+### 1. Add a queue-readiness checker
+
+Why it made the top three:
+
+- It closes the most important remaining gap between the V1 architecture and the actual toolchain.
+- The design packet says only `ACCEPTED` should release the next prompt, but the repo does not yet provide a concrete way to answer that question.
+- It is bounded, low-risk, and directly improves repeated operational use.
+
+### 2. Add a review backlog / unreviewed-run lister
+
+Why it made the top three:
+
+- Review write-back now works, but locating records that still need review is manual.
+- This is the next most obvious operational friction once more records accumulate in `notes/`.
+- It improves usability without expanding the workflow into a broader engine.
+
+### 3. Add lightweight contract validation for `run_prompt.py` and `review_run.py`
+
+Why it made the top three:
+
+- The current V1 flow depends on exact markdown structure and field stability.
+- A small regression harness would protect the record contract before more helpers are built on top of it.
+- It is a clean bounded step that reduces accidental drift risk.
+
+## Which One Should Happen Next
+
+The next step should be: `Add a queue-readiness checker`.
+
+Reason:
+
+- It addresses the highest-value remaining seam in the current V1 workflow.
+- It is directly demanded by the architecture and review-gate docs.
+- It reduces the current ambiguity around whether the next prompt may proceed, which is the most important missing operational behavior.
+
+## What Should Explicitly Wait
+
+These should wait until after the top option:
+
+- broader workflow engines or status systems
+- retry orchestration beyond minimal linkage
+- multi-module runner refactors
+- dashboards or aggregated reporting
+- broad platform expansion
+
+These can also wait briefly behind the top option, but remain good bounded follow-ups:
+
+- doc-alignment cleanup for `V1_Bridge_Runner_Change_Spec.md`
+- rejected-run / retry-linkage support
+- environment-focused diagnosis tooling
+
+## Practical Recommendation
+
+Do the next step in this order:
+
+1. queue-readiness checker
+2. review backlog / unreviewed-run lister
+3. lightweight contract validation
+
+That sequence best strengthens the current repo for repeated reviewed use while staying inside the existing V1 boundary.
diff --git a/notes/012_v1_pipeline_options_review__20260416_000819.md b/notes/012_v1_pipeline_options_review__20260416_000819.md
new file mode 100644
index 0000000000000000000000000000000000000000..0f6cc92d9ee3b3f247d902d8d7eb7c676a696429
--- /dev/null
+++ b/notes/012_v1_pipeline_options_review__20260416_000819.md
@@ -0,0 +1,188 @@
+# 012 V1 Pipeline Options Review
+
+## Short Summary Of Current Pipeline Maturity
+
+The V1 pipeline is now a usable thin slice, not just a design packet.
+
+It can:
+
+- execute one prompt through `tools/codex/run_prompt.py`
+- write one durable V1 execution record in `notes/`
+- preserve execution facts separately from review facts
+- accept manual review write-back through `tools/codex/review_run.py`
+
+That means the core V1 artifact boundary is real and operationally demonstrable. The main remaining weakness is not missing architecture. It is missing operational support around repeated use: finding what needs review, deciding what is actually ready to run next, and keeping the docs aligned with the now-real implementation.
+
+## What Is Working Now
+
+### 1. Execution path
+
+- Prompt resolution exists and is conservative. `run_prompt.py` preserves the same direct-path, `codex_prompts/`, and unique-prefix lookup behavior as the bridge runner baseline. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:31).
+- Codex invocation is still thin and inspectable. The runner shells out through `codex exec -C <repo_root> --output-last-message <tempfile> -` and captures return code, final output, and stderr. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:74).
+- Execution-record creation exists and matches the intended V1 section order. The runner writes header, execution facts, review facts, failure analysis, resource facts, prompt text, final output, and stderr into one markdown file. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:97).
+- Resource and failure evidence capture exists in the actual record body. The runner records `return_code`, `elapsed_seconds`, output/stderr character counts, and full stderr text when present. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:144) and the failed record [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:8).
+- The separation between execution outcome and acceptance outcome is implemented correctly. New records start at `review_status: UNREVIEWED` regardless of `EXECUTED` vs `EXECUTION_FAILED`. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:130).
+- The failure path is demonstrated, not hypothetical. The failed smoke-test run preserved the read-only/session-start stderr cleanly while still producing a reviewable record. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).
+
+### 2. Review path
+
+- Manual review write-back exists as a separate thin companion tool. `review_run.py` updates an existing record in place rather than creating sidecars or replacing the record. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:57).
+- Record structure preservation is strong. The helper validates section order and required field lines before writing, then replaces only the targeted review fields. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:122) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:140).
+- The helper is consistent with the V1 review gate. It only allows `ACCEPTED` or `REJECTED`, auto-fills `reviewed_at_utc`, and limits failure-analysis updates to rejected runs. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:12) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:148).
+- Review is operationally usable now. The accepted smoke-test record shows the exact intended lifecycle: executed record first, then in-place write-back to `ACCEPTED`. See [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md:16).
+
+### 3. Workflow usability
+
+- The thin V1 flow is practical for single reviewed runs:
+  - run prompt
+  - inspect record
+  - apply manual review
+- The repo already has evidence of both important paths:
+  - successful execution plus accepted review
+  - failed execution artifact with preserved stderr
+- The implementation stayed within the intended V1 boundary and avoided premature platform growth.
+
+## Seams And Gaps That Remain
+
+### 1. Conservative queue progression is still architectural, not operational
+
+The architecture says only an accepted reviewed run should release the next prompt, but there is no helper yet that actually answers:
+
+- what is the latest run for prompt N
+- whether it is still `UNREVIEWED`
+- whether the next prompt is ready
+
+This is the biggest repeated-use seam. The design expects it, but the current repo still leaves the decision manual.
+
+### 2. Review is usable, but review discovery is manual
+
+`review_run.py` can update a chosen record, but nothing helps a human find:
+
+- all unreviewed records
+- the latest record for a prompt
+- the current review backlog
+
+That becomes awkward as `notes/` grows.
+
+### 3. Stable identity is mostly right, but not perfectly aligned with the current doc
+
+The execution-record artifact says the stable V1 `run_id` is `<prompt_stem>__<started_at_utc>`, but `run_prompt.py` adds `__2`, `__3`, etc. on same-second collisions. See [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:37) versus [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:59).
+
+This is a small practical improvement in code, but it is still a spec mismatch.
+
+### 4. Documentation drift exists
+
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md` still targets `baby_run_prompt.py` as the implementation target and still names the runner as `tools/codex/baby_run_prompt.py`. See [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:5), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:17), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:156), and [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:252).
+- The design packet now implicitly assumes a review write-back step, but the architecture document still describes conservative queue progression as part of the V1 shape without a concrete implementation artifact for that responsibility. See [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:161) and [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:215).
+
+### 5. Repeated-use confidence is under-supported
+
+The runner and review helper are both small and clear, but there is no explicit fixture-based validation or script-level regression harness around:
+
+- record section ordering
+- same-record write-back safety
+- rejection-field behavior
+- collision behavior
+
+The code is simple enough that this is not a correctness emergency, but for repeated operational use it is a credible next bounded improvement.
+
+### 6. Rejected-run and retry linkage are still mostly placeholders
+
+The record shape includes `retry_of_run_id` and manual failure-analysis fields, but the current tools do not help a reviewer connect a rejected run to a retry decision or derive the next retry target.
+
+### 7. Environment fragility still appears in the repo history
+
+The failed smoke-test run shows that Codex session startup can fail before task work begins. That is not a runner-design flaw, but it does mean the workflow still benefits from a small amount of environment-oriented diagnosis or runbook support when repeated use hits startup failures. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).
+
+## Documentation Alignment Assessment
+
+Overall alignment is good at the architectural level and good enough at the artifact level.
+
+What still drifts:
+
+- implementation target naming drift: the bridge-runner spec still speaks in terms of `baby_run_prompt.py`, while the actual V1 runner is `run_prompt.py`
+- runner identity drift: the spec examples still encode `tools/codex/baby_run_prompt.py` while the real records encode `tools/codex/run_prompt.py`
+- stable-identity drift: the spec describes a strict no-suffix `run_id`, while code uses a pragmatic collision suffix
+
+These are small mismatches, but they are no longer theoretical because the repo now contains real V1 records and a real review helper.
+
+## Realistic Next Bounded Options
+
+### Option 1. Add a queue-readiness checker
+
+- What it would build or improve:
+  - A small helper that determines whether the next prompt is ready based on the latest execution record and whether that record is `ACCEPTED`.
+- Why it matters:
+  - This closes the biggest gap between the architecture and current operational reality.
+  - It makes the V1 rule "only accepted review releases the next prompt" actually inspectable in use rather than purely manual.
+- Expected risk level: `low`
+- Expected payoff level: `high`
+
+### Option 2. Add a review backlog / unreviewed-run lister
+
+- What it would build or improve:
+  - A small helper that scans V1 execution records in `notes/` and lists runs still at `UNREVIEWED`, plus the latest record per prompt.
+- Why it matters:
+  - Review write-back already works, but finding what needs review is still manual.
+  - This directly improves repeated operational use without broadening the workflow.
+- Expected risk level: `low`
+- Expected payoff level: `high`
+
+### Option 3. Add lightweight contract validation for the V1 scripts
+
+- What it would build or improve:
+  - A small test or fixture harness around `run_prompt.py` and `review_run.py` that locks in the record shape, allowed review transitions, and in-place write-back behavior.
+- Why it matters:
+  - The workflow now depends on markdown field stability.
+  - A small regression harness would reduce silent drift while keeping the implementation thin.
+- Expected risk level: `low`
+- Expected payoff level: `medium`
+
+### Option 4. Align the V1 design docs to the implemented runner and helper
+
+- What it would build or improve:
+  - A cleanup pass on the architecture/spec packet so it reflects `run_prompt.py`, the existing review write-back helper, and the real collision rule.
+- Why it matters:
+  - The current doc drift is small but inspectable.
+  - Fixing it would remove avoidable ambiguity before more workflow support is added.
+- Expected risk level: `low`
+- Expected payoff level: `medium`
+
+### Option 5. Add rejected-run / retry-linkage support
+
+- What it would build or improve:
+  - A small helper or convention for linking a retry run to `retry_of_run_id` and surfacing the latest rejected run for a prompt.
+- Why it matters:
+  - The record model already reserves space for retries, but the workflow does not yet help humans use it consistently.
+  - This would make failure analysis more actionable without introducing a larger retry engine.
+- Expected risk level: `medium`
+- Expected payoff level: `medium`
+
+### Option 6. Add a minimal environment-diagnosis/runbook note or helper
+
+- What it would build or improve:
+  - A bounded way to capture or explain Codex startup failures such as read-only session initialization issues.
+- Why it matters:
+  - The repo contains real evidence of environment-level failure before task execution.
+  - A small runbook or diagnostic helper could reduce confusion when failures are not prompt-related.
+- Expected risk level: `low`
+- Expected payoff level: `low`
+
+## Assessment Of The Current V1 Flow
+
+The current V1 flow is now operationally usable for intentional, reviewed single-step work. It is no longer just a design direction.
+
+What still feels awkward for regular use:
+
+- finding the latest relevant run by hand in `notes/`
+- identifying what is awaiting review
+- deciding whether the next prompt is actually released
+- keeping the docs and record semantics in sync as the thin tools stabilize
+
+What is missing if this is going to be used regularly:
+
+- one small readiness/checking helper
+- one small review-discovery helper
+- one small validation or contract layer to protect the record format
+
+That is enough to make the next step clear: the strongest next work is around operational support for repeated reviewed use, not workflow expansion.

codex
The two notes are written. I’m doing a quick verification pass on the filenames and contents, then I’ll close with the paths and the main recommendation.
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
/bin/bash -lc "sed -n '1,220p' notes/012_top_three_next_options__20260416_000819.md" in /data/git/handson-ml3
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

exec
/bin/bash -lc 'git status --short notes/012_v1_pipeline_options_review__20260416_000819.md notes/012_top_three_next_options__20260416_000819.md' in /data/git/handson-ml3
 succeeded in 0ms:
?? notes/012_top_three_next_options__20260416_000819.md
?? notes/012_v1_pipeline_options_review__20260416_000819.md

diff --git a/notes/012_top_three_next_options__20260416_000819.md b/notes/012_top_three_next_options__20260416_000819.md
new file mode 100644
index 0000000000000000000000000000000000000000..c9c7fb98ff7eeabfb54bbcb6485b3b8729de70e4
--- /dev/null
+++ b/notes/012_top_three_next_options__20260416_000819.md
@@ -0,0 +1,63 @@
+# 012 Top Three Next Options
+
+## Ranked Top Three
+
+### 1. Add a queue-readiness checker
+
+Why it made the top three:
+
+- It closes the most important remaining gap between the V1 architecture and the actual toolchain.
+- The design packet says only `ACCEPTED` should release the next prompt, but the repo does not yet provide a concrete way to answer that question.
+- It is bounded, low-risk, and directly improves repeated operational use.
+
+### 2. Add a review backlog / unreviewed-run lister
+
+Why it made the top three:
+
+- Review write-back now works, but locating records that still need review is manual.
+- This is the next most obvious operational friction once more records accumulate in `notes/`.
+- It improves usability without expanding the workflow into a broader engine.
+
+### 3. Add lightweight contract validation for `run_prompt.py` and `review_run.py`
+
+Why it made the top three:
+
+- The current V1 flow depends on exact markdown structure and field stability.
+- A small regression harness would protect the record contract before more helpers are built on top of it.
+- It is a clean bounded step that reduces accidental drift risk.
+
+## Which One Should Happen Next
+
+The next step should be: `Add a queue-readiness checker`.
+
+Reason:
+
+- It addresses the highest-value remaining seam in the current V1 workflow.
+- It is directly demanded by the architecture and review-gate docs.
+- It reduces the current ambiguity around whether the next prompt may proceed, which is the most important missing operational behavior.
+
+## What Should Explicitly Wait
+
+These should wait until after the top option:
+
+- broader workflow engines or status systems
+- retry orchestration beyond minimal linkage
+- multi-module runner refactors
+- dashboards or aggregated reporting
+- broad platform expansion
+
+These can also wait briefly behind the top option, but remain good bounded follow-ups:
+
+- doc-alignment cleanup for `V1_Bridge_Runner_Change_Spec.md`
+- rejected-run / retry-linkage support
+- environment-focused diagnosis tooling
+
+## Practical Recommendation
+
+Do the next step in this order:
+
+1. queue-readiness checker
+2. review backlog / unreviewed-run lister
+3. lightweight contract validation
+
+That sequence best strengthens the current repo for repeated reviewed use while staying inside the existing V1 boundary.
diff --git a/notes/012_v1_pipeline_options_review__20260416_000819.md b/notes/012_v1_pipeline_options_review__20260416_000819.md
new file mode 100644
index 0000000000000000000000000000000000000000..0f6cc92d9ee3b3f247d902d8d7eb7c676a696429
--- /dev/null
+++ b/notes/012_v1_pipeline_options_review__20260416_000819.md
@@ -0,0 +1,188 @@
+# 012 V1 Pipeline Options Review
+
+## Short Summary Of Current Pipeline Maturity
+
+The V1 pipeline is now a usable thin slice, not just a design packet.
+
+It can:
+
+- execute one prompt through `tools/codex/run_prompt.py`
+- write one durable V1 execution record in `notes/`
+- preserve execution facts separately from review facts
+- accept manual review write-back through `tools/codex/review_run.py`
+
+That means the core V1 artifact boundary is real and operationally demonstrable. The main remaining weakness is not missing architecture. It is missing operational support around repeated use: finding what needs review, deciding what is actually ready to run next, and keeping the docs aligned with the now-real implementation.
+
+## What Is Working Now
+
+### 1. Execution path
+
+- Prompt resolution exists and is conservative. `run_prompt.py` preserves the same direct-path, `codex_prompts/`, and unique-prefix lookup behavior as the bridge runner baseline. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:31).
+- Codex invocation is still thin and inspectable. The runner shells out through `codex exec -C <repo_root> --output-last-message <tempfile> -` and captures return code, final output, and stderr. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:74).
+- Execution-record creation exists and matches the intended V1 section order. The runner writes header, execution facts, review facts, failure analysis, resource facts, prompt text, final output, and stderr into one markdown file. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:97).
+- Resource and failure evidence capture exists in the actual record body. The runner records `return_code`, `elapsed_seconds`, output/stderr character counts, and full stderr text when present. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:144) and the failed record [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:8).
+- The separation between execution outcome and acceptance outcome is implemented correctly. New records start at `review_status: UNREVIEWED` regardless of `EXECUTED` vs `EXECUTION_FAILED`. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:130).
+- The failure path is demonstrated, not hypothetical. The failed smoke-test run preserved the read-only/session-start stderr cleanly while still producing a reviewable record. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).
+
+### 2. Review path
+
+- Manual review write-back exists as a separate thin companion tool. `review_run.py` updates an existing record in place rather than creating sidecars or replacing the record. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:57).
+- Record structure preservation is strong. The helper validates section order and required field lines before writing, then replaces only the targeted review fields. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:122) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:140).
+- The helper is consistent with the V1 review gate. It only allows `ACCEPTED` or `REJECTED`, auto-fills `reviewed_at_utc`, and limits failure-analysis updates to rejected runs. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:12) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:148).
+- Review is operationally usable now. The accepted smoke-test record shows the exact intended lifecycle: executed record first, then in-place write-back to `ACCEPTED`. See [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md:16).
+
+### 3. Workflow usability
+
+- The thin V1 flow is practical for single reviewed runs:
+  - run prompt
+  - inspect record
+  - apply manual review
+- The repo already has evidence of both important paths:
+  - successful execution plus accepted review
+  - failed execution artifact with preserved stderr
+- The implementation stayed within the intended V1 boundary and avoided premature platform growth.
+
+## Seams And Gaps That Remain
+
+### 1. Conservative queue progression is still architectural, not operational
+
+The architecture says only an accepted reviewed run should release the next prompt, but there is no helper yet that actually answers:
+
+- what is the latest run for prompt N
+- whether it is still `UNREVIEWED`
+- whether the next prompt is ready
+
+This is the biggest repeated-use seam. The design expects it, but the current repo still leaves the decision manual.
+
+### 2. Review is usable, but review discovery is manual
+
+`review_run.py` can update a chosen record, but nothing helps a human find:
+
+- all unreviewed records
+- the latest record for a prompt
+- the current review backlog
+
+That becomes awkward as `notes/` grows.
+
+### 3. Stable identity is mostly right, but not perfectly aligned with the current doc
+
+The execution-record artifact says the stable V1 `run_id` is `<prompt_stem>__<started_at_utc>`, but `run_prompt.py` adds `__2`, `__3`, etc. on same-second collisions. See [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:37) versus [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:59).
+
+This is a small practical improvement in code, but it is still a spec mismatch.
+
+### 4. Documentation drift exists
+
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md` still targets `baby_run_prompt.py` as the implementation target and still names the runner as `tools/codex/baby_run_prompt.py`. See [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:5), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:17), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:156), and [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:252).
+- The design packet now implicitly assumes a review write-back step, but the architecture document still describes conservative queue progression as part of the V1 shape without a concrete implementation artifact for that responsibility. See [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:161) and [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:215).
+
+### 5. Repeated-use confidence is under-supported
+
+The runner and review helper are both small and clear, but there is no explicit fixture-based validation or script-level regression harness around:
+
+- record section ordering
+- same-record write-back safety
+- rejection-field behavior
+- collision behavior
+
+The code is simple enough that this is not a correctness emergency, but for repeated operational use it is a credible next bounded improvement.
+
+### 6. Rejected-run and retry linkage are still mostly placeholders
+
+The record shape includes `retry_of_run_id` and manual failure-analysis fields, but the current tools do not help a reviewer connect a rejected run to a retry decision or derive the next retry target.
+
+### 7. Environment fragility still appears in the repo history
+
+The failed smoke-test run shows that Codex session startup can fail before task work begins. That is not a runner-design flaw, but it does mean the workflow still benefits from a small amount of environment-oriented diagnosis or runbook support when repeated use hits startup failures. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).
+
+## Documentation Alignment Assessment
+
+Overall alignment is good at the architectural level and good enough at the artifact level.
+
+What still drifts:
+
+- implementation target naming drift: the bridge-runner spec still speaks in terms of `baby_run_prompt.py`, while the actual V1 runner is `run_prompt.py`
+- runner identity drift: the spec examples still encode `tools/codex/baby_run_prompt.py` while the real records encode `tools/codex/run_prompt.py`
+- stable-identity drift: the spec describes a strict no-suffix `run_id`, while code uses a pragmatic collision suffix
+
+These are small mismatches, but they are no longer theoretical because the repo now contains real V1 records and a real review helper.
+
+## Realistic Next Bounded Options
+
+### Option 1. Add a queue-readiness checker
+
+- What it would build or improve:
+  - A small helper that determines whether the next prompt is ready based on the latest execution record and whether that record is `ACCEPTED`.
+- Why it matters:
+  - This closes the biggest gap between the architecture and current operational reality.
+  - It makes the V1 rule "only accepted review releases the next prompt" actually inspectable in use rather than purely manual.
+- Expected risk level: `low`
+- Expected payoff level: `high`
+
+### Option 2. Add a review backlog / unreviewed-run lister
+
+- What it would build or improve:
+  - A small helper that scans V1 execution records in `notes/` and lists runs still at `UNREVIEWED`, plus the latest record per prompt.
+- Why it matters:
+  - Review write-back already works, but finding what needs review is still manual.
+  - This directly improves repeated operational use without broadening the workflow.
+- Expected risk level: `low`
+- Expected payoff level: `high`
+
+### Option 3. Add lightweight contract validation for the V1 scripts
+
+- What it would build or improve:
+  - A small test or fixture harness around `run_prompt.py` and `review_run.py` that locks in the record shape, allowed review transitions, and in-place write-back behavior.
+- Why it matters:
+  - The workflow now depends on markdown field stability.
+  - A small regression harness would reduce silent drift while keeping the implementation thin.
+- Expected risk level: `low`
+- Expected payoff level: `medium`
+
+### Option 4. Align the V1 design docs to the implemented runner and helper
+
+- What it would build or improve:
+  - A cleanup pass on the architecture/spec packet so it reflects `run_prompt.py`, the existing review write-back helper, and the real collision rule.
+- Why it matters:
+  - The current doc drift is small but inspectable.
+  - Fixing it would remove avoidable ambiguity before more workflow support is added.
+- Expected risk level: `low`
+- Expected payoff level: `medium`
+
+### Option 5. Add rejected-run / retry-linkage support
+
+- What it would build or improve:
+  - A small helper or convention for linking a retry run to `retry_of_run_id` and surfacing the latest rejected run for a prompt.
+- Why it matters:
+  - The record model already reserves space for retries, but the workflow does not yet help humans use it consistently.
+  - This would make failure analysis more actionable without introducing a larger retry engine.
+- Expected risk level: `medium`
+- Expected payoff level: `medium`
+
+### Option 6. Add a minimal environment-diagnosis/runbook note or helper
+
+- What it would build or improve:
+  - A bounded way to capture or explain Codex startup failures such as read-only session initialization issues.
+- Why it matters:
+  - The repo contains real evidence of environment-level failure before task execution.
+  - A small runbook or diagnostic helper could reduce confusion when failures are not prompt-related.
+- Expected risk level: `low`
+- Expected payoff level: `low`
+
+## Assessment Of The Current V1 Flow
+
+The current V1 flow is now operationally usable for intentional, reviewed single-step work. It is no longer just a design direction.
+
+What still feels awkward for regular use:
+
+- finding the latest relevant run by hand in `notes/`
+- identifying what is awaiting review
+- deciding whether the next prompt is actually released
+- keeping the docs and record semantics in sync as the thin tools stabilize
+
+What is missing if this is going to be used regularly:
+
+- one small readiness/checking helper
+- one small review-discovery helper
+- one small validation or contract layer to protect the record format
+
+That is enough to make the next step clear: the strongest next work is around operational support for repeated reviewed use, not workflow expansion.

  ✓ Read the specified design docs, workflow code, and recent notes to map the current V1 pipeline state
  ✓ Synthesize working pieces, seams, and bounded next-step options grounded in repo evidence
  ✓ Write the two requested notes under notes/ with timestamped filenames
diff --git a/notes/012_top_three_next_options__20260416_000819.md b/notes/012_top_three_next_options__20260416_000819.md
new file mode 100644
index 0000000000000000000000000000000000000000..c9c7fb98ff7eeabfb54bbcb6485b3b8729de70e4
--- /dev/null
+++ b/notes/012_top_three_next_options__20260416_000819.md
@@ -0,0 +1,63 @@
+# 012 Top Three Next Options
+
+## Ranked Top Three
+
+### 1. Add a queue-readiness checker
+
+Why it made the top three:
+
+- It closes the most important remaining gap between the V1 architecture and the actual toolchain.
+- The design packet says only `ACCEPTED` should release the next prompt, but the repo does not yet provide a concrete way to answer that question.
+- It is bounded, low-risk, and directly improves repeated operational use.
+
+### 2. Add a review backlog / unreviewed-run lister
+
+Why it made the top three:
+
+- Review write-back now works, but locating records that still need review is manual.
+- This is the next most obvious operational friction once more records accumulate in `notes/`.
+- It improves usability without expanding the workflow into a broader engine.
+
+### 3. Add lightweight contract validation for `run_prompt.py` and `review_run.py`
+
+Why it made the top three:
+
+- The current V1 flow depends on exact markdown structure and field stability.
+- A small regression harness would protect the record contract before more helpers are built on top of it.
+- It is a clean bounded step that reduces accidental drift risk.
+
+## Which One Should Happen Next
+
+The next step should be: `Add a queue-readiness checker`.
+
+Reason:
+
+- It addresses the highest-value remaining seam in the current V1 workflow.
+- It is directly demanded by the architecture and review-gate docs.
+- It reduces the current ambiguity around whether the next prompt may proceed, which is the most important missing operational behavior.
+
+## What Should Explicitly Wait
+
+These should wait until after the top option:
+
+- broader workflow engines or status systems
+- retry orchestration beyond minimal linkage
+- multi-module runner refactors
+- dashboards or aggregated reporting
+- broad platform expansion
+
+These can also wait briefly behind the top option, but remain good bounded follow-ups:
+
+- doc-alignment cleanup for `V1_Bridge_Runner_Change_Spec.md`
+- rejected-run / retry-linkage support
+- environment-focused diagnosis tooling
+
+## Practical Recommendation
+
+Do the next step in this order:
+
+1. queue-readiness checker
+2. review backlog / unreviewed-run lister
+3. lightweight contract validation
+
+That sequence best strengthens the current repo for repeated reviewed use while staying inside the existing V1 boundary.
diff --git a/notes/012_v1_pipeline_options_review__20260416_000819.md b/notes/012_v1_pipeline_options_review__20260416_000819.md
new file mode 100644
index 0000000000000000000000000000000000000000..0f6cc92d9ee3b3f247d902d8d7eb7c676a696429
--- /dev/null
+++ b/notes/012_v1_pipeline_options_review__20260416_000819.md
@@ -0,0 +1,188 @@
+# 012 V1 Pipeline Options Review
+
+## Short Summary Of Current Pipeline Maturity
+
+The V1 pipeline is now a usable thin slice, not just a design packet.
+
+It can:
+
+- execute one prompt through `tools/codex/run_prompt.py`
+- write one durable V1 execution record in `notes/`
+- preserve execution facts separately from review facts
+- accept manual review write-back through `tools/codex/review_run.py`
+
+That means the core V1 artifact boundary is real and operationally demonstrable. The main remaining weakness is not missing architecture. It is missing operational support around repeated use: finding what needs review, deciding what is actually ready to run next, and keeping the docs aligned with the now-real implementation.
+
+## What Is Working Now
+
+### 1. Execution path
+
+- Prompt resolution exists and is conservative. `run_prompt.py` preserves the same direct-path, `codex_prompts/`, and unique-prefix lookup behavior as the bridge runner baseline. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:31).
+- Codex invocation is still thin and inspectable. The runner shells out through `codex exec -C <repo_root> --output-last-message <tempfile> -` and captures return code, final output, and stderr. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:74).
+- Execution-record creation exists and matches the intended V1 section order. The runner writes header, execution facts, review facts, failure analysis, resource facts, prompt text, final output, and stderr into one markdown file. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:97).
+- Resource and failure evidence capture exists in the actual record body. The runner records `return_code`, `elapsed_seconds`, output/stderr character counts, and full stderr text when present. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:144) and the failed record [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:8).
+- The separation between execution outcome and acceptance outcome is implemented correctly. New records start at `review_status: UNREVIEWED` regardless of `EXECUTED` vs `EXECUTION_FAILED`. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:130).
+- The failure path is demonstrated, not hypothetical. The failed smoke-test run preserved the read-only/session-start stderr cleanly while still producing a reviewable record. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).
+
+### 2. Review path
+
+- Manual review write-back exists as a separate thin companion tool. `review_run.py` updates an existing record in place rather than creating sidecars or replacing the record. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:57).
+- Record structure preservation is strong. The helper validates section order and required field lines before writing, then replaces only the targeted review fields. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:122) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:140).
+- The helper is consistent with the V1 review gate. It only allows `ACCEPTED` or `REJECTED`, auto-fills `reviewed_at_utc`, and limits failure-analysis updates to rejected runs. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:12) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:148).
+- Review is operationally usable now. The accepted smoke-test record shows the exact intended lifecycle: executed record first, then in-place write-back to `ACCEPTED`. See [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md:16).
+
+### 3. Workflow usability
+
+- The thin V1 flow is practical for single reviewed runs:
+  - run prompt
+  - inspect record
+  - apply manual review
+- The repo already has evidence of both important paths:
+  - successful execution plus accepted review
+  - failed execution artifact with preserved stderr
+- The implementation stayed within the intended V1 boundary and avoided premature platform growth.
+
+## Seams And Gaps That Remain
+
+### 1. Conservative queue progression is still architectural, not operational
+
+The architecture says only an accepted reviewed run should release the next prompt, but there is no helper yet that actually answers:
+
+- what is the latest run for prompt N
+- whether it is still `UNREVIEWED`
+- whether the next prompt is ready
+
+This is the biggest repeated-use seam. The design expects it, but the current repo still leaves the decision manual.
+
+### 2. Review is usable, but review discovery is manual
+
+`review_run.py` can update a chosen record, but nothing helps a human find:
+
+- all unreviewed records
+- the latest record for a prompt
+- the current review backlog
+
+That becomes awkward as `notes/` grows.
+
+### 3. Stable identity is mostly right, but not perfectly aligned with the current doc
+
+The execution-record artifact says the stable V1 `run_id` is `<prompt_stem>__<started_at_utc>`, but `run_prompt.py` adds `__2`, `__3`, etc. on same-second collisions. See [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:37) versus [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:59).
+
+This is a small practical improvement in code, but it is still a spec mismatch.
+
+### 4. Documentation drift exists
+
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md` still targets `baby_run_prompt.py` as the implementation target and still names the runner as `tools/codex/baby_run_prompt.py`. See [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:5), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:17), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:156), and [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:252).
+- The design packet now implicitly assumes a review write-back step, but the architecture document still describes conservative queue progression as part of the V1 shape without a concrete implementation artifact for that responsibility. See [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:161) and [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:215).
+
+### 5. Repeated-use confidence is under-supported
+
+The runner and review helper are both small and clear, but there is no explicit fixture-based validation or script-level regression harness around:
+
+- record section ordering
+- same-record write-back safety
+- rejection-field behavior
+- collision behavior
+
+The code is simple enough that this is not a correctness emergency, but for repeated operational use it is a credible next bounded improvement.
+
+### 6. Rejected-run and retry linkage are still mostly placeholders
+
+The record shape includes `retry_of_run_id` and manual failure-analysis fields, but the current tools do not help a reviewer connect a rejected run to a retry decision or derive the next retry target.
+
+### 7. Environment fragility still appears in the repo history
+
+The failed smoke-test run shows that Codex session startup can fail before task work begins. That is not a runner-design flaw, but it does mean the workflow still benefits from a small amount of environment-oriented diagnosis or runbook support when repeated use hits startup failures. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).
+
+## Documentation Alignment Assessment
+
+Overall alignment is good at the architectural level and good enough at the artifact level.
+
+What still drifts:
+
+- implementation target naming drift: the bridge-runner spec still speaks in terms of `baby_run_prompt.py`, while the actual V1 runner is `run_prompt.py`
+- runner identity drift: the spec examples still encode `tools/codex/baby_run_prompt.py` while the real records encode `tools/codex/run_prompt.py`
+- stable-identity drift: the spec describes a strict no-suffix `run_id`, while code uses a pragmatic collision suffix
+
+These are small mismatches, but they are no longer theoretical because the repo now contains real V1 records and a real review helper.
+
+## Realistic Next Bounded Options
+
+### Option 1. Add a queue-readiness checker
+
+- What it would build or improve:
+  - A small helper that determines whether the next prompt is ready based on the latest execution record and whether that record is `ACCEPTED`.
+- Why it matters:
+  - This closes the biggest gap between the architecture and current operational reality.
+  - It makes the V1 rule "only accepted review releases the next prompt" actually inspectable in use rather than purely manual.
+- Expected risk level: `low`
+- Expected payoff level: `high`
+
+### Option 2. Add a review backlog / unreviewed-run lister
+
+- What it would build or improve:
+  - A small helper that scans V1 execution records in `notes/` and lists runs still at `UNREVIEWED`, plus the latest record per prompt.
+- Why it matters:
+  - Review write-back already works, but finding what needs review is still manual.
+  - This directly improves repeated operational use without broadening the workflow.
+- Expected risk level: `low`
+- Expected payoff level: `high`
+
+### Option 3. Add lightweight contract validation for the V1 scripts
+
+- What it would build or improve:
+  - A small test or fixture harness around `run_prompt.py` and `review_run.py` that locks in the record shape, allowed review transitions, and in-place write-back behavior.
+- Why it matters:
+  - The workflow now depends on markdown field stability.
+  - A small regression harness would reduce silent drift while keeping the implementation thin.
+- Expected risk level: `low`
+- Expected payoff level: `medium`
+
+### Option 4. Align the V1 design docs to the implemented runner and helper
+
+- What it would build or improve:
+  - A cleanup pass on the architecture/spec packet so it reflects `run_prompt.py`, the existing review write-back helper, and the real collision rule.
+- Why it matters:
+  - The current doc drift is small but inspectable.
+  - Fixing it would remove avoidable ambiguity before more workflow support is added.
+- Expected risk level: `low`
+- Expected payoff level: `medium`
+
+### Option 5. Add rejected-run / retry-linkage support
+
+- What it would build or improve:
+  - A small helper or convention for linking a retry run to `retry_of_run_id` and surfacing the latest rejected run for a prompt.
+- Why it matters:
+  - The record model already reserves space for retries, but the workflow does not yet help humans use it consistently.
+  - This would make failure analysis more actionable without introducing a larger retry engine.
+- Expected risk level: `medium`
+- Expected payoff level: `medium`
+
+### Option 6. Add a minimal environment-diagnosis/runbook note or helper
+
+- What it would build or improve:
+  - A bounded way to capture or explain Codex startup failures such as read-only session initialization issues.
+- Why it matters:
+  - The repo contains real evidence of environment-level failure before task execution.
+  - A small runbook or diagnostic helper could reduce confusion when failures are not prompt-related.
+- Expected risk level: `low`
+- Expected payoff level: `low`
+
+## Assessment Of The Current V1 Flow
+
+The current V1 flow is now operationally usable for intentional, reviewed single-step work. It is no longer just a design direction.
+
+What still feels awkward for regular use:
+
+- finding the latest relevant run by hand in `notes/`
+- identifying what is awaiting review
+- deciding whether the next prompt is actually released
+- keeping the docs and record semantics in sync as the thin tools stabilize
+
+What is missing if this is going to be used regularly:
+
+- one small readiness/checking helper
+- one small review-discovery helper
+- one small validation or contract layer to protect the record format
+
+That is enough to make the next step clear: the strongest next work is around operational support for repeated reviewed use, not workflow expansion.

codex
Created the two requested artifacts:

- [012_v1_pipeline_options_review__20260416_000819.md](/data/git/handson-ml3/notes/012_v1_pipeline_options_review__20260416_000819.md)
- [012_top_three_next_options__20260416_000819.md](/data/git/handson-ml3/notes/012_top_three_next_options__20260416_000819.md)

The review concludes the V1 pipe is now a real usable thin slice: `run_prompt.py` creates durable execution records, `review_run.py` performs in-place manual review write-back, and the main remaining weakness is operational support for repeated use rather than missing architecture. The top recommendation is to add a small queue-readiness checker next, followed by an unreviewed-run lister and then lightweight contract validation for the two V1 scripts.

I did not modify `tools/codex/baby_run_prompt.py`, `tools/codex/run_prompt.py`, or `tools/codex/review_run.py`.
diff --git a/notes/012_top_three_next_options__20260416_000819.md b/notes/012_top_three_next_options__20260416_000819.md
new file mode 100644
index 0000000000000000000000000000000000000000..c9c7fb98ff7eeabfb54bbcb6485b3b8729de70e4
--- /dev/null
+++ b/notes/012_top_three_next_options__20260416_000819.md
@@ -0,0 +1,63 @@
+# 012 Top Three Next Options
+
+## Ranked Top Three
+
+### 1. Add a queue-readiness checker
+
+Why it made the top three:
+
+- It closes the most important remaining gap between the V1 architecture and the actual toolchain.
+- The design packet says only `ACCEPTED` should release the next prompt, but the repo does not yet provide a concrete way to answer that question.
+- It is bounded, low-risk, and directly improves repeated operational use.
+
+### 2. Add a review backlog / unreviewed-run lister
+
+Why it made the top three:
+
+- Review write-back now works, but locating records that still need review is manual.
+- This is the next most obvious operational friction once more records accumulate in `notes/`.
+- It improves usability without expanding the workflow into a broader engine.
+
+### 3. Add lightweight contract validation for `run_prompt.py` and `review_run.py`
+
+Why it made the top three:
+
+- The current V1 flow depends on exact markdown structure and field stability.
+- A small regression harness would protect the record contract before more helpers are built on top of it.
+- It is a clean bounded step that reduces accidental drift risk.
+
+## Which One Should Happen Next
+
+The next step should be: `Add a queue-readiness checker`.
+
+Reason:
+
+- It addresses the highest-value remaining seam in the current V1 workflow.
+- It is directly demanded by the architecture and review-gate docs.
+- It reduces the current ambiguity around whether the next prompt may proceed, which is the most important missing operational behavior.
+
+## What Should Explicitly Wait
+
+These should wait until after the top option:
+
+- broader workflow engines or status systems
+- retry orchestration beyond minimal linkage
+- multi-module runner refactors
+- dashboards or aggregated reporting
+- broad platform expansion
+
+These can also wait briefly behind the top option, but remain good bounded follow-ups:
+
+- doc-alignment cleanup for `V1_Bridge_Runner_Change_Spec.md`
+- rejected-run / retry-linkage support
+- environment-focused diagnosis tooling
+
+## Practical Recommendation
+
+Do the next step in this order:
+
+1. queue-readiness checker
+2. review backlog / unreviewed-run lister
+3. lightweight contract validation
+
+That sequence best strengthens the current repo for repeated reviewed use while staying inside the existing V1 boundary.
diff --git a/notes/012_v1_pipeline_options_review__20260416_000819.md b/notes/012_v1_pipeline_options_review__20260416_000819.md
new file mode 100644
index 0000000000000000000000000000000000000000..0f6cc92d9ee3b3f247d902d8d7eb7c676a696429
--- /dev/null
+++ b/notes/012_v1_pipeline_options_review__20260416_000819.md
@@ -0,0 +1,188 @@
+# 012 V1 Pipeline Options Review
+
+## Short Summary Of Current Pipeline Maturity
+
+The V1 pipeline is now a usable thin slice, not just a design packet.
+
+It can:
+
+- execute one prompt through `tools/codex/run_prompt.py`
+- write one durable V1 execution record in `notes/`
+- preserve execution facts separately from review facts
+- accept manual review write-back through `tools/codex/review_run.py`
+
+That means the core V1 artifact boundary is real and operationally demonstrable. The main remaining weakness is not missing architecture. It is missing operational support around repeated use: finding what needs review, deciding what is actually ready to run next, and keeping the docs aligned with the now-real implementation.
+
+## What Is Working Now
+
+### 1. Execution path
+
+- Prompt resolution exists and is conservative. `run_prompt.py` preserves the same direct-path, `codex_prompts/`, and unique-prefix lookup behavior as the bridge runner baseline. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:31).
+- Codex invocation is still thin and inspectable. The runner shells out through `codex exec -C <repo_root> --output-last-message <tempfile> -` and captures return code, final output, and stderr. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:74).
+- Execution-record creation exists and matches the intended V1 section order. The runner writes header, execution facts, review facts, failure analysis, resource facts, prompt text, final output, and stderr into one markdown file. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:97).
+- Resource and failure evidence capture exists in the actual record body. The runner records `return_code`, `elapsed_seconds`, output/stderr character counts, and full stderr text when present. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:144) and the failed record [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:8).
+- The separation between execution outcome and acceptance outcome is implemented correctly. New records start at `review_status: UNREVIEWED` regardless of `EXECUTED` vs `EXECUTION_FAILED`. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:130).
+- The failure path is demonstrated, not hypothetical. The failed smoke-test run preserved the read-only/session-start stderr cleanly while still producing a reviewable record. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).
+
+### 2. Review path
+
+- Manual review write-back exists as a separate thin companion tool. `review_run.py` updates an existing record in place rather than creating sidecars or replacing the record. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:57).
+- Record structure preservation is strong. The helper validates section order and required field lines before writing, then replaces only the targeted review fields. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:122) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:140).
+- The helper is consistent with the V1 review gate. It only allows `ACCEPTED` or `REJECTED`, auto-fills `reviewed_at_utc`, and limits failure-analysis updates to rejected runs. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:12) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:148).
+- Review is operationally usable now. The accepted smoke-test record shows the exact intended lifecycle: executed record first, then in-place write-back to `ACCEPTED`. See [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md:16).
+
+### 3. Workflow usability
+
+- The thin V1 flow is practical for single reviewed runs:
+  - run prompt
+  - inspect record
+  - apply manual review
+- The repo already has evidence of both important paths:
+  - successful execution plus accepted review
+  - failed execution artifact with preserved stderr
+- The implementation stayed within the intended V1 boundary and avoided premature platform growth.
+
+## Seams And Gaps That Remain
+
+### 1. Conservative queue progression is still architectural, not operational
+
+The architecture says only an accepted reviewed run should release the next prompt, but there is no helper yet that actually answers:
+
+- what is the latest run for prompt N
+- whether it is still `UNREVIEWED`
+- whether the next prompt is ready
+
+This is the biggest repeated-use seam. The design expects it, but the current repo still leaves the decision manual.
+
+### 2. Review is usable, but review discovery is manual
+
+`review_run.py` can update a chosen record, but nothing helps a human find:
+
+- all unreviewed records
+- the latest record for a prompt
+- the current review backlog
+
+That becomes awkward as `notes/` grows.
+
+### 3. Stable identity is mostly right, but not perfectly aligned with the current doc
+
+The execution-record artifact says the stable V1 `run_id` is `<prompt_stem>__<started_at_utc>`, but `run_prompt.py` adds `__2`, `__3`, etc. on same-second collisions. See [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:37) versus [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:59).
+
+This is a small practical improvement in code, but it is still a spec mismatch.
+
+### 4. Documentation drift exists
+
+- `tools/codex/V1_Bridge_Runner_Change_Spec.md` still targets `baby_run_prompt.py` as the implementation target and still names the runner as `tools/codex/baby_run_prompt.py`. See [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:5), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:17), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:156), and [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:252).
+- The design packet now implicitly assumes a review write-back step, but the architecture document still describes conservative queue progression as part of the V1 shape without a concrete implementation artifact for that responsibility. See [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:161) and [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:215).
+
+### 5. Repeated-use confidence is under-supported
+
+The runner and review helper are both small and clear, but there is no explicit fixture-based validation or script-level regression harness around:
+
+- record section ordering
+- same-record write-back safety
+- rejection-field behavior
+- collision behavior
+
+The code is simple enough that this is not a correctness emergency, but for repeated operational use it is a credible next bounded improvement.
+
+### 6. Rejected-run and retry linkage are still mostly placeholders
+
+The record shape includes `retry_of_run_id` and manual failure-analysis fields, but the current tools do not help a reviewer connect a rejected run to a retry decision or derive the next retry target.
+
+### 7. Environment fragility still appears in the repo history
+
+The failed smoke-test run shows that Codex session startup can fail before task work begins. That is not a runner-design flaw, but it does mean the workflow still benefits from a small amount of environment-oriented diagnosis or runbook support when repeated use hits startup failures. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).
+
+## Documentation Alignment Assessment
+
+Overall alignment is good at the architectural level and good enough at the artifact level.
+
+What still drifts:
+
+- implementation target naming drift: the bridge-runner spec still speaks in terms of `baby_run_prompt.py`, while the actual V1 runner is `run_prompt.py`
+- runner identity drift: the spec examples still encode `tools/codex/baby_run_prompt.py` while the real records encode `tools/codex/run_prompt.py`
+- stable-identity drift: the spec describes a strict no-suffix `run_id`, while code uses a pragmatic collision suffix
+
+These are small mismatches, but they are no longer theoretical because the repo now contains real V1 records and a real review helper.
+
+## Realistic Next Bounded Options
+
+### Option 1. Add a queue-readiness checker
+
+- What it would build or improve:
+  - A small helper that determines whether the next prompt is ready based on the latest execution record and whether that record is `ACCEPTED`.
+- Why it matters:
+  - This closes the biggest gap between the architecture and current operational reality.
+  - It makes the V1 rule "only accepted review releases the next prompt" actually inspectable in use rather than purely manual.
+- Expected risk level: `low`
+- Expected payoff level: `high`
+
+### Option 2. Add a review backlog / unreviewed-run lister
+
+- What it would build or improve:
+  - A small helper that scans V1 execution records in `notes/` and lists runs still at `UNREVIEWED`, plus the latest record per prompt.
+- Why it matters:
+  - Review write-back already works, but finding what needs review is still manual.
+  - This directly improves repeated operational use without broadening the workflow.
+- Expected risk level: `low`
+- Expected payoff level: `high`
+
+### Option 3. Add lightweight contract validation for the V1 scripts
+
+- What it would build or improve:
+  - A small test or fixture harness around `run_prompt.py` and `review_run.py` that locks in the record shape, allowed review transitions, and in-place write-back behavior.
+- Why it matters:
+  - The workflow now depends on markdown field stability.
+  - A small regression harness would reduce silent drift while keeping the implementation thin.
+- Expected risk level: `low`
+- Expected payoff level: `medium`
+
+### Option 4. Align the V1 design docs to the implemented runner and helper
+
+- What it would build or improve:
+  - A cleanup pass on the architecture/spec packet so it reflects `run_prompt.py`, the existing review write-back helper, and the real collision rule.
+- Why it matters:
+  - The current doc drift is small but inspectable.
+  - Fixing it would remove avoidable ambiguity before more workflow support is added.
+- Expected risk level: `low`
+- Expected payoff level: `medium`
+
+### Option 5. Add rejected-run / retry-linkage support
+
+- What it would build or improve:
+  - A small helper or convention for linking a retry run to `retry_of_run_id` and surfacing the latest rejected run for a prompt.
+- Why it matters:
+  - The record model already reserves space for retries, but the workflow does not yet help humans use it consistently.
+  - This would make failure analysis more actionable without introducing a larger retry engine.
+- Expected risk level: `medium`
+- Expected payoff level: `medium`
+
+### Option 6. Add a minimal environment-diagnosis/runbook note or helper
+
+- What it would build or improve:
+  - A bounded way to capture or explain Codex startup failures such as read-only session initialization issues.
+- Why it matters:
+  - The repo contains real evidence of environment-level failure before task execution.
+  - A small runbook or diagnostic helper could reduce confusion when failures are not prompt-related.
+- Expected risk level: `low`
+- Expected payoff level: `low`
+
+## Assessment Of The Current V1 Flow
+
+The current V1 flow is now operationally usable for intentional, reviewed single-step work. It is no longer just a design direction.
+
+What still feels awkward for regular use:
+
+- finding the latest relevant run by hand in `notes/`
+- identifying what is awaiting review
+- deciding whether the next prompt is actually released
+- keeping the docs and record semantics in sync as the thin tools stabilize
+
+What is missing if this is going to be used regularly:
+
+- one small readiness/checking helper
+- one small review-discovery helper
+- one small validation or contract layer to protect the record format
+
+That is enough to make the next step clear: the strongest next work is around operational support for repeated reviewed use, not workflow expansion.

tokens used
71,570
```
