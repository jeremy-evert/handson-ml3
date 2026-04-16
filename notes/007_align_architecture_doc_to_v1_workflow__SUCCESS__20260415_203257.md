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

Build a small, clean, reusable prompt workflow system for Codex work inside a repository.

This system should be conservative first:

* easy to inspect
* easy to test
* easy to extend
* composed of small parts
* safe by default

The design should emphasize **separation of concerns** so each piece does one job clearly.

---

## Why We Are Re-Architecting

The first draft bundled too many responsibilities into one script. That made it harder to:

* understand
* trust
* test
* evolve
* reuse across repositories

We want a cleaner structure that can become a template for future repos.

---

## What the First Script Was Doing

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

---

## What I Would Like This System To Do Eventually

These are ideas worth considering, but they should not all be built at once.

### A. Better architecture and reuse

* work as a reusable template across repositories
* isolate config from code
* support per-repo conventions without rewriting internals

### B. Structured metadata

* optionally store prompt metadata in frontmatter or sidecar files
* support fields like:

  * title
  * objective
  * tags
  * priority
  * dependencies
  * owner

### C. Better status model

Instead of only:

* UNRUN
* SUCCESS
* FAIL

We may want:

* READY
* RUNNING
* BLOCKED
* NEEDS_REVIEW
* PARTIAL
* SKIPPED
* ARCHIVED

### D. Retry intelligence

* carry forward the previous failed note automatically
* include multiple prior failures, not just the latest one
* summarize recurring failure patterns

### E. Prompt preparation

* generate a clean “execution bundle” for Codex
* include:

  * prompt text
  * repo context
  * previous failure note if retrying
  * explicit success criteria

### F. Manual and automated modes

* manual mode for conservative workflows
* dry-run mode for seeing what would happen
* later, optional Codex CLI integration

### G. Report generation

* generate queue reports
* generate status snapshots
* generate audit/history reports over prompt activity

### H. Better note/content model

* move from filename-only state to richer note contents
* enforce a consistent note template
* optionally add structured machine-readable metadata inside notes

### I. Dependency and sequencing support

* allow prompts to depend on other prompts
* prevent prompts from running before prerequisites are complete

### J. Prompt families or lanes

* support categories such as:

  * setup
  * audit
  * refactor
  * notebook generation
  * documentation

### K. Approval gates

* require human confirmation before marking success
* require review before allowing dependent prompts to proceed

### L. Logging and diagnostics

* maintain an execution log
* record tool errors separately from task failures
* distinguish infrastructure failure from prompt failure

### M. File layout that scales cleanly

* support growth without turning into a junk drawer

---

## Proposed Folder Architecture

A better first structure would be:

```text
tools/
  codex/
    README.md
    architecture.md
    config.py
    paths.py
    prompts.py
    notes.py
    status.py
    retry.py
    cli.py
    templates/
      note_success.md
      note_fail.md
```

For an even more conservative start, we can go smaller:

```text
tools/
  codex/
    README.md
    architecture.md
    paths.py
    prompts.py
    notes.py
    status.py
```

Then add a tiny entrypoint later if needed.


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
/bin/bash -lc "sed -n '1,260p' notes/004_architecture_and_bridge_runner_review__20260415_195538.md" in /data/git/handson-ml3
 succeeded in 0ms:
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

exec
/bin/bash -lc "sed -n '1,260p' notes/005_prompt_queue_plan__20260415_202557.md" in /data/git/handson-ml3
 succeeded in 0ms:
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

codex
The gap is clear: the architecture doc is still jumping from decomposition to future ideas, while the governing workflow expects an explicit V1 slice, artifact set, review stop, and deferral line. I’m pulling the remaining workflow sections now so the revision follows that sequence directly.
exec
/bin/bash -lc "sed -n '221,420p' tools/Project_Design_Workflow.md" in /data/git/handson-ml3
 succeeded in 0ms:
* What modules or interfaces should exist?
* What examples, fixtures, or test inputs should exist?
* What outputs prove the path works?
* What notes, logs, or reports should be kept as durable project memory?

Output:

* deliverables list
* artifact inventory

---

### Phase 6: Sequence the work

This is the "what order reduces pain and risk?" phase.

Questions:

* What must come first because other things depend on it?
* What can be tested independently?
* What pieces should be proven before automation or convenience layers are added?
* What order keeps the build inspectable?

Output:

* implementation order
* dependency chain

---

### Phase 7: Define validation

This is the "how will we know each layer works?" phase.

Questions:

* What is the smoke test for each part?
* What is a good manual test before automation?
* What failure modes do we expect?
* What evidence counts as success?
* What review should happen before the next step begins?

Output:

* validation checklist
* smoke tests
* example inputs and outputs
* review points between iterations

---

### Phase 8: Execute one bounded slice

Only now do we begin implementation.

Rules:

* build one thin slice
* keep the task bounded and inspectable
* use bridge tooling only when it remains thin and subordinate to the design
* validate the slice
* inspect the result before issuing the next step
* do not sprint ahead because the first part felt good

Output:

* one completed slice
* evidence of validation
* notes about what the design got right or wrong

---

### Phase 9: Review and refine

This is the "did the design survive contact with reality?" phase.

Questions:

* What felt clean?
* What felt awkward?
* What assumptions broke?
* What should be renamed, split, simplified, or deferred?
* What should change before the next bounded step?

Output:

* refinement notes
* updated architecture or scope if needed
* revised next-step plan

---

## The Iteration Loop

After the initial design work, many projects should move through a repeating bounded loop:

1. clarify the next chunk
2. define a bounded task
3. state the success criteria
4. execute
5. validate and inspect results
6. review what changed
7. refine the plan
8. issue the next bounded task only after review

This loop should stay small enough that:

* the task can be understood before execution
* the result can be reviewed without guesswork
* failure teaches something specific
* refinement happens while context is still fresh

If a project cannot explain the next chunk in a few clear sentences, the chunk is probably still too large.

---

## The Practical Decomposition Pattern

When a design feels big, decompose it in this order:

### 1. Purpose

What is the system for?

### 2. Boundaries

What is in and out?

### 3. Components

What parts exist?

### 4. Responsibilities

What does each part own?

### 5. Artifacts

What files or outputs must exist?

### 6. Sequence

What gets built first?

### 7. Validation

How do we test each step?

### 8. Review points

Where do we stop and inspect before continuing?

### 9. Extension path

What comes later, but not now?

That pattern is portable and should work across many projects.

---

## Failure Analysis as Part of the Workflow

When a bounded step fails, do not treat the failure as noise.
Capture it, inspect it, and decide whether the problem is in the task, the decomposition, the criteria, or the environment.

Useful questions include:

* Was the task too large?
* Was the task poorly decomposed?
* Were success criteria unclear or incomplete?
* Did the task depend on hidden assumptions?
* Was the failure caused by tooling or infrastructure rather than task difficulty?
* Would a smaller or differently framed task have worked better?
* Did the review happen too late?
* Did retries produce new information, or only repeat cost?

Useful outputs include:

* a short failure note or report
* updated task boundaries
* revised success criteria
* a smaller follow-up slice
* a decision to fix environment issues before retrying

A clean retry is often possible, but it should come after analysis rather than instead of it.

---

## Resource and Cost Awareness

Projects benefit from tracking lightweight evidence about execution cost and quality.
This does not need to be elaborate, but it should be enough to notice patterns.

Examples of useful observations:

* elapsed time
* execution size
* review effort
* repeated retries
* failure frequency

exec
/bin/bash -lc "nl -ba tools/Codex_Prompt_Workflow_Architecture.md | sed -n '1,260p'" in /data/git/handson-ml3
 succeeded in 0ms:
     1	# Codex Prompt Workflow Architecture
     2	
     3	## Goal
     4	
     5	Build a small, clean, reusable prompt workflow system for Codex work inside a repository.
     6	
     7	This system should be conservative first:
     8	
     9	* easy to inspect
    10	* easy to test
    11	* easy to extend
    12	* composed of small parts
    13	* safe by default
    14	
    15	The design should emphasize **separation of concerns** so each piece does one job clearly.
    16	
    17	---
    18	
    19	## Why We Are Re-Architecting
    20	
    21	The first draft bundled too many responsibilities into one script. That made it harder to:
    22	
    23	* understand
    24	* trust
    25	* test
    26	* evolve
    27	* reuse across repositories
    28	
    29	We want a cleaner structure that can become a template for future repos.
    30	
    31	---
    32	
    33	## What the First Script Was Doing
    34	
    35	Below is a decomposition of the responsibilities that were bundled together.
    36	
    37	### 1. Repo path discovery
    38	
    39	* figure out where the repository root is
    40	* infer where `codex_prompts/` and `notes/` live
    41	
    42	### 2. Directory validation
    43	
    44	* verify the expected folders exist
    45	* fail cleanly if they do not
    46	
    47	### 3. Prompt discovery
    48	
    49	* scan `codex_prompts/`
    50	* identify valid prompt files
    51	* ignore files that do not match the naming convention
    52	
    53	### 4. Prompt parsing and indexing
    54	
    55	* extract numeric prefixes from filenames
    56	* sort prompts in execution order
    57	* establish a stable prompt identity from filename/stem
    58	
    59	### 5. Note discovery
    60	
    61	* scan `notes/`
    62	* find note files that match the naming convention
    63	* ignore unrelated markdown files
    64	
    65	### 6. Note parsing
    66	
    67	* extract prompt name, success/fail state, and timestamp from note filenames
    68	* turn note filenames into structured metadata
    69	
    70	### 7. Status reconstruction
    71	
    72	* map prompts to matching notes
    73	* determine whether each prompt is:
    74	
    75	  * UNRUN
    76	  * SUCCESS
    77	  * FAIL
    78	* pick the latest note when there are multiple notes for one prompt
    79	
    80	### 8. Prompt selection
    81	
    82	* find the next unrun prompt
    83	* find the first failed prompt
    84	* find a prompt by numeric prefix, base name, or full filename
    85	
    86	### 9. Prompt display
    87	
    88	* print a selected prompt to the terminal
    89	
    90	### 10. Retry context assembly
    91	
    92	* find the latest failed note for a prompt
    93	* display the original prompt plus the latest failed note
    94	* present instructions for retrying
    95	
    96	### 11. Note writing
    97	
    98	* create timestamped note filenames
    99	* write notes in markdown
   100	* optionally include the previous note as context
   101	
   102	### 12. Manual status marking
   103	
   104	* let the user record a success or failure
   105	* attach summary/details text to the note
   106	
   107	### 13. Command-line interface
   108	
   109	* parse subcommands like `list`, `next`, `show`, `mark`, `retry-failed`
   110	* route commands to the right behaviors
   111	
   112	### 14. Terminal presentation
   113	
   114	* format output for human readability
   115	* present status tables and retry blocks
   116	
   117	---
   118	
   119	## What I Would Like This System To Do Eventually
   120	
   121	These are ideas worth considering, but they should not all be built at once.
   122	
   123	### A. Better architecture and reuse
   124	
   125	* work as a reusable template across repositories
   126	* isolate config from code
   127	* support per-repo conventions without rewriting internals
   128	
   129	### B. Structured metadata
   130	
   131	* optionally store prompt metadata in frontmatter or sidecar files
   132	* support fields like:
   133	
   134	  * title
   135	  * objective
   136	  * tags
   137	  * priority
   138	  * dependencies
   139	  * owner
   140	
   141	### C. Better status model
   142	
   143	Instead of only:
   144	
   145	* UNRUN
   146	* SUCCESS
   147	* FAIL
   148	
   149	We may want:
   150	
   151	* READY
   152	* RUNNING
   153	* BLOCKED
   154	* NEEDS_REVIEW
   155	* PARTIAL
   156	* SKIPPED
   157	* ARCHIVED
   158	
   159	### D. Retry intelligence
   160	
   161	* carry forward the previous failed note automatically
   162	* include multiple prior failures, not just the latest one
   163	* summarize recurring failure patterns
   164	
   165	### E. Prompt preparation
   166	
   167	* generate a clean “execution bundle” for Codex
   168	* include:
   169	
   170	  * prompt text
   171	  * repo context
   172	  * previous failure note if retrying
   173	  * explicit success criteria
   174	
   175	### F. Manual and automated modes
   176	
   177	* manual mode for conservative workflows
   178	* dry-run mode for seeing what would happen
   179	* later, optional Codex CLI integration
   180	
   181	### G. Report generation
   182	
   183	* generate queue reports
   184	* generate status snapshots
   185	* generate audit/history reports over prompt activity
   186	
   187	### H. Better note/content model
   188	
   189	* move from filename-only state to richer note contents
   190	* enforce a consistent note template
   191	* optionally add structured machine-readable metadata inside notes
   192	
   193	### I. Dependency and sequencing support
   194	
   195	* allow prompts to depend on other prompts
   196	* prevent prompts from running before prerequisites are complete
   197	
   198	### J. Prompt families or lanes
   199	
   200	* support categories such as:
   201	
   202	  * setup
   203	  * audit
   204	  * refactor
   205	  * notebook generation
   206	  * documentation
   207	
   208	### K. Approval gates
   209	
   210	* require human confirmation before marking success
   211	* require review before allowing dependent prompts to proceed
   212	
   213	### L. Logging and diagnostics
   214	
   215	* maintain an execution log
   216	* record tool errors separately from task failures
   217	* distinguish infrastructure failure from prompt failure
   218	
   219	### M. File layout that scales cleanly
   220	
   221	* support growth without turning into a junk drawer
   222	
   223	---
   224	
   225	## Proposed Folder Architecture
   226	
   227	A better first structure would be:
   228	
   229	```text
   230	tools/
   231	  codex/
   232	    README.md
   233	    architecture.md
   234	    config.py
   235	    paths.py
   236	    prompts.py
   237	    notes.py
   238	    status.py
   239	    retry.py
   240	    cli.py
   241	    templates/
   242	      note_success.md
   243	      note_fail.md
   244	```
   245	
   246	For an even more conservative start, we can go smaller:
   247	
   248	```text
   249	tools/
   250	  codex/
   251	    README.md
   252	    architecture.md
   253	    paths.py
   254	    prompts.py
   255	    notes.py
   256	    status.py
   257	```
   258	
   259	Then add a tiny entrypoint later if needed.
   260	

codex
I have enough context. I’m revising the architecture doc in place so it explicitly follows the governing workflow sequence: problem, boundary, artifacts, minimum slice, implementation order, validation/review, and deferrals tied to the V1 execution record and review gate.
apply patch
patch: completed
/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md
diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index d7907ae43df90629a57549ae7c3129053f608eba..6e565317400cbb364e78db8a66ac79150f29140a
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -2,9 +2,9 @@
 
 ## Goal
 
-Build a small, clean, reusable prompt workflow system for Codex work inside a repository.
+Build a small, conservative prompt workflow system for Codex work inside a repository.
 
-This system should be conservative first:
+This system should stay:
 
 * easy to inspect
 * easy to test
@@ -16,489 +16,452 @@
 
 ---
 
-## Why We Are Re-Architecting
+## V1 Problem Statement
 
-The first draft bundled too many responsibilities into one script. That made it harder to:
+The immediate V1 problem is not "how do we build a larger prompt platform?"
 
-* understand
-* trust
-* test
-* evolve
-* reuse across repositories
+It is:
 
-We want a cleaner structure that can become a template for future repos.
+* how to execute one bounded prompt run
+* how to preserve a durable record of that run
+* how to stop for human review before the next prompt is treated as ready
 
----
+The current gap is that prompt execution exists, but the workflow boundary between:
 
-## What the First Script Was Doing
+* execution
+* durable evidence
+* accepted progress
 
-Below is a decomposition of the responsibilities that were bundled together.
+is still too loose.
 
-### 1. Repo path discovery
+That gap matters because the governing workflow requires:
 
-* figure out where the repository root is
-* infer where `codex_prompts/` and `notes/` live
+* thin slices before broad automation
+* review between iterations
+* durable local history
+* failure evidence that supports refinement instead of blind retry
 
-### 2. Directory validation
+So V1 should solve the smallest workflow problem that makes the sequence inspectable:
 
-* verify the expected folders exist
-* fail cleanly if they do not
+* one run
+* one durable execution record
+* one manual review gate
+* one conservative rule for whether the next prompt may proceed
 
-### 3. Prompt discovery
+---
 
-* scan `codex_prompts/`
-* identify valid prompt files
-* ignore files that do not match the naming convention
+## Why We Are Re-Architecting
 
-### 4. Prompt parsing and indexing
+The first draft bundled too many responsibilities into one script. That made it harder to:
 
-* extract numeric prefixes from filenames
-* sort prompts in execution order
-* establish a stable prompt identity from filename/stem
+* understand
+* trust
+* test
+* evolve
+* reuse across repositories
 
-### 5. Note discovery
+We want a cleaner structure, but V1 should only separate what is necessary to support the governing workflow.
 
-* scan `notes/`
-* find note files that match the naming convention
-* ignore unrelated markdown files
+This is still a design-alignment step, not a broad implementation push.
 
-### 6. Note parsing
+---
 
-* extract prompt name, success/fail state, and timestamp from note filenames
-* turn note filenames into structured metadata
+## V1 Scope
 
-### 7. Status reconstruction
+V1 covers the minimum workflow slice from prompt execution through later human review.
 
-* map prompts to matching notes
-* determine whether each prompt is:
+It includes:
 
-  * UNRUN
-  * SUCCESS
-  * FAIL
-* pick the latest note when there are multiple notes for one prompt
+* selecting and executing one prompt
+* writing one durable execution record in `notes/`
+* keeping execution outcome separate from review outcome
+* stopping the queue at `UNREVIEWED`
+* allowing only an explicit accepted review outcome to release the next prompt
 
-### 8. Prompt selection
+V1 is intentionally about workflow control and artifact clarity, not about architectural breadth.
 
-* find the next unrun prompt
-* find the first failed prompt
-* find a prompt by numeric prefix, base name, or full filename
+---
 
-### 9. Prompt display
+## Out Of Scope For V1
 
-* print a selected prompt to the terminal
+The following items are explicitly out of scope in this stage:
 
-### 10. Retry context assembly
-
-* find the latest failed note for a prompt
-* display the original prompt plus the latest failed note
-* present instructions for retrying
-
-### 11. Note writing
+* runner refactor into many modules
+* dependency-aware scheduling
+* richer queue state machines
+* automatic approval or review routing
+* aggregated reports and dashboards
+* retry intelligence beyond minimal retry linkage
+* token accounting beyond optional lightweight fields
+* broad CLI redesign
+* reusable multi-repo configuration systems
+* structured sidecars, databases, or platform services
 
-* create timestamped note filenames
-* write notes in markdown
-* optionally include the previous note as context
+These may become useful later, but they should not shape V1 beyond clear deferral notes.
 
-### 12. Manual status marking
+---
 
-* let the user record a success or failure
-* attach summary/details text to the note
+## Minimum Artifact Inventory
 
-### 13. Command-line interface
+The governing workflow expects the minimum required artifacts to be explicit.
 
-* parse subcommands like `list`, `next`, `show`, `mark`, `retry-failed`
-* route commands to the right behaviors
+For V1, the minimum inventory is:
 
-### 14. Terminal presentation
+### 1. Governing workflow
 
-* format output for human readability
-* present status tables and retry blocks
+`tools/Project_Design_Workflow.md`
 
----
+This remains the controlling sequence for:
 
-## What I Would Like This System To Do Eventually
+* bounded execution
+* validation
+* review between iterations
+* deferral discipline
 
-These are ideas worth considering, but they should not all be built at once.
+### 2. Architecture document
 
-### A. Better architecture and reuse
+`tools/Codex_Prompt_Workflow_Architecture.md`
 
-* work as a reusable template across repositories
-* isolate config from code
-* support per-repo conventions without rewriting internals
+This document should explain:
 
-### B. Structured metadata
+* the V1 problem being solved
+* the V1 scope boundary
+* the minimum parts and artifacts
+* the conservative implementation order
+* what remains deferred
 
-* optionally store prompt metadata in frontmatter or sidecar files
-* support fields like:
+### 3. V1 execution record definition
 
-  * title
-  * objective
-  * tags
-  * priority
-  * dependencies
-  * owner
+`tools/codex/V1_Execution_Record_Artifact.md`
 
-### C. Better status model
+This defines the minimum durable source of truth for one run.
 
-Instead of only:
+Its role in the architecture is to make one run inspectable without requiring a larger system.
 
-* UNRUN
-* SUCCESS
-* FAIL
+### 4. V1 review gate definition
 
-We may want:
+`tools/codex/V1_Run_Review_Gate.md`
 
-* READY
-* RUNNING
-* BLOCKED
-* NEEDS_REVIEW
-* PARTIAL
-* SKIPPED
-* ARCHIVED
+This defines the smallest manual stop between one completed run and the next prompt becoming ready.
 
-### D. Retry intelligence
+Its role in the architecture is to keep execution success separate from accepted progress.
 
-* carry forward the previous failed note automatically
-* include multiple prior failures, not just the latest one
-* summarize recurring failure patterns
+### 5. Execution record files in `notes/`
 
-### E. Prompt preparation
+For V1, one run should produce one markdown execution-record file in `notes/`.
 
-* generate a clean “execution bundle” for Codex
-* include:
+That record is the minimum durable run artifact.
 
-  * prompt text
-  * repo context
-  * previous failure note if retrying
-  * explicit success criteria
+No additional required queue file, database, or sidecar should be introduced in V1.
 
-### F. Manual and automated modes
+---
 
-* manual mode for conservative workflows
-* dry-run mode for seeing what would happen
-* later, optional Codex CLI integration
+## Minimum Viable Slice
 
-### G. Report generation
+The minimum viable slice is:
 
-* generate queue reports
-* generate status snapshots
-* generate audit/history reports over prompt activity
+1. execute one prompt
+2. write one execution record with `review_status: UNREVIEWED`
+3. stop for human review
+4. update the same execution record to either `ACCEPTED` or `REJECTED`
+5. treat only `ACCEPTED` as releasing the next prompt in sequence
 
-### H. Better note/content model
+This is the smallest slice that proves the workflow rather than only the runner.
 
-* move from filename-only state to richer note contents
-* enforce a consistent note template
-* optionally add structured machine-readable metadata inside notes
+It is intentionally narrower than:
 
-### I. Dependency and sequencing support
+* a generalized queue engine
+* a full prompt-status system
+* a modular tool suite
+* an automation framework
 
-* allow prompts to depend on other prompts
-* prevent prompts from running before prerequisites are complete
+---
 
-### J. Prompt families or lanes
+## Role Of The Execution Record
 
-* support categories such as:
+The execution record is the central V1 artifact.
 
-  * setup
-  * audit
-  * refactor
-  * notebook generation
-  * documentation
+Its purpose is to preserve, in one inspectable markdown file:
 
-### K. Approval gates
+* what prompt was run
+* what happened during execution
+* what Codex returned
+* what lightweight failure or resource evidence was observed
+* what a human later decided about the run
 
-* require human confirmation before marking success
-* require review before allowing dependent prompts to proceed
+For V1:
 
-### L. Logging and diagnostics
+* the record body is the source of truth
+* the file in `notes/` is the durable local history unit
+* one run should not be split across multiple required files
 
-* maintain an execution log
-* record tool errors separately from task failures
-* distinguish infrastructure failure from prompt failure
+This keeps V1 small while still supporting:
 
-### M. File layout that scales cleanly
+* review between iterations
+* failure analysis
+* lightweight cost awareness
+* stable run identity
 
-* support growth without turning into a junk drawer
+The execution record defined in `tools/codex/V1_Execution_Record_Artifact.md` should therefore be treated as a required architectural element, not an implementation detail.
 
 ---
 
-## Proposed Folder Architecture
+## Role Of The Review Gate
 
-A better first structure would be:
+The review gate is the minimum manual checkpoint after execution record creation.
 
-```text
-tools/
-  codex/
-    README.md
-    architecture.md
-    config.py
-    paths.py
-    prompts.py
-    notes.py
-    status.py
-    retry.py
-    cli.py
-    templates/
-      note_success.md
-      note_fail.md
-```
-
-For an even more conservative start, we can go smaller:
-
-```text
-tools/
-  codex/
-    README.md
-    architecture.md
-    paths.py
-    prompts.py
-    notes.py
-    status.py
-```
+Its purpose is to enforce the governing workflow rule that the next bounded step should not proceed only because a process finished successfully.
 
-Then add a tiny entrypoint later if needed.
-
----
-
-## Recommended Responsibility Split
-
-### `paths.py`
+For V1:
 
-Owns:
+* execution writes a record
+* new records begin as `UNREVIEWED`
+* human review decides `ACCEPTED` or `REJECTED`
+* only `ACCEPTED` allows the next prompt to be treated as ready
 
-* repo-root discovery
-* locating `codex_prompts/`
-* locating `notes/`
-* validating required folders
+This means the review gate is not a separate platform subsystem.
 
-Should not own:
+It is the manual transition of a single execution record from:
 
-* note parsing
-* prompt parsing
-* CLI logic
+* `UNREVIEWED`
 
-### `prompts.py`
+to one of:
 
-Owns:
+* `ACCEPTED`
+* `REJECTED`
 
-* prompt discovery
-* filename validation
-* numeric prefix parsing
-* prompt sorting
-* prompt lookup by selector
+The review gate defined in `tools/codex/V1_Run_Review_Gate.md` is therefore part of the architecture boundary for V1, not an optional later add-on.
 
-Should not own:
+---
 
-* note history
-* note writing
-* terminal display
+## What The First Script Was Doing
 
-### `notes.py`
+Below is a decomposition of the responsibilities that were bundled together.
 
-Owns:
+### 1. Repo path discovery
 
-* note discovery
-* note filename parsing
-* latest-note selection
-* note writing
-* note naming convention
+* figure out where the repository root is
+* infer where `codex_prompts/` and `notes/` live
 
-Should not own:
+### 2. Directory validation
 
-* prompt queue decisions
-* CLI logic
+* verify the expected folders exist
+* fail cleanly if they do not
 
-### `status.py`
+### 3. Prompt discovery
 
-Owns:
+* scan `codex_prompts/`
+* identify valid prompt files
+* ignore files that do not match the naming convention
 
-* combining prompt data and note data
-* reconstructing current state
-* selecting unrun/failed/successful prompts
+### 4. Prompt parsing and indexing
 
-Should not own:
+* extract numeric prefixes from filenames
+* sort prompts in execution order
+* establish a stable prompt identity from filename/stem
 
-* printing or file writing
+### 5. Note discovery
 
-### `retry.py`
+* scan `notes/`
+* find note files that match the naming convention
+* ignore unrelated markdown files
 
-Owns:
+### 6. Note parsing
 
-* collecting retry context
-* pairing original prompt with latest failed note
-* preparing a retry bundle
+* extract prompt name, success/fail state, and timestamp from note filenames
+* turn note filenames into structured metadata
 
-Should not own:
+### 7. Status reconstruction
 
-* scanning the repo broadly if other modules already do that
+* map prompts to matching notes
+* determine whether each prompt is:
+  * UNRUN
+  * SUCCESS
+  * FAIL
+* pick the latest note when there are multiple notes for one prompt
 
-### `cli.py`
+### 8. Prompt selection
 
-Owns:
+* find the next unrun prompt
+* find the first failed prompt
+* find a prompt by numeric prefix, base name, or full filename
 
-* argument parsing
-* wiring commands to lower-level modules
-* user-facing terminal behavior
+### 9. Prompt display
 
-Should not own:
+* print a selected prompt to the terminal
 
-* core business logic beyond orchestration
+### 10. Retry context assembly
 
----
+* find the latest failed note for a prompt
+* display the original prompt plus the latest failed note
+* present instructions for retrying
 
-## Minimal First Build
+### 11. Note writing
 
-If we build carefully, the first implementation should do only these things:
+* create timestamped note filenames
+* write notes in markdown
+* optionally include the previous note as context
 
-### Phase 1
+### 12. Manual status marking
 
-1. discover prompts
-2. discover notes
-3. reconstruct status
-4. print status table
+* let the user record a success or failure
+* attach summary/details text to the note
 
-### Phase 2
+### 13. Command-line interface
 
-5. show a prompt
-6. show next unrun prompt
-7. write a manual success/fail note
+* parse subcommands like `list`, `next`, `show`, `mark`, `retry-failed`
+* route commands to the right behaviors
 
-### Phase 3
+### 14. Terminal presentation
 
-8. assemble retry context from failed note + original prompt
+* format output for human readability
+* present status tables and retry blocks
 
-That is enough to start using the system without committing to automation yet.
+This decomposition is still useful, but V1 should not treat every decomposed responsibility as something that must be built now.
 
 ---
 
-## Naming Conventions
+## V1 Architectural Shape
 
-### Prompt files
+The conservative V1 shape is smaller than a full prompt-workflow toolkit.
 
-```text
-001_smoke_test_pipeline.md
-002_repo_inventory_and_status.md
-```
+It needs only these practical responsibilities:
 
-### Note files
+### 1. Prompt resolution
 
-```text
-001_smoke_test_pipeline__SUCCESS__20260415_094500.md
-002_repo_inventory_and_status__FAIL__20260415_095212.md
-```
+Enough logic to identify the intended prompt file safely and reproducibly.
 
-These are simple, inspectable, and script-friendly.
+### 2. Bounded execution
 
----
+Enough runner behavior to execute one prompt and capture the immediate execution facts.
 
-## Design Principles
+### 3. Execution-record writing
 
-### 1. Plain files first
+Enough note policy to write one execution record in the V1 format defined by `tools/codex/V1_Execution_Record_Artifact.md`.
 
-Use the filesystem as the source of truth before introducing databases or hidden state.
+### 4. Review write-back
 
-### 2. Small parts
+Enough manual workflow support to update that same record with the review facts defined by `tools/codex/V1_Run_Review_Gate.md`.
 
-Each file/module should have one clear job.
+### 5. Conservative queue progression
 
-### 3. Human-readable state
+Enough status logic to preserve one rule:
 
-A human should be able to inspect prompts and notes without special tooling.
+* only an accepted reviewed run releases the next prompt
 
-### 4. Conservative defaults
+Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
-No automatic execution unless explicitly enabled later.
+---
 
-### 5. Reusability
+## Implementation Order
 
-This should work as a template for future repositories.
+The implementation order should follow the governing workflow and the queued design decisions:
 
-### 6. Honest state
+1. define the V1 review gate
+2. align the architecture document to the workflow, the execution record, and the review gate
+3. define the smallest bridge-runner change spec needed to emit the V1 execution record
+4. only then implement the bounded runner changes for that slice
+5. validate the slice and inspect the resulting run record before any broader refactor
 
-Do not pretend a task succeeded merely because a command ran.
+This order is conservative on purpose.
 
-### 7. Grow only when pressure demands it
+It reduces the risk of:
 
-No extra cleverness until we truly need it.
+* hard-coding the wrong note model
+* coupling queue progression to process exit
+* building convenience layers before the review loop is stable
+* refactoring modules before the actual V1 artifact boundary is settled
 
 ---
 
-## Questions / Design Decisions To Review
+## Validation And Review Posture
 
-### A. What should count as identity?
-
-Should prompt identity be based only on filename stem, or should we eventually support IDs inside prompt files?
-
-### B. How rich should notes be?
+Validation in V1 should stay small and inspectable.
 
-Do we want plain markdown only, or markdown plus structured frontmatter?
+The minimum posture is:
 
-### C. What statuses do we really want in V1?
+* confirm one prompt can be executed intentionally
+* confirm one execution record is written in the expected V1 shape
+* confirm execution status and review status remain separate
+* confirm a new record stops at `UNREVIEWED`
+* confirm only `ACCEPTED` would release the next prompt
+* confirm `REJECTED` stops progression and preserves enough evidence for the next human decision
 
-Is `UNRUN / SUCCESS / FAIL` enough at first?
+This is not a deep automation plan.
 
-### D. How should retries work?
+It is the minimum validation needed to show that the workflow sequence is being honored:
 
-Should a retry include only the latest failed note, or should it include a short history?
+* execute
+* preserve evidence
+* review
+* decide whether to continue
 
-### E. Should prompts ever depend on one another?
+---
 
-If yes, that affects architecture early.
+## Explicit Deferred Items
 
-### F. When do we introduce Codex execution?
+The following remain intentionally deferred until after the V1 slice is proven:
 
-Do we want:
+### Better architecture and reuse
 
-* never in core tools
-* later as an adapter
-* later as an optional plugin layer
+* per-repo configuration layers
+* reusable templates across many repositories
 
-### G. What belongs in notes versus separate logs?
+### Richer metadata and content models
 
-Human-facing narrative and machine-facing diagnostics may eventually deserve separate homes.
+* prompt frontmatter
+* sidecar metadata
+* machine-readable structured note bodies beyond the V1 markdown record
 
----
+### Richer status and queue models
 
-## Recommended Next Step
+* `READY`
+* `RUNNING`
+* `BLOCKED`
+* `PARTIAL`
+* `SKIPPED`
+* `ARCHIVED`
+* dependency-aware release rules
 
-Before writing new code, review this architecture and decide:
+### Retry intelligence
 
-1. the minimal V1 feature set
-2. the final folder layout for `tools/codex/`
-3. the exact status model for V1
-4. whether notes stay plain markdown only
-5. whether retries should include one failed note or a short chain
+* multi-run retry synthesis
+* recurring failure summaries
+* automatic retry context assembly beyond minimal linkage
 
-After that, implement only the smallest slice needed to make the workflow real.
+### Broader execution preparation
 
----
+* richer execution bundles
+* automated repo-context assembly
+* broader prompt packaging
 
-## Suggested Initial Build Target
+### Reporting and diagnostics
 
-My current recommendation for the smallest clean starting point is:
+* aggregated queue reports
+* dashboards
+* cross-run audit summaries
+* separate diagnostic subsystems
 
-* `tools/codex/paths.py`
-* `tools/codex/prompts.py`
-* `tools/codex/notes.py`
-* `tools/codex/status.py`
-* `tools/codex/README.md`
+### Larger implementation restructuring
 
-No CLI yet unless we decide it is truly needed.
+* broad module decomposition
+* larger CLI surface
+* extensive folder expansion
 
-That would let us validate the architecture before we wrap it in commands.
+These are extension paths, not V1 requirements.
 
 ---
 
-## Closing Thought
+## Extension Path After V1
 
-The right next move is not “build the runner.”
+If V1 proves clean, the next steps should still remain bounded.
 
-The right next move is:
+A sensible post-V1 path would be:
 
-* define the pieces
-* define the responsibilities
-* define the boundaries
-* build the smallest useful slice
-
-That gives us something we can trust, reuse, and extend without regret.
+1. keep the execution record as the stable source artifact
+2. keep the review gate explicit
+3. add only the smallest runner changes needed to support that record reliably
+4. review whether any module split is justified by actual friction
+5. defer broader queue features until the single-run slice has been used enough to reveal real needs
 
+This keeps the architecture reusable and inspectable without turning the design document into a platform roadmap.

diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index d7907ae43df90629a57549ae7c3129053f608eba..6e565317400cbb364e78db8a66ac79150f29140a
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -2,9 +2,9 @@
 
 ## Goal
 
-Build a small, clean, reusable prompt workflow system for Codex work inside a repository.
+Build a small, conservative prompt workflow system for Codex work inside a repository.
 
-This system should be conservative first:
+This system should stay:
 
 * easy to inspect
 * easy to test
@@ -16,489 +16,452 @@
 
 ---
 
-## Why We Are Re-Architecting
+## V1 Problem Statement
 
-The first draft bundled too many responsibilities into one script. That made it harder to:
+The immediate V1 problem is not "how do we build a larger prompt platform?"
 
-* understand
-* trust
-* test
-* evolve
-* reuse across repositories
+It is:
 
-We want a cleaner structure that can become a template for future repos.
+* how to execute one bounded prompt run
+* how to preserve a durable record of that run
+* how to stop for human review before the next prompt is treated as ready
 
----
+The current gap is that prompt execution exists, but the workflow boundary between:
 
-## What the First Script Was Doing
+* execution
+* durable evidence
+* accepted progress
 
-Below is a decomposition of the responsibilities that were bundled together.
+is still too loose.
 
-### 1. Repo path discovery
+That gap matters because the governing workflow requires:
 
-* figure out where the repository root is
-* infer where `codex_prompts/` and `notes/` live
+* thin slices before broad automation
+* review between iterations
+* durable local history
+* failure evidence that supports refinement instead of blind retry
 
-### 2. Directory validation
+So V1 should solve the smallest workflow problem that makes the sequence inspectable:
 
-* verify the expected folders exist
-* fail cleanly if they do not
+* one run
+* one durable execution record
+* one manual review gate
+* one conservative rule for whether the next prompt may proceed
 
-### 3. Prompt discovery
+---
 
-* scan `codex_prompts/`
-* identify valid prompt files
-* ignore files that do not match the naming convention
+## Why We Are Re-Architecting
 
-### 4. Prompt parsing and indexing
+The first draft bundled too many responsibilities into one script. That made it harder to:
 
-* extract numeric prefixes from filenames
-* sort prompts in execution order
-* establish a stable prompt identity from filename/stem
+* understand
+* trust
+* test
+* evolve
+* reuse across repositories
 
-### 5. Note discovery
+We want a cleaner structure, but V1 should only separate what is necessary to support the governing workflow.
 
-* scan `notes/`
-* find note files that match the naming convention
-* ignore unrelated markdown files
+This is still a design-alignment step, not a broad implementation push.
 
-### 6. Note parsing
+---
 
-* extract prompt name, success/fail state, and timestamp from note filenames
-* turn note filenames into structured metadata
+## V1 Scope
 
-### 7. Status reconstruction
+V1 covers the minimum workflow slice from prompt execution through later human review.
 
-* map prompts to matching notes
-* determine whether each prompt is:
+It includes:
 
-  * UNRUN
-  * SUCCESS
-  * FAIL
-* pick the latest note when there are multiple notes for one prompt
+* selecting and executing one prompt
+* writing one durable execution record in `notes/`
+* keeping execution outcome separate from review outcome
+* stopping the queue at `UNREVIEWED`
+* allowing only an explicit accepted review outcome to release the next prompt
 
-### 8. Prompt selection
+V1 is intentionally about workflow control and artifact clarity, not about architectural breadth.
 
-* find the next unrun prompt
-* find the first failed prompt
-* find a prompt by numeric prefix, base name, or full filename
+---
 
-### 9. Prompt display
+## Out Of Scope For V1
 
-* print a selected prompt to the terminal
+The following items are explicitly out of scope in this stage:
 
-### 10. Retry context assembly
-
-* find the latest failed note for a prompt
-* display the original prompt plus the latest failed note
-* present instructions for retrying
-
-### 11. Note writing
+* runner refactor into many modules
+* dependency-aware scheduling
+* richer queue state machines
+* automatic approval or review routing
+* aggregated reports and dashboards
+* retry intelligence beyond minimal retry linkage
+* token accounting beyond optional lightweight fields
+* broad CLI redesign
+* reusable multi-repo configuration systems
+* structured sidecars, databases, or platform services
 
-* create timestamped note filenames
-* write notes in markdown
-* optionally include the previous note as context
+These may become useful later, but they should not shape V1 beyond clear deferral notes.
 
-### 12. Manual status marking
+---
 
-* let the user record a success or failure
-* attach summary/details text to the note
+## Minimum Artifact Inventory
 
-### 13. Command-line interface
+The governing workflow expects the minimum required artifacts to be explicit.
 
-* parse subcommands like `list`, `next`, `show`, `mark`, `retry-failed`
-* route commands to the right behaviors
+For V1, the minimum inventory is:
 
-### 14. Terminal presentation
+### 1. Governing workflow
 
-* format output for human readability
-* present status tables and retry blocks
+`tools/Project_Design_Workflow.md`
 
----
+This remains the controlling sequence for:
 
-## What I Would Like This System To Do Eventually
+* bounded execution
+* validation
+* review between iterations
+* deferral discipline
 
-These are ideas worth considering, but they should not all be built at once.
+### 2. Architecture document
 
-### A. Better architecture and reuse
+`tools/Codex_Prompt_Workflow_Architecture.md`
 
-* work as a reusable template across repositories
-* isolate config from code
-* support per-repo conventions without rewriting internals
+This document should explain:
 
-### B. Structured metadata
+* the V1 problem being solved
+* the V1 scope boundary
+* the minimum parts and artifacts
+* the conservative implementation order
+* what remains deferred
 
-* optionally store prompt metadata in frontmatter or sidecar files
-* support fields like:
+### 3. V1 execution record definition
 
-  * title
-  * objective
-  * tags
-  * priority
-  * dependencies
-  * owner
+`tools/codex/V1_Execution_Record_Artifact.md`
 
-### C. Better status model
+This defines the minimum durable source of truth for one run.
 
-Instead of only:
+Its role in the architecture is to make one run inspectable without requiring a larger system.
 
-* UNRUN
-* SUCCESS
-* FAIL
+### 4. V1 review gate definition
 
-We may want:
+`tools/codex/V1_Run_Review_Gate.md`
 
-* READY
-* RUNNING
-* BLOCKED
-* NEEDS_REVIEW
-* PARTIAL
-* SKIPPED
-* ARCHIVED
+This defines the smallest manual stop between one completed run and the next prompt becoming ready.
 
-### D. Retry intelligence
+Its role in the architecture is to keep execution success separate from accepted progress.
 
-* carry forward the previous failed note automatically
-* include multiple prior failures, not just the latest one
-* summarize recurring failure patterns
+### 5. Execution record files in `notes/`
 
-### E. Prompt preparation
+For V1, one run should produce one markdown execution-record file in `notes/`.
 
-* generate a clean “execution bundle” for Codex
-* include:
+That record is the minimum durable run artifact.
 
-  * prompt text
-  * repo context
-  * previous failure note if retrying
-  * explicit success criteria
+No additional required queue file, database, or sidecar should be introduced in V1.
 
-### F. Manual and automated modes
+---
 
-* manual mode for conservative workflows
-* dry-run mode for seeing what would happen
-* later, optional Codex CLI integration
+## Minimum Viable Slice
 
-### G. Report generation
+The minimum viable slice is:
 
-* generate queue reports
-* generate status snapshots
-* generate audit/history reports over prompt activity
+1. execute one prompt
+2. write one execution record with `review_status: UNREVIEWED`
+3. stop for human review
+4. update the same execution record to either `ACCEPTED` or `REJECTED`
+5. treat only `ACCEPTED` as releasing the next prompt in sequence
 
-### H. Better note/content model
+This is the smallest slice that proves the workflow rather than only the runner.
 
-* move from filename-only state to richer note contents
-* enforce a consistent note template
-* optionally add structured machine-readable metadata inside notes
+It is intentionally narrower than:
 
-### I. Dependency and sequencing support
+* a generalized queue engine
+* a full prompt-status system
+* a modular tool suite
+* an automation framework
 
-* allow prompts to depend on other prompts
-* prevent prompts from running before prerequisites are complete
+---
 
-### J. Prompt families or lanes
+## Role Of The Execution Record
 
-* support categories such as:
+The execution record is the central V1 artifact.
 
-  * setup
-  * audit
-  * refactor
-  * notebook generation
-  * documentation
+Its purpose is to preserve, in one inspectable markdown file:
 
-### K. Approval gates
+* what prompt was run
+* what happened during execution
+* what Codex returned
+* what lightweight failure or resource evidence was observed
+* what a human later decided about the run
 
-* require human confirmation before marking success
-* require review before allowing dependent prompts to proceed
+For V1:
 
-### L. Logging and diagnostics
+* the record body is the source of truth
+* the file in `notes/` is the durable local history unit
+* one run should not be split across multiple required files
 
-* maintain an execution log
-* record tool errors separately from task failures
-* distinguish infrastructure failure from prompt failure
+This keeps V1 small while still supporting:
 
-### M. File layout that scales cleanly
+* review between iterations
+* failure analysis
+* lightweight cost awareness
+* stable run identity
 
-* support growth without turning into a junk drawer
+The execution record defined in `tools/codex/V1_Execution_Record_Artifact.md` should therefore be treated as a required architectural element, not an implementation detail.
 
 ---
 
-## Proposed Folder Architecture
+## Role Of The Review Gate
 
-A better first structure would be:
+The review gate is the minimum manual checkpoint after execution record creation.
 
-```text
-tools/
-  codex/
-    README.md
-    architecture.md
-    config.py
-    paths.py
-    prompts.py
-    notes.py
-    status.py
-    retry.py
-    cli.py
-    templates/
-      note_success.md
-      note_fail.md
-```
-
-For an even more conservative start, we can go smaller:
-
-```text
-tools/
-  codex/
-    README.md
-    architecture.md
-    paths.py
-    prompts.py
-    notes.py
-    status.py
-```
+Its purpose is to enforce the governing workflow rule that the next bounded step should not proceed only because a process finished successfully.
 
-Then add a tiny entrypoint later if needed.
-
----
-
-## Recommended Responsibility Split
-
-### `paths.py`
+For V1:
 
-Owns:
+* execution writes a record
+* new records begin as `UNREVIEWED`
+* human review decides `ACCEPTED` or `REJECTED`
+* only `ACCEPTED` allows the next prompt to be treated as ready
 
-* repo-root discovery
-* locating `codex_prompts/`
-* locating `notes/`
-* validating required folders
+This means the review gate is not a separate platform subsystem.
 
-Should not own:
+It is the manual transition of a single execution record from:
 
-* note parsing
-* prompt parsing
-* CLI logic
+* `UNREVIEWED`
 
-### `prompts.py`
+to one of:
 
-Owns:
+* `ACCEPTED`
+* `REJECTED`
 
-* prompt discovery
-* filename validation
-* numeric prefix parsing
-* prompt sorting
-* prompt lookup by selector
+The review gate defined in `tools/codex/V1_Run_Review_Gate.md` is therefore part of the architecture boundary for V1, not an optional later add-on.
 
-Should not own:
+---
 
-* note history
-* note writing
-* terminal display
+## What The First Script Was Doing
 
-### `notes.py`
+Below is a decomposition of the responsibilities that were bundled together.
 
-Owns:
+### 1. Repo path discovery
 
-* note discovery
-* note filename parsing
-* latest-note selection
-* note writing
-* note naming convention
+* figure out where the repository root is
+* infer where `codex_prompts/` and `notes/` live
 
-Should not own:
+### 2. Directory validation
 
-* prompt queue decisions
-* CLI logic
+* verify the expected folders exist
+* fail cleanly if they do not
 
-### `status.py`
+### 3. Prompt discovery
 
-Owns:
+* scan `codex_prompts/`
+* identify valid prompt files
+* ignore files that do not match the naming convention
 
-* combining prompt data and note data
-* reconstructing current state
-* selecting unrun/failed/successful prompts
+### 4. Prompt parsing and indexing
 
-Should not own:
+* extract numeric prefixes from filenames
+* sort prompts in execution order
+* establish a stable prompt identity from filename/stem
 
-* printing or file writing
+### 5. Note discovery
 
-### `retry.py`
+* scan `notes/`
+* find note files that match the naming convention
+* ignore unrelated markdown files
 
-Owns:
+### 6. Note parsing
 
-* collecting retry context
-* pairing original prompt with latest failed note
-* preparing a retry bundle
+* extract prompt name, success/fail state, and timestamp from note filenames
+* turn note filenames into structured metadata
 
-Should not own:
+### 7. Status reconstruction
 
-* scanning the repo broadly if other modules already do that
+* map prompts to matching notes
+* determine whether each prompt is:
+  * UNRUN
+  * SUCCESS
+  * FAIL
+* pick the latest note when there are multiple notes for one prompt
 
-### `cli.py`
+### 8. Prompt selection
 
-Owns:
+* find the next unrun prompt
+* find the first failed prompt
+* find a prompt by numeric prefix, base name, or full filename
 
-* argument parsing
-* wiring commands to lower-level modules
-* user-facing terminal behavior
+### 9. Prompt display
 
-Should not own:
+* print a selected prompt to the terminal
 
-* core business logic beyond orchestration
+### 10. Retry context assembly
 
----
+* find the latest failed note for a prompt
+* display the original prompt plus the latest failed note
+* present instructions for retrying
 
-## Minimal First Build
+### 11. Note writing
 
-If we build carefully, the first implementation should do only these things:
+* create timestamped note filenames
+* write notes in markdown
+* optionally include the previous note as context
 
-### Phase 1
+### 12. Manual status marking
 
-1. discover prompts
-2. discover notes
-3. reconstruct status
-4. print status table
+* let the user record a success or failure
+* attach summary/details text to the note
 
-### Phase 2
+### 13. Command-line interface
 
-5. show a prompt
-6. show next unrun prompt
-7. write a manual success/fail note
+* parse subcommands like `list`, `next`, `show`, `mark`, `retry-failed`
+* route commands to the right behaviors
 
-### Phase 3
+### 14. Terminal presentation
 
-8. assemble retry context from failed note + original prompt
+* format output for human readability
+* present status tables and retry blocks
 
-That is enough to start using the system without committing to automation yet.
+This decomposition is still useful, but V1 should not treat every decomposed responsibility as something that must be built now.
 
 ---
 
-## Naming Conventions
+## V1 Architectural Shape
 
-### Prompt files
+The conservative V1 shape is smaller than a full prompt-workflow toolkit.
 
-```text
-001_smoke_test_pipeline.md
-002_repo_inventory_and_status.md
-```
+It needs only these practical responsibilities:
 
-### Note files
+### 1. Prompt resolution
 
-```text
-001_smoke_test_pipeline__SUCCESS__20260415_094500.md
-002_repo_inventory_and_status__FAIL__20260415_095212.md
-```
+Enough logic to identify the intended prompt file safely and reproducibly.
 
-These are simple, inspectable, and script-friendly.
+### 2. Bounded execution
 
----
+Enough runner behavior to execute one prompt and capture the immediate execution facts.
 
-## Design Principles
+### 3. Execution-record writing
 
-### 1. Plain files first
+Enough note policy to write one execution record in the V1 format defined by `tools/codex/V1_Execution_Record_Artifact.md`.
 
-Use the filesystem as the source of truth before introducing databases or hidden state.
+### 4. Review write-back
 
-### 2. Small parts
+Enough manual workflow support to update that same record with the review facts defined by `tools/codex/V1_Run_Review_Gate.md`.
 
-Each file/module should have one clear job.
+### 5. Conservative queue progression
 
-### 3. Human-readable state
+Enough status logic to preserve one rule:
 
-A human should be able to inspect prompts and notes without special tooling.
+* only an accepted reviewed run releases the next prompt
 
-### 4. Conservative defaults
+Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
-No automatic execution unless explicitly enabled later.
+---
 
-### 5. Reusability
+## Implementation Order
 
-This should work as a template for future repositories.
+The implementation order should follow the governing workflow and the queued design decisions:
 
-### 6. Honest state
+1. define the V1 review gate
+2. align the architecture document to the workflow, the execution record, and the review gate
+3. define the smallest bridge-runner change spec needed to emit the V1 execution record
+4. only then implement the bounded runner changes for that slice
+5. validate the slice and inspect the resulting run record before any broader refactor
 
-Do not pretend a task succeeded merely because a command ran.
+This order is conservative on purpose.
 
-### 7. Grow only when pressure demands it
+It reduces the risk of:
 
-No extra cleverness until we truly need it.
+* hard-coding the wrong note model
+* coupling queue progression to process exit
+* building convenience layers before the review loop is stable
+* refactoring modules before the actual V1 artifact boundary is settled
 
 ---
 
-## Questions / Design Decisions To Review
+## Validation And Review Posture
 
-### A. What should count as identity?
-
-Should prompt identity be based only on filename stem, or should we eventually support IDs inside prompt files?
-
-### B. How rich should notes be?
+Validation in V1 should stay small and inspectable.
 
-Do we want plain markdown only, or markdown plus structured frontmatter?
+The minimum posture is:
 
-### C. What statuses do we really want in V1?
+* confirm one prompt can be executed intentionally
+* confirm one execution record is written in the expected V1 shape
+* confirm execution status and review status remain separate
+* confirm a new record stops at `UNREVIEWED`
+* confirm only `ACCEPTED` would release the next prompt
+* confirm `REJECTED` stops progression and preserves enough evidence for the next human decision
 
-Is `UNRUN / SUCCESS / FAIL` enough at first?
+This is not a deep automation plan.
 
-### D. How should retries work?
+It is the minimum validation needed to show that the workflow sequence is being honored:
 
-Should a retry include only the latest failed note, or should it include a short history?
+* execute
+* preserve evidence
+* review
+* decide whether to continue
 
-### E. Should prompts ever depend on one another?
+---
 
-If yes, that affects architecture early.
+## Explicit Deferred Items
 
-### F. When do we introduce Codex execution?
+The following remain intentionally deferred until after the V1 slice is proven:
 
-Do we want:
+### Better architecture and reuse
 
-* never in core tools
-* later as an adapter
-* later as an optional plugin layer
+* per-repo configuration layers
+* reusable templates across many repositories
 
-### G. What belongs in notes versus separate logs?
+### Richer metadata and content models
 
-Human-facing narrative and machine-facing diagnostics may eventually deserve separate homes.
+* prompt frontmatter
+* sidecar metadata
+* machine-readable structured note bodies beyond the V1 markdown record
 
----
+### Richer status and queue models
 
-## Recommended Next Step
+* `READY`
+* `RUNNING`
+* `BLOCKED`
+* `PARTIAL`
+* `SKIPPED`
+* `ARCHIVED`
+* dependency-aware release rules
 
-Before writing new code, review this architecture and decide:
+### Retry intelligence
 
-1. the minimal V1 feature set
-2. the final folder layout for `tools/codex/`
-3. the exact status model for V1
-4. whether notes stay plain markdown only
-5. whether retries should include one failed note or a short chain
+* multi-run retry synthesis
+* recurring failure summaries
+* automatic retry context assembly beyond minimal linkage
 
-After that, implement only the smallest slice needed to make the workflow real.
+### Broader execution preparation
 
----
+* richer execution bundles
+* automated repo-context assembly
+* broader prompt packaging
 
-## Suggested Initial Build Target
+### Reporting and diagnostics
 
-My current recommendation for the smallest clean starting point is:
+* aggregated queue reports
+* dashboards
+* cross-run audit summaries
+* separate diagnostic subsystems
 
-* `tools/codex/paths.py`
-* `tools/codex/prompts.py`
-* `tools/codex/notes.py`
-* `tools/codex/status.py`
-* `tools/codex/README.md`
+### Larger implementation restructuring
 
-No CLI yet unless we decide it is truly needed.
+* broad module decomposition
+* larger CLI surface
+* extensive folder expansion
 
-That would let us validate the architecture before we wrap it in commands.
+These are extension paths, not V1 requirements.
 
 ---
 
-## Closing Thought
+## Extension Path After V1
 
-The right next move is not “build the runner.”
+If V1 proves clean, the next steps should still remain bounded.
 
-The right next move is:
+A sensible post-V1 path would be:
 
-* define the pieces
-* define the responsibilities
-* define the boundaries
-* build the smallest useful slice
-
-That gives us something we can trust, reuse, and extend without regret.
+1. keep the execution record as the stable source artifact
+2. keep the review gate explicit
+3. add only the smallest runner changes needed to support that record reliably
+4. review whether any module split is justified by actual friction
+5. defer broader queue features until the single-run slice has been used enough to reveal real needs
 
+This keeps the architecture reusable and inspectable without turning the design document into a platform roadmap.

codex
The document is rewritten around the V1 seam now. I’m doing a quick readback to check that the workflow order, artifact roles, and deferral lines are explicit and that I didn’t accidentally broaden the scope.
exec
/bin/bash -lc "sed -n '1,320p' tools/Codex_Prompt_Workflow_Architecture.md" in /data/git/handson-ml3
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

diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index d7907ae43df90629a57549ae7c3129053f608eba..6e565317400cbb364e78db8a66ac79150f29140a
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -2,9 +2,9 @@
 
 ## Goal
 
-Build a small, clean, reusable prompt workflow system for Codex work inside a repository.
+Build a small, conservative prompt workflow system for Codex work inside a repository.
 
-This system should be conservative first:
+This system should stay:
 
 * easy to inspect
 * easy to test
@@ -16,489 +16,452 @@
 
 ---
 
-## Why We Are Re-Architecting
+## V1 Problem Statement
 
-The first draft bundled too many responsibilities into one script. That made it harder to:
+The immediate V1 problem is not "how do we build a larger prompt platform?"
 
-* understand
-* trust
-* test
-* evolve
-* reuse across repositories
+It is:
 
-We want a cleaner structure that can become a template for future repos.
+* how to execute one bounded prompt run
+* how to preserve a durable record of that run
+* how to stop for human review before the next prompt is treated as ready
 
----
+The current gap is that prompt execution exists, but the workflow boundary between:
 
-## What the First Script Was Doing
+* execution
+* durable evidence
+* accepted progress
 
-Below is a decomposition of the responsibilities that were bundled together.
+is still too loose.
 
-### 1. Repo path discovery
+That gap matters because the governing workflow requires:
 
-* figure out where the repository root is
-* infer where `codex_prompts/` and `notes/` live
+* thin slices before broad automation
+* review between iterations
+* durable local history
+* failure evidence that supports refinement instead of blind retry
 
-### 2. Directory validation
+So V1 should solve the smallest workflow problem that makes the sequence inspectable:
 
-* verify the expected folders exist
-* fail cleanly if they do not
+* one run
+* one durable execution record
+* one manual review gate
+* one conservative rule for whether the next prompt may proceed
 
-### 3. Prompt discovery
+---
 
-* scan `codex_prompts/`
-* identify valid prompt files
-* ignore files that do not match the naming convention
+## Why We Are Re-Architecting
 
-### 4. Prompt parsing and indexing
+The first draft bundled too many responsibilities into one script. That made it harder to:
 
-* extract numeric prefixes from filenames
-* sort prompts in execution order
-* establish a stable prompt identity from filename/stem
+* understand
+* trust
+* test
+* evolve
+* reuse across repositories
 
-### 5. Note discovery
+We want a cleaner structure, but V1 should only separate what is necessary to support the governing workflow.
 
-* scan `notes/`
-* find note files that match the naming convention
-* ignore unrelated markdown files
+This is still a design-alignment step, not a broad implementation push.
 
-### 6. Note parsing
+---
 
-* extract prompt name, success/fail state, and timestamp from note filenames
-* turn note filenames into structured metadata
+## V1 Scope
 
-### 7. Status reconstruction
+V1 covers the minimum workflow slice from prompt execution through later human review.
 
-* map prompts to matching notes
-* determine whether each prompt is:
+It includes:
 
-  * UNRUN
-  * SUCCESS
-  * FAIL
-* pick the latest note when there are multiple notes for one prompt
+* selecting and executing one prompt
+* writing one durable execution record in `notes/`
+* keeping execution outcome separate from review outcome
+* stopping the queue at `UNREVIEWED`
+* allowing only an explicit accepted review outcome to release the next prompt
 
-### 8. Prompt selection
+V1 is intentionally about workflow control and artifact clarity, not about architectural breadth.
 
-* find the next unrun prompt
-* find the first failed prompt
-* find a prompt by numeric prefix, base name, or full filename
+---
 
-### 9. Prompt display
+## Out Of Scope For V1
 
-* print a selected prompt to the terminal
+The following items are explicitly out of scope in this stage:
 
-### 10. Retry context assembly
-
-* find the latest failed note for a prompt
-* display the original prompt plus the latest failed note
-* present instructions for retrying
-
-### 11. Note writing
+* runner refactor into many modules
+* dependency-aware scheduling
+* richer queue state machines
+* automatic approval or review routing
+* aggregated reports and dashboards
+* retry intelligence beyond minimal retry linkage
+* token accounting beyond optional lightweight fields
+* broad CLI redesign
+* reusable multi-repo configuration systems
+* structured sidecars, databases, or platform services
 
-* create timestamped note filenames
-* write notes in markdown
-* optionally include the previous note as context
+These may become useful later, but they should not shape V1 beyond clear deferral notes.
 
-### 12. Manual status marking
+---
 
-* let the user record a success or failure
-* attach summary/details text to the note
+## Minimum Artifact Inventory
 
-### 13. Command-line interface
+The governing workflow expects the minimum required artifacts to be explicit.
 
-* parse subcommands like `list`, `next`, `show`, `mark`, `retry-failed`
-* route commands to the right behaviors
+For V1, the minimum inventory is:
 
-### 14. Terminal presentation
+### 1. Governing workflow
 
-* format output for human readability
-* present status tables and retry blocks
+`tools/Project_Design_Workflow.md`
 
----
+This remains the controlling sequence for:
 
-## What I Would Like This System To Do Eventually
+* bounded execution
+* validation
+* review between iterations
+* deferral discipline
 
-These are ideas worth considering, but they should not all be built at once.
+### 2. Architecture document
 
-### A. Better architecture and reuse
+`tools/Codex_Prompt_Workflow_Architecture.md`
 
-* work as a reusable template across repositories
-* isolate config from code
-* support per-repo conventions without rewriting internals
+This document should explain:
 
-### B. Structured metadata
+* the V1 problem being solved
+* the V1 scope boundary
+* the minimum parts and artifacts
+* the conservative implementation order
+* what remains deferred
 
-* optionally store prompt metadata in frontmatter or sidecar files
-* support fields like:
+### 3. V1 execution record definition
 
-  * title
-  * objective
-  * tags
-  * priority
-  * dependencies
-  * owner
+`tools/codex/V1_Execution_Record_Artifact.md`
 
-### C. Better status model
+This defines the minimum durable source of truth for one run.
 
-Instead of only:
+Its role in the architecture is to make one run inspectable without requiring a larger system.
 
-* UNRUN
-* SUCCESS
-* FAIL
+### 4. V1 review gate definition
 
-We may want:
+`tools/codex/V1_Run_Review_Gate.md`
 
-* READY
-* RUNNING
-* BLOCKED
-* NEEDS_REVIEW
-* PARTIAL
-* SKIPPED
-* ARCHIVED
+This defines the smallest manual stop between one completed run and the next prompt becoming ready.
 
-### D. Retry intelligence
+Its role in the architecture is to keep execution success separate from accepted progress.
 
-* carry forward the previous failed note automatically
-* include multiple prior failures, not just the latest one
-* summarize recurring failure patterns
+### 5. Execution record files in `notes/`
 
-### E. Prompt preparation
+For V1, one run should produce one markdown execution-record file in `notes/`.
 
-* generate a clean “execution bundle” for Codex
-* include:
+That record is the minimum durable run artifact.
 
-  * prompt text
-  * repo context
-  * previous failure note if retrying
-  * explicit success criteria
+No additional required queue file, database, or sidecar should be introduced in V1.
 
-### F. Manual and automated modes
+---
 
-* manual mode for conservative workflows
-* dry-run mode for seeing what would happen
-* later, optional Codex CLI integration
+## Minimum Viable Slice
 
-### G. Report generation
+The minimum viable slice is:
 
-* generate queue reports
-* generate status snapshots
-* generate audit/history reports over prompt activity
+1. execute one prompt
+2. write one execution record with `review_status: UNREVIEWED`
+3. stop for human review
+4. update the same execution record to either `ACCEPTED` or `REJECTED`
+5. treat only `ACCEPTED` as releasing the next prompt in sequence
 
-### H. Better note/content model
+This is the smallest slice that proves the workflow rather than only the runner.
 
-* move from filename-only state to richer note contents
-* enforce a consistent note template
-* optionally add structured machine-readable metadata inside notes
+It is intentionally narrower than:
 
-### I. Dependency and sequencing support
+* a generalized queue engine
+* a full prompt-status system
+* a modular tool suite
+* an automation framework
 
-* allow prompts to depend on other prompts
-* prevent prompts from running before prerequisites are complete
+---
 
-### J. Prompt families or lanes
+## Role Of The Execution Record
 
-* support categories such as:
+The execution record is the central V1 artifact.
 
-  * setup
-  * audit
-  * refactor
-  * notebook generation
-  * documentation
+Its purpose is to preserve, in one inspectable markdown file:
 
-### K. Approval gates
+* what prompt was run
+* what happened during execution
+* what Codex returned
+* what lightweight failure or resource evidence was observed
+* what a human later decided about the run
 
-* require human confirmation before marking success
-* require review before allowing dependent prompts to proceed
+For V1:
 
-### L. Logging and diagnostics
+* the record body is the source of truth
+* the file in `notes/` is the durable local history unit
+* one run should not be split across multiple required files
 
-* maintain an execution log
-* record tool errors separately from task failures
-* distinguish infrastructure failure from prompt failure
+This keeps V1 small while still supporting:
 
-### M. File layout that scales cleanly
+* review between iterations
+* failure analysis
+* lightweight cost awareness
+* stable run identity
 
-* support growth without turning into a junk drawer
+The execution record defined in `tools/codex/V1_Execution_Record_Artifact.md` should therefore be treated as a required architectural element, not an implementation detail.
 
 ---
 
-## Proposed Folder Architecture
+## Role Of The Review Gate
 
-A better first structure would be:
+The review gate is the minimum manual checkpoint after execution record creation.
 
-```text
-tools/
-  codex/
-    README.md
-    architecture.md
-    config.py
-    paths.py
-    prompts.py
-    notes.py
-    status.py
-    retry.py
-    cli.py
-    templates/
-      note_success.md
-      note_fail.md
-```
-
-For an even more conservative start, we can go smaller:
-
-```text
-tools/
-  codex/
-    README.md
-    architecture.md
-    paths.py
-    prompts.py
-    notes.py
-    status.py
-```
+Its purpose is to enforce the governing workflow rule that the next bounded step should not proceed only because a process finished successfully.
 
-Then add a tiny entrypoint later if needed.
-
----
-
-## Recommended Responsibility Split
-
-### `paths.py`
+For V1:
 
-Owns:
+* execution writes a record
+* new records begin as `UNREVIEWED`
+* human review decides `ACCEPTED` or `REJECTED`
+* only `ACCEPTED` allows the next prompt to be treated as ready
 
-* repo-root discovery
-* locating `codex_prompts/`
-* locating `notes/`
-* validating required folders
+This means the review gate is not a separate platform subsystem.
 
-Should not own:
+It is the manual transition of a single execution record from:
 
-* note parsing
-* prompt parsing
-* CLI logic
+* `UNREVIEWED`
 
-### `prompts.py`
+to one of:
 
-Owns:
+* `ACCEPTED`
+* `REJECTED`
 
-* prompt discovery
-* filename validation
-* numeric prefix parsing
-* prompt sorting
-* prompt lookup by selector
+The review gate defined in `tools/codex/V1_Run_Review_Gate.md` is therefore part of the architecture boundary for V1, not an optional later add-on.
 
-Should not own:
+---
 
-* note history
-* note writing
-* terminal display
+## What The First Script Was Doing
 
-### `notes.py`
+Below is a decomposition of the responsibilities that were bundled together.
 
-Owns:
+### 1. Repo path discovery
 
-* note discovery
-* note filename parsing
-* latest-note selection
-* note writing
-* note naming convention
+* figure out where the repository root is
+* infer where `codex_prompts/` and `notes/` live
 
-Should not own:
+### 2. Directory validation
 
-* prompt queue decisions
-* CLI logic
+* verify the expected folders exist
+* fail cleanly if they do not
 
-### `status.py`
+### 3. Prompt discovery
 
-Owns:
+* scan `codex_prompts/`
+* identify valid prompt files
+* ignore files that do not match the naming convention
 
-* combining prompt data and note data
-* reconstructing current state
-* selecting unrun/failed/successful prompts
+### 4. Prompt parsing and indexing
 
-Should not own:
+* extract numeric prefixes from filenames
+* sort prompts in execution order
+* establish a stable prompt identity from filename/stem
 
-* printing or file writing
+### 5. Note discovery
 
-### `retry.py`
+* scan `notes/`
+* find note files that match the naming convention
+* ignore unrelated markdown files
 
-Owns:
+### 6. Note parsing
 
-* collecting retry context
-* pairing original prompt with latest failed note
-* preparing a retry bundle
+* extract prompt name, success/fail state, and timestamp from note filenames
+* turn note filenames into structured metadata
 
-Should not own:
+### 7. Status reconstruction
 
-* scanning the repo broadly if other modules already do that
+* map prompts to matching notes
+* determine whether each prompt is:
+  * UNRUN
+  * SUCCESS
+  * FAIL
+* pick the latest note when there are multiple notes for one prompt
 
-### `cli.py`
+### 8. Prompt selection
 
-Owns:
+* find the next unrun prompt
+* find the first failed prompt
+* find a prompt by numeric prefix, base name, or full filename
 
-* argument parsing
-* wiring commands to lower-level modules
-* user-facing terminal behavior
+### 9. Prompt display
 
-Should not own:
+* print a selected prompt to the terminal
 
-* core business logic beyond orchestration
+### 10. Retry context assembly
 
----
+* find the latest failed note for a prompt
+* display the original prompt plus the latest failed note
+* present instructions for retrying
 
-## Minimal First Build
+### 11. Note writing
 
-If we build carefully, the first implementation should do only these things:
+* create timestamped note filenames
+* write notes in markdown
+* optionally include the previous note as context
 
-### Phase 1
+### 12. Manual status marking
 
-1. discover prompts
-2. discover notes
-3. reconstruct status
-4. print status table
+* let the user record a success or failure
+* attach summary/details text to the note
 
-### Phase 2
+### 13. Command-line interface
 
-5. show a prompt
-6. show next unrun prompt
-7. write a manual success/fail note
+* parse subcommands like `list`, `next`, `show`, `mark`, `retry-failed`
+* route commands to the right behaviors
 
-### Phase 3
+### 14. Terminal presentation
 
-8. assemble retry context from failed note + original prompt
+* format output for human readability
+* present status tables and retry blocks
 
-That is enough to start using the system without committing to automation yet.
+This decomposition is still useful, but V1 should not treat every decomposed responsibility as something that must be built now.
 
 ---
 
-## Naming Conventions
+## V1 Architectural Shape
 
-### Prompt files
+The conservative V1 shape is smaller than a full prompt-workflow toolkit.
 
-```text
-001_smoke_test_pipeline.md
-002_repo_inventory_and_status.md
-```
+It needs only these practical responsibilities:
 
-### Note files
+### 1. Prompt resolution
 
-```text
-001_smoke_test_pipeline__SUCCESS__20260415_094500.md
-002_repo_inventory_and_status__FAIL__20260415_095212.md
-```
+Enough logic to identify the intended prompt file safely and reproducibly.
 
-These are simple, inspectable, and script-friendly.
+### 2. Bounded execution
 
----
+Enough runner behavior to execute one prompt and capture the immediate execution facts.
 
-## Design Principles
+### 3. Execution-record writing
 
-### 1. Plain files first
+Enough note policy to write one execution record in the V1 format defined by `tools/codex/V1_Execution_Record_Artifact.md`.
 
-Use the filesystem as the source of truth before introducing databases or hidden state.
+### 4. Review write-back
 
-### 2. Small parts
+Enough manual workflow support to update that same record with the review facts defined by `tools/codex/V1_Run_Review_Gate.md`.
 
-Each file/module should have one clear job.
+### 5. Conservative queue progression
 
-### 3. Human-readable state
+Enough status logic to preserve one rule:
 
-A human should be able to inspect prompts and notes without special tooling.
+* only an accepted reviewed run releases the next prompt
 
-### 4. Conservative defaults
+Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
-No automatic execution unless explicitly enabled later.
+---
 
-### 5. Reusability
+## Implementation Order
 
-This should work as a template for future repositories.
+The implementation order should follow the governing workflow and the queued design decisions:
 
-### 6. Honest state
+1. define the V1 review gate
+2. align the architecture document to the workflow, the execution record, and the review gate
+3. define the smallest bridge-runner change spec needed to emit the V1 execution record
+4. only then implement the bounded runner changes for that slice
+5. validate the slice and inspect the resulting run record before any broader refactor
 
-Do not pretend a task succeeded merely because a command ran.
+This order is conservative on purpose.
 
-### 7. Grow only when pressure demands it
+It reduces the risk of:
 
-No extra cleverness until we truly need it.
+* hard-coding the wrong note model
+* coupling queue progression to process exit
+* building convenience layers before the review loop is stable
+* refactoring modules before the actual V1 artifact boundary is settled
 
 ---
 
-## Questions / Design Decisions To Review
+## Validation And Review Posture
 
-### A. What should count as identity?
-
-Should prompt identity be based only on filename stem, or should we eventually support IDs inside prompt files?
-
-### B. How rich should notes be?
+Validation in V1 should stay small and inspectable.
 
-Do we want plain markdown only, or markdown plus structured frontmatter?
+The minimum posture is:
 
-### C. What statuses do we really want in V1?
+* confirm one prompt can be executed intentionally
+* confirm one execution record is written in the expected V1 shape
+* confirm execution status and review status remain separate
+* confirm a new record stops at `UNREVIEWED`
+* confirm only `ACCEPTED` would release the next prompt
+* confirm `REJECTED` stops progression and preserves enough evidence for the next human decision
 
-Is `UNRUN / SUCCESS / FAIL` enough at first?
+This is not a deep automation plan.
 
-### D. How should retries work?
+It is the minimum validation needed to show that the workflow sequence is being honored:
 
-Should a retry include only the latest failed note, or should it include a short history?
+* execute
+* preserve evidence
+* review
+* decide whether to continue
 
-### E. Should prompts ever depend on one another?
+---
 
-If yes, that affects architecture early.
+## Explicit Deferred Items
 
-### F. When do we introduce Codex execution?
+The following remain intentionally deferred until after the V1 slice is proven:
 
-Do we want:
+### Better architecture and reuse
 
-* never in core tools
-* later as an adapter
-* later as an optional plugin layer
+* per-repo configuration layers
+* reusable templates across many repositories
 
-### G. What belongs in notes versus separate logs?
+### Richer metadata and content models
 
-Human-facing narrative and machine-facing diagnostics may eventually deserve separate homes.
+* prompt frontmatter
+* sidecar metadata
+* machine-readable structured note bodies beyond the V1 markdown record
 
----
+### Richer status and queue models
 
-## Recommended Next Step
+* `READY`
+* `RUNNING`
+* `BLOCKED`
+* `PARTIAL`
+* `SKIPPED`
+* `ARCHIVED`
+* dependency-aware release rules
 
-Before writing new code, review this architecture and decide:
+### Retry intelligence
 
-1. the minimal V1 feature set
-2. the final folder layout for `tools/codex/`
-3. the exact status model for V1
-4. whether notes stay plain markdown only
-5. whether retries should include one failed note or a short chain
+* multi-run retry synthesis
+* recurring failure summaries
+* automatic retry context assembly beyond minimal linkage
 
-After that, implement only the smallest slice needed to make the workflow real.
+### Broader execution preparation
 
----
+* richer execution bundles
+* automated repo-context assembly
+* broader prompt packaging
 
-## Suggested Initial Build Target
+### Reporting and diagnostics
 
-My current recommendation for the smallest clean starting point is:
+* aggregated queue reports
+* dashboards
+* cross-run audit summaries
+* separate diagnostic subsystems
 
-* `tools/codex/paths.py`
-* `tools/codex/prompts.py`
-* `tools/codex/notes.py`
-* `tools/codex/status.py`
-* `tools/codex/README.md`
+### Larger implementation restructuring
 
-No CLI yet unless we decide it is truly needed.
+* broad module decomposition
+* larger CLI surface
+* extensive folder expansion
 
-That would let us validate the architecture before we wrap it in commands.
+These are extension paths, not V1 requirements.
 
 ---
 
-## Closing Thought
+## Extension Path After V1
 
-The right next move is not “build the runner.”
+If V1 proves clean, the next steps should still remain bounded.
 
-The right next move is:
+A sensible post-V1 path would be:
 
-* define the pieces
-* define the responsibilities
-* define the boundaries
-* build the smallest useful slice
-
-That gives us something we can trust, reuse, and extend without regret.
+1. keep the execution record as the stable source artifact
+2. keep the review gate explicit
+3. add only the smallest runner changes needed to support that record reliably
+4. review whether any module split is justified by actual friction
+5. defer broader queue features until the single-run slice has been used enough to reveal real needs
 
+This keeps the architecture reusable and inspectable without turning the design document into a platform roadmap.

exec
/bin/bash -lc "sed -n '321,520p' tools/Codex_Prompt_Workflow_Architecture.md" in /data/git/handson-ml3
 succeeded in 0ms:

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

diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index d7907ae43df90629a57549ae7c3129053f608eba..6e565317400cbb364e78db8a66ac79150f29140a
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -2,9 +2,9 @@
 
 ## Goal
 
-Build a small, clean, reusable prompt workflow system for Codex work inside a repository.
+Build a small, conservative prompt workflow system for Codex work inside a repository.
 
-This system should be conservative first:
+This system should stay:
 
 * easy to inspect
 * easy to test
@@ -16,489 +16,452 @@
 
 ---
 
-## Why We Are Re-Architecting
+## V1 Problem Statement
 
-The first draft bundled too many responsibilities into one script. That made it harder to:
+The immediate V1 problem is not "how do we build a larger prompt platform?"
 
-* understand
-* trust
-* test
-* evolve
-* reuse across repositories
+It is:
 
-We want a cleaner structure that can become a template for future repos.
+* how to execute one bounded prompt run
+* how to preserve a durable record of that run
+* how to stop for human review before the next prompt is treated as ready
 
----
+The current gap is that prompt execution exists, but the workflow boundary between:
 
-## What the First Script Was Doing
+* execution
+* durable evidence
+* accepted progress
 
-Below is a decomposition of the responsibilities that were bundled together.
+is still too loose.
 
-### 1. Repo path discovery
+That gap matters because the governing workflow requires:
 
-* figure out where the repository root is
-* infer where `codex_prompts/` and `notes/` live
+* thin slices before broad automation
+* review between iterations
+* durable local history
+* failure evidence that supports refinement instead of blind retry
 
-### 2. Directory validation
+So V1 should solve the smallest workflow problem that makes the sequence inspectable:
 
-* verify the expected folders exist
-* fail cleanly if they do not
+* one run
+* one durable execution record
+* one manual review gate
+* one conservative rule for whether the next prompt may proceed
 
-### 3. Prompt discovery
+---
 
-* scan `codex_prompts/`
-* identify valid prompt files
-* ignore files that do not match the naming convention
+## Why We Are Re-Architecting
 
-### 4. Prompt parsing and indexing
+The first draft bundled too many responsibilities into one script. That made it harder to:
 
-* extract numeric prefixes from filenames
-* sort prompts in execution order
-* establish a stable prompt identity from filename/stem
+* understand
+* trust
+* test
+* evolve
+* reuse across repositories
 
-### 5. Note discovery
+We want a cleaner structure, but V1 should only separate what is necessary to support the governing workflow.
 
-* scan `notes/`
-* find note files that match the naming convention
-* ignore unrelated markdown files
+This is still a design-alignment step, not a broad implementation push.
 
-### 6. Note parsing
+---
 
-* extract prompt name, success/fail state, and timestamp from note filenames
-* turn note filenames into structured metadata
+## V1 Scope
 
-### 7. Status reconstruction
+V1 covers the minimum workflow slice from prompt execution through later human review.
 
-* map prompts to matching notes
-* determine whether each prompt is:
+It includes:
 
-  * UNRUN
-  * SUCCESS
-  * FAIL
-* pick the latest note when there are multiple notes for one prompt
+* selecting and executing one prompt
+* writing one durable execution record in `notes/`
+* keeping execution outcome separate from review outcome
+* stopping the queue at `UNREVIEWED`
+* allowing only an explicit accepted review outcome to release the next prompt
 
-### 8. Prompt selection
+V1 is intentionally about workflow control and artifact clarity, not about architectural breadth.
 
-* find the next unrun prompt
-* find the first failed prompt
-* find a prompt by numeric prefix, base name, or full filename
+---
 
-### 9. Prompt display
+## Out Of Scope For V1
 
-* print a selected prompt to the terminal
+The following items are explicitly out of scope in this stage:
 
-### 10. Retry context assembly
-
-* find the latest failed note for a prompt
-* display the original prompt plus the latest failed note
-* present instructions for retrying
-
-### 11. Note writing
+* runner refactor into many modules
+* dependency-aware scheduling
+* richer queue state machines
+* automatic approval or review routing
+* aggregated reports and dashboards
+* retry intelligence beyond minimal retry linkage
+* token accounting beyond optional lightweight fields
+* broad CLI redesign
+* reusable multi-repo configuration systems
+* structured sidecars, databases, or platform services
 
-* create timestamped note filenames
-* write notes in markdown
-* optionally include the previous note as context
+These may become useful later, but they should not shape V1 beyond clear deferral notes.
 
-### 12. Manual status marking
+---
 
-* let the user record a success or failure
-* attach summary/details text to the note
+## Minimum Artifact Inventory
 
-### 13. Command-line interface
+The governing workflow expects the minimum required artifacts to be explicit.
 
-* parse subcommands like `list`, `next`, `show`, `mark`, `retry-failed`
-* route commands to the right behaviors
+For V1, the minimum inventory is:
 
-### 14. Terminal presentation
+### 1. Governing workflow
 
-* format output for human readability
-* present status tables and retry blocks
+`tools/Project_Design_Workflow.md`
 
----
+This remains the controlling sequence for:
 
-## What I Would Like This System To Do Eventually
+* bounded execution
+* validation
+* review between iterations
+* deferral discipline
 
-These are ideas worth considering, but they should not all be built at once.
+### 2. Architecture document
 
-### A. Better architecture and reuse
+`tools/Codex_Prompt_Workflow_Architecture.md`
 
-* work as a reusable template across repositories
-* isolate config from code
-* support per-repo conventions without rewriting internals
+This document should explain:
 
-### B. Structured metadata
+* the V1 problem being solved
+* the V1 scope boundary
+* the minimum parts and artifacts
+* the conservative implementation order
+* what remains deferred
 
-* optionally store prompt metadata in frontmatter or sidecar files
-* support fields like:
+### 3. V1 execution record definition
 
-  * title
-  * objective
-  * tags
-  * priority
-  * dependencies
-  * owner
+`tools/codex/V1_Execution_Record_Artifact.md`
 
-### C. Better status model
+This defines the minimum durable source of truth for one run.
 
-Instead of only:
+Its role in the architecture is to make one run inspectable without requiring a larger system.
 
-* UNRUN
-* SUCCESS
-* FAIL
+### 4. V1 review gate definition
 
-We may want:
+`tools/codex/V1_Run_Review_Gate.md`
 
-* READY
-* RUNNING
-* BLOCKED
-* NEEDS_REVIEW
-* PARTIAL
-* SKIPPED
-* ARCHIVED
+This defines the smallest manual stop between one completed run and the next prompt becoming ready.
 
-### D. Retry intelligence
+Its role in the architecture is to keep execution success separate from accepted progress.
 
-* carry forward the previous failed note automatically
-* include multiple prior failures, not just the latest one
-* summarize recurring failure patterns
+### 5. Execution record files in `notes/`
 
-### E. Prompt preparation
+For V1, one run should produce one markdown execution-record file in `notes/`.
 
-* generate a clean “execution bundle” for Codex
-* include:
+That record is the minimum durable run artifact.
 
-  * prompt text
-  * repo context
-  * previous failure note if retrying
-  * explicit success criteria
+No additional required queue file, database, or sidecar should be introduced in V1.
 
-### F. Manual and automated modes
+---
 
-* manual mode for conservative workflows
-* dry-run mode for seeing what would happen
-* later, optional Codex CLI integration
+## Minimum Viable Slice
 
-### G. Report generation
+The minimum viable slice is:
 
-* generate queue reports
-* generate status snapshots
-* generate audit/history reports over prompt activity
+1. execute one prompt
+2. write one execution record with `review_status: UNREVIEWED`
+3. stop for human review
+4. update the same execution record to either `ACCEPTED` or `REJECTED`
+5. treat only `ACCEPTED` as releasing the next prompt in sequence
 
-### H. Better note/content model
+This is the smallest slice that proves the workflow rather than only the runner.
 
-* move from filename-only state to richer note contents
-* enforce a consistent note template
-* optionally add structured machine-readable metadata inside notes
+It is intentionally narrower than:
 
-### I. Dependency and sequencing support
+* a generalized queue engine
+* a full prompt-status system
+* a modular tool suite
+* an automation framework
 
-* allow prompts to depend on other prompts
-* prevent prompts from running before prerequisites are complete
+---
 
-### J. Prompt families or lanes
+## Role Of The Execution Record
 
-* support categories such as:
+The execution record is the central V1 artifact.
 
-  * setup
-  * audit
-  * refactor
-  * notebook generation
-  * documentation
+Its purpose is to preserve, in one inspectable markdown file:
 
-### K. Approval gates
+* what prompt was run
+* what happened during execution
+* what Codex returned
+* what lightweight failure or resource evidence was observed
+* what a human later decided about the run
 
-* require human confirmation before marking success
-* require review before allowing dependent prompts to proceed
+For V1:
 
-### L. Logging and diagnostics
+* the record body is the source of truth
+* the file in `notes/` is the durable local history unit
+* one run should not be split across multiple required files
 
-* maintain an execution log
-* record tool errors separately from task failures
-* distinguish infrastructure failure from prompt failure
+This keeps V1 small while still supporting:
 
-### M. File layout that scales cleanly
+* review between iterations
+* failure analysis
+* lightweight cost awareness
+* stable run identity
 
-* support growth without turning into a junk drawer
+The execution record defined in `tools/codex/V1_Execution_Record_Artifact.md` should therefore be treated as a required architectural element, not an implementation detail.
 
 ---
 
-## Proposed Folder Architecture
+## Role Of The Review Gate
 
-A better first structure would be:
+The review gate is the minimum manual checkpoint after execution record creation.
 
-```text
-tools/
-  codex/
-    README.md
-    architecture.md
-    config.py
-    paths.py
-    prompts.py
-    notes.py
-    status.py
-    retry.py
-    cli.py
-    templates/
-      note_success.md
-      note_fail.md
-```
-
-For an even more conservative start, we can go smaller:
-
-```text
-tools/
-  codex/
-    README.md
-    architecture.md
-    paths.py
-    prompts.py
-    notes.py
-    status.py
-```
+Its purpose is to enforce the governing workflow rule that the next bounded step should not proceed only because a process finished successfully.
 
-Then add a tiny entrypoint later if needed.
-
----
-
-## Recommended Responsibility Split
-
-### `paths.py`
+For V1:
 
-Owns:
+* execution writes a record
+* new records begin as `UNREVIEWED`
+* human review decides `ACCEPTED` or `REJECTED`
+* only `ACCEPTED` allows the next prompt to be treated as ready
 
-* repo-root discovery
-* locating `codex_prompts/`
-* locating `notes/`
-* validating required folders
+This means the review gate is not a separate platform subsystem.
 
-Should not own:
+It is the manual transition of a single execution record from:
 
-* note parsing
-* prompt parsing
-* CLI logic
+* `UNREVIEWED`
 
-### `prompts.py`
+to one of:
 
-Owns:
+* `ACCEPTED`
+* `REJECTED`
 
-* prompt discovery
-* filename validation
-* numeric prefix parsing
-* prompt sorting
-* prompt lookup by selector
+The review gate defined in `tools/codex/V1_Run_Review_Gate.md` is therefore part of the architecture boundary for V1, not an optional later add-on.
 
-Should not own:
+---
 
-* note history
-* note writing
-* terminal display
+## What The First Script Was Doing
 
-### `notes.py`
+Below is a decomposition of the responsibilities that were bundled together.
 
-Owns:
+### 1. Repo path discovery
 
-* note discovery
-* note filename parsing
-* latest-note selection
-* note writing
-* note naming convention
+* figure out where the repository root is
+* infer where `codex_prompts/` and `notes/` live
 
-Should not own:
+### 2. Directory validation
 
-* prompt queue decisions
-* CLI logic
+* verify the expected folders exist
+* fail cleanly if they do not
 
-### `status.py`
+### 3. Prompt discovery
 
-Owns:
+* scan `codex_prompts/`
+* identify valid prompt files
+* ignore files that do not match the naming convention
 
-* combining prompt data and note data
-* reconstructing current state
-* selecting unrun/failed/successful prompts
+### 4. Prompt parsing and indexing
 
-Should not own:
+* extract numeric prefixes from filenames
+* sort prompts in execution order
+* establish a stable prompt identity from filename/stem
 
-* printing or file writing
+### 5. Note discovery
 
-### `retry.py`
+* scan `notes/`
+* find note files that match the naming convention
+* ignore unrelated markdown files
 
-Owns:
+### 6. Note parsing
 
-* collecting retry context
-* pairing original prompt with latest failed note
-* preparing a retry bundle
+* extract prompt name, success/fail state, and timestamp from note filenames
+* turn note filenames into structured metadata
 
-Should not own:
+### 7. Status reconstruction
 
-* scanning the repo broadly if other modules already do that
+* map prompts to matching notes
+* determine whether each prompt is:
+  * UNRUN
+  * SUCCESS
+  * FAIL
+* pick the latest note when there are multiple notes for one prompt
 
-### `cli.py`
+### 8. Prompt selection
 
-Owns:
+* find the next unrun prompt
+* find the first failed prompt
+* find a prompt by numeric prefix, base name, or full filename
 
-* argument parsing
-* wiring commands to lower-level modules
-* user-facing terminal behavior
+### 9. Prompt display
 
-Should not own:
+* print a selected prompt to the terminal
 
-* core business logic beyond orchestration
+### 10. Retry context assembly
 
----
+* find the latest failed note for a prompt
+* display the original prompt plus the latest failed note
+* present instructions for retrying
 
-## Minimal First Build
+### 11. Note writing
 
-If we build carefully, the first implementation should do only these things:
+* create timestamped note filenames
+* write notes in markdown
+* optionally include the previous note as context
 
-### Phase 1
+### 12. Manual status marking
 
-1. discover prompts
-2. discover notes
-3. reconstruct status
-4. print status table
+* let the user record a success or failure
+* attach summary/details text to the note
 
-### Phase 2
+### 13. Command-line interface
 
-5. show a prompt
-6. show next unrun prompt
-7. write a manual success/fail note
+* parse subcommands like `list`, `next`, `show`, `mark`, `retry-failed`
+* route commands to the right behaviors
 
-### Phase 3
+### 14. Terminal presentation
 
-8. assemble retry context from failed note + original prompt
+* format output for human readability
+* present status tables and retry blocks
 
-That is enough to start using the system without committing to automation yet.
+This decomposition is still useful, but V1 should not treat every decomposed responsibility as something that must be built now.
 
 ---
 
-## Naming Conventions
+## V1 Architectural Shape
 
-### Prompt files
+The conservative V1 shape is smaller than a full prompt-workflow toolkit.
 
-```text
-001_smoke_test_pipeline.md
-002_repo_inventory_and_status.md
-```
+It needs only these practical responsibilities:
 
-### Note files
+### 1. Prompt resolution
 
-```text
-001_smoke_test_pipeline__SUCCESS__20260415_094500.md
-002_repo_inventory_and_status__FAIL__20260415_095212.md
-```
+Enough logic to identify the intended prompt file safely and reproducibly.
 
-These are simple, inspectable, and script-friendly.
+### 2. Bounded execution
 
----
+Enough runner behavior to execute one prompt and capture the immediate execution facts.
 
-## Design Principles
+### 3. Execution-record writing
 
-### 1. Plain files first
+Enough note policy to write one execution record in the V1 format defined by `tools/codex/V1_Execution_Record_Artifact.md`.
 
-Use the filesystem as the source of truth before introducing databases or hidden state.
+### 4. Review write-back
 
-### 2. Small parts
+Enough manual workflow support to update that same record with the review facts defined by `tools/codex/V1_Run_Review_Gate.md`.
 
-Each file/module should have one clear job.
+### 5. Conservative queue progression
 
-### 3. Human-readable state
+Enough status logic to preserve one rule:
 
-A human should be able to inspect prompts and notes without special tooling.
+* only an accepted reviewed run releases the next prompt
 
-### 4. Conservative defaults
+Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
-No automatic execution unless explicitly enabled later.
+---
 
-### 5. Reusability
+## Implementation Order
 
-This should work as a template for future repositories.
+The implementation order should follow the governing workflow and the queued design decisions:
 
-### 6. Honest state
+1. define the V1 review gate
+2. align the architecture document to the workflow, the execution record, and the review gate
+3. define the smallest bridge-runner change spec needed to emit the V1 execution record
+4. only then implement the bounded runner changes for that slice
+5. validate the slice and inspect the resulting run record before any broader refactor
 
-Do not pretend a task succeeded merely because a command ran.
+This order is conservative on purpose.
 
-### 7. Grow only when pressure demands it
+It reduces the risk of:
 
-No extra cleverness until we truly need it.
+* hard-coding the wrong note model
+* coupling queue progression to process exit
+* building convenience layers before the review loop is stable
+* refactoring modules before the actual V1 artifact boundary is settled
 
 ---
 
-## Questions / Design Decisions To Review
+## Validation And Review Posture
 
-### A. What should count as identity?
-
-Should prompt identity be based only on filename stem, or should we eventually support IDs inside prompt files?
-
-### B. How rich should notes be?
+Validation in V1 should stay small and inspectable.
 
-Do we want plain markdown only, or markdown plus structured frontmatter?
+The minimum posture is:
 
-### C. What statuses do we really want in V1?
+* confirm one prompt can be executed intentionally
+* confirm one execution record is written in the expected V1 shape
+* confirm execution status and review status remain separate
+* confirm a new record stops at `UNREVIEWED`
+* confirm only `ACCEPTED` would release the next prompt
+* confirm `REJECTED` stops progression and preserves enough evidence for the next human decision
 
-Is `UNRUN / SUCCESS / FAIL` enough at first?
+This is not a deep automation plan.
 
-### D. How should retries work?
+It is the minimum validation needed to show that the workflow sequence is being honored:
 
-Should a retry include only the latest failed note, or should it include a short history?
+* execute
+* preserve evidence
+* review
+* decide whether to continue
 
-### E. Should prompts ever depend on one another?
+---
 
-If yes, that affects architecture early.
+## Explicit Deferred Items
 
-### F. When do we introduce Codex execution?
+The following remain intentionally deferred until after the V1 slice is proven:
 
-Do we want:
+### Better architecture and reuse
 
-* never in core tools
-* later as an adapter
-* later as an optional plugin layer
+* per-repo configuration layers
+* reusable templates across many repositories
 
-### G. What belongs in notes versus separate logs?
+### Richer metadata and content models
 
-Human-facing narrative and machine-facing diagnostics may eventually deserve separate homes.
+* prompt frontmatter
+* sidecar metadata
+* machine-readable structured note bodies beyond the V1 markdown record
 
----
+### Richer status and queue models
 
-## Recommended Next Step
+* `READY`
+* `RUNNING`
+* `BLOCKED`
+* `PARTIAL`
+* `SKIPPED`
+* `ARCHIVED`
+* dependency-aware release rules
 
-Before writing new code, review this architecture and decide:
+### Retry intelligence
 
-1. the minimal V1 feature set
-2. the final folder layout for `tools/codex/`
-3. the exact status model for V1
-4. whether notes stay plain markdown only
-5. whether retries should include one failed note or a short chain
+* multi-run retry synthesis
+* recurring failure summaries
+* automatic retry context assembly beyond minimal linkage
 
-After that, implement only the smallest slice needed to make the workflow real.
+### Broader execution preparation
 
----
+* richer execution bundles
+* automated repo-context assembly
+* broader prompt packaging
 
-## Suggested Initial Build Target
+### Reporting and diagnostics
 
-My current recommendation for the smallest clean starting point is:
+* aggregated queue reports
+* dashboards
+* cross-run audit summaries
+* separate diagnostic subsystems
 
-* `tools/codex/paths.py`
-* `tools/codex/prompts.py`
-* `tools/codex/notes.py`
-* `tools/codex/status.py`
-* `tools/codex/README.md`
+### Larger implementation restructuring
 
-No CLI yet unless we decide it is truly needed.
+* broad module decomposition
+* larger CLI surface
+* extensive folder expansion
 
-That would let us validate the architecture before we wrap it in commands.
+These are extension paths, not V1 requirements.
 
 ---
 
-## Closing Thought
+## Extension Path After V1
 
-The right next move is not “build the runner.”
+If V1 proves clean, the next steps should still remain bounded.
 
-The right next move is:
+A sensible post-V1 path would be:
 
-* define the pieces
-* define the responsibilities
-* define the boundaries
-* build the smallest useful slice
-
-That gives us something we can trust, reuse, and extend without regret.
+1. keep the execution record as the stable source artifact
+2. keep the review gate explicit
+3. add only the smallest runner changes needed to support that record reliably
+4. review whether any module split is justified by actual friction
+5. defer broader queue features until the single-run slice has been used enough to reveal real needs
 
+This keeps the architecture reusable and inspectable without turning the design document into a platform roadmap.

codex
Revised [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md) to align it explicitly with the governing workflow and the V1 artifacts.

The document now states the V1 problem, scope boundary, out-of-scope items, minimum artifact inventory, minimum viable slice, implementation order, validation/review posture, and explicit deferrals. It also makes the roles of [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md) and [V1_Run_Review_Gate.md](/data/git/handson-ml3/tools/codex/V1_Run_Review_Gate.md) architectural rather than optional future ideas, and it keeps the conservative tone without expanding into a broader platform roadmap.

No code was changed and no tests were run, since this pass was a document alignment edit only.
diff --git a/tools/Codex_Prompt_Workflow_Architecture.md b/tools/Codex_Prompt_Workflow_Architecture.md
index d7907ae43df90629a57549ae7c3129053f608eba..6e565317400cbb364e78db8a66ac79150f29140a
--- a/tools/Codex_Prompt_Workflow_Architecture.md
+++ b/tools/Codex_Prompt_Workflow_Architecture.md
@@ -2,9 +2,9 @@
 
 ## Goal
 
-Build a small, clean, reusable prompt workflow system for Codex work inside a repository.
+Build a small, conservative prompt workflow system for Codex work inside a repository.
 
-This system should be conservative first:
+This system should stay:
 
 * easy to inspect
 * easy to test
@@ -16,489 +16,452 @@
 
 ---
 
-## Why We Are Re-Architecting
+## V1 Problem Statement
 
-The first draft bundled too many responsibilities into one script. That made it harder to:
+The immediate V1 problem is not "how do we build a larger prompt platform?"
 
-* understand
-* trust
-* test
-* evolve
-* reuse across repositories
+It is:
 
-We want a cleaner structure that can become a template for future repos.
+* how to execute one bounded prompt run
+* how to preserve a durable record of that run
+* how to stop for human review before the next prompt is treated as ready
 
----
+The current gap is that prompt execution exists, but the workflow boundary between:
 
-## What the First Script Was Doing
+* execution
+* durable evidence
+* accepted progress
 
-Below is a decomposition of the responsibilities that were bundled together.
+is still too loose.
 
-### 1. Repo path discovery
+That gap matters because the governing workflow requires:
 
-* figure out where the repository root is
-* infer where `codex_prompts/` and `notes/` live
+* thin slices before broad automation
+* review between iterations
+* durable local history
+* failure evidence that supports refinement instead of blind retry
 
-### 2. Directory validation
+So V1 should solve the smallest workflow problem that makes the sequence inspectable:
 
-* verify the expected folders exist
-* fail cleanly if they do not
+* one run
+* one durable execution record
+* one manual review gate
+* one conservative rule for whether the next prompt may proceed
 
-### 3. Prompt discovery
+---
 
-* scan `codex_prompts/`
-* identify valid prompt files
-* ignore files that do not match the naming convention
+## Why We Are Re-Architecting
 
-### 4. Prompt parsing and indexing
+The first draft bundled too many responsibilities into one script. That made it harder to:
 
-* extract numeric prefixes from filenames
-* sort prompts in execution order
-* establish a stable prompt identity from filename/stem
+* understand
+* trust
+* test
+* evolve
+* reuse across repositories
 
-### 5. Note discovery
+We want a cleaner structure, but V1 should only separate what is necessary to support the governing workflow.
 
-* scan `notes/`
-* find note files that match the naming convention
-* ignore unrelated markdown files
+This is still a design-alignment step, not a broad implementation push.
 
-### 6. Note parsing
+---
 
-* extract prompt name, success/fail state, and timestamp from note filenames
-* turn note filenames into structured metadata
+## V1 Scope
 
-### 7. Status reconstruction
+V1 covers the minimum workflow slice from prompt execution through later human review.
 
-* map prompts to matching notes
-* determine whether each prompt is:
+It includes:
 
-  * UNRUN
-  * SUCCESS
-  * FAIL
-* pick the latest note when there are multiple notes for one prompt
+* selecting and executing one prompt
+* writing one durable execution record in `notes/`
+* keeping execution outcome separate from review outcome
+* stopping the queue at `UNREVIEWED`
+* allowing only an explicit accepted review outcome to release the next prompt
 
-### 8. Prompt selection
+V1 is intentionally about workflow control and artifact clarity, not about architectural breadth.
 
-* find the next unrun prompt
-* find the first failed prompt
-* find a prompt by numeric prefix, base name, or full filename
+---
 
-### 9. Prompt display
+## Out Of Scope For V1
 
-* print a selected prompt to the terminal
+The following items are explicitly out of scope in this stage:
 
-### 10. Retry context assembly
-
-* find the latest failed note for a prompt
-* display the original prompt plus the latest failed note
-* present instructions for retrying
-
-### 11. Note writing
+* runner refactor into many modules
+* dependency-aware scheduling
+* richer queue state machines
+* automatic approval or review routing
+* aggregated reports and dashboards
+* retry intelligence beyond minimal retry linkage
+* token accounting beyond optional lightweight fields
+* broad CLI redesign
+* reusable multi-repo configuration systems
+* structured sidecars, databases, or platform services
 
-* create timestamped note filenames
-* write notes in markdown
-* optionally include the previous note as context
+These may become useful later, but they should not shape V1 beyond clear deferral notes.
 
-### 12. Manual status marking
+---
 
-* let the user record a success or failure
-* attach summary/details text to the note
+## Minimum Artifact Inventory
 
-### 13. Command-line interface
+The governing workflow expects the minimum required artifacts to be explicit.
 
-* parse subcommands like `list`, `next`, `show`, `mark`, `retry-failed`
-* route commands to the right behaviors
+For V1, the minimum inventory is:
 
-### 14. Terminal presentation
+### 1. Governing workflow
 
-* format output for human readability
-* present status tables and retry blocks
+`tools/Project_Design_Workflow.md`
 
----
+This remains the controlling sequence for:
 
-## What I Would Like This System To Do Eventually
+* bounded execution
+* validation
+* review between iterations
+* deferral discipline
 
-These are ideas worth considering, but they should not all be built at once.
+### 2. Architecture document
 
-### A. Better architecture and reuse
+`tools/Codex_Prompt_Workflow_Architecture.md`
 
-* work as a reusable template across repositories
-* isolate config from code
-* support per-repo conventions without rewriting internals
+This document should explain:
 
-### B. Structured metadata
+* the V1 problem being solved
+* the V1 scope boundary
+* the minimum parts and artifacts
+* the conservative implementation order
+* what remains deferred
 
-* optionally store prompt metadata in frontmatter or sidecar files
-* support fields like:
+### 3. V1 execution record definition
 
-  * title
-  * objective
-  * tags
-  * priority
-  * dependencies
-  * owner
+`tools/codex/V1_Execution_Record_Artifact.md`
 
-### C. Better status model
+This defines the minimum durable source of truth for one run.
 
-Instead of only:
+Its role in the architecture is to make one run inspectable without requiring a larger system.
 
-* UNRUN
-* SUCCESS
-* FAIL
+### 4. V1 review gate definition
 
-We may want:
+`tools/codex/V1_Run_Review_Gate.md`
 
-* READY
-* RUNNING
-* BLOCKED
-* NEEDS_REVIEW
-* PARTIAL
-* SKIPPED
-* ARCHIVED
+This defines the smallest manual stop between one completed run and the next prompt becoming ready.
 
-### D. Retry intelligence
+Its role in the architecture is to keep execution success separate from accepted progress.
 
-* carry forward the previous failed note automatically
-* include multiple prior failures, not just the latest one
-* summarize recurring failure patterns
+### 5. Execution record files in `notes/`
 
-### E. Prompt preparation
+For V1, one run should produce one markdown execution-record file in `notes/`.
 
-* generate a clean “execution bundle” for Codex
-* include:
+That record is the minimum durable run artifact.
 
-  * prompt text
-  * repo context
-  * previous failure note if retrying
-  * explicit success criteria
+No additional required queue file, database, or sidecar should be introduced in V1.
 
-### F. Manual and automated modes
+---
 
-* manual mode for conservative workflows
-* dry-run mode for seeing what would happen
-* later, optional Codex CLI integration
+## Minimum Viable Slice
 
-### G. Report generation
+The minimum viable slice is:
 
-* generate queue reports
-* generate status snapshots
-* generate audit/history reports over prompt activity
+1. execute one prompt
+2. write one execution record with `review_status: UNREVIEWED`
+3. stop for human review
+4. update the same execution record to either `ACCEPTED` or `REJECTED`
+5. treat only `ACCEPTED` as releasing the next prompt in sequence
 
-### H. Better note/content model
+This is the smallest slice that proves the workflow rather than only the runner.
 
-* move from filename-only state to richer note contents
-* enforce a consistent note template
-* optionally add structured machine-readable metadata inside notes
+It is intentionally narrower than:
 
-### I. Dependency and sequencing support
+* a generalized queue engine
+* a full prompt-status system
+* a modular tool suite
+* an automation framework
 
-* allow prompts to depend on other prompts
-* prevent prompts from running before prerequisites are complete
+---
 
-### J. Prompt families or lanes
+## Role Of The Execution Record
 
-* support categories such as:
+The execution record is the central V1 artifact.
 
-  * setup
-  * audit
-  * refactor
-  * notebook generation
-  * documentation
+Its purpose is to preserve, in one inspectable markdown file:
 
-### K. Approval gates
+* what prompt was run
+* what happened during execution
+* what Codex returned
+* what lightweight failure or resource evidence was observed
+* what a human later decided about the run
 
-* require human confirmation before marking success
-* require review before allowing dependent prompts to proceed
+For V1:
 
-### L. Logging and diagnostics
+* the record body is the source of truth
+* the file in `notes/` is the durable local history unit
+* one run should not be split across multiple required files
 
-* maintain an execution log
-* record tool errors separately from task failures
-* distinguish infrastructure failure from prompt failure
+This keeps V1 small while still supporting:
 
-### M. File layout that scales cleanly
+* review between iterations
+* failure analysis
+* lightweight cost awareness
+* stable run identity
 
-* support growth without turning into a junk drawer
+The execution record defined in `tools/codex/V1_Execution_Record_Artifact.md` should therefore be treated as a required architectural element, not an implementation detail.
 
 ---
 
-## Proposed Folder Architecture
+## Role Of The Review Gate
 
-A better first structure would be:
+The review gate is the minimum manual checkpoint after execution record creation.
 
-```text
-tools/
-  codex/
-    README.md
-    architecture.md
-    config.py
-    paths.py
-    prompts.py
-    notes.py
-    status.py
-    retry.py
-    cli.py
-    templates/
-      note_success.md
-      note_fail.md
-```
-
-For an even more conservative start, we can go smaller:
-
-```text
-tools/
-  codex/
-    README.md
-    architecture.md
-    paths.py
-    prompts.py
-    notes.py
-    status.py
-```
+Its purpose is to enforce the governing workflow rule that the next bounded step should not proceed only because a process finished successfully.
 
-Then add a tiny entrypoint later if needed.
-
----
-
-## Recommended Responsibility Split
-
-### `paths.py`
+For V1:
 
-Owns:
+* execution writes a record
+* new records begin as `UNREVIEWED`
+* human review decides `ACCEPTED` or `REJECTED`
+* only `ACCEPTED` allows the next prompt to be treated as ready
 
-* repo-root discovery
-* locating `codex_prompts/`
-* locating `notes/`
-* validating required folders
+This means the review gate is not a separate platform subsystem.
 
-Should not own:
+It is the manual transition of a single execution record from:
 
-* note parsing
-* prompt parsing
-* CLI logic
+* `UNREVIEWED`
 
-### `prompts.py`
+to one of:
 
-Owns:
+* `ACCEPTED`
+* `REJECTED`
 
-* prompt discovery
-* filename validation
-* numeric prefix parsing
-* prompt sorting
-* prompt lookup by selector
+The review gate defined in `tools/codex/V1_Run_Review_Gate.md` is therefore part of the architecture boundary for V1, not an optional later add-on.
 
-Should not own:
+---
 
-* note history
-* note writing
-* terminal display
+## What The First Script Was Doing
 
-### `notes.py`
+Below is a decomposition of the responsibilities that were bundled together.
 
-Owns:
+### 1. Repo path discovery
 
-* note discovery
-* note filename parsing
-* latest-note selection
-* note writing
-* note naming convention
+* figure out where the repository root is
+* infer where `codex_prompts/` and `notes/` live
 
-Should not own:
+### 2. Directory validation
 
-* prompt queue decisions
-* CLI logic
+* verify the expected folders exist
+* fail cleanly if they do not
 
-### `status.py`
+### 3. Prompt discovery
 
-Owns:
+* scan `codex_prompts/`
+* identify valid prompt files
+* ignore files that do not match the naming convention
 
-* combining prompt data and note data
-* reconstructing current state
-* selecting unrun/failed/successful prompts
+### 4. Prompt parsing and indexing
 
-Should not own:
+* extract numeric prefixes from filenames
+* sort prompts in execution order
+* establish a stable prompt identity from filename/stem
 
-* printing or file writing
+### 5. Note discovery
 
-### `retry.py`
+* scan `notes/`
+* find note files that match the naming convention
+* ignore unrelated markdown files
 
-Owns:
+### 6. Note parsing
 
-* collecting retry context
-* pairing original prompt with latest failed note
-* preparing a retry bundle
+* extract prompt name, success/fail state, and timestamp from note filenames
+* turn note filenames into structured metadata
 
-Should not own:
+### 7. Status reconstruction
 
-* scanning the repo broadly if other modules already do that
+* map prompts to matching notes
+* determine whether each prompt is:
+  * UNRUN
+  * SUCCESS
+  * FAIL
+* pick the latest note when there are multiple notes for one prompt
 
-### `cli.py`
+### 8. Prompt selection
 
-Owns:
+* find the next unrun prompt
+* find the first failed prompt
+* find a prompt by numeric prefix, base name, or full filename
 
-* argument parsing
-* wiring commands to lower-level modules
-* user-facing terminal behavior
+### 9. Prompt display
 
-Should not own:
+* print a selected prompt to the terminal
 
-* core business logic beyond orchestration
+### 10. Retry context assembly
 
----
+* find the latest failed note for a prompt
+* display the original prompt plus the latest failed note
+* present instructions for retrying
 
-## Minimal First Build
+### 11. Note writing
 
-If we build carefully, the first implementation should do only these things:
+* create timestamped note filenames
+* write notes in markdown
+* optionally include the previous note as context
 
-### Phase 1
+### 12. Manual status marking
 
-1. discover prompts
-2. discover notes
-3. reconstruct status
-4. print status table
+* let the user record a success or failure
+* attach summary/details text to the note
 
-### Phase 2
+### 13. Command-line interface
 
-5. show a prompt
-6. show next unrun prompt
-7. write a manual success/fail note
+* parse subcommands like `list`, `next`, `show`, `mark`, `retry-failed`
+* route commands to the right behaviors
 
-### Phase 3
+### 14. Terminal presentation
 
-8. assemble retry context from failed note + original prompt
+* format output for human readability
+* present status tables and retry blocks
 
-That is enough to start using the system without committing to automation yet.
+This decomposition is still useful, but V1 should not treat every decomposed responsibility as something that must be built now.
 
 ---
 
-## Naming Conventions
+## V1 Architectural Shape
 
-### Prompt files
+The conservative V1 shape is smaller than a full prompt-workflow toolkit.
 
-```text
-001_smoke_test_pipeline.md
-002_repo_inventory_and_status.md
-```
+It needs only these practical responsibilities:
 
-### Note files
+### 1. Prompt resolution
 
-```text
-001_smoke_test_pipeline__SUCCESS__20260415_094500.md
-002_repo_inventory_and_status__FAIL__20260415_095212.md
-```
+Enough logic to identify the intended prompt file safely and reproducibly.
 
-These are simple, inspectable, and script-friendly.
+### 2. Bounded execution
 
----
+Enough runner behavior to execute one prompt and capture the immediate execution facts.
 
-## Design Principles
+### 3. Execution-record writing
 
-### 1. Plain files first
+Enough note policy to write one execution record in the V1 format defined by `tools/codex/V1_Execution_Record_Artifact.md`.
 
-Use the filesystem as the source of truth before introducing databases or hidden state.
+### 4. Review write-back
 
-### 2. Small parts
+Enough manual workflow support to update that same record with the review facts defined by `tools/codex/V1_Run_Review_Gate.md`.
 
-Each file/module should have one clear job.
+### 5. Conservative queue progression
 
-### 3. Human-readable state
+Enough status logic to preserve one rule:
 
-A human should be able to inspect prompts and notes without special tooling.
+* only an accepted reviewed run releases the next prompt
 
-### 4. Conservative defaults
+Everything beyond those responsibilities should be presumed deferred unless it is required to preserve the V1 slice.
 
-No automatic execution unless explicitly enabled later.
+---
 
-### 5. Reusability
+## Implementation Order
 
-This should work as a template for future repositories.
+The implementation order should follow the governing workflow and the queued design decisions:
 
-### 6. Honest state
+1. define the V1 review gate
+2. align the architecture document to the workflow, the execution record, and the review gate
+3. define the smallest bridge-runner change spec needed to emit the V1 execution record
+4. only then implement the bounded runner changes for that slice
+5. validate the slice and inspect the resulting run record before any broader refactor
 
-Do not pretend a task succeeded merely because a command ran.
+This order is conservative on purpose.
 
-### 7. Grow only when pressure demands it
+It reduces the risk of:
 
-No extra cleverness until we truly need it.
+* hard-coding the wrong note model
+* coupling queue progression to process exit
+* building convenience layers before the review loop is stable
+* refactoring modules before the actual V1 artifact boundary is settled
 
 ---
 
-## Questions / Design Decisions To Review
+## Validation And Review Posture
 
-### A. What should count as identity?
-
-Should prompt identity be based only on filename stem, or should we eventually support IDs inside prompt files?
-
-### B. How rich should notes be?
+Validation in V1 should stay small and inspectable.
 
-Do we want plain markdown only, or markdown plus structured frontmatter?
+The minimum posture is:
 
-### C. What statuses do we really want in V1?
+* confirm one prompt can be executed intentionally
+* confirm one execution record is written in the expected V1 shape
+* confirm execution status and review status remain separate
+* confirm a new record stops at `UNREVIEWED`
+* confirm only `ACCEPTED` would release the next prompt
+* confirm `REJECTED` stops progression and preserves enough evidence for the next human decision
 
-Is `UNRUN / SUCCESS / FAIL` enough at first?
+This is not a deep automation plan.
 
-### D. How should retries work?
+It is the minimum validation needed to show that the workflow sequence is being honored:
 
-Should a retry include only the latest failed note, or should it include a short history?
+* execute
+* preserve evidence
+* review
+* decide whether to continue
 
-### E. Should prompts ever depend on one another?
+---
 
-If yes, that affects architecture early.
+## Explicit Deferred Items
 
-### F. When do we introduce Codex execution?
+The following remain intentionally deferred until after the V1 slice is proven:
 
-Do we want:
+### Better architecture and reuse
 
-* never in core tools
-* later as an adapter
-* later as an optional plugin layer
+* per-repo configuration layers
+* reusable templates across many repositories
 
-### G. What belongs in notes versus separate logs?
+### Richer metadata and content models
 
-Human-facing narrative and machine-facing diagnostics may eventually deserve separate homes.
+* prompt frontmatter
+* sidecar metadata
+* machine-readable structured note bodies beyond the V1 markdown record
 
----
+### Richer status and queue models
 
-## Recommended Next Step
+* `READY`
+* `RUNNING`
+* `BLOCKED`
+* `PARTIAL`
+* `SKIPPED`
+* `ARCHIVED`
+* dependency-aware release rules
 
-Before writing new code, review this architecture and decide:
+### Retry intelligence
 
-1. the minimal V1 feature set
-2. the final folder layout for `tools/codex/`
-3. the exact status model for V1
-4. whether notes stay plain markdown only
-5. whether retries should include one failed note or a short chain
+* multi-run retry synthesis
+* recurring failure summaries
+* automatic retry context assembly beyond minimal linkage
 
-After that, implement only the smallest slice needed to make the workflow real.
+### Broader execution preparation
 
----
+* richer execution bundles
+* automated repo-context assembly
+* broader prompt packaging
 
-## Suggested Initial Build Target
+### Reporting and diagnostics
 
-My current recommendation for the smallest clean starting point is:
+* aggregated queue reports
+* dashboards
+* cross-run audit summaries
+* separate diagnostic subsystems
 
-* `tools/codex/paths.py`
-* `tools/codex/prompts.py`
-* `tools/codex/notes.py`
-* `tools/codex/status.py`
-* `tools/codex/README.md`
+### Larger implementation restructuring
 
-No CLI yet unless we decide it is truly needed.
+* broad module decomposition
+* larger CLI surface
+* extensive folder expansion
 
-That would let us validate the architecture before we wrap it in commands.
+These are extension paths, not V1 requirements.
 
 ---
 
-## Closing Thought
+## Extension Path After V1
 
-The right next move is not “build the runner.”
+If V1 proves clean, the next steps should still remain bounded.
 
-The right next move is:
+A sensible post-V1 path would be:
 
-* define the pieces
-* define the responsibilities
-* define the boundaries
-* build the smallest useful slice
-
-That gives us something we can trust, reuse, and extend without regret.
+1. keep the execution record as the stable source artifact
+2. keep the review gate explicit
+3. add only the smallest runner changes needed to support that record reliably
+4. review whether any module split is justified by actual friction
+5. defer broader queue features until the single-run slice has been used enough to reveal real needs
 
+This keeps the architecture reusable and inspectable without turning the design document into a platform roadmap.

tokens used
41,584
```
