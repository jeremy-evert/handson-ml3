# Staged Notebook Wrapper MVP Contract

## Executive Summary
The staged notebook wrapper exists to break notebook-treatment work into bounded Codex-sized runs without weakening the current V1 runner, record, and review workflow. The MVP thin slice is a Stage 1 only wrapper that can select a narrow notebook scope, determine whether each notebook needs a chapter-intro treatment, and launch one bounded intro-treatment run at a time through `tools/codex/run_prompt.py`, stopping at the existing review gate after each run.

## Problem This Wrapper Solves

The current repo has two useful but disconnected pieces:

- a conservative runner-centered workflow built around `tools/codex/run_prompt.py`, `notes/` V1 execution records, `tools/codex/review_run.py`, `tools/codex/check_queue_readiness.py`, and `tools/codex/list_review_backlog.py`
- notebook-treatment experiments in `tools/notebook_enricher/notebook_scanner.py` and `tools/notebook_enricher/prompt_builder.py` that show how narrow notebook mutations can be targeted

What is missing is a thin repo-specific bridge that can turn notebook-treatment intent into repeated bounded runs without collapsing back into one giant notebook-enrichment prompt or inventing a second workflow platform.

In this repo, the wrapper exists to solve a narrow coordination problem:

- select a bounded notebook-treatment target
- use scanner-first evidence to decide whether treatment is needed
- emit one narrow prompt for one bounded notebook-treatment step
- run that prompt through the existing runner
- stop for human review before any further progression

The wrapper does not exist to replace the runner, redesign the queue, or create a broader notebook-processing system.

## MVP Thin Slice

The MVP is allowed to do exactly one meaningful thin slice first:

- Stage 1 only
- chapter-intro detection and treatment only
- one bounded notebook-treatment run at a time
- runner-mediated execution only

For the first MVP pass, the wrapper may:

- accept a narrow notebook scope using a later-defined minimal input contract
- inspect candidate notebooks with scanner-first logic before mutation
- determine whether a notebook is missing or has a non-substantive chapter intro
- generate a bounded Stage 1 prompt for exactly one notebook at a time
- launch that prompt through `tools/codex/run_prompt.py`
- stop after each run and require the normal V1 review decision before any subsequent wrapper-driven run

The MVP is not allowed to start with all three stages, all notebooks in one pass, or an autonomous multi-step workflow. The first slice is deliberately a Stage 1 launcher around existing V1 mechanics.

## Required Invariants

The wrapper must preserve these repo-specific invariants:

- The workflow remains runner-centered. The wrapper is subordinate to `tools/codex/run_prompt.py`, not an alternative execution path.
- Bounded work still goes through `tools/codex/run_prompt.py`. The wrapper must not bypass it with direct `codex exec` calls for normal bounded notebook-treatment runs.
- The V1 execution record in `notes/` remains the canonical artifact for each bounded run. One wrapper-launched run still means one V1 execution record markdown file.
- The wrapper must not change the V1 execution record format defined by `tools/codex/V1_Execution_Record_Artifact.md` and validated by `tools/codex/v1_record_validation.py`.
- The existing review gate remains in force. `UNREVIEWED`, `ACCEPTED`, and `REJECTED` keep their current meaning, and only `ACCEPTED` may release the next bounded step.
- The wrapper must not become a second source of truth for queue progression. `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py` already derive progression and backlog status from the V1 records in `notes/`; the wrapper must not supersede that with wrapper-local queue state.
- Review write-back remains manual through the existing model embodied by `tools/codex/review_run.py`.
- Any wrapper-local artifacts introduced later may support targeting or resume behavior, but they must not redefine whether a run happened, whether it was reviewed, or whether the next prompt is ready.

## Mandatory Safety Rules

The wrapper MVP must enforce these safety rules:

- No direct code-cell edits.
- No broad notebook rewrites framed as enrichment.
- Only deterministic, bounded insert-or-replace notebook mutations are allowed.
- Scanner-first targeting is mandatory before any mutation prompt is generated.
- Mutation targeting must be derived from concrete notebook positions, not vague chapter-wide instructions.
- The wrapper may only launch prompts that are narrow enough for a reviewer to verify against one notebook and one treatment goal.
- No silent automatic progression past review boundaries. A completed run must stop at `UNREVIEWED` until a human applies the existing review gate.
- No mutation prompt may authorize opportunistic cleanup, style passes, notebook reformatting, output regeneration, metadata churn, or unrelated edits.
- For the Stage 1 MVP slice, allowed mutations are limited to a single chapter-intro insert or replace action for the targeted notebook.

These safety rules follow directly from the notebook scanner and prompt-builder experiments in `tools/notebook_enricher/`, which already assume cell-adjacent, deterministic treatment decisions rather than open-ended notebook rewriting.

## Explicit Deferrals

This contract pass intentionally does not decide the following. Later prompts must define them:

- the concrete notebook scope or selection config format
- the concrete progress-state structure for wrapper-local resume support
- the stop/resume file schema
- the detailed scan or report artifact shape
- the stage-specific prompt-generation rules and prompt text contracts
- the detailed orchestration-loop behavior
- the exact wrapper CLI surface
- the concrete write path for wrapper-generated prompt files or wrapper-local notes
- Stage 2 rules for markdown-before-code treatment
- Stage 3 rules for markdown-after-code treatment
- retry behavior beyond the existing V1 rejected-run stop-and-decide model

This note sets boundaries and deferrals only. It does not settle the concrete design of those deferred surfaces.

## Non-Goals

The staged notebook wrapper MVP is not:

- a redesign of `tools/codex/run_prompt.py`
- a redesign of the V1 execution record
- a new queue engine
- a richer queue-state model
- retry orchestration
- a dashboard or reporting UI
- a background daemon or watcher
- a parallel execution system
- an automatic reviewer or approval system
- a general notebook-refactoring platform
- a repo-wide enrichment pass that treats all stages as one unit

If a proposed design requires any of the above to make the MVP feel complete, that design is too broad for this pass.

## Why This Contract Comes Before Concrete Design

This repo already favors boundaries before breadth and thin slices before larger automation, as stated in `tools/Project_Design_Workflow.md` and `tools/Codex_Prompt_Workflow_Architecture.md`. The current V1 workflow is deliberately small: one run, one record, one review gate, one conservative readiness rule. A staged notebook wrapper can fit into that workflow cleanly only if the boundary is settled before implementation details start expanding.

This contract comes first to prevent three common failure modes:

- treating the wrapper as a replacement execution path instead of a thin layer around `tools/codex/run_prompt.py`
- letting wrapper-local progress artifacts become a second source of truth for queue progression
- overdefining stop/resume, scan artifacts, and orchestration behavior before the MVP thin slice is proven

By fixing the contract first, later prompts can define the minimum concrete scope/progress/resume surface without accidentally redesigning the existing V1 workflow.

## Recommended Next Prompt

The next prompt should be:

- `Build the minimal Stage 1 wrapper scope/progress/resume contract`

That prompt should define only the smallest concrete contract for:

- notebook scope input
- wrapper-local progress tracking sufficient for stop/resume
- the minimum resume boundary that does not compete with V1 execution records or review status

It should not yet define full orchestration behavior, Stage 2 or Stage 3 treatment rules, or any broader wrapper platform concerns.
