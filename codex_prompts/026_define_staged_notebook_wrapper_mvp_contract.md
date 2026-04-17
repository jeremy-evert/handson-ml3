# Prompt: Define the staged notebook-wrapper MVP contract

You are working inside this repository.

Your task is to define the MVP contract for the staged notebook-treatment wrapper.

This is a contract-and-boundary pass only.

Do NOT implement the wrapper.
Do NOT modify any `.py` files.
Do NOT modify any notebooks.
Do NOT redesign `tools/codex/run_prompt.py`.
Do NOT change the V1 execution record format.
Do NOT create a new queue engine.
Do NOT define detailed stop/resume file schemas in this pass.
Do NOT define detailed notebook-selection config syntax in this pass.
Do NOT define detailed orchestration-loop behavior in this pass.

## Goal

Create one repo-specific markdown note that defines the narrow MVP contract for the staged notebook-wrapper effort.

This prompt exists to settle the boundary and deferrals before any concrete wrapper design or implementation work begins.

The note must make later prompts safer by clearly stating:

- why the wrapper exists
- what the MVP is allowed to do first
- what invariants it must preserve from the current V1 workflow
- what safety rules are mandatory
- what is explicitly deferred to later prompts

## Context

The current repo already has:

- a runner-centered workflow built around `tools/codex/run_prompt.py`
- one-run / one-record V1 execution records in `notes/`
- manual review write-back via `tools/codex/review_run.py`
- readiness checks via `tools/codex/check_queue_readiness.py`
- backlog inspection via `tools/codex/list_review_backlog.py`
- notebook scanning and prompt-building experiments in:
  - `tools/notebook_enricher/notebook_scanner.py`
  - `tools/notebook_enricher/prompt_builder.py`

The wrapper under discussion is meant to launch narrow notebook-treatment work in stages, not replace the runner or invent a broader platform.

## Files to inspect first

Inspect at minimum:

- `notes/023_tools_codex_assessment__20260417_171721.md`
- `notes/024_generate_staged_notebook_wrapper_prompts__20260417_172452.md`
- `notes/025_first_prompt_audit__20260417_125331.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `tools/codex/v1_record_validation.py`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/Project_Design_Workflow.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/notebook_enricher/notebook_scanner.py`
- `tools/notebook_enricher/prompt_builder.py`

## What this contract note must decide

Your note must answer these questions directly.

### 1. Why does the wrapper exist?
Explain the exact problem the wrapper is meant to solve in this repo.

### 2. What is the MVP allowed to do first?
Choose the smallest meaningful thin slice.
Prefer one narrow stage-first slice over an all-stages MVP.

### 3. What invariants must the wrapper preserve?
At minimum, address:

- the wrapper remains runner-centered
- bounded work still goes through `tools/codex/run_prompt.py`
- the V1 execution record in `notes/` remains the canonical run artifact
- the existing review gate remains in force
- the wrapper must not become a second source of truth for queue progression

### 4. What safety rules are mandatory?
At minimum, address:

- no direct code-cell edits
- no broad notebook rewrites
- only deterministic, bounded insert-or-replace notebook mutations
- scanner-first targeting before mutation
- no silent automatic progression past review boundaries

### 5. What is explicitly deferred?
State clearly what later prompts must decide instead of this one.

At minimum, defer:

- concrete scope/config input format
- concrete progress-state structure
- stop/resume file schema
- detailed scan/report artifact shape
- stage-specific prompt-generation rules
- orchestration-loop behavior

### 6. What is the non-goal list?
State clearly what this wrapper MVP is not.

At minimum, exclude:

- runner redesign
- V1 record redesign
- a new queue engine
- richer queue states
- retry orchestration
- dashboards
- background daemons
- parallel execution

## Deliverable

Write exactly one markdown note into `notes/`.

Use this filename if possible:

`notes/025_define_staged_notebook_wrapper_mvp_contract__<TIMESTAMP>.md`

## Required note structure

# Staged Notebook Wrapper MVP Contract

## Executive Summary
- one short paragraph stating the narrow MVP direction

## Problem This Wrapper Solves

## MVP Thin Slice
- state what the wrapper is allowed to do first
- keep this deliberately narrow

## Required Invariants

## Mandatory Safety Rules

## Explicit Deferrals

## Non-Goals

## Why This Contract Comes Before Concrete Design

## Recommended Next Prompt
- name the next design prompt that should follow this contract pass
- that next prompt should define the minimal concrete scope/progress/resume contract

## Output rules

- Be concrete
- Be repo-specific
- Prefer strong boundaries over broad ambition
- Do not write code
- Do not modify repo files except for the single note in `notes/`
- At the end of your final response, print only the path to the note you created
