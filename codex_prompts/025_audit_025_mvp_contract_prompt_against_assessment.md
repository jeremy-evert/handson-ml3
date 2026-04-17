# Prompt: Audit the proposed MVP-contract prompt against the wrapper assessment

You are working inside this repository.

Your task is to audit exactly one proposed wrapper-build prompt against the earlier staged notebook-wrapper assessment and the current Codex runner-centered workflow.

This is an audit-only pass.

Do NOT implement the wrapper.
Do NOT modify any `.py` files.
Do NOT modify any notebooks.
Do NOT rewrite prompt files in this pass.

## Goal

The previous planning pass proposed that the first wrapper-build prompt should be:

`codex_prompts/025_define_staged_notebook_wrapper_mvp_contract.md`

Your job is to assess whether that proposed first prompt is the right first move.

You are auditing exactly one prompt idea, not the full future sequence.

## Important constraints

- Audit only the proposed `025_define_staged_notebook_wrapper_mvp_contract.md`
- Do NOT evaluate later prompts except where needed to explain whether this first prompt is correctly scoped
- Do NOT create code
- Do NOT mutate notebooks
- Do NOT redesign `tools/codex/run_prompt.py`
- Do NOT change the V1 execution record format
- This pass is for judgment, not construction

## Files and evidence to inspect

Inspect at minimum:

- the earlier wrapper assessment note
- the latest prompt-generation run note
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `tools/codex/v1_record_validation.py`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/Project_Design_Workflow.md`
- `tools/notebook_enricher/notebook_scanner.py`
- `tools/notebook_enricher/prompt_builder.py`

If `codex_prompts/025_define_staged_notebook_wrapper_mvp_contract.md` does not actually exist yet, treat this as an audit of the proposed prompt concept and the intended role described in the planning evidence.

## Questions to answer

Answer these directly and concretely.

### 1. Is this the right first prompt?
Should the first wrapper-build prompt really be an MVP-contract prompt?

### 2. Is the scope right?
Should this prompt cover:
- MVP boundary
- wrapper state model
- scope/config inputs
- reporting artifacts
- non-goals

Or is that too much for one prompt?

### 3. Is anything missing?
What absolutely must be included in this first prompt so later work does not drift?

### 4. Is anything premature?
What should NOT be included in this first prompt because it belongs in a later prompt?

### 5. Does this align with the assessment?
Does this first prompt preserve:
- runner-centered design
- stop/resume concerns
- notebook mutation safety
- MVP-first discipline
- thin-slice sequencing

### 6. Final judgment
Assign one verdict to the proposed first prompt:
- KEEP
- REVISE
- SPLIT
- DROP

Then explain why.

## Deliverable

Write exactly one markdown note into `notes/`.

Use a filename close to:

`notes/025_first_prompt_audit__YYYYMMDD_HHMMSS.md`

## Required report structure

# First Prompt Audit Report

## Executive Summary
- one short paragraph stating whether the proposed MVP-contract prompt is the right first move

## Evidence Reviewed
- list the notes and repo files inspected

## What This First Prompt Should Cover
## What This First Prompt Should Not Cover
## Alignment With The Wrapper Assessment
## Risks If This Prompt Is Too Broad Or Too Thin
## Verdict
- KEEP / REVISE / SPLIT / DROP
- with explanation

## Recommendation For The Next Step
- explain whether the next action should be:
  - write the actual 025 prompt,
  - revise the idea for 025 first,
  - or replace it with a different first prompt

## Output rules

- Be concrete
- Be repo-specific
- Audit exactly one prompt idea
- Do not rewrite the prompt file in this pass
- At the end of your final response, print only the path to the note you created
