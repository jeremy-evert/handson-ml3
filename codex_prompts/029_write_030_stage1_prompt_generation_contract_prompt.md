# Prompt: Write the next wrapper-build prompt for the Stage 1 prompt-generation contract

You are working inside this repository.

Your task is to write exactly one new prompt file in `codex_prompts/`.

Do NOT implement the wrapper.
Do NOT modify any `.py` files.
Do NOT modify any notebooks.
Do NOT redesign `tools/codex/run_prompt.py`.
Do NOT change the V1 execution record format.

## Goal

Use the existing wrapper assessment, prompt-plan evidence, first-prompt audit, MVP contract note, and minimal scope/state/resume contract note to write the next actual build prompt in the sequence.

The prompt file you must create is:

`codex_prompts/030_define_stage1_prompt_generation_contract.md`

This new prompt must be a concrete design prompt, not an implementation prompt.

Its purpose is to define the smallest practical contract for **Stage 1 prompt generation** only, meaning the prompt-generation rules and safety boundaries for chapter-intro treatment work.

It must not define Stage 2 or Stage 3.
It must not implement code.
It must not define the orchestration loop.

## Files to inspect first

Inspect at minimum:

- `notes/023_tools_codex_assessment__20260417_171721.md`
- `notes/024_generate_staged_notebook_wrapper_prompts__20260417_172452.md`
- `notes/025_first_prompt_audit__20260417_125331.md`
- `notes/025_define_staged_notebook_wrapper_mvp_contract__20260417_175751.md`
- `notes/026_define_staged_notebook_wrapper_mvp_contract__20260417_175702.md`
- `notes/028_minimal_wrapper_scope_state_and_resume_contract__20260417_190146.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `tools/codex/v1_record_validation.py`
- `tools/notebook_enricher/notebook_scanner.py`
- `tools/notebook_enricher/prompt_builder.py`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/Project_Design_Workflow.md`

## What the new 030 prompt must do

The prompt you write must instruct a future Codex run to define the minimal concrete contract for **Stage 1 prompt generation**.

That future 030 prompt should answer questions like:

- what inputs Stage 1 prompt generation requires
- what the scanner must already determine before a Stage 1 prompt is generated
- what one Stage 1 prompt is allowed to target
- what chapter-intro insertion vs replacement decisions are allowed
- what output contract a Stage 1 prompt must enforce
- what notebook-safety rules must be non-negotiable
- what Stage 1 prompt-generation concerns remain local and what must stay canonical in `notes/`

## Constraints for the new 030 prompt

The prompt you write must ensure that the future 030 design prompt:

- stays runner-centered
- preserves the V1 review gate
- preserves `notes/` as the canonical execution/review truth
- does not become an orchestration prompt
- does not define Python implementation details
- does not define Stage 2 or Stage 3 behavior
- does not permit direct code-cell edits
- does not permit broad notebook rewrites
- keeps Stage 1 prompt generation deterministic and bounded

## Deliverables

Create exactly:

1. `codex_prompts/030_define_stage1_prompt_generation_contract.md`
2. one short note in `notes/` explaining:
   - why this is the right next prompt
   - what design surface it covers
   - what it intentionally defers

## Required qualities of the new 030 prompt

The prompt you create must:

- be concrete
- be repo-specific
- be narrow
- define a design surface, not an implementation surface
- explicitly separate Stage 1 contract decisions from later implementation work
- clearly defer Stage 2, Stage 3, orchestration, and code generation

## Output rules

- Write the actual prompt file, not commentary about a prompt
- Make the new prompt concrete and runnable
- Keep it narrow and disciplined
- At the end of your final response, print only the path to the note you created
