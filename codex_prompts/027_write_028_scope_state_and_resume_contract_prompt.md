# Prompt: Write the next wrapper-build prompt for minimal scope, state, and resume design

You are working inside this repository.

Your task is to write exactly one new prompt file in `codex_prompts/`.

Do NOT implement the wrapper.
Do NOT modify any `.py` files.
Do NOT modify any notebooks.
Do NOT redesign `tools/codex/run_prompt.py`.
Do NOT change the V1 execution record format.

## Goal

Use the existing wrapper assessment, prompt-plan evidence, audit result, and MVP contract note to write the next actual build prompt in the sequence.

The prompt file you must create is:

`codex_prompts/028_define_minimal_wrapper_scope_state_and_resume_contract.md`

This new prompt should be a concrete design prompt, not an implementation prompt.

Its purpose is to define the smallest practical concrete contract for:

- notebook scope selection
- wrapper-local progress state
- stop/resume mechanics

without competing with the V1 execution record in `notes/`.

## Files to inspect first

Inspect at minimum:

- `notes/023_tools_codex_assessment__20260417_171721.md`
- `notes/024_generate_staged_notebook_wrapper_prompts__20260417_172452.md`
- `notes/025_first_prompt_audit__20260417_125331.md`
- `notes/026_define_staged_notebook_wrapper_mvp_contract__20260417_175702.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `tools/codex/v1_record_validation.py`
- `tools/notebook_enricher/notebook_scanner.py`
- `tools/notebook_enricher/prompt_builder.py`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Execution_Record_Artifact.md`

## What the new 028 prompt must do

The prompt you write must instruct a future Codex run to define the minimal concrete contract for wrapper scope/state/resume behavior.

That future 028 prompt should answer questions like:

- how notebooks are selected for one wrapper run
- how the wrapper knows which stage is active
- how the wrapper knows which notebook is next
- what minimal wrapper-local state is allowed
- how stop/resume works without replacing the V1 execution record
- what is allowed to be wrapper-local versus what remains canonical in `notes/`

## Constraints for the new 028 prompt

The prompt you write must ensure that the future 028 design prompt:

- stays runner-centered
- preserves the V1 review gate
- avoids becoming a queue engine
- avoids redefining execution truth outside `notes/`
- keeps the wrapper-local state minimal
- does not define orchestration-loop behavior yet
- does not define stage-specific mutation rules yet
- does not implement code

## Deliverables

Create exactly:

1. `codex_prompts/028_define_minimal_wrapper_scope_state_and_resume_contract.md`
2. one short note in `notes/` explaining:
   - why this is the right next prompt
   - what design surface it covers
   - what it still intentionally defers

## Output rules

- Write the actual prompt file, not commentary about a prompt
- Make the new prompt concrete and runnable
- Keep it narrow
- At the end of your final response, print only the path to the note you created
