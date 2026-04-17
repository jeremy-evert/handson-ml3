# Prompt: Define the minimal wrapper scope, state, and resume contract

You are working inside this repository.

Your task is to define the smallest practical concrete contract for the staged notebook wrapper's:

- notebook scope selection
- wrapper-local progress state
- stop/resume mechanics

This is a design pass only.

Do NOT implement the wrapper.
Do NOT modify any `.py` files.
Do NOT modify any notebooks.
Do NOT redesign `tools/codex/run_prompt.py`.
Do NOT change the V1 execution record format.
Do NOT define orchestration-loop behavior in this pass.
Do NOT define stage-specific mutation rules in this pass.
Do NOT turn the wrapper into a queue engine.

## Goal

Write one repo-specific design note that defines the minimum concrete contract needed for a Stage 1 wrapper to:

- select a bounded notebook scope
- remember enough local progress to stop and resume safely
- know which notebook is next within that bounded scope
- preserve `notes/` V1 execution records as the canonical execution and review truth

This prompt exists because the MVP boundary has already been set. The next unresolved design surface is not implementation details or stage logic. It is the minimum concrete contract for wrapper-local scope and resume behavior that can coexist cleanly with the existing V1 record and review model.

## Files to inspect first

Inspect at minimum:

- `notes/023_tools_codex_assessment__20260417_171721.md`
- `notes/024_generate_staged_notebook_wrapper_prompts__20260417_172452.md`
- `notes/025_first_prompt_audit__20260417_125331.md`
- `notes/025_define_staged_notebook_wrapper_mvp_contract__20260417_175751.md`
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
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/Project_Design_Workflow.md`

## What this design note must decide

Your note must answer these questions directly and concretely.

### 1. What is one wrapper run allowed to scope?

Define the smallest practical scope contract for the MVP slice.

At minimum, decide:

- whether scope is one notebook, an ordered notebook list, or some similarly narrow batch
- whether the contract is Stage 1 only for now
- what the wrapper must know at start time versus what it may derive
- how scope stays narrow enough for review and resume

Do not define a broad multi-stage or repo-wide orchestration model.

### 2. What minimal wrapper-local state is allowed?

Define the minimum wrapper-local state needed to support stop/resume without competing with `notes/`.

At minimum, address:

- active stage identity
- bounded notebook scope identity
- current notebook position or equivalent next-target marker
- whether scan-derived treatment status may be cached locally or must be recomputed
- what wrapper-local fields are allowed versus disallowed

Be explicit about what must remain canonical in V1 execution records instead.

### 3. How does the wrapper know what notebook is next?

Define the minimum rule for "next notebook" selection inside one bounded wrapper run.

At minimum, address:

- deterministic notebook ordering
- how already-reviewed work affects next-target choice
- how a wrapper distinguishes "not yet run", "run but awaiting review", and "accepted enough to advance" without creating a second execution-truth system

This must remain compatible with the current `UNREVIEWED` / `ACCEPTED` / `REJECTED` review gate.

### 4. How should stop/resume work?

Define the smallest practical stop/resume contract.

At minimum, answer:

- what event causes the wrapper to stop
- what information must exist to allow resume
- what information may be reconstructed from `notes/`
- when the wrapper is allowed to advance after resume
- how resume behavior handles a latest run that is still `UNREVIEWED`
- how resume behavior handles a latest run that is `REJECTED`

The answer must preserve the current V1 stop-and-decide model.

### 5. What is wrapper-local versus canonical?

State clearly which facts may live in wrapper-local state and which facts must stay canonical in `notes/`.

At minimum, distinguish:

- wrapper convenience state
- derived targeting state
- canonical execution facts
- canonical review facts
- readiness/progression truth

The note must make clear that wrapper-local state may support resume, but may not redefine whether a bounded run happened or whether it is accepted.

### 6. What should this prompt explicitly defer?

State clearly what later prompts must define instead of this one.

At minimum, defer:

- exact orchestration-loop behavior
- prompt-generation rules for Stage 1
- any Stage 2 or Stage 3 design
- notebook write contracts
- retry policy beyond current V1 review outcomes
- reporting or dashboard artifacts beyond the minimum needed for scope/resume support
- implementation choices in Python

## Deliverable

Write exactly one markdown note into `notes/`.

Use this filename if possible:

`notes/028_minimal_wrapper_scope_state_and_resume_contract__<TIMESTAMP>.md`

## Required note structure

# Minimal Wrapper Scope, State, and Resume Contract

## Executive Summary
- one short paragraph stating the contract direction

## Why This Design Surface Comes Next

## Proposed Scope Contract

## Minimum Wrapper-Local State

## Next-Notebook Selection Rule

## Stop/Resume Contract

## Wrapper-Local Versus Canonical Truth

## Guardrails Against Queue-Engine Drift

## Explicit Deferrals

## Recommended Follow-On Prompt
- name the next prompt that should define the next concrete surface after this one

## Output rules

- Be concrete
- Be repo-specific
- Prefer the smallest workable contract over speculative flexibility
- Keep the wrapper runner-centered
- Preserve the V1 review gate exactly as it exists
- Do not write code
- Do not modify repo files except for the single note in `notes/`
- At the end of your final response, print only the path to the note you created
