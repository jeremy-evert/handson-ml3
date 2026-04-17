# Prompt: Define the minimal Stage 1 prompt-generation contract for chapter-intro treatment

You are working inside this repository.

Your task is to define the smallest practical concrete contract for **Stage 1 prompt generation** only.

This is a design-contract pass only.

Do NOT implement the wrapper.
Do NOT modify any `.py` files.
Do NOT modify any notebooks.
Do NOT redesign `tools/codex/run_prompt.py`.
Do NOT change the V1 execution record format.
Do NOT define Stage 2 or Stage 3 behavior.
Do NOT define the orchestration loop.
Do NOT define Python implementation details in this pass.

## Goal

Create one repo-specific markdown note that defines the minimal contract for how Stage 1 prompts are generated for chapter-intro treatment work.

This prompt exists to settle the Stage 1 prompt-generation surface before any build prompt tries to implement prompt generation or wrapper orchestration.

The note must make later work safer by clearly stating:

- what inputs Stage 1 prompt generation requires
- what the scanner must already determine before a Stage 1 prompt may be generated
- what one Stage 1 prompt is allowed to target
- what intro insertion vs replacement decisions are allowed
- what output contract every Stage 1 prompt must enforce
- what notebook-safety rules are non-negotiable
- what remains wrapper-local versus what stays canonical in `notes/`

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

## What this contract note must decide

Your note must answer these questions directly.

### 1. What inputs are required to generate one Stage 1 prompt?

Define the minimum concrete inputs needed before prompt generation may happen.

At minimum, address:

- the targeted notebook path
- the fixed active stage identity
- the scanner-derived chapter-intro status
- the scanner-derived existing intro cell index, if any
- the scanner-derived setup-cell index, if any
- any notebook metadata that is necessary to keep the prompt bounded

Do not drift into CLI syntax or Python types. Define the contract surface, not the implementation.

### 2. What must the scanner already determine?

State what Stage 1 prompt generation is allowed to assume has already been determined by scanner-first logic.

At minimum, address:

- whether Stage 1 is needed at all
- whether an existing intro is `missing`, `heading`, `thin`, or `substantive`
- whether the intro action is `insert` or `replace`
- what exact notebook position is eligible for the change
- what evidence is sufficient to avoid generating a Stage 1 prompt

Make clear that prompt generation must not be the place where open-ended notebook discovery happens.

### 3. What is one Stage 1 prompt allowed to target?

Define the smallest practical unit of Stage 1 prompt work.

At minimum, address:

- one prompt targets exactly one notebook
- one prompt covers exactly one Stage 1 chapter-intro treatment decision
- one prompt must not become a broad notebook rewrite
- one prompt must not include Stage 2 or Stage 3 work

### 4. What intro decisions are allowed?

Define the allowed Stage 1 treatment choices.

At minimum, address:

- when insertion is allowed
- when replacement is allowed
- when skip / no-prompt is required
- what kinds of opportunistic edits are forbidden

Be explicit that Stage 1 must not authorize cleanup beyond the chapter-intro decision.

### 5. What output contract must every Stage 1 prompt enforce?

Define the prompt-side output contract that a future implementation must preserve.

At minimum, address:

- the prompt must direct one bounded notebook mutation only
- the prompt must require writing the complete modified notebook artifact, not partial fragments
- the prompt must preserve notebook validity
- the prompt must prohibit direct code-cell edits
- the prompt must prohibit unrelated cell reordering, broad rewrites, and metadata churn
- the prompt must be reviewable as one bounded V1 step

You may refer to temporary notebook-output behavior at a contract level if needed, but do not define implementation code.

### 6. What notebook-safety rules are non-negotiable?

State the hard safety boundaries for Stage 1 prompt generation.

At minimum, include:

- no direct code-cell edits
- no broad notebook rewrites
- no Stage 2 or Stage 3 edits
- no opportunistic style cleanup
- deterministic targeting before mutation
- bounded insert-or-replace action only
- no silent progression past the V1 review gate

### 7. What remains local versus canonical?

Clarify what Stage 1 prompt generation may treat as local convenience versus what must remain canonical in `notes/`.

At minimum, address:

- wrapper-local prompt-generation inputs or targeting facts may exist
- execution and review truth remain canonical in V1 records under `notes/`
- prompt-generation state must not redefine queue readiness
- Stage 1 prompt generation must stay subordinate to the runner-centered workflow

## Required boundaries

Your note must preserve these boundaries:

- The design remains runner-centered around `tools/codex/run_prompt.py`.
- The V1 review gate remains unchanged.
- `notes/` remains the canonical execution and review history.
- This pass defines only the Stage 1 prompt-generation contract.
- This pass does not define orchestration.
- This pass does not define Stage 2 or Stage 3.
- This pass does not define Python implementation details.

## Deliverable

Write exactly one markdown note into `notes/`.

Use this filename if possible:

`notes/030_define_stage1_prompt_generation_contract__<TIMESTAMP>.md`

## Required note structure

# Stage 1 Prompt-Generation Contract

## Executive Summary
- one short paragraph stating the narrow contract surface

## Why This Design Surface Comes Next

## Required Inputs For One Stage 1 Prompt

## Scanner Preconditions

## Allowed Stage 1 Target Unit

## Allowed Intro Decisions

## Stage 1 Prompt Output Contract

## Mandatory Notebook-Safety Rules

## Local Prompt-Generation Facts Versus Canonical Truth

## Explicit Deferrals
- clearly defer Stage 2, Stage 3, orchestration, and implementation work

## Recommended Follow-On Prompt
- name the next design or build prompt that should follow once this contract is settled

## Output rules

- Be concrete
- Be repo-specific
- Keep the contract narrow and deterministic
- Prefer explicit boundaries over speculative flexibility
- Do not write code
- Do not modify repo files except for the single note in `notes/`
- At the end of your final response, print only the path to the note you created
