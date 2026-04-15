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
