# Task: Review the new V1 runner candidate and recommend the smallest next step

You are working in this repository.

Your task is to review the newly built V1 runner candidate, inspect the sample execution record it produced, and recommend the smallest sensible next move.

## Important framing

This is a review-and-recommendation task.

Do NOT implement code changes in this pass.
Do NOT rewrite major documents.
Do NOT refactor the runner.
Do NOT modify `tools/codex/baby_run_prompt.py`.

Your job is to inspect what now exists and decide the smallest next improvement or validation step.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/run_prompt.py`
- `notes/009_run_prompt_candidate_build__20260415_233407.md`
- `notes/001_smoke_test_pipeline__20260415_233343.md`

You may also read, for comparison only:

- `tools/codex/baby_run_prompt.py`
- `notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md`
- `notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md`

## Goal

Determine whether the next bounded step should be one of these:

1. a tiny polish to `tools/codex/run_prompt.py`
2. a small helper or workflow aid for manual review write-back
3. a focused environment-diagnosis prompt for the read-only session issue seen in the validation run

You must recommend only **one** next step.

## Questions to answer

Please answer these questions in your review:

### 1. Runner assessment
- Does `tools/codex/run_prompt.py` successfully implement the V1 design intent?
- What is strong about it?
- Are there any small correctness or clarity issues that should be fixed before further build-out?

### 2. Artifact assessment
- Does `notes/001_smoke_test_pipeline__20260415_233343.md` match the intended V1 execution-record shape?
- Is the separation between execution and review clear?
- Did the record preserve useful failure evidence?

### 3. Operational assessment
- Is the most important next issue now design-related, implementation-related, or environment-related?
- Does the read-only filesystem/session error appear to be mainly a runner problem or a Codex execution environment problem?

### 4. Smallest next move
Choose exactly one of the following as the recommended next move:

- tiny runner polish
- review write-back helper/workflow aid
- environment diagnosis prompt

Explain why that is the smallest and safest next move.

## Required output artifacts

Create exactly two artifacts.

### Artifact 1
Create a review note at:

`notes/010_run_prompt_candidate_review__TIMESTAMP.md`

This note should include:

- short summary
- runner assessment
- artifact assessment
- operational assessment
- exactly one recommended next move

### Artifact 2
Create a short recommendation note at:

`notes/010_next_step_recommendation__TIMESTAMP.md`

This should contain only:

- the chosen next move
- why it should happen next
- what it should produce
- what should explicitly wait

Keep it brief and decisive.

## Constraints

1. Use the exact file paths listed above.
2. Recommend only one next step.
3. Do not produce a broad roadmap.
4. Do not modify code or design documents in this pass.
5. Keep the recommendation small enough to become the next Codex prompt.

## Success criteria

This task is successful if:

- the review clearly assesses both `run_prompt.py` and the sample execution record
- the next move is grounded in what actually happened
- only one next step is recommended
- the result helps us continue in a slow, deliberate, low-risk way
