# Task: Define the V1 run-review gate for prompt execution

You are working in this repository.

Your task is to define the minimum review gate that must sit between:

- a prompt run finishing
- the next prompt in sequence being treated as ready

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/baby_run_prompt.py`
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
- `notes/005_prompt_queue_plan__20260415_202557.md`

## Goal

Use `tools/Project_Design_Workflow.md` as governing.

Define the smallest practical V1 review gate that preserves:

- review between iterations
- separation of execution outcome from accepted outcome
- a conservative manual step before the next prompt is treated as ready

## Important framing

This is a design task.

Do NOT implement the gate in code in this pass.
Do NOT create a large state machine.
Do NOT redesign the whole queue system.

## Questions to settle

Please settle these points:

1. What exact event marks a run as awaiting review?
2. What minimum human checks must happen before a run can be accepted?
3. What review outcomes are needed in V1?
4. What outcome should let the next prompt proceed?
5. What outcome should stop the queue and force a new design or retry decision?
6. What information must be written back into the run record during review?

## Required output artifact

Create one markdown design note at:

`tools/codex/V1_Run_Review_Gate.md`

The note should include:

- purpose
- scope
- the review trigger
- the minimum manual checklist
- allowed V1 review outcomes
- how the outcome affects queue progression
- how this connects to `tools/codex/V1_Execution_Record_Artifact.md`
- what is intentionally deferred

## Constraints

1. Use the exact file paths listed above.
2. Keep the gate manual and conservative.
3. Do not implement CLI or runner changes in this pass.
4. Do not invent a large workflow beyond the next reviewed step.
5. Keep the decision small enough to review before any implementation prompt follows.

## Success criteria

This task is successful if:

- the review gate is explicit and easy to apply
- execution success is clearly separated from accepted outcome
- queue progression rules are clear for V1
- the result is small enough to guide the next prompt without expanding scope
