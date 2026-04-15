# Task: Define the V1 execution-record artifact and generate the next bounded prompt queue

You are working in this repository.

Your task has two tightly related parts:

1. define the V1 execution-record artifact for one prompt run
2. use that design decision to generate the next bounded sequence of prompt files needed to reduce the current workflow/architecture/runner misalignment

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/baby_run_prompt.py`
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
- `notes/004_next_design_step_recommendation__20260415_195538.md`

## Goal

Use the current workflow as the governing law.

Then:

### Part 1
Define the V1 execution-record artifact for one prompt run.

This artifact should clarify:

- what the source of truth is for one run
- what is captured automatically
- what remains manual for human review
- how execution outcome is separated from accepted/reviewed outcome
- what minimal failure-analysis fields exist
- what minimal cost/resource fields exist
- what should explicitly wait until later

### Part 2
Using that execution-record decision, generate the next small sequence of prompt files needed to reduce the remaining misalignment between:

- the workflow we now want
- the architecture we say we want
- the bridge runner we currently have

## Important framing

This is still a design task, not a broad implementation task.

Do NOT refactor the runner in this pass.
Do NOT implement multiple modules in this pass.
Do NOT produce a giant roadmap.

Instead, define the execution-record artifact and then produce the smallest useful queue of next prompts.

## Requirements for Part 1

Create one markdown design artifact at:

`tools/codex/V1_Execution_Record_Artifact.md`

This document should include:

- purpose
- scope
- source of truth
- stable identity for a run
- required fields
- optional fields
- automatic vs manual fields
- execution status vs review status
- minimum failure-analysis section
- minimum resource/cost section
- what is intentionally deferred from V1

Keep it practical, small, and reusable.

## Requirements for Part 2

Create a short sequence of prompt files in `codex_prompts/`.

Create between **3 and 5** prompt files total.
Do not create more than 5.

Each prompt must:

- have one primary goal
- produce one primary artifact or decision
- be small enough to review before the next prompt is run
- follow from the execution-record design
- reduce a specific misalignment already identified in the review notes

Each prompt file should have:

- a clear filename with numeric prefix
- a focused task
- exact file paths where relevant
- explicit constraints
- explicit success criteria

## Also create one companion sequence note

Create a short note at:

`notes/005_prompt_queue_plan__TIMESTAMP.md`

This note should summarize:

- why these prompts were chosen
- why this order reduces risk
- what each prompt is meant to settle
- what larger work is intentionally deferred

## Constraints

1. Use the exact file paths listed above.
2. Treat `tools/Project_Design_Workflow.md` as governing.
3. Keep the execution-record artifact conservative and V1-sized.
4. Do not let the prompt queue become a giant backlog.
5. Prefer the smallest sequence that meaningfully reduces misalignment.
6. Do not implement the later prompts in this pass. Only write them.
7. Do not rewrite the workflow doc in this pass.

## Success criteria

This task is successful if:

- `tools/codex/V1_Execution_Record_Artifact.md` is clear and usable
- the execution-record design separates execution from review/acceptance
- the prompt queue contains only 3 to 5 bounded next prompts
- the prompts are ordered sensibly
- the outputs help us continue one reviewed step at a time
