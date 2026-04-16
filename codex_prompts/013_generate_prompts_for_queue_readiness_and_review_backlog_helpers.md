# Task: Generate the next two implementation prompts for the V1 workflow pipeline

You are working in this repository.

Your task is to write the next two bounded Codex prompt files for the V1 pipeline based on the current repo state and the recent options review.

## Important framing

This is a prompt-generation task only.

Do NOT implement either helper in this pass.
Do NOT modify existing code or design documents in this pass.
Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`

Your job is to generate the next two implementation prompts as files in `codex_prompts/`.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `notes/012_v1_pipeline_options_review__20260416_000819.md`
- `notes/012_top_three_next_options__20260416_000819.md`

## Goal

Generate exactly two prompt files:

1. a prompt to build the recommended **queue-readiness checker**
2. a prompt to build the recommended **review backlog / unreviewed-run lister**

These prompts should be ready for later execution through the current V1 pipeline.

## Prompt 1 requirements

Create a prompt file for the queue-readiness checker.

This future helper should answer a bounded operational question such as:

- given the current prompt set and current records in `notes/`, is the next prompt ready to run?
- what is the latest run for the current or previous prompt?
- is that latest run still `UNREVIEWED`, `ACCEPTED`, or `REJECTED`?
- should the next prompt be treated as ready under the current V1 rules?

The prompt should keep the helper small, inspectable, and consistent with the current architecture and review gate.

## Prompt 2 requirements

Create a prompt file for the review backlog / unreviewed-run lister.

This future helper should answer bounded operational questions such as:

- which execution records are still `UNREVIEWED`?
- what are the latest records per prompt?
- what likely needs human review next?

The prompt should keep the helper small, inspectable, and avoid turning into a broader dashboard or queue engine.

## What each generated prompt must include

Each prompt file must:

- have a clear numeric filename prefix
- define one primary goal only
- name exact files to inspect
- define exact output artifacts to create
- include constraints that prevent broad workflow-engine growth
- define success criteria
- stay small enough to review comfortably before execution

## Also create one planning note

Create a short note at:

`notes/013_next_two_prompt_plan__TIMESTAMP.md`

This note should summarize:

- why these two prompts were chosen
- why they are ordered this way
- which one should be executed first
- what should explicitly wait until later

## Constraints

1. Generate exactly two prompt files, no more.
2. Do not implement the helpers in this pass.
3. Keep both generated prompts bounded and V1-sized.
4. Avoid speculative expansion into dashboards, dependency engines, or broader automation systems.
5. Use the current repo evidence and the options review to shape the prompts.

## Success criteria

This task is successful if:

- exactly two prompt files are created
- one is for the queue-readiness checker
- one is for the review backlog / unreviewed-run lister
- both prompts are implementation-ready and bounded
- the planning note clearly explains the sequence
