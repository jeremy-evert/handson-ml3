# Task: Generate the next two implementation prompts for the current V1 workflow cleanup

You are working in this repository.

Your task is to write the next two bounded implementation prompt files based on the current architecture sweep and prioritized remaining work.

## Important framing

This is a prompt-generation task only.

Do NOT implement code or doc edits in this pass.
Do NOT modify existing tools or design documents in this pass.

Your job is to create the next two implementation prompts as files in `codex_prompts/`.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
- `notes/018_prioritized_remaining_work__20260416_005130.md`

## Goal

Generate exactly two prompt files:

1. a prompt to perform the recommended **doc/spec alignment cleanup**
2. a prompt to implement the recommended **lightweight record-contract validation**

These prompts should be implementation-ready, bounded, and consistent with the current repo state.

## Prompt 1 requirements

Create a prompt file for doc/spec alignment cleanup.

This future task should focus on aligning the design packet with the actual V1 toolset and current behavior, including at least:

- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- any stale references to `baby_run_prompt.py` where the actual V1 runner is now `run_prompt.py`
- any stale implication that readiness/backlog helpers do not yet exist
- the current run-id collision suffix behavior where relevant

The prompt should keep the cleanup bounded and should not turn into a broad architecture rewrite.

## Prompt 2 requirements

Create a prompt file for lightweight record-contract validation.

This future task should focus on creating a small repeatable validation layer for the shared V1 markdown execution-record shape used by:

- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

The prompt should keep the validation lightweight and inspectable.

Avoid proposing a large test framework, platform service, or dashboard.

## What each generated prompt must include

Each prompt file must:

- have a clear numeric filename prefix
- define one primary goal only
- name exact files to inspect
- define exact output artifacts to create
- include constraints that prevent broad platform growth
- define validation requirements
- define success criteria
- stay small enough to review comfortably before execution

## Also create one planning note

Create a short note at:

`notes/019_next_two_cleanup_prompts_plan__TIMESTAMP.md`

This note should summarize:

- why these two prompts were chosen
- why they are ordered this way
- which one should be executed first
- what should explicitly wait until later

## Constraints

1. Generate exactly two prompt files, no more.
2. Do not implement the cleanup tasks in this pass.
3. Keep both generated prompts bounded and V1-sized.
4. Do not introduce a broader orchestration layer, queue engine, retry manager, or platform expansion.
5. Let the architecture sweep drive the prompts.

## Success criteria

This task is successful if:

- exactly two prompt files are created
- one is for doc/spec alignment cleanup
- one is for lightweight record-contract validation
- both prompts are implementation-ready and bounded
- the planning note clearly explains the sequence
