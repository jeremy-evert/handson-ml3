# Task: Build a small V1 queue-readiness checker for the next prompt

You are working in this repository.

Your task is to implement a small helper that answers the bounded V1 operational question:

`Is the next prompt in sequence ready to run under the current review-gate rules?`

## Important framing

This is a small helper task.

Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`

Those existing tools should remain unchanged in this pass.

You are building a thin read-only companion helper beside them, not refactoring the pipeline.

## Files to inspect

Read these exact files before making changes:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `notes/012_v1_pipeline_options_review__20260416_000819.md`
- `notes/012_top_three_next_options__20260416_000819.md`

You may also inspect the current prompt set and current execution records in:

- `codex_prompts/`
- `notes/`

## Goal

Create a small helper at:

`tools/codex/check_queue_readiness.py`

This helper should determine, from the current prompt files and current V1 execution records in `notes/`, whether the next prompt should be treated as ready under the current V1 review-gate rules.

The helper should stay small, inspectable, and conservative.

## What the helper must answer

The helper should answer these bounded questions:

1. what is the ordered prompt list based on numeric filename prefixes in `codex_prompts/`
2. what is the latest execution record for the relevant prompt
3. what is that record's `execution_status`
4. what is that record's `review_status`
5. based on the current V1 rule, is the next prompt ready

For V1, the key release rule is:

- only an `ACCEPTED` latest run for the immediately previous prompt releases the next prompt
- `UNREVIEWED` stops the queue
- `REJECTED` stops the queue
- missing prior run evidence stops the queue

## Minimum CLI behavior

Keep the CLI small and explicit.

Support one primary mode:

- no positional arguments required

Optional arguments are acceptable only if they remain small and directly useful, for example:

- `--prompt` to check readiness for a specific prompt file or numeric prefix

Do not add many modes or subcommands.

## Expected output

The helper should print a short human-readable readiness summary to stdout.

That summary should include at least:

- target prompt
- previous prompt, if one exists
- latest run record path considered
- latest run `execution_status`
- latest run `review_status`
- final readiness decision
- short reason

Keep the output textual and inspectable.

Do not create JSON outputs, databases, caches, or sidecar state.

## Important behavior rules

### 1. Preserve V1 boundaries

Do NOT:

- run prompts automatically
- update review fields automatically
- create or modify execution records
- release multiple future prompts
- build dependency-aware scheduling
- build a queue engine or status platform

This helper should only inspect and report the current readiness decision.

### 2. Respect the execution record as source of truth

Use the markdown execution record body in `notes/` as the source of truth for:

- `run_id`
- `prompt_file`
- `prompt_stem`
- `started_at_utc`
- `execution_status`
- `review_status`

Do not invent a separate state file.

### 3. Stay conservative about "latest"

If multiple records exist for the same prompt, pick the latest V1 run record using the execution-record evidence that is already present and inspectable.

Keep this logic simple and explainable.
Do not build a complex retry or history synthesizer.

### 4. Fail clearly on ambiguous or malformed inputs

If prompt ordering or record parsing is too malformed to make a safe decision, exit nonzero with a short error.

### 5. Keep the helper read-only

This helper should not modify files.

## Implementation guidance

Keep the implementation single-file unless a tiny helper is truly necessary.

A simple approach is preferred:

- discover prompt files with numeric prefixes
- sort them in sequence
- parse minimal field lines from candidate execution records in `notes/`
- identify the latest relevant record
- apply the V1 readiness rule
- print a short decision summary

Do not build a full markdown parser unless truly necessary.

## Required artifacts

### Artifact 1

Create:

`tools/codex/check_queue_readiness.py`

### Artifact 2

Create a short implementation note at:

`notes/014_queue_readiness_checker_build__TIMESTAMP.md`

This note should summarize:

- what was built
- what rule it applies for readiness
- what it intentionally does not do
- what validation was performed

## Validation requirements

After implementation, validate at least these points:

1. the helper can identify the ordered prompt sequence from `codex_prompts/`
2. the helper can find the latest run record for a prompt from `notes/`
3. the helper distinguishes `UNREVIEWED`, `ACCEPTED`, and `REJECTED`
4. the helper reports not-ready when the previous prompt is not accepted
5. the helper reports ready only when the previous prompt's latest run is accepted
6. `tools/codex/run_prompt.py` remains unchanged
7. `tools/codex/review_run.py` remains unchanged
8. `tools/codex/baby_run_prompt.py` remains unchanged

Include the validation outcome in the implementation note.

## Constraints

1. Use the exact file paths listed above.
2. Keep the helper small and read-only.
3. Do not modify the runner or review-writeback tools.
4. Do not alter the design documents in this pass.
5. Do not expand into dashboards, queue engines, retry orchestration, or broader automation.
6. Keep the implementation small enough to review comfortably.

## Success criteria

This task is successful if:

- `tools/codex/check_queue_readiness.py` exists and works
- it gives a conservative readiness decision grounded in current prompt order and latest execution-record review status
- it remains read-only and inspectable
- the implementation note explains the rule and validation clearly
- the existing runner and review-writeback tools remain unchanged
