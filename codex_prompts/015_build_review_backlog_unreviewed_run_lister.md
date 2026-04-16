# Task: Build a small V1 review backlog and unreviewed-run lister

You are working in this repository.

Your task is to implement a small helper that answers the bounded V1 operational question:

`What execution records still need human review, and what are the latest records per prompt?`

## Important framing

This is a small helper task.

Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`

Those existing tools should remain unchanged in this pass.

You are building a thin read-only review-discovery helper, not a dashboard or workflow engine.

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

You may also inspect the current execution records in:

- `notes/`

## Goal

Create a small helper at:

`tools/codex/list_review_backlog.py`

This helper should scan V1 execution records in `notes/` and produce a small human-readable review backlog summary.

The helper should stay focused on review discovery, not queue control.

## What the helper must answer

The helper should answer these bounded questions:

1. which execution records are still `UNREVIEWED`
2. what the latest execution record is for each prompt
3. which prompts most likely need human review next based on latest-record status

For V1, "likely needs human review next" should stay simple and inspectable:

- latest record is `UNREVIEWED`
- latest record path is known
- prompt identity is known

Do not try to prioritize by a rich policy engine.

## Minimum CLI behavior

Keep the CLI small and explicit.

Support one primary mode:

- no positional arguments required

Optional small filters are acceptable only if they remain clearly bounded, for example:

- `--unreviewed-only`

Do not add many modes, interactive behavior, or subcommands.

## Expected output

The helper should print a short human-readable summary to stdout.

That summary should include at least:

- a section or block listing all `UNREVIEWED` records found
- the latest record per prompt
- a short "needs review next" summary derived from those latest records

For each listed record, include at least:

- record path
- prompt stem or prompt file
- started timestamp
- execution status
- review status

Keep the output textual and inspectable.

Do not create JSON outputs, databases, caches, sidecars, or a broader reporting layer.

## Important behavior rules

### 1. Preserve V1 boundaries

Do NOT:

- update records automatically
- release prompts automatically
- compute broader queue readiness policy
- build a dashboard, TUI, or web view
- add retry orchestration
- add analytics beyond the immediate review backlog summary

This helper should only surface the current review backlog from existing records.

### 2. Respect the execution record as source of truth

Use the markdown execution record body in `notes/` as the source of truth for:

- `run_id`
- `prompt_file`
- `prompt_stem`
- `started_at_utc`
- `execution_status`
- `review_status`

Do not introduce a separate index file.

### 3. Stay conservative about "latest"

If multiple records exist for the same prompt, choose the latest record using simple, inspectable logic grounded in the existing record evidence.

Do not build retry graphs or full history synthesis.

### 4. Fail clearly on malformed inputs

If a record cannot be parsed well enough for the minimal summary, either skip it with a short warning or fail clearly if that is safer.

Pick one small consistent policy and document it in the implementation note.

### 5. Keep the helper read-only

This helper should not modify files.

## Implementation guidance

Keep the implementation single-file unless a tiny helper is truly necessary.

A simple approach is preferred:

- scan markdown execution records in `notes/`
- parse minimal field lines
- group records by prompt
- identify latest record per prompt
- list `UNREVIEWED` records
- print a small review backlog summary

Do not build a full markdown parser unless truly necessary.

## Required artifacts

### Artifact 1

Create:

`tools/codex/list_review_backlog.py`

### Artifact 2

Create a short implementation note at:

`notes/015_review_backlog_lister_build__TIMESTAMP.md`

This note should summarize:

- what was built
- what backlog view it provides
- what it intentionally does not do
- what validation was performed

## Validation requirements

After implementation, validate at least these points:

1. the helper can find V1 execution records in `notes/`
2. the helper can list records still marked `UNREVIEWED`
3. the helper can identify the latest record per prompt
4. the helper can produce a small "needs review next" summary
5. the helper remains read-only
6. `tools/codex/run_prompt.py` remains unchanged
7. `tools/codex/review_run.py` remains unchanged
8. `tools/codex/baby_run_prompt.py` remains unchanged

Include the validation outcome in the implementation note.

## Constraints

1. Use the exact file paths listed above.
2. Keep the helper small and read-only.
3. Do not modify the runner or review-writeback tools.
4. Do not alter the design documents in this pass.
5. Do not expand into dashboards, queue engines, analytics platforms, or broader automation.
6. Keep the implementation small enough to review comfortably.

## Success criteria

This task is successful if:

- `tools/codex/list_review_backlog.py` exists and works
- it can surface current `UNREVIEWED` records and latest records per prompt
- it remains a small review-discovery helper rather than a broader system
- the implementation note explains the scope and validation clearly
- the existing runner and review-writeback tools remain unchanged
