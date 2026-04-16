# Task: Polish the queue-readiness checker output for missing V1 history gaps

You are working in this repository.

Your task is to make one small usability improvement to the queue-readiness helper.

## Important framing

This is a small polish task.

Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/list_review_backlog.py`

Do NOT rewrite design documents in this pass.

You may modify only:

- `tools/codex/check_queue_readiness.py`

and create the required implementation note.

## Files to inspect

Read these exact files before making changes:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/check_queue_readiness.py`
- `notes/016_queue_and_backlog_helper_validation__20260416_003710.md`
- `notes/016_next_improvement_recommendation__20260416_003710.md`

Also inspect current repo contents in:

- `codex_prompts/`
- `notes/`

## Goal

Improve the stdout output of `tools/codex/check_queue_readiness.py` so that when the default target is chosen conservatively because earlier prompts have no V1 execution-record history, the output includes a short explanation that makes this easier for a human to understand.

This is a usability clarification, not a policy change.

## Problem to address

The current helper is making the correct conservative decision, but in the current repo state the default result can be confusing because:

- prompts `002` through `010` have legacy `__SUCCESS__` notes
- those legacy notes are not V1 execution records
- the helper correctly ignores them
- the default output does not currently explain that gap clearly enough

## Required behavior change

Keep the current readiness logic unchanged.

Add a short explanatory line or small explanatory block to the human-readable output when all of the following are true:

1. the helper is using default target selection
2. the chosen target is blocked or selected because prior prompts lack V1 execution-record history
3. legacy-looking success notes or otherwise non-V1 history may make the result surprising to a human

The explanation should stay small, practical, and inspectable.

It should clarify that:

- the helper is using only V1 execution records in `notes/`
- legacy `__SUCCESS__` notes do not count as V1 queue history
- this is why the default target may point back to an earlier prompt than a human might first expect

## Important constraints

### 1. Do not change the queue rule
Do NOT change:
- how prompt order is computed
- how latest records are selected
- the `ACCEPTED` / `UNREVIEWED` / `REJECTED` logic
- the meaning of missing prior V1 evidence

This pass is only about clearer output.

### 2. Keep output conservative and small
Do not add:
- dashboards
- verbose debug dumps
- broad repo scans in the output
- speculative migration behavior
- any automatic inference from legacy notes

### 3. Keep the helper read-only
Do not modify any files other than the helper source itself and the implementation note.

## Required artifacts

### Artifact 1
Update:

`tools/codex/check_queue_readiness.py`

### Artifact 2
Create a short implementation note at:

`notes/017_queue_readiness_gap_explanation_polish__TIMESTAMP.md`

This note should summarize:

- what output was changed
- why the change was made
- what logic was intentionally left unchanged
- what validation was performed

## Validation requirements

After the change, validate at least these points:

1. `python3 tools/codex/check_queue_readiness.py` still succeeds
2. the default output now includes a clearer explanation when the repo state includes missing V1 history for earlier prompts
3. `python3 tools/codex/check_queue_readiness.py --prompt 002` still reports readiness correctly
4. `python3 tools/codex/check_queue_readiness.py --prompt 013` still reports not-ready correctly
5. `tools/codex/list_review_backlog.py` remains unchanged
6. `tools/codex/run_prompt.py` remains unchanged
7. `tools/codex/review_run.py` remains unchanged
8. `tools/codex/baby_run_prompt.py` remains unchanged

Include the validation outcome in the implementation note.

## Success criteria

This task is successful if:

- `tools/codex/check_queue_readiness.py` keeps the same readiness logic
- the default output is easier for a human to understand in the current repo state
- the helper stays small, read-only, and conservative
- the implementation note clearly explains the polish and validation
