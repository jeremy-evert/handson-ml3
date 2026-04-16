# Task: Validate the new queue-readiness and review-backlog helpers against the current repo state

You are working in this repository.

Your task is to exercise the two newly built read-only helpers against the current prompt and note history, then write a short validation report.

## Important framing

This is a validation and review task.

Do NOT modify existing code in this pass.
Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

Do NOT rewrite design documents in this pass.

Your job is to run the current helpers, inspect their outputs against the current repo state, and summarize whether they behave correctly and conservatively.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`

Also inspect current repo contents in:

- `codex_prompts/`
- `notes/`

## Goal

Validate whether the two helpers behave correctly against the repo’s current V1 artifacts.

The two helpers are:

- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

## What to do

### 1. Run the queue-readiness helper
Run at least these checks:

- `python3 tools/codex/check_queue_readiness.py`
- `python3 tools/codex/check_queue_readiness.py --prompt 001`
- `python3 tools/codex/check_queue_readiness.py --prompt 002`
- `python3 tools/codex/check_queue_readiness.py --prompt 013`
- `python3 tools/codex/check_queue_readiness.py --prompt 014`
- `python3 tools/codex/check_queue_readiness.py --prompt 015`

### 2. Run the review-backlog helper
Run at least these checks:

- `python3 tools/codex/list_review_backlog.py`
- `python3 tools/codex/list_review_backlog.py --unreviewed-only`

### 3. Compare outputs against actual repo evidence
Check whether the helper outputs agree with:

- actual prompt ordering in `codex_prompts/`
- actual review statuses in the current execution records in `notes/`
- the intended V1 rules in the design documents

## Questions to answer

Please answer these questions in the report:

### A. Queue-readiness correctness
- Does `check_queue_readiness.py` appear to choose the correct target and prior prompt?
- Do its readiness decisions match the current record evidence?
- Are there any edge cases or confusing behaviors visible from the current repo state?

### B. Review-backlog correctness
- Does `list_review_backlog.py` appear to identify `UNREVIEWED` records correctly?
- Does it pick the latest record per prompt correctly?
- Does its “likely needs human review next” view match the repo evidence?

### C. Consistency between the two helpers
- Do the two helpers tell a coherent story about the current repo state?
- Is there any mismatch between what the readiness checker says and what the backlog lister says?

### D. Smallest next improvement
- Based on this validation pass, what is the single smallest next improvement or cleanup step?
- Prefer one bounded suggestion only.

## Required output artifacts

Create exactly two artifacts.

### Artifact 1
Create a validation report at:

`notes/016_queue_and_backlog_helper_validation__TIMESTAMP.md`

This report should include:

- short summary
- queue-readiness validation findings
- review-backlog validation findings
- consistency findings
- one recommended next improvement

### Artifact 2
Create a short recommendation note at:

`notes/016_next_improvement_recommendation__TIMESTAMP.md`

This should contain only:

- the recommended next bounded improvement
- why it should happen next
- what should explicitly wait

## Constraints

1. Use the exact file paths listed above.
2. Do not modify the helpers in this pass.
3. Do not create a broad roadmap.
4. Recommend only one bounded next improvement.
5. Keep the validation practical and evidence-based.

## Success criteria

This task is successful if:

- both helpers are exercised against the current repo state
- the report compares their outputs to actual prompt and note evidence
- any mismatches or edge cases are clearly identified
- one small next improvement is recommended
- the result helps decide the next deliberate step with confidence
