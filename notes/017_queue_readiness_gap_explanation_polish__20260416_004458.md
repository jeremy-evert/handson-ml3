# 017 Queue Readiness Gap Explanation Polish

## Summary

Updated the human-readable stdout of `tools/codex/check_queue_readiness.py` for the default-target case where missing earlier V1 execution-record history can make the result look surprising.

The helper now prints one small `Queue note:` line when:

- default target selection is being used
- later prompts already have V1 execution records
- one or more earlier prompts still have no V1 record
- those missing-V1 prompts also have legacy `__SUCCESS__` notes that a human might otherwise mistake for queue history

## Why This Change Was Made

The current repo state includes legacy `__SUCCESS__` notes for prompts `002` through `010`, but those notes are not V1 execution records.

The helper was already making the correct conservative decision by defaulting back to prompt `002`. The usability issue was that the output did not clearly explain why older-looking success notes were being ignored.

The added note clarifies that:

- only V1 execution records in `notes/` count
- legacy `__SUCCESS__` notes do not count as V1 queue history
- missing V1 evidence for earlier prompts can pull the default target earlier than a human might first expect

## Logic Intentionally Left Unchanged

This pass did not change queue policy or readiness logic.

Specifically left unchanged:

- prompt discovery and ordering
- default target selection behavior
- latest-record selection behavior
- `ACCEPTED` / `UNREVIEWED` / `REJECTED` handling
- the meaning of missing prior V1 evidence
- treatment of legacy notes as non-authoritative for V1 queue readiness

The only behavior change is an additional explanatory output line in the narrow default-mode gap case.

## Validation

Validation performed in the current repo state:

1. `python3 tools/codex/check_queue_readiness.py`
   Outcome: succeeded. Default output still selected `codex_prompts/002_repo_inventory_and_status.md` and now included:
   `Queue note: default selection uses only V1 execution records in notes/. Legacy __SUCCESS__ notes do not count as V1 queue history, so prompts 002, 003, 004, 005, 006, 007, 008, 009, 010 still count as missing V1 evidence and can pull the default target earlier than older notes suggest.`

2. `python3 tools/codex/check_queue_readiness.py --prompt 002`
   Outcome: succeeded and still reported readiness correctly:
   `Ready: YES`

3. `python3 tools/codex/check_queue_readiness.py --prompt 013`
   Outcome: succeeded and still reported not-ready correctly:
   `Ready: NO`
   Reason remained `latest V1 run for the immediately previous prompt is UNREVIEWED`.

4. Protected files unchanged check:
   `git diff --name-only -- tools/codex/baby_run_prompt.py tools/codex/run_prompt.py tools/codex/review_run.py tools/codex/list_review_backlog.py`
   Outcome: no output, confirming those files remained unchanged.
