# 011 Review Write-Back Helper Build

## What Was Built

Built [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py), a thin V1 companion helper that updates one existing execution record in `notes/` in place.

The helper:

- accepts a record path plus explicit review arguments
- verifies the file is under `notes/` and matches the minimum V1 execution-record structure
- updates only the targeted field lines
- writes the changes back to the same markdown file
- prints the updated file path
- exits nonzero on invalid input or malformed structure

## V1 Fields It Updates

Always updates:

- `review_status`
- `review_summary`
- `reviewed_at_utc`

Optionally updates:

- `reviewed_by`

For `REJECTED` runs, it can also update:

- `failure_type`
- `failure_symptom`
- `likely_cause`
- `recommended_next_action`

## What It Intentionally Does Not Do

- does not modify `tools/codex/run_prompt.py`
- does not modify `tools/codex/baby_run_prompt.py`
- does not create queue state, sidecars, databases, or extra workflow files
- does not release the next prompt automatically
- does not redesign the V1 record format
- does not rewrite or reorder non-review sections

## Validation

Validation commands performed:

- `python3 -m py_compile tools/codex/review_run.py`
- `python3 tools/codex/review_run.py notes/001_smoke_test_pipeline__20260415_234918.md --review-status ACCEPTED --review-summary 'Smoke test output and artifact are complete enough to accept this bounded step.'`
- `git diff -- tools/codex/run_prompt.py tools/codex/baby_run_prompt.py`

Validation outcome:

- `review_status` in [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md) changed from `UNREVIEWED` to `ACCEPTED`.
- `review_summary` was written in place.
- `reviewed_at_utc` was auto-filled as `20260415_235508`.
- All other sections remained unchanged; the diff was limited to the three review lines above.
- The updated record still matches the expected V1 section order and required field layout.
- `tools/codex/run_prompt.py` remained unchanged.
- `tools/codex/baby_run_prompt.py` remained unchanged.

## Scope Note

This helper is intentionally a minimum V1 review write-back tool. It updates review facts on one existing execution record and stops there.
