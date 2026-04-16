# 014 Queue Readiness Checker Build

## What Was Built

Built a small read-only helper at [check_queue_readiness.py](/data/git/handson-ml3/tools/codex/check_queue_readiness.py) that:

- discovers prompt order from numeric prefixes in `codex_prompts/`
- parses minimal V1 execution-record field lines from markdown files in `notes/`
- finds the latest V1 record for the relevant prior prompt
- reports whether the target prompt is ready under the current V1 review gate

The helper supports:

- default queue-head checking with no positional arguments
- `--prompt` for a specific prompt file, filename stem, or numeric prefix

## Readiness Rule Applied

The helper applies the current V1 gate conservatively:

- the first prompt in sequence is ready because it has no prior review gate
- any later prompt is ready only if the latest V1 run for the immediately previous prompt has `review_status: ACCEPTED`
- `UNREVIEWED` stops the queue
- `REJECTED` stops the queue
- missing prior V1 run evidence stops the queue

For latest-record selection, the helper uses the execution-record body as the source of truth and picks the highest `started_at_utc` for the prompt, with the run-id collision suffix used only as a simple same-second tiebreaker.

## What It Intentionally Does Not Do

- does not run prompts
- does not modify execution records
- does not update review fields
- does not create sidecar state, caches, or JSON outputs
- does not release multiple future prompts
- does not build a queue engine or dashboard

## Validation Performed

1. Confirmed ordered prompt discovery from `codex_prompts/` with the helper default output.
2. Confirmed latest-run lookup from `notes/` by checking `--prompt 002`, which selected [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md) as the latest relevant prior record.
3. Confirmed review-status distinction:
   - `--prompt 002` reported ready because prior prompt `001` is `ACCEPTED`
   - `--prompt 013` reported not ready because prior prompt `012` is `UNREVIEWED`
   - a small in-memory check against `evaluate_readiness(...)` reported not ready when the prior prompt's latest run was `REJECTED`
4. Confirmed not-ready behavior when the previous prompt is not accepted or is missing:
   - `--prompt 013` and `--prompt 014` stopped on `UNREVIEWED`
   - `--prompt 003` stopped because prompt `002` has no current V1 run record
5. Confirmed ready behavior only when the previous prompt's latest run is accepted via `--prompt 002`.
6. Confirmed first-prompt handling via `--prompt 001`, which reported ready with no prior gate.
7. Confirmed `tools/codex/run_prompt.py` remained unchanged.
8. Confirmed `tools/codex/review_run.py` remained unchanged.
9. Confirmed `tools/codex/baby_run_prompt.py` remained unchanged.

Validation outcome: passed for the bounded V1 behavior above.
