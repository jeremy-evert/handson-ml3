# 015 Review Backlog Lister Build

## What Was Built

Built a small read-only helper at [list_review_backlog.py](/data/git/handson-ml3/tools/codex/list_review_backlog.py) that:

- scans `notes/` for top-level V1 execution-record files
- parses the execution-record body as the source of truth for `run_id`, `prompt_file`, `prompt_stem`, `started_at_utc`, `execution_status`, and `review_status`
- lists all current `UNREVIEWED` records
- identifies the latest execution record per prompt
- surfaces a small "likely needs human review next" view by selecting prompts whose latest record is `UNREVIEWED`

The CLI stays small:

- default summary output with no positional arguments
- optional `--unreviewed-only` filter to narrow the latest-per-prompt view to prompts whose latest record is still `UNREVIEWED`

## Conservative Parsing Policy

The helper intentionally does not use a full markdown parser, but it also does not treat every note with copied field lines as a record.

It only accepts files that look like real top-level V1 execution records:

- the file starts with a markdown title
- the title matches the record `run_id`
- the required V1 sections exist in the expected order
- the required minimal V1 fields are present and valid

Policy on malformed inputs:

- non-record notes are skipped
- record-like files with incomplete or malformed required V1 structure fail clearly with an error

This keeps the backlog view conservative and inspectable while avoiding false positives from design notes that embed record examples.

## What Backlog View It Provides

The helper prints a short stdout summary with:

- all `UNREVIEWED` records found
- the latest record per prompt
- the prompts that likely need human review next because their latest record is `UNREVIEWED`

Each listed record includes:

- record path
- prompt file
- started timestamp
- execution status
- review status

For latest-record selection, it uses the record body and picks the highest `started_at_utc` for each prompt, with the run-id same-second suffix used only as a simple tiebreaker.

## What It Intentionally Does Not Do

- does not modify any files
- does not update review fields
- does not release prompts or compute queue readiness
- does not create JSON, caches, sidecars, or indexes
- does not build a dashboard, TUI, or broader reporting layer
- does not change `tools/codex/baby_run_prompt.py`
- does not change `tools/codex/run_prompt.py`
- does not change `tools/codex/review_run.py`

## Validation Performed

1. Confirmed the helper finds V1 execution records in `notes/` by running:
   `python3 tools/codex/list_review_backlog.py`
2. Confirmed it lists records still marked `UNREVIEWED` from the current V1 set.
3. Confirmed it identifies the latest record per prompt, including prompt `001` where the accepted later record supersedes the earlier failed record.
4. Confirmed it produces a small "likely needs human review next" summary from latest-record status.
5. Confirmed the helper remains read-only by inspecting the implementation and verifying it only reads `notes/*.md` and prints stdout.
6. Confirmed `tools/codex/run_prompt.py` remains unchanged.
7. Confirmed `tools/codex/review_run.py` remains unchanged.
8. Confirmed `tools/codex/baby_run_prompt.py` remains unchanged.
9. Confirmed the optional bounded filter works by running:
   `python3 tools/codex/list_review_backlog.py --unreviewed-only`

Validation outcome: passed for the bounded V1 review-discovery behavior above.
