# 021 Record Contract Validation

## What Changed

- Added `tools/codex/v1_record_validation.py` as one small shared parser/validator for the V1 markdown execution-record contract.
- Moved the repeated section-order, field-line, status-value, `run_id`, and prompt-identity checks out of `run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py`.
- Kept the current V1 markdown artifact and tool boundaries unchanged.

## Contract Covered

- required section presence and order
- required field-line presence
- allowed `execution_status` values
- allowed `review_status` values
- `run_id` pattern: `<prompt_stem>__<started_at_utc>` with optional same-second `__<n>` suffix
- `title == run_id`
- `prompt_file` stem matches `prompt_stem`
- filename stem matches `run_id` when validating a record file

## Validation Evidence

- Syntax check: `python -m py_compile tools/codex/run_prompt.py tools/codex/review_run.py tools/codex/check_queue_readiness.py tools/codex/list_review_backlog.py tools/codex/v1_record_validation.py`
- Existing-record parse path: shared validator accepted `notes/011_build_v1_review_writeback_helper__20260415_235346.md`
- Existing-note non-record path: shared validator returned `None` for `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
- Existing-record reject path: mutating `review_status` from the accepted record text to ``BROKEN`` raised `record has invalid review_status`
- Runner write path: `run_prompt.py` was exercised with a stubbed `codex` executable, wrote a temporary V1 record, and that record was accepted by the shared validator before the temporary note was removed
- Review write-back path: `review_run.py` was exercised against that temporary record and the updated record still validated
- Read-only helpers: `python tools/codex/list_review_backlog.py --unreviewed-only` and `python tools/codex/check_queue_readiness.py --prompt 021` both ran successfully with the shared validator in place

## Intentional Limits

- no new framework, database, sidecar, or alternate record format
- no queue-semantics expansion
- no broader orchestration or retry tooling
