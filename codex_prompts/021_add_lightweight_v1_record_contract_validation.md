# Task: Add lightweight validation for the shared V1 execution-record contract

You are working in this repository.

Your task is to add a small, repeatable validation layer for the shared markdown execution-record shape used across the current V1 tools.

Keep this implementation lightweight and inspectable.

## Primary goal

Create one small validation path that protects the shared V1 markdown record contract without introducing a larger framework or platform surface.

## Files to inspect

Read these exact files before editing:

- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
- `notes/018_prioritized_remaining_work__20260416_005130.md`

## Required implementation scope

Create a lightweight validation layer for the shared V1 markdown execution-record shape used by:

- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

The implementation should cover only the minimum contract that these tools already depend on, such as:

- required section presence and order
- required field-line presence
- allowed execution-status values
- allowed review-status values
- the expected `run_id` pattern including the optional same-second numeric suffix
- basic consistency checks like title/run-id agreement and prompt-file/prompt-stem agreement where appropriate

## Output artifacts to create

Create exactly these artifacts:

1. one small shared validation module under `tools/codex/`
2. only the minimal script changes needed so the current V1 tools reuse that validation instead of carrying separate ad hoc contract checks
3. one short implementation note:
   - `notes/021_record_contract_validation__TIMESTAMP.md`

## Constraints

- Do not build a larger test framework
- Do not introduce a service, daemon, dashboard, or platform layer
- Do not add a database, JSON sidecar, or alternate record format
- Do not redesign the V1 markdown artifact
- Do not expand queue semantics, retry tooling, or orchestration behavior
- Do not add broad dependency or configuration systems
- Keep the validation readable enough that a reviewer can inspect it quickly in one sitting

## Validation requirements

Validate the work by doing all of the following:

1. Run a lightweight syntax check on the touched Python files.
2. Run at least one small direct validation path against existing repo records in `notes/` so the shared validator proves it can parse or reject records using the current contract.
3. Confirm `run_prompt.py` still writes records that the shared validator accepts.
4. Confirm `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py` still work with the shared validator in place.
5. Keep validation evidence in the implementation note concise and concrete.

## Success criteria

This task is successful if:

1. The shared V1 markdown record contract is enforced through one lightweight reusable validation path.
2. The four V1 scripts no longer each carry their own independent, partially duplicated contract assumptions where simple sharing would suffice.
3. The validator remains small, inspectable, and local to the current V1 workflow.
4. No broader platform growth is introduced.
5. The result is still easy to review before execution.
