# Task: Build a small V1 review write-back helper for execution records

You are working in this repository.

Your task is to implement a small helper that updates an existing V1 execution record in `notes/` with manual review information.

## Important framing

This is a small helper task.

Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`

Those runners should remain unchanged in this pass.

You are building a companion tool beside them, not refactoring them.

## Files to inspect

Read these exact files before making changes:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/run_prompt.py`
- `notes/001_smoke_test_pipeline__20260415_234918.md`

You may also read for context:

- `tools/codex/baby_run_prompt.py`
- `notes/009_run_prompt_candidate_build__20260415_233407.md`
- `notes/010_run_prompt_candidate_review__20260415_234559.md`
- `notes/010_next_step_recommendation__20260415_234559.md`

## Goal

Create a small helper at:

`tools/codex/review_run.py`

This helper should let a human reviewer update an existing V1 execution record with review information while preserving the existing record structure.

The helper should support the minimum V1 review write-back flow, not a broader workflow engine.

## What the helper must do

The helper should:

1. accept a path to an existing execution-record markdown file in `notes/`
2. verify that the file looks like a V1 execution record
3. allow the reviewer to set:
   - `review_status`
   - `review_summary`
4. optionally allow the reviewer to set:
   - `reviewed_by`
   - `reviewed_at_utc`
5. when `review_status` is `REJECTED`, optionally allow the reviewer to also set:
   - `failure_type`
   - `failure_symptom`
   - `likely_cause`
   - `recommended_next_action`
6. write the updates back into the same markdown file
7. preserve all other existing content unchanged
8. print the updated file path
9. exit nonzero on invalid input or malformed record structure

## Minimum CLI behavior

Keep the CLI small and explicit.

Required arguments:

- `record` → path to the record file to update
- `--review-status` → must be one of:
  - `ACCEPTED`
  - `REJECTED`

Required:
- `--review-summary`

Optional:
- `--reviewed-by`
- `--reviewed-at-utc`

Optional rejection-only fields:
- `--failure-type`
- `--failure-symptom`
- `--likely-cause`
- `--recommended-next-action`

## Important behavior rules

### 1. Preserve V1 boundaries
Do NOT:
- release the next prompt automatically
- compute queue state
- create additional files
- create sidecars or databases
- add automation beyond updating the one record

### 2. Respect the review gate
The helper should behave consistently with `tools/codex/V1_Run_Review_Gate.md`.

This means:
- it updates review facts
- it may update failure-analysis fields
- it does not try to redesign the workflow

### 3. Preserve record structure
The helper should update the existing markdown record in place.

Do not rewrite the record into a different format.
Do not reorder sections unnecessarily.
Do not remove prompt text, output, stderr, or resource facts.

### 4. Handle reviewed timestamp sensibly
If `--reviewed-at-utc` is not supplied, the helper should fill it automatically with current UTC time.

## Implementation guidance

Keep this helper small and inspectable.

A simple strategy is acceptable:
- read the whole file as text
- replace the specific field lines in the known V1 structure
- validate only the minimum required sections/fields before writing back

Do not build a large markdown parser unless it is truly necessary.

## Required artifacts

### Artifact 1
Create:

`tools/codex/review_run.py`

### Artifact 2
Create a short implementation note at:

`notes/011_review_writeback_helper_build__TIMESTAMP.md`

This note should summarize:

- what was built
- what V1 fields it updates
- what it intentionally does not do
- what validation was performed

### Artifact 3
Validate the helper by applying it to this record:

`notes/001_smoke_test_pipeline__20260415_234918.md`

Use a review outcome of:

- `ACCEPTED`

Use a short review summary that makes sense for the smoke test.

The goal is to prove that the helper can update an existing V1 execution record in place.

## Validation requirements

After implementation, validate at least these points:

1. the helper updates `review_status`
2. the helper updates `review_summary`
3. the helper fills `reviewed_at_utc`
4. the helper preserves all other sections
5. the updated record still looks like a V1 execution record
6. `tools/codex/run_prompt.py` remains unchanged
7. `tools/codex/baby_run_prompt.py` remains unchanged

Include the validation outcome in the implementation note.

## Constraints

1. Use the exact file paths listed above.
2. Keep the helper single-file unless a tiny helper is truly necessary.
3. Do not modify the runners in this pass.
4. Do not alter the design documents in this pass.
5. Do not build broader queue or workflow logic.
6. Keep the implementation small enough to review comfortably.

## Success criteria

This task is successful if:

- `tools/codex/review_run.py` exists and works
- it can update a V1 execution record in place
- it writes back review facts consistent with the V1 review gate
- the validation record is successfully updated to `ACCEPTED`
- both runners remain unchanged
- the helper remains thin and inspectable
