# Task: Build the first proper V1 runner candidate beside baby_run_prompt.py

You are working in this repository.

Your task is to implement a new runner candidate at:

`tools/codex/run_prompt.py`

This new runner should be the first proper V1 implementation candidate for prompt execution records.

## Important framing

Do NOT modify:

- `tools/codex/baby_run_prompt.py`

That file is a working bootstrap artifact and should remain untouched in this pass.

You may read it for reference or inspiration, but you must build the new runner as a separate implementation.

## Files to inspect

Read these exact files before making changes:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/baby_run_prompt.py`

## Goal

Create a new runner at:

`tools/codex/run_prompt.py`

The new runner should implement the V1 execution-record behavior described in the design documents while preserving a thin, inspectable scope.

This is a replacement candidate, not a broad framework.

## What the new runner must do

The new runner should:

1. resolve one prompt file from the provided argument
2. run `codex exec` in the repository root
3. capture the final Codex message via `--output-last-message`
4. capture stderr and subprocess return code
5. write one V1 execution-record markdown file into `notes/`
6. use stable run identity naming based on:
   - `<prompt_stem>__<started_at_utc>`
7. initialize:
   - `review_status: UNREVIEWED`
   - blank review fields
   - blank failure-analysis fields
8. separate:
   - execution outcome
   - review outcome
9. capture the minimum automatic resource/cost metrics defined by the spec
10. print the written execution-record path
11. exit with the subprocess return code

## What the new runner must NOT do

Do NOT:

- modify `baby_run_prompt.py`
- implement review write-back
- implement queue progression logic
- implement dependency handling
- implement retry orchestration
- redesign the broader CLI
- split the system into multiple modules unless one tiny helper is absolutely necessary
- introduce JSON sidecars, databases, or other new persistence layers

## Collision rule

The spec left one open question about same-second collisions for `run_id`.

Settle it in the smallest practical way for V1:

- default to second-precision UTC timestamps
- if the target run-record path already exists, append a short numeric suffix such as `__2`, `__3`, etc.
- document this briefly in the implementation note you create below

Do not invent a larger identity system.

## Implementation guidance

Preserve the spirit of the current bootstrap runner where it is still useful:

- simple CLI with one required prompt argument
- repo-root execution
- prompt resolution from path / repo-relative path / `codex_prompts/` / unique prefix match
- direct subprocess use for `codex exec`

But the output artifact must follow the V1 execution-record design rather than the old success/failure note model.

## Required artifacts

### Artifact 1
Create or update:

`tools/codex/run_prompt.py`

### Artifact 2
Create a short implementation note at:

`notes/009_run_prompt_candidate_build__TIMESTAMP.md`

This note should summarize:

- what was built
- what behavior was intentionally preserved from `baby_run_prompt.py`
- what changed to support the V1 execution record
- how the collision rule was handled
- what was intentionally deferred

### Artifact 3
Create one sample execution-record file by running the new runner against this prompt:

`codex_prompts/001_smoke_test_pipeline.md`

Use the new runner itself for this validation run.

The goal is to prove that the new runner can emit a correctly shaped V1 execution record in `notes/`.

## Validation requirements

After implementation, validate at least these points:

1. the new runner can resolve and execute the sample prompt
2. the written record filename follows the V1 naming rule
3. the record includes:
   - run identity
   - execution facts
   - review facts
   - failure-analysis section
   - resource/cost section
   - prompt text
   - Codex final output
   - stderr section
4. `review_status` starts as `UNREVIEWED`
5. `execution_status` is derived only from subprocess result
6. `baby_run_prompt.py` remains unchanged

Include the validation outcome in the implementation note.

## Constraints

1. Use the exact file paths listed above.
2. Keep the new runner thin and inspectable.
3. Prefer one file unless a tiny helper is truly necessary.
4. Do not perform a broader refactor in this pass.
5. Do not alter the design documents in this pass.
6. Do not change the bootstrap runner in this pass.

## Success criteria

This task is successful if:

- `tools/codex/run_prompt.py` exists and runs
- it emits a V1 execution record rather than the old success/failure note format
- execution and review are clearly separated in the written record
- the validation run produces one correctly shaped record in `notes/`
- `baby_run_prompt.py` is unchanged
- the implementation remains small enough to review comfortably
