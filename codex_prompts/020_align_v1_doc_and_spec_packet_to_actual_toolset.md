# Task: Align the V1 doc/spec packet to the actual implemented toolset

You are working in this repository.

Your task is to perform a bounded doc/spec alignment cleanup so the current V1 design packet matches the implemented workflow.

This is a cleanup pass, not a new architecture pass.

## Primary goal

Make the current V1 design packet accurately describe the actual implemented toolset and current behavior without expanding scope or redesigning the workflow.

## Files to inspect

Read these exact files before editing:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
- `notes/018_prioritized_remaining_work__20260416_005130.md`

## Required cleanup scope

Keep the work bounded to doc/spec alignment for the current V1 slice.

The cleanup must address at least these items:

1. `tools/codex/V1_Bridge_Runner_Change_Spec.md`
   - remove stale references to `tools/codex/baby_run_prompt.py`
   - align the document to the real runner at `tools/codex/run_prompt.py`
   - stop describing review write-back and queue support as unimplemented when helper scripts now exist

2. Any stale references in the inspected design packet that still imply:
   - the runner is `baby_run_prompt.py`
   - readiness and backlog helpers do not yet exist
   - the V1 run id is always unsuffixed

3. The current run-id collision behavior where relevant
   - document that the base V1 identity is `<prompt_stem>__<started_at_utc>`
   - document that `run_prompt.py` adds a numeric suffix such as `__2` only when needed to avoid same-second collisions
   - keep that explanation small and implementation-accurate

## Output artifacts to create

Create exactly one note:

- `notes/020_doc_spec_alignment_cleanup__TIMESTAMP.md`

Update only the minimum necessary design/spec files from the inspected list.

## Constraints

- Do not modify `tools/codex/run_prompt.py`
- Do not modify `tools/codex/review_run.py`
- Do not modify `tools/codex/check_queue_readiness.py`
- Do not modify `tools/codex/list_review_backlog.py`
- Do not rewrite the overall architecture
- Do not introduce new workflow states, new tools, or new platform layers
- Do not expand into retry tooling, scheduling, queue engines, dashboards, or orchestration
- Keep this as a V1-sized doc/spec cleanup only

## Validation requirements

Validate the cleanup by doing all of the following:

1. Confirm the updated design/spec files consistently name `tools/codex/run_prompt.py` as the active V1 runner where applicable.
2. Confirm the updated packet no longer implies that review write-back, readiness checking, or backlog listing are still future work if those helpers are already present.
3. Confirm the run-id wording is consistent with current code behavior, including same-second collision suffixes where relevant.
4. Use a search pass to verify there are no remaining stale `baby_run_prompt.py` references inside the edited design/spec files unless a historical reference is explicitly intentional and clearly marked as historical.

## Success criteria

This task is successful if:

1. The V1 design/spec packet matches the current implemented toolset closely enough that a new prompt author would not be misled.
2. `tools/codex/V1_Bridge_Runner_Change_Spec.md` no longer reads like a pre-implementation bridge spec for a different runner.
3. The packet reflects that the V1 workflow now includes:
   - `run_prompt.py`
   - `review_run.py`
   - `check_queue_readiness.py`
   - `list_review_backlog.py`
4. The run-id collision suffix behavior is described accurately but briefly.
5. The cleanup stays comfortably reviewable and does not turn into a broader rewrite.
