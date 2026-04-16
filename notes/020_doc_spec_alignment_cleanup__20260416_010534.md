# 020 Doc Spec Alignment Cleanup

## Scope

This pass stayed bounded to doc/spec alignment for the current V1 slice.

Updated files:

- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`

No runner or helper code was changed.

## What Was Aligned

### Active V1 runner naming

- Replaced stale bridge-runner references to `tools/codex/baby_run_prompt.py` with `tools/codex/run_prompt.py` in the edited packet.
- Updated the bridge-runner spec so it reads as an alignment document for the implemented runner instead of a pre-implementation change request for a different file.

### Implemented helper set

- The packet now reflects that the current V1 workflow includes:
  - `tools/codex/run_prompt.py`
  - `tools/codex/review_run.py`
  - `tools/codex/check_queue_readiness.py`
  - `tools/codex/list_review_backlog.py`
- Removed stale framing in the bridge spec that treated review write-back or review/backlog helper support as still pending.

### Run-id wording

- Kept the base V1 identity as `<prompt_stem>__<started_at_utc>`.
- Added the implementation-accurate note that `run_prompt.py` appends a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt.

## Validation

1. Confirmed the edited design/spec files now name `tools/codex/run_prompt.py` as the active V1 runner where applicable.
2. Confirmed the edited packet no longer implies review write-back, readiness checking, or backlog listing are future work.
3. Confirmed run-id wording in the edited packet now matches current code behavior: base identity first, collision suffix only when needed.
4. Ran a search pass across the edited design/spec files and found no remaining `baby_run_prompt.py` references.

## Notes

This was intentionally not a broader architecture rewrite.
It only brought the current V1 packet back into alignment with the implemented toolset and record behavior.
