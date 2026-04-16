# Task: Sweep the architecture document against the actual V1 implementation and prioritize remaining work

You are working in this repository.

Your task is to compare the current architecture/design documents against the actual V1 implementation and artifacts now present in the repo, then produce a short prioritized list of what remains to build or clean up.

## Important framing

This is an architecture-review and prioritization task.

Do NOT implement code in this pass.
Do NOT rewrite major documents in this pass.
Do NOT modify:

- `tools/codex/baby_run_prompt.py`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

Do NOT build new helpers in this pass.

Your job is to compare the intended architecture to the repo as it actually exists and decide what is left, what matters most, and what should wait.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

Read these recent notes:

- `notes/012_v1_pipeline_options_review__20260416_000819.md`
- `notes/014_queue_readiness_checker_build__20260416_002419.md`
- `notes/015_review_backlog_lister_build__20260416_010500.md`
- `notes/016_queue_and_backlog_helper_validation__20260416_003710.md`
- `notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md`

Also inspect current repo contents in:

- `codex_prompts/`
- `notes/`
- `tools/codex/`

## Goal

Determine:

1. what the architecture/design packet says should exist
2. what actually exists now
3. what still appears missing, stale, drifting, or deferred
4. what remaining work should be prioritized next

## Questions to answer

### 1. Architecture alignment
- Which parts of the intended V1 architecture are now implemented and operational?
- Which parts are only partially implemented?
- Which parts of the architecture doc are stale, misleading, or lagging the actual repo state?

### 2. Remaining work
Identify the realistic remaining work items that are still justified by the current repo state.

Prefer items such as:
- small usability gaps
- doc/spec drift cleanup
- lightweight validation/contract checks
- missing thin operational helpers
- one or two small pieces needed before using the system regularly for real work

Avoid speculative platform expansion.

### 3. Prioritization
Rank the remaining items in a short prioritized list.

For each item include:
- short name
- why it matters
- expected risk: low / medium / high
- expected payoff: low / medium / high
- recommended timing:
  - next
  - soon
  - later
  - explicitly defer

### 4. Practical stopping point
Based on the current repo state, answer this clearly:

- Is the V1 prompt workflow system now good enough to use for real work?
- If yes, what is the smallest remaining thing to clean up before using it heavily?
- If no, what single missing piece still blocks that?

## Required output artifacts

Create exactly two artifacts.

### Artifact 1
Create a review report at:

`notes/018_architecture_vs_actual_sweep__TIMESTAMP.md`

This report should include:

- short summary
- implemented vs intended comparison
- remaining-work list
- prioritized ranking
- judgment about whether V1 is ready for real work

### Artifact 2
Create a short recommendation note at:

`notes/018_prioritized_remaining_work__TIMESTAMP.md`

This note should contain only:

- the top prioritized remaining items
- which one should happen next
- what should explicitly wait

Keep it brief and decisive.

## Constraints

1. Use the exact file paths listed above.
2. Do not implement code in this pass.
3. Do not rewrite the architecture doc in this pass.
4. Do not produce a giant roadmap.
5. Prioritize bounded, evidence-based remaining work only.
6. Keep the result practical enough to drive the next useful step.

## Success criteria

This task is successful if:

- the report accurately compares the architecture packet to the actual repo state
- the remaining work list is short and grounded
- the prioritization is practical
- the result helps decide whether to keep polishing the tool or start using it for the real job that motivated it
