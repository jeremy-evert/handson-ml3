# 018 Architecture Vs Actual Sweep

## Short Summary

The V1 prompt workflow is now real, not just specified. The repo has a working runner at `tools/codex/run_prompt.py`, an in-place review write-back helper at `tools/codex/review_run.py`, a queue-readiness checker at `tools/codex/check_queue_readiness.py`, and a review-backlog lister at `tools/codex/list_review_backlog.py`.

The main remaining work is no longer core architecture implementation. It is cleanup around doc/spec drift, lightweight contract protection for the markdown record format, and a small amount of operational tightening before heavier repeated use.

## Implemented Vs Intended

### Implemented and operational

- One bounded prompt run can be executed through `tools/codex/run_prompt.py`.
- One durable V1 execution record is written into `notes/` with execution facts, review facts, failure-analysis placeholders, resource facts, prompt text, final output, and stderr.
- Execution status and review status are cleanly separated. New records start at `review_status: UNREVIEWED`.
- Manual review write-back exists through `tools/codex/review_run.py` and updates the same record in place.
- Conservative queue progression is operational through `tools/codex/check_queue_readiness.py`.
  - Only `ACCEPTED` on the immediately previous prompt releases the next prompt.
- Review discovery is operational through `tools/codex/list_review_backlog.py`.
  - The repo currently shows 9 V1 execution records, 8 unreviewed records, and latest unreviewed records for prompts `011` through `017`.

### Partially implemented or narrower than the docs imply

- Retry linkage exists only as a placeholder field. `retry_of_run_id` is present in the artifact but there is no thin helper or established operating path for using it consistently.
- Validation exists as ad hoc note-based evidence, not as a durable contract test layer. The system currently depends on markdown shape stability across four scripts.
- The architecture packet describes the core V1 shape well, but it does not reflect the current operational helper set as clearly as the repo now warrants.

### Stale or lagging parts of the design packet

- `tools/codex/V1_Bridge_Runner_Change_Spec.md` is the clearest stale document.
  - It still targets `tools/codex/baby_run_prompt.py`.
  - It still names the runner field as `tools/codex/baby_run_prompt.py`.
  - It still frames review write-back and queue logic as not yet implemented, while the repo now has separate helpers for both.
- The execution-record spec says the stable V1 `run_id` is `<prompt_stem>__<started_at_utc>`, but `tools/codex/run_prompt.py` now adds same-second suffixes like `__2` when needed. The code is pragmatic; the spec lags.
- `tools/Codex_Prompt_Workflow_Architecture.md` is broadly right, but it now underspecifies actual repo reality. The practical V1 toolset is no longer just runner plus manual review conceptually; it includes explicit readiness and backlog helpers.

## Remaining Work

### 1. Align the design packet to the actual V1 toolset

Why it matters:
The repo has outgrown parts of the design packet. The next prompt or reviewer could still be steered by stale references to `baby_run_prompt.py`, by the old strict run-id rule, or by the older assumption that readiness/backlog support does not exist.

Evidence:
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- actual scripts in `tools/codex/`

### 2. Add lightweight contract checks for the V1 markdown record

Why it matters:
`run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py` all rely on the same markdown field and section conventions. That contract is now central to the workflow, but it is only protected by careful coding and prior notes, not by a small repeatable test layer.

Evidence:
- repeated required-section and required-field parsing logic in `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py`
- current validation is documented in notes rather than locked in as an executable check

### 3. Tighten operational guidance around legacy notes and the existing review backlog

Why it matters:
The tooling is correct, but current repo state is mixed:
- legacy `__SUCCESS__` notes still exist for older prompts
- latest V1 reviews for `011` through `017` are still `UNREVIEWED`

That is not a V1 architecture failure, but it does create friction when interpreting readiness and when deciding whether to trust the queue as a day-to-day operational surface.

Evidence:
- `tools/codex/check_queue_readiness.py` needed a special explanatory `Queue note:`
- `tools/codex/list_review_backlog.py` currently reports 8 unreviewed records

### 4. Defer retry-linkage tooling and richer queue semantics

Why it matters:
These are plausible future improvements, but they are not required to use the current V1 slice for real work. Building them now would expand the system before the present workflow has been used enough to justify it.

Evidence:
- `retry_of_run_id` remains unused in normal flow
- current V1 already covers run, record, review, backlog, and readiness

## Prioritized Ranking

1. `Doc/spec alignment cleanup`
Why: removes the most immediate source of incorrect future work and makes the architecture packet trustworthy again.
Expected risk: `low`
Expected payoff: `high`
Recommended timing: `next`

2. `Record-contract validation`
Why: protects the central markdown artifact contract shared by all current V1 tools.
Expected risk: `low`
Expected payoff: `high`
Recommended timing: `soon`

3. `Operational guidance cleanup for legacy notes and open review backlog`
Why: reduces day-to-day confusion without adding new platform surface.
Expected risk: `low`
Expected payoff: `medium`
Recommended timing: `soon`

4. `Retry-linkage helper or richer queue semantics`
Why: useful later, but not justified before the current V1 flow sees more real use.
Expected risk: `medium`
Expected payoff: `medium`
Recommended timing: `explicitly defer`

## Judgment

Yes. The V1 prompt workflow system is good enough to use for real work now, provided it is used in the intended conservative way:

- run one prompt
- inspect the record
- write back manual review
- use readiness/backlog helpers to decide what is next

The smallest remaining thing to clean up before using it heavily is doc/spec alignment, especially:

- update the bridge-runner spec to reflect `run_prompt.py`
- reflect that review write-back, readiness, and backlog helpers now exist
- align the run-id collision rule with actual behavior

That is the shortest path to making the current V1 implementation and the architecture packet agree on what the system actually is.
