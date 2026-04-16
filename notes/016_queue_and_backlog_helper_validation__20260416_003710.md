# 016 Queue And Backlog Helper Validation

## Short Summary

I validated `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py` against the current prompt order in `codex_prompts/` and the current V1 execution records in `notes/`.

Both helpers behaved conservatively and consistently with the V1 design documents. The main visible usability issue is that the default queue-readiness result points back to prompt `002`, because prompts `002` through `010` do not have V1 execution records even though later prompts `011` through `015` do.

Current V1 record evidence in `notes/`:

- `notes/001_smoke_test_pipeline__20260415_233343.md`: `EXECUTION_FAILED`, `UNREVIEWED`
- `notes/001_smoke_test_pipeline__20260415_234918.md`: `EXECUTED`, `ACCEPTED`
- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`: `EXECUTED`, `UNREVIEWED`
- `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`: `EXECUTED`, `UNREVIEWED`
- `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`: `EXECUTED`, `UNREVIEWED`
- `notes/014_build_queue_readiness_checker__20260416_002319.md`: `EXECUTED`, `UNREVIEWED`
- `notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md`: `EXECUTED`, `UNREVIEWED`

## Queue-Readiness Validation Findings

Prompt ordering was correct. The helper discovered the current sequence `001` through `016` from `codex_prompts/` in numeric order.

The default run of `python3 tools/codex/check_queue_readiness.py` selected `codex_prompts/002_repo_inventory_and_status.md` as the target, with `001` as the previous prompt and `notes/001_smoke_test_pipeline__20260415_234918.md` as the latest prior V1 record. That is correct under the V1 rule set: the latest V1 run for `001` is `ACCEPTED`, and there is no V1 run evidence for `002`, so `002` is the first unreleased prompt in sequence.

The prompt-specific checks also matched the repo evidence:

- `--prompt 001`: `Ready: YES`, because the first prompt has no prior gate.
- `--prompt 002`: `Ready: YES`, because latest `001` is `ACCEPTED`.
- `--prompt 013`: `Ready: NO`, because latest `012` is `UNREVIEWED` in `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`.
- `--prompt 014`: `Ready: NO`, because latest `013` is `UNREVIEWED` in `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`.
- `--prompt 015`: `Ready: NO`, because latest `014` is `UNREVIEWED` in `notes/014_build_queue_readiness_checker__20260416_002319.md`.

This matches the V1 rule in `tools/codex/V1_Run_Review_Gate.md`: only an `ACCEPTED` latest run for the immediately previous prompt releases the next prompt.

Visible edge case from the current repo state:

- The default target being `002` is correct but potentially surprising, because later prompts `011` through `015` already have V1 execution records. The helper is correctly ignoring older non-V1 `__SUCCESS__` notes for prompts `002` through `010`, but that gap is not explained in the output.

## Review-Backlog Validation Findings

`python3 tools/codex/list_review_backlog.py` reported:

- `Discovered V1 execution records: 7`
- `Unreviewed records: 6`
- `Prompts with latest record: 6`

Those counts match the current repo evidence exactly: there are 7 V1 records total, 6 of them are still `UNREVIEWED`, and those 7 records cover 6 prompts because prompt `001` has two V1 records.

Its latest-per-prompt selection was also correct:

- For `001`, it chose `notes/001_smoke_test_pipeline__20260415_234918.md`, which is newer than the failed `001` run and has `review_status: ACCEPTED`.
- For `011` through `015`, each prompt has one V1 record and each latest record is `UNREVIEWED`.

Its `UNREVIEWED records` section also behaved correctly. It includes the older failed `001` run because that record is still literally `UNREVIEWED`, even though it is no longer the latest record for prompt `001`.

Its `Likely needs human review next` view is the right conservative reduction of the backlog for the current repo state. It lists only prompts whose latest record is `UNREVIEWED`, which means:

- `011`
- `012`
- `013`
- `014`
- `015`

That matches the actual latest-record evidence.

The `--unreviewed-only` mode also behaved consistently. It filtered the `Latest record per prompt` section down to latest records that are themselves `UNREVIEWED`, while leaving the full `UNREVIEWED records` section intact.

## Consistency Findings

The two helpers tell a coherent story.

- `check_queue_readiness.py` says the queue is only V1-ready up to prompt `001`, so the next ready prompt is `002`.
- `list_review_backlog.py` says the current latest records needing review are `011` through `015`.

Those outputs are not contradictory. They are answering different bounded questions:

- readiness asks whether the reviewed V1 queue may advance from the start of the prompt sequence
- backlog asks which existing V1 run records still await human review

The stale failed `001` record is the main point that could look inconsistent at first glance, but the helpers remain coherent:

- backlog includes it in the all-unreviewed list because it is still unreviewed
- backlog excludes it from "likely needs human review next" because latest `001` is already accepted
- readiness also ignores it for queue release because latest `001` is accepted

## One Recommended Next Improvement

The single smallest next improvement is to make `tools/codex/check_queue_readiness.py` print a short explanation when the default target is blocked by a missing V1 history gap, especially in a repo like this one where prompts `002` through `010` have legacy notes but no V1 execution records.

Why this should happen next:

- the helper is already making the correct decision
- the current repo state makes that correct decision easy to misread
- a one-line explanation would reduce confusion without changing queue policy or adding workflow complexity

What should explicitly wait:

- any change to queue rules
- any backlog prioritization policy
- any automatic bridging from legacy `__SUCCESS__` notes into V1 state
- any broader dashboard or workflow-engine work
