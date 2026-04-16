# 016 Next Improvement Recommendation

Recommended next bounded improvement:

- Add a short gap-explanation line to `tools/codex/check_queue_readiness.py` when the default target is blocked by missing V1 execution-record history for earlier prompts.

Why it should happen next:

- The helper is making the correct conservative decision now, but in the current repo state that decision is easy to misread because prompts `002` through `010` have legacy notes but no V1 execution records.
- A small output clarification would improve operator understanding without changing any queue logic.

What should explicitly wait:

- any change to the V1 release rule
- any migration or inference from legacy `__SUCCESS__` notes
- any backlog prioritization feature
- any broader queue engine or dashboard work
