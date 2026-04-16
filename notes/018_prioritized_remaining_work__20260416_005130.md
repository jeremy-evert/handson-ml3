# 018 Prioritized Remaining Work

Top remaining items:

1. `Doc/spec alignment cleanup`
- Do this next.
- The repo now has `run_prompt.py`, `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py`, but parts of the design packet still describe the older bridge-runner state and strict unsuffixed run ids.

2. `Lightweight record-contract validation`
- Do this soon.
- The V1 workflow now depends on one markdown record shape shared across four scripts. A small repeatable check is the main technical safeguard still missing.

3. `Operational guidance cleanup for legacy notes and the current unreviewed backlog`
- Do this soon.
- The system works, but mixed legacy notes plus unreviewed latest records still make regular use noisier than it needs to be.

What should explicitly wait:

- retry-linkage tooling
- richer queue states or scheduling
- broader runner refactors
- any platform-style expansion beyond the current V1 slice
