# 012 Top Three Next Options

## Ranked Top Three

### 1. Add a queue-readiness checker

Why it made the top three:

- It closes the most important remaining gap between the V1 architecture and the actual toolchain.
- The design packet says only `ACCEPTED` should release the next prompt, but the repo does not yet provide a concrete way to answer that question.
- It is bounded, low-risk, and directly improves repeated operational use.

### 2. Add a review backlog / unreviewed-run lister

Why it made the top three:

- Review write-back now works, but locating records that still need review is manual.
- This is the next most obvious operational friction once more records accumulate in `notes/`.
- It improves usability without expanding the workflow into a broader engine.

### 3. Add lightweight contract validation for `run_prompt.py` and `review_run.py`

Why it made the top three:

- The current V1 flow depends on exact markdown structure and field stability.
- A small regression harness would protect the record contract before more helpers are built on top of it.
- It is a clean bounded step that reduces accidental drift risk.

## Which One Should Happen Next

The next step should be: `Add a queue-readiness checker`.

Reason:

- It addresses the highest-value remaining seam in the current V1 workflow.
- It is directly demanded by the architecture and review-gate docs.
- It reduces the current ambiguity around whether the next prompt may proceed, which is the most important missing operational behavior.

## What Should Explicitly Wait

These should wait until after the top option:

- broader workflow engines or status systems
- retry orchestration beyond minimal linkage
- multi-module runner refactors
- dashboards or aggregated reporting
- broad platform expansion

These can also wait briefly behind the top option, but remain good bounded follow-ups:

- doc-alignment cleanup for `V1_Bridge_Runner_Change_Spec.md`
- rejected-run / retry-linkage support
- environment-focused diagnosis tooling

## Practical Recommendation

Do the next step in this order:

1. queue-readiness checker
2. review backlog / unreviewed-run lister
3. lightweight contract validation

That sequence best strengthens the current repo for repeated reviewed use while staying inside the existing V1 boundary.
