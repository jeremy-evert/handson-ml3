# 012 V1 Pipeline Options Review

## Short Summary Of Current Pipeline Maturity

The V1 pipeline is now a usable thin slice, not just a design packet.

It can:

- execute one prompt through `tools/codex/run_prompt.py`
- write one durable V1 execution record in `notes/`
- preserve execution facts separately from review facts
- accept manual review write-back through `tools/codex/review_run.py`

That means the core V1 artifact boundary is real and operationally demonstrable. The main remaining weakness is not missing architecture. It is missing operational support around repeated use: finding what needs review, deciding what is actually ready to run next, and keeping the docs aligned with the now-real implementation.

## What Is Working Now

### 1. Execution path

- Prompt resolution exists and is conservative. `run_prompt.py` preserves the same direct-path, `codex_prompts/`, and unique-prefix lookup behavior as the bridge runner baseline. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:31).
- Codex invocation is still thin and inspectable. The runner shells out through `codex exec -C <repo_root> --output-last-message <tempfile> -` and captures return code, final output, and stderr. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:74).
- Execution-record creation exists and matches the intended V1 section order. The runner writes header, execution facts, review facts, failure analysis, resource facts, prompt text, final output, and stderr into one markdown file. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:97).
- Resource and failure evidence capture exists in the actual record body. The runner records `return_code`, `elapsed_seconds`, output/stderr character counts, and full stderr text when present. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:144) and the failed record [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:8).
- The separation between execution outcome and acceptance outcome is implemented correctly. New records start at `review_status: UNREVIEWED` regardless of `EXECUTED` vs `EXECUTION_FAILED`. See [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:130).
- The failure path is demonstrated, not hypothetical. The failed smoke-test run preserved the read-only/session-start stderr cleanly while still producing a reviewable record. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).

### 2. Review path

- Manual review write-back exists as a separate thin companion tool. `review_run.py` updates an existing record in place rather than creating sidecars or replacing the record. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:57).
- Record structure preservation is strong. The helper validates section order and required field lines before writing, then replaces only the targeted review fields. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:122) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:140).
- The helper is consistent with the V1 review gate. It only allows `ACCEPTED` or `REJECTED`, auto-fills `reviewed_at_utc`, and limits failure-analysis updates to rejected runs. See [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:12) and [review_run.py](/data/git/handson-ml3/tools/codex/review_run.py:148).
- Review is operationally usable now. The accepted smoke-test record shows the exact intended lifecycle: executed record first, then in-place write-back to `ACCEPTED`. See [001_smoke_test_pipeline__20260415_234918.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_234918.md:16).

### 3. Workflow usability

- The thin V1 flow is practical for single reviewed runs:
  - run prompt
  - inspect record
  - apply manual review
- The repo already has evidence of both important paths:
  - successful execution plus accepted review
  - failed execution artifact with preserved stderr
- The implementation stayed within the intended V1 boundary and avoided premature platform growth.

## Seams And Gaps That Remain

### 1. Conservative queue progression is still architectural, not operational

The architecture says only an accepted reviewed run should release the next prompt, but there is no helper yet that actually answers:

- what is the latest run for prompt N
- whether it is still `UNREVIEWED`
- whether the next prompt is ready

This is the biggest repeated-use seam. The design expects it, but the current repo still leaves the decision manual.

### 2. Review is usable, but review discovery is manual

`review_run.py` can update a chosen record, but nothing helps a human find:

- all unreviewed records
- the latest record for a prompt
- the current review backlog

That becomes awkward as `notes/` grows.

### 3. Stable identity is mostly right, but not perfectly aligned with the current doc

The execution-record artifact says the stable V1 `run_id` is `<prompt_stem>__<started_at_utc>`, but `run_prompt.py` adds `__2`, `__3`, etc. on same-second collisions. See [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:37) versus [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py:59).

This is a small practical improvement in code, but it is still a spec mismatch.

### 4. Documentation drift exists

- `tools/codex/V1_Bridge_Runner_Change_Spec.md` still targets `baby_run_prompt.py` as the implementation target and still names the runner as `tools/codex/baby_run_prompt.py`. See [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:5), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:17), [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:156), and [V1_Bridge_Runner_Change_Spec.md](/data/git/handson-ml3/tools/codex/V1_Bridge_Runner_Change_Spec.md:252).
- The design packet now implicitly assumes a review write-back step, but the architecture document still describes conservative queue progression as part of the V1 shape without a concrete implementation artifact for that responsibility. See [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:161) and [Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:215).

### 5. Repeated-use confidence is under-supported

The runner and review helper are both small and clear, but there is no explicit fixture-based validation or script-level regression harness around:

- record section ordering
- same-record write-back safety
- rejection-field behavior
- collision behavior

The code is simple enough that this is not a correctness emergency, but for repeated operational use it is a credible next bounded improvement.

### 6. Rejected-run and retry linkage are still mostly placeholders

The record shape includes `retry_of_run_id` and manual failure-analysis fields, but the current tools do not help a reviewer connect a rejected run to a retry decision or derive the next retry target.

### 7. Environment fragility still appears in the repo history

The failed smoke-test run shows that Codex session startup can fail before task work begins. That is not a runner-design flaw, but it does mean the workflow still benefits from a small amount of environment-oriented diagnosis or runbook support when repeated use hits startup failures. See [001_smoke_test_pipeline__20260415_233343.md](/data/git/handson-ml3/notes/001_smoke_test_pipeline__20260415_233343.md:83).

## Documentation Alignment Assessment

Overall alignment is good at the architectural level and good enough at the artifact level.

What still drifts:

- implementation target naming drift: the bridge-runner spec still speaks in terms of `baby_run_prompt.py`, while the actual V1 runner is `run_prompt.py`
- runner identity drift: the spec examples still encode `tools/codex/baby_run_prompt.py` while the real records encode `tools/codex/run_prompt.py`
- stable-identity drift: the spec describes a strict no-suffix `run_id`, while code uses a pragmatic collision suffix

These are small mismatches, but they are no longer theoretical because the repo now contains real V1 records and a real review helper.

## Realistic Next Bounded Options

### Option 1. Add a queue-readiness checker

- What it would build or improve:
  - A small helper that determines whether the next prompt is ready based on the latest execution record and whether that record is `ACCEPTED`.
- Why it matters:
  - This closes the biggest gap between the architecture and current operational reality.
  - It makes the V1 rule "only accepted review releases the next prompt" actually inspectable in use rather than purely manual.
- Expected risk level: `low`
- Expected payoff level: `high`

### Option 2. Add a review backlog / unreviewed-run lister

- What it would build or improve:
  - A small helper that scans V1 execution records in `notes/` and lists runs still at `UNREVIEWED`, plus the latest record per prompt.
- Why it matters:
  - Review write-back already works, but finding what needs review is still manual.
  - This directly improves repeated operational use without broadening the workflow.
- Expected risk level: `low`
- Expected payoff level: `high`

### Option 3. Add lightweight contract validation for the V1 scripts

- What it would build or improve:
  - A small test or fixture harness around `run_prompt.py` and `review_run.py` that locks in the record shape, allowed review transitions, and in-place write-back behavior.
- Why it matters:
  - The workflow now depends on markdown field stability.
  - A small regression harness would reduce silent drift while keeping the implementation thin.
- Expected risk level: `low`
- Expected payoff level: `medium`

### Option 4. Align the V1 design docs to the implemented runner and helper

- What it would build or improve:
  - A cleanup pass on the architecture/spec packet so it reflects `run_prompt.py`, the existing review write-back helper, and the real collision rule.
- Why it matters:
  - The current doc drift is small but inspectable.
  - Fixing it would remove avoidable ambiguity before more workflow support is added.
- Expected risk level: `low`
- Expected payoff level: `medium`

### Option 5. Add rejected-run / retry-linkage support

- What it would build or improve:
  - A small helper or convention for linking a retry run to `retry_of_run_id` and surfacing the latest rejected run for a prompt.
- Why it matters:
  - The record model already reserves space for retries, but the workflow does not yet help humans use it consistently.
  - This would make failure analysis more actionable without introducing a larger retry engine.
- Expected risk level: `medium`
- Expected payoff level: `medium`

### Option 6. Add a minimal environment-diagnosis/runbook note or helper

- What it would build or improve:
  - A bounded way to capture or explain Codex startup failures such as read-only session initialization issues.
- Why it matters:
  - The repo contains real evidence of environment-level failure before task execution.
  - A small runbook or diagnostic helper could reduce confusion when failures are not prompt-related.
- Expected risk level: `low`
- Expected payoff level: `low`

## Assessment Of The Current V1 Flow

The current V1 flow is now operationally usable for intentional, reviewed single-step work. It is no longer just a design direction.

What still feels awkward for regular use:

- finding the latest relevant run by hand in `notes/`
- identifying what is awaiting review
- deciding whether the next prompt is actually released
- keeping the docs and record semantics in sync as the thin tools stabilize

What is missing if this is going to be used regularly:

- one small readiness/checking helper
- one small review-discovery helper
- one small validation or contract layer to protect the record format

That is enough to make the next step clear: the strongest next work is around operational support for repeated reviewed use, not workflow expansion.
