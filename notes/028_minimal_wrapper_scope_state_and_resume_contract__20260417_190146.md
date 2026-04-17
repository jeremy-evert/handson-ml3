# Minimal Wrapper Scope, State, and Resume Contract

## Executive Summary
The smallest workable Stage 1 wrapper contract is: one wrapper run owns one explicit ordered notebook list for Stage 1 only, launches at most one notebook-treatment prompt at a time through `tools/codex/run_prompt.py`, stores only enough wrapper-local state to remember that bounded list and the current position, and always reconstructs execution/review truth from V1 records in `notes/` before deciding whether it may resume or advance.

## Why This Design Surface Comes Next

`notes/025_define_staged_notebook_wrapper_mvp_contract__20260417_175751.md` fixed the boundary but intentionally deferred the first concrete seam: how a Stage 1 wrapper can hold a narrow notebook scope, stop after each bounded run, and resume later without competing with the V1 record model. That seam now matters because the existing repo already has:

- one-run / one-record execution truth in `notes/` via `tools/codex/run_prompt.py`
- manual review write-back via `tools/codex/review_run.py`
- readiness logic that advances only on latest `ACCEPTED` review state via `tools/codex/check_queue_readiness.py`
- scanner-derived notebook treatment signals in `tools/notebook_enricher/notebook_scanner.py`

The next design step therefore is not orchestration breadth. It is the minimum contract that lets a wrapper target a bounded Stage 1 notebook slice and survive interruption while preserving the current stop-and-decide workflow.

## Proposed Scope Contract

One wrapper run is allowed to scope:

- Stage 1 only
- one explicit ordered notebook list
- no other stages
- no repo-wide autonomous traversal

The wrapper run scope is the smallest practical batch that still permits stop/resume. One notebook only would technically work, but it would make wrapper-local resume state almost pointless. A bounded ordered list is the smallest useful scope because it lets the wrapper remember "which notebook in this declared Stage 1 slice is next" without becoming a queue engine.

At wrapper start time, the wrapper must know:

- `stage_id`, fixed to Stage 1 for this MVP
- the explicit notebook path list for this run
- the deterministic notebook order for that list

The wrapper may derive at runtime:

- current Stage 1 treatment need from `tools/notebook_enricher/notebook_scanner.py`
- latest run and review status for a notebook by reading V1 records in `notes/`
- whether the wrapper may advance past the current notebook

The scope stays narrow enough for review and resume because:

- the notebook list is fixed for the life of one wrapper run
- ordering is explicit and deterministic
- the wrapper may only care about the next notebook in that list
- the wrapper stops after each bounded notebook run reaches the existing V1 review gate

This is a Stage 1-only contract for now. Later prompts may decide whether later stages reuse the same shape, but this note does not.

## Minimum Wrapper-Local State

Allowed wrapper-local state is limited to convenience data needed to resume the bounded Stage 1 slice:

- wrapper run id
- `stage_id`
- ordered notebook scope list
- scope fingerprint or equivalent immutable scope identity
- current notebook index or explicit next notebook path
- latest wrapper-launched Stage 1 run record path or run id for that notebook, if present

The wrapper may also cache scan-derived treatment status for convenience, but that cache is non-canonical and must be treated as disposable. On resume, Stage 1 need should be recomputed from the notebook scan rather than trusted from wrapper-local state, because notebook contents can change outside the wrapper.

Allowed wrapper-local fields are therefore:

- identity of the bounded wrapper run
- identity of the bounded notebook scope
- position within that scope
- a pointer to the last wrapper-attempted notebook/run

Disallowed wrapper-local fields are:

- execution status as authoritative truth
- review status as authoritative truth
- acceptance/progression truth
- durable "completed" notebook truth independent of `notes/`
- repo-wide backlog state
- retry policy state beyond a pointer to the latest relevant run
- any second status taxonomy such as `BLOCKED`, `DONE`, or `WAITING`

Canonical execution and review facts must remain in the V1 execution record instead:

- whether a bounded run happened
- when it happened
- what prompt ran
- whether it executed successfully
- whether it is `UNREVIEWED`, `ACCEPTED`, or `REJECTED`
- any reviewer-written summary or failure analysis

## Next-Notebook Selection Rule

Inside one bounded wrapper run, notebooks are considered strictly in the declared ordered list. The wrapper does not discover new notebooks mid-run and does not reprioritize.

Deterministic ordering rule:

- preserve the explicit order of the wrapper's input notebook list
- if the input mechanism later allows a set or glob, it must resolve once at wrapper start into a stable sorted list and persist that exact order in wrapper-local state

Next notebook rule:

1. Start at the first notebook in the declared ordered list.
2. For the current notebook, recompute Stage 1 need from the scanner.
3. Look up the latest Stage 1 wrapper-launched V1 record for that notebook.
4. Interpret that latest record only through the existing V1 gate:
- no relevant run record means "not yet run"
- latest relevant run with `review_status: UNREVIEWED` means "run but awaiting review"
- latest relevant run with `review_status: REJECTED` means "stopped on rejection"
- latest relevant run with `review_status: ACCEPTED` plus scanner showing no remaining Stage 1 need means "accepted enough to advance"
5. Advance only when the notebook is either:
- scan-clean for Stage 1 without needing a wrapper-launched run in this bounded slice, or
- backed by an `ACCEPTED` latest relevant run and no longer needs Stage 1 treatment

This avoids a second execution-truth system because "accepted enough to advance" is not stored as an independent wrapper status. It is derived from two things only:

- the latest relevant V1 review outcome in `notes/`
- current scanner evidence that the notebook no longer needs Stage 1 treatment

## Stop/Resume Contract

The wrapper must stop on any event that would otherwise require a human decision:

- after creating any new V1 execution record, because the new record begins `UNREVIEWED`
- when the latest relevant run for the current notebook is already `UNREVIEWED`
- when the latest relevant run for the current notebook is `REJECTED`
- when the wrapper reaches the end of the bounded notebook list

Information that must exist to allow resume:

- bounded ordered notebook scope
- `stage_id`
- current notebook position or equivalent next-target marker

Information that may be reconstructed from `notes/` on resume:

- whether the current notebook has any relevant prior Stage 1 wrapper-launched run
- the latest relevant run id / record path
- latest execution status
- latest review status
- whether progression is blocked by `UNREVIEWED` or `REJECTED`

Resume rule:

1. Reload wrapper-local scope and current position.
2. Re-scan the current notebook for Stage 1 need.
3. Re-read the latest relevant V1 record for that notebook from `notes/`.
4. Decide from those facts whether to stop, rerun nothing, or advance.

After resume, the wrapper is allowed to advance only when:

- there is no remaining Stage 1 need for the current notebook and the latest relevant accepted work, if any, is consistent with that, or
- the notebook needed no Stage 1 work in the first place

If the latest run is still `UNREVIEWED`, resume must stop immediately and surface that the wrapper is waiting on the existing manual review gate. It must not generate another prompt for that notebook and must not move to the next notebook.

If the latest run is `REJECTED`, resume must also stop immediately. The wrapper must preserve the current V1 stop-and-decide model and require an explicit human decision outside this contract before any retry or redesign occurs.

## Wrapper-Local Versus Canonical Truth

Wrapper-local convenience state:

- bounded wrapper run identity
- declared notebook scope
- declared notebook order
- current position / next-target pointer
- last relevant run pointer

Derived targeting state:

- current Stage 1 treatment need from `notebook_scanner.py`
- whether the wrapper may advance past the current notebook, derived from scanner output plus latest V1 review state

Canonical execution facts:

- everything written by `tools/codex/run_prompt.py` into the V1 record in `notes/`

Canonical review facts:

- everything written back by `tools/codex/review_run.py`, especially `review_status` and `review_summary`

Canonical readiness/progression truth:

- the existing V1 review gate semantics from `tools/codex/V1_Run_Review_Gate.md`
- specifically, only `ACCEPTED` may release progression, and `UNREVIEWED` / `REJECTED` stop the flow

Wrapper-local state may support resume, but it may not redefine whether a bounded run occurred, whether the latest run is accepted, or whether review is still pending.

## Guardrails Against Queue-Engine Drift

The wrapper must not:

- maintain a repo-wide work queue
- dynamically discover future notebook work after the bounded scope is set
- add new progression states beyond V1 review outcomes
- auto-skip human review
- auto-retry rejected work
- treat wrapper-local state as the canonical backlog or readiness source

The wrapper is therefore a runner-centered bounded-slice driver, not a scheduling system. Its scope is "remember my declared Stage 1 notebook slice and where I stopped," not "manage notebook work across the repository."

## Explicit Deferrals

This note intentionally defers:

- exact orchestration-loop behavior
- Stage 1 prompt-generation rules
- any Stage 2 or Stage 3 design
- notebook write contracts
- retry policy beyond current V1 review outcomes
- reporting or dashboard artifacts beyond the minimum wrapper-local resume support
- implementation choices in Python
- exact wrapper file format or CLI syntax
- exact rule for identifying which V1 records count as wrapper-launched Stage 1 runs for a notebook

That final item is deferred because this note defines the contract need for such linkage, but not the implementation mechanism.

## Recommended Follow-On Prompt

`Define the minimal Stage 1 wrapper run-to-record linkage and prompt identity contract`

That prompt should define how a wrapper-launched Stage 1 notebook run is identified in prompt files and V1 records so resume logic can reliably find the relevant latest run for one notebook without altering the V1 record format.
