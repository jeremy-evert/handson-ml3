# 031_build_stage1_wrapper_mvp_from_contracts__20260417_191833

- run_id: `031_build_stage1_wrapper_mvp_from_contracts__20260417_191833`
- prompt_file: `codex_prompts/031_build_stage1_wrapper_mvp_from_contracts.md`
- prompt_stem: `031_build_stage1_wrapper_mvp_from_contracts`
- started_at_utc: `20260417_191833`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260417_192056`
- runner: `tools/codex/run_prompt.py`
- return_code: `0`
- retry_of_run_id:

## Review Facts

- review_status: `UNREVIEWED`
- review_summary:
- reviewed_by:
- reviewed_at_utc:

## Failure Analysis

- failure_type:
- failure_symptom:
- likely_cause:
- recommended_next_action:

## Resource / Cost Facts

- elapsed_seconds: `142.850`
- final_output_char_count: `955`
- stderr_char_count: `189833`

## Prompt Text

```md
# Prompt: Build the Stage 1 wrapper MVP from the existing contracts

You are working inside this repository.

Your task is to implement the thinnest practical Stage 1 wrapper MVP that is justified by the existing design contracts.

This is an implementation prompt.

## Hard boundaries

Do NOT implement Stage 2.
Do NOT implement Stage 3.
Do NOT redesign `tools/codex/run_prompt.py`.
Do NOT change the V1 execution record format.
Do NOT build a general queue engine.
Do NOT add parallel execution.
Do NOT add dashboards or background daemons.
Do NOT make the wrapper auto-run generated prompts in this pass.
Do NOT modify any notebooks in this pass.

## Goal

Build a small Stage 1-only wrapper that:

- accepts an explicit ordered notebook list as input
- uses scanner-first logic to determine whether Stage 1 is needed for each notebook
- generates Stage 1 prompt files only for notebooks that need chapter-intro treatment
- skips notebooks that do not need Stage 1 treatment
- writes one small report describing what it generated, what it skipped, and why

This wrapper must stop at prompt generation.
It must not execute the generated Stage 1 prompts yet.

## Files to inspect first

Inspect at minimum:

- `notes/025_define_staged_notebook_wrapper_mvp_contract__20260417_175751.md`
- `notes/028_minimal_wrapper_scope_state_and_resume_contract__20260417_190146.md`
- `notes/030_define_stage1_prompt_generation_contract__20260417_191113.md`
- `tools/notebook_enricher/notebook_scanner.py`
- `tools/notebook_enricher/prompt_builder.py`
- `tools/codex/run_prompt.py`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Execution_Record_Artifact.md`

Also inspect the repository layout enough to confirm where notebooks live and where generated prompt artifacts should go.

## What to build

Build only the minimum needed to support this Stage 1 prompt-generation slice.

### Required deliverables

Create or update only the smallest reasonable set of files needed for this slice.

At minimum, the implementation should include:

1. A Stage 1 wrapper entry point under `tools/notebook_enricher/`
   - choose a practical filename consistent with the current repo style
   - this script should be the thin entry point for Stage 1 prompt generation

2. Any minimal helper changes needed inside `tools/notebook_enricher/`
   - only if they are actually required
   - prefer reusing existing scanner / prompt-builder logic as-is

3. One markdown report in `notes/` for this run
   - summarize:
     - target notebook list
     - which notebooks needed Stage 1 prompt generation
     - which notebooks were skipped
     - why each notebook was generated or skipped
     - what prompt files were created

## Required behavior

The wrapper MVP must do all of the following:

### 1. Input model
Support a narrow explicit notebook scope as input.

Use one of these approaches:
- positional notebook-path arguments
- or one explicit text/json file containing an ordered notebook list

Choose the simpler option.

The wrapper must not do repo-wide autonomous discovery in this pass.

### 2. Scanner-first behavior
For each notebook in the explicit ordered scope:

- run scanner logic first
- determine whether Stage 1 is needed
- determine the intro status
- determine whether the action should be insert, replace, or skip

Prompt generation must depend on scanner evidence, not ad hoc notebook guessing.

### 3. Prompt generation only
For notebooks that need Stage 1 treatment:

- generate exactly one Stage 1 prompt file per notebook
- use the existing prompt-builder patterns where practical
- keep the prompt bounded to one notebook and one Stage 1 decision

For notebooks that do not need Stage 1 treatment:

- do not generate a prompt
- record the skip in the report with the reason

### 4. Output location
Write generated prompt files into `codex_prompts/`.

Use a clear naming convention that keeps these generated Stage 1 prompts obviously distinct from the hand-authored design/build prompts already in the repo.

Do NOT overwrite existing hand-authored prompts.

Use a naming pattern that is deterministic and reviewable.

### 5. No auto-run
Do not invoke `tools/codex/run_prompt.py` on the generated prompt files in this pass.

This slice ends at “prompt files generated.”

### 6. Safety rules
Preserve all current boundaries:

- no notebook mutation in this wrapper pass
- no direct code-cell edits
- no Stage 2 or Stage 3 work
- no orchestration-loop breadth
- no wrapper-local redefinition of canonical execution/review truth in `notes/`

## Implementation guidance

Follow the existing contracts.

The implementation should reflect these already-set boundaries:

- Stage 1 first
- explicit ordered notebook scope
- runner-centered workflow remains intact
- scanner-derived evidence drives prompt generation
- Stage 1 prompt generation is one notebook / one intro decision / bounded action only

Prefer the smallest implementation that works.

Do not gold-plate this.

## Naming and reviewability

Generated prompt files should be easy for a human to inspect before running.

The report in `notes/` should make it easy to answer:

- what notebooks were requested
- what the wrapper decided
- what files were generated
- what still remains deferred

## Validation

Before finishing, do at least these checks:

1. Confirm the wrapper can accept an explicit ordered notebook scope.
2. Confirm it can scan notebooks and classify Stage 1 need.
3. Confirm it generates prompt files only for notebooks that need Stage 1.
4. Confirm it does not run those generated prompts.
5. Confirm the report clearly explains generated vs skipped notebooks.
6. Confirm no notebooks were modified during this wrapper build pass.

If practical, use a very small test scope such as one notebook needing treatment and one notebook not needing treatment.

## Success criteria

This prompt is successful only if ALL of the following are true:

1. There is now a thin runnable Stage 1 wrapper MVP under `tools/notebook_enricher/`.
2. It accepts an explicit ordered notebook scope.
3. It uses scanner-first logic.
4. It generates Stage 1 prompt files only for notebooks that need intro treatment.
5. It does not run generated prompts.
6. It does not modify notebooks.
7. It writes a small report in `notes/`.
8. The implementation stays narrow and does not drift into Stage 2, Stage 3, or orchestration breadth.

## Output rules

- Be concrete
- Reuse existing code where practical
- Keep the change set as small as possible
- Favor a thin working slice over a flexible framework
- At the end of your final response, print only the path to the note you created
```

## Codex Final Output

Implemented a thin Stage 1-only wrapper at [tools/notebook_enricher/stage1_prompt_wrapper.py](/home/jevert/git/handson-ml3/tools/notebook_enricher/stage1_prompt_wrapper.py). It accepts an explicit ordered notebook list as positional args, scans each notebook first, classifies the Stage 1 action as `insert`, `replace`, or `skip`, writes Stage 1 prompt files only for notebooks that need treatment, and stops there without invoking `tools/codex/run_prompt.py`.

I validated it on `05_support_vector_machines.ipynb` and `06_decision_trees.ipynb`. That produced [codex_prompts/generated_stage1__05_support_vector_machines.md](/home/jevert/git/handson-ml3/codex_prompts/generated_stage1__05_support_vector_machines.md) and skipped `06_decision_trees.ipynb` because the scanner classified its intro as substantive. `git diff --name-only -- '*.ipynb'` returned no changes, so no notebooks were modified.

notes/031_stage1_wrapper_mvp_report__20260417_000000.md

## Stderr

```text
Runner context:
- sandbox: workspace-write
- full_auto: False
- model: (default)
- profile: (default)
- json: False
- codex_command: codex exec -C /home/jevert/git/handson-ml3 --sandbox workspace-write --output-last-message /tmp/codex-last-message-wmlq0hfj.txt -
OpenAI Codex v0.120.0 (research preview)
--------
workdir: /home/jevert/git/handson-ml3
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, /home/jevert/.codex/memories]
reasoning effort: none
reasoning summaries: none
session id: 019d9ce1-767d-7d01-ad01-8d011e7b127c
--------
user
# Prompt: Build the Stage 1 wrapper MVP from the existing contracts

You are working inside this repository.

Your task is to implement the thinnest practical Stage 1 wrapper MVP that is justified by the existing design contracts.

This is an implementation prompt.

## Hard boundaries

Do NOT implement Stage 2.
Do NOT implement Stage 3.
Do NOT redesign `tools/codex/run_prompt.py`.
Do NOT change the V1 execution record format.
Do NOT build a general queue engine.
Do NOT add parallel execution.
Do NOT add dashboards or background daemons.
Do NOT make the wrapper auto-run generated prompts in this pass.
Do NOT modify any notebooks in this pass.

## Goal

Build a small Stage 1-only wrapper that:

- accepts an explicit ordered notebook list as input
- uses scanner-first logic to determine whether Stage 1 is needed for each notebook
- generates Stage 1 prompt files only for notebooks that need chapter-intro treatment
- skips notebooks that do not need Stage 1 treatment
- writes one small report describing what it generated, what it skipped, and why

This wrapper must stop at prompt generation.
It must not execute the generated Stage 1 prompts yet.

## Files to inspect first

Inspect at minimum:

- `notes/025_define_staged_notebook_wrapper_mvp_contract__20260417_175751.md`
- `notes/028_minimal_wrapper_scope_state_and_resume_contract__20260417_190146.md`
- `notes/030_define_stage1_prompt_generation_contract__20260417_191113.md`
- `tools/notebook_enricher/notebook_scanner.py`
- `tools/notebook_enricher/prompt_builder.py`
- `tools/codex/run_prompt.py`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Execution_Record_Artifact.md`

Also inspect the repository layout enough to confirm where notebooks live and where generated prompt artifacts should go.

## What to build

Build only the minimum needed to support this Stage 1 prompt-generation slice.

### Required deliverables

Create or update only the smallest reasonable set of files needed for this slice.

At minimum, the implementation should include:

1. A Stage 1 wrapper entry point under `tools/notebook_enricher/`
   - choose a practical filename consistent with the current repo style
   - this script should be the thin entry point for Stage 1 prompt generation

2. Any minimal helper changes needed inside `tools/notebook_enricher/`
   - only if they are actually required
   - prefer reusing existing scanner / prompt-builder logic as-is

3. One markdown report in `notes/` for this run
   - summarize:
     - target notebook list
     - which notebooks needed Stage 1 prompt generation
     - which notebooks were skipped
     - why each notebook was generated or skipped
     - what prompt files were created

## Required behavior

The wrapper MVP must do all of the following:

### 1. Input model
Support a narrow explicit notebook scope as input.

Use one of these approaches:
- positional notebook-path arguments
- or one explicit text/json file containing an ordered notebook list

Choose the simpler option.

The wrapper must not do repo-wide autonomous discovery in this pass.

### 2. Scanner-first behavior
For each notebook in the explicit ordered scope:

- run scanner logic first
- determine whether Stage 1 is needed
- determine the intro status
- determine whether the action should be insert, replace, or skip

Prompt generation must depend on scanner evidence, not ad hoc notebook guessing.

### 3. Prompt generation only
For notebooks that need Stage 1 treatment:

- generate exactly one Stage 1 prompt file per notebook
- use the existing prompt-builder patterns where practical
- keep the prompt bounded to one notebook and one Stage 1 decision

For notebooks that do not need Stage 1 treatment:

- do not generate a prompt
- record the skip in the report with the reason

### 4. Output location
Write generated prompt files into `codex_prompts/`.

Use a clear naming convention that keeps these generated Stage 1 prompts obviously distinct from the hand-authored design/build prompts already in the repo.

Do NOT overwrite existing hand-authored prompts.

Use a naming pattern that is deterministic and reviewable.

### 5. No auto-run
Do not invoke `tools/codex/run_prompt.py` on the generated prompt files in this pass.

This slice ends at “prompt files generated.”

### 6. Safety rules
Preserve all current boundaries:

- no notebook mutation in this wrapper pass
- no direct code-cell edits
- no Stage 2 or Stage 3 work
- no orchestration-loop breadth
- no wrapper-local redefinition of canonical execution/review truth in `notes/`

## Implementation guidance

Follow the existing contracts.

The implementation should reflect these already-set boundaries:

- Stage 1 first
- explicit ordered notebook scope
- runner-centered workflow remains intact
- scanner-derived evidence drives prompt generation
- Stage 1 prompt generation is one notebook / one intro decision / bounded action only

Prefer the smallest implementation that works.

Do not gold-plate this.

## Naming and reviewability

Generated prompt files should be easy for a human to inspect before running.

The report in `notes/` should make it easy to answer:

- what notebooks were requested
- what the wrapper decided
- what files were generated
- what still remains deferred

## Validation

Before finishing, do at least these checks:

1. Confirm the wrapper can accept an explicit ordered notebook scope.
2. Confirm it can scan notebooks and classify Stage 1 need.
3. Confirm it generates prompt files only for notebooks that need Stage 1.
4. Confirm it does not run those generated prompts.
5. Confirm the report clearly explains generated vs skipped notebooks.
6. Confirm no notebooks were modified during this wrapper build pass.

If practical, use a very small test scope such as one notebook needing treatment and one notebook not needing treatment.

## Success criteria

This prompt is successful only if ALL of the following are true:

1. There is now a thin runnable Stage 1 wrapper MVP under `tools/notebook_enricher/`.
2. It accepts an explicit ordered notebook scope.
3. It uses scanner-first logic.
4. It generates Stage 1 prompt files only for notebooks that need intro treatment.
5. It does not run generated prompts.
6. It does not modify notebooks.
7. It writes a small report in `notes/`.
8. The implementation stays narrow and does not drift into Stage 2, Stage 3, or orchestration breadth.

## Output rules

- Be concrete
- Reuse existing code where practical
- Keep the change set as small as possible
- Favor a thin working slice over a flexible framework
- At the end of your final response, print only the path to the note you created

codex
Implementing the Stage 1-only wrapper slice now. I’m starting by reading the stated contracts and the existing scanner/prompt-builder/runner code, then I’ll keep the change set constrained to a thin prompt-generation entry point and one run report.
exec
/bin/bash -lc "sed -n '1,240p' notes/025_define_staged_notebook_wrapper_mvp_contract__20260417_175751.md" in /home/jevert/git/handson-ml3
exec
/bin/bash -lc "sed -n '1,240p' notes/028_minimal_wrapper_scope_state_and_resume_contract__20260417_190146.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
# Staged Notebook Wrapper MVP Contract

## Executive Summary
The staged notebook wrapper exists to break notebook-treatment work into bounded Codex-sized runs without weakening the current V1 runner, record, and review workflow. The MVP thin slice is a Stage 1 only wrapper that can select a narrow notebook scope, determine whether each notebook needs a chapter-intro treatment, and launch one bounded intro-treatment run at a time through `tools/codex/run_prompt.py`, stopping at the existing review gate after each run.

## Problem This Wrapper Solves

The current repo has two useful but disconnected pieces:

- a conservative runner-centered workflow built around `tools/codex/run_prompt.py`, `notes/` V1 execution records, `tools/codex/review_run.py`, `tools/codex/check_queue_readiness.py`, and `tools/codex/list_review_backlog.py`
- notebook-treatment experiments in `tools/notebook_enricher/notebook_scanner.py` and `tools/notebook_enricher/prompt_builder.py` that show how narrow notebook mutations can be targeted

What is missing is a thin repo-specific bridge that can turn notebook-treatment intent into repeated bounded runs without collapsing back into one giant notebook-enrichment prompt or inventing a second workflow platform.

In this repo, the wrapper exists to solve a narrow coordination problem:

- select a bounded notebook-treatment target
- use scanner-first evidence to decide whether treatment is needed
- emit one narrow prompt for one bounded notebook-treatment step
- run that prompt through the existing runner
- stop for human review before any further progression

The wrapper does not exist to replace the runner, redesign the queue, or create a broader notebook-processing system.

## MVP Thin Slice

The MVP is allowed to do exactly one meaningful thin slice first:

- Stage 1 only
- chapter-intro detection and treatment only
- one bounded notebook-treatment run at a time
- runner-mediated execution only

For the first MVP pass, the wrapper may:

- accept a narrow notebook scope using a later-defined minimal input contract
- inspect candidate notebooks with scanner-first logic before mutation
- determine whether a notebook is missing or has a non-substantive chapter intro
- generate a bounded Stage 1 prompt for exactly one notebook at a time
- launch that prompt through `tools/codex/run_prompt.py`
- stop after each run and require the normal V1 review decision before any subsequent wrapper-driven run

The MVP is not allowed to start with all three stages, all notebooks in one pass, or an autonomous multi-step workflow. The first slice is deliberately a Stage 1 launcher around existing V1 mechanics.

## Required Invariants

The wrapper must preserve these repo-specific invariants:

- The workflow remains runner-centered. The wrapper is subordinate to `tools/codex/run_prompt.py`, not an alternative execution path.
- Bounded work still goes through `tools/codex/run_prompt.py`. The wrapper must not bypass it with direct `codex exec` calls for normal bounded notebook-treatment runs.
- The V1 execution record in `notes/` remains the canonical artifact for each bounded run. One wrapper-launched run still means one V1 execution record markdown file.
- The wrapper must not change the V1 execution record format defined by `tools/codex/V1_Execution_Record_Artifact.md` and validated by `tools/codex/v1_record_validation.py`.
- The existing review gate remains in force. `UNREVIEWED`, `ACCEPTED`, and `REJECTED` keep their current meaning, and only `ACCEPTED` may release the next bounded step.
- The wrapper must not become a second source of truth for queue progression. `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py` already derive progression and backlog status from the V1 records in `notes/`; the wrapper must not supersede that with wrapper-local queue state.
- Review write-back remains manual through the existing model embodied by `tools/codex/review_run.py`.
- Any wrapper-local artifacts introduced later may support targeting or resume behavior, but they must not redefine whether a run happened, whether it was reviewed, or whether the next prompt is ready.

## Mandatory Safety Rules

The wrapper MVP must enforce these safety rules:

- No direct code-cell edits.
- No broad notebook rewrites framed as enrichment.
- Only deterministic, bounded insert-or-replace notebook mutations are allowed.
- Scanner-first targeting is mandatory before any mutation prompt is generated.
- Mutation targeting must be derived from concrete notebook positions, not vague chapter-wide instructions.
- The wrapper may only launch prompts that are narrow enough for a reviewer to verify against one notebook and one treatment goal.
- No silent automatic progression past review boundaries. A completed run must stop at `UNREVIEWED` until a human applies the existing review gate.
- No mutation prompt may authorize opportunistic cleanup, style passes, notebook reformatting, output regeneration, metadata churn, or unrelated edits.
- For the Stage 1 MVP slice, allowed mutations are limited to a single chapter-intro insert or replace action for the targeted notebook.

These safety rules follow directly from the notebook scanner and prompt-builder experiments in `tools/notebook_enricher/`, which already assume cell-adjacent, deterministic treatment decisions rather than open-ended notebook rewriting.

## Explicit Deferrals

This contract pass intentionally does not decide the following. Later prompts must define them:

- the concrete notebook scope or selection config format
- the concrete progress-state structure for wrapper-local resume support
- the stop/resume file schema
- the detailed scan or report artifact shape
- the stage-specific prompt-generation rules and prompt text contracts
- the detailed orchestration-loop behavior
- the exact wrapper CLI surface
- the concrete write path for wrapper-generated prompt files or wrapper-local notes
- Stage 2 rules for markdown-before-code treatment
- Stage 3 rules for markdown-after-code treatment
- retry behavior beyond the existing V1 rejected-run stop-and-decide model

This note sets boundaries and deferrals only. It does not settle the concrete design of those deferred surfaces.

## Non-Goals

The staged notebook wrapper MVP is not:

- a redesign of `tools/codex/run_prompt.py`
- a redesign of the V1 execution record
- a new queue engine
- a richer queue-state model
- retry orchestration
- a dashboard or reporting UI
- a background daemon or watcher
- a parallel execution system
- an automatic reviewer or approval system
- a general notebook-refactoring platform
- a repo-wide enrichment pass that treats all stages as one unit

If a proposed design requires any of the above to make the MVP feel complete, that design is too broad for this pass.

## Why This Contract Comes Before Concrete Design

This repo already favors boundaries before breadth and thin slices before larger automation, as stated in `tools/Project_Design_Workflow.md` and `tools/Codex_Prompt_Workflow_Architecture.md`. The current V1 workflow is deliberately small: one run, one record, one review gate, one conservative readiness rule. A staged notebook wrapper can fit into that workflow cleanly only if the boundary is settled before implementation details start expanding.

This contract comes first to prevent three common failure modes:

- treating the wrapper as a replacement execution path instead of a thin layer around `tools/codex/run_prompt.py`
- letting wrapper-local progress artifacts become a second source of truth for queue progression
- overdefining stop/resume, scan artifacts, and orchestration behavior before the MVP thin slice is proven

By fixing the contract first, later prompts can define the minimum concrete scope/progress/resume surface without accidentally redesigning the existing V1 workflow.

## Recommended Next Prompt

The next prompt should be:

- `Build the minimal Stage 1 wrapper scope/progress/resume contract`

That prompt should define only the smallest concrete contract for:

- notebook scope input
- wrapper-local progress tracking sufficient for stop/resume
- the minimum resume boundary that does not compete with V1 execution records or review status

It should not yet define full orchestration behavior, Stage 2 or Stage 3 treatment rules, or any broader wrapper platform concerns.

 succeeded in 0ms:
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

exec
/bin/bash -lc "sed -n '1,260p' notes/030_define_stage1_prompt_generation_contract__20260417_191113.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
# Stage 1 Prompt-Generation Contract

## Executive Summary
Stage 1 prompt generation is a narrow scanner-driven contract for one chapter-intro treatment decision on one notebook. A Stage 1 prompt may be generated only after `tools/notebook_enricher/notebook_scanner.py` has already determined that Stage 1 is needed, classified the current intro state, and identified the exact eligible insert-or-replace position. The prompt itself stays bounded to one notebook mutation, preserves notebook validity, and remains subordinate to the existing runner-centered V1 workflow.

## Why This Design Surface Comes Next

`notes/025_define_staged_notebook_wrapper_mvp_contract__20260417_175751.md` fixed the Stage 1-first MVP boundary, and `notes/028_minimal_wrapper_scope_state_and_resume_contract__20260417_190146.md` fixed the minimal scope/state/resume seam. The next unresolved surface is narrower: what concrete facts must exist before a Stage 1 prompt is even allowed to exist.

This comes next because the repo already separates:

- execution and review truth, which remain in V1 records under `notes/`
- scanner-derived notebook evidence, which already exists in `tools/notebook_enricher/notebook_scanner.py`
- prompt execution, which remains runner-centered around `tools/codex/run_prompt.py`

Without a Stage 1 prompt-generation contract, later build work would be free to let prompt generation drift into open-ended notebook discovery, broad notebook rewriting, or wrapper-local redefinition of canonical truth. This note prevents that drift before any implementation starts.

## Required Inputs For One Stage 1 Prompt

One Stage 1 prompt may be generated only when these minimum concrete inputs are already available for the targeted notebook:

- the target notebook path
- the fixed active stage identity, which must be Stage 1 and nothing else
- the scanner-derived `chapter_intro_status`
- the scanner-derived `chapter_intro_index`
- the scanner-derived `setup_cell_index`
- the scanner-derived Stage 1 need decision
- the scanner-derived intro action decision: `insert`, `replace`, or `skip`
- enough notebook identity context to keep the prompt bounded to one notebook, specifically the notebook stem and total cell count or equivalent bounded notebook-size context already available from the scanner inventory

For this repo, those inputs should align with the existing `NotebookInventory` surface in `tools/notebook_enricher/notebook_scanner.py`, which already provides:

- `path`
- `notebook_stem`
- `total_cells`
- `chapter_intro_status`
- `chapter_intro_index`
- `setup_cell_index`
- `needs_stage1()`

Stage 1 prompt generation does not require CLI syntax, wrapper state format, or Python type decisions in this pass. It requires only the concrete targeting facts above.

## Scanner Preconditions

Stage 1 prompt generation is allowed to assume scanner-first logic has already determined all notebook-discovery questions needed for the Stage 1 decision. Prompt generation is not the place where notebook discovery happens.

Before a Stage 1 prompt may be generated, the scanner must already determine:

- whether Stage 1 is needed at all
- whether the existing chapter intro is `missing`, `heading`, `thin`, or `substantive`
- whether the eligible Stage 1 action is `insert`, `replace`, or `skip`
- what exact notebook position is eligible for the Stage 1 change
- whether the notebook has a `# Setup` boundary that constrains where an inserted intro may go

For this repo, the scanner evidence is sufficient when it yields these conclusions:

- `chapter_intro_status: substantive` means no Stage 1 prompt should be generated
- `chapter_intro_status: missing` means insertion is eligible at the scanner-approved intro position before the setup boundary when present
- `chapter_intro_status: heading` means replacement is eligible only for the scanner-identified intro cell
- `chapter_intro_status: thin` means replacement is eligible only for the scanner-identified intro cell

Evidence is sufficient to avoid generating a Stage 1 prompt when either of these is true:

- the scanner says Stage 1 is not needed because the intro is already `substantive`
- the scanner cannot determine a single eligible bounded insert-or-replace target position, in which case Stage 1 prompt generation must stop rather than broaden the task

The prompt generator must not perform open-ended reasoning such as:

- reinterpreting whether another markdown cell elsewhere might be a better intro candidate
- deciding to rewrite multiple front-matter cells
- discovering Stage 2 or Stage 3 work while preparing a Stage 1 prompt

## Allowed Stage 1 Target Unit

The smallest practical Stage 1 target unit is:

- exactly one notebook
- exactly one Stage 1 chapter-intro treatment decision
- exactly one bounded intro mutation opportunity within that notebook

One Stage 1 prompt must not become:

- a broad notebook rewrite
- a front-matter cleanup pass
- a mixed Stage 1 plus Stage 2 or Stage 3 treatment prompt
- a notebook-wide style or narrative improvement pass

For this contract, one prompt targets one notebook and resolves only one of these outcomes:

- insert one new chapter-intro markdown cell
- replace one existing intro markdown cell
- no prompt at all because Stage 1 is unnecessary or not safely targetable

## Allowed Intro Decisions

Stage 1 allows only these intro decisions:

- `insert` when the scanner classifies the intro as `missing` and identifies a single eligible insertion position near the front of the notebook, bounded by the current intro-detection window and the setup boundary
- `replace` when the scanner classifies the existing intro as `heading` or `thin` and identifies the exact existing intro cell index eligible for replacement
- `skip` or no prompt when the scanner classifies the intro as `substantive`
- `skip` or no prompt when scanner evidence is not strong enough to support one deterministic bounded insert-or-replace action

Stage 1 does not authorize:

- replacing more than one cell
- inserting more than one new markdown cell
- rewriting neighboring markdown cells for consistency
- cleaning up title cells, notebook-description cells, or HTML link tables
- moving the setup cell
- reformatting markdown outside the single targeted intro treatment

This means Stage 1 is a chapter-intro decision only. It is not a license for opportunistic cleanup before, around, or after that decision.

## Stage 1 Prompt Output Contract

Every Stage 1 prompt generated under this contract must enforce the same bounded output shape.

The prompt must require:

- one bounded notebook mutation only
- reading the full source notebook artifact for the targeted notebook
- writing the complete modified notebook artifact, not partial notebook fragments
- preserving valid Jupyter notebook structure
- preserving all non-targeted cells in place
- preserving code-cell source, outputs, and metadata unchanged
- keeping the mutation reviewable as one bounded V1 step

The prompt must prohibit:

- direct code-cell edits
- unrelated markdown rewrites
- unrelated cell insertion or deletion
- cell reordering
- notebook-wide metadata churn
- broad front-matter rewrites
- Stage 2 or Stage 3 edits

At a contract level, the prompt may direct output to a temporary modified notebook artifact rather than in-place mutation, but this note does not define the implementation path. What is canonical here is that the Stage 1 prompt must require a complete modified notebook artifact that is valid and reviewable as one bounded run result.

## Mandatory Notebook-Safety Rules

These notebook-safety rules are non-negotiable for Stage 1 prompt generation in this repo:

- no direct code-cell edits
- no broad notebook rewrites
- no Stage 2 or Stage 3 edits
- no opportunistic style cleanup
- deterministic targeting before mutation
- bounded insert-or-replace action only
- no silent progression past the V1 review gate
- no mutation prompt when scanner evidence does not support one exact bounded target
- no metadata churn unrelated to the one Stage 1 intro treatment

The safety boundary is therefore:

- scanner first
- one notebook
- one intro decision
- one bounded mutation
- one V1-reviewed run

This keeps Stage 1 compatible with the current review gate in `tools/codex/V1_Run_Review_Gate.md` and the single-run execution-record model in `tools/codex/V1_Execution_Record_Artifact.md`.

## Local Prompt-Generation Facts Versus Canonical Truth

Stage 1 prompt generation may use wrapper-local targeting facts as convenience inputs. For example, wrapper-local state may remember:

- which notebook is currently being considered
- that the active stage is Stage 1
- the scanner-derived intro status and target position used to generate the prompt

But those local prompt-generation facts are not canonical execution truth.

What must remain canonical in `notes/` is:

- whether a Stage 1 run actually happened
- what prompt was executed
- what execution outcome was recorded
- what review decision was recorded
- whether the run remains `UNREVIEWED`, was `ACCEPTED`, or was `REJECTED`

Stage 1 prompt generation must not:

- redefine queue readiness
- replace the V1 review gate with wrapper-local state
- treat prompt-generation facts as proof that execution or review occurred
- compete with `tools/codex/check_queue_readiness.py` or `tools/codex/list_review_backlog.py` as the source of progression truth

The design remains runner-centered:

- prompt generation may be wrapper-local
- execution and review truth remain canonical in V1 records under `notes/`
- Stage 1 prompt generation stays subordinate to `tools/codex/run_prompt.py` and the existing review workflow

## Explicit Deferrals

This note intentionally does not define:

- Stage 2 prompt-generation rules
- Stage 3 prompt-generation rules
- wrapper orchestration behavior
- the orchestration loop
- run-to-record linkage mechanics
- wrapper-local file formats
- CLI syntax
- Python implementation details
- how prompt files are named or stored
- notebook-write implementation code

This pass defines only the Stage 1 prompt-generation contract surface and its hard boundaries.

## Recommended Follow-On Prompt

`Build the minimal Stage 1 prompt-generation implementation around scanner output and runner-centered execution`

That follow-on should implement only this contract surface without expanding into Stage 2, Stage 3, orchestration redesign, or V1 record changes.

exec
/bin/bash -lc "sed -n '1,260p' tools/notebook_enricher/notebook_scanner.py" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
"""
notebook_scanner.py — Cell inventory and treatment-state detection.

Reads a .ipynb file and classifies every code cell's treatment state:
  - has_goal_before  : 'canonical', 'legacy', 'thin', 'missing'
  - has_impl_after   : 'canonical', 'legacy', 'thin', 'missing'

Also detects the chapter intro status and the Setup cell position.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Detection vocabulary
# ---------------------------------------------------------------------------

GOAL_CANONICAL = "### Goal Before This Cell"
IMPL_CANONICAL = "### Implementation Notes After This Cell"

GOAL_LEGACY_KEYWORDS = [
    "**Why run this cell",
    "**Goal:",
    "**What it is",
    "**Principle:",
]

IMPL_LEGACY_KEYWORDS = [
    "**Result**:",
    "**Why this matters",
]

# A cell whose source starts with `# Setup` marks the boundary for intro detection.
SETUP_MARKERS = ("# setup", "## setup")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CellState:
    index: int
    cell_type: str          # 'code' or 'markdown'
    is_empty: bool          # True if source is whitespace only
    has_goal_before: str    # 'canonical', 'legacy', 'thin', 'missing'
    has_impl_after: str     # 'canonical', 'legacy', 'thin', 'missing'
    goal_cell_index: int    # index of preceding markdown cell, or -1
    impl_cell_index: int    # index of following markdown cell, or -1

    def needs_goal(self) -> bool:
        return self.has_goal_before in ("missing", "legacy", "thin") and not self.is_empty

    def needs_impl(self) -> bool:
        return self.has_impl_after in ("missing", "legacy", "thin") and not self.is_empty


@dataclass
class NotebookInventory:
    path: Path
    notebook_stem: str
    total_cells: int
    code_cell_states: list[CellState]   # one entry per code cell
    chapter_intro_status: str           # 'substantive', 'thin', 'heading', 'missing'
    chapter_intro_index: int            # cell index of existing intro, -1 if missing
    setup_cell_index: int               # index of the '# Setup' cell, -1 if not found

    def needs_stage1(self) -> bool:
        """Stage 1 is needed when chapter intro is not already substantive."""
        return self.chapter_intro_status != "substantive"

    def cells_needing_goal(self) -> list[CellState]:
        return [cs for cs in self.code_cell_states if cs.needs_goal()]

    def cells_needing_impl(self) -> list[CellState]:
        return [cs for cs in self.code_cell_states if cs.needs_impl()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cell_source(cell: dict) -> str:
    """Return the full source text of a notebook cell."""
    return "".join(cell.get("source", []))


def word_count(text: str) -> int:
    return len(text.split())


def _classify_goal(src: str) -> str:
    if GOAL_CANONICAL in src:
        return "canonical"
    if any(kw in src for kw in GOAL_LEGACY_KEYWORDS):
        return "legacy"
    # Anything else (section header, book narrative, etc.) is 'thin'
    return "thin"


def _classify_impl(src: str) -> str:
    if IMPL_CANONICAL in src:
        return "canonical"
    if any(kw in src for kw in IMPL_LEGACY_KEYWORDS):
        return "legacy"
    # Short bold-starting cell counts as legacy per spec
    if src.strip().startswith("**") and word_count(src) < 100:
        return "legacy"
    return "thin"


# ---------------------------------------------------------------------------
# Chapter intro detection
# ---------------------------------------------------------------------------

def _is_setup_cell(src: str) -> bool:
    stripped = src.strip().lower()
    return any(stripped.startswith(m) for m in SETUP_MARKERS)


def _is_prose_candidate(cell: dict, src: str) -> bool:
    """True if this markdown cell could be the chapter intro (not a heading-only or HTML cell)."""
    if cell.get("cell_type") != "markdown":
        return False
    stripped = src.strip()
    # Skip the bold-title cell like **Chapter 7 – ...**
    if stripped.startswith("**") and "\n" not in stripped:
        return False
    # Skip the italic notebook-description line
    if stripped.startswith("_This notebook contains") and "\n" not in stripped:
        return False
    # Skip HTML tables (colab/kaggle links)
    if stripped.startswith("<table"):
        return False
    return True


def detect_chapter_intro(cells: list[dict]) -> tuple[str, int, int]:
    """
    Scan cells 0-5 (or up to # Setup) and identify the chapter intro.

    Returns:
        (status, intro_index, setup_index)

    status values: 'substantive', 'thin', 'heading', 'missing'
    intro_index: cell index of the intro, or -1 if missing
    setup_index: cell index of the # Setup cell, or -1 if not found
    """
    setup_index = -1
    for i, cell in enumerate(cells):
        if _is_setup_cell(cell_source(cell)):
            setup_index = i
            break

    upper_bound = min(6, setup_index if setup_index != -1 else len(cells))

    best_index = -1
    best_wc = -1
    for i in range(upper_bound):
        cell = cells[i]
        src = cell_source(cell)
        if not _is_prose_candidate(cell, src):
            continue
        wc = word_count(src)
        if wc > best_wc:
            best_wc = wc
            best_index = i

    if best_index == -1 or best_wc < 10:
        return "missing", -1, setup_index

    if best_wc > 200:
        status = "substantive"
    elif best_wc >= 50:
        status = "thin"
    else:
        status = "heading"

    return status, best_index, setup_index


# ---------------------------------------------------------------------------
# Main scan function
# ---------------------------------------------------------------------------

def scan_notebook(path: Path) -> NotebookInventory:
    """Read a notebook and return a full cell inventory."""
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    intro_status, intro_index, setup_index = detect_chapter_intro(cells)

    code_cell_states: list[CellState] = []

    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        src = cell_source(cell)
        is_empty = not src.strip()

        # Classify goal_before
        goal_cell_idx = -1
        if i > 0 and cells[i - 1].get("cell_type") == "markdown":
            goal_cell_idx = i - 1
            goal_status = _classify_goal(cell_source(cells[i - 1]))
        else:
            goal_status = "missing"

        # Classify impl_after
        impl_cell_idx = -1
        if i < len(cells) - 1 and cells[i + 1].get("cell_type") == "markdown":
            impl_cell_idx = i + 1
            impl_status = _classify_impl(cell_source(cells[i + 1]))
        else:
            impl_status = "missing"

        code_cell_states.append(CellState(
            index=i,
            cell_type="code",
            is_empty=is_empty,
            has_goal_before=goal_status,
            has_impl_after=impl_status,
            goal_cell_index=goal_cell_idx,
            impl_cell_index=impl_cell_idx,
        ))

    return NotebookInventory(
        path=path,
        notebook_stem=path.stem,
        total_cells=len(cells),
        code_cell_states=code_cell_states,
        chapter_intro_status=intro_status,
        chapter_intro_index=intro_index,
        setup_cell_index=setup_index,
    )


def format_inventory_report(inv: NotebookInventory) -> str:
    """Human-readable summary of a notebook scan result."""
    lines = [
        f"Notebook: {inv.notebook_stem}",
        f"Total cells: {inv.total_cells}",
        f"Code cells: {len(inv.code_cell_states)}",
        f"Chapter intro: {inv.chapter_intro_status} (cell {inv.chapter_intro_index})",
        f"Setup cell: {inv.setup_cell_index}",
        "",
        f"Stage 1 needed: {inv.needs_stage1()}",
        f"Cells needing goal (Stage 2): {len(inv.cells_needing_goal())}",
        f"Cells needing impl (Stage 3): {len(inv.cells_needing_impl())}",
        "",
        "Code cell detail:",
    ]
    for cs in inv.code_cell_states:
        if cs.is_empty:

exec
/bin/bash -lc "sed -n '1,260p' tools/notebook_enricher/prompt_builder.py" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
"""
prompt_builder.py — Generates per-job prompt files for codex exec.

Each (notebook, stage) pair gets a focused prompt that tells codex exactly:
- Which notebook to modify
- What stage is being run
- Which cells need treatment (from scanner output)
- The treatment specification
- A gold standard example from NB06
- The output contract (write to .tmp, don't touch code cells)
"""

from __future__ import annotations

import json
from pathlib import Path

from notebook_scanner import NotebookInventory, cell_source


TREATMENT_SPEC_SOURCE = "prompts/001_assess_and_finish_05_06_07.md"
GOLD_STANDARD_NOTEBOOK = "06_decision_trees.ipynb"
ACTIVE_PROMPT_NAME = "active_prompt"  # used as prompt_stem base


def _repo_root(this_file: Path) -> Path:
    return this_file.resolve().parents[2]


def _load_treatment_spec(root: Path) -> str:
    path = root / TREATMENT_SPEC_SOURCE
    if not path.exists():
        return "[Treatment spec not found — see prompts/001_assess_and_finish_05_06_07.md]"
    text = path.read_text(encoding="utf-8")
    # Extract just the treatment template section
    start = text.find("#### Treatment structure")
    end = text.find("---", start + 10) if start != -1 else -1
    if start != -1 and end != -1:
        return text[start:end].strip()
    # Fallback: return the whole thing
    return text


def _load_gold_standard(root: Path) -> str:
    """Pull the chapter intro + one Goal/Code/Impl trio from NB06."""
    path = root / GOLD_STANDARD_NOTEBOOK
    if not path.exists():
        return "[Gold standard notebook not found]"

    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]

    # Find the chapter intro cell (cell 3 in NB06 after prompt 001 treatment)
    intro_text = ""
    for cell in cells[:6]:
        src = cell_source(cell)
        if "## Chapter Overview" in src or "Chapter Overview" in src:
            intro_text = src[:800] + ("..." if len(src) > 800 else "")
            break

    # Find one Goal/Code/Impl trio
    trio_text = ""
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        if not src.strip():
            continue
        goal_src = cell_source(cells[i - 1]) if i > 0 and cells[i - 1].get("cell_type") == "markdown" else ""
        impl_src = cell_source(cells[i + 1]) if i < len(cells) - 1 and cells[i + 1].get("cell_type") == "markdown" else ""
        if "### Goal Before This Cell" in goal_src and "### Implementation Notes After This Cell" in impl_src:
            trio_text = (
                f"### GOAL BEFORE CELL:\n{goal_src[:600]}\n\n"
                f"### CODE CELL:\n```python\n{src[:400]}\n```\n\n"
                f"### IMPLEMENTATION NOTES AFTER CELL:\n{impl_src[:600]}"
            )
            break

    parts = []
    if intro_text:
        parts.append("**Chapter Intro Example:**\n" + intro_text)
    if trio_text:
        parts.append("**Goal/Code/Implementation Trio Example:**\n" + trio_text)
    return "\n\n".join(parts) if parts else "[No gold standard found]"


def _cell_source_snippet(src: str, max_chars: int = 300) -> str:
    src = src.strip()
    if len(src) <= max_chars:
        return src
    return src[:max_chars] + "..."


def build_stage1_prompt(root: Path, inventory: NotebookInventory) -> str:
    """Generate the stage 1 prompt: add/replace the chapter intro."""
    nb_path = inventory.path.relative_to(root).as_posix()
    tmp_path = str(inventory.path.with_suffix(".ipynb.tmp").name)
    chapter_num = inventory.notebook_stem.split("_")[0].lstrip("0") or "?"

    with open(inventory.path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]

    # Describe current intro state
    intro_idx = inventory.chapter_intro_index
    if intro_idx == -1:
        intro_description = "MISSING — no chapter intro exists. Insert a new one."
        insert_instruction = (
            f"INSERT a new markdown cell at index {max(3, inventory.setup_cell_index - 1 if inventory.setup_cell_index > 0 else 3)}. "
            "Place it after the Colab/Kaggle links table and before the # Setup cell."
        )
        current_content = "(no existing intro)"
    else:
        current_src = cell_source(cells[intro_idx])
        intro_description = f"{inventory.chapter_intro_status.upper()} at cell index {intro_idx} (~{len(current_src.split())} words)."
        insert_instruction = f"REPLACE the markdown cell at index {intro_idx} with the full chapter intro."
        current_content = _cell_source_snippet(current_src)

    treatment_spec = _load_treatment_spec(root)
    gold_standard = _load_gold_standard(root)

    return f"""# Stage 1: Chapter Intro Enrichment

## Target Notebook
- Path: {nb_path}
- Chapter: {chapter_num}
- Notebook stem: {inventory.notebook_stem}

## Current Chapter Intro State
Status: {intro_description}

Current content (if any):
```
{current_content}
```

## Your Task
{insert_instruction}

The new intro must follow the treatment specification below exactly.

## Treatment Specification

{treatment_spec}

## Gold Standard Example (from 06_decision_trees.ipynb — the finished notebook)

{gold_standard}

## Output Contract

1. Read the full notebook from: `{nb_path}`
2. {insert_instruction}
3. Write the COMPLETE modified notebook as valid JSON to: `{tmp_path}`
   (write it to the same directory as the original notebook)
4. Do NOT modify any `code` cells — not their source, outputs, or metadata
5. Do NOT add, remove, or reorder any cell other than the one change described above
6. The new intro cell must have `"cell_type": "markdown"` and a `"source"` field

## Hard Constraints

- NEVER modify a cell with `"cell_type": "code"`
- The intro must be 300–500 words covering: what this chapter is about, 3–5 learning
  objectives, why this topic matters in ML, where it fits in the course, key vocabulary
- Write clean JSON — the same structure as the input notebook
- The output file must be a valid Jupyter notebook that can be opened in Jupyter Lab
"""


def build_stage2_prompt(root: Path, inventory: NotebookInventory) -> str:
    """Generate the stage 2 prompt: add Goal Before cells."""
    nb_path = inventory.path.relative_to(root).as_posix()
    tmp_path = str(inventory.path.with_suffix(".ipynb.tmp").name)

    cells_needing = inventory.cells_needing_goal()
    if not cells_needing:
        return f"# Stage 2: No Goal Cells Needed\n\nNotebook {nb_path} already has goal cells for all code cells. No action required."

    treatment_spec = _load_treatment_spec(root)
    gold_standard = _load_gold_standard(root)

    cell_list = []
    with open(inventory.path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]

    for cs in cells_needing:
        src = cell_source(cells[cs.index])
        snippet = _cell_source_snippet(src, 200)
        action = "REPLACE" if cs.goal_cell_index != -1 else "INSERT BEFORE"
        cell_list.append(
            f"- Code cell index {cs.index} (current goal status: {cs.has_goal_before}): "
            f"{action} goal cell\n  Code preview: `{snippet[:100]}`"
        )

    cell_list_str = "\n".join(cell_list)

    return f"""# Stage 2: Goal-Before-Cell Enrichment

## Target Notebook
- Path: {nb_path}

## Cells Needing Goal Treatment ({len(cells_needing)} total)

{cell_list_str}

## Treatment Specification

{treatment_spec}

## Gold Standard Example

{gold_standard}

## Rules Per Cell

For each code cell in the list above:
- If action is INSERT BEFORE: add a new markdown cell immediately before the code cell
- If action is REPLACE: replace the immediately-preceding markdown cell with a canonical goal cell
- The new cell must start with `### Goal Before This Cell`
- Then 4–8 sentences: goal, why it matters for ML, ⚙️ Plumbing / ✨ Better practice labels

## Output Contract

1. Read the full notebook from: `{nb_path}`
2. Apply all the cell insertions/replacements listed above
3. Write the COMPLETE modified notebook as valid JSON to: `{tmp_path}`
4. Do NOT modify any `code` cells
5. Do NOT touch any cell not in the list above

## Hard Constraints

- NEVER modify a cell with `"cell_type": "code"`
- Every new goal cell must start with `### Goal Before This Cell`
- Maintain correct cell ordering — inserted cells go immediately BEFORE the target code cell
"""


def build_stage3_prompt(root: Path, inventory: NotebookInventory) -> str:
    """Generate the stage 3 prompt: add Implementation Notes cells."""
    nb_path = inventory.path.relative_to(root).as_posix()
    tmp_path = str(inventory.path.with_suffix(".ipynb.tmp").name)

    cells_needing = inventory.cells_needing_impl()
    if not cells_needing:
        return f"# Stage 3: No Impl Cells Needed\n\nNotebook {nb_path} already has implementation notes for all code cells. No action required."

    treatment_spec = _load_treatment_spec(root)
    gold_standard = _load_gold_standard(root)

    cell_list = []
    with open(inventory.path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]

    for cs in cells_needing:
        src = cell_source(cells[cs.index])
        snippet = _cell_source_snippet(src, 200)
        action = "REPLACE" if cs.impl_cell_index != -1 else "INSERT AFTER"

exec
/bin/bash -lc "sed -n '1,260p' tools/codex/run_prompt.py" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from v1_record_validation import validate_record_text


PROMPTS_DIR = "codex_prompts"
NOTES_DIR = "notes"
RUNNER_PATH = "tools/codex/run_prompt.py"
VALID_SANDBOXES = {"read-only", "workspace-write", "danger-full-access"}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def utc_timestamp(moment: dt.datetime) -> str:
    return moment.strftime("%Y%m%d_%H%M%S")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def fail(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
    prompt = Path(prompt_arg)

    if prompt.is_absolute():
        return prompt

    direct = root / prompt
    if direct.exists():
        return direct

    in_prompts = root / PROMPTS_DIR / prompt
    if in_prompts.exists():
        return in_prompts

    matches = sorted((root / PROMPTS_DIR).glob(f"{prompt_arg}*"))
    if len(matches) == 1:
        return matches[0]

    return in_prompts


def prompt_file_label(root: Path, prompt_path: Path) -> str:
    try:
        return prompt_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(prompt_path.resolve())


def build_record_path(notes_dir: Path, prompt_stem: str, started_at_utc: str) -> tuple[str, Path]:
    base_run_id = f"{prompt_stem}__{started_at_utc}"
    candidate = notes_dir / f"{base_run_id}.md"
    if not candidate.exists():
        return base_run_id, candidate

    suffix = 2
    while True:
        run_id = f"{base_run_id}__{suffix}"
        candidate = notes_dir / f"{run_id}.md"
        if not candidate.exists():
            return run_id, candidate
        suffix += 1


def fenced_block(text: str, fence: str, info: str) -> str:
    body = text.rstrip("\n")
    return f"{fence}{info}\n{body}\n{fence}" if body else f"{fence}{info}\n\n{fence}"


def build_codex_command(
    *,
    root: Path,
    output_path: Path,
    sandbox: str,
    full_auto: bool,
    model: str | None,
    profile: str | None,
    use_json: bool,
) -> list[str]:
    command = ["codex", "exec", "-C", str(root)]

    # full-auto is the Codex low-friction preset. If explicitly requested,
    # prefer it over a manual sandbox flag.
    if full_auto:
        command.append("--full-auto")
    else:
        command.extend(["--sandbox", sandbox])

    if model:
        command.extend(["--model", model])

    if profile:
        command.extend(["--profile", profile])

    if use_json:
        command.append("--json")

    command.extend(["--output-last-message", str(output_path), "-"])
    return command


def format_runner_context(
    *,
    command: list[str],
    sandbox: str,
    full_auto: bool,
    model: str | None,
    profile: str | None,
    use_json: bool,
) -> str:
    context_lines = [
        "Runner context:",
        f"- sandbox: {sandbox}",
        f"- full_auto: {full_auto}",
        f"- model: {model or '(default)'}",
        f"- profile: {profile or '(default)'}",
        f"- json: {use_json}",
        f"- codex_command: {shlex.join(command)}",
        "",
    ]
    return "\n".join(context_lines)


def run_codex(
    *,
    root: Path,
    prompt_text: str,
    sandbox: str,
    full_auto: bool,
    model: str | None,
    profile: str | None,
    use_json: bool,
) -> tuple[int, str, str]:
    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
        output_path = Path(handle.name)

    command = build_codex_command(
        root=root,
        output_path=output_path,
        sandbox=sandbox,
        full_auto=full_auto,
        model=model,
        profile=profile,
        use_json=use_json,
    )

    context_prefix = format_runner_context(
        command=command,
        sandbox=sandbox,
        full_auto=full_auto,
        model=model,
        profile=profile,
        use_json=use_json,
    )

    try:
        result = subprocess.run(
            command,
            input=prompt_text,
            text=True,
            capture_output=True,
            check=False,
        )
        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        stderr_text = context_prefix + (result.stderr or "")
        return result.returncode, final_output, stderr_text
    except FileNotFoundError as exc:
        stderr_text = context_prefix + f"codex executable not found: {exc}\n"
        return 127, "", stderr_text
    finally:
        output_path.unlink(missing_ok=True)


def build_record_content(
    *,
    run_id: str,
    prompt_file: str,
    prompt_stem: str,
    started_at_utc: str,
    execution_status: str,
    finished_at_utc: str,
    return_code: int,
    prompt_text: str,
    codex_output: str,
    stderr_text: str,
    elapsed_seconds: float,
) -> str:
    final_output = codex_output.rstrip()
    stderr_body = stderr_text.rstrip()

    sections = [
        f"# {run_id}",
        "",
        f"- run_id: `{run_id}`",
        f"- prompt_file: `{prompt_file}`",
        f"- prompt_stem: `{prompt_stem}`",
        f"- started_at_utc: `{started_at_utc}`",
        "",
        "## Execution Facts",
        "",
        f"- execution_status: `{execution_status}`",
        f"- finished_at_utc: `{finished_at_utc}`",
        f"- runner: `{RUNNER_PATH}`",
        f"- return_code: `{return_code}`",
        "- retry_of_run_id:",
        "",
        "## Review Facts",
        "",
        "- review_status: `UNREVIEWED`",
        "- review_summary:",
        "- reviewed_by:",
        "- reviewed_at_utc:",
        "",
        "## Failure Analysis",
        "",
        "- failure_type:",
        "- failure_symptom:",
        "- likely_cause:",
        "- recommended_next_action:",
        "",
        "## Resource / Cost Facts",
        "",
        f"- elapsed_seconds: `{elapsed_seconds:.3f}`",
        f"- final_output_char_count: `{len(codex_output)}`",
        f"- stderr_char_count: `{len(stderr_text)}`",
        "",
        "## Prompt Text",
        "",
        fenced_block(prompt_text, "```", "md"),
        "",
        "## Codex Final Output",
        "",
        final_output if final_output else "*No output captured.*",
        "",

codex
The contracts are aligned: scanner evidence drives a one-notebook Stage 1 decision, and this slice should stop before `run_prompt.py`. I’m checking the repo layout and the exact V1 docs next so the wrapper names and report stay consistent with the existing workflow rather than inventing new conventions.
exec
/bin/bash -lc "sed -n '1,220p' tools/codex/V1_Run_Review_Gate.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
# V1 Run Review Gate

## Purpose

Define the smallest practical review gate between one prompt run finishing and the next prompt in sequence being treated as ready.

This gate exists to preserve three things from the governing workflow:

- review between iterations
- separation of execution outcome from accepted outcome
- a conservative human stop before queue progression

## Scope

V1 covers only the decision point immediately after one run record is written.

It defines:

- when a run enters review
- the minimum manual checks
- the allowed review outcomes
- how those outcomes affect whether the next prompt may proceed
- what review information must be written into the run record

It does not define:

- automation of review
- a larger queue state machine
- dependency-aware scheduling
- retry orchestration
- richer review taxonomies such as `PARTIAL` or `BLOCKED`

## Review Trigger

A run becomes `awaiting review` when its execution phase has finished and its V1 execution record has been written to `notes/` with:

- `execution_status` set to either `EXECUTED` or `EXECUTION_FAILED`
- `review_status` still set to `UNREVIEWED`

That is the exact review trigger for V1.

Process exit alone does not make the next prompt ready.
Record creation plus `UNREVIEWED` review status means the run must stop for human inspection.

## Minimum Manual Checklist

Before a run can be accepted, a human reviewer must check only these items:

1. Confirm the execution record is complete enough to review.
   Required minimum: prompt identity, execution status, return code, prompt text, Codex final output, and any captured stderr.

2. Check whether the run actually addressed the prompt that was executed.
   This is a scope-and-intent check, not a deep redesign review.

3. Check whether there is enough evidence to treat the result as an accepted outcome for the current bounded step.
   Minimum evidence may be direct output inspection, referenced file changes, or stated validation evidence in the run record or surrounding repo state.

4. Check whether any obvious failure, mismatch, or ambiguity remains that would make automatic progression unsafe.
   Examples: execution failed, output is incomplete, acceptance is unclear, validation is missing, or the result suggests the design/task should be refined before continuing.

This checklist is intentionally small.
V1 only needs enough manual review to keep execution success separate from accepted progress.

## Allowed V1 Review Outcomes

V1 should allow exactly these review outcomes:

- `UNREVIEWED`
- `ACCEPTED`
- `REJECTED`

Meaning:

- `UNREVIEWED`: default state after execution record creation; queue must stop here.
- `ACCEPTED`: the bounded step is accepted after human review.
- `REJECTED`: the run is not accepted for progression, whether because execution failed, the result is inadequate, or the outcome requires redesign or retry analysis first.

No additional V1 outcomes are needed.

## Queue Progression Rule

Only `ACCEPTED` allows the next prompt in sequence to be treated as ready.

Rules:

- `UNREVIEWED` stops the queue pending manual review.
- `ACCEPTED` releases exactly the next reviewed step.
- `REJECTED` stops the queue and forces an explicit human decision about what happens next.

`EXECUTED` does not release the queue.
`EXECUTION_FAILED` does not itself decide the queue either, but it will usually lead to `REJECTED` on review.

## Stop-And-Decide Rule

`REJECTED` is the V1 outcome that stops the queue and forces a new design or retry decision.

That decision is intentionally outside this gate.
It may result in:

- a revised prompt
- a smaller follow-up slice
- an environment fix
- a deliberate retry
- an architecture or scope adjustment

V1 only requires that the queue does not continue past a rejected run.

## Run Record Write-Back

During review, the run record defined by [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md) must be updated with these review facts:

- `review_status`
- `review_summary`

When available, V1 should also write:

- `reviewed_by`
- `reviewed_at_utc`

For rejected runs, the reviewer should also fill the small failure-analysis fields when useful:

- `failure_type`
- `failure_symptom`
- `likely_cause`
- `recommended_next_action`

This keeps execution facts and review judgment in one durable record.

## Connection To The Execution Record

This gate depends on the separation already defined in [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:77):

- execution status answers "what happened when the runner executed the prompt?"
- review status answers "did a human accept this bounded step as good enough to progress?"

The V1 review gate is therefore not a new artifact.
It is the manual transition of one execution record from `UNREVIEWED` to either `ACCEPTED` or `REJECTED`.

## Intentionally Deferred

V1 intentionally defers:

- automatic queue release
- automatic reviewer assignment
- separate queue-level state files
- richer review outcomes such as `PARTIAL`, `BLOCKED`, or `NEEDS_RETRY`
- policy for choosing among multiple retry or redesign options
- automatic extraction of validation evidence from repo changes
- broader workflow redesign beyond the next reviewed step

## V1 Decision Summary

The minimum V1 review gate is:

1. execution finishes
2. a run record is written with `review_status: UNREVIEWED`
3. human review applies the minimum checklist
4. reviewer writes back `ACCEPTED` or `REJECTED`
5. only `ACCEPTED` makes the next prompt ready

That is small enough to guide the next implementation prompt without expanding the system into a larger workflow engine.

exec
/bin/bash -lc "sed -n '1,240p' tools/codex/V1_Execution_Record_Artifact.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
# V1 Execution Record Artifact

## Purpose

Define the minimum durable record for one prompt run so the workflow preserves:

- what was executed
- what happened during execution
- what a human later decided about the result
- enough failure and resource evidence to guide the next reviewed step

This artifact is meant to support the current conservative workflow, not a larger automation system.

## Scope

V1 covers one prompt run from prompt submission through later human review.

It does not define:

- multi-run reporting
- dependency graphs
- approval automation
- module decomposition
- a full state machine

## Source Of Truth

For V1, the source of truth for one run is one markdown execution-record file in `notes/`.

The record body is authoritative.
The filename is a locator and stable identifier carrier, but the body is the place where execution facts and review fields must be read from.

V1 should avoid splitting one run across multiple required files.

## Stable Identity For A Run

Each run must have a stable `run_id` that does not change after the file is created.

V1 format:

`<prompt_stem>__<started_at_utc>`

Example:

`005_define_execution_record_and_generate_next_prompt_queue__20260415_202557`

Why this is enough for V1:

- it is human-readable
- it distinguishes repeated runs of the same prompt
- it does not depend on later review outcome
- it can be created cheaply by the bridge runner

`tools/codex/run_prompt.py` uses that base identity and adds a numeric suffix such as `__2` only when needed to avoid a same-second filename collision for the same prompt.

The filename should use the same stable identity and should not encode accepted or rejected review outcome.

## Required Fields

Every V1 execution record should contain these fields.

### Run Identity

- `run_id`
- `prompt_file`
- `prompt_stem`
- `started_at_utc`

### Execution Facts

- `execution_status`
- `finished_at_utc`
- `runner`
- `return_code`
- `prompt_text`
- `codex_final_output`

### Review Facts

- `review_status`
- `review_summary`

### Minimum Failure / Retry Linkage

- `retry_of_run_id`

### Minimum Resource / Cost Facts

- `elapsed_seconds`
- `final_output_char_count`
- `stderr_char_count`

## Optional Fields

These fields are useful when available, but should not block V1.

- `reviewed_by`
- `reviewed_at_utc`
- `stderr_text`
- `failure_analysis`
- `recommended_next_action`
- `model_name`
- `token_usage`
- `attempt_index_for_prompt`
- `workspace_notes`

## Automatic Vs Manual Fields

### Automatic In V1

These should be captured automatically when the runner can do so cheaply and reliably:

- `run_id`
- `prompt_file`
- `prompt_stem`
- `started_at_utc`
- `finished_at_utc`
- `execution_status`
- `runner`
- `return_code`
- `prompt_text`
- `codex_final_output`
- `elapsed_seconds`
- `final_output_char_count`
- `stderr_char_count`
- `stderr_text` when present
- `retry_of_run_id` only if the runner was explicitly given that context

### Manual In V1

These remain human-owned in V1:

- `review_status`
- `review_summary`
- `reviewed_by`
- `reviewed_at_utc`
- `failure_analysis`
- `recommended_next_action`

This split is intentional.
Execution capture should be automatic.
Acceptance and interpretation should remain manual until the review loop is defined more clearly.

## Execution Status Vs Review Status

V1 must keep these separate.

### Execution Status

This records what happened when the runner invoked Codex.

Allowed V1 values:

- `EXECUTED`
- `EXECUTION_FAILED`

`EXECUTED` means the subprocess completed successfully enough to return code `0`.
It does not mean the work is accepted.

`EXECUTION_FAILED` means the subprocess failed or the run artifact could only be completed with failure evidence.

### Review Status

This records the later human judgment.

Allowed V1 values:

- `UNREVIEWED`
- `ACCEPTED`
- `REJECTED`

Rules:

- a new run record should start as `UNREVIEWED`
- `EXECUTED` must not imply `ACCEPTED`
- `EXECUTION_FAILED` does not automatically imply `REJECTED`, though rejection will often follow
- the next prompt in a reviewed sequence should wait for explicit human review, not only process exit success

## Minimum Failure-Analysis Section

V1 failure analysis should stay small.
It should exist as a short section that can be filled in manually when useful, especially for `EXECUTION_FAILED` or `REJECTED` runs.

Minimum fields:

- `failure_type`
- `failure_symptom`
- `likely_cause`
- `recommended_next_action`

Suggested `failure_type` values:

- `environment`
- `runner`
- `prompt_scope`
- `prompt_clarity`
- `implementation`
- `validation`
- `unknown`

This is enough for V1 because it supports learning from failure without requiring a larger taxonomy.

## Minimum Resource / Cost Section

V1 should record only lightweight fields that are cheap to capture or inspect:

- `started_at_utc`
- `finished_at_utc`
- `elapsed_seconds`
- `return_code`
- `final_output_char_count`
- `stderr_char_count`

Optional later additions:

- `model_name`
- `token_usage`
- `tool_call_count`
- `file_change_count`
- `estimated_review_minutes`

V1 should observe resource use, not build a billing system.

## Practical Record Shape

A V1 markdown record should have these sections in this order:

1. Header / identity
2. Execution facts
3. Review facts
4. Failure analysis
5. Resource / cost facts
6. Prompt text
7. Codex final output
8. Stderr or supplemental notes

This keeps one run easy to inspect in a single file.

## Intentionally Deferred From V1


exec
/bin/bash -lc "rg --files -g '*.ipynb'" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
15_processing_sequences_using_rnns_and_cnns.ipynb
index.ipynb
07_ensemble_learning_and_random_forests.ipynb
extra_gradient_descent_comparison.ipynb
ch04/Gradient_Descent_From_Scratch.ipynb
16_nlp_with_rnns_and_attention.ipynb
extra_autodiff.ipynb
17_autoencoders_gans_and_diffusion_models.ipynb
Weather_Agreement_Lab/04_agreement_metrics.ipynb
Weather_Agreement_Lab/01_data_pull.ipynb
Weather_Agreement_Lab/02_alignment_and_cleaning.ipynb
Weather_Agreement_Lab/03_forecast_uncertainty.ipynb
tools_pandas.ipynb
Weather_Agreement_Lab/05_data_fusion.ipynb
tools_numpy.ipynb
math_differential_calculus.ipynb
11_training_deep_neural_networks.ipynb
math_linear_algebra.ipynb
bird_api_demo/bird_call_api_practice.ipynb
19_training_and_deploying_at_scale.ipynb
extra_ann_architectures.ipynb
03_classification.ipynb
13_loading_and_preprocessing_data.ipynb
01_the_machine_learning_landscape.ipynb
10_neural_nets_with_keras.ipynb
Gradient_descent.ipynb
05_support_vector_machines.ipynb
09_unsupervised_learning.ipynb
18_reinforcement_learning.ipynb
api_practice/api_practice.ipynb
08_dimensionality_reduction.ipynb
14_deep_computer_vision_with_cnns.ipynb
04_training_linear_models.ipynb
06_decision_trees.ipynb
tools_matplotlib.ipynb
12_custom_models_and_training_with_tensorflow.ipynb
02_end_to_end_machine_learning_project.ipynb

exec
/bin/bash -lc 'rg --files codex_prompts notes tools/notebook_enricher' in /home/jevert/git/handson-ml3
 succeeded in 0ms:
tools/notebook_enricher/notebook_scanner.py
tools/notebook_enricher/prompt_builder.py
notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md
notes/022_scaffolding_cleanup_plan__20260416_023016.md
notes/024_generate_staged_notebook_wrapper_prompts__20260417_172452.md
notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md
notes/022_repo_lay_of_the_land__20260416_105729.md
notes/027_write_028_scope_state_and_resume_contract_prompt__20260417_185855.md
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md
notes/019_next_two_cleanup_prompts_plan__20260416_010251.md
notes/030_define_stage1_prompt_generation_contract__20260417_191029.md
notes/016_next_improvement_recommendation__20260416_003710.md
notes/001_notebook_finish_report.md
notes/009_run_prompt_candidate_build__20260415_233407.md
codex_prompts/015_build_review_backlog_unreviewed_run_lister.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md
notes/030_define_stage1_prompt_generation_contract__20260417_191113.md
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md
codex_prompts/022_audit_and_classify_scaffolding_for_summary_and_attic.md
notes/011_review_writeback_helper_build__20260415_235514.md
codex_prompts/027_write_028_scope_state_and_resume_contract_prompt.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md
codex_prompts/024_generate_staged_notebook_wrapper_prompts.md
notes/026_define_staged_notebook_wrapper_mvp_contract__20260417_175702.md
codex_prompts/030_define_stage1_prompt_generation_contract.md
codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md
notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md
notes/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work__20260416_005034.md
codex_prompts/025_audit_025_mvp_contract_prompt_against_assessment.md
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md
codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md
notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md
codex_prompts/017_polish_queue_readiness_gap_explanation.md
codex_prompts/011_build_v1_review_writeback_helper.md
codex_prompts/002_repo_inventory_and_status.md
notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md
codex_prompts/029_write_030_stage1_prompt_generation_contract_prompt.md
notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md
codex_prompts/028_define_minimal_wrapper_scope_state_and_resume_contract.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md
codex_prompts/001_smoke_test_pipeline.md
codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md
notes/013_next_two_prompt_plan__20260416_002005.md
codex_prompts/014_build_queue_readiness_checker.md
notes/004_architecture_and_bridge_runner_review__20260415_195538.md
codex_prompts/021_add_lightweight_v1_record_contract_validation.md
codex_prompts/022_generate_repo_lay_of_the_land_summary.md
codex_prompts/026_define_staged_notebook_wrapper_mvp_contract.md
notes/021_add_lightweight_v1_record_contract_validation__20260416_010810.md
codex_prompts/019_generate_prompts_for_doc_alignment_and_record_contract_validation.md
notes/029_write_030_stage1_prompt_generation_contract_prompt__20260417_190640.md
codex_prompts/003_revise_Project_Deisgn_workflow_document.md
codex_prompts/023_tools_codex_assessment.md
codex_prompts/020_align_v1_doc_and_spec_packet_to_actual_toolset.md
codex_prompts/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work.md
codex_prompts/031_build_stage1_wrapper_mvp_from_contracts.md
codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md
codex_prompts/008_define_minimal_bridge_runner_change_spec.md
codex_prompts/007_align_architecture_doc_to_v1_workflow.md
codex_prompts/006_define_v1_run_review_gate.md
codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md
notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md
codex_prompts/009_build_v1_run_prompt_candidate.md
codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md
notes/019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209.md
notes/027_write_028_scope_state_and_resume_contract_prompt__20260417_120000.md
notes/014_queue_readiness_checker_build__20260416_002419.md
notes/028_minimal_wrapper_scope_state_and_resume_contract__20260417_190146.md
notes/025_audit_025_mvp_contract_prompt_against_assessment__20260417_173903.md
notes/005_prompt_queue_plan__20260415_202557.md
notes/010_run_prompt_candidate_review__20260415_234559.md
notes/010_next_step_recommendation__20260415_234559.md
notes/016_queue_and_backlog_helper_validation__20260416_003710.md
notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md
notes/001_smoke_test_pipeline__20260415_234918.md
notes/014_build_queue_readiness_checker__20260416_002319.md
notes/025_audit_025_mvp_contract_prompt_against_assessment__20260417_175240.md
notes/004_next_design_step_recommendation__20260415_195538.md
notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md
notes/028_define_minimal_wrapper_scope_state_and_resume_contract__20260417_190057.md
notes/029_write_030_stage1_prompt_generation_contract_prompt__20260417_190736.md
notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md
notes/018_architecture_vs_actual_sweep__20260416_005130.md
notes/015_review_backlog_lister_build__20260416_010500.md
notes/012_v1_pipeline_options_review__20260416_000819.md
notes/025_define_staged_notebook_wrapper_mvp_contract__20260417_175751.md
notes/025_audit_025_mvp_contract_prompt_against_assessment__20260417_173231.md
notes/003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md
notes/022_scaffolding_classification_report__20260416_023016.md
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md
notes/021_record_contract_validation__20260416_011314.md
notes/020_doc_spec_alignment_cleanup__20260416_010534.md
notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md
notes/018_prioritized_remaining_work__20260416_005130.md
notes/001_smoke_test_pipeline__20260415_233343.md
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md
notes/025_first_prompt_audit__20260417_125331.md
notes/023_tools_codex_assessment__20260417_171721.md
notes/011_build_v1_review_writeback_helper__20260415_235346.md
notes/001_notebook_triage_05_06_07.md
notes/020_align_v1_doc_and_spec_packet_to_actual_toolset__20260416_010453.md
notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md
notes/012_top_three_next_options__20260416_000819.md

exec
/bin/bash -lc 'git status --short' in /home/jevert/git/handson-ml3
 succeeded in 0ms:
?? codex_prompts/031_build_stage1_wrapper_mvp_from_contracts.md

codex
The repository layout is straightforward: notebooks live mostly at repo root, prompts go in `codex_prompts/`, and nothing in the current contracts requires runner integration for this pass. I’m now checking how the scanner classifies a small notebook sample so the wrapper and the report can be validated against an actual generate/skip split.
exec
/bin/bash -lc "python - <<'PY'
from pathlib import Path
from tools.notebook_enricher.notebook_scanner import scan_notebook
for path_str in ['06_decision_trees.ipynb','05_support_vector_machines.ipynb','01_the_machine_learning_landscape.ipynb','03_classification.ipynb']:
    inv = scan_notebook(Path(path_str))
    print(path_str, inv.chapter_intro_status, inv.chapter_intro_index, inv.setup_cell_index, inv.needs_stage1())
PY" in /home/jevert/git/handson-ml3
 exited 127 in 0ms:
/bin/bash: line 1: python: command not found

exec
/bin/bash -lc "sed -n '1,220p' codex_prompts/024_generate_staged_notebook_wrapper_prompts.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
# Prompt: Generate the staged prompt set for the notebook-wrapper build

You are working inside this repository.

Your task is to use the existing assessment and current tooling layout to create the first draft prompt set that will later be used to build the staged notebook-treatment wrapper.

Do NOT build the wrapper itself yet.

Do NOT modify any `.py` files.
Do NOT modify any notebooks.
Do NOT refactor the current runner.
Do NOT change the V1 record format.

## Goal

We already fenced the problem and assessed what the wrapper should be.

Now generate the prompt files that would let Codex build this wrapper in careful stages.

The wrapper concept is:

- stay centered around Codex calls
- keep the current `tools/codex/run_prompt.py` flow intact
- use narrow, bounded tasks instead of one giant notebook-enrichment run
- support these staged notebook-treatment passes:
  - Stage 1: chapter intro detection / insertion
  - Stage 2: markdown-before-code detection / insertion
  - Stage 3: markdown-after-code detection / insertion

## What to inspect first

Inspect at minimum:

- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `tools/codex/v1_record_validation.py`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/Project_Design_Workflow.md`
- the existing assessment note for this wrapper idea
- `tools/notebook_enricher/notebook_scanner.py`
- `tools/notebook_enricher/prompt_builder.py`

## Your job

Create a practical first-draft prompt set that breaks the future implementation into small Codex-sized tasks.

These prompts should be designed so that each one:
- has a narrow scope
- can be run through the current Codex runner
- produces durable notes and visible progress
- minimizes risk of notebook corruption
- supports stop/resume behavior
- avoids overengineering

## Deliverables

Create:

1. A small sequence of new prompt files in `codex_prompts/` for the wrapper build
2. One planning note in `notes/` that explains:
   - what prompt files you created
   - what each prompt is meant to accomplish
   - why the sequence is ordered this way
   - any open concerns or assumptions

## Requirements for the generated prompt set

The prompt set should cover, in some sensible order:

- defining the wrapper state model
- defining notebook scope/config input
- defining progress tracking and stop/resume behavior
- defining notebook scan output / treatment detection
- defining safe prompt-generation rules for Stage 1, Stage 2, Stage 3
- defining notebook-write safeguards
- defining a minimal orchestration loop
- defining reporting / note artifacts for the wrapper runs
- defining what the MVP should do first

The prompts should not yet ask Codex to:
- enrich all notebooks
- process the whole repo in one pass
- redesign the V1 runner
- add parallel execution
- add a UI
- add advanced scheduling
- add automatic self-healing
- add background daemons

## Naming

Use the next available numeric prefixes in `codex_prompts/`.

Make the prompt filenames descriptive and consistent with the existing repo style.

## Output note structure

Write one markdown note into `notes/` with this structure:

# Staged Notebook Wrapper Prompt Plan

## Executive Summary
## Prompt Files Created
## Why This Sequence
## Scope Boundaries
## Risks Still Open
## Recommended Next Prompt To Run

## Output rules

- Be practical
- Be repo-specific
- Keep the prompt set small and disciplined
- Prefer the minimum viable sequence over a giant speculative backlog
- At the end of your final response, print only the path to the planning note you created

exec
/bin/bash -lc "sed -n '1,220p' codex_prompts/031_build_stage1_wrapper_mvp_from_contracts.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
# Prompt: Build the Stage 1 wrapper MVP from the existing contracts

You are working inside this repository.

Your task is to implement the thinnest practical Stage 1 wrapper MVP that is justified by the existing design contracts.

This is an implementation prompt.

## Hard boundaries

Do NOT implement Stage 2.
Do NOT implement Stage 3.
Do NOT redesign `tools/codex/run_prompt.py`.
Do NOT change the V1 execution record format.
Do NOT build a general queue engine.
Do NOT add parallel execution.
Do NOT add dashboards or background daemons.
Do NOT make the wrapper auto-run generated prompts in this pass.
Do NOT modify any notebooks in this pass.

## Goal

Build a small Stage 1-only wrapper that:

- accepts an explicit ordered notebook list as input
- uses scanner-first logic to determine whether Stage 1 is needed for each notebook
- generates Stage 1 prompt files only for notebooks that need chapter-intro treatment
- skips notebooks that do not need Stage 1 treatment
- writes one small report describing what it generated, what it skipped, and why

This wrapper must stop at prompt generation.
It must not execute the generated Stage 1 prompts yet.

## Files to inspect first

Inspect at minimum:

- `notes/025_define_staged_notebook_wrapper_mvp_contract__20260417_175751.md`
- `notes/028_minimal_wrapper_scope_state_and_resume_contract__20260417_190146.md`
- `notes/030_define_stage1_prompt_generation_contract__20260417_191113.md`
- `tools/notebook_enricher/notebook_scanner.py`
- `tools/notebook_enricher/prompt_builder.py`
- `tools/codex/run_prompt.py`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Execution_Record_Artifact.md`

Also inspect the repository layout enough to confirm where notebooks live and where generated prompt artifacts should go.

## What to build

Build only the minimum needed to support this Stage 1 prompt-generation slice.

### Required deliverables

Create or update only the smallest reasonable set of files needed for this slice.

At minimum, the implementation should include:

1. A Stage 1 wrapper entry point under `tools/notebook_enricher/`
   - choose a practical filename consistent with the current repo style
   - this script should be the thin entry point for Stage 1 prompt generation

2. Any minimal helper changes needed inside `tools/notebook_enricher/`
   - only if they are actually required
   - prefer reusing existing scanner / prompt-builder logic as-is

3. One markdown report in `notes/` for this run
   - summarize:
     - target notebook list
     - which notebooks needed Stage 1 prompt generation
     - which notebooks were skipped
     - why each notebook was generated or skipped
     - what prompt files were created

## Required behavior

The wrapper MVP must do all of the following:

### 1. Input model
Support a narrow explicit notebook scope as input.

Use one of these approaches:
- positional notebook-path arguments
- or one explicit text/json file containing an ordered notebook list

Choose the simpler option.

The wrapper must not do repo-wide autonomous discovery in this pass.

### 2. Scanner-first behavior
For each notebook in the explicit ordered scope:

- run scanner logic first
- determine whether Stage 1 is needed
- determine the intro status
- determine whether the action should be insert, replace, or skip

Prompt generation must depend on scanner evidence, not ad hoc notebook guessing.

### 3. Prompt generation only
For notebooks that need Stage 1 treatment:

- generate exactly one Stage 1 prompt file per notebook
- use the existing prompt-builder patterns where practical
- keep the prompt bounded to one notebook and one Stage 1 decision

For notebooks that do not need Stage 1 treatment:

- do not generate a prompt
- record the skip in the report with the reason

### 4. Output location
Write generated prompt files into `codex_prompts/`.

Use a clear naming convention that keeps these generated Stage 1 prompts obviously distinct from the hand-authored design/build prompts already in the repo.

Do NOT overwrite existing hand-authored prompts.

Use a naming pattern that is deterministic and reviewable.

### 5. No auto-run
Do not invoke `tools/codex/run_prompt.py` on the generated prompt files in this pass.

This slice ends at “prompt files generated.”

### 6. Safety rules
Preserve all current boundaries:

- no notebook mutation in this wrapper pass
- no direct code-cell edits
- no Stage 2 or Stage 3 work
- no orchestration-loop breadth
- no wrapper-local redefinition of canonical execution/review truth in `notes/`

## Implementation guidance

Follow the existing contracts.

The implementation should reflect these already-set boundaries:

- Stage 1 first
- explicit ordered notebook scope
- runner-centered workflow remains intact
- scanner-derived evidence drives prompt generation
- Stage 1 prompt generation is one notebook / one intro decision / bounded action only

Prefer the smallest implementation that works.

Do not gold-plate this.

## Naming and reviewability

Generated prompt files should be easy for a human to inspect before running.

The report in `notes/` should make it easy to answer:

- what notebooks were requested
- what the wrapper decided
- what files were generated
- what still remains deferred

## Validation

Before finishing, do at least these checks:

1. Confirm the wrapper can accept an explicit ordered notebook scope.
2. Confirm it can scan notebooks and classify Stage 1 need.
3. Confirm it generates prompt files only for notebooks that need Stage 1.
4. Confirm it does not run those generated prompts.
5. Confirm the report clearly explains generated vs skipped notebooks.
6. Confirm no notebooks were modified during this wrapper build pass.

If practical, use a very small test scope such as one notebook needing treatment and one notebook not needing treatment.

## Success criteria

This prompt is successful only if ALL of the following are true:

1. There is now a thin runnable Stage 1 wrapper MVP under `tools/notebook_enricher/`.
2. It accepts an explicit ordered notebook scope.
3. It uses scanner-first logic.
4. It generates Stage 1 prompt files only for notebooks that need intro treatment.
5. It does not run generated prompts.
6. It does not modify notebooks.
7. It writes a small report in `notes/`.
8. The implementation stays narrow and does not drift into Stage 2, Stage 3, or orchestration breadth.

## Output rules

- Be concrete
- Reuse existing code where practical
- Keep the change set as small as possible
- Favor a thin working slice over a flexible framework
- At the end of your final response, print only the path to the note you created

codex
I’ve got enough context to keep this narrow. The implementation plan is: add one wrapper script under `tools/notebook_enricher/`, keep scanner and prompt-builder reuse intact, then run the wrapper on a tiny explicit notebook list to generate prompts and a report without touching notebooks or invoking the runner.
exec
/bin/bash -lc "python3 - <<'PY'
from pathlib import Path
from tools.notebook_enricher.notebook_scanner import scan_notebook
for path_str in ['06_decision_trees.ipynb','05_support_vector_machines.ipynb','01_the_machine_learning_landscape.ipynb','03_classification.ipynb','07_ensemble_learning_and_random_forests.ipynb']:
    inv = scan_notebook(Path(path_str))
    print(f\"{path_str}: status={inv.chapter_intro_status}, intro_index={inv.chapter_intro_index}, setup_index={inv.setup_cell_index}, needs_stage1={inv.needs_stage1()}\")
PY" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
06_decision_trees.ipynb: status=substantive, intro_index=3, setup_index=4, needs_stage1=False
05_support_vector_machines.ipynb: status=missing, intro_index=-1, setup_index=3, needs_stage1=True
01_the_machine_learning_landscape.ipynb: status=thin, intro_index=0, setup_index=2, needs_stage1=True
03_classification.ipynb: status=missing, intro_index=-1, setup_index=3, needs_stage1=True
07_ensemble_learning_and_random_forests.ipynb: status=thin, intro_index=2, setup_index=4, needs_stage1=True

exec
/bin/bash -lc "sed -n '1,260p' tools/notebook_enricher/notebook_scanner.py" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
"""
notebook_scanner.py — Cell inventory and treatment-state detection.

Reads a .ipynb file and classifies every code cell's treatment state:
  - has_goal_before  : 'canonical', 'legacy', 'thin', 'missing'
  - has_impl_after   : 'canonical', 'legacy', 'thin', 'missing'

Also detects the chapter intro status and the Setup cell position.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Detection vocabulary
# ---------------------------------------------------------------------------

GOAL_CANONICAL = "### Goal Before This Cell"
IMPL_CANONICAL = "### Implementation Notes After This Cell"

GOAL_LEGACY_KEYWORDS = [
    "**Why run this cell",
    "**Goal:",
    "**What it is",
    "**Principle:",
]

IMPL_LEGACY_KEYWORDS = [
    "**Result**:",
    "**Why this matters",
]

# A cell whose source starts with `# Setup` marks the boundary for intro detection.
SETUP_MARKERS = ("# setup", "## setup")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CellState:
    index: int
    cell_type: str          # 'code' or 'markdown'
    is_empty: bool          # True if source is whitespace only
    has_goal_before: str    # 'canonical', 'legacy', 'thin', 'missing'
    has_impl_after: str     # 'canonical', 'legacy', 'thin', 'missing'
    goal_cell_index: int    # index of preceding markdown cell, or -1
    impl_cell_index: int    # index of following markdown cell, or -1

    def needs_goal(self) -> bool:
        return self.has_goal_before in ("missing", "legacy", "thin") and not self.is_empty

    def needs_impl(self) -> bool:
        return self.has_impl_after in ("missing", "legacy", "thin") and not self.is_empty


@dataclass
class NotebookInventory:
    path: Path
    notebook_stem: str
    total_cells: int
    code_cell_states: list[CellState]   # one entry per code cell
    chapter_intro_status: str           # 'substantive', 'thin', 'heading', 'missing'
    chapter_intro_index: int            # cell index of existing intro, -1 if missing
    setup_cell_index: int               # index of the '# Setup' cell, -1 if not found

    def needs_stage1(self) -> bool:
        """Stage 1 is needed when chapter intro is not already substantive."""
        return self.chapter_intro_status != "substantive"

    def cells_needing_goal(self) -> list[CellState]:
        return [cs for cs in self.code_cell_states if cs.needs_goal()]

    def cells_needing_impl(self) -> list[CellState]:
        return [cs for cs in self.code_cell_states if cs.needs_impl()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cell_source(cell: dict) -> str:
    """Return the full source text of a notebook cell."""
    return "".join(cell.get("source", []))


def word_count(text: str) -> int:
    return len(text.split())


def _classify_goal(src: str) -> str:
    if GOAL_CANONICAL in src:
        return "canonical"
    if any(kw in src for kw in GOAL_LEGACY_KEYWORDS):
        return "legacy"
    # Anything else (section header, book narrative, etc.) is 'thin'
    return "thin"


def _classify_impl(src: str) -> str:
    if IMPL_CANONICAL in src:
        return "canonical"
    if any(kw in src for kw in IMPL_LEGACY_KEYWORDS):
        return "legacy"
    # Short bold-starting cell counts as legacy per spec
    if src.strip().startswith("**") and word_count(src) < 100:
        return "legacy"
    return "thin"


# ---------------------------------------------------------------------------
# Chapter intro detection
# ---------------------------------------------------------------------------

def _is_setup_cell(src: str) -> bool:
    stripped = src.strip().lower()
    return any(stripped.startswith(m) for m in SETUP_MARKERS)


def _is_prose_candidate(cell: dict, src: str) -> bool:
    """True if this markdown cell could be the chapter intro (not a heading-only or HTML cell)."""
    if cell.get("cell_type") != "markdown":
        return False
    stripped = src.strip()
    # Skip the bold-title cell like **Chapter 7 – ...**
    if stripped.startswith("**") and "\n" not in stripped:
        return False
    # Skip the italic notebook-description line
    if stripped.startswith("_This notebook contains") and "\n" not in stripped:
        return False
    # Skip HTML tables (colab/kaggle links)
    if stripped.startswith("<table"):
        return False
    return True


def detect_chapter_intro(cells: list[dict]) -> tuple[str, int, int]:
    """
    Scan cells 0-5 (or up to # Setup) and identify the chapter intro.

    Returns:
        (status, intro_index, setup_index)

    status values: 'substantive', 'thin', 'heading', 'missing'
    intro_index: cell index of the intro, or -1 if missing
    setup_index: cell index of the # Setup cell, or -1 if not found
    """
    setup_index = -1
    for i, cell in enumerate(cells):
        if _is_setup_cell(cell_source(cell)):
            setup_index = i
            break

    upper_bound = min(6, setup_index if setup_index != -1 else len(cells))

    best_index = -1
    best_wc = -1
    for i in range(upper_bound):
        cell = cells[i]
        src = cell_source(cell)
        if not _is_prose_candidate(cell, src):
            continue
        wc = word_count(src)
        if wc > best_wc:
            best_wc = wc
            best_index = i

    if best_index == -1 or best_wc < 10:
        return "missing", -1, setup_index

    if best_wc > 200:
        status = "substantive"
    elif best_wc >= 50:
        status = "thin"
    else:
        status = "heading"

    return status, best_index, setup_index


# ---------------------------------------------------------------------------
# Main scan function
# ---------------------------------------------------------------------------

def scan_notebook(path: Path) -> NotebookInventory:
    """Read a notebook and return a full cell inventory."""
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    intro_status, intro_index, setup_index = detect_chapter_intro(cells)

    code_cell_states: list[CellState] = []

    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        src = cell_source(cell)
        is_empty = not src.strip()

        # Classify goal_before
        goal_cell_idx = -1
        if i > 0 and cells[i - 1].get("cell_type") == "markdown":
            goal_cell_idx = i - 1
            goal_status = _classify_goal(cell_source(cells[i - 1]))
        else:
            goal_status = "missing"

        # Classify impl_after
        impl_cell_idx = -1
        if i < len(cells) - 1 and cells[i + 1].get("cell_type") == "markdown":
            impl_cell_idx = i + 1
            impl_status = _classify_impl(cell_source(cells[i + 1]))
        else:
            impl_status = "missing"

        code_cell_states.append(CellState(
            index=i,
            cell_type="code",
            is_empty=is_empty,
            has_goal_before=goal_status,
            has_impl_after=impl_status,
            goal_cell_index=goal_cell_idx,
            impl_cell_index=impl_cell_idx,
        ))

    return NotebookInventory(
        path=path,
        notebook_stem=path.stem,
        total_cells=len(cells),
        code_cell_states=code_cell_states,
        chapter_intro_status=intro_status,
        chapter_intro_index=intro_index,
        setup_cell_index=setup_index,
    )


def format_inventory_report(inv: NotebookInventory) -> str:
    """Human-readable summary of a notebook scan result."""
    lines = [
        f"Notebook: {inv.notebook_stem}",
        f"Total cells: {inv.total_cells}",
        f"Code cells: {len(inv.code_cell_states)}",
        f"Chapter intro: {inv.chapter_intro_status} (cell {inv.chapter_intro_index})",
        f"Setup cell: {inv.setup_cell_index}",
        "",
        f"Stage 1 needed: {inv.needs_stage1()}",
        f"Cells needing goal (Stage 2): {len(inv.cells_needing_goal())}",
        f"Cells needing impl (Stage 3): {len(inv.cells_needing_impl())}",
        "",
        "Code cell detail:",
    ]
    for cs in inv.code_cell_states:
        if cs.is_empty:

codex
I’m at the edit point. The wrapper will stay self-contained so I don’t have to widen the existing scanner or prompt-builder API; it will compute the Stage 1 action from scanner output, write generated prompts under a distinct `generated_stage1__...` naming scheme, and emit one markdown report with the validation run details.
apply patch
patch: completed
/home/jevert/git/handson-ml3/tools/notebook_enricher/stage1_prompt_wrapper.py
diff --git a/tools/notebook_enricher/stage1_prompt_wrapper.py b/tools/notebook_enricher/stage1_prompt_wrapper.py
new file mode 100644
index 0000000000000000000000000000000000000000..06ebb33701bd867acefbf5703d0317758f9a43b2
--- /dev/null
+++ b/tools/notebook_enricher/stage1_prompt_wrapper.py
@@ -0,0 +1,296 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+from dataclasses import dataclass
+from pathlib import Path
+import sys
+
+from notebook_scanner import NotebookInventory, scan_notebook
+from prompt_builder import build_stage1_prompt
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+GENERATED_PROMPT_PREFIX = "generated_stage1__"
+REPORT_PREFIX = "031_stage1_wrapper_mvp_report__"
+
+
+@dataclass
+class Stage1Decision:
+    inventory: NotebookInventory
+    action: str
+    reason: str
+
+    @property
+    def needs_prompt(self) -> bool:
+        return self.action in {"insert", "replace"}
+
+
+@dataclass
+class PromptArtifact:
+    notebook_path: Path
+    prompt_path: Path
+    action: str
+    reason: str
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def utc_timestamp() -> str:
+    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Generate Stage 1 notebook-enrichment prompts for an explicit notebook list.",
+    )
+    parser.add_argument(
+        "notebooks",
+        nargs="+",
+        help="Ordered notebook paths, relative to the repository root or absolute.",
+    )
+    parser.add_argument(
+        "--report-path",
+        help="Optional explicit markdown report path. Defaults to notes/ with a UTC timestamp.",
+    )
+    parser.add_argument(
+        "--overwrite-generated",
+        action="store_true",
+        help="Allow overwriting an existing generated Stage 1 prompt file for the same notebook.",
+    )
+    return parser.parse_args()
+
+
+def resolve_notebook_path(root: Path, notebook_arg: str) -> Path:
+    path = Path(notebook_arg)
+    if not path.is_absolute():
+        path = root / path
+    return path.resolve()
+
+
+def relative_repo_path(root: Path, path: Path) -> str:
+    return path.relative_to(root).as_posix()
+
+
+def classify_stage1_decision(inventory: NotebookInventory) -> Stage1Decision:
+    status = inventory.chapter_intro_status
+
+    if status == "substantive":
+        return Stage1Decision(
+            inventory=inventory,
+            action="skip",
+            reason="scanner classified the existing chapter intro as substantive",
+        )
+
+    if status == "missing":
+        return Stage1Decision(
+            inventory=inventory,
+            action="insert",
+            reason="scanner found no eligible chapter intro prose cell before the setup boundary",
+        )
+
+    if status in {"heading", "thin"} and inventory.chapter_intro_index >= 0:
+        return Stage1Decision(
+            inventory=inventory,
+            action="replace",
+            reason=(
+                "scanner found an existing intro candidate that is not yet substantive "
+                f"({status})"
+            ),
+        )
+
+    return Stage1Decision(
+        inventory=inventory,
+        action="skip",
+        reason="scanner evidence did not support one deterministic Stage 1 insert-or-replace target",
+    )
+
+
+def prompt_filename_for(root: Path, notebook_path: Path) -> str:
+    relative = notebook_path.relative_to(root).with_suffix("")
+    slug = "__".join(relative.parts)
+    return f"{GENERATED_PROMPT_PREFIX}{slug}.md"
+
+
+def write_prompt_file(
+    *,
+    root: Path,
+    decision: Stage1Decision,
+    overwrite_generated: bool,
+) -> PromptArtifact:
+    prompts_dir = root / PROMPTS_DIR
+    prompts_dir.mkdir(parents=True, exist_ok=True)
+
+    prompt_path = prompts_dir / prompt_filename_for(root, decision.inventory.path)
+    if prompt_path.exists() and not overwrite_generated:
+        raise FileExistsError(
+            f"generated prompt already exists: {relative_repo_path(root, prompt_path)} "
+            "(rerun with --overwrite-generated to replace it)"
+        )
+
+    prompt_text = build_stage1_prompt(root, decision.inventory)
+    prompt_path.write_text(prompt_text, encoding="utf-8")
+
+    return PromptArtifact(
+        notebook_path=decision.inventory.path,
+        prompt_path=prompt_path,
+        action=decision.action,
+        reason=decision.reason,
+    )
+
+
+def default_report_path(root: Path) -> Path:
+    return root / NOTES_DIR / f"{REPORT_PREFIX}{utc_timestamp()}.md"
+
+
+def build_report_text(
+    *,
+    root: Path,
+    notebook_paths: list[Path],
+    generated: list[PromptArtifact],
+    skipped: list[Stage1Decision],
+) -> str:
+    lines: list[str] = [
+        "# Stage 1 Wrapper MVP Report",
+        "",
+        "## Executive Summary",
+        "",
+        (
+            f"Requested {len(notebook_paths)} notebook(s). Generated {len(generated)} "
+            f"Stage 1 prompt(s) and skipped {len(skipped)} notebook(s)."
+        ),
+        "This wrapper run stopped at prompt generation and did not invoke `tools/codex/run_prompt.py`.",
+        "",
+        "## Target Notebook List",
+        "",
+    ]
+
+    for idx, notebook_path in enumerate(notebook_paths, start=1):
+        lines.append(f"{idx}. `{relative_repo_path(root, notebook_path)}`")
+
+    lines.extend(
+        [
+            "",
+            "## Generated Prompts",
+            "",
+        ]
+    )
+
+    if generated:
+        for artifact in generated:
+            lines.append(
+                (
+                    f"- `{relative_repo_path(root, artifact.notebook_path)}`: generated "
+                    f"`{relative_repo_path(root, artifact.prompt_path)}` "
+                    f"because action is `{artifact.action}` and {artifact.reason}."
+                )
+            )
+    else:
+        lines.append("- None.")
+
+    lines.extend(
+        [
+            "",
+            "## Skipped Notebooks",
+            "",
+        ]
+    )
+
+    if skipped:
+        for decision in skipped:
+            inv = decision.inventory
+            lines.append(
+                (
+                    f"- `{relative_repo_path(root, inv.path)}`: skipped because action is "
+                    f"`{decision.action}` and {decision.reason}. "
+                    f"(intro_status=`{inv.chapter_intro_status}`, "
+                    f"intro_index=`{inv.chapter_intro_index}`, "
+                    f"setup_index=`{inv.setup_cell_index}`)"
+                )
+            )
+    else:
+        lines.append("- None.")
+
+    lines.extend(
+        [
+            "",
+            "## Deferred Work",
+            "",
+            "- No generated prompts were executed in this pass.",
+            "- No notebooks were modified in this pass.",
+            "- Stage 2 and Stage 3 remain out of scope.",
+        ]
+    )
+
+    return "\n".join(lines) + "\n"
+
+
+def main() -> int:
+    args = parse_args()
+    root = repo_root()
+
+    notebook_paths: list[Path] = []
+    for notebook_arg in args.notebooks:
+        notebook_path = resolve_notebook_path(root, notebook_arg)
+        if not notebook_path.exists():
+            print(f"ERROR: notebook not found: {notebook_arg}", file=sys.stderr)
+            return 1
+        try:
+            notebook_path.relative_to(root)
+        except ValueError:
+            print(
+                f"ERROR: notebook path is outside the repository root: {notebook_path}",
+                file=sys.stderr,
+            )
+            return 1
+        if notebook_path.suffix != ".ipynb":
+            print(f"ERROR: not a notebook path: {notebook_arg}", file=sys.stderr)
+            return 1
+        notebook_paths.append(notebook_path)
+
+    generated: list[PromptArtifact] = []
+    skipped: list[Stage1Decision] = []
+
+    for notebook_path in notebook_paths:
+        inventory = scan_notebook(notebook_path)
+        decision = classify_stage1_decision(inventory)
+        if decision.needs_prompt:
+            artifact = write_prompt_file(
+                root=root,
+                decision=decision,
+                overwrite_generated=args.overwrite_generated,
+            )
+            generated.append(artifact)
+            print(
+                f"GENERATED {relative_repo_path(root, artifact.prompt_path)} "
+                f"for {relative_repo_path(root, notebook_path)} "
+                f"({decision.action})"
+            )
+        else:
+            skipped.append(decision)
+            print(
+                f"SKIPPED {relative_repo_path(root, notebook_path)} "
+                f"({decision.reason})"
+            )
+
+    report_path = Path(args.report_path).resolve() if args.report_path else default_report_path(root)
+    report_path.parent.mkdir(parents=True, exist_ok=True)
+    report_text = build_report_text(
+        root=root,
+        notebook_paths=notebook_paths,
+        generated=generated,
+        skipped=skipped,
+    )
+    report_path.write_text(report_text, encoding="utf-8")
+    print(f"REPORT {relative_repo_path(root, report_path)}")
+
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

diff --git a/tools/notebook_enricher/stage1_prompt_wrapper.py b/tools/notebook_enricher/stage1_prompt_wrapper.py
new file mode 100644
index 0000000000000000000000000000000000000000..06ebb33701bd867acefbf5703d0317758f9a43b2
--- /dev/null
+++ b/tools/notebook_enricher/stage1_prompt_wrapper.py
@@ -0,0 +1,296 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+from dataclasses import dataclass
+from pathlib import Path
+import sys
+
+from notebook_scanner import NotebookInventory, scan_notebook
+from prompt_builder import build_stage1_prompt
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+GENERATED_PROMPT_PREFIX = "generated_stage1__"
+REPORT_PREFIX = "031_stage1_wrapper_mvp_report__"
+
+
+@dataclass
+class Stage1Decision:
+    inventory: NotebookInventory
+    action: str
+    reason: str
+
+    @property
+    def needs_prompt(self) -> bool:
+        return self.action in {"insert", "replace"}
+
+
+@dataclass
+class PromptArtifact:
+    notebook_path: Path
+    prompt_path: Path
+    action: str
+    reason: str
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def utc_timestamp() -> str:
+    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Generate Stage 1 notebook-enrichment prompts for an explicit notebook list.",
+    )
+    parser.add_argument(
+        "notebooks",
+        nargs="+",
+        help="Ordered notebook paths, relative to the repository root or absolute.",
+    )
+    parser.add_argument(
+        "--report-path",
+        help="Optional explicit markdown report path. Defaults to notes/ with a UTC timestamp.",
+    )
+    parser.add_argument(
+        "--overwrite-generated",
+        action="store_true",
+        help="Allow overwriting an existing generated Stage 1 prompt file for the same notebook.",
+    )
+    return parser.parse_args()
+
+
+def resolve_notebook_path(root: Path, notebook_arg: str) -> Path:
+    path = Path(notebook_arg)
+    if not path.is_absolute():
+        path = root / path
+    return path.resolve()
+
+
+def relative_repo_path(root: Path, path: Path) -> str:
+    return path.relative_to(root).as_posix()
+
+
+def classify_stage1_decision(inventory: NotebookInventory) -> Stage1Decision:
+    status = inventory.chapter_intro_status
+
+    if status == "substantive":
+        return Stage1Decision(
+            inventory=inventory,
+            action="skip",
+            reason="scanner classified the existing chapter intro as substantive",
+        )
+
+    if status == "missing":
+        return Stage1Decision(
+            inventory=inventory,
+            action="insert",
+            reason="scanner found no eligible chapter intro prose cell before the setup boundary",
+        )
+
+    if status in {"heading", "thin"} and inventory.chapter_intro_index >= 0:
+        return Stage1Decision(
+            inventory=inventory,
+            action="replace",
+            reason=(
+                "scanner found an existing intro candidate that is not yet substantive "
+                f"({status})"
+            ),
+        )
+
+    return Stage1Decision(
+        inventory=inventory,
+        action="skip",
+        reason="scanner evidence did not support one deterministic Stage 1 insert-or-replace target",
+    )
+
+
+def prompt_filename_for(root: Path, notebook_path: Path) -> str:
+    relative = notebook_path.relative_to(root).with_suffix("")
+    slug = "__".join(relative.parts)
+    return f"{GENERATED_PROMPT_PREFIX}{slug}.md"
+
+
+def write_prompt_file(
+    *,
+    root: Path,
+    decision: Stage1Decision,
+    overwrite_generated: bool,
+) -> PromptArtifact:
+    prompts_dir = root / PROMPTS_DIR
+    prompts_dir.mkdir(parents=True, exist_ok=True)
+
+    prompt_path = prompts_dir / prompt_filename_for(root, decision.inventory.path)
+    if prompt_path.exists() and not overwrite_generated:
+        raise FileExistsError(
+            f"generated prompt already exists: {relative_repo_path(root, prompt_path)} "
+            "(rerun with --overwrite-generated to replace it)"
+        )
+
+    prompt_text = build_stage1_prompt(root, decision.inventory)
+    prompt_path.write_text(prompt_text, encoding="utf-8")
+
+    return PromptArtifact(
+        notebook_path=decision.inventory.path,
+        prompt_path=prompt_path,
+        action=decision.action,
+        reason=decision.reason,
+    )
+
+
+def default_report_path(root: Path) -> Path:
+    return root / NOTES_DIR / f"{REPORT_PREFIX}{utc_timestamp()}.md"
+
+
+def build_report_text(
+    *,
+    root: Path,
+    notebook_paths: list[Path],
+    generated: list[PromptArtifact],
+    skipped: list[Stage1Decision],
+) -> str:
+    lines: list[str] = [
+        "# Stage 1 Wrapper MVP Report",
+        "",
+        "## Executive Summary",
+        "",
+        (
+            f"Requested {len(notebook_paths)} notebook(s). Generated {len(generated)} "
+            f"Stage 1 prompt(s) and skipped {len(skipped)} notebook(s)."
+        ),
+        "This wrapper run stopped at prompt generation and did not invoke `tools/codex/run_prompt.py`.",
+        "",
+        "## Target Notebook List",
+        "",
+    ]
+
+    for idx, notebook_path in enumerate(notebook_paths, start=1):
+        lines.append(f"{idx}. `{relative_repo_path(root, notebook_path)}`")
+
+    lines.extend(
+        [
+            "",
+            "## Generated Prompts",
+            "",
+        ]
+    )
+
+    if generated:
+        for artifact in generated:
+            lines.append(
+                (
+                    f"- `{relative_repo_path(root, artifact.notebook_path)}`: generated "
+                    f"`{relative_repo_path(root, artifact.prompt_path)}` "
+                    f"because action is `{artifact.action}` and {artifact.reason}."
+                )
+            )
+    else:
+        lines.append("- None.")
+
+    lines.extend(
+        [
+            "",
+            "## Skipped Notebooks",
+            "",
+        ]
+    )
+
+    if skipped:
+        for decision in skipped:
+            inv = decision.inventory
+            lines.append(
+                (
+                    f"- `{relative_repo_path(root, inv.path)}`: skipped because action is "
+                    f"`{decision.action}` and {decision.reason}. "
+                    f"(intro_status=`{inv.chapter_intro_status}`, "
+                    f"intro_index=`{inv.chapter_intro_index}`, "
+                    f"setup_index=`{inv.setup_cell_index}`)"
+                )
+            )
+    else:
+        lines.append("- None.")
+
+    lines.extend(
+        [
+            "",
+            "## Deferred Work",
+            "",
+            "- No generated prompts were executed in this pass.",
+            "- No notebooks were modified in this pass.",
+            "- Stage 2 and Stage 3 remain out of scope.",
+        ]
+    )
+
+    return "\n".join(lines) + "\n"
+
+
+def main() -> int:
+    args = parse_args()
+    root = repo_root()
+
+    notebook_paths: list[Path] = []
+    for notebook_arg in args.notebooks:
+        notebook_path = resolve_notebook_path(root, notebook_arg)
+        if not notebook_path.exists():
+            print(f"ERROR: notebook not found: {notebook_arg}", file=sys.stderr)
+            return 1
+        try:
+            notebook_path.relative_to(root)
+        except ValueError:
+            print(
+                f"ERROR: notebook path is outside the repository root: {notebook_path}",
+                file=sys.stderr,
+            )
+            return 1
+        if notebook_path.suffix != ".ipynb":
+            print(f"ERROR: not a notebook path: {notebook_arg}", file=sys.stderr)
+            return 1
+        notebook_paths.append(notebook_path)
+
+    generated: list[PromptArtifact] = []
+    skipped: list[Stage1Decision] = []
+
+    for notebook_path in notebook_paths:
+        inventory = scan_notebook(notebook_path)
+        decision = classify_stage1_decision(inventory)
+        if decision.needs_prompt:
+            artifact = write_prompt_file(
+                root=root,
+                decision=decision,
+                overwrite_generated=args.overwrite_generated,
+            )
+            generated.append(artifact)
+            print(
+                f"GENERATED {relative_repo_path(root, artifact.prompt_path)} "
+                f"for {relative_repo_path(root, notebook_path)} "
+                f"({decision.action})"
+            )
+        else:
+            skipped.append(decision)
+            print(
+                f"SKIPPED {relative_repo_path(root, notebook_path)} "
+                f"({decision.reason})"
+            )
+
+    report_path = Path(args.report_path).resolve() if args.report_path else default_report_path(root)
+    report_path.parent.mkdir(parents=True, exist_ok=True)
+    report_text = build_report_text(
+        root=root,
+        notebook_paths=notebook_paths,
+        generated=generated,
+        skipped=skipped,
+    )
+    report_path.write_text(report_text, encoding="utf-8")
+    print(f"REPORT {relative_repo_path(root, report_path)}")
+
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

codex
The wrapper file is in place. I’m tightening one small path-handling edge now, then I’ll run the actual validation scope and inspect the generated prompt/report artifacts before finishing.
apply patch
patch: completed
/home/jevert/git/handson-ml3/tools/notebook_enricher/stage1_prompt_wrapper.py
diff --git a/tools/notebook_enricher/stage1_prompt_wrapper.py b/tools/notebook_enricher/stage1_prompt_wrapper.py
new file mode 100644
index 0000000000000000000000000000000000000000..ccef40439f9d87bde89c38ad4b3386256b87f94b
--- /dev/null
+++ b/tools/notebook_enricher/stage1_prompt_wrapper.py
@@ -0,0 +1,303 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+from dataclasses import dataclass
+from pathlib import Path
+import sys
+
+from notebook_scanner import NotebookInventory, scan_notebook
+from prompt_builder import build_stage1_prompt
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+GENERATED_PROMPT_PREFIX = "generated_stage1__"
+REPORT_PREFIX = "031_stage1_wrapper_mvp_report__"
+
+
+@dataclass
+class Stage1Decision:
+    inventory: NotebookInventory
+    action: str
+    reason: str
+
+    @property
+    def needs_prompt(self) -> bool:
+        return self.action in {"insert", "replace"}
+
+
+@dataclass
+class PromptArtifact:
+    notebook_path: Path
+    prompt_path: Path
+    action: str
+    reason: str
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def utc_timestamp() -> str:
+    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Generate Stage 1 notebook-enrichment prompts for an explicit notebook list.",
+    )
+    parser.add_argument(
+        "notebooks",
+        nargs="+",
+        help="Ordered notebook paths, relative to the repository root or absolute.",
+    )
+    parser.add_argument(
+        "--report-path",
+        help="Optional explicit markdown report path. Defaults to notes/ with a UTC timestamp.",
+    )
+    parser.add_argument(
+        "--overwrite-generated",
+        action="store_true",
+        help="Allow overwriting an existing generated Stage 1 prompt file for the same notebook.",
+    )
+    return parser.parse_args()
+
+
+def resolve_notebook_path(root: Path, notebook_arg: str) -> Path:
+    path = Path(notebook_arg)
+    if not path.is_absolute():
+        path = root / path
+    return path.resolve()
+
+
+def resolve_output_path(root: Path, output_arg: str) -> Path:
+    path = Path(output_arg)
+    if not path.is_absolute():
+        path = root / path
+    return path.resolve()
+
+
+def relative_repo_path(root: Path, path: Path) -> str:
+    return path.relative_to(root).as_posix()
+
+
+def classify_stage1_decision(inventory: NotebookInventory) -> Stage1Decision:
+    status = inventory.chapter_intro_status
+
+    if status == "substantive":
+        return Stage1Decision(
+            inventory=inventory,
+            action="skip",
+            reason="scanner classified the existing chapter intro as substantive",
+        )
+
+    if status == "missing":
+        return Stage1Decision(
+            inventory=inventory,
+            action="insert",
+            reason="scanner found no eligible chapter intro prose cell before the setup boundary",
+        )
+
+    if status in {"heading", "thin"} and inventory.chapter_intro_index >= 0:
+        return Stage1Decision(
+            inventory=inventory,
+            action="replace",
+            reason=(
+                "scanner found an existing intro candidate that is not yet substantive "
+                f"({status})"
+            ),
+        )
+
+    return Stage1Decision(
+        inventory=inventory,
+        action="skip",
+        reason="scanner evidence did not support one deterministic Stage 1 insert-or-replace target",
+    )
+
+
+def prompt_filename_for(root: Path, notebook_path: Path) -> str:
+    relative = notebook_path.relative_to(root).with_suffix("")
+    slug = "__".join(relative.parts)
+    return f"{GENERATED_PROMPT_PREFIX}{slug}.md"
+
+
+def write_prompt_file(
+    *,
+    root: Path,
+    decision: Stage1Decision,
+    overwrite_generated: bool,
+) -> PromptArtifact:
+    prompts_dir = root / PROMPTS_DIR
+    prompts_dir.mkdir(parents=True, exist_ok=True)
+
+    prompt_path = prompts_dir / prompt_filename_for(root, decision.inventory.path)
+    if prompt_path.exists() and not overwrite_generated:
+        raise FileExistsError(
+            f"generated prompt already exists: {relative_repo_path(root, prompt_path)} "
+            "(rerun with --overwrite-generated to replace it)"
+        )
+
+    prompt_text = build_stage1_prompt(root, decision.inventory)
+    prompt_path.write_text(prompt_text, encoding="utf-8")
+
+    return PromptArtifact(
+        notebook_path=decision.inventory.path,
+        prompt_path=prompt_path,
+        action=decision.action,
+        reason=decision.reason,
+    )
+
+
+def default_report_path(root: Path) -> Path:
+    return root / NOTES_DIR / f"{REPORT_PREFIX}{utc_timestamp()}.md"
+
+
+def build_report_text(
+    *,
+    root: Path,
+    notebook_paths: list[Path],
+    generated: list[PromptArtifact],
+    skipped: list[Stage1Decision],
+) -> str:
+    lines: list[str] = [
+        "# Stage 1 Wrapper MVP Report",
+        "",
+        "## Executive Summary",
+        "",
+        (
+            f"Requested {len(notebook_paths)} notebook(s). Generated {len(generated)} "
+            f"Stage 1 prompt(s) and skipped {len(skipped)} notebook(s)."
+        ),
+        "This wrapper run stopped at prompt generation and did not invoke `tools/codex/run_prompt.py`.",
+        "",
+        "## Target Notebook List",
+        "",
+    ]
+
+    for idx, notebook_path in enumerate(notebook_paths, start=1):
+        lines.append(f"{idx}. `{relative_repo_path(root, notebook_path)}`")
+
+    lines.extend(
+        [
+            "",
+            "## Generated Prompts",
+            "",
+        ]
+    )
+
+    if generated:
+        for artifact in generated:
+            lines.append(
+                (
+                    f"- `{relative_repo_path(root, artifact.notebook_path)}`: generated "
+                    f"`{relative_repo_path(root, artifact.prompt_path)}` "
+                    f"because action is `{artifact.action}` and {artifact.reason}."
+                )
+            )
+    else:
+        lines.append("- None.")
+
+    lines.extend(
+        [
+            "",
+            "## Skipped Notebooks",
+            "",
+        ]
+    )
+
+    if skipped:
+        for decision in skipped:
+            inv = decision.inventory
+            lines.append(
+                (
+                    f"- `{relative_repo_path(root, inv.path)}`: skipped because action is "
+                    f"`{decision.action}` and {decision.reason}. "
+                    f"(intro_status=`{inv.chapter_intro_status}`, "
+                    f"intro_index=`{inv.chapter_intro_index}`, "
+                    f"setup_index=`{inv.setup_cell_index}`)"
+                )
+            )
+    else:
+        lines.append("- None.")
+
+    lines.extend(
+        [
+            "",
+            "## Deferred Work",
+            "",
+            "- No generated prompts were executed in this pass.",
+            "- No notebooks were modified in this pass.",
+            "- Stage 2 and Stage 3 remain out of scope.",
+        ]
+    )
+
+    return "\n".join(lines) + "\n"
+
+
+def main() -> int:
+    args = parse_args()
+    root = repo_root()
+
+    notebook_paths: list[Path] = []
+    for notebook_arg in args.notebooks:
+        notebook_path = resolve_notebook_path(root, notebook_arg)
+        if not notebook_path.exists():
+            print(f"ERROR: notebook not found: {notebook_arg}", file=sys.stderr)
+            return 1
+        try:
+            notebook_path.relative_to(root)
+        except ValueError:
+            print(
+                f"ERROR: notebook path is outside the repository root: {notebook_path}",
+                file=sys.stderr,
+            )
+            return 1
+        if notebook_path.suffix != ".ipynb":
+            print(f"ERROR: not a notebook path: {notebook_arg}", file=sys.stderr)
+            return 1
+        notebook_paths.append(notebook_path)
+
+    generated: list[PromptArtifact] = []
+    skipped: list[Stage1Decision] = []
+
+    for notebook_path in notebook_paths:
+        inventory = scan_notebook(notebook_path)
+        decision = classify_stage1_decision(inventory)
+        if decision.needs_prompt:
+            artifact = write_prompt_file(
+                root=root,
+                decision=decision,
+                overwrite_generated=args.overwrite_generated,
+            )
+            generated.append(artifact)
+            print(
+                f"GENERATED {relative_repo_path(root, artifact.prompt_path)} "
+                f"for {relative_repo_path(root, notebook_path)} "
+                f"({decision.action})"
+            )
+        else:
+            skipped.append(decision)
+            print(
+                f"SKIPPED {relative_repo_path(root, notebook_path)} "
+                f"({decision.reason})"
+            )
+
+    report_path = resolve_output_path(root, args.report_path) if args.report_path else default_report_path(root)
+    report_path.parent.mkdir(parents=True, exist_ok=True)
+    report_text = build_report_text(
+        root=root,
+        notebook_paths=notebook_paths,
+        generated=generated,
+        skipped=skipped,
+    )
+    report_path.write_text(report_text, encoding="utf-8")
+    print(f"REPORT {relative_repo_path(root, report_path)}")
+
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

diff --git a/tools/notebook_enricher/stage1_prompt_wrapper.py b/tools/notebook_enricher/stage1_prompt_wrapper.py
new file mode 100644
index 0000000000000000000000000000000000000000..ccef40439f9d87bde89c38ad4b3386256b87f94b
--- /dev/null
+++ b/tools/notebook_enricher/stage1_prompt_wrapper.py
@@ -0,0 +1,303 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+from dataclasses import dataclass
+from pathlib import Path
+import sys
+
+from notebook_scanner import NotebookInventory, scan_notebook
+from prompt_builder import build_stage1_prompt
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+GENERATED_PROMPT_PREFIX = "generated_stage1__"
+REPORT_PREFIX = "031_stage1_wrapper_mvp_report__"
+
+
+@dataclass
+class Stage1Decision:
+    inventory: NotebookInventory
+    action: str
+    reason: str
+
+    @property
+    def needs_prompt(self) -> bool:
+        return self.action in {"insert", "replace"}
+
+
+@dataclass
+class PromptArtifact:
+    notebook_path: Path
+    prompt_path: Path
+    action: str
+    reason: str
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def utc_timestamp() -> str:
+    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Generate Stage 1 notebook-enrichment prompts for an explicit notebook list.",
+    )
+    parser.add_argument(
+        "notebooks",
+        nargs="+",
+        help="Ordered notebook paths, relative to the repository root or absolute.",
+    )
+    parser.add_argument(
+        "--report-path",
+        help="Optional explicit markdown report path. Defaults to notes/ with a UTC timestamp.",
+    )
+    parser.add_argument(
+        "--overwrite-generated",
+        action="store_true",
+        help="Allow overwriting an existing generated Stage 1 prompt file for the same notebook.",
+    )
+    return parser.parse_args()
+
+
+def resolve_notebook_path(root: Path, notebook_arg: str) -> Path:
+    path = Path(notebook_arg)
+    if not path.is_absolute():
+        path = root / path
+    return path.resolve()
+
+
+def resolve_output_path(root: Path, output_arg: str) -> Path:
+    path = Path(output_arg)
+    if not path.is_absolute():
+        path = root / path
+    return path.resolve()
+
+
+def relative_repo_path(root: Path, path: Path) -> str:
+    return path.relative_to(root).as_posix()
+
+
+def classify_stage1_decision(inventory: NotebookInventory) -> Stage1Decision:
+    status = inventory.chapter_intro_status
+
+    if status == "substantive":
+        return Stage1Decision(
+            inventory=inventory,
+            action="skip",
+            reason="scanner classified the existing chapter intro as substantive",
+        )
+
+    if status == "missing":
+        return Stage1Decision(
+            inventory=inventory,
+            action="insert",
+            reason="scanner found no eligible chapter intro prose cell before the setup boundary",
+        )
+
+    if status in {"heading", "thin"} and inventory.chapter_intro_index >= 0:
+        return Stage1Decision(
+            inventory=inventory,
+            action="replace",
+            reason=(
+                "scanner found an existing intro candidate that is not yet substantive "
+                f"({status})"
+            ),
+        )
+
+    return Stage1Decision(
+        inventory=inventory,
+        action="skip",
+        reason="scanner evidence did not support one deterministic Stage 1 insert-or-replace target",
+    )
+
+
+def prompt_filename_for(root: Path, notebook_path: Path) -> str:
+    relative = notebook_path.relative_to(root).with_suffix("")
+    slug = "__".join(relative.parts)
+    return f"{GENERATED_PROMPT_PREFIX}{slug}.md"
+
+
+def write_prompt_file(
+    *,
+    root: Path,
+    decision: Stage1Decision,
+    overwrite_generated: bool,
+) -> PromptArtifact:
+    prompts_dir = root / PROMPTS_DIR
+    prompts_dir.mkdir(parents=True, exist_ok=True)
+
+    prompt_path = prompts_dir / prompt_filename_for(root, decision.inventory.path)
+    if prompt_path.exists() and not overwrite_generated:
+        raise FileExistsError(
+            f"generated prompt already exists: {relative_repo_path(root, prompt_path)} "
+            "(rerun with --overwrite-generated to replace it)"
+        )
+
+    prompt_text = build_stage1_prompt(root, decision.inventory)
+    prompt_path.write_text(prompt_text, encoding="utf-8")
+
+    return PromptArtifact(
+        notebook_path=decision.inventory.path,
+        prompt_path=prompt_path,
+        action=decision.action,
+        reason=decision.reason,
+    )
+
+
+def default_report_path(root: Path) -> Path:
+    return root / NOTES_DIR / f"{REPORT_PREFIX}{utc_timestamp()}.md"
+
+
+def build_report_text(
+    *,
+    root: Path,
+    notebook_paths: list[Path],
+    generated: list[PromptArtifact],
+    skipped: list[Stage1Decision],
+) -> str:
+    lines: list[str] = [
+        "# Stage 1 Wrapper MVP Report",
+        "",
+        "## Executive Summary",
+        "",
+        (
+            f"Requested {len(notebook_paths)} notebook(s). Generated {len(generated)} "
+            f"Stage 1 prompt(s) and skipped {len(skipped)} notebook(s)."
+        ),
+        "This wrapper run stopped at prompt generation and did not invoke `tools/codex/run_prompt.py`.",
+        "",
+        "## Target Notebook List",
+        "",
+    ]
+
+    for idx, notebook_path in enumerate(notebook_paths, start=1):
+        lines.append(f"{idx}. `{relative_repo_path(root, notebook_path)}`")
+
+    lines.extend(
+        [
+            "",
+            "## Generated Prompts",
+            "",
+        ]
+    )
+
+    if generated:
+        for artifact in generated:
+            lines.append(
+                (
+                    f"- `{relative_repo_path(root, artifact.notebook_path)}`: generated "
+                    f"`{relative_repo_path(root, artifact.prompt_path)}` "
+                    f"because action is `{artifact.action}` and {artifact.reason}."
+                )
+            )
+    else:
+        lines.append("- None.")
+
+    lines.extend(
+        [
+            "",
+            "## Skipped Notebooks",
+            "",
+        ]
+    )
+
+    if skipped:
+        for decision in skipped:
+            inv = decision.inventory
+            lines.append(
+                (
+                    f"- `{relative_repo_path(root, inv.path)}`: skipped because action is "
+                    f"`{decision.action}` and {decision.reason}. "
+                    f"(intro_status=`{inv.chapter_intro_status}`, "
+                    f"intro_index=`{inv.chapter_intro_index}`, "
+                    f"setup_index=`{inv.setup_cell_index}`)"
+                )
+            )
+    else:
+        lines.append("- None.")
+
+    lines.extend(
+        [
+            "",
+            "## Deferred Work",
+            "",
+            "- No generated prompts were executed in this pass.",
+            "- No notebooks were modified in this pass.",
+            "- Stage 2 and Stage 3 remain out of scope.",
+        ]
+    )
+
+    return "\n".join(lines) + "\n"
+
+
+def main() -> int:
+    args = parse_args()
+    root = repo_root()
+
+    notebook_paths: list[Path] = []
+    for notebook_arg in args.notebooks:
+        notebook_path = resolve_notebook_path(root, notebook_arg)
+        if not notebook_path.exists():
+            print(f"ERROR: notebook not found: {notebook_arg}", file=sys.stderr)
+            return 1
+        try:
+            notebook_path.relative_to(root)
+        except ValueError:
+            print(
+                f"ERROR: notebook path is outside the repository root: {notebook_path}",
+                file=sys.stderr,
+            )
+            return 1
+        if notebook_path.suffix != ".ipynb":
+            print(f"ERROR: not a notebook path: {notebook_arg}", file=sys.stderr)
+            return 1
+        notebook_paths.append(notebook_path)
+
+    generated: list[PromptArtifact] = []
+    skipped: list[Stage1Decision] = []
+
+    for notebook_path in notebook_paths:
+        inventory = scan_notebook(notebook_path)
+        decision = classify_stage1_decision(inventory)
+        if decision.needs_prompt:
+            artifact = write_prompt_file(
+                root=root,
+                decision=decision,
+                overwrite_generated=args.overwrite_generated,
+            )
+            generated.append(artifact)
+            print(
+                f"GENERATED {relative_repo_path(root, artifact.prompt_path)} "
+                f"for {relative_repo_path(root, notebook_path)} "
+                f"({decision.action})"
+            )
+        else:
+            skipped.append(decision)
+            print(
+                f"SKIPPED {relative_repo_path(root, notebook_path)} "
+                f"({decision.reason})"
+            )
+
+    report_path = resolve_output_path(root, args.report_path) if args.report_path else default_report_path(root)
+    report_path.parent.mkdir(parents=True, exist_ok=True)
+    report_text = build_report_text(
+        root=root,
+        notebook_paths=notebook_paths,
+        generated=generated,
+        skipped=skipped,
+    )
+    report_path.write_text(report_text, encoding="utf-8")
+    print(f"REPORT {relative_repo_path(root, report_path)}")
+
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

exec
/bin/bash -lc 'git diff --stat' in /home/jevert/git/handson-ml3
 succeeded in 0ms:
exec
/bin/bash -lc 'python3 tools/notebook_enricher/stage1_prompt_wrapper.py --help' in /home/jevert/git/handson-ml3
 succeeded in 0ms:
usage: stage1_prompt_wrapper.py [-h] [--report-path REPORT_PATH]
                                [--overwrite-generated]
                                notebooks [notebooks ...]

Generate Stage 1 notebook-enrichment prompts for an explicit notebook list.

positional arguments:
  notebooks             Ordered notebook paths, relative to the repository
                        root or absolute.

options:
  -h, --help            show this help message and exit
  --report-path REPORT_PATH
                        Optional explicit markdown report path. Defaults to
                        notes/ with a UTC timestamp.
  --overwrite-generated
                        Allow overwriting an existing generated Stage 1 prompt
                        file for the same notebook.

diff --git a/tools/notebook_enricher/stage1_prompt_wrapper.py b/tools/notebook_enricher/stage1_prompt_wrapper.py
new file mode 100644
index 0000000000000000000000000000000000000000..ccef40439f9d87bde89c38ad4b3386256b87f94b
--- /dev/null
+++ b/tools/notebook_enricher/stage1_prompt_wrapper.py
@@ -0,0 +1,303 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+from dataclasses import dataclass
+from pathlib import Path
+import sys
+
+from notebook_scanner import NotebookInventory, scan_notebook
+from prompt_builder import build_stage1_prompt
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+GENERATED_PROMPT_PREFIX = "generated_stage1__"
+REPORT_PREFIX = "031_stage1_wrapper_mvp_report__"
+
+
+@dataclass
+class Stage1Decision:
+    inventory: NotebookInventory
+    action: str
+    reason: str
+
+    @property
+    def needs_prompt(self) -> bool:
+        return self.action in {"insert", "replace"}
+
+
+@dataclass
+class PromptArtifact:
+    notebook_path: Path
+    prompt_path: Path
+    action: str
+    reason: str
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def utc_timestamp() -> str:
+    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Generate Stage 1 notebook-enrichment prompts for an explicit notebook list.",
+    )
+    parser.add_argument(
+        "notebooks",
+        nargs="+",
+        help="Ordered notebook paths, relative to the repository root or absolute.",
+    )
+    parser.add_argument(
+        "--report-path",
+        help="Optional explicit markdown report path. Defaults to notes/ with a UTC timestamp.",
+    )
+    parser.add_argument(
+        "--overwrite-generated",
+        action="store_true",
+        help="Allow overwriting an existing generated Stage 1 prompt file for the same notebook.",
+    )
+    return parser.parse_args()
+
+
+def resolve_notebook_path(root: Path, notebook_arg: str) -> Path:
+    path = Path(notebook_arg)
+    if not path.is_absolute():
+        path = root / path
+    return path.resolve()
+
+
+def resolve_output_path(root: Path, output_arg: str) -> Path:
+    path = Path(output_arg)
+    if not path.is_absolute():
+        path = root / path
+    return path.resolve()
+
+
+def relative_repo_path(root: Path, path: Path) -> str:
+    return path.relative_to(root).as_posix()
+
+
+def classify_stage1_decision(inventory: NotebookInventory) -> Stage1Decision:
+    status = inventory.chapter_intro_status
+
+    if status == "substantive":
+        return Stage1Decision(
+            inventory=inventory,
+            action="skip",
+            reason="scanner classified the existing chapter intro as substantive",
+        )
+
+    if status == "missing":
+        return Stage1Decision(
+            inventory=inventory,
+            action="insert",
+            reason="scanner found no eligible chapter intro prose cell before the setup boundary",
+        )
+
+    if status in {"heading", "thin"} and inventory.chapter_intro_index >= 0:
+        return Stage1Decision(
+            inventory=inventory,
+            action="replace",
+            reason=(
+                "scanner found an existing intro candidate that is not yet substantive "
+                f"({status})"
+            ),
+        )
+
+    return Stage1Decision(
+        inventory=inventory,
+        action="skip",
+        reason="scanner evidence did not support one deterministic Stage 1 insert-or-replace target",
+    )
+
+
+def prompt_filename_for(root: Path, notebook_path: Path) -> str:
+    relative = notebook_path.relative_to(root).with_suffix("")
+    slug = "__".join(relative.parts)
+    return f"{GENERATED_PROMPT_PREFIX}{slug}.md"
+
+
+def write_prompt_file(
+    *,
+    root: Path,
+    decision: Stage1Decision,
+    overwrite_generated: bool,
+) -> PromptArtifact:
+    prompts_dir = root / PROMPTS_DIR
+    prompts_dir.mkdir(parents=True, exist_ok=True)
+
+    prompt_path = prompts_dir / prompt_filename_for(root, decision.inventory.path)
+    if prompt_path.exists() and not overwrite_generated:
+        raise FileExistsError(
+            f"generated prompt already exists: {relative_repo_path(root, prompt_path)} "
+            "(rerun with --overwrite-generated to replace it)"
+        )
+
+    prompt_text = build_stage1_prompt(root, decision.inventory)
+    prompt_path.write_text(prompt_text, encoding="utf-8")
+
+    return PromptArtifact(
+        notebook_path=decision.inventory.path,
+        prompt_path=prompt_path,
+        action=decision.action,
+        reason=decision.reason,
+    )
+
+
+def default_report_path(root: Path) -> Path:
+    return root / NOTES_DIR / f"{REPORT_PREFIX}{utc_timestamp()}.md"
+
+
+def build_report_text(
+    *,
+    root: Path,
+    notebook_paths: list[Path],
+    generated: list[PromptArtifact],
+    skipped: list[Stage1Decision],
+) -> str:
+    lines: list[str] = [
+        "# Stage 1 Wrapper MVP Report",
+        "",
+        "## Executive Summary",
+        "",
+        (
+            f"Requested {len(notebook_paths)} notebook(s). Generated {len(generated)} "
+            f"Stage 1 prompt(s) and skipped {len(skipped)} notebook(s)."
+        ),
+        "This wrapper run stopped at prompt generation and did not invoke `tools/codex/run_prompt.py`.",
+        "",
+        "## Target Notebook List",
+        "",
+    ]
+
+    for idx, notebook_path in enumerate(notebook_paths, start=1):
+        lines.append(f"{idx}. `{relative_repo_path(root, notebook_path)}`")
+
+    lines.extend(
+        [
+            "",
+            "## Generated Prompts",
+            "",
+        ]
+    )
+
+    if generated:
+        for artifact in generated:
+            lines.append(
+                (
+                    f"- `{relative_repo_path(root, artifact.notebook_path)}`: generated "
+                    f"`{relative_repo_path(root, artifact.prompt_path)}` "
+                    f"because action is `{artifact.action}` and {artifact.reason}."
+                )
+            )
+    else:
+        lines.append("- None.")
+
+    lines.extend(
+        [
+            "",
+            "## Skipped Notebooks",
+            "",
+        ]
+    )
+
+    if skipped:
+        for decision in skipped:
+            inv = decision.inventory
+            lines.append(
+                (
+                    f"- `{relative_repo_path(root, inv.path)}`: skipped because action is "
+                    f"`{decision.action}` and {decision.reason}. "
+                    f"(intro_status=`{inv.chapter_intro_status}`, "
+                    f"intro_index=`{inv.chapter_intro_index}`, "
+                    f"setup_index=`{inv.setup_cell_index}`)"
+                )
+            )
+    else:
+        lines.append("- None.")
+
+    lines.extend(
+        [
+            "",
+            "## Deferred Work",
+            "",
+            "- No generated prompts were executed in this pass.",
+            "- No notebooks were modified in this pass.",
+            "- Stage 2 and Stage 3 remain out of scope.",
+        ]
+    )
+
+    return "\n".join(lines) + "\n"
+
+
+def main() -> int:
+    args = parse_args()
+    root = repo_root()
+
+    notebook_paths: list[Path] = []
+    for notebook_arg in args.notebooks:
+        notebook_path = resolve_notebook_path(root, notebook_arg)
+        if not notebook_path.exists():
+            print(f"ERROR: notebook not found: {notebook_arg}", file=sys.stderr)
+            return 1
+        try:
+            notebook_path.relative_to(root)
+        except ValueError:
+            print(
+                f"ERROR: notebook path is outside the repository root: {notebook_path}",
+                file=sys.stderr,
+            )
+            return 1
+        if notebook_path.suffix != ".ipynb":
+            print(f"ERROR: not a notebook path: {notebook_arg}", file=sys.stderr)
+            return 1
+        notebook_paths.append(notebook_path)
+
+    generated: list[PromptArtifact] = []
+    skipped: list[Stage1Decision] = []
+
+    for notebook_path in notebook_paths:
+        inventory = scan_notebook(notebook_path)
+        decision = classify_stage1_decision(inventory)
+        if decision.needs_prompt:
+            artifact = write_prompt_file(
+                root=root,
+                decision=decision,
+                overwrite_generated=args.overwrite_generated,
+            )
+            generated.append(artifact)
+            print(
+                f"GENERATED {relative_repo_path(root, artifact.prompt_path)} "
+                f"for {relative_repo_path(root, notebook_path)} "
+                f"({decision.action})"
+            )
+        else:
+            skipped.append(decision)
+            print(
+                f"SKIPPED {relative_repo_path(root, notebook_path)} "
+                f"({decision.reason})"
+            )
+
+    report_path = resolve_output_path(root, args.report_path) if args.report_path else default_report_path(root)
+    report_path.parent.mkdir(parents=True, exist_ok=True)
+    report_text = build_report_text(
+        root=root,
+        notebook_paths=notebook_paths,
+        generated=generated,
+        skipped=skipped,
+    )
+    report_path.write_text(report_text, encoding="utf-8")
+    print(f"REPORT {relative_repo_path(root, report_path)}")
+
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

codex
The CLI shape checks out. I’m running the wrapper on a two-notebook scope now: one notebook the scanner marks for Stage 1 generation and one already-substantive notebook that should be skipped, which gives the narrowest useful end-to-end validation for this slice.
exec
/bin/bash -lc 'python3 tools/notebook_enricher/stage1_prompt_wrapper.py 05_support_vector_machines.ipynb 06_decision_trees.ipynb --report-path notes/031_stage1_wrapper_mvp_report__20260417_000000.md' in /home/jevert/git/handson-ml3
 succeeded in 0ms:
GENERATED codex_prompts/generated_stage1__05_support_vector_machines.md for 05_support_vector_machines.ipynb (insert)
SKIPPED 06_decision_trees.ipynb (scanner classified the existing chapter intro as substantive)
REPORT notes/031_stage1_wrapper_mvp_report__20260417_000000.md

exec
/bin/bash -lc 'git status --short' in /home/jevert/git/handson-ml3
 succeeded in 0ms:
?? codex_prompts/031_build_stage1_wrapper_mvp_from_contracts.md
?? codex_prompts/generated_stage1__05_support_vector_machines.md
?? notes/031_stage1_wrapper_mvp_report__20260417_000000.md
?? tools/notebook_enricher/stage1_prompt_wrapper.py

diff --git a/tools/notebook_enricher/stage1_prompt_wrapper.py b/tools/notebook_enricher/stage1_prompt_wrapper.py
new file mode 100644
index 0000000000000000000000000000000000000000..ccef40439f9d87bde89c38ad4b3386256b87f94b
--- /dev/null
+++ b/tools/notebook_enricher/stage1_prompt_wrapper.py
@@ -0,0 +1,303 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+from dataclasses import dataclass
+from pathlib import Path
+import sys
+
+from notebook_scanner import NotebookInventory, scan_notebook
+from prompt_builder import build_stage1_prompt
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+GENERATED_PROMPT_PREFIX = "generated_stage1__"
+REPORT_PREFIX = "031_stage1_wrapper_mvp_report__"
+
+
+@dataclass
+class Stage1Decision:
+    inventory: NotebookInventory
+    action: str
+    reason: str
+
+    @property
+    def needs_prompt(self) -> bool:
+        return self.action in {"insert", "replace"}
+
+
+@dataclass
+class PromptArtifact:
+    notebook_path: Path
+    prompt_path: Path
+    action: str
+    reason: str
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def utc_timestamp() -> str:
+    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Generate Stage 1 notebook-enrichment prompts for an explicit notebook list.",
+    )
+    parser.add_argument(
+        "notebooks",
+        nargs="+",
+        help="Ordered notebook paths, relative to the repository root or absolute.",
+    )
+    parser.add_argument(
+        "--report-path",
+        help="Optional explicit markdown report path. Defaults to notes/ with a UTC timestamp.",
+    )
+    parser.add_argument(
+        "--overwrite-generated",
+        action="store_true",
+        help="Allow overwriting an existing generated Stage 1 prompt file for the same notebook.",
+    )
+    return parser.parse_args()
+
+
+def resolve_notebook_path(root: Path, notebook_arg: str) -> Path:
+    path = Path(notebook_arg)
+    if not path.is_absolute():
+        path = root / path
+    return path.resolve()
+
+
+def resolve_output_path(root: Path, output_arg: str) -> Path:
+    path = Path(output_arg)
+    if not path.is_absolute():
+        path = root / path
+    return path.resolve()
+
+
+def relative_repo_path(root: Path, path: Path) -> str:
+    return path.relative_to(root).as_posix()
+
+
+def classify_stage1_decision(inventory: NotebookInventory) -> Stage1Decision:
+    status = inventory.chapter_intro_status
+
+    if status == "substantive":
+        return Stage1Decision(
+            inventory=inventory,
+            action="skip",
+            reason="scanner classified the existing chapter intro as substantive",
+        )
+
+    if status == "missing":
+        return Stage1Decision(
+            inventory=inventory,
+            action="insert",
+            reason="scanner found no eligible chapter intro prose cell before the setup boundary",
+        )
+
+    if status in {"heading", "thin"} and inventory.chapter_intro_index >= 0:
+        return Stage1Decision(
+            inventory=inventory,
+            action="replace",
+            reason=(
+                "scanner found an existing intro candidate that is not yet substantive "
+                f"({status})"
+            ),
+        )
+
+    return Stage1Decision(
+        inventory=inventory,
+        action="skip",
+        reason="scanner evidence did not support one deterministic Stage 1 insert-or-replace target",
+    )
+
+
+def prompt_filename_for(root: Path, notebook_path: Path) -> str:
+    relative = notebook_path.relative_to(root).with_suffix("")
+    slug = "__".join(relative.parts)
+    return f"{GENERATED_PROMPT_PREFIX}{slug}.md"
+
+
+def write_prompt_file(
+    *,
+    root: Path,
+    decision: Stage1Decision,
+    overwrite_generated: bool,
+) -> PromptArtifact:
+    prompts_dir = root / PROMPTS_DIR
+    prompts_dir.mkdir(parents=True, exist_ok=True)
+
+    prompt_path = prompts_dir / prompt_filename_for(root, decision.inventory.path)
+    if prompt_path.exists() and not overwrite_generated:
+        raise FileExistsError(
+            f"generated prompt already exists: {relative_repo_path(root, prompt_path)} "
+            "(rerun with --overwrite-generated to replace it)"
+        )
+
+    prompt_text = build_stage1_prompt(root, decision.inventory)
+    prompt_path.write_text(prompt_text, encoding="utf-8")
+
+    return PromptArtifact(
+        notebook_path=decision.inventory.path,
+        prompt_path=prompt_path,
+        action=decision.action,
+        reason=decision.reason,
+    )
+
+
+def default_report_path(root: Path) -> Path:
+    return root / NOTES_DIR / f"{REPORT_PREFIX}{utc_timestamp()}.md"
+
+
+def build_report_text(
+    *,
+    root: Path,
+    notebook_paths: list[Path],
+    generated: list[PromptArtifact],
+    skipped: list[Stage1Decision],
+) -> str:
+    lines: list[str] = [
+        "# Stage 1 Wrapper MVP Report",
+        "",
+        "## Executive Summary",
+        "",
+        (
+            f"Requested {len(notebook_paths)} notebook(s). Generated {len(generated)} "
+            f"Stage 1 prompt(s) and skipped {len(skipped)} notebook(s)."
+        ),
+        "This wrapper run stopped at prompt generation and did not invoke `tools/codex/run_prompt.py`.",
+        "",
+        "## Target Notebook List",
+        "",
+    ]
+
+    for idx, notebook_path in enumerate(notebook_paths, start=1):
+        lines.append(f"{idx}. `{relative_repo_path(root, notebook_path)}`")
+
+    lines.extend(
+        [
+            "",
+            "## Generated Prompts",
+            "",
+        ]
+    )
+
+    if generated:
+        for artifact in generated:
+            lines.append(
+                (
+                    f"- `{relative_repo_path(root, artifact.notebook_path)}`: generated "
+                    f"`{relative_repo_path(root, artifact.prompt_path)}` "
+                    f"because action is `{artifact.action}` and {artifact.reason}."
+                )
+            )
+    else:
+        lines.append("- None.")
+
+    lines.extend(
+        [
+            "",
+            "## Skipped Notebooks",
+            "",
+        ]
+    )
+
+    if skipped:
+        for decision in skipped:
+            inv = decision.inventory
+            lines.append(
+                (
+                    f"- `{relative_repo_path(root, inv.path)}`: skipped because action is "
+                    f"`{decision.action}` and {decision.reason}. "
+                    f"(intro_status=`{inv.chapter_intro_status}`, "
+                    f"intro_index=`{inv.chapter_intro_index}`, "
+                    f"setup_index=`{inv.setup_cell_index}`)"
+                )
+            )
+    else:
+        lines.append("- None.")
+
+    lines.extend(
+        [
+            "",
+            "## Deferred Work",
+            "",
+            "- No generated prompts were executed in this pass.",
+            "- No notebooks were modified in this pass.",
+            "- Stage 2 and Stage 3 remain out of scope.",
+        ]
+    )
+
+    return "\n".join(lines) + "\n"
+
+
+def main() -> int:
+    args = parse_args()
+    root = repo_root()
+
+    notebook_paths: list[Path] = []
+    for notebook_arg in args.notebooks:
+        notebook_path = resolve_notebook_path(root, notebook_arg)
+        if not notebook_path.exists():
+            print(f"ERROR: notebook not found: {notebook_arg}", file=sys.stderr)
+            return 1
+        try:
+            notebook_path.relative_to(root)
+        except ValueError:
+            print(
+                f"ERROR: notebook path is outside the repository root: {notebook_path}",
+                file=sys.stderr,
+            )
+            return 1
+        if notebook_path.suffix != ".ipynb":
+            print(f"ERROR: not a notebook path: {notebook_arg}", file=sys.stderr)
+            return 1
+        notebook_paths.append(notebook_path)
+
+    generated: list[PromptArtifact] = []
+    skipped: list[Stage1Decision] = []
+
+    for notebook_path in notebook_paths:
+        inventory = scan_notebook(notebook_path)
+        decision = classify_stage1_decision(inventory)
+        if decision.needs_prompt:
+            artifact = write_prompt_file(
+                root=root,
+                decision=decision,
+                overwrite_generated=args.overwrite_generated,
+            )
+            generated.append(artifact)
+            print(
+                f"GENERATED {relative_repo_path(root, artifact.prompt_path)} "
+                f"for {relative_repo_path(root, notebook_path)} "
+                f"({decision.action})"
+            )
+        else:
+            skipped.append(decision)
+            print(
+                f"SKIPPED {relative_repo_path(root, notebook_path)} "
+                f"({decision.reason})"
+            )
+
+    report_path = resolve_output_path(root, args.report_path) if args.report_path else default_report_path(root)
+    report_path.parent.mkdir(parents=True, exist_ok=True)
+    report_text = build_report_text(
+        root=root,
+        notebook_paths=notebook_paths,
+        generated=generated,
+        skipped=skipped,
+    )
+    report_path.write_text(report_text, encoding="utf-8")
+    print(f"REPORT {relative_repo_path(root, report_path)}")
+
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

exec
/bin/bash -lc "sed -n '1,260p' codex_prompts/generated_stage1__05_support_vector_machines.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
# Stage 1: Chapter Intro Enrichment

## Target Notebook
- Path: 05_support_vector_machines.ipynb
- Chapter: 5
- Notebook stem: 05_support_vector_machines

## Current Chapter Intro State
Status: MISSING — no chapter intro exists. Insert a new one.

Current content (if any):
```
(no existing intro)
```

## Your Task
INSERT a new markdown cell at index 3. Place it after the Colab/Kaggle links table and before the # Setup cell.

The new intro must follow the treatment specification below exactly.

## Treatment Specification

#### Treatment structure for the entire notebook:

```
[CHAPTER INTRO MARKDOWN CELL]
  - What is this chapter about?
  - What are the 3-5 main concepts a student should walk away understanding?
  - Why does this topic matter in the broader ML landscape?
  - Where does it sit relative to what came before and what comes next?
  - Any key vocabulary terms to know before diving in
  (Aim for 300-500 words. This is the "sit down, let me tell you what we're about to do" cell.)

Then for each logical section or code cell in the notebook:

[GOAL MARKDOWN CELL — before the code]
  - What is the goal of the next code block?
  - Why does this matter for ML? (Not just "it runs the model" — why do we care?)
  - What is this code doing that is a *better practice* worth noting?
  - What is this code doing that is just *plumbing* (necessary but not pedagogically deep)?
  (Aim for 4-8 sentences. Make the distinction between "this is important" and "this is boilerplate" explicit.)

[PYTHON CODE CELL]
  (unchanged)

[IMPLEMENTATION DETAIL MARKDOWN CELL — after the code]
  - What did we just see happen?
  - What are the implementation choices worth noticing? (e.g., why this hyperparameter,
    why this data split ratio, why this particular sklearn API call?)
  - What might go wrong here in practice, and how would you know?
  - If there is output, what should the student be looking for in that output?
  (Aim for 3-6 sentences. This is the "here's what's interesting about what we just ran" cell.)
```

## Gold Standard Example (from 06_decision_trees.ipynb — the finished notebook)

**Chapter Intro Example:**
## Chapter Overview: Decision Trees

Decision trees are one of the most intuitive and interpretable machine learning algorithms you will encounter. They make predictions by asking a sequence of yes/no questions about the input features, following branches down to a leaf node that contains the final answer. If you have ever played Twenty Questions, you already understand the core idea.

**What you should walk away understanding:**

1. **How decision trees make decisions** — the splitting criterion (Gini impurity and entropy), how the tree chooses which feature to split on at each node, and how it decides when to stop splitting.
2. **Regularization and the overfitting problem** — unconstrained trees will memorize training data perfectly and generalize poorly. Hyperparameters like `max_depth`...

**Goal/Code/Implementation Trio Example:**
### GOAL BEFORE CELL:
### Goal Before This Cell

**Goal:** Verify the Python environment and notebook prerequisites before doing any modeling.

**Why this matters for machine learning:** This cell contributes to the larger workflow of building, inspecting, evaluating, or explaining a decision tree model.

### CODE CELL:
```python
import sys

assert sys.version_info >= (3, 7)

#%pip install graphviz
#Note: Using %pip (with the percent sign) is better than !pip inside notebooks because it guarantees installation into the kernel's specific virtual environment rather than a global system Python.
```

### IMPLEMENTATION NOTES AFTER CELL:
### Implementation Notes After This Cell

Failing fast on environment problems is a practical ML skill because many notebook errors come from setup drift rather than model logic.

**Broader skill:** Being able to connect model behavior to code-level implementation details is one of the most valuable habits you can build in machine learning.

## Output Contract

1. Read the full notebook from: `05_support_vector_machines.ipynb`
2. INSERT a new markdown cell at index 3. Place it after the Colab/Kaggle links table and before the # Setup cell.
3. Write the COMPLETE modified notebook as valid JSON to: `05_support_vector_machines.ipynb.tmp`
   (write it to the same directory as the original notebook)
4. Do NOT modify any `code` cells — not their source, outputs, or metadata
5. Do NOT add, remove, or reorder any cell other than the one change described above
6. The new intro cell must have `"cell_type": "markdown"` and a `"source"` field

## Hard Constraints

- NEVER modify a cell with `"cell_type": "code"`
- The intro must be 300–500 words covering: what this chapter is about, 3–5 learning
  objectives, why this topic matters in ML, where it fits in the course, key vocabulary
- Write clean JSON — the same structure as the input notebook
- The output file must be a valid Jupyter notebook that can be opened in Jupyter Lab

exec
/bin/bash -lc "sed -n '1,240p' notes/031_stage1_wrapper_mvp_report__20260417_000000.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
# Stage 1 Wrapper MVP Report

## Executive Summary

Requested 2 notebook(s). Generated 1 Stage 1 prompt(s) and skipped 1 notebook(s).
This wrapper run stopped at prompt generation and did not invoke `tools/codex/run_prompt.py`.

## Target Notebook List

1. `05_support_vector_machines.ipynb`
2. `06_decision_trees.ipynb`

## Generated Prompts

- `05_support_vector_machines.ipynb`: generated `codex_prompts/generated_stage1__05_support_vector_machines.md` because action is `insert` and scanner found no eligible chapter intro prose cell before the setup boundary.

## Skipped Notebooks

- `06_decision_trees.ipynb`: skipped because action is `skip` and scanner classified the existing chapter intro as substantive. (intro_status=`substantive`, intro_index=`3`, setup_index=`4`)

## Deferred Work

- No generated prompts were executed in this pass.
- No notebooks were modified in this pass.
- Stage 2 and Stage 3 remain out of scope.

exec
/bin/bash -lc "git diff --name-only -- '*.ipynb'" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
diff --git a/tools/notebook_enricher/stage1_prompt_wrapper.py b/tools/notebook_enricher/stage1_prompt_wrapper.py
new file mode 100644
index 0000000000000000000000000000000000000000..ccef40439f9d87bde89c38ad4b3386256b87f94b
--- /dev/null
+++ b/tools/notebook_enricher/stage1_prompt_wrapper.py
@@ -0,0 +1,303 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+from dataclasses import dataclass
+from pathlib import Path
+import sys
+
+from notebook_scanner import NotebookInventory, scan_notebook
+from prompt_builder import build_stage1_prompt
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+GENERATED_PROMPT_PREFIX = "generated_stage1__"
+REPORT_PREFIX = "031_stage1_wrapper_mvp_report__"
+
+
+@dataclass
+class Stage1Decision:
+    inventory: NotebookInventory
+    action: str
+    reason: str
+
+    @property
+    def needs_prompt(self) -> bool:
+        return self.action in {"insert", "replace"}
+
+
+@dataclass
+class PromptArtifact:
+    notebook_path: Path
+    prompt_path: Path
+    action: str
+    reason: str
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def utc_timestamp() -> str:
+    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Generate Stage 1 notebook-enrichment prompts for an explicit notebook list.",
+    )
+    parser.add_argument(
+        "notebooks",
+        nargs="+",
+        help="Ordered notebook paths, relative to the repository root or absolute.",
+    )
+    parser.add_argument(
+        "--report-path",
+        help="Optional explicit markdown report path. Defaults to notes/ with a UTC timestamp.",
+    )
+    parser.add_argument(
+        "--overwrite-generated",
+        action="store_true",
+        help="Allow overwriting an existing generated Stage 1 prompt file for the same notebook.",
+    )
+    return parser.parse_args()
+
+
+def resolve_notebook_path(root: Path, notebook_arg: str) -> Path:
+    path = Path(notebook_arg)
+    if not path.is_absolute():
+        path = root / path
+    return path.resolve()
+
+
+def resolve_output_path(root: Path, output_arg: str) -> Path:
+    path = Path(output_arg)
+    if not path.is_absolute():
+        path = root / path
+    return path.resolve()
+
+
+def relative_repo_path(root: Path, path: Path) -> str:
+    return path.relative_to(root).as_posix()
+
+
+def classify_stage1_decision(inventory: NotebookInventory) -> Stage1Decision:
+    status = inventory.chapter_intro_status
+
+    if status == "substantive":
+        return Stage1Decision(
+            inventory=inventory,
+            action="skip",
+            reason="scanner classified the existing chapter intro as substantive",
+        )
+
+    if status == "missing":
+        return Stage1Decision(
+            inventory=inventory,
+            action="insert",
+            reason="scanner found no eligible chapter intro prose cell before the setup boundary",
+        )
+
+    if status in {"heading", "thin"} and inventory.chapter_intro_index >= 0:
+        return Stage1Decision(
+            inventory=inventory,
+            action="replace",
+            reason=(
+                "scanner found an existing intro candidate that is not yet substantive "
+                f"({status})"
+            ),
+        )
+
+    return Stage1Decision(
+        inventory=inventory,
+        action="skip",
+        reason="scanner evidence did not support one deterministic Stage 1 insert-or-replace target",
+    )
+
+
+def prompt_filename_for(root: Path, notebook_path: Path) -> str:
+    relative = notebook_path.relative_to(root).with_suffix("")
+    slug = "__".join(relative.parts)
+    return f"{GENERATED_PROMPT_PREFIX}{slug}.md"
+
+
+def write_prompt_file(
+    *,
+    root: Path,
+    decision: Stage1Decision,
+    overwrite_generated: bool,
+) -> PromptArtifact:
+    prompts_dir = root / PROMPTS_DIR
+    prompts_dir.mkdir(parents=True, exist_ok=True)
+
+    prompt_path = prompts_dir / prompt_filename_for(root, decision.inventory.path)
+    if prompt_path.exists() and not overwrite_generated:
+        raise FileExistsError(
+            f"generated prompt already exists: {relative_repo_path(root, prompt_path)} "
+            "(rerun with --overwrite-generated to replace it)"
+        )
+
+    prompt_text = build_stage1_prompt(root, decision.inventory)
+    prompt_path.write_text(prompt_text, encoding="utf-8")
+
+    return PromptArtifact(
+        notebook_path=decision.inventory.path,
+        prompt_path=prompt_path,
+        action=decision.action,
+        reason=decision.reason,
+    )
+
+
+def default_report_path(root: Path) -> Path:
+    return root / NOTES_DIR / f"{REPORT_PREFIX}{utc_timestamp()}.md"
+
+
+def build_report_text(
+    *,
+    root: Path,
+    notebook_paths: list[Path],
+    generated: list[PromptArtifact],
+    skipped: list[Stage1Decision],
+) -> str:
+    lines: list[str] = [
+        "# Stage 1 Wrapper MVP Report",
+        "",
+        "## Executive Summary",
+        "",
+        (
+            f"Requested {len(notebook_paths)} notebook(s). Generated {len(generated)} "
+            f"Stage 1 prompt(s) and skipped {len(skipped)} notebook(s)."
+        ),
+        "This wrapper run stopped at prompt generation and did not invoke `tools/codex/run_prompt.py`.",
+        "",
+        "## Target Notebook List",
+        "",
+    ]
+
+    for idx, notebook_path in enumerate(notebook_paths, start=1):
+        lines.append(f"{idx}. `{relative_repo_path(root, notebook_path)}`")
+
+    lines.extend(
+        [
+            "",
+            "## Generated Prompts",
+            "",
+        ]
+    )
+
+    if generated:
+        for artifact in generated:
+            lines.append(
+                (
+                    f"- `{relative_repo_path(root, artifact.notebook_path)}`: generated "
+                    f"`{relative_repo_path(root, artifact.prompt_path)}` "
+                    f"because action is `{artifact.action}` and {artifact.reason}."
+                )
+            )
+    else:
+        lines.append("- None.")
+
+    lines.extend(
+        [
+            "",
+            "## Skipped Notebooks",
+            "",
+        ]
+    )
+
+    if skipped:
+        for decision in skipped:
+            inv = decision.inventory
+            lines.append(
+                (
+                    f"- `{relative_repo_path(root, inv.path)}`: skipped because action is "
+                    f"`{decision.action}` and {decision.reason}. "
+                    f"(intro_status=`{inv.chapter_intro_status}`, "
+                    f"intro_index=`{inv.chapter_intro_index}`, "
+                    f"setup_index=`{inv.setup_cell_index}`)"
+                )
+            )
+    else:
+        lines.append("- None.")
+
+    lines.extend(
+        [
+            "",
+            "## Deferred Work",
+            "",
+            "- No generated prompts were executed in this pass.",
+            "- No notebooks were modified in this pass.",
+            "- Stage 2 and Stage 3 remain out of scope.",
+        ]
+    )
+
+    return "\n".join(lines) + "\n"
+
+
+def main() -> int:
+    args = parse_args()
+    root = repo_root()
+
+    notebook_paths: list[Path] = []
+    for notebook_arg in args.notebooks:
+        notebook_path = resolve_notebook_path(root, notebook_arg)
+        if not notebook_path.exists():
+            print(f"ERROR: notebook not found: {notebook_arg}", file=sys.stderr)
+            return 1
+        try:
+            notebook_path.relative_to(root)
+        except ValueError:
+            print(
+                f"ERROR: notebook path is outside the repository root: {notebook_path}",
+                file=sys.stderr,
+            )
+            return 1
+        if notebook_path.suffix != ".ipynb":
+            print(f"ERROR: not a notebook path: {notebook_arg}", file=sys.stderr)
+            return 1
+        notebook_paths.append(notebook_path)
+
+    generated: list[PromptArtifact] = []
+    skipped: list[Stage1Decision] = []
+
+    for notebook_path in notebook_paths:
+        inventory = scan_notebook(notebook_path)
+        decision = classify_stage1_decision(inventory)
+        if decision.needs_prompt:
+            artifact = write_prompt_file(
+                root=root,
+                decision=decision,
+                overwrite_generated=args.overwrite_generated,
+            )
+            generated.append(artifact)
+            print(
+                f"GENERATED {relative_repo_path(root, artifact.prompt_path)} "
+                f"for {relative_repo_path(root, notebook_path)} "
+                f"({decision.action})"
+            )
+        else:
+            skipped.append(decision)
+            print(
+                f"SKIPPED {relative_repo_path(root, notebook_path)} "
+                f"({decision.reason})"
+            )
+
+    report_path = resolve_output_path(root, args.report_path) if args.report_path else default_report_path(root)
+    report_path.parent.mkdir(parents=True, exist_ok=True)
+    report_text = build_report_text(
+        root=root,
+        notebook_paths=notebook_paths,
+        generated=generated,
+        skipped=skipped,
+    )
+    report_path.write_text(report_text, encoding="utf-8")
+    print(f"REPORT {relative_repo_path(root, report_path)}")
+
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

codex
Implemented a thin Stage 1-only wrapper at [tools/notebook_enricher/stage1_prompt_wrapper.py](/home/jevert/git/handson-ml3/tools/notebook_enricher/stage1_prompt_wrapper.py). It accepts an explicit ordered notebook list as positional args, scans each notebook first, classifies the Stage 1 action as `insert`, `replace`, or `skip`, writes Stage 1 prompt files only for notebooks that need treatment, and stops there without invoking `tools/codex/run_prompt.py`.

I validated it on `05_support_vector_machines.ipynb` and `06_decision_trees.ipynb`. That produced [codex_prompts/generated_stage1__05_support_vector_machines.md](/home/jevert/git/handson-ml3/codex_prompts/generated_stage1__05_support_vector_machines.md) and skipped `06_decision_trees.ipynb` because the scanner classified its intro as substantive. `git diff --name-only -- '*.ipynb'` returned no changes, so no notebooks were modified.

notes/031_stage1_wrapper_mvp_report__20260417_000000.md
diff --git a/tools/notebook_enricher/stage1_prompt_wrapper.py b/tools/notebook_enricher/stage1_prompt_wrapper.py
new file mode 100644
index 0000000000000000000000000000000000000000..ccef40439f9d87bde89c38ad4b3386256b87f94b
--- /dev/null
+++ b/tools/notebook_enricher/stage1_prompt_wrapper.py
@@ -0,0 +1,303 @@
+#!/usr/bin/env python3
+
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+from dataclasses import dataclass
+from pathlib import Path
+import sys
+
+from notebook_scanner import NotebookInventory, scan_notebook
+from prompt_builder import build_stage1_prompt
+
+
+PROMPTS_DIR = "codex_prompts"
+NOTES_DIR = "notes"
+GENERATED_PROMPT_PREFIX = "generated_stage1__"
+REPORT_PREFIX = "031_stage1_wrapper_mvp_report__"
+
+
+@dataclass
+class Stage1Decision:
+    inventory: NotebookInventory
+    action: str
+    reason: str
+
+    @property
+    def needs_prompt(self) -> bool:
+        return self.action in {"insert", "replace"}
+
+
+@dataclass
+class PromptArtifact:
+    notebook_path: Path
+    prompt_path: Path
+    action: str
+    reason: str
+
+
+def repo_root() -> Path:
+    return Path(__file__).resolve().parents[2]
+
+
+def utc_timestamp() -> str:
+    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Generate Stage 1 notebook-enrichment prompts for an explicit notebook list.",
+    )
+    parser.add_argument(
+        "notebooks",
+        nargs="+",
+        help="Ordered notebook paths, relative to the repository root or absolute.",
+    )
+    parser.add_argument(
+        "--report-path",
+        help="Optional explicit markdown report path. Defaults to notes/ with a UTC timestamp.",
+    )
+    parser.add_argument(
+        "--overwrite-generated",
+        action="store_true",
+        help="Allow overwriting an existing generated Stage 1 prompt file for the same notebook.",
+    )
+    return parser.parse_args()
+
+
+def resolve_notebook_path(root: Path, notebook_arg: str) -> Path:
+    path = Path(notebook_arg)
+    if not path.is_absolute():
+        path = root / path
+    return path.resolve()
+
+
+def resolve_output_path(root: Path, output_arg: str) -> Path:
+    path = Path(output_arg)
+    if not path.is_absolute():
+        path = root / path
+    return path.resolve()
+
+
+def relative_repo_path(root: Path, path: Path) -> str:
+    return path.relative_to(root).as_posix()
+
+
+def classify_stage1_decision(inventory: NotebookInventory) -> Stage1Decision:
+    status = inventory.chapter_intro_status
+
+    if status == "substantive":
+        return Stage1Decision(
+            inventory=inventory,
+            action="skip",
+            reason="scanner classified the existing chapter intro as substantive",
+        )
+
+    if status == "missing":
+        return Stage1Decision(
+            inventory=inventory,
+            action="insert",
+            reason="scanner found no eligible chapter intro prose cell before the setup boundary",
+        )
+
+    if status in {"heading", "thin"} and inventory.chapter_intro_index >= 0:
+        return Stage1Decision(
+            inventory=inventory,
+            action="replace",
+            reason=(
+                "scanner found an existing intro candidate that is not yet substantive "
+                f"({status})"
+            ),
+        )
+
+    return Stage1Decision(
+        inventory=inventory,
+        action="skip",
+        reason="scanner evidence did not support one deterministic Stage 1 insert-or-replace target",
+    )
+
+
+def prompt_filename_for(root: Path, notebook_path: Path) -> str:
+    relative = notebook_path.relative_to(root).with_suffix("")
+    slug = "__".join(relative.parts)
+    return f"{GENERATED_PROMPT_PREFIX}{slug}.md"
+
+
+def write_prompt_file(
+    *,
+    root: Path,
+    decision: Stage1Decision,
+    overwrite_generated: bool,
+) -> PromptArtifact:
+    prompts_dir = root / PROMPTS_DIR
+    prompts_dir.mkdir(parents=True, exist_ok=True)
+
+    prompt_path = prompts_dir / prompt_filename_for(root, decision.inventory.path)
+    if prompt_path.exists() and not overwrite_generated:
+        raise FileExistsError(
+            f"generated prompt already exists: {relative_repo_path(root, prompt_path)} "
+            "(rerun with --overwrite-generated to replace it)"
+        )
+
+    prompt_text = build_stage1_prompt(root, decision.inventory)
+    prompt_path.write_text(prompt_text, encoding="utf-8")
+
+    return PromptArtifact(
+        notebook_path=decision.inventory.path,
+        prompt_path=prompt_path,
+        action=decision.action,
+        reason=decision.reason,
+    )
+
+
+def default_report_path(root: Path) -> Path:
+    return root / NOTES_DIR / f"{REPORT_PREFIX}{utc_timestamp()}.md"
+
+
+def build_report_text(
+    *,
+    root: Path,
+    notebook_paths: list[Path],
+    generated: list[PromptArtifact],
+    skipped: list[Stage1Decision],
+) -> str:
+    lines: list[str] = [
+        "# Stage 1 Wrapper MVP Report",
+        "",
+        "## Executive Summary",
+        "",
+        (
+            f"Requested {len(notebook_paths)} notebook(s). Generated {len(generated)} "
+            f"Stage 1 prompt(s) and skipped {len(skipped)} notebook(s)."
+        ),
+        "This wrapper run stopped at prompt generation and did not invoke `tools/codex/run_prompt.py`.",
+        "",
+        "## Target Notebook List",
+        "",
+    ]
+
+    for idx, notebook_path in enumerate(notebook_paths, start=1):
+        lines.append(f"{idx}. `{relative_repo_path(root, notebook_path)}`")
+
+    lines.extend(
+        [
+            "",
+            "## Generated Prompts",
+            "",
+        ]
+    )
+
+    if generated:
+        for artifact in generated:
+            lines.append(
+                (
+                    f"- `{relative_repo_path(root, artifact.notebook_path)}`: generated "
+                    f"`{relative_repo_path(root, artifact.prompt_path)}` "
+                    f"because action is `{artifact.action}` and {artifact.reason}."
+                )
+            )
+    else:
+        lines.append("- None.")
+
+    lines.extend(
+        [
+            "",
+            "## Skipped Notebooks",
+            "",
+        ]
+    )
+
+    if skipped:
+        for decision in skipped:
+            inv = decision.inventory
+            lines.append(
+                (
+                    f"- `{relative_repo_path(root, inv.path)}`: skipped because action is "
+                    f"`{decision.action}` and {decision.reason}. "
+                    f"(intro_status=`{inv.chapter_intro_status}`, "
+                    f"intro_index=`{inv.chapter_intro_index}`, "
+                    f"setup_index=`{inv.setup_cell_index}`)"
+                )
+            )
+    else:
+        lines.append("- None.")
+
+    lines.extend(
+        [
+            "",
+            "## Deferred Work",
+            "",
+            "- No generated prompts were executed in this pass.",
+            "- No notebooks were modified in this pass.",
+            "- Stage 2 and Stage 3 remain out of scope.",
+        ]
+    )
+
+    return "\n".join(lines) + "\n"
+
+
+def main() -> int:
+    args = parse_args()
+    root = repo_root()
+
+    notebook_paths: list[Path] = []
+    for notebook_arg in args.notebooks:
+        notebook_path = resolve_notebook_path(root, notebook_arg)
+        if not notebook_path.exists():
+            print(f"ERROR: notebook not found: {notebook_arg}", file=sys.stderr)
+            return 1
+        try:
+            notebook_path.relative_to(root)
+        except ValueError:
+            print(
+                f"ERROR: notebook path is outside the repository root: {notebook_path}",
+                file=sys.stderr,
+            )
+            return 1
+        if notebook_path.suffix != ".ipynb":
+            print(f"ERROR: not a notebook path: {notebook_arg}", file=sys.stderr)
+            return 1
+        notebook_paths.append(notebook_path)
+
+    generated: list[PromptArtifact] = []
+    skipped: list[Stage1Decision] = []
+
+    for notebook_path in notebook_paths:
+        inventory = scan_notebook(notebook_path)
+        decision = classify_stage1_decision(inventory)
+        if decision.needs_prompt:
+            artifact = write_prompt_file(
+                root=root,
+                decision=decision,
+                overwrite_generated=args.overwrite_generated,
+            )
+            generated.append(artifact)
+            print(
+                f"GENERATED {relative_repo_path(root, artifact.prompt_path)} "
+                f"for {relative_repo_path(root, notebook_path)} "
+                f"({decision.action})"
+            )
+        else:
+            skipped.append(decision)
+            print(
+                f"SKIPPED {relative_repo_path(root, notebook_path)} "
+                f"({decision.reason})"
+            )
+
+    report_path = resolve_output_path(root, args.report_path) if args.report_path else default_report_path(root)
+    report_path.parent.mkdir(parents=True, exist_ok=True)
+    report_text = build_report_text(
+        root=root,
+        notebook_paths=notebook_paths,
+        generated=generated,
+        skipped=skipped,
+    )
+    report_path.write_text(report_text, encoding="utf-8")
+    print(f"REPORT {relative_repo_path(root, report_path)}")
+
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())

tokens used
60,257
```
