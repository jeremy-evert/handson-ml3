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
