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
