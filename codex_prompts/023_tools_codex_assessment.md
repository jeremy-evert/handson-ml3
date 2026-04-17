# Prompt: Assess wrapper design for staged notebook-treatment runs using the existing Codex runner

You are working inside this repository.

Your job is NOT to implement the notebook wrapper yet.

Your job is to:
1. inspect the existing Codex runner tooling and related design notes,
2. assess how a staged notebook-treatment wrapper could be built around the current Codex-centered workflow,
3. write a clear implementation-planning report into `notes/`,
4. make no code changes.

## Core intent

We want to keep the workflow centered around Codex calls.

The desired future behavior is a wrapper that can launch one narrow Codex task at a time against a notebook set, instead of asking Codex to solve the entire notebook in one pass.

The staged workflow we are considering is:

### Stage 1
For each notebook:
- inspect whether there is a strong chapter-level markdown introduction near the front
- if missing, write it
- if already present and sufficient, skip
- then move to the next notebook

### Stage 2
For each notebook:
- inspect each non-empty code cell
- determine whether there is a markdown cell immediately before it that explains:
  - the goal of the code cell
  - why this matters for machine learning
- if missing, add it
- if present and sufficient, skip
- continue for all code cells
- continue for all notebooks

### Stage 3
For each notebook:
- inspect each non-empty code cell
- determine whether there is a markdown cell immediately after it that explains:
  - how the implementation works
  - why this is a good practice
  - what other methods could accomplish something similar
- if missing, add it
- if present and sufficient, skip
- continue for all code cells
- continue for all notebooks

## Important constraints

- Do NOT implement the wrapper yet
- Do NOT modify any `.py` files
- Do NOT modify any notebooks
- Do NOT create a new runner
- Do NOT refactor existing tooling
- Do NOT stage or commit anything
- Do NOT change the V1 execution record system
- Only assess what it would take to build this cleanly using the current Codex-centered approach

## Files and areas to inspect

You should inspect the current Codex tooling and any repo notes that help you understand the intended architecture, including at minimum:

- `tools/codex/run_prompt.py`
- the related validation / review / queue scripts in `tools/codex/`
- the V1 workflow markdown files in `tools/codex/` if present
- existing notes or reports that discuss the runner, bridge workflow, notebook treatment, or staged design
- the relevant notebooks only as reference examples if needed

You should especially pay attention to:
- how the current runner executes one prompt
- how records are written into `notes/`
- how queue/review logic works
- what a wrapper would need to do without breaking the current model
- whether this should be a thin orchestration layer that emits prompt files and calls the current runner repeatedly
- how progress and idempotency could be tracked
- what risks exist around notebook mutation
- what minimum viable version should look like

## Specific questions to answer in the report

Your report must answer these questions directly and concretely:

1. Can this staged notebook-treatment workflow be built cleanly while staying centered on the current Codex runner?
2. What is the smallest viable wrapper design that would make this work?
3. Should the wrapper:
   - generate prompt files and call `run_prompt.py` repeatedly,
   - call `codex exec` directly,
   - or use some other thin orchestration approach?
4. What state would need to be tracked between runs?
5. How should the wrapper know:
   - which notebooks are in scope,
   - which stage is currently being worked,
   - which notebook/cell is next,
   - which work is already complete,
   - and when it is safe to stop or resume?
6. What existing parts of the repo can be reused as-is?
7. What new pieces would likely be required?
8. What are the biggest technical risks?
9. What should the MVP do first, before any more advanced features?
10. What should NOT be built yet?

## Deliverable

Write exactly one markdown report into `notes/`.

Use a filename that matches this pattern:

`notes/XXX_staged_notebook_wrapper_assessment__YYYYMMDD_HHMMSS.md`

If you cannot create that exact timestamped filename, create a close equivalent that is clear and unique.

## Report structure

Use this structure:

# Staged Notebook Wrapper Assessment

## Executive Summary
- one short paragraph stating whether the idea is viable

## What Exists Today
- summarize the current runner-centered workflow
- identify which current files are relevant

## Target Workflow
- restate the three-stage notebook-treatment concept in implementation terms

## Recommended Wrapper Design
- describe the thinnest clean design
- explain how it would interact with the current runner

## Reusable Existing Components
- list specific files / functions / patterns already present that should be reused

## New Components Likely Needed
- list the minimum new modules / files / artifacts required

## State and Progress Tracking
- explain how progress could be tracked safely across runs

## Risks and Safeguards
- explain notebook-integrity risks and how to reduce them

## MVP Recommendation
- explain the smallest version worth building first

## What To Avoid
- explain overengineering traps or design mistakes to avoid

## Suggested Build Sequence
- provide an ordered, practical build sequence

## Final Recommendation
- a concise final judgment

## Output rules

- Be concrete
- Be repo-specific
- Name files and scripts explicitly where possible
- Prefer practical design over abstract architecture talk
- Do not write code
- Do not modify repo files except for the single report in `notes/`
- At the end of your final response, print only the path to the note you created
