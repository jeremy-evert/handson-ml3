# Task: Audit notes and prompt scaffolding, then classify what should be kept, summarized, or moved to an attic

You are working in this repository.

Your task is to inspect the current `notes/` and `codex_prompts/` folders and produce a conservative classification plan for cleanup.

## Important framing

This is an audit and classification task only.

Do NOT move files.
Do NOT delete files.
Do NOT rewrite tools.
Do NOT rewrite major design documents.
Do NOT create the attic in this pass.

Your job is to classify the current scaffolding so later passes can safely summarize and move it without losing durable knowledge.

## Files and folders to inspect

Read and inspect:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `tools/codex/v1_record_validation.py`
- all files in `notes/`
- all files in `codex_prompts/`

You may also inspect the current repo tree to understand the broader context, but the classification target is specifically:

- `notes/`
- `codex_prompts/`

## Goal

Produce a conservative classification plan that distinguishes:

1. permanent residents
2. summarize, then move to attic
3. move to attic without summary
4. uncertain / needs human review

## Classification standard

A file should be treated as worth summarizing if it contains one or more of the following:

- a decision that still governs current behavior
- explanation of why the current design exists
- validation evidence for a still-live tool
- meaningful scope or priority decisions that still matter
- important failure findings that shaped the final design
- operational guidance that would still help a future maintainer

A file should be treated as likely scaffolding if it is mainly:

- a transient execution receipt
- an intermediate prompt-generation step
- a superseded planning or sequencing artifact
- a redundant success note from before the V1 system stabilized
- a local construction artifact whose main purpose was building the current toolset

## Required output artifacts

Create exactly two artifacts.

### Artifact 1

Create:

`notes/022_scaffolding_classification_report__TIMESTAMP.md`

This report should include:

- short summary
- classification criteria used
- permanent residents
- summarize-then-attic list
- attic-without-summary list
- uncertain list

For each listed file, include a short reason.

Be specific enough that a later prompt can act on this report without re-arguing every classification.

### Artifact 2

Create:

`notes/022_scaffolding_cleanup_plan__TIMESTAMP.md`

This note should contain only:

- what should happen in the next pass
- what should happen after that
- what should explicitly not happen yet

Keep this second note brief and operational.

## Constraints

1. Be conservative.
2. Prefer `uncertain` over aggressive classification when in doubt.
3. Do not move or delete anything in this pass.
4. Do not collapse the entire build history into one sentence.
5. Keep the plan practical enough to drive the next prompt.
6. Focus the classification on `notes/` and `codex_prompts/`, while using the tool and design docs only as context for deciding what is durable.

## Success criteria

This task is successful if:

- the current notes and prompts are classified into the four categories above
- the reasoning is concrete and repo-specific
- the result is safe enough to use as the basis for a summary-extraction pass
- no files are moved or deleted
