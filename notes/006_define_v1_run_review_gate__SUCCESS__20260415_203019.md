# 006_define_v1_run_review_gate - SUCCESS

- Prompt file: `006_define_v1_run_review_gate.md`
- Timestamp (UTC): `20260415_203019`
- Status: `SUCCESS`

## Original Prompt

```md
# Task: Define the V1 run-review gate for prompt execution

You are working in this repository.

Your task is to define the minimum review gate that must sit between:

- a prompt run finishing
- the next prompt in sequence being treated as ready

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/baby_run_prompt.py`
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
- `notes/005_prompt_queue_plan__20260415_202557.md`

## Goal

Use `tools/Project_Design_Workflow.md` as governing.

Define the smallest practical V1 review gate that preserves:

- review between iterations
- separation of execution outcome from accepted outcome
- a conservative manual step before the next prompt is treated as ready

## Important framing

This is a design task.

Do NOT implement the gate in code in this pass.
Do NOT create a large state machine.
Do NOT redesign the whole queue system.

## Questions to settle

Please settle these points:

1. What exact event marks a run as awaiting review?
2. What minimum human checks must happen before a run can be accepted?
3. What review outcomes are needed in V1?
4. What outcome should let the next prompt proceed?
5. What outcome should stop the queue and force a new design or retry decision?
6. What information must be written back into the run record during review?

## Required output artifact

Create one markdown design note at:

`tools/codex/V1_Run_Review_Gate.md`

The note should include:

- purpose
- scope
- the review trigger
- the minimum manual checklist
- allowed V1 review outcomes
- how the outcome affects queue progression
- how this connects to `tools/codex/V1_Execution_Record_Artifact.md`
- what is intentionally deferred

## Constraints

1. Use the exact file paths listed above.
2. Keep the gate manual and conservative.
3. Do not implement CLI or runner changes in this pass.
4. Do not invent a large workflow beyond the next reviewed step.
5. Keep the decision small enough to review before any implementation prompt follows.

## Success criteria

This task is successful if:

- the review gate is explicit and easy to apply
- execution success is clearly separated from accepted outcome
- queue progression rules are clear for V1
- the result is small enough to guide the next prompt without expanding scope
```

## Codex Output

Created [V1_Run_Review_Gate.md](/data/git/handson-ml3/tools/codex/V1_Run_Review_Gate.md).

It defines the minimal V1 gate as: execution finishes, a run record is written with `review_status: UNREVIEWED`, a human performs a short manual checklist, and only `ACCEPTED` allows the next prompt to be treated as ready. `REJECTED` is the stop condition that forces an explicit redesign or retry decision, and the note ties that directly back to the fields in [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md).

No code or CLI changes were made.

## Notes

Codex stderr:
```text
OpenAI Codex v0.120.0 (research preview)
--------
workdir: /data/git/handson-ml3
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, /home/jevert/.codex/memories]
reasoning effort: none
reasoning summaries: none
session id: 019d92d6-6e64-7182-9253-261cc553a5fb
--------
user
# Task: Define the V1 run-review gate for prompt execution

You are working in this repository.

Your task is to define the minimum review gate that must sit between:

- a prompt run finishing
- the next prompt in sequence being treated as ready

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/baby_run_prompt.py`
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
- `notes/005_prompt_queue_plan__20260415_202557.md`

## Goal

Use `tools/Project_Design_Workflow.md` as governing.

Define the smallest practical V1 review gate that preserves:

- review between iterations
- separation of execution outcome from accepted outcome
- a conservative manual step before the next prompt is treated as ready

## Important framing

This is a design task.

Do NOT implement the gate in code in this pass.
Do NOT create a large state machine.
Do NOT redesign the whole queue system.

## Questions to settle

Please settle these points:

1. What exact event marks a run as awaiting review?
2. What minimum human checks must happen before a run can be accepted?
3. What review outcomes are needed in V1?
4. What outcome should let the next prompt proceed?
5. What outcome should stop the queue and force a new design or retry decision?
6. What information must be written back into the run record during review?

## Required output artifact

Create one markdown design note at:

`tools/codex/V1_Run_Review_Gate.md`

The note should include:

- purpose
- scope
- the review trigger
- the minimum manual checklist
- allowed V1 review outcomes
- how the outcome affects queue progression
- how this connects to `tools/codex/V1_Execution_Record_Artifact.md`
- what is intentionally deferred

## Constraints

1. Use the exact file paths listed above.
2. Keep the gate manual and conservative.
3. Do not implement CLI or runner changes in this pass.
4. Do not invent a large workflow beyond the next reviewed step.
5. Keep the decision small enough to review before any implementation prompt follows.

## Success criteria

This task is successful if:

- the review gate is explicit and easy to apply
- execution success is clearly separated from accepted outcome
- queue progression rules are clear for V1
- the result is small enough to guide the next prompt without expanding scope

codex
I’m reading the specified workflow and architecture files first, then I’ll draft a minimal V1 review-gate note that stays within the current design boundaries.
exec
/bin/bash -lc "sed -n '1,240p' tools/Project_Design_Workflow.md" in /data/git/handson-ml3
 succeeded in 0ms:
# Project Design Workflow

## Goal

Define a repeatable workflow for turning an idea or architecture document into a clean, working system without skipping the thinking that keeps the build safe, clear, and reusable.

This document is meant to sit beside architecture documents and help answer a practical question:

**What are the steps between "this design makes sense" and "the thing works"?**

---

## Why This Exists

Sometimes a project feels sound at the architecture level but still feels fuzzy at the build level.

That happens because architecture is not the same thing as execution.

Architecture tells us:

* what parts exist
* what they are responsible for
* how they relate
* what should not be coupled

But architecture does not yet tell us:

* what to build first
* what order reduces risk
* what to test first
* what can wait
* how to know whether the design is holding
* how to recover cleanly when a step fails

This workflow bridges that gap.

---

## Relationship to a Larger Design Process

This workflow does **not** replace broader project thinking.
It fits inside it.

A useful mapping looks like this:

### 1. Conversation

What are we trying to do?
What hurts right now?
What would "better" feel like?

### 2. Goals

What must become true for this project to count as successful?

### 3. Deliverables

What concrete artifacts must exist?
Examples:

* architecture doc
* workflow doc
* module layout
* interface sketch
* notes folder
* validation checklist
* first working slice

### 4. Tasks

What actions create those deliverables?
Examples:

* define V1 scope
* define module boundaries
* choose naming conventions
* write first thin component
* validate one path end to end
* capture review notes

### 5. Implementation

Only after the earlier layers are stable do we write implementation.

So this workflow is not separate from design.
It is the part that turns design into bounded execution.

---

## Core Principles

### 1. Design before build

Do not use implementation momentum to hide unclear thinking.

### 2. Boundaries before breadth

A project becomes safer when scope, responsibility, and deferral lines are visible.

### 3. Thin slices before large pushes

Prefer a bounded slice that proves something real over a large batch that creates many unknowns at once.

### 4. Review between iterations

Each bounded execution step should be inspected before the next one is issued.
Do not assume a sequence is healthy just because the previous step completed.

### 5. Validation is part of design

Tests, inspections, examples, and acceptance checks are not extra work.
They are how the design proves it is real.

### 6. Bridge tooling is allowed, but subordinate

Temporary or bridge tooling can be useful when it stays thin, inspectable, and reversible.
It may help move work forward, gather evidence, or reduce manual friction.
It should not become a substitute for architecture, clear interfaces, or good decomposition.

### 7. Durable local history matters

Notes, logs, reports, generated outputs, and similar local artifacts can serve as durable project memory.
They help preserve decisions, validation results, failure patterns, and the reasoning behind refinements.

### 8. Failure should produce analysis, not just retries

When a step fails, the useful question is not only "how do we try again?"
It is also "what did this failure reveal about the task, decomposition, criteria, or environment?"

### 9. Resource use should be observed

Project effort has a cost, whether that cost is human time, machine time, review effort, retries, execution size, or external spend.
Large tasks should justify themselves.

---

## The Core Sequence

Here is the recommended design workflow.

### Phase 1: Clarify the problem

This is the "why are we doing this?" phase.

Questions:

* What pain are we removing?
* What confusion are we reducing?
* What repeated work are we trying to standardize?
* What is dangerous if we build too fast?

Output:

* short problem statement
* short success statement

---

### Phase 2: Define the system boundary

This is the "what belongs in this project and what does not?" phase.

Questions:

* What should this system do?
* What should it explicitly not do yet?
* What decisions are deferred?
* What adjacent problems are real but out of scope?

Output:

* scope statement
* out-of-scope list

---

### Phase 3: Draft the architecture

This is the "what pieces exist and what are their jobs?" phase.

Questions:

* What modules, components, or services are needed?
* What is each one responsible for?
* What should each one never own?
* Where are the seams between parts?

Output:

* architecture doc
* responsibility split
* proposed file, package, or interface layout

---

### Phase 4: Identify the minimum viable slice

This is the "what is the smallest useful thing we can build that proves the design?" phase.

Questions:

* What is the thinnest vertical slice that is actually useful?
* What can we test without building the whole system?
* What gives us signal early?
* What first slice is unlikely to trap us later?

Output:

* V1 feature list
* initial build plan

---

### Phase 5: Define the artifacts

This is the "what files or outputs must exist?" phase.

Questions:

* What documents should exist?
* What modules or interfaces should exist?
* What examples, fixtures, or test inputs should exist?
* What outputs prove the path works?
* What notes, logs, or reports should be kept as durable project memory?

Output:

* deliverables list
* artifact inventory

---

### Phase 6: Sequence the work

This is the "what order reduces pain and risk?" phase.

Questions:

* What must come first because other things depend on it?
* What can be tested independently?

exec
/bin/bash -lc "sed -n '1,260p' tools/codex/V1_Execution_Record_Artifact.md" in /data/git/handson-ml3
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

The following should wait:

- automatic acceptance or approval gating
- dependency-aware prompt release
- richer review states such as `BLOCKED` or `PARTIAL`
- structured JSON sidecars or databases
- normalized token accounting across runners
- automatic failure classification
- aggregated reports and dashboards
- full retry history synthesis
- broader refactor of the runner into multiple modules

V1 is only trying to establish a durable, reviewable single-run artifact that cleanly separates execution from acceptance.

exec
/bin/bash -lc "sed -n '1,260p' tools/Codex_Prompt_Workflow_Architecture.md" in /data/git/handson-ml3
 succeeded in 0ms:
# Codex Prompt Workflow Architecture

## Goal

Build a small, clean, reusable prompt workflow system for Codex work inside a repository.

This system should be conservative first:

* easy to inspect
* easy to test
* easy to extend
* composed of small parts
* safe by default

The design should emphasize **separation of concerns** so each piece does one job clearly.

---

## Why We Are Re-Architecting

The first draft bundled too many responsibilities into one script. That made it harder to:

* understand
* trust
* test
* evolve
* reuse across repositories

We want a cleaner structure that can become a template for future repos.

---

## What the First Script Was Doing

Below is a decomposition of the responsibilities that were bundled together.

### 1. Repo path discovery

* figure out where the repository root is
* infer where `codex_prompts/` and `notes/` live

### 2. Directory validation

* verify the expected folders exist
* fail cleanly if they do not

### 3. Prompt discovery

* scan `codex_prompts/`
* identify valid prompt files
* ignore files that do not match the naming convention

### 4. Prompt parsing and indexing

* extract numeric prefixes from filenames
* sort prompts in execution order
* establish a stable prompt identity from filename/stem

### 5. Note discovery

* scan `notes/`
* find note files that match the naming convention
* ignore unrelated markdown files

### 6. Note parsing

* extract prompt name, success/fail state, and timestamp from note filenames
* turn note filenames into structured metadata

### 7. Status reconstruction

* map prompts to matching notes
* determine whether each prompt is:

  * UNRUN
  * SUCCESS
  * FAIL
* pick the latest note when there are multiple notes for one prompt

### 8. Prompt selection

* find the next unrun prompt
* find the first failed prompt
* find a prompt by numeric prefix, base name, or full filename

### 9. Prompt display

* print a selected prompt to the terminal

### 10. Retry context assembly

* find the latest failed note for a prompt
* display the original prompt plus the latest failed note
* present instructions for retrying

### 11. Note writing

* create timestamped note filenames
* write notes in markdown
* optionally include the previous note as context

### 12. Manual status marking

* let the user record a success or failure
* attach summary/details text to the note

### 13. Command-line interface

* parse subcommands like `list`, `next`, `show`, `mark`, `retry-failed`
* route commands to the right behaviors

### 14. Terminal presentation

* format output for human readability
* present status tables and retry blocks

---

## What I Would Like This System To Do Eventually

These are ideas worth considering, but they should not all be built at once.

### A. Better architecture and reuse

* work as a reusable template across repositories
* isolate config from code
* support per-repo conventions without rewriting internals

### B. Structured metadata

* optionally store prompt metadata in frontmatter or sidecar files
* support fields like:

  * title
  * objective
  * tags
  * priority
  * dependencies
  * owner

### C. Better status model

Instead of only:

* UNRUN
* SUCCESS
* FAIL

We may want:

* READY
* RUNNING
* BLOCKED
* NEEDS_REVIEW
* PARTIAL
* SKIPPED
* ARCHIVED

### D. Retry intelligence

* carry forward the previous failed note automatically
* include multiple prior failures, not just the latest one
* summarize recurring failure patterns

### E. Prompt preparation

* generate a clean “execution bundle” for Codex
* include:

  * prompt text
  * repo context
  * previous failure note if retrying
  * explicit success criteria

### F. Manual and automated modes

* manual mode for conservative workflows
* dry-run mode for seeing what would happen
* later, optional Codex CLI integration

### G. Report generation

* generate queue reports
* generate status snapshots
* generate audit/history reports over prompt activity

### H. Better note/content model

* move from filename-only state to richer note contents
* enforce a consistent note template
* optionally add structured machine-readable metadata inside notes

### I. Dependency and sequencing support

* allow prompts to depend on other prompts
* prevent prompts from running before prerequisites are complete

### J. Prompt families or lanes

* support categories such as:

  * setup
  * audit
  * refactor
  * notebook generation
  * documentation

### K. Approval gates

* require human confirmation before marking success
* require review before allowing dependent prompts to proceed

### L. Logging and diagnostics

* maintain an execution log
* record tool errors separately from task failures
* distinguish infrastructure failure from prompt failure

### M. File layout that scales cleanly

* support growth without turning into a junk drawer

---

## Proposed Folder Architecture

A better first structure would be:

```text
tools/
  codex/
    README.md
    architecture.md
    config.py
    paths.py
    prompts.py
    notes.py
    status.py
    retry.py
    cli.py
    templates/
      note_success.md
      note_fail.md
```

For an even more conservative start, we can go smaller:

```text
tools/
  codex/
    README.md
    architecture.md
    paths.py
    prompts.py
    notes.py
    status.py
```

Then add a tiny entrypoint later if needed.


exec
/bin/bash -lc "sed -n '1,260p' tools/codex/baby_run_prompt.py" in /data/git/handson-ml3
 succeeded in 0ms:
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import tempfile
from pathlib import Path


PROMPTS_DIR = "codex_prompts"
NOTES_DIR = "notes"


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def build_note_path(root: Path, prompt_path: Path, status: str, timestamp: str) -> Path:
    return root / NOTES_DIR / f"{prompt_path.stem}__{status}__{timestamp}.md"


def build_note_content(
    prompt_path: Path,
    timestamp: str,
    status: str,
    prompt_text: str,
    codex_output: str,
    notes: str = "",
) -> str:
    return f"""# {prompt_path.stem} - {status}

- Prompt file: `{prompt_path.name}`
- Timestamp (UTC): `{timestamp}`
- Status: `{status}`

## Original Prompt

```md
{prompt_text.rstrip()}
```

## Codex Output

{codex_output.rstrip() or "*No output captured.*"}

## Notes

{notes.rstrip()}
"""


def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
        output_path = Path(handle.name)

    try:
        result = subprocess.run(
            ["codex", "exec", "-C", str(root), "--output-last-message", str(output_path), "-"],
            input=prompt_text,
            text=True,
            capture_output=True,
            check=False,
        )

        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        return result.returncode, final_output, result.stderr
    finally:
        output_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one prompt file through codex exec and write a note.")
    parser.add_argument("prompt", help="Prompt filename, numeric prefix, or path")
    args = parser.parse_args()

    root = repo_root()
    prompts_dir = root / PROMPTS_DIR
    notes_dir = root / NOTES_DIR

    if not prompts_dir.exists():
        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
        return 1

    if not notes_dir.exists():
        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
        return 1

    prompt_path = resolve_prompt_path(root, args.prompt)
    if not prompt_path.exists():
        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
        return 1

    prompt_text = prompt_path.read_text(encoding="utf-8")
    timestamp = utc_timestamp()

    returncode, codex_output, stderr_text = run_codex(root, prompt_text)
    status = "SUCCESS" if returncode == 0 else "FAILED"
    note_path = build_note_path(root, prompt_path, status, timestamp)

    notes = ""
    if stderr_text.strip():
        notes = f"""Codex stderr:
```text
{stderr_text.rstrip()}
```"""

    note_path.write_text(
        build_note_content(
            prompt_path=prompt_path,
            timestamp=timestamp,
            status=status,
            prompt_text=prompt_text,
            codex_output=codex_output,
            notes=notes,
        ),
        encoding="utf-8",
    )

    print(note_path)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())

exec
/bin/bash -lc "sed -n '1,260p' notes/004_architecture_and_bridge_runner_review__20260415_195538.md" in /data/git/handson-ml3
 succeeded in 0ms:
# Architecture And Bridge Runner Review

## Summary

`tools/Codex_Prompt_Workflow_Architecture.md` is directionally aligned with the governing workflow because it pushes toward smaller parts, clearer ownership, and conservative growth. `tools/codex/baby_run_prompt.py` is acceptable as thin bridge tooling for a single bounded slice, but the current architecture doc does not yet translate the workflow into a clear V1 boundary, validation plan, review loop, or failure/resource evidence model. The next step should stay at the design layer: define the minimum execution-record artifact that every run must produce before more tooling is added.

## Architecture Alignment Findings

### Strong alignment

- The architecture doc is explicitly conservative, small-part oriented, and separation-of-concerns driven, which matches design-before-build and boundaries-before-breadth in the workflow ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:92), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:5)).
- Its decomposition of the original script into repo paths, prompt parsing, note handling, status reconstruction, retry handling, and CLI concerns is a good responsibility split and a useful architecture starting point ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:177), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:33)).
- It repeatedly says not to build everything at once and proposes a smaller folder shape before a larger one, which is consistent with thin slices before large pushes ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:100), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:121), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:246)).

### Stale, incomplete, or misaligned

- The architecture doc does not yet reflect the workflow's full sequence. It jumps from decomposition into a wish list and folder layout, but it does not define a short problem statement, explicit scope fence, out-of-scope list, minimum viable slice, artifact inventory, implementation order, or validation checklist as the governing workflow expects ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:141), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:159), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:196), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:214), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:233), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:251)).
- The "eventually" section is useful as an extension path, but it is carrying too much conceptual breadth for a workflow that emphasizes bounded next steps. There is no explicit line saying which of those items are intentionally deferred from V1 ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:96), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:374), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:119)).
- The document mentions logging and diagnostics, richer notes, retry intelligence, approval gates, and dependencies, but it does not specify the source of truth or the stable identity of important records beyond prompt stems. That leaves the durable-history model underdesigned relative to the workflow's emphasis on preserved local memory and stable identities ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:120), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:374), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:187)).
- Review points between iterations are largely absent. The architecture doc talks about manual status marking and approval gates eventually, but not about the concrete review stop that should happen after each bounded run before the next run proceeds ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:262), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:208)).

## Runner Alignment Findings

### Useful and appropriately thin

- The runner does one bounded thing: resolve a prompt, execute `codex exec`, and write a local note. That fits the workflow's allowance for thin bridge tooling that gathers evidence while remaining inspectable and reversible ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:114), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:99)).
- It validates basic directory prerequisites and fails cleanly when expected folders or prompt files are missing, which is a reasonable minimal safeguard for a bridge script ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:109), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:108)).
- It writes timestamped markdown notes into `notes/`, which gives the system at least some durable local history of what was run and what Codex returned ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:120), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:46), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:135)).

### Responsibilities it is carrying that should move elsewhere later

- Prompt resolution policy is embedded in the runner. Selection by absolute path, repo-relative path, `codex_prompts/` path, or globbed prefix is helpful now, but prompt identity and lookup rules belong in a prompt-discovery/status layer if the system grows ([tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:47), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:25)).
- Note-path and note-content policy are also embedded here. That is acceptable for a bridge script, but note schema, status vocabulary, and execution-record structure should eventually be owned by a note/history module rather than the process launcher ([tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:96), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:187), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:50)).
- The runner currently decides final state as `SUCCESS` or `FAILED` based only on the subprocess exit code. That is an expedient bridge behavior, but review judgment, validation outcome, and richer state modeling should not stay inside the launcher if the workflow matures ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:141), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:124)).

### Current misalignment

- As bridge tooling it is still acceptable, but it is only partially subordinate to workflow. It captures execution, yet it does not encode the review point that should separate execution from acceptance. A zero exit code becomes `SUCCESS` immediately, which skips the workflow's "validate and inspect before the next step" posture ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:271), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:125)).
- It stores the original prompt text and final output, but not explicit success criteria, validation evidence, or reviewer judgment. That means the note is an execution artifact, not yet a full workflow artifact ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:251), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:58)).

## Failure-Analysis Findings

- The current setup provides some raw failure evidence because nonzero exit runs still write a note and stderr is preserved when present ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:343), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:128)).
- That said, it does not support useful post-failure analysis very well. The note structure has no fields for failure classification, whether the failure was task/design/environment related, whether the task was too large, what changed on retry, or what the recommended next action is. The workflow explicitly wants failure to produce analysis rather than mere retry cost ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:125), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:343)).
- The smallest improvement here is not a larger retry engine. It is a defined execution-note schema with a short "failure analysis" block for failed runs, even if the user fills some of it manually. That preserves inspectable local history without expanding the runner's responsibilities much.

## Resource-Awareness Findings

- The setup preserves only minimal cost evidence today: timestamp in the note filename/content and the captured final output/stderr. That gives some weak signal about artifact volume, but almost nothing about runtime, retries, or service usage ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:382), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:17), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:58)).
- There is no recorded elapsed time, no attempt count or retry linkage, no output-size metric, and no token usage capture if Codex exposes it. The workflow asks for lightweight observation, not a billing system, and the current setup falls short of even that lightweight bar ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:130), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:382)).
- The smallest improvement is to define a tiny metrics block in the execution record. Start with fields the runner can always capture cheaply: start time, end time, elapsed seconds, subprocess return code, stderr present yes/no, and output character count. Token usage can remain optional when available later.

## Single Recommended Next Design Step

Create a short design artifact that defines the V1 execution record for one prompt run.

That artifact should do four things only:

1. Declare the source of truth for a run record.
2. Define the minimum fields for execution, review, failure analysis, and lightweight metrics.
3. Separate execution outcome from reviewed outcome.
4. State what bridge tooling may capture now versus what remains manual for this stage.

This is the best next step because both the architecture doc and the runner are currently weak at the same seam: they can run prompts and save notes, but they do not yet define what a durable, reviewable run artifact must contain. Defining that record is smaller and safer than refactoring modules now, and it directly supports failure analysis, resource awareness, and review between iterations.

exec
/bin/bash -lc "sed -n '1,260p' notes/005_prompt_queue_plan__20260415_202557.md" in /data/git/handson-ml3
 succeeded in 0ms:
# Prompt Queue Plan

## Why These Prompts Were Chosen

These prompts were chosen because the current misalignment is concentrated at one seam:
the system can execute prompts, but it does not yet define the review checkpoint, the V1 architectural boundary around that checkpoint, or the smallest runner behavior needed to support it.

The queue stays small on purpose.
It settles the workflow gap before any implementation prompt is allowed to widen scope.

## Why This Order Reduces Risk

The order moves from policy to architecture to bridge-tooling:

1. define the review gate
2. align the architecture doc to that gate and the execution record
3. define the smallest runner change spec that fits those decisions

This reduces the risk of hard-coding the wrong note model or the wrong queue semantics into the runner before the manual review loop is settled.

## What Each Prompt Is Meant To Settle

`006_define_v1_run_review_gate.md`

- settles the missing review stop between execution and accepted progress

`007_align_architecture_doc_to_v1_workflow.md`

- settles the V1 boundary, artifact inventory, validation posture, and deferrals inside the architecture document

`008_define_minimal_bridge_runner_change_spec.md`

- settles the smallest bridge-runner behavior needed to emit the V1 execution record without a broader refactor

## What Is Intentionally Deferred

The following work is intentionally not in this queue:

- runner refactor into multiple modules
- richer state machines
- retry intelligence
- dependency-aware scheduling
- aggregated reporting
- token accounting beyond optional fields
- broad CLI redesign

codex
I have the core context. I’m pulling the remaining sections of the governing workflow so the gate language matches its sequencing, validation, and review requirements rather than inferring from fragments.
exec
/bin/bash -lc "sed -n '241,420p' tools/Project_Design_Workflow.md" in /data/git/handson-ml3
 succeeded in 0ms:
* What pieces should be proven before automation or convenience layers are added?
* What order keeps the build inspectable?

Output:

* implementation order
* dependency chain

---

### Phase 7: Define validation

This is the "how will we know each layer works?" phase.

Questions:

* What is the smoke test for each part?
* What is a good manual test before automation?
* What failure modes do we expect?
* What evidence counts as success?
* What review should happen before the next step begins?

Output:

* validation checklist
* smoke tests
* example inputs and outputs
* review points between iterations

---

### Phase 8: Execute one bounded slice

Only now do we begin implementation.

Rules:

* build one thin slice
* keep the task bounded and inspectable
* use bridge tooling only when it remains thin and subordinate to the design
* validate the slice
* inspect the result before issuing the next step
* do not sprint ahead because the first part felt good

Output:

* one completed slice
* evidence of validation
* notes about what the design got right or wrong

---

### Phase 9: Review and refine

This is the "did the design survive contact with reality?" phase.

Questions:

* What felt clean?
* What felt awkward?
* What assumptions broke?
* What should be renamed, split, simplified, or deferred?
* What should change before the next bounded step?

Output:

* refinement notes
* updated architecture or scope if needed
* revised next-step plan

---

## The Iteration Loop

After the initial design work, many projects should move through a repeating bounded loop:

1. clarify the next chunk
2. define a bounded task
3. state the success criteria
4. execute
5. validate and inspect results
6. review what changed
7. refine the plan
8. issue the next bounded task only after review

This loop should stay small enough that:

* the task can be understood before execution
* the result can be reviewed without guesswork
* failure teaches something specific
* refinement happens while context is still fresh

If a project cannot explain the next chunk in a few clear sentences, the chunk is probably still too large.

---

## The Practical Decomposition Pattern

When a design feels big, decompose it in this order:

### 1. Purpose

What is the system for?

### 2. Boundaries

What is in and out?

### 3. Components

What parts exist?

### 4. Responsibilities

What does each part own?

### 5. Artifacts

What files or outputs must exist?

### 6. Sequence

What gets built first?

### 7. Validation

How do we test each step?

### 8. Review points

Where do we stop and inspect before continuing?

### 9. Extension path

What comes later, but not now?

That pattern is portable and should work across many projects.

---

## Failure Analysis as Part of the Workflow

When a bounded step fails, do not treat the failure as noise.
Capture it, inspect it, and decide whether the problem is in the task, the decomposition, the criteria, or the environment.

Useful questions include:

* Was the task too large?
* Was the task poorly decomposed?
* Were success criteria unclear or incomplete?
* Did the task depend on hidden assumptions?
* Was the failure caused by tooling or infrastructure rather than task difficulty?
* Would a smaller or differently framed task have worked better?
* Did the review happen too late?
* Did retries produce new information, or only repeat cost?

Useful outputs include:

* a short failure note or report
* updated task boundaries
* revised success criteria
* a smaller follow-up slice
* a decision to fix environment issues before retrying

A clean retry is often possible, but it should come after analysis rather than instead of it.

---

## Resource and Cost Awareness

Projects benefit from tracking lightweight evidence about execution cost and quality.
This does not need to be elaborate, but it should be enough to notice patterns.

Examples of useful observations:

* elapsed time
* execution size
* review effort
* repeated retries
* failure frequency

codex
The governing rule is clear: execute, then validate and inspect, then decide whether the next bounded step is even eligible. I’m writing the note to keep that stop explicit without adding extra queue machinery.
exec
/bin/bash -lc 'rg --files tools/codex' in /data/git/handson-ml3
 succeeded in 0ms:
tools/codex/baby_run_prompt.py
tools/codex/V1_Execution_Record_Artifact.md

exec
/bin/bash -lc 'git status --short' in /data/git/handson-ml3
 succeeded in 0ms:
?? codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md
?? codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md
?? codex_prompts/006_define_v1_run_review_gate.md
?? codex_prompts/007_align_architecture_doc_to_v1_workflow.md
?? codex_prompts/008_define_minimal_bridge_runner_change_spec.md
?? notes/004_architecture_and_bridge_runner_review__20260415_195538.md
?? notes/004_next_design_step_recommendation__20260415_195538.md
?? notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md
?? notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md
?? notes/005_prompt_queue_plan__20260415_202557.md
?? tools/codex/V1_Execution_Record_Artifact.md

codex
I’ve checked the worktree so I can add only the requested design note and avoid touching the in-progress prompt artifacts around it.
apply patch
patch: completed
/data/git/handson-ml3/tools/codex/V1_Run_Review_Gate.md
diff --git a/tools/codex/V1_Run_Review_Gate.md b/tools/codex/V1_Run_Review_Gate.md
new file mode 100644
index 0000000000000000000000000000000000000000..8738b403163228dba1600691d8ff591f37d0c37f
--- /dev/null
+++ b/tools/codex/V1_Run_Review_Gate.md
@@ -0,0 +1,161 @@
+# V1 Run Review Gate
+
+## Purpose
+
+Define the smallest practical review gate between one prompt run finishing and the next prompt in sequence being treated as ready.
+
+This gate exists to preserve three things from the governing workflow:
+
+- review between iterations
+- separation of execution outcome from accepted outcome
+- a conservative human stop before queue progression
+
+## Scope
+
+V1 covers only the decision point immediately after one run record is written.
+
+It defines:
+
+- when a run enters review
+- the minimum manual checks
+- the allowed review outcomes
+- how those outcomes affect whether the next prompt may proceed
+- what review information must be written into the run record
+
+It does not define:
+
+- automation of review
+- a larger queue state machine
+- dependency-aware scheduling
+- retry orchestration
+- richer review taxonomies such as `PARTIAL` or `BLOCKED`
+
+## Review Trigger
+
+A run becomes `awaiting review` when its execution phase has finished and its V1 execution record has been written to `notes/` with:
+
+- `execution_status` set to either `EXECUTED` or `EXECUTION_FAILED`
+- `review_status` still set to `UNREVIEWED`
+
+That is the exact review trigger for V1.
+
+Process exit alone does not make the next prompt ready.
+Record creation plus `UNREVIEWED` review status means the run must stop for human inspection.
+
+## Minimum Manual Checklist
+
+Before a run can be accepted, a human reviewer must check only these items:
+
+1. Confirm the execution record is complete enough to review.
+   Required minimum: prompt identity, execution status, return code, prompt text, Codex final output, and any captured stderr.
+
+2. Check whether the run actually addressed the prompt that was executed.
+   This is a scope-and-intent check, not a deep redesign review.
+
+3. Check whether there is enough evidence to treat the result as an accepted outcome for the current bounded step.
+   Minimum evidence may be direct output inspection, referenced file changes, or stated validation evidence in the run record or surrounding repo state.
+
+4. Check whether any obvious failure, mismatch, or ambiguity remains that would make automatic progression unsafe.
+   Examples: execution failed, output is incomplete, acceptance is unclear, validation is missing, or the result suggests the design/task should be refined before continuing.
+
+This checklist is intentionally small.
+V1 only needs enough manual review to keep execution success separate from accepted progress.
+
+## Allowed V1 Review Outcomes
+
+V1 should allow exactly these review outcomes:
+
+- `UNREVIEWED`
+- `ACCEPTED`
+- `REJECTED`
+
+Meaning:
+
+- `UNREVIEWED`: default state after execution record creation; queue must stop here.
+- `ACCEPTED`: the bounded step is accepted after human review.
+- `REJECTED`: the run is not accepted for progression, whether because execution failed, the result is inadequate, or the outcome requires redesign or retry analysis first.
+
+No additional V1 outcomes are needed.
+
+## Queue Progression Rule
+
+Only `ACCEPTED` allows the next prompt in sequence to be treated as ready.
+
+Rules:
+
+- `UNREVIEWED` stops the queue pending manual review.
+- `ACCEPTED` releases exactly the next reviewed step.
+- `REJECTED` stops the queue and forces an explicit human decision about what happens next.
+
+`EXECUTED` does not release the queue.
+`EXECUTION_FAILED` does not itself decide the queue either, but it will usually lead to `REJECTED` on review.
+
+## Stop-And-Decide Rule
+
+`REJECTED` is the V1 outcome that stops the queue and forces a new design or retry decision.
+
+That decision is intentionally outside this gate.
+It may result in:
+
+- a revised prompt
+- a smaller follow-up slice
+- an environment fix
+- a deliberate retry
+- an architecture or scope adjustment
+
+V1 only requires that the queue does not continue past a rejected run.
+
+## Run Record Write-Back
+
+During review, the run record defined by [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md) must be updated with these review facts:
+
+- `review_status`
+- `review_summary`
+
+When available, V1 should also write:
+
+- `reviewed_by`
+- `reviewed_at_utc`
+
+For rejected runs, the reviewer should also fill the small failure-analysis fields when useful:
+
+- `failure_type`
+- `failure_symptom`
+- `likely_cause`
+- `recommended_next_action`
+
+This keeps execution facts and review judgment in one durable record.
+
+## Connection To The Execution Record
+
+This gate depends on the separation already defined in [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:77):
+
+- execution status answers "what happened when the runner executed the prompt?"
+- review status answers "did a human accept this bounded step as good enough to progress?"
+
+The V1 review gate is therefore not a new artifact.
+It is the manual transition of one execution record from `UNREVIEWED` to either `ACCEPTED` or `REJECTED`.
+
+## Intentionally Deferred
+
+V1 intentionally defers:
+
+- automatic queue release
+- automatic reviewer assignment
+- separate queue-level state files
+- richer review outcomes such as `PARTIAL`, `BLOCKED`, or `NEEDS_RETRY`
+- policy for choosing among multiple retry or redesign options
+- automatic extraction of validation evidence from repo changes
+- broader workflow redesign beyond the next reviewed step
+
+## V1 Decision Summary
+
+The minimum V1 review gate is:
+
+1. execution finishes
+2. a run record is written with `review_status: UNREVIEWED`
+3. human review applies the minimum checklist
+4. reviewer writes back `ACCEPTED` or `REJECTED`
+5. only `ACCEPTED` makes the next prompt ready
+
+That is small enough to guide the next implementation prompt without expanding the system into a larger workflow engine.

diff --git a/tools/codex/V1_Run_Review_Gate.md b/tools/codex/V1_Run_Review_Gate.md
new file mode 100644
index 0000000000000000000000000000000000000000..8738b403163228dba1600691d8ff591f37d0c37f
--- /dev/null
+++ b/tools/codex/V1_Run_Review_Gate.md
@@ -0,0 +1,161 @@
+# V1 Run Review Gate
+
+## Purpose
+
+Define the smallest practical review gate between one prompt run finishing and the next prompt in sequence being treated as ready.
+
+This gate exists to preserve three things from the governing workflow:
+
+- review between iterations
+- separation of execution outcome from accepted outcome
+- a conservative human stop before queue progression
+
+## Scope
+
+V1 covers only the decision point immediately after one run record is written.
+
+It defines:
+
+- when a run enters review
+- the minimum manual checks
+- the allowed review outcomes
+- how those outcomes affect whether the next prompt may proceed
+- what review information must be written into the run record
+
+It does not define:
+
+- automation of review
+- a larger queue state machine
+- dependency-aware scheduling
+- retry orchestration
+- richer review taxonomies such as `PARTIAL` or `BLOCKED`
+
+## Review Trigger
+
+A run becomes `awaiting review` when its execution phase has finished and its V1 execution record has been written to `notes/` with:
+
+- `execution_status` set to either `EXECUTED` or `EXECUTION_FAILED`
+- `review_status` still set to `UNREVIEWED`
+
+That is the exact review trigger for V1.
+
+Process exit alone does not make the next prompt ready.
+Record creation plus `UNREVIEWED` review status means the run must stop for human inspection.
+
+## Minimum Manual Checklist
+
+Before a run can be accepted, a human reviewer must check only these items:
+
+1. Confirm the execution record is complete enough to review.
+   Required minimum: prompt identity, execution status, return code, prompt text, Codex final output, and any captured stderr.
+
+2. Check whether the run actually addressed the prompt that was executed.
+   This is a scope-and-intent check, not a deep redesign review.
+
+3. Check whether there is enough evidence to treat the result as an accepted outcome for the current bounded step.
+   Minimum evidence may be direct output inspection, referenced file changes, or stated validation evidence in the run record or surrounding repo state.
+
+4. Check whether any obvious failure, mismatch, or ambiguity remains that would make automatic progression unsafe.
+   Examples: execution failed, output is incomplete, acceptance is unclear, validation is missing, or the result suggests the design/task should be refined before continuing.
+
+This checklist is intentionally small.
+V1 only needs enough manual review to keep execution success separate from accepted progress.
+
+## Allowed V1 Review Outcomes
+
+V1 should allow exactly these review outcomes:
+
+- `UNREVIEWED`
+- `ACCEPTED`
+- `REJECTED`
+
+Meaning:
+
+- `UNREVIEWED`: default state after execution record creation; queue must stop here.
+- `ACCEPTED`: the bounded step is accepted after human review.
+- `REJECTED`: the run is not accepted for progression, whether because execution failed, the result is inadequate, or the outcome requires redesign or retry analysis first.
+
+No additional V1 outcomes are needed.
+
+## Queue Progression Rule
+
+Only `ACCEPTED` allows the next prompt in sequence to be treated as ready.
+
+Rules:
+
+- `UNREVIEWED` stops the queue pending manual review.
+- `ACCEPTED` releases exactly the next reviewed step.
+- `REJECTED` stops the queue and forces an explicit human decision about what happens next.
+
+`EXECUTED` does not release the queue.
+`EXECUTION_FAILED` does not itself decide the queue either, but it will usually lead to `REJECTED` on review.
+
+## Stop-And-Decide Rule
+
+`REJECTED` is the V1 outcome that stops the queue and forces a new design or retry decision.
+
+That decision is intentionally outside this gate.
+It may result in:
+
+- a revised prompt
+- a smaller follow-up slice
+- an environment fix
+- a deliberate retry
+- an architecture or scope adjustment
+
+V1 only requires that the queue does not continue past a rejected run.
+
+## Run Record Write-Back
+
+During review, the run record defined by [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md) must be updated with these review facts:
+
+- `review_status`
+- `review_summary`
+
+When available, V1 should also write:
+
+- `reviewed_by`
+- `reviewed_at_utc`
+
+For rejected runs, the reviewer should also fill the small failure-analysis fields when useful:
+
+- `failure_type`
+- `failure_symptom`
+- `likely_cause`
+- `recommended_next_action`
+
+This keeps execution facts and review judgment in one durable record.
+
+## Connection To The Execution Record
+
+This gate depends on the separation already defined in [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:77):
+
+- execution status answers "what happened when the runner executed the prompt?"
+- review status answers "did a human accept this bounded step as good enough to progress?"
+
+The V1 review gate is therefore not a new artifact.
+It is the manual transition of one execution record from `UNREVIEWED` to either `ACCEPTED` or `REJECTED`.
+
+## Intentionally Deferred
+
+V1 intentionally defers:
+
+- automatic queue release
+- automatic reviewer assignment
+- separate queue-level state files
+- richer review outcomes such as `PARTIAL`, `BLOCKED`, or `NEEDS_RETRY`
+- policy for choosing among multiple retry or redesign options
+- automatic extraction of validation evidence from repo changes
+- broader workflow redesign beyond the next reviewed step
+
+## V1 Decision Summary
+
+The minimum V1 review gate is:
+
+1. execution finishes
+2. a run record is written with `review_status: UNREVIEWED`
+3. human review applies the minimum checklist
+4. reviewer writes back `ACCEPTED` or `REJECTED`
+5. only `ACCEPTED` makes the next prompt ready
+
+That is small enough to guide the next implementation prompt without expanding the system into a larger workflow engine.

codex
The note is in place. I’m doing one pass to verify it answers each required design question and stays within the V1/manual-only boundary.
exec
/bin/bash -lc "sed -n '1,260p' tools/codex/V1_Run_Review_Gate.md" in /data/git/handson-ml3
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

diff --git a/tools/codex/V1_Run_Review_Gate.md b/tools/codex/V1_Run_Review_Gate.md
new file mode 100644
index 0000000000000000000000000000000000000000..8738b403163228dba1600691d8ff591f37d0c37f
--- /dev/null
+++ b/tools/codex/V1_Run_Review_Gate.md
@@ -0,0 +1,161 @@
+# V1 Run Review Gate
+
+## Purpose
+
+Define the smallest practical review gate between one prompt run finishing and the next prompt in sequence being treated as ready.
+
+This gate exists to preserve three things from the governing workflow:
+
+- review between iterations
+- separation of execution outcome from accepted outcome
+- a conservative human stop before queue progression
+
+## Scope
+
+V1 covers only the decision point immediately after one run record is written.
+
+It defines:
+
+- when a run enters review
+- the minimum manual checks
+- the allowed review outcomes
+- how those outcomes affect whether the next prompt may proceed
+- what review information must be written into the run record
+
+It does not define:
+
+- automation of review
+- a larger queue state machine
+- dependency-aware scheduling
+- retry orchestration
+- richer review taxonomies such as `PARTIAL` or `BLOCKED`
+
+## Review Trigger
+
+A run becomes `awaiting review` when its execution phase has finished and its V1 execution record has been written to `notes/` with:
+
+- `execution_status` set to either `EXECUTED` or `EXECUTION_FAILED`
+- `review_status` still set to `UNREVIEWED`
+
+That is the exact review trigger for V1.
+
+Process exit alone does not make the next prompt ready.
+Record creation plus `UNREVIEWED` review status means the run must stop for human inspection.
+
+## Minimum Manual Checklist
+
+Before a run can be accepted, a human reviewer must check only these items:
+
+1. Confirm the execution record is complete enough to review.
+   Required minimum: prompt identity, execution status, return code, prompt text, Codex final output, and any captured stderr.
+
+2. Check whether the run actually addressed the prompt that was executed.
+   This is a scope-and-intent check, not a deep redesign review.
+
+3. Check whether there is enough evidence to treat the result as an accepted outcome for the current bounded step.
+   Minimum evidence may be direct output inspection, referenced file changes, or stated validation evidence in the run record or surrounding repo state.
+
+4. Check whether any obvious failure, mismatch, or ambiguity remains that would make automatic progression unsafe.
+   Examples: execution failed, output is incomplete, acceptance is unclear, validation is missing, or the result suggests the design/task should be refined before continuing.
+
+This checklist is intentionally small.
+V1 only needs enough manual review to keep execution success separate from accepted progress.
+
+## Allowed V1 Review Outcomes
+
+V1 should allow exactly these review outcomes:
+
+- `UNREVIEWED`
+- `ACCEPTED`
+- `REJECTED`
+
+Meaning:
+
+- `UNREVIEWED`: default state after execution record creation; queue must stop here.
+- `ACCEPTED`: the bounded step is accepted after human review.
+- `REJECTED`: the run is not accepted for progression, whether because execution failed, the result is inadequate, or the outcome requires redesign or retry analysis first.
+
+No additional V1 outcomes are needed.
+
+## Queue Progression Rule
+
+Only `ACCEPTED` allows the next prompt in sequence to be treated as ready.
+
+Rules:
+
+- `UNREVIEWED` stops the queue pending manual review.
+- `ACCEPTED` releases exactly the next reviewed step.
+- `REJECTED` stops the queue and forces an explicit human decision about what happens next.
+
+`EXECUTED` does not release the queue.
+`EXECUTION_FAILED` does not itself decide the queue either, but it will usually lead to `REJECTED` on review.
+
+## Stop-And-Decide Rule
+
+`REJECTED` is the V1 outcome that stops the queue and forces a new design or retry decision.
+
+That decision is intentionally outside this gate.
+It may result in:
+
+- a revised prompt
+- a smaller follow-up slice
+- an environment fix
+- a deliberate retry
+- an architecture or scope adjustment
+
+V1 only requires that the queue does not continue past a rejected run.
+
+## Run Record Write-Back
+
+During review, the run record defined by [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md) must be updated with these review facts:
+
+- `review_status`
+- `review_summary`
+
+When available, V1 should also write:
+
+- `reviewed_by`
+- `reviewed_at_utc`
+
+For rejected runs, the reviewer should also fill the small failure-analysis fields when useful:
+
+- `failure_type`
+- `failure_symptom`
+- `likely_cause`
+- `recommended_next_action`
+
+This keeps execution facts and review judgment in one durable record.
+
+## Connection To The Execution Record
+
+This gate depends on the separation already defined in [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:77):
+
+- execution status answers "what happened when the runner executed the prompt?"
+- review status answers "did a human accept this bounded step as good enough to progress?"
+
+The V1 review gate is therefore not a new artifact.
+It is the manual transition of one execution record from `UNREVIEWED` to either `ACCEPTED` or `REJECTED`.
+
+## Intentionally Deferred
+
+V1 intentionally defers:
+
+- automatic queue release
+- automatic reviewer assignment
+- separate queue-level state files
+- richer review outcomes such as `PARTIAL`, `BLOCKED`, or `NEEDS_RETRY`
+- policy for choosing among multiple retry or redesign options
+- automatic extraction of validation evidence from repo changes
+- broader workflow redesign beyond the next reviewed step
+
+## V1 Decision Summary
+
+The minimum V1 review gate is:
+
+1. execution finishes
+2. a run record is written with `review_status: UNREVIEWED`
+3. human review applies the minimum checklist
+4. reviewer writes back `ACCEPTED` or `REJECTED`
+5. only `ACCEPTED` makes the next prompt ready
+
+That is small enough to guide the next implementation prompt without expanding the system into a larger workflow engine.

codex
Created [V1_Run_Review_Gate.md](/data/git/handson-ml3/tools/codex/V1_Run_Review_Gate.md).

It defines the minimal V1 gate as: execution finishes, a run record is written with `review_status: UNREVIEWED`, a human performs a short manual checklist, and only `ACCEPTED` allows the next prompt to be treated as ready. `REJECTED` is the stop condition that forces an explicit redesign or retry decision, and the note ties that directly back to the fields in [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md).

No code or CLI changes were made.
diff --git a/tools/codex/V1_Run_Review_Gate.md b/tools/codex/V1_Run_Review_Gate.md
new file mode 100644
index 0000000000000000000000000000000000000000..8738b403163228dba1600691d8ff591f37d0c37f
--- /dev/null
+++ b/tools/codex/V1_Run_Review_Gate.md
@@ -0,0 +1,161 @@
+# V1 Run Review Gate
+
+## Purpose
+
+Define the smallest practical review gate between one prompt run finishing and the next prompt in sequence being treated as ready.
+
+This gate exists to preserve three things from the governing workflow:
+
+- review between iterations
+- separation of execution outcome from accepted outcome
+- a conservative human stop before queue progression
+
+## Scope
+
+V1 covers only the decision point immediately after one run record is written.
+
+It defines:
+
+- when a run enters review
+- the minimum manual checks
+- the allowed review outcomes
+- how those outcomes affect whether the next prompt may proceed
+- what review information must be written into the run record
+
+It does not define:
+
+- automation of review
+- a larger queue state machine
+- dependency-aware scheduling
+- retry orchestration
+- richer review taxonomies such as `PARTIAL` or `BLOCKED`
+
+## Review Trigger
+
+A run becomes `awaiting review` when its execution phase has finished and its V1 execution record has been written to `notes/` with:
+
+- `execution_status` set to either `EXECUTED` or `EXECUTION_FAILED`
+- `review_status` still set to `UNREVIEWED`
+
+That is the exact review trigger for V1.
+
+Process exit alone does not make the next prompt ready.
+Record creation plus `UNREVIEWED` review status means the run must stop for human inspection.
+
+## Minimum Manual Checklist
+
+Before a run can be accepted, a human reviewer must check only these items:
+
+1. Confirm the execution record is complete enough to review.
+   Required minimum: prompt identity, execution status, return code, prompt text, Codex final output, and any captured stderr.
+
+2. Check whether the run actually addressed the prompt that was executed.
+   This is a scope-and-intent check, not a deep redesign review.
+
+3. Check whether there is enough evidence to treat the result as an accepted outcome for the current bounded step.
+   Minimum evidence may be direct output inspection, referenced file changes, or stated validation evidence in the run record or surrounding repo state.
+
+4. Check whether any obvious failure, mismatch, or ambiguity remains that would make automatic progression unsafe.
+   Examples: execution failed, output is incomplete, acceptance is unclear, validation is missing, or the result suggests the design/task should be refined before continuing.
+
+This checklist is intentionally small.
+V1 only needs enough manual review to keep execution success separate from accepted progress.
+
+## Allowed V1 Review Outcomes
+
+V1 should allow exactly these review outcomes:
+
+- `UNREVIEWED`
+- `ACCEPTED`
+- `REJECTED`
+
+Meaning:
+
+- `UNREVIEWED`: default state after execution record creation; queue must stop here.
+- `ACCEPTED`: the bounded step is accepted after human review.
+- `REJECTED`: the run is not accepted for progression, whether because execution failed, the result is inadequate, or the outcome requires redesign or retry analysis first.
+
+No additional V1 outcomes are needed.
+
+## Queue Progression Rule
+
+Only `ACCEPTED` allows the next prompt in sequence to be treated as ready.
+
+Rules:
+
+- `UNREVIEWED` stops the queue pending manual review.
+- `ACCEPTED` releases exactly the next reviewed step.
+- `REJECTED` stops the queue and forces an explicit human decision about what happens next.
+
+`EXECUTED` does not release the queue.
+`EXECUTION_FAILED` does not itself decide the queue either, but it will usually lead to `REJECTED` on review.
+
+## Stop-And-Decide Rule
+
+`REJECTED` is the V1 outcome that stops the queue and forces a new design or retry decision.
+
+That decision is intentionally outside this gate.
+It may result in:
+
+- a revised prompt
+- a smaller follow-up slice
+- an environment fix
+- a deliberate retry
+- an architecture or scope adjustment
+
+V1 only requires that the queue does not continue past a rejected run.
+
+## Run Record Write-Back
+
+During review, the run record defined by [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md) must be updated with these review facts:
+
+- `review_status`
+- `review_summary`
+
+When available, V1 should also write:
+
+- `reviewed_by`
+- `reviewed_at_utc`
+
+For rejected runs, the reviewer should also fill the small failure-analysis fields when useful:
+
+- `failure_type`
+- `failure_symptom`
+- `likely_cause`
+- `recommended_next_action`
+
+This keeps execution facts and review judgment in one durable record.
+
+## Connection To The Execution Record
+
+This gate depends on the separation already defined in [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:77):
+
+- execution status answers "what happened when the runner executed the prompt?"
+- review status answers "did a human accept this bounded step as good enough to progress?"
+
+The V1 review gate is therefore not a new artifact.
+It is the manual transition of one execution record from `UNREVIEWED` to either `ACCEPTED` or `REJECTED`.
+
+## Intentionally Deferred
+
+V1 intentionally defers:
+
+- automatic queue release
+- automatic reviewer assignment
+- separate queue-level state files
+- richer review outcomes such as `PARTIAL`, `BLOCKED`, or `NEEDS_RETRY`
+- policy for choosing among multiple retry or redesign options
+- automatic extraction of validation evidence from repo changes
+- broader workflow redesign beyond the next reviewed step
+
+## V1 Decision Summary
+
+The minimum V1 review gate is:
+
+1. execution finishes
+2. a run record is written with `review_status: UNREVIEWED`
+3. human review applies the minimum checklist
+4. reviewer writes back `ACCEPTED` or `REJECTED`
+5. only `ACCEPTED` makes the next prompt ready
+
+That is small enough to guide the next implementation prompt without expanding the system into a larger workflow engine.

tokens used
35,050
```
