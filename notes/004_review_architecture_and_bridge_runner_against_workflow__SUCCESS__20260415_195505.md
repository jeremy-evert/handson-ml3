# 004_review_architecture_and_bridge_runner_against_workflow - SUCCESS

- Prompt file: `004_review_architecture_and_bridge_runner_against_workflow.md`
- Timestamp (UTC): `20260415_195505`
- Status: `SUCCESS`

## Original Prompt

```md
# Task: Review the Codex prompt workflow architecture and the baby bridge runner against the current project design workflow

You are working in this repository.

Your task is to review two files against the current workflow guidance and produce a small design assessment.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/baby_run_prompt.py`

## Goal

Use `tools/Project_Design_Workflow.md` as the governing workflow document.

Then inspect:

- the architecture we say we want
- the bridge runner we actually have

Your job is to identify:

1. where the architecture and runner align with the workflow
2. where they do not align
3. what the smallest next design step should be before more implementation happens

## Important framing

This is a design-review task, not a large implementation task.

Do NOT do a broad rewrite of the system.
Do NOT build multiple new modules.
Do NOT refactor the runner in this pass.

This pass should stay focused on assessment, mismatch detection, and next-step recommendation.

## What to look for

Please evaluate the architecture doc and the bridge runner using the workflow principles in `tools/Project_Design_Workflow.md`, especially:

- design before build
- boundaries before breadth
- thin slices before large pushes
- review between iterations
- bridge tooling is allowed, but subordinate
- durable local history matters
- failure should produce analysis, not just retries
- resource use should be observed

## Questions to answer

Please answer these questions in the report:

### 1. Architecture alignment
- Does `tools/Codex_Prompt_Workflow_Architecture.md` reflect the workflow in `tools/Project_Design_Workflow.md`?
- Where is it strong?
- Where is it stale, incomplete, or misaligned?

### 2. Runner alignment
- Does `tools/codex/baby_run_prompt.py` behave like acceptable bridge tooling under the workflow?
- What parts of it are useful and appropriately thin?
- What responsibilities is it currently carrying that should eventually move elsewhere?

### 3. Failure-analysis support
- Does the current runner/workflow setup support useful post-failure analysis?
- If not, what is the smallest next improvement that would help?

### 4. Resource-awareness support
- Does the current setup preserve any useful execution-cost evidence?
- What is the smallest next improvement that would let us track lightweight metrics such as runtime, retries, token usage if available, or output size?

### 5. Smallest next design step
- What is the single best next design artifact or design decision to create before more implementation?
- Prefer one bounded next step, not a large roadmap.

## Required output artifacts

Please create exactly two artifacts.

### Artifact 1
Create a markdown report at:

`notes/004_architecture_and_bridge_runner_review__TIMESTAMP.md`

This report should include:

- a short summary
- architecture alignment findings
- runner alignment findings
- failure-analysis findings
- resource-awareness findings
- the single recommended next design step

Keep it practical and inspectable.

### Artifact 2
Create a short markdown file at:

`notes/004_next_design_step_recommendation__TIMESTAMP.md`

This should contain only:

- the recommended next step
- why it should happen next
- what artifact it should produce
- what should explicitly wait until later

Keep this one brief and decisive.

## Constraints

1. Use the exact file paths listed above.
2. Do not edit the workflow doc in this pass.
3. Do not perform a major refactor in this pass.
4. Do not produce a giant implementation plan.
5. Prefer specific observations over vague advice.
6. Keep the recommendation bounded enough that it can become the next Codex prompt.

## Success criteria

This task is successful if:

- the report clearly compares desired architecture vs current runner
- the assessment is grounded in `tools/Project_Design_Workflow.md`
- the next step is small, concrete, and design-focused
- the outputs help us decide the next prompt with confidence
```

## Codex Output

Created the two requested artifacts:

- [004_architecture_and_bridge_runner_review__20260415_195538.md](/data/git/handson-ml3/notes/004_architecture_and_bridge_runner_review__20260415_195538.md)
- [004_next_design_step_recommendation__20260415_195538.md](/data/git/handson-ml3/notes/004_next_design_step_recommendation__20260415_195538.md)

The assessment’s main conclusion is that the architecture doc is directionally right on decomposition and conservative growth, and the runner is thin enough to count as acceptable bridge tooling, but both are weak at the same seam: they do not yet define a durable, reviewable execution record that separates raw execution from reviewed outcome and captures minimal failure/resource evidence.

I did not modify the workflow doc or refactor the runner. The recommendation is intentionally bounded: define the V1 execution-record artifact next, before more implementation.

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
session id: 019d92b6-2caf-7452-b496-1a1a9adc8574
--------
user
# Task: Review the Codex prompt workflow architecture and the baby bridge runner against the current project design workflow

You are working in this repository.

Your task is to review two files against the current workflow guidance and produce a small design assessment.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/baby_run_prompt.py`

## Goal

Use `tools/Project_Design_Workflow.md` as the governing workflow document.

Then inspect:

- the architecture we say we want
- the bridge runner we actually have

Your job is to identify:

1. where the architecture and runner align with the workflow
2. where they do not align
3. what the smallest next design step should be before more implementation happens

## Important framing

This is a design-review task, not a large implementation task.

Do NOT do a broad rewrite of the system.
Do NOT build multiple new modules.
Do NOT refactor the runner in this pass.

This pass should stay focused on assessment, mismatch detection, and next-step recommendation.

## What to look for

Please evaluate the architecture doc and the bridge runner using the workflow principles in `tools/Project_Design_Workflow.md`, especially:

- design before build
- boundaries before breadth
- thin slices before large pushes
- review between iterations
- bridge tooling is allowed, but subordinate
- durable local history matters
- failure should produce analysis, not just retries
- resource use should be observed

## Questions to answer

Please answer these questions in the report:

### 1. Architecture alignment
- Does `tools/Codex_Prompt_Workflow_Architecture.md` reflect the workflow in `tools/Project_Design_Workflow.md`?
- Where is it strong?
- Where is it stale, incomplete, or misaligned?

### 2. Runner alignment
- Does `tools/codex/baby_run_prompt.py` behave like acceptable bridge tooling under the workflow?
- What parts of it are useful and appropriately thin?
- What responsibilities is it currently carrying that should eventually move elsewhere?

### 3. Failure-analysis support
- Does the current runner/workflow setup support useful post-failure analysis?
- If not, what is the smallest next improvement that would help?

### 4. Resource-awareness support
- Does the current setup preserve any useful execution-cost evidence?
- What is the smallest next improvement that would let us track lightweight metrics such as runtime, retries, token usage if available, or output size?

### 5. Smallest next design step
- What is the single best next design artifact or design decision to create before more implementation?
- Prefer one bounded next step, not a large roadmap.

## Required output artifacts

Please create exactly two artifacts.

### Artifact 1
Create a markdown report at:

`notes/004_architecture_and_bridge_runner_review__TIMESTAMP.md`

This report should include:

- a short summary
- architecture alignment findings
- runner alignment findings
- failure-analysis findings
- resource-awareness findings
- the single recommended next design step

Keep it practical and inspectable.

### Artifact 2
Create a short markdown file at:

`notes/004_next_design_step_recommendation__TIMESTAMP.md`

This should contain only:

- the recommended next step
- why it should happen next
- what artifact it should produce
- what should explicitly wait until later

Keep this one brief and decisive.

## Constraints

1. Use the exact file paths listed above.
2. Do not edit the workflow doc in this pass.
3. Do not perform a major refactor in this pass.
4. Do not produce a giant implementation plan.
5. Prefer specific observations over vague advice.
6. Keep the recommendation bounded enough that it can become the next Codex prompt.

## Success criteria

This task is successful if:

- the report clearly compares desired architecture vs current runner
- the assessment is grounded in `tools/Project_Design_Workflow.md`
- the next step is small, concrete, and design-focused
- the outputs help us decide the next prompt with confidence

codex
I’m reviewing the workflow doc first, then comparing the architecture note and the baby bridge runner directly against it. After that I’ll write the two requested markdown artifacts under `notes/` with a bounded recommendation, without changing the workflow or refactoring code.
  → Read governing workflow, architecture doc, and bridge runner
  • Assess alignment and gaps against workflow principles
  • Write the two required markdown artifacts in notes/
exec
/bin/bash -lc "sed -n '1,260p' tools/Codex_Prompt_Workflow_Architecture.md" in /data/git/handson-ml3
exec
/bin/bash -lc "sed -n '1,260p' tools/codex/baby_run_prompt.py" in /data/git/handson-ml3
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
/bin/bash -lc "sed -n '1,260p' tools/Project_Design_Workflow.md" in /data/git/handson-ml3
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

codex
I have the governing principles and the current runner shape. I’m pulling the rest of the workflow guidance now so the assessment stays tied to the full sequence, especially validation, review loops, failure handling, and artifact expectations.
exec
/bin/bash -lc "sed -n '261,520p' tools/Project_Design_Workflow.md" in /data/git/handson-ml3
 succeeded in 0ms:
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
* output quality
* machine or service usage

In some environments, teams may also track:

* token usage
* compute time
* API cost
* artifact volume

The purpose is not accounting for its own sake.
The purpose is to notice whether larger, more expensive, or less bounded tasks correlate with more failures, lower quality, or weaker reviewability.

If those patterns appear, the workflow should respond by shrinking task size, clarifying criteria, or improving decomposition.

---

## The Right Questions Before Building

Before implementation begins, these questions should be answered.

### 1. What is the smallest version that is truly useful?

Do not build past that line yet.

### 2. What is the source of truth?

What artifacts, interfaces, or records define reality for this system?

### 3. What is the stable identity of each important object?

Names, IDs, paths, interfaces, and records should not be ambiguous.

### 4. What states do we actually need in V1?

Probably fewer than we are tempted to add.

### 5. What should remain manually controlled?

Human judgment should stay in the loop where truth is fuzzy.

### 6. What bridge tooling is acceptable?

If temporary helpers are needed, how do we keep them thin, inspectable, and easy to replace?

### 7. What history should be preserved?

What notes, outputs, validation records, or failure reports will be worth having later?

### 8. What is the next thing we are intentionally not building?

This keeps the scope fence visible.

---

## Design Workflow as a Reusable Template

For many projects, a reliable progression is:

1. conversation
2. problem statement
3. goals
4. scope and boundaries
5. architecture
6. workflow and build plan
7. artifact list
8. implementation sequence
9. validation plan
10. bounded execution loop
11. review and refinement
12. failure analysis when needed

This is slow in the beginning and fast later.
It feels like more thought up front because it is.
But it prevents the kind of momentum that sends a project downhill with shopping carts tied to its feet.

---

## Closing Thought

A good design workflow does not slow building down.
It moves confusion to the front, where it is cheap.

That is the whole game.

Think clearly first.
Name the parts.
Set the boundaries.
Choose the first slice.
Review each iteration.
Study failure instead of hiding it.
Build only what is earned.
Then move forward with confidence.

exec
/bin/bash -lc "nl -ba tools/Project_Design_Workflow.md | sed -n '1,260p'" in /data/git/handson-ml3
 succeeded in 0ms:
     1	# Project Design Workflow
     2	
     3	## Goal
     4	
     5	Define a repeatable workflow for turning an idea or architecture document into a clean, working system without skipping the thinking that keeps the build safe, clear, and reusable.
     6	
     7	This document is meant to sit beside architecture documents and help answer a practical question:
     8	
     9	**What are the steps between "this design makes sense" and "the thing works"?**
    10	
    11	---
    12	
    13	## Why This Exists
    14	
    15	Sometimes a project feels sound at the architecture level but still feels fuzzy at the build level.
    16	
    17	That happens because architecture is not the same thing as execution.
    18	
    19	Architecture tells us:
    20	
    21	* what parts exist
    22	* what they are responsible for
    23	* how they relate
    24	* what should not be coupled
    25	
    26	But architecture does not yet tell us:
    27	
    28	* what to build first
    29	* what order reduces risk
    30	* what to test first
    31	* what can wait
    32	* how to know whether the design is holding
    33	* how to recover cleanly when a step fails
    34	
    35	This workflow bridges that gap.
    36	
    37	---
    38	
    39	## Relationship to a Larger Design Process
    40	
    41	This workflow does **not** replace broader project thinking.
    42	It fits inside it.
    43	
    44	A useful mapping looks like this:
    45	
    46	### 1. Conversation
    47	
    48	What are we trying to do?
    49	What hurts right now?
    50	What would "better" feel like?
    51	
    52	### 2. Goals
    53	
    54	What must become true for this project to count as successful?
    55	
    56	### 3. Deliverables
    57	
    58	What concrete artifacts must exist?
    59	Examples:
    60	
    61	* architecture doc
    62	* workflow doc
    63	* module layout
    64	* interface sketch
    65	* notes folder
    66	* validation checklist
    67	* first working slice
    68	
    69	### 4. Tasks
    70	
    71	What actions create those deliverables?
    72	Examples:
    73	
    74	* define V1 scope
    75	* define module boundaries
    76	* choose naming conventions
    77	* write first thin component
    78	* validate one path end to end
    79	* capture review notes
    80	
    81	### 5. Implementation
    82	
    83	Only after the earlier layers are stable do we write implementation.
    84	
    85	So this workflow is not separate from design.
    86	It is the part that turns design into bounded execution.
    87	
    88	---
    89	
    90	## Core Principles
    91	
    92	### 1. Design before build
    93	
    94	Do not use implementation momentum to hide unclear thinking.
    95	
    96	### 2. Boundaries before breadth
    97	
    98	A project becomes safer when scope, responsibility, and deferral lines are visible.
    99	
   100	### 3. Thin slices before large pushes
   101	
   102	Prefer a bounded slice that proves something real over a large batch that creates many unknowns at once.
   103	
   104	### 4. Review between iterations
   105	
   106	Each bounded execution step should be inspected before the next one is issued.
   107	Do not assume a sequence is healthy just because the previous step completed.
   108	
   109	### 5. Validation is part of design
   110	
   111	Tests, inspections, examples, and acceptance checks are not extra work.
   112	They are how the design proves it is real.
   113	
   114	### 6. Bridge tooling is allowed, but subordinate
   115	
   116	Temporary or bridge tooling can be useful when it stays thin, inspectable, and reversible.
   117	It may help move work forward, gather evidence, or reduce manual friction.
   118	It should not become a substitute for architecture, clear interfaces, or good decomposition.
   119	
   120	### 7. Durable local history matters
   121	
   122	Notes, logs, reports, generated outputs, and similar local artifacts can serve as durable project memory.
   123	They help preserve decisions, validation results, failure patterns, and the reasoning behind refinements.
   124	
   125	### 8. Failure should produce analysis, not just retries
   126	
   127	When a step fails, the useful question is not only "how do we try again?"
   128	It is also "what did this failure reveal about the task, decomposition, criteria, or environment?"
   129	
   130	### 9. Resource use should be observed
   131	
   132	Project effort has a cost, whether that cost is human time, machine time, review effort, retries, execution size, or external spend.
   133	Large tasks should justify themselves.
   134	
   135	---
   136	
   137	## The Core Sequence
   138	
   139	Here is the recommended design workflow.
   140	
   141	### Phase 1: Clarify the problem
   142	
   143	This is the "why are we doing this?" phase.
   144	
   145	Questions:
   146	
   147	* What pain are we removing?
   148	* What confusion are we reducing?
   149	* What repeated work are we trying to standardize?
   150	* What is dangerous if we build too fast?
   151	
   152	Output:
   153	
   154	* short problem statement
   155	* short success statement
   156	
   157	---
   158	
   159	### Phase 2: Define the system boundary
   160	
   161	This is the "what belongs in this project and what does not?" phase.
   162	
   163	Questions:
   164	
   165	* What should this system do?
   166	* What should it explicitly not do yet?
   167	* What decisions are deferred?
   168	* What adjacent problems are real but out of scope?
   169	
   170	Output:
   171	
   172	* scope statement
   173	* out-of-scope list
   174	
   175	---
   176	
   177	### Phase 3: Draft the architecture
   178	
   179	This is the "what pieces exist and what are their jobs?" phase.
   180	
   181	Questions:
   182	
   183	* What modules, components, or services are needed?
   184	* What is each one responsible for?
   185	* What should each one never own?
   186	* Where are the seams between parts?
   187	
   188	Output:
   189	
   190	* architecture doc
   191	* responsibility split
   192	* proposed file, package, or interface layout
   193	
   194	---
   195	
   196	### Phase 4: Identify the minimum viable slice
   197	
   198	This is the "what is the smallest useful thing we can build that proves the design?" phase.
   199	
   200	Questions:
   201	
   202	* What is the thinnest vertical slice that is actually useful?
   203	* What can we test without building the whole system?
   204	* What gives us signal early?
   205	* What first slice is unlikely to trap us later?
   206	
   207	Output:
   208	
   209	* V1 feature list
   210	* initial build plan
   211	
   212	---
   213	
   214	### Phase 5: Define the artifacts
   215	
   216	This is the "what files or outputs must exist?" phase.
   217	
   218	Questions:
   219	
   220	* What documents should exist?
   221	* What modules or interfaces should exist?
   222	* What examples, fixtures, or test inputs should exist?
   223	* What outputs prove the path works?
   224	* What notes, logs, or reports should be kept as durable project memory?
   225	
   226	Output:
   227	
   228	* deliverables list
   229	* artifact inventory
   230	
   231	---
   232	
   233	### Phase 6: Sequence the work
   234	
   235	This is the "what order reduces pain and risk?" phase.
   236	
   237	Questions:
   238	
   239	* What must come first because other things depend on it?
   240	* What can be tested independently?
   241	* What pieces should be proven before automation or convenience layers are added?
   242	* What order keeps the build inspectable?
   243	
   244	Output:
   245	
   246	* implementation order
   247	* dependency chain
   248	
   249	---
   250	
   251	### Phase 7: Define validation
   252	
   253	This is the "how will we know each layer works?" phase.
   254	
   255	Questions:
   256	
   257	* What is the smoke test for each part?
   258	* What is a good manual test before automation?
   259	* What failure modes do we expect?
   260	* What evidence counts as success?

exec
/bin/bash -lc "nl -ba tools/Codex_Prompt_Workflow_Architecture.md | sed -n '1,260p'" in /data/git/handson-ml3
 succeeded in 0ms:
     1	# Codex Prompt Workflow Architecture
     2	
     3	## Goal
     4	
     5	Build a small, clean, reusable prompt workflow system for Codex work inside a repository.
     6	
     7	This system should be conservative first:
     8	
     9	* easy to inspect
    10	* easy to test
    11	* easy to extend
    12	* composed of small parts
    13	* safe by default
    14	
    15	The design should emphasize **separation of concerns** so each piece does one job clearly.
    16	
    17	---
    18	
    19	## Why We Are Re-Architecting
    20	
    21	The first draft bundled too many responsibilities into one script. That made it harder to:
    22	
    23	* understand
    24	* trust
    25	* test
    26	* evolve
    27	* reuse across repositories
    28	
    29	We want a cleaner structure that can become a template for future repos.
    30	
    31	---
    32	
    33	## What the First Script Was Doing
    34	
    35	Below is a decomposition of the responsibilities that were bundled together.
    36	
    37	### 1. Repo path discovery
    38	
    39	* figure out where the repository root is
    40	* infer where `codex_prompts/` and `notes/` live
    41	
    42	### 2. Directory validation
    43	
    44	* verify the expected folders exist
    45	* fail cleanly if they do not
    46	
    47	### 3. Prompt discovery
    48	
    49	* scan `codex_prompts/`
    50	* identify valid prompt files
    51	* ignore files that do not match the naming convention
    52	
    53	### 4. Prompt parsing and indexing
    54	
    55	* extract numeric prefixes from filenames
    56	* sort prompts in execution order
    57	* establish a stable prompt identity from filename/stem
    58	
    59	### 5. Note discovery
    60	
    61	* scan `notes/`
    62	* find note files that match the naming convention
    63	* ignore unrelated markdown files
    64	
    65	### 6. Note parsing
    66	
    67	* extract prompt name, success/fail state, and timestamp from note filenames
    68	* turn note filenames into structured metadata
    69	
    70	### 7. Status reconstruction
    71	
    72	* map prompts to matching notes
    73	* determine whether each prompt is:
    74	
    75	  * UNRUN
    76	  * SUCCESS
    77	  * FAIL
    78	* pick the latest note when there are multiple notes for one prompt
    79	
    80	### 8. Prompt selection
    81	
    82	* find the next unrun prompt
    83	* find the first failed prompt
    84	* find a prompt by numeric prefix, base name, or full filename
    85	
    86	### 9. Prompt display
    87	
    88	* print a selected prompt to the terminal
    89	
    90	### 10. Retry context assembly
    91	
    92	* find the latest failed note for a prompt
    93	* display the original prompt plus the latest failed note
    94	* present instructions for retrying
    95	
    96	### 11. Note writing
    97	
    98	* create timestamped note filenames
    99	* write notes in markdown
   100	* optionally include the previous note as context
   101	
   102	### 12. Manual status marking
   103	
   104	* let the user record a success or failure
   105	* attach summary/details text to the note
   106	
   107	### 13. Command-line interface
   108	
   109	* parse subcommands like `list`, `next`, `show`, `mark`, `retry-failed`
   110	* route commands to the right behaviors
   111	
   112	### 14. Terminal presentation
   113	
   114	* format output for human readability
   115	* present status tables and retry blocks
   116	
   117	---
   118	
   119	## What I Would Like This System To Do Eventually
   120	
   121	These are ideas worth considering, but they should not all be built at once.
   122	
   123	### A. Better architecture and reuse
   124	
   125	* work as a reusable template across repositories
   126	* isolate config from code
   127	* support per-repo conventions without rewriting internals
   128	
   129	### B. Structured metadata
   130	
   131	* optionally store prompt metadata in frontmatter or sidecar files
   132	* support fields like:
   133	
   134	  * title
   135	  * objective
   136	  * tags
   137	  * priority
   138	  * dependencies
   139	  * owner
   140	
   141	### C. Better status model
   142	
   143	Instead of only:
   144	
   145	* UNRUN
   146	* SUCCESS
   147	* FAIL
   148	
   149	We may want:
   150	
   151	* READY
   152	* RUNNING
   153	* BLOCKED
   154	* NEEDS_REVIEW
   155	* PARTIAL
   156	* SKIPPED
   157	* ARCHIVED
   158	
   159	### D. Retry intelligence
   160	
   161	* carry forward the previous failed note automatically
   162	* include multiple prior failures, not just the latest one
   163	* summarize recurring failure patterns
   164	
   165	### E. Prompt preparation
   166	
   167	* generate a clean “execution bundle” for Codex
   168	* include:
   169	
   170	  * prompt text
   171	  * repo context
   172	  * previous failure note if retrying
   173	  * explicit success criteria
   174	
   175	### F. Manual and automated modes
   176	
   177	* manual mode for conservative workflows
   178	* dry-run mode for seeing what would happen
   179	* later, optional Codex CLI integration
   180	
   181	### G. Report generation
   182	
   183	* generate queue reports
   184	* generate status snapshots
   185	* generate audit/history reports over prompt activity
   186	
   187	### H. Better note/content model
   188	
   189	* move from filename-only state to richer note contents
   190	* enforce a consistent note template
   191	* optionally add structured machine-readable metadata inside notes
   192	
   193	### I. Dependency and sequencing support
   194	
   195	* allow prompts to depend on other prompts
   196	* prevent prompts from running before prerequisites are complete
   197	
   198	### J. Prompt families or lanes
   199	
   200	* support categories such as:
   201	
   202	  * setup
   203	  * audit
   204	  * refactor
   205	  * notebook generation
   206	  * documentation
   207	
   208	### K. Approval gates
   209	
   210	* require human confirmation before marking success
   211	* require review before allowing dependent prompts to proceed
   212	
   213	### L. Logging and diagnostics
   214	
   215	* maintain an execution log
   216	* record tool errors separately from task failures
   217	* distinguish infrastructure failure from prompt failure
   218	
   219	### M. File layout that scales cleanly
   220	
   221	* support growth without turning into a junk drawer
   222	
   223	---
   224	
   225	## Proposed Folder Architecture
   226	
   227	A better first structure would be:
   228	
   229	```text
   230	tools/
   231	  codex/
   232	    README.md
   233	    architecture.md
   234	    config.py
   235	    paths.py
   236	    prompts.py
   237	    notes.py
   238	    status.py
   239	    retry.py
   240	    cli.py
   241	    templates/
   242	      note_success.md
   243	      note_fail.md
   244	```
   245	
   246	For an even more conservative start, we can go smaller:
   247	
   248	```text
   249	tools/
   250	  codex/
   251	    README.md
   252	    architecture.md
   253	    paths.py
   254	    prompts.py
   255	    notes.py
   256	    status.py
   257	```
   258	
   259	Then add a tiny entrypoint later if needed.
   260	

exec
/bin/bash -lc "nl -ba tools/codex/baby_run_prompt.py | sed -n '1,220p'" in /data/git/handson-ml3
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	
     3	from __future__ import annotations
     4	
     5	import argparse
     6	import datetime as dt
     7	import subprocess
     8	import sys
     9	import tempfile
    10	from pathlib import Path
    11	
    12	
    13	PROMPTS_DIR = "codex_prompts"
    14	NOTES_DIR = "notes"
    15	
    16	
    17	def utc_timestamp() -> str:
    18	    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    19	
    20	
    21	def repo_root() -> Path:
    22	    return Path(__file__).resolve().parents[2]
    23	
    24	
    25	def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
    26	    prompt = Path(prompt_arg)
    27	
    28	    if prompt.is_absolute():
    29	        return prompt
    30	
    31	    direct = root / prompt
    32	    if direct.exists():
    33	        return direct
    34	
    35	    in_prompts = root / PROMPTS_DIR / prompt
    36	    if in_prompts.exists():
    37	        return in_prompts
    38	
    39	    matches = sorted((root / PROMPTS_DIR).glob(f"{prompt_arg}*"))
    40	    if len(matches) == 1:
    41	        return matches[0]
    42	
    43	    return in_prompts
    44	
    45	
    46	def build_note_path(root: Path, prompt_path: Path, status: str, timestamp: str) -> Path:
    47	    return root / NOTES_DIR / f"{prompt_path.stem}__{status}__{timestamp}.md"
    48	
    49	
    50	def build_note_content(
    51	    prompt_path: Path,
    52	    timestamp: str,
    53	    status: str,
    54	    prompt_text: str,
    55	    codex_output: str,
    56	    notes: str = "",
    57	) -> str:
    58	    return f"""# {prompt_path.stem} - {status}
    59	
    60	- Prompt file: `{prompt_path.name}`
    61	- Timestamp (UTC): `{timestamp}`
    62	- Status: `{status}`
    63	
    64	## Original Prompt
    65	
    66	```md
    67	{prompt_text.rstrip()}
    68	```
    69	
    70	## Codex Output
    71	
    72	{codex_output.rstrip() or "*No output captured.*"}
    73	
    74	## Notes
    75	
    76	{notes.rstrip()}
    77	"""
    78	
    79	
    80	def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
    81	    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
    82	        output_path = Path(handle.name)
    83	
    84	    try:
    85	        result = subprocess.run(
    86	            ["codex", "exec", "-C", str(root), "--output-last-message", str(output_path), "-"],
    87	            input=prompt_text,
    88	            text=True,
    89	            capture_output=True,
    90	            check=False,
    91	        )
    92	
    93	        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    94	        return result.returncode, final_output, result.stderr
    95	    finally:
    96	        output_path.unlink(missing_ok=True)
    97	
    98	
    99	def main() -> int:
   100	    parser = argparse.ArgumentParser(description="Run one prompt file through codex exec and write a note.")
   101	    parser.add_argument("prompt", help="Prompt filename, numeric prefix, or path")
   102	    args = parser.parse_args()
   103	
   104	    root = repo_root()
   105	    prompts_dir = root / PROMPTS_DIR
   106	    notes_dir = root / NOTES_DIR
   107	
   108	    if not prompts_dir.exists():
   109	        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
   110	        return 1
   111	
   112	    if not notes_dir.exists():
   113	        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
   114	        return 1
   115	
   116	    prompt_path = resolve_prompt_path(root, args.prompt)
   117	    if not prompt_path.exists():
   118	        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
   119	        return 1
   120	
   121	    prompt_text = prompt_path.read_text(encoding="utf-8")
   122	    timestamp = utc_timestamp()
   123	
   124	    returncode, codex_output, stderr_text = run_codex(root, prompt_text)
   125	    status = "SUCCESS" if returncode == 0 else "FAILED"
   126	    note_path = build_note_path(root, prompt_path, status, timestamp)
   127	
   128	    notes = ""
   129	    if stderr_text.strip():
   130	        notes = f"""Codex stderr:
   131	```text
   132	{stderr_text.rstrip()}
   133	```"""
   134	
   135	    note_path.write_text(
   136	        build_note_content(
   137	            prompt_path=prompt_path,
   138	            timestamp=timestamp,
   139	            status=status,
   140	            prompt_text=prompt_text,
   141	            codex_output=codex_output,
   142	            notes=notes,
   143	        ),
   144	        encoding="utf-8",
   145	    )
   146	
   147	    print(note_path)
   148	    return returncode
   149	
   150	
   151	if __name__ == "__main__":
   152	    raise SystemExit(main())

codex
The mismatch pattern is fairly clear now: the architecture note decomposes responsibilities well, but it still reads like an aspiration list more than a workflow-sequenced design; the runner is thin enough to be useful, but it collapses failure analysis and cost evidence into a single success/fail note. I’m creating the two requested artifacts next, with the recommendation narrowed to one design decision rather than a roadmap.
exec
/bin/bash -lc 'date -u +%Y%m%d_%H%M%S' in /data/git/handson-ml3
 succeeded in 0ms:
20260415_195538

exec
/bin/bash -lc 'ls -d notes' in /data/git/handson-ml3
 succeeded in 0ms:
notes

codex
I’m writing the assessment files now. The report will stay anchored to the workflow principles and the existing runner behavior, and the second file will name one bounded design artifact to produce next while explicitly deferring broader refactoring.
apply patch
patch: completed
/data/git/handson-ml3/notes/004_architecture_and_bridge_runner_review__20260415_195538.md
/data/git/handson-ml3/notes/004_next_design_step_recommendation__20260415_195538.md
diff --git a/notes/004_architecture_and_bridge_runner_review__20260415_195538.md b/notes/004_architecture_and_bridge_runner_review__20260415_195538.md
new file mode 100644
index 0000000000000000000000000000000000000000..d68704808225218db458996857cf657779e7b246
--- /dev/null
+++ b/notes/004_architecture_and_bridge_runner_review__20260415_195538.md
@@ -0,0 +1,64 @@
+# Architecture And Bridge Runner Review
+
+## Summary
+
+`tools/Codex_Prompt_Workflow_Architecture.md` is directionally aligned with the governing workflow because it pushes toward smaller parts, clearer ownership, and conservative growth. `tools/codex/baby_run_prompt.py` is acceptable as thin bridge tooling for a single bounded slice, but the current architecture doc does not yet translate the workflow into a clear V1 boundary, validation plan, review loop, or failure/resource evidence model. The next step should stay at the design layer: define the minimum execution-record artifact that every run must produce before more tooling is added.
+
+## Architecture Alignment Findings
+
+### Strong alignment
+
+- The architecture doc is explicitly conservative, small-part oriented, and separation-of-concerns driven, which matches design-before-build and boundaries-before-breadth in the workflow ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:92), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:5)).
+- Its decomposition of the original script into repo paths, prompt parsing, note handling, status reconstruction, retry handling, and CLI concerns is a good responsibility split and a useful architecture starting point ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:177), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:33)).
+- It repeatedly says not to build everything at once and proposes a smaller folder shape before a larger one, which is consistent with thin slices before large pushes ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:100), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:121), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:246)).
+
+### Stale, incomplete, or misaligned
+
+- The architecture doc does not yet reflect the workflow's full sequence. It jumps from decomposition into a wish list and folder layout, but it does not define a short problem statement, explicit scope fence, out-of-scope list, minimum viable slice, artifact inventory, implementation order, or validation checklist as the governing workflow expects ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:141), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:159), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:196), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:214), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:233), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:251)).
+- The "eventually" section is useful as an extension path, but it is carrying too much conceptual breadth for a workflow that emphasizes bounded next steps. There is no explicit line saying which of those items are intentionally deferred from V1 ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:96), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:374), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:119)).
+- The document mentions logging and diagnostics, richer notes, retry intelligence, approval gates, and dependencies, but it does not specify the source of truth or the stable identity of important records beyond prompt stems. That leaves the durable-history model underdesigned relative to the workflow's emphasis on preserved local memory and stable identities ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:120), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:374), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:187)).
+- Review points between iterations are largely absent. The architecture doc talks about manual status marking and approval gates eventually, but not about the concrete review stop that should happen after each bounded run before the next run proceeds ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:262), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:208)).
+
+## Runner Alignment Findings
+
+### Useful and appropriately thin
+
+- The runner does one bounded thing: resolve a prompt, execute `codex exec`, and write a local note. That fits the workflow's allowance for thin bridge tooling that gathers evidence while remaining inspectable and reversible ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:114), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:99)).
+- It validates basic directory prerequisites and fails cleanly when expected folders or prompt files are missing, which is a reasonable minimal safeguard for a bridge script ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:109), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:108)).
+- It writes timestamped markdown notes into `notes/`, which gives the system at least some durable local history of what was run and what Codex returned ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:120), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:46), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:135)).
+
+### Responsibilities it is carrying that should move elsewhere later
+
+- Prompt resolution policy is embedded in the runner. Selection by absolute path, repo-relative path, `codex_prompts/` path, or globbed prefix is helpful now, but prompt identity and lookup rules belong in a prompt-discovery/status layer if the system grows ([tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:47), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:25)).
+- Note-path and note-content policy are also embedded here. That is acceptable for a bridge script, but note schema, status vocabulary, and execution-record structure should eventually be owned by a note/history module rather than the process launcher ([tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:96), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:187), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:50)).
+- The runner currently decides final state as `SUCCESS` or `FAILED` based only on the subprocess exit code. That is an expedient bridge behavior, but review judgment, validation outcome, and richer state modeling should not stay inside the launcher if the workflow matures ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:141), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:124)).
+
+### Current misalignment
+
+- As bridge tooling it is still acceptable, but it is only partially subordinate to workflow. It captures execution, yet it does not encode the review point that should separate execution from acceptance. A zero exit code becomes `SUCCESS` immediately, which skips the workflow's "validate and inspect before the next step" posture ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:271), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:125)).
+- It stores the original prompt text and final output, but not explicit success criteria, validation evidence, or reviewer judgment. That means the note is an execution artifact, not yet a full workflow artifact ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:251), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:58)).
+
+## Failure-Analysis Findings
+
+- The current setup provides some raw failure evidence because nonzero exit runs still write a note and stderr is preserved when present ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:343), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:128)).
+- That said, it does not support useful post-failure analysis very well. The note structure has no fields for failure classification, whether the failure was task/design/environment related, whether the task was too large, what changed on retry, or what the recommended next action is. The workflow explicitly wants failure to produce analysis rather than mere retry cost ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:125), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:343)).
+- The smallest improvement here is not a larger retry engine. It is a defined execution-note schema with a short "failure analysis" block for failed runs, even if the user fills some of it manually. That preserves inspectable local history without expanding the runner's responsibilities much.
+
+## Resource-Awareness Findings
+
+- The setup preserves only minimal cost evidence today: timestamp in the note filename/content and the captured final output/stderr. That gives some weak signal about artifact volume, but almost nothing about runtime, retries, or service usage ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:382), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:17), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:58)).
+- There is no recorded elapsed time, no attempt count or retry linkage, no output-size metric, and no token usage capture if Codex exposes it. The workflow asks for lightweight observation, not a billing system, and the current setup falls short of even that lightweight bar ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:130), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:382)).
+- The smallest improvement is to define a tiny metrics block in the execution record. Start with fields the runner can always capture cheaply: start time, end time, elapsed seconds, subprocess return code, stderr present yes/no, and output character count. Token usage can remain optional when available later.
+
+## Single Recommended Next Design Step
+
+Create a short design artifact that defines the V1 execution record for one prompt run.
+
+That artifact should do four things only:
+
+1. Declare the source of truth for a run record.
+2. Define the minimum fields for execution, review, failure analysis, and lightweight metrics.
+3. Separate execution outcome from reviewed outcome.
+4. State what bridge tooling may capture now versus what remains manual for this stage.
+
+This is the best next step because both the architecture doc and the runner are currently weak at the same seam: they can run prompts and save notes, but they do not yet define what a durable, reviewable run artifact must contain. Defining that record is smaller and safer than refactoring modules now, and it directly supports failure analysis, resource awareness, and review between iterations.
diff --git a/notes/004_next_design_step_recommendation__20260415_195538.md b/notes/004_next_design_step_recommendation__20260415_195538.md
new file mode 100644
index 0000000000000000000000000000000000000000..bd504fee4abbdde468c28efe6842d2ea85685552
--- /dev/null
+++ b/notes/004_next_design_step_recommendation__20260415_195538.md
@@ -0,0 +1,21 @@
+# Next Design Step Recommendation
+
+Define the V1 execution-record artifact for a single prompt run.
+
+This should happen next because the current system can execute prompts and write notes, but it does not yet define the minimum durable record needed for review, failure analysis, and lightweight cost observation. Without that decision, more implementation risks hard-coding the wrong note model and the wrong success semantics.
+
+The artifact should be a short design note that specifies:
+
+- the source of truth for one run record
+- the minimum fields for execution facts, reviewed outcome, failure-analysis notes, and lightweight metrics
+- which fields are captured automatically now
+- which fields remain manual until later
+
+What should wait until later:
+
+- module refactors and file splits
+- richer status/state machines
+- retry intelligence
+- dependency handling
+- approval-gate automation
+- reporting layers beyond the single-run record

diff --git a/notes/004_architecture_and_bridge_runner_review__20260415_195538.md b/notes/004_architecture_and_bridge_runner_review__20260415_195538.md
new file mode 100644
index 0000000000000000000000000000000000000000..d68704808225218db458996857cf657779e7b246
--- /dev/null
+++ b/notes/004_architecture_and_bridge_runner_review__20260415_195538.md
@@ -0,0 +1,64 @@
+# Architecture And Bridge Runner Review
+
+## Summary
+
+`tools/Codex_Prompt_Workflow_Architecture.md` is directionally aligned with the governing workflow because it pushes toward smaller parts, clearer ownership, and conservative growth. `tools/codex/baby_run_prompt.py` is acceptable as thin bridge tooling for a single bounded slice, but the current architecture doc does not yet translate the workflow into a clear V1 boundary, validation plan, review loop, or failure/resource evidence model. The next step should stay at the design layer: define the minimum execution-record artifact that every run must produce before more tooling is added.
+
+## Architecture Alignment Findings
+
+### Strong alignment
+
+- The architecture doc is explicitly conservative, small-part oriented, and separation-of-concerns driven, which matches design-before-build and boundaries-before-breadth in the workflow ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:92), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:5)).
+- Its decomposition of the original script into repo paths, prompt parsing, note handling, status reconstruction, retry handling, and CLI concerns is a good responsibility split and a useful architecture starting point ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:177), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:33)).
+- It repeatedly says not to build everything at once and proposes a smaller folder shape before a larger one, which is consistent with thin slices before large pushes ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:100), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:121), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:246)).
+
+### Stale, incomplete, or misaligned
+
+- The architecture doc does not yet reflect the workflow's full sequence. It jumps from decomposition into a wish list and folder layout, but it does not define a short problem statement, explicit scope fence, out-of-scope list, minimum viable slice, artifact inventory, implementation order, or validation checklist as the governing workflow expects ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:141), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:159), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:196), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:214), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:233), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:251)).
+- The "eventually" section is useful as an extension path, but it is carrying too much conceptual breadth for a workflow that emphasizes bounded next steps. There is no explicit line saying which of those items are intentionally deferred from V1 ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:96), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:374), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:119)).
+- The document mentions logging and diagnostics, richer notes, retry intelligence, approval gates, and dependencies, but it does not specify the source of truth or the stable identity of important records beyond prompt stems. That leaves the durable-history model underdesigned relative to the workflow's emphasis on preserved local memory and stable identities ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:120), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:374), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:187)).
+- Review points between iterations are largely absent. The architecture doc talks about manual status marking and approval gates eventually, but not about the concrete review stop that should happen after each bounded run before the next run proceeds ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:262), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:208)).
+
+## Runner Alignment Findings
+
+### Useful and appropriately thin
+
+- The runner does one bounded thing: resolve a prompt, execute `codex exec`, and write a local note. That fits the workflow's allowance for thin bridge tooling that gathers evidence while remaining inspectable and reversible ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:114), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:99)).
+- It validates basic directory prerequisites and fails cleanly when expected folders or prompt files are missing, which is a reasonable minimal safeguard for a bridge script ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:109), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:108)).
+- It writes timestamped markdown notes into `notes/`, which gives the system at least some durable local history of what was run and what Codex returned ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:120), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:46), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:135)).
+
+### Responsibilities it is carrying that should move elsewhere later
+
+- Prompt resolution policy is embedded in the runner. Selection by absolute path, repo-relative path, `codex_prompts/` path, or globbed prefix is helpful now, but prompt identity and lookup rules belong in a prompt-discovery/status layer if the system grows ([tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:47), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:25)).
+- Note-path and note-content policy are also embedded here. That is acceptable for a bridge script, but note schema, status vocabulary, and execution-record structure should eventually be owned by a note/history module rather than the process launcher ([tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:96), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:187), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:50)).
+- The runner currently decides final state as `SUCCESS` or `FAILED` based only on the subprocess exit code. That is an expedient bridge behavior, but review judgment, validation outcome, and richer state modeling should not stay inside the launcher if the workflow matures ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:141), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:124)).
+
+### Current misalignment
+
+- As bridge tooling it is still acceptable, but it is only partially subordinate to workflow. It captures execution, yet it does not encode the review point that should separate execution from acceptance. A zero exit code becomes `SUCCESS` immediately, which skips the workflow's "validate and inspect before the next step" posture ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:271), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:125)).
+- It stores the original prompt text and final output, but not explicit success criteria, validation evidence, or reviewer judgment. That means the note is an execution artifact, not yet a full workflow artifact ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:251), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:58)).
+
+## Failure-Analysis Findings
+
+- The current setup provides some raw failure evidence because nonzero exit runs still write a note and stderr is preserved when present ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:343), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:128)).
+- That said, it does not support useful post-failure analysis very well. The note structure has no fields for failure classification, whether the failure was task/design/environment related, whether the task was too large, what changed on retry, or what the recommended next action is. The workflow explicitly wants failure to produce analysis rather than mere retry cost ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:125), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:343)).
+- The smallest improvement here is not a larger retry engine. It is a defined execution-note schema with a short "failure analysis" block for failed runs, even if the user fills some of it manually. That preserves inspectable local history without expanding the runner's responsibilities much.
+
+## Resource-Awareness Findings
+
+- The setup preserves only minimal cost evidence today: timestamp in the note filename/content and the captured final output/stderr. That gives some weak signal about artifact volume, but almost nothing about runtime, retries, or service usage ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:382), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:17), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:58)).
+- There is no recorded elapsed time, no attempt count or retry linkage, no output-size metric, and no token usage capture if Codex exposes it. The workflow asks for lightweight observation, not a billing system, and the current setup falls short of even that lightweight bar ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:130), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:382)).
+- The smallest improvement is to define a tiny metrics block in the execution record. Start with fields the runner can always capture cheaply: start time, end time, elapsed seconds, subprocess return code, stderr present yes/no, and output character count. Token usage can remain optional when available later.
+
+## Single Recommended Next Design Step
+
+Create a short design artifact that defines the V1 execution record for one prompt run.
+
+That artifact should do four things only:
+
+1. Declare the source of truth for a run record.
+2. Define the minimum fields for execution, review, failure analysis, and lightweight metrics.
+3. Separate execution outcome from reviewed outcome.
+4. State what bridge tooling may capture now versus what remains manual for this stage.
+
+This is the best next step because both the architecture doc and the runner are currently weak at the same seam: they can run prompts and save notes, but they do not yet define what a durable, reviewable run artifact must contain. Defining that record is smaller and safer than refactoring modules now, and it directly supports failure analysis, resource awareness, and review between iterations.
diff --git a/notes/004_next_design_step_recommendation__20260415_195538.md b/notes/004_next_design_step_recommendation__20260415_195538.md
new file mode 100644
index 0000000000000000000000000000000000000000..bd504fee4abbdde468c28efe6842d2ea85685552
--- /dev/null
+++ b/notes/004_next_design_step_recommendation__20260415_195538.md
@@ -0,0 +1,21 @@
+# Next Design Step Recommendation
+
+Define the V1 execution-record artifact for a single prompt run.
+
+This should happen next because the current system can execute prompts and write notes, but it does not yet define the minimum durable record needed for review, failure analysis, and lightweight cost observation. Without that decision, more implementation risks hard-coding the wrong note model and the wrong success semantics.
+
+The artifact should be a short design note that specifies:
+
+- the source of truth for one run record
+- the minimum fields for execution facts, reviewed outcome, failure-analysis notes, and lightweight metrics
+- which fields are captured automatically now
+- which fields remain manual until later
+
+What should wait until later:
+
+- module refactors and file splits
+- richer status/state machines
+- retry intelligence
+- dependency handling
+- approval-gate automation
+- reporting layers beyond the single-run record

codex
The two notes are written. I’m doing a quick verification pass on the generated files now to make sure the assessment stayed bounded, answered the requested questions, and kept the recommendation to one concrete design artifact.
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
/bin/bash -lc "sed -n '1,220p' notes/004_next_design_step_recommendation__20260415_195538.md" in /data/git/handson-ml3
 succeeded in 0ms:
# Next Design Step Recommendation

Define the V1 execution-record artifact for a single prompt run.

This should happen next because the current system can execute prompts and write notes, but it does not yet define the minimum durable record needed for review, failure analysis, and lightweight cost observation. Without that decision, more implementation risks hard-coding the wrong note model and the wrong success semantics.

The artifact should be a short design note that specifies:

- the source of truth for one run record
- the minimum fields for execution facts, reviewed outcome, failure-analysis notes, and lightweight metrics
- which fields are captured automatically now
- which fields remain manual until later

What should wait until later:

- module refactors and file splits
- richer status/state machines
- retry intelligence
- dependency handling
- approval-gate automation
- reporting layers beyond the single-run record

diff --git a/notes/004_architecture_and_bridge_runner_review__20260415_195538.md b/notes/004_architecture_and_bridge_runner_review__20260415_195538.md
new file mode 100644
index 0000000000000000000000000000000000000000..d68704808225218db458996857cf657779e7b246
--- /dev/null
+++ b/notes/004_architecture_and_bridge_runner_review__20260415_195538.md
@@ -0,0 +1,64 @@
+# Architecture And Bridge Runner Review
+
+## Summary
+
+`tools/Codex_Prompt_Workflow_Architecture.md` is directionally aligned with the governing workflow because it pushes toward smaller parts, clearer ownership, and conservative growth. `tools/codex/baby_run_prompt.py` is acceptable as thin bridge tooling for a single bounded slice, but the current architecture doc does not yet translate the workflow into a clear V1 boundary, validation plan, review loop, or failure/resource evidence model. The next step should stay at the design layer: define the minimum execution-record artifact that every run must produce before more tooling is added.
+
+## Architecture Alignment Findings
+
+### Strong alignment
+
+- The architecture doc is explicitly conservative, small-part oriented, and separation-of-concerns driven, which matches design-before-build and boundaries-before-breadth in the workflow ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:92), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:5)).
+- Its decomposition of the original script into repo paths, prompt parsing, note handling, status reconstruction, retry handling, and CLI concerns is a good responsibility split and a useful architecture starting point ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:177), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:33)).
+- It repeatedly says not to build everything at once and proposes a smaller folder shape before a larger one, which is consistent with thin slices before large pushes ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:100), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:121), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:246)).
+
+### Stale, incomplete, or misaligned
+
+- The architecture doc does not yet reflect the workflow's full sequence. It jumps from decomposition into a wish list and folder layout, but it does not define a short problem statement, explicit scope fence, out-of-scope list, minimum viable slice, artifact inventory, implementation order, or validation checklist as the governing workflow expects ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:141), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:159), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:196), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:214), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:233), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:251)).
+- The "eventually" section is useful as an extension path, but it is carrying too much conceptual breadth for a workflow that emphasizes bounded next steps. There is no explicit line saying which of those items are intentionally deferred from V1 ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:96), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:374), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:119)).
+- The document mentions logging and diagnostics, richer notes, retry intelligence, approval gates, and dependencies, but it does not specify the source of truth or the stable identity of important records beyond prompt stems. That leaves the durable-history model underdesigned relative to the workflow's emphasis on preserved local memory and stable identities ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:120), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:374), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:187)).
+- Review points between iterations are largely absent. The architecture doc talks about manual status marking and approval gates eventually, but not about the concrete review stop that should happen after each bounded run before the next run proceeds ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:262), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:208)).
+
+## Runner Alignment Findings
+
+### Useful and appropriately thin
+
+- The runner does one bounded thing: resolve a prompt, execute `codex exec`, and write a local note. That fits the workflow's allowance for thin bridge tooling that gathers evidence while remaining inspectable and reversible ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:114), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:99)).
+- It validates basic directory prerequisites and fails cleanly when expected folders or prompt files are missing, which is a reasonable minimal safeguard for a bridge script ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:109), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:108)).
+- It writes timestamped markdown notes into `notes/`, which gives the system at least some durable local history of what was run and what Codex returned ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:120), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:46), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:135)).
+
+### Responsibilities it is carrying that should move elsewhere later
+
+- Prompt resolution policy is embedded in the runner. Selection by absolute path, repo-relative path, `codex_prompts/` path, or globbed prefix is helpful now, but prompt identity and lookup rules belong in a prompt-discovery/status layer if the system grows ([tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:47), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:25)).
+- Note-path and note-content policy are also embedded here. That is acceptable for a bridge script, but note schema, status vocabulary, and execution-record structure should eventually be owned by a note/history module rather than the process launcher ([tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:96), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:187), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:50)).
+- The runner currently decides final state as `SUCCESS` or `FAILED` based only on the subprocess exit code. That is an expedient bridge behavior, but review judgment, validation outcome, and richer state modeling should not stay inside the launcher if the workflow matures ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:141), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:124)).
+
+### Current misalignment
+
+- As bridge tooling it is still acceptable, but it is only partially subordinate to workflow. It captures execution, yet it does not encode the review point that should separate execution from acceptance. A zero exit code becomes `SUCCESS` immediately, which skips the workflow's "validate and inspect before the next step" posture ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:271), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:125)).
+- It stores the original prompt text and final output, but not explicit success criteria, validation evidence, or reviewer judgment. That means the note is an execution artifact, not yet a full workflow artifact ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:251), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:58)).
+
+## Failure-Analysis Findings
+
+- The current setup provides some raw failure evidence because nonzero exit runs still write a note and stderr is preserved when present ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:343), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:128)).
+- That said, it does not support useful post-failure analysis very well. The note structure has no fields for failure classification, whether the failure was task/design/environment related, whether the task was too large, what changed on retry, or what the recommended next action is. The workflow explicitly wants failure to produce analysis rather than mere retry cost ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:125), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:343)).
+- The smallest improvement here is not a larger retry engine. It is a defined execution-note schema with a short "failure analysis" block for failed runs, even if the user fills some of it manually. That preserves inspectable local history without expanding the runner's responsibilities much.
+
+## Resource-Awareness Findings
+
+- The setup preserves only minimal cost evidence today: timestamp in the note filename/content and the captured final output/stderr. That gives some weak signal about artifact volume, but almost nothing about runtime, retries, or service usage ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:382), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:17), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:58)).
+- There is no recorded elapsed time, no attempt count or retry linkage, no output-size metric, and no token usage capture if Codex exposes it. The workflow asks for lightweight observation, not a billing system, and the current setup falls short of even that lightweight bar ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:130), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:382)).
+- The smallest improvement is to define a tiny metrics block in the execution record. Start with fields the runner can always capture cheaply: start time, end time, elapsed seconds, subprocess return code, stderr present yes/no, and output character count. Token usage can remain optional when available later.
+
+## Single Recommended Next Design Step
+
+Create a short design artifact that defines the V1 execution record for one prompt run.
+
+That artifact should do four things only:
+
+1. Declare the source of truth for a run record.
+2. Define the minimum fields for execution, review, failure analysis, and lightweight metrics.
+3. Separate execution outcome from reviewed outcome.
+4. State what bridge tooling may capture now versus what remains manual for this stage.
+
+This is the best next step because both the architecture doc and the runner are currently weak at the same seam: they can run prompts and save notes, but they do not yet define what a durable, reviewable run artifact must contain. Defining that record is smaller and safer than refactoring modules now, and it directly supports failure analysis, resource awareness, and review between iterations.
diff --git a/notes/004_next_design_step_recommendation__20260415_195538.md b/notes/004_next_design_step_recommendation__20260415_195538.md
new file mode 100644
index 0000000000000000000000000000000000000000..bd504fee4abbdde468c28efe6842d2ea85685552
--- /dev/null
+++ b/notes/004_next_design_step_recommendation__20260415_195538.md
@@ -0,0 +1,21 @@
+# Next Design Step Recommendation
+
+Define the V1 execution-record artifact for a single prompt run.
+
+This should happen next because the current system can execute prompts and write notes, but it does not yet define the minimum durable record needed for review, failure analysis, and lightweight cost observation. Without that decision, more implementation risks hard-coding the wrong note model and the wrong success semantics.
+
+The artifact should be a short design note that specifies:
+
+- the source of truth for one run record
+- the minimum fields for execution facts, reviewed outcome, failure-analysis notes, and lightweight metrics
+- which fields are captured automatically now
+- which fields remain manual until later
+
+What should wait until later:
+
+- module refactors and file splits
+- richer status/state machines
+- retry intelligence
+- dependency handling
+- approval-gate automation
+- reporting layers beyond the single-run record

Assessment completed and both requested notes were created with a single bounded next-step recommendation.
  ✓ Read governing workflow, architecture doc, and bridge runner
  ✓ Assess alignment and gaps against workflow principles
  ✓ Write the two required markdown artifacts in notes/
diff --git a/notes/004_architecture_and_bridge_runner_review__20260415_195538.md b/notes/004_architecture_and_bridge_runner_review__20260415_195538.md
new file mode 100644
index 0000000000000000000000000000000000000000..d68704808225218db458996857cf657779e7b246
--- /dev/null
+++ b/notes/004_architecture_and_bridge_runner_review__20260415_195538.md
@@ -0,0 +1,64 @@
+# Architecture And Bridge Runner Review
+
+## Summary
+
+`tools/Codex_Prompt_Workflow_Architecture.md` is directionally aligned with the governing workflow because it pushes toward smaller parts, clearer ownership, and conservative growth. `tools/codex/baby_run_prompt.py` is acceptable as thin bridge tooling for a single bounded slice, but the current architecture doc does not yet translate the workflow into a clear V1 boundary, validation plan, review loop, or failure/resource evidence model. The next step should stay at the design layer: define the minimum execution-record artifact that every run must produce before more tooling is added.
+
+## Architecture Alignment Findings
+
+### Strong alignment
+
+- The architecture doc is explicitly conservative, small-part oriented, and separation-of-concerns driven, which matches design-before-build and boundaries-before-breadth in the workflow ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:92), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:5)).
+- Its decomposition of the original script into repo paths, prompt parsing, note handling, status reconstruction, retry handling, and CLI concerns is a good responsibility split and a useful architecture starting point ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:177), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:33)).
+- It repeatedly says not to build everything at once and proposes a smaller folder shape before a larger one, which is consistent with thin slices before large pushes ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:100), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:121), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:246)).
+
+### Stale, incomplete, or misaligned
+
+- The architecture doc does not yet reflect the workflow's full sequence. It jumps from decomposition into a wish list and folder layout, but it does not define a short problem statement, explicit scope fence, out-of-scope list, minimum viable slice, artifact inventory, implementation order, or validation checklist as the governing workflow expects ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:141), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:159), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:196), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:214), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:233), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:251)).
+- The "eventually" section is useful as an extension path, but it is carrying too much conceptual breadth for a workflow that emphasizes bounded next steps. There is no explicit line saying which of those items are intentionally deferred from V1 ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:96), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:374), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:119)).
+- The document mentions logging and diagnostics, richer notes, retry intelligence, approval gates, and dependencies, but it does not specify the source of truth or the stable identity of important records beyond prompt stems. That leaves the durable-history model underdesigned relative to the workflow's emphasis on preserved local memory and stable identities ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:120), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:374), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:187)).
+- Review points between iterations are largely absent. The architecture doc talks about manual status marking and approval gates eventually, but not about the concrete review stop that should happen after each bounded run before the next run proceeds ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:262), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:208)).
+
+## Runner Alignment Findings
+
+### Useful and appropriately thin
+
+- The runner does one bounded thing: resolve a prompt, execute `codex exec`, and write a local note. That fits the workflow's allowance for thin bridge tooling that gathers evidence while remaining inspectable and reversible ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:114), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:99)).
+- It validates basic directory prerequisites and fails cleanly when expected folders or prompt files are missing, which is a reasonable minimal safeguard for a bridge script ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:109), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:108)).
+- It writes timestamped markdown notes into `notes/`, which gives the system at least some durable local history of what was run and what Codex returned ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:120), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:46), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:135)).
+
+### Responsibilities it is carrying that should move elsewhere later
+
+- Prompt resolution policy is embedded in the runner. Selection by absolute path, repo-relative path, `codex_prompts/` path, or globbed prefix is helpful now, but prompt identity and lookup rules belong in a prompt-discovery/status layer if the system grows ([tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:47), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:25)).
+- Note-path and note-content policy are also embedded here. That is acceptable for a bridge script, but note schema, status vocabulary, and execution-record structure should eventually be owned by a note/history module rather than the process launcher ([tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:96), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:187), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:50)).
+- The runner currently decides final state as `SUCCESS` or `FAILED` based only on the subprocess exit code. That is an expedient bridge behavior, but review judgment, validation outcome, and richer state modeling should not stay inside the launcher if the workflow matures ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:141), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:124)).
+
+### Current misalignment
+
+- As bridge tooling it is still acceptable, but it is only partially subordinate to workflow. It captures execution, yet it does not encode the review point that should separate execution from acceptance. A zero exit code becomes `SUCCESS` immediately, which skips the workflow's "validate and inspect before the next step" posture ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:271), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:125)).
+- It stores the original prompt text and final output, but not explicit success criteria, validation evidence, or reviewer judgment. That means the note is an execution artifact, not yet a full workflow artifact ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:251), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:58)).
+
+## Failure-Analysis Findings
+
+- The current setup provides some raw failure evidence because nonzero exit runs still write a note and stderr is preserved when present ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:343), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:128)).
+- That said, it does not support useful post-failure analysis very well. The note structure has no fields for failure classification, whether the failure was task/design/environment related, whether the task was too large, what changed on retry, or what the recommended next action is. The workflow explicitly wants failure to produce analysis rather than mere retry cost ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:125), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:343)).
+- The smallest improvement here is not a larger retry engine. It is a defined execution-note schema with a short "failure analysis" block for failed runs, even if the user fills some of it manually. That preserves inspectable local history without expanding the runner's responsibilities much.
+
+## Resource-Awareness Findings
+
+- The setup preserves only minimal cost evidence today: timestamp in the note filename/content and the captured final output/stderr. That gives some weak signal about artifact volume, but almost nothing about runtime, retries, or service usage ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:382), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:17), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:58)).
+- There is no recorded elapsed time, no attempt count or retry linkage, no output-size metric, and no token usage capture if Codex exposes it. The workflow asks for lightweight observation, not a billing system, and the current setup falls short of even that lightweight bar ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:130), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:382)).
+- The smallest improvement is to define a tiny metrics block in the execution record. Start with fields the runner can always capture cheaply: start time, end time, elapsed seconds, subprocess return code, stderr present yes/no, and output character count. Token usage can remain optional when available later.
+
+## Single Recommended Next Design Step
+
+Create a short design artifact that defines the V1 execution record for one prompt run.
+
+That artifact should do four things only:
+
+1. Declare the source of truth for a run record.
+2. Define the minimum fields for execution, review, failure analysis, and lightweight metrics.
+3. Separate execution outcome from reviewed outcome.
+4. State what bridge tooling may capture now versus what remains manual for this stage.
+
+This is the best next step because both the architecture doc and the runner are currently weak at the same seam: they can run prompts and save notes, but they do not yet define what a durable, reviewable run artifact must contain. Defining that record is smaller and safer than refactoring modules now, and it directly supports failure analysis, resource awareness, and review between iterations.
diff --git a/notes/004_next_design_step_recommendation__20260415_195538.md b/notes/004_next_design_step_recommendation__20260415_195538.md
new file mode 100644
index 0000000000000000000000000000000000000000..bd504fee4abbdde468c28efe6842d2ea85685552
--- /dev/null
+++ b/notes/004_next_design_step_recommendation__20260415_195538.md
@@ -0,0 +1,21 @@
+# Next Design Step Recommendation
+
+Define the V1 execution-record artifact for a single prompt run.
+
+This should happen next because the current system can execute prompts and write notes, but it does not yet define the minimum durable record needed for review, failure analysis, and lightweight cost observation. Without that decision, more implementation risks hard-coding the wrong note model and the wrong success semantics.
+
+The artifact should be a short design note that specifies:
+
+- the source of truth for one run record
+- the minimum fields for execution facts, reviewed outcome, failure-analysis notes, and lightweight metrics
+- which fields are captured automatically now
+- which fields remain manual until later
+
+What should wait until later:
+
+- module refactors and file splits
+- richer status/state machines
+- retry intelligence
+- dependency handling
+- approval-gate automation
+- reporting layers beyond the single-run record

codex
Created the two requested artifacts:

- [004_architecture_and_bridge_runner_review__20260415_195538.md](/data/git/handson-ml3/notes/004_architecture_and_bridge_runner_review__20260415_195538.md)
- [004_next_design_step_recommendation__20260415_195538.md](/data/git/handson-ml3/notes/004_next_design_step_recommendation__20260415_195538.md)

The assessment’s main conclusion is that the architecture doc is directionally right on decomposition and conservative growth, and the runner is thin enough to count as acceptable bridge tooling, but both are weak at the same seam: they do not yet define a durable, reviewable execution record that separates raw execution from reviewed outcome and captures minimal failure/resource evidence.

I did not modify the workflow doc or refactor the runner. The recommendation is intentionally bounded: define the V1 execution-record artifact next, before more implementation.
diff --git a/notes/004_architecture_and_bridge_runner_review__20260415_195538.md b/notes/004_architecture_and_bridge_runner_review__20260415_195538.md
new file mode 100644
index 0000000000000000000000000000000000000000..d68704808225218db458996857cf657779e7b246
--- /dev/null
+++ b/notes/004_architecture_and_bridge_runner_review__20260415_195538.md
@@ -0,0 +1,64 @@
+# Architecture And Bridge Runner Review
+
+## Summary
+
+`tools/Codex_Prompt_Workflow_Architecture.md` is directionally aligned with the governing workflow because it pushes toward smaller parts, clearer ownership, and conservative growth. `tools/codex/baby_run_prompt.py` is acceptable as thin bridge tooling for a single bounded slice, but the current architecture doc does not yet translate the workflow into a clear V1 boundary, validation plan, review loop, or failure/resource evidence model. The next step should stay at the design layer: define the minimum execution-record artifact that every run must produce before more tooling is added.
+
+## Architecture Alignment Findings
+
+### Strong alignment
+
+- The architecture doc is explicitly conservative, small-part oriented, and separation-of-concerns driven, which matches design-before-build and boundaries-before-breadth in the workflow ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:92), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:5)).
+- Its decomposition of the original script into repo paths, prompt parsing, note handling, status reconstruction, retry handling, and CLI concerns is a good responsibility split and a useful architecture starting point ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:177), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:33)).
+- It repeatedly says not to build everything at once and proposes a smaller folder shape before a larger one, which is consistent with thin slices before large pushes ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:100), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:121), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:246)).
+
+### Stale, incomplete, or misaligned
+
+- The architecture doc does not yet reflect the workflow's full sequence. It jumps from decomposition into a wish list and folder layout, but it does not define a short problem statement, explicit scope fence, out-of-scope list, minimum viable slice, artifact inventory, implementation order, or validation checklist as the governing workflow expects ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:141), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:159), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:196), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:214), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:233), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:251)).
+- The "eventually" section is useful as an extension path, but it is carrying too much conceptual breadth for a workflow that emphasizes bounded next steps. There is no explicit line saying which of those items are intentionally deferred from V1 ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:96), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:374), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:119)).
+- The document mentions logging and diagnostics, richer notes, retry intelligence, approval gates, and dependencies, but it does not specify the source of truth or the stable identity of important records beyond prompt stems. That leaves the durable-history model underdesigned relative to the workflow's emphasis on preserved local memory and stable identities ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:120), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:374), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:187)).
+- Review points between iterations are largely absent. The architecture doc talks about manual status marking and approval gates eventually, but not about the concrete review stop that should happen after each bounded run before the next run proceeds ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:262), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:208)).
+
+## Runner Alignment Findings
+
+### Useful and appropriately thin
+
+- The runner does one bounded thing: resolve a prompt, execute `codex exec`, and write a local note. That fits the workflow's allowance for thin bridge tooling that gathers evidence while remaining inspectable and reversible ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:114), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:99)).
+- It validates basic directory prerequisites and fails cleanly when expected folders or prompt files are missing, which is a reasonable minimal safeguard for a bridge script ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:109), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:108)).
+- It writes timestamped markdown notes into `notes/`, which gives the system at least some durable local history of what was run and what Codex returned ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:120), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:46), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:135)).
+
+### Responsibilities it is carrying that should move elsewhere later
+
+- Prompt resolution policy is embedded in the runner. Selection by absolute path, repo-relative path, `codex_prompts/` path, or globbed prefix is helpful now, but prompt identity and lookup rules belong in a prompt-discovery/status layer if the system grows ([tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:47), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:25)).
+- Note-path and note-content policy are also embedded here. That is acceptable for a bridge script, but note schema, status vocabulary, and execution-record structure should eventually be owned by a note/history module rather than the process launcher ([tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:96), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:187), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:50)).
+- The runner currently decides final state as `SUCCESS` or `FAILED` based only on the subprocess exit code. That is an expedient bridge behavior, but review judgment, validation outcome, and richer state modeling should not stay inside the launcher if the workflow matures ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Codex_Prompt_Workflow_Architecture.md](/data/git/handson-ml3/tools/Codex_Prompt_Workflow_Architecture.md:141), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:124)).
+
+### Current misalignment
+
+- As bridge tooling it is still acceptable, but it is only partially subordinate to workflow. It captures execution, yet it does not encode the review point that should separate execution from acceptance. A zero exit code becomes `SUCCESS` immediately, which skips the workflow's "validate and inspect before the next step" posture ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:104), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:271), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:125)).
+- It stores the original prompt text and final output, but not explicit success criteria, validation evidence, or reviewer judgment. That means the note is an execution artifact, not yet a full workflow artifact ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:251), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:58)).
+
+## Failure-Analysis Findings
+
+- The current setup provides some raw failure evidence because nonzero exit runs still write a note and stderr is preserved when present ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:343), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:128)).
+- That said, it does not support useful post-failure analysis very well. The note structure has no fields for failure classification, whether the failure was task/design/environment related, whether the task was too large, what changed on retry, or what the recommended next action is. The workflow explicitly wants failure to produce analysis rather than mere retry cost ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:125), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:343)).
+- The smallest improvement here is not a larger retry engine. It is a defined execution-note schema with a short "failure analysis" block for failed runs, even if the user fills some of it manually. That preserves inspectable local history without expanding the runner's responsibilities much.
+
+## Resource-Awareness Findings
+
+- The setup preserves only minimal cost evidence today: timestamp in the note filename/content and the captured final output/stderr. That gives some weak signal about artifact volume, but almost nothing about runtime, retries, or service usage ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:382), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:17), [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py:58)).
+- There is no recorded elapsed time, no attempt count or retry linkage, no output-size metric, and no token usage capture if Codex exposes it. The workflow asks for lightweight observation, not a billing system, and the current setup falls short of even that lightweight bar ([tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:130), [tools/Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md:382)).
+- The smallest improvement is to define a tiny metrics block in the execution record. Start with fields the runner can always capture cheaply: start time, end time, elapsed seconds, subprocess return code, stderr present yes/no, and output character count. Token usage can remain optional when available later.
+
+## Single Recommended Next Design Step
+
+Create a short design artifact that defines the V1 execution record for one prompt run.
+
+That artifact should do four things only:
+
+1. Declare the source of truth for a run record.
+2. Define the minimum fields for execution, review, failure analysis, and lightweight metrics.
+3. Separate execution outcome from reviewed outcome.
+4. State what bridge tooling may capture now versus what remains manual for this stage.
+
+This is the best next step because both the architecture doc and the runner are currently weak at the same seam: they can run prompts and save notes, but they do not yet define what a durable, reviewable run artifact must contain. Defining that record is smaller and safer than refactoring modules now, and it directly supports failure analysis, resource awareness, and review between iterations.
diff --git a/notes/004_next_design_step_recommendation__20260415_195538.md b/notes/004_next_design_step_recommendation__20260415_195538.md
new file mode 100644
index 0000000000000000000000000000000000000000..bd504fee4abbdde468c28efe6842d2ea85685552
--- /dev/null
+++ b/notes/004_next_design_step_recommendation__20260415_195538.md
@@ -0,0 +1,21 @@
+# Next Design Step Recommendation
+
+Define the V1 execution-record artifact for a single prompt run.
+
+This should happen next because the current system can execute prompts and write notes, but it does not yet define the minimum durable record needed for review, failure analysis, and lightweight cost observation. Without that decision, more implementation risks hard-coding the wrong note model and the wrong success semantics.
+
+The artifact should be a short design note that specifies:
+
+- the source of truth for one run record
+- the minimum fields for execution facts, reviewed outcome, failure-analysis notes, and lightweight metrics
+- which fields are captured automatically now
+- which fields remain manual until later
+
+What should wait until later:
+
+- module refactors and file splits
+- richer status/state machines
+- retry intelligence
+- dependency handling
+- approval-gate automation
+- reporting layers beyond the single-run record

tokens used
42,737
```
