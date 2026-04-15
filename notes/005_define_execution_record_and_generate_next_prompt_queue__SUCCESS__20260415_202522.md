# 005_define_execution_record_and_generate_next_prompt_queue - SUCCESS

- Prompt file: `005_define_execution_record_and_generate_next_prompt_queue.md`
- Timestamp (UTC): `20260415_202522`
- Status: `SUCCESS`

## Original Prompt

```md
# Task: Define the V1 execution-record artifact and generate the next bounded prompt queue

You are working in this repository.

Your task has two tightly related parts:

1. define the V1 execution-record artifact for one prompt run
2. use that design decision to generate the next bounded sequence of prompt files needed to reduce the current workflow/architecture/runner misalignment

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/baby_run_prompt.py`
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
- `notes/004_next_design_step_recommendation__20260415_195538.md`

## Goal

Use the current workflow as the governing law.

Then:

### Part 1
Define the V1 execution-record artifact for one prompt run.

This artifact should clarify:

- what the source of truth is for one run
- what is captured automatically
- what remains manual for human review
- how execution outcome is separated from accepted/reviewed outcome
- what minimal failure-analysis fields exist
- what minimal cost/resource fields exist
- what should explicitly wait until later

### Part 2
Using that execution-record decision, generate the next small sequence of prompt files needed to reduce the remaining misalignment between:

- the workflow we now want
- the architecture we say we want
- the bridge runner we currently have

## Important framing

This is still a design task, not a broad implementation task.

Do NOT refactor the runner in this pass.
Do NOT implement multiple modules in this pass.
Do NOT produce a giant roadmap.

Instead, define the execution-record artifact and then produce the smallest useful queue of next prompts.

## Requirements for Part 1

Create one markdown design artifact at:

`tools/codex/V1_Execution_Record_Artifact.md`

This document should include:

- purpose
- scope
- source of truth
- stable identity for a run
- required fields
- optional fields
- automatic vs manual fields
- execution status vs review status
- minimum failure-analysis section
- minimum resource/cost section
- what is intentionally deferred from V1

Keep it practical, small, and reusable.

## Requirements for Part 2

Create a short sequence of prompt files in `codex_prompts/`.

Create between **3 and 5** prompt files total.
Do not create more than 5.

Each prompt must:

- have one primary goal
- produce one primary artifact or decision
- be small enough to review before the next prompt is run
- follow from the execution-record design
- reduce a specific misalignment already identified in the review notes

Each prompt file should have:

- a clear filename with numeric prefix
- a focused task
- exact file paths where relevant
- explicit constraints
- explicit success criteria

## Also create one companion sequence note

Create a short note at:

`notes/005_prompt_queue_plan__TIMESTAMP.md`

This note should summarize:

- why these prompts were chosen
- why this order reduces risk
- what each prompt is meant to settle
- what larger work is intentionally deferred

## Constraints

1. Use the exact file paths listed above.
2. Treat `tools/Project_Design_Workflow.md` as governing.
3. Keep the execution-record artifact conservative and V1-sized.
4. Do not let the prompt queue become a giant backlog.
5. Prefer the smallest sequence that meaningfully reduces misalignment.
6. Do not implement the later prompts in this pass. Only write them.
7. Do not rewrite the workflow doc in this pass.

## Success criteria

This task is successful if:

- `tools/codex/V1_Execution_Record_Artifact.md` is clear and usable
- the execution-record design separates execution from review/acceptance
- the prompt queue contains only 3 to 5 bounded next prompts
- the prompts are ordered sensibly
- the outputs help us continue one reviewed step at a time
```

## Codex Output

Created the V1 execution-record design at [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:1). It makes one `notes/` markdown record the V1 source of truth for a run, gives each run a stable `run_id`, separates `execution_status` from `review_status`, defines the minimum manual failure-analysis and lightweight resource fields, and explicitly defers broader state machines, reporting, and refactors.

I also added the next bounded prompt queue:
[006_define_v1_run_review_gate.md](/data/git/handson-ml3/codex_prompts/006_define_v1_run_review_gate.md:1),
[007_align_architecture_doc_to_v1_workflow.md](/data/git/handson-ml3/codex_prompts/007_align_architecture_doc_to_v1_workflow.md:1),
and [008_define_minimal_bridge_runner_change_spec.md](/data/git/handson-ml3/codex_prompts/008_define_minimal_bridge_runner_change_spec.md:1).
The companion sequencing note is [005_prompt_queue_plan__20260415_202557.md](/data/git/handson-ml3/notes/005_prompt_queue_plan__20260415_202557.md:1).

Verification was by direct file inspection; no implementation or runner refactor was performed.

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
session id: 019d92d1-ea45-71c3-8fbd-c83284522cdc
--------
user
# Task: Define the V1 execution-record artifact and generate the next bounded prompt queue

You are working in this repository.

Your task has two tightly related parts:

1. define the V1 execution-record artifact for one prompt run
2. use that design decision to generate the next bounded sequence of prompt files needed to reduce the current workflow/architecture/runner misalignment

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/baby_run_prompt.py`
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
- `notes/004_next_design_step_recommendation__20260415_195538.md`

## Goal

Use the current workflow as the governing law.

Then:

### Part 1
Define the V1 execution-record artifact for one prompt run.

This artifact should clarify:

- what the source of truth is for one run
- what is captured automatically
- what remains manual for human review
- how execution outcome is separated from accepted/reviewed outcome
- what minimal failure-analysis fields exist
- what minimal cost/resource fields exist
- what should explicitly wait until later

### Part 2
Using that execution-record decision, generate the next small sequence of prompt files needed to reduce the remaining misalignment between:

- the workflow we now want
- the architecture we say we want
- the bridge runner we currently have

## Important framing

This is still a design task, not a broad implementation task.

Do NOT refactor the runner in this pass.
Do NOT implement multiple modules in this pass.
Do NOT produce a giant roadmap.

Instead, define the execution-record artifact and then produce the smallest useful queue of next prompts.

## Requirements for Part 1

Create one markdown design artifact at:

`tools/codex/V1_Execution_Record_Artifact.md`

This document should include:

- purpose
- scope
- source of truth
- stable identity for a run
- required fields
- optional fields
- automatic vs manual fields
- execution status vs review status
- minimum failure-analysis section
- minimum resource/cost section
- what is intentionally deferred from V1

Keep it practical, small, and reusable.

## Requirements for Part 2

Create a short sequence of prompt files in `codex_prompts/`.

Create between **3 and 5** prompt files total.
Do not create more than 5.

Each prompt must:

- have one primary goal
- produce one primary artifact or decision
- be small enough to review before the next prompt is run
- follow from the execution-record design
- reduce a specific misalignment already identified in the review notes

Each prompt file should have:

- a clear filename with numeric prefix
- a focused task
- exact file paths where relevant
- explicit constraints
- explicit success criteria

## Also create one companion sequence note

Create a short note at:

`notes/005_prompt_queue_plan__TIMESTAMP.md`

This note should summarize:

- why these prompts were chosen
- why this order reduces risk
- what each prompt is meant to settle
- what larger work is intentionally deferred

## Constraints

1. Use the exact file paths listed above.
2. Treat `tools/Project_Design_Workflow.md` as governing.
3. Keep the execution-record artifact conservative and V1-sized.
4. Do not let the prompt queue become a giant backlog.
5. Prefer the smallest sequence that meaningfully reduces misalignment.
6. Do not implement the later prompts in this pass. Only write them.
7. Do not rewrite the workflow doc in this pass.

## Success criteria

This task is successful if:

- `tools/codex/V1_Execution_Record_Artifact.md` is clear and usable
- the execution-record design separates execution from review/acceptance
- the prompt queue contains only 3 to 5 bounded next prompts
- the prompts are ordered sensibly
- the outputs help us continue one reviewed step at a time

codex
I’m inspecting the workflow, architecture, runner, and the two review notes first so the execution-record design stays anchored to the current governing workflow rather than drifting into implementation.
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
/bin/bash -lc "sed -n '1,260p' notes/004_next_design_step_recommendation__20260415_195538.md" in /data/git/handson-ml3
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

codex
I have the governing constraints and the main seam: the current bridge runner records execution, but it conflates execution success with reviewed acceptance and leaves the run artifact underdefined. I’m checking the existing prompt queue so the next sequence stays small and fits the repository’s current naming pattern.
exec
/bin/bash -lc 'rg --files codex_prompts' in /data/git/handson-ml3
 succeeded in 0ms:
codex_prompts/003_revise_Project_Deisgn_workflow_document.md
codex_prompts/002_repo_inventory_and_status.md
codex_prompts/001_smoke_test_pipeline.md
codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md
codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md

exec
/bin/bash -lc 'ls -1 tools/codex' in /data/git/handson-ml3
 succeeded in 0ms:
__pycache__
baby_run_prompt.py

exec
/bin/bash -lc 'ls -1 notes | tail -n 20' in /data/git/handson-ml3
 succeeded in 0ms:
001_smoke_test_pipeline__SUCCESS__20260415_133235.md
001_smoke_test_pipeline__SUCCESS__20260415_183223.md
002_repo_inventory_and_status__SUCCESS__20260415_133347.md
002_repo_inventory_and_status__SUCCESS__20260415_183259.md
003_project_design_workflow_revision__SUCCESS__20260415_144244.md
003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md
004_architecture_and_bridge_runner_review__20260415_195538.md
004_next_design_step_recommendation__20260415_195538.md
004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md

codex
I found the prompt naming pattern and the current queue head. Next I’m reading the latest prompt files so the new bounded prompts match the existing task format and numbering style instead of inventing a parallel convention.
exec
/bin/bash -lc "sed -n '1,240p' codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md" in /data/git/handson-ml3
 succeeded in 0ms:
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

exec
/bin/bash -lc "sed -n '1,260p' codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md" in /data/git/handson-ml3
 succeeded in 0ms:
# Task: Define the V1 execution-record artifact and generate the next bounded prompt queue

You are working in this repository.

Your task has two tightly related parts:

1. define the V1 execution-record artifact for one prompt run
2. use that design decision to generate the next bounded sequence of prompt files needed to reduce the current workflow/architecture/runner misalignment

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/baby_run_prompt.py`
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
- `notes/004_next_design_step_recommendation__20260415_195538.md`

## Goal

Use the current workflow as the governing law.

Then:

### Part 1
Define the V1 execution-record artifact for one prompt run.

This artifact should clarify:

- what the source of truth is for one run
- what is captured automatically
- what remains manual for human review
- how execution outcome is separated from accepted/reviewed outcome
- what minimal failure-analysis fields exist
- what minimal cost/resource fields exist
- what should explicitly wait until later

### Part 2
Using that execution-record decision, generate the next small sequence of prompt files needed to reduce the remaining misalignment between:

- the workflow we now want
- the architecture we say we want
- the bridge runner we currently have

## Important framing

This is still a design task, not a broad implementation task.

Do NOT refactor the runner in this pass.
Do NOT implement multiple modules in this pass.
Do NOT produce a giant roadmap.

Instead, define the execution-record artifact and then produce the smallest useful queue of next prompts.

## Requirements for Part 1

Create one markdown design artifact at:

`tools/codex/V1_Execution_Record_Artifact.md`

This document should include:

- purpose
- scope
- source of truth
- stable identity for a run
- required fields
- optional fields
- automatic vs manual fields
- execution status vs review status
- minimum failure-analysis section
- minimum resource/cost section
- what is intentionally deferred from V1

Keep it practical, small, and reusable.

## Requirements for Part 2

Create a short sequence of prompt files in `codex_prompts/`.

Create between **3 and 5** prompt files total.
Do not create more than 5.

Each prompt must:

- have one primary goal
- produce one primary artifact or decision
- be small enough to review before the next prompt is run
- follow from the execution-record design
- reduce a specific misalignment already identified in the review notes

Each prompt file should have:

- a clear filename with numeric prefix
- a focused task
- exact file paths where relevant
- explicit constraints
- explicit success criteria

## Also create one companion sequence note

Create a short note at:

`notes/005_prompt_queue_plan__TIMESTAMP.md`

This note should summarize:

- why these prompts were chosen
- why this order reduces risk
- what each prompt is meant to settle
- what larger work is intentionally deferred

## Constraints

1. Use the exact file paths listed above.
2. Treat `tools/Project_Design_Workflow.md` as governing.
3. Keep the execution-record artifact conservative and V1-sized.
4. Do not let the prompt queue become a giant backlog.
5. Prefer the smallest sequence that meaningfully reduces misalignment.
6. Do not implement the later prompts in this pass. Only write them.
7. Do not rewrite the workflow doc in this pass.

## Success criteria

This task is successful if:

- `tools/codex/V1_Execution_Record_Artifact.md` is clear and usable
- the execution-record design separates execution from review/acceptance
- the prompt queue contains only 3 to 5 bounded next prompts
- the prompts are ordered sensibly
- the outputs help us continue one reviewed step at a time

codex
I have enough context to write this conservatively. The edits will add one V1 execution-record design note, a short 3-step next prompt queue, and one companion planning note that explains why that order is the smallest useful sequence.
exec
/bin/bash -lc 'date -u +%Y%m%d_%H%M%S' in /data/git/handson-ml3
 succeeded in 0ms:
20260415_202557

apply patch
patch: completed
/data/git/handson-ml3/codex_prompts/006_define_v1_run_review_gate.md
/data/git/handson-ml3/codex_prompts/007_align_architecture_doc_to_v1_workflow.md
/data/git/handson-ml3/codex_prompts/008_define_minimal_bridge_runner_change_spec.md
/data/git/handson-ml3/notes/005_prompt_queue_plan__20260415_202557.md
/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md
diff --git a/codex_prompts/006_define_v1_run_review_gate.md b/codex_prompts/006_define_v1_run_review_gate.md
new file mode 100644
index 0000000000000000000000000000000000000000..3b21bed977ca97d9be3dd7b51cb78b8e74602781
--- /dev/null
+++ b/codex_prompts/006_define_v1_run_review_gate.md
@@ -0,0 +1,82 @@
+# Task: Define the V1 run-review gate for prompt execution
+
+You are working in this repository.
+
+Your task is to define the minimum review gate that must sit between:
+
+- a prompt run finishing
+- the next prompt in sequence being treated as ready
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/baby_run_prompt.py`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__TIMESTAMP.md`
+
+## Goal
+
+Use `tools/Project_Design_Workflow.md` as governing.
+
+Define the smallest practical V1 review gate that preserves:
+
+- review between iterations
+- separation of execution outcome from accepted outcome
+- a conservative manual step before the next prompt is treated as ready
+
+## Important framing
+
+This is a design task.
+
+Do NOT implement the gate in code in this pass.
+Do NOT create a large state machine.
+Do NOT redesign the whole queue system.
+
+## Questions to settle
+
+Please settle these points:
+
+1. What exact event marks a run as awaiting review?
+2. What minimum human checks must happen before a run can be accepted?
+3. What review outcomes are needed in V1?
+4. What outcome should let the next prompt proceed?
+5. What outcome should stop the queue and force a new design or retry decision?
+6. What information must be written back into the run record during review?
+
+## Required output artifact
+
+Create one markdown design note at:
+
+`tools/codex/V1_Run_Review_Gate.md`
+
+The note should include:
+
+- purpose
+- scope
+- the review trigger
+- the minimum manual checklist
+- allowed V1 review outcomes
+- how the outcome affects queue progression
+- how this connects to `tools/codex/V1_Execution_Record_Artifact.md`
+- what is intentionally deferred
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Keep the gate manual and conservative.
+3. Do not implement CLI or runner changes in this pass.
+4. Do not invent a large workflow beyond the next reviewed step.
+5. Keep the decision small enough to review before any implementation prompt follows.
+
+## Success criteria
+
+This task is successful if:
+
+- the review gate is explicit and easy to apply
+- execution success is clearly separated from accepted outcome
+- queue progression rules are clear for V1
+- the result is small enough to guide the next prompt without expanding scope
diff --git a/codex_prompts/007_align_architecture_doc_to_v1_workflow.md b/codex_prompts/007_align_architecture_doc_to_v1_workflow.md
new file mode 100644
index 0000000000000000000000000000000000000000..840d018ed4e23e74f8704c350a60f42fef84b960
--- /dev/null
+++ b/codex_prompts/007_align_architecture_doc_to_v1_workflow.md
@@ -0,0 +1,76 @@
+# Task: Align the Codex prompt architecture document to the governing workflow and V1 run artifact
+
+You are working in this repository.
+
+Your task is to bring the architecture document into explicit V1 alignment with:
+
+- the governing workflow
+- the V1 execution record
+- the V1 review gate
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Run_Review_Gate.md`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__TIMESTAMP.md`
+
+## Goal
+
+Update the architecture document so it no longer skips the workflow steps that matter for V1.
+
+The revised architecture should make clear:
+
+- the V1 problem being solved
+- the V1 scope boundary
+- the minimum artifact inventory
+- the role of the execution record
+- the role of the review gate
+- the conservative implementation order
+- what remains deferred
+
+## Important framing
+
+This is still a design-alignment task.
+
+Do NOT implement the architecture in code in this pass.
+Do NOT expand the document into a giant platform roadmap.
+Do NOT rewrite the governing workflow document.
+
+## Required output artifact
+
+Revise this file directly:
+
+`tools/Codex_Prompt_Workflow_Architecture.md`
+
+The revision should add or clarify:
+
+- short problem statement
+- V1 scope
+- out-of-scope items
+- minimum artifact inventory
+- minimum viable slice
+- implementation order
+- validation and review posture
+- explicit deferred items
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Treat `tools/Project_Design_Workflow.md` as governing.
+3. Preserve the conservative tone of the architecture doc.
+4. Keep the document reusable and inspectable.
+5. Do not add speculative subsystems that are not needed for V1.
+
+## Success criteria
+
+This task is successful if:
+
+- the architecture doc clearly reflects the governing workflow sequence
+- the V1 execution record and review gate are integrated into the architecture
+- V1 boundaries and deferrals are explicit
+- the result reduces the mismatch identified in the review note without becoming a large rewrite
diff --git a/codex_prompts/008_define_minimal_bridge_runner_change_spec.md b/codex_prompts/008_define_minimal_bridge_runner_change_spec.md
new file mode 100644
index 0000000000000000000000000000000000000000..29044a2d5c39fae180b12a36fa9d788d87c05c6b
--- /dev/null
+++ b/codex_prompts/008_define_minimal_bridge_runner_change_spec.md
@@ -0,0 +1,80 @@
+# Task: Define the minimal bridge-runner change spec for V1 execution records
+
+You are working in this repository.
+
+Your task is to define the smallest change spec for `tools/codex/baby_run_prompt.py` so it can support the V1 execution record model without a larger refactor.
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Run_Review_Gate.md`
+- `tools/codex/baby_run_prompt.py`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__TIMESTAMP.md`
+
+## Goal
+
+Produce a narrow implementation spec for the current bridge runner.
+
+The spec should define only the smallest changes needed to make the runner:
+
+- emit a V1 execution record instead of a success-implies-acceptance note
+- preserve stable run identity
+- capture the minimum automatic fields
+- leave manual review fields untouched or initialized for later completion
+
+## Important framing
+
+This is a design-spec task.
+
+Do NOT implement the changes in this pass.
+Do NOT split the runner into multiple modules in this pass.
+Do NOT introduce a large CLI redesign.
+
+## Questions to settle
+
+Please settle these points:
+
+1. What exact filename pattern should the runner write for a run record?
+2. What exact markdown sections and fields should it populate automatically?
+3. What review fields should be initialized but left manual?
+4. How should execution status be derived from the subprocess result?
+5. What minimal runtime and output metrics should be captured now?
+6. What current behaviors in `baby_run_prompt.py` should remain unchanged for V1?
+
+## Required output artifact
+
+Create one markdown spec at:
+
+`tools/codex/V1_Bridge_Runner_Change_Spec.md`
+
+The spec should include:
+
+- purpose
+- scope
+- current behavior summary
+- required V1 changes
+- non-goals
+- exact data/field mapping from runner output to execution record
+- open questions, if any, that must be resolved before implementation
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Keep the spec single-file and bridge-sized.
+3. Preserve the current runner's thin role where possible.
+4. Do not design a large future module layout here.
+5. Keep the change set small enough that it could be reviewed and implemented in one later prompt.
+
+## Success criteria
+
+This task is successful if:
+
+- the runner change scope is small and explicit
+- the spec is directly grounded in the V1 execution record and review gate
+- execution and review are no longer conflated in the target behavior
+- the spec avoids a premature refactor
diff --git a/notes/005_prompt_queue_plan__20260415_202557.md b/notes/005_prompt_queue_plan__20260415_202557.md
new file mode 100644
index 0000000000000000000000000000000000000000..9b4ec8df08d7061cf79ca0e7b6950b95c2da254c
--- /dev/null
+++ b/notes/005_prompt_queue_plan__20260415_202557.md
@@ -0,0 +1,45 @@
+# Prompt Queue Plan
+
+## Why These Prompts Were Chosen
+
+These prompts were chosen because the current misalignment is concentrated at one seam:
+the system can execute prompts, but it does not yet define the review checkpoint, the V1 architectural boundary around that checkpoint, or the smallest runner behavior needed to support it.
+
+The queue stays small on purpose.
+It settles the workflow gap before any implementation prompt is allowed to widen scope.
+
+## Why This Order Reduces Risk
+
+The order moves from policy to architecture to bridge-tooling:
+
+1. define the review gate
+2. align the architecture doc to that gate and the execution record
+3. define the smallest runner change spec that fits those decisions
+
+This reduces the risk of hard-coding the wrong note model or the wrong queue semantics into the runner before the manual review loop is settled.
+
+## What Each Prompt Is Meant To Settle
+
+`006_define_v1_run_review_gate.md`
+
+- settles the missing review stop between execution and accepted progress
+
+`007_align_architecture_doc_to_v1_workflow.md`
+
+- settles the V1 boundary, artifact inventory, validation posture, and deferrals inside the architecture document
+
+`008_define_minimal_bridge_runner_change_spec.md`
+
+- settles the smallest bridge-runner behavior needed to emit the V1 execution record without a broader refactor
+
+## What Is Intentionally Deferred
+
+The following work is intentionally not in this queue:
+
+- runner refactor into multiple modules
+- richer state machines
+- retry intelligence
+- dependency-aware scheduling
+- aggregated reporting
+- token accounting beyond optional fields
+- broad CLI redesign
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
new file mode 100644
index 0000000000000000000000000000000000000000..2b3a9bd724ef6c395b9312859e80c60691a7a668
--- /dev/null
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -0,0 +1,251 @@
+# V1 Execution Record Artifact
+
+## Purpose
+
+Define the minimum durable record for one prompt run so the workflow preserves:
+
+- what was executed
+- what happened during execution
+- what a human later decided about the result
+- enough failure and resource evidence to guide the next reviewed step
+
+This artifact is meant to support the current conservative workflow, not a larger automation system.
+
+## Scope
+
+V1 covers one prompt run from prompt submission through later human review.
+
+It does not define:
+
+- multi-run reporting
+- dependency graphs
+- approval automation
+- module decomposition
+- a full state machine
+
+## Source Of Truth
+
+For V1, the source of truth for one run is one markdown execution-record file in `notes/`.
+
+The record body is authoritative.
+The filename is a locator and stable identifier carrier, but the body is the place where execution facts and review fields must be read from.
+
+V1 should avoid splitting one run across multiple required files.
+
+## Stable Identity For A Run
+
+Each run must have a stable `run_id` that does not change after the file is created.
+
+V1 format:
+
+`<prompt_stem>__<started_at_utc>`
+
+Example:
+
+`005_define_execution_record_and_generate_next_prompt_queue__20260415_202557`
+
+Why this is enough for V1:
+
+- it is human-readable
+- it distinguishes repeated runs of the same prompt
+- it does not depend on later review outcome
+- it can be created cheaply by the bridge runner
+
+The filename should use the same stable identity and should not encode accepted or rejected review outcome.
+
+## Required Fields
+
+Every V1 execution record should contain these fields.
+
+### Run Identity
+
+- `run_id`
+- `prompt_file`
+- `prompt_stem`
+- `started_at_utc`
+
+### Execution Facts
+
+- `execution_status`
+- `finished_at_utc`
+- `runner`
+- `return_code`
+- `prompt_text`
+- `codex_final_output`
+
+### Review Facts
+
+- `review_status`
+- `review_summary`
+
+### Minimum Failure / Retry Linkage
+
+- `retry_of_run_id`
+
+### Minimum Resource / Cost Facts
+
+- `elapsed_seconds`
+- `final_output_char_count`
+- `stderr_char_count`
+
+## Optional Fields
+
+These fields are useful when available, but should not block V1.
+
+- `reviewed_by`
+- `reviewed_at_utc`
+- `stderr_text`
+- `failure_analysis`
+- `recommended_next_action`
+- `model_name`
+- `token_usage`
+- `attempt_index_for_prompt`
+- `workspace_notes`
+
+## Automatic Vs Manual Fields
+
+### Automatic In V1
+
+These should be captured automatically when the runner can do so cheaply and reliably:
+
+- `run_id`
+- `prompt_file`
+- `prompt_stem`
+- `started_at_utc`
+- `finished_at_utc`
+- `execution_status`
+- `runner`
+- `return_code`
+- `prompt_text`
+- `codex_final_output`
+- `elapsed_seconds`
+- `final_output_char_count`
+- `stderr_char_count`
+- `stderr_text` when present
+- `retry_of_run_id` only if the runner was explicitly given that context
+
+### Manual In V1
+
+These remain human-owned in V1:
+
+- `review_status`
+- `review_summary`
+- `reviewed_by`
+- `reviewed_at_utc`
+- `failure_analysis`
+- `recommended_next_action`
+
+This split is intentional.
+Execution capture should be automatic.
+Acceptance and interpretation should remain manual until the review loop is defined more clearly.
+
+## Execution Status Vs Review Status
+
+V1 must keep these separate.
+
+### Execution Status
+
+This records what happened when the runner invoked Codex.
+
+Allowed V1 values:
+
+- `EXECUTED`
+- `EXECUTION_FAILED`
+
+`EXECUTED` means the subprocess completed successfully enough to return code `0`.
+It does not mean the work is accepted.
+
+`EXECUTION_FAILED` means the subprocess failed or the run artifact could only be completed with failure evidence.
+
+### Review Status
+
+This records the later human judgment.
+
+Allowed V1 values:
+
+- `UNREVIEWED`
+- `ACCEPTED`
+- `REJECTED`
+
+Rules:
+
+- a new run record should start as `UNREVIEWED`
+- `EXECUTED` must not imply `ACCEPTED`
+- `EXECUTION_FAILED` does not automatically imply `REJECTED`, though rejection will often follow
+- the next prompt in a reviewed sequence should wait for explicit human review, not only process exit success
+
+## Minimum Failure-Analysis Section
+
+V1 failure analysis should stay small.
+It should exist as a short section that can be filled in manually when useful, especially for `EXECUTION_FAILED` or `REJECTED` runs.
+
+Minimum fields:
+
+- `failure_type`
+- `failure_symptom`
+- `likely_cause`
+- `recommended_next_action`
+
+Suggested `failure_type` values:
+
+- `environment`
+- `runner`
+- `prompt_scope`
+- `prompt_clarity`
+- `implementation`
+- `validation`
+- `unknown`
+
+This is enough for V1 because it supports learning from failure without requiring a larger taxonomy.
+
+## Minimum Resource / Cost Section
+
+V1 should record only lightweight fields that are cheap to capture or inspect:
+
+- `started_at_utc`
+- `finished_at_utc`
+- `elapsed_seconds`
+- `return_code`
+- `final_output_char_count`
+- `stderr_char_count`
+
+Optional later additions:
+
+- `model_name`
+- `token_usage`
+- `tool_call_count`
+- `file_change_count`
+- `estimated_review_minutes`
+
+V1 should observe resource use, not build a billing system.
+
+## Practical Record Shape
+
+A V1 markdown record should have these sections in this order:
+
+1. Header / identity
+2. Execution facts
+3. Review facts
+4. Failure analysis
+5. Resource / cost facts
+6. Prompt text
+7. Codex final output
+8. Stderr or supplemental notes
+
+This keeps one run easy to inspect in a single file.
+
+## Intentionally Deferred From V1
+
+The following should wait:
+
+- automatic acceptance or approval gating
+- dependency-aware prompt release
+- richer review states such as `BLOCKED` or `PARTIAL`
+- structured JSON sidecars or databases
+- normalized token accounting across runners
+- automatic failure classification
+- aggregated reports and dashboards
+- full retry history synthesis
+- broader refactor of the runner into multiple modules
+
+V1 is only trying to establish a durable, reviewable single-run artifact that cleanly separates execution from acceptance.

diff --git a/codex_prompts/006_define_v1_run_review_gate.md b/codex_prompts/006_define_v1_run_review_gate.md
new file mode 100644
index 0000000000000000000000000000000000000000..3b21bed977ca97d9be3dd7b51cb78b8e74602781
--- /dev/null
+++ b/codex_prompts/006_define_v1_run_review_gate.md
@@ -0,0 +1,82 @@
+# Task: Define the V1 run-review gate for prompt execution
+
+You are working in this repository.
+
+Your task is to define the minimum review gate that must sit between:
+
+- a prompt run finishing
+- the next prompt in sequence being treated as ready
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/baby_run_prompt.py`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__TIMESTAMP.md`
+
+## Goal
+
+Use `tools/Project_Design_Workflow.md` as governing.
+
+Define the smallest practical V1 review gate that preserves:
+
+- review between iterations
+- separation of execution outcome from accepted outcome
+- a conservative manual step before the next prompt is treated as ready
+
+## Important framing
+
+This is a design task.
+
+Do NOT implement the gate in code in this pass.
+Do NOT create a large state machine.
+Do NOT redesign the whole queue system.
+
+## Questions to settle
+
+Please settle these points:
+
+1. What exact event marks a run as awaiting review?
+2. What minimum human checks must happen before a run can be accepted?
+3. What review outcomes are needed in V1?
+4. What outcome should let the next prompt proceed?
+5. What outcome should stop the queue and force a new design or retry decision?
+6. What information must be written back into the run record during review?
+
+## Required output artifact
+
+Create one markdown design note at:
+
+`tools/codex/V1_Run_Review_Gate.md`
+
+The note should include:
+
+- purpose
+- scope
+- the review trigger
+- the minimum manual checklist
+- allowed V1 review outcomes
+- how the outcome affects queue progression
+- how this connects to `tools/codex/V1_Execution_Record_Artifact.md`
+- what is intentionally deferred
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Keep the gate manual and conservative.
+3. Do not implement CLI or runner changes in this pass.
+4. Do not invent a large workflow beyond the next reviewed step.
+5. Keep the decision small enough to review before any implementation prompt follows.
+
+## Success criteria
+
+This task is successful if:
+
+- the review gate is explicit and easy to apply
+- execution success is clearly separated from accepted outcome
+- queue progression rules are clear for V1
+- the result is small enough to guide the next prompt without expanding scope
diff --git a/codex_prompts/007_align_architecture_doc_to_v1_workflow.md b/codex_prompts/007_align_architecture_doc_to_v1_workflow.md
new file mode 100644
index 0000000000000000000000000000000000000000..840d018ed4e23e74f8704c350a60f42fef84b960
--- /dev/null
+++ b/codex_prompts/007_align_architecture_doc_to_v1_workflow.md
@@ -0,0 +1,76 @@
+# Task: Align the Codex prompt architecture document to the governing workflow and V1 run artifact
+
+You are working in this repository.
+
+Your task is to bring the architecture document into explicit V1 alignment with:
+
+- the governing workflow
+- the V1 execution record
+- the V1 review gate
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Run_Review_Gate.md`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__TIMESTAMP.md`
+
+## Goal
+
+Update the architecture document so it no longer skips the workflow steps that matter for V1.
+
+The revised architecture should make clear:
+
+- the V1 problem being solved
+- the V1 scope boundary
+- the minimum artifact inventory
+- the role of the execution record
+- the role of the review gate
+- the conservative implementation order
+- what remains deferred
+
+## Important framing
+
+This is still a design-alignment task.
+
+Do NOT implement the architecture in code in this pass.
+Do NOT expand the document into a giant platform roadmap.
+Do NOT rewrite the governing workflow document.
+
+## Required output artifact
+
+Revise this file directly:
+
+`tools/Codex_Prompt_Workflow_Architecture.md`
+
+The revision should add or clarify:
+
+- short problem statement
+- V1 scope
+- out-of-scope items
+- minimum artifact inventory
+- minimum viable slice
+- implementation order
+- validation and review posture
+- explicit deferred items
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Treat `tools/Project_Design_Workflow.md` as governing.
+3. Preserve the conservative tone of the architecture doc.
+4. Keep the document reusable and inspectable.
+5. Do not add speculative subsystems that are not needed for V1.
+
+## Success criteria
+
+This task is successful if:
+
+- the architecture doc clearly reflects the governing workflow sequence
+- the V1 execution record and review gate are integrated into the architecture
+- V1 boundaries and deferrals are explicit
+- the result reduces the mismatch identified in the review note without becoming a large rewrite
diff --git a/codex_prompts/008_define_minimal_bridge_runner_change_spec.md b/codex_prompts/008_define_minimal_bridge_runner_change_spec.md
new file mode 100644
index 0000000000000000000000000000000000000000..29044a2d5c39fae180b12a36fa9d788d87c05c6b
--- /dev/null
+++ b/codex_prompts/008_define_minimal_bridge_runner_change_spec.md
@@ -0,0 +1,80 @@
+# Task: Define the minimal bridge-runner change spec for V1 execution records
+
+You are working in this repository.
+
+Your task is to define the smallest change spec for `tools/codex/baby_run_prompt.py` so it can support the V1 execution record model without a larger refactor.
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Run_Review_Gate.md`
+- `tools/codex/baby_run_prompt.py`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__TIMESTAMP.md`
+
+## Goal
+
+Produce a narrow implementation spec for the current bridge runner.
+
+The spec should define only the smallest changes needed to make the runner:
+
+- emit a V1 execution record instead of a success-implies-acceptance note
+- preserve stable run identity
+- capture the minimum automatic fields
+- leave manual review fields untouched or initialized for later completion
+
+## Important framing
+
+This is a design-spec task.
+
+Do NOT implement the changes in this pass.
+Do NOT split the runner into multiple modules in this pass.
+Do NOT introduce a large CLI redesign.
+
+## Questions to settle
+
+Please settle these points:
+
+1. What exact filename pattern should the runner write for a run record?
+2. What exact markdown sections and fields should it populate automatically?
+3. What review fields should be initialized but left manual?
+4. How should execution status be derived from the subprocess result?
+5. What minimal runtime and output metrics should be captured now?
+6. What current behaviors in `baby_run_prompt.py` should remain unchanged for V1?
+
+## Required output artifact
+
+Create one markdown spec at:
+
+`tools/codex/V1_Bridge_Runner_Change_Spec.md`
+
+The spec should include:
+
+- purpose
+- scope
+- current behavior summary
+- required V1 changes
+- non-goals
+- exact data/field mapping from runner output to execution record
+- open questions, if any, that must be resolved before implementation
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Keep the spec single-file and bridge-sized.
+3. Preserve the current runner's thin role where possible.
+4. Do not design a large future module layout here.
+5. Keep the change set small enough that it could be reviewed and implemented in one later prompt.
+
+## Success criteria
+
+This task is successful if:
+
+- the runner change scope is small and explicit
+- the spec is directly grounded in the V1 execution record and review gate
+- execution and review are no longer conflated in the target behavior
+- the spec avoids a premature refactor
diff --git a/notes/005_prompt_queue_plan__20260415_202557.md b/notes/005_prompt_queue_plan__20260415_202557.md
new file mode 100644
index 0000000000000000000000000000000000000000..9b4ec8df08d7061cf79ca0e7b6950b95c2da254c
--- /dev/null
+++ b/notes/005_prompt_queue_plan__20260415_202557.md
@@ -0,0 +1,45 @@
+# Prompt Queue Plan
+
+## Why These Prompts Were Chosen
+
+These prompts were chosen because the current misalignment is concentrated at one seam:
+the system can execute prompts, but it does not yet define the review checkpoint, the V1 architectural boundary around that checkpoint, or the smallest runner behavior needed to support it.
+
+The queue stays small on purpose.
+It settles the workflow gap before any implementation prompt is allowed to widen scope.
+
+## Why This Order Reduces Risk
+
+The order moves from policy to architecture to bridge-tooling:
+
+1. define the review gate
+2. align the architecture doc to that gate and the execution record
+3. define the smallest runner change spec that fits those decisions
+
+This reduces the risk of hard-coding the wrong note model or the wrong queue semantics into the runner before the manual review loop is settled.
+
+## What Each Prompt Is Meant To Settle
+
+`006_define_v1_run_review_gate.md`
+
+- settles the missing review stop between execution and accepted progress
+
+`007_align_architecture_doc_to_v1_workflow.md`
+
+- settles the V1 boundary, artifact inventory, validation posture, and deferrals inside the architecture document
+
+`008_define_minimal_bridge_runner_change_spec.md`
+
+- settles the smallest bridge-runner behavior needed to emit the V1 execution record without a broader refactor
+
+## What Is Intentionally Deferred
+
+The following work is intentionally not in this queue:
+
+- runner refactor into multiple modules
+- richer state machines
+- retry intelligence
+- dependency-aware scheduling
+- aggregated reporting
+- token accounting beyond optional fields
+- broad CLI redesign
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
new file mode 100644
index 0000000000000000000000000000000000000000..2b3a9bd724ef6c395b9312859e80c60691a7a668
--- /dev/null
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -0,0 +1,251 @@
+# V1 Execution Record Artifact
+
+## Purpose
+
+Define the minimum durable record for one prompt run so the workflow preserves:
+
+- what was executed
+- what happened during execution
+- what a human later decided about the result
+- enough failure and resource evidence to guide the next reviewed step
+
+This artifact is meant to support the current conservative workflow, not a larger automation system.
+
+## Scope
+
+V1 covers one prompt run from prompt submission through later human review.
+
+It does not define:
+
+- multi-run reporting
+- dependency graphs
+- approval automation
+- module decomposition
+- a full state machine
+
+## Source Of Truth
+
+For V1, the source of truth for one run is one markdown execution-record file in `notes/`.
+
+The record body is authoritative.
+The filename is a locator and stable identifier carrier, but the body is the place where execution facts and review fields must be read from.
+
+V1 should avoid splitting one run across multiple required files.
+
+## Stable Identity For A Run
+
+Each run must have a stable `run_id` that does not change after the file is created.
+
+V1 format:
+
+`<prompt_stem>__<started_at_utc>`
+
+Example:
+
+`005_define_execution_record_and_generate_next_prompt_queue__20260415_202557`
+
+Why this is enough for V1:
+
+- it is human-readable
+- it distinguishes repeated runs of the same prompt
+- it does not depend on later review outcome
+- it can be created cheaply by the bridge runner
+
+The filename should use the same stable identity and should not encode accepted or rejected review outcome.
+
+## Required Fields
+
+Every V1 execution record should contain these fields.
+
+### Run Identity
+
+- `run_id`
+- `prompt_file`
+- `prompt_stem`
+- `started_at_utc`
+
+### Execution Facts
+
+- `execution_status`
+- `finished_at_utc`
+- `runner`
+- `return_code`
+- `prompt_text`
+- `codex_final_output`
+
+### Review Facts
+
+- `review_status`
+- `review_summary`
+
+### Minimum Failure / Retry Linkage
+
+- `retry_of_run_id`
+
+### Minimum Resource / Cost Facts
+
+- `elapsed_seconds`
+- `final_output_char_count`
+- `stderr_char_count`
+
+## Optional Fields
+
+These fields are useful when available, but should not block V1.
+
+- `reviewed_by`
+- `reviewed_at_utc`
+- `stderr_text`
+- `failure_analysis`
+- `recommended_next_action`
+- `model_name`
+- `token_usage`
+- `attempt_index_for_prompt`
+- `workspace_notes`
+
+## Automatic Vs Manual Fields
+
+### Automatic In V1
+
+These should be captured automatically when the runner can do so cheaply and reliably:
+
+- `run_id`
+- `prompt_file`
+- `prompt_stem`
+- `started_at_utc`
+- `finished_at_utc`
+- `execution_status`
+- `runner`
+- `return_code`
+- `prompt_text`
+- `codex_final_output`
+- `elapsed_seconds`
+- `final_output_char_count`
+- `stderr_char_count`
+- `stderr_text` when present
+- `retry_of_run_id` only if the runner was explicitly given that context
+
+### Manual In V1
+
+These remain human-owned in V1:
+
+- `review_status`
+- `review_summary`
+- `reviewed_by`
+- `reviewed_at_utc`
+- `failure_analysis`
+- `recommended_next_action`
+
+This split is intentional.
+Execution capture should be automatic.
+Acceptance and interpretation should remain manual until the review loop is defined more clearly.
+
+## Execution Status Vs Review Status
+
+V1 must keep these separate.
+
+### Execution Status
+
+This records what happened when the runner invoked Codex.
+
+Allowed V1 values:
+
+- `EXECUTED`
+- `EXECUTION_FAILED`
+
+`EXECUTED` means the subprocess completed successfully enough to return code `0`.
+It does not mean the work is accepted.
+
+`EXECUTION_FAILED` means the subprocess failed or the run artifact could only be completed with failure evidence.
+
+### Review Status
+
+This records the later human judgment.
+
+Allowed V1 values:
+
+- `UNREVIEWED`
+- `ACCEPTED`
+- `REJECTED`
+
+Rules:
+
+- a new run record should start as `UNREVIEWED`
+- `EXECUTED` must not imply `ACCEPTED`
+- `EXECUTION_FAILED` does not automatically imply `REJECTED`, though rejection will often follow
+- the next prompt in a reviewed sequence should wait for explicit human review, not only process exit success
+
+## Minimum Failure-Analysis Section
+
+V1 failure analysis should stay small.
+It should exist as a short section that can be filled in manually when useful, especially for `EXECUTION_FAILED` or `REJECTED` runs.
+
+Minimum fields:
+
+- `failure_type`
+- `failure_symptom`
+- `likely_cause`
+- `recommended_next_action`
+
+Suggested `failure_type` values:
+
+- `environment`
+- `runner`
+- `prompt_scope`
+- `prompt_clarity`
+- `implementation`
+- `validation`
+- `unknown`
+
+This is enough for V1 because it supports learning from failure without requiring a larger taxonomy.
+
+## Minimum Resource / Cost Section
+
+V1 should record only lightweight fields that are cheap to capture or inspect:
+
+- `started_at_utc`
+- `finished_at_utc`
+- `elapsed_seconds`
+- `return_code`
+- `final_output_char_count`
+- `stderr_char_count`
+
+Optional later additions:
+
+- `model_name`
+- `token_usage`
+- `tool_call_count`
+- `file_change_count`
+- `estimated_review_minutes`
+
+V1 should observe resource use, not build a billing system.
+
+## Practical Record Shape
+
+A V1 markdown record should have these sections in this order:
+
+1. Header / identity
+2. Execution facts
+3. Review facts
+4. Failure analysis
+5. Resource / cost facts
+6. Prompt text
+7. Codex final output
+8. Stderr or supplemental notes
+
+This keeps one run easy to inspect in a single file.
+
+## Intentionally Deferred From V1
+
+The following should wait:
+
+- automatic acceptance or approval gating
+- dependency-aware prompt release
+- richer review states such as `BLOCKED` or `PARTIAL`
+- structured JSON sidecars or databases
+- normalized token accounting across runners
+- automatic failure classification
+- aggregated reports and dashboards
+- full retry history synthesis
+- broader refactor of the runner into multiple modules
+
+V1 is only trying to establish a durable, reviewable single-run artifact that cleanly separates execution from acceptance.

codex
The first draft is written. I’m doing a quick verification pass for path exactness and queue quality; the main thing I’m checking is that the new prompts point to concrete files rather than placeholders.
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
/bin/bash -lc "sed -n '1,240p' codex_prompts/006_define_v1_run_review_gate.md" in /data/git/handson-ml3
 succeeded in 0ms:
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
- `notes/005_prompt_queue_plan__TIMESTAMP.md`

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

exec
/bin/bash -lc "sed -n '1,240p' codex_prompts/007_align_architecture_doc_to_v1_workflow.md" in /data/git/handson-ml3
 succeeded in 0ms:
# Task: Align the Codex prompt architecture document to the governing workflow and V1 run artifact

You are working in this repository.

Your task is to bring the architecture document into explicit V1 alignment with:

- the governing workflow
- the V1 execution record
- the V1 review gate

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
- `notes/005_prompt_queue_plan__TIMESTAMP.md`

## Goal

Update the architecture document so it no longer skips the workflow steps that matter for V1.

The revised architecture should make clear:

- the V1 problem being solved
- the V1 scope boundary
- the minimum artifact inventory
- the role of the execution record
- the role of the review gate
- the conservative implementation order
- what remains deferred

## Important framing

This is still a design-alignment task.

Do NOT implement the architecture in code in this pass.
Do NOT expand the document into a giant platform roadmap.
Do NOT rewrite the governing workflow document.

## Required output artifact

Revise this file directly:

`tools/Codex_Prompt_Workflow_Architecture.md`

The revision should add or clarify:

- short problem statement
- V1 scope
- out-of-scope items
- minimum artifact inventory
- minimum viable slice
- implementation order
- validation and review posture
- explicit deferred items

## Constraints

1. Use the exact file paths listed above.
2. Treat `tools/Project_Design_Workflow.md` as governing.
3. Preserve the conservative tone of the architecture doc.
4. Keep the document reusable and inspectable.
5. Do not add speculative subsystems that are not needed for V1.

## Success criteria

This task is successful if:

- the architecture doc clearly reflects the governing workflow sequence
- the V1 execution record and review gate are integrated into the architecture
- V1 boundaries and deferrals are explicit
- the result reduces the mismatch identified in the review note without becoming a large rewrite

exec
/bin/bash -lc "sed -n '1,240p' codex_prompts/008_define_minimal_bridge_runner_change_spec.md" in /data/git/handson-ml3
 succeeded in 0ms:
# Task: Define the minimal bridge-runner change spec for V1 execution records

You are working in this repository.

Your task is to define the smallest change spec for `tools/codex/baby_run_prompt.py` so it can support the V1 execution record model without a larger refactor.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/baby_run_prompt.py`
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
- `notes/005_prompt_queue_plan__TIMESTAMP.md`

## Goal

Produce a narrow implementation spec for the current bridge runner.

The spec should define only the smallest changes needed to make the runner:

- emit a V1 execution record instead of a success-implies-acceptance note
- preserve stable run identity
- capture the minimum automatic fields
- leave manual review fields untouched or initialized for later completion

## Important framing

This is a design-spec task.

Do NOT implement the changes in this pass.
Do NOT split the runner into multiple modules in this pass.
Do NOT introduce a large CLI redesign.

## Questions to settle

Please settle these points:

1. What exact filename pattern should the runner write for a run record?
2. What exact markdown sections and fields should it populate automatically?
3. What review fields should be initialized but left manual?
4. How should execution status be derived from the subprocess result?
5. What minimal runtime and output metrics should be captured now?
6. What current behaviors in `baby_run_prompt.py` should remain unchanged for V1?

## Required output artifact

Create one markdown spec at:

`tools/codex/V1_Bridge_Runner_Change_Spec.md`

The spec should include:

- purpose
- scope
- current behavior summary
- required V1 changes
- non-goals
- exact data/field mapping from runner output to execution record
- open questions, if any, that must be resolved before implementation

## Constraints

1. Use the exact file paths listed above.
2. Keep the spec single-file and bridge-sized.
3. Preserve the current runner's thin role where possible.
4. Do not design a large future module layout here.
5. Keep the change set small enough that it could be reviewed and implemented in one later prompt.

## Success criteria

This task is successful if:

- the runner change scope is small and explicit
- the spec is directly grounded in the V1 execution record and review gate
- execution and review are no longer conflated in the target behavior
- the spec avoids a premature refactor

exec
/bin/bash -lc "sed -n '1,220p' notes/005_prompt_queue_plan__20260415_202557.md" in /data/git/handson-ml3
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

diff --git a/codex_prompts/006_define_v1_run_review_gate.md b/codex_prompts/006_define_v1_run_review_gate.md
new file mode 100644
index 0000000000000000000000000000000000000000..3b21bed977ca97d9be3dd7b51cb78b8e74602781
--- /dev/null
+++ b/codex_prompts/006_define_v1_run_review_gate.md
@@ -0,0 +1,82 @@
+# Task: Define the V1 run-review gate for prompt execution
+
+You are working in this repository.
+
+Your task is to define the minimum review gate that must sit between:
+
+- a prompt run finishing
+- the next prompt in sequence being treated as ready
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/baby_run_prompt.py`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__TIMESTAMP.md`
+
+## Goal
+
+Use `tools/Project_Design_Workflow.md` as governing.
+
+Define the smallest practical V1 review gate that preserves:
+
+- review between iterations
+- separation of execution outcome from accepted outcome
+- a conservative manual step before the next prompt is treated as ready
+
+## Important framing
+
+This is a design task.
+
+Do NOT implement the gate in code in this pass.
+Do NOT create a large state machine.
+Do NOT redesign the whole queue system.
+
+## Questions to settle
+
+Please settle these points:
+
+1. What exact event marks a run as awaiting review?
+2. What minimum human checks must happen before a run can be accepted?
+3. What review outcomes are needed in V1?
+4. What outcome should let the next prompt proceed?
+5. What outcome should stop the queue and force a new design or retry decision?
+6. What information must be written back into the run record during review?
+
+## Required output artifact
+
+Create one markdown design note at:
+
+`tools/codex/V1_Run_Review_Gate.md`
+
+The note should include:
+
+- purpose
+- scope
+- the review trigger
+- the minimum manual checklist
+- allowed V1 review outcomes
+- how the outcome affects queue progression
+- how this connects to `tools/codex/V1_Execution_Record_Artifact.md`
+- what is intentionally deferred
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Keep the gate manual and conservative.
+3. Do not implement CLI or runner changes in this pass.
+4. Do not invent a large workflow beyond the next reviewed step.
+5. Keep the decision small enough to review before any implementation prompt follows.
+
+## Success criteria
+
+This task is successful if:
+
+- the review gate is explicit and easy to apply
+- execution success is clearly separated from accepted outcome
+- queue progression rules are clear for V1
+- the result is small enough to guide the next prompt without expanding scope
diff --git a/codex_prompts/007_align_architecture_doc_to_v1_workflow.md b/codex_prompts/007_align_architecture_doc_to_v1_workflow.md
new file mode 100644
index 0000000000000000000000000000000000000000..840d018ed4e23e74f8704c350a60f42fef84b960
--- /dev/null
+++ b/codex_prompts/007_align_architecture_doc_to_v1_workflow.md
@@ -0,0 +1,76 @@
+# Task: Align the Codex prompt architecture document to the governing workflow and V1 run artifact
+
+You are working in this repository.
+
+Your task is to bring the architecture document into explicit V1 alignment with:
+
+- the governing workflow
+- the V1 execution record
+- the V1 review gate
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Run_Review_Gate.md`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__TIMESTAMP.md`
+
+## Goal
+
+Update the architecture document so it no longer skips the workflow steps that matter for V1.
+
+The revised architecture should make clear:
+
+- the V1 problem being solved
+- the V1 scope boundary
+- the minimum artifact inventory
+- the role of the execution record
+- the role of the review gate
+- the conservative implementation order
+- what remains deferred
+
+## Important framing
+
+This is still a design-alignment task.
+
+Do NOT implement the architecture in code in this pass.
+Do NOT expand the document into a giant platform roadmap.
+Do NOT rewrite the governing workflow document.
+
+## Required output artifact
+
+Revise this file directly:
+
+`tools/Codex_Prompt_Workflow_Architecture.md`
+
+The revision should add or clarify:
+
+- short problem statement
+- V1 scope
+- out-of-scope items
+- minimum artifact inventory
+- minimum viable slice
+- implementation order
+- validation and review posture
+- explicit deferred items
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Treat `tools/Project_Design_Workflow.md` as governing.
+3. Preserve the conservative tone of the architecture doc.
+4. Keep the document reusable and inspectable.
+5. Do not add speculative subsystems that are not needed for V1.
+
+## Success criteria
+
+This task is successful if:
+
+- the architecture doc clearly reflects the governing workflow sequence
+- the V1 execution record and review gate are integrated into the architecture
+- V1 boundaries and deferrals are explicit
+- the result reduces the mismatch identified in the review note without becoming a large rewrite
diff --git a/codex_prompts/008_define_minimal_bridge_runner_change_spec.md b/codex_prompts/008_define_minimal_bridge_runner_change_spec.md
new file mode 100644
index 0000000000000000000000000000000000000000..29044a2d5c39fae180b12a36fa9d788d87c05c6b
--- /dev/null
+++ b/codex_prompts/008_define_minimal_bridge_runner_change_spec.md
@@ -0,0 +1,80 @@
+# Task: Define the minimal bridge-runner change spec for V1 execution records
+
+You are working in this repository.
+
+Your task is to define the smallest change spec for `tools/codex/baby_run_prompt.py` so it can support the V1 execution record model without a larger refactor.
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Run_Review_Gate.md`
+- `tools/codex/baby_run_prompt.py`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__TIMESTAMP.md`
+
+## Goal
+
+Produce a narrow implementation spec for the current bridge runner.
+
+The spec should define only the smallest changes needed to make the runner:
+
+- emit a V1 execution record instead of a success-implies-acceptance note
+- preserve stable run identity
+- capture the minimum automatic fields
+- leave manual review fields untouched or initialized for later completion
+
+## Important framing
+
+This is a design-spec task.
+
+Do NOT implement the changes in this pass.
+Do NOT split the runner into multiple modules in this pass.
+Do NOT introduce a large CLI redesign.
+
+## Questions to settle
+
+Please settle these points:
+
+1. What exact filename pattern should the runner write for a run record?
+2. What exact markdown sections and fields should it populate automatically?
+3. What review fields should be initialized but left manual?
+4. How should execution status be derived from the subprocess result?
+5. What minimal runtime and output metrics should be captured now?
+6. What current behaviors in `baby_run_prompt.py` should remain unchanged for V1?
+
+## Required output artifact
+
+Create one markdown spec at:
+
+`tools/codex/V1_Bridge_Runner_Change_Spec.md`
+
+The spec should include:
+
+- purpose
+- scope
+- current behavior summary
+- required V1 changes
+- non-goals
+- exact data/field mapping from runner output to execution record
+- open questions, if any, that must be resolved before implementation
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Keep the spec single-file and bridge-sized.
+3. Preserve the current runner's thin role where possible.
+4. Do not design a large future module layout here.
+5. Keep the change set small enough that it could be reviewed and implemented in one later prompt.
+
+## Success criteria
+
+This task is successful if:
+
+- the runner change scope is small and explicit
+- the spec is directly grounded in the V1 execution record and review gate
+- execution and review are no longer conflated in the target behavior
+- the spec avoids a premature refactor
diff --git a/notes/005_prompt_queue_plan__20260415_202557.md b/notes/005_prompt_queue_plan__20260415_202557.md
new file mode 100644
index 0000000000000000000000000000000000000000..9b4ec8df08d7061cf79ca0e7b6950b95c2da254c
--- /dev/null
+++ b/notes/005_prompt_queue_plan__20260415_202557.md
@@ -0,0 +1,45 @@
+# Prompt Queue Plan
+
+## Why These Prompts Were Chosen
+
+These prompts were chosen because the current misalignment is concentrated at one seam:
+the system can execute prompts, but it does not yet define the review checkpoint, the V1 architectural boundary around that checkpoint, or the smallest runner behavior needed to support it.
+
+The queue stays small on purpose.
+It settles the workflow gap before any implementation prompt is allowed to widen scope.
+
+## Why This Order Reduces Risk
+
+The order moves from policy to architecture to bridge-tooling:
+
+1. define the review gate
+2. align the architecture doc to that gate and the execution record
+3. define the smallest runner change spec that fits those decisions
+
+This reduces the risk of hard-coding the wrong note model or the wrong queue semantics into the runner before the manual review loop is settled.
+
+## What Each Prompt Is Meant To Settle
+
+`006_define_v1_run_review_gate.md`
+
+- settles the missing review stop between execution and accepted progress
+
+`007_align_architecture_doc_to_v1_workflow.md`
+
+- settles the V1 boundary, artifact inventory, validation posture, and deferrals inside the architecture document
+
+`008_define_minimal_bridge_runner_change_spec.md`
+
+- settles the smallest bridge-runner behavior needed to emit the V1 execution record without a broader refactor
+
+## What Is Intentionally Deferred
+
+The following work is intentionally not in this queue:
+
+- runner refactor into multiple modules
+- richer state machines
+- retry intelligence
+- dependency-aware scheduling
+- aggregated reporting
+- token accounting beyond optional fields
+- broad CLI redesign
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
new file mode 100644
index 0000000000000000000000000000000000000000..2b3a9bd724ef6c395b9312859e80c60691a7a668
--- /dev/null
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -0,0 +1,251 @@
+# V1 Execution Record Artifact
+
+## Purpose
+
+Define the minimum durable record for one prompt run so the workflow preserves:
+
+- what was executed
+- what happened during execution
+- what a human later decided about the result
+- enough failure and resource evidence to guide the next reviewed step
+
+This artifact is meant to support the current conservative workflow, not a larger automation system.
+
+## Scope
+
+V1 covers one prompt run from prompt submission through later human review.
+
+It does not define:
+
+- multi-run reporting
+- dependency graphs
+- approval automation
+- module decomposition
+- a full state machine
+
+## Source Of Truth
+
+For V1, the source of truth for one run is one markdown execution-record file in `notes/`.
+
+The record body is authoritative.
+The filename is a locator and stable identifier carrier, but the body is the place where execution facts and review fields must be read from.
+
+V1 should avoid splitting one run across multiple required files.
+
+## Stable Identity For A Run
+
+Each run must have a stable `run_id` that does not change after the file is created.
+
+V1 format:
+
+`<prompt_stem>__<started_at_utc>`
+
+Example:
+
+`005_define_execution_record_and_generate_next_prompt_queue__20260415_202557`
+
+Why this is enough for V1:
+
+- it is human-readable
+- it distinguishes repeated runs of the same prompt
+- it does not depend on later review outcome
+- it can be created cheaply by the bridge runner
+
+The filename should use the same stable identity and should not encode accepted or rejected review outcome.
+
+## Required Fields
+
+Every V1 execution record should contain these fields.
+
+### Run Identity
+
+- `run_id`
+- `prompt_file`
+- `prompt_stem`
+- `started_at_utc`
+
+### Execution Facts
+
+- `execution_status`
+- `finished_at_utc`
+- `runner`
+- `return_code`
+- `prompt_text`
+- `codex_final_output`
+
+### Review Facts
+
+- `review_status`
+- `review_summary`
+
+### Minimum Failure / Retry Linkage
+
+- `retry_of_run_id`
+
+### Minimum Resource / Cost Facts
+
+- `elapsed_seconds`
+- `final_output_char_count`
+- `stderr_char_count`
+
+## Optional Fields
+
+These fields are useful when available, but should not block V1.
+
+- `reviewed_by`
+- `reviewed_at_utc`
+- `stderr_text`
+- `failure_analysis`
+- `recommended_next_action`
+- `model_name`
+- `token_usage`
+- `attempt_index_for_prompt`
+- `workspace_notes`
+
+## Automatic Vs Manual Fields
+
+### Automatic In V1
+
+These should be captured automatically when the runner can do so cheaply and reliably:
+
+- `run_id`
+- `prompt_file`
+- `prompt_stem`
+- `started_at_utc`
+- `finished_at_utc`
+- `execution_status`
+- `runner`
+- `return_code`
+- `prompt_text`
+- `codex_final_output`
+- `elapsed_seconds`
+- `final_output_char_count`
+- `stderr_char_count`
+- `stderr_text` when present
+- `retry_of_run_id` only if the runner was explicitly given that context
+
+### Manual In V1
+
+These remain human-owned in V1:
+
+- `review_status`
+- `review_summary`
+- `reviewed_by`
+- `reviewed_at_utc`
+- `failure_analysis`
+- `recommended_next_action`
+
+This split is intentional.
+Execution capture should be automatic.
+Acceptance and interpretation should remain manual until the review loop is defined more clearly.
+
+## Execution Status Vs Review Status
+
+V1 must keep these separate.
+
+### Execution Status
+
+This records what happened when the runner invoked Codex.
+
+Allowed V1 values:
+
+- `EXECUTED`
+- `EXECUTION_FAILED`
+
+`EXECUTED` means the subprocess completed successfully enough to return code `0`.
+It does not mean the work is accepted.
+
+`EXECUTION_FAILED` means the subprocess failed or the run artifact could only be completed with failure evidence.
+
+### Review Status
+
+This records the later human judgment.
+
+Allowed V1 values:
+
+- `UNREVIEWED`
+- `ACCEPTED`
+- `REJECTED`
+
+Rules:
+
+- a new run record should start as `UNREVIEWED`
+- `EXECUTED` must not imply `ACCEPTED`
+- `EXECUTION_FAILED` does not automatically imply `REJECTED`, though rejection will often follow
+- the next prompt in a reviewed sequence should wait for explicit human review, not only process exit success
+
+## Minimum Failure-Analysis Section
+
+V1 failure analysis should stay small.
+It should exist as a short section that can be filled in manually when useful, especially for `EXECUTION_FAILED` or `REJECTED` runs.
+
+Minimum fields:
+
+- `failure_type`
+- `failure_symptom`
+- `likely_cause`
+- `recommended_next_action`
+
+Suggested `failure_type` values:
+
+- `environment`
+- `runner`
+- `prompt_scope`
+- `prompt_clarity`
+- `implementation`
+- `validation`
+- `unknown`
+
+This is enough for V1 because it supports learning from failure without requiring a larger taxonomy.
+
+## Minimum Resource / Cost Section
+
+V1 should record only lightweight fields that are cheap to capture or inspect:
+
+- `started_at_utc`
+- `finished_at_utc`
+- `elapsed_seconds`
+- `return_code`
+- `final_output_char_count`
+- `stderr_char_count`
+
+Optional later additions:
+
+- `model_name`
+- `token_usage`
+- `tool_call_count`
+- `file_change_count`
+- `estimated_review_minutes`
+
+V1 should observe resource use, not build a billing system.
+
+## Practical Record Shape
+
+A V1 markdown record should have these sections in this order:
+
+1. Header / identity
+2. Execution facts
+3. Review facts
+4. Failure analysis
+5. Resource / cost facts
+6. Prompt text
+7. Codex final output
+8. Stderr or supplemental notes
+
+This keeps one run easy to inspect in a single file.
+
+## Intentionally Deferred From V1
+
+The following should wait:
+
+- automatic acceptance or approval gating
+- dependency-aware prompt release
+- richer review states such as `BLOCKED` or `PARTIAL`
+- structured JSON sidecars or databases
+- normalized token accounting across runners
+- automatic failure classification
+- aggregated reports and dashboards
+- full retry history synthesis
+- broader refactor of the runner into multiple modules
+
+V1 is only trying to establish a durable, reviewable single-run artifact that cleanly separates execution from acceptance.

codex
I found one concrete issue in the generated prompts: three of them still reference the queue note with a `TIMESTAMP` placeholder. I’m correcting those to the exact note path now so the follow-on prompts are directly runnable.
apply patch
patch: completed
/data/git/handson-ml3/codex_prompts/006_define_v1_run_review_gate.md
/data/git/handson-ml3/codex_prompts/007_align_architecture_doc_to_v1_workflow.md
/data/git/handson-ml3/codex_prompts/008_define_minimal_bridge_runner_change_spec.md
diff --git a/codex_prompts/006_define_v1_run_review_gate.md b/codex_prompts/006_define_v1_run_review_gate.md
new file mode 100644
index 0000000000000000000000000000000000000000..474c616cd105b06aae41f921b2229fcfbd38d117
--- /dev/null
+++ b/codex_prompts/006_define_v1_run_review_gate.md
@@ -0,0 +1,82 @@
+# Task: Define the V1 run-review gate for prompt execution
+
+You are working in this repository.
+
+Your task is to define the minimum review gate that must sit between:
+
+- a prompt run finishing
+- the next prompt in sequence being treated as ready
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/baby_run_prompt.py`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+
+## Goal
+
+Use `tools/Project_Design_Workflow.md` as governing.
+
+Define the smallest practical V1 review gate that preserves:
+
+- review between iterations
+- separation of execution outcome from accepted outcome
+- a conservative manual step before the next prompt is treated as ready
+
+## Important framing
+
+This is a design task.
+
+Do NOT implement the gate in code in this pass.
+Do NOT create a large state machine.
+Do NOT redesign the whole queue system.
+
+## Questions to settle
+
+Please settle these points:
+
+1. What exact event marks a run as awaiting review?
+2. What minimum human checks must happen before a run can be accepted?
+3. What review outcomes are needed in V1?
+4. What outcome should let the next prompt proceed?
+5. What outcome should stop the queue and force a new design or retry decision?
+6. What information must be written back into the run record during review?
+
+## Required output artifact
+
+Create one markdown design note at:
+
+`tools/codex/V1_Run_Review_Gate.md`
+
+The note should include:
+
+- purpose
+- scope
+- the review trigger
+- the minimum manual checklist
+- allowed V1 review outcomes
+- how the outcome affects queue progression
+- how this connects to `tools/codex/V1_Execution_Record_Artifact.md`
+- what is intentionally deferred
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Keep the gate manual and conservative.
+3. Do not implement CLI or runner changes in this pass.
+4. Do not invent a large workflow beyond the next reviewed step.
+5. Keep the decision small enough to review before any implementation prompt follows.
+
+## Success criteria
+
+This task is successful if:
+
+- the review gate is explicit and easy to apply
+- execution success is clearly separated from accepted outcome
+- queue progression rules are clear for V1
+- the result is small enough to guide the next prompt without expanding scope
diff --git a/codex_prompts/007_align_architecture_doc_to_v1_workflow.md b/codex_prompts/007_align_architecture_doc_to_v1_workflow.md
new file mode 100644
index 0000000000000000000000000000000000000000..5ac69c8f729baba517263fb8d6e834941c404b4e
--- /dev/null
+++ b/codex_prompts/007_align_architecture_doc_to_v1_workflow.md
@@ -0,0 +1,76 @@
+# Task: Align the Codex prompt architecture document to the governing workflow and V1 run artifact
+
+You are working in this repository.
+
+Your task is to bring the architecture document into explicit V1 alignment with:
+
+- the governing workflow
+- the V1 execution record
+- the V1 review gate
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Run_Review_Gate.md`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+
+## Goal
+
+Update the architecture document so it no longer skips the workflow steps that matter for V1.
+
+The revised architecture should make clear:
+
+- the V1 problem being solved
+- the V1 scope boundary
+- the minimum artifact inventory
+- the role of the execution record
+- the role of the review gate
+- the conservative implementation order
+- what remains deferred
+
+## Important framing
+
+This is still a design-alignment task.
+
+Do NOT implement the architecture in code in this pass.
+Do NOT expand the document into a giant platform roadmap.
+Do NOT rewrite the governing workflow document.
+
+## Required output artifact
+
+Revise this file directly:
+
+`tools/Codex_Prompt_Workflow_Architecture.md`
+
+The revision should add or clarify:
+
+- short problem statement
+- V1 scope
+- out-of-scope items
+- minimum artifact inventory
+- minimum viable slice
+- implementation order
+- validation and review posture
+- explicit deferred items
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Treat `tools/Project_Design_Workflow.md` as governing.
+3. Preserve the conservative tone of the architecture doc.
+4. Keep the document reusable and inspectable.
+5. Do not add speculative subsystems that are not needed for V1.
+
+## Success criteria
+
+This task is successful if:
+
+- the architecture doc clearly reflects the governing workflow sequence
+- the V1 execution record and review gate are integrated into the architecture
+- V1 boundaries and deferrals are explicit
+- the result reduces the mismatch identified in the review note without becoming a large rewrite
diff --git a/codex_prompts/008_define_minimal_bridge_runner_change_spec.md b/codex_prompts/008_define_minimal_bridge_runner_change_spec.md
new file mode 100644
index 0000000000000000000000000000000000000000..9d4d0df20196a88f4e4fc874a8c4abe51ef7e788
--- /dev/null
+++ b/codex_prompts/008_define_minimal_bridge_runner_change_spec.md
@@ -0,0 +1,80 @@
+# Task: Define the minimal bridge-runner change spec for V1 execution records
+
+You are working in this repository.
+
+Your task is to define the smallest change spec for `tools/codex/baby_run_prompt.py` so it can support the V1 execution record model without a larger refactor.
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Run_Review_Gate.md`
+- `tools/codex/baby_run_prompt.py`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+
+## Goal
+
+Produce a narrow implementation spec for the current bridge runner.
+
+The spec should define only the smallest changes needed to make the runner:
+
+- emit a V1 execution record instead of a success-implies-acceptance note
+- preserve stable run identity
+- capture the minimum automatic fields
+- leave manual review fields untouched or initialized for later completion
+
+## Important framing
+
+This is a design-spec task.
+
+Do NOT implement the changes in this pass.
+Do NOT split the runner into multiple modules in this pass.
+Do NOT introduce a large CLI redesign.
+
+## Questions to settle
+
+Please settle these points:
+
+1. What exact filename pattern should the runner write for a run record?
+2. What exact markdown sections and fields should it populate automatically?
+3. What review fields should be initialized but left manual?
+4. How should execution status be derived from the subprocess result?
+5. What minimal runtime and output metrics should be captured now?
+6. What current behaviors in `baby_run_prompt.py` should remain unchanged for V1?
+
+## Required output artifact
+
+Create one markdown spec at:
+
+`tools/codex/V1_Bridge_Runner_Change_Spec.md`
+
+The spec should include:
+
+- purpose
+- scope
+- current behavior summary
+- required V1 changes
+- non-goals
+- exact data/field mapping from runner output to execution record
+- open questions, if any, that must be resolved before implementation
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Keep the spec single-file and bridge-sized.
+3. Preserve the current runner's thin role where possible.
+4. Do not design a large future module layout here.
+5. Keep the change set small enough that it could be reviewed and implemented in one later prompt.
+
+## Success criteria
+
+This task is successful if:
+
+- the runner change scope is small and explicit
+- the spec is directly grounded in the V1 execution record and review gate
+- execution and review are no longer conflated in the target behavior
+- the spec avoids a premature refactor
diff --git a/notes/005_prompt_queue_plan__20260415_202557.md b/notes/005_prompt_queue_plan__20260415_202557.md
new file mode 100644
index 0000000000000000000000000000000000000000..9b4ec8df08d7061cf79ca0e7b6950b95c2da254c
--- /dev/null
+++ b/notes/005_prompt_queue_plan__20260415_202557.md
@@ -0,0 +1,45 @@
+# Prompt Queue Plan
+
+## Why These Prompts Were Chosen
+
+These prompts were chosen because the current misalignment is concentrated at one seam:
+the system can execute prompts, but it does not yet define the review checkpoint, the V1 architectural boundary around that checkpoint, or the smallest runner behavior needed to support it.
+
+The queue stays small on purpose.
+It settles the workflow gap before any implementation prompt is allowed to widen scope.
+
+## Why This Order Reduces Risk
+
+The order moves from policy to architecture to bridge-tooling:
+
+1. define the review gate
+2. align the architecture doc to that gate and the execution record
+3. define the smallest runner change spec that fits those decisions
+
+This reduces the risk of hard-coding the wrong note model or the wrong queue semantics into the runner before the manual review loop is settled.
+
+## What Each Prompt Is Meant To Settle
+
+`006_define_v1_run_review_gate.md`
+
+- settles the missing review stop between execution and accepted progress
+
+`007_align_architecture_doc_to_v1_workflow.md`
+
+- settles the V1 boundary, artifact inventory, validation posture, and deferrals inside the architecture document
+
+`008_define_minimal_bridge_runner_change_spec.md`
+
+- settles the smallest bridge-runner behavior needed to emit the V1 execution record without a broader refactor
+
+## What Is Intentionally Deferred
+
+The following work is intentionally not in this queue:
+
+- runner refactor into multiple modules
+- richer state machines
+- retry intelligence
+- dependency-aware scheduling
+- aggregated reporting
+- token accounting beyond optional fields
+- broad CLI redesign
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
new file mode 100644
index 0000000000000000000000000000000000000000..2b3a9bd724ef6c395b9312859e80c60691a7a668
--- /dev/null
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -0,0 +1,251 @@
+# V1 Execution Record Artifact
+
+## Purpose
+
+Define the minimum durable record for one prompt run so the workflow preserves:
+
+- what was executed
+- what happened during execution
+- what a human later decided about the result
+- enough failure and resource evidence to guide the next reviewed step
+
+This artifact is meant to support the current conservative workflow, not a larger automation system.
+
+## Scope
+
+V1 covers one prompt run from prompt submission through later human review.
+
+It does not define:
+
+- multi-run reporting
+- dependency graphs
+- approval automation
+- module decomposition
+- a full state machine
+
+## Source Of Truth
+
+For V1, the source of truth for one run is one markdown execution-record file in `notes/`.
+
+The record body is authoritative.
+The filename is a locator and stable identifier carrier, but the body is the place where execution facts and review fields must be read from.
+
+V1 should avoid splitting one run across multiple required files.
+
+## Stable Identity For A Run
+
+Each run must have a stable `run_id` that does not change after the file is created.
+
+V1 format:
+
+`<prompt_stem>__<started_at_utc>`
+
+Example:
+
+`005_define_execution_record_and_generate_next_prompt_queue__20260415_202557`
+
+Why this is enough for V1:
+
+- it is human-readable
+- it distinguishes repeated runs of the same prompt
+- it does not depend on later review outcome
+- it can be created cheaply by the bridge runner
+
+The filename should use the same stable identity and should not encode accepted or rejected review outcome.
+
+## Required Fields
+
+Every V1 execution record should contain these fields.
+
+### Run Identity
+
+- `run_id`
+- `prompt_file`
+- `prompt_stem`
+- `started_at_utc`
+
+### Execution Facts
+
+- `execution_status`
+- `finished_at_utc`
+- `runner`
+- `return_code`
+- `prompt_text`
+- `codex_final_output`
+
+### Review Facts
+
+- `review_status`
+- `review_summary`
+
+### Minimum Failure / Retry Linkage
+
+- `retry_of_run_id`
+
+### Minimum Resource / Cost Facts
+
+- `elapsed_seconds`
+- `final_output_char_count`
+- `stderr_char_count`
+
+## Optional Fields
+
+These fields are useful when available, but should not block V1.
+
+- `reviewed_by`
+- `reviewed_at_utc`
+- `stderr_text`
+- `failure_analysis`
+- `recommended_next_action`
+- `model_name`
+- `token_usage`
+- `attempt_index_for_prompt`
+- `workspace_notes`
+
+## Automatic Vs Manual Fields
+
+### Automatic In V1
+
+These should be captured automatically when the runner can do so cheaply and reliably:
+
+- `run_id`
+- `prompt_file`
+- `prompt_stem`
+- `started_at_utc`
+- `finished_at_utc`
+- `execution_status`
+- `runner`
+- `return_code`
+- `prompt_text`
+- `codex_final_output`
+- `elapsed_seconds`
+- `final_output_char_count`
+- `stderr_char_count`
+- `stderr_text` when present
+- `retry_of_run_id` only if the runner was explicitly given that context
+
+### Manual In V1
+
+These remain human-owned in V1:
+
+- `review_status`
+- `review_summary`
+- `reviewed_by`
+- `reviewed_at_utc`
+- `failure_analysis`
+- `recommended_next_action`
+
+This split is intentional.
+Execution capture should be automatic.
+Acceptance and interpretation should remain manual until the review loop is defined more clearly.
+
+## Execution Status Vs Review Status
+
+V1 must keep these separate.
+
+### Execution Status
+
+This records what happened when the runner invoked Codex.
+
+Allowed V1 values:
+
+- `EXECUTED`
+- `EXECUTION_FAILED`
+
+`EXECUTED` means the subprocess completed successfully enough to return code `0`.
+It does not mean the work is accepted.
+
+`EXECUTION_FAILED` means the subprocess failed or the run artifact could only be completed with failure evidence.
+
+### Review Status
+
+This records the later human judgment.
+
+Allowed V1 values:
+
+- `UNREVIEWED`
+- `ACCEPTED`
+- `REJECTED`
+
+Rules:
+
+- a new run record should start as `UNREVIEWED`
+- `EXECUTED` must not imply `ACCEPTED`
+- `EXECUTION_FAILED` does not automatically imply `REJECTED`, though rejection will often follow
+- the next prompt in a reviewed sequence should wait for explicit human review, not only process exit success
+
+## Minimum Failure-Analysis Section
+
+V1 failure analysis should stay small.
+It should exist as a short section that can be filled in manually when useful, especially for `EXECUTION_FAILED` or `REJECTED` runs.
+
+Minimum fields:
+
+- `failure_type`
+- `failure_symptom`
+- `likely_cause`
+- `recommended_next_action`
+
+Suggested `failure_type` values:
+
+- `environment`
+- `runner`
+- `prompt_scope`
+- `prompt_clarity`
+- `implementation`
+- `validation`
+- `unknown`
+
+This is enough for V1 because it supports learning from failure without requiring a larger taxonomy.
+
+## Minimum Resource / Cost Section
+
+V1 should record only lightweight fields that are cheap to capture or inspect:
+
+- `started_at_utc`
+- `finished_at_utc`
+- `elapsed_seconds`
+- `return_code`
+- `final_output_char_count`
+- `stderr_char_count`
+
+Optional later additions:
+
+- `model_name`
+- `token_usage`
+- `tool_call_count`
+- `file_change_count`
+- `estimated_review_minutes`
+
+V1 should observe resource use, not build a billing system.
+
+## Practical Record Shape
+
+A V1 markdown record should have these sections in this order:
+
+1. Header / identity
+2. Execution facts
+3. Review facts
+4. Failure analysis
+5. Resource / cost facts
+6. Prompt text
+7. Codex final output
+8. Stderr or supplemental notes
+
+This keeps one run easy to inspect in a single file.
+
+## Intentionally Deferred From V1
+
+The following should wait:
+
+- automatic acceptance or approval gating
+- dependency-aware prompt release
+- richer review states such as `BLOCKED` or `PARTIAL`
+- structured JSON sidecars or databases
+- normalized token accounting across runners
+- automatic failure classification
+- aggregated reports and dashboards
+- full retry history synthesis
+- broader refactor of the runner into multiple modules
+
+V1 is only trying to establish a durable, reviewable single-run artifact that cleanly separates execution from acceptance.

diff --git a/codex_prompts/006_define_v1_run_review_gate.md b/codex_prompts/006_define_v1_run_review_gate.md
new file mode 100644
index 0000000000000000000000000000000000000000..474c616cd105b06aae41f921b2229fcfbd38d117
--- /dev/null
+++ b/codex_prompts/006_define_v1_run_review_gate.md
@@ -0,0 +1,82 @@
+# Task: Define the V1 run-review gate for prompt execution
+
+You are working in this repository.
+
+Your task is to define the minimum review gate that must sit between:
+
+- a prompt run finishing
+- the next prompt in sequence being treated as ready
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/baby_run_prompt.py`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+
+## Goal
+
+Use `tools/Project_Design_Workflow.md` as governing.
+
+Define the smallest practical V1 review gate that preserves:
+
+- review between iterations
+- separation of execution outcome from accepted outcome
+- a conservative manual step before the next prompt is treated as ready
+
+## Important framing
+
+This is a design task.
+
+Do NOT implement the gate in code in this pass.
+Do NOT create a large state machine.
+Do NOT redesign the whole queue system.
+
+## Questions to settle
+
+Please settle these points:
+
+1. What exact event marks a run as awaiting review?
+2. What minimum human checks must happen before a run can be accepted?
+3. What review outcomes are needed in V1?
+4. What outcome should let the next prompt proceed?
+5. What outcome should stop the queue and force a new design or retry decision?
+6. What information must be written back into the run record during review?
+
+## Required output artifact
+
+Create one markdown design note at:
+
+`tools/codex/V1_Run_Review_Gate.md`
+
+The note should include:
+
+- purpose
+- scope
+- the review trigger
+- the minimum manual checklist
+- allowed V1 review outcomes
+- how the outcome affects queue progression
+- how this connects to `tools/codex/V1_Execution_Record_Artifact.md`
+- what is intentionally deferred
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Keep the gate manual and conservative.
+3. Do not implement CLI or runner changes in this pass.
+4. Do not invent a large workflow beyond the next reviewed step.
+5. Keep the decision small enough to review before any implementation prompt follows.
+
+## Success criteria
+
+This task is successful if:
+
+- the review gate is explicit and easy to apply
+- execution success is clearly separated from accepted outcome
+- queue progression rules are clear for V1
+- the result is small enough to guide the next prompt without expanding scope
diff --git a/codex_prompts/007_align_architecture_doc_to_v1_workflow.md b/codex_prompts/007_align_architecture_doc_to_v1_workflow.md
new file mode 100644
index 0000000000000000000000000000000000000000..5ac69c8f729baba517263fb8d6e834941c404b4e
--- /dev/null
+++ b/codex_prompts/007_align_architecture_doc_to_v1_workflow.md
@@ -0,0 +1,76 @@
+# Task: Align the Codex prompt architecture document to the governing workflow and V1 run artifact
+
+You are working in this repository.
+
+Your task is to bring the architecture document into explicit V1 alignment with:
+
+- the governing workflow
+- the V1 execution record
+- the V1 review gate
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Run_Review_Gate.md`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+
+## Goal
+
+Update the architecture document so it no longer skips the workflow steps that matter for V1.
+
+The revised architecture should make clear:
+
+- the V1 problem being solved
+- the V1 scope boundary
+- the minimum artifact inventory
+- the role of the execution record
+- the role of the review gate
+- the conservative implementation order
+- what remains deferred
+
+## Important framing
+
+This is still a design-alignment task.
+
+Do NOT implement the architecture in code in this pass.
+Do NOT expand the document into a giant platform roadmap.
+Do NOT rewrite the governing workflow document.
+
+## Required output artifact
+
+Revise this file directly:
+
+`tools/Codex_Prompt_Workflow_Architecture.md`
+
+The revision should add or clarify:
+
+- short problem statement
+- V1 scope
+- out-of-scope items
+- minimum artifact inventory
+- minimum viable slice
+- implementation order
+- validation and review posture
+- explicit deferred items
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Treat `tools/Project_Design_Workflow.md` as governing.
+3. Preserve the conservative tone of the architecture doc.
+4. Keep the document reusable and inspectable.
+5. Do not add speculative subsystems that are not needed for V1.
+
+## Success criteria
+
+This task is successful if:
+
+- the architecture doc clearly reflects the governing workflow sequence
+- the V1 execution record and review gate are integrated into the architecture
+- V1 boundaries and deferrals are explicit
+- the result reduces the mismatch identified in the review note without becoming a large rewrite
diff --git a/codex_prompts/008_define_minimal_bridge_runner_change_spec.md b/codex_prompts/008_define_minimal_bridge_runner_change_spec.md
new file mode 100644
index 0000000000000000000000000000000000000000..9d4d0df20196a88f4e4fc874a8c4abe51ef7e788
--- /dev/null
+++ b/codex_prompts/008_define_minimal_bridge_runner_change_spec.md
@@ -0,0 +1,80 @@
+# Task: Define the minimal bridge-runner change spec for V1 execution records
+
+You are working in this repository.
+
+Your task is to define the smallest change spec for `tools/codex/baby_run_prompt.py` so it can support the V1 execution record model without a larger refactor.
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Run_Review_Gate.md`
+- `tools/codex/baby_run_prompt.py`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+
+## Goal
+
+Produce a narrow implementation spec for the current bridge runner.
+
+The spec should define only the smallest changes needed to make the runner:
+
+- emit a V1 execution record instead of a success-implies-acceptance note
+- preserve stable run identity
+- capture the minimum automatic fields
+- leave manual review fields untouched or initialized for later completion
+
+## Important framing
+
+This is a design-spec task.
+
+Do NOT implement the changes in this pass.
+Do NOT split the runner into multiple modules in this pass.
+Do NOT introduce a large CLI redesign.
+
+## Questions to settle
+
+Please settle these points:
+
+1. What exact filename pattern should the runner write for a run record?
+2. What exact markdown sections and fields should it populate automatically?
+3. What review fields should be initialized but left manual?
+4. How should execution status be derived from the subprocess result?
+5. What minimal runtime and output metrics should be captured now?
+6. What current behaviors in `baby_run_prompt.py` should remain unchanged for V1?
+
+## Required output artifact
+
+Create one markdown spec at:
+
+`tools/codex/V1_Bridge_Runner_Change_Spec.md`
+
+The spec should include:
+
+- purpose
+- scope
+- current behavior summary
+- required V1 changes
+- non-goals
+- exact data/field mapping from runner output to execution record
+- open questions, if any, that must be resolved before implementation
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Keep the spec single-file and bridge-sized.
+3. Preserve the current runner's thin role where possible.
+4. Do not design a large future module layout here.
+5. Keep the change set small enough that it could be reviewed and implemented in one later prompt.
+
+## Success criteria
+
+This task is successful if:
+
+- the runner change scope is small and explicit
+- the spec is directly grounded in the V1 execution record and review gate
+- execution and review are no longer conflated in the target behavior
+- the spec avoids a premature refactor
diff --git a/notes/005_prompt_queue_plan__20260415_202557.md b/notes/005_prompt_queue_plan__20260415_202557.md
new file mode 100644
index 0000000000000000000000000000000000000000..9b4ec8df08d7061cf79ca0e7b6950b95c2da254c
--- /dev/null
+++ b/notes/005_prompt_queue_plan__20260415_202557.md
@@ -0,0 +1,45 @@
+# Prompt Queue Plan
+
+## Why These Prompts Were Chosen
+
+These prompts were chosen because the current misalignment is concentrated at one seam:
+the system can execute prompts, but it does not yet define the review checkpoint, the V1 architectural boundary around that checkpoint, or the smallest runner behavior needed to support it.
+
+The queue stays small on purpose.
+It settles the workflow gap before any implementation prompt is allowed to widen scope.
+
+## Why This Order Reduces Risk
+
+The order moves from policy to architecture to bridge-tooling:
+
+1. define the review gate
+2. align the architecture doc to that gate and the execution record
+3. define the smallest runner change spec that fits those decisions
+
+This reduces the risk of hard-coding the wrong note model or the wrong queue semantics into the runner before the manual review loop is settled.
+
+## What Each Prompt Is Meant To Settle
+
+`006_define_v1_run_review_gate.md`
+
+- settles the missing review stop between execution and accepted progress
+
+`007_align_architecture_doc_to_v1_workflow.md`
+
+- settles the V1 boundary, artifact inventory, validation posture, and deferrals inside the architecture document
+
+`008_define_minimal_bridge_runner_change_spec.md`
+
+- settles the smallest bridge-runner behavior needed to emit the V1 execution record without a broader refactor
+
+## What Is Intentionally Deferred
+
+The following work is intentionally not in this queue:
+
+- runner refactor into multiple modules
+- richer state machines
+- retry intelligence
+- dependency-aware scheduling
+- aggregated reporting
+- token accounting beyond optional fields
+- broad CLI redesign
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
new file mode 100644
index 0000000000000000000000000000000000000000..2b3a9bd724ef6c395b9312859e80c60691a7a668
--- /dev/null
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -0,0 +1,251 @@
+# V1 Execution Record Artifact
+
+## Purpose
+
+Define the minimum durable record for one prompt run so the workflow preserves:
+
+- what was executed
+- what happened during execution
+- what a human later decided about the result
+- enough failure and resource evidence to guide the next reviewed step
+
+This artifact is meant to support the current conservative workflow, not a larger automation system.
+
+## Scope
+
+V1 covers one prompt run from prompt submission through later human review.
+
+It does not define:
+
+- multi-run reporting
+- dependency graphs
+- approval automation
+- module decomposition
+- a full state machine
+
+## Source Of Truth
+
+For V1, the source of truth for one run is one markdown execution-record file in `notes/`.
+
+The record body is authoritative.
+The filename is a locator and stable identifier carrier, but the body is the place where execution facts and review fields must be read from.
+
+V1 should avoid splitting one run across multiple required files.
+
+## Stable Identity For A Run
+
+Each run must have a stable `run_id` that does not change after the file is created.
+
+V1 format:
+
+`<prompt_stem>__<started_at_utc>`
+
+Example:
+
+`005_define_execution_record_and_generate_next_prompt_queue__20260415_202557`
+
+Why this is enough for V1:
+
+- it is human-readable
+- it distinguishes repeated runs of the same prompt
+- it does not depend on later review outcome
+- it can be created cheaply by the bridge runner
+
+The filename should use the same stable identity and should not encode accepted or rejected review outcome.
+
+## Required Fields
+
+Every V1 execution record should contain these fields.
+
+### Run Identity
+
+- `run_id`
+- `prompt_file`
+- `prompt_stem`
+- `started_at_utc`
+
+### Execution Facts
+
+- `execution_status`
+- `finished_at_utc`
+- `runner`
+- `return_code`
+- `prompt_text`
+- `codex_final_output`
+
+### Review Facts
+
+- `review_status`
+- `review_summary`
+
+### Minimum Failure / Retry Linkage
+
+- `retry_of_run_id`
+
+### Minimum Resource / Cost Facts
+
+- `elapsed_seconds`
+- `final_output_char_count`
+- `stderr_char_count`
+
+## Optional Fields
+
+These fields are useful when available, but should not block V1.
+
+- `reviewed_by`
+- `reviewed_at_utc`
+- `stderr_text`
+- `failure_analysis`
+- `recommended_next_action`
+- `model_name`
+- `token_usage`
+- `attempt_index_for_prompt`
+- `workspace_notes`
+
+## Automatic Vs Manual Fields
+
+### Automatic In V1
+
+These should be captured automatically when the runner can do so cheaply and reliably:
+
+- `run_id`
+- `prompt_file`
+- `prompt_stem`
+- `started_at_utc`
+- `finished_at_utc`
+- `execution_status`
+- `runner`
+- `return_code`
+- `prompt_text`
+- `codex_final_output`
+- `elapsed_seconds`
+- `final_output_char_count`
+- `stderr_char_count`
+- `stderr_text` when present
+- `retry_of_run_id` only if the runner was explicitly given that context
+
+### Manual In V1
+
+These remain human-owned in V1:
+
+- `review_status`
+- `review_summary`
+- `reviewed_by`
+- `reviewed_at_utc`
+- `failure_analysis`
+- `recommended_next_action`
+
+This split is intentional.
+Execution capture should be automatic.
+Acceptance and interpretation should remain manual until the review loop is defined more clearly.
+
+## Execution Status Vs Review Status
+
+V1 must keep these separate.
+
+### Execution Status
+
+This records what happened when the runner invoked Codex.
+
+Allowed V1 values:
+
+- `EXECUTED`
+- `EXECUTION_FAILED`
+
+`EXECUTED` means the subprocess completed successfully enough to return code `0`.
+It does not mean the work is accepted.
+
+`EXECUTION_FAILED` means the subprocess failed or the run artifact could only be completed with failure evidence.
+
+### Review Status
+
+This records the later human judgment.
+
+Allowed V1 values:
+
+- `UNREVIEWED`
+- `ACCEPTED`
+- `REJECTED`
+
+Rules:
+
+- a new run record should start as `UNREVIEWED`
+- `EXECUTED` must not imply `ACCEPTED`
+- `EXECUTION_FAILED` does not automatically imply `REJECTED`, though rejection will often follow
+- the next prompt in a reviewed sequence should wait for explicit human review, not only process exit success
+
+## Minimum Failure-Analysis Section
+
+V1 failure analysis should stay small.
+It should exist as a short section that can be filled in manually when useful, especially for `EXECUTION_FAILED` or `REJECTED` runs.
+
+Minimum fields:
+
+- `failure_type`
+- `failure_symptom`
+- `likely_cause`
+- `recommended_next_action`
+
+Suggested `failure_type` values:
+
+- `environment`
+- `runner`
+- `prompt_scope`
+- `prompt_clarity`
+- `implementation`
+- `validation`
+- `unknown`
+
+This is enough for V1 because it supports learning from failure without requiring a larger taxonomy.
+
+## Minimum Resource / Cost Section
+
+V1 should record only lightweight fields that are cheap to capture or inspect:
+
+- `started_at_utc`
+- `finished_at_utc`
+- `elapsed_seconds`
+- `return_code`
+- `final_output_char_count`
+- `stderr_char_count`
+
+Optional later additions:
+
+- `model_name`
+- `token_usage`
+- `tool_call_count`
+- `file_change_count`
+- `estimated_review_minutes`
+
+V1 should observe resource use, not build a billing system.
+
+## Practical Record Shape
+
+A V1 markdown record should have these sections in this order:
+
+1. Header / identity
+2. Execution facts
+3. Review facts
+4. Failure analysis
+5. Resource / cost facts
+6. Prompt text
+7. Codex final output
+8. Stderr or supplemental notes
+
+This keeps one run easy to inspect in a single file.
+
+## Intentionally Deferred From V1
+
+The following should wait:
+
+- automatic acceptance or approval gating
+- dependency-aware prompt release
+- richer review states such as `BLOCKED` or `PARTIAL`
+- structured JSON sidecars or databases
+- normalized token accounting across runners
+- automatic failure classification
+- aggregated reports and dashboards
+- full retry history synthesis
+- broader refactor of the runner into multiple modules
+
+V1 is only trying to establish a durable, reviewable single-run artifact that cleanly separates execution from acceptance.

exec
/bin/bash -lc 'ls -1 codex_prompts | sort' in /data/git/handson-ml3
 succeeded in 0ms:
001_smoke_test_pipeline.md
002_repo_inventory_and_status.md
003_revise_Project_Deisgn_workflow_document.md
004_review_architecture_and_bridge_runner_against_workflow.md
005_define_execution_record_and_generate_next_prompt_queue.md
006_define_v1_run_review_gate.md
007_align_architecture_doc_to_v1_workflow.md
008_define_minimal_bridge_runner_change_spec.md

exec
/bin/bash -lc 'git diff -- tools/codex/V1_Execution_Record_Artifact.md codex_prompts/006_define_v1_run_review_gate.md codex_prompts/007_align_architecture_doc_to_v1_workflow.md codex_prompts/008_define_minimal_bridge_runner_change_spec.md notes/005_prompt_queue_plan__20260415_202557.md' in /data/git/handson-ml3
 succeeded in 0ms:
diff --git a/codex_prompts/006_define_v1_run_review_gate.md b/codex_prompts/006_define_v1_run_review_gate.md
new file mode 100644
index 0000000000000000000000000000000000000000..474c616cd105b06aae41f921b2229fcfbd38d117
--- /dev/null
+++ b/codex_prompts/006_define_v1_run_review_gate.md
@@ -0,0 +1,82 @@
+# Task: Define the V1 run-review gate for prompt execution
+
+You are working in this repository.
+
+Your task is to define the minimum review gate that must sit between:
+
+- a prompt run finishing
+- the next prompt in sequence being treated as ready
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/baby_run_prompt.py`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+
+## Goal
+
+Use `tools/Project_Design_Workflow.md` as governing.
+
+Define the smallest practical V1 review gate that preserves:
+
+- review between iterations
+- separation of execution outcome from accepted outcome
+- a conservative manual step before the next prompt is treated as ready
+
+## Important framing
+
+This is a design task.
+
+Do NOT implement the gate in code in this pass.
+Do NOT create a large state machine.
+Do NOT redesign the whole queue system.
+
+## Questions to settle
+
+Please settle these points:
+
+1. What exact event marks a run as awaiting review?
+2. What minimum human checks must happen before a run can be accepted?
+3. What review outcomes are needed in V1?
+4. What outcome should let the next prompt proceed?
+5. What outcome should stop the queue and force a new design or retry decision?
+6. What information must be written back into the run record during review?
+
+## Required output artifact
+
+Create one markdown design note at:
+
+`tools/codex/V1_Run_Review_Gate.md`
+
+The note should include:
+
+- purpose
+- scope
+- the review trigger
+- the minimum manual checklist
+- allowed V1 review outcomes
+- how the outcome affects queue progression
+- how this connects to `tools/codex/V1_Execution_Record_Artifact.md`
+- what is intentionally deferred
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Keep the gate manual and conservative.
+3. Do not implement CLI or runner changes in this pass.
+4. Do not invent a large workflow beyond the next reviewed step.
+5. Keep the decision small enough to review before any implementation prompt follows.
+
+## Success criteria
+
+This task is successful if:
+
+- the review gate is explicit and easy to apply
+- execution success is clearly separated from accepted outcome
+- queue progression rules are clear for V1
+- the result is small enough to guide the next prompt without expanding scope
diff --git a/codex_prompts/007_align_architecture_doc_to_v1_workflow.md b/codex_prompts/007_align_architecture_doc_to_v1_workflow.md
new file mode 100644
index 0000000000000000000000000000000000000000..5ac69c8f729baba517263fb8d6e834941c404b4e
--- /dev/null
+++ b/codex_prompts/007_align_architecture_doc_to_v1_workflow.md
@@ -0,0 +1,76 @@
+# Task: Align the Codex prompt architecture document to the governing workflow and V1 run artifact
+
+You are working in this repository.
+
+Your task is to bring the architecture document into explicit V1 alignment with:
+
+- the governing workflow
+- the V1 execution record
+- the V1 review gate
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Run_Review_Gate.md`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+
+## Goal
+
+Update the architecture document so it no longer skips the workflow steps that matter for V1.
+
+The revised architecture should make clear:
+
+- the V1 problem being solved
+- the V1 scope boundary
+- the minimum artifact inventory
+- the role of the execution record
+- the role of the review gate
+- the conservative implementation order
+- what remains deferred
+
+## Important framing
+
+This is still a design-alignment task.
+
+Do NOT implement the architecture in code in this pass.
+Do NOT expand the document into a giant platform roadmap.
+Do NOT rewrite the governing workflow document.
+
+## Required output artifact
+
+Revise this file directly:
+
+`tools/Codex_Prompt_Workflow_Architecture.md`
+
+The revision should add or clarify:
+
+- short problem statement
+- V1 scope
+- out-of-scope items
+- minimum artifact inventory
+- minimum viable slice
+- implementation order
+- validation and review posture
+- explicit deferred items
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Treat `tools/Project_Design_Workflow.md` as governing.
+3. Preserve the conservative tone of the architecture doc.
+4. Keep the document reusable and inspectable.
+5. Do not add speculative subsystems that are not needed for V1.
+
+## Success criteria
+
+This task is successful if:
+
+- the architecture doc clearly reflects the governing workflow sequence
+- the V1 execution record and review gate are integrated into the architecture
+- V1 boundaries and deferrals are explicit
+- the result reduces the mismatch identified in the review note without becoming a large rewrite
diff --git a/codex_prompts/008_define_minimal_bridge_runner_change_spec.md b/codex_prompts/008_define_minimal_bridge_runner_change_spec.md
new file mode 100644
index 0000000000000000000000000000000000000000..9d4d0df20196a88f4e4fc874a8c4abe51ef7e788
--- /dev/null
+++ b/codex_prompts/008_define_minimal_bridge_runner_change_spec.md
@@ -0,0 +1,80 @@
+# Task: Define the minimal bridge-runner change spec for V1 execution records
+
+You are working in this repository.
+
+Your task is to define the smallest change spec for `tools/codex/baby_run_prompt.py` so it can support the V1 execution record model without a larger refactor.
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Run_Review_Gate.md`
+- `tools/codex/baby_run_prompt.py`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+
+## Goal
+
+Produce a narrow implementation spec for the current bridge runner.
+
+The spec should define only the smallest changes needed to make the runner:
+
+- emit a V1 execution record instead of a success-implies-acceptance note
+- preserve stable run identity
+- capture the minimum automatic fields
+- leave manual review fields untouched or initialized for later completion
+
+## Important framing
+
+This is a design-spec task.
+
+Do NOT implement the changes in this pass.
+Do NOT split the runner into multiple modules in this pass.
+Do NOT introduce a large CLI redesign.
+
+## Questions to settle
+
+Please settle these points:
+
+1. What exact filename pattern should the runner write for a run record?
+2. What exact markdown sections and fields should it populate automatically?
+3. What review fields should be initialized but left manual?
+4. How should execution status be derived from the subprocess result?
+5. What minimal runtime and output metrics should be captured now?
+6. What current behaviors in `baby_run_prompt.py` should remain unchanged for V1?
+
+## Required output artifact
+
+Create one markdown spec at:
+
+`tools/codex/V1_Bridge_Runner_Change_Spec.md`
+
+The spec should include:
+
+- purpose
+- scope
+- current behavior summary
+- required V1 changes
+- non-goals
+- exact data/field mapping from runner output to execution record
+- open questions, if any, that must be resolved before implementation
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Keep the spec single-file and bridge-sized.
+3. Preserve the current runner's thin role where possible.
+4. Do not design a large future module layout here.
+5. Keep the change set small enough that it could be reviewed and implemented in one later prompt.
+
+## Success criteria
+
+This task is successful if:
+
+- the runner change scope is small and explicit
+- the spec is directly grounded in the V1 execution record and review gate
+- execution and review are no longer conflated in the target behavior
+- the spec avoids a premature refactor
diff --git a/notes/005_prompt_queue_plan__20260415_202557.md b/notes/005_prompt_queue_plan__20260415_202557.md
new file mode 100644
index 0000000000000000000000000000000000000000..9b4ec8df08d7061cf79ca0e7b6950b95c2da254c
--- /dev/null
+++ b/notes/005_prompt_queue_plan__20260415_202557.md
@@ -0,0 +1,45 @@
+# Prompt Queue Plan
+
+## Why These Prompts Were Chosen
+
+These prompts were chosen because the current misalignment is concentrated at one seam:
+the system can execute prompts, but it does not yet define the review checkpoint, the V1 architectural boundary around that checkpoint, or the smallest runner behavior needed to support it.
+
+The queue stays small on purpose.
+It settles the workflow gap before any implementation prompt is allowed to widen scope.
+
+## Why This Order Reduces Risk
+
+The order moves from policy to architecture to bridge-tooling:
+
+1. define the review gate
+2. align the architecture doc to that gate and the execution record
+3. define the smallest runner change spec that fits those decisions
+
+This reduces the risk of hard-coding the wrong note model or the wrong queue semantics into the runner before the manual review loop is settled.
+
+## What Each Prompt Is Meant To Settle
+
+`006_define_v1_run_review_gate.md`
+
+- settles the missing review stop between execution and accepted progress
+
+`007_align_architecture_doc_to_v1_workflow.md`
+
+- settles the V1 boundary, artifact inventory, validation posture, and deferrals inside the architecture document
+
+`008_define_minimal_bridge_runner_change_spec.md`
+
+- settles the smallest bridge-runner behavior needed to emit the V1 execution record without a broader refactor
+
+## What Is Intentionally Deferred
+
+The following work is intentionally not in this queue:
+
+- runner refactor into multiple modules
+- richer state machines
+- retry intelligence
+- dependency-aware scheduling
+- aggregated reporting
+- token accounting beyond optional fields
+- broad CLI redesign
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
new file mode 100644
index 0000000000000000000000000000000000000000..2b3a9bd724ef6c395b9312859e80c60691a7a668
--- /dev/null
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -0,0 +1,251 @@
+# V1 Execution Record Artifact
+
+## Purpose
+
+Define the minimum durable record for one prompt run so the workflow preserves:
+
+- what was executed
+- what happened during execution
+- what a human later decided about the result
+- enough failure and resource evidence to guide the next reviewed step
+
+This artifact is meant to support the current conservative workflow, not a larger automation system.
+
+## Scope
+
+V1 covers one prompt run from prompt submission through later human review.
+
+It does not define:
+
+- multi-run reporting
+- dependency graphs
+- approval automation
+- module decomposition
+- a full state machine
+
+## Source Of Truth
+
+For V1, the source of truth for one run is one markdown execution-record file in `notes/`.
+
+The record body is authoritative.
+The filename is a locator and stable identifier carrier, but the body is the place where execution facts and review fields must be read from.
+
+V1 should avoid splitting one run across multiple required files.
+
+## Stable Identity For A Run
+
+Each run must have a stable `run_id` that does not change after the file is created.
+
+V1 format:
+
+`<prompt_stem>__<started_at_utc>`
+
+Example:
+
+`005_define_execution_record_and_generate_next_prompt_queue__20260415_202557`
+
+Why this is enough for V1:
+
+- it is human-readable
+- it distinguishes repeated runs of the same prompt
+- it does not depend on later review outcome
+- it can be created cheaply by the bridge runner
+
+The filename should use the same stable identity and should not encode accepted or rejected review outcome.
+
+## Required Fields
+
+Every V1 execution record should contain these fields.
+
+### Run Identity
+
+- `run_id`
+- `prompt_file`
+- `prompt_stem`
+- `started_at_utc`
+
+### Execution Facts
+
+- `execution_status`
+- `finished_at_utc`
+- `runner`
+- `return_code`
+- `prompt_text`
+- `codex_final_output`
+
+### Review Facts
+
+- `review_status`
+- `review_summary`
+
+### Minimum Failure / Retry Linkage
+
+- `retry_of_run_id`
+
+### Minimum Resource / Cost Facts
+
+- `elapsed_seconds`
+- `final_output_char_count`
+- `stderr_char_count`
+
+## Optional Fields
+
+These fields are useful when available, but should not block V1.
+
+- `reviewed_by`
+- `reviewed_at_utc`
+- `stderr_text`
+- `failure_analysis`
+- `recommended_next_action`
+- `model_name`
+- `token_usage`
+- `attempt_index_for_prompt`
+- `workspace_notes`
+
+## Automatic Vs Manual Fields
+
+### Automatic In V1
+
+These should be captured automatically when the runner can do so cheaply and reliably:
+
+- `run_id`
+- `prompt_file`
+- `prompt_stem`
+- `started_at_utc`
+- `finished_at_utc`
+- `execution_status`
+- `runner`
+- `return_code`
+- `prompt_text`
+- `codex_final_output`
+- `elapsed_seconds`
+- `final_output_char_count`
+- `stderr_char_count`
+- `stderr_text` when present
+- `retry_of_run_id` only if the runner was explicitly given that context
+
+### Manual In V1
+
+These remain human-owned in V1:
+
+- `review_status`
+- `review_summary`
+- `reviewed_by`
+- `reviewed_at_utc`
+- `failure_analysis`
+- `recommended_next_action`
+
+This split is intentional.
+Execution capture should be automatic.
+Acceptance and interpretation should remain manual until the review loop is defined more clearly.
+
+## Execution Status Vs Review Status
+
+V1 must keep these separate.
+
+### Execution Status
+
+This records what happened when the runner invoked Codex.
+
+Allowed V1 values:
+
+- `EXECUTED`
+- `EXECUTION_FAILED`
+
+`EXECUTED` means the subprocess completed successfully enough to return code `0`.
+It does not mean the work is accepted.
+
+`EXECUTION_FAILED` means the subprocess failed or the run artifact could only be completed with failure evidence.
+
+### Review Status
+
+This records the later human judgment.
+
+Allowed V1 values:
+
+- `UNREVIEWED`
+- `ACCEPTED`
+- `REJECTED`
+
+Rules:
+
+- a new run record should start as `UNREVIEWED`
+- `EXECUTED` must not imply `ACCEPTED`
+- `EXECUTION_FAILED` does not automatically imply `REJECTED`, though rejection will often follow
+- the next prompt in a reviewed sequence should wait for explicit human review, not only process exit success
+
+## Minimum Failure-Analysis Section
+
+V1 failure analysis should stay small.
+It should exist as a short section that can be filled in manually when useful, especially for `EXECUTION_FAILED` or `REJECTED` runs.
+
+Minimum fields:
+
+- `failure_type`
+- `failure_symptom`
+- `likely_cause`
+- `recommended_next_action`
+
+Suggested `failure_type` values:
+
+- `environment`
+- `runner`
+- `prompt_scope`
+- `prompt_clarity`
+- `implementation`
+- `validation`
+- `unknown`
+
+This is enough for V1 because it supports learning from failure without requiring a larger taxonomy.
+
+## Minimum Resource / Cost Section
+
+V1 should record only lightweight fields that are cheap to capture or inspect:
+
+- `started_at_utc`
+- `finished_at_utc`
+- `elapsed_seconds`
+- `return_code`
+- `final_output_char_count`
+- `stderr_char_count`
+
+Optional later additions:
+
+- `model_name`
+- `token_usage`
+- `tool_call_count`
+- `file_change_count`
+- `estimated_review_minutes`
+
+V1 should observe resource use, not build a billing system.
+
+## Practical Record Shape
+
+A V1 markdown record should have these sections in this order:
+
+1. Header / identity
+2. Execution facts
+3. Review facts
+4. Failure analysis
+5. Resource / cost facts
+6. Prompt text
+7. Codex final output
+8. Stderr or supplemental notes
+
+This keeps one run easy to inspect in a single file.
+
+## Intentionally Deferred From V1
+
+The following should wait:
+
+- automatic acceptance or approval gating
+- dependency-aware prompt release
+- richer review states such as `BLOCKED` or `PARTIAL`
+- structured JSON sidecars or databases
+- normalized token accounting across runners
+- automatic failure classification
+- aggregated reports and dashboards
+- full retry history synthesis
+- broader refactor of the runner into multiple modules
+
+V1 is only trying to establish a durable, reviewable single-run artifact that cleanly separates execution from acceptance.

codex
Created the V1 execution-record design at [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:1). It makes one `notes/` markdown record the V1 source of truth for a run, gives each run a stable `run_id`, separates `execution_status` from `review_status`, defines the minimum manual failure-analysis and lightweight resource fields, and explicitly defers broader state machines, reporting, and refactors.

I also added the next bounded prompt queue:
[006_define_v1_run_review_gate.md](/data/git/handson-ml3/codex_prompts/006_define_v1_run_review_gate.md:1),
[007_align_architecture_doc_to_v1_workflow.md](/data/git/handson-ml3/codex_prompts/007_align_architecture_doc_to_v1_workflow.md:1),
and [008_define_minimal_bridge_runner_change_spec.md](/data/git/handson-ml3/codex_prompts/008_define_minimal_bridge_runner_change_spec.md:1).
The companion sequencing note is [005_prompt_queue_plan__20260415_202557.md](/data/git/handson-ml3/notes/005_prompt_queue_plan__20260415_202557.md:1).

Verification was by direct file inspection; no implementation or runner refactor was performed.
diff --git a/codex_prompts/006_define_v1_run_review_gate.md b/codex_prompts/006_define_v1_run_review_gate.md
new file mode 100644
index 0000000000000000000000000000000000000000..474c616cd105b06aae41f921b2229fcfbd38d117
--- /dev/null
+++ b/codex_prompts/006_define_v1_run_review_gate.md
@@ -0,0 +1,82 @@
+# Task: Define the V1 run-review gate for prompt execution
+
+You are working in this repository.
+
+Your task is to define the minimum review gate that must sit between:
+
+- a prompt run finishing
+- the next prompt in sequence being treated as ready
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/baby_run_prompt.py`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+
+## Goal
+
+Use `tools/Project_Design_Workflow.md` as governing.
+
+Define the smallest practical V1 review gate that preserves:
+
+- review between iterations
+- separation of execution outcome from accepted outcome
+- a conservative manual step before the next prompt is treated as ready
+
+## Important framing
+
+This is a design task.
+
+Do NOT implement the gate in code in this pass.
+Do NOT create a large state machine.
+Do NOT redesign the whole queue system.
+
+## Questions to settle
+
+Please settle these points:
+
+1. What exact event marks a run as awaiting review?
+2. What minimum human checks must happen before a run can be accepted?
+3. What review outcomes are needed in V1?
+4. What outcome should let the next prompt proceed?
+5. What outcome should stop the queue and force a new design or retry decision?
+6. What information must be written back into the run record during review?
+
+## Required output artifact
+
+Create one markdown design note at:
+
+`tools/codex/V1_Run_Review_Gate.md`
+
+The note should include:
+
+- purpose
+- scope
+- the review trigger
+- the minimum manual checklist
+- allowed V1 review outcomes
+- how the outcome affects queue progression
+- how this connects to `tools/codex/V1_Execution_Record_Artifact.md`
+- what is intentionally deferred
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Keep the gate manual and conservative.
+3. Do not implement CLI or runner changes in this pass.
+4. Do not invent a large workflow beyond the next reviewed step.
+5. Keep the decision small enough to review before any implementation prompt follows.
+
+## Success criteria
+
+This task is successful if:
+
+- the review gate is explicit and easy to apply
+- execution success is clearly separated from accepted outcome
+- queue progression rules are clear for V1
+- the result is small enough to guide the next prompt without expanding scope
diff --git a/codex_prompts/007_align_architecture_doc_to_v1_workflow.md b/codex_prompts/007_align_architecture_doc_to_v1_workflow.md
new file mode 100644
index 0000000000000000000000000000000000000000..5ac69c8f729baba517263fb8d6e834941c404b4e
--- /dev/null
+++ b/codex_prompts/007_align_architecture_doc_to_v1_workflow.md
@@ -0,0 +1,76 @@
+# Task: Align the Codex prompt architecture document to the governing workflow and V1 run artifact
+
+You are working in this repository.
+
+Your task is to bring the architecture document into explicit V1 alignment with:
+
+- the governing workflow
+- the V1 execution record
+- the V1 review gate
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Run_Review_Gate.md`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+
+## Goal
+
+Update the architecture document so it no longer skips the workflow steps that matter for V1.
+
+The revised architecture should make clear:
+
+- the V1 problem being solved
+- the V1 scope boundary
+- the minimum artifact inventory
+- the role of the execution record
+- the role of the review gate
+- the conservative implementation order
+- what remains deferred
+
+## Important framing
+
+This is still a design-alignment task.
+
+Do NOT implement the architecture in code in this pass.
+Do NOT expand the document into a giant platform roadmap.
+Do NOT rewrite the governing workflow document.
+
+## Required output artifact
+
+Revise this file directly:
+
+`tools/Codex_Prompt_Workflow_Architecture.md`
+
+The revision should add or clarify:
+
+- short problem statement
+- V1 scope
+- out-of-scope items
+- minimum artifact inventory
+- minimum viable slice
+- implementation order
+- validation and review posture
+- explicit deferred items
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Treat `tools/Project_Design_Workflow.md` as governing.
+3. Preserve the conservative tone of the architecture doc.
+4. Keep the document reusable and inspectable.
+5. Do not add speculative subsystems that are not needed for V1.
+
+## Success criteria
+
+This task is successful if:
+
+- the architecture doc clearly reflects the governing workflow sequence
+- the V1 execution record and review gate are integrated into the architecture
+- V1 boundaries and deferrals are explicit
+- the result reduces the mismatch identified in the review note without becoming a large rewrite
diff --git a/codex_prompts/008_define_minimal_bridge_runner_change_spec.md b/codex_prompts/008_define_minimal_bridge_runner_change_spec.md
new file mode 100644
index 0000000000000000000000000000000000000000..9d4d0df20196a88f4e4fc874a8c4abe51ef7e788
--- /dev/null
+++ b/codex_prompts/008_define_minimal_bridge_runner_change_spec.md
@@ -0,0 +1,80 @@
+# Task: Define the minimal bridge-runner change spec for V1 execution records
+
+You are working in this repository.
+
+Your task is to define the smallest change spec for `tools/codex/baby_run_prompt.py` so it can support the V1 execution record model without a larger refactor.
+
+## Files to inspect
+
+Read these exact files:
+
+- `tools/Project_Design_Workflow.md`
+- `tools/Codex_Prompt_Workflow_Architecture.md`
+- `tools/codex/V1_Execution_Record_Artifact.md`
+- `tools/codex/V1_Run_Review_Gate.md`
+- `tools/codex/baby_run_prompt.py`
+- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
+- `notes/005_prompt_queue_plan__20260415_202557.md`
+
+## Goal
+
+Produce a narrow implementation spec for the current bridge runner.
+
+The spec should define only the smallest changes needed to make the runner:
+
+- emit a V1 execution record instead of a success-implies-acceptance note
+- preserve stable run identity
+- capture the minimum automatic fields
+- leave manual review fields untouched or initialized for later completion
+
+## Important framing
+
+This is a design-spec task.
+
+Do NOT implement the changes in this pass.
+Do NOT split the runner into multiple modules in this pass.
+Do NOT introduce a large CLI redesign.
+
+## Questions to settle
+
+Please settle these points:
+
+1. What exact filename pattern should the runner write for a run record?
+2. What exact markdown sections and fields should it populate automatically?
+3. What review fields should be initialized but left manual?
+4. How should execution status be derived from the subprocess result?
+5. What minimal runtime and output metrics should be captured now?
+6. What current behaviors in `baby_run_prompt.py` should remain unchanged for V1?
+
+## Required output artifact
+
+Create one markdown spec at:
+
+`tools/codex/V1_Bridge_Runner_Change_Spec.md`
+
+The spec should include:
+
+- purpose
+- scope
+- current behavior summary
+- required V1 changes
+- non-goals
+- exact data/field mapping from runner output to execution record
+- open questions, if any, that must be resolved before implementation
+
+## Constraints
+
+1. Use the exact file paths listed above.
+2. Keep the spec single-file and bridge-sized.
+3. Preserve the current runner's thin role where possible.
+4. Do not design a large future module layout here.
+5. Keep the change set small enough that it could be reviewed and implemented in one later prompt.
+
+## Success criteria
+
+This task is successful if:
+
+- the runner change scope is small and explicit
+- the spec is directly grounded in the V1 execution record and review gate
+- execution and review are no longer conflated in the target behavior
+- the spec avoids a premature refactor
diff --git a/notes/005_prompt_queue_plan__20260415_202557.md b/notes/005_prompt_queue_plan__20260415_202557.md
new file mode 100644
index 0000000000000000000000000000000000000000..9b4ec8df08d7061cf79ca0e7b6950b95c2da254c
--- /dev/null
+++ b/notes/005_prompt_queue_plan__20260415_202557.md
@@ -0,0 +1,45 @@
+# Prompt Queue Plan
+
+## Why These Prompts Were Chosen
+
+These prompts were chosen because the current misalignment is concentrated at one seam:
+the system can execute prompts, but it does not yet define the review checkpoint, the V1 architectural boundary around that checkpoint, or the smallest runner behavior needed to support it.
+
+The queue stays small on purpose.
+It settles the workflow gap before any implementation prompt is allowed to widen scope.
+
+## Why This Order Reduces Risk
+
+The order moves from policy to architecture to bridge-tooling:
+
+1. define the review gate
+2. align the architecture doc to that gate and the execution record
+3. define the smallest runner change spec that fits those decisions
+
+This reduces the risk of hard-coding the wrong note model or the wrong queue semantics into the runner before the manual review loop is settled.
+
+## What Each Prompt Is Meant To Settle
+
+`006_define_v1_run_review_gate.md`
+
+- settles the missing review stop between execution and accepted progress
+
+`007_align_architecture_doc_to_v1_workflow.md`
+
+- settles the V1 boundary, artifact inventory, validation posture, and deferrals inside the architecture document
+
+`008_define_minimal_bridge_runner_change_spec.md`
+
+- settles the smallest bridge-runner behavior needed to emit the V1 execution record without a broader refactor
+
+## What Is Intentionally Deferred
+
+The following work is intentionally not in this queue:
+
+- runner refactor into multiple modules
+- richer state machines
+- retry intelligence
+- dependency-aware scheduling
+- aggregated reporting
+- token accounting beyond optional fields
+- broad CLI redesign
diff --git a/tools/codex/V1_Execution_Record_Artifact.md b/tools/codex/V1_Execution_Record_Artifact.md
new file mode 100644
index 0000000000000000000000000000000000000000..2b3a9bd724ef6c395b9312859e80c60691a7a668
--- /dev/null
+++ b/tools/codex/V1_Execution_Record_Artifact.md
@@ -0,0 +1,251 @@
+# V1 Execution Record Artifact
+
+## Purpose
+
+Define the minimum durable record for one prompt run so the workflow preserves:
+
+- what was executed
+- what happened during execution
+- what a human later decided about the result
+- enough failure and resource evidence to guide the next reviewed step
+
+This artifact is meant to support the current conservative workflow, not a larger automation system.
+
+## Scope
+
+V1 covers one prompt run from prompt submission through later human review.
+
+It does not define:
+
+- multi-run reporting
+- dependency graphs
+- approval automation
+- module decomposition
+- a full state machine
+
+## Source Of Truth
+
+For V1, the source of truth for one run is one markdown execution-record file in `notes/`.
+
+The record body is authoritative.
+The filename is a locator and stable identifier carrier, but the body is the place where execution facts and review fields must be read from.
+
+V1 should avoid splitting one run across multiple required files.
+
+## Stable Identity For A Run
+
+Each run must have a stable `run_id` that does not change after the file is created.
+
+V1 format:
+
+`<prompt_stem>__<started_at_utc>`
+
+Example:
+
+`005_define_execution_record_and_generate_next_prompt_queue__20260415_202557`
+
+Why this is enough for V1:
+
+- it is human-readable
+- it distinguishes repeated runs of the same prompt
+- it does not depend on later review outcome
+- it can be created cheaply by the bridge runner
+
+The filename should use the same stable identity and should not encode accepted or rejected review outcome.
+
+## Required Fields
+
+Every V1 execution record should contain these fields.
+
+### Run Identity
+
+- `run_id`
+- `prompt_file`
+- `prompt_stem`
+- `started_at_utc`
+
+### Execution Facts
+
+- `execution_status`
+- `finished_at_utc`
+- `runner`
+- `return_code`
+- `prompt_text`
+- `codex_final_output`
+
+### Review Facts
+
+- `review_status`
+- `review_summary`
+
+### Minimum Failure / Retry Linkage
+
+- `retry_of_run_id`
+
+### Minimum Resource / Cost Facts
+
+- `elapsed_seconds`
+- `final_output_char_count`
+- `stderr_char_count`
+
+## Optional Fields
+
+These fields are useful when available, but should not block V1.
+
+- `reviewed_by`
+- `reviewed_at_utc`
+- `stderr_text`
+- `failure_analysis`
+- `recommended_next_action`
+- `model_name`
+- `token_usage`
+- `attempt_index_for_prompt`
+- `workspace_notes`
+
+## Automatic Vs Manual Fields
+
+### Automatic In V1
+
+These should be captured automatically when the runner can do so cheaply and reliably:
+
+- `run_id`
+- `prompt_file`
+- `prompt_stem`
+- `started_at_utc`
+- `finished_at_utc`
+- `execution_status`
+- `runner`
+- `return_code`
+- `prompt_text`
+- `codex_final_output`
+- `elapsed_seconds`
+- `final_output_char_count`
+- `stderr_char_count`
+- `stderr_text` when present
+- `retry_of_run_id` only if the runner was explicitly given that context
+
+### Manual In V1
+
+These remain human-owned in V1:
+
+- `review_status`
+- `review_summary`
+- `reviewed_by`
+- `reviewed_at_utc`
+- `failure_analysis`
+- `recommended_next_action`
+
+This split is intentional.
+Execution capture should be automatic.
+Acceptance and interpretation should remain manual until the review loop is defined more clearly.
+
+## Execution Status Vs Review Status
+
+V1 must keep these separate.
+
+### Execution Status
+
+This records what happened when the runner invoked Codex.
+
+Allowed V1 values:
+
+- `EXECUTED`
+- `EXECUTION_FAILED`
+
+`EXECUTED` means the subprocess completed successfully enough to return code `0`.
+It does not mean the work is accepted.
+
+`EXECUTION_FAILED` means the subprocess failed or the run artifact could only be completed with failure evidence.
+
+### Review Status
+
+This records the later human judgment.
+
+Allowed V1 values:
+
+- `UNREVIEWED`
+- `ACCEPTED`
+- `REJECTED`
+
+Rules:
+
+- a new run record should start as `UNREVIEWED`
+- `EXECUTED` must not imply `ACCEPTED`
+- `EXECUTION_FAILED` does not automatically imply `REJECTED`, though rejection will often follow
+- the next prompt in a reviewed sequence should wait for explicit human review, not only process exit success
+
+## Minimum Failure-Analysis Section
+
+V1 failure analysis should stay small.
+It should exist as a short section that can be filled in manually when useful, especially for `EXECUTION_FAILED` or `REJECTED` runs.
+
+Minimum fields:
+
+- `failure_type`
+- `failure_symptom`
+- `likely_cause`
+- `recommended_next_action`
+
+Suggested `failure_type` values:
+
+- `environment`
+- `runner`
+- `prompt_scope`
+- `prompt_clarity`
+- `implementation`
+- `validation`
+- `unknown`
+
+This is enough for V1 because it supports learning from failure without requiring a larger taxonomy.
+
+## Minimum Resource / Cost Section
+
+V1 should record only lightweight fields that are cheap to capture or inspect:
+
+- `started_at_utc`
+- `finished_at_utc`
+- `elapsed_seconds`
+- `return_code`
+- `final_output_char_count`
+- `stderr_char_count`
+
+Optional later additions:
+
+- `model_name`
+- `token_usage`
+- `tool_call_count`
+- `file_change_count`
+- `estimated_review_minutes`
+
+V1 should observe resource use, not build a billing system.
+
+## Practical Record Shape
+
+A V1 markdown record should have these sections in this order:
+
+1. Header / identity
+2. Execution facts
+3. Review facts
+4. Failure analysis
+5. Resource / cost facts
+6. Prompt text
+7. Codex final output
+8. Stderr or supplemental notes
+
+This keeps one run easy to inspect in a single file.
+
+## Intentionally Deferred From V1
+
+The following should wait:
+
+- automatic acceptance or approval gating
+- dependency-aware prompt release
+- richer review states such as `BLOCKED` or `PARTIAL`
+- structured JSON sidecars or databases
+- normalized token accounting across runners
+- automatic failure classification
+- aggregated reports and dashboards
+- full retry history synthesis
+- broader refactor of the runner into multiple modules
+
+V1 is only trying to establish a durable, reviewable single-run artifact that cleanly separates execution from acceptance.

tokens used
35,502
```
