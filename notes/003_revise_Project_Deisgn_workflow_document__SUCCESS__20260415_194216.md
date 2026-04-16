# 003_revise_Project_Deisgn_workflow_document - SUCCESS

- Prompt file: `003_revise_Project_Deisgn_workflow_document.md`
- Timestamp (UTC): `20260415_194216`
- Status: `SUCCESS`

## Original Prompt

```md
# Task: Revise Project_Design_Workflow.md to remain project-agnostic and include failure-analysis discipline

You are working in this repository.

Your task is to revise the document `Project_Design_Workflow.md`.

## Goal

Produce an improved version of `Project_Design_Workflow.md` that remains:

- project-agnostic
- reusable across many repositories and workflows
- grounded in careful design-before-build discipline
- compatible with a prompt-by-prompt execution style

The revised document must NOT become narrowly tailored to the current Codex prompt workflow project.
It must stay general enough to guide many different projects.

## Important design intent

The workflow document should describe a reusable way to move from:

- idea
- architecture
- design clarification
- bounded implementation
- validation
- refinement

It should not depend on the current maturity level of any one project.

In other words:

- do not make the workflow care whether a project is early, midstream, or late
- do not make it depend on specific current tools being present
- do not overfit it to this repo

## New ideas that MUST be incorporated

Please revise the workflow so it clearly supports these general principles:

### 1. Bounded iterative execution
A project may move forward one carefully scoped chunk at a time.
The workflow should support iterative cycles of:

- clarify next chunk
- define bounded task
- execute
- inspect results
- refine plan
- repeat

This should be expressed in a project-agnostic way.

### 2. Bridge tooling is allowed
The workflow should acknowledge that a project may use temporary or bridge tooling during execution, as long as that tooling remains:

- thin
- inspectable
- subordinate to the design
- not a substitute for good architecture

Do not mention any specific tool names unless needed as a light example.
Keep the principle general.

### 3. Durable local history matters
The workflow should acknowledge that local artifacts, notes, logs, outputs, and reports can serve as durable project memory.
This should be expressed as a reusable principle, not tied to this specific project.

### 4. Review between iterations
The workflow should reinforce that each bounded execution step should be reviewed before the next one is issued.

### 5. Failure should trigger analysis
Add a reusable principle and likely a dedicated section explaining that after a failed execution step, the workflow should encourage a second-pass review or report that asks questions such as:

- Was the task too large?
- Was the task poorly decomposed?
- Were success criteria unclear?
- Was the failure caused by tooling/infrastructure rather than task difficulty?
- Would a smaller or differently framed prompt/task have worked better?

This failure-analysis concept should be included as part of the workflow, not as an afterthought.

### 6. Resource/cost awareness
Add a reusable principle that project minutes, notes, or reports may record metrics such as:

- token usage
- execution size
- elapsed time
- repeated retries
- failure frequency

The document should encourage observing whether larger or more expensive tasks correlate with failure or lower-quality results.

This should remain general and not assume a single specific model or platform.

## Constraints

1. Keep the tone consistent with the current document:
   - clear
   - calm
   - practical
   - thoughtful
   - not flashy

2. Preserve the strengths of the current document:
   - phased workflow
   - emphasis on boundaries
   - emphasis on validation
   - emphasis on refinement
   - reusable decomposition pattern

3. Do NOT turn this into a tool-specific runbook.
4. Do NOT overfit it to Codex.
5. Do NOT make it depend on the current state of this project.
6. Do NOT remove useful existing structure unless replacement is clearly better.

## Preferred output

Please do the following:

1. Read the existing `Project_Design_Workflow.md`
2. Revise it directly into a stronger replacement
3. Preserve the filename
4. Make the revised document feel like a better version of the current one, not a completely different manifesto

## Also create a short companion note

After revising the workflow doc, create a short markdown note in `notes/` with a timestamped filename that summarizes:

- what changed
- why those changes were made
- what new reusable principles were added
- any tradeoffs or open questions that remain

The note should be concise and practical.

## Success criteria

The task is successful if:

- `Project_Design_Workflow.md` remains reusable across many projects
- it now clearly supports bounded iterative execution
- it includes a clear failure-analysis concept
- it includes resource/cost-awareness as a reusable practice
- it still feels grounded, conservative, and design-centered
- it does not become narrowly tailored to this specific repo
```

## Codex Output

Revised [Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md) into a stronger project-agnostic version. The update keeps the phased design-before-build structure, but removes the repo-specific application section and adds explicit guidance for bounded iterative execution, review between iterations, bridge tooling, durable local history, failure analysis, and resource/cost awareness.

I also added a short companion note at [003_project_design_workflow_revision__SUCCESS__20260415_144244.md](/data/git/handson-ml3/notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md) summarizing what changed, why, the new reusable principles, and the remaining tradeoff that broader reuse means less project-specific concreteness.

No code tests were needed for this task.

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
session id: 019d92aa-7447-7302-b4d3-ee30d88317dd
--------
user
# Task: Revise Project_Design_Workflow.md to remain project-agnostic and include failure-analysis discipline

You are working in this repository.

Your task is to revise the document `Project_Design_Workflow.md`.

## Goal

Produce an improved version of `Project_Design_Workflow.md` that remains:

- project-agnostic
- reusable across many repositories and workflows
- grounded in careful design-before-build discipline
- compatible with a prompt-by-prompt execution style

The revised document must NOT become narrowly tailored to the current Codex prompt workflow project.
It must stay general enough to guide many different projects.

## Important design intent

The workflow document should describe a reusable way to move from:

- idea
- architecture
- design clarification
- bounded implementation
- validation
- refinement

It should not depend on the current maturity level of any one project.

In other words:

- do not make the workflow care whether a project is early, midstream, or late
- do not make it depend on specific current tools being present
- do not overfit it to this repo

## New ideas that MUST be incorporated

Please revise the workflow so it clearly supports these general principles:

### 1. Bounded iterative execution
A project may move forward one carefully scoped chunk at a time.
The workflow should support iterative cycles of:

- clarify next chunk
- define bounded task
- execute
- inspect results
- refine plan
- repeat

This should be expressed in a project-agnostic way.

### 2. Bridge tooling is allowed
The workflow should acknowledge that a project may use temporary or bridge tooling during execution, as long as that tooling remains:

- thin
- inspectable
- subordinate to the design
- not a substitute for good architecture

Do not mention any specific tool names unless needed as a light example.
Keep the principle general.

### 3. Durable local history matters
The workflow should acknowledge that local artifacts, notes, logs, outputs, and reports can serve as durable project memory.
This should be expressed as a reusable principle, not tied to this specific project.

### 4. Review between iterations
The workflow should reinforce that each bounded execution step should be reviewed before the next one is issued.

### 5. Failure should trigger analysis
Add a reusable principle and likely a dedicated section explaining that after a failed execution step, the workflow should encourage a second-pass review or report that asks questions such as:

- Was the task too large?
- Was the task poorly decomposed?
- Were success criteria unclear?
- Was the failure caused by tooling/infrastructure rather than task difficulty?
- Would a smaller or differently framed prompt/task have worked better?

This failure-analysis concept should be included as part of the workflow, not as an afterthought.

### 6. Resource/cost awareness
Add a reusable principle that project minutes, notes, or reports may record metrics such as:

- token usage
- execution size
- elapsed time
- repeated retries
- failure frequency

The document should encourage observing whether larger or more expensive tasks correlate with failure or lower-quality results.

This should remain general and not assume a single specific model or platform.

## Constraints

1. Keep the tone consistent with the current document:
   - clear
   - calm
   - practical
   - thoughtful
   - not flashy

2. Preserve the strengths of the current document:
   - phased workflow
   - emphasis on boundaries
   - emphasis on validation
   - emphasis on refinement
   - reusable decomposition pattern

3. Do NOT turn this into a tool-specific runbook.
4. Do NOT overfit it to Codex.
5. Do NOT make it depend on the current state of this project.
6. Do NOT remove useful existing structure unless replacement is clearly better.

## Preferred output

Please do the following:

1. Read the existing `Project_Design_Workflow.md`
2. Revise it directly into a stronger replacement
3. Preserve the filename
4. Make the revised document feel like a better version of the current one, not a completely different manifesto

## Also create a short companion note

After revising the workflow doc, create a short markdown note in `notes/` with a timestamped filename that summarizes:

- what changed
- why those changes were made
- what new reusable principles were added
- any tradeoffs or open questions that remain

The note should be concise and practical.

## Success criteria

The task is successful if:

- `Project_Design_Workflow.md` remains reusable across many projects
- it now clearly supports bounded iterative execution
- it includes a clear failure-analysis concept
- it includes resource/cost-awareness as a reusable practice
- it still feels grounded, conservative, and design-centered
- it does not become narrowly tailored to this specific repo

codex
I’m reviewing the existing workflow doc and surrounding notes first so I can preserve its structure while tightening the general guidance and adding the new failure-analysis and iteration principles.
exec
/bin/bash -lc "sed -n '1,260p' Project_Design_Workflow.md" in /data/git/handson-ml3
 exited 2 in 0ms:
sed: can't read Project_Design_Workflow.md: No such file or directory

exec
/bin/bash -lc 'rg --files notes | tail -n 20' in /data/git/handson-ml3
 succeeded in 0ms:
notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md
notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md
notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md

codex
The workflow file is not at repo root. I’m locating it and checking the existing notes naming pattern before editing so the replacement lands in the right place and the companion note matches repo conventions.
exec
/bin/bash -lc 'find notes -maxdepth 1 -type f | sort' in /data/git/handson-ml3
 succeeded in 0ms:
notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md
notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md
notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md

exec
/bin/bash -lc "rg --files | rg '("'^|/)Project_Design_Workflow'"\\.md"'$'"'" in /data/git/handson-ml3
 succeeded in 104ms:
tools/Project_Design_Workflow.md

codex
I found the document at `tools/Project_Design_Workflow.md`. Next I’m reading it in full, then I’ll revise it in place and add a short note under `notes/` that captures the reasoning and new reusable principles.
exec
/bin/bash -lc "sed -n '1,320p' tools/Project_Design_Workflow.md" in /data/git/handson-ml3
 succeeded in 0ms:
# Project Design Workflow

## Goal

Define a repeatable workflow for turning an idea or architecture document into a clean, working system without skipping the thinking that keeps the build safe, clear, and reusable.

This document is meant to sit beside architecture documents and help answer a practical question:

**What are the steps between “this design makes sense” and “the thing works”?**

---

## Why This Exists

Sometimes a project feels good at the architecture level but still feels fuzzy at the build level.

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

This workflow bridges that gap.

---

## Relationship to the Thought Garden

This workflow does **not** replace the Thought Garden.
It fits inside it.

A useful mapping looks like this:

### 1. Conversation

What are we trying to do?
What hurts right now?
What would “better” feel like?

### 2. Goals

What must become true for this project to count as successful?

### 3. Deliverables

What concrete artifacts must exist?
Examples:

* architecture doc
* workflow doc
* module layout
* prompt folder
* notes folder
* test prompt
* first working slice

### 4. Tasks

What actions create those deliverables?
Examples:

* define V1 scope
* define module boundaries
* choose naming conventions
* write `paths.py`
* test prompt discovery
* write note parser

### 5. Code / Libraries

Only after the earlier layers are stable do we write implementation.

So the current project is not outside the Thought Garden at all.
It is just sitting in a very explicit design phase before code is allowed onto the stage.

---

## The Core Sequence

Here is the recommended design workflow.

### Phase 1: Clarify the problem

This is the “why are we doing this?” phase.

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

This is the “what belongs in this project and what does not?” phase.

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

This is the “what pieces exist and what are their jobs?” phase.

Questions:

* What modules/components are needed?
* What is each one responsible for?
* What should each one never own?
* Where are the seams between parts?

Output:

* architecture doc
* responsibility split
* proposed file/folder layout

---

### Phase 4: Identify the minimum viable slice

This is the “what is the smallest useful thing we can build that proves the design?” phase.

Questions:

* What is the thinnest vertical slice that is actually useful?
* What can we test without building the whole system?
* What gives us signal early?
* What first slice is unlikely to trap us later?

Output:

* V1 feature list
* phase-by-phase build plan

---

### Phase 5: Define the artifacts

This is the “what files or outputs must exist?” phase.

Questions:

* What markdown docs should exist?
* What modules should exist?
* What example prompts or test files should exist?
* What outputs prove the pipe works?

Output:

* deliverables list
* artifact inventory

---

### Phase 6: Sequence the work

This is the “what order reduces pain and risk?” phase.

Questions:

* What must come first because other things depend on it?
* What can be tested independently?
* What pieces should be proven before we add CLI or automation?
* What order keeps the build inspectable?

Output:

* implementation order
* dependency chain

---

### Phase 7: Define validation

This is the “how will we know each layer works?” phase.

Questions:

* What is the smoke test for each module?
* What is a good manual test before automation?
* What failure modes do we expect?
* What evidence counts as success?

Output:

* validation checklist
* smoke tests
* example inputs and outputs

---

### Phase 8: Build the first slice

Only now do we begin implementation.

Rules:

* build one thin slice
* test it
* inspect it
* adjust the design if reality disagrees
* do not sprint ahead because the first part felt good

Output:

* first working slice
* notes about what the design got right/wrong

---

### Phase 9: Review and refine

This is the “did the design survive contact with reality?” phase.

Questions:

* What felt clean?
* What felt awkward?
* What assumptions broke?
* What should be renamed, split, or deferred?

Output:

* refinement notes
* updated architecture if needed

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

### 8. Extension path

What comes later, but not now?

That pattern is portable and should work for many future repos.

---

## Applying This to the Codex Prompt Workflow

For the current project, the decomposition looks like this.

### Purpose

Create a clean, reusable prompt workflow system for Codex work inside a repo.

### Boundaries

In now:

* prompt discovery
* note discovery
* status reconstruction
* retry context preparation
* conservative manual workflow

Out for now:

* direct Codex execution
* rich metadata system
* dependency graphing

exec
/bin/bash -lc "sed -n '321,520p' tools/Project_Design_Workflow.md" in /data/git/handson-ml3
 succeeded in 0ms:
* advanced logging
* multi-repo orchestration

### Components

* `paths.py`
* `prompts.py`
* `notes.py`
* `status.py`
* later maybe `retry.py`
* later maybe `cli.py`

### Artifacts

* architecture doc
* workflow doc
* prompt folder
* notes folder
* example prompts
* minimal modules
* validation notes

### Build order

1. finalize architecture and workflow
2. define V1 scope
3. create folder layout
4. implement `paths.py`
5. implement `prompts.py`
6. implement `notes.py`
7. implement `status.py`
8. manually validate status reconstruction
9. add retry preparation
10. add thin CLI only if needed

### Validation

* can we discover prompt files?
* can we discover note files?
* can we reconstruct current prompt status correctly?
* can we identify next unrun?
* can we pair a failed prompt with its most recent failed note?

### Extension path

* structured note templates
* dry-run bundle preparation
* optional Codex adapter
* dependency handling
* reusable template packaging

---

## The Right Questions Before Building

Before implementation begins, these questions should be answered.

### 1. What is the smallest version that is truly useful?

Do not build past that line yet.

### 2. What is the source of truth?

Filesystem only? Filenames only? Markdown contents too?

### 3. What is the stable identity of a prompt?

Filename? Internal ID? Both?

### 4. What states do we actually need in V1?

Probably fewer than we are tempted to add.

### 5. What should be manually controlled?

Human judgment should stay in the loop where truth is fuzzy.

### 6. What is the next thing we are intentionally not building?

This keeps the scope fence visible.

---

## Design Workflow as a Reusable Template

For future projects, a reliable progression is:

1. conversation
2. problem statement
3. goals
4. scope / boundaries
5. architecture
6. workflow / build plan
7. artifact list
8. implementation sequence
9. validation plan
10. first build slice
11. review / refinement

This is slow in the beginning and fast later.
It feels like more thought up front because it is.
But it prevents the kind of momentum that sends a project downhill with shopping carts tied to its feet.

---

## What Comes Next for This Project

The next practical step is not coding yet.
The next step is to turn the architecture into a **focused V1 build plan**.

That means deciding:

* exact V1 features
* exact folder/module layout
* exact naming rules
* exact validation steps

Once those are fixed, implementation becomes much safer.

---

## Closing Thought

A good design workflow does not slow building down.
It moves confusion to the front, where it is cheap.

That is the whole game.

Think clearly first.
Name the parts.
Set the boundaries.
Choose the first slice.
Build only what is earned.
Then move forward with confidence.


codex
I’ve read the current document. The main revision is straightforward: keep the phased structure and decomposition pattern, but remove the repo-specific application section and fold in an explicit bounded iteration loop, bridge-tooling guidance, durable history, and post-failure analysis.
exec
/bin/bash -lc 'date +%Y%m%d_%H%M%S' in /data/git/handson-ml3
 succeeded in 0ms:
20260415_144244

codex
I’m editing the workflow doc now. The replacement will stay conservative and reusable, with the new principles integrated into the workflow itself rather than appended as project-specific advice.
apply patch
patch: completed
/data/git/handson-ml3/tools/Project_Design_Workflow.md
diff --git a/tools/Project_Design_Workflow.md b/tools/Project_Design_Workflow.md
deleted file mode 100644
index 10aa2005ff8b68da1682930d6f4f2f81445e8150..0000000000000000000000000000000000000000
--- a/tools/Project_Design_Workflow.md
+++ /dev/null
@@ -1,455 +0,0 @@
-# Project Design Workflow
-
-## Goal
-
-Define a repeatable workflow for turning an idea or architecture document into a clean, working system without skipping the thinking that keeps the build safe, clear, and reusable.
-
-This document is meant to sit beside architecture documents and help answer a practical question:
-
-**What are the steps between “this design makes sense” and “the thing works”?**
-
----
-
-## Why This Exists
-
-Sometimes a project feels good at the architecture level but still feels fuzzy at the build level.
-
-That happens because architecture is not the same thing as execution.
-
-Architecture tells us:
-
-* what parts exist
-* what they are responsible for
-* how they relate
-* what should not be coupled
-
-But architecture does not yet tell us:
-
-* what to build first
-* what order reduces risk
-* what to test first
-* what can wait
-* how to know whether the design is holding
-
-This workflow bridges that gap.
-
----
-
-## Relationship to the Thought Garden
-
-This workflow does **not** replace the Thought Garden.
-It fits inside it.
-
-A useful mapping looks like this:
-
-### 1. Conversation
-
-What are we trying to do?
-What hurts right now?
-What would “better” feel like?
-
-### 2. Goals
-
-What must become true for this project to count as successful?
-
-### 3. Deliverables
-
-What concrete artifacts must exist?
-Examples:
-
-* architecture doc
-* workflow doc
-* module layout
-* prompt folder
-* notes folder
-* test prompt
-* first working slice
-
-### 4. Tasks
-
-What actions create those deliverables?
-Examples:
-
-* define V1 scope
-* define module boundaries
-* choose naming conventions
-* write `paths.py`
-* test prompt discovery
-* write note parser
-
-### 5. Code / Libraries
-
-Only after the earlier layers are stable do we write implementation.
-
-So the current project is not outside the Thought Garden at all.
-It is just sitting in a very explicit design phase before code is allowed onto the stage.
-
----
-
-## The Core Sequence
-
-Here is the recommended design workflow.
-
-### Phase 1: Clarify the problem
-
-This is the “why are we doing this?” phase.
-
-Questions:
-
-* What pain are we removing?
-* What confusion are we reducing?
-* What repeated work are we trying to standardize?
-* What is dangerous if we build too fast?
-
-Output:
-
-* short problem statement
-* short success statement
-
----
-
-### Phase 2: Define the system boundary
-
-This is the “what belongs in this project and what does not?” phase.
-
-Questions:
-
-* What should this system do?
-* What should it explicitly not do yet?
-* What decisions are deferred?
-* What adjacent problems are real but out of scope?
-
-Output:
-
-* scope statement
-* out-of-scope list
-
----
-
-### Phase 3: Draft the architecture
-
-This is the “what pieces exist and what are their jobs?” phase.
-
-Questions:
-
-* What modules/components are needed?
-* What is each one responsible for?
-* What should each one never own?
-* Where are the seams between parts?
-
-Output:
-
-* architecture doc
-* responsibility split
-* proposed file/folder layout
-
----
-
-### Phase 4: Identify the minimum viable slice
-
-This is the “what is the smallest useful thing we can build that proves the design?” phase.
-
-Questions:
-
-* What is the thinnest vertical slice that is actually useful?
-* What can we test without building the whole system?
-* What gives us signal early?
-* What first slice is unlikely to trap us later?
-
-Output:
-
-* V1 feature list
-* phase-by-phase build plan
-
----
-
-### Phase 5: Define the artifacts
-
-This is the “what files or outputs must exist?” phase.
-
-Questions:
-
-* What markdown docs should exist?
-* What modules should exist?
-* What example prompts or test files should exist?
-* What outputs prove the pipe works?
-
-Output:
-
-* deliverables list
-* artifact inventory
-
----
-
-### Phase 6: Sequence the work
-
-This is the “what order reduces pain and risk?” phase.
-
-Questions:
-
-* What must come first because other things depend on it?
-* What can be tested independently?
-* What pieces should be proven before we add CLI or automation?
-* What order keeps the build inspectable?
-
-Output:
-
-* implementation order
-* dependency chain
-
----
-
-### Phase 7: Define validation
-
-This is the “how will we know each layer works?” phase.
-
-Questions:
-
-* What is the smoke test for each module?
-* What is a good manual test before automation?
-* What failure modes do we expect?
-* What evidence counts as success?
-
-Output:
-
-* validation checklist
-* smoke tests
-* example inputs and outputs
-
----
-
-### Phase 8: Build the first slice
-
-Only now do we begin implementation.
-
-Rules:
-
-* build one thin slice
-* test it
-* inspect it
-* adjust the design if reality disagrees
-* do not sprint ahead because the first part felt good
-
-Output:
-
-* first working slice
-* notes about what the design got right/wrong
-
----
-
-### Phase 9: Review and refine
-
-This is the “did the design survive contact with reality?” phase.
-
-Questions:
-
-* What felt clean?
-* What felt awkward?
-* What assumptions broke?
-* What should be renamed, split, or deferred?
-
-Output:
-
-* refinement notes
-* updated architecture if needed
-
----
-
-## The Practical Decomposition Pattern
-
-When a design feels big, decompose it in this order:
-
-### 1. Purpose
-
-What is the system for?
-
-### 2. Boundaries
-
-What is in and out?
-
-### 3. Components
-
-What parts exist?
-
-### 4. Responsibilities
-
-What does each part own?
-
-### 5. Artifacts
-
-What files or outputs must exist?
-
-### 6. Sequence
-
-What gets built first?
-
-### 7. Validation
-
-How do we test each step?
-
-### 8. Extension path
-
-What comes later, but not now?
-
-That pattern is portable and should work for many future repos.
-
----
-
-## Applying This to the Codex Prompt Workflow
-
-For the current project, the decomposition looks like this.
-
-### Purpose
-
-Create a clean, reusable prompt workflow system for Codex work inside a repo.
-
-### Boundaries
-
-In now:
-
-* prompt discovery
-* note discovery
-* status reconstruction
-* retry context preparation
-* conservative manual workflow
-
-Out for now:
-
-* direct Codex execution
-* rich metadata system
-* dependency graphing
-* advanced logging
-* multi-repo orchestration
-
-### Components
-
-* `paths.py`
-* `prompts.py`
-* `notes.py`
-* `status.py`
-* later maybe `retry.py`
-* later maybe `cli.py`
-
-### Artifacts
-
-* architecture doc
-* workflow doc
-* prompt folder
-* notes folder
-* example prompts
-* minimal modules
-* validation notes
-
-### Build order
-
-1. finalize architecture and workflow
-2. define V1 scope
-3. create folder layout
-4. implement `paths.py`
-5. implement `prompts.py`
-6. implement `notes.py`
-7. implement `status.py`
-8. manually validate status reconstruction
-9. add retry preparation
-10. add thin CLI only if needed
-
-### Validation
-
-* can we discover prompt files?
-* can we discover note files?
-* can we reconstruct current prompt status correctly?
-* can we identify next unrun?
-* can we pair a failed prompt with its most recent failed note?
-
-### Extension path
-
-* structured note templates
-* dry-run bundle preparation
-* optional Codex adapter
-* dependency handling
-* reusable template packaging
-
----
-
-## The Right Questions Before Building
-
-Before implementation begins, these questions should be answered.
-
-### 1. What is the smallest version that is truly useful?
-
-Do not build past that line yet.
-
-### 2. What is the source of truth?
-
-Filesystem only? Filenames only? Markdown contents too?
-
-### 3. What is the stable identity of a prompt?
-
-Filename? Internal ID? Both?
-
-### 4. What states do we actually need in V1?
-
-Probably fewer than we are tempted to add.
-
-### 5. What should be manually controlled?
-
-Human judgment should stay in the loop where truth is fuzzy.
-
-### 6. What is the next thing we are intentionally not building?
-
-This keeps the scope fence visible.
-
----
-
-## Design Workflow as a Reusable Template
-
-For future projects, a reliable progression is:
-
-1. conversation
-2. problem statement
-3. goals
-4. scope / boundaries
-5. architecture
-6. workflow / build plan
-7. artifact list
-8. implementation sequence
-9. validation plan
-10. first build slice
-11. review / refinement
-
-This is slow in the beginning and fast later.
-It feels like more thought up front because it is.
-But it prevents the kind of momentum that sends a project downhill with shopping carts tied to its feet.
-
----
-
-## What Comes Next for This Project
-
-The next practical step is not coding yet.
-The next step is to turn the architecture into a **focused V1 build plan**.
-
-That means deciding:
-
-* exact V1 features
-* exact folder/module layout
-* exact naming rules
-* exact validation steps
-
-Once those are fixed, implementation becomes much safer.
-
----
-
-## Closing Thought
-
-A good design workflow does not slow building down.
-It moves confusion to the front, where it is cheap.
-
-That is the whole game.
-
-Think clearly first.
-Name the parts.
-Set the boundaries.
-Choose the first slice.
-Build only what is earned.
-Then move forward with confidence.
-

diff --git a/tools/Project_Design_Workflow.md b/tools/Project_Design_Workflow.md
deleted file mode 100644
index 10aa2005ff8b68da1682930d6f4f2f81445e8150..0000000000000000000000000000000000000000
--- a/tools/Project_Design_Workflow.md
+++ /dev/null
@@ -1,455 +0,0 @@
-# Project Design Workflow
-
-## Goal
-
-Define a repeatable workflow for turning an idea or architecture document into a clean, working system without skipping the thinking that keeps the build safe, clear, and reusable.
-
-This document is meant to sit beside architecture documents and help answer a practical question:
-
-**What are the steps between “this design makes sense” and “the thing works”?**
-
----
-
-## Why This Exists
-
-Sometimes a project feels good at the architecture level but still feels fuzzy at the build level.
-
-That happens because architecture is not the same thing as execution.
-
-Architecture tells us:
-
-* what parts exist
-* what they are responsible for
-* how they relate
-* what should not be coupled
-
-But architecture does not yet tell us:
-
-* what to build first
-* what order reduces risk
-* what to test first
-* what can wait
-* how to know whether the design is holding
-
-This workflow bridges that gap.
-
----
-
-## Relationship to the Thought Garden
-
-This workflow does **not** replace the Thought Garden.
-It fits inside it.
-
-A useful mapping looks like this:
-
-### 1. Conversation
-
-What are we trying to do?
-What hurts right now?
-What would “better” feel like?
-
-### 2. Goals
-
-What must become true for this project to count as successful?
-
-### 3. Deliverables
-
-What concrete artifacts must exist?
-Examples:
-
-* architecture doc
-* workflow doc
-* module layout
-* prompt folder
-* notes folder
-* test prompt
-* first working slice
-
-### 4. Tasks
-
-What actions create those deliverables?
-Examples:
-
-* define V1 scope
-* define module boundaries
-* choose naming conventions
-* write `paths.py`
-* test prompt discovery
-* write note parser
-
-### 5. Code / Libraries
-
-Only after the earlier layers are stable do we write implementation.
-
-So the current project is not outside the Thought Garden at all.
-It is just sitting in a very explicit design phase before code is allowed onto the stage.
-
----
-
-## The Core Sequence
-
-Here is the recommended design workflow.
-
-### Phase 1: Clarify the problem
-
-This is the “why are we doing this?” phase.
-
-Questions:
-
-* What pain are we removing?
-* What confusion are we reducing?
-* What repeated work are we trying to standardize?
-* What is dangerous if we build too fast?
-
-Output:
-
-* short problem statement
-* short success statement
-
----
-
-### Phase 2: Define the system boundary
-
-This is the “what belongs in this project and what does not?” phase.
-
-Questions:
-
-* What should this system do?
-* What should it explicitly not do yet?
-* What decisions are deferred?
-* What adjacent problems are real but out of scope?
-
-Output:
-
-* scope statement
-* out-of-scope list
-
----
-
-### Phase 3: Draft the architecture
-
-This is the “what pieces exist and what are their jobs?” phase.
-
-Questions:
-
-* What modules/components are needed?
-* What is each one responsible for?
-* What should each one never own?
-* Where are the seams between parts?
-
-Output:
-
-* architecture doc
-* responsibility split
-* proposed file/folder layout
-
----
-
-### Phase 4: Identify the minimum viable slice
-
-This is the “what is the smallest useful thing we can build that proves the design?” phase.
-
-Questions:
-
-* What is the thinnest vertical slice that is actually useful?
-* What can we test without building the whole system?
-* What gives us signal early?
-* What first slice is unlikely to trap us later?
-
-Output:
-
-* V1 feature list
-* phase-by-phase build plan
-
----
-
-### Phase 5: Define the artifacts
-
-This is the “what files or outputs must exist?” phase.
-
-Questions:
-
-* What markdown docs should exist?
-* What modules should exist?
-* What example prompts or test files should exist?
-* What outputs prove the pipe works?
-
-Output:
-
-* deliverables list
-* artifact inventory
-
----
-
-### Phase 6: Sequence the work
-
-This is the “what order reduces pain and risk?” phase.
-
-Questions:
-
-* What must come first because other things depend on it?
-* What can be tested independently?
-* What pieces should be proven before we add CLI or automation?
-* What order keeps the build inspectable?
-
-Output:
-
-* implementation order
-* dependency chain
-
----
-
-### Phase 7: Define validation
-
-This is the “how will we know each layer works?” phase.
-
-Questions:
-
-* What is the smoke test for each module?
-* What is a good manual test before automation?
-* What failure modes do we expect?
-* What evidence counts as success?
-
-Output:
-
-* validation checklist
-* smoke tests
-* example inputs and outputs
-
----
-
-### Phase 8: Build the first slice
-
-Only now do we begin implementation.
-
-Rules:
-
-* build one thin slice
-* test it
-* inspect it
-* adjust the design if reality disagrees
-* do not sprint ahead because the first part felt good
-
-Output:
-
-* first working slice
-* notes about what the design got right/wrong
-
----
-
-### Phase 9: Review and refine
-
-This is the “did the design survive contact with reality?” phase.
-
-Questions:
-
-* What felt clean?
-* What felt awkward?
-* What assumptions broke?
-* What should be renamed, split, or deferred?
-
-Output:
-
-* refinement notes
-* updated architecture if needed
-
----
-
-## The Practical Decomposition Pattern
-
-When a design feels big, decompose it in this order:
-
-### 1. Purpose
-
-What is the system for?
-
-### 2. Boundaries
-
-What is in and out?
-
-### 3. Components
-
-What parts exist?
-
-### 4. Responsibilities
-
-What does each part own?
-
-### 5. Artifacts
-
-What files or outputs must exist?
-
-### 6. Sequence
-
-What gets built first?
-
-### 7. Validation
-
-How do we test each step?
-
-### 8. Extension path
-
-What comes later, but not now?
-
-That pattern is portable and should work for many future repos.
-
----
-
-## Applying This to the Codex Prompt Workflow
-
-For the current project, the decomposition looks like this.
-
-### Purpose
-
-Create a clean, reusable prompt workflow system for Codex work inside a repo.
-
-### Boundaries
-
-In now:
-
-* prompt discovery
-* note discovery
-* status reconstruction
-* retry context preparation
-* conservative manual workflow
-
-Out for now:
-
-* direct Codex execution
-* rich metadata system
-* dependency graphing
-* advanced logging
-* multi-repo orchestration
-
-### Components
-
-* `paths.py`
-* `prompts.py`
-* `notes.py`
-* `status.py`
-* later maybe `retry.py`
-* later maybe `cli.py`
-
-### Artifacts
-
-* architecture doc
-* workflow doc
-* prompt folder
-* notes folder
-* example prompts
-* minimal modules
-* validation notes
-
-### Build order
-
-1. finalize architecture and workflow
-2. define V1 scope
-3. create folder layout
-4. implement `paths.py`
-5. implement `prompts.py`
-6. implement `notes.py`
-7. implement `status.py`
-8. manually validate status reconstruction
-9. add retry preparation
-10. add thin CLI only if needed
-
-### Validation
-
-* can we discover prompt files?
-* can we discover note files?
-* can we reconstruct current prompt status correctly?
-* can we identify next unrun?
-* can we pair a failed prompt with its most recent failed note?
-
-### Extension path
-
-* structured note templates
-* dry-run bundle preparation
-* optional Codex adapter
-* dependency handling
-* reusable template packaging
-
----
-
-## The Right Questions Before Building
-
-Before implementation begins, these questions should be answered.
-
-### 1. What is the smallest version that is truly useful?
-
-Do not build past that line yet.
-
-### 2. What is the source of truth?
-
-Filesystem only? Filenames only? Markdown contents too?
-
-### 3. What is the stable identity of a prompt?
-
-Filename? Internal ID? Both?
-
-### 4. What states do we actually need in V1?
-
-Probably fewer than we are tempted to add.
-
-### 5. What should be manually controlled?
-
-Human judgment should stay in the loop where truth is fuzzy.
-
-### 6. What is the next thing we are intentionally not building?
-
-This keeps the scope fence visible.
-
----
-
-## Design Workflow as a Reusable Template
-
-For future projects, a reliable progression is:
-
-1. conversation
-2. problem statement
-3. goals
-4. scope / boundaries
-5. architecture
-6. workflow / build plan
-7. artifact list
-8. implementation sequence
-9. validation plan
-10. first build slice
-11. review / refinement
-
-This is slow in the beginning and fast later.
-It feels like more thought up front because it is.
-But it prevents the kind of momentum that sends a project downhill with shopping carts tied to its feet.
-
----
-
-## What Comes Next for This Project
-
-The next practical step is not coding yet.
-The next step is to turn the architecture into a **focused V1 build plan**.
-
-That means deciding:
-
-* exact V1 features
-* exact folder/module layout
-* exact naming rules
-* exact validation steps
-
-Once those are fixed, implementation becomes much safer.
-
----
-
-## Closing Thought
-
-A good design workflow does not slow building down.
-It moves confusion to the front, where it is cheap.
-
-That is the whole game.
-
-Think clearly first.
-Name the parts.
-Set the boundaries.
-Choose the first slice.
-Build only what is earned.
-Then move forward with confidence.
-

apply patch
patch: completed
/data/git/handson-ml3/tools/Project_Design_Workflow.md
diff --git a/tools/Project_Design_Workflow.md b/tools/Project_Design_Workflow.md
index 10aa2005ff8b68da1682930d6f4f2f81445e8150..812efe01ef35dc5dde789875c10cd96ceaca542a
--- a/tools/Project_Design_Workflow.md
+++ b/tools/Project_Design_Workflow.md
@@ -6,13 +6,13 @@
 
 This document is meant to sit beside architecture documents and help answer a practical question:
 
-**What are the steps between “this design makes sense” and “the thing works”?**
+**What are the steps between "this design makes sense" and "the thing works"?**
 
 ---
 
 ## Why This Exists
 
-Sometimes a project feels good at the architecture level but still feels fuzzy at the build level.
+Sometimes a project feels sound at the architecture level but still feels fuzzy at the build level.
 
 That happens because architecture is not the same thing as execution.
 
@@ -30,14 +30,15 @@
 * what to test first
 * what can wait
 * how to know whether the design is holding
+* how to recover cleanly when a step fails
 
 This workflow bridges that gap.
 
 ---
 
-## Relationship to the Thought Garden
+## Relationship to a Larger Design Process
 
-This workflow does **not** replace the Thought Garden.
+This workflow does **not** replace broader project thinking.
 It fits inside it.
 
 A useful mapping looks like this:
@@ -46,7 +47,7 @@
 
 What are we trying to do?
 What hurts right now?
-What would “better” feel like?
+What would "better" feel like?
 
 ### 2. Goals
 
@@ -60,9 +61,9 @@
 * architecture doc
 * workflow doc
 * module layout
-* prompt folder
+* interface sketch
 * notes folder
-* test prompt
+* validation checklist
 * first working slice
 
 ### 4. Tasks
@@ -73,26 +74,73 @@
 * define V1 scope
 * define module boundaries
 * choose naming conventions
-* write `paths.py`
-* test prompt discovery
-* write note parser
+* write first thin component
+* validate one path end to end
+* capture review notes
 
-### 5. Code / Libraries
+### 5. Implementation
 
 Only after the earlier layers are stable do we write implementation.
 
-So the current project is not outside the Thought Garden at all.
-It is just sitting in a very explicit design phase before code is allowed onto the stage.
+So this workflow is not separate from design.
+It is the part that turns design into bounded execution.
 
 ---
 
+## Core Principles
+
+### 1. Design before build
+
+Do not use implementation momentum to hide unclear thinking.
+
+### 2. Boundaries before breadth
+
+A project becomes safer when scope, responsibility, and deferral lines are visible.
+
+### 3. Thin slices before large pushes
+
+Prefer a bounded slice that proves something real over a large batch that creates many unknowns at once.
+
+### 4. Review between iterations
+
+Each bounded execution step should be inspected before the next one is issued.
+Do not assume a sequence is healthy just because the previous step completed.
+
+### 5. Validation is part of design
+
+Tests, inspections, examples, and acceptance checks are not extra work.
+They are how the design proves it is real.
+
+### 6. Bridge tooling is allowed, but subordinate
+
+Temporary or bridge tooling can be useful when it stays thin, inspectable, and reversible.
+It may help move work forward, gather evidence, or reduce manual friction.
+It should not become a substitute for architecture, clear interfaces, or good decomposition.
+
+### 7. Durable local history matters
+
+Notes, logs, reports, generated outputs, and similar local artifacts can serve as durable project memory.
+They help preserve decisions, validation results, failure patterns, and the reasoning behind refinements.
+
+### 8. Failure should produce analysis, not just retries
+
+When a step fails, the useful question is not only "how do we try again?"
+It is also "what did this failure reveal about the task, decomposition, criteria, or environment?"
+
+### 9. Resource use should be observed
+
+Project effort has a cost, whether that cost is human time, machine time, review effort, retries, execution size, or external spend.
+Large tasks should justify themselves.
+
+---
+
 ## The Core Sequence
 
 Here is the recommended design workflow.
 
 ### Phase 1: Clarify the problem
 
-This is the “why are we doing this?” phase.
+This is the "why are we doing this?" phase.
 
 Questions:
 
@@ -110,7 +158,7 @@
 
 ### Phase 2: Define the system boundary
 
-This is the “what belongs in this project and what does not?” phase.
+This is the "what belongs in this project and what does not?" phase.
 
 Questions:
 
@@ -128,11 +176,11 @@
 
 ### Phase 3: Draft the architecture
 
-This is the “what pieces exist and what are their jobs?” phase.
+This is the "what pieces exist and what are their jobs?" phase.
 
 Questions:
 
-* What modules/components are needed?
+* What modules, components, or services are needed?
 * What is each one responsible for?
 * What should each one never own?
 * Where are the seams between parts?
@@ -141,13 +189,13 @@
 
 * architecture doc
 * responsibility split
-* proposed file/folder layout
+* proposed file, package, or interface layout
 
 ---
 
 ### Phase 4: Identify the minimum viable slice
 
-This is the “what is the smallest useful thing we can build that proves the design?” phase.
+This is the "what is the smallest useful thing we can build that proves the design?" phase.
 
 Questions:
 
@@ -159,20 +207,21 @@
 Output:
 
 * V1 feature list
-* phase-by-phase build plan
+* initial build plan
 
 ---
 
 ### Phase 5: Define the artifacts
 
-This is the “what files or outputs must exist?” phase.
+This is the "what files or outputs must exist?" phase.
 
 Questions:
 
-* What markdown docs should exist?
-* What modules should exist?
-* What example prompts or test files should exist?
-* What outputs prove the pipe works?
+* What documents should exist?
+* What modules or interfaces should exist?
+* What examples, fixtures, or test inputs should exist?
+* What outputs prove the path works?
+* What notes, logs, or reports should be kept as durable project memory?
 
 Output:
 
@@ -183,13 +232,13 @@
 
 ### Phase 6: Sequence the work
 
-This is the “what order reduces pain and risk?” phase.
+This is the "what order reduces pain and risk?" phase.
 
 Questions:
 
 * What must come first because other things depend on it?
 * What can be tested independently?
-* What pieces should be proven before we add CLI or automation?
+* What pieces should be proven before automation or convenience layers are added?
 * What order keeps the build inspectable?
 
 Output:
@@ -201,57 +250,87 @@
 
 ### Phase 7: Define validation
 
-This is the “how will we know each layer works?” phase.
+This is the "how will we know each layer works?" phase.
 
 Questions:
 
-* What is the smoke test for each module?
+* What is the smoke test for each part?
 * What is a good manual test before automation?
 * What failure modes do we expect?
 * What evidence counts as success?
+* What review should happen before the next step begins?
 
 Output:
 
 * validation checklist
 * smoke tests
 * example inputs and outputs
+* review points between iterations
 
 ---
 
-### Phase 8: Build the first slice
+### Phase 8: Execute one bounded slice
 
 Only now do we begin implementation.
 
 Rules:
 
 * build one thin slice
-* test it
-* inspect it
-* adjust the design if reality disagrees
+* keep the task bounded and inspectable
+* use bridge tooling only when it remains thin and subordinate to the design
+* validate the slice
+* inspect the result before issuing the next step
 * do not sprint ahead because the first part felt good
 
 Output:
 
-* first working slice
-* notes about what the design got right/wrong
+* one completed slice
+* evidence of validation
+* notes about what the design got right or wrong
 
 ---
 
 ### Phase 9: Review and refine
 
-This is the “did the design survive contact with reality?” phase.
+This is the "did the design survive contact with reality?" phase.
 
 Questions:
 
 * What felt clean?
 * What felt awkward?
 * What assumptions broke?
-* What should be renamed, split, or deferred?
+* What should be renamed, split, simplified, or deferred?
+* What should change before the next bounded step?
 
 Output:
 
 * refinement notes
-* updated architecture if needed
+* updated architecture or scope if needed
+* revised next-step plan
+
+---
+
+## The Iteration Loop
+
+After the initial design work, many projects should move through a repeating bounded loop:
+
+1. clarify the next chunk
+2. define a bounded task
+3. state the success criteria
+4. execute
+5. validate and inspect results
+6. review what changed
+7. refine the plan
+8. issue the next bounded task only after review
+
+This loop should stay small enough that:
+
+* the task can be understood before execution
+* the result can be reviewed without guesswork
+* failure teaches something specific
+* refinement happens while context is still fresh
+
+If a project cannot explain the next chunk in a few clear sentences, the chunk is probably still too large.
 
 ---
 
@@ -287,87 +366,72 @@
 
 How do we test each step?
 
-### 8. Extension path
+### 8. Review points
+
+Where do we stop and inspect before continuing?
+
+### 9. Extension path
 
 What comes later, but not now?
 
-That pattern is portable and should work for many future repos.
+That pattern is portable and should work across many projects.
 
 ---
 
-## Applying This to the Codex Prompt Workflow
+## Failure Analysis as Part of the Workflow
 
-For the current project, the decomposition looks like this.
+When a bounded step fails, do not treat the failure as noise.
+Capture it, inspect it, and decide whether the problem is in the task, the decomposition, the criteria, or the environment.
 
-### Purpose
+Useful questions include:
 
-Create a clean, reusable prompt workflow system for Codex work inside a repo.
+* Was the task too large?
+* Was the task poorly decomposed?
+* Were success criteria unclear or incomplete?
+* Did the task depend on hidden assumptions?
+* Was the failure caused by tooling or infrastructure rather than task difficulty?
+* Would a smaller or differently framed task have worked better?
+* Did the review happen too late?
+* Did retries produce new information, or only repeat cost?
 
-### Boundaries
+Useful outputs include:
 
-In now:
+* a short failure note or report
+* updated task boundaries
+* revised success criteria
+* a smaller follow-up slice
+* a decision to fix environment issues before retrying
 
-* prompt discovery
-* note discovery
-* status reconstruction
-* retry context preparation
-* conservative manual workflow
+A clean retry is often possible, but it should come after analysis rather than instead of it.
 
-Out for now:
+---
 
-* direct Codex execution
-* rich metadata system
-* dependency graphing
-* advanced logging
-* multi-repo orchestration
+## Resource and Cost Awareness
 
-### Components
+Projects benefit from tracking lightweight evidence about execution cost and quality.
+This does not need to be elaborate, but it should be enough to notice patterns.
 
-* `paths.py`
-* `prompts.py`
-* `notes.py`
-* `status.py`
-* later maybe `retry.py`
-* later maybe `cli.py`
+Examples of useful observations:
 
-### Artifacts
+* elapsed time
+* execution size
+* review effort
+* repeated retries
+* failure frequency
+* output quality
+* machine or service usage
 
-* architecture doc
-* workflow doc
-* prompt folder
-* notes folder
-* example prompts
-* minimal modules
-* validation notes
+In some environments, teams may also track:
 
-### Build order
-
-1. finalize architecture and workflow
-2. define V1 scope
-3. create folder layout
-4. implement `paths.py`
-5. implement `prompts.py`
-6. implement `notes.py`
-7. implement `status.py`
-8. manually validate status reconstruction
-9. add retry preparation
-10. add thin CLI only if needed
-
-### Validation
-
-* can we discover prompt files?
-* can we discover note files?
-* can we reconstruct current prompt status correctly?
-* can we identify next unrun?
-* can we pair a failed prompt with its most recent failed note?
+* token usage
+* compute time
+* API cost
+* artifact volume
 
-### Extension path
+The purpose is not accounting for its own sake.
+The purpose is to notice whether larger, more expensive, or less bounded tasks correlate with more failures, lower quality, or weaker reviewability.
 
-* structured note templates
-* dry-run bundle preparation
-* optional Codex adapter
-* dependency handling
-* reusable template packaging
+If those patterns appear, the workflow should respond by shrinking task size, clarifying criteria, or improving decomposition.
 
 ---
 
@@ -381,41 +445,50 @@
 
 ### 2. What is the source of truth?
 
-Filesystem only? Filenames only? Markdown contents too?
+What artifacts, interfaces, or records define reality for this system?
 
-### 3. What is the stable identity of a prompt?
+### 3. What is the stable identity of each important object?
 
-Filename? Internal ID? Both?
+Names, IDs, paths, interfaces, and records should not be ambiguous.
 
 ### 4. What states do we actually need in V1?
 
 Probably fewer than we are tempted to add.
 
-### 5. What should be manually controlled?
+### 5. What should remain manually controlled?
 
 Human judgment should stay in the loop where truth is fuzzy.
 
-### 6. What is the next thing we are intentionally not building?
+### 6. What bridge tooling is acceptable?
 
+If temporary helpers are needed, how do we keep them thin, inspectable, and easy to replace?
+
+### 7. What history should be preserved?
+
+What notes, outputs, validation records, or failure reports will be worth having later?
+
+### 8. What is the next thing we are intentionally not building?
+
 This keeps the scope fence visible.
 
 ---
 
 ## Design Workflow as a Reusable Template
 
-For future projects, a reliable progression is:
+For many projects, a reliable progression is:
 
 1. conversation
 2. problem statement
 3. goals
-4. scope / boundaries
+4. scope and boundaries
 5. architecture
-6. workflow / build plan
+6. workflow and build plan
 7. artifact list
 8. implementation sequence
 9. validation plan
-10. first build slice
-11. review / refinement
+10. bounded execution loop
+11. review and refinement
+12. failure analysis when needed
 
 This is slow in the beginning and fast later.
 It feels like more thought up front because it is.
@@ -423,22 +496,6 @@
 
 ---
 
-## What Comes Next for This Project
-
-The next practical step is not coding yet.
-The next step is to turn the architecture into a **focused V1 build plan**.
-
-That means deciding:
-
-* exact V1 features
-* exact folder/module layout
-* exact naming rules
-* exact validation steps
-
-Once those are fixed, implementation becomes much safer.
-
----
-
 ## Closing Thought
 
 A good design workflow does not slow building down.
@@ -450,6 +507,7 @@
 Name the parts.
 Set the boundaries.
 Choose the first slice.
+Review each iteration.
+Study failure instead of hiding it.
 Build only what is earned.
 Then move forward with confidence.
-

diff --git a/tools/Project_Design_Workflow.md b/tools/Project_Design_Workflow.md
index 10aa2005ff8b68da1682930d6f4f2f81445e8150..812efe01ef35dc5dde789875c10cd96ceaca542a
--- a/tools/Project_Design_Workflow.md
+++ b/tools/Project_Design_Workflow.md
@@ -6,13 +6,13 @@
 
 This document is meant to sit beside architecture documents and help answer a practical question:
 
-**What are the steps between “this design makes sense” and “the thing works”?**
+**What are the steps between "this design makes sense" and "the thing works"?**
 
 ---
 
 ## Why This Exists
 
-Sometimes a project feels good at the architecture level but still feels fuzzy at the build level.
+Sometimes a project feels sound at the architecture level but still feels fuzzy at the build level.
 
 That happens because architecture is not the same thing as execution.
 
@@ -30,14 +30,15 @@
 * what to test first
 * what can wait
 * how to know whether the design is holding
+* how to recover cleanly when a step fails
 
 This workflow bridges that gap.
 
 ---
 
-## Relationship to the Thought Garden
+## Relationship to a Larger Design Process
 
-This workflow does **not** replace the Thought Garden.
+This workflow does **not** replace broader project thinking.
 It fits inside it.
 
 A useful mapping looks like this:
@@ -46,7 +47,7 @@
 
 What are we trying to do?
 What hurts right now?
-What would “better” feel like?
+What would "better" feel like?
 
 ### 2. Goals
 
@@ -60,9 +61,9 @@
 * architecture doc
 * workflow doc
 * module layout
-* prompt folder
+* interface sketch
 * notes folder
-* test prompt
+* validation checklist
 * first working slice
 
 ### 4. Tasks
@@ -73,26 +74,73 @@
 * define V1 scope
 * define module boundaries
 * choose naming conventions
-* write `paths.py`
-* test prompt discovery
-* write note parser
+* write first thin component
+* validate one path end to end
+* capture review notes
 
-### 5. Code / Libraries
+### 5. Implementation
 
 Only after the earlier layers are stable do we write implementation.
 
-So the current project is not outside the Thought Garden at all.
-It is just sitting in a very explicit design phase before code is allowed onto the stage.
+So this workflow is not separate from design.
+It is the part that turns design into bounded execution.
 
 ---
 
+## Core Principles
+
+### 1. Design before build
+
+Do not use implementation momentum to hide unclear thinking.
+
+### 2. Boundaries before breadth
+
+A project becomes safer when scope, responsibility, and deferral lines are visible.
+
+### 3. Thin slices before large pushes
+
+Prefer a bounded slice that proves something real over a large batch that creates many unknowns at once.
+
+### 4. Review between iterations
+
+Each bounded execution step should be inspected before the next one is issued.
+Do not assume a sequence is healthy just because the previous step completed.
+
+### 5. Validation is part of design
+
+Tests, inspections, examples, and acceptance checks are not extra work.
+They are how the design proves it is real.
+
+### 6. Bridge tooling is allowed, but subordinate
+
+Temporary or bridge tooling can be useful when it stays thin, inspectable, and reversible.
+It may help move work forward, gather evidence, or reduce manual friction.
+It should not become a substitute for architecture, clear interfaces, or good decomposition.
+
+### 7. Durable local history matters
+
+Notes, logs, reports, generated outputs, and similar local artifacts can serve as durable project memory.
+They help preserve decisions, validation results, failure patterns, and the reasoning behind refinements.
+
+### 8. Failure should produce analysis, not just retries
+
+When a step fails, the useful question is not only "how do we try again?"
+It is also "what did this failure reveal about the task, decomposition, criteria, or environment?"
+
+### 9. Resource use should be observed
+
+Project effort has a cost, whether that cost is human time, machine time, review effort, retries, execution size, or external spend.
+Large tasks should justify themselves.
+
+---
+
 ## The Core Sequence
 
 Here is the recommended design workflow.
 
 ### Phase 1: Clarify the problem
 
-This is the “why are we doing this?” phase.
+This is the "why are we doing this?" phase.
 
 Questions:
 
@@ -110,7 +158,7 @@
 
 ### Phase 2: Define the system boundary
 
-This is the “what belongs in this project and what does not?” phase.
+This is the "what belongs in this project and what does not?" phase.
 
 Questions:
 
@@ -128,11 +176,11 @@
 
 ### Phase 3: Draft the architecture
 
-This is the “what pieces exist and what are their jobs?” phase.
+This is the "what pieces exist and what are their jobs?" phase.
 
 Questions:
 
-* What modules/components are needed?
+* What modules, components, or services are needed?
 * What is each one responsible for?
 * What should each one never own?
 * Where are the seams between parts?
@@ -141,13 +189,13 @@
 
 * architecture doc
 * responsibility split
-* proposed file/folder layout
+* proposed file, package, or interface layout
 
 ---
 
 ### Phase 4: Identify the minimum viable slice
 
-This is the “what is the smallest useful thing we can build that proves the design?” phase.
+This is the "what is the smallest useful thing we can build that proves the design?" phase.
 
 Questions:
 
@@ -159,20 +207,21 @@
 Output:
 
 * V1 feature list
-* phase-by-phase build plan
+* initial build plan
 
 ---
 
 ### Phase 5: Define the artifacts
 
-This is the “what files or outputs must exist?” phase.
+This is the "what files or outputs must exist?" phase.
 
 Questions:
 
-* What markdown docs should exist?
-* What modules should exist?
-* What example prompts or test files should exist?
-* What outputs prove the pipe works?
+* What documents should exist?
+* What modules or interfaces should exist?
+* What examples, fixtures, or test inputs should exist?
+* What outputs prove the path works?
+* What notes, logs, or reports should be kept as durable project memory?
 
 Output:
 
@@ -183,13 +232,13 @@
 
 ### Phase 6: Sequence the work
 
-This is the “what order reduces pain and risk?” phase.
+This is the "what order reduces pain and risk?" phase.
 
 Questions:
 
 * What must come first because other things depend on it?
 * What can be tested independently?
-* What pieces should be proven before we add CLI or automation?
+* What pieces should be proven before automation or convenience layers are added?
 * What order keeps the build inspectable?
 
 Output:
@@ -201,57 +250,87 @@
 
 ### Phase 7: Define validation
 
-This is the “how will we know each layer works?” phase.
+This is the "how will we know each layer works?" phase.
 
 Questions:
 
-* What is the smoke test for each module?
+* What is the smoke test for each part?
 * What is a good manual test before automation?
 * What failure modes do we expect?
 * What evidence counts as success?
+* What review should happen before the next step begins?
 
 Output:
 
 * validation checklist
 * smoke tests
 * example inputs and outputs
+* review points between iterations
 
 ---
 
-### Phase 8: Build the first slice
+### Phase 8: Execute one bounded slice
 
 Only now do we begin implementation.
 
 Rules:
 
 * build one thin slice
-* test it
-* inspect it
-* adjust the design if reality disagrees
+* keep the task bounded and inspectable
+* use bridge tooling only when it remains thin and subordinate to the design
+* validate the slice
+* inspect the result before issuing the next step
 * do not sprint ahead because the first part felt good
 
 Output:
 
-* first working slice
-* notes about what the design got right/wrong
+* one completed slice
+* evidence of validation
+* notes about what the design got right or wrong
 
 ---
 
 ### Phase 9: Review and refine
 
-This is the “did the design survive contact with reality?” phase.
+This is the "did the design survive contact with reality?" phase.
 
 Questions:
 
 * What felt clean?
 * What felt awkward?
 * What assumptions broke?
-* What should be renamed, split, or deferred?
+* What should be renamed, split, simplified, or deferred?
+* What should change before the next bounded step?
 
 Output:
 
 * refinement notes
-* updated architecture if needed
+* updated architecture or scope if needed
+* revised next-step plan
+
+---
+
+## The Iteration Loop
+
+After the initial design work, many projects should move through a repeating bounded loop:
+
+1. clarify the next chunk
+2. define a bounded task
+3. state the success criteria
+4. execute
+5. validate and inspect results
+6. review what changed
+7. refine the plan
+8. issue the next bounded task only after review
+
+This loop should stay small enough that:
+
+* the task can be understood before execution
+* the result can be reviewed without guesswork
+* failure teaches something specific
+* refinement happens while context is still fresh
+
+If a project cannot explain the next chunk in a few clear sentences, the chunk is probably still too large.
 
 ---
 
@@ -287,87 +366,72 @@
 
 How do we test each step?
 
-### 8. Extension path
+### 8. Review points
+
+Where do we stop and inspect before continuing?
+
+### 9. Extension path
 
 What comes later, but not now?
 
-That pattern is portable and should work for many future repos.
+That pattern is portable and should work across many projects.
 
 ---
 
-## Applying This to the Codex Prompt Workflow
+## Failure Analysis as Part of the Workflow
 
-For the current project, the decomposition looks like this.
+When a bounded step fails, do not treat the failure as noise.
+Capture it, inspect it, and decide whether the problem is in the task, the decomposition, the criteria, or the environment.
 
-### Purpose
+Useful questions include:
 
-Create a clean, reusable prompt workflow system for Codex work inside a repo.
+* Was the task too large?
+* Was the task poorly decomposed?
+* Were success criteria unclear or incomplete?
+* Did the task depend on hidden assumptions?
+* Was the failure caused by tooling or infrastructure rather than task difficulty?
+* Would a smaller or differently framed task have worked better?
+* Did the review happen too late?
+* Did retries produce new information, or only repeat cost?
 
-### Boundaries
+Useful outputs include:
 
-In now:
+* a short failure note or report
+* updated task boundaries
+* revised success criteria
+* a smaller follow-up slice
+* a decision to fix environment issues before retrying
 
-* prompt discovery
-* note discovery
-* status reconstruction
-* retry context preparation
-* conservative manual workflow
+A clean retry is often possible, but it should come after analysis rather than instead of it.
 
-Out for now:
+---
 
-* direct Codex execution
-* rich metadata system
-* dependency graphing
-* advanced logging
-* multi-repo orchestration
+## Resource and Cost Awareness
 
-### Components
+Projects benefit from tracking lightweight evidence about execution cost and quality.
+This does not need to be elaborate, but it should be enough to notice patterns.
 
-* `paths.py`
-* `prompts.py`
-* `notes.py`
-* `status.py`
-* later maybe `retry.py`
-* later maybe `cli.py`
+Examples of useful observations:
 
-### Artifacts
+* elapsed time
+* execution size
+* review effort
+* repeated retries
+* failure frequency
+* output quality
+* machine or service usage
 
-* architecture doc
-* workflow doc
-* prompt folder
-* notes folder
-* example prompts
-* minimal modules
-* validation notes
+In some environments, teams may also track:
 
-### Build order
-
-1. finalize architecture and workflow
-2. define V1 scope
-3. create folder layout
-4. implement `paths.py`
-5. implement `prompts.py`
-6. implement `notes.py`
-7. implement `status.py`
-8. manually validate status reconstruction
-9. add retry preparation
-10. add thin CLI only if needed
-
-### Validation
-
-* can we discover prompt files?
-* can we discover note files?
-* can we reconstruct current prompt status correctly?
-* can we identify next unrun?
-* can we pair a failed prompt with its most recent failed note?
+* token usage
+* compute time
+* API cost
+* artifact volume
 
-### Extension path
+The purpose is not accounting for its own sake.
+The purpose is to notice whether larger, more expensive, or less bounded tasks correlate with more failures, lower quality, or weaker reviewability.
 
-* structured note templates
-* dry-run bundle preparation
-* optional Codex adapter
-* dependency handling
-* reusable template packaging
+If those patterns appear, the workflow should respond by shrinking task size, clarifying criteria, or improving decomposition.
 
 ---
 
@@ -381,41 +445,50 @@
 
 ### 2. What is the source of truth?
 
-Filesystem only? Filenames only? Markdown contents too?
+What artifacts, interfaces, or records define reality for this system?
 
-### 3. What is the stable identity of a prompt?
+### 3. What is the stable identity of each important object?
 
-Filename? Internal ID? Both?
+Names, IDs, paths, interfaces, and records should not be ambiguous.
 
 ### 4. What states do we actually need in V1?
 
 Probably fewer than we are tempted to add.
 
-### 5. What should be manually controlled?
+### 5. What should remain manually controlled?
 
 Human judgment should stay in the loop where truth is fuzzy.
 
-### 6. What is the next thing we are intentionally not building?
+### 6. What bridge tooling is acceptable?
 
+If temporary helpers are needed, how do we keep them thin, inspectable, and easy to replace?
+
+### 7. What history should be preserved?
+
+What notes, outputs, validation records, or failure reports will be worth having later?
+
+### 8. What is the next thing we are intentionally not building?
+
 This keeps the scope fence visible.
 
 ---
 
 ## Design Workflow as a Reusable Template
 
-For future projects, a reliable progression is:
+For many projects, a reliable progression is:
 
 1. conversation
 2. problem statement
 3. goals
-4. scope / boundaries
+4. scope and boundaries
 5. architecture
-6. workflow / build plan
+6. workflow and build plan
 7. artifact list
 8. implementation sequence
 9. validation plan
-10. first build slice
-11. review / refinement
+10. bounded execution loop
+11. review and refinement
+12. failure analysis when needed
 
 This is slow in the beginning and fast later.
 It feels like more thought up front because it is.
@@ -423,22 +496,6 @@
 
 ---
 
-## What Comes Next for This Project
-
-The next practical step is not coding yet.
-The next step is to turn the architecture into a **focused V1 build plan**.
-
-That means deciding:
-
-* exact V1 features
-* exact folder/module layout
-* exact naming rules
-* exact validation steps
-
-Once those are fixed, implementation becomes much safer.
-
----
-
 ## Closing Thought
 
 A good design workflow does not slow building down.
@@ -450,6 +507,7 @@
 Name the parts.
 Set the boundaries.
 Choose the first slice.
+Review each iteration.
+Study failure instead of hiding it.
 Build only what is earned.
 Then move forward with confidence.
-

codex
The main doc rewrite is in place. I’m adding the short companion note now with a timestamped filename, keeping it concise and focused on what changed, why, and what tradeoffs remain.
apply patch
patch: completed
/data/git/handson-ml3/notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md
diff --git a/notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md b/notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md
new file mode 100644
index 0000000000000000000000000000000000000000..8a0f0c6f5c6bfd6828fd2294aa3d091fb4f847d6
--- /dev/null
+++ b/notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md
@@ -0,0 +1,27 @@
+# Project Design Workflow Revision
+
+## What Changed
+
+Revised `tools/Project_Design_Workflow.md` into a more project-agnostic workflow document.
+Removed the project-specific application section and replaced it with reusable guidance.
+Added explicit sections for bounded iteration, failure analysis, resource and cost awareness, and durable local history.
+
+## Why These Changes Were Made
+
+The earlier version had a strong phased structure, but part of it was tied to one specific workflow and repo context.
+The revision keeps the design-first, boundary-aware approach while making the document reusable across different projects, tools, and maturity levels.
+
+## New Reusable Principles Added
+
+* bounded execution in small reviewable slices
+* review between iterations before issuing the next step
+* bridge tooling allowed when thin, inspectable, and subordinate
+* durable local notes, logs, outputs, and reports as project memory
+* failure triggering analysis rather than reflexive retry
+* lightweight observation of cost, time, retries, and failure patterns
+
+## Tradeoffs and Open Questions
+
+The document is now more broadly reusable, but less concrete for any one project.
+That tradeoff seems correct for a shared workflow template.
+If later needed, a separate project-specific companion doc could show how this workflow maps onto a particular repo without narrowing the core template.
diff --git a/tools/Project_Design_Workflow.md b/tools/Project_Design_Workflow.md
index 10aa2005ff8b68da1682930d6f4f2f81445e8150..812efe01ef35dc5dde789875c10cd96ceaca542a
--- a/tools/Project_Design_Workflow.md
+++ b/tools/Project_Design_Workflow.md
@@ -6,13 +6,13 @@
 
 This document is meant to sit beside architecture documents and help answer a practical question:
 
-**What are the steps between “this design makes sense” and “the thing works”?**
+**What are the steps between "this design makes sense" and "the thing works"?**
 
 ---
 
 ## Why This Exists
 
-Sometimes a project feels good at the architecture level but still feels fuzzy at the build level.
+Sometimes a project feels sound at the architecture level but still feels fuzzy at the build level.
 
 That happens because architecture is not the same thing as execution.
 
@@ -30,14 +30,15 @@
 * what to test first
 * what can wait
 * how to know whether the design is holding
+* how to recover cleanly when a step fails
 
 This workflow bridges that gap.
 
 ---
 
-## Relationship to the Thought Garden
+## Relationship to a Larger Design Process
 
-This workflow does **not** replace the Thought Garden.
+This workflow does **not** replace broader project thinking.
 It fits inside it.
 
 A useful mapping looks like this:
@@ -46,7 +47,7 @@
 
 What are we trying to do?
 What hurts right now?
-What would “better” feel like?
+What would "better" feel like?
 
 ### 2. Goals
 
@@ -60,9 +61,9 @@
 * architecture doc
 * workflow doc
 * module layout
-* prompt folder
+* interface sketch
 * notes folder
-* test prompt
+* validation checklist
 * first working slice
 
 ### 4. Tasks
@@ -73,26 +74,73 @@
 * define V1 scope
 * define module boundaries
 * choose naming conventions
-* write `paths.py`
-* test prompt discovery
-* write note parser
+* write first thin component
+* validate one path end to end
+* capture review notes
 
-### 5. Code / Libraries
+### 5. Implementation
 
 Only after the earlier layers are stable do we write implementation.
 
-So the current project is not outside the Thought Garden at all.
-It is just sitting in a very explicit design phase before code is allowed onto the stage.
+So this workflow is not separate from design.
+It is the part that turns design into bounded execution.
 
 ---
 
+## Core Principles
+
+### 1. Design before build
+
+Do not use implementation momentum to hide unclear thinking.
+
+### 2. Boundaries before breadth
+
+A project becomes safer when scope, responsibility, and deferral lines are visible.
+
+### 3. Thin slices before large pushes
+
+Prefer a bounded slice that proves something real over a large batch that creates many unknowns at once.
+
+### 4. Review between iterations
+
+Each bounded execution step should be inspected before the next one is issued.
+Do not assume a sequence is healthy just because the previous step completed.
+
+### 5. Validation is part of design
+
+Tests, inspections, examples, and acceptance checks are not extra work.
+They are how the design proves it is real.
+
+### 6. Bridge tooling is allowed, but subordinate
+
+Temporary or bridge tooling can be useful when it stays thin, inspectable, and reversible.
+It may help move work forward, gather evidence, or reduce manual friction.
+It should not become a substitute for architecture, clear interfaces, or good decomposition.
+
+### 7. Durable local history matters
+
+Notes, logs, reports, generated outputs, and similar local artifacts can serve as durable project memory.
+They help preserve decisions, validation results, failure patterns, and the reasoning behind refinements.
+
+### 8. Failure should produce analysis, not just retries
+
+When a step fails, the useful question is not only "how do we try again?"
+It is also "what did this failure reveal about the task, decomposition, criteria, or environment?"
+
+### 9. Resource use should be observed
+
+Project effort has a cost, whether that cost is human time, machine time, review effort, retries, execution size, or external spend.
+Large tasks should justify themselves.
+
+---
+
 ## The Core Sequence
 
 Here is the recommended design workflow.
 
 ### Phase 1: Clarify the problem
 
-This is the “why are we doing this?” phase.
+This is the "why are we doing this?" phase.
 
 Questions:
 
@@ -110,7 +158,7 @@
 
 ### Phase 2: Define the system boundary
 
-This is the “what belongs in this project and what does not?” phase.
+This is the "what belongs in this project and what does not?" phase.
 
 Questions:
 
@@ -128,11 +176,11 @@
 
 ### Phase 3: Draft the architecture
 
-This is the “what pieces exist and what are their jobs?” phase.
+This is the "what pieces exist and what are their jobs?" phase.
 
 Questions:
 
-* What modules/components are needed?
+* What modules, components, or services are needed?
 * What is each one responsible for?
 * What should each one never own?
 * Where are the seams between parts?
@@ -141,13 +189,13 @@
 
 * architecture doc
 * responsibility split
-* proposed file/folder layout
+* proposed file, package, or interface layout
 
 ---
 
 ### Phase 4: Identify the minimum viable slice
 
-This is the “what is the smallest useful thing we can build that proves the design?” phase.
+This is the "what is the smallest useful thing we can build that proves the design?" phase.
 
 Questions:
 
@@ -159,20 +207,21 @@
 Output:
 
 * V1 feature list
-* phase-by-phase build plan
+* initial build plan
 
 ---
 
 ### Phase 5: Define the artifacts
 
-This is the “what files or outputs must exist?” phase.
+This is the "what files or outputs must exist?" phase.
 
 Questions:
 
-* What markdown docs should exist?
-* What modules should exist?
-* What example prompts or test files should exist?
-* What outputs prove the pipe works?
+* What documents should exist?
+* What modules or interfaces should exist?
+* What examples, fixtures, or test inputs should exist?
+* What outputs prove the path works?
+* What notes, logs, or reports should be kept as durable project memory?
 
 Output:
 
@@ -183,13 +232,13 @@
 
 ### Phase 6: Sequence the work
 
-This is the “what order reduces pain and risk?” phase.
+This is the "what order reduces pain and risk?" phase.
 
 Questions:
 
 * What must come first because other things depend on it?
 * What can be tested independently?
-* What pieces should be proven before we add CLI or automation?
+* What pieces should be proven before automation or convenience layers are added?
 * What order keeps the build inspectable?
 
 Output:
@@ -201,57 +250,87 @@
 
 ### Phase 7: Define validation
 
-This is the “how will we know each layer works?” phase.
+This is the "how will we know each layer works?" phase.
 
 Questions:
 
-* What is the smoke test for each module?
+* What is the smoke test for each part?
 * What is a good manual test before automation?
 * What failure modes do we expect?
 * What evidence counts as success?
+* What review should happen before the next step begins?
 
 Output:
 
 * validation checklist
 * smoke tests
 * example inputs and outputs
+* review points between iterations
 
 ---
 
-### Phase 8: Build the first slice
+### Phase 8: Execute one bounded slice
 
 Only now do we begin implementation.
 
 Rules:
 
 * build one thin slice
-* test it
-* inspect it
-* adjust the design if reality disagrees
+* keep the task bounded and inspectable
+* use bridge tooling only when it remains thin and subordinate to the design
+* validate the slice
+* inspect the result before issuing the next step
 * do not sprint ahead because the first part felt good
 
 Output:
 
-* first working slice
-* notes about what the design got right/wrong
+* one completed slice
+* evidence of validation
+* notes about what the design got right or wrong
 
 ---
 
 ### Phase 9: Review and refine
 
-This is the “did the design survive contact with reality?” phase.
+This is the "did the design survive contact with reality?" phase.
 
 Questions:
 
 * What felt clean?
 * What felt awkward?
 * What assumptions broke?
-* What should be renamed, split, or deferred?
+* What should be renamed, split, simplified, or deferred?
+* What should change before the next bounded step?
 
 Output:
 
 * refinement notes
-* updated architecture if needed
+* updated architecture or scope if needed
+* revised next-step plan
+
+---
+
+## The Iteration Loop
+
+After the initial design work, many projects should move through a repeating bounded loop:
+
+1. clarify the next chunk
+2. define a bounded task
+3. state the success criteria
+4. execute
+5. validate and inspect results
+6. review what changed
+7. refine the plan
+8. issue the next bounded task only after review
+
+This loop should stay small enough that:
+
+* the task can be understood before execution
+* the result can be reviewed without guesswork
+* failure teaches something specific
+* refinement happens while context is still fresh
+
+If a project cannot explain the next chunk in a few clear sentences, the chunk is probably still too large.
 
 ---
 
@@ -287,87 +366,72 @@
 
 How do we test each step?
 
-### 8. Extension path
+### 8. Review points
+
+Where do we stop and inspect before continuing?
+
+### 9. Extension path
 
 What comes later, but not now?
 
-That pattern is portable and should work for many future repos.
+That pattern is portable and should work across many projects.
 
 ---
 
-## Applying This to the Codex Prompt Workflow
+## Failure Analysis as Part of the Workflow
 
-For the current project, the decomposition looks like this.
+When a bounded step fails, do not treat the failure as noise.
+Capture it, inspect it, and decide whether the problem is in the task, the decomposition, the criteria, or the environment.
 
-### Purpose
+Useful questions include:
 
-Create a clean, reusable prompt workflow system for Codex work inside a repo.
+* Was the task too large?
+* Was the task poorly decomposed?
+* Were success criteria unclear or incomplete?
+* Did the task depend on hidden assumptions?
+* Was the failure caused by tooling or infrastructure rather than task difficulty?
+* Would a smaller or differently framed task have worked better?
+* Did the review happen too late?
+* Did retries produce new information, or only repeat cost?
 
-### Boundaries
+Useful outputs include:
 
-In now:
+* a short failure note or report
+* updated task boundaries
+* revised success criteria
+* a smaller follow-up slice
+* a decision to fix environment issues before retrying
 
-* prompt discovery
-* note discovery
-* status reconstruction
-* retry context preparation
-* conservative manual workflow
+A clean retry is often possible, but it should come after analysis rather than instead of it.
 
-Out for now:
+---
 
-* direct Codex execution
-* rich metadata system
-* dependency graphing
-* advanced logging
-* multi-repo orchestration
+## Resource and Cost Awareness
 
-### Components
+Projects benefit from tracking lightweight evidence about execution cost and quality.
+This does not need to be elaborate, but it should be enough to notice patterns.
 
-* `paths.py`
-* `prompts.py`
-* `notes.py`
-* `status.py`
-* later maybe `retry.py`
-* later maybe `cli.py`
+Examples of useful observations:
 
-### Artifacts
+* elapsed time
+* execution size
+* review effort
+* repeated retries
+* failure frequency
+* output quality
+* machine or service usage
 
-* architecture doc
-* workflow doc
-* prompt folder
-* notes folder
-* example prompts
-* minimal modules
-* validation notes
+In some environments, teams may also track:
 
-### Build order
-
-1. finalize architecture and workflow
-2. define V1 scope
-3. create folder layout
-4. implement `paths.py`
-5. implement `prompts.py`
-6. implement `notes.py`
-7. implement `status.py`
-8. manually validate status reconstruction
-9. add retry preparation
-10. add thin CLI only if needed
-
-### Validation
-
-* can we discover prompt files?
-* can we discover note files?
-* can we reconstruct current prompt status correctly?
-* can we identify next unrun?
-* can we pair a failed prompt with its most recent failed note?
+* token usage
+* compute time
+* API cost
+* artifact volume
 
-### Extension path
+The purpose is not accounting for its own sake.
+The purpose is to notice whether larger, more expensive, or less bounded tasks correlate with more failures, lower quality, or weaker reviewability.
 
-* structured note templates
-* dry-run bundle preparation
-* optional Codex adapter
-* dependency handling
-* reusable template packaging
+If those patterns appear, the workflow should respond by shrinking task size, clarifying criteria, or improving decomposition.
 
 ---
 
@@ -381,41 +445,50 @@
 
 ### 2. What is the source of truth?
 
-Filesystem only? Filenames only? Markdown contents too?
+What artifacts, interfaces, or records define reality for this system?
 
-### 3. What is the stable identity of a prompt?
+### 3. What is the stable identity of each important object?
 
-Filename? Internal ID? Both?
+Names, IDs, paths, interfaces, and records should not be ambiguous.
 
 ### 4. What states do we actually need in V1?
 
 Probably fewer than we are tempted to add.
 
-### 5. What should be manually controlled?
+### 5. What should remain manually controlled?
 
 Human judgment should stay in the loop where truth is fuzzy.
 
-### 6. What is the next thing we are intentionally not building?
+### 6. What bridge tooling is acceptable?
 
+If temporary helpers are needed, how do we keep them thin, inspectable, and easy to replace?
+
+### 7. What history should be preserved?
+
+What notes, outputs, validation records, or failure reports will be worth having later?
+
+### 8. What is the next thing we are intentionally not building?
+
 This keeps the scope fence visible.
 
 ---
 
 ## Design Workflow as a Reusable Template
 
-For future projects, a reliable progression is:
+For many projects, a reliable progression is:
 
 1. conversation
 2. problem statement
 3. goals
-4. scope / boundaries
+4. scope and boundaries
 5. architecture
-6. workflow / build plan
+6. workflow and build plan
 7. artifact list
 8. implementation sequence
 9. validation plan
-10. first build slice
-11. review / refinement
+10. bounded execution loop
+11. review and refinement
+12. failure analysis when needed
 
 This is slow in the beginning and fast later.
 It feels like more thought up front because it is.
@@ -423,22 +496,6 @@
 
 ---
 
-## What Comes Next for This Project
-
-The next practical step is not coding yet.
-The next step is to turn the architecture into a **focused V1 build plan**.
-
-That means deciding:
-
-* exact V1 features
-* exact folder/module layout
-* exact naming rules
-* exact validation steps
-
-Once those are fixed, implementation becomes much safer.
-
----
-
 ## Closing Thought
 
 A good design workflow does not slow building down.
@@ -450,6 +507,7 @@
 Name the parts.
 Set the boundaries.
 Choose the first slice.
+Review each iteration.
+Study failure instead of hiding it.
 Build only what is earned.
 Then move forward with confidence.
-

diff --git a/notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md b/notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md
new file mode 100644
index 0000000000000000000000000000000000000000..8a0f0c6f5c6bfd6828fd2294aa3d091fb4f847d6
--- /dev/null
+++ b/notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md
@@ -0,0 +1,27 @@
+# Project Design Workflow Revision
+
+## What Changed
+
+Revised `tools/Project_Design_Workflow.md` into a more project-agnostic workflow document.
+Removed the project-specific application section and replaced it with reusable guidance.
+Added explicit sections for bounded iteration, failure analysis, resource and cost awareness, and durable local history.
+
+## Why These Changes Were Made
+
+The earlier version had a strong phased structure, but part of it was tied to one specific workflow and repo context.
+The revision keeps the design-first, boundary-aware approach while making the document reusable across different projects, tools, and maturity levels.
+
+## New Reusable Principles Added
+
+* bounded execution in small reviewable slices
+* review between iterations before issuing the next step
+* bridge tooling allowed when thin, inspectable, and subordinate
+* durable local notes, logs, outputs, and reports as project memory
+* failure triggering analysis rather than reflexive retry
+* lightweight observation of cost, time, retries, and failure patterns
+
+## Tradeoffs and Open Questions
+
+The document is now more broadly reusable, but less concrete for any one project.
+That tradeoff seems correct for a shared workflow template.
+If later needed, a separate project-specific companion doc could show how this workflow maps onto a particular repo without narrowing the core template.
diff --git a/tools/Project_Design_Workflow.md b/tools/Project_Design_Workflow.md
index 10aa2005ff8b68da1682930d6f4f2f81445e8150..812efe01ef35dc5dde789875c10cd96ceaca542a
--- a/tools/Project_Design_Workflow.md
+++ b/tools/Project_Design_Workflow.md
@@ -6,13 +6,13 @@
 
 This document is meant to sit beside architecture documents and help answer a practical question:
 
-**What are the steps between “this design makes sense” and “the thing works”?**
+**What are the steps between "this design makes sense" and "the thing works"?**
 
 ---
 
 ## Why This Exists
 
-Sometimes a project feels good at the architecture level but still feels fuzzy at the build level.
+Sometimes a project feels sound at the architecture level but still feels fuzzy at the build level.
 
 That happens because architecture is not the same thing as execution.
 
@@ -30,14 +30,15 @@
 * what to test first
 * what can wait
 * how to know whether the design is holding
+* how to recover cleanly when a step fails
 
 This workflow bridges that gap.
 
 ---
 
-## Relationship to the Thought Garden
+## Relationship to a Larger Design Process
 
-This workflow does **not** replace the Thought Garden.
+This workflow does **not** replace broader project thinking.
 It fits inside it.
 
 A useful mapping looks like this:
@@ -46,7 +47,7 @@
 
 What are we trying to do?
 What hurts right now?
-What would “better” feel like?
+What would "better" feel like?
 
 ### 2. Goals
 
@@ -60,9 +61,9 @@
 * architecture doc
 * workflow doc
 * module layout
-* prompt folder
+* interface sketch
 * notes folder
-* test prompt
+* validation checklist
 * first working slice
 
 ### 4. Tasks
@@ -73,26 +74,73 @@
 * define V1 scope
 * define module boundaries
 * choose naming conventions
-* write `paths.py`
-* test prompt discovery
-* write note parser
+* write first thin component
+* validate one path end to end
+* capture review notes
 
-### 5. Code / Libraries
+### 5. Implementation
 
 Only after the earlier layers are stable do we write implementation.
 
-So the current project is not outside the Thought Garden at all.
-It is just sitting in a very explicit design phase before code is allowed onto the stage.
+So this workflow is not separate from design.
+It is the part that turns design into bounded execution.
 
 ---
 
+## Core Principles
+
+### 1. Design before build
+
+Do not use implementation momentum to hide unclear thinking.
+
+### 2. Boundaries before breadth
+
+A project becomes safer when scope, responsibility, and deferral lines are visible.
+
+### 3. Thin slices before large pushes
+
+Prefer a bounded slice that proves something real over a large batch that creates many unknowns at once.
+
+### 4. Review between iterations
+
+Each bounded execution step should be inspected before the next one is issued.
+Do not assume a sequence is healthy just because the previous step completed.
+
+### 5. Validation is part of design
+
+Tests, inspections, examples, and acceptance checks are not extra work.
+They are how the design proves it is real.
+
+### 6. Bridge tooling is allowed, but subordinate
+
+Temporary or bridge tooling can be useful when it stays thin, inspectable, and reversible.
+It may help move work forward, gather evidence, or reduce manual friction.
+It should not become a substitute for architecture, clear interfaces, or good decomposition.
+
+### 7. Durable local history matters
+
+Notes, logs, reports, generated outputs, and similar local artifacts can serve as durable project memory.
+They help preserve decisions, validation results, failure patterns, and the reasoning behind refinements.
+
+### 8. Failure should produce analysis, not just retries
+
+When a step fails, the useful question is not only "how do we try again?"
+It is also "what did this failure reveal about the task, decomposition, criteria, or environment?"
+
+### 9. Resource use should be observed
+
+Project effort has a cost, whether that cost is human time, machine time, review effort, retries, execution size, or external spend.
+Large tasks should justify themselves.
+
+---
+
 ## The Core Sequence
 
 Here is the recommended design workflow.
 
 ### Phase 1: Clarify the problem
 
-This is the “why are we doing this?” phase.
+This is the "why are we doing this?" phase.
 
 Questions:
 
@@ -110,7 +158,7 @@
 
 ### Phase 2: Define the system boundary
 
-This is the “what belongs in this project and what does not?” phase.
+This is the "what belongs in this project and what does not?" phase.
 
 Questions:
 
@@ -128,11 +176,11 @@
 
 ### Phase 3: Draft the architecture
 
-This is the “what pieces exist and what are their jobs?” phase.
+This is the "what pieces exist and what are their jobs?" phase.
 
 Questions:
 
-* What modules/components are needed?
+* What modules, components, or services are needed?
 * What is each one responsible for?
 * What should each one never own?
 * Where are the seams between parts?
@@ -141,13 +189,13 @@
 
 * architecture doc
 * responsibility split
-* proposed file/folder layout
+* proposed file, package, or interface layout
 
 ---
 
 ### Phase 4: Identify the minimum viable slice
 
-This is the “what is the smallest useful thing we can build that proves the design?” phase.
+This is the "what is the smallest useful thing we can build that proves the design?" phase.
 
 Questions:
 
@@ -159,20 +207,21 @@
 Output:
 
 * V1 feature list
-* phase-by-phase build plan
+* initial build plan
 
 ---
 
 ### Phase 5: Define the artifacts
 
-This is the “what files or outputs must exist?” phase.
+This is the "what files or outputs must exist?" phase.
 
 Questions:
 
-* What markdown docs should exist?
-* What modules should exist?
-* What example prompts or test files should exist?
-* What outputs prove the pipe works?
+* What documents should exist?
+* What modules or interfaces should exist?
+* What examples, fixtures, or test inputs should exist?
+* What outputs prove the path works?
+* What notes, logs, or reports should be kept as durable project memory?
 
 Output:
 
@@ -183,13 +232,13 @@
 
 ### Phase 6: Sequence the work
 
-This is the “what order reduces pain and risk?” phase.
+This is the "what order reduces pain and risk?" phase.
 
 Questions:
 
 * What must come first because other things depend on it?
 * What can be tested independently?
-* What pieces should be proven before we add CLI or automation?
+* What pieces should be proven before automation or convenience layers are added?
 * What order keeps the build inspectable?
 
 Output:
@@ -201,57 +250,87 @@
 
 ### Phase 7: Define validation
 
-This is the “how will we know each layer works?” phase.
+This is the "how will we know each layer works?" phase.
 
 Questions:
 
-* What is the smoke test for each module?
+* What is the smoke test for each part?
 * What is a good manual test before automation?
 * What failure modes do we expect?
 * What evidence counts as success?
+* What review should happen before the next step begins?
 
 Output:
 
 * validation checklist
 * smoke tests
 * example inputs and outputs
+* review points between iterations
 
 ---
 
-### Phase 8: Build the first slice
+### Phase 8: Execute one bounded slice
 
 Only now do we begin implementation.
 
 Rules:
 
 * build one thin slice
-* test it
-* inspect it
-* adjust the design if reality disagrees
+* keep the task bounded and inspectable
+* use bridge tooling only when it remains thin and subordinate to the design
+* validate the slice
+* inspect the result before issuing the next step
 * do not sprint ahead because the first part felt good
 
 Output:
 
-* first working slice
-* notes about what the design got right/wrong
+* one completed slice
+* evidence of validation
+* notes about what the design got right or wrong
 
 ---
 
 ### Phase 9: Review and refine
 
-This is the “did the design survive contact with reality?” phase.
+This is the "did the design survive contact with reality?" phase.
 
 Questions:
 
 * What felt clean?
 * What felt awkward?
 * What assumptions broke?
-* What should be renamed, split, or deferred?
+* What should be renamed, split, simplified, or deferred?
+* What should change before the next bounded step?
 
 Output:
 
 * refinement notes
-* updated architecture if needed
+* updated architecture or scope if needed
+* revised next-step plan
+
+---
+
+## The Iteration Loop
+
+After the initial design work, many projects should move through a repeating bounded loop:
+
+1. clarify the next chunk
+2. define a bounded task
+3. state the success criteria
+4. execute
+5. validate and inspect results
+6. review what changed
+7. refine the plan
+8. issue the next bounded task only after review
+
+This loop should stay small enough that:
+
+* the task can be understood before execution
+* the result can be reviewed without guesswork
+* failure teaches something specific
+* refinement happens while context is still fresh
+
+If a project cannot explain the next chunk in a few clear sentences, the chunk is probably still too large.
 
 ---
 
@@ -287,87 +366,72 @@
 
 How do we test each step?
 
-### 8. Extension path
+### 8. Review points
+
+Where do we stop and inspect before continuing?
+
+### 9. Extension path
 
 What comes later, but not now?
 
-That pattern is portable and should work for many future repos.
+That pattern is portable and should work across many projects.
 
 ---
 
-## Applying This to the Codex Prompt Workflow
+## Failure Analysis as Part of the Workflow
 
-For the current project, the decomposition looks like this.
+When a bounded step fails, do not treat the failure as noise.
+Capture it, inspect it, and decide whether the problem is in the task, the decomposition, the criteria, or the environment.
 
-### Purpose
+Useful questions include:
 
-Create a clean, reusable prompt workflow system for Codex work inside a repo.
+* Was the task too large?
+* Was the task poorly decomposed?
+* Were success criteria unclear or incomplete?
+* Did the task depend on hidden assumptions?
+* Was the failure caused by tooling or infrastructure rather than task difficulty?
+* Would a smaller or differently framed task have worked better?
+* Did the review happen too late?
+* Did retries produce new information, or only repeat cost?
 
-### Boundaries
+Useful outputs include:
 
-In now:
+* a short failure note or report
+* updated task boundaries
+* revised success criteria
+* a smaller follow-up slice
+* a decision to fix environment issues before retrying
 
-* prompt discovery
-* note discovery
-* status reconstruction
-* retry context preparation
-* conservative manual workflow
+A clean retry is often possible, but it should come after analysis rather than instead of it.
 
-Out for now:
+---
 
-* direct Codex execution
-* rich metadata system
-* dependency graphing
-* advanced logging
-* multi-repo orchestration
+## Resource and Cost Awareness
 
-### Components
+Projects benefit from tracking lightweight evidence about execution cost and quality.
+This does not need to be elaborate, but it should be enough to notice patterns.
 
-* `paths.py`
-* `prompts.py`
-* `notes.py`
-* `status.py`
-* later maybe `retry.py`
-* later maybe `cli.py`
+Examples of useful observations:
 
-### Artifacts
+* elapsed time
+* execution size
+* review effort
+* repeated retries
+* failure frequency
+* output quality
+* machine or service usage
 
-* architecture doc
-* workflow doc
-* prompt folder
-* notes folder
-* example prompts
-* minimal modules
-* validation notes
+In some environments, teams may also track:
 
-### Build order
-
-1. finalize architecture and workflow
-2. define V1 scope
-3. create folder layout
-4. implement `paths.py`
-5. implement `prompts.py`
-6. implement `notes.py`
-7. implement `status.py`
-8. manually validate status reconstruction
-9. add retry preparation
-10. add thin CLI only if needed
-
-### Validation
-
-* can we discover prompt files?
-* can we discover note files?
-* can we reconstruct current prompt status correctly?
-* can we identify next unrun?
-* can we pair a failed prompt with its most recent failed note?
+* token usage
+* compute time
+* API cost
+* artifact volume
 
-### Extension path
+The purpose is not accounting for its own sake.
+The purpose is to notice whether larger, more expensive, or less bounded tasks correlate with more failures, lower quality, or weaker reviewability.
 
-* structured note templates
-* dry-run bundle preparation
-* optional Codex adapter
-* dependency handling
-* reusable template packaging
+If those patterns appear, the workflow should respond by shrinking task size, clarifying criteria, or improving decomposition.
 
 ---
 
@@ -381,41 +445,50 @@
 
 ### 2. What is the source of truth?
 
-Filesystem only? Filenames only? Markdown contents too?
+What artifacts, interfaces, or records define reality for this system?
 
-### 3. What is the stable identity of a prompt?
+### 3. What is the stable identity of each important object?
 
-Filename? Internal ID? Both?
+Names, IDs, paths, interfaces, and records should not be ambiguous.
 
 ### 4. What states do we actually need in V1?
 
 Probably fewer than we are tempted to add.
 
-### 5. What should be manually controlled?
+### 5. What should remain manually controlled?
 
 Human judgment should stay in the loop where truth is fuzzy.
 
-### 6. What is the next thing we are intentionally not building?
+### 6. What bridge tooling is acceptable?
 
+If temporary helpers are needed, how do we keep them thin, inspectable, and easy to replace?
+
+### 7. What history should be preserved?
+
+What notes, outputs, validation records, or failure reports will be worth having later?
+
+### 8. What is the next thing we are intentionally not building?
+
 This keeps the scope fence visible.
 
 ---
 
 ## Design Workflow as a Reusable Template
 
-For future projects, a reliable progression is:
+For many projects, a reliable progression is:
 
 1. conversation
 2. problem statement
 3. goals
-4. scope / boundaries
+4. scope and boundaries
 5. architecture
-6. workflow / build plan
+6. workflow and build plan
 7. artifact list
 8. implementation sequence
 9. validation plan
-10. first build slice
-11. review / refinement
+10. bounded execution loop
+11. review and refinement
+12. failure analysis when needed
 
 This is slow in the beginning and fast later.
 It feels like more thought up front because it is.
@@ -423,22 +496,6 @@
 
 ---
 
-## What Comes Next for This Project
-
-The next practical step is not coding yet.
-The next step is to turn the architecture into a **focused V1 build plan**.
-
-That means deciding:
-
-* exact V1 features
-* exact folder/module layout
-* exact naming rules
-* exact validation steps
-
-Once those are fixed, implementation becomes much safer.
-
----
-
 ## Closing Thought
 
 A good design workflow does not slow building down.
@@ -450,6 +507,7 @@
 Name the parts.
 Set the boundaries.
 Choose the first slice.
+Review each iteration.
+Study failure instead of hiding it.
 Build only what is earned.
 Then move forward with confidence.
-

exec
/bin/bash -lc "sed -n '1,320p' tools/Project_Design_Workflow.md" in /data/git/handson-ml3
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

exec
/bin/bash -lc "sed -n '1,220p' notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md" in /data/git/handson-ml3
 succeeded in 0ms:
# Project Design Workflow Revision

## What Changed

Revised `tools/Project_Design_Workflow.md` into a more project-agnostic workflow document.
Removed the project-specific application section and replaced it with reusable guidance.
Added explicit sections for bounded iteration, failure analysis, resource and cost awareness, and durable local history.

## Why These Changes Were Made

The earlier version had a strong phased structure, but part of it was tied to one specific workflow and repo context.
The revision keeps the design-first, boundary-aware approach while making the document reusable across different projects, tools, and maturity levels.

## New Reusable Principles Added

* bounded execution in small reviewable slices
* review between iterations before issuing the next step
* bridge tooling allowed when thin, inspectable, and subordinate
* durable local notes, logs, outputs, and reports as project memory
* failure triggering analysis rather than reflexive retry
* lightweight observation of cost, time, retries, and failure patterns

## Tradeoffs and Open Questions

The document is now more broadly reusable, but less concrete for any one project.
That tradeoff seems correct for a shared workflow template.
If later needed, a separate project-specific companion doc could show how this workflow maps onto a particular repo without narrowing the core template.

diff --git a/notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md b/notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md
new file mode 100644
index 0000000000000000000000000000000000000000..8a0f0c6f5c6bfd6828fd2294aa3d091fb4f847d6
--- /dev/null
+++ b/notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md
@@ -0,0 +1,27 @@
+# Project Design Workflow Revision
+
+## What Changed
+
+Revised `tools/Project_Design_Workflow.md` into a more project-agnostic workflow document.
+Removed the project-specific application section and replaced it with reusable guidance.
+Added explicit sections for bounded iteration, failure analysis, resource and cost awareness, and durable local history.
+
+## Why These Changes Were Made
+
+The earlier version had a strong phased structure, but part of it was tied to one specific workflow and repo context.
+The revision keeps the design-first, boundary-aware approach while making the document reusable across different projects, tools, and maturity levels.
+
+## New Reusable Principles Added
+
+* bounded execution in small reviewable slices
+* review between iterations before issuing the next step
+* bridge tooling allowed when thin, inspectable, and subordinate
+* durable local notes, logs, outputs, and reports as project memory
+* failure triggering analysis rather than reflexive retry
+* lightweight observation of cost, time, retries, and failure patterns
+
+## Tradeoffs and Open Questions
+
+The document is now more broadly reusable, but less concrete for any one project.
+That tradeoff seems correct for a shared workflow template.
+If later needed, a separate project-specific companion doc could show how this workflow maps onto a particular repo without narrowing the core template.
diff --git a/tools/Project_Design_Workflow.md b/tools/Project_Design_Workflow.md
index 10aa2005ff8b68da1682930d6f4f2f81445e8150..812efe01ef35dc5dde789875c10cd96ceaca542a
--- a/tools/Project_Design_Workflow.md
+++ b/tools/Project_Design_Workflow.md
@@ -6,13 +6,13 @@
 
 This document is meant to sit beside architecture documents and help answer a practical question:
 
-**What are the steps between “this design makes sense” and “the thing works”?**
+**What are the steps between "this design makes sense" and "the thing works"?**
 
 ---
 
 ## Why This Exists
 
-Sometimes a project feels good at the architecture level but still feels fuzzy at the build level.
+Sometimes a project feels sound at the architecture level but still feels fuzzy at the build level.
 
 That happens because architecture is not the same thing as execution.
 
@@ -30,14 +30,15 @@
 * what to test first
 * what can wait
 * how to know whether the design is holding
+* how to recover cleanly when a step fails
 
 This workflow bridges that gap.
 
 ---
 
-## Relationship to the Thought Garden
+## Relationship to a Larger Design Process
 
-This workflow does **not** replace the Thought Garden.
+This workflow does **not** replace broader project thinking.
 It fits inside it.
 
 A useful mapping looks like this:
@@ -46,7 +47,7 @@
 
 What are we trying to do?
 What hurts right now?
-What would “better” feel like?
+What would "better" feel like?
 
 ### 2. Goals
 
@@ -60,9 +61,9 @@
 * architecture doc
 * workflow doc
 * module layout
-* prompt folder
+* interface sketch
 * notes folder
-* test prompt
+* validation checklist
 * first working slice
 
 ### 4. Tasks
@@ -73,26 +74,73 @@
 * define V1 scope
 * define module boundaries
 * choose naming conventions
-* write `paths.py`
-* test prompt discovery
-* write note parser
+* write first thin component
+* validate one path end to end
+* capture review notes
 
-### 5. Code / Libraries
+### 5. Implementation
 
 Only after the earlier layers are stable do we write implementation.
 
-So the current project is not outside the Thought Garden at all.
-It is just sitting in a very explicit design phase before code is allowed onto the stage.
+So this workflow is not separate from design.
+It is the part that turns design into bounded execution.
 
 ---
 
+## Core Principles
+
+### 1. Design before build
+
+Do not use implementation momentum to hide unclear thinking.
+
+### 2. Boundaries before breadth
+
+A project becomes safer when scope, responsibility, and deferral lines are visible.
+
+### 3. Thin slices before large pushes
+
+Prefer a bounded slice that proves something real over a large batch that creates many unknowns at once.
+
+### 4. Review between iterations
+
+Each bounded execution step should be inspected before the next one is issued.
+Do not assume a sequence is healthy just because the previous step completed.
+
+### 5. Validation is part of design
+
+Tests, inspections, examples, and acceptance checks are not extra work.
+They are how the design proves it is real.
+
+### 6. Bridge tooling is allowed, but subordinate
+
+Temporary or bridge tooling can be useful when it stays thin, inspectable, and reversible.
+It may help move work forward, gather evidence, or reduce manual friction.
+It should not become a substitute for architecture, clear interfaces, or good decomposition.
+
+### 7. Durable local history matters
+
+Notes, logs, reports, generated outputs, and similar local artifacts can serve as durable project memory.
+They help preserve decisions, validation results, failure patterns, and the reasoning behind refinements.
+
+### 8. Failure should produce analysis, not just retries
+
+When a step fails, the useful question is not only "how do we try again?"
+It is also "what did this failure reveal about the task, decomposition, criteria, or environment?"
+
+### 9. Resource use should be observed
+
+Project effort has a cost, whether that cost is human time, machine time, review effort, retries, execution size, or external spend.
+Large tasks should justify themselves.
+
+---
+
 ## The Core Sequence
 
 Here is the recommended design workflow.
 
 ### Phase 1: Clarify the problem
 
-This is the “why are we doing this?” phase.
+This is the "why are we doing this?" phase.
 
 Questions:
 
@@ -110,7 +158,7 @@
 
 ### Phase 2: Define the system boundary
 
-This is the “what belongs in this project and what does not?” phase.
+This is the "what belongs in this project and what does not?" phase.
 
 Questions:
 
@@ -128,11 +176,11 @@
 
 ### Phase 3: Draft the architecture
 
-This is the “what pieces exist and what are their jobs?” phase.
+This is the "what pieces exist and what are their jobs?" phase.
 
 Questions:
 
-* What modules/components are needed?
+* What modules, components, or services are needed?
 * What is each one responsible for?
 * What should each one never own?
 * Where are the seams between parts?
@@ -141,13 +189,13 @@
 
 * architecture doc
 * responsibility split
-* proposed file/folder layout
+* proposed file, package, or interface layout
 
 ---
 
 ### Phase 4: Identify the minimum viable slice
 
-This is the “what is the smallest useful thing we can build that proves the design?” phase.
+This is the "what is the smallest useful thing we can build that proves the design?" phase.
 
 Questions:
 
@@ -159,20 +207,21 @@
 Output:
 
 * V1 feature list
-* phase-by-phase build plan
+* initial build plan
 
 ---
 
 ### Phase 5: Define the artifacts
 
-This is the “what files or outputs must exist?” phase.
+This is the "what files or outputs must exist?" phase.
 
 Questions:
 
-* What markdown docs should exist?
-* What modules should exist?
-* What example prompts or test files should exist?
-* What outputs prove the pipe works?
+* What documents should exist?
+* What modules or interfaces should exist?
+* What examples, fixtures, or test inputs should exist?
+* What outputs prove the path works?
+* What notes, logs, or reports should be kept as durable project memory?
 
 Output:
 
@@ -183,13 +232,13 @@
 
 ### Phase 6: Sequence the work
 
-This is the “what order reduces pain and risk?” phase.
+This is the "what order reduces pain and risk?" phase.
 
 Questions:
 
 * What must come first because other things depend on it?
 * What can be tested independently?
-* What pieces should be proven before we add CLI or automation?
+* What pieces should be proven before automation or convenience layers are added?
 * What order keeps the build inspectable?
 
 Output:
@@ -201,57 +250,87 @@
 
 ### Phase 7: Define validation
 
-This is the “how will we know each layer works?” phase.
+This is the "how will we know each layer works?" phase.
 
 Questions:
 
-* What is the smoke test for each module?
+* What is the smoke test for each part?
 * What is a good manual test before automation?
 * What failure modes do we expect?
 * What evidence counts as success?
+* What review should happen before the next step begins?
 
 Output:
 
 * validation checklist
 * smoke tests
 * example inputs and outputs
+* review points between iterations
 
 ---
 
-### Phase 8: Build the first slice
+### Phase 8: Execute one bounded slice
 
 Only now do we begin implementation.
 
 Rules:
 
 * build one thin slice
-* test it
-* inspect it
-* adjust the design if reality disagrees
+* keep the task bounded and inspectable
+* use bridge tooling only when it remains thin and subordinate to the design
+* validate the slice
+* inspect the result before issuing the next step
 * do not sprint ahead because the first part felt good
 
 Output:
 
-* first working slice
-* notes about what the design got right/wrong
+* one completed slice
+* evidence of validation
+* notes about what the design got right or wrong
 
 ---
 
 ### Phase 9: Review and refine
 
-This is the “did the design survive contact with reality?” phase.
+This is the "did the design survive contact with reality?" phase.
 
 Questions:
 
 * What felt clean?
 * What felt awkward?
 * What assumptions broke?
-* What should be renamed, split, or deferred?
+* What should be renamed, split, simplified, or deferred?
+* What should change before the next bounded step?
 
 Output:
 
 * refinement notes
-* updated architecture if needed
+* updated architecture or scope if needed
+* revised next-step plan
+
+---
+
+## The Iteration Loop
+
+After the initial design work, many projects should move through a repeating bounded loop:
+
+1. clarify the next chunk
+2. define a bounded task
+3. state the success criteria
+4. execute
+5. validate and inspect results
+6. review what changed
+7. refine the plan
+8. issue the next bounded task only after review
+
+This loop should stay small enough that:
+
+* the task can be understood before execution
+* the result can be reviewed without guesswork
+* failure teaches something specific
+* refinement happens while context is still fresh
+
+If a project cannot explain the next chunk in a few clear sentences, the chunk is probably still too large.
 
 ---
 
@@ -287,87 +366,72 @@
 
 How do we test each step?
 
-### 8. Extension path
+### 8. Review points
+
+Where do we stop and inspect before continuing?
+
+### 9. Extension path
 
 What comes later, but not now?
 
-That pattern is portable and should work for many future repos.
+That pattern is portable and should work across many projects.
 
 ---
 
-## Applying This to the Codex Prompt Workflow
+## Failure Analysis as Part of the Workflow
 
-For the current project, the decomposition looks like this.
+When a bounded step fails, do not treat the failure as noise.
+Capture it, inspect it, and decide whether the problem is in the task, the decomposition, the criteria, or the environment.
 
-### Purpose
+Useful questions include:
 
-Create a clean, reusable prompt workflow system for Codex work inside a repo.
+* Was the task too large?
+* Was the task poorly decomposed?
+* Were success criteria unclear or incomplete?
+* Did the task depend on hidden assumptions?
+* Was the failure caused by tooling or infrastructure rather than task difficulty?
+* Would a smaller or differently framed task have worked better?
+* Did the review happen too late?
+* Did retries produce new information, or only repeat cost?
 
-### Boundaries
+Useful outputs include:
 
-In now:
+* a short failure note or report
+* updated task boundaries
+* revised success criteria
+* a smaller follow-up slice
+* a decision to fix environment issues before retrying
 
-* prompt discovery
-* note discovery
-* status reconstruction
-* retry context preparation
-* conservative manual workflow
+A clean retry is often possible, but it should come after analysis rather than instead of it.
 
-Out for now:
+---
 
-* direct Codex execution
-* rich metadata system
-* dependency graphing
-* advanced logging
-* multi-repo orchestration
+## Resource and Cost Awareness
 
-### Components
+Projects benefit from tracking lightweight evidence about execution cost and quality.
+This does not need to be elaborate, but it should be enough to notice patterns.
 
-* `paths.py`
-* `prompts.py`
-* `notes.py`
-* `status.py`
-* later maybe `retry.py`
-* later maybe `cli.py`
+Examples of useful observations:
 
-### Artifacts
+* elapsed time
+* execution size
+* review effort
+* repeated retries
+* failure frequency
+* output quality
+* machine or service usage
 
-* architecture doc
-* workflow doc
-* prompt folder
-* notes folder
-* example prompts
-* minimal modules
-* validation notes
+In some environments, teams may also track:
 
-### Build order
-
-1. finalize architecture and workflow
-2. define V1 scope
-3. create folder layout
-4. implement `paths.py`
-5. implement `prompts.py`
-6. implement `notes.py`
-7. implement `status.py`
-8. manually validate status reconstruction
-9. add retry preparation
-10. add thin CLI only if needed
-
-### Validation
-
-* can we discover prompt files?
-* can we discover note files?
-* can we reconstruct current prompt status correctly?
-* can we identify next unrun?
-* can we pair a failed prompt with its most recent failed note?
+* token usage
+* compute time
+* API cost
+* artifact volume
 
-### Extension path
+The purpose is not accounting for its own sake.
+The purpose is to notice whether larger, more expensive, or less bounded tasks correlate with more failures, lower quality, or weaker reviewability.
 
-* structured note templates
-* dry-run bundle preparation
-* optional Codex adapter
-* dependency handling
-* reusable template packaging
+If those patterns appear, the workflow should respond by shrinking task size, clarifying criteria, or improving decomposition.
 
 ---
 
@@ -381,41 +445,50 @@
 
 ### 2. What is the source of truth?
 
-Filesystem only? Filenames only? Markdown contents too?
+What artifacts, interfaces, or records define reality for this system?
 
-### 3. What is the stable identity of a prompt?
+### 3. What is the stable identity of each important object?
 
-Filename? Internal ID? Both?
+Names, IDs, paths, interfaces, and records should not be ambiguous.
 
 ### 4. What states do we actually need in V1?
 
 Probably fewer than we are tempted to add.
 
-### 5. What should be manually controlled?
+### 5. What should remain manually controlled?
 
 Human judgment should stay in the loop where truth is fuzzy.
 
-### 6. What is the next thing we are intentionally not building?
+### 6. What bridge tooling is acceptable?
 
+If temporary helpers are needed, how do we keep them thin, inspectable, and easy to replace?
+
+### 7. What history should be preserved?
+
+What notes, outputs, validation records, or failure reports will be worth having later?
+
+### 8. What is the next thing we are intentionally not building?
+
 This keeps the scope fence visible.
 
 ---
 
 ## Design Workflow as a Reusable Template
 
-For future projects, a reliable progression is:
+For many projects, a reliable progression is:
 
 1. conversation
 2. problem statement
 3. goals
-4. scope / boundaries
+4. scope and boundaries
 5. architecture
-6. workflow / build plan
+6. workflow and build plan
 7. artifact list
 8. implementation sequence
 9. validation plan
-10. first build slice
-11. review / refinement
+10. bounded execution loop
+11. review and refinement
+12. failure analysis when needed
 
 This is slow in the beginning and fast later.
 It feels like more thought up front because it is.
@@ -423,22 +496,6 @@
 
 ---
 
-## What Comes Next for This Project
-
-The next practical step is not coding yet.
-The next step is to turn the architecture into a **focused V1 build plan**.
-
-That means deciding:
-
-* exact V1 features
-* exact folder/module layout
-* exact naming rules
-* exact validation steps
-
-Once those are fixed, implementation becomes much safer.
-
----
-
 ## Closing Thought
 
 A good design workflow does not slow building down.
@@ -450,6 +507,7 @@
 Name the parts.
 Set the boundaries.
 Choose the first slice.
+Review each iteration.
+Study failure instead of hiding it.
 Build only what is earned.
 Then move forward with confidence.
-

codex
Revised [Project_Design_Workflow.md](/data/git/handson-ml3/tools/Project_Design_Workflow.md) into a stronger project-agnostic version. The update keeps the phased design-before-build structure, but removes the repo-specific application section and adds explicit guidance for bounded iterative execution, review between iterations, bridge tooling, durable local history, failure analysis, and resource/cost awareness.

I also added a short companion note at [003_project_design_workflow_revision__SUCCESS__20260415_144244.md](/data/git/handson-ml3/notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md) summarizing what changed, why, the new reusable principles, and the remaining tradeoff that broader reuse means less project-specific concreteness.

No code tests were needed for this task.
diff --git a/notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md b/notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md
new file mode 100644
index 0000000000000000000000000000000000000000..8a0f0c6f5c6bfd6828fd2294aa3d091fb4f847d6
--- /dev/null
+++ b/notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md
@@ -0,0 +1,27 @@
+# Project Design Workflow Revision
+
+## What Changed
+
+Revised `tools/Project_Design_Workflow.md` into a more project-agnostic workflow document.
+Removed the project-specific application section and replaced it with reusable guidance.
+Added explicit sections for bounded iteration, failure analysis, resource and cost awareness, and durable local history.
+
+## Why These Changes Were Made
+
+The earlier version had a strong phased structure, but part of it was tied to one specific workflow and repo context.
+The revision keeps the design-first, boundary-aware approach while making the document reusable across different projects, tools, and maturity levels.
+
+## New Reusable Principles Added
+
+* bounded execution in small reviewable slices
+* review between iterations before issuing the next step
+* bridge tooling allowed when thin, inspectable, and subordinate
+* durable local notes, logs, outputs, and reports as project memory
+* failure triggering analysis rather than reflexive retry
+* lightweight observation of cost, time, retries, and failure patterns
+
+## Tradeoffs and Open Questions
+
+The document is now more broadly reusable, but less concrete for any one project.
+That tradeoff seems correct for a shared workflow template.
+If later needed, a separate project-specific companion doc could show how this workflow maps onto a particular repo without narrowing the core template.
diff --git a/tools/Project_Design_Workflow.md b/tools/Project_Design_Workflow.md
index 10aa2005ff8b68da1682930d6f4f2f81445e8150..812efe01ef35dc5dde789875c10cd96ceaca542a
--- a/tools/Project_Design_Workflow.md
+++ b/tools/Project_Design_Workflow.md
@@ -6,13 +6,13 @@
 
 This document is meant to sit beside architecture documents and help answer a practical question:
 
-**What are the steps between “this design makes sense” and “the thing works”?**
+**What are the steps between "this design makes sense" and "the thing works"?**
 
 ---
 
 ## Why This Exists
 
-Sometimes a project feels good at the architecture level but still feels fuzzy at the build level.
+Sometimes a project feels sound at the architecture level but still feels fuzzy at the build level.
 
 That happens because architecture is not the same thing as execution.
 
@@ -30,14 +30,15 @@
 * what to test first
 * what can wait
 * how to know whether the design is holding
+* how to recover cleanly when a step fails
 
 This workflow bridges that gap.
 
 ---
 
-## Relationship to the Thought Garden
+## Relationship to a Larger Design Process
 
-This workflow does **not** replace the Thought Garden.
+This workflow does **not** replace broader project thinking.
 It fits inside it.
 
 A useful mapping looks like this:
@@ -46,7 +47,7 @@
 
 What are we trying to do?
 What hurts right now?
-What would “better” feel like?
+What would "better" feel like?
 
 ### 2. Goals
 
@@ -60,9 +61,9 @@
 * architecture doc
 * workflow doc
 * module layout
-* prompt folder
+* interface sketch
 * notes folder
-* test prompt
+* validation checklist
 * first working slice
 
 ### 4. Tasks
@@ -73,26 +74,73 @@
 * define V1 scope
 * define module boundaries
 * choose naming conventions
-* write `paths.py`
-* test prompt discovery
-* write note parser
+* write first thin component
+* validate one path end to end
+* capture review notes
 
-### 5. Code / Libraries
+### 5. Implementation
 
 Only after the earlier layers are stable do we write implementation.
 
-So the current project is not outside the Thought Garden at all.
-It is just sitting in a very explicit design phase before code is allowed onto the stage.
+So this workflow is not separate from design.
+It is the part that turns design into bounded execution.
 
 ---
 
+## Core Principles
+
+### 1. Design before build
+
+Do not use implementation momentum to hide unclear thinking.
+
+### 2. Boundaries before breadth
+
+A project becomes safer when scope, responsibility, and deferral lines are visible.
+
+### 3. Thin slices before large pushes
+
+Prefer a bounded slice that proves something real over a large batch that creates many unknowns at once.
+
+### 4. Review between iterations
+
+Each bounded execution step should be inspected before the next one is issued.
+Do not assume a sequence is healthy just because the previous step completed.
+
+### 5. Validation is part of design
+
+Tests, inspections, examples, and acceptance checks are not extra work.
+They are how the design proves it is real.
+
+### 6. Bridge tooling is allowed, but subordinate
+
+Temporary or bridge tooling can be useful when it stays thin, inspectable, and reversible.
+It may help move work forward, gather evidence, or reduce manual friction.
+It should not become a substitute for architecture, clear interfaces, or good decomposition.
+
+### 7. Durable local history matters
+
+Notes, logs, reports, generated outputs, and similar local artifacts can serve as durable project memory.
+They help preserve decisions, validation results, failure patterns, and the reasoning behind refinements.
+
+### 8. Failure should produce analysis, not just retries
+
+When a step fails, the useful question is not only "how do we try again?"
+It is also "what did this failure reveal about the task, decomposition, criteria, or environment?"
+
+### 9. Resource use should be observed
+
+Project effort has a cost, whether that cost is human time, machine time, review effort, retries, execution size, or external spend.
+Large tasks should justify themselves.
+
+---
+
 ## The Core Sequence
 
 Here is the recommended design workflow.
 
 ### Phase 1: Clarify the problem
 
-This is the “why are we doing this?” phase.
+This is the "why are we doing this?" phase.
 
 Questions:
 
@@ -110,7 +158,7 @@
 
 ### Phase 2: Define the system boundary
 
-This is the “what belongs in this project and what does not?” phase.
+This is the "what belongs in this project and what does not?" phase.
 
 Questions:
 
@@ -128,11 +176,11 @@
 
 ### Phase 3: Draft the architecture
 
-This is the “what pieces exist and what are their jobs?” phase.
+This is the "what pieces exist and what are their jobs?" phase.
 
 Questions:
 
-* What modules/components are needed?
+* What modules, components, or services are needed?
 * What is each one responsible for?
 * What should each one never own?
 * Where are the seams between parts?
@@ -141,13 +189,13 @@
 
 * architecture doc
 * responsibility split
-* proposed file/folder layout
+* proposed file, package, or interface layout
 
 ---
 
 ### Phase 4: Identify the minimum viable slice
 
-This is the “what is the smallest useful thing we can build that proves the design?” phase.
+This is the "what is the smallest useful thing we can build that proves the design?" phase.
 
 Questions:
 
@@ -159,20 +207,21 @@
 Output:
 
 * V1 feature list
-* phase-by-phase build plan
+* initial build plan
 
 ---
 
 ### Phase 5: Define the artifacts
 
-This is the “what files or outputs must exist?” phase.
+This is the "what files or outputs must exist?" phase.
 
 Questions:
 
-* What markdown docs should exist?
-* What modules should exist?
-* What example prompts or test files should exist?
-* What outputs prove the pipe works?
+* What documents should exist?
+* What modules or interfaces should exist?
+* What examples, fixtures, or test inputs should exist?
+* What outputs prove the path works?
+* What notes, logs, or reports should be kept as durable project memory?
 
 Output:
 
@@ -183,13 +232,13 @@
 
 ### Phase 6: Sequence the work
 
-This is the “what order reduces pain and risk?” phase.
+This is the "what order reduces pain and risk?" phase.
 
 Questions:
 
 * What must come first because other things depend on it?
 * What can be tested independently?
-* What pieces should be proven before we add CLI or automation?
+* What pieces should be proven before automation or convenience layers are added?
 * What order keeps the build inspectable?
 
 Output:
@@ -201,57 +250,87 @@
 
 ### Phase 7: Define validation
 
-This is the “how will we know each layer works?” phase.
+This is the "how will we know each layer works?" phase.
 
 Questions:
 
-* What is the smoke test for each module?
+* What is the smoke test for each part?
 * What is a good manual test before automation?
 * What failure modes do we expect?
 * What evidence counts as success?
+* What review should happen before the next step begins?
 
 Output:
 
 * validation checklist
 * smoke tests
 * example inputs and outputs
+* review points between iterations
 
 ---
 
-### Phase 8: Build the first slice
+### Phase 8: Execute one bounded slice
 
 Only now do we begin implementation.
 
 Rules:
 
 * build one thin slice
-* test it
-* inspect it
-* adjust the design if reality disagrees
+* keep the task bounded and inspectable
+* use bridge tooling only when it remains thin and subordinate to the design
+* validate the slice
+* inspect the result before issuing the next step
 * do not sprint ahead because the first part felt good
 
 Output:
 
-* first working slice
-* notes about what the design got right/wrong
+* one completed slice
+* evidence of validation
+* notes about what the design got right or wrong
 
 ---
 
 ### Phase 9: Review and refine
 
-This is the “did the design survive contact with reality?” phase.
+This is the "did the design survive contact with reality?" phase.
 
 Questions:
 
 * What felt clean?
 * What felt awkward?
 * What assumptions broke?
-* What should be renamed, split, or deferred?
+* What should be renamed, split, simplified, or deferred?
+* What should change before the next bounded step?
 
 Output:
 
 * refinement notes
-* updated architecture if needed
+* updated architecture or scope if needed
+* revised next-step plan
+
+---
+
+## The Iteration Loop
+
+After the initial design work, many projects should move through a repeating bounded loop:
+
+1. clarify the next chunk
+2. define a bounded task
+3. state the success criteria
+4. execute
+5. validate and inspect results
+6. review what changed
+7. refine the plan
+8. issue the next bounded task only after review
+
+This loop should stay small enough that:
+
+* the task can be understood before execution
+* the result can be reviewed without guesswork
+* failure teaches something specific
+* refinement happens while context is still fresh
+
+If a project cannot explain the next chunk in a few clear sentences, the chunk is probably still too large.
 
 ---
 
@@ -287,87 +366,72 @@
 
 How do we test each step?
 
-### 8. Extension path
+### 8. Review points
+
+Where do we stop and inspect before continuing?
+
+### 9. Extension path
 
 What comes later, but not now?
 
-That pattern is portable and should work for many future repos.
+That pattern is portable and should work across many projects.
 
 ---
 
-## Applying This to the Codex Prompt Workflow
+## Failure Analysis as Part of the Workflow
 
-For the current project, the decomposition looks like this.
+When a bounded step fails, do not treat the failure as noise.
+Capture it, inspect it, and decide whether the problem is in the task, the decomposition, the criteria, or the environment.
 
-### Purpose
+Useful questions include:
 
-Create a clean, reusable prompt workflow system for Codex work inside a repo.
+* Was the task too large?
+* Was the task poorly decomposed?
+* Were success criteria unclear or incomplete?
+* Did the task depend on hidden assumptions?
+* Was the failure caused by tooling or infrastructure rather than task difficulty?
+* Would a smaller or differently framed task have worked better?
+* Did the review happen too late?
+* Did retries produce new information, or only repeat cost?
 
-### Boundaries
+Useful outputs include:
 
-In now:
+* a short failure note or report
+* updated task boundaries
+* revised success criteria
+* a smaller follow-up slice
+* a decision to fix environment issues before retrying
 
-* prompt discovery
-* note discovery
-* status reconstruction
-* retry context preparation
-* conservative manual workflow
+A clean retry is often possible, but it should come after analysis rather than instead of it.
 
-Out for now:
+---
 
-* direct Codex execution
-* rich metadata system
-* dependency graphing
-* advanced logging
-* multi-repo orchestration
+## Resource and Cost Awareness
 
-### Components
+Projects benefit from tracking lightweight evidence about execution cost and quality.
+This does not need to be elaborate, but it should be enough to notice patterns.
 
-* `paths.py`
-* `prompts.py`
-* `notes.py`
-* `status.py`
-* later maybe `retry.py`
-* later maybe `cli.py`
+Examples of useful observations:
 
-### Artifacts
+* elapsed time
+* execution size
+* review effort
+* repeated retries
+* failure frequency
+* output quality
+* machine or service usage
 
-* architecture doc
-* workflow doc
-* prompt folder
-* notes folder
-* example prompts
-* minimal modules
-* validation notes
+In some environments, teams may also track:
 
-### Build order
-
-1. finalize architecture and workflow
-2. define V1 scope
-3. create folder layout
-4. implement `paths.py`
-5. implement `prompts.py`
-6. implement `notes.py`
-7. implement `status.py`
-8. manually validate status reconstruction
-9. add retry preparation
-10. add thin CLI only if needed
-
-### Validation
-
-* can we discover prompt files?
-* can we discover note files?
-* can we reconstruct current prompt status correctly?
-* can we identify next unrun?
-* can we pair a failed prompt with its most recent failed note?
+* token usage
+* compute time
+* API cost
+* artifact volume
 
-### Extension path
+The purpose is not accounting for its own sake.
+The purpose is to notice whether larger, more expensive, or less bounded tasks correlate with more failures, lower quality, or weaker reviewability.
 
-* structured note templates
-* dry-run bundle preparation
-* optional Codex adapter
-* dependency handling
-* reusable template packaging
+If those patterns appear, the workflow should respond by shrinking task size, clarifying criteria, or improving decomposition.
 
 ---
 
@@ -381,41 +445,50 @@
 
 ### 2. What is the source of truth?
 
-Filesystem only? Filenames only? Markdown contents too?
+What artifacts, interfaces, or records define reality for this system?
 
-### 3. What is the stable identity of a prompt?
+### 3. What is the stable identity of each important object?
 
-Filename? Internal ID? Both?
+Names, IDs, paths, interfaces, and records should not be ambiguous.
 
 ### 4. What states do we actually need in V1?
 
 Probably fewer than we are tempted to add.
 
-### 5. What should be manually controlled?
+### 5. What should remain manually controlled?
 
 Human judgment should stay in the loop where truth is fuzzy.
 
-### 6. What is the next thing we are intentionally not building?
+### 6. What bridge tooling is acceptable?
 
+If temporary helpers are needed, how do we keep them thin, inspectable, and easy to replace?
+
+### 7. What history should be preserved?
+
+What notes, outputs, validation records, or failure reports will be worth having later?
+
+### 8. What is the next thing we are intentionally not building?
+
 This keeps the scope fence visible.
 
 ---
 
 ## Design Workflow as a Reusable Template
 
-For future projects, a reliable progression is:
+For many projects, a reliable progression is:
 
 1. conversation
 2. problem statement
 3. goals
-4. scope / boundaries
+4. scope and boundaries
 5. architecture
-6. workflow / build plan
+6. workflow and build plan
 7. artifact list
 8. implementation sequence
 9. validation plan
-10. first build slice
-11. review / refinement
+10. bounded execution loop
+11. review and refinement
+12. failure analysis when needed
 
 This is slow in the beginning and fast later.
 It feels like more thought up front because it is.
@@ -423,22 +496,6 @@
 
 ---
 
-## What Comes Next for This Project
-
-The next practical step is not coding yet.
-The next step is to turn the architecture into a **focused V1 build plan**.
-
-That means deciding:
-
-* exact V1 features
-* exact folder/module layout
-* exact naming rules
-* exact validation steps
-
-Once those are fixed, implementation becomes much safer.
-
----
-
 ## Closing Thought
 
 A good design workflow does not slow building down.
@@ -450,6 +507,7 @@
 Name the parts.
 Set the boundaries.
 Choose the first slice.
+Review each iteration.
+Study failure instead of hiding it.
 Build only what is earned.
 Then move forward with confidence.
-

tokens used
36,091
```
