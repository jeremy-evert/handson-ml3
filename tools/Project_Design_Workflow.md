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

