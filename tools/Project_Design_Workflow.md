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
