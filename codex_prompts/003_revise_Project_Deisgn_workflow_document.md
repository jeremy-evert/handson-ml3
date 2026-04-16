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
