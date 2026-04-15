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
