# Stage 1 Prompt-Generation Prompt Handoff

## Why this is next

The MVP boundary and the minimal scope/state/resume contract are already set. The next unresolved seam is the Stage 1 prompt-generation contract itself: what evidence must exist before generating a prompt, what one prompt may target, and what safety and output rules must bound that prompt.

## Design surface covered

This prompt is limited to Stage 1 chapter-intro prompt generation. It defines required inputs, scanner preconditions, insert-versus-replace decisions, prompt output expectations, notebook-safety rules, and the split between wrapper-local targeting facts and canonical V1 truth in `notes/`.

## Intentionally deferred

It defers Stage 2, Stage 3, orchestration behavior, run-to-record linkage details, and all Python implementation work.
