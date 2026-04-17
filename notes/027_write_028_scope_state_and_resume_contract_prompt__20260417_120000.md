# Next Wrapper Prompt Note

This is the right next prompt because the MVP boundary is already defined, but the smallest concrete contract for wrapper scope, wrapper-local state, and resume behavior is still unresolved. That seam has to be settled before any scan layer, prompt-generation rules, or orchestration logic can be designed without drifting into a second queue system.

This prompt covers only the minimal design surface needed to answer how one bounded Stage 1 wrapper run selects notebooks, tracks its current position, and resumes safely while keeping `notes/` V1 records as the canonical execution and review truth. It intentionally defers orchestration-loop behavior, stage-specific mutation rules, notebook-write contracts, retry policy expansion, and implementation.
