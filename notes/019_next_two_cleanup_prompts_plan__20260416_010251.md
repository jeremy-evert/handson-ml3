# 019 Next Two Cleanup Prompts Plan

These two prompts were chosen directly from the architecture sweep and prioritized remaining-work note because they are the highest-payoff low-risk items still left in the current V1 workflow.

They are ordered this way because doc/spec alignment should happen first. The design packet still contains stale references to `baby_run_prompt.py`, outdated assumptions about missing helpers, and unsuffixed run-id wording. Cleaning that up first makes the repo's written guidance match the actual V1 toolset before adding more implementation hardening.

The first prompt to execute should be `codex_prompts/020_align_v1_doc_and_spec_packet_to_actual_toolset.md`. After that, execute `codex_prompts/021_add_lightweight_v1_record_contract_validation.md` to protect the shared markdown execution-record shape that the current tools already depend on.

What should wait until later:

- operational guidance cleanup for legacy `__SUCCESS__` notes and the current review backlog
- retry-linkage tooling around `retry_of_run_id`
- richer queue states, scheduling, or orchestration
- broader runner refactors or any platform-style expansion beyond the current V1 slice
