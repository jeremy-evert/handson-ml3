# 022 Scaffolding Classification Report

## Short Summary

`notes/` currently mixes four different things:

- legacy pre-V1 success receipts
- durable design and implementation notes
- current V1 execution records that the live helpers still read
- small planning / recommendation notes that were useful during construction but are now mostly superseded

`codex_prompts/` is almost entirely construction scaffolding for building the current V1 toolset. The durable knowledge from those prompts now lives in the tool/docs packet and in a smaller set of implementation and review notes.

The main conservative rule in this report is:

- current V1 execution-record files stay out of attic decisions for now, because moving them would change the behavior of `tools/codex/check_queue_readiness.py` and `tools/codex/list_review_backlog.py`

## Classification Criteria Used

- `permanent residents`: still useful as standing maintainer context in their current form
- `summarize, then move to attic`: contains durable decisions, validation evidence, or design rationale, but does not need to stay in `notes/` once that knowledge is extracted
- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
- `uncertain / needs human review`: not safe to move yet because it is still operationally active, still part of the current source-of-truth set, or still too close to the active cleanup pass

## Permanent Residents

- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
  Reason: best compact repo-specific statement of what the V1 system currently is, what remains missing, and whether it is ready for real use.
- `notes/020_doc_spec_alignment_cleanup__20260416_010534.md`
  Reason: explains why the current design/spec packet changed and records the final doc-alignment decisions that still govern the written packet.
- `notes/021_record_contract_validation__20260416_011314.md`
  Reason: durable contract and validation evidence for the shared V1 execution-record parser used by live tools.

## Summarize, Then Move To Attic

- `notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md`
  Reason: useful initial repo-context snapshot, but stale as a standing note.
- `notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md`
  Reason: explains why `tools/Project_Design_Workflow.md` was revised and what reusable principles were added.
- `notes/004_architecture_and_bridge_runner_review__20260415_195538.md`
  Reason: foundational mismatch review that drove the V1 execution-record and review-gate design.
- `notes/005_prompt_queue_plan__20260415_202557.md`
  Reason: captures why the early V1 sequence was ordered policy first, architecture second, runner change last.
- `notes/009_run_prompt_candidate_build__20260415_233407.md`
  Reason: records what behavior `run_prompt.py` intentionally preserved and what changed for V1 records.
- `notes/010_run_prompt_candidate_review__20260415_234559.md`
  Reason: captures the first serious assessment of the new runner and the environment-vs-runner distinction.
- `notes/011_review_writeback_helper_build__20260415_235514.md`
  Reason: implementation and validation note for the still-live `review_run.py` helper.
- `notes/012_v1_pipeline_options_review__20260416_000819.md`
  Reason: useful maturity review and bounded option set before operational helpers were added.
- `notes/014_queue_readiness_checker_build__20260416_002419.md`
  Reason: implementation and validation note for the still-live readiness helper.
- `notes/015_review_backlog_lister_build__20260416_010500.md`
  Reason: implementation and validation note for the still-live review backlog helper.
- `notes/016_queue_and_backlog_helper_validation__20260416_003710.md`
  Reason: repo-specific validation evidence showing how readiness and backlog behavior interact with mixed legacy and V1 notes.
- `notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md`
  Reason: records the rationale for the current `Queue note:` behavior in the readiness helper.

## Move To Attic Without Summary

### notes/

- `notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md`
  Reason: legacy smoke-test receipt; superseded by later V1 records and duplicates.
- `notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md`
  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
- `notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md`
  Reason: duplicate legacy smoke-test receipt with no new durable decision.
- `notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md`
  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
- `notes/003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md`
  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
- `notes/004_next_design_step_recommendation__20260415_195538.md`
  Reason: brief recommendation fully subsumed by the main review and later execution-record work.
- `notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md`
  Reason: success receipt; the main review note is the durable artifact.
- `notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md`
  Reason: success receipt; resulting doc and queue-plan note carry the lasting value.
- `notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md`
  Reason: success receipt; durable knowledge lives in `tools/codex/V1_Run_Review_Gate.md`.
- `notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md`
  Reason: success receipt; durable knowledge lives in the architecture doc.
- `notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md`
  Reason: success receipt; durable knowledge lives in the bridge-runner spec.
- `notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md`
  Reason: implementation receipt; the build note and the code itself carry the durable content.
- `notes/010_next_step_recommendation__20260415_234559.md`
  Reason: brief next-step recommendation that was quickly superseded by the actual sequence taken.
- `notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md`
  Reason: success receipt; the review note carries the useful content.
- `notes/012_top_three_next_options__20260416_000819.md`
  Reason: short ranking note largely duplicated by the fuller options review and later implementation history.
- `notes/013_next_two_prompt_plan__20260416_002005.md`
  Reason: prompt-generation plan already realized by `014` and `015`.
- `notes/016_next_improvement_recommendation__20260416_003710.md`
  Reason: one-line recommendation already realized by `017`.
- `notes/018_prioritized_remaining_work__20260416_005130.md`
  Reason: concise ranking duplicated by the fuller architecture sweep and the implemented follow-up work.
- `notes/019_next_two_cleanup_prompts_plan__20260416_010251.md`
  Reason: prompt-generation plan already realized by `020` and `021`.

### codex_prompts/

- `codex_prompts/001_smoke_test_pipeline.md`
  Reason: one-use bootstrap prompt; durable outcome was only the existence of a working notes path.
- `codex_prompts/002_repo_inventory_and_status.md`
  Reason: one-use repo-inventory scaffold; durable value is in the resulting note, not the prompt text.
- `codex_prompts/003_revise_Project_Deisgn_workflow_document.md`
  Reason: implementation instruction for a completed doc revision; resulting document and revision note now carry the knowledge.
- `codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md`
  Reason: review scaffold whose durable output is the architecture/runner assessment note.
- `codex_prompts/005_define_execution_record_and_generate_next_prompt_queue.md`
  Reason: queue-generation scaffold; durable outputs are the execution-record doc and queue-plan note.
- `codex_prompts/006_define_v1_run_review_gate.md`
  Reason: design scaffold whose durable output is `tools/codex/V1_Run_Review_Gate.md`.
- `codex_prompts/007_align_architecture_doc_to_v1_workflow.md`
  Reason: alignment scaffold whose durable output is the revised architecture doc.
- `codex_prompts/008_define_minimal_bridge_runner_change_spec.md`
  Reason: spec-writing scaffold whose durable output is the bridge-runner spec.
- `codex_prompts/009_build_v1_run_prompt_candidate.md`
  Reason: implementation prompt for completed runner work; code and build note now carry the value.
- `codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md`
  Reason: review scaffold whose durable output is the runner-review note.
- `codex_prompts/011_build_v1_review_writeback_helper.md`
  Reason: implementation prompt for completed helper work; code and build note now carry the value.
- `codex_prompts/012_review_v1_pipeline_and_recommend_next_options.md`
  Reason: review scaffold whose durable output is the options review note.
- `codex_prompts/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md`
  Reason: prompt-generation scaffold already realized by later prompt files and notes.
- `codex_prompts/014_build_queue_readiness_checker.md`
  Reason: implementation prompt for completed helper work; code and build note now carry the value.
- `codex_prompts/015_build_review_backlog_unreviewed_run_lister.md`
  Reason: implementation prompt for completed helper work; code and build note now carry the value.
- `codex_prompts/016_validate_queue_and_backlog_helpers_against_current_repo.md`
  Reason: validation scaffold whose durable content is in the validation note.
- `codex_prompts/017_polish_queue_readiness_gap_explanation.md`
  Reason: small polish prompt for completed helper behavior; code and polish note now carry the value.
- `codex_prompts/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work.md`
  Reason: review scaffold whose durable output is the architecture sweep note.
- `codex_prompts/019_generate_prompts_for_doc_alignment_and_record_contract_validation.md`
  Reason: prompt-generation scaffold already realized by `020` and `021`.
- `codex_prompts/020_align_v1_doc_and_spec_packet_to_actual_toolset.md`
  Reason: implementation prompt for completed doc cleanup; the edited docs and cleanup note now carry the value.
- `codex_prompts/021_add_lightweight_v1_record_contract_validation.md`
  Reason: implementation prompt for completed validator work; code and validation note now carry the value.

## Uncertain / Needs Human Review

### notes/

- `notes/001_smoke_test_pipeline__20260415_233343.md`
  Reason: current V1 execution record; moving it changes live run history and helper output.
- `notes/001_smoke_test_pipeline__20260415_234918.md`
  Reason: current V1 execution record and current accepted prior evidence for prompt `001`.
- `notes/011_build_v1_review_writeback_helper__20260415_235346.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/014_build_queue_readiness_checker__20260416_002319.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work__20260416_005034.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/020_align_v1_doc_and_spec_packet_to_actual_toolset__20260416_010453.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.
- `notes/021_add_lightweight_v1_record_contract_validation__20260416_010810.md`
  Reason: current V1 execution record; still part of backlog/readiness evidence.

### codex_prompts/

- `codex_prompts/022_audit_and_classify_scaffolding_for_summary_and_attic.md`
  Reason: current audit prompt for the cleanup sequence; do not classify as attic material until this report has been acted on and superseded.

## Practical Read Of The Repo

- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
- The second cleanup target is the durable-but-noisy implementation notes, but only after their key rationale is extracted into a smaller standing summary.
- The current V1 execution records should not move in a cleanup pass unless the repo first decides how the readiness/backlog tools should treat archived records.
