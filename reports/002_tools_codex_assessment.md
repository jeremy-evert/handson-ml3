# Assessment: tools/codex
**Prompt:** 002_repo_and_tooling_recon  
**Date:** 2026-04-17

---

## Part A — Orientation Summary

`tools/Codex_Prompt_Workflow_Architecture.md` defines a minimal V1 workflow for executing a single prompt, writing a durable execution record to `notes/`, stopping for human review, and only releasing the next prompt after explicit `ACCEPTED` status. It emphasizes four core roles: execution, record, review gate, and conservative queue progression — and explicitly defers everything broader.

`tools/Project_Design_Workflow.md` is a meta-process document: design before build, thin slices, review between iterations, study failure rather than hide it. It's the governing philosophy behind the codex tooling and applies equally to any future runner work in this repo.

---

## Part B — Script-by-Script Assessment

---

### `run_prompt.py` — Current V1 Runner

**What it does:** Executes one prompt file through `codex exec`, writes a durable V1 execution record to `notes/`.

**Input:** A prompt filename, numeric prefix, or path argument. Looks up the file in `codex_prompts/` using a multi-step resolver (absolute → direct → codex_prompts/ prefix → glob match).

**Output:** A single markdown file at `notes/{prompt_stem}__{started_at_utc}.md`. If a same-second collision exists, appends `__2`, `__3`, etc. Prints the written path to stdout and exits with the subprocess return code.

**External state:** Requires `codex_prompts/` and `notes/` directories to exist. Calls `codex exec -C {root} --output-last-message {tmpfile} -` as a subprocess.

**Record format:** Fully V1-compliant. Sections: identity → Execution Facts → Review Facts → Failure Analysis → Resource / Cost Facts → Prompt Text → Codex Final Output → Stderr. Initializes `review_status: UNREVIEWED`. Calls `validate_record_text()` on the written file.

**Status: Functional and clean.** This is the right script to run. Well-structured, validates its own output.

---

### `review_run.py` — Review Write-Back

**What it does:** Accepts a review decision and writes it back into an existing V1 execution record.

**Input:** Path to an existing `notes/*.md` V1 record + `--review-status ACCEPTED|REJECTED` + `--review-summary "..."` + optional `--reviewed-by`, `--reviewed-at-utc`. For `REJECTED`, also accepts `--failure-type`, `--failure-symptom`, `--likely-cause`, `--recommended-next-action`.

**Output:** Modifies the target file in place. Uses regex field replacement, validates before and after writing.

**External state:** Must be run against a file that exists under `notes/` (enforced). Requires the file to be a valid V1 record.

**Status: Functional and clean.** Correctly enforces that rejection-only fields cannot be set on an ACCEPTED review.

---

### `check_queue_readiness.py` — Queue Gate Check

**What it does:** Reports whether the next prompt in the codex_prompts queue is ready to run, based on whether the immediately previous prompt's latest V1 record is ACCEPTED.

**Input:** `codex_prompts/*.md` (discovers all prompts, sorts by numeric prefix) + `notes/*.md` (parses V1 records only, skips non-V1 files). Accepts `--prompt` to check a specific prompt; otherwise defaults to the first prompt without an ACCEPTED latest record.

**Output:** Human-readable summary: ordered prompt list, target prompt, previous prompt, latest record status, ready/not-ready verdict.

**External state:** Reads `codex_prompts/` and `notes/`. Calls `v1_record_validation.parse_record_file()` on every markdown file in `notes/`.

**Status: CURRENTLY BROKEN.** `codex_prompts/` contains two files with prefix `022`:
- `022_audit_and_classify_scaffolding_for_summary_and_attic.md`  
- `022_generate_repo_lay_of_the_land_summary.md`

`discover_prompts()` raises `ReadinessError` on duplicate prefixes and the script exits with an error. Verified:
```
ERROR: multiple prompt files share the same numeric prefix: ...022_audit... and ...022_generate...
```
One of these prompts needs to be renumbered before `check_queue_readiness.py` can be used again.

---

### `list_review_backlog.py` — Backlog Inspector

**What it does:** Reports all UNREVIEWED V1 records in notes/, grouped by record and by latest-per-prompt.

**Input:** `notes/*.md`. Parses V1 records only; silently skips non-V1 files.

**Output:** Summary: total discovered, unreviewed count, latest-per-prompt list, "likely needs human review next" list.

**Status: Functional.** Does not read `codex_prompts/`, so unaffected by the duplicate-022 prefix bug. Current state as of 2026-04-17: 14 V1 records discovered, 13 UNREVIEWED, 1 ACCEPTED (the second run of `001_smoke_test_pipeline`).

---

### `v1_record_validation.py` — Shared Validation Library

**What it does:** Provides the shared contract for V1 execution records: `validate_record_text()`, `parse_record_file()`, `V1Record` dataclass, `looks_like_v1_record()`.

**Input:** Text string or file path.

**Output:** `V1Record` dataclass or raises `ValidationError`.

**External state:** None. Pure logic module.

**Key behaviors:**
- `looks_like_v1_record()` is a cheap pre-filter: checks for `run_id` field OR both "## Execution Facts" and "## Review Facts" sections. Files that fail this check are silently skipped by the other tools — important for the many non-V1 markdown files in `notes/`.
- `validate_record_text()` checks sections exist and are in order, all 23 required fields exist, timestamps are valid, `execution_status` and `review_status` are in known sets, `prompt_file/prompt_stem` match, filename matches `run_id` (when given a Path).

**Status: Functional.** This is the load-bearing shared module. All four scripts depend on it.

**One gap worth noting:** The `V1_Execution_Record_Artifact.md` spec lists `reviewed_by` and `reviewed_at_utc` as optional, but `REQUIRED_FIELDS` in this module lists them as required. In practice this works because `run_prompt.py` initializes them as blank placeholder lines — the validator parses them as empty strings, which passes. Not a bug, but a slight tension between spec and implementation.

---

### `baby_run_prompt.py` — Pre-V1 Runner (Historical)

**What it does:** The original runner, before V1 execution records existed. Executes one prompt through `codex exec`, writes a simpler note in the legacy format.

**Input:** Same prompt argument as `run_prompt.py`.

**Output:** A markdown file at `notes/{prompt_stem}__{STATUS}__{timestamp}.md` where STATUS is `SUCCESS` or `FAILED`. Note format: identity header → Original Prompt → Codex Output → Notes (stderr if present).

**External state:** Same directory requirements as `run_prompt.py`. Does NOT import or use `v1_record_validation`.

**Difference from `run_prompt.py`:**
- Output filename encodes success/failure in the name (the old format the architecture docs explicitly say to stop using)
- No V1 sections, no `review_status`, no `run_id` field
- No validation of the written record
- Output is NOT recognized by `check_queue_readiness.py` or `list_review_backlog.py` (fails `looks_like_v1_record()`)

**Status: Functional but obsolete.** Should not be used for new runs. Its output is invisible to the V1 tooling. Safe to retain as reference but should not be on any PATH or mentioned in workflow docs as a current tool.

---

### V1 Markdown Documents Assessment

#### `V1_Execution_Record_Artifact.md`
**Status: Live reference, well-aligned.** Defines the record contract. Implementation in `run_prompt.py` and `v1_record_validation.py` closely matches. The minor `reviewed_by`/`reviewed_at_utc` optional-vs-required tension noted above is the only delta.

#### `V1_Run_Review_Gate.md`
**Status: Live reference, well-aligned.** Defines the three-state review gate (UNREVIEWED → ACCEPTED|REJECTED) and the queue-progression rule. Implemented faithfully by `review_run.py` and `check_queue_readiness.py`.

#### `V1_Bridge_Runner_Change_Spec.md`
**Status: Historical artifact, no longer actionable.** This document was a change spec for what `run_prompt.py` needed to become. All changes it described have been implemented. The spec section titled "Required V1 Changes" now describes the current state, not future work. Its remaining value is as a retrospective explanation of why the V1 format looks the way it does. Could be archived or relabeled to avoid confusion.

---

## Specific Questions

### 1. End-to-End Workflow

A complete prompt execution cycle:

1. **Human** creates or confirms `codex_prompts/{NNN}_{name}.md` exists
2. **`check_queue_readiness.py`** — confirms previous prompt's latest V1 record is ACCEPTED (currently broken due to duplicate-022 prefix; fix by renumbering one prompt)
3. **`run_prompt.py {NNN}`** — resolves prompt path, calls `codex exec`, writes `notes/{prompt_stem}__{started_at_utc}.md` with `review_status: UNREVIEWED`, prints path, exits with subprocess return code
4. **Human** opens the written execution record in `notes/`, reads Codex output, checks the repo for actual changes, applies the review checklist from `V1_Run_Review_Gate.md`
5. **`review_run.py {record_path} --review-status ACCEPTED --review-summary "..."`** — writes review fields back into the record
6. **`list_review_backlog.py`** — confirms backlog is clear / shows remaining UNREVIEWED records
7. **`check_queue_readiness.py`** — confirms next prompt is now ready
8. Repeat from step 2

---

### 2. What Is Working Well

- **The V1 record format** is clean, human-readable, and inspectable. Sections are ordered logically.
- **`v1_record_validation.py`** as a shared library keeps the contract in one place. The `looks_like_v1_record()` pre-filter is smart — it prevents non-V1 files from crashing the tools without requiring a curated file list.
- **`run_prompt.py`** validates its own output immediately after writing. This means a malformed record is caught at write time, not discovered hours later during review.
- **`review_run.py`** validates before AND after writing. Double-validation is the right call for a write-back operation.
- **The execution/review status separation** is well-enforced. `EXECUTED` does not mean `ACCEPTED`. This is the core design principle and it holds.

---

### 3. What Is Problematic

1. **Duplicate prefix 022 in `codex_prompts/`** — `check_queue_readiness.py` is currently broken. Fix: renumber one of `022_audit_and_classify_scaffolding_for_summary_and_attic.md` or `022_generate_repo_lay_of_the_land_summary.md`.

2. **`baby_run_prompt.py` still present** — No documentation labels it obsolete. A new developer could run it thinking it's the current tool. It produces output invisible to V1 tooling. Should be clearly marked or removed.

3. **`V1_Bridge_Runner_Change_Spec.md` title is misleading** — "Change Spec" implies future work. All changes are done. Should be retitled or archived.

4. **notes/ contains 53 files across three incompatible naming conventions:**
   - 14 V1 records (`{stem}__{timestamp}.md`) — recognized by V1 tools
   - 14 legacy SUCCESS/FAILED notes (`{stem}__SUCCESS__{timestamp}.md`) — invisible to V1 tools
   - 25 free-form output notes (various stems that don't match prompt stems, e.g., `004_architecture_and_bridge_runner_review__20260415_195538.md`) — invisible to V1 tools
   
   The 39 non-V1 files are silently skipped, not errors. But they create confusion about what "has been done."

5. **Prompts 002–010 have no V1 execution records.** They have legacy notes from `baby_run_prompt.py`, so the V1 tooling treats them as UNRUN. `check_queue_readiness.py` (once the 022 issue is fixed) would report them as missing V1 evidence and block the queue. The check_queue_readiness.py code does detect legacy SUCCESS notes and emits a gap explanation, but this creates a confusing state.

6. **Stem mismatch in notes/:** `003_project_design_workflow_revision__SUCCESS__20260415_144244.md` does not match prompt `003_revise_Project_Deisgn_workflow_document.md`. The prompt was apparently renamed after the note was written. Minor — doesn't break anything — but represents drift between note history and current prompt names.

---

### 4. `baby_run_prompt.py` vs `run_prompt.py`

`baby_run_prompt.py` is the pre-V1 runner. It writes `notes/{stem}__{SUCCESS|FAILED}__{timestamp}.md` with a simple 4-section note body (no `run_id`, no `review_status`, no execution record structure). It does not call `v1_record_validation`. Its output is completely invisible to `check_queue_readiness.py` and `list_review_backlog.py`.

`run_prompt.py` is the current V1 runner. Same CLI interface. Writes `notes/{stem}__{timestamp}.md` with the full V1 execution record. Calls validation after writing. Recognized by all V1 tools.

They are parallel implementations of the same interface at different epochs of the project. `baby_run_prompt.py` should be clearly deprecated.

---

### 5. The V1_*.md Files

**`V1_Execution_Record_Artifact.md`** — Living reference document. Implementation matches it. Worth keeping current.

**`V1_Run_Review_Gate.md`** — Living reference document. Implementation matches it. Worth keeping current.

**`V1_Bridge_Runner_Change_Spec.md`** — Historical artifact. The changes it described are done. It now documents what was built, not what needs to be built. Confusingly titled for a developer who encounters it fresh.

---

### 6. `codex_prompts/` → `notes/` Naming

**The convention:** prompts are named `{NNN}_{description}.md`; V1 records are named `{prompt_stem}__{started_at_utc}.md`. This means a V1 record for prompt `014_build_queue_readiness_checker.md` lives at `notes/014_build_queue_readiness_checker__{timestamp}.md`. The mapping is: prompt stem = note file prefix = `prompt_stem` field in the record body. Clean and unambiguous.

**The messiness:** 39 of 53 files in `notes/` don't follow V1 convention (legacy format + free-form). The V1 tooling silently skips them (correct behavior), but the directory looks chaotic.

**Prompts with no V1 records:** 002–010 (have legacy notes only), `022_generate_repo_lay_of_the_land_summary.md` (has no notes at all — only `022_audit_*` has a V1 record).

**Orphaned V1 records:** None. All 14 V1 records match their corresponding prompt files.

---

### 7. `v1_record_validation.py` — Is It Actually Called?

Yes. Called in three places:

- `run_prompt.py` line 13: `from v1_record_validation import validate_record_text` — calls `validate_record_text()` on the written record immediately after writing (line 219)
- `review_run.py` lines 11-12: imports `REVIEW_STATUS_VALUES` and `validate_record_text` — calls it before and after writing review fields (lines 142, 148)
- `check_queue_readiness.py` line 11: `from v1_record_validation import V1Record, ValidationError, parse_record_file` — calls `parse_record_file()` on every `.md` in notes/ (line 136)
- `list_review_backlog.py` line 9: same import pattern as check_queue_readiness — calls `parse_record_file()` on every `.md` in notes/ (line 50)

It is not "just sitting there." It is the shared backbone of the entire V1 pipeline.
