# Stage 1 Prompt-Generation Contract

## Executive Summary
Stage 1 prompt generation is a narrow scanner-driven contract for one chapter-intro treatment decision on one notebook. A Stage 1 prompt may be generated only after `tools/notebook_enricher/notebook_scanner.py` has already determined that Stage 1 is needed, classified the current intro state, and identified the exact eligible insert-or-replace position. The prompt itself stays bounded to one notebook mutation, preserves notebook validity, and remains subordinate to the existing runner-centered V1 workflow.

## Why This Design Surface Comes Next

`notes/025_define_staged_notebook_wrapper_mvp_contract__20260417_175751.md` fixed the Stage 1-first MVP boundary, and `notes/028_minimal_wrapper_scope_state_and_resume_contract__20260417_190146.md` fixed the minimal scope/state/resume seam. The next unresolved surface is narrower: what concrete facts must exist before a Stage 1 prompt is even allowed to exist.

This comes next because the repo already separates:

- execution and review truth, which remain in V1 records under `notes/`
- scanner-derived notebook evidence, which already exists in `tools/notebook_enricher/notebook_scanner.py`
- prompt execution, which remains runner-centered around `tools/codex/run_prompt.py`

Without a Stage 1 prompt-generation contract, later build work would be free to let prompt generation drift into open-ended notebook discovery, broad notebook rewriting, or wrapper-local redefinition of canonical truth. This note prevents that drift before any implementation starts.

## Required Inputs For One Stage 1 Prompt

One Stage 1 prompt may be generated only when these minimum concrete inputs are already available for the targeted notebook:

- the target notebook path
- the fixed active stage identity, which must be Stage 1 and nothing else
- the scanner-derived `chapter_intro_status`
- the scanner-derived `chapter_intro_index`
- the scanner-derived `setup_cell_index`
- the scanner-derived Stage 1 need decision
- the scanner-derived intro action decision: `insert`, `replace`, or `skip`
- enough notebook identity context to keep the prompt bounded to one notebook, specifically the notebook stem and total cell count or equivalent bounded notebook-size context already available from the scanner inventory

For this repo, those inputs should align with the existing `NotebookInventory` surface in `tools/notebook_enricher/notebook_scanner.py`, which already provides:

- `path`
- `notebook_stem`
- `total_cells`
- `chapter_intro_status`
- `chapter_intro_index`
- `setup_cell_index`
- `needs_stage1()`

Stage 1 prompt generation does not require CLI syntax, wrapper state format, or Python type decisions in this pass. It requires only the concrete targeting facts above.

## Scanner Preconditions

Stage 1 prompt generation is allowed to assume scanner-first logic has already determined all notebook-discovery questions needed for the Stage 1 decision. Prompt generation is not the place where notebook discovery happens.

Before a Stage 1 prompt may be generated, the scanner must already determine:

- whether Stage 1 is needed at all
- whether the existing chapter intro is `missing`, `heading`, `thin`, or `substantive`
- whether the eligible Stage 1 action is `insert`, `replace`, or `skip`
- what exact notebook position is eligible for the Stage 1 change
- whether the notebook has a `# Setup` boundary that constrains where an inserted intro may go

For this repo, the scanner evidence is sufficient when it yields these conclusions:

- `chapter_intro_status: substantive` means no Stage 1 prompt should be generated
- `chapter_intro_status: missing` means insertion is eligible at the scanner-approved intro position before the setup boundary when present
- `chapter_intro_status: heading` means replacement is eligible only for the scanner-identified intro cell
- `chapter_intro_status: thin` means replacement is eligible only for the scanner-identified intro cell

Evidence is sufficient to avoid generating a Stage 1 prompt when either of these is true:

- the scanner says Stage 1 is not needed because the intro is already `substantive`
- the scanner cannot determine a single eligible bounded insert-or-replace target position, in which case Stage 1 prompt generation must stop rather than broaden the task

The prompt generator must not perform open-ended reasoning such as:

- reinterpreting whether another markdown cell elsewhere might be a better intro candidate
- deciding to rewrite multiple front-matter cells
- discovering Stage 2 or Stage 3 work while preparing a Stage 1 prompt

## Allowed Stage 1 Target Unit

The smallest practical Stage 1 target unit is:

- exactly one notebook
- exactly one Stage 1 chapter-intro treatment decision
- exactly one bounded intro mutation opportunity within that notebook

One Stage 1 prompt must not become:

- a broad notebook rewrite
- a front-matter cleanup pass
- a mixed Stage 1 plus Stage 2 or Stage 3 treatment prompt
- a notebook-wide style or narrative improvement pass

For this contract, one prompt targets one notebook and resolves only one of these outcomes:

- insert one new chapter-intro markdown cell
- replace one existing intro markdown cell
- no prompt at all because Stage 1 is unnecessary or not safely targetable

## Allowed Intro Decisions

Stage 1 allows only these intro decisions:

- `insert` when the scanner classifies the intro as `missing` and identifies a single eligible insertion position near the front of the notebook, bounded by the current intro-detection window and the setup boundary
- `replace` when the scanner classifies the existing intro as `heading` or `thin` and identifies the exact existing intro cell index eligible for replacement
- `skip` or no prompt when the scanner classifies the intro as `substantive`
- `skip` or no prompt when scanner evidence is not strong enough to support one deterministic bounded insert-or-replace action

Stage 1 does not authorize:

- replacing more than one cell
- inserting more than one new markdown cell
- rewriting neighboring markdown cells for consistency
- cleaning up title cells, notebook-description cells, or HTML link tables
- moving the setup cell
- reformatting markdown outside the single targeted intro treatment

This means Stage 1 is a chapter-intro decision only. It is not a license for opportunistic cleanup before, around, or after that decision.

## Stage 1 Prompt Output Contract

Every Stage 1 prompt generated under this contract must enforce the same bounded output shape.

The prompt must require:

- one bounded notebook mutation only
- reading the full source notebook artifact for the targeted notebook
- writing the complete modified notebook artifact, not partial notebook fragments
- preserving valid Jupyter notebook structure
- preserving all non-targeted cells in place
- preserving code-cell source, outputs, and metadata unchanged
- keeping the mutation reviewable as one bounded V1 step

The prompt must prohibit:

- direct code-cell edits
- unrelated markdown rewrites
- unrelated cell insertion or deletion
- cell reordering
- notebook-wide metadata churn
- broad front-matter rewrites
- Stage 2 or Stage 3 edits

At a contract level, the prompt may direct output to a temporary modified notebook artifact rather than in-place mutation, but this note does not define the implementation path. What is canonical here is that the Stage 1 prompt must require a complete modified notebook artifact that is valid and reviewable as one bounded run result.

## Mandatory Notebook-Safety Rules

These notebook-safety rules are non-negotiable for Stage 1 prompt generation in this repo:

- no direct code-cell edits
- no broad notebook rewrites
- no Stage 2 or Stage 3 edits
- no opportunistic style cleanup
- deterministic targeting before mutation
- bounded insert-or-replace action only
- no silent progression past the V1 review gate
- no mutation prompt when scanner evidence does not support one exact bounded target
- no metadata churn unrelated to the one Stage 1 intro treatment

The safety boundary is therefore:

- scanner first
- one notebook
- one intro decision
- one bounded mutation
- one V1-reviewed run

This keeps Stage 1 compatible with the current review gate in `tools/codex/V1_Run_Review_Gate.md` and the single-run execution-record model in `tools/codex/V1_Execution_Record_Artifact.md`.

## Local Prompt-Generation Facts Versus Canonical Truth

Stage 1 prompt generation may use wrapper-local targeting facts as convenience inputs. For example, wrapper-local state may remember:

- which notebook is currently being considered
- that the active stage is Stage 1
- the scanner-derived intro status and target position used to generate the prompt

But those local prompt-generation facts are not canonical execution truth.

What must remain canonical in `notes/` is:

- whether a Stage 1 run actually happened
- what prompt was executed
- what execution outcome was recorded
- what review decision was recorded
- whether the run remains `UNREVIEWED`, was `ACCEPTED`, or was `REJECTED`

Stage 1 prompt generation must not:

- redefine queue readiness
- replace the V1 review gate with wrapper-local state
- treat prompt-generation facts as proof that execution or review occurred
- compete with `tools/codex/check_queue_readiness.py` or `tools/codex/list_review_backlog.py` as the source of progression truth

The design remains runner-centered:

- prompt generation may be wrapper-local
- execution and review truth remain canonical in V1 records under `notes/`
- Stage 1 prompt generation stays subordinate to `tools/codex/run_prompt.py` and the existing review workflow

## Explicit Deferrals

This note intentionally does not define:

- Stage 2 prompt-generation rules
- Stage 3 prompt-generation rules
- wrapper orchestration behavior
- the orchestration loop
- run-to-record linkage mechanics
- wrapper-local file formats
- CLI syntax
- Python implementation details
- how prompt files are named or stored
- notebook-write implementation code

This pass defines only the Stage 1 prompt-generation contract surface and its hard boundaries.

## Recommended Follow-On Prompt

`Build the minimal Stage 1 prompt-generation implementation around scanner output and runner-centered execution`

That follow-on should implement only this contract surface without expanding into Stage 2, Stage 3, orchestration redesign, or V1 record changes.
