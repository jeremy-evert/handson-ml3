# Prompt 002 — Tooling Assessment and Runner Design Recon
## Claude Code Prompt | handson-ml3/prompts/

---

## Context

This repo is a fork of the Hands-On Machine Learning (3rd ed.) Jupyter notebook collection
extended for university ML instruction. The 19 numbered chapter notebooks (`01_` through `19_`)
each teach a chapter from the book. A pedagogical treatment is being applied to each notebook
(see `prompts/001_assess_and_finish_05_06_07.md` for the full treatment spec).

Prompt 001 finished notebook 06. Notebook 07 is partially treated with format inconsistencies.
The rest are largely untouched. An automated runner will be built in a later prompt to apply
the treatment across all remaining notebooks.

This prompt is about **understanding the existing tooling** and **informing the runner design**.
Read-only except for writing to `reports/`. Do not modify notebooks or tool files.

---

## Part A — Orientation (brief)

Read `tools/Codex_Prompt_Workflow_Architecture.md` and `tools/Project_Design_Workflow.md`.
Write two or three sentences in your report summarizing what these documents say the system
is supposed to do. This is orientation only — the real work is Part B.

---

## Part B — tools/codex Deep Dive

Read every file in `tools/codex/`. Produce a complete assessment.

For each Python script, explain:
- What it does
- What it reads as input
- What it writes as output
- What external state it depends on (files in `codex_prompts/`, `notes/`, etc.)
- Whether it is currently functional, partially functional, or broken/stale

Then answer these specific questions:

1. **End-to-end workflow**: Walk a complete prompt execution cycle from "prompt file exists
   in `codex_prompts/`" to "run is complete and reviewed." Name every script and file
   touched, in order.

2. **What is working well?** Clean, sensible, worth keeping.

3. **What is problematic?** Broken, over-engineered, inconsistently named, or unused.
   Name files and be specific.

4. **What is `baby_run_prompt.py`?** How does it differ from `run_prompt.py`?

5. **The `V1_*.md` files** — living design documents or historical artifacts that no longer
   match the code?

6. **`codex_prompts/` → `notes/` naming**: Is the convention consistent? Orphaned notes?
   Prompts with no corresponding notes?

7. **`v1_record_validation.py`** — what does it validate, and is it actually being called
   by the other scripts, or just sitting there?

Write this to: `reports/002_tools_codex_assessment.md`

---

## Part D — Synthesis for the Runner Design

Using what you learned in Part B, plus a look at two reference notebooks, answer the
questions a runner builder needs answered before writing a single line of code.

### Reference notebooks to read:

**`06_decision_trees.ipynb`** — the gold standard. Prompt 001 just finished this one.
It has a complete chapter intro and full Goal Before / Implementation After treatment on
all 58 code cells. This is what "done" looks like.

**`07_ensemble_learning_and_random_forests.ipynb`** — the workload preview. About 23%
treated, with a format mismatch: existing markdown cells use `**Why run this cell**` and
`**Result**:` instead of the canonical `### Goal` / implementation notes format.
This is what the runner will encounter on most notebooks.

### Questions to answer:

1. **Parallel track or integration?**
   Should the notebook enrichment runner integrate with the existing `tools/codex/`
   pipeline (reusing its queue, execution records, run/review cycle), or should it be
   a clean parallel track that only borrows its conventions?
   Give a concrete recommendation with reasoning.

2. **What does the runner need to detect?**
   Based on 07's format mismatch and 06's clean state, what cell-level conditions must
   the runner identify before deciding what to do? Be specific about the detection logic.

3. **What does the runner need to handle that the current tooling does not?**
   The `tools/codex/` pipeline is prompt-file-in, notes-file-out. The notebook runner
   needs to read and write `.ipynb` JSON, detect partial treatment, normalize mismatched
   formats, and insert new cells without disturbing code cells. What gaps exist?

4. **Risk assessment**: What is the highest-risk operation in the runner, and what
   safeguard should be built in before anything gets written to a notebook?

Write this to: `reports/002_synthesis_for_runner_design.md`

---

## Output Summary

| File | Contents |
|------|---------|
| `reports/002_tools_codex_assessment.md` | Full tools/codex assessment |
| `reports/002_synthesis_for_runner_design.md` | Runner design questions answered |

Do not commit. Leave as untracked files for human review.

---

## Tone

Write as if briefing a developer who is smart but new to this repo.
Be specific. Use file names and line numbers where relevant.
If something is broken, say it is broken. If something is well-designed, say so.
No hedging.
