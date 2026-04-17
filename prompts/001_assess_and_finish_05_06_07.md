# Prompt 001 — Assess and Finish Notebooks 05, 06, 07
## Claude Code Prompt | handson-ml3/prompts/

---

## Context

This repo is a fork of the Hands-On Machine Learning (3rd ed.) Jupyter notebook collection,
being extended with a consistent pedagogical treatment for use in a university ML course.

The treatment was partially applied to notebooks 05, 06, and 07 before work was interrupted.
Those three notebooks are in an unknown intermediate state — some cells may already have the
full treatment, some may have partial markdown, and some may have none at all.

This is time-sensitive. One of these notebooks needs to be fully presentable for a class session
within the next 90 minutes.

---

## Your Mission

### Step 1 — Read and assess all three notebooks

Read the full contents of:
- `05_support_vector_machines.ipynb`
- `06_decision_trees.ipynb`
- `07_ensemble_learning_and_random_forests.ipynb`

For each notebook, produce a triage report covering:

1. **Completion estimate (0–100%)** — What fraction of code cells already have the full
   treatment applied (preceding goal markdown + following implementation markdown)?
2. **Top section status** — Does the notebook have a chapter-level intro markdown cell at the top?
   If yes, is it substantive or just a heading?
3. **Biggest gap** — What is the largest untreated stretch in the notebook (e.g., "cells 12–28
   have no markdown context at all")?
4. **Recommendation** — Given the state, which notebook is closest to done and should be
   finished first?

Write this triage to: `notes/001_notebook_triage_05_06_07.md`

---

### Step 2 — Finish the recommended notebook

Apply the full treatment to whichever notebook you identified as closest to done.

#### The Treatment Template

**Rule:** Preserve all existing code cells exactly. Do not modify, reorder, or delete any Python.

**Rule:** If a treatment element already exists and is substantive (>2 sentences, clearly written),
leave it alone or lightly augment. Do not replace good work.

**Rule:** If a treatment element is thin (a heading only, one vague sentence, placeholder text),
replace it with a full version.

**Rule:** If a treatment element is missing entirely, add it.

---

#### Treatment structure for the entire notebook:

```
[CHAPTER INTRO MARKDOWN CELL]
  - What is this chapter about?
  - What are the 3-5 main concepts a student should walk away understanding?
  - Why does this topic matter in the broader ML landscape?
  - Where does it sit relative to what came before and what comes next?
  - Any key vocabulary terms to know before diving in
  (Aim for 300-500 words. This is the "sit down, let me tell you what we're about to do" cell.)

Then for each logical section or code cell in the notebook:

[GOAL MARKDOWN CELL — before the code]
  - What is the goal of the next code block?
  - Why does this matter for ML? (Not just "it runs the model" — why do we care?)
  - What is this code doing that is a *better practice* worth noting?
  - What is this code doing that is just *plumbing* (necessary but not pedagogically deep)?
  (Aim for 4-8 sentences. Make the distinction between "this is important" and "this is boilerplate" explicit.)

[PYTHON CODE CELL]
  (unchanged)

[IMPLEMENTATION DETAIL MARKDOWN CELL — after the code]
  - What did we just see happen?
  - What are the implementation choices worth noticing? (e.g., why this hyperparameter,
    why this data split ratio, why this particular sklearn API call?)
  - What might go wrong here in practice, and how would you know?
  - If there is output, what should the student be looking for in that output?
  (Aim for 3-6 sentences. This is the "here's what's interesting about what we just ran" cell.)
```

---

### Step 3 — Write a completion note

After finishing the notebook, write a brief note to:
`notes/001_notebook_finish_report.md`

Include:
- Which notebook was finished
- Which cells were added vs. augmented vs. left alone
- Any cells that were skipped and why
- Honest assessment: is this notebook ready to teach from, or are there still rough patches?

---

## Output files

| File | Purpose |
|------|---------|
| `notes/001_notebook_triage_05_06_07.md` | Triage report for all three notebooks |
| `notes/001_notebook_finish_report.md` | Completion summary for the finished notebook |
| `05_support_vector_machines.ipynb` (possibly) | Updated in place if chosen |
| `06_decision_trees.ipynb` (possibly) | Updated in place if chosen |
| `07_ensemble_learning_and_random_forests.ipynb` (possibly) | Updated in place if chosen |

---

## Hard constraints

- **Do not touch any notebook other than the one you are finishing.**
- **Do not modify Python code cells.** Only add or edit markdown cells.
- **Do not create new Python cells.** If you think one is missing, note it in the finish report
  but do not add it.
- **Commit your changes** when done with a message like:
  `feat: apply full pedagogical treatment to 06_decision_trees`

---

## Tone and voice for markdown cells

Write as a knowledgeable instructor talking to a student who is smart but new to ML.
Avoid condescension. Avoid over-hedging. Be direct. Use concrete examples where helpful.
Prefer "here is why this matters" over "it is important to note that."

When distinguishing better practices from plumbing, be explicit:
> ⚙️ **Plumbing:** This line imports the dataset. Every notebook needs it; nothing special here.
> ✨ **Better practice:** We use `train_test_split` with `stratify=y` here — this matters because
>    class imbalance would otherwise skew our validation results.
