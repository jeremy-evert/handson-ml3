# Synthesis for the Notebook Runner Design
**Prompt:** 002_repo_and_tooling_recon  
**Date:** 2026-04-17  
**Reference notebooks:** `06_decision_trees.ipynb` (gold standard), `07_ensemble_learning_and_random_forests.ipynb` (workload preview)

---

## Reference Notebook Summary

### NB06 — What "Done" Looks Like

229 cells: 59 code, 170 markdown. 100% treated. Every non-empty code cell is flanked by:
- A "### Goal Before This Cell" markdown cell immediately preceding it
- A "### Implementation Notes After This Cell" markdown cell immediately following it

Plus a 517-word chapter intro at cell 3 (added by prompt 001). Format is consistent throughout. This is the target state.

### NB07 — What the Runner Will Encounter

183 cells: 74 code, 109 markdown. ~23% treated by a useful definition. The treatment that exists uses different vocabulary: `**Why run this cell**:`, `**Result**:`, `**What it is**:`, `**Principle:**` — not the canonical `### Goal Before This Cell` / `### Implementation Notes After This Cell` headers. The chapter intro (cell 2) exists but is ~140 words, below the 300-500 word target. The setup section (cells 4-12) has no treatment at all. Largest untreated stretch: cells 42–178, covering 60 code cells (Out-of-Bag, Random Forests, Boosting, Stacking, exercises).

---

## Question 1: Parallel Track or Integration?

**Recommendation: Parallel track that borrows conventions.**

The existing codex pipeline is tightly coupled to one execution model: `codex exec` subprocess → markdown notes in `notes/`. That model assumes a human-written prompt file in `codex_prompts/`, a Codex CLI binary on PATH, and output captured from a subprocess. None of that maps to notebook enrichment.

The notebook runner needs to:
- Read and write `.ipynb` JSON directly
- Generate markdown cell content via Claude API calls (not `codex exec`)
- Make cell-level decisions about insertion, replacement, or skip
- Work notebook-by-notebook with its own progress tracking

Integrating that into `run_prompt.py` would either require significant surgery to an already-working V1 script, or produce a monster that does two unrelated things. The cost is high; the benefit is nil.

**What to borrow from the codex pipeline:**
- The execution record format: write a durable markdown execution record to `notes/` after each notebook run (same V1 sections, same `run_id` pattern using `{notebook_stem}__{started_at_utc}`)
- The review gate: same three states — UNREVIEWED → ACCEPTED|REJECTED — with the same queue-progression rule
- `v1_record_validation.py`: import it directly rather than duplicating the contract
- The naming convention for records

This gives the notebook runner first-class status in the workflow system (its runs show up in `list_review_backlog.py`, obey the review gate) without forcing a bad architectural coupling.

The one prerequisite: the notebook runner needs to know to write its records with `prompt_file` pointing to whatever triggered it (the notebook filename itself, or a treatment-spec prompt file if one is added to `codex_prompts/`).

---

## Question 2: What Does the Runner Need to Detect?

Before deciding what to do with any cell, the runner needs to detect:

### 2a. Chapter intro presence and quality
- **Detect:** Is there a substantive chapter-level markdown cell before `# Setup`?
- **Criteria:** Position is cell 0–4 (before Setup). Length > 200 words. Contains either the canonical intro structure or a previous attempt.
- **Action slots:** MISSING (add), THIN (replace), PRESENT (leave alone)

### 2b. For each non-empty code cell: does it have a Goal Before?
- **Detect:** Is the immediately preceding cell a markdown cell? Does it contain canonical or legacy "goal" vocabulary?
  - Canonical: `### Goal Before This Cell` in the source
  - Legacy: `**Goal:**`, `**Why run this cell**:`, `**What it is**:` in the source
- **Action slots:** MISSING (add), LEGACY (replace or augment), PRESENT (leave alone)
- **Complication:** What if the preceding cell is another code cell (two code cells in a row)? The runner must handle this gracefully — it inserts between them.

### 2c. For each non-empty code cell: does it have an Implementation Notes After?
- **Detect:** Is the immediately following cell a markdown cell? Does it contain canonical or legacy "implementation" vocabulary?
  - Canonical: `### Implementation Notes After This Cell`
  - Legacy: `**Result**:`, `**Why this matters**:`, `**AdaBoost in action**:`
- **Action slots:** MISSING (add), LEGACY (replace or augment), PRESENT (leave alone)
- **Complication:** Two code cells back-to-back with no markdown between them requires insertion.

### 2d. Should this code cell be skipped?
- **Skip:** Empty code cells (only whitespace or a final empty cell)
- **Skip (maybe):** "extra code" cells — cells whose source starts with `# extra code`. The treatment spec says to treat these as real code cells, but their pedagogical depth is often lower. The runner should probably detect these and generate lighter-weight treatment, not skip entirely.

### 2e. Format mismatch severity
- **Canonical:** Uses `### Goal Before This Cell` / `### Implementation Notes After This Cell` headers
- **Partial legacy:** Uses `**Why run this cell**:` style — functionally similar but doesn't match the heading structure
- **Thin:** One sentence or a short heading with no substance
- **This matters** because the runner needs different handling: canonical-present cells get left alone, thin/legacy cells get replaced.

---

## Question 3: What Does the Runner Need That Current Tooling Does Not?

The codex pipeline is: prompt-file-in, markdown-record-out. The notebook runner needs substantially more:

### `.ipynb` JSON read/write
None of the current tools touch `.ipynb` files. The runner needs:
- Read: `json.load()` the notebook, access `cells` array
- Identify cell type, source, outputs, metadata
- Insert new cells at specific positions without touching any existing cell
- Write: `json.dump()` back, preserving structure (cell IDs, kernel metadata, output cells)

### Cell insertion logic
The V1 tools only write to `notes/`. The notebook runner needs to insert markdown cells at specific indices in the `.ipynb` cell array. This is structurally different — it's in-place surgery on a structured JSON document.

### Treatment-state detection
The V1 tools rely on a simple: "is there a V1 record for this prompt?" The notebook runner needs per-cell state: which cells are treated, which are partially treated, which use the legacy vocabulary. This is new logic with no analog in the current tooling.

### Format normalization
NB07 has cells written in a different voice and structure than the canonical format. The runner needs to recognize these and either replace them or augment them. The V1 tooling has no concept of "partially matching" records.

### Claude API integration
The V1 runner calls `codex exec` via subprocess. The notebook runner cannot do this — it needs to generate markdown cell content that is contextually aware of the surrounding cells (the code cell content, section header, chapter topic). This requires a direct API call or a different execution method that passes notebook context.

### Idempotency guarantee
If the runner is interrupted and restarted, it must not double-treat cells that were already treated in a previous partial run. The V1 runner sidesteps this by creating a new record file each time. The notebook runner modifies a file in place — it needs to check treatment state before acting.

### Notebook-level execution record
The V1 record format (Section 1: identity, Section 2: execution facts, etc.) should be extended or adapted to include notebook-specific facts: which notebook was processed, how many cells were added vs. augmented vs. left alone, whether any cells were skipped and why.

---

## Question 4: Risk Assessment

**Highest-risk operation: writing the modified `.ipynb` JSON back to disk.**

If this goes wrong:
- The notebook's JSON is malformed → notebook is unreadable in Jupyter
- A code cell was accidentally modified → students run different code than intended, and the bug may not be visible until runtime
- Cells were inserted at the wrong index → the logical flow of the notebook is broken

These failures are not obvious until the notebook is opened and run. Code cell corruption in particular is silent until execution.

**Required safeguard: verify-before-write, with backup.**

Before writing any modified notebook back to disk:

1. **Verify code cell integrity.** Re-read the original notebook's code cells and compare them byte-for-byte against the corresponding cells in the modified notebook. If any code cell source differs, abort and report the mismatch. This check should be a hard stop, not a warning.

2. **Verify cell count sanity.** The modified notebook should have exactly N more cells than the original, where N equals the number of markdown cells inserted. If the delta is wrong, abort.

3. **Write to a temp file first.** Write the modified notebook to `{notebook_stem}.ipynb.tmp`, validate the JSON round-trips cleanly (`json.loads(json.dumps(nb))`), then rename to the final path. This prevents partial writes from corrupting the original.

4. **Record what was done in the execution record.** The V1-style run record for this notebook should enumerate every cell that was added or modified, by index, so a human reviewer can spot-check specific cells rather than reading the entire notebook.

The fundamental risk is low-visibility corruption. The safeguard is: make the diff reviewable, make the code cell verification automatic, and never assume the write succeeded until it has been verified.
