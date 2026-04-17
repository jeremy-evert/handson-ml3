"""
prompt_builder.py — Generates per-job prompt files for codex exec.

Each (notebook, stage) pair gets a focused prompt that tells codex exactly:
- Which notebook to modify
- What stage is being run
- Which cells need treatment (from scanner output)
- The treatment specification
- A gold standard example from NB06
- The output contract (write to .tmp, don't touch code cells)
"""

from __future__ import annotations

import json
from pathlib import Path

from notebook_scanner import NotebookInventory, cell_source


TREATMENT_SPEC_SOURCE = "prompts/001_assess_and_finish_05_06_07.md"
GOLD_STANDARD_NOTEBOOK = "06_decision_trees.ipynb"
ACTIVE_PROMPT_NAME = "active_prompt"  # used as prompt_stem base


def _repo_root(this_file: Path) -> Path:
    return this_file.resolve().parents[2]


def _load_treatment_spec(root: Path) -> str:
    path = root / TREATMENT_SPEC_SOURCE
    if not path.exists():
        return "[Treatment spec not found — see prompts/001_assess_and_finish_05_06_07.md]"
    text = path.read_text(encoding="utf-8")
    # Extract just the treatment template section
    start = text.find("#### Treatment structure")
    end = text.find("---", start + 10) if start != -1 else -1
    if start != -1 and end != -1:
        return text[start:end].strip()
    # Fallback: return the whole thing
    return text


def _load_gold_standard(root: Path) -> str:
    """Pull the chapter intro + one Goal/Code/Impl trio from NB06."""
    path = root / GOLD_STANDARD_NOTEBOOK
    if not path.exists():
        return "[Gold standard notebook not found]"

    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]

    # Find the chapter intro cell (cell 3 in NB06 after prompt 001 treatment)
    intro_text = ""
    for cell in cells[:6]:
        src = cell_source(cell)
        if "## Chapter Overview" in src or "Chapter Overview" in src:
            intro_text = src[:800] + ("..." if len(src) > 800 else "")
            break

    # Find one Goal/Code/Impl trio
    trio_text = ""
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        if not src.strip():
            continue
        goal_src = cell_source(cells[i - 1]) if i > 0 and cells[i - 1].get("cell_type") == "markdown" else ""
        impl_src = cell_source(cells[i + 1]) if i < len(cells) - 1 and cells[i + 1].get("cell_type") == "markdown" else ""
        if "### Goal Before This Cell" in goal_src and "### Implementation Notes After This Cell" in impl_src:
            trio_text = (
                f"### GOAL BEFORE CELL:\n{goal_src[:600]}\n\n"
                f"### CODE CELL:\n```python\n{src[:400]}\n```\n\n"
                f"### IMPLEMENTATION NOTES AFTER CELL:\n{impl_src[:600]}"
            )
            break

    parts = []
    if intro_text:
        parts.append("**Chapter Intro Example:**\n" + intro_text)
    if trio_text:
        parts.append("**Goal/Code/Implementation Trio Example:**\n" + trio_text)
    return "\n\n".join(parts) if parts else "[No gold standard found]"


def _cell_source_snippet(src: str, max_chars: int = 300) -> str:
    src = src.strip()
    if len(src) <= max_chars:
        return src
    return src[:max_chars] + "..."


def build_stage1_prompt(root: Path, inventory: NotebookInventory) -> str:
    """Generate the stage 1 prompt: add/replace the chapter intro."""
    nb_path = inventory.path.relative_to(root).as_posix()
    tmp_path = str(inventory.path.with_suffix(".ipynb.tmp").name)
    chapter_num = inventory.notebook_stem.split("_")[0].lstrip("0") or "?"

    with open(inventory.path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]

    # Describe current intro state
    intro_idx = inventory.chapter_intro_index
    if intro_idx == -1:
        intro_description = "MISSING — no chapter intro exists. Insert a new one."
        insert_instruction = (
            f"INSERT a new markdown cell at index {max(3, inventory.setup_cell_index - 1 if inventory.setup_cell_index > 0 else 3)}. "
            "Place it after the Colab/Kaggle links table and before the # Setup cell."
        )
        current_content = "(no existing intro)"
    else:
        current_src = cell_source(cells[intro_idx])
        intro_description = f"{inventory.chapter_intro_status.upper()} at cell index {intro_idx} (~{len(current_src.split())} words)."
        insert_instruction = f"REPLACE the markdown cell at index {intro_idx} with the full chapter intro."
        current_content = _cell_source_snippet(current_src)

    treatment_spec = _load_treatment_spec(root)
    gold_standard = _load_gold_standard(root)

    return f"""# Stage 1: Chapter Intro Enrichment

## Target Notebook
- Path: {nb_path}
- Chapter: {chapter_num}
- Notebook stem: {inventory.notebook_stem}

## Current Chapter Intro State
Status: {intro_description}

Current content (if any):
```
{current_content}
```

## Your Task
{insert_instruction}

The new intro must follow the treatment specification below exactly.

## Treatment Specification

{treatment_spec}

## Gold Standard Example (from 06_decision_trees.ipynb — the finished notebook)

{gold_standard}

## Output Contract

1. Read the full notebook from: `{nb_path}`
2. {insert_instruction}
3. Write the COMPLETE modified notebook as valid JSON to: `{tmp_path}`
   (write it to the same directory as the original notebook)
4. Do NOT modify any `code` cells — not their source, outputs, or metadata
5. Do NOT add, remove, or reorder any cell other than the one change described above
6. The new intro cell must have `"cell_type": "markdown"` and a `"source"` field

## Hard Constraints

- NEVER modify a cell with `"cell_type": "code"`
- The intro must be 300–500 words covering: what this chapter is about, 3–5 learning
  objectives, why this topic matters in ML, where it fits in the course, key vocabulary
- Write clean JSON — the same structure as the input notebook
- The output file must be a valid Jupyter notebook that can be opened in Jupyter Lab
"""


def build_stage2_prompt(root: Path, inventory: NotebookInventory) -> str:
    """Generate the stage 2 prompt: add Goal Before cells."""
    nb_path = inventory.path.relative_to(root).as_posix()
    tmp_path = str(inventory.path.with_suffix(".ipynb.tmp").name)

    cells_needing = inventory.cells_needing_goal()
    if not cells_needing:
        return f"# Stage 2: No Goal Cells Needed\n\nNotebook {nb_path} already has goal cells for all code cells. No action required."

    treatment_spec = _load_treatment_spec(root)
    gold_standard = _load_gold_standard(root)

    cell_list = []
    with open(inventory.path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]

    for cs in cells_needing:
        src = cell_source(cells[cs.index])
        snippet = _cell_source_snippet(src, 200)
        action = "REPLACE" if cs.goal_cell_index != -1 else "INSERT BEFORE"
        cell_list.append(
            f"- Code cell index {cs.index} (current goal status: {cs.has_goal_before}): "
            f"{action} goal cell\n  Code preview: `{snippet[:100]}`"
        )

    cell_list_str = "\n".join(cell_list)

    return f"""# Stage 2: Goal-Before-Cell Enrichment

## Target Notebook
- Path: {nb_path}

## Cells Needing Goal Treatment ({len(cells_needing)} total)

{cell_list_str}

## Treatment Specification

{treatment_spec}

## Gold Standard Example

{gold_standard}

## Rules Per Cell

For each code cell in the list above:
- If action is INSERT BEFORE: add a new markdown cell immediately before the code cell
- If action is REPLACE: replace the immediately-preceding markdown cell with a canonical goal cell
- The new cell must start with `### Goal Before This Cell`
- Then 4–8 sentences: goal, why it matters for ML, ⚙️ Plumbing / ✨ Better practice labels

## Output Contract

1. Read the full notebook from: `{nb_path}`
2. Apply all the cell insertions/replacements listed above
3. Write the COMPLETE modified notebook as valid JSON to: `{tmp_path}`
4. Do NOT modify any `code` cells
5. Do NOT touch any cell not in the list above

## Hard Constraints

- NEVER modify a cell with `"cell_type": "code"`
- Every new goal cell must start with `### Goal Before This Cell`
- Maintain correct cell ordering — inserted cells go immediately BEFORE the target code cell
"""


def build_stage3_prompt(root: Path, inventory: NotebookInventory) -> str:
    """Generate the stage 3 prompt: add Implementation Notes cells."""
    nb_path = inventory.path.relative_to(root).as_posix()
    tmp_path = str(inventory.path.with_suffix(".ipynb.tmp").name)

    cells_needing = inventory.cells_needing_impl()
    if not cells_needing:
        return f"# Stage 3: No Impl Cells Needed\n\nNotebook {nb_path} already has implementation notes for all code cells. No action required."

    treatment_spec = _load_treatment_spec(root)
    gold_standard = _load_gold_standard(root)

    cell_list = []
    with open(inventory.path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]

    for cs in cells_needing:
        src = cell_source(cells[cs.index])
        snippet = _cell_source_snippet(src, 200)
        action = "REPLACE" if cs.impl_cell_index != -1 else "INSERT AFTER"
        cell_list.append(
            f"- Code cell index {cs.index} (current impl status: {cs.has_impl_after}): "
            f"{action} impl cell\n  Code preview: `{snippet[:100]}`"
        )

    cell_list_str = "\n".join(cell_list)

    return f"""# Stage 3: Implementation-Notes-After-Cell Enrichment

## Target Notebook
- Path: {nb_path}

## Cells Needing Implementation Notes ({len(cells_needing)} total)

{cell_list_str}

## Treatment Specification

{treatment_spec}

## Gold Standard Example

{gold_standard}

## Rules Per Cell

For each code cell in the list above:
- If action is INSERT AFTER: add a new markdown cell immediately after the code cell
- If action is REPLACE: replace the immediately-following markdown cell with a canonical impl cell
- The new cell must start with `### Implementation Notes After This Cell`
- Then 3–6 sentences: what we just saw, implementation choices worth noticing, what might go wrong,
  what to look for in the output

## Output Contract

1. Read the full notebook from: `{nb_path}`
2. Apply all the cell insertions/replacements listed above
3. Write the COMPLETE modified notebook as valid JSON to: `{tmp_path}`
4. Do NOT modify any `code` cells
5. Do NOT touch any cell not in the list above

## Hard Constraints

- NEVER modify a cell with `"cell_type": "code"`
- Every new impl cell must start with `### Implementation Notes After This Cell`
- Maintain correct cell ordering — inserted cells go immediately AFTER the target code cell
"""


def write_prompt_file(root: Path, notebook_stem: str, stage_num: int, content: str) -> Path:
    """
    Save the generated prompt to notes/enrichment/{notebook_stem}_stage{N}.md
    and also to the active_prompt.md slot used by codex exec.
    Returns the saved prompt path (used as prompt_file in the V1 record).
    """
    enrichment_dir = root / "notes" / "enrichment"
    enrichment_dir.mkdir(parents=True, exist_ok=True)

    # Save under a stable name for use as the V1 record's prompt_file
    prompt_path = enrichment_dir / f"{notebook_stem}_stage{stage_num}.md"
    prompt_path.write_text(content, encoding="utf-8")

    # Also write to active_prompt.md for human inspection
    active_path = enrichment_dir / "active_prompt.md"
    active_path.write_text(content, encoding="utf-8")

    return prompt_path


def build_prompt(root: Path, inventory: NotebookInventory, stage: int) -> str:
    """Dispatch to the correct stage prompt builder."""
    if stage == 1:
        return build_stage1_prompt(root, inventory)
    if stage == 2:
        return build_stage2_prompt(root, inventory)
    if stage == 3:
        return build_stage3_prompt(root, inventory)
    raise ValueError(f"Unknown stage: {stage}")
