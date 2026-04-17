"""
notebook_scanner.py — Cell inventory and treatment-state detection.

Reads a .ipynb file and classifies every code cell's treatment state:
  - has_goal_before  : 'canonical', 'legacy', 'thin', 'missing'
  - has_impl_after   : 'canonical', 'legacy', 'thin', 'missing'

Also detects the chapter intro status and the Setup cell position.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Detection vocabulary
# ---------------------------------------------------------------------------

GOAL_CANONICAL = "### Goal Before This Cell"
IMPL_CANONICAL = "### Implementation Notes After This Cell"

GOAL_LEGACY_KEYWORDS = [
    "**Why run this cell",
    "**Goal:",
    "**What it is",
    "**Principle:",
]

IMPL_LEGACY_KEYWORDS = [
    "**Result**:",
    "**Why this matters",
]

# A cell whose source starts with `# Setup` marks the boundary for intro detection.
SETUP_MARKERS = ("# setup", "## setup")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CellState:
    index: int
    cell_type: str          # 'code' or 'markdown'
    is_empty: bool          # True if source is whitespace only
    has_goal_before: str    # 'canonical', 'legacy', 'thin', 'missing'
    has_impl_after: str     # 'canonical', 'legacy', 'thin', 'missing'
    goal_cell_index: int    # index of preceding markdown cell, or -1
    impl_cell_index: int    # index of following markdown cell, or -1

    def needs_goal(self) -> bool:
        return self.has_goal_before in ("missing", "legacy", "thin") and not self.is_empty

    def needs_impl(self) -> bool:
        return self.has_impl_after in ("missing", "legacy", "thin") and not self.is_empty


@dataclass
class NotebookInventory:
    path: Path
    notebook_stem: str
    total_cells: int
    code_cell_states: list[CellState]   # one entry per code cell
    chapter_intro_status: str           # 'substantive', 'thin', 'heading', 'missing'
    chapter_intro_index: int            # cell index of existing intro, -1 if missing
    setup_cell_index: int               # index of the '# Setup' cell, -1 if not found

    def needs_stage1(self) -> bool:
        """Stage 1 is needed when chapter intro is not already substantive."""
        return self.chapter_intro_status != "substantive"

    def cells_needing_goal(self) -> list[CellState]:
        return [cs for cs in self.code_cell_states if cs.needs_goal()]

    def cells_needing_impl(self) -> list[CellState]:
        return [cs for cs in self.code_cell_states if cs.needs_impl()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cell_source(cell: dict) -> str:
    """Return the full source text of a notebook cell."""
    return "".join(cell.get("source", []))


def word_count(text: str) -> int:
    return len(text.split())


def _classify_goal(src: str) -> str:
    if GOAL_CANONICAL in src:
        return "canonical"
    if any(kw in src for kw in GOAL_LEGACY_KEYWORDS):
        return "legacy"
    # Anything else (section header, book narrative, etc.) is 'thin'
    return "thin"


def _classify_impl(src: str) -> str:
    if IMPL_CANONICAL in src:
        return "canonical"
    if any(kw in src for kw in IMPL_LEGACY_KEYWORDS):
        return "legacy"
    # Short bold-starting cell counts as legacy per spec
    if src.strip().startswith("**") and word_count(src) < 100:
        return "legacy"
    return "thin"


# ---------------------------------------------------------------------------
# Chapter intro detection
# ---------------------------------------------------------------------------

def _is_setup_cell(src: str) -> bool:
    stripped = src.strip().lower()
    return any(stripped.startswith(m) for m in SETUP_MARKERS)


def _is_prose_candidate(cell: dict, src: str) -> bool:
    """True if this markdown cell could be the chapter intro (not a heading-only or HTML cell)."""
    if cell.get("cell_type") != "markdown":
        return False
    stripped = src.strip()
    # Skip the bold-title cell like **Chapter 7 – ...**
    if stripped.startswith("**") and "\n" not in stripped:
        return False
    # Skip the italic notebook-description line
    if stripped.startswith("_This notebook contains") and "\n" not in stripped:
        return False
    # Skip HTML tables (colab/kaggle links)
    if stripped.startswith("<table"):
        return False
    return True


def detect_chapter_intro(cells: list[dict]) -> tuple[str, int, int]:
    """
    Scan cells 0-5 (or up to # Setup) and identify the chapter intro.

    Returns:
        (status, intro_index, setup_index)

    status values: 'substantive', 'thin', 'heading', 'missing'
    intro_index: cell index of the intro, or -1 if missing
    setup_index: cell index of the # Setup cell, or -1 if not found
    """
    setup_index = -1
    for i, cell in enumerate(cells):
        if _is_setup_cell(cell_source(cell)):
            setup_index = i
            break

    upper_bound = min(6, setup_index if setup_index != -1 else len(cells))

    best_index = -1
    best_wc = -1
    for i in range(upper_bound):
        cell = cells[i]
        src = cell_source(cell)
        if not _is_prose_candidate(cell, src):
            continue
        wc = word_count(src)
        if wc > best_wc:
            best_wc = wc
            best_index = i

    if best_index == -1 or best_wc < 10:
        return "missing", -1, setup_index

    if best_wc > 200:
        status = "substantive"
    elif best_wc >= 50:
        status = "thin"
    else:
        status = "heading"

    return status, best_index, setup_index


# ---------------------------------------------------------------------------
# Main scan function
# ---------------------------------------------------------------------------

def scan_notebook(path: Path) -> NotebookInventory:
    """Read a notebook and return a full cell inventory."""
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    intro_status, intro_index, setup_index = detect_chapter_intro(cells)

    code_cell_states: list[CellState] = []

    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        src = cell_source(cell)
        is_empty = not src.strip()

        # Classify goal_before
        goal_cell_idx = -1
        if i > 0 and cells[i - 1].get("cell_type") == "markdown":
            goal_cell_idx = i - 1
            goal_status = _classify_goal(cell_source(cells[i - 1]))
        else:
            goal_status = "missing"

        # Classify impl_after
        impl_cell_idx = -1
        if i < len(cells) - 1 and cells[i + 1].get("cell_type") == "markdown":
            impl_cell_idx = i + 1
            impl_status = _classify_impl(cell_source(cells[i + 1]))
        else:
            impl_status = "missing"

        code_cell_states.append(CellState(
            index=i,
            cell_type="code",
            is_empty=is_empty,
            has_goal_before=goal_status,
            has_impl_after=impl_status,
            goal_cell_index=goal_cell_idx,
            impl_cell_index=impl_cell_idx,
        ))

    return NotebookInventory(
        path=path,
        notebook_stem=path.stem,
        total_cells=len(cells),
        code_cell_states=code_cell_states,
        chapter_intro_status=intro_status,
        chapter_intro_index=intro_index,
        setup_cell_index=setup_index,
    )


def format_inventory_report(inv: NotebookInventory) -> str:
    """Human-readable summary of a notebook scan result."""
    lines = [
        f"Notebook: {inv.notebook_stem}",
        f"Total cells: {inv.total_cells}",
        f"Code cells: {len(inv.code_cell_states)}",
        f"Chapter intro: {inv.chapter_intro_status} (cell {inv.chapter_intro_index})",
        f"Setup cell: {inv.setup_cell_index}",
        "",
        f"Stage 1 needed: {inv.needs_stage1()}",
        f"Cells needing goal (Stage 2): {len(inv.cells_needing_goal())}",
        f"Cells needing impl (Stage 3): {len(inv.cells_needing_impl())}",
        "",
        "Code cell detail:",
    ]
    for cs in inv.code_cell_states:
        if cs.is_empty:
            lines.append(f"  [{cs.index:3d}] (empty)")
        else:
            lines.append(
                f"  [{cs.index:3d}] goal={cs.has_goal_before:9s} impl={cs.has_impl_after:9s}"
                f"  (goal_cell={cs.goal_cell_index}, impl_cell={cs.impl_cell_index})"
            )
    return "\n".join(lines)
