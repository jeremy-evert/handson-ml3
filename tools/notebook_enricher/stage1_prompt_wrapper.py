#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
import sys

from notebook_scanner import NotebookInventory, scan_notebook
from prompt_builder import build_stage1_prompt


PROMPTS_DIR = "codex_prompts"
NOTES_DIR = "notes"
GENERATED_PROMPT_PREFIX = "generated_stage1__"
REPORT_PREFIX = "031_stage1_wrapper_mvp_report__"


@dataclass
class Stage1Decision:
    inventory: NotebookInventory
    action: str
    reason: str

    @property
    def needs_prompt(self) -> bool:
        return self.action in {"insert", "replace"}


@dataclass
class PromptArtifact:
    notebook_path: Path
    prompt_path: Path
    action: str
    reason: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Stage 1 notebook-enrichment prompts for an explicit notebook list.",
    )
    parser.add_argument(
        "notebooks",
        nargs="+",
        help="Ordered notebook paths, relative to the repository root or absolute.",
    )
    parser.add_argument(
        "--report-path",
        help="Optional explicit markdown report path. Defaults to notes/ with a UTC timestamp.",
    )
    parser.add_argument(
        "--overwrite-generated",
        action="store_true",
        help="Allow overwriting an existing generated Stage 1 prompt file for the same notebook.",
    )
    return parser.parse_args()


def resolve_notebook_path(root: Path, notebook_arg: str) -> Path:
    path = Path(notebook_arg)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def resolve_output_path(root: Path, output_arg: str) -> Path:
    path = Path(output_arg)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def relative_repo_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def classify_stage1_decision(inventory: NotebookInventory) -> Stage1Decision:
    status = inventory.chapter_intro_status

    if status == "substantive":
        return Stage1Decision(
            inventory=inventory,
            action="skip",
            reason="scanner classified the existing chapter intro as substantive",
        )

    if status == "missing":
        return Stage1Decision(
            inventory=inventory,
            action="insert",
            reason="scanner found no eligible chapter intro prose cell before the setup boundary",
        )

    if status in {"heading", "thin"} and inventory.chapter_intro_index >= 0:
        return Stage1Decision(
            inventory=inventory,
            action="replace",
            reason=(
                "scanner found an existing intro candidate that is not yet substantive "
                f"({status})"
            ),
        )

    return Stage1Decision(
        inventory=inventory,
        action="skip",
        reason="scanner evidence did not support one deterministic Stage 1 insert-or-replace target",
    )


def prompt_filename_for(root: Path, notebook_path: Path) -> str:
    relative = notebook_path.relative_to(root).with_suffix("")
    slug = "__".join(relative.parts)
    return f"{GENERATED_PROMPT_PREFIX}{slug}.md"


def write_prompt_file(
    *,
    root: Path,
    decision: Stage1Decision,
    overwrite_generated: bool,
) -> PromptArtifact:
    prompts_dir = root / PROMPTS_DIR
    prompts_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = prompts_dir / prompt_filename_for(root, decision.inventory.path)
    if prompt_path.exists() and not overwrite_generated:
        raise FileExistsError(
            f"generated prompt already exists: {relative_repo_path(root, prompt_path)} "
            "(rerun with --overwrite-generated to replace it)"
        )

    prompt_text = build_stage1_prompt(root, decision.inventory)
    prompt_path.write_text(prompt_text, encoding="utf-8")

    return PromptArtifact(
        notebook_path=decision.inventory.path,
        prompt_path=prompt_path,
        action=decision.action,
        reason=decision.reason,
    )


def default_report_path(root: Path) -> Path:
    return root / NOTES_DIR / f"{REPORT_PREFIX}{utc_timestamp()}.md"


def build_report_text(
    *,
    root: Path,
    notebook_paths: list[Path],
    generated: list[PromptArtifact],
    skipped: list[Stage1Decision],
) -> str:
    lines: list[str] = [
        "# Stage 1 Wrapper MVP Report",
        "",
        "## Executive Summary",
        "",
        (
            f"Requested {len(notebook_paths)} notebook(s). Generated {len(generated)} "
            f"Stage 1 prompt(s) and skipped {len(skipped)} notebook(s)."
        ),
        "This wrapper run stopped at prompt generation and did not invoke `tools/codex/run_prompt.py`.",
        "",
        "## Target Notebook List",
        "",
    ]

    for idx, notebook_path in enumerate(notebook_paths, start=1):
        lines.append(f"{idx}. `{relative_repo_path(root, notebook_path)}`")

    lines.extend(
        [
            "",
            "## Generated Prompts",
            "",
        ]
    )

    if generated:
        for artifact in generated:
            lines.append(
                (
                    f"- `{relative_repo_path(root, artifact.notebook_path)}`: generated "
                    f"`{relative_repo_path(root, artifact.prompt_path)}` "
                    f"because action is `{artifact.action}` and {artifact.reason}."
                )
            )
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Skipped Notebooks",
            "",
        ]
    )

    if skipped:
        for decision in skipped:
            inv = decision.inventory
            lines.append(
                (
                    f"- `{relative_repo_path(root, inv.path)}`: skipped because action is "
                    f"`{decision.action}` and {decision.reason}. "
                    f"(intro_status=`{inv.chapter_intro_status}`, "
                    f"intro_index=`{inv.chapter_intro_index}`, "
                    f"setup_index=`{inv.setup_cell_index}`)"
                )
            )
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Deferred Work",
            "",
            "- No generated prompts were executed in this pass.",
            "- No notebooks were modified in this pass.",
            "- Stage 2 and Stage 3 remain out of scope.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    root = repo_root()

    notebook_paths: list[Path] = []
    for notebook_arg in args.notebooks:
        notebook_path = resolve_notebook_path(root, notebook_arg)
        if not notebook_path.exists():
            print(f"ERROR: notebook not found: {notebook_arg}", file=sys.stderr)
            return 1
        try:
            notebook_path.relative_to(root)
        except ValueError:
            print(
                f"ERROR: notebook path is outside the repository root: {notebook_path}",
                file=sys.stderr,
            )
            return 1
        if notebook_path.suffix != ".ipynb":
            print(f"ERROR: not a notebook path: {notebook_arg}", file=sys.stderr)
            return 1
        notebook_paths.append(notebook_path)

    generated: list[PromptArtifact] = []
    skipped: list[Stage1Decision] = []

    for notebook_path in notebook_paths:
        inventory = scan_notebook(notebook_path)
        decision = classify_stage1_decision(inventory)
        if decision.needs_prompt:
            artifact = write_prompt_file(
                root=root,
                decision=decision,
                overwrite_generated=args.overwrite_generated,
            )
            generated.append(artifact)
            print(
                f"GENERATED {relative_repo_path(root, artifact.prompt_path)} "
                f"for {relative_repo_path(root, notebook_path)} "
                f"({decision.action})"
            )
        else:
            skipped.append(decision)
            print(
                f"SKIPPED {relative_repo_path(root, notebook_path)} "
                f"({decision.reason})"
            )

    report_path = resolve_output_path(root, args.report_path) if args.report_path else default_report_path(root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = build_report_text(
        root=root,
        notebook_paths=notebook_paths,
        generated=generated,
        skipped=skipped,
    )
    report_path.write_text(report_text, encoding="utf-8")
    print(f"REPORT {relative_repo_path(root, report_path)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
