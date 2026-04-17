#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from notebook_scanner import scan_notebook
from prompt_builder import build_stage2_prompt, build_stage3_prompt


PROMPTS_DIR = "codex_prompts"
NOTES_DIR = "notes"
STAGE2_PREFIX = "generated_stage2__"
STAGE3_PREFIX = "generated_stage3__"


@dataclass
class NotebookResult:
    notebook: str
    stage2: str
    stage3: str
    stage2_prompt: str | None = None
    stage3_prompt: str | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def chapter_notebooks(root: Path) -> list[Path]:
    notebooks: list[Path] = []
    for i in range(1, 20):
        prefix = f"{i:02d}_"
        matches = sorted(root.glob(f"{prefix}*.ipynb"))
        if len(matches) == 1:
            notebooks.append(matches[0])
        elif len(matches) == 0:
            print(f"WARNING: no notebook found for prefix {prefix}", file=sys.stderr)
        else:
            print(
                f"WARNING: multiple notebooks found for prefix {prefix}: "
                + ", ".join(m.name for m in matches),
                file=sys.stderr,
            )
    return notebooks


def resolve_notebooks(root: Path, args_notebooks: list[str] | None) -> list[Path]:
    if not args_notebooks:
        return chapter_notebooks(root)

    resolved: list[Path] = []
    for item in args_notebooks:
        p = Path(item)
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Notebook not found: {item}")
        if p.suffix != ".ipynb":
            raise ValueError(f"Not a notebook path: {item}")
        resolved.append(p)
    return resolved


def prompt_path_for(root: Path, notebook_path: Path, stage: int) -> Path:
    stem = notebook_path.relative_to(root).with_suffix("").as_posix().replace("/", "__")
    if stage == 2:
        return root / PROMPTS_DIR / f"{STAGE2_PREFIX}{stem}.md"
    if stage == 3:
        return root / PROMPTS_DIR / f"{STAGE3_PREFIX}{stem}.md"
    raise ValueError(f"Unknown stage: {stage}")


def write_prompt(prompt_path: Path, content: str, overwrite: bool) -> None:
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    if prompt_path.exists() and not overwrite:
        raise FileExistsError(
            f"Prompt already exists: {prompt_path}. Use --overwrite-prompts to replace it."
        )
    prompt_path.write_text(content, encoding="utf-8")


def run_codex_prompt(root: Path, prompt_path: Path) -> subprocess.CompletedProcess[str]:
    runner = root / "tools" / "codex" / "run_prompt.py"
    return subprocess.run(
        [str(runner), str(prompt_path)],
        text=True,
        capture_output=True,
        cwd=root,
        check=False,
    )


def status_line(label: str, value: str) -> str:
    return f"{label:<8} {value}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and run Stage 2 and Stage 3 notebook-enrichment prompts "
            "for an explicit list of notebooks or for chapters 01..19."
        )
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="Optional notebook paths. If omitted, runs against chapter notebooks 01..19.",
    )
    parser.add_argument(
        "--overwrite-prompts",
        action="store_true",
        help="Overwrite previously generated stage2/stage3 prompt files.",
    )
    parser.add_argument(
        "--stage2-only",
        action="store_true",
        help="Run only Stage 2 (goal-before explanations).",
    )
    parser.add_argument(
        "--stage3-only",
        action="store_true",
        help="Run only Stage 3 (implementation-after explanations).",
    )
    args = parser.parse_args()

    if args.stage2_only and args.stage3_only:
        print("ERROR: choose only one of --stage2-only or --stage3-only", file=sys.stderr)
        return 1

    root = repo_root()
    try:
        notebooks = resolve_notebooks(root, args.notebooks)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not notebooks:
        print("ERROR: no notebooks to process", file=sys.stderr)
        return 1

    run_stage2 = not args.stage3_only
    run_stage3 = not args.stage2_only

    results: list[NotebookResult] = []
    failures = 0

    for nb_path in notebooks:
        rel_nb = nb_path.relative_to(root).as_posix()
        print("=" * 72)
        print(rel_nb)
        print("=" * 72)

        result = NotebookResult(notebook=rel_nb, stage2="SKIPPED", stage3="SKIPPED")

        # ----- Stage 2 -----
        if run_stage2:
            inv = scan_notebook(nb_path)
            cells_needing_goal = inv.cells_needing_goal()

            if not cells_needing_goal:
                result.stage2 = "SKIPPED (no goal-before work needed)"
                print(status_line("Stage 2:", result.stage2))
            else:
                stage2_prompt = prompt_path_for(root, nb_path, 2)
                try:
                    prompt_text = build_stage2_prompt(root, inv)
                    write_prompt(stage2_prompt, prompt_text, overwrite=args.overwrite_prompts)
                    result.stage2_prompt = stage2_prompt.relative_to(root).as_posix()
                    print(status_line("Stage 2:", f"GENERATED {result.stage2_prompt}"))
                except Exception as exc:
                    result.stage2 = f"FAILED (prompt generation: {exc})"
                    failures += 1
                    print(status_line("Stage 2:", result.stage2))
                    results.append(result)
                    print()
                    continue

                proc = run_codex_prompt(root, stage2_prompt)
                if proc.returncode != 0:
                    result.stage2 = f"FAILED (runner rc={proc.returncode})"
                    failures += 1
                    print(status_line("Stage 2:", result.stage2))
                    if proc.stdout.strip():
                        print(proc.stdout.strip())
                    if proc.stderr.strip():
                        print(proc.stderr.strip(), file=sys.stderr)
                    results.append(result)
                    print()
                    continue

                result.stage2 = "DONE"
                print(status_line("Stage 2:", result.stage2))
                if proc.stdout.strip():
                    print(proc.stdout.strip())

        # ----- Stage 3 -----
        if run_stage3:
            inv = scan_notebook(nb_path)
            cells_needing_impl = inv.cells_needing_impl()

            if not cells_needing_impl:
                result.stage3 = "SKIPPED (no implementation-after work needed)"
                print(status_line("Stage 3:", result.stage3))
            else:
                stage3_prompt = prompt_path_for(root, nb_path, 3)
                try:
                    prompt_text = build_stage3_prompt(root, inv)
                    write_prompt(stage3_prompt, prompt_text, overwrite=args.overwrite_prompts)
                    result.stage3_prompt = stage3_prompt.relative_to(root).as_posix()
                    print(status_line("Stage 3:", f"GENERATED {result.stage3_prompt}"))
                except Exception as exc:
                    result.stage3 = f"FAILED (prompt generation: {exc})"
                    failures += 1
                    print(status_line("Stage 3:", result.stage3))
                    results.append(result)
                    print()
                    continue

                proc = run_codex_prompt(root, stage3_prompt)
                if proc.returncode != 0:
                    result.stage3 = f"FAILED (runner rc={proc.returncode})"
                    failures += 1
                    print(status_line("Stage 3:", result.stage3))
                    if proc.stdout.strip():
                        print(proc.stdout.strip())
                    if proc.stderr.strip():
                        print(proc.stderr.strip(), file=sys.stderr)
                    results.append(result)
                    print()
                    continue

                result.stage3 = "DONE"
                print(status_line("Stage 3:", result.stage3))
                if proc.stdout.strip():
                    print(proc.stdout.strip())

        results.append(result)
        print()

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for r in results:
        print(r.notebook)
        print(f"  Stage 2: {r.stage2}")
        if r.stage2_prompt:
            print(f"    prompt: {r.stage2_prompt}")
        print(f"  Stage 3: {r.stage3}")
        if r.stage3_prompt:
            print(f"    prompt: {r.stage3_prompt}")
        print()

    if failures:
        print(f"Completed with {failures} failure(s).", file=sys.stderr)
        return 1

    print("Completed without runner failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
