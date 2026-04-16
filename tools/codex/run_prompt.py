#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import tempfile
import time
from pathlib import Path


PROMPTS_DIR = "codex_prompts"
NOTES_DIR = "notes"
RUNNER_PATH = "tools/codex/run_prompt.py"


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def utc_timestamp(moment: dt.datetime) -> str:
    return moment.strftime("%Y%m%d_%H%M%S")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
    prompt = Path(prompt_arg)

    if prompt.is_absolute():
        return prompt

    direct = root / prompt
    if direct.exists():
        return direct

    in_prompts = root / PROMPTS_DIR / prompt
    if in_prompts.exists():
        return in_prompts

    matches = sorted((root / PROMPTS_DIR).glob(f"{prompt_arg}*"))
    if len(matches) == 1:
        return matches[0]

    return in_prompts


def prompt_file_label(root: Path, prompt_path: Path) -> str:
    try:
        return prompt_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(prompt_path.resolve())


def build_record_path(notes_dir: Path, prompt_stem: str, started_at_utc: str) -> tuple[str, Path]:
    base_run_id = f"{prompt_stem}__{started_at_utc}"
    candidate = notes_dir / f"{base_run_id}.md"
    if not candidate.exists():
        return base_run_id, candidate

    suffix = 2
    while True:
        run_id = f"{base_run_id}__{suffix}"
        candidate = notes_dir / f"{run_id}.md"
        if not candidate.exists():
            return run_id, candidate
        suffix += 1


def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
        output_path = Path(handle.name)

    try:
        result = subprocess.run(
            ["codex", "exec", "-C", str(root), "--output-last-message", str(output_path), "-"],
            input=prompt_text,
            text=True,
            capture_output=True,
            check=False,
        )
        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        return result.returncode, final_output, result.stderr
    finally:
        output_path.unlink(missing_ok=True)


def fenced_block(text: str, fence: str, info: str) -> str:
    body = text.rstrip("\n")
    return f"{fence}{info}\n{body}\n{fence}" if body else f"{fence}{info}\n\n{fence}"


def build_record_content(
    *,
    run_id: str,
    prompt_file: str,
    prompt_stem: str,
    started_at_utc: str,
    execution_status: str,
    finished_at_utc: str,
    return_code: int,
    prompt_text: str,
    codex_output: str,
    stderr_text: str,
    elapsed_seconds: float,
) -> str:
    final_output = codex_output.rstrip()
    stderr_body = stderr_text.rstrip()

    sections = [
        f"# {run_id}",
        "",
        f"- run_id: `{run_id}`",
        f"- prompt_file: `{prompt_file}`",
        f"- prompt_stem: `{prompt_stem}`",
        f"- started_at_utc: `{started_at_utc}`",
        "",
        "## Execution Facts",
        "",
        f"- execution_status: `{execution_status}`",
        f"- finished_at_utc: `{finished_at_utc}`",
        f"- runner: `{RUNNER_PATH}`",
        f"- return_code: `{return_code}`",
        "- retry_of_run_id:",
        "",
        "## Review Facts",
        "",
        "- review_status: `UNREVIEWED`",
        "- review_summary:",
        "- reviewed_by:",
        "- reviewed_at_utc:",
        "",
        "## Failure Analysis",
        "",
        "- failure_type:",
        "- failure_symptom:",
        "- likely_cause:",
        "- recommended_next_action:",
        "",
        "## Resource / Cost Facts",
        "",
        f"- elapsed_seconds: `{elapsed_seconds:.3f}`",
        f"- final_output_char_count: `{len(codex_output)}`",
        f"- stderr_char_count: `{len(stderr_text)}`",
        "",
        "## Prompt Text",
        "",
        fenced_block(prompt_text, "```", "md"),
        "",
        "## Codex Final Output",
        "",
        final_output if final_output else "*No output captured.*",
        "",
        "## Stderr",
        "",
        fenced_block(stderr_body, "```", "text") if stderr_body else "*No stderr captured.*",
        "",
    ]
    return "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one prompt file through codex exec and write a V1 execution record."
    )
    parser.add_argument("prompt", help="Prompt filename, numeric prefix, or path")
    args = parser.parse_args()

    root = repo_root()
    prompts_dir = root / PROMPTS_DIR
    notes_dir = root / NOTES_DIR

    if not prompts_dir.exists():
        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
        return 1

    if not notes_dir.exists():
        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
        return 1

    prompt_path = resolve_prompt_path(root, args.prompt)
    if not prompt_path.exists():
        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
        return 1

    prompt_text = prompt_path.read_text(encoding="utf-8")
    started_at = utc_now()
    started_at_utc = utc_timestamp(started_at)
    run_id, record_path = build_record_path(notes_dir, prompt_path.stem, started_at_utc)

    monotonic_start = time.monotonic()
    return_code, codex_output, stderr_text = run_codex(root, prompt_text)
    elapsed_seconds = time.monotonic() - monotonic_start
    finished_at_utc = utc_timestamp(utc_now())
    execution_status = "EXECUTED" if return_code == 0 else "EXECUTION_FAILED"

    record_path.write_text(
        build_record_content(
            run_id=run_id,
            prompt_file=prompt_file_label(root, prompt_path),
            prompt_stem=prompt_path.stem,
            started_at_utc=started_at_utc,
            execution_status=execution_status,
            finished_at_utc=finished_at_utc,
            return_code=return_code,
            prompt_text=prompt_text,
            codex_output=codex_output,
            stderr_text=stderr_text,
            elapsed_seconds=elapsed_seconds,
        ),
        encoding="utf-8",
    )

    print(record_path)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
