#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import tempfile
from pathlib import Path


PROMPTS_DIR = "codex_prompts"
NOTES_DIR = "notes"


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


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


def build_note_path(root: Path, prompt_path: Path, status: str, timestamp: str) -> Path:
    return root / NOTES_DIR / f"{prompt_path.stem}__{status}__{timestamp}.md"


def build_note_content(
    prompt_path: Path,
    timestamp: str,
    status: str,
    prompt_text: str,
    codex_output: str,
    notes: str = "",
) -> str:
    return f"""# {prompt_path.stem} - {status}

- Prompt file: `{prompt_path.name}`
- Timestamp (UTC): `{timestamp}`
- Status: `{status}`

## Original Prompt

```md
{prompt_text.rstrip()}
```

## Codex Output

{codex_output.rstrip() or "*No output captured.*"}

## Notes

{notes.rstrip()}
"""


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one prompt file through codex exec and write a note.")
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
    timestamp = utc_timestamp()

    returncode, codex_output, stderr_text = run_codex(root, prompt_text)
    status = "SUCCESS" if returncode == 0 else "FAILED"
    note_path = build_note_path(root, prompt_path, status, timestamp)

    notes = ""
    if stderr_text.strip():
        notes = f"""Codex stderr:
```text
{stderr_text.rstrip()}
```"""

    note_path.write_text(
        build_note_content(
            prompt_path=prompt_path,
            timestamp=timestamp,
            status=status,
            prompt_text=prompt_text,
            codex_output=codex_output,
            notes=notes,
        ),
        encoding="utf-8",
    )

    print(note_path)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
