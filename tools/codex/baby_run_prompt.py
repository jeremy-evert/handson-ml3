#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path


PROMPTS_DIR = "codex_prompts"
NOTES_DIR = "notes"


def utc_timestamp() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def repo_root_from_script(script_path: Path) -> Path:
    return script_path.resolve().parents[2]


def resolve_prompt_path(repo_root: Path, prompt_arg: str) -> Path:
    prompt_path = Path(prompt_arg)

    if prompt_path.is_absolute():
        return prompt_path

    direct = repo_root / prompt_arg
    if direct.exists():
        return direct

    in_prompts = repo_root / PROMPTS_DIR / prompt_arg
    if in_prompts.exists():
        return in_prompts

    return in_prompts


def build_note_path(repo_root: Path, prompt_path: Path, status: str) -> Path:
    ts = utc_timestamp()
    base_name = prompt_path.stem
    return repo_root / NOTES_DIR / f"{base_name}__{status}__{ts}.md"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one Codex prompt in a conservative file-based workflow."
    )
    parser.add_argument("prompt", help="Prompt filename or path")
    parser.add_argument(
        "--status",
        default="STARTED",
        help="Initial note status label (default: STARTED)",
    )
    args = parser.parse_args()

    script_path = Path(__file__)
    repo_root = repo_root_from_script(script_path)

    prompts_dir = repo_root / PROMPTS_DIR
    notes_dir = repo_root / NOTES_DIR

    if not prompts_dir.exists():
        print(f"ERROR: Missing prompt directory: {prompts_dir}", file=sys.stderr)
        return 1

    if not notes_dir.exists():
        print(f"ERROR: Missing notes directory: {notes_dir}", file=sys.stderr)
        return 1

    prompt_path = resolve_prompt_path(repo_root, args.prompt)

    if not prompt_path.exists():
        print(f"ERROR: Prompt file not found: {prompt_path}", file=sys.stderr)
        return 1

    prompt_text = prompt_path.read_text(encoding="utf-8")
    ts = utc_timestamp()
    status = args.status.upper()
    note_path = build_note_path(repo_root, prompt_path, status)

    content = f"""# {prompt_path.stem} - {status}

- Prompt file: `{prompt_path.name}`
- Timestamp (UTC): `{ts}`
- Status: `{status}`

## Prompt

```md
{prompt_text.rstrip()}
```

## Codex Output

*Paste or append Codex output here.*

## Notes

*Add observations, fixes, or follow-up thoughts here.*
"""

    note_path.write_text(content, encoding="utf-8")

    print(f"Prompt: {prompt_path}")
    print(f"Note created: {note_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
