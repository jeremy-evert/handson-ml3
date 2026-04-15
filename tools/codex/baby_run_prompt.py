#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PROMPTS_DIR = "codex_prompts"
NOTES_DIR = "notes"
CODEX_HOME_DIR = ".codex-home"
CODEX_RUN_TIMEOUT_SECONDS = 30
MANUAL_STATUS = "HANDOFF"


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_prompt_path(root: Path, prompt_arg: str) -> Path:
    prompt_path = Path(prompt_arg)
    if prompt_path.is_absolute():
        return prompt_path

    direct = root / prompt_path
    if direct.exists():
        return direct

    in_prompts = root / PROMPTS_DIR / prompt_path
    if in_prompts.exists():
        return in_prompts

    return in_prompts


def build_note_path(root: Path, prompt_path: Path, status: str, timestamp: str) -> Path:
    return root / NOTES_DIR / f"{prompt_path.stem}__{status}__{timestamp}.md"


def build_note_content(prompt_path: Path, timestamp: str, status: str, prompt_text: str) -> str:
    return f"""# {prompt_path.stem} - {status}

- Prompt file: `{prompt_path.name}`
- Timestamp (UTC): `{timestamp}`
- Status: `{status}`

## Original Prompt

```md
{prompt_text.rstrip()}
```

## Codex Output

_Pending_

## Notes

"""


def build_manual_command(root: Path) -> str:
    return f'codex -C "{root}"'


def update_note(note_path: Path, prompt_path: Path, timestamp: str, status: str, prompt_text: str, codex_output: str, notes_text: str = "") -> None:
    content = f"""# {prompt_path.stem} - {status}

- Prompt file: `{prompt_path.name}`
- Timestamp (UTC): `{timestamp}`
- Status: `{status}`

## Original Prompt

```md
{prompt_text.rstrip()}
```

## Codex Output

{codex_output.rstrip() or "_No output captured._"}

## Notes

{notes_text.rstrip()}
"""
    note_path.write_text(content.rstrip() + "\n", encoding="utf-8")


def build_codex_env(root: Path) -> dict[str, str]:
    codex_home = root / CODEX_HOME_DIR
    xdg_dirs = {
        "XDG_CONFIG_HOME": codex_home / ".config",
        "XDG_STATE_HOME": codex_home / ".local" / "state",
        "XDG_DATA_HOME": codex_home / ".local" / "share",
        "XDG_CACHE_HOME": codex_home / ".cache",
    }

    for path in (codex_home, *xdg_dirs.values()):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # Avoid inheriting the parent Codex session and force writable state paths.
    env.pop("CODEX_THREAD_ID", None)
    env["HOME"] = str(codex_home)
    for key, path in xdg_dirs.items():
        env[key] = str(path)
    return env


def run_codex(root: Path, prompt_text: str) -> tuple[int, str, str]:
    with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt", delete=False) as handle:
        output_path = Path(handle.name)

    try:
        result = subprocess.run(
            [
                "codex",
                "exec",
                "--ephemeral",
                "-C",
                str(root),
                "--output-last-message",
                str(output_path),
                "-",
            ],
            input=prompt_text,
            text=True,
            capture_output=True,
            check=False,
            env=build_codex_env(root),
            timeout=CODEX_RUN_TIMEOUT_SECONDS,
        )
        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        return result.returncode, final_output, result.stderr
    except subprocess.TimeoutExpired as exc:
        final_output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        raw_stderr = exc.stderr or ""
        if isinstance(raw_stderr, bytes):
            raw_stderr = raw_stderr.decode("utf-8", errors="replace")
        stderr_text = raw_stderr.rstrip()
        if stderr_text:
            stderr_text += "\n"
        stderr_text += f"Codex execution timed out after {CODEX_RUN_TIMEOUT_SECONDS} seconds."
        return 124, final_output, stderr_text
    finally:
        output_path.unlink(missing_ok=True)


def summarize_failure(stderr_text: str) -> list[str]:
    notes_lines: list[str] = []
    if not stderr_text.strip():
        return notes_lines

    if "Failed to create session" in stderr_text or "error creating thread" in stderr_text:
        notes_lines.append("Codex could not initialize its local session storage.")

    if "api.openai.com" in stderr_text and "Operation not permitted" in stderr_text:
        notes_lines.append("Codex could not reach the API from this environment, likely due to blocked network access.")

    notes_lines.append("Codex stderr:")
    notes_lines.append("```text")
    notes_lines.append(stderr_text.rstrip())
    notes_lines.append("```")
    return notes_lines


def build_manual_notes(root: Path, prompt_path: Path, note_path: Path) -> str:
    command = build_manual_command(root)
    return "\n".join(
        [
            "Manual handoff required.",
            f"Prompt path: `{prompt_path}`",
            f"Note path: `{note_path}`",
            "Launch Codex in a terminal/environment where interactive TTY access and networking both work:",
            "```bash",
            command,
            "```",
            "Then paste the contents of the prompt file into the interactive Codex session.",
            "After Codex finishes, paste or append the final output into this note.",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one prompt file directly through Codex.")
    parser.add_argument("--manual", action="store_true", help="Prepare a manual Codex handoff instead of running codex exec")
    parser.add_argument("prompt", help="Prompt filename or path")
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
    initial_status = MANUAL_STATUS if args.manual else "STARTED"
    started_path = build_note_path(root, prompt_path, initial_status, timestamp)
    started_path.write_text(
        build_note_content(prompt_path, timestamp, initial_status, prompt_text),
        encoding="utf-8",
    )

    if args.manual:
        manual_notes = build_manual_notes(root, prompt_path, started_path)
        update_note(
            note_path=started_path,
            prompt_path=prompt_path,
            timestamp=timestamp,
            status=MANUAL_STATUS,
            prompt_text=prompt_text,
            codex_output="_Manual handoff prepared. No automatic Codex execution was attempted._",
            notes_text=manual_notes,
        )
        print(started_path)
        print()
        print(manual_notes)
        return 0

    returncode, codex_output, stderr_text = run_codex(root, prompt_text)
    final_status = "SUCCESS" if returncode == 0 else "FAILED"
    final_path = build_note_path(root, prompt_path, final_status, timestamp)

    notes_lines = []
    if final_path != started_path:
        started_path.rename(final_path)

    notes_lines.extend(summarize_failure(stderr_text))

    if returncode != 0:
        notes_lines.append(f"Codex exited with status `{returncode}`.")

    update_note(
        note_path=final_path,
        prompt_path=prompt_path,
        timestamp=timestamp,
        status=final_status,
        prompt_text=prompt_text,
        codex_output=codex_output,
        notes_text="\n".join(notes_lines).strip(),
    )

    print(final_path)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
