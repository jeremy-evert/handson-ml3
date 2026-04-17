# 024_generate_staged_notebook_wrapper_prompts__20260417_172452

- run_id: `024_generate_staged_notebook_wrapper_prompts__20260417_172452`
- prompt_file: `codex_prompts/024_generate_staged_notebook_wrapper_prompts.md`
- prompt_stem: `024_generate_staged_notebook_wrapper_prompts`
- started_at_utc: `20260417_172452`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260417_172708`
- runner: `tools/codex/run_prompt.py`
- return_code: `0`
- retry_of_run_id:

## Review Facts

- review_status: `UNREVIEWED`
- review_summary:
- reviewed_by:
- reviewed_at_utc:

## Failure Analysis

- failure_type:
- failure_symptom:
- likely_cause:
- recommended_next_action:

## Resource / Cost Facts

- elapsed_seconds: `136.114`
- final_output_char_count: `1728`
- stderr_char_count: `163173`

## Prompt Text

```md
# Prompt: Generate the staged prompt set for the notebook-wrapper build

You are working inside this repository.

Your task is to use the existing assessment and current tooling layout to create the first draft prompt set that will later be used to build the staged notebook-treatment wrapper.

Do NOT build the wrapper itself yet.

Do NOT modify any `.py` files.
Do NOT modify any notebooks.
Do NOT refactor the current runner.
Do NOT change the V1 record format.

## Goal

We already fenced the problem and assessed what the wrapper should be.

Now generate the prompt files that would let Codex build this wrapper in careful stages.

The wrapper concept is:

- stay centered around Codex calls
- keep the current `tools/codex/run_prompt.py` flow intact
- use narrow, bounded tasks instead of one giant notebook-enrichment run
- support these staged notebook-treatment passes:
  - Stage 1: chapter intro detection / insertion
  - Stage 2: markdown-before-code detection / insertion
  - Stage 3: markdown-after-code detection / insertion

## What to inspect first

Inspect at minimum:

- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `tools/codex/v1_record_validation.py`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/Project_Design_Workflow.md`
- the existing assessment note for this wrapper idea
- `tools/notebook_enricher/notebook_scanner.py`
- `tools/notebook_enricher/prompt_builder.py`

## Your job

Create a practical first-draft prompt set that breaks the future implementation into small Codex-sized tasks.

These prompts should be designed so that each one:
- has a narrow scope
- can be run through the current Codex runner
- produces durable notes and visible progress
- minimizes risk of notebook corruption
- supports stop/resume behavior
- avoids overengineering

## Deliverables

Create:

1. A small sequence of new prompt files in `codex_prompts/` for the wrapper build
2. One planning note in `notes/` that explains:
   - what prompt files you created
   - what each prompt is meant to accomplish
   - why the sequence is ordered this way
   - any open concerns or assumptions

## Requirements for the generated prompt set

The prompt set should cover, in some sensible order:

- defining the wrapper state model
- defining notebook scope/config input
- defining progress tracking and stop/resume behavior
- defining notebook scan output / treatment detection
- defining safe prompt-generation rules for Stage 1, Stage 2, Stage 3
- defining notebook-write safeguards
- defining a minimal orchestration loop
- defining reporting / note artifacts for the wrapper runs
- defining what the MVP should do first

The prompts should not yet ask Codex to:
- enrich all notebooks
- process the whole repo in one pass
- redesign the V1 runner
- add parallel execution
- add a UI
- add advanced scheduling
- add automatic self-healing
- add background daemons

## Naming

Use the next available numeric prefixes in `codex_prompts/`.

Make the prompt filenames descriptive and consistent with the existing repo style.

## Output note structure

Write one markdown note into `notes/` with this structure:

# Staged Notebook Wrapper Prompt Plan

## Executive Summary
## Prompt Files Created
## Why This Sequence
## Scope Boundaries
## Risks Still Open
## Recommended Next Prompt To Run

## Output rules

- Be practical
- Be repo-specific
- Keep the prompt set small and disciplined
- Prefer the minimum viable sequence over a giant speculative backlog
- At the end of your final response, print only the path to the planning note you created
```

## Codex Final Output

I could not write the files because this workspace is mounted read-only. The draft prompt set I prepared is:

- `codex_prompts/025_define_staged_notebook_wrapper_mvp_contract.md`
  Defines the MVP boundary, wrapper state model, scope/config inputs, reporting artifacts, and explicit non-goals before any code build.
- `codex_prompts/026_build_wrapper_scope_state_and_resume_contract.md`
  Builds the minimal structured scope/progress contract for notebook selection, per-stage position, stop/resume behavior, and durable progress notes.
- `codex_prompts/027_build_wrapper_scan_and_treatment_report_layer.md`
  Adds the narrow scan/report layer that reuses `tools/notebook_enricher/notebook_scanner.py` and emits deterministic treatment decisions without mutating notebooks.
- `codex_prompts/028_build_stage1_intro_prompt_generation_rules.md`
  Implements only Stage 1 prompt-generation rules for chapter intro detection/insertion, with safe write constraints and narrow output contracts.
- `codex_prompts/029_build_stage2_goal_before_and_stage3_impl_after_prompt_generation_rules.md`
  Implements the Stage 2 and Stage 3 prompt-generation rules, focused on bounded cell lists, replace-vs-insert decisions, and notebook-write safeguards.
- `codex_prompts/030_build_minimal_notebook_wrapper_orchestration_loop.md`
  Implements the thinnest orchestration loop around `tools/codex/run_prompt.py`, using generated prompt files plus wrapper progress artifacts, with no runner redesign.

The planning note I drafted would summarize that sequence as: contract first, then resumable state, then scan output, then per-stage prompt safety, then minimal orchestration last.

notes/025_staged_notebook_wrapper_prompt_plan__20260417_181500.md

## Stderr

```text
OpenAI Codex v0.120.0 (research preview)
--------
workdir: /home/jevert/git/handson-ml3
model: gpt-5.4
provider: openai
approval: never
sandbox: read-only
reasoning effort: none
reasoning summaries: none
session id: 019d9c79-5fbc-7c12-b124-521d10f4d887
--------
user
# Prompt: Generate the staged prompt set for the notebook-wrapper build

You are working inside this repository.

Your task is to use the existing assessment and current tooling layout to create the first draft prompt set that will later be used to build the staged notebook-treatment wrapper.

Do NOT build the wrapper itself yet.

Do NOT modify any `.py` files.
Do NOT modify any notebooks.
Do NOT refactor the current runner.
Do NOT change the V1 record format.

## Goal

We already fenced the problem and assessed what the wrapper should be.

Now generate the prompt files that would let Codex build this wrapper in careful stages.

The wrapper concept is:

- stay centered around Codex calls
- keep the current `tools/codex/run_prompt.py` flow intact
- use narrow, bounded tasks instead of one giant notebook-enrichment run
- support these staged notebook-treatment passes:
  - Stage 1: chapter intro detection / insertion
  - Stage 2: markdown-before-code detection / insertion
  - Stage 3: markdown-after-code detection / insertion

## What to inspect first

Inspect at minimum:

- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `tools/codex/v1_record_validation.py`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/Project_Design_Workflow.md`
- the existing assessment note for this wrapper idea
- `tools/notebook_enricher/notebook_scanner.py`
- `tools/notebook_enricher/prompt_builder.py`

## Your job

Create a practical first-draft prompt set that breaks the future implementation into small Codex-sized tasks.

These prompts should be designed so that each one:
- has a narrow scope
- can be run through the current Codex runner
- produces durable notes and visible progress
- minimizes risk of notebook corruption
- supports stop/resume behavior
- avoids overengineering

## Deliverables

Create:

1. A small sequence of new prompt files in `codex_prompts/` for the wrapper build
2. One planning note in `notes/` that explains:
   - what prompt files you created
   - what each prompt is meant to accomplish
   - why the sequence is ordered this way
   - any open concerns or assumptions

## Requirements for the generated prompt set

The prompt set should cover, in some sensible order:

- defining the wrapper state model
- defining notebook scope/config input
- defining progress tracking and stop/resume behavior
- defining notebook scan output / treatment detection
- defining safe prompt-generation rules for Stage 1, Stage 2, Stage 3
- defining notebook-write safeguards
- defining a minimal orchestration loop
- defining reporting / note artifacts for the wrapper runs
- defining what the MVP should do first

The prompts should not yet ask Codex to:
- enrich all notebooks
- process the whole repo in one pass
- redesign the V1 runner
- add parallel execution
- add a UI
- add advanced scheduling
- add automatic self-healing
- add background daemons

## Naming

Use the next available numeric prefixes in `codex_prompts/`.

Make the prompt filenames descriptive and consistent with the existing repo style.

## Output note structure

Write one markdown note into `notes/` with this structure:

# Staged Notebook Wrapper Prompt Plan

## Executive Summary
## Prompt Files Created
## Why This Sequence
## Scope Boundaries
## Risks Still Open
## Recommended Next Prompt To Run

## Output rules

- Be practical
- Be repo-specific
- Keep the prompt set small and disciplined
- Prefer the minimum viable sequence over a giant speculative backlog
- At the end of your final response, print only the path to the planning note you created

codex
I’m inspecting the current Codex tooling, workflow docs, and the existing notebook-enricher pieces first so the prompt set matches the repo’s current shape rather than inventing a parallel design.
exec
/bin/bash -lc "sed -n '1,220p' tools/codex/review_run.py" in /home/jevert/git/handson-ml3
exec
/bin/bash -lc "sed -n '1,220p' tools/codex/run_prompt.py" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

from v1_record_validation import REVIEW_STATUS_VALUES, validate_record_text


REVIEW_STATUSES = REVIEW_STATUS_VALUES - {"UNREVIEWED"}
FAILURE_FIELDS = (
    "failure_type",
    "failure_symptom",
    "likely_cause",
    "recommended_next_action",
)


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write manual V1 review fields back into an existing execution record."
    )
    parser.add_argument("record", help="Path to an existing execution-record markdown file")
    parser.add_argument(
        "--review-status",
        required=True,
        choices=sorted(REVIEW_STATUSES),
        help="Manual review outcome",
    )
    parser.add_argument(
        "--review-summary",
        required=True,
        help="Short manual review summary",
    )
    parser.add_argument("--reviewed-by", help="Reviewer identifier")
    parser.add_argument(
        "--reviewed-at-utc",
        help="Review timestamp in UTC using YYYYMMDD_HHMMSS; defaults to current UTC time",
    )
    parser.add_argument("--failure-type", help="Short failure category for rejected runs")
    parser.add_argument("--failure-symptom", help="Observed failure symptom for rejected runs")
    parser.add_argument("--likely-cause", help="Likely root cause for rejected runs")
    parser.add_argument(
        "--recommended-next-action",
        help="Manual next action recommendation for rejected runs",
    )
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def require_single_line(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be a single line")
    return value.strip()


def resolve_record_path(record_arg: str) -> Path:
    record_path = Path(record_arg)
    if record_path.is_absolute():
        return record_path
    return repo_root() / record_path


def validate_record_path(record_path: Path) -> None:
    if not record_path.exists():
        raise ValueError(f"record file not found: {record_path}")
    if not record_path.is_file():
        raise ValueError(f"record path is not a file: {record_path}")

    root = repo_root().resolve()
    notes_dir = root / "notes"
    try:
        record_path.resolve().relative_to(notes_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"record must be under {notes_dir}") from exc


def replace_field(text: str, field: str, value: str | None, *, code: bool = False) -> str:
    rendered = f"`{value}`" if code and value else (value or "")
    pattern = re.compile(rf"(?m)^- {re.escape(field)}:.*$")
    if pattern.search(text) is None:
        raise ValueError(f"record is missing field line: {field}")
    return pattern.sub(f"- {field}: {rendered}", text, count=1)


def build_updates(args: argparse.Namespace) -> dict[str, tuple[str | None, bool]]:
    review_summary = require_single_line("review_summary", args.review_summary)
    reviewed_by = require_single_line("reviewed_by", args.reviewed_by)
    reviewed_at_utc = require_single_line("reviewed_at_utc", args.reviewed_at_utc) or utc_timestamp()
    failure_values = {
        "failure_type": require_single_line("failure_type", args.failure_type),
        "failure_symptom": require_single_line("failure_symptom", args.failure_symptom),
        "likely_cause": require_single_line("likely_cause", args.likely_cause),
        "recommended_next_action": require_single_line(
            "recommended_next_action", args.recommended_next_action
        ),
    }

    if args.review_status != "REJECTED" and any(value for value in failure_values.values()):
        raise ValueError("rejection-only failure fields may only be set when --review-status REJECTED")

    updates: dict[str, tuple[str | None, bool]] = {
        "review_status": (args.review_status, True),
        "review_summary": (review_summary, False),
        "reviewed_at_utc": (reviewed_at_utc, True),
    }
    if reviewed_by is not None:
        updates["reviewed_by"] = (reviewed_by, False)

    if args.review_status == "REJECTED":
        for field, value in failure_values.items():
            if value is not None:
                updates[field] = (value, False)

    return updates


def main() -> int:
    try:
        args = parse_args()
        record_path = resolve_record_path(args.record)
        validate_record_path(record_path)
        text = record_path.read_text(encoding="utf-8")
        validate_record_text(text, source=record_path)

        updated = text
        for field, (value, code) in build_updates(args).items():
            updated = replace_field(updated, field, value, code=code)

        validate_record_text(updated, source=record_path)
        record_path.write_text(updated, encoding="utf-8")
    except ValueError as exc:
        return fail(str(exc))

    print(record_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

 succeeded in 0ms:
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from v1_record_validation import validate_record_text


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
    validate_record_text(record_path.read_text(encoding="utf-8"), source=record_path)


exec
/bin/bash -lc "sed -n '1,220p' tools/codex/check_queue_readiness.py" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from v1_record_validation import V1Record, ValidationError, parse_record_file


PROMPTS_DIR = "codex_prompts"
NOTES_DIR = "notes"
PROMPT_NAME_RE = re.compile(r"^(?P<prefix>\d+)_")


@dataclass(frozen=True)
class PromptEntry:
    prefix: int
    path: Path

    @property
    def label(self) -> str:
        return self.path.as_posix()


@dataclass(frozen=True)
class ReadinessResult:
    target: PromptEntry
    previous: PromptEntry | None
    latest_record: V1Record | None
    ready: bool
    reason: str


class ReadinessError(Exception):
    pass


def format_prefixes(prefixes: list[int]) -> str:
    return ", ".join(f"{prefix:03d}" for prefix in prefixes)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report whether the next prompt is ready under the current V1 review gate."
    )
    parser.add_argument(
        "--prompt",
        help="Specific prompt file, filename, or numeric prefix to check",
    )
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def parse_prompt_prefix(path: Path) -> int:
    match = PROMPT_NAME_RE.match(path.stem)
    if match is None:
        raise ReadinessError(f"prompt file is missing a numeric prefix: {path.as_posix()}")
    return int(match.group("prefix"))


def discover_prompts(root: Path) -> list[PromptEntry]:
    prompts_dir = root / PROMPTS_DIR
    if not prompts_dir.exists():
        raise ReadinessError(f"missing prompt directory: {prompts_dir}")

    entries: list[PromptEntry] = []
    seen_prefixes: dict[int, Path] = {}

    for path in sorted(prompts_dir.glob("*.md")):
        prefix = parse_prompt_prefix(path)
        if prefix in seen_prefixes:
            raise ReadinessError(
                "multiple prompt files share the same numeric prefix: "
                f"{seen_prefixes[prefix].as_posix()} and {path.as_posix()}"
            )
        seen_prefixes[prefix] = path
        entries.append(PromptEntry(prefix=prefix, path=path.relative_to(root)))

    if not entries:
        raise ReadinessError(f"no markdown prompt files found in {prompts_dir}")

    return sorted(entries, key=lambda entry: entry.prefix)


def resolve_prompt(prompts: list[PromptEntry], prompt_arg: str) -> PromptEntry:
    trimmed = prompt_arg.strip()
    if not trimmed:
        raise ReadinessError("--prompt must not be empty")

    if trimmed.isdigit():
        prefix = int(trimmed)
        matches = [prompt for prompt in prompts if prompt.prefix == prefix]
        if len(matches) != 1:
            raise ReadinessError(f"no prompt found for numeric prefix: {trimmed}")
        return matches[0]

    normalized = trimmed.rstrip("/")
    matches = [
        prompt
        for prompt in prompts
        if normalized in {prompt.label, prompt.path.name, prompt.path.stem}
    ]
    if len(matches) == 1:
        return matches[0]

    prefix_matches = [prompt for prompt in prompts if prompt.path.name.startswith(normalized)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        raise ReadinessError(f"prompt selector is ambiguous: {trimmed}")

    raise ReadinessError(f"prompt not found: {trimmed}")


def discover_run_records(root: Path) -> list[V1Record]:
    notes_dir = root / NOTES_DIR
    if not notes_dir.exists():
        raise ReadinessError(f"missing notes directory: {notes_dir}")

    records: list[V1Record] = []
    for path in sorted(notes_dir.glob("*.md")):
        try:
            record = parse_record_file(path)
        except ValidationError as exc:
            raise ReadinessError(str(exc)) from exc
        if record is not None:
            records.append(record)
    return records


def discover_legacy_success_prefixes(root: Path) -> set[int]:
    notes_dir = root / NOTES_DIR
    prefixes: set[int] = set()
    for path in sorted(notes_dir.glob("*__SUCCESS__*.md")):
        match = PROMPT_NAME_RE.match(path.stem)
        if match is not None:
            prefixes.add(int(match.group("prefix")))
    return prefixes


def latest_record_for_prompt(records: list[V1Record], prompt: PromptEntry) -> V1Record | None:
    relevant = [record for record in records if record.prompt_file == prompt.label]
    if not relevant:
        return None
    return max(relevant, key=lambda record: (record.started_at_utc, record.run_suffix))


def default_target(prompts: list[PromptEntry], records: list[V1Record]) -> PromptEntry:
    for index, prompt in enumerate(prompts):
        if index == 0:
            latest = latest_record_for_prompt(records, prompt)
            if latest is None or latest.review_status != "ACCEPTED":
                return prompt
            continue

        previous = prompts[index - 1]
        latest_previous = latest_record_for_prompt(records, previous)
        if latest_previous is None or latest_previous.review_status != "ACCEPTED":
            return prompt

        latest_current = latest_record_for_prompt(records, prompt)
        if latest_current is None or latest_current.review_status != "ACCEPTED":
            return prompt

    raise ReadinessError("all discovered prompts already have ACCEPTED latest V1 runs; no next prompt remains")


def evaluate_readiness(
    prompts: list[PromptEntry],
    records: list[V1Record],
    target: PromptEntry,
) -> ReadinessResult:
    index = next((i for i, prompt in enumerate(prompts) if prompt == target), None)
    if index is None:
        raise ReadinessError(f"target prompt is not in the discovered prompt list: {target.label}")

    if index == 0:
        return ReadinessResult(
            target=target,
            previous=None,
            latest_record=None,
            ready=True,
            reason="first prompt in sequence has no prior review gate",
        )

    previous = prompts[index - 1]
    latest = latest_record_for_prompt(records, previous)
    if latest is None:
        return ReadinessResult(
            target=target,
            previous=previous,
            latest_record=None,
            ready=False,
            reason="missing V1 run evidence for the immediately previous prompt",
        )

    if latest.review_status == "ACCEPTED":
        return ReadinessResult(
            target=target,
            previous=previous,
            latest_record=latest,
            ready=True,
            reason="latest V1 run for the immediately previous prompt is ACCEPTED",
        )

    if latest.review_status == "UNREVIEWED":
        reason = "latest V1 run for the immediately previous prompt is UNREVIEWED"
    else:

exec
/bin/bash -lc "sed -n '1,220p' tools/codex/list_review_backlog.py" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from v1_record_validation import V1Record, ValidationError, parse_record_file


NOTES_DIR = "notes"


class BacklogError(Exception):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List the current V1 review backlog from execution records in notes/."
    )
    parser.add_argument(
        "--unreviewed-only",
        action="store_true",
        help="Limit the latest-per-prompt and needs-review views to UNREVIEWED latest records.",
    )
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def discover_run_records(root: Path) -> list[V1Record]:
    notes_dir = root / NOTES_DIR
    if not notes_dir.exists():
        raise BacklogError(f"missing notes directory: {notes_dir}")

    records: list[V1Record] = []
    for path in sorted(notes_dir.glob("*.md")):
        try:
            record = parse_record_file(path)
        except ValidationError as exc:
            raise BacklogError(str(exc)) from exc
        if record is not None:
            records.append(record)
    return records


def latest_records_by_prompt(records: list[V1Record]) -> list[V1Record]:
    latest_by_prompt: dict[str, V1Record] = {}
    for record in records:
        current = latest_by_prompt.get(record.prompt_file)
        if current is None or (record.started_at_utc, record.run_suffix) > (
            current.started_at_utc,
            current.run_suffix,
        ):
            latest_by_prompt[record.prompt_file] = record

    return sorted(latest_by_prompt.values(), key=lambda record: record.prompt_stem)


def render_record(record: V1Record) -> str:
    return (
        f"- {record.path.as_posix()} | prompt={record.prompt_file} | started={record.started_at_utc} | "
        f"execution={record.execution_status} | review={record.review_status}"
    )


def print_section(title: str, records: list[V1Record]) -> None:
    print(title)
    if not records:
        print("- none")
        return

    for record in records:
        print(render_record(record))


def main() -> int:
    try:
        args = parse_args()
        root = repo_root()
        records = discover_run_records(root)
        unreviewed = sorted(
            [record for record in records if record.review_status == "UNREVIEWED"],
            key=lambda record: (record.started_at_utc, record.run_suffix, record.prompt_stem),
        )
        latest = latest_records_by_prompt(records)
        needs_review_next = [record for record in latest if record.review_status == "UNREVIEWED"]

        latest_view = latest if not args.unreviewed_only else needs_review_next

        print(f"V1 review backlog summary from {NOTES_DIR}/")
        print(f"Discovered V1 execution records: {len(records)}")
        print(f"Unreviewed records: {len(unreviewed)}")
        print(f"Prompts with latest record: {len(latest)}")
        print()

        print_section("UNREVIEWED records:", unreviewed)
        print()
        print_section("Latest record per prompt:", latest_view)
        print()
        print_section("Likely needs human review next:", needs_review_next)
        return 0
    except BacklogError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

codex
I have the runner/review gate shape now. Next I’m locating the wrapper assessment note and the notebook-enricher inputs so the new prompt sequence fits the existing evidence trail and avoids asking for incompatible artifacts.
exec
/bin/bash -lc "sed -n '1,260p' tools/codex/v1_record_validation.py" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


EXECUTION_STATUS_VALUES = {"EXECUTED", "EXECUTION_FAILED"}
REVIEW_STATUS_VALUES = {"UNREVIEWED", "ACCEPTED", "REJECTED"}
REQUIRED_SECTIONS = (
    "## Execution Facts",
    "## Review Facts",
    "## Failure Analysis",
    "## Resource / Cost Facts",
    "## Prompt Text",
    "## Codex Final Output",
    "## Stderr",
)
REQUIRED_FIELDS = (
    "run_id",
    "prompt_file",
    "prompt_stem",
    "started_at_utc",
    "execution_status",
    "finished_at_utc",
    "runner",
    "return_code",
    "retry_of_run_id",
    "review_status",
    "review_summary",
    "reviewed_by",
    "reviewed_at_utc",
    "failure_type",
    "failure_symptom",
    "likely_cause",
    "recommended_next_action",
    "elapsed_seconds",
    "final_output_char_count",
    "stderr_char_count",
)
TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class V1Record:
    path: Path | None
    title: str
    run_id: str
    prompt_file: str
    prompt_stem: str
    started_at_utc: str
    execution_status: str
    finished_at_utc: str
    runner: str
    return_code: str
    retry_of_run_id: str
    review_status: str
    review_summary: str
    reviewed_by: str
    reviewed_at_utc: str
    failure_type: str
    failure_symptom: str
    likely_cause: str
    recommended_next_action: str
    elapsed_seconds: str
    final_output_char_count: str
    stderr_char_count: str
    run_suffix: int


def _source_label(source: Path | str | None) -> str:
    if source is None:
        return "<text>"
    if isinstance(source, Path):
        return source.as_posix()
    return source


def parse_field_line(text: str, field: str) -> str | None:
    match = re.search(rf"(?m)^- {re.escape(field)}:\s*(.*)$", text)
    if match is None:
        return None

    value = match.group(1).strip()
    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return value


def looks_like_v1_record(text: str) -> bool:
    preamble = text.split("```", 1)[0]
    has_run_id = parse_field_line(preamble, "run_id") is not None
    has_execution_section = "## Execution Facts" in preamble
    has_review_section = "## Review Facts" in preamble
    return has_run_id or (has_execution_section and has_review_section)


def parse_run_suffix(prompt_stem: str, started_at_utc: str, run_id: str) -> int:
    base_run_id = f"{prompt_stem}__{started_at_utc}"
    if run_id == base_run_id:
        return 1

    match = re.fullmatch(rf"{re.escape(base_run_id)}__(\d+)", run_id)
    if match is None:
        raise ValidationError(f"run_id does not match V1 identity pattern: {run_id}")

    suffix = int(match.group(1))
    if suffix < 2:
        raise ValidationError(f"run_id suffix must be >= 2 when present: {run_id}")
    return suffix


def validate_record_text(text: str, *, source: Path | str | None = None) -> V1Record:
    label = _source_label(source)
    if not text.startswith("# "):
        raise ValidationError(f"record does not start with a markdown title: {label}")

    positions: list[int] = []
    for section in REQUIRED_SECTIONS:
        pos = text.find(section)
        if pos == -1:
            raise ValidationError(f"record is missing section: {section} ({label})")
        positions.append(pos)
    if positions != sorted(positions):
        raise ValidationError(f"record sections are out of the expected V1 order: {label}")

    values: dict[str, str] = {}
    for field in REQUIRED_FIELDS:
        value = parse_field_line(text, field)
        if value is None:
            raise ValidationError(f"record is missing field line: {field} ({label})")
        values[field] = value

    title = text.splitlines()[0][2:].strip()
    run_id = values["run_id"]
    prompt_file = values["prompt_file"]
    prompt_stem = values["prompt_stem"]
    started_at_utc = values["started_at_utc"]
    execution_status = values["execution_status"]
    review_status = values["review_status"]

    if title != run_id:
        raise ValidationError(f"record title/run_id mismatch: {label}")
    if not TIMESTAMP_RE.fullmatch(started_at_utc):
        raise ValidationError(f"record has invalid started_at_utc timestamp: {label}")
    if execution_status not in EXECUTION_STATUS_VALUES:
        raise ValidationError(f"record has invalid execution_status: {label}")
    if review_status not in REVIEW_STATUS_VALUES:
        raise ValidationError(f"record has invalid review_status: {label}")

    if Path(prompt_file).stem != prompt_stem:
        raise ValidationError(f"record prompt_file/prompt_stem mismatch: {label}")

    run_suffix = parse_run_suffix(prompt_stem, started_at_utc, run_id)

    if isinstance(source, Path) and source.suffix == ".md" and source.stem != run_id:
        raise ValidationError(f"record filename/run_id mismatch: {label}")

    return V1Record(
        path=source if isinstance(source, Path) else None,
        title=title,
        run_id=run_id,
        prompt_file=prompt_file,
        prompt_stem=prompt_stem,
        started_at_utc=started_at_utc,
        execution_status=execution_status,
        finished_at_utc=values["finished_at_utc"],
        runner=values["runner"],
        return_code=values["return_code"],
        retry_of_run_id=values["retry_of_run_id"],
        review_status=review_status,
        review_summary=values["review_summary"],
        reviewed_by=values["reviewed_by"],
        reviewed_at_utc=values["reviewed_at_utc"],
        failure_type=values["failure_type"],
        failure_symptom=values["failure_symptom"],
        likely_cause=values["likely_cause"],
        recommended_next_action=values["recommended_next_action"],
        elapsed_seconds=values["elapsed_seconds"],
        final_output_char_count=values["final_output_char_count"],
        stderr_char_count=values["stderr_char_count"],
        run_suffix=run_suffix,
    )


def parse_record_file(path: Path) -> V1Record | None:
    text = path.read_text(encoding="utf-8")
    if not looks_like_v1_record(text):
        return None
    return validate_record_text(text, source=path)

exec
/bin/bash -lc "sed -n '1,260p' tools/Codex_Prompt_Workflow_Architecture.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
# Codex Prompt Workflow Architecture

## Goal

Build a small, conservative prompt workflow system for Codex work inside a repository.

This system should stay:

* easy to inspect
* easy to test
* easy to extend
* composed of small parts
* safe by default

The design should emphasize **separation of concerns** so each piece does one job clearly.

---

## V1 Problem Statement

The immediate V1 problem is not "how do we build a larger prompt platform?"

It is:

* how to execute one bounded prompt run
* how to preserve a durable record of that run
* how to stop for human review before the next prompt is treated as ready

The current gap is that prompt execution exists, but the workflow boundary between:

* execution
* durable evidence
* accepted progress

is still too loose.

That gap matters because the governing workflow requires:

* thin slices before broad automation
* review between iterations
* durable local history
* failure evidence that supports refinement instead of blind retry

So V1 should solve the smallest workflow problem that makes the sequence inspectable:

* one run
* one durable execution record
* one manual review gate
* one conservative rule for whether the next prompt may proceed

---

## Why We Are Re-Architecting

The first draft bundled too many responsibilities into one script. That made it harder to:

* understand
* trust
* test
* evolve
* reuse across repositories

We want a cleaner structure, but V1 should only separate what is necessary to support the governing workflow.

This is still a design-alignment step, not a broad implementation push.

---

## V1 Scope

V1 covers the minimum workflow slice from prompt execution through later human review.

It includes:

* selecting and executing one prompt
* writing one durable execution record in `notes/`
* keeping execution outcome separate from review outcome
* stopping the queue at `UNREVIEWED`
* allowing only an explicit accepted review outcome to release the next prompt

V1 is intentionally about workflow control and artifact clarity, not about architectural breadth.

---

## Out Of Scope For V1

The following items are explicitly out of scope in this stage:

* runner refactor into many modules
* dependency-aware scheduling
* richer queue state machines
* automatic approval or review routing
* aggregated reports and dashboards
* retry intelligence beyond minimal retry linkage
* token accounting beyond optional lightweight fields
* broad CLI redesign
* reusable multi-repo configuration systems
* structured sidecars, databases, or platform services

These may become useful later, but they should not shape V1 beyond clear deferral notes.

---

## Minimum Artifact Inventory

The governing workflow expects the minimum required artifacts to be explicit.

For V1, the minimum inventory is:

### 1. Governing workflow

`tools/Project_Design_Workflow.md`

This remains the controlling sequence for:

* bounded execution
* validation
* review between iterations
* deferral discipline

### 2. Architecture document

`tools/Codex_Prompt_Workflow_Architecture.md`

This document should explain:

* the V1 problem being solved
* the V1 scope boundary
* the minimum parts and artifacts
* the conservative implementation order
* what remains deferred

### 3. V1 execution record definition

`tools/codex/V1_Execution_Record_Artifact.md`

This defines the minimum durable source of truth for one run.

Its role in the architecture is to make one run inspectable without requiring a larger system.

### 4. V1 review gate definition

`tools/codex/V1_Run_Review_Gate.md`

This defines the smallest manual stop between one completed run and the next prompt becoming ready.

Its role in the architecture is to keep execution success separate from accepted progress.

### 5. Execution record files in `notes/`

For V1, one run should produce one markdown execution-record file in `notes/`.

That record is the minimum durable run artifact.

No additional required queue file, database, or sidecar should be introduced in V1.

### 6. Current V1 helper scripts

The current implemented V1 toolset is:

* `tools/codex/run_prompt.py` for bounded execution and execution-record creation
* `tools/codex/review_run.py` for manual review write-back into the same record
* `tools/codex/check_queue_readiness.py` for conservative next-prompt readiness checks
* `tools/codex/list_review_backlog.py` for review backlog inspection from `notes/`

These helpers operate on the same markdown execution-record contract.
They are part of the current V1 workflow surface, not future placeholders.

---

## Minimum Viable Slice

The minimum viable slice is:

1. execute one prompt
2. write one execution record with `review_status: UNREVIEWED`
3. stop for human review
4. update the same execution record to either `ACCEPTED` or `REJECTED`
5. treat only `ACCEPTED` as releasing the next prompt in sequence

This is the smallest slice that proves the workflow rather than only the runner.

It is intentionally narrower than:

* a generalized queue engine
* a full prompt-status system
* a modular tool suite
* an automation framework

---

## Role Of The Execution Record

The execution record is the central V1 artifact.

Its purpose is to preserve, in one inspectable markdown file:

* what prompt was run
* what happened during execution
* what Codex returned
* what lightweight failure or resource evidence was observed
* what a human later decided about the run

For V1:

* the record body is the source of truth
* the file in `notes/` is the durable local history unit
* one run should not be split across multiple required files

This keeps V1 small while still supporting:

* review between iterations
* failure analysis
* lightweight cost awareness
* stable run identity

The execution record defined in `tools/codex/V1_Execution_Record_Artifact.md` should therefore be treated as a required architectural element, not an implementation detail.

---

## Role Of The Review Gate

The review gate is the minimum manual checkpoint after execution record creation.

Its purpose is to enforce the governing workflow rule that the next bounded step should not proceed only because a process finished successfully.

For V1:

* execution writes a record
* new records begin as `UNREVIEWED`
* human review decides `ACCEPTED` or `REJECTED`
* only `ACCEPTED` allows the next prompt to be treated as ready

This means the review gate is not a separate platform subsystem.

It is the manual transition of a single execution record from:

* `UNREVIEWED`

to one of:

* `ACCEPTED`
* `REJECTED`

The review gate defined in `tools/codex/V1_Run_Review_Gate.md` is therefore part of the architecture boundary for V1, not an optional later add-on.

---

## What The First Script Was Doing

Below is a decomposition of the responsibilities that were bundled together.

### 1. Repo path discovery

* figure out where the repository root is
* infer where `codex_prompts/` and `notes/` live

### 2. Directory validation

* verify the expected folders exist

exec
/bin/bash -lc "sed -n '1,260p' tools/Project_Design_Workflow.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
# Project Design Workflow

## Goal

Define a repeatable workflow for turning an idea or architecture document into a clean, working system without skipping the thinking that keeps the build safe, clear, and reusable.

This document is meant to sit beside architecture documents and help answer a practical question:

**What are the steps between "this design makes sense" and "the thing works"?**

---

## Why This Exists

Sometimes a project feels sound at the architecture level but still feels fuzzy at the build level.

That happens because architecture is not the same thing as execution.

Architecture tells us:

* what parts exist
* what they are responsible for
* how they relate
* what should not be coupled

But architecture does not yet tell us:

* what to build first
* what order reduces risk
* what to test first
* what can wait
* how to know whether the design is holding
* how to recover cleanly when a step fails

This workflow bridges that gap.

---

## Relationship to a Larger Design Process

This workflow does **not** replace broader project thinking.
It fits inside it.

A useful mapping looks like this:

### 1. Conversation

What are we trying to do?
What hurts right now?
What would "better" feel like?

### 2. Goals

What must become true for this project to count as successful?

### 3. Deliverables

What concrete artifacts must exist?
Examples:

* architecture doc
* workflow doc
* module layout
* interface sketch
* notes folder
* validation checklist
* first working slice

### 4. Tasks

What actions create those deliverables?
Examples:

* define V1 scope
* define module boundaries
* choose naming conventions
* write first thin component
* validate one path end to end
* capture review notes

### 5. Implementation

Only after the earlier layers are stable do we write implementation.

So this workflow is not separate from design.
It is the part that turns design into bounded execution.

---

## Core Principles

### 1. Design before build

Do not use implementation momentum to hide unclear thinking.

### 2. Boundaries before breadth

A project becomes safer when scope, responsibility, and deferral lines are visible.

### 3. Thin slices before large pushes

Prefer a bounded slice that proves something real over a large batch that creates many unknowns at once.

### 4. Review between iterations

Each bounded execution step should be inspected before the next one is issued.
Do not assume a sequence is healthy just because the previous step completed.

### 5. Validation is part of design

Tests, inspections, examples, and acceptance checks are not extra work.
They are how the design proves it is real.

### 6. Bridge tooling is allowed, but subordinate

Temporary or bridge tooling can be useful when it stays thin, inspectable, and reversible.
It may help move work forward, gather evidence, or reduce manual friction.
It should not become a substitute for architecture, clear interfaces, or good decomposition.

### 7. Durable local history matters

Notes, logs, reports, generated outputs, and similar local artifacts can serve as durable project memory.
They help preserve decisions, validation results, failure patterns, and the reasoning behind refinements.

### 8. Failure should produce analysis, not just retries

When a step fails, the useful question is not only "how do we try again?"
It is also "what did this failure reveal about the task, decomposition, criteria, or environment?"

### 9. Resource use should be observed

Project effort has a cost, whether that cost is human time, machine time, review effort, retries, execution size, or external spend.
Large tasks should justify themselves.

---

## The Core Sequence

Here is the recommended design workflow.

### Phase 1: Clarify the problem

This is the "why are we doing this?" phase.

Questions:

* What pain are we removing?
* What confusion are we reducing?
* What repeated work are we trying to standardize?
* What is dangerous if we build too fast?

Output:

* short problem statement
* short success statement

---

### Phase 2: Define the system boundary

This is the "what belongs in this project and what does not?" phase.

Questions:

* What should this system do?
* What should it explicitly not do yet?
* What decisions are deferred?
* What adjacent problems are real but out of scope?

Output:

* scope statement
* out-of-scope list

---

### Phase 3: Draft the architecture

This is the "what pieces exist and what are their jobs?" phase.

Questions:

* What modules, components, or services are needed?
* What is each one responsible for?
* What should each one never own?
* Where are the seams between parts?

Output:

* architecture doc
* responsibility split
* proposed file, package, or interface layout

---

### Phase 4: Identify the minimum viable slice

This is the "what is the smallest useful thing we can build that proves the design?" phase.

Questions:

* What is the thinnest vertical slice that is actually useful?
* What can we test without building the whole system?
* What gives us signal early?
* What first slice is unlikely to trap us later?

Output:

* V1 feature list
* initial build plan

---

### Phase 5: Define the artifacts

This is the "what files or outputs must exist?" phase.

Questions:

* What documents should exist?
* What modules or interfaces should exist?
* What examples, fixtures, or test inputs should exist?
* What outputs prove the path works?
* What notes, logs, or reports should be kept as durable project memory?

Output:

* deliverables list
* artifact inventory

---

### Phase 6: Sequence the work

This is the "what order reduces pain and risk?" phase.

Questions:

* What must come first because other things depend on it?
* What can be tested independently?
* What pieces should be proven before automation or convenience layers are added?
* What order keeps the build inspectable?

Output:

* implementation order
* dependency chain

---

### Phase 7: Define validation

This is the "how will we know each layer works?" phase.

Questions:

* What is the smoke test for each part?
* What is a good manual test before automation?
* What failure modes do we expect?
* What evidence counts as success?

exec
/bin/bash -lc "rg -n \"wrapper|notebook-treatment|staged notebook|notebook wrapper|assessment\" notes tools codex_prompts -g '*.md'" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
codex_prompts/024_generate_staged_notebook_wrapper_prompts.md:1:# Prompt: Generate the staged prompt set for the notebook-wrapper build
codex_prompts/024_generate_staged_notebook_wrapper_prompts.md:5:Your task is to use the existing assessment and current tooling layout to create the first draft prompt set that will later be used to build the staged notebook-treatment wrapper.
codex_prompts/024_generate_staged_notebook_wrapper_prompts.md:7:Do NOT build the wrapper itself yet.
codex_prompts/024_generate_staged_notebook_wrapper_prompts.md:16:We already fenced the problem and assessed what the wrapper should be.
codex_prompts/024_generate_staged_notebook_wrapper_prompts.md:18:Now generate the prompt files that would let Codex build this wrapper in careful stages.
codex_prompts/024_generate_staged_notebook_wrapper_prompts.md:20:The wrapper concept is:
codex_prompts/024_generate_staged_notebook_wrapper_prompts.md:25:- support these staged notebook-treatment passes:
codex_prompts/024_generate_staged_notebook_wrapper_prompts.md:41:- the existing assessment note for this wrapper idea
codex_prompts/024_generate_staged_notebook_wrapper_prompts.md:61:1. A small sequence of new prompt files in `codex_prompts/` for the wrapper build
codex_prompts/024_generate_staged_notebook_wrapper_prompts.md:72:- defining the wrapper state model
codex_prompts/024_generate_staged_notebook_wrapper_prompts.md:79:- defining reporting / note artifacts for the wrapper runs
codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md:5:Your task is to review two files against the current workflow guidance and produce a small design assessment.
codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md:38:This pass should stay focused on assessment, mismatch detection, and next-step recommendation.
codex_prompts/004_review_architecture_and_bridge_runner_against_workflow.md:127:- the assessment is grounded in `tools/Project_Design_Workflow.md`
codex_prompts/023_tools_codex_assessment.md:1:# Prompt: Assess wrapper design for staged notebook-treatment runs using the existing Codex runner
codex_prompts/023_tools_codex_assessment.md:5:Your job is NOT to implement the notebook wrapper yet.
codex_prompts/023_tools_codex_assessment.md:9:2. assess how a staged notebook-treatment wrapper could be built around the current Codex-centered workflow,
codex_prompts/023_tools_codex_assessment.md:17:The desired future behavior is a wrapper that can launch one narrow Codex task at a time against a notebook set, instead of asking Codex to solve the entire notebook in one pass.
codex_prompts/023_tools_codex_assessment.md:53:- Do NOT implement the wrapper yet
codex_prompts/023_tools_codex_assessment.md:76:- what a wrapper would need to do without breaking the current model
codex_prompts/023_tools_codex_assessment.md:86:1. Can this staged notebook-treatment workflow be built cleanly while staying centered on the current Codex runner?
codex_prompts/023_tools_codex_assessment.md:87:2. What is the smallest viable wrapper design that would make this work?
codex_prompts/023_tools_codex_assessment.md:88:3. Should the wrapper:
codex_prompts/023_tools_codex_assessment.md:93:5. How should the wrapper know:
codex_prompts/023_tools_codex_assessment.md:111:`notes/XXX_staged_notebook_wrapper_assessment__YYYYMMDD_HHMMSS.md`
codex_prompts/023_tools_codex_assessment.md:129:- restate the three-stage notebook-treatment concept in implementation terms
codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md:51:### 1. Runner assessment
codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md:56:### 2. Artifact assessment
codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md:61:### 3. Operational assessment
codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md:86:- runner assessment
codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md:87:- artifact assessment
codex_prompts/010_review_run_prompt_candidate_and_recommend_next_step.md:88:- operational assessment
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:60:### 1. Runner assessment
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:65:### 2. Artifact assessment
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:70:### 3. Operational assessment
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:95:- runner assessment
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:96:- artifact assessment
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:97:- operational assessment
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:207:### 1. Runner assessment
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:212:### 2. Artifact assessment
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:217:### 3. Operational assessment
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:242:- runner assessment
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:243:- artifact assessment
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md:244:- operational assessment
notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md:3375:### 1. Runner assessment
notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md:3380:### 2. Artifact assessment
notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md:3385:### 3. Operational assessment
notes/022_scaffolding_classification_report__20260416_023016.md:22:- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
notes/022_scaffolding_classification_report__20260416_023016.md:47:  Reason: captures the first serious assessment of the new runner and the environment-vs-runner distinction.
notes/022_scaffolding_classification_report__20260416_023016.md:68:  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
notes/022_scaffolding_classification_report__20260416_023016.md:72:  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
notes/022_scaffolding_classification_report__20260416_023016.md:74:  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
notes/022_scaffolding_classification_report__20260416_023016.md:113:  Reason: review scaffold whose durable output is the architecture/runner assessment note.
notes/022_scaffolding_classification_report__20260416_023016.md:187:- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md:1123:Your task is to review two files against the current workflow guidance and produce a small design assessment.
notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md:1156:This pass should stay focused on assessment, mismatch detection, and next-step recommendation.
notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md:1245:- the assessment is grounded in `tools/Project_Design_Workflow.md`
notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md:14:Your task is to review two files against the current workflow guidance and produce a small design assessment.
notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md:47:This pass should stay focused on assessment, mismatch detection, and next-step recommendation.
notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md:136:- the assessment is grounded in `tools/Project_Design_Workflow.md`
notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md:148:The assessment’s main conclusion is that the architecture doc is directionally right on decomposition and conservative growth, and the runner is thin enough to count as acceptable bridge tooling, but both are weak at the same seam: they do not yet define a durable, reviewable execution record that separates raw execution from reviewed outcome and captures minimal failure/resource evidence.
notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md:172:Your task is to review two files against the current workflow guidance and produce a small design assessment.
notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md:205:This pass should stay focused on assessment, mismatch detection, and next-step recommendation.
notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md:294:- the assessment is grounded in `tools/Project_Design_Workflow.md`
notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md:988:I have the governing principles and the current runner shape. I’m pulling the rest of the workflow guidance now so the assessment stays tied to the full sequence, especially validation, review loops, failure handling, and artifact expectations.
notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md:1943:I’m writing the assessment files now. The report will stay anchored to the workflow principles and the existing runner behavior, and the second file will name one bounded design artifact to produce next while explicitly deferring broader refactoring.
notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md:2145:The two notes are written. I’m doing a quick verification pass on the generated files now to make sure the assessment stayed bounded, answered the requested questions, and kept the recommendation to one concrete design artifact.
notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md:2445:The assessment’s main conclusion is that the architecture doc is directionally right on decomposition and conservative growth, and the runner is thin enough to count as acceptable bridge tooling, but both are weak at the same seam: they do not yet define a durable, reviewable execution record that separates raw execution from reviewed outcome and captures minimal failure/resource evidence.
notes/023_tools_codex_assessment__20260417_171721.md:1:# 023_tools_codex_assessment__20260417_171721
notes/023_tools_codex_assessment__20260417_171721.md:3:- run_id: `023_tools_codex_assessment__20260417_171721`
notes/023_tools_codex_assessment__20260417_171721.md:4:- prompt_file: `codex_prompts/023_tools_codex_assessment.md`
notes/023_tools_codex_assessment__20260417_171721.md:5:- prompt_stem: `023_tools_codex_assessment`
notes/023_tools_codex_assessment__20260417_171721.md:39:# Prompt: Assess wrapper design for staged notebook-treatment runs using the existing Codex runner
notes/023_tools_codex_assessment__20260417_171721.md:43:Your job is NOT to implement the notebook wrapper yet.
notes/023_tools_codex_assessment__20260417_171721.md:47:2. assess how a staged notebook-treatment wrapper could be built around the current Codex-centered workflow,
notes/023_tools_codex_assessment__20260417_171721.md:55:The desired future behavior is a wrapper that can launch one narrow Codex task at a time against a notebook set, instead of asking Codex to solve the entire notebook in one pass.
notes/023_tools_codex_assessment__20260417_171721.md:91:- Do NOT implement the wrapper yet
notes/023_tools_codex_assessment__20260417_171721.md:114:- what a wrapper would need to do without breaking the current model
notes/023_tools_codex_assessment__20260417_171721.md:124:1. Can this staged notebook-treatment workflow be built cleanly while staying centered on the current Codex runner?
notes/023_tools_codex_assessment__20260417_171721.md:125:2. What is the smallest viable wrapper design that would make this work?
notes/023_tools_codex_assessment__20260417_171721.md:126:3. Should the wrapper:
notes/023_tools_codex_assessment__20260417_171721.md:131:5. How should the wrapper know:
notes/023_tools_codex_assessment__20260417_171721.md:149:`notes/XXX_staged_notebook_wrapper_assessment__YYYYMMDD_HHMMSS.md`
notes/023_tools_codex_assessment__20260417_171721.md:167:- restate the three-stage notebook-treatment concept in implementation terms
notes/023_tools_codex_assessment__20260417_171721.md:210:Unable to create `/home/jevert/git/handson-ml3/notes/023_staged_notebook_wrapper_assessment__20260417_121843.md` because the workspace is read-only.
notes/023_tools_codex_assessment__20260417_171721.md:227:# Prompt: Assess wrapper design for staged notebook-treatment runs using the existing Codex runner
notes/023_tools_codex_assessment__20260417_171721.md:231:Your job is NOT to implement the notebook wrapper yet.
notes/023_tools_codex_assessment__20260417_171721.md:235:2. assess how a staged notebook-treatment wrapper could be built around the current Codex-centered workflow,
notes/023_tools_codex_assessment__20260417_171721.md:243:The desired future behavior is a wrapper that can launch one narrow Codex task at a time against a notebook set, instead of asking Codex to solve the entire notebook in one pass.
notes/023_tools_codex_assessment__20260417_171721.md:279:- Do NOT implement the wrapper yet
notes/023_tools_codex_assessment__20260417_171721.md:302:- what a wrapper would need to do without breaking the current model
notes/023_tools_codex_assessment__20260417_171721.md:312:1. Can this staged notebook-treatment workflow be built cleanly while staying centered on the current Codex runner?
notes/023_tools_codex_assessment__20260417_171721.md:313:2. What is the smallest viable wrapper design that would make this work?
notes/023_tools_codex_assessment__20260417_171721.md:314:3. Should the wrapper:
notes/023_tools_codex_assessment__20260417_171721.md:319:5. How should the wrapper know:
notes/023_tools_codex_assessment__20260417_171721.md:337:`notes/XXX_staged_notebook_wrapper_assessment__YYYYMMDD_HHMMSS.md`
notes/023_tools_codex_assessment__20260417_171721.md:355:- restate the three-stage notebook-treatment concept in implementation terms
notes/023_tools_codex_assessment__20260417_171721.md:396:I’m inspecting the existing `tools/codex/` runner flow and related notes first, then I’ll map that to the staged notebook-wrapper idea and produce a single planning report in `notes/` if the workspace allows it.
notes/023_tools_codex_assessment__20260417_171721.md:467:?? codex_prompts/023_tools_codex_assessment.md
notes/023_tools_codex_assessment__20260417_171721.md:470:I’ve identified the current runner scripts and the main V1 design docs. Next I’m reading the runner, review, validation, and prior notes to pin down what the wrapper can reuse without changing the current execution model.
notes/023_tools_codex_assessment__20260417_171721.md:1509:The runner is very small: one prompt file in, one `codex exec` call, one V1 record out to `notes/`. I’m reading the design notes now to see how strongly the repo expects wrappers to stay outside that contract versus extending it.
notes/023_tools_codex_assessment__20260417_171721.md:2382:/bin/bash -lc "rg -n \"notebook|treatment|staged|wrapper|bridge runner|run_prompt|review gate|queue\" notes tools -g '*.md'" in /home/jevert/git/handson-ml3
notes/023_tools_codex_assessment__20260417_171721.md:2666:notes/022_scaffolding_classification_report__20260416_023016.md:22:- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
notes/023_tools_codex_assessment__20260417_171721.md:2674:notes/022_scaffolding_classification_report__20260416_023016.md:68:  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
notes/023_tools_codex_assessment__20260417_171721.md:2675:notes/022_scaffolding_classification_report__20260416_023016.md:72:  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
notes/023_tools_codex_assessment__20260417_171721.md:2676:notes/022_scaffolding_classification_report__20260416_023016.md:74:  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
notes/023_tools_codex_assessment__20260417_171721.md:2693:notes/022_scaffolding_classification_report__20260416_023016.md:187:- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
notes/023_tools_codex_assessment__20260417_171721.md:7179:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13031:+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
notes/023_tools_codex_assessment__20260417_171721.md:7187:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13077:+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
notes/023_tools_codex_assessment__20260417_171721.md:7188:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13081:+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
notes/023_tools_codex_assessment__20260417_171721.md:7189:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13083:+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
notes/023_tools_codex_assessment__20260417_171721.md:7206:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13196:+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
notes/023_tools_codex_assessment__20260417_171721.md:7208:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13227:+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
notes/023_tools_codex_assessment__20260417_171721.md:7216:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13273:+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
notes/023_tools_codex_assessment__20260417_171721.md:7217:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13277:+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
notes/023_tools_codex_assessment__20260417_171721.md:7218:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13279:+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
notes/023_tools_codex_assessment__20260417_171721.md:7235:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13392:+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
notes/023_tools_codex_assessment__20260417_171721.md:7237:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13426:+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
notes/023_tools_codex_assessment__20260417_171721.md:7245:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13472:+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
notes/023_tools_codex_assessment__20260417_171721.md:7246:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13476:+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
notes/023_tools_codex_assessment__20260417_171721.md:7247:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13478:+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
notes/023_tools_codex_assessment__20260417_171721.md:7264:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13591:+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
notes/023_tools_codex_assessment__20260417_171721.md:7266:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13648:+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
notes/023_tools_codex_assessment__20260417_171721.md:7274:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13694:+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
notes/023_tools_codex_assessment__20260417_171721.md:7275:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13698:+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
notes/023_tools_codex_assessment__20260417_171721.md:7276:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13700:+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
notes/023_tools_codex_assessment__20260417_171721.md:7293:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13813:+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
notes/023_tools_codex_assessment__20260417_171721.md:7295:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13875:- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
notes/023_tools_codex_assessment__20260417_171721.md:7303:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13921:  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
notes/023_tools_codex_assessment__20260417_171721.md:7304:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13925:  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
notes/023_tools_codex_assessment__20260417_171721.md:7305:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13927:  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
notes/023_tools_codex_assessment__20260417_171721.md:7322:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14040:- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
notes/023_tools_codex_assessment__20260417_171721.md:7324:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14095:+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
notes/023_tools_codex_assessment__20260417_171721.md:7332:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14141:+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
notes/023_tools_codex_assessment__20260417_171721.md:7333:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14145:+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
notes/023_tools_codex_assessment__20260417_171721.md:7334:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14147:+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
notes/023_tools_codex_assessment__20260417_171721.md:7351:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14260:+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
notes/023_tools_codex_assessment__20260417_171721.md:7354:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14321:+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
notes/023_tools_codex_assessment__20260417_171721.md:7362:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14367:+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
notes/023_tools_codex_assessment__20260417_171721.md:7363:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14371:+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
notes/023_tools_codex_assessment__20260417_171721.md:7364:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14373:+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
notes/023_tools_codex_assessment__20260417_171721.md:7381:notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14486:+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
notes/023_tools_codex_assessment__20260417_171721.md:8360:I found a notebook-triage note that’s directly relevant: it already frames chapter intro coverage and per-code-cell treatment coverage in the same terms as your proposed stages. I’m checking a few notebook structures now just to confirm what a wrapper would need to target cell-by-cell.
notes/023_tools_codex_assessment__20260417_171721.md:8563:/bin/bash -lc "sed -n '1,240p' codex_prompts/023_tools_codex_assessment.md" in /home/jevert/git/handson-ml3
notes/023_tools_codex_assessment__20260417_171721.md:8565:# Prompt: Assess wrapper design for staged notebook-treatment runs using the existing Codex runner
notes/023_tools_codex_assessment__20260417_171721.md:8569:Your job is NOT to implement the notebook wrapper yet.
notes/023_tools_codex_assessment__20260417_171721.md:8573:2. assess how a staged notebook-treatment wrapper could be built around the current Codex-centered workflow,
notes/023_tools_codex_assessment__20260417_171721.md:8581:The desired future behavior is a wrapper that can launch one narrow Codex task at a time against a notebook set, instead of asking Codex to solve the entire notebook in one pass.
notes/023_tools_codex_assessment__20260417_171721.md:8617:- Do NOT implement the wrapper yet
notes/023_tools_codex_assessment__20260417_171721.md:8640:- what a wrapper would need to do without breaking the current model
notes/023_tools_codex_assessment__20260417_171721.md:8650:1. Can this staged notebook-treatment workflow be built cleanly while staying centered on the current Codex runner?
notes/023_tools_codex_assessment__20260417_171721.md:8651:2. What is the smallest viable wrapper design that would make this work?
notes/023_tools_codex_assessment__20260417_171721.md:8652:3. Should the wrapper:
notes/023_tools_codex_assessment__20260417_171721.md:8657:5. How should the wrapper know:
notes/023_tools_codex_assessment__20260417_171721.md:8675:`notes/XXX_staged_notebook_wrapper_assessment__YYYYMMDD_HHMMSS.md`
notes/023_tools_codex_assessment__20260417_171721.md:8693:- restate the three-stage notebook-treatment concept in implementation terms
notes/023_tools_codex_assessment__20260417_171721.md:8734:I have enough context to draft the assessment. I’m preparing the report content now and will try to write the single `notes/` file directly; if the workspace blocks writes, I’ll stop at that point and report the constraint instead of inventing a file.
notes/023_tools_codex_assessment__20260417_171721.md:8741:I’m writing one note under `notes/` and nothing else. The report is scoped to the current runner contract, V1 records, and the staged notebook-treatment workflow you described.
notes/023_tools_codex_assessment__20260417_171721.md:8744:Unable to create `/home/jevert/git/handson-ml3/notes/023_staged_notebook_wrapper_assessment__20260417_121843.md` because the workspace is read-only.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:2818:### 1. Runner assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:2823:### 2. Artifact assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:2828:### 3. Operational assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:2853:- runner assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:2854:- artifact assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:2855:- operational assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:5023:Your task is to review two files against the current workflow guidance and produce a small design assessment.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:5056:This pass should stay focused on assessment, mismatch detection, and next-step recommendation.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:5145:- the assessment is grounded in `tools/Project_Design_Workflow.md`
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:5157:The assessment’s main conclusion is that the architecture doc is directionally right on decomposition and conservative growth, and the runner is thin enough to count as acceptable bridge tooling, but both are weak at the same seam: they do not yet define a durable, reviewable execution record that separates raw execution from reviewed outcome and captures minimal failure/resource evidence.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:5181:Your task is to review two files against the current workflow guidance and produce a small design assessment.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:5214:This pass should stay focused on assessment, mismatch detection, and next-step recommendation.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:6777:### 1. Runner assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:6782:### 2. Artifact assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:6787:### 3. Operational assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:6812:- runner assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:6813:- artifact assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:6814:- operational assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:6924:### 1. Runner assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:6929:### 2. Artifact assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:6934:### 3. Operational assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:6959:- runner assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:6960:- artifact assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:6961:- operational assessment
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:10976:  Your task is to review two files against the current workflow guidance and produce a small design assessment.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13031:+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13056:+  Reason: captures the first serious assessment of the new runner and the environment-vs-runner distinction.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13077:+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13081:+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13083:+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13122:+  Reason: review scaffold whose durable output is the architecture/runner assessment note.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13196:+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13227:+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13252:+  Reason: captures the first serious assessment of the new runner and the environment-vs-runner distinction.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13273:+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13277:+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13279:+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13318:+  Reason: review scaffold whose durable output is the architecture/runner assessment note.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13392:+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13426:+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13451:+  Reason: captures the first serious assessment of the new runner and the environment-vs-runner distinction.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13472:+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13476:+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13478:+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13517:+  Reason: review scaffold whose durable output is the architecture/runner assessment note.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13591:+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13648:+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13673:+  Reason: captures the first serious assessment of the new runner and the environment-vs-runner distinction.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13694:+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13698:+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13700:+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13739:+  Reason: review scaffold whose durable output is the architecture/runner assessment note.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13813:+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13875:- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13900:  Reason: captures the first serious assessment of the new runner and the environment-vs-runner distinction.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13921:  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13925:  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13927:  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:13966:  Reason: review scaffold whose durable output is the architecture/runner assessment note.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14040:- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14095:+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14120:+  Reason: captures the first serious assessment of the new runner and the environment-vs-runner distinction.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14141:+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14145:+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14147:+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14186:+  Reason: review scaffold whose durable output is the architecture/runner assessment note.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14260:+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14321:+- `move to attic without summary`: mainly a transient receipt, wrapper, prompt-generation step, or superseded construction artifact whose durable content already exists elsewhere
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14346:+  Reason: captures the first serious assessment of the new runner and the environment-vs-runner distinction.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14367:+  Reason: verbose legacy success wrapper; duplicates the earlier smoke-test result and embeds prompt/output noise.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14371:+  Reason: wrapper receipt around the repo-inventory task; the earlier inventory note holds the useful content.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14373:+  Reason: success wrapper for the workflow-doc revision; the companion revision note carries the durable rationale.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14412:+  Reason: review scaffold whose durable output is the architecture/runner assessment note.
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md:14486:+- The safest first cleanup target is old prompt files plus legacy success wrappers and short recommendation receipts.

exec
/bin/bash -lc "sed -n '1,260p' tools/notebook_enricher/notebook_scanner.py" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
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

exec
/bin/bash -lc "sed -n '1,260p' tools/notebook_enricher/prompt_builder.py" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
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

codex
The existing notebook-enricher logic is much more notebook-mutating than the new wrapper should be. I’m pulling the wrapper assessment note and current prompt numbering now so I can draft a smaller, safer sequence aligned to the current V1 note-and-review workflow.
exec
/bin/bash -lc "sed -n '1,240p' notes/023_tools_codex_assessment__20260417_171721.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
# 023_tools_codex_assessment__20260417_171721

- run_id: `023_tools_codex_assessment__20260417_171721`
- prompt_file: `codex_prompts/023_tools_codex_assessment.md`
- prompt_stem: `023_tools_codex_assessment`
- started_at_utc: `20260417_171721`

## Execution Facts

- execution_status: `EXECUTED`
- finished_at_utc: `20260417_171956`
- runner: `tools/codex/run_prompt.py`
- return_code: `0`
- retry_of_run_id:

## Review Facts

- review_status: `UNREVIEWED`
- review_summary:
- reviewed_by:
- reviewed_at_utc:

## Failure Analysis

- failure_type:
- failure_symptom:
- likely_cause:
- recommended_next_action:

## Resource / Cost Facts

- elapsed_seconds: `155.095`
- final_output_char_count: `148`
- stderr_char_count: `1164500`

## Prompt Text

```md
# Prompt: Assess wrapper design for staged notebook-treatment runs using the existing Codex runner

You are working inside this repository.

Your job is NOT to implement the notebook wrapper yet.

Your job is to:
1. inspect the existing Codex runner tooling and related design notes,
2. assess how a staged notebook-treatment wrapper could be built around the current Codex-centered workflow,
3. write a clear implementation-planning report into `notes/`,
4. make no code changes.

## Core intent

We want to keep the workflow centered around Codex calls.

The desired future behavior is a wrapper that can launch one narrow Codex task at a time against a notebook set, instead of asking Codex to solve the entire notebook in one pass.

The staged workflow we are considering is:

### Stage 1
For each notebook:
- inspect whether there is a strong chapter-level markdown introduction near the front
- if missing, write it
- if already present and sufficient, skip
- then move to the next notebook

### Stage 2
For each notebook:
- inspect each non-empty code cell
- determine whether there is a markdown cell immediately before it that explains:
  - the goal of the code cell
  - why this matters for machine learning
- if missing, add it
- if present and sufficient, skip
- continue for all code cells
- continue for all notebooks

### Stage 3
For each notebook:
- inspect each non-empty code cell
- determine whether there is a markdown cell immediately after it that explains:
  - how the implementation works
  - why this is a good practice
  - what other methods could accomplish something similar
- if missing, add it
- if present and sufficient, skip
- continue for all code cells
- continue for all notebooks

## Important constraints

- Do NOT implement the wrapper yet
- Do NOT modify any `.py` files
- Do NOT modify any notebooks
- Do NOT create a new runner
- Do NOT refactor existing tooling
- Do NOT stage or commit anything
- Do NOT change the V1 execution record system
- Only assess what it would take to build this cleanly using the current Codex-centered approach

## Files and areas to inspect

You should inspect the current Codex tooling and any repo notes that help you understand the intended architecture, including at minimum:

- `tools/codex/run_prompt.py`
- the related validation / review / queue scripts in `tools/codex/`
- the V1 workflow markdown files in `tools/codex/` if present
- existing notes or reports that discuss the runner, bridge workflow, notebook treatment, or staged design
- the relevant notebooks only as reference examples if needed

You should especially pay attention to:
- how the current runner executes one prompt
- how records are written into `notes/`
- how queue/review logic works
- what a wrapper would need to do without breaking the current model
- whether this should be a thin orchestration layer that emits prompt files and calls the current runner repeatedly
- how progress and idempotency could be tracked
- what risks exist around notebook mutation
- what minimum viable version should look like

## Specific questions to answer in the report

Your report must answer these questions directly and concretely:

1. Can this staged notebook-treatment workflow be built cleanly while staying centered on the current Codex runner?
2. What is the smallest viable wrapper design that would make this work?
3. Should the wrapper:
   - generate prompt files and call `run_prompt.py` repeatedly,
   - call `codex exec` directly,
   - or use some other thin orchestration approach?
4. What state would need to be tracked between runs?
5. How should the wrapper know:
   - which notebooks are in scope,
   - which stage is currently being worked,
   - which notebook/cell is next,
   - which work is already complete,
   - and when it is safe to stop or resume?
6. What existing parts of the repo can be reused as-is?
7. What new pieces would likely be required?
8. What are the biggest technical risks?
9. What should the MVP do first, before any more advanced features?
10. What should NOT be built yet?

## Deliverable

Write exactly one markdown report into `notes/`.

Use a filename that matches this pattern:

`notes/XXX_staged_notebook_wrapper_assessment__YYYYMMDD_HHMMSS.md`

If you cannot create that exact timestamped filename, create a close equivalent that is clear and unique.

## Report structure

Use this structure:

# Staged Notebook Wrapper Assessment

## Executive Summary
- one short paragraph stating whether the idea is viable

## What Exists Today
- summarize the current runner-centered workflow
- identify which current files are relevant

## Target Workflow
- restate the three-stage notebook-treatment concept in implementation terms

## Recommended Wrapper Design
- describe the thinnest clean design
- explain how it would interact with the current runner

## Reusable Existing Components
- list specific files / functions / patterns already present that should be reused

## New Components Likely Needed
- list the minimum new modules / files / artifacts required

## State and Progress Tracking
- explain how progress could be tracked safely across runs

## Risks and Safeguards
- explain notebook-integrity risks and how to reduce them

## MVP Recommendation
- explain the smallest version worth building first

## What To Avoid
- explain overengineering traps or design mistakes to avoid

## Suggested Build Sequence
- provide an ordered, practical build sequence

## Final Recommendation
- a concise final judgment

## Output rules

- Be concrete
- Be repo-specific
- Name files and scripts explicitly where possible
- Prefer practical design over abstract architecture talk
- Do not write code
- Do not modify repo files except for the single report in `notes/`
- At the end of your final response, print only the path to the note you created
```

## Codex Final Output

Unable to create `/home/jevert/git/handson-ml3/notes/023_staged_notebook_wrapper_assessment__20260417_121843.md` because the workspace is read-only.

## Stderr

```text
OpenAI Codex v0.120.0 (research preview)
--------
workdir: /home/jevert/git/handson-ml3
model: gpt-5.4
provider: openai
approval: never
sandbox: read-only
reasoning effort: none
reasoning summaries: none
session id: 019d9c72-7f50-7d42-9c4e-99baefb4661f
--------
user
# Prompt: Assess wrapper design for staged notebook-treatment runs using the existing Codex runner

You are working inside this repository.

Your job is NOT to implement the notebook wrapper yet.

Your job is to:
1. inspect the existing Codex runner tooling and related design notes,
2. assess how a staged notebook-treatment wrapper could be built around the current Codex-centered workflow,
3. write a clear implementation-planning report into `notes/`,
4. make no code changes.

## Core intent


exec
/bin/bash -lc "sed -n '241,520p' notes/023_tools_codex_assessment__20260417_171721.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
We want to keep the workflow centered around Codex calls.

The desired future behavior is a wrapper that can launch one narrow Codex task at a time against a notebook set, instead of asking Codex to solve the entire notebook in one pass.

The staged workflow we are considering is:

### Stage 1
For each notebook:
- inspect whether there is a strong chapter-level markdown introduction near the front
- if missing, write it
- if already present and sufficient, skip
- then move to the next notebook

### Stage 2
For each notebook:
- inspect each non-empty code cell
- determine whether there is a markdown cell immediately before it that explains:
  - the goal of the code cell
  - why this matters for machine learning
- if missing, add it
- if present and sufficient, skip
- continue for all code cells
- continue for all notebooks

### Stage 3
For each notebook:
- inspect each non-empty code cell
- determine whether there is a markdown cell immediately after it that explains:
  - how the implementation works
  - why this is a good practice
  - what other methods could accomplish something similar
- if missing, add it
- if present and sufficient, skip
- continue for all code cells
- continue for all notebooks

## Important constraints

- Do NOT implement the wrapper yet
- Do NOT modify any `.py` files
- Do NOT modify any notebooks
- Do NOT create a new runner
- Do NOT refactor existing tooling
- Do NOT stage or commit anything
- Do NOT change the V1 execution record system
- Only assess what it would take to build this cleanly using the current Codex-centered approach

## Files and areas to inspect

You should inspect the current Codex tooling and any repo notes that help you understand the intended architecture, including at minimum:

- `tools/codex/run_prompt.py`
- the related validation / review / queue scripts in `tools/codex/`
- the V1 workflow markdown files in `tools/codex/` if present
- existing notes or reports that discuss the runner, bridge workflow, notebook treatment, or staged design
- the relevant notebooks only as reference examples if needed

You should especially pay attention to:
- how the current runner executes one prompt
- how records are written into `notes/`
- how queue/review logic works
- what a wrapper would need to do without breaking the current model
- whether this should be a thin orchestration layer that emits prompt files and calls the current runner repeatedly
- how progress and idempotency could be tracked
- what risks exist around notebook mutation
- what minimum viable version should look like

## Specific questions to answer in the report

Your report must answer these questions directly and concretely:

1. Can this staged notebook-treatment workflow be built cleanly while staying centered on the current Codex runner?
2. What is the smallest viable wrapper design that would make this work?
3. Should the wrapper:
   - generate prompt files and call `run_prompt.py` repeatedly,
   - call `codex exec` directly,
   - or use some other thin orchestration approach?
4. What state would need to be tracked between runs?
5. How should the wrapper know:
   - which notebooks are in scope,
   - which stage is currently being worked,
   - which notebook/cell is next,
   - which work is already complete,
   - and when it is safe to stop or resume?
6. What existing parts of the repo can be reused as-is?
7. What new pieces would likely be required?
8. What are the biggest technical risks?
9. What should the MVP do first, before any more advanced features?
10. What should NOT be built yet?

## Deliverable

Write exactly one markdown report into `notes/`.

Use a filename that matches this pattern:

`notes/XXX_staged_notebook_wrapper_assessment__YYYYMMDD_HHMMSS.md`

If you cannot create that exact timestamped filename, create a close equivalent that is clear and unique.

## Report structure

Use this structure:

# Staged Notebook Wrapper Assessment

## Executive Summary
- one short paragraph stating whether the idea is viable

## What Exists Today
- summarize the current runner-centered workflow
- identify which current files are relevant

## Target Workflow
- restate the three-stage notebook-treatment concept in implementation terms

## Recommended Wrapper Design
- describe the thinnest clean design
- explain how it would interact with the current runner

## Reusable Existing Components
- list specific files / functions / patterns already present that should be reused

## New Components Likely Needed
- list the minimum new modules / files / artifacts required

## State and Progress Tracking
- explain how progress could be tracked safely across runs

## Risks and Safeguards
- explain notebook-integrity risks and how to reduce them

## MVP Recommendation
- explain the smallest version worth building first

## What To Avoid
- explain overengineering traps or design mistakes to avoid

## Suggested Build Sequence
- provide an ordered, practical build sequence

## Final Recommendation
- a concise final judgment

## Output rules

- Be concrete
- Be repo-specific
- Name files and scripts explicitly where possible
- Prefer practical design over abstract architecture talk
- Do not write code
- Do not modify repo files except for the single report in `notes/`
- At the end of your final response, print only the path to the note you created

codex
I’m inspecting the existing `tools/codex/` runner flow and related notes first, then I’ll map that to the staged notebook-wrapper idea and produce a single planning report in `notes/` if the workspace allows it.
exec
/bin/bash -lc 'rg --files tools/codex notes | sort' in /home/jevert/git/handson-ml3
 succeeded in 0ms:
notes/001_notebook_finish_report.md
notes/001_notebook_triage_05_06_07.md
notes/001_smoke_test_pipeline__20260415_233343.md
notes/001_smoke_test_pipeline__20260415_234918.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_133235.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_183223.md
notes/001_smoke_test_pipeline__SUCCESS__20260415_184932.md
notes/002_repo_inventory_and_status__SUCCESS__20260415_133347.md
notes/002_repo_inventory_and_status__SUCCESS__20260415_183259.md
notes/003_project_design_workflow_revision__SUCCESS__20260415_144244.md
notes/003_revise_Project_Deisgn_workflow_document__SUCCESS__20260415_194216.md
notes/004_architecture_and_bridge_runner_review__20260415_195538.md
notes/004_next_design_step_recommendation__20260415_195538.md
notes/004_review_architecture_and_bridge_runner_against_workflow__SUCCESS__20260415_195505.md
notes/005_define_execution_record_and_generate_next_prompt_queue__SUCCESS__20260415_202522.md
notes/005_prompt_queue_plan__20260415_202557.md
notes/006_define_v1_run_review_gate__SUCCESS__20260415_203019.md
notes/007_align_architecture_doc_to_v1_workflow__SUCCESS__20260415_203257.md
notes/008_define_minimal_bridge_runner_change_spec__SUCCESS__20260415_203548.md
notes/009_build_v1_run_prompt_candidate__SUCCESS__20260415_233222.md
notes/009_run_prompt_candidate_build__20260415_233407.md
notes/010_next_step_recommendation__20260415_234559.md
notes/010_review_run_prompt_candidate_and_recommend_next_step__SUCCESS__20260415_234523.md
notes/010_run_prompt_candidate_review__20260415_234559.md
notes/011_build_v1_review_writeback_helper__20260415_235346.md
notes/011_review_writeback_helper_build__20260415_235514.md
notes/012_review_v1_pipeline_and_recommend_next_options__20260416_000658.md
notes/012_top_three_next_options__20260416_000819.md
notes/012_v1_pipeline_options_review__20260416_000819.md
notes/013_generate_prompts_for_queue_readiness_and_review_backlog_helpers__20260416_001937.md
notes/013_next_two_prompt_plan__20260416_002005.md
notes/014_build_queue_readiness_checker__20260416_002319.md
notes/014_queue_readiness_checker_build__20260416_002419.md
notes/015_build_review_backlog_unreviewed_run_lister__20260416_003109.md
notes/015_review_backlog_lister_build__20260416_010500.md
notes/016_next_improvement_recommendation__20260416_003710.md
notes/016_queue_and_backlog_helper_validation__20260416_003710.md
notes/016_validate_queue_and_backlog_helpers_against_current_repo__20260416_003601.md
notes/017_polish_queue_readiness_gap_explanation__20260416_004355.md
notes/017_queue_readiness_gap_explanation_polish__20260416_004458.md
notes/018_architecture_vs_actual_sweep__20260416_005130.md
notes/018_prioritized_remaining_work__20260416_005130.md
notes/018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work__20260416_005034.md
notes/019_generate_prompts_for_doc_alignment_and_record_contract_validation__20260416_010209.md
notes/019_next_two_cleanup_prompts_plan__20260416_010251.md
notes/020_align_v1_doc_and_spec_packet_to_actual_toolset__20260416_010453.md
notes/020_doc_spec_alignment_cleanup__20260416_010534.md
notes/021_add_lightweight_v1_record_contract_validation__20260416_010810.md
notes/021_record_contract_validation__20260416_011314.md
notes/022_audit_and_classify_scaffolding_for_summary_and_attic__20260416_022902.md
notes/022_repo_lay_of_the_land__20260416_105729.md
notes/022_scaffolding_classification_report__20260416_023016.md
notes/022_scaffolding_cleanup_plan__20260416_023016.md
tools/codex/V1_Bridge_Runner_Change_Spec.md
tools/codex/V1_Execution_Record_Artifact.md
tools/codex/V1_Run_Review_Gate.md
tools/codex/baby_run_prompt.py
tools/codex/check_queue_readiness.py
tools/codex/list_review_backlog.py
tools/codex/review_run.py
tools/codex/run_prompt.py
tools/codex/v1_record_validation.py

exec
/bin/bash -lc 'pwd && git status --short' in /home/jevert/git/handson-ml3
 succeeded in 0ms:
/home/jevert/git/handson-ml3
?? codex_prompts/023_tools_codex_assessment.md

codex
I’ve identified the current runner scripts and the main V1 design docs. Next I’m reading the runner, review, validation, and prior notes to pin down what the wrapper can reuse without changing the current execution model.
exec
/bin/bash -lc "sed -n '1,260p' tools/codex/review_run.py" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

from v1_record_validation import REVIEW_STATUS_VALUES, validate_record_text


REVIEW_STATUSES = REVIEW_STATUS_VALUES - {"UNREVIEWED"}
FAILURE_FIELDS = (
    "failure_type",
    "failure_symptom",
    "likely_cause",
    "recommended_next_action",
)


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write manual V1 review fields back into an existing execution record."
    )
    parser.add_argument("record", help="Path to an existing execution-record markdown file")
    parser.add_argument(
        "--review-status",
        required=True,
        choices=sorted(REVIEW_STATUSES),
        help="Manual review outcome",
    )
    parser.add_argument(
        "--review-summary",
        required=True,
        help="Short manual review summary",
    )
    parser.add_argument("--reviewed-by", help="Reviewer identifier")

exec
/bin/bash -lc 'ls codex_prompts | sort' in /home/jevert/git/handson-ml3
 succeeded in 0ms:
001_smoke_test_pipeline.md
002_repo_inventory_and_status.md
003_revise_Project_Deisgn_workflow_document.md
004_review_architecture_and_bridge_runner_against_workflow.md
005_define_execution_record_and_generate_next_prompt_queue.md
006_define_v1_run_review_gate.md
007_align_architecture_doc_to_v1_workflow.md
008_define_minimal_bridge_runner_change_spec.md
009_build_v1_run_prompt_candidate.md
010_review_run_prompt_candidate_and_recommend_next_step.md
011_build_v1_review_writeback_helper.md
012_review_v1_pipeline_and_recommend_next_options.md
013_generate_prompts_for_queue_readiness_and_review_backlog_helpers.md
014_build_queue_readiness_checker.md
015_build_review_backlog_unreviewed_run_lister.md
016_validate_queue_and_backlog_helpers_against_current_repo.md
017_polish_queue_readiness_gap_explanation.md
018_sweep_architecture_against_actual_v1_and_prioritize_remaining_work.md
019_generate_prompts_for_doc_alignment_and_record_contract_validation.md
020_align_v1_doc_and_spec_packet_to_actual_toolset.md
021_add_lightweight_v1_record_contract_validation.md
022_audit_and_classify_scaffolding_for_summary_and_attic.md
022_generate_repo_lay_of_the_land_summary.md
023_tools_codex_assessment.md
024_generate_staged_notebook_wrapper_prompts.md

exec
/bin/bash -lc "sed -n '1,220p' codex_prompts/023_tools_codex_assessment.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
# Prompt: Assess wrapper design for staged notebook-treatment runs using the existing Codex runner

You are working inside this repository.

Your job is NOT to implement the notebook wrapper yet.

Your job is to:
1. inspect the existing Codex runner tooling and related design notes,
2. assess how a staged notebook-treatment wrapper could be built around the current Codex-centered workflow,
3. write a clear implementation-planning report into `notes/`,
4. make no code changes.

## Core intent

We want to keep the workflow centered around Codex calls.

The desired future behavior is a wrapper that can launch one narrow Codex task at a time against a notebook set, instead of asking Codex to solve the entire notebook in one pass.

The staged workflow we are considering is:

### Stage 1
For each notebook:
- inspect whether there is a strong chapter-level markdown introduction near the front
- if missing, write it
- if already present and sufficient, skip
- then move to the next notebook

### Stage 2
For each notebook:
- inspect each non-empty code cell
- determine whether there is a markdown cell immediately before it that explains:
  - the goal of the code cell
  - why this matters for machine learning
- if missing, add it
- if present and sufficient, skip
- continue for all code cells
- continue for all notebooks

### Stage 3
For each notebook:
- inspect each non-empty code cell
- determine whether there is a markdown cell immediately after it that explains:
  - how the implementation works
  - why this is a good practice
  - what other methods could accomplish something similar
- if missing, add it
- if present and sufficient, skip
- continue for all code cells
- continue for all notebooks

## Important constraints

- Do NOT implement the wrapper yet
- Do NOT modify any `.py` files
- Do NOT modify any notebooks
- Do NOT create a new runner
- Do NOT refactor existing tooling
- Do NOT stage or commit anything
- Do NOT change the V1 execution record system
- Only assess what it would take to build this cleanly using the current Codex-centered approach

## Files and areas to inspect

You should inspect the current Codex tooling and any repo notes that help you understand the intended architecture, including at minimum:

- `tools/codex/run_prompt.py`
- the related validation / review / queue scripts in `tools/codex/`
- the V1 workflow markdown files in `tools/codex/` if present
- existing notes or reports that discuss the runner, bridge workflow, notebook treatment, or staged design
- the relevant notebooks only as reference examples if needed

You should especially pay attention to:
- how the current runner executes one prompt
- how records are written into `notes/`
- how queue/review logic works
- what a wrapper would need to do without breaking the current model
- whether this should be a thin orchestration layer that emits prompt files and calls the current runner repeatedly
- how progress and idempotency could be tracked
- what risks exist around notebook mutation
- what minimum viable version should look like

## Specific questions to answer in the report

Your report must answer these questions directly and concretely:

1. Can this staged notebook-treatment workflow be built cleanly while staying centered on the current Codex runner?
2. What is the smallest viable wrapper design that would make this work?
3. Should the wrapper:
   - generate prompt files and call `run_prompt.py` repeatedly,
   - call `codex exec` directly,
   - or use some other thin orchestration approach?
4. What state would need to be tracked between runs?
5. How should the wrapper know:
   - which notebooks are in scope,
   - which stage is currently being worked,
   - which notebook/cell is next,
   - which work is already complete,
   - and when it is safe to stop or resume?
6. What existing parts of the repo can be reused as-is?
7. What new pieces would likely be required?
8. What are the biggest technical risks?
9. What should the MVP do first, before any more advanced features?
10. What should NOT be built yet?

## Deliverable

Write exactly one markdown report into `notes/`.

Use a filename that matches this pattern:

`notes/XXX_staged_notebook_wrapper_assessment__YYYYMMDD_HHMMSS.md`

If you cannot create that exact timestamped filename, create a close equivalent that is clear and unique.

## Report structure

Use this structure:

# Staged Notebook Wrapper Assessment

## Executive Summary
- one short paragraph stating whether the idea is viable

## What Exists Today
- summarize the current runner-centered workflow
- identify which current files are relevant

## Target Workflow
- restate the three-stage notebook-treatment concept in implementation terms

## Recommended Wrapper Design
- describe the thinnest clean design
- explain how it would interact with the current runner

## Reusable Existing Components
- list specific files / functions / patterns already present that should be reused

## New Components Likely Needed
- list the minimum new modules / files / artifacts required

## State and Progress Tracking
- explain how progress could be tracked safely across runs

## Risks and Safeguards
- explain notebook-integrity risks and how to reduce them

## MVP Recommendation
- explain the smallest version worth building first

## What To Avoid
- explain overengineering traps or design mistakes to avoid

## Suggested Build Sequence
- provide an ordered, practical build sequence

## Final Recommendation
- a concise final judgment

## Output rules

- Be concrete
- Be repo-specific
- Name files and scripts explicitly where possible
- Prefer practical design over abstract architecture talk
- Do not write code
- Do not modify repo files except for the single report in `notes/`
- At the end of your final response, print only the path to the note you created

codex
I have enough of the runner contract now. I’m checking a few existing prompt files for tone and granularity, then I’ll assemble the draft prompt set and planning note content in repo style, but I may be blocked from actually writing because this workspace is mounted read-only.
exec
/bin/bash -lc "sed -n '1,220p' codex_prompts/019_generate_prompts_for_doc_alignment_and_record_contract_validation.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
# Task: Generate the next two implementation prompts for the current V1 workflow cleanup

You are working in this repository.

Your task is to write the next two bounded implementation prompt files based on the current architecture sweep and prioritized remaining work.

## Important framing

This is a prompt-generation task only.

Do NOT implement code or doc edits in this pass.
Do NOT modify existing tools or design documents in this pass.

Your job is to create the next two implementation prompts as files in `codex_prompts/`.

## Files to inspect

Read these exact files:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
- `notes/018_prioritized_remaining_work__20260416_005130.md`

## Goal

Generate exactly two prompt files:

1. a prompt to perform the recommended **doc/spec alignment cleanup**
2. a prompt to implement the recommended **lightweight record-contract validation**

These prompts should be implementation-ready, bounded, and consistent with the current repo state.

## Prompt 1 requirements

Create a prompt file for doc/spec alignment cleanup.

This future task should focus on aligning the design packet with the actual V1 toolset and current behavior, including at least:

- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- any stale references to `baby_run_prompt.py` where the actual V1 runner is now `run_prompt.py`
- any stale implication that readiness/backlog helpers do not yet exist
- the current run-id collision suffix behavior where relevant

The prompt should keep the cleanup bounded and should not turn into a broad architecture rewrite.

## Prompt 2 requirements

Create a prompt file for lightweight record-contract validation.

This future task should focus on creating a small repeatable validation layer for the shared V1 markdown execution-record shape used by:

- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

The prompt should keep the validation lightweight and inspectable.

Avoid proposing a large test framework, platform service, or dashboard.

## What each generated prompt must include

Each prompt file must:

- have a clear numeric filename prefix
- define one primary goal only
- name exact files to inspect
- define exact output artifacts to create
- include constraints that prevent broad platform growth
- define validation requirements
- define success criteria
- stay small enough to review comfortably before execution

## Also create one planning note

Create a short note at:

`notes/019_next_two_cleanup_prompts_plan__TIMESTAMP.md`

This note should summarize:

- why these two prompts were chosen
- why they are ordered this way
- which one should be executed first
- what should explicitly wait until later

## Constraints

1. Generate exactly two prompt files, no more.
2. Do not implement the cleanup tasks in this pass.
3. Keep both generated prompts bounded and V1-sized.
4. Do not introduce a broader orchestration layer, queue engine, retry manager, or platform expansion.
5. Let the architecture sweep drive the prompts.

## Success criteria

This task is successful if:

- exactly two prompt files are created
- one is for doc/spec alignment cleanup
- one is for lightweight record-contract validation
- both prompts are implementation-ready and bounded
- the planning note clearly explains the sequence

exec
/bin/bash -lc "sed -n '1,220p' codex_prompts/020_align_v1_doc_and_spec_packet_to_actual_toolset.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
# Task: Align the V1 doc/spec packet to the actual implemented toolset

You are working in this repository.

Your task is to perform a bounded doc/spec alignment cleanup so the current V1 design packet matches the implemented workflow.

This is a cleanup pass, not a new architecture pass.

## Primary goal

Make the current V1 design packet accurately describe the actual implemented toolset and current behavior without expanding scope or redesigning the workflow.

## Files to inspect

Read these exact files before editing:

- `tools/Project_Design_Workflow.md`
- `tools/Codex_Prompt_Workflow_Architecture.md`
- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/V1_Bridge_Runner_Change_Spec.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
- `notes/018_prioritized_remaining_work__20260416_005130.md`

## Required cleanup scope

Keep the work bounded to doc/spec alignment for the current V1 slice.

The cleanup must address at least these items:

1. `tools/codex/V1_Bridge_Runner_Change_Spec.md`
   - remove stale references to `tools/codex/baby_run_prompt.py`
   - align the document to the real runner at `tools/codex/run_prompt.py`
   - stop describing review write-back and queue support as unimplemented when helper scripts now exist

2. Any stale references in the inspected design packet that still imply:
   - the runner is `baby_run_prompt.py`
   - readiness and backlog helpers do not yet exist
   - the V1 run id is always unsuffixed

3. The current run-id collision behavior where relevant
   - document that the base V1 identity is `<prompt_stem>__<started_at_utc>`
   - document that `run_prompt.py` adds a numeric suffix such as `__2` only when needed to avoid same-second collisions
   - keep that explanation small and implementation-accurate

## Output artifacts to create

Create exactly one note:

- `notes/020_doc_spec_alignment_cleanup__TIMESTAMP.md`

Update only the minimum necessary design/spec files from the inspected list.

## Constraints

- Do not modify `tools/codex/run_prompt.py`
- Do not modify `tools/codex/review_run.py`
- Do not modify `tools/codex/check_queue_readiness.py`
- Do not modify `tools/codex/list_review_backlog.py`
- Do not rewrite the overall architecture
- Do not introduce new workflow states, new tools, or new platform layers
- Do not expand into retry tooling, scheduling, queue engines, dashboards, or orchestration
- Keep this as a V1-sized doc/spec cleanup only

## Validation requirements

Validate the cleanup by doing all of the following:

1. Confirm the updated design/spec files consistently name `tools/codex/run_prompt.py` as the active V1 runner where applicable.
2. Confirm the updated packet no longer implies that review write-back, readiness checking, or backlog listing are still future work if those helpers are already present.
3. Confirm the run-id wording is consistent with current code behavior, including same-second collision suffixes where relevant.
4. Use a search pass to verify there are no remaining stale `baby_run_prompt.py` references inside the edited design/spec files unless a historical reference is explicitly intentional and clearly marked as historical.

## Success criteria

This task is successful if:

1. The V1 design/spec packet matches the current implemented toolset closely enough that a new prompt author would not be misled.
2. `tools/codex/V1_Bridge_Runner_Change_Spec.md` no longer reads like a pre-implementation bridge spec for a different runner.
3. The packet reflects that the V1 workflow now includes:
   - `run_prompt.py`
   - `review_run.py`
   - `check_queue_readiness.py`
   - `list_review_backlog.py`
4. The run-id collision suffix behavior is described accurately but briefly.
5. The cleanup stays comfortably reviewable and does not turn into a broader rewrite.

exec
/bin/bash -lc "sed -n '1,220p' codex_prompts/021_add_lightweight_v1_record_contract_validation.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
# Task: Add lightweight validation for the shared V1 execution-record contract

You are working in this repository.

Your task is to add a small, repeatable validation layer for the shared markdown execution-record shape used across the current V1 tools.

Keep this implementation lightweight and inspectable.

## Primary goal

Create one small validation path that protects the shared V1 markdown record contract without introducing a larger framework or platform surface.

## Files to inspect

Read these exact files before editing:

- `tools/codex/V1_Execution_Record_Artifact.md`
- `tools/codex/V1_Run_Review_Gate.md`
- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`
- `notes/018_architecture_vs_actual_sweep__20260416_005130.md`
- `notes/018_prioritized_remaining_work__20260416_005130.md`

## Required implementation scope

Create a lightweight validation layer for the shared V1 markdown execution-record shape used by:

- `tools/codex/run_prompt.py`
- `tools/codex/review_run.py`
- `tools/codex/check_queue_readiness.py`
- `tools/codex/list_review_backlog.py`

The implementation should cover only the minimum contract that these tools already depend on, such as:

- required section presence and order
- required field-line presence
- allowed execution-status values
- allowed review-status values
- the expected `run_id` pattern including the optional same-second numeric suffix
- basic consistency checks like title/run-id agreement and prompt-file/prompt-stem agreement where appropriate

## Output artifacts to create

Create exactly these artifacts:

1. one small shared validation module under `tools/codex/`
2. only the minimal script changes needed so the current V1 tools reuse that validation instead of carrying separate ad hoc contract checks
3. one short implementation note:
   - `notes/021_record_contract_validation__TIMESTAMP.md`

## Constraints

- Do not build a larger test framework
- Do not introduce a service, daemon, dashboard, or platform layer
- Do not add a database, JSON sidecar, or alternate record format
- Do not redesign the V1 markdown artifact
- Do not expand queue semantics, retry tooling, or orchestration behavior
- Do not add broad dependency or configuration systems
- Keep the validation readable enough that a reviewer can inspect it quickly in one sitting

## Validation requirements

Validate the work by doing all of the following:

1. Run a lightweight syntax check on the touched Python files.
2. Run at least one small direct validation path against existing repo records in `notes/` so the shared validator proves it can parse or reject records using the current contract.
3. Confirm `run_prompt.py` still writes records that the shared validator accepts.
4. Confirm `review_run.py`, `check_queue_readiness.py`, and `list_review_backlog.py` still work with the shared validator in place.
5. Keep validation evidence in the implementation note concise and concrete.

## Success criteria

This task is successful if:

1. The shared V1 markdown record contract is enforced through one lightweight reusable validation path.
2. The four V1 scripts no longer each carry their own independent, partially duplicated contract assumptions where simple sharing would suffice.
3. The validator remains small, inspectable, and local to the current V1 workflow.
4. No broader platform growth is introduced.
5. The result is still easy to review before execution.

exec
/bin/bash -lc "sed -n '1,220p' tools/codex/V1_Execution_Record_Artifact.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
# V1 Execution Record Artifact

## Purpose

Define the minimum durable record for one prompt run so the workflow preserves:

- what was executed
- what happened during execution
- what a human later decided about the result
- enough failure and resource evidence to guide the next reviewed step

This artifact is meant to support the current conservative workflow, not a larger automation system.

## Scope

V1 covers one prompt run from prompt submission through later human review.

It does not define:

- multi-run reporting
- dependency graphs
- approval automation
- module decomposition
- a full state machine

## Source Of Truth

For V1, the source of truth for one run is one markdown execution-record file in `notes/`.

The record body is authoritative.
The filename is a locator and stable identifier carrier, but the body is the place where execution facts and review fields must be read from.

V1 should avoid splitting one run across multiple required files.

## Stable Identity For A Run

Each run must have a stable `run_id` that does not change after the file is created.

V1 format:

`<prompt_stem>__<started_at_utc>`

Example:

`005_define_execution_record_and_generate_next_prompt_queue__20260415_202557`

Why this is enough for V1:

- it is human-readable
- it distinguishes repeated runs of the same prompt
- it does not depend on later review outcome
- it can be created cheaply by the bridge runner

`tools/codex/run_prompt.py` uses that base identity and adds a numeric suffix such as `__2` only when needed to avoid a same-second filename collision for the same prompt.

The filename should use the same stable identity and should not encode accepted or rejected review outcome.

## Required Fields

Every V1 execution record should contain these fields.

### Run Identity

- `run_id`
- `prompt_file`
- `prompt_stem`
- `started_at_utc`

### Execution Facts

- `execution_status`
- `finished_at_utc`
- `runner`
- `return_code`
- `prompt_text`
- `codex_final_output`

### Review Facts

- `review_status`
- `review_summary`

### Minimum Failure / Retry Linkage

- `retry_of_run_id`

### Minimum Resource / Cost Facts

- `elapsed_seconds`
- `final_output_char_count`
- `stderr_char_count`

## Optional Fields

These fields are useful when available, but should not block V1.

- `reviewed_by`
- `reviewed_at_utc`
- `stderr_text`
- `failure_analysis`
- `recommended_next_action`
- `model_name`
- `token_usage`
- `attempt_index_for_prompt`
- `workspace_notes`

## Automatic Vs Manual Fields

### Automatic In V1

These should be captured automatically when the runner can do so cheaply and reliably:

- `run_id`
- `prompt_file`
- `prompt_stem`
- `started_at_utc`
- `finished_at_utc`
- `execution_status`
- `runner`
- `return_code`
- `prompt_text`
- `codex_final_output`
- `elapsed_seconds`
- `final_output_char_count`
- `stderr_char_count`
- `stderr_text` when present
- `retry_of_run_id` only if the runner was explicitly given that context

### Manual In V1

These remain human-owned in V1:

- `review_status`
- `review_summary`
- `reviewed_by`
- `reviewed_at_utc`
- `failure_analysis`
- `recommended_next_action`

This split is intentional.
Execution capture should be automatic.
Acceptance and interpretation should remain manual until the review loop is defined more clearly.

## Execution Status Vs Review Status

V1 must keep these separate.

### Execution Status

This records what happened when the runner invoked Codex.

Allowed V1 values:

- `EXECUTED`
- `EXECUTION_FAILED`

`EXECUTED` means the subprocess completed successfully enough to return code `0`.
It does not mean the work is accepted.

`EXECUTION_FAILED` means the subprocess failed or the run artifact could only be completed with failure evidence.

### Review Status

This records the later human judgment.

Allowed V1 values:

- `UNREVIEWED`
- `ACCEPTED`
- `REJECTED`

Rules:

- a new run record should start as `UNREVIEWED`
- `EXECUTED` must not imply `ACCEPTED`
- `EXECUTION_FAILED` does not automatically imply `REJECTED`, though rejection will often follow
- the next prompt in a reviewed sequence should wait for explicit human review, not only process exit success

## Minimum Failure-Analysis Section

V1 failure analysis should stay small.
It should exist as a short section that can be filled in manually when useful, especially for `EXECUTION_FAILED` or `REJECTED` runs.

Minimum fields:

- `failure_type`
- `failure_symptom`
- `likely_cause`
- `recommended_next_action`

Suggested `failure_type` values:

- `environment`
- `runner`
- `prompt_scope`
- `prompt_clarity`
- `implementation`
- `validation`
- `unknown`

This is enough for V1 because it supports learning from failure without requiring a larger taxonomy.

## Minimum Resource / Cost Section

V1 should record only lightweight fields that are cheap to capture or inspect:

- `started_at_utc`
- `finished_at_utc`
- `elapsed_seconds`
- `return_code`
- `final_output_char_count`
- `stderr_char_count`

Optional later additions:

- `model_name`
- `token_usage`
- `tool_call_count`
- `file_change_count`
- `estimated_review_minutes`

exec
/bin/bash -lc "sed -n '1,220p' tools/codex/V1_Run_Review_Gate.md" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
# V1 Run Review Gate

## Purpose

Define the smallest practical review gate between one prompt run finishing and the next prompt in sequence being treated as ready.

This gate exists to preserve three things from the governing workflow:

- review between iterations
- separation of execution outcome from accepted outcome
- a conservative human stop before queue progression

## Scope

V1 covers only the decision point immediately after one run record is written.

It defines:

- when a run enters review
- the minimum manual checks
- the allowed review outcomes
- how those outcomes affect whether the next prompt may proceed
- what review information must be written into the run record

It does not define:

- automation of review
- a larger queue state machine
- dependency-aware scheduling
- retry orchestration
- richer review taxonomies such as `PARTIAL` or `BLOCKED`

## Review Trigger

A run becomes `awaiting review` when its execution phase has finished and its V1 execution record has been written to `notes/` with:

- `execution_status` set to either `EXECUTED` or `EXECUTION_FAILED`
- `review_status` still set to `UNREVIEWED`

That is the exact review trigger for V1.

Process exit alone does not make the next prompt ready.
Record creation plus `UNREVIEWED` review status means the run must stop for human inspection.

## Minimum Manual Checklist

Before a run can be accepted, a human reviewer must check only these items:

1. Confirm the execution record is complete enough to review.
   Required minimum: prompt identity, execution status, return code, prompt text, Codex final output, and any captured stderr.

2. Check whether the run actually addressed the prompt that was executed.
   This is a scope-and-intent check, not a deep redesign review.

3. Check whether there is enough evidence to treat the result as an accepted outcome for the current bounded step.
   Minimum evidence may be direct output inspection, referenced file changes, or stated validation evidence in the run record or surrounding repo state.

4. Check whether any obvious failure, mismatch, or ambiguity remains that would make automatic progression unsafe.
   Examples: execution failed, output is incomplete, acceptance is unclear, validation is missing, or the result suggests the design/task should be refined before continuing.

This checklist is intentionally small.
V1 only needs enough manual review to keep execution success separate from accepted progress.

## Allowed V1 Review Outcomes

V1 should allow exactly these review outcomes:

- `UNREVIEWED`
- `ACCEPTED`
- `REJECTED`

Meaning:

- `UNREVIEWED`: default state after execution record creation; queue must stop here.
- `ACCEPTED`: the bounded step is accepted after human review.
- `REJECTED`: the run is not accepted for progression, whether because execution failed, the result is inadequate, or the outcome requires redesign or retry analysis first.

No additional V1 outcomes are needed.

## Queue Progression Rule

Only `ACCEPTED` allows the next prompt in sequence to be treated as ready.

Rules:

- `UNREVIEWED` stops the queue pending manual review.
- `ACCEPTED` releases exactly the next reviewed step.
- `REJECTED` stops the queue and forces an explicit human decision about what happens next.

`EXECUTED` does not release the queue.
`EXECUTION_FAILED` does not itself decide the queue either, but it will usually lead to `REJECTED` on review.

## Stop-And-Decide Rule

`REJECTED` is the V1 outcome that stops the queue and forces a new design or retry decision.

That decision is intentionally outside this gate.
It may result in:

- a revised prompt
- a smaller follow-up slice
- an environment fix
- a deliberate retry
- an architecture or scope adjustment

V1 only requires that the queue does not continue past a rejected run.

## Run Record Write-Back

During review, the run record defined by [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md) must be updated with these review facts:

- `review_status`
- `review_summary`

When available, V1 should also write:

- `reviewed_by`
- `reviewed_at_utc`

For rejected runs, the reviewer should also fill the small failure-analysis fields when useful:

- `failure_type`
- `failure_symptom`
- `likely_cause`
- `recommended_next_action`

This keeps execution facts and review judgment in one durable record.

## Connection To The Execution Record

This gate depends on the separation already defined in [V1_Execution_Record_Artifact.md](/data/git/handson-ml3/tools/codex/V1_Execution_Record_Artifact.md:77):

- execution status answers "what happened when the runner executed the prompt?"
- review status answers "did a human accept this bounded step as good enough to progress?"

The V1 review gate is therefore not a new artifact.
It is the manual transition of one execution record from `UNREVIEWED` to either `ACCEPTED` or `REJECTED`.

## Intentionally Deferred

V1 intentionally defers:

- automatic queue release
- automatic reviewer assignment
- separate queue-level state files
- richer review outcomes such as `PARTIAL`, `BLOCKED`, or `NEEDS_RETRY`
- policy for choosing among multiple retry or redesign options
- automatic extraction of validation evidence from repo changes
- broader workflow redesign beyond the next reviewed step

## V1 Decision Summary

The minimum V1 review gate is:

1. execution finishes
2. a run record is written with `review_status: UNREVIEWED`
3. human review applies the minimum checklist
4. reviewer writes back `ACCEPTED` or `REJECTED`
5. only `ACCEPTED` makes the next prompt ready

That is small enough to guide the next implementation prompt without expanding the system into a larger workflow engine.

exec
/bin/bash -lc "sed -n '260,520p' tools/notebook_enricher/prompt_builder.py" in /home/jevert/git/handson-ml3
 succeeded in 0ms:
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

codex
The repository already records the key constraint: this same wrapper-assessment flow previously failed only because the workspace is read-only, not because the prompt was unclear. I’m drafting the actual prompt set and planning note content now so I can hand you the exact files to add once writes are possible.
codex
I could not write the files because this workspace is mounted read-only. The draft prompt set I prepared is:

- `codex_prompts/025_define_staged_notebook_wrapper_mvp_contract.md`
  Defines the MVP boundary, wrapper state model, scope/config inputs, reporting artifacts, and explicit non-goals before any code build.
- `codex_prompts/026_build_wrapper_scope_state_and_resume_contract.md`
  Builds the minimal structured scope/progress contract for notebook selection, per-stage position, stop/resume behavior, and durable progress notes.
- `codex_prompts/027_build_wrapper_scan_and_treatment_report_layer.md`
  Adds the narrow scan/report layer that reuses `tools/notebook_enricher/notebook_scanner.py` and emits deterministic treatment decisions without mutating notebooks.
- `codex_prompts/028_build_stage1_intro_prompt_generation_rules.md`
  Implements only Stage 1 prompt-generation rules for chapter intro detection/insertion, with safe write constraints and narrow output contracts.
- `codex_prompts/029_build_stage2_goal_before_and_stage3_impl_after_prompt_generation_rules.md`
  Implements the Stage 2 and Stage 3 prompt-generation rules, focused on bounded cell lists, replace-vs-insert decisions, and notebook-write safeguards.
- `codex_prompts/030_build_minimal_notebook_wrapper_orchestration_loop.md`
  Implements the thinnest orchestration loop around `tools/codex/run_prompt.py`, using generated prompt files plus wrapper progress artifacts, with no runner redesign.

The planning note I drafted would summarize that sequence as: contract first, then resumable state, then scan output, then per-stage prompt safety, then minimal orchestration last.

notes/025_staged_notebook_wrapper_prompt_plan__20260417_181500.md
tokens used
68,287
```
