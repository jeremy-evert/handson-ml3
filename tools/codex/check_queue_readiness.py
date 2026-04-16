#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


PROMPTS_DIR = "codex_prompts"
NOTES_DIR = "notes"
REQUIRED_RECORD_FIELDS = (
    "run_id",
    "prompt_file",
    "prompt_stem",
    "started_at_utc",
    "execution_status",
    "review_status",
)
IDENTITY_FIELDS = (
    "run_id",
    "prompt_file",
    "prompt_stem",
    "started_at_utc",
)
EXECUTION_STATUSES = {"EXECUTED", "EXECUTION_FAILED"}
REVIEW_STATUSES = {"UNREVIEWED", "ACCEPTED", "REJECTED"}
PROMPT_NAME_RE = re.compile(r"^(?P<prefix>\d+)_")
TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


@dataclass(frozen=True)
class PromptEntry:
    prefix: int
    path: Path

    @property
    def label(self) -> str:
        return self.path.as_posix()


@dataclass(frozen=True)
class RunRecord:
    path: Path
    run_id: str
    prompt_file: str
    prompt_stem: str
    started_at_utc: str
    execution_status: str
    review_status: str
    run_suffix: int


@dataclass(frozen=True)
class ReadinessResult:
    target: PromptEntry
    previous: PromptEntry | None
    latest_record: RunRecord | None
    ready: bool
    reason: str


class ReadinessError(Exception):
    pass


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


def parse_field_line(text: str, field: str) -> str | None:
    match = re.search(rf"(?m)^- {re.escape(field)}:\s*(.*)$", text)
    if match is None:
        return None

    value = match.group(1).strip()
    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return value


def parse_run_suffix(prompt_stem: str, started_at_utc: str, run_id: str) -> int:
    base_run_id = f"{prompt_stem}__{started_at_utc}"
    if run_id == base_run_id:
        return 1

    suffix_match = re.fullmatch(rf"{re.escape(base_run_id)}__(\d+)", run_id)
    if suffix_match is None:
        raise ReadinessError(f"run_id does not match V1 identity pattern: {run_id}")
    return int(suffix_match.group(1))


def parse_record_file(root: Path, path: Path) -> RunRecord | None:
    text = path.read_text(encoding="utf-8")
    has_execution_section = "## Execution Facts" in text
    has_review_section = "## Review Facts" in text
    if not has_execution_section and not has_review_section:
        return None
    if has_execution_section != has_review_section:
        raise ReadinessError(
            f"record-like note has incomplete V1 sections: {path.relative_to(root).as_posix()}"
        )

    header_block = text.split("\n## ", 1)[0]
    identity_values = {field: parse_field_line(header_block, field) for field in IDENTITY_FIELDS}
    if identity_values["run_id"] is None:
        return None

    metadata_block = text.split("\n## Prompt Text\n", 1)[0]
    values = {field: parse_field_line(metadata_block, field) for field in REQUIRED_RECORD_FIELDS}

    missing_fields = [field for field, value in values.items() if value is None]
    if missing_fields:
        raise ReadinessError(
            f"record-like note is missing required V1 fields: {path.relative_to(root).as_posix()} "
            f"missing {', '.join(missing_fields)}"
        )

    run_id = values["run_id"] or ""
    prompt_file = values["prompt_file"] or ""
    prompt_stem = values["prompt_stem"] or ""
    started_at_utc = values["started_at_utc"] or ""
    execution_status = values["execution_status"] or ""
    review_status = values["review_status"] or ""

    if not TIMESTAMP_RE.fullmatch(started_at_utc):
        raise ReadinessError(
            f"record has invalid started_at_utc timestamp: {path.relative_to(root).as_posix()}"
        )
    if execution_status not in EXECUTION_STATUSES:
        raise ReadinessError(
            f"record has invalid execution_status: {path.relative_to(root).as_posix()}"
        )
    if review_status not in REVIEW_STATUSES:
        raise ReadinessError(f"record has invalid review_status: {path.relative_to(root).as_posix()}")

    prompt_path = Path(prompt_file)
    if prompt_path.stem != prompt_stem:
        raise ReadinessError(
            f"record prompt_file/prompt_stem mismatch: {path.relative_to(root).as_posix()}"
        )

    return RunRecord(
        path=path.relative_to(root),
        run_id=run_id,
        prompt_file=prompt_file,
        prompt_stem=prompt_stem,
        started_at_utc=started_at_utc,
        execution_status=execution_status,
        review_status=review_status,
        run_suffix=parse_run_suffix(prompt_stem, started_at_utc, run_id),
    )


def discover_run_records(root: Path) -> list[RunRecord]:
    notes_dir = root / NOTES_DIR
    if not notes_dir.exists():
        raise ReadinessError(f"missing notes directory: {notes_dir}")

    records: list[RunRecord] = []
    for path in sorted(notes_dir.glob("*.md")):
        record = parse_record_file(root, path)
        if record is not None:
            records.append(record)
    return records


def latest_record_for_prompt(records: list[RunRecord], prompt: PromptEntry) -> RunRecord | None:
    relevant = [record for record in records if record.prompt_file == prompt.label]
    if not relevant:
        return None
    return max(relevant, key=lambda record: (record.started_at_utc, record.run_suffix))


def default_target(prompts: list[PromptEntry], records: list[RunRecord]) -> PromptEntry:
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
    records: list[RunRecord],
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
        reason = "latest V1 run for the immediately previous prompt is REJECTED"

    return ReadinessResult(
        target=target,
        previous=previous,
        latest_record=latest,
        ready=False,
        reason=reason,
    )


def print_summary(prompts: list[PromptEntry], result: ReadinessResult) -> None:
    print("Ordered prompts:")
    for prompt in prompts:
        print(f"- {prompt.prefix:03d}: {prompt.label}")

    latest_record = result.latest_record
    latest_record_path = latest_record.path.as_posix() if latest_record else "none"
    latest_execution_status = latest_record.execution_status if latest_record else "n/a"
    latest_review_status = latest_record.review_status if latest_record else "n/a"

    print()
    print(f"Target prompt: {result.target.label}")
    print(f"Previous prompt: {result.previous.label if result.previous else 'none'}")
    print(f"Latest run record: {latest_record_path}")
    print(f"Latest run execution_status: {latest_execution_status}")
    print(f"Latest run review_status: {latest_review_status}")
    print(f"Ready: {'YES' if result.ready else 'NO'}")
    print(f"Reason: {result.reason}")


def main() -> int:
    try:
        args = parse_args()
        root = repo_root()
        prompts = discover_prompts(root)
        records = discover_run_records(root)
        target = resolve_prompt(prompts, args.prompt) if args.prompt else default_target(prompts, records)
        result = evaluate_readiness(prompts, records, target)
        print_summary(prompts, result)
        return 0
    except ReadinessError as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
