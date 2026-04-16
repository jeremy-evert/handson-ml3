#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path


REVIEW_STATUSES = {"ACCEPTED", "REJECTED"}
FAILURE_FIELDS = (
    "failure_type",
    "failure_symptom",
    "likely_cause",
    "recommended_next_action",
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
    *FAILURE_FIELDS,
    "elapsed_seconds",
    "final_output_char_count",
    "stderr_char_count",
)
REQUIRED_SECTIONS = (
    "## Execution Facts",
    "## Review Facts",
    "## Failure Analysis",
    "## Resource / Cost Facts",
    "## Prompt Text",
    "## Codex Final Output",
    "## Stderr",
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


def validate_v1_record_structure(text: str) -> None:
    if not text.startswith("# "):
        raise ValueError("record does not start with a markdown title")

    positions: list[int] = []
    for section in REQUIRED_SECTIONS:
        pos = text.find(section)
        if pos == -1:
            raise ValueError(f"record is missing section: {section}")
        positions.append(pos)
    if positions != sorted(positions):
        raise ValueError("record sections are out of the expected V1 order")

    for field in REQUIRED_FIELDS:
        if re.search(rf"^- {re.escape(field)}:", text, flags=re.MULTILINE) is None:
            raise ValueError(f"record is missing field line: {field}")


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
        validate_v1_record_structure(text)

        updated = text
        for field, (value, code) in build_updates(args).items():
            updated = replace_field(updated, field, value, code=code)

        record_path.write_text(updated, encoding="utf-8")
    except ValueError as exc:
        return fail(str(exc))

    print(record_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
