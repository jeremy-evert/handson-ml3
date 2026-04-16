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
