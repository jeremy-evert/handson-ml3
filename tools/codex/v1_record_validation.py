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
