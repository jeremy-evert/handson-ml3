# V1 Bridge Runner Change Spec

## Purpose

Describe the current V1 bridge runner and adjacent helpers so this spec matches the implemented workflow without a broader refactor.

The target outcome is:

- execution produces one durable V1 execution record
- execution outcome and review outcome are kept separate
- run identity is stable
- the runner captures only the minimum automatic fields
- human review fields remain manual

## Scope

This spec covers the current V1 runner at [run_prompt.py](/data/git/handson-ml3/tools/codex/run_prompt.py) and the small helper scripts that operate on the same execution-record contract.

It defines:

- the record filename pattern
- the markdown section order
- the exact fields the runner should populate automatically
- the manual review fields the runner should initialize for later completion
- the minimal mapping from subprocess result to execution status
- the minimal metrics to capture now
- the current helper support around review write-back, queue readiness, and review backlog inspection

## Current Behavior Summary

The current runner at `tools/codex/run_prompt.py`:

- resolves one prompt path from an argument using the existing lookup rules
- validates that `codex_prompts/` and `notes/` exist
- reads the prompt text from the selected file
- runs `codex exec -C <repo_root> --output-last-message <tempfile> -`
- captures subprocess return code, last-message output, and stderr
- writes one markdown execution record into `notes/`
- uses base run identity `<prompt_stem>__<started_at_utc>`
- adds a numeric suffix such as `__2` only when needed to avoid same-second collisions for the same prompt
- initializes `review_status: UNREVIEWED`
- prints the written note path
- exits with the subprocess return code

The current helper set also includes:

- `tools/codex/review_run.py` for manual review write-back into the same record
- `tools/codex/check_queue_readiness.py` for conservative readiness checks
- `tools/codex/list_review_backlog.py` for backlog inspection from `notes/`

The current gap is therefore no longer missing helper support.
The remaining need is only to keep this document aligned to the implemented V1 behavior.

## Required V1 Changes

### 1. Replace success/failure note naming with stable run identity naming

The runner should stop writing filenames in the form:

`<prompt_stem>__<status>__<timestamp>.md`

It should instead write:

`notes/<prompt_stem>__<started_at_utc>.md`

This is the base V1 identity.
If that path already exists for the same prompt and second, `run_prompt.py` appends a numeric suffix:

`notes/<prompt_stem>__<started_at_utc>__2.md`

This filename is the run's stable identity carrier and must not encode:

- `SUCCESS`
- `FAILED`
- `ACCEPTED`
- `REJECTED`

### 2. Emit a V1 execution record instead of a success-implies-acceptance note

The runner should replace the current freeform note body with a V1 execution record shaped around:

1. Header / identity
2. Execution facts
3. Review facts
4. Failure analysis
5. Resource / cost facts
6. Prompt text
7. Codex final output
8. Stderr or supplemental notes

### 3. Initialize review fields but leave review work manual

The runner should create a new record with:

- `review_status: UNREVIEWED`
- `review_summary:` blank placeholder

It should not attempt to:

- infer acceptance
- mark `ACCEPTED`
- mark `REJECTED`
- fill reviewer identity
- fill review timestamp
- classify the failure automatically

### 4. Derive execution status only from subprocess execution result

The minimal V1 mapping is:

- subprocess `return_code == 0` -> `execution_status: EXECUTED`
- subprocess `return_code != 0` -> `execution_status: EXECUTION_FAILED`

This mapping controls only execution facts.
It must not affect `review_status` beyond leaving it at `UNREVIEWED`.

### 5. Capture only the minimum automatic metrics now

The runner should capture and write:

- `started_at_utc`
- `finished_at_utc`
- `elapsed_seconds`
- `return_code`
- `final_output_char_count`
- `stderr_char_count`

It should also write:

- `stderr_text` when present

No broader token accounting, file-diff analysis, validation harvesting, or queue-state calculation should be added in this pass.

## Non-Goals

This spec does not authorize:

- splitting the runner into multiple modules
- a larger CLI redesign
- new queue states or queue engines beyond the current V1 readiness rule
- retry orchestration
- dependency-aware scheduling
- broader note/history discovery logic
- structured sidecars, JSON records, or databases

This document is an alignment spec for the current V1 slice.
It is not a request to redesign the already-implemented helper scripts.

## Exact Record Shape

The record should remain plain markdown and use these sections in this order.

### 1. Header / identity

Suggested format:

```md
# <run_id>

- run_id: `<run_id>`
- prompt_file: `<prompt_file>`
- prompt_stem: `<prompt_stem>`
- started_at_utc: `<started_at_utc>`
```

### 2. Execution facts

```md
## Execution Facts

- execution_status: `<EXECUTED|EXECUTION_FAILED>`
- finished_at_utc: `<finished_at_utc>`
- runner: `tools/codex/run_prompt.py`
- return_code: `<int>`
- retry_of_run_id: ``
```

### 3. Review facts

```md
## Review Facts

- review_status: `UNREVIEWED`
- review_summary:
- reviewed_by:
- reviewed_at_utc:
```

### 4. Failure analysis

```md
## Failure Analysis

- failure_type:
- failure_symptom:
- likely_cause:
- recommended_next_action:
```

These fields are initialized only.
They remain manual in V1.

### 5. Resource / cost facts

```md
## Resource / Cost Facts

- elapsed_seconds: `<float_or_decimal_seconds>`
- final_output_char_count: `<int>`
- stderr_char_count: `<int>`
```

### 6. Prompt text

````md
## Prompt Text

```md
<full prompt text>
```
````

### 7. Codex final output

```md
## Codex Final Output

<captured last message, or `*No output captured.*`>
```

### 8. Stderr or supplemental notes

If stderr exists:

````md
## Stderr

```text
<stderr text>
```
````

If stderr is empty:

```md
## Stderr

*No stderr captured.*
```

## Exact Data / Field Mapping

### Run identity mapping

- base V1 identity -> `f"{prompt_path.stem}__{started_at_utc}"`
- `run_id` -> base identity unless a same-second collision exists for the same prompt, then `f"{base_run_id}__{n}"`
- filename -> `notes/{run_id}.md`
- `prompt_stem` -> `prompt_path.stem`
- `started_at_utc` -> timestamp captured immediately before `run_codex(...)`

### Prompt file mapping

- `prompt_file` -> repo-relative POSIX path when the resolved prompt is inside the repo root
- fallback `prompt_file` -> absolute path string only if the resolved prompt is outside the repo root

This is the smallest precise mapping that preserves current prompt resolution behavior without introducing a new prompt identity system.

### Execution mapping

- `runner` -> literal `tools/codex/run_prompt.py`
- `return_code` -> subprocess return code from `subprocess.run(...)`
- `execution_status` -> `EXECUTED` when return code is `0`, else `EXECUTION_FAILED`
- `finished_at_utc` -> timestamp captured immediately after `run_codex(...)` returns
- `prompt_text` section -> full prompt file contents as currently read
- `codex_final_output` section -> current `--output-last-message` file contents

### Review mapping

- `review_status` -> literal `UNREVIEWED`
- `review_summary` -> blank
- `reviewed_by` -> blank
- `reviewed_at_utc` -> blank

### Failure / retry mapping

- `retry_of_run_id` -> blank in this pass
- `failure_type` -> blank
- `failure_symptom` -> blank
- `likely_cause` -> blank
- `recommended_next_action` -> blank

This preserves the manual review boundary from [V1_Run_Review_Gate.md](/data/git/handson-ml3/tools/codex/V1_Run_Review_Gate.md).

### Metrics mapping

- `elapsed_seconds` -> `finished_at_utc - started_at_utc` in seconds
- `final_output_char_count` -> `len(codex_output)`
- `stderr_char_count` -> `len(stderr_text)`
- `stderr_text` section -> full stderr text when non-empty

These are the only required automatic metrics for V1 because they are cheap, inspectable, and already available from the current subprocess flow.

## Review Fields Left Manual

The runner should initialize but not populate these beyond blanks:

- `review_summary`
- `reviewed_by`
- `reviewed_at_utc`
- `failure_type`
- `failure_symptom`
- `likely_cause`
- `recommended_next_action`

The runner should set only one review field automatically:

- `review_status: UNREVIEWED`

## Behaviors That Should Remain Unchanged For V1

The following current behaviors should stay as they are:

- repository-root discovery via `Path(__file__).resolve().parents[2]`
- requirement that `codex_prompts/` and `notes/` already exist
- prompt resolution behavior for absolute paths, repo-relative paths, `codex_prompts/` paths, and unique prefix matches
- reading the entire prompt file as UTF-8 text
- invoking `codex exec` through the existing thin subprocess path
- using `--output-last-message` and preserving stderr capture
- writing exactly one markdown artifact per run into `notes/`
- printing the written record path to stdout
- returning the subprocess return code as the script exit code

These preserved behaviors keep the implementation bridge-sized and avoid a premature runner redesign.

## Current Helper Support Around The Runner

The current V1 workflow around the runner is:

1. `tools/codex/run_prompt.py` writes the execution record with `review_status: UNREVIEWED`
2. `tools/codex/review_run.py` writes back `ACCEPTED` or `REJECTED` review fields into that same file
3. `tools/codex/check_queue_readiness.py` evaluates whether the next prompt is ready under the V1 review gate
4. `tools/codex/list_review_backlog.py` reports the current review backlog from `notes/`

These helpers are intentionally small.
They do not introduce a larger orchestration layer, but they do mean review write-back, readiness checking, and backlog listing are implemented parts of the current V1 slice.
