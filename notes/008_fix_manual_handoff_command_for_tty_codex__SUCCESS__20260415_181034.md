# 008_fix_manual_handoff_command_for_tty_codex - SUCCESS

- Prompt file: `008_fix_manual_handoff_command_for_tty_codex.md`
- Timestamp (UTC): `20260415_181034`
- Status: `SUCCESS`

## Original Prompt

```md
# 008 Fix Manual Handoff Command for TTY Codex

Your task is to fix the manual-handoff behavior in `tools/codex/baby_run_prompt.py`.

## Context

The script now supports a manual mode:

```bash
./tools/codex/baby_run_prompt.py --manual <prompt>
```

That mode creates a handoff note and prints a suggested Codex command for a human to run externally.

However, the current suggested command is broken for interactive Codex usage.

Right now it prints a command like:

```bash
codex -C "/data/git/handson-ml3" - < "/data/git/handson-ml3/codex_prompts/001_smoke_test_pipeline.md"
```

When run, Codex responds with:

```text
Error: stdin is not a terminal
```

So the current manual handoff is not actually usable.

## Goal

Update `tools/codex/baby_run_prompt.py` so manual handoff produces a command and workflow that actually works for a human using the interactive Codex CLI.

## Required behavior

When manual mode is used, the script should still:

1. resolve the prompt file
2. create a timestamped note in `notes/`
3. mark it clearly as a handoff/manual state
4. write the original prompt into the note

But it should no longer print a broken stdin-redirection command for interactive Codex.

## What to determine

Inspect the available conservative options and choose the simplest one that is actually usable.

Possible directions include:

* printing a `codex` command that opens the interactive TUI with an initial prompt argument instead of stdin redirection
* printing a short sequence of manual steps rather than a single broken command
* generating a temporary or persistent handoff file that the human can paste from
* using a different Codex invocation mode only if it is clearly appropriate

Do not assume stdin piping into plain `codex` is acceptable if the CLI requires a TTY.

## Constraints

* Keep the script small.
* Use only the Python standard library.
* Do not build a framework.
* Do not add queue orchestration.
* Do not modify unrelated files.

## Deliverable

Update:

```text
tools/codex/baby_run_prompt.py
```

## Success criteria

After the change:

```bash
./tools/codex/baby_run_prompt.py --manual codex_prompts/001_smoke_test_pipeline.md
```

should create a handoff note and print instructions that are actually usable with Codex.

The output should no longer direct the user to run a command that fails with:

```text
Error: stdin is not a terminal
```

Be explicit about whether the user is expected to:

* launch interactive Codex with an initial prompt argument
* open Codex first and paste text manually
* or follow another small manual sequence

The handoff should be honest, clear, and workable.
```

## Codex Output

Created the next indexed prompt at `codex_prompts/008_fix_manual_handoff_command_for_tty_codex.md`.

This prompt targets the actual failure in the current manual workflow:

* `baby_run_prompt.py --manual ...` prints a handoff command
* that command uses stdin redirection into plain `codex`
* the interactive Codex CLI rejects that path with `Error: stdin is not a terminal`

The prompt asks for a conservative fix that keeps the handoff model but changes the command and instructions so they are usable with a TTY-based Codex workflow.

## Notes

This note records prompt creation only. The prompt itself has not been executed yet.
