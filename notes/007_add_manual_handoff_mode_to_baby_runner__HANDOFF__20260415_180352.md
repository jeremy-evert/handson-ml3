# 007_add_manual_handoff_mode_to_baby_runner - HANDOFF

- Prompt file: `007_add_manual_handoff_mode_to_baby_runner.md`
- Timestamp (UTC): `20260415_180352`
- Status: `HANDOFF`

## Original Prompt

```md
# 007 Add Manual Handoff Mode to Baby Runner

Your task is to make `tools/codex/baby_run_prompt.py` usable when automatic `codex exec` cannot complete in the current environment.

## Context

The script already:

* resolves prompt files from `codex_prompts/`
* creates timestamped notes in `notes/`
* attempts direct `codex exec`
* avoids the earlier read-only session initialization failure
* records infrastructure failures cleanly

The remaining practical problem is that direct `codex exec` may still be blocked by environment constraints such as missing network access.

We want a conservative fallback that still helps a human get the work done.

## Goal

Add an explicit manual-handoff mode to `tools/codex/baby_run_prompt.py`.

## Required behavior

Support a command like:

```bash
./tools/codex/baby_run_prompt.py --manual 007_add_manual_handoff_mode_to_baby_runner.md
```

In manual mode, the script should:

1. resolve the prompt file as usual
2. generate a UTC timestamp
3. create a note in `notes/` using the existing naming style, but with a status that clearly indicates handoff
4. write the original prompt into the note
5. include a short manual handoff section in the note
6. print a ready-to-run `codex` command that a human can execute in a terminal/environment where Codex networking works
7. avoid claiming that the prompt was executed automatically

## Handoff content

The manual handoff output should be explicit and easy to use.

It should include:

* the note path
* the prompt path
* an exact suggested `codex` command
* a short instruction that the user should run that command externally and then paste or append the final result into the note

If helpful, the script may also create a small handoff bundle file, but keep this optional and minimal.

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

After the change, the script should support both:

```bash
./tools/codex/baby_run_prompt.py 001_smoke_test_pipeline.md
./tools/codex/baby_run_prompt.py --manual 007_add_manual_handoff_mode_to_baby_runner.md
```

The automatic path should keep its current behavior.

The manual path should create a clear handoff note and print a usable external command without pretending the task already succeeded.
```

## Codex Output

_Manual handoff prepared. No automatic Codex execution was attempted._

## Notes

Manual handoff required.
Prompt path: `/data/git/handson-ml3/codex_prompts/007_add_manual_handoff_mode_to_baby_runner.md`
Note path: `/data/git/handson-ml3/notes/007_add_manual_handoff_mode_to_baby_runner__HANDOFF__20260415_180352.md`
Run this command in a terminal/environment where Codex networking works:
```bash
codex -C "/data/git/handson-ml3" - < "/data/git/handson-ml3/codex_prompts/007_add_manual_handoff_mode_to_baby_runner.md"
```
Then paste or append Codex's final output into this note.
