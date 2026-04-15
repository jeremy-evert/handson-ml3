# 005 Make Baby Prompt Runner Resilient to Codex Session Storage Failures

Your task is to inspect and fix the remaining failure in `tools/codex/baby_run_prompt.py`.

## Context

The immediate Python compatibility bug in timestamp generation has already been fixed.

The current remaining failure happens when this command is run:

```bash
./tools/codex/baby_run_prompt.py 001_smoke_test_pipeline.md
```

The script now starts correctly, creates the timestamped note, and then fails during the `codex exec` subprocess call.

The failure recorded in the note is:

```text
WARNING: proceeding, even though we could not update PATH: Read-only file system (os error 30)
ERROR codex_core::codex: Failed to create session: Read-only file system (os error 30)
Error: thread/start: thread/start failed: error creating thread: Fatal error: Failed to initialize session: Read-only file system (os error 30)
```

That means the runner is no longer blocked on Python itself. It is now blocked on how `codex exec` behaves in the current environment.

## Goal

Update `tools/codex/baby_run_prompt.py` so it is resilient to this Codex session/storage problem and still provides a useful workflow.

## Required approach

Treat this as an environment-compatibility and graceful-degradation problem.

The work should:

1. Inspect how `codex exec` is being invoked now.
2. Determine whether the failure can be avoided by setting a writable environment location for Codex session/state data.
3. If that is possible, update the script to do so in a small, explicit, inspectable way.
4. If that is not possible, make the script fail gracefully and leave behind a clearly useful note instead of a brittle partial result.

## Expected behavior

After the fix, the script should do the best available thing in this environment:

1. Still resolve prompts from `codex_prompts/`.
2. Still create a timestamped note in `notes/`.
3. Still attempt the direct Codex run path first.
4. Avoid hidden assumptions about writable home/session locations.
5. Either:
   - successfully run Codex by redirecting its writable state into a safe writable location inside the repo or another known writable path, or
   - detect the infrastructure failure cleanly and record a short, explicit fallback result in the note.

## Constraints

* Keep the script small.
* Use only the Python standard library.
* Do not build a framework.
* Do not add queueing or orchestration features.
* Do not modify unrelated files unless a very small supporting change is clearly necessary.
* Prefer an explicit environment override over a fragile workaround.

## Implementation guidance

Consider whether the subprocess environment for `codex exec` should override variables such as `HOME`, `XDG_*`, or another Codex-related writable path, but only if inspection shows that is the correct direction.

If you introduce a fallback mode, keep it conservative:

* preserve the note file
* explain that Codex execution was blocked by environment/session initialization
* keep the output readable
* return a nonzero exit status when the direct run still cannot complete

Do not hide infrastructure failure by pretending the prompt succeeded.

## Deliverable

Update:

```text
tools/codex/baby_run_prompt.py
```

If needed, make one very small supporting change elsewhere, but only if justified by the implementation.

## Success criteria

The resulting script should be robust in the current repo environment and easy to inspect.

At minimum:

```bash
./tools/codex/baby_run_prompt.py 001_smoke_test_pipeline.md
```

should no longer fail for a trivial Python-version issue, and it should handle the Codex session/storage problem in a deliberate, readable way.
