# 006 Investigate TUI or Manual Handoff Workaround for Baby Runner

Your task is to investigate whether `tools/codex/baby_run_prompt.py` should support a TUI-based or manual-handoff fallback when `codex exec` cannot complete in the current environment.

## Context

`baby_run_prompt.py` has already been improved so that:

1. it no longer fails on older Python versions due to `datetime.UTC`
2. it no longer fails immediately on Codex session-state initialization
3. it records clean failure notes when `codex exec` cannot reach the API or times out

The remaining limitation is environmental:

* in the current sandbox, `codex exec` cannot reach the Codex API
* that means direct non-interactive execution still cannot complete here

There is a question about whether the script could work around this by launching the interactive Codex TUI instead.

## Problem to solve

Determine whether launching the TUI is actually a meaningful workaround.

Do not assume that “interactive” automatically means “works.”

The TUI may still share the same:

* network requirements
* auth requirements
* sandbox restrictions
* API connectivity requirements

So this needs to be evaluated carefully.

## Goals

1. Inspect whether launching plain `codex` interactively would realistically bypass the current failure mode.
2. Distinguish between these possibilities:
   - no, TUI would fail for the same underlying reason
   - yes, TUI could work if launched outside the current constrained environment
   - yes, but only in a manual-assist mode where the script prepares context and hands control to a human
3. If TUI is not a true workaround, identify the best conservative fallback design for the baby runner.
4. If a useful fallback exists, implement it in a small and explicit way.

## Preferred decision criteria

Use a conservative workflow judgment.

A good solution should:

* keep the script small
* remain easy to inspect
* avoid pretending automation succeeded when it did not
* improve real usability for a human operator

## Likely acceptable outcomes

One of these is likely correct:

### Option A: TUI is not a real fix here

If the TUI uses the same blocked backend/network path, then adding a TUI launch mode is not a real solution.

In that case, the script should probably keep the direct runner path and optionally add a **manual handoff bundle** that:

* writes the prompt into the note
* prints a suggested `codex` launch command
* tells the user what to paste or review
* keeps the result clearly marked as manual follow-up required

### Option B: TUI is useful as an operator handoff

If the TUI does not solve the sandbox problem by itself but still helps in practice when run by a human in a different terminal/environment, then the script may add an explicit fallback mode such as:

* `--manual`
* `--tui`
* `--emit-handoff`

That mode should not claim to have executed the prompt automatically. It should only prepare the handoff cleanly.

### Option C: TUI can work directly in a specific configuration

If inspection shows that the TUI can succeed with a different provider or launch style, document exactly what must change and keep the implementation narrow.

## Constraints

* Keep changes small.
* Use only the Python standard library if code changes are made.
* Do not build a framework.
* Do not add queue orchestration.
* Do not modify unrelated files.

## Deliverable

Inspect the current behavior and then either:

1. update `tools/codex/baby_run_prompt.py` with a conservative TUI/manual-handoff fallback, or
2. decide that no code change should be made and record a clear note explaining why

## Success criteria

The result should make the workflow more honest and more usable.

If a fallback is added, it should be explicit about whether it is:

* automatic execution
* manual operator handoff
* interactive-only assistance

Do not blur those categories.
