# 003 Upgrade Baby Prompt Runner

Your task is to inspect and improve `tools/codex/baby_run_prompt.py`.

## Context

Right now, `baby_run_prompt.py` reads a prompt file and creates a timestamped markdown note in `notes/`, but it does **not** actually execute Codex or capture any real Codex output.

That means the current script is functioning only as a note scaffold generator, not as a useful prompt runner.

We want to improve it conservatively.

## Goals

Update `tools/codex/baby_run_prompt.py` so that it becomes a better manual-assist prompt runner.

The improved version should:

1. Read a prompt file from `codex_prompts/`
2. Create a timestamped note file in `notes/`
3. Write the prompt content into that note file
4. Also generate a clearly separated **execution bundle** for manual use with Codex
5. Print that execution bundle to stdout so it can be copied directly into Codex
6. Include enough structure that the user can manually paste Codex output back into the note file
7. Remain small, readable, and dependency-light

## Conservative design requirements

Do **not** build a large framework.

Do **not** add queue logic, retry orchestration, status reconstruction, or automatic Codex API integration.

Do **not** add unnecessary abstraction.

Keep the script focused on one job:
- prepare one prompt
- create one note
- emit one clean execution bundle

## Suggested behavior

When run like this:

```bash
./tools/codex/baby_run_prompt.py 002_repo_inventory_and_status.md
