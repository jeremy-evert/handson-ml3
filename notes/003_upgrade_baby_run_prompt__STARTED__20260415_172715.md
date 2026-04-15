# 003_upgrade_baby_run_prompt - STARTED

- Prompt file: `003_upgrade_baby_run_prompt.md`
- Timestamp (UTC): `20260415_172715`
- Status: `STARTED`

## Prompt

```md
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
```

## Codex Output

*Paste or append Codex output here.*

## Notes

*Add observations, fixes, or follow-up thoughts here.*


## Codex Output

Updated [tools/codex/baby_run_prompt.py](/data/git/handson-ml3/tools/codex/baby_run_prompt.py) so it now does the full manual-assist flow: it still reads one prompt and creates one timestamped note, but it also builds a clearly delimited execution bundle, writes that bundle into the note, and prints the same bundle to stdout for direct copy/paste into Codex.

The note now includes separate `Prompt`, `Execution Bundle`, `Codex Output`, and `Notes` sections, so there’s an obvious place to paste Codex’s final response back into the file. I kept it as a small single-file script with a couple of formatting helpers and no framework/API logic.

Verification: ran `python3 tools/codex/baby_run_prompt.py 002_repo_inventory_and_status.md` successfully. It created [notes/002_repo_inventory_and_status__STARTED__20260415_172816.md](/data/git/handson-ml3/notes/002_repo_inventory_and_status__STARTED__20260415_172816.md) and printed the execution bundle to stdout as intended. The script file is now executable (`100755`), which matches the intended `./tools/codex/baby_run_prompt.py ...` usage.