# 004_simplify_baby_run_prompt_to_direct_runner - STARTED

- Prompt file: `004_simplify_baby_run_prompt_to_direct_runner.md`
- Note file: `004_simplify_baby_run_prompt_to_direct_runner__STARTED__20260415_173759.md`
- Timestamp (UTC): `20260415_173759`
- Status: `STARTED`

## Prompt

```md
# 004 Simplify Baby Prompt Runner to Direct Runner

Your task is to rewrite `tools/codex/baby_run_prompt.py` into a very small direct-run prompt runner.

## Context

The current version of `baby_run_prompt.py` is too ceremonial for the immediate need.

Right now we do **not** want:
- a manual execution bundle
- queue logic
- retry logic
- orchestration
- framework structure
- extra abstractions

We want one small script that does one job:

- take one prompt document
- run Codex on it
- store the result in a timestamped note file

## Goal

Replace the current behavior with a simpler direct-run workflow.

When this command is run:

```bash
./tools/codex/baby_run_prompt.py 002_repo_inventory_and_status.md
````

the script should:

1. resolve the prompt file from `codex_prompts/` if needed
2. generate a UTC timestamp
3. create a note file in `notes/` with this pattern:

```text
<prompt_stem>__STARTED__YYYYMMDD_HHMMSS.md
```

4. write a markdown note that includes:

   * prompt file name
   * timestamp
   * status
   * the full original prompt text
   * a `## Codex Output` section
   * a `## Notes` section

5. run Codex directly using a subprocess call equivalent to:

```bash
codex exec -C . --output-last-message <tempfile> - < <promptfile>
```

6. append the final Codex output into the `## Codex Output` section of the note
7. print the final note path to stdout
8. exit with a nonzero code if Codex execution fails

## Constraints

* Keep this as a **single small Python file**
* Use only the Python standard library
* Do not add a framework
* Do not add queueing
* Do not add retries
* Do not add status reconstruction
* Do not print an execution bundle
* Do not require manual copy/paste for the normal path
* Do not modify unrelated files

## Implementation guidance

Use small, readable helpers where needed, but keep the script simple.

A good structure would be:

* timestamp helper
* repo root helper
* prompt path resolver
* note path builder
* note content builder
* Codex execution helper
* main

Use `subprocess.run()` to invoke Codex.

Use a temporary file for `--output-last-message`.

Read the prompt file and pass its contents to Codex through stdin.

Append Codex’s final output into the note after the note is created.

If Codex fails:

* keep the note file
* include an error section or note
* return a nonzero exit code

## Quality bar

This script should feel like a tiny utility, not a platform.

It should be easy for a human to inspect in one sitting.

It should solve the immediate need cleanly:

* pass prompts as docs
* get reports back as docs

## Deliverable

Update only:

```text
tools/codex/baby_run_prompt.py
```

## Success criteria

After the rewrite, this should work:

```bash
./tools/codex/baby_run_prompt.py 001_smoke_test_pipeline.md
./tools/codex/baby_run_prompt.py 002_repo_inventory_and_status.md
```

and each run should leave behind a timestamped markdown note in `notes/` containing:

* the original prompt
* Codex’s final response
* a place for follow-up notes
```

## Execution Bundle

Copy the bundle below into Codex for a manual run.

```text
=== CODEX EXECUTION BUNDLE START ===
Prompt file: 004_simplify_baby_run_prompt_to_direct_runner.md
Note file: /data/git/handson-ml3/notes/004_simplify_baby_run_prompt_to_direct_runner__STARTED__20260415_173759.md
Prepared at (UTC): 20260415_173759

Instructions:
1. Paste this entire bundle into Codex.
2. Let Codex complete the task.
3. Paste Codex's final response into the note file under "Codex Output".
4. Add any local follow-up details under "Notes" if needed.

Task Prompt:

# 004 Simplify Baby Prompt Runner to Direct Runner

Your task is to rewrite `tools/codex/baby_run_prompt.py` into a very small direct-run prompt runner.

## Context

The current version of `baby_run_prompt.py` is too ceremonial for the immediate need.

Right now we do **not** want:
- a manual execution bundle
- queue logic
- retry logic
- orchestration
- framework structure
- extra abstractions

We want one small script that does one job:

- take one prompt document
- run Codex on it
- store the result in a timestamped note file

## Goal

Replace the current behavior with a simpler direct-run workflow.

When this command is run:

```bash
./tools/codex/baby_run_prompt.py 002_repo_inventory_and_status.md
````

the script should:

1. resolve the prompt file from `codex_prompts/` if needed
2. generate a UTC timestamp
3. create a note file in `notes/` with this pattern:

```text
<prompt_stem>__STARTED__YYYYMMDD_HHMMSS.md
```

4. write a markdown note that includes:

   * prompt file name
   * timestamp
   * status
   * the full original prompt text
   * a `## Codex Output` section
   * a `## Notes` section

5. run Codex directly using a subprocess call equivalent to:

```bash
codex exec -C . --output-last-message <tempfile> - < <promptfile>
```

6. append the final Codex output into the `## Codex Output` section of the note
7. print the final note path to stdout
8. exit with a nonzero code if Codex execution fails

## Constraints

* Keep this as a **single small Python file**
* Use only the Python standard library
* Do not add a framework
* Do not add queueing
* Do not add retries
* Do not add status reconstruction
* Do not print an execution bundle
* Do not require manual copy/paste for the normal path
* Do not modify unrelated files

## Implementation guidance

Use small, readable helpers where needed, but keep the script simple.

A good structure would be:

* timestamp helper
* repo root helper
* prompt path resolver
* note path builder
* note content builder
* Codex execution helper
* main

Use `subprocess.run()` to invoke Codex.

Use a temporary file for `--output-last-message`.

Read the prompt file and pass its contents to Codex through stdin.

Append Codex’s final output into the note after the note is created.

If Codex fails:

* keep the note file
* include an error section or note
* return a nonzero exit code

## Quality bar

This script should feel like a tiny utility, not a platform.

It should be easy for a human to inspect in one sitting.

It should solve the immediate need cleanly:

* pass prompts as docs
* get reports back as docs

## Deliverable

Update only:

```text
tools/codex/baby_run_prompt.py
```

## Success criteria

After the rewrite, this should work:

```bash
./tools/codex/baby_run_prompt.py 001_smoke_test_pipeline.md
./tools/codex/baby_run_prompt.py 002_repo_inventory_and_status.md
```

and each run should leave behind a timestamped markdown note in `notes/` containing:

* the original prompt
* Codex’s final response
* a place for follow-up notes

Response Format:
## Codex Output
Paste Codex's final response here.

## Notes
Add any local observations, edits, commands run, or follow-up items here.
=== CODEX EXECUTION BUNDLE END ===
```

## Codex Output

Paste Codex's final response here.

## Notes

Add observations, edits, commands run, or follow-up items here.
