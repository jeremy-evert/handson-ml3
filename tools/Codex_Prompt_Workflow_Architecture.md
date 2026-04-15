# Codex Prompt Workflow Architecture

## Goal

Build a small, clean, reusable prompt workflow system for Codex work inside a repository.

This system should be conservative first:

* easy to inspect
* easy to test
* easy to extend
* composed of small parts
* safe by default

The design should emphasize **separation of concerns** so each piece does one job clearly.

---

## Why We Are Re-Architecting

The first draft bundled too many responsibilities into one script. That made it harder to:

* understand
* trust
* test
* evolve
* reuse across repositories

We want a cleaner structure that can become a template for future repos.

---

## What the First Script Was Doing

Below is a decomposition of the responsibilities that were bundled together.

### 1. Repo path discovery

* figure out where the repository root is
* infer where `codex_prompts/` and `notes/` live

### 2. Directory validation

* verify the expected folders exist
* fail cleanly if they do not

### 3. Prompt discovery

* scan `codex_prompts/`
* identify valid prompt files
* ignore files that do not match the naming convention

### 4. Prompt parsing and indexing

* extract numeric prefixes from filenames
* sort prompts in execution order
* establish a stable prompt identity from filename/stem

### 5. Note discovery

* scan `notes/`
* find note files that match the naming convention
* ignore unrelated markdown files

### 6. Note parsing

* extract prompt name, success/fail state, and timestamp from note filenames
* turn note filenames into structured metadata

### 7. Status reconstruction

* map prompts to matching notes
* determine whether each prompt is:

  * UNRUN
  * SUCCESS
  * FAIL
* pick the latest note when there are multiple notes for one prompt

### 8. Prompt selection

* find the next unrun prompt
* find the first failed prompt
* find a prompt by numeric prefix, base name, or full filename

### 9. Prompt display

* print a selected prompt to the terminal

### 10. Retry context assembly

* find the latest failed note for a prompt
* display the original prompt plus the latest failed note
* present instructions for retrying

### 11. Note writing

* create timestamped note filenames
* write notes in markdown
* optionally include the previous note as context

### 12. Manual status marking

* let the user record a success or failure
* attach summary/details text to the note

### 13. Command-line interface

* parse subcommands like `list`, `next`, `show`, `mark`, `retry-failed`
* route commands to the right behaviors

### 14. Terminal presentation

* format output for human readability
* present status tables and retry blocks

---

## What I Would Like This System To Do Eventually

These are ideas worth considering, but they should not all be built at once.

### A. Better architecture and reuse

* work as a reusable template across repositories
* isolate config from code
* support per-repo conventions without rewriting internals

### B. Structured metadata

* optionally store prompt metadata in frontmatter or sidecar files
* support fields like:

  * title
  * objective
  * tags
  * priority
  * dependencies
  * owner

### C. Better status model

Instead of only:

* UNRUN
* SUCCESS
* FAIL

We may want:

* READY
* RUNNING
* BLOCKED
* NEEDS_REVIEW
* PARTIAL
* SKIPPED
* ARCHIVED

### D. Retry intelligence

* carry forward the previous failed note automatically
* include multiple prior failures, not just the latest one
* summarize recurring failure patterns

### E. Prompt preparation

* generate a clean “execution bundle” for Codex
* include:

  * prompt text
  * repo context
  * previous failure note if retrying
  * explicit success criteria

### F. Manual and automated modes

* manual mode for conservative workflows
* dry-run mode for seeing what would happen
* later, optional Codex CLI integration

### G. Report generation

* generate queue reports
* generate status snapshots
* generate audit/history reports over prompt activity

### H. Better note/content model

* move from filename-only state to richer note contents
* enforce a consistent note template
* optionally add structured machine-readable metadata inside notes

### I. Dependency and sequencing support

* allow prompts to depend on other prompts
* prevent prompts from running before prerequisites are complete

### J. Prompt families or lanes

* support categories such as:

  * setup
  * audit
  * refactor
  * notebook generation
  * documentation

### K. Approval gates

* require human confirmation before marking success
* require review before allowing dependent prompts to proceed

### L. Logging and diagnostics

* maintain an execution log
* record tool errors separately from task failures
* distinguish infrastructure failure from prompt failure

### M. File layout that scales cleanly

* support growth without turning into a junk drawer

---

## Proposed Folder Architecture

A better first structure would be:

```text
tools/
  codex/
    README.md
    architecture.md
    config.py
    paths.py
    prompts.py
    notes.py
    status.py
    retry.py
    cli.py
    templates/
      note_success.md
      note_fail.md
```

For an even more conservative start, we can go smaller:

```text
tools/
  codex/
    README.md
    architecture.md
    paths.py
    prompts.py
    notes.py
    status.py
```

Then add a tiny entrypoint later if needed.

---

## Recommended Responsibility Split

### `paths.py`

Owns:

* repo-root discovery
* locating `codex_prompts/`
* locating `notes/`
* validating required folders

Should not own:

* note parsing
* prompt parsing
* CLI logic

### `prompts.py`

Owns:

* prompt discovery
* filename validation
* numeric prefix parsing
* prompt sorting
* prompt lookup by selector

Should not own:

* note history
* note writing
* terminal display

### `notes.py`

Owns:

* note discovery
* note filename parsing
* latest-note selection
* note writing
* note naming convention

Should not own:

* prompt queue decisions
* CLI logic

### `status.py`

Owns:

* combining prompt data and note data
* reconstructing current state
* selecting unrun/failed/successful prompts

Should not own:

* printing or file writing

### `retry.py`

Owns:

* collecting retry context
* pairing original prompt with latest failed note
* preparing a retry bundle

Should not own:

* scanning the repo broadly if other modules already do that

### `cli.py`

Owns:

* argument parsing
* wiring commands to lower-level modules
* user-facing terminal behavior

Should not own:

* core business logic beyond orchestration

---

## Minimal First Build

If we build carefully, the first implementation should do only these things:

### Phase 1

1. discover prompts
2. discover notes
3. reconstruct status
4. print status table

### Phase 2

5. show a prompt
6. show next unrun prompt
7. write a manual success/fail note

### Phase 3

8. assemble retry context from failed note + original prompt

That is enough to start using the system without committing to automation yet.

---

## Naming Conventions

### Prompt files

```text
001_smoke_test_pipeline.md
002_repo_inventory_and_status.md
```

### Note files

```text
001_smoke_test_pipeline__SUCCESS__20260415_094500.md
002_repo_inventory_and_status__FAIL__20260415_095212.md
```

These are simple, inspectable, and script-friendly.

---

## Design Principles

### 1. Plain files first

Use the filesystem as the source of truth before introducing databases or hidden state.

### 2. Small parts

Each file/module should have one clear job.

### 3. Human-readable state

A human should be able to inspect prompts and notes without special tooling.

### 4. Conservative defaults

No automatic execution unless explicitly enabled later.

### 5. Reusability

This should work as a template for future repositories.

### 6. Honest state

Do not pretend a task succeeded merely because a command ran.

### 7. Grow only when pressure demands it

No extra cleverness until we truly need it.

---

## Questions / Design Decisions To Review

### A. What should count as identity?

Should prompt identity be based only on filename stem, or should we eventually support IDs inside prompt files?

### B. How rich should notes be?

Do we want plain markdown only, or markdown plus structured frontmatter?

### C. What statuses do we really want in V1?

Is `UNRUN / SUCCESS / FAIL` enough at first?

### D. How should retries work?

Should a retry include only the latest failed note, or should it include a short history?

### E. Should prompts ever depend on one another?

If yes, that affects architecture early.

### F. When do we introduce Codex execution?

Do we want:

* never in core tools
* later as an adapter
* later as an optional plugin layer

### G. What belongs in notes versus separate logs?

Human-facing narrative and machine-facing diagnostics may eventually deserve separate homes.

---

## Recommended Next Step

Before writing new code, review this architecture and decide:

1. the minimal V1 feature set
2. the final folder layout for `tools/codex/`
3. the exact status model for V1
4. whether notes stay plain markdown only
5. whether retries should include one failed note or a short chain

After that, implement only the smallest slice needed to make the workflow real.

---

## Suggested Initial Build Target

My current recommendation for the smallest clean starting point is:

* `tools/codex/paths.py`
* `tools/codex/prompts.py`
* `tools/codex/notes.py`
* `tools/codex/status.py`
* `tools/codex/README.md`

No CLI yet unless we decide it is truly needed.

That would let us validate the architecture before we wrap it in commands.

---

## Closing Thought

The right next move is not “build the runner.”

The right next move is:

* define the pieces
* define the responsibilities
* define the boundaries
* build the smallest useful slice

That gives us something we can trust, reuse, and extend without regret.

