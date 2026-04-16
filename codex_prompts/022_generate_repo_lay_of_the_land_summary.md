# Task: Generate a quick lay-of-the-land summary for this repository

You are working in this repository.

Your task is to produce a short, practical repository inventory that helps a human quickly understand what is here, what the main folders are for, and what the most important top-level files are for.

Think of this as a lightweight, annotated tree-style overview.

Keep it concise, readable, and useful for someone getting re-oriented before doing more work in the repo.

## Primary goal

Create one short markdown summary that:

1. lists the major top-level folders in the repository
2. gives each major folder a one-line or two-line explanation
3. lists the most important top-level files
4. gives each important file a one-line explanation
5. helps a reviewer understand the current shape of the repo without reading everything

## Files and areas to inspect first

Start by inspecting at least:

- `README.md`
- `codex_prompts/`
- `notes/`
- `tools/`
- `Weather_Agreement_Lab/`
- `Codex_Weather_Fusion/`
- the top-level notebook files
- any obviously important top-level markdown files

You may inspect additional files as needed, but stay lightweight.

## Required output artifact

Create exactly one new note:

- `notes/022_repo_lay_of_the_land__TIMESTAMP.md`

## What the note should contain

The note should include these sections in this general order:

### 1. Repo snapshot

A short paragraph that explains the overall character of the repo.

Example kinds of things to capture:

- whether this looks like a book-study repo, an experiments repo, a tooling repo, or a hybrid
- whether recent work appears concentrated in a few folders
- whether there are obvious active workflow areas

### 2. Top-level folders

Provide a concise annotated list of the main top-level folders.

For each folder, include:

- folder name
- short purpose line
- optional second line if needed for context

Do not try to document every tiny folder in depth.
Focus on the ones that matter most.

### 3. Important top-level files

Provide a concise annotated list of the most important top-level files.

For each file, include:

- file name
- short purpose line

Group similar files together where helpful, such as:

- environment/setup files
- documentation files
- notebook files
- generated outputs / images

### 4. Current center of gravity

Write a short section describing where the repo seems most active right now.

This should identify the folders or files that look like the current working area.

### 5. Suggested “start here” reading order

Give a very short recommended reading path for a human who wants to understand the repo quickly.

For example, something like:

1. `README.md`
2. `codex_prompts/`
3. `tools/codex/`
4. recent `notes/`

Adjust the order based on what you actually find.

## Constraints

- Do not rewrite or reorganize the repo
- Do not rename files
- Do not create a large architecture document
- Do not create a giant recursive inventory of every file
- Do not produce raw `tree` output only
- Do not be overly verbose
- Keep the final note compact enough that someone could read it in a few minutes

## Style requirements

- Be concrete
- Be brief
- Prefer useful explanation over exhaustiveness
- Use markdown headings and bullet lists
- Make it feel like an annotated repo map, not a wall of prose

## Validation requirements

Before finishing:

1. Make sure the note actually reflects the current repo contents.
2. Make sure each major folder listed has a short explanation.
3. Make sure the most important top-level files are called out explicitly.
4. Keep the summary short enough to be practical.

## Success criteria

This task is successful if:

1. A reviewer can open one note and quickly understand the repo’s overall shape.
2. The main folders are listed with useful explanations.
3. The important top-level files are listed with useful explanations.
4. The result feels like a human-friendly annotated tree summary rather than a raw dump.