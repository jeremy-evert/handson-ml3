# 002 Repo Inventory and Status Report

Your task is to inspect the current repository and create a concise status report.

## Goals
1. Inventory the top-level structure of the repository
2. Identify important files and directories
3. Summarize the apparent purpose of the repo
4. Identify obvious missing pieces, unfinished work, or areas that may need attention
5. Write a report to the `notes/` folder

## Report filename
Create a markdown note in `notes/` whose filename includes:
- `002_repo_inventory_and_status`
- a success or fail marker
- a timestamp

Example:
- `002_repo_inventory_and_status__SUCCESS__YYYYMMDD_HHMMSS.md`

## Report contents
Include these sections:

### 1. Scope
Briefly describe what repository was inspected.

### 2. Top-Level Inventory
List the important top-level files and directories.

### 3. Purpose Guess
Based on the visible contents, explain what this repository appears to be for.

### 4. Current State
Summarize signs of maturity, incompleteness, structure, documentation, scripts, notebooks, and other notable characteristics.

### 5. Risks or Gaps
List anything obviously missing or worth attention, such as:
- missing documentation
- no clear entry point
- no requirements file
- no tests
- incomplete notebooks
- naming inconsistency
- prompt workflow not yet automated

### 6. Summary
Give a short overall repo status summary.

## Constraints
- Do not modify repo contents except for writing the report in `notes/`
- Be concise but useful
- If the repo cannot be inspected, mark the run as FAIL and explain why
