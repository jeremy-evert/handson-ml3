# Project Design Workflow Revision

## What Changed

Revised `tools/Project_Design_Workflow.md` into a more project-agnostic workflow document.
Removed the project-specific application section and replaced it with reusable guidance.
Added explicit sections for bounded iteration, failure analysis, resource and cost awareness, and durable local history.

## Why These Changes Were Made

The earlier version had a strong phased structure, but part of it was tied to one specific workflow and repo context.
The revision keeps the design-first, boundary-aware approach while making the document reusable across different projects, tools, and maturity levels.

## New Reusable Principles Added

* bounded execution in small reviewable slices
* review between iterations before issuing the next step
* bridge tooling allowed when thin, inspectable, and subordinate
* durable local notes, logs, outputs, and reports as project memory
* failure triggering analysis rather than reflexive retry
* lightweight observation of cost, time, retries, and failure patterns

## Tradeoffs and Open Questions

The document is now more broadly reusable, but less concrete for any one project.
That tradeoff seems correct for a shared workflow template.
If later needed, a separate project-specific companion doc could show how this workflow maps onto a particular repo without narrowing the core template.
