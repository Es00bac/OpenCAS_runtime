# OpenCAS Documentation Map

Last updated: 2026-04-15

Purpose:
- define which documents are current source-of-truth
- prevent older audits and comparison notes from silently becoming stale live guidance
- keep collaborators from branching documentation unnecessarily

## Current Source Of Truth

- [TaskList.md](../TaskList.md)
  - current execution list
  - use this first when deciding what to work on

- [production-readiness-status-2026-04-09.md](production-readiness-status-2026-04-09.md)
  - current readiness assessment
  - capability vs gap analysis
  - next qualification-driven work

- [opencas-deep-system-audit-2026-04-09.md](opencas-deep-system-audit-2026-04-09.md)
  - current deep architectural and behavioral audit
  - subsystem interaction map
  - present-vs-target capability assessment

- [first-regular-use-deployment-checklist.md](first-regular-use-deployment-checklist.md)
  - explicit gate for first regular-use deployment testing
  - current deployment decision document

- [opencas-production-program-plan-2026-04-08.md](opencas-production-program-plan-2026-04-08.md)
  - long-horizon program plan
  - milestone structure and acceptance direction

- [opencas-continuation-program-2026-04-15.md](opencas-continuation-program-2026-04-15.md)
  - current continuation program for the commitment-continuity track
  - use as subsystem context and acceptance guidance for promise/commitment behavior, not as the primary cleanup task board

- [opencas-cleanup-program-2026-04-15.md](opencas-cleanup-program-2026-04-15.md)
  - current bounded cleanup program for deduplication, doc-truth maintenance, and subsystem-friendly refactors
  - this is the primary planning doc when the active work is structural cleanup/refactor rather than new capability delivery

- [testing-execution-plan-2026-04-09.md](qualification/testing-execution-plan-2026-04-09.md)
  - active testing workflow
  - process hygiene expectations
  - mini-vs-High role split

- [long-scenario-matrix.md](qualification/long-scenario-matrix.md)
  - current longer daily-use scenario definitions

- [live_validation_summary.md](qualification/live_validation_summary.md)
  - current qualification snapshot

- [qualification_remediation_rollup.md](qualification/qualification_remediation_rollup.md)
  - current rerun-to-action guidance

- [CLAUDE.md](../CLAUDE.md)
  - collaborator guidance and current runtime conventions
  - contains historical milestone context; pair with `TaskList.md` for live state

- [AGENTS.md](../AGENTS.md)
  - project context and non-negotiable directives
  - contains 2026-04-08 historical context plus refreshed 2026-04-15 guidance

## Active Reference Documents

- [handoff-2026-04-15-commitment-consolidation.md](handoff-2026-04-15-commitment-consolidation.md)
  - focused handoff for the unfinished Claude 2026-04-14 through 2026-04-15 commitment work
  - use alongside the 2026-04-15 continuation program, not instead of it
  - parts of it are now historical because `PR-019` through `PR-021` have already corrected some of the originally described gaps

- [opencas-production-readiness-audit-2026-04-08.md](opencas-production-readiness-audit-2026-04-08.md)
  - still useful as the baseline broad audit
  - not the live execution plan

## Historical / Reference-Only Documents

- [claude-codex-handoff-2026-04-08.md](claude-codex-handoff-2026-04-08.md)
- [opencas-comprehensive-audit.md](opencas-comprehensive-audit.md)
- [opencas-architecture-audit.md](opencas-architecture-audit.md)
- [opencas-architecture-and-comparison.md](opencas-architecture-and-comparison.md)
- [notes/comprehensive-comparison-report.md](../notes/comprehensive-comparison-report.md)
- [notes/deep-code-audit-2026-04-06.md](../notes/deep-code-audit-2026-04-06.md)
- [notes/opencass-vs-openbulma-v4-realistic-assessment.md](../notes/opencass-vs-openbulma-v4-realistic-assessment.md)

Use these for context and historical reasoning only. Do not treat them as the current plan or task source.

## Qualification Artifacts

- [live_validation_summary.json](qualification/live_validation_summary.json)
- [qualification_remediation_rollup.json](qualification/qualification_remediation_rollup.json)
- [audio](qualification/audio)

These are generated or semi-generated artifacts. Keep them current, but do not treat them as the only narrative documentation.

## Maintenance Rule

When creating or updating docs:
- prefer updating one of the current source-of-truth files above
- only create a new doc when it has a distinct long-lived purpose
- if a new doc replaces an old one, add the old doc here under historical/reference-only rather than leaving the relationship implicit
