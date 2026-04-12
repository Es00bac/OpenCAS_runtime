# OpenCAS Documentation Map

Last updated: 2026-04-09

Purpose:
- define which documents are current source-of-truth
- prevent older audits and comparison notes from silently becoming stale live guidance
- keep collaborators from branching documentation unnecessarily

## Current Source Of Truth

- [TaskList.md](/mnt/xtra/OpenCAS/TaskList.md)
  - current execution list
  - use this first when deciding what to work on

- [production-readiness-status-2026-04-09.md](/mnt/xtra/OpenCAS/docs/production-readiness-status-2026-04-09.md)
  - current readiness assessment
  - capability vs gap analysis
  - next qualification-driven work

- [opencas-deep-system-audit-2026-04-09.md](/mnt/xtra/OpenCAS/docs/opencas-deep-system-audit-2026-04-09.md)
  - current deep architectural and behavioral audit
  - subsystem interaction map
  - present-vs-target capability assessment

- [first-regular-use-deployment-checklist.md](/mnt/xtra/OpenCAS/docs/first-regular-use-deployment-checklist.md)
  - explicit gate for first regular-use deployment testing
  - current deployment decision document

- [opencas-production-program-plan-2026-04-08.md](/mnt/xtra/OpenCAS/docs/opencas-production-program-plan-2026-04-08.md)
  - long-horizon program plan
  - milestone structure and acceptance direction

- [testing-execution-plan-2026-04-09.md](/mnt/xtra/OpenCAS/docs/qualification/testing-execution-plan-2026-04-09.md)
  - active testing workflow
  - process hygiene expectations
  - mini-vs-High role split

- [long-scenario-matrix.md](/mnt/xtra/OpenCAS/docs/qualification/long-scenario-matrix.md)
  - current longer daily-use scenario definitions

- [live_validation_summary.md](/mnt/xtra/OpenCAS/docs/qualification/live_validation_summary.md)
  - current qualification snapshot

- [qualification_remediation_rollup.md](/mnt/xtra/OpenCAS/docs/qualification/qualification_remediation_rollup.md)
  - current rerun-to-action guidance

- [CLAUDE.md](/mnt/xtra/OpenCAS/CLAUDE.md)
  - collaborator guidance and current runtime conventions

- [AGENTS.md](/mnt/xtra/OpenCAS/AGENTS.md)
  - project context and non-negotiable directives

## Active Reference Documents

- [claude-codex-handoff-2026-04-08.md](/mnt/xtra/OpenCAS/docs/claude-codex-handoff-2026-04-08.md)
  - current handoff reference
  - keep updated when the active frontier changes

- [opencas-production-readiness-audit-2026-04-08.md](/mnt/xtra/OpenCAS/docs/opencas-production-readiness-audit-2026-04-08.md)
  - still useful as the baseline broad audit
  - not the live execution plan

## Historical / Reference-Only Documents

- [opencas-comprehensive-audit.md](/mnt/xtra/OpenCAS/docs/opencas-comprehensive-audit.md)
- [opencas-architecture-audit.md](/mnt/xtra/OpenCAS/docs/opencas-architecture-audit.md)
- [opencas-architecture-and-comparison.md](/mnt/xtra/OpenCAS/docs/opencas-architecture-and-comparison.md)
- [notes/comprehensive-comparison-report.md](/mnt/xtra/OpenCAS/notes/comprehensive-comparison-report.md)
- [notes/deep-code-audit-2026-04-06.md](/mnt/xtra/OpenCAS/notes/deep-code-audit-2026-04-06.md)
- [notes/opencass-vs-openbulma-v4-realistic-assessment.md](/mnt/xtra/OpenCAS/notes/opencass-vs-openbulma-v4-realistic-assessment.md)

Use these for context and historical reasoning only. Do not treat them as the current plan or task source.

## Qualification Artifacts

- [live_validation_summary.json](/mnt/xtra/OpenCAS/docs/qualification/live_validation_summary.json)
- [qualification_remediation_rollup.json](/mnt/xtra/OpenCAS/docs/qualification/qualification_remediation_rollup.json)
- [audio](/mnt/xtra/OpenCAS/docs/qualification/audio)

These are generated or semi-generated artifacts. Keep them current, but do not treat them as the only narrative documentation.

## Maintenance Rule

When creating or updating docs:
- prefer updating one of the current source-of-truth files above
- only create a new doc when it has a distinct long-lived purpose
- if a new doc replaces an old one, add the old doc here under historical/reference-only rather than leaving the relationship implicit
