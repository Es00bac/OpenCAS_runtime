# OpenCAS Documentation Map

Last updated: 2026-04-09

Purpose:
- define which documents are current source-of-truth
- prevent older audits and comparison notes from silently becoming stale live guidance
- keep collaborators from branching documentation unnecessarily

## Current Source Of Truth

- [TaskList.md]((workspace_root)/TaskList.md)
  - current execution list
  - use this first when deciding what to work on

- [production-readiness-status-2026-04-09.md]((workspace_root)/docs/production-readiness-status-2026-04-09.md)
  - current readiness assessment
  - capability vs gap analysis
  - next qualification-driven work

- [opencas-deep-system-audit-2026-04-09.md]((workspace_root)/docs/opencas-deep-system-audit-2026-04-09.md)
  - current deep architectural and behavioral audit
  - subsystem interaction map
  - present-vs-target capability assessment

- [first-regular-use-deployment-checklist.md]((workspace_root)/docs/first-regular-use-deployment-checklist.md)
  - explicit gate for first regular-use deployment testing
  - current deployment decision document

- [opencas-production-program-plan-2026-04-08.md]((workspace_root)/docs/opencas-production-program-plan-2026-04-08.md)
  - long-horizon program plan
  - milestone structure and acceptance direction

- [testing-execution-plan-2026-04-09.md]((workspace_root)/docs/qualification/testing-execution-plan-2026-04-09.md)
  - active testing workflow
  - process hygiene expectations
  - mini-vs-High role split

- [long-scenario-matrix.md]((workspace_root)/docs/qualification/long-scenario-matrix.md)
  - current longer daily-use scenario definitions

- [live_validation_summary.md]((workspace_root)/docs/qualification/live_validation_summary.md)
  - current qualification snapshot

- [qualification_remediation_rollup.md]((workspace_root)/docs/qualification/qualification_remediation_rollup.md)
  - current rerun-to-action guidance

- [CLAUDE.md]((workspace_root)/CLAUDE.md)
  - collaborator guidance and current runtime conventions

- [AGENTS.md]((workspace_root)/AGENTS.md)
  - project context and non-negotiable directives

## Active Reference Documents

- [claude-codex-handoff-2026-04-08.md]((workspace_root)/docs/claude-codex-handoff-2026-04-08.md)
  - current handoff reference
  - keep updated when the active frontier changes

- [opencas-production-readiness-audit-2026-04-08.md]((workspace_root)/docs/opencas-production-readiness-audit-2026-04-08.md)
  - still useful as the baseline broad audit
  - not the live execution plan

## Historical / Reference-Only Documents

- [opencas-comprehensive-audit.md]((workspace_root)/docs/opencas-comprehensive-audit.md)
- [opencas-architecture-audit.md]((workspace_root)/docs/opencas-architecture-audit.md)
- [opencas-architecture-and-comparison.md]((workspace_root)/docs/opencas-architecture-and-comparison.md)
- [notes/comprehensive-comparison-report.md]((workspace_root)/notes/comprehensive-comparison-report.md)
- [notes/deep-code-audit-2026-04-06.md]((workspace_root)/notes/deep-code-audit-2026-04-06.md)
- [notes/opencass-vs-legacy_agent_v4-realistic-assessment.md]((workspace_root)/notes/opencass-vs-legacy_agent_v4-realistic-assessment.md)

Use these for context and historical reasoning only. Do not treat them as the current plan or task source.

## Qualification Artifacts

- [live_validation_summary.json]((workspace_root)/docs/qualification/live_validation_summary.json)
- [qualification_remediation_rollup.json]((workspace_root)/docs/qualification/qualification_remediation_rollup.json)
- [audio]((workspace_root)/docs/qualification/audio)

These are generated or semi-generated artifacts. Keep them current, but do not treat them as the only narrative documentation.

## Maintenance Rule

When creating or updating docs:
- prefer updating one of the current source-of-truth files above
- only create a new doc when it has a distinct long-lived purpose
- if a new doc replaces an old one, add the old doc here under historical/reference-only rather than leaving the relationship implicit
