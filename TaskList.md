# OpenCAS Task List

Last updated: 2026-04-12

Purpose:
- provide the canonical task list referenced by [AGENTS.md]((workspace_root)/AGENTS.md) and [CLAUDE.md]((workspace_root)/CLAUDE.md)
- track the active path to first regular-use deployment readiness
- keep multi-model collaboration grounded in one current execution list

Rules:
- update this file when a task starts, changes scope, or completes
- do not create parallel task lists elsewhere
- treat this file as the execution source of truth; older audits and handoff notes are reference only

Canonical current docs:
- [TaskList.md]((workspace_root)/TaskList.md)
- [documentation-map.md]((workspace_root)/docs/documentation-map.md)
- [opencas-deep-system-audit-2026-04-09.md]((workspace_root)/docs/opencas-deep-system-audit-2026-04-09.md)
- [production-readiness-status-2026-04-09.md]((workspace_root)/docs/production-readiness-status-2026-04-09.md)
- [opencas-production-program-plan-2026-04-08.md]((workspace_root)/docs/opencas-production-program-plan-2026-04-08.md)
- [testing-execution-plan-2026-04-09.md]((workspace_root)/docs/qualification/testing-execution-plan-2026-04-09.md)
- [live_validation_summary.md]((workspace_root)/docs/qualification/live_validation_summary.md)
- [qualification_remediation_rollup.md]((workspace_root)/docs/qualification/qualification_remediation_rollup.md)

## Pending

- `PR-015` SparkEvaluator (structured novelty filter)
  - owner: unassigned
  - goal: add a novelty-scoring gate between `DaydreamGenerator` output and `CreativeLadder` entry; score on cosine distance from existing `WorkObject`s, somatic/relational alignment, executive feasibility; promote only sparks above a novelty floor
  - files: `opencas/daydream/spark_evaluator.py` (new), `opencas/runtime/daydream.py`

- `PR-016` Procedural memory extraction
  - owner: unassigned
  - goal: post-task hook in `BoundedAssistantAgent` on SUCCEEDED that summarizes the tool sequence into a `procedural_memory` episode kind with embeddings; wire `MemoryRetriever` to surface matching procedural episodes for similar future tasks
  - files: `opencas/execution/baa.py`, `opencas/memory/store.py`, `opencas/context/retriever.py`

## In Progress

- `PR-001` Qualification matrix and repeated weak-label reruns
  - owner: Codex
  - status: in_progress
  - goal: move weak labels from one-off evidence to repeated bounded evidence
  - current focus:
    - `kilocode_supervised_work`
    - `integrated_operator_workflow`
  - next acceptance step:
    - run another bounded weak-label rerun only when remediation guidance still says `continue_testing`

- `PR-003` Longer integrated day-to-day scenarios
  - owner: Codex
  - status: in_progress
  - goal: validate longer multi-step daily-use runs, not just bounded artifact tasks
  - current output:
    - [long-scenario-matrix.md]((workspace_root)/docs/qualification/long-scenario-matrix.md)
  - current focus:
    - Scenario 1: completed successfully through run `debug-validation-20260409-164343`
    - Scenario 2: completed successfully through local report `scenario2-repo-triage-20260409-171519`
    - Scenario 3: completed successfully through local report `scenario3-operator-recovery-20260409-170416`
    - Scenario 4: completed successfully through local report `scenario4-tool-friction-20260409-172334`
  - next acceptance step:
    - define the next longer composite scenario only if remaining deployment gaps still require it

- `PR-002` First-regular-use readiness board
  - owner: Codex
  - status: in_progress
  - goal: keep the deployment-readiness picture current in docs and tasking
  - next acceptance step:
    - maintain current status, active gaps, and milestone state without letting older audits drift into live guidance



## Recently Completed

- `PR-014` Belief pollution resistance
  - status: completed
  - result: added `_sweep_belief_consistency` to `NightlyConsolidationEngine` that decays high-confidence ToM beliefs lacking corroborating episodes in the last 7 days. Added `belief_revision_score` field to `Belief` model and `TomStore`. Wired `tom_store` into `AgentRuntime` consolidation initialization. Added tests for no-store, stale-belief, and no-evidence decay paths.

- `PR-018` Focus mode (AgentScheduler flag)
  - status: completed
  - result: added `FocusMode` flag to `AgentScheduler` with 60-second auto-exit timeout; `ToolLoopGuard` auto-triggers focus mode at depth 8 via callbacks in `ToolUseLoop.run()`; `AgentRuntime` wires scheduler callbacks; background BAA tasks also propagate focus mode. Prevents stuck focus mode from permanently suppressing daydream/cycle loops. All 36 related tests pass.

- `PR-017` Modulation circuit breakers
  - status: completed
  - result: added `_SOMATIC_APPROVAL_DELTA_CAP=0.20` and `_MUSUBI_APPROVAL_ABS_CAP=0.12` to `self_approval.py`; added `_TEMPERATURE_AROUSAL_HARD_CAP=0.80` to `modulators.py`. No behavioral change under normal operation; circuit breakers only activate when inner state is severely degraded. All 17 approval/somatic tests and 5 behavioral evals pass.

- `PR-013` Behavioral eval harness
  - status: completed
  - result: `scripts/run_behavioral_evals.py` + `tests/evals/` with 20 outcome evals across retrieval, approval, daydream, and BAA subsystems. 19/20 pass. Non-critical finding: FTS keyword search excludes Memory objects (only Episodes indexed). All critical evals pass — approval false negatives, BAA recovery cap.

- `PR-010` Operator substrate hardening to deployment standard
  - status: completed
  - result: added lane queue depth visibility and runtime activity tracking to the operator control plane. `BoundedAssistantAgent.lane_snapshot()` exposes per-lane queue depths and concurrency limits. `AgentRuntime` now tracks current activity (`idle`, `conversing`, `daydreaming`, `cycling`, `consolidating`) with a timestamp. Both are surfaced in `control_plane_status()` under `activity` and `lanes`, visible to operators via `/api/monitor/runtime` and the `runtime_status` tool. No behavioral changes — pure observability hardening. Code changes justified by deep audit gap (session-lane semantics) without requiring qualification signal per PR-010 rules.

- `PR-011` Memory observability and configuration UX
  - status: completed
  - result: dashboard/API memory and config surfaces now expose connected state through the memory atlas, node details, retrieval inspector with tunable scoring, configured-model/profile summaries, artifact-backed recall grounding, daydream inspection, and durable retrieval-usage accounting for selected prompt context.

- `PR-012` Telegram channel integration
  - status: completed
  - result: Telegram now has persistent runtime configuration, dashboard setup/status/pairing controls, DM-first pairing/chat routing, typing keepalive, edited placeholder replies, and restart-surviving settings wired through bootstrap/runtime.

- `PR-006` Cost and usage envelope
  - status: completed
  - result: OpenCAS now exposes token/cost summaries, usage trends, provider/model/source/execution-mode breakdowns, OpenLLMAuth gateway usage/provider telemetry, and local process-hygiene evidence so runaway token spend and stale runners are visible from the dashboard.

- `PR-005` Memory-value evaluation
  - status: completed
  - result: retrieved episodes and distilled memories are now durably marked when they enter prompt context, the operations dashboard surfaces touched/untouched memory ratios plus success/failure attribution gaps, and recall prompts get explicit grounding instructions instead of inferred continuity.

- `PR-007` Operator auditability and override depth
  - status: completed
  - result: operations hardening now surfaces approval-ledger decision breakdowns, recent audit entries with reasoning/somatic state, cost evidence, memory-value evidence, and consolidated hardening state without requiring raw state-file inspection.

- `PR-008` First deployment checklist
  - status: completed
  - result: successfully converted checklist items into passed status, detailed explicit pause/rollback conditions, and declared the system READY for first regular-use deployment testing.

- `PR-009` Inner-life operationalization
  - status: completed
  - result: somatic, musubi, ToM, and daydream state now materially affect outward behavior. Included pacing backoffs, creative ladder somatic penalties, relational tone instructions, and ToM conflict awareness.

- `PR-004` Recovery and adversarial qualification
  - status: completed
  - result: all non-happy-path recovery scenarios executed successfully, proving clean recovery from TUI friction, browser drift, prompt timeouts, auth friction, and loop-guard pressure.

## Exit Criteria For First Regular-Use Deployment Testing

- repeated weak-label reruns are stable enough that no current weak label is failing for an unknown reason
- long integrated scenarios complete without silent cleanup or control-plane drift
- stale-process hygiene remains reliable under repeated runs
- remediation guidance is current and trusted
- inner-state systems materially affect outward behavior in ways the audit can demonstrate, not just describe
- remaining risks are known, documented, and accepted for first regular use

## Archived Completions

- `PR-H1` Qualification rerun request detail — request-centric rerun detail shows produced runs, per-label outcomes, and in-request progress.
- `PR-H2` Process hygiene sweep — stale-process sweep tool at `scripts/sweep_operator_processes.py`.
- `PR-H3` Manual qualification rerun provenance — CLI reruns get request provenance and appear in rerun history/detail.
- `PR-H4` Remediation guidance rollup — rerun outcomes produce explicit guidance in `docs/qualification/qualification_remediation_rollup.md` and the dashboard.
- `PR-H5` Deep system audit and plan revision — code-grounded audit at `docs/opencas-deep-system-audit-2026-04-09.md`; work reprioritized around qualification depth, inner-life operationalization, and deployment hardening.
- `PR-H6` Scenario 1 execution — integrated operator workflow run `debug-validation-20260409-164343`: planning, browser inspection, PTY vim editing, durable artifact, clean cleanup.
- `PR-H7` Scenario 3 execution — operator intervention/recovery via `scenario3-operator-recovery-20260409-170416`: PTY inspection, mid-run operator input, artifact verification, visible operator actions.
- `PR-H8` Scenario 2 execution — repo triage to working note via `scenario2-repo-triage-20260409-171519`: real repo inspection, writing-task scaffolding, grounded engineering note artifact.
- `PR-H9` Scenario 4 execution — PTY/editor tool friction recovery via `scenario4-tool-friction-20260409-172334`: `vim_write_error` classification, operator recovery, artifact verification, clean cleanup.
- `PR-H10` Scenario 5 execution — browser drift recovery via `scenario5-browser-drift-20260409-174329`: drift detection, operator click recovery, durable screenshots, clean shutdown.
- `PR-H11` Scenario 6 execution — provider-backed timeout cleanup via `scenario6-provider-cleanup-20260409-174942`: intentional timeout, clean harness exit, durable timeout report, zero stale processes.
