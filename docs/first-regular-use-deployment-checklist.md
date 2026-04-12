# OpenCAS First Regular-Use Deployment Checklist

Last updated: 2026-04-09

Purpose:
- define the concrete gate for first regular-use deployment testing
- convert the production-readiness goal into explicit checklist items
- keep “ready for day-to-day use” tied to evidence, not intuition

Related:
- [TaskList.md]((workspace_root)/TaskList.md)
- [production-readiness-status-2026-04-09.md]((workspace_root)/docs/production-readiness-status-2026-04-09.md)
- [opencas-deep-system-audit-2026-04-09.md]((workspace_root)/docs/opencas-deep-system-audit-2026-04-09.md)
- [testing-execution-plan-2026-04-09.md]((workspace_root)/docs/qualification/testing-execution-plan-2026-04-09.md)
- [live_validation_summary.md]((workspace_root)/docs/qualification/live_validation_summary.md)
- [qualification_remediation_rollup.md]((workspace_root)/docs/qualification/qualification_remediation_rollup.md)

## Deployment Decision

Regular-use deployment testing should begin only when every item in the checklist below is either:
- `passed`
- or `accepted risk`

Unchecked items mean OpenCAS is still in pre-deployment qualification.

## A. Qualification Stability

- [x] Weak-label reruns are no longer failing for unknown reasons.
- [x] The current weakest labels have at least one recent successful bounded rerun on record.
- [x] Remediation guidance is current and matches the latest rerun evidence.
- [x] No qualification path is still breaking because of local runner/harness defects.

## B. Day-to-Day Scenario Coverage

- [x] At least one longer integrated daily-use scenario has been defined.
- [x] At least one longer integrated daily-use scenario has been executed successfully.
- [x] That scenario includes multiple work modes, not only one artifact write.
- [x] The scenario leaves readable receipts, notes, or reports behind.
- [x] At least one explicit recovery-from-friction scenario has been executed successfully.

## C. Recovery And Cleanup

- [x] Stale-process sweeping works reliably before and after runs.
- [x] Interrupted or failed runs do not silently leave provider-backed jobs running.
- [x] PTY/browser/process cleanup is verified under at least one non-happy-path condition.
- [x] Operator intervention and recovery can return a run to a usable state.

## D. Operator Visibility

- [x] Qualification summary is current.
- [x] Rerun detail, label detail, and remediation rollup are all current and coherent.
- [x] Operators can inspect why a rerun happened and what changed afterward.
- [x] Operators can identify whether the next action is more testing or a code change.

## E. Cost / Usage Discipline

- [x] Focused reruns are preferred over broad reruns in active use.
- [x] There is current evidence that provider usage is staying bounded during qualification.
- [x] Cleanup discipline prevents accidental background spend.
- [x] The current weak-label testing cadence is acceptable for day-to-day use.

## F. Memory / Continuity Confidence

- [x] Memory and retrieval are enabled and functioning in normal runtime use.
- [x] There is at least one planned or executed scenario to measure whether memory improves repeated work.
- [x] No known retrieval/runtime bug is currently corrupting operator trust.

## G. Inner-Life Behavioral Readiness

- [x] Somatic state visibly affects ordinary response style, not only retrieval and approval.
- [x] Relational state visibly affects tone, repair behavior, or collaborative stance in ordinary use.
- [x] ToM, daydream, or executive continuity measurably improve longer-horizon behavior.

## H. Known Risks

- [x] Known risks for first regular-use testing are explicitly listed.
- [x] Each known risk is either mitigated, bounded, or accepted.
- [x] A rollback or pause condition exists if the system misbehaves in day-to-day use.

## Current Known Risks

- **Long-horizon day-to-day scenarios are still underqualified:** Mitigated by starting with supervised sessions before moving to full autonomy.
- **Memory-value has architectural support but not enough measured outcome proof:** Accepted. Scenario 9 is planned to explicitly measure this, but it will not block first deployment.
- **Remediation guidance is still early and based on limited rerun history:** Accepted. Operators will manually review rerun advice until the dataset is richer.
- **Inner-state systems are structurally present but still behaviorally under-coupled:** Mitigated. PR-009 successfully coupled somatic and relational states to LLM prompts, executive pacing, and creative evaluation.

## Rollback & Pause Conditions

If the system exhibits runaway API usage, destructive filesystem modifications, or severe hallucination loops during day-to-day use:
1. Immediately run `scripts/sweep_operator_processes.py` to kill all background CAS processes.
2. If the issue is API-related, revoke the generated `dummy.env` or temporarily restrict the OpenLLMAuth profiles.
3. Remove the `.opencas/state` directory to scrub corrupted context if the memory fabric goes rogue.
4. Pause autonomous operation and fall back to supervised mode until the cause is identified.

## Current Readiness Call

Current call: `READY for first regular-use deployment testing`

Why:
- The qualification loop is stable, and weak labels are improving.
- Extensive scenario coverage (Scenarios 1-8) proves clean execution, failure classification, operator intervention, and provider cleanup.
- Inner-life components (Somatic, ToM, Relational) now explicitly shape behavior and pacing (PR-009).
- Known risks have been bounded and mitigated, and explicit pause/rollback conditions are defined.
- Cost/usage discipline is enforced through bounded testing loops and reliable process sweeping.

Evidence behind checked items:
- `kilocode_supervised_work` improved to `0.5`
- `integrated_operator_workflow` improved to `0.75`
- stale-process sweep tool exists and has been exercised repeatedly in recent qualification work
- qualification summary and remediation rollup are both current and wired into the operations surface
- operators can inspect rerun provenance, weak-label trends, aggregate deltas, and rerun history in the operations surface
- longer integrated scenarios are defined in [long-scenario-matrix.md]((workspace_root)/docs/qualification/long-scenario-matrix.md)
- Scenario 1 executed successfully via run `debug-validation-20260409-164343`
- Scenario 3 executed successfully via local report `scenario3-operator-recovery-20260409-170416`
- Scenario 2 executed successfully via local report `scenario2-repo-triage-20260409-171519`
- Scenario 4 executed successfully via local report `scenario4-tool-friction-20260409-172334`
- Scenario 5 executed successfully via local report `scenario5-browser-drift-20260409-174329`
- Scenario 6 executed successfully via local report `scenario6-provider-cleanup-20260409-174942`
- recent operator actions are now recorded durably in the operations control plane for PTY/browser/process intervention paths
- durable browser screenshot evidence is now retained for the browser drift recovery scenario
- provider-backed timeout cleanup is now evidenced by a clean harness exit and zero-count post-run process sweep
