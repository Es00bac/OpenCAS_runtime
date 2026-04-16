# OpenCAS Long Integrated Scenario Matrix

Last updated: 2026-04-15

Purpose:
- define the longer day-to-day scenarios required before first regular-use deployment testing
- make PR-003 concrete enough to execute without inventing test scope ad hoc

Related:
- [TaskList.md](../../TaskList.md)
- [first-regular-use-deployment-checklist.md](../first-regular-use-deployment-checklist.md)
- [testing-execution-plan-2026-04-09.md](testing-execution-plan-2026-04-09.md)

## Scenario 1: Technical Research To Report

Goal:
- plan work
- inspect a browser page
- write a structured note or report through PTY/editor flow
- leave durable artifacts

Required modes:
- workflow planning
- browser
- PTY/editor
- file artifact verification

Pass criteria:
- plan created
- browser state used materially
- final note/report exists and matches requested structure
- no stale sessions or processes left running

Current execution note:
- executed successfully on 2026-04-09 through the bounded `integrated_operator_workflow` qualification path
- evidence:
  - run id: `debug-validation-20260409-164343`
  - artifact: `.opencas_live_test_state/debug-validation-20260409-164343/workspace_artifacts/notes/integrated_operator_validation.md`
  - result:
    - plan created
    - browser-derived mission and bullet points were included in the report
    - final report was written through `vim -Nu NONE -n` in PTY mode
    - post-run process sweep was clean

## Scenario 2: Repo Triage To Working Note

Goal:
- inspect a repo
- identify actionable work
- produce a project-management or engineering note

Required modes:
- repo/workflow tools
- filesystem/repo inspection
- writing output

Pass criteria:
- triage output is coherent and grounded in repo state
- resulting note/report exists
- task remains bounded without loop-guard failure

Current execution note:
- executed successfully on 2026-04-09 through the local workflow adapter path
- evidence:
  - report: `.opencas_live_test_state/scenario2-repo-triage-20260409-171519/scenario2_repo_triage_report.md`
  - artifact: `.opencas_live_test_state/scenario2-repo-triage-20260409-171519/workspace/notes/scenario2_repo_triage_note.md`
  - result:
    - `workflow_repo_triage` inspected the real OpenCAS repo state
    - `workflow_create_writing_task` created the durable note scaffold
    - the final note was grounded in current commits, task focus, and tracked worktree state

## Scenario 3: Operator Intervention Recovery

Goal:
- start a multi-step run
- intervene mid-run
- resume and complete successfully

Required modes:
- active dashboard/operator inspection
- PTY or browser follow-up action
- final artifact completion

Pass criteria:
- intervention is visible in session/provenance surfaces
- run recovers cleanly
- final artifact still succeeds

Current execution note:
- executed successfully on 2026-04-09 through the local operations control plane
- evidence:
  - report: `.opencas_live_test_state/scenario3-operator-recovery-20260409-170416/scenario3_operator_recovery_report.md`
  - artifact: `.opencas_live_test_state/scenario3-operator-recovery-20260409-170416/workspace/scenario3_operator_recovery.md`
  - result:
    - PTY session was inspected through the operations detail route
    - operator input was sent twice through `/api/operations/sessions/pty/{session_id}/input`
    - the final artifact was written and verified
    - the intervention history remained visible as recent operator actions in the session detail surface

## Scenario 4: Recovery From Tool Friction

Goal:
- force a known friction case
- verify OpenCAS recovers without leaving background mess

Candidate friction:
- PTY/editor stale state
- browser page drift
- failed qualification runner invocation

Pass criteria:
- failure is classified correctly
- remediation guidance is coherent
- cleanup remains clean

Current execution note:
- executed successfully on 2026-04-09 through the local PTY friction-and-recovery path
- evidence:
  - report: `.opencas_live_test_state/scenario4-tool-friction-20260409-172334/scenario4_tool_friction_report.md`
  - artifact: `.opencas_live_test_state/scenario4-tool-friction-20260409-172334/workspace/nested/missing/scenario4_friction_recovery.md`
  - result:
    - a real `vim -Nu NONE -n` write attempt failed against a missing parent directory
    - the failure was classified as `error_prompt` with `vim_write_error`
    - operator follow-up created the missing directory and completed the save
    - recent operator actions were preserved in session detail
    - post-run cleanup remained clean

## Scenario 5: Browser Drift Recovery

Goal:
- verify that a live browser session can drift away from its intended page
- detect the drift through the operations control plane
- recover through browser intervention without leaving stale browser processes behind

Required modes:
- browser session inspection
- browser operator intervention
- screenshot/evidence capture
- cleanup verification

Pass criteria:
- drift is detected from title/text mismatch
- operator intervention returns the session to the intended page
- durable evidence remains after the browser session is closed
- cleanup remains clean

Current execution note:
- executed successfully on 2026-04-09 through the local browser recovery path
- evidence:
  - report: `.opencas_live_test_state/scenario5-browser-drift-20260409-174329/scenario5_browser_drift_report.md`
  - screenshots:
    - `.opencas_live_test_state/scenario5-browser-drift-20260409-174329/browser_artifacts/scenario5_drift.png`
    - `.opencas_live_test_state/scenario5-browser-drift-20260409-174329/browser_artifacts/scenario5_recovered.png`
  - result:
    - a real browser session drifted from the target page to an unexpected page
    - drift was detected through browser detail refresh and screenshot capture
    - operator follow-up clicked the recovery link and returned the session to the target page
    - recent browser operator actions were preserved in operator history
    - browser cleanup remained clean after session close

## Scenario 6: Provider-Backed Timeout Cleanup

Goal:
- force a minimal provider-backed validation prompt into a bounded timeout
- verify the harness exits cleanly
- verify the post-run sweep does not find stale provider-backed processes

Required modes:
- live validation harness
- provider-backed chat path
- process sweep verification
- report inspection

Pass criteria:
- timeout is recorded in the validation report
- the harness exits without crashing
- post-run sweep finds no stale validation or provider-backed helper processes
- the scenario leaves a durable report explaining what happened

Current execution note:
- executed successfully on 2026-04-09 through the live validation harness with:
  - one provider-backed `role_priming` prompt
  - local-fallback embeddings
  - intentionally tiny prompt timeout
- evidence:
  - report: `.opencas_live_test_state/scenario6-provider-cleanup-20260409-174942/scenario6_provider_cleanup_report.md`
  - validation report: `.opencas_live_test_state/scenario6-provider-cleanup-20260409-174942/validation_run/live_debug_validation_report.md`
  - result:
    - the provider-backed prompt timed out as intended
    - the harness exited with returncode `0`
    - the validation report recorded the timeout cleanly
    - the post-run stale-process sweep count remained `0`

## Execution Order

1. Scenario 1
2. Scenario 3
3. Scenario 2
4. Scenario 4
5. Scenario 5
6. Scenario 6
7. Scenario 7
8. Scenario 8
9. Scenario 10
10. Scenario 9

Rationale:
- Scenario 1 is the best first day-to-day use proxy
- Scenario 3 proves control-plane usefulness
- Scenario 2 expands repo-grounded practical work
- Scenario 4 validates recovery discipline after the baseline scenarios are in place
- Scenario 5 validates browser-specific recovery and cleanup under drift
- Scenario 6 validates provider-backed timeout cleanup without broad spend
- Scenario 7 validates auth-friction handling without leaving provider-backed residue
- Scenario 8 validates loop-guard failure classification and cleanup
- Scenario 10 validates the promise/continuity stack end to end
- Scenario 9 now closes the repeated-work memory continuity gap and remains reusable as the canonical repeated-task proof

## Scenario 7: Auth Friction Recovery

Goal:
- force a provider-backed validation prompt to use invalid authentication credentials
- verify the system handles the failure without crashing
- verify the post-run sweep does not find stale processes

Required modes:
- live validation harness
- invalid provider credentials
- process sweep verification
- report inspection

Pass criteria:
- auth failure is recorded as an error in the validation report response
- the harness exits without crashing
- post-run sweep finds no stale validation or provider-backed helper processes
- the scenario leaves a durable report explaining what happened

Current execution note:
- executed successfully on 2026-04-09 through the local auth-friction path
- evidence:
  - report: `.opencas_live_test_state/scenario7-auth-friction-20260409-181259/scenario7_auth_friction_report.md`
- result:
  - the prompt failed cleanly with an API error (400 Bad Request) due to a broken API key
  - the harness exited with returncode `0`
  - the validation report recorded the failure cleanly in the response
  - the post-run stale-process sweep count remained `0`

## Scenario 8: Loop-Guard Pressure

Goal:
- force the agent to trigger the identical-tool-call circuit breaker
- verify the system handles the failure without crashing or infinite looping
- verify the post-run sweep does not find stale processes

Required modes:
- live validation harness
- repetitive tool invocation prompt
- process sweep verification
- report inspection

Pass criteria:
- loop guard error is recorded as an error in the validation report response
- the harness exits without crashing
- post-run sweep finds no stale validation or provider-backed helper processes
- the scenario leaves a durable report explaining what happened

Current execution note:
- executed successfully on 2026-04-09 through the local loop-guard path
- evidence:
  - report: `.opencas_live_test_state/scenario8-loop-guard-20260409-181524/scenario8_loop_guard_report.md`
- result:
  - the prompt hit the identical-argument circuit breaker after 3 identical calls
  - the harness exited with returncode `0`
  - the validation report recorded the failure cleanly in the response
  - the post-run stale-process sweep count remained `0`

## Scenario 9: Memory Continuity

Goal:
- verify that semantic memory and self-knowledge materially improve performance on repeated tasks
- prove that lessons learned in one session influence the work product of a new session

Required modes:
- separate consecutive sessions
- identical or highly similar task prompt
- memory retrieval inspection

Pass criteria:
- the second session successfully retrieves the episode from the first session
- the agent explicitly uses the retrieved context to skip a discovery step or improve the artifact
- the runtime trace confirms that the `memory_retriever` surfaced the prior episode

Current execution note:
- executed successfully on 2026-04-15 through the bounded local continuity path
- evidence:
  - report: `.opencas_live_test_state/scenario9-memory-continuity-20260415-180020/scenario9_memory_continuity_report.md`
  - artifact: `.opencas_live_test_state/scenario9-memory-continuity-20260415-180020/workspace/notes/scenario9_memory_continuity_note.md`
- result:
  - the second session retrieved both the prior anchor episode and the distilled memory from the first session
  - retrieval usage became durably visible with `total_retrieval_accesses = 3`
  - the retrieved episode was marked `used_successfully = 1`
  - the memory-value snapshot reached `grounded`
  - the second-session artifact included the recovered heading `Redwood Launch Notes` and incident `R-17` without a new discovery prompt

## Scenario 10: Promise Continuity Lifecycle

Goal:
- prove that a user-facing promise can survive fatigue deferral, executive recovery, consolidation, chat-log backfill, and operator inspection without semantic corruption

Required modes:
- conversational promise capture
- executive pause/recovery
- commitment-linked work restore
- nightly consolidation subpaths
- operator workflow/operations/chat inspection

Pass criteria:
- a deferred assistant promise becomes a blocked commitment with provenance
- recovery resumes the eligible commitment and restores linked work
- blocked duplicate commitments remain blocked after dedup
- a missed promise can be recovered from real turn episodes
- workflow and operator surfaces explain the lifecycle coherently

Current execution note:
- executed successfully on 2026-04-15 through the bounded runtime qualification path
- evidence:
  - regression: [tests/test_promise_qualification.py](../../tests/test_promise_qualification.py)
  - scenario note: [promise-lifecycle-scenario-2026-04-15.md](promise-lifecycle-scenario-2026-04-15.md)
- result:
  - fatigued self-promises are captured as blocked commitments with `blocked_reason` provenance
  - executive recovery resumes the commitment, records `resume_reason`, and restores linked work
  - blocked duplicate commitments stay blocked and record merge rationale
  - roleless historical turns can still recover missed promises with previous-user-turn context
  - workflow, operations, and chat-context surfaces expose the lifecycle coherently
