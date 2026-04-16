# Promise Continuity Scenario

Date:
- 2026-04-15

Purpose:
- prove the current promise-continuity stack as one bounded lifecycle instead of treating pause/resume, consolidation, chat recovery, and observability as unrelated regressions

Scope:
- synthetic bounded runtime qualification
- local stores only
- no browser, PTY, or live provider dependency

Scenario:
1. A user asks for future work in chat.
2. The assistant responds with a deferred promise while the executive is fatigued.
3. The promise becomes a blocked self-commitment with explicit provenance.
4. Recovery clears the executive pause and restores linked work.
5. Nightly commitment dedup merges blocked duplicates without reactivating them.
6. A missed promise in conversation logs is recovered from real turn episodes, including roleless historical turns.
7. Operator surfaces explain the lifecycle through workflow, operations, and chat-context summaries.

Primary evidence:
- regression: [tests/test_promise_qualification.py](../../tests/test_promise_qualification.py)
- key proof: `test_promise_lifecycle_qualification_scenario`

What the scenario proves:
- blocked self-commitments preserve `blocked_reason` provenance when created under executive fatigue
- executive recovery resumes only the eligible commitment and records `resume_reason` / `previous_blocked_reason`
- linked work is restored to the executive queue when the commitment becomes active again
- commitment dedup preserves blocked status and records merge rationale instead of silently reactivating blocked duplicates
- chat-log extraction can recover missed promises from the real episode model, including roleless assistant turns with previous-user-turn context
- operator surfaces can explain why a commitment is active, blocked, resumed, merged, or recovered from chat

Support fixes surfaced while executing the scenario:
- Python 3.14 local runtime needed an `aiosqlite` compatibility shim because fresh store bootstrap was hanging during connection setup
- bootstrap had a duplicated `workspace_index` startup block that leaked a background worker and interfered with clean qualification teardown
- chat-log recovery now falls back to content-based candidate matching when LLM candidate ids drift from prompt ordering, so recovered commitments keep the right session and `previous_user_turn` provenance

Known boundary:
- this is a bounded qualification scenario, not a live end-user session
- it proves the lifecycle contract and operator visibility, but it does not replace future live qualification for longer autonomous runs
