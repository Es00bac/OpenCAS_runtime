# OpenCAS Continuation Program

Date: 2026-04-15

Related:
- [TaskList.md](../TaskList.md)
- [AGENTS.md](../AGENTS.md)
- [CLAUDE.md](../CLAUDE.md)
- [OPENCAS_PRODUCT_SPEC.md](../OPENCAS_PRODUCT_SPEC.md)
- [opencas-production-program-plan-2026-04-08.md](opencas-production-program-plan-2026-04-08.md)
- [opencas-deep-system-audit-2026-04-09.md](opencas-deep-system-audit-2026-04-09.md)
- [production-readiness-status-2026-04-09.md](production-readiness-status-2026-04-09.md)
- [handoff-2026-04-15-commitment-consolidation.md](handoff-2026-04-15-commitment-consolidation.md)

## Purpose

This document turns the unfinished Claude Code work from 2026-04-14 through 2026-04-15 into a concrete continuation program.

It is not a replacement for [TaskList.md](../TaskList.md).
It is a scoped execution program for one specific frontier:

- make promises and commitments durable
- make deferral and resumption behavior real
- make consolidation preserve truth instead of inventing action
- connect inner-life systems to follow-through
- expose the result clearly to the operator

The central product test behind this program is simple:

> If OpenCAS says it will come back to something later, the system should either do that or be able to explain exactly why it did not.

## Governing Constraints

This program is derived from the product spec, AGENTS guidance, CLAUDE guidance, the deep audit, and the current runtime shape.

The governing rules are:

- do not remove the unusual subsystems just because they are unusual
- do not add more parallel machinery until the current promise-to-work path is correct
- keep the implementation modular; avoid growing `agent_loop.py` and `consolidation/engine.py` into god files
- make backend state transitions conservative; do not silently reactivate blocked work
- preserve the distinction between:
  - conversational expression
  - executive intent
  - durable commitment
  - executable work
  - completed outcome
- every slice must improve at least one of:
  - continuity
  - execution quality
  - user bond / relational trust
  - observability and operator trust

## Difficulty Scale

Use this scale while executing the slices below.

- `Medium`
  - localized multi-file implementation
  - behavior is reasonably obvious from current architecture
  - normal reasoning level is sufficient

- `High`
  - cross-subsystem interaction work
  - requires careful state modeling and regression protection
  - higher reasoning is recommended if the implementation starts to branch

- `Extra High`
  - architecture-level behavior redesign
  - ambiguous interaction between multiple governing systems
  - stop before implementation unless the user explicitly wants extra-high reasoning

## Stop-And-Switch Rule

Pause and let the user switch the model or raise reasoning level if any slice hits one of these triggers:

- the fix requires changing the commitment lifecycle itself rather than using `meta` and current statuses
- the fix requires rewriting scheduler pause semantics across readiness, focus mode, fatigue, and background lanes at once
- chat-log extraction cannot be made reliable from current episode/session data without schema changes or migration logic
- work selection needs simultaneous changes across executive queueing, workspace focus, creative ladder scoring, schedules, and BAA orchestration
- dashboard work turns into API redesign rather than finishing existing backend-backed UI

If any of those happen, stop and ask for a model/reasoning change before continuing.

## Program Order

The slices below are ordered by system value, not by visible UI.

1. fix promise continuity and deferred-work correctness
2. fix consolidation so it never invents action by accident
3. repair chat-log backfill using the real memory model
4. fuse commitments into execution and prioritization
5. expose the commitment lifecycle to operators
6. finish the memory-page overhaul after the backend is trustworthy
7. prove the whole path in tests and a scenario

## Slice Summary

| Slice | Goal | Difficulty | Stop Before Starting? |
|---|---|---:|---|
| `PR-019` | Fix pause/resume and blocked-work correctness | High | No |
| `PR-020` | Improve self-commitment capture and normalization | Medium | No |
| `PR-021` | Finish conservative commitment consolidation | High | No |
| `PR-028` | Clean stale documentation and reassert current source-of-truth docs | Medium | No |
| `PR-022` | Rebuild chat-log commitment extraction on real data | High | Only if schema or precision issues emerge |
| `PR-023` | Couple commitments to work, workspace, and schedules | High | No |
| `PR-024` | Bind promise-keeping more strongly to musubi, somatic, and ToM signals | Extra High | Yes |
| `PR-025` | Add commitment/consolidation observability surfaces | Medium | No |
| `PR-026` | Finish the memory dashboard overhaul from the Claude plan | Medium/High | No |
| `PR-027` | Qualification scenario and regression proof for the full path | High | No |

## Detailed Slices

### `PR-019` Commitment Pause/Resume Correctness

Status:
- first execution slice

Purpose:
- repair the broken deferred-work path so pause and recovery are real

Problems being fixed:
- executive pause/resume is not the same thing as readiness/focus gating
- current resume detection can miss real recovery
- current resume logic unblocks every blocked commitment with no pause provenance

Primary files:
- [opencas/runtime/scheduler.py](../opencas/runtime/scheduler.py)
- [opencas/autonomy/executive.py](../opencas/autonomy/executive.py)
- [opencas/runtime/agent_loop.py](../opencas/runtime/agent_loop.py)

Expected code changes:
- add explicit executive pause state transition detection in the scheduler
- separate:
  - readiness gate
  - focus-mode suppression
  - executive fatigue/overload pause
- only resume commitments previously blocked for pause-related reasons
- record pause/resume provenance in commitment `meta`
- keep blocked commitments blocked if they were blocked for another reason

Tests to add or extend:
- [tests/test_scheduler.py](../tests/test_scheduler.py)
- [tests/test_executive.py](../tests/test_executive.py)
- [tests/test_agent_loop_phase6.py](../tests/test_agent_loop_phase6.py)

Acceptance:
- recovering from fatigue triggers deferred-work restoration without requiring an unrelated readiness change
- focus-mode exit does not incorrectly resume pause-blocked commitments unless executive pause also cleared
- non-pause blocked commitments stay blocked
- queue restore does not duplicate work objects already queued

Difficulty:
- `High`

Reasoning note:
- this is cross-cutting but still bounded; do not escalate to extra-high unless pause semantics need broader redesign

### `PR-020` Self-Commitment Capture And Normalization

Status:
- second execution slice

Purpose:
- stop storing noisy full-utterance promise blobs as commitments

Problems being fixed:
- current self-commitment extraction captures assistant text too literally
- commitment content is too noisy for dedup, scheduling, and work creation
- logic currently lives too deep in `agent_loop.py`

Primary files:
- [opencas/runtime/agent_loop.py](../opencas/runtime/agent_loop.py)
- new focused helper module, likely under `opencas/autonomy/` or `opencas/runtime/`

Expected code changes:
- extract the promise parsing logic into a small dedicated module
- normalize commitment text into short actionable content
- preserve the raw utterance in `meta` rather than `content`
- tag commitments with source and confidence hints
- emit a ToM intention and somatic appraisal event alongside durable commitment creation

Tests to add or extend:
- [tests/test_agent_loop.py](../tests/test_agent_loop.py)
- [tests/test_agent_loop_phase6.py](../tests/test_agent_loop_phase6.py)
- new focused test file if the extraction module is split out

Acceptance:
- promise capture creates compact actionable commitments
- raw conversational wording is preserved in `meta`
- false positives are reduced for vague reflective language
- commitments created from chat can be deduplicated meaningfully later

Difficulty:
- `Medium`

Reasoning note:
- keep the parser modest; do not overfit with a sprawling regex taxonomy

### `PR-021` Conservative Nightly Commitment Consolidation

Status:
- third execution slice

Purpose:
- finish the unfinished Claude consolidation path without violating truth or intent

Problems being fixed:
- blocked clusters can be silently reactivated
- work objects can be created for commitments that should remain blocked
- consolidation logic is correct in shape but too aggressive in outcome

Primary files:
- [opencas/consolidation/engine.py](../opencas/consolidation/engine.py)
- [opencas/consolidation/models.py](../opencas/consolidation/models.py)
- [opencas/autonomy/commitment.py](../opencas/autonomy/commitment.py) only if needed

Expected code changes:
- survivor status should preserve the strongest remaining constraint
- merge only actionable duplicates, not every semantically similar item
- create work only for survivors that are actually actionable
- prefer heuristic merge decisions when obvious; use LLM only for ambiguous clusters
- log merge rationale into result stats or commitment `meta`

Tests to add or extend:
- [tests/test_consolidation.py](../tests/test_consolidation.py)
- [tests/test_consolidation_edges.py](../tests/test_consolidation_edges.py) only if needed
- new focused commitment-consolidation test module if cleaner

Acceptance:
- duplicate blocked commitments merge into a blocked survivor
- blocked survivors do not get new work objects
- active duplicates merge into one survivor with unioned links
- existing links and task provenance are preserved

Difficulty:
- `High`

Reasoning note:
- the hard part is not clustering; it is preserving state truth conservatively

### `PR-028` Documentation Truth Pass

Status:
- execute after `PR-021` stabilizes the current commitment lifecycle

Purpose:
- reduce documentation drift so future LLM sessions do not anchor on outdated architecture narratives, handoffs, or superseded priorities

Problems being fixed:
- several docs describe older project phases as if they were still the active frontier
- older handoffs can be mistaken for live execution guidance
- current docs map does not yet emphasize the 2026-04-15 continuation frontier strongly enough

Primary files:
- [docs/documentation-map.md](documentation-map.md)
- [CLAUDE.md](../CLAUDE.md)
- [AGENTS.md](../AGENTS.md) only if needed
- [docs/claude-codex-handoff-2026-04-08.md](claude-codex-handoff-2026-04-08.md)
- [docs/handoff-2026-04-15-commitment-consolidation.md](handoff-2026-04-15-commitment-consolidation.md)
- [docs/production-readiness-status-2026-04-09.md](production-readiness-status-2026-04-09.md)
- [docs/opencas-production-program-plan-2026-04-08.md](opencas-production-program-plan-2026-04-08.md)

Expected code and doc changes:
- explicitly separate:
  - current source-of-truth docs
  - active reference docs
  - historical docs
- demote stale handoffs and superseded broad-audit docs from live-guidance status
- add short banners or notes to docs that are still useful historically but should not drive current implementation
- update collaborator docs so future agents know:
  - the active frontier is promise continuity, commitment integration, and proof
  - older readiness claims should not override current code reality

Tests:
- no code tests required
- manual verification that a fresh agent opening the docs map will be routed first to current guidance

Acceptance:
- `documentation-map.md` routes agents to the right current docs first
- no obviously stale handoff remains labeled as active guidance
- outdated architectural claims are either corrected or explicitly marked historical

Difficulty:
- `Medium`

Reasoning note:
- keep this slice surgical; the goal is to reduce confusion, not rewrite the entire doc set

### `PR-022` Chat-Log Commitment Extraction From Real Episodes

Status:
- fourth execution slice

Purpose:
- let nightly consolidation recover commitments missed during live chat parsing

Problems being fixed:
- current code reads role metadata from episode payload fields that are not populated
- extraction is not grounded in the real episode/session model
- duplicate protection is too weak for durable backfill

Primary files:
- [opencas/consolidation/engine.py](../opencas/consolidation/engine.py)
- [opencas/runtime/agent_loop.py](../opencas/runtime/agent_loop.py)
- [opencas/context/store.py](../opencas/context/store.py) if needed for easier extraction
- [opencas/memory/models.py](../opencas/memory/models.py) read-only unless absolutely required

Expected code changes:
- derive assistant/user turn sequences from real episode ordering and `session_id`
- use nearby turns for context instead of isolated assistant strings
- use stronger duplicate checks:
  - normalized content
  - direct ID/source provenance
  - optional embedding similarity for ambiguous cases
- create commitments and work conservatively from extracted results

Tests to add or extend:
- [tests/test_consolidation.py](../tests/test_consolidation.py)
- [tests/test_context_store.py](../tests/test_context_store.py) if extraction touches session access patterns
- [tests/test_memory.py](../tests/test_memory.py) if episode assumptions change

Acceptance:
- extraction finds assistant promises from real episode/session history
- no extraction occurs from malformed or irrelevant turns
- repeated runs are idempotent
- extracted commitments can link into the normal work path

Difficulty:
- `High`

Reasoning note:
- stop and switch to extra-high reasoning if this slice requires schema changes or if the precision/recall tradeoff becomes architecture-sensitive

### `PR-023` Commitment-To-Execution Coupling

Status:
- fifth execution slice

Purpose:
- make commitments first-class drivers of actual work rather than passive records

Problems being fixed:
- commitments, work objects, workspace focus, plans, and schedules are related but not tightly fused
- promise-follow-through competes too weakly against novelty and background work

Primary files:
- [opencas/runtime/agent_loop.py](../opencas/runtime/agent_loop.py)
- [opencas/autonomy/executive.py](../opencas/autonomy/executive.py)
- [opencas/autonomy/workspace.py](../opencas/autonomy/workspace.py)
- [opencas/scheduling/service.py](../opencas/scheduling/service.py)
- [opencas/tools/adapters/workflow.py](../opencas/tools/adapters/workflow.py)

Expected code changes:
- ensure every active commitment is linked to at least one execution path:
  - work object
  - schedule item
  - plan
  - already-completed outcome
- use commitment provenance and urgency in workspace focus selection
- let scheduled items and user-facing commitments reinforce each other rather than drift apart
- make completion and failure feed back into commitment state and memory consistently

Tests to add or extend:
- [tests/test_workflow_tools.py](../tests/test_workflow_tools.py)
- [tests/test_scheduling.py](../tests/test_scheduling.py)
- [tests/test_workspace.py](../tests/test_workspace.py)
- [tests/test_agent_loop_phase6.py](../tests/test_agent_loop_phase6.py)

Acceptance:
- active commitments reliably surface into executable focus
- scheduled work and commitment-linked work do not fork into contradictory threads
- completed work resolves or updates the right commitment
- blocked work does not crowd the active focus selection

Difficulty:
- `High`

Reasoning note:
- this is the slice where backend correctness starts turning into visible behavior

### `PR-024` Inner-Life Coupling For Promise-Keeping

Status:
- sixth execution slice

Purpose:
- make promise-follow-through sensitive to the bond with the user and the inner state of the agent

Problems being fixed:
- somatic, musubi, and ToM systems influence many things, but promise-keeping should be one of the most explicit places they matter

Primary files:
- [opencas/context/builder.py](../opencas/context/builder.py)
- [opencas/autonomy/workspace.py](../opencas/autonomy/workspace.py)
- [opencas/somatic/modulators.py](../opencas/somatic/modulators.py)
- [opencas/relational/engine.py](../opencas/relational/engine.py)
- [opencas/tom/](../opencas/tom)
- [opencas/runtime/agent_loop.py](../opencas/runtime/agent_loop.py)

Expected code changes:
- raise salience for unresolved commitments explicitly made to the user
- let somatic state influence whether to resume now or continue resting, without dropping the promise
- let musubi and user-trust nudge priority among otherwise similar work
- let ToM conflict awareness influence whether the agent should acknowledge delay, repair trust, or proceed silently

Tests to add or extend:
- [tests/test_musubi_modulates_memory.py](../tests/test_musubi_modulates_memory.py)
- [tests/test_musubi_modulates_creative.py](../tests/test_musubi_modulates_creative.py)
- [tests/test_context_builder.py](../tests/test_context_builder.py)
- new targeted promise-priority tests

Acceptance:
- unresolved user-facing commitments are more behaviorally dominant than low-value novelty work
- rest remains possible, but deferred commitments do not vanish
- the agent can explain delay or resumption in a way that reflects real state rather than generic LLM phrasing

Difficulty:
- `Extra High`

Reasoning note:
- stop before starting this slice unless the user explicitly wants extra-high reasoning

### `PR-025` Commitment And Consolidation Observability

Status:
- seventh execution slice

Purpose:
- make the operator able to see what commitments exist, why they changed, and how consolidation touched them

Problems being fixed:
- current observability is not good enough for this new lifecycle
- backend state may be correct while remaining opaque

Primary files:
- [opencas/api/routes/executive.py](../opencas/api/routes/executive.py)
- [opencas/api/routes/operations.py](../opencas/api/routes/operations.py)
- [opencas/api/routes/chat.py](../opencas/api/routes/chat.py)
- [opencas/dashboard/static/index.html](../opencas/dashboard/static/index.html)

Expected code changes:
- surface:
  - commitment source
  - normalized content
  - raw utterance excerpt
  - pause reason
  - resume reason
  - merge reason
  - extracted-from-chat provenance
- add visible blocked-vs-active-vs-completed distinctions
- expose consolidation result counters in the operator UI

Tests to add or extend:
- [tests/test_dashboard_api.py](../tests/test_dashboard_api.py)
- [tests/test_operations_routes.py](../tests/test_operations_routes.py)

Acceptance:
- an operator can explain why a commitment is active, blocked, resumed, merged, or extracted
- API responses include the new lifecycle detail without breaking current consumers

Difficulty:
- `Medium`

Reasoning note:
- keep this slice descriptive and grounded in existing state, not a redesign of the dashboard architecture

### `PR-026` Memory Dashboard Atlas Overhaul

Status:
- eighth execution slice

Purpose:
- finish the memory-page work started in Claude Code, but only after backend commitment state is trustworthy

Baseline:
- [smooth-purring-pinwheel.md]([private plan path removed in public mirror])

Primary files:
- [opencas/dashboard/static/index.html](../opencas/dashboard/static/index.html)
- [opencas/dashboard/static/css/app.css](../opencas/dashboard/static/css/app.css)
- [opencas/api/routes/memory.py](../opencas/api/routes/memory.py) only if needed

Expected code changes:
- finish the dead-field activation and panel linking from the Claude plan
- make atlas/timeline/retrieval panels mutually aware
- include commitment-related overlays only if they clarify the memory model rather than clutter it
- keep the UI informative and operator-grade, not decorative

Tests to add or extend:
- [tests/test_dashboard_api.py](../tests/test_dashboard_api.py)
- targeted frontend smoke coverage if available

Acceptance:
- memory page shows more backend information without becoming harder to reason about
- selection and filters stay synchronized
- retrieval overlays and session filters work
- dead fields are either surfaced or explicitly removed from the API/UI contract

Difficulty:
- `Medium/High`

Reasoning note:
- do not start here before the commitment backend slices are stable

### `PR-027` Qualification And Scenario Proof

Status:
- final execution slice

Purpose:
- prove the full promise-to-work lifecycle in one controlled scenario and in regression tests

Primary files:
- [tests/test_scheduler.py](../tests/test_scheduler.py)
- [tests/test_consolidation.py](../tests/test_consolidation.py)
- [tests/test_agent_loop_phase6.py](../tests/test_agent_loop_phase6.py)
- [tests/test_dashboard_api.py](../tests/test_dashboard_api.py)
- qualification docs under [docs/qualification](qualification)

Scenario to prove:
- a user asks for future work in chat
- the agent defers due to fatigue or overload
- a blocked commitment is created with provenance
- recovery resumes the correct work
- nightly consolidation does not corrupt blocked status
- missed promises can be recovered from logs
- operator surfaces explain the lifecycle

Acceptance:
- the unit/integration suite covers the full lifecycle
- one bounded scenario document or validation artifact exists
- no known regression remains in the current unfinished Claude area

Difficulty:
- `High`

Reasoning note:
- this slice should end with confidence, not just green unit tests

## Execution Guidance

When implementing this program:

- start with `PR-019`, `PR-020`, and `PR-021` before touching the memory atlas work
- run `PR-028` immediately after those backend-correctness slices so future agents read current guidance first
- keep new logic out of `agent_loop.py` and `consolidation/engine.py` when it can live in a focused helper
- prefer provenance in `meta` over new top-level schema fields unless repeated access patterns prove a new field is necessary
- use the existing stores and lifecycle states unless a genuine data-model failure forces a broader change
- maintain idempotence in nightly processes
- treat blocked state as semantically meaningful, not as a temporary inconvenience

## What To Avoid

- do not let the LLM decide everything in consolidation when heuristics are sufficient
- do not create executable work from ambiguous or blocked commitments
- do not let promise extraction turn into generic intent mining from every conversation
- do not make the dashboard prettier at the expense of explaining real state
- do not start `PR-024` without explicit user approval to use extra-high reasoning if the design questions remain cross-cutting

## Immediate Next Move

If implementation begins now, the first active slice should be:

- `PR-019` Commitment Pause/Resume Correctness

The reason:

- it fixes a real broken promise path already present in the uncommitted Claude work
- it is the highest-leverage trust repair
- it unblocks the rest of the continuation program
