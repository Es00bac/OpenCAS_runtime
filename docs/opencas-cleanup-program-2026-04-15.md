# OpenCAS Cleanup Program

Date: 2026-04-15

Purpose:
- reduce maintenance risk without flattening the parts of OpenCAS that are intentionally unusual
- remove duplicated logic that currently has to be fixed in parallel
- correct live documentation so future collaborators do not anchor on contradictory state
- preserve behavior, identity continuity, autonomy, and real-work usefulness while cleaning the substrate

## Cleanup Principles

Every cleanup slice should satisfy all of these constraints:
- preserve current user-visible behavior unless the current behavior is clearly a bug
- reduce duplicate logic or contradictory source-of-truth docs
- improve subsystem integration rather than just moving code around
- land with focused verification, not speculative rewrites
- keep OpenCAS recognizable as OpenCAS: persistent, identity-bearing, autonomous, relational, and useful
- prefer extracting cohesive service/helper modules out of large route/runtime files instead of adding more responsibilities to existing god objects
- do not centralize unrelated logic into one new helper file; every extraction should have a narrow reason to exist

## Current Audit Findings

1. Chat transport duplication has been removed.
- `/chat`, `/api/chat/send`, and websocket chat now share the same transport helpers.
- Attachment execution and refusal-turn persistence no longer drift across chat entry points.

2. The dashboard frontend is cleaner but still not finished.
- `opencas/dashboard/static/index.html` lost major operations and memory applet slabs to helper modules, but the SPA shell still carries too much fetch/render glue.
- The remaining dashboard debt is now concentrated rather than smeared across unrelated blocks.

3. `AgentRuntime` remains the highest-risk integration seam.
- `opencas/runtime/agent_loop.py` has shed conversation, cycle, lifecycle, episodic, reflection, telegram, tool-registration, maintenance, and status-view blocks.
- The remaining cleanup work is now about preserving that decomposition momentum without weakening runtime behavior, while normalizing the final context-building seams around it.

4. `operations.py` is now mostly route assembly instead of a god object.
- `opencas/api/routes/operations.py` has been reduced to a thin route surface over dedicated browser, session, tasking, activity, monitoring, qualification, and operator-action helpers.
- The next API cleanup work should target smaller remaining route surfaces and documentation truth, not try to re-solve the already-extracted seams.

5. The bootstrap TUI shell is no longer a monolith.
- `opencas/bootstrap/tui.py` now acts as the app shell and screen registry.
- Profile, setup, runtime, state, bootstrap, and shared-widget responsibilities live in focused helper modules, so future work should preserve that split instead of regrowing the shell.

6. Live docs still need active normalization.
- `TaskList.md` remains the single live execution source.
- Historical docs are mostly demoted, but wording drift still needs correction whenever file boundaries or active frontiers change.

7. Integration correctness is part of cleanup.
- Subsystems such as somatic state, ToM, musubi, commitments, workspace focus, dashboard surfaces, and qualification evidence should all have visible, maintained connection points.
- Dead surfaces and stale persisted references are still forms of technical debt here.

## Bounded Cleanup Slices

| Slice | Goal | Difficulty | Stop Gate |
| --- | --- | --- | --- |
| `PR-030` | Establish cleanup program and normalize live task/doc source-of-truth | Medium | No |
| `PR-031` | Centralize chat transport and attachment execution helpers | Medium | No |
| `PR-032` | Extract qualification/readiness services out of `opencas/api/routes/operations.py` | Medium | No |
| `PR-033` | Normalize session/context store surfaces and remove API-layer drift around session metadata and status filtering | Medium | No |
| `PR-034` | Extract shared dashboard fetch/render helpers from the monolithic chat/overview frontend | High | No |
| `PR-035` | Documentation truth pass focused on live subsystem integration surfaces and current operational entry points | Medium | No |
| `PR-036` | Decompose `AgentRuntime.converse()` into smaller verified units without changing behavior | Extra High | Yes |

## Slice Notes

### `PR-030` Cleanup program and live doc normalization
- Create one canonical cleanup program doc.
- Keep `TaskList.md` aligned with the live cleanup frontier.
- Remove obviously contradictory headings/state in tasking docs.

### `PR-031` Chat transport consolidation
- Centralize common chat turn execution.
- Keep attachment resolution, somatic serialization, and session-id resolution in one place.
- Ensure every chat surface continues to behave the same after refactor.

### `PR-032` Qualification/readiness service extraction
- Move qualification artifact loading, rerun history handling, and recommendation helpers out of `operations.py`.
- Keep the operations routes thin and preserve the existing API payloads.
- Use this as the first concrete god-object reduction slice after chat transport consolidation.

### `PR-033` Session/context surface normalization
- Remove drift between route expectations, context-store APIs, and frontend session behavior.
- Tighten session-status filtering and session metadata handling so chat/session surfaces stay coherent.

### `PR-034` Dashboard fetch/render helper extraction
- Extract repeated `fetch`/error/loading/update behavior into shared helpers inside the SPA.
- Reduce repeated route-specific shell code where the logic is the same.
- Avoid cosmetic churn; focus on maintainability and correctness.

### `PR-035` Documentation truth pass
- Re-check `TaskList.md`, `documentation-map.md`, readiness docs, release docs, and current operator entry points against the actual codebase.
- Make the live path obvious and historical material clearly historical.

### `PR-036` `AgentRuntime` decomposition
- Split the conversation/refusal/logging/tool-loop/commitment path into smaller helpers or service modules.
- Preserve exact behavior and verification semantics.
- This is the first slice that should be treated as `Extra High`.

## Current Progress Snapshot

Completed foundation slices:
- `PR-030` live cleanup program and task/doc normalization
- `PR-031` shared chat transport and attachment execution consolidation
- `PR-032` qualification/readiness extraction out of the operations route layer
- `PR-033` session/context surface normalization
- `PR-034` dashboard helper extraction
- `PR-035` documentation truth-pass foundation

Completed bounded follow-on slices:
- route-model extraction plus dedicated operations helpers for monitoring, operator actions, browser, sessions, tasking, activity, and qualification
- bootstrap TUI extraction into state, bootstrap, shared widget, intro/user/profile/setup/runtime screens, and bootstrap helpers
- workspace/path containment normalization plus persisted Chronicle/workspace reference repair
- `AgentRuntime` seam extraction across conversation, cycle, lifecycle, episodic, reflection, maintenance, telegram, status-view, and tool-registration paths
- `MemoryStore` schema/serialization plus episode/edge helper extraction
- retrieval helper extraction into query, ranking, and search modules
- consolidation helper extraction into commitment cleanup and signal maintenance modules
- daydream and qualification support-module splits
- advanced tool-registration, workflow adapter, and bootstrap pipeline support splits

Current status:
1. The bounded cleanup program has reached a satisfactory stopping point: the major god-object route, bootstrap, workflow, memory, retriever, consolidation, and runtime slabs have all been split into focused helper modules with stable facade entry points.
2. Repo-wide documentation is now expected to stay aligned incrementally as feature work resumes; cleanup is no longer the primary frontier.
3. Workspace-containment and persisted-state repair remain standing maintenance rules whenever path policy or artifact locations change.

Rationale:
- The largest route, bootstrap, memory-store, retriever, consolidation, and config-control-plane monoliths have all been materially reduced.
- The high-risk cleanup work that motivated this program has been addressed, and the remaining larger modules are now cohesive domain surfaces rather than mixed responsibility god objects.
- Future maintenance should preserve the extracted seams and continue as feature-driven incremental edits instead of reopening a broad cleanup program by default.
