# OpenCAS Task List

Last updated: 2026-04-15

Purpose:
- provide the canonical task list referenced by [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md)
- track the active path to first regular-use deployment readiness
- keep multi-model collaboration grounded in one current execution list

Rules:
- update this file when a task starts, changes scope, or completes
- do not create parallel task lists elsewhere
- treat this file as the execution source of truth; older audits and handoff notes are reference only

Current active stance:
- the bounded cleanup program has reached a satisfactory stopping point
- future maintenance should preserve the extracted seams and keep documentation aligned as feature work resumes

Canonical current docs:
- [TaskList.md](TaskList.md)
- [documentation-map.md](docs/documentation-map.md)
- [opencas-deep-system-audit-2026-04-09.md](docs/opencas-deep-system-audit-2026-04-09.md)
- [production-readiness-status-2026-04-09.md](docs/production-readiness-status-2026-04-09.md)
- [opencas-production-program-plan-2026-04-08.md](docs/opencas-production-program-plan-2026-04-08.md)
- [opencas-continuation-program-2026-04-15.md](docs/opencas-continuation-program-2026-04-15.md)
- [opencas-cleanup-program-2026-04-15.md](docs/opencas-cleanup-program-2026-04-15.md)
- [testing-execution-plan-2026-04-09.md](docs/qualification/testing-execution-plan-2026-04-09.md)
- [live_validation_summary.md](docs/qualification/live_validation_summary.md)
- [qualification_remediation_rollup.md](docs/qualification/qualification_remediation_rollup.md)

## In Progress

- no cleanup slices currently in progress

## Recently Completed

- `PR-086` Config control-plane split
  - owner: Codex
  - status: completed
  - result:
    - extracted config overview/serialization logic into `opencas/api/config_overview.py` and gateway mutation helpers into `opencas/api/config_mutations.py`
    - reduced `opencas/api/routes/config.py` to a thin route assembly layer while preserving the existing dashboard/operator config surface
    - preserved config-route behavior by re-running the focused dashboard config regression subset after the split

- `PR-085` Bootstrap service-stage extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted relational/plugin/governance/planning/harness/diagnostics startup into `opencas/bootstrap/pipeline_services.py`
    - reduced `bootstrap/pipeline.py` to a staged orchestrator plus the remaining provider and embedding assembly instead of another mixed mid-pipeline service slab
    - preserved bootstrap behavior by re-running the focused bootstrap, server, and integration phase 1 regression subset after the split

- `PR-036` `AgentRuntime` decomposition
  - owner: Codex
  - status: completed
  - goal: split `AgentRuntime.converse()` and adjacent runtime orchestration into smaller verified units without changing behavior
  - current output:
    - extracted the first conversation-turn seam into `opencas/runtime/conversation_turns.py`, moving refusal handling, user-turn persistence, tool-loop execution, intermediate message persistence, and post-response state updates out of `agent_loop.py`
    - tightened the tool-loop heuristic so reflective/plain chat turns stay tool-free instead of falling back to the default exploration subset
    - extracted the next cycle seam into `opencas/runtime/cycle_phases.py`, moving promoted-work enqueueing, workspace intervention evaluation, and executive queue draining out of `AgentRuntime.run_cycle()`
    - extracted runtime control-plane/workflow/consolidation status assembly into `opencas/runtime/status_views.py`, so the monitoring/operator snapshot path no longer lives inline inside `AgentRuntime`
    - extracted autonomous scheduler/server/shutdown orchestration into `opencas/runtime/lifecycle.py`, so `AgentRuntime` no longer carries the runtime-mode lifecycle block inline
    - extracted episodic-memory and continuity helpers into `opencas/runtime/episodic_runtime.py`, moving continuity decay, goal-directive parsing, self-commitment capture, and episode-edge persistence out of `agent_loop.py`
    - extracted daydream, reflection, and identity-rebuild helpers into `opencas/runtime/reflection_runtime.py`, so the runtime’s inner-life maintenance path is now isolated from the main orchestration file
    - extracted Telegram runtime control into `opencas/runtime/telegram_runtime.py`, moving config loading, service assembly, status/configure flows, and pairing approval out of the main runtime file
    - extracted default tool registration into `opencas/runtime/tool_registration.py`, so adapter wiring and schema policy now have one dedicated module instead of living inline in `AgentRuntime.__init__`
    - split runtime tool registration into `tool_registration_foundation.py`, `tool_registration_workflow.py`, and `tool_registration_advanced.py`, so the post-refactor registration layer no longer re-formed as a single helper monolith
    - split `tool_registration_advanced.py` into focused coding, interactive-planning, and runtime-integration/workspace registration helpers so the last advanced registration slab is now just an assembly shell
    - split `opencas/tools/adapters/workflow.py` into managed-workspace path helpers, tasking CRUD helpers, and PTY supervision helpers so the workflow tool surface no longer relies on one mixed operator adapter module
    - extracted `BootstrapContext.close()` and bootstrap support helpers into dedicated modules so `bootstrap/pipeline.py` no longer mixes shutdown bookkeeping, guard/runtime support, and substrate assembly in one file
    - extracted generic `ContextBuilder` support helpers into `opencas/context/builder_support.py` so `builder.py` now focuses on system-prompt assembly rather than also owning token heuristics, identity-anchor formatting, redundancy pruning, and retrieval-usage bookkeeping
    - extracted compaction, consolidation, BAA completion handling, snapshot syncing, and runtime trace/response helpers into `opencas/runtime/maintenance_runtime.py`
    - final runtime/context cleanup reached a stable stopping point after the retriever, maintenance, lifecycle, cycle, episodic, reflection, telegram, tool-registration, and bootstrap seams were all extracted and re-verified

## Recently Completed

- `PR-084` Bootstrap store-stage extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted foundational store connection and executive snapshot restoration into `opencas/bootstrap/pipeline_stores.py` behind a typed `RuntimeStoreBundle`
    - reduced `bootstrap/pipeline.py` so the early substrate boot path is a staged orchestrator rather than another long mixed store/executive setup slab
    - preserved bootstrap behavior by re-running the focused bootstrap, server, and integration phase 1 regression subset after the split

- `PR-083` Bootstrap context and workspace-index assembly split
  - owner: Codex
  - status: completed
  - result:
    - moved `BootstrapContext` into `opencas/bootstrap/context.py` so lifecycle ownership is no longer embedded inside `bootstrap/pipeline.py`
    - extracted workspace-index startup and final context assembly into `opencas/bootstrap/pipeline_context.py`, reducing `pipeline.py` to staged substrate orchestration plus the remaining bootstrap sequence
    - preserved bootstrap behavior by re-running the focused bootstrap, server, and integration phase 1 regression subset after the split

- `PR-082` Retriever candidate fusion and MMR split
  - owner: Codex
  - status: completed
  - result:
    - extracted candidate-map assembly, graph-expansion normalization, and weighted fusion into `opencas/context/retrieval_candidates.py`
    - extracted MMR vector-resolution and reranking into `opencas/context/retrieval_mmr.py` while preserving the `MemoryRetriever` wrapper surface
    - reduced `opencas/context/retriever.py` so it focuses on retrieval orchestration and public adapter methods instead of also carrying the full inspection/fusion slab inline
    - preserved retrieval behavior by re-running the focused retriever regression suite after the split

- `PR-081` Multi-model routing and gateway control-plane upgrade
  - owner: Codex
  - status: completed
  - result:
    - added persisted OpenCAS model-routing policy with single-model and Light/Standard/High/Extra-High tiered modes, plus runtime-aware lane resolution inside `LLMClient` and the active tool-use loop
    - annotated core LLM callsites with explicit complexity tiers so low-cost work stays light while harder planning, retries, and deep tool loops can climb toward higher-capability models
    - extended the System dashboard and config API so users can assign models to complexity tiers, configure provider presets through OpenLLMAuth, manage auth profiles/providers/custom models, and verify provider connectivity against the active gateway material
    - preserved behavior with focused regression coverage for routing resolution, persisted bootstrap loading, and config-route mutation of app-local OpenLLMAuth state

- `PR-080` Context builder support split
  - owner: Codex
  - status: completed
  - result:
    - extracted identity-anchor formatting, token estimation, redundancy pruning, retrieval-usage bookkeeping, and memory-entry conversion into `opencas/context/builder_support.py` while preserving the existing `ContextBuilder` method surface
    - reduced `opencas/context/builder.py` so it focuses more tightly on system-prompt composition and manifest assembly instead of also carrying generic retrieval-support helpers
    - preserved context-builder behavior by compiling the split modules and re-running the focused context-builder regression suite after the extraction

- `PR-079` Bootstrap pipeline support split
  - owner: Codex
  - status: completed
  - result:
    - extracted bootstrap shutdown bookkeeping into `opencas/bootstrap/context_close.py` and support helpers into `opencas/bootstrap/pipeline_support.py` while preserving the `BootstrapPipeline` method surface
    - reduced `opencas/bootstrap/pipeline.py` so it focuses more tightly on substrate assembly rather than also owning shutdown traversal, embed-model fallback logic, runtime guards, and stage logging helpers
    - preserved bootstrap behavior by compiling the split modules and re-running the focused bootstrap pipeline and shutdown regression subset after the extraction

- `PR-078` Workflow adapter split
  - owner: Codex
  - status: completed
  - result:
    - extracted managed-workspace path resolution into `workflow_paths.py`, tasking/scheduling/writing/repo helpers into `workflow_tasking.py`, and PTY supervision into `workflow_supervision.py` behind the existing `WorkflowToolAdapter` facade
    - reduced `opencas/tools/adapters/workflow.py` to a thin dispatch shell so the workflow tool surface no longer mixes path policy, CRUD orchestration, repo triage, and interactive PTY supervision in one module
    - preserved the workflow tool contract by compiling the split modules and re-running the focused workflow adapter test suite after the extraction

- `PR-077` Advanced tool registration split
  - owner: Codex
  - status: completed
  - result:
    - extracted advanced coding/introspection, interactive-planning, and runtime integration/workspace registration into focused runtime modules behind the existing `register_advanced_tools()` entry point
    - reduced `tool_registration_advanced.py` to an assembly shell so the registration surface no longer hides one remaining large advanced helper slab
    - preserved the runtime tool surface by compiling the new modules and re-running the focused tool registration and registry test subset after the split

- `PR-076` Tool registration foundation and workflow split
  - owner: Codex
  - status: completed
  - result:
    - extracted filesystem/search/edit, shell/process/pty, workflow state/tasking, and web/browser tool registration into focused runtime modules backed by a narrow `ToolRegistrationSpec` helper
    - reduced `tool_registration_foundation.py` and `tool_registration_workflow.py` to assembly shells so the registration surface no longer lives in two large single-function slabs
    - preserved the runtime tool surface by compiling the new modules and re-running the focused tool registration and registry test subset after the split

- `PR-075` Consolidation signal maintenance extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted stale-reference detection, salience reweighting, strong-signal promotion, orphan recovery, identity-core promotion, and cluster hashing into `opencas/consolidation/signal_maintenance.py`
    - reduced `opencas/consolidation/engine.py` so it focuses on nightly cycle orchestration, episode clustering, commitment cleanup delegation, and belief/identity updates instead of also owning the full signal-maintenance slab
    - preserved consolidation behavior by re-running the full consolidation regression subset and promise-lifecycle qualification after the split

- `PR-074` Memory store episode and edge split
  - owner: Codex
  - status: completed
  - result:
    - extracted episode persistence/query helpers into `opencas/memory/store_episodes.py` and edge persistence/query helpers into `opencas/memory/store_edges.py`
    - reduced `opencas/memory/store.py` so it acts as the stable `MemoryStore` facade instead of inlining episode upserts, edge upserts, FTS queries, and graph-edge maintenance SQL
    - preserved the public `MemoryStore` API by keeping thin wrappers on the class and re-running the full memory and retriever regression subset after the split

- `PR-073` Retriever search helper extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted semantic search, keyword search, graph expansion, graph candidate merging, relational seed scoring, emotion boosting, and reciprocal-rank fusion helpers into `opencas/context/retrieval_search.py`
    - reduced `opencas/context/retriever.py` so it focuses on retrieval inspection, score fusion, and MMR reranking while keeping the existing `MemoryRetriever` wrapper surface stable
    - preserved the retrieval behavior by fixing the wrapper compatibility break immediately and re-running the full retrieval regression subset after the split

- `PR-072` Consolidation commitment cleanup extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted commitment deduplication and chat-recovery logic into `opencas/consolidation/commitment_cleanup.py` while leaving thin wrappers on `NightlyConsolidationEngine`
    - reduced `opencas/consolidation/engine.py` so it focuses on episode clustering, salience reweighting, belief sweeping, and identity updates instead of also owning the full commitment cleanup subsystem
    - preserved the existing consolidation and promise-lifecycle behavior by verifying the full consolidation regression subset and qualification scenario after the split

- `PR-071` Memory store schema and serialization extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted the SQLite schema and migration list into `opencas/memory/store_schema.py` and the row/parameter adapters into `opencas/memory/store_serialization.py`
    - reduced `opencas/memory/store.py` so it focuses on query and persistence orchestration instead of inlining every schema string, migration, and model-serialization detail
    - preserved the `MemoryStore` API while reusing the new helpers across episode, edge, memory, and compaction persistence paths

- `PR-070` Qualification analysis and run-loader split
  - owner: Codex
  - status: completed
  - result:
    - split comparison/remediation/recommendation logic into `opencas/api/qualification_analysis.py` and validation-run/rerun detail loading into `opencas/api/qualification_runs.py`
    - reduced `opencas/api/qualification_service.py` to a thin compatibility facade so the route layer kept one stable import surface during the refactor
    - preserved the qualification operations behavior by reusing the same loaders and rerun-command assembly under the new module boundaries

- `PR-069` Daydream store split
  - owner: Codex
  - status: completed
  - result:
    - split `opencas/daydream/store.py` into `opencas/daydream/daydream_store.py` and `opencas/daydream/conflict_store.py` with a shared `opencas/daydream/sqlite_base.py` lifecycle shell
    - left `opencas/daydream/store.py` as a thin compatibility export so existing imports and bootstrap wiring stayed stable during the refactor
    - added direct compatibility coverage while preserving the existing daydream/conflict persistence behavior and regression suite

- `PR-068` Retriever query and ranking helper extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted query-intent parsing into `opencas/context/retrieval_query.py` and ranking post-processing into `opencas/context/retrieval_ranking.py`
    - reduced `opencas/context/retriever.py` so it focuses on candidate fusion, semantic search, keyword search, and graph expansion instead of carrying every pure helper inline
    - preserved the `MemoryRetriever` method surface with thin wrappers and verified the retrieval regression suite directly

- `PR-067` Qualification service model/history extraction
  - owner: Codex
  - status: completed
  - result:
    - split the qualification request/response models into `opencas/api/qualification_models.py` and the rerun-history persistence/loaders into `opencas/api/qualification_history.py`
    - reduced `opencas/api/qualification_service.py` so it focuses on summary/detail assembly and recommendation enrichment instead of owning every support type and history helper inline
    - preserved the qualification route surface by restoring recent-rerun comparison and rate-window enrichment at the service layer and verifying the qualification route subset directly

- `PR-066` Bootstrap profile-screen bundle split
  - owner: Codex
  - status: completed
  - result:
    - split the profile/onboarding screen bundle into `opencas/bootstrap/tui_screens_intro.py` and `opencas/bootstrap/tui_screens_user.py` so the bootstrap wizard no longer depends on one large profile-screen module
    - left `opencas/bootstrap/tui_screens_profile.py` as a thin compatibility export while moving the TUI registry and tests onto the new modules
    - verified the bootstrap registry/component/state subset so the screen flow stayed stable through the split

- `PR-065` Runtime constructor setup extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted the boot-time `AgentRuntime` wiring into `opencas/runtime/runtime_setup.py`, separating autonomy, execution, memory, and channel assembly from the constructor shell
    - reduced `opencas/runtime/agent_loop.py` to the runtime method surface and orchestration wrappers while preserving the existing public runtime API and boot ordering
    - fixed the local telegram-service rebuild path by importing `build_runtime_telegram_service()` where `_build_telegram_service()` actually uses it

- `PR-064` Cleanup documentation truth pass refresh
  - owner: Codex
  - status: completed
  - result:
    - updated the cleanup program, documentation map, and collaborator guidance so they reflect the post-split route and bootstrap layout instead of treating already-extracted seams as still pending
    - clarified that the cleanup program is the primary planning doc for the current structural frontier while the continuation program remains subsystem guidance for commitment continuity
    - recorded the current standing cleanup state so future agent sessions do not re-open already-finished `operations.py` or bootstrap TUI reduction work by mistake

- `PR-063` Operations browser route extraction
  - owner: Codex
  - status: completed
  - result:
    - moved the remaining browser session route declarations into `opencas/api/operations_browser.py` so the browser control surface now lives with the browser session service instead of inside `opencas/api/routes/operations.py`
    - reduced `opencas/api/routes/operations.py` to route assembly plus non-browser operations seams while preserving the existing browser route surface and operator-action history behavior
    - verified the full browser route subset directly, including refresh, action, screenshot, close, and clear flows

- `PR-062` Bootstrap runtime-screen extraction
  - owner: Codex
  - status: completed
  - result:
    - moved the bootstrap progress and runtime-launch screen into `opencas/bootstrap/tui_runtime.py`
    - reduced `opencas/bootstrap/tui.py` to the app shell and screen registry while preserving the bootstrap flow and runtime startup behavior
    - extended the TUI registry coverage so the bootstrap runtime screen is still resolved through `BootstrapTUI.get_screen()` after the split

- `PR-061` Bootstrap TUI setup-screen extraction
  - owner: Codex
  - status: completed
  - result:
    - moved the workspace, credentials, models, advanced, and review screens into `opencas/bootstrap/tui_screens_setup.py`
    - reduced `opencas/bootstrap/tui.py` to the bootstrap progress screen and app wiring while preserving the screen registry and wizard flow
    - extended the registry coverage so profile, setup, and runtime screens are all resolved through `BootstrapTUI.get_screen()` after the split

- `PR-060` Operations qualification helper extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted qualification summary, label detail, rerun detail, validation-run lookup, and rerun-launch behavior into `opencas/api/operations_qualification.py`
    - reduced `opencas/api/routes/operations.py` again by removing the remaining qualification orchestration block and fixing the latent `uuid4` rerun-launch path while preserving the existing route surface
    - verified the qualification routes directly, including rerun launch and rerun provenance behavior

- `PR-059` Bootstrap TUI profile-screen extraction
  - owner: Codex
  - status: completed
  - result:
    - moved the onboarding and profile-oriented bootstrap screens into `opencas/bootstrap/tui_screens_profile.py`
    - reduced `opencas/bootstrap/tui.py` to runtime/bootstrap orchestration, late-stage screens, and app wiring while preserving the existing screen registry and behavior
    - added focused registry coverage so moved profile screens and local runtime screens are both resolved through `BootstrapTUI.get_screen()`

- `PR-058` Operations activity helper extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted receipt and background-task route behavior into `opencas/api/operations_activity.py`
    - reduced `opencas/api/routes/operations.py` again by removing the remaining receipt/task listing and detail logic while preserving response shapes
    - added focused task-list and task-detail route tests so the extracted activity path is covered directly instead of relying on incidental route coverage

- `PR-057` Operations tasking helper extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted work, commitment, and plan route behavior into `opencas/api/operations_tasking.py`
    - reduced `opencas/api/routes/operations.py` by removing the repeated tasking CRUD and serialization blocks while keeping route behavior and validation responses unchanged
    - verified the affected work, commitment, and plan routes so persistence, operator snapshots, and plan action rendering stayed stable during the extraction

- `PR-056` Operations process and PTY session extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted process-session inspection, PTY-session inspection, PTY input handling, and session-list aggregation into `opencas/api/operations_sessions.py`
    - reduced `opencas/api/routes/operations.py` by replacing the repeated process and PTY orchestration blocks with one shared session helper path
    - verified the affected session routes so rerun provenance, PTY observation, and operator-action behavior stayed unchanged during the extraction

- `PR-055` Operations browser-session helper extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted browser session lookup, observation merging, refresh handling, and operator-action wiring into `opencas/api/operations_browser.py`
    - reduced `opencas/api/routes/operations.py` by replacing the repeated browser navigate/click/type/press/wait/capture orchestration blocks with one shared helper path
    - verified the browser operations route subset so response shapes and operator-action history stayed unchanged during the extraction

- `PR-054` Operations route model extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted operations request/response models into `opencas/api/operations_models.py`
    - reduced `opencas/api/routes/operations.py` by moving the large model slab out of the router module while keeping route behavior and response schemas unchanged
    - verified the affected operations and dashboard paths with focused route regressions after restoring the session-entry adapter helper

- `PR-053` Bootstrap TUI shared-widget extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted `StepHeader`, `HelpText`, and `NavButtons` into `opencas/bootstrap/tui_components.py`
    - reduced `opencas/bootstrap/tui.py` by removing the shared widget definitions while keeping the wizard screen flow unchanged
    - added focused widget tests so future TUI cleanup work keeps the shared component contract stable

- `PR-052` Bootstrap TUI bootstrap-helper extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted user-bio composition, questionnaire payload persistence, and `BootstrapConfig` construction into `opencas/bootstrap/tui_bootstrap.py`
    - reduced the bootstrap screen logic in `opencas/bootstrap/tui.py` while keeping the questionnaire save path and bootstrap pipeline behavior unchanged
    - added focused tests covering fallback bio composition, questionnaire persistence, and config construction from wizard state

- `PR-051` Operations monitoring and operator-action extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted readiness and hardening snapshot builders into `opencas/api/operations_monitoring.py`
    - extracted operator-action persistence and history loading into `opencas/api/operator_actions.py`
    - reduced `opencas/api/routes/operations.py` while keeping the existing operations route surface and operator-audit behavior unchanged

- `PR-050` Bootstrap TUI state extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted bootstrap wizard state and provider/model discovery into `opencas/bootstrap/tui_state.py`
    - reduced `opencas/bootstrap/tui.py` by removing the global state/config discovery slab while keeping the screen flow and persisted questionnaire behavior unchanged
    - added focused tests for auth-profile scanning and model-discovery fallback behavior so the TUI bootstrap path stays stable during future cleanup work

- `PR-049` Memory API helper extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted memory-route serialization helpers into `opencas/api/memory_serialization.py`
    - extracted embedding projection and retriever bootstrap helpers into `opencas/api/memory_projection.py`
    - kept the `/api/memory` route surface unchanged while shrinking `opencas/api/routes/memory.py` and preserving the dashboard-facing response shapes

- `PR-048` Runtime grouped tool-registration extraction
  - owner: Codex
  - status: completed
  - result:
    - split the runtime tool registration layer into `tool_registration_foundation.py`, `tool_registration_workflow.py`, and `tool_registration_advanced.py`
    - kept `register_runtime_default_tools()` as the stable runtime entry point while removing the new helper-module re-monolithization risk
    - preserved the runtime tool surface for workflow, browser, PTY, runtime-status, MCP, and workspace-index tools

- `PR-046` Runtime maintenance and event-hook extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted session compaction, nightly consolidation, somatic snapshot persistence, BAA completion handling, executive snapshot syncing, and trace/response helpers into `opencas/runtime/maintenance_runtime.py`
    - kept the public `AgentRuntime` method surface unchanged so lifecycle hooks, scheduler flows, and operator paths continue calling the same runtime methods
    - added focused housekeeping tests to prove the extracted module without depending on the flaky broader runtime fixture teardown path

- `PR-045` Runtime default tool registration extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted the default tool adapter/schema registration block into `opencas/runtime/tool_registration.py`
    - reduced `opencas/runtime/agent_loop.py` by moving the largest remaining constructor-time registration seam behind a single runtime helper
    - preserved the existing runtime tool surface so workflow, browser, PTY, runtime-status, and workspace-index tools continue registering through the same `AgentRuntime` method

- `PR-044` Runtime Telegram extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted Telegram config loading, service assembly, status/configure behavior, and pairing approval into `opencas/runtime/telegram_runtime.py`
    - kept the public runtime control surface unchanged so lifecycle helpers and dashboard routes still call the same `AgentRuntime` methods
    - added focused helper tests and direct route verification to keep the extraction behavior-true

- `PR-043` Repo-local maintenance bootstrap normalization
  - owner: Codex
  - status: completed
  - result:
    - added `opencas/maintenance/script_config.py` so repo-local maintenance scripts share one managed-workspace/state bootstrap helper instead of re-encoding local defaults
    - updated `scripts/sync_workspace_memories.py`, `scripts/sync_chronicle_summaries.py`, and `scripts/repair_workspace_references.py` to use the shared helper, keeping workspace-containment policy aligned across maintenance entry points
    - added focused regression coverage for the helper’s default and override behavior

- `PR-042` Runtime episodic and reflection extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted continuity decay, user-goal parsing, self-commitment capture, episode persistence, and episode-edge creation into `opencas/runtime/episodic_runtime.py`
    - extracted daydream/reflection execution plus identity rebuild and metacognition helpers into `opencas/runtime/reflection_runtime.py`
    - reduced `opencas/runtime/agent_loop.py` by moving another dense runtime seam behind narrow helper modules while preserving the existing runtime method surface

- `PR-039` Dashboard memory surface extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted the memory atlas applet plus its memory-specific render globals into `opencas/dashboard/static/js/memory_app.js`
    - reduced `opencas/dashboard/static/index.html` by moving the largest remaining memory-specific dashboard block behind one dedicated module while preserving the existing Alpine/HTMX contract
    - added dashboard smoke coverage so the static shell now asserts the external memory module is wired in alongside the memory routes it drives

- `PR-040` Runtime tool and plugin execution extraction
  - owner: Codex
  - status: completed
  - result:
    - extracted the tool/plugin/MCP execution seam into `opencas/runtime/tool_runtime.py`, moving tool-use context hydration, MCP registration, approval-wrapped tool execution, plugin lifecycle calls, and repair submission out of `AgentRuntime`
    - kept the `AgentRuntime` method surface stable by turning the runtime methods into thin wrappers over the extracted module
    - verified the moved execution path with direct execution checks plus targeted tool, repair, and plugin/runtime regressions

- `PR-041` Bootstrap background task cleanup
  - owner: Codex
  - status: completed
  - result:
    - taught `BootstrapContext` to track and cancel bootstrap-owned background tasks during `close()`
    - moved the embedding backfill task into tracked context state and made the backfill worker exit quietly on cancellation
    - taught `WorkspaceIndexService` to track and cancel its initial full-scan task instead of leaving it detached past shutdown
    - added focused bootstrap coverage so short-lived runtimes stop leaking background backfill tasks into teardown

- `PR-038` Workspace reference normalization and live state repair
  - owner: Codex
  - status: completed
  - result:
    - added `opencas/maintenance/workspace_references.py` plus `scripts/repair_workspace_references.py` to normalize stale Chronicle/workspace references inside SQLite-backed state
    - updated the system prompt workspace orientation to use the configured primary and managed workspace roots instead of a hard-coded repo path
    - repaired the live `.opencas` databases so old root-level and legacy-workspace Chronicle references now point at `workspace/Chronicles`

- `PR-037` Managed workspace containment for agent-created artifacts
  - owner: Codex
  - status: completed
  - result:
    - added a dedicated managed-workspace root policy to `BootstrapConfig` so agent-created artifacts default into a project-local `workspace/` directory instead of spilling across the host environment
    - routed workflow-created writing artifacts through that managed workspace root and reject output paths outside it
    - exposed the managed workspace root in the runtime control-plane status, updated chronicle sync scripts to look under `workspace/Chronicles`, and added tests for default containment, explicit overrides, and out-of-bounds path rejection

- `PR-034` Dashboard fetch/render helper extraction
  - owner: Codex
  - status: completed
  - result:
    - added `opencas/dashboard/static/js/http_helpers.js` as a shared HTTP utility layer for the dashboard
    - migrated the chat and daydream applets off handwritten fetch/query boilerplate onto the shared helper layer
    - extracted the operations detail/session/qualification mutation layer into `opencas/dashboard/static/js/operations_helpers.js`
    - removed the duplicated operations action handlers from `index.html` while preserving the existing operator-facing routes and markup contracts

- `PR-035` Documentation truth pass and collaborator conventions
  - status: completed
  - result: updated the cleanup program, [CLAUDE.md](CLAUDE.md), [AGENTS.md](AGENTS.md), and [TaskList.md](TaskList.md) so the live collaborator guidance now points at the cleanup frontier and explicitly treats god-object growth as a regression.

- `PR-033` Session/context surface normalization
  - status: completed
  - result: added real session metadata support to `SessionContextStore` with persistent name/status fields, empty-session creation without hidden placeholder messages, session search/list/get/update methods, and status-aware chat session routes so the frontend, API, and context substrate now agree on one session contract.

- `PR-032` Qualification/readiness service extraction
  - status: completed
  - result: extracted qualification artifact loading, rerun history handling, validation-run detail loading, and rerun command construction into `opencas/api/qualification_service.py`, reducing `opencas/api/routes/operations.py` by a large self-contained subsystem while preserving the existing operations API surface.

- `PR-030` Cleanup program and live source-of-truth normalization
  - status: completed
  - result: added [opencas-cleanup-program-2026-04-15.md](docs/opencas-cleanup-program-2026-04-15.md), aligned [TaskList.md](TaskList.md) and [documentation-map.md](docs/documentation-map.md) with the live cleanup frontier, and updated the program so further work explicitly targets bounded extraction from the largest god-object files instead of helper sprawl.

- `PR-031` Shared chat transport and attachment execution
  - status: completed
  - result: extracted shared chat turn execution, attachment resolution, upload persistence, session-id resolution, and somatic serialization into `opencas/api/chat_service.py`, so `/chat`, `/api/chat/send`, and websocket chat now reuse one implementation instead of carrying parallel logic.

- `PR-029` Chat attachment handling and refusal-path truthfulness
  - status: completed
  - result: text and markdown uploads from the chat UI now become real queued attachments instead of being pasted into the compose textarea, the chat send route materializes attachment content for model context while preserving attachment metadata in session history, and refusal-path turns now persist the triggering user message so uploads and refusals stop disappearing from the transcript.

- `PR-016` Procedural memory extraction
  - status: completed
  - result: added post-task hook in `BoundedAssistantAgent` that summarizes successful task tool sequences into `PROCEDURAL` episodes with embeddings. Wired `MemoryRetriever._semantic_search` to also fetch episodes by embedding_id so procedural memories surface for similar future tasks. Added `list_episodes_by_embedding_ids` to `MemoryStore`. All 5 new tests and 18 related tests pass.

- `PR-015` SparkEvaluator (structured novelty filter)
  - status: completed
  - result: created `SparkEvaluator` in `opencas/daydream/spark_evaluator.py` with cosine distance, somatic alignment, relational alignment, and executive feasibility scoring. Wired into `DaydreamGenerator` and `AgentRuntime`. Added 8 tests covering all scoring dimensions and DaydreamGenerator integration; all pass.

## Completed 2026-04-15 Continuation Slices

- `PR-019` Commitment pause/resume correctness
  - owner: Codex
  - status: completed
  - goal: make deferred work restoration track real executive pause recovery without reactivating unrelated blocked work
  - result:
    - scheduler now detects executive pause transitions separately from readiness/focus gating
    - executive resume now only unblocks auto-resumable pause-blocked commitments and preserves blocked provenance
    - queue restoration skips work linked to non-active commitments
    - focused scheduler/executive regressions pass for fatigue recovery and focus-mode non-resume behavior

- `PR-020` Self-commitment capture and normalization
  - owner: Codex
  - status: completed
  - goal: turn assistant promises into compact durable commitments with normalized content and preserved raw provenance
  - result:
    - promise parsing now lives in a dedicated helper instead of staying embedded in `agent_loop.py`
    - self-commitments now store compact normalized content instead of whole assistant-turn blobs
    - matched conversational wording, normalization source, and capture confidence are preserved in metadata
    - capture path now mirrors normalized commitments into ToM and emits a somatic self-response event

- `PR-021` Conservative nightly commitment consolidation
  - owner: Codex
  - status: completed
  - goal: finish the unfinished Claude consolidation path without reactivating blocked commitments or inventing action
  - result:
    - blocked duplicate commitments now merge into blocked survivors instead of being silently reactivated
    - work creation stays limited to active survivors only
    - embedding clusters are refined with conservative lexical duplicate checks before merge, so same-shaped but distinct commitments stay separate
    - exact duplicate active commitments merge heuristically without unnecessary LLM arbitration, preserving linked work/task provenance and merge rationale
    - role-less historical turn episodes can still seed recovered commitments, and new turn episodes now record explicit role payloads

## Additional Completed Readiness Slices

- `PR-022` Chat-log commitment recovery on the real memory model
  - owner: Codex
  - status: completed
  - goal: make nightly chat-log backfill depend on real episode/session structure instead of brittle assumptions, with stronger normalization and duplicate checks
  - result:
    - chat-log recovery now builds normalized candidates from real turn episodes grouped by session instead of dumping raw recent turn text
    - recovered candidates preserve source session, source episode, source sentence, previous user turn, and role provenance in commitment metadata
    - in-session duplicate candidate promises collapse before LLM review, and recovered commitments dedupe conservatively against existing tracked commitments
    - historical role-less assistant turns can still be recovered conservatively, while new turn episodes now carry explicit role payloads for future cycles

- `PR-023` Commitment coupling to work, workspace, and schedules
  - owner: Codex
  - status: completed
  - goal: make durable commitments materially influence execution selection instead of only existing as stored records
  - result:
    - user-facing promise-backed work now receives a lightweight execution bias when restored or enqueued
    - executive dequeue order now respects promise/user-facing bias before raw promotion score
    - executive workspace scoring now boosts active commitment-linked work and surfaces user-facing commitment work with operator affinity
    - targeted tests prove promise-backed work moves ahead of unrelated background work without bypassing executive capacity or safety controls

- `PR-024` Promise-keeping integration across somatic, musubi, and ToM
  - owner: Codex
  - status: completed
  - goal: bind promise-keeping more strongly to inner-state systems so the agent prioritizes follow-through in a way that reflects bond, fatigue, and self-model coherence
  - result:
    - somatic modulators now expose promise-followthrough guidance so fatigue and tension raise promise salience without forcing immediate resumption
    - relational musubi and trust now provide a bounded promise-priority nudge for user-facing commitments
    - ToM now interprets unresolved self-commitments into a follow-through signal that distinguishes resume-now, acknowledge-delay, and repair-trust pressure
    - the context builder now injects pending user-facing commitments and explicit delay/repair guidance into the system prompt when relevant
    - executive workspace rebuild now keeps deferred user-facing commitments visible during pause states instead of letting them disappear from focus, and intervention no longer treats non-task focus as deletable work

- `PR-025` Commitment and consolidation observability
  - owner: Codex
  - status: completed
  - goal: make the operator able to see what commitments exist, why they changed, and how consolidation touched them
  - result:
    - commitments now expose lifecycle/provenance snapshots across executive, operations, chat-context, and runtime workflow surfaces
    - operator APIs now show source, raw excerpt, blocked reason, resume reason, merge rationale, chat-recovery provenance, and consolidation counters
    - the dashboard now surfaces blocked-vs-active state distinctions and latest consolidation outcomes directly in the commitment views
    - route hardening now drops non-dict consolidation payloads instead of wedging response serialization

- `PR-026` Memory dashboard atlas overhaul
  - owner: Codex
  - status: completed
  - goal: finish the memory-page work so the atlas, timeline, retrieval, and value panels operate like one operator-grade surface
  - result:
    - the memory page now has a compact health header showing compaction ratio, identity-core density, avg salience, and affect-lane visibility
    - the lower memory workbench is now tabbed across timeline, retrieval inspector, and memory-value evidence without forcing the atlas off-screen
    - atlas legend chips are now interactive filters for kind and affect lanes, with identity-core halos and richer point shapes in the scatter plot
    - control/detail panels are resizable, the atlas grows with viewport height, and hidden-all filter states clear the chart instead of leaving stale points onscreen

- `PR-027` Qualification and scenario proof
  - owner: Codex
  - status: completed
  - goal: prove the full promise-to-work lifecycle in one controlled scenario and in regression/qualification coverage
  - result:
    - added a bounded end-to-end qualification scenario covering promise capture, fatigue blocking, executive recovery, blocked dedup, chat-log recovery, and operator-surface inspection
    - documented the scenario in the qualification notes and long-scenario matrix so future runs can reuse one canonical proof artifact
    - hardened chat-log recovery to fall back to content-based candidate matching when LLM candidate ids drift from prompt ordering, preserving correct session/user-turn provenance
    - fixed Python 3.14 sqlite bootstrap compatibility for local stores and removed a duplicate `workspace_index` bootstrap/start path that was leaking background workers during qualification

- `PR-028` Documentation truth pass
  - owner: Codex
  - status: completed
  - goal: demote stale docs and reassert the current source-of-truth set after the commitment backend is corrected
  - result:
    - `AGENTS.md` now clearly distinguishes historical 2026-04-08 context from the 2026-04-15 live frontier
    - `CLAUDE.md` now points collaborators to the live status docs first and demotes stale phase-completion assumptions
    - `documentation-map.md` now warns which guidance files include historical context
    - the 2026-04-15 commitment handoff is explicitly marked as partially superseded historical context



## Earlier Completed Readiness And Capability Slices

- `PR-001` Qualification matrix and repeated weak-label reruns
  - status: completed
  - result: qualification reruns are now graduated instead of staying artificially sticky after stable successes. The remediation rollup classifies repeated clean outcomes as `watch_only`, `integrated_operator_workflow` returned `2/2 artifact_verified` in request `7ddf2492ba1946328ca6398e7b541fed`, and `kilocode_supervised_work` was reproduced, fixed in `workflow_supervise_session`, then confirmed with `2/2 artifact_verified` in request `b404b54a8f414e36a6f96d531708b6bf`. The remaining historical failing request (`0249235e74ae4b8382c83c76e30f8e91`) is retained as defect evidence, not as an active rerun obligation.

- `PR-003` Longer integrated day-to-day scenarios
  - status: completed
  - result: the long-scenario matrix is now backed by executed evidence across Scenarios 1-10 where needed, including auth-friction, loop-guard, promise continuity, and memory continuity. Scenario 9 specifically proved repeated-task memory reuse with grounded retrieval-value evidence and a second-session artifact that reused recovered project details without rediscovery.

- `PR-002` First-regular-use readiness board
  - status: completed
  - result: production-readiness status, deployment checklist, and long-scenario references are now synchronized with the current rerun history and executed scenario set, so the readiness board no longer lags behind the actual evidence frontier.

- `PR-015` SparkEvaluator (structured novelty filter)
  - status: completed
  - result: created `SparkEvaluator` in `opencas/daydream/spark_evaluator.py` with cosine distance, somatic alignment, relational alignment, and executive feasibility scoring. Wired into `DaydreamGenerator` and `AgentRuntime`. Added 8 tests covering all scoring dimensions and DaydreamGenerator integration; all pass.

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
