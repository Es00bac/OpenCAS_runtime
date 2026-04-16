# AGENTS.md — OpenCAS Project Context

> **For:** Codex, Claude, Gemini, Kimi, and any future collaborator.  
> **Written:** 2026-04-08 after completion of the unified 4-plan enhancement suite.  
> **Refreshed:** 2026-04-15 for the current commitment-continuity frontier.  
> **Primary human intent:** The user wants OpenCAS to be a genuinely autonomous, learning partner—not a tool that superficially appears busy. Every line of code must earn its keep.

> **Current-use rule:** This file still contains historical context from the 2026-04-08 phase. For current execution state and active work, use [TaskList.md](TaskList.md), [docs/documentation-map.md](docs/documentation-map.md), and [docs/opencas-continuation-program-2026-04-15.md](docs/opencas-continuation-program-2026-04-15.md) first.

---

## 1. What This Project Is

**OpenCAS** (Open Computational Autonomous System) is a local-first, persistent autonomous AI agent written in Python. It is not a chatbot wrapper. It is designed to:

- Remember across sessions via SQLite-backed episodic memory and semantic embeddings.
- Self-approve ordinary actions and escalate only for genuinely high-risk or ambiguous cases.
- Learn from experience (usage feedback, belief/intention tracking, skill acquisition).
- Maintain a continuously running creative/execution loop (daydreaming, task decomposition, bounded assistant execution).
- Judge safety dynamically and ask for help only in extreme cases.

The human user and the CAS agent are **partners**. The primary user of the system is the agent itself running in this environment. It must use judgment gained over time, trust metrics, and its own history.

---

## 2. Historical 2026-04-08 Enhancement Suite

The user authorized a comprehensive 4-plan autonomous pass. Every phase was implemented, tested, and committed.

### Phase A — OpenLLMAuth Configurability
**Goal:** Break the global singleton so OpenCAS and OpenBulma-v4 can share `open_llm_auth` without collision, and make `kimi` / `kimi k2.5` first-class choices.

**Done:**
- `ProviderManager` now accepts `config_path` and `env_path` constructor args.
- `load_config()` reads optional per-project config/env files with proper precedence (process env > `env_path` > `config_path` > `~/.open_llm_auth/config.json`).
- Kimi heuristic added to `_infer_provider_for_model_id`.
- OpenCAS `BootstrapConfig` extended with `provider_config_path` and `provider_env_path`.
- OpenCAS `BootstrapPipeline` passes these into `ProviderManager`.

### Phase B — OpenCAS Memory Retrieval "Inner-Mind"
**Goal:** Port OpenBulma-v4's rich multi-signal scoring into OpenCAS so retrieval feels organic.

**Done:**
- New module: `opencas/context/resonance.py` with `compute_emotional_resonance`, `compute_temporal_echo`, `compute_reliability_score`, `compute_edge_strength`.
- `SomaticModulators` now returns a `RetrievalAdjustment` (recency, salience, emotional resonance, temporal echo, graph bonuses) consumed by `MemoryRetriever`.
- `MemoryRetriever` constructor now accepts `somatic_manager` and `relational_engine`.
- Retrieval fusion formula updated: semantic (0.30), keyword (0.20), recency (0.15), salience (0.10), graph (0.10), emotional resonance (0.08), temporal echo (0.04), reliability (0.03), plus somatic bonus terms and reliability scaling.
- Graph expansion now uses `compute_edge_strength()` instead of raw `edge.confidence`.
- Relational musubi modifier applied when `relational_engine` is present.
- Wired in `ContextBuilder` and `BootstrapPipeline`.

### Phase B — OpenLLMAuth Advanced Web GUI
**Goal:** Upgrade the vanilla JS dashboard into a modern admin interface.

**Done:**
- New SQLite `UsageStore` (`usage_records` schema) in `open_llm_auth/src/open_llm_auth/server/usage_store.py`.
- Usage interception in `open_llm_auth/src/open_llm_auth/server/routes.py` for chat completions, embeddings, and universal API (timed for non-streaming, estimated for streaming).
- Dashboard template: `open_llm_auth/src/open_llm_auth/server/templates/dashboard.html` (~45KB) using `htmx + Alpine.js + Chart.js`.
- `config_routes.py` extended with profile snapshot endpoints (save, activate, import, export) and usage aggregation endpoints.
- `main.py` wired to serve `dashboard.html` from `/` via Jinja2.
- Tests: `test_usage_store.py` (4 tests), `test_config_routes_dashboard.py` (8 tests), `test_main_templates.py` (1 test). All pass.

### Phase C — OpenCAS Comprehensive Dashboard
**Goal:** Expose every subsystem (config, diagnostics, chat, memory graphs, embedding visualizations) through a web UI.

**Done:**
- New API routers mounted in `opencas/api/server.py`:
  - `/api/config` — redacted `BootstrapConfig`, provider profiles
  - `/api/monitor` — health checks, BAA queue, embedding latency, event samples
  - `/api/chat` — session list, history, traces, active plan
  - `/api/memory` — paginated episodes, graph neighbors, stats, search, 2D embedding projection (UMAP/PCA/random)
- New directory: `opencas/api/routes/{config,monitor,chat,memory}.py`.
- Dashboard SPA: `opencas/dashboard/static/index.html` + `css/app.css` (htmx + Alpine.js + Chart.js).
- `SessionContextStore` extended with `list_session_ids()`.
- `MemoryStore` extended with `get_stats()`.
- Tests: `tests/test_dashboard_api.py` (4 tests, all pass).

---

## 3. Critical User Directives (Non-negotiable)

1. **No "bust work."** The user explicitly said: *"I also do not want you to do bust work, just to appear like you are working for a long time, everything you do has to be meaningful. If this is less complicated than I think it is, stop when you are done."*
   - **Rule:** Do not generate busy-work, excessive comments, speculative abstractions, or premature generalizations. If a task is done, stop.

2. **Agent-as-primary-user.**
   - The agent must operate with judgment gained from experience, trust in the human user, and its own history.
   - It should acquire, reuse, and adapt skills abstractly.
   - Human intervention is only for extreme cases.

3. **Safety through learning.**
   - The agent should learn what is safe and have good practices for when it is unsure.
   - The `SelfApprovalLadder`, `HookBus`, `CommandSafetyValidator`, and `ConversationalRefusalGate` exist for this purpose. Do not bypass them.

4. **Keep generated work contained.**
   - Agent-created notes, project scaffolds, and workflow-generated artifacts should live under the project-local `workspace/` directory, not as ad hoc files scattered across the repo root or host environment.
   - If a path is optional, default it into that managed workspace.
   - If a caller supplies a path, reject escapes outside the managed workspace unless the task explicitly requires broader host access and the relevant policy is updated.
   - If the managed workspace policy changes, repair persisted Chronicle/workspace references in SQLite-backed state with `scripts/repair_workspace_references.py` so memory and workflow surfaces do not keep stale paths alive.
   - Repo-local maintenance scripts should derive their BootstrapConfig through `opencas.maintenance.build_repo_local_bootstrap_config()` so workspace/state defaults stay consistent across utilities.

5. **Git discipline.**
   - Both `.` and `../open_llm_auth` are now git repositories.
   - The user specifically requested this: *"make it a git repo"* and *"all projects need to have a git repo, Local"*.
   - Create commits for meaningful milestones. Do not leave repos in a broken state.

6. **Do not run the Bulma importer.**
   - `opencas/legacy/importer.py` and `AgentRuntime.import_bulma()` are **forbidden** until this directive is removed. Running it causes unrecoverable data loss.

7. **Code organization discipline.**
   - Future agentic workflows should keep the workspace organized and avoid growing large god-object files.
   - Prefer extracting coherent service/helper modules when a route or runtime file starts carrying multiple responsibilities.
   - Reuse one implementation path for shared behavior so fixes land once.
   - Do not replace one god object with another generic helper blob; extracted modules should have a narrow, durable purpose.

---

## 4. Workspace Layout

```
./          # Main OpenCAS repo (Python)
  opencas/
    api/                    # FastAPI server + new dashboard routers
    bootstrap/              # BootstrapPipeline, BootstrapContext, BootstrapConfig
    context/                # SessionContextStore, ContextBuilder, MemoryRetriever, resonance
    memory/                 # MemoryStore, EpisodeGraph, models
    embeddings/             # EmbeddingService, QdrantVectorBackend, HnswVectorBackend
    runtime/                # AgentRuntime, AgentScheduler
    autonomy/               # SelfApprovalLadder, CreativeLadder, ExecutiveState
    somatic/                # SomaticManager, SomaticModulators
    relational/             # RelationalEngine (musubi)
    tom/                    # ToMEngine, TomStore
    diagnostics/            # Doctor, HealthMonitor
    dashboard/              # New static SPA (htmx + Alpine.js + Chart.js)
    ...
  tests/                    # pytest suite
  requirements.txt
  CLAUDE.md                 # Multi-model collaboration guidelines
  AGENTS.md                 # This file

../open_llm_auth/    # Editable dependency, now a git repo
  src/open_llm_auth/
    server/
      templates/dashboard.html
      static/
      usage_store.py
      config_routes.py
      routes.py
    auth/manager.py
    config.py
  tests/
```

---

## 5. Build / Test / Run Commands

```bash
# Activate venv (required for all Python commands)
source .venv/bin/activate

# Run tests
pytest

# Run specific files
pytest tests/test_dashboard_api.py -v
pytest tests/test_memory.py -v

# Run OpenCAS with server
python -m opencas --with-server
```

---

## 6. Current State (As of 2026-04-15)

- The 2026-04-08 enhancement suite is historical context, not the current frontier.
- The bounded cleanup program has now reached a stable stopping point, after advancing well past the original foundation slices.
- Cleanup slices completed in this pass family include:
  - `PR-031` through `PR-035` foundation cleanup and documentation normalization
  - `PR-036` major `AgentRuntime` decomposition work across conversation, cycle, lifecycle, reflection, episodic, maintenance, telegram, and tool-registration seams
  - `PR-069` through `PR-075` daydream store, qualification support, memory store, retriever, and consolidation-engine helper extraction
  - `PR-076` through `PR-079` tool-registration, workflow-adapter, and bootstrap-pipeline support extraction
  - `PR-082` through `PR-086` retriever fusion/MMR, bootstrap-context/service/store assembly, and config control-plane extraction
- Notable current cleanup state:
  - `opencas/api/routes/operations.py` is a thin assembly layer over focused helper modules instead of a route god object
  - `opencas/bootstrap/tui.py` is the app shell and screen registry, with setup/profile/runtime/state/bootstrap/widgets extracted into dedicated modules
  - `opencas/runtime/agent_loop.py`, `opencas/context/retriever.py`, `opencas/memory/store.py`, and `opencas/consolidation/engine.py` are all materially smaller than they were at the start of the cleanup program, with retriever fusion/MMR now living in dedicated helper modules
  - `opencas/tools/adapters/workflow.py`, `opencas/bootstrap/pipeline.py`, and `opencas/api/routes/config.py` are now thin facades over focused helper modules instead of mixed operator/bootstrap/control-plane slabs
  - workspace containment and persisted Chronicle/workspace reference repair are part of the standing maintenance policy, not optional follow-up work
- Future structural risk is now mostly about preserving the extracted seams during feature work rather than breaking down any remaining god-object route, bootstrap, workflow, or retriever monoliths.
- The authoritative current task state is in [TaskList.md](TaskList.md).
- The authoritative current cleanup plan is in [docs/opencas-cleanup-program-2026-04-15.md](docs/opencas-cleanup-program-2026-04-15.md).
- The working tree is expected to be kept clean at milestone boundaries; do not assume older docs describing long-lived dirty WIP are still accurate.
- OpenLLMAuth is still the editable dependency at `../open_llm_auth/`.

---

## 7. How to Continue Working

If you are picking up this project, here is the suggested order of operations:

1. **Read `CLAUDE.md`** for multi-model collaboration conventions.
2. **Check [TaskList.md](TaskList.md)** to see what is actually pending or in progress.
3. **Check [docs/documentation-map.md](docs/documentation-map.md)** so you do not anchor on stale docs.
4. **Run targeted pytest** for the slice you are changing before widening scope.
5. **Reference the full session transcript** (see below) only when the current docs are insufficient.
6. **Do not create new abstractions** unless three similar patterns already exist and the user explicitly asked for cleanup.
7. **Use agent-oriented comments.** Leave short comments or module docstrings at real orchestration seams, invariants, or contract boundaries so future agents can navigate intent without re-deriving it.
8. **When editing:** prefer the current source-of-truth docs over older handoffs. Verify tests pass after changes. Commit meaningful milestones.

---

## 8. Session Artifacts & Full History

This AGENTS.md is a summary. The most relevant transcripts are:

- historical 2026-04-08 substrate / hardening transcript:

```
[private transcript path removed in public mirror]
```

- current 2026-04-14 through 2026-04-15 commitment-continuity transcript:

```
[private transcript path removed in public mirror]
```

If you need to understand *why* a specific file was changed, what test failed repeatedly before passing, or the exact wording of a user constraint, use the transcript that matches the active frontier rather than defaulting to the older one.

---

## 9. Key Contacts / References

- **Spec:** `OPENCAS_PRODUCT_SPEC.md` (authoritative requirements).
- **Comparison repos:** `../openbulma-v4/` (Bulma patterns), `claw-code` patterns in spec §16.
- **LLM Gateway:** `open_llm_auth` (editable install in same workspace).
- **Default models:**
  - Chat: `anthropic/claude-sonnet-4-6` (override via `default_llm_model`)
  - Embeddings: `google/gemini-embedding-2-preview`

---

## 10. TL;DR for Codex

- OpenCAS is a persistent autonomous agent.
- The 2026-04-08 enhancement suite is historical context, not the live execution state.
- The current frontier is promise-keeping: commitment capture, pause/resume correctness, conservative consolidation, and doc cleanup.
- The user hates busy-work and wants the agent to act as a partner, not a tool.
- **Do not run the Bulma importer.**
- When in doubt, read `TaskList.md`, `documentation-map.md`, and the 2026-04-15 continuation program before older handoffs.
