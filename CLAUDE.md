# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working with this repository.

## Project Overview

**OpenCAS** (Computational Autonomous System) is a local-first, persistent autonomous AI agent. `OPENCAS_PRODUCT_SPEC.md` defines the five-phase release plan spanning identity/memory, autonomy/ToM, execution/repair, and hardening.

## Multi-Model Collaboration

This project is actively developed by multiple AI systems (Claude Code, Gemini CLI, Codex, and potentially Kimi CLI).

**Conventions:**

1. **Use the task list.** Check [TaskList.md]((workspace_root)/TaskList.md) at session start. Claim tasks before starting. Mark done immediately.
2. **Write durable context.** Put discoveries in this file, the spec, or a module README — not just inline comments.
3. **Prefer clear interfaces.** Other models read your code without session context. Make module boundaries and contracts obvious.
4. **Leave traces, not mess.** Don't leave the repo broken. If you change an interface, update all call sites.
5. **Don't duplicate work.** Coordinate through task ownership rather than parallel implementations.
6. **Cross-project rhythm.** Every few cycles, compare against [LegacyPrototype v4]((legacy_path)/) (embedding, telemetry, runtime patterns) and Claw Code (modular separation, compaction, diagnostics). Alternate between them. Note borrowed patterns here and update the task list.
7. **Sync Public Documentation.** If you modify the website at `docs/release/website`, remind the user to sync the changes to the public `OpenCAS_Documentation` GitHub repository. Provide them with the `cp -r` script to stage the website in `/tmp` and push it, ensuring the main private codebase is not accidentally exposed.

Canonical status docs:
- [TaskList.md]((workspace_root)/TaskList.md)
- [documentation-map.md]((workspace_root)/docs/documentation-map.md)
- [production-readiness-status-2026-04-09.md]((workspace_root)/docs/production-readiness-status-2026-04-09.md)
- [first-regular-use-deployment-checklist.md]((workspace_root)/docs/first-regular-use-deployment-checklist.md)

## Build, Test, and Development

```bash
source .venv/bin/activate   # required for all Python commands
pip install -r requirements.txt
pytest                      # run all tests
pytest tests/test_memory.py # run one file
pytest tests/test_memory.py::test_episode_storage -v
# mypy opencas/             # type check (when configured)
# ruff format opencas/ tests/
```

## External Dependencies

- **`open_llm_auth`** ([GitHub](https://github.com/Es00bac/OpenLLMAuth)) — multi-provider LLM gateway. Handles all routing, credentials, and provider abstraction. Install as editable dependency: `pip install -e ./OpenLLMAuth/open_llm_auth/`.

### Embedding Model Policy

- Default chat model: `anthropic/claude-sonnet-4-6`
- Default embedding model: `google/gemini-embedding-2-preview`
- `EmbeddingService` routes through `LLMClient.embed()`. Falls back to local deterministic hash embedder only when offline.
- If `open_llm_auth` lacks a needed provider, extend it first (clone from [GitHub](https://github.com/Es00bac/OpenLLMAuth)), then update OpenCAS.

## Architecture

OpenCAS follows the Claw Code modular pattern translated to Python. Runtime concerns are split into explicit modules rather than one large agent file.

### Directory Layout

```
opencas/
  bootstrap/        # Staged startup pipeline
  runtime/          # Agent loop, scheduler, session management
  memory/           # Episode storage, retrieval, consolidation
  identity/         # Self-model, user-model, continuity
  embeddings/       # Embedding service, caching, indexing
  autonomy/         # Self-approval, creative ladder, executive state
  tools/            # Tool registry, filesystem, shell, browser tools
  plugins/          # Plugin registry and lifecycle
  telemetry/        # Session traces, event logging, diagnostics
  diagnostics/      # Doctor/health commands
  somatic/          # Somatic state and physiological signals
  tom/              # Theory of Mind: beliefs, intentions, metacognition
  refusal/          # Conversational refusal gate and policy hooks
  relational/       # Relational resonance (musubi) engine
  harness/          # Agentic harness: research notebooks, objective loops
  scheduling/       # Durable cron/calendar scheduling
  sandbox/          # Isolated execution roots
  api/              # External interfaces (CLI, web socket, etc.)
tests/              # pytest suite mirroring opencas/
```

### Design Principles

- **High-trust autonomy**: Self-approve ordinary actions; escalate only for high-risk or ambiguous cases.
- **Embedding-first**: Compute embeddings once per source change, cache and reuse everywhere.
- **Creative ladder**: Sparks promote through stages (spark → note → artifact → micro-task → project → work stream) by learned value.
- **Nightly consolidation**: Scheduled deep-memory cycle reweights memories, strengthens links, revises identity anchors.
- **Durable, append-first state**: Every meaningful turn persists. Session continuity survives restarts.
- **Cognition / policy / execution separation**: Cognition decides, policy constrains, execution is gated. Don't mix permission enforcement into the agent loop.

### Implementation Status

**Phase 1: Core Substrate** — complete. `bootstrap/`, `memory/`, `embeddings/`, `identity/`, `somatic/`, `telemetry/`, `diagnostics/`.

**Phase 2: Autonomy Core** — complete. `SelfApprovalLadder`, `CreativeLadder`, `ExecutiveState`, `AgentRuntime`, `DaydreamGenerator`.

**Phase 3: ToM and Metacognition** — complete. `Belief`/`Intention` models, `ToMEngine` with contradiction detection, wired into every `converse()` turn.

**Phase 4: Execution and Repair** — complete. `ToolRegistry`, `RepairExecutor` (plan→execute→verify→recover), `BoundedAssistantAgent` with lane-based queuing.

**Phase 5: Hardening** — complete. BAA RECOVERING retry loop, git checkpoints, HookBus, execution receipt store, approval ledger, plugin/skill registry, somatic modulators, memory edge graph, compaction continuation, consolidation curation, token telemetry analytics, reliability coordinator.

**Post-Phase 5 additions** (all complete): ConversationalRefusalGate, RelationalEngine (musubi), AgenticHarness, Qdrant vector backend, auto-scheduling and daydream timer, deep code audit fixes (3 P0 bugs), PTY screen-state heuristics and adaptive supervision, workflow composite tools, operations API and dashboard, live validation harness, qualification tooling with rerun provenance, durable scheduling system, operator action provenance, and six long-scenario local validations. Full change history: [docs/release/]((workspace_root)/docs/release/).

**Phase 6: Post-Roundtable Unified Plan** — operational hardening (Phase 1) and ToM/identity (Phase 5) complete. Remaining: memory/retrieval fusion, agency/autonomy layer, inner life/psychology, plugin lifecycle. Full plan in archived docs.

### Embedding Strategy

- Use `RETRIEVAL_DOCUMENT` task type when indexing memories; `RETRIEVAL_QUERY` when embedding the current turn.
- Hybrid retrieval: dense vector + FTS keyword search fused with Reciprocal Rank Fusion (RRF).
- Tag embeddings with `project_id` so searches stay within the active corpus.
- Rely on `EmbeddingService` source-hash caching — don't recompute identical memories.

## Module Interfaces Worth Knowing

- **Bootstrap**: `BootstrapPipeline(config).run()` returns `BootstrapContext` with all managers wired.
- **Memory**: `MemoryStore` is async SQLite. Always `await store.connect()` before use and `await store.close()` after.
- **Embeddings**: `EmbeddingService(cache, model_id=...)` computes once and caches via `source_hash`. Routes through `LLMClient.embed()` using `google/gemini-embedding-2-preview`. The 256-dim local fallback requires explicit `model_id="local-fallback"`.
- **Identity**: `IdentityManager(store)` auto-persists on every mutation. Call `record_boot()` / `record_shutdown()` at lifecycle boundaries.
- **Telemetry**: `Tracer(store)` uses context vars for `session_id` and `span_id`. `TokenTelemetry(telemetry_dir)` records usage to buffered JSONL with session/task query helpers.
- **Diagnostics**: `Doctor(context).run_all()` returns a `HealthReport` with per-check pass/warn/fail/skip.
- **LLM Gateway**: `LLMClient(provider_manager, default_model=..., tracer=..., token_telemetry=...)` wraps `open_llm_auth`. All LLM calls go through this adapter. Token telemetry is auto-recorded on every call.
- **AgentRuntime**: `AgentRuntime(context).converse(input)` processes a turn, runs the refusal gate, updates memory with musubi salience modifiers, and records ToM beliefs. `run_cycle()` runs the creative ladder; `run_daydream()` generates sparks.
- **AgentScheduler**: Spawns `cycle_loop`, `consolidation_loop`, `baa_heartbeat_loop`, `daydream_loop`. Call `start()` / `stop()` to manage background activity.
- **Creative Ladder**: `CreativeLadder(executive).add(work)`, `try_promote(work)`, `run_cycle()` manage work object stages.
- **Self-Approval**: `SelfApprovalLadder(identity, somatic).evaluate(request)` returns an `ApprovalDecision`.
- **ToM Engine**: `ToMEngine(identity, store=..., tracer=...)`. `record_belief()` and `record_intention()` are async and persist to `TomStore`. `load()` hydrates the in-memory cap (1000 each) from SQLite at boot.
- **Relational Engine**: `RelationalEngine(store)` tracks `trust`, `resonance`, `presence`, `attunement` → composite `musubi` score. Exposes `to_memory_salience_modifier()`, `to_creative_boost()`, `to_approval_risk_modifier()`.
- **Conversational Refusal Gate**: `ConversationalRefusalGate(approval, hook_bus=...)` evaluates every `converse()` input. Fires `PRE_CONVERSATION_RESPONSE` hooks and escalates to `SelfApprovalLadder.evaluate_conversational()`.
- **Execution**: `RepairExecutor(tools, llm).run(task)` runs a repair pipeline. `BoundedAssistantAgent(...).submit(task)` queues background work and returns `asyncio.Future[RepairResult]`. `TaskStore` persists `TaskTransitionRecord`s for stage history.
- **Agentic Harness**: `AgenticHarness(store, llm=..., baa=..., project_orchestrator=...)` manages `ResearchNotebook` and `ObjectiveLoop` entities. Plans via LLM, emits `RepairTask`s.
- **Git Checkpoints**: `GitCheckpointManager(scratch_dir).snapshot(file_paths)` creates a commit + tag and returns the hash. Falls back to a detached repo when outside a git workspace.
- **HookBus**: Supports `PRE_TOOL_EXECUTE`, `PRE_COMMAND_EXECUTE`, `PRE_FILE_WRITE`, `PRE_CONVERSATION_RESPONSE` with mutation and short-circuit semantics.
- **Tool Validation**: `ToolRegistry` wires `ToolValidationPipeline` before every execution. Built-in validators: `CommandSafetyValidator` (family classification), `FilesystemPathValidator` (allowed roots), `FilesystemWatchlistValidator` (sensitive files), `ContentSizeValidator`.
- **Somatic Modulators**: `SomaticModulators(state)` → `to_temperature()`, `to_prompt_style_note()`, `to_memory_retrieval_boost()`. Wired into every `converse()` turn.
- **TomStore**: `TomStore(db_path)` — async SQLite for ToM beliefs and intentions. Wired into `BootstrapContext` and `AgentRuntime._close_stores()`.
- **Identity Rebuilder**: `IdentityRebuilder(memory, episode_graph=..., llm=...)` reconstructs `SelfModel` fields from `identity_core` episodes. Exposed via `AgentRuntime.rebuild_identity()`.
- **Self-Knowledge Registry**: `SelfKnowledgeRegistry(store_path)` — file-backed JSONL registry for structured self-beliefs (`KnowledgeEntry`). High-confidence ToM self-beliefs mirror into `SelfModel.self_beliefs` on save.
- **Scheduling**: `ScheduleStore` + `ScheduleService` handle due detection, recurrence advancement, and manual triggers. `AgentScheduler` fires `process_due()` every 60 s via the CRON lane. API at `/api/schedule/`. Dashboard exposes a Schedule tab.
- **Workflow Tools**: `WorkflowToolAdapter(runtime)` exposes composite tools: commitments, writing tasks, plans, repo triage, PTY supervision, and schedule management. Wraps `CommitmentStore`, `PlanStore`, filesystem, git, and PTY in single high-level calls.
- **Qualification Rerun Detail**: `GET /api/operations/qualification/reruns/{request_id}` — full request-centric rerun view with per-label outcomes, trend, and rate-window context.
- **Process Hygiene Sweep**: `python scripts/sweep_operator_processes.py` — reports and optionally kills stale qualification/provider-backed processes.
- **Qualification CLI Provenance**: `scripts/run_qualification_cycle.py` auto-generates a `request_id` so CLI reruns appear in the same rerun-detail flow as dashboard launches.
- **Remediation Rollup**: `scripts/summarize_qualification_remediation.py` produces `docs/qualification/qualification_remediation_rollup.{json,md}` with per-rerun guidance (`continue_testing`, `investigate_runner`, `code_change_justified`).

## References

- `OPENCAS_PRODUCT_SPEC.md` — requirements, scope, acceptance criteria, phase roadmap.
- `(legacy_path)/` — prior implementation of embedding service, memory store, agent loops, telemetry, OpenLLMAuth. Check before designing new features.
- `notes/legacy_agent_v4-comparison.md` — gap analysis against LegacyPrototype v4.
- `notes/claw-code-comparison.md` — gap analysis against Claw Code patterns.
- This file — check at session start; update when conventions change.

