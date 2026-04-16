# OpenCAS: Architecture, Subsystems, and Comparative Assessment

**Date:** 2026-04-08  
**Scope:** Complete architectural survey of OpenCAS, comparison against OpenBulma-v4, and assessment of design maturity and integration quality.

---

## 1. What OpenCAS Is

OpenCAS (the **Open Computational Autonomous System**) is a local-first, persistent autonomous AI agent written in Python/asyncio. Its core thesis is simple but ambitious: *the agent should not wait to be told what to do next if it already knows enough to act.*

Unlike a stateless chatbot, OpenCAS maintains:
- **Episodic memory** of every conversation and action
- **Semantic memory** of consolidated knowledge
- **A self-model** that evolves over time (identity, values, goals)
- **A theory of mind** tracking beliefs and intentions about the user
- **A somatic state** (arousal, fatigue, tension, valence) that modulates cognition
- **A relational field** (`musubi`) measuring trust and attunement with the user
- **Creative autonomy** that generates daydreams, promotes them into tasks, and executes them in the background

OpenCAS is designed as a *living computational collaborator* rather than a disposable assistant.

---

## 2. What OpenCAS Is Based On

OpenCAS draws heavily from two prior projects:

### 2.1 OpenBulma-v4 (TypeScript/Node.js)

OpenBulma-v4 is the most direct ancestor. It was a working autonomous agent with:
- WebSocket and HTTP chat interfaces
- A `BulmaAssistantAgent` (BAA) for background task execution
- `MemoryFabric` with Qdrant-backed semantic search
- EventBus-driven reactive coordination
- Token telemetry and safety policy

Many OpenCAS subsystems are direct Python translations or evolutions of Bulma patterns:
- `BoundedAssistantAgent` ← `BulmaAssistantAgent`
- `ToolValidationPipeline` ← Bulma safety/validation validators
- `EventBus` / `HookBus` ← Bulma's typed event emitter
- `TokenTelemetry` ← Bulma's JSONL telemetry
- `NightlyConsolidationEngine` ← Bulma's memory consolidation cycle
- `GitCheckpointManager` ← Bulma's checkpoint/rollback system

### 2.2 Claw Code (Rust)

Claw Code provided the *architectural philosophy*:
- **Explicit bootstrap as composition root** rather than ad-hoc wiring
- **Separate runtime concerns** into modules (`runtime/`, `tools/`, `plugins/`, `telemetry/`)
- **Session durability** with append-first persistence
- **Compaction** as first-class context management
- **Permission enforcement separate from cognition**
- **Sandboxing and filesystem boundaries explicit**
- **Doctor/health-check pattern** for runtime diagnostics

The comparison notes in `notes/claw-code-comparison.md` and `notes/openbulma-v4-comparison.md` explicitly map these borrowings.

---

## 3. Directory Structure and Subsystem Overview

```
opencas/
  bootstrap/        # Staged startup pipeline (Claw Code pattern)
  runtime/          # AgentRuntime, scheduler, readiness
  memory/           # Episode store, semantic memory, episode graph
  embeddings/       # Embedding service, cache, Qdrant backend
  identity/         # Self-model, user-model, continuity
  autonomy/         # Self-approval, creative ladder, executive state
  tools/            # Tool registry, adapters, validation pipeline
  plugins/          # Plugin lifecycle, skill registry, manifest loader
  execution/        # BAA, RepairExecutor, lanes, reliability
  telemetry/        # Tracer, token telemetry, event bus
  diagnostics/      # Doctor, health monitor
  somatic/          # Physiological state and appraisal
  tom/              # Theory of mind engine
  relational/       # Relational resonance (musubi)
  refusal/          # Conversational safety gate
  harness/          # Research notebooks and objective loops
  context/          # Prompt context assembly
  consolidation/    # Nightly deep-memory cycle
  compaction/       # Conversation compaction
  sandbox/          # Execution boundaries (Docker, workspace)
  api/              # LLM gateway, FastAPI server
```

---

## 4. Subsystem-by-Subsystem Deep Dive

### 4.1 `bootstrap/` — The Composition Root

**Key files:** `pipeline.py`, `config.py`

**What it does:**
`BootstrapPipeline.run()` performs explicit, staged initialization of the entire agent substrate. It constructs a `BootstrapContext` dataclass that holds every manager, store, and registry.

**Boot stages (simplified):**
1. Telemetry & tracing
2. Event bus & hook bus
3. Identity & continuity
4. Memory backend (SQLite)
5. Task, work, commitment, portfolio stores
6. LLM gateway (`LLMClient` via `open_llm_auth`)
7. Embedding service + optional Qdrant backend
8. Sandbox configuration
9. Somatic state & relational engine
10. Plugin/skill registry + lifecycle manager
11. Project orchestrator & harness
12. ToM store & diagnostics monitor
13. Readiness state machine

**Integration:**
Everything downstream depends on `BootstrapContext`. `AgentRuntime` receives it and coordinates all subsystems through it.

**Design quality:** This is one of OpenCAS's strongest areas. The staged pipeline is testable, explicit, and far cleaner than OpenBulma-v4's monolithic `src/index.ts` imperative wiring.

---

### 4.2 `runtime/` — The Central Coordinator

**Key files:** `agent_loop.py`, `scheduler.py`, `readiness.py`, `daydream.py`

**What it does:**
`AgentRuntime` is the orchestrator. At ~1,767 lines, it is arguably a "god object," but it serves a necessary purpose: it mediates between conversation, memory, tools, creative cycles, execution, and autonomous scheduling.

**Key methods:**
- `converse(user_input)` — full user turn: refusal gate, episode recording, context building, tool loop, goal extraction, ToM belief recording, relational interaction recording
- `run_daydream()` — generates reflections/sparks when idle, evaluates them, routes to creative ladder
- `run_cycle()` — promotes work objects, drains executive queue to BAA, evaluates intervention policy, runs harness cycle
- `execute_tool(name, args)` — self-approval gating + tool execution + post-execution somatic bumps

**Scheduler:**
`AgentScheduler` spawns four background lanes (`CHAT`, `BAA`, `CONSOLIDATION`, `CRON`) via `LaneManager`. Each lane has worker coroutines on independent intervals.

**Integration:**
- Calls `MemoryStore.save_episode()` after every turn
- Calls `ContextBuilder.build()` to assemble LLM prompts
- Calls `ToolUseLoop.run()` for ReAct-style tool execution
- Calls `CreativeLadder.run_cycle()` and `ExecutiveState.dequeue()`
- Subscribes to `BaaCompletedEvent` for goal resolution

**Concerns:**
`AgentRuntime` coordinates too many concerns. While better than OpenBulma-v4's `IntegrationHub` (which was ~4,800 lines of callback injection), the centralization creates a single point of churn.

---

### 4.3 `memory/` and `embeddings/` — Persistent Semantic Infrastructure

**Key files:**
- `memory/store.py` — async SQLite episode store
- `memory/fabric/graph.py` — `EpisodeGraph` for BFS traversal
- `memory/fabric/builder.py` — `FabricBuilder` rebuilds edges
- `embeddings/service.py` — `EmbeddingService` with cache
- `embeddings/qdrant_backend.py` — optional Qdrant acceleration

**What it does:**
`MemoryStore` persists episodes (conversational turns, tool executions, creative events), semantic `Memory` records, compactions, and typed `episode_edges`. It includes an FTS5 virtual table for lexical search.

`EmbeddingService` computes embeddings once per source text, caches them forever via `source_hash` deduplication, and supports three search tiers:
1. **Qdrant** (optional ANN search with `model_id`/`project_id` filtering)
2. **SQLite brute-force** (loads all vectors, computes cosine similarity in Python/numpy)
3. **Lexical fallback** (Jaccard overlap for empty vector results)

`EpisodeGraph` supports BFS walks and subgraph extraction, used during consolidation and identity rebuilding.

**Integration:**
- `AgentRuntime._record_episode()` embeds and saves every turn
- `ContextBuilder` / `MemoryRetriever` use semantic search to recall relevant memories
- `NightlyConsolidationEngine` calls `FabricBuilder.rebuild()` to reconstruct edges

**Scaling concern:** When Qdrant is unavailable, the SQLite brute-force scan is **O(N)**. For thousands of episodes, this degrades linearly. The codebase *does* have a `QdrantVectorBackend`, but it is optional. There is no local ANN index (e.g., faiss, hnswlib) as a middle ground.

**Question answered:** *Does OpenCAS expect Qdrant to be mandatory in production?*  
**Answer:** No. The design treats Qdrant as an optional acceleration layer. The fallback brute-force scan works for small-scale deployments but will not scale to production memory volumes without Qdrant or another ANN backend.

---

### 4.4 `tools/` — Capability Registry with Policy Hooks

**Key files:** `registry.py`, `validation.py`, `loop.py`, `loop_guard.py`, `adapters/*.py`

**What it does:**
`ToolRegistry` registers ~25 core tools (filesystem, shell, edit, search, web, Python REPL, LSP, process, agent, plan mode, etc.). Each tool has:
- A JSON schema for LLM function calling
- An `ActionRiskTier` (`READONLY`, `WORKSPACE_WRITE`, `SHELL_LOCAL`, `NETWORK`, `EXTERNAL_WRITE`, `DESTRUCTIVE`)
- An adapter implementing the actual logic

`ToolValidationPipeline` runs validators before execution:
- `CommandSafetyValidator` — parses commands, blocks dangerous families
- `SmartCommandValidator` — learnable grey-list for unknown commands
- `FilesystemPathValidator` — enforces allowed roots
- `FilesystemWatchlistValidator` — blocks sensitive paths
- `ContentSizeValidator` — enforces `max_write_bytes`

`ToolUseLoop.run()` implements a ReAct loop with concurrent tool call batching (READONLY tools run in parallel; mutating tools run serially). `ToolLoopGuard` prevents infinite loops (max 16 rounds) and detects identical-call repetition.

**Integration:**
- `AgentRuntime._register_default_tools()` populates the registry at init
- `AgentRuntime._register_skills()` adds plugin-provided tools
- `execute_tool()` runs self-approval before calling `ToolRegistry.execute_async()`
- `HookBus` provides `PRE_TOOL_EXECUTE`, `PRE_COMMAND_EXECUTE`, `PRE_FILE_WRITE`, and `PRE_CONVERSATION_RESPONSE` policy hooks

**Comparison note:** OpenCAS has a more mature tool governance model than OpenBulma-v4, with explicit risk tiers and validation pipelines. However, OpenBulma-v4 had **MCP (Model Context Protocol)** integration, which OpenCAS currently lacks.

---

### 4.5 `plugins/` — Manifest-Driven Extension Loading

**Key files:** `lifecycle.py`, `loader.py`, `store.py`, `registry.py`, `models.py`

**What it does:**
`PluginLifecycleManager` handles install, enable, disable, and uninstall of plugins. `PluginStore` persists state to SQLite. `load_plugin_from_manifest()` reads `plugin.json` files, imports entrypoint modules, registers skills and hooks, and extracts `on_load`/`on_unload` lifecycle functions.

Plugins can declare:
- `id`, `name`, `description`, `version`
- `skills` — skill entrypoints
- `hooks` — `{hook_name, handler, priority}` definitions
- `dependencies` — other plugin IDs required
- `entrypoint` — Python module to import

**Integration:**
- Bootstrap loads builtins from `opencas/plugins/skills/` and user plugins from `~/.opencas/plugins/`
- `AgentRuntime` exposes `install_plugin()`, `uninstall_plugin()`, `enable_plugin()`, `disable_plugin()`
- `execute_tool()` blocks tools from disabled plugins via `_plugin_tools` ownership mapping

**Recent fixes:** Dependency validation during install/enable, kwargs validation in `TypedHookRegistry`, and correct preservation of manifest plugin tools at runtime were all recently corrected.

---

### 4.6 `execution/` — Background Task Engine

**Key files:** `baa.py`, `executor.py`, `lanes.py`, `reliability.py`, `git_checkpoint.py`, `store.py`

**What it does:**
`BoundedAssistantAgent` (BAA) manages background task execution using lane-based concurrency. It supports dependency holds, futures for results, and auto-resume of pending tasks from `TaskStore` on boot.

`RepairExecutor` executes repair tasks through explicit phases:
`DETECT → SNAPSHOT (git checkpoint) → PLAN → EXECUTE → VERIFY → POSTCHECK`

It includes:
- **Git checkpoints** (`GitCheckpointManager`) for rollback on failure
- **Convergence guards** that hash execution output and detect non-improving loops
- **Retry budgeting** with exponential backoff up to 10 recovery attempts

`ReliabilityCoordinator` listens to `BaaCompletedEvent` and emits `BaaPauseEvent` when failure rates spike above a threshold.

**Integration:**
- `AgentRuntime.submit_repair(task)` queues tasks
- `AgentScheduler` runs a BAA heartbeat loop
- `EventBus` notifies subscribers on task completion

**Question answered:** *How does BAA recover from a pause?*  
**Answer:** It doesn't, automatically. `ReliabilityCoordinator` emits `BaaPauseEvent`, but **no code in the BAA subscribes to this event** to actually pause execution. The BAA has no pause handler. Recovery would require operator intervention or a future feature to listen for `BaaPauseEvent` and temporarily stop dequeuing tasks.

**Question answered:** *Is there duplicate LaneManager in BAA and scheduler?*  
**Answer:** Not exactly. `BoundedAssistantAgent` has its own `LaneManager` for task execution lanes. `AgentScheduler` has a separate `LaneManager` for background scheduling loops (`cycle_loop`, `consolidation_loop`, `baa_heartbeat_loop`, `daydream_loop`). They serve different purposes. This is a design choice, not a bug, though it means two queue systems exist.

---

### 4.7 `autonomy/` — Self-Approval and Creative Ladder

**Key files:** `self_approval.py`, `creative_ladder.py`, `executive.py`, `spark_router.py`, `workspace.py`

**What it does:**
`SelfApprovalLadder` evaluates every action request against trust history, somatic state, boundary checks, and relational modifiers, producing one of four approval levels.

`CreativeLadder` promotes `WorkObject`s through stages:
`SPARK → NOTE → ARTIFACT → MICRO_TASK → PROJECT_SEED → PROJECT → DURABLE_WORK_STREAM`

Promotion considers semantic similarity, goal relevance, daydream alignment, and musubi boost.

`ExecutiveState` manages a capacity-limited task queue (max 5), active goals, and current intention.

`InterventionPolicy` evaluates the executive workspace and can recommend launching background work, surfacing approvals, or retiring stalled focus items.

**Integration:**
- `AgentRuntime.converse()` runs self-approval before every tool execution
- `AgentRuntime.run_cycle()` runs the creative ladder and drains the executive queue
- High-scoring daydreams are routed through `SparkRouter` and may create commitments or portfolio clusters

---

### 4.8 `tom/` and `relational/` — Social Cognition

**Key files:** `tom/engine.py`, `relational/engine.py`

**Theory of Mind (`ToMEngine`):**
- Records beliefs about SELF and USER
- Tracks active intentions
- Runs metacognitive consistency checks after each turn
- Mirrors high-confidence self-beliefs into `SelfKnowledgeRegistry`

**Relational Engine (`RelationalEngine`):**
- Tracks four dimensions: `trust`, `resonance`, `presence`, `attunement`
- Computes composite `musubi` score
- Modulates memory salience, creative boosts, and self-approval risk appetite

**Integration:**
- `AgentRuntime.converse()` records a user belief and triggers consistency checks
- `AgentRuntime` records relational interactions after conversation and creative collaboration

**Comparison note:** OpenBulma-v4 had a more sophisticated affective computing layer (`SomaticStateService`) with emotional resonance directly integrated into memory retrieval. OpenCAS's affect is attached to episodes but does not yet feed retrieval scoring as richly.

---

### 4.9 `consolidation/`, `compaction/`, `context/` — Memory Lifecycle

**Consolidation (`consolidation/engine.py`):**
`NightlyConsolidationEngine` performs deep memory maintenance:
- Greedy clustering of non-compacted episodes by embedding similarity
- LLM summarization into `Memory` records
- Promotion of strong individual signals
- Edge rebuilding via `FabricBuilder`
- Salience reweighting and pruning
- Identity anchor updates

**Compaction (`compaction/compactor.py`):**
`ConversationCompactor` summarizes old context when token budgets are exceeded, injecting a synthetic continuation message so the agent retains continuity.

**Context (`context/builder.py`, `context/retriever.py`):**
`ContextBuilder` assembles LLM prompts from system persona, recent history, and retrieved memories. `MemoryRetriever` fuses semantic search with episode graph traversal.

**Question answered:** *Is plan mode persisted?*  
**Answer:** No. `ToolUseContext.plan_mode` is a boolean flag in memory. If the agent restarts during plan mode, the state is lost. There is no recovery from `SessionContextStore`.

---

### 4.10 `somatic/` — Agent Body State

**Key files:** `manager.py`, `modulators.py`, `appraisal.py`

**What it does:**
`SomaticManager` tracks dimensions like arousal, fatigue, tension, valence, focus, energy, and certainty. `SomaticModulators` translates this state into:
- LLM `temperature` adjustments
- Prompt `style_note` injections
- Memory retrieval emotional boosts

Appraisal uses a keyword map with negation windows to infer affect from events.

**Question answered:** *Any plans for a learned somatic appraiser?*  
**Answer:** None found in the codebase, spec, or notes. The current appraiser is heuristic-based. Improving it would likely require either a lightweight classifier or LLM-based appraisal.

---

### 4.11 `telemetry/` and `diagnostics/`

**Telemetry:**
- `Tracer` logs structured events to JSONL with span support
- `TokenTelemetry` records LLM prompt/completion tokens, latency, and cost per call
- `EventBus` provides async typed pub/sub for reactive coordination

**Diagnostics:**
- `Doctor` runs health checks across all subsystems
- `HealthMonitor` runs `Doctor` checks every 60s and emits `HealthCheckEvent`

---

## 5. OpenCAS vs. OpenBulma-v4: Detailed Comparison

### 5.1 Integration Architecture

| Dimension | OpenBulma-v4 | OpenCAS |
|-----------|--------------|---------|
| **Composition root** | Monolithic `src/index.ts` (~864 lines), imperative wiring | **Staged `BootstrapPipeline` with explicit `BootstrapContext`** |
| **Runtime coordinator** | `IntegrationHub` (~4,826 lines) with heavy callback injection | `AgentRuntime` (~1,767 lines) with context-based mediation |
| **Coupling** | High — hub knows every subsystem directly | Moderate — runtime mediates via context |
| **Async model** | Node.js `EventEmitter` (sync callbacks) | **True async `EventBus` with `asyncio.gather`** |
| **Concurrency** | Task-based with event emitters | **Lane-based workers (`CHAT`, `BAA`, `CONSOLIDATION`, `CRON`)** |

**Verdict:** OpenCAS has a dramatically cleaner bootstrap and runtime architecture. OpenBulma-v4's `IntegrationHub` was powerful but became a bottleneck where every change rippled across the system.

---

### 5.2 Memory and Embeddings

| Dimension | OpenBulma-v4 | OpenCAS |
|-----------|--------------|---------|
| **Primary store** | Postgres + Qdrant | SQLite + optional Qdrant |
| **Vector search** | Qdrant HNSW (fast, scalable) | **Brute-force SQLite scan (O(N))** or optional Qdrant |
| **Retrieval scoring** | **Multi-factor** (jaccard, vector, emotional, recency, salience, graph, confidence) | Semantic similarity + graph BFS + salience (simpler) |
| **Emotional memory** | ** Emotional vectors in retrieval** | Affect attached to episodes, limited retrieval impact |
| **Graph lifecycle** | Implicit via reindexing | **Explicit orphan recovery, edge rebuild, identity core promotion** |

**Verdict:** OpenBulma-v4 had a richer, more battle-tested memory system, especially for retrieval. OpenCAS's brute-force vector scan is a genuine scaling liability without Qdrant. However, OpenCAS's explicit graph lifecycle management (orphan recovery, edge rebuild) is more systematic.

---

### 5.3 Tools and Execution

| Dimension | OpenBulma-v4 | OpenCAS |
|-----------|--------------|---------|
| **Tool governance** | Basic | **Risk tiers + validation pipeline + pre-execution hooks** |
| **MCP support** | **Yes** | No |
| **BAA concurrency** | Task isolation | **Lane-based worker pools** |
| **BAA lifecycle** | **8-phase repair lifecycle** (detect/snapshot/reproduce/diagnose/patch/verify/deploy/postcheck) | Execution stages with futures/dependencies |
| **Checkpoints** | Git-based | Git-based (same pattern) |
| **Convergence guards** | Yes | Yes |
| **Task persistence** | **Persistent `TaskStore`** | Persistent `TaskStore` with auto-resume |

**Verdict:** OpenCAS has better tool governance and cleaner concurrency. OpenBulma-v4 had a richer BAA repair lifecycle and MCP integration. Both now have task persistence and convergence guards (OpenCAS added these in later phases).

---

### 5.4 Presence and Observability

| Dimension | OpenBulma-v4 | OpenCAS |
|-----------|--------------|---------|
| **UI channels** | **Web dashboard, WebSocket chat, Telegram bot, TUI** | CLI-only, optional FastAPI server |
| **Background loops** | **Actually scheduled and running** (`ExecutiveLoopRunner`, `ConsolidationCoordinator`) | Implemented in code, scheduled in `AgentScheduler`, but mostly test-driven |
| **Token telemetry** | Buffered JSONL with query APIs | Buffered JSONL with query APIs (borrowed directly) |
| **Health checks** | Basic | **Systematic `Doctor` + `HealthMonitor`** |

**Verdict:** OpenBulma-v4 is far ahead in user-facing presence. It was a running, online agent. OpenCAS is currently a backend/runtime that requires explicit invocation or the FastAPI server to be useful.

---

### 5.5 Autonomy and Psychology

| Dimension | OpenBulma-v4 | OpenCAS |
|-----------|--------------|---------|
| **Self-approval** | Static policy-based | **Dynamic trust/history/somatic/relational scoring** |
| **Creative ladder** | Present but less structured | **Explicit 7-stage ladder with portfolio clustering** |
| **Daydreaming** | `DaydreamCoordinator` | `DaydreamGenerator` + reflection evaluator + self-compassion mirror |
| **ToM** | Basic belief tracking | **Persistent belief/intention store with consistency checks** |
| **Relational field** | `SomaticStateService` with musubi dynamics | **Explicit `RelationalEngine` with 4 dimensions** |
| **Somatic state** | **More sophisticated, directly impacts retrieval** | Good state tracking, weaker retrieval integration |

**Verdict:** OpenCAS has a more elaborate theoretical model of autonomy and psychology. Whether this translates to better behavior is harder to assess because OpenCAS has not been run end-to-end as extensively as Bulma.

---

## 6. What's On Par, What's Better, What Needs Work

### What's Better in OpenCAS
1. **Bootstrap architecture** — staged, explicit, testable
2. **Async event bus** — true async pub/sub vs. sync EventEmitter
3. **Tool governance** — risk tiers, validation pipeline, hooks
4. **Lane-based execution** — cleaner concurrency control
5. **Graph lifecycle** — explicit orphan recovery and edge rebuild
6. **Self-approval sophistication** — multi-factor dynamic scoring
7. **Psychological architecture** — more explicit ToM, relational, and creative ladder models
8. **Test coverage** — 644 pytest tests covering stores, models, and integration

### What's On Par
1. **Token telemetry** — direct translation of Bulma's pattern
2. **Git checkpoints** — both use the same approach
3. **Task persistence** — both have persistent task stores
4. **Consolidation concept** — both have nightly deep-memory cycles

### What Needs Work (OpenCAS Gaps)
1. **Vector search scaling** — brute-force O(N) scan without Qdrant is not production-viable for large memory stores
2. **User-facing presence** — no dashboard, no TUI, no Telegram bot; just CLI and a bare FastAPI server
3. **MCP integration** — missing entirely; limits ecosystem tool access
4. **BAA pause handling** — `BaaPauseEvent` is emitted but never consumed by the BAA
5. **Plan mode persistence** — in-memory only, lost on restart
6. **LLM source attribution** — ad-hoc strings (`tool_use_loop`, `compaction`, etc.) with no central schema
7. **Somatic retrieval integration** — affect does not yet meaningfully modulate memory retrieval scoring
8. **Real-world shakedown** — 644 tests, but largely unit/integration; no end-to-end validation against live LLMs and embeddings at scale

---

## 7. Integration Assessment: Is OpenCAS More Integrated Than OpenBulma?

**Yes, architecturally. No, operationally.**

**Architecturally:** OpenCAS subsystems have clearer boundaries, explicit contracts, and a proper composition root. The directory layout (`bootstrap/`, `runtime/`, `memory/`, `autonomy/`, `execution/`, etc.) enforces separation of concerns. The event bus is truly async. The tool registry has explicit risk tiers and validation. Plugins have a manifest-driven lifecycle.

**Operationally:** OpenBulma-v4 was a *running system*. Its subsystems were welded together haphazardly in `IntegrationHub`, but they actually ran together 24/7, serving users over WebSocket and Telegram, scheduling background loops, and maintaining a continuous presence. OpenCAS's subsystems are better *designed* but less *proven in composition*. Many of its advanced features (daydream scheduling, consolidation, harness cycles) exist as correct-looking code that has primarily been exercised through unit tests rather than long-running autonomous operation.

In short: **OpenBulma-v4 was a working prototype with architectural debt. OpenCAS is a cleaner foundation that still needs to be driven around the block.**

---

## 8. Unanswered Questions (For Future Investigation)

1. **Qdrant operational strategy:** Should Qdrant be treated as mandatory for production, or should a local ANN library (faiss, hnswlib) be added as a middle-tier fallback?

2. **BAA pause recovery:** Who should subscribe to `BaaPauseEvent` to actually halt task execution, and what should the auto-recovery policy be?

3. **MCP roadmap:** Is MCP integration planned, or is OpenCAS intentionally avoiding the protocol in favor of its own plugin manifest system?

4. **AgentToolAdapter supervision:** Subagents spawned via `agent` tool run in a completely isolated `ToolUseLoop`. Should there be a parent-child supervision hierarchy (timeouts, cancellation propagation, result validation)?

5. **Plan mode durability:** Should `enter_plan_mode` persist state to `SessionContextStore` so it survives restarts?

6. **Somatic appraisal evolution:** Is the keyword-based appraiser considered sufficient long-term, or is there a plan to migrate to LLM-based or learned affect inference?

7. **Real e2e validation:** When will OpenCAS run a sustained autonomous session against live LLM and embedding providers to validate integration at scale?

---

## 9. Conclusion

OpenCAS represents a significant architectural evolution over OpenBulma-v4. It takes the *ideas* that worked in Bulma (BAA, token telemetry, checkpoints, consolidation) and reimplements them with cleaner boundaries, better async semantics, and more sophisticated autonomy models. It also adds genuinely new subsystems (Theory of Mind, relational resonance, plugin lifecycle, intervention policy) that Bulma lacked.

However, OpenBulma-v4 still wins on **proven runtime behavior** and **user-facing richness**. Its memory retrieval was more sophisticated, its agent was actually online, and its subsystems had been stress-tested in continuous operation. OpenCAS's challenge is to close the gap between elegant design and field-hardened execution—especially in vector search scaling, background loop reliability, and real-world LLM prompt stability.

If the goal is a *better foundation for the next generation*, OpenCAS is on the right track. If the goal is an agent that can run unsupervised tonight, Bulma remains the safer reference point.
