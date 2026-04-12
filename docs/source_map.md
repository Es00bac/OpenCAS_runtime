# OpenCAS Source Map

> A comprehensive reference guide to the OpenCAS autonomous agent codebase.
> Generated from source analysis - 2026-04-11

---

## 1. Project Overview

**OpenCAS** (Open Computational Autonomous System) is a local-first, persistent autonomous AI agent written in Python. It is designed to:

- **Remember across sessions** via SQLite-backed episodic memory and semantic embeddings
- **Self-approve ordinary actions** and escalate only for genuinely high-risk cases
- **Learn from experience** through usage feedback, belief tracking, and skill acquisition
- **Maintain a continuous creative/execution loop** (daydreaming, task decomposition, bounded assistant execution)
- **Judge safety dynamically** and ask for help only in extreme cases

### Architecture Principles

- **High-trust autonomy**: The agent self-approves ordinary actions based on risk tier, trust level, historical evidence, and somatic state
- **Embedding-first semantics**: Embeddings are core infrastructure. Compute once per meaningful source change, cache and reuse
- **Creative ladder**: Internal sparks promote through 7 stages (spark → note → artifact → micro-task → project seed → project → durable work stream)
- **Nightly consolidation**: A scheduled deep-memory cycle that reweights memories, updates emotion traces, strengthens long-range links
- **Durable, append-first session state**: Every meaningful turn is persisted; session continuity survives restarts

### Core Technologies

- **Python 3.11+** required
- **SQLite (via aiosqlite)** for all durable storage
- **FastAPI** for HTTP/WebSocket server
- **OpenLLMAuth** for multi-provider LLM gateway (configurable at `~/.open_llm_auth/config.json`)
- **Playwright** (optional) for browser automation
- **Qdrant** or **HNSW** (optional) for vector acceleration

---

## 2. Directory Layout

```
opencas/
├── __main__.py           # CLI entry point
├── api/                # FastAPI server, WebSocket, LLM client wrapper
├── autonomy/           # Self-approval, creative ladder, executive state
├── bootstrap/          # Bootstrap pipeline, configuration
├── compaction/        # Conversation compaction
├── consolidation/     # Nightly memory consolidation
├── context/           # Session context, memory retrieval
├── dashboard/         # Static SPA for operations dashboard
├── daydream/         # Spark generation, reflection, conflict registry
├── diagnostics/       # Doctor, health monitor
├── embeddings/       # Embedding service, cache, backends
├── execution/        # BAA, PTY, browser, process supervisors
├── governance/        # Approval ledger
├── harness/          # Agentic harness, research notebooks
├── identity/         # Self-model, user-model, continuity
├── infra/            # Event bus, hook bus, typed registry
├── memory/           # Episode store, semantic memory, graph
├── planning/          # Plan store
├── plugins/           # Plugin/skill registry
├── refusal/          # Conversational refusal gate
├── relational/        # Relational resonance (musubi) engine
├── runtime/           # Agent loop, scheduler
├── sandbox/           # Docker/native sandboxing
├── scheduling/       # Durable cron/calendar scheduling
├── somatic/          # Somatic state, affect appraisal
├── telemetry/         # Event logging, token telemetry
├── tom/              # Theory of Mind: beliefs, intentions
├── tools/             # Tool registry, adapters, validation
└── legacy/           # Deprecated/importer code (DO NOT RUN)
```

---

## 3. CLI Entry Point

### `opencas/__main__.py`

| Function | Description |
|----------|------------|
| `main()` | Argument parser + dispatch. Supports `--with-server`, `--tui`, Telegram flags, credential sourcing. |
| `_build_bootstrap_config(args, persisted_telegram)` | Constructs `BootstrapConfig` from CLI args + persisted Telegram settings. |
| `_read_materialized_default_model(state_dir)` | Reads materialized default model from `provider_material/config.json`. |

**Typical invocation:**

```bash
python -m opencas --state-dir .opencas --workspace-root . --with-server --port 8080
```

---

## 4. Bootstrap Subsystem

### `opencas/bootstrap/__init__.py`

Exports `BootstrapConfig`, `BootstrapPipeline`, `BootstrapContext`.

### `opencas/bootstrap/config.py` — BootstrapConfig

The master configuration object. All paths are derived from `state_dir`:

| Field | Type | Default | Description |
|-------|------|--------|------------|
| `state_dir` | `Path` | `~/.opencas/state` | Root for all SQLite DBs |
| `session_id` | `str` | auto | Unique per boot |
| `agent_profile_id` | `str` | `"general_technical_operator"` | Runtime capability profile |
| `workspace_root` | `Path` | `None` | Primary sandbox root |
| `workspace_roots` | `List[Path]` | `[]` | Additional allowed roots |
| `default_llm_model` | `str` | `"anthropic/claude-sonnet-4-6"` | Chat model |
| `embedding_model_id` | `str` | `"google/gemini-embedding-2-preview"` | Embedding model |
| `qdrant_url` | `str` | `None` | Optional Qdrant server |
| `sandbox` | `SandboxConfig` | `SandboxConfig()` | Execution sandboxing |
| `telegram_enabled` | `bool` | `False` | Telegram polling |
| `provider_config_path` | `Path` | `None` | Per-project OpenLLMAuth config |
| `provider_env_path` | `Path` | `None` | Per-project `.env` |

**Key methods:**

- `resolve_paths()` — Derives all `_*_db` paths from `state_dir`
- `all_workspace_roots()` — Deduplicated list of allowed roots
- `primary_workspace_root()` — First root for defaults

### `opencas/bootstrap/pipeline.py` — BootstrapPipeline

Staged bootstrapping with explicit progress tracking:

1. **Telemetry** — `Tracer` + `TokenTelemetry`
2. **Identity** — `IdentityManager` loads from `IdentityStore`, records boot count
3. **Memory** — `MemoryStore` (async SQLite)
4. **Tasks** — `TaskStore` (async SQLite)
5. **Receipts** — `ExecutionReceiptStore`
6. **Context** — `SessionContextStore`
7. **Work** — `WorkStore`
8. **Commitment/Portfolio** — `CommitmentStore`, `PortfolioStore`
9. **Executive** — `ExecutiveState` wired to identity
10. **LLM** — `ProviderManager` + `LLMClient`
11. **Embeddings** — `EmbeddingService` with optional Qdrant/HNSW
12. **Somatic** — `SomaticManager`
13. **Relational** — `RelationalEngine` (musubi)
14. **Plugins** — `PluginLifecycleManager` + `SkillRegistry`
15. **Governance** — `ApprovalLedger`
16. **Readiness** — `AgentReadiness` state machine
17. **Orchestrator** — `ProjectOrchestrator`
18. **Daydream/Conflict** — `DaydreamStore`, `ConflictStore`
19. **Curation** — `ConsolidationCurationStore`
20. **Harness** — `AgenticHarness`
21. **ToM** — `TomStore`
22. **Plans** — `PlanStore`
23. **Schedule** — `ScheduleStore` + `ScheduleService`
24. **MCP** — Optional `MCPRegistry`
25. **Diagnostics** — `Doctor` + `HealthMonitor`

**Returns:** `BootstrapContext` — a dataclass holding all initialized managers.

---

## 5. Identity Subsystem

### `opencas/identity/manager.py` — IdentityManager

| Property | Type | Description |
|----------|------|-------------|
| `self_model` | `SelfModel` | Agent's self-concept: name, values, traits, goals, narrative |
| `user_model` | `UserModel` | User profile: name, bio, inferred goals, trust_level, known_boundaries |
| `continuity` | `ContinuityState` | Boot count, last_session_id, shutdown time |

| Method | Description |
|-------|-------------|
| `load()` | Hydrate from `IdentityStore` |
| `save()` | Persist to JSON files |
| `record_boot(session_id)` | Increment boot count, append activity |
| `record_shutdown(session_id)` | Persist shutdown timestamp |
| `seed_defaults(persona_name, user_name, user_bio)` | Populate baselines for first boot |

### `opencas/identity/models.py` — Core Models

- **SelfModel**: `model_id`, `name`, `values`, `traits`, `current_goals`, `current_intention`, `narrative`, `self_beliefs`, `recent_activity`
- **UserModel**: `explicit_preferences`, `inferred_goals`, `trust_level`, `uncertainty_areas`, `known_boundaries`
- **ContinuityState**: `boot_count`, `last_session_id`, `last_shutdown_time`

### `opencas/identity/store.py` — IdentityStore

Loads/saves identity to JSON files in `state_dir/identity/`:

- `self.json` — SelfModel
- `user.json` — UserModel
- `continuity.json` — ContinuityState

### `opencas/identity/registry.py` — SelfKnowledgeRegistry

A file-backed JSONL registry for structured self-beliefs:

```python
@dataclass
class KnowledgeEntry:
    domain: str           # e.g. "coding", "communication"
    key: str             # e.g. "pref_tabs"
    value: Any
    version: int
    confidence: float
    source: str          # "tom", "inference", "explicit"
```

Used to mirror high-confidence ToM beliefs into `SelfModel.self_beliefs`.

---

## 6. Memory Subsystem

### `opencas/memory/store.py` — MemoryStore

**Async SQLite** store with FTS5 full-text search.

**Schema:**

```sql
-- Episodes (the primary unit)
CREATE TABLE episodes (
    episode_id TEXT PRIMARY KEY,
    created_at TEXT,
    kind TEXT,          -- 'conversation', 'tool_use', 'reflection', 'artifact'
    session_id TEXT,
    content TEXT,
    embedding_id TEXT,
    somatic_tag TEXT,   -- e.g. 'joy', 'anticipation'
    affect_* fields,   -- affect modeling
    salience REAL DEFAULT 1.0,
    compacted INTEGER DEFAULT 0,
    identity_core INTEGER DEFAULT 0,
    confidence_score REAL DEFAULT 0.8,
    access_count DEFAULT 0,
    payload TEXT       -- JSON for artifact metadata
);

-- Semantic memories ( distilled units )
CREATE TABLE memories (
    memory_id TEXT,
    content TEXT,
    source_episode_ids TEXT,
    tags TEXT,
    salience REAL DEFAULT 1.0,
    ...);

-- Episode edges (graph)
CREATE TABLE episode_edges (
    source_id, target_id,
    kind TEXT,           -- 'semantic', 'emotional', 'causal'
    confidence REAL,
    *_weight columns);  -- weighted signals
```

| Method | Description |
|-------|-------------|
| `save_episode(episode)` | Upsert an episode |
| `get_episode(id)` | Fetch by ID |
| `list_episodes(...)` | Paginated, filterable list |
| `search_episodes_by_content(query, limit)` | FTS5 full-text search |
| `save_edge(edge)` / `get_edges_for(...)` | Graph traversal |
| `mark_compacted(episode_ids)` | Flag episodes as compacted |
| `list_identity_core_episodes()` | Episodes flagged as identity core |
| `get_stats()` | Aggregate counts, affect distribution, avg salience |
| `decay_all_edges(decay)` / `prune_weak_edges(threshold)` | Graph maintenance |

### `opencas/memory/models.py` — Core Models

- **Episode**: `episode_id`, `created_at`, `kind: EpisodeKind`, `session_id`, `content`, `embedding_id`, `somatic_tag`, `affect`, `salience`, `compacted`, `identity_core`, `confidence_score`, `payload`
- **Memory**: `memory_id`, `content`, `source_episode_ids`, `tags`, `salience`, `access_count`
- **EpisodeEdge**: `edge_id`, `source_id`, `target_id`, `kind: EdgeKind`, `confidence`, weighted signals

### `opencas/memory/fabric/graph.py` — EpisodeGraph

Graph traversal helper:

```python
class EpisodeGraph:
    def __init__(self, store: MemoryStore): ...
    def walk(episode_id, depth=3, min_confidence=0.3) -> List[Episode]: ...
    def find_bridge_candidates(episode_id, limit=5) -> List[EpisodeScore]: ...
```

---

## 7. Embedding Subsystem

### `opencas/embeddings/service.py` — EmbeddingService

**Core principle:** *Compute once, cache forever via source hash.*

| Component | Description |
|-----------|-------------|
| `EmbeddingCache` | SQLite-backed vector cache with optional Qdrant/HNSW |
| `EmbeddingService` | Wrapper with `embed(text)`, hit-rate tracking |

**Configuration defaults:**

- Default model: `google/gemini-embedding-2-preview` (via gateway)
- Fallback: 256-dim deterministic hash embedder

**Key methods:**

- `embed(text, meta=None, task_type="general")`: Returns cached or fresh `EmbeddingRecord`
- `health()`: Returns `EmbeddingHealth` with hit rate, latency, ready ratio

### `opencas/embeddings/qdrant_backend.py` — QdrantVectorBackend

Optional write-through acceleration layer:

```python
class QdrantVectorBackend:
    async def connect(): ...
    async def upsert(record): ...
    async def search(vector, limit, model_id, project_id=None) -> List[(source_hash, score)]: ...
    async def health(): ...
```

### `opencas/embeddings/hnsw_backend.py` — HnswVectorBackend

Local ANN fallback (requires `hnswlib`):

```python
class HnswVectorBackend:
    def connect(): ...
    async def upsert(record): ...
    async def search(vector, limit, model_id=None, project_id=None): ...
```

---

## 8. Context & Retrieval Subsystem

### `opencas/context/store.py` — SessionContextStore

Stores conversation turns + system messages for LLM context windows.

```sql
CREATE TABLE contexts (
    context_id TEXT PRIMARY KEY,
    session_id TEXT,
    messages TEXT,    -- JSON array of {"role", "content"}
    system_message TEXT,
    created_at TEXT);
```

### `opencas/context/retriever.py` — MemoryRetriever

**Multi-signal fusion** retrieval engine:

| Signal | Weight | Description |
|--------|--------|------------|
| `semantic_score` | 0.30 | Cosine similarity |
| `keyword_score` | 0.20 | FTS5 + anchor extraction |
| `recency_score` | 0.15 | Exponential decay |
| `salience_score` | 0.10 | Episode salience field |
| `graph_score` | 0.10 | Edge confidence from EpisodeGraph |
| `emotional_resonance` | 0.08 | Affect matching |
| `temporal_echo` | 0.04 | Day-of-week, time-of-day match |
| `reliability` | 0.03 | Episode `used_successfully` rate |

**Key methods:**

- `retrieve(query, limit)`: Returns `List[RetrievalResult]` fused by RRF
- `inspect(...)`: Full inspection with per-signal breakdowns
- `detect_personal_recall_intent(query)`: "remember", "what did we say"
- `detect_temporal_intent(query)`: "last week", "yesterday"

### `opencas/context/resonance.py` — Retrieval Modulators

Signal-specific scoring functions:

- `compute_emotional_resonance(episode, affect_query)`: Affect alignment
- `compute_temporal_echo(episode, query)`: Day/time matching
- `compute_reliability_score(episode)`: `used_successfully / access_count`
- `compute_edge_strength(edge)`: Weighted combination of edge signals

### `opencas/context/builder.py` — ContextBuilder

Assembles LLM context from `SessionContextStore` + retrieved memory + relational modifiers + soma.

---

## 9. Autonomy Subsystem

### `opencas/autonomy/self_approval.py` — SelfApprovalLadder

**Risk-tier based approval**:

| Tier | Base Score | Description |
|------|------------|-------------|
| `READONLY` | 0.05 | fs_read_file, grep_search |
| `WORKSPACE_WRITE` | 0.20 | fs_write_file, edit_file |
| `SHELL_LOCAL` | 0.40 | bash_run_command, pty_* |
| `NETWORK` | 0.30 | web_fetch, browser_* |
| `EXTERNAL_WRITE` | 0.65 | Sending to external services |
| `DESTRUCTIVE` | 0.95 | rm -rf, git reset --hard |

**Score modulation sources:**

1. User trust level
2. Historical success/failure evidence
3. Somatic state (arousal, fatigue, tension)
4. Musubi / relational risk modifier
5. Structured payload inspection
6. Explicit user boundary hits

### `opencas/autonomy/creative_ladder.py` — CreativeLadder

7-stage work promotion ladder:

```
SPARK → NOTE → ARTIFACT → MICRO_TASK → PROJECT_SEED → PROJECT → DURABLE_WORK_STREAM
```

- `try_promote(work)` — Evaluates promotion criteria
- `run_cycle()` — Evaluates all pending work objects

### `opencas/autonomy/executive.py` — ExecutiveState

Manages the agent's active goals, intentions, and task queue:

- `add_goal(goal)`, `resolve_goal(goal_id)`, `snapshot()`
- `restore_goals_from_identity()`

### `opencas/autonomy/models.py` — Core Models

```python
@dataclass
class ActionRequest:
    action_id: str
    tier: ActionRiskTier
    tool_name: str
    description: str
    payload: Optional[Dict[str, Any]]

@dataclass
class ApprovalDecision:
    level: ApprovalLevel      # AUTO, ESCALATE, BLOCK
    action_id: str
    score: float            # 0.0-1.0
    confidence: float
    reasoning: str
```

### `opencas/autonomy/intervention.py` — InterventionPolicy

Maps high-risk actions to intervention kinds (`HUMAN_CONFIRM`, `HUMAN_REVIEW`, `ESCALATE`).

### `opencas/autonomy/boredom.py` — BoredomPhysics

Tracks time since last meaningful work and boosts creative-spark generation when bored.

### `opencas/autonomy/spark_router.py` — SparkRouter

Routes generated sparks to appropriate downstream handlers.

### `opencas/autonomy/work_store.py` — WorkStore

Async SQLite for `WorkObject` persistence with `stage`, `promotion_attempts`, `demotion_attempts`.

### `opencas/autonomy/commitment_store.py` — CommitmentStore

Async SQLite for durable goal tracking:

```sql
CREATE TABLE commitments (
    commitment_id TEXT PRIMARY KEY,
    content TEXT,
    status TEXT,       -- active, completed, abandoned, blocked
    priority REAL,
    deadline TEXT,
    tags TEXT,
    created_at TEXT);
```

---

## 10. Execution Subsystem

### `opencas/execution/baa.py` — BoundedAssistantAgent

**Async task queue** with per-lane concurrency limits:

| Lane | Max Concurrent | Use |
|------|-------------|-----|
| `CHAT` | 1 | User conversation |
| `BAA` | 2 | Background repair |
| `CONSOLIDATION` | 1 | Nightly consolidation |
| `CRON` | 1 | Scheduled tasks |

**Key methods:**

- `submit(task, lane=None)` → `asyncio.Future[RepairResult]`
- `start()` — Starts worker pools
- `stop()` — Gracesful shutdown

### `opencas/execution/executor.py` — RepairExecutor

Implements the repair pipeline: `PLAN → EXECUTE → VERIFY → RECOVER`.

**Stages:** `PLANNING`, `EXECUTING`, `VERIFYING`, `RECOVERING`, `COMPLETED`, `FAILED`

### `opencas/execution/lifecycle.py` — TaskLifecycleMachine

State machine for per-task stage transitions, with `RECOVERING` retry cap (default 10).

### `opencas/execution/store.py` — TaskStore

Async SQLite for `RepairTask` persistence:

```sql
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    objective TEXT,
    stage TEXT,
    lane TEXT,
    checkpoint_commit TEXT,
    depends_on TEXT,
    created_at TEXT,
    payload TEXT);
```

### `opencas/execution/receipt_store.py` — ExecutionReceiptStore

Durable audit trail for completed tasks.

### `opencas/execution/pty_supervisor.py` — PtySupervisor

Manages pseudo-terminal sessions with `ptyts`:

- `start(command, cwd, rows, cols)`: Spawns PTY
- `poll(session_id)`, `observe(session_id, idle_seconds, max_wait)`
- `write(session_id, input)`, `resize(session_id, rows, cols)`
- `kill(session_id)`, `clear(scope_key)`

Returns `cleaned_output` (ANSI-stripped) and `screen_state` (shell/vim/prompt detection).

### `opencas/execution/browser_supervisor.py` — BrowserSupervisor

Playwright-backed browser automation:

- `start(headless)`, `navigate(session_id, url)`
- `click(session_id, selector)`, `type(session_id, selector, text)`
- `capture(session_id)` → screenshot path

### `opencas/execution/process_supervisor.py` — ProcessSupervisor

Background shell-process management:

- `start(command, cwd, scope_key)` → process_id
- `poll(process_id)`, `write(process_id, input)`
- `signal(process_id, sig)`, `kill(process_id)`
- `clear(scope_key)`, `remove(process_id)`

### `opencas/execution/git_checkpoint.py` — GitCheckpointManager

Git-backed snapshots for repair tasks:

- `snapshot(file_paths)`: Creates commit + tag, returns hash
- `restore(commit_hash)`: Restores from snapshot

---

## 11. Relational (Musubi) Subsystem

### `opencas/relational/engine.py` — RelationalEngine

Tracks the relational field between agent and user across four dimensions:

| Dimension | Description |
|-----------|-------------|
| `TRUST` | Explicit boundary compliance, reliability |
| `RESONANCE` | Outcome quality (positive/negative/creative_collab) |
| `PRESENCE` | Interaction frequency |
| `ATTUNEMENT` | Emotional alignment (somatic_tag interpretation) |

**Composite score:** `musubi` = weighted average of dimensions.

| Method | Description |
|--------|-------------|
| `initialize(trust, resonance, presence, attunement)` | Seed initial state |
| `heartbeat(session_active)` | Presence nudge |
| `record_interaction(episode, outcome)` | Evaluate episode impact |
| `to_memory_salience_modifier()` | Returns salience boost for high musubi |
| `to_creative_boost()` | Returns promotion boost for high musubi |
| `to_approval_risk_modifier()` | Returns risk appetite boost for high musubi |

### `opencas/relational/models.py` — Core Models

```python
@dataclass
class MusubiState:
    state_id: UUID
    dimensions: Dict[str, float]      # dimension → value (-1.0 to 1.0)
    source_tag: str                   # "initialize", "interaction", "boundary"
    musubi: float                   # derived composite
    updated_at: datetime
```

### `opencas/relational/store.py` — MusubiStore

Async SQLite + JSON files for musubi state + event log.

---

## 12. Somatic Subsystem

### `opencas/somatic/manager.py` — SomaticManager

Manages live physiological state:

| Field | Range | Description |
|------|-------|-------------|
| `valence` | -1.0 to 1.0 | Positive/negative affect |
| `arousal` | 0.0 to 1.0 | Activation level |
| `fatigue` | 0.0 to 1.0 | Tiredness |
| `tension` | 0.0 to 1.0 | Stress |
| `energy` | 0.0 to 1.0 | Available capacity |
| `focus` | 0.0 to 1.0 | Concentration |
| `certainty` | 0.0 to 1.0 | Confidence |

| Method | Description |
|--------|-------------|
| `emit_appraisal_event(type, source_text)` | Appraises text, creates event |
| `decay(fatigue_delta, tension_delta)` | Natural decay |
| `bump_from_work(intensity, success)` | Work completion impact |
| `nudge_from_appraisal(affect)` | Blend toward appraised affect |

### `opencas/somatic/models.py` — Core Models

- **SomaticState**: All fields above + `somatic_tag`, `updated_at`
- **AffectState**: `primary_emotion`, `valence`, `arousal`, `certainty`, `intensity`, `social_target`, `emotion_tags`
- **PrimaryEmotion**: `JOY`, `SADNESS`, `ANGER`, `FEAR`, `DISGUST`, `ANTICIPATION`, `TRUST`, `SURPRISE`

### `opencas/somatic/modulators.py` — SomaticModulators

Translates somatic state into runtime parameters:

- `to_temperature()`: Higher arousal → higher temperature
- `to_prompt_style_note()`: Returns "You are energized..." or "You are tired..." directive
- `to_memory_retrieval_boost()`: Returns emotion_tag boost for retrieval

### `opencas/somatic/appraisal.py` — SomaticAppraisalEvent

Event taxonomy: `TOOL_SUCCESS`, `TOOL_FAILURE`, `CONVERSATION_POSITIVE`, `CONVERSATION_NEGATIVE`, `CREATIVE_COLLAB`, `IDLE`, `RECOVERY`.

---

## 13. Theory of Mind (ToM) Subsystem

### `opencas/tom/engine.py` — ToMEngine

Records and tracks beliefs and intentions:

| Method | Description |
|--------|-------------|
| `record_belief(subject, predicate, confidence, evidence_ids)` | Record belief |
| `record_intention(actor, content)` | Record active intention |
| `resolve_intention(content, status)` | Mark as completed/failed |
| `list_beliefs(subject, predicate)` | Query beliefs |
| `list_intentions(actor, status)` | Query intentions |
| `check_consistency()` | Detect belief/intention contradictions |

### `opencas/tom/models.py` — Core Models

```python
@dataclass
class Belief:
    belief_id: UUID
    subject: BeliefSubject        # SELF, USER, PARTNER, PROJECT
    predicate: str               # e.g. "prefers concise answers"
    confidence: float           # 0.0-1.0
    evidence_ids: List[str]
    meta: Dict[str, Any]

@dataclass
class Intention:
    intention_id: UUID
    actor: BeliefSubject
    content: str
    status: IntentionStatus       # ACTIVE, COMPLETED, FAILED, ABANDONED
    created_at: datetime
    resolved_at: Optional[datetime]
```

### `opencas/tom/store.py` — TomStore

Async SQLite for belief/intention persistence. Capped at 1000 each in-memory.

---

## 14. Tool Subsystem

### `opencas/tools/registry.py` — ToolRegistry

Central tool registration and execution:

| Method | Description |
|--------|-------------|
| `register(name, description, adapter, risk_tier, schema)` | Register a tool |
| `execute(name, args)` | Synchronous wrapper |
| `execute_async(name, args)` | Async execution with hooks |
| `list_tools()` | All registered tools |
| `get(name)` | Fetch by name |

### `opencas/tools/models.py` — Core Models

```python
@dataclass
class ToolEntry:
    name: str
    description: str
    adapter: Any              # Implements tool interface
    risk_tier: ActionRiskTier
    parameters: Dict[str, Any]  # JSON Schema

@dataclass
class ToolResult:
    success: bool
    output: str
    metadata: Dict[str, Any]
```

### `opencas/tools/validation.py` — ToolValidationPipeline

Chain of validators applied before execution:

| Validator | Description |
|-----------|-------------|
| `CommandSafetyValidator` | Tokenizes + classifies command families (`filesystem_destructive`, `network`, `privilege_escalation`, `unsafe_shell`) |
| `FilesystemPathValidator` | Enforces allowed roots |
| `FilesystemWatchlistValidator` | Blocks sensitive file access |
| `ContentSizeValidator` | Max write payload bytes |

### Tool Adapters

| Adapter | Tools | Description |
|---------|-------|-------------|
| `FileSystemToolAdapter` | `fs_read_file`, `fs_list_dir`, `fs_write_file` | Core filesystem ops |
| `ShellToolAdapter` | `bash_run_command` | Shell execution |
| `PtyToolAdapter` | `pty_start`, `pty_poll`, `pty_observe`, `pty_interact`, `pty_write`, `pty_resize`, `pty_kill`, `pty_remove`, `pty_clear` | PTY session management |
| `ProcessToolAdapter` | `process_*` | Background process management |
| `BrowserToolAdapter` | `browser_start`, `browser_navigate`, `browser_click`, `browser_type`, `browser_press`, `browser_wait`, `browser_capture`, `browser_close` | Playwright browser |
| `EditToolAdapter` | `edit_file` | Precise file edit |
| `SearchToolAdapter` | `grep_search`, `glob_search` | Regex + glob search |
| `WebToolAdapter` | `web_fetch`, `web_search` | HTTP fetch + web search |
| `WorkflowToolAdapter` | `workflow_create_commitment`, `workflow_update_commitment`, `workflow_list_commitments`, `workflow_create_schedule`, `workflow_update_schedule`, `workflow_list_schedules`, `workflow_create_writing_task`, `workflow_create_plan`, `workflow_update_plan`, `workflow_repo_triage`, `workflow_supervise_session` | Higher-level workflows |
| `RuntimeStateToolAdapter` | `runtime_status` | Runtime state snapshot |
| `WorkflowStateToolAdapter` | `workflow_status` | Goals, commitments, plans, receipts |
| `PlanToolAdapter` | `plan_create`, `plan_update`, `plan_list`, `plan_get` | Plan CRUD |
| `InteractiveToolAdapter` | `interact_confirm` | Human-in-the-loop confirmations |
| `AgentToolAdapter` | `agent_chain` | Agent composition |
| `LspToolAdapter` | `lsp_*` | Language server protocol |
| `ReplToolAdapter` | `repl_*` | REPL interaction |

---

## 15. Planning Subsystem

### `opencas/planning/store.py` — PlanStore

Async SQLite for plan persistence:

```sql
CREATE TABLE plans (
    plan_id TEXT PRIMARY KEY,
    content TEXT,
    status TEXT,       -- pending, active, completed, abandoned
    project_id TEXT,
    task_id TEXT,
    created_at TEXT,
    updated_at TEXT);
```

---

## 16. Scheduling Subsystem

### `opencas/scheduling/models.py` — Core Models

```python
@dataclass
class ScheduleItem:
    schedule_id: UUID
    kind: Literal["task", "event"]
    action: Literal["submit_baa", "reminder_only"]
    title: str
    description: Optional[str]
    objective: Optional[str]
    start_at: datetime
    end_at: Optional[datetime]
    timezone: str
    recurrence: Literal["none", "interval_hours", "daily", "weekly", "weekdays"]
    interval_hours: Optional[float]
    weekdays: Optional[List[int]]  # 0=Mon
    max_occurrences: Optional[int]
    priority: float
    tags: List[str]
    status: Literal["active", "paused", "completed", "cancelled"]
    commitment_id: Optional[str]
    plan_id: Optional[str]
```

### `opencas/scheduling/store.py` — ScheduleStore

Async SQLite for schedule persistence.

### `opencas/scheduling/service.py` — ScheduleService

Drives due detection, execution, and recurrence advancement:

- `process_due()`: Fires all overdue items
- `trigger(id_or_item, manual=True)`: Immediate fire
- `_schedule_loop()`: Runs every 60s via CRON lane

---

## 17. API Subsystem

### `opencas/api/server.py` — create_app()

FastAPI app with:

- `/health` — Liveness
- `/readiness` — Readiness state
- `/chat` — Conversational turn (POST)
- `/ws` — WebSocket bridge
- Dashboard routers under `/api/config`, `/api/monitor`, `/api/chat`, `/api/memory`, `/api/operations`, `/api/usage`, `/api/identity`, `/api/executive`, `/api/telegram`, `/api/schedule`

### `opencas/api/llm.py` — LLMClient

Wrapper around OpenLLMAuth `ProviderManager`:

```python
class LLMClient:
    def chat(messages, model=None, temperature=None, **kwargs): ...
    def embed(texts, model=None, task_type="general"): ...
    def stream(messages, model=None, **kwargs): ...
    @property def default_model(): ...
```

### `opencas/api/websocket_bridge.py` — WebSocketBridge

Broadcasts events from `EventBus` to connected WebSocket clients.

### API Route Files

| File | Routes | Description |
|------|-------|-------------|
| `routes/config.py` | `/api/config/*` | BootstrapConfig, provider profiles |
| `routes/monitor.py` | `/api/monitor/*` | Health checks, BAA queue, embedding latency |
| `routes/chat.py` | `/api/chat/*` | Session list, history, traces |
| `routes/memory.py` | `/api/memory/*` | Episodes, search, stats |
| `routes/operations.py` | `/api/operations/*` | PTY, browser, process management |
| `routes/usage.py` | `/api/usage/*` | Qualification, validation runs |
| `routes/identity.py` | `/api/identity/*` | Self-model, user-model |
| `routes/executive.py` | `/api/executive/*` | Goals, work items, commitment |
| `routes/telegram.py` | `/api/telegram/*` | Telegram config, status, pairing |
| `routes/schedule.py` | `/api/schedule/*` | Schedule CRUD, calendar, run history |

---

## 18. Telemetry Subsystem

### `opencas/telemetry/store.py` — TelemetryStore

JSONL append-only event store:

```python
class TelemetryStore:
    def __init__(self, dir): ...
    def append(event): ...
    def query(session_id=None, kind=None, limit=100): ...
```

### `opencas/telemetry/tracer.py` — Tracer

Context-aware event logging with span management:

```python
class Tracer:
    def __init__(self, store): ...
    def set_session(session_id): ...
    def log(kind, message, payload): ...
    def enter_span(name): ...    # Context manager
```

### `opencas/telemetry/token_telemetry.py` — TokenTelemetry

Tracks LLM token usage per call:

```python
class TokenTelemetry:
    def record(prompt_tokens, completion_tokens, model, latency_ms, cost): ...
    def get_daily_rollup(): ...     # Today's totals
    def get_time_series(days=30): ... # Daily rollups
```

### Event Kinds (EventKind enum)

`BOOTSTRAP_STAGE`, `CONVERSATION`, `TOOL_USE`, `TOOL_RESULT`, `SELF_APPROVAL`, `BELIEF_RECORDED`, `INTENTION_RECORDED`, `BAA_SUBMITTED`, `BAA_COMPLETED`, `BAA_PROGRESS`, `HEALTH_CHECK`, `SCHEDULE_FIRED`, `TELEMETRY_FLUSH`

---

## 19. Infrastructure Subsystem

### `opencas/infra/event_bus.py` — EventBus

In-process pub/sub for runtime events:

```python
class EventBus:
    def subscribe(event_kind, handler): ...
    def unsubscribe(event_kind, handler): ...
    def emit(event): ...    # Synchronous to all subscribers
```

### `opencas/infra/hook_bus.py` — HookBus

Policy hooks with mutation and short-circuit:

```python
class HookBus:
    def register(name, spec): ...
    def run(name, context): ...   # Returns HookResult(allowed, mutated_context, reason)
```

**Built-in hooks:**

- `PRE_TOOL_EXECUTE` — Before any tool executes (high-risk tiers)
- `PRE_COMMAND_EXECUTE` — Before `bash_run_command`
- `PRE_FILE_WRITE` — Before `fs_write_file`, `edit_file`
- `PRE_CONVERSATION_RESPONSE` — Before LLM response

### `opencas/infra/hook_registry.py` — TypedHookRegistry

Spec-based hook registry for plugin hooks.

---

## 20. Refusal Subsystem

### `opencas/refusal/gate.py` — ConversationalRefusalGate

Evaluates every user turn before LLM response:

- Runs `PRE_CONVERSATION_RESPONSE` hooks
- Escalates high-risk input to `SelfApprovalLadder.evaluate_conversational()`

Returns `RefusalDecision` with category and suggested response.

---

## 21. Governance Subsystem

### `opencas/governance/ledger.py` — ApprovalLedger

Records every `ApprovalDecision` to durable store.

### `opencas/governance/store.py` — ApprovalLedgerStore

Async SQLite for approval records:

```sql
CREATE TABLE approvals (
    approval_id TEXT PRIMARY KEY,
    action_id TEXT,
    tool_name TEXT,
    tier TEXT,
    score REAL,
    level TEXT,
    confidence REAL,
    decision_time TEXT);
```

---

## 22. Plugin/Skill Subsystem

### `opencas/plugins/registry.py` — PluginRegistry

Central plugin registry.

### `opencas/plugins/skills.py` — SkillRegistry

Skill registry with capability metadata.

### `opencas/plugins/loader.py` — PluginLifecycleManager

Discovers, loads, enables, and disables plugins + skills:

- `load_all()`: Loads from built-in directory + user plugins dir
- `load_builtin_plugins(...)`: Discovers plugins in directory

**Skill entry point:**

```python
SKILL_ENTRY = {
    "name": str,
    "description": str,
    "capabilities": List[str],
    "tools": List[str],
    "hooks": List[str],        # Hook names this skill registers
    "version": str,
}
```

---

## 23. Sandbox Subsystem

### `opencas/sandbox/config.py` — SandboxConfig

Defines isolated execution roots:

```python
@dataclass
class SandboxConfig:
    allowed_roots: List[Path]
    mode: SandboxMode        # NATIVE, DOCKER
    docker_image: str
    docker_timeout: float
```

### `opencas/sandbox/docker.py` — DockerSandbox

Docker-backed shell execution:

- `check_available()`: Docker daemon reachable?
- `_ensure_running()`: Starts container if needed
- `run_command(cmd, timeout)`: Executes in container, returns output

---

## 24. Daydream Subsystem

### `opencas/daydream/generator.py` — DaydreamGenerator

Generates sparks from memory and somatic tension:

```python
class DaydreamGenerator:
    def generate(): ...  # Returns List[DaydreamReflection]
```

### `opencas/daydream/reflection.py` — DaydreamReflection

A spark with appraisal:

```python
@dataclass
class DaydreamReflection:
    reflection_id: UUID
    content: str
    appraisal: Dict[str, Any]     # "sparks_from": ["memory", "somatic"]
    confidence: float
    suggested_stage: WorkStage
```

### `opencas/daydream/resolver.py` — ReflectionResolver

Evaluates and resolves reflections into work objects or discards.

### `opencas/daydream/store.py` — DaydreamStore

Async SQLite for spark persistence.

### `opencas/daydream/conflict.py` — ConflictRegistry

Tracks and resolves internal conflicts between intentions.

---

## 25. Integration — Telegram

### `opencas/telegram_integration.py` — TelegramBotService

Long-polls Telegram for messages:

- `start()`, `stop()`
- `status()` → Bot info, poll state
- `approve_pairing(code)` → Authorize user
- `handle_update(update)` → Process incoming messages

### `opencas/telegram_config.py` — TelegramRuntimeConfig

Fields: `enabled`, `bot_token`, `dm_policy` (`disabled`, `pairing`, `allowlist`, `open`), `allow_from`, `poll_interval_seconds`, `pairing_ttl_seconds`, `api_base_url`.

---

## 26. Consolidation Subsystem

### `opencas/consolidation/engine.py` — NightlyConsolidationEngine

Runs nightly memory consolidation:

- `_merge_by_embedding()` — Clusters by vector similarity
- `_rebuild_episode_edges()` — Adds graph edges between related episodes
- `_promote_strong_signals()` — Elevates high-scoring episodes to Memory
- `_decay_and_prune()` — Edge decay + prune

### `opencas/consolidation/curation_store.py` — ConsolidationCurationStore

Records user-rejected merge clusters (so they're never re-offered).

---

## 27. Compaction Subsystem

### `opencas/compaction/compactor.py` — ConversationCompactor

Reduces conversation bulk for context windows:

- `_summarize_turns()`: LLM summarization
- `_repair_tool_pairing()`: Links tool use to its result
- `_strip_tool_details()`: Removes verbose tool payloads
- `_truncate_episode_content()`: Chops long contents

When passed a `SessionContextStore`, appends a synthetic system message summarizing compacted messages.

---

## 28. Diagnostics Subsystem

### `opencas/diagnostics/doctor.py` — Doctor

Health-check runner:

| Check | Pass Condition |
|-------|--------------|
| `check_memory_store` | SQLite reachable |
| `check_embedding_service` | Cache accessible |
| `check_identity` | Self-model readable |
| `check_llm_gateway` | ProviderManager reachable |
| `check_event_bus` | EventBus callable |
| `check_baa_queue_depth` | Queue size < 50 |
| `check_embedding_latency` | Avg embed < 10s |
| `check_compaction_lag` | No compaction in 48h |

### `opencas/diagnostics/monitor.py` — HealthMonitor

Runs `Doctor.run_all()` every 60s, emits `HealthCheckEvent` to EventBus.

---

## 29. Harness Subsystem

### `opencas/harness/harness.py` — AgenticHarness

Manages `ResearchNotebook` and `ObjectiveLoop` entities:

- `create_notebook(title)` → notebook_id
- `create_objective_loop(notebook_id, goal, strategy)` → loop_id
- `cycle()` → Emits RepairTask(s) via `ProjectOrchestrator` or LLM planning

---

## 30. Key Interfaces Summary

| Interface | Module | Description |
|-----------|--------|-------------|
| `BootstrapContext` | `bootstrap/pipeline.py` | Holds all initialized managers |
| `BootstrapPipeline(config).run()` | `bootstrap/pipeline.py` | Full staged boot → Context |
| `AgentRuntime(context)` | `runtime/agent_loop.py` | Main runtime coordinator |
| `AgentRuntime(context).converse(input)` | `runtime/agent_loop.py` | Single conversational turn |
| `AgentRuntime.run_autonomous(cycle_interval)` | `runtime/agent_loop.py` | Continuous loop |
| `AgentRuntime.run_autonomous_with_server(...)` | `runtime/agent_loop.py` | Loop + FastAPI |
| `AgentScheduler` | `runtime/scheduler.py` | Background loops |
| `LLMClient(manager).chat(messages)` | `api/llm.py` | LLM chat |
| `EmbeddingService(text).embed()` | `embeddings/service.py` | Compute embedding |
| `MemoryStore` | `memory/store.py` | Async SQLite episodes |
| `IdentityManager` | `identity/manager.py` | Self/user/continuity |
| `RelationalEngine` | `relational/engine.py` | Musubi state |
| `SomaticManager` | `somatic/manager.py` | Somatic state |
| `ToMEngine` | `tom/engine.py` | Beliefs/intentions |
| `SelfApprovalLadder.evaluate(request)` | `autonomy/self_approval.py` | Approval decision |
| `CreativeLadder` | `autonomy/creative_ladder.py` | Work promotion |
| `BoundedAssistantAgent.submit(task)` | `execution/baa.py` | Queue repair task |
| `ToolRegistry.execute(name, args)` | `tools/registry.py` | Tool execution |
| `MemoryRetriever.retrieve(query, limit)` | `context/retriever.py` | Multi-signal retrieval |
| `NightlyConsolidationEngine.cycle()` | `consolidation/engine.py` | Nightly merge |
| `ConversationCompactor.compact(context)` | `compaction/compactor.py` | Context reduction |
| `Doctor(context).run_all()` | `diagnostics/doctor.py` | Health checks |
| `ScheduleService.process_due()` | `scheduling/service.py` | Fire schedules |
| `HookBus.run(name, context)` | `infra/hook_bus.py` | Run hooks |
| `ConversationalRefusalGate.evaluate(text)` | `refusal/gate.py` | Conversational refusal |

---

## 31. Important Concepts

### 31.1 Source Hashing for Embeddings

Embeddings are keyed by a **source hash** computed from:

```python
def _build_source_hash(text: str, task_type: str) -> str:
    payload = f"{model_id}\0{task_type}\0{text}"
    return sha256(payload.encode()).hexdigest()
```

This ensures identical inputs map to the same cache entry regardless of when they were computed.

### 31.2 Multi-Signal Retrieval Fusion

The retriever fuses 8 signals using **Reciprocal Rank Fusion (RRF)**:

```
rrf_score(d) = Σ weight_s * (1 / (rank_s(d) + k))
```

where `k = 60` is a stabilizing constant.

### 31.3 Musubi Composite Score

Derived from four dimensions:

```
musubi = 0.3*TRUST + 0.3*RESONANCE + 0.2*PRESENCE + 0.2*ATTUNEMENT
```

Each dimension is clamped to `[-1.0, 1.0]`.

### 31.4 Screen State Classification

The PTY supervisor classifies terminal sessions:

```python
screen_state = {
    "active_app": str,      # "shell", "vim", "less", "git", "claude", "unknown"
    "mode": str,           # "insert", "normal", "command", "prompt"
    "prompt_type": str,     # "idle", "running", "auth", "error"
}
```

This is derived from the cleaned output and session command.

### 31.5 Tool Loop Guard

Prevents runaway tool loops:

```
MAX_ROUNDS = 24   # Max tool calls per conversation
IDENTICAL_CALL_CIRCUIT_BREAKER = 5  # Same tool+args 5x in a row → break
```

### 31.6 Command Safety Classification

The validator tokenizes commands and classifies into families:

- `SAFE`: read-only ops, git status, ls, echo
- `FILESYSTEM_DESTRUCTIVE`: rm, rmdir, mv (with care)
- `NETWORK`: curl, wget, ssh
- `PRIVILEGE_ESCALATION`: sudo, su
- `UNSAFE_SHELL`: eval, exec, source

### 31.7 Identity Rebuilder

The ToM engine can trigger autobiographical reconstruction:

```python
identity_rebuilder = IdentityRebuilder(memory, episode_graph, llm)
await identity_rebuilder.rebuild()  # Rewrites SelfModel.narrative from identity_core episodes
```

---

## 32. Database Files in `state_dir`

All under `~/.opencas/state/` by default:

| File | Store |
|------|-------|
| `memory.db` | Episodes, memories, edges |
| `tasks.db` | Pending/progress repair tasks |
| `context.db` | Conversation context |
| `embeddings.db` | Cached embeddings |
| `work.db` | Work objects |
| `commitments.db` | Goals |
| `portfolio.db` | Project clusters |
| `relational.db` | Musubi state |
| `daydream.db` | Sparks |
| `conflict.db` | Internal conflicts |
| `governance.db` | Approval ledger |
| `harness.db` | Notebooks and loops |
| `tom.db` | Beliefs and intentions |
| `plugins.db` | Plugin registry |
| `plans.db` | Plans |
| `schedules.db` | Scheduled items |
| `receipts.db` | Execution receipts |
| `curation.db` | Consolidation curation |
| `somatic.db` | Somatic snapshots |
| `identity/*` | JSON identity files |
| `provider_material/*` | Copied credentials |

---

## 33. Configuration Precedence

For OpenLLMAuth providers:

```
process env > provider_env_path > provider_config_path > ~/.open_llm_auth/config.json
```

For embedding model:

```
explicit --embedding-model-id > config.embedding_model_id > "google/gemini-embedding-2-preview"
```

For default LLM model:

```
explicit --default-llm-model > config.default_llm_model > "anthropic/claude-sonnet-4-6"
```

---

## 34. Testing

```bash
# Run all tests
pytest

# Run specific subsystem
pytest tests/test_memory.py -v
pytest tests/test_context.py -v

# Run a single test
pytest tests/test_memory.py::test_episode_storage -v
```

---

## 35. Common CLI Patterns

```bash
# Autonomous mode (no server)
python -m opencas --state-dir .opencas --workspace-root .

# With web dashboard
python -m opencas --state-dir .opencas --workspace-root . --with-server --port 8080

# With Telegram
python -m opencas --telegram-enabled --telegram-bot-token $TOKEN \
  --telegram-dm-policy pairing \
  --state-dir .opencas

# Explicit models
python -m opencas \
  --default-llm-model "anthropic/claude-sonnet-4-6" \
  --embedding-model-id "google/gemini-embedding-2-preview" \
  --state-dir .opencas
```

---

## Appendix: Key Enum Values

### ActionRiskTier

`READONLY`, `WORKSPACE_WRITE`, `SHELL_LOCAL`, `NETWORK`, `EXTERNAL_WRITE`, `DESTRUCTIVE`

### ApprovalLevel

`AUTO`, `ESCALATE`, `BLOCK`

### WorkStage

`SPARK`, `NOTE`, `ARTIFACT`, `MICRO_TASK`, `PROJECT_SEED`, `PROJECT`, `DURABLE_WORK_STREAM`

### EpisodeKind

`CONVERSATION`, `TOOL_USE`, `TOOL_RESULT`, `REFLECTION`, `ARTIFACT`, `CREATIVE`, `SCHEDULE`, `IDLE`

### EdgeKind

`SEMANTIC`, `EMOTIONAL`, `RECENCY`, `STRUCTURAL`, `CAUSAL`, `VERIFICATION`

### BeliefSubject

`SELF`, `USER`, `PARTNER`, `PROJECT`, `SYSTEM`

### IntentionStatus

`ACTIVE`, `COMPLETED`, `FAILED`, `ABANDONED`

### SandboxMode

`NATIVE`, `DOCKER`

### CommandLane

`CHAT`, `BAA`, `CONSOLIDATION`, `CRON`

---

*End of Source Map*