# OpenCAS Key Terminology

This page defines the vocabulary used throughout OpenCAS documentation, code, and the operator dashboard. Terms are grouped by the subsystem they belong to.

---

## Core Concepts

**Agent**
The running OpenCAS process. It holds a persistent identity, memory, and autonomous background activity. Unlike a stateless chatbot, the agent continues operating between conversations: daydreaming, consolidating memory, and executing queued work.

**Bootstrap / BootstrapContext**
The startup sequence that initializes every subsystem in order and assembles a `BootstrapContext` object containing all wired managers. Nothing runs until bootstrap completes. The `ready` readiness state is emitted at the end of bootstrap.

**Operator**
The human (or system) that owns and configures the agent. The operator accesses the agent through the dashboard, API, or CLI. The operator is distinct from "the user" in conversations — in single-user deployments they are often the same person.

**State Directory**
The local directory where all persistent state is stored. Defaults to `./.opencas` in the current working directory. Contains SQLite databases, telemetry JSONL files, provider material, and identity snapshots. Moving or backing up this directory is how you migrate an agent between machines.

**Provider Material**
Credentials and configuration for external model providers, managed by `open_llm_auth`. Stored under `state_dir/provider_material/`. Consists of a `config.json` (model routing) and a `.env` (API keys). The TUI bootstrap wizard copies these from an existing `open_llm_auth` installation.

---

## Memory

**Episode**
A single turn or event stored in memory. Episodes have content, a source tag, timestamp, embedding, and optional tags. They are the atomic unit of memory and the input to retrieval.

**Distilled Memory**
A higher-level memory object created by consolidating related episodes into a durable, semantically stable entry. Distilled memories have their own embeddings and edge links and survive compaction.

**Memory Edge**
A weighted link between two memory objects (episodes or distilled memories). Edges are built by the consolidation engine and decayed over time. High-confidence edges surface related memories during retrieval without requiring a direct text match.

**Consolidation**
The nightly (or periodic) background process that reweights memory edges, promotes high-signal episodes to distilled memories, rebuilds autobiographical continuity, and prunes weak links. Controlled by `NightlyConsolidationEngine`.

**Compaction**
Reducing the active context window by summarizing older conversation turns into a synthetic system message. Prevents unbounded context growth without losing continuity. Managed by `ConversationCompactor`.

**Retrieval Fusion**
The process of combining multiple retrieval signals (semantic vector similarity, keyword match, recency, graph edges, salience, emotional resonance, temporal echo, reliability) into a single ranked result list using Reciprocal Rank Fusion (RRF). This is how the agent selects relevant memories for a given conversation turn.

**Embedding**
A dense vector representation of a piece of text, used for semantic similarity search. OpenCAS uses `google/gemini-embedding-2-preview` via `open_llm_auth` by default. Embeddings are cached to SQLite and never recomputed for identical source text. A deterministic local fallback embedder is used when the provider is unavailable.

**Embedding Backfill**
A background task that computes embeddings for any memory records that were stored without one (e.g. during offline operation or before the embedding model was configured).

---

## Identity

**Self-Model**
The agent's structured representation of itself: name, persona, values, traits, goals, narrative, and current somatic/relational state. Persisted across restarts and updated by identity recording, ToM belief sync, and nightly consolidation.

**User Model**
The agent's representation of the operator/user: name, bio, known preferences, and recent interaction history. Updated on each conversational turn.

**Continuity**
The identity record that tracks boot count, session IDs, and the `recent_activity` log. Gives the agent a minimal autobiographical timeline that survives restarts.

**Self-Knowledge Registry**
A versioned JSONL file of structured self-beliefs keyed by `domain` and `key` (e.g. `domain="capability"`, `key="code_writing"`). High-confidence ToM beliefs about the self are mirrored here.

---

## Somatic State

**Somatic State**
A set of physiological-analogue dimensions that modulate how the agent behaves: `arousal`, `fatigue`, `tension`, and `valence`. These are not cosmetic — they directly influence LLM temperature, prompt style, memory salience, and whether the agent recommends pausing background work.

**Somatic Modulators**
Derived from the live somatic state: `to_temperature()` adjusts LLM sampling heat, `to_prompt_style_note()` injects a directive into the system prompt, and `to_memory_retrieval_boost()` emotionally tunes memory ranking.

---

## Relational State (Musubi)

**Musubi**
The composite relational score derived from four dimensions: `trust`, `resonance`, `presence`, and `attunement`. Named after the Japanese concept of generative connection. A higher musubi score loosens self-approval thresholds, boosts creative promotion, and increases memory salience for relational content.

**Relational Engine**
The subsystem (`RelationalEngine`) that tracks the four musubi dimensions and exposes modifiers for memory, creative ladder, and self-approval. Updated over time as the agent and operator interact.

---

## Autonomy

**Self-Approval Ladder**
A risk-tiered evaluation system that decides whether the agent can proceed with an action without operator intervention. Tiers range from low-risk (auto-approved) to high-risk (escalated or blocked). Every evaluation is recorded to the durable approval ledger.

**Creative Ladder**
A seven-stage promotion system for internal creative work: `spark → note → artifact → micro_task → project_seed → project → durable_work_stream`. Work objects are promoted based on intrinsic value signals and relational boost. Demotion also occurs when value signals are weak.

**Work Object**
A unit of autonomous work tracked through the creative ladder. Has a stage, content, value signal, and optional plan and commitment links.

**Executive State**
The agent's current intention, active goals, and capacity-limited task queue. Tracks whether the agent should recommend pausing (based on somatic fatigue) and manages goal persistence across restarts.

**Daydream**
A background creative process that generates sparks from memory and somatic tension during idle periods. High-value daydreams ("keeper daydreams") are persisted to durable memory with embeddings.

---

## Execution

**BAA (Bounded Assistant Agent)**
The background execution engine. Accepts `RepairTask` objects, queues them, and executes them with concurrency limits and lane routing. Supports retry, recovery, and failure-rate throttling via `ReliabilityCoordinator`. BAA tasks feed from scheduled items, harness objectives, and operator-submitted work.

**RepairTask**
The unit of work submitted to the BAA. Has an objective, optional commitment and project links, stage history, and metadata. Progresses through `PENDING → RUNNING → SUCCEEDED / FAILED / RECOVERING`.

**Repair Executor**
Executes a `RepairTask` through a plan → execute → verify → recover loop using the tool registry and optional LLM planning. Records a git checkpoint before destructive steps when git is available.

**Lane**
A named execution queue with configurable concurrency. Current lanes: `chat` (conversational turns), `baa` (background repair tasks), `consolidation` (nightly memory work), `cron` (daydream + schedule processing). Prevents background work from crowding out conversation.

**Execution Receipt**
A durable audit record created at the end of every terminal BAA task. Stored in `receipts.db` and surfaced through the dashboard Operations tab. Answers "what did the agent actually do?"

---

## Scheduling

**Schedule Item**
A durable cron or calendar entry with a recurrence rule, action, priority, and optional commitment/plan links. Kinds: `task` (triggers BAA execution) and `event` (reminder-only, records without executing). Recurrence: `none`, `interval_hours`, `daily`, `weekly`, `weekdays`.

**Schedule Run**
An audit record for each time a schedule item fires, whether automatically or via manual trigger. Stores the resulting BAA task ID, status, and any error.

**Schedule Service**
The subsystem (`ScheduleService`) that detects due items every 60 seconds and fires them. Applies a catch-up policy: if the agent was offline and multiple occurrences were missed, it fires once for the latest due occurrence and then advances to the next future time.

---

## Theory of Mind (ToM)

**Belief**
A structured record of something the agent infers about the user or itself: subject, predicate, confidence, and optional evidence. Beliefs about the self are mirrored into the self-knowledge registry when confidence is high.

**Intention**
A record of a user or agent goal inferred from a conversation turn. Tracked with confidence and staleness so outdated intentions can be pruned.

**ToM Engine**
The subsystem (`ToMEngine`) that records beliefs and intentions on every conversational turn, detects contradictions between existing beliefs, and syncs high-confidence self-beliefs into the identity self-model.

---

## Tools and Workflow

**Tool Registry**
The central registry of callable tools available to the agent. Tools are validated through a pipeline (command safety, filesystem path enforcement, content size limits) before execution. Plugin-provided tools are auto-registered at startup.

**Workflow Tools**
Higher-level composite tools built on top of base tools. Examples: `workflow_create_writing_task` (scaffold + commitment + plan in one call), `workflow_repo_triage` (quick repo and work summary), `workflow_supervise_session` (start/observe/interact with a PTY session), and `workflow_create_schedule` (create a durable scheduled task). Designed to reduce tool-call round-trips for common operator patterns.

**Tool Loop Guard**
A circuit breaker inside the agent's tool-use loop. Stops execution after `MAX_ROUNDS` (currently 24) or if the same tool call is made identically twice in a row. Prevents infinite tool loops.

**PTY Session**
A pseudo-terminal session managed by `PtySupervisor`. Lets the agent interact with terminal applications (vim, external AI coding tools, shells) as a human would. Sessions have screen-state heuristics that classify what the terminal application is currently doing.

**HookBus**
An event bus for pre-execution policy hooks. Hook points: `PRE_TOOL_EXECUTE`, `PRE_COMMAND_EXECUTE`, `PRE_FILE_WRITE`, `PRE_CONVERSATION_RESPONSE`. Hooks can mutate or short-circuit an action before it runs.

---

## Observability

**Telemetry Store**
An append-only JSONL event log. Every meaningful runtime event (bootstrap stages, tool calls, memory operations, LLM calls) is logged with a session ID and optional span ID. Queryable through `/api/monitor/events`.

**Token Telemetry**
Per-call LLM usage recording: prompt tokens, completion tokens, latency, and estimated cost. Rolled up by session and by day. Surfaced in the Usage dashboard tab.

**Health Monitor**
A background process that runs `Doctor` checks every 60 seconds and emits health events to the event bus. Checks include memory store connectivity, embedding latency, BAA queue depth, and compaction lag.

**Doctor**
The diagnostic subsystem (`Doctor`) that runs named health checks across the full substrate and returns a `HealthReport` with `pass / warn / fail / skip` results per check. Also exposed at `/api/monitor/health`.

**Readiness State**
A state machine (`booting → ready / degraded`) that tracks whether the agent is fully initialized. The `/readiness` endpoint exposes the current state, the time it entered that state, and a history of transitions.

---

## Dashboard Tabs

| Tab | What it shows |
| --- | --- |
| **Overview** | System summary, quick status |
| **Health** | Doctor checks and health event feed |
| **Chat** | Conversational interface |
| **Operations** | PTY/browser/process sessions, receipts, work, commitments, plans, qualification |
| **Usage** | Token usage, cost, and latency over time |
| **Daydream** | Daydream sparks, keeper history |
| **Memory** | Episode and distilled memory browser |
| **Identity** | Self-model, user model, somatic, relational state |
| **Executive** | Goals, task queue, executive snapshot |
| **Schedule** | Scheduled tasks and events, calendar view, run history |
| **System** | Configuration, model routing, bootstrap config |
