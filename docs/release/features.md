# OpenCAS Features

This document describes the features present in the current repo state.

## Memory And Retrieval

### Persistent Memory

OpenCAS persists state locally through SQLite-backed stores for episodes, distilled memories, context history, tasks, work, plans, daydreaming, schedule runs, telemetry, and platform state.

### Retrieval Fusion

Current retrieval combines these signals:

| Signal | Meaning |
| --- | --- |
| Semantic | Vector similarity |
| Keyword | Text match |
| Recency | Time proximity |
| Salience | Importance weighting |
| Graph | Memory-edge connectivity |
| Emotional resonance | Affective alignment |
| Temporal echo | Time-pattern affinity |
| Reliability | Confidence weighting |

Somatic and relational adjustments can further modulate ranking.

### Artifact-Backed Autobiographical Memory

Authored artifacts under the managed state can be bridged into memory so recall can ground on prior authored work instead of only raw file reads.

## Embeddings

- Provider-backed embeddings are routed through `open_llm_auth`
- Default embedding model: `google/embeddinggemma-300m` with native 768-dimensional local vectors
- Embedding backfill can align stale records onto the active model
- A deterministic local fallback embedder exists when provider-backed embeddings are unavailable

## Somatic And Relational State

Current somatic state tracks:

- arousal
- fatigue
- tension
- valence
- focus
- energy
- certainty
- derived somatic tag / primary emotion

OpenCAS also exposes relational continuity through musubi and identity surfaces.

## Autonomy And Execution

### Self-Approval

OpenCAS uses a tiered approval path for ordinary versus risky actions, with evidence, historical behavior, somatic state, and boundary handling feeding the decision.

### Creative Ladder

Work can move through:

- spark
- note
- artifact
- micro-task
- project seed
- project
- durable work

### Daydreaming

Idle-time daydreaming can generate:

- reflections
- keeper memories
- conflict records
- promoted work objects

### Background Execution

The bounded assistant and retry pipeline keep long-running work from drifting:

- queued background tasks are lane-limited
- receipts record what actually happened
- retry and salvage state preserve blocked-vs-resumable intent instead of blindly replaying failures
- git and provenance checkpoints help operators inspect what changed

### Project Return

Unfinished work can be captured as a resumable project return instead of being lost after a chat turn. Return records preserve canonical artifacts, recent attempts, blocked-vs-resumable state, creative continuity, and the next intended step so the agent can decide when to return without requiring repeated operator prompts.

### Tool Intelligence

Tool selection is supported by more than a flat tool list:

- tool manifests describe capabilities and boundaries
- the semantic tool router can rank likely tools for a task
- compact tool-use memory records which tools worked for similar work
- adaptive tool-call budgets can expand for meaningful research and shrink when loops become repetitive
- task-specific objective contracts can be drafted for the work being attempted instead of relying only on static boilerplate

### Scheduling

Durable scheduled work is part of the current system:

- task schedules can trigger BAA execution
- event schedules can record reminders without execution
- recurring items support interval, daily, weekly, and weekday patterns
- schedule runs are queryable in the dashboard and API

## Operator Control Plane

### Dashboard

Current tabs:

- Overview
- Health
- Chat
- Operations
- Schedule
- Usage
- Daydream
- Memory
- Identity
- Executive
- Platform
- System
- Logs

### Operations

Operator-facing operations include:

- task and work inspection
- commitments and plans
- qualification reports and rerun tracking
- PTY, browser, and process session visibility/control
- execution receipts
- approval audit visibility
- hardening and memory-value views

### Usage Monitoring

Current usage monitoring includes:

- token telemetry
- model/source breakdowns
- recent large events
- provider telemetry notes when available
- stale-process/process-hygiene context

### Platform And Trust

OpenCAS now exposes a platform surface for extensions and capability inspection:

- canonical capability inventory
- extension install, update, disable, enable, and uninstall flows
- bundle inspection and compatibility checks
- plugin trust policies for publishers, signers, checksums, and feeds

### Logs

The logs view exposes the runtime telemetry event stream:

- event filtering by kind, session, and text
- recent session discovery
- event counts by kind
- a time-windowed event feed for operator inspection

## Channels

### Dashboard Chat

The Chat surface includes:

- session history
- current provider/model lane
- somatic state panel
- current work and executive context
- lane-aware message history
- voice capture and voice synthesis controls

### Voice

The chat surface can transcribe microphone input and synthesize spoken replies. Voice output metadata is preserved alongside the chat history so the operator can see which messages were spoken.

### Phone

OpenCAS includes a Twilio-backed phone bridge with:

- persisted phone configuration
- owner and caller-specific workspaces
- public and owner screening flows
- employer-safe caller handling
- live call status and recent-call inspection

### Telegram

Telegram integration currently supports:

- persisted configuration
- pairing and DM policy control
- typing indicators
- edited replies
- image/media attachment description for chat context when supported by the runtime channel

### Desktop Context

The optional desktop-context plugin can capture explicit operator-approved desktop context for review. It is opt-in and is intended for grounded assistance, not hidden surveillance.
- dashboard setup and status

## API Surface

The current server exposes these main API domains:

- config
- monitor
- chat
- daydream
- memory
- operations
- usage
- identity
- executive
- platform
- phone
- schedule
- telemetry
- Telegram

See [API Reference](api/README.md) for details.

## Release Truths

- OpenCAS is local-state and operator-owned.
- Chat, voice, and embedding traffic normally uses configured providers through `open_llm_auth`.
- The system is not accurately described as cloud-free by default.
