# OpenCAS Features

This document describes the features present in the current repo state.

## Memory And Retrieval

### Persistent Memory

OpenCAS persists state locally through SQLite-backed stores for episodes, distilled memories, context history, tasks, work, plans, daydreaming, and telemetry.

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
- Default embedding model: `google/gemini-embedding-2-preview`
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

## Autonomy

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

## Operator Control Plane

### Dashboard

Current tabs:

- Overview
- Health
- Chat
- Operations
- Usage
- Daydream
- Memory
- Identity
- Executive
- System

### Operations

Operator-facing operations include:

- task and work inspection
- commitments and plans
- qualification reports and rerun tracking
- PTY, browser, and process session visibility/control
- execution receipts
- approval audit visibility

### Usage Monitoring

Current usage monitoring includes:

- token telemetry
- model/source breakdowns
- recent large events
- provider telemetry notes when available
- stale-process/process-hygiene context

## Channels

### Dashboard Chat

The Chat surface includes:

- session history
- current provider/model lane
- somatic state panel
- current work and executive context
- lane-aware message history

### Telegram

Telegram integration currently supports:

- persisted configuration
- pairing and DM policy control
- typing indicators
- edited replies
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
- Telegram

See [API Reference](api/README.md) for details.

## Release Truths

- OpenCAS is local-state and operator-owned.
- Chat and embedding traffic normally uses configured providers through `open_llm_auth`.
- It is not accurate to describe the default setup as cloud-free.
