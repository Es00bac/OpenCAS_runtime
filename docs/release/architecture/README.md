# OpenCAS Architecture

## Overview

OpenCAS is structured as explicit subsystems rather than one monolithic agent file. The major runtime concerns are separated into bootstrap, memory, context/retrieval, autonomy, telemetry, API, dashboard, and channel integrations.

## Primary Subsystems

| Path | Role |
| --- | --- |
| `opencas/bootstrap/` | Configuration resolution, store setup, provider material, runtime assembly |
| `opencas/runtime/` | Conversational loop, scheduler, autonomous cycles, Telegram lifecycle |
| `opencas/memory/` | Episodes, distilled memories, graph edges, artifact-backed memory ingestion |
| `opencas/context/` | Session context, retrieval fusion, resonance logic, prompt assembly |
| `opencas/embeddings/` | Embedding service, cache, local fallback, vector backend wiring |
| `opencas/autonomy/` | Self-approval, executive state, creative ladder, work store |
| `opencas/daydream/` | Reflection and conflict storage for the daydream subsystem |
| `opencas/identity/` | Self-model, user model, continuity |
| `opencas/somatic/` | Somatic dimensions and modulators |
| `opencas/relational/` | Musubi and relational influence |
| `opencas/tom/` | Theory-of-mind belief and intention tracking |
| `opencas/api/` | FastAPI server and route groups |
| `opencas/dashboard/` | Operator-facing SPA |

## Runtime Loops

### Conversation Loop

1. User message arrives through dashboard, API, or Telegram.
2. Context is assembled from recent session state and retrieval results.
3. The runtime decides whether tool use is needed.
4. The assistant response is persisted with lane and somatic metadata.

### Creative / Background Loop

1. Scheduler checks for idle/background opportunities.
2. Creative ladder and work queues are evaluated.
3. Daydreaming may run if cooldown and readiness conditions allow it.
4. Promoted work and keeper memories are persisted.

### Consolidation Loop

1. Periodic memory maintenance runs.
2. Edges and long-horizon continuity state can be reweighted or rebuilt.

## Persistence

The state directory is configurable. Under the current CLI, the default is:

```text
./.opencas
```

Important persisted state includes:

- `memory.db`
- `context.db`
- `tasks.db`
- `work.db`
- `plans.db`
- `daydream.db`
- `tom.db`
- `telemetry/`
- `provider_material/config.json`
- `provider_material/.env`

## Provider And Model Architecture

OpenCAS uses `open_llm_auth` as its model gateway.

- chat and tool-use completions use the configured default model lane
- embeddings use the configured embedding lane
- provider material can be linked from an existing config/env or copied into app-local state
- the dashboard System tab exposes both configured defaults and effective runtime models

This means the system is local-state and operator-controlled, but the default model execution path is provider-backed rather than purely local.

## API And Dashboard

The FastAPI app mounts the following main route groups:

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

The dashboard surfaces those through the current tab set:

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

## Architectural Truths

- The repo currently has a strong operator surface and broad internal observability.
- The repo is not yet documented here as a polished package manager install.
- The release docs should not describe the system as cloud-free unless the configured model lanes are truly local.
