# OpenCAS Architecture

## Overview

OpenCAS is structured as explicit subsystems rather than one monolithic agent file. The major runtime concerns are separated into bootstrap, memory, context/retrieval, autonomy, scheduling, telemetry, API, dashboard, platform trust, phone, and channel integrations.

## Primary Subsystems

| Path | Role |
| --- | --- |
| `opencas/bootstrap/` | Configuration resolution, store setup, provider material, runtime assembly |
| `opencas/runtime/` | Conversational loop, scheduler, autonomous cycles, Telegram and phone lifecycle |
| `opencas/memory/` | Episodes, distilled memories, graph edges, artifact-backed memory ingestion |
| `opencas/context/` | Session context, retrieval fusion, resonance logic, prompt assembly |
| `opencas/embeddings/` | Embedding service, cache, native 768-dimensional Gemma lane, local fallback, vector backend wiring |
| `opencas/autonomy/` | Self-approval, executive state, creative ladder, commitment and work stores |
| `opencas/scheduling/` | Durable scheduled tasks, calendar views, and run history |
| `opencas/platform/` | Capability inventory, extension descriptors, trust surfaces |
| `opencas/telemetry/` | Append-only runtime event stream and log/query helpers |
| `opencas/daydream/` | Reflection and conflict storage for the daydream subsystem |
| `opencas/desktop_context/` | Explicit opt-in desktop capture and context packaging |
| `opencas/identity/` | Self-model, user model, continuity |
| `opencas/somatic/` | Somatic dimensions and modulators |
| `opencas/relational/` | Musubi and relational influence |
| `opencas/tom/` | Theory-of-mind belief and intention tracking |
| `opencas/api/` | FastAPI server and route groups |
| `opencas/dashboard/` | Operator-facing SPA |

## Runtime Loops

### Conversation And Voice Loop

1. User message arrives through dashboard, API, Telegram, or phone bridge.
2. Context is assembled from recent session state and retrieval results.
3. The runtime decides whether tool use is needed.
4. Voice input can be transcribed and voice output can be synthesized when the chat lane is configured for it.
5. The assistant response is persisted with lane and somatic metadata.

### Creative / Background Loop

1. Scheduler checks for idle/background opportunities.
2. Creative ladder and work queues are evaluated.
3. Daydreaming may run if cooldown and readiness conditions allow it.
4. Promoted work and keeper memories are persisted.

### Scheduling Loop

1. The schedule service checks for due items on its fixed cadence.
2. Task schedules can submit new BAA work.
3. Reminder schedules can emit durable run records without execution.
4. Missed runs are advanced conservatively instead of being blindly replayed.

### Project Return Loop

1. Long-running creative or execution work records a return snapshot when it stops before true completion.
2. The snapshot preserves canonical artifacts, attempts, blocked state, and a bounded next step.
3. The runtime can surface or requeue the work later when it has capacity or new context.
4. Completion checks reject false "done" states when required artifact changes did not happen.

### Consolidation Loop

1. Periodic memory maintenance runs.
2. Edges and long-horizon continuity state can be reweighted or rebuilt.
3. Retried work keeps salvage and blocked-state provenance instead of being reshuffled into unrelated work.

### Telemetry Loop

1. Runtime events are appended to the telemetry store.
2. Logs and usage views query the same event and usage sources.
3. Operators can inspect the event stream through the dashboard and API without touching raw files.

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
- schedule and phone configuration under the runtime-managed workspace

## Provider And Model Architecture

OpenCAS uses `open_llm_auth` as its model gateway.

- chat and tool-use completions use the configured default model lane
- voice lanes and embeddings use the configured provider material when available
- the default embedding lane is `google/embeddinggemma-300m`, native 768 dimensions
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
- platform
- phone
- schedule
- telemetry
- Telegram

The dashboard surfaces those through the current tab set:

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

## Architectural Truths

- The repo currently has a strong operator surface and broad internal observability.
- The repo is documented here as an editable source checkout, not a package-manager install.
- The release docs should not describe the system as cloud-free unless the configured model lanes are truly local.
- Phone, voice, schedule, platform, and telemetry are first-class runtime surfaces, not side experiments.
