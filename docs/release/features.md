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
- Default embedding model: `google/embeddinggemma-300m` (upcast to 3072-dim canonical storage)
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

### Desktop Context / Body Double

The desktop-context plugin can capture an active-desktop observation and store it as runtime context. Current inputs include:

- screenshot metadata and optional OCR text
- MPRIS media state, including play/pause/seek/start/stop changes
- retrieved YouTube transcripts when a playable YouTube URL is detected
- timestamp-aligned transcript excerpts when caption timestamps and playback position are both usable
- optional local Whisper transcription of short system-audio windows for currently playing media

Live transcription is disabled by default and is enabled through `desktop_context_configure` or automatically when media commentary mode is activated from a video-commentary request. The live path records a short Pulse/PipeWire monitor-source window through `ffmpeg`, runs the local `whisper` CLI, and stores the transcript excerpt in the same observation payload as the prefetched transcript.

When both sources are present, prompts treat the retrieved transcript as the timestamped map/history and the local Whisper excerpt as the current heard segment. For livestreams or videos without usable prefetched captions, the live Whisper excerpt is the current transcript source. Live transcription is skipped when the target media is paused or when multiple playing media identities make the target ambiguous.

For a single clear probable livestream, Body Double can pause playback while speaking and then resume with a bounded catch-up speed request. It uses MPRIS `Rate` when supported and falls back to YouTube browser playback-speed shortcuts when Firefox exposes the stream through MPRIS but rejects rate writes.

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
