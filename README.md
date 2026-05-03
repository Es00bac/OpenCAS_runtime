# OpenCAS Release Docs

OpenCAS is a persistent autonomous agent with local state, durable memory, a web control plane, and provider-routed model access through `open_llm_auth`.

This release bundle reflects the codebase as it exists now. It does not describe an aspirational future package. If a command, endpoint, or UI surface is listed here, it should exist in the repo and in the running server.

## What This Release Includes

- Persistent episodic and distilled memory backed by SQLite
- Provider-backed chat, voice, and embedding lanes via `open_llm_auth`
- Memory inspection, retrieval inspection, and connected atlas views
- Chat, operations, usage, daydream, identity, executive, schedule, platform, logs, and system dashboard surfaces
- Twilio-backed phone bridge support with owner and caller workspace separation
- Telegram pairing and chat integration
- Background daydreaming, creative ladder promotion, and task orchestration
- Operator-facing audit, receipt, qualification, telemetry, and plugin-trust APIs

## Ground Truth About Deployment

- OpenCAS keeps its state locally under the configured state directory.
- The default CLI state directory is `./.opencas`.
- Chat, voice, and embedding traffic normally goes to whichever provider/model you configure through `open_llm_auth`.
- The default embedding model is `google/embeddinggemma-300m`.
- Embeddings have a deterministic local fallback path when provider-backed embeddings are unavailable.
- The dashboard server defaults to `127.0.0.1:8080`.

That means the project is local-state and operator-owned, but not "fully local" in the sense of requiring no external model providers.

## OpenCAS: Durable Work Stream

OpenCAS isn't just a system — it's a concept big enough to fill a 73-minute cyber-noir rap opera. *Durable Work Stream* is a 15-track animated album that walks through the entire architecture in verse: cold boots, memory fabric, the BAA repair pipeline, Musubi weather, the self-approval ladder, and the finale that ties it all together.

If you want to understand OpenCAS in one sitting, watch the full animated video:

https://github.com/Es00bac/OpenCAS_runtime/releases/download/media-2026-04-14/OpenCAS_Animated_Final.mp4

_If the link above doesn't stream inline in your browser, click through to the release page to download or stream it directly._

## Recommended First Run

```bash
git clone https://github.com/Es00bac/OpenCAS_runtime.git OpenCAS
cd OpenCAS
git clone https://github.com/Es00bac/OpenLLMAuth.git ../open_llm_auth
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m opencas --tui
```

The TUI bootstrap is the most user-friendly way to configure provider material, model selection, and Telegram settings for the current repo state.

First boot is responsibility-gated. OpenCAS creates persistent continuity, not a disposable chat session. If you later delete the state directory, you delete that agent's continuity. The TUI asks you to acknowledge this before creation.

After configuration:

```bash
python -m opencas --with-server
```

For a non-TUI fresh bootstrap, acknowledge that boundary explicitly:

```bash
python -m opencas --with-server --accept-bootstrap-responsibility
```

Then open:

```text
http://127.0.0.1:8080/dashboard
```

## Dashboard Surface

The current dashboard includes these top-level tabs:

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

## Documentation Index

| Document | Purpose |
| --- | --- |
| [Installation Guide](docs/release/installation.md) | Accurate setup instructions for the current repo layout |
| [Usage Guide](docs/release/usage.md) | How to run OpenCAS and use its operator surfaces |
| [Features](docs/release/features.md) | Product capabilities and subsystem summary |
| [Key Terminology](docs/release/terminology.md) | Definitions for the OpenCAS vocabulary used in docs, code, and the dashboard |
| [API Reference](docs/release/api/README.md) | HTTP and WebSocket surfaces exposed by the running server |
| [Architecture](docs/release/architecture/README.md) | Runtime structure, loops, persistence, and subsystem boundaries |
| [Changelog](docs/release/CHANGELOG.md) | Release notes for this documentation bundle |
| [Release Website](docs/release/website/index.html) | Standalone release landing page |

## Current Release Boundaries

This repo is not yet packaged as a polished PyPI install. The current `requirements.txt` expects the editable gateway dependency at:

```text
../open_llm_auth/
```

The public gateway repo is:

```text
https://github.com/Es00bac/OpenLLMAuth.git
```

If you move the repo to another machine or directory layout, update that dependency path or install `open_llm_auth` separately before running OpenCAS.

## Verification Checklist

Before calling a release artifact accurate, verify these commands on the current code:

```bash
source .venv/bin/activate
python -m opencas --help
python -m opencas --with-server --accept-bootstrap-responsibility
pytest tests/test_dashboard_api.py -q
```

## Licensing Note

OpenCAS is released under AGPL-3.0-or-later; see `LICENSE`.
