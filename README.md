# OpenCAS Release Docs

OpenCAS is a persistent autonomous agent with local state, durable memory, a web control plane, and provider-routed model access through [`open_llm_auth`](https://github.com/Es00bac/OpenLLMAuth).

This release bundle reflects the codebase as it exists now. It does not describe an aspirational future package. If a command, endpoint, or UI surface is listed here, it should exist in the repo and in the running server.

## What This Release Includes

- Persistent episodic and distilled memory backed by SQLite
- Provider-backed chat and embedding lanes via [`open_llm_auth`](https://github.com/Es00bac/OpenLLMAuth)
- Memory inspection, retrieval inspection, and connected atlas views
- Chat, operations, usage, daydream, identity, executive, and system dashboard surfaces
- Telegram pairing and chat integration
- Background daydreaming, creative ladder promotion, and task orchestration
- Operator-facing audit, receipt, qualification, and usage telemetry APIs

## Ground Truth About Deployment

- OpenCAS keeps its state locally under the configured state directory.
- The default CLI state directory is `./.opencas`.
- Chat and embedding traffic normally goes to whichever provider/model you configure through `open_llm_auth`.
- The default embedding model is `google/gemini-embedding-2-preview`.
- Embeddings have a deterministic local fallback path when provider-backed embeddings are unavailable.
- The dashboard server defaults to `127.0.0.1:8080`.

That means the project is local-state and operator-owned, but not “fully local” in the sense of requiring no external model providers.

## OpenCAS: Durable Work Stream

OpenCAS isn't just a system — it's a concept big enough to fill a 73-minute cyber-noir rap opera. *Durable Work Stream* is a 15-track animated album that walks through the entire architecture in verse: cold boots, memory fabric, the BAA repair pipeline, Musubi weather, the self-approval ladder, and the finale that ties it all together.

If you want to understand OpenCAS in one sitting, watch the full animated video:

<video src="https://github.com/Es00bac/OpenCAS_runtime/releases/download/media-2026-04-14/OpenCAS_Animated_Final.mp4" controls width="100%"></video>

_If the player doesn't load, [download or stream it directly from the release page](https://github.com/Es00bac/OpenCAS_runtime/releases/tag/media-2026-04-14)._

## Recommended First Run

```bash
git clone https://github.com/Es00bac/OpenCAS_runtime.git
cd opencas
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m opencas --tui
```

The TUI bootstrap is the most user-friendly way to configure provider material, model selection, and Telegram settings for the current repo state.

After configuration:

```bash
python -m opencas --with-server
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
- Usage
- Daydream
- Memory
- Identity
- Executive
- System

## Documentation Index

| Document | Purpose |
| --- | --- |
| [Installation Guide](installation.md) | Accurate setup instructions for the current repo layout |
| [Usage Guide](usage.md) | How to run OpenCAS and use its operator surfaces |
| [Features](features.md) | Product capabilities and subsystem summary |
| [Key Terminology](terminology.md) | Definitions for all OpenCAS vocabulary used in docs, code, and the dashboard |
| [API Reference](api/README.md) | HTTP and WebSocket surfaces exposed by the running server |
| [Architecture](architecture/README.md) | Runtime structure, loops, persistence, and subsystem boundaries |
| [Changelog](CHANGELOG.md) | Release notes for this documentation bundle |
| [Release Website](website/index.html) | Standalone release landing page |

## Current Release Boundaries

This repo is not yet packaged as a polished PyPI install. The `requirements.txt` references `open_llm_auth` as an editable dependency. To install it:

```bash
git clone https://github.com/Es00bac/OpenLLMAuth.git
pip install -e ./OpenLLMAuth/open_llm_auth/
```

Then install OpenCAS's remaining dependencies with `pip install -r requirements.txt`.

## Verification Checklist

Before calling a release artifact accurate, verify these commands on the current code:

```bash
source .venv/bin/activate
python -m opencas --help
python -m opencas --with-server
pytest tests/test_dashboard_api.py -q
```
