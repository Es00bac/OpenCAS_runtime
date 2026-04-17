# OpenCAS

[![License: AGPL v3+](https://img.shields.io/badge/license-AGPLv3%2B-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Linux-1f6feb)
![Python](https://img.shields.io/badge/python-3.11%2B-3776ab)
![Status](https://img.shields.io/badge/status-experimental-c97b18)

OpenCAS is a persistent autonomous system with local state, durable memory, a web control plane, and provider-routed model access through [`OpenLLMAuth`](https://github.com/Es00bac/OpenLLMAuth).

It is designed to run as an operator-owned, long-lived system rather than a stateless chat wrapper.

## Core capabilities

- Persistent episodic and distilled memory backed by SQLite
- Provider-backed chat and embedding lanes via [`OpenLLMAuth`](https://github.com/Es00bac/OpenLLMAuth)
- Memory inspection, retrieval inspection, and connected atlas views
- Chat, operations, usage, daydream, identity, executive, and system dashboard surfaces
- Telegram pairing and chat integration
- Background daydreaming, creative ladder promotion, and task orchestration
- Operator-facing audit, receipt, qualification, and usage telemetry APIs

## Quickstart

OpenCAS currently depends on a local editable install of `OpenLLMAuth`.

```bash
git clone https://github.com/Es00bac/OpenCAS_runtime.git
git clone https://github.com/Es00bac/OpenLLMAuth.git
cd OpenCAS_runtime
python -m venv .venv
source .venv/bin/activate
pip install -e ../OpenLLMAuth/open_llm_auth/
pip install -r requirements.txt
python -m opencas --tui
```

The TUI bootstrap is the easiest way to configure provider material, model selection, and Telegram settings for the current repo state.

After configuration:

```bash
python -m opencas --with-server
```

Then open:

```text
http://127.0.0.1:8080/dashboard
```

## Operational model

Ground-truth deployment boundaries for the current repo:

- OpenCAS keeps its state locally under the configured state directory.
- The default CLI state directory is `./.opencas`.
- Chat and embedding traffic normally goes to whichever provider and model you configure through `OpenLLMAuth`.
- The default embedding model is `google/gemini-embedding-2-preview`.
- Embeddings have a deterministic local fallback path when provider-backed embeddings are unavailable.
- The dashboard server defaults to `127.0.0.1:8080`.

That means the project is local-state and operator-owned, but not fully local in the sense of requiring no external model provider.

## Dashboard surface

Current top-level dashboard tabs:

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

## Documentation index

| Document | Purpose |
| --- | --- |
| [Installation Guide](installation.md) | Accurate setup instructions for the current repo layout |
| [Usage Guide](usage.md) | How to run OpenCAS and use its operator surfaces |
| [Features](features.md) | Product capabilities and subsystem summary |
| [Key Terminology](terminology.md) | Definitions for OpenCAS vocabulary used in docs, code, and the dashboard |
| [API Reference](api/README.md) | HTTP and WebSocket surfaces exposed by the running server |
| [Architecture](architecture/README.md) | Runtime structure, loops, persistence, and subsystem boundaries |
| [Changelog](CHANGELOG.md) | Release notes for this documentation bundle |
| [Release Website](website/index.html) | Standalone release landing page |

## Durable Work Stream

OpenCAS also has a long-form animated companion release, *Durable Work Stream*, which walks through the architecture in verse: cold boots, memory fabric, the BAA repair pipeline, Musubi weather, the self-approval ladder, and the final system arc.

Watch it here:

<a href="https://es00bac.github.io/OpenCAS_Documentation/OpenCAS_Animated_Final_480p.mp4">
  <img src="https://es00bac.github.io/OpenCAS_Documentation/OpenCAS_Thumbnail.jpg" alt="OpenCAS: Durable Work Stream" width="100%">
</a>

_If the embedded player does not load, [download the original from the release page](https://github.com/Es00bac/OpenCAS_runtime/releases/tag/media-2026-04-14)._ 

## Verification checklist

Before calling the current release state accurate, verify these commands against the repo as shipped:

```bash
source .venv/bin/activate
python -m opencas --help
python -m opencas --with-server
pytest tests/test_dashboard_api.py -q
```

## Release checklist

- Verify the editable `OpenLLMAuth` dependency path in the quickstart still matches the current repo layout.
- Confirm the dashboard boots cleanly at `127.0.0.1:8080` and the documented tabs still exist.
- Re-run the documented verification commands before publishing release docs or screenshots.
- Make sure no local state directories, provider credentials, or operator-specific data are staged.
- Re-check the docs index links after moving or renaming release docs.

## License

OpenCAS is licensed under the GNU Affero General Public License v3.0 or later.

Copyright remains with contributors. The AGPL preserves copyright notices, requires source disclosure for redistributed and modified network deployments, and is the strongest standard copyleft fit for a networked autonomous system like this one.
