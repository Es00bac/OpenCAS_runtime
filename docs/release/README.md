# OpenCAS Release Docs

OpenCAS is a persistent autonomous agent with local state, durable memory, a web control plane, and provider-routed model access through `open_llm_auth`.

This release bundle reflects the codebase as it exists now. It does not describe an aspirational future package. If a command, endpoint, or UI surface is listed here, it should exist in the repo and in the running server.

## What This Release Includes

- Persistent episodic and distilled memory backed by SQLite
- Provider-backed chat and embedding lanes via `open_llm_auth`
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

## Recommended First Run

```bash
git clone https://github.com/Es00bac/OpenCAS.git
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

This repo is not yet packaged as a polished PyPI install. The current `requirements.txt` expects the editable gateway dependency at:

```text
../open_llm_auth/
```

If you move the repo to another machine or directory layout, update that dependency path or install `open_llm_auth` separately before running OpenCAS.

## Verification Checklist

Before calling a release artifact accurate, verify these commands on the current code:

```bash
source .venv/bin/activate
python -m opencas --help
python -m opencas --with-server
pytest tests/test_dashboard_api.py -q
```

## Licensing Note

This checkout does not currently include a root `LICENSE` file. Confirm licensing material before publishing this release bundle externally.
