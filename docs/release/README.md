# OpenCAS Release Docs

OpenCAS is a persistent autonomous agent with local state, durable memory, a web control plane, and provider-routed model access through `open_llm_auth`.

This release bundle reflects the repo as it exists now. It is not an aspirational roadmap. If a command, endpoint, dashboard surface, or website section is listed here, it should exist in the running system.

## What This Release Includes

- Persistent episodic and distilled memory backed by SQLite
- Provider-backed chat, voice, and embedding lanes via `open_llm_auth`
- Memory inspection, retrieval inspection, and connected atlas views
- Chat, operations, usage, daydream, identity, executive, schedule, platform, logs, and system dashboard surfaces
- Twilio-backed phone bridge support with owner and caller workspace separation
- Telegram pairing and chat integration
- Telegram media attachment handling for image/message context
- Compact tool-use memory and semantic tool routing for better tool selection
- Autonomous project return scheduling for unfinished work
- Opt-in desktop context capture and review tools
- Background daydreaming, creative ladder promotion, retry-aware recovery, and task orchestration
- Operator-facing audit, receipt, qualification, telemetry, and plugin-trust APIs

## Ground Truth About Deployment

- OpenCAS keeps its state locally under the configured state directory.
- The default CLI state directory is `./.opencas`.
- Chat, voice, and embedding traffic normally goes to whichever provider/model you configure through `open_llm_auth`.
- The default embedding model is now `google/embeddinggemma-300m`.
- `google/embeddinggemma-300m` is treated as the native 768-dimensional local embedding lane. Older 3072-dimensional compatibility records are historical migration concerns, not the current default.
- A deterministic local hash fallback path is retained for environments where embedding generation is blocked.
- The dashboard server defaults to `127.0.0.1:8080`.

That means the project is local-state and operator-owned, but not “fully local” in the sense of requiring no external model providers.

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
| [Installation Guide](installation.md) | Accurate setup instructions for the current repo layout |
| [Usage Guide](usage.md) | How to run OpenCAS and use its operator surfaces |
| [Features](features.md) | Product capabilities and subsystem summary |
| [Key Terminology](terminology.md) | Definitions for the OpenCAS vocabulary used in docs, code, and the dashboard |
| [API Reference](api/README.md) | HTTP and WebSocket surfaces exposed by the running server |
| [Architecture](architecture/README.md) | Runtime structure, loops, persistence, and subsystem boundaries |
| [Changelog](CHANGELOG.md) | Release notes for this documentation bundle |
| [Release Website](website/index.html) | Standalone release landing page |

## Current Release Boundaries

This release is documented as an editable source checkout. The current `requirements.txt` expects the editable gateway dependency at:

```text
../open_llm_auth/
```

The recommended first-run command clones that dependency from:

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

This checkout includes a root `LICENSE` file and is published as AGPL-3.0-or-later.
