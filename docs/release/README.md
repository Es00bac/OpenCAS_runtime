# OpenCAS Release Documentation

This documentation describes the public OpenCAS runtime snapshot in this repository. It focuses on what is present in the code and dashboard surfaces, not on comparisons with other projects.

OpenCAS is a Python runtime for a persistent autonomous agent. It stores local state, builds memory over time, exposes a web dashboard, and routes model calls through `open_llm_auth`.

## Current Scope

OpenCAS currently includes:

- SQLite-backed episodic memory, distilled memory, context history, tasks, work, plans, daydream state, schedule runs, telemetry, and platform state
- provider-routed chat, voice, and embedding lanes through `open_llm_auth`
- memory search, retrieval inspection, graph/atlas views, and embedding projection surfaces
- dashboard tabs for Overview, Health, Chat, Operations, Schedule, Usage, Daydream, Memory, Identity, Executive, Platform, System, and Logs
- background daydreaming, creative ladder work promotion, bounded assistant execution, retry/salvage receipts, and schedule-triggered work
- Telegram configuration and chat integration
- Twilio-backed phone bridge support when configured
- extension inventory, bundle inspection, lifecycle controls, and plugin trust policy surfaces

## Deployment Model

- State is stored locally under the configured state directory.
- The default CLI state directory is `./.opencas`.
- Provider credentials and model routing are managed through `open_llm_auth`.
- Chat, voice, and embedding calls normally use configured external providers unless you replace those lanes with local provider implementations.
- The dashboard server defaults to `127.0.0.1:8080`.
- The default embedding model in this snapshot is `google/embeddinggemma-300m`.
- Embedding generation has a deterministic local fallback for unavailable provider-backed embeddings.

## Install

```bash
git clone https://github.com/Es00bac/OpenCAS_runtime.git OpenCAS
cd OpenCAS
git clone https://github.com/Es00bac/OpenLLMAuth.git ../open_llm_auth
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

The TUI bootstrap is the recommended setup path for this snapshot:

```bash
python -m opencas --tui
```

It can configure provider material, model selection, and Telegram settings.

Fresh non-TUI bootstraps require an explicit acknowledgement because a new state directory creates persistent agent continuity:

```bash
python -m opencas --with-server --accept-bootstrap-responsibility
```

After the state directory already exists, the acknowledgement flag is not required.

## Run

```bash
source .venv/bin/activate
python -m opencas --with-server
```

Then open:

```text
http://127.0.0.1:8080/dashboard
```

## Documentation Index

| Document | Purpose |
| --- | --- |
| [Installation Guide](installation.md) | Setup steps for the current repository layout |
| [Usage Guide](usage.md) | Running OpenCAS and using its dashboard/API surfaces |
| [Features](features.md) | Current subsystem and capability summary |
| [Terminology](terminology.md) | Vocabulary used by the docs, dashboard, and code |
| [API Reference](api/README.md) | HTTP and WebSocket surfaces exposed by the server |
| [Architecture](architecture/README.md) | Runtime structure, loops, persistence, and subsystem boundaries |
| [Changelog](CHANGELOG.md) | Release notes for this public snapshot |
| [Website Source](website/index.html) | Static documentation site included with the release |

The separate GitHub Pages site is maintained at:

```text
https://es00bac.github.io/OpenCAS_Documentation/
```

## Media

The release includes an optional architecture video, *OpenCAS: Durable Work Stream*. It is a generated media artifact that walks through several subsystem names and runtime concepts.

```text
https://github.com/Es00bac/OpenCAS_runtime/releases/tag/media-2026-04-14
```

## Current Boundaries

- This repository is not packaged as a PyPI release.
- `requirements.txt` expects the editable `open_llm_auth` dependency at `../open_llm_auth/`.
- The public snapshot should not contain `.opencas`, provider `.env` files, private operator notes, or live state databases.
- The docs describe the current public snapshot and may lag behind private live development.

## Verification

Useful checks for this snapshot:

```bash
source .venv/bin/activate
python -m opencas --help
python -m opencas --with-server --accept-bootstrap-responsibility
pytest tests/test_dashboard_api.py -q
```

## License

OpenCAS is released under AGPL-3.0-or-later. See `LICENSE`.
