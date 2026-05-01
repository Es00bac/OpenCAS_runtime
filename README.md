# OpenCAS

OpenCAS is a local-state autonomous agent runtime with persistent memory, a dashboard control plane, provider-routed model access, scheduled work, daydreaming, background execution, and reviewable tool/plugin surfaces.

This public repository is a clean runtime release. It includes source, public tests, release documentation, the static documentation website, and generic maintenance utilities. It does not include private state, personal workspaces, deployment secrets, local runtime databases, or a preloaded agent identity.

## What The System Includes

- `opencas/`: the Python runtime, FastAPI server, dashboard assets, memory stores, autonomy loops, scheduling, phone/Telegram/voice integrations, tool registry, and plugin trust surfaces.
- `plugins/`: bundled example plugins with manifests for calculator, codec, diff, HTTP request, JSON tools, notes, system stats, time tools, and opt-in desktop context.
- `tests/`: the public regression suite for the shipped runtime surface.
- `docs/release/`: comprehensive release docs plus a standalone website under `docs/release/website/`.
- `OPENCAS_PRODUCT_SPEC.md`: product principles and target architecture for the autonomous-agent system.

## Quick Start

`requirements.txt` expects a sibling checkout of `open_llm_auth` at `../open_llm_auth/`. Replace that editable dependency if your deployment installs the gateway another way.

```bash
git clone https://github.com/Es00bac/OpenCAS_runtime.git OpenCAS
cd OpenCAS
git clone <open-llm-auth-repo-url> ../open_llm_auth
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m opencas --tui
```

After initial configuration:

```bash
python -m opencas --with-server
```

Then open `http://127.0.0.1:8080/dashboard`.

First boot is responsibility-gated. OpenCAS creates persistent continuity, not a disposable chat session. If you later delete the state directory, you delete that agent's continuity. The TUI asks you to acknowledge this before creation.

For a non-TUI fresh bootstrap, acknowledge that boundary explicitly:

```bash
python -m opencas --with-server --accept-bootstrap-responsibility
```

## Runtime Defaults

- State defaults to `./.opencas/` unless `--state-dir` is set.
- The default embedding model is `google/embeddinggemma-300m`, resolved as native 768-dimensional local embeddings.
- Chat, voice, and non-local model traffic route through the configured `open_llm_auth` provider material.
- Qdrant is optional; the local HNSW/vector cache path works for a clean install.
- Voice, phone, Telegram, and desktop-context features require operator-supplied credentials or explicit enablement.

## Documentation

- [Release docs](docs/release/README.md)
- [Installation](docs/release/installation.md)
- [Usage](docs/release/usage.md)
- [Features](docs/release/features.md)
- [Architecture](docs/release/architecture/README.md)
- [API reference](docs/release/api/README.md)
- [Website](docs/release/website/index.html)

Serve the website locally:

```bash
./serve_docs.sh
```

## Basic Verification

```bash
source .venv/bin/activate
python -m opencas --help
pytest tests/test_dashboard_api.py tests/test_phone_integration.py -q
```

OpenCAS is released under AGPL-3.0-or-later; see `LICENSE`.
