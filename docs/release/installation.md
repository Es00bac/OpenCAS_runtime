# OpenCAS Installation Guide

This guide documents the repo as it exists today, not a hypothetical packaged future release.

## Prerequisites

- Python `3.11+`
- A POSIX shell for the commands below
- Network access to whichever model providers you plan to use through `open_llm_auth`
- The sibling `open_llm_auth` workspace available at `../open_llm_auth/`, because `requirements.txt` installs it as an editable dependency

Optional:

- Qdrant if you want an external vector backend instead of the default local HNSW path
- Telegram bot token if you want Telegram integration

## Install From Repo

```bash
git clone https://github.com/Es00bac/OpenCAS.git
cd opencas
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Confirm The CLI Surface

```bash
source .venv/bin/activate
python -m opencas --help
```

Current important flags:

- `--with-server`
- `--host`
- `--port`
- `--state-dir`
- `--default-llm-model`
- `--embedding-model-id`
- `--provider-config-path`
- `--provider-env-path`
- `--credential-source-config-path`
- `--credential-source-env-path`
- `--credential-profile-id`
- `--credential-env-key`
- `--telegram-enabled`
- `--telegram-disabled`
- `--tui`

## Recommended Setup Path

The most accurate and operator-friendly setup path in the current repo is the TUI bootstrap:

```bash
source .venv/bin/activate
python -m opencas --tui
```

Use it to:

- choose copied-local versus linked provider material
- select available chat and embedding models from discovered configured models
- copy specific auth profiles and environment keys into app-local provider material
- configure Telegram basics

## Manual Launch

If you already have provider material configured:

```bash
source .venv/bin/activate
python -m opencas --with-server
```

Default server address:

```text
http://127.0.0.1:8080/dashboard
```

## Manual Provider Examples

Explicit project config/env paths:

```bash
python -m opencas \
  --provider-config-path /path/to/open_llm_auth/config.json \
  --provider-env-path /path/to/open_llm_auth/.env \
  --with-server
```

Copy selected credential material into app-local state:

```bash
python -m opencas \
  --credential-source-config-path /path/to/open_llm_auth/config.json \
  --credential-source-env-path /path/to/open_llm_auth/.env \
  --credential-profile-id kimi-coding:default \
  --credential-env-key MOONSHOT_API_KEY \
  --with-server
```

Override models directly:

```bash
python -m opencas \
  --default-llm-model kimi-coding/k2p5 \
  --embedding-model-id google/gemini-embedding-2-preview \
  --with-server
```

## State And Storage

Current CLI default:

```text
./.opencas
```

Important local state files under the state directory include:

- `memory.db`
- `context.db`
- `tasks.db`
- `work.db`
- `daydream.db`
- `tom.db`
- `plans.db`
- `telemetry/`
- `provider_material/config.json`
- `provider_material/.env`

## Optional Qdrant

OpenCAS can use Qdrant, but it is optional.

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

Then set the Qdrant configuration in your bootstrap configuration before running OpenCAS.

## Troubleshooting

### `ModuleNotFoundError: open_llm_auth`

The current repo depends on the editable sibling checkout in `requirements.txt`. Make sure `../open_llm_auth/` exists, or change that dependency path before installing.

### Dashboard Does Not Open

Verify that the server is running on the current default port:

```bash
python -m opencas --with-server
```

Then open:

```text
http://127.0.0.1:8080/dashboard
```

### Wrong Provider Or Model

Open the dashboard **System** tab or run the TUI bootstrap again. The live config overview exposes:

- configured default chat model
- effective runtime chat model
- configured embedding model
- materialized provider profiles
- copied environment keys

## Next Step

Continue with the [Usage Guide](usage.md).
