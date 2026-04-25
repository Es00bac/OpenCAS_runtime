# OpenCAS

OpenCAS is a local-first autonomous agent runtime with persistent memory, a web dashboard, provider-routed model access, and bounded execution surfaces.

This public repository is intentionally limited to the code needed to bootstrap a fresh agent. It does not include private state, personal workspaces, deployment secrets, operator documents, or local runtime artifacts.

## What’s Included

- The `opencas/` runtime and dashboard code
- The public test suite for the shipped code surface
- Generic maintenance utilities that are safe to publish

## Quick Start

`requirements.txt` expects a sibling checkout of `open_llm_auth` at `../open_llm_auth/`.

```bash
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

## Notes

- State defaults to a local `.opencas/` directory unless you configure a different location.
- Provider credentials and optional voice/phone integrations are expected to be configured in your own environment.
- The public tree does not ship any preloaded agent identity or personal workspace content.
- OpenCAS is released under AGPL-3.0-or-later; see `LICENSE`.

## Basic Verification

```bash
source .venv/bin/activate
python -m opencas --help
pytest tests/test_dashboard_api.py tests/test_phone_integration.py -q
```
