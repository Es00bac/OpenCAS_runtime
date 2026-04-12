# OpenCAS Usage Guide

## Recommended Operator Path

The current repo is best operated through the dashboard and API server:

```bash
source .venv/bin/activate
python -m opencas --with-server
```

Default address:

```text
http://127.0.0.1:8080/dashboard
```

## Important Accuracy Note

The current CLI does **not** expose a separate interactive `python -m opencas chat` terminal mode. Conversation happens through:

- the dashboard Chat tab
- `POST /api/chat/send`
- the WebSocket bridge at `/ws`
- Telegram, when enabled and configured

## Useful CLI Flags

```bash
python -m opencas --help
```

High-value current flags:

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

## Dashboard Tabs

| Tab | Purpose |
| --- | --- |
| Overview | High-level runtime, health, token, and memory summaries |
| Health | Doctor state, health history, runtime and event visibility |
| Chat | Sessions, current lane, somatic context, current work, and message send |
| Operations | Tasks, work, commitments, plans, receipts, qualification, and live process control |
| Usage | Token telemetry, dominant sources/models, provider notes, and process hygiene clues |
| Daydream | Reflections, conflicts, keeper status, and daydream-origin work |
| Memory | Atlas, search, embedding projection, node detail, and retrieval inspection |
| Identity | Self-model, user-model, continuity, musubi, and somatic state |
| Executive | Intent, goals, commitments, plans, and executive snapshot |
| System | Effective config, configured model options, and Telegram control |

## Chat Behavior

- The Chat sidebar shows the active provider/model lane currently in use.
- New assistant messages persist lane metadata in message history.
- Older assistant messages that predate lane metadata should be treated as legacy history.
- The chat context rail also surfaces somatic state, current work, executive intent, and task counts.

## Memory And Recall

OpenCAS memory behavior is currently exposed through both runtime behavior and the Memory dashboard tab:

- episodic and distilled memories
- semantic retrieval using provider-backed embeddings by default
- artifact-backed autobiographical memory for authored plans/stories/notes
- retrieval inspection endpoints and UI
- connected graph and landscape views

## Daydreaming

Daydreaming is a background autonomy path, not a chat command. When idle conditions and cooldown rules allow it, OpenCAS can produce:

- reflections
- keeper memories
- daydream-origin work objects
- conflict records

These are visible through the Daydream dashboard tab and the `/api/daydream/*` routes.

## Telegram

Telegram is a first-class channel in the current repo. Use the System tab or the TUI bootstrap to configure:

- bot token
- pairing/allowlist/open DM policy
- allowlisted Telegram user ids
- poll interval and pairing TTL

## Usage And Cost Monitoring

The Usage tab is the operator-facing place to inspect:

- token totals
- dominant usage sources
- dominant models
- recent large usage events
- provider telemetry notes when available
- process hygiene context for runaway spend investigations

## Practical Workflow

1. Launch with `python -m opencas --with-server`.
2. Open the Chat tab and confirm the active provider/model lane.
3. Use the System tab if the effective model looks wrong.
4. Use Memory when debugging recall quality.
5. Use Operations when debugging task state, receipts, or background processes.
6. Use Usage when debugging spend or rate-limit surprises.

## Stopping OpenCAS

Stop the server with `Ctrl+C` in the terminal running the process.

## Related Docs

- [Installation Guide](installation.md)
- [Features](features.md)
- [API Reference](api/README.md)
- [Architecture](architecture/README.md)
- [Release Website](website/index.html)
