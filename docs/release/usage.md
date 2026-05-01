# OpenCAS Usage Guide

## Recommended Operator Path

The current repo is best operated through the dashboard and API server:

```bash
source .venv/bin/activate
python -m opencas --with-server --accept-bootstrap-responsibility
```

The acknowledgement flag is required only for non-TUI fresh bootstraps. It makes the first-boot boundary explicit: OpenCAS creates persistent continuity, not a disposable chat session, and deleting the state directory deletes that agent's continuity. Later launches against an existing state directory do not require the flag.

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
- `--accept-bootstrap-responsibility`

## Dashboard Tabs

| Tab | Purpose |
| --- | --- |
| Overview | High-level runtime, health, token, and memory summaries |
| Health | Doctor status, recent health history, runtime and event visibility |
| Chat | Session history, current provider/model lane, somatic context, current work, voice controls, and message send |
| Operations | Receipts, tasks, work objects, qualification runs, approval audit, and live process control |
| Schedule | Durable schedule items, calendar view, and run history |
| Usage | Token telemetry, dominant models and sources, provider telemetry notes, and process hygiene clues |
| Daydream | Reflections, conflicts, keeper promotion, and daydream-origin work |
| Memory | Atlas, search, embedding projection, node detail, and retrieval inspection |
| Identity | Self-model, user-model, continuity, musubi, and somatic state |
| Executive | Goals, plans, commitments, and executive snapshot |
| Platform | Capability inventory, extension lifecycle, and plugin trust control |
| System | Effective config, configured model options, Telegram, and phone setup/status |
| Logs | Runtime telemetry event stream and filtering controls |

## Chat Behavior

- The Chat sidebar shows the active lane: provider, resolved model, auth profile, and auth source.
- New assistant messages carry their own lane metadata in history.
- Older assistant messages that predate lane metadata should be treated as legacy historical entries.
- The chat context panel also surfaces somatic state, current work, executive intent, and recent background-task counts.
- Voice controls let the operator record a prompt, transcribe it, and synthesize spoken replies when configured.

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

## Schedule

Use the Schedule tab when you need durable recurring work instead of a one-off chat action:

- create task or reminder schedules
- inspect future run windows
- trigger or cancel a schedule from the dashboard
- review run history for missed or repeated items

## Platform

Use the Platform tab when you need to inspect or change the extension surface:

- view canonical capabilities
- inspect extension bundles before install or update
- manage extension lifecycle state
- adjust plugin trust policies and trust feed sync

## Phone

Phone configuration is exposed through the System surface and the `/api/phone/*` routes:

- owner and caller workspace separation
- public base URL and webhook settings
- Twilio and voice model configuration
- recent call review and owner call initiation

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

## Logs

The Logs tab is the event stream when you need to answer “what happened?” quickly:

- filter by event kind
- filter by session
- search by text
- inspect a recent timeline without opening the raw telemetry store

## Practical Workflow

1. Launch with `python -m opencas --with-server`, adding `--accept-bootstrap-responsibility` if this is a non-TUI fresh bootstrap.
2. Open the Chat tab and confirm the active provider/model lane.
3. Use the System tab if the effective model, Telegram, phone, or voice setup looks wrong.
4. Use Memory when debugging recall quality.
5. Use Operations when debugging task state, receipts, or background processes.
6. Use Schedule when debugging recurring work or missed runs.
7. Use Platform when debugging extension installation or trust policy issues.
8. Use Logs when you need the raw event sequence.
9. Use Usage when debugging spend or rate-limit surprises.

## Stopping OpenCAS

Stop the server with `Ctrl+C` in the terminal running the process.

## Related Docs

- [Installation Guide](installation.md)
- [Features](features.md)
- [API Reference](api/README.md)
- [Architecture](architecture/README.md)
- [Release Website](website/index.html)
