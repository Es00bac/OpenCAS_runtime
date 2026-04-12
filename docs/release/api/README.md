# OpenCAS API Reference

## Base Address

Current default server address:

```text
http://127.0.0.1:8080
```

API base:

```text
http://127.0.0.1:8080/api
```

## Core Non-API Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Basic liveness check |
| `GET` | `/readiness` | Runtime readiness snapshot |
| `GET` | `/dashboard` | Dashboard SPA |
| `POST` | `/chat` | Simple compatibility chat endpoint |
| `WS` | `/ws` | WebSocket bridge for chat/event traffic |

## API Domains

### Config

| Method | Path |
| --- | --- |
| `GET` | `/api/config` |
| `GET` | `/api/config/providers` |
| `GET` | `/api/config/overview` |

### Monitor

| Method | Path |
| --- | --- |
| `GET` | `/api/monitor/health` |
| `GET` | `/api/monitor/health/history` |
| `GET` | `/api/monitor/baa` |
| `GET` | `/api/monitor/embeddings` |
| `GET` | `/api/monitor/events` |
| `GET` | `/api/monitor/runtime` |

### Chat

| Method | Path |
| --- | --- |
| `GET` | `/api/chat/sessions` |
| `POST` | `/api/chat/sessions` |
| `PATCH` | `/api/chat/sessions/{session_id}` |
| `POST` | `/api/chat/sessions/{session_id}/archive` |
| `POST` | `/api/chat/sessions/{session_id}/unarchive` |
| `GET` | `/api/chat/sessions/{session_id}/history` |
| `GET` | `/api/chat/sessions/{session_id}/traces` |
| `GET` | `/api/chat/plan` |
| `GET` | `/api/chat/context-summary` |
| `POST` | `/api/chat/send` |

### Daydream

| Method | Path |
| --- | --- |
| `GET` | `/api/daydream/summary` |
| `GET` | `/api/daydream/reflections` |
| `GET` | `/api/daydream/conflicts` |
| `GET` | `/api/daydream/promotions` |

### Memory

| Method | Path |
| --- | --- |
| `GET` | `/api/memory/episodes` |
| `GET` | `/api/memory/graph` |
| `GET` | `/api/memory/stats` |
| `GET` | `/api/memory/search` |
| `GET` | `/api/memory/embedding-projection` |
| `GET` | `/api/memory/landscape` |
| `GET` | `/api/memory/node-detail` |
| `GET` | `/api/memory/retrieval-inspect` |

### Operations

Representative current operations surface:

| Method | Path |
| --- | --- |
| `GET` | `/api/operations/qualification` |
| `GET` | `/api/operations/qualification/labels/{label}` |
| `POST` | `/api/operations/qualification/reruns` |
| `GET` | `/api/operations/qualification/reruns/{request_id}` |
| `GET` | `/api/operations/validation-runs` |
| `GET` | `/api/operations/validation-runs/{run_id}` |
| `GET` | `/api/operations/hardening` |
| `GET` | `/api/operations/memory-value` |
| `GET` | `/api/operations/approval-audit` |
| `GET` | `/api/operations/costs` |
| `GET` | `/api/operations/sessions` |
| `GET` | `/api/operations/receipts` |
| `GET` | `/api/operations/tasks` |
| `GET` | `/api/operations/work` |
| `GET` | `/api/operations/commitments` |
| `GET` | `/api/operations/plans` |

There are also detailed PTY, browser, process, receipt, task, work, plan, and commitment routes mounted under `/api/operations/...`.

### Usage

| Method | Path |
| --- | --- |
| `GET` | `/api/usage/overview` |

### Identity

| Method | Path |
| --- | --- |
| `GET` | `/api/identity` |
| `GET` | `/api/identity/self` |
| `GET` | `/api/identity/user` |
| `GET` | `/api/identity/continuity` |
| `GET` | `/api/identity/musubi` |
| `GET` | `/api/identity/somatic` |

### Executive

| Method | Path |
| --- | --- |
| `GET` | `/api/executive` |
| `GET` | `/api/executive/snapshot` |
| `GET` | `/api/executive/commitments` |
| `GET` | `/api/executive/plans` |

### Telegram

| Method | Path |
| --- | --- |
| `GET` | `/api/telegram/status` |
| `POST` | `/api/telegram/config` |
| `POST` | `/api/telegram/pairings/{code}/approve` |

## Example Requests

Send a chat message:

```bash
curl -X POST http://127.0.0.1:8080/api/chat/send \
  -H "Content-Type: application/json" \
  -d '{"session_id":"default","message":"Hello OpenCAS"}'
```

Fetch chat context summary:

```bash
curl "http://127.0.0.1:8080/api/chat/context-summary"
```

Inspect memory landscape:

```bash
curl "http://127.0.0.1:8080/api/memory/landscape?limit=40&edge_kind=semantic"
```

Fetch usage overview:

```bash
curl "http://127.0.0.1:8080/api/usage/overview?window_days=7&bucket_hours=6"
```

## OpenAPI

The running server also exposes:

- `/openapi.json`
- `/docs`
- `/redoc`

Use those together with this file when you need the exact request/response schema for a running instance.
