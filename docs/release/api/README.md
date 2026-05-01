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
| `POST` | `/api/config/model-routing` |
| `POST` | `/api/config/provider-setups` |
| `POST` | `/api/config/provider-test` |
| `DELETE` | `/api/config/auth-profiles/{profile_id}` |
| `DELETE` | `/api/config/providers/{provider_id}` |
| `DELETE` | `/api/config/providers/{provider_id}/models/{model_id}` |
| `GET` | `/api/config/web-trust` |
| `POST` | `/api/config/web-trust/policies` |
| `DELETE` | `/api/config/web-trust/policies/{domain}` |
| `GET` | `/api/config/plugin-trust` |
| `POST` | `/api/config/plugin-trust/policies` |
| `POST` | `/api/config/plugin-trust/feeds/sync` |
| `DELETE` | `/api/config/plugin-trust/policies/{scope}/{value}` |

### Monitor

| Method | Path |
| --- | --- |
| `GET` | `/api/monitor/health` |
| `GET` | `/api/monitor/health/history` |
| `GET` | `/api/monitor/baa` |
| `GET` | `/api/monitor/embeddings` |
| `GET` | `/api/monitor/events` |
| `GET` | `/api/monitor/task-beacon` |
| `GET` | `/api/monitor/runtime` |
| `GET` | `/api/monitor/meaningful-loop` |
| `GET` | `/api/monitor/affective-examinations` |
| `GET` | `/api/monitor/shadow-registry` |
| `GET` | `/api/monitor/shadow-registry/cluster` |
| `POST` | `/api/monitor/shadow-registry/cluster/triage` |
| `GET` | `/api/monitor/web-trust` |
| `GET` | `/api/monitor/plugin-trust` |

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
| `GET` | `/api/chat/voice/status` |
| `POST` | `/api/chat/voice/transcribe` |
| `POST` | `/api/chat/voice/synthesize` |
| `POST` | `/api/chat/upload` |
| `GET` | `/api/chat/uploads/{filename}` |

### Daydream

| Method | Path |
| --- | --- |
| `GET` | `/api/daydream/summary` |
| `GET` | `/api/daydream/reflections` |
| `GET` | `/api/daydream/conflicts` |
| `GET` | `/api/daydream/promotions` |
| `GET` | `/api/daydream/sparks` |
| `GET` | `/api/daydream/initiatives` |
| `GET` | `/api/daydream/outcomes` |
| `GET` | `/api/daydream/notifications` |
| `GET` | `/api/daydream/lifecycle/{spark_id}` |

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

Current operations surface:

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
| `GET` | `/api/operations/sessions/process/{process_id}` |
| `DELETE` | `/api/operations/sessions/process/{process_id}` |
| `DELETE` | `/api/operations/sessions/process` |
| `GET` | `/api/operations/sessions/pty/{session_id}` |
| `POST` | `/api/operations/sessions/pty/{session_id}/input` |
| `DELETE` | `/api/operations/sessions/pty/{session_id}` |
| `DELETE` | `/api/operations/sessions/pty` |
| `GET` | `/api/operations/receipts` |
| `GET` | `/api/operations/receipts/{receipt_id}` |
| `GET` | `/api/operations/tasks` |
| `GET` | `/api/operations/tasks/{task_id}` |
| `GET` | `/api/operations/tasks/{task_id}/salvage` |
| `GET` | `/api/operations/work` |
| `GET` | `/api/operations/work/{work_id}` |
| `PATCH` | `/api/operations/work/{work_id}` |
| `GET` | `/api/operations/commitments` |
| `GET` | `/api/operations/commitments/{commitment_id}` |
| `PATCH` | `/api/operations/commitments/{commitment_id}` |
| `GET` | `/api/operations/plans` |
| `GET` | `/api/operations/plans/{plan_id}` |
| `PATCH` | `/api/operations/plans/{plan_id}` |

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
| `PATCH` | `/api/identity/somatic` |
| `GET` | `/api/identity/tom` |

### Executive

| Method | Path |
| --- | --- |
| `GET` | `/api/executive` |
| `GET` | `/api/executive/snapshot` |
| `POST` | `/api/executive/park-goal` |
| `GET` | `/api/executive/commitments` |
| `GET` | `/api/executive/plans` |
| `GET` | `/api/executive/events/summary` |
| `GET` | `/api/executive/events/search` |

### Platform

| Method | Path |
| --- | --- |
| `GET` | `/api/platform/capabilities` |
| `GET` | `/api/platform/capabilities/{capability_id}` |
| `GET` | `/api/platform/extensions` |
| `GET` | `/api/platform/extensions/{extension_id}` |
| `POST` | `/api/platform/extensions/install` |
| `POST` | `/api/platform/extensions/{extension_id}/update` |
| `POST` | `/api/platform/extensions/{extension_id}/enable` |
| `POST` | `/api/platform/extensions/{extension_id}/disable` |
| `DELETE` | `/api/platform/extensions/{extension_id}` |
| `POST` | `/api/platform/extensions/inspect-bundle` |
| `GET` | `/api/platform/policies/install-update` |

### Phone

| Method | Path |
| --- | --- |
| `GET` | `/api/phone/status` |
| `GET` | `/api/phone/recent-calls` |
| `GET` | `/api/phone/recent-calls/{call_sid}` |
| `POST` | `/api/phone/config` |
| `POST` | `/api/phone/autoconfigure` |
| `POST` | `/api/phone/session-profiles` |
| `POST` | `/api/phone/menu-config` |
| `POST` | `/api/phone/call-owner` |
| `POST` | `/api/phone/twilio/voice` |
| `POST` | `/api/phone/twilio/gather` |
| `POST` | `/api/phone/twilio/poll` |

### Schedule

| Method | Path |
| --- | --- |
| `GET` | `/api/schedule/items` |
| `POST` | `/api/schedule/items` |
| `GET` | `/api/schedule/items/{schedule_id}` |
| `PATCH` | `/api/schedule/items/{schedule_id}` |
| `DELETE` | `/api/schedule/items/{schedule_id}` |
| `GET` | `/api/schedule/calendar` |
| `GET` | `/api/schedule/agenda` |
| `GET` | `/api/schedule/runs` |
| `POST` | `/api/schedule/items/{schedule_id}/trigger` |

### Telemetry

| Method | Path |
| --- | --- |
| `GET` | `/api/telemetry/events` |
| `GET` | `/api/telemetry/kinds` |
| `GET` | `/api/telemetry/sessions` |
| `GET` | `/api/telemetry/stats` |
| `GET` | `/api/telemetry/stream` |

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
