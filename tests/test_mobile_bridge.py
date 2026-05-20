"""Tests for the OpenCAS Pocket Relay mobile bridge."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opencas.api.routes.mobile import build_mobile_router


class _FakeReadiness:
    def snapshot(self) -> dict:
        return {"state": "ready", "ready_since": "2026-05-16T00:00:00+00:00"}


class _FakeRuntime:
    def __init__(self, tmp_path: Path) -> None:
        self.ctx = SimpleNamespace(
            config=SimpleNamespace(state_dir=tmp_path),
            readiness=_FakeReadiness(),
            identity=SimpleNamespace(self_model=SimpleNamespace(name="Bulma")),
            llm=SimpleNamespace(default_model="openai/gpt-5.5"),
        )
        self.mobile_messages: list[dict] = []
        self.mobile_approval_responses: list[dict] = []

    async def converse(self, text: str, session_id: str | None = None, user_meta=None):
        self.mobile_messages.append(
            {"text": text, "session_id": session_id, "user_meta": dict(user_meta or {})}
        )
        return f"mobile reply to {text}"

    async def handle_mobile_approval_response(self, payload: dict) -> dict:
        self.mobile_approval_responses.append(dict(payload))
        return {"handled": True, "approval_id": payload["approval_id"]}


def _client(tmp_path: Path) -> tuple[TestClient, _FakeRuntime]:
    runtime = _FakeRuntime(tmp_path)
    app = FastAPI()
    app.include_router(build_mobile_router(runtime))
    return TestClient(app), runtime


def _pair(client: TestClient) -> dict:
    started = client.post(
        "/api/mobile/pair/start",
        json={"operator_label": "Jarrod", "device_label": "Pixel test"},
    )
    assert started.status_code == 200
    payload = started.json()
    completed = client.post(
        "/api/mobile/pair/complete",
        json={
            "pairing_id": payload["pairing_id"],
            "pairing_secret": payload["pairing_secret"],
            "device_label": "Pixel test",
            "platform": "android",
            "app_version": "0.1-test",
        },
    )
    assert completed.status_code == 200
    return completed.json()


def _auth_headers(pairing: dict) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {pairing['device_token']}",
        "X-OpenCAS-Device-ID": pairing["device_id"],
    }


def _event(pairing: dict, *, idempotency_key: str = "stable-event-1") -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "event_id": "11111111-1111-4111-8111-111111111111",
        "device_id": pairing["device_id"],
        "kind": "device.presence",
        "created_at": now,
        "observed_at": now,
        "source": "android_companion",
        "permission_scope": "device_presence",
        "precision": "device",
        "confidence": 1.0,
        "operator_visible": True,
        "payload": {"reachable": True},
        "idempotency_key": idempotency_key,
    }


def test_pairing_flow_issues_scoped_device_credentials(tmp_path: Path) -> None:
    client, _runtime = _client(tmp_path)

    started = client.post(
        "/api/mobile/pair/start",
        json={"operator_label": "Jarrod", "device_label": "Pixel test"},
    )

    assert started.status_code == 200
    start_payload = started.json()
    assert start_payload["pairing_id"]
    assert start_payload["pairing_secret"]
    assert start_payload["qr_payload"].startswith("opencas-pocket-relay://pair?")
    assert start_payload["expires_at"]

    completed = client.post(
        "/api/mobile/pair/complete",
        json={
            "pairing_id": start_payload["pairing_id"],
            "pairing_secret": start_payload["pairing_secret"],
            "device_label": "Pixel test",
            "platform": "android",
            "app_version": "0.1-test",
        },
    )

    assert completed.status_code == 200
    completed_payload = completed.json()
    assert completed_payload["device_id"]
    assert completed_payload["device_token"]
    assert completed_payload["token_type"] == "Bearer"
    assert completed_payload["device"]["device_label"] == "Pixel test"
    assert "device_token" not in completed_payload["device"]


def test_pairing_page_creates_scannable_deep_link_qr(tmp_path: Path) -> None:
    client, _runtime = _client(tmp_path)

    response = client.get(
        "/api/mobile/pair/new?server=http%3A%2F%2F127.0.0.1%3A32147&device_label=Pixel"
    )

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "OpenCAS Pocket Relay Pairing" in response.text
    assert "opencas-pocket-relay://pair?" in response.text
    assert "server=http%3A%2F%2F127.0.0.1%3A32147" in response.text
    assert "<svg" in response.text or "QR generator unavailable" in response.text


def test_mobile_events_require_auth_and_are_idempotent(tmp_path: Path) -> None:
    client, _runtime = _client(tmp_path)
    pairing = _pair(client)
    event = _event(pairing)

    unauthenticated = client.post("/api/mobile/events", json={"events": [event]})
    assert unauthenticated.status_code == 401

    first = client.post(
        "/api/mobile/events",
        headers=_auth_headers(pairing),
        json={"events": [event]},
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["accepted"] == 1
    assert first_payload["duplicates"] == 0
    assert first_payload["acks"][0]["kind"] == "device.presence"

    duplicate = client.post(
        "/api/mobile/events",
        headers=_auth_headers(pairing),
        json={"events": [event]},
    )
    assert duplicate.status_code == 200
    duplicate_payload = duplicate.json()
    assert duplicate_payload["accepted"] == 0
    assert duplicate_payload["duplicates"] == 1
    assert duplicate_payload["acks"][0]["duplicate"] is True


def test_mobile_event_rejects_missing_required_provenance(tmp_path: Path) -> None:
    client, _runtime = _client(tmp_path)
    pairing = _pair(client)
    event = _event(pairing)
    event.pop("permission_scope")

    response = client.post(
        "/api/mobile/events",
        headers=_auth_headers(pairing),
        json={"events": [event]},
    )

    assert response.status_code == 422
    assert "permission_scope" in response.text


def test_mobile_state_exposes_operator_safe_snapshot(tmp_path: Path) -> None:
    client, _runtime = _client(tmp_path)
    pairing = _pair(client)

    response = client.get("/api/mobile/state", headers=_auth_headers(pairing))

    assert response.status_code == 200
    payload = response.json()
    assert payload["presence"]["online"] is True
    assert payload["presence"]["agent_name"] == "Bulma"
    assert payload["presence"]["active_model"] == "openai/gpt-5.5"
    assert payload["readiness"]["state"] == "ready"
    assert payload["context_shared"]["device_id"] == pairing["device_id"]
    assert "raw_logs" not in payload


def test_mobile_operator_message_uses_mobile_actor_metadata(tmp_path: Path) -> None:
    client, runtime = _client(tmp_path)
    pairing = _pair(client)

    response = client.post(
        "/api/mobile/operator-message",
        headers=_auth_headers(pairing),
        json={
            "message": "What are you working on?",
            "message_kind": "question",
            "client_message_id": "msg-1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response"] == "mobile reply to What are you working on?"
    assert runtime.mobile_messages == [
        {
            "text": "What are you working on?",
            "session_id": f"mobile:{pairing['device_id']}",
            "user_meta": {
                "actor_type": "mobile_companion",
                "actor_label": "OpenCAS Pocket Relay",
                "actor_note": "Operator message from paired Android Pocket Relay app.",
                "client_message_id": "msg-1",
                "message_kind": "question",
                "device_id": pairing["device_id"],
            },
        }
    ]


def test_mobile_approval_response_is_recorded_and_forwarded(tmp_path: Path) -> None:
    client, runtime = _client(tmp_path)
    pairing = _pair(client)

    response = client.post(
        "/api/mobile/approval-response",
        headers=_auth_headers(pairing),
        json={
            "approval_id": "approval-123",
            "decision": "approved",
            "reason": "Yes, send it.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["recorded"] is True
    assert payload["forwarded"] == {"handled": True, "approval_id": "approval-123"}
    assert runtime.mobile_approval_responses == [
        {
            "approval_id": "approval-123",
            "decision": "approved",
            "reason": "Yes, send it.",
            "device_id": pairing["device_id"],
        }
    ]
