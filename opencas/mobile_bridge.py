"""SQLite-backed bridge for the Android Pocket Relay companion app."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
from typing import Any, Mapping
from urllib.parse import urlencode
from uuid import uuid4

from opencas.identity.agent_name import resolve_agent_name


REQUIRED_EVENT_FIELDS = (
    "event_id",
    "device_id",
    "kind",
    "created_at",
    "observed_at",
    "source",
    "permission_scope",
    "idempotency_key",
)


class MobileBridgeError(Exception):
    """Base class for mobile bridge errors."""


class MobileBridgeAuthError(MobileBridgeError):
    """Raised when a mobile request is not authenticated."""


class MobileBridgeValidationError(MobileBridgeError):
    """Raised when a mobile payload is not acceptable."""


@dataclass(frozen=True)
class MobileDeviceRecord:
    """One paired Pocket Relay device."""

    device_id: str
    device_label: str
    platform: str
    app_version: str
    created_at: str
    last_seen_at: str | None
    status: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_label": self.device_label,
            "platform": self.platform,
            "app_version": self.app_version,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
            "status": self.status,
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _new_secret() -> str:
    return secrets.token_urlsafe(32)


def build_pairing_payload(
    *,
    public_base_url: str,
    pairing_id: str,
    pairing_secret: str,
) -> str:
    """Build the deep-link payload consumed by Pocket Relay."""

    query = urlencode(
        {
            "server": public_base_url.rstrip("/"),
            "pairing_id": pairing_id,
            "pairing_secret": pairing_secret,
        }
    )
    return f"opencas-pocket-relay://pair?{query}"


class MobileBridgeStore:
    """Persist mobile pairings, devices, events, and acknowledgement cursors."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS mobile_pairings (
                    pairing_id TEXT PRIMARY KEY,
                    secret_hash TEXT NOT NULL,
                    operator_label TEXT NOT NULL DEFAULT '',
                    requested_device_label TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    completed_device_id TEXT
                );

                CREATE TABLE IF NOT EXISTS mobile_devices (
                    device_id TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL,
                    device_label TEXT NOT NULL DEFAULT '',
                    platform TEXT NOT NULL DEFAULT 'android',
                    app_version TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT,
                    status TEXT NOT NULL DEFAULT 'active'
                );

                CREATE TABLE IF NOT EXISTS mobile_events (
                    ack_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    permission_scope TEXT NOT NULL,
                    precision TEXT,
                    confidence REAL,
                    operator_visible INTEGER NOT NULL DEFAULT 1,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    idempotency_key TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    UNIQUE(device_id, idempotency_key)
                );

                CREATE INDEX IF NOT EXISTS idx_mobile_events_device_ack
                    ON mobile_events(device_id, ack_id);
                CREATE INDEX IF NOT EXISTS idx_mobile_events_kind
                    ON mobile_events(kind);
                """
            )

    def create_pairing(
        self,
        *,
        operator_label: str = "",
        device_label: str = "",
        ttl_seconds: int = 600,
    ) -> dict[str, Any]:
        pairing_id = str(uuid4())
        pairing_secret = _new_secret()
        now = _utc_now()
        expires_at = now + timedelta(seconds=max(60, int(ttl_seconds)))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO mobile_pairings (
                    pairing_id, secret_hash, operator_label, requested_device_label,
                    created_at, expires_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    pairing_id,
                    _hash_secret(pairing_secret),
                    str(operator_label or "").strip(),
                    str(device_label or "").strip(),
                    _iso(now),
                    _iso(expires_at),
                ),
            )
        return {
            "pairing_id": pairing_id,
            "pairing_secret": pairing_secret,
            "created_at": _iso(now),
            "expires_at": _iso(expires_at),
        }

    def complete_pairing(
        self,
        *,
        pairing_id: str,
        pairing_secret: str,
        device_label: str,
        platform: str = "android",
        app_version: str = "",
    ) -> tuple[MobileDeviceRecord, str]:
        now = _utc_now()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM mobile_pairings WHERE pairing_id = ?",
                (str(pairing_id or "").strip(),),
            ).fetchone()
            if row is None:
                raise MobileBridgeValidationError("pairing_id was not found")
            if row["status"] != "pending":
                raise MobileBridgeValidationError("pairing_id is no longer pending")
            if row["secret_hash"] != _hash_secret(str(pairing_secret or "").strip()):
                raise MobileBridgeAuthError("pairing_secret is invalid")
            expires_at = datetime.fromisoformat(row["expires_at"])
            if expires_at <= now:
                conn.execute(
                    "UPDATE mobile_pairings SET status = 'expired' WHERE pairing_id = ?",
                    (row["pairing_id"],),
                )
                raise MobileBridgeValidationError("pairing_id has expired")

            token = _new_secret()
            device = MobileDeviceRecord(
                device_id=str(uuid4()),
                device_label=str(device_label or row["requested_device_label"] or "Android device").strip(),
                platform=str(platform or "android").strip() or "android",
                app_version=str(app_version or "").strip(),
                created_at=_iso(now),
                last_seen_at=_iso(now),
                status="active",
            )
            conn.execute(
                """
                INSERT INTO mobile_devices (
                    device_id, token_hash, device_label, platform, app_version,
                    created_at, last_seen_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    device.device_id,
                    _hash_secret(token),
                    device.device_label,
                    device.platform,
                    device.app_version,
                    device.created_at,
                    device.last_seen_at,
                    device.status,
                ),
            )
            conn.execute(
                """
                UPDATE mobile_pairings
                SET status = 'completed', completed_device_id = ?
                WHERE pairing_id = ?
                """,
                (device.device_id, row["pairing_id"]),
            )
        return device, token

    def authenticate(self, *, device_id: str, token: str) -> MobileDeviceRecord:
        normalized_device_id = str(device_id or "").strip()
        normalized_token = str(token or "").strip()
        if not normalized_device_id or not normalized_token:
            raise MobileBridgeAuthError("device_id and bearer token are required")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM mobile_devices WHERE device_id = ?",
                (normalized_device_id,),
            ).fetchone()
            if row is None or row["status"] != "active":
                raise MobileBridgeAuthError("device is not paired")
            if row["token_hash"] != _hash_secret(normalized_token):
                raise MobileBridgeAuthError("bearer token is invalid")
            now = _iso(_utc_now())
            conn.execute(
                "UPDATE mobile_devices SET last_seen_at = ? WHERE device_id = ?",
                (now, normalized_device_id),
            )
            return MobileDeviceRecord(
                device_id=row["device_id"],
                device_label=row["device_label"],
                platform=row["platform"],
                app_version=row["app_version"],
                created_at=row["created_at"],
                last_seen_at=now,
                status=row["status"],
            )

    def record_event(self, device: MobileDeviceRecord, envelope: Mapping[str, Any]) -> dict[str, Any]:
        missing = [
            key
            for key in REQUIRED_EVENT_FIELDS
            if key not in envelope or envelope.get(key) in (None, "")
        ]
        if missing:
            raise MobileBridgeValidationError(
                "mobile event is missing required field(s): " + ", ".join(missing)
            )
        if str(envelope["device_id"]) != device.device_id:
            raise MobileBridgeAuthError("event device_id does not match authenticated device")

        payload = envelope.get("payload") or {}
        if not isinstance(payload, Mapping):
            raise MobileBridgeValidationError("payload must be an object")
        confidence = envelope.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError) as exc:
                raise MobileBridgeValidationError("confidence must be numeric") from exc
            if confidence < 0.0 or confidence > 1.0:
                raise MobileBridgeValidationError("confidence must be between 0.0 and 1.0")

        received_at = _iso(_utc_now())
        values = (
            str(envelope["event_id"]),
            device.device_id,
            str(envelope["kind"]),
            str(envelope["created_at"]),
            str(envelope["observed_at"]),
            str(envelope["source"]),
            str(envelope["permission_scope"]),
            None if envelope.get("precision") is None else str(envelope.get("precision")),
            confidence,
            1 if bool(envelope.get("operator_visible", True)) else 0,
            json.dumps(dict(payload), sort_keys=True),
            str(envelope["idempotency_key"]),
            received_at,
        )
        with self._connect() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO mobile_events (
                        event_id, device_id, kind, created_at, observed_at, source,
                        permission_scope, precision, confidence, operator_visible,
                        payload_json, idempotency_key, received_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                ack_id = int(cursor.lastrowid)
                duplicate = False
            except sqlite3.IntegrityError:
                row = conn.execute(
                    """
                    SELECT ack_id FROM mobile_events
                    WHERE device_id = ? AND idempotency_key = ?
                    """,
                    (device.device_id, str(envelope["idempotency_key"])),
                ).fetchone()
                ack_id = int(row["ack_id"]) if row else 0
                duplicate = True
        return {
            "ack_id": ack_id,
            "event_id": str(envelope["event_id"]),
            "kind": str(envelope["kind"]),
            "idempotency_key": str(envelope["idempotency_key"]),
            "received_at": received_at,
            "duplicate": duplicate,
        }

    def list_acks(
        self,
        device: MobileDeviceRecord,
        *,
        after_ack_id: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ack_id, event_id, kind, idempotency_key, received_at
                FROM mobile_events
                WHERE device_id = ? AND ack_id > ?
                ORDER BY ack_id ASC
                LIMIT ?
                """,
                (device.device_id, max(0, int(after_ack_id)), max(1, min(int(limit), 500))),
            ).fetchall()
        return [
            {
                "ack_id": int(row["ack_id"]),
                "event_id": row["event_id"],
                "kind": row["kind"],
                "idempotency_key": row["idempotency_key"],
                "received_at": row["received_at"],
                "duplicate": False,
            }
            for row in rows
        ]

    def recent_events(self, device: MobileDeviceRecord, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ack_id, kind, permission_scope, precision, operator_visible,
                       payload_json, received_at
                FROM mobile_events
                WHERE device_id = ?
                ORDER BY ack_id DESC
                LIMIT ?
                """,
                (device.device_id, max(1, min(int(limit), 100))),
            ).fetchall()
        return [
            {
                "ack_id": int(row["ack_id"]),
                "kind": row["kind"],
                "permission_scope": row["permission_scope"],
                "precision": row["precision"],
                "operator_visible": bool(row["operator_visible"]),
                "payload": json.loads(row["payload_json"] or "{}"),
                "received_at": row["received_at"],
            }
            for row in rows
        ]


class MobileBridgeService:
    """Runtime-facing API used by Pocket Relay routes."""

    def __init__(self, runtime: Any, state_dir: Path | str) -> None:
        self.runtime = runtime
        self.store = MobileBridgeStore(Path(state_dir) / "mobile" / "mobile_bridge.db")

    def start_pairing(
        self,
        *,
        public_base_url: str,
        operator_label: str = "",
        device_label: str = "",
    ) -> dict[str, Any]:
        payload = self.store.create_pairing(
            operator_label=operator_label,
            device_label=device_label,
        )
        qr_payload = build_pairing_payload(
            public_base_url=public_base_url,
            pairing_id=payload["pairing_id"],
            pairing_secret=payload["pairing_secret"],
        )
        return {
            **payload,
            "qr_payload": qr_payload,
            "operator_label": str(operator_label or "").strip(),
            "device_label": str(device_label or "").strip(),
        }

    def complete_pairing(
        self,
        *,
        pairing_id: str,
        pairing_secret: str,
        device_label: str,
        platform: str = "android",
        app_version: str = "",
    ) -> dict[str, Any]:
        device, token = self.store.complete_pairing(
            pairing_id=pairing_id,
            pairing_secret=pairing_secret,
            device_label=device_label,
            platform=platform,
            app_version=app_version,
        )
        return {
            "device_id": device.device_id,
            "device_token": token,
            "token_type": "Bearer",
            "device": device.public_dict(),
        }

    def authenticate(self, *, device_id: str, token: str) -> MobileDeviceRecord:
        return self.store.authenticate(device_id=device_id, token=token)

    def ingest_events(
        self,
        device: MobileDeviceRecord,
        events: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        acks = [self.store.record_event(device, event) for event in events]
        duplicate_count = sum(1 for ack in acks if ack["duplicate"])
        accepted_count = len(acks) - duplicate_count
        self._trace(
            "mobile_events_ingested",
            {
                "device_id": device.device_id,
                "accepted": accepted_count,
                "duplicates": duplicate_count,
                "event_kinds": [ack["kind"] for ack in acks],
            },
        )
        return {
            "accepted": accepted_count,
            "duplicates": duplicate_count,
            "acks": acks,
        }

    def ack_snapshot(
        self,
        device: MobileDeviceRecord,
        *,
        after_ack_id: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        return {
            "device_id": device.device_id,
            "acks": self.store.list_acks(device, after_ack_id=after_ack_id, limit=limit),
        }

    def state_snapshot(self, device: MobileDeviceRecord) -> dict[str, Any]:
        ctx = getattr(self.runtime, "ctx", None)
        readiness = getattr(ctx, "readiness", None)
        readiness_payload = (
            readiness.snapshot()
            if readiness is not None and callable(getattr(readiness, "snapshot", None))
            else {"state": "unknown"}
        )
        llm = getattr(ctx, "llm", None)
        recent_events = self.store.recent_events(device, limit=12)
        shared_scopes = sorted({event["permission_scope"] for event in recent_events})
        return {
            "generated_at": _iso(_utc_now()),
            "presence": {
                "online": True,
                "agent_name": resolve_agent_name(runtime=self.runtime),
                "active_model": getattr(llm, "default_model", None),
                "current_activity": getattr(self.runtime, "_activity", None) or getattr(self.runtime, "activity", None),
            },
            "readiness": readiness_payload,
            "attention": {
                "summary": getattr(self.runtime, "_current_intention", None)
                or "No current mobile-safe attention summary is available.",
                "evidence": [],
            },
            "commitments": [],
            "approvals": [],
            "context_shared": {
                "device_id": device.device_id,
                "device_label": device.device_label,
                "shared_permission_scopes": shared_scopes,
                "recent_event_count": len(recent_events),
            },
            "recent_mobile_events": recent_events,
        }

    async def operator_message(
        self,
        device: MobileDeviceRecord,
        *,
        message: str,
        message_kind: str = "question",
        client_message_id: str = "",
    ) -> dict[str, Any]:
        cleaned = str(message or "").strip()
        if not cleaned:
            raise MobileBridgeValidationError("message is required")
        session_id = f"mobile:{device.device_id}"
        user_meta = {
            "actor_type": "mobile_companion",
            "actor_label": "OpenCAS Pocket Relay",
            "actor_note": "Operator message from paired Android Pocket Relay app.",
            "client_message_id": str(client_message_id or "").strip(),
            "message_kind": str(message_kind or "question").strip() or "question",
            "device_id": device.device_id,
        }
        converse = getattr(self.runtime, "converse", None)
        if not callable(converse):
            raise MobileBridgeValidationError("runtime conversation path is not available")
        response = await converse(cleaned, session_id=session_id, user_meta=user_meta)
        self._trace(
            "mobile_operator_message",
            {
                "device_id": device.device_id,
                "session_id": session_id,
                "message_kind": user_meta["message_kind"],
                "client_message_id": user_meta["client_message_id"],
            },
        )
        return {
            "session_id": session_id,
            "response": str(response),
        }

    async def approval_response(
        self,
        device: MobileDeviceRecord,
        *,
        approval_id: str,
        decision: str,
        reason: str = "",
    ) -> dict[str, Any]:
        payload = {
            "approval_id": str(approval_id or "").strip(),
            "decision": str(decision or "").strip(),
            "reason": str(reason or "").strip(),
            "device_id": device.device_id,
        }
        if not payload["approval_id"]:
            raise MobileBridgeValidationError("approval_id is required")
        if payload["decision"] not in {"approved", "rejected", "deferred"}:
            raise MobileBridgeValidationError("decision must be approved, rejected, or deferred")
        forwarded: dict[str, Any] | None = None
        handler = getattr(self.runtime, "handle_mobile_approval_response", None)
        if callable(handler):
            forwarded = await handler(payload)
        self._trace("mobile_approval_response", payload)
        return {
            "recorded": True,
            "forwarded": forwarded,
        }

    def _trace(self, event: str, payload: Mapping[str, Any]) -> None:
        trace = getattr(self.runtime, "_trace", None)
        if callable(trace):
            trace(event, dict(payload))


def mobile_bridge_service(runtime: Any) -> MobileBridgeService:
    """Return a cached mobile bridge service for *runtime*."""

    existing = getattr(runtime, "_mobile_bridge", None)
    if isinstance(existing, MobileBridgeService):
        return existing
    ctx = getattr(runtime, "ctx", None)
    config = getattr(ctx, "config", None)
    state_dir = getattr(config, "state_dir", Path(".opencas"))
    created = MobileBridgeService(runtime=runtime, state_dir=state_dir)
    setattr(runtime, "_mobile_bridge", created)
    return created
