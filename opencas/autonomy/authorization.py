"""Scoped standing authorizations for low-risk tool use.

This module records explicit user authorizations such as "check my email" as
durable, expiring scopes. The approval ladder can then recognize the matching
read-only tool request without treating every future Gmail action as globally
safe.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import ActionRequest, ActionRiskTier


DEFAULT_AUTHORIZATION_TTL = timedelta(hours=24)
GMAIL_READ_ACTION_CLASS = "gmail_read"
GMAIL_SCOPE = "google_workspace:gmail"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS authorizations (
    authorization_id TEXT PRIMARY KEY,
    action_class TEXT NOT NULL,
    scope TEXT NOT NULL,
    granted_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    evidence_episode_id TEXT,
    granted_by TEXT NOT NULL,
    session_id TEXT,
    meta TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_authorizations_scope
ON authorizations(action_class, scope, expires_at);

CREATE INDEX IF NOT EXISTS idx_authorizations_granted_at
ON authorizations(granted_at);
"""

_EMAIL_TARGET_RE = re.compile(r"\b(email|e-mail|gmail|inbox|mailbox)\b", re.IGNORECASE)
_EMAIL_ACTION_RE = re.compile(
    r"\b(check|read|look\s+at|show|scan|search|find|inspect|review)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Authorization:
    """A durable, expiring grant for one action class and scope."""

    authorization_id: str
    action_class: str
    scope: str
    granted_at: datetime
    expires_at: datetime
    evidence_episode_id: str | None = None
    granted_by: str = "user"
    session_id: str | None = None
    meta: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "authorization_id": self.authorization_id,
            "action_class": self.action_class,
            "scope": self.scope,
            "granted_at": self.granted_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "evidence_episode_id": self.evidence_episode_id,
            "granted_by": self.granted_by,
            "session_id": self.session_id,
            "meta": dict(self.meta or {}),
        }


class AuthorizationStore:
    """SQLite-backed scoped authorization store.

    The store opens short-lived SQLite connections so runtime shutdown does not
    need another close hook.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._ensure_schema()

    def grant(
        self,
        *,
        action_class: str,
        scope: str,
        evidence_episode_id: str | None = None,
        granted_by: str = "user",
        session_id: str | None = None,
        granted_at: datetime | None = None,
        expires_at: datetime | None = None,
        ttl: timedelta = DEFAULT_AUTHORIZATION_TTL,
        meta: dict[str, Any] | None = None,
    ) -> Authorization:
        granted_at = _aware_utc(granted_at)
        expires_at = _aware_utc(expires_at) if expires_at else granted_at + ttl
        authorization = Authorization(
            authorization_id=str(uuid4()),
            action_class=action_class,
            scope=scope,
            granted_at=granted_at,
            expires_at=expires_at,
            evidence_episode_id=evidence_episode_id,
            granted_by=granted_by,
            session_id=session_id,
            meta=dict(meta or {}),
        )
        self._ensure_schema()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO authorizations (
                    authorization_id, action_class, scope, granted_at, expires_at,
                    evidence_episode_id, granted_by, session_id, meta
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    authorization.authorization_id,
                    authorization.action_class,
                    authorization.scope,
                    authorization.granted_at.isoformat(),
                    authorization.expires_at.isoformat(),
                    authorization.evidence_episode_id,
                    authorization.granted_by,
                    authorization.session_id,
                    json.dumps(authorization.meta or {}, sort_keys=True),
                ),
            )
        return authorization

    def find_valid(
        self,
        action_class: str,
        scope: str,
        *,
        session_id: str | None = None,
        now: datetime | None = None,
    ) -> Authorization | None:
        now = _aware_utc(now)
        self._ensure_schema()
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM authorizations
                WHERE action_class = ? AND scope = ?
                ORDER BY granted_at DESC
                LIMIT 25
                """,
                (action_class, scope),
            ).fetchall()
        authorizations = [_authorization_from_row(row) for row in rows]
        if session_id:
            authorizations.sort(
                key=lambda authorization: (
                    authorization.session_id == session_id,
                    authorization.granted_at,
                ),
                reverse=True,
            )
        for authorization in authorizations:
            if authorization.expires_at >= now:
                return authorization
        return None

    def _ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.executescript(_SCHEMA)


def authorization_scope_for_request(
    request: ActionRequest,
) -> tuple[str, str] | None:
    """Return the standing authorization scope that can cover *request*."""
    if request.tier is not ActionRiskTier.READONLY:
        return None
    tool_name = (request.tool_name or "").strip().lower()
    if tool_name.startswith("google_workspace_gmail_"):
        return GMAIL_READ_ACTION_CLASS, GMAIL_SCOPE
    if tool_name in {"gmail_headlines", "gmail_get_message", "gmail_search"}:
        return GMAIL_READ_ACTION_CLASS, GMAIL_SCOPE
    return None


def authorization_scope_for_conversation(text: str) -> tuple[str, str] | None:
    """Return the standing authorization scope implied by conversational text."""
    if _is_email_read_authorization(text):
        return GMAIL_READ_ACTION_CLASS, GMAIL_SCOPE
    return None


def grant_authorizations_from_user_text(
    store: AuthorizationStore,
    text: str,
    *,
    session_id: str | None = None,
    evidence_episode_id: str | None = None,
    now: datetime | None = None,
    ttl: timedelta = DEFAULT_AUTHORIZATION_TTL,
) -> list[Authorization]:
    """Extract explicit low-risk authorizations from a user turn."""
    if not _is_email_read_authorization(text):
        return []
    grant = store.grant(
        action_class=GMAIL_READ_ACTION_CLASS,
        scope=GMAIL_SCOPE,
        evidence_episode_id=evidence_episode_id,
        granted_by="user",
        session_id=session_id,
        granted_at=now,
        ttl=ttl,
        meta={"source": "user_turn", "matched_intent": "email_read"},
    )
    return [grant]


def _is_email_read_authorization(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return False
    return bool(_EMAIL_TARGET_RE.search(normalized) and _EMAIL_ACTION_RE.search(normalized))


def _authorization_from_row(row: sqlite3.Row) -> Authorization:
    try:
        meta = json.loads(row["meta"] or "{}")
    except json.JSONDecodeError:
        meta = {}
    return Authorization(
        authorization_id=str(row["authorization_id"]),
        action_class=str(row["action_class"]),
        scope=str(row["scope"]),
        granted_at=_parse_datetime(str(row["granted_at"])),
        expires_at=_parse_datetime(str(row["expires_at"])),
        evidence_episode_id=row["evidence_episode_id"],
        granted_by=str(row["granted_by"]),
        session_id=row["session_id"],
        meta=meta if isinstance(meta, dict) else {},
    )


def _aware_utc(value: datetime | None = None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_datetime(value: str) -> datetime:
    return _aware_utc(datetime.fromisoformat(value))
