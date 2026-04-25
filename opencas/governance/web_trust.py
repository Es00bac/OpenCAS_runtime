"""Web trust policy and learned domain-confidence service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from urllib.parse import urlparse

import aiosqlite


class WebActionClass(str, Enum):
    """Risk-relevant classes of web interaction."""

    SEARCH = "search"
    FETCH = "fetch"
    NAVIGATE = "navigate"
    OBSERVE = "observe"
    INTERACT = "interact"


class WebTrustLevel(str, Enum):
    """Domain trust levels used by the self-approval ladder."""

    UNKNOWN = "unknown"
    GRAY = "gray"
    TRUSTED = "trusted"
    USER_APPROVED = "user_approved"
    BLOCKED = "blocked"


@dataclass
class WebTrustPolicy:
    """Explicit domain policy record."""

    domain: str
    level: WebTrustLevel
    source: str = "user"
    note: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class WebDomainObservation:
    """Aggregated observed outcomes for a domain and action class."""

    domain: str
    action_class: WebActionClass
    success_count: int = 0
    failure_count: int = 0
    last_success_at: Optional[str] = None
    last_failure_at: Optional[str] = None


@dataclass
class WebTrustAssessment:
    """Approval-facing summary for a specific domain/action request."""

    domain: Optional[str]
    action_class: WebActionClass
    level: WebTrustLevel
    certainty: float
    risk_delta: float
    blocked: bool = False
    matched_policy_domain: Optional[str] = None
    reasons: List[str] = field(default_factory=list)


class WebTrustStore:
    """Async SQLite store for web trust policies and observations."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS web_domain_policies (
        domain TEXT PRIMARY KEY,
        level TEXT NOT NULL,
        source TEXT NOT NULL,
        note TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS web_domain_observations (
        domain TEXT NOT NULL,
        action_class TEXT NOT NULL,
        success_count INTEGER NOT NULL DEFAULT 0,
        failure_count INTEGER NOT NULL DEFAULT 0,
        last_success_at TEXT,
        last_failure_at TEXT,
        PRIMARY KEY (domain, action_class)
    );

    CREATE INDEX IF NOT EXISTS idx_web_observations_domain ON web_domain_observations(domain);
    """

    def __init__(self, path) -> None:
        from pathlib import Path

        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "WebTrustStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(self._SCHEMA)
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def list_policies(self) -> List[WebTrustPolicy]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT domain, level, source, note, created_at, updated_at
            FROM web_domain_policies
            ORDER BY domain ASC
            """
        )
        rows = await cursor.fetchall()
        return [
            WebTrustPolicy(
                domain=row["domain"],
                level=WebTrustLevel(row["level"]),
                source=row["source"],
                note=row["note"] or "",
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    async def list_observations(self) -> List[WebDomainObservation]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT domain, action_class, success_count, failure_count, last_success_at, last_failure_at
            FROM web_domain_observations
            ORDER BY domain ASC, action_class ASC
            """
        )
        rows = await cursor.fetchall()
        return [
            WebDomainObservation(
                domain=row["domain"],
                action_class=WebActionClass(row["action_class"]),
                success_count=int(row["success_count"] or 0),
                failure_count=int(row["failure_count"] or 0),
                last_success_at=row["last_success_at"],
                last_failure_at=row["last_failure_at"],
            )
            for row in rows
        ]

    async def upsert_policy(self, policy: WebTrustPolicy) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO web_domain_policies (domain, level, source, note, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(domain) DO UPDATE SET
                level = excluded.level,
                source = excluded.source,
                note = excluded.note,
                updated_at = excluded.updated_at
            """,
            (
                policy.domain,
                policy.level.value,
                policy.source,
                policy.note,
                policy.created_at.isoformat(),
                policy.updated_at.isoformat(),
            ),
        )
        await self._db.commit()

    async def delete_policy(self, domain: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "DELETE FROM web_domain_policies WHERE domain = ?",
            (domain,),
        )
        await self._db.commit()

    async def increment_observation(
        self,
        domain: str,
        action_class: WebActionClass,
        success: bool,
        *,
        timestamp: Optional[str] = None,
    ) -> None:
        assert self._db is not None
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        success_inc = 1 if success else 0
        failure_inc = 0 if success else 1
        await self._db.execute(
            """
            INSERT INTO web_domain_observations (
                domain, action_class, success_count, failure_count, last_success_at, last_failure_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(domain, action_class) DO UPDATE SET
                success_count = web_domain_observations.success_count + excluded.success_count,
                failure_count = web_domain_observations.failure_count + excluded.failure_count,
                last_success_at = CASE
                    WHEN excluded.success_count > 0 THEN excluded.last_success_at
                    ELSE web_domain_observations.last_success_at
                END,
                last_failure_at = CASE
                    WHEN excluded.failure_count > 0 THEN excluded.last_failure_at
                    ELSE web_domain_observations.last_failure_at
                END
            """,
            (
                domain,
                action_class.value,
                success_inc,
                failure_inc,
                ts if success else None,
                ts if not success else None,
            ),
        )
        await self._db.commit()


class WebTrustService:
    """Persisted domain trust model with explicit policy and learned evidence."""

    def __init__(self, store: WebTrustStore) -> None:
        self.store = store
        self._policies: Dict[str, WebTrustPolicy] = {}
        self._observations: Dict[tuple[str, WebActionClass], WebDomainObservation] = {}

    async def connect(self) -> "WebTrustService":
        await self.store.connect()
        self._policies = {policy.domain: policy for policy in await self.store.list_policies()}
        self._observations = {
            (row.domain, row.action_class): row
            for row in await self.store.list_observations()
        }
        return self

    async def close(self) -> None:
        await self.store.close()

    async def set_policy(
        self,
        domain_or_url: str,
        level: WebTrustLevel | str,
        *,
        source: str = "user",
        note: str = "",
    ) -> WebTrustPolicy:
        normalized = normalize_web_domain(domain_or_url)
        if not normalized:
            raise ValueError("A valid domain or URL is required")
        policy = WebTrustPolicy(
            domain=normalized,
            level=WebTrustLevel(level),
            source=str(source or "user"),
            note=str(note or ""),
            created_at=self._policies.get(normalized, WebTrustPolicy(normalized, WebTrustLevel.GRAY)).created_at,
            updated_at=datetime.now(timezone.utc),
        )
        await self.store.upsert_policy(policy)
        self._policies[normalized] = policy
        return policy

    async def remove_policy(self, domain_or_url: str) -> None:
        normalized = normalize_web_domain(domain_or_url)
        if not normalized:
            return
        await self.store.delete_policy(normalized)
        self._policies.pop(normalized, None)

    async def record_outcome(
        self,
        *,
        url: Optional[str],
        domain: Optional[str],
        action_class: WebActionClass | str,
        success: bool,
    ) -> None:
        normalized = normalize_web_domain(domain or url or "")
        if not normalized:
            return
        action = WebActionClass(action_class)
        await self.store.increment_observation(normalized, action, success)
        key = (normalized, action)
        current = self._observations.get(key)
        if current is None:
            current = WebDomainObservation(domain=normalized, action_class=action)
            self._observations[key] = current
        if success:
            current.success_count += 1
            current.last_success_at = datetime.now(timezone.utc).isoformat()
        else:
            current.failure_count += 1
            current.last_failure_at = datetime.now(timezone.utc).isoformat()

    def assess(
        self,
        *,
        url: Optional[str],
        domain: Optional[str],
        action_class: WebActionClass | str,
    ) -> Optional[WebTrustAssessment]:
        normalized = normalize_web_domain(domain or url or "")
        action = WebActionClass(action_class)
        if not normalized:
            if action == WebActionClass.SEARCH:
                return WebTrustAssessment(
                    domain=None,
                    action_class=action,
                    level=WebTrustLevel.GRAY,
                    certainty=0.5,
                    risk_delta=-0.04,
                    reasons=["web_search is treated as bounded read-only discovery"],
                )
            return None

        policy, matched_policy_domain = self._match_policy(normalized)
        certainty = self._certainty(normalized)
        learned_level = self._learned_level(normalized, action, certainty)
        level = policy.level if policy is not None else learned_level
        reasons = [f"web_domain={normalized}", f"web_action={action.value}"]
        if policy is not None:
            reasons.append(f"web_policy={policy.level.value}@{matched_policy_domain}")
        else:
            reasons.append(f"web_learned={learned_level.value}")
        reasons.append(f"web_certainty={certainty:.3f}")

        if level is WebTrustLevel.BLOCKED:
            return WebTrustAssessment(
                domain=normalized,
                action_class=action,
                level=level,
                certainty=certainty,
                risk_delta=1.0,
                blocked=True,
                matched_policy_domain=matched_policy_domain,
                reasons=reasons,
            )

        delta = self._risk_delta(level, action)
        return WebTrustAssessment(
            domain=normalized,
            action_class=action,
            level=level,
            certainty=certainty,
            risk_delta=delta,
            blocked=False,
            matched_policy_domain=matched_policy_domain,
            reasons=reasons,
        )

    async def summary(self, limit: int = 20) -> Dict[str, object]:
        return self.snapshot(limit=limit)

    def snapshot(self, limit: int = 20) -> Dict[str, object]:
        domains = sorted(
            {policy.domain for policy in self._policies.values()}
            | {domain for domain, _ in self._observations.keys()}
        )
        entries = []
        for domain in domains[: max(1, limit)]:
            policy, matched = self._match_policy(domain)
            certainty = self._certainty(domain)
            entries.append(
                {
                    "domain": domain,
                    "policy": policy.level.value if policy is not None else None,
                    "policy_source": policy.source if policy is not None else None,
                    "matched_policy_domain": matched,
                    "certainty": round(certainty, 3),
                    "learned_level": self._learned_level(domain, WebActionClass.OBSERVE, certainty).value,
                    "observations": self._domain_observation_summary(domain),
                }
            )
        return {
            "available": True,
            "policy_count": len(self._policies),
            "observed_domain_count": len({domain for domain, _ in self._observations.keys()}),
            "entries": entries,
        }

    def _match_policy(self, domain: str) -> tuple[Optional[WebTrustPolicy], Optional[str]]:
        for candidate in iter_domain_candidates(domain):
            policy = self._policies.get(candidate)
            if policy is not None:
                return policy, candidate
        return None, None

    def _certainty(self, domain: str) -> float:
        observe = self._observations.get((domain, WebActionClass.OBSERVE))
        navigate = self._observations.get((domain, WebActionClass.NAVIGATE))
        fetch = self._observations.get((domain, WebActionClass.FETCH))
        interact = self._observations.get((domain, WebActionClass.INTERACT))
        success = sum(
            row.success_count
            for row in (observe, navigate, fetch, interact)
            if row is not None
        )
        failure = sum(
            row.failure_count
            for row in (observe, navigate, fetch, interact)
            if row is not None
        )
        total = success + failure
        if total <= 0:
            return 0.0
        success_ratio = success / total
        sample_weight = min(total / 12.0, 1.0)
        return max(0.0, min(1.0, success_ratio * sample_weight))

    def _learned_level(
        self,
        domain: str,
        action: WebActionClass,
        certainty: Optional[float] = None,
    ) -> WebTrustLevel:
        certainty = self._certainty(domain) if certainty is None else certainty
        summary = self._domain_observation_summary(domain)
        success = summary["success_count"]
        failure = summary["failure_count"]
        total = success + failure
        success_ratio = success / total if total else 0.0

        if total <= 0:
            return WebTrustLevel.UNKNOWN
        if success >= 12 and success_ratio >= 0.97 and certainty >= 0.96:
            return WebTrustLevel.TRUSTED
        if action in {WebActionClass.FETCH, WebActionClass.NAVIGATE, WebActionClass.OBSERVE}:
            if success >= 2 and success_ratio >= 0.80:
                return WebTrustLevel.GRAY
        if action == WebActionClass.INTERACT and success >= 4 and success_ratio >= 0.90 and certainty >= 0.55:
            return WebTrustLevel.GRAY
        return WebTrustLevel.UNKNOWN

    def _domain_observation_summary(self, domain: str) -> Dict[str, int]:
        success = 0
        failure = 0
        for (seen_domain, _), row in self._observations.items():
            if seen_domain != domain:
                continue
            success += row.success_count
            failure += row.failure_count
        return {"success_count": success, "failure_count": failure}

    @staticmethod
    def _risk_delta(level: WebTrustLevel, action: WebActionClass) -> float:
        if action == WebActionClass.SEARCH:
            return -0.04

        if action in {WebActionClass.FETCH, WebActionClass.NAVIGATE, WebActionClass.OBSERVE}:
            if level == WebTrustLevel.USER_APPROVED:
                return -0.16
            if level == WebTrustLevel.TRUSTED:
                return -0.12
            if level == WebTrustLevel.GRAY:
                return -0.03
            if level == WebTrustLevel.UNKNOWN:
                return 0.04

        if action == WebActionClass.INTERACT:
            if level == WebTrustLevel.USER_APPROVED:
                return -0.16
            if level == WebTrustLevel.TRUSTED:
                return -0.12
            if level == WebTrustLevel.GRAY:
                return 0.08
            if level == WebTrustLevel.UNKNOWN:
                return 0.24

        return 0.0


def normalize_web_domain(domain_or_url: str) -> Optional[str]:
    """Normalize a domain or URL into a lowercase hostname."""
    raw = str(domain_or_url or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host or None


def iter_domain_candidates(domain_or_url: str) -> List[str]:
    """Return exact domain plus parent domains for policy lookup."""
    normalized = normalize_web_domain(domain_or_url)
    if not normalized:
        return []
    parts = normalized.split(".")
    candidates = [normalized]
    for idx in range(1, max(1, len(parts) - 1)):
        candidate = ".".join(parts[idx:])
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def classify_web_action(tool_name: str) -> Optional[WebActionClass]:
    """Map a tool name to a web action class."""
    name = str(tool_name or "").strip().lower()
    if name == "web_search":
        return WebActionClass.SEARCH
    if name == "web_fetch":
        return WebActionClass.FETCH
    if name == "browser_navigate":
        return WebActionClass.NAVIGATE
    if name in {"browser_snapshot", "browser_wait"}:
        return WebActionClass.OBSERVE
    if name in {"browser_click", "browser_type", "browser_press"}:
        return WebActionClass.INTERACT
    return None
