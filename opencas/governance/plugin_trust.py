"""Trust policy and assessment helpers for packaged plugin distribution."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

import aiosqlite
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

SUPPORTED_PLUGIN_TRUST_FEED_VERSIONS = {1}
_SUPPORTED_PLUGIN_TRUST_FEED_SIGNATURE_ALGORITHMS = {"ed25519"}


class PluginTrustScope(str, Enum):
    """Explicit trust policy scopes for packaged extensions."""

    PUBLISHER = "publisher"
    CHECKSUM = "checksum"
    SIGNER = "signer"


class PluginTrustLevel(str, Enum):
    """Trust levels for packaged extension provenance."""

    UNKNOWN = "unknown"
    GRAY = "gray"
    TRUSTED = "trusted"
    USER_APPROVED = "user_approved"
    BLOCKED = "blocked"


@dataclass
class PluginTrustPolicy:
    """Explicit operator-managed trust policy."""

    scope: PluginTrustScope
    value: str
    level: PluginTrustLevel
    source: str = "user"
    note: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PluginTrustAssessment:
    """Operator-facing trust summary for a bundle or installed plugin."""

    level: PluginTrustLevel
    certainty: float
    blocked: bool = False
    publisher: Optional[str] = None
    checksum: Optional[str] = None
    signer_ids: List[str] = field(default_factory=list)
    verified_signer_ids: List[str] = field(default_factory=list)
    signature_count: int = 0
    verified_signature_count: int = 0
    matched_policies: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


@dataclass
class PluginTrustFeedSyncResult:
    """Summary of one external trust-feed synchronization run."""

    source_id: str
    source: str
    format_version: int
    verification: Dict[str, object] = field(default_factory=dict)
    imported: List[Dict[str, object]] = field(default_factory=list)
    removed: List[Dict[str, object]] = field(default_factory=list)
    skipped_conflicts: List[Dict[str, object]] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)


def normalize_plugin_publisher(value: str | None) -> str | None:
    """Normalize publisher names for stable policy matching."""

    if not value:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def normalize_plugin_checksum(value: str | None) -> str | None:
    """Normalize SHA-256 strings used for pinned bundle trust policies."""

    if not value:
        return None
    normalized = str(value).strip().lower()
    if len(normalized) != 64:
        return None
    if any(ch not in "0123456789abcdef" for ch in normalized):
        return None
    return normalized


def normalize_plugin_signer_id(value: str | None) -> str | None:
    """Normalize signer key identifiers for stable policy matching."""

    if not value:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def normalize_plugin_public_key(value: str | None) -> str | None:
    """Normalize public-key text into a stable raw-base64 Ed25519 form."""

    if not value:
        return None
    from opencas.plugins.package import PluginPackageError, _normalize_public_key_text

    try:
        return _normalize_public_key_text(value)
    except PluginPackageError:
        return None


def _compute_feed_payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_plugin_trust_feed_signature_payload(
    *,
    format_version: int,
    source_id: str,
    policies: List[Dict[str, object]],
) -> bytes:
    """Build the canonical signed payload for one trust feed document."""

    payload = {
        "format_version": int(format_version),
        "source_id": str(source_id),
        "policies": list(policies),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


class PluginTrustStore:
    """Async SQLite store for plugin bundle trust policies."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS plugin_trust_policies (
        scope TEXT NOT NULL,
        value TEXT NOT NULL,
        level TEXT NOT NULL,
        source TEXT NOT NULL,
        note TEXT,
        metadata TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (scope, value)
    );

    CREATE INDEX IF NOT EXISTS idx_plugin_trust_scope ON plugin_trust_policies(scope);
    """

    def __init__(self, path) -> None:
        from pathlib import Path

        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "PluginTrustStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(self._SCHEMA)
        columns_cursor = await self._db.execute("PRAGMA table_info(plugin_trust_policies)")
        columns = {row[1] for row in await columns_cursor.fetchall()}
        if "metadata" not in columns:
            await self._db.execute("ALTER TABLE plugin_trust_policies ADD COLUMN metadata TEXT")
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def list_policies(self) -> List[PluginTrustPolicy]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT scope, value, level, source, note, metadata, created_at, updated_at
            FROM plugin_trust_policies
            ORDER BY scope ASC, value ASC
            """
        )
        rows = await cursor.fetchall()
        return [
            PluginTrustPolicy(
                scope=PluginTrustScope(row["scope"]),
                value=row["value"],
                level=PluginTrustLevel(row["level"]),
                source=row["source"],
                note=row["note"] or "",
                metadata=json.loads(row["metadata"] or "{}"),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    async def upsert_policy(self, policy: PluginTrustPolicy) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO plugin_trust_policies (scope, value, level, source, note, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope, value) DO UPDATE SET
                level = excluded.level,
                source = excluded.source,
                note = excluded.note,
                metadata = excluded.metadata,
                updated_at = excluded.updated_at
            """,
            (
                policy.scope.value,
                policy.value,
                policy.level.value,
                policy.source,
                policy.note,
                json.dumps(policy.metadata, sort_keys=True),
                policy.created_at.isoformat(),
                policy.updated_at.isoformat(),
            ),
        )
        await self._db.commit()

    async def delete_policy(self, scope: PluginTrustScope, value: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "DELETE FROM plugin_trust_policies WHERE scope = ? AND value = ?",
            (scope.value, value),
        )
        await self._db.commit()


class PluginTrustService:
    """Persisted trust model for packaged extension provenance and checksums."""

    def __init__(self, store: PluginTrustStore) -> None:
        self.store = store
        self._policies: Dict[tuple[PluginTrustScope, str], PluginTrustPolicy] = {}

    async def connect(self) -> "PluginTrustService":
        await self.store.connect()
        self._policies = {
            (policy.scope, policy.value): policy
            for policy in await self.store.list_policies()
        }
        return self

    async def close(self) -> None:
        await self.store.close()

    async def set_policy(
        self,
        scope: PluginTrustScope | str,
        value: str,
        level: PluginTrustLevel | str,
        *,
        source: str = "user",
        note: str = "",
        metadata: Optional[Dict[str, object]] = None,
    ) -> PluginTrustPolicy:
        normalized_scope = PluginTrustScope(scope)
        normalized_value = self._normalize_scope_value(normalized_scope, value)
        normalized_metadata = self._normalize_scope_metadata(normalized_scope, metadata or {})
        policy = PluginTrustPolicy(
            scope=normalized_scope,
            value=normalized_value,
            level=PluginTrustLevel(level),
            source=str(source or "user"),
            note=str(note or ""),
            metadata=normalized_metadata,
            created_at=self._policies.get(
                (normalized_scope, normalized_value),
                PluginTrustPolicy(
                    scope=normalized_scope,
                    value=normalized_value,
                    level=PluginTrustLevel.GRAY,
                ),
            ).created_at,
            updated_at=datetime.now(timezone.utc),
        )
        await self.store.upsert_policy(policy)
        self._policies[(normalized_scope, normalized_value)] = policy
        return policy

    async def remove_policy(self, scope: PluginTrustScope | str, value: str) -> None:
        normalized_scope = PluginTrustScope(scope)
        normalized_value = self._normalize_scope_value(normalized_scope, value)
        await self.store.delete_policy(normalized_scope, normalized_value)
        self._policies.pop((normalized_scope, normalized_value), None)

    def _normalize_feed_policies(
        self,
        policies_raw: object,
        *,
        rejected: List[str],
    ) -> Dict[tuple[PluginTrustScope, str], PluginTrustPolicy]:
        if not isinstance(policies_raw, list):
            raise ValueError("plugin trust feed policies must be a list")

        desired: Dict[tuple[PluginTrustScope, str], PluginTrustPolicy] = {}
        for index, raw_policy in enumerate(policies_raw):
            entry_path = f"policies[{index}]"
            if not isinstance(raw_policy, dict):
                rejected.append(f"{entry_path} must be an object")
                continue
            try:
                scope = PluginTrustScope(raw_policy.get("scope"))
            except Exception:
                rejected.append(f"{entry_path}.scope must be one of {[item.value for item in PluginTrustScope]}")
                continue
            if scope is PluginTrustScope.CHECKSUM:
                rejected.append(f"{entry_path}.scope=checksum is not supported in synchronized trust feeds")
                continue
            value = raw_policy.get("value")
            level = raw_policy.get("level")
            note = raw_policy.get("note") or ""
            metadata = raw_policy.get("metadata") if isinstance(raw_policy.get("metadata"), dict) else {}
            if not isinstance(note, str):
                rejected.append(f"{entry_path}.note must be a string")
                continue
            try:
                normalized_value = self._normalize_scope_value(scope, str(value))
                normalized_level = PluginTrustLevel(level)
                normalized_metadata = self._normalize_scope_metadata(scope, dict(metadata))
            except (ValueError, TypeError) as exc:
                rejected.append(f"{entry_path}: {exc}")
                continue
            desired[(scope, normalized_value)] = PluginTrustPolicy(
                scope=scope,
                value=normalized_value,
                level=normalized_level,
                source="",
                note=note,
                metadata=normalized_metadata,
            )
        return desired

    def _verify_feed_signatures(
        self,
        *,
        source_id: str,
        format_version: int,
        desired: Dict[tuple[PluginTrustScope, str], PluginTrustPolicy],
        signatures_raw: object,
    ) -> Dict[str, object]:
        if not isinstance(signatures_raw, list) or not signatures_raw:
            raise ValueError("plugin trust feed must include at least one signature entry")

        normalized_policies = [
            {
                "scope": policy.scope.value,
                "value": policy.value,
                "level": policy.level.value,
                "note": policy.note,
                "metadata": dict(policy.metadata),
            }
            for _, policy in sorted(desired.items(), key=lambda item: (item[0][0].value, item[0][1]))
        ]
        payload = build_plugin_trust_feed_signature_payload(
            format_version=format_version,
            source_id=source_id,
            policies=normalized_policies,
        )
        payload_sha256 = _compute_feed_payload_sha256(payload)
        verification = {
            "signature_count": len(signatures_raw),
            "verified_signature_count": 0,
            "verified_signer_ids": [],
            "trusted_signer_ids": [],
            "matched_policies": [],
            "payload_sha256": payload_sha256,
            "reasons": [],
        }

        for index, raw_entry in enumerate(signatures_raw):
            entry_path = f"signatures[{index}]"
            if not isinstance(raw_entry, dict):
                verification["reasons"].append(f"{entry_path} must be an object")
                continue

            key_id = normalize_plugin_signer_id(raw_entry.get("key_id"))
            algorithm = str(raw_entry.get("algorithm") or "").strip().lower()
            public_key_value = raw_entry.get("public_key")
            signature_value = raw_entry.get("signature")
            if key_id is None:
                verification["reasons"].append(f"{entry_path}.key_id must be a non-empty string")
                continue
            if algorithm not in _SUPPORTED_PLUGIN_TRUST_FEED_SIGNATURE_ALGORITHMS:
                verification["reasons"].append(f"{entry_path}.algorithm must be one of {sorted(_SUPPORTED_PLUGIN_TRUST_FEED_SIGNATURE_ALGORITHMS)}")
                continue
            if not isinstance(public_key_value, str) or not public_key_value.strip():
                verification["reasons"].append(f"{entry_path}.public_key must be a non-empty string")
                continue
            if not isinstance(signature_value, str) or not signature_value.strip():
                verification["reasons"].append(f"{entry_path}.signature must be a non-empty string")
                continue

            from opencas.plugins.package import PluginPackageError, _load_ed25519_public_key

            try:
                public_key = _load_ed25519_public_key(public_key_value)
                signature_bytes = base64.b64decode(signature_value.encode("ascii"), validate=True)
                public_key.verify(signature_bytes, payload)
            except (InvalidSignature, PluginPackageError, ValueError, TypeError, binascii.Error) as exc:
                message = "signature verification failed" if isinstance(exc, InvalidSignature) else str(exc)
                verification["reasons"].append(f"{entry_path}: {message}")
                continue

            verification["verified_signature_count"] += 1
            verification["verified_signer_ids"].append(key_id)

            signer_policy = self._policies.get((PluginTrustScope.SIGNER, key_id))
            if signer_policy is None:
                verification["reasons"].append(f"{entry_path}: signer {key_id} is not trusted yet")
                continue
            if signer_policy.level not in {PluginTrustLevel.TRUSTED, PluginTrustLevel.USER_APPROVED}:
                verification["reasons"].append(
                    f"{entry_path}: signer {key_id} policy level {signer_policy.level.value} is not sufficient for trust-feed sync"
                )
                continue
            policy_public_key = normalize_plugin_public_key(
                signer_policy.metadata.get("public_key")
                if isinstance(signer_policy.metadata.get("public_key"), str)
                else None
            )
            entry_public_key = normalize_plugin_public_key(public_key_value)
            if policy_public_key is not None and entry_public_key != policy_public_key:
                verification["reasons"].append(f"{entry_path}: signer policy public key mismatch for {key_id}")
                continue
            bound_source_id = signer_policy.metadata.get("source_id")
            if isinstance(bound_source_id, str) and bound_source_id.strip() and bound_source_id.strip() != source_id:
                verification["reasons"].append(
                    f"{entry_path}: signer policy source binding mismatch for {key_id}; expected {bound_source_id.strip()}"
                )
                continue

            verification["trusted_signer_ids"].append(key_id)
            verification["matched_policies"].append(f"signer:{key_id}")

        verification["verified_signer_ids"] = sorted(set(verification["verified_signer_ids"]))
        verification["trusted_signer_ids"] = sorted(set(verification["trusted_signer_ids"]))
        verification["matched_policies"] = sorted(set(verification["matched_policies"]))

        if verification["verified_signature_count"] <= 0:
            raise ValueError("plugin trust feed declares signatures, but none verified successfully")
        if not verification["trusted_signer_ids"]:
            raise ValueError("plugin trust feed is not anchored to a trusted signer policy")
        return verification

    async def sync_feed(self, feed: Dict[str, object]) -> PluginTrustFeedSyncResult:
        """Reconcile feed-owned trust policies from one explicit external source."""

        format_version = feed.get("format_version")
        if not isinstance(format_version, int) or isinstance(format_version, bool):
            raise ValueError("plugin trust feed format_version must be an integer")
        if format_version not in SUPPORTED_PLUGIN_TRUST_FEED_VERSIONS:
            raise ValueError(
                f"plugin trust feed format_version {format_version} is not supported; expected one of {sorted(SUPPORTED_PLUGIN_TRUST_FEED_VERSIONS)}"
            )
        source_id_raw = feed.get("source_id")
        if not isinstance(source_id_raw, str) or not source_id_raw.strip():
            raise ValueError("plugin trust feed source_id must be a non-empty string")

        source_id = source_id_raw.strip()
        source = f"feed:{source_id}"
        result = PluginTrustFeedSyncResult(
            source_id=source_id,
            source=source,
            format_version=format_version,
        )
        desired = self._normalize_feed_policies(feed.get("policies"), rejected=result.rejected)
        for policy in desired.values():
            policy.source = source
        result.verification = self._verify_feed_signatures(
            source_id=source_id,
            format_version=format_version,
            desired=desired,
            signatures_raw=feed.get("signatures"),
        )

        desired_keys = set(desired.keys())
        existing_feed_keys = {
            key for key, policy in self._policies.items() if policy.source == source
        }
        stale_keys = sorted(existing_feed_keys - desired_keys, key=lambda item: (item[0].value, item[1]))

        for key, desired_policy in sorted(desired.items(), key=lambda item: (item[0][0].value, item[0][1])):
            existing = self._policies.get(key)
            if existing is not None and existing.source != source:
                result.skipped_conflicts.append(
                    {
                        "scope": desired_policy.scope.value,
                        "value": desired_policy.value,
                        "existing_source": existing.source,
                        "incoming_source": source,
                        "reason": "existing policy from a different source must be removed or edited manually before feed sync can replace it",
                    }
                )
                continue

            created_at = existing.created_at if existing is not None else datetime.now(timezone.utc)
            desired_policy.created_at = created_at
            desired_policy.updated_at = datetime.now(timezone.utc)
            await self.store.upsert_policy(desired_policy)
            self._policies[key] = desired_policy
            result.imported.append(
                {
                    "scope": desired_policy.scope.value,
                    "value": desired_policy.value,
                    "level": desired_policy.level.value,
                    "metadata": dict(desired_policy.metadata),
                }
            )

        for scope, value in stale_keys:
            await self.store.delete_policy(scope, value)
            self._policies.pop((scope, value), None)
            result.removed.append({"scope": scope.value, "value": value})

        return result

    def assess(
        self,
        *,
        provenance: Optional[dict],
        bundle: Optional[dict],
    ) -> PluginTrustAssessment:
        provenance = provenance if isinstance(provenance, dict) else {}
        bundle = bundle if isinstance(bundle, dict) else {}
        publisher = normalize_plugin_publisher(provenance.get("publisher"))
        checksum = normalize_plugin_checksum(bundle.get("sha256"))
        signature_state = bundle.get("signatures") if isinstance(bundle.get("signatures"), dict) else {}
        signature_entries = signature_state.get("entries") if isinstance(signature_state.get("entries"), list) else []
        signer_ids = [
            signer_id
            for signer_id in (
                normalize_plugin_signer_id(entry.get("key_id"))
                for entry in signature_entries
                if isinstance(entry, dict)
            )
            if signer_id is not None
        ]
        verified_entries = [
            entry
            for entry in signature_entries
            if isinstance(entry, dict) and entry.get("verified") is True
        ]
        verified_signer_ids = [
            signer_id
            for signer_id in (
                normalize_plugin_signer_id(entry.get("key_id"))
                for entry in verified_entries
            )
            if signer_id is not None
        ]
        reasons: List[str] = []
        matched_policies: List[str] = []

        checksum_policy = (
            self._policies.get((PluginTrustScope.CHECKSUM, checksum))
            if checksum is not None
            else None
        )
        publisher_policy = (
            self._policies.get((PluginTrustScope.PUBLISHER, publisher))
            if publisher is not None
            else None
        )
        signer_policies = {
            signer_id: self._policies.get((PluginTrustScope.SIGNER, signer_id))
            for signer_id in verified_signer_ids
        }

        if checksum_policy is not None:
            matched_policies.append(f"checksum:{checksum_policy.value}")
            reasons.append(f"checksum policy={checksum_policy.level.value}")
            return PluginTrustAssessment(
                level=checksum_policy.level,
                certainty=1.0,
                blocked=checksum_policy.level is PluginTrustLevel.BLOCKED,
                publisher=publisher,
                checksum=checksum,
                signer_ids=signer_ids,
                verified_signer_ids=verified_signer_ids,
                signature_count=len(signature_entries),
                verified_signature_count=len(verified_entries),
                matched_policies=matched_policies,
                reasons=reasons,
            )

        for signer_id, signer_policy in signer_policies.items():
            if signer_policy is None:
                continue
            matched_policies.append(f"signer:{signer_id}")
            policy_publisher = signer_policy.metadata.get("publisher")
            bound_publisher = normalize_plugin_publisher(
                policy_publisher if isinstance(policy_publisher, str) else None
            )
            if bound_publisher is not None and bound_publisher != publisher:
                reasons.append(
                    f"signer policy publisher binding mismatch: expected {bound_publisher}, observed {publisher or '<missing>'}"
                )
                return PluginTrustAssessment(
                    level=PluginTrustLevel.BLOCKED,
                    certainty=1.0,
                    blocked=True,
                    publisher=publisher,
                    checksum=checksum,
                    signer_ids=signer_ids,
                    verified_signer_ids=verified_signer_ids,
                    signature_count=len(signature_entries),
                    verified_signature_count=len(verified_entries),
                    matched_policies=matched_policies,
                    reasons=reasons,
                )
            signer_public_key = signer_policy.metadata.get("public_key")
            policy_public_key = normalize_plugin_public_key(
                signer_public_key if isinstance(signer_public_key, str) else None
            )
            entry_public_key = normalize_plugin_public_key(
                next(
                    (
                        str(entry.get("public_key", ""))
                        for entry in verified_entries
                        if normalize_plugin_signer_id(entry.get("key_id")) == signer_id
                    ),
                    "",
                )
                or None
            )
            if policy_public_key is not None and entry_public_key is not None and policy_public_key != entry_public_key:
                reasons.append(f"signer policy public key mismatch for {signer_id}")
                return PluginTrustAssessment(
                    level=PluginTrustLevel.BLOCKED,
                    certainty=1.0,
                    blocked=True,
                    publisher=publisher,
                    checksum=checksum,
                    signer_ids=signer_ids,
                    verified_signer_ids=verified_signer_ids,
                    signature_count=len(signature_entries),
                    verified_signature_count=len(verified_entries),
                    matched_policies=matched_policies,
                    reasons=reasons,
                )
            reasons.append(f"signer policy={signer_policy.level.value}")
            return PluginTrustAssessment(
                level=signer_policy.level,
                certainty=0.98,
                blocked=signer_policy.level is PluginTrustLevel.BLOCKED,
                publisher=publisher,
                checksum=checksum,
                signer_ids=signer_ids,
                verified_signer_ids=verified_signer_ids,
                signature_count=len(signature_entries),
                verified_signature_count=len(verified_entries),
                matched_policies=matched_policies,
                reasons=reasons,
            )

        if publisher_policy is not None:
            matched_policies.append(f"publisher:{publisher_policy.value}")
            reasons.append(f"publisher policy={publisher_policy.level.value}")
            return PluginTrustAssessment(
                level=publisher_policy.level,
                certainty=0.92,
                blocked=publisher_policy.level is PluginTrustLevel.BLOCKED,
                publisher=publisher,
                checksum=checksum,
                signer_ids=signer_ids,
                verified_signer_ids=verified_signer_ids,
                signature_count=len(signature_entries),
                verified_signature_count=len(verified_entries),
                matched_policies=matched_policies,
                reasons=reasons,
            )

        level = PluginTrustLevel.UNKNOWN
        certainty = 0.0
        if verified_entries:
            level = PluginTrustLevel.GRAY
            certainty = 0.7
            reasons.append("bundle signature verifies cryptographically, but no explicit signer trust policy is configured")
        elif signature_entries:
            level = PluginTrustLevel.BLOCKED
            certainty = 1.0
            reasons.append("bundle declares signatures, but none verified successfully")
        elif publisher is not None and checksum is not None:
            level = PluginTrustLevel.GRAY
            certainty = 0.4
            reasons.append("bundle has publisher metadata and checksum but no explicit trust policy")
        elif publisher is not None:
            level = PluginTrustLevel.GRAY
            certainty = 0.3
            reasons.append("bundle has publisher metadata but no explicit trust policy")
        elif checksum is not None:
            certainty = 0.2
            reasons.append("bundle has checksum metadata but no explicit trust policy")
        else:
            reasons.append("bundle has no explicit trust signals beyond basic metadata")

        return PluginTrustAssessment(
            level=level,
            certainty=certainty,
            blocked=level is PluginTrustLevel.BLOCKED,
            publisher=publisher,
            checksum=checksum,
            signer_ids=signer_ids,
            verified_signer_ids=verified_signer_ids,
            signature_count=len(signature_entries),
            verified_signature_count=len(verified_entries),
            matched_policies=matched_policies,
            reasons=reasons,
        )

    async def summary(self, limit: int = 20) -> Dict[str, object]:
        return self.snapshot(limit=limit)

    def snapshot(self, limit: int = 20) -> Dict[str, object]:
        entries = []
        for policy in sorted(
            self._policies.values(),
            key=lambda item: (item.scope.value, item.value),
        )[: max(1, limit)]:
            entries.append(
                {
                    "scope": policy.scope.value,
                    "value": policy.value,
                    "level": policy.level.value,
                    "source": policy.source,
                    "note": policy.note,
                    "metadata": dict(policy.metadata),
                    "updated_at": policy.updated_at.isoformat(),
                }
            )
        return {
            "available": True,
            "policy_count": len(self._policies),
            "publisher_policy_count": sum(
                1 for policy in self._policies.values() if policy.scope is PluginTrustScope.PUBLISHER
            ),
            "checksum_policy_count": sum(
                1 for policy in self._policies.values() if policy.scope is PluginTrustScope.CHECKSUM
            ),
            "signer_policy_count": sum(
                1 for policy in self._policies.values() if policy.scope is PluginTrustScope.SIGNER
            ),
            "feed_policy_count": sum(
                1 for policy in self._policies.values() if str(policy.source).startswith("feed:")
            ),
            "feed_source_count": len(
                {
                    str(policy.source)
                    for policy in self._policies.values()
                    if str(policy.source).startswith("feed:")
                }
            ),
            "entries": entries,
        }

    def _normalize_scope_value(self, scope: PluginTrustScope, value: str) -> str:
        if scope is PluginTrustScope.PUBLISHER:
            normalized = normalize_plugin_publisher(value)
            if normalized is None:
                raise ValueError("A non-empty publisher value is required")
            return normalized
        if scope is PluginTrustScope.SIGNER:
            normalized = normalize_plugin_signer_id(value)
            if normalized is None:
                raise ValueError("A non-empty signer key id is required")
            return normalized
        normalized = normalize_plugin_checksum(value)
        if normalized is None:
            raise ValueError("A valid 64-character SHA-256 checksum is required")
        return normalized

    def _normalize_scope_metadata(
        self,
        scope: PluginTrustScope,
        metadata: Dict[str, object],
    ) -> Dict[str, object]:
        normalized = dict(metadata or {})
        if scope is not PluginTrustScope.SIGNER:
            return normalized
        public_key_value = normalized.get("public_key")
        public_key = normalize_plugin_public_key(
            public_key_value if isinstance(public_key_value, str) else None
        )
        if public_key is None:
            raise ValueError("Signer trust policies require a valid Ed25519 public_key")
        normalized["public_key"] = public_key
        publisher_value = normalized.get("publisher")
        publisher = normalize_plugin_publisher(
            publisher_value if isinstance(publisher_value, str) else None
        )
        if publisher is not None:
            normalized["publisher"] = publisher
        else:
            normalized.pop("publisher", None)
        return normalized
