"""Service layer for turning loose outputs into retrievable peripheral beads."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .models import (
    BeadEntry,
    BeadSourceKind,
    BeadStatus,
    BeadValidationResult,
    ThreadAnchor,
    ThreadStatus,
)
from .store import ThreadRegistryStore


def normalize_content_for_hash(content: str) -> str:
    """Normalize line endings without truncating the content being hashed."""
    return str(content or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def compute_content_hashes(content: str) -> tuple[str, str]:
    """Return full SHA-256 and a short display hash for *all* provided content."""
    full_hash = hashlib.sha256(normalize_content_for_hash(content).encode("utf-8")).hexdigest()
    return full_hash, full_hash[:12]


def make_anchor_id(title: str) -> str:
    """Build a stable human-readable id from a title without semantic matching."""
    normalized = " ".join(str(title or "").lower().split())
    chars: list[str] = []
    last_was_dash = False
    for char in normalized:
        if char.isalnum():
            chars.append(char)
            last_was_dash = False
        elif not last_was_dash:
            chars.append("-")
            last_was_dash = True
    candidate = "".join(chars).strip("-")
    return candidate or f"thread-{uuid4().hex[:12]}"


def make_bead_id(thread_anchor_id: str, title: str) -> str:
    """Build a readable unique bead id."""
    return f"{make_anchor_id(thread_anchor_id or title)}-{uuid4().hex[:12]}"


class ThreadRegistryService:
    """Coordinate thread anchors, bead validation, and artifact ingestion."""

    def __init__(self, *, store: ThreadRegistryStore, workspace_root: Path | str) -> None:
        self.store = store
        self.workspace_root = Path(workspace_root)

    async def ensure_thread_anchor(
        self,
        *,
        title: str,
        kind: str = "theme",
        status: ThreadStatus | str = ThreadStatus.PERIPHERAL,
        anchor_id: str | None = None,
    ) -> ThreadAnchor:
        resolved_anchor_id = anchor_id or make_anchor_id(title)
        existing = await self.store.get_thread_anchor(resolved_anchor_id)
        if existing is not None:
            return existing
        now = datetime.now(timezone.utc)
        anchor = ThreadAnchor(
            anchor_id=resolved_anchor_id,
            title=title,
            kind=kind,
            status=status,
            created_at=now,
            updated_at=now,
        )
        await self.store.save_thread_anchor(anchor)
        return anchor

    async def create_candidate_bead(
        self,
        *,
        thread_anchor_id: str,
        title: str,
        summary: str,
        source_kind: BeadSourceKind | str,
        source_ref: str,
        content: str,
        user_commissioned: bool = False,
    ) -> BeadEntry:
        content_hash_full, content_hash_short = compute_content_hashes(content)
        existing = await self.store.find_bead_by_source_hash(
            source_ref=source_ref,
            content_hash_full=content_hash_full,
        )
        if existing is not None:
            return existing

        anchor = await self.store.get_thread_anchor(thread_anchor_id)
        validation = self._validate_candidate(
            anchor=anchor,
            title=title,
            summary=summary,
            source_ref=source_ref,
            content=content,
        )
        if validation.dangling:
            status = BeadStatus.DANGLING
            committed_at = None
        elif validation.pickup_intelligible:
            status = BeadStatus.PERIPHERAL
            committed_at = datetime.now(timezone.utc)
        else:
            status = BeadStatus.CANDIDATE
            committed_at = None

        bead = BeadEntry(
            bead_id=make_bead_id(thread_anchor_id, title),
            thread_anchor_id=thread_anchor_id,
            title=title,
            summary=summary,
            source_kind=source_kind,
            source_ref=source_ref,
            content_hash_full=content_hash_full,
            content_hash_short=content_hash_short,
            status=status,
            user_commissioned=bool(user_commissioned),
            committed_at=committed_at,
            validation=validation,
        )
        await self.store.save_bead(bead)
        return bead

    async def ingest_autonomous_artifact(
        self,
        *,
        source_ref: str,
        content: str,
        thread_title: str,
        title: str,
        summary: str,
        thread_kind: str = "system_insight",
        user_commissioned: bool = False,
    ) -> BeadEntry:
        anchor = await self.ensure_thread_anchor(
            title=thread_title,
            kind=thread_kind,
            status=ThreadStatus.PERIPHERAL,
        )
        return await self.create_candidate_bead(
            thread_anchor_id=anchor.anchor_id,
            title=title,
            summary=summary,
            source_kind=BeadSourceKind.AUTONOMOUS_ARTIFACT,
            source_ref=source_ref,
            content=content,
            user_commissioned=user_commissioned,
        )

    async def list_beads(
        self,
        *,
        thread_anchor_id: str | None = None,
        status: BeadStatus | str | None = None,
        source_kind: BeadSourceKind | str | None = None,
        limit: int = 50,
    ) -> list[BeadEntry]:
        return await self.store.list_beads(
            thread_anchor_id=thread_anchor_id,
            status=status,
            source_kind=source_kind,
            limit=limit,
        )

    async def list_thread_anchors(
        self,
        *,
        status: ThreadStatus | str | None = None,
        limit: int = 50,
    ) -> list[ThreadAnchor]:
        return await self.store.list_thread_anchors(status=status, limit=limit)

    def _validate_candidate(
        self,
        *,
        anchor: ThreadAnchor | None,
        title: str,
        summary: str,
        source_ref: str,
        content: str,
    ) -> BeadValidationResult:
        reasons: list[str] = []
        title_ok = bool(str(title or "").strip())
        summary_ok = len(" ".join(str(summary or "").split())) >= 20
        source_ok = bool(str(source_ref or "").strip())
        content_ok = len(normalize_content_for_hash(content)) >= 20
        complete_fields = title_ok and summary_ok and source_ok and content_ok
        if not title_ok:
            reasons.append("title_missing")
        if not summary_ok:
            reasons.append("summary_too_short")
        if not source_ok:
            reasons.append("source_ref_missing")
        if not content_ok:
            reasons.append("content_too_short")

        anchor_resolved = anchor is not None
        if not anchor_resolved:
            reasons.append("anchor_unresolved")

        pickup_intelligible = complete_fields and anchor_resolved
        if pickup_intelligible:
            reasons.append("pickup_intelligible")
        return BeadValidationResult(
            complete_fields=complete_fields,
            anchor_resolved=anchor_resolved,
            pickup_intelligible=pickup_intelligible,
            dangling=not anchor_resolved,
            should_create_task=False,
            reasons=reasons,
            confidence=0.85 if pickup_intelligible else 0.35,
        )
