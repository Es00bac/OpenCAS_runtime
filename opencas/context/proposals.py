"""Durable reflective proposals for dual-context coordination."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

import aiosqlite
from pydantic import BaseModel, Field

from .hemispheres import ContextAuthority, ContextLane

_SCHEMA = """
CREATE TABLE IF NOT EXISTS context_proposals (
    proposal_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    source_lane TEXT NOT NULL,
    source_epoch INTEGER NOT NULL,
    source_snapshot_id TEXT NOT NULL,
    proposal_kind TEXT NOT NULL,
    status TEXT NOT NULL,
    authority TEXT NOT NULL,
    project_id TEXT,
    commitment_id TEXT,
    schedule_id TEXT,
    task_id TEXT,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_context_proposals_project
    ON context_proposals(project_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_context_proposals_task
    ON context_proposals(task_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_context_proposals_status
    ON context_proposals(status, created_at DESC);
"""


class ProposalStatus(StrEnum):
    """Lifecycle state for a context proposal."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STALE = "stale"
    SUPERSEDED = "superseded"


class ContextProposal(BaseModel):
    """A source-labeled idea that cannot act until accepted by executive truth."""

    proposal_id: str = Field(default_factory=lambda: f"proposal:{uuid4()}")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_lane: ContextLane
    source_snapshot_id: str
    source_epoch: int
    proposal_kind: str
    project_id: Optional[str] = None
    commitment_id: Optional[str] = None
    schedule_id: Optional[str] = None
    task_id: Optional[str] = None
    content: str
    evidence_refs: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    authority: ContextAuthority = ContextAuthority.PROPOSAL
    status: ProposalStatus = ProposalStatus.PENDING
    validation: Dict[str, Any] = Field(default_factory=dict)


class ContextProposalStore:
    """SQLite store for reflective proposals and their validation lifecycle."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "ContextProposalStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def save(self, proposal: ContextProposal) -> None:
        assert self._db is not None
        proposal.updated_at = datetime.now(timezone.utc)
        await self._db.execute(
            """
            INSERT INTO context_proposals (
                proposal_id, created_at, updated_at, source_lane, source_epoch,
                source_snapshot_id, proposal_kind, status, authority, project_id,
                commitment_id, schedule_id, task_id, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(proposal_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                source_lane = excluded.source_lane,
                source_epoch = excluded.source_epoch,
                source_snapshot_id = excluded.source_snapshot_id,
                proposal_kind = excluded.proposal_kind,
                status = excluded.status,
                authority = excluded.authority,
                project_id = excluded.project_id,
                commitment_id = excluded.commitment_id,
                schedule_id = excluded.schedule_id,
                task_id = excluded.task_id,
                payload = excluded.payload
            """,
            self._row_values(proposal),
        )
        await self._db.commit()

    async def get(self, proposal_id: str) -> Optional[ContextProposal]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT payload FROM context_proposals WHERE proposal_id = ?",
            (proposal_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return ContextProposal.model_validate_json(row["payload"])

    async def mark_accepted(
        self,
        proposal_id: str,
        *,
        accepted_by: str,
        arbiter_decision_id: str | None = None,
    ) -> Optional[ContextProposal]:
        proposal = await self.get(proposal_id)
        if proposal is None:
            return None
        proposal.status = ProposalStatus.ACCEPTED
        proposal.authority = ContextAuthority.EXECUTIVE_COMMITTED
        decision_id = str(arbiter_decision_id or "").strip() or f"accepted:{proposal_id}"
        proposal.validation = {
            **proposal.validation,
            "accepted_by": accepted_by,
            "arbiter_decision_id": decision_id,
            "proposal_validation_status": ProposalStatus.ACCEPTED.value,
        }
        await self.save(proposal)
        return proposal

    async def mark_rejected(self, proposal_id: str, *, reason: str) -> Optional[ContextProposal]:
        proposal = await self.get(proposal_id)
        if proposal is None:
            return None
        proposal.status = ProposalStatus.REJECTED
        proposal.validation = {**proposal.validation, "reason": reason}
        await self.save(proposal)
        return proposal

    async def mark_stale(self, proposal_id: str, *, current_epoch: int) -> Optional[ContextProposal]:
        proposal = await self.get(proposal_id)
        if proposal is None:
            return None
        proposal.status = ProposalStatus.STALE
        proposal.validation = {**proposal.validation, "current_epoch": current_epoch}
        await self.save(proposal)
        return proposal

    async def list_by_project(
        self,
        project_id: str,
        *,
        include_terminal: bool = False,
        limit: int = 20,
    ) -> List[ContextProposal]:
        where = "project_id = ?"
        args: list[Any] = [project_id]
        if not include_terminal:
            where += " AND status = ?"
            args.append(ProposalStatus.PENDING.value)
        return await self._list_where(where, args, limit=limit)

    async def list_by_task(
        self,
        task_id: str,
        *,
        include_terminal: bool = True,
        limit: int = 20,
    ) -> List[ContextProposal]:
        where = "task_id = ?"
        args: list[Any] = [task_id]
        if not include_terminal:
            where += " AND status = ?"
            args.append(ProposalStatus.PENDING.value)
        return await self._list_where(where, args, limit=limit)

    async def list_recent(
        self,
        *,
        status: Optional[ProposalStatus | str] = None,
        limit: int = 20,
    ) -> List[ContextProposal]:
        """Return recent proposals, optionally filtered by lifecycle status."""

        where = "1 = 1"
        args: list[Any] = []
        if status is not None:
            status_value = status.value if isinstance(status, ProposalStatus) else str(status)
            where = "status = ?"
            args.append(status_value)
        return await self._list_where(where, args, limit=limit, newest_first=True)

    async def count_by_status(self) -> Dict[str, int]:
        """Return proposal counts keyed by status."""

        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM context_proposals
            GROUP BY status
            """
        )
        rows = await cursor.fetchall()
        return {str(row["status"]): int(row["count"] or 0) for row in rows}

    async def search_text(
        self,
        query: str,
        *,
        include_terminal: bool = True,
        limit: int = 8,
    ) -> List[ContextProposal]:
        """Return proposals whose payload text overlaps a natural-language query."""

        terms = self._query_terms(query)
        if not terms:
            return []
        where = "(" + " OR ".join("LOWER(payload) LIKE ?" for _ in terms) + ")"
        args: list[Any] = [f"%{term}%" for term in terms]
        if not include_terminal:
            where += " AND status = ?"
            args.append(ProposalStatus.PENDING.value)
        candidates = await self._list_where(
            where,
            args,
            limit=max(limit * 8, 24),
            newest_first=True,
        )
        minimum_score = min(2, len(terms))
        scored: list[tuple[int, datetime, ContextProposal]] = []
        for proposal in candidates:
            text = self._proposal_search_text(proposal)
            score = sum(1 for term in terms if term in text)
            if score < minimum_score:
                continue
            scored.append((score, proposal.created_at, proposal))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [proposal for _score, _created_at, proposal in scored[:limit]]

    @staticmethod
    def _proposal_search_text(proposal: ContextProposal) -> str:
        return " ".join(
            str(value or "")
            for value in (
                proposal.proposal_id,
                proposal.proposal_kind,
                proposal.project_id,
                proposal.commitment_id,
                proposal.schedule_id,
                proposal.task_id,
                proposal.content,
                " ".join(proposal.evidence_refs),
            )
        ).lower()

    async def _list_where(
        self,
        where: str,
        args: list[Any],
        *,
        limit: int,
        newest_first: bool = False,
    ) -> List[ContextProposal]:
        assert self._db is not None
        direction = "DESC" if newest_first else "ASC"
        cursor = await self._db.execute(
            f"""
            SELECT payload FROM context_proposals
            WHERE {where}
            ORDER BY created_at {direction}, proposal_id ASC
            LIMIT ?
            """,
            [*args, limit],
        )
        rows = await cursor.fetchall()
        return [ContextProposal.model_validate_json(row["payload"]) for row in rows]

    @staticmethod
    def _query_terms(query: str) -> List[str]:
        terms: list[str] = []
        stopwords = {
            "about",
            "again",
            "that",
            "this",
            "then",
            "there",
            "what",
            "when",
            "where",
            "while",
            "with",
            "work",
            "context",
            "could",
            "memory",
            "project",
            "please",
            "return",
            "should",
            "would",
        }
        for raw in re.findall(r"[A-Za-z0-9_:-]{3,}", str(query or "").lower()):
            if raw in stopwords or raw in terms:
                continue
            terms.append(raw)
            if len(terms) >= 8:
                break
        return terms

    @staticmethod
    def _row_values(proposal: ContextProposal) -> tuple[Any, ...]:
        return (
            proposal.proposal_id,
            proposal.created_at.isoformat(),
            proposal.updated_at.isoformat(),
            proposal.source_lane.value,
            proposal.source_epoch,
            proposal.source_snapshot_id,
            proposal.proposal_kind,
            proposal.status.value,
            proposal.authority.value,
            proposal.project_id,
            proposal.commitment_id,
            proposal.schedule_id,
            proposal.task_id,
            proposal.model_dump_json(),
        )
