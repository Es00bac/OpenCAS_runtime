"""Persisted cognitive state and feedback-loop spine for OpenCAS.

The store in this module is intentionally small and general: it gives higher
level cognitive analogues one shared place to leave evidence, and gives prompt
assembly/tools one shared place to retrieve that evidence again. That is the
main guard against new write-only "inner life" records.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import aiosqlite
from pydantic import BaseModel, Field, field_validator

from opencas.memory import EpisodeKind


class CognitiveEventKind(str, Enum):
    """Kinds of cognitive state evidence OpenCAS can record and retrieve."""

    ATTENTION = "attention"
    WORKING_MEMORY = "working_memory"
    SURPRISE = "surprise"
    PROSPECTIVE_MEMORY = "prospective_memory"
    EMOTIONAL_MEMORY = "emotional_memory"
    CURIOSITY = "curiosity"
    PROCEDURAL_SKILL = "procedural_skill"
    COUNTERFACTUAL = "counterfactual"
    HABIT = "habit"
    LIFE_PRIORITY = "life_priority"
    NARRATIVE = "narrative"
    TELEMETRY_AFFECT = "telemetry_affect"
    AFFECTIVE_TREND = "affective_trend"
    SELF_INSPECTION_GAP = "self_inspection_gap"
    RECALL_FAILURE_MEMORY = "recall_failure_memory"
    MAINTENANCE_OUTCOME = "maintenance_outcome"
    SOMATIC_STRATEGY = "somatic_strategy"
    UNCERTAINTY_SEEKING = "uncertainty_seeking"
    SOCIAL_MODEL = "social_model"


class CognitiveEvent(BaseModel):
    """A compact, retrievable event in the cognitive feedback spine."""

    event_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    kind: CognitiveEventKind
    summary: str
    content: str = ""
    source: str = ""
    session_id: str | None = None
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    salience: float = Field(default=1.0, ge=0.0, le=10.0)
    status: str = "active"
    evidence_refs: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("summary", "content", "source", "status", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str:
        return " ".join(str(value or "").split())

    @field_validator("session_id", mode="before")
    @classmethod
    def _clean_optional_text(cls, value: Any) -> str | None:
        text = " ".join(str(value or "").split())
        return text or None


class AttentionTarget(BaseModel):
    """A persisted selective-attention target."""

    target_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    label: str
    strength: float = Field(default=0.6, ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.08, ge=0.0, le=1.0)
    source: str = ""
    status: str = "active"
    evidence_refs: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("label", "source", "status", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str:
        return " ".join(str(value or "").split())


class WorkingMemoryItem(BaseModel):
    """A bounded short-term memory slot."""

    item_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    slot: str
    content: str
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = ""
    status: str = "active"
    expires_at: datetime | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("slot", "content", "source", "status", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str:
        return " ".join(str(value or "").split())


class ProspectiveMemory(BaseModel):
    """A future-intention record that can be checked or acted on later."""

    intent_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action: str
    trigger_at: datetime | None = None
    condition: str = ""
    status: str = "active"
    proof_ref: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action", "condition", "status", "proof_ref", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str:
        return " ".join(str(value or "").split())


class LearnedSkill(BaseModel):
    """A reusable procedure extracted from lived tool/action experience."""

    skill_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    name: str
    description: str
    tool_sequence: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    activation_status: str = "candidate"
    risk_tier: str = "workspace_write"
    evidence_refs: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "description", "activation_status", "risk_tier", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str:
        return " ".join(str(value or "").split())


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cognitive_events (
    event_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    summary TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    session_id TEXT,
    confidence REAL NOT NULL DEFAULT 0.6,
    salience REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'active',
    evidence_refs TEXT NOT NULL DEFAULT '[]',
    raw TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cognitive_events_kind_created_at
    ON cognitive_events(kind, created_at);
CREATE INDEX IF NOT EXISTS idx_cognitive_events_session_created_at
    ON cognitive_events(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cognitive_events_status_salience
    ON cognitive_events(status, salience);

CREATE TABLE IF NOT EXISTS attention_targets (
    target_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    label TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 0.6,
    decay_rate REAL NOT NULL DEFAULT 0.08,
    source TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    evidence_refs TEXT NOT NULL DEFAULT '[]',
    raw TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_attention_status_strength
    ON attention_targets(status, strength);

CREATE TABLE IF NOT EXISTS working_memory_items (
    item_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    slot TEXT NOT NULL,
    content TEXT NOT NULL,
    priority REAL NOT NULL DEFAULT 0.5,
    source TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    expires_at TEXT,
    evidence_refs TEXT NOT NULL DEFAULT '[]',
    raw TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_working_memory_status_priority
    ON working_memory_items(status, priority);

CREATE TABLE IF NOT EXISTS prospective_memories (
    intent_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    action TEXT NOT NULL,
    trigger_at TEXT,
    condition TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    proof_ref TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.5,
    evidence_refs TEXT NOT NULL DEFAULT '[]',
    raw TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prospective_status_trigger
    ON prospective_memories(status, trigger_at);

CREATE TABLE IF NOT EXISTS learned_skills (
    skill_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    tool_sequence TEXT NOT NULL DEFAULT '[]',
    preconditions TEXT NOT NULL DEFAULT '[]',
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0.4,
    activation_status TEXT NOT NULL DEFAULT 'candidate',
    risk_tier TEXT NOT NULL DEFAULT 'workspace_write',
    evidence_refs TEXT NOT NULL DEFAULT '[]',
    raw TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_learned_skills_activation_confidence
    ON learned_skills(activation_status, confidence);
"""


class CognitiveStateStore:
    """SQLite-backed state for cognitive analogues and their feedback loops."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "CognitiveStateStore":
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

    async def record_event(
        self,
        kind: CognitiveEventKind | str,
        summary: str,
        *,
        content: str = "",
        source: str = "",
        session_id: str | None = None,
        confidence: float = 0.6,
        salience: float = 1.0,
        status: str = "active",
        evidence_refs: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> CognitiveEvent:
        """Append or replace a cognitive event."""
        assert self._db is not None
        kind_value = kind.value if isinstance(kind, CognitiveEventKind) else str(kind)
        event = CognitiveEvent(
            kind=CognitiveEventKind(kind_value),
            summary=summary,
            content=content,
            source=source,
            session_id=session_id,
            confidence=confidence,
            salience=salience,
            status=status,
            evidence_refs=list(evidence_refs or []),
            payload=dict(payload or {}),
        )
        await self._db.execute(
            """
            INSERT INTO cognitive_events (
                event_id, created_at, kind, summary, content, source, session_id,
                confidence, salience, status, evidence_refs, raw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                created_at = excluded.created_at,
                kind = excluded.kind,
                summary = excluded.summary,
                content = excluded.content,
                source = excluded.source,
                session_id = excluded.session_id,
                confidence = excluded.confidence,
                salience = excluded.salience,
                status = excluded.status,
                evidence_refs = excluded.evidence_refs,
                raw = excluded.raw
            """,
            (
                str(event.event_id),
                event.created_at.isoformat(),
                event.kind.value,
                event.summary,
                event.content,
                event.source,
                event.session_id,
                event.confidence,
                event.salience,
                event.status,
                json.dumps(event.evidence_refs),
                event.model_dump_json(),
            ),
        )
        await self._db.commit()
        return event

    async def list_recent_events(
        self,
        *,
        kind: CognitiveEventKind | str | None = None,
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[CognitiveEvent]:
        """Return recent cognitive events in chronological order."""
        assert self._db is not None
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind.value if isinstance(kind, CognitiveEventKind) else str(kind))
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        params.append(max(1, int(limit)))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = await self._db.execute(
            f"""
            SELECT raw FROM cognitive_events
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        rows = await cursor.fetchall()
        events = [CognitiveEvent.model_validate_json(row["raw"]) for row in rows]
        events.reverse()
        return events

    async def upsert_attention(
        self,
        label: str,
        *,
        strength: float = 0.6,
        decay_rate: float = 0.08,
        source: str = "",
        status: str = "active",
        evidence_refs: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> AttentionTarget:
        """Create or refresh an attention target."""
        assert self._db is not None
        now = datetime.now(timezone.utc)
        target_id = _stable_id("attention", label.lower().strip())
        existing = await self._get_raw("attention_targets", "target_id", target_id)
        created_at = now
        merged_evidence = list(evidence_refs or [])
        if existing:
            previous = AttentionTarget.model_validate_json(existing)
            created_at = previous.created_at
            merged_evidence = _merge_refs(previous.evidence_refs, merged_evidence)
        target = AttentionTarget(
            target_id=target_id,
            created_at=created_at,
            updated_at=now,
            label=label,
            strength=max(0.0, min(1.0, float(strength))),
            decay_rate=max(0.0, min(1.0, float(decay_rate))),
            source=source,
            status=status,
            evidence_refs=merged_evidence,
            payload=dict(payload or {}),
        )
        await self._db.execute(
            """
            INSERT INTO attention_targets (
                target_id, created_at, updated_at, label, strength, decay_rate,
                source, status, evidence_refs, raw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(target_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                label = excluded.label,
                strength = excluded.strength,
                decay_rate = excluded.decay_rate,
                source = excluded.source,
                status = excluded.status,
                evidence_refs = excluded.evidence_refs,
                raw = excluded.raw
            """,
            (
                target.target_id,
                target.created_at.isoformat(),
                target.updated_at.isoformat(),
                target.label,
                target.strength,
                target.decay_rate,
                target.source,
                target.status,
                json.dumps(target.evidence_refs),
                target.model_dump_json(),
            ),
        )
        await self._db.commit()
        await self.record_event(
            CognitiveEventKind.ATTENTION,
            f"Attention target active: {target.label}",
            source=source or "attention_store",
            confidence=target.strength,
            salience=1.0 + target.strength,
            evidence_refs=target.evidence_refs,
            payload={"target_id": target.target_id},
        )
        return target

    async def list_attention(self, *, active_only: bool = True, limit: int = 8) -> list[AttentionTarget]:
        assert self._db is not None
        where = "WHERE status = 'active'" if active_only else ""
        cursor = await self._db.execute(
            f"""
            SELECT raw FROM attention_targets
            {where}
            ORDER BY strength DESC, updated_at DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        )
        rows = await cursor.fetchall()
        return [AttentionTarget.model_validate_json(row["raw"]) for row in rows]

    async def upsert_working_memory(
        self,
        slot: str,
        content: str,
        *,
        priority: float = 0.5,
        source: str = "",
        status: str = "active",
        expires_at: datetime | None = None,
        evidence_refs: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> WorkingMemoryItem:
        """Create or refresh a bounded working-memory slot."""
        assert self._db is not None
        now = datetime.now(timezone.utc)
        item_id = _stable_id("working_memory", slot.lower().strip())
        existing = await self._get_raw("working_memory_items", "item_id", item_id)
        created_at = now
        merged_evidence = list(evidence_refs or [])
        if existing:
            previous = WorkingMemoryItem.model_validate_json(existing)
            created_at = previous.created_at
            merged_evidence = _merge_refs(previous.evidence_refs, merged_evidence)
        item = WorkingMemoryItem(
            item_id=item_id,
            created_at=created_at,
            updated_at=now,
            slot=slot,
            content=content,
            priority=max(0.0, min(1.0, float(priority))),
            source=source,
            status=status,
            expires_at=expires_at,
            evidence_refs=merged_evidence,
            payload=dict(payload or {}),
        )
        await self._db.execute(
            """
            INSERT INTO working_memory_items (
                item_id, created_at, updated_at, slot, content, priority, source,
                status, expires_at, evidence_refs, raw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                content = excluded.content,
                priority = excluded.priority,
                source = excluded.source,
                status = excluded.status,
                expires_at = excluded.expires_at,
                evidence_refs = excluded.evidence_refs,
                raw = excluded.raw
            """,
            (
                item.item_id,
                item.created_at.isoformat(),
                item.updated_at.isoformat(),
                item.slot,
                item.content,
                item.priority,
                item.source,
                item.status,
                item.expires_at.isoformat() if item.expires_at else None,
                json.dumps(item.evidence_refs),
                item.model_dump_json(),
            ),
        )
        await self._db.commit()
        await self.record_event(
            CognitiveEventKind.WORKING_MEMORY,
            f"Working memory slot '{item.slot}' updated",
            content=item.content,
            source=source or "working_memory",
            confidence=item.priority,
            salience=1.0 + item.priority,
            evidence_refs=item.evidence_refs,
            payload={"item_id": item.item_id},
        )
        return item

    async def list_working_memory(self, *, active_only: bool = True, limit: int = 8) -> list[WorkingMemoryItem]:
        assert self._db is not None
        now = datetime.now(timezone.utc).isoformat()
        clauses = []
        params: list[Any] = []
        if active_only:
            clauses.append("status = 'active'")
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(now)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        cursor = await self._db.execute(
            f"""
            SELECT raw FROM working_memory_items
            {where}
            ORDER BY priority DESC, updated_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        rows = await cursor.fetchall()
        return [WorkingMemoryItem.model_validate_json(row["raw"]) for row in rows]

    async def upsert_prospective_memory(
        self,
        action: str,
        *,
        trigger_at: datetime | None = None,
        condition: str = "",
        status: str = "active",
        proof_ref: str = "",
        confidence: float = 0.5,
        evidence_refs: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ProspectiveMemory:
        """Create or refresh a future-intention record."""
        assert self._db is not None
        now = datetime.now(timezone.utc)
        intent_id = _stable_id("prospective", action.lower().strip(), condition.lower().strip())
        existing = await self._get_raw("prospective_memories", "intent_id", intent_id)
        created_at = now
        merged_evidence = list(evidence_refs or [])
        if existing:
            previous = ProspectiveMemory.model_validate_json(existing)
            created_at = previous.created_at
            merged_evidence = _merge_refs(previous.evidence_refs, merged_evidence)
        memory = ProspectiveMemory(
            intent_id=intent_id,
            created_at=created_at,
            updated_at=now,
            action=action,
            trigger_at=trigger_at,
            condition=condition,
            status=status,
            proof_ref=proof_ref,
            confidence=confidence,
            evidence_refs=merged_evidence,
            payload=dict(payload or {}),
        )
        await self._db.execute(
            """
            INSERT INTO prospective_memories (
                intent_id, created_at, updated_at, action, trigger_at, condition,
                status, proof_ref, confidence, evidence_refs, raw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(intent_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                trigger_at = excluded.trigger_at,
                condition = excluded.condition,
                status = excluded.status,
                proof_ref = excluded.proof_ref,
                confidence = excluded.confidence,
                evidence_refs = excluded.evidence_refs,
                raw = excluded.raw
            """,
            (
                memory.intent_id,
                memory.created_at.isoformat(),
                memory.updated_at.isoformat(),
                memory.action,
                memory.trigger_at.isoformat() if memory.trigger_at else None,
                memory.condition,
                memory.status,
                memory.proof_ref,
                memory.confidence,
                json.dumps(memory.evidence_refs),
                memory.model_dump_json(),
            ),
        )
        await self._db.commit()
        await self.record_event(
            CognitiveEventKind.PROSPECTIVE_MEMORY,
            f"Prospective memory active: {memory.action}",
            source="prospective_memory",
            confidence=memory.confidence,
            salience=1.4,
            evidence_refs=memory.evidence_refs,
            payload={"intent_id": memory.intent_id, "condition": memory.condition},
        )
        return memory

    async def list_prospective_memories(self, *, active_only: bool = True, limit: int = 10) -> list[ProspectiveMemory]:
        assert self._db is not None
        where = "WHERE status = 'active'" if active_only else ""
        cursor = await self._db.execute(
            f"""
            SELECT raw FROM prospective_memories
            {where}
            ORDER BY
                CASE WHEN trigger_at IS NULL THEN 1 ELSE 0 END ASC,
                trigger_at ASC,
                confidence DESC,
                updated_at DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        )
        rows = await cursor.fetchall()
        return [ProspectiveMemory.model_validate_json(row["raw"]) for row in rows]

    async def upsert_learned_skill(self, skill: LearnedSkill) -> LearnedSkill:
        """Create or update an evidence-gated learned procedure."""
        assert self._db is not None
        if not skill.skill_id:
            skill.skill_id = _stable_id("skill", " ".join(skill.tool_sequence), skill.name.lower())
        existing = await self._get_raw("learned_skills", "skill_id", skill.skill_id)
        if existing:
            previous = LearnedSkill.model_validate_json(existing)
            new_refs = [ref for ref in skill.evidence_refs if ref not in previous.evidence_refs]
            success_count = previous.success_count + (skill.success_count if new_refs else 0)
            failure_count = previous.failure_count + (skill.failure_count if new_refs else 0)
            evidence_refs = _merge_refs(previous.evidence_refs, skill.evidence_refs)
            confidence = _skill_confidence(success_count, failure_count)
            skill = skill.model_copy(
                update={
                    "created_at": previous.created_at,
                    "updated_at": datetime.now(timezone.utc),
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "confidence": confidence,
                    "activation_status": _activation_status(confidence, skill.risk_tier),
                    "evidence_refs": evidence_refs,
                }
            )
        else:
            confidence = _skill_confidence(skill.success_count, skill.failure_count)
            skill = skill.model_copy(
                update={
                    "updated_at": datetime.now(timezone.utc),
                    "confidence": confidence,
                    "activation_status": _activation_status(confidence, skill.risk_tier),
                }
            )
        await self._db.execute(
            """
            INSERT INTO learned_skills (
                skill_id, created_at, updated_at, name, description, tool_sequence,
                preconditions, success_count, failure_count, confidence,
                activation_status, risk_tier, evidence_refs, raw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(skill_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                name = excluded.name,
                description = excluded.description,
                tool_sequence = excluded.tool_sequence,
                preconditions = excluded.preconditions,
                success_count = excluded.success_count,
                failure_count = excluded.failure_count,
                confidence = excluded.confidence,
                activation_status = excluded.activation_status,
                risk_tier = excluded.risk_tier,
                evidence_refs = excluded.evidence_refs,
                raw = excluded.raw
            """,
            (
                skill.skill_id,
                skill.created_at.isoformat(),
                skill.updated_at.isoformat(),
                skill.name,
                skill.description,
                json.dumps(skill.tool_sequence),
                json.dumps(skill.preconditions),
                skill.success_count,
                skill.failure_count,
                skill.confidence,
                skill.activation_status,
                skill.risk_tier,
                json.dumps(skill.evidence_refs),
                skill.model_dump_json(),
            ),
        )
        await self._db.commit()
        await self.record_event(
            CognitiveEventKind.PROCEDURAL_SKILL,
            f"Learned procedure {skill.activation_status}: {skill.name}",
            content=skill.description,
            source="procedural_memory",
            confidence=skill.confidence,
            salience=1.0 + skill.confidence,
            evidence_refs=skill.evidence_refs,
            payload={"skill_id": skill.skill_id, "tool_sequence": skill.tool_sequence},
        )
        return skill

    async def list_learned_skills(
        self,
        *,
        activation_status: str | None = None,
        query: str = "",
        limit: int = 10,
    ) -> list[LearnedSkill]:
        assert self._db is not None
        clauses: list[str] = []
        params: list[Any] = []
        if activation_status is not None:
            clauses.append("activation_status = ?")
            params.append(activation_status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query_tokens = _tokens(query)
        candidate_limit = max(1, int(limit))
        if query_tokens:
            candidate_limit = max(candidate_limit, 50)
        params.append(candidate_limit)
        cursor = await self._db.execute(
            f"""
            SELECT raw FROM learned_skills
            {where}
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        rows = await cursor.fetchall()
        skills = [LearnedSkill.model_validate_json(row["raw"]) for row in rows]
        if query_tokens:
            skills = sorted(
                skills,
                key=lambda skill: (
                    _skill_query_score(query_tokens, skill),
                    skill.confidence,
                    skill.updated_at,
                ),
                reverse=True,
            )
            skills = [skill for skill in skills if _skill_query_score(query_tokens, skill) > 0]
        return skills[: max(1, int(limit))]

    async def extract_skills_from_procedural_memory(self, memory_store: Any, *, limit: int = 200) -> dict[str, Any]:
        """Mine procedural episodes into reusable skill candidates."""
        list_recent = getattr(memory_store, "list_recent_episodes", None)
        if not callable(list_recent):
            return {"available": False, "skills_created_or_updated": 0}
        episodes = await list_recent(limit=limit)
        procedural = [ep for ep in episodes if getattr(ep, "kind", None) == EpisodeKind.PROCEDURAL]
        updated: list[LearnedSkill] = []
        for episode in procedural:
            tools = _extract_tool_sequence(str(getattr(episode, "content", "") or ""))
            if not tools:
                continue
            objective = _extract_objective(str(getattr(episode, "content", "") or ""))
            risk_tier = _skill_risk_tier(tools)
            evidence_ref = f"episode:{getattr(episode, 'episode_id', '')}"
            success = _episode_succeeded(episode)
            skill = LearnedSkill(
                name=_skill_name(objective, tools),
                description=(
                    f"Procedure learned from procedural memory. Objective: {objective or 'unspecified'}. "
                    f"Tool sequence: {' -> '.join(tools)}."
                ),
                tool_sequence=tools,
                preconditions=[objective] if objective else [],
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                risk_tier=risk_tier,
                evidence_refs=[evidence_ref],
                payload={
                    "source_session_id": getattr(episode, "session_id", None),
                    "source_episode_id": str(getattr(episode, "episode_id", "")),
                },
            )
            updated.append(await self.upsert_learned_skill(skill))
        return {
            "available": True,
            "procedural_episodes_examined": len(procedural),
            "skills_created_or_updated": len(updated),
            "auto_usable": sum(1 for item in updated if item.activation_status == "evidence_gated_auto_use"),
            "skill_ids": [item.skill_id for item in updated],
        }

    async def query_context(self, query: str, *, limit: int = 8) -> dict[str, Any]:
        """Return compact cognitive evidence ranked by simple query relevance."""
        query_tokens = _tokens(query)
        events = await self.list_recent_events(limit=80)
        attention = await self.list_attention(limit=8)
        working = await self.list_working_memory(limit=8)
        skills = await self.list_learned_skills(query=query, limit=12)
        scored_events = sorted(
            (
                (
                    _score_text(query_tokens, f"{event.kind.value} {event.summary} {event.content}")
                    + event.salience * 0.05,
                    event,
                )
                for event in events
            ),
            key=lambda item: (item[0], item[1].created_at),
            reverse=True,
        )
        return {
            "available": True,
            "query": query,
            "attention": [item.model_dump(mode="json") for item in attention[:limit]],
            "working_memory": [item.model_dump(mode="json") for item in working[:limit]],
            "events": [
                event.model_dump(mode="json")
                for score, event in scored_events[:limit]
                if score > 0 or not query_tokens
            ],
            "learned_skills": [item.model_dump(mode="json") for item in skills[:limit]],
        }

    async def prompt_block(
        self,
        *,
        query: str = "",
        session_id: str | None = None,
        char_budget: int = 1400,
    ) -> str:
        """Render a concise model-facing cognitive-state block."""
        attention = await self.list_attention(limit=4)
        working = await self.list_working_memory(limit=5)
        prospective = await self.list_prospective_memories(limit=3)
        query_tokens = _tokens(query)
        recent = await self.list_recent_events(session_id=session_id, limit=40)
        if not recent:
            recent = await self.list_recent_events(limit=40)
        scored = sorted(
            (
                (
                    _score_text(query_tokens, f"{event.kind.value} {event.summary} {event.content}"),
                    event.salience,
                    event,
                )
                for event in recent
            ),
            key=lambda item: (item[0] + item[1] * 0.04, item[2].created_at),
            reverse=True,
        )
        relevant = [
            event
            for query_score, salience, event in scored
            if query_score > 0 or (not query_tokens and salience >= 1.4)
        ][:5]
        skills = await self.list_learned_skills(
            activation_status="evidence_gated_auto_use",
            query=query,
            limit=3,
        )
        if query_tokens and not skills:
            skills = await self.list_learned_skills(activation_status="evidence_gated_auto_use", limit=3)

        lines: list[str] = []
        if attention:
            lines.append("Cognitive state evidence:")
            lines.append("Attention/focus:")
            lines.extend(
                f"- {item.label} (strength {item.strength:.2f}; source {item.source or 'unknown'})"
                for item in attention[:4]
            )
        if working:
            if not lines:
                lines.append("Cognitive state evidence:")
            lines.append("Working memory:")
            lines.extend(f"- {item.slot}: {item.content}" for item in working[:5])
        if prospective:
            if not lines:
                lines.append("Cognitive state evidence:")
            lines.append("Prospective memory:")
            lines.extend(
                f"- {item.action} ({item.condition or item.trigger_at or 'future condition'})"
                for item in prospective[:3]
            )
        if relevant:
            if not lines:
                lines.append("Cognitive state evidence:")
            lines.append("Recent cognitive feedback:")
            lines.extend(
                f"- [{event.kind.value}] {event.summary}"
                for event in relevant[:5]
            )
        if skills:
            if not lines:
                lines.append("Cognitive state evidence:")
            lines.append("Evidence-gated learned procedures:")
            lines.extend(
                f"- {skill.name} (confidence {skill.confidence:.2f}; tools {' -> '.join(skill.tool_sequence[:5])})"
                for skill in skills[:3]
            )
        if not lines:
            return ""
        lines.append(
            "Use this as compact autobiographical/cognitive evidence. It is not a response script; "
            "apply it only when relevant and keep claims grounded in the evidence shown. "
            "When a turn reveals a reusable procedure or the operator asks to teach/remember a workflow, "
            "preserve it with cognitive_skill_create using concrete evidence_refs and tool_sequence."
        )
        block = "\n".join(lines)
        return block[: max(200, int(char_budget))]

    async def _get_raw(self, table: str, key_name: str, key_value: str) -> str | None:
        assert self._db is not None
        cursor = await self._db.execute(
            f"SELECT raw FROM {table} WHERE {key_name} = ?",
            (key_value,),
        )
        row = await cursor.fetchone()
        return str(row["raw"]) if row else None


def _stable_id(*parts: str) -> str:
    material = "\x1f".join(parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _merge_refs(left: list[str], right: list[str]) -> list[str]:
    merged: list[str] = []
    for ref in [*left, *right]:
        text = str(ref or "").strip()
        if text and text not in merged:
            merged.append(text)
    return merged


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_./:-]{3,}", str(text or "").lower())
        if token not in {"the", "and", "for", "that", "with", "this"}
    }


def _score_text(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = _tokens(text)
    if not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return overlap / max(1, len(query_tokens))


def _skill_query_score(query_tokens: set[str], skill: LearnedSkill) -> float:
    """Score a learned skill by task relevance, not global confidence alone."""

    if not query_tokens:
        return 0.0
    name_score = _score_text(query_tokens, skill.name) * 1.6
    description_score = _score_text(query_tokens, skill.description)
    precondition_score = _score_text(query_tokens, " ".join(skill.preconditions)) * 1.3
    tool_score = _score_text(query_tokens, " ".join(skill.tool_sequence)) * 0.7
    return name_score + description_score + precondition_score + tool_score


def _extract_tool_sequence(content: str) -> list[str]:
    tools: list[str] = []
    for match in re.finditer(r"\btool\s+([A-Za-z0-9_:-]+)", content):
        name = match.group(1).rstrip(":")
        if name and (not tools or tools[-1] != name):
            tools.append(name)
    if tools:
        return tools[:12]
    for match in re.finditer(r"[-*]\s*([A-Za-z0-9_:-]+)", content):
        name = match.group(1).rstrip(":")
        if "_" in name and name not in tools:
            tools.append(name)
    return tools[:12]


def _extract_objective(content: str) -> str:
    for pattern in (r"Objective:\s*(.+)", r"objective\s*=\s*['\"]?([^'\n\"]+)"):
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return " ".join(match.group(1).split())[:180]
    first = " ".join(content.splitlines()[0].split()) if content.splitlines() else ""
    return first[:180]


def _episode_succeeded(episode: Any) -> bool:
    content = str(getattr(episode, "content", "") or "").lower()
    if "outcome: failure" in content or "failed" in content:
        return False
    return "success" in content or float(getattr(episode, "confidence_score", 0.8) or 0.0) >= 0.5


def _skill_risk_tier(tools: list[str]) -> str:
    joined = " ".join(tools)
    if any(token in joined for token in ("rm", "delete", "cancel", "send_email", "calendar_delete")):
        return "external_write"
    if any(token in joined for token in ("bash", "pty", "process", "fs_write", "edit", "apply_patch")):
        return "workspace_write"
    return "readonly"


def _skill_confidence(success_count: int, failure_count: int) -> float:
    total = success_count + failure_count
    if total <= 0:
        return 0.35
    base = success_count / max(1, total)
    evidence_bonus = min(0.2, total * 0.04)
    return round(max(0.0, min(0.95, 0.35 + base * 0.4 + evidence_bonus - failure_count * 0.03)), 3)


def _activation_status(confidence: float, risk_tier: str) -> str:
    if confidence >= 0.58 and risk_tier in {"readonly", "workspace_write"}:
        return "evidence_gated_auto_use"
    if confidence >= 0.72:
        return "review_required"
    return "candidate"


def _skill_name(objective: str, tools: list[str]) -> str:
    if objective:
        cleaned = re.sub(r"[^A-Za-z0-9 _-]+", "", objective).strip()
        if cleaned:
            return "Procedure: " + cleaned[:72]
    return "Procedure: " + " -> ".join(tools[:3])
