"""Possibility signals produced from grounded daydream thoughts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from opencas.cognition import CognitionGrounding


class PossibilitySignalRoute(str, Enum):
    """Routing decisions available to the daydream signal loop."""

    DISCARD = "discard"
    INCUBATE = "incubate"
    THREAD = "thread"
    SELF_NOTE = "self_note"
    SELF_EXPERIMENT = "self_experiment"
    SELF_PROTOTYPE = "self_prototype"
    RESEARCH = "research"
    WORK_CANDIDATE = "work_candidate"
    ASK_USER = "ask_user"
    COMPOST = "compost"
    ROUTE_FAILED = "route_failed"


class ContactPosture(str, Enum):
    """How owner contact should be considered for a signal."""

    SILENT = "silent"
    ASK_FIRST = "ask_first"
    SHARE_AFTER_ARTIFACT = "share_after_artifact"
    SHARE_FAILURE = "share_failure"
    URGENT = "urgent"


class SelfWorkKind(str, Enum):
    """Contained self-work artifact type suggested by a signal."""

    NONE = "none"
    NOTE = "note"
    RESEARCH = "research"
    EXPERIMENT = "experiment"
    PROTOTYPE = "prototype"
    COMPOST = "compost"


class PossibilitySignal(BaseModel):
    """A daydream thought normalized into an inspectable routing unit."""

    signal_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_reflection_id: str
    source_thought_index: int = 0
    source_mode: str = "waking_daydream"
    summary: str
    imaginative_branch: str = ""
    practical_branch: str = ""
    bridge: str = ""
    novelty: float = Field(default=0.5, ge=0.0, le=1.0)
    usefulness: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    risk: float = Field(default=0.0, ge=0.0, le=1.0)
    grounding: List[CognitionGrounding] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    suggested_route: PossibilitySignalRoute = PossibilitySignalRoute.INCUBATE
    suggested_handler: str = ""
    contact_posture: ContactPosture = ContactPosture.SILENT
    self_work_kind: SelfWorkKind = SelfWorkKind.NONE
    self_work_intent: str = ""
    route_status: str = "pending"
    route_reason: str = ""
    routed_at: Optional[datetime] = None
    artifact_paths: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class SelfWorkReceipt(BaseModel):
    """Receipt for a self-work artifact or compost record."""

    receipt_id: str = Field(default_factory=lambda: str(uuid4()))
    signal_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    route: PossibilitySignalRoute
    kind: SelfWorkKind = SelfWorkKind.NONE
    outcome: str = ""
    summary: str = ""
    artifact_paths: List[str] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)
