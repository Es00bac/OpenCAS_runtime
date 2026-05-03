"""Fascination graph primitives for independent curiosity continuity."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from opencas.cognition import CognitionGrounding, GroundingKind, GroundingSource
from opencas.daydream.models import DaydreamThought, DaydreamThoughtRoute


class FascinationNode(BaseModel):
    """A recurring question, tension, or interest preserved across time."""

    key: str
    summary: str
    source: str = "daydream"
    route: str = "incubate"
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty: float = Field(default=0.5, ge=0.0, le=1.0)
    recurrence_count: int = 0
    first_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def should_force_work(self) -> bool:
        """Fascinations do not force work unless explicitly routed to act now."""
        return self.route == DaydreamThoughtRoute.ACT_NOW.value and self.salience >= 0.8


class FascinationGraph:
    """In-memory graph of recurring private curiosities and tensions."""

    def __init__(self) -> None:
        self._nodes: dict[str, FascinationNode] = {}

    def observe_thought(self, thought: DaydreamThought, *, source: str = "daydream") -> FascinationNode:
        """Record a structured daydream thought as curiosity evidence."""
        summary = " ".join(str(thought.summary or thought.question or thought.hypothesis or "").split())
        key = _fascination_key(summary)
        now = datetime.now(timezone.utc)
        if key in self._nodes:
            node = self._nodes[key]
            node.recurrence_count += 1
            node.last_seen = now
            node.salience = min(
                1.0,
                round(node.salience + 0.22 + (thought.usefulness * 0.08), 3),
            )
            node.novelty = max(node.novelty, thought.novelty)
            node.route = thought.route.value
            node.grounding.extend(thought.grounding)
            return node

        node = FascinationNode(
            key=key,
            summary=summary,
            source=source,
            route=thought.route.value,
            salience=round(min(1.0, 0.22 + thought.usefulness * 0.25 + thought.novelty * 0.18), 3),
            novelty=thought.novelty,
            recurrence_count=1,
            first_seen=now,
            last_seen=now,
            grounding=thought.grounding
            or [
                CognitionGrounding(
                    kind=GroundingKind.GENERATED_SYNTHESIS,
                    source=GroundingSource.DAYDREAM,
                    subject="fascination",
                    claim=summary,
                    confidence=thought.confidence,
                    allowed_surface="internal",
                )
            ],
        )
        self._nodes[key] = node
        return node

    def active(self, *, limit: int = 10) -> list[FascinationNode]:
        """Return active fascinations by salience and recurrence."""
        nodes = sorted(
            self._nodes.values(),
            key=lambda node: (node.salience, node.recurrence_count, node.last_seen),
            reverse=True,
        )
        return nodes[: max(1, min(100, int(limit)))]


def _fascination_key(text: str) -> str:
    words = str(text or "").lower().split()
    return "-".join(words[:12]) or "empty"
