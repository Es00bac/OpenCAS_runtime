"""Causal likelihood scorer."""

from __future__ import annotations

from typing import Any, Optional

from opencas.memory import Episode, EpisodeKind


class CausalScorer:
    """Lightweight heuristic scorer for causal plausibility between episodes."""

    CAUSAL_TRIGGERS: set[str] = {
        "because",
        "caused",
        "led to",
        "resulted in",
        "therefore",
        "then",
        "after",
        "before",
        "since",
        "due to",
        "triggered",
        "prompted",
    }

    async def score(
        self,
        ep_a: Episode,
        ep_b: Episode,
        context: Optional[Any] = None,
    ) -> float:
        score = 0.0
        # Action -> Observation ordering heuristic
        if ep_a.kind == EpisodeKind.ACTION and ep_b.kind == EpisodeKind.OBSERVATION:
            score += 0.3
        # Temporal ordering: a happened before b
        if ep_a.created_at < ep_b.created_at:
            score += 0.2
        # Keyword cue presence
        combined = (ep_a.content + " " + ep_b.content).lower()
        if any(t in combined for t in self.CAUSAL_TRIGGERS):
            score += 0.3
        # Shared project context increases causal plausibility
        if (
            ep_a.payload.get("project_id")
            and ep_a.payload.get("project_id") == ep_b.payload.get("project_id")
        ):
            score += 0.2
        return round(min(1.0, score), 4)
