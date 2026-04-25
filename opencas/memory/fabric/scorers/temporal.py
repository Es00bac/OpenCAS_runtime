"""Temporal (time-decay) scorer."""

from __future__ import annotations

from typing import Any, Optional

from opencas.memory import Episode


class TemporalScorer:
    """Score temporal proximity with exponential decay over days."""

    def __init__(self, decay_days: float = 120.0) -> None:
        self.decay_days = decay_days

    async def score(
        self,
        ep_a: Episode,
        ep_b: Episode,
        context: Optional[Any] = None,
    ) -> float:
        delta = abs((ep_a.created_at - ep_b.created_at).total_seconds())
        days = delta / 86_400
        return round(max(0.0, 1.0 - (days / self.decay_days)), 4)
