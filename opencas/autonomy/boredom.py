"""Boredom physics engine for modulating daydream frequency and creative pacing."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional


class BoredomPhysics:
    """Track idle time and compute boredom/motivation scores."""

    def __init__(self) -> None:
        self._last_activity_at: datetime = datetime.now(timezone.utc)
        self._last_reset_at: datetime = datetime.now(timezone.utc)

    def record_activity(self) -> None:
        """Reset boredom clocks on significant activity (e.g., user message)."""
        self._last_activity_at = datetime.now(timezone.utc)
        self._last_reset_at = self._last_activity_at

    def record_reset(self) -> None:
        """Reset boredom after a daydream or other autonomy reset."""
        self._last_reset_at = datetime.now(timezone.utc)

    def compute_boredom(self, now: Optional[datetime] = None) -> float:
        """Return boredom in [0, 1] based on hours since last activity or reset."""
        if now is None:
            now = datetime.now(timezone.utc)
        last = max(self._last_activity_at, self._last_reset_at)
        idle_hours = max(0.0, (now - last).total_seconds() / 3600.0)
        return math.tanh(idle_hours / 2.0)

    def compute_motivation(
        self,
        somatic_readiness: float = 0.5,
        now: Optional[datetime] = None,
    ) -> float:
        """Return motivation as weighted blend of boredom and somatic state."""
        boredom = self.compute_boredom(now)
        # Clamp somatic readiness to [0, 1]
        readiness = max(0.0, min(1.0, somatic_readiness))
        return 0.76 * boredom + 0.24 * readiness

    def should_daydream(
        self,
        somatic_readiness: float = 0.5,
        motivation_threshold: float = 0.55,
        now: Optional[datetime] = None,
    ) -> bool:
        """Return True if boredom physics says a daydream is warranted."""
        motivation = self.compute_motivation(somatic_readiness, now)
        return motivation >= motivation_threshold
