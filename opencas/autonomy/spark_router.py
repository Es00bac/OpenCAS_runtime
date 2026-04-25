"""Spark router: decides which rung a daydream spark should land on."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from opencas.autonomy.models import WorkObject

from .workspace import ExecutiveWorkspace


class SparkRung(str, Enum):
    """Possible routing decisions for a spark."""

    REJECT = "reject"
    NOTE = "note"
    MICRO_TASK = "micro_task"
    FULL_TASK = "full_task"


class SparkRouter:
    """Routes daydream sparks into the creative ladder or rejection."""

    def __init__(self) -> None:
        # track rejections for persistent intent bypass: source_key -> list of rejection timestamps
        self._rejection_history: Dict[str, List[datetime]] = {}

    def route(
        self,
        spark: WorkObject,
        workspace: Optional[ExecutiveWorkspace],
        boredom: float,
        now: Optional[datetime] = None,
    ) -> SparkRung:
        """Decide the rung for a spark."""
        if now is None:
            now = datetime.now(timezone.utc)

        score = spark.promotion_score
        source_key = self._source_key(spark)

        # Persistent intent bypass
        intensity = float(spark.meta.get("intensity", 0)) if spark.meta else 0.0
        if self._persistent_intent_bypass(source_key, score, intensity, now):
            return SparkRung.FULL_TASK

        if boredom < 0.3 or score < 0.25:
            self._record_rejection(source_key, now)
            return SparkRung.REJECT
        if score < 0.45:
            return SparkRung.NOTE
        if score < 0.65:
            return SparkRung.MICRO_TASK
        return SparkRung.FULL_TASK

    @staticmethod
    def _source_key(spark: WorkObject) -> str:
        """Derive a stable source key from meta or content hash."""
        origin = spark.meta.get("origin") if spark.meta else None
        if origin:
            return str(origin)
        # fallback to first 60 chars of content
        return spark.content[:60].strip().lower()

    def _persistent_intent_bypass(
        self,
        source_key: str,
        score: float,
        intensity: float,
        now: datetime,
    ) -> bool:
        """Allow a full-task bypass if a source has been repeatedly rejected."""
        if score < 0.6 or intensity < 0.6:
            return False
        history = self._rejection_history.get(source_key, [])
        # Count rejections in last 24h
        cutoff = now.astimezone(timezone.utc) - __import__("datetime").timedelta(hours=24)
        recent = [t for t in history if t >= cutoff]
        if len(recent) >= 3:
            # Consume the bypass by clearing recent history for this source
            self._rejection_history[source_key] = [t for t in history if t < cutoff]
            return True
        return False

    def _record_rejection(self, source_key: str, now: datetime) -> None:
        self._rejection_history.setdefault(source_key, []).append(now)
