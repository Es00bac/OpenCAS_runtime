"""Relational (project/session/target overlap) scorer."""

from __future__ import annotations

from typing import Any, Optional

from opencas.memory import Episode


class RelationalScorer:
    """Score relational proximity between two episodes."""

    async def score(
        self,
        ep_a: Episode,
        ep_b: Episode,
        context: Optional[Any] = None,
    ) -> float:
        score = 0.0
        same_project = (
            ep_a.payload.get("project_id")
            and ep_a.payload.get("project_id") == ep_b.payload.get("project_id")
        )
        same_session = ep_a.session_id and ep_a.session_id == ep_b.session_id
        same_target = False
        if ep_a.affect and ep_b.affect:
            same_target = (ep_a.affect.social_target or None) == (
                ep_b.affect.social_target or None
            )
        if same_project:
            score += 0.4
        if same_session:
            score += 0.3
        if same_target:
            score += 0.3
        return round(min(1.0, score), 4)
