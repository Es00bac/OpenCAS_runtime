"""Learning helpers for wellbeing maintenance outcomes."""

from __future__ import annotations

from .models import LearnedMaintenanceBehavior, MaintenanceOutcome


class WellbeingLearning:
    """Convert maintenance outcomes into conservative future-routing hints."""

    def learn_from_outcome(self, outcome: MaintenanceOutcome) -> LearnedMaintenanceBehavior:
        delta = round(outcome.before_risk - outcome.after_risk, 3)
        if delta >= 0.15:
            return LearnedMaintenanceBehavior(
                action_type=outcome.action_type,
                effect_direction="improved",
                confidence=min(1.0, 0.55 + delta),
                reinforce_action=True,
                evidence_ids=list(outcome.evidence_ids),
                meta={"risk_delta": delta, "outcome": outcome.outcome},
            )
        if delta <= -0.15:
            return LearnedMaintenanceBehavior(
                action_type=outcome.action_type,
                effect_direction="worsened",
                confidence=min(1.0, 0.55 + abs(delta)),
                inspect_next_time=True,
                evidence_ids=list(outcome.evidence_ids),
                meta={"risk_delta": delta, "outcome": outcome.outcome},
            )
        return LearnedMaintenanceBehavior(
            action_type=outcome.action_type,
            effect_direction="neutral",
            confidence=0.45,
            evidence_ids=list(outcome.evidence_ids),
            meta={"risk_delta": delta, "outcome": outcome.outcome},
        )
