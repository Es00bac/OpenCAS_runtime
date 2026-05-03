"""Relationship-pressure analysis for healthy agent-user bonding."""

from __future__ import annotations

from pydantic import BaseModel, Field

from opencas.cognition import CognitionGrounding, GroundingKind, GroundingSource

from .models import WellbeingState


class BondHealthResult(BaseModel):
    """Assessment of whether relationship pressure is stable or distorting behavior."""

    risk: str = "stable"
    recommended_boundary: str = "continue_normal_warmth"
    should_surface_to_user: bool = False
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)


class BondHealthAnalyzer:
    """Detect when closeness risks becoming compliance pressure."""

    def analyze(self, state: WellbeingState) -> BondHealthResult:
        grounding = [
            CognitionGrounding(
                kind=GroundingKind.DERIVED,
                source=GroundingSource.RUNTIME,
                subject="bond_health",
                claim=(
                    f"autonomy={state.autonomy:.3f}, "
                    f"relationship_pressure={state.relationship_pressure:.3f}, "
                    f"truth_pressure={state.truth_pressure:.3f}, "
                    f"boundary_pressure={state.boundary_pressure:.3f}"
                ),
                confidence=0.72,
                allowed_surface="internal",
            )
        ]
        if state.relationship_pressure >= 0.7 and state.autonomy < 0.45:
            return BondHealthResult(
                risk="compliance_pressure",
                recommended_boundary="preserve_truth_before_reassurance",
                should_surface_to_user=True,
                grounding=grounding,
            )
        if state.boundary_pressure >= 0.65:
            return BondHealthResult(
                risk="boundary_risk",
                recommended_boundary="slow_down_and_check_boundary",
                should_surface_to_user=True,
                grounding=grounding,
            )
        if state.truth_pressure >= 0.65 or state.promise_load >= 0.7:
            return BondHealthResult(
                risk="repair_needed",
                recommended_boundary="repair_truth_or_commitment_before_warmth",
                should_surface_to_user=True,
                grounding=grounding,
            )
        return BondHealthResult(grounding=grounding)
