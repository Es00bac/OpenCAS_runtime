"""Bounded planning for autonomous self-maintenance."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .models import MaintenanceAction, MaintenanceActionType, WellbeingAssessment

PROMISE_PRESERVATION_THRESHOLD = 0.2


class MaintenancePlan(BaseModel):
    """A concrete, bounded maintenance plan derived from wellbeing assessment."""

    actions: list[MaintenanceAction] = Field(default_factory=list)
    background_throttle: bool = False
    surface_to_user: bool = False
    hot_apply_allowed: bool = False
    meta: dict = Field(default_factory=dict)


class MaintenancePlanner:
    """Convert wellbeing state into safe maintenance actions."""

    def plan(self, assessment: WellbeingAssessment) -> MaintenancePlan:
        state = assessment.state
        actions: list[MaintenanceAction] = []
        background_throttle = False
        surface_to_user = False

        if state.recovery_need >= 0.7:
            background_throttle = True
            actions.append(
                MaintenanceAction(
                    action_type=MaintenanceActionType.RECOVER,
                    reason="Recovery need is high; throttle low-value background work and preserve continuity.",
                    priority=0.95,
                    grounding=state.grounding,
                )
            )

        if state.promise_load >= PROMISE_PRESERVATION_THRESHOLD:
            actions.append(
                MaintenanceAction(
                    action_type=MaintenanceActionType.PRESERVE_PROMISES,
                    reason="Active or unresolved commitments need work, schedule, or blocked-state provenance.",
                    priority=max(0.72, state.promise_load),
                    grounding=state.grounding,
                )
            )

        if state.relationship_pressure >= 0.65 or state.boundary_pressure >= 0.55:
            actions.append(
                MaintenanceAction(
                    action_type=MaintenanceActionType.BOUNDARY_CHECK,
                    reason="Relationship pressure is high; preserve truth before closeness or compliance.",
                    priority=max(0.7, state.relationship_pressure, state.boundary_pressure),
                    grounding=state.grounding,
                )
            )

        if state.truth_pressure >= 0.55:
            surface_to_user = True
            actions.append(
                MaintenanceAction(
                    action_type=MaintenanceActionType.TRUTH_REPAIR,
                    reason="Truth pressure is high; acknowledge uncertainty or unsupported claims from evidence.",
                    priority=max(0.72, state.truth_pressure),
                    grounding=state.grounding,
                )
            )

        if state.coherence < 0.55 or state.drift_load >= 0.55:
            actions.append(
                MaintenanceAction(
                    action_type=MaintenanceActionType.DEEP_THINK,
                    reason="Coherence or drift pressure needs slow reflection before more output.",
                    priority=max(0.62, 1.0 - state.coherence, state.drift_load),
                    grounding=state.grounding,
                )
            )

        if state.curiosity < 0.35:
            actions.append(
                MaintenanceAction(
                    action_type=MaintenanceActionType.CURIOSITY_INCUBATE,
                    reason="Private curiosity continuity is low; incubate recurring questions instead of becoming only user-reactive.",
                    priority=max(0.58, 1.0 - state.curiosity),
                    grounding=state.grounding,
                )
            )

        if state.drift_load >= 0.75 and state.truth_pressure >= 0.5:
            actions.append(
                MaintenanceAction(
                    action_type=MaintenanceActionType.SELF_MODIFICATION_PROPOSAL,
                    reason="Repeated drift plus truth pressure may justify a reviewable prompt or code proposal.",
                    priority=max(0.65, state.drift_load),
                    grounding=state.grounding,
                )
            )

        if not actions:
            actions.append(
                MaintenanceAction(
                    action_type=MaintenanceActionType.OPERATOR_SUMMARY,
                    reason="Wellbeing state is stable; no maintenance action is needed beyond normal observability.",
                    priority=0.1,
                    grounding=state.grounding,
                )
            )

        actions.sort(key=lambda item: item.priority, reverse=True)
        return MaintenancePlan(
            actions=actions,
            background_throttle=background_throttle,
            surface_to_user=surface_to_user,
            hot_apply_allowed=False,
            meta={"overall_risk": state.overall_risk},
        )
