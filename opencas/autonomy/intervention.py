"""Intervention policy for the executive workspace."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .workspace import ExecutionMode, ExecutiveWorkspace, WorkspaceAffinity


class InterventionKind(str, Enum):
    """Possible executive interventions."""

    LAUNCH_BACKGROUND = "launch_background"
    SURFACE_CLARIFICATION = "surface_clarification"
    SURFACE_APPROVAL = "surface_approval"
    VERIFY_COMPLETED_WORK = "verify_completed_work"
    RECLAIM_TO_FOREGROUND = "reclaim_to_foreground"
    RETIRE_OR_DEFER_FOCUS = "retire_or_defer_focus"
    NO_INTERVENTION = "no_intervention"


class InterventionDecision(BaseModel):
    """Result of evaluating the intervention policy."""

    decision_id: UUID = Field(default_factory=uuid4)
    kind: InterventionKind
    target_item_id: Optional[str] = None
    reason: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)


class InterventionPolicy:
    """Pure decision function for executive layer interventions."""

    @staticmethod
    def evaluate(
        workspace: ExecutiveWorkspace,
        baa_queue_depth: int = 0,
        held_count: int = 0,
        somatic_recommends_pause: bool = False,
        live_work_orders: Optional[List[Dict[str, Any]]] = None,
    ) -> InterventionDecision:
        """Decide what the executive layer should do next."""
        live_orders = live_work_orders or []

        # 1. Surface blocked work orders
        for order in live_orders:
            if order.get("stage") == "needs_clarification":
                return InterventionDecision(
                    kind=InterventionKind.SURFACE_CLARIFICATION,
                    target_item_id=order.get("task_id"),
                    reason="Live work order needs clarification",
                )
            if order.get("stage") == "needs_approval":
                return InterventionDecision(
                    kind=InterventionKind.SURFACE_APPROVAL,
                    target_item_id=order.get("task_id"),
                    reason="Live work order needs approval",
                )

        focus = workspace.focus
        if focus is None:
            return InterventionDecision(
                kind=InterventionKind.NO_INTERVENTION,
                reason="Workspace is empty",
            )

        # 2. Verify completed but unverified work
        if focus.meta.get("verified") is False:
            return InterventionDecision(
                kind=InterventionKind.VERIFY_COMPLETED_WORK,
                target_item_id=str(focus.item_id),
                reason="Focus item is completed but unverified",
            )

        # 3. Reclaim stale personal items to foreground
        if focus.affinity.value == "personal" and focus.meta.get("stale") is True:
            return InterventionDecision(
                kind=InterventionKind.RECLAIM_TO_FOREGROUND,
                target_item_id=str(focus.item_id),
                reason="Stale personal item requires foreground attention",
            )

        # 4. Launch background work if conditions are met
        if focus.execution_mode == ExecutionMode.BACKGROUND_AGENT:
            if baa_queue_depth < 5 and not somatic_recommends_pause:
                return InterventionDecision(
                    kind=InterventionKind.LAUNCH_BACKGROUND,
                    target_item_id=str(focus.item_id),
                    reason="Background agent execution is appropriate and capacity exists",
                )

        # 5. Retire or defer low-score / paused focus
        if focus.total_score < 0.2 or somatic_recommends_pause:
            return InterventionDecision(
                kind=InterventionKind.RETIRE_OR_DEFER_FOCUS,
                target_item_id=str(focus.item_id),
                reason="Focus score too low or somatic state recommends pause",
            )

        return InterventionDecision(
            kind=InterventionKind.NO_INTERVENTION,
            target_item_id=str(focus.item_id),
            reason="No intervention required at this time",
        )
