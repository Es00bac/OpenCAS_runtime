"""Follow-up surfacing for repeated record-only maintenance loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .drift import (
    annotate_drift_drain_effect,
    outcome_is_pending_drift_validation,
    split_drift_observations,
)
from .models import (
    MaintenanceAction,
    MaintenanceActionType,
    MaintenanceOutcome,
    WellbeingRecommendation,
)
from .store import WellbeingStore


@dataclass(frozen=True)
class RecordOnlyMaintenanceLoop:
    """A repeated maintenance receipt whose effect is still unmeasured."""

    signature: str
    action_type: str
    outcome: str
    repeat_count: int
    outcome_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RecordOnlyFollowupSummary:
    """Operator-facing summary of record-only maintenance loops."""

    stuck_loop_count: int = 0
    repeat_count: int = 0
    next_followup_action: dict | None = None
    linked_followup_ids: list[str] = field(default_factory=list)
    linked_followup_refs: list[dict] = field(default_factory=list)

    def model_dump(self) -> dict:
        return {
            "stuck_loop_count": self.stuck_loop_count,
            "repeat_count": self.repeat_count,
            "next_followup_action": self.next_followup_action,
            "linked_followup_ids": list(self.linked_followup_ids),
            "linked_followup_refs": list(self.linked_followup_refs),
        }


class RecordOnlyMaintenanceFollowupService:
    """Promote repeated record-only maintenance outcomes into review work."""

    def __init__(
        self,
        store: WellbeingStore,
        *,
        self_inspection_store: Any | None = None,
        repeat_threshold: int = 3,
        scan_limit: int = 50,
    ) -> None:
        self.store = store
        self.self_inspection_store = self_inspection_store
        self.repeat_threshold = max(2, int(repeat_threshold))
        self.scan_limit = max(self.repeat_threshold, min(200, int(scan_limit)))

    async def evaluate(self) -> RecordOnlyFollowupSummary:
        outcomes = await self.store.list_maintenance_outcomes(limit=self.scan_limit)
        estimated_followup = await self._verify_estimated_drift_drain(outcomes)
        loops = self._find_stuck_loops(outcomes)
        if not loops:
            return estimated_followup or RecordOnlyFollowupSummary()

        primary = max(loops, key=lambda loop: loop.repeat_count)
        recommendation = await self._ensure_followup_recommendation(primary)
        recommendation_id = str(recommendation.recommendation_id)
        next_action = {
            "type": MaintenanceActionType.SELF_MODIFICATION_PROPOSAL.value,
            "reason": (
                f"{primary.action_type} has produced {primary.repeat_count} repeated "
                "record-only follow-up receipts; create a reviewable change proposal "
                "or operator review item before treating it as repaired."
            ),
            "source_action_type": primary.action_type,
            "source_outcome": primary.outcome,
            "repeat_count": primary.repeat_count,
            "effect_basis": "record_only",
            "effect_remains_unmeasured": True,
            "recommendation_id": recommendation_id,
        }
        proposal_path = await self._proposal_path_for_recommendation(recommendation_id)
        if proposal_path:
            next_action["proposal_artifact_path"] = proposal_path
        return RecordOnlyFollowupSummary(
            stuck_loop_count=len(loops),
            repeat_count=primary.repeat_count,
            next_followup_action=next_action,
            linked_followup_ids=[recommendation_id],
            linked_followup_refs=[
                {
                    "kind": "wellbeing_recommendation",
                    "id": recommendation_id,
                    "source_signature": primary.signature,
                }
            ],
        )

    async def _verify_estimated_drift_drain(
        self,
        outcomes: list[MaintenanceOutcome],
    ) -> RecordOnlyFollowupSummary | None:
        if self.self_inspection_store is None:
            return None
        list_recent = getattr(self.self_inspection_store, "list_recent", None)
        if not callable(list_recent):
            return None

        pending_by_action: dict[str, list[MaintenanceOutcome]] = {}
        for outcome in outcomes:
            if outcome_is_pending_drift_validation(outcome):
                pending_by_action.setdefault(outcome.action_type, []).append(outcome)

        for action_type, pending in pending_by_action.items():
            pending = sorted(pending, key=lambda item: item.created_at)
            if len(pending) < 3:
                continue
            records = list(await list_recent(limit=200))
            before_count, after_count = _drift_counts_around(records, pending[0].created_at)
            if before_count > 0 and after_count <= before_count * 0.9:
                target = pending[-1]
                decrease_fraction = round((after_count - before_count) / before_count, 3)
                target.meta = {
                    **(target.meta or {}),
                    "effect_basis": "measured",
                    "effect_direction": "measured_lower",
                    "risk_delta": max(-1.0, min(0.0, decrease_fraction)),
                    "drift_verification": {
                        "before_observation_count": before_count,
                        "after_observation_count": after_count,
                        "decrease_fraction": decrease_fraction,
                    },
                }
                await annotate_drift_drain_effect(self.self_inspection_store, target)
                await self.store.save_maintenance_outcome(target)
                return None

            if len(pending) >= 5:
                recommendation = await self._ensure_estimated_drift_recommendation(
                    action_type=action_type,
                    outcomes=pending,
                    before_count=before_count,
                    after_count=after_count,
                )
                recommendation_id = str(recommendation.recommendation_id)
                next_action = {
                    "type": MaintenanceActionType.SELF_MODIFICATION_PROPOSAL.value,
                    "reason": (
                        f"{action_type} has {len(pending)} estimated drift-lowering "
                        "maintenance receipts without a measured drop in the drift "
                        "observation rate; create a reviewable proposal before "
                        "treating the action as effective."
                    ),
                    "source_action_type": action_type,
                    "source_outcome": "estimated_drift_decrease_unverified",
                    "repeat_count": len(pending),
                    "effect_basis": "estimated",
                    "effect_remains_unmeasured": True,
                    "recommendation_id": recommendation_id,
                }
                proposal_path = await self._proposal_path_for_recommendation(
                    recommendation_id
                )
                if proposal_path:
                    next_action["proposal_artifact_path"] = proposal_path
                return RecordOnlyFollowupSummary(
                    stuck_loop_count=1,
                    repeat_count=len(pending),
                    next_followup_action=next_action,
                    linked_followup_ids=[recommendation_id],
                    linked_followup_refs=[
                        {
                            "kind": "wellbeing_recommendation",
                            "id": recommendation_id,
                            "source_signature": f"{action_type}|estimated_drift_drain",
                        }
                    ],
                )
        return None

    def _find_stuck_loops(
        self,
        outcomes: list[MaintenanceOutcome],
    ) -> list[RecordOnlyMaintenanceLoop]:
        grouped: dict[str, list[MaintenanceOutcome]] = {}
        for outcome in outcomes:
            meta = outcome.meta or {}
            if meta.get("effect_basis") != "record_only":
                continue
            if meta.get("followup_required") is not True:
                continue
            signature = self._signature(outcome)
            grouped.setdefault(signature, []).append(outcome)

        loops: list[RecordOnlyMaintenanceLoop] = []
        for signature, items in grouped.items():
            if len(items) < self.repeat_threshold:
                continue
            newest_first = sorted(items, key=lambda item: item.created_at, reverse=True)
            first = newest_first[0]
            loops.append(
                RecordOnlyMaintenanceLoop(
                    signature=signature,
                    action_type=first.action_type,
                    outcome=first.outcome,
                    repeat_count=len(items),
                    outcome_ids=[str(item.outcome_id) for item in newest_first],
                )
            )
        return loops

    async def _ensure_followup_recommendation(
        self,
        loop: RecordOnlyMaintenanceLoop,
    ) -> WellbeingRecommendation:
        existing = await self.store.list_recommendations(status="proposed", limit=50)
        for recommendation in existing:
            meta = recommendation.meta or {}
            if (
                meta.get("followup_kind") == "repeated_record_only_maintenance"
                and meta.get("source_signature") == loop.signature
            ):
                if int(meta.get("repeat_count") or 0) < loop.repeat_count:
                    recommendation.meta = {
                        **meta,
                        "source_outcome_ids": loop.outcome_ids,
                        "repeat_count": loop.repeat_count,
                    }
                    for action in recommendation.actions:
                        action.meta = {
                            **(action.meta or {}),
                            "repeat_count": loop.repeat_count,
                        }
                    await self.store.save_recommendation(recommendation)
                return recommendation

        recommendation = WellbeingRecommendation(
            reason=(
                f"Repeated {loop.action_type} maintenance outcomes are still record-only; "
                "the effect remains unmeasured and needs a reviewable follow-up action."
            ),
            actions=[
                MaintenanceAction(
                    action_type=MaintenanceActionType.SELF_MODIFICATION_PROPOSAL,
                    reason=(
                        "Create a reviewable self-maintenance change proposal or operator "
                        "review item for the repeated unmeasured maintenance loop."
                    ),
                    priority=0.82,
                    meta={
                        "source_signature": loop.signature,
                        "repeat_count": loop.repeat_count,
                        "effect_basis": "record_only",
                        "effect_remains_unmeasured": True,
                    },
                )
            ],
            status="proposed",
            meta={
                "followup_kind": "repeated_record_only_maintenance",
                "source_signature": loop.signature,
                "source_action_type": loop.action_type,
                "source_outcome": loop.outcome,
                "source_outcome_ids": loop.outcome_ids,
                "repeat_count": loop.repeat_count,
                "threshold": self.repeat_threshold,
                "effect_basis": "record_only",
                "effect_remains_unmeasured": True,
                "hot_applied": False,
            },
        )
        await self.store.save_recommendation(recommendation)
        return recommendation

    async def _ensure_estimated_drift_recommendation(
        self,
        *,
        action_type: str,
        outcomes: list[MaintenanceOutcome],
        before_count: int,
        after_count: int,
    ) -> WellbeingRecommendation:
        signature = f"{action_type}|estimated_drift_drain"
        existing = await self.store.list_recommendations(status="proposed", limit=50)
        for recommendation in existing:
            meta = recommendation.meta or {}
            if (
                meta.get("followup_kind") == "estimated_drift_drain_unverified"
                and meta.get("source_signature") == signature
            ):
                return recommendation

        outcome_ids = [str(outcome.outcome_id) for outcome in outcomes]
        recommendation = WellbeingRecommendation(
            reason=(
                f"{action_type} maintenance outcomes claim estimated drift reduction, "
                "but recent drift observations did not decrease enough to validate it."
            ),
            actions=[
                MaintenanceAction(
                    action_type=MaintenanceActionType.SELF_MODIFICATION_PROPOSAL,
                    reason=(
                        "Create a reviewable proposal for a different drift-maintenance "
                        "strategy or a stricter measurement gate."
                    ),
                    priority=0.84,
                    meta={
                        "source_signature": signature,
                        "repeat_count": len(outcomes),
                        "effect_basis": "estimated",
                        "effect_remains_unmeasured": True,
                    },
                )
            ],
            status="proposed",
            meta={
                "followup_kind": "estimated_drift_drain_unverified",
                "source_signature": signature,
                "source_action_type": action_type,
                "source_outcome_ids": outcome_ids,
                "repeat_count": len(outcomes),
                "threshold": 5,
                "before_observation_count": before_count,
                "after_observation_count": after_count,
                "effect_basis": "estimated",
                "effect_remains_unmeasured": True,
                "hot_applied": False,
            },
        )
        await self.store.save_recommendation(recommendation)
        return recommendation

    async def _proposal_path_for_recommendation(self, recommendation_id: str) -> str:
        proposals = await self.store.list_self_modification_proposals(limit=100)
        for proposal in proposals:
            if (proposal.meta or {}).get("recommendation_id") == recommendation_id:
                return proposal.artifact_path
        return ""

    @staticmethod
    def _signature(outcome: MaintenanceOutcome) -> str:
        effect_direction = str((outcome.meta or {}).get("effect_direction") or "")
        return "|".join(
            [
                outcome.action_type,
                outcome.outcome,
                "record_only",
                effect_direction,
            ]
        )


def _drift_counts_around(
    records: list[Any],
    split_at: datetime,
) -> tuple[int, int]:
    if split_at.tzinfo is None:
        split_at = split_at.replace(tzinfo=timezone.utc)
    before = 0
    after = 0
    for record in records:
        created_at = getattr(record, "created_at", None)
        if created_at is None:
            continue
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        open_observations, _ = split_drift_observations([record])
        if created_at < split_at:
            before += len(open_observations)
        else:
            after += len(open_observations)
    return before, after
