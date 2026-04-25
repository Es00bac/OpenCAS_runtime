"""ApprovalLedger wraps the store and tracer for SelfApprovalLadder integration."""

from __future__ import annotations

from typing import Optional

from opencas.autonomy.models import ActionRequest, ApprovalDecision
from opencas.telemetry import EventKind, Tracer

from .models import ApprovalLedgerEntry
from .store import ApprovalLedgerStore


class ApprovalLedger:
    """Wraps ApprovalLedgerStore with tracing and query helpers."""

    def __init__(
        self,
        store: ApprovalLedgerStore,
        tracer: Optional[Tracer] = None,
    ) -> None:
        self.store = store
        self.tracer = tracer

    async def record(
        self,
        decision: ApprovalDecision,
        request: ActionRequest,
        score: float,
        somatic_state: Optional[str] = None,
    ) -> ApprovalLedgerEntry:
        """Record a decision to the ledger."""
        entry = ApprovalLedgerEntry(
            decision_id=decision.decision_id,
            action_id=request.action_id,
            level=decision.level.value,
            score=score,
            reasoning=decision.reasoning,
            tool_name=request.tool_name,
            tier=request.tier,
            somatic_state=somatic_state,
        )
        await self.store.save(entry)
        if self.tracer:
            self.tracer.log(
                EventKind.SELF_APPROVAL,
                "ApprovalLedger: recorded",
                {
                    "entry_id": str(entry.entry_id),
                    "decision_id": str(entry.decision_id),
                    "action_id": str(entry.action_id),
                    "level": entry.level,
                    "tier": entry.tier.value,
                },
            )
        return entry

    async def query_stats(self, window_days: int = 7) -> dict:
        return await self.store.query_stats(window_days)
