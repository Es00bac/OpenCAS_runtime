from __future__ import annotations

from typing import Any

from opencas.recovery.models import ContinuityPacket, RecoverySummary


class AutonomousRecoveryCoordinator:
    def __init__(
        self,
        *,
        scanner: Any,
        classifier: Any,
        continuity_builder: Any,
        planner: Any,
        executor: Any,
        ledger: Any,
        tracer: Any = None,
        activity_sink: Any = None,
    ) -> None:
        self.scanner = scanner
        self.classifier = classifier
        self.continuity_builder = continuity_builder
        self.planner = planner
        self.executor = executor
        self.ledger = ledger
        self.tracer = tracer
        self.activity_sink = activity_sink

    async def run_once(self, *, limit_per_store: int = 100, max_actions: int = 5) -> RecoverySummary:
        candidates = await self.scanner.scan(limit_per_store=limit_per_store)
        summary = RecoverySummary(scanned=len(candidates))
        actions = 0
        for candidate in candidates:
            if actions >= max_actions:
                break
            if await self.ledger.should_skip_candidate(candidate.candidate_id):
                summary = _increment(summary, "skipped_by_ledger")
                continue
            classification = self.classifier.classify(candidate)
            summary = _increment(summary, classification.category)
            packet: ContinuityPacket | None = None
            if classification.category == "needs_artifact_continuity" and self.continuity_builder is not None:
                packet = await self.continuity_builder.build(candidate)
            plan = self.planner.plan(candidate, classification, continuity_packet=packet)
            decision = await self.executor.execute(plan)
            await self.ledger.record_decision(decision)
            if decision.result == "submitted":
                summary = _increment(summary, "submitted")
                actions += 1
            elif decision.result in {"blocked", "no_executor_available", "no_progress", "scheduled"}:
                actions += 1
        self._trace(summary)
        await self._post_activity(summary)
        return summary

    async def _post_activity(self, summary: RecoverySummary) -> None:
        if self.activity_sink is None:
            return
        post = getattr(self.activity_sink, "post_activity", None)
        if not callable(post):
            return
        await post(
            agent="AutonomousRecoveryCoordinator",
            summary=f"Recovery pass scanned {summary.scanned} candidates and submitted {summary.submitted} actions.",
            status="completed",
            files_changed=[],
            details=summary.__dict__,
        )

    def _trace(self, summary: RecoverySummary) -> None:
        if self.tracer is None:
            return
        trace = getattr(self.tracer, "trace", None) or getattr(self.tracer, "record", None)
        if callable(trace):
            trace("autonomous_recovery_pass", summary.__dict__)


def _increment(summary: RecoverySummary, field_name: str) -> RecoverySummary:
    if not hasattr(summary, field_name):
        return summary
    values = dict(summary.__dict__)
    values[field_name] += 1
    return RecoverySummary(**values)
