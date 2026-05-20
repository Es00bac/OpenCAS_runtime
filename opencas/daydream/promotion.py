"""Route daydream possibility signals through existing OpenCAS systems."""

from __future__ import annotations

import json
from typing import Any

from opencas.autonomy import WorkObject, WorkStage
from opencas.thread_registry import BeadSourceKind, ThreadStatus

from .self_workspace import SelfWorkspaceService
from .signal_builder import DaydreamSignalBuilder
from .signal_store import DaydreamSignalStore
from .signals import PossibilitySignal, PossibilitySignalRoute, SelfWorkReceipt


class DaydreamPromotionService:
    """Promote possibility signals without becoming a second executor."""

    def __init__(
        self,
        *,
        signal_store: DaydreamSignalStore,
        self_workspace: SelfWorkspaceService,
        signal_builder: DaydreamSignalBuilder | None = None,
        thread_registry_service: Any | None = None,
        creative: Any | None = None,
        initiative_contact: Any | None = None,
    ) -> None:
        self.signal_store = signal_store
        self.self_workspace = self_workspace
        self.signal_builder = signal_builder or DaydreamSignalBuilder()
        self.thread_registry_service = thread_registry_service
        self.creative = creative
        self.initiative_contact = initiative_contact

    async def route_reflection(self, reflection: Any) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for signal in self.signal_builder.build_from_reflection(reflection):
            await self.signal_store.save_signal(signal)
            results.append(await self.route_signal(signal))
        return results

    async def route_signal(self, signal: PossibilitySignal) -> dict[str, Any]:
        try:
            route = signal.suggested_route
            if route == PossibilitySignalRoute.INCUBATE:
                return await self._record(signal, status="incubated", reason=signal.route_reason or "incubated")
            if route == PossibilitySignalRoute.DISCARD:
                return await self._record(signal, status="discarded", reason=signal.route_reason or "discarded")
            if route == PossibilitySignalRoute.THREAD:
                return await self._route_thread(signal)
            if route == PossibilitySignalRoute.SELF_NOTE:
                receipt = await self.self_workspace.write_note(signal, reason=signal.route_reason)
                return await self._record_receipt(signal, receipt)
            if route == PossibilitySignalRoute.SELF_EXPERIMENT:
                receipt = await self.self_workspace.write_experiment(signal, reason=signal.route_reason)
                return await self._record_receipt(signal, receipt)
            if route == PossibilitySignalRoute.SELF_PROTOTYPE:
                receipt = await self.self_workspace.write_prototype(signal, reason=signal.route_reason)
                return await self._record_receipt(signal, receipt)
            if route == PossibilitySignalRoute.COMPOST:
                receipt = await self.self_workspace.compost(signal, reason=signal.route_reason)
                return await self._record_receipt(signal, receipt)
            if route == PossibilitySignalRoute.WORK_CANDIDATE:
                return await self._route_work_candidate(signal)
            if route == PossibilitySignalRoute.ASK_USER:
                return await self._route_ask_user(signal)
            if route == PossibilitySignalRoute.RESEARCH:
                receipt = await self.self_workspace.write_research(signal, reason=signal.route_reason)
                return await self._record_receipt(signal, receipt)
            return await self._record(signal, status="held", reason=f"unsupported route {route.value}")
        except Exception as exc:
            return await self._record(
                signal,
                route=PossibilitySignalRoute.ROUTE_FAILED,
                status="route_failed",
                reason=str(exc),
            )

    async def backfill_signal_thread_beads(self, *, limit: int = 50) -> dict[str, Any]:
        if self.thread_registry_service is None:
            return {"available": False, "scanned": 0, "recorded": 0, "skipped": 0}
        signals = await self.signal_store.list_recent(limit=max(1, min(200, int(limit))))
        recorded: list[str] = []
        skipped = 0
        for signal in signals:
            if signal.suggested_route == PossibilitySignalRoute.DISCARD:
                skipped += 1
                continue
            if signal.route_status in {"pending", "discarded"}:
                skipped += 1
                continue
            bead_id = await self._record_signal_thread_bead(
                signal,
                route=signal.suggested_route,
                status=signal.route_status,
                reason=signal.route_reason or "backfilled routed daydream signal",
                artifact_paths=signal.artifact_paths,
                raw={"backfill": True},
            )
            if bead_id:
                recorded.append(bead_id)
            else:
                skipped += 1
        return {
            "available": True,
            "scanned": len(signals),
            "recorded": len(recorded),
            "skipped": skipped,
            "bead_ids": recorded,
        }

    async def _route_thread(self, signal: PossibilitySignal) -> dict[str, Any]:
        if self.thread_registry_service is None:
            return await self._record(signal, status="held", reason="thread registry unavailable")
        anchor = await self.thread_registry_service.ensure_thread_anchor(
            title=signal.summary[:96],
            kind="daydream_signal",
            status=ThreadStatus.PERIPHERAL,
        )
        bead = await self.thread_registry_service.create_candidate_bead(
            thread_anchor_id=anchor.anchor_id,
            title=signal.summary[:96],
            summary=signal.practical_branch or signal.summary,
            source_kind=BeadSourceKind.DAYDREAM_SIGNAL,
            source_ref=f"daydream_signal:{signal.signal_id}",
            content=self._signal_content(signal),
            user_commissioned=False,
        )
        return await self._record(
            signal,
            status="routed",
            reason=signal.route_reason or "recorded as peripheral thread bead",
            raw={
                "bead_id": getattr(bead, "bead_id", ""),
                "thread_bead_id": getattr(bead, "bead_id", ""),
            },
            record_signal_bead=False,
        )

    async def _route_work_candidate(self, signal: PossibilitySignal) -> dict[str, Any]:
        if self.creative is None or not hasattr(self.creative, "add"):
            return await self._record(signal, status="held", reason="creative ladder unavailable")
        work = WorkObject(
            content=signal.summary,
            stage=WorkStage.SPARK,
            promotion_score=max(signal.usefulness, signal.confidence),
            meta={
                "origin": "daydream_signal",
                "signal_id": signal.signal_id,
                "source_reflection_id": signal.source_reflection_id,
            },
        )
        self.creative.add(work)
        return await self._record(
            signal,
            status="routed",
            reason=signal.route_reason or "added to creative ladder",
            raw={"work_id": str(work.work_id)},
        )

    async def _route_ask_user(self, signal: PossibilitySignal) -> dict[str, Any]:
        if self.initiative_contact is None or not hasattr(self.initiative_contact, "consider_candidate"):
            return await self._record(signal, status="held", reason="initiative contact unavailable")
        candidate = {
            "source_id": signal.signal_id,
            "source_kind": "daydream_signal",
            "label": signal.summary[:160],
            "summary": signal.summary,
            "intensity": max(signal.usefulness, signal.novelty, signal.confidence),
            "reason": signal.route_reason or "signal asks for owner judgment",
            "tags": ["daydream_signal", signal.suggested_route.value],
            "raw": {
                "signal_id": signal.signal_id,
                "source_reflection_id": signal.source_reflection_id,
                "imaginative_branch": signal.imaginative_branch,
                "practical_branch": signal.practical_branch,
                "bridge": signal.bridge,
                "contact_posture": signal.contact_posture.value,
                "artifact_paths": signal.artifact_paths,
            },
        }
        result = await self.initiative_contact.consider_candidate(candidate)
        status = str(result.get("status") or "held") if isinstance(result, dict) else "held"
        if status == "held":
            status = "deferred"
        return await self._record(
            signal,
            status=status,
            reason=(
                str(result.get("reason") or signal.route_reason or "contact considered")
                if isinstance(result, dict)
                else "contact considered"
            ),
            raw={"contact_result": result, "candidate": candidate},
        )

    async def _record_receipt(
        self,
        signal: PossibilitySignal,
        receipt: SelfWorkReceipt,
    ) -> dict[str, Any]:
        await self.signal_store.save_receipt(receipt)
        return await self._record(
            signal,
            status="routed",
            reason=receipt.outcome,
            artifact_paths=receipt.artifact_paths,
            raw={"receipt_id": receipt.receipt_id},
        )

    async def _record(
        self,
        signal: PossibilitySignal,
        *,
        status: str,
        reason: str,
        route: PossibilitySignalRoute | None = None,
        artifact_paths: list[str] | None = None,
        raw: dict[str, Any] | None = None,
        record_signal_bead: bool = True,
    ) -> dict[str, Any]:
        route_value = route or signal.suggested_route
        raw_payload = dict(raw or {})
        if record_signal_bead:
            bead_id = await self._record_signal_thread_bead(
                signal,
                route=route_value,
                status=status,
                reason=reason,
                artifact_paths=artifact_paths or [],
                raw=raw_payload,
            )
            if bead_id:
                raw_payload.setdefault("thread_bead_id", bead_id)
        event = await self.signal_store.record_route(
            signal.signal_id,
            route=route_value,
            status=status,
            reason=reason,
            artifact_paths=artifact_paths or [],
            raw=raw_payload or None,
        )
        result = {
            "status": status,
            "reason": reason,
            "signal_id": signal.signal_id,
            "route": route_value.value,
            "event": event,
        }
        if raw_payload.get("thread_bead_id"):
            result["thread_bead_id"] = raw_payload["thread_bead_id"]
        return result

    async def _record_signal_thread_bead(
        self,
        signal: PossibilitySignal,
        *,
        route: PossibilitySignalRoute,
        status: str,
        reason: str,
        artifact_paths: list[str],
        raw: dict[str, Any],
    ) -> str | None:
        if self.thread_registry_service is None:
            return None
        if route == PossibilitySignalRoute.DISCARD:
            return None
        try:
            anchor = await self.thread_registry_service.ensure_thread_anchor(
                title=signal.summary[:96],
                kind="daydream_signal",
                status=ThreadStatus.PERIPHERAL,
            )
            bead = await self.thread_registry_service.create_candidate_bead(
                thread_anchor_id=anchor.anchor_id,
                title=signal.summary[:96],
                summary=self._signal_bead_summary(signal, route=route, status=status, reason=reason),
                source_kind=BeadSourceKind.DAYDREAM_SIGNAL,
                source_ref=f"daydream_signal:{signal.signal_id}",
                content=self._signal_content(
                    signal,
                    route=route,
                    status=status,
                    reason=reason,
                    artifact_paths=artifact_paths,
                    raw=raw,
                ),
                user_commissioned=False,
            )
            return str(getattr(bead, "bead_id", "") or "") or None
        except Exception:
            return None

    @staticmethod
    def _signal_content(
        signal: PossibilitySignal,
        *,
        route: PossibilitySignalRoute | None = None,
        status: str | None = None,
        reason: str | None = None,
        artifact_paths: list[str] | None = None,
        raw: dict[str, Any] | None = None,
    ) -> str:
        payload = signal.model_dump(mode="json")
        if route is not None:
            payload["routed_route"] = route.value
        if status is not None:
            payload["route_status"] = status
        if reason is not None:
            payload["route_reason"] = reason
        if artifact_paths:
            payload["artifact_paths"] = artifact_paths
        if raw:
            payload["route_raw"] = raw
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def _signal_bead_summary(
        signal: PossibilitySignal,
        *,
        route: PossibilitySignalRoute,
        status: str,
        reason: str,
    ) -> str:
        branch = signal.practical_branch or signal.summary
        return " ".join(
            part
            for part in [
                branch,
                f"route={route.value}",
                f"status={status}",
                reason,
            ]
            if str(part or "").strip()
        )
