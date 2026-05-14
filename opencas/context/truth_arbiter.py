"""Versioned current-truth snapshots for dual-context coordination."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class TruthSnapshot(BaseModel):
    """Compact, source-labeled view of the runtime state that can authorize work."""

    snapshot_id: str
    epoch: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""
    source_hash: str
    executive: Dict[str, Any] = Field(default_factory=dict)
    baa: Dict[str, Any] = Field(default_factory=dict)
    schedules: Dict[str, Any] = Field(default_factory=dict)
    commitments: Dict[str, Any] = Field(default_factory=dict)
    receipts: Dict[str, Any] = Field(default_factory=dict)
    runtime_activity: Dict[str, Any] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)


class TruthArbiter:
    """Issue stable truth epochs from authoritative runtime state."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._epoch = 0
        self._last_source_hash: str | None = None
        self._lock = asyncio.Lock()

    @property
    def epoch(self) -> int:
        return self._epoch

    async def issue_snapshot(self, *, reason: str = "") -> TruthSnapshot:
        executive = self._executive_snapshot()
        baa = self._baa_snapshot()
        schedules = await self._schedule_snapshot()
        commitments = await self._commitment_snapshot()
        receipts = await self._receipt_snapshot()
        runtime_activity = self._runtime_activity_snapshot()
        sources = [
            "executive",
            "baa",
            "schedules",
            "commitments",
            "receipts",
            "runtime_activity",
        ]
        source_payload = {
            "executive": executive,
            "baa": baa,
            "schedules": schedules,
            "commitments": commitments,
            "receipts": receipts,
            "runtime_activity": self._strip_volatile_fields(runtime_activity),
        }
        source_hash = self._stable_hash(source_payload)
        async with self._lock:
            if source_hash != self._last_source_hash:
                self._epoch += 1
                self._last_source_hash = source_hash
            epoch = self._epoch
        snapshot_id = f"truth:{epoch}:{source_hash[:16]}"
        return TruthSnapshot(
            snapshot_id=snapshot_id,
            epoch=epoch,
            reason=reason,
            source_hash=source_hash,
            executive=executive,
            baa=baa,
            schedules=schedules,
            commitments=commitments,
            receipts=receipts,
            runtime_activity=runtime_activity,
            sources=sources,
        )

    def _executive_snapshot(self) -> Dict[str, Any]:
        executive = getattr(self.runtime, "executive", None)
        if executive is None:
            executive = getattr(getattr(self.runtime, "ctx", None), "executive", None)
        if executive is None:
            return {"available": False}
        raw = executive.snapshot() if callable(getattr(executive, "snapshot", None)) else {}
        if not isinstance(raw, dict):
            raw = {}
        payload = dict(raw)
        intention_source = payload.get("intention_source")
        has_foreground = bool(payload.get("active_goals")) or int(payload.get("queue_size") or 0) > 0
        if intention_source == "active_work" and payload.get("intention") and not has_foreground:
            payload["intention"] = None
            payload["intention_source"] = "stale_active_work"
        payload.setdefault("available", True)
        return payload

    def _baa_snapshot(self) -> Dict[str, Any]:
        baa = getattr(self.runtime, "baa", None)
        if baa is None:
            return {"available": False, "live_task_ids": [], "live_task_count": 0}
        live_tasks = getattr(baa, "_live_tasks", {}) or {}
        live_task_ids = sorted(str(task_id) for task_id in live_tasks.keys())
        lane_snapshot = {}
        if callable(getattr(baa, "lane_snapshot", None)):
            try:
                lane_snapshot = baa.lane_snapshot()
            except Exception:
                lane_snapshot = {}
        return {
            "available": True,
            "live_task_ids": live_task_ids,
            "live_task_count": len(live_task_ids),
            "queue_size": int(getattr(baa, "queue_size", 0) or 0),
            "held_size": int(getattr(baa, "held_size", 0) or 0),
            "active_count": int(getattr(baa, "active_count", 0) or 0),
            "lanes": lane_snapshot,
        }

    async def _schedule_snapshot(self) -> Dict[str, Any]:
        service = getattr(self.runtime, "schedule_service", None)
        if service is None:
            service = getattr(getattr(self.runtime, "ctx", None), "schedule_service", None)
        if service is None:
            return {"available": False}
        agenda = getattr(service, "temporal_agenda", None)
        if callable(agenda):
            try:
                result = agenda()
                if inspect.isawaitable(result):
                    result = await result
                if isinstance(result, dict):
                    return self._compact_mapping(result)
            except Exception as exc:
                return {"available": True, "error": str(exc)}
        return {"available": True}

    async def _commitment_snapshot(self) -> Dict[str, Any]:
        store = getattr(self.runtime, "commitment_store", None)
        if store is None:
            store = getattr(getattr(self.runtime, "ctx", None), "commitment_store", None)
        if store is None:
            return {"available": False}
        count_by_status = getattr(store, "count_by_status", None)
        if not callable(count_by_status):
            return {"available": True}
        counts: Dict[str, int] = {}
        for status in ("active", "blocked", "completed", "abandoned"):
            try:
                result = count_by_status(status)
                if inspect.isawaitable(result):
                    result = await result
                counts[status] = int(result or 0)
            except Exception:
                continue
        return {"available": True, "status_counts": counts}

    async def _receipt_snapshot(self) -> Dict[str, Any]:
        store = getattr(getattr(self.runtime, "ctx", None), "receipt_store", None)
        if store is None:
            return {"available": False}
        list_recent = getattr(store, "list_recent", None)
        if not callable(list_recent):
            return {"available": True}
        try:
            result = list_recent(limit=5)
            if inspect.isawaitable(result):
                result = await result
            receipt_ids = [str(getattr(item, "receipt_id", "")) for item in (result or [])]
            return {"available": True, "recent_receipt_ids": [item for item in receipt_ids if item]}
        except Exception as exc:
            return {"available": True, "error": str(exc)}

    def _runtime_activity_snapshot(self) -> Dict[str, Any]:
        return {
            "activity": getattr(self.runtime, "_activity", None),
            "readiness": (
                getattr(getattr(self.runtime, "readiness", None), "snapshot", lambda: None)()
                if getattr(self.runtime, "readiness", None) is not None
                else None
            ),
        }

    @staticmethod
    def _compact_mapping(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Keep large agenda-like payloads suitable for stable hashing and task metadata."""

        compact = dict(payload)
        for key in ("upcoming", "recent_runs", "items", "runs"):
            value = compact.get(key)
            if isinstance(value, list):
                compact[key] = value[:10]
        return TruthArbiter._strip_volatile_fields(compact)

    @staticmethod
    def _strip_volatile_fields(value: Any) -> Any:
        """Remove clock-derived fields that would churn truth epochs without state changes."""

        volatile_keys = {"now", "horizon_end", "seconds_until", "activity"}
        if isinstance(value, dict):
            return {
                key: TruthArbiter._strip_volatile_fields(item)
                for key, item in value.items()
                if key not in volatile_keys
            }
        if isinstance(value, list):
            return [TruthArbiter._strip_volatile_fields(item) for item in value]
        return value

    @staticmethod
    def _stable_hash(payload: Dict[str, Any]) -> str:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
