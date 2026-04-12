"""Daydream API routes for reflections, conflicts, and promotion lineage."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _truncate(value: Optional[str], limit: int = 160) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _reflection_to_dict(reflection: Any) -> Dict[str, Any]:
    return {
        "reflection_id": str(getattr(reflection, "reflection_id", "")),
        "created_at": _iso(getattr(reflection, "created_at", None)),
        "spark_content": getattr(reflection, "spark_content", ""),
        "spark_preview": _truncate(getattr(reflection, "spark_content", ""), 120),
        "recollection": getattr(reflection, "recollection", ""),
        "interpretation": getattr(reflection, "interpretation", ""),
        "synthesis": getattr(reflection, "synthesis", ""),
        "open_question": getattr(reflection, "open_question", None),
        "changed_self_view": getattr(reflection, "changed_self_view", ""),
        "tension_hints": list(getattr(reflection, "tension_hints", []) or []),
        "alignment_score": float(getattr(reflection, "alignment_score", 0.0) or 0.0),
        "novelty_score": float(getattr(reflection, "novelty_score", 0.0) or 0.0),
        "keeper": bool(getattr(reflection, "keeper", False)),
    }


def _conflict_to_dict(conflict: Any) -> Dict[str, Any]:
    somatic = getattr(conflict, "somatic_context", None)
    return {
        "conflict_id": str(getattr(conflict, "conflict_id", "")),
        "created_at": _iso(getattr(conflict, "created_at", None)),
        "resolved_at": _iso(getattr(conflict, "resolved_at", None)),
        "kind": getattr(conflict, "kind", ""),
        "description": getattr(conflict, "description", ""),
        "source_daydream_id": getattr(conflict, "source_daydream_id", None),
        "occurrence_count": int(getattr(conflict, "occurrence_count", 0) or 0),
        "resolved": bool(getattr(conflict, "resolved", False)),
        "auto_resolved": bool(getattr(conflict, "auto_resolved", False)),
        "resolution_notes": getattr(conflict, "resolution_notes", ""),
        "somatic_context": (
            somatic.model_dump(mode="json") if somatic is not None and hasattr(somatic, "model_dump") else None
        ),
    }


def _work_to_dict(work: Any) -> Dict[str, Any]:
    meta = dict(getattr(work, "meta", {}) or {})
    return {
        "work_id": str(getattr(work, "work_id", "")),
        "created_at": _iso(getattr(work, "created_at", None)),
        "updated_at": _iso(getattr(work, "updated_at", None)),
        "stage": getattr(getattr(work, "stage", None), "value", getattr(work, "stage", None)),
        "content": getattr(work, "content", ""),
        "content_preview": _truncate(getattr(work, "content", ""), 120),
        "promotion_score": float(getattr(work, "promotion_score", 0.0) or 0.0),
        "project_id": getattr(work, "project_id", None),
        "commitment_id": getattr(work, "commitment_id", None),
        "portfolio_id": getattr(work, "portfolio_id", None),
        "blocked_by": list(getattr(work, "blocked_by", []) or []),
        "source_memory_ids": list(getattr(work, "source_memory_ids", []) or []),
        "meta": meta,
        "title": meta.get("title") or _truncate(getattr(work, "content", ""), 88),
    }


def _memory_to_dict(memory: Any) -> Dict[str, Any]:
    tags = list(getattr(memory, "tags", []) or [])
    return {
        "memory_id": str(getattr(memory, "memory_id", "")),
        "created_at": _iso(getattr(memory, "created_at", None)),
        "updated_at": _iso(getattr(memory, "updated_at", None)),
        "content": getattr(memory, "content", ""),
        "content_preview": _truncate(getattr(memory, "content", ""), 120),
        "tags": tags,
        "salience": float(getattr(memory, "salience", 0.0) or 0.0),
        "access_count": int(getattr(memory, "access_count", 0) or 0),
        "last_accessed": _iso(getattr(memory, "last_accessed", None)),
        "source_episode_ids": list(getattr(memory, "source_episode_ids", []) or []),
    }


async def _load_reflections(runtime: Any, *, limit: int, keeper_only: Optional[bool]) -> List[Dict[str, Any]]:
    store = getattr(getattr(runtime, "ctx", None), "daydream_store", None)
    if store is None or not hasattr(store, "list_recent"):
        return []
    reflections = await store.list_recent(limit=limit, keeper_only=keeper_only)
    return [_reflection_to_dict(item) for item in reflections]


async def _load_conflicts(runtime: Any, *, limit: int, resolved: Optional[bool]) -> List[Dict[str, Any]]:
    store = getattr(getattr(runtime, "ctx", None), "conflict_store", None)
    if store is None:
        registry = getattr(runtime, "conflict_registry", None)
        if registry is not None:
            store = getattr(registry, "store", None)
    if store is None:
        return []
    if hasattr(store, "list_conflicts"):
        conflicts = await store.list_conflicts(limit=limit, resolved=resolved)
    elif resolved in (None, False) and hasattr(store, "list_active_conflicts"):
        conflicts = await store.list_active_conflicts(limit=limit)
    else:
        conflicts = []
    return [_conflict_to_dict(item) for item in conflicts]


async def _load_work_promotions(runtime: Any, *, limit: int) -> List[Dict[str, Any]]:
    store = getattr(getattr(runtime, "ctx", None), "work_store", None)
    if store is None:
        creative = getattr(runtime, "creative", None)
        store = getattr(creative, "store", None)
    if store is None:
        return []
    if hasattr(store, "list_by_origin"):
        items = await store.list_by_origin("daydream", limit=limit)
    elif hasattr(store, "list_all"):
        items = [
            item
            for item in await store.list_all(limit=max(limit * 4, limit))
            if str((getattr(item, "meta", {}) or {}).get("origin", "")) == "daydream"
        ][:limit]
    else:
        items = []
    return [_work_to_dict(item) for item in items]


async def _load_keeper_memories(runtime: Any, *, limit: int) -> List[Dict[str, Any]]:
    memory = getattr(runtime, "memory", None) or getattr(getattr(runtime, "ctx", None), "memory", None)
    if memory is None:
        return []
    if hasattr(memory, "list_memories_by_tag"):
        items = await memory.list_memories_by_tag("daydream", limit=limit)
    elif hasattr(memory, "list_memories"):
        items = [
            item
            for item in await memory.list_memories(limit=max(limit * 4, limit))
            if "daydream" in list(getattr(item, "tags", []) or [])
        ][:limit]
    else:
        items = []
    return [_memory_to_dict(item) for item in items]


def _iter_persisted_daydream_runs(runtime: Any, *, window_days: int) -> List[Dict[str, Any]]:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    state_dir = getattr(config, "state_dir", None)
    if not state_dir:
        return []
    telemetry_dir = Path(state_dir) / "telemetry"
    if not telemetry_dir.exists():
        return []
    runs: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    days = [now - timedelta(days=offset) for offset in range(max(1, window_days))]
    for day in sorted(days):
        path = telemetry_dir / f"{day.date().isoformat()}.jsonl"
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            if str(event.get("message", "")) != "AgentScheduler: daydream_complete":
                continue
            payload = dict(event.get("payload", {}) or {})
            runs.append(
                {
                    "timestamp": _iso(event.get("timestamp")),
                    "reflections": int(payload.get("reflections", 0) or 0),
                    "keepers": int(payload.get("keepers", 0) or 0),
                    "memories_created": int(payload.get("daydream_memories_created", 0) or 0),
                    "promoted_work_count": len(payload.get("daydream_work_objects", []) or []),
                    "had_activity": bool((payload.get("reflections", 0) or 0) > 0 or (payload.get("keepers", 0) or 0) > 0),
                }
            )
    runs.sort(key=lambda item: item.get("timestamp") or "")
    return runs


def _load_recent_runs(runtime: Any, *, limit: int, window_days: int) -> List[Dict[str, Any]]:
    persisted_runs = _iter_persisted_daydream_runs(runtime, window_days=window_days)
    if persisted_runs:
        return persisted_runs[-limit:]
    tracer = getattr(runtime, "tracer", None)
    store = getattr(tracer, "store", None)
    if store is None or not hasattr(store, "query"):
        return []
    try:
        events = store.query(limit=max(limit * 10, 100))
    except TypeError:
        events = store.query(limit=max(limit * 10, 100), kinds=None)
    runs: List[Dict[str, Any]] = []
    for event in reversed(list(events)):
        if str(getattr(event, "message", "")) != "AgentScheduler: daydream_complete":
            continue
        payload = dict(getattr(event, "payload", {}) or {})
        runs.append(
            {
                "timestamp": _iso(getattr(event, "timestamp", None)),
                "reflections": int(payload.get("reflections", 0) or 0),
                "keepers": int(payload.get("keepers", 0) or 0),
                "memories_created": int(payload.get("daydream_memories_created", 0) or 0),
                "promoted_work_count": len(payload.get("daydream_work_objects", []) or []),
                "had_activity": bool((payload.get("reflections", 0) or 0) > 0 or (payload.get("keepers", 0) or 0) > 0),
            }
        )
    return list(reversed(runs[-limit:]))


async def _build_summary(runtime: Any, *, window_days: int, reflection_limit: int) -> Dict[str, Any]:
    reflections = await _load_reflections(runtime, limit=reflection_limit, keeper_only=None)
    keepers = [item for item in reflections if item["keeper"]]
    conflicts = await _load_conflicts(runtime, limit=40, resolved=None)
    work = await _load_work_promotions(runtime, limit=20)
    memories = await _load_keeper_memories(runtime, limit=20)
    recent_runs = _load_recent_runs(runtime, limit=16, window_days=window_days)

    store = getattr(getattr(runtime, "ctx", None), "daydream_store", None)
    if store is not None and hasattr(store, "get_summary"):
        summary = await store.get_summary(window_days=window_days)
    else:
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - (max(1, window_days) * 86400)
        summary = {
            "total_reflections": len(reflections),
            "total_keepers": len(keepers),
            "window_days": max(1, window_days),
            "window_reflections": sum(
                1
                for item in reflections
                if item["created_at"] and datetime.fromisoformat(item["created_at"]).timestamp() >= cutoff
            ),
            "window_keepers": sum(
                1
                for item in keepers
                if item["created_at"] and datetime.fromisoformat(item["created_at"]).timestamp() >= cutoff
            ),
            "latest_reflection_at": reflections[0]["created_at"] if reflections else None,
        }

    active_conflicts = [item for item in conflicts if not item["resolved"]]
    resolved_conflicts = [item for item in conflicts if item["resolved"]]
    stage_counts = dict(Counter(item["stage"] or "unknown" for item in work))

    latest_active_run = next((item for item in recent_runs if item["had_activity"]), None)
    if latest_active_run is None and recent_runs:
        latest_active_run = recent_runs[0]

    return {
        "summary": {
            **summary,
            "active_conflicts": len(active_conflicts),
            "resolved_conflicts": len(resolved_conflicts),
            "promoted_work_count": len(work),
            "keeper_memory_count": len(memories),
            "promotion_stage_counts": stage_counts,
            "recent_run_count": len(recent_runs),
            "latest_active_run": latest_active_run,
        },
        "recent_runs": recent_runs,
        "recent_reflections": reflections[:8],
        "active_conflicts": active_conflicts[:8],
        "recent_promotions": work[:8],
        "recent_keeper_memories": memories[:8],
    }


def build_daydream_router(runtime: Any) -> APIRouter:
    """Build daydream routes wired to *runtime*."""
    r = APIRouter(prefix="/api/daydream", tags=["daydream"])

    @r.get("/summary")
    async def get_daydream_summary(window_days: int = 7, limit: int = 24) -> Dict[str, Any]:
        return await _build_summary(runtime, window_days=max(1, min(window_days, 30)), reflection_limit=max(8, min(limit, 80)))

    @r.get("/reflections")
    async def list_reflections(limit: int = 30, keeper_only: Optional[bool] = None) -> Dict[str, Any]:
        items = await _load_reflections(runtime, limit=max(1, min(limit, 200)), keeper_only=keeper_only)
        return {
            "count": len(items),
            "keeper_only": keeper_only,
            "items": items,
        }

    @r.get("/conflicts")
    async def list_conflicts(limit: int = 20, state: str = "all") -> Dict[str, Any]:
        resolved: Optional[bool]
        if state == "active":
            resolved = False
        elif state == "resolved":
            resolved = True
        else:
            resolved = None
        items = await _load_conflicts(runtime, limit=max(1, min(limit, 100)), resolved=resolved)
        return {
            "count": len(items),
            "state": state,
            "items": items,
        }

    @r.get("/promotions")
    async def get_promotions(limit: int = 20) -> Dict[str, Any]:
        bounded = max(1, min(limit, 100))
        work = await _load_work_promotions(runtime, limit=bounded)
        memories = await _load_keeper_memories(runtime, limit=bounded)
        return {
            "work_count": len(work),
            "keeper_memory_count": len(memories),
            "work_items": work,
            "keeper_memories": memories,
        }

    @r.get("/sparks")
    async def list_sparks(limit: int = 50) -> Dict[str, Any]:
        store = getattr(getattr(runtime, "ctx", None), "daydream_store", None)
        if store is None or not hasattr(store, "list_sparks"):
            return {"count": 0, "items": []}
        items = await store.list_sparks(limit=max(1, min(limit, 200)))
        return {"count": len(items), "items": [item.model_dump(mode="json") for item in items]}

    @r.get("/initiatives")
    async def list_initiatives(limit: int = 50) -> Dict[str, Any]:
        store = getattr(getattr(runtime, "ctx", None), "daydream_store", None)
        if store is None or not hasattr(store, "list_initiatives"):
            return {"count": 0, "items": []}
        items = await store.list_initiatives(limit=max(1, min(limit, 200)))
        return {"count": len(items), "items": [item.model_dump(mode="json") for item in items]}

    @r.get("/outcomes")
    async def list_outcomes(limit: int = 50) -> Dict[str, Any]:
        store = getattr(getattr(runtime, "ctx", None), "daydream_store", None)
        if store is None or not hasattr(store, "list_outcomes"):
            return {"count": 0, "items": []}
        items = await store.list_outcomes(limit=max(1, min(limit, 200)))
        return {"count": len(items), "items": [item.model_dump(mode="json") for item in items]}

    @r.get("/notifications")
    async def list_notifications(limit: int = 50) -> Dict[str, Any]:
        store = getattr(getattr(runtime, "ctx", None), "daydream_store", None)
        if store is None or not hasattr(store, "list_notifications"):
            return {"count": 0, "items": []}
        items = await store.list_notifications(limit=max(1, min(limit, 200)))
        return {"count": len(items), "items": [item.model_dump(mode="json") for item in items]}

    @r.get("/lifecycle/{spark_id}")
    async def get_lifecycle(spark_id: str) -> Dict[str, Any]:
        store = getattr(getattr(runtime, "ctx", None), "daydream_store", None)
        if store is None or not hasattr(store, "get_lifecycle_for_spark"):
            return {"spark": None, "initiatives": [], "outcomes": [], "notifications": []}
        return await store.get_lifecycle_for_spark(spark_id)

    return r
