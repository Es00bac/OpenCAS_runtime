"""Daydream API routes for reflections, conflicts, and promotion lineage."""

from __future__ import annotations

import json
import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

from opencas.daydream.association_memory import DAYDREAM_ASSOCIATION_TAG


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


def _compact_sequence(values: Any, *, limit: int = 12) -> List[Any]:
    if not isinstance(values, list):
        return []
    compacted = values[:limit]
    if len(values) > limit:
        compacted.append(f"... {len(values) - limit} more")
    return compacted


def _compact_mapping(mapping: Any, *, text_limit: int = 240, item_limit: int = 12) -> Dict[str, Any]:
    if not isinstance(mapping, dict):
        return {}
    compacted: Dict[str, Any] = {}
    for index, (key, value) in enumerate(mapping.items()):
        if index >= item_limit:
            compacted["__truncated__"] = len(mapping) - item_limit
            break
        if isinstance(value, str):
            compacted[str(key)] = _truncate(value, text_limit)
        elif isinstance(value, list):
            compacted[str(key)] = _compact_sequence(value, limit=8)
        elif isinstance(value, dict):
            compacted[str(key)] = _compact_mapping(value, text_limit=text_limit, item_limit=8)
        else:
            compacted[str(key)] = value
    return compacted


def _compact_summary_item(item: Dict[str, Any]) -> Dict[str, Any]:
    compacted = dict(item)
    text_limits = {
        "spark_content": 420,
        "recollection": 420,
        "interpretation": 420,
        "synthesis": 420,
        "open_question": 420,
        "changed_self_view": 420,
        "description": 420,
        "resolution_notes": 420,
        "content": 420,
        "summary": 420,
        "imaginative_branch": 420,
        "practical_branch": 420,
        "bridge": 420,
        "outcome": 420,
    }
    for key, limit in text_limits.items():
        if key in compacted:
            compacted[key] = _truncate(compacted.get(key), limit)
    for key in ("source_episode_ids", "source_memory_ids", "evidence_refs", "artifact_paths", "tags", "blocked_by"):
        if key in compacted:
            compacted[key] = _compact_sequence(compacted.get(key), limit=12)
    for key in ("meta", "validation", "raw", "somatic_context"):
        if key in compacted:
            compacted[key] = _compact_mapping(compacted.get(key), text_limit=240, item_limit=12)
    if "thoughts" in compacted:
        compacted["thoughts_included"] = False
        compacted.pop("thoughts", None)
    return compacted


def _compact_summary_items(items: List[Dict[str, Any]], *, compact: bool) -> List[Dict[str, Any]]:
    if not compact:
        return items
    return [_compact_summary_item(item) for item in items]


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _with_route_semantics(payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    try:
        route_version = int(meta.get("route_schema_version") or 1)
    except (TypeError, ValueError):
        route_version = 1
    payload["route_schema_version"] = max(1, route_version)
    payload["stale_route_semantics"] = payload["route_schema_version"] < 2
    return payload


def _reflection_to_dict(reflection: Any) -> Dict[str, Any]:
    thoughts = [
        thought.model_dump(mode="json") if hasattr(thought, "model_dump") else dict(thought)
        for thought in list(getattr(reflection, "thoughts", []) or [])
        if hasattr(thought, "model_dump") or isinstance(thought, dict)
    ]
    inner_dialogue_turn_count = sum(
        len(item.get("inner_dialogue", []) or [])
        for item in thoughts
        if isinstance(item, dict)
    )
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
        "thought_count": len(thoughts),
        "inner_dialogue_turn_count": inner_dialogue_turn_count,
        "thoughts": thoughts,
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


def _proposal_to_dict(proposal: Any) -> Dict[str, Any]:
    validation = getattr(proposal, "validation", {}) or {}
    return {
        "proposal_id": str(getattr(proposal, "proposal_id", "")),
        "created_at": _iso(getattr(proposal, "created_at", None)),
        "updated_at": _iso(getattr(proposal, "updated_at", None)),
        "source_lane": str(_enum_value(getattr(proposal, "source_lane", ""))),
        "source_snapshot_id": getattr(proposal, "source_snapshot_id", ""),
        "source_epoch": int(getattr(proposal, "source_epoch", 0) or 0),
        "proposal_kind": getattr(proposal, "proposal_kind", ""),
        "project_id": getattr(proposal, "project_id", None),
        "commitment_id": getattr(proposal, "commitment_id", None),
        "schedule_id": getattr(proposal, "schedule_id", None),
        "task_id": getattr(proposal, "task_id", None),
        "content": getattr(proposal, "content", ""),
        "content_preview": _truncate(getattr(proposal, "content", ""), 160),
        "evidence_refs": list(getattr(proposal, "evidence_refs", []) or []),
        "confidence": float(getattr(proposal, "confidence", 0.0) or 0.0),
        "authority": str(_enum_value(getattr(proposal, "authority", ""))),
        "status": str(_enum_value(getattr(proposal, "status", ""))),
        "validation": dict(validation) if isinstance(validation, dict) else {},
    }


def _signal_to_dict(signal: Any) -> Dict[str, Any]:
    if hasattr(signal, "model_dump"):
        data = signal.model_dump(mode="json")
        return _with_route_semantics(dict(data))
    payload = {
        "signal_id": str(getattr(signal, "signal_id", "")),
        "created_at": _iso(getattr(signal, "created_at", None)),
        "source_reflection_id": getattr(signal, "source_reflection_id", ""),
        "source_thought_index": int(getattr(signal, "source_thought_index", 0) or 0),
        "source_mode": getattr(signal, "source_mode", ""),
        "summary": getattr(signal, "summary", ""),
        "summary_preview": _truncate(getattr(signal, "summary", ""), 120),
        "imaginative_branch": getattr(signal, "imaginative_branch", ""),
        "practical_branch": getattr(signal, "practical_branch", ""),
        "bridge": getattr(signal, "bridge", ""),
        "novelty": float(getattr(signal, "novelty", 0.0) or 0.0),
        "usefulness": float(getattr(signal, "usefulness", 0.0) or 0.0),
        "confidence": float(getattr(signal, "confidence", 0.0) or 0.0),
        "risk": float(getattr(signal, "risk", 0.0) or 0.0),
        "suggested_route": _enum_value(getattr(signal, "suggested_route", "")),
        "contact_posture": _enum_value(getattr(signal, "contact_posture", "")),
        "self_work_kind": _enum_value(getattr(signal, "self_work_kind", "")),
        "route_status": getattr(signal, "route_status", ""),
        "route_reason": getattr(signal, "route_reason", ""),
        "artifact_paths": list(getattr(signal, "artifact_paths", []) or []),
        "evidence_ids": list(getattr(signal, "evidence_ids", []) or []),
        "meta": dict(getattr(signal, "meta", {}) or {}),
    }
    return _with_route_semantics(payload)


def _receipt_to_dict(receipt: Any) -> Dict[str, Any]:
    if hasattr(receipt, "model_dump"):
        data = receipt.model_dump(mode="json")
        return dict(data)
    return {
        "receipt_id": str(getattr(receipt, "receipt_id", "")),
        "signal_id": str(getattr(receipt, "signal_id", "")),
        "created_at": _iso(getattr(receipt, "created_at", None)),
        "route": _enum_value(getattr(receipt, "route", "")),
        "kind": _enum_value(getattr(receipt, "kind", "")),
        "outcome": getattr(receipt, "outcome", ""),
        "summary": getattr(receipt, "summary", ""),
        "artifact_paths": list(getattr(receipt, "artifact_paths", []) or []),
        "raw": dict(getattr(receipt, "raw", {}) or {}),
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
        items = [
            item
            for item in await memory.list_memories_by_tag("keeper", limit=max(limit * 4, limit))
            if "daydream" in list(getattr(item, "tags", []) or [])
        ][:limit]
    elif hasattr(memory, "list_memories"):
        items = [
            item
            for item in await memory.list_memories(limit=max(limit * 4, limit))
            if "daydream" in list(getattr(item, "tags", []) or [])
            and "keeper" in list(getattr(item, "tags", []) or [])
        ][:limit]
    else:
        items = []
    return [_memory_to_dict(item) for item in items]


async def _load_association_memories(runtime: Any, *, limit: int) -> List[Dict[str, Any]]:
    memory = getattr(runtime, "memory", None) or getattr(getattr(runtime, "ctx", None), "memory", None)
    if memory is None:
        return []
    if hasattr(memory, "list_memories_by_tag"):
        items = await memory.list_memories_by_tag(DAYDREAM_ASSOCIATION_TAG, limit=limit)
    elif hasattr(memory, "list_memories"):
        items = [
            item
            for item in await memory.list_memories(limit=max(limit * 4, limit))
            if DAYDREAM_ASSOCIATION_TAG in list(getattr(item, "tags", []) or [])
        ][:limit]
    else:
        items = []
    return [_memory_to_dict(item) for item in items]


def _context_proposal_store(runtime: Any) -> Any:
    return (
        getattr(runtime, "context_proposals", None)
        or getattr(getattr(runtime, "ctx", None), "context_proposal_store", None)
    )


async def _load_context_proposals(
    runtime: Any,
    *,
    limit: int,
    search: Optional[str] = None,
) -> List[Dict[str, Any]]:
    store = _context_proposal_store(runtime)
    if store is None:
        return []
    try:
        if search and hasattr(store, "search_text"):
            items = await store.search_text(search, include_terminal=True, limit=limit)
        elif hasattr(store, "list_recent"):
            items = await store.list_recent(limit=limit)
        else:
            items = []
    except Exception:
        items = []
    return [_proposal_to_dict(item) for item in items]


async def _context_proposal_status_counts(runtime: Any) -> Dict[str, int]:
    store = _context_proposal_store(runtime)
    if store is None or not hasattr(store, "count_by_status"):
        return {}
    try:
        return dict(await store.count_by_status())
    except Exception:
        return {}


async def _count_memories_by_tag(
    runtime: Any,
    tag: str,
    *,
    required_tag: Optional[str] = None,
    scan_limit: int = 10000,
) -> int:
    memory = getattr(runtime, "memory", None) or getattr(getattr(runtime, "ctx", None), "memory", None)
    if memory is None:
        return 0
    count_by_tag = getattr(memory, "count_memories_by_tag", None)
    if callable(count_by_tag):
        try:
            return int(await count_by_tag(tag, required_tag=required_tag))
        except TypeError:
            return int(await count_by_tag(tag))
        except Exception:
            pass
    if hasattr(memory, "list_memories_by_tag"):
        items = await memory.list_memories_by_tag(tag, limit=scan_limit)
    elif hasattr(memory, "list_memories"):
        items = [
            item
            for item in await memory.list_memories(limit=scan_limit)
            if tag in list(getattr(item, "tags", []) or [])
        ]
    else:
        return 0
    if required_tag:
        items = [
            item
            for item in items
            if required_tag in list(getattr(item, "tags", []) or [])
        ]
    return len(items)


def _signal_store(runtime: Any) -> Any:
    return (
        getattr(getattr(runtime, "ctx", None), "daydream_signal_store", None)
        or getattr(runtime, "daydream_signal_store", None)
    )


async def _load_signal_summary(runtime: Any, *, window_days: int) -> Dict[str, Any]:
    store = _signal_store(runtime)
    if store is None:
        return {
            "total_signals": 0,
            "window_days": window_days,
            "window_signals": 0,
            "route_counts": {},
            "status_counts": {},
            "window_self_work_receipts": 0,
        }
    if hasattr(store, "get_summary"):
        return dict(await store.get_summary(window_days=window_days))
    signals = await _call_store_list(store, "list_recent", limit=100)
    receipts = await _call_store_list(store, "list_receipts", limit=100)
    return {
        "total_signals": len(signals),
        "window_days": window_days,
        "window_signals": len(signals),
        "route_counts": dict(
            Counter(
                str(_enum_value(getattr(item, "suggested_route", "unknown")))
                for item in signals
            )
        ),
        "status_counts": dict(
            Counter(
                str(getattr(item, "route_status", "unknown") or "unknown")
                for item in signals
            )
        ),
        "window_self_work_receipts": len(receipts),
    }


async def _load_signals(runtime: Any, *, limit: int) -> List[Dict[str, Any]]:
    store = _signal_store(runtime)
    if store is None:
        return []
    return [_signal_to_dict(item) for item in await _call_store_list(store, "list_recent", limit=limit)]


async def _load_self_work_receipts(runtime: Any, *, limit: int) -> List[Dict[str, Any]]:
    store = _signal_store(runtime)
    if store is None:
        return []
    return [_receipt_to_dict(item) for item in await _call_store_list(store, "list_receipts", limit=limit)]


async def _call_store_list(store: Any, method_name: str, *, limit: int) -> list[Any]:
    method = getattr(store, method_name, None)
    if not callable(method):
        return []
    return list(await method(limit=limit) or [])


def _daydream_run_from_event(message: str, timestamp: Any, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if message in {"AgentScheduler: daydream_complete", "AgentRuntime: daydream_complete"}:
        quality_status = str(payload.get("quality_status") or _quality_status(payload))
        run = {
            "event": "daydream_complete",
            "timestamp": _iso(timestamp),
            "reflections": int(payload.get("reflections", 0) or 0),
            "keepers": int(payload.get("keepers", 0) or 0),
            "memories_created": int(payload.get("daydream_memories_created", 0) or 0),
            "association_memories_created": int(
                payload.get("daydream_association_memories_created", 0) or 0
            ),
            "context_proposals_created": int(
                payload.get("daydream_context_proposals_created", 0) or 0
            ),
            "promoted_work_count": len(payload.get("daydream_work_objects", []) or []),
            "had_activity": bool(
                (payload.get("reflections", 0) or 0) > 0
                or (payload.get("keepers", 0) or 0) > 0
                or (payload.get("daydream_association_memories_created", 0) or 0) > 0
                or (payload.get("daydream_context_proposals_created", 0) or 0) > 0
            ),
            "quality_status": quality_status,
        }
        if quality_status.lower() == "skipped" or payload.get("skip_reason") or payload.get("skip_reasons"):
            skip_reason = payload.get("skip_reason") or payload.get("reason")
            scheduler_skip_reason = payload.get("scheduler_skip_reason")
            if skip_reason is not None:
                run["skip_reason"] = str(skip_reason)
            if scheduler_skip_reason is not None:
                run["scheduler_skip_reason"] = str(scheduler_skip_reason)
            if isinstance(payload.get("skip_reasons"), list):
                run["skip_reasons"] = [str(item) for item in payload.get("skip_reasons", [])]
            for key in (
                "motivation",
                "motivation_threshold",
                "somatic_readiness",
                "cooldown_ok",
                "cooldown_seconds_remaining",
                "cooldown_until",
                "last_daydream_at",
            ):
                if key in payload:
                    run[key] = payload[key]
        return run
    if message not in {"AgentScheduler: daydream_skipped", "AgentRuntime: daydream_skipped"}:
        return None
    skip_reason = payload.get("skip_reason") or payload.get("reason") or payload.get("scheduler_skip_reason")
    scheduler_skip_reason = payload.get("scheduler_skip_reason") or payload.get("reason") or skip_reason
    run = {
        "event": "daydream_skipped",
        "timestamp": _iso(timestamp),
        "reflections": 0,
        "keepers": 0,
        "memories_created": 0,
        "association_memories_created": 0,
        "context_proposals_created": 0,
        "promoted_work_count": 0,
        "had_activity": False,
        "quality_status": "skipped",
        "skip_reason": str(skip_reason) if skip_reason is not None else None,
        "scheduler_skip_reason": str(scheduler_skip_reason) if scheduler_skip_reason is not None else None,
    }
    for key in (
        "motivation",
        "cooldown_seconds_remaining",
        "cooldown_until",
        "last_daydream_at",
    ):
        if key in payload:
            run[key] = payload[key]
    return run


def _tail_text_lines(path: Path, *, max_lines: int, block_size: int = 65536) -> List[str]:
    """Read the last *max_lines* text lines without loading the whole telemetry file."""

    if max_lines <= 0:
        return []
    blocks: List[bytes] = []
    newline_count = 0
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        while position > 0 and newline_count <= max_lines:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            blocks.append(chunk)
            newline_count += chunk.count(b"\n")
    data = b"".join(reversed(blocks))
    return data.decode("utf-8", errors="replace").splitlines()[-max_lines:]


def _iter_persisted_daydream_runs(
    runtime: Any,
    *,
    window_days: int,
    limit: int,
) -> List[Dict[str, Any]]:
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
    for day in days:
        path = telemetry_dir / f"{day.date().isoformat()}.jsonl"
        if not path.exists():
            continue
        try:
            lines = _tail_text_lines(path, max_lines=max(limit * 40, 1000))
        except Exception:
            continue
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            payload = dict(event.get("payload", {}) or {})
            run = _daydream_run_from_event(
                str(event.get("message", "")),
                event.get("timestamp"),
                payload,
            )
            if run is not None:
                runs.append(run)
                if len(runs) >= limit:
                    runs.sort(key=lambda item: item.get("timestamp") or "")
                    return runs
    runs.sort(key=lambda item: item.get("timestamp") or "")
    return runs


async def _load_recent_runs(runtime: Any, *, limit: int, window_days: int) -> List[Dict[str, Any]]:
    persisted_runs = await asyncio.to_thread(
        _iter_persisted_daydream_runs,
        runtime,
        window_days=window_days,
        limit=max(limit, 50),
    )
    tracer = getattr(runtime, "tracer", None)
    store = getattr(tracer, "store", None)
    runs: List[Dict[str, Any]] = []
    if store is not None and hasattr(store, "query"):
        try:
            events = store.query(limit=max(limit * 10, 100))
        except TypeError:
            events = store.query(limit=max(limit * 10, 100), kinds=None)
        for event in reversed(list(events)):
            payload = dict(getattr(event, "payload", {}) or {})
            run = _daydream_run_from_event(
                str(getattr(event, "message", "")),
                getattr(event, "timestamp", None),
                payload,
            )
            if run is not None:
                runs.append(run)
    runs.extend(persisted_runs)
    deduped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for run in runs:
        timestamp = str(run.get("timestamp") or "")
        status = str(run.get("status") or "")
        key = (timestamp, status)
        deduped[key] = run
    merged = sorted(deduped.values(), key=lambda item: item.get("timestamp") or "")
    return merged[-limit:]


def _quality_status(payload: Dict[str, Any]) -> str:
    if payload.get("error"):
        return "failed"
    promoted = len(payload.get("daydream_work_objects", []) or [])
    keepers = int(payload.get("keepers", 0) or 0)
    memories = int(payload.get("daydream_memories_created", 0) or 0)
    association_memories = int(payload.get("daydream_association_memories_created", 0) or 0)
    context_proposals = int(payload.get("daydream_context_proposals_created", 0) or 0)
    reflections = int(payload.get("reflections", 0) or 0)
    if promoted > 0:
        return "promoted"
    if keepers > 0 or memories > 0 or association_memories > 0 or context_proposals > 0:
        return "useful"
    if reflections > 0:
        return "observed"
    if payload.get("skip_reason"):
        return "skipped"
    return "empty"


def _consecutive_skip_count(recent_runs: List[Dict[str, Any]]) -> int:
    count = 0
    for item in reversed(recent_runs):
        if not _run_is_skip_like(item):
            break
        count += 1
    return count


def _run_is_skip_like(item: Dict[str, Any]) -> bool:
    if item.get("event") == "daydream_skipped":
        return True
    return (
        bool(item.get("had_activity")) is False
        and str(item.get("quality_status") or "").lower() == "skipped"
    )


def _last_successful_run(recent_runs: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    return next((item for item in reversed(recent_runs) if item.get("had_activity")), None)


def _skip_diagnostic(
    runtime: Any,
    recent_runs: List[Dict[str, Any]],
    consecutive_skip_count: int,
) -> Dict[str, Any]:
    latest_skip = next(
        (item for item in reversed(recent_runs) if _run_is_skip_like(item)),
        None,
    )
    if latest_skip is None:
        return {
            "status": "no_recent_skips",
            "reason": None,
            "action": "No scheduler skip diagnosis is needed from recent daydream telemetry.",
        }
    latest_explicit_reason = next(
        (
            item
            for item in reversed(recent_runs)
            if _run_is_skip_like(item)
            and (item.get("scheduler_skip_reason") or item.get("skip_reason"))
        ),
        latest_skip,
    )
    reason = latest_explicit_reason.get("scheduler_skip_reason") or latest_explicit_reason.get("skip_reason")
    if recent_runs and not _run_is_skip_like(recent_runs[-1]):
        return {
            "status": "recovered_after_skip",
            "reason": reason,
            "status_for_operator": "active_after_prior_skip",
            "pause_condition": None,
            "condition_clears": "a later daydream run produced activity",
            "explanation": (
                "A prior scheduler skip is still visible in telemetry, but a later "
                "daydream run produced activity. Treat the skip as historical, not "
                "the current operating state."
            ),
            "telemetry": _skip_telemetry(latest_explicit_reason),
            "action": "Use latest_active_run and consecutive_skip_count before interpreting old skips as current failure.",
        }
    operator_status = _operator_skip_status(runtime, reason)
    telemetry = _skip_telemetry(latest_explicit_reason)
    if consecutive_skip_count >= 2:
        return {
            "status": "repeated_skips",
            "reason": reason,
            **operator_status,
            "telemetry": telemetry,
            "action": (
                "Inspect executive pause, queue pressure, conversation quiet, and cooldown "
                "telemetry before expecting the next daydream run."
            ),
        }
    return {
        "status": "recent_skip",
        "reason": reason,
        **operator_status,
        "telemetry": telemetry,
        "action": "Check the scheduler skip reason against live runtime state before treating daydream as broken.",
    }


def _skip_telemetry(run: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "motivation",
        "motivation_threshold",
        "boredom",
        "somatic_readiness",
        "cooldown_ok",
        "cooldown_seconds_remaining",
        "cooldown_until",
        "last_daydream_at",
    )
    return {key: run[key] for key in keys if key in run}


def _operator_skip_status(runtime: Any, reason: Any) -> Dict[str, Any]:
    reason_text = str(reason or "")
    if reason_text == "executive_recommended_pause":
        executive = getattr(runtime, "executive", None)
        pause_reason = None
        recommend_pause = False
        if executive is not None:
            try:
                pause_reason_fn = getattr(executive, "pause_reason", None)
                if callable(pause_reason_fn):
                    pause_reason = pause_reason_fn()
            except Exception:
                pause_reason = None
            try:
                recommend_pause = bool(executive.recommend_pause())
            except Exception:
                recommend_pause = pause_reason is not None
        if recommend_pause:
            condition = str(pause_reason or "executive pause")
            clears = _pause_clear_condition(condition)
            return {
                "status_for_operator": "healthy_pause",
                "pause_condition": condition,
                "condition_clears": clears,
                "explanation": (
                    f"Daydream is correctly paused while {condition}. "
                    f"Will resume when {clears}."
                ),
            }
        return {
            "status_for_operator": "stale_pause_cleared",
            "pause_condition": None,
            "condition_clears": "the next scheduler attempt or operator-triggered daydream run records fresh telemetry",
            "explanation": (
                "The latest explicit skip reason says executive pause, but the live "
                "executive state does not currently recommend pause. Treat this as "
                "stale skip telemetry until a fresh daydream attempt records the new state."
            ),
        }
    if reason_text == "baa_busy":
        return {
            "status_for_operator": "healthy_pause",
            "pause_condition": "baa_busy",
            "condition_clears": "BAA queue, held, and active counts return to zero",
            "explanation": "Daydream is correctly paused while the bounded assistant is busy.",
        }
    if reason_text == "recent_user_activity":
        return {
            "status_for_operator": "healthy_pause",
            "pause_condition": "recent_user_activity",
            "condition_clears": "the conversation quiet window elapses",
            "explanation": "Daydream is correctly paused while recent user activity is still inside the quiet window.",
        }
    if reason_text == "motivation_below_threshold":
        return {
            "status_for_operator": "healthy_pause",
            "pause_condition": "motivation_below_threshold",
            "condition_clears": "motivation rises to the daydream threshold",
            "explanation": "Daydream is paused because boredom/readiness motivation is below the generation threshold.",
        }
    if reason_text == "cooldown":
        return {
            "status_for_operator": "healthy_pause",
            "pause_condition": "cooldown",
            "condition_clears": "cooldown_seconds_remaining reaches zero",
            "explanation": "Daydream is correctly paused while its cooldown window is active.",
        }
    return {
        "status_for_operator": "unknown_pause",
        "pause_condition": reason_text or None,
        "condition_clears": None,
        "explanation": "No live operator classification is available for this skip reason.",
    }


def _pause_clear_condition(condition: str) -> str:
    if condition == "overload":
        return "queue load drops below executive capacity"
    if condition == "fatigue":
        return "somatic fatigue returns to 0.7 or lower"
    return "the executive pause condition clears"


async def _build_summary(
    runtime: Any,
    *,
    window_days: int,
    reflection_limit: int,
    include_details: bool = True,
    compact_details: bool = True,
) -> Dict[str, Any]:
    detail_limit = 20 if include_details else 0
    reflections = await _load_reflections(runtime, limit=reflection_limit, keeper_only=None)
    keepers = [item for item in reflections if item["keeper"]]
    conflicts = await _load_conflicts(runtime, limit=40 if include_details else 8, resolved=None)
    work = await _load_work_promotions(runtime, limit=detail_limit) if include_details else []
    memories = await _load_keeper_memories(runtime, limit=detail_limit) if include_details else []
    association_memories = (
        await _load_association_memories(runtime, limit=detail_limit)
        if include_details
        else []
    )
    context_proposals = (
        await _load_context_proposals(runtime, limit=detail_limit)
        if include_details
        else []
    )
    context_proposal_status_counts = await _context_proposal_status_counts(runtime)
    keeper_memory_count = (
        await _count_memories_by_tag(runtime, "keeper", required_tag="daydream")
        if include_details
        else 0
    )
    association_memory_count = (
        await _count_memories_by_tag(runtime, DAYDREAM_ASSOCIATION_TAG)
        if include_details
        else 0
    )
    signal_summary = await _load_signal_summary(runtime, window_days=window_days)
    recent_signals = await _load_signals(runtime, limit=8)
    recent_self_work_receipts = await _load_self_work_receipts(runtime, limit=8)
    consecutive_window_runs = await _load_recent_runs(runtime, limit=200, window_days=window_days)
    recent_runs = consecutive_window_runs[-16:]

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
    consecutive_skip_count = _consecutive_skip_count(consecutive_window_runs)

    latest_successful_run = _last_successful_run(consecutive_window_runs)
    latest_active_run = _last_successful_run(recent_runs)
    if latest_active_run is None and recent_runs:
        latest_active_run = recent_runs[-1]

    return {
        "summary": {
            **summary,
            "active_conflicts": len(active_conflicts),
            "resolved_conflicts": len(resolved_conflicts),
            "promoted_work_count": len(work),
            "keeper_memory_count": keeper_memory_count,
            "association_memory_count": association_memory_count,
            "details_included": include_details,
            "details_compacted": compact_details,
            "context_proposal_count": sum(context_proposal_status_counts.values()),
            "context_proposal_status_counts": context_proposal_status_counts,
            "promotion_stage_counts": stage_counts,
            "recent_run_count": len(recent_runs),
            "latest_active_run": latest_active_run,
            "last_successful_run_at": (
                latest_successful_run.get("timestamp") if latest_successful_run else None
            ),
            "consecutive_skip_count": consecutive_skip_count,
            "skip_diagnostic": _skip_diagnostic(
                runtime,
                consecutive_window_runs,
                consecutive_skip_count,
            ),
            "signal_count": int(signal_summary.get("total_signals", 0) or 0),
            "signal_window_count": int(signal_summary.get("window_signals", 0) or 0),
            "signal_route_counts": dict(signal_summary.get("route_counts", {}) or {}),
            "signal_status_counts": dict(signal_summary.get("status_counts", {}) or {}),
            "self_work_receipt_count": int(signal_summary.get("window_self_work_receipts", 0) or 0),
        },
        "recent_runs": recent_runs,
        "recent_reflections": _compact_summary_items(reflections[:8], compact=compact_details),
        "active_conflicts": _compact_summary_items(active_conflicts[:8], compact=compact_details),
        "recent_promotions": _compact_summary_items(work[:8], compact=compact_details),
        "recent_keeper_memories": _compact_summary_items(memories[:8], compact=compact_details),
        "recent_association_memories": _compact_summary_items(association_memories[:8], compact=compact_details),
        "recent_context_proposals": _compact_summary_items(context_proposals[:8], compact=compact_details),
        "recent_signals": _compact_summary_items(recent_signals, compact=compact_details),
        "recent_self_work_receipts": _compact_summary_items(recent_self_work_receipts, compact=compact_details),
    }


def build_daydream_router(runtime: Any) -> APIRouter:
    """Build daydream routes wired to *runtime*."""
    r = APIRouter(prefix="/api/daydream", tags=["daydream"])

    @r.get("/summary")
    async def get_daydream_summary(
        window_days: int = 7,
        limit: int = 24,
        include_details: bool = True,
        compact_details: bool = True,
    ) -> Dict[str, Any]:
        return await _build_summary(
            runtime,
            window_days=max(1, min(window_days, 30)),
            reflection_limit=max(8, min(limit, 80)),
            include_details=include_details,
            compact_details=compact_details,
        )

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
        association_memories = await _load_association_memories(runtime, limit=bounded)
        context_proposals = await _load_context_proposals(runtime, limit=bounded)
        context_proposal_status_counts = await _context_proposal_status_counts(runtime)
        return {
            "work_count": len(work),
            "keeper_memory_count": await _count_memories_by_tag(runtime, "keeper", required_tag="daydream"),
            "association_memory_count": await _count_memories_by_tag(runtime, DAYDREAM_ASSOCIATION_TAG),
            "context_proposal_count": sum(context_proposal_status_counts.values()),
            "context_proposal_status_counts": context_proposal_status_counts,
            "work_items": work,
            "keeper_memories": memories,
            "association_memories": association_memories,
            "context_proposals": context_proposals,
        }

    @r.get("/proposals")
    async def list_context_proposals(limit: int = 50, search: Optional[str] = None) -> Dict[str, Any]:
        bounded = max(1, min(limit, 200))
        items = await _load_context_proposals(runtime, limit=bounded, search=search)
        return {
            "count": len(items),
            "status_counts": await _context_proposal_status_counts(runtime),
            "search": search,
            "items": items,
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

    @r.get("/signals")
    async def list_signals(limit: int = 50) -> Dict[str, Any]:
        items = await _load_signals(runtime, limit=max(1, min(limit, 200)))
        return {"count": len(items), "items": items}

    @r.get("/self-work")
    async def list_self_work(limit: int = 50) -> Dict[str, Any]:
        items = await _load_self_work_receipts(runtime, limit=max(1, min(limit, 200)))
        return {"count": len(items), "items": items}

    @r.get("/lifecycle/{spark_id}")
    async def get_lifecycle(spark_id: str) -> Dict[str, Any]:
        store = getattr(getattr(runtime, "ctx", None), "daydream_store", None)
        if store is None or not hasattr(store, "get_lifecycle_for_spark"):
            return {"spark": None, "initiatives": [], "outcomes": [], "notifications": []}
        return await store.get_lifecycle_for_spark(spark_id)

    @r.post("/run")
    async def run_daydream_now(
        reason: str = "operator_nudge",
        force: bool = False,
        reflective_only: bool = False,
    ) -> Dict[str, Any]:
        """Run one daydream pass through the live runtime for operator verification."""
        runner = getattr(runtime, "run_daydream", None)
        if not callable(runner):
            return {
                "available": False,
                "ran": False,
                "reason": "runtime_daydream_unavailable",
            }
        result = await runner(force=force, reflective_only=reflective_only)
        trace = getattr(runtime, "_trace", None)
        if callable(trace) and isinstance(result, dict):
            trace_payload = {
                key: value
                for key, value in result.items()
                if key not in {"daydream_work_objects", "reflections_list"}
            }
            trace_payload["trigger"] = "operator_daydream_run"
            trace_payload["operator_reason"] = reason
            trace_payload["force"] = force
            trace_payload["reflective_only"] = reflective_only
            event_name = "daydream_skipped" if result.get("skip_reason") else "daydream_complete"
            try:
                trace(event_name, trace_payload)
            except Exception:
                pass
        return {
            "available": True,
            "ran": True,
            "reason": reason,
            "force": force,
            "reflective_only": reflective_only,
            "result": result,
        }

    return r
