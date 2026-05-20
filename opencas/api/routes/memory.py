"""Memory API routes for the OpenCAS dashboard."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


from opencas.api.memory_projection import (
    build_memory_retriever,
    collect_embedding_points,
    enrich_nodes_with_embeddings,
    get_total_episode_count,
)
from opencas.context.resonance import compute_edge_strength
from opencas.api.memory_serialization import (
    affect_to_dict,
    edge_signal_summary,
    edge_to_dict,
    episode_context_metadata,
    episode_to_dict,
    memory_context_metadata,
    memory_to_dict,
    truncate_memory_text,
)
from opencas.context.retriever import MemoryRetriever
from opencas.memory import EdgeKind
from opencas.telemetry import EventKind

router = APIRouter(tags=["memory"])


class EpisodesResponse(BaseModel):
    episodes: List[Dict[str, Any]]
    total: int


class GraphResponse(BaseModel):
    episode_id: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class MemoryStatsResponse(BaseModel):
    episode_count: int
    memory_count: int
    edge_count: int
    compacted_count: int
    identity_core_count: int
    avg_salience: float
    affect_distribution: Dict[str, int]


class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]


class ProjectionResponse(BaseModel):
    points: List[Dict[str, Any]]
    method: str
    groups: List[Dict[str, Any]] = Field(default_factory=list)


class MemoryLandscapeResponse(BaseModel):
    stats: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    projection: Dict[str, Any]


class RetrievalInspectResponse(BaseModel):
    query: str
    weights: Dict[str, float]
    candidates: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    meta: Dict[str, Any]


class NodeDetailResponse(BaseModel):
    node: Dict[str, Any]
    neighbors: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    stats: Dict[str, Any]


class MemoryActivityResponse(BaseModel):
    generated_at: str
    window_seconds: int
    event_count: int
    active_node_ids: List[str]
    active_nodes: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    stats: Dict[str, Any] = Field(default_factory=dict)


class CognitiveFlowResponse(BaseModel):
    generated_at: str
    window_seconds: int
    event_count: int
    span_count: int
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    stats: Dict[str, Any] = Field(default_factory=dict)


def _safe_str(value: Any) -> str:
    return str(value or "")


def _ensure_tz(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _safe_event_timestamp(value: Any, now: datetime) -> datetime:
    if value is None:
        return now
    if isinstance(value, datetime):
        return _ensure_tz(value)
    if isinstance(value, str):
        try:
            normalized = value.strip()
            try:
                return _ensure_tz(datetime.fromisoformat(normalized))
            except ValueError:
                return _ensure_tz(datetime.fromisoformat(normalized.replace("Z", "+00:00")))
        except ValueError:
            return now
        return now


_TEMPORAL_STRING_KEYS = (
    "timestamp",
    "event_timestamp",
    "created_at",
    "updated_at",
    "last_activated_at",
    "activated_at",
    "occurred_at",
    "event_time",
    "timestamp_iso",
    "routed_at",
    "generated_at",
)
_TEMPORAL_NUMERIC_KEYS = (
    "temporal_timestamp_ms",
    "timestamp_ms",
    "event_timestamp_ms",
    "activated_at_ms",
    "last_activated_at_ms",
    "created_at_ms",
    "updated_at_ms",
)


def _parse_temporal_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_tz(value)
    if isinstance(value, (int, float)):
        raw = float(value)
        if raw <= 0:
            return None
        if raw > 10_000_000_000:
            raw = raw / 1000.0
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        try:
            numeric = float(normalized)
        except ValueError:
            numeric = None
        if numeric is not None:
            return _parse_temporal_datetime(numeric)
        try:
            return _ensure_tz(datetime.fromisoformat(normalized))
        except ValueError:
            try:
                return _ensure_tz(datetime.fromisoformat(normalized.replace("Z", "+00:00")))
            except ValueError:
                return None
    return None


def _temporal_fields(timestamp: Any, *, source: str, now: datetime | None = None) -> Dict[str, Any]:
    parsed = _parse_temporal_datetime(timestamp)
    if parsed is None:
        return {
            "temporal_has_metadata": False,
            "temporal_source": "missing",
        }
    parsed = _ensure_tz(parsed)
    now = now or datetime.now(timezone.utc)
    return {
        "timestamp": parsed.isoformat(),
        "temporal_timestamp_ms": int(parsed.timestamp() * 1000),
        "temporal_has_metadata": True,
        "temporal_source": source,
        "age_seconds": round(max(0.0, (now - parsed).total_seconds()), 3),
        "age_days": round(max(0.0, (now - parsed).total_seconds()) / 86400.0, 3),
    }


def _with_temporal_metadata(
    payload: Dict[str, Any],
    *,
    timestamp: Any = None,
    source: str = "timestamp",
    now: datetime | None = None,
) -> Dict[str, Any]:
    payload = dict(payload)
    selected_source = source
    selected_timestamp = timestamp
    if selected_timestamp is None:
        for key in _TEMPORAL_STRING_KEYS:
            if payload.get(key) not in (None, ""):
                selected_timestamp = payload.get(key)
                selected_source = key
                break
    if selected_timestamp is None:
        for key in _TEMPORAL_NUMERIC_KEYS:
            if payload.get(key) not in (None, ""):
                selected_timestamp = payload.get(key)
                selected_source = key
                break
    if selected_timestamp is not None:
        fields = _temporal_fields(selected_timestamp, source=selected_source, now=now)
        if fields["temporal_has_metadata"]:
            original_age_days = payload.get("age_days")
            payload.update(fields)
            if original_age_days is not None and selected_source not in {"age_days"}:
                payload["relative_age_days"] = original_age_days
            return payload
    if payload.get("age_days") is not None:
        payload["temporal_has_metadata"] = True
        payload["temporal_source"] = "age_days"
        return payload
    payload["temporal_has_metadata"] = False
    payload["temporal_source"] = "missing"
    return payload


def _infer_subsystem(
    *,
    kind: str,
    message: str,
    payload: Dict[str, Any],
) -> str:
    explicit = _safe_str(
        payload.get("subsystem")
        or payload.get("system")
        or payload.get("pipeline")
        or payload.get("source")
    )
    if explicit:
        return explicit
    source_type = _safe_str(payload.get("source_type") or payload.get("source_name"))
    if source_type:
        return source_type
    source_type = _safe_str(payload.get("source_id"))
    if source_type:
        return source_type
    cleaned_kind = kind.replace("_", " ")
    if "span_start" in kind or "span_end" in kind:
        return "runtime_span"
    if "agent_loop" in cleaned_kind or "agent_loop" in message.lower():
        return "agent_loop"
    if "agent runtime" in message.lower() or "agentruntime" in message.lower():
        return "agent_runtime"
    if "tool" in cleaned_kind or "tool" in message.lower():
        return "tooling"
    if ":" in message:
        token = message.split(":", 1)[0].strip()
        if token:
            return token.replace(".", "_").lower()
    return "runtime"


def _infer_pipeline_stage(
    *,
    kind: str,
    message: str,
    payload: Dict[str, Any],
) -> str:
    operation = _safe_str(payload.get("operation") or payload.get("name") or payload.get("stage"))
    if operation:
        return operation
    text = f"{kind} {message.lower()}"
    if "span_start" in kind or "span_end" in kind:
        span_name = _safe_str(payload.get("name"))
        if span_name:
            return span_name
        return "span"
    if "decision" in text or "select" in text or "choose" in text or "promote" in text:
        return "decision"
    if "converse" in text or "conversation" in text:
        return "conversation"
    if "cycle" in text:
        return "cycle"
    if "daydream" in text:
        return "daydream"
    if "tool_call" in kind or "llm_call" in kind:
        return "execution"
    if "consolidation" in text:
        return "consolidation"
    if "memory" in text:
        return "memory"
    if "split" in text or "route" in text:
        return "routing"
    return "processing"


def _infer_decision_signal(
    *,
    kind: str,
    message: str,
    payload: Dict[str, Any],
) -> bool:
    text = (f"{kind} {message} " + _safe_str(payload.get("operation"))).lower()
    return any(token in text for token in ["decision", "decide", "select", "route", "promote", "intervention", "choose", "gate"])


def _compact_flow_text(value: Any, limit: int = 900) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: max(0, limit - 3)] + "..."
    return text


def _first_payload_text(payload: Dict[str, Any], keys: List[str], *, limit: int = 900) -> str:
    for key in keys:
        text = _compact_flow_text(payload.get(key), limit=limit)
        if text:
            return text
    nested = payload.get("payload")
    if isinstance(nested, dict):
        for key in keys:
            text = _compact_flow_text(nested.get(key), limit=limit)
            if text:
                return text
    return ""


def _is_low_information_flow_event(
    *,
    kind: str,
    message: str,
    payload: Dict[str, Any],
) -> bool:
    text = f"{kind} {message}".lower()
    if kind == EventKind.DIAGNOSTIC_RUN.value and (
        "doctor telemetry probe" in text or "healthmonitor sweep" in text
    ):
        return True
    if "toolregistry" in text and "tool_registered" in text:
        return True
    if "tool_registered" in text and not _first_payload_text(
        payload, ["content_preview", "summary", "reasoning", "reason", "result"]
    ):
        return True
    lifecycle_markers = (
        "scheduler_start",
        "scheduler_stop",
        "health_monitor_started",
        "health_monitor_stopped",
        "telegram_started",
        "telegram_stopped",
        "baa_heartbeat",
        "identity_persistence_heartbeat",
        "telemetry_pruned",
        "workspace_sync_complete",
        "thread_backfill",
        "thread_beads",
        "capability_drift_episode_recorded",
    )
    if any(marker in text for marker in lifecycle_markers):
        return True
    if kind == EventKind.BOOTSTRAP_STAGE.value:
        return True
    if kind == EventKind.LLM_CALL.value and "embed_batch" in text:
        return True
    if (
        "schedule_triggered" in text
        and _safe_str(payload.get("status") or payload.get("result")).lower() == "skipped"
        and not _first_payload_text(payload, ["reasoning", "reason", "diagnostic", "summary", "result"])
    ):
        return True
    if "poll" in text and not payload:
        return True
    return False


def _flow_priority(
    *,
    kind: str,
    message: str,
    payload: Dict[str, Any],
    decision_signal: bool,
    is_noise: bool,
) -> float:
    if is_noise:
        return 0.05
    if decision_signal:
        return 0.95
    text = f"{kind} {message}".lower()
    if kind in {EventKind.MEMORY_ACTIVATED.value, EventKind.ACTION_BACKLINK.value}:
        return 0.9
    if kind in {EventKind.SELF_APPROVAL.value, EventKind.TOOL_CALL.value}:
        return 0.86
    if kind == EventKind.LLM_CALL.value:
        return 0.78
    if "daydream" in text or "reflection" in text or "proposal" in text:
        return 0.8
    if "consolidation" in text or "schedule" in text or "initiative" in text:
        return 0.7
    if _first_payload_text(payload, ["content_preview", "summary", "query", "reasoning", "reason"]):
        return 0.6
    return 0.25


def _flow_event_selection_score(event: Any) -> float:
    payload = dict(getattr(event, "payload", {}) or {})
    kind = _enum_value(getattr(event, "kind", ""))
    message = str(getattr(event, "message", "") or "")
    decision_signal = _infer_decision_signal(kind=kind, message=message, payload=payload)
    is_noise = _is_low_information_flow_event(kind=kind, message=message, payload=payload)
    return _flow_priority(
        kind=kind,
        message=message,
        payload=payload,
        decision_signal=decision_signal,
        is_noise=is_noise,
    )


def _flow_event_is_noise(event: Any) -> bool:
    payload = dict(getattr(event, "payload", {}) or {})
    kind = _enum_value(getattr(event, "kind", ""))
    message = str(getattr(event, "message", "") or "")
    return _is_low_information_flow_event(kind=kind, message=message, payload=payload)


def _is_readable_thought_node(event: Dict[str, Any]) -> bool:
    if bool(event.get("is_noise")):
        return False
    priority = float(event.get("flow_priority") or 0.0)
    thought = _safe_str(event.get("thought_text"))
    if not thought:
        return False
    reasoning = _safe_str(event.get("reasoning_text"))
    evidence = _safe_str(event.get("evidence_text"))
    kind = _safe_str(event.get("kind"))
    lane = _safe_str(event.get("flow_lane"))
    event_payload = event.get("event_payload") if isinstance(event.get("event_payload"), dict) else {}
    tool_result = _safe_str(event_payload.get("result")).lower()
    text = f"{event.get('label') or ''} {event.get('event_message') or event.get('message') or ''}".lower()
    generic = thought == _compact_flow_text(
        f"{event.get('subsystem') or ''} {event.get('pipeline_stage') or 'processing'}: {event.get('event_message') or event.get('message') or ''}",
        limit=1100,
    )
    if generic and not reasoning and not evidence and priority < 0.6:
        return False
    if reasoning or evidence or bool(event.get("decision_signal")):
        return priority >= 0.25
    if kind == EventKind.TOOL_CALL.value:
        if "schedule" in text and tool_result in {"", "skipped", "skip"}:
            return False
        if tool_result in {"", "ok", "success", "complete", "completed", "skipped", "skip"}:
            return False
        return priority >= 0.55
    if kind in {EventKind.MEMORY_ACTIVATED.value, EventKind.SELF_APPROVAL.value, EventKind.LLM_CALL.value}:
        return priority >= 0.55
    if "daydream" in lane or "reflection" in lane:
        return priority >= 0.55
    return priority >= 0.7


def _recent_distinct_readable_thoughts(
    readable_thoughts: List[Dict[str, Any]],
    *,
    limit: int = 12,
) -> List[Dict[str, Any]]:
    """Return the newest readable thought records without repeated telemetry spam."""

    selected: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str]] = set()
    for event in reversed(readable_thoughts):
        signature = (
            _safe_str(event.get("flow_lane")).lower(),
            _safe_str(event.get("label")).lower(),
            _safe_str(event.get("thought_text")).lower(),
            _safe_str(event.get("reasoning_text")).lower(),
        )
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(event)
        if len(selected) >= limit:
            break
    return list(reversed(selected))


def _observable_thought_text(
    *,
    kind: str,
    message: str,
    payload: Dict[str, Any],
    subsystem: str,
    pipeline_stage: str,
) -> str:
    reasoning = _first_payload_text(
        payload,
        [
            "reasoning",
            "reason",
            "bounded_reason",
            "diagnostic",
            "route_reason",
            "hypothesis",
            "possible_experiment",
            "bridge",
            "open_question",
        ],
    )
    content = _first_payload_text(
        payload,
        [
            "content_preview",
            "summary",
            "synthesis",
            "interpretation",
            "recollection",
            "spark_content",
            "query",
            "objective",
            "message",
            "result",
            "status",
        ],
    )
    tool_name = _safe_str(payload.get("tool") or payload.get("tool_name") or payload.get("name"))
    model = _safe_str(payload.get("model") or payload.get("model_id") or payload.get("resolved_model"))
    if kind == EventKind.MEMORY_ACTIVATED.value:
        rank = _safe_str(payload.get("rank"))
        score = _safe_str(payload.get("score"))
        parts = ["Accessed memory/context evidence"]
        if rank:
            parts.append(f"rank {rank}")
        if score:
            parts.append(f"score {score}")
        if content:
            parts.append(content)
        if reasoning:
            parts.append(f"why: {reasoning}")
        return _compact_flow_text("; ".join(parts), limit=1100)
    if kind == EventKind.SELF_APPROVAL.value:
        action = _safe_str(payload.get("tool_name") or payload.get("action_id") or tool_name)
        level = _safe_str(payload.get("level") or payload.get("tier"))
        base = f"Self-approval decision for {action or 'action'}"
        if level:
            base += f": {level}"
        if reasoning:
            base += f"; {reasoning}"
        return _compact_flow_text(base, limit=1100)
    if kind == EventKind.TOOL_CALL.value:
        status = _safe_str(payload.get("status") or payload.get("result") or "")
        lower_message = message.lower()
        lower_reason = reasoning.lower()
        if "task_held_executive_pause" in lower_message:
            detail = "Held bounded-assistant work because executive load is too high"
            if reasoning:
                detail += f": {reasoning}"
            return _compact_flow_text(detail, limit=1100)
        if "cycle_backoff" in lower_message:
            detail = "Backed off an autonomous work cycle"
            if reasoning:
                detail += f" because {reasoning}"
            return _compact_flow_text(detail, limit=1100)
        if "desktop_context_skipped" in lower_message:
            detail = "Skipped desktop-context observation"
            if reasoning:
                detail += f" because {reasoning}"
            return _compact_flow_text(detail, limit=1100)
        if "schedule_triggered" in lower_message and status.lower() in {"skipped", "skip"}:
            detail = "Skipped scheduled work"
            if reasoning:
                detail += f" because {reasoning}"
            return _compact_flow_text(detail, limit=1100)
        if reasoning and status:
            return _compact_flow_text(
                f"Tool activity {tool_name or message}: {status}; reason: {reasoning}",
                limit=1100,
            )
        if lower_reason and lower_reason not in lower_message:
            return _compact_flow_text(
                f"Tool activity {tool_name or message}; reason: {reasoning}",
                limit=1100,
            )
        return _compact_flow_text(
            f"Tool activity {tool_name or message}: {status or content or message}",
            limit=1100,
        )
    if kind == EventKind.LLM_CALL.value:
        source = _safe_str(payload.get("source"))
        latency = _safe_str(payload.get("latency_ms"))
        detail = f"LLM lane {model or 'selected model'}"
        if source:
            detail += f" for {source}"
        if latency:
            detail += f"; latency {latency} ms"
        return _compact_flow_text(detail, limit=1100)
    if reasoning and content:
        return _compact_flow_text(f"{content}; reasoning evidence: {reasoning}", limit=1100)
    if reasoning:
        return _compact_flow_text(reasoning, limit=1100)
    if content:
        return _compact_flow_text(content, limit=1100)
    return _compact_flow_text(f"{subsystem} {pipeline_stage}: {message}", limit=1100)


def _flow_lane_label(
    *,
    kind: str,
    message: str,
    subsystem: str,
    pipeline_stage: str,
    is_noise: bool,
) -> str:
    text = f"{kind} {message} {subsystem} {pipeline_stage}".lower()
    if is_noise:
        return "background telemetry"
    if "memory" in text or "retriev" in text or "recall" in text:
        return "memory and recall"
    if "daydream" in text or "reflection" in text or "proposal" in text:
        return "reflection and daydream"
    if "self-approval" in text or "approval" in text or "decision" in text or "gate" in text:
        return "decision and approval"
    if "schedule" in text or "initiative" in text or "commitment" in text:
        return "scheduling and initiative"
    if "tool" in text or kind == EventKind.TOOL_CALL.value:
        return "tools and execution"
    if kind == EventKind.LLM_CALL.value:
        return "model reasoning"
    if "tom" in text or "relational" in text or "musubi" in text:
        return "social model"
    if subsystem:
        return subsystem.replace("_", " ")
    return pipeline_stage.replace("_", " ") or "runtime"


def _span_depth_for_id(
    span_id: str,
    span_parent: Dict[str, str],
    span_depth_cache: Dict[str, int],
) -> int:
    if not span_id:
        return 0
    cached = span_depth_cache.get(span_id)
    if cached is not None:
        return cached

    visited: set[str] = set()
    current = span_id
    depth = 0
    while current in span_parent:
        if current in visited:
            break
        visited.add(current)
        current = span_parent[current]
        if not current:
            break
        depth += 1
        cached = span_depth_cache.get(current)
        if cached is not None:
            depth = cached + 1
            break

    span_depth_cache[span_id] = depth
    return depth


class _CognitiveFlowCache:
    """Short-lived single-flight cache for bursty cognitive-flow polls."""

    def __init__(self, ttl_seconds: float = 0.75) -> None:
        self.ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
        self._key: Tuple[int, int, str, str, str] | None = None
        self._created_at = 0.0
        self._response: CognitiveFlowResponse | None = None

    def _fresh(self, key: Tuple[int, int, str, str, str]) -> bool:
        return (
            self._response is not None
            and self._key == key
            and (time.monotonic() - self._created_at) <= self.ttl_seconds
        )

    async def get(
        self,
        key: Tuple[int, int, str, str, str],
        builder: Callable[[], CognitiveFlowResponse | Awaitable[CognitiveFlowResponse]],
    ) -> CognitiveFlowResponse:
        if self._fresh(key):
            return self._response  # type: ignore[return-value]
        async with self._lock:
            if self._fresh(key):
                return self._response  # type: ignore[return-value]
            response = builder()
            if hasattr(response, "__await__"):
                response = await response
            self._key = key
            self._created_at = time.monotonic()
            self._response = response
            return response


def _cognitive_flow_node(event: Any, *, now: datetime, span_depth: int) -> Dict[str, Any]:
    timestamp = getattr(event, "timestamp", now)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    payload = dict(getattr(event, "payload", {}) or {})
    event_id = str(getattr(event, "event_id", ""))
    kind = _enum_value(getattr(event, "kind", ""))
    message = str(getattr(event, "message", "") or "")
    activation_source = str(payload.get("activation_source") or "")
    source_type = _safe_str(payload.get("source_type") or "telemetry")
    source_id = _safe_str(payload.get("source_id") or "")
    node_ref = _safe_str(payload.get("node_id") or (f"{source_type}:{source_id}" if source_type and source_id else ""))

    raw_label = payload.get("name") or getattr(event, "message", "")
    if not raw_label:
        raw_label = payload.get("operation") or payload.get("operation_name") or payload.get("stage") or kind or event_id
    subsystem = _infer_subsystem(kind=kind, message=message, payload=payload)
    pipeline_stage = _infer_pipeline_stage(kind=kind, message=message, payload=payload)
    decision_signal = _infer_decision_signal(kind=kind, message=message, payload=payload)
    is_noise = _is_low_information_flow_event(kind=kind, message=message, payload=payload)
    priority = _flow_priority(
        kind=kind,
        message=message,
        payload=payload,
        decision_signal=decision_signal,
        is_noise=is_noise,
    )
    reasoning_text = _first_payload_text(
        payload,
        [
            "reasoning",
            "reason",
            "bounded_reason",
            "diagnostic",
            "route_reason",
            "hypothesis",
            "possible_experiment",
            "bridge",
            "open_question",
        ],
    )
    thought_text = _observable_thought_text(
        kind=kind,
        message=message,
        payload=payload,
        subsystem=subsystem,
        pipeline_stage=pipeline_stage,
    )
    source_label = _safe_str(payload.get("source") or payload.get("source_name"))
    operation = _safe_str(payload.get("operation") or payload.get("operation_name") or payload.get("stage"))
    flow_lane = _flow_lane_label(
        kind=kind,
        message=message,
        subsystem=subsystem,
        pipeline_stage=pipeline_stage,
        is_noise=is_noise,
    )

    return _with_temporal_metadata({
        "event_id": event_id,
        "node_id": event_id,
        "kind": kind,
        "label": str(raw_label)[:240],
        "message": str(getattr(event, "message", "") or "")[:420],
        "timestamp": timestamp.isoformat(),
        "session_id": _safe_str(getattr(event, "session_id", None)),
        "span_id": _safe_str(getattr(event, "span_id", None)),
        "parent_span_id": _safe_str(getattr(event, "parent_span_id", None)),
        "event_message": message,
        "subsystem": subsystem,
        "pipeline_stage": pipeline_stage,
        "decision_signal": decision_signal,
        "flow_priority": priority,
        "is_noise": is_noise,
        "flow_lane": flow_lane,
        "lane_label": flow_lane,
        "thought_text": thought_text,
        "reasoning_text": reasoning_text,
        "evidence_text": _first_payload_text(
            payload,
            ["content_preview", "query", "source_id", "node_id", "referenced_event_id"],
        ),
        "continuity_backlink": _safe_str(getattr(event, "continuity_backlink", None)),
        "noted_as_such": bool(getattr(event, "noted_as_such", False)),
        "activation_timestamp": (
            getattr(event, "activated_at", None).isoformat()
            if getattr(event, "activated_at", None) is not None
            else ""
        ),
        "content_preview": str(payload.get("content_preview") or payload.get("query") or activation_source or ""),
        "node_ref": node_ref,
        "source_type": source_type,
        "source_id": source_id,
        "source_label": source_label,
        "activation_source": activation_source or "telemetry",
        "referenced_event_id": _safe_str(payload.get("referenced_event_id") or ""),
        "operation": operation,
        "span_depth": span_depth,
        "is_span_gate": kind in {EventKind.SPAN_START.value, EventKind.SPAN_END.value},
        "event_payload": {
            "name": _safe_str(payload.get("name") or ""),
            "operation": operation,
            "tool": _safe_str(payload.get("tool") or payload.get("tool_name") or ""),
            "result": _safe_str(payload.get("result") or payload.get("status") or ""),
            "payload": _truncate_payload_dict(payload),
        },
    }, timestamp=timestamp, source="telemetry_event.timestamp", now=now)


def _truncate_payload_dict(payload: Dict[str, Any], max_items: int = 40) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    reduced = []
    for index, item in enumerate(payload.items()):
        if index >= max_items:
            break
        reduced.append(item)
    return dict(reduced)


class _MemoryActivityCache:
    """Short-lived single-flight cache for bursty dashboard activity polls."""

    def __init__(self, ttl_seconds: float = 0.75) -> None:
        self.ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
        self._key: Tuple[int, int] | None = None
        self._created_at = 0.0
        self._response: MemoryActivityResponse | None = None

    def _fresh(self, key: Tuple[int, int]) -> bool:
        return (
            self._response is not None
            and self._key == key
            and (time.monotonic() - self._created_at) <= self.ttl_seconds
        )

    async def get(
        self,
        key: Tuple[int, int],
        builder: Callable[[], MemoryActivityResponse | Awaitable[MemoryActivityResponse]],
    ) -> MemoryActivityResponse:
        if self._fresh(key):
            return self._response  # type: ignore[return-value]
        async with self._lock:
            if self._fresh(key):
                return self._response  # type: ignore[return-value]
            response = builder()
            if hasattr(response, "__await__"):
                response = await response
            self._key = key
            self._created_at = time.monotonic()
            self._response = response
            return response


def _context_proposal_store(runtime: Any) -> Any:
    return getattr(runtime, "context_proposals", None) or getattr(
        getattr(runtime, "ctx", None),
        "context_proposal_store",
        None,
    )


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _proposal_node(proposal: Any, now: datetime) -> Dict[str, Any]:
    created_at = getattr(proposal, "created_at", now)
    updated_at = getattr(proposal, "updated_at", created_at)
    age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    status = _enum_value(getattr(proposal, "status", ""))
    authority = _enum_value(getattr(proposal, "authority", ""))
    content = str(getattr(proposal, "content", "") or "")
    confidence = float(getattr(proposal, "confidence", 0.5) or 0.0)
    can_authorize = status == "accepted" and authority == "executive_committed"
    return _with_temporal_metadata({
        "node_id": f"context_proposal:{getattr(proposal, 'proposal_id', '')}",
        "node_type": "context_proposal",
        "proposal_id": str(getattr(proposal, "proposal_id", "")),
        "label": truncate_memory_text(content, 72),
        "content": content,
        "created_at": created_at.isoformat(),
        "updated_at": updated_at.isoformat(),
        "age_days": round(age_days, 3),
        "kind": "context_proposal",
        "proposal_kind": str(getattr(proposal, "proposal_kind", "") or "context_proposal"),
        "proposal_status": status,
        "source_lane": _enum_value(getattr(proposal, "source_lane", "")),
        "source_lane_inferred": False,
        "lane_source": "proposal_store",
        "source_snapshot_id": str(getattr(proposal, "source_snapshot_id", "") or ""),
        "source_epoch": int(getattr(proposal, "source_epoch", 0) or 0),
        "authority": authority,
        "context_material": "context_proposal",
        "can_authorize_work": can_authorize,
        "project_id": getattr(proposal, "project_id", None),
        "commitment_id": getattr(proposal, "commitment_id", None),
        "schedule_id": getattr(proposal, "schedule_id", None),
        "task_id": getattr(proposal, "task_id", None),
        "evidence_refs": list(getattr(proposal, "evidence_refs", []) or []),
        "validation": dict(getattr(proposal, "validation", {}) or {}),
        "session_id": None,
        "salience": None,
        "confidence_score": confidence,
        "proposal_confidence": confidence,
        "somatic_tag": "reflective",
        "compacted": False,
        "identity_core": False,
        "used_successfully": 0,
        "used_unsuccessfully": 0,
        "affect": None,
        "embedding_id": None,
        "connection_count": 0,
        "dual_context": {
            "source_lane": _enum_value(getattr(proposal, "source_lane", "")),
            "source_lane_inferred": False,
            "lane_source": "proposal_store",
            "authority": authority,
            "context_material": "context_proposal",
            "can_authorize_work": can_authorize,
            "source_snapshot_id": str(getattr(proposal, "source_snapshot_id", "") or ""),
            "source_epoch": int(getattr(proposal, "source_epoch", 0) or 0),
            "proposal_status": status,
        },
    }, timestamp=created_at, source="created_at", now=now)


async def _load_context_proposals(
    runtime: Any,
    *,
    query: Optional[str],
    limit: int,
) -> tuple[List[Any], int]:
    applied_limit = min(200, max(8, int(limit)))
    store = _context_proposal_store(runtime)
    if store is None:
        return [], applied_limit
    if query:
        search_text = getattr(store, "search_text", None)
        if callable(search_text):
            return list(await search_text(query, include_terminal=True, limit=applied_limit)), applied_limit
    list_recent = getattr(store, "list_recent", None)
    if callable(list_recent):
        return list(await list_recent(limit=applied_limit)), applied_limit
    return [], applied_limit


def _proposal_evidence_node_ids(proposal: Any, node_map: Dict[str, Dict[str, Any]]) -> List[str]:
    node_ids: List[str] = []
    for raw_ref in list(getattr(proposal, "evidence_refs", []) or []):
        ref = str(raw_ref or "").strip()
        if not ref:
            continue
        candidates = [ref]
        if ref.startswith("daydream_reflection:"):
            candidates.append(f"context_proposal:{ref}")
        if not ref.startswith(("episode:", "memory:", "context_proposal:")):
            candidates.extend([f"episode:{ref}", f"memory:{ref}"])
        for candidate in candidates:
            if candidate in node_map and candidate not in node_ids:
                node_ids.append(candidate)
                break
    return node_ids


def _proposal_edge_payload(proposal: Any, source_node_id: str, target_node_id: str) -> Dict[str, Any]:
    confidence = float(getattr(proposal, "confidence", 0.5) or 0.0)
    created_at = getattr(proposal, "created_at", datetime.now(timezone.utc))
    return {
        "edge_id": f"proposal-evidence:{getattr(proposal, 'proposal_id', '')}:{target_node_id}",
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "kind": "proposal_evidence",
        "confidence": round(confidence, 6),
        "strength": round(max(0.2, min(1.0, confidence)), 6),
        "created_at": created_at.isoformat(),
        "time_distance_days": None,
        "weights": {"proposal_confidence": confidence},
        "synthetic": True,
    }


def _memory_activity_payload(event: Any, now: datetime, window_seconds: int) -> Dict[str, Any] | None:
    payload = dict(getattr(event, "payload", {}) or {})
    node_id = str(payload.get("node_id") or "").strip()
    source_type = str(payload.get("source_type") or "").strip()
    source_id = str(payload.get("source_id") or "").strip()
    if not node_id and source_type and source_id:
        node_id = f"{source_type}:{source_id}"
    if not node_id:
        return None
    timestamp = getattr(event, "activated_at", None) or getattr(event, "timestamp", None) or now
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    age_seconds = max(0.0, (now - timestamp).total_seconds())
    if age_seconds > window_seconds:
        return None
    base_intensity = max(0.0, 1.0 - (age_seconds / max(1, window_seconds)))
    score = payload.get("score")
    if score is not None:
        base_intensity = max(base_intensity, min(1.0, float(score) * 0.75))
    kind = _enum_value(getattr(event, "kind", ""))
    source_key = "activated_at" if getattr(event, "activated_at", None) is not None else "timestamp"
    return _with_temporal_metadata({
        "event_id": str(getattr(event, "event_id", "")),
        "timestamp": timestamp.isoformat(),
        "age_seconds": round(age_seconds, 3),
        "event_kind": kind,
        "node_id": node_id,
        "source_type": source_type or node_id.split(":", 1)[0],
        "source_id": source_id or node_id.split(":", 1)[-1],
        "activation_source": str(payload.get("activation_source") or kind or "memory"),
        "source_lane": str(payload.get("source_lane") or payload.get("origin_context_lane") or ""),
        "source_lane_inferred": bool(payload.get("source_lane_inferred", False)),
        "lane_source": str(payload.get("lane_source") or "telemetry_payload"),
        "authority": str(payload.get("authority") or payload.get("context_authority") or ""),
        "context_material": str(payload.get("context_material") or ""),
        "query": payload.get("query"),
        "score": score,
        "rank": payload.get("rank"),
        "content_preview": payload.get("content_preview"),
        "intensity": round(min(1.0, base_intensity), 3),
        "noted_as_such": bool(getattr(event, "noted_as_such", False)),
    }, timestamp=timestamp, source=source_key, now=now)


def _activity_context_fields(node: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_lane": str(node.get("source_lane") or ""),
        "source_lane_inferred": bool(node.get("source_lane_inferred", False)),
        "lane_source": str(node.get("lane_source") or ""),
        "authority": str(node.get("authority") or ""),
        "context_material": str(node.get("context_material") or ""),
        "content_preview": node.get("content_preview") or node.get("label") or node.get("content"),
    }


async def _activity_metadata_for_nodes(
    runtime: Any,
    node_ids: List[str],
    now: datetime,
) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    requested = {str(node_id or "").strip() for node_id in node_ids}
    requested.discard("")
    if not requested:
        return metadata

    episode_ids = [node_id.split(":", 1)[1] for node_id in requested if node_id.startswith("episode:")]
    memory_ids = [node_id.split(":", 1)[1] for node_id in requested if node_id.startswith("memory:")]
    proposal_ids = [
        node_id.split(":", 1)[1]
        for node_id in requested
        if node_id.startswith("context_proposal:")
    ]

    memory = getattr(runtime, "memory", None)
    get_episodes = getattr(memory, "get_episodes_by_ids", None)
    if callable(get_episodes) and episode_ids:
        try:
            for episode in await get_episodes(episode_ids):
                node_id = f"episode:{getattr(episode, 'episode_id', '')}"
                metadata[node_id] = _activity_context_fields(episode_to_dict(episode))
        except Exception:
            pass

    get_memory = getattr(memory, "get_memory", None)
    if callable(get_memory) and memory_ids:
        for memory_id in memory_ids:
            try:
                memory_item = await get_memory(memory_id)
            except Exception:
                memory_item = None
            if memory_item is not None:
                metadata[f"memory:{memory_id}"] = _activity_context_fields(memory_to_dict(memory_item))

    proposal_store = _context_proposal_store(runtime)
    get_proposal = getattr(proposal_store, "get", None)
    if callable(get_proposal) and proposal_ids:
        for proposal_id in proposal_ids:
            try:
                proposal = await get_proposal(proposal_id)
            except Exception:
                proposal = None
            if proposal is not None:
                metadata[f"context_proposal:{proposal_id}"] = _activity_context_fields(
                    _proposal_node(proposal, now)
                )

    return metadata


def _enrich_activity_events(
    events: List[Dict[str, Any]],
    metadata_by_node: Dict[str, Dict[str, Any]],
) -> None:
    for event in events:
        meta = metadata_by_node.get(str(event.get("node_id") or ""))
        if not meta:
            continue
        if not event.get("source_lane") and meta.get("source_lane"):
            event["source_lane"] = meta["source_lane"]
            event["source_lane_inferred"] = True
            event["lane_source"] = meta.get("lane_source") or "node_metadata"
        elif not event.get("lane_source"):
            event["lane_source"] = "telemetry_payload"
        for key in ("authority", "context_material", "content_preview"):
            if not event.get(key) and meta.get(key):
                event[key] = meta[key]


def build_memory_router(runtime: Any) -> APIRouter:
    """Build memory routes wired to *runtime*."""
    r = APIRouter(prefix="/api/memory", tags=["memory"])
    activity_cache = _MemoryActivityCache()
    cognitive_flow_cache = _CognitiveFlowCache()

    @r.get("/episodes", response_model=EpisodesResponse)
    async def list_episodes(limit: int = 50, offset: int = 0) -> EpisodesResponse:
        mem = runtime.memory
        episodes = await mem.list_episodes(limit=limit, offset=offset)
        total = await get_total_episode_count(mem)
        return EpisodesResponse(
            episodes=[episode_to_dict(ep) for ep in episodes],
            total=total,
        )

    @r.get("/graph", response_model=GraphResponse)
    async def get_graph(episode_id: str, limit: int = 24) -> GraphResponse:
        graph = runtime.episode_graph
        edges = await graph.get_neighbors(episode_id, limit=limit)
        seen = {episode_id}
        node_ids = [episode_id]
        for edge in edges:
            other = edge.target_id if edge.source_id == episode_id else edge.source_id
            if other not in seen:
                seen.add(other)
                node_ids.append(other)
        episodes = await runtime.memory.get_episodes_by_ids(node_ids)
        return GraphResponse(
            episode_id=episode_id,
            nodes=[episode_to_dict(ep) for ep in episodes],
            edges=[edge_to_dict(e) for e in edges],
        )

    @r.get("/stats", response_model=MemoryStatsResponse)
    async def get_memory_stats() -> MemoryStatsResponse:
        stats = await runtime.memory.get_stats()
        return MemoryStatsResponse(
            episode_count=stats["episode_count"],
            memory_count=stats["memory_count"],
            edge_count=stats["edge_count"],
            compacted_count=stats["compacted_count"],
            identity_core_count=stats["identity_core_count"],
            avg_salience=stats["avg_salience"],
            affect_distribution=stats["affect_distribution"],
        )

    @r.get("/search", response_model=SearchResponse)
    async def search_memories(query: str, limit: int = 20) -> SearchResponse:
        episodes = await runtime.memory.search_episodes_by_content(query, limit=limit)
        return SearchResponse(
            query=query,
            results=[episode_to_dict(ep) for ep in episodes],
        )

    @r.get("/activity", response_model=MemoryActivityResponse)
    async def memory_activity(
        window_seconds: int = 180,
        limit: int = 80,
    ) -> MemoryActivityResponse:
        window_seconds = max(5, min(int(window_seconds), 3600))
        limit = max(1, min(int(limit), 300))

        async def _build_response() -> MemoryActivityResponse:
            now = datetime.now(timezone.utc)
            tracer = getattr(runtime, "tracer", None)
            store = getattr(tracer, "store", None)
            if store is None:
                return MemoryActivityResponse(
                    generated_at=now.isoformat(),
                    window_seconds=window_seconds,
                    event_count=0,
                    active_node_ids=[],
                    active_nodes=[],
                    events=[],
                    stats={"available": False},
                )
            events = store.query(
                kinds=[EventKind.MEMORY_ACTIVATED, EventKind.MEMORY_WRITE, EventKind.MEMORY_COMPACT],
                limit=max(limit * 4, limit),
            )
            activity_events = [
                item
                for event in events
                if (item := _memory_activity_payload(event, now, window_seconds)) is not None
            ][-limit:]
            metadata_by_node = await _activity_metadata_for_nodes(
                runtime,
                [
                    event["node_id"]
                    for event in activity_events
                    if not event.get("source_lane")
                    or not event.get("authority")
                    or not event.get("context_material")
                    or not event.get("content_preview")
                ],
                now,
            )
            _enrich_activity_events(activity_events, metadata_by_node)
            aggregate: Dict[str, Dict[str, Any]] = {}
            for event in activity_events:
                node_id = event["node_id"]
                current = aggregate.setdefault(
                    node_id,
                    {
                        "node_id": node_id,
                        "source_type": event["source_type"],
                        "source_id": event["source_id"],
                        "activation_source": event["activation_source"],
                        "event_count": 0,
                        "intensity": 0.0,
                        "last_activated_at": event["timestamp"],
                        "source_lane": event.get("source_lane"),
                        "source_lane_inferred": event.get("source_lane_inferred", False),
                        "lane_source": event.get("lane_source"),
                        "authority": event.get("authority"),
                        "context_material": event.get("context_material"),
                        "query": event.get("query"),
                        "content_preview": event.get("content_preview"),
                    },
                )
                current["event_count"] += 1
                current["intensity"] = min(1.0, float(current["intensity"]) + float(event["intensity"]))
                if event["timestamp"] >= str(current["last_activated_at"]):
                    current["last_activated_at"] = event["timestamp"]
                    current["query"] = event.get("query")
                    current["content_preview"] = event.get("content_preview")
                    current["activation_source"] = event["activation_source"]
                    current["source_lane"] = event.get("source_lane")
                    current["source_lane_inferred"] = event.get("source_lane_inferred", False)
                    current["lane_source"] = event.get("lane_source")
                    current["authority"] = event.get("authority")
                    current["context_material"] = event.get("context_material")
            active_nodes = sorted(
                (
                    _with_temporal_metadata(
                        {**node, "intensity": round(float(node["intensity"]), 3)},
                        timestamp=node.get("last_activated_at"),
                        source="last_activated_at",
                        now=now,
                    )
                    for node in aggregate.values()
                ),
                key=lambda item: (float(item["intensity"]), str(item["last_activated_at"])),
                reverse=True,
            )
            return MemoryActivityResponse(
                generated_at=now.isoformat(),
                window_seconds=window_seconds,
                event_count=len(activity_events),
                active_node_ids=[node["node_id"] for node in active_nodes],
                active_nodes=active_nodes,
                events=sorted(activity_events, key=lambda item: item["timestamp"], reverse=True),
                stats={
                    "available": True,
                    "active_node_count": len(active_nodes),
                    "event_kind_distribution": dict(Counter(event["event_kind"] for event in activity_events)),
                    "activation_source_distribution": dict(Counter(event["activation_source"] for event in activity_events)),
                    "source_lane_distribution": dict(
                        Counter(event.get("source_lane") or "unknown" for event in activity_events)
                    ),
                    "context_material_distribution": dict(
                        Counter(event.get("context_material") or "unknown" for event in activity_events)
                    ),
                },
            )

        return await activity_cache.get((window_seconds, limit), _build_response)

    @r.get("/cognitive-flow", response_model=CognitiveFlowResponse)
    async def cognitive_flow(
        window_seconds: int = 180,
        limit: int = 180,
        session_id: Optional[str] = None,
        kind_filter: Optional[str] = None,
        include_noise: bool = True,
        show_span_flow: bool = True,
        show_sequential_links: bool = True,
    ) -> CognitiveFlowResponse:
        window_seconds = max(5, min(int(window_seconds), 7200))
        limit = max(20, min(int(limit), 600))

        requested_kinds: Optional[List[EventKind]] = None
        if kind_filter:
            requested_kinds = []
            for raw_kind in str(kind_filter).split(","):
                normalized = raw_kind.strip()
                if not normalized:
                    continue
                try:
                    requested_kinds.append(EventKind(normalized))
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported kind_filter value: {normalized}",
                    ) from None
            if not requested_kinds:
                requested_kinds = None

        cache_key = (
            window_seconds,
            limit,
            session_id or "",
            kind_filter or "",
            f"{include_noise}:{show_span_flow}:{show_sequential_links}",
        )

        async def _build_response() -> CognitiveFlowResponse:
            now = datetime.now(timezone.utc)
            tracer = getattr(runtime, "tracer", None)
            store = getattr(tracer, "store", None)
            window_start = now - timedelta(seconds=window_seconds)
            if store is None:
                return CognitiveFlowResponse(
                    generated_at=now.isoformat(),
                    window_seconds=window_seconds,
                    event_count=0,
                    span_count=0,
                    nodes=[],
                    edges=[],
                    stats={"available": False, "reason": "telemetry_store_missing"},
                )

            events = store.query(
                kinds=requested_kinds,
                session_id=session_id,
                limit=max(limit * 8, limit),
            )
            timed_events: List[Tuple[datetime, Any]] = []
            for event in events:
                event_timestamp_value = _safe_event_timestamp(
                    getattr(event, "timestamp", None),
                    now,
                )
                if event_timestamp_value >= window_start:
                    timed_events.append((event_timestamp_value, event))
            timed_events.sort(key=lambda item: item[0])
            candidate_event_count = len(timed_events)
            excluded_noise_count = 0
            if not include_noise:
                without_noise = [
                    event_item
                    for event_item in timed_events
                    if not _flow_event_is_noise(event_item[1])
                ]
                excluded_noise_count = len(timed_events) - len(without_noise)
                timed_events = without_noise
            if len(timed_events) > limit:
                ranked_events = sorted(
                    enumerate(timed_events),
                    key=lambda item: (_flow_event_selection_score(item[1][1]), item[1][0]),
                    reverse=True,
                )
                selected_indexes = {index for index, _event_item in ranked_events[:limit]}
                timed_events = [
                    event_item
                    for index, event_item in enumerate(timed_events)
                    if index in selected_indexes
                ]

            span_parent_by_id: Dict[str, str] = {}
            event_timestamp: Dict[str, datetime] = {}
            for event_timestamp_value, event in timed_events:
                event_id = str(getattr(event, "event_id", ""))
                if not event_id:
                    continue
                event_timestamp[event_id] = event_timestamp_value
                span_id = _safe_str(getattr(event, "span_id", None))
                if not span_id:
                    continue
                parent_span_id = _safe_str(getattr(event, "parent_span_id", None))
                span_parent_by_id.setdefault(span_id, parent_span_id)

            span_depth_cache: Dict[str, int] = {}
            nodes: List[Dict[str, Any]] = []
            event_by_id: Dict[str, Dict[str, Any]] = {}
            for event_timestamp_value, event in timed_events:
                event_id = str(getattr(event, "event_id", ""))
                if not event_id:
                    continue
                span_id = _safe_str(getattr(event, "span_id", None))
                span_depth = _span_depth_for_id(span_id, span_parent_by_id, span_depth_cache)
                node = _cognitive_flow_node(event, now=now, span_depth=span_depth)
                node["timestamp"] = event_timestamp_value.isoformat()
                event_by_id[event_id] = node
                nodes.append(node)

            lane_order: Dict[str, int] = {}
            for node in nodes:
                flow_lane = _safe_str(
                    node.get("flow_lane")
                    or node.get("lane_label")
                    or node.get("subsystem")
                    or "runtime"
                )
                if flow_lane not in lane_order:
                    lane_order[flow_lane] = len(lane_order)
                node["flow_lane"] = flow_lane
                node["lane_label"] = flow_lane
                node["lane_index"] = lane_order[flow_lane]

            event_ids = set(event_by_id.keys())
            edges: List[Dict[str, Any]] = []
            edge_ids: set[str] = set()

            def _add_flow_edge(
                *,
                source_node_id: str,
                target_node_id: str,
                kind: str,
                created_at: Any,
                strength: float,
                weight_kind: str,
            ) -> None:
                if not source_node_id or not target_node_id or source_node_id == target_node_id:
                    return
                if source_node_id not in event_by_id or target_node_id not in event_by_id:
                    return
                edge_id = f"{kind}:{source_node_id}:{target_node_id}"
                if edge_id in edge_ids:
                    return
                edge_timestamp = _safe_event_timestamp(created_at, now)
                edges.append(
                    _with_temporal_metadata(
                        {
                            "edge_id": edge_id,
                            "source_node_id": source_node_id,
                            "target_node_id": target_node_id,
                            "kind": kind,
                            "created_at": edge_timestamp.isoformat(),
                            "strength": strength,
                            "weights": {"kind": weight_kind},
                        },
                        timestamp=edge_timestamp,
                        source="created_at",
                        now=now,
                    )
                )
                edge_ids.add(edge_id)

            if show_sequential_links:
                by_span_latest: Dict[str, str] = {}
                by_lane_latest: Dict[str, str] = {}
                previous_high_signal: str = ""
                for node in nodes:
                    node_id = str(node.get("event_id") or "")
                    if not node_id:
                        continue
                    span_id = str(node.get("span_id") or "")
                    if not span_id:
                        lane_key = str(node.get("flow_lane") or node.get("subsystem") or "runtime")
                        previous_lane = by_lane_latest.get(lane_key)
                        if previous_lane:
                            _add_flow_edge(
                                source_node_id=previous_lane,
                                target_node_id=node_id,
                                kind="lane_order",
                                created_at=max(
                                    event_timestamp.get(previous_lane, now),
                                    event_timestamp.get(node_id, now),
                                ),
                                strength=0.38 if node.get("is_noise") else 0.55,
                                weight_kind="lane_temporal",
                            )
                        by_lane_latest[lane_key] = node_id
                    else:
                        previous = by_span_latest.get(span_id)
                        if previous:
                            _add_flow_edge(
                                source_node_id=previous,
                                target_node_id=node_id,
                                kind="flow_order",
                                created_at=max(
                                    event_timestamp.get(previous, now),
                                    event_timestamp.get(node_id, now),
                                ),
                                strength=0.65,
                                weight_kind="temporal",
                            )
                        by_span_latest[span_id] = node_id
                        lane_key = str(node.get("flow_lane") or node.get("subsystem") or "runtime")
                        by_lane_latest[lane_key] = node_id

                    high_signal = (
                        not bool(node.get("is_noise"))
                        and float(node.get("flow_priority") or 0.0) >= 0.25
                    )
                    if high_signal:
                        if previous_high_signal:
                            _add_flow_edge(
                                source_node_id=previous_high_signal,
                                target_node_id=node_id,
                                kind="cognitive_sequence",
                                created_at=max(
                                    event_timestamp.get(previous_high_signal, now),
                                    event_timestamp.get(node_id, now),
                                ),
                                strength=0.72,
                                weight_kind="global_temporal",
                            )
                        previous_high_signal = node_id

            if show_span_flow:
                child_span_started: set[str] = set()
                first_event_in_span: Dict[str, str] = {}
                for node in nodes:
                    span_id = str(node.get("span_id") or "")
                    if not span_id:
                        continue
                    node_id = str(node.get("event_id") or "")
                    if not node_id:
                        continue
                    if span_id not in first_event_in_span:
                        first_event_in_span[span_id] = str(node.get("event_id") or "")
                for node in nodes:
                    parent_span = str(node.get("parent_span_id") or "")
                    if not parent_span:
                        continue
                    if parent_span not in first_event_in_span:
                        continue
                    child_span = str(node.get("span_id") or "")
                    if not child_span or child_span == parent_span:
                        continue
                    child_key = f"{parent_span}:{child_span}"
                    if child_key in child_span_started:
                        continue
                    source_node = first_event_in_span.get(parent_span)
                    target_node = first_event_in_span.get(child_span)
                    if not source_node or not target_node:
                        continue
                    _add_flow_edge(
                        source_node_id=source_node,
                        target_node_id=target_node,
                        kind="span_hierarchy",
                        created_at=node.get("timestamp"),
                        strength=0.75,
                        weight_kind="span",
                    )
                    child_span_started.add(child_key)

            for node in nodes:
                continuity = str(node.get("continuity_backlink") or "")
                if continuity and continuity in event_ids and continuity != node["event_id"]:
                    _add_flow_edge(
                        source_node_id=continuity,
                        target_node_id=node["event_id"],
                        kind="continuity_backlink",
                        created_at=node.get("timestamp"),
                        strength=0.9,
                        weight_kind="continuity",
                    )
                if node.get("kind") == EventKind.ACTION_BACKLINK.value:
                    memory_event_id = str(_safe_str(node.get("event_payload", {}).get("memory_event_id")))
                    if memory_event_id and memory_event_id in event_ids:
                        _add_flow_edge(
                            source_node_id=memory_event_id,
                            target_node_id=node["event_id"],
                            kind="action_backlink",
                            created_at=node.get("timestamp"),
                            strength=0.9,
                            weight_kind="action",
                        )

            kind_distribution = Counter(event.get("kind") for event in nodes)
            session_distribution = Counter(event.get("session_id") or "unknown" for event in nodes)
            span_depth_distribution = Counter(
                int(event.get("span_depth", 0) or 0) for event in nodes
            )
            lane_distribution = Counter(_safe_str(event.get("flow_lane") or "runtime") for event in nodes)
            event_ids_with_source = Counter(_safe_str(event.get("source_type") or "unknown") for event in nodes)
            subsystem_distribution = Counter(_safe_str(event.get("subsystem") or "unknown") for event in nodes)
            stage_distribution = Counter(
                _safe_str(event.get("pipeline_stage") or "processing") for event in nodes
            )
            decision_count = sum(1 for event in nodes if bool(event.get("decision_signal")))
            nodes = sorted(nodes, key=lambda item: str(item.get("timestamp")))
            for sequence_index, node in enumerate(nodes, start=1):
                node["sequence_index"] = sequence_index
                node["sequence_total"] = len(nodes)
                node["temporal_order_label"] = f"{sequence_index}/{len(nodes)}"
            edges.sort(key=lambda item: _safe_event_timestamp(item.get("created_at"), now))
            temporal_missing_node_count = sum(
                1 for event in nodes if not bool(event.get("temporal_has_metadata"))
            )
            temporal_missing_edge_count = sum(
                1 for edge in edges if not bool(edge.get("temporal_has_metadata"))
            )
            readable_thoughts = [
                {
                    "event_id": event.get("event_id"),
                    "timestamp": event.get("timestamp"),
                    "sequence_index": event.get("sequence_index"),
                    "temporal_order_label": event.get("temporal_order_label"),
                    "flow_lane": event.get("flow_lane"),
                    "subsystem": event.get("subsystem"),
                    "pipeline_stage": event.get("pipeline_stage"),
                    "kind": event.get("kind"),
                    "label": event.get("label"),
                    "thought_text": event.get("thought_text"),
                    "reasoning_text": event.get("reasoning_text"),
                    "evidence_text": event.get("evidence_text"),
                    "flow_priority": event.get("flow_priority"),
                    "decision_signal": event.get("decision_signal"),
                }
                for event in nodes
                if _is_readable_thought_node(event)
            ]

            return CognitiveFlowResponse(
                generated_at=now.isoformat(),
                window_seconds=window_seconds,
                event_count=len(nodes),
                span_count=len({node.get("span_id") for node in nodes if node.get("span_id")} ),
                nodes=nodes,
                edges=edges,
                stats={
                    "available": True,
                    "limit": limit,
                    "session_id": session_id,
                    "window_start": window_start.isoformat(),
                    "span_count": len({node.get("span_id") for node in nodes if node.get("span_id")}),
                    "kind_distribution": dict(kind_distribution),
                    "session_distribution": dict(session_distribution),
                    "span_depth_distribution": dict(span_depth_distribution),
                    "source_type_distribution": dict(event_ids_with_source),
                    "lane_distribution": dict(lane_distribution),
                    "edge_count": len(edges),
                    "noise_event_count": sum(1 for event in nodes if bool(event.get("is_noise"))),
                    "high_signal_event_count": sum(
                        1
                        for event in nodes
                        if not bool(event.get("is_noise")) and float(event.get("flow_priority") or 0.0) >= 0.25
                    ),
                    "readable_thought_count": len(readable_thoughts),
                    "recent_readable_thoughts": _recent_distinct_readable_thoughts(
                        readable_thoughts,
                        limit=12,
                    ),
                    "subsystem_distribution": dict(subsystem_distribution),
                    "pipeline_stage_distribution": dict(stage_distribution),
                    "decision_event_count": decision_count,
                    "temporal_missing_node_count": temporal_missing_node_count,
                    "temporal_missing_edge_count": temporal_missing_edge_count,
                    "candidate_event_count": candidate_event_count,
                    "excluded_noise_event_count": excluded_noise_count,
                    "selected_event_count": len(nodes),
                    "truncated_by_limit": candidate_event_count > len(nodes),
                    "include_noise": include_noise,
                    "show_span_flow": show_span_flow,
                    "show_sequential_links": show_sequential_links,
                    "kind_filter": kind_filter,
                },
            )

        return await cognitive_flow_cache.get(cache_key, _build_response)


    @r.get("/embedding-projection", response_model=ProjectionResponse)
    async def embedding_projection(
        limit: int = 500,
        method: str = "auto",
    ) -> ProjectionResponse:
        mem = runtime.memory
        episodes = await mem.list_episodes(limit=limit)
        points_data = await collect_embedding_points(
            runtime.ctx.embeddings,
            [
                (
                    "episode",
                    {
                        "node_id": f"episode:{ep.episode_id}",
                        "episode_id": str(ep.episode_id),
                        "salience": ep.salience,
                        "kind": ep.kind.value,
                    },
                    ep.embedding_id,
                )
                for ep in episodes
            ],
            requested_method=method,
        )

        points: List[Dict[str, Any]] = []
        for ep in episodes:
            point = points_data["points"].get(f"episode:{ep.episode_id}")
            if point is None:
                continue
            points.append(
                {
                    "episode_id": str(ep.episode_id),
                    "x": point["x"],
                    "y": point["y"],
                    "salience": ep.salience,
                    "kind": ep.kind.value,
                    "projection_group": point["projection_group"],
                    "projection_dimension": point["projection_dimension"],
                    "embedding_model_id": point["embedding_model_id"],
                }
            )

        return ProjectionResponse(
            points=points,
            method=points_data["method"],
            groups=points_data["groups"],
        )

    @r.get("/landscape", response_model=MemoryLandscapeResponse)
    async def memory_landscape(
        limit: int = 140,
        query: Optional[str] = None,
        session_id: Optional[str] = None,
        kind: Optional[str] = None,
        emotion: Optional[str] = None,
        max_age_days: Optional[float] = None,
        min_edge_confidence: float = 0.18,
        edge_kind: Optional[str] = None,
        include_memories: bool = True,
        include_context_proposals: bool = True,
        method: str = "auto",
    ) -> MemoryLandscapeResponse:
        memory = runtime.memory
        now = datetime.now(timezone.utc)
        if query:
            episodes = await memory.search_episodes_by_content(query, limit=limit)
        else:
            episodes = await memory.list_episodes(limit=limit)

        filtered_episodes = []
        for episode in episodes:
            if session_id and episode.session_id != session_id:
                continue
            if kind and episode.kind.value != kind:
                continue
            episode_age_days = max(0.0, (now - episode.created_at).total_seconds() / 86400.0)
            if max_age_days is not None and episode_age_days > max_age_days:
                continue
            primary_emotion = getattr(getattr(episode, "affect", None), "primary_emotion", None)
            if emotion and getattr(primary_emotion, "value", None) != emotion:
                continue
            filtered_episodes.append(episode)

        memory_items = await memory.list_memories(limit=max(25, min(limit, 120))) if include_memories else []
        memories = []
        node_map: Dict[str, Dict[str, Any]] = {}
        episode_lookup: Dict[str, Any] = {}

        for episode in filtered_episodes:
            age_days = max(0.0, (now - episode.created_at).total_seconds() / 86400.0)
            node_id = f"episode:{episode.episode_id}"
            context_meta = episode_context_metadata(episode)
            node_map[node_id] = {
                "node_id": node_id,
                "node_type": "episode",
                "episode_id": str(episode.episode_id),
                "label": truncate_memory_text(episode.content, 72),
                "content": episode.content,
                "created_at": episode.created_at.isoformat(),
                "age_days": round(age_days, 3),
                "kind": episode.kind.value,
                "session_id": episode.session_id,
                "salience": episode.salience,
                "confidence_score": episode.confidence_score,
                "somatic_tag": episode.somatic_tag,
                "compacted": episode.compacted,
                "identity_core": episode.identity_core,
                "used_successfully": episode.used_successfully,
                "used_unsuccessfully": episode.used_unsuccessfully,
                "affect": affect_to_dict(episode.affect),
                "embedding_id": episode.embedding_id,
                "connection_count": 0,
                **context_meta,
                "dual_context": context_meta,
            }
            node_map[node_id] = _with_temporal_metadata(
                node_map[node_id],
                timestamp=episode.created_at,
                source="created_at",
                now=now,
            )
            episode_lookup[str(episode.episode_id)] = episode

        for memory_item in memory_items:
            age_days = max(0.0, (now - memory_item.updated_at).total_seconds() / 86400.0)
            if max_age_days is not None and age_days > max_age_days:
                continue
            memories.append(memory_item)
            node_id = f"memory:{memory_item.memory_id}"
            context_meta = memory_context_metadata(memory_item)
            node_map[node_id] = {
                "node_id": node_id,
                "node_type": "memory",
                "memory_id": str(memory_item.memory_id),
                "label": truncate_memory_text(memory_item.content, 72),
                "content": memory_item.content,
                "created_at": memory_item.created_at.isoformat(),
                "updated_at": memory_item.updated_at.isoformat(),
                "age_days": round(age_days, 3),
                "kind": "memory",
                "session_id": None,
                "salience": memory_item.salience,
                "confidence_score": getattr(memory_item, 'confidence_score', None),
                "somatic_tag": None,
                "compacted": False,
                "identity_core": False,
                "used_successfully": 0,
                "used_unsuccessfully": 0,
                "affect": None,
                "embedding_id": memory_item.embedding_id,
                "tags": list(memory_item.tags),
                "access_count": memory_item.access_count,
                "last_accessed": memory_item.last_accessed.isoformat() if memory_item.last_accessed else None,
                "source_episode_ids": list(memory_item.source_episode_ids),
                "connection_count": 0,
                **context_meta,
                "dual_context": context_meta,
            }
            node_map[node_id] = _with_temporal_metadata(
                node_map[node_id],
                timestamp=memory_item.updated_at,
                source="updated_at",
                now=now,
            )

        context_proposals: List[Any] = []
        context_proposal_limit = min(200, max(8, int(limit)))
        if include_context_proposals:
            context_proposals, context_proposal_limit = await _load_context_proposals(
                runtime,
                query=query,
                limit=limit,
            )
            for proposal in context_proposals:
                node = _proposal_node(proposal, now)
                if max_age_days is not None and float(node["age_days"]) > max_age_days:
                    continue
                if kind and kind not in {node["kind"], node.get("proposal_kind")}:
                    continue
                node_map[node["node_id"]] = node

        projection = await collect_embedding_points(
            runtime.ctx.embeddings,
            [
                (node["node_type"], node, node.get("embedding_id"))
                for node in node_map.values()
            ],
            requested_method=method,
        )
        for node_id, point in projection["points"].items():
            node_map[node_id].update(
                {
                    "x": point["x"],
                    "y": point["y"],
                    "projection_group": point["projection_group"],
                    "projection_dimension": point["projection_dimension"],
                    "projection_method": point["projection_method"],
                    "embedding_model_id": point["embedding_model_id"],
                }
            )

        edges_out: List[Dict[str, Any]] = []
        episode_ids = list(episode_lookup.keys())
        if episode_ids:
            edges = await memory.get_edges_for_batch(
                episode_ids,
                min_confidence=min_edge_confidence,
            )
            for edge in edges:
                source_node = f"episode:{edge.source_id}"
                target_node = f"episode:{edge.target_id}"
                if source_node not in node_map or target_node not in node_map:
                    continue
                if edge_kind and edge.kind.value != edge_kind:
                    continue
                source_episode = episode_lookup.get(edge.source_id)
                target_episode = episode_lookup.get(edge.target_id)
                time_distance_days = None
                if source_episode is not None and target_episode is not None:
                    delta = abs(source_episode.created_at - target_episode.created_at)
                    time_distance_days = round(delta.total_seconds() / 86400.0, 3)
                edges_out.append(
                    {
                        "edge_id": str(edge.edge_id),
                        "source_node_id": source_node,
                        "target_node_id": target_node,
                        "kind": edge.kind.value,
                        "confidence": round(float(edge.confidence), 6),
                        "strength": round(float(compute_edge_strength(edge)), 6),
                        "created_at": edge.created_at.isoformat(),
                        "time_distance_days": time_distance_days,
                        "weights": {
                            "semantic": edge.semantic_weight,
                            "emotional": edge.emotional_weight,
                            "recency": edge.recency_weight,
                            "structural": edge.structural_weight,
                            "salience": edge.salience_weight,
                            "causal": edge.causal_weight,
                            "verification": edge.verification_weight,
                            "actor_affinity": edge.actor_affinity_weight,
                        },
                    }
                )
                node_map[source_node]["connection_count"] += 1
                node_map[target_node]["connection_count"] += 1

        if include_memories:
            for memory_item in memories:
                memory_node_id = f"memory:{memory_item.memory_id}"
                for source_episode_id in memory_item.source_episode_ids:
                    source_node_id = f"episode:{source_episode_id}"
                    if source_node_id not in node_map:
                        continue
                    strength = min(1.0, 0.35 + float(memory_item.salience) / 12.0)
                    if edge_kind and edge_kind != "distilled_from":
                        continue
                    edges_out.append(
                        {
                            "edge_id": f"memory-link:{memory_item.memory_id}:{source_episode_id}",
                            "source_node_id": source_node_id,
                            "target_node_id": memory_node_id,
                            "kind": "distilled_from",
                            "confidence": round(strength, 6),
                            "strength": round(strength, 6),
                            "created_at": memory_item.updated_at.isoformat(),
                            "time_distance_days": None,
                            "weights": {},
                            "synthetic": True,
                        }
                    )
                    node_map[source_node_id]["connection_count"] += 1
                    node_map[memory_node_id]["connection_count"] += 1

        for proposal in context_proposals:
            proposal_node_id = f"context_proposal:{getattr(proposal, 'proposal_id', '')}"
            if proposal_node_id not in node_map:
                continue
            for evidence_node_id in _proposal_evidence_node_ids(proposal, node_map):
                if edge_kind and edge_kind != "proposal_evidence":
                    continue
                edges_out.append(_proposal_edge_payload(proposal, proposal_node_id, evidence_node_id))
                node_map[proposal_node_id]["connection_count"] += 1
                node_map[evidence_node_id]["connection_count"] += 1

        edges_out = [
            _with_temporal_metadata(
                edge,
                timestamp=edge.get("created_at"),
                source="created_at",
                now=now,
            )
            for edge in edges_out
        ]

        emotion_distribution = Counter(
            node.get("affect", {}).get("primary_emotion")
            for node in node_map.values()
            if node.get("affect")
        )
        kind_distribution = Counter(node.get("kind") for node in node_map.values())
        node_type_distribution = Counter(node.get("node_type") for node in node_map.values())
        edge_kind_distribution = Counter(edge.get("kind") for edge in edges_out)
        model_distribution = Counter(
            node.get("embedding_model_id")
            for node in node_map.values()
            if node.get("embedding_model_id")
        )
        lane_distribution = Counter(
            node.get("source_lane") or "unknown"
            for node in node_map.values()
        )
        authority_distribution = Counter(
            node.get("authority") or "unknown"
            for node in node_map.values()
        )
        context_material_distribution = Counter(
            node.get("context_material") or "unknown"
            for node in node_map.values()
        )
        lane_source_distribution = Counter(
            "inferred" if node.get("source_lane_inferred") else "explicit"
            for node in node_map.values()
        )
        proposal_status_distribution = Counter(
            node.get("proposal_status")
            for node in node_map.values()
            if node.get("node_type") == "context_proposal"
        )
        age_values = [node["age_days"] for node in node_map.values()]
        edge_strength_values = [float(edge.get("strength", 0.0)) for edge in edges_out]

        stats = {
            "query": query,
            "session_id": session_id,
            "visible_episode_count": len(filtered_episodes),
            "visible_memory_count": len(memories) if include_memories else 0,
            "visible_context_proposal_count": sum(
                1 for node in node_map.values() if node.get("node_type") == "context_proposal"
            ),
            "visible_edge_count": len(edges_out),
            "projection_method": projection["method"],
            "projection_groups": projection["groups"],
            "embeddingless_node_count": projection["missing_count"],
            "kind_distribution": dict(kind_distribution),
            "node_type_distribution": dict(node_type_distribution),
            "edge_kind_distribution": dict(edge_kind_distribution),
            "emotion_distribution": {k: v for k, v in emotion_distribution.items() if k},
            "embedding_model_distribution": dict(model_distribution),
            "proposal_status_distribution": {k: v for k, v in proposal_status_distribution.items() if k},
            "lane_distribution": dict(lane_distribution),
            "authority_distribution": dict(authority_distribution),
            "context_material_distribution": dict(context_material_distribution),
            "lane_source_distribution": dict(lane_source_distribution),
            "can_authorize_work_count": sum(
                1 for node in node_map.values() if node.get("can_authorize_work")
            ),
            "proposal_kind_distribution": {
                key: value
                for key, value in Counter(
                    node.get("proposal_kind")
                    for node in node_map.values()
                    if node.get("node_type") == "context_proposal"
                ).items()
                if key
            },
            "context_proposal_limit": context_proposal_limit,
            "context_proposal_limit_clamped": include_context_proposals
            and context_proposal_limit < max(8, int(limit)),
            "time_span_days": round(max(age_values) if age_values else 0.0, 3),
            "freshest_visible_age_days": round(min(age_values) if age_values else 0.0, 3),
            "average_edge_strength": round(sum(edge_strength_values) / len(edge_strength_values), 4) if edge_strength_values else 0.0,
            "temporal_missing_node_count": sum(
                1 for node in node_map.values() if not bool(node.get("temporal_has_metadata"))
            ),
            "temporal_missing_edge_count": sum(
                1 for edge in edges_out if not bool(edge.get("temporal_has_metadata"))
            ),
            "max_age_days": max_age_days,
            "min_edge_confidence": min_edge_confidence,
            "edge_kind": edge_kind,
        }

        nodes = sorted(
            node_map.values(),
            key=lambda item: (
                item["node_type"] != "episode",
                item.get("age_days", 0.0),
            ),
        )
        return MemoryLandscapeResponse(
            stats=stats,
            nodes=nodes,
            edges=edges_out,
            projection={
                "method": projection["method"],
                "groups": projection["groups"],
            },
        )

    @r.get("/node-detail", response_model=NodeDetailResponse)
    async def node_detail(
        node_id: str,
        limit: int = 18,
        min_confidence: float = 0.12,
        edge_kind: Optional[str] = None,
    ) -> NodeDetailResponse:
        memory = runtime.memory
        now = datetime.now(timezone.utc)
        kind_filter = None
        synthetic_edge_kinds = {"distilled_from", "proposal_evidence"}
        if edge_kind and edge_kind not in synthetic_edge_kinds:
            try:
                kind_filter = EdgeKind(edge_kind)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Unsupported edge kind: {edge_kind}") from exc

        if ":" not in node_id:
            raise HTTPException(
                status_code=400,
                detail="node_id must be prefixed with episode:, memory:, or context_proposal:",
            )
        node_type, raw_id = node_id.split(":", 1)

        node_payload: Optional[Dict[str, Any]] = None
        neighbors: List[Dict[str, Any]] = []
        edge_payloads: List[Dict[str, Any]] = []

        if node_type == "episode":
            episode = await memory.get_episode(raw_id)
            if episode is None:
                raise HTTPException(status_code=404, detail=f"Unknown episode node: {raw_id}")
            age_days = max(0.0, (now - episode.created_at).total_seconds() / 86400.0)
            node_payload = {
                "node_id": node_id,
                "node_type": "episode",
                **episode_to_dict(episode),
                "label": truncate_memory_text(episode.content, 72),
                "age_days": round(age_days, 3),
            }

            episode_edges = await memory.get_edges_for(
                raw_id,
                min_confidence=min_confidence,
                limit=limit,
                kind=kind_filter,
            )
            neighbor_ids = []
            for edge in episode_edges:
                other_id = edge.target_id if edge.source_id == raw_id else edge.source_id
                neighbor_ids.append(other_id)
            episode_lookup = {
                str(item.episode_id): item
                for item in await memory.get_episodes_by_ids(list(dict.fromkeys(neighbor_ids)))
            }

            for edge in episode_edges:
                other_id = edge.target_id if edge.source_id == raw_id else edge.source_id
                other_episode = episode_lookup.get(other_id)
                if other_episode is None:
                    continue
                other_age_days = max(0.0, (now - other_episode.created_at).total_seconds() / 86400.0)
                neighbor_payload = {
                    "node_id": f"episode:{other_id}",
                    "node_type": "episode",
                    **episode_to_dict(other_episode),
                    "label": truncate_memory_text(other_episode.content, 72),
                    "age_days": round(other_age_days, 3),
                }
                neighbors.append(neighbor_payload)
                edge_dict = edge_to_dict(edge)
                edge_dict.update(
                    {
                        "source_node_id": f"episode:{edge.source_id}",
                        "target_node_id": f"episode:{edge.target_id}",
                        "other_node_id": f"episode:{other_id}",
                        "time_distance_days": round(
                            abs((other_episode.created_at - episode.created_at).total_seconds()) / 86400.0,
                            3,
                        ),
                    }
                )
                edge_dict.update(edge_signal_summary(edge_dict))
                edge_payloads.append(edge_dict)

            if edge_kind in (None, "", "distilled_from"):
                memory_items = await memory.list_memories(limit=max(limit * 4, 80))
                connected_memories = [
                    memory_item
                    for memory_item in memory_items
                    if raw_id in list(memory_item.source_episode_ids)
                ][:limit]
                for memory_item in connected_memories:
                    memory_age_days = max(0.0, (now - memory_item.updated_at).total_seconds() / 86400.0)
                    neighbor_payload = {
                        "node_id": f"memory:{memory_item.memory_id}",
                        "node_type": "memory",
                        **memory_to_dict(memory_item),
                        "label": truncate_memory_text(memory_item.content, 72),
                        "age_days": round(memory_age_days, 3),
                    }
                    neighbors.append(neighbor_payload)
                    strength = min(1.0, 0.35 + float(memory_item.salience) / 12.0)
                    edge_payloads.append(
                        {
                            "edge_id": f"memory-link:{memory_item.memory_id}:{raw_id}",
                            "source_node_id": node_id,
                            "target_node_id": neighbor_payload["node_id"],
                            "other_node_id": neighbor_payload["node_id"],
                            "kind": "distilled_from",
                            "confidence": round(strength, 6),
                            "strength": round(strength, 6),
                            "created_at": memory_item.updated_at.isoformat(),
                            "time_distance_days": None,
                            "strongest_signal": "salience",
                            "strongest_signal_weight": round(float(memory_item.salience), 6),
                            "signal_weights": {"salience": float(memory_item.salience)},
                        }
                    )
        elif node_type == "memory":
            memory_item = await memory.get_memory(raw_id)
            if memory_item is None:
                raise HTTPException(status_code=404, detail=f"Unknown memory node: {raw_id}")
            age_days = max(0.0, (now - memory_item.updated_at).total_seconds() / 86400.0)
            node_payload = {
                "node_id": node_id,
                "node_type": "memory",
                **memory_to_dict(memory_item),
                "label": truncate_memory_text(memory_item.content, 72),
                "age_days": round(age_days, 3),
            }

            source_episodes = await memory.get_episodes_by_ids(list(memory_item.source_episode_ids))
            for episode in source_episodes[:limit]:
                other_age_days = max(0.0, (now - episode.created_at).total_seconds() / 86400.0)
                neighbor_payload = {
                    "node_id": f"episode:{episode.episode_id}",
                    "node_type": "episode",
                    **episode_to_dict(episode),
                    "label": truncate_memory_text(episode.content, 72),
                    "age_days": round(other_age_days, 3),
                }
                neighbors.append(neighbor_payload)
                strength = min(1.0, 0.35 + float(memory_item.salience) / 12.0)
                edge_payloads.append(
                    {
                        "edge_id": f"memory-link:{memory_item.memory_id}:{episode.episode_id}",
                        "source_node_id": neighbor_payload["node_id"],
                        "target_node_id": node_id,
                        "other_node_id": neighbor_payload["node_id"],
                        "kind": "distilled_from",
                        "confidence": round(strength, 6),
                        "strength": round(strength, 6),
                        "created_at": memory_item.updated_at.isoformat(),
                        "time_distance_days": round(
                            abs((memory_item.updated_at - episode.created_at).total_seconds()) / 86400.0,
                            3,
                        ),
                        "strongest_signal": "salience",
                        "strongest_signal_weight": round(float(memory_item.salience), 6),
                        "signal_weights": {"salience": float(memory_item.salience)},
                    }
                )
        elif node_type == "context_proposal":
            store = _context_proposal_store(runtime)
            get_proposal = getattr(store, "get", None)
            proposal = await get_proposal(raw_id) if callable(get_proposal) else None
            if proposal is None:
                raise HTTPException(status_code=404, detail=f"Unknown context proposal node: {raw_id}")
            node_payload = _proposal_node(proposal, now)
            evidence_refs = list(getattr(proposal, "evidence_refs", []) or [])
            episode_ids: List[str] = []
            memory_ids: List[str] = []
            for raw_ref in evidence_refs:
                ref = str(raw_ref or "").strip()
                if not ref:
                    continue
                if ref.startswith("episode:"):
                    episode_ids.append(ref.split(":", 1)[1])
                elif ref.startswith("memory:"):
                    memory_ids.append(ref.split(":", 1)[1])
                else:
                    episode_ids.append(ref)
                    memory_ids.append(ref)
            episode_lookup = {
                str(item.episode_id): item
                for item in await memory.get_episodes_by_ids(list(dict.fromkeys(episode_ids)))
            }
            for episode_id, episode in list(episode_lookup.items())[:limit]:
                other_age_days = max(0.0, (now - episode.created_at).total_seconds() / 86400.0)
                neighbor_payload = {
                    "node_id": f"episode:{episode_id}",
                    "node_type": "episode",
                    **episode_to_dict(episode),
                    "label": truncate_memory_text(episode.content, 72),
                    "age_days": round(other_age_days, 3),
                }
                neighbors.append(neighbor_payload)
                edge = _proposal_edge_payload(proposal, node_payload["node_id"], neighbor_payload["node_id"])
                edge.update(
                    {
                        "other_node_id": neighbor_payload["node_id"],
                        "strongest_signal": "proposal_confidence",
                        "strongest_signal_weight": edge["strength"],
                        "signal_weights": {"proposal_confidence": edge["strength"]},
                    }
                )
                edge_payloads.append(edge)
            get_memory = getattr(memory, "get_memory", None)
            if callable(get_memory):
                for memory_id in list(dict.fromkeys(memory_ids))[:limit]:
                    memory_item = await get_memory(memory_id)
                    if memory_item is None:
                        continue
                    memory_age_days = max(0.0, (now - memory_item.updated_at).total_seconds() / 86400.0)
                    neighbor_payload = {
                        "node_id": f"memory:{memory_item.memory_id}",
                        "node_type": "memory",
                        **memory_to_dict(memory_item),
                        "label": truncate_memory_text(memory_item.content, 72),
                        "age_days": round(memory_age_days, 3),
                    }
                    neighbors.append(neighbor_payload)
                    edge = _proposal_edge_payload(proposal, node_payload["node_id"], neighbor_payload["node_id"])
                    edge.update(
                        {
                            "other_node_id": neighbor_payload["node_id"],
                            "strongest_signal": "proposal_confidence",
                            "strongest_signal_weight": edge["strength"],
                            "signal_weights": {"proposal_confidence": edge["strength"]},
                        }
                    )
                    edge_payloads.append(edge)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported node type: {node_type}")

        unique_neighbors: Dict[str, Dict[str, Any]] = {}
        for neighbor in neighbors:
            unique_neighbors[neighbor["node_id"]] = neighbor
        neighbors = list(unique_neighbors.values())

        await enrich_nodes_with_embeddings(
            runtime.ctx.embeddings,
            [node_payload, *neighbors],
        )

        edge_payloads = [
            _with_temporal_metadata(
                edge,
                timestamp=edge.get("created_at"),
                source="created_at",
                now=now,
            )
            for edge in edge_payloads
        ]
        edge_kind_distribution = Counter(edge.get("kind") for edge in edge_payloads)
        neighbor_type_distribution = Counter(node.get("node_type") for node in neighbors)
        return NodeDetailResponse(
            node=node_payload,
            neighbors=sorted(
                neighbors,
                key=lambda item: (
                    item.get("node_type") != "episode",
                    item.get("age_days", 0.0),
                ),
            ),
            edges=sorted(edge_payloads, key=lambda item: float(item.get("strength", 0.0)), reverse=True),
            stats={
                "neighbor_count": len(neighbors),
                "edge_count": len(edge_payloads),
                "edge_kind_distribution": dict(edge_kind_distribution),
                "neighbor_type_distribution": dict(neighbor_type_distribution),
            },
        )

    @r.get("/retrieval-inspect", response_model=RetrievalInspectResponse)
    async def retrieval_inspect(
        query: str,
        limit: int = 12,
        session_id: Optional[str] = None,
        min_confidence: float = 0.15,
        lambda_param: float = 0.5,
        expand_graph: bool = True,
        semantic_weight: float = MemoryRetriever.DEFAULT_FUSION_WEIGHTS["semantic_score"],
        keyword_weight: float = MemoryRetriever.DEFAULT_FUSION_WEIGHTS["keyword_score"],
        recency_weight: float = MemoryRetriever.DEFAULT_FUSION_WEIGHTS["recency_score"],
        salience_weight: float = MemoryRetriever.DEFAULT_FUSION_WEIGHTS["salience_score"],
        graph_weight: float = MemoryRetriever.DEFAULT_FUSION_WEIGHTS["graph_score"],
        emotional_weight: float = MemoryRetriever.DEFAULT_FUSION_WEIGHTS["emotional_resonance"],
        temporal_weight: float = MemoryRetriever.DEFAULT_FUSION_WEIGHTS["temporal_echo"],
        reliability_weight: float = MemoryRetriever.DEFAULT_FUSION_WEIGHTS["reliability"],
    ) -> RetrievalInspectResponse:
        retriever = build_memory_retriever(runtime)
        inspection = await retriever.inspect(
            query=query,
            session_id=session_id,
            limit=limit,
            expand_graph=expand_graph,
            min_confidence=min_confidence,
            lambda_param=lambda_param,
            weights={
                "semantic_score": semantic_weight,
                "keyword_score": keyword_weight,
                "recency_score": recency_weight,
                "salience_score": salience_weight,
                "graph_score": graph_weight,
                "emotional_resonance": emotional_weight,
                "temporal_echo": temporal_weight,
                "reliability": reliability_weight,
            },
        )

        results = []
        for item in inspection["results"]:
            payload = {
                "source_type": item.source_type,
                "source_id": item.source_id,
                "content": item.content,
                "content_preview": truncate_memory_text(item.content, 200),
                "score": round(float(item.score), 6),
            }
            if item.episode is not None:
                payload["episode"] = episode_to_dict(item.episode)
                payload["node_id"] = f"episode:{item.source_id}"
            if item.memory is not None:
                payload["memory"] = memory_to_dict(item.memory)
                payload["node_id"] = f"memory:{item.source_id}"
            results.append(payload)

        return RetrievalInspectResponse(
            query=query,
            weights=inspection["weights"],
            candidates=inspection["candidates"],
            results=results,
            meta=inspection["meta"],
        )

    return r
