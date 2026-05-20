"""Grounded working-set collection for reflective/daydream cognition.

The daydream lane should not invent what the agent is "thinking about".  This
module builds a bounded, evidence-backed set of currently active material from
the runtime surfaces that already exist: current turn state, conversation
history, cognitive state, commitments, plans, tasks, telemetry, and recent
memory observations.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from opencas.identity.text_hygiene import sanitize_identity_text


@dataclass
class ReflectiveWorkingSetItem:
    """One grounded item available to reflective cognition."""

    kind: str
    source: str
    label: str
    text: str
    created_at: str | None = None
    observed_at: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    weight: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        temporal = {
            "created_at": self.created_at,
            "observed_at": self.observed_at,
            "time_basis": "source_created_at" if self.created_at else "runtime_observed_at",
        }
        return {
            "kind": self.kind,
            "source": self.source,
            "label": self.label,
            "text": self.text,
            "created_at": self.created_at,
            "observed_at": self.observed_at,
            "temporal": temporal,
            "evidence_ids": list(self.evidence_ids),
            "weight": round(float(self.weight), 3),
            "meta": dict(self.meta),
        }


async def collect_reflective_working_set(
    runtime: Any,
    *,
    limit: int = 18,
) -> list[dict[str, Any]]:
    """Collect current grounded material for daydream/reflection generation."""

    items: list[ReflectiveWorkingSetItem] = []
    seen: set[str] = set()
    session_id = _current_session_id(runtime)
    observed_at = _iso(datetime.now(timezone.utc))

    def add(
        *,
        kind: str,
        source: str,
        label: str,
        text: Any,
        created_at: Any = None,
        evidence_ids: list[str] | None = None,
        weight: float = 0.5,
        meta: dict[str, Any] | None = None,
    ) -> None:
        clean_text = _compact(text, 360)
        clean_label = _compact(label, 80)
        if not clean_text or not clean_label:
            return
        clean_kind = _compact(kind, 64) or "observation"
        clean_source = _compact(source, 96) or "runtime"
        clean_created_at = _iso(created_at)
        item_meta = dict(meta or {})
        item_meta.setdefault("authority", _infer_authority(clean_kind, clean_source, item_meta))
        item_meta.setdefault("source_family", _source_family(clean_kind, clean_source, item_meta))
        item_meta.setdefault(
            "novelty_pressure",
            _item_novelty_pressure(
                kind=clean_kind,
                source=clean_source,
                text=clean_text,
                weight=weight,
                created_at=clean_created_at,
            ),
        )
        item_meta.setdefault("temporal_basis", "source_created_at" if clean_created_at else "runtime_observed_at")
        key = f"{kind}|{source}|{clean_label}|{clean_text}".lower()
        if key in seen:
            return
        seen.add(key)
        items.append(
            ReflectiveWorkingSetItem(
                kind=clean_kind,
                source=clean_source,
                label=clean_label,
                text=clean_text,
                created_at=clean_created_at,
                observed_at=observed_at,
                evidence_ids=[_compact(eid, 120) for eid in list(evidence_ids or []) if _compact(eid, 120)],
                weight=max(0.0, min(1.0, float(weight))),
                meta=item_meta,
            )
        )

    _collect_runtime_focus(runtime, add)
    await _collect_recent_dialogue(runtime, session_id, add)
    await _collect_cognitive_state(runtime, session_id, add)
    _collect_identity_interest_state(runtime, add)
    await _collect_tom_user_model(runtime, add)
    await _collect_active_work(runtime, add)
    await _collect_recent_self_work(runtime, add)
    _collect_recent_telemetry(runtime, session_id, add)
    await _collect_recent_memory(runtime, session_id, add)

    items.sort(key=lambda item: (item.weight, item.created_at or ""), reverse=True)
    selected = _diversify_items(items, max(1, int(limit)))
    return [item.to_dict() for item in selected]


def working_set_source_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize working-set source coverage for traces and reflection context."""

    counts: dict[str, int] = {}
    for item in items:
        source = str(item.get("source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def working_set_source_family_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize higher-level source-family coverage for daydream intake."""

    counts: dict[str, int] = {}
    for item in items:
        meta = item.get("meta") if isinstance(item, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        family = _compact(meta.get("source_family"), 80) or _source_family(
            str(item.get("kind") or ""),
            str(item.get("source") or ""),
            meta,
        )
        counts[family] = counts.get(family, 0) + 1
    return counts


def working_set_novelty_pressure(items: list[dict[str, Any]]) -> float:
    """Estimate how much current evidence should pull reflection forward.

    This is not execution authority. It only says the reflective lane has fresh,
    diverse, grounded material worth thinking about even if boredom is low.
    """

    if not items:
        return 0.0
    pressures: list[float] = []
    sources: set[str] = set()
    operator_items = 0
    for item in items:
        meta = item.get("meta") if isinstance(item, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        try:
            pressure = float(meta.get("novelty_pressure", 0.0) or 0.0)
        except (TypeError, ValueError):
            pressure = 0.0
        pressures.append(max(0.0, min(1.0, pressure)))
        source = str(item.get("source") or "")
        if source:
            sources.add(source)
        if meta.get("authority") == "operator":
            operator_items += 1
    max_pressure = max(pressures) if pressures else 0.0
    avg_pressure = sum(pressures) / max(1, len(pressures))
    diversity_bonus = min(0.10, max(0, len(sources) - 1) * 0.035)
    operator_bonus = min(0.06, operator_items * 0.02)
    return round(min(0.42, (max_pressure * 0.22) + (avg_pressure * 0.14) + diversity_bonus + operator_bonus), 3)


def _collect_runtime_focus(runtime: Any, add: Any) -> None:
    activity = getattr(runtime, "_activity", None)
    if activity:
        add(
            kind="runtime_activity",
            source="runtime.activity",
            label="Current runtime activity",
            text=str(activity),
            created_at=getattr(runtime, "_activity_since", None),
            weight=0.75,
        )

    cognition = getattr(runtime, "current_cognition", None)
    payload = _model_payload(cognition)
    if not payload:
        return
    actor = _actor_label(payload)
    session_id = payload.get("session_id")
    for key in ("user_input", "user_text", "input", "content", "message"):
        if payload.get(key):
            add(
                kind="active_turn",
                source="runtime.current_cognition",
                label=f"Current turn from {actor}",
                text=payload.get(key),
                created_at=payload.get("created_at") or payload.get("timestamp"),
                evidence_ids=[str(v) for v in (payload.get("evidence_ids") or [])],
                weight=0.98,
                meta={"session_id": session_id, "actor": actor},
            )
            break


async def _collect_recent_dialogue(runtime: Any, session_id: str | None, add: Any) -> None:
    store = getattr(getattr(runtime, "ctx", None), "context_store", None)
    if store is None or not session_id:
        return
    try:
        entries = await _maybe_await(store.list_recent(session_id, limit=6, include_hidden=True))
    except TypeError:
        entries = await _maybe_await(store.list_recent(session_id, limit=6))
    except Exception:
        return
    for entry in entries[-6:]:
        meta = getattr(entry, "meta", {}) or {}
        actor = _actor_label(meta)
        role = getattr(getattr(entry, "role", None), "value", getattr(entry, "role", "message"))
        label = f"Recent {role}"
        if actor and actor != "operator":
            label = f"{label} ({actor})"
        add(
            kind="recent_dialogue",
            source="context_store",
            label=label,
            text=getattr(entry, "content", ""),
            created_at=getattr(entry, "created_at", None),
            evidence_ids=[str(getattr(entry, "message_id", ""))],
            weight=0.82 if role == "user" else 0.68,
            meta={"session_id": session_id, "actor": actor},
        )


async def _collect_cognitive_state(runtime: Any, session_id: str | None, add: Any) -> None:
    store = (
        getattr(getattr(runtime, "ctx", None), "cognitive_state_store", None)
        or getattr(runtime, "cognitive_state_store", None)
    )
    if store is None:
        return
    try:
        for target in await _maybe_await(store.list_attention(limit=6)):
            add(
                kind="attention",
                source="cognitive_state.attention",
                label=getattr(target, "label", "attention"),
                text=getattr(target, "label", ""),
                created_at=getattr(target, "updated_at", None),
                evidence_ids=list(getattr(target, "evidence_refs", []) or []),
                weight=float(getattr(target, "strength", 0.6) or 0.6),
            )
    except Exception:
        pass
    try:
        for item in await _maybe_await(store.list_working_memory(limit=6)):
            add(
                kind="working_memory",
                source="cognitive_state.working_memory",
                label=getattr(item, "slot", "working memory"),
                text=getattr(item, "content", ""),
                created_at=getattr(item, "updated_at", None),
                evidence_ids=list(getattr(item, "evidence_refs", []) or []),
                weight=float(getattr(item, "priority", 0.6) or 0.6),
            )
    except Exception:
        pass
    try:
        for item in await _maybe_await(store.list_prospective_memories(limit=6)):
            add(
                kind="prospective_memory",
                source="cognitive_state.prospective_memory",
                label="Prospective memory",
                text=" ".join(
                    part
                    for part in (
                        getattr(item, "action", ""),
                        getattr(item, "condition", ""),
                    )
                    if part
                ),
                created_at=getattr(item, "updated_at", None),
                evidence_ids=list(getattr(item, "evidence_refs", []) or []),
                weight=float(getattr(item, "confidence", 0.6) or 0.6),
                meta={"trigger_at": getattr(item, "trigger_at", None), "proof_ref": getattr(item, "proof_ref", "")},
            )
    except Exception:
        pass
    try:
        events = await _maybe_await(store.list_recent_events(session_id=session_id, status="active", limit=8))
        for event in events[-8:]:
            add(
                kind=f"cognitive_event:{getattr(getattr(event, 'kind', None), 'value', getattr(event, 'kind', 'event'))}",
                source="cognitive_state.events",
                label=getattr(event, "summary", "cognitive event"),
                text=getattr(event, "content", "") or getattr(event, "summary", ""),
                created_at=getattr(event, "created_at", None),
                evidence_ids=[str(getattr(event, "event_id", "")), *list(getattr(event, "evidence_refs", []) or [])],
                weight=min(1.0, float(getattr(event, "salience", 1.0) or 1.0) / 3.0),
            )
    except Exception:
        pass


def _collect_identity_interest_state(runtime: Any, add: Any) -> None:
    identity = getattr(getattr(runtime, "ctx", None), "identity", None) or getattr(runtime, "identity", None)
    if identity is None:
        return

    self_model = getattr(identity, "self_model", None)
    beliefs = getattr(self_model, "self_beliefs", {}) or {}
    daydream = beliefs.get("daydream") if isinstance(beliefs, dict) else None
    if isinstance(daydream, dict):
        config = daydream.get("bulma_config") if isinstance(daydream.get("bulma_config"), dict) else {}
        status = daydream.get("bulma_status") if isinstance(daydream.get("bulma_status"), dict) else {}
        current_interest = status.get("currentInterest") or status.get("current_interest")
        if current_interest:
            add(
                kind="personal_curiosity",
                source="identity.self_beliefs.daydream",
                label="Current self-directed interest",
                text=current_interest,
                created_at=status.get("updated_at") or daydream.get("updated_at"),
                weight=0.7,
                meta={"authority": "runtime", "self_directed": True},
            )
        raw_seeds = config.get("hobbySeeds") or config.get("hobby_seeds") or []
        if isinstance(raw_seeds, list):
            for seed in raw_seeds[:6]:
                add(
                    kind="personal_curiosity",
                    source="identity.self_beliefs.daydream",
                    label="Hobby or curiosity seed",
                    text=seed,
                    created_at=config.get("updated_at") or daydream.get("updated_at"),
                    weight=0.62,
                    meta={"authority": "runtime", "self_directed": True},
                )

    user_model = getattr(identity, "user_model", None)
    preferences = getattr(user_model, "explicit_preferences", {}) or {}
    if isinstance(preferences, dict):
        for key in (
            "interests",
            "interest",
            "hobbies",
            "hobby",
            "topics",
            "news_topics",
            "research_interests",
            "communication_style",
        ):
            value = preferences.get(key)
            if value:
                add(
                    kind="user_interest_or_preference",
                    source="identity.user_model",
                    label=f"User model {key}",
                    text=value,
                    created_at=getattr(user_model, "updated_at", None),
                    weight=0.66,
                    meta={"authority": "operator", "preference_key": key},
                )
    inferred_goals = getattr(user_model, "inferred_goals", []) or []
    for goal in list(inferred_goals)[:5]:
        payload = _model_payload(goal)
        text = _first_payload_text(payload, ("text", "goal", "summary", "description"))
        if text:
            add(
                kind="inferred_user_goal",
                source="identity.user_model",
                label="Inferred user goal",
                text=text,
                created_at=payload.get("updated_at") or getattr(user_model, "updated_at", None),
                weight=0.58,
                meta={"authority": "runtime", "provenance": payload.get("provenance")},
            )


async def _collect_tom_user_model(runtime: Any, add: Any) -> None:
    tom = getattr(getattr(runtime, "ctx", None), "tom", None) or getattr(runtime, "tom", None)
    method = getattr(tom, "list_beliefs", None)
    if not callable(method):
        return
    try:
        records = await _maybe_await(method(subject="user", limit=12))
    except TypeError:
        try:
            records = await _maybe_await(method(limit=12))
        except TypeError:
            try:
                records = await _maybe_await(method())
            except Exception:
                return
        except Exception:
            return
    except Exception:
        return
    for record in list(records or [])[:12]:
        payload = _model_payload(record)
        subject = _enum_text(payload.get("subject") or getattr(record, "subject", None)).lower()
        if subject and subject not in {"user", "operator", "owner", "human"}:
            continue
        relation = _compact(
            payload.get("relation") or payload.get("kind") or payload.get("type"),
            80,
        )
        predicate = _compact(payload.get("predicate") or payload.get("claim") or payload.get("summary"), 180)
        obj = _compact(payload.get("object") or payload.get("value") or payload.get("content"), 240)
        confidence = _safe_float(payload.get("confidence"), default=0.5)
        text = _tom_belief_text(relation=relation, predicate=predicate, obj=obj)
        if not text:
            continue
        relation_l = relation.lower()
        is_preference = any(
            token in f"{relation_l} {predicate.lower()} {obj.lower()}"
            for token in (
                "prefer",
                "interest",
                "hobby",
                "like",
                "want",
                "need",
                "goal",
                "asked",
                "communication",
                "autonomy",
                "assistant",
            )
        )
        add(
            kind="tom_user_belief",
            source="tom.user_beliefs",
            label=relation or "User belief",
            text=text,
            created_at=payload.get("updated_at") or payload.get("created_at") or payload.get("timestamp"),
            evidence_ids=_tom_evidence_ids(payload),
            weight=0.74 if is_preference else 0.58,
            meta={
                "authority": "runtime",
                "relation": relation,
                "predicate": predicate,
                "object": obj,
                "confidence": round(confidence, 3),
                "source_kind": _compact(payload.get("source_kind") or payload.get("provenance"), 80),
            },
        )


async def _collect_active_work(runtime: Any, add: Any) -> None:
    executive = getattr(runtime, "executive", None)
    for goal in list(getattr(executive, "active_goals", []) or [])[:6]:
        add(
            kind="active_goal",
            source="executive.active_goals",
            label="Active goal",
            text=goal,
            weight=0.9,
        )

    for store_attr, method_name, kind, source, text_keys in (
        ("commitment_store", "list_active", "active_commitment", "commitments", ("text", "summary", "description", "title")),
        ("plan_store", "list_active", "active_plan", "plans", ("summary", "title", "description")),
        ("tasks", "list_pending", "pending_task", "tasks", ("description", "summary", "title")),
    ):
        store = getattr(runtime, store_attr, None) or getattr(getattr(runtime, "ctx", None), store_attr, None)
        method = getattr(store, method_name, None)
        if not callable(method):
            continue
        try:
            records = await _maybe_await(method(limit=6))
        except TypeError:
            try:
                records = await _maybe_await(method())
            except Exception:
                continue
        except Exception:
            continue
        for record in list(records or [])[:6]:
            payload = _model_payload(record)
            text = _first_payload_text(payload, text_keys)
            add(
                kind=kind,
                source=source,
                label=_first_payload_text(payload, ("title", "summary", "description", "text")) or kind,
                text=text,
                created_at=payload.get("updated_at") or payload.get("created_at"),
                evidence_ids=[str(payload.get("commitment_id") or payload.get("plan_id") or payload.get("task_id") or "")],
                weight=0.82,
            )


async def _collect_recent_self_work(runtime: Any, add: Any) -> None:
    store = (
        getattr(runtime, "daydream_signal_store", None)
        or getattr(getattr(runtime, "ctx", None), "daydream_signal_store", None)
    )
    method = getattr(store, "list_receipts", None)
    if not callable(method):
        return
    try:
        receipts = await _maybe_await(method(limit=6))
    except TypeError:
        try:
            receipts = await _maybe_await(method())
        except Exception:
            return
    except Exception:
        return
    for receipt in list(receipts or [])[:6]:
        payload = _model_payload(receipt)
        raw = payload.get("raw") if isinstance(payload.get("raw"), dict) else {}
        artifact_paths = payload.get("artifact_paths") if isinstance(payload.get("artifact_paths"), list) else []
        validation_status = _compact(raw.get("validation_status"), 80)
        outcome = _compact(payload.get("outcome"), 80)
        route = _compact(payload.get("route"), 80)
        kind_value = _compact(payload.get("kind"), 80)
        summary = _first_payload_text(payload, ("summary", "outcome", "route")) or "self work receipt"
        artifact_summary = ", ".join(_compact(path, 120) for path in artifact_paths[:3] if _compact(path, 120))
        status_bits = [
            f"route={route}" if route else "",
            f"kind={kind_value}" if kind_value else "",
            f"outcome={outcome}" if outcome else "",
            f"validation={validation_status}" if validation_status else "",
            f"artifacts={artifact_summary}" if artifact_summary else "",
        ]
        text = " | ".join(bit for bit in [summary, *status_bits] if bit)
        receipt_id = _compact(payload.get("receipt_id"), 120)
        signal_id = _compact(payload.get("signal_id"), 120)
        evidence_ids = [item for item in [receipt_id, signal_id, *artifact_paths[:3]] if item]
        add(
            kind="self_work_scaffold" if validation_status == "scaffold_only" else "self_work_receipt",
            source="daydream.self_work",
            label=summary,
            text=text,
            created_at=payload.get("created_at"),
            evidence_ids=evidence_ids,
            weight=0.78 if validation_status == "scaffold_only" else 0.64,
            meta={
                "route": route,
                "kind": kind_value,
                "outcome": outcome,
                "validation_status": validation_status,
                "artifact_paths": artifact_paths,
                "self_directed": True,
            },
        )


def _collect_recent_telemetry(runtime: Any, session_id: str | None, add: Any) -> None:
    tracer = getattr(runtime, "tracer", None)
    store = getattr(tracer, "store", None)
    query = getattr(store, "query", None)
    if not callable(query):
        return
    try:
        events = query(session_id=session_id, limit=16) if session_id else query(limit=16)
    except Exception:
        return
    for event in list(events or [])[-10:]:
        payload = getattr(event, "payload", {}) or {}
        kind_value = _event_kind_value(event)
        summary = _first_payload_text(
            payload,
            (
                "summary",
                "reason",
                "route",
                "stage",
                "status",
                "action",
                "tool",
                "tool_name",
                "query",
                "title",
                "subject",
                "headline",
                "url",
            ),
        )
        text = summary or getattr(event, "message", "")
        blob = _payload_text_blob(payload, getattr(event, "message", ""))
        is_tool = kind_value == "tool_call" or any(key in payload for key in ("tool", "tool_name", "tool_call_id"))
        is_external = _looks_external(blob)
        tool_name = _compact(payload.get("tool") or payload.get("tool_name") or payload.get("name"), 80)
        if is_tool and tool_name:
            tool_subject = _first_payload_text(
                payload,
                ("query", "headline", "subject", "title", "url", "summary", "reason", "action", "status"),
            )
            text = " | ".join(bit for bit in (tool_name, tool_subject or summary or getattr(event, "message", "")) if bit)
        source = "telemetry.tool_call" if is_tool else "telemetry.external" if is_external else "telemetry"
        item_kind = "tool_observation" if is_tool else "external_observation" if is_external else f"telemetry:{kind_value}"
        add(
            kind=item_kind,
            source=source,
            label=getattr(event, "message", "") or "telemetry event",
            text=text,
            created_at=getattr(event, "timestamp", None),
            evidence_ids=[str(getattr(event, "event_id", ""))],
            weight=0.7 if is_tool or is_external else 0.58,
            meta={
                "span_id": getattr(event, "span_id", None),
                "parent_span_id": getattr(event, "parent_span_id", None),
                "event_kind": kind_value,
                "tool_name": tool_name,
            },
        )


async def _collect_recent_memory(runtime: Any, session_id: str | None, add: Any) -> None:
    memory = getattr(runtime, "memory", None)
    if memory is None:
        return
    loader = getattr(memory, "list_recent_episodes", None) or getattr(memory, "list_episodes", None)
    if not callable(loader):
        return
    try:
        episodes = await _maybe_await(loader(session_id=session_id, limit=12))
    except TypeError:
        try:
            episodes = await _maybe_await(loader(limit=12))
        except Exception:
            return
    except Exception:
        return
    for episode in list(episodes or [])[-8:]:
        payload = _model_payload(episode)
        content = payload.get("content", "")
        meta_text = " ".join(str(v) for v in (payload.get("meta") or {}).values()) if isinstance(payload.get("meta"), dict) else ""
        is_external = _looks_external(f"{content} {meta_text}")
        add(
            kind="external_observation" if is_external else "recent_memory",
            source="memory",
            label="Recent external observation" if is_external else "Recent memory",
            text=content,
            created_at=payload.get("updated_at") or payload.get("created_at") or payload.get("timestamp"),
            evidence_ids=[str(payload.get("episode_id") or payload.get("memory_id") or "")],
            weight=0.72 if is_external else 0.56,
            meta={"session_id": payload.get("session_id")},
        )


def _current_session_id(runtime: Any) -> str | None:
    cognition_payload = _model_payload(getattr(runtime, "current_cognition", None))
    for value in (
        cognition_payload.get("session_id"),
        getattr(getattr(getattr(runtime, "ctx", None), "config", None), "session_id", None),
        "default",
    ):
        text = _compact(value, 120)
        if text:
            return text
    return None


def _actor_label(payload: dict[str, Any]) -> str:
    actor = payload.get("conversation_actor")
    if isinstance(actor, dict):
        return _compact(actor.get("label") or actor.get("type") or "operator", 80) or "operator"
    user_meta = payload.get("user_meta")
    if isinstance(user_meta, dict):
        return _actor_label(user_meta)
    return "operator"


def _infer_authority(kind: str, source: str, meta: dict[str, Any]) -> str:
    actor = str(meta.get("actor") or "").strip().lower()
    source_l = source.lower()
    kind_l = kind.lower()
    if "collaborator" in actor or actor in {"codex", "claude", "gemini", "kimi"}:
        return "collaborator_agent"
    if actor and actor != "operator":
        return "collaborator_agent"
    if source_l in {"context_store", "runtime.current_cognition"} or kind_l in {"active_turn", "recent_dialogue"}:
        return "operator"
    return "runtime"


def _item_novelty_pressure(
    *,
    kind: str,
    source: str,
    text: str,
    weight: float,
    created_at: str | None,
) -> float:
    kind_l = kind.lower()
    source_l = source.lower()
    text_l = text.lower()
    base = 0.18
    if kind_l == "active_turn":
        base = 0.95
    elif kind_l == "recent_dialogue":
        base = 0.78
    elif kind_l == "external_observation":
        base = 0.84
    elif kind_l == "tool_observation":
        base = 0.80
    elif kind_l == "tom_user_belief":
        base = 0.68
    elif kind_l in {"active_goal", "active_commitment", "active_plan", "pending_task"}:
        base = 0.62
    elif kind_l in {"self_work_scaffold", "self_work_receipt"} or source_l == "daydream.self_work":
        base = 0.72 if kind_l == "self_work_scaffold" else 0.58
    elif kind_l.startswith("telemetry:"):
        base = 0.46
    elif source_l.startswith("cognitive_state"):
        base = 0.54
    elif source_l == "memory":
        base = 0.34
    if any(token in text_l for token in ("email", "gmail", "headline", "news", "browser", "web", "rss", "research", "fetched", "url")):
        base = max(base, 0.76)
    if not created_at:
        base *= 0.9
    try:
        weight_f = max(0.0, min(1.0, float(weight)))
    except (TypeError, ValueError):
        weight_f = 0.5
    return round(max(0.0, min(1.0, (base * 0.7) + (weight_f * 0.3))), 3)


def _model_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return dict(model_dump(mode="json"))
        except TypeError:
            try:
                return dict(model_dump())
            except Exception:
                return {}
        except Exception:
            return {}
    if hasattr(value, "__dict__"):
        return dict(getattr(value, "__dict__", {}) or {})
    return {}


def _first_payload_text(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if value:
            text = _compact(value, 360)
            if text:
                return text
    return ""


def _source_family(kind: str, source: str, meta: dict[str, Any]) -> str:
    explicit = _compact(meta.get("source_family"), 80) if isinstance(meta, dict) else ""
    if explicit:
        return explicit
    kind_l = kind.lower()
    source_l = source.lower()
    if kind_l == "active_turn" or source_l == "runtime.current_cognition":
        return "active_turn"
    if kind_l == "recent_dialogue" or source_l == "context_store":
        return "conversation"
    if kind_l in {"active_goal", "active_commitment", "active_plan", "pending_task"}:
        return "active_work"
    if kind_l == "tom_user_belief" or source_l.startswith("tom."):
        return "tom_user_model"
    if kind_l in {"personal_curiosity", "user_interest_or_preference", "inferred_user_goal"}:
        return "identity_interest"
    if kind_l == "tool_observation" or source_l == "telemetry.tool_call":
        return "tool_observation"
    if kind_l == "external_observation" or source_l == "telemetry.external":
        return "external_observation"
    if kind_l in {"self_work_scaffold", "self_work_receipt"} or source_l == "daydream.self_work":
        return "self_work"
    if source_l.startswith("cognitive_state"):
        return "cognitive_state"
    if source_l == "memory":
        return "memory"
    if source_l.startswith("runtime"):
        return "runtime"
    if source_l.startswith("telemetry"):
        return "telemetry"
    return "other"


def _diversify_items(items: list[ReflectiveWorkingSetItem], limit: int) -> list[ReflectiveWorkingSetItem]:
    """Keep high-salience items while preventing one subsystem from crowding out the rest."""

    if len(items) <= limit:
        return list(items)
    priority = [
        "active_turn",
        "conversation",
        "active_work",
        "tom_user_model",
        "identity_interest",
        "external_observation",
        "tool_observation",
        "self_work",
        "cognitive_state",
        "memory",
        "telemetry",
        "runtime",
    ]
    selected: list[ReflectiveWorkingSetItem] = []
    selected_ids: set[int] = set()

    def take(item: ReflectiveWorkingSetItem) -> bool:
        ident = id(item)
        if ident in selected_ids or len(selected) >= limit:
            return False
        selected.append(item)
        selected_ids.add(ident)
        return True

    for family in priority:
        for item in items:
            if item.meta.get("source_family") == family:
                take(item)
                break
        if len(selected) >= limit:
            return selected

    for item in items:
        take(item)
        if len(selected) >= limit:
            break
    return selected


def _event_kind_value(event: Any) -> str:
    kind = getattr(event, "kind", None)
    return _compact(getattr(kind, "value", kind), 80) or "event"


def _payload_text_blob(payload: dict[str, Any], message: Any = "") -> str:
    values = [str(message or "")]
    for value in payload.values():
        if isinstance(value, (str, int, float, bool)):
            values.append(str(value))
        elif isinstance(value, dict):
            values.extend(str(v) for v in value.values() if isinstance(v, (str, int, float, bool)))
        elif isinstance(value, list):
            values.extend(str(v) for v in value[:8] if isinstance(v, (str, int, float, bool)))
    return " ".join(values)


def _looks_external(text: Any) -> bool:
    text_l = str(text or "").lower()
    return any(
        token in text_l
        for token in (
            "email",
            "gmail",
            "headline",
            "news",
            "browser",
            "web",
            "rss",
            "http://",
            "https://",
            "url",
            "fetched",
            "search",
        )
    )


def _enum_text(value: Any) -> str:
    return _compact(getattr(value, "value", value), 120)


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _tom_belief_text(*, relation: str, predicate: str, obj: str) -> str:
    if relation and obj:
        text = f"{relation}: {obj}"
        if predicate and predicate.lower() not in text.lower():
            text = f"{text} ({predicate})"
        return text
    return predicate or obj or relation


def _tom_evidence_ids(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("evidence_ids") or payload.get("evidence_refs") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raw = []
    belief_id = payload.get("belief_id") or payload.get("id")
    return [
        _compact(value, 120)
        for value in [belief_id, *raw]
        if _compact(value, 120)
    ]


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    text = _compact(value, 80)
    return text or None


def _compact(value: Any, limit: int) -> str:
    text = sanitize_identity_text(str(value or ""))
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "..."
    return text
