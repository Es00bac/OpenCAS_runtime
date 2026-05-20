"""Runtime cognitive feedback-loop maintenance."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from opencas.cognition import CognitiveEventKind, recommended_counterfactual
from opencas.telemetry.affect_analyzer import compute_trajectory_from_snapshots
from opencas.telemetry.affect_graph import AffectQualityGraph
from opencas.telemetry.affect_interventions import AnomalyDetector, InterventionEngine
from opencas.telemetry.affect_models import AffectDimension, AffectSnapshot, MoodSignalSource, QualitySignal
from opencas.telemetry.affect_store import AffectStore
from opencas.thread_registry import BeadSourceKind
from opencas.runtime.commitment_followthrough import seed_active_support_commitments
from opencas.runtime.context_proposal_intake import intake_context_proposals
from opencas.runtime.cognition_bus import NarratorRedundancyObserved
from opencas.runtime.priority_patterns import record_priority_patterns


async def run_runtime_cognitive_maintenance(runtime: Any) -> dict[str, Any]:
    """Close cognitive feedback loops and refresh prompt-facing state.

    This pass is deliberately evidence-gated and cheap: it mines existing
    runtime records into the cognitive spine so the next prompt/tool loop can
    use them. It does not hot-patch code, config, or prompts.
    """

    store = _cognitive_store(runtime)
    if store is None:
        return {"available": False, "reason": "cognitive_state_store_unavailable"}

    result: dict[str, Any] = {
        "available": True,
        "attention_updates": 0,
        "working_memory_updates": 0,
        "events_recorded": 0,
        "skills": {},
        "promotions": {},
        "priority_patterns": {},
        "curiosity_promotions": 0,
        "commitment_followthrough_seeds": 0,
        "context_proposal_intake": {},
        "telemetry_affect": {},
    }

    result["commitment_followthrough_seeds"] += await seed_active_support_commitments(runtime)
    result["attention_updates"] += await _refresh_attention(runtime, store)
    result["working_memory_updates"] += await _refresh_working_memory(runtime, store)
    result["events_recorded"] += await _record_affective_registry_trend(runtime, store)
    result["events_recorded"] += await _record_self_inspection_feedback(runtime, store)
    result["events_recorded"] += await _record_recall_failure_memories(runtime, store)
    result["events_recorded"] += await _record_curiosity_state(runtime, store)
    result["curiosity_promotions"] += await _promote_curiosity_to_creative(runtime, store)
    result["events_recorded"] += await _record_habit_candidates(runtime, store)
    result["events_recorded"] += await _record_counterfactual_need(runtime, store)
    result["events_recorded"] += await _record_somatic_strategy_pressure(runtime, store)
    result["events_recorded"] += await _record_uncertainty_pressure(runtime, store)
    result["events_recorded"] += await _record_social_agent_models(runtime, store)
    result["events_recorded"] += await _record_narrative_summary(runtime, store)
    result["context_proposal_intake"] = await intake_context_proposals(runtime, store)
    result["priority_patterns"] = await record_priority_patterns(runtime, store)
    result["promotions"] = await _promote_cognitive_followups(runtime, store)
    result["telemetry_affect"] = await _run_telemetry_affect(runtime, store)
    result["skills"] = await store.extract_skills_from_procedural_memory(
        getattr(runtime, "memory", None),
        limit=240,
    )

    await store.record_event(
        CognitiveEventKind.MAINTENANCE_OUTCOME,
        "Cognitive maintenance refreshed feedback-loop state",
        source="cognitive_maintenance",
        confidence=0.75,
        salience=1.1,
        payload=result,
    )
    result["events_recorded"] += 1

    trace = getattr(runtime, "_trace", None)
    if callable(trace):
        trace("cognitive_maintenance", result)
    return result


def _cognitive_store(runtime: Any) -> Any:
    return getattr(runtime, "cognitive_state_store", None) or getattr(
        getattr(runtime, "ctx", None),
        "cognitive_state_store",
        None,
    )


async def _refresh_attention(runtime: Any, store: Any) -> int:
    count = 0
    executive = getattr(runtime, "executive", None)
    intention = str(getattr(executive, "intention", "") or "").strip()
    if intention:
        await store.upsert_attention(
            intention,
            strength=0.82,
            source="executive_intention",
            evidence_refs=["executive:intention"],
        )
        count += 1
    for goal in list(getattr(executive, "active_goals", []) or [])[:3]:
        label = str(goal or "").strip()
        if not label:
            continue
        await store.upsert_attention(
            label,
            strength=0.65,
            source="executive_active_goal",
            evidence_refs=["executive:active_goals"],
        )
        count += 1
    activity = str(getattr(runtime, "_activity", "") or "").strip()
    if activity and activity != "idle":
        await store.upsert_attention(
            f"runtime activity: {activity}",
            strength=0.7,
            source="runtime_activity",
            evidence_refs=["runtime:activity"],
        )
        count += 1
    return count


async def _refresh_working_memory(runtime: Any, store: Any) -> int:
    count = 0
    activity = str(getattr(runtime, "_activity", "idle") or "idle")
    since = getattr(runtime, "_activity_since", None)
    content = f"Current runtime activity is {activity}."
    if isinstance(since, datetime):
        content += f" Started at {since.astimezone(timezone.utc).isoformat()}."
    await store.upsert_working_memory(
        "runtime_activity",
        content,
        priority=0.5 if activity == "idle" else 0.75,
        source="runtime_activity",
        evidence_refs=["runtime:activity"],
    )
    count += 1
    builder = getattr(runtime, "builder", None)
    latest_wellbeing = getattr(builder, "latest_wellbeing_state", None)
    if latest_wellbeing is not None:
        risk = float(getattr(latest_wellbeing, "overall_risk", 0.0) or 0.0)
        await store.upsert_working_memory(
            "wellbeing_risk",
            f"Latest wellbeing risk is {risk:.3f}; use it to pace and close loops, not to script feelings.",
            priority=min(0.9, 0.4 + risk),
            source="wellbeing_runtime",
            evidence_refs=["wellbeing:latest_state"],
        )
        count += 1
    return count


async def _record_affective_registry_trend(runtime: Any, store: Any) -> int:
    writer = getattr(getattr(runtime, "ctx", None), "affective_registry_writer", None)
    latest = getattr(writer, "get_latest", None)
    if not callable(latest):
        return 0
    try:
        entries = latest(25)
    except Exception:
        return 0
    if not entries:
        return 0
    tension = _avg(_extract_affect(entries, "tension"))
    fatigue = _avg(_extract_affect(entries, "fatigue"))
    certainty = _avg(_extract_affect(entries, "certainty"))
    focus = _avg(_extract_affect(entries, "focus"))
    summary = (
        f"Affective registry trend over {len(entries)} entries: "
        f"tension {tension:.2f}, fatigue {fatigue:.2f}, certainty {certainty:.2f}, focus {focus:.2f}."
    )
    await store.record_event(
        CognitiveEventKind.AFFECTIVE_TREND,
        summary,
        source="affective_registry",
        confidence=0.7,
        salience=1.2 + max(tension, fatigue),
        payload={
            "entry_count": len(entries),
            "avg_tension": tension,
            "avg_fatigue": fatigue,
            "avg_certainty": certainty,
            "avg_focus": focus,
        },
    )
    if tension >= 0.55 or fatigue >= 0.55 or certainty <= 0.35:
        await store.upsert_working_memory(
            "affective_risk",
            summary + " Treat this as pacing/verification pressure.",
            priority=0.7,
            source="affective_registry",
            evidence_refs=[f"affective_registry:{getattr(entries[-1], 'entry_id', '')}"],
        )
    return 1


async def _record_self_inspection_feedback(runtime: Any, store: Any) -> int:
    inspection_store = getattr(runtime, "self_inspection_store", None)
    list_recent = getattr(inspection_store, "list_recent", None)
    if not callable(list_recent):
        return 0
    try:
        recent = await list_recent(limit=16)
    except Exception:
        return 0
    if not recent:
        return 0
    recent = [
        record
        for record in recent
        if not bool((getattr(record, "meta", None) or {}).get("audit_only"))
    ]
    if not recent:
        return 0
    count = 0
    gap_counter: Counter[str] = Counter()
    drift_counter: Counter[str] = Counter()
    valence_counter: Counter[str] = Counter()
    for record in recent:
        for gap in getattr(record, "commitment_gaps", []) or []:
            status = str(getattr(getattr(gap, "status", ""), "value", getattr(gap, "status", "")))
            if status in {"open", "watch"}:
                promised = str(getattr(gap, "promised", "") or "").strip()
                if promised:
                    gap_counter[promised] += 1
        for drift in getattr(record, "drift_observations", []) or []:
            reason = str(getattr(drift, "reason", "") or "").strip()
            if reason:
                drift_counter[reason] += 1
        for source in getattr(record, "valence_sources", []) or []:
            label = str(getattr(source, "source", "") or "").strip()
            if label:
                valence_counter[label] += 1
    for promised, seen in gap_counter.most_common(3):
        await store.record_event(
            CognitiveEventKind.SELF_INSPECTION_GAP,
            f"Repeated unresolved commitment gap: {promised}",
            source="self_inspection",
            confidence=min(0.95, 0.45 + seen * 0.15),
            salience=1.5 + min(1.0, seen * 0.2),
            evidence_refs=[f"self_inspection:{getattr(record, 'record_id', '')}" for record in recent[-3:]],
            payload={"repeat_count": seen, "gap": promised},
        )
        await store.upsert_working_memory(
            "commitment_gap",
            f"Unresolved commitment gap needs action or proof: {promised}",
            priority=0.85,
            source="self_inspection",
            evidence_refs=[f"self_inspection:{getattr(recent[-1], 'record_id', '')}"],
        )
        count += 1
    for reason, seen in drift_counter.most_common(2):
        if seen < 2:
            continue
        await store.record_event(
            CognitiveEventKind.SURPRISE,
            f"Repeated response drift detected: {reason}",
            source="self_inspection",
            confidence=min(0.9, 0.4 + seen * 0.12),
            salience=1.3 + min(1.0, seen * 0.15),
            payload={"repeat_count": seen, "drift": reason},
        )
        await _record_repeated_drift_thread_bead(
            runtime,
            reason=reason,
            repeat_count=seen,
        )
        count += 1
    for label, seen in valence_counter.most_common(2):
        if seen < 2:
            continue
        await store.record_event(
            CognitiveEventKind.EMOTIONAL_MEMORY,
            f"Repeated valence source in recent turns: {label}",
            source="self_inspection",
            confidence=min(0.85, 0.4 + seen * 0.1),
            salience=1.2,
            payload={"repeat_count": seen, "source": label},
        )
        count += 1
    return count


async def _record_repeated_drift_thread_bead(
    runtime: Any,
    *,
    reason: str,
    repeat_count: int,
) -> bool:
    service = getattr(runtime, "thread_registry_service", None)
    ensure_anchor = getattr(service, "ensure_thread_anchor", None)
    create_bead = getattr(service, "create_candidate_bead", None)
    if not callable(ensure_anchor) or not callable(create_bead):
        return False

    compact_reason = " ".join(str(reason or "").split())
    if not compact_reason:
        return False
    source_key = _promotion_key("self_inspection_drift", compact_reason)
    source_ref = f"self_inspection:drift:{source_key}"
    content = "\n".join(
        [
            "Repeated self-inspection drift pattern",
            f"reason: {compact_reason}",
            "consumer: thread-registry continuity cues",
            (
                "instruction: treat as behavioral feedback; gather fresh evidence or "
                "change route before repeating the same framing."
            ),
        ]
    )
    try:
        anchor = await ensure_anchor(
            title="Repeated self-inspection drift",
            kind="self_inspection_drift",
            status="peripheral",
            anchor_id="self-inspection-drift",
        )
        await create_bead(
            thread_anchor_id=anchor.anchor_id,
            title=f"Drift pattern: {_short_title(compact_reason)}",
            summary=(
                "Repeated self-inspection drift pattern needs behavioral follow-through: "
                f"{compact_reason}"
            ),
            source_kind=BeadSourceKind.SELF_INSPECTION_DRIFT,
            source_ref=source_ref,
            content=content,
            user_commissioned=False,
        )
    except Exception as exc:
        trace = getattr(runtime, "_trace", None)
        if callable(trace):
            trace(
                "self_inspection_drift_thread_bead_failed",
                {"reason": compact_reason, "repeat_count": repeat_count, "error": str(exc)},
            )
        return False
    return True


async def _record_recall_failure_memories(runtime: Any, store: Any) -> int:
    memory = getattr(runtime, "memory", None)
    list_recent = getattr(memory, "list_recent_episodes", None)
    if not callable(list_recent):
        return 0
    try:
        episodes = await list_recent(limit=80)
    except Exception:
        return 0
    markers = ("failed to recall", "failed recall", "memory gap", "didn't remember", "did not remember")
    recorded = 0
    for episode in episodes:
        content = str(getattr(episode, "content", "") or "")
        if not any(marker in content.lower() for marker in markers):
            continue
        await store.record_event(
            CognitiveEventKind.RECALL_FAILURE_MEMORY,
            "Prior failed recall is autobiographical evidence to handle gently",
            content=content[:500],
            source="memory_episode",
            confidence=0.75,
            salience=1.4,
            evidence_refs=[f"episode:{getattr(episode, 'episode_id', '')}"],
            payload={"session_id": getattr(episode, "session_id", None)},
        )
        recorded += 1
        if recorded >= 3:
            break
    return recorded


async def _record_curiosity_state(runtime: Any, store: Any) -> int:
    graph = getattr(runtime, "fascination_graph", None)
    active = getattr(graph, "active", None)
    if not callable(active):
        return 0
    try:
        nodes = active(limit=5)
    except Exception:
        return 0
    count = 0
    for node in nodes[:3]:
        title = str(
            getattr(node, "title", "")
            or getattr(node, "label", "")
            or getattr(node, "summary", "")
            or ""
        ).strip()
        if not title:
            continue
        salience = float(getattr(node, "salience", 0.5) or 0.5)
        await store.record_event(
            CognitiveEventKind.CURIOSITY,
            f"Active curiosity/fascination trail: {title}",
            source="fascination_graph",
            confidence=min(0.9, 0.45 + salience * 0.4),
            salience=1.0 + salience,
            payload=_model_dump(node),
        )
        count += 1
    return count


async def _promote_curiosity_to_creative(runtime: Any, store: Any) -> int:
    graph = getattr(runtime, "fascination_graph", None)
    active = getattr(graph, "active", None)
    creative = getattr(runtime, "creative", None)
    add_work = getattr(creative, "add", None)
    if not callable(active) or not callable(add_work):
        return 0
    try:
        nodes = active(limit=5)
    except Exception:
        return 0
    promoted = 0
    for node in nodes:
        salience = float(getattr(node, "salience", 0.0) or 0.0)
        novelty = float(getattr(node, "novelty", 0.0) or 0.0)
        should_force = bool(getattr(node, "should_force_work", False))
        if not should_force and (salience < 0.72 or novelty < 0.45):
            continue
        summary = str(getattr(node, "summary", "") or "").strip()
        if not summary:
            continue
        key = _promotion_key("curiosity", str(getattr(node, "key", summary)))
        if await _curiosity_work_exists(runtime, key):
            continue
        from opencas.autonomy import WorkObject, WorkStage

        work = WorkObject(
            content=summary,
            stage=WorkStage.SPARK,
            promotion_score=max(salience, novelty),
            meta={
                "origin": "cognitive_curiosity",
                "promotion_key": key,
                "source": str(getattr(node, "source", "fascination_graph") or "fascination_graph"),
                "route": str(getattr(node, "route", "") or ""),
                "salience": salience,
                "novelty": novelty,
                "recurrence_count": int(getattr(node, "recurrence_count", 0) or 0),
            },
        )
        try:
            add_work(work)
        except Exception:
            continue
        await store.record_event(
            CognitiveEventKind.CURIOSITY,
            f"Curiosity promoted into creative ladder: {summary[:160]}",
            source="fascination_graph",
            confidence=max(0.55, min(0.9, salience)),
            salience=1.5 + max(salience, novelty) * 0.3,
            payload={"work_id": str(work.work_id), "promotion_key": key},
        )
        promoted += 1
    return promoted


async def _curiosity_work_exists(runtime: Any, promotion_key: str) -> bool:
    creative = getattr(runtime, "creative", None)
    for work in list(getattr(creative, "_ladder", []) or []):
        meta = getattr(work, "meta", {}) or {}
        if isinstance(meta, dict) and meta.get("promotion_key") == promotion_key:
            return True
    work_store = getattr(getattr(runtime, "ctx", None), "work_store", None)
    list_by_origin = getattr(work_store, "list_by_origin", None)
    if callable(list_by_origin):
        try:
            existing = await list_by_origin("cognitive_curiosity", limit=100)
        except Exception:
            existing = []
        for work in existing:
            meta = getattr(work, "meta", {}) or {}
            if isinstance(meta, dict) and meta.get("promotion_key") == promotion_key:
                return True
    return False


async def _record_habit_candidates(runtime: Any, store: Any) -> int:
    memory = getattr(runtime, "memory", None)
    list_recent = getattr(memory, "list_recent_episodes", None)
    if not callable(list_recent):
        return 0
    try:
        episodes = await list_recent(limit=160)
    except Exception:
        return 0
    normalized: Counter[str] = Counter()
    refs: dict[str, list[str]] = {}
    for episode in episodes:
        if str(getattr(getattr(episode, "kind", ""), "value", getattr(episode, "kind", ""))) != "turn":
            continue
        payload = getattr(episode, "payload", {}) or {}
        role = str(payload.get("role") or payload.get("message_role") or "").lower()
        if role and role != "user":
            continue
        text = _normalize_habit_text(str(getattr(episode, "content", "") or ""))
        if len(text) < 12:
            continue
        normalized[text] += 1
        refs.setdefault(text, []).append(f"episode:{getattr(episode, 'episode_id', '')}")
    for text, count in normalized.most_common(2):
        if count < 3:
            continue
        await store.record_event(
            CognitiveEventKind.HABIT,
            f"Possible repeated user routine: {text}",
            source="episodic_pattern",
            confidence=min(0.9, 0.35 + count * 0.12),
            salience=1.2 + min(1.0, count * 0.1),
            evidence_refs=refs.get(text, [])[:5],
            payload={"repeat_count": count},
        )
        return 1
    return 0


async def _record_counterfactual_need(runtime: Any, store: Any) -> int:
    memory = getattr(runtime, "memory", None)
    list_recent = getattr(memory, "list_recent_episodes", None)
    if not callable(list_recent):
        return 0
    try:
        episodes = await list_recent(limit=80)
    except Exception:
        return 0
    failure_markers = ("failed", "error", "blocked", "loop")
    failures = [
        ep
        for ep in episodes
        if any(marker in str(getattr(ep, "content", "") or "").lower() for marker in failure_markers)
    ]
    if len(failures) < 2:
        return 0
    failure_summary = "\n".join(str(getattr(ep, "content", "") or "")[:240] for ep in failures[:5])
    counterfactual = recommended_counterfactual(
        objective=str(getattr(getattr(runtime, "executive", None), "intention", "") or ""),
        failure_summary=failure_summary,
        prior_attempts=len(failures),
    )
    recommended = counterfactual.get("recommended") or {}
    await store.record_event(
        CognitiveEventKind.COUNTERFACTUAL,
        "Recent failures or blocked loops produced a scored counterfactual retry plan",
        source="episodic_failure_pattern",
        confidence=0.72,
        salience=1.6,
        evidence_refs=[f"episode:{getattr(ep, 'episode_id', '')}" for ep in failures[:5]],
        payload={"failure_count": len(failures), "counterfactual": counterfactual},
    )
    await store.upsert_working_memory(
        "counterfactual_review",
        (
            "Before repeating a recently failed path, use the scored counterfactual "
            f"recommendation: {recommended.get('strategy', 'narrow_scope')} - "
            f"{recommended.get('action', 'compare alternatives against evidence')}"
        ),
        priority=0.78,
        source="episodic_failure_pattern",
        evidence_refs=[f"episode:{getattr(ep, 'episode_id', '')}" for ep in failures[:3]],
        payload={"counterfactual": counterfactual},
    )
    return 1


async def _record_somatic_strategy_pressure(runtime: Any, store: Any) -> int:
    somatic = getattr(getattr(runtime, "ctx", None), "somatic", None)
    state = getattr(somatic, "state", None)
    if state is None:
        return 0
    tension = float(getattr(state, "tension", 0.0) or 0.0)
    fatigue = float(getattr(state, "fatigue", 0.0) or 0.0)
    certainty = float(getattr(state, "certainty", 0.5) or 0.5)
    valence = float(getattr(state, "valence", 0.0) or 0.0)
    if tension < 0.55 and fatigue < 0.65 and certainty > 0.35:
        return 0

    summary = (
        "Somatic strategy pressure: "
        f"tension {tension:.2f}, fatigue {fatigue:.2f}, certainty {certainty:.2f}, valence {valence:.2f}."
    )
    strategy = (
        summary
        + " Use slower tool loops, explicit proof checks, and direct uncertainty labels "
        "instead of smoothing over pressure."
    )
    await store.record_event(
        CognitiveEventKind.SOMATIC_STRATEGY,
        summary,
        content=strategy,
        source="somatic_state",
        confidence=0.72,
        salience=1.4 + max(tension, fatigue, 1.0 - certainty) * 0.4,
        payload={
            "tension": tension,
            "fatigue": fatigue,
            "certainty": certainty,
            "valence": valence,
            "strategy": "slow_verify_label_uncertainty",
        },
    )
    await store.upsert_working_memory(
        "somatic_strategy",
        strategy,
        priority=min(0.9, 0.55 + max(tension, fatigue, 1.0 - certainty) * 0.3),
        source="somatic_state",
        evidence_refs=["somatic:current_state"],
    )
    return 1


async def _record_uncertainty_pressure(runtime: Any, store: Any) -> int:
    signals: list[str] = []
    builder = getattr(runtime, "builder", None)
    latest_wellbeing = getattr(builder, "latest_wellbeing_state", None)
    if latest_wellbeing is not None:
        uncertainty = float(getattr(latest_wellbeing, "uncertainty_load", 0.0) or 0.0)
        if uncertainty >= 0.55:
            signals.append(f"wellbeing uncertainty_load {uncertainty:.2f}")
    somatic = getattr(getattr(runtime, "ctx", None), "somatic", None)
    state = getattr(somatic, "state", None)
    certainty = float(getattr(state, "certainty", 0.5) or 0.5) if state is not None else 0.5
    if certainty <= 0.35:
        signals.append(f"somatic certainty {certainty:.2f}")
    tom = getattr(runtime, "tom", None)
    check = getattr(tom, "check_consistency", None)
    if callable(check):
        try:
            consistency = check()
            if not bool(getattr(consistency, "consistent", True)):
                signals.append("ToM consistency gaps")
        except Exception:
            pass
    if not signals:
        return 0

    summary = "Uncertainty pressure should trigger evidence seeking: " + "; ".join(signals)
    evidence_policy = {
        "required_before_assertion": True,
        "preferred_tools": [
            "search_memories",
            "recall_autobiography",
            "recall_concepts",
            "artifact_lookup",
            "runtime_status",
            "workflow_status",
            "fs_read_file",
            "fs_list_dir",
            "web_search",
            "web_fetch",
        ],
        "allowed_resolution": "Use a relevant evidence tool or ask one focused question when evidence is unavailable.",
    }
    await store.record_event(
        CognitiveEventKind.UNCERTAINTY_SEEKING,
        summary,
        source="cognitive_maintenance",
        confidence=0.72,
        salience=1.45,
        payload={"signals": signals, "evidence_policy": evidence_policy},
    )
    await store.upsert_working_memory(
        "uncertainty_seeking",
        (
            summary
            + ". Do not finalize uncertain claims until a relevant evidence tool has been "
            "used, or until a focused question has been asked because the evidence path is unavailable."
        ),
        priority=0.82,
        source="cognitive_maintenance",
        evidence_refs=["cognition:uncertainty_pressure"],
        payload={"signals": signals, "evidence_policy": evidence_policy},
    )
    return 1


async def _record_social_agent_models(runtime: Any, store: Any) -> int:
    tom = getattr(runtime, "tom", None)
    list_models = getattr(tom, "list_agent_models", None)
    if not callable(list_models):
        return 0
    try:
        models = list_models(limit=5)
    except Exception:
        return 0
    if not models:
        return 0
    strongest = models[0]
    label = str(strongest.get("entity_label") or strongest.get("entity_id") or "agent")
    beliefs = strongest.get("beliefs") if isinstance(strongest.get("beliefs"), list) else []
    intentions = strongest.get("intentions") if isinstance(strongest.get("intentions"), list) else []
    summary = (
        f"Third-party agent model active for {label}: "
        f"{len(beliefs)} beliefs and {len(intentions)} intentions available."
    )
    await store.record_event(
        CognitiveEventKind.SOCIAL_MODEL,
        summary,
        source="tom",
        confidence=0.72,
        salience=1.25,
        payload={"agent_models": models},
    )
    await store.upsert_working_memory(
        "social_agent_models",
        summary + " Keep this separate from self and operator/user beliefs.",
        priority=0.62,
        source="tom",
        evidence_refs=[f"tom:agent:{strongest.get('entity_id')}"],
        payload={"agent_models": models[:3]},
    )
    return 1


async def _record_narrative_summary(runtime: Any, store: Any) -> int:
    events = await store.list_recent_events(limit=12)
    if not events:
        return 0
    strongest = sorted(events, key=lambda item: (item.salience, item.created_at), reverse=True)[:4]
    theme_counts = Counter(event.kind.value for event in events)
    summary = "Recent cognitive arc: " + "; ".join(
        f"{event.kind.value}: {event.summary}" for event in strongest
    )
    evidence_refs = [f"cognitive_event:{event.event_id}" for event in strongest]
    await store.record_event(
        CognitiveEventKind.NARRATIVE,
        summary[:700],
        source="cognitive_maintenance",
        confidence=0.65,
        salience=1.0,
        evidence_refs=evidence_refs,
        payload={
            "theme_counts": dict(theme_counts),
            "strongest_event_ids": [str(event.event_id) for event in strongest],
        },
    )
    identity = getattr(getattr(runtime, "ctx", None), "identity", None)
    record_breadcrumb = getattr(identity, "record_continuity_breadcrumb", None)
    if callable(record_breadcrumb):
        try:
            record_breadcrumb(
                intent="Integrate recent cognitive feedback into continuity",
                decision=summary[:320],
                next_step="Use this narrative only as evidence-grounded context for future action.",
                note="Recorded by cognitive maintenance narrative construction.",
            )
        except Exception:
            pass
    record_self_knowledge = getattr(identity, "record_self_knowledge", None)
    if callable(record_self_knowledge):
        recent_arc_value = {
            "summary": summary[:700],
            "theme_counts": dict(theme_counts),
            "evidence_refs": evidence_refs,
        }
        try:
            registry = getattr(identity, "registry", None)
            duplicate = (
                registry is not None
                and callable(getattr(registry, "recent_value_hash_seen", None))
                and registry.recent_value_hash_seen(
                    "cognitive_narrative",
                    "recent_arc",
                    recent_arc_value,
                    limit=3,
                    exclude_keys=["evidence_refs"],
                )
            )
            if duplicate:
                bus = getattr(runtime, "cognition_bus", None)
                publish = getattr(bus, "publish", None)
                content_hash = registry.value_hash(
                    recent_arc_value,
                    exclude_keys=["evidence_refs"],
                )
                if callable(publish):
                    await publish(
                        NarratorRedundancyObserved(
                            kind="narrator.redundancy_observed",
                            source="cognitive_runtime._record_narrative_summary",
                            payload={
                                "domain": "cognitive_narrative",
                                "key": "recent_arc",
                                "content_hash": content_hash,
                                "skipped": True,
                                "summary": summary[:240],
                                "evidence_refs": evidence_refs,
                            },
                            evidence_ids=evidence_refs,
                        )
                    )
            else:
                meta = {
                    "source": "cognitive_maintenance",
                    "comparison_hash": registry.value_hash(
                        recent_arc_value,
                        exclude_keys=["evidence_refs"],
                    )
                    if registry is not None
                    and callable(getattr(registry, "value_hash", None))
                    else None,
                }
                meta = {key: value for key, value in meta.items() if value is not None}
                record_self_knowledge(
                    domain="cognitive_narrative",
                    key="recent_arc",
                    value=recent_arc_value,
                    confidence=0.7,
                    evidence_ids=evidence_refs,
                    meta=meta,
                )
        except Exception:
            pass
    set_recent_themes = getattr(identity, "set_recent_themes", None)
    if callable(set_recent_themes):
        try:
            set_recent_themes(
                [
                    {"term": kind, "count": count}
                    for kind, count in theme_counts.most_common(10)
                ]
            )
        except Exception:
            pass
    return 1


async def _promote_cognitive_followups(runtime: Any, store: Any) -> dict[str, Any]:
    schedule_service = getattr(runtime, "schedule_service", None) or getattr(
        getattr(runtime, "ctx", None),
        "schedule_service",
        None,
    )
    create_schedule = getattr(schedule_service, "create_schedule", None)
    list_items = getattr(getattr(schedule_service, "store", None), "list_items", None)
    if not callable(create_schedule) or not callable(list_items):
        return {"available": False, "reason": "schedule_service_unavailable"}

    now = datetime.now(timezone.utc)
    result = {
        "available": True,
        "prospective_promoted": 0,
        "self_inspection_promoted": 0,
        "habit_promoted": 0,
        "skipped_existing": 0,
    }
    prospective = await store.list_prospective_memories(limit=16)
    for item in prospective:
        if getattr(item, "proof_ref", ""):
            continue
        trigger_at = getattr(item, "trigger_at", None)
        if trigger_at is None:
            continue
        if float(getattr(item, "confidence", 0.0) or 0.0) < 0.55:
            continue
        promotion_key = _promotion_key("prospective_memory", item.action, item.condition)
        if await _promotion_schedule_exists(list_items, promotion_key):
            result["skipped_existing"] += 1
            continue
        item_payload = dict(getattr(item, "payload", {}) or {})
        schedule = await _create_cognitive_schedule(
            create_schedule,
            title=f"Prospective memory: {_short_title(item.action)}",
            objective=(
                f"Follow through on this prospective memory: {item.action}. "
                "Return or store proof of completion, or record the concrete blocker."
            ),
            start_at=trigger_at,
            promotion_key=promotion_key,
            promotion_kind="prospective_memory",
            evidence_refs=list(getattr(item, "evidence_refs", []) or []),
            meta_extra={"intent_id": item.intent_id, "condition": item.condition},
            priority=float(item_payload.get("schedule_priority") or 7.0),
            tags=list(item_payload.get("schedule_tags") or []),
        )
        await store.upsert_prospective_memory(
            item.action,
            trigger_at=trigger_at,
            condition=item.condition,
            proof_ref=f"schedule:{schedule.schedule_id}",
            confidence=item.confidence,
            evidence_refs=list(getattr(item, "evidence_refs", []) or []),
            payload={**dict(getattr(item, "payload", {}) or {}), "promoted_schedule_id": str(schedule.schedule_id)},
        )
        result["prospective_promoted"] += 1

    gaps = await store.list_recent_events(
        kind=CognitiveEventKind.SELF_INSPECTION_GAP,
        status="active",
        limit=12,
    )
    for event in gaps:
        if float(getattr(event, "confidence", 0.0) or 0.0) < 0.65:
            continue
        promotion_key = _promotion_key("self_inspection_gap", event.summary)
        if await _promotion_schedule_exists(list_items, promotion_key):
            result["skipped_existing"] += 1
            continue
        await _create_cognitive_schedule(
            create_schedule,
            title=f"Metacognitive follow-up: {_short_title(event.summary)}",
            objective=(
                f"Resolve this repeated self-inspection gap: {event.summary}. "
                "Check durable evidence, repair the underlying loop if safe, and create "
                "proof or a bounded blocker note."
            ),
            start_at=now + timedelta(minutes=15),
            promotion_key=promotion_key,
            promotion_kind="self_inspection_gap",
            evidence_refs=list(getattr(event, "evidence_refs", []) or []),
            meta_extra={"cognitive_event_id": str(event.event_id)},
        )
        result["self_inspection_promoted"] += 1

    habits = await store.list_recent_events(kind=CognitiveEventKind.HABIT, status="active", limit=12)
    for event in habits:
        payload = dict(getattr(event, "payload", {}) or {})
        repeat_count = int(payload.get("repeat_count") or 0)
        if repeat_count < 3 or float(getattr(event, "confidence", 0.0) or 0.0) < 0.65:
            continue
        promotion_key = _promotion_key("habit", event.summary)
        if await _promotion_schedule_exists(list_items, promotion_key):
            result["skipped_existing"] += 1
            continue
        await _create_cognitive_schedule(
            create_schedule,
            title=f"Habit review: {_short_title(event.summary)}",
            objective=(
                f"Review this possible repeated routine: {event.summary}. "
                "Only propose or create automation if evidence supports usefulness and safety."
            ),
            start_at=now + timedelta(hours=12),
            promotion_key=promotion_key,
            promotion_kind="habit",
            evidence_refs=list(getattr(event, "evidence_refs", []) or []),
            meta_extra={"cognitive_event_id": str(event.event_id), "repeat_count": repeat_count},
        )
        result["habit_promoted"] += 1
    return result


async def _run_telemetry_affect(runtime: Any, store: Any) -> dict[str, Any]:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    state_dir = getattr(config, "state_dir", None)
    if state_dir is None:
        return {"available": False, "reason": "state_dir_unavailable"}
    affect_store = getattr(runtime, "telemetry_affect_store", None)
    if affect_store is None:
        affect_store = AffectStore(Path(state_dir) / "telemetry_affect")
        runtime.telemetry_affect_store = affect_store
    sync_result = await _sync_affective_registry_to_telemetry(runtime, affect_store)
    quality_result = await _sync_execution_receipts_to_quality(runtime, affect_store)
    graph_result = _build_telemetry_affect_graph(affect_store)
    engine = InterventionEngine(
        affect_store,
        AnomalyDetector(affect_store),
    )
    try:
        alerts = engine.run_health_check()
        dashboard = engine.get_team_health_dashboard()
    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc),
            "registry_sync": sync_result,
            "quality_sync": quality_result,
            "graph": graph_result,
        }
    if (
        alerts
        or dashboard.get("unresolved_alerts")
        or sync_result.get("snapshots_created")
        or sync_result.get("trajectories_created")
        or quality_result.get("signals_created")
        or graph_result.get("edges_created")
    ):
        await store.record_event(
            CognitiveEventKind.TELEMETRY_AFFECT,
            (
                "Telemetry affect processed "
                f"{sync_result.get('snapshots_created', 0)} new snapshots, "
                f"{sync_result.get('trajectories_created', 0)} trajectories, and "
                f"{graph_result.get('edges_created', 0)} graph edges, with "
                f"{dashboard.get('unresolved_alerts', 0)} unresolved alerts."
            ),
            source="telemetry_affect",
            confidence=0.7,
            salience=1.4 + min(1.0, float(dashboard.get("critical_alerts", 0) or 0) * 0.3),
            payload={
                "dashboard": dashboard,
                "registry_sync": sync_result,
                "quality_sync": quality_result,
                "graph": graph_result,
            },
        )
    return {
        "available": True,
        "alerts_created": len(alerts),
        "dashboard": dashboard,
        "registry_sync": sync_result,
        "quality_sync": quality_result,
        "graph": graph_result,
    }


async def _sync_affective_registry_to_telemetry(runtime: Any, affect_store: AffectStore) -> dict[str, Any]:
    writer = getattr(getattr(runtime, "ctx", None), "affective_registry_writer", None)
    latest = getattr(writer, "get_latest", None)
    if not callable(latest):
        return {"available": False, "reason": "affective_registry_unavailable"}
    try:
        entries = latest(80)
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    if not entries:
        return {"available": True, "entries_seen": 0, "snapshots_created": 0, "trajectories_created": 0}

    seen_entry_ids: set[str] = set()
    try:
        for snapshot in affect_store.iter_snapshots():
            entry_id = str(snapshot.source_payload.get("affective_registry_entry_id") or "")
            if entry_id:
                seen_entry_ids.add(entry_id)
    except Exception:
        seen_entry_ids = set()

    created: list[AffectSnapshot] = []
    for entry in entries:
        entry_id = str(getattr(entry, "entry_id", "") or "")
        if not entry_id or entry_id in seen_entry_ids:
            continue
        snapshot = _affective_entry_to_snapshot(entry)
        try:
            affect_store.save_snapshot(snapshot)
        except Exception:
            continue
        created.append(snapshot)

    trajectories_created = 0
    by_session: dict[str, list[AffectSnapshot]] = {}
    for snapshot in created:
        by_session.setdefault(snapshot.session_id, []).append(snapshot)
    for session_id, new_snapshots in by_session.items():
        try:
            snapshots = sorted(
                affect_store.get_snapshots_for_session(session_id),
                key=lambda item: item.timestamp,
            )
        except Exception:
            snapshots = sorted(new_snapshots, key=lambda item: item.timestamp)
        if len(snapshots) < 2:
            continue
        try:
            artifact_id = snapshots[-1].artifact_id or "runtime"
            trajectory = compute_trajectory_from_snapshots(
                snapshots,
                session_id=session_id,
                artifact_id=artifact_id,
                actor="opencas_agent",
            )
            affect_store.save_trajectory(trajectory)
            trajectories_created += 1
        except Exception:
            continue
    return {
        "available": True,
        "entries_seen": len(entries),
        "snapshots_created": len(created),
        "trajectories_created": trajectories_created,
    }


def _affective_entry_to_snapshot(entry: Any) -> AffectSnapshot:
    affective = getattr(entry, "affective_state", None)
    context = getattr(entry, "execution_context", None)
    session_id = str(getattr(context, "session_id", "") or "runtime")
    tension = _bounded_unit(getattr(affective, "tension", 0.0))
    fatigue = _bounded_unit(getattr(affective, "fatigue", 0.0))
    focus = _bounded_unit(getattr(affective, "focus", 0.5))
    certainty = _bounded_unit(getattr(affective, "certainty", 0.5))
    arousal = _bounded_unit(getattr(affective, "arousal", 0.5))
    dimensions = {
        AffectDimension.VALENCE.value: _bounded_signed(getattr(affective, "valence", 0.0)),
        AffectDimension.AROUSAL.value: arousal,
        AffectDimension.CERTAINTY.value: (certainty * 2.0) - 1.0,
        AffectDimension.COHERENCE.value: (focus * 2.0) - 1.0,
        AffectDimension.URGENCY.value: max(tension, fatigue * 0.7),
    }
    raw = (
        f"phase={getattr(getattr(entry, 'phase', ''), 'value', getattr(entry, 'phase', ''))}; "
        f"emotion={getattr(affective, 'primary_emotion', 'neutral')}; "
        f"tension={tension:.3f}; fatigue={fatigue:.3f}; certainty={certainty:.3f}"
    )
    payload = getattr(entry, "payload", {}) if isinstance(getattr(entry, "payload", None), dict) else {}
    return AffectSnapshot(
        timestamp=getattr(entry, "timestamp", datetime.now(timezone.utc)),
        session_id=session_id,
        artifact_id=str(payload.get("artifact_id") or "runtime"),
        actor="opencas_agent",
        dimensions=dimensions,
        source=MoodSignalSource.INFERRED_FROM_PATTERN,
        source_payload={
            "source": "affective_registry",
            "affective_registry_entry_id": str(getattr(entry, "entry_id", "")),
            "phase": str(getattr(getattr(entry, "phase", ""), "value", getattr(entry, "phase", ""))),
            "payload": payload,
        },
        raw_signal=raw,
        signal_hash=_promotion_key("affective_registry_entry", str(getattr(entry, "entry_id", ""))),
        confidence=0.72,
    )


async def _sync_execution_receipts_to_quality(runtime: Any, affect_store: AffectStore) -> dict[str, Any]:
    receipt_store = getattr(getattr(runtime, "ctx", None), "receipt_store", None)
    list_recent = getattr(receipt_store, "list_recent", None)
    if not callable(list_recent):
        return {"available": False, "reason": "receipt_store_unavailable"}
    try:
        receipts = await list_recent(limit=40)
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    existing_receipt_ids: set[str] = set()
    try:
        for signal in affect_store.iter_quality_signals(artifact_id="runtime"):
            payload_id = str(getattr(signal, "session_id", "") or "")
            if payload_id:
                existing_receipt_ids.add(payload_id)
    except Exception:
        existing_receipt_ids = set()

    created = 0
    for receipt in receipts:
        receipt_id = str(getattr(receipt, "receipt_id", "") or "")
        if not receipt_id or receipt_id in existing_receipt_ids:
            continue
        success = bool(getattr(receipt, "success", False))
        verification = getattr(receipt, "verification_result", None)
        phases = list(getattr(receipt, "phases", []) or [])
        signal = QualitySignal(
            artifact_id="runtime",
            session_id=receipt_id,
            ci_failure_rate=0.0 if success else 1.0,
            review_rounds=len(phases) if phases else None,
            test_flakiness=0.0 if verification is True else 1.0 if verification is False else None,
            rollback_frequency=0.0 if success else 0.2,
        )
        signal.composite_quality = signal.compute_composite()
        try:
            affect_store.save_quality_signal(signal)
        except Exception:
            continue
        created += 1
    return {"available": True, "receipts_seen": len(receipts), "signals_created": created}


def _build_telemetry_affect_graph(affect_store: AffectStore) -> dict[str, Any]:
    graph = AffectQualityGraph(affect_store)
    try:
        edges = graph.build_edges_for_artifact("runtime")
        patterns = graph.find_predictive_patterns(min_correlation=0.5)
        risk = graph.get_artifact_risk_profile("runtime")
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    return {
        "available": True,
        "edges_created": len(edges),
        "predictive_patterns": patterns[:5],
        "runtime_risk_profile": risk,
    }


def _extract_affect(entries: list[Any], name: str) -> list[float]:
    values: list[float] = []
    for entry in entries:
        state = getattr(entry, "affective_state", None)
        value = getattr(state, name, None)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bounded_unit(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _bounded_signed(value: Any) -> float:
    try:
        return max(-1.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _model_dump(value: Any) -> dict[str, Any]:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if isinstance(value, dict):
        return value
    return {}


def _normalize_habit_text(text: str) -> str:
    compact = " ".join(text.lower().split())
    for prefix in ("jarrod:", "user:", "operator:"):
        if compact.startswith(prefix):
            compact = compact[len(prefix) :].strip()
    return compact[:120]


async def _create_cognitive_schedule(
    create_schedule: Any,
    *,
    title: str,
    objective: str,
    start_at: datetime,
    promotion_key: str,
    promotion_kind: str,
    evidence_refs: list[str],
    meta_extra: dict[str, Any],
    priority: float = 7.0,
    tags: list[str] | None = None,
) -> Any:
    from opencas.scheduling.models import ScheduleAction, ScheduleKind, ScheduleRecurrence

    return await create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title=title,
        description=f"Promoted by cognitive maintenance from {promotion_kind}.",
        objective=objective,
        start_at=start_at,
        recurrence=ScheduleRecurrence.NONE,
        priority=max(1.0, min(10.0, priority)),
        tags=["cognitive-maintenance", promotion_kind, *(tags or [])],
        meta={
            "source": "cognitive_maintenance",
            "promotion_kind": promotion_kind,
            "promotion_key": promotion_key,
            "evidence_refs": evidence_refs[:10],
            **meta_extra,
        },
    )


async def _promotion_schedule_exists(list_items: Any, promotion_key: str) -> bool:
    from opencas.scheduling.models import ScheduleStatus

    try:
        items = await list_items(status=ScheduleStatus.ACTIVE, limit=1000)
    except Exception:
        return False
    for item in items:
        meta = getattr(item, "meta", {}) or {}
        if isinstance(meta, dict) and meta.get("promotion_key") == promotion_key:
            return True
    return False


def _promotion_key(*parts: str) -> str:
    material = "\x1f".join(str(part or "").strip().lower() for part in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _short_title(value: str, *, max_chars: int = 72) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
