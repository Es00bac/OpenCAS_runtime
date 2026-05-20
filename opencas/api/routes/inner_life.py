"""Diagnostics and bounded manual proof actions for inner-life substrate wiring."""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter

from opencas.proof_chain import summarize_claims


def build_inner_life_router(runtime: Any) -> APIRouter:
    """Expose configured/empty/populated state for inner-life stores."""

    router = APIRouter(prefix="/api/inner-life", tags=["inner-life"])

    @router.get("/summary")
    async def summary() -> dict[str, Any]:
        daydreams = await _daydream_summary(runtime)
        daydream_signals = await _daydream_signal_summary(runtime)
        context_proposals = await _context_proposal_summary(runtime)
        thread_registry = await _thread_registry_summary(runtime)
        wellbeing = await _wellbeing_summary(runtime)
        cognition = await _cognition_summary(runtime)
        dreaming = await _dreaming_summary(runtime)
        proof_chain = await _proof_summary(runtime)
        return {
            "available": True,
            "daydreams": daydreams,
            "daydream_signals": daydream_signals,
            "context_proposals": context_proposals,
            "thread_registry": thread_registry,
            "wellbeing": wellbeing,
            "cognition": cognition,
            "dreaming": dreaming,
            "proof_chain": proof_chain,
            "configured_empty_count": sum(
                1
                for item in (
                    daydreams,
                    daydream_signals,
                    context_proposals,
                    thread_registry,
                    wellbeing,
                    cognition,
                    dreaming,
                    proof_chain,
                )
                if item.get("configured_empty")
            ),
        }

    @router.get("/runtime-truth")
    async def runtime_truth(
        include_all: bool = False,
        limit: int = 40,
        offset: int = 0,
    ) -> dict[str, Any]:
        return await _runtime_truth_packet(
            runtime,
            include_all=include_all,
            limit=limit,
            offset=offset,
        )

    @router.post("/nightly-dream")
    async def nightly_dream(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = dict(payload or {})
        runner = getattr(runtime, "run_nightly_dream", None)
        if not callable(runner):
            return {"available": False, "reason": "nightly_dreaming_unavailable"}
        now = datetime.now(timezone.utc)
        consolidation_result = dict(getattr(runtime, "_last_consolidation_result", None) or {})
        if not consolidation_result:
            manual_bucket = now.replace(second=0, microsecond=0)
            consolidation_result = {
                "result_id": f"manual-nightly-dream-{manual_bucket.isoformat()}",
                "timestamp": now.isoformat(),
                "manual_truth_check": True,
                "reason": str(payload.get("reason") or "manual inner-life dream trigger"),
            }
        result = runner(
            mode=str(payload.get("mode") or "light"),
            consolidation_result=consolidation_result,
        )
        if inspect.isawaitable(result):
            result = await result
        proof_store = getattr(runtime, "proof_store", None)
        dream_claims = []
        if proof_store is not None:
            dream_claims = await proof_store.list_claims(claim_type="dream", limit=20)
        return {
            **dict(result or {}),
            "side_effectful": True,
            "side_effects": [
                "dream_record",
                "dream_artifact",
                "proof_claim",
            ],
            "proof": {
                "dream_claims": len(dream_claims),
                **summarize_claims(dream_claims),
            },
        }

    return router


async def _daydream_summary(runtime: Any) -> dict[str, Any]:
    store = getattr(getattr(runtime, "ctx", None), "daydream_store", None)
    if store is None:
        return {"available": False, "recent_reflections": 0, "recent_thoughts": 0}
    recent = await _call_list(store, "list_recent", limit=8)
    thought_count = sum(len(getattr(item, "thoughts", []) or []) for item in recent)
    return {
        "available": True,
        "recent_reflections": len(recent),
        "recent_thoughts": thought_count,
        "configured_empty": len(recent) == 0,
        "thoughts_empty": thought_count == 0,
    }


async def _daydream_signal_summary(runtime: Any) -> dict[str, Any]:
    store = (
        getattr(getattr(runtime, "ctx", None), "daydream_signal_store", None)
        or getattr(runtime, "daydream_signal_store", None)
    )
    if store is None:
        return {"available": False, "recent_signals": 0, "recent_receipts": 0}
    signals = await _call_list(store, "list_recent", limit=8)
    receipts = await _call_list(store, "list_receipts", limit=8)
    return {
        "available": True,
        "recent_signals": len(signals),
        "recent_receipts": len(receipts),
        "configured_empty": len(signals) == 0 and len(receipts) == 0,
    }


async def _context_proposal_summary(runtime: Any) -> dict[str, Any]:
    store = (
        getattr(runtime, "context_proposals", None)
        or getattr(getattr(runtime, "ctx", None), "context_proposal_store", None)
    )
    if store is None:
        return {
            "available": False,
            "recent_proposals": 0,
            "status_counts": {},
        }
    recent = await _call_list(store, "list_recent", limit=8)
    count_by_status = getattr(store, "count_by_status", None)
    try:
        status_counts = dict(await count_by_status()) if callable(count_by_status) else {}
    except Exception:
        status_counts = {}
    return {
        "available": True,
        "recent_proposals": len(recent),
        "status_counts": status_counts,
        "configured_empty": len(recent) == 0 and not status_counts,
    }


async def _thread_registry_summary(runtime: Any) -> dict[str, Any]:
    store = getattr(runtime, "thread_registry_store", None)
    if store is None:
        return {"available": False, "recent_beads": 0}
    beads = await _call_list(store, "list_beads", limit=8)
    return {
        "available": True,
        "recent_beads": len(beads),
        "configured_empty": len(beads) == 0,
    }


async def _wellbeing_summary(runtime: Any) -> dict[str, Any]:
    store = getattr(runtime, "wellbeing_store", None)
    if store is None:
        return {"available": False, "recent_states": 0, "maintenance_outcomes": 0}
    states = await _call_list(store, "list_states", limit=5)
    outcomes = await _call_list(store, "list_maintenance_outcomes", limit=8)
    recommendations = await _call_list(store, "list_recommendations", limit=8)
    return {
        "available": True,
        "recent_states": len(states),
        "recommendations": len(recommendations),
        "maintenance_outcomes": len(outcomes),
        "configured_empty": not states and not outcomes and not recommendations,
    }


async def _cognition_summary(runtime: Any) -> dict[str, Any]:
    store = getattr(runtime, "cognitive_state_store", None) or getattr(
        getattr(runtime, "ctx", None),
        "cognitive_state_store",
        None,
    )
    if store is None:
        return {"available": False, "recent_events": 0}
    events = await _call_list(store, "list_recent_events", limit=8)
    attention = await _call_list(store, "list_attention", limit=5)
    working = await _call_list(store, "list_working_memory", limit=5)
    skills = await _call_list(store, "list_learned_skills", limit=5)
    return {
        "available": True,
        "recent_events": len(events),
        "attention_targets": len(attention),
        "working_memory_items": len(working),
        "learned_skills": len(skills),
        "configured_empty": not events and not attention and not working and not skills,
    }


async def _dreaming_summary(runtime: Any) -> dict[str, Any]:
    store = getattr(runtime, "dream_store", None)
    if store is None:
        return {"available": False, "recent_dreams": 0}
    dreams = await _call_list(store, "list_recent", limit=8)
    return {
        "available": True,
        "recent_dreams": len(dreams),
        "configured_empty": len(dreams) == 0,
    }


async def _proof_summary(runtime: Any) -> dict[str, Any]:
    store = getattr(runtime, "proof_store", None)
    if store is None:
        return {"available": False, "recent_claims": 0}
    claims = await _call_list(store, "list_claims", limit=20)
    return {
        "available": True,
        "recent_claims": len(claims),
        **summarize_claims(claims),
        "configured_empty": len(claims) == 0,
    }


async def _runtime_truth_packet(
    runtime: Any,
    *,
    include_all: bool = False,
    limit: int = 40,
    offset: int = 0,
) -> dict[str, Any]:
    profile = getattr(runtime, "agent_profile", None)
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    tools = _tool_entries(runtime)
    capabilities = _capability_entries(runtime)
    if include_all:
        daydreams = await _daydream_summary(runtime)
        daydream_signals = await _daydream_signal_summary(runtime)
        context_proposals = await _context_proposal_summary(runtime)
        thread_registry = await _thread_registry_summary(runtime)
        wellbeing = await _wellbeing_summary(runtime)
        dreaming = await _dreaming_summary(runtime)
        proof_chain = await _proof_summary(runtime)
        summary_mode = "full"
    else:
        daydreams = _availability_summary(runtime, "ctx.daydream_store")
        daydream_signals = _availability_summary(runtime, "ctx.daydream_signal_store")
        context_proposals = _availability_summary(runtime, "ctx.context_proposal_store")
        thread_registry = _availability_summary(runtime, "thread_registry_store")
        wellbeing = _availability_summary(runtime, "wellbeing_store")
        dreaming = _availability_summary(runtime, "dream_store")
        proof_chain = _availability_summary(runtime, "proof_store")
        summary_mode = "availability_fast"
    inner_life = {
        "daydreams": daydreams,
        "daydream_signals": daydream_signals,
        "context_proposals": context_proposals,
        "thread_registry": thread_registry,
        "wellbeing": wellbeing,
        "dreaming": dreaming,
        "proof_chain": proof_chain,
    }
    configured_empty = [
        name for name, payload in inner_life.items() if payload.get("configured_empty")
    ]
    offset = max(0, int(offset or 0))
    limit = max(1, min(500, int(limit or 40)))
    if include_all:
        visible_tools = tools
        visible_capabilities = capabilities
    else:
        visible_tools = tools[offset : offset + limit]
        visible_capabilities = capabilities[offset : offset + limit]
    return {
        "available": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agent": {
            "profile_id": getattr(profile, "profile_id", ""),
            "display_name": getattr(profile, "display_name", ""),
            "autonomy_style": getattr(profile, "autonomy_style", ""),
        },
        "workspace": {
            "session_id": getattr(config, "session_id", ""),
            "state_dir": str(getattr(config, "state_dir", "")),
            "primary_root": str(config.primary_workspace_root()) if config is not None else "",
            "managed_root": str(config.agent_workspace_root()) if config is not None else "",
        },
        "capabilities": {
            "tool_count": len(tools),
            "capability_count": len(capabilities),
            "tools": visible_tools,
            "capabilities": visible_capabilities,
            "inventory_offset": 0 if include_all else offset,
            "inventory_limit": None if include_all else limit,
            "inventory_truncated": not include_all
            and (offset + limit < len(tools) or offset + limit < len(capabilities)),
        },
        "tool_count": len(tools),
        "tools": visible_tools,
        "accessible_tool_count": len(tools),
        "accessible_tools": visible_tools,
        "capability_count": len(capabilities),
        "accessible_capability_count": len(capabilities),
        "accessible_capabilities": visible_capabilities,
        "inner_life": inner_life,
        "caveats": {
            "summary_mode": summary_mode,
            "configured_empty_surfaces": configured_empty,
            "truth_rule": (
                "Configured but empty means the mechanism exists but has no lived records yet."
            ),
        },
    }


def _tool_entries(runtime: Any) -> list[dict[str, Any]]:
    registry = getattr(runtime, "tools", None)
    if registry is None or not hasattr(registry, "list_tools"):
        return []
    try:
        entries = list(registry.list_tools())
    except Exception:
        return []
    return [
        {
            "name": str(getattr(entry, "name", "")),
            "risk_tier": str(getattr(getattr(entry, "risk_tier", None), "value", "")),
            "description": str(getattr(entry, "description", "")),
        }
        for entry in entries
        if str(getattr(entry, "name", "")).strip()
    ]


def _capability_entries(runtime: Any) -> list[dict[str, Any]]:
    registry = getattr(runtime, "capability_registry", None) or getattr(
        getattr(runtime, "ctx", None),
        "capability_registry",
        None,
    )
    if registry is None or not hasattr(registry, "list_capabilities"):
        return []
    try:
        entries = list(registry.list_capabilities())
    except Exception:
        return []
    return [
        {
            "capability_id": str(getattr(entry, "capability_id", "")),
            "status": str(getattr(getattr(entry, "status", None), "value", "")),
            "description": str(getattr(entry, "description", "")),
            "tool_names": list(getattr(entry, "tool_names", []) or []),
        }
        for entry in entries
        if str(getattr(entry, "capability_id", "")).strip()
    ]


def _availability_summary(runtime: Any, path: str) -> dict[str, Any]:
    target: Any = runtime
    for part in path.split("."):
        target = getattr(target, part, None)
        if target is None:
            return {"available": False, "counts_deferred": True}
    return {"available": True, "counts_deferred": True, "configured_empty": False}


async def _call_list(store: Any, method_name: str, *, limit: int) -> list[Any]:
    method = getattr(store, method_name, None)
    if not callable(method):
        return []
    try:
        return list(await method(limit=limit) or [])
    except Exception:
        return []
