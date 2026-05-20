"""Dual-context API routes for dashboard and manager surfaces."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _proposal_store(runtime: Any) -> Any:
    return getattr(runtime, "context_proposals", None) or getattr(
        getattr(runtime, "ctx", None),
        "context_proposal_store",
        None,
    )


def _proposal_payload(proposal: Any) -> Dict[str, Any]:
    status = _enum_value(getattr(proposal, "status", ""))
    authority = _enum_value(getattr(proposal, "authority", ""))
    source_lane = _enum_value(getattr(proposal, "source_lane", ""))
    confidence = float(getattr(proposal, "confidence", 0.0) or 0.0)
    return {
        "proposal_id": str(getattr(proposal, "proposal_id", "")),
        "created_at": _iso(getattr(proposal, "created_at", None)),
        "updated_at": _iso(getattr(proposal, "updated_at", None)),
        "source_lane": source_lane,
        "source_snapshot_id": str(getattr(proposal, "source_snapshot_id", "") or ""),
        "source_epoch": int(getattr(proposal, "source_epoch", 0) or 0),
        "proposal_kind": str(getattr(proposal, "proposal_kind", "") or ""),
        "status": status,
        "authority": authority,
        "confidence": confidence,
        "project_id": getattr(proposal, "project_id", None),
        "commitment_id": getattr(proposal, "commitment_id", None),
        "schedule_id": getattr(proposal, "schedule_id", None),
        "task_id": getattr(proposal, "task_id", None),
        "evidence_refs": list(getattr(proposal, "evidence_refs", []) or []),
        "validation": dict(getattr(proposal, "validation", {}) or {}),
        "content_preview": " ".join(str(getattr(proposal, "content", "") or "").split())[:240],
        "can_authorize_work": status == "accepted" and authority == "executive_committed",
    }


async def _recent_proposals(runtime: Any, *, limit: int = 200) -> List[Dict[str, Any]]:
    store = _proposal_store(runtime)
    if store is None:
        return []
    list_recent = getattr(store, "list_recent", None)
    if not callable(list_recent):
        return []
    try:
        return [_proposal_payload(item) for item in await list_recent(limit=limit)]
    except Exception:
        return []


async def _proposal_status_counts(runtime: Any) -> Dict[str, int]:
    store = _proposal_store(runtime)
    if store is None:
        return {}
    count_by_status = getattr(store, "count_by_status", None)
    if callable(count_by_status):
        try:
            return dict(await count_by_status())
        except Exception:
            return {}
    return {}


async def _truth_snapshot(runtime: Any) -> Dict[str, Any]:
    arbiter = getattr(runtime, "truth_arbiter", None) or getattr(
        getattr(runtime, "ctx", None),
        "truth_arbiter",
        None,
    )
    issue_snapshot = getattr(arbiter, "issue_snapshot", None)
    if not callable(issue_snapshot):
        return {"available": False}
    try:
        snapshot = await issue_snapshot(reason="api_dual_context_status")
    except Exception as exc:
        return {"available": True, "error": str(exc)}
    return {
        "available": True,
        "snapshot_id": snapshot.snapshot_id,
        "epoch": snapshot.epoch,
        "generated_at": snapshot.generated_at.isoformat(),
        "reason": snapshot.reason,
        "source_hash": snapshot.source_hash,
        "sources": list(snapshot.sources),
        "executive": snapshot.executive,
        "baa": snapshot.baa,
        "schedules": snapshot.schedules,
        "commitments": snapshot.commitments,
        "receipts": snapshot.receipts,
        "runtime_activity": snapshot.runtime_activity,
    }


def _packet_summary(packet: Any) -> Dict[str, Any]:
    sections = list(getattr(packet, "sections", []) or [])
    return {
        "lane": _enum_value(getattr(packet, "lane", "")),
        "authority_mode": str(getattr(packet, "authority_mode", "") or ""),
        "truth_snapshot_id": str(getattr(packet, "truth_snapshot_id", "") or ""),
        "truth_epoch": int(getattr(packet, "truth_epoch", 0) or 0),
        "token_budget": int(getattr(packet, "token_budget", 0) or 0),
        "can_write": bool(getattr(packet, "can_write", False)),
        "section_count": len(sections),
        "sections": [
            {
                "title": str(getattr(section, "title", "") or ""),
                "label": str(getattr(section, "label", "") or ""),
                "authority": _enum_value(getattr(section, "authority", "")),
                "refs": list(getattr(section, "refs", []) or []),
            }
            for section in sections
        ],
    }


async def _packet_previews(runtime: Any) -> Dict[str, Any]:
    builder = getattr(runtime, "context_packet_builder", None) or getattr(
        getattr(runtime, "ctx", None),
        "context_packet_builder",
        None,
    )
    if builder is None:
        return {"available": False}
    result: Dict[str, Any] = {"available": True}
    for lane, method_name in (
        ("executive", "build_executive_packet"),
        ("reflective", "build_reflective_packet"),
    ):
        method = getattr(builder, method_name, None)
        if not callable(method):
            continue
        try:
            packet = await method(user_input="Dashboard dual-context status inspection.")
            result[lane] = _packet_summary(packet)
        except Exception as exc:
            result[lane] = {"error": str(exc)}
    return result


def _proposal_stats(proposals: List[Dict[str, Any]], status_counts: Dict[str, int]) -> Dict[str, Any]:
    return {
        "count": sum(status_counts.values()) if status_counts else len(proposals),
        "recent_count": len(proposals),
        "status_counts": status_counts,
        "source_lane_counts": dict(Counter(item.get("source_lane") or "unknown" for item in proposals)),
        "authority_counts": dict(Counter(item.get("authority") or "unknown" for item in proposals)),
        "proposal_kind_counts": dict(Counter(item.get("proposal_kind") or "unknown" for item in proposals)),
        "authorizing_count": sum(1 for item in proposals if item.get("can_authorize_work")),
    }


def build_context_router(runtime: Any) -> APIRouter:
    """Build context routes wired to *runtime*."""

    router = APIRouter(prefix="/api/context", tags=["context"])

    @router.get("/dual")
    async def dual_context_status(limit: int = 40) -> Dict[str, Any]:
        bounded = max(1, min(int(limit), 200))
        proposals = await _recent_proposals(runtime, limit=max(bounded, 80))
        status_counts = await _proposal_status_counts(runtime)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "architecture": {
                "lanes": ["executive", "reflective"],
                "bridge": "truth_arbiter",
                "proposal_store_available": _proposal_store(runtime) is not None,
                "rule": (
                    "Reflective material may be remembered and proposed; only executive "
                    "truth, user requests, due schedules, or accepted proposals may authorize work."
                ),
            },
            "lanes": {
                "executive": {
                    "source_lane": "executive",
                    "authority_mode": "live_truth_and_committed_actions",
                    "can_write": True,
                    "material": [
                        "direct_user_request",
                        "live_observation",
                        "accepted_proposal",
                        "schedule_due",
                        "executive_committed",
                    ],
                },
                "reflective": {
                    "source_lane": "reflective",
                    "authority_mode": "proposals_only",
                    "can_write": False,
                    "material": [
                        "daydream",
                        "memory_association",
                        "self_note",
                        "risk_or_blocker",
                        "bad_idea_to_avoid",
                    ],
                },
            },
            "truth_snapshot": await _truth_snapshot(runtime),
            "context_packets": await _packet_previews(runtime),
            "proposals": {
                "stats": _proposal_stats(proposals, status_counts),
                "recent": proposals[:bounded],
            },
            "dashboard_contract": {
                "node_fields": [
                    "source_lane",
                    "source_lane_inferred",
                    "authority",
                    "context_material",
                    "can_authorize_work",
                    "source_snapshot_id",
                    "source_epoch",
                    "proposal_status",
                ],
                "memory_landscape_stats": [
                    "lane_distribution",
                    "authority_distribution",
                    "context_material_distribution",
                    "lane_source_distribution",
                    "can_authorize_work_count",
                ],
            },
        }

    return router
