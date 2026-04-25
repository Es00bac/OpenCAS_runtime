"""Operations monitoring snapshot helpers.

These helpers keep the operations router focused on HTTP wiring while the
SQLite-backed readiness, approval, and cost snapshots live in one reusable
module.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional


def window_bounds_ms(window_days: int) -> tuple[int, int]:
    clamped_days = max(1, min(365, int(window_days)))
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (clamped_days * 24 * 60 * 60 * 1000)
    return start_ms, end_ms


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 3)


def json_list(value: Any, key: Optional[str] = None) -> List[Any]:
    if isinstance(value, str) and value:
        try:
            value = json.loads(value)
        except Exception:
            return []
    if isinstance(value, dict) and key and key in value:
        nested = value.get(key)
        if isinstance(nested, list):
            return nested
        return []
    if isinstance(value, list):
        return value
    return []


async def build_memory_value_snapshot(runtime: Any) -> Dict[str, Any]:
    store = getattr(runtime, "memory", None) or getattr(getattr(runtime, "ctx", None), "memory", None)
    snapshot: Dict[str, Any] = {
        "available": store is not None,
        "evidence_level": "insufficient",
        "stats": {},
        "retrieval_usage": {
            "touched_episode_count": 0,
            "untouched_episode_count": 0,
            "total_episode_accesses": 0,
            "touched_memory_count": 0,
            "untouched_memory_count": 0,
            "total_memory_accesses": 0,
            "total_retrieval_accesses": 0,
            "touched_episode_ratio": 0.0,
            "touched_memory_ratio": 0.0,
        },
        "outcomes": {
            "outcome_instrumented_episode_count": 0,
            "total_success_uses": 0,
            "total_failed_uses": 0,
        },
        "top_episode_reuse": [],
        "top_memory_reuse": [],
        "notes": [],
    }
    if store is None:
        snapshot["notes"].append("Memory store is not available in the current runtime.")
        return snapshot

    if hasattr(store, "get_stats"):
        try:
            snapshot["stats"] = await store.get_stats()
        except Exception as exc:
            snapshot["notes"].append(f"Unable to load memory stats: {exc}")

    db = getattr(store, "_db", None)
    if db is None:
        snapshot["notes"].append("Memory store is running without a queryable SQLite connection, so value evidence is limited.")
        return snapshot

    episode_cursor = await db.execute(
        """
        SELECT
            COUNT(*) AS episode_count,
            SUM(CASE WHEN access_count > 0 THEN 1 ELSE 0 END) AS touched_episode_count,
            COALESCE(SUM(access_count), 0) AS total_episode_accesses,
            SUM(CASE WHEN used_successfully > 0 OR used_unsuccessfully > 0 THEN 1 ELSE 0 END) AS outcome_instrumented_episode_count,
            COALESCE(SUM(used_successfully), 0) AS total_success_uses,
            COALESCE(SUM(used_unsuccessfully), 0) AS total_failed_uses
        FROM episodes
        """
    )
    episode_row = await episode_cursor.fetchone()
    memory_cursor = await db.execute(
        """
        SELECT
            COUNT(*) AS memory_count,
            SUM(CASE WHEN access_count > 0 THEN 1 ELSE 0 END) AS touched_memory_count,
            COALESCE(SUM(access_count), 0) AS total_memory_accesses
        FROM memories
        """
    )
    memory_row = await memory_cursor.fetchone()
    top_episode_cursor = await db.execute(
        """
        SELECT
            episode_id,
            kind,
            session_id,
            content,
            salience,
            confidence_score,
            access_count,
            last_accessed,
            used_successfully,
            used_unsuccessfully
        FROM episodes
        WHERE access_count > 0 OR used_successfully > 0 OR used_unsuccessfully > 0
        ORDER BY access_count DESC, used_successfully DESC, used_unsuccessfully DESC, created_at DESC
        LIMIT 8
        """
    )
    top_episode_rows = await top_episode_cursor.fetchall()
    top_memory_cursor = await db.execute(
        """
        SELECT
            memory_id,
            content,
            salience,
            access_count,
            last_accessed,
            source_episode_ids,
            tags
        FROM memories
        WHERE access_count > 0
        ORDER BY access_count DESC, salience DESC, updated_at DESC
        LIMIT 8
        """
    )
    top_memory_rows = await top_memory_cursor.fetchall()

    episode_count = int(episode_row["episode_count"] or 0)
    touched_episode_count = int(episode_row["touched_episode_count"] or 0)
    total_episode_accesses = int(episode_row["total_episode_accesses"] or 0)
    outcome_instrumented_episode_count = int(episode_row["outcome_instrumented_episode_count"] or 0)
    total_success_uses = int(episode_row["total_success_uses"] or 0)
    total_failed_uses = int(episode_row["total_failed_uses"] or 0)

    memory_count = int(memory_row["memory_count"] or 0)
    touched_memory_count = int(memory_row["touched_memory_count"] or 0)
    total_memory_accesses = int(memory_row["total_memory_accesses"] or 0)
    total_retrieval_accesses = total_episode_accesses + total_memory_accesses

    snapshot["retrieval_usage"] = {
        "touched_episode_count": touched_episode_count,
        "untouched_episode_count": max(0, episode_count - touched_episode_count),
        "total_episode_accesses": total_episode_accesses,
        "touched_memory_count": touched_memory_count,
        "untouched_memory_count": max(0, memory_count - touched_memory_count),
        "total_memory_accesses": total_memory_accesses,
        "total_retrieval_accesses": total_retrieval_accesses,
        "touched_episode_ratio": ratio(touched_episode_count, episode_count),
        "touched_memory_ratio": ratio(touched_memory_count, memory_count),
    }
    snapshot["outcomes"] = {
        "outcome_instrumented_episode_count": outcome_instrumented_episode_count,
        "total_success_uses": total_success_uses,
        "total_failed_uses": total_failed_uses,
    }
    snapshot["top_episode_reuse"] = [
        {
            "episode_id": row["episode_id"],
            "kind": row["kind"],
            "session_id": row["session_id"],
            "content": row["content"],
            "salience": row["salience"],
            "confidence_score": row["confidence_score"],
            "access_count": row["access_count"],
            "last_accessed": row["last_accessed"],
            "used_successfully": row["used_successfully"],
            "used_unsuccessfully": row["used_unsuccessfully"],
        }
        for row in top_episode_rows
    ]
    snapshot["top_memory_reuse"] = [
        {
            "memory_id": row["memory_id"],
            "content": row["content"],
            "salience": row["salience"],
            "access_count": row["access_count"],
            "last_accessed": row["last_accessed"],
            "source_episode_ids": json_list(row["source_episode_ids"], key="source_episode_ids"),
            "tags": json_list(row["tags"], key="tags"),
        }
        for row in top_memory_rows
    ]

    if total_retrieval_accesses <= 0:
        snapshot["evidence_level"] = "insufficient"
        snapshot["notes"].append("No retrieved memories have been durably recorded as used yet.")
    elif outcome_instrumented_episode_count <= 0:
        snapshot["evidence_level"] = "partial"
        snapshot["notes"].append("Retrieval access is now visible, but success and failure attribution still needs more downstream outcome coverage.")
    else:
        snapshot["evidence_level"] = "grounded"
        snapshot["notes"].append("The runtime has both retrieval-access evidence and outcome-tagged episode reuse to inspect.")

    if total_success_uses <= 0 and total_failed_uses <= 0:
        snapshot["notes"].append("No episode has been marked successful or unsuccessful yet, so value claims remain provisional.")
    return snapshot


async def build_approval_audit_snapshot(runtime: Any, *, window_days: int, limit: int) -> Dict[str, Any]:
    ledger = getattr(getattr(runtime, "ctx", None), "ledger", None)
    snapshot: Dict[str, Any] = {
        "available": ledger is not None,
        "window_days": window_days,
        "total_decisions": 0,
        "level_counts": {},
        "tier_counts": {},
        "breakdown": [],
        "recent_entries": [],
        "notes": [],
    }
    if ledger is None:
        snapshot["notes"].append("Approval ledger is not available in the current runtime.")
        return snapshot

    stats = await ledger.query_stats(window_days=window_days)
    breakdown = stats.get("breakdown", []) or []
    level_counts: Dict[str, int] = {}
    tier_counts: Dict[str, int] = {}
    total_decisions = 0
    for item in breakdown:
        level = str(item.get("level", "unknown") or "unknown")
        tier = str(item.get("tier", "unknown") or "unknown")
        count = int(item.get("count", 0) or 0)
        total_decisions += count
        level_counts[level] = level_counts.get(level, 0) + count
        tier_counts[tier] = tier_counts.get(tier, 0) + count

    recent_entries = []
    store = getattr(ledger, "store", None)
    if store is not None and hasattr(store, "list_recent"):
        for entry in await store.list_recent(limit=limit):
            recent_entries.append(
                {
                    "entry_id": str(entry.entry_id),
                    "decision_id": str(entry.decision_id),
                    "action_id": str(entry.action_id),
                    "created_at": entry.created_at.isoformat(),
                    "level": entry.level,
                    "tier": entry.tier.value if hasattr(entry.tier, "value") else str(entry.tier),
                    "score": entry.score,
                    "tool_name": entry.tool_name,
                    "reasoning": entry.reasoning,
                    "somatic_state": entry.somatic_state,
                }
            )

    snapshot.update(
        {
            "total_decisions": total_decisions,
            "level_counts": level_counts,
            "tier_counts": tier_counts,
            "breakdown": breakdown,
            "recent_entries": recent_entries,
        }
    )
    if total_decisions <= 0:
        snapshot["notes"].append("No approval decisions have been recorded in the current window.")
    if not recent_entries:
        snapshot["notes"].append("No recent approval-ledger entries are available for drill-down.")
    return snapshot


async def build_cost_snapshot(runtime: Any, *, window_days: int, bucket_hours: int) -> Dict[str, Any]:
    ctx = getattr(runtime, "ctx", None)
    telemetry = getattr(ctx, "token_telemetry", None)
    snapshot: Dict[str, Any] = {
        "available": telemetry is not None,
        "window_days": window_days,
        "bucket_hours": bucket_hours,
        "summary": {},
        "session_summary": {},
        "daily_rollup": [],
        "time_series": [],
        "recent_receipts": {"count": 0, "success_count": 0, "failure_count": 0, "success_rate": 0.0},
        "notes": [],
    }
    if telemetry is None:
        snapshot["notes"].append("Token telemetry is not available in the current runtime.")
        return snapshot

    start_ms, end_ms = window_bounds_ms(window_days)
    bucket_ms = max(1, min(24, int(bucket_hours))) * 60 * 60 * 1000
    summary = telemetry.get_summary(start_ms, end_ms).to_dict()
    session_id = getattr(getattr(ctx, "config", None), "session_id", None)
    session_summary = telemetry.get_session_summary(session_id).to_dict() if session_id else {}
    daily_rollup = [item.to_dict() for item in telemetry.get_daily_rollup(start_ms, end_ms)]
    time_series = [item.to_dict() for item in telemetry.get_time_series(start_ms, end_ms, bucket_ms=bucket_ms)]

    recent_receipts_summary = {"count": 0, "success_count": 0, "failure_count": 0, "success_rate": 0.0}
    receipt_store = getattr(ctx, "receipt_store", None)
    if receipt_store is not None:
        recent_receipts = await receipt_store.list_recent(limit=40)
        success_count = sum(1 for item in recent_receipts if bool(getattr(item, "success", False)))
        receipt_count = len(recent_receipts)
        recent_receipts_summary = {
            "count": receipt_count,
            "success_count": success_count,
            "failure_count": max(0, receipt_count - success_count),
            "success_rate": ratio(success_count, receipt_count),
        }

    snapshot.update(
        {
            "summary": summary,
            "session_summary": session_summary,
            "daily_rollup": daily_rollup,
            "time_series": time_series,
            "recent_receipts": recent_receipts_summary,
        }
    )
    if int(summary.get("totalCalls", 0) or 0) <= 0:
        snapshot["notes"].append("No token usage has been recorded in the selected window.")
    if recent_receipts_summary["count"] <= 0:
        snapshot["notes"].append("No recent execution receipts are available to compare against token usage.")
    return snapshot


async def build_hardening_snapshot(runtime: Any, *, window_days: int, bucket_hours: int, decision_limit: int) -> Dict[str, Any]:
    memory_value = await build_memory_value_snapshot(runtime)
    approval_audit = await build_approval_audit_snapshot(runtime, window_days=window_days, limit=decision_limit)
    costs = await build_cost_snapshot(runtime, window_days=window_days, bucket_hours=bucket_hours)

    observable_signals = sum(
        [
            1 if memory_value["retrieval_usage"]["total_retrieval_accesses"] > 0 else 0,
            1 if approval_audit["total_decisions"] > 0 else 0,
            1 if int(costs["summary"].get("totalCalls", 0) or 0) > 0 else 0,
        ]
    )
    if memory_value.get("evidence_level") == "grounded" and observable_signals >= 3:
        overall_state = "grounded"
    elif observable_signals > 0:
        overall_state = "observable"
    else:
        overall_state = "emerging"

    return {
        "overall_state": overall_state,
        "window_days": window_days,
        "memory_value": {
            "evidence_level": memory_value.get("evidence_level"),
            "total_retrieval_accesses": memory_value["retrieval_usage"]["total_retrieval_accesses"],
            "outcome_instrumented_episode_count": memory_value["outcomes"]["outcome_instrumented_episode_count"],
        },
        "approval_audit": {
            "total_decisions": approval_audit.get("total_decisions", 0),
            "level_counts": approval_audit.get("level_counts", {}),
        },
        "costs": {
            "total_calls": int(costs["summary"].get("totalCalls", 0) or 0),
            "total_tokens": int(costs["summary"].get("totalTokens", 0) or 0),
            "cost_estimate": float(costs["summary"].get("costEstimate", 0.0) or 0.0),
            "recent_receipt_success_rate": costs["recent_receipts"]["success_rate"],
        },
        "notes": [
            *memory_value.get("notes", []),
            *approval_audit.get("notes", []),
            *costs.get("notes", []),
        ][:6],
    }
