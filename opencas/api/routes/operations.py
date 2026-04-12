"""Operations API routes for session inspection, receipts, work management, and qualification state."""

from __future__ import annotations

import time
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from opencas.autonomy.commitment import CommitmentStatus
from opencas.autonomy.models import WorkStage

REPO_ROOT = Path(__file__).resolve().parents[3]
QUALIFICATION_SUMMARY_PATH = REPO_ROOT / "docs" / "qualification" / "live_validation_summary.json"
QUALIFICATION_REMEDIATION_PATH = REPO_ROOT / "docs" / "qualification" / "qualification_remediation_rollup.json"
VALIDATION_RUNS_DIR = REPO_ROOT / ".opencas_live_test_state"
QUALIFICATION_RERUN_HISTORY_PATH = VALIDATION_RUNS_DIR / "qualification_rerun_history.jsonl"
DEFAULT_OPERATOR_ACTIONS_PATH = VALIDATION_RUNS_DIR / "operator_action_history.jsonl"


class SessionEntry(BaseModel):
    session_id: str
    pid: Optional[int] = None
    scope_key: str
    command: str
    cwd: Optional[str] = None
    running: bool
    returncode: Optional[int] = None
    rows: Optional[int] = None
    cols: Optional[int] = None
    created_at: Optional[float] = None
    last_observed_at: Optional[float] = None
    last_screen_state: Dict[str, Any] = Field(default_factory=dict)
    last_cleaned_output: Optional[str] = None


class SessionScopeEntry(BaseModel):
    scope_key: str
    process_count: int = 0
    pty_count: int = 0
    browser_count: int = 0


def _build_session_entry(entry: Dict[str, Any]) -> SessionEntry:
    return SessionEntry(
        session_id=entry["session_id"],
        pid=entry.get("pid"),
        scope_key=entry.get("scope_key", ""),
        command=entry.get("command", ""),
        cwd=entry.get("cwd"),
        running=entry.get("running", False),
        returncode=entry.get("returncode"),
        rows=entry.get("rows"),
        cols=entry.get("cols"),
        created_at=entry.get("created_at"),
        last_observed_at=entry.get("last_observed_at"),
        last_screen_state=entry.get("last_screen_state", {}) or {},
        last_cleaned_output=entry.get("last_cleaned_output"),
    )


class SessionListResponse(BaseModel):
    processes: List[Dict[str, Any]] = Field(default_factory=list)
    pty: List[SessionEntry] = Field(default_factory=list)
    browser: List[Dict[str, Any]] = Field(default_factory=list)
    scopes: List[SessionScopeEntry] = Field(default_factory=list)
    current_scope: Optional[str] = None
    total_processes: int = 0
    total_pty: int = 0
    total_browser: int = 0


class PtyInputRequest(BaseModel):
    input: str
    observe: bool = True
    idle_seconds: float = 0.25
    max_wait_seconds: float = 1.5


class BrowserNavigateRequest(BaseModel):
    url: str
    wait_until: str = "load"
    timeout_ms: int = 30000
    refresh: bool = True


class BrowserClickRequest(BaseModel):
    selector: str
    timeout_ms: int = 5000
    refresh: bool = True


class BrowserTypeRequest(BaseModel):
    selector: str
    text: str
    clear: bool = True
    timeout_ms: int = 5000
    refresh: bool = True


class BrowserPressRequest(BaseModel):
    key: str
    refresh: bool = True


class BrowserWaitRequest(BaseModel):
    timeout_ms: int = 5000
    load_state: str = "load"
    selector: Optional[str] = None
    refresh: bool = True


class BrowserCaptureRequest(BaseModel):
    full_page: bool = False


class ReceiptEntry(BaseModel):
    receipt_id: str
    task_id: str
    status: str
    tool_name: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: Optional[int] = None


class ReceiptListResponse(BaseModel):
    count: int
    items: List[ReceiptEntry]


class TaskEntry(BaseModel):
    task_id: str
    title: str
    objective: str
    status: str
    stage: str
    source: Optional[str] = None
    project_id: Optional[str] = None
    commitment_id: Optional[str] = None
    updated_at: Optional[str] = None
    duplicate_objective_count: int = 1


class TaskListResponse(BaseModel):
    counts: Dict[str, int]
    items: List[TaskEntry]


class WorkItemEntry(BaseModel):
    work_id: str
    title: str
    stage: str
    project_id: Optional[str] = None
    blocked_by: Optional[List[str]] = None


class WorkListResponse(BaseModel):
    counts: Dict[str, int]
    items: List[WorkItemEntry]


class WorkUpdateRequest(BaseModel):
    stage: Optional[WorkStage] = None
    content: Optional[str] = None
    blocked_by: Optional[List[str]] = None


class CommitmentEntry(BaseModel):
    commitment_id: str
    content: str
    status: str
    priority: float
    tags: List[str] = Field(default_factory=list)
    deadline: Optional[str] = None


class CommitmentListResponse(BaseModel):
    count: int
    items: List[CommitmentEntry]


class CommitmentUpdateRequest(BaseModel):
    status: Optional[CommitmentStatus] = None
    content: Optional[str] = None
    priority: Optional[float] = None
    tags: Optional[List[str]] = None


class PlanSummary(BaseModel):
    plan_id: str
    status: str
    content_preview: str
    project_id: Optional[str] = None
    updated_at: Optional[str] = None


class PlanListResponse(BaseModel):
    count: int
    items: List[PlanSummary]


class PlanUpdateRequest(BaseModel):
    status: Optional[Literal["draft", "active", "completed", "abandoned"]] = None
    content: Optional[str] = None


class QualificationSummaryResponse(BaseModel):
    found: bool
    path: Optional[str] = None
    updated_at: Optional[str] = None
    summary: Dict[str, Any] = Field(default_factory=dict)
    weakest_checks: List[Dict[str, Any]] = Field(default_factory=list)
    recommended_reruns: List[Dict[str, Any]] = Field(default_factory=list)
    active_reruns: List[Dict[str, Any]] = Field(default_factory=list)
    recent_runs: List[Dict[str, Any]] = Field(default_factory=list)
    recent_rerun_history: List[Dict[str, Any]] = Field(default_factory=list)
    remediation_rollup: Dict[str, Any] = Field(default_factory=dict)


class QualificationLabelDetailResponse(BaseModel):
    found: bool
    label: Optional[str] = None
    detail: Dict[str, Any] = Field(default_factory=dict)


class QualificationRerunDetailResponse(BaseModel):
    found: bool
    request_id: Optional[str] = None
    detail: Dict[str, Any] = Field(default_factory=dict)


class ValidationRunEntry(BaseModel):
    run_id: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    model: Optional[str] = None
    direct_successes: int = 0
    direct_total: int = 0
    agent_successes: int = 0
    agent_total: int = 0
    duration_seconds: float = 0.0
    report_path: Optional[str] = None
    report_markdown_path: Optional[str] = None
    aborted: bool = False
    failed_labels: List[str] = Field(default_factory=list)


class ValidationRunListResponse(BaseModel):
    count: int
    label_filter: Optional[str] = None
    items: List[ValidationRunEntry] = Field(default_factory=list)


class ValidationRunDetailResponse(BaseModel):
    found: bool
    run: Dict[str, Any] = Field(default_factory=dict)


class QualificationRerunRequest(BaseModel):
    label: str
    iterations: int = 2
    include_direct_checks: bool = False
    prompt_timeout_seconds: float = 180.0
    run_timeout_seconds: float = 420.0
    source_label: Optional[str] = None
    source_note: Optional[str] = None


class ProcessDetailResponse(BaseModel):
    found: bool
    process: Dict[str, Any] = Field(default_factory=dict)


def _human_title(text: Optional[str], fallback: str = "Untitled") -> str:
    raw = str(text or "").strip()
    if not raw:
        return fallback
    first_line = raw.splitlines()[0].strip()
    compact = " ".join(first_line.split())
    if len(compact) <= 88:
        return compact
    return compact[:85].rstrip() + "..."


def _task_ui_status(stage: str, status: str) -> str:
    stage_key = str(stage or "").strip().lower()
    status_key = str(status or "").strip().lower()
    if stage_key in {"done"} or status_key in {"completed", "success"}:
        return "completed"
    if stage_key in {"failed"} or status_key in {"failed", "error"}:
        return "failed"
    if stage_key == "needs_approval":
        return "needs approval"
    if stage_key == "needs_clarification":
        return "needs clarification"
    if stage_key in {"queued", "planning", "executing", "verifying", "recovering"}:
        return stage_key
    if status_key:
        return status_key.replace("_", " ")
    return "unknown"


def _window_bounds_ms(window_days: int) -> tuple[int, int]:
    clamped_days = max(1, min(365, int(window_days)))
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (clamped_days * 24 * 60 * 60 * 1000)
    return start_ms, end_ms


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 3)


def _json_list(value: Any, key: Optional[str] = None) -> List[Any]:
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


async def _build_memory_value_snapshot(runtime: Any) -> Dict[str, Any]:
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
        "touched_episode_ratio": _ratio(touched_episode_count, episode_count),
        "touched_memory_ratio": _ratio(touched_memory_count, memory_count),
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
            "source_episode_ids": _json_list(row["source_episode_ids"], key="source_episode_ids"),
            "tags": _json_list(row["tags"], key="tags"),
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


async def _build_approval_audit_snapshot(runtime: Any, *, window_days: int, limit: int) -> Dict[str, Any]:
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


async def _build_cost_snapshot(runtime: Any, *, window_days: int, bucket_hours: int) -> Dict[str, Any]:
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

    start_ms, end_ms = _window_bounds_ms(window_days)
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
            "success_rate": _ratio(success_count, receipt_count),
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


async def _build_hardening_snapshot(runtime: Any, *, window_days: int, bucket_hours: int, decision_limit: int) -> Dict[str, Any]:
    memory_value = await _build_memory_value_snapshot(runtime)
    approval_audit = await _build_approval_audit_snapshot(runtime, window_days=window_days, limit=decision_limit)
    costs = await _build_cost_snapshot(runtime, window_days=window_days, bucket_hours=bucket_hours)

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


def _resolve_operator_actions_path(runtime: Any) -> Path:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    state_dir = getattr(config, "state_dir", None)
    if isinstance(state_dir, (str, Path)):
        return Path(state_dir) / "operator_action_history.jsonl"
    return DEFAULT_OPERATOR_ACTIONS_PATH


def _truncate_text(value: Optional[str], limit: int = 160) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _append_operator_action(runtime: Any, entry: Dict[str, Any]) -> Dict[str, Any]:
    path = _resolve_operator_actions_path(runtime)
    payload = {
        "event_id": uuid4().hex,
        "timestamp": time.time(),
        **entry,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return payload


def _load_recent_operator_actions(
    runtime: Any,
    *,
    target_kind: str,
    target_id: str,
    scope_key: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    path = _resolve_operator_actions_path(runtime)
    if not path.exists():
        return []

    items: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if str(payload.get("target_kind", "") or "") != target_kind:
            continue
        if str(payload.get("target_id", "") or "") != target_id:
            continue
        if scope_key is not None and str(payload.get("scope_key", "") or "default") != str(scope_key):
            continue
        items.append(payload)
        if len(items) >= limit:
            break
    return items


def _build_qualification_recommendations(checks: Dict[str, Any]) -> List[Dict[str, Any]]:
    recommendations: List[Dict[str, Any]] = []
    for label, payload in checks.items():
        failures = int(payload.get("failures", 0) or 0)
        timeouts = int(payload.get("timeouts", 0) or 0)
        if failures <= 0 and timeouts <= 0:
            continue
        success_rate = payload.get("success_rate")
        reasons: List[str] = []
        if failures:
            reasons.append(f"{failures} failure(s)")
        if timeouts:
            reasons.append(f"{timeouts} timeout(s)")
        recent_failures = payload.get("recent_failures", []) or []
        note = "Rerun this check in a bounded cycle and inspect the most recent failure before changing behavior."
        if "kilocode" in label:
            note = "Treat this as a PTY/TUI workflow issue first: verify readiness, submission, and timeout behavior before changing models."
        elif "integrated" in label:
            note = "Treat this as a coordination-budget issue first: inspect tool-loop churn and verify the workflow stayed bounded."
        elif "writing" in label:
            note = "Treat this as a workflow-selection issue first: verify the agent chose the higher-level writing tools instead of low-level choreography."
        recommendations.append({
            "label": label,
            "success_rate": success_rate,
            "failures": failures,
            "timeouts": timeouts,
            "reason": ", ".join(reasons),
            "recent_failures": recent_failures,
            "note": note,
            "comparison": _label_run_comparison(label),
            "rate_window": _label_rate_window(label),
            "command": [
                "python",
                "scripts/run_qualification_cycle.py",
                "--agent-check-label",
                label,
                "--iterations",
                "2",
            ],
        })
    recommendations.sort(
        key=lambda item: (
            item["success_rate"] if item["success_rate"] is not None else 1.0,
            -item["failures"],
            -item["timeouts"],
            item["label"],
        )
    )
    return recommendations[:5]


def _annotate_recommendation_runtime_state(
    recommendations: List[Dict[str, Any]],
    active_reruns: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    request_history, completion_history = _load_latest_rerun_history_by_label()
    active_by_label: Dict[str, Dict[str, Any]] = {}
    for entry in active_reruns:
        if not entry.get("running", False):
            continue
        metadata = entry.get("metadata", {}) or {}
        label = str(metadata.get("source_label", "") or "")
        if not label or label in active_by_label:
            continue
        active_by_label[label] = {
            "process_id": entry.get("process_id"),
            "requested_at": metadata.get("requested_at"),
            "note": metadata.get("source_note"),
        }

    for item in recommendations:
        label = str(item.get("label", "") or "")
        active = active_by_label.get(label)
        comparison = item.get("comparison") or {}
        latest = comparison.get("latest") or None
        item["active_rerun"] = active
        item["last_completed_run"] = latest
        item["last_request"] = request_history.get(label)
        item["last_completion_event"] = completion_history.get(label)
    return recommendations


def _load_qualification_summary() -> QualificationSummaryResponse:
    path = QUALIFICATION_SUMMARY_PATH
    if not path.exists():
        return QualificationSummaryResponse(found=False, path=str(path))

    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return QualificationSummaryResponse(
            found=False,
            path=str(path),
            summary={"error": str(exc)},
        )

    checks = summary.get("agent_checks", {}) or {}
    weakest = sorted(
        (
            {
                "label": label,
                "success_rate": payload.get("success_rate"),
                "failures": payload.get("failures", 0),
                "timeouts": payload.get("timeouts", 0),
            }
            for label, payload in checks.items()
        ),
        key=lambda item: (
            item["success_rate"] if item["success_rate"] is not None else 1.0,
            -item["failures"],
            -item["timeouts"],
            item["label"],
        ),
    )[:5]
    return QualificationSummaryResponse(
        found=True,
        path=str(path),
        updated_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        summary=summary,
        weakest_checks=weakest,
        recommended_reruns=_build_qualification_recommendations(checks),
    )


def _load_qualification_remediation() -> Dict[str, Any]:
    path = QUALIFICATION_REMEDIATION_PATH
    if not path.exists():
        return {"found": False, "path": str(path), "items": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"found": False, "path": str(path), "error": str(exc), "items": []}
    payload["found"] = True
    payload["path"] = str(path)
    return payload


def _direct_check_success(payload: Dict[str, Any]) -> bool:
    if "success" in payload:
        return bool(payload.get("success"))
    return bool(payload.get("available"))


def _agent_check_success(item: Dict[str, Any]) -> bool:
    if "material_success" in item:
        return bool(item.get("material_success"))
    if item.get("timed_out", False):
        return False
    if "expected_file" in item:
        return bool(item.get("expected_file_exists", False))
    return True


def _label_run_comparison(label: str) -> Optional[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    for path in sorted(VALIDATION_RUNS_DIR.glob("*/live_debug_validation_report.json"), reverse=True):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_id = str(report.get("run_id", path.parent.name))
        finished_at = report.get("finished_at") or report.get("started_at")
        for item in report.get("agent_checks", []) or []:
            if str(item.get("label", "")) != label:
                continue
            matched.append({
                "run_id": run_id,
                "finished_at": finished_at,
                "success": _agent_check_success(item),
                "outcome": item.get("outcome"),
            })
            break
        if len(matched) >= 2:
            break

    if not matched:
        return None

    latest = matched[0]
    previous = matched[1] if len(matched) > 1 else None
    if previous is None:
        trend = "first_observed"
    elif latest["success"] and not previous["success"]:
        trend = "improved"
    elif not latest["success"] and previous["success"]:
        trend = "regressed"
    elif latest["success"]:
        trend = "stable_pass"
    else:
        trend = "stable_fail"

    return {"trend": trend, "latest": latest, "previous": previous}


def _label_rate_window(label: str, window_size: int = 3) -> Optional[Dict[str, Any]]:
    matched: List[bool] = []
    for path in sorted(VALIDATION_RUNS_DIR.glob("*/live_debug_validation_report.json"), reverse=True):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in report.get("agent_checks", []) or []:
            if str(item.get("label", "")) != label:
                continue
            matched.append(_agent_check_success(item))
            break

    if not matched:
        return None

    recent = matched[:window_size]
    previous = matched[window_size : window_size * 2]
    recent_rate = round(sum(1 for item in recent if item) / len(recent), 3) if recent else None
    previous_rate = round(sum(1 for item in previous if item) / len(previous), 3) if previous else None
    delta = None
    if recent_rate is not None and previous_rate is not None:
        delta = round(recent_rate - previous_rate, 3)

    return {
        "window_size": window_size,
        "recent_runs": len(recent),
        "previous_runs": len(previous),
        "recent_success_rate": recent_rate,
        "previous_success_rate": previous_rate,
        "delta_success_rate": delta,
    }


def _duration_seconds(report: Dict[str, Any]) -> float:
    started = report.get("started_at")
    finished = report.get("finished_at")
    if not started or not finished:
        return 0.0
    try:
        started_at = datetime.fromisoformat(str(started))
        finished_at = datetime.fromisoformat(str(finished))
    except ValueError:
        return 0.0
    return max(0.0, (finished_at - started_at).total_seconds())


def _append_qualification_rerun_history(entry: Dict[str, Any]) -> None:
    path = QUALIFICATION_RERUN_HISTORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _load_latest_rerun_history_by_label(
    limit: int = 200,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    path = QUALIFICATION_RERUN_HISTORY_PATH
    if not path.exists():
        return {}, {}

    latest_requests: Dict[str, Dict[str, Any]] = {}
    latest_completions: Dict[str, Dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {}, {}

    for line in reversed(lines[-limit:]):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        event = str(payload.get("event", "requested") or "requested")
        if event == "completed":
            labels = [str(item) for item in payload.get("labels", []) if str(item)]
            for label in labels:
                if label not in latest_completions:
                    latest_completions[label] = payload
            continue

        label = str(payload.get("label", "") or "")
        if not label or label in latest_requests:
            continue
        latest_requests[label] = payload
    return latest_requests, latest_completions


def _load_recent_rerun_history(limit: int = 10) -> List[Dict[str, Any]]:
    path = QUALIFICATION_RERUN_HISTORY_PATH
    if not path.exists():
        return []

    items: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        event = str(payload.get("event", "requested") or "requested")
        timestamp = payload.get("completed_at") if event == "completed" else payload.get("requested_at")
        labels = payload.get("labels")
        if labels is None:
            label = payload.get("label")
            labels = [label] if label else []
        labels = [str(item) for item in labels if str(item)]
        comparison = None
        rate_window = None
        if event == "completed" and len(labels) == 1:
            comparison = _label_run_comparison(labels[0])
            rate_window = _label_rate_window(labels[0])
        items.append({
            **payload,
            "event": event,
            "event_time": timestamp,
            "labels": labels,
            "comparison": comparison,
            "rate_window": rate_window if event == "completed" and len(labels) == 1 else None,
        })
        if len(items) >= limit:
            break
    return items


def _load_all_rerun_history(limit: int = 400) -> List[Dict[str, Any]]:
    path = QUALIFICATION_RERUN_HISTORY_PATH
    if not path.exists():
        return []

    items: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    for line in reversed(lines[-limit:]):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        event = str(payload.get("event", "requested") or "requested")
        labels = payload.get("labels")
        if labels is None:
            label = payload.get("label")
            labels = [label] if label else []
        items.append({
            **payload,
            "event": event,
            "labels": [str(item) for item in labels if str(item)],
        })
    return items


def _find_rerun_history_by_request_id(request_id: Optional[str]) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not request_id:
        return None, None
    request_entry: Optional[Dict[str, Any]] = None
    completion_entry: Optional[Dict[str, Any]] = None
    for item in _load_all_rerun_history():
        if str(item.get("request_id", "") or "") != str(request_id):
            continue
        if item.get("event") == "completed" and completion_entry is None:
            completion_entry = item
        elif item.get("event") == "requested" and request_entry is None:
            request_entry = item
        if request_entry is not None and completion_entry is not None:
            break
    return request_entry, completion_entry


def _find_rerun_history_for_run(run_id: str, label: Optional[str] = None) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    request_entry: Optional[Dict[str, Any]] = None
    completion_entry: Optional[Dict[str, Any]] = None
    for item in _load_all_rerun_history():
        if item.get("event") != "completed":
            continue
        generated_ids = [str(value) for value in item.get("generated_run_ids", []) if str(value)]
        latest_run_id = str(item.get("latest_run_id", "") or "")
        labels = [str(value) for value in item.get("labels", []) if str(value)]
        if run_id not in generated_ids and run_id != latest_run_id:
            continue
        if label and labels and label not in labels:
            continue
        completion_entry = item
        request_entry, _ = _find_rerun_history_by_request_id(str(item.get("request_id", "") or ""))
        break
    return request_entry, completion_entry


def _load_validation_runs(limit: int = 10, label: Optional[str] = None) -> ValidationRunListResponse:
    items: List[ValidationRunEntry] = []
    for path in sorted(VALIDATION_RUNS_DIR.glob("*/live_debug_validation_report.json"), reverse=True):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        direct_checks = report.get("direct_checks", {}) or {}
        agent_checks = report.get("agent_checks", []) or []
        if label:
            if not any(str(item.get("label", "")) == label for item in agent_checks):
                continue
        direct_successes = sum(1 for payload in direct_checks.values() if _direct_check_success(payload))
        agent_successes = sum(1 for item in agent_checks if _agent_check_success(item))
        failed_labels = [str(item.get("label", "")) for item in agent_checks if not _agent_check_success(item)]
        items.append(ValidationRunEntry(
            run_id=str(report.get("run_id", path.parent.name)),
            started_at=report.get("started_at"),
            finished_at=report.get("finished_at"),
            model=report.get("model"),
            direct_successes=direct_successes,
            direct_total=len(direct_checks),
            agent_successes=agent_successes,
            agent_total=len(agent_checks),
            duration_seconds=_duration_seconds(report),
            report_path=str(path),
            report_markdown_path=str(path.with_name("live_debug_validation_report.md")),
            aborted=bool(report.get("aborted", False)),
            failed_labels=failed_labels,
        ))
        if len(items) >= limit:
            break
    return ValidationRunListResponse(count=len(items), label_filter=label, items=items)


def _load_validation_run_detail(run_id: str, label: Optional[str] = None) -> ValidationRunDetailResponse:
    path = VALIDATION_RUNS_DIR / run_id / "live_debug_validation_report.json"
    if not path.exists():
        return ValidationRunDetailResponse(found=False)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ValidationRunDetailResponse(found=False, run={"error": str(exc)})

    direct_checks = report.get("direct_checks", {}) or {}
    agent_checks = report.get("agent_checks", []) or []
    failed_agent_checks = [
        item for item in agent_checks if not _agent_check_success(item)
    ]
    matching_agent_checks = [
        item for item in agent_checks if not label or str(item.get("label", "")) == label
    ]
    rerun_request, rerun_completion = _find_rerun_history_for_run(run_id, label=label)
    return ValidationRunDetailResponse(
        found=True,
        run={
            "run_id": str(report.get("run_id", run_id)),
            "started_at": report.get("started_at"),
            "finished_at": report.get("finished_at"),
            "model": report.get("model"),
            "embedding_model": report.get("embedding_model"),
            "state_dir": report.get("state_dir"),
            "workspace_root": report.get("workspace_root"),
            "aborted": bool(report.get("aborted", False)),
            "abort_reason": report.get("abort_reason"),
            "focus_label": label,
            "direct_checks": direct_checks,
            "agent_checks": agent_checks,
            "matching_agent_checks": matching_agent_checks,
            "failed_agent_checks": failed_agent_checks,
            "rerun_request": rerun_request,
            "rerun_completion": rerun_completion,
            "report_path": str(path),
            "report_markdown_path": str(path.with_name("live_debug_validation_report.md")),
            "duration_seconds": _duration_seconds(report),
        },
    )


def _load_qualification_label_detail(label: str) -> QualificationLabelDetailResponse:
    summary = _load_qualification_summary()
    if not summary.found:
        return QualificationLabelDetailResponse(found=False, label=label, detail={"summary_missing": True, "path": summary.path})

    agent_checks = summary.summary.get("agent_checks", {}) or {}
    stats = agent_checks.get(label)
    if not stats:
        return QualificationLabelDetailResponse(found=False, label=label)

    recommendations = [item for item in summary.recommended_reruns if str(item.get("label", "")) == label]
    recent_runs = [item.model_dump(mode="json") for item in _load_validation_runs(limit=10, label=label).items]
    recent_rerun_history = [item for item in _load_recent_rerun_history(limit=20) if label in (item.get("labels") or [])]
    active_reruns = [item for item in _load_recent_rerun_history(limit=20) if item.get("event") == "requested" and label in (item.get("labels") or [])]
    return QualificationLabelDetailResponse(
        found=True,
        label=label,
        detail={
            "stats": stats,
            "comparison": _label_run_comparison(label),
            "rate_window": _label_rate_window(label),
            "recommendation": recommendations[0] if recommendations else None,
            "recent_runs": recent_runs,
            "recent_rerun_history": recent_rerun_history,
            "active_rerun_requests": active_reruns[:5],
        },
    )


def _load_qualification_rerun_detail(request_id: str) -> QualificationRerunDetailResponse:
    request_entry, completion_entry = _find_rerun_history_by_request_id(request_id)
    if request_entry is None and completion_entry is None:
        return QualificationRerunDetailResponse(found=False, request_id=request_id)

    labels: List[str] = []
    if completion_entry:
        labels = [str(item) for item in completion_entry.get("labels", []) if str(item)]
    elif request_entry:
        label = str(request_entry.get("label", "") or "")
        labels = [label] if label else []

    latest_run_id = None
    latest_run_detail: Optional[Dict[str, Any]] = None
    generated_runs: List[Dict[str, Any]] = []
    run_index: Dict[str, Dict[str, Any]] = {}
    generated_run_details: Dict[str, Dict[str, Any]] = {}
    if completion_entry:
        latest_run_id = str(completion_entry.get("latest_run_id", "") or "") or None
        run_index = {
            item.run_id: item.model_dump(mode="json")
            for item in _load_validation_runs(limit=200).items
        }
        for run_id in [str(item) for item in completion_entry.get("generated_run_ids", []) if str(item)]:
            matched = run_index.get(run_id)
            if matched is not None:
                generated_runs.append(matched)
            run_detail = _load_validation_run_detail(run_id)
            if not run_detail.found:
                continue
            run = run_detail.run
            generated_run_details[run_id] = run
            if matched is None:
                direct_checks = run.get("direct_checks", {}) or {}
                agent_checks = run.get("agent_checks", []) or []
                generated_runs.append({
                    "run_id": run.get("run_id"),
                    "started_at": run.get("started_at"),
                    "finished_at": run.get("finished_at"),
                    "model": run.get("model"),
                    "direct_successes": sum(1 for payload in direct_checks.values() if _direct_check_success(payload)),
                    "direct_total": len(direct_checks),
                    "agent_successes": sum(1 for item in agent_checks if _agent_check_success(item)),
                    "agent_total": len(agent_checks),
                    "duration_seconds": run.get("duration_seconds", 0.0),
                    "report_path": run.get("report_path"),
                    "report_markdown_path": run.get("report_markdown_path"),
                    "aborted": bool(run.get("aborted", False)),
                    "failed_labels": [str(item.get("label", "")) for item in agent_checks if not _agent_check_success(item)],
                })
    if latest_run_id:
        label_hint = labels[0] if len(labels) == 1 else None
        run_detail = _load_validation_run_detail(latest_run_id, label=label_hint)
        latest_run_detail = run_detail.run if run_detail.found else None

    label_outcomes: List[Dict[str, Any]] = []
    latest_checks = latest_run_detail.get("agent_checks", []) if latest_run_detail else []
    for label in labels:
        matching_check = next((item for item in latest_checks if str(item.get("label", "")) == label), None)
        label_outcomes.append({
            "label": label,
            "latest_run_id": latest_run_id,
            "latest_success": _agent_check_success(matching_check) if matching_check is not None else None,
            "latest_outcome": matching_check.get("outcome") if matching_check is not None else None,
            "latest_response": matching_check.get("response") if matching_check is not None else None,
            "comparison": _label_run_comparison(label),
            "rate_window": _label_rate_window(label),
        })

    request_progress: List[Dict[str, Any]] = []
    generated_run_ids = [str(item.get("run_id", "")) for item in generated_runs if str(item.get("run_id", ""))]
    first_run_id = generated_run_ids[0] if generated_run_ids else None
    for label in labels:
        first_check = None
        latest_check = None
        if first_run_id and first_run_id in generated_run_details:
            first_check = next(
                (item for item in generated_run_details[first_run_id].get("agent_checks", []) if str(item.get("label", "")) == label),
                None,
            )
        if latest_run_id and latest_run_id in generated_run_details:
            latest_check = next(
                (item for item in generated_run_details[latest_run_id].get("agent_checks", []) if str(item.get("label", "")) == label),
                None,
            )
        first_success = _agent_check_success(first_check) if first_check is not None else None
        latest_success = _agent_check_success(latest_check) if latest_check is not None else None
        if first_success is None or latest_success is None:
            trend = "insufficient_data"
        elif latest_success and not first_success:
            trend = "improved"
        elif not latest_success and first_success:
            trend = "regressed"
        elif latest_success:
            trend = "stable_pass"
        else:
            trend = "stable_fail"
        request_progress.append({
            "label": label,
            "first_run_id": first_run_id,
            "latest_run_id": latest_run_id,
            "first_success": first_success,
            "latest_success": latest_success,
            "first_outcome": first_check.get("outcome") if first_check is not None else None,
            "latest_outcome": latest_check.get("outcome") if latest_check is not None else None,
            "trend": trend,
        })

    detail: Dict[str, Any] = {
        "request": request_entry,
        "completion": completion_entry,
        "labels": labels,
        "latest_run_id": latest_run_id,
        "latest_run_detail": latest_run_detail,
        "generated_runs": generated_runs,
        "label_outcomes": label_outcomes,
        "request_progress": request_progress,
    }
    return QualificationRerunDetailResponse(found=True, request_id=request_id, detail=detail)


def _build_qualification_rerun_command(payload: QualificationRerunRequest, *, request_id: str) -> str:
    args = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_qualification_cycle.py"),
        "--agent-check-label",
        payload.label,
        "--iterations",
        str(max(1, payload.iterations)),
        "--prompt-timeout-seconds",
        str(payload.prompt_timeout_seconds),
        "--run-timeout-seconds",
        str(payload.run_timeout_seconds),
        "--request-id",
        request_id,
        "--rerun-history-path",
        str(QUALIFICATION_RERUN_HISTORY_PATH),
    ]
    if payload.include_direct_checks:
        args.append("--include-direct-checks")
    return " ".join(shlex.quote(arg) for arg in args)


def build_operations_router(runtime: Any) -> APIRouter:
    """Build operations routes for session inspection and work management."""
    r = APIRouter(prefix="/api/operations", tags=["operations"])

    def _find_browser_session_entry(scope_key: str, session_id: str) -> Optional[Dict[str, Any]]:
        snapshot = runtime.browser_supervisor.snapshot(scope_key=scope_key)
        return next((item for item in snapshot.get("entries", []) if item.get("session_id") == session_id), None)

    def _merge_browser_observation(
        entry: Dict[str, Any],
        observed: Optional[Dict[str, Any]],
        *,
        url_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not observed or not observed.get("found", False):
            return entry
        return {
            **entry,
            "url": url_hint or observed.get("url", entry.get("url")),
            "title": observed.get("title"),
            "last_snapshot_text": observed.get("text"),
            "last_snapshot_links": observed.get("links", []),
            "last_snapshot_screenshot": observed.get("screenshot_path"),
            "last_observed_at": time.time(),
        }

    # ── Session Inspection ────────────────────────────────────────────

    @r.get("/qualification", response_model=QualificationSummaryResponse)
    async def get_qualification_summary() -> QualificationSummaryResponse:
        summary = _load_qualification_summary()
        if hasattr(runtime, "process_supervisor"):
            snapshot = runtime.process_supervisor.snapshot(scope_key="qualification")
            summary.active_reruns = snapshot.get("entries", [])
        summary.recommended_reruns = _annotate_recommendation_runtime_state(
            summary.recommended_reruns,
            summary.active_reruns,
        )
        summary.recent_runs = [item.model_dump(mode="json") for item in _load_validation_runs(limit=5).items]
        summary.recent_rerun_history = _load_recent_rerun_history(limit=10)
        summary.remediation_rollup = _load_qualification_remediation()
        return summary

    @r.get("/qualification/labels/{label}", response_model=QualificationLabelDetailResponse)
    async def get_qualification_label(label: str) -> QualificationLabelDetailResponse:
        detail = _load_qualification_label_detail(label)
        if detail.found and hasattr(runtime, "process_supervisor"):
            snapshot = runtime.process_supervisor.snapshot(scope_key="qualification")
            active = []
            for item in snapshot.get("entries", []):
                metadata = item.get("metadata", {}) or {}
                if str(metadata.get("source_label", "") or "") == label and item.get("running", False):
                    active.append(item)
            detail.detail["active_reruns"] = active
        return detail

    @r.get("/qualification/reruns/{request_id}", response_model=QualificationRerunDetailResponse)
    async def get_qualification_rerun(request_id: str) -> QualificationRerunDetailResponse:
        detail = _load_qualification_rerun_detail(request_id)
        if detail.found and hasattr(runtime, "process_supervisor"):
            snapshot = runtime.process_supervisor.snapshot(scope_key="qualification")
            active = []
            for item in snapshot.get("entries", []):
                metadata = item.get("metadata", {}) or {}
                if str(metadata.get("request_id", "") or "") == request_id and item.get("running", False):
                    active.append(item)
            detail.detail["active_processes"] = active
        return detail

    @r.get("/validation-runs", response_model=ValidationRunListResponse)
    async def list_validation_runs(limit: int = 10, label: Optional[str] = None) -> ValidationRunListResponse:
        return _load_validation_runs(limit=max(1, min(limit, 50)), label=label)

    @r.get("/validation-runs/{run_id}", response_model=ValidationRunDetailResponse)
    async def get_validation_run(run_id: str, label: Optional[str] = None) -> ValidationRunDetailResponse:
        return _load_validation_run_detail(run_id, label=label)

    @r.post("/qualification/reruns")
    async def start_qualification_rerun(payload: QualificationRerunRequest) -> Dict[str, Any]:
        if not hasattr(runtime, "process_supervisor"):
            return {"ok": False, "error": "Process supervisor not available"}
        request_id = uuid4().hex
        command = _build_qualification_rerun_command(payload, request_id=request_id)
        metadata = {
            "kind": "qualification_rerun",
            "request_id": request_id,
            "source_label": payload.source_label or payload.label,
            "source_note": payload.source_note or "",
            "requested_at": time.time(),
        }
        process_id = runtime.process_supervisor.start(
            "qualification",
            command,
            cwd=str(REPO_ROOT),
            metadata=metadata,
        )
        history_entry = {
            "event": "requested",
            "request_id": request_id,
            "process_id": process_id,
            "label": payload.label,
            "source_label": metadata["source_label"],
            "source_note": metadata["source_note"],
            "requested_at": metadata["requested_at"],
            "command": command,
            "iterations": max(1, payload.iterations),
            "include_direct_checks": bool(payload.include_direct_checks),
        }
        _append_qualification_rerun_history(history_entry)
        return {
            "ok": True,
            "process_id": process_id,
            "scope_key": "qualification",
            "command": command,
            "metadata": metadata,
            "history_entry": history_entry,
        }

    @r.get("/hardening")
    async def get_hardening_summary(
        window_days: int = 7,
        bucket_hours: int = 6,
        decision_limit: int = 12,
    ) -> Dict[str, Any]:
        return await _build_hardening_snapshot(
            runtime,
            window_days=max(1, min(window_days, 30)),
            bucket_hours=max(1, min(bucket_hours, 24)),
            decision_limit=max(1, min(decision_limit, 50)),
        )

    @r.get("/memory-value")
    async def get_memory_value() -> Dict[str, Any]:
        return await _build_memory_value_snapshot(runtime)

    @r.get("/approval-audit")
    async def get_approval_audit(window_days: int = 7, limit: int = 12) -> Dict[str, Any]:
        return await _build_approval_audit_snapshot(
            runtime,
            window_days=max(1, min(window_days, 30)),
            limit=max(1, min(limit, 50)),
        )

    @r.get("/costs")
    async def get_costs(window_days: int = 7, bucket_hours: int = 6) -> Dict[str, Any]:
        return await _build_cost_snapshot(
            runtime,
            window_days=max(1, min(window_days, 30)),
            bucket_hours=max(1, min(bucket_hours, 24)),
        )

    @r.get("/sessions", response_model=SessionListResponse)
    async def list_sessions(scope_key: Optional[str] = None) -> SessionListResponse:
        process_entries: List[Dict[str, Any]] = []
        pty_entries: List[SessionEntry] = []
        browser_entries: List[Dict[str, Any]] = []
        scope_summary: Dict[str, Dict[str, int]] = {}
        total_processes = 0
        total_pty = 0
        total_browser = 0

        if hasattr(runtime, "process_supervisor"):
            p_snapshot = runtime.process_supervisor.snapshot(scope_key=scope_key)
            total_processes = p_snapshot.get("total_count", 0)
            process_entries = p_snapshot.get("entries", [])
            for entry in process_entries:
                entry_scope = str(entry.get("scope_key", "") or "default")
                scope_summary.setdefault(entry_scope, {"process_count": 0, "pty_count": 0, "browser_count": 0})["process_count"] += 1

        if hasattr(runtime, "pty_supervisor"):
            snapshot = runtime.pty_supervisor.snapshot(scope_key=scope_key)
            total_pty = snapshot.get("total_count", 0)
            for entry in snapshot.get("entries", []):
                entry_scope = str(entry.get("scope_key", "") or "default")
                scope_summary.setdefault(entry_scope, {"process_count": 0, "pty_count": 0, "browser_count": 0})["pty_count"] += 1
                pty_entries.append(_build_session_entry(entry))

        if hasattr(runtime, "browser_supervisor"):
            b_snapshot = runtime.browser_supervisor.snapshot(scope_key=scope_key)
            total_browser = b_snapshot.get("total_count", 0)
            browser_entries = b_snapshot.get("entries", [])
            for entry in browser_entries:
                entry_scope = str(entry.get("scope_key", "") or "default")
                scope_summary.setdefault(entry_scope, {"process_count": 0, "pty_count": 0, "browser_count": 0})["browser_count"] += 1

        return SessionListResponse(
            processes=process_entries,
            pty=pty_entries,
            browser=browser_entries,
            scopes=[
                SessionScopeEntry(scope_key=scope, **counts)
                for scope, counts in sorted(scope_summary.items(), key=lambda item: item[0])
            ],
            current_scope=scope_key or None,
            total_processes=total_processes,
            total_pty=total_pty,
            total_browser=total_browser,
        )

    @r.get("/sessions/process/{process_id}", response_model=ProcessDetailResponse)
    async def get_process_session(
        process_id: str,
        scope_key: str = "default",
        refresh: bool = True,
    ) -> ProcessDetailResponse:
        if not hasattr(runtime, "process_supervisor"):
            return ProcessDetailResponse(found=False, process={"error": "Process supervisor not available"})
        snapshot = runtime.process_supervisor.snapshot(scope_key=scope_key)
        entry = next((item for item in snapshot.get("entries", []) if item.get("process_id") == process_id), None)
        if entry is None:
            return ProcessDetailResponse(found=False)
        polled = runtime.process_supervisor.poll(scope_key, process_id) if refresh else {}
        metadata = entry.get("metadata", {}) or {}
        rerun_request, rerun_completion = _find_rerun_history_by_request_id(metadata.get("request_id"))
        return ProcessDetailResponse(
            found=True,
            process={
                **entry,
                "running": polled.get("running", entry.get("running", False)),
                "returncode": polled.get("returncode", entry.get("returncode")),
                "polled": polled,
                "rerun_request": rerun_request,
                "rerun_completion": rerun_completion,
                "recent_operator_actions": _load_recent_operator_actions(
                    runtime,
                    target_kind="process",
                    target_id=process_id,
                    scope_key=scope_key,
                ),
            },
        )

    @r.delete("/sessions/process/{process_id}")
    async def kill_process_session(
        process_id: str,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        if not hasattr(runtime, "process_supervisor"):
            return {"ok": False, "error": "Process supervisor not available"}
        ok = runtime.process_supervisor.kill(scope_key, process_id)
        runtime.process_supervisor.remove(scope_key, process_id)
        _append_operator_action(
            runtime,
            {
                "action": "kill_process",
                "target_kind": "process",
                "target_id": process_id,
                "scope_key": scope_key,
                "ok": bool(ok),
            },
        )
        return {"ok": ok, "process_id": process_id}

    @r.delete("/sessions/process")
    async def clear_process_sessions(scope_key: str = "default") -> Dict[str, Any]:
        if not hasattr(runtime, "process_supervisor"):
            return {"ok": False, "error": "Process supervisor not available"}
        removed = runtime.process_supervisor.clear(scope_key)
        return {"ok": True, "removed": removed, "scope_key": scope_key}

    @r.delete("/sessions/pty/{session_id}")
    async def kill_pty_session(
        session_id: str, scope_key: str = "default"
    ) -> Dict[str, Any]:
        if not hasattr(runtime, "pty_supervisor"):
            return {"ok": False, "error": "PTY supervisor not available"}
        ok = runtime.pty_supervisor.kill(scope_key, session_id)
        runtime.pty_supervisor.remove(scope_key, session_id)
        return {"ok": ok, "session_id": session_id}

    @r.delete("/sessions/pty")
    async def clear_pty_sessions(scope_key: str = "default") -> Dict[str, Any]:
        if not hasattr(runtime, "pty_supervisor"):
            return {"ok": False, "error": "PTY supervisor not available"}
        removed = runtime.pty_supervisor.clear(scope_key)
        return {"ok": True, "removed": removed, "scope_key": scope_key}

    @r.get("/sessions/pty/{session_id}")
    async def get_pty_session(
        session_id: str,
        scope_key: str = "default",
        refresh: bool = False,
        idle_seconds: float = 0.25,
        max_wait_seconds: float = 1.5,
    ) -> Dict[str, Any]:
        if not hasattr(runtime, "pty_supervisor"):
            return {"found": False, "error": "PTY supervisor not available"}

        snapshot = runtime.pty_supervisor.snapshot(scope_key=scope_key, sample_limit=500)
        entry = next((item for item in snapshot.get("entries", []) if item.get("session_id") == session_id), None)
        if entry is None:
            return {"found": False}

        observed = None
        if refresh and entry.get("running", False):
            observed = runtime.pty_supervisor.observe_until_quiet(
                scope_key,
                session_id,
                idle_seconds=idle_seconds,
                max_wait_seconds=max_wait_seconds,
            )
            entry = {
                **entry,
                "running": observed.get("running", entry.get("running", False)),
                "returncode": observed.get("returncode", entry.get("returncode")),
                "last_screen_state": observed.get("screen_state", entry.get("last_screen_state", {})),
                "last_cleaned_output": observed.get("cleaned_combined_output", entry.get("last_cleaned_output")),
                "last_observed_at": time.time() if observed.get("elapsed_ms") is not None else entry.get("last_observed_at"),
            }

        return {
            "found": True,
            "session": _build_session_entry(entry).model_dump(mode="json"),
            "observed": observed,
            "recent_operator_actions": _load_recent_operator_actions(
                runtime,
                target_kind="pty",
                target_id=session_id,
                scope_key=scope_key,
            ),
        }

    @r.post("/sessions/pty/{session_id}/input")
    async def send_pty_input(
        session_id: str,
        payload: PtyInputRequest,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        if not hasattr(runtime, "pty_supervisor"):
            return {"found": False, "error": "PTY supervisor not available"}

        snapshot = runtime.pty_supervisor.snapshot(scope_key=scope_key, sample_limit=500)
        entry = next((item for item in snapshot.get("entries", []) if item.get("session_id") == session_id), None)
        if entry is None:
            return {"found": False}

        ok = runtime.pty_supervisor.write(scope_key, session_id, payload.input)
        if not ok:
            return {"found": False, "error": "Failed to write PTY input"}

        observed = None
        if payload.observe:
            observed = runtime.pty_supervisor.observe_until_quiet(
                scope_key,
                session_id,
                idle_seconds=payload.idle_seconds,
                max_wait_seconds=payload.max_wait_seconds,
            )

        refreshed = runtime.pty_supervisor.snapshot(scope_key=scope_key, sample_limit=500)
        updated_entry = next(
            (item for item in refreshed.get("entries", []) if item.get("session_id") == session_id),
            entry,
        )
        _append_operator_action(
            runtime,
            {
                "action": "pty_input",
                "target_kind": "pty",
                "target_id": session_id,
                "scope_key": scope_key,
                "ok": True,
                "input_length": len(payload.input or ""),
                "input_preview": _truncate_text(payload.input),
                "observe": bool(payload.observe),
                "observed_mode": (observed or {}).get("screen_state", {}).get("mode"),
            },
        )
        return {
            "found": True,
            "ok": True,
            "session": _build_session_entry(updated_entry).model_dump(mode="json"),
            "observed": observed,
            "recent_operator_actions": _load_recent_operator_actions(
                runtime,
                target_kind="pty",
                target_id=session_id,
                scope_key=scope_key,
            ),
        }

    @r.get("/sessions/browser/{session_id}")
    async def get_browser_session(
        session_id: str,
        scope_key: str = "default",
        refresh: bool = False,
        capture_screenshot: bool = False,
        full_page: bool = False,
    ) -> Dict[str, Any]:
        if not hasattr(runtime, "browser_supervisor"):
            return {"found": False, "error": "Browser supervisor not available"}

        entry = _find_browser_session_entry(scope_key=scope_key, session_id=session_id)
        if entry is None:
            return {"found": False}

        observed = None
        if refresh:
            observed = await runtime.browser_supervisor.snapshot_page(
                scope_key=scope_key,
                session_id=session_id,
                capture_screenshot=capture_screenshot,
                full_page=full_page,
            )
            if not observed.get("found", False):
                return {"found": False, "error": observed.get("error", "Browser session not found")}
            entry = _merge_browser_observation(entry, observed)

        return {
            "found": True,
            "session": entry,
            "observed": observed,
            "recent_operator_actions": _load_recent_operator_actions(
                runtime,
                target_kind="browser",
                target_id=session_id,
                scope_key=scope_key,
            ),
        }

    @r.delete("/sessions/browser/{session_id}")
    async def close_browser_session(
        session_id: str,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        if not hasattr(runtime, "browser_supervisor"):
            return {"ok": False, "error": "Browser supervisor not available"}
        ok = await runtime.browser_supervisor.close(scope_key=scope_key, session_id=session_id)
        _append_operator_action(
            runtime,
            {
                "action": "close_browser",
                "target_kind": "browser",
                "target_id": session_id,
                "scope_key": scope_key,
                "ok": bool(ok),
            },
        )
        return {"ok": ok, "session_id": session_id}

    @r.delete("/sessions/browser")
    async def clear_browser_sessions(scope_key: str = "default") -> Dict[str, Any]:
        if not hasattr(runtime, "browser_supervisor"):
            return {"ok": False, "error": "Browser supervisor not available"}
        removed = await runtime.browser_supervisor.clear(scope_key=scope_key)
        return {"ok": True, "removed": removed, "scope_key": scope_key}

    @r.post("/sessions/browser/{session_id}/navigate")
    async def navigate_browser_session(
        session_id: str,
        payload: BrowserNavigateRequest,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        if not hasattr(runtime, "browser_supervisor"):
            return {"found": False, "error": "Browser supervisor not available"}

        entry = _find_browser_session_entry(scope_key=scope_key, session_id=session_id)
        if entry is None:
            return {"found": False}

        result = await runtime.browser_supervisor.navigate(
            scope_key=scope_key,
            session_id=session_id,
            url=payload.url,
            wait_until=payload.wait_until,
            timeout_ms=payload.timeout_ms,
        )
        if not result.get("found", False):
            return {"found": False, "error": result.get("error", "Browser session not found")}

        observed = None
        if payload.refresh:
            observed = await runtime.browser_supervisor.snapshot_page(
                scope_key=scope_key,
                session_id=session_id,
            )

        refreshed_entry = _find_browser_session_entry(scope_key=scope_key, session_id=session_id) or entry
        updated_entry = _merge_browser_observation(
            refreshed_entry,
            observed,
            url_hint=result.get("url"),
        )
        return {
            "found": True,
            "session": updated_entry,
            "observed": observed,
            "navigate": result,
            "recent_operator_actions": (
                _append_operator_action(
                    runtime,
                    {
                        "action": "browser_navigate",
                        "target_kind": "browser",
                        "target_id": session_id,
                        "scope_key": scope_key,
                        "ok": True,
                        "url": payload.url,
                        "wait_until": payload.wait_until,
                        "timeout_ms": payload.timeout_ms,
                    },
                ),
                _load_recent_operator_actions(
                    runtime,
                    target_kind="browser",
                    target_id=session_id,
                    scope_key=scope_key,
                ),
            )[1],
        }

    @r.post("/sessions/browser/{session_id}/click")
    async def click_browser_session(
        session_id: str,
        payload: BrowserClickRequest,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        if not hasattr(runtime, "browser_supervisor"):
            return {"found": False, "error": "Browser supervisor not available"}

        entry = _find_browser_session_entry(scope_key=scope_key, session_id=session_id)
        if entry is None:
            return {"found": False}

        result = await runtime.browser_supervisor.click(
            scope_key=scope_key,
            session_id=session_id,
            selector=payload.selector,
            timeout_ms=payload.timeout_ms,
        )
        if not result.get("found", False):
            return {"found": False, "error": result.get("error", "Browser session not found")}

        observed = None
        if payload.refresh:
            observed = await runtime.browser_supervisor.snapshot_page(
                scope_key=scope_key,
                session_id=session_id,
            )

        refreshed_entry = _find_browser_session_entry(scope_key=scope_key, session_id=session_id) or entry
        updated_entry = _merge_browser_observation(
            refreshed_entry,
            observed,
            url_hint=result.get("url"),
        )
        return {
            "found": True,
            "session": updated_entry,
            "observed": observed,
            "click": result,
            "recent_operator_actions": (
                _append_operator_action(
                    runtime,
                    {
                        "action": "browser_click",
                        "target_kind": "browser",
                        "target_id": session_id,
                        "scope_key": scope_key,
                        "ok": True,
                        "selector": payload.selector,
                        "timeout_ms": payload.timeout_ms,
                    },
                ),
                _load_recent_operator_actions(
                    runtime,
                    target_kind="browser",
                    target_id=session_id,
                    scope_key=scope_key,
                ),
            )[1],
        }

    @r.post("/sessions/browser/{session_id}/type")
    async def type_browser_session(
        session_id: str,
        payload: BrowserTypeRequest,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        if not hasattr(runtime, "browser_supervisor"):
            return {"found": False, "error": "Browser supervisor not available"}

        entry = _find_browser_session_entry(scope_key=scope_key, session_id=session_id)
        if entry is None:
            return {"found": False}

        result = await runtime.browser_supervisor.type_text(
            scope_key=scope_key,
            session_id=session_id,
            selector=payload.selector,
            text=payload.text,
            clear=payload.clear,
            timeout_ms=payload.timeout_ms,
        )
        if not result.get("found", False):
            return {"found": False, "error": result.get("error", "Browser session not found")}

        observed = None
        if payload.refresh:
            observed = await runtime.browser_supervisor.snapshot_page(
                scope_key=scope_key,
                session_id=session_id,
            )

        refreshed_entry = _find_browser_session_entry(scope_key=scope_key, session_id=session_id) or entry
        updated_entry = _merge_browser_observation(
            refreshed_entry,
            observed,
            url_hint=result.get("url"),
        )
        return {
            "found": True,
            "session": updated_entry,
            "observed": observed,
            "type": result,
            "recent_operator_actions": (
                _append_operator_action(
                    runtime,
                    {
                        "action": "browser_type",
                        "target_kind": "browser",
                        "target_id": session_id,
                        "scope_key": scope_key,
                        "ok": True,
                        "selector": payload.selector,
                        "text_length": len(payload.text or ""),
                        "text_preview": _truncate_text(payload.text),
                        "clear": bool(payload.clear),
                        "timeout_ms": payload.timeout_ms,
                    },
                ),
                _load_recent_operator_actions(
                    runtime,
                    target_kind="browser",
                    target_id=session_id,
                    scope_key=scope_key,
                ),
            )[1],
        }

    @r.post("/sessions/browser/{session_id}/press")
    async def press_browser_session(
        session_id: str,
        payload: BrowserPressRequest,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        if not hasattr(runtime, "browser_supervisor"):
            return {"found": False, "error": "Browser supervisor not available"}

        entry = _find_browser_session_entry(scope_key=scope_key, session_id=session_id)
        if entry is None:
            return {"found": False}

        result = await runtime.browser_supervisor.press(
            scope_key=scope_key,
            session_id=session_id,
            key=payload.key,
        )
        if not result.get("found", False):
            return {"found": False, "error": result.get("error", "Browser session not found")}

        observed = None
        if payload.refresh:
            observed = await runtime.browser_supervisor.snapshot_page(
                scope_key=scope_key,
                session_id=session_id,
            )

        refreshed_entry = _find_browser_session_entry(scope_key=scope_key, session_id=session_id) or entry
        updated_entry = _merge_browser_observation(
            refreshed_entry,
            observed,
            url_hint=result.get("url"),
        )
        return {
            "found": True,
            "session": updated_entry,
            "observed": observed,
            "press": result,
            "recent_operator_actions": (
                _append_operator_action(
                    runtime,
                    {
                        "action": "browser_press",
                        "target_kind": "browser",
                        "target_id": session_id,
                        "scope_key": scope_key,
                        "ok": True,
                        "key": payload.key,
                    },
                ),
                _load_recent_operator_actions(
                    runtime,
                    target_kind="browser",
                    target_id=session_id,
                    scope_key=scope_key,
                ),
            )[1],
        }

    @r.post("/sessions/browser/{session_id}/wait")
    async def wait_browser_session(
        session_id: str,
        payload: BrowserWaitRequest,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        if not hasattr(runtime, "browser_supervisor"):
            return {"found": False, "error": "Browser supervisor not available"}

        entry = _find_browser_session_entry(scope_key=scope_key, session_id=session_id)
        if entry is None:
            return {"found": False}

        result = await runtime.browser_supervisor.wait(
            scope_key=scope_key,
            session_id=session_id,
            timeout_ms=payload.timeout_ms,
            load_state=payload.load_state,
            selector=payload.selector,
        )
        if not result.get("found", False):
            return {"found": False, "error": result.get("error", "Browser session not found")}

        observed = None
        if payload.refresh:
            observed = await runtime.browser_supervisor.snapshot_page(
                scope_key=scope_key,
                session_id=session_id,
            )

        refreshed_entry = _find_browser_session_entry(scope_key=scope_key, session_id=session_id) or entry
        updated_entry = _merge_browser_observation(
            refreshed_entry,
            observed,
            url_hint=result.get("url"),
        )
        return {
            "found": True,
            "session": updated_entry,
            "observed": observed,
            "wait": result,
            "recent_operator_actions": (
                _append_operator_action(
                    runtime,
                    {
                        "action": "browser_wait",
                        "target_kind": "browser",
                        "target_id": session_id,
                        "scope_key": scope_key,
                        "ok": True,
                        "selector": payload.selector,
                        "load_state": payload.load_state,
                        "timeout_ms": payload.timeout_ms,
                    },
                ),
                _load_recent_operator_actions(
                    runtime,
                    target_kind="browser",
                    target_id=session_id,
                    scope_key=scope_key,
                ),
            )[1],
        }

    @r.post("/sessions/browser/{session_id}/capture")
    async def capture_browser_session(
        session_id: str,
        payload: BrowserCaptureRequest,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        if not hasattr(runtime, "browser_supervisor"):
            return {"found": False, "error": "Browser supervisor not available"}

        entry = _find_browser_session_entry(scope_key=scope_key, session_id=session_id)
        if entry is None:
            return {"found": False}

        observed = await runtime.browser_supervisor.snapshot_page(
            scope_key=scope_key,
            session_id=session_id,
            capture_screenshot=True,
            full_page=payload.full_page,
        )
        if not observed.get("found", False):
            return {"found": False, "error": observed.get("error", "Browser session not found")}

        refreshed_entry = _find_browser_session_entry(scope_key=scope_key, session_id=session_id) or entry
        updated_entry = _merge_browser_observation(
            refreshed_entry,
            observed,
            url_hint=observed.get("url"),
        )
        return {
            "found": True,
            "session": updated_entry,
            "observed": observed,
            "capture": {
                "screenshot_path": observed.get("screenshot_path"),
                "full_page": payload.full_page,
            },
            "recent_operator_actions": (
                _append_operator_action(
                    runtime,
                    {
                        "action": "browser_capture",
                        "target_kind": "browser",
                        "target_id": session_id,
                        "scope_key": scope_key,
                        "ok": bool(observed.get("found", False)),
                        "full_page": bool(payload.full_page),
                        "screenshot_path": observed.get("screenshot_path"),
                    },
                ),
                _load_recent_operator_actions(
                    runtime,
                    target_kind="browser",
                    target_id=session_id,
                    scope_key=scope_key,
                ),
            )[1],
        }

    @r.get("/sessions/browser/{session_id}/screenshot")
    async def get_browser_session_screenshot(
        session_id: str,
        scope_key: str = "default",
    ) -> Any:
        if not hasattr(runtime, "browser_supervisor"):
            return {"found": False, "error": "Browser supervisor not available"}

        entry = _find_browser_session_entry(scope_key=scope_key, session_id=session_id)
        if entry is None:
            return {"found": False}

        screenshot_path = entry.get("last_snapshot_screenshot")
        if not screenshot_path:
            return {"found": False, "error": "No captured browser screenshot for this session"}
        path = Path(screenshot_path)
        if not path.exists():
            return {"found": False, "error": "Captured browser screenshot is missing"}
        return FileResponse(path, media_type="image/png", filename=path.name)

    # ── Receipts ──────────────────────────────────────────────────────

    @r.get("/receipts", response_model=ReceiptListResponse)
    async def list_receipts(limit: int = 50) -> ReceiptListResponse:
        store = getattr(runtime.ctx, "receipt_store", None)
        if store is None:
            return ReceiptListResponse(count=0, items=[])

        receipts = await store.list_recent(limit=limit)
        items = []
        for receipt in receipts:
            items.append(ReceiptEntry(
                receipt_id=str(getattr(receipt, "receipt_id", "")),
                task_id=str(getattr(receipt, "task_id", "")),
                status=str(getattr(receipt, "status", "")),
                tool_name=getattr(receipt, "tool_name", None),
                started_at=receipt.started_at.isoformat() if getattr(receipt, "started_at", None) else None,
                finished_at=receipt.finished_at.isoformat() if getattr(receipt, "finished_at", None) else None,
                duration_ms=getattr(receipt, "duration_ms", None),
            ))
        return ReceiptListResponse(count=len(items), items=items)

    @r.get("/receipts/{receipt_id}")
    async def get_receipt(receipt_id: str) -> Dict[str, Any]:
        store = getattr(runtime.ctx, "receipt_store", None)
        if store is None:
            return {"found": False, "error": "Receipt store not available"}
        receipt = await store.get(receipt_id)
        if receipt is None:
            return {"found": False}
        return {"found": True, "receipt": receipt.model_dump(mode="json")}

    # ── Background Tasks ──────────────────────────────────────────────

    @r.get("/tasks", response_model=TaskListResponse)
    async def list_tasks(limit: int = 50) -> TaskListResponse:
        store = getattr(runtime.ctx, "tasks", None)
        if store is None:
            return TaskListResponse(
                counts={
                    "total": 0,
                    "active": 0,
                    "waiting": 0,
                    "completed": 0,
                    "failed": 0,
                },
                items=[],
            )

        sample = await store.list_all(limit=max(limit, 250))
        objective_counts: Dict[str, int] = {}
        for item in sample:
            objective_counts[item.objective] = objective_counts.get(item.objective, 0) + 1

        counts = {
            "total": len(sample),
            "active": 0,
            "waiting": 0,
            "completed": 0,
            "failed": 0,
        }
        items: List[TaskEntry] = []
        for item in sample[:limit]:
            ui_status = _task_ui_status(
                item.stage.value if hasattr(item.stage, "value") else str(item.stage),
                item.status,
            )
            if ui_status in {"queued", "planning", "executing", "verifying", "recovering"}:
                counts["active"] += 1
            elif ui_status in {"needs approval", "needs clarification"}:
                counts["waiting"] += 1
            elif ui_status == "completed":
                counts["completed"] += 1
            elif ui_status == "failed":
                counts["failed"] += 1
            items.append(
                TaskEntry(
                    task_id=str(item.task_id),
                    title=_human_title(item.meta.get("title") or item.objective, fallback="Background task"),
                    objective=item.objective,
                    status=ui_status,
                    stage=item.stage.value if hasattr(item.stage, "value") else str(item.stage),
                    source=str(item.meta.get("source", "") or "") or None,
                    project_id=item.project_id,
                    commitment_id=item.commitment_id,
                    updated_at=item.updated_at.isoformat(),
                    duplicate_objective_count=objective_counts.get(item.objective, 1),
                )
            )
        if len(sample) > limit:
            for item in sample[limit:]:
                ui_status = _task_ui_status(
                    item.stage.value if hasattr(item.stage, "value") else str(item.stage),
                    item.status,
                )
                if ui_status in {"queued", "planning", "executing", "verifying", "recovering"}:
                    counts["active"] += 1
                elif ui_status in {"needs approval", "needs clarification"}:
                    counts["waiting"] += 1
                elif ui_status == "completed":
                    counts["completed"] += 1
                elif ui_status == "failed":
                    counts["failed"] += 1
        return TaskListResponse(counts=counts, items=items)

    @r.get("/tasks/{task_id}")
    async def get_task(task_id: str) -> Dict[str, Any]:
        store = getattr(runtime.ctx, "tasks", None)
        if store is None:
            return {"found": False, "error": "Task store not available"}
        task = await store.get(task_id)
        if task is None:
            return {"found": False}
        result = await store.get_result(task_id)
        lifecycle = await store.list_lifecycle_transitions(task_id, limit=50)
        related = await store.list_all(limit=100)
        duplicate_count = sum(1 for item in related if item.objective == task.objective)
        return {
            "found": True,
            "task": {
                "task_id": str(task.task_id),
                "title": _human_title(task.meta.get("title") or task.objective, fallback="Background task"),
                "objective": task.objective,
                "status": _task_ui_status(
                    task.stage.value if hasattr(task.stage, "value") else str(task.stage),
                    task.status,
                ),
                "raw_status": task.status,
                "stage": task.stage.value if hasattr(task.stage, "value") else str(task.stage),
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat(),
                "source": str(task.meta.get("source", "") or "") or None,
                "project_id": task.project_id,
                "commitment_id": task.commitment_id,
                "depends_on": task.depends_on,
                "attempt": task.attempt,
                "max_attempts": task.max_attempts,
                "duplicate_objective_count": duplicate_count,
                "meta": task.meta,
                "phases": [phase.model_dump(mode="json") for phase in task.phases],
                "result": result.model_dump(mode="json") if result is not None else None,
            },
            "transitions": [
                {
                    "from_stage": item.get("from_stage"),
                    "to_stage": item.get("to_stage"),
                    "reason": item.get("reason"),
                    "timestamp": item.get("timestamp").isoformat()
                    if item.get("timestamp") is not None
                    else None,
                    "context": item.get("context", {}),
                }
                for item in lifecycle
            ],
        }

    # ── Work Items ────────────────────────────────────────────────────

    @r.get("/work", response_model=WorkListResponse)
    async def list_work(
        project_id: Optional[str] = None, limit: int = 50
    ) -> WorkListResponse:
        store = getattr(runtime.ctx, "work_store", None)
        if store is None:
            return WorkListResponse(counts={"total": 0, "ready": 0, "blocked": 0}, items=[])

        counts = await store.summary_counts()
        if project_id:
            raw_items = await store.list_by_project(project_id, limit=limit)
        else:
            raw_items = await store.list_all(limit=limit)

        items = []
        for item in raw_items:
            items.append(WorkItemEntry(
                work_id=str(item.work_id),
                title=getattr(item, "title", "") or str(item.content)[:80],
                stage=item.stage.value if hasattr(item.stage, "value") else str(item.stage),
                project_id=getattr(item, "project_id", None),
                blocked_by=getattr(item, "blocked_by", None),
            ))
        return WorkListResponse(counts=counts, items=items)

    @r.get("/work/{work_id}")
    async def get_work_item(work_id: str) -> Dict[str, Any]:
        store = getattr(runtime.ctx, "work_store", None)
        if store is None:
            return {"found": False, "error": "Work store not available"}
        item = await store.get(work_id)
        if item is None:
            return {"found": False}
        return {
            "found": True,
            "item": {
                "work_id": str(item.work_id),
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
                "stage": item.stage.value if hasattr(item.stage, "value") else str(item.stage),
                "content": item.content,
                "project_id": item.project_id,
                "commitment_id": item.commitment_id,
                "portfolio_id": item.portfolio_id,
                "dependency_ids": item.dependency_ids,
                "blocked_by": item.blocked_by,
                "meta": item.meta,
            },
        }

    @r.patch("/work/{work_id}")
    async def update_work_item(work_id: str, payload: WorkUpdateRequest) -> Dict[str, Any]:
        store = getattr(runtime.ctx, "work_store", None)
        if store is None:
            return {"found": False, "error": "Work store not available"}
        item = await store.get(work_id)
        if item is None:
            return {"found": False}

        updated = item.model_copy(deep=True)
        changed = False
        if payload.stage is not None:
            updated.stage = payload.stage
            changed = True
        if payload.content is not None:
            updated.content = payload.content
            changed = True
        if payload.blocked_by is not None:
            updated.blocked_by = payload.blocked_by
            changed = True
        if not changed:
            return {"found": True, "item": {
                "work_id": str(updated.work_id),
                "created_at": updated.created_at.isoformat(),
                "updated_at": updated.updated_at.isoformat(),
                "stage": updated.stage.value if hasattr(updated.stage, "value") else str(updated.stage),
                "content": updated.content,
                "project_id": updated.project_id,
                "commitment_id": updated.commitment_id,
                "portfolio_id": updated.portfolio_id,
                "dependency_ids": updated.dependency_ids,
                "blocked_by": updated.blocked_by,
                "meta": updated.meta,
            }}

        updated.updated_at = datetime.now(timezone.utc)
        await store.save(updated)
        return {
            "found": True,
            "item": {
                "work_id": str(updated.work_id),
                "created_at": updated.created_at.isoformat(),
                "updated_at": updated.updated_at.isoformat(),
                "stage": updated.stage.value if hasattr(updated.stage, "value") else str(updated.stage),
                "content": updated.content,
                "project_id": updated.project_id,
                "commitment_id": updated.commitment_id,
                "portfolio_id": updated.portfolio_id,
                "dependency_ids": updated.dependency_ids,
                "blocked_by": updated.blocked_by,
                "meta": updated.meta,
            },
        }

    # ── Commitments ───────────────────────────────────────────────────

    @r.get("/commitments", response_model=CommitmentListResponse)
    async def list_commitments(
        status: str = "active", limit: int = 50
    ) -> CommitmentListResponse:
        store = getattr(runtime, "commitment_store", None)
        if store is None:
            return CommitmentListResponse(count=0, items=[])

        from opencas.autonomy.commitment import CommitmentStatus
        status_map = {
            "active": CommitmentStatus.ACTIVE,
            "completed": CommitmentStatus.COMPLETED,
            "abandoned": CommitmentStatus.ABANDONED,
            "blocked": CommitmentStatus.BLOCKED,
        }
        cs = status_map.get(status, CommitmentStatus.ACTIVE)
        raw_items = await store.list_by_status(cs, limit=limit)

        items = []
        for item in raw_items:
            items.append(CommitmentEntry(
                commitment_id=str(item.commitment_id),
                content=item.content,
                status=item.status.value,
                priority=item.priority,
                tags=item.tags,
                deadline=item.deadline.isoformat() if item.deadline else None,
            ))
        return CommitmentListResponse(count=len(items), items=items)

    @r.get("/commitments/{commitment_id}")
    async def get_commitment(commitment_id: str) -> Dict[str, Any]:
        store = getattr(runtime, "commitment_store", None)
        if store is None:
            return {"found": False, "error": "Commitment store not available"}
        item = await store.get(commitment_id)
        if item is None:
            return {"found": False}
        return {
            "found": True,
            "commitment": {
                "commitment_id": str(item.commitment_id),
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
                "content": item.content,
                "status": item.status.value,
                "priority": item.priority,
                "tags": item.tags,
                "deadline": item.deadline.isoformat() if item.deadline else None,
                "linked_work_ids": item.linked_work_ids,
                "linked_task_ids": item.linked_task_ids,
                "meta": item.meta,
            },
        }

    @r.patch("/commitments/{commitment_id}")
    async def update_commitment(
        commitment_id: str,
        payload: CommitmentUpdateRequest,
    ) -> Dict[str, Any]:
        store = getattr(runtime, "commitment_store", None)
        if store is None:
            return {"found": False, "error": "Commitment store not available"}
        item = await store.get(commitment_id)
        if item is None:
            return {"found": False}

        updated = item.model_copy(deep=True)
        changed = False
        if payload.status is not None:
            updated.status = payload.status
            changed = True
        if payload.content is not None:
            updated.content = payload.content
            changed = True
        if payload.priority is not None:
            updated.priority = payload.priority
            changed = True
        if payload.tags is not None:
            updated.tags = payload.tags
            changed = True
        if changed:
            updated.updated_at = datetime.now(timezone.utc)
            await store.save(updated)
        return {
            "found": True,
            "commitment": {
                "commitment_id": str(updated.commitment_id),
                "created_at": updated.created_at.isoformat(),
                "updated_at": updated.updated_at.isoformat(),
                "content": updated.content,
                "status": updated.status.value,
                "priority": updated.priority,
                "tags": updated.tags,
                "deadline": updated.deadline.isoformat() if updated.deadline else None,
                "linked_work_ids": updated.linked_work_ids,
                "linked_task_ids": updated.linked_task_ids,
                "meta": updated.meta,
            },
        }

    # ── Plans ─────────────────────────────────────────────────────────

    @r.get("/plans", response_model=PlanListResponse)
    async def list_plans(
        project_id: Optional[str] = None, limit: int = 20
    ) -> PlanListResponse:
        store = getattr(runtime.ctx, "plan_store", None)
        if store is None:
            return PlanListResponse(count=0, items=[])

        plans = await store.list_active(project_id=project_id)
        items = []
        for plan in plans[:limit]:
            items.append(PlanSummary(
                plan_id=plan.plan_id,
                status=plan.status,
                content_preview=plan.content[:200],
                project_id=getattr(plan, "project_id", None),
                updated_at=plan.updated_at.isoformat() if getattr(plan, "updated_at", None) else None,
            ))
        return PlanListResponse(count=len(items), items=items)

    @r.get("/plans/{plan_id}")
    async def get_plan(plan_id: str) -> Dict[str, Any]:
        store = getattr(runtime.ctx, "plan_store", None)
        if store is None:
            return {"found": False, "error": "Plan store not available"}
        plan = await store.get_plan(plan_id)
        if plan is None:
            return {"found": False}
        actions = await store.get_actions(plan_id, limit=50)
        return {
            "found": True,
            "plan": {
                "plan_id": plan.plan_id,
                "status": plan.status,
                "content": plan.content,
                "project_id": getattr(plan, "project_id", None),
                "task_id": getattr(plan, "task_id", None),
                "updated_at": plan.updated_at.isoformat() if getattr(plan, "updated_at", None) else None,
            },
            "actions": [
                {
                    "tool_name": a.tool_name,
                    "success": a.success,
                    "result_summary": a.result_summary[:200] if a.result_summary else None,
                    "created_at": a.timestamp.isoformat() if getattr(a, "timestamp", None) else None,
                }
                for a in actions
            ],
        }

    @r.patch("/plans/{plan_id}")
    async def update_plan(plan_id: str, payload: PlanUpdateRequest) -> Dict[str, Any]:
        store = getattr(runtime.ctx, "plan_store", None)
        if store is None:
            return {"found": False, "error": "Plan store not available"}
        plan = await store.get_plan(plan_id)
        if plan is None:
            return {"found": False}

        if payload.content is not None:
            await store.update_content(plan_id, payload.content)
        if payload.status is not None:
            await store.set_status(plan_id, str(payload.status))

        updated_plan = await store.get_plan(plan_id)
        if updated_plan is None:
            return {"found": False}
        actions = await store.get_actions(plan_id, limit=50)
        return {
            "found": True,
            "plan": {
                "plan_id": updated_plan.plan_id,
                "status": updated_plan.status,
                "content": updated_plan.content,
                "project_id": getattr(updated_plan, "project_id", None),
                "task_id": getattr(updated_plan, "task_id", None),
                "updated_at": updated_plan.updated_at.isoformat() if getattr(updated_plan, "updated_at", None) else None,
            },
            "actions": [
                {
                    "tool_name": a.tool_name,
                    "success": a.success,
                    "result_summary": a.result_summary[:200] if a.result_summary else None,
                    "created_at": a.timestamp.isoformat() if getattr(a, "timestamp", None) else None,
                }
                for a in actions
            ],
        }

    return r
