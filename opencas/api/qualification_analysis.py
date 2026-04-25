"""Qualification comparison and recommendation helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from opencas.api.qualification_history import (
    load_latest_rerun_history_by_label,
    load_recent_rerun_history,
)
from opencas.api.qualification_models import (
    QualificationArtifactsPaths,
    QualificationSummaryResponse,
)


def direct_check_success(payload: Dict[str, Any]) -> bool:
    if "success" in payload:
        return bool(payload.get("success"))
    return bool(payload.get("available"))


def agent_check_success(item: Dict[str, Any]) -> bool:
    if "material_success" in item:
        return bool(item.get("material_success"))
    if item.get("timed_out", False):
        return False
    if "expected_file" in item:
        return bool(item.get("expected_file_exists", False))
    return True


def duration_seconds(report: Dict[str, Any]) -> float:
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


def load_qualification_remediation(paths: QualificationArtifactsPaths) -> Dict[str, Any]:
    path = paths.remediation_path
    if not path.exists():
        return {"found": False, "path": str(path), "items": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"found": False, "path": str(path), "error": str(exc), "items": []}
    payload["found"] = True
    payload["path"] = str(path)
    return payload


def label_run_comparison(paths: QualificationArtifactsPaths, label: str) -> Optional[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    for path in sorted(paths.validation_runs_dir.glob("*/live_debug_validation_report.json"), reverse=True):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_id = str(report.get("run_id", path.parent.name))
        finished_at = report.get("finished_at") or report.get("started_at")
        for item in report.get("agent_checks", []) or []:
            if str(item.get("label", "")) != label:
                continue
            matched.append(
                {
                    "run_id": run_id,
                    "finished_at": finished_at,
                    "success": agent_check_success(item),
                    "outcome": item.get("outcome"),
                }
            )
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


def label_rate_window(
    paths: QualificationArtifactsPaths,
    label: str,
    window_size: int = 3,
) -> Optional[Dict[str, Any]]:
    matched: List[bool] = []
    for path in sorted(paths.validation_runs_dir.glob("*/live_debug_validation_report.json"), reverse=True):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in report.get("agent_checks", []) or []:
            if str(item.get("label", "")) != label:
                continue
            matched.append(agent_check_success(item))
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


def build_qualification_recommendations(
    paths: QualificationArtifactsPaths,
    checks: Dict[str, Any],
) -> List[Dict[str, Any]]:
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
        recommendations.append(
            {
                "label": label,
                "success_rate": success_rate,
                "failures": failures,
                "timeouts": timeouts,
                "reason": ", ".join(reasons),
                "recent_failures": recent_failures,
                "note": note,
                "comparison": label_run_comparison(paths, label),
                "rate_window": label_rate_window(paths, label),
                "command": [
                    "python",
                    "scripts/run_qualification_cycle.py",
                    "--agent-check-label",
                    label,
                    "--iterations",
                    "2",
                ],
            }
        )
    recommendations.sort(
        key=lambda item: (
            item["success_rate"] if item["success_rate"] is not None else 1.0,
            -item["failures"],
            -item["timeouts"],
            item["label"],
        )
    )
    return recommendations[:5]


def load_qualification_summary(paths: QualificationArtifactsPaths) -> QualificationSummaryResponse:
    path = paths.summary_path
    if not path.exists():
        return QualificationSummaryResponse(found=False, path=str(path))

    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return QualificationSummaryResponse(found=False, path=str(path), summary={"error": str(exc)})

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
        recommended_reruns=build_qualification_recommendations(paths, checks),
    )


def annotate_recent_rerun_history(
    paths: QualificationArtifactsPaths,
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    for item in items:
        labels = [str(value) for value in item.get("labels", []) if str(value)]
        if item.get("event") == "completed" and len(labels) == 1:
            comparison = label_run_comparison(paths, labels[0])
            rate_window = label_rate_window(paths, labels[0])
        else:
            comparison = None
            rate_window = None
        annotated.append({**item, "comparison": comparison, "rate_window": rate_window})
    return annotated


def annotate_recommendation_runtime_state(
    paths: QualificationArtifactsPaths,
    recommendations: List[Dict[str, Any]],
    active_reruns: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    request_history, completion_history = load_latest_rerun_history_by_label(paths)
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
