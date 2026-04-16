"""Qualification validation-run and rerun detail loaders."""

from __future__ import annotations

import json
import shlex
import sys
from typing import Any, Dict, List, Optional

from opencas.api.qualification_analysis import (
    agent_check_success,
    direct_check_success,
    duration_seconds,
    label_rate_window,
    label_run_comparison,
    load_qualification_summary,
    annotate_recent_rerun_history,
)
from opencas.api.qualification_history import (
    find_rerun_history_by_request_id,
    find_rerun_history_for_run,
    load_recent_rerun_history,
)
from opencas.api.qualification_models import (
    QualificationArtifactsPaths,
    QualificationLabelDetailResponse,
    QualificationRerunDetailResponse,
    QualificationRerunRequest,
    ValidationRunDetailResponse,
    ValidationRunEntry,
    ValidationRunListResponse,
)


def load_validation_runs(
    paths: QualificationArtifactsPaths,
    limit: int = 10,
    label: Optional[str] = None,
) -> ValidationRunListResponse:
    items: List[ValidationRunEntry] = []
    for path in sorted(paths.validation_runs_dir.glob("*/live_debug_validation_report.json"), reverse=True):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        direct_checks = report.get("direct_checks", {}) or {}
        agent_checks = report.get("agent_checks", []) or []
        if label and not any(str(item.get("label", "")) == label for item in agent_checks):
            continue
        direct_successes = sum(1 for payload in direct_checks.values() if direct_check_success(payload))
        agent_successes = sum(1 for item in agent_checks if agent_check_success(item))
        failed_labels = [str(item.get("label", "")) for item in agent_checks if not agent_check_success(item)]
        items.append(
            ValidationRunEntry(
                run_id=str(report.get("run_id", path.parent.name)),
                started_at=report.get("started_at"),
                finished_at=report.get("finished_at"),
                model=report.get("model"),
                direct_successes=direct_successes,
                direct_total=len(direct_checks),
                agent_successes=agent_successes,
                agent_total=len(agent_checks),
                duration_seconds=duration_seconds(report),
                report_path=str(path),
                report_markdown_path=str(path.with_name("live_debug_validation_report.md")),
                aborted=bool(report.get("aborted", False)),
                failed_labels=failed_labels,
            )
        )
        if len(items) >= limit:
            break
    return ValidationRunListResponse(count=len(items), label_filter=label, items=items)


def load_validation_run_detail(
    paths: QualificationArtifactsPaths,
    run_id: str,
    label: Optional[str] = None,
) -> ValidationRunDetailResponse:
    path = paths.validation_runs_dir / run_id / "live_debug_validation_report.json"
    if not path.exists():
        return ValidationRunDetailResponse(found=False)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ValidationRunDetailResponse(found=False, run={"error": str(exc)})

    direct_checks = report.get("direct_checks", {}) or {}
    agent_checks = report.get("agent_checks", []) or []
    failed_agent_checks = [item for item in agent_checks if not agent_check_success(item)]
    matching_agent_checks = [item for item in agent_checks if not label or str(item.get("label", "")) == label]
    rerun_request, rerun_completion = find_rerun_history_for_run(paths, run_id, label=label)
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
            "duration_seconds": duration_seconds(report),
        },
    )


def load_qualification_label_detail(
    paths: QualificationArtifactsPaths,
    label: str,
) -> QualificationLabelDetailResponse:
    summary = load_qualification_summary(paths)
    if not summary.found:
        return QualificationLabelDetailResponse(found=False, label=label, detail={"summary_missing": True, "path": summary.path})

    agent_checks = summary.summary.get("agent_checks", {}) or {}
    stats = agent_checks.get(label)
    if not stats:
        return QualificationLabelDetailResponse(found=False, label=label)

    recommendations = [item for item in summary.recommended_reruns if str(item.get("label", "")) == label]
    recent_runs = [item.model_dump(mode="json") for item in load_validation_runs(paths, limit=10, label=label).items]
    recent_rerun_history = [
        item
        for item in annotate_recent_rerun_history(paths, load_recent_rerun_history(paths, limit=20))
        if label in (item.get("labels") or [])
    ]
    active_reruns = [
        item
        for item in load_recent_rerun_history(paths, limit=20)
        if item.get("event") == "requested" and label in (item.get("labels") or [])
    ]
    return QualificationLabelDetailResponse(
        found=True,
        label=label,
        detail={
            "stats": stats,
            "comparison": label_run_comparison(paths, label),
            "rate_window": label_rate_window(paths, label),
            "recommendation": recommendations[0] if recommendations else None,
            "recent_runs": recent_runs,
            "recent_rerun_history": recent_rerun_history,
            "active_rerun_requests": active_reruns[:5],
        },
    )


def load_qualification_rerun_detail(
    paths: QualificationArtifactsPaths,
    request_id: str,
) -> QualificationRerunDetailResponse:
    request_entry, completion_entry = find_rerun_history_by_request_id(paths, request_id)
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
        run_index = {item.run_id: item.model_dump(mode="json") for item in load_validation_runs(paths, limit=200).items}
        for run_id in [str(item) for item in completion_entry.get("generated_run_ids", []) if str(item)]:
            matched = run_index.get(run_id)
            if matched is not None:
                generated_runs.append(matched)
            run_detail = load_validation_run_detail(paths, run_id)
            if not run_detail.found:
                continue
            run = run_detail.run
            generated_run_details[run_id] = run
            if matched is None:
                direct_checks = run.get("direct_checks", {}) or {}
                agent_checks = run.get("agent_checks", []) or []
                generated_runs.append(
                    {
                        "run_id": run.get("run_id"),
                        "started_at": run.get("started_at"),
                        "finished_at": run.get("finished_at"),
                        "model": run.get("model"),
                        "direct_successes": sum(1 for payload in direct_checks.values() if direct_check_success(payload)),
                        "direct_total": len(direct_checks),
                        "agent_successes": sum(1 for item in agent_checks if agent_check_success(item)),
                        "agent_total": len(agent_checks),
                        "duration_seconds": run.get("duration_seconds", 0.0),
                        "report_path": run.get("report_path"),
                        "report_markdown_path": run.get("report_markdown_path"),
                        "aborted": bool(run.get("aborted", False)),
                        "failed_labels": [str(item.get("label", "")) for item in agent_checks if not agent_check_success(item)],
                    }
                )
    if latest_run_id:
        label_hint = labels[0] if len(labels) == 1 else None
        run_detail = load_validation_run_detail(paths, latest_run_id, label=label_hint)
        latest_run_detail = run_detail.run if run_detail.found else None

    label_outcomes: List[Dict[str, Any]] = []
    latest_checks = latest_run_detail.get("agent_checks", []) if latest_run_detail else []
    for label in labels:
        matching_check = next((item for item in latest_checks if str(item.get("label", "")) == label), None)
        label_outcomes.append(
            {
                "label": label,
                "latest_run_id": latest_run_id,
                "latest_success": agent_check_success(matching_check) if matching_check is not None else None,
                "latest_outcome": matching_check.get("outcome") if matching_check is not None else None,
                "latest_response": matching_check.get("response") if matching_check is not None else None,
                "comparison": label_run_comparison(paths, label),
                "rate_window": label_rate_window(paths, label),
            }
        )

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
        first_success = agent_check_success(first_check) if first_check is not None else None
        latest_success = agent_check_success(latest_check) if latest_check is not None else None
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
        request_progress.append(
            {
                "label": label,
                "first_run_id": first_run_id,
                "latest_run_id": latest_run_id,
                "first_success": first_success,
                "latest_success": latest_success,
                "first_outcome": first_check.get("outcome") if first_check is not None else None,
                "latest_outcome": latest_check.get("outcome") if latest_check is not None else None,
                "trend": trend,
            }
        )

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


def build_qualification_rerun_command(
    paths: QualificationArtifactsPaths,
    payload: QualificationRerunRequest,
    *,
    request_id: str,
) -> str:
    args = [
        sys.executable,
        str(paths.repo_root / "scripts" / "run_qualification_cycle.py"),
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
        str(paths.rerun_history_path),
    ]
    if payload.include_direct_checks:
        args.append("--include-direct-checks")
    return " ".join(shlex.quote(arg) for arg in args)
