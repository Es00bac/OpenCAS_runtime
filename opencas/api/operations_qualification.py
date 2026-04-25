"""Shared qualification helpers for operations routes."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from opencas.api.qualification_models import (
    QualificationArtifactsPaths,
    QualificationLabelDetailResponse,
    QualificationRerunDetailResponse,
    QualificationRerunRequest,
    QualificationSummaryResponse,
    ValidationRunDetailResponse,
    ValidationRunListResponse,
)
from opencas.api.qualification_history import (
    append_qualification_rerun_history,
    find_rerun_history_by_request_id as _find_rerun_history_by_request_id_in_paths,
    load_recent_rerun_history,
)
from opencas.api.qualification_service import (
    annotate_recent_rerun_history,
    annotate_recommendation_runtime_state,
    build_qualification_rerun_command,
    load_qualification_label_detail,
    load_qualification_remediation,
    load_qualification_rerun_detail,
    load_qualification_summary,
    load_validation_run_detail,
    load_validation_runs,
)



def find_rerun_history_by_request_id(
    paths_provider: Callable[[], QualificationArtifactsPaths],
    request_id: Optional[str],
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    return _find_rerun_history_by_request_id_in_paths(paths_provider(), request_id)


class QualificationOperationsService:
    """Collect qualification route behavior behind one seam."""

    def __init__(
        self,
        runtime: Any,
        *,
        paths_provider: Callable[[], QualificationArtifactsPaths],
        repo_root: Path,
    ) -> None:
        self.runtime = runtime
        self._paths_provider = paths_provider
        self._repo_root = repo_root

    def get_summary(self) -> QualificationSummaryResponse:
        paths = self._paths_provider()
        summary = load_qualification_summary(paths)
        if hasattr(self.runtime, "process_supervisor"):
            snapshot = self.runtime.process_supervisor.snapshot(scope_key="qualification")
            summary.active_reruns = snapshot.get("entries", [])
        summary.recommended_reruns = annotate_recommendation_runtime_state(
            paths,
            summary.recommended_reruns,
            summary.active_reruns,
        )
        summary.recent_runs = [item.model_dump(mode="json") for item in load_validation_runs(paths, limit=5).items]
        summary.recent_rerun_history = annotate_recent_rerun_history(paths, load_recent_rerun_history(paths, limit=10))
        summary.remediation_rollup = load_qualification_remediation(paths)
        return summary

    def get_label_detail(self, label: str) -> QualificationLabelDetailResponse:
        detail = load_qualification_label_detail(self._paths_provider(), label)
        if detail.found and hasattr(self.runtime, "process_supervisor"):
            snapshot = self.runtime.process_supervisor.snapshot(scope_key="qualification")
            active = []
            for item in snapshot.get("entries", []):
                metadata = item.get("metadata", {}) or {}
                if str(metadata.get("source_label", "") or "") == label and item.get("running", False):
                    active.append(item)
            detail.detail["active_reruns"] = active
        return detail

    def get_rerun_detail(self, request_id: str) -> QualificationRerunDetailResponse:
        detail = load_qualification_rerun_detail(self._paths_provider(), request_id)
        if detail.found and hasattr(self.runtime, "process_supervisor"):
            snapshot = self.runtime.process_supervisor.snapshot(scope_key="qualification")
            active = []
            for item in snapshot.get("entries", []):
                metadata = item.get("metadata", {}) or {}
                if str(metadata.get("request_id", "") or "") == request_id and item.get("running", False):
                    active.append(item)
            detail.detail["active_processes"] = active
        return detail

    def list_validation_runs(self, *, limit: int = 10, label: Optional[str] = None) -> ValidationRunListResponse:
        return load_validation_runs(self._paths_provider(), limit=max(1, min(limit, 50)), label=label)

    def get_validation_run(self, run_id: str, *, label: Optional[str] = None) -> ValidationRunDetailResponse:
        return load_validation_run_detail(self._paths_provider(), run_id, label=label)

    def start_rerun(self, payload: QualificationRerunRequest) -> Dict[str, Any]:
        if not hasattr(self.runtime, "process_supervisor"):
            return {"ok": False, "error": "Process supervisor not available"}
        paths = self._paths_provider()
        request_id = uuid4().hex
        command = build_qualification_rerun_command(paths, payload, request_id=request_id)
        metadata = {
            "kind": "qualification_rerun",
            "request_id": request_id,
            "source_label": payload.source_label or payload.label,
            "source_note": payload.source_note or "",
            "requested_at": time.time(),
        }
        process_id = self.runtime.process_supervisor.start(
            "qualification",
            command,
            cwd=str(self._repo_root),
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
        append_qualification_rerun_history(paths, history_entry)
        return {
            "ok": True,
            "process_id": process_id,
            "scope_key": "qualification",
            "command": command,
            "metadata": metadata,
            "history_entry": history_entry,
        }
