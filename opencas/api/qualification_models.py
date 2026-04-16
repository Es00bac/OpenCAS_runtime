"""Shared models and path descriptors for qualification artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


@dataclass(frozen=True)
class QualificationArtifactsPaths:
    repo_root: Path
    summary_path: Path
    remediation_path: Path
    validation_runs_dir: Path
    rerun_history_path: Path
