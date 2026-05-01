"""Shared request and response models for operations routes."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from opencas.autonomy.commitment import CommitmentStatus
from opencas.autonomy.models import WorkStage


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
    retry_blocked: bool = False
    loop_stop_cause: Optional[str] = None
    latest_meaningful_signal: Optional[str] = None
    latest_artifact: Optional[str] = None
    latest_evidence: Optional[str] = None
    latest_blocker: Optional[str] = None


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
    lifecycle: Dict[str, Any] = Field(default_factory=dict)


class CommitmentListResponse(BaseModel):
    count: int
    items: List[CommitmentEntry]
    summary: Dict[str, Any] = Field(default_factory=dict)


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


class ProcessDetailResponse(BaseModel):
    found: bool
    process: Dict[str, Any] = Field(default_factory=dict)
