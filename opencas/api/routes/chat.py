"""Chat API routes for the OpenCAS dashboard."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from .identity import SomaticStateResponse
from pydantic import BaseModel, Field
from pathlib import Path

from opencas.api.chat_service import (
    chat_upload_dir,
    perform_chat_turn,
    store_uploaded_file,
)
from opencas.bootstrap.task_beacon import (
    build_task_beacon,
    public_task_beacon_payload,
    runtime_task_beacon_fragments,
)
from opencas.bootstrap.live_objective import read_tasklist_live_objective
from opencas.api.voice_service import (
    VoiceSynthesisResult,
    VoiceTranscriptionResult,
    synthesize_speech,
    transcribe_audio,
    voice_status,
)
from opencas.runtime.lane_metadata import build_runtime_lane_meta

router = APIRouter(tags=["chat"])


class SessionListResponse(BaseModel):
    sessions: List[Dict[str, Any]]


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]


class SessionTracesResponse(BaseModel):
    session_id: str
    traces: List[Dict[str, Any]]


class ActivePlanResponse(BaseModel):
    active_plan: Optional[Dict[str, Any]]


class ChatSendResponse(BaseModel):
    session_id: str
    response: str
    somatic: Optional[SomaticStateResponse] = None
    voice_output: Optional[Dict[str, Any]] = None


class UploadResponse(BaseModel):
    filename: str
    path: str
    url: str
    media_type: str
    size_bytes: int


class ChatAttachmentInput(BaseModel):
    filename: str
    url: Optional[str] = None
    path: Optional[str] = None
    media_type: Optional[str] = None


class ChatVoiceInput(BaseModel):
    provider: str
    mode: str
    model: str
    warning: Optional[str] = None
    audio: Dict[str, Any]


class ChatVoiceSynthesisRequest(BaseModel):
    text: str
    prefer_local: bool = True
    expressive: bool = False


class ChatVoiceOutputResponse(BaseModel):
    provider: str
    mode: str
    model: str
    expressive: bool = False
    voice_id: Optional[str] = None
    voice_name: Optional[str] = None
    warning: Optional[str] = None
    filename: str
    url: str
    media_type: str
    size_bytes: int
    path: Optional[str] = None


class ChatVoiceStatusResponse(BaseModel):
    elevenlabs_available: bool
    local_stt_available: bool
    local_tts_available: bool
    elevenlabs_voice_id: str
    local_voice_name: str
    local_voice_resolved: str
    expressive_supported: bool


class ChatVoiceTranscriptionResponse(BaseModel):
    transcript: str
    voice_input: ChatVoiceInput


class ChatSendRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = ""
    attachments: List[ChatAttachmentInput] = Field(default_factory=list)
    voice_input: Optional[ChatVoiceInput] = None
    speak_response: bool = False
    voice_prefer_local: bool = True
    voice_expressive: bool = False


class CreateSessionResponse(BaseModel):
    session_id: str
    created: bool = True


class UpdateSessionRequest(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None  # "active" or "archived"


class UpdateSessionResponse(BaseModel):
    session_id: str
    updated: bool = True


class ChatContextSummaryResponse(BaseModel):
    session_id: str
    somatic: Optional[Dict[str, Any]] = None
    lane: Dict[str, Any] = Field(default_factory=dict)
    last_lane: Dict[str, Any] = Field(default_factory=dict)
    executive: Dict[str, Any] = Field(default_factory=dict)
    current_work: Optional[Dict[str, Any]] = None
    tasks: Dict[str, Any] = Field(default_factory=dict)
    task_beacon: Dict[str, Any] = Field(default_factory=dict)
    consolidation: Dict[str, Any] = Field(default_factory=dict)


_CURRENT_WORK_STAGES = {"micro_task", "project_seed", "project"}
_TASKLIST_ENTRY_RE = re.compile(r"^- `(?:PR|TASK)-[A-Z0-9-]+` (?P<title>.+)$")
_TASKLIST_STALE_SECTIONS = {
    "Recently Completed",
    "Completed 2026-04-15 Continuation Slices",
    "Additional Completed Readiness Slices",
    "Earlier Completed Readiness And Capability Slices",
    "Archived Completions",
}


def _human_title(text: Optional[str], fallback: str = "Untitled") -> str:
    raw = str(text or "").strip()
    if not raw:
        return fallback
    compact = " ".join(raw.splitlines()[0].split())
    return compact if len(compact) <= 88 else compact[:85].rstrip() + "..."


def _task_ui_status(stage: str, status: str) -> str:
    stage_key = str(stage or "").strip().lower()
    status_key = str(status or "").strip().lower()
    if stage_key == "done" or status_key in {"completed", "success"}:
        return "completed"
    if stage_key == "failed" or status_key in {"failed", "error"}:
        return "failed"
    if stage_key == "needs_approval":
        return "needs approval"
    if stage_key == "needs_clarification":
        return "needs clarification"
    if stage_key:
        return stage_key.replace("_", " ")
    return status_key.replace("_", " ") if status_key else "unknown"


def _is_current_work_candidate(item: Dict[str, Any]) -> bool:
    stage = str(item.get("stage") or "").strip().lower()
    return stage in _CURRENT_WORK_STAGES


def _current_work_from_items(work_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for item in work_items:
        if not _is_current_work_candidate(item):
            continue
        return {
            "work_id": item.get("work_id"),
            "title": _human_title(item.get("meta", {}).get("title") or item.get("content"), "Current work"),
            "stage": item.get("stage"),
            "project_id": item.get("project_id"),
            "blocked_by": item.get("blocked_by") or [],
        }
    return None


def _current_work_from_queue_items(queue_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for item in queue_items:
        state = str(item.get("state") or "").strip().lower()
        bearing = str(item.get("bearing") or "").strip().lower()
        is_active = bool(item.get("is_active"))
        if not is_active and state != "active":
            continue
        title = item.get("title") or item.get("content") or item.get("objective")
        return {
            "work_id": item.get("work_id"),
            "title": _human_title(title, "Current work"),
            "stage": item.get("stage"),
            "project_id": item.get("project_id"),
            "blocked_by": item.get("blocked_by") or [],
            "bearing": bearing or None,
            "source": "executive_queue",
        }
    return None


def _runtime_intention_source(runtime: Any) -> Optional[str]:
    executive = getattr(runtime, "executive", None) or getattr(getattr(runtime, "ctx", None), "executive", None)
    return getattr(executive, "intention_source", None)


def _normalize_tasklist_title(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _tasklist_section_for_title(workspace_root: Any, title: Any) -> Optional[str]:
    normalized_title = _normalize_tasklist_title(title)
    if not normalized_title or workspace_root is None:
        return None
    tasklist_path = Path(workspace_root) / "TaskList.md"
    if not tasklist_path.exists():
        return None
    section = ""
    try:
        lines = tasklist_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for raw_line in lines:
        line = raw_line.rstrip()
        if line.startswith("## "):
            section = line.strip().lstrip("#").strip()
            continue
        match = _TASKLIST_ENTRY_RE.match(line.strip())
        if match and _normalize_tasklist_title(match.group("title")) == normalized_title:
            return section or None
    return None


def _normalize_session_status(status: str) -> str:
    normalized = str(status or "").strip().lower() or "active"
    if normalized not in {"active", "archived", "all"}:
        raise HTTPException(status_code=400, detail=f"Unsupported session status: {status}")
    return normalized


def build_chat_router(runtime: Any) -> APIRouter:
    """Build chat routes wired to *runtime*."""
    import uuid

    from opencas.context.models import MessageRole

    r = APIRouter(prefix="/api/chat", tags=["chat"])
    upload_dir = chat_upload_dir(runtime)

    @r.get("/sessions", response_model=SessionListResponse)
    async def list_sessions(
        limit: int = 20,
        status: str = "active",
        q: Optional[str] = None,
    ) -> SessionListResponse:
        normalized_status = _normalize_session_status(status)
        if q and hasattr(runtime.ctx.context_store, "search_sessions"):
            sessions = await runtime.ctx.context_store.search_sessions(
                q,
                status=normalized_status,
                limit=limit,
            )
        else:
            sessions = await runtime.ctx.context_store.list_session_ids(
                limit=limit,
                status=normalized_status,
            )
        return SessionListResponse(sessions=sessions)

    @r.post("/sessions", response_model=CreateSessionResponse)
    async def create_session() -> CreateSessionResponse:
        sid = str(uuid.uuid4())
        # Pre-register the session so it appears in the list immediately
        await runtime.ctx.context_store.ensure_session(sid)
        return CreateSessionResponse(session_id=sid)

    @r.patch("/sessions/{session_id}", response_model=UpdateSessionResponse)
    async def update_session(session_id: str, req: UpdateSessionRequest) -> UpdateSessionResponse:
        if req.name is not None:
            await runtime.ctx.context_store.update_session_name(session_id, req.name)
        if req.status is not None:
            await runtime.ctx.context_store.set_session_status(session_id, _normalize_session_status(req.status))
        return UpdateSessionResponse(session_id=session_id)

    @r.post("/sessions/{session_id}/archive", response_model=UpdateSessionResponse)
    async def archive_session(session_id: str) -> UpdateSessionResponse:
        await runtime.ctx.context_store.set_session_status(session_id, "archived")
        return UpdateSessionResponse(session_id=session_id)

    @r.post("/sessions/{session_id}/unarchive", response_model=UpdateSessionResponse)
    async def unarchive_session(session_id: str) -> UpdateSessionResponse:
        await runtime.ctx.context_store.set_session_status(session_id, "active")
        return UpdateSessionResponse(session_id=session_id)

    @r.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
    async def get_session_history(session_id: str, limit: int = 100) -> SessionHistoryResponse:
        entries = await runtime.ctx.context_store.list_recent(session_id, limit=limit)
        return SessionHistoryResponse(
            session_id=session_id,
            messages=[
                {
                    "message_id": str(e.message_id),
                    "role": e.role.value,
                    "content": e.content,
                    "created_at": e.created_at.isoformat(),
                    "meta": e.meta,
                }
                for e in entries
            ],
        )

    @r.get("/sessions/{session_id}/traces", response_model=SessionTracesResponse)
    async def get_session_traces(session_id: str, limit: int = 100) -> SessionTracesResponse:
        events = runtime.tracer.store.query(session_id=session_id, limit=limit)
        return SessionTracesResponse(
            session_id=session_id,
            traces=[
                {
                    "timestamp": e.timestamp.isoformat(),
                    "kind": e.kind.value,
                    "message": e.message,
                    "payload": e.payload,
                    "span_id": e.span_id,
                }
                for e in events
            ],
        )

    @r.get("/plan", response_model=ActivePlanResponse)
    async def get_active_plan() -> ActivePlanResponse:
        plan_store = getattr(runtime.ctx, "plan_store", None)
        if plan_store is None:
            return ActivePlanResponse(active_plan=None)
        try:
            active = await plan_store.list_active()
            if active:
                return ActivePlanResponse(active_plan=active[0].model_dump(mode="json"))
        except Exception:
            pass
        return ActivePlanResponse(active_plan=None)

    @r.get("/context-summary", response_model=ChatContextSummaryResponse)
    async def get_context_summary(
        session_id: Optional[str] = None,
        task_limit: int = 6,
    ) -> ChatContextSummaryResponse:
        sid = session_id or runtime.ctx.config.session_id or "default"
        somatic = None
        somatic_mgr = getattr(runtime.ctx, "somatic", None)
        if somatic_mgr is not None:
            try:
                ss = somatic_mgr.state
                somatic = {
                    "somatic_tag": ss.somatic_tag,
                    "arousal": ss.arousal,
                    "energy": ss.energy,
                    "focus": ss.focus,
                    "fatigue": ss.fatigue,
                    "tension": ss.tension,
                    "valence": ss.valence,
                    "certainty": ss.certainty,
                    "updated_at": ss.updated_at.isoformat(),
                }
            except Exception:
                somatic = None

        lane: Dict[str, Any] = build_runtime_lane_meta(runtime, prefer_current=False)
        last_lane: Dict[str, Any] = build_runtime_lane_meta(runtime, prefer_current=True)
        if last_lane == lane:
            last_lane = {}

        workflow = await runtime.workflow_status(limit=task_limit)
        workspace_root = getattr(getattr(runtime.ctx, "config", None), "workspace_root", None)
        task_beacon = public_task_beacon_payload(
            build_task_beacon(
                workspace_root,
                limit_per_state=1,
                live_fragments=runtime_task_beacon_fragments(runtime),
            ),
            include_details=True,
            include_items=True,
        )
        work_items = workflow.get("work", {}).get("items", []) or []
        current_work = _current_work_from_items(work_items)
        executive_payload = workflow.get("executive", {}) or {}
        effective_intention = executive_payload.get("intention")
        intention_source = _runtime_intention_source(runtime) or "workflow"
        tasklist_live_objective = read_tasklist_live_objective(workspace_root)
        if current_work is None:
            current_work = _current_work_from_queue_items(
                ((executive_payload.get("queue") or {}).get("items") or [])
            )
            if current_work is not None:
                effective_intention = current_work["title"]
                intention_source = "active_queue"
            elif tasklist_live_objective:
                effective_intention = tasklist_live_objective
                intention_source = "tasklist_live_objective"
            elif _tasklist_section_for_title(workspace_root, effective_intention) in _TASKLIST_STALE_SECTIONS:
                effective_intention = None
                intention_source = "stale_tasklist_completed"

        task_entries = []
        task_counts = {"active": 0, "waiting": 0, "completed": 0, "failed": 0, "total": 0}
        task_store = getattr(runtime.ctx, "tasks", None)
        if task_store is not None:
            tasks = await task_store.list_all(limit=max(task_limit, 50))
            task_counts["total"] = len(tasks)
            objective_counts: Dict[str, int] = {}
            for task in tasks:
                objective_counts[task.objective] = objective_counts.get(task.objective, 0) + 1
            for task in tasks:
                ui_status = _task_ui_status(
                    task.stage.value if hasattr(task.stage, "value") else str(task.stage),
                    task.status,
                )
                if ui_status in {"queued", "planning", "executing", "verifying", "recovering"}:
                    task_counts["active"] += 1
                elif ui_status in {"needs approval", "needs clarification"}:
                    task_counts["waiting"] += 1
                elif ui_status == "completed":
                    task_counts["completed"] += 1
                elif ui_status == "failed":
                    task_counts["failed"] += 1
            for task in tasks[:task_limit]:
                task_entries.append(
                    {
                        "task_id": str(task.task_id),
                        "title": _human_title(task.meta.get("title") or task.objective, "Background task"),
                        "status": _task_ui_status(
                            task.stage.value if hasattr(task.stage, "value") else str(task.stage),
                            task.status,
                        ),
                        "stage": task.stage.value if hasattr(task.stage, "value") else str(task.stage),
                        "source": task.meta.get("source"),
                        "updated_at": task.updated_at.isoformat(),
                        "duplicate_objective_count": objective_counts.get(task.objective, 1),
                    }
                )

        return ChatContextSummaryResponse(
            session_id=sid,
            somatic=somatic,
            lane=lane,
            last_lane=last_lane,
            executive={
                "intention": effective_intention,
                "intention_source": intention_source,
                "active_goals": executive_payload.get("active_goals", []),
                "recommend_pause": executive_payload.get("recommend_pause", False),
                "queued_work_count": executive_payload.get("queued_work_count", 0),
                "capacity_remaining": executive_payload.get("capacity_remaining", 0),
            },
            current_work=current_work,
            tasks={
                "counts": task_counts,
                "items": task_entries,
            },
            task_beacon=task_beacon,
            consolidation=workflow.get("consolidation", {}),
        )

    @r.post("/send", response_model=ChatSendResponse)
    async def send_message(req: ChatSendRequest) -> ChatSendResponse:
        result = await perform_chat_turn(
            runtime,
            session_id=req.session_id,
            message=req.message,
            attachments=req.attachments,
            voice_input=req.voice_input.model_dump(mode="json") if req.voice_input else None,
            speak_response=req.speak_response,
            voice_prefer_local=req.voice_prefer_local,
            voice_expressive=req.voice_expressive,
        )
        return ChatSendResponse(
            session_id=result.session_id,
            response=result.response,
            somatic=result.somatic,
            voice_output=result.voice_output,
        )

    @r.get("/voice/status", response_model=ChatVoiceStatusResponse)
    async def get_voice_status() -> ChatVoiceStatusResponse:
        return ChatVoiceStatusResponse(**voice_status().to_dict())

    @r.post("/voice/transcribe", response_model=ChatVoiceTranscriptionResponse)
    async def transcribe_voice(
        file: UploadFile = File(...),
        prefer_local: bool = Form(True),
        language_code: Optional[str] = Form(None),
    ) -> ChatVoiceTranscriptionResponse:
        audio_bytes = file.file.read()
        result = await transcribe_audio(
            upload_dir,
            audio_bytes=audio_bytes,
            filename=file.filename or f"voice_input_{uuid.uuid4().hex}.webm",
            media_type=file.content_type,
            prefer_local=prefer_local,
            language_code=language_code,
        )
        return ChatVoiceTranscriptionResponse(
            transcript=result.text,
            voice_input=ChatVoiceInput(**result.to_meta()),
        )

    @r.post("/voice/synthesize", response_model=ChatVoiceOutputResponse)
    async def synthesize_voice(req: ChatVoiceSynthesisRequest) -> ChatVoiceOutputResponse:
        result = await synthesize_speech(
            upload_dir,
            text=req.text,
            prefer_local=req.prefer_local,
            expressive=req.expressive,
        )
        return ChatVoiceOutputResponse(**result.to_meta())

    @r.post("/upload", response_model=UploadResponse)
    async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
        payload = store_uploaded_file(
            upload_dir,
            filename=file.filename,
            content_type=file.content_type,
            fileobj=file.file,
        )
        return UploadResponse(
            filename=payload["filename"],
            path=payload["path"],
            url=payload["url"],
            media_type=payload["media_type"],
            size_bytes=payload["size_bytes"],
        )

    from starlette.responses import FileResponse

    @r.get("/uploads/{filename}")
    async def serve_upload(filename: str) -> Any:
        target = upload_dir / filename
        try:
            target.resolve().relative_to(upload_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid filename")
        if not target.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(target)

    return r
