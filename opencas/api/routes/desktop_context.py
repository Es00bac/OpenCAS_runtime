"""Desktop-context API routes for the Body Double manager program."""

from __future__ import annotations

import inspect
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, ConfigDict


class DesktopContextCaptureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    force: bool = False


def _require_desktop_context(runtime: Any) -> Any:
    service = getattr(runtime, "desktop_context", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Desktop context service is not available")
    return service


def _desktop_context_plugin_disabled(runtime: Any) -> bool:
    ctx = getattr(runtime, "ctx", None)
    lifecycle = getattr(ctx, "plugin_lifecycle", None)
    is_tool_disabled = getattr(lifecycle, "is_tool_disabled", None)
    if not callable(is_tool_disabled):
        return False
    for tool_name in ("desktop_context_configure", "desktop_context_observe", "desktop_context_speak"):
        try:
            if is_tool_disabled(tool_name):
                return True
        except Exception:
            return False
    return False


def _requests_live_body_double_enable(updates: Dict[str, Any]) -> bool:
    return any(
        updates.get(key) is True
        for key in (
            "enabled",
            "media_commentary_mode_enabled",
            "proactive_video_commentary_enabled",
            "live_transcription_enabled",
        )
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def build_desktop_context_router(runtime: Any) -> APIRouter:
    """Build routes backed by the live DesktopContextService instance."""

    router = APIRouter(prefix="/api/desktop-context", tags=["desktop-context"])

    @router.get("/status")
    async def status() -> Dict[str, Any]:
        service = _require_desktop_context(runtime)
        status_fn = getattr(service, "status", None)
        if not callable(status_fn):
            raise HTTPException(status_code=503, detail="Desktop context status is not available")
        snapshot = await _maybe_await(status_fn())
        return dict(snapshot or {})

    @router.patch("/config")
    async def configure(updates: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
        service = _require_desktop_context(runtime)
        clean_updates = dict(updates or {})
        if _desktop_context_plugin_disabled(runtime) and _requests_live_body_double_enable(clean_updates):
            raise HTTPException(
                status_code=409,
                detail="desktop_context plugin is disabled; enable the extension before turning Body Double on",
            )
        configure_fn = getattr(service, "configure", None)
        if not callable(configure_fn):
            raise HTTPException(status_code=503, detail="Desktop context configuration is not available")
        result = await _maybe_await(configure_fn(**clean_updates))
        status_fn = getattr(service, "status", None)
        snapshot = await _maybe_await(status_fn()) if callable(status_fn) else {}
        return {"ok": True, "result": result, "status": dict(snapshot or {})}

    @router.post("/capture")
    async def capture(
        request: DesktopContextCaptureRequest = Body(default_factory=DesktopContextCaptureRequest),
    ) -> Dict[str, Any]:
        service = _require_desktop_context(runtime)
        capture_fn = getattr(service, "capture_once", None)
        if not callable(capture_fn):
            raise HTTPException(status_code=503, detail="Desktop context capture is not available")
        result = await _maybe_await(capture_fn(force=request.force))
        status_fn = getattr(service, "status", None)
        snapshot = await _maybe_await(status_fn()) if callable(status_fn) else {}
        return {"ok": True, "result": result, "status": dict(snapshot or {})}

    return router
