"""FastAPI server for OpenCAS continuous presence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from opencas.telemetry.models import TelemetryEvent
from .chat_service import perform_chat_turn
from .websocket_bridge import WebSocketBridge


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    response: str


def create_app(runtime: Any) -> FastAPI:
    """Build a FastAPI app wired to *runtime*."""
    app = FastAPI(title="OpenCAS")
    bridge = WebSocketBridge(runtime.ctx.event_bus)

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/readiness")
    async def readiness() -> Dict[str, Any]:
        if runtime.ctx.readiness:
            return runtime.ctx.readiness.snapshot()
        return {"state": "unknown"}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest) -> ChatResponse:
        try:
            result = await perform_chat_turn(
                runtime,
                session_id=body.session_id,
                message=body.message,
            )
            return ChatResponse(response=result.response)
        except Exception as exc:
            return ChatResponse(response=f"[Error: {exc}]")

    @app.get("/telemetry/recent")
    async def telemetry_recent(
        session_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        store = runtime.tracer.store
        events = store.query(session_id=session_id, limit=limit)
        return [_event_to_dict(e) for e in events]

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await bridge.connect(websocket)
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": "Invalid JSON"})
                    )
                    continue

                msg_type = msg.get("type")
                if msg_type == "chat":
                    payload = msg.get("payload", "")
                    sid = msg.get("session_id")
                    try:
                        result = await perform_chat_turn(
                            runtime,
                            session_id=sid,
                            message=payload,
                        )
                        response = result.response
                    except Exception as exc:
                        response = f"[Error: {exc}]"
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "chat_response",
                                "session_id": sid,
                                "payload": response,
                            }
                        )
                    )
                else:
                    await websocket.send_text(
                        json.dumps(
                            {"type": "error", "message": f"Unknown type: {msg_type}"}
                        )
                    )
        except WebSocketDisconnect:
            pass
        finally:
            bridge.disconnect(websocket)

    # Dashboard routers
    from .routes.config import build_config_router
    from .routes.monitor import build_monitor_router
    from .routes.platform import build_platform_router
    from .routes.chat import build_chat_router
    from .routes.daydream import build_daydream_router
    from .routes.memory import build_memory_router
    from .routes.operations import build_operations_router
    from .routes.usage import build_usage_router
    from .routes.identity import build_identity_router
    from .routes.executive import build_executive_router
    from .routes.phone import build_phone_router
    from .routes.telegram import build_telegram_router
    from .routes.schedule import build_schedule_router
    from .routes.telemetry import build_telemetry_router

    app.include_router(build_config_router(runtime))
    app.include_router(build_monitor_router(runtime))
    app.include_router(build_platform_router(runtime))
    app.include_router(build_chat_router(runtime))
    app.include_router(build_daydream_router(runtime))
    app.include_router(build_memory_router(runtime))
    app.include_router(build_operations_router(runtime))
    app.include_router(build_usage_router(runtime))
    app.include_router(build_identity_router(runtime))
    app.include_router(build_executive_router(runtime))
    app.include_router(build_phone_router(runtime))
    app.include_router(build_telegram_router(runtime))
    app.include_router(build_schedule_router(runtime))
    app.include_router(build_telemetry_router(runtime))

    # Static dashboard SPA
    dashboard_dir = Path(__file__).parent.parent / "dashboard" / "static"
    if dashboard_dir.exists():
        dashboard_static = StaticFiles(directory=str(dashboard_dir))
        app.mount("/dashboard/static", dashboard_static, name="dashboard-static")
        app.mount("/opencas/dashboard/static", StaticFiles(directory=str(dashboard_dir)), name="opencas-dashboard-static")

    @app.get("/dashboard")
    async def dashboard_root(request: Request) -> HTMLResponse:
        index = dashboard_dir / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text(encoding="utf-8"))
        return HTMLResponse((dashboard_dir / "index.html").read_text(encoding="utf-8"))

    @app.get("/opencas")
    @app.get("/opencas/")
    async def opencas_dashboard_root(request: Request) -> HTMLResponse:
        index = dashboard_dir / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text(encoding="utf-8"))
        return HTMLResponse((dashboard_dir / "index.html").read_text(encoding="utf-8"))

    @app.get("/")
    async def root(request: Request) -> HTMLResponse:
        index = dashboard_dir / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text(encoding="utf-8"))
        return HTMLResponse((dashboard_dir / "index.html").read_text(encoding="utf-8"))

    return app


def _event_to_dict(event: TelemetryEvent) -> Dict[str, Any]:
    return {
        "timestamp": event.timestamp.isoformat(),
        "kind": event.kind.value,
        "message": event.message,
        "payload": event.payload,
        "session_id": event.session_id,
        "span_id": event.span_id,
    }
