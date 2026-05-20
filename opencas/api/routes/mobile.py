"""Pocket Relay mobile companion routes."""

from __future__ import annotations

from html import escape
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from opencas.mobile_bridge import (
    MobileBridgeAuthError,
    MobileBridgeValidationError,
    MobileDeviceRecord,
    mobile_bridge_service,
)


class MobilePairStartRequest(BaseModel):
    operator_label: str = ""
    device_label: str = ""


class MobilePairCompleteRequest(BaseModel):
    pairing_id: str
    pairing_secret: str
    device_label: str = "Android device"
    platform: str = "android"
    app_version: str = ""


class MobileEventsRequest(BaseModel):
    events: List[Dict[str, Any]] = Field(default_factory=list)


class MobileOperatorMessageRequest(BaseModel):
    message: str
    message_kind: str = "question"
    client_message_id: str = ""


class MobileApprovalResponseRequest(BaseModel):
    approval_id: str
    decision: str
    reason: str = ""


def _bearer_token(authorization: Optional[str]) -> str:
    raw = str(authorization or "").strip()
    if not raw.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Bearer authorization is required")
    token = raw[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Bearer authorization is required")
    return token


def _authenticated_device(
    runtime: Any,
    *,
    authorization: Optional[str],
    device_id: Optional[str],
) -> MobileDeviceRecord:
    service = mobile_bridge_service(runtime)
    try:
        return service.authenticate(
            device_id=str(device_id or "").strip(),
            token=_bearer_token(authorization),
        )
    except MobileBridgeAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _public_base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _qr_svg(payload: str) -> str | None:
    """Render a QR SVG with the system qrencode tool when available."""

    if not shutil.which("qrencode"):
        return None
    try:
        result = subprocess.run(
            ["qrencode", "-t", "SVG", "-o", "-", "--", payload],
            check=True,
            capture_output=True,
            text=True,
            timeout=4,
        )
    except Exception:
        return None
    svg = result.stdout.strip()
    if not svg.startswith("<?xml") and "<svg" not in svg[:200]:
        return None
    return svg


def _pairing_page_html(*, qr_payload: str, qr_svg: str | None, server: str, expires_at: str) -> str:
    qr_block = (
        qr_svg
        if qr_svg
        else "<p><strong>QR generator unavailable.</strong> Copy the pairing payload into the app instead.</p>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenCAS Pocket Relay Pairing</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 48rem; }}
    .qr {{ width: min(80vw, 360px); }}
    .qr svg {{ width: 100%; height: auto; }}
    code, textarea {{ width: 100%; font-family: ui-monospace, monospace; }}
    textarea {{ min-height: 7rem; }}
  </style>
</head>
<body>
  <h1>OpenCAS Pocket Relay Pairing</h1>
  <p>Scan this with the Android camera app. It should open OpenCAS Pocket Relay and complete pairing.</p>
  <div class="qr">{qr_block}</div>
  <p><strong>Server:</strong> <code>{escape(server)}</code></p>
  <p><strong>Expires:</strong> <code>{escape(expires_at)}</code></p>
  <p>If the camera does not open Pocket Relay, paste this into the app's pairing field:</p>
  <textarea readonly>{escape(qr_payload)}</textarea>
</body>
</html>"""


def build_mobile_router(runtime: Any) -> APIRouter:
    """Build routes for paired Android Pocket Relay clients."""

    router = APIRouter(prefix="/api/mobile", tags=["mobile"])

    @router.post("/pair/start")
    async def pair_start(req: MobilePairStartRequest, request: Request) -> Dict[str, Any]:
        return mobile_bridge_service(runtime).start_pairing(
            public_base_url=_public_base_url(request),
            operator_label=req.operator_label,
            device_label=req.device_label,
        )

    @router.get("/pair/new", response_class=HTMLResponse)
    async def pair_new(
        request: Request,
        server: Optional[str] = None,
        operator_label: str = "",
        device_label: str = "Android phone",
    ) -> HTMLResponse:
        public_base_url = str(server or "").strip().rstrip("/") or _public_base_url(request)
        payload = mobile_bridge_service(runtime).start_pairing(
            public_base_url=public_base_url,
            operator_label=operator_label,
            device_label=device_label,
        )
        return HTMLResponse(
            _pairing_page_html(
                qr_payload=payload["qr_payload"],
                qr_svg=_qr_svg(payload["qr_payload"]),
                server=public_base_url,
                expires_at=payload["expires_at"],
            )
        )

    @router.post("/pair/complete")
    async def pair_complete(req: MobilePairCompleteRequest) -> Dict[str, Any]:
        try:
            return mobile_bridge_service(runtime).complete_pairing(
                pairing_id=req.pairing_id,
                pairing_secret=req.pairing_secret,
                device_label=req.device_label,
                platform=req.platform,
                app_version=req.app_version,
            )
        except MobileBridgeAuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except MobileBridgeValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/events")
    async def post_events(
        req: MobileEventsRequest,
        authorization: Optional[str] = Header(default=None),
        x_opencas_device_id: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        device = _authenticated_device(
            runtime,
            authorization=authorization,
            device_id=x_opencas_device_id,
        )
        if not req.events:
            raise HTTPException(status_code=422, detail="events must include at least one envelope")
        try:
            return mobile_bridge_service(runtime).ingest_events(device, req.events)
        except MobileBridgeAuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except MobileBridgeValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/events/ack")
    async def get_event_acks(
        after_ack_id: int = 0,
        limit: int = 100,
        authorization: Optional[str] = Header(default=None),
        x_opencas_device_id: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        device = _authenticated_device(
            runtime,
            authorization=authorization,
            device_id=x_opencas_device_id,
        )
        return mobile_bridge_service(runtime).ack_snapshot(
            device,
            after_ack_id=after_ack_id,
            limit=limit,
        )

    @router.get("/state")
    async def get_state(
        authorization: Optional[str] = Header(default=None),
        x_opencas_device_id: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        device = _authenticated_device(
            runtime,
            authorization=authorization,
            device_id=x_opencas_device_id,
        )
        return mobile_bridge_service(runtime).state_snapshot(device)

    @router.post("/operator-message")
    async def post_operator_message(
        req: MobileOperatorMessageRequest,
        authorization: Optional[str] = Header(default=None),
        x_opencas_device_id: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        device = _authenticated_device(
            runtime,
            authorization=authorization,
            device_id=x_opencas_device_id,
        )
        try:
            return await mobile_bridge_service(runtime).operator_message(
                device,
                message=req.message,
                message_kind=req.message_kind,
                client_message_id=req.client_message_id,
            )
        except MobileBridgeValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/approval-response")
    async def post_approval_response(
        req: MobileApprovalResponseRequest,
        authorization: Optional[str] = Header(default=None),
        x_opencas_device_id: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        device = _authenticated_device(
            runtime,
            authorization=authorization,
            device_id=x_opencas_device_id,
        )
        try:
            return await mobile_bridge_service(runtime).approval_response(
                device,
                approval_id=req.approval_id,
                decision=req.decision,
                reason=req.reason,
            )
        except MobileBridgeValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router
