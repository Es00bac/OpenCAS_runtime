"""Phone bridge routes for Twilio-backed voice access."""

from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import Response
from pydantic import BaseModel, Field

from opencas.phone_config import PhoneContactPolicy, PhoneRuntimeConfig


class PhoneConfigUpdateRequest(BaseModel):
    enabled: bool = False
    public_base_url: Optional[str] = None
    webhook_signature_required: bool = True
    webhook_secret: Optional[str] = None
    twilio_from_number: Optional[str] = None
    owner_phone_number: Optional[str] = None
    owner_display_name: str = "Operator"
    owner_workspace_subdir: str = "phone/owner"
    contacts: list[PhoneContactPolicy] = Field(default_factory=list)


class PhoneCallOwnerRequest(BaseModel):
    message: str = ""
    reason: str = ""


class PhoneAutoconfigureRequest(BaseModel):
    enabled: Optional[bool] = None
    public_base_url: Optional[str] = None
    webhook_signature_required: Optional[bool] = None
    webhook_secret: Optional[str] = None
    twilio_from_number: Optional[str] = None
    owner_phone_number: Optional[str] = None
    owner_display_name: Optional[str] = None
    owner_workspace_subdir: Optional[str] = None


def _configured_public_base_url(runtime: Any) -> Optional[str]:
    settings = getattr(runtime, "phone_settings", None)
    current = settings() if callable(settings) else settings
    if current is not None:
        public_base_url = getattr(current, "public_base_url", None)
        if public_base_url:
            return str(public_base_url).rstrip("/")
    return None


def _external_request_url(runtime: Any, request: Request) -> str:
    public_base_url = _configured_public_base_url(runtime)
    if not public_base_url:
        return str(request.url)

    split = urlsplit(str(request.url))
    query = urlencode(parse_qsl(split.query, keep_blank_values=True), doseq=True)
    return urlunsplit(
        (
            urlsplit(public_base_url).scheme,
            urlsplit(public_base_url).netloc,
            split.path,
            query,
            "",
        )
    )


def _external_base_url(runtime: Any, request: Request) -> str:
    public_base_url = _configured_public_base_url(runtime)
    if public_base_url:
        return public_base_url
    return str(request.base_url).rstrip("/")


def _external_websocket_url(runtime: Any, websocket: WebSocket) -> str:
    public_base_url = _configured_public_base_url(runtime)
    if not public_base_url:
        return str(websocket.url)

    split = urlsplit(str(websocket.url))
    public_split = urlsplit(public_base_url)
    scheme = "wss" if public_split.scheme == "https" else "ws"
    query = urlencode(parse_qsl(split.query, keep_blank_values=True), doseq=True)
    return urlunsplit((scheme, public_split.netloc, split.path, query, ""))


def build_phone_router(runtime: Any) -> APIRouter:
    """Build phone-bridge routes wired to *runtime*."""

    r = APIRouter(prefix="/api/phone", tags=["phone"])

    @r.get("/status")
    async def get_status() -> Dict[str, Any]:
        getter = getattr(runtime, "phone_status", None)
        if not callable(getter):
            raise HTTPException(status_code=503, detail="Phone bridge is not available")
        return await getter()

    @r.post("/config")
    async def update_config(req: PhoneConfigUpdateRequest) -> Dict[str, Any]:
        configurer = getattr(runtime, "configure_phone", None)
        if not callable(configurer):
            raise HTTPException(status_code=503, detail="Phone bridge is not available")
        settings = PhoneRuntimeConfig(
            enabled=req.enabled,
            public_base_url=req.public_base_url,
            webhook_signature_required=req.webhook_signature_required,
            webhook_secret=req.webhook_secret,
            twilio_from_number=req.twilio_from_number,
            owner_phone_number=req.owner_phone_number,
            owner_display_name=req.owner_display_name,
            owner_workspace_subdir=req.owner_workspace_subdir,
            contacts=req.contacts,
        )
        return await configurer(settings)

    @r.post("/autoconfigure")
    async def autoconfigure(req: PhoneAutoconfigureRequest) -> Dict[str, Any]:
        autoconfigure_phone = getattr(runtime, "autoconfigure_phone", None)
        if not callable(autoconfigure_phone):
            raise HTTPException(status_code=503, detail="Phone bridge is not available")
        return await autoconfigure_phone(
            enabled=req.enabled,
            public_base_url=req.public_base_url,
            webhook_signature_required=req.webhook_signature_required,
            webhook_secret=req.webhook_secret,
            twilio_from_number=req.twilio_from_number,
            owner_phone_number=req.owner_phone_number,
            owner_display_name=req.owner_display_name,
            owner_workspace_subdir=req.owner_workspace_subdir,
        )

    @r.post("/call-owner")
    async def call_owner(req: PhoneCallOwnerRequest) -> Dict[str, Any]:
        caller = getattr(runtime, "call_owner_via_phone", None)
        if not callable(caller):
            raise HTTPException(status_code=503, detail="Phone bridge is not available")
        return await caller(message=req.message, reason=req.reason)

    @r.post("/twilio/voice")
    async def twilio_voice(request: Request) -> Response:
        handler = getattr(runtime, "handle_phone_voice_webhook", None)
        if not callable(handler):
            raise HTTPException(status_code=503, detail="Phone bridge is not available")
        form = await request.form()
        xml = await handler(
            request_url=_external_request_url(runtime, request),
            webhook_base_url=_external_base_url(runtime, request),
            form_data=form,
            provided_signature=request.headers.get("X-Twilio-Signature"),
            call_token=request.query_params.get("call_token"),
            bridge_token=request.query_params.get("bridge_token"),
        )
        return Response(content=xml, media_type="text/xml")

    @r.post("/twilio/gather")
    async def twilio_gather(request: Request) -> Response:
        handler = getattr(runtime, "handle_phone_gather_webhook", None)
        if not callable(handler):
            raise HTTPException(status_code=503, detail="Phone bridge is not available")
        form = await request.form()
        xml = await handler(
            request_url=_external_request_url(runtime, request),
            webhook_base_url=_external_base_url(runtime, request),
            form_data=form,
            provided_signature=request.headers.get("X-Twilio-Signature"),
            call_token=request.query_params.get("call_token"),
            bridge_token=request.query_params.get("bridge_token"),
        )
        return Response(content=xml, media_type="text/xml")

    @r.post("/twilio/poll")
    async def twilio_poll(request: Request) -> Response:
        handler = getattr(runtime, "handle_phone_poll_webhook", None)
        if not callable(handler):
            raise HTTPException(status_code=503, detail="Phone bridge is not available")
        form = await request.form()
        xml = await handler(
            request_url=_external_request_url(runtime, request),
            webhook_base_url=_external_base_url(runtime, request),
            form_data=form,
            provided_signature=request.headers.get("X-Twilio-Signature"),
            call_token=request.query_params.get("call_token"),
            bridge_token=request.query_params.get("bridge_token"),
            reply_token=request.query_params.get("reply_token"),
        )
        return Response(content=xml, media_type="text/xml")

    @r.websocket("/twilio/media/{stream_secret}")
    async def twilio_media(websocket: WebSocket, stream_secret: str) -> None:
        handler = getattr(runtime, "handle_phone_media_stream", None)
        if not callable(handler):
            await websocket.close(code=1013)
            return
        await handler(
            websocket=websocket,
            request_url=_external_websocket_url(runtime, websocket),
            provided_signature=websocket.headers.get("x-twilio-signature"),
            stream_secret=stream_secret,
        )

    return r
