"""Phone bridge routes for Twilio-backed voice access."""

from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import Response
from pydantic import BaseModel, Field

from opencas.phone_config import PhoneAllowedAction, PhoneContactPolicy, PhoneRuntimeConfig


class PhoneConfigUpdateRequest(BaseModel):
    enabled: bool = False
    public_base_url: Optional[str] = None
    webhook_signature_required: bool = True
    webhook_secret: Optional[str] = None
    menu_config_path: Optional[str] = None
    twilio_env_path: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    twilio_api_key_sid: Optional[str] = None
    twilio_api_secret: Optional[str] = None
    clear_twilio_api_secret: bool = False
    twilio_auth_token: Optional[str] = None
    clear_twilio_auth_token: bool = False
    twilio_from_number: Optional[str] = None
    owner_phone_number: Optional[str] = None
    owner_display_name: str = "Operator"
    owner_workspace_subdir: str = "phone/owner"
    owner_pin: Optional[str] = None
    clear_owner_pin: bool = False
    elevenlabs_env_path: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    clear_elevenlabs_api_key: bool = False
    elevenlabs_voice_id: Optional[str] = None
    elevenlabs_stt_model: str = "scribe_v2"
    elevenlabs_fast_model: str = "eleven_flash_v2_5"
    elevenlabs_expressive_model: str = "eleven_v3"
    edge_tts_voice: str = "Aira"
    phone_tts_mode: str = "fast"
    phone_min_utterance_bytes: int = 800
    phone_silence_gap_seconds: float = 1.0
    phone_speech_rms_threshold: int = 180
    phone_preroll_ms: int = 320
    contacts: list[PhoneContactPolicy] = Field(default_factory=list)


class PhoneCallOwnerRequest(BaseModel):
    message: str = ""
    reason: str = ""


class PhoneAutoconfigureRequest(BaseModel):
    enabled: Optional[bool] = None
    public_base_url: Optional[str] = None
    webhook_signature_required: Optional[bool] = None
    webhook_secret: Optional[str] = None
    twilio_env_path: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    twilio_api_key_sid: Optional[str] = None
    twilio_api_secret: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_from_number: Optional[str] = None
    owner_phone_number: Optional[str] = None
    owner_display_name: Optional[str] = None
    owner_workspace_subdir: Optional[str] = None


class PhoneSessionProfilesUpdateRequest(BaseModel):
    owner_entry_prompt: str = ""
    owner_entry_reprompt: str = ""
    owner_continue_digit: str = "1"
    owner_fallback_digit: str = "2"
    owner_pin_prompt: str = ""
    owner_pin_retry_prompt: str = ""
    owner_pin_success_message: str = ""
    owner_pin_failure_message: str = ""
    public_prompt: str = ""
    public_reprompt: str = ""
    employer_enabled: bool = True
    employer_digit: str = "1"
    employer_label: str = "Potential employer"
    employer_phrases: list[str] = Field(default_factory=list)
    employer_greeting: str = ""
    employer_prompt_profile: str = "worksafe_owner"
    employer_allowed_actions: list[PhoneAllowedAction] = Field(
        default_factory=lambda: ["leave_message", "knowledge_qa"]
    )
    employer_shared_workspace_subdir: str = "phone/employer_shared"
    employer_caller_workspace_subdir: str = "phone/employers/{phone_digits}"
    reject_enabled: bool = True
    reject_digit: str = "2"
    reject_label: str = "Not for this line"
    reject_phrases: list[str] = Field(default_factory=list)
    reject_message: str = ""


class PhoneMenuConfigUpdateRequest(BaseModel):
    config: Dict[str, Any] = Field(default_factory=dict)


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


def _current_phone_settings(runtime: Any) -> PhoneRuntimeConfig:
    settings = getattr(runtime, "phone_settings", None)
    if callable(settings):
        current = settings()
        if isinstance(current, PhoneRuntimeConfig):
            return current
    elif isinstance(settings, PhoneRuntimeConfig):
        return settings
    existing = getattr(runtime, "_phone_config", None)
    if isinstance(existing, PhoneRuntimeConfig):
        return existing
    return PhoneRuntimeConfig()


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

    @r.get("/recent-calls")
    async def get_recent_calls(limit: int = 10) -> Dict[str, Any]:
        getter = getattr(runtime, "recent_phone_calls", None)
        if not callable(getter):
            raise HTTPException(status_code=503, detail="Phone bridge is not available")
        return await getter(limit=limit)

    @r.get("/recent-calls/{call_sid}")
    async def get_call_detail(call_sid: str) -> Dict[str, Any]:
        getter = getattr(runtime, "phone_call_detail", None)
        if not callable(getter):
            raise HTTPException(status_code=503, detail="Phone bridge is not available")
        detail = await getter(call_sid)
        if not detail.get("found"):
            raise HTTPException(status_code=404, detail=f"Phone call {call_sid} was not found")
        return detail

    @r.post("/config")
    async def update_config(req: PhoneConfigUpdateRequest) -> Dict[str, Any]:
        configurer = getattr(runtime, "configure_phone", None)
        if not callable(configurer):
            raise HTTPException(status_code=503, detail="Phone bridge is not available")
        current = _current_phone_settings(runtime)
        provided = set(req.model_fields_set)
        settings = PhoneRuntimeConfig(
            enabled=req.enabled,
            public_base_url=req.public_base_url,
            webhook_signature_required=req.webhook_signature_required,
            webhook_secret=req.webhook_secret if req.webhook_secret is not None else current.webhook_secret,
            menu_config_path=req.menu_config_path if "menu_config_path" in provided else current.menu_config_path,
            twilio_env_path=req.twilio_env_path if "twilio_env_path" in provided else current.twilio_env_path,
            twilio_account_sid=req.twilio_account_sid if "twilio_account_sid" in provided else current.twilio_account_sid,
            twilio_api_key_sid=req.twilio_api_key_sid if "twilio_api_key_sid" in provided else current.twilio_api_key_sid,
            twilio_api_secret=(
                None
                if req.clear_twilio_api_secret
                else req.twilio_api_secret
                if req.twilio_api_secret is not None
                else current.twilio_api_secret
            ),
            twilio_auth_token=(
                None
                if req.clear_twilio_auth_token
                else req.twilio_auth_token
                if req.twilio_auth_token is not None
                else current.twilio_auth_token
            ),
            twilio_from_number=req.twilio_from_number,
            owner_phone_number=req.owner_phone_number,
            owner_display_name=req.owner_display_name,
            owner_workspace_subdir=req.owner_workspace_subdir,
            owner_pin=(
                None
                if req.clear_owner_pin
                else req.owner_pin
                if req.owner_pin is not None
                else current.owner_pin
            ),
            elevenlabs_env_path=(
                req.elevenlabs_env_path if "elevenlabs_env_path" in provided else current.elevenlabs_env_path
            ),
            elevenlabs_api_key=(
                None
                if req.clear_elevenlabs_api_key
                else req.elevenlabs_api_key
                if req.elevenlabs_api_key is not None
                else current.elevenlabs_api_key
            ),
            elevenlabs_voice_id=(
                req.elevenlabs_voice_id if "elevenlabs_voice_id" in provided else current.elevenlabs_voice_id
            ),
            elevenlabs_stt_model=(
                req.elevenlabs_stt_model if "elevenlabs_stt_model" in provided else current.elevenlabs_stt_model
            ),
            elevenlabs_fast_model=(
                req.elevenlabs_fast_model if "elevenlabs_fast_model" in provided else current.elevenlabs_fast_model
            ),
            elevenlabs_expressive_model=(
                req.elevenlabs_expressive_model
                if "elevenlabs_expressive_model" in provided
                else current.elevenlabs_expressive_model
            ),
            edge_tts_voice=req.edge_tts_voice if "edge_tts_voice" in provided else current.edge_tts_voice,
            phone_tts_mode=req.phone_tts_mode if "phone_tts_mode" in provided else current.phone_tts_mode,
            phone_min_utterance_bytes=(
                req.phone_min_utterance_bytes
                if "phone_min_utterance_bytes" in provided
                else current.phone_min_utterance_bytes
            ),
            phone_silence_gap_seconds=(
                req.phone_silence_gap_seconds
                if "phone_silence_gap_seconds" in provided
                else current.phone_silence_gap_seconds
            ),
            phone_speech_rms_threshold=(
                req.phone_speech_rms_threshold
                if "phone_speech_rms_threshold" in provided
                else current.phone_speech_rms_threshold
            ),
            phone_preroll_ms=(
                req.phone_preroll_ms if "phone_preroll_ms" in provided else current.phone_preroll_ms
            ),
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
            twilio_env_path=req.twilio_env_path,
            twilio_account_sid=req.twilio_account_sid,
            twilio_api_key_sid=req.twilio_api_key_sid,
            twilio_api_secret=req.twilio_api_secret,
            twilio_auth_token=req.twilio_auth_token,
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

    @r.post("/session-profiles")
    async def update_session_profiles(req: PhoneSessionProfilesUpdateRequest) -> Dict[str, Any]:
        configurer = getattr(runtime, "configure_phone_session_profiles", None)
        if not callable(configurer):
            raise HTTPException(status_code=503, detail="Phone bridge is not available")
        return await configurer(req.model_dump(mode="json"))

    @r.post("/menu-config")
    async def update_menu_config(req: PhoneMenuConfigUpdateRequest) -> Dict[str, Any]:
        configurer = getattr(runtime, "configure_phone_menu_config", None)
        if not callable(configurer):
            raise HTTPException(status_code=503, detail="Phone bridge is not available")
        return await configurer(req.config)

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
