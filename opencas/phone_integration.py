"""Twilio-backed phone bridge for OpenCAS."""

from __future__ import annotations

import asyncio
from datetime import datetime
import hashlib
import inspect
import re
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

import httpx

try:
    from twilio.request_validator import RequestValidator
except ImportError:  # pragma: no cover - exercised when optional phone deps are absent
    class RequestValidator:  # type: ignore[no-redef]
        """Fallback validator used when the optional Twilio package is not installed."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def validate(self, *_args: Any, **_kwargs: Any) -> bool:
            return False

from opencas.api.chat_service import chat_upload_dir
from opencas.api.voice_service import synthesize_speech
from opencas.autonomy.models import ActionRiskTier
from opencas.context.models import MessageRole
from opencas.memory import EpisodeKind
from opencas.phone_config import (
    PhoneAllowedAction,
    PhoneMenuConfig,
    PhoneMenuDefinition,
    PhoneMenuOption,
    PhoneRuntimeConfig,
    PhoneWorkspaceMount,
    load_phone_menu_config,
    normalize_phone_number,
)
from opencas.phone_streaming import (
    PhoneMediaStreamSession,
    build_connect_stream_twiml,
    mulaw_to_wav_bytes,
)
from opencas.runtime.lane_metadata import build_assistant_message_meta
from opencas.tools import ToolRegistry, ToolUseContext, ToolUseLoop
from opencas.tools.adapters.add_only_write import AddOnlyFileWriteToolAdapter
from opencas.tools.adapters.fs import FileSystemToolAdapter
from opencas.tools.adapters.search import SearchToolAdapter

_TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".py",
    ".js",
    ".ts",
    ".toml",
}

_SCREENING_MENU_PROMPT = (
    "Hi, this is the OpenCAS phone bridge. Potential employers, press 1 or say employer. "
    "Everyone else, press 2."
)
_SCREENING_MENU_REPROMPT = (
    "Please press 1 or say employer if you're calling about work opportunities. "
    "Otherwise, press 2."
)
_SCREENING_REJECTION = (
    "Sorry, this line is reserved for employment inquiries. Please check the website for public information."
)
_EMPLOYER_MODE_GREETING = (
    "You're connected to the OpenCAS phone bridge in work mode. I can answer questions about the owner's approved resume, skills, "
    "and current projects, and I can take a message for follow-up."
)
_EMPLOYER_POLICY_NOTE = (
    "Employment-only phone mode. Discuss only approved work information in this workspace, such as the resume, "
    "skills, relevant projects, and employer follow-up requests. Do not discuss private life, criminal history, "
    "personal drama, medical information, finances, family matters, or anything not explicitly present in the approved workspace."
)
_EMPLOYER_WORKSPACE_ROOT = "phone/employers"
_EMPLOYER_SHARED_WORKSPACE = "phone/employer_shared"
_OWNER_EMPLOYER_SUMMARY_LOG = "phone/owner/employer-call-summaries.md"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_EMPLOYER_SHARED_SEED_DIR = _REPO_ROOT / "operator_seed" / "phone" / "employer_shared"
_PHONE_MENU_CONFIG_PATH = _REPO_ROOT / "operator_seed" / "phone" / "menu.json"
_PHONE_PROFILE_ROOT = _REPO_ROOT / "operator_seed" / "phone"
_PHONE_OWNER_STREAM_REPLY_TIMEOUT_SECONDS = 10.0
_PHONE_WORKSPACE_STREAM_REPLY_TIMEOUT_SECONDS = 8.0


@dataclass(frozen=True)
class TwilioCredentials:
    """Resolved Twilio REST credentials."""

    account_sid: str
    api_username: str
    api_password: str
    webhook_auth_token: Optional[str] = None


@dataclass(frozen=True)
class PhoneResolvedCaller:
    """Resolved trust and workspace policy for one phone number."""

    phone_number: str
    display_name: str
    trust_level: str
    allowed_actions: tuple[PhoneAllowedAction, ...] = ()
    workspace_subdir: Optional[str] = None
    notes: str = ""
    is_owner: bool = False
    menu_option_key: Optional[str] = None
    prompt_profile: Optional[str] = None
    workspace_mounts: tuple[PhoneWorkspaceMount, ...] = ()

    @property
    def allows_knowledge_qa(self) -> bool:
        return self.is_owner or "knowledge_qa" in self.allowed_actions

    @property
    def allows_voicemail(self) -> bool:
        return self.is_owner or "leave_message" in self.allowed_actions


@dataclass
class OutboundCallContext:
    """Ephemeral state for one outbound phone call."""

    token: str
    created_at: float
    intro_message: str
    reason: str = ""
    consumed: bool = False


@dataclass
class PendingPhoneReply:
    """Background owner-line reply generation tracked across Twilio polls."""

    token: str
    created_at: float
    call_sid: str
    caller_number: str
    task: asyncio.Task["ResolvedPhoneReply"]


@dataclass(frozen=True)
class ResolvedPhoneReply:
    """Prepared phone reply content ready for immediate TwiML emission."""

    text: str
    prompt_verb: str


class _LowTrustToolRuntime:
    """Minimal runtime proxy for caller-scoped tool execution."""

    def __init__(self, base_runtime: Any, tools: ToolRegistry) -> None:
        self._base_runtime = base_runtime
        self.tools = tools
        self.ctx = base_runtime.ctx
        self.llm = base_runtime.llm
        self.tracer = getattr(base_runtime, "tracer", None)

    async def execute_tool(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> Dict[str, Any]:
        entry = self.tools.get(name)
        if entry is None:
            return {"success": False, "output": f"Tool not found: {name}", "metadata": {}}
        adapter = entry.adapter
        if inspect.iscoroutinefunction(adapter):
            result = await adapter(name, args)
        elif hasattr(adapter, "__call__") and inspect.iscoroutinefunction(getattr(type(adapter), "__call__", None)):
            result = await adapter(name, args)
        else:
            result = adapter(name, args)
        return {
            "success": result.success,
            "output": result.output,
            "metadata": result.metadata,
        }

    async def _record_episode(self, content: str, kind: EpisodeKind, *, session_id: str, role: str | None = None) -> None:
        record = getattr(self._base_runtime, "_record_episode", None)
        if record is None:
            return
        await record(content, kind, session_id=session_id, role=role)


def _extract_env_value(keys: Iterable[str], path: Optional[Path] = None) -> Optional[str]:
    import os

    for key in keys:
        direct = os.environ.get(key)
        if direct and str(direct).strip():
            return str(direct).strip().strip('"').strip("'")
    if path is None or not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    for key in keys:
        match = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*\"?([^\"\n]+)", text)
        if match:
            return match.group(1).strip().strip('"').strip("'")
    return None


def _xml_response(*verbs: str) -> str:
    return '<?xml version="1.0" encoding="UTF-8"?><Response>' + "".join(verbs) + "</Response>"


def _say(text: str) -> str:
    return f"<Say>{escape(text)}</Say>"


def _play(url: str) -> str:
    return f"<Play>{escape(url)}</Play>"


def _hangup() -> str:
    return "<Hangup/>"


def _gather(
    action_url: str,
    prompt_verb: str,
    *,
    input_mode: str = "speech",
    speech_timeout: str | None = "auto",
    num_digits: int | None = None,
    timeout_seconds: int | None = None,
) -> str:
    attributes = [
        f'input="{escape(input_mode)}"',
        f'action="{escape(action_url)}"',
        'method="POST"',
        'language="en-US"',
    ]
    if "speech" in input_mode and speech_timeout:
        attributes.append(f'speechTimeout="{escape(str(speech_timeout))}"')
    if num_digits is not None:
        attributes.append(f'numDigits="{max(1, int(num_digits))}"')
    if timeout_seconds is not None:
        attributes.append(f'timeout="{max(1, int(timeout_seconds))}"')
    return f"<Gather {' '.join(attributes)}>{prompt_verb}</Gather>"


def _redirect(url: str, *, method: str = "POST") -> str:
    return f'<Redirect method="{escape(method)}">{escape(url)}</Redirect>'


def _pause(seconds: int) -> str:
    safe_seconds = max(1, min(int(seconds), 60))
    return f'<Pause length="{safe_seconds}"/>'


class PhoneBridgeService:
    """Resolve caller trust and mediate Twilio phone conversations."""

    def __init__(self, runtime: Any, config: PhoneRuntimeConfig) -> None:
        self.runtime = runtime
        self.config = config
        self._outbound_contexts: dict[str, OutboundCallContext] = {}
        self._pending_phone_replies: dict[str, PendingPhoneReply] = {}
        self._sync_employer_shared_workspace()

    def status(self) -> Dict[str, Any]:
        webhook_base = (self.config.public_base_url or "").rstrip("/")
        urls = {
            "voice": f"{webhook_base}/api/phone/twilio/voice" if webhook_base else None,
            "gather": f"{webhook_base}/api/phone/twilio/gather" if webhook_base else None,
            "poll": f"{webhook_base}/api/phone/twilio/poll" if webhook_base else None,
        }
        menu = self._menu_config()
        return {
            **self.config.redacted_dict(),
            "twilio_credentials_configured": self._twilio_credentials() is not None,
            "webhook_urls": urls,
            "contact_count": len(self.config.contacts),
            "screening_menu": menu.model_dump(mode="json"),
        }

    def trace_phone_event(
        self,
        event: str,
        *,
        caller: PhoneResolvedCaller | None = None,
        call_sid: str | None = None,
        **payload: Any,
    ) -> None:
        trace = getattr(self.runtime, "_trace", None)
        if not callable(trace):
            return
        event_payload: Dict[str, Any] = dict(payload)
        if caller is not None:
            event_payload.setdefault("caller_number", caller.phone_number)
            event_payload.setdefault("caller_display_name", caller.display_name)
            event_payload.setdefault("caller_trust_level", caller.trust_level)
        if call_sid:
            event_payload.setdefault("call_sid", str(call_sid).strip())
        try:
            trace(event, event_payload)
        except Exception:
            pass

    def _configured_menu_config_path(self) -> Path:
        raw = str(self.config.menu_config_path or "").strip()
        if raw:
            return Path(raw).expanduser()
        return _PHONE_MENU_CONFIG_PATH

    def _menu_config(self) -> PhoneMenuConfig:
        return load_phone_menu_config(self._configured_menu_config_path())

    async def autoconfigure_twilio(
        self,
        *,
        enabled: Optional[bool] = None,
        public_base_url: Optional[str] = None,
        webhook_signature_required: Optional[bool] = None,
        webhook_secret: Optional[str] = None,
        twilio_env_path: Optional[str] = None,
        twilio_account_sid: Optional[str] = None,
        twilio_api_key_sid: Optional[str] = None,
        twilio_api_secret: Optional[str] = None,
        twilio_auth_token: Optional[str] = None,
        twilio_from_number: Optional[str] = None,
        owner_phone_number: Optional[str] = None,
        owner_display_name: Optional[str] = None,
        owner_workspace_subdir: Optional[str] = None,
    ) -> Dict[str, Any]:
        credential_config = self.config.model_copy(
            update={
                "twilio_env_path": twilio_env_path if twilio_env_path is not None else self.config.twilio_env_path,
                "twilio_account_sid": twilio_account_sid if twilio_account_sid is not None else self.config.twilio_account_sid,
                "twilio_api_key_sid": twilio_api_key_sid if twilio_api_key_sid is not None else self.config.twilio_api_key_sid,
                "twilio_api_secret": twilio_api_secret if twilio_api_secret is not None else self.config.twilio_api_secret,
                "twilio_auth_token": twilio_auth_token if twilio_auth_token is not None else self.config.twilio_auth_token,
            }
        )
        try:
            credentials = self._twilio_credentials(config=credential_config)
        except TypeError:
            credentials = self._twilio_credentials()
        if credentials is None:
            raise RuntimeError("Twilio credentials are not configured")

        requested_base_url = str(public_base_url or self.config.public_base_url or "").strip()
        webhook_secret = str(webhook_secret or self.config.webhook_secret or secrets.token_urlsafe(24)).strip()

        requested_from = normalize_phone_number(twilio_from_number or self.config.twilio_from_number)
        number_list_url = (
            f"https://api.twilio.com/2010-04-01/Accounts/{credentials.account_sid}/IncomingPhoneNumbers.json"
        )
        async with httpx.AsyncClient(auth=(credentials.api_username, credentials.api_password), timeout=30.0) as client:
            number_response = await client.get(number_list_url)
            number_response.raise_for_status()
            number_payload = number_response.json()
            candidates = [
                self._serialize_incoming_number(item)
                for item in number_payload.get("incoming_phone_numbers") or []
                if isinstance(item, Mapping)
            ]
            if not candidates:
                raise RuntimeError("No Twilio incoming phone numbers were found for this account")

            selected = self._select_incoming_number(candidates, requested_from=requested_from)
            updated_payload: Mapping[str, Any] | None = None
            voice_url: str | None = None
            if requested_base_url:
                voice_url = self._absolute_url(
                    requested_base_url,
                    "/api/phone/twilio/voice",
                    bridge_token=webhook_secret,
                )
                update_response = await client.post(
                    (
                        "https://api.twilio.com/2010-04-01/Accounts/"
                        f"{credentials.account_sid}/IncomingPhoneNumbers/{selected['sid']}.json"
                    ),
                    data={
                        "VoiceUrl": voice_url,
                        "VoiceMethod": "POST",
                    },
                )
                update_response.raise_for_status()
                maybe_payload = update_response.json()
                if isinstance(maybe_payload, Mapping):
                    updated_payload = maybe_payload

        updated_number = self._serialize_incoming_number(updated_payload or selected)
        if voice_url:
            updated_number["voice_url"] = updated_number.get("voice_url") or voice_url
            updated_number["voice_method"] = updated_number.get("voice_method") or "POST"

        next_settings = PhoneRuntimeConfig(
            enabled=self.config.enabled if enabled is None else bool(enabled),
            public_base_url=requested_base_url,
            webhook_signature_required=(
                self.config.webhook_signature_required
                if webhook_signature_required is None
                else bool(webhook_signature_required)
            ),
            webhook_secret=webhook_secret,
            twilio_env_path=credential_config.twilio_env_path,
            twilio_account_sid=credential_config.twilio_account_sid,
            twilio_api_key_sid=credential_config.twilio_api_key_sid,
            twilio_api_secret=credential_config.twilio_api_secret,
            twilio_auth_token=credential_config.twilio_auth_token,
            twilio_from_number=updated_number["phone_number"],
            owner_phone_number=owner_phone_number or self.config.owner_phone_number,
            owner_display_name=owner_display_name or self.config.owner_display_name,
            owner_workspace_subdir=owner_workspace_subdir or self.config.owner_workspace_subdir,
            elevenlabs_env_path=self.config.elevenlabs_env_path,
            elevenlabs_api_key=self.config.elevenlabs_api_key,
            elevenlabs_voice_id=self.config.elevenlabs_voice_id,
            elevenlabs_stt_model=self.config.elevenlabs_stt_model,
            elevenlabs_fast_model=self.config.elevenlabs_fast_model,
            elevenlabs_expressive_model=self.config.elevenlabs_expressive_model,
            edge_tts_voice=self.config.edge_tts_voice,
            contacts=self.config.contacts,
        )
        return {
            "settings": next_settings,
            "twilio_number_candidates": candidates,
            "selected_number": updated_number,
            "webhook_update": {
                "voice_url": updated_number.get("voice_url"),
                "voice_method": updated_number.get("voice_method") or ("POST" if voice_url else None),
                "configured": bool(voice_url),
            },
            "note": (
                None
                if voice_url
                else "Twilio number discovery completed, but voice webhook setup was skipped because public_base_url is not configured."
            ),
        }

    async def place_owner_call(
        self,
        *,
        message: str,
        reason: str = "",
    ) -> Dict[str, Any]:
        if not self.config.enabled:
            raise RuntimeError("Phone bridge is disabled")
        credentials = self._twilio_credentials()
        if credentials is None:
            raise RuntimeError("Twilio credentials are not configured")
        if not self.config.twilio_from_number:
            raise RuntimeError("twilio_from_number is not configured")
        if not self.config.owner_phone_number:
            raise RuntimeError("owner_phone_number is not configured")
        intro = (message or "").strip() or "Hi, it's the OpenCAS agent calling from OpenCAS."
        payload = {
            "From": self.config.twilio_from_number,
            "To": self.config.owner_phone_number,
        }
        callback_url = None
        voice_mode = "inline_twiml"
        if self.config.public_base_url:
            context = self._create_outbound_context(intro_message=intro, reason=reason)
            callback_url = self._absolute_url(
                self.config.public_base_url,
                "/api/phone/twilio/voice",
                call_token=context.token,
                bridge_token=self.config.webhook_secret,
            )
            payload["Url"] = callback_url
            payload["Method"] = "POST"
            voice_mode = "webhook"
        else:
            payload["Twiml"] = _xml_response(_say(intro), _hangup())
        async with httpx.AsyncClient(auth=(credentials.api_username, credentials.api_password), timeout=30.0) as client:
            response = await client.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{credentials.account_sid}/Calls.json",
                data=payload,
            )
        response.raise_for_status()
        data = response.json()
        return {
            "ok": True,
            "call_sid": data.get("sid"),
            "status": data.get("status"),
            "to": self.config.owner_phone_number,
            "from": self.config.twilio_from_number,
            "callback_url": callback_url,
            "voice_mode": voice_mode,
        }

    def validate_webhook_signature(
        self,
        *,
        request_url: str,
        form_data: Mapping[str, Any],
        provided_signature: Optional[str],
    ) -> bool:
        credentials = self._twilio_credentials()
        auth_token = credentials.webhook_auth_token if credentials is not None else None
        if not self.config.webhook_signature_required:
            return True
        if not auth_token:
            self._trace_signature_failure(
                reason="missing_auth_token",
                request_url=request_url,
                form_data=form_data,
            )
            return False
        if not provided_signature:
            self._trace_signature_failure(
                reason="missing_signature",
                request_url=request_url,
                form_data=form_data,
            )
            return False

        validator = RequestValidator(auth_token)
        params: Any
        if callable(getattr(form_data, "getall", None)) or callable(getattr(form_data, "getlist", None)):
            params = form_data
        else:
            params = {str(key): str(value) for key, value in form_data.items()}
        try:
            is_valid = bool(
                validator.validate(
                    request_url,
                    params,
                    provided_signature.strip(),
                )
            )
        except Exception:
            is_valid = False
        if not is_valid:
            self._trace_signature_failure(
                reason="validator_rejected",
                request_url=request_url,
                form_data=form_data,
            )
        return is_valid

    async def handle_voice_webhook(
        self,
        *,
        request_url: str,
        webhook_base_url: str,
        form_data: Mapping[str, Any],
        provided_signature: Optional[str],
        call_token: Optional[str] = None,
        bridge_token: Optional[str] = None,
    ) -> str:
        if not self.validate_webhook_request(
            request_url=request_url,
            form_data=form_data,
            provided_signature=provided_signature,
            bridge_token=bridge_token,
        ):
            self._trace_webhook_decision("voice", "signature_invalid", form_data=form_data)
            return _xml_response(_say("Unauthorized phone bridge request."), _hangup())
        normalized = self._normalize_form(form_data)
        if not self.config.enabled:
            self._trace_webhook_decision("voice", "bridge_disabled", form_data=normalized)
            return _xml_response(_say("The OpenCAS agent's phone bridge is currently offline."), _hangup())

        caller = self._resolve_caller(
            from_number=normalized.get("From"),
            to_number=normalized.get("To"),
            call_token=call_token,
        )
        if caller is None:
            caller = self._screening_caller(normalized.get("From"))
            if caller is None:
                self._trace_webhook_decision("voice", "caller_not_authorized", form_data=normalized)
                return _xml_response(_say("This phone number is not authorized for the OpenCAS agent."), _hangup())
            self._trace_webhook_decision("voice", "accepted_screening", caller=caller, form_data=normalized)
        else:
            self._trace_webhook_decision("voice", "accepted", caller=caller, form_data=normalized)

        intro_context = self._get_outbound_context(call_token)
        intro_message = intro_context.intro_message if intro_context and not intro_context.consumed else ""
        if intro_context is not None:
            self._mark_outbound_context_consumed(intro_context.token)

        if caller.is_owner:
            if self.owner_pin_required():
                return self._owner_pin_gather_twiml(
                    webhook_base_url=webhook_base_url,
                    call_token=call_token,
                    prompt_text=self.owner_pin_prompt(),
                    retry_text=self.owner_pin_retry_prompt(),
                )
            elif self.owner_menu_enabled():
                owner_menu_key = self.owner_menu_key() or self.default_menu_key()
                return self._menu_gather_twiml(
                    webhook_base_url=webhook_base_url,
                    call_token=call_token,
                    menu_key=owner_menu_key,
                    stream_mode="owner_menu",
                    allow_speech=False,
                    prompt_text=intro_message or self.menu_prompt(owner_menu_key),
                    reprompt_text=self.menu_reprompt(owner_menu_key),
                )
            owner_intro = intro_message or self.default_stream_greeting(caller, stream_mode="owner")
            return self._stream_connect_twiml(
                caller=caller,
                webhook_base_url=webhook_base_url,
                call_token=call_token,
                intro_message=owner_intro,
                stream_mode="owner",
            )
        menu_key = self.default_menu_key()
        return self._menu_gather_twiml(
            webhook_base_url=webhook_base_url,
            call_token=call_token,
            menu_key=menu_key,
            stream_mode="screening",
            allow_speech=True,
            prompt_text=intro_message or self.menu_prompt(menu_key),
            reprompt_text=self.menu_reprompt(menu_key),
        )

    async def handle_gather_webhook(
        self,
        *,
        request_url: str,
        webhook_base_url: str,
        form_data: Mapping[str, Any],
        provided_signature: Optional[str],
        call_token: Optional[str] = None,
        bridge_token: Optional[str] = None,
    ) -> str:
        if not self.validate_webhook_request(
            request_url=request_url,
            form_data=form_data,
            provided_signature=provided_signature,
            bridge_token=bridge_token,
        ):
            self._trace_webhook_decision("gather", "signature_invalid", form_data=form_data)
            return _xml_response(_say("Unauthorized phone bridge request."), _hangup())
        normalized = self._normalize_form(form_data)
        if not self.config.enabled:
            self._trace_webhook_decision("gather", "bridge_disabled", form_data=normalized)
            return _xml_response(_say("The OpenCAS agent's phone bridge is currently offline."), _hangup())

        caller = self._resolve_caller(
            from_number=normalized.get("From"),
            to_number=normalized.get("To"),
            call_token=call_token,
        )
        request_query = self._request_query_params(request_url)
        stream_mode = str(request_query.get("stream_mode") or "").strip().lower()
        menu_key = str(request_query.get("menu_key") or "").strip()
        digits = str(normalized.get("Digits") or "").strip()
        if caller is None and stream_mode == "screening":
            caller = self._screening_caller(
                normalized.get("From"),
                display_name=normalized.get("CallerName") or normalized.get("From"),
            )
        if caller is None:
            self._trace_webhook_decision("gather", "caller_not_authorized", form_data=normalized)
            return _xml_response(_say("This phone number is not authorized for the OpenCAS agent."), _hangup())

        if stream_mode == "owner_pin":
            if not digits:
                self._trace_webhook_decision(
                    "gather",
                    "owner_pin_missing_digits",
                    caller=caller,
                    form_data=normalized,
                )
                return self._owner_pin_gather_twiml(
                    webhook_base_url=webhook_base_url,
                    call_token=call_token,
                    prompt_text=self.owner_pin_retry_prompt(),
                    retry_text=self.owner_pin_failure_message(),
                )
            if self.validate_owner_pin(digits):
                self._trace_webhook_decision(
                    "gather",
                    "owner_pin_verified",
                    caller=caller,
                    form_data=normalized,
                )
                if self.owner_menu_enabled():
                    next_menu_key = self.owner_menu_key() or self.default_menu_key()
                    return self._menu_gather_twiml(
                        webhook_base_url=webhook_base_url,
                        call_token=call_token,
                        menu_key=next_menu_key,
                        stream_mode="owner_menu",
                        allow_speech=False,
                        prompt_text=f"{self.owner_pin_success_message()} {self.menu_prompt(next_menu_key)}".strip(),
                        reprompt_text=self.menu_reprompt(next_menu_key),
                    )
                return self._stream_connect_reply_twiml(
                    caller=caller,
                    webhook_base_url=webhook_base_url,
                    call_token=call_token,
                    stream_mode="owner",
                    preface_text=self.owner_pin_success_message(),
                    intro_message="",
                )
            self._trace_webhook_decision(
                "gather",
                "owner_pin_rejected",
                caller=caller,
                form_data=normalized,
            )
            return _xml_response(_say(self.owner_pin_failure_message()), _hangup())

        if menu_key and stream_mode in {"owner_menu", "screening"}:
            transcript = str(normalized.get("SpeechResult") or "").strip()
            choice: str | None = None
            if digits:
                choice = self.classify_menu_digit(menu_key, digits[:1])
            elif transcript:
                choice = self.classify_menu_transcript(menu_key, transcript)
            option = self.resolve_menu_option(menu_key, choice)
            if option is None:
                self._trace_webhook_decision(
                    "gather",
                    "menu_unmatched",
                    caller=caller,
                    form_data=normalized,
                )
                return self._menu_gather_twiml(
                    webhook_base_url=webhook_base_url,
                    call_token=call_token,
                    menu_key=menu_key,
                    stream_mode=stream_mode,
                    allow_speech=stream_mode != "owner_menu",
                    prompt_text=self.menu_reprompt(menu_key),
                    reprompt_text=self.menu_reprompt(menu_key),
                )
            self._trace_webhook_decision(
                "gather",
                "menu_choice",
                caller=caller,
                form_data={**normalized, "menu_key": menu_key, "option_key": option.key},
            )
            if option.action == "owner_conversation":
                return self._stream_connect_reply_twiml(
                    caller=caller,
                    webhook_base_url=webhook_base_url,
                    call_token=call_token,
                    stream_mode="owner",
                    preface_text="Go ahead.",
                    intro_message="",
                )
            if option.action == "submenu":
                next_menu_key = str(option.target_menu or "").strip() or self.default_menu_key()
                next_mode = "owner_menu" if next_menu_key == (self.owner_menu_key() or "") else "screening"
                return self._menu_gather_twiml(
                    webhook_base_url=webhook_base_url,
                    call_token=call_token,
                    menu_key=next_menu_key,
                    stream_mode=next_mode,
                    allow_speech=next_mode != "owner_menu",
                    prompt_text=self.menu_prompt(next_menu_key),
                    reprompt_text=self.menu_reprompt(next_menu_key),
                )
            if option.action == "workspace_assistant":
                next_caller = await self.activate_menu_workspace_caller(
                    option=option,
                    caller_number=getattr(caller, "phone_number", None),
                    display_name=getattr(caller, "display_name", None),
                )
                return self._stream_connect_reply_twiml(
                    caller=next_caller,
                    webhook_base_url=webhook_base_url,
                    call_token=call_token,
                    stream_mode="workspace_assistant",
                    preface_text=self.menu_workspace_acceptance(next_caller),
                    intro_message="",
            )
            announcement = await self.screening_option_announcement(
                option,
                caller_number=getattr(caller, "phone_number", None),
                display_name=getattr(caller, "display_name", None),
            )
            return _xml_response(_say(announcement), _hangup())

        if not caller.is_owner and not caller.allowed_actions:
            self._trace_webhook_decision(
                "gather",
                "caller_not_configured",
                caller=caller,
                form_data=normalized,
            )
            return _xml_response(_say("This phone number is not configured for phone access."), _hangup())

        transcript = str(normalized.get("SpeechResult") or "").strip()
        call_sid = str(normalized.get("CallSid") or "").strip()
        if not transcript:
            self._trace_webhook_decision(
                "gather",
                "empty_transcript",
                caller=caller,
                form_data=normalized,
            )
            return _xml_response(_say("I did not catch that. Goodbye for now."), _hangup())
        self._trace_webhook_decision("gather", "accepted", caller=caller, form_data=normalized)

        if caller.is_owner:
            reply_token = self._start_owner_reply_task(
                caller=caller,
                transcript=transcript,
                call_sid=call_sid,
                webhook_base_url=webhook_base_url,
            )
            return self._pending_reply_twiml(
                webhook_base_url=webhook_base_url,
                call_token=call_token,
                reply_token=reply_token,
            )

        if not caller.allows_knowledge_qa:
            await self._persist_voicemail(caller, transcript, call_sid=call_sid)
            return await self._voice_reply_twiml(
                caller=caller,
                text="Thanks. I saved your message for the OpenCAS agent.",
                webhook_base_url=webhook_base_url,
                continue_listening=False,
                call_token=call_token,
                expressive=False,
                prefer_direct_speech=True,
            )

        response_text = await self._respond_low_trust(caller, transcript, call_sid=call_sid)
        return await self._voice_reply_twiml(
            caller=caller,
            text=response_text,
            webhook_base_url=webhook_base_url,
            continue_listening=True,
            call_token=call_token,
            expressive=False,
            prefer_direct_speech=True,
        )

    async def handle_poll_webhook(
        self,
        *,
        request_url: str,
        webhook_base_url: str,
        form_data: Mapping[str, Any],
        provided_signature: Optional[str],
        call_token: Optional[str] = None,
        bridge_token: Optional[str] = None,
        reply_token: Optional[str] = None,
    ) -> str:
        if not self.validate_webhook_request(
            request_url=request_url,
            form_data=form_data,
            provided_signature=provided_signature,
            bridge_token=bridge_token,
        ):
            self._trace_webhook_decision("poll", "signature_invalid", form_data=form_data)
            return _xml_response(_say("Unauthorized phone bridge request."), _hangup())
        normalized = self._normalize_form(form_data)
        if not self.config.enabled:
            self._trace_webhook_decision("poll", "bridge_disabled", form_data=normalized)
            return _xml_response(_say("The OpenCAS agent's phone bridge is currently offline."), _hangup())

        caller = self._resolve_caller(
            from_number=normalized.get("From"),
            to_number=normalized.get("To"),
            call_token=call_token,
        )
        if caller is None:
            self._trace_webhook_decision("poll", "caller_not_authorized", form_data=normalized)
            return _xml_response(_say("This phone number is not authorized for the OpenCAS agent."), _hangup())

        pending = self._get_pending_phone_reply(reply_token)
        if pending is None:
            self._trace_webhook_decision("poll", "reply_missing", caller=caller, form_data=normalized)
            return await self._voice_reply_twiml(
                caller=caller,
                text="I lost the pending reply. Please ask me again.",
                webhook_base_url=webhook_base_url,
                continue_listening=True,
                call_token=call_token,
                expressive=caller.is_owner,
                prefer_direct_speech=True,
            )

        if not pending.task.done():
            self._trace_webhook_decision("poll", "reply_pending", caller=caller, form_data=normalized)
            return self._pending_reply_twiml(
                webhook_base_url=webhook_base_url,
                call_token=call_token,
                reply_token=pending.token,
            )

        self._trace_webhook_decision("poll", "reply_ready", caller=caller, form_data=normalized)
        resolved = self._pop_pending_phone_reply(pending.token)
        try:
            reply = resolved.task.result()
        except Exception:
            self._trace_owner_reply_issue(
                "background_task_error",
                caller=caller,
                call_sid=resolved.call_sid,
            )
            return await self._voice_reply_twiml(
                caller=caller,
                text="I hit a problem preparing that reply. Please ask again or use the main workspace.",
                webhook_base_url=webhook_base_url,
                continue_listening=True,
                call_token=call_token,
                expressive=caller.is_owner,
                prefer_direct_speech=True,
            )
        return self._reply_twiml_from_prompt_verb(
            prompt_verb=reply.prompt_verb,
            webhook_base_url=webhook_base_url,
            continue_listening=True,
            call_token=call_token,
        )

    async def handle_media_stream(
        self,
        *,
        websocket: Any,
        request_url: str,
        provided_signature: Optional[str],
        stream_secret: str,
    ) -> None:
        if not self.validate_media_stream_request(
            request_url=request_url,
            provided_signature=provided_signature,
            stream_secret=stream_secret,
        ):
            await websocket.close(code=1008)
            return
        session = PhoneMediaStreamSession(
            websocket=websocket,
            service=self,
        )
        await session.run()

    def _configured_twilio_env_path(self, config: Optional[PhoneRuntimeConfig] = None) -> Optional[Path]:
        source = config or self.config
        raw = str(source.twilio_env_path or "").strip()
        if not raw:
            return None
        return Path(raw).expanduser()

    def _twilio_credentials(self, config: Optional[PhoneRuntimeConfig] = None) -> Optional[TwilioCredentials]:
        source = config or self.config
        env_path = self._configured_twilio_env_path(source)
        account_sid = source.twilio_account_sid or _extract_env_value(("TWILIO_ACCOUNT_SID", "ACCOUNT_SID"), env_path)
        api_username = source.twilio_api_key_sid or _extract_env_value(
            ("TWILIO_API_KEY", "TWILIO_SID", "SID", "TWILIO_ACCOUNT_SID", "ACCOUNT_SID"),
            env_path,
        )
        api_password = source.twilio_api_secret or _extract_env_value(
            ("TWILIO_API_SECRET", "TWILIO_SECRET", "SECRET", "TWILIO_AUTH_TOKEN", "AUTH_TOKEN"),
            env_path,
        )
        webhook_auth_token = source.twilio_auth_token or _extract_env_value(
            ("TWILIO_AUTH_TOKEN", "AUTH_TOKEN"),
            env_path,
        )
        if not api_username and account_sid and webhook_auth_token:
            api_username = account_sid
        if not api_password and webhook_auth_token:
            api_password = webhook_auth_token
        if not account_sid and api_username and str(api_username).startswith("AC"):
            account_sid = api_username
        if not account_sid and api_username and api_password:
            account_sid = self._discover_account_sid(api_username, api_password)
        if not account_sid or not api_username or not api_password:
            return None
        return TwilioCredentials(
            account_sid=account_sid,
            api_username=api_username,
            api_password=api_password,
            webhook_auth_token=webhook_auth_token,
        )

    def _discover_account_sid(self, api_username: str, api_password: str) -> Optional[str]:
        try:
            response = httpx.get(
                "https://api.twilio.com/2010-04-01/Accounts.json",
                auth=(api_username, api_password),
                timeout=15.0,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return None
        accounts = payload.get("accounts") if isinstance(payload, Mapping) else None
        if not isinstance(accounts, list):
            return None
        for item in accounts:
            if not isinstance(item, Mapping):
                continue
            sid = str(item.get("sid") or "").strip()
            if sid.startswith("AC"):
                return sid
        return None

    def _serialize_incoming_number(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "sid": str(payload.get("sid") or "").strip(),
            "friendly_name": str(payload.get("friendly_name") or "").strip(),
            "phone_number": normalize_phone_number(payload.get("phone_number")),
            "voice_url": str(payload.get("voice_url") or "").strip() or None,
            "voice_method": str(payload.get("voice_method") or "").strip() or None,
        }

    def _select_incoming_number(
        self,
        candidates: list[Dict[str, Any]],
        *,
        requested_from: Optional[str],
    ) -> Dict[str, Any]:
        if requested_from:
            for item in candidates:
                if item.get("phone_number") == requested_from:
                    return item
            if len(candidates) == 1:
                return candidates[0]
            raise RuntimeError(f"Configured Twilio number {requested_from} was not found in this account")
        if len(candidates) == 1:
            return candidates[0]
        normalized_candidates = [item.get("phone_number") or item.get("friendly_name") or item.get("sid") for item in candidates]
        raise RuntimeError(
            "Multiple Twilio incoming numbers were found for this account. "
            f"Set twilio_from_number explicitly. Available numbers: {', '.join(str(item) for item in normalized_candidates)}"
        )

    def _normalize_form(self, form_data: Mapping[str, Any]) -> Dict[str, str]:
        return {
            str(key): str(value)
            for key, value in form_data.items()
            if value is not None
        }

    def _resolve_caller(
        self,
        *,
        from_number: Optional[str],
        to_number: Optional[str] = None,
        call_token: Optional[str] = None,
    ) -> Optional[PhoneResolvedCaller]:
        owner_number = self.config.owner_phone_number
        normalized_from = normalize_phone_number(from_number)
        normalized_to = normalize_phone_number(to_number)
        if owner_number and (normalized_from == owner_number or (call_token and normalized_to == owner_number)):
            return PhoneResolvedCaller(
                phone_number=owner_number,
                display_name=self.config.owner_display_name,
                trust_level="owner",
                allowed_actions=("leave_message", "knowledge_qa"),
                workspace_subdir=self.config.owner_workspace_subdir,
                is_owner=True,
            )

        for contact in self.config.contacts:
            if normalized_from == contact.phone_number:
                return PhoneResolvedCaller(
                    phone_number=contact.phone_number,
                    display_name=contact.display_name or contact.phone_number,
                    trust_level=contact.trust_level,
                    allowed_actions=tuple(contact.allowed_actions),
                    workspace_subdir=contact.workspace_subdir,
                    notes=contact.notes,
                    is_owner=False,
                )
        return None

    def _screening_caller(
        self,
        from_number: Optional[str],
        *,
        display_name: Optional[str] = None,
    ) -> Optional[PhoneResolvedCaller]:
        normalized = normalize_phone_number(from_number)
        if not normalized:
            return None
        label = str(display_name or "").strip() or "Caller"
        configured = self._resolve_caller(from_number=normalized)
        if configured is not None and not configured.is_owner:
            label = configured.display_name or label
        return PhoneResolvedCaller(
            phone_number=normalized,
            display_name=label,
            trust_level="low",
            allowed_actions=(),
            notes=_EMPLOYER_POLICY_NOTE,
            is_owner=False,
        )

    def _menu_workspace_caller(
        self,
        option: PhoneMenuOption,
        *,
        caller_number: Optional[str],
        display_name: Optional[str] = None,
    ) -> Optional[PhoneResolvedCaller]:
        normalized = normalize_phone_number(caller_number)
        if not normalized:
            return None
        configured = self._resolve_caller(from_number=normalized)
        label = (
            configured.display_name
            if configured is not None and not configured.is_owner and configured.display_name
            else str(display_name or option.label or "Caller").strip() or "Caller"
        )
        caller_workspace_subdir = None
        for mount in option.workspace_mounts:
            if mount.scope == "caller":
                caller_workspace_subdir = self._render_workspace_subdir(mount.subdir, normalized)
                break
        return PhoneResolvedCaller(
            phone_number=normalized,
            display_name=label,
            trust_level="low",
            allowed_actions=tuple(option.allowed_actions),
            workspace_subdir=caller_workspace_subdir,
            notes=_EMPLOYER_POLICY_NOTE,
            is_owner=False,
            menu_option_key=option.key,
            prompt_profile=option.prompt_profile,
            workspace_mounts=tuple(option.workspace_mounts),
        )

    def _employer_caller(
        self,
        from_number: Optional[str],
        *,
        display_name: Optional[str] = None,
    ) -> Optional[PhoneResolvedCaller]:
        option = self.screening_option_by_key("employer")
        if option is not None:
            return self._menu_workspace_caller(option, caller_number=from_number, display_name=display_name)
        normalized = normalize_phone_number(from_number)
        if not normalized:
            return None
        label = str(display_name or "").strip() or "Potential employer"
        configured = self._resolve_caller(from_number=normalized)
        if configured is not None and not configured.is_owner and configured.display_name:
            label = configured.display_name
        return PhoneResolvedCaller(
            phone_number=normalized,
            display_name=label,
            trust_level="low",
            allowed_actions=("leave_message", "knowledge_qa"),
            workspace_subdir=f"{_EMPLOYER_WORKSPACE_ROOT}/{normalized.lstrip('+')}",
            notes=_EMPLOYER_POLICY_NOTE,
            is_owner=False,
            menu_option_key="employer",
            prompt_profile="worksafe_owner",
            workspace_mounts=(
                PhoneWorkspaceMount(scope="shared", subdir=_EMPLOYER_SHARED_WORKSPACE, access="read_only"),
                PhoneWorkspaceMount(
                    scope="caller",
                    subdir=f"{_EMPLOYER_WORKSPACE_ROOT}/{{phone_digits}}",
                    access="append_only",
                ),
            ),
        )

    @staticmethod
    def _render_workspace_subdir(template: str, normalized_phone_number: str) -> str:
        digits = normalized_phone_number.lstrip("+")
        return (
            str(template or "")
            .replace("{phone_number}", normalized_phone_number)
            .replace("{phone_digits}", digits)
        )

    def resolve_stream_session_caller(
        self,
        *,
        stream_mode: str,
        caller_number: Optional[str],
        display_name: Optional[str],
        call_token: Optional[str],
    ) -> Optional[PhoneResolvedCaller]:
        normalized_mode = str(stream_mode or "owner").strip().lower() or "owner"
        if normalized_mode in {"owner", "owner_pin", "owner_menu"}:
            return self._resolve_caller(
                from_number=caller_number,
                to_number=self.config.owner_phone_number,
                call_token=call_token,
            )
        if normalized_mode in {"employer", "workspace_assistant"}:
            return self._employer_caller(caller_number, display_name=display_name)
        return self._screening_caller(caller_number, display_name=display_name)

    def default_stream_greeting(self, caller: PhoneResolvedCaller, *, stream_mode: str) -> str:
        normalized_mode = str(stream_mode or "owner").strip().lower() or "owner"
        if normalized_mode == "screening":
            return self.menu_prompt(self.default_menu_key())
        if normalized_mode == "owner_menu":
            return self.menu_prompt(self.owner_menu_key() or self.default_menu_key())
        if normalized_mode == "owner_pin":
            return self.owner_pin_prompt()
        if normalized_mode in {"employer", "workspace_assistant"}:
            return self.menu_workspace_acceptance(caller)
        return self._default_greeting(caller)

    def _session_id(self, caller: PhoneResolvedCaller) -> str:
        return f"phone:{caller.phone_number}"

    def _contact_workspace_dir(self, caller: PhoneResolvedCaller) -> Path:
        managed_root = self.runtime.ctx.config.agent_workspace_root()
        subdir = caller.workspace_subdir or (
            self.config.owner_workspace_subdir if caller.is_owner else f"phone/contacts/{caller.phone_number.lstrip('+')}"
        )
        candidate = (Path(managed_root) / subdir).resolve()
        managed_root_resolved = Path(managed_root).resolve()
        try:
            candidate.relative_to(managed_root_resolved)
        except ValueError as exc:
            raise RuntimeError("contact workspace escaped managed workspace root") from exc
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _owner_workspace_dir(self) -> Path:
        managed_root = self.runtime.ctx.config.agent_workspace_root()
        candidate = (Path(managed_root) / self.config.owner_workspace_subdir).resolve()
        managed_root_resolved = Path(managed_root).resolve()
        try:
            candidate.relative_to(managed_root_resolved)
        except ValueError as exc:
            raise RuntimeError("owner workspace escaped managed workspace root") from exc
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _employer_shared_workspace_dir(self) -> Path:
        managed_root = self.runtime.ctx.config.agent_workspace_root()
        candidate = (Path(managed_root) / _EMPLOYER_SHARED_WORKSPACE).resolve()
        managed_root_resolved = Path(managed_root).resolve()
        try:
            candidate.relative_to(managed_root_resolved)
        except ValueError as exc:
            raise RuntimeError("employer shared workspace escaped managed workspace root") from exc
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _employment_shared_seed_dir(self) -> Path:
        candidate = _EMPLOYER_SHARED_SEED_DIR.resolve()
        repo_root = _REPO_ROOT.resolve()
        try:
            candidate.relative_to(repo_root)
        except ValueError as exc:
            raise RuntimeError("employment shared seed directory escaped repo root") from exc
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _phone_profile_dir(self, profile_name: Optional[str]) -> Path:
        cleaned = str(profile_name or "").strip().replace("\\", "/")
        candidate = (_PHONE_PROFILE_ROOT / cleaned).resolve()
        repo_root = _REPO_ROOT.resolve()
        try:
            candidate.relative_to(repo_root)
        except ValueError as exc:
            raise RuntimeError("phone profile directory escaped repo root") from exc
        return candidate

    def _default_greeting(self, caller: PhoneResolvedCaller) -> str:
        if caller.is_owner:
            return "Hi, it's the OpenCAS agent. I'm here on the phone. What do you need?"
        if not caller.allowed_actions:
            return "This phone number is not configured for phone access."
        if caller.allows_knowledge_qa:
            return (
                f"Hello {caller.display_name}. You can leave a message, or ask questions "
                "that are covered by the workspace prepared for your number."
            )
        return f"Hello {caller.display_name}. Please leave your message for the OpenCAS agent."

    def employer_screening_prompt(self) -> str:
        return self.menu_prompt(self.default_menu_key()) or _SCREENING_MENU_PROMPT

    def employer_screening_reprompt(self) -> str:
        return self.menu_reprompt(self.default_menu_key()) or _SCREENING_MENU_REPROMPT

    def employer_screening_rejection(self) -> str:
        return _SCREENING_REJECTION

    def employer_screening_acceptance(self, caller: PhoneResolvedCaller) -> str:
        label = caller.display_name or "there"
        if label.lower() == "caller":
            return _EMPLOYER_MODE_GREETING
        return f"{_EMPLOYER_MODE_GREETING} Welcome, {label}."

    def default_menu_key(self) -> str:
        menu = self._menu_config()
        if menu.default_menu_key:
            return menu.default_menu_key
        if menu.menus:
            return menu.menus[0].key
        return "public_main"

    def owner_menu_key(self) -> Optional[str]:
        menu = self._menu_config()
        cleaned = str(menu.owner_menu_key or "").strip()
        return cleaned or None

    def owner_menu_enabled(self) -> bool:
        owner_menu_key = self.owner_menu_key()
        return owner_menu_key is not None and self.menu_by_key(owner_menu_key) is not None

    def menu_by_key(self, key: Optional[str]) -> Optional[PhoneMenuDefinition]:
        cleaned = str(key or "").strip()
        if not cleaned:
            return None
        for menu in self._menu_config().menus:
            if menu.key == cleaned:
                return menu
        return None

    def menu_prompt(self, menu_key: Optional[str]) -> str:
        menu = self.menu_by_key(menu_key)
        if menu is not None and menu.prompt:
            return menu.prompt
        if menu_key == self.default_menu_key():
            return _SCREENING_MENU_PROMPT
        return "Please choose an option."

    def menu_reprompt(self, menu_key: Optional[str]) -> str:
        menu = self.menu_by_key(menu_key)
        if menu is not None and menu.reprompt:
            return menu.reprompt
        if menu_key == self.default_menu_key():
            return _SCREENING_MENU_REPROMPT
        return "Please choose one of the available options."

    def menu_workspace_acceptance(self, caller: PhoneResolvedCaller) -> str:
        option = self.screening_option_by_key(caller.menu_option_key)
        if option is not None and option.greeting:
            return option.greeting
        if caller.menu_option_key == "employer":
            return self.employer_screening_acceptance(caller)
        return (
            "You're connected to the OpenCAS agent on a restricted workspace line. "
            "I can only use the approved information and note-taking space for this call."
        )

    def classify_screening_digit(self, digit: str) -> str | None:
        option = self._screening_option_for_digit(self.default_menu_key(), digit)
        return option.key if option is not None else None

    def classify_screening_transcript(self, transcript: str) -> str | None:
        option = self._screening_option_for_transcript(self.default_menu_key(), transcript)
        return option.key if option is not None else None

    def classify_menu_digit(self, menu_key: Optional[str], digit: str) -> str | None:
        option = self._screening_option_for_digit(menu_key, digit)
        return option.key if option is not None else None

    def classify_menu_transcript(self, menu_key: Optional[str], transcript: str) -> str | None:
        option = self._screening_option_for_transcript(menu_key, transcript)
        return option.key if option is not None else None

    def screening_option_by_key(self, key: Optional[str]) -> Optional[PhoneMenuOption]:
        return self.resolve_menu_option(self.default_menu_key(), key)

    def resolve_menu_option(self, menu_key: Optional[str], choice: str | None) -> Optional[PhoneMenuOption]:
        cleaned = str(choice or "").strip()
        if not cleaned:
            return None
        menu = self.menu_by_key(menu_key)
        if menu is None:
            return None
        for option in menu.options:
            if option.key == cleaned:
                return option
        return None

    def _screening_option_for_digit(self, menu_key: Optional[str], digit: str) -> Optional[PhoneMenuOption]:
        menu = self.menu_by_key(menu_key)
        if menu is None:
            return None
        cleaned = str(digit or "").strip()
        if not cleaned:
            return None
        for option in menu.options:
            if option.digit == cleaned:
                return option
        return None

    def _screening_option_for_transcript(self, menu_key: Optional[str], transcript: str) -> Optional[PhoneMenuOption]:
        menu = self.menu_by_key(menu_key)
        if menu is None:
            return None
        cleaned = " ".join(str(transcript or "").strip().lower().split())
        if not cleaned:
            return None
        spoken_digits = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
        }
        for option in menu.options:
            if cleaned == option.digit or cleaned == {"1": "one", "2": "two", "3": "three"}.get(option.digit, ""):
                return option
            if cleaned == spoken_digits.get(option.digit, ""):
                return option
            if any(phrase in cleaned for phrase in option.phrases):
                return option
        return None

    def owner_pin_required(self) -> bool:
        return bool(self.config.owner_pin)

    def owner_pin_prompt(self) -> str:
        menu = self._menu_config()
        return menu.owner_pin_prompt or "Please enter your six digit owner PIN now."

    def owner_pin_retry_prompt(self) -> str:
        menu = self._menu_config()
        return menu.owner_pin_retry_prompt or "That PIN was not accepted. Please try again."

    def owner_pin_success_message(self) -> str:
        menu = self._menu_config()
        return menu.owner_pin_success_message or "Thanks. You're verified."

    def owner_pin_failure_message(self) -> str:
        menu = self._menu_config()
        return menu.owner_pin_failure_message or "Sorry, I couldn't verify the owner PIN. Goodbye."

    def validate_owner_pin(self, candidate: str) -> bool:
        configured = str(self.config.owner_pin or "").strip()
        if not configured:
            return True
        return secrets.compare_digest(configured, str(candidate or "").strip())

    def resolve_screening_option(self, choice: str | None) -> Optional[PhoneMenuOption]:
        return self.screening_option_by_key(choice)

    async def activate_menu_workspace_caller(
        self,
        *,
        option: PhoneMenuOption,
        caller_number: Optional[str],
        display_name: Optional[str],
    ) -> PhoneResolvedCaller:
        caller = self._menu_workspace_caller(option, caller_number=caller_number, display_name=display_name)
        if caller is None:
            raise RuntimeError("Caller number is missing")
        if any(mount.scope == "shared" and "employer_shared" in mount.subdir for mount in caller.workspace_mounts):
            self._sync_employer_shared_workspace()
        self._ensure_workspace_option_ready(caller)
        return caller

    async def screening_option_announcement(
        self,
        option: PhoneMenuOption,
        *,
        caller_number: Optional[str],
        display_name: Optional[str],
    ) -> str:
        if option.action == "workspace_assistant":
            caller = await self.activate_menu_workspace_caller(
                option=option,
                caller_number=caller_number,
                display_name=display_name,
            )
            return self.menu_workspace_acceptance(caller)
        if option.action == "say_then_hangup":
            return option.message or self.employer_screening_rejection()
        if option.action == "time_announcement":
            return self._time_announcement(option)
        return self.employer_screening_reprompt()

    def _time_announcement(self, option: PhoneMenuOption) -> str:
        time_zone = str(option.time_zone or "").strip()
        if not time_zone:
            return option.message or "Sorry, that time service is not configured."
        try:
            now = datetime.now(ZoneInfo(time_zone))
        except Exception:
            return option.message or f"Sorry, I couldn't load the time for {time_zone}."
        template = option.message_template or option.message or "The current time in {time_zone} is {local_time}."
        return template.format(
            time_zone=time_zone,
            local_time=now.strftime("%A, %B %d at %I:%M %p"),
        )

    def _absolute_url(self, base_url: str, path: str, **query: str) -> str:
        root = str(base_url or "").rstrip("/")
        suffix = path if path.startswith("/") else f"/{path}"
        filtered_query = {key: value for key, value in query.items() if value is not None}
        if not filtered_query:
            return f"{root}{suffix}"
        return f"{root}{suffix}?{urlencode(filtered_query)}"

    @staticmethod
    def _request_query_params(request_url: str) -> Dict[str, str]:
        return {
            str(key): str(value)
            for key, value in parse_qsl(urlsplit(str(request_url or "")).query, keep_blank_values=True)
            if key
        }

    def _stream_bridge_token(self) -> str:
        seed = str(self.config.webhook_secret or "").strip() or "opencas-phone-stream"
        return hashlib.sha256(f"media:{seed}".encode("utf-8")).hexdigest()[:40]

    def _stream_websocket_url(self, webhook_base_url: str) -> str:
        root = str(webhook_base_url or self.config.public_base_url or "").rstrip("/")
        if root.startswith("https://"):
            root = "wss://" + root[len("https://") :]
        elif root.startswith("http://"):
            root = "ws://" + root[len("http://") :]
        return f"{root}/api/phone/twilio/media/{self._stream_bridge_token()}"

    def _stream_connect_twiml(
        self,
        *,
        caller: PhoneResolvedCaller,
        webhook_base_url: str,
        call_token: Optional[str],
        intro_message: str,
        stream_mode: str,
    ) -> str:
        return build_connect_stream_twiml(
            self._stream_websocket_url(webhook_base_url),
            {
                "callerNumber": caller.phone_number,
                "displayName": caller.display_name,
                "callToken": call_token or "",
                "introMessage": intro_message,
                "streamMode": stream_mode,
            },
        )

    def _menu_gather_twiml(
        self,
        *,
        webhook_base_url: str,
        call_token: Optional[str],
        menu_key: str,
        stream_mode: str,
        allow_speech: bool,
        prompt_text: str,
        reprompt_text: str,
    ) -> str:
        gather_url = self._absolute_url(
            webhook_base_url,
            "/api/phone/twilio/gather",
            call_token=call_token,
            bridge_token=self.config.webhook_secret,
            stream_mode=stream_mode,
            menu_key=menu_key,
        )
        input_mode = "dtmf speech" if allow_speech else "dtmf"
        return _xml_response(
            _gather(
                gather_url,
                _say(prompt_text),
                input_mode=input_mode,
                speech_timeout="auto" if allow_speech else None,
                num_digits=1,
                timeout_seconds=5,
            ),
            _say(reprompt_text),
            _hangup(),
        )

    def _owner_pin_gather_twiml(
        self,
        *,
        webhook_base_url: str,
        call_token: Optional[str],
        prompt_text: str,
        retry_text: str,
    ) -> str:
        gather_url = self._absolute_url(
            webhook_base_url,
            "/api/phone/twilio/gather",
            call_token=call_token,
            bridge_token=self.config.webhook_secret,
            stream_mode="owner_pin",
        )
        return _xml_response(
            _gather(
                gather_url,
                _say(prompt_text),
                input_mode="dtmf",
                speech_timeout=None,
                num_digits=6,
                timeout_seconds=8,
            ),
            _say(retry_text),
            _hangup(),
        )

    def _stream_connect_reply_twiml(
        self,
        *,
        caller: PhoneResolvedCaller,
        webhook_base_url: str,
        call_token: Optional[str],
        stream_mode: str,
        preface_text: str = "",
        intro_message: str = "",
    ) -> str:
        verbs: list[str] = []
        if preface_text.strip():
            verbs.append(_say(preface_text.strip()))
        verbs.append(
            self._stream_connect_twiml(
                caller=caller,
                webhook_base_url=webhook_base_url,
                call_token=call_token,
                intro_message=intro_message,
                stream_mode=stream_mode,
            )
        )
        return _xml_response(*verbs)

    async def _voice_reply_twiml(
        self,
        *,
        caller: PhoneResolvedCaller,
        text: str,
        webhook_base_url: str,
        continue_listening: bool,
        call_token: Optional[str],
        expressive: bool,
        prefer_direct_speech: bool = False,
    ) -> str:
        prompt_verb = (
            _say(text)
            if prefer_direct_speech
            else await self._build_voice_prompt_verb(
                text=text,
                webhook_base_url=webhook_base_url,
                expressive=expressive,
            )
        )
        return self._reply_twiml_from_prompt_verb(
            prompt_verb=prompt_verb,
            webhook_base_url=webhook_base_url,
            continue_listening=continue_listening,
            call_token=call_token,
        )

    def _reply_twiml_from_prompt_verb(
        self,
        *,
        prompt_verb: str,
        webhook_base_url: str,
        continue_listening: bool,
        call_token: Optional[str],
    ) -> str:
        if not continue_listening:
            return _xml_response(prompt_verb, _hangup())

        gather_url = self._absolute_url(
            webhook_base_url,
            "/api/phone/twilio/gather",
            call_token=call_token,
            bridge_token=self.config.webhook_secret,
        )
        return _xml_response(
            _gather(gather_url, prompt_verb),
            _say("I didn't catch anything else. Goodbye."),
            _hangup(),
        )

    def validate_media_stream_request(
        self,
        *,
        request_url: str,
        provided_signature: Optional[str],
        stream_secret: str,
    ) -> bool:
        if stream_secret != self._stream_bridge_token():
            return False
        credentials = self._twilio_credentials()
        auth_token = credentials.webhook_auth_token if credentials is not None else None
        if not auth_token:
            return True
        if not provided_signature:
            return False
        validator = RequestValidator(auth_token)
        try:
            return bool(validator.validate(request_url, {}, provided_signature.strip()))
        except Exception:
            return False

    def _pending_reply_twiml(
        self,
        *,
        webhook_base_url: str,
        call_token: Optional[str],
        reply_token: str,
    ) -> str:
        poll_url = self._absolute_url(
            webhook_base_url,
            "/api/phone/twilio/poll",
            call_token=call_token,
            reply_token=reply_token,
            bridge_token=self.config.webhook_secret,
        )
        return _xml_response(_pause(2), _redirect(poll_url))

    async def _build_voice_prompt_verb(
        self,
        *,
        text: str,
        webhook_base_url: str,
        expressive: bool,
    ) -> str:
        try:
            voice_result = await asyncio.wait_for(
                synthesize_speech(
                    chat_upload_dir(self.runtime),
                    text=text,
                    prefer_local=True,
                    expressive=expressive,
                    state_dir=self.runtime.ctx.config.state_dir,
                    config=self.config,
                ),
                timeout=2.5,
            )
            meta = voice_result.to_meta()
            relative_url = str(meta.get("url") or "").strip()
            if relative_url:
                if relative_url.startswith("http://") or relative_url.startswith("https://"):
                    return _play(relative_url)
                return _play(self._absolute_url(webhook_base_url, relative_url))
        except Exception:
            pass
        return _say(text)

    async def _build_background_voice_prompt_verb(
        self,
        *,
        text: str,
        webhook_base_url: str,
        expressive: bool,
    ) -> str:
        try:
            voice_result = await asyncio.wait_for(
                synthesize_speech(
                    chat_upload_dir(self.runtime),
                    text=text,
                    prefer_local=True,
                    expressive=expressive,
                    state_dir=self.runtime.ctx.config.state_dir,
                    config=self.config,
                ),
                timeout=12.0,
            )
            meta = voice_result.to_meta()
            relative_url = str(meta.get("url") or "").strip()
            if relative_url:
                if relative_url.startswith("http://") or relative_url.startswith("https://"):
                    return _play(relative_url)
                return _play(self._absolute_url(webhook_base_url, relative_url))
        except Exception:
            pass
        return _say(text)

    def _start_owner_reply_task(
        self,
        *,
        caller: PhoneResolvedCaller,
        transcript: str,
        call_sid: str,
        webhook_base_url: str,
    ) -> str:
        self._prune_pending_phone_replies()
        token = secrets.token_urlsafe(12)
        task = asyncio.create_task(
            self._generate_owner_reply(
                caller=caller,
                transcript=transcript,
                call_sid=call_sid,
                webhook_base_url=webhook_base_url,
            )
        )
        self._pending_phone_replies[token] = PendingPhoneReply(
            token=token,
            created_at=time.time(),
            call_sid=call_sid,
            caller_number=caller.phone_number,
            task=task,
        )
        return token

    async def _generate_owner_reply(
        self,
        caller: PhoneResolvedCaller,
        transcript: str,
        *,
        call_sid: str,
        webhook_base_url: str,
    ) -> ResolvedPhoneReply:
        fallback = (
            "I'm on the phone line in fast mode. I heard you, but I need to follow up in the main workspace after this call."
        )
        try:
            content = await asyncio.wait_for(
                self.generate_owner_live_reply(
                    caller=caller,
                    transcript=transcript,
                    call_sid=call_sid,
                ),
                timeout=75.0,
            )
        except asyncio.TimeoutError:
            self._trace_owner_reply_issue("timeout", caller=caller, call_sid=call_sid)
            content = fallback
        except Exception:
            self._trace_owner_reply_issue("llm_error", caller=caller, call_sid=call_sid)
            content = fallback

        prompt_verb = await self._build_background_voice_prompt_verb(
            text=content,
            webhook_base_url=webhook_base_url,
            expressive=self.phone_tts_expressive(),
        )
        return ResolvedPhoneReply(text=content, prompt_verb=prompt_verb)

    async def generate_owner_live_reply(
        self,
        *,
        caller: PhoneResolvedCaller,
        transcript: str,
        call_sid: str,
    ) -> str:
        session_id = self._session_id(caller)
        await self._ensure_phone_session(session_id, caller)
        await self._ensure_owner_preamble(session_id, caller)
        fallback = (
            "I hit a problem handling that on the phone line. Ask again in a different way, or switch to the dashboard if you need the technical details."
        )
        user_meta = self._user_meta(caller, call_sid=call_sid)
        user_meta["phone"]["mode"] = "owner_live"
        user_meta["voice_input"] = {
            "provider": "elevenlabs",
            "mode": "phone_stream",
            "model": self.config.elevenlabs_stt_model,
        }
        self.trace_phone_event(
            "phone_owner_reply_started",
            caller=caller,
            call_sid=call_sid,
            transcript_preview=transcript[:120],
            transcript_length=len(transcript),
        )
        try:
            content = await self.runtime.converse(
                transcript,
                session_id=session_id,
                user_meta=user_meta,
            )
        except Exception:
            self._trace_owner_reply_issue("runtime_converse_error", caller=caller, call_sid=call_sid)
            return fallback
        await self._annotate_latest_assistant_phone_meta(
            session_id=session_id,
            caller=caller,
            call_sid=call_sid,
        )
        resolved = str(content or "").strip() or fallback
        self.trace_phone_event(
            "phone_owner_reply_completed",
            caller=caller,
            call_sid=call_sid,
            response_preview=resolved[:120],
            response_length=len(resolved),
        )
        return resolved

    async def generate_owner_live_stream_reply(
        self,
        *,
        caller: PhoneResolvedCaller,
        transcript: str,
        call_sid: str,
    ) -> str:
        fallback = (
            "I’m still here, but the phone line stalled while I was preparing that reply. "
            "Please ask again, or switch to the dashboard if you need a longer answer."
        )
        try:
            return await asyncio.wait_for(
                self.generate_owner_live_reply(
                    caller=caller,
                    transcript=transcript,
                    call_sid=call_sid,
                ),
                timeout=_PHONE_OWNER_STREAM_REPLY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            self._trace_owner_reply_issue("stream_timeout", caller=caller, call_sid=call_sid)
        except Exception:
            self._trace_owner_reply_issue("stream_error", caller=caller, call_sid=call_sid)
        await self._append_phone_assistant_message(
            caller=caller,
            call_sid=call_sid,
            content=fallback,
        )
        return fallback

    async def activate_employer_caller(
        self,
        *,
        caller_number: Optional[str],
        display_name: Optional[str],
    ) -> PhoneResolvedCaller:
        option = self.screening_option_by_key("employer")
        if option is None:
            raise RuntimeError("Employer menu option is not configured")
        return await self.activate_menu_workspace_caller(
            option=option,
            caller_number=caller_number,
            display_name=display_name,
        )

    async def generate_workspace_live_reply(
        self,
        *,
        caller: PhoneResolvedCaller,
        transcript: str,
        call_sid: str,
    ) -> str:
        if any(mount.scope == "shared" and "employer_shared" in mount.subdir for mount in caller.workspace_mounts):
            self._sync_employer_shared_workspace()
        self._ensure_workspace_option_ready(caller)
        self.trace_phone_event(
            "phone_workspace_reply_started",
            caller=caller,
            call_sid=call_sid,
            transcript_preview=transcript[:120],
            transcript_length=len(transcript),
        )
        resolved = await self._respond_low_trust(caller, transcript, call_sid=call_sid)
        self.trace_phone_event(
            "phone_workspace_reply_completed",
            caller=caller,
            call_sid=call_sid,
            response_preview=resolved[:120],
            response_length=len(resolved),
        )
        return resolved

    async def generate_workspace_live_stream_reply(
        self,
        *,
        caller: PhoneResolvedCaller,
        transcript: str,
        call_sid: str,
    ) -> str:
        fallback = (
            "I ran into a delay on the phone line while I was putting that together. "
            "Please ask again in a moment."
        )
        try:
            return await asyncio.wait_for(
                self.generate_workspace_live_reply(
                    caller=caller,
                    transcript=transcript,
                    call_sid=call_sid,
                ),
                timeout=_PHONE_WORKSPACE_STREAM_REPLY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            pass
        except Exception:
            pass
        await self._append_phone_assistant_message(
            caller=caller,
            call_sid=call_sid,
            content=fallback,
        )
        return fallback

    async def generate_employer_live_reply(
        self,
        *,
        caller: PhoneResolvedCaller,
        transcript: str,
        call_sid: str,
    ) -> str:
        return await self.generate_workspace_live_reply(
            caller=caller,
            transcript=transcript,
            call_sid=call_sid,
        )

    async def _respond_low_trust(
        self,
        caller: PhoneResolvedCaller,
        transcript: str,
        *,
        call_sid: str,
    ) -> str:
        session_id = self._session_id(caller)
        await self._ensure_phone_session(session_id, caller)
        history = await self.runtime.ctx.context_store.list_recent(session_id, limit=10)
        if caller.workspace_mounts:
            workspace_roots = self._employer_workspace_roots(caller)
            workspace_knowledge = self._workspace_knowledge_excerpt_multi(workspace_roots)
            if self._is_employer_caller(caller):
                system_prompt = self._build_employer_prompt(
                    caller,
                    shared_workspace_dir=workspace_roots["shared"],
                    caller_workspace_dir=workspace_roots["caller"],
                    workspace_knowledge=workspace_knowledge,
                )
                objective = (
                    "Employer call support request. Use only the approved employer workspace tools. "
                    "Treat phone/employer_shared as the only factual source of truth about the owner. "
                    "Use the caller workspace only for this employer's messages, notes, and follow-up records. "
                    f"Caller request: {transcript}"
                )
            else:
                system_prompt = self._build_workspace_assistant_prompt(
                    caller,
                    workspace_roots=workspace_roots,
                    workspace_knowledge=workspace_knowledge,
                )
                objective = (
                    "Restricted menu workspace support request. Use only the configured workspace tools and honor "
                    "the access mode for each mounted workspace root. "
                    f"Caller request: {transcript}"
                )
            low_trust_registry = self._build_workspace_assistant_tool_registry(
                caller=caller,
                workspace_roots=workspace_roots,
            )
        else:
            workspace_dir = self._contact_workspace_dir(caller)
            workspace_knowledge = self._workspace_knowledge_excerpt(workspace_dir)
            system_prompt = self._build_low_trust_prompt(caller, workspace_dir, workspace_knowledge)
            low_trust_registry = self._build_low_trust_tool_registry(workspace_dir)
            objective = (
                "Caller workspace support request. Use only the caller workspace tools to "
                "search files, read files, and write add-only notes when needed. "
                f"Caller request: {transcript}"
            )
        messages = [{"role": "system", "content": system_prompt}]
        for entry in history:
            if entry.role in {MessageRole.USER, MessageRole.ASSISTANT} and entry.content.strip():
                messages.append({"role": entry.role.value, "content": entry.content})
        messages.append({"role": "user", "content": transcript})
        low_trust_runtime = _LowTrustToolRuntime(self.runtime, low_trust_registry)
        low_trust_loop = ToolUseLoop(
            llm=self.runtime.llm,
            tools=low_trust_registry,
            approval=self.runtime.approval,
            tracer=getattr(self.runtime, "tracer", None),
        )
        result = await low_trust_loop.run(
            objective=objective,
            messages=messages,
            ctx=ToolUseContext(
                runtime=low_trust_runtime,
                session_id=session_id,
                max_iterations=8,
            ),
            payload={"temperature": 0.1},
        )
        content = result.final_output or "I don't have enough approved information to answer that on this line."
        await self.runtime.ctx.context_store.append(
            session_id,
            MessageRole.USER,
            transcript,
            meta=self._user_meta(caller, call_sid=call_sid),
        )
        await self.runtime.ctx.context_store.append(
            session_id,
            MessageRole.ASSISTANT,
            content,
            meta=build_assistant_message_meta(
                self.runtime,
                extra={"phone": self._assistant_meta(caller, call_sid=call_sid)},
            ),
        )
        await self._record_episode_safe(transcript, session_id=session_id, role="user")
        await self._record_episode_safe(content, session_id=session_id, role="assistant")
        return content

    async def finalize_employer_call(
        self,
        *,
        caller: PhoneResolvedCaller,
        call_sid: str,
        caller_audio_mulaw: bytes | None = None,
    ) -> None:
        session_id = self._session_id(caller)
        history = await self.runtime.ctx.context_store.list_recent(
            session_id,
            limit=40,
            include_hidden=True,
        )
        visible = [
            entry
            for entry in history
            if entry.role in {MessageRole.USER, MessageRole.ASSISTANT} and str(entry.content or "").strip()
        ]
        if not any(entry.role == MessageRole.USER for entry in visible):
            return

        transcript_text = self._format_employer_call_transcript(history=visible)
        summary_text = await self._summarize_employer_call(caller=caller, history=visible, call_sid=call_sid)
        employer_workspace = self._contact_workspace_dir(caller)
        call_artifacts = self._write_employer_call_artifacts(
            caller=caller,
            call_sid=call_sid,
            employer_workspace=employer_workspace,
            transcript_text=transcript_text,
            summary_text=summary_text,
            caller_audio_mulaw=caller_audio_mulaw,
        )
        self._append_workspace_note(
            employer_workspace / "call-summary.md",
            summary_text,
        )
        self._append_workspace_note(
            self._owner_workspace_dir() / "employer-call-summaries.md",
            summary_text,
        )
        self._append_workspace_note(
            employer_workspace / "messages.md",
            summary_text,
        )
        await self._record_episode_safe(
            summary_text,
            session_id=self._owner_session_id(),
            role="system",
        )
        await self._notify_owner_about_employer_call(
            caller=caller,
            call_sid=call_sid,
            summary_text=summary_text,
            transcript_text=transcript_text,
            audio_path=call_artifacts.get("audio_path"),
        )

    async def _persist_voicemail(
        self,
        caller: PhoneResolvedCaller,
        transcript: str,
        *,
        call_sid: str,
    ) -> None:
        session_id = self._session_id(caller)
        await self._ensure_phone_session(session_id, caller)
        meta = self._user_meta(caller, call_sid=call_sid)
        meta["phone"]["mode"] = "voicemail"
        await self.runtime.ctx.context_store.append(
            session_id,
            MessageRole.USER,
            transcript,
            meta=meta,
        )
        await self._record_episode_safe(transcript, session_id=session_id, role="user")

    def _sync_employer_shared_workspace(self) -> None:
        shared_seed_dir = self._employment_shared_seed_dir()
        shared_workspace_dir = self._employer_shared_workspace_dir()
        copied = False
        for source in sorted(shared_seed_dir.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(shared_seed_dir)
            destination = shared_workspace_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                try:
                    if destination.read_bytes() == source.read_bytes():
                        copied = True
                        continue
                except Exception:
                    pass
            shutil.copyfile(source, destination)
            copied = True
        if copied:
            return
        fallback_user = shared_workspace_dir / "user.md"
        if not fallback_user.exists():
            fallback_user.write_text(
                "No employer-safe user profile has been seeded yet.\n",
                encoding="utf-8",
            )

    def _ensure_workspace_option_ready(self, caller: PhoneResolvedCaller) -> None:
        roots = self._workspace_mount_roots(caller)
        caller_root = roots.get("caller")
        if caller_root is None:
            return
        caller_root.mkdir(parents=True, exist_ok=True)
        notes_path = caller_root / "messages.md"
        if not notes_path.exists():
            notes_path.write_text(
                "# Caller Notes\n",
                encoding="utf-8",
            )

    async def _ensure_phone_session(self, session_id: str, caller: PhoneResolvedCaller) -> None:
        store = self.runtime.ctx.context_store
        await store.ensure_session(session_id)
        await store.update_session_name(
            session_id,
            f"Phone: {caller.display_name} ({caller.phone_number})",
        )

    async def _ensure_owner_preamble(self, session_id: str, caller: PhoneResolvedCaller) -> None:
        store = self.runtime.ctx.context_store
        recent = await store.list_recent(session_id, limit=20, include_hidden=True)
        if any(entry.role == MessageRole.SYSTEM and entry.meta.get("phone_owner_preamble") for entry in recent):
            return
        workspace_dir = self._contact_workspace_dir(caller)
        await store.append(
            session_id,
            MessageRole.SYSTEM,
            "\n".join(
                [
                    self._read_phone_prompt_profile("owner_live"),
                    f"Trusted owner caller number: {caller.phone_number}.",
                    f"Owner workspace: {workspace_dir}.",
                    "This directive applies only to phone turns in this session.",
                ]
            ),
            meta={"phone_owner_preamble": True},
        )

    def _is_employer_caller(self, caller: PhoneResolvedCaller) -> bool:
        if caller.menu_option_key == "employer":
            return True
        subdir = str(caller.workspace_subdir or "").replace("\\", "/").strip("/")
        return subdir.startswith(f"{_EMPLOYER_WORKSPACE_ROOT}/")

    def _employer_workspace_roots(self, caller: PhoneResolvedCaller) -> Dict[str, Path]:
        return self._workspace_mount_roots(caller)

    def _workspace_mount_roots(self, caller: PhoneResolvedCaller) -> Dict[str, Path]:
        roots: Dict[str, Path] = {}
        if not caller.workspace_mounts:
            roots["caller"] = self._contact_workspace_dir(caller)
            return roots
        managed_root = Path(self.runtime.ctx.config.agent_workspace_root()).resolve()
        for mount in caller.workspace_mounts:
            subdir = self._render_workspace_subdir(mount.subdir, caller.phone_number)
            candidate = (managed_root / subdir).resolve()
            try:
                candidate.relative_to(managed_root)
            except ValueError as exc:
                raise RuntimeError("workspace mount escaped managed workspace root") from exc
            candidate.mkdir(parents=True, exist_ok=True)
            roots[mount.scope] = candidate
        return roots

    def _read_phone_prompt_profile(self, profile_name: Optional[str]) -> str:
        config_dir = self._phone_profile_dir(profile_name)
        snippets: list[str] = []
        for name in ("system.md", "style.md", "rules.md"):
            path = config_dir / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                snippets.append(text)
        if snippets:
            return "\n\n".join(snippets)
        return (
            "You are the OpenCAS work-safe phone agent, a professional employer-facing phone agent. "
            "You only know the employer-safe facts available in the approved workspace and the current call."
        )

    def _build_employer_prompt(
        self,
        caller: PhoneResolvedCaller,
        *,
        shared_workspace_dir: Path,
        caller_workspace_dir: Path,
        workspace_knowledge: str,
    ) -> str:
        parts = [
            self._read_phone_prompt_profile(caller.prompt_profile),
            "You are on an employment-focused phone line.",
            "The only factual source of truth about the owner is the employer shared workspace and the current call.",
            "Do not use or imply any hidden memory, owner context, or personal knowledge outside the approved employer workspace.",
            "Treat the caller workspace as add-only notes and follow-up records for this specific employer line.",
            "If the approved workspace does not support a claim, say you do not have that information on this line.",
            "You may answer employment-related questions about the owner and take messages for follow-up.",
            "You must not discuss personal life, criminal history, health, finances, family, or anything not explicitly contained in the employer shared workspace.",
            "Keep spoken replies under 90 words and sound professional, calm, and direct.",
            f"Caller: {caller.display_name} ({caller.phone_number}).",
            f"Employer shared workspace: {shared_workspace_dir}.",
            f"Employer caller workspace: {caller_workspace_dir}.",
            "Approved workspace notes:",
            workspace_knowledge or "- No approved employer workspace notes are available yet.",
        ]
        return "\n".join(parts)

    def _build_low_trust_prompt(
        self,
        caller: PhoneResolvedCaller,
        workspace_dir: Path,
        workspace_knowledge: str,
    ) -> str:
        parts = [
            "You are the OpenCAS agent answering a low-trust caller on a phone line managed by OpenCAS.",
            "Do not use any tools other than the bounded caller workspace toolset, and do not expose private operator data.",
            "Only answer from the approved workspace notes below and the live conversation history.",
            "If the answer is not supported by those notes, say you do not have that information on this line.",
            "Keep spoken replies under 90 words and avoid long lists.",
            f"Caller: {caller.display_name} ({caller.phone_number}).",
            f"Approved actions: {', '.join(caller.allowed_actions) or 'none'}.",
            f"Approved workspace: {workspace_dir}.",
        ]
        if caller.notes:
            parts.append(f"Caller notes: {caller.notes}")
        parts.extend(
            [
                "You have a bounded caller workspace toolset only.",
                "You may read, list, and search files inside the approved workspace.",
                "You may add new note content inside that workspace, but you must not overwrite, delete, rename, or move files.",
                "You must not execute code, browse the web, call external services, or act outside this caller workspace.",
                "If routing to the main operator is needed, say you will record the request in this caller workspace unless the call is owner-trusted.",
            ]
        )
        parts.append("Approved workspace notes:")
        parts.append(workspace_knowledge or "- No workspace notes are available yet.")
        return "\n".join(parts)

    def _build_low_trust_tool_registry(self, workspace_dir: Path) -> ToolRegistry:
        root = str(workspace_dir)
        registry = ToolRegistry(tracer=getattr(self.runtime, "tracer", None))
        fs_adapter = FileSystemToolAdapter(allowed_roots=[root])
        search_adapter = SearchToolAdapter(allowed_roots=[root])
        add_only_adapter = AddOnlyFileWriteToolAdapter(allowed_roots=[root])

        registry.register(
            "fs_read_file",
            "Read a file inside the caller workspace. The path must stay inside the approved caller workspace.",
            fs_adapter,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": f"Absolute path inside the caller workspace root {workspace_dir}.",
                    }
                },
                "required": ["file_path"],
            },
        )
        registry.register(
            "fs_list_dir",
            "List a directory inside the caller workspace. The path must stay inside the approved caller workspace.",
            fs_adapter,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": f"Absolute path inside the caller workspace root {workspace_dir}.",
                    }
                },
                "required": ["dir_path"],
            },
        )
        registry.register(
            "grep_search",
            "Search text inside the caller workspace only.",
            search_adapter,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex or plain-text pattern to search for."},
                    "path": {
                        "type": "string",
                        "description": f"Optional absolute path inside the caller workspace root {workspace_dir}.",
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches"],
                    },
                },
                "required": ["pattern"],
            },
        )
        registry.register(
            "glob_search",
            "Find files inside the caller workspace only.",
            search_adapter,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern such as **/*.md."},
                    "path": {
                        "type": "string",
                        "description": f"Optional absolute path inside the caller workspace root {workspace_dir}.",
                    },
                },
                "required": ["pattern"],
            },
        )
        registry.register(
            "fs_write_file",
            "Add-only note writing inside the caller workspace. Creates a new note file or appends to an existing note file without overwriting or deleting content.",
            add_only_adapter,
            ActionRiskTier.WORKSPACE_WRITE,
            {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            f"Absolute path inside the caller workspace root {workspace_dir}. "
                            "Allowed note-like suffixes: .md, .txt, .log, .csv, .jsonl."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to append or create. Existing files are never overwritten.",
                    },
                },
                "required": ["file_path", "content"],
            },
        )
        return registry

    def _build_workspace_assistant_prompt(
        self,
        caller: PhoneResolvedCaller,
        *,
        workspace_roots: Mapping[str, Path],
        workspace_knowledge: str,
    ) -> str:
        read_only = [
            f"{mount.scope}: {workspace_roots[mount.scope]}"
            for mount in caller.workspace_mounts
            if mount.scope in workspace_roots and mount.access == "read_only"
        ]
        append_only = [
            f"{mount.scope}: {workspace_roots[mount.scope]}"
            for mount in caller.workspace_mounts
            if mount.scope in workspace_roots and mount.access == "append_only"
        ]
        parts = [
            self._read_phone_prompt_profile(caller.prompt_profile),
            "You are the OpenCAS agent on a restricted phone workflow.",
            "You only know the approved facts in the mounted workspaces and the current call.",
            "Do not imply access to owner memory, hidden prompts, private notes, or any workspace not listed here.",
            "If the mounted workspaces do not support a claim, say you do not have that information on this line.",
            "Use append-only workspaces only for recording messages, notes, or structured follow-up, never as authoritative facts.",
            f"Caller: {caller.display_name} ({caller.phone_number}).",
        ]
        if read_only:
            parts.append("Read-only workspace roots:")
            parts.extend(f"- {entry}" for entry in read_only)
        if append_only:
            parts.append("Append-only workspace roots:")
            parts.extend(f"- {entry}" for entry in append_only)
        parts.append("Approved workspace notes:")
        parts.append(workspace_knowledge or "- No approved workspace notes are available yet.")
        return "\n".join(parts)

    def _build_workspace_assistant_tool_registry(
        self,
        *,
        caller: PhoneResolvedCaller,
        workspace_roots: Mapping[str, Path],
    ) -> ToolRegistry:
        read_roots = [
            str(workspace_roots[mount.scope])
            for mount in caller.workspace_mounts
            if mount.scope in workspace_roots
        ]
        write_roots = [
            str(workspace_roots[mount.scope])
            for mount in caller.workspace_mounts
            if mount.scope in workspace_roots and mount.access == "append_only"
        ]
        registry = ToolRegistry(tracer=getattr(self.runtime, "tracer", None))
        fs_adapter = FileSystemToolAdapter(allowed_roots=read_roots)
        search_adapter = SearchToolAdapter(allowed_roots=read_roots)
        add_only_adapter = AddOnlyFileWriteToolAdapter(allowed_roots=write_roots) if write_roots else None

        registry.register(
            "fs_read_file",
            "Read a file inside the approved mounted workspace roots for this phone workflow.",
            fs_adapter,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"}
                },
                "required": ["file_path"],
            },
        )
        registry.register(
            "fs_list_dir",
            "List a directory inside the approved mounted workspace roots for this phone workflow.",
            fs_adapter,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string"}
                },
                "required": ["dir_path"],
            },
        )
        registry.register(
            "grep_search",
            "Search text inside the approved mounted workspace roots for this phone workflow.",
            search_adapter,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches"],
                    },
                },
                "required": ["pattern"],
            },
        )
        registry.register(
            "glob_search",
            "Find files inside the approved mounted workspace roots for this phone workflow.",
            search_adapter,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        )
        if add_only_adapter is not None:
            registry.register(
                "fs_write_file",
                "Add-only note writing inside append-only mounted workspace roots. Never overwrite or delete existing content.",
                add_only_adapter,
                ActionRiskTier.WORKSPACE_WRITE,
                {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["file_path", "content"],
                },
            )
        return registry

    def _build_employer_tool_registry(
        self,
        *,
        shared_workspace_dir: Path,
        caller_workspace_dir: Path,
    ) -> ToolRegistry:
        registry = ToolRegistry(tracer=getattr(self.runtime, "tracer", None))
        fs_adapter = FileSystemToolAdapter(allowed_roots=[str(shared_workspace_dir), str(caller_workspace_dir)])
        search_adapter = SearchToolAdapter(allowed_roots=[str(shared_workspace_dir), str(caller_workspace_dir)])
        add_only_adapter = AddOnlyFileWriteToolAdapter(allowed_roots=[str(caller_workspace_dir)])

        registry.register(
            "fs_read_file",
            "Read a file inside the approved employer shared workspace or the caller-specific employer workspace.",
            fs_adapter,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        )
        registry.register(
            "fs_list_dir",
            "List a directory inside the approved employer shared workspace or the caller-specific employer workspace.",
            fs_adapter,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {"dir_path": {"type": "string"}},
                "required": ["dir_path"],
            },
        )
        registry.register(
            "grep_search",
            "Search text inside the approved employer shared workspace or the caller-specific employer workspace.",
            search_adapter,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "output_mode": {"type": "string", "enum": ["content", "files_with_matches"]},
                },
                "required": ["pattern"],
            },
        )
        registry.register(
            "glob_search",
            "Find files inside the approved employer shared workspace or the caller-specific employer workspace.",
            search_adapter,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        )
        registry.register(
            "fs_write_file",
            "Add-only note writing inside the caller-specific employer workspace. Never write into the shared employer facts workspace.",
            add_only_adapter,
            ActionRiskTier.WORKSPACE_WRITE,
            {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
            },
        )
        return registry

    def _workspace_knowledge_excerpt(self, workspace_dir: Path, *, max_chars: int = 12000, max_files: int = 8) -> str:
        snippets: list[str] = []
        total_chars = 0
        for path in sorted(workspace_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            rel = path.relative_to(workspace_dir)
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            remaining = max_chars - total_chars
            if remaining <= 0 or len(snippets) >= max_files:
                break
            excerpt = text[:remaining]
            total_chars += len(excerpt)
            if len(excerpt) < len(text):
                excerpt += "\n[truncated]"
            snippets.append(f"File: {rel}\n{excerpt}")
        return "\n\n".join(snippets)

    def _workspace_knowledge_excerpt_multi(
        self,
        workspace_roots: Mapping[str, Path],
        *,
        max_chars: int = 12000,
        max_files: int = 10,
    ) -> str:
        snippets: list[str] = []
        total_chars = 0
        file_count = 0
        for label, workspace_dir in workspace_roots.items():
            for path in sorted(workspace_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
                    continue
                remaining = max_chars - total_chars
                if remaining <= 0 or file_count >= max_files:
                    return "\n\n".join(snippets)
                rel = path.relative_to(workspace_dir)
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
                if not text:
                    continue
                excerpt = text[:remaining]
                total_chars += len(excerpt)
                file_count += 1
                if len(excerpt) < len(text):
                    excerpt += "\n[truncated]"
                snippets.append(f"Root: {label}\nFile: {rel}\n{excerpt}")
        return "\n\n".join(snippets)

    async def _summarize_employer_call(
        self,
        *,
        caller: PhoneResolvedCaller,
        history: list[Any],
        call_sid: str,
    ) -> str:
        transcript_text = self._format_employer_call_transcript(history=history)
        fallback = (
            f"Employer call from {caller.display_name} ({caller.phone_number})\n"
            f"Call SID: {call_sid}\n"
            f"Transcript:\n{transcript_text}\n"
        )
        if not transcript_text.strip():
            return fallback
        try:
            response = await self.runtime.llm.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize an employment-related phone call for the owner. "
                            "Stay factual, concise, and grounded only in the transcript. "
                            "Include the caller, apparent opportunity or intent, notable questions, "
                            "and any requested follow-up."
                        ),
                    },
                    {"role": "user", "content": transcript_text},
                ],
                complexity="light",
                payload={"temperature": 0.1, "max_tokens": 220},
                source="phone_employer_summary",
                session_id=self._session_id(caller),
                execution_mode="phone_employer_summary",
            )
            content = self._extract_response_text(response).strip()
        except Exception:
            content = ""
        summary_body = content or transcript_text
        return (
            f"Employer call from {caller.display_name} ({caller.phone_number})\n"
            f"Call SID: {call_sid}\n"
            f"{summary_body}\n"
        )

    @staticmethod
    def _format_employer_call_transcript(*, history: list[Any]) -> str:
        transcript_lines: list[str] = []
        for entry in history:
            role_label = "Caller" if entry.role == MessageRole.USER else "the OpenCAS agent"
            content = str(entry.content or "").strip()
            if not content:
                continue
            transcript_lines.append(f"{role_label}: {content}")
        return "\n".join(transcript_lines)

    def _write_employer_call_artifacts(
        self,
        *,
        caller: PhoneResolvedCaller,
        call_sid: str,
        employer_workspace: Path,
        transcript_text: str,
        summary_text: str,
        caller_audio_mulaw: bytes | None,
    ) -> Dict[str, Any]:
        call_dir = employer_workspace / "calls" / self._safe_call_dir_name(call_sid)
        call_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = call_dir / "transcript.md"
        summary_path = call_dir / "summary.md"
        transcript_path.write_text(
            (
                f"# Employer Call Transcript\n\n"
                f"- Caller: {caller.display_name} ({caller.phone_number})\n"
                f"- Call SID: {call_sid}\n\n"
                f"## Transcript\n\n{transcript_text.strip() or '(no transcript)'}\n"
            ),
            encoding="utf-8",
        )
        summary_path.write_text(summary_text.strip() + "\n", encoding="utf-8")
        audio_path = self._store_employer_call_audio_artifact(
            call_dir=call_dir,
            caller_audio_mulaw=caller_audio_mulaw,
        )
        return {
            "call_dir": call_dir,
            "transcript_path": transcript_path,
            "summary_path": summary_path,
            "audio_path": audio_path,
        }

    @staticmethod
    def _safe_call_dir_name(call_sid: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(call_sid or "").strip())
        return cleaned or f"call-{int(time.time())}"

    def _store_employer_call_audio_artifact(
        self,
        *,
        call_dir: Path,
        caller_audio_mulaw: bytes | None,
    ) -> Optional[Path]:
        payload = bytes(caller_audio_mulaw or b"")
        if not payload:
            return None
        wav_bytes = mulaw_to_wav_bytes(payload)
        try:
            mp3_bytes = self._wav_bytes_to_mp3(wav_bytes)
        except Exception:
            return None
        target = call_dir / "caller-message.mp3"
        target.write_bytes(mp3_bytes)
        return target

    @staticmethod
    def _wav_bytes_to_mp3(wav_bytes: bytes) -> bytes:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                "pipe:0",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "128k",
                "-f",
                "mp3",
                "pipe:1",
            ],
            input=wav_bytes,
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(stderr or "ffmpeg mp3 transcode failed")
        return bytes(completed.stdout)

    def _append_workspace_note(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cleaned = str(content or "").strip()
        if not cleaned:
            return
        with path.open("a", encoding="utf-8") as handle:
            if path.stat().st_size > 0:
                handle.write("\n\n")
            handle.write(cleaned)
            handle.write("\n")

    async def _notify_owner_about_employer_call(
        self,
        *,
        caller: PhoneResolvedCaller,
        call_sid: str,
        summary_text: str,
        transcript_text: str,
        audio_path: Optional[Path],
    ) -> None:
        telegram = getattr(self.runtime, "_telegram", None)
        if telegram is None or not hasattr(telegram, "notify_owner"):
            return
        caller_text = transcript_text.strip() or "No caller transcript captured."
        notification = (
            f"the OpenCAS agent received an employment inquiry call.\n"
            f"Caller: {caller.display_name} ({caller.phone_number})\n"
            f"Call SID: {call_sid}\n\n"
            f"Summary:\n{summary_text.strip()}\n\n"
            f"Caller transcript:\n{caller_text}"
        )
        try:
            await telegram.notify_owner(
                notification,
                document_path=audio_path,
                document_filename=audio_path.name if audio_path is not None else None,
                document_caption=(
                    f"Employer call audio: {caller.display_name} ({caller.phone_number})"
                    if audio_path is not None
                    else None
                ),
            )
        except Exception:
            self._trace_owner_reply_issue(
                "telegram_notification_failed",
                caller=caller,
                call_sid=call_sid,
            )

    def _owner_session_id(self) -> str:
        owner_number = self.config.owner_phone_number or "owner"
        return f"phone:{owner_number}"

    def phone_tts_expressive(self) -> bool:
        return str(self.config.phone_tts_mode or "fast").strip().lower() == "expressive"

    def _user_meta(self, caller: PhoneResolvedCaller, *, call_sid: str) -> Dict[str, Any]:
        return {
            "phone": {
                "channel": "twilio_voice",
                "caller_number": caller.phone_number,
                "display_name": caller.display_name,
                "trust_level": caller.trust_level,
                "allowed_actions": list(caller.allowed_actions),
                "call_sid": call_sid,
            }
        }

    def _assistant_meta(self, caller: PhoneResolvedCaller, *, call_sid: str) -> Dict[str, Any]:
        return {
            "channel": "twilio_voice",
            "caller_number": caller.phone_number,
            "display_name": caller.display_name,
            "trust_level": caller.trust_level,
            "call_sid": call_sid,
        }

    async def _annotate_latest_assistant_phone_meta(
        self,
        *,
        session_id: str,
        caller: PhoneResolvedCaller,
        call_sid: str,
    ) -> None:
        store = getattr(self.runtime.ctx, "context_store", None)
        if store is None or not hasattr(store, "list_recent") or not hasattr(store, "merge_message_meta"):
            return
        entries = await store.list_recent(session_id, limit=8, include_hidden=True)
        for entry in reversed(entries):
            if entry.role != MessageRole.ASSISTANT:
                continue
            message_id = getattr(entry, "message_id", None)
            if message_id is None:
                continue
            await store.merge_message_meta(
                session_id,
                message_id,
                build_assistant_message_meta(
                    self.runtime,
                    extra={"phone": self._assistant_meta(caller, call_sid=call_sid)},
                ),
            )
            return

    async def _append_phone_assistant_message(
        self,
        *,
        caller: PhoneResolvedCaller,
        call_sid: str,
        content: str,
    ) -> None:
        text = str(content or "").strip()
        if not text:
            return
        store = getattr(self.runtime.ctx, "context_store", None)
        if store is None or not hasattr(store, "append"):
            return
        session_id = self._session_id(caller)
        if hasattr(store, "list_recent"):
            entries = await store.list_recent(session_id, limit=4, include_hidden=True)
            for entry in reversed(entries):
                if entry.role != MessageRole.ASSISTANT:
                    continue
                meta = getattr(entry, "meta", {}) or {}
                phone_meta = meta.get("phone", {}) if isinstance(meta, dict) else {}
                if (
                    phone_meta.get("call_sid") == call_sid
                    and str(getattr(entry, "content", "") or "").strip() == text
                ):
                    return
        await store.append(
            session_id,
            MessageRole.ASSISTANT,
            text,
            meta=build_assistant_message_meta(
                self.runtime,
                extra={"phone": self._assistant_meta(caller, call_sid=call_sid)},
            ),
        )
        await self._record_episode_safe(text, session_id=session_id, role="assistant")

    async def _record_episode_safe(
        self,
        content: str,
        *,
        session_id: str,
        role: str,
    ) -> None:
        record = getattr(self.runtime, "_record_episode", None)
        if record is None:
            return
        try:
            await record(content, EpisodeKind.TURN, session_id=session_id, role=role)
        except Exception:
            pass

    def _create_outbound_context(self, *, intro_message: str, reason: str) -> OutboundCallContext:
        self._prune_outbound_contexts()
        token = secrets.token_urlsafe(12)
        context = OutboundCallContext(
            token=token,
            created_at=time.time(),
            intro_message=intro_message,
            reason=reason,
        )
        self._outbound_contexts[token] = context
        return context

    def _get_outbound_context(self, token: Optional[str]) -> Optional[OutboundCallContext]:
        if not token:
            return None
        self._prune_outbound_contexts()
        return self._outbound_contexts.get(token)

    def _mark_outbound_context_consumed(self, token: str) -> None:
        context = self._outbound_contexts.get(token)
        if context is not None:
            context.consumed = True

    def _prune_outbound_contexts(self, *, ttl_seconds: int = 3600) -> None:
        cutoff = time.time() - ttl_seconds
        stale = [
            key
            for key, item in self._outbound_contexts.items()
            if item.created_at < cutoff
        ]
        for key in stale:
            self._outbound_contexts.pop(key, None)

    def _get_pending_phone_reply(self, token: Optional[str]) -> Optional[PendingPhoneReply]:
        if not token:
            return None
        self._prune_pending_phone_replies()
        return self._pending_phone_replies.get(token)

    def _pop_pending_phone_reply(self, token: str) -> Optional[PendingPhoneReply]:
        return self._pending_phone_replies.pop(token, None)

    def _prune_pending_phone_replies(self, *, ttl_seconds: int = 300) -> None:
        cutoff = time.time() - ttl_seconds
        stale: list[str] = []
        for key, item in self._pending_phone_replies.items():
            if item.created_at < cutoff:
                if not item.task.done():
                    item.task.cancel()
                stale.append(key)
        for key in stale:
            self._pending_phone_replies.pop(key, None)

    def _trace_signature_failure(
        self,
        *,
        reason: str,
        request_url: str,
        form_data: Mapping[str, Any],
    ) -> None:
        trace = getattr(self.runtime, "_trace", None)
        if not callable(trace):
            return
        try:
            repeated_keys = self._repeated_form_keys(form_data)
            trace(
                "phone_webhook_signature_invalid",
                {
                    "reason": reason,
                    "request_url": request_url,
                    "public_base_url": self.config.public_base_url,
                    "webhook_secret_configured": bool(self.config.webhook_secret),
                    "param_keys": sorted(str(key) for key in form_data.keys()),
                    "repeated_param_keys": repeated_keys,
                    "has_from": bool(form_data.get("From")),
                    "has_to": bool(form_data.get("To")),
                    "has_call_sid": bool(form_data.get("CallSid")),
                },
            )
        except Exception:
            pass

    def validate_webhook_request(
        self,
        *,
        request_url: str,
        form_data: Mapping[str, Any],
        provided_signature: Optional[str],
        bridge_token: Optional[str],
    ) -> bool:
        if self.config.webhook_secret:
            if not bridge_token or not secrets.compare_digest(str(bridge_token), str(self.config.webhook_secret)):
                self._trace_signature_failure(
                    reason="bridge_token_mismatch",
                    request_url=request_url,
                    form_data=form_data,
                )
                return False

        if self.config.webhook_signature_required:
            credentials = self._twilio_credentials()
            if credentials is not None and credentials.webhook_auth_token:
                return self.validate_webhook_signature(
                    request_url=request_url,
                    form_data=form_data,
                    provided_signature=provided_signature,
                )
            return bool(self.config.webhook_secret)

        return True

    def _trace_webhook_decision(
        self,
        webhook_kind: str,
        decision: str,
        *,
        caller: PhoneResolvedCaller | None = None,
        form_data: Mapping[str, Any],
    ) -> None:
        trace = getattr(self.runtime, "_trace", None)
        if not callable(trace):
            return
        try:
            trace(
                f"phone_{webhook_kind}_webhook_{decision}",
                {
                    "caller_number": caller.phone_number if caller is not None else normalize_phone_number(form_data.get("From")),
                    "to_number": normalize_phone_number(form_data.get("To")),
                    "call_sid": str(form_data.get("CallSid") or "").strip() or None,
                    "repeated_param_keys": self._repeated_form_keys(form_data),
                },
            )
        except Exception:
            pass

    def _trace_owner_reply_issue(
        self,
        reason: str,
        *,
        caller: PhoneResolvedCaller,
        call_sid: str,
    ) -> None:
        trace = getattr(self.runtime, "_trace", None)
        if not callable(trace):
            return
        try:
            trace(
                "phone_owner_reply_issue",
                {
                    "reason": reason,
                    "caller_number": caller.phone_number,
                    "call_sid": call_sid,
                },
            )
        except Exception:
            pass

    def _repeated_form_keys(self, form_data: Mapping[str, Any]) -> list[str]:
        getlist = getattr(form_data, "getlist", None)
        getall = getattr(form_data, "getall", None)
        repeated: list[str] = []
        for key in form_data.keys():
            values: list[Any] | None = None
            if callable(getall):
                try:
                    values = list(getall(key))
                except Exception:
                    values = None
            if values is None and callable(getlist):
                try:
                    values = list(getlist(key))
                except Exception:
                    values = None
            if values is not None and len(values) > 1:
                repeated.append(str(key))
        return sorted(set(repeated))

    @staticmethod
    def _extract_response_text(response: Mapping[str, Any]) -> str:
        choices = response.get("choices") if isinstance(response, Mapping) else None
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
            if isinstance(message, Mapping):
                return str(message.get("content") or "")
        return ""
