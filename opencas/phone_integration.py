"""Twilio-backed phone bridge for OpenCAS."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
from urllib.parse import urlencode
from xml.sax.saxutils import escape

import httpx
from twilio.request_validator import RequestValidator

from opencas.api.chat_service import chat_upload_dir
from opencas.api.voice_service import synthesize_speech
from opencas.autonomy.models import ActionRiskTier
from opencas.context.models import MessageRole
from opencas.memory import EpisodeKind
from opencas.phone_config import (
    PhoneAllowedAction,
    PhoneRuntimeConfig,
    normalize_phone_number,
)
from opencas.phone_streaming import (
    OwnerPhoneMediaStreamSession,
    build_connect_stream_twiml,
)
from opencas.runtime.lane_metadata import build_assistant_message_meta
from opencas.tools import ToolRegistry, ToolUseContext, ToolUseLoop
from opencas.tools.adapters.add_only_write import AddOnlyFileWriteToolAdapter
from opencas.tools.adapters.fs import FileSystemToolAdapter
from opencas.tools.adapters.search import SearchToolAdapter

_TWILIO_ENV_PATH = Path(
    os.environ.get("OPENCAS_TWILIO_ENV_FILE")
    or os.environ.get("TWILIO_ENV_FILE")
    or (Path.home() / ".opencasenv" / ".twilio")
).expanduser()
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

    async def execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
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


def _extract_env_value(keys: Iterable[str], path: Path = _TWILIO_ENV_PATH) -> Optional[str]:
    import os

    for key in keys:
        direct = os.environ.get(key)
        if direct and str(direct).strip():
            return str(direct).strip().strip('"').strip("'")
    if not path.exists():
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


def _gather(action_url: str, prompt_verb: str) -> str:
    return (
        f'<Gather input="speech" action="{escape(action_url)}" method="POST" '
        'speechTimeout="auto" language="en-US">'
        f"{prompt_verb}</Gather>"
    )


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

    def status(self) -> Dict[str, Any]:
        webhook_base = (self.config.public_base_url or "").rstrip("/")
        urls = {
            "voice": f"{webhook_base}/api/phone/twilio/voice" if webhook_base else None,
            "gather": f"{webhook_base}/api/phone/twilio/gather" if webhook_base else None,
            "poll": f"{webhook_base}/api/phone/twilio/poll" if webhook_base else None,
        }
        return {
            **self.config.redacted_dict(),
            "twilio_credentials_configured": self._twilio_credentials() is not None,
            "webhook_urls": urls,
            "contact_count": len(self.config.contacts),
        }

    async def autoconfigure_twilio(
        self,
        *,
        enabled: Optional[bool] = None,
        public_base_url: Optional[str] = None,
        webhook_signature_required: Optional[bool] = None,
        webhook_secret: Optional[str] = None,
        twilio_from_number: Optional[str] = None,
        owner_phone_number: Optional[str] = None,
        owner_display_name: Optional[str] = None,
        owner_workspace_subdir: Optional[str] = None,
    ) -> Dict[str, Any]:
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
            twilio_from_number=updated_number["phone_number"],
            owner_phone_number=owner_phone_number or self.config.owner_phone_number,
            owner_display_name=owner_display_name or self.config.owner_display_name,
            owner_workspace_subdir=owner_workspace_subdir or self.config.owner_workspace_subdir,
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
        intro = (message or "").strip() or "Hi, it's your OpenCAS assistant calling."
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
            return _xml_response(_say("The phone bridge is currently offline."), _hangup())

        caller = self._resolve_caller(
            from_number=normalized.get("From"),
            to_number=normalized.get("To"),
            call_token=call_token,
        )
        if caller is None:
            self._trace_webhook_decision("voice", "caller_not_authorized", form_data=normalized)
            return _xml_response(_say("This phone number is not authorized for this assistant."), _hangup())
        if not caller.is_owner and not caller.allowed_actions:
            self._trace_webhook_decision(
                "voice",
                "caller_not_configured",
                caller=caller,
                form_data=normalized,
            )
            return _xml_response(_say("This phone number is not configured for phone access."), _hangup())
        self._trace_webhook_decision("voice", "accepted", caller=caller, form_data=normalized)

        intro_context = self._get_outbound_context(call_token)
        intro_message = intro_context.intro_message if intro_context and not intro_context.consumed else ""
        if intro_context is not None:
            self._mark_outbound_context_consumed(intro_context.token)

        if caller.is_owner:
            return self._owner_stream_connect_twiml(
                caller=caller,
                webhook_base_url=webhook_base_url,
                call_token=call_token,
                intro_message=intro_message or self._default_greeting(caller),
            )

        greeting = intro_message or self._default_greeting(caller)
        return await self._voice_reply_twiml(
            caller=caller,
            text=greeting,
            webhook_base_url=webhook_base_url,
            continue_listening=True,
            call_token=call_token,
            expressive=caller.is_owner,
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
            return _xml_response(_say("The phone bridge is currently offline."), _hangup())

        caller = self._resolve_caller(
            from_number=normalized.get("From"),
            to_number=normalized.get("To"),
            call_token=call_token,
        )
        if caller is None:
            self._trace_webhook_decision("gather", "caller_not_authorized", form_data=normalized)
            return _xml_response(_say("This phone number is not authorized for this assistant."), _hangup())
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
                text="Thanks. I saved your message for the operator.",
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
            return _xml_response(_say("The phone bridge is currently offline."), _hangup())

        caller = self._resolve_caller(
            from_number=normalized.get("From"),
            to_number=normalized.get("To"),
            call_token=call_token,
        )
        if caller is None:
            self._trace_webhook_decision("poll", "caller_not_authorized", form_data=normalized)
            return _xml_response(_say("This phone number is not authorized for this assistant."), _hangup())

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
        caller = self._resolve_caller(from_number=self.config.owner_phone_number)
        if caller is None or not caller.is_owner:
            await websocket.close(code=1008)
            return
        session = OwnerPhoneMediaStreamSession(
            websocket=websocket,
            service=self,
            caller=caller,
            call_sid="",
        )
        await session.run()

    def _twilio_credentials(self) -> Optional[TwilioCredentials]:
        account_sid = _extract_env_value(("TWILIO_ACCOUNT_SID", "ACCOUNT_SID"))
        api_username = _extract_env_value(("TWILIO_API_KEY", "TWILIO_SID", "SID", "TWILIO_ACCOUNT_SID", "ACCOUNT_SID"))
        api_password = _extract_env_value(("TWILIO_API_SECRET", "TWILIO_SECRET", "SECRET", "TWILIO_AUTH_TOKEN", "AUTH_TOKEN"))
        webhook_auth_token = _extract_env_value(("TWILIO_AUTH_TOKEN", "AUTH_TOKEN"))
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

    def _default_greeting(self, caller: PhoneResolvedCaller) -> str:
        if caller.is_owner:
            return "Hi, it's your OpenCAS assistant. I'm here on the phone. What do you need?"
        if not caller.allowed_actions:
            return "This phone number is not configured for phone access."
        if caller.allows_knowledge_qa:
            return (
                f"Hello {caller.display_name}. You can leave a message, or ask questions "
                "that are covered by the workspace prepared for your number."
            )
        return f"Hello {caller.display_name}. Please leave your message for the operator."

    def _absolute_url(self, base_url: str, path: str, **query: str) -> str:
        root = str(base_url or "").rstrip("/")
        suffix = path if path.startswith("/") else f"/{path}"
        filtered_query = {key: value for key, value in query.items() if value is not None}
        if not filtered_query:
            return f"{root}{suffix}"
        return f"{root}{suffix}?{urlencode(filtered_query)}"

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

    def _owner_stream_connect_twiml(
        self,
        *,
        caller: PhoneResolvedCaller,
        webhook_base_url: str,
        call_token: Optional[str],
        intro_message: str,
    ) -> str:
        return build_connect_stream_twiml(
            self._stream_websocket_url(webhook_base_url),
            {
                "callerNumber": caller.phone_number,
                "displayName": caller.display_name,
                "callToken": call_token or "",
                "introMessage": intro_message,
            },
        )

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
                    prefer_local=False,
                    expressive=expressive,
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
                    prefer_local=False,
                    expressive=expressive,
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
            expressive=True,
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

        heuristic = self._owner_phone_heuristic_reply(transcript)
        if heuristic is not None:
            await self._persist_owner_fast_reply(
                caller=caller,
                transcript=transcript,
                response_text=heuristic,
                call_sid=call_sid,
            )
            return heuristic

        fallback = (
            "I'm handling that on the phone line now. If it needs deeper work, I'll follow up in the main workspace after the call."
        )
        history = await self.runtime.ctx.context_store.list_recent(
            session_id,
            limit=12,
            include_hidden=True,
        )
        messages = [{"role": "system", "content": self._build_owner_phone_prompt(caller)}]
        for entry in history:
            if entry.role not in {MessageRole.USER, MessageRole.ASSISTANT}:
                continue
            content = str(entry.content or "").strip()
            if not content:
                continue
            messages.append({"role": entry.role.value, "content": content})
        messages.append({"role": "user", "content": transcript})

        response = await self.runtime.llm.chat_completion(
            messages=messages,
            complexity="light",
            payload={"temperature": 0.35, "max_tokens": 220},
            source="phone_owner_live",
            session_id=session_id,
            execution_mode="phone_owner_live",
        )
        content = self._extract_response_text(response).strip() or fallback
        await self._persist_owner_fast_reply(
            caller=caller,
            transcript=transcript,
            response_text=content,
            call_sid=call_sid,
        )
        return content

    async def _persist_owner_fast_reply(
        self,
        *,
        caller: PhoneResolvedCaller,
        transcript: str,
        response_text: str,
        call_sid: str,
    ) -> None:
        session_id = self._session_id(caller)
        user_meta = self._user_meta(caller, call_sid=call_sid)
        await self.runtime.ctx.context_store.append(
            session_id,
            MessageRole.USER,
            transcript,
            meta=user_meta,
        )
        await self._record_episode_safe(transcript, session_id=session_id, role="user")
        await self.runtime.ctx.context_store.append(
            session_id,
            MessageRole.ASSISTANT,
            response_text,
            meta=build_assistant_message_meta(
                self.runtime,
                extra={"phone": self._assistant_meta(caller, call_sid=call_sid)},
            ),
        )
        await self._record_episode_safe(response_text, session_id=session_id, role="assistant")

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
        workspace_dir = self._contact_workspace_dir(caller)
        workspace_knowledge = self._workspace_knowledge_excerpt(workspace_dir)
        system_prompt = self._build_low_trust_prompt(caller, workspace_dir, workspace_knowledge)
        messages = [{"role": "system", "content": system_prompt}]
        for entry in history:
            if entry.role in {MessageRole.USER, MessageRole.ASSISTANT} and entry.content.strip():
                messages.append({"role": entry.role.value, "content": entry.content})
        messages.append({"role": "user", "content": transcript})
        low_trust_registry = self._build_low_trust_tool_registry(workspace_dir)
        low_trust_runtime = _LowTrustToolRuntime(self.runtime, low_trust_registry)
        low_trust_loop = ToolUseLoop(
            llm=self.runtime.llm,
            tools=low_trust_registry,
            approval=self.runtime.approval,
            tracer=getattr(self.runtime, "tracer", None),
        )
        result = await low_trust_loop.run(
            objective=(
                "Caller workspace support request. Use only the caller workspace tools to "
                "search files, read files, and write add-only notes when needed. "
                f"Caller request: {transcript}"
            ),
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
            (
                "Phone session preamble: you are speaking live with your primary operator over the phone. "
                "Keep answers concise and spoken-language friendly. You may use your normal tool/runtime surface "
                "because this number is the trusted owner line. If a reply depends on the operator's voice context, "
                f"remember that this call is anchored to {caller.phone_number} and the owner workspace {workspace_dir}."
            ),
            meta={"phone_owner_preamble": True},
        )

    def _build_owner_phone_prompt(self, caller: PhoneResolvedCaller) -> str:
        workspace_dir = self._contact_workspace_dir(caller)
        return "\n".join(
            [
                "You are the OpenCAS assistant speaking live with your trusted owner over the phone.",
                "Reply in short spoken-language sentences.",
                "Do not call tools from this phone reply path.",
                "If the request needs deeper work, acknowledge it briefly and say you will handle it in the main workspace after the call.",
                "Keep answers under 70 words and avoid lists unless the caller explicitly asks for them.",
                f"Owner caller number: {caller.phone_number}.",
                f"Owner workspace: {workspace_dir}.",
            ]
        )

    def _owner_phone_heuristic_reply(self, transcript: str) -> str | None:
        cleaned = " ".join(str(transcript or "").strip().lower().split())
        if not cleaned:
            return None
        if any(phrase in cleaned for phrase in ("what can you do", "how can you help", "what do you do on this line")):
            return (
                "On this line I can take short requests, confirm status, leave notes for follow-up, "
                "and route work back into the main OpenCAS workspace."
            )
        if any(phrase in cleaned for phrase in ("are you there", "can you hear me", "hello", "hi", "hey")):
            return "I'm here. Give me a short request and I'll handle what I can on the phone line."
        if "status" in cleaned and any(word in cleaned for word in ("phone", "line", "bridge")):
            return "The secure phone bridge is live and this is the trusted owner line."
        return None

    def _build_low_trust_prompt(
        self,
        caller: PhoneResolvedCaller,
        workspace_dir: Path,
        workspace_knowledge: str,
    ) -> str:
        parts = [
            "You are the OpenCAS assistant answering a low-trust caller on a phone line managed by OpenCAS.",
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
