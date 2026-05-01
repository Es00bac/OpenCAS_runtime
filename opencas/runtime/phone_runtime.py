"""Phone bridge runtime helpers for OpenCAS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from opencas.api.provenance_store import ProvenanceTransitionKind, record_provenance_transition
from opencas.phone_config import (
    PhoneMenuConfig,
    PhoneMenuDefinition,
    PhoneMenuOption,
    PhoneRuntimeConfig,
    PhoneWorkspaceMount,
    phone_config_path,
    load_phone_menu_config,
    normalize_phone_number,
    load_phone_runtime_config,
    phone_dashboard_menu_path,
    save_phone_runtime_config,
    save_phone_menu_config,
    summarize_phone_session_profiles,
)
from opencas.phone_integration_service import PhoneBridgeService
from opencas.phone_session_state import analyze_phone_call_timeline
from opencas.provenance_events_adapter import ProvenanceEventType, emit_provenance_event
from opencas.telemetry import EventKind

_DEFAULT_PHONE_MENU_PATH = Path("operator_seed/phone/menu.json")
_PHONE_TRACE_PREFIX = "AgentRuntime: "
_PHONE_TRACE_SCAN_LIMIT = 240
_PHONE_TRACE_DISPLAY_LIMIT = 24
_PHONE_CALL_DISPLAY_LIMIT = 10


def _record_phone_settings_provenance(
    runtime: Any,
    *,
    target_entity: str,
    trigger_action: str,
    source_artifact: str,
    saved_path: Path,
    details: Mapping[str, Any] | None = None,
    linked_transition_ids: list[str] | tuple[str, ...] | None = None,
) -> None:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    state_dir = getattr(config, "state_dir", None)
    if state_dir is None:
        return
    raw_session_id = getattr(config, "session_id", None) if config is not None else None
    session_id = raw_session_id.strip() if isinstance(raw_session_id, str) and raw_session_id.strip() else target_entity
    payload = dict(details or {})
    payload.setdefault("saved_path", str(saved_path))
    record_provenance_transition(
        state_dir=state_dir,
        kind=ProvenanceTransitionKind.MUTATION,
        session_id=session_id,
        entity_id=target_entity,
        status="mutated",
        trigger_artifact=source_artifact,
        source_artifact=source_artifact,
        trigger_action=trigger_action,
        parent_transition_id=str(saved_path),
        linked_transition_ids=linked_transition_ids,
        target_entity=target_entity,
        origin_action_id=str(saved_path),
        details=payload,
    )


def build_runtime_phone_service(runtime: Any) -> PhoneBridgeService:
    """Instantiate the phone bridge service from the runtime's current config."""

    return PhoneBridgeService(runtime=runtime, config=runtime._phone_config)


def initialize_runtime_phone(runtime: Any, state_dir: Path | str) -> None:
    """Load persisted phone settings and rebuild the service handle."""

    runtime._phone_config = load_phone_runtime_config(state_dir)
    runtime._phone = build_runtime_phone_service(runtime)


def runtime_phone_settings(runtime: Any) -> PhoneRuntimeConfig:
    """Return the current phone bridge settings."""

    return runtime._phone_config


def _runtime_phone_menu_config_path(runtime: Any) -> Path:
    raw = str(getattr(runtime, "_phone_config", PhoneRuntimeConfig()).menu_config_path or "").strip()
    if raw:
        return Path(raw).expanduser()
    return _DEFAULT_PHONE_MENU_PATH


def _runtime_phone_menu(runtime: Any) -> PhoneMenuConfig:
    menu = load_phone_menu_config(_runtime_phone_menu_config_path(runtime))
    if menu.menus:
        return menu
    return load_phone_menu_config(_DEFAULT_PHONE_MENU_PATH)


def _ensure_menu(menu: PhoneMenuConfig, key: str) -> PhoneMenuDefinition:
    for item in menu.menus:
        if item.key == key:
            return item
    created = PhoneMenuDefinition(key=key)
    menu.menus.append(created)
    return created


def _ensure_option(menu: PhoneMenuDefinition, key: str, *, action: str, digit: str) -> PhoneMenuOption:
    for item in menu.options:
        if item.key == key:
            item.action = action  # type: ignore[assignment]
            item.digit = digit
            return item
    created = PhoneMenuOption(key=key, action=action, digit=digit)
    menu.options.append(created)
    return created


def _remove_option(menu: PhoneMenuDefinition, key: str) -> None:
    menu.options = [item for item in menu.options if item.key != key]


def _friendly_phone_event_label(event_name: str) -> str:
    return event_name.removeprefix("phone_").replace("_", " ").strip() or event_name


def _phone_trace_payload_summary(event_name: str, payload: Mapping[str, Any]) -> str:
    if event_name == "phone_session_state_changed":
        from_state = str(payload.get("from_state") or "").strip()
        to_state = str(payload.get("to_state") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        if from_state and to_state:
            return f"{from_state} -> {to_state}" + (f" ({reason})" if reason else "")
        if to_state:
            return to_state + (f" ({reason})" if reason else "")
        return reason or "state changed"
    if event_name == "phone_stream_dtmf":
        digit = str(payload.get("digit") or "").strip()
        return f"digit {digit}" if digit else "digit received"
    if event_name in {"phone_stream_transcribed", "phone_owner_reply_started", "phone_workspace_reply_started"}:
        preview = str(payload.get("transcript_preview") or "").strip()
        return preview[:120] or "speech captured"
    if event_name in {"phone_owner_reply_completed", "phone_workspace_reply_completed"}:
        preview = str(payload.get("response_preview") or "").strip()
        return preview[:120] or "reply completed"
    if event_name == "phone_stream_menu_choice":
        action = str(payload.get("action") or "").strip()
        choice = str(payload.get("choice") or "").strip()
        if action and choice:
            return f"{choice} -> {action}"
        return action or choice or "menu choice"
    if event_name == "phone_stream_closed":
        reason = str(payload.get("reason") or "").strip()
        return reason or "stream closed"
    if event_name in {"phone_owner_reply_issue", "phone_stream_tts_attempt_failed", "phone_stream_tts_failed"}:
        reason = str(payload.get("reason") or payload.get("error") or "").strip()
        return reason[:120] or "issue recorded"
    return str(payload.get("note") or "").strip()[:120]


def _recent_phone_telemetry(runtime: Any) -> Dict[str, list[Dict[str, Any]]]:
    tracer = getattr(runtime, "tracer", None)
    store = getattr(tracer, "store", None)
    if store is None or not hasattr(store, "query"):
        return {"recent_calls": [], "recent_events": []}
    try:
        candidates = store.query(limit=_PHONE_TRACE_SCAN_LIMIT)
    except Exception:
        return {"recent_calls": [], "recent_events": []}

    filtered = []
    for event in candidates:
        if getattr(event, "kind", None) != EventKind.TOM_EVAL:
            continue
        message = str(getattr(event, "message", "") or "")
        if not message.startswith(_PHONE_TRACE_PREFIX):
            continue
        event_name = message.removeprefix(_PHONE_TRACE_PREFIX).strip()
        if not event_name.startswith("phone_"):
            continue
        filtered.append(event)

    recent_events: list[Dict[str, Any]] = []
    calls: dict[str, Dict[str, Any]] = {}
    call_timelines: dict[str, list[Dict[str, Any]]] = {}
    for event in filtered[-_PHONE_TRACE_DISPLAY_LIMIT:]:
        payload = dict(getattr(event, "payload", {}) or {})
        event_name = str(getattr(event, "message", "")).removeprefix(_PHONE_TRACE_PREFIX).strip()
        call_sid = str(payload.get("call_sid") or "").strip() or None
        caller_number = normalize_phone_number(payload.get("caller_number"))
        entry = {
            "timestamp": event.timestamp.isoformat(),
            "event": event_name,
            "label": _friendly_phone_event_label(event_name),
            "summary": _phone_trace_payload_summary(event_name, payload),
            "call_sid": call_sid,
            "caller_number": caller_number,
            "mode": str(payload.get("mode") or "").strip() or None,
            "stream_sid": str(payload.get("stream_sid") or "").strip() or None,
        }
        recent_events.append(entry)
        if call_sid:
            call_timelines.setdefault(call_sid, []).append({**entry, "payload": payload})

    for event in filtered:
        payload = dict(getattr(event, "payload", {}) or {})
        event_name = str(getattr(event, "message", "")).removeprefix(_PHONE_TRACE_PREFIX).strip()
        call_sid = str(payload.get("call_sid") or "").strip()
        if not call_sid:
            continue
        caller_number = normalize_phone_number(payload.get("caller_number"))
        timestamp = event.timestamp.isoformat()
        record = calls.setdefault(
            call_sid,
            {
                "call_sid": call_sid,
                "caller_number": caller_number,
                "mode": str(payload.get("mode") or "").strip() or None,
                "started_at": timestamp,
                "last_at": timestamp,
                "last_event": event_name,
                "last_summary": _phone_trace_payload_summary(event_name, payload),
                "event_count": 0,
                "issue_count": 0,
                "hangup_reason": None,
                "recent_labels": [],
            },
        )
        record["last_at"] = timestamp
        record["last_event"] = event_name
        record["last_summary"] = _phone_trace_payload_summary(event_name, payload)
        record["event_count"] += 1
        if caller_number and not record.get("caller_number"):
            record["caller_number"] = caller_number
        mode = str(payload.get("mode") or "").strip()
        if mode and not record.get("mode"):
            record["mode"] = mode
        if "issue" in event_name or event_name.endswith("_failed") or event_name.endswith("_invalid"):
            record["issue_count"] += 1
        if event_name == "phone_stream_closed":
            reason = str(payload.get("reason") or "").strip()
            if reason:
                record["hangup_reason"] = reason
        labels = record["recent_labels"]
        labels.append(_friendly_phone_event_label(event_name))
        if len(labels) > 5:
            del labels[0 : len(labels) - 5]

    for call_sid, timeline in call_timelines.items():
        record = calls.get(call_sid)
        if record is None:
            continue
        diagnostics = analyze_phone_call_timeline(timeline)
        record["current_state"] = diagnostics.get("current_state")
        record["terminal_action"] = diagnostics.get("terminal_action")
        record["hangup_class"] = diagnostics.get("hangup_class")
        record["visited_states"] = diagnostics.get("visited_states") or []

    recent_calls = sorted(calls.values(), key=lambda item: item["last_at"], reverse=True)[:_PHONE_CALL_DISPLAY_LIMIT]
    return {"recent_calls": recent_calls, "recent_events": list(reversed(recent_events))}


def _phone_call_detail(runtime: Any, call_sid: str) -> Dict[str, Any]:
    tracer = getattr(runtime, "tracer", None)
    store = getattr(tracer, "store", None)
    normalized_call_sid = str(call_sid or "").strip()
    if not normalized_call_sid or store is None or not hasattr(store, "query"):
        return {"found": False, "call_sid": normalized_call_sid, "events": [], "phase_durations": {}}
    try:
        candidates = store.query(limit=max(_PHONE_TRACE_SCAN_LIMIT * 2, 480))
    except Exception:
        return {"found": False, "call_sid": normalized_call_sid, "events": [], "phase_durations": {}}

    timeline: list[Dict[str, Any]] = []
    for event in candidates:
        if getattr(event, "kind", None) != EventKind.TOM_EVAL:
            continue
        message = str(getattr(event, "message", "") or "")
        if not message.startswith(_PHONE_TRACE_PREFIX):
            continue
        event_name = message.removeprefix(_PHONE_TRACE_PREFIX).strip()
        if not event_name.startswith("phone_"):
            continue
        payload = dict(getattr(event, "payload", {}) or {})
        if str(payload.get("call_sid") or "").strip() != normalized_call_sid:
            continue
        timeline.append(
            {
                "timestamp": event.timestamp.isoformat(),
                "event": event_name,
                "label": _friendly_phone_event_label(event_name),
                "summary": _phone_trace_payload_summary(event_name, payload),
                "payload": payload,
            }
        )

    timeline.sort(key=lambda item: item["timestamp"])
    if not timeline:
        return {"found": False, "call_sid": normalized_call_sid, "events": [], "phase_durations": {}}

    first_payload = dict(timeline[0].get("payload") or {})
    diagnostics = analyze_phone_call_timeline(timeline)

    issue_count = sum(
        1
        for item in timeline
        if "issue" in item["event"] or item["event"].endswith("_failed") or item["event"].endswith("_invalid")
    )
    call = {
        "call_sid": normalized_call_sid,
        "caller_number": normalize_phone_number(first_payload.get("caller_number")),
        "caller_display_name": str(first_payload.get("caller_display_name") or "").strip() or None,
        "mode": str(first_payload.get("mode") or "").strip() or None,
        "started_at": timeline[0]["timestamp"],
        "last_at": timeline[-1]["timestamp"],
        "hangup_reason": diagnostics.get("hangup_reason"),
        "hangup_class": diagnostics.get("hangup_class"),
        "event_count": len(timeline),
        "issue_count": issue_count,
        "last_event": timeline[-1]["event"],
        "last_summary": timeline[-1]["summary"],
        "current_state": diagnostics.get("current_state"),
        "terminal_action": diagnostics.get("terminal_action"),
        "visited_states": diagnostics.get("visited_states") or [],
    }
    return {
        "found": True,
        "call_sid": normalized_call_sid,
        "call": call,
        "events": timeline,
        "phase_durations": diagnostics.get("phase_durations") or {},
        "state_timeline": diagnostics.get("state_timeline") or [],
    }


async def get_runtime_phone_status(runtime: Any) -> Dict[str, Any]:
    """Return the effective phone-bridge status."""

    if getattr(runtime, "_phone", None) is not None:
        status = runtime._phone.status()
    else:
        status = {
        **runtime._phone_config.redacted_dict(),
        "twilio_credentials_configured": False,
        "webhook_urls": {"voice": None, "gather": None},
        "contact_count": 0,
        }
    menu = _runtime_phone_menu(runtime)
    status["session_profiles"] = summarize_phone_session_profiles(menu)
    status["menu_config_source"] = {
        "path": str(_runtime_phone_menu_config_path(runtime)),
        "editable_path": str(phone_dashboard_menu_path(runtime.ctx.config.state_dir)),
        "using_override": bool(getattr(runtime, "_phone_config", PhoneRuntimeConfig()).menu_config_path),
    }
    status["menu_config"] = menu.model_dump(mode="json")
    status.update(_recent_phone_telemetry(runtime))
    return status


async def get_runtime_recent_phone_calls(runtime: Any, *, limit: int = _PHONE_CALL_DISPLAY_LIMIT) -> Dict[str, Any]:
    snapshot = _recent_phone_telemetry(runtime)
    return {
        "calls": list(snapshot.get("recent_calls") or [])[: max(1, int(limit))],
        "events": list(snapshot.get("recent_events") or []),
    }


async def get_runtime_phone_call_detail(runtime: Any, *, call_sid: str) -> Dict[str, Any]:
    return _phone_call_detail(runtime, call_sid)


async def configure_runtime_phone(runtime: Any, settings: PhoneRuntimeConfig) -> Dict[str, Any]:
    """Persist new phone settings and return fresh status."""

    runtime._phone_config = settings
    saved_path = save_phone_runtime_config(runtime.ctx.config.state_dir, settings) or phone_config_path(runtime.ctx.config.state_dir)
    runtime._phone = build_runtime_phone_service(runtime)
    status = await get_runtime_phone_status(runtime)
    status["saved"] = True
    emit_provenance_event(
        status,
        event_type=ProvenanceEventType.MUTATION,
        triggering_artifact="setting|phone|runtime",
        triggering_action="UPDATE",
        parent_link_id=str(saved_path),
        linked_link_ids=[str(saved_path)],
        details={
            "enabled": settings.enabled,
            "owner_phone_number": settings.owner_phone_number,
            "twilio_from_number": settings.twilio_from_number,
        },
    )
    return status


async def configure_runtime_phone_session_profiles(runtime: Any, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Persist the structured phone-session profile editor into an editable menu config."""

    menu = _runtime_phone_menu(runtime)
    menu.default_menu_key = "public_main"
    menu.owner_menu_key = "owner_entry"
    menu.owner_pin_prompt = str(payload.get("owner_pin_prompt") or "").strip()
    menu.owner_pin_retry_prompt = str(payload.get("owner_pin_retry_prompt") or "").strip()
    menu.owner_pin_success_message = str(payload.get("owner_pin_success_message") or "").strip()
    menu.owner_pin_failure_message = str(payload.get("owner_pin_failure_message") or "").strip()

    owner_menu = _ensure_menu(menu, "owner_entry")
    owner_menu.prompt = str(payload.get("owner_entry_prompt") or "").strip()
    owner_menu.reprompt = str(payload.get("owner_entry_reprompt") or "").strip()
    owner_continue = _ensure_option(
        owner_menu,
        "owner_continue",
        action="owner_conversation",
        digit=str(payload.get("owner_continue_digit") or "1").strip() or "1",
    )
    owner_continue.label = "Continue as owner"
    owner_continue.phrases = ["continue", "owner", "jarrod", "me"]
    owner_main_menu = _ensure_option(
        owner_menu,
        "owner_main_menu",
        action="submenu",
        digit=str(payload.get("owner_fallback_digit") or "2").strip() or "2",
    )
    owner_main_menu.label = "Main menu"
    owner_main_menu.phrases = ["main menu", "menu", "public"]
    owner_main_menu.target_menu = "public_main"

    public_menu = _ensure_menu(menu, "public_main")
    public_menu.prompt = str(payload.get("public_prompt") or "").strip()
    public_menu.reprompt = str(payload.get("public_reprompt") or "").strip()

    employer_enabled = bool(payload.get("employer_enabled", True))
    if employer_enabled:
        employer = _ensure_option(
            public_menu,
            "employer",
            action="workspace_assistant",
            digit=str(payload.get("employer_digit") or "1").strip() or "1",
        )
        employer.label = str(payload.get("employer_label") or "Potential employer").strip()
        employer.phrases = [
            str(item).strip().lower()
            for item in list(payload.get("employer_phrases") or [])
            if str(item).strip()
        ]
        employer.greeting = str(payload.get("employer_greeting") or "").strip()
        employer.prompt_profile = str(payload.get("employer_prompt_profile") or "worksafe_owner").strip() or "worksafe_owner"
        employer.allowed_actions = list(payload.get("employer_allowed_actions") or ["leave_message", "knowledge_qa"])
        employer.workspace_mounts = [
            PhoneWorkspaceMount(
                scope="shared",
                subdir=str(payload.get("employer_shared_workspace_subdir") or "phone/employer_shared").strip() or "phone/employer_shared",
                access="read_only",
            ),
            PhoneWorkspaceMount(
                scope="caller",
                subdir=str(payload.get("employer_caller_workspace_subdir") or "phone/employers/{phone_digits}").strip() or "phone/employers/{phone_digits}",
                access="append_only",
            ),
        ]
    else:
        _remove_option(public_menu, "employer")

    reject_enabled = bool(payload.get("reject_enabled", True))
    if reject_enabled:
        reject = _ensure_option(
            public_menu,
            "reject",
            action="say_then_hangup",
            digit=str(payload.get("reject_digit") or "2").strip() or "2",
        )
        reject.label = str(payload.get("reject_label") or "Not for this line").strip()
        reject.phrases = [
            str(item).strip().lower()
            for item in list(payload.get("reject_phrases") or [])
            if str(item).strip()
        ]
        reject.message = str(payload.get("reject_message") or "").strip()
        reject.greeting = ""
        reject.prompt_profile = None
        reject.target_menu = None
        reject.allowed_actions = ["leave_message", "knowledge_qa"]
        reject.workspace_mounts = []
    else:
        _remove_option(public_menu, "reject")

    menu = PhoneMenuConfig.model_validate(menu.model_dump(mode="json"))
    menu_path = phone_dashboard_menu_path(runtime.ctx.config.state_dir)
    save_phone_menu_config(menu_path, menu)

    runtime._phone_config = runtime._phone_config.model_copy(
        update={"menu_config_path": str(menu_path)}
    )
    saved_phone_config_path = save_phone_runtime_config(runtime.ctx.config.state_dir, runtime._phone_config) or phone_config_path(runtime.ctx.config.state_dir)
    runtime._phone = build_runtime_phone_service(runtime)
    status = await get_runtime_phone_status(runtime)
    status["saved"] = True
    status["session_profiles_saved"] = True
    emit_provenance_event(
        status,
        event_type=ProvenanceEventType.MUTATION,
        triggering_artifact="setting|phone|session-profiles",
        triggering_action="UPDATE",
        parent_link_id=str(menu_path),
        linked_link_ids=[str(menu_path), str(saved_phone_config_path)],
        details={
            "menu_config_path": str(menu_path),
            "employer_enabled": employer_enabled,
            "reject_enabled": reject_enabled,
        },
    )
    return status


async def configure_runtime_phone_menu_config(runtime: Any, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Persist an advanced phone menu config from the dashboard."""

    menu = PhoneMenuConfig.model_validate(payload)
    menu_path = phone_dashboard_menu_path(runtime.ctx.config.state_dir)
    save_phone_menu_config(menu_path, menu)

    runtime._phone_config = runtime._phone_config.model_copy(
        update={"menu_config_path": str(menu_path)}
    )
    saved_phone_config_path = save_phone_runtime_config(runtime.ctx.config.state_dir, runtime._phone_config) or phone_config_path(runtime.ctx.config.state_dir)
    runtime._phone = build_runtime_phone_service(runtime)
    status = await get_runtime_phone_status(runtime)
    status["saved"] = True
    status["menu_config_saved"] = True
    emit_provenance_event(
        status,
        event_type=ProvenanceEventType.MUTATION,
        triggering_artifact="setting|phone|menu-config",
        triggering_action="UPDATE",
        parent_link_id=str(menu_path),
        linked_link_ids=[str(menu_path), str(saved_phone_config_path)],
        details={
            "menu_config_path": str(menu_path),
            "menu_count": len(menu.menus),
        },
    )
    return status


async def autoconfigure_runtime_phone(
    runtime: Any,
    *,
    enabled: bool | None = None,
    public_base_url: str | None = None,
    webhook_signature_required: bool | None = None,
    webhook_secret: str | None = None,
    twilio_env_path: str | None = None,
    twilio_account_sid: str | None = None,
    twilio_api_key_sid: str | None = None,
    twilio_api_secret: str | None = None,
    twilio_auth_token: str | None = None,
    twilio_from_number: str | None = None,
    owner_phone_number: str | None = None,
    owner_display_name: str | None = None,
    owner_workspace_subdir: str | None = None,
) -> Dict[str, Any]:
    """Resolve Twilio resources, persist the resulting config, and return status."""

    if getattr(runtime, "_phone", None) is None:
        runtime._phone = build_runtime_phone_service(runtime)
    result = await runtime._phone.autoconfigure_twilio(
        enabled=enabled,
        public_base_url=public_base_url,
        webhook_signature_required=webhook_signature_required,
        webhook_secret=webhook_secret,
        twilio_env_path=twilio_env_path,
        twilio_account_sid=twilio_account_sid,
        twilio_api_key_sid=twilio_api_key_sid,
        twilio_api_secret=twilio_api_secret,
        twilio_auth_token=twilio_auth_token,
        twilio_from_number=twilio_from_number,
        owner_phone_number=owner_phone_number,
        owner_display_name=owner_display_name,
        owner_workspace_subdir=owner_workspace_subdir,
    )
    settings = result["settings"]
    runtime._phone_config = settings
    saved_path = save_phone_runtime_config(runtime.ctx.config.state_dir, settings) or phone_config_path(runtime.ctx.config.state_dir)
    runtime._phone = build_runtime_phone_service(runtime)
    status = await get_runtime_phone_status(runtime)
    status["saved"] = True
    status["autoconfigured"] = True
    status["selected_number"] = result.get("selected_number")
    status["twilio_number_candidates"] = result.get("twilio_number_candidates", [])
    status["webhook_update"] = result.get("webhook_update", {})
    status["note"] = result.get("note")
    emit_provenance_event(
        status,
        event_type=ProvenanceEventType.MUTATION,
        triggering_artifact="setting|phone|autoconfig",
        triggering_action="UPDATE",
        parent_link_id=str(saved_path),
        linked_link_ids=[str(saved_path)],
        details={
            "enabled": settings.enabled,
            "selected_number": result.get("selected_number", {}),
            "note": result.get("note"),
        },
    )
    return status


async def call_owner_via_runtime_phone(
    runtime: Any,
    *,
    message: str,
    reason: str = "",
) -> Dict[str, Any]:
    """Trigger an outbound owner call through the configured phone bridge."""

    if getattr(runtime, "_phone", None) is None:
        raise RuntimeError("Phone bridge is not available")
    result = await runtime._phone.place_owner_call(message=message, reason=reason)
    runtime._trace(
        "phone_owner_call_requested",
        {
            "to": result.get("to"),
            "call_sid": result.get("call_sid"),
            "status": result.get("status"),
        },
    )
    return result


async def handle_runtime_phone_voice_webhook(
    runtime: Any,
    *,
    request_url: str,
    webhook_base_url: str,
    form_data: Mapping[str, Any],
    provided_signature: str | None,
    call_token: str | None = None,
    bridge_token: str | None = None,
) -> str:
    """Render TwiML for the initial Twilio voice webhook."""

    if getattr(runtime, "_phone", None) is None:
        raise RuntimeError("Phone bridge is not available")
    return await runtime._phone.handle_voice_webhook(
        request_url=request_url,
        webhook_base_url=webhook_base_url,
        form_data=form_data,
        provided_signature=provided_signature,
        call_token=call_token,
        bridge_token=bridge_token,
    )


async def handle_runtime_phone_gather_webhook(
    runtime: Any,
    *,
    request_url: str,
    webhook_base_url: str,
    form_data: Mapping[str, Any],
    provided_signature: str | None,
    call_token: str | None = None,
    bridge_token: str | None = None,
) -> str:
    """Render TwiML for a speech-gather continuation webhook."""

    if getattr(runtime, "_phone", None) is None:
        raise RuntimeError("Phone bridge is not available")
    return await runtime._phone.handle_gather_webhook(
        request_url=request_url,
        webhook_base_url=webhook_base_url,
        form_data=form_data,
        provided_signature=provided_signature,
        call_token=call_token,
        bridge_token=bridge_token,
    )


async def handle_runtime_phone_poll_webhook(
    runtime: Any,
    *,
    request_url: str,
    webhook_base_url: str,
    form_data: Mapping[str, Any],
    provided_signature: str | None,
    call_token: str | None = None,
    bridge_token: str | None = None,
    reply_token: str | None = None,
) -> str:
    """Render TwiML while waiting for a background phone reply to complete."""

    if getattr(runtime, "_phone", None) is None:
        raise RuntimeError("Phone bridge is not available")
    return await runtime._phone.handle_poll_webhook(
        request_url=request_url,
        webhook_base_url=webhook_base_url,
        form_data=form_data,
        provided_signature=provided_signature,
        call_token=call_token,
        bridge_token=bridge_token,
        reply_token=reply_token,
    )


async def handle_runtime_phone_media_stream(
    runtime: Any,
    *,
    websocket: Any,
    request_url: str,
    provided_signature: str | None,
    stream_secret: str,
) -> None:
    """Handle a live phone websocket session."""

    if getattr(runtime, "_phone", None) is None:
        raise RuntimeError("Phone bridge is not available")
    await runtime._phone.handle_media_stream(
        websocket=websocket,
        request_url=request_url,
        provided_signature=provided_signature,
        stream_secret=stream_secret,
    )
