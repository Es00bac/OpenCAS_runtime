"""Lane-policy decisions for phone stream entry and menu routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PhoneStreamStartDecision:
    """Resolved caller/mode/greeting for the start of a live stream."""

    caller: Any
    stream_mode: str
    active_menu_key: str
    greeting: str
    employer_mode_active: bool = False


@dataclass(frozen=True)
class PhoneMenuRouteDecision:
    """Resolved action to apply after a recognized phone menu choice."""

    next_mode: str
    next_menu_key: str
    announcement: str
    next_caller: Any | None = None
    employer_mode_active: bool | None = None
    hangup_after_speech: bool = False
    terminal_action: str | None = None


def resolve_stream_start(
    service: Any,
    *,
    stream_mode: str,
    caller_number: str | None,
    display_name: str | None,
    call_token: str | None,
    intro_message: str,
) -> PhoneStreamStartDecision | None:
    caller = service.resolve_stream_session_caller(
        stream_mode=stream_mode,
        caller_number=caller_number,
        display_name=display_name,
        call_token=call_token,
    )
    if caller is None:
        return None
    normalized_mode = str(stream_mode or "owner").strip().lower() or "owner"
    active_menu_key = ""
    if normalized_mode == "owner_menu":
        active_menu_key = service.owner_menu_key() or service.default_menu_key()
    elif normalized_mode == "screening":
        active_menu_key = service.default_menu_key()
    greeting = str(intro_message or "").strip() or service.default_stream_greeting(caller, stream_mode=normalized_mode)
    return PhoneStreamStartDecision(
        caller=caller,
        stream_mode=normalized_mode,
        active_menu_key=active_menu_key,
        greeting=greeting,
        employer_mode_active=bool(getattr(caller, "menu_option_key", None) == "employer"),
    )


async def resolve_menu_route(
    service: Any,
    *,
    option: Any,
    caller: Any,
    active_menu_key: str,
) -> PhoneMenuRouteDecision:
    action = str(option.action or "").strip()
    if action == "owner_conversation":
        return PhoneMenuRouteDecision(
            next_mode="owner",
            next_menu_key="",
            announcement=service.default_stream_greeting(caller, stream_mode="owner"),
        )
    if action == "submenu":
        next_menu_key = str(option.target_menu or "").strip()
        next_mode = "owner_menu" if next_menu_key == (service.owner_menu_key() or "") else "screening"
        return PhoneMenuRouteDecision(
            next_mode=next_mode,
            next_menu_key=next_menu_key,
            announcement=service.menu_prompt(next_menu_key) if next_menu_key else service.menu_reprompt(active_menu_key),
        )
    if action == "workspace_assistant":
        next_caller = await service.activate_menu_workspace_caller(
            option=option,
            caller_number=getattr(caller, "phone_number", None),
            display_name=getattr(caller, "display_name", None),
        )
        return PhoneMenuRouteDecision(
            next_mode="workspace_assistant",
            next_menu_key="",
            next_caller=next_caller,
            announcement=service.menu_workspace_acceptance(next_caller),
            employer_mode_active=bool(getattr(next_caller, "menu_option_key", None) == "employer"),
        )
    if action in {"say_then_hangup", "time_announcement"}:
        announcement = await service.screening_option_announcement(
            option,
            caller_number=getattr(caller, "phone_number", None),
            display_name=getattr(caller, "display_name", None),
        )
        return PhoneMenuRouteDecision(
            next_mode=str(getattr(caller, "menu_option_key", None) or getattr(caller, "trust_level", "") or "screening"),
            next_menu_key=active_menu_key,
            announcement=announcement,
            hangup_after_speech=True,
            terminal_action=action,
        )
    return PhoneMenuRouteDecision(
        next_mode="owner_menu" if active_menu_key == (service.owner_menu_key() or "") else "screening",
        next_menu_key=active_menu_key,
        announcement=service.menu_reprompt(active_menu_key),
    )
