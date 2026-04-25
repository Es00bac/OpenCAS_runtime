"""Phone bridge runtime helpers for OpenCAS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from opencas.phone_config import (
    PhoneRuntimeConfig,
    load_phone_runtime_config,
    save_phone_runtime_config,
)
from opencas.phone_integration import PhoneBridgeService


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


async def get_runtime_phone_status(runtime: Any) -> Dict[str, Any]:
    """Return the effective phone-bridge status."""

    if getattr(runtime, "_phone", None) is not None:
        return runtime._phone.status()
    return {
        **runtime._phone_config.redacted_dict(),
        "twilio_credentials_configured": False,
        "webhook_urls": {"voice": None, "gather": None},
        "contact_count": 0,
    }


async def configure_runtime_phone(runtime: Any, settings: PhoneRuntimeConfig) -> Dict[str, Any]:
    """Persist new phone settings and return fresh status."""

    runtime._phone_config = settings
    save_phone_runtime_config(runtime.ctx.config.state_dir, settings)
    runtime._phone = build_runtime_phone_service(runtime)
    status = await get_runtime_phone_status(runtime)
    status["saved"] = True
    return status


async def autoconfigure_runtime_phone(
    runtime: Any,
    *,
    enabled: bool | None = None,
    public_base_url: str | None = None,
    webhook_signature_required: bool | None = None,
    webhook_secret: str | None = None,
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
        twilio_from_number=twilio_from_number,
        owner_phone_number=owner_phone_number,
        owner_display_name=owner_display_name,
        owner_workspace_subdir=owner_workspace_subdir,
    )
    settings = result["settings"]
    runtime._phone_config = settings
    save_phone_runtime_config(runtime.ctx.config.state_dir, settings)
    runtime._phone = build_runtime_phone_service(runtime)
    status = await get_runtime_phone_status(runtime)
    status["saved"] = True
    status["autoconfigured"] = True
    status["selected_number"] = result.get("selected_number")
    status["twilio_number_candidates"] = result.get("twilio_number_candidates", [])
    status["webhook_update"] = result.get("webhook_update", {})
    status["note"] = result.get("note")
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
    """Handle a live owner phone websocket session."""

    if getattr(runtime, "_phone", None) is None:
        raise RuntimeError("Phone bridge is not available")
    await runtime._phone.handle_media_stream(
        websocket=websocket,
        request_url=request_url,
        provided_signature=provided_signature,
        stream_secret=stream_secret,
    )
