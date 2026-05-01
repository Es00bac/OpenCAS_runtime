"""Telegram runtime helpers for OpenCAS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from opencas.telegram_config import (
    TelegramRuntimeConfig,
    telegram_config_path,
    load_telegram_runtime_config,
    save_telegram_runtime_config,
)
from opencas.telegram_integration import TelegramBotService
from opencas.provenance_events_adapter import ProvenanceEventType, emit_provenance_event


def build_runtime_telegram_service(runtime: Any) -> Optional[TelegramBotService]:
    """Instantiate TelegramBotService from the runtime's current config."""
    cfg = runtime._telegram_config
    if not (cfg.enabled and cfg.bot_token):
        return None
    return TelegramBotService(
        runtime=runtime,
        enabled=True,
        token=cfg.bot_token,
        state_dir=runtime.ctx.config.state_dir,
        dm_policy=cfg.dm_policy,
        allow_from=cfg.allow_from,
        poll_interval_seconds=cfg.poll_interval_seconds,
        pairing_ttl_seconds=cfg.pairing_ttl_seconds,
        api_base_url=cfg.api_base_url,
        tracer=runtime.tracer,
    )


def initialize_runtime_telegram(runtime: Any, state_dir: Path | str) -> None:
    """Load persisted Telegram config and rebuild the service handle."""
    runtime._telegram_config = load_telegram_runtime_config(state_dir)
    runtime._telegram = build_runtime_telegram_service(runtime)


async def start_runtime_telegram(runtime: Any) -> None:
    """Start the Telegram polling service if configured. Errors are traced, not raised."""
    if runtime._telegram is None:
        return
    try:
        await runtime._telegram.start()
        runtime._trace("telegram_started", {})
    except Exception as exc:
        runtime._trace("telegram_start_failed", {"error": str(exc)})


def runtime_telegram_settings(runtime: Any) -> TelegramRuntimeConfig:
    """Return the current Telegram runtime settings."""
    return runtime._telegram_config


async def get_runtime_telegram_status(runtime: Any) -> Dict[str, Any]:
    """Return live Telegram status or a stable fallback when disabled."""
    if runtime._telegram is not None:
        return await runtime._telegram.status()
    cfg = runtime._telegram_config
    return {
        "enabled": cfg.enabled,
        "configured": bool(cfg.bot_token),
        "token_configured": bool(cfg.bot_token),
        "running": False,
        "dm_policy": cfg.dm_policy,
        "allow_from": cfg.allow_from,
        "bot": {"id": None, "username": None, "first_name": None, "link": None},
        "last_update_id": None,
        "last_error": None,
        "pairings": {},
    }


async def configure_runtime_telegram(runtime: Any, settings: TelegramRuntimeConfig) -> Dict[str, Any]:
    """Persist new Telegram settings, rebuild the service, and return fresh status."""
    if runtime._telegram is not None:
        try:
            await runtime._telegram.stop()
        except Exception:
            pass
    runtime._telegram_config = settings
    saved_path = save_telegram_runtime_config(runtime.ctx.config.state_dir, settings) or telegram_config_path(runtime.ctx.config.state_dir)
    runtime._telegram = build_runtime_telegram_service(runtime)
    await start_runtime_telegram(runtime)
    status = await get_runtime_telegram_status(runtime)
    status["saved"] = True
    emit_provenance_event(
        status,
        event_type=ProvenanceEventType.MUTATION,
        triggering_artifact="setting|telegram|runtime",
        triggering_action="UPDATE",
        parent_link_id=str(saved_path),
        linked_link_ids=[str(saved_path)],
        details={
            "enabled": settings.enabled,
            "dm_policy": settings.dm_policy,
            "bot_token_configured": bool(settings.bot_token),
        },
    )
    return status


async def approve_runtime_telegram_pairing(runtime: Any, code: str) -> bool:
    """Approve a pending pairing request if a Telegram service is active."""
    if runtime._telegram is None:
        return False
    result = await runtime._telegram.approve_pairing(code)
    return result is not None
