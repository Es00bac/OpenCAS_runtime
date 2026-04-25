"""Telegram control-plane routes for the OpenCAS dashboard."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from opencas.telegram_config import TelegramRuntimeConfig


class TelegramConfigUpdateRequest(BaseModel):
    enabled: bool = False
    bot_token: Optional[str] = None
    clear_bot_token: bool = False
    dm_policy: str = "pairing"
    allow_from: List[str] = Field(default_factory=list)
    poll_interval_seconds: float = 1.0
    pairing_ttl_seconds: int = 3600


def build_telegram_router(runtime: Any) -> APIRouter:
    """Build Telegram setup and status routes wired to *runtime*."""

    r = APIRouter(prefix="/api/telegram", tags=["telegram"])

    @r.get("/status")
    async def get_status() -> Dict[str, Any]:
        return await runtime.telegram_status()

    @r.post("/config")
    async def update_config(req: TelegramConfigUpdateRequest) -> Dict[str, Any]:
        current = runtime.telegram_settings
        bot_token = None
        if not req.clear_bot_token:
            bot_token = (
                req.bot_token.strip()
                if req.bot_token is not None and req.bot_token.strip()
                else current.bot_token
            )
        settings = TelegramRuntimeConfig(
            enabled=req.enabled,
            bot_token=bot_token,
            dm_policy=req.dm_policy,
            allow_from=req.allow_from,
            poll_interval_seconds=req.poll_interval_seconds,
            pairing_ttl_seconds=req.pairing_ttl_seconds,
            api_base_url=current.api_base_url,
        )
        return await runtime.configure_telegram(settings)

    @r.post("/pairings/{code}/approve")
    async def approve_pairing(code: str) -> Dict[str, Any]:
        approved = await runtime.approve_telegram_pairing(code)
        if not approved:
            raise HTTPException(status_code=404, detail="Pairing code not found")
        return await runtime.telegram_status()

    return r
