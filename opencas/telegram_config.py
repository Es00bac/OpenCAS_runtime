"""Persistent Telegram runtime configuration for OpenCAS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


TelegramDmPolicy = Literal["disabled", "pairing", "allowlist", "open"]


class TelegramRuntimeConfig(BaseModel):
    """Persisted Telegram integration settings."""

    enabled: bool = False
    bot_token: Optional[str] = None
    dm_policy: TelegramDmPolicy = "pairing"
    allow_from: List[str] = Field(default_factory=list)
    poll_interval_seconds: float = 1.0
    pairing_ttl_seconds: int = 3600
    api_base_url: str = "https://api.telegram.org"

    @field_validator("bot_token")
    @classmethod
    def _normalize_token(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @field_validator("allow_from", mode="before")
    @classmethod
    def _normalize_allow_from(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = value.replace(",", "\n").splitlines()
        else:
            raw_items = list(value)
        normalized = []
        seen = set()
        for item in raw_items:
            cleaned = str(item).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized

    @field_validator("poll_interval_seconds")
    @classmethod
    def _normalize_poll_interval(cls, value: float) -> float:
        return max(0.2, float(value))

    @field_validator("pairing_ttl_seconds")
    @classmethod
    def _normalize_pairing_ttl(cls, value: int) -> int:
        return max(60, int(value))

    def redacted_dict(self) -> Dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["token_configured"] = bool(self.bot_token)
        payload["bot_token"] = "***" if self.bot_token else None
        return payload


def telegram_config_path(state_dir: Path | str) -> Path:
    return Path(state_dir).expanduser() / "telegram" / "config.json"


def load_telegram_runtime_config(state_dir: Path | str) -> TelegramRuntimeConfig:
    path = telegram_config_path(state_dir)
    if not path.exists():
        return TelegramRuntimeConfig()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return TelegramRuntimeConfig.model_validate(payload)
    except Exception:
        pass
    return TelegramRuntimeConfig()


def save_telegram_runtime_config(
    state_dir: Path | str,
    config: TelegramRuntimeConfig,
) -> Path:
    path = telegram_config_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.model_dump(mode="json"), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return path
