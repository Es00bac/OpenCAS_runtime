"""Persistent phone bridge configuration for OpenCAS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


PhoneTrustLevel = Literal["owner", "low"]
PhoneAllowedAction = Literal["leave_message", "knowledge_qa"]


def normalize_phone_number(value: Any) -> Optional[str]:
    """Normalize a phone number into a stable E.164-style string when possible."""

    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    has_plus = raw.startswith("+")
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None

    if has_plus:
        return f"+{digits}"
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    if len(digits) >= 8:
        return f"+{digits}"
    return raw


class PhoneContactPolicy(BaseModel):
    """Per-number trust and capability rules for the phone bridge."""

    phone_number: str
    display_name: str = ""
    trust_level: PhoneTrustLevel = "low"
    allowed_actions: List[PhoneAllowedAction] = Field(default_factory=lambda: ["leave_message"])
    workspace_subdir: Optional[str] = None
    notes: str = ""

    @field_validator("phone_number")
    @classmethod
    def _normalize_phone_number(cls, value: Any) -> str:
        normalized = normalize_phone_number(value)
        if not normalized:
            raise ValueError("phone_number is required")
        return normalized

    @field_validator("display_name", "notes")
    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("trust_level")
    @classmethod
    def _normalize_trust_level(cls, value: Any) -> PhoneTrustLevel:
        cleaned = str(value or "low").strip() or "low"
        if cleaned == "owner":
            raise ValueError("contacts must use low trust; owner_phone_number defines the trusted line")
        return cleaned  # type: ignore[return-value]

    @field_validator("workspace_subdir")
    @classmethod
    def _normalize_workspace_subdir(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        raw = str(value).strip().replace("\\", "/")
        if not raw:
            return None
        candidate = Path(raw)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("workspace_subdir must stay inside the managed workspace")
        return "/".join(part for part in candidate.parts if part not in {"."})

    @field_validator("allowed_actions", mode="before")
    @classmethod
    def _normalize_allowed_actions(cls, value: Any) -> List[PhoneAllowedAction]:
        if value is None:
            return ["leave_message"]
        raw_items = value if isinstance(value, list) else [value]
        normalized: List[PhoneAllowedAction] = []
        seen = set()
        for item in raw_items:
            cleaned = str(item or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)  # type: ignore[arg-type]
        return normalized


class PhoneRuntimeConfig(BaseModel):
    """Persisted runtime settings for Twilio phone integration."""

    enabled: bool = False
    public_base_url: Optional[str] = None
    webhook_signature_required: bool = True
    webhook_secret: Optional[str] = None
    twilio_from_number: Optional[str] = None
    owner_phone_number: Optional[str] = None
    owner_display_name: str = "Operator"
    owner_workspace_subdir: str = "phone/owner"
    contacts: List[PhoneContactPolicy] = Field(default_factory=list)

    @field_validator("public_base_url")
    @classmethod
    def _normalize_public_base_url(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip().rstrip("/")
        if cleaned and not cleaned.startswith(("http://", "https://")):
            raise ValueError("public_base_url must start with http:// or https://")
        return cleaned or None

    @field_validator("twilio_from_number", "owner_phone_number")
    @classmethod
    def _normalize_optional_phone(cls, value: Any) -> Optional[str]:
        return normalize_phone_number(value)

    @field_validator("owner_display_name")
    @classmethod
    def _normalize_owner_display_name(cls, value: Any) -> str:
        cleaned = str(value or "").strip()
        return cleaned or "Operator"

    @field_validator("webhook_secret")
    @classmethod
    def _normalize_webhook_secret(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @field_validator("owner_workspace_subdir")
    @classmethod
    def _normalize_owner_workspace_subdir(cls, value: Any) -> str:
        cleaned = PhoneContactPolicy._normalize_workspace_subdir(value)
        return cleaned or "phone/owner"

    @field_validator("contacts", mode="before")
    @classmethod
    def _dedupe_contacts(cls, value: Any) -> List[PhoneContactPolicy]:
        raw_items = list(value or [])
        deduped: List[PhoneContactPolicy] = []
        seen = set()
        for item in raw_items:
            contact = item if isinstance(item, PhoneContactPolicy) else PhoneContactPolicy.model_validate(item)
            if contact.phone_number in seen:
                continue
            seen.add(contact.phone_number)
            deduped.append(contact)
        return deduped

    @model_validator(mode="after")
    def _validate_owner_uniqueness(self) -> "PhoneRuntimeConfig":
        if self.owner_phone_number and any(
            contact.phone_number == self.owner_phone_number for contact in self.contacts
        ):
            raise ValueError("owner_phone_number must not also appear in contacts")
        return self

    def owner_summary(self) -> Dict[str, Any]:
        return {
            "display_name": self.owner_display_name,
            "phone_number": self.owner_phone_number,
            "workspace_subdir": self.owner_workspace_subdir,
            "configured": bool(self.owner_phone_number),
        }

    def redacted_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "public_base_url": self.public_base_url,
            "webhook_signature_required": self.webhook_signature_required,
            "webhook_secret_configured": bool(self.webhook_secret),
            "twilio_from_number": self.twilio_from_number,
            "owner": self.owner_summary(),
            "contacts": [contact.model_dump(mode="json") for contact in self.contacts],
        }


def phone_config_path(state_dir: Path | str) -> Path:
    return Path(state_dir).expanduser() / "phone" / "config.json"


def load_phone_runtime_config(state_dir: Path | str) -> PhoneRuntimeConfig:
    path = phone_config_path(state_dir)
    if not path.exists():
        return PhoneRuntimeConfig()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return PhoneRuntimeConfig.model_validate(payload)
    except Exception:
        pass
    return PhoneRuntimeConfig()


def save_phone_runtime_config(
    state_dir: Path | str,
    config: PhoneRuntimeConfig,
) -> Path:
    path = phone_config_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.model_dump(mode="json"), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return path
