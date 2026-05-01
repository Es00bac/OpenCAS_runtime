"""Persistent phone bridge configuration for OpenCAS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


PhoneTrustLevel = Literal["owner", "low"]
PhoneAllowedAction = Literal["leave_message", "knowledge_qa"]
PhoneWorkspaceAccess = Literal["read_only", "append_only"]
PhoneWorkspaceScope = Literal["shared", "caller"]
PhoneMenuActionKind = Literal[
    "workspace_assistant",
    "say_then_hangup",
    "time_announcement",
    "submenu",
    "owner_conversation",
]
PhoneTtsMode = Literal["fast", "expressive"]


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


def _normalize_workspace_subdir(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip().replace("\\", "/")
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("workspace_subdir must stay inside the managed workspace")
    return "/".join(part for part in candidate.parts if part not in {"."})


def _normalize_allowed_actions(
    value: Any,
    *,
    default: Optional[List[PhoneAllowedAction]] = None,
) -> List[PhoneAllowedAction]:
    if value is None:
        return list(default or [])
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
        return _normalize_workspace_subdir(value)

    @field_validator("allowed_actions", mode="before")
    @classmethod
    def _normalize_allowed_actions(cls, value: Any) -> List[PhoneAllowedAction]:
        return _normalize_allowed_actions(value, default=["leave_message"])


class PhoneWorkspaceMount(BaseModel):
    """One workspace mount available to a menu workflow."""

    scope: PhoneWorkspaceScope = "caller"
    subdir: str
    access: PhoneWorkspaceAccess = "read_only"

    @field_validator("subdir")
    @classmethod
    def _normalize_workspace_subdir(cls, value: Any) -> str:
        cleaned = _normalize_workspace_subdir(value)
        if not cleaned:
            raise ValueError("workspace mount subdir is required")
        return cleaned


class PhoneMenuOption(BaseModel):
    """One configurable phone menu option."""

    key: str
    digit: str
    action: PhoneMenuActionKind
    label: str = ""
    phrases: List[str] = Field(default_factory=list)
    greeting: str = ""
    message: str = ""
    message_template: str = ""
    prompt_profile: Optional[str] = None
    target_menu: Optional[str] = None
    allowed_actions: List[PhoneAllowedAction] = Field(default_factory=lambda: ["leave_message", "knowledge_qa"])
    workspace_mounts: List[PhoneWorkspaceMount] = Field(default_factory=list)
    time_zone: Optional[str] = None

    @field_validator("key", "label", "greeting", "message", "message_template")
    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("digit")
    @classmethod
    def _normalize_digit(cls, value: Any) -> str:
        cleaned = str(value or "").strip()
        if len(cleaned) != 1:
            raise ValueError("menu digit must be a single character")
        return cleaned

    @field_validator("prompt_profile", "target_menu")
    @classmethod
    def _normalize_optional_profile(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip().replace("\\", "/")
        if not cleaned:
            return None
        candidate = Path(cleaned)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("prompt_profile must stay inside the phone profiles directory")
        return "/".join(part for part in candidate.parts if part not in {"."})

    @field_validator("time_zone")
    @classmethod
    def _normalize_optional_time_zone(cls, value: Any) -> Optional[str]:
        cleaned = str(value or "").strip()
        return cleaned or None

    @field_validator("phrases", mode="before")
    @classmethod
    def _normalize_phrases(cls, value: Any) -> List[str]:
        raw_items = value if isinstance(value, list) else ([] if value is None else [value])
        normalized: List[str] = []
        seen = set()
        for item in raw_items:
            cleaned = " ".join(str(item or "").strip().lower().split())
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized

    @field_validator("allowed_actions", mode="before")
    @classmethod
    def _normalize_allowed_actions(cls, value: Any) -> List[PhoneAllowedAction]:
        return _normalize_allowed_actions(value, default=["leave_message", "knowledge_qa"])

    @field_validator("workspace_mounts", mode="before")
    @classmethod
    def _normalize_workspace_mounts(cls, value: Any) -> List[PhoneWorkspaceMount]:
        raw_items = list(value or [])
        mounts: List[PhoneWorkspaceMount] = []
        seen = set()
        for item in raw_items:
            mount = item if isinstance(item, PhoneWorkspaceMount) else PhoneWorkspaceMount.model_validate(item)
            key = (mount.scope, mount.subdir, mount.access)
            if key in seen:
                continue
            seen.add(key)
            mounts.append(mount)
        return mounts

    @model_validator(mode="after")
    def _validate_action_requirements(self) -> "PhoneMenuOption":
        if self.action == "workspace_assistant" and not self.workspace_mounts:
            raise ValueError("workspace_assistant menu options require workspace_mounts")
        if self.action == "time_announcement" and not self.time_zone:
            raise ValueError("time_announcement menu options require time_zone")
        if self.action == "submenu" and not self.target_menu:
            raise ValueError("submenu menu options require target_menu")
        return self


class PhoneMenuDefinition(BaseModel):
    """One named phone menu."""

    key: str
    prompt: str = ""
    reprompt: str = ""
    options: List[PhoneMenuOption] = Field(default_factory=list)

    @field_validator("key", "prompt", "reprompt")
    @classmethod
    def _normalize_menu_text(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("options", mode="before")
    @classmethod
    def _dedupe_options(cls, value: Any) -> List[PhoneMenuOption]:
        raw_items = list(value or [])
        deduped: List[PhoneMenuOption] = []
        seen_digits = set()
        seen_keys = set()
        for item in raw_items:
            option = item if isinstance(item, PhoneMenuOption) else PhoneMenuOption.model_validate(item)
            if option.digit in seen_digits or option.key in seen_keys:
                continue
            seen_digits.add(option.digit)
            seen_keys.add(option.key)
            deduped.append(option)
        return deduped


class PhoneMenuConfig(BaseModel):
    """External JSON-configured phone menus."""

    default_menu_key: str = "public_main"
    owner_menu_key: Optional[str] = None
    menu_prompt: str = ""
    menu_reprompt: str = ""
    owner_pin_prompt: str = ""
    owner_pin_retry_prompt: str = ""
    owner_pin_success_message: str = ""
    owner_pin_failure_message: str = ""
    menus: List[PhoneMenuDefinition] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_menu_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if "menus" in value:
            return value
        default_menu_key = str(value.get("default_menu_key") or "public_main").strip() or "public_main"
        legacy_menu = {
            "key": default_menu_key,
            "prompt": value.get("menu_prompt", ""),
            "reprompt": value.get("menu_reprompt", ""),
            "options": value.get("options", []),
        }
        return {
            **value,
            "default_menu_key": default_menu_key,
            "menus": [legacy_menu],
        }

    @field_validator(
        "default_menu_key",
        "owner_menu_key",
        "menu_prompt",
        "menu_reprompt",
        "owner_pin_prompt",
        "owner_pin_retry_prompt",
        "owner_pin_success_message",
        "owner_pin_failure_message",
    )
    @classmethod
    def _normalize_prompt_text(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("menus", mode="before")
    @classmethod
    def _dedupe_menus(cls, value: Any) -> List[PhoneMenuDefinition]:
        raw_items = list(value or [])
        deduped: List[PhoneMenuDefinition] = []
        seen_keys = set()
        for item in raw_items:
            menu = item if isinstance(item, PhoneMenuDefinition) else PhoneMenuDefinition.model_validate(item)
            if not menu.key or menu.key in seen_keys:
                continue
            seen_keys.add(menu.key)
            deduped.append(menu)
        return deduped

    @model_validator(mode="after")
    def _validate_menu_references(self) -> "PhoneMenuConfig":
        menu_keys = {menu.key for menu in self.menus if menu.key}
        if self.default_menu_key and self.default_menu_key not in menu_keys and self.menus:
            self.default_menu_key = self.menus[0].key
        if self.owner_menu_key and self.owner_menu_key not in menu_keys:
            self.owner_menu_key = None
        for menu in self.menus:
            for option in menu.options:
                if option.action == "submenu" and option.target_menu not in menu_keys:
                    raise ValueError(f"submenu target_menu '{option.target_menu}' is not defined")
        return self


class PhoneRuntimeConfig(BaseModel):
    """Persisted runtime settings for Twilio phone integration."""

    enabled: bool = False
    public_base_url: Optional[str] = None
    webhook_signature_required: bool = True
    webhook_secret: Optional[str] = None
    menu_config_path: Optional[str] = None
    twilio_env_path: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    twilio_api_key_sid: Optional[str] = None
    twilio_api_secret: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_from_number: Optional[str] = None
    owner_phone_number: Optional[str] = None
    owner_display_name: str = "Operator"
    owner_workspace_subdir: str = "phone/owner"
    owner_pin: Optional[str] = None
    elevenlabs_env_path: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_voice_id: Optional[str] = None
    elevenlabs_stt_model: str = "scribe_v2"
    elevenlabs_fast_model: str = "eleven_flash_v2_5"
    elevenlabs_expressive_model: str = "eleven_v3"
    edge_tts_voice: str = "Aira"
    phone_tts_mode: PhoneTtsMode = "fast"
    phone_min_utterance_bytes: int = 800
    phone_silence_gap_seconds: float = 1.0
    phone_speech_rms_threshold: int = 180
    phone_preroll_ms: int = 320
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

    @field_validator("menu_config_path")
    @classmethod
    def _normalize_optional_config_path(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @field_validator(
        "twilio_env_path",
        "twilio_account_sid",
        "twilio_api_key_sid",
        "twilio_api_secret",
        "twilio_auth_token",
        "owner_pin",
        "elevenlabs_env_path",
        "elevenlabs_api_key",
        "elevenlabs_voice_id",
    )
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @field_validator("owner_pin")
    @classmethod
    def _normalize_owner_pin(cls, value: Any) -> Optional[str]:
        cleaned = str(value or "").strip()
        if not cleaned:
            return None
        digits = "".join(ch for ch in cleaned if ch.isdigit())
        if len(digits) != 6:
            raise ValueError("owner_pin must be exactly 6 digits")
        return digits

    @field_validator(
        "elevenlabs_stt_model",
        "elevenlabs_fast_model",
        "elevenlabs_expressive_model",
        "edge_tts_voice",
    )
    @classmethod
    def _normalize_required_text(cls, value: Any, info: Any) -> str:
        default_map = {
            "elevenlabs_stt_model": "scribe_v2",
            "elevenlabs_fast_model": "eleven_flash_v2_5",
            "elevenlabs_expressive_model": "eleven_v3",
            "edge_tts_voice": "Aira",
        }
        cleaned = str(value or "").strip()
        return cleaned or default_map.get(str(info.field_name), "")

    @field_validator("phone_tts_mode")
    @classmethod
    def _normalize_phone_tts_mode(cls, value: Any) -> PhoneTtsMode:
        cleaned = str(value or "fast").strip() or "fast"
        return cleaned  # type: ignore[return-value]

    @field_validator("phone_min_utterance_bytes", mode="before")
    @classmethod
    def _normalize_phone_min_utterance_bytes(cls, value: Any) -> int:
        try:
            cleaned = int(value)
        except Exception:
            cleaned = 800
        return max(160, min(cleaned, 16000))

    @field_validator("phone_silence_gap_seconds", mode="before")
    @classmethod
    def _normalize_phone_silence_gap_seconds(cls, value: Any) -> float:
        try:
            cleaned = float(value)
        except Exception:
            cleaned = 1.0
        return max(0.3, min(cleaned, 3.0))

    @field_validator("phone_speech_rms_threshold", mode="before")
    @classmethod
    def _normalize_phone_speech_rms_threshold(cls, value: Any) -> int:
        try:
            cleaned = int(value)
        except Exception:
            cleaned = 180
        return max(40, min(cleaned, 2000))

    @field_validator("phone_preroll_ms", mode="before")
    @classmethod
    def _normalize_phone_preroll_ms(cls, value: Any) -> int:
        try:
            cleaned = int(value)
        except Exception:
            cleaned = 320
        return max(0, min(cleaned, 2000))

    @field_validator("owner_workspace_subdir")
    @classmethod
    def _normalize_owner_workspace_subdir(cls, value: Any) -> str:
        cleaned = _normalize_workspace_subdir(value)
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
            "pin_configured": bool(self.owner_pin),
            "configured": bool(self.owner_phone_number),
        }

    def redacted_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "public_base_url": self.public_base_url,
            "webhook_signature_required": self.webhook_signature_required,
            "webhook_secret_configured": bool(self.webhook_secret),
            "menu_config_path": self.menu_config_path,
            "twilio_env_path": self.twilio_env_path,
            "twilio_account_sid": self.twilio_account_sid,
            "twilio_api_key_sid": self.twilio_api_key_sid,
            "twilio_api_secret_configured": bool(self.twilio_api_secret),
            "twilio_auth_token_configured": bool(self.twilio_auth_token),
            "twilio_from_number": self.twilio_from_number,
            "owner": self.owner_summary(),
            "elevenlabs_env_path": self.elevenlabs_env_path,
            "elevenlabs_api_key_configured": bool(self.elevenlabs_api_key),
            "elevenlabs_voice_id": self.elevenlabs_voice_id,
            "elevenlabs_stt_model": self.elevenlabs_stt_model,
            "elevenlabs_fast_model": self.elevenlabs_fast_model,
            "elevenlabs_expressive_model": self.elevenlabs_expressive_model,
            "edge_tts_voice": self.edge_tts_voice,
            "phone_tts_mode": self.phone_tts_mode,
            "phone_min_utterance_bytes": self.phone_min_utterance_bytes,
            "phone_silence_gap_seconds": self.phone_silence_gap_seconds,
            "phone_speech_rms_threshold": self.phone_speech_rms_threshold,
            "phone_preroll_ms": self.phone_preroll_ms,
            "contacts": [contact.model_dump(mode="json") for contact in self.contacts],
        }


def phone_config_path(state_dir: Path | str) -> Path:
    return Path(state_dir).expanduser() / "phone" / "config.json"


def phone_dashboard_menu_path(state_dir: Path | str) -> Path:
    return Path(state_dir).expanduser() / "phone" / "dashboard_menu.json"


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


def save_phone_menu_config(path: Path | str, config: PhoneMenuConfig) -> Path:
    config_path = Path(path).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config.model_dump(mode="json"), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return config_path


def default_phone_menu_config() -> PhoneMenuConfig:
    """Return the built-in public phone menu for fresh installs."""

    return PhoneMenuConfig(
        default_menu_key="public_main",
        owner_menu_key="owner_entry",
        owner_pin_prompt="Please enter your six digit owner PIN now.",
        owner_pin_retry_prompt="That PIN did not match. Please try once more.",
        owner_pin_success_message="PIN verified.",
        owner_pin_failure_message="Sorry, I couldn't verify the owner PIN. Goodbye.",
        menus=[
            {
                "key": "owner_entry",
                "prompt": "Press 1 to continue as the owner, or press 2 for the main menu.",
                "reprompt": "Press 1 for owner mode, or press 2 for the main menu.",
                "options": [
                    {
                        "key": "owner_continue",
                        "digit": "1",
                        "action": "owner_conversation",
                        "label": "Owner conversation",
                        "message": "Go ahead.",
                    },
                    {
                        "key": "owner_main_menu",
                        "digit": "2",
                        "action": "submenu",
                        "label": "Main menu",
                        "target_menu": "public_main",
                    },
                ],
            },
            {
                "key": "public_main",
                "prompt": (
                    "Hi, this is the OpenCAS phone bridge. Potential employers, press 1 or say employer. "
                    "Everyone else, press 2."
                ),
                "reprompt": (
                    "Please press 1 or say employer if you're calling about work opportunities. "
                    "Otherwise, press 2."
                ),
                "options": [
                    {
                        "key": "employer",
                        "digit": "1",
                        "action": "workspace_assistant",
                        "label": "Potential employer",
                        "phrases": ["employer", "work", "recruiter", "hiring", "job"],
                        "greeting": (
                            "You're connected to the OpenCAS phone bridge in work mode. "
                            "I can answer questions about the owner's approved resume, skills, "
                            "and current projects, and I can take a message for follow-up."
                        ),
                        "prompt_profile": "worksafe_owner",
                        "allowed_actions": ["leave_message", "knowledge_qa"],
                        "workspace_mounts": [
                            {
                                "scope": "shared",
                                "subdir": "phone/employer_shared",
                                "access": "read_only",
                            },
                            {
                                "scope": "caller",
                                "subdir": "phone/employers/{phone_digits}",
                                "access": "append_only",
                            },
                        ],
                    },
                    {
                        "key": "reject",
                        "digit": "2",
                        "action": "say_then_hangup",
                        "label": "Not for this line",
                        "phrases": ["other", "not employer", "personal"],
                        "message": (
                            "Sorry, this line is reserved for employment inquiries. "
                            "Please check the website for public information."
                        ),
                    },
                ],
            },
        ],
    )


def load_phone_menu_config(path: Path | str) -> PhoneMenuConfig:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        return default_phone_menu_config()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return PhoneMenuConfig.model_validate(payload)
    except Exception:
        pass
    return default_phone_menu_config()


def summarize_phone_session_profiles(menu: PhoneMenuConfig) -> Dict[str, Any]:
    def _menu_by_key(key: str) -> Optional[PhoneMenuDefinition]:
        for item in menu.menus:
            if item.key == key:
                return item
        return None

    def _option_by_key(menu_def: Optional[PhoneMenuDefinition], key: str) -> Optional[PhoneMenuOption]:
        if menu_def is None:
            return None
        for item in menu_def.options:
            if item.key == key:
                return item
        return None

    def _workspace_mount_subdir(
        option: Optional[PhoneMenuOption],
        *,
        scope: PhoneWorkspaceScope,
        access: PhoneWorkspaceAccess,
    ) -> Optional[str]:
        if option is None:
            return None
        for mount in option.workspace_mounts:
            if mount.scope == scope and mount.access == access:
                return mount.subdir
        return None

    owner_menu = _menu_by_key(menu.owner_menu_key or "owner_entry") or _menu_by_key("owner_entry")
    public_menu = _menu_by_key(menu.default_menu_key or "public_main") or _menu_by_key("public_main")
    owner_continue = _option_by_key(owner_menu, "owner_continue")
    owner_main_menu = _option_by_key(owner_menu, "owner_main_menu")
    employer = _option_by_key(public_menu, "employer")
    reject = _option_by_key(public_menu, "reject")

    return {
        "owner_entry": {
            "menu_key": owner_menu.key if owner_menu else None,
            "prompt": owner_menu.prompt if owner_menu else "",
            "reprompt": owner_menu.reprompt if owner_menu else "",
            "continue_digit": owner_continue.digit if owner_continue else "1",
            "fallback_digit": owner_main_menu.digit if owner_main_menu else "2",
        },
        "public_main": {
            "menu_key": public_menu.key if public_menu else None,
            "prompt": public_menu.prompt if public_menu else "",
            "reprompt": public_menu.reprompt if public_menu else "",
        },
        "owner_pin": {
            "prompt": menu.owner_pin_prompt,
            "retry_prompt": menu.owner_pin_retry_prompt,
            "success_message": menu.owner_pin_success_message,
            "failure_message": menu.owner_pin_failure_message,
        },
        "employer": {
            "enabled": employer is not None,
            "digit": employer.digit if employer else "1",
            "label": employer.label if employer else "Potential employer",
            "phrases": list(employer.phrases) if employer else [],
            "greeting": employer.greeting if employer else "",
            "prompt_profile": employer.prompt_profile if employer else "worksafe_owner",
            "allowed_actions": list(employer.allowed_actions) if employer else ["leave_message", "knowledge_qa"],
            "shared_workspace_subdir": _workspace_mount_subdir(
                employer, scope="shared", access="read_only"
            )
            or "phone/employer_shared",
            "caller_workspace_subdir": _workspace_mount_subdir(
                employer, scope="caller", access="append_only"
            )
            or "phone/employers/{phone_digits}",
        },
        "reject": {
            "enabled": reject is not None,
            "digit": reject.digit if reject else "2",
            "label": reject.label if reject else "Not for this line",
            "phrases": list(reject.phrases) if reject else [],
            "message": reject.message if reject else "",
        },
    }
