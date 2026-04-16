"""Bootstrap TUI persistence and config-building helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from opencas.bootstrap import BootstrapConfig
from opencas.model_routing import ModelRoutingConfig, ModelRoutingMode
from opencas.sandbox import SandboxConfig
from opencas.sandbox.config import SandboxMode
from opencas.bootstrap.tui_state import WizardState


def compose_user_bio(state: WizardState) -> str:
    """Compose a rich user_bio from questionnaire answers if none was given."""
    if state.user_bio:
        return state.user_bio
    parts: List[str] = []
    if state.user_job:
        parts.append(f"Role: {state.user_job}")
    if state.user_interests:
        parts.append(f"Interests: {state.user_interests}")
    if state.vision_main_help:
        parts.append(f"Wants help with: {state.vision_main_help}")
    if state.vision_engagement_style:
        parts.append(f"Engagement preference: {state.vision_engagement_style.replace('_', ' ')}")
    if state.vision_success_six_months:
        parts.append(f"6-month success: {state.vision_success_six_months}")
    if state.learning_preference:
        parts.append(f"Learning preference: {state.learning_preference.replace('_', ' ')}")
    if state.feedback_style:
        parts.append(f"Feedback preference: {state.feedback_style.replace('_', ' ')}")
    if state.help_style:
        parts.append(f"Help preference: {state.help_style.replace('_', ' ')}")
    if state.happy_makers:
        parts.append(f"Energized by: {state.happy_makers}")
    if state.sad_drainers:
        parts.append(f"Drained by: {state.sad_drainers}")
    if state.angry_triggers:
        parts.append(f"Frustrated by: {state.angry_triggers}")
    if state.agent_avoid:
        parts.append(f"Avoid: {state.agent_avoid}")
    if state.bad_day_help:
        parts.append(f"Recovery: {state.bad_day_help}")
    return "\n".join(parts)


def questionnaire_payload(state: WizardState) -> Dict[str, Dict[str, object]]:
    return {
        "vision": {
            "main_help": state.vision_main_help,
            "engagement_style": state.vision_engagement_style,
            "success_six_months": state.vision_success_six_months,
            "working_notes": state.vision_working_notes,
        },
        "about_me": {
            "job": state.user_job,
            "interests": state.user_interests,
            "communication_style": state.user_comm_style,
        },
        "learning_style": {
            "preference": state.learning_preference,
            "feedback_style": state.feedback_style,
            "help_style": state.help_style,
            "collab_pair": state.collab_pair,
            "collab_async": state.collab_async,
            "collab_backforth": state.collab_backforth,
            "collab_minimal": state.collab_minimal,
        },
        "emotional_landscape": {
            "happy_makers": state.happy_makers,
            "sad_drainers": state.sad_drainers,
            "angry_triggers": state.angry_triggers,
            "agent_avoid": state.agent_avoid,
            "bad_day_help": state.bad_day_help,
        },
        "initial_goals": {
            "goal_1": state.goal_1,
            "goal_2": state.goal_2,
            "goal_3": state.goal_3,
            "timeframe": state.goal_timeframe,
        },
        "persona_theme": {
            "accent": state.persona_accent,
        },
    }


def save_questionnaire(state: WizardState, state_dir: Path) -> Path:
    path = state_dir / "bootstrap_questionnaire.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(questionnaire_payload(state), indent=2), encoding="utf-8")
    return path


def _optional_path(value: str) -> Path | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    return Path(cleaned)


def _optional_text(value: str) -> str | None:
    cleaned = value.strip()
    return cleaned or None


def _split_items(value: str) -> List[str]:
    tokens = value.replace("\n", ",").split(",")
    return [token.strip() for token in tokens if token.strip()]


def _parse_int(value: str, *, default: int, field_name: str) -> int:
    cleaned = value.strip()
    if not cleaned:
        return default
    try:
        return int(cleaned)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _parse_float(value: str, *, default: float, field_name: str) -> float:
    cleaned = value.strip()
    if not cleaned:
        return default
    try:
        return float(cleaned)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number") from exc


def _parse_mcp_servers(value: str) -> List[Dict[str, Any]]:
    cleaned = value.strip()
    if not cleaned:
        return []
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("MCP servers must be valid JSON") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError("MCP servers JSON must be a list of objects")
    return parsed


def build_bootstrap_config(state: WizardState) -> BootstrapConfig:
    extra_roots = [r.strip() for r in state.workspace_extra.split(",") if r.strip()]
    state_dir = Path(state.state_dir)
    provider_config_path = None
    provider_env_path = None
    credential_source_config_path = None
    credential_source_env_path = None
    credential_profile_ids: List[str] = []
    credential_env_keys: List[str] = []

    if state.provider_mode == "custom":
        provider_config_path = _optional_path(state.provider_config_path)
        provider_env_path = _optional_path(state.provider_env_path)
    elif state.provider_mode == "copy":
        credential_source_config_path = _optional_path(state.credential_source_config)
        credential_source_env_path = _optional_path(state.credential_source_env_path)
        credential_profile_ids = list(state.selected_profiles)
        credential_env_keys = list(state.credential_env_keys)

    return BootstrapConfig(
        state_dir=state_dir,
        session_id=None,
        agent_profile_id=state.agent_profile_id,
        workspace_root=Path(state.workspace_root),
        workspace_roots=[Path(r) for r in extra_roots],
        managed_workspace_root=_optional_path(state.managed_workspace_root),
        default_llm_model=state.default_llm_model,
        embedding_model_id=state.embedding_model_id,
        model_routing=ModelRoutingConfig(
            mode=ModelRoutingMode(state.model_routing_mode),
            single_model=state.default_llm_model if state.model_routing_mode == "single" else None,
            light_model=_optional_text(state.routing_light_model),
            standard_model=_optional_text(state.routing_standard_model),
            high_model=_optional_text(state.routing_high_model),
            extra_high_model=_optional_text(state.routing_extra_high_model),
            auto_escalation=state.routing_auto_escalation,
        ),
        provider_config_path=provider_config_path,
        provider_env_path=provider_env_path,
        credential_source_config_path=credential_source_config_path,
        credential_source_env_path=credential_source_env_path,
        credential_profile_ids=credential_profile_ids,
        credential_env_keys=credential_env_keys,
        qdrant_url=_optional_text(state.qdrant_url),
        qdrant_api_key=_optional_text(state.qdrant_api_key),
        qdrant_collection=state.qdrant_collection.strip() or "opencas_embeddings",
        hnsw_enabled=state.hnsw_enabled,
        hnsw_m=_parse_int(state.hnsw_m, default=16, field_name="HNSW M"),
        hnsw_ef_construction=_parse_int(
            state.hnsw_ef_construction,
            default=200,
            field_name="HNSW ef_construction",
        ),
        mcp_servers=_parse_mcp_servers(state.mcp_servers_json),
        mcp_auto_register=state.mcp_auto_register,
        telegram_enabled=state.telegram_enabled,
        telegram_bot_token=_optional_text(state.telegram_bot_token),
        telegram_dm_policy=state.telegram_dm_policy,
        telegram_allow_from=_split_items(state.telegram_allow_from),
        telegram_poll_interval_seconds=_parse_float(
            state.telegram_poll_interval_seconds,
            default=1.0,
            field_name="Telegram poll interval",
        ),
        telegram_pairing_ttl_seconds=_parse_int(
            state.telegram_pairing_ttl_seconds,
            default=3600,
            field_name="Telegram pairing TTL",
        ),
        telegram_api_base_url=state.telegram_api_base_url.strip() or "https://api.telegram.org",
        sandbox=SandboxConfig(
            mode=SandboxMode(state.sandbox_mode),
            allowed_roots=[Path(root) for root in _split_items(state.sandbox_allowed_roots)],
        ),
        persona_name=state.persona_name,
        user_name=state.user_name or None,
        user_bio=compose_user_bio(state) or None,
    )
