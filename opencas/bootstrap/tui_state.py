"""Shared state and provider/model discovery helpers for the bootstrap TUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from open_llm_auth.config import load_config
from open_llm_auth.provider_catalog import get_builtin_provider_models


class WizardState:
    def __init__(self) -> None:
        # Core config
        self.agent_profile_id: str = "general_technical_operator"
        self.persona_name: str = "OpenCAS"
        self.user_name: str = ""
        self.user_bio: str = ""
        self.state_dir: str = "./.opencas"
        self.workspace_root: str = "."
        self.workspace_extra: str = ""
        self.managed_workspace_root: str = ""
        self.provider_mode: str = "auto"
        self.provider_config_path: str = ""
        self.provider_env_path: str = ""
        self.credential_source_config: str = ""
        self.credential_source_env_path: str = ""
        self.credential_env_keys: List[str] = []
        self.selected_profiles: List[str] = []
        self.default_llm_model: Optional[str] = None
        self.embedding_model_id: str = "google/embeddinggemma-300m"
        self.model_routing_mode: str = "single"
        self.routing_light_model: str = ""
        self.routing_standard_model: str = ""
        self.routing_high_model: str = ""
        self.routing_extra_high_model: str = ""
        self.routing_auto_escalation: bool = True
        self.qdrant_url: str = ""
        self.qdrant_api_key: str = ""
        self.qdrant_collection: str = "opencas_embeddings"
        self.hnsw_enabled: bool = True
        self.hnsw_m: str = "16"
        self.hnsw_ef_construction: str = "200"
        self.mcp_auto_register: bool = False
        self.mcp_servers_json: str = ""
        self.telegram_enabled: bool = False
        self.telegram_bot_token: str = ""
        self.telegram_dm_policy: str = "pairing"
        self.telegram_allow_from: str = ""
        self.telegram_poll_interval_seconds: str = "1.0"
        self.telegram_pairing_ttl_seconds: str = "3600"
        self.telegram_api_base_url: str = "https://api.telegram.org"
        self.sandbox_mode: str = "workspace-only"
        self.sandbox_allowed_roots: str = ""
        self.use_server: bool = True
        self.host: str = "127.0.0.1"
        self.port: str = "8080"
        self.cycle_interval: str = "600"
        self.daydream_interval: str = "720"
        self.baa_heartbeat_interval: str = "120"
        self.consolidation_interval: str = "86400"
        self.with_embeddings: bool = True
        self.accepted_warning: bool = False

        # Partnership vision
        self.vision_main_help: str = ""
        self.vision_engagement_style: str = "collaborative"
        self.vision_success_six_months: str = ""
        self.vision_working_notes: str = ""

        # About me
        self.user_job: str = ""
        self.user_interests: str = ""
        self.user_comm_style: str = "mixed"

        # Learning style
        self.learning_preference: str = "hands_on"
        self.feedback_style: str = "mixed"
        self.help_style: str = "mixed"
        self.collab_pair: bool = False
        self.collab_async: bool = True
        self.collab_backforth: bool = False
        self.collab_minimal: bool = False

        # Emotional landscape
        self.happy_makers: str = ""
        self.sad_drainers: str = ""
        self.angry_triggers: str = ""
        self.agent_avoid: str = ""
        self.bad_day_help: str = ""

        # Initial goals
        self.goal_1: str = ""
        self.goal_2: str = ""
        self.goal_3: str = ""
        self.goal_timeframe: str = "mixed"

        # Persona theme
        self.persona_accent: str = "amber"


STATE = WizardState()


def scan_openllmauth_profiles() -> List[tuple[str, str]]:
    """Scan ~/.open_llm_auth/config.json for available auth profiles."""
    config_path = Path.home() / ".open_llm_auth" / "config.json"
    if not config_path.exists():
        return []
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        profiles = data.get("authProfiles", {})
        return [
            (pid, f"{pid} ({info.get('provider', 'unknown')})")
            for pid, info in profiles.items()
        ]
    except Exception:
        return []


def discover_model_choices(state: WizardState) -> Dict[str, List[tuple[str, str]]]:
    """Return chat and embedding choices from the active provider material."""
    config_path: Optional[Path] = None
    env_path: Optional[Path] = None
    if state.provider_mode == "custom":
        if state.provider_config_path:
            config_path = Path(state.provider_config_path).expanduser()
        if state.provider_env_path:
            env_path = Path(state.provider_env_path).expanduser()
    elif state.provider_mode == "copy":
        if state.credential_source_config:
            config_path = Path(state.credential_source_config).expanduser()
        if state.credential_source_env_path:
            env_path = Path(state.credential_source_env_path).expanduser()
    else:
        default_path = Path.home() / ".open_llm_auth" / "config.json"
        if default_path.exists():
            config_path = default_path

    providers: set[str] = set()
    chat_models: Dict[str, str] = {}
    embedding_models: Dict[str, str] = {"local-fallback": "local-fallback (offline fallback)"}

    try:
        cfg = load_config(config_path=config_path, env_path=env_path)
        provider_map = cfg.all_provider_configs()
        profile_map = cfg.all_auth_profiles()
        if state.provider_mode == "copy" and state.selected_profiles:
            for profile_id in state.selected_profiles:
                profile = profile_map.get(profile_id)
                if profile is not None:
                    providers.add(profile.provider)
        providers.update(provider_map.keys())
        providers.update(profile.provider for profile in profile_map.values())

        for provider_id in sorted(providers):
            provider_cfg = provider_map.get(provider_id)
            model_defs = list(getattr(provider_cfg, "models", []) or [])
            if not model_defs:
                for model in get_builtin_provider_models(provider_id):
                    try:
                        model_defs.append(type("ModelDef", (), model))
                    except Exception:
                        continue
            for model in model_defs:
                model_id = getattr(model, "id", None)
                if not model_id:
                    continue
                model_name = getattr(model, "name", None) or model_id
                full_ref = f"{provider_id}/{model_id}"
                label = f"{provider_id} / {model_name}"
                chat_models.setdefault(full_ref, label)
                if "embedding" in full_ref.lower():
                    embedding_models.setdefault(full_ref, label)
    except Exception:
        pass

    chat_models.setdefault(state.default_llm_model, state.default_llm_model)
    embedding_models.setdefault(state.embedding_model_id, state.embedding_model_id)

    return {
        "chat": sorted(((label, value) for value, label in chat_models.items()), key=lambda item: item[1].lower()),
        "embedding": sorted(((label, value) for value, label in embedding_models.items()), key=lambda item: item[1].lower()),
    }
