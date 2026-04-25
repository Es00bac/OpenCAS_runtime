"""Utilities for app-local OpenLLMAuth credential materialization."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from open_llm_auth.config import AuthProfile, Config, load_config


@dataclass
class ProviderMaterialBundle:
    """Paths to app-local copied provider material."""

    config_path: Path
    env_path: Path
    copied_profile_ids: list[str]
    copied_env_keys: list[str]


def materialize_provider_material(
    target_dir: Path,
    *,
    source_config_path: Optional[Path] = None,
    source_env_path: Optional[Path] = None,
    profile_ids: Iterable[str] = (),
    env_keys: Iterable[str] = (),
    default_model: Optional[str] = None,
) -> ProviderMaterialBundle:
    """Copy selected provider profiles and env vars into an app-local bundle."""
    target_dir.mkdir(parents=True, exist_ok=True)
    config_path = target_dir / "config.json"
    env_path = target_dir / ".env"

    source_config = _load_source_config(
        source_config_path=source_config_path,
        source_env_path=source_env_path,
    )
    source_profiles = source_config.all_auth_profiles()
    source_provider_configs = source_config.all_provider_configs()

    copied_profiles: Dict[str, AuthProfile] = {}
    copied_provider_ids: set[str] = set()
    copied_profile_ids: list[str] = []
    for profile_id in profile_ids:
        profile = source_profiles.get(profile_id)
        if profile is None:
            continue
        copied_profiles[profile_id] = _copy_profile_with_resolved_secret(profile_id, profile)
        copied_provider_ids.add(profile.provider)
        copied_profile_ids.append(profile_id)

    cfg = Config()
    if default_model:
        cfg.default_model = default_model
    cfg.auth_profiles = copied_profiles
    for provider_id in copied_provider_ids:
        provider_cfg = source_provider_configs.get(provider_id)
        if provider_cfg is not None:
            cfg.providers[provider_id] = provider_cfg
        ordered = source_config.all_auth_order().get(provider_id, [])
        filtered = [profile_id for profile_id in ordered if profile_id in copied_profiles]
        if filtered:
            cfg.auth_order[provider_id] = filtered

    config_path.write_text(
        cfg.model_dump_json(indent=2, by_alias=True),
        encoding="utf-8",
    )

    source_env = _read_env_file(source_env_path)
    copied_env_keys: list[str] = []
    env_lines: list[str] = []
    for key in env_keys:
        value = source_env.get(key)
        if value is None:
            value = os.getenv(key)
        if value is None:
            continue
        copied_env_keys.append(key)
        env_lines.append(f"{key}={json.dumps(value)}")
    env_path.write_text("\n".join(env_lines) + ("\n" if env_lines else ""), encoding="utf-8")

    return ProviderMaterialBundle(
        config_path=config_path,
        env_path=env_path,
        copied_profile_ids=copied_profile_ids,
        copied_env_keys=copied_env_keys,
    )


def _load_source_config(
    *,
    source_config_path: Optional[Path],
    source_env_path: Optional[Path],
) -> Config:
    if source_config_path is not None:
        return load_config(config_path=source_config_path, env_path=source_env_path)
    default_config = Path.home() / ".open_llm_auth" / "config.json"
    if default_config.exists():
        return load_config(config_path=default_config, env_path=source_env_path)
    return Config()


def _copy_profile_with_resolved_secret(profile_id: str, profile: AuthProfile) -> AuthProfile:
    copied = profile.model_copy(deep=True)
    copied.id = profile_id
    secret = profile.secret()
    if copied.type == "api_key":
        copied.key = secret
    elif copied.type == "token":
        copied.token = secret
    else:
        copied.access = secret
    return copied


def _read_env_file(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        values[key] = value
    return values
