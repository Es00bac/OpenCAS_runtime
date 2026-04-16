"""Helpers for editing the active OpenLLMAuth material from OpenCAS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from open_llm_auth.config import CONFIG_DIR, CONFIG_FILE, Config, load_config


@dataclass
class ActiveGatewayMaterial:
    """Resolved OpenLLMAuth config/env files that the runtime is using."""

    mode: str
    config_path: Path
    env_path: Optional[Path]


def resolve_active_gateway_material(runtime_config: Any) -> ActiveGatewayMaterial:
    """Resolve the current OpenLLMAuth config and env files for this runtime."""
    state_dir = Path(getattr(runtime_config, "state_dir")).expanduser()
    bundle_dir = state_dir / "provider_material"
    bundle_config = bundle_dir / "config.json"
    bundle_env = bundle_dir / ".env"
    if (
        getattr(runtime_config, "credential_source_config_path", None) is not None
        or getattr(runtime_config, "credential_source_env_path", None) is not None
        or bundle_config.exists()
    ):
        return ActiveGatewayMaterial(
            mode="copied-local",
            config_path=bundle_config,
            env_path=bundle_env,
        )

    provider_config_path = getattr(runtime_config, "provider_config_path", None)
    provider_env_path = getattr(runtime_config, "provider_env_path", None)
    if provider_config_path is not None or provider_env_path is not None:
        return ActiveGatewayMaterial(
            mode="custom-linked",
            config_path=Path(provider_config_path or CONFIG_FILE).expanduser(),
            env_path=Path(provider_env_path).expanduser() if provider_env_path else None,
        )

    default_env = CONFIG_DIR / ".env"
    return ActiveGatewayMaterial(
        mode="shared-default",
        config_path=CONFIG_FILE,
        env_path=default_env,
    )


def load_active_gateway_config(runtime_config: Any) -> tuple[Config, ActiveGatewayMaterial]:
    """Load the effective OpenLLMAuth config for the current runtime."""
    material = resolve_active_gateway_material(runtime_config)
    cfg = load_config(
        config_path=material.config_path,
        env_path=material.env_path if material.env_path and material.env_path.exists() else None,
    )
    return cfg, material


def save_active_gateway_config(material: ActiveGatewayMaterial, cfg: Config) -> None:
    """Persist the edited OpenLLMAuth config back to the active material path."""
    material.config_path.parent.mkdir(parents=True, exist_ok=True)
    material.config_path.write_text(
        cfg.model_dump_json(indent=2, by_alias=True),
        encoding="utf-8",
    )
    if material.env_path is not None:
        material.env_path.parent.mkdir(parents=True, exist_ok=True)
        if not material.env_path.exists():
            material.env_path.write_text("", encoding="utf-8")


def reload_runtime_gateway(runtime: Any, material: ActiveGatewayMaterial) -> None:
    """Retarget and reload the runtime provider manager after config edits."""
    llm = getattr(runtime.ctx, "llm", None)
    mgr = getattr(llm, "manager", None)
    if mgr is None:
        mgr = getattr(llm, "provider_manager", None)
    if mgr is None:
        return
    if hasattr(mgr, "_config_path"):
        mgr._config_path = material.config_path
    if hasattr(mgr, "_env_path"):
        mgr._env_path = material.env_path
    if hasattr(mgr, "reload"):
        mgr.reload()
