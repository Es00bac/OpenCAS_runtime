"""Config API routes for the OpenCAS dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel
from open_llm_auth.config import load_config

router = APIRouter(tags=["config"])


class ConfigResponse(BaseModel):
    config: Dict[str, Any]


class ProviderConfigResponse(BaseModel):
    providers: Dict[str, Any]


class ConfigOverviewResponse(BaseModel):
    overview: Dict[str, Any]


def _redact_secrets(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy and redit sensitive keys from a config dict."""
    import copy

    result = copy.deepcopy(config_dict)
    sensitive = {"api_key", "api_secret", "token", "password", "secret"}

    def _redact(obj):
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if any(s in key.lower() for s in sensitive):
                    obj[key] = "***"
                else:
                    _redact(value)
        elif isinstance(obj, list):
            for item in obj:
                _redact(item)

    _redact(result)
    return result


def _provider_summary(provider_id: str, provider_cfg: Any, profile_ids: List[str]) -> Dict[str, Any]:
    model_ids = [model.id for model in provider_cfg.models] if getattr(provider_cfg, "models", None) else []
    return {
        "provider_id": provider_id,
        "base_url": getattr(provider_cfg, "base_url", None),
        "auth": getattr(provider_cfg, "auth", None),
        "api": getattr(provider_cfg, "api", None),
        "configured_model_ids": model_ids,
        "configured_model_count": len(model_ids),
        "headers_present": bool(getattr(provider_cfg, "headers", None)),
        "profile_ids": profile_ids,
    }


def _profile_summary(profile_id: str, profile: Any) -> Dict[str, Any]:
    metadata = getattr(profile, "metadata", {}) or {}
    return {
        "profile_id": profile_id,
        "provider": getattr(profile, "provider", None),
        "type": getattr(profile, "type", None),
        "expired": bool(profile.is_expired()) if hasattr(profile, "is_expired") else False,
        "base_url": getattr(profile, "base_url", None),
        "account_id": getattr(profile, "account_id", None),
        "gateway_id": getattr(profile, "gateway_id", None),
        "metadata_keys": sorted(metadata.keys()),
    }


def _path_or_none(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _path_status(value: Any) -> Dict[str, Any]:
    path_str = _path_or_none(value)
    if not path_str:
        return {"path": None, "exists": False, "kind": None}
    path = Path(path_str).expanduser()
    kind = "dir" if path.is_dir() else "file" if path.exists() else None
    return {
        "path": str(path),
        "exists": path.exists(),
        "kind": kind,
    }


def _read_env_keys(path: Path) -> List[str]:
    if not path.exists():
        return []
    keys: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _value = line.split("=", 1)
        key = key.strip()
        if key:
            keys.append(key)
    return sorted(set(keys))


def _summarize_materialized_bundle(state_dir: Any) -> Dict[str, Any]:
    if not state_dir:
        return {
            "exists": False,
            "config_exists": False,
            "env_exists": False,
            "profile_count": 0,
            "provider_count": 0,
            "env_key_count": 0,
            "profile_ids": [],
            "provider_ids": [],
            "env_keys": [],
            "default_model": None,
            "config_path": None,
            "env_path": None,
        }

    bundle_dir = Path(_path_or_none(state_dir)).expanduser() / "provider_material"
    config_path = bundle_dir / "config.json"
    env_path = bundle_dir / ".env"
    summary = {
        "exists": bundle_dir.exists(),
        "config_exists": config_path.exists(),
        "env_exists": env_path.exists(),
        "profile_count": 0,
        "provider_count": 0,
        "env_key_count": 0,
        "profile_ids": [],
        "provider_ids": [],
        "env_keys": _read_env_keys(env_path),
        "default_model": None,
        "config_path": str(config_path),
        "env_path": str(env_path),
    }
    summary["env_key_count"] = len(summary["env_keys"])
    if not config_path.exists():
        return summary

    try:
        cfg = load_config(
            config_path=config_path,
            env_path=env_path if env_path.exists() else None,
        )
        profiles = cfg.all_auth_profiles() if hasattr(cfg, "all_auth_profiles") else {}
        providers = cfg.all_provider_configs() if hasattr(cfg, "all_provider_configs") else {}
        summary.update(
            {
                "profile_count": len(profiles),
                "provider_count": len(providers),
                "profile_ids": sorted(profiles.keys()),
                "provider_ids": sorted(providers.keys()),
                "default_model": getattr(cfg, "default_model", None),
            }
        )
    except Exception:
        pass
    return summary


def _detect_config_mode(runtime_config: Any, materialized: Dict[str, Any]) -> str:
    if (
        getattr(runtime_config, "credential_source_config_path", None) is not None
        or getattr(runtime_config, "credential_source_env_path", None) is not None
        or materialized.get("config_exists")
    ):
        return "copied-local"
    if (
        getattr(runtime_config, "provider_config_path", None) is not None
        or getattr(runtime_config, "provider_env_path", None) is not None
    ):
        return "custom-linked"
    return "shared-default"


def build_config_router(runtime: Any) -> APIRouter:
    """Build config routes wired to *runtime*."""
    r = APIRouter(prefix="/api/config", tags=["config"])

    @r.get("", response_model=ConfigResponse)
    async def get_config() -> ConfigResponse:
        raw = runtime.ctx.config.model_dump(mode="json")
        return ConfigResponse(config=_redact_secrets(raw))

    @r.get("/providers", response_model=ProviderConfigResponse)
    async def get_providers() -> ProviderConfigResponse:
        mgr = getattr(runtime.ctx.llm, "manager", None)
        if mgr is None:
            mgr = getattr(runtime.ctx.llm, "provider_manager", None)
        cfg = getattr(mgr, "_config", None)
        providers: Dict[str, Any] = {}
        if cfg is not None:
            for name, profile in cfg.providers.items():
                providers[name] = _redact_secrets(profile.model_dump(mode="json"))
        return ProviderConfigResponse(providers=providers)

    @r.get("/overview", response_model=ConfigOverviewResponse)
    async def get_config_overview() -> ConfigOverviewResponse:
        mgr = getattr(runtime.ctx.llm, "manager", None)
        if mgr is None:
            mgr = getattr(runtime.ctx.llm, "provider_manager", None)
        cfg = getattr(mgr, "_config", None)

        available_models: List[str] = []
        if mgr is not None and hasattr(mgr, "list_models"):
            try:
                model_list = await mgr.list_models()
                available_models = sorted(
                    {
                        str(item.get("id"))
                        for item in model_list
                        if isinstance(item, dict) and item.get("id")
                    }
                )
            except Exception:
                available_models = []

        providers: List[Dict[str, Any]] = []
        profiles: List[Dict[str, Any]] = []
        if cfg is not None:
            profile_map = cfg.all_auth_profiles() if hasattr(cfg, "all_auth_profiles") else {}
            provider_map = cfg.all_provider_configs() if hasattr(cfg, "all_provider_configs") else {}
            for profile_id, profile in sorted(profile_map.items()):
                profiles.append(_profile_summary(profile_id, profile))
            profiles_by_provider: Dict[str, List[str]] = {}
            for entry in profiles:
                provider_id = entry.get("provider")
                if not provider_id:
                    continue
                profiles_by_provider.setdefault(provider_id, []).append(entry["profile_id"])
            for provider_id, provider_cfg in sorted(provider_map.items()):
                providers.append(
                    _provider_summary(
                        provider_id,
                        provider_cfg,
                        sorted(profiles_by_provider.get(provider_id, [])),
                    )
                )

        runtime_config = runtime.ctx.config
        configured_chat_model = getattr(runtime_config, "default_llm_model", None)
        configured_embedding_model = getattr(runtime_config, "embedding_model_id", None)
        effective_chat_model = getattr(getattr(runtime.ctx, "llm", None), "default_model", None) or configured_chat_model
        effective_embedding_model = getattr(getattr(runtime.ctx, "embeddings", None), "model_id", None) or configured_embedding_model
        embedding_models = sorted(
            {
                model_id
                for model_id in available_models
                if "embedding" in model_id.lower()
            }
        )
        if effective_embedding_model:
            embedding_models = sorted(set(embedding_models + [effective_embedding_model]))
        if "local-fallback" not in embedding_models:
            embedding_models.append("local-fallback")
        state_dir = getattr(runtime_config, "state_dir", None)
        materialized = _summarize_materialized_bundle(state_dir)
        config_mode = _detect_config_mode(runtime_config, materialized)
        expired_profiles = sum(1 for profile in profiles if profile.get("expired"))
        healthy_profiles = len(profiles) - expired_profiles

        overview = {
            "config_mode": config_mode,
            "paths": {
                "state_dir": _path_or_none(getattr(runtime_config, "state_dir", None)),
                "provider_config_path": _path_or_none(getattr(runtime_config, "provider_config_path", None)),
                "provider_env_path": _path_or_none(getattr(runtime_config, "provider_env_path", None)),
                "credential_source_config_path": _path_or_none(getattr(runtime_config, "credential_source_config_path", None)),
                "credential_source_env_path": _path_or_none(getattr(runtime_config, "credential_source_env_path", None)),
            },
            "path_status": {
                "state_dir": _path_status(getattr(runtime_config, "state_dir", None)),
                "provider_config_path": _path_status(getattr(runtime_config, "provider_config_path", None)),
                "provider_env_path": _path_status(getattr(runtime_config, "provider_env_path", None)),
                "credential_source_config_path": _path_status(getattr(runtime_config, "credential_source_config_path", None)),
                "credential_source_env_path": _path_status(getattr(runtime_config, "credential_source_env_path", None)),
            },
            "current": {
                "configured_default_llm_model": configured_chat_model,
                "configured_embedding_model_id": configured_embedding_model,
                "default_llm_model": effective_chat_model,
                "embedding_model_id": effective_embedding_model,
                "chat_model_available": bool(effective_chat_model and effective_chat_model in available_models),
                "embedding_model_available": bool(
                    effective_embedding_model
                    and effective_embedding_model in set(embedding_models)
                ),
            },
            "credential_health": {
                "provider_count": len(providers),
                "profile_count": len(profiles),
                "healthy_profile_count": healthy_profiles,
                "expired_profile_count": expired_profiles,
                "available_model_count": len(available_models),
                "embedding_model_count": len(embedding_models),
            },
            "credential_copy": {
                "profile_ids": list(getattr(runtime_config, "credential_profile_ids", []) or []),
                "env_keys": list(getattr(runtime_config, "credential_env_keys", []) or []),
            },
            "materialized_bundle": materialized,
            "available_models": available_models,
            "available_embedding_models": embedding_models,
            "providers": providers,
            "auth_profiles": profiles,
        }
        return ConfigOverviewResponse(overview=overview)

    return r
