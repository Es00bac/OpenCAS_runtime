"""Mutation helpers for the config control plane."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from open_llm_auth.auth.manager import ProviderManager
from open_llm_auth.config import AuthProfile, ModelDefinitionConfig, ProviderConfig
from open_llm_auth.provider_catalog import get_builtin_provider_config, get_builtin_provider_models, normalize_provider_id
from open_llm_auth.provider_setup_catalog import get_provider_setup_preset

from opencas.api.gateway_admin import load_active_gateway_config, reload_runtime_gateway, save_active_gateway_config
from opencas.governance import PluginTrustLevel, PluginTrustScope, WebTrustLevel
from opencas.model_routing import (
    ComplexityTier,
    ModelRoutingConfig,
    PersistedModelRoutingState,
    sanitize_model_routing_state,
    save_persisted_model_routing_state,
)
from opencas.provenance_events_adapter import ProvenanceEventType, emit_provenance_event


def slugify_profile_label(value: str) -> str:
    lowered = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value or "").strip())
    lowered = "-".join(part for part in lowered.split("-") if part)
    return lowered or "default"


def dedupe_profile_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def merge_custom_models(existing: List[Any], custom_model_ids: List[str]) -> List[ModelDefinitionConfig]:
    by_id: Dict[str, ModelDefinitionConfig] = {}
    for item in existing or []:
        model_id = str(getattr(item, "id", "")).strip()
        if not model_id:
            continue
        if isinstance(item, ModelDefinitionConfig):
            by_id[model_id] = item
        else:
            by_id[model_id] = ModelDefinitionConfig(id=model_id, name=model_id)
    for model_id in custom_model_ids:
        clean = str(model_id or "").strip()
        if not clean:
            continue
        by_id.setdefault(clean, ModelDefinitionConfig(id=clean, name=clean))
    return list(by_id.values())


def available_gateway_model_ids(cfg: Any) -> List[str]:
    provider_map = cfg.all_provider_configs() if hasattr(cfg, "all_provider_configs") else {}
    ordered: List[str] = []
    seen: set[str] = set()

    def add_model_ref(provider_id: str, model_id: str) -> None:
        clean = str(model_id or "").strip()
        if not clean:
            return
        ref = clean if "/" in clean else f"{provider_id}/{clean}"
        if ref in seen:
            return
        seen.add(ref)
        ordered.append(ref)

    for provider_id, provider_cfg in provider_map.items():
        for model in getattr(provider_cfg, "models", None) or []:
            add_model_ref(provider_id, getattr(model, "id", ""))
        builtins = list(get_builtin_provider_models(provider_id))
        preferred = [item for item in builtins if bool(item.get("reasoning"))]
        preferred.extend(item for item in builtins if not bool(item.get("reasoning")))
        for model in preferred:
            add_model_ref(provider_id, model.get("id") or "")
    return ordered


def _emit_settings_provenance_event(
    payload: Dict[str, Any],
    *,
    event_type: ProvenanceEventType,
    triggering_artifact: str,
    triggering_action: str,
    parent_link_id: str | None = None,
    linked_link_ids: List[str] | None = None,
    details: Dict[str, Any],
) -> Dict[str, Any]:
    emit_provenance_event(
        payload,
        event_type=event_type,
        triggering_artifact=triggering_artifact,
        triggering_action=triggering_action,
        parent_link_id=parent_link_id,
        linked_link_ids=linked_link_ids,
        details=details,
    )
    return payload


def apply_sanitized_runtime_model_selection(runtime: Any, cfg: Any) -> Dict[str, Any]:
    current_default = (
        getattr(runtime.ctx.config, "default_llm_model", None)
        or getattr(getattr(runtime.ctx, "llm", None), "default_model", None)
        or getattr(cfg, "default_model", None)
    )
    current_routing = getattr(runtime.ctx.config, "model_routing", None) or ModelRoutingConfig()
    sanitized = sanitize_model_routing_state(
        current_default,
        current_routing,
        available_gateway_model_ids(cfg),
    )
    runtime.ctx.config.default_llm_model = sanitized.default_llm_model
    runtime.ctx.config.model_routing = sanitized.model_routing
    llm = getattr(runtime.ctx, "llm", None)
    if llm is not None and hasattr(llm, "set_model_routing"):
        llm.set_model_routing(
            default_model=sanitized.default_llm_model,
            model_routing=sanitized.model_routing,
        )
    cfg.default_model = sanitized.default_llm_model
    settings_path = save_persisted_model_routing_state(
        Path(runtime.ctx.config.state_dir),
        sanitized,
    )
    response = {"state": sanitized, "settings_path": str(settings_path)}
    return _emit_settings_provenance_event(
        response,
        event_type=ProvenanceEventType.MUTATION,
        triggering_artifact="setting|config|model-routing",
        triggering_action="UPDATE",
        parent_link_id=str(settings_path),
        linked_link_ids=[str(settings_path)],
        details={
            "settings_path": str(settings_path),
            "default_llm_model": sanitized.default_llm_model,
        },
    )


def build_provider_config_for_preset(
    provider_id: str,
    *,
    base_url: Optional[str],
    headers: Dict[str, str],
    custom_model_ids: List[str],
    existing_cfg: Optional[ProviderConfig],
) -> ProviderConfig:
    builtin_raw = get_builtin_provider_config(provider_id) or {}
    builtin_cfg = ProviderConfig.model_validate(builtin_raw)
    merged_headers = dict(getattr(existing_cfg, "headers", {}) or {})
    merged_headers.update(headers or {})
    models = merge_custom_models(
        list(getattr(existing_cfg, "models", []) or []),
        custom_model_ids,
    )
    return builtin_cfg.model_copy(
        update={
            "base_url": base_url or getattr(existing_cfg, "base_url", None) or builtin_cfg.base_url,
            "headers": merged_headers,
            "models": models,
        }
    )


async def save_model_routing(runtime: Any, payload: Any) -> Dict[str, Any]:
    cfg, material = load_active_gateway_config(runtime.ctx.config)
    default_model = (
        payload.default_llm_model
        or getattr(runtime.ctx.config, "default_llm_model", None)
        or getattr(getattr(runtime.ctx, "llm", None), "default_model", None)
    )
    sanitized = sanitize_model_routing_state(
        default_model,
        payload.model_routing,
        available_gateway_model_ids(cfg),
    )
    routing = sanitized.model_routing
    standard_model = sanitized.default_llm_model or routing.resolve_model(
        default_model=default_model,
        complexity=ComplexityTier.STANDARD,
    )
    if not standard_model:
        raise HTTPException(status_code=400, detail="A standard/default chat model is required")

    runtime.ctx.config.default_llm_model = standard_model
    runtime.ctx.config.model_routing = routing
    llm = getattr(runtime.ctx, "llm", None)
    if llm is not None and hasattr(llm, "set_model_routing"):
        llm.set_model_routing(default_model=standard_model, model_routing=routing)

    state = PersistedModelRoutingState(
        default_llm_model=standard_model,
        model_routing=routing,
    )
    settings_path = save_persisted_model_routing_state(Path(runtime.ctx.config.state_dir), state)

    cfg.default_model = standard_model
    save_active_gateway_config(material, cfg)
    reload_runtime_gateway(runtime, material)

    response = {
        "status": "ok",
        "default_llm_model": standard_model,
        "model_routing": routing.model_dump(mode="json"),
        "settings_path": str(settings_path),
    }
    return _emit_settings_provenance_event(
        response,
        event_type=ProvenanceEventType.MUTATION,
        triggering_artifact="setting|config|model-routing",
        triggering_action="UPDATE",
        parent_link_id=str(settings_path),
        linked_link_ids=[str(settings_path)],
        details={
            "default_llm_model": standard_model,
            "settings_path": str(settings_path),
        },
    )


async def save_guided_provider_setup(runtime: Any, payload: Any) -> Dict[str, Any]:
    preset = get_provider_setup_preset(payload.family_id, payload.preset_id)
    if preset is None:
        raise HTTPException(status_code=404, detail="Provider setup preset not found")

    provider_id = normalize_provider_id(str(preset["provider_id"]))
    cfg, material = load_active_gateway_config(runtime.ctx.config)
    existing_provider_cfg = cfg.all_provider_configs().get(provider_id)
    provider_cfg = build_provider_config_for_preset(
        provider_id,
        base_url=payload.base_url,
        headers=payload.headers,
        custom_model_ids=payload.custom_model_ids,
        existing_cfg=existing_provider_cfg,
    )
    cfg.providers[provider_id] = provider_cfg

    created_profile_id: Optional[str] = None
    auth_kind = str(preset.get("auth_kind") or "api_key")
    if preset.get("supports_auth_profile"):
        profile_label = slugify_profile_label(payload.profile_label)
        created_profile_id = f"{provider_id}:{profile_label}"
        existing_profile = cfg.all_auth_profiles().get(created_profile_id)
        if auth_kind == "api_key":
            secret = (payload.api_key or "").strip()
            if not secret and existing_profile and getattr(existing_profile, "type", None) == "api_key":
                secret = getattr(existing_profile, "key", None) or ""
            if not secret:
                raise HTTPException(status_code=400, detail="API key is required for this provider setup")
            profile = AuthProfile(
                id=created_profile_id,
                provider=provider_id,
                type="api_key",
                key=secret,
                base_url=payload.base_url,
            )
        elif auth_kind == "oauth":
            access = (payload.access_token or "").strip()
            if not access and existing_profile and getattr(existing_profile, "type", None) == "oauth":
                access = getattr(existing_profile, "access", None) or ""
            if not access:
                raise HTTPException(status_code=400, detail="Access token is required for this provider setup")
            profile = AuthProfile(
                id=created_profile_id,
                provider=provider_id,
                type="oauth",
                access=access,
                refresh=((payload.refresh_token or "").strip() or getattr(existing_profile, "refresh", None)),
                expires=payload.expires_at if payload.expires_at is not None else getattr(existing_profile, "expires", None),
                base_url=payload.base_url,
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported guided authentication mode")
        cfg.auth_profiles[created_profile_id] = profile
        current_order = list(cfg.all_auth_order().get(provider_id, []))
        cfg.auth_order[provider_id] = dedupe_profile_order([created_profile_id, *current_order])

    if not cfg.default_model and preset.get("default_model"):
        cfg.default_model = str(preset["default_model"])

    sanitize_info = apply_sanitized_runtime_model_selection(runtime, cfg)
    save_active_gateway_config(material, cfg)
    reload_runtime_gateway(runtime, material)

    response = {
        "status": "ok",
        "provider_id": provider_id,
        "profile_id": created_profile_id,
        "default_model": cfg.default_model,
        "config_mode": material.mode,
        "settings_path": sanitize_info["settings_path"],
    }
    return _emit_settings_provenance_event(
        response,
        event_type=ProvenanceEventType.MUTATION,
        triggering_artifact=f"setting|config|provider-setup|{provider_id}",
        triggering_action="UPDATE",
        parent_link_id=f"provider:{provider_id}",
        linked_link_ids=[created_profile_id, sanitize_info["settings_path"]],
        details={
            "profile_id": created_profile_id,
            "settings_path": sanitize_info["settings_path"],
        },
    )


async def delete_auth_profile(runtime: Any, profile_id: str) -> Dict[str, Any]:
    cfg, material = load_active_gateway_config(runtime.ctx.config)
    removed = False
    if profile_id in cfg.auth_profiles:
        del cfg.auth_profiles[profile_id]
        removed = True
    if profile_id in cfg.auth.profiles:
        del cfg.auth.profiles[profile_id]
        removed = True
    for provider_id, ordered in list(cfg.auth_order.items()):
        cfg.auth_order[provider_id] = [item for item in ordered if item != profile_id]
    for provider_id, ordered in list(cfg.auth.order.items()):
        cfg.auth.order[provider_id] = [item for item in ordered if item != profile_id]
    if not removed:
        raise HTTPException(status_code=404, detail="Auth profile not found")
    save_active_gateway_config(material, cfg)
    reload_runtime_gateway(runtime, material)
    response = {"status": "ok", "deleted": profile_id}
    return _emit_settings_provenance_event(
        response,
        event_type=ProvenanceEventType.MUTATION,
        triggering_artifact=f"setting|config|auth-profile|{profile_id}",
        triggering_action="DELETE",
        parent_link_id=f"auth-profile:{profile_id}",
        linked_link_ids=[profile_id],
        details={"profile_id": profile_id},
    )


async def delete_provider(runtime: Any, provider_id: str) -> Dict[str, Any]:
    cfg, material = load_active_gateway_config(runtime.ctx.config)
    normalized = normalize_provider_id(provider_id)
    removed = False
    for mapping_name in ("providers",):
        mapping = getattr(cfg, mapping_name)
        for key in list(mapping.keys()):
            if normalize_provider_id(key) == normalized:
                del mapping[key]
                removed = True
    for mapping_name in ("providers",):
        mapping = getattr(cfg.models, mapping_name)
        for key in list(mapping.keys()):
            if normalize_provider_id(key) == normalized:
                del mapping[key]
                removed = True
    for mapping_name in ("auth_profiles",):
        mapping = getattr(cfg, mapping_name)
        for key in list(mapping.keys()):
            profile = mapping[key]
            if normalize_provider_id(getattr(profile, "provider", "")) == normalized:
                del mapping[key]
                removed = True
    for mapping_name in ("profiles",):
        mapping = getattr(cfg.auth, mapping_name)
        for key in list(mapping.keys()):
            profile = mapping[key]
            if normalize_provider_id(getattr(profile, "provider", "")) == normalized:
                del mapping[key]
                removed = True
    cfg.auth_order.pop(normalized, None)
    cfg.auth.order.pop(normalized, None)
    if not removed:
        raise HTTPException(status_code=404, detail="Provider not found")
    sanitize_info = apply_sanitized_runtime_model_selection(runtime, cfg)
    save_active_gateway_config(material, cfg)
    reload_runtime_gateway(runtime, material)
    response = {
        "status": "ok",
        "deleted": normalized,
        "settings_path": sanitize_info["settings_path"],
    }
    return _emit_settings_provenance_event(
        response,
        event_type=ProvenanceEventType.MUTATION,
        triggering_artifact=f"setting|config|provider|{normalized}",
        triggering_action="DELETE",
        parent_link_id=f"provider:{normalized}",
        linked_link_ids=[normalized, sanitize_info["settings_path"]],
        details={"provider_id": normalized, "settings_path": sanitize_info["settings_path"]},
    )


async def delete_provider_model(runtime: Any, provider_id: str, model_id: str) -> Dict[str, Any]:
    cfg, material = load_active_gateway_config(runtime.ctx.config)
    normalized_provider = normalize_provider_id(provider_id)
    normalized_model = str(model_id or "").strip()
    if not normalized_model:
        raise HTTPException(status_code=400, detail="Model id is required")

    removed = False
    for mapping in (cfg.providers, cfg.models.providers):
        provider_cfg = None
        matched_key = None
        for key, value in mapping.items():
            if normalize_provider_id(key) == normalized_provider:
                matched_key = key
                provider_cfg = value
                break
        if provider_cfg is None or matched_key is None:
            continue
        remaining = [
            model
            for model in list(getattr(provider_cfg, "models", []) or [])
            if str(getattr(model, "id", "")).strip() != normalized_model
        ]
        if len(remaining) != len(list(getattr(provider_cfg, "models", []) or [])):
            mapping[matched_key] = provider_cfg.model_copy(update={"models": remaining})
            removed = True

    if not removed:
        raise HTTPException(status_code=404, detail="Configured model not found")

    sanitize_info = apply_sanitized_runtime_model_selection(runtime, cfg)
    save_active_gateway_config(material, cfg)
    reload_runtime_gateway(runtime, material)
    response = {
        "status": "ok",
        "provider_id": normalized_provider,
        "deleted_model_id": normalized_model,
        "settings_path": sanitize_info["settings_path"],
    }
    return _emit_settings_provenance_event(
        response,
        event_type=ProvenanceEventType.MUTATION,
        triggering_artifact=f"setting|config|provider-model|{normalized_provider}/{normalized_model}",
        triggering_action="DELETE",
        parent_link_id=f"provider-model:{normalized_provider}/{normalized_model}",
        linked_link_ids=[normalized_provider, normalized_model, sanitize_info["settings_path"]],
        details={
            "provider_id": normalized_provider,
            "model_id": normalized_model,
            "settings_path": sanitize_info["settings_path"],
        },
    )


async def test_provider_connection(runtime: Any, payload: Any) -> Dict[str, Any]:
    provider_id = normalize_provider_id(payload.provider_id)
    cfg, material = load_active_gateway_config(runtime.ctx.config)
    manager = ProviderManager(config_path=material.config_path, env_path=material.env_path)
    model_ref = payload.model_ref
    if not model_ref:
        provider_cfg = cfg.all_provider_configs().get(provider_id)
        model_ids = [
            str(model.id)
            for model in getattr(provider_cfg, "models", []) or []
            if getattr(model, "id", None)
        ]
        if not model_ids:
            model_ids = [
                str(model.get("id"))
                for model in get_builtin_provider_models(provider_id)
                if model.get("id")
            ]
        model_ref = f"{provider_id}/{model_ids[0]}" if model_ids else cfg.default_model or f"{provider_id}/assistant"
    try:
        resolved = manager.resolve(model_ref, preferred_profile=payload.profile_id)
        models = await resolved.provider.list_models()
        response = {
            "status": "ok",
            "provider_id": resolved.provider_id,
            "profile_id": resolved.profile_id,
            "model_ref": model_ref,
            "models_found": len(models),
        }
    except Exception as exc:
        response = {
            "status": "error",
            "provider_id": provider_id,
            "profile_id": payload.profile_id,
            "model_ref": model_ref,
            "detail": str(exc),
        }
        return _emit_settings_provenance_event(
            response,
            event_type=ProvenanceEventType.BLOCKED,
            triggering_artifact=f"setting|config|provider-test|{provider_id}",
            triggering_action="CHECK",
            parent_link_id=f"provider-test:{provider_id}",
            linked_link_ids=[provider_id, model_ref],
            details={
                "profile_id": payload.profile_id,
                "model_ref": model_ref,
                "detail": str(exc),
            },
        )
    else:
        return _emit_settings_provenance_event(
            response,
            event_type=ProvenanceEventType.CHECK,
            triggering_artifact=f"setting|config|provider-test|{provider_id}",
            triggering_action="CHECK",
            parent_link_id=f"provider-test:{provider_id}",
            linked_link_ids=[provider_id, model_ref],
            details={
                "profile_id": resolved.profile_id,
                "model_ref": model_ref,
                "models_found": len(models),
            },
        )


async def save_web_trust_policy(
    runtime: Any,
    *,
    domain: str,
    level: str,
    note: Optional[str] = None,
    source: str = "user",
) -> Dict[str, Any]:
    service = getattr(runtime.ctx, "web_trust", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Web trust service is unavailable")
    try:
        policy = await service.set_policy(
            domain,
            WebTrustLevel(level),
            note=note or "",
            source=source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response = {
        "status": "ok",
        "policy": {
            "domain": policy.domain,
            "level": policy.level.value,
            "source": policy.source,
            "note": policy.note,
            "updated_at": policy.updated_at.isoformat(),
        },
    }
    return _emit_settings_provenance_event(
        response,
        event_type=ProvenanceEventType.MUTATION,
        triggering_artifact=f"setting|trust|web|{policy.domain}",
        triggering_action="UPDATE",
        parent_link_id=f"trust:web:{policy.domain}",
        linked_link_ids=[policy.domain],
        details={"domain": policy.domain, "level": policy.level.value},
    )


async def save_plugin_trust_policy(
    runtime: Any,
    *,
    scope: str,
    value: str,
    level: str,
    note: Optional[str] = None,
    source: str = "user",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    service = getattr(runtime.ctx, "plugin_trust", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Plugin trust service is unavailable")
    try:
        policy = await service.set_policy(
            PluginTrustScope(scope),
            value,
            PluginTrustLevel(level),
            note=note or "",
            source=source,
            metadata=metadata or {},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response = {
        "status": "ok",
        "policy": {
            "scope": policy.scope.value,
            "value": policy.value,
            "level": policy.level.value,
            "source": policy.source,
            "note": policy.note,
            "metadata": dict(policy.metadata),
            "updated_at": policy.updated_at.isoformat(),
        },
    }
    return _emit_settings_provenance_event(
        response,
        event_type=ProvenanceEventType.MUTATION,
        triggering_artifact=f"setting|trust|plugin|{policy.scope.value}:{policy.value}",
        triggering_action="UPDATE",
        parent_link_id=f"trust:plugin:{policy.scope.value}:{policy.value}",
        linked_link_ids=[policy.scope.value, policy.value],
        details={"scope": policy.scope.value, "value": policy.value, "level": policy.level.value},
    )


async def sync_plugin_trust_feed(
    runtime: Any,
    *,
    feed: Dict[str, Any],
) -> Dict[str, Any]:
    service = getattr(runtime.ctx, "plugin_trust", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Plugin trust service is unavailable")
    try:
        result = await service.sync_feed(feed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response = {
        "status": "ok",
        "feed": {
            "source_id": result.source_id,
            "source": result.source,
            "format_version": result.format_version,
            "verification": dict(result.verification),
            "imported": list(result.imported),
            "removed": list(result.removed),
            "skipped_conflicts": list(result.skipped_conflicts),
            "rejected": list(result.rejected),
        },
    }
    return _emit_settings_provenance_event(
        response,
        event_type=ProvenanceEventType.MUTATION,
        triggering_artifact=f"setting|trust|plugin-feed|{result.source_id}",
        triggering_action="UPDATE",
        parent_link_id=f"trust-feed:{result.source_id}",
        linked_link_ids=[result.source_id],
        details={
            "source_id": result.source_id,
            "imported_count": len(result.imported),
            "removed_count": len(result.removed),
        },
    )


async def delete_plugin_trust_policy(
    runtime: Any,
    *,
    scope: str,
    value: str,
) -> Dict[str, Any]:
    service = getattr(runtime.ctx, "plugin_trust", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Plugin trust service is unavailable")
    try:
        await service.remove_policy(PluginTrustScope(scope), value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response = {"status": "ok", "deleted": {"scope": scope, "value": value}}
    return _emit_settings_provenance_event(
        response,
        event_type=ProvenanceEventType.MUTATION,
        triggering_artifact=f"setting|trust|plugin|{scope}:{value}",
        triggering_action="DELETE",
        parent_link_id=f"trust:plugin:{scope}:{value}",
        linked_link_ids=[scope, value],
        details={"scope": scope, "value": value},
    )


async def delete_web_trust_policy(runtime: Any, *, domain: str) -> Dict[str, Any]:
    service = getattr(runtime.ctx, "web_trust", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Web trust service is unavailable")
    await service.remove_policy(domain)
    response = {"status": "ok", "deleted": domain}
    return _emit_settings_provenance_event(
        response,
        event_type=ProvenanceEventType.MUTATION,
        triggering_artifact=f"setting|trust|web|{domain}",
        triggering_action="DELETE",
        parent_link_id=f"trust:web:{domain}",
        linked_link_ids=[domain],
        details={"domain": domain},
    )
