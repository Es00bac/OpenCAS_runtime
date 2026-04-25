"""Config API routes for the OpenCAS dashboard."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field
from open_llm_auth.provider_setup_catalog import get_provider_setup_catalog

from opencas.api.config_mutations import (
    delete_auth_profile as delete_auth_profile_impl,
    delete_plugin_trust_policy as delete_plugin_trust_policy_impl,
    delete_provider as delete_provider_impl,
    delete_provider_model as delete_provider_model_impl,
    delete_web_trust_policy as delete_web_trust_policy_impl,
    save_plugin_trust_policy as save_plugin_trust_policy_impl,
    save_guided_provider_setup as save_guided_provider_setup_impl,
    save_model_routing as save_model_routing_impl,
    save_web_trust_policy as save_web_trust_policy_impl,
    sync_plugin_trust_feed as sync_plugin_trust_feed_impl,
    test_provider_connection as test_provider_connection_impl,
)
from opencas.api.config_overview import (
    build_config_overview_payload,
    redact_secrets,
)
from opencas.governance import PluginTrustLevel, PluginTrustScope, WebTrustLevel
from opencas.model_routing import ModelRoutingConfig

router = APIRouter(tags=["config"])


class ConfigResponse(BaseModel):
    config: Dict[str, Any]


class ProviderConfigResponse(BaseModel):
    providers: Dict[str, Any]


class ConfigOverviewResponse(BaseModel):
    overview: Dict[str, Any]


class ModelRoutingUpdateRequest(BaseModel):
    default_llm_model: Optional[str] = None
    model_routing: ModelRoutingConfig


class GuidedProviderSetupRequest(BaseModel):
    family_id: str
    preset_id: str
    profile_label: str = "default"
    api_key: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[int] = None
    base_url: Optional[str] = None
    custom_model_ids: List[str] = []
    headers: Dict[str, str] = {}


class ProviderConnectionTestRequest(BaseModel):
    provider_id: str
    profile_id: Optional[str] = None
    model_ref: Optional[str] = None


class WebTrustPolicyRequest(BaseModel):
    domain: str
    level: WebTrustLevel
    note: Optional[str] = None
    source: str = "user"


class PluginTrustPolicyRequest(BaseModel):
    scope: PluginTrustScope
    value: str
    level: PluginTrustLevel
    note: Optional[str] = None
    source: str = "user"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PluginTrustFeedEntryRequest(BaseModel):
    scope: PluginTrustScope
    value: str
    level: PluginTrustLevel
    note: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PluginTrustFeedSignatureRequest(BaseModel):
    key_id: str
    algorithm: str = "ed25519"
    public_key: str
    signature: str


class PluginTrustFeedSyncRequest(BaseModel):
    format_version: int = 1
    source_id: str
    policies: List[PluginTrustFeedEntryRequest]
    signatures: List[PluginTrustFeedSignatureRequest] = Field(default_factory=list)


def build_config_router(runtime: Any) -> APIRouter:
    """Build config routes wired to *runtime*."""
    r = APIRouter(prefix="/api/config", tags=["config"])

    @r.get("", response_model=ConfigResponse)
    async def get_config() -> ConfigResponse:
        raw = runtime.ctx.config.model_dump(mode="json")
        return ConfigResponse(config=redact_secrets(raw))

    @r.get("/providers", response_model=ProviderConfigResponse)
    async def get_providers() -> ProviderConfigResponse:
        mgr = getattr(runtime.ctx.llm, "manager", None)
        if mgr is None:
            mgr = getattr(runtime.ctx.llm, "provider_manager", None)
        cfg = getattr(mgr, "_config", None)
        providers: Dict[str, Any] = {}
        if cfg is not None:
            for name, profile in cfg.providers.items():
                providers[name] = redact_secrets(profile.model_dump(mode="json"))
        return ProviderConfigResponse(providers=providers)

    @r.get("/overview", response_model=ConfigOverviewResponse)
    async def get_config_overview() -> ConfigOverviewResponse:
        overview = await build_config_overview_payload(runtime)
        overview["provider_setup_catalog"] = list(get_provider_setup_catalog().values())
        return ConfigOverviewResponse(overview=overview)

    @r.post("/model-routing")
    async def save_model_routing(
        payload: ModelRoutingUpdateRequest,
    ) -> Dict[str, Any]:
        return await save_model_routing_impl(runtime, payload)

    @r.post("/provider-setups")
    async def save_guided_provider_setup(
        payload: GuidedProviderSetupRequest,
    ) -> Dict[str, Any]:
        return await save_guided_provider_setup_impl(runtime, payload)

    @r.delete("/auth-profiles/{profile_id:path}")
    async def delete_auth_profile(profile_id: str) -> Dict[str, Any]:
        return await delete_auth_profile_impl(runtime, profile_id)

    @r.delete("/providers/{provider_id}")
    async def delete_provider(provider_id: str) -> Dict[str, Any]:
        return await delete_provider_impl(runtime, provider_id)

    @r.delete("/providers/{provider_id}/models/{model_id:path}")
    async def delete_provider_model(provider_id: str, model_id: str) -> Dict[str, Any]:
        return await delete_provider_model_impl(runtime, provider_id, model_id)

    @r.post("/provider-test")
    async def test_provider_connection(
        payload: ProviderConnectionTestRequest,
    ) -> Dict[str, Any]:
        return await test_provider_connection_impl(runtime, payload)

    @r.get("/web-trust")
    async def get_web_trust_overview(limit: int = 20) -> Dict[str, Any]:
        service = getattr(runtime.ctx, "web_trust", None)
        if service is None:
            return {"available": False, "entries": []}
        return await service.summary(limit=limit)

    @r.get("/plugin-trust")
    async def get_plugin_trust_overview(limit: int = 20) -> Dict[str, Any]:
        service = getattr(runtime.ctx, "plugin_trust", None)
        if service is None:
            return {"available": False, "entries": []}
        return await service.summary(limit=limit)

    @r.post("/plugin-trust/policies")
    async def save_plugin_trust_policy(
        payload: PluginTrustPolicyRequest,
    ) -> Dict[str, Any]:
        return await save_plugin_trust_policy_impl(
            runtime,
            scope=payload.scope.value,
            value=payload.value,
            level=payload.level.value,
            note=payload.note,
            source=payload.source,
            metadata=payload.metadata,
        )

    @r.post("/plugin-trust/feeds/sync")
    async def sync_plugin_trust_feed(
        payload: PluginTrustFeedSyncRequest,
    ) -> Dict[str, Any]:
        return await sync_plugin_trust_feed_impl(
            runtime,
            feed=payload.model_dump(mode="json"),
        )

    @r.delete("/plugin-trust/policies/{scope}/{value:path}")
    async def delete_plugin_trust_policy(scope: PluginTrustScope, value: str) -> Dict[str, Any]:
        return await delete_plugin_trust_policy_impl(runtime, scope=scope.value, value=value)

    @r.post("/web-trust/policies")
    async def save_web_trust_policy(
        payload: WebTrustPolicyRequest,
    ) -> Dict[str, Any]:
        return await save_web_trust_policy_impl(
            runtime,
            domain=payload.domain,
            level=payload.level.value,
            note=payload.note,
            source=payload.source,
        )

    @r.delete("/web-trust/policies/{domain:path}")
    async def delete_web_trust_policy(domain: str) -> Dict[str, Any]:
        return await delete_web_trust_policy_impl(runtime, domain=domain)

    return r
