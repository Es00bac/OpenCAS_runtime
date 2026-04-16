"""Config API routes for the OpenCAS dashboard."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel
from open_llm_auth.provider_setup_catalog import get_provider_setup_catalog

from opencas.api.config_mutations import (
    delete_auth_profile as delete_auth_profile_impl,
    delete_provider as delete_provider_impl,
    delete_provider_model as delete_provider_model_impl,
    save_guided_provider_setup as save_guided_provider_setup_impl,
    save_model_routing as save_model_routing_impl,
    test_provider_connection as test_provider_connection_impl,
)
from opencas.api.config_overview import (
    build_config_overview_payload,
    redact_secrets,
)
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

    return r
