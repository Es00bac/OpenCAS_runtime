"""Platform inventory API routes for canonical capability inspection."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from opencas import __version__ as OPENCAS_VERSION
from opencas.governance import PluginTrustAssessment
from opencas.platform import CapabilitySource, CapabilityStatus
from opencas.platform.projections import build_extension_descriptors
from opencas.plugins import classify_plugin_update, evaluate_plugin_compatibility
from opencas.plugins.manifest import SUPPORTED_PLUGIN_MANIFEST_VERSIONS
from opencas.plugins.package import PluginPackageError, inspect_plugin_bundle_details


class CapabilityPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str
    display_name: str
    kind: str
    source: CapabilitySource
    owner_id: str
    status: CapabilityStatus
    description: str
    tool_names: List[str]
    declared_dependencies: List[str]
    config_schema: Dict[str, Any]
    entrypoint: str | None
    manifest_path: str | None
    source_path: str | None
    validation_errors: List[str]
    metadata: Dict[str, Any]


class CapabilityInventoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capabilities: List[CapabilityPayload] = Field(default_factory=list)


class CapabilityDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: CapabilityPayload


class ExtensionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extension_id: str
    extension_kind: str
    display_name: str
    status: CapabilityStatus
    capability_ids: List[str] = Field(default_factory=list)
    manifest_version: int | None = None
    version: str | None = None
    manifest_path: str | None = None
    dependencies: List[str] = Field(default_factory=list)
    compatibility: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    release_notes: str = ""
    bundle: Dict[str, Any] = Field(default_factory=dict)
    trust: Dict[str, Any] = Field(default_factory=dict)
    config_schema_summary: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtensionInventoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extensions: List[ExtensionPayload] = Field(default_factory=list)


class ExtensionDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extension: ExtensionPayload
    capabilities: List[CapabilityPayload] = Field(default_factory=list)


class ExtensionLifecycleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    extension_id: str
    action: str


class ExtensionInstallResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    extension_id: str
    action: str
    stored_bundle: str | None = None
    manifest_version: int | None = None
    version: str | None = None
    previous_version: str | None = None
    change_type: str = "install"
    compatibility: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    release_notes: str = ""
    bundle: Dict[str, Any] = Field(default_factory=dict)
    trust: Dict[str, Any] = Field(default_factory=dict)


class ExtensionInstallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class ExtensionBundleInspectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extension_id: str
    manifest_version: int | None = None
    version: str | None = None
    previous_version: str | None = None
    change_type: str = "install"
    compatibility: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    release_notes: str = ""
    bundle: Dict[str, Any] = Field(default_factory=dict)
    trust: Dict[str, Any] = Field(default_factory=dict)


class PolicyFieldPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str


class PolicyRulePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    label: str
    description: str
    operator_guidance: str
    requires_existing: bool = False


class ExtensionInstallUpdatePolicyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime_version: str
    bundle_suffix: str
    supported_manifest_versions: List[int] = Field(default_factory=list)
    change_types: List[PolicyRulePayload] = Field(default_factory=list)
    compatibility_fields: List[PolicyFieldPayload] = Field(default_factory=list)
    provenance_fields: List[PolicyFieldPayload] = Field(default_factory=list)
    bundle_fields: List[PolicyFieldPayload] = Field(default_factory=list)
    lifecycle_actions: List[PolicyFieldPayload] = Field(default_factory=list)
    evaluation_rules: List[str] = Field(default_factory=list)
    operator_notes: List[str] = Field(default_factory=list)


def _capability_registry(runtime: Any) -> Any:
    registry = getattr(runtime, "capability_registry", None)
    if registry is not None:
        return registry
    ctx = getattr(runtime, "ctx", None)
    if ctx is not None:
        return getattr(ctx, "capability_registry", None)
    return None


def _require_capability_registry(runtime: Any) -> Any:
    registry = _capability_registry(runtime)
    if registry is None:
        raise HTTPException(status_code=503, detail="Capability registry is not available")
    return registry


def _capability_to_model(descriptor: Any) -> CapabilityPayload:
    return CapabilityPayload(
        capability_id=descriptor.capability_id,
        display_name=descriptor.display_name,
        kind=descriptor.kind,
        source=descriptor.source,
        owner_id=descriptor.owner_id,
        status=descriptor.status,
        description=descriptor.description,
        tool_names=list(descriptor.tool_names),
        declared_dependencies=list(descriptor.declared_dependencies),
        config_schema=dict(descriptor.config_schema),
        entrypoint=descriptor.entrypoint,
        manifest_path=descriptor.manifest_path,
        source_path=descriptor.source_path,
        validation_errors=list(descriptor.validation_errors),
        metadata=dict(descriptor.metadata),
    )


def _extension_to_model(descriptor: Any) -> ExtensionPayload:
    return ExtensionPayload(
        extension_id=descriptor.extension_id,
        extension_kind=descriptor.extension_kind,
        display_name=descriptor.display_name,
        status=descriptor.status,
        capability_ids=list(descriptor.capability_ids),
        manifest_version=descriptor.manifest_version,
        version=descriptor.version,
        manifest_path=descriptor.manifest_path,
        dependencies=list(descriptor.dependencies),
        compatibility=dict(descriptor.compatibility),
        provenance=dict(descriptor.provenance),
        release_notes=descriptor.release_notes,
        bundle={},
        trust={},
        config_schema_summary=dict(descriptor.config_schema_summary),
        errors=list(descriptor.errors),
        metadata=dict(descriptor.metadata),
    )


def _normalize_query_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _require_extension_descriptor(runtime: Any, extension_id: str) -> Any:
    registry = _require_capability_registry(runtime)
    for item in build_extension_descriptors(registry):
        if item.extension_id == extension_id:
            return item
    raise HTTPException(status_code=404, detail=f"Extension not found: {extension_id}")


def _find_extension_descriptor(runtime: Any, extension_id: str) -> Any | None:
    registry = _require_capability_registry(runtime)
    for item in build_extension_descriptors(registry):
        if item.extension_id == extension_id:
            return item
    return None


async def _installed_plugin_rows(runtime: Any) -> Dict[str, Dict[str, Any]]:
    store = getattr(runtime.ctx, "plugin_store", None)
    if store is None:
        return {}
    rows = await store.list_installed()
    return {
        str(row.get("plugin_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("plugin_id")
    }


def _serialize_plugin_trust_assessment(assessment: PluginTrustAssessment | None) -> Dict[str, Any]:
    if assessment is None:
        return {}
    return {
        "level": getattr(getattr(assessment, "level", None), "value", getattr(assessment, "level", None)),
        "certainty": round(float(getattr(assessment, "certainty", 0.0)), 3),
        "blocked": bool(getattr(assessment, "blocked", False)),
        "publisher": getattr(assessment, "publisher", None),
        "checksum": getattr(assessment, "checksum", None),
        "signer_ids": list(getattr(assessment, "signer_ids", []) or []),
        "verified_signer_ids": list(getattr(assessment, "verified_signer_ids", []) or []),
        "signature_count": int(getattr(assessment, "signature_count", 0) or 0),
        "verified_signature_count": int(getattr(assessment, "verified_signature_count", 0) or 0),
        "matched_policies": list(getattr(assessment, "matched_policies", []) or []),
        "reasons": list(getattr(assessment, "reasons", []) or []),
    }


def _assess_plugin_trust(
    runtime: Any,
    *,
    provenance: Dict[str, Any] | None,
    bundle: Dict[str, Any] | None,
) -> Dict[str, Any]:
    service = getattr(runtime.ctx, "plugin_trust", None)
    if service is None:
        return {}
    assessment = service.assess(provenance=provenance, bundle=bundle)
    return _serialize_plugin_trust_assessment(assessment)


def _require_plugin_extension(extension: Any) -> None:
    if extension.extension_kind != "plugin":
        raise HTTPException(
            status_code=409,
            detail=f"Extension does not support lifecycle actions: {extension.extension_id}",
        )


def _require_extension_lifecycle_backend(runtime: Any) -> None:
    enable_plugin = getattr(runtime, "enable_plugin", None)
    disable_plugin = getattr(runtime, "disable_plugin", None)
    uninstall_plugin = getattr(runtime, "uninstall_plugin", None)
    if not callable(enable_plugin) and not callable(disable_plugin) and not callable(uninstall_plugin):
        raise HTTPException(status_code=503, detail="Plugin lifecycle backend is not available")


async def _run_extension_lifecycle_action(
    runtime: Any,
    extension_id: str,
    action: str,
) -> ExtensionLifecycleResponse:
    extension = _require_extension_descriptor(runtime, extension_id)
    _require_plugin_extension(extension)
    _require_extension_lifecycle_backend(runtime)

    if action == "enable":
        enable_plugin = getattr(runtime, "enable_plugin", None)
        if not callable(enable_plugin):
            raise HTTPException(status_code=503, detail="Plugin lifecycle backend is not available")
        await enable_plugin(extension_id)
    elif action == "disable":
        disable_plugin = getattr(runtime, "disable_plugin", None)
        if not callable(disable_plugin):
            raise HTTPException(status_code=503, detail="Plugin lifecycle backend is not available")
        await disable_plugin(extension_id)
    elif action == "uninstall":
        uninstall_plugin = getattr(runtime, "uninstall_plugin", None)
        if not callable(uninstall_plugin):
            raise HTTPException(status_code=503, detail="Plugin lifecycle backend is not available")
        await uninstall_plugin(extension_id)
    else:
        raise HTTPException(status_code=500, detail=f"Unsupported lifecycle action: {action}")

    return ExtensionLifecycleResponse(extension_id=extension_id, action=action)


async def _install_extension_bundle(
    runtime: Any,
    path: str,
    *,
    action: str,
    expected_extension_id: str | None = None,
) -> ExtensionInstallResponse:
    install_plugin = getattr(runtime, "install_plugin", None)
    if not callable(install_plugin):
        raise HTTPException(status_code=503, detail="Plugin install backend is not available")

    inspection = _inspect_extension_bundle(
        runtime,
        path,
        expected_extension_id=expected_extension_id,
    )
    if inspection.trust.get("blocked", False):
        reasons = inspection.trust.get("reasons", [])
        reason_text = "; ".join(reasons) if reasons else "bundle trust policy blocks this extension"
        raise HTTPException(status_code=409, detail=reason_text)
    if not inspection.compatibility.get("compatible", True):
        reasons = inspection.compatibility.get("reasons", [])
        reason_text = "; ".join(reasons) if reasons else "bundle is not compatible with this runtime"
        raise HTTPException(status_code=409, detail=reason_text)

    plugin = await install_plugin(path)
    if plugin is None:
        raise HTTPException(status_code=400, detail="Failed to install extension bundle")

    return ExtensionInstallResponse(
        extension_id=plugin.plugin_id,
        action=action,
        stored_bundle=str(path),
        manifest_version=getattr(plugin, "manifest_version", None),
        version=getattr(plugin, "version", None),
        previous_version=inspection.previous_version,
        change_type=inspection.change_type,
        compatibility=dict(inspection.compatibility),
        provenance=dict(inspection.provenance),
        release_notes=inspection.release_notes,
        bundle=dict(inspection.bundle),
        trust=dict(inspection.trust),
    )


def _inspect_extension_bundle(
    runtime: Any,
    path: str,
    *,
    expected_extension_id: str | None = None,
) -> ExtensionBundleInspectionResponse:
    try:
        inspection = inspect_plugin_bundle_details(path)
    except PluginPackageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    manifest = inspection["manifest"]
    bundle = inspection["bundle"]

    extension_id = str(manifest["id"])
    if expected_extension_id is not None and extension_id != expected_extension_id:
        raise HTTPException(
            status_code=409,
            detail=f"Bundle extension id {extension_id} does not match expected {expected_extension_id}",
        )

    compatibility = evaluate_plugin_compatibility(manifest)
    existing = _find_extension_descriptor(runtime, extension_id)
    previous_version = getattr(existing, "version", None) if existing is not None else None
    change_type = classify_plugin_update(previous_version, manifest.get("version"))
    trust = _assess_plugin_trust(
        runtime,
        provenance=dict(manifest.get("distribution", {})),
        bundle=dict(bundle),
    )
    return ExtensionBundleInspectionResponse(
        extension_id=extension_id,
        manifest_version=manifest.get("manifest_version"),
        version=manifest.get("version"),
        previous_version=previous_version,
        change_type=change_type,
        compatibility=compatibility,
        provenance=dict(manifest.get("distribution", {})),
        release_notes=str(manifest.get("release_notes", "")),
        bundle=dict(bundle),
        trust=trust,
    )


def _build_extension_install_update_policy() -> ExtensionInstallUpdatePolicyResponse:
    return ExtensionInstallUpdatePolicyResponse(
        runtime_version=OPENCAS_VERSION,
        bundle_suffix=".opencas-plugin.zip",
        supported_manifest_versions=sorted(SUPPORTED_PLUGIN_MANIFEST_VERSIONS),
        change_types=[
            PolicyRulePayload(
                rule_id="install",
                label="Install",
                description="The bundle targets an extension id that is not currently installed.",
                operator_guidance="Use install for a new plugin or for a bundle whose extension id is not already present in the runtime inventory.",
                requires_existing=False,
            ),
            PolicyRulePayload(
                rule_id="upgrade",
                label="Upgrade",
                description="The bundle version is higher than the currently installed version for the same extension id.",
                operator_guidance="Upgrade replaces the installed plugin in place after compatibility passes. Review release notes and provenance before proceeding.",
                requires_existing=True,
            ),
            PolicyRulePayload(
                rule_id="downgrade",
                label="Downgrade",
                description="The bundle version is lower than the currently installed version for the same extension id.",
                operator_guidance="Downgrades are allowed when compatibility passes, but operators should confirm the rollback is intentional and verify changed behavior afterward.",
                requires_existing=True,
            ),
            PolicyRulePayload(
                rule_id="reinstall",
                label="Reinstall",
                description="The bundle version matches the currently installed version for the same extension id.",
                operator_guidance="Reinstall refreshes the packaged contents in place. Use it to recover from local drift or to reapply a known-good bundle.",
                requires_existing=True,
            ),
            PolicyRulePayload(
                rule_id="replace",
                label="Replace",
                description="The bundle cannot be compared as a numeric version transition, so the runtime treats it as a same-id replacement.",
                operator_guidance="Replace is the most conservative update classification. Inspect provenance, checksum, and release notes because version semantics were not strong enough to classify the transition more precisely.",
                requires_existing=True,
            ),
        ],
        compatibility_fields=[
            PolicyFieldPayload(
                name="min_opencas_version",
                description="Optional lower runtime bound declared by the plugin manifest. The bundle is blocked if the running OpenCAS version is below this value.",
            ),
            PolicyFieldPayload(
                name="max_opencas_version",
                description="Optional upper runtime bound declared by the plugin manifest. The bundle is blocked if the running OpenCAS version is above this value.",
            ),
            PolicyFieldPayload(
                name="reasons",
                description="Human-readable incompatibility reasons returned by the runtime when the bundle cannot be installed or updated safely.",
            ),
        ],
        provenance_fields=[
            PolicyFieldPayload(
                name="publisher",
                description="Who produced the bundle according to the manifest distribution metadata.",
            ),
            PolicyFieldPayload(
                name="channel",
                description="Declared release channel such as stable, beta, or internal. Operators can use this to distinguish high-trust releases from experimental ones.",
            ),
            PolicyFieldPayload(
                name="source_url",
                description="Upstream location for the bundle or source repository, when the manifest provides one.",
            ),
            PolicyFieldPayload(
                name="changelog_url",
                description="Optional external changelog link supplied by the bundle author.",
            ),
            PolicyFieldPayload(
                name="signatures",
                description="Optional manifest signature entries. Each entry can declare a signer key id, an Ed25519 public key, and a signature over the canonical bundle payload.",
            ),
        ],
        bundle_fields=[
            PolicyFieldPayload(
                name="filename",
                description="Uploaded archive name used for operator review and audit trails.",
            ),
            PolicyFieldPayload(
                name="sha256",
                description="Content digest for the uploaded archive. Operators should compare this when verifying bundle provenance out of band.",
            ),
            PolicyFieldPayload(
                name="size_bytes",
                description="Archive size in bytes, useful for spotting unexpectedly large or tiny bundles.",
            ),
            PolicyFieldPayload(
                name="member_count",
                description="Number of files inside the archive after safe member validation.",
            ),
            PolicyFieldPayload(
                name="signatures",
                description="Bundle inspection summary for cryptographic signatures, including how many signatures were present, how many verified, and per-signer verification status.",
            ),
        ],
        lifecycle_actions=[
            PolicyFieldPayload(
                name="enable",
                description="Marks an installed plugin extension as enabled through the runtime lifecycle manager.",
            ),
            PolicyFieldPayload(
                name="disable",
                description="Marks an installed plugin extension as disabled without removing it from disk.",
            ),
            PolicyFieldPayload(
                name="uninstall",
                description="Removes the installed plugin from the live registry, plugin store, and managed plugin directory.",
            ),
        ],
        evaluation_rules=[
            "Bundle inspection always runs before install or update, and the extension id in the manifest is authoritative.",
            "Update operations require the uploaded bundle extension id to match the targeted installed extension id exactly.",
            "Compatibility is evaluated against the running OpenCAS version before any install or update mutation occurs.",
            "Incompatible bundles are rejected with a blocking 409 response and are not passed to the plugin install backend.",
            "The change type is derived from the existing installed version for the same extension id plus the uploaded bundle version.",
            "When signatures are present, bundle inspection verifies each signature against the canonical payload built from manifest metadata and hashed archive members.",
            "Signer trust policies can block or approve a bundle only when the signer key id is verified successfully against the uploaded bundle.",
        ],
        operator_notes=[
            "Install and update share the same inspection pipeline. The action button changes, but the runtime still computes the actual change type from extension id and version data.",
            "Publisher and source metadata still help with human review, but cryptographic verification now comes from optional Ed25519 signer entries plus operator-managed signer trust policies.",
            "A verified signature without an explicit signer trust policy improves operator evidence, but it does not automatically make the bundle trusted.",
            "Use the bundle checksum, verified signer ids, publisher, source URL, and release notes together when deciding whether a replacement or downgrade is acceptable.",
        ],
    )


def build_platform_router(runtime: Any) -> APIRouter:
    """Build platform inventory routes wired to *runtime*."""
    r = APIRouter(prefix="/api/platform", tags=["platform"])

    @r.get("/capabilities", response_model=CapabilityInventoryResponse)
    async def list_capabilities(
        source: CapabilitySource | None = None,
        owner_id: str | None = None,
        status: CapabilityStatus | None = None,
        kind: str | None = None,
    ) -> CapabilityInventoryResponse:
        registry = _require_capability_registry(runtime)
        normalized_kind = _normalize_query_value(kind)
        capabilities = [
            _capability_to_model(item)
            for item in registry.list_capabilities(
                source=source,
                owner_id=_normalize_query_value(owner_id),
                status=status,
            )
            if normalized_kind is None or item.kind == normalized_kind
        ]
        return CapabilityInventoryResponse(capabilities=capabilities)

    @r.get("/capabilities/{capability_id:path}", response_model=CapabilityDetailResponse)
    async def get_capability(capability_id: str) -> CapabilityDetailResponse:
        registry = _require_capability_registry(runtime)
        descriptor = registry.get(capability_id)
        if descriptor is None:
            raise HTTPException(status_code=404, detail=f"Capability not found: {capability_id}")
        return CapabilityDetailResponse(capability=_capability_to_model(descriptor))

    @r.get("/extensions", response_model=ExtensionInventoryResponse)
    async def list_extensions() -> ExtensionInventoryResponse:
        registry = _require_capability_registry(runtime)
        installed_rows = await _installed_plugin_rows(runtime)
        extensions = []
        for item in build_extension_descriptors(registry):
            row = installed_rows.get(item.extension_id, {})
            payload = _extension_to_model(item)
            payload.bundle = dict(row.get("bundle_metadata") or {})
            payload.trust = _assess_plugin_trust(
                runtime,
                provenance=payload.provenance,
                bundle=payload.bundle,
            )
            extensions.append(payload)
        return ExtensionInventoryResponse(extensions=extensions)

    @r.get(
        "/policies/install-update",
        response_model=ExtensionInstallUpdatePolicyResponse,
    )
    async def get_extension_install_update_policy() -> ExtensionInstallUpdatePolicyResponse:
        return _build_extension_install_update_policy()

    @r.get("/extensions/{extension_id:path}", response_model=ExtensionDetailResponse)
    async def get_extension(extension_id: str) -> ExtensionDetailResponse:
        registry = _require_capability_registry(runtime)
        extension = _require_extension_descriptor(runtime, extension_id)
        installed_rows = await _installed_plugin_rows(runtime)
        row = installed_rows.get(extension_id, {})
        capabilities = [
            _capability_to_model(descriptor)
            for descriptor in registry.list_capabilities(owner_id=extension_id)
        ]
        extension_payload = _extension_to_model(extension)
        extension_payload.bundle = dict(row.get("bundle_metadata") or {})
        extension_payload.trust = _assess_plugin_trust(
            runtime,
            provenance=extension_payload.provenance,
            bundle=extension_payload.bundle,
        )
        return ExtensionDetailResponse(
            extension=extension_payload,
            capabilities=capabilities,
        )

    @r.post("/extensions/{extension_id:path}/enable", response_model=ExtensionLifecycleResponse)
    async def enable_extension(extension_id: str) -> ExtensionLifecycleResponse:
        return await _run_extension_lifecycle_action(runtime, extension_id, "enable")

    @r.post("/extensions/{extension_id:path}/disable", response_model=ExtensionLifecycleResponse)
    async def disable_extension(extension_id: str) -> ExtensionLifecycleResponse:
        return await _run_extension_lifecycle_action(runtime, extension_id, "disable")

    @r.delete("/extensions/{extension_id:path}", response_model=ExtensionLifecycleResponse)
    async def uninstall_extension(extension_id: str) -> ExtensionLifecycleResponse:
        return await _run_extension_lifecycle_action(runtime, extension_id, "uninstall")

    @r.post("/extensions/install", response_model=ExtensionInstallResponse)
    async def install_extension_bundle(req: ExtensionInstallRequest) -> ExtensionInstallResponse:
        return await _install_extension_bundle(runtime, req.path, action="install")

    @r.post("/extensions/inspect-bundle", response_model=ExtensionBundleInspectionResponse)
    async def inspect_extension_bundle(req: ExtensionInstallRequest) -> ExtensionBundleInspectionResponse:
        return _inspect_extension_bundle(runtime, req.path)

    @r.post("/extensions/{extension_id:path}/update", response_model=ExtensionInstallResponse)
    async def update_extension_bundle(
        extension_id: str,
        req: ExtensionInstallRequest,
    ) -> ExtensionInstallResponse:
        return await _install_extension_bundle(
            runtime,
            req.path,
            action="update",
            expected_extension_id=extension_id,
        )

    return r
