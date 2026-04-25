"""Projection helpers for operator-facing platform views."""

from __future__ import annotations

from collections import defaultdict

from .models import CapabilityStatus, ExtensionDescriptor


def _first_non_null(values):
    for value in values:
        if value is not None:
            return value
    return None


def _build_config_schema_summary(items: list) -> dict:
    schema = _first_non_null(item.metadata.get("plugin_config_schema") for item in items)
    default_config = _first_non_null(item.metadata.get("plugin_default_config") for item in items)
    if not isinstance(schema, dict) or not schema:
        return {}

    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict):
        properties = {}
    if not isinstance(required, list):
        required = []
    if not isinstance(default_config, dict):
        default_config = {}

    return {
        "type": schema.get("type", "object"),
        "property_count": len(properties),
        "properties": sorted(properties.keys()),
        "required": [item for item in required if isinstance(item, str)],
        "has_default_config": bool(default_config),
        "default_config_keys": sorted(default_config.keys()),
    }


def _build_compatibility_summary(items: list) -> dict:
    compatibility = _first_non_null(item.metadata.get("compatibility") for item in items)
    if not isinstance(compatibility, dict):
        return {}
    constraints = compatibility.get("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}
    reasons = compatibility.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = []
    return {
        "runtime_version": compatibility.get("runtime_version"),
        "compatible": bool(compatibility.get("compatible", True)),
        "constraints": {
            "min_opencas_version": constraints.get("min_opencas_version"),
            "max_opencas_version": constraints.get("max_opencas_version"),
        },
        "reasons": [item for item in reasons if isinstance(item, str)],
    }


def _build_provenance_summary(items: list) -> dict:
    distribution = _first_non_null(item.metadata.get("distribution") for item in items)
    if not isinstance(distribution, dict):
        return {}
    return {
        key: value
        for key, value in distribution.items()
        if isinstance(key, str) and isinstance(value, str) and value.strip()
    }


def _build_release_notes(items: list) -> str:
    release_notes = _first_non_null(item.metadata.get("release_notes") for item in items)
    if not isinstance(release_notes, str):
        return ""
    return release_notes.strip()


def build_extension_descriptors(registry) -> list[ExtensionDescriptor]:
    """Group capability descriptors into extension descriptors for operators."""

    grouped: dict[str, list] = defaultdict(list)
    for capability in registry.list_capabilities():
        grouped[capability.owner_id].append(capability)

    extensions: list[ExtensionDescriptor] = []
    for owner_id, items in sorted(grouped.items()):
        first = items[0]
        status = CapabilityStatus.ENABLED
        if any(item.status is CapabilityStatus.FAILED_VALIDATION for item in items):
            status = CapabilityStatus.FAILED_VALIDATION
        elif any(item.status is CapabilityStatus.MISSING_DEPENDENCY for item in items):
            status = CapabilityStatus.MISSING_DEPENDENCY
        elif any(item.status is CapabilityStatus.DISABLED for item in items):
            status = CapabilityStatus.DISABLED
        elif any(item.status is CapabilityStatus.UNAVAILABLE for item in items):
            status = CapabilityStatus.UNAVAILABLE

        if first.source.value == "plugin":
            extension_kind = "plugin"
        elif first.source.value == "mcp":
            extension_kind = "mcp_server"
        else:
            extension_kind = "core_bundle"

        owner_name = _first_non_null(item.metadata.get("owner_name") for item in items)
        manifest_version = _first_non_null(item.metadata.get("manifest_version") for item in items)
        version = _first_non_null(item.metadata.get("version") for item in items)
        manifest_path = _first_non_null(item.manifest_path for item in items)

        extensions.append(
            ExtensionDescriptor(
                extension_id=owner_id,
                extension_kind=extension_kind,
                display_name=owner_name if owner_name is not None else owner_id,
                status=status,
                capability_ids=sorted(item.capability_id for item in items),
                manifest_version=manifest_version,
                version=version,
                manifest_path=manifest_path,
                dependencies=sorted(
                    {
                        dependency
                        for item in items
                        for dependency in item.declared_dependencies
                    }
                ),
                compatibility=_build_compatibility_summary(items),
                provenance=_build_provenance_summary(items),
                release_notes=_build_release_notes(items),
                config_schema_summary=_build_config_schema_summary(items),
                errors=[error for item in items for error in item.validation_errors],
            )
        )

    return extensions
