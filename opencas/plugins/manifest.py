"""Versioned plugin manifest helpers for the phase-two extension SDK surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from opencas import __version__ as OPENCAS_VERSION

SUPPORTED_PLUGIN_MANIFEST_VERSIONS = {1}
DEFAULT_PLUGIN_MANIFEST_VERSION = 1

_SCHEMA_TYPES = {"object", "array", "string", "integer", "number", "boolean"}
_SUPPORTED_SIGNATURE_ALGORITHMS = {"ed25519"}


class PluginManifestError(ValueError):
    """Raised when a plugin manifest fails structural validation."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = list(errors)
        super().__init__("Invalid plugin manifest: " + "; ".join(self.errors))


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_version_string(value: str) -> tuple[int, ...] | None:
    parts = value.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def compare_version_strings(left: str, right: str) -> int:
    """Compare dotted numeric version strings."""

    left_parts = _parse_version_string(left)
    right_parts = _parse_version_string(right)
    if left_parts is None or right_parts is None:
        raise ValueError(f"invalid version comparison: {left!r} vs {right!r}")
    max_len = max(len(left_parts), len(right_parts))
    left_norm = left_parts + (0,) * (max_len - len(left_parts))
    right_norm = right_parts + (0,) * (max_len - len(right_parts))
    if left_norm < right_norm:
        return -1
    if left_norm > right_norm:
        return 1
    return 0


def validate_config_schema(schema: Any, *, path: str = "config_schema") -> List[str]:
    """Validate the shape of a lightweight JSON-schema-like config object."""

    if schema in ({}, None):
        return []
    if not isinstance(schema, dict):
        return [f"{path} must be an object"]

    errors: List[str] = []
    schema_type = schema.get("type")
    if schema_type is not None and schema_type not in _SCHEMA_TYPES:
        errors.append(f"{path}.type must be one of {sorted(_SCHEMA_TYPES)}")

    properties = schema.get("properties")
    if properties is not None:
        if schema_type not in (None, "object"):
            errors.append(f"{path}.properties requires type 'object'")
        if not isinstance(properties, dict):
            errors.append(f"{path}.properties must be an object")
        else:
            for key, child_schema in properties.items():
                child_path = f"{path}.properties.{key}"
                if not isinstance(child_schema, dict):
                    errors.append(f"{child_path} must be an object")
                    continue
                errors.extend(validate_config_schema(child_schema, path=child_path))

    required = schema.get("required")
    if required is not None:
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            errors.append(f"{path}.required must be a list of strings")

    enum_values = schema.get("enum")
    if enum_values is not None and not isinstance(enum_values, list):
        errors.append(f"{path}.enum must be a list")

    items = schema.get("items")
    if items is not None:
        if schema_type not in (None, "array"):
            errors.append(f"{path}.items requires type 'array'")
        if not isinstance(items, dict):
            errors.append(f"{path}.items must be an object")
        else:
            errors.extend(validate_config_schema(items, path=f"{path}.items"))

    additional_properties = schema.get("additionalProperties")
    if additional_properties is not None:
        if not isinstance(additional_properties, (bool, dict)):
            errors.append(f"{path}.additionalProperties must be a boolean or object")
        elif isinstance(additional_properties, dict):
            errors.extend(
                validate_config_schema(
                    additional_properties,
                    path=f"{path}.additionalProperties",
                )
            )

    return errors


def validate_config_payload(
    schema: Dict[str, Any],
    payload: Any,
    *,
    path: str = "config",
) -> List[str]:
    """Validate *payload* against a lightweight JSON-schema-like object."""

    if not schema:
        return []

    errors: List[str] = []
    schema_type = schema.get("type")
    enum_values = schema.get("enum")
    if enum_values is not None and payload not in enum_values:
        errors.append(f"{path} must be one of {enum_values}")

    if schema_type == "object":
        if not isinstance(payload, dict):
            return [f"{path} must be an object"]
        required = schema.get("required", [])
        for key in required:
            if key not in payload:
                errors.append(f"{path}.{key} is required")
        properties = schema.get("properties", {})
        additional_properties = schema.get("additionalProperties", True)
        for key, value in payload.items():
            if key in properties:
                errors.extend(
                    validate_config_payload(
                        properties[key],
                        value,
                        path=f"{path}.{key}",
                    )
                )
            elif additional_properties is False:
                errors.append(f"{path}.{key} is not allowed")
            elif isinstance(additional_properties, dict):
                errors.extend(
                    validate_config_payload(
                        additional_properties,
                        value,
                        path=f"{path}.{key}",
                    )
                )
        return errors

    if schema_type == "array":
        if not isinstance(payload, list):
            return [f"{path} must be an array"]
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, value in enumerate(payload):
                errors.extend(
                    validate_config_payload(
                        item_schema,
                        value,
                        path=f"{path}[{index}]",
                    )
                )
        return errors

    if schema_type == "string" and not isinstance(payload, str):
        return [f"{path} must be a string"]
    if schema_type == "integer" and not _is_integer(payload):
        return [f"{path} must be an integer"]
    if schema_type == "number" and not (isinstance(payload, (int, float)) and not isinstance(payload, bool)):
        return [f"{path} must be a number"]
    if schema_type == "boolean" and not isinstance(payload, bool):
        return [f"{path} must be a boolean"]

    return errors


def normalize_plugin_manifest(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return a manifest dict with backward-compatible defaults applied."""

    manifest = dict(raw)
    manifest.setdefault("manifest_version", DEFAULT_PLUGIN_MANIFEST_VERSION)
    manifest.setdefault("version", "0.0.1")
    manifest.setdefault("skills", [])
    manifest.setdefault("hooks", [])
    manifest.setdefault("dependencies", [])
    manifest.setdefault("capabilities", [])
    manifest.setdefault("config_schema", {})
    manifest.setdefault("default_config", {})
    manifest.setdefault("compatibility", {})
    manifest.setdefault("distribution", {})
    manifest.setdefault("release_notes", "")
    return manifest


def validate_plugin_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Validate the normalized plugin manifest structure."""

    errors: List[str] = []

    manifest_version = manifest.get("manifest_version")
    if not _is_integer(manifest_version):
        errors.append("manifest_version must be an integer")
    elif manifest_version not in SUPPORTED_PLUGIN_MANIFEST_VERSIONS:
        errors.append(
            f"manifest_version {manifest_version} is not supported; expected one of {sorted(SUPPORTED_PLUGIN_MANIFEST_VERSIONS)}"
        )

    plugin_id = manifest.get("id")
    if not isinstance(plugin_id, str) or not plugin_id.strip():
        errors.append("id is required and must be a non-empty string")

    for field_name in ("name", "description", "version", "entrypoint"):
        value = manifest.get(field_name)
        if value is not None and not isinstance(value, str):
            errors.append(f"{field_name} must be a string")

    release_notes = manifest.get("release_notes", "")
    if release_notes is not None and not isinstance(release_notes, str):
        errors.append("release_notes must be a string")

    for list_field in ("skills", "dependencies"):
        values = manifest.get(list_field)
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            errors.append(f"{list_field} must be a list of strings")

    hooks = manifest.get("hooks")
    if not isinstance(hooks, list):
        errors.append("hooks must be a list")
    else:
        for index, hook in enumerate(hooks):
            if not isinstance(hook, dict):
                errors.append(f"hooks[{index}] must be an object")
                continue
            for required_field in ("hook_name", "handler"):
                value = hook.get(required_field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"hooks[{index}].{required_field} must be a non-empty string")

    top_level_schema = manifest.get("config_schema", {})
    errors.extend(validate_config_schema(top_level_schema, path="config_schema"))
    if top_level_schema and top_level_schema.get("type") not in (None, "object"):
        errors.append("config_schema.type must be 'object' for plugin configuration")

    default_config = manifest.get("default_config", {})
    if not isinstance(default_config, dict):
        errors.append("default_config must be an object")
    else:
        errors.extend(
            validate_config_payload(
                top_level_schema if isinstance(top_level_schema, dict) else {},
                default_config,
                path="default_config",
            )
        )

    compatibility = manifest.get("compatibility", {})
    if compatibility in ({}, None):
        compatibility = {}
    if not isinstance(compatibility, dict):
        errors.append("compatibility must be an object")
    else:
        allowed_keys = {"min_opencas_version", "max_opencas_version"}
        for key in compatibility.keys():
            if key not in allowed_keys:
                errors.append(f"compatibility.{key} is not supported")
        min_version = compatibility.get("min_opencas_version")
        max_version = compatibility.get("max_opencas_version")
        if min_version is not None:
            if not isinstance(min_version, str) or _parse_version_string(min_version) is None:
                errors.append("compatibility.min_opencas_version must be a dotted numeric version string")
        if max_version is not None:
            if not isinstance(max_version, str) or _parse_version_string(max_version) is None:
                errors.append("compatibility.max_opencas_version must be a dotted numeric version string")
        if (
            isinstance(min_version, str)
            and isinstance(max_version, str)
            and _parse_version_string(min_version) is not None
            and _parse_version_string(max_version) is not None
            and compare_version_strings(min_version, max_version) > 0
        ):
            errors.append(
                "compatibility.min_opencas_version cannot be greater than compatibility.max_opencas_version"
            )

    distribution = manifest.get("distribution", {})
    if distribution in ({}, None):
        distribution = {}
    if not isinstance(distribution, dict):
        errors.append("distribution must be an object")
    else:
        allowed_keys = {
            "publisher",
            "channel",
            "source_url",
            "homepage_url",
            "documentation_url",
            "changelog_url",
            "signatures",
        }
        for key in distribution.keys():
            if key not in allowed_keys:
                errors.append(f"distribution.{key} is not supported")
        for key, value in distribution.items():
            if key == "signatures":
                if value in ({}, None):
                    continue
                if not isinstance(value, list):
                    errors.append("distribution.signatures must be a list")
                    continue
                for index, entry in enumerate(value):
                    entry_path = f"distribution.signatures[{index}]"
                    if not isinstance(entry, dict):
                        errors.append(f"{entry_path} must be an object")
                        continue
                    for entry_key in entry.keys():
                        if entry_key not in {"key_id", "algorithm", "signature", "public_key"}:
                            errors.append(f"{entry_path}.{entry_key} is not supported")
                    key_id = entry.get("key_id")
                    if not isinstance(key_id, str) or not key_id.strip():
                        errors.append(f"{entry_path}.key_id must be a non-empty string")
                    algorithm = entry.get("algorithm")
                    if algorithm is None:
                        errors.append(f"{entry_path}.algorithm is required")
                    elif not isinstance(algorithm, str):
                        errors.append(f"{entry_path}.algorithm must be a string")
                    elif algorithm not in _SUPPORTED_SIGNATURE_ALGORITHMS:
                        errors.append(
                            f"{entry_path}.algorithm must be one of {sorted(_SUPPORTED_SIGNATURE_ALGORITHMS)}"
                        )
                    signature = entry.get("signature")
                    if not isinstance(signature, str) or not signature.strip():
                        errors.append(f"{entry_path}.signature must be a non-empty string")
                    public_key = entry.get("public_key")
                    if not isinstance(public_key, str) or not public_key.strip():
                        errors.append(f"{entry_path}.public_key must be a non-empty string")
                continue
            if value is not None and not isinstance(value, str):
                errors.append(f"distribution.{key} must be a string")

    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, list):
        errors.append("capabilities must be a list")
    else:
        for index, capability in enumerate(capabilities):
            if not isinstance(capability, dict):
                errors.append(f"capabilities[{index}] must be an object")
                continue
            capability_id = capability.get("capability_id")
            if not isinstance(capability_id, str) or not capability_id.strip():
                errors.append(f"capabilities[{index}].capability_id must be a non-empty string")
            display_name = capability.get("display_name")
            if display_name is not None and not isinstance(display_name, str):
                errors.append(f"capabilities[{index}].display_name must be a string")
            dependencies = capability.get("dependencies", [])
            if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
                errors.append(f"capabilities[{index}].dependencies must be a list of strings")
            tool_names = capability.get("tool_names", [])
            if not isinstance(tool_names, list) or any(not isinstance(item, str) for item in tool_names):
                errors.append(f"capabilities[{index}].tool_names must be a list of strings")
            metadata = capability.get("metadata", {})
            if metadata is not None and not isinstance(metadata, dict):
                errors.append(f"capabilities[{index}].metadata must be an object")
            errors.extend(
                validate_config_schema(
                    capability.get("config_schema", {}),
                    path=f"capabilities[{index}].config_schema",
                )
            )

    return errors


def evaluate_plugin_compatibility(
    manifest: Dict[str, Any],
    *,
    runtime_version: str = OPENCAS_VERSION,
) -> Dict[str, Any]:
    """Evaluate OpenCAS runtime compatibility for a validated plugin manifest."""

    compatibility = manifest.get("compatibility", {})
    if not isinstance(compatibility, dict):
        compatibility = {}
    min_version = compatibility.get("min_opencas_version")
    max_version = compatibility.get("max_opencas_version")
    reasons: List[str] = []
    compatible = True
    if isinstance(min_version, str) and compare_version_strings(runtime_version, min_version) < 0:
        compatible = False
        reasons.append(f"requires OpenCAS >= {min_version}, current runtime is {runtime_version}")
    if isinstance(max_version, str) and compare_version_strings(runtime_version, max_version) > 0:
        compatible = False
        reasons.append(f"requires OpenCAS <= {max_version}, current runtime is {runtime_version}")
    return {
        "runtime_version": runtime_version,
        "compatible": compatible,
        "constraints": {
            "min_opencas_version": min_version,
            "max_opencas_version": max_version,
        },
        "reasons": reasons,
    }


def classify_plugin_update(
    previous_version: str | None,
    next_version: str | None,
) -> str:
    """Classify an install/update transition using dotted numeric versions."""

    if not previous_version:
        return "install"
    if not next_version:
        return "replace"

    try:
        comparison = compare_version_strings(previous_version, next_version)
    except ValueError:
        return "replace"

    if comparison < 0:
        return "upgrade"
    if comparison > 0:
        return "downgrade"
    return "reinstall"


def load_plugin_manifest(path: Path | str) -> Dict[str, Any]:
    """Load, normalize, and validate a plugin manifest from disk."""

    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exercised through loader compatibility
        raise PluginManifestError([f"failed to parse manifest: {exc}"]) from exc

    if not isinstance(raw, dict):
        raise PluginManifestError(["manifest root must be a JSON object"])

    manifest = normalize_plugin_manifest(raw)
    errors = validate_plugin_manifest(manifest)
    if errors:
        raise PluginManifestError(errors)
    return manifest
