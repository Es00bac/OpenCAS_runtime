"""Plugin and skill registry for OpenCAS."""

from .lifecycle import PluginLifecycleManager
from .loader import (
    load_builtin_plugins,
    load_builtin_skills,
    load_plugin_from_manifest,
    load_skill_from_path,
)
from .manifest import (
    PluginManifestError,
    classify_plugin_update,
    compare_version_strings,
    evaluate_plugin_compatibility,
    load_plugin_manifest,
    normalize_plugin_manifest,
    validate_config_payload,
    validate_config_schema,
    validate_plugin_manifest,
)
from .package import (
    PLUGIN_BUNDLE_SUFFIX,
    PluginPackageError,
    build_plugin_bundle,
    build_plugin_signature_payload,
    extract_plugin_bundle,
    inspect_plugin_bundle,
    inspect_plugin_bundle_details,
    sign_plugin_directory,
)
from .templates import PluginTemplateSpec, scaffold_plugin_template
from .models import PluginEntry, SkillEntry
from .registry import PluginRegistry, SkillRegistry
from .store import PluginStore

__all__ = [
    "PLUGIN_BUNDLE_SUFFIX",
    "PluginManifestError",
    "PluginPackageError",
    "PluginEntry",
    "PluginLifecycleManager",
    "PluginRegistry",
    "PluginStore",
    "PluginTemplateSpec",
    "SkillEntry",
    "SkillRegistry",
    "build_plugin_bundle",
    "build_plugin_signature_payload",
    "classify_plugin_update",
    "compare_version_strings",
    "evaluate_plugin_compatibility",
    "extract_plugin_bundle",
    "inspect_plugin_bundle",
    "inspect_plugin_bundle_details",
    "load_builtin_plugins",
    "load_plugin_manifest",
    "load_builtin_skills",
    "load_plugin_from_manifest",
    "load_skill_from_path",
    "normalize_plugin_manifest",
    "sign_plugin_directory",
    "scaffold_plugin_template",
    "validate_config_payload",
    "validate_config_schema",
    "validate_plugin_manifest",
]
