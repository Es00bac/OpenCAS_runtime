"""Skill and plugin loader for OpenCAS."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from opencas.platform import CapabilityDescriptor, CapabilityRegistry, CapabilitySource, CapabilityStatus
from opencas.tools import ToolRegistry
from opencas.infra.hook_registry import TypedHookRegistry

from .manifest import (
    PluginManifestError,
    evaluate_plugin_compatibility,
    load_plugin_manifest,
)
from .models import PluginCapabilityEntry, PluginEntry, SkillEntry
from .registry import PluginRegistry, SkillRegistry


class _PluginScopedToolRegistry:
    """ToolRegistry facade that stamps plugin ownership onto new tools."""

    def __init__(self, tools: ToolRegistry, plugin_id: str) -> None:
        self._tools = tools
        self._plugin_id = plugin_id

    def register(
        self,
        name: str,
        description: str,
        adapter,
        risk_tier,
        parameters: Optional[Dict[str, Any]] = None,
        plugin_id: Optional[str] = None,
    ) -> None:
        self._tools.register(
            name,
            description,
            adapter,
            risk_tier,
            parameters,
            plugin_id=plugin_id if plugin_id is not None else self._plugin_id,
        )

    def __getattr__(self, item: str) -> Any:
        return getattr(self._tools, item)


def _load_module_from_path(path: Path) -> Any:
    """Dynamically import a Python module from *path*."""
    module_name = f"_opencas_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _skill_entry_from_module(
    module: Any,
    plugin_id: Optional[str],
    path: Path,
) -> Optional[SkillEntry]:
    """Create a SkillEntry from a loaded module, if possible."""
    entry: Optional[SkillEntry] = None
    if hasattr(module, "SKILL_ENTRY"):
        raw = module.SKILL_ENTRY
        if isinstance(raw, SkillEntry):
            entry = raw
        elif isinstance(raw, dict):
            entry = SkillEntry(**raw)

    register_fn = None
    if hasattr(module, "register_skills"):
        register_fn = getattr(module, "register_skills")

    if entry is None and register_fn is not None:
        entry = SkillEntry(
            skill_id=path.stem,
            name=path.stem,
            description="",
            entrypoint=str(path),
        )

    if entry is not None:
        entry.entrypoint = entry.entrypoint or str(path)
        if register_fn is not None:
            entry.register_fn = register_fn
        entry.plugin_id = plugin_id

    return entry


def _plugin_capability_entry_from_manifest(raw: Any) -> Optional[PluginCapabilityEntry]:
    """Normalize a manifest capability entry into a dataclass."""

    if isinstance(raw, PluginCapabilityEntry):
        return raw
    if not isinstance(raw, dict):
        return None

    capability_id = raw.get("capability_id")
    if not capability_id:
        return None

    display_name = raw.get("display_name") or raw.get("name") or capability_id
    return PluginCapabilityEntry(
        capability_id=str(capability_id),
        display_name=str(display_name),
        kind=str(raw.get("kind", "tool")),
        description=str(raw.get("description", "")),
        tool_names=list(raw.get("tool_names", [])),
        dependencies=list(raw.get("dependencies", [])),
        config_schema=dict(raw.get("config_schema", {})),
        metadata=dict(raw.get("metadata", {})),
    )


def _register_plugin_capabilities(
    *,
    plugin_id: str,
    plugin_name: str,
    plugin_description: str,
    manifest_version: int | None,
    version: str,
    manifest_path: Path,
    source_path: Path,
    entrypoint: Optional[str],
    plugin_dependencies: List[str],
    plugin_config_schema: Dict[str, Any],
    plugin_default_config: Dict[str, Any],
    plugin_compatibility: Dict[str, Any],
    plugin_distribution: Dict[str, Any],
    plugin_release_notes: str,
    capability_entries: List[PluginCapabilityEntry],
    capability_registry: Optional[CapabilityRegistry],
    initial_status: CapabilityStatus = CapabilityStatus.ENABLED,
    validation_errors: Optional[List[str]] = None,
) -> None:
    """Project manifest capabilities into the canonical registry."""

    if capability_registry is None:
        return

    entries = capability_entries
    use_plugin_dependencies = False
    if not entries:
        use_plugin_dependencies = True
        entries = [
            PluginCapabilityEntry(
                capability_id=f"plugin:{plugin_id}",
                display_name=plugin_name or plugin_id,
                kind="plugin",
                description=plugin_description,
                metadata={"owner_name": plugin_name or plugin_id},
            )
        ]

    for entry in entries:
        metadata = dict(entry.metadata)
        metadata["manifest_version"] = manifest_version
        metadata["version"] = version
        metadata["owner_name"] = plugin_name or plugin_id
        metadata["plugin_config_schema"] = dict(plugin_config_schema)
        metadata["plugin_default_config"] = dict(plugin_default_config)
        metadata["compatibility"] = dict(plugin_compatibility)
        metadata["distribution"] = dict(plugin_distribution)
        metadata["release_notes"] = plugin_release_notes
        capability_registry.register(
            CapabilityDescriptor(
                capability_id=entry.capability_id,
                display_name=entry.display_name,
                kind=entry.kind,
                source=CapabilitySource.PLUGIN,
                owner_id=plugin_id,
                status=initial_status,
                description=entry.description,
                tool_names=list(entry.tool_names),
                declared_dependencies=list(entry.dependencies)
                if entry.dependencies or not use_plugin_dependencies
                else list(plugin_dependencies),
                config_schema=dict(entry.config_schema),
                entrypoint=entrypoint,
                manifest_path=str(manifest_path),
                source_path=str(source_path),
                validation_errors=list(validation_errors or []),
                metadata=metadata,
            )
        )


def load_skill_from_path(
    path: Path | str,
    registry: SkillRegistry,
    plugin_id: Optional[str] = None,
) -> Optional[SkillEntry]:
    """Load a single skill module from *path* and register it."""
    path = Path(path)
    if not path.exists() or not path.is_file() or path.name.startswith("_"):
        return None

    module = _load_module_from_path(path)
    if module is None:
        return None

    entry = _skill_entry_from_module(module, plugin_id, path)
    if entry is not None:
        registry.register(entry)

    return entry


def load_builtin_skills(
    directory: Path | str,
    registry: SkillRegistry,
    plugin_id_prefix: Optional[str] = None,
) -> List[SkillEntry]:
    """Discover and load all .py skill files in *directory*."""
    directory = Path(directory)
    loaded: List[SkillEntry] = []
    if not directory.exists() or not directory.is_dir():
        return loaded
    for file_path in sorted(directory.glob("*.py")):
        pid = plugin_id_prefix or file_path.stem
        entry = load_skill_from_path(file_path, registry, plugin_id=pid)
        if entry is not None:
            loaded.append(entry)
    return loaded


def load_plugin_from_manifest(
    manifest_path: Path | str,
    plugin_registry: PluginRegistry,
    skill_registry: SkillRegistry,
    tools: ToolRegistry,
    hook_registry: Optional[TypedHookRegistry] = None,
    source: str = "installed",
    *,
    capability_registry: Optional[CapabilityRegistry] = None,
) -> Optional[PluginEntry]:
    """Load a plugin from a *plugin.json* manifest."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists() or not manifest_path.is_file():
        return None

    try:
        manifest = load_plugin_manifest(manifest_path)
    except PluginManifestError:
        return None

    plugin_id = manifest.get("id")
    name = manifest.get("name", plugin_id or "")
    description = manifest.get("description", "")
    manifest_version = manifest.get("manifest_version")
    version = manifest.get("version", "0.0.1")
    entrypoint_name = manifest.get("entrypoint")
    skills_cfg = manifest.get("skills", [])
    hooks_cfg = manifest.get("hooks", [])
    dependencies = manifest.get("dependencies", [])
    config_schema = dict(manifest.get("config_schema", {}))
    default_config = dict(manifest.get("default_config", {}))
    compatibility = evaluate_plugin_compatibility(manifest)
    distribution = dict(manifest.get("distribution", {}))
    release_notes = str(manifest.get("release_notes", ""))
    validation_errors = list(compatibility.get("reasons", [])) if not compatibility.get("compatible", True) else []
    capability_entries = [
        entry
        for entry in (
            _plugin_capability_entry_from_manifest(raw)
            for raw in manifest.get("capabilities", [])
        )
        if entry is not None
    ]

    if not plugin_id:
        return None

    plugin_dir = manifest_path.parent
    skills: List[SkillEntry] = []
    on_load_fn = None
    on_unload_fn = None

    if compatibility.get("compatible", True):
        module = None
        if entrypoint_name:
            entrypoint_path = plugin_dir / entrypoint_name
            if entrypoint_path.exists() and entrypoint_path.is_file():
                module = _load_module_from_path(entrypoint_path)

        # Register skills
        if module is not None and hasattr(module, "register_skills"):
            register_fn = getattr(module, "register_skills")
            register_fn(skill_registry, _PluginScopedToolRegistry(tools, plugin_id))
            # Also create entries for any exported SKILL_ENTRY or skill manifests
            if hasattr(module, "SKILL_ENTRY"):
                raw = module.SKILL_ENTRY
                if isinstance(raw, SkillEntry):
                    raw.plugin_id = plugin_id
                    if raw.skill_id not in {s.skill_id for s in skill_registry.list_skills()}:
                        skill_registry.register(raw)
                    skills.append(raw)
                elif isinstance(raw, dict):
                    raw.setdefault("plugin_id", plugin_id)
                    se = SkillEntry(**raw)
                    skill_registry.register(se)
                    skills.append(se)
        elif skills_cfg:
            for skill_file in skills_cfg:
                skill_path = plugin_dir / skill_file
                entry = load_skill_from_path(skill_path, skill_registry, plugin_id=plugin_id)
                if entry is not None:
                    skills.append(entry)

        # Register hooks
        if module is not None and hasattr(module, "register_hooks"):
            if hook_registry is not None:
                register_hooks_fn = getattr(module, "register_hooks")
                register_hooks_fn(hook_registry)
        elif hooks_cfg and hook_registry is not None:
            for hook_def in hooks_cfg:
                if module is not None and hasattr(module, hook_def.get("handler")):
                    handler = getattr(module, hook_def["handler"])
                    hook_registry.register(
                        hook_def["hook_name"],
                        handler,
                        priority=hook_def.get("priority", 0),
                        source=plugin_id,
                    )

        if module is not None:
            if hasattr(module, "on_load"):
                on_load_fn = getattr(module, "on_load")
            if hasattr(module, "on_unload"):
                on_unload_fn = getattr(module, "on_unload")

    _register_plugin_capabilities(
        plugin_id=plugin_id,
        plugin_name=name,
        plugin_description=description,
        manifest_version=manifest_version,
        version=version,
        manifest_path=manifest_path,
        source_path=plugin_dir,
        entrypoint=entrypoint_name if entrypoint_name else None,
        plugin_dependencies=list(dependencies),
        plugin_config_schema=config_schema,
        plugin_default_config=default_config,
        plugin_compatibility=compatibility,
        plugin_distribution=distribution,
        plugin_release_notes=release_notes,
        capability_entries=capability_entries,
        capability_registry=capability_registry,
        initial_status=(
            CapabilityStatus.ENABLED
            if compatibility.get("compatible", True)
            else CapabilityStatus.FAILED_VALIDATION
        ),
        validation_errors=validation_errors,
    )

    plugin = PluginEntry(
        plugin_id=plugin_id,
        name=name,
        description=description,
        manifest_version=manifest_version,
        version=version,
        source=source,
        path=str(manifest_path),
        manifest=manifest,
        config_schema=config_schema,
        default_config=default_config,
        compatibility=compatibility,
        distribution=distribution,
        release_notes=release_notes,
        validation_errors=validation_errors,
        enabled=compatibility.get("compatible", True),
        capabilities=capability_entries,
        skills=skills,
        on_load_fn=on_load_fn,
        on_unload_fn=on_unload_fn,
    )
    plugin_registry.register(plugin)
    return plugin


def load_builtin_plugins(
    directory: Path | str,
    plugin_registry: PluginRegistry,
    skill_registry: SkillRegistry,
    tools: ToolRegistry,
    hook_registry: Optional[TypedHookRegistry] = None,
    capability_registry: Optional[CapabilityRegistry] = None,
) -> List[PluginEntry]:
    """Discover and load all plugins in *directory*.

    Directories containing *plugin.json* are loaded as manifest plugins.
    Standalone *.py files are loaded as single-skill builtin plugins.
    """
    directory = Path(directory)
    loaded: List[PluginEntry] = []
    if not directory.exists() or not directory.is_dir():
        return loaded

    # Manifest-based plugins first
    for manifest_path in sorted(directory.glob("*/plugin.json")):
        plugin = load_plugin_from_manifest(
            manifest_path,
            plugin_registry,
            skill_registry,
            tools,
            hook_registry,
            source="builtin",
            capability_registry=capability_registry,
        )
        if plugin is not None:
            loaded.append(plugin)

    # Standalone skill files as legacy-style builtin plugins
    for file_path in sorted(directory.glob("*.py")):
        if file_path.name.startswith("_"):
            continue
        plugin_id = file_path.stem
        entry = load_skill_from_path(file_path, skill_registry, plugin_id=plugin_id)
        if entry is not None:
            plugin = PluginEntry(
                plugin_id=plugin_id,
                name=entry.name,
                description=entry.description,
                source="builtin",
                path=str(file_path),
                enabled=True,
                skills=[entry],
            )
            plugin_registry.register(plugin)
            loaded.append(plugin)

    return loaded
