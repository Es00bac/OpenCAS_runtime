"""Skill and plugin loader for OpenCAS."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from opencas.tools import ToolRegistry
from opencas.infra.hook_registry import TypedHookRegistry

from .models import PluginEntry, SkillEntry
from .registry import PluginRegistry, SkillRegistry


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
) -> Optional[PluginEntry]:
    """Load a plugin from a *plugin.json* manifest."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists() or not manifest_path.is_file():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    plugin_id = manifest.get("id")
    name = manifest.get("name", plugin_id or "")
    description = manifest.get("description", "")
    version = manifest.get("version", "0.0.1")
    entrypoint_name = manifest.get("entrypoint")
    skills_cfg = manifest.get("skills", [])
    hooks_cfg = manifest.get("hooks", [])
    dependencies = manifest.get("dependencies", [])

    if not plugin_id:
        return None

    plugin_dir = manifest_path.parent
    module = None
    if entrypoint_name:
        entrypoint_path = plugin_dir / entrypoint_name
        if entrypoint_path.exists() and entrypoint_path.is_file():
            module = _load_module_from_path(entrypoint_path)

    # Register skills
    skills: List[SkillEntry] = []
    if module is not None and hasattr(module, "register_skills"):
        register_fn = getattr(module, "register_skills")
        register_fn(skill_registry, tools)
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

    on_load_fn = None
    on_unload_fn = None
    if module is not None:
        if hasattr(module, "on_load"):
            on_load_fn = getattr(module, "on_load")
        if hasattr(module, "on_unload"):
            on_unload_fn = getattr(module, "on_unload")

    plugin = PluginEntry(
        plugin_id=plugin_id,
        name=name,
        description=description,
        version=version,
        source=source,
        path=str(manifest_path),
        manifest=manifest,
        enabled=True,
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
