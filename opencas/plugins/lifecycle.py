"""Plugin lifecycle manager for OpenCAS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from opencas.infra.hook_registry import TypedHookRegistry
from opencas.telemetry import EventKind, Tracer
from opencas.tools import ToolRegistry

from .loader import load_builtin_plugins, load_plugin_from_manifest
from .models import PluginEntry
from .registry import PluginRegistry, SkillRegistry
from .store import PluginStore


class PluginLifecycleManager:
    """Manages install, enable, disable, and uninstall for plugins."""

    def __init__(
        self,
        store: PluginStore,
        plugin_registry: PluginRegistry,
        skill_registry: SkillRegistry,
        tools: ToolRegistry,
        hook_registry: Optional[TypedHookRegistry] = None,
        builtin_dir: Optional[Path] = None,
        tracer: Optional[Tracer] = None,
    ) -> None:
        self.store = store
        self.plugin_registry = plugin_registry
        self.skill_registry = skill_registry
        self.tools = tools
        self.hook_registry = hook_registry
        self.builtin_dir = builtin_dir
        self.tracer = tracer
        self._disabled_tools: set[str] = set()

    def _trace(self, event: str, payload: Dict[str, Any]) -> None:
        if self.tracer:
            self.tracer.log(EventKind.BOOTSTRAP_STAGE, f"PluginLifecycle: {event}", payload)

    async def _check_dependencies(self, plugin: PluginEntry, require_enabled: bool = False) -> Optional[str]:
        """Return missing dependency plugin_id, or None if all satisfied."""
        dependencies = plugin.manifest.get("dependencies", [])
        for dep_id in dependencies:
            if not await self.store.is_installed(dep_id):
                return dep_id
            if require_enabled and not await self.store.is_enabled(dep_id):
                return dep_id
        return None

    async def install(self, path: Path | str) -> Optional[PluginEntry]:
        """Install a plugin from a directory or manifest file."""
        path = Path(path)
        manifest_path = path if path.is_file() and path.name == "plugin.json" else path / "plugin.json"

        plugin = load_plugin_from_manifest(
            manifest_path,
            self.plugin_registry,
            self.skill_registry,
            self.tools,
            self.hook_registry,
            source="installed",
        )
        if plugin is None:
            self._trace("install_failed", {"path": str(path)})
            return None

        missing = await self._check_dependencies(plugin, require_enabled=False)
        if missing is not None:
            self._trace("install_failed", {"plugin_id": plugin.plugin_id, "reason": f"missing dependency: {missing}"})
            return None

        await self.store.install(
            plugin_id=plugin.plugin_id,
            name=plugin.name,
            description=plugin.description,
            source="installed",
            path=str(manifest_path),
            manifest=plugin.manifest,
        )
        await self.enable(plugin.plugin_id)
        self._trace("installed", {"plugin_id": plugin.plugin_id, "name": plugin.name})
        return plugin

    async def uninstall(self, plugin_id: str) -> None:
        """Uninstall a plugin, disabling it first."""
        plugin = self.plugin_registry.get(plugin_id)
        if plugin is not None:
            await self.disable(plugin_id)
            self.plugin_registry.unregister(plugin_id)
        await self.store.uninstall(plugin_id)
        self._trace("uninstalled", {"plugin_id": plugin_id})

    async def enable(self, plugin_id: str) -> None:
        """Enable a plugin: register its skills, hooks, and call on_load."""
        plugin = self.plugin_registry.get(plugin_id)
        if plugin is None:
            # Attempt to restore from store if path is known
            installed = await self.store.list_installed()
            for row in installed:
                if row["plugin_id"] == plugin_id:
                    manifest_path = Path(row["path"])
                    plugin = load_plugin_from_manifest(
                        manifest_path,
                        self.plugin_registry,
                        self.skill_registry,
                        self.tools,
                        self.hook_registry,
                        source=row["source"],
                    )
                    break

        if plugin is None:
            self._trace("enable_failed", {"plugin_id": plugin_id, "reason": "plugin not found"})
            return

        missing = await self._check_dependencies(plugin, require_enabled=True)
        if missing is not None:
            self._trace("enable_failed", {"plugin_id": plugin_id, "reason": f"missing dependency: {missing}"})
            return

        plugin.enabled = True
        await self.store.set_enabled(plugin_id, True)

        # Re-enable any previously disabled tools for this plugin
        if hasattr(self.tools, "_plugin_tools"):
            for tool_name, owner_id in list(self.tools._plugin_tools.items()):
                if owner_id == plugin_id:
                    self._disabled_tools.discard(tool_name)

        if plugin.on_load_fn is not None:
            try:
                plugin.on_load_fn()
            except Exception as exc:
                self._trace("on_load_failed", {"plugin_id": plugin_id, "error": str(exc)})

        self._trace("enabled", {"plugin_id": plugin_id})

    async def disable(self, plugin_id: str) -> None:
        """Disable a plugin: call on_unload, unregister hooks, block its tools."""
        plugin = self.plugin_registry.get(plugin_id)
        if plugin is None:
            self._trace("disable_failed", {"plugin_id": plugin_id, "reason": "plugin not found"})
            return

        plugin.enabled = False
        await self.store.set_enabled(plugin_id, False)

        if plugin.on_unload_fn is not None:
            try:
                plugin.on_unload_fn()
            except Exception as exc:
                self._trace("on_unload_failed", {"plugin_id": plugin_id, "error": str(exc)})

        if self.hook_registry is not None:
            self.hook_registry.clear_source(plugin_id)

        # Block this plugin's tools
        for skill in plugin.skills:
            if skill.capabilities:
                for tool_name in skill.capabilities:
                    self._disabled_tools.add(tool_name)

        # Use ToolRegistry plugin ownership map if available
        if hasattr(self.tools, "_plugin_tools"):
            for tool_name, owner_id in list(self.tools._plugin_tools.items()):
                if owner_id == plugin_id:
                    self._disabled_tools.add(tool_name)

        self._trace("disabled", {"plugin_id": plugin_id})

    async def load_all(self) -> List[PluginEntry]:
        """Load builtins and all enabled installed plugins at boot."""
        loaded: List[PluginEntry] = []

        # 1. Builtins
        if self.builtin_dir is not None:
            builtins = load_builtin_plugins(
                self.builtin_dir,
                self.plugin_registry,
                self.skill_registry,
                self.tools,
                self.hook_registry,
            )
            loaded.extend(builtins)
            for plugin in builtins:
                # Builtins are implicitly installed/enabled
                if not await self.store.is_installed(plugin.plugin_id):
                    await self.store.install(
                        plugin_id=plugin.plugin_id,
                        name=plugin.name,
                        description=plugin.description,
                        source="builtin",
                        path=plugin.path or "",
                        manifest=plugin.manifest,
                    )

        # 2. Enabled installed plugins
        enabled_rows = await self.store.list_enabled()
        for row in enabled_rows:
            plugin_id = row["plugin_id"]
            if self.plugin_registry.get(plugin_id) is not None:
                continue
            manifest_path = Path(row["path"])
            if not manifest_path.exists():
                # Fallback to directory if path points to missing manifest
                manifest_path = Path(row["path"]).parent / "plugin.json"
            plugin = load_plugin_from_manifest(
                manifest_path,
                self.plugin_registry,
                self.skill_registry,
                self.tools,
                self.hook_registry,
                source="installed",
            )
            if plugin is not None:
                plugin.enabled = True
                loaded.append(plugin)

        for plugin in loaded:
            if plugin.on_load_fn is not None:
                try:
                    plugin.on_load_fn()
                except Exception as exc:
                    self._trace("on_load_failed", {"plugin_id": plugin.plugin_id, "error": str(exc)})

        self._trace("load_all", {"builtin_count": len(builtins) if self.builtin_dir else 0, "installed_enabled_count": len(enabled_rows)})
        return loaded

    def is_tool_disabled(self, tool_name: str) -> bool:
        """Return True if *tool_name* belongs to a disabled plugin."""
        return tool_name in self._disabled_tools
