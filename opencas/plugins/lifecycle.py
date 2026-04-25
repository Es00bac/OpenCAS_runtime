"""Plugin lifecycle manager for OpenCAS."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from opencas.platform import CapabilityRegistry, CapabilityStatus
from opencas.infra.hook_registry import TypedHookRegistry
from opencas.telemetry import EventKind, Tracer
from opencas.tools import ToolRegistry

from .loader import load_builtin_plugins, load_plugin_from_manifest
from .models import PluginEntry
from .package import PluginPackageError, extract_plugin_bundle, inspect_plugin_bundle_details
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
        install_root: Optional[Path] = None,
        tracer: Optional[Tracer] = None,
        capability_registry: Optional[CapabilityRegistry] = None,
    ) -> None:
        self.store = store
        self.plugin_registry = plugin_registry
        self.skill_registry = skill_registry
        self.tools = tools
        self.hook_registry = hook_registry
        self.builtin_dir = builtin_dir
        self.install_root = install_root
        self.tracer = tracer
        self.capability_registry = capability_registry
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

    def _set_plugin_capability_status(
        self,
        plugin_id: str,
        status: CapabilityStatus,
        errors: list[str] | None = None,
    ) -> None:
        """Project plugin lifecycle state onto all owned capability descriptors."""

        if self.capability_registry is None:
            return

        for descriptor in self.capability_registry.list_capabilities(owner_id=plugin_id):
            self.capability_registry.update_status(
                descriptor.capability_id,
                status,
                errors=errors,
            )

    def _mark_dependency_failure(
        self,
        plugin: PluginEntry,
        missing_dependency: str,
    ) -> None:
        """Project a missing dependency onto plugin-local and canonical state."""

        reason = f"missing dependency: {missing_dependency}"
        plugin.enabled = False
        self._set_plugin_capability_status(
            plugin.plugin_id,
            CapabilityStatus.MISSING_DEPENDENCY,
            errors=[reason],
        )

    def _mark_validation_failure(
        self,
        plugin: PluginEntry,
        errors: list[str] | None = None,
    ) -> None:
        """Project manifest/runtime validation failures onto canonical state."""

        reasons = list(errors if errors is not None else plugin.validation_errors)
        plugin.enabled = False
        plugin.validation_errors = reasons
        self._set_plugin_capability_status(
            plugin.plugin_id,
            CapabilityStatus.FAILED_VALIDATION,
            errors=reasons,
        )

    def _rollback_plugin_surface(self, plugin_id: str, *, remove_capabilities: bool = False) -> None:
        """Remove live runtime state for *plugin_id* without touching persistence."""

        removed_tools: list[str] = []
        if hasattr(self.tools, "unregister_owner"):
            removed_tools = self.tools.unregister_owner(plugin_id)
        else:
            for tool_name, owner_id in list(getattr(self.tools, "_plugin_tools", {}).items()):
                if owner_id == plugin_id:
                    removed_tools.append(tool_name)
                    self.tools.unregister(tool_name)
        for tool_name in removed_tools:
            self._disabled_tools.discard(tool_name)

        for skill in list(self.skill_registry.list_skills()):
            if skill.plugin_id == plugin_id:
                self.skill_registry.unregister(skill.skill_id)

        if self.hook_registry is not None:
            self.hook_registry.clear_source(plugin_id)

        if remove_capabilities and self.capability_registry is not None:
            self.capability_registry.unregister_owner(plugin_id)

        self.plugin_registry.unregister(plugin_id)

    async def install(self, path: Path | str) -> Optional[PluginEntry]:
        """Install a plugin from a directory or manifest file."""
        path = Path(path)
        manifest = None
        bundle_metadata: Dict[str, Any] | None = None
        if path.is_file() and path.suffix == ".zip":
            if self.install_root is None:
                self._trace(
                    "install_failed",
                    {"path": str(path), "reason": "bundle install requires install_root"},
                )
                return None
            try:
                details = inspect_plugin_bundle_details(path)
                bundle_metadata = dict(details.get("bundle", {}))
                manifest_path, manifest = extract_plugin_bundle(path, self.install_root)
            except PluginPackageError as exc:
                self._trace("install_failed", {"path": str(path), "reason": str(exc)})
                return None
        else:
            manifest_path = path if path.is_file() and path.name == "plugin.json" else path / "plugin.json"
            try:
                from .manifest import load_plugin_manifest

                manifest = load_plugin_manifest(manifest_path)
            except Exception:
                manifest = None

        plugin_id = None
        if isinstance(manifest, dict):
            plugin_id = manifest.get("id")
        if isinstance(plugin_id, str) and plugin_id:
            if self.plugin_registry.get(plugin_id) is not None or await self.store.is_installed(plugin_id):
                self._rollback_plugin_surface(plugin_id, remove_capabilities=True)

        plugin = load_plugin_from_manifest(
            manifest_path,
            self.plugin_registry,
            self.skill_registry,
            self.tools,
            self.hook_registry,
            source="installed",
            capability_registry=self.capability_registry,
        )
        if plugin is None:
            self._trace("install_failed", {"path": str(path)})
            return None

        if plugin.validation_errors:
            self._mark_validation_failure(plugin)
            self._rollback_plugin_surface(plugin.plugin_id, remove_capabilities=True)
            self._trace(
                "install_failed",
                {
                    "plugin_id": plugin.plugin_id,
                    "reason": "; ".join(plugin.validation_errors),
                },
            )
            return None

        missing = await self._check_dependencies(plugin, require_enabled=False)
        if missing is not None:
            self._mark_dependency_failure(plugin, missing)
            self._rollback_plugin_surface(plugin.plugin_id)
            self._trace("install_failed", {"plugin_id": plugin.plugin_id, "reason": f"missing dependency: {missing}"})
            return None

        await self.store.install(
            plugin_id=plugin.plugin_id,
            name=plugin.name,
            description=plugin.description,
            source="installed",
            path=str(manifest_path),
            manifest=plugin.manifest,
            bundle_metadata=bundle_metadata,
        )
        await self.enable(plugin.plugin_id)
        self._trace("installed", {"plugin_id": plugin.plugin_id, "name": plugin.name})
        return plugin

    async def uninstall(self, plugin_id: str) -> None:
        """Uninstall a plugin, disabling it first."""
        plugin = self.plugin_registry.get(plugin_id)
        if plugin is not None:
            await self.disable(plugin_id)
        installed_rows = await self.store.list_installed()
        stored = next((row for row in installed_rows if row["plugin_id"] == plugin_id), None)
        self._rollback_plugin_surface(plugin_id, remove_capabilities=True)
        await self.store.uninstall(plugin_id)
        removed_path: str | None = None
        if stored is not None and stored.get("source") == "installed" and self.install_root is not None:
            manifest_path = Path(stored.get("path") or "")
            if manifest_path.name == "plugin.json":
                candidate_root = manifest_path.parent
            else:
                candidate_root = manifest_path
            try:
                resolved_install_root = self.install_root.resolve()
                resolved_candidate = candidate_root.resolve()
                if (
                    resolved_candidate.exists()
                    and resolved_candidate != resolved_install_root
                    and resolved_install_root in resolved_candidate.parents
                ):
                    shutil.rmtree(resolved_candidate)
                    removed_path = str(resolved_candidate)
            except OSError as exc:
                self._trace(
                    "uninstall_cleanup_failed",
                    {"plugin_id": plugin_id, "path": str(candidate_root), "error": str(exc)},
                )
        payload: Dict[str, Any] = {"plugin_id": plugin_id}
        if removed_path is not None:
            payload["removed_path"] = removed_path
        self._trace("uninstalled", payload)

    async def enable(self, plugin_id: str) -> None:
        """Enable a plugin: register its skills, hooks, and call on_load."""
        plugin = self.plugin_registry.get(plugin_id)
        loaded_from_store = False
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
                        capability_registry=self.capability_registry,
                    )
                    loaded_from_store = plugin is not None
                    break

        if plugin is None:
            self._trace("enable_failed", {"plugin_id": plugin_id, "reason": "plugin not found"})
            return

        if plugin.validation_errors:
            self._mark_validation_failure(plugin)
            await self.store.set_enabled(plugin_id, False)
            self._trace(
                "enable_failed",
                {"plugin_id": plugin_id, "reason": "; ".join(plugin.validation_errors)},
            )
            return

        missing = await self._check_dependencies(plugin, require_enabled=True)
        if missing is not None:
            self._mark_dependency_failure(plugin, missing)
            if loaded_from_store:
                self._rollback_plugin_surface(plugin_id)
            await self.store.set_enabled(plugin_id, False)
            self._trace("enable_failed", {"plugin_id": plugin_id, "reason": f"missing dependency: {missing}"})
            return

        plugin.enabled = True
        await self.store.set_enabled(plugin_id, True)
        self._set_plugin_capability_status(plugin_id, CapabilityStatus.ENABLED)

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
        self._set_plugin_capability_status(plugin_id, CapabilityStatus.DISABLED)

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
                capability_registry=self.capability_registry,
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
                if plugin.validation_errors:
                    self._mark_validation_failure(plugin)
                    await self.store.set_enabled(plugin.plugin_id, False)

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
                capability_registry=self.capability_registry,
            )
            if plugin is not None:
                if plugin.validation_errors:
                    self._mark_validation_failure(plugin)
                    await self.store.set_enabled(plugin.plugin_id, False)
                    continue
                plugin.enabled = True
                loaded.append(plugin)

        for plugin in loaded:
            if plugin.validation_errors:
                continue
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
