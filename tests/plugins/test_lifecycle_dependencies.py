"""Tests for plugin lifecycle dependency validation."""

from pathlib import Path
import pytest
import pytest_asyncio

from opencas.plugins import PluginRegistry, PluginStore, SkillRegistry
from opencas.plugins.lifecycle import PluginLifecycleManager
from opencas.tools import ToolRegistry


@pytest_asyncio.fixture
async def lifecycle(tmp_path: Path):
    store = PluginStore(tmp_path / "plugins.db")
    await store.connect()
    plugin_registry = PluginRegistry()
    skill_registry = SkillRegistry()
    tools = ToolRegistry()
    mgr = PluginLifecycleManager(
        store=store,
        plugin_registry=plugin_registry,
        skill_registry=skill_registry,
        tools=tools,
        hook_registry=None,
        builtin_dir=None,
    )
    yield mgr
    await store.close()


@pytest.mark.asyncio
async def test_install_requires_dependencies_installed(
    lifecycle: PluginLifecycleManager, tmp_path: Path
) -> None:
    dep_dir = tmp_path / "dep_plugin"
    dep_dir.mkdir()
    dep_manifest = {
        "id": "dep_plugin",
        "name": "Dependency Plugin",
        "description": "",
        "version": "1.0.0",
    }
    (dep_dir / "plugin.json").write_text(__import__("json").dumps(dep_manifest))

    plugin_dir = tmp_path / "needs_dep"
    plugin_dir.mkdir()
    manifest = {
        "id": "needs_dep",
        "name": "Needs Dep",
        "description": "",
        "version": "1.0.0",
        "dependencies": ["dep_plugin"],
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))

    # Without dep installed, install fails
    result = await lifecycle.install(plugin_dir)
    assert result is None

    # Install dep first
    dep_result = await lifecycle.install(dep_dir)
    assert dep_result is not None

    # Now plugin can install
    result = await lifecycle.install(plugin_dir)
    assert result is not None
    assert result.plugin_id == "needs_dep"


@pytest.mark.asyncio
async def test_enable_requires_dependencies_enabled(
    lifecycle: PluginLifecycleManager, tmp_path: Path
) -> None:
    dep_dir = tmp_path / "dep_plugin"
    dep_dir.mkdir()
    dep_manifest = {
        "id": "dep_plugin",
        "name": "Dependency Plugin",
        "description": "",
        "version": "1.0.0",
    }
    (dep_dir / "plugin.json").write_text(__import__("json").dumps(dep_manifest))
    await lifecycle.install(dep_dir)

    plugin_dir = tmp_path / "needs_dep"
    plugin_dir.mkdir()
    manifest = {
        "id": "needs_dep",
        "name": "Needs Dep",
        "description": "",
        "version": "1.0.0",
        "dependencies": ["dep_plugin"],
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))
    await lifecycle.install(plugin_dir)

    await lifecycle.disable("dep_plugin")
    await lifecycle.disable("needs_dep")
    # can't enable needs_dep while dep_plugin is disabled
    await lifecycle.enable("needs_dep")
    assert not await lifecycle.store.is_enabled("needs_dep")

    await lifecycle.enable("dep_plugin")
    await lifecycle.enable("needs_dep")
    assert await lifecycle.store.is_enabled("needs_dep")
