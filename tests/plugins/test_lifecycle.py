"""Tests for PluginLifecycleManager."""

from pathlib import Path
import pytest
import pytest_asyncio

from opencas.infra.hook_registry import TypedHookRegistry
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
    hook_registry = TypedHookRegistry()
    mgr = PluginLifecycleManager(
        store=store,
        plugin_registry=plugin_registry,
        skill_registry=skill_registry,
        tools=tools,
        hook_registry=hook_registry,
        builtin_dir=None,
    )
    yield mgr
    await store.close()


@pytest.mark.asyncio
async def test_install_plugin(lifecycle: PluginLifecycleManager, tmp_path: Path) -> None:
    plugin_dir = tmp_path / "my_plugin"
    plugin_dir.mkdir()
    manifest = {
        "id": "my_plugin",
        "name": "My Plugin",
        "description": "",
        "version": "1.0.0",
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))

    plugin = await lifecycle.install(plugin_dir)
    assert plugin is not None
    assert plugin.plugin_id == "my_plugin"
    assert await lifecycle.store.is_installed("my_plugin")


@pytest.mark.asyncio
async def test_disable_blocks_tools(lifecycle: PluginLifecycleManager, tmp_path: Path) -> None:
    plugin_dir = tmp_path / "tool_plugin"
    plugin_dir.mkdir()
    manifest = {
        "id": "tool_plugin",
        "name": "Tool Plugin",
        "description": "",
        "version": "1.0.0",
        "entrypoint": "main.py",
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))
    (plugin_dir / "main.py").write_text(
        "from opencas.autonomy.models import ActionRiskTier\n"
        "from opencas.tools.models import ToolResult\n"
        "def register_skills(skill_registry, tools):\n"
        "    tools.register('tool_a', 'A tool', lambda n, a: ToolResult(success=True, output='ok', metadata={}), ActionRiskTier.READONLY, {}, plugin_id='tool_plugin')\n"
    )

    plugin = await lifecycle.install(plugin_dir)
    assert plugin is not None
    assert not lifecycle.is_tool_disabled("tool_a")

    await lifecycle.disable("tool_plugin")
    assert lifecycle.is_tool_disabled("tool_a")

    await lifecycle.enable("tool_plugin")
    assert not lifecycle.is_tool_disabled("tool_a")


@pytest.mark.asyncio
async def test_uninstall_removes_plugin(lifecycle: PluginLifecycleManager, tmp_path: Path) -> None:
    plugin_dir = tmp_path / "removable"
    plugin_dir.mkdir()
    manifest = {"id": "removable", "name": "Removable", "description": "", "version": "1.0.0"}
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))

    await lifecycle.install(plugin_dir)
    assert await lifecycle.store.is_installed("removable")
    await lifecycle.uninstall("removable")
    assert not await lifecycle.store.is_installed("removable")
    assert lifecycle.plugin_registry.get("removable") is None


@pytest.mark.asyncio
async def test_load_all_restores_enabled_plugins(lifecycle: PluginLifecycleManager, tmp_path: Path) -> None:
    plugin_dir = tmp_path / "restorable"
    plugin_dir.mkdir()
    manifest = {"id": "restorable", "name": "Restorable", "description": "", "version": "1.0.0"}
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))

    await lifecycle.install(plugin_dir)
    await lifecycle.disable("restorable")

    # Reset registry to simulate fresh boot
    lifecycle.plugin_registry = PluginRegistry()
    lifecycle.skill_registry = SkillRegistry()
    if lifecycle.hook_registry is not None:
        lifecycle.hook_registry._handlers.clear()

    await lifecycle.load_all()
    # Since it was disabled, load_all should NOT re-register it as active
    # (it loads builtins only; installed disabled plugins are skipped)
    assert lifecycle.plugin_registry.get("restorable") is None

    # Enable and re-test
    await lifecycle.store.set_enabled("restorable", True)
    await lifecycle.load_all()
    restored = lifecycle.plugin_registry.get("restorable")
    assert restored is not None
    assert restored.enabled is True
