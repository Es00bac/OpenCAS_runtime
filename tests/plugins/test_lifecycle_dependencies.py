"""Tests for plugin lifecycle dependency validation."""

from pathlib import Path
import pytest
import pytest_asyncio

from opencas.platform import CapabilityRegistry, CapabilityStatus
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
    capability_registry = CapabilityRegistry()
    mgr = PluginLifecycleManager(
        store=store,
        plugin_registry=plugin_registry,
        skill_registry=skill_registry,
        tools=tools,
        hook_registry=None,
        builtin_dir=None,
        capability_registry=capability_registry,
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
        "entrypoint": "main.py",
        "dependencies": ["dep_plugin"],
        "capabilities": [
            {
                "capability_id": "plugin:needs_dep.echo",
                "display_name": "Echo",
                "tool_names": ["needs_dep_tool"],
            }
        ],
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))
    (plugin_dir / "main.py").write_text(
        "from opencas.autonomy.models import ActionRiskTier\n"
        "from opencas.plugins.models import SkillEntry\n"
        "from opencas.tools.models import ToolResult\n"
        "SKILL_ENTRY = {\n"
        "    'skill_id': 'needs_dep_skill',\n"
        "    'name': 'Needs Dep Skill',\n"
        "    'description': '',\n"
        "    'capabilities': ['needs_dep_tool'],\n"
        "}\n"
        "def register_skills(skill_registry, tools):\n"
        "    tools.register('needs_dep_tool', 'Needs Dep Tool', lambda n, a: ToolResult(success=True, output='ok', metadata={}), ActionRiskTier.READONLY, {})\n"
    )

    # Without dep installed, install fails
    result = await lifecycle.install(plugin_dir)
    assert result is None

    assert lifecycle.plugin_registry.get("needs_dep") is None
    assert lifecycle.tools.get("needs_dep_tool") is None
    execution = await lifecycle.tools.execute_async("needs_dep_tool", {})
    assert not execution.success
    assert execution.output == "Tool not found: needs_dep_tool"
    assert lifecycle.skill_registry.get("needs_dep_skill") is None
    descriptor = lifecycle.capability_registry.get("plugin:needs_dep.echo")
    assert descriptor is not None
    assert descriptor.status is CapabilityStatus.MISSING_DEPENDENCY
    assert descriptor.validation_errors == ["missing dependency: dep_plugin"]

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
        "entrypoint": "main.py",
        "dependencies": ["dep_plugin"],
        "capabilities": [
            {
                "capability_id": "plugin:needs_dep.echo",
                "display_name": "Echo",
                "tool_names": ["needs_dep_tool"],
            }
        ],
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))
    (plugin_dir / "main.py").write_text(
        "from opencas.autonomy.models import ActionRiskTier\n"
        "from opencas.plugins.models import SkillEntry\n"
        "from opencas.tools.models import ToolResult\n"
        "SKILL_ENTRY = {\n"
        "    'skill_id': 'needs_dep_skill',\n"
        "    'name': 'Needs Dep Skill',\n"
        "    'description': '',\n"
        "    'capabilities': ['needs_dep_tool'],\n"
        "}\n"
        "def register_skills(skill_registry, tools):\n"
        "    tools.register('needs_dep_tool', 'Needs Dep Tool', lambda n, a: ToolResult(success=True, output='ok', metadata={}), ActionRiskTier.READONLY, {})\n"
    )
    await lifecycle.install(plugin_dir)

    await lifecycle.disable("dep_plugin")
    await lifecycle.disable("needs_dep")
    lifecycle.plugin_registry = PluginRegistry()
    # can't enable needs_dep while dep_plugin is disabled
    await lifecycle.enable("needs_dep")
    assert not await lifecycle.store.is_enabled("needs_dep")

    assert lifecycle.plugin_registry.get("needs_dep") is None
    assert lifecycle.tools.get("needs_dep_tool") is None
    execution = await lifecycle.tools.execute_async("needs_dep_tool", {})
    assert not execution.success
    assert execution.output == "Tool not found: needs_dep_tool"
    assert lifecycle.skill_registry.get("needs_dep_skill") is None
    descriptor = lifecycle.capability_registry.get("plugin:needs_dep.echo")
    assert descriptor is not None
    assert descriptor.status is CapabilityStatus.MISSING_DEPENDENCY
    assert descriptor.validation_errors == ["missing dependency: dep_plugin"]

    await lifecycle.enable("dep_plugin")
    await lifecycle.enable("needs_dep")
    assert await lifecycle.store.is_enabled("needs_dep")
