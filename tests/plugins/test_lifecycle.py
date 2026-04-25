"""Tests for PluginLifecycleManager."""

from pathlib import Path
import pytest
import pytest_asyncio

from opencas.platform import CapabilityRegistry, CapabilityStatus
from opencas.infra.hook_registry import TypedHookRegistry
from opencas.plugins import (
    PluginRegistry,
    PluginStore,
    SkillRegistry,
    build_plugin_bundle,
)
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
    capability_registry = CapabilityRegistry()
    mgr = PluginLifecycleManager(
        store=store,
        plugin_registry=plugin_registry,
        skill_registry=skill_registry,
        tools=tools,
        hook_registry=hook_registry,
        builtin_dir=None,
        install_root=tmp_path / "installed_plugins",
        capability_registry=capability_registry,
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
async def test_install_plugin_preserves_versioned_manifest_contract(
    lifecycle: PluginLifecycleManager, tmp_path: Path
) -> None:
    plugin_dir = tmp_path / "configured_plugin"
    plugin_dir.mkdir()
    manifest = {
        "id": "configured_plugin",
        "name": "Configured Plugin",
        "description": "",
        "manifest_version": 1,
        "version": "2.0.0",
        "config_schema": {
            "type": "object",
            "properties": {
                "profile": {"type": "string"},
            },
            "required": ["profile"],
        },
        "default_config": {"profile": "safe"},
        "capabilities": [
            {
                "capability_id": "plugin:configured_plugin.echo",
                "display_name": "Echo",
                "tool_names": ["echo_tool"],
            }
        ],
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))

    plugin = await lifecycle.install(plugin_dir)

    assert plugin is not None
    assert plugin.manifest_version == 1
    assert plugin.config_schema == manifest["config_schema"]
    assert plugin.default_config == manifest["default_config"]
    descriptor = lifecycle.capability_registry.get("plugin:configured_plugin.echo")
    assert descriptor is not None
    assert descriptor.metadata["manifest_version"] == 1
    assert descriptor.metadata["plugin_config_schema"] == manifest["config_schema"]
    assert descriptor.metadata["plugin_default_config"] == manifest["default_config"]


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
async def test_disable_updates_plugin_capability_status(
    lifecycle: PluginLifecycleManager, tmp_path: Path
) -> None:
    plugin_dir = tmp_path / "status_plugin"
    plugin_dir.mkdir()
    manifest = {
        "id": "status_plugin",
        "name": "Status Plugin",
        "description": "",
        "version": "1.0.0",
        "capabilities": [
            {
                "capability_id": "plugin:status_plugin.echo",
                "display_name": "Echo",
                "tool_names": ["tool_a"],
            }
        ],
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))

    plugin = await lifecycle.install(plugin_dir)
    assert plugin is not None

    descriptor = lifecycle.capability_registry.get("plugin:status_plugin.echo")
    assert descriptor is not None
    assert descriptor.status is CapabilityStatus.ENABLED

    await lifecycle.disable("status_plugin")

    disabled_descriptor = lifecycle.capability_registry.get("plugin:status_plugin.echo")
    assert disabled_descriptor is not None
    assert disabled_descriptor.status is CapabilityStatus.DISABLED


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
async def test_uninstall_removes_installed_plugin_directory(
    lifecycle: PluginLifecycleManager, tmp_path: Path
) -> None:
    plugin_dir = tmp_path / "packaged_removable"
    plugin_dir.mkdir()
    manifest = {
        "id": "packaged_removable",
        "name": "Packaged Removable",
        "description": "",
        "version": "1.0.0",
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))
    bundle_path = build_plugin_bundle(plugin_dir)

    plugin = await lifecycle.install(bundle_path)

    assert plugin is not None
    installed_dir = lifecycle.install_root / "packaged_removable"
    assert installed_dir.exists()

    await lifecycle.uninstall("packaged_removable")

    assert not installed_dir.exists()
    assert not await lifecycle.store.is_installed("packaged_removable")


@pytest.mark.asyncio
async def test_uninstall_removes_owned_capabilities_from_registry(
    lifecycle: PluginLifecycleManager, tmp_path: Path
) -> None:
    plugin_dir = tmp_path / "capable"
    plugin_dir.mkdir()
    manifest = {
        "id": "capable",
        "name": "Capable",
        "description": "",
        "version": "1.0.0",
        "entrypoint": "main.py",
        "capabilities": [
            {
                "capability_id": "plugin:capable.echo",
                "display_name": "Echo",
                "tool_names": ["echo_tool"],
            }
        ],
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))
    (plugin_dir / "main.py").write_text(
        "from opencas.autonomy.models import ActionRiskTier\n"
        "from opencas.plugins.models import SkillEntry\n"
        "from opencas.tools.models import ToolResult\n"
        "SKILL_ENTRY = {\n"
        "    'skill_id': 'capable_skill',\n"
        "    'name': 'Capable Skill',\n"
        "    'description': '',\n"
        "    'capabilities': ['echo_tool'],\n"
        "}\n"
        "def register_skills(skill_registry, tools):\n"
        "    tools.register('echo_tool', 'Echo tool', lambda n, a: ToolResult(success=True, output='ok', metadata={}), ActionRiskTier.READONLY, {})\n"
    )

    await lifecycle.install(plugin_dir)
    assert lifecycle.capability_registry.get("plugin:capable.echo") is not None
    assert lifecycle.tools.get("echo_tool") is not None

    await lifecycle.uninstall("capable")

    assert lifecycle.capability_registry.get("plugin:capable.echo") is None
    assert lifecycle.capability_registry.list_capabilities(owner_id="capable") == []
    assert lifecycle.tools.get("echo_tool") is None
    execution = await lifecycle.tools.execute_async("echo_tool", {})
    assert not execution.success
    assert execution.output == "Tool not found: echo_tool"
    assert lifecycle.skill_registry.get("capable_skill") is None


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


@pytest.mark.asyncio
async def test_install_plugin_bundle_extracts_into_install_root(
    lifecycle: PluginLifecycleManager, tmp_path: Path
) -> None:
    plugin_dir = tmp_path / "bundle_plugin"
    plugin_dir.mkdir()
    manifest = {
        "id": "bundle_plugin",
        "name": "Bundle Plugin",
        "description": "",
        "manifest_version": 1,
        "version": "1.0.0",
        "entrypoint": "main.py",
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))
    (plugin_dir / "main.py").write_text("def register_skills(skill_registry, tools):\n    pass\n")
    bundle_path = build_plugin_bundle(plugin_dir)

    plugin = await lifecycle.install(bundle_path)

    assert plugin is not None
    assert plugin.plugin_id == "bundle_plugin"
    assert plugin.path == str(tmp_path / "installed_plugins" / "bundle_plugin" / "plugin.json")
    assert (tmp_path / "installed_plugins" / "bundle_plugin" / "plugin.json").exists()


@pytest.mark.asyncio
async def test_reinstall_plugin_replaces_owned_capabilities(
    lifecycle: PluginLifecycleManager, tmp_path: Path
) -> None:
    plugin_dir = tmp_path / "replaceable_plugin"
    plugin_dir.mkdir()
    initial_manifest = {
        "id": "replaceable_plugin",
        "name": "Replaceable Plugin",
        "description": "",
        "manifest_version": 1,
        "version": "1.0.0",
        "capabilities": [
            {
                "capability_id": "plugin:replaceable_plugin.echo",
                "display_name": "Echo",
            }
        ],
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(initial_manifest))
    await lifecycle.install(plugin_dir)
    assert lifecycle.capability_registry.get("plugin:replaceable_plugin.echo") is not None

    updated_manifest = {
        "id": "replaceable_plugin",
        "name": "Replaceable Plugin",
        "description": "",
        "manifest_version": 1,
        "version": "2.0.0",
        "capabilities": [
            {
                "capability_id": "plugin:replaceable_plugin.status",
                "display_name": "Status",
            }
        ],
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(updated_manifest))

    plugin = await lifecycle.install(plugin_dir)

    assert plugin is not None
    assert plugin.version == "2.0.0"
    assert lifecycle.capability_registry.get("plugin:replaceable_plugin.echo") is None
    replacement = lifecycle.capability_registry.get("plugin:replaceable_plugin.status")
    assert replacement is not None
    assert replacement.metadata["version"] == "2.0.0"


@pytest.mark.asyncio
async def test_install_rejects_runtime_incompatible_plugin(
    lifecycle: PluginLifecycleManager, tmp_path: Path
) -> None:
    plugin_dir = tmp_path / "incompatible_plugin"
    plugin_dir.mkdir()
    manifest = {
        "id": "incompatible_plugin",
        "name": "Incompatible Plugin",
        "description": "",
        "manifest_version": 1,
        "version": "1.0.0",
        "compatibility": {
            "min_opencas_version": "9.0.0",
        },
        "capabilities": [
            {
                "capability_id": "plugin:incompatible_plugin.echo",
                "display_name": "Echo",
            }
        ],
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))

    plugin = await lifecycle.install(plugin_dir)

    assert plugin is None
    assert not await lifecycle.store.is_installed("incompatible_plugin")
    assert lifecycle.plugin_registry.get("incompatible_plugin") is None
    assert lifecycle.capability_registry.get("plugin:incompatible_plugin.echo") is None


@pytest.mark.asyncio
async def test_load_all_marks_incompatible_installed_plugin_failed_validation(
    lifecycle: PluginLifecycleManager, tmp_path: Path
) -> None:
    plugin_dir = tmp_path / "persisted_incompatible_plugin"
    plugin_dir.mkdir()
    manifest = {
        "id": "persisted_incompatible_plugin",
        "name": "Persisted Incompatible Plugin",
        "description": "",
        "manifest_version": 1,
        "version": "1.0.0",
        "compatibility": {
            "min_opencas_version": "9.0.0",
        },
        "capabilities": [
            {
                "capability_id": "plugin:persisted_incompatible_plugin.echo",
                "display_name": "Echo",
            }
        ],
    }
    manifest_path = plugin_dir / "plugin.json"
    manifest_path.write_text(__import__("json").dumps(manifest))
    await lifecycle.store.install(
        "persisted_incompatible_plugin",
        "Persisted Incompatible Plugin",
        "",
        "installed",
        str(manifest_path),
        manifest=manifest,
    )

    loaded = await lifecycle.load_all()

    assert loaded == []
    plugin = lifecycle.plugin_registry.get("persisted_incompatible_plugin")
    assert plugin is not None
    assert plugin.enabled is False
    assert plugin.validation_errors == [
        "requires OpenCAS >= 9.0.0, current runtime is 0.1.0"
    ]
    descriptor = lifecycle.capability_registry.get("plugin:persisted_incompatible_plugin.echo")
    assert descriptor is not None
    assert descriptor.status is CapabilityStatus.FAILED_VALIDATION
    assert descriptor.validation_errors == [
        "requires OpenCAS >= 9.0.0, current runtime is 0.1.0"
    ]
    assert not await lifecycle.store.is_enabled("persisted_incompatible_plugin")
