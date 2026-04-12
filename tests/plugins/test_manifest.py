"""Tests for plugin manifest loading."""

from pathlib import Path

from opencas.infra.hook_registry import TypedHookRegistry
from opencas.plugins import (
    PluginRegistry,
    PluginStore,
    SkillRegistry,
    load_builtin_plugins,
    load_plugin_from_manifest,
)
from opencas.plugins.lifecycle import PluginLifecycleManager
from opencas.tools import ToolRegistry


def test_load_plugin_from_manifest(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "test_plugin"
    plugin_dir.mkdir()
    manifest = {
        "id": "test_plugin",
        "name": "Test Plugin",
        "description": "A test plugin.",
        "version": "1.0.0",
        "entrypoint": "main.py",
        "skills": ["skill.py"],
    }
    manifest_path = plugin_dir / "plugin.json"
    manifest_path.write_text(__import__("json").dumps(manifest))

    # entrypoint module registers skills directly
    (plugin_dir / "main.py").write_text(
        "def register_skills(skill_registry, tools):\n"
        "    from opencas.plugins import SkillEntry\n"
        "    skill_registry.register(SkillEntry(skill_id='main_skill', name='Main', description=''))\n"
        "    skill_registry.register(SkillEntry(skill_id='second_skill', name='Second', description=''))\n"
    )

    plugin_registry = PluginRegistry()
    skill_registry = SkillRegistry()
    tools = ToolRegistry()

    plugin = load_plugin_from_manifest(
        manifest_path,
        plugin_registry,
        skill_registry,
        tools,
        source="installed",
    )

    assert plugin is not None
    assert plugin.plugin_id == "test_plugin"
    assert plugin.name == "Test Plugin"
    assert plugin_registry.get("test_plugin") == plugin
    # entrypoint registered both skills
    assert skill_registry.get("main_skill") is not None
    assert skill_registry.get("second_skill") is not None


def test_load_builtin_plugins_with_manifest(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "manifest_plugin"
    plugin_dir.mkdir()
    manifest = {
        "id": "manifest_plugin",
        "name": "Manifest Plugin",
        "description": "",
        "version": "1.0.0",
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))

    plugin_registry = PluginRegistry()
    skill_registry = SkillRegistry()
    tools = ToolRegistry()

    plugins = load_builtin_plugins(
        tmp_path,
        plugin_registry,
        skill_registry,
        tools,
    )

    assert len(plugins) == 1
    assert plugins[0].plugin_id == "manifest_plugin"
    assert plugins[0].source == "builtin"


def test_load_builtin_plugins_with_legacy_skills(tmp_path: Path) -> None:
    (tmp_path / "legacy_skill.py").write_text(
        "from opencas.plugins import SkillEntry\n"
        "SKILL_ENTRY = SkillEntry(skill_id='legacy', name='Legacy', description='')\n"
    )

    plugin_registry = PluginRegistry()
    skill_registry = SkillRegistry()
    tools = ToolRegistry()

    plugins = load_builtin_plugins(
        tmp_path,
        plugin_registry,
        skill_registry,
        tools,
    )

    assert len(plugins) == 1
    assert plugins[0].plugin_id == "legacy_skill"
    assert plugins[0].source == "builtin"
    assert skill_registry.get("legacy") is not None


def test_manifest_hooks_registration(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "hook_plugin"
    plugin_dir.mkdir()
    manifest = {
        "id": "hook_plugin",
        "name": "Hook Plugin",
        "description": "",
        "version": "1.0.0",
        "entrypoint": "hooks.py",
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))
    (plugin_dir / "hooks.py").write_text(
        "from opencas.infra.hook_registry import HookResult\n"
        "def my_hook(_, ctx):\n"
        "    return HookResult(allowed=False, reason='hooked')\n"
        "def register_hooks(registry):\n"
        "    registry.register('my_hook', my_hook, priority=5, source='hook_plugin')\n"
    )

    plugin_registry = PluginRegistry()
    skill_registry = SkillRegistry()
    tools = ToolRegistry()
    hook_registry = TypedHookRegistry()
    hook_registry.register_spec(__import__("opencas.infra.hook_registry", fromlist=["HookSpec"]).HookSpec(name="my_hook"))

    plugin = load_plugin_from_manifest(
        plugin_dir / "plugin.json",
        plugin_registry,
        skill_registry,
        tools,
        hook_registry,
    )

    assert plugin is not None
    result = hook_registry.run("my_hook", {})
    assert result.allowed is False
    assert result.reason == "hooked"
