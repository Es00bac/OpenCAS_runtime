"""Tests for plugin manifest loading."""

from pathlib import Path

from opencas.platform import CapabilityRegistry, CapabilitySource, CapabilityStatus
from opencas.infra.hook_registry import TypedHookRegistry
from opencas.plugins import (
    PluginManifestError,
    PluginRegistry,
    PluginStore,
    SkillRegistry,
    classify_plugin_update,
    evaluate_plugin_compatibility,
    load_plugin_manifest,
    load_builtin_plugins,
    load_plugin_from_manifest,
)
from opencas.plugins.lifecycle import PluginLifecycleManager
from opencas.tools import ToolRegistry


def test_load_plugin_manifest_applies_phase_two_defaults(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "manifest_defaults"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "manifest_defaults",
                "name": "Manifest Defaults",
                "description": "Defaults should be applied.",
            }
        )
    )

    manifest = load_plugin_manifest(plugin_dir / "plugin.json")

    assert manifest["manifest_version"] == 1
    assert manifest["version"] == "0.0.1"
    assert manifest["config_schema"] == {}
    assert manifest["default_config"] == {}
    assert manifest["distribution"] == {}
    assert manifest["release_notes"] == ""


def test_load_plugin_manifest_validates_plugin_level_config_schema_and_defaults(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "configured_manifest"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "configured_manifest",
                "name": "Configured Manifest",
                "description": "Plugin-level config schema should be validated.",
                "manifest_version": 1,
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "profile": {"type": "string"},
                        "retries": {"type": "integer"},
                    },
                    "required": ["profile"],
                    "additionalProperties": False,
                },
                "default_config": {"profile": "default", "retries": 2},
            }
        )
    )

    manifest = load_plugin_manifest(plugin_dir / "plugin.json")

    assert manifest["manifest_version"] == 1
    assert manifest["config_schema"]["properties"]["profile"]["type"] == "string"
    assert manifest["default_config"] == {"profile": "default", "retries": 2}


def test_load_plugin_manifest_rejects_invalid_plugin_level_config_schema(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "invalid_schema_manifest"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "invalid_schema_manifest",
                "name": "Invalid Schema Manifest",
                "description": "",
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "profile": {"type": "bogus"},
                    },
                },
            }
        )
    )

    try:
        load_plugin_manifest(plugin_dir / "plugin.json")
    except PluginManifestError as exc:
        assert "config_schema.properties.profile.type must be one of" in exc.errors[0]
    else:  # pragma: no cover - defensive
        raise AssertionError("expected PluginManifestError")


def test_load_plugin_manifest_rejects_invalid_default_config(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "invalid_default_manifest"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "invalid_default_manifest",
                "name": "Invalid Default Manifest",
                "description": "",
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "profile": {"type": "string"},
                    },
                    "required": ["profile"],
                    "additionalProperties": False,
                },
                "default_config": {"profile": 42},
            }
        )
    )

    try:
        load_plugin_manifest(plugin_dir / "plugin.json")
    except PluginManifestError as exc:
        assert exc.errors == ["default_config.profile must be a string"]
    else:  # pragma: no cover - defensive
        raise AssertionError("expected PluginManifestError")


def test_load_plugin_manifest_validates_compatibility_bounds(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "compat_manifest"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "compat_manifest",
                "name": "Compat Manifest",
                "description": "",
                "compatibility": {
                    "min_opencas_version": "0.1.0",
                    "max_opencas_version": "0.3.0",
                },
            }
        )
    )

    manifest = load_plugin_manifest(plugin_dir / "plugin.json")

    assert manifest["compatibility"] == {
        "min_opencas_version": "0.1.0",
        "max_opencas_version": "0.3.0",
    }


def test_load_plugin_manifest_accepts_distribution_and_release_notes(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "distribution_manifest"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "distribution_manifest",
                "name": "Distribution Manifest",
                "description": "",
                "distribution": {
                    "publisher": "OpenCAS Labs",
                    "channel": "beta",
                    "source_url": "https://example.com/distribution-manifest",
                    "documentation_url": "https://example.com/distribution-manifest/docs",
                    "changelog_url": "https://example.com/distribution-manifest/changelog",
                },
                "release_notes": "Adds provenance visibility to the platform control plane.",
            }
        )
    )

    manifest = load_plugin_manifest(plugin_dir / "plugin.json")

    assert manifest["distribution"]["publisher"] == "OpenCAS Labs"
    assert manifest["distribution"]["channel"] == "beta"
    assert manifest["release_notes"] == "Adds provenance visibility to the platform control plane."


def test_load_plugin_manifest_accepts_distribution_signatures(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "signed_distribution_manifest"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "signed_distribution_manifest",
                "name": "Signed Distribution Manifest",
                "description": "",
                "distribution": {
                    "publisher": "OpenCAS Labs",
                    "signatures": [
                        {
                            "key_id": "opencas-labs-main",
                            "algorithm": "ed25519",
                            "public_key": "QUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUE=",
                            "signature": "QUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQQ==",
                        }
                    ],
                },
            }
        )
    )

    manifest = load_plugin_manifest(plugin_dir / "plugin.json")

    assert manifest["distribution"]["signatures"][0]["key_id"] == "opencas-labs-main"


def test_load_plugin_manifest_rejects_invalid_distribution_signature_shape(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "invalid_signature_distribution_manifest"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "invalid_signature_distribution_manifest",
                "name": "Invalid Signature Distribution Manifest",
                "description": "",
                "distribution": {
                    "signatures": [
                        {
                            "key_id": "",
                            "algorithm": "rsa",
                            "signature": "",
                        }
                    ],
                },
            }
        )
    )

    try:
        load_plugin_manifest(plugin_dir / "plugin.json")
    except PluginManifestError as exc:
        assert "distribution.signatures[0].key_id must be a non-empty string" in exc.errors
        assert "distribution.signatures[0].algorithm must be one of ['ed25519']" in exc.errors
        assert "distribution.signatures[0].signature must be a non-empty string" in exc.errors
        assert "distribution.signatures[0].public_key must be a non-empty string" in exc.errors
    else:  # pragma: no cover - defensive
        raise AssertionError("expected PluginManifestError")


def test_load_plugin_manifest_rejects_invalid_distribution_shape(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "invalid_distribution_manifest"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "invalid_distribution_manifest",
                "name": "Invalid Distribution Manifest",
                "description": "",
                "distribution": {
                    "publisher": ["not", "a", "string"],
                    "unexpected": "value",
                },
            }
        )
    )

    try:
        load_plugin_manifest(plugin_dir / "plugin.json")
    except PluginManifestError as exc:
        assert "distribution.publisher must be a string" in exc.errors
        assert "distribution.unexpected is not supported" in exc.errors
    else:  # pragma: no cover - defensive
        raise AssertionError("expected PluginManifestError")


def test_load_plugin_manifest_rejects_invalid_compatibility_range(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "invalid_compat_manifest"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "invalid_compat_manifest",
                "name": "Invalid Compat Manifest",
                "description": "",
                "compatibility": {
                    "min_opencas_version": "0.4.0",
                    "max_opencas_version": "0.3.0",
                },
            }
        )
    )

    try:
        load_plugin_manifest(plugin_dir / "plugin.json")
    except PluginManifestError as exc:
        assert exc.errors == [
            "compatibility.min_opencas_version cannot be greater than compatibility.max_opencas_version"
        ]
    else:  # pragma: no cover - defensive
        raise AssertionError("expected PluginManifestError")


def test_evaluate_plugin_compatibility_reports_incompatible_runtime() -> None:
    compatibility = evaluate_plugin_compatibility(
        {
            "compatibility": {
                "min_opencas_version": "0.2.0",
                "max_opencas_version": "0.9.0",
            }
        },
        runtime_version="0.1.0",
    )

    assert compatibility["compatible"] is False
    assert compatibility["constraints"] == {
        "min_opencas_version": "0.2.0",
        "max_opencas_version": "0.9.0",
    }
    assert compatibility["reasons"] == [
        "requires OpenCAS >= 0.2.0, current runtime is 0.1.0"
    ]


def test_classify_plugin_update_distinguishes_install_upgrade_downgrade() -> None:
    assert classify_plugin_update(None, "1.0.0") == "install"
    assert classify_plugin_update("1.0.0", "1.2.0") == "upgrade"
    assert classify_plugin_update("1.2.0", "1.0.0") == "downgrade"
    assert classify_plugin_update("1.2.0", "1.2.0") == "reinstall"


def test_load_plugin_from_manifest_registers_capabilities_in_platform_registry(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "capability_plugin"
    plugin_dir.mkdir()
    manifest = {
        "id": "capability_plugin",
        "name": "Capability Plugin",
        "description": "A plugin with declared capabilities.",
        "version": "2.1.0",
        "entrypoint": "main.py",
        "capabilities": [
            {
                "capability_id": "plugin:capability_plugin.echo",
                "display_name": "Echo",
                "description": "Echo messages.",
                "tool_names": ["echo_tool"],
                "dependencies": ["network"],
                "config_schema": {"type": "object"},
                "metadata": {"owner_name": "Capability Plugin Owner"},
            }
        ],
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))
    (plugin_dir / "main.py").write_text("def register_skills(skill_registry, tools):\n    pass\n")

    plugin_registry = PluginRegistry()
    skill_registry = SkillRegistry()
    tools = ToolRegistry()
    capability_registry = CapabilityRegistry()

    plugin = load_plugin_from_manifest(
        plugin_dir / "plugin.json",
        plugin_registry,
        skill_registry,
        tools,
        capability_registry=capability_registry,
        source="installed",
    )

    assert plugin is not None
    descriptor = capability_registry.get("plugin:capability_plugin.echo")
    assert descriptor is not None
    assert descriptor.source is CapabilitySource.PLUGIN
    assert descriptor.owner_id == "capability_plugin"
    assert descriptor.status is CapabilityStatus.ENABLED
    assert descriptor.entrypoint == "main.py"
    assert descriptor.manifest_path == str(plugin_dir / "plugin.json")
    assert descriptor.source_path == str(plugin_dir)
    assert descriptor.metadata["version"] == "2.1.0"
    assert descriptor.metadata["owner_name"] == "Capability Plugin"
    assert descriptor.metadata["manifest_version"] == 1
    assert descriptor.metadata["plugin_config_schema"] == {}
    assert descriptor.metadata["plugin_default_config"] == {}
    assert descriptor.metadata["compatibility"]["runtime_version"] == "0.1.0"
    assert descriptor.metadata["compatibility"]["compatible"] is True
    assert descriptor.tool_names == ["echo_tool"]
    assert descriptor.declared_dependencies == ["network"]
    assert descriptor.config_schema == {"type": "object"}


def test_load_plugin_from_manifest_registers_fallback_capability_descriptor(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "fallback_plugin"
    plugin_dir.mkdir()
    manifest = {
        "id": "fallback_plugin",
        "name": "Fallback Plugin",
        "description": "A plugin without declared capabilities.",
        "version": "1.0.0",
        "entrypoint": "main.py",
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))
    (plugin_dir / "main.py").write_text("def register_skills(skill_registry, tools):\n    pass\n")

    plugin_registry = PluginRegistry()
    skill_registry = SkillRegistry()
    tools = ToolRegistry()
    capability_registry = CapabilityRegistry()

    plugin = load_plugin_from_manifest(
        plugin_dir / "plugin.json",
        plugin_registry,
        skill_registry,
        tools,
        capability_registry=capability_registry,
        source="installed",
    )

    assert plugin is not None
    descriptor = capability_registry.get("plugin:fallback_plugin")
    assert descriptor is not None
    assert descriptor.kind == "plugin"
    assert descriptor.source is CapabilitySource.PLUGIN
    assert descriptor.owner_id == "fallback_plugin"
    assert descriptor.status is CapabilityStatus.ENABLED
    assert descriptor.entrypoint == "main.py"
    assert descriptor.manifest_path == str(plugin_dir / "plugin.json")
    assert descriptor.source_path == str(plugin_dir)
    assert descriptor.description == "A plugin without declared capabilities."


def test_load_plugin_from_manifest_keeps_positional_source_compatibility(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "compat_plugin"
    plugin_dir.mkdir()
    manifest = {
        "id": "compat_plugin",
        "name": "Compat Plugin",
        "description": "A plugin for compatibility coverage.",
        "version": "1.0.0",
        "capabilities": [
            {
                "capability_id": "plugin:compat_plugin.echo",
                "display_name": "Echo",
            }
        ],
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))

    plugin_registry = PluginRegistry()
    skill_registry = SkillRegistry()
    tools = ToolRegistry()
    capability_registry = CapabilityRegistry()

    plugin = load_plugin_from_manifest(
        plugin_dir / "plugin.json",
        plugin_registry,
        skill_registry,
        tools,
        None,
        "installed",
        capability_registry=capability_registry,
    )

    assert plugin is not None
    descriptor = capability_registry.get("plugin:compat_plugin.echo")
    assert descriptor is not None
    assert descriptor.source is CapabilitySource.PLUGIN
    assert descriptor.metadata["version"] == "1.0.0"


def test_load_plugin_from_manifest_preserves_nested_entrypoint_string(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "nested_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "pkg").mkdir()
    manifest = {
        "id": "nested_plugin",
        "name": "Nested Plugin",
        "description": "A plugin with a nested entrypoint.",
        "version": "1.0.0",
        "entrypoint": "pkg/main.py",
        "capabilities": [
            {
                "capability_id": "plugin:nested_plugin.echo",
                "display_name": "Echo",
            }
        ],
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))
    (plugin_dir / "pkg" / "main.py").write_text(
        "def register_skills(skill_registry, tools):\n"
        "    pass\n"
    )

    plugin_registry = PluginRegistry()
    skill_registry = SkillRegistry()
    tools = ToolRegistry()
    capability_registry = CapabilityRegistry()

    plugin = load_plugin_from_manifest(
        plugin_dir / "plugin.json",
        plugin_registry,
        skill_registry,
        tools,
        capability_registry=capability_registry,
        source="installed",
    )

    assert plugin is not None
    descriptor = capability_registry.get("plugin:nested_plugin.echo")
    assert descriptor is not None
    assert descriptor.entrypoint == "pkg/main.py"
    assert descriptor.source_path == str(plugin_dir)
    assert descriptor.metadata["version"] == "1.0.0"


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
    assert plugin.manifest_version == 1
    assert plugin.config_schema == {}
    assert plugin.default_config == {}
    assert plugin_registry.get("test_plugin") == plugin
    # entrypoint registered both skills
    assert skill_registry.get("main_skill") is not None
    assert skill_registry.get("second_skill") is not None


def test_load_plugin_from_manifest_surfaces_plugin_level_config_contract(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "configured_plugin"
    plugin_dir.mkdir()
    manifest = {
        "id": "configured_plugin",
        "name": "Configured Plugin",
        "description": "A plugin with plugin-level config.",
        "version": "1.0.0",
        "manifest_version": 1,
        "config_schema": {
            "type": "object",
            "properties": {
                "profile": {"type": "string"},
                "mode": {"type": "string", "enum": ["safe", "fast"]},
            },
            "required": ["profile"],
        },
        "default_config": {"profile": "safe-default", "mode": "safe"},
        "capabilities": [
            {
                "capability_id": "plugin:configured_plugin.echo",
                "display_name": "Echo",
            }
        ],
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))

    plugin_registry = PluginRegistry()
    skill_registry = SkillRegistry()
    tools = ToolRegistry()
    capability_registry = CapabilityRegistry()

    plugin = load_plugin_from_manifest(
        plugin_dir / "plugin.json",
        plugin_registry,
        skill_registry,
        tools,
        capability_registry=capability_registry,
        source="installed",
    )

    assert plugin is not None
    assert plugin.manifest_version == 1
    assert plugin.config_schema == manifest["config_schema"]
    assert plugin.default_config == manifest["default_config"]

    descriptor = capability_registry.get("plugin:configured_plugin.echo")
    assert descriptor is not None
    assert descriptor.metadata["manifest_version"] == 1
    assert descriptor.metadata["plugin_config_schema"] == manifest["config_schema"]
    assert descriptor.metadata["plugin_default_config"] == manifest["default_config"]


def test_load_plugin_from_manifest_returns_none_for_invalid_versioned_manifest(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "invalid_loader_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "invalid_loader_plugin",
                "name": "Invalid Loader Plugin",
                "description": "",
                "manifest_version": 99,
            }
        )
    )

    plugin_registry = PluginRegistry()
    skill_registry = SkillRegistry()
    tools = ToolRegistry()

    plugin = load_plugin_from_manifest(
        plugin_dir / "plugin.json",
        plugin_registry,
        skill_registry,
        tools,
    )

    assert plugin is None


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


def test_load_builtin_plugins_emits_manifest_capabilities(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "builtin_manifest_plugin"
    plugin_dir.mkdir()
    manifest = {
        "id": "builtin_manifest_plugin",
        "name": "Builtin Manifest Plugin",
        "description": "Builtin plugin with capabilities.",
        "version": "3.0.0",
        "capabilities": [
            {
                "capability_id": "plugin:builtin_manifest_plugin.echo",
                "display_name": "Echo",
                "tool_names": ["echo_tool"],
            }
        ],
    }
    (plugin_dir / "plugin.json").write_text(__import__("json").dumps(manifest))

    plugin_registry = PluginRegistry()
    skill_registry = SkillRegistry()
    tools = ToolRegistry()
    capability_registry = CapabilityRegistry()

    plugins = load_builtin_plugins(
        tmp_path,
        plugin_registry,
        skill_registry,
        tools,
        capability_registry=capability_registry,
    )

    assert len(plugins) == 1
    descriptor = capability_registry.get("plugin:builtin_manifest_plugin.echo")
    assert descriptor is not None
    assert descriptor.source is CapabilitySource.PLUGIN
    assert descriptor.owner_id == "builtin_manifest_plugin"
    assert descriptor.source_path == str(plugin_dir)


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
