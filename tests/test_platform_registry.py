from __future__ import annotations

from opencas.platform.models import CapabilityDescriptor, CapabilitySource, CapabilityStatus
from opencas.platform.projections import build_extension_descriptors
from opencas.platform.registry import CapabilityRegistry


def test_registry_stores_capabilities_by_id() -> None:
    registry = CapabilityRegistry()
    descriptor = CapabilityDescriptor(
        capability_id="core:fs.read",
        display_name="Filesystem Read",
        kind="tool",
        source=CapabilitySource.CORE,
        owner_id="core",
        status=CapabilityStatus.ENABLED,
        tool_names=["fs_read_file"],
    )

    registry.register(descriptor)

    assert registry.get("core:fs.read") == descriptor
    assert [item.capability_id for item in registry.list_capabilities()] == ["core:fs.read"]


def test_registry_updates_status_without_replacing_identity() -> None:
    registry = CapabilityRegistry()
    descriptor = CapabilityDescriptor(
        capability_id="plugin:test.echo",
        display_name="Echo",
        kind="tool",
        source=CapabilitySource.PLUGIN,
        owner_id="test_plugin",
        status=CapabilityStatus.ENABLED,
        tool_names=["echo_tool"],
    )
    registry.register(descriptor)

    registry.update_status(
        "plugin:test.echo",
        CapabilityStatus.DISABLED,
        errors=["disabled by operator"],
    )

    updated = registry.get("plugin:test.echo")
    assert updated is not None
    assert updated.status is CapabilityStatus.DISABLED
    assert updated.validation_errors == ["disabled by operator"]


def test_registry_unregisters_all_capabilities_for_owner() -> None:
    registry = CapabilityRegistry()
    registry.register(
        CapabilityDescriptor(
            capability_id="plugin:test.echo",
            display_name="Echo",
            kind="tool",
            source=CapabilitySource.PLUGIN,
            owner_id="test_plugin",
            status=CapabilityStatus.ENABLED,
            tool_names=["echo_tool"],
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="plugin:test.status",
            display_name="Status",
            kind="tool",
            source=CapabilitySource.PLUGIN,
            owner_id="test_plugin",
            status=CapabilityStatus.DISABLED,
            tool_names=["status_tool"],
        )
    )

    registry.unregister_owner("test_plugin")

    assert registry.get("plugin:test.echo") is None
    assert registry.get("plugin:test.status") is None
    assert registry.list_capabilities(owner_id="test_plugin") == []


def test_registry_filters_by_source_and_owner() -> None:
    registry = CapabilityRegistry()
    registry.register(
        CapabilityDescriptor(
            capability_id="core:bash.run",
            display_name="Run Shell Command",
            kind="tool",
            source=CapabilitySource.CORE,
            owner_id="core",
            status=CapabilityStatus.ENABLED,
            tool_names=["bash_run_command"],
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="mcp:filesystem.read",
            display_name="MCP Filesystem Read",
            kind="tool",
            source=CapabilitySource.MCP,
            owner_id="mcp:filesystem",
            status=CapabilityStatus.ENABLED,
            tool_names=["mcp__filesystem__read_file"],
        )
    )

    assert [item.capability_id for item in registry.list_capabilities(source="mcp")] == [
        "mcp:filesystem.read"
    ]
    assert [item.capability_id for item in registry.list_capabilities(owner_id="core")] == [
        "core:bash.run"
    ]


def test_build_extension_descriptors_groups_capabilities_by_owner() -> None:
    registry = CapabilityRegistry()
    registry.register(
        CapabilityDescriptor(
            capability_id="plugin:test.echo",
            display_name="Echo",
            kind="tool",
            source=CapabilitySource.PLUGIN,
            owner_id="test_plugin",
            status=CapabilityStatus.ENABLED,
            tool_names=["echo_tool"],
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="plugin:test.status",
            display_name="Status",
            kind="tool",
            source=CapabilitySource.PLUGIN,
            owner_id="test_plugin",
            status=CapabilityStatus.DISABLED,
            tool_names=["status_tool"],
            metadata={"version": "1.0.0"},
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="plugin:test.validate",
            display_name="Validate",
            kind="tool",
            source=CapabilitySource.PLUGIN,
            owner_id="test_plugin",
            status=CapabilityStatus.FAILED_VALIDATION,
            tool_names=["validate_tool"],
            manifest_path="/tmp/test_plugin/plugin.json",
        )
    )

    extensions = build_extension_descriptors(registry)

    assert len(extensions) == 1
    assert extensions[0].extension_id == "test_plugin"
    assert extensions[0].extension_kind == "plugin"
    assert sorted(extensions[0].capability_ids) == [
        "plugin:test.echo",
        "plugin:test.status",
        "plugin:test.validate",
    ]
    assert extensions[0].display_name == "test_plugin"
    assert extensions[0].version == "1.0.0"
    assert extensions[0].manifest_path == "/tmp/test_plugin/plugin.json"
    assert extensions[0].status is CapabilityStatus.FAILED_VALIDATION


def test_build_extension_descriptors_prefers_failed_validation_over_disabled() -> None:
    registry = CapabilityRegistry()
    registry.register(
        CapabilityDescriptor(
            capability_id="plugin:test.disabled",
            display_name="Disabled",
            kind="tool",
            source=CapabilitySource.PLUGIN,
            owner_id="another_plugin",
            status=CapabilityStatus.DISABLED,
            tool_names=["disabled_tool"],
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="plugin:test.failed",
            display_name="Failed",
            kind="tool",
            source=CapabilitySource.PLUGIN,
            owner_id="another_plugin",
            status=CapabilityStatus.FAILED_VALIDATION,
            tool_names=["failed_tool"],
        )
    )

    extensions = build_extension_descriptors(registry)

    assert len(extensions) == 1
    assert extensions[0].status is CapabilityStatus.FAILED_VALIDATION


def test_build_extension_descriptors_exposes_manifest_contract_summary() -> None:
    registry = CapabilityRegistry()
    registry.register(
        CapabilityDescriptor(
            capability_id="plugin:test.configured",
            display_name="Configured",
            kind="tool",
            source=CapabilitySource.PLUGIN,
            owner_id="configured_plugin",
            status=CapabilityStatus.ENABLED,
            tool_names=["configured_tool"],
            manifest_path="/tmp/configured/plugin.json",
            metadata={
                "owner_name": "Configured Plugin",
                "manifest_version": 1,
                "version": "4.2.0",
                "plugin_config_schema": {
                    "type": "object",
                    "properties": {
                        "profile": {"type": "string"},
                        "mode": {"type": "string"},
                    },
                    "required": ["profile"],
                },
                "plugin_default_config": {"profile": "default"},
                "compatibility": {
                    "runtime_version": "0.1.0",
                    "compatible": True,
                    "constraints": {
                        "min_opencas_version": "0.1.0",
                        "max_opencas_version": None,
                    },
                    "reasons": [],
                },
            },
        )
    )

    extensions = build_extension_descriptors(registry)

    assert len(extensions) == 1
    extension = extensions[0]
    assert extension.extension_id == "configured_plugin"
    assert extension.manifest_version == 1
    assert extension.version == "4.2.0"
    assert extension.config_schema_summary == {
        "type": "object",
        "property_count": 2,
        "properties": ["mode", "profile"],
        "required": ["profile"],
        "has_default_config": True,
        "default_config_keys": ["profile"],
    }
    assert extension.compatibility == {
        "runtime_version": "0.1.0",
        "compatible": True,
        "constraints": {
            "min_opencas_version": "0.1.0",
            "max_opencas_version": None,
        },
        "reasons": [],
    }
