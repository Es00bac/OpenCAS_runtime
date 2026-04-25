"""Tests for the read-only platform inventory API routes."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opencas.api.server import create_app
from opencas.api.routes.platform import build_platform_router
from opencas.governance import PluginTrustLevel
from opencas.platform import CapabilityDescriptor, CapabilityRegistry, CapabilitySource, CapabilityStatus
from opencas.plugins import build_plugin_bundle


def _make_runtime():
    registry = CapabilityRegistry()
    registry.register(
        CapabilityDescriptor(
            capability_id="core:bash.run",
            display_name="Run Shell Command",
            kind="tool",
            source=CapabilitySource.CORE,
            owner_id="core",
            status=CapabilityStatus.ENABLED,
            description="Run a shell command.",
            tool_names=["bash_run_command"],
            config_schema={"type": "object"},
            metadata={"risk_tier": "read_only"},
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="plugin:test.echo",
            display_name="Echo",
            kind="tool",
            source=CapabilitySource.PLUGIN,
            owner_id="test_plugin",
            status=CapabilityStatus.DISABLED,
            tool_names=["echo_tool"],
            manifest_path="/tmp/test_plugin/plugin.json",
            metadata={
                "owner_name": "Test Plugin",
                "manifest_version": 1,
                "version": "1.2.3",
                "distribution": {
                    "publisher": "OpenCAS Labs",
                    "channel": "stable",
                    "source_url": "https://example.com/test-plugin",
                    "changelog_url": "https://example.com/test-plugin/changelog",
                },
                "release_notes": "Stable release with platform compatibility metadata.",
                "compatibility": {
                    "runtime_version": "0.1.0",
                    "compatible": True,
                    "constraints": {
                        "min_opencas_version": "0.1.0",
                        "max_opencas_version": None,
                    },
                    "reasons": [],
                },
                "plugin_config_schema": {
                    "type": "object",
                    "properties": {
                        "profile": {"type": "string"},
                        "mode": {"type": "string"},
                    },
                    "required": ["profile"],
                },
                "plugin_default_config": {"profile": "safe"},
            },
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="mcp:filesystem.read",
            display_name="Filesystem Read",
            kind="tool",
            source=CapabilitySource.MCP,
            owner_id="mcp:filesystem",
            status=CapabilityStatus.FAILED_VALIDATION,
            tool_names=["mcp__filesystem__read_file"],
            validation_errors=["missing transport config"],
            metadata={"owner_name": "Filesystem"},
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="core:fs.read",
            display_name="Read File",
            kind="file",
            source=CapabilitySource.CORE,
            owner_id="core",
            status=CapabilityStatus.ENABLED,
            description="Read a file from disk.",
            tool_names=["fs_read_file"],
            metadata={"risk_tier": "read_only"},
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="plugin:test/path.echo",
            display_name="Path Echo",
            kind="tool",
            source=CapabilitySource.PLUGIN,
            owner_id="test_plugin/nested",
            status=CapabilityStatus.ENABLED,
            tool_names=["echo_path_tool"],
            metadata={"owner_name": "Nested Plugin", "version": "9.9.9"},
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="plugin:group.read",
            display_name="Group Read",
            kind="tool",
            source=CapabilitySource.PLUGIN,
            owner_id="grouped/plugin",
            status=CapabilityStatus.ENABLED,
            tool_names=["group_read_tool"],
            declared_dependencies=["alpha-lib"],
            metadata={"owner_name": "Grouped Plugin", "version": "2.0.0"},
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="plugin:group.write",
            display_name="Group Write",
            kind="tool",
            source=CapabilitySource.PLUGIN,
            owner_id="grouped/plugin",
            status=CapabilityStatus.FAILED_VALIDATION,
            tool_names=["group_write_tool"],
            declared_dependencies=["beta-lib", "alpha-lib"],
            validation_errors=["missing beta-lib"],
            metadata={"owner_name": "Grouped Plugin"},
        )
    )

    class _FakePluginStore:
        async def list_installed(self):
            return [
                {
                    "plugin_id": "test_plugin",
                    "bundle_metadata": {
                        "filename": "test_plugin-1.2.3.opencas-plugin.zip",
                        "sha256": "b" * 64,
                        "size_bytes": 2048,
                        "member_count": 3,
                    },
                }
            ]

    class _FakePluginTrust:
        def assess(self, *, provenance, bundle):
            publisher = (provenance or {}).get("publisher")
            checksum = (bundle or {}).get("sha256")
            level = PluginTrustLevel.GRAY
            blocked = False
            reasons = ["bundle has publisher metadata and checksum but no explicit trust policy"]
            certainty = 0.4
            matched = []
            if checksum == "b" * 64:
                level = PluginTrustLevel.TRUSTED
                certainty = 1.0
                matched = [f"checksum:{'b' * 64}"]
                reasons = ["checksum policy=trusted"]
            elif publisher == "OpenCAS Labs":
                level = PluginTrustLevel.USER_APPROVED
                certainty = 0.92
                matched = ["publisher:OpenCAS Labs"]
                reasons = ["publisher policy=user_approved"]
            return SimpleNamespace(
                level=level,
                certainty=certainty,
                blocked=blocked,
                publisher=(publisher or "").lower() or None,
                checksum=checksum,
                matched_policies=matched,
                reasons=reasons,
            )

    return SimpleNamespace(
        ctx=SimpleNamespace(
            config=SimpleNamespace(state_dir="/tmp/opencas-test-state"),
            event_bus=SimpleNamespace(subscribe=lambda *args, **kwargs: None),
            readiness=SimpleNamespace(snapshot=lambda: {"state": "ready"}),
            plugin_store=_FakePluginStore(),
            plugin_trust=_FakePluginTrust(),
        ),
        capability_registry=registry,
    )


def _make_test_app(runtime):
    app = FastAPI()
    app.include_router(build_platform_router(runtime))
    return app


def test_platform_capabilities_and_extensions_are_exposed_via_router() -> None:
    runtime = _make_runtime()
    client = TestClient(_make_test_app(runtime))

    capabilities = client.get("/api/platform/capabilities")
    assert capabilities.status_code == 200
    capability_payload = capabilities.json()
    assert [item["capability_id"] for item in capability_payload["capabilities"]] == [
        "core:bash.run",
        "core:fs.read",
        "mcp:filesystem.read",
        "plugin:group.read",
        "plugin:group.write",
        "plugin:test.echo",
        "plugin:test/path.echo",
    ]
    assert capability_payload["capabilities"][0]["source"] == "core"
    assert capability_payload["capabilities"][1]["kind"] == "file"
    assert capability_payload["capabilities"][2]["status"] == "failed_validation"
    assert capability_payload["capabilities"][3]["capability_id"] == "plugin:group.read"
    assert capability_payload["capabilities"][3]["metadata"]["owner_name"] == "Grouped Plugin"
    assert capability_payload["capabilities"][4]["capability_id"] == "plugin:group.write"
    assert capability_payload["capabilities"][5]["capability_id"] == "plugin:test.echo"
    assert capability_payload["capabilities"][6]["capability_id"] == "plugin:test/path.echo"

    filtered = client.get("/api/platform/capabilities", params={"source": "plugin", "owner_id": "test_plugin"})
    assert filtered.status_code == 200
    filtered_payload = filtered.json()
    assert [item["capability_id"] for item in filtered_payload["capabilities"]] == ["plugin:test.echo"]

    filtered = client.get("/api/platform/capabilities", params={"status": "enabled", "kind": "tool"})
    assert filtered.status_code == 200
    filtered_payload = filtered.json()
    assert [item["capability_id"] for item in filtered_payload["capabilities"]] == [
        "core:bash.run",
        "plugin:group.read",
        "plugin:test/path.echo",
    ]

    capability_detail = client.get("/api/platform/capabilities/plugin:test/path.echo")
    assert capability_detail.status_code == 200
    detail_payload = capability_detail.json()
    assert detail_payload["capability"]["capability_id"] == "plugin:test/path.echo"
    assert detail_payload["capability"]["source"] == "plugin"
    assert detail_payload["capability"]["status"] == "enabled"
    assert detail_payload["capability"]["manifest_path"] is None

    grouped = client.get("/api/platform/capabilities", params={"owner_id": "grouped/plugin"})
    assert grouped.status_code == 200
    grouped_payload = grouped.json()
    assert [item["capability_id"] for item in grouped_payload["capabilities"]] == [
        "plugin:group.read",
        "plugin:group.write",
    ]

    extensions = client.get("/api/platform/extensions")
    assert extensions.status_code == 200
    extension_payload = extensions.json()
    assert [item["extension_id"] for item in extension_payload["extensions"]] == [
        "core",
        "grouped/plugin",
        "mcp:filesystem",
        "test_plugin",
        "test_plugin/nested",
    ]
    assert extension_payload["extensions"][0]["extension_kind"] == "core_bundle"
    assert extension_payload["extensions"][1]["status"] == "failed_validation"
    assert extension_payload["extensions"][1]["dependencies"] == ["alpha-lib", "beta-lib"]
    assert extension_payload["extensions"][1]["errors"] == ["missing beta-lib"]
    assert extension_payload["extensions"][2]["display_name"] == "Filesystem"
    assert extension_payload["extensions"][3]["display_name"] == "Test Plugin"
    assert extension_payload["extensions"][3]["manifest_version"] == 1
    assert extension_payload["extensions"][3]["compatibility"] == {
        "runtime_version": "0.1.0",
        "compatible": True,
        "constraints": {
            "min_opencas_version": "0.1.0",
            "max_opencas_version": None,
        },
        "reasons": [],
    }
    assert extension_payload["extensions"][3]["provenance"] == {
        "publisher": "OpenCAS Labs",
        "channel": "stable",
        "source_url": "https://example.com/test-plugin",
        "changelog_url": "https://example.com/test-plugin/changelog",
    }
    assert extension_payload["extensions"][3]["bundle"] == {
        "filename": "test_plugin-1.2.3.opencas-plugin.zip",
        "sha256": "b" * 64,
        "size_bytes": 2048,
        "member_count": 3,
    }
    assert extension_payload["extensions"][3]["trust"] == {
        "level": "trusted",
        "certainty": 1.0,
        "blocked": False,
        "publisher": "opencas labs",
        "checksum": "b" * 64,
        "signer_ids": [],
        "verified_signer_ids": [],
        "signature_count": 0,
        "verified_signature_count": 0,
        "matched_policies": [f"checksum:{'b' * 64}"],
        "reasons": ["checksum policy=trusted"],
    }
    assert extension_payload["extensions"][3]["release_notes"] == (
        "Stable release with platform compatibility metadata."
    )
    assert extension_payload["extensions"][3]["config_schema_summary"] == {
        "type": "object",
        "property_count": 2,
        "properties": ["mode", "profile"],
        "required": ["profile"],
        "has_default_config": True,
        "default_config_keys": ["profile"],
    }
    assert extension_payload["extensions"][4]["display_name"] == "Nested Plugin"

    extension_detail = client.get("/api/platform/extensions/grouped/plugin")
    assert extension_detail.status_code == 200
    extension_detail_payload = extension_detail.json()
    assert extension_detail_payload["extension"]["extension_id"] == "grouped/plugin"
    assert extension_detail_payload["extension"]["capability_ids"] == [
        "plugin:group.read",
        "plugin:group.write",
    ]
    assert extension_detail_payload["extension"]["status"] == "failed_validation"
    assert extension_detail_payload["extension"]["version"] == "2.0.0"
    assert extension_detail_payload["extension"]["dependencies"] == ["alpha-lib", "beta-lib"]
    assert extension_detail_payload["extension"]["errors"] == ["missing beta-lib"]
    assert [item["capability_id"] for item in extension_detail_payload["capabilities"]] == [
        "plugin:group.read",
        "plugin:group.write",
    ]
    assert extension_detail_payload["capabilities"][0]["source"] == "plugin"
    assert extension_detail_payload["capabilities"][1]["status"] == "failed_validation"
    assert extension_detail_payload["capabilities"][1]["validation_errors"] == ["missing beta-lib"]

    plugin_detail = client.get("/api/platform/extensions/test_plugin")
    assert plugin_detail.status_code == 200
    plugin_detail_payload = plugin_detail.json()
    assert plugin_detail_payload["extension"]["manifest_version"] == 1
    assert plugin_detail_payload["extension"]["compatibility"] == {
        "runtime_version": "0.1.0",
        "compatible": True,
        "constraints": {
            "min_opencas_version": "0.1.0",
            "max_opencas_version": None,
        },
        "reasons": [],
    }
    assert plugin_detail_payload["extension"]["provenance"] == {
        "publisher": "OpenCAS Labs",
        "channel": "stable",
        "source_url": "https://example.com/test-plugin",
        "changelog_url": "https://example.com/test-plugin/changelog",
    }
    assert plugin_detail_payload["extension"]["bundle"] == {
        "filename": "test_plugin-1.2.3.opencas-plugin.zip",
        "sha256": "b" * 64,
        "size_bytes": 2048,
        "member_count": 3,
    }
    assert plugin_detail_payload["extension"]["trust"] == {
        "level": "trusted",
        "certainty": 1.0,
        "blocked": False,
        "publisher": "opencas labs",
        "checksum": "b" * 64,
        "signer_ids": [],
        "verified_signer_ids": [],
        "signature_count": 0,
        "verified_signature_count": 0,
        "matched_policies": [f"checksum:{'b' * 64}"],
        "reasons": ["checksum policy=trusted"],
    }
    assert plugin_detail_payload["extension"]["release_notes"] == (
        "Stable release with platform compatibility metadata."
    )
    assert plugin_detail_payload["extension"]["config_schema_summary"] == {
        "type": "object",
        "property_count": 2,
        "properties": ["mode", "profile"],
        "required": ["profile"],
        "has_default_config": True,
        "default_config_keys": ["profile"],
    }


def test_platform_routes_return_404_for_unknown_inventory_entries() -> None:
    runtime = _make_runtime()
    client = TestClient(_make_test_app(runtime))

    assert client.get("/api/platform/capabilities/unknown").status_code == 404
    assert client.get("/api/platform/extensions/unknown").status_code == 404


def test_platform_install_update_policy_is_exposed_via_router() -> None:
    runtime = _make_runtime()
    client = TestClient(_make_test_app(runtime))

    response = client.get("/api/platform/policies/install-update")

    assert response.status_code == 200
    body = response.json()
    assert body["runtime_version"] == "0.1.0"
    assert body["bundle_suffix"] == ".opencas-plugin.zip"
    assert body["supported_manifest_versions"] == [1]
    assert [item["rule_id"] for item in body["change_types"]] == [
        "install",
        "upgrade",
        "downgrade",
        "reinstall",
        "replace",
    ]
    assert body["change_types"][1]["requires_existing"] is True
    assert body["compatibility_fields"] == [
        {
            "name": "min_opencas_version",
            "description": "Optional lower runtime bound declared by the plugin manifest. The bundle is blocked if the running OpenCAS version is below this value.",
        },
        {
            "name": "max_opencas_version",
            "description": "Optional upper runtime bound declared by the plugin manifest. The bundle is blocked if the running OpenCAS version is above this value.",
        },
        {
            "name": "reasons",
            "description": "Human-readable incompatibility reasons returned by the runtime when the bundle cannot be installed or updated safely.",
        },
    ]
    assert [item["name"] for item in body["provenance_fields"]] == [
        "publisher",
        "channel",
        "source_url",
        "changelog_url",
        "signatures",
    ]
    assert [item["name"] for item in body["bundle_fields"]] == [
        "filename",
        "sha256",
        "size_bytes",
        "member_count",
        "signatures",
    ]
    assert [item["name"] for item in body["lifecycle_actions"]] == [
        "enable",
        "disable",
        "uninstall",
    ]
    assert body["evaluation_rules"] == [
        "Bundle inspection always runs before install or update, and the extension id in the manifest is authoritative.",
        "Update operations require the uploaded bundle extension id to match the targeted installed extension id exactly.",
        "Compatibility is evaluated against the running OpenCAS version before any install or update mutation occurs.",
        "Incompatible bundles are rejected with a blocking 409 response and are not passed to the plugin install backend.",
        "The change type is derived from the existing installed version for the same extension id plus the uploaded bundle version.",
        "When signatures are present, bundle inspection verifies each signature against the canonical payload built from manifest metadata and hashed archive members.",
        "Signer trust policies can block or approve a bundle only when the signer key id is verified successfully against the uploaded bundle.",
    ]
    assert body["operator_notes"] == [
        "Install and update share the same inspection pipeline. The action button changes, but the runtime still computes the actual change type from extension id and version data.",
        "Publisher and source metadata still help with human review, but cryptographic verification now comes from optional Ed25519 signer entries plus operator-managed signer trust policies.",
        "A verified signature without an explicit signer trust policy improves operator evidence, but it does not automatically make the bundle trusted.",
        "Use the bundle checksum, verified signer ids, publisher, source URL, and release notes together when deciding whether a replacement or downgrade is acceptable.",
    ]


def test_platform_routes_return_503_when_registry_is_missing() -> None:
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            config=SimpleNamespace(state_dir="/tmp/opencas-test-state"),
            event_bus=SimpleNamespace(subscribe=lambda *args, **kwargs: None),
            readiness=SimpleNamespace(snapshot=lambda: {"state": "ready"}),
        )
    )
    client = TestClient(_make_test_app(runtime))

    assert client.get("/api/platform/capabilities").status_code == 503
    assert client.get("/api/platform/capabilities/core:bash.run").status_code == 503
    assert client.get("/api/platform/extensions").status_code == 503
    assert client.get("/api/platform/extensions/core").status_code == 503


def test_platform_routes_reject_invalid_enum_filters() -> None:
    runtime = _make_runtime()
    client = TestClient(_make_test_app(runtime))

    invalid_source = client.get("/api/platform/capabilities", params={"source": "bogus"})
    assert invalid_source.status_code == 422

    invalid_status = client.get("/api/platform/capabilities", params={"status": "bogus"})
    assert invalid_status.status_code == 422


def test_platform_extension_enable_calls_runtime_lifecycle() -> None:
    called = []

    async def _enable(plugin_id: str) -> None:
        called.append(("enable", plugin_id))

    runtime = SimpleNamespace(
        capability_registry=_make_runtime().capability_registry,
        enable_plugin=_enable,
        ctx=SimpleNamespace(
            config=SimpleNamespace(state_dir="/tmp/opencas-test-state"),
            event_bus=SimpleNamespace(subscribe=lambda *args, **kwargs: None),
            readiness=SimpleNamespace(snapshot=lambda: {"state": "ready"}),
        ),
    )
    client = TestClient(_make_test_app(runtime))

    response = client.post("/api/platform/extensions/test_plugin/enable")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "extension_id": "test_plugin",
        "action": "enable",
    }
    assert called == [("enable", "test_plugin")]


def test_platform_extension_disable_calls_runtime_lifecycle() -> None:
    called = []

    async def _disable(plugin_id: str) -> None:
        called.append(("disable", plugin_id))

    runtime = SimpleNamespace(
        capability_registry=_make_runtime().capability_registry,
        disable_plugin=_disable,
        ctx=SimpleNamespace(
            config=SimpleNamespace(state_dir="/tmp/opencas-test-state"),
            event_bus=SimpleNamespace(subscribe=lambda *args, **kwargs: None),
            readiness=SimpleNamespace(snapshot=lambda: {"state": "ready"}),
        ),
    )
    client = TestClient(_make_test_app(runtime))

    response = client.post("/api/platform/extensions/grouped/plugin/disable")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "extension_id": "grouped/plugin",
        "action": "disable",
    }
    assert called == [("disable", "grouped/plugin")]


def test_platform_extension_uninstall_calls_runtime_lifecycle() -> None:
    called = []

    async def _uninstall(plugin_id: str) -> None:
        called.append(("uninstall", plugin_id))

    runtime = SimpleNamespace(
        capability_registry=_make_runtime().capability_registry,
        uninstall_plugin=_uninstall,
        ctx=SimpleNamespace(
            config=SimpleNamespace(state_dir="/tmp/opencas-test-state"),
            event_bus=SimpleNamespace(subscribe=lambda *args, **kwargs: None),
            readiness=SimpleNamespace(snapshot=lambda: {"state": "ready"}),
        ),
    )
    client = TestClient(_make_test_app(runtime))

    response = client.delete("/api/platform/extensions/test_plugin")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "extension_id": "test_plugin",
        "action": "uninstall",
    }
    assert called == [("uninstall", "test_plugin")]


def test_platform_extension_enable_returns_404_for_unknown_extension() -> None:
    runtime = _make_runtime()
    client = TestClient(_make_test_app(runtime))

    response = client.post("/api/platform/extensions/does-not-exist/enable")

    assert response.status_code == 404
    assert response.json()["detail"] == "Extension not found: does-not-exist"


def test_platform_install_extension_bundle_calls_runtime_install(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "upload_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "upload_plugin",
                "name": "Upload Plugin",
                "description": "",
                "manifest_version": 1,
                "version": "1.0.0",
                "distribution": {
                    "publisher": "OpenCAS Labs",
                    "channel": "stable",
                    "source_url": "https://example.com/upload-plugin",
                },
                "release_notes": "Initial packaged release.",
            }
        )
    )
    bundle_path = build_plugin_bundle(plugin_dir)
    calls = []

    async def _install(path: str) -> object:
        calls.append(path)
        return SimpleNamespace(plugin_id="upload_plugin", manifest_version=1, version="1.0.0")

    runtime = _make_runtime()
    runtime.install_plugin = _install
    client = TestClient(_make_test_app(runtime))

    response = client.post(
        "/api/platform/extensions/install",
        json={"path": str(bundle_path)},
    )

    assert response.status_code == 200
    assert response.json()["action"] == "install"
    assert response.json()["extension_id"] == "upload_plugin"
    assert response.json()["manifest_version"] == 1
    assert response.json()["version"] == "1.0.0"
    assert response.json()["previous_version"] is None
    assert response.json()["change_type"] == "install"
    assert response.json()["compatibility"] == {
        "runtime_version": "0.1.0",
        "compatible": True,
        "constraints": {
            "min_opencas_version": None,
            "max_opencas_version": None,
        },
        "reasons": [],
    }
    assert response.json()["provenance"] == {
        "publisher": "OpenCAS Labs",
        "channel": "stable",
        "source_url": "https://example.com/upload-plugin",
    }
    assert response.json()["trust"] == {
        "level": "user_approved",
        "certainty": 0.92,
        "blocked": False,
        "publisher": "opencas labs",
        "checksum": response.json()["bundle"]["sha256"],
        "signer_ids": [],
        "verified_signer_ids": [],
        "signature_count": 0,
        "verified_signature_count": 0,
        "matched_policies": ["publisher:OpenCAS Labs"],
        "reasons": ["publisher policy=user_approved"],
    }
    assert response.json()["release_notes"] == "Initial packaged release."
    assert response.json()["bundle"]["filename"] == bundle_path.name
    assert response.json()["bundle"]["size_bytes"] == bundle_path.stat().st_size
    assert response.json()["bundle"]["member_count"] == 1
    assert response.json()["bundle"]["signatures"] == {
        "present": False,
        "count": 0,
        "verified_count": 0,
        "entries": [],
    }
    assert len(response.json()["bundle"]["sha256"]) == 64
    assert response.json()["stored_bundle"] == str(bundle_path)
    assert calls == [str(bundle_path)]


def test_platform_inspect_extension_bundle_reports_change_type(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "inspect_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "test_plugin",
                "name": "Test Plugin",
                "description": "",
                "manifest_version": 1,
                "version": "2.0.0",
                "distribution": {
                    "publisher": "Phase Two Labs",
                    "channel": "beta",
                    "source_url": "https://example.com/test-plugin/v2",
                },
                "release_notes": "Upgrade adds provenance previews for operators.",
            }
        )
    )
    bundle_path = build_plugin_bundle(plugin_dir)
    runtime = _make_runtime()
    client = TestClient(_make_test_app(runtime))

    response = client.post(
        "/api/platform/extensions/inspect-bundle",
        json={"path": str(bundle_path)},
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "extension_id": "test_plugin",
        "manifest_version": 1,
        "version": "2.0.0",
        "previous_version": "1.2.3",
        "change_type": "upgrade",
        "compatibility": {
            "runtime_version": "0.1.0",
            "compatible": True,
            "constraints": {
                "min_opencas_version": None,
                "max_opencas_version": None,
            },
            "reasons": [],
        },
        "provenance": {
            "publisher": "Phase Two Labs",
            "channel": "beta",
            "source_url": "https://example.com/test-plugin/v2",
        },
        "release_notes": "Upgrade adds provenance previews for operators.",
        "trust": {
            "level": "gray",
            "certainty": 0.4,
            "blocked": False,
            "publisher": "phase two labs",
            "checksum": body["bundle"]["sha256"],
            "signer_ids": [],
            "verified_signer_ids": [],
            "signature_count": 0,
            "verified_signature_count": 0,
            "matched_policies": [],
            "reasons": ["bundle has publisher metadata and checksum but no explicit trust policy"],
        },
        "bundle": {
            "filename": bundle_path.name,
            "sha256": body["bundle"]["sha256"],
            "size_bytes": bundle_path.stat().st_size,
            "member_count": 1,
            "signatures": {
                "present": False,
                "count": 0,
                "verified_count": 0,
                "entries": [],
            },
        },
    }
    assert len(body["bundle"]["sha256"]) == 64


def test_platform_update_extension_bundle_rejects_bundle_id_mismatch(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "wrong_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "wrong_plugin",
                "name": "Wrong Plugin",
                "description": "",
                "manifest_version": 1,
                "version": "1.0.0",
            }
        )
    )
    bundle_path = build_plugin_bundle(plugin_dir)

    runtime = _make_runtime()

    async def _install(_: str) -> object:
        raise AssertionError("install_plugin should not be called on id mismatch")

    runtime.install_plugin = _install
    client = TestClient(_make_test_app(runtime))

    response = client.post(
        "/api/platform/extensions/test_plugin/update",
        json={"path": str(bundle_path)},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Bundle extension id wrong_plugin does not match expected test_plugin"
    )


def test_platform_install_extension_bundle_rejects_incompatible_runtime(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "incompatible_upload_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "incompatible_upload_plugin",
                "name": "Incompatible Upload Plugin",
                "description": "",
                "manifest_version": 1,
                "version": "1.0.0",
                "compatibility": {
                    "min_opencas_version": "9.0.0",
                },
            }
        )
    )
    bundle_path = build_plugin_bundle(plugin_dir)

    runtime = _make_runtime()

    async def _install(_: str) -> object:
        raise AssertionError("install_plugin should not be called for incompatible bundles")

    runtime.install_plugin = _install
    client = TestClient(_make_test_app(runtime))

    response = client.post(
        "/api/platform/extensions/install",
        json={"path": str(bundle_path)},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "requires OpenCAS >= 9.0.0, current runtime is 0.1.0"


def test_platform_install_extension_bundle_rejects_blocked_bundle_trust(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "blocked_upload_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "blocked_upload_plugin",
                "name": "Blocked Upload Plugin",
                "description": "",
                "manifest_version": 1,
                "version": "1.0.0",
                "distribution": {
                    "publisher": "Blocked Labs",
                },
            }
        )
    )
    bundle_path = build_plugin_bundle(plugin_dir)

    runtime = _make_runtime()

    class _BlockedPluginTrust:
        def assess(self, *, provenance, bundle):
            return SimpleNamespace(
                level=PluginTrustLevel.BLOCKED,
                certainty=1.0,
                blocked=True,
                publisher=(provenance or {}).get("publisher"),
                checksum=(bundle or {}).get("sha256"),
                matched_policies=["publisher:Blocked Labs"],
                reasons=["publisher policy=blocked"],
            )

    runtime.ctx.plugin_trust = _BlockedPluginTrust()

    async def _install(_: str) -> object:
        raise AssertionError("install_plugin should not be called for blocked bundles")

    runtime.install_plugin = _install
    client = TestClient(_make_test_app(runtime))

    response = client.post(
        "/api/platform/extensions/install",
        json={"path": str(bundle_path)},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "publisher policy=blocked"


def test_platform_extension_enable_rejects_non_plugin_extensions() -> None:
    runtime = _make_runtime()
    client = TestClient(_make_test_app(runtime))

    response = client.post("/api/platform/extensions/core/enable")

    assert response.status_code == 409
    assert response.json()["detail"] == "Extension does not support lifecycle actions: core"


def test_platform_extension_enable_returns_503_when_lifecycle_backend_missing() -> None:
    runtime = SimpleNamespace(
        capability_registry=_make_runtime().capability_registry,
        ctx=SimpleNamespace(
            config=SimpleNamespace(state_dir="/tmp/opencas-test-state"),
            event_bus=SimpleNamespace(subscribe=lambda *args, **kwargs: None),
            readiness=SimpleNamespace(snapshot=lambda: {"state": "ready"}),
        ),
    )
    client = TestClient(_make_test_app(runtime))

    response = client.post("/api/platform/extensions/test_plugin/enable")

    assert response.status_code == 503
    assert response.json()["detail"] == "Plugin lifecycle backend is not available"


def test_create_app_mounts_platform_router() -> None:
    runtime = _make_runtime()
    from unittest.mock import patch

    from fastapi import APIRouter

    with patch("opencas.api.routes.chat.build_chat_router", return_value=APIRouter()):
        app = create_app(runtime)
    capability_route = next(
        route for route in app.routes if getattr(route, "path", None) == "/api/platform/capabilities/{capability_id:path}"
    )
    extension_route = next(
        route for route in app.routes if getattr(route, "path", None) == "/api/platform/extensions/{extension_id:path}"
    )
    policy_route = next(
        route for route in app.routes if getattr(route, "path", None) == "/api/platform/policies/install-update"
    )
    assert any(getattr(route, "path", "").startswith("/api/platform/capabilities") for route in app.routes)
    assert any(getattr(route, "path", "").startswith("/api/platform/extensions") for route in app.routes)
    assert any(getattr(route, "path", "").startswith("/api/platform/policies") for route in app.routes)
    assert ".*" in capability_route.path_regex.pattern
    assert ".*" in extension_route.path_regex.pattern
    assert policy_route.methods == {"GET"}
