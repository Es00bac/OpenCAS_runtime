"""Tests for schema-driven plugin authoring templates."""

from pathlib import Path

from opencas.plugins import (
    PluginTemplateSpec,
    build_plugin_bundle,
    inspect_plugin_bundle,
    load_plugin_manifest,
    scaffold_plugin_template,
)


def test_scaffold_plugin_template_generates_valid_tool_plugin(tmp_path: Path) -> None:
    target_dir = tmp_path / "template_tool"
    spec = PluginTemplateSpec(
        plugin_id="template_tool",
        name="Template Tool",
        description="A generated tool plugin.",
        template_kind="tool",
        config_schema={
            "type": "object",
            "properties": {
                "profile": {"type": "string"},
            },
            "required": ["profile"],
        },
        default_config={"profile": "default"},
        distribution={
            "publisher": "OpenCAS Labs",
            "channel": "stable",
            "source_url": "https://example.com/template-tool",
        },
        release_notes="Initial scaffold for the template tool plugin.",
        tool_names=["template_tool_run"],
    )

    manifest_path = scaffold_plugin_template(target_dir, spec)
    manifest = load_plugin_manifest(manifest_path)
    entrypoint = (target_dir / "main.py").read_text(encoding="utf-8")

    assert manifest["id"] == "template_tool"
    assert manifest["manifest_version"] == 1
    assert manifest["default_config"] == {"profile": "default"}
    assert manifest["distribution"]["publisher"] == "OpenCAS Labs"
    assert manifest["release_notes"] == "Initial scaffold for the template tool plugin."
    assert manifest["capabilities"][0]["tool_names"] == ["template_tool_run"]
    assert "def register_skills" in entrypoint
    assert "tools.register" in entrypoint
    assert (target_dir / "README.md").exists()


def test_scaffold_plugin_template_generates_valid_hook_plugin(tmp_path: Path) -> None:
    target_dir = tmp_path / "template_hook"
    spec = PluginTemplateSpec(
        plugin_id="template_hook",
        name="Template Hook",
        description="A generated hook plugin.",
        template_kind="hook",
    )

    manifest_path = scaffold_plugin_template(target_dir, spec)
    manifest = load_plugin_manifest(manifest_path)
    entrypoint = (target_dir / "main.py").read_text(encoding="utf-8")

    assert manifest["capabilities"][0]["tool_names"] == []
    assert "def register_hooks" in entrypoint
    assert "def register_skills" not in entrypoint


def test_scaffolded_plugin_template_packages_cleanly(tmp_path: Path) -> None:
    target_dir = tmp_path / "template_hybrid"
    spec = PluginTemplateSpec(
        plugin_id="template_hybrid",
        name="Template Hybrid",
        description="A generated hybrid plugin.",
        template_kind="hybrid",
    )

    scaffold_plugin_template(target_dir, spec)
    bundle_path = build_plugin_bundle(target_dir)
    manifest = inspect_plugin_bundle(bundle_path)

    assert manifest["id"] == "template_hybrid"
    assert manifest["manifest_version"] == 1
