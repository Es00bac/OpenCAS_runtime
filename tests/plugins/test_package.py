"""Tests for plugin bundle packaging helpers."""

from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from opencas.plugins import (
    PLUGIN_BUNDLE_SUFFIX,
    PluginPackageError,
    build_plugin_bundle,
    sign_plugin_directory,
    extract_plugin_bundle,
    inspect_plugin_bundle,
    inspect_plugin_bundle_details,
)


def test_build_and_extract_plugin_bundle_round_trip(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "bundle_source"
    plugin_dir.mkdir()
    (plugin_dir / "pkg").mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "bundle_source",
                "name": "Bundle Source",
                "description": "",
                "manifest_version": 1,
                "version": "1.2.3",
            }
        )
    )
    (plugin_dir / "pkg" / "main.py").write_text("VALUE = 1\n")

    bundle_path = build_plugin_bundle(plugin_dir)

    assert bundle_path.name == f"bundle_source-1.2.3{PLUGIN_BUNDLE_SUFFIX}"
    manifest_path, manifest = extract_plugin_bundle(bundle_path, tmp_path / "installed")

    assert manifest["id"] == "bundle_source"
    assert manifest_path == tmp_path / "installed" / "bundle_source" / "plugin.json"
    assert manifest_path.exists()
    assert (manifest_path.parent / "pkg" / "main.py").read_text() == "VALUE = 1\n"


def test_extract_plugin_bundle_rejects_path_traversal(tmp_path: Path) -> None:
    bundle_path = tmp_path / f"malicious{PLUGIN_BUNDLE_SUFFIX}"
    with __import__("zipfile").ZipFile(bundle_path, "w") as archive:
        archive.writestr("../escape.txt", "nope")
        archive.writestr(
            "plugin.json",
            __import__("json").dumps(
                {
                    "id": "malicious",
                    "name": "Malicious",
                    "description": "",
                }
            ),
        )

    with pytest.raises(PluginPackageError, match="escapes package root"):
        extract_plugin_bundle(bundle_path, tmp_path / "installed")


def test_inspect_plugin_bundle_validates_manifest_without_extracting(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "inspect_source"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "inspect_source",
                "name": "Inspect Source",
                "description": "",
                "manifest_version": 1,
                "version": "9.1.0",
            }
        )
    )
    bundle_path = build_plugin_bundle(plugin_dir)

    manifest = inspect_plugin_bundle(bundle_path)

    assert manifest["id"] == "inspect_source"
    assert manifest["manifest_version"] == 1
    assert manifest["version"] == "9.1.0"


def test_inspect_plugin_bundle_details_returns_bundle_metadata(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "inspect_details"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "inspect_details",
                "name": "Inspect Details",
                "description": "",
                "manifest_version": 1,
                "version": "1.0.0",
                "distribution": {
                    "publisher": "OpenCAS Labs",
                    "channel": "stable",
                    "source_url": "https://example.com/inspect-details",
                },
                "release_notes": "Adds platform provenance previews.",
            }
        )
    )
    bundle_path = build_plugin_bundle(plugin_dir)

    details = inspect_plugin_bundle_details(bundle_path)

    assert details["manifest"]["id"] == "inspect_details"
    assert details["manifest"]["distribution"]["publisher"] == "OpenCAS Labs"
    assert details["manifest"]["release_notes"] == "Adds platform provenance previews."
    assert details["bundle"]["filename"] == bundle_path.name
    assert details["bundle"]["size_bytes"] == bundle_path.stat().st_size
    assert details["bundle"]["member_count"] == 1
    assert len(details["bundle"]["sha256"]) == 64


def test_sign_plugin_directory_yields_verified_bundle_signature(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "signed_details"
    plugin_dir.mkdir()
    (plugin_dir / "pkg").mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "signed_details",
                "name": "Signed Details",
                "description": "",
                "manifest_version": 1,
                "version": "1.0.0",
                "distribution": {
                    "publisher": "OpenCAS Labs",
                },
            }
        )
    )
    (plugin_dir / "pkg" / "main.py").write_text("VALUE = 7\n")

    private_key = ed25519.Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    signature = sign_plugin_directory(
        plugin_dir,
        key_id="opencas-labs-main",
        private_key=private_pem,
    )
    bundle_path = build_plugin_bundle(plugin_dir)

    details = inspect_plugin_bundle_details(bundle_path)

    assert signature["key_id"] == "opencas-labs-main"
    assert details["bundle"]["signatures"]["present"] is True
    assert details["bundle"]["signatures"]["count"] == 1
    assert details["bundle"]["signatures"]["verified_count"] == 1
    assert details["bundle"]["signatures"]["entries"][0]["key_id"] == "opencas-labs-main"
    assert details["bundle"]["signatures"]["entries"][0]["verified"] is True
    assert details["bundle"]["signatures"]["entries"][0]["public_key_fingerprint"] == signature["public_key_fingerprint"]


def test_inspect_plugin_bundle_details_reports_invalid_signature_after_tamper(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "tampered_signature"
    plugin_dir.mkdir()
    (plugin_dir / "pkg").mkdir()
    (plugin_dir / "plugin.json").write_text(
        __import__("json").dumps(
            {
                "id": "tampered_signature",
                "name": "Tampered Signature",
                "description": "",
                "manifest_version": 1,
                "version": "1.0.0",
                "distribution": {
                    "publisher": "OpenCAS Labs",
                },
            }
        )
    )
    (plugin_dir / "pkg" / "main.py").write_text("VALUE = 1\n")

    private_key = ed25519.Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    sign_plugin_directory(
        plugin_dir,
        key_id="opencas-labs-main",
        private_key=private_pem,
    )
    (plugin_dir / "pkg" / "main.py").write_text("VALUE = 2\n")
    bundle_path = build_plugin_bundle(plugin_dir)

    details = inspect_plugin_bundle_details(bundle_path)

    assert details["bundle"]["signatures"]["verified_count"] == 0
    assert details["bundle"]["signatures"]["entries"][0]["verified"] is False
    assert "signature" in details["bundle"]["signatures"]["entries"][0]["error"].lower()
