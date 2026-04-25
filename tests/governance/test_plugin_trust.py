"""Tests for persisted plugin bundle trust policies."""

from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from opencas.governance import (
    PluginTrustLevel,
    PluginTrustScope,
    PluginTrustService,
    PluginTrustStore,
    build_plugin_trust_feed_signature_payload,
    normalize_plugin_publisher,
)


def _signed_feed(
    *,
    source_id: str,
    private_key: ed25519.Ed25519PrivateKey,
    public_key_b64: str,
    policies: list[dict],
    key_id: str = "opencas-feed-root",
) -> dict:
    canonical_policies = []
    for raw in policies:
        entry = dict(raw)
        if str(entry.get("scope")) == "publisher":
            entry["value"] = normalize_plugin_publisher(str(entry.get("value") or "")) or ""
        metadata = dict(entry.get("metadata") or {})
        publisher = metadata.get("publisher")
        if isinstance(publisher, str):
            normalized_publisher = normalize_plugin_publisher(publisher)
            if normalized_publisher is not None:
                metadata["publisher"] = normalized_publisher
        entry["metadata"] = metadata
        canonical_policies.append(entry)
    payload = build_plugin_trust_feed_signature_payload(
        format_version=1,
        source_id=source_id,
        policies=canonical_policies,
    )
    signature = private_key.sign(payload)
    return {
        "format_version": 1,
        "source_id": source_id,
        "policies": policies,
        "signatures": [
            {
                "key_id": key_id,
                "algorithm": "ed25519",
                "public_key": public_key_b64,
                "signature": base64.b64encode(signature).decode("ascii"),
            }
        ],
    }


@pytest.mark.asyncio
async def test_plugin_trust_assesses_gray_without_explicit_policy(tmp_path) -> None:
    service = await PluginTrustService(PluginTrustStore(tmp_path / "plugin_trust.db")).connect()

    assessment = service.assess(
        provenance={"publisher": "OpenCAS Labs", "channel": "stable"},
        bundle={"sha256": "a" * 64},
    )

    assert assessment.level == PluginTrustLevel.GRAY
    assert assessment.blocked is False
    assert assessment.publisher == "opencas labs"
    assert assessment.checksum == "a" * 64
    assert assessment.certainty == 0.4

    await service.close()


@pytest.mark.asyncio
async def test_plugin_trust_checksum_policy_overrides_publisher_policy(tmp_path) -> None:
    service = await PluginTrustService(PluginTrustStore(tmp_path / "plugin_trust.db")).connect()
    await service.set_policy(
        PluginTrustScope.PUBLISHER,
        "OpenCAS Labs",
        PluginTrustLevel.TRUSTED,
        note="Trusted publisher",
    )
    await service.set_policy(
        PluginTrustScope.CHECKSUM,
        "b" * 64,
        PluginTrustLevel.BLOCKED,
        note="Pinned bad bundle",
    )

    assessment = service.assess(
        provenance={"publisher": "OpenCAS Labs"},
        bundle={"sha256": "b" * 64},
    )

    assert assessment.level == PluginTrustLevel.BLOCKED
    assert assessment.blocked is True
    assert assessment.matched_policies == [f"checksum:{'b' * 64}"]

    snapshot = service.snapshot(limit=10)
    assert snapshot["policy_count"] == 2
    assert snapshot["publisher_policy_count"] == 1
    assert snapshot["checksum_policy_count"] == 1

    await service.close()


@pytest.mark.asyncio
async def test_plugin_trust_verified_signer_policy_upgrades_bundle(tmp_path) -> None:
    service = await PluginTrustService(PluginTrustStore(tmp_path / "plugin_trust.db")).connect()
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_key_b64 = __import__("base64").b64encode(public_key).decode("ascii")

    await service.set_policy(
        PluginTrustScope.SIGNER,
        "opencas-labs-main",
        PluginTrustLevel.TRUSTED,
        metadata={"public_key": public_key_b64, "publisher": "OpenCAS Labs"},
    )

    assessment = service.assess(
        provenance={"publisher": "OpenCAS Labs"},
        bundle={
            "sha256": "c" * 64,
            "signatures": {
                "entries": [
                    {
                        "key_id": "opencas-labs-main",
                        "verified": True,
                        "public_key": public_key_b64,
                    }
                ]
            },
        },
    )

    assert assessment.level == PluginTrustLevel.TRUSTED
    assert assessment.verified_signer_ids == ["opencas-labs-main"]
    assert assessment.matched_policies == ["signer:opencas-labs-main"]

    await service.close()


@pytest.mark.asyncio
async def test_plugin_trust_blocks_signer_public_key_mismatch(tmp_path) -> None:
    service = await PluginTrustService(PluginTrustStore(tmp_path / "plugin_trust.db")).connect()
    trusted_key = ed25519.Ed25519PrivateKey.generate().public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    other_key = ed25519.Ed25519PrivateKey.generate().public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trusted_b64 = __import__("base64").b64encode(trusted_key).decode("ascii")
    other_b64 = __import__("base64").b64encode(other_key).decode("ascii")

    await service.set_policy(
        PluginTrustScope.SIGNER,
        "opencas-labs-main",
        PluginTrustLevel.TRUSTED,
        metadata={"public_key": trusted_b64},
    )

    assessment = service.assess(
        provenance={"publisher": "OpenCAS Labs"},
        bundle={
            "signatures": {
                "entries": [
                    {
                        "key_id": "opencas-labs-main",
                        "verified": True,
                        "public_key": other_b64,
                    }
                ]
            },
        },
    )

    assert assessment.level == PluginTrustLevel.BLOCKED
    assert assessment.blocked is True
    assert "public key mismatch" in assessment.reasons[0]

    await service.close()


@pytest.mark.asyncio
async def test_plugin_trust_blocks_bundle_when_declared_signatures_do_not_verify(tmp_path) -> None:
    service = await PluginTrustService(PluginTrustStore(tmp_path / "plugin_trust.db")).connect()

    assessment = service.assess(
        provenance={"publisher": "OpenCAS Labs"},
        bundle={
            "signatures": {
                "entries": [
                    {
                        "key_id": "opencas-labs-main",
                        "verified": False,
                        "error": "signature verification failed",
                    }
                ]
            },
        },
    )

    assert assessment.level == PluginTrustLevel.BLOCKED
    assert assessment.blocked is True
    assert assessment.reasons == ["bundle declares signatures, but none verified successfully"]

    await service.close()


@pytest.mark.asyncio
async def test_plugin_trust_feed_sync_imports_and_prunes_feed_owned_policies(tmp_path) -> None:
    service = await PluginTrustService(PluginTrustStore(tmp_path / "plugin_trust.db")).connect()
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_key_b64 = base64.b64encode(public_key).decode("ascii")
    await service.set_policy(
        PluginTrustScope.SIGNER,
        "opencas-feed-root",
        PluginTrustLevel.TRUSTED,
        metadata={"public_key": public_key_b64, "source_id": "opencas-labs"},
    )

    first = await service.sync_feed(
        _signed_feed(
            source_id="opencas-labs",
            private_key=private_key,
            public_key_b64=public_key_b64,
            key_id="opencas-feed-root",
            policies=[
                {
                    "scope": "publisher",
                    "value": "OpenCAS Labs",
                    "level": "trusted",
                    "note": "",
                    "metadata": {},
                },
                {
                    "scope": "signer",
                    "value": "opencas-labs-main",
                    "level": "trusted",
                    "note": "",
                    "metadata": {
                        "public_key": public_key_b64,
                        "publisher": "OpenCAS Labs",
                        "source_id": "opencas-labs",
                    },
                },
            ],
        )
    )

    assert len(first.imported) == 2
    assert first.verification["verified_signature_count"] == 1
    assert first.verification["trusted_signer_ids"] == ["opencas-feed-root"]
    snapshot = service.snapshot(limit=10)
    assert snapshot["feed_policy_count"] == 2
    assert snapshot["feed_source_count"] == 1

    second = await service.sync_feed(
        _signed_feed(
            source_id="opencas-labs",
            private_key=private_key,
            public_key_b64=public_key_b64,
            key_id="opencas-feed-root",
            policies=[
                {
                    "scope": "signer",
                    "value": "opencas-labs-main",
                    "level": "trusted",
                    "note": "",
                    "metadata": {
                        "public_key": public_key_b64,
                        "publisher": "OpenCAS Labs",
                        "source_id": "opencas-labs",
                    },
                }
            ],
        )
    )

    assert second.removed == [{"scope": "publisher", "value": "opencas labs"}]
    snapshot_after = service.snapshot(limit=10)
    assert snapshot_after["policy_count"] == 2
    assert snapshot_after["publisher_policy_count"] == 0
    assert snapshot_after["signer_policy_count"] == 2

    await service.close()


@pytest.mark.asyncio
async def test_plugin_trust_feed_sync_preserves_manual_conflicts(tmp_path) -> None:
    service = await PluginTrustService(PluginTrustStore(tmp_path / "plugin_trust.db")).connect()
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_key_b64 = base64.b64encode(public_key).decode("ascii")
    await service.set_policy(
        PluginTrustScope.SIGNER,
        "opencas-feed-root",
        PluginTrustLevel.TRUSTED,
        metadata={"public_key": public_key_b64, "source_id": "opencas-labs"},
    )
    await service.set_policy(
        PluginTrustScope.PUBLISHER,
        "OpenCAS Labs",
        PluginTrustLevel.USER_APPROVED,
        source="dashboard",
        note="manual policy",
    )

    result = await service.sync_feed(
        _signed_feed(
            source_id="opencas-labs",
            private_key=private_key,
            public_key_b64=public_key_b64,
            key_id="opencas-feed-root",
            policies=[
                {
                    "scope": "publisher",
                    "value": "OpenCAS Labs",
                    "level": "trusted",
                    "note": "",
                    "metadata": {},
                }
            ],
        )
    )

    assert result.imported == []
    assert len(result.skipped_conflicts) == 1
    snapshot = service.snapshot(limit=10)
    assert snapshot["policy_count"] == 2
    assert snapshot["feed_policy_count"] == 0
    assert any(item["source"] == "dashboard" for item in snapshot["entries"])

    await service.close()


@pytest.mark.asyncio
async def test_plugin_trust_feed_sync_rejects_unsigned_feed(tmp_path) -> None:
    service = await PluginTrustService(PluginTrustStore(tmp_path / "plugin_trust.db")).connect()

    with pytest.raises(ValueError, match="must include at least one signature entry"):
        await service.sync_feed(
            {
                "format_version": 1,
                "source_id": "opencas-labs",
                "policies": [
                    {
                        "scope": "publisher",
                        "value": "OpenCAS Labs",
                        "level": "trusted",
                    }
                ],
            }
        )

    await service.close()


@pytest.mark.asyncio
async def test_plugin_trust_feed_sync_rejects_untrusted_signer(tmp_path) -> None:
    service = await PluginTrustService(PluginTrustStore(tmp_path / "plugin_trust.db")).connect()
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_key_b64 = base64.b64encode(public_key).decode("ascii")

    with pytest.raises(ValueError, match="not anchored to a trusted signer policy"):
        await service.sync_feed(
            _signed_feed(
                source_id="opencas-labs",
                private_key=private_key,
                public_key_b64=public_key_b64,
                policies=[
                    {
                        "scope": "publisher",
                        "value": "OpenCAS Labs",
                        "level": "trusted",
                        "note": "",
                        "metadata": {},
                    }
                ],
            )
        )

    await service.close()


@pytest.mark.asyncio
async def test_plugin_trust_feed_sync_rejects_signer_source_binding_mismatch(tmp_path) -> None:
    service = await PluginTrustService(PluginTrustStore(tmp_path / "plugin_trust.db")).connect()
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_key_b64 = base64.b64encode(public_key).decode("ascii")
    await service.set_policy(
        PluginTrustScope.SIGNER,
        "opencas-feed-root",
        PluginTrustLevel.TRUSTED,
        metadata={"public_key": public_key_b64, "source_id": "other-source"},
    )

    with pytest.raises(ValueError, match="not anchored to a trusted signer policy"):
        await service.sync_feed(
            _signed_feed(
                source_id="opencas-labs",
                private_key=private_key,
                public_key_b64=public_key_b64,
                key_id="opencas-feed-root",
                policies=[
                    {
                        "scope": "publisher",
                        "value": "OpenCAS Labs",
                        "level": "trusted",
                        "note": "",
                        "metadata": {},
                    }
                ],
            )
        )

    await service.close()
