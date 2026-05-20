from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.governance import ShadowRegistry, ShadowRegistryStore
from opencas.thread_registry.shadow_bridge import (
    record_shadow_registry_thread_beads,
)
from opencas.thread_registry import (
    BeadSourceKind,
    BeadStatus,
    ThreadRegistryService,
    ThreadRegistryStore,
)


@pytest.mark.asyncio
async def test_shadow_registry_mature_cluster_records_one_shadow_bead(tmp_path: Path):
    registry = ShadowRegistry(ShadowRegistryStore(tmp_path / "shadow_registry"))
    first = registry.capture_retry_blocked(
        {
            "objective": "Continue writing project 4246 from the existing manuscript.",
            "canonical_artifact_path": "workspace/writing/4246/story_4246.md",
            "attempt": 1,
            "reason": "RetryGovernor blocked a broad retry with no new evidence.",
        }
    )
    registry.capture_retry_blocked(
        {
            "objective": "Continue writing project 4246 from the existing manuscript.",
            "canonical_artifact_path": "workspace/writing/4246/story_4246.md",
            "attempt": 2,
            "reason": "RetryGovernor blocked another broad retry with no new evidence.",
        }
    )
    thread_store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    service = ThreadRegistryService(store=thread_store, workspace_root=tmp_path / "workspace")
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(shadow_registry=registry),
        thread_registry_service=service,
    )

    try:
        result = await record_shadow_registry_thread_beads(runtime, limit=10)

        assert result["available"] is True
        assert result["recorded"] == 1
        beads = await thread_store.list_beads(source_kind=BeadSourceKind.SHADOW_INTENTION)
        assert len(beads) == 1
        assert beads[0].source_ref == f"shadow_registry:{first.fingerprint}"
        assert beads[0].status == BeadStatus.PERIPHERAL
        assert beads[0].user_commissioned is False
        assert "retry_blocked" in beads[0].summary
        assert "story_4246.md" in beads[0].summary
    finally:
        await thread_store.close()


@pytest.mark.asyncio
async def test_shadow_registry_bridge_skips_singletons_and_dismissed_clusters(
    tmp_path: Path,
):
    registry = ShadowRegistry(ShadowRegistryStore(tmp_path / "shadow_registry"))
    dismissed = registry.capture_retry_blocked(
        {
            "objective": "Continue writing project 4246 from the existing manuscript.",
            "canonical_artifact_path": "workspace/writing/4246/story_4246.md",
            "attempt": 1,
            "reason": "RetryGovernor blocked a broad retry with no new evidence.",
        }
    )
    registry.capture_retry_blocked(
        {
            "objective": "Continue writing project 4246 from the existing manuscript.",
            "canonical_artifact_path": "workspace/writing/4246/story_4246.md",
            "attempt": 2,
            "reason": "RetryGovernor blocked another broad retry with no new evidence.",
        }
    )
    registry.triage_cluster(dismissed.fingerprint, dismissed=True)
    registry.capture_retry_blocked(
        {
            "objective": "Inspect a one-off dashboard failure.",
            "canonical_artifact_path": "workspace/dashboard.md",
            "attempt": 1,
            "reason": "RetryGovernor blocked a one-off retry.",
        }
    )
    thread_store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    service = ThreadRegistryService(store=thread_store, workspace_root=tmp_path / "workspace")
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(shadow_registry=registry),
        thread_registry_service=service,
    )

    try:
        result = await record_shadow_registry_thread_beads(runtime, limit=10)

        assert result["recorded"] == 0
        beads = await thread_store.list_beads(source_kind=BeadSourceKind.SHADOW_INTENTION)
        assert beads == []
    finally:
        await thread_store.close()


@pytest.mark.asyncio
async def test_shadow_registry_thread_bridge_is_idempotent(tmp_path: Path):
    registry = ShadowRegistry(ShadowRegistryStore(tmp_path / "shadow_registry"))
    registry.capture_retry_blocked(
        {
            "objective": "Continue writing project 4246 from the existing manuscript.",
            "canonical_artifact_path": "workspace/writing/4246/story_4246.md",
            "attempt": 1,
            "reason": "RetryGovernor blocked a broad retry with no new evidence.",
        }
    )
    registry.capture_retry_blocked(
        {
            "objective": "Continue writing project 4246 from the existing manuscript.",
            "canonical_artifact_path": "workspace/writing/4246/story_4246.md",
            "attempt": 2,
            "reason": "RetryGovernor blocked another broad retry with no new evidence.",
        }
    )
    thread_store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    service = ThreadRegistryService(store=thread_store, workspace_root=tmp_path / "workspace")
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(shadow_registry=registry),
        thread_registry_service=service,
    )

    try:
        await record_shadow_registry_thread_beads(runtime, limit=10)
        await record_shadow_registry_thread_beads(runtime, limit=10)

        beads = await thread_store.list_beads(source_kind=BeadSourceKind.SHADOW_INTENTION)
        assert len(beads) == 1
    finally:
        await thread_store.close()
