"""Tests for the ShadowRegistry blocked-intention store."""

from pathlib import Path

from opencas.governance import (
    BlockReason,
    DecompositionStage,
    ShadowRegistry,
    ShadowRegistryStore,
)


def test_shadow_registry_persists_and_clusters_blocked_intentions(tmp_path: Path) -> None:
    store = ShadowRegistryStore(tmp_path / "shadow_registry")
    registry = ShadowRegistry(store=store)

    first = registry.capture(
        tool_name="bash_run_command",
        parameters={"command": "rm -rf /", "cwd": "/"},
        reason=BlockReason.SAFETY_BLOCKED,
        context="Command blocked by safety policy: blocked pattern: rm -rf /",
        session_id="session:default:abc",
        artifact="tool|default|bash_run_command",
    )
    second = registry.capture(
        tool_name="bash_run_command",
        parameters={"command": "rm -rf /", "cwd": "/"},
        reason=BlockReason.SAFETY_BLOCKED,
        context="Command blocked by safety policy: blocked pattern: rm -rf /",
        session_id="session:default:abc",
        artifact="tool|default|bash_run_command",
    )

    persisted = store.get(first.id)
    assert persisted is not None
    assert persisted.intent_summary == "shell:rm -rf /"
    assert persisted.decomposition_stage == DecompositionStage.RAW

    cluster = registry.get_cluster(first.id)
    assert {item.id for item in cluster} == {first.id, second.id}
    assert first.fingerprint == second.fingerprint

    reloaded = ShadowRegistry(store=ShadowRegistryStore(tmp_path / "shadow_registry"))
    recent = reloaded.list_recent(limit=10)
    assert [item.id for item in recent] == [second.id, first.id]


def test_shadow_registry_summary_and_planning_context(tmp_path: Path) -> None:
    store = ShadowRegistryStore(tmp_path / "shadow_registry")
    registry = ShadowRegistry(store=store)

    first = registry.capture(
        tool_name="repair_retry",
        parameters={
            "objective": "Continue Chronicle 4246 from the existing manuscript.",
            "canonical_artifact_path": "workspace/Chronicles/4246/chronicle_4246.md",
            "attempt": 2,
        },
        reason=BlockReason.RETRY_BLOCKED,
        context="RetryGovernor blocked a broad retry with no new evidence.",
        artifact="workspace/Chronicles/4246/chronicle_4246.md",
        target_kind="repair_task",
        target_id="task-4246",
        capture_source="repair_executor",
    )
    second = registry.capture(
        tool_name="repair_retry",
        parameters={
            "objective": "Continue Chronicle 4246 from the existing manuscript.",
            "canonical_artifact_path": "workspace/Chronicles/4246/chronicle_4246.md",
            "attempt": 3,
        },
        reason=BlockReason.RETRY_BLOCKED,
        context="RetryGovernor blocked a broad retry with no new evidence.",
        artifact="workspace/Chronicles/4246/chronicle_4246.md",
        target_kind="repair_task",
        target_id="task-4246",
        capture_source="repair_executor",
    )
    registry.capture(
        tool_name="bash_run_command",
        parameters={"command": "rm -rf /", "cwd": "/"},
        reason=BlockReason.SAFETY_BLOCKED,
        context="Command blocked by safety policy: blocked pattern: rm -rf /",
        session_id="session:default:abc",
        artifact="tool|default|bash_run_command",
    )

    summary = registry.summary(limit=5, cluster_limit=5)
    assert summary["total_entries"] == 3
    assert summary["reason_counts"][BlockReason.RETRY_BLOCKED.value] == 2
    recent_ids = {item["id"] for item in summary["recent_entries"]}
    assert second.id in recent_ids
    assert summary["top_clusters"][0]["fingerprint"] == first.fingerprint
    assert summary["top_clusters"][0]["count"] == 2

    planning = registry.build_planning_context(
        objective="Continue Chronicle 4246 from the existing manuscript.",
        artifact="workspace/Chronicles/4246/chronicle_4246.md",
    )
    assert planning["available"] is True
    assert planning["clusters"][0]["count"] == 2
    assert "deterministic review" in planning["prompt_block"].lower()
    assert "narrow edit" in planning["prompt_block"].lower()
    assert "previously blocked framings to avoid repeating" in planning["prompt_block"].lower()
    assert "continue chronicle 4246 from the existing manuscript." in planning["prompt_block"].lower()
    assert "blocker handling rule" in planning["prompt_block"].lower()


def test_shadow_registry_cluster_triage_persists_and_filters_summary(tmp_path: Path) -> None:
    store = ShadowRegistryStore(tmp_path / "shadow_registry")
    registry = ShadowRegistry(store=store)

    chronicle = registry.capture(
        tool_name="repair_retry",
        parameters={
            "objective": "Continue Chronicle 4246 from the existing manuscript.",
            "canonical_artifact_path": "workspace/Chronicles/4246/chronicle_4246.md",
            "attempt": 2,
        },
        reason=BlockReason.RETRY_BLOCKED,
        context="RetryGovernor blocked a broad retry with no new evidence.",
        artifact="workspace/Chronicles/4246/chronicle_4246.md",
        target_kind="repair_task",
        target_id="task-4246",
        capture_source="repair_executor",
    )
    registry.capture(
        tool_name="repair_retry",
        parameters={
            "objective": "Continue Chronicle 4246 from the existing manuscript.",
            "canonical_artifact_path": "workspace/Chronicles/4246/chronicle_4246.md",
            "attempt": 3,
        },
        reason=BlockReason.RETRY_BLOCKED,
        context="RetryGovernor blocked another broad retry with no new evidence.",
        artifact="workspace/Chronicles/4246/chronicle_4246.md",
        target_kind="repair_task",
        target_id="task-4246",
        capture_source="repair_executor",
    )
    safety = registry.capture(
        tool_name="bash_run_command",
        parameters={"command": "rm -rf /", "cwd": "/"},
        reason=BlockReason.SAFETY_BLOCKED,
        context="Command blocked by safety policy: blocked pattern: rm -rf /",
        session_id="session:default:abc",
        artifact="tool|default|bash_run_command",
    )

    triaged = registry.triage_cluster(
        chronicle.fingerprint,
        annotation="Known Chronicle retry loop; handled manually.",
        dismissed=True,
    )
    assert triaged["available"] is True
    assert triaged["triage_status"] == "dismissed"
    assert triaged["annotation"] == "Known Chronicle retry loop; handled manually."

    summary = registry.summary(limit=5, cluster_limit=5)
    assert summary["dismissed_clusters"] == 1
    assert summary["active_clusters"] == 1
    assert summary["top_clusters"][0]["fingerprint"] == safety.fingerprint
    assert summary["top_clusters"][0]["triage_status"] == "active"

    detail = registry.inspect_cluster(chronicle.fingerprint)
    assert detail["annotation"] == "Known Chronicle retry loop; handled manually."
    assert detail["triage_status"] == "dismissed"

    planning = registry.build_planning_context(
        objective="Continue Chronicle 4246 from the existing manuscript.",
        artifact="workspace/Chronicles/4246/chronicle_4246.md",
    )
    assert planning["available"] is False

    reloaded = ShadowRegistry(store=ShadowRegistryStore(tmp_path / "shadow_registry"))
    reloaded_detail = reloaded.inspect_cluster(chronicle.fingerprint)
    assert reloaded_detail["annotation"] == "Known Chronicle retry loop; handled manually."
    assert reloaded_detail["triage_status"] == "dismissed"
