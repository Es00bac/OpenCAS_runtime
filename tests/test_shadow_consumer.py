from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.cognition import CognitiveEventKind
from opencas.governance import BlockReason, ShadowRegistry, ShadowRegistryStore
from opencas.proof_chain import ProofClaimType, ProofEvidenceKind


class FakeCognitiveStore:
    def __init__(self) -> None:
        self.skills = []
        self.events = []

    async def upsert_learned_skill(self, skill):
        self.skills.append(skill)
        return skill

    async def record_event(self, kind, summary, **kwargs):
        event = SimpleNamespace(
            event_id=f"event-{len(self.events) + 1}",
            kind=kind,
            summary=summary,
            **kwargs,
        )
        self.events.append(event)
        return event


class FakeProofChain:
    def __init__(self) -> None:
        self.claims = []
        self.links = []

    async def record_claim(self, **kwargs):
        claim = SimpleNamespace(claim_id=f"claim-{len(self.claims) + 1}", **kwargs)
        self.claims.append(claim)
        return claim

    async def link_evidence(self, claim_id, **kwargs):
        self.links.append({"claim_id": claim_id, **kwargs})
        return SimpleNamespace(claim_id=claim_id)


class FakeTracer:
    def __init__(self) -> None:
        self.events = []

    def log(self, kind, message, payload):
        self.events.append({"kind": kind, "message": message, "payload": payload})


def _runtime(registry: ShadowRegistry) -> SimpleNamespace:
    return SimpleNamespace(
        ctx=SimpleNamespace(shadow_registry=registry),
        cognitive_state_store=FakeCognitiveStore(),
        proof_chain=FakeProofChain(),
        tracer=FakeTracer(),
    )


def _capture_cluster(
    registry: ShadowRegistry,
    *,
    tool_name: str,
    reason: BlockReason,
    summary_seed: str,
    captured_at: datetime,
):
    item = registry.capture(
        tool_name=tool_name,
        parameters={"path": summary_seed},
        reason=reason,
        context=f"{reason.value} for {summary_seed}",
        target_kind="test",
        target_id=summary_seed,
        capture_source="test",
    )
    item.captured_at = captured_at
    return registry.store.save(item)


@pytest.mark.asyncio
async def test_shadow_consumer_writes_learned_skill_before_dismissing_cohort(tmp_path: Path) -> None:
    from opencas.runtime.shadow_consumer_runtime import consume_shadow_registry

    now = datetime(2026, 5, 7, tzinfo=timezone.utc)
    registry = ShadowRegistry(ShadowRegistryStore(tmp_path / "shadow_registry"))
    for index in range(5):
        _capture_cluster(
            registry,
            tool_name="fs_read_file",
            reason=BlockReason.VALIDATION_BLOCKED,
            summary_seed=f"/outside/root/{index}.md",
            captured_at=now - timedelta(days=2),
        )
    runtime = _runtime(registry)

    result = await consume_shadow_registry(runtime, now=now, max_dismissals=20)

    assert result["dismissed_count"] == 5
    assert result["learned_skill_count"] == 1
    assert len(runtime.cognitive_state_store.skills) == 1
    skill = runtime.cognitive_state_store.skills[0]
    assert skill.skill_id == "shadow:fs_read_file:validation_blocked"
    assert len(skill.evidence_refs) == 5
    assert "allowed roots" in " ".join(skill.preconditions).lower()
    assert registry.summary()["dismissed_clusters"] == 5
    assert len(runtime.proof_chain.claims) == 5
    assert runtime.proof_chain.claims[0].claim_type == ProofClaimType.MAINTENANCE
    assert {link["evidence_kind"] for link in runtime.proof_chain.links} == {
        ProofEvidenceKind.RUNTIME_EVENT
    }
    assert len(runtime.tracer.events) == 5
    assert runtime.tracer.events[0]["payload"]["reversible"] is True


@pytest.mark.asyncio
async def test_shadow_consumer_records_counterfactual_for_time_decay_only_cluster(
    tmp_path: Path,
) -> None:
    from opencas.runtime.shadow_consumer_runtime import consume_shadow_registry

    now = datetime(2026, 5, 7, tzinfo=timezone.utc)
    registry = ShadowRegistry(ShadowRegistryStore(tmp_path / "shadow_registry"))
    item = _capture_cluster(
        registry,
        tool_name="single_tool",
        reason=BlockReason.SAFETY_BLOCKED,
        summary_seed="dangerous command",
        captured_at=now - timedelta(days=8),
    )
    runtime = _runtime(registry)

    result = await consume_shadow_registry(runtime, now=now, max_dismissals=20)

    assert result["dismissed_count"] == 1
    assert result["counterfactual_count"] == 1
    assert runtime.cognitive_state_store.events[0].kind == CognitiveEventKind.COUNTERFACTUAL
    assert runtime.cognitive_state_store.events[0].evidence_refs == [
        f"shadow_registry:{item.fingerprint}"
    ]
    detail = registry.inspect_cluster(item.fingerprint)
    assert detail["triage_status"] == "dismissed"
    assert "cognitive_event:event-1" in detail["annotation"]


@pytest.mark.asyncio
async def test_shadow_consumer_enforces_per_tick_bound(tmp_path: Path) -> None:
    from opencas.runtime.shadow_consumer_runtime import consume_shadow_registry

    now = datetime(2026, 5, 7, tzinfo=timezone.utc)
    registry = ShadowRegistry(ShadowRegistryStore(tmp_path / "shadow_registry"))
    for index in range(25):
        _capture_cluster(
            registry,
            tool_name=f"single_tool_{index}",
            reason=BlockReason.SAFETY_BLOCKED,
            summary_seed=f"old blocked command {index}",
            captured_at=now - timedelta(days=8),
        )
    runtime = _runtime(registry)

    result = await consume_shadow_registry(runtime, now=now, max_dismissals=20)

    assert result["dismissed_count"] == 20
    assert registry.summary()["dismissed_clusters"] == 20


@pytest.mark.asyncio
async def test_shadow_consumer_skips_recent_clusters_even_when_cohort_matches(tmp_path: Path) -> None:
    from opencas.runtime.shadow_consumer_runtime import consume_shadow_registry

    now = datetime(2026, 5, 7, tzinfo=timezone.utc)
    registry = ShadowRegistry(ShadowRegistryStore(tmp_path / "shadow_registry"))
    for index in range(5):
        _capture_cluster(
            registry,
            tool_name="bash_run_command",
            reason=BlockReason.APPROVAL_DENIED,
            summary_seed=f"recent shell {index}",
            captured_at=now - timedelta(hours=2),
        )
    runtime = _runtime(registry)

    result = await consume_shadow_registry(runtime, now=now, max_dismissals=20)

    assert result["dismissed_count"] == 0
    assert registry.summary()["active_clusters"] == 5
