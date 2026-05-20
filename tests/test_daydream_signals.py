from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.api.routes.daydream import _signal_to_dict
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.cognition import CognitionGrounding, GroundingKind, GroundingSource
from opencas.daydream.promotion import DaydreamPromotionService
from opencas.daydream.self_workspace import SelfWorkspaceService
from opencas.daydream.signal_builder import DaydreamSignalBuilder
from opencas.daydream.signal_store import DaydreamSignalStore
from opencas.daydream.signals import (
    ContactPosture,
    PossibilitySignal,
    PossibilitySignalRoute,
    SelfWorkKind,
)
from opencas.runtime.agent_loop import AgentRuntime
from opencas.thread_registry import BeadSourceKind, ThreadRegistryService, ThreadRegistryStore


def test_possibility_signal_model_preserves_branches_and_route() -> None:
    signal = PossibilitySignal(
        source_reflection_id="reflection-1",
        source_thought_index=0,
        source_mode="waking_daydream",
        summary="Build a tiny proof browser for claims.",
        imaginative_branch="A thought could become a little lantern showing where it came from.",
        practical_branch="A proof browser would make claims inspectable before trust is spent.",
        bridge="The lantern image points at a UI that exposes provenance instead of prose confidence.",
        novelty=0.82,
        usefulness=0.91,
        confidence=0.77,
        suggested_route=PossibilitySignalRoute.SELF_EXPERIMENT,
        contact_posture=ContactPosture.SHARE_AFTER_ARTIFACT,
        self_work_kind=SelfWorkKind.EXPERIMENT,
    )

    assert signal.signal_id
    assert signal.suggested_route == PossibilitySignalRoute.SELF_EXPERIMENT
    assert signal.contact_posture == ContactPosture.SHARE_AFTER_ARTIFACT
    assert signal.self_work_kind == SelfWorkKind.EXPERIMENT
    assert "lantern" in signal.imaginative_branch


def test_daydream_signal_api_labels_pre_fix_route_semantics() -> None:
    old_signal = PossibilitySignal(
        source_reflection_id="reflection-old",
        source_thought_index=0,
        summary="Investigate an old research route.",
        suggested_route=PossibilitySignalRoute.RESEARCH,
        route_status="held",
        route_reason="research handler not wired",
    )
    current_signal = PossibilitySignal(
        source_reflection_id="reflection-new",
        source_thought_index=0,
        summary="Investigate a current research route.",
        suggested_route=PossibilitySignalRoute.RESEARCH,
        route_status="routed",
        route_reason="self_research_note_written",
        meta={"route_schema_version": 2},
    )

    old_payload = _signal_to_dict(old_signal)
    current_payload = _signal_to_dict(current_signal)

    assert old_payload["route_schema_version"] == 1
    assert old_payload["stale_route_semantics"] is True
    assert current_payload["route_schema_version"] == 2
    assert current_payload["stale_route_semantics"] is False


@pytest.mark.asyncio
async def test_signal_store_round_trips_signal_and_route(tmp_path: Path) -> None:
    store = await DaydreamSignalStore(tmp_path / "signals.db").connect()
    signal = PossibilitySignal(
        source_reflection_id="reflection-1",
        source_thought_index=0,
        source_mode="waking_daydream",
        summary="Try an inspectable proof browser.",
        imaginative_branch="A lantern over each claim.",
        practical_branch="Operators can see claim evidence.",
        bridge="The image becomes a provenance UI sketch.",
        novelty=0.7,
        usefulness=0.8,
        confidence=0.75,
        suggested_route=PossibilitySignalRoute.THREAD,
    )

    await store.save_signal(signal)
    loaded = await store.get_signal(signal.signal_id)
    assert loaded is not None
    assert loaded.practical_branch == "Operators can see claim evidence."

    await store.record_route(
        signal.signal_id,
        route=PossibilitySignalRoute.THREAD,
        status="routed",
        reason="high-usefulness peripheral idea",
        artifact_paths=["workspace/self/notes/example.md"],
    )
    recent = await store.list_recent(limit=5)
    summary = await store.get_summary(window_days=7)

    assert recent[0].signal_id == signal.signal_id
    assert recent[0].route_status == "routed"
    assert summary["total_signals"] == 1
    assert summary["route_counts"]["thread"] == 1
    assert summary["status_counts"]["routed"] == 1
    await store.close()


def test_signal_builder_turns_fallback_reflection_into_incubated_signal() -> None:
    from opencas.daydream import DaydreamReflection, DaydreamThought

    reflection_id = "11111111-1111-4111-8111-111111111111"
    reflection = DaydreamReflection(
        reflection_id=reflection_id,
        spark_content="just a plain text spark",
        thoughts=[
            DaydreamThought(
                summary="just a plain text spark",
                confidence=0.4,
                usefulness=0.35,
                novelty=0.5,
            )
        ],
    )

    signals = DaydreamSignalBuilder().build_from_reflection(reflection)

    assert len(signals) == 1
    assert signals[0].source_reflection_id == reflection_id
    assert signals[0].summary == "just a plain text spark"
    assert signals[0].suggested_route == PossibilitySignalRoute.INCUBATE
    assert signals[0].confidence == 0.4
    assert signals[0].practical_branch == ""


def test_signal_builder_uses_enriched_thought_route_when_grounded() -> None:
    from opencas.daydream import (
        DaydreamDialogueTurn,
        DaydreamReflection,
        DaydreamThought,
        DaydreamThoughtRoute,
    )

    reflection = DaydreamReflection(
        reflection_id="22222222-2222-4222-8222-222222222222",
        spark_content="A tiny proof browser might be worth prototyping.",
        thoughts=[
            DaydreamThought(
                route=DaydreamThoughtRoute.DEEP_THINK,
                summary="A tiny proof browser might be worth prototyping.",
                imaginative_branch="Claims become little lit windows.",
                practical_branch="A small artifact could show unresolved claims at a glance.",
                bridge="The lit-window image maps to a compact provenance panel.",
                suggested_handler="self_experiment",
                contact_posture="share_after_artifact",
                self_work_intent="sketch a local proof-browser experiment",
                inner_dialogue=[
                    DaydreamDialogueTurn(
                        voice="curious",
                        stance="opening question",
                        text="What would make unresolved proof visible?",
                    ),
                    DaydreamDialogueTurn(
                        voice="guardian",
                        stance="authority boundary",
                        text="Keep it as self-work until it has evidence.",
                    ),
                ],
                confidence=0.82,
                usefulness=0.86,
                novelty=0.72,
                grounding=[
                    CognitionGrounding(
                        kind=GroundingKind.OBSERVED,
                        source=GroundingSource.RUNTIME,
                        subject="proof_browser",
                        claim="A proof-browser idea appeared in current daydream context.",
                        evidence_ids=["daydream:test-grounded-route"],
                    )
                ],
            )
        ],
    )

    signals = DaydreamSignalBuilder().build_from_reflection(reflection)

    assert signals[0].suggested_route == PossibilitySignalRoute.SELF_EXPERIMENT
    assert signals[0].contact_posture == ContactPosture.SHARE_AFTER_ARTIFACT
    assert signals[0].self_work_kind == SelfWorkKind.EXPERIMENT
    assert signals[0].self_work_intent == "sketch a local proof-browser experiment"
    assert signals[0].meta["inner_dialogue_turn_count"] == 2
    assert "guardian" in signals[0].meta["inner_dialogue_voices"]


def test_signal_builder_incubates_high_confidence_action_without_grounding() -> None:
    from opencas.daydream import DaydreamReflection, DaydreamThought, DaydreamThoughtRoute

    reflection = DaydreamReflection(
        reflection_id="33333333-3333-4333-8333-333333333333",
        spark_content="Build an autonomous subsystem immediately.",
        thoughts=[
            DaydreamThought(
                route=DaydreamThoughtRoute.ACT_NOW,
                summary="Build an autonomous subsystem immediately.",
                practical_branch="This would create work, but the thought lacks source evidence.",
                confidence=0.92,
                usefulness=0.91,
                novelty=0.8,
            )
        ],
    )

    signals = DaydreamSignalBuilder().build_from_reflection(reflection)

    assert signals[0].suggested_route == PossibilitySignalRoute.INCUBATE
    assert signals[0].route_reason == "high-action route requires grounding or evidence ids"
    assert signals[0].meta["original_route"] == "work_candidate"


@pytest.mark.asyncio
async def test_self_workspace_writes_note_and_receipt_under_self_root(tmp_path: Path) -> None:
    signal = PossibilitySignal(
        source_reflection_id="reflection-1",
        source_thought_index=0,
        summary="Sketch an inspectable proof browser.",
        imaginative_branch="Claims become lit windows.",
        practical_branch="Operators can inspect unresolved claims.",
        bridge="The image maps to a small provenance panel.",
        suggested_route=PossibilitySignalRoute.SELF_NOTE,
        self_work_kind=SelfWorkKind.NOTE,
    )
    service = SelfWorkspaceService(workspace_root=tmp_path / "workspace")

    receipt = await service.write_note(signal, reason="useful enough to preserve")

    assert receipt.signal_id == signal.signal_id
    assert receipt.route == PossibilitySignalRoute.SELF_NOTE
    assert receipt.kind == SelfWorkKind.NOTE
    assert len(receipt.artifact_paths) == 2
    for artifact in receipt.artifact_paths:
        path = Path(artifact)
        assert path.exists()
        assert (tmp_path / "workspace" / "self").resolve() in path.resolve().parents


@pytest.mark.asyncio
async def test_self_workspace_marks_prototype_as_scaffold_until_validated(tmp_path: Path) -> None:
    signal = PossibilitySignal(
        source_reflection_id="reflection-prototype",
        source_thought_index=0,
        summary="Prototype a small claim browser.",
        imaginative_branch="Claims become lit panes.",
        practical_branch="A scaffold can preserve the idea before a real runnable implementation.",
        bridge="The image maps to a local artifact plan.",
        suggested_route=PossibilitySignalRoute.SELF_PROTOTYPE,
        self_work_kind=SelfWorkKind.PROTOTYPE,
    )
    service = SelfWorkspaceService(workspace_root=tmp_path / "workspace")

    receipt = await service.write_prototype(signal, reason="prototype requires validation")

    assert receipt.outcome == "self_prototype_scaffolded"
    assert receipt.raw["validation_status"] == "scaffold_only"
    assert any(path.endswith("README.md") for path in receipt.artifact_paths)
    assert any(path.endswith("PROTOTYPE_MANIFEST.json") for path in receipt.artifact_paths)
    assert any(path.endswith("VALIDATION.md") for path in receipt.artifact_paths)


@pytest.mark.asyncio
async def test_self_workspace_compost_records_salvage_and_discard_metadata(tmp_path: Path) -> None:
    signal = PossibilitySignal(
        source_reflection_id="reflection-compost",
        source_thought_index=0,
        summary="Discard a weak project scaffold but keep the useful question.",
        imaginative_branch="A failed scaffold becomes soil for a clearer later idea.",
        practical_branch="The useful question should survive while the artifact stays inactive.",
        bridge="Compost keeps salvage separate from active work.",
        suggested_route=PossibilitySignalRoute.COMPOST,
        self_work_kind=SelfWorkKind.COMPOST,
        evidence_ids=["self-work:weak-scaffold"],
    )
    service = SelfWorkspaceService(workspace_root=tmp_path / "workspace")

    receipt = await service.compost(signal, reason="failed usefulness review")

    assert receipt.outcome == "self_work_composted"
    assert receipt.raw["compost_status"] == "composted"
    assert any(path.endswith("SALVAGE_INDEX.json") for path in receipt.artifact_paths)
    salvage_path = next(Path(path) for path in receipt.artifact_paths if path.endswith("SALVAGE_INDEX.json"))
    assert '"salvage_candidates"' in salvage_path.read_text(encoding="utf-8")


def test_self_workspace_rejects_path_escape(tmp_path: Path) -> None:
    service = SelfWorkspaceService(workspace_root=tmp_path / "workspace")

    with pytest.raises(ValueError, match="outside self workspace"):
        service.resolve_self_path("../outside.md")


class _FakeThreadRegistry:
    def __init__(self) -> None:
        self.beads: list[dict] = []

    async def ensure_thread_anchor(self, **kwargs):
        return SimpleNamespace(anchor_id="anchor-1")

    async def create_candidate_bead(self, **kwargs):
        self.beads.append(kwargs)
        return SimpleNamespace(bead_id="bead-1")


class _FakeInitiativeContact:
    def __init__(self) -> None:
        self.candidates: list[dict] = []

    async def consider_candidate(self, candidate: dict):
        self.candidates.append(candidate)
        return {"status": "held", "reason": "test"}


class _FakeCreative:
    def __init__(self) -> None:
        self.items: list = []

    def add(self, item) -> None:
        self.items.append(item)


@pytest.mark.asyncio
async def test_promotion_routes_thread_signal_to_thread_registry(tmp_path: Path) -> None:
    store = await DaydreamSignalStore(tmp_path / "signals.db").connect()
    thread_registry = _FakeThreadRegistry()
    signal = PossibilitySignal(
        source_reflection_id="reflection-1",
        source_thought_index=0,
        summary="A provenance bead should be preserved.",
        imaginative_branch="A small breadcrumb shines beside each claim.",
        practical_branch="The bead keeps a loose system insight recoverable.",
        bridge="The breadcrumb image maps to peripheral thread pickup.",
        suggested_route=PossibilitySignalRoute.THREAD,
        usefulness=0.8,
        confidence=0.8,
    )
    await store.save_signal(signal)
    service = DaydreamPromotionService(
        signal_store=store,
        self_workspace=SelfWorkspaceService(workspace_root=tmp_path / "workspace"),
        thread_registry_service=thread_registry,
    )

    result = await service.route_signal(signal)
    recent = await store.list_recent(limit=1)

    assert result["status"] == "routed"
    assert thread_registry.beads[0]["source_ref"] == f"daydream_signal:{signal.signal_id}"
    assert recent[0].route_status == "routed"
    await store.close()


@pytest.mark.asyncio
async def test_promotion_writes_self_note_receipt(tmp_path: Path) -> None:
    store = await DaydreamSignalStore(tmp_path / "signals.db").connect()
    thread_registry = _FakeThreadRegistry()
    signal = PossibilitySignal(
        source_reflection_id="reflection-2",
        source_thought_index=0,
        summary="Write a note about proof browser shape.",
        imaginative_branch="Claims become lit windows.",
        practical_branch="A note can preserve the useful UI shape.",
        bridge="The image maps to a small inspectable artifact.",
        suggested_route=PossibilitySignalRoute.SELF_NOTE,
        self_work_kind=SelfWorkKind.NOTE,
        usefulness=0.8,
        confidence=0.8,
    )
    await store.save_signal(signal)
    service = DaydreamPromotionService(
        signal_store=store,
        self_workspace=SelfWorkspaceService(workspace_root=tmp_path / "workspace"),
        thread_registry_service=thread_registry,
    )

    result = await service.route_signal(signal)
    receipts = await store.list_receipts(limit=5)

    assert result["status"] == "routed"
    assert receipts[0].signal_id == signal.signal_id
    assert receipts[0].kind == SelfWorkKind.NOTE
    assert any(path.endswith(".md") for path in receipts[0].artifact_paths)
    assert thread_registry.beads
    assert thread_registry.beads[0]["source_ref"] == f"daydream_signal:{signal.signal_id}"
    assert thread_registry.beads[0]["source_kind"] == BeadSourceKind.DAYDREAM_SIGNAL
    assert "self_note" in thread_registry.beads[0]["content"]
    await store.close()


@pytest.mark.asyncio
async def test_backfill_records_existing_routed_daydream_signals_as_thread_beads(tmp_path: Path) -> None:
    store = await DaydreamSignalStore(tmp_path / "signals.db").connect()
    thread_registry = _FakeThreadRegistry()
    signal = PossibilitySignal(
        source_reflection_id="reflection-backfill",
        source_thought_index=1,
        summary="A routed signal should not stay write-only.",
        imaginative_branch="The old spark is still glowing in the ledger.",
        practical_branch="Backfill a bead so existing signals can be consumed.",
        bridge="The image maps to producer-consumer closure.",
        suggested_route=PossibilitySignalRoute.SELF_PROTOTYPE,
        self_work_kind=SelfWorkKind.PROTOTYPE,
        route_status="routed",
        route_reason="prototype saved earlier",
        usefulness=0.8,
        confidence=0.8,
        artifact_paths=["workspace/self/prototypes/proof-browser.md"],
    )
    await store.save_signal(signal)
    service = DaydreamPromotionService(
        signal_store=store,
        self_workspace=SelfWorkspaceService(workspace_root=tmp_path / "workspace"),
        thread_registry_service=thread_registry,
    )

    result = await service.backfill_signal_thread_beads(limit=10)

    assert result["available"] is True
    assert result["scanned"] == 1
    assert result["recorded"] == 1
    assert thread_registry.beads[0]["source_ref"] == f"daydream_signal:{signal.signal_id}"
    assert thread_registry.beads[0]["source_kind"] == BeadSourceKind.DAYDREAM_SIGNAL
    assert "workspace/self/prototypes/proof-browser.md" in thread_registry.beads[0]["content"]
    await store.close()


@pytest.mark.asyncio
async def test_backfill_signal_thread_beads_is_idempotent_across_restarts(tmp_path: Path) -> None:
    store = await DaydreamSignalStore(tmp_path / "signals.db").connect()
    thread_store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    thread_registry = ThreadRegistryService(
        store=thread_store,
        workspace_root=tmp_path / "workspace",
    )
    signal = PossibilitySignal(
        source_reflection_id="reflection-idempotent",
        source_thought_index=1,
        summary="A routed signal should stay single after scheduler restart.",
        imaginative_branch="The bead should not echo with every boot.",
        practical_branch="A repeated backfill should reuse the source/content hash.",
        bridge="The image maps to scheduler restart idempotence.",
        suggested_route=PossibilitySignalRoute.SELF_NOTE,
        self_work_kind=SelfWorkKind.NOTE,
        route_status="routed",
        route_reason="note saved earlier",
        usefulness=0.8,
        confidence=0.8,
        artifact_paths=["workspace/self/notes/restart-idempotence.md"],
    )
    await store.save_signal(signal)
    service = DaydreamPromotionService(
        signal_store=store,
        self_workspace=SelfWorkspaceService(workspace_root=tmp_path / "workspace"),
        thread_registry_service=thread_registry,
    )

    first = await service.backfill_signal_thread_beads(limit=10)
    bead_count_after_first = len(
        await thread_store.list_beads(source_kind=BeadSourceKind.DAYDREAM_SIGNAL)
    )
    second = await service.backfill_signal_thread_beads(limit=10)
    beads_after_second = await thread_store.list_beads(
        source_kind=BeadSourceKind.DAYDREAM_SIGNAL
    )

    assert first["recorded"] == 1
    assert second["recorded"] == 1
    assert bead_count_after_first == 1
    assert len(beads_after_second) == 1
    assert beads_after_second[0].source_ref == f"daydream_signal:{signal.signal_id}"
    await store.close()
    await thread_store.close()


@pytest.mark.asyncio
async def test_promotion_routes_research_signal_to_self_workspace(tmp_path: Path) -> None:
    store = await DaydreamSignalStore(tmp_path / "signals.db").connect()
    signal = PossibilitySignal(
        source_reflection_id="reflection-research",
        source_thought_index=0,
        summary="Investigate whether proof summaries are undercounting evidence links.",
        imaginative_branch="A ledger row can look verified while hiding its receipts.",
        practical_branch="A contained research note can preserve the question and evidence IDs.",
        bridge="The image maps to a proof-summary query.",
        suggested_route=PossibilitySignalRoute.RESEARCH,
        self_work_kind=SelfWorkKind.RESEARCH,
        evidence_ids=["proof-summary:live-audit"],
        usefulness=0.86,
        confidence=0.84,
    )
    await store.save_signal(signal)
    service = DaydreamPromotionService(
        signal_store=store,
        self_workspace=SelfWorkspaceService(workspace_root=tmp_path / "workspace"),
    )

    result = await service.route_signal(signal)
    receipts = await store.list_receipts(limit=5)

    assert result["status"] == "routed"
    assert result["reason"] == "self_research_note_written"
    assert receipts[0].kind == SelfWorkKind.RESEARCH
    assert receipts[0].outcome == "self_research_note_written"
    assert any("/self/research/" in path for path in receipts[0].artifact_paths)
    await store.close()


@pytest.mark.asyncio
async def test_promotion_incubates_weak_signal_without_contact_or_work(tmp_path: Path) -> None:
    store = await DaydreamSignalStore(tmp_path / "signals.db").connect()
    contact = _FakeInitiativeContact()
    creative = _FakeCreative()
    signal = PossibilitySignal(
        source_reflection_id="reflection-3",
        source_thought_index=0,
        summary="Maybe something vague.",
        confidence=0.2,
        usefulness=0.2,
        suggested_route=PossibilitySignalRoute.INCUBATE,
    )
    await store.save_signal(signal)
    service = DaydreamPromotionService(
        signal_store=store,
        self_workspace=SelfWorkspaceService(workspace_root=tmp_path / "workspace"),
        initiative_contact=contact,
        creative=creative,
    )

    result = await service.route_signal(signal)
    recent = await store.list_recent(limit=1)

    assert result["status"] == "incubated"
    assert recent[0].route_status == "incubated"
    assert contact.candidates == []
    assert creative.items == []
    await store.close()


@pytest.mark.asyncio
async def test_promotion_ask_user_sends_enriched_candidate(tmp_path: Path) -> None:
    store = await DaydreamSignalStore(tmp_path / "signals.db").connect()
    contact = _FakeInitiativeContact()
    signal = PossibilitySignal(
        source_reflection_id="reflection-4",
        source_thought_index=0,
        summary="Ask whether a proof browser would be useful now.",
        imaginative_branch="A lantern could show claim provenance.",
        practical_branch="The operator can decide if this is worth self-work time.",
        bridge="The image becomes a concrete question about an inspectable UI.",
        suggested_route=PossibilitySignalRoute.ASK_USER,
        contact_posture=ContactPosture.ASK_FIRST,
        confidence=0.7,
        usefulness=0.75,
    )
    await store.save_signal(signal)
    service = DaydreamPromotionService(
        signal_store=store,
        self_workspace=SelfWorkspaceService(workspace_root=tmp_path / "workspace"),
        initiative_contact=contact,
    )

    result = await service.route_signal(signal)

    assert result["status"] == "deferred"
    assert contact.candidates[0]["source_id"] == signal.signal_id
    assert contact.candidates[0]["raw"]["imaginative_branch"] == signal.imaginative_branch
    assert "message" not in contact.candidates[0]
    await store.close()


@pytest.mark.asyncio
async def test_bootstrap_and_runtime_expose_daydream_signal_services(tmp_path: Path) -> None:
    config = BootstrapConfig(
        state_dir=tmp_path / "state",
        workspace_root=tmp_path / "repo",
    )
    ctx = await BootstrapPipeline(config).run()
    try:
        assert ctx.daydream_signal_store is not None
        assert ctx.daydream_signal_store.path == ctx.config.state_dir / "daydream_signals.db"

        runtime = AgentRuntime(ctx)

        assert runtime.daydream_signal_builder is not None
        assert runtime.self_workspace.self_root == ctx.config.agent_workspace_root() / "self"
        assert runtime.daydream_promotion.signal_store is ctx.daydream_signal_store
        assert runtime.daydream_promotion.initiative_contact is runtime.initiative_contact
    finally:
        await ctx.close()
