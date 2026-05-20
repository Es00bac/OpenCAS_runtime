from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from opencas.api.server import create_app
from opencas.bootstrap import BootstrapConfig
from opencas.bootstrap.pipeline_stores import initialize_runtime_stores
from opencas.cognition import ToolCallTransit
from opencas.daydream import (
    DaydreamReflection,
    DaydreamThought,
    DaydreamThoughtKind,
    DaydreamThoughtRoute,
)
from opencas.identity import IdentityManager, IdentityStore
from opencas.thread_registry.runtime_bridge import (
    record_failed_tool_call_transit_thread_beads,
    record_runtime_daydream_thread_beads,
)
from opencas.telemetry import TelemetryStore, Tracer
from opencas.thread_registry import (
    BeadSourceKind,
    BeadStatus,
    ThreadRegistryService,
    ThreadRegistryStore,
    ThreadStatus,
    compute_content_hashes,
)
from opencas.tools.adapters.thread_registry import ThreadRegistryToolAdapter


@pytest.mark.asyncio
async def test_registry_hashes_full_content_not_prefix(tmp_path: Path) -> None:
    store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    service = ThreadRegistryService(store=store, workspace_root=tmp_path / "workspace")

    prefix = "same visible opening " * 20
    content_a = prefix + "ending A"
    content_b = prefix + "ending B"

    full_a, short_a = compute_content_hashes(content_a)
    full_b, short_b = compute_content_hashes(content_b)

    assert len(full_a) == 64
    assert short_a == full_a[:12]
    assert full_a != full_b
    assert short_a != short_b

    bead = await service.ingest_autonomous_artifact(
        source_ref="workspace/if_i_am_the_operator/manuscript_draft.md",
        content=content_a,
        thread_title="Operator attention registry",
        title="Manuscript registry insight",
        summary="A proposal to keep autonomous artifacts retrievable without turning them into obligations.",
    )

    assert bead.content_hash_full == full_a
    assert bead.content_hash_short == short_a
    assert bead.user_commissioned is False
    assert bead.status == BeadStatus.PERIPHERAL
    assert bead.validation is not None
    assert bead.validation.dangling is False
    assert bead.validation.should_create_task is False

    await store.close()


@pytest.mark.asyncio
async def test_unanchored_candidate_stays_dangling_not_task(tmp_path: Path) -> None:
    store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    service = ThreadRegistryService(store=store, workspace_root=tmp_path / "workspace")

    bead = await service.create_candidate_bead(
        thread_anchor_id="unknown-thread",
        title="Loose observation",
        summary="An observation that cannot yet be placed in an intelligible ongoing thread.",
        source_kind=BeadSourceKind.DAYDREAM_REFLECTION,
        source_ref="daydream:reflection-1",
        content="This needs a later operator to decide whether it belongs anywhere.",
        user_commissioned=False,
    )

    assert bead.status == BeadStatus.DANGLING
    assert bead.validation is not None
    assert bead.validation.anchor_resolved is False
    assert bead.validation.dangling is True
    assert bead.validation.should_create_task is False

    listed = await store.list_beads(status=BeadStatus.DANGLING)
    assert [item.bead_id for item in listed] == [bead.bead_id]

    await store.close()


@pytest.mark.asyncio
async def test_store_round_trips_threads_and_filters_beads(tmp_path: Path) -> None:
    store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    service = ThreadRegistryService(store=store, workspace_root=tmp_path / "workspace")

    anchor = await service.ensure_thread_anchor(
        title="Curiosity capture",
        kind="system_insight",
        status=ThreadStatus.PERIPHERAL,
    )
    first = await service.create_candidate_bead(
        thread_anchor_id=anchor.anchor_id,
        title="First bead",
        summary="A finished-enough unit of meaning that can be picked up later.",
        source_kind=BeadSourceKind.MANUAL,
        source_ref="manual:test",
        content="A concise but recoverable thought.",
        user_commissioned=True,
    )
    second = await service.create_candidate_bead(
        thread_anchor_id=anchor.anchor_id,
        title="Second bead",
        summary="Another recoverable unit that should remain attached to the same thread.",
        source_kind=BeadSourceKind.FASCINATION,
        source_ref="fascination:test",
        content="Another concise thought.",
        user_commissioned=False,
    )

    loaded_anchor = await store.get_thread_anchor(anchor.anchor_id)
    assert loaded_anchor is not None
    assert loaded_anchor.title == "Curiosity capture"

    all_for_thread = await store.list_beads(thread_anchor_id=anchor.anchor_id)
    assert {item.bead_id for item in all_for_thread} == {first.bead_id, second.bead_id}

    manual = await store.list_beads(source_kind=BeadSourceKind.MANUAL)
    assert [item.bead_id for item in manual] == [first.bead_id]

    await store.close()


@pytest.mark.asyncio
async def test_thread_registry_tool_adapter_creates_and_queries(tmp_path: Path) -> None:
    store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    service = ThreadRegistryService(store=store, workspace_root=tmp_path / "workspace")

    class Runtime:
        thread_registry_service = service

    adapter = ThreadRegistryToolAdapter(Runtime())
    create_result = await adapter(
        "thread_registry_create_candidate",
        {
            "thread_title": "Operator self-debugging",
            "title": "Need a searchable trace",
            "summary": "A meta-tool-call inspector idea should be retained as a peripheral design bead.",
            "source_kind": "autonomous_artifact",
            "source_ref": "workspace/if_i_am_the_operator/manuscript_draft.md",
            "content": "A trace inspector lets later reasoning compare tool intent against tool use.",
            "user_commissioned": False,
        },
    )

    assert create_result.success is True
    assert create_result.metadata["bead"]["status"] == "peripheral"
    assert create_result.metadata["bead"]["user_commissioned"] is False

    query_result = await adapter(
        "thread_registry_query",
        {
            "thread_anchor_id": create_result.metadata["bead"]["thread_anchor_id"],
            "limit": 5,
        },
    )

    assert query_result.success is True
    assert query_result.metadata["bead_count"] == 1
    assert query_result.metadata["beads"][0]["title"] == "Need a searchable trace"

    await store.close()


@pytest.mark.asyncio
async def test_failed_tool_call_transit_records_one_peripheral_bead(tmp_path: Path) -> None:
    store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    service = ThreadRegistryService(store=store, workspace_root=tmp_path / "workspace")
    runtime = SimpleNamespace(thread_registry_service=service)
    failed = ToolCallTransit(
        chain_id="chain-1",
        call_id="call-failed",
        tool_name="workflow_status",
        entry_intent="Verify the current workflow state before answering.",
        trust_context="operator_requested",
        success=False,
        result_shape={"tags": ["failed", "blocked"], "output_chars": 31},
    )
    successful = failed.model_copy(update={"call_id": "call-ok", "success": True})

    result = await record_failed_tool_call_transit_thread_beads(
        runtime,
        session_id="s1",
        user_input="What is the workflow state?",
        tool_call_transits=[failed, successful],
    )

    assert result["available"] is True
    assert result["recorded"] == 1
    beads = await store.list_beads(source_kind=BeadSourceKind.TOOL_CALL_TRANSIT)
    assert len(beads) == 1
    assert beads[0].source_ref == "tool_call_transit:s1:chain-1:call-failed"
    assert beads[0].status == BeadStatus.PERIPHERAL
    assert beads[0].user_commissioned is False
    assert "workflow_status" in beads[0].summary
    assert "failed" in beads[0].summary

    await store.close()


@pytest.mark.asyncio
async def test_failed_tool_call_transit_backfill_is_idempotent(tmp_path: Path) -> None:
    store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    service = ThreadRegistryService(store=store, workspace_root=tmp_path / "workspace")
    runtime = SimpleNamespace(thread_registry_service=service)
    failed = ToolCallTransit(
        chain_id="chain-1",
        call_id="call-failed",
        tool_name="workflow_status",
        entry_intent="Verify the current workflow state before answering.",
        trust_context="operator_requested",
        success=False,
        result_shape={"tags": ["failed", "blocked"], "output_chars": 31},
    )

    await record_failed_tool_call_transit_thread_beads(
        runtime,
        session_id="s1",
        user_input="What is the workflow state?",
        tool_call_transits=[failed],
    )
    await record_failed_tool_call_transit_thread_beads(
        runtime,
        session_id="s1",
        user_input="What is the workflow state?",
        tool_call_transits=[failed],
    )

    beads = await store.list_beads(source_kind=BeadSourceKind.TOOL_CALL_TRANSIT)
    assert len(beads) == 1

    await store.close()


@pytest.mark.asyncio
async def test_bootstrap_store_bundle_includes_thread_registry(tmp_path: Path) -> None:
    config = BootstrapConfig(
        state_dir=tmp_path / "state",
        workspace_root=tmp_path / "repo",
    ).resolve_paths()
    identity = IdentityManager(IdentityStore(config.state_dir / "identity"))
    identity.load()
    tracer = Tracer(TelemetryStore(config.state_dir / "telemetry"))

    stores = await initialize_runtime_stores(
        config,
        identity=identity,
        tracer=tracer,
        stage=lambda _name, _meta=None: None,
    )

    assert stores.thread_registry_store is not None
    assert stores.thread_registry_store.path == config.state_dir / "thread_registry.db"

    await stores.memory.close()
    await stores.tasks.close()
    await stores.receipt_store.close()
    await stores.context_store.close()
    await stores.work_store.close()
    await stores.commitment_store.close()
    await stores.self_inspection_store.close()
    await stores.wellbeing_store.close()
    await stores.thread_registry_store.close()
    await stores.portfolio_store.close()


@pytest.mark.asyncio
async def test_daydream_thread_registry_records_incubated_thought_as_peripheral_bead(
    tmp_path: Path,
) -> None:
    store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    service = ThreadRegistryService(store=store, workspace_root=tmp_path / "workspace")

    class Runtime:
        thread_registry_service = service

        def _trace(self, _event, _payload=None):
            return None

    reflection = DaydreamReflection(
        spark_content="A loose thought about improving curiosity capture.",
        synthesis="The thought should be retrievable later without becoming an active obligation.",
        fascination_thread="Curiosity capture",
        thoughts=[
            DaydreamThought(
                kind=DaydreamThoughtKind.SYSTEM_INSIGHT,
                route=DaydreamThoughtRoute.INCUBATE,
                summary="Curiosity becomes more useful when an insight can rest in the periphery.",
                question="How can a later operator pick this up without guessing?",
                confidence=0.7,
            )
        ],
    )

    result = await record_runtime_daydream_thread_beads(Runtime(), reflection)

    assert result["available"] is True
    assert result["recorded"] == 1
    beads = await store.list_beads(source_kind=BeadSourceKind.DAYDREAM_REFLECTION)
    assert len(beads) == 1
    assert beads[0].status == BeadStatus.PERIPHERAL
    assert beads[0].user_commissioned is False
    assert beads[0].validation is not None
    assert beads[0].validation.should_create_task is False

    await store.close()


@pytest.mark.asyncio
async def test_thread_registry_api_lists_threads_and_beads(tmp_path: Path) -> None:
    store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    service = ThreadRegistryService(store=store, workspace_root=tmp_path / "workspace")
    bead = await service.ingest_autonomous_artifact(
        source_ref="workspace/if_i_am_the_operator/manuscript_draft.md",
        content="A registry bead should be visible to operator dashboards.",
        thread_title="Operator observability",
        title="Dashboard-visible bead",
        summary="The registry needs a read-only API so operator surfaces can inspect peripheral beads.",
    )
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            config=BootstrapConfig(
                state_dir=tmp_path / "state",
                workspace_root=tmp_path / "repo",
            ).resolve_paths(),
            event_bus=SimpleNamespace(),
        ),
        thread_registry_service=service,
    )
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        threads_response = await client.get("/api/thread-registry/threads")
        beads_response = await client.get(
            "/api/thread-registry/beads",
            params={"thread_anchor_id": bead.thread_anchor_id},
        )

    assert threads_response.status_code == 200
    assert beads_response.status_code == 200
    assert threads_response.json()["count"] == 1
    assert beads_response.json()["count"] == 1
    assert beads_response.json()["beads"][0]["title"] == "Dashboard-visible bead"

    await store.close()
