from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencas.bootstrap import BootstrapConfig
from opencas.bootstrap.pipeline_stores import initialize_runtime_stores
from opencas.identity import IdentityManager, IdentityStore
from opencas.runtime.runtime_setup import initialize_runtime_execution
from opencas.telemetry import TelemetryStore, Tracer


@pytest.mark.asyncio
async def test_runtime_store_bundle_includes_context_proposal_store(tmp_path) -> None:
    config = BootstrapConfig(state_dir=tmp_path / "state").resolve_paths()
    identity = IdentityManager(IdentityStore(config.state_dir / "identity"))
    identity.load()
    tracer = Tracer(TelemetryStore(config.state_dir / "telemetry"))

    stores = await initialize_runtime_stores(
        config,
        identity=identity,
        tracer=tracer,
        stage=lambda _name, _meta=None: None,
    )
    try:
        assert stores.context_proposal_store is not None
        assert stores.context_proposal_store.path == config.state_dir / "context_proposals.db"
    finally:
        await stores.memory.close()
        await stores.tasks.close()
        await stores.receipt_store.close()
        await stores.context_store.close()
        await stores.context_proposal_store.close()
        await stores.work_store.close()
        await stores.commitment_store.close()
        await stores.self_inspection_store.close()
        await stores.cognitive_state_store.close()
        await stores.wellbeing_store.close()
        await stores.dream_store.close()
        await stores.proof_store.close()
        await stores.thread_registry_store.close()
        await stores.portfolio_store.close()


def test_initialize_runtime_execution_wires_truth_arbiter_and_packet_builder() -> None:
    proposal_store = SimpleNamespace()
    runtime = SimpleNamespace(
        tracer=None,
        ctx=SimpleNamespace(hook_bus=None),
        plugin_lifecycle=None,
        capability_registry=None,
        _register_default_tools=lambda: None,
        _register_skills=lambda: None,
        orchestrator=SimpleNamespace(),
        llm=None,
        approval=SimpleNamespace(),
    )
    context = SimpleNamespace(
        capability_registry=None,
        plugin_lifecycle=None,
        tasks=None,
        event_bus=None,
        receipt_store=None,
        memory=None,
        embeddings=None,
        context_proposal_store=proposal_store,
    )

    initialize_runtime_execution(runtime, context)

    assert runtime.truth_arbiter.runtime is runtime
    assert runtime.context_proposals is proposal_store
    assert runtime.context_packet_builder.truth_arbiter is runtime.truth_arbiter
    assert runtime.context_packet_builder.proposal_store is proposal_store
