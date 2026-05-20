from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencas.bootstrap.context import BootstrapContext
from opencas.runtime.scheduler import AgentScheduler


def test_bootstrap_context_accepts_recovery_components() -> None:
    context = BootstrapContext(
        config=SimpleNamespace(session_id=None),
        tracer=None,
        identity=None,
        memory=None,
        tasks=None,
        receipt_store=None,
        embeddings=None,
        somatic=None,
        llm=None,
        token_telemetry=None,
        event_bus=None,
        hook_bus=None,
        typed_hook_registry=None,
        ledger=None,
        shadow_registry=None,
        web_trust=None,
        plugin_trust=None,
        skill_registry=None,
        sandbox=None,
        readiness=None,
        context_store=None,
        context_proposal_store=None,
        work_store=None,
        project_orchestrator=None,
        relational=None,
        daydream_store=None,
        daydream_signal_store=None,
        conflict_store=None,
        somatic_store=None,
        executive=None,
        curation_store=None,
        harness=None,
        doctor=None,
        workspace_index=None,
        health_monitor=None,
        commitment_store=None,
        portfolio_store=None,
        tom_store=None,
        self_knowledge_registry=None,
        plugin_store=None,
        plugin_lifecycle=None,
        capability_registry=None,
        plan_store=None,
        schedule_store=None,
        schedule_service=None,
        recovery_ledger="ledger",
        recovery_coordinator="coordinator",
    )

    assert context.recovery_ledger == "ledger"
    assert context.recovery_coordinator == "coordinator"


@pytest.mark.asyncio
async def test_scheduler_recovery_loop_invokes_coordinator_when_allowed(monkeypatch) -> None:
    calls = []

    class FakeCoordinator:
        async def run_once(self, *, limit_per_store: int, max_actions: int):
            calls.append((limit_per_store, max_actions))
            return SimpleNamespace(scanned=1, submitted=1)

    scheduler = AgentScheduler(SimpleNamespace(recovery_coordinator=FakeCoordinator()))
    monkeypatch.setattr(scheduler, "_should_run_cycle", lambda: True)
    monkeypatch.setattr(scheduler, "_background_llm_block_reason", lambda **_: None)

    await scheduler._run_recovery_once_for_test()

    assert calls == [(100, 5)]


@pytest.mark.asyncio
async def test_scheduler_recovery_requires_quiet_baa_before_running(monkeypatch) -> None:
    calls = []
    gate_requirements = []

    class FakeCoordinator:
        async def run_once(self, *, limit_per_store: int, max_actions: int):
            calls.append((limit_per_store, max_actions))
            return SimpleNamespace(scanned=1, submitted=1)

    def block_reason(*, require_quiet_baa: bool = False, **_) -> str | None:
        gate_requirements.append(require_quiet_baa)
        if require_quiet_baa:
            return "baa_busy"
        return None

    scheduler = AgentScheduler(SimpleNamespace(recovery_coordinator=FakeCoordinator()))
    monkeypatch.setattr(scheduler, "_should_run_cycle", lambda: True)
    monkeypatch.setattr(scheduler, "_background_llm_block_reason", block_reason)

    await scheduler._run_recovery_once_for_test()

    assert gate_requirements == [True]
    assert calls == []
