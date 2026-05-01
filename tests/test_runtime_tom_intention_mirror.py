from types import SimpleNamespace

import pytest

from opencas.execution.models import ExecutionStage, RepairTask
from opencas.runtime.tom_intention_mirror import reconcile_completed_runtime_intentions
from opencas.tom import BeliefSubject, Intention, IntentionStatus


@pytest.mark.asyncio
async def test_reconcile_completed_runtime_intentions_resolves_completed_active_work() -> None:
    intention = Intention(
        actor=BeliefSubject.SELF,
        content="already completed duplicate",
        status=IntentionStatus.ACTIVE,
        meta={"source": "active_work_dispatch"},
    )

    class FakeTom:
        def __init__(self) -> None:
            self.intentions = [intention]

        def list_intentions(self, actor=None, status=None):
            results = self.intentions
            if actor is not None:
                results = [item for item in results if item.actor == actor]
            if status is not None:
                results = [item for item in results if item.status == status]
            return results

        async def resolve_intention(self, content, status=IntentionStatus.COMPLETED):
            for item in self.intentions:
                if item.content == content:
                    item.status = status
                    return True
            return False

    class FakeTaskStore:
        async def list_all(self, limit=500):
            return [
                RepairTask(
                    objective="Already Completed Duplicate",
                    stage=ExecutionStage.DONE,
                    status="completed",
                )
            ]

    runtime = SimpleNamespace(
        tom=FakeTom(),
        ctx=SimpleNamespace(tasks=FakeTaskStore()),
        _trace=lambda *args, **kwargs: None,
    )

    resolved = await reconcile_completed_runtime_intentions(runtime)

    assert resolved == 1
    assert intention.status == IntentionStatus.COMPLETED


@pytest.mark.asyncio
async def test_reconcile_completed_runtime_intentions_abandons_failed_active_work() -> None:
    intention = Intention(
        actor=BeliefSubject.SELF,
        content="terminal failed work",
        status=IntentionStatus.ACTIVE,
        meta={"source": "active_work_dispatch"},
    )

    class FakeTom:
        def __init__(self) -> None:
            self.intentions = [intention]

        def list_intentions(self, actor=None, status=None):
            results = self.intentions
            if actor is not None:
                results = [item for item in results if item.actor == actor]
            if status is not None:
                results = [item for item in results if item.status == status]
            return results

        async def resolve_intention(self, content, status=IntentionStatus.COMPLETED):
            for item in self.intentions:
                if item.content == content:
                    item.status = status
                    return True
            return False

    class FakeTaskStore:
        async def list_all(self, limit=500):
            return [
                RepairTask(
                    objective="Terminal Failed Work",
                    stage=ExecutionStage.FAILED,
                    status="failed",
                )
            ]

    runtime = SimpleNamespace(
        tom=FakeTom(),
        ctx=SimpleNamespace(tasks=FakeTaskStore()),
        _trace=lambda *args, **kwargs: None,
    )

    resolved = await reconcile_completed_runtime_intentions(runtime)

    assert resolved == 1
    assert intention.status == IntentionStatus.ABANDONED
