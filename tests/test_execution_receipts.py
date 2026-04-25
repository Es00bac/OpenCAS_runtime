"""Tests for ExecutionReceiptStore."""

import pytest
import pytest_asyncio

from opencas.execution.models import (
    ExecutionPhase,
    ExecutionStage,
    PhaseRecord,
    RepairResult,
    RepairTask,
)
from opencas.execution.receipt_store import ExecutionReceiptStore


@pytest_asyncio.fixture
async def receipt_store(tmp_path):
    store = ExecutionReceiptStore(tmp_path / "receipts.db")
    await store.connect()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_receipt_store_round_trip(receipt_store: ExecutionReceiptStore) -> None:
    task = RepairTask(objective="test receipt")
    task.artifacts.append("plan:do something")
    task.phases.append(PhaseRecord(phase=ExecutionPhase.DETECT, success=True, output="file.txt"))
    task.checkpoint_commit = "abc123"
    result = RepairResult(
        task_id=task.task_id,
        success=True,
        stage=ExecutionStage.DONE,
        output="all good",
    )

    receipt = await receipt_store.save(task, result)
    assert receipt.task_id == task.task_id
    assert receipt.plan == "do something"
    assert receipt.success is True
    assert receipt.checkpoint_commit == "abc123"

    fetched = await receipt_store.get(str(receipt.receipt_id))
    assert fetched is not None
    assert fetched.objective == "test receipt"
    assert fetched.output == "all good"
    assert len(fetched.phases) == 1
    assert fetched.phases[0].phase == ExecutionPhase.DETECT


@pytest.mark.asyncio
async def test_receipt_store_list_by_task(receipt_store: ExecutionReceiptStore) -> None:
    task = RepairTask(objective="list test")
    result = RepairResult(
        task_id=task.task_id,
        success=False,
        stage=ExecutionStage.FAILED,
        output="nope",
    )
    receipt1 = await receipt_store.save(task, result)
    receipt2 = await receipt_store.save(task, result)

    items = await receipt_store.list_by_task(str(task.task_id))
    assert len(items) == 2
    ids = {str(r.receipt_id) for r in items}
    assert str(receipt1.receipt_id) in ids
    assert str(receipt2.receipt_id) in ids


@pytest.mark.asyncio
async def test_receipt_store_verification_result(receipt_store: ExecutionReceiptStore) -> None:
    task = RepairTask(objective="verify test")
    task.phases.append(PhaseRecord(phase=ExecutionPhase.VERIFY, success=True, output="ok"))
    result = RepairResult(
        task_id=task.task_id,
        success=True,
        stage=ExecutionStage.DONE,
        output="done",
    )
    receipt = await receipt_store.save(task, result)
    assert receipt.verification_result is True

    fetched = await receipt_store.get(str(receipt.receipt_id))
    assert fetched is not None
    assert fetched.verification_result is True


@pytest.mark.asyncio
async def test_receipt_store_list_recent(receipt_store: ExecutionReceiptStore) -> None:
    t1 = RepairTask(objective="recent one")
    t2 = RepairTask(objective="recent two")
    r1 = RepairResult(task_id=t1.task_id, success=True, stage=ExecutionStage.DONE, output="ok")
    r2 = RepairResult(task_id=t2.task_id, success=False, stage=ExecutionStage.FAILED, output="no")
    await receipt_store.save(t1, r1)
    await receipt_store.save(t2, r2)

    recent = await receipt_store.list_recent(limit=2)
    assert len(recent) == 2
    assert {item.objective for item in recent} == {"recent one", "recent two"}
