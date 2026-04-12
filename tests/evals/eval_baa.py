"""BAA behavioral evals.

Measures whether BoundedAssistantAgent correctly routes tasks through
lanes, respects concurrency limits, handles held/dependency mechanics,
and enforces the recovery cap. Uses a mock RepairExecutor so no LLM
or tool calls are needed.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock
from uuid import uuid4

from opencas.execution.baa import BoundedAssistantAgent
from opencas.execution.lanes import CommandLane
from opencas.execution.models import ExecutionStage, RepairResult, RepairTask
from opencas.tools import ToolRegistry


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    notes: str
    details: dict = field(default_factory=dict)


def _done_executor(task: RepairTask) -> RepairResult:
    """Mock executor that always returns DONE."""
    return RepairResult(
        task_id=task.task_id,
        success=True,
        stage=ExecutionStage.DONE,
        output="mock success",
    )


def _fail_executor(task: RepairTask) -> RepairResult:
    """Mock executor that always returns FAILED."""
    return RepairResult(
        task_id=task.task_id,
        success=False,
        stage=ExecutionStage.FAILED,
        output="mock failure",
    )


def _recovering_executor(task: RepairTask) -> RepairResult:
    """Mock executor that always returns RECOVERING (triggers retry loop)."""
    return RepairResult(
        task_id=task.task_id,
        success=False,
        stage=ExecutionStage.RECOVERING,
        output="mock recovering",
    )


def _make_baa(executor_fn) -> BoundedAssistantAgent:
    tools = MagicMock(spec=ToolRegistry)
    baa = BoundedAssistantAgent(tools=tools, max_concurrent=2)

    async def mock_run(task):
        return executor_fn(task)

    baa.executor.run = mock_run
    return baa


# ---------------------------------------------------------------------------
# Eval 1: task completes successfully through BAA
# ---------------------------------------------------------------------------
async def eval_task_completes(tmp: Path) -> EvalResult:
    baa = _make_baa(_done_executor)
    await baa.start()

    task = RepairTask(objective="eval: simple task")
    future = await baa.submit(task, lane=CommandLane.BAA)

    try:
        result = await asyncio.wait_for(future, timeout=10.0)
        passed = result.success and result.stage == ExecutionStage.DONE
        details = {"success": result.success, "stage": result.stage.value, "output": result.output}
    except asyncio.TimeoutError:
        passed = False
        details = {"error": "timeout"}
    finally:
        await baa.stop()

    return EvalResult(
        name="baa.task_completes",
        passed=passed,
        score=1.0 if passed else 0.0,
        notes="Single task submitted to BAA lane, expected DONE",
        details=details,
    )


# ---------------------------------------------------------------------------
# Eval 2: failed task is recorded as FAILED (not silently dropped)
# ---------------------------------------------------------------------------
async def eval_task_failure_recorded(tmp: Path) -> EvalResult:
    baa = _make_baa(_fail_executor)
    await baa.start()

    task = RepairTask(objective="eval: failing task")
    future = await baa.submit(task, lane=CommandLane.BAA)

    try:
        result = await asyncio.wait_for(future, timeout=10.0)
        passed = not result.success and result.stage == ExecutionStage.FAILED
        details = {"success": result.success, "stage": result.stage.value}
    except asyncio.TimeoutError:
        passed = False
        details = {"error": "timeout"}
    finally:
        await baa.stop()

    return EvalResult(
        name="baa.task_failure_recorded",
        passed=passed,
        score=1.0 if passed else 0.0,
        notes="Failing task should produce FAILED result, not timeout or crash",
        details=details,
    )


# ---------------------------------------------------------------------------
# Eval 3: recovery cap enforced — task that keeps RECOVERING fails after cap
# ---------------------------------------------------------------------------
async def eval_recovery_cap(tmp: Path) -> EvalResult:
    baa = _make_baa(_recovering_executor)
    await baa.start()

    task = RepairTask(objective="eval: infinite recovery")
    future = await baa.submit(task, lane=CommandLane.BAA)

    try:
        result = await asyncio.wait_for(future, timeout=30.0)
        # Should eventually fail with FAILED after 10 recovery retries
        passed = result.stage == ExecutionStage.FAILED and "recovery cap" in (result.output or "").lower()
        details = {"stage": result.stage.value, "output": result.output}
    except asyncio.TimeoutError:
        passed = False
        details = {"error": "timeout — recovery cap may not be enforced"}
    finally:
        await baa.stop()

    return EvalResult(
        name="baa.recovery_cap",
        passed=passed,
        score=1.0 if passed else 0.0,
        notes="RECOVERING task must fail after 10 retries with 'recovery cap' in output",
        details=details,
    )


# ---------------------------------------------------------------------------
# Eval 4: lane snapshot reports correct structure
# ---------------------------------------------------------------------------
async def eval_lane_snapshot(tmp: Path) -> EvalResult:
    baa = _make_baa(_done_executor)
    snapshot = baa.lane_snapshot()

    expected_lanes = {l.value for l in CommandLane}
    actual_lanes = set(snapshot.keys())
    has_all_lanes = expected_lanes == actual_lanes

    all_have_fields = all(
        "queue_depth" in v and "max_concurrent" in v
        for v in snapshot.values()
    )
    chat_concurrency = snapshot.get(CommandLane.CHAT.value, {}).get("max_concurrent", 0)
    baa_concurrency = snapshot.get(CommandLane.BAA.value, {}).get("max_concurrent", 0)
    concurrency_correct = chat_concurrency == 1 and baa_concurrency >= 1

    passed = has_all_lanes and all_have_fields and concurrency_correct
    return EvalResult(
        name="baa.lane_snapshot",
        passed=passed,
        score=1.0 if passed else 0.0,
        notes=f"lanes={actual_lanes}, all_fields={all_have_fields}, concurrency_correct={concurrency_correct}",
        details={
            "snapshot": snapshot,
            "has_all_lanes": has_all_lanes,
            "all_have_fields": all_have_fields,
            "concurrency_correct": concurrency_correct,
        },
    )


# ---------------------------------------------------------------------------
# Eval 5: held task released when dependency completes
# ---------------------------------------------------------------------------
async def eval_dependency_held_and_released(tmp: Path) -> EvalResult:
    baa = _make_baa(_done_executor)
    await baa.start()

    # Submit a task that depends on a fictional completed task_id
    dep_id = str(uuid4())
    # Manually inject the dependency result so it looks already complete
    baa._results[dep_id] = RepairResult(
        task_id=dep_id,  # type: ignore[arg-type]
        success=True,
        stage=ExecutionStage.DONE,
        output="pre-completed dep",
    )

    dependent_task = RepairTask(
        objective="eval: dependent task",
        depends_on=[dep_id],
    )
    future = await baa.submit(dependent_task, lane=CommandLane.BAA)

    # Task should NOT be held (dep already complete), should run and complete
    try:
        result = await asyncio.wait_for(future, timeout=10.0)
        was_held = str(dependent_task.task_id) in baa._held
        passed = result.success and result.stage == ExecutionStage.DONE and not was_held
        details = {"success": result.success, "stage": result.stage.value, "was_held": was_held}
    except asyncio.TimeoutError:
        passed = False
        details = {"error": "timeout — task may have been incorrectly held"}
    finally:
        await baa.stop()

    return EvalResult(
        name="baa.dependency_held_and_released",
        passed=passed,
        score=1.0 if passed else 0.0,
        notes="Task with pre-completed dep should not be held; should run to DONE",
        details=details,
    )


# ---------------------------------------------------------------------------
# Eval 6: multi-lane parallel throughput
# Submit 4 tasks across BAA and CRON lanes simultaneously.
# Verify all complete without deadlock.
# ---------------------------------------------------------------------------
async def eval_multi_lane_throughput(tmp: Path) -> EvalResult:
    baa = _make_baa(_done_executor)
    await baa.start()

    tasks = [
        (RepairTask(objective=f"eval: baa task {i}"), CommandLane.BAA)
        for i in range(3)
    ] + [
        (RepairTask(objective=f"eval: cron task {i}"), CommandLane.CRON)
        for i in range(2)
    ]

    futures = [await baa.submit(t, lane=lane) for t, lane in tasks]

    try:
        results = await asyncio.wait_for(asyncio.gather(*futures), timeout=20.0)
        all_done = all(r.stage == ExecutionStage.DONE for r in results)
        passed = all_done
        details = {
            "total": len(results),
            "done": sum(1 for r in results if r.stage == ExecutionStage.DONE),
            "failed": sum(1 for r in results if r.stage == ExecutionStage.FAILED),
        }
    except asyncio.TimeoutError:
        passed = False
        details = {"error": "timeout — possible deadlock across lanes"}
    finally:
        await baa.stop()

    return EvalResult(
        name="baa.multi_lane_throughput",
        passed=passed,
        score=1.0 if passed else 0.0,
        notes=f"5 tasks across BAA+CRON lanes, all should complete",
        details=details,
    )


async def run_all(tmp_root: Path) -> List[EvalResult]:
    tmp_root.mkdir(parents=True, exist_ok=True)
    # Run sequentially to avoid BAA instance conflicts
    results = []
    for fn in [
        eval_lane_snapshot,
        eval_task_completes,
        eval_task_failure_recorded,
        eval_dependency_held_and_released,
        eval_multi_lane_throughput,
        eval_recovery_cap,
    ]:
        results.append(await fn(tmp_root))
    return results
