"""Tests for TaskLifecycleMachine and stage transitions."""

import pytest

from opencas.execution.lifecycle import LifecycleStage, TaskLifecycleMachine


def test_valid_spark_to_note() -> None:
    transition = TaskLifecycleMachine.transition(
        task_id="t1", from_stage=LifecycleStage.SPARK, to_stage=LifecycleStage.NOTE
    )
    assert transition.from_stage == LifecycleStage.SPARK
    assert transition.to_stage == LifecycleStage.NOTE


def test_valid_queued_to_executing() -> None:
    transition = TaskLifecycleMachine.transition(
        task_id="t1", from_stage=LifecycleStage.QUEUED, to_stage=LifecycleStage.EXECUTING
    )
    assert transition.to_stage == LifecycleStage.EXECUTING


def test_valid_executing_to_done_path() -> None:
    t1 = TaskLifecycleMachine.transition(
        task_id="t1", from_stage=LifecycleStage.EXECUTING, to_stage=LifecycleStage.VERIFYING
    )
    t2 = TaskLifecycleMachine.transition(
        task_id="t1", from_stage=LifecycleStage.VERIFYING, to_stage=LifecycleStage.REPORTING
    )
    t3 = TaskLifecycleMachine.transition(
        task_id="t1", from_stage=LifecycleStage.REPORTING, to_stage=LifecycleStage.DONE
    )
    assert t3.to_stage == LifecycleStage.DONE


def test_invalid_transition_raises() -> None:
    with pytest.raises(ValueError):
        TaskLifecycleMachine.transition(
            task_id="t1", from_stage=LifecycleStage.DONE, to_stage=LifecycleStage.EXECUTING
        )


def test_retry_from_failed() -> None:
    transition = TaskLifecycleMachine.transition(
        task_id="t1", from_stage=LifecycleStage.FAILED, to_stage=LifecycleStage.QUEUED,
        reason="retry scheduled"
    )
    assert transition.to_stage == LifecycleStage.QUEUED
    assert transition.reason == "retry scheduled"


def test_executing_to_queued_for_retry() -> None:
    # BAA retry semantics: executing task may go back to queued
    transition = TaskLifecycleMachine.transition(
        task_id="t1", from_stage=LifecycleStage.EXECUTING, to_stage=LifecycleStage.QUEUED
    )
    assert transition.to_stage == LifecycleStage.QUEUED


def test_needs_approval_resolved() -> None:
    t1 = TaskLifecycleMachine.transition(
        task_id="t1", from_stage=LifecycleStage.NEEDS_APPROVAL, to_stage=LifecycleStage.EXECUTING
    )
    assert t1.to_stage == LifecycleStage.EXECUTING


def test_is_valid_helper() -> None:
    assert TaskLifecycleMachine.is_valid(LifecycleStage.SPARK, LifecycleStage.ARTIFACT) is True
    assert TaskLifecycleMachine.is_valid(LifecycleStage.DONE, LifecycleStage.FAILED) is False
