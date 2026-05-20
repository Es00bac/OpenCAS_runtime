"""Tests for the executive state tracker."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.autonomy import WorkObject, WorkStage
from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.autonomy.executive import ExecutiveState
from opencas.autonomy.work_store import WorkStore
from opencas.identity import IdentityManager, IdentityStore
from opencas.somatic import SomaticManager
from opencas.telemetry import TelemetryStore, Tracer


@pytest.fixture
def identity(tmp_path: Path):
    store = IdentityStore(tmp_path / "identity")
    mgr = IdentityManager(store)
    mgr.load()
    return mgr


@pytest.fixture
def somatic(tmp_path: Path):
    return SomaticManager(tmp_path / "somatic.json")


@pytest.fixture
def tracer(tmp_path: Path):
    store = TelemetryStore(tmp_path / "telemetry")
    return Tracer(store)


@pytest.fixture
def executive(identity, somatic, tracer):
    return ExecutiveState(identity=identity, somatic=somatic, tracer=tracer)


def test_intention_set(executive: ExecutiveState) -> None:
    executive.set_intention("plan the day")
    assert executive.intention == "plan the day"


def test_intention_set_syncs_placeholder_identity_intention(
    executive: ExecutiveState,
    identity: IdentityManager,
) -> None:
    identity.self_model.current_intention = "establish trust and understanding"
    identity.save()

    executive.set_intention("plan the day")

    assert identity.self_model.current_intention == "plan the day"


def test_intention_set_preserves_explicit_identity_intention(
    executive: ExecutiveState,
    identity: IdentityManager,
) -> None:
    identity.self_model.current_intention = "protect an explicitly chosen intention"
    identity.save()

    executive.set_intention("plan the day")

    assert identity.self_model.current_intention == "protect an explicitly chosen intention"


def test_tasklist_live_objective_replaces_stale_identity_intention(
    executive: ExecutiveState,
    identity: IdentityManager,
) -> None:
    identity.self_model.current_intention = "Convert the Writing Project truth-repair lens."
    identity.save()

    executive.set_intention(
        "Whole-system standards conformance pass",
        source="tasklist_live_objective",
    )

    assert identity.self_model.current_intention == "Whole-system standards conformance pass"


def test_set_intention_from_work_preserves_explicit_objective(executive: ExecutiveState) -> None:
    executive.set_intention("Continuity surface reconciliation decision bead")

    executive.set_intention_from_work(
        WorkObject(content="A quiet task beacon that collapses dozens of build/test fragments")
    )

    assert executive.intention == "Continuity surface reconciliation decision bead"
    assert executive.intention_source == "explicit"


def test_set_intention_from_work_replaces_prior_work_driven_focus(executive: ExecutiveState) -> None:
    executive.set_intention_from_work(WorkObject(content="first active work"))
    executive.set_intention_from_work(WorkObject(content="second active work"))

    assert executive.intention == "second active work"
    assert executive.intention_source == "active_work"


def test_load_snapshot_syncs_placeholder_identity_intention(
    identity: IdentityManager,
    somatic: SomaticManager,
    tracer: Tracer,
    tmp_path: Path,
) -> None:
    identity.self_model.current_intention = "establish trust and understanding"
    identity.save()

    executive = ExecutiveState(identity=identity, somatic=somatic, tracer=tracer)
    snapshot_path = tmp_path / "executive.json"
    snapshot_path.write_text(
        '{'
        '"updated_at":"2026-04-22T00:00:00+00:00",'
        '"intention":"revising the story and doing what you need to to complete it",'
        '"intention_source":"active_work",'
        '"active_goals":[],'
        '"queue_metadata":[]'
        '}',
        encoding="utf-8",
    )

    executive.load_snapshot(snapshot_path)

    assert identity.self_model.current_intention == "revising the story and doing what you need to to complete it"
    assert executive.intention_source == "active_work"


def test_load_snapshot_rejects_loaded_self_referential_suppression_metadata(
    identity: IdentityManager,
    somatic: SomaticManager,
    tracer: Tracer,
    tmp_path: Path,
) -> None:
    recursive_goal = (
        "The browser_click blocked memory is in soft focus; do not lose it, "
        "but do not activate it now."
    )
    user_goal = "verify tsconfig"
    snapshot_path = tmp_path / "executive.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "updated_at": "2026-05-06T00:00:00+00:00",
                "intention": "repair parked goal hygiene",
                "intention_source": "explicit",
                "active_goals": [],
                "parked_goals": [user_goal, recursive_goal],
                "parked_goal_reasons": {
                    user_goal: "evidence_deferred",
                    recursive_goal: "low_divergence_reframe",
                },
                "parked_goal_metadata": {
                    user_goal: {
                        "reason": "evidence_deferred",
                        "source_artifact": "tsconfig",
                    },
                    recursive_goal: {
                        "reason": "low_divergence_reframe",
                        "source_artifact": recursive_goal,
                        "failed_framings": [recursive_goal],
                        "duplicate_of_task_id": "task-1",
                    },
                },
                "queue_metadata": [],
            }
        ),
        encoding="utf-8",
    )
    executive = ExecutiveState(identity=identity, somatic=somatic, tracer=tracer)

    executive.load_snapshot(snapshot_path)

    assert user_goal in executive.parked_goals
    assert recursive_goal not in executive.parked_goals
    assert recursive_goal not in executive.parked_goal_metadata
    rejected = executive.consume_suppressed_parked_goal_rejections()
    assert len(rejected) == 1
    assert rejected[0]["goal"] == recursive_goal
    assert rejected[0]["metadata"]["duplicate_of_task_id"] == "task-1"
    assert executive.consume_suppressed_parked_goal_rejections() == []


def test_goals_sync_to_identity(executive: ExecutiveState, identity: IdentityManager) -> None:
    executive.add_goal("learn rust")
    assert "learn rust" in identity.self_model.current_goals
    executive.remove_goal("learn rust")
    assert "learn rust" not in identity.self_model.current_goals


def test_park_goal_moves_active_goal_to_parked_with_wake_condition(
    executive: ExecutiveState,
    identity: IdentityManager,
) -> None:
    executive.add_goal("verify tsconfig")

    changed = executive.park_goal(
        "verify tsconfig",
        reason="evidence_deferred",
        wake_trigger="TypeScript failure, tsconfig file change, objective dependency, or direct user request",
        source_artifact="tsconfig",
    )

    assert changed is True
    assert "verify tsconfig" not in executive.active_goals
    assert "verify tsconfig" in executive.parked_goals
    assert "verify tsconfig" not in identity.self_model.current_goals
    metadata = executive.parked_goal_metadata["verify tsconfig"]
    assert metadata["reason"] == "evidence_deferred"
    assert metadata["wake_trigger"] == (
        "TypeScript failure, tsconfig file change, objective dependency, or direct user request"
    )
    assert metadata["source_artifact"] == "tsconfig"


def test_park_goal_merges_reframe_details_into_metadata(executive: ExecutiveState) -> None:
    changed = executive.park_goal(
        "continue creative_writing",
        reason="low_divergence_reframe",
        details={
            "reframe_hint": "Resume from workspace/writing/4246/story_4246.md with one narrow edit.",
            "failed_framings": [
                "Continue writing project 4246 from the existing manuscript.",
            ],
            "reframe_rule": "Do not retry this line with cosmetic rewording.",
        },
    )

    assert changed is True
    metadata = executive.parked_goal_metadata["continue creative_writing"]
    assert metadata["reason"] == "low_divergence_reframe"
    assert metadata["wake_trigger"] == (
        "fresh evidence, relevant artifact change, materially different framing, or direct user request"
    )
    assert metadata["reframe_hint"] == (
        "Resume from workspace/writing/4246/story_4246.md with one narrow edit."
    )
    assert metadata["failed_framings"] == [
        "Continue writing project 4246 from the existing manuscript.",
    ]
    assert metadata["reframe_rule"] == "Do not retry this line with cosmetic rewording."


def test_park_goal_rejects_self_referential_suppression_metadata(
    executive: ExecutiveState,
) -> None:
    goal = "browser_click: Click at x=1590, y=890 to click lightsaber"
    executive.add_goal(goal)

    changed = executive.park_goal(
        goal,
        reason="low_divergence_reframe",
        source_artifact=goal,
        details={
            "failed_framings": [goal],
            "duplicate_of_task_id": "task-1",
            "last_result_stage": "failed",
        },
    )

    assert changed is False
    assert goal in executive.active_goals
    assert goal not in executive.parked_goals
    assert goal not in executive.parked_goal_metadata


def test_park_goal_rejects_duplicate_low_divergence_objective_self_reference(
    executive: ExecutiveState,
) -> None:
    goal = "browser_click blocked memory should not be retried as a goal"

    changed = executive.park_goal(
        goal,
        reason="duplicate_low_divergence_objective",
        source_artifact=goal,
        details={
            "failed_framings": [goal],
            "duplicate_of_task_id": "task-2",
        },
    )

    assert changed is False
    assert goal not in executive.parked_goals
    assert goal not in executive.parked_goal_metadata


def test_park_goal_archives_extra_low_divergence_reframes(executive: ExecutiveState) -> None:
    for index in range(3):
        executive.park_goal(
            f"duplicate reframe {index}",
            reason="low_divergence_reframe",
            details={
                "reframe_hint": f"Use materially different evidence path {index}.",
                "failed_framings": [f"duplicate reframe {index}"],
            },
        )

    assert executive.parked_goals == ["duplicate reframe 2"]
    assert "duplicate reframe 0" in executive.archived_parked_goals
    assert "duplicate reframe 1" in executive.archived_parked_goals
    archived = executive.archived_parked_goal_metadata["duplicate reframe 0"]
    assert archived["reason"] == "low_divergence_reframe"
    assert archived["archive_reason"] == "low_divergence_reframe_limit"
    assert archived["reframe_hint"] == "Use materially different evidence path 0."


def test_refresh_structural_load_applies_parked_pressure_after_somatic_attach(
    identity: IdentityManager,
    somatic: SomaticManager,
    tracer: Tracer,
) -> None:
    executive = ExecutiveState(identity=identity, somatic=None, tracer=tracer)
    for goal in (
        "repair /package",
        "repair /npx",
        "test /jsonlogger",
        "build /reports/problems/problems-report",
        "build /outputs/apk/debug/app-debug",
        "verify tsconfig",
    ):
        executive.park_goal(goal)

    executive.somatic = somatic
    executive.refresh_structural_load()

    assert somatic.state.somatic_tag == "continuity_pressure"


def test_enqueue_and_dequeue(executive: ExecutiveState) -> None:
    w1 = WorkObject(content="a", stage=WorkStage.SPARK, promotion_score=1.0)
    w2 = WorkObject(content="b", stage=WorkStage.NOTE, promotion_score=2.0)
    assert executive.enqueue(w1) is True
    assert executive.enqueue(w2) is True
    assert executive.capacity_remaining == 3

    highest = executive.dequeue()
    assert highest is not None
    assert highest.work_id == w2.work_id  # higher score first


def test_dequeue_prioritizes_user_facing_commitment_work(executive: ExecutiveState) -> None:
    linked = WorkObject(content="linked", stage=WorkStage.MICRO_TASK, promotion_score=0.3)
    background = WorkObject(content="background", stage=WorkStage.MICRO_TASK, promotion_score=0.8)
    commitment = Commitment(
        content="Return to linked work",
        priority=9.0,
        meta={"source": "assistant_response"},
    )
    ExecutiveState.apply_commitment_execution_bias(linked, commitment)

    assert executive.enqueue(background) is True
    assert executive.enqueue(linked) is True

    highest = executive.dequeue()
    assert highest is not None
    assert highest.work_id == linked.work_id


def test_capacity_limit(executive: ExecutiveState) -> None:
    for i in range(5):
        assert executive.enqueue(WorkObject(content=str(i))) is True
    rejected = WorkObject(content="overflow")
    assert executive.enqueue(rejected) is False
    assert executive.is_overloaded is True
    assert executive.capacity_remaining == 0


def test_recommend_pause_overload(executive: ExecutiveState) -> None:
    executive._max_queue_depth = 20
    for i in range(14):
        executive.enqueue(WorkObject(content=str(i)))
    assert executive.recommend_pause() is True


def test_recommend_pause_fatigue(executive: ExecutiveState, somatic: SomaticManager) -> None:
    somatic.set_fatigue(0.8)
    assert executive.recommend_pause() is True


def test_recommend_pause_operator_rest_window(
    executive: ExecutiveState,
    somatic: SomaticManager,
) -> None:
    somatic.set_fatigue(0.2)
    somatic.set_rest_until(datetime.now(timezone.utc) + timedelta(minutes=10))
    assert executive.pause_reason() == "operator_rest"
    assert executive.recommend_pause() is True


def test_snapshot(executive: ExecutiveState) -> None:
    executive.add_goal("test goal")
    executive.add_goal("repair /package")
    executive.set_intention("plan the day")
    executive.enqueue(WorkObject(content="work", stage=WorkStage.SPARK))
    snap = executive.snapshot()
    assert snap["intention"] == "plan the day"
    assert snap["intention_source"] == "explicit"
    assert snap["active_goals"] == ["test goal"]
    assert snap["parked_goals"] == []
    assert snap["parked_goal_count"] == 0
    assert snap["archived_parked_goal_count"] == 0
    assert snap["capacity_remaining"] == 4
    assert snap["queue_size"] == 1
    assert snap["queue_stages"] == ["spark"]
    assert "timestamp" in snap


def test_enqueue_promoted_work_populates_active_goal(executive: ExecutiveState) -> None:
    work = WorkObject(
        content="Build a bounded provenance consumer validation note.",
        stage=WorkStage.MICRO_TASK,
    )

    accepted = executive.enqueue(work)

    assert accepted is True
    assert executive.active_goals == [
        "Advance promoted work: Build a bounded provenance consumer validation note."
    ]


def test_reconcile_work_update_removes_demoted_work_from_live_focus(
    executive: ExecutiveState,
    identity: IdentityManager,
) -> None:
    work = WorkObject(
        content="Build a bounded provenance consumer validation note.",
        stage=WorkStage.MICRO_TASK,
    )
    assert executive.enqueue(work) is True
    assert executive.snapshot()["queue_size"] == 1

    work.stage = WorkStage.NOTE
    changed = executive.reconcile_work_update(work)

    assert changed is True
    assert executive.snapshot()["queue_size"] == 0
    assert executive.active_goals == []
    assert identity.self_model.current_goals == []


def test_enqueue_rejects_stale_promoted_project_fragment(executive: ExecutiveState) -> None:
    work = WorkObject(
        content="Return to project: from individual chapters",
        stage=WorkStage.MICRO_TASK,
    )

    accepted = executive.enqueue(work)

    assert accepted is False
    assert executive.active_goals == []
    assert executive.snapshot()["queue_size"] == 0
    assert work.blocked_by == ["goal_hygiene:stale_promoted_work_fragment"]
    assert work.meta["goal_hygiene_rejected_reason"] == "stale_promoted_work_fragment"


@pytest.mark.asyncio
async def test_restore_queue_populates_active_goal(tmp_path: Path, identity: IdentityManager) -> None:
    store = WorkStore(tmp_path / "work.db")
    await store.connect()
    work = WorkObject(
        content="Write the live stabilization validation report.",
        stage=WorkStage.MICRO_TASK,
    )
    await store.save(work)
    executive = ExecutiveState(identity=identity, work_store=store)

    restored = await executive.restore_queue()

    assert restored == 1
    assert executive.active_goals == [
        "Advance promoted work: Write the live stabilization validation report."
    ]
    await store.close()


@pytest.mark.asyncio
async def test_restore_queue_blocks_stale_promoted_project_fragment(
    tmp_path: Path,
    identity: IdentityManager,
) -> None:
    store = WorkStore(tmp_path / "work.db")
    await store.connect()
    work = WorkObject(
        content="Return to project: output showing its contents",
        stage=WorkStage.MICRO_TASK,
    )
    await store.save(work)
    executive = ExecutiveState(identity=identity, work_store=store)

    restored = await executive.restore_queue()
    refreshed = await store.get(str(work.work_id))

    assert restored == 0
    assert executive.active_goals == []
    assert executive.snapshot()["queue_size"] == 0
    assert refreshed is not None
    assert refreshed.blocked_by == ["goal_hygiene:stale_promoted_work_fragment"]
    assert refreshed.meta["goal_hygiene_rejected_source"] == "restore_queue"
    await store.close()


def test_structural_load_updates_somatic_from_queue_pressure(
    executive: ExecutiveState,
    somatic: SomaticManager,
) -> None:
    executive._max_queue_depth = 20
    for i in range(6):
        assert executive.enqueue(WorkObject(content=f"work-{i}", stage=WorkStage.MICRO_TASK)) is True

    assert somatic.state.tension > 0.0
    assert somatic.state.arousal > 0.0
    assert somatic.state.somatic_tag in {"task_pressure", "crowded"}


def test_queue_metadata_marks_one_item_active_and_labels_the_rest(executive: ExecutiveState) -> None:
    top = WorkObject(content="top priority", stage=WorkStage.MICRO_TASK, promotion_score=3.0)
    backup = WorkObject(content="backup priority", stage=WorkStage.PROJECT_SEED, promotion_score=2.0)
    tail = WorkObject(content="tail priority", stage=WorkStage.PROJECT, promotion_score=1.0)

    assert executive.enqueue(tail) is True
    assert executive.enqueue(backup) is True
    assert executive.enqueue(top) is True

    snap = executive.snapshot()
    assert [item["state"] for item in snap["queue_metadata"]] == ["active", "held", "held"]
    assert [item["state_label"] for item in snap["queue_metadata"]] == ["Active", "Held", "Held"]
    assert [item["role"] for item in snap["queue_metadata"]] == ["active", "held", "held"]
    assert [item["role_label"] for item in snap["queue_metadata"]] == ["Active", "Held", "Held"]
    assert [item["bearing"] for item in snap["queue_metadata"]] == ["ready", "queued", "waiting"]
    assert [item["bearing_label"] for item in snap["queue_metadata"]] == ["Ready", "Queued", "Waiting"]
    assert [item["is_active"] for item in snap["queue_metadata"]] == [True, False, False]
    assert snap["queue_metadata"][0]["work_id"] == str(top.work_id)
    assert snap["queue_metadata"][0]["title"] == "top priority"
    assert snap["queue_metadata"][1]["work_id"] == str(backup.work_id)
    assert snap["queue_metadata"][2]["work_id"] == str(tail.work_id)

    dequeued = executive.dequeue()
    assert dequeued is not None
    assert dequeued.work_id == top.work_id

    resumed = executive.snapshot()
    assert [item["state"] for item in resumed["queue_metadata"]] == ["active", "held"]
    assert [item["role"] for item in resumed["queue_metadata"]] == ["active", "held"]
    assert [item["bearing"] for item in resumed["queue_metadata"]] == ["ready", "waiting"]
    assert [item["bearing_label"] for item in resumed["queue_metadata"]] == ["Ready", "Waiting"]
    assert resumed["queue_metadata"][0]["work_id"] == str(backup.work_id)
    assert resumed["queue_metadata"][0]["is_active"] is True


@pytest.mark.asyncio
async def test_workflow_status_uses_canonical_queue_metadata(identity: IdentityManager) -> None:
    from opencas.runtime.status_views import build_workflow_status

    class FakeExecutive:
        def __init__(self) -> None:
            self.intention = "focus on the queue"
            self.active_goals = ["keep one active"]
            self.capacity_remaining = 4

        def recommend_pause(self) -> bool:
            return False

        def queue_metadata(self):
            return [
                {"work_id": "w1", "state": "active", "bearing": "ready"},
                {"work_id": "w2", "state": "held", "bearing": "queued"},
                {"work_id": "w3", "state": "held", "bearing": "waiting"},
            ]

    runtime = SimpleNamespace(
        agent_profile=SimpleNamespace(model_dump=lambda mode="json": {"name": "fake"}),
        executive=FakeExecutive(),
        commitment_store=None,
        ctx=SimpleNamespace(
            work_store=None,
            plan_store=None,
            receipt_store=None,
        ),
        project_resume=None,
        _last_consolidation_result=None,
    )

    workflow = await build_workflow_status(runtime)

    assert workflow["executive"]["queued_work_count"] == 3
    assert workflow["executive"]["queue"]["counts"] == {
        "total": 3,
        "active": 1,
        "held": 2,
        "ready": 1,
        "queued": 1,
        "waiting": 1,
    }
    assert workflow["executive"]["queue"]["items"][0]["state"] == "active"


def test_remove_work(executive: ExecutiveState) -> None:
    w = WorkObject(content="x")
    executive.enqueue(w)
    assert executive.remove_work(str(w.work_id)) is True
    assert executive.remove_work(str(w.work_id)) is False


@pytest.mark.asyncio
async def test_restore_queue_skips_non_active_commitment_work(
    identity: IdentityManager,
    somatic: SomaticManager,
    tracer: Tracer,
) -> None:
    class FakeCommitmentStore:
        def __init__(self, commitments):
            self.commitments = {str(c.commitment_id): c for c in commitments}

        async def get(self, commitment_id: str):
            return self.commitments.get(commitment_id)

    class FakeWorkStore:
        def __init__(self, work_items):
            self.work_items = {str(w.work_id): w for w in work_items}

        async def list_ready(self, limit: int = 100):
            return list(self.work_items.values())[:limit]

        async def save(self, work):
            self.work_items[str(work.work_id)] = work

    executive = ExecutiveState(
        identity=identity,
        somatic=somatic,
        tracer=tracer,
    )

    active_commitment = Commitment(content="Active commitment")
    blocked_commitment = Commitment(
        content="Blocked commitment",
        status=CommitmentStatus.BLOCKED,
        meta={"blocked_reason": "manual_hold"},
    )
    active_work = WorkObject(
        content="active work",
        stage=WorkStage.MICRO_TASK,
        commitment_id=str(active_commitment.commitment_id),
    )
    blocked_work = WorkObject(
        content="blocked work",
        stage=WorkStage.MICRO_TASK,
        commitment_id=str(blocked_commitment.commitment_id),
    )
    executive.commitment_store = FakeCommitmentStore([active_commitment, blocked_commitment])
    executive.work_store = FakeWorkStore([active_work, blocked_work])

    restored = await executive.restore_queue()
    queued_ids = {str(item.work_id) for item in executive.task_queue}
    assert restored == 1
    assert str(active_work.work_id) in queued_ids
    assert str(blocked_work.work_id) not in queued_ids


@pytest.mark.asyncio
async def test_resume_deferred_work_only_unblocks_auto_resumable_commitments(
    identity: IdentityManager,
    somatic: SomaticManager,
    tracer: Tracer,
) -> None:
    class FakeCommitmentStore:
        def __init__(self, commitments):
            self.commitments = {str(c.commitment_id): c for c in commitments}

        async def list_by_status(self, status, limit: int = 100):
            return [
                commitment
                for commitment in self.commitments.values()
                if commitment.status == status
            ][:limit]

        async def get(self, commitment_id: str):
            return self.commitments.get(commitment_id)

        async def save(self, commitment):
            self.commitments[str(commitment.commitment_id)] = commitment

    class FakeWorkStore:
        def __init__(self, work_items):
            self.work_items = {str(w.work_id): w for w in work_items}

        async def get(self, work_id: str):
            return self.work_items.get(work_id)

        async def list_ready(self, limit: int = 100):
            return list(self.work_items.values())[:limit]

        async def save(self, work):
            self.work_items[str(work.work_id)] = work

    executive = ExecutiveState(
        identity=identity,
        somatic=somatic,
        tracer=tracer,
    )

    resumable = Commitment(
        content="Resume after rest",
        status=CommitmentStatus.BLOCKED,
        tags=["self_commitment"],
        meta={
            "source": "assistant_response",
            "resume_policy": "auto_on_executive_recovery",
            "blocked_reason": "executive_fatigue",
        },
    )
    resumable_operator_rest_without_policy = Commitment(
        content="Resume after operator rest without explicit policy",
        status=CommitmentStatus.BLOCKED,
        meta={"blocked_reason": "executive_operator_rest"},
    )
    manual_hold = Commitment(
        content="Do not resume automatically",
        status=CommitmentStatus.BLOCKED,
        meta={"blocked_reason": "manual_hold"},
    )
    resumable_work = WorkObject(
        content="resume work",
        stage=WorkStage.MICRO_TASK,
        commitment_id=str(resumable.commitment_id),
    )
    resumable_operator_rest_work = WorkObject(
        content="legacy resume work",
        stage=WorkStage.MICRO_TASK,
        commitment_id=str(resumable_operator_rest_without_policy.commitment_id),
    )
    manual_hold_work = WorkObject(
        content="manual hold work",
        stage=WorkStage.MICRO_TASK,
        commitment_id=str(manual_hold.commitment_id),
    )
    resumable.linked_work_ids.append(str(resumable_work.work_id))
    resumable_operator_rest_without_policy.linked_work_ids.append(
        str(resumable_operator_rest_work.work_id)
    )
    manual_hold.linked_work_ids.append(str(manual_hold_work.work_id))
    executive.commitment_store = FakeCommitmentStore(
        [resumable, resumable_operator_rest_without_policy, manual_hold]
    )
    executive.work_store = FakeWorkStore(
        [resumable_work, resumable_operator_rest_work, manual_hold_work]
    )

    result = await executive.resume_deferred_work()
    await __import__("asyncio").sleep(0)
    queued_ids = {str(item.work_id) for item in executive.task_queue}
    resumed = await executive.commitment_store.get(str(resumable.commitment_id))
    held = await executive.commitment_store.get(str(manual_hold.commitment_id))
    resumed_without_policy = await executive.commitment_store.get(
        str(resumable_operator_rest_without_policy.commitment_id)
    )

    assert result["unblocked_commitments"] == 2
    assert resumed is not None
    assert resumed.status == CommitmentStatus.ACTIVE
    assert resumed.meta["resume_reason"] == "executive_recovery"
    assert held is not None
    assert held.status == CommitmentStatus.BLOCKED
    assert resumed_without_policy is not None
    assert resumed_without_policy.status == CommitmentStatus.ACTIVE
    assert str(resumable_work.work_id) in queued_ids
    assert str(resumable_operator_rest_work.work_id) in queued_ids
    assert str(manual_hold_work.work_id) not in queued_ids
