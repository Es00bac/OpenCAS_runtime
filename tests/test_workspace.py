"""Tests for ExecutiveWorkspace and InterventionPolicy."""

from datetime import datetime, timezone
from pathlib import Path
import pytest

from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.autonomy.executive import ExecutiveState
from opencas.autonomy.intervention import InterventionKind, InterventionPolicy
from opencas.autonomy.models import WorkObject, WorkStage
from opencas.autonomy.workspace import (
    ExecutiveWorkspace,
    ExecutionMode,
    PortfolioBoost,
    WorkspaceAffinity,
    WorkspaceItem,
    WorkspaceItemKind,
)
from opencas.identity import IdentityManager, IdentityStore


@pytest.fixture
def identity(tmp_path: Path):
    store = IdentityStore(tmp_path / "identity")
    mgr = IdentityManager(store)
    mgr.load()
    return mgr


@pytest.fixture
def executive(identity: IdentityManager):
    return ExecutiveState(identity=identity)


def test_workspace_rebuild_scoring() -> None:
    c = Commitment(content="deadline task", priority=10.0, deadline=datetime.now(timezone.utc))
    wo = WorkObject(content="background work", stage=WorkStage.MICRO_TASK, promotion_score=0.8)
    workspace = ExecutiveWorkspace.rebuild(
        commitments=[c],
        work_objects=[wo],
    )

    assert workspace.focus is not None
    # Commitment with deadline should score high
    assert workspace.queue[0].kind == WorkspaceItemKind.COMMITMENT
    assert workspace.queue[0].total_score > 0.5


def test_workspace_rebuild_with_portfolio_boost() -> None:
    wo = WorkObject(content="boosted work", stage=WorkStage.MICRO_TASK, promotion_score=0.5)
    wo.portfolio_id = "port-1"
    boost = PortfolioBoost(portfolio_id="port-1", spark_count=5, boost=0.1)
    workspace = ExecutiveWorkspace.rebuild(
        commitments=[],
        work_objects=[wo],
        portfolio_boosts={"port-1": boost},
    )

    item = [i for i in workspace.queue if i.content == "boosted work"][0]
    # base total without boost would be around 0.4*0.5 + 0.4*0.5 + 0.2*0.2 = 0.44
    # with boost +0.1 => ~0.54
    assert item.total_score > 0.5


def test_intervention_surface_clarification(executive: ExecutiveState) -> None:
    workspace = ExecutiveWorkspace(focus=None, queue=[])
    live_orders = [{"task_id": "t1", "stage": "needs_clarification"}]
    decision = InterventionPolicy.evaluate(
        workspace=workspace,
        live_work_orders=live_orders,
    )
    assert decision.kind == InterventionKind.SURFACE_CLARIFICATION
    assert decision.target_item_id == "t1"


def test_intervention_surface_approval(executive: ExecutiveState) -> None:
    workspace = ExecutiveWorkspace(focus=None, queue=[])
    live_orders = [{"task_id": "t2", "stage": "needs_approval"}]
    decision = InterventionPolicy.evaluate(
        workspace=workspace,
        live_work_orders=live_orders,
    )
    assert decision.kind == InterventionKind.SURFACE_APPROVAL


def test_intervention_launch_background(executive: ExecutiveState) -> None:
    focus = WorkspaceItem(
        kind=WorkspaceItemKind.TASK,
        content="bg task",
        execution_mode=ExecutionMode.BACKGROUND_AGENT,
    )
    workspace = ExecutiveWorkspace(focus=focus, queue=[focus])
    decision = InterventionPolicy.evaluate(workspace=workspace, baa_queue_depth=2)
    assert decision.kind == InterventionKind.LAUNCH_BACKGROUND


def test_intervention_retire_low_score(executive: ExecutiveState) -> None:
    focus = WorkspaceItem(
        kind=WorkspaceItemKind.TASK,
        content="low value",
        total_score=0.1,
        execution_mode=ExecutionMode.FOREGROUND_TOOLS,
    )
    workspace = ExecutiveWorkspace(focus=focus, queue=[focus])
    decision = InterventionPolicy.evaluate(workspace=workspace)
    assert decision.kind == InterventionKind.RETIRE_OR_DEFER_FOCUS


def test_intervention_reclaim_stale_personal(executive: ExecutiveState) -> None:
    focus = WorkspaceItem(
        kind=WorkspaceItemKind.COMMITMENT,
        content="stale personal",
        affinity=WorkspaceAffinity.PERSONAL,
        meta={"stale": True},
    )
    workspace = ExecutiveWorkspace(focus=focus, queue=[focus])
    decision = InterventionPolicy.evaluate(workspace=workspace)
    assert decision.kind == InterventionKind.RECLAIM_TO_FOREGROUND


def test_intervention_verify_completed_work(executive: ExecutiveState) -> None:
    focus = WorkspaceItem(
        kind=WorkspaceItemKind.TASK,
        content="done but unchecked",
        meta={"verified": False},
    )
    workspace = ExecutiveWorkspace(focus=focus, queue=[focus])
    decision = InterventionPolicy.evaluate(workspace=workspace)
    assert decision.kind == InterventionKind.VERIFY_COMPLETED_WORK


def test_intervention_no_intervention_empty_workspace(executive: ExecutiveState) -> None:
    workspace = ExecutiveWorkspace(focus=None, queue=[])
    decision = InterventionPolicy.evaluate(workspace=workspace)
    assert decision.kind == InterventionKind.NO_INTERVENTION
