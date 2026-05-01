"""Connected-store bootstrap helpers for ``BootstrapPipeline``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from opencas.autonomy.commitment_store import CommitmentStore
from opencas.autonomy.executive import ExecutiveState
from opencas.autonomy.portfolio import PortfolioStore
from opencas.autonomy.work_store import WorkStore
from opencas.context import SessionContextStore
from opencas.execution import TaskStore
from opencas.execution.receipt_store import ExecutionReceiptStore
from opencas.memory import MemoryStore

from .config import BootstrapConfig
from .live_objective import read_tasklist_live_objective

if TYPE_CHECKING:
    from opencas.identity import IdentityManager
    from opencas.telemetry import Tracer


@dataclass
class RuntimeStoreBundle:
    """Grouped connected stores needed before provider and somatic startup."""

    memory: MemoryStore
    tasks: TaskStore
    receipt_store: ExecutionReceiptStore
    context_store: SessionContextStore
    work_store: WorkStore
    commitment_store: CommitmentStore
    portfolio_store: PortfolioStore
    executive: ExecutiveState


async def initialize_runtime_stores(
    config: BootstrapConfig,
    *,
    identity: "IdentityManager",
    tracer: "Tracer",
    stage: Callable[[str, Optional[dict]], None],
) -> RuntimeStoreBundle:
    """Connect the foundational runtime stores and restore executive state."""
    memory = MemoryStore(config.memory_db)
    await memory.connect()
    stage("memory_online")

    tasks = TaskStore(config.tasks_db)
    await tasks.connect()
    stage("tasks_online")

    receipt_store = ExecutionReceiptStore(config.state_dir / "receipts.db")
    await receipt_store.connect()
    stage("execution_receipts_online")

    context_store = SessionContextStore(config.context_db)
    await context_store.connect()
    stage("context_store_online")

    work_store = WorkStore(config.work_db)
    await work_store.connect()
    stage("work_store_online")

    commitment_store = CommitmentStore(config.state_dir / "commitments.db")
    await commitment_store.connect()
    portfolio_store = PortfolioStore(config.state_dir / "portfolio.db")
    await portfolio_store.connect()
    stage("commitment_portfolio_online")

    executive = ExecutiveState(
        identity=identity,
        somatic=None,
        tracer=tracer,
        work_store=work_store,
        commitment_store=commitment_store,
    )
    executive.load_snapshot(config.state_dir / "executive.json")
    executive.restore_goals_from_identity()
    live_objective = read_tasklist_live_objective(config.workspace_root)
    if live_objective:
        executive.set_intention(live_objective, source="tasklist_live_objective")
    elif executive.intention_source == "tasklist_live_objective":
        executive.set_intention(None)
    stage("executive_online")

    return RuntimeStoreBundle(
        memory=memory,
        tasks=tasks,
        receipt_store=receipt_store,
        context_store=context_store,
        work_store=work_store,
        commitment_store=commitment_store,
        portfolio_store=portfolio_store,
        executive=executive,
    )
