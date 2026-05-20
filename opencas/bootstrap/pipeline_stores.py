"""Connected-store bootstrap helpers for ``BootstrapPipeline``."""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from opencas.autonomy.commitment_store import CommitmentStore
from opencas.autonomy.executive import ExecutiveState
from opencas.autonomy.portfolio import PortfolioStore
from opencas.autonomy.work_store import WorkStore
from opencas.cognition import CognitiveStateStore, SelfInspectionStore
from opencas.context import ContextProposalStore, SessionContextStore
from opencas.dreaming import DreamStore
from opencas.execution import TaskStore
from opencas.execution.receipt_store import ExecutionReceiptStore
from opencas.memory import MemoryStore
from opencas.proof_chain import ProofStore
from opencas.runtime.suppressed_reframe_runtime import record_loaded_suppressed_reframe_thread_beads
from opencas.thread_registry import ThreadRegistryStore
from opencas.thread_registry.service import ThreadRegistryService
from opencas.wellbeing import WellbeingStore

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
    context_proposal_store: ContextProposalStore
    work_store: WorkStore
    commitment_store: CommitmentStore
    self_inspection_store: SelfInspectionStore
    cognitive_state_store: CognitiveStateStore
    wellbeing_store: WellbeingStore
    dream_store: DreamStore
    proof_store: ProofStore
    thread_registry_store: ThreadRegistryStore
    portfolio_store: PortfolioStore
    executive: ExecutiveState


async def initialize_runtime_stores(
    config: BootstrapConfig,
    *,
    identity: "IdentityManager",
    tracer: "Tracer",
    stage: Callable[[str, Optional[dict]], None],
    exit_stack: Optional[AsyncExitStack] = None,
) -> RuntimeStoreBundle:
    """Connect the foundational runtime stores and restore executive state."""
    def _register_close(obj: Any) -> None:
        if exit_stack is not None:
            exit_stack.push_async_callback(obj.close)

    memory = MemoryStore(config.memory_db)
    await memory.connect()
    _register_close(memory)
    stage("memory_online")

    tasks = TaskStore(config.tasks_db)
    await tasks.connect()
    _register_close(tasks)
    stage("tasks_online")

    receipt_store = ExecutionReceiptStore(config.state_dir / "receipts.db")
    await receipt_store.connect()
    _register_close(receipt_store)
    stage("execution_receipts_online")

    context_store = SessionContextStore(config.context_db)
    await context_store.connect()
    _register_close(context_store)
    stage("context_store_online")

    context_proposal_store = ContextProposalStore(config.state_dir / "context_proposals.db")
    await context_proposal_store.connect()
    _register_close(context_proposal_store)
    stage("context_proposals_online")

    work_store = WorkStore(config.work_db)
    await work_store.connect()
    _register_close(work_store)
    stage("work_store_online")

    commitment_store = CommitmentStore(config.state_dir / "commitments.db")
    await commitment_store.connect()
    _register_close(commitment_store)
    self_inspection_store = SelfInspectionStore(config.state_dir / "self_inspection.db")
    await self_inspection_store.connect()
    _register_close(self_inspection_store)
    cognitive_state_store = CognitiveStateStore(config.state_dir / "cognitive_state.db")
    await cognitive_state_store.connect()
    _register_close(cognitive_state_store)
    wellbeing_store = WellbeingStore(config.state_dir / "wellbeing.db")
    await wellbeing_store.connect()
    _register_close(wellbeing_store)
    dream_store = DreamStore(config.state_dir / "dreaming.db")
    await dream_store.connect()
    _register_close(dream_store)
    proof_store = ProofStore(config.state_dir / "proof_chain.db")
    await proof_store.connect()
    _register_close(proof_store)
    thread_registry_store = ThreadRegistryStore(config.state_dir / "thread_registry.db")
    await thread_registry_store.connect()
    _register_close(thread_registry_store)
    portfolio_store = PortfolioStore(config.state_dir / "portfolio.db")
    await portfolio_store.connect()
    _register_close(portfolio_store)
    stage("commitment_self_inspection_cognitive_wellbeing_thread_registry_portfolio_online")

    executive = ExecutiveState(
        identity=identity,
        somatic=None,
        tracer=tracer,
        work_store=work_store,
        commitment_store=commitment_store,
    )
    executive.load_snapshot(config.state_dir / "executive.json")
    thread_registry_service = ThreadRegistryService(
        store=thread_registry_store,
        workspace_root=config.agent_workspace_root(),
    )
    suppressed_reframe_result = await record_loaded_suppressed_reframe_thread_beads(
        executive=executive,
        service=thread_registry_service,
        tracer=tracer,
    )
    if suppressed_reframe_result.get("recorded_count"):
        executive.save_snapshot(config.state_dir / "executive.json")
    executive.restore_goals_from_identity()
    await executive.restore_queue(limit=100)
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
        context_proposal_store=context_proposal_store,
        work_store=work_store,
        commitment_store=commitment_store,
        self_inspection_store=self_inspection_store,
        cognitive_state_store=cognitive_state_store,
        wellbeing_store=wellbeing_store,
        dream_store=dream_store,
        proof_store=proof_store,
        thread_registry_store=thread_registry_store,
        portfolio_store=portfolio_store,
        executive=executive,
    )
