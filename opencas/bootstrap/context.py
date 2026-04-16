"""Shared bootstrap context container for initialized OpenCAS substrate managers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from .context_close import close_bootstrap_context

if TYPE_CHECKING:
    from open_llm_auth.auth.manager import ProviderManager

    from opencas.api import LLMClient
    from opencas.autonomy.commitment_store import CommitmentStore
    from opencas.autonomy.executive import ExecutiveState
    from opencas.autonomy.portfolio import PortfolioStore
    from opencas.autonomy.project_orchestrator import ProjectOrchestrator
    from opencas.autonomy.work_store import WorkStore
    from opencas.consolidation import ConsolidationCurationStore
    from opencas.diagnostics import Doctor, HealthMonitor
    from opencas.embeddings import EmbeddingService
    from opencas.execution import TaskStore
    from opencas.execution.receipt_store import ExecutionReceiptStore
    from opencas.governance import ApprovalLedger
    from opencas.harness import AgenticHarness
    from opencas.identity import IdentityManager, SelfKnowledgeRegistry
    from opencas.infra import EventBus, HookBus, TypedHookRegistry
    from opencas.memory import MemoryStore
    from opencas.planning import PlanStore
    from opencas.plugins import PluginLifecycleManager, PluginStore, SkillRegistry
    from opencas.relational import RelationalEngine
    from opencas.runtime.readiness import AgentReadiness
    from opencas.sandbox import SandboxConfig
    from opencas.scheduling import ScheduleService, ScheduleStore
    from opencas.somatic import SomaticManager, SomaticStore
    from opencas.telemetry import TokenTelemetry, Tracer
    from opencas.tom import TomStore
    from opencas.workspace.service import WorkspaceIndexService

    from .config import BootstrapConfig
    from opencas.context import SessionContextStore
    from opencas.daydream import ConflictStore, DaydreamStore


@dataclass
class BootstrapContext:
    """Holds all initialized substrate managers after a successful boot."""

    config: BootstrapConfig
    tracer: Tracer
    identity: IdentityManager
    memory: MemoryStore
    tasks: TaskStore
    receipt_store: ExecutionReceiptStore
    embeddings: EmbeddingService
    somatic: SomaticManager
    llm: LLMClient
    token_telemetry: TokenTelemetry
    event_bus: EventBus
    hook_bus: HookBus
    typed_hook_registry: TypedHookRegistry
    ledger: ApprovalLedger
    skill_registry: SkillRegistry
    sandbox: SandboxConfig
    readiness: AgentReadiness
    context_store: SessionContextStore
    work_store: WorkStore
    project_orchestrator: ProjectOrchestrator
    relational: RelationalEngine
    daydream_store: DaydreamStore
    conflict_store: ConflictStore
    somatic_store: SomaticStore
    executive: ExecutiveState
    curation_store: ConsolidationCurationStore
    harness: AgenticHarness
    doctor: Doctor
    workspace_index: WorkspaceIndexService
    health_monitor: HealthMonitor
    commitment_store: CommitmentStore
    portfolio_store: PortfolioStore
    tom_store: TomStore
    self_knowledge_registry: SelfKnowledgeRegistry
    plugin_store: PluginStore
    plugin_lifecycle: PluginLifecycleManager
    plan_store: PlanStore
    schedule_store: ScheduleStore
    schedule_service: ScheduleService
    mcp_registry: Optional[Any] = None
    background_tasks: tuple[asyncio.Task[Any], ...] = ()

    async def close(self) -> None:
        """Close all owned runtime stores and services."""
        await close_bootstrap_context(self)
