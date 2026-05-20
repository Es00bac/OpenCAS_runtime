"""Shared bootstrap context container for initialized OpenCAS substrate managers."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from .context_close import close_bootstrap_context

if TYPE_CHECKING:

    from opencas.affective import AffectiveExaminationService
    from opencas.affective_registry import AffectiveRegistryWriter
    from opencas.api import LLMClient
    from opencas.autonomy.commitment_store import CommitmentStore
    from opencas.autonomy.executive import ExecutiveState
    from opencas.autonomy.portfolio import PortfolioStore
    from opencas.autonomy.project_orchestrator import ProjectOrchestrator
    from opencas.autonomy.work_store import WorkStore
    from opencas.cognition import CognitiveStateStore, SelfInspectionStore
    from opencas.consolidation import ConsolidationCurationStore
    from opencas.context import ContextProposalStore, SessionContextStore
    from opencas.daydream import ConflictStore, DaydreamSignalStore, DaydreamStore
    from opencas.diagnostics import Doctor, HealthMonitor
    from opencas.dreaming import DreamStore
    from opencas.embeddings import EmbeddingService
    from opencas.execution import TaskStore
    from opencas.execution.receipt_store import ExecutionReceiptStore
    from opencas.governance import ApprovalLedger, PluginTrustService, ShadowRegistry, WebTrustService
    from opencas.harness import AgenticHarness
    from opencas.identity import IdentityManager, SelfKnowledgeRegistry
    from opencas.infra import EventBus, HookBus, TypedHookRegistry
    from opencas.memory import ArtifactMemoryBridge, MemoryStore
    from opencas.memory.autobiography import (
        AutobiographyReconstructor,
        SessionAnchorStore,
        SessionAutobiographyComposer,
    )
    from opencas.planning import PlanStore
    from opencas.platform import CapabilityRegistry
    from opencas.plugins import PluginLifecycleManager, PluginStore, SkillRegistry
    from opencas.proof_chain import ProofStore
    from opencas.relational import RelationalEngine
    from opencas.runtime.readiness import AgentReadiness
    from opencas.sandbox import SandboxConfig
    from opencas.scheduling import ScheduleService, ScheduleStore
    from opencas.somatic import SomaticManager, SomaticStore
    from opencas.telemetry import TokenTelemetry, Tracer
    from opencas.thread_registry import ThreadRegistryStore
    from opencas.tom import TomStore
    from opencas.wellbeing import WellbeingStore
    from opencas.workspace.service import WorkspaceIndexService

    from .config import BootstrapConfig


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
    shadow_registry: ShadowRegistry
    web_trust: WebTrustService
    plugin_trust: PluginTrustService
    skill_registry: SkillRegistry
    sandbox: SandboxConfig
    readiness: AgentReadiness
    context_store: SessionContextStore
    context_proposal_store: ContextProposalStore
    work_store: WorkStore
    project_orchestrator: ProjectOrchestrator
    relational: RelationalEngine
    daydream_store: DaydreamStore
    daydream_signal_store: DaydreamSignalStore
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
    capability_registry: CapabilityRegistry
    plan_store: PlanStore
    schedule_store: ScheduleStore
    schedule_service: ScheduleService
    recovery_ledger: Any | None = None
    recovery_coordinator: Any | None = None
    artifact_bridge: Optional[ArtifactMemoryBridge] = None
    autobiography_anchor_store: Optional[SessionAnchorStore] = None
    autobiography_composer: Optional[SessionAutobiographyComposer] = None
    autobiography_reconstructor: Optional[AutobiographyReconstructor] = None
    affective_examinations: Optional[AffectiveExaminationService] = None
    affective_registry_writer: Optional[AffectiveRegistryWriter] = None
    self_inspection_store: Optional[SelfInspectionStore] = None
    cognitive_state_store: Optional[CognitiveStateStore] = None
    wellbeing_store: Optional[WellbeingStore] = None
    dream_store: Optional[DreamStore] = None
    proof_store: Optional[ProofStore] = None
    thread_registry_store: Optional[ThreadRegistryStore] = None
    mcp_registry: Optional[Any] = None
    background_tasks: tuple[asyncio.Task[Any], ...] = ()
    _exit_stack: Optional[AsyncExitStack] = None

    async def close(self) -> None:
        """Close all owned runtime stores and services."""
        await close_bootstrap_context(self)
