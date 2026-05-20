"""Boot-time assembly helpers for AgentRuntime.

These helpers keep AgentRuntime.__init__ readable while preserving the wiring
order between autonomy, execution, memory, and operator-facing channels.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from opencas.autonomy.boredom import BoredomPhysics
from opencas.autonomy.authorization import AuthorizationStore
from opencas.autonomy.creative_ladder import CreativeLadder
from opencas.autonomy.mode_utils import normalize_approval_mode
from opencas.autonomy.models import ApprovalMode
from opencas.autonomy.project_resume import ProjectResumeResolver
from opencas.autonomy.self_approval import SelfApprovalLadder
from opencas.autonomy.spark_router import SparkRouter
from opencas.autonomy.trust_engine import TrustEngine
from opencas.compaction import ConversationCompactor
from opencas.consolidation import NightlyConsolidationEngine
from opencas.context import ContextBuilder, ContextPacketBuilder, MemoryRetriever, TruthArbiter
from opencas.daydream import (
    ConflictRegistry,
    DaydreamPromotionService,
    DaydreamSignalBuilder,
    ReflectionEvaluator,
    ReflectionResolver,
    SelfCompassionMirror,
    SelfWorkspaceService,
)
from opencas.daydream.spark_evaluator import SparkEvaluator
from opencas.desktop_context import DesktopContextService
from opencas.dreaming import NightlyDreamingService
from opencas.execution import (
    BoundedAssistantAgent,
    BrowserSupervisor,
    ProcessSupervisor,
    PtySupervisor,
    ReliabilityCoordinator,
)
from opencas.governance import (
    AutoReviewerSubagent,
    AutoReviewMode,
    AutoReviewPolicy,
    normalize_auto_review_mode,
)
from opencas.identity import IdentityRebuilder
from opencas.initiative_contact import InitiativeContactService
from opencas.memory.fabric.graph import EpisodeGraph
from opencas.phone_config import PhoneRuntimeConfig
from opencas.platform import CapabilityRegistry
from opencas.proof_chain import ProofChainService
from opencas.somatic import SomaticModulators
from opencas.telegram_config import TelegramRuntimeConfig
from opencas.thread_registry import ThreadRegistryService
from opencas.tom import ToMEngine
from opencas.tools import ToolRegistry, ToolUseLoop
from opencas.wellbeing import FascinationGraph, MaintenancePlanner, WellbeingEngine

from .capability_snapshot import (
    CapabilityDriftReport,
    detect_capability_drift,
    record_drift_episode,
)
from .autobiography_hooks import (
    register_autobiography_hooks,
    schedule_autobiography_boot_recovery,
)
from .daydream import DaydreamGenerator
from .phone_runtime import initialize_runtime_phone
from .provenance_hooks import register_runtime_provenance_hooks
from .shadow_registry_hooks import register_runtime_shadow_registry_hooks
from opencas.platform.telegram_runtime import initialize_runtime_telegram


def build_runtime_auto_review_policy(runtime: Any, context: Any) -> AutoReviewPolicy:
    """Build the approval auto-review policy from runtime configuration."""
    raw_mode = getattr(getattr(context, "config", None), "approval_mode", AutoReviewMode.DEFAULT.value)
    try:
        mode = normalize_auto_review_mode(raw_mode)
    except ValueError:
        mode = AutoReviewMode.DEFAULT
    reviewer = None
    if mode is AutoReviewMode.AUTO_REVIEW:
        reviewer = AutoReviewerSubagent(llm=getattr(runtime, "llm", None))
    return AutoReviewPolicy(mode=mode, reviewer=reviewer)


def initialize_runtime_autonomy(runtime: Any, context: Any) -> None:
    """Wire cognition, approval, and relationship-aware autonomy components."""
    runtime.executive = context.executive
    runtime.creative = CreativeLadder(
        executive=runtime.executive,
        embeddings=context.embeddings,
        tracer=runtime.tracer,
        work_store=context.work_store,
        relational=getattr(context, "relational", None),
        task_store=getattr(context, "tasks", None),
    )
    runtime.orchestrator = context.project_orchestrator

    from opencas.refusal import ConversationalRefusalGate
    approval_mode = normalize_approval_mode(
        getattr(getattr(context, "config", None), "approval_mode", ApprovalMode.DEFAULT.value)
    )
    trust_engine = TrustEngine(
        identity=context.identity,
        base_trust=getattr(getattr(context, "config", None), "base_trust", 0.5),
    )
    context.trust_engine = trust_engine
    authorization_store = AuthorizationStore(context.config.state_dir / "authorizations.db")
    runtime.authorization_store = authorization_store

    runtime.approval = SelfApprovalLadder(
        identity=context.identity,
        somatic=context.somatic,
        tracer=runtime.tracer,
        relational=getattr(context, "relational", None),
        ledger=getattr(context, "ledger", None),
        web_trust=getattr(context, "web_trust", None),
        mode=approval_mode,
        trust_engine=trust_engine,
        authorization_store=authorization_store,
    )
    runtime.trust_engine = trust_engine
    runtime.auto_review = build_runtime_auto_review_policy(runtime, context)
    runtime.refusal_gate = ConversationalRefusalGate(
        approval=runtime.approval,
        hook_bus=runtime.ctx.hook_bus,
    )
    runtime.spark_evaluator = SparkEvaluator(
        embeddings=context.embeddings,
        work_store=getattr(context, "work_store", None),
        executive=runtime.executive,
        somatic=context.somatic,
        relational=getattr(context, "relational", None),
        novelty_floor=0.3,
    )
    runtime.daydream = DaydreamGenerator(
        llm=runtime.llm,
        memory=runtime.memory,
        tracer=runtime.tracer,
        identity=context.identity,
        somatic=context.somatic,
        relational=getattr(context, "relational", None),
        daydream_store=getattr(context, "daydream_store", None),
        spark_evaluator=runtime.spark_evaluator,
    )
    runtime.reflection_evaluator = ReflectionEvaluator()
    runtime.reflection_resolver = ReflectionResolver(mirror=SelfCompassionMirror())
    runtime.conflict_registry = None
    if getattr(runtime.ctx, "conflict_store", None):
        runtime.conflict_registry = ConflictRegistry(runtime.ctx.conflict_store)
    runtime._last_daydream_time: Optional[datetime] = None
    runtime.boredom = BoredomPhysics()
    runtime.spark_router = SparkRouter()
    runtime.commitment_store = getattr(context, "commitment_store", None)
    runtime.self_inspection_store = getattr(context, "self_inspection_store", None)
    runtime.cognitive_state_store = getattr(context, "cognitive_state_store", None)
    runtime.wellbeing_store = getattr(context, "wellbeing_store", None)
    runtime.dream_store = getattr(context, "dream_store", None)
    runtime.nightly_dreaming = None
    if runtime.dream_store is not None:
        runtime.nightly_dreaming = NightlyDreamingService(
            store=runtime.dream_store,
            workspace_root=context.config.agent_workspace_root(),
        )
    runtime.proof_store = getattr(context, "proof_store", None)
    runtime.proof_chain = None
    if runtime.proof_store is not None:
        runtime.proof_chain = ProofChainService(runtime.proof_store)
    runtime.thread_registry_store = getattr(context, "thread_registry_store", None)
    runtime.thread_registry_service = None
    if runtime.thread_registry_store is not None:
        runtime.thread_registry_service = ThreadRegistryService(
            store=runtime.thread_registry_store,
            workspace_root=context.config.agent_workspace_root(),
        )
    runtime.daydream_signal_store = getattr(context, "daydream_signal_store", None)
    runtime.daydream_signal_builder = DaydreamSignalBuilder()
    runtime.self_workspace = SelfWorkspaceService(
        workspace_root=context.config.agent_workspace_root(),
        artifact_bridge=getattr(context, "artifact_bridge", None),
    )
    runtime.daydream_promotion = None
    if runtime.daydream_signal_store is not None:
        runtime.daydream_promotion = DaydreamPromotionService(
            signal_store=runtime.daydream_signal_store,
            self_workspace=runtime.self_workspace,
            signal_builder=runtime.daydream_signal_builder,
            thread_registry_service=runtime.thread_registry_service,
            creative=runtime.creative,
        )
    runtime.wellbeing_engine = WellbeingEngine()
    runtime.maintenance_planner = MaintenancePlanner()
    runtime.fascination_graph = FascinationGraph()
    runtime.portfolio_store = getattr(context, "portfolio_store", None)
    runtime.schedule_service = getattr(context, "schedule_service", None)
    if runtime.schedule_service is not None:
        runtime.schedule_service.runtime = runtime
    runtime.tom = ToMEngine(
        identity=context.identity,
        tracer=runtime.tracer,
        store=getattr(context, "tom_store", None),
    )


def _wire_capability_drift(runtime: Any, context: Any) -> None:
    """Compute the per-boot tool diff and prepare the async episode recorder.

    The sync diff runs immediately so the prompt-time capability context can warn
    the LLM about lost tools on the very first turn. The episode write is deferred
    to scheduler startup because it needs the event loop and embedding service.
    """
    state_dir = getattr(getattr(context, "config", None), "state_dir", None)
    current_names: list[str] = []
    try:
        current_names = [str(getattr(entry, "name", "")) for entry in runtime.tools.list_tools()]
    except Exception:
        current_names = []
    if state_dir is None:
        runtime.capability_drift_report = CapabilityDriftReport()
        runtime.record_capability_drift_episode = None
        return
    report = detect_capability_drift(
        state_dir=state_dir,
        current_tools=current_names,
        tracer=getattr(runtime, "tracer", None),
    )
    runtime.capability_drift_report = report

    memory = getattr(context, "memory", None)
    embeddings = getattr(context, "embeddings", None)

    async def _record() -> dict[str, Any]:
        if not report.has_drift:
            return {"recorded": False, "reason": "no_drift"}
        await record_drift_episode(memory=memory, embeddings=embeddings, report=report)
        return {
            "recorded": bool(report.lost),
            "lost_count": len(report.lost),
            "gained_count": len(report.gained),
        }

    runtime.record_capability_drift_episode = _record


def initialize_runtime_execution(runtime: Any, context: Any) -> None:
    """Wire supervisors, tools, and bounded execution components."""
    runtime.process_supervisor = ProcessSupervisor()
    runtime.pty_supervisor = PtySupervisor()
    runtime.browser_supervisor = BrowserSupervisor()
    shared_capability_registry = getattr(context, "capability_registry", None)
    if shared_capability_registry is not None:
        runtime.capability_registry = shared_capability_registry
    elif getattr(runtime, "capability_registry", None) is None:
        runtime.capability_registry = CapabilityRegistry()
    runtime.plugin_lifecycle = getattr(context, "plugin_lifecycle", None)
    if runtime.plugin_lifecycle is not None and shared_capability_registry is not None:
        runtime.plugin_lifecycle.capability_registry = shared_capability_registry
    if runtime.plugin_lifecycle is not None:
        runtime.tools = runtime.plugin_lifecycle.tools
    else:
        runtime.tools = ToolRegistry(tracer=runtime.tracer, hook_bus=runtime.ctx.hook_bus)
    runtime.tools.runtime = runtime
    register_runtime_provenance_hooks(runtime)
    register_autobiography_hooks(runtime)
    register_runtime_shadow_registry_hooks(runtime)
    runtime._register_default_tools()
    runtime._register_skills()
    _wire_capability_drift(runtime, context)
    runtime.baa = BoundedAssistantAgent(
        tools=runtime.tools,
        llm=runtime.llm,
        tracer=runtime.tracer,
        max_concurrent=2,
        store=context.tasks,
        event_bus=context.event_bus,
        receipt_store=getattr(context, "receipt_store", None),
        runtime=runtime,
        memory=getattr(context, "memory", None),
        embeddings=getattr(context, "embeddings", None),
    )
    runtime.recovery_coordinator = getattr(context, "recovery_coordinator", None)
    recovery_executor = getattr(runtime.recovery_coordinator, "executor", None)
    if recovery_executor is not None:
        recovery_executor.baa = runtime.baa
    runtime.truth_arbiter = TruthArbiter(runtime)
    context.truth_arbiter = runtime.truth_arbiter
    runtime.context_proposals = getattr(context, "context_proposal_store", None)
    runtime.context_packet_builder = ContextPacketBuilder(
        runtime=runtime,
        truth_arbiter=runtime.truth_arbiter,
        proposal_store=runtime.context_proposals,
    )
    runtime.orchestrator.baa = runtime.baa
    runtime.tool_loop = ToolUseLoop(
        llm=runtime.llm,
        tools=runtime.tools,
        approval=runtime.approval,
        tracer=runtime.tracer,
    )
    if context.event_bus:
        from opencas.infra import BaaCompletedEvent

        context.event_bus.subscribe(BaaCompletedEvent, runtime._on_baa_completed)
    runtime.reliability = None
    runtime.scheduler = None
    if context.event_bus:
        runtime.reliability = ReliabilityCoordinator(
            event_bus=context.event_bus,
            window_size=10,
            failure_threshold=0.7,
            cooldown_seconds=300,
        )


def initialize_runtime_memory_surfaces(runtime: Any, context: Any) -> None:
    """Wire memory retrieval, compaction, consolidation, and identity rebuild surfaces."""
    runtime.autobiography_anchor_store = getattr(context, "autobiography_anchor_store", None)
    runtime.autobiography_composer = getattr(context, "autobiography_composer", None)
    runtime.autobiography_reconstructor = getattr(context, "autobiography_reconstructor", None)
    runtime.episode_graph = EpisodeGraph(store=runtime.memory)
    runtime.rebuilder = IdentityRebuilder(
        memory=runtime.memory,
        episode_graph=runtime.episode_graph,
        llm=runtime.llm,
    )
    runtime.retriever = MemoryRetriever(
        memory=runtime.memory,
        embeddings=context.embeddings,
        episode_graph=runtime.episode_graph,
        somatic_manager=context.somatic,
        relational_engine=context.relational,
        affective_examinations=getattr(context, "affective_examinations", None),
        cognitive_state_store=getattr(context, "cognitive_state_store", None),
        tracer=runtime.tracer,
    )
    runtime.project_resume = ProjectResumeResolver(
        memory=runtime.memory,
        work_store=getattr(context, "work_store", None),
        plan_store=getattr(context, "plan_store", None),
        harness_store=getattr(getattr(context, "harness", None), "store", None),
    )
    runtime.modulators = SomaticModulators(context.somatic.state)
    runtime.builder = ContextBuilder(
        store=context.context_store,
        retriever=runtime.retriever,
        identity=context.identity,
        executive=runtime.executive,
        agent_profile=runtime.agent_profile,
        config=context.config,
        modulators=runtime.modulators,
        relational=getattr(context, "relational", None),
        tom=runtime.tom,
        project_resume_resolver=runtime.project_resume,
        affective_examinations=getattr(context, "affective_examinations", None),
        self_inspection_store=getattr(context, "self_inspection_store", None),
        cognitive_state_store=getattr(context, "cognitive_state_store", None),
        schedule_service=getattr(context, "schedule_service", None),
        daydream_store=getattr(context, "daydream_store", None),
        context_proposal_store=getattr(context, "context_proposal_store", None),
        autobiography_reconstructor=getattr(context, "autobiography_reconstructor", None),
        thread_registry_store=getattr(context, "thread_registry_store", None),
        commitment_store=runtime.commitment_store,
        llm=runtime.llm,
    )
    runtime.compactor = ConversationCompactor(
        memory=runtime.memory,
        llm=runtime.llm,
        tracer=runtime.tracer,
        context_store=runtime.ctx.context_store,
        identity=context.identity,
        embeddings=context.embeddings,
    )
    runtime.consolidation = NightlyConsolidationEngine(
        memory=runtime.memory,
        embeddings=context.embeddings,
        llm=runtime.llm,
        identity=context.identity,
        tracer=runtime.tracer,
        curation_store=getattr(context, "curation_store", None),
        tom_store=getattr(context, "tom_store", None),
        commitment_store=runtime.commitment_store,
        work_store=getattr(context, "work_store", None),
    )
    runtime.harness = getattr(context, "harness", None)
    if runtime.harness:
        runtime.harness.baa = runtime.baa
        runtime.harness.project_resume_resolver = runtime.project_resume
    schedule_autobiography_boot_recovery(runtime)


def initialize_runtime_channels(runtime: Any, context: Any) -> None:
    """Wire operator-facing channels and mutable runtime status handles."""
    runtime._telegram_config = TelegramRuntimeConfig()
    runtime._telegram = None
    initialize_runtime_telegram(runtime, context.config.state_dir)
    runtime._phone_config = PhoneRuntimeConfig()
    runtime._phone = None
    initialize_runtime_phone(runtime, context.config.state_dir)
    runtime.initiative_contact = InitiativeContactService(
        runtime=runtime,
        state_dir=context.config.state_dir,
    )
    if getattr(runtime, "daydream_promotion", None) is not None:
        runtime.daydream_promotion.initiative_contact = runtime.initiative_contact
    runtime.desktop_context = DesktopContextService(
        runtime=runtime,
        state_dir=context.config.state_dir,
    )
