"""Boot-time assembly helpers for AgentRuntime.

These helpers keep AgentRuntime.__init__ readable while preserving the wiring
order between autonomy, execution, memory, and operator-facing channels.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from opencas.autonomy.boredom import BoredomPhysics
from opencas.autonomy.creative_ladder import CreativeLadder
from opencas.autonomy.self_approval import SelfApprovalLadder
from opencas.autonomy.spark_router import SparkRouter
from opencas.daydream import (
    ConflictRegistry,
    ReflectionEvaluator,
    ReflectionResolver,
    SelfCompassionMirror,
)
from opencas.daydream.spark_evaluator import SparkEvaluator
from opencas.execution import (
    BoundedAssistantAgent,
    BrowserSupervisor,
    ProcessSupervisor,
    PtySupervisor,
    ReliabilityCoordinator,
)
from opencas.context import ContextBuilder, MemoryRetriever
from opencas.compaction import ConversationCompactor
from opencas.consolidation import NightlyConsolidationEngine
from opencas.identity import IdentityRebuilder
from opencas.memory.fabric.graph import EpisodeGraph
from opencas.somatic import SomaticModulators
from opencas.telemetry import Tracer
from opencas.tools import ToolRegistry, ToolUseLoop
from opencas.tom import ToMEngine
from opencas.telegram_config import TelegramRuntimeConfig

from .daydream import DaydreamGenerator
from .telegram_runtime import initialize_runtime_telegram


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

    runtime.approval = SelfApprovalLadder(
        identity=context.identity,
        somatic=context.somatic,
        tracer=runtime.tracer,
        relational=getattr(context, "relational", None),
        ledger=getattr(context, "ledger", None),
    )
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
    runtime.portfolio_store = getattr(context, "portfolio_store", None)
    runtime.schedule_service = getattr(context, "schedule_service", None)
    if runtime.schedule_service is not None:
        runtime.schedule_service.runtime = runtime
    runtime.tom = ToMEngine(
        identity=context.identity,
        tracer=runtime.tracer,
        store=getattr(context, "tom_store", None),
    )


def initialize_runtime_execution(runtime: Any, context: Any) -> None:
    """Wire supervisors, tools, and bounded execution components."""
    runtime.process_supervisor = ProcessSupervisor()
    runtime.pty_supervisor = PtySupervisor()
    runtime.browser_supervisor = BrowserSupervisor()
    runtime.plugin_lifecycle = getattr(context, "plugin_lifecycle", None)
    if runtime.plugin_lifecycle is not None:
        runtime.tools = runtime.plugin_lifecycle.tools
    else:
        runtime.tools = ToolRegistry(tracer=runtime.tracer, hook_bus=runtime.ctx.hook_bus)
    runtime._register_default_tools()
    runtime._register_skills()
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
    )
    runtime.compactor = ConversationCompactor(
        memory=runtime.memory,
        llm=runtime.llm,
        tracer=runtime.tracer,
        context_store=runtime.ctx.context_store,
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


def initialize_runtime_channels(runtime: Any, context: Any) -> None:
    """Wire operator-facing channels and mutable runtime status handles."""
    runtime._telegram_config = TelegramRuntimeConfig()
    runtime._telegram = None
    initialize_runtime_telegram(runtime, context.config.state_dir)
