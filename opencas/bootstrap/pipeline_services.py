"""Mid-pipeline service initialization helpers for ``BootstrapPipeline``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from opencas.autonomy.project_orchestrator import ProjectOrchestrator
from opencas.consolidation import ConsolidationCurationStore
from opencas.daydream import ConflictStore, DaydreamStore
from opencas.diagnostics import Doctor, HealthMonitor
from opencas.governance import ApprovalLedger, ApprovalLedgerStore
from opencas.harness import AgenticHarness, HarnessStore
from opencas.planning import PlanStore
from opencas.plugins import (
    PluginLifecycleManager,
    PluginRegistry,
    PluginStore,
    SkillRegistry,
)
from opencas.relational import MusubiStore, RelationalEngine
from opencas.runtime.readiness import AgentReadiness
from opencas.scheduling import ScheduleService, ScheduleStore
from opencas.tom import TomStore
from opencas.tools import ToolRegistry

from .config import BootstrapConfig

if TYPE_CHECKING:
    from opencas.api import LLMClient
    from opencas.autonomy.work_store import WorkStore
    from opencas.identity import IdentityManager
    from opencas.infra import EventBus, HookBus, TypedHookRegistry
    from opencas.somatic import SomaticManager
    from opencas.telemetry import Tracer


@dataclass
class RuntimeServiceBundle:
    """Grouped service objects constructed after core stores and embeddings."""

    plugin_store: PluginStore
    skill_registry: SkillRegistry
    plugin_lifecycle: PluginLifecycleManager
    ledger: ApprovalLedger
    readiness: AgentReadiness
    project_orchestrator: ProjectOrchestrator
    relational: RelationalEngine
    daydream_store: DaydreamStore
    conflict_store: ConflictStore
    curation_store: ConsolidationCurationStore
    harness: AgenticHarness
    tom_store: TomStore
    plan_store: PlanStore
    schedule_store: ScheduleStore
    schedule_service: ScheduleService
    mcp_registry: Optional[Any]
    doctor: Doctor
    health_monitor: HealthMonitor


async def initialize_runtime_services(
    config: BootstrapConfig,
    *,
    identity: "IdentityManager",
    llm: "LLMClient",
    tracer: "Tracer",
    somatic: "SomaticManager",
    event_bus: "EventBus",
    hook_bus: "HookBus",
    typed_hook_registry: "TypedHookRegistry",
    work_store: "WorkStore",
    stage: Callable[[str, Optional[dict]], None],
    is_first_boot: bool,
    clean_boot: bool,
) -> RuntimeServiceBundle:
    """Initialize the mid-pipeline relational, plugin, harness, and planning services."""
    relational_store = MusubiStore(config.relational_db)
    relational = RelationalEngine(store=relational_store, tracer=tracer)
    await relational.connect()
    if is_first_boot or clean_boot:
        await relational.initialize(
            trust=0.5,
            resonance=0.0,
            presence=0.0,
            attunement=0.0,
            note="Initial relational field. Awaiting contact.",
        )
        identity.self_model.relational_state_id = str(relational.state.state_id)
        identity.save()
    stage("relational_online", {"musubi": relational.state.musubi})

    plugin_store = PluginStore(config.plugins_db)
    await plugin_store.connect()
    plugin_registry = PluginRegistry()
    skill_registry = SkillRegistry()
    plugin_lifecycle = PluginLifecycleManager(
        store=plugin_store,
        plugin_registry=plugin_registry,
        skill_registry=skill_registry,
        tools=ToolRegistry(tracer=tracer, hook_bus=hook_bus),
        hook_registry=typed_hook_registry,
        builtin_dir=None,
        tracer=tracer,
    )
    plugins_dir = config.state_dir.parent / "plugins"
    import opencas.plugins.skills as skills_pkg

    builtin_skills_dir = Path(skills_pkg.__file__).parent
    plugin_lifecycle.builtin_dir = builtin_skills_dir
    loaded_plugins = await plugin_lifecycle.load_all()
    if plugins_dir.exists() and plugins_dir.is_dir():
        from opencas.plugins.loader import load_builtin_plugins

        user_plugins = load_builtin_plugins(
            plugins_dir,
            plugin_registry,
            skill_registry,
            plugin_lifecycle.tools,
            typed_hook_registry,
        )
        for plugin in user_plugins:
            if not await plugin_store.is_installed(plugin.plugin_id):
                await plugin_store.install(
                    plugin_id=plugin.plugin_id,
                    name=plugin.name,
                    description=plugin.description,
                    source="builtin",
                    path=plugin.path or "",
                    manifest=plugin.manifest,
                )
        loaded_plugins.extend(user_plugins)
    stage(
        "plugins_online",
        {
            "plugin_count": len(loaded_plugins),
            "plugin_dir": str(plugins_dir),
            "builtin_dir": str(builtin_skills_dir),
        },
    )

    ledger_store = ApprovalLedgerStore(config.state_dir / "governance.db")
    await ledger_store.connect()
    ledger = ApprovalLedger(store=ledger_store, tracer=tracer)
    stage("governance_online")

    readiness = AgentReadiness()
    stage("readiness_booting")

    project_orchestrator = ProjectOrchestrator(
        llm=llm,
        baa=None,
        work_store=work_store,
        event_bus=event_bus,
    )
    stage("project_orchestrator_online")

    daydream_store = DaydreamStore(config.daydream_db)
    await daydream_store.connect()
    conflict_store = ConflictStore(config.conflict_db)
    await conflict_store.connect()
    stage("daydream_stores_online")

    curation_store = ConsolidationCurationStore(config.state_dir / "curation.db")
    await curation_store.connect()
    stage("curation_store_online")

    harness_store = HarnessStore(config.harness_db)
    await harness_store.connect()
    harness = AgenticHarness(
        store=harness_store,
        llm=llm,
        tracer=tracer,
        work_store=work_store,
        project_orchestrator=project_orchestrator,
    )
    stage("harness_online")

    tom_store = TomStore(config.tom_db)
    await tom_store.connect()
    stage("tom_store_online")

    plan_store = PlanStore(config.plans_db)
    await plan_store.connect()
    stage("plan_store_online")

    schedule_store = ScheduleStore(config.schedules_db)
    await schedule_store.connect()
    schedule_service = ScheduleService(schedule_store, tracer=tracer)
    stage("schedule_store_online")

    mcp_registry = None
    if config.mcp_servers:
        try:
            from opencas.tools.mcp_registry import MCPRegistry, MCPServerConfig

            configs = [MCPServerConfig(**server) for server in config.mcp_servers]
            mcp_registry = MCPRegistry(configs)
        except Exception:
            pass
    stage("mcp_registry_online", {"configured_servers": len(config.mcp_servers or [])})

    doctor = Doctor(context=None)
    health_monitor = HealthMonitor(
        doctor=doctor,
        event_bus=event_bus,
        interval_seconds=60.0,
        tracer=tracer,
    )
    stage("diagnostics_online")

    return RuntimeServiceBundle(
        plugin_store=plugin_store,
        skill_registry=skill_registry,
        plugin_lifecycle=plugin_lifecycle,
        ledger=ledger,
        readiness=readiness,
        project_orchestrator=project_orchestrator,
        relational=relational,
        daydream_store=daydream_store,
        conflict_store=conflict_store,
        curation_store=curation_store,
        harness=harness,
        tom_store=tom_store,
        plan_store=plan_store,
        schedule_store=schedule_store,
        schedule_service=schedule_service,
        mcp_registry=mcp_registry,
        doctor=doctor,
        health_monitor=health_monitor,
    )
