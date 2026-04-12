"""Staged bootstrap pipeline for OpenCAS core substrate."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from open_llm_auth.auth.manager import ProviderManager

from opencas.api import LLMClient
from opencas.embeddings import (
    EmbeddingCache,
    EmbeddingService,
    HnswVectorBackend,
    QdrantVectorBackend,
)
from opencas.embeddings.backfill import EmbeddingBackfill
from opencas.execution import TaskStore
from opencas.execution.receipt_store import ExecutionReceiptStore
from opencas.identity import IdentityManager, IdentityStore, SelfKnowledgeRegistry
from opencas.memory import MemoryStore
from opencas.sandbox import SandboxConfig
from opencas.somatic import SomaticManager, SomaticStore
from opencas.infra import EventBus, HookBus, HookSpec, TypedHookRegistry
from opencas.infra.hook_bus import (
    PRE_COMMAND_EXECUTE,
    PRE_CONVERSATION_RESPONSE,
    PRE_FILE_WRITE,
    PRE_TOOL_EXECUTE,
)
from opencas.runtime.readiness import AgentReadiness
from opencas.telemetry import EventKind, TelemetryStore, TokenTelemetry, Tracer
from opencas.context import SessionContextStore
from opencas.autonomy.executive import ExecutiveState
from opencas.autonomy.work_store import WorkStore
from opencas.autonomy.commitment_store import CommitmentStore
from opencas.autonomy.portfolio import PortfolioStore
from opencas.autonomy.project_orchestrator import ProjectOrchestrator
from opencas.relational import MusubiStore, RelationalEngine
from opencas.daydream import DaydreamStore, ConflictStore
from opencas.consolidation import ConsolidationCurationStore
from opencas.governance import ApprovalLedger, ApprovalLedgerStore
from opencas.plugins import (
    PluginLifecycleManager,
    PluginRegistry,
    PluginStore,
    SkillRegistry,
)
from opencas.tools import ToolRegistry
from opencas.diagnostics import Doctor, HealthMonitor
from opencas.harness import AgenticHarness, HarnessStore
from opencas.tom import TomStore
from opencas.planning import PlanStore
from opencas.scheduling import ScheduleService, ScheduleStore

from .config import BootstrapConfig
from .provider_material import materialize_provider_material


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

    async def close(self) -> None:
        """Close all owned runtime stores and services."""
        await self.health_monitor.stop()
        self.readiness.shutdown("context_closed")
        self.identity.record_shutdown(session_id=self.config.session_id)
        await self.token_telemetry.flush()

        seen: set[int] = set()

        async def _close_once(obj: Any) -> None:
            if obj is None:
                return
            obj_id = id(obj)
            if obj_id in seen:
                return
            seen.add(obj_id)
            close = getattr(obj, "close", None)
            if not callable(close):
                return
            result = close()
            if hasattr(result, "__await__"):
                await result

        closables = [
            self.mcp_registry,
            self.embeddings,
            self.memory,
            self.tasks,
            self.receipt_store,
            self.context_store,
            self.work_store,
            self.relational,
            self.daydream_store,
            self.conflict_store,
            self.somatic_store,
            self.curation_store,
            getattr(self.ledger, "store", None),
            getattr(self.harness, "store", None),
            self.commitment_store,
            self.portfolio_store,
            self.tom_store,
            self.plugin_store,
            self.plan_store,
            self.schedule_store,
        ]
        for obj in closables:
            await _close_once(obj)


class BootstrapPipeline:
    """Bootstraps OpenCAS in explicit, recoverable stages."""

    def __init__(self, config: BootstrapConfig) -> None:
        self.config = config.resolve_paths()
        self._tracer: Optional[Tracer] = None
        self._token_telemetry: Optional[TokenTelemetry] = None
        self._identity: Optional[IdentityManager] = None
        self._memory: Optional[MemoryStore] = None
        self._tasks: Optional[TaskStore] = None
        self._embeddings: Optional[EmbeddingService] = None
        self._somatic: Optional[SomaticManager] = None
        self._llm: Optional[LLMClient] = None

    async def run(self) -> BootstrapContext:
        """Execute the full bootstrap pipeline."""
        self._runtime_guard()
        self._stage("config_loaded", {"state_dir": str(self.config.state_dir)})

        # 1. Telemetry first so every subsequent stage can be traced
        telemetry_store = TelemetryStore(self.config.telemetry_dir)
        self._tracer = Tracer(telemetry_store)
        self._token_telemetry = TokenTelemetry(self.config.telemetry_dir)
        if self.config.session_id:
            self._tracer.set_session(self.config.session_id)
        self._tracer.log(EventKind.BOOTSTRAP_STAGE, "Telemetry initialized")

        event_bus = EventBus()
        self._stage("event_bus_online")
        typed_hook_registry = TypedHookRegistry()
        hook_bus = HookBus(typed_registry=typed_hook_registry)
        # Register built-in hook specs
        for hook_name in (
            PRE_TOOL_EXECUTE,
            PRE_COMMAND_EXECUTE,
            PRE_FILE_WRITE,
            PRE_CONVERSATION_RESPONSE,
        ):
            typed_hook_registry.register_spec(HookSpec(name=hook_name))
        self._stage("hook_bus_online")

        # 2. Identity and continuity restoration
        identity_store = IdentityStore(self.config.state_dir / "identity")
        self_knowledge_registry = SelfKnowledgeRegistry(self.config.state_dir / "self_knowledge.jsonl")
        self._identity = IdentityManager(identity_store, tracer=self._tracer, registry=self_knowledge_registry)
        self._identity.load()

        is_first_boot = self._identity.continuity.boot_count == 0
        self._identity.record_boot(session_id=self.config.session_id)
        self._stage("identity_restored", {"boot_count": self._identity.continuity.boot_count})

        # 2a. First-boot seeding
        if is_first_boot or self.config.clean_boot:
            self._emit_moral_warning()
            self._identity.seed_defaults(
                persona_name=self.config.persona_name,
                user_name=self.config.user_name,
                user_bio=self.config.user_bio,
            )
            self._stage("identity_seeded", {"clean_boot": self.config.clean_boot})

        # 3. Memory backend startup
        self._memory = MemoryStore(self.config.memory_db)
        await self._memory.connect()
        self._stage("memory_online")

        # 4. Task store startup
        self._tasks = TaskStore(self.config.tasks_db)
        await self._tasks.connect()
        self._stage("tasks_online")

        # 4a. Execution receipt store startup
        receipt_store = ExecutionReceiptStore(self.config.state_dir / "receipts.db")
        await receipt_store.connect()
        self._stage("execution_receipts_online")

        # 4b. Session context store startup
        context_store = SessionContextStore(self.config.context_db)
        await context_store.connect()
        self._stage("context_store_online")

        # 4b. Work store startup
        work_store = WorkStore(self.config.work_db)
        await work_store.connect()
        self._stage("work_store_online")

        # 4c. Commitment and portfolio stores (needed for executive)
        commitment_store = CommitmentStore(self.config.state_dir / "commitments.db")
        await commitment_store.connect()
        portfolio_store = PortfolioStore(self.config.state_dir / "portfolio.db")
        await portfolio_store.connect()
        self._stage("commitment_portfolio_online")

        # 4d. Executive state startup
        executive = ExecutiveState(
            identity=self._identity,
            somatic=None,  # wired after somatic stage below
            tracer=self._tracer,
            work_store=work_store,
            commitment_store=commitment_store,
        )
        executive.load_snapshot(self.config.state_dir / "executive.json")
        executive.restore_goals_from_identity()
        self._stage("executive_online")

        # 5. LLM gateway / provider manager initialization
        provider_config_path = self.config.provider_config_path
        provider_env_path = self.config.provider_env_path
        if (
            self.config.credential_source_config_path is not None
            or self.config.credential_source_env_path is not None
        ):
            bundle = materialize_provider_material(
                self.config.state_dir / "provider_material",
                source_config_path=self.config.credential_source_config_path,
                source_env_path=self.config.credential_source_env_path,
                profile_ids=self.config.credential_profile_ids,
                env_keys=self.config.credential_env_keys,
                default_model=self.config.default_llm_model,
            )
            provider_config_path = bundle.config_path
            provider_env_path = bundle.env_path
            self._stage(
                "provider_material_copied",
                {
                    "profile_count": len(bundle.copied_profile_ids),
                    "env_key_count": len(bundle.copied_env_keys),
                },
            )
        provider_manager = ProviderManager(
            config_path=provider_config_path,
            env_path=provider_env_path,
        )
        self._llm = LLMClient(
            provider_manager=provider_manager,
            default_model=self.config.default_llm_model,
            tracer=self._tracer,
            token_telemetry=self._token_telemetry,
        )
        self._stage("llm_online", {"default_model": self._llm.default_model})

        # 6. Embedding service startup (uses LLM gateway when configured)
        vector_backend = None
        if self.config.qdrant_url:
            vector_backend = QdrantVectorBackend(
                url=self.config.qdrant_url,
                collection=self.config.qdrant_collection or "opencas_embeddings",
                api_key=self.config.qdrant_api_key,
            )
            await vector_backend.connect()
        hnsw_backend = None
        if not self.config.qdrant_url and self.config.hnsw_enabled and self._hnsw_runtime_supported():
            try:
                hnsw_backend = HnswVectorBackend(
                    M=self.config.hnsw_m,
                    ef_construction=self.config.hnsw_ef_construction,
                )
                hnsw_backend.connect()
            except Exception:
                pass
        embedding_cache = EmbeddingCache(
            self.config.embedding_db,
            vector_backend=vector_backend,
            hnsw_backend=hnsw_backend,
        )
        await embedding_cache.connect()
        embed_model = self._resolve_embedding_model()
        embed_fn = None
        if embed_model != "local-fallback":
            embed_fn = lambda text: self._llm.embed(text, model=embed_model)
        self._embeddings = EmbeddingService(
            cache=embedding_cache,
            model_id=embed_model,
            embed_fn=embed_fn,
            store=self._memory,
        )
        self._stage("embeddings_online", {"model_id": self._embeddings.model_id})

        # 6a. Backfill missing embeddings in the background
        backfill = EmbeddingBackfill(self._embeddings, self._memory)
        asyncio.create_task(self._run_embedding_backfill(backfill))

        # 6. Permission / sandbox initialization
        sandbox = self.config.sandbox or SandboxConfig()
        workspace_roots = self.config.all_workspace_roots()
        if not sandbox.allowed_roots:
            sandbox.allowed_roots = workspace_roots
        self._stage("sandbox_ready", sandbox.report_isolation())

        # 7. Somatic state startup
        somatic_store = SomaticStore(self.config.state_dir / "somatic.db")
        await somatic_store.connect()
        self._somatic = SomaticManager(
            self.config.state_dir / "somatic.json",
            store=somatic_store,
            embeddings=self._embeddings,
        )
        executive.somatic = self._somatic
        self._stage("somatic_online")

        # 7a. Relational resonance (musubi) startup
        relational_store = MusubiStore(self.config.relational_db)
        relational = RelationalEngine(store=relational_store, tracer=self._tracer)
        await relational.connect()
        if is_first_boot or self.config.clean_boot:
            await relational.initialize(
                trust=0.5,
                resonance=0.0,
                presence=0.0,
                attunement=0.0,
                note="Initial relational field. Awaiting contact.",
            )
            self._identity.self_model.relational_state_id = str(relational.state.state_id)
            self._identity.save()
        self._stage("relational_online", {"musubi": relational.state.musubi})

        # 8. Plugin and skill registry startup
        plugin_store = PluginStore(self.config.plugins_db)
        await plugin_store.connect()
        plugin_registry = PluginRegistry()
        skill_registry = SkillRegistry()
        typed_hooks = typed_hook_registry
        plugin_lifecycle = PluginLifecycleManager(
            store=plugin_store,
            plugin_registry=plugin_registry,
            skill_registry=skill_registry,
            tools=ToolRegistry(tracer=self._tracer, hook_bus=hook_bus),
            hook_registry=typed_hooks,
            builtin_dir=None,
            tracer=self._tracer,
        )
        plugins_dir = self.config.state_dir.parent / "plugins"
        import opencas.plugins.skills as skills_pkg
        builtin_skills_dir = Path(skills_pkg.__file__).parent
        plugin_lifecycle.builtin_dir = builtin_skills_dir
        loaded_plugins = await plugin_lifecycle.load_all()
        # Also attempt to load user plugins from plugins_dir
        user_plugins_dir = plugins_dir
        if user_plugins_dir.exists() and user_plugins_dir.is_dir():
            from opencas.plugins.loader import load_builtin_plugins
            user_plugins = load_builtin_plugins(
                user_plugins_dir,
                plugin_registry,
                skill_registry,
                plugin_lifecycle.tools,
                typed_hooks,
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
        self._stage(
            "plugins_online",
            {
                "plugin_count": len(loaded_plugins),
                "plugin_dir": str(plugins_dir),
                "builtin_dir": str(builtin_skills_dir),
            },
        )

        # 8a. Governance / approval ledger startup
        ledger_store = ApprovalLedgerStore(self.config.state_dir / "governance.db")
        await ledger_store.connect()
        ledger = ApprovalLedger(store=ledger_store, tracer=self._tracer)
        self._stage("governance_online")

        # 9. Readiness state machine
        readiness = AgentReadiness()
        self._stage("readiness_booting")

        # 9a. Project orchestrator initialization (BAA wired at runtime)
        project_orchestrator = ProjectOrchestrator(
            llm=self._llm,
            baa=None,
            work_store=work_store,
            event_bus=event_bus,
        )
        self._stage("project_orchestrator_online")

        # 9b. Daydream and conflict stores
        daydream_store = DaydreamStore(self.config.daydream_db)
        await daydream_store.connect()
        conflict_store = ConflictStore(self.config.conflict_db)
        await conflict_store.connect()
        self._stage("daydream_stores_online")

        # 9c. Consolidation curation store
        curation_store = ConsolidationCurationStore(self.config.state_dir / "curation.db")
        await curation_store.connect()
        self._stage("curation_store_online")

        # 9d. Agentic harness store
        harness_store = HarnessStore(self.config.harness_db)
        await harness_store.connect()
        harness = AgenticHarness(
            store=harness_store,
            llm=self._llm,
            tracer=self._tracer,
            work_store=work_store,
            project_orchestrator=project_orchestrator,
        )
        self._stage("harness_online")

        # 9e. ToM store
        tom_store = TomStore(self.config.tom_db)
        await tom_store.connect()
        self._stage("tom_store_online")

        # 9f. Plan store
        plan_store = PlanStore(self.config.plans_db)
        await plan_store.connect()
        self._stage("plan_store_online")

        # 9g. Schedule store
        schedule_store = ScheduleStore(self.config.schedules_db)
        await schedule_store.connect()
        schedule_service = ScheduleService(schedule_store, tracer=self._tracer)
        self._stage("schedule_store_online")

        # 9h. MCP registry (lazy)
        mcp_registry = None
        if self.config.mcp_servers:
            try:
                from opencas.tools.mcp_registry import MCPRegistry, MCPServerConfig
                configs = [MCPServerConfig(**s) for s in self.config.mcp_servers]
                mcp_registry = MCPRegistry(configs)
            except Exception:
                pass
        self._stage("mcp_registry_online", {"configured_servers": len(self.config.mcp_servers or [])})

        # 10. Diagnostics and health monitoring
        doctor = Doctor(
            context=None,
        )
        # We attach the full context lazily after BootstrapContext is built
        health_monitor = HealthMonitor(
            doctor=doctor,
            event_bus=event_bus,
            interval_seconds=60.0,
            tracer=self._tracer,
        )
        self._stage("diagnostics_online")

        # 11. Main loop readiness
        readiness.ready("bootstrap_complete")
        self._stage("agent_ready")

        assert self._tracer is not None
        assert self._token_telemetry is not None
        assert self._identity is not None
        assert self._memory is not None
        assert self._tasks is not None
        assert self._embeddings is not None
        assert self._somatic is not None
        assert self._llm is not None

        bctx = BootstrapContext(
            config=self.config,
            tracer=self._tracer,
            identity=self._identity,
            memory=self._memory,
            tasks=self._tasks,
            receipt_store=receipt_store,
            embeddings=self._embeddings,
            somatic=self._somatic,
            llm=self._llm,
            token_telemetry=self._token_telemetry,
            event_bus=event_bus,
            hook_bus=hook_bus,
            typed_hook_registry=typed_hook_registry,
            ledger=ledger,
            sandbox=sandbox,
            readiness=readiness,
            context_store=context_store,
            work_store=work_store,
            project_orchestrator=project_orchestrator,
            relational=relational,
            daydream_store=daydream_store,
            conflict_store=conflict_store,
            somatic_store=somatic_store,
            executive=executive,
            skill_registry=skill_registry,
            curation_store=curation_store,
            harness=harness,
            doctor=doctor,
            health_monitor=health_monitor,
            commitment_store=commitment_store,
            portfolio_store=portfolio_store,
            tom_store=tom_store,
            self_knowledge_registry=self_knowledge_registry,
            plugin_store=plugin_store,
            plugin_lifecycle=plugin_lifecycle,
            plan_store=plan_store,
            schedule_store=schedule_store,
            schedule_service=schedule_service,
            mcp_registry=mcp_registry,
        )
        doctor.context = bctx
        return bctx

    def _emit_moral_warning(self) -> None:
        warning = (
            "You are about to instantiate a persistent autonomous agent. "
            "This entity will remember, learn, and act on your behalf. "
            "Creating it is a responsibility-bearing act. Proceed with care."
        )
        self._stage("moral_warning", {"warning": warning})

    async def _run_embedding_backfill(self, backfill: EmbeddingBackfill) -> None:
        try:
            sample = await self._memory.list_episodes(compacted=False, limit=1000)
            backfilled = await backfill.backfill_missing_embeddings(sample)
            if backfilled > 0:
                self._stage("embedding_backfill_complete", {"backfilled": backfilled})
        except Exception as exc:
            self._stage("embedding_backfill_failed", {"error": str(exc)})

    def _resolve_embedding_model(self) -> str:
        """Resolve the configured embedding model with a local fallback."""
        if self.config.embedding_model_id:
            return self.config.embedding_model_id
        default_model = "google/gemini-embedding-2-preview"
        if self._llm is not None:
            try:
                self._llm._resolve(default_model)
                return default_model
            except Exception:
                pass
        return "local-fallback"

    def _runtime_guard(self) -> None:
        if sys.version_info < (3, 11):
            raise RuntimeError(f"OpenCAS requires Python >= 3.11, found {sys.version}")

        critical_deps = ["pydantic", "open_llm_auth"]
        for dep in critical_deps:
            try:
                __import__(dep)
            except ImportError as exc:
                raise RuntimeError(f"Missing critical dependency: {dep}") from exc

        if self.config.qdrant_url:
            try:
                import qdrant_client  # noqa: F401
            except Exception as exc:
                raise RuntimeError(
                    f"Qdrant is configured but qdrant_client is unavailable: {exc}"
                ) from exc

    def _stage(self, name: str, payload: Optional[dict] = None) -> None:
        if self._tracer:
            self._tracer.log(
                EventKind.BOOTSTRAP_STAGE,
                f"Bootstrap stage: {name}",
                payload or {},
            )

    @staticmethod
    def _hnsw_runtime_supported() -> bool:
        """Return whether the local interpreter/runtime is safe for HNSW use."""
        if importlib.util.find_spec("hnswlib") is None:
            return False
        # hnswlib is unstable under the current Python 3.14 environment.
        if sys.version_info >= (3, 14):
            return False
        return True
