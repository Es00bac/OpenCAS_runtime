from __future__ import annotations
"""Staged bootstrap pipeline for OpenCAS core substrate."""


import asyncio
import sys
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
from opencas.execution.receipt_store import ExecutionReceiptStore
from opencas.identity import IdentityManager, IdentityStore, SelfKnowledgeRegistry
from opencas.sandbox import SandboxConfig
from opencas.somatic import SomaticManager, SomaticStore
from opencas.infra import EventBus, HookBus, HookSpec, TypedHookRegistry
from opencas.infra.hook_bus import (
    PRE_COMMAND_EXECUTE,
    PRE_CONVERSATION_RESPONSE,
    PRE_FILE_WRITE,
    PRE_TOOL_EXECUTE,
)
from opencas.telemetry import EventKind, TelemetryStore, TokenTelemetry, Tracer

from .config import BootstrapConfig
from .context import BootstrapContext
from .pipeline_support import (
    emit_moral_warning,
    hnsw_runtime_supported,
    resolve_embedding_model,
    run_embedding_backfill,
    runtime_guard,
    stage,
)
from .pipeline_context import build_bootstrap_context, initialize_workspace_index
from .pipeline_services import initialize_runtime_services
from .pipeline_stores import initialize_runtime_stores
from .provider_material import materialize_provider_material


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

        stores = await initialize_runtime_stores(
            self.config,
            identity=self._identity,
            tracer=self._tracer,
            stage=self._stage,
        )
        self._memory = stores.memory
        self._tasks = stores.tasks
        receipt_store = stores.receipt_store
        context_store = stores.context_store
        work_store = stores.work_store
        commitment_store = stores.commitment_store
        portfolio_store = stores.portfolio_store
        executive = stores.executive

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
            model_routing=self.config.model_routing,
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
        backfill_task = asyncio.create_task(self._run_embedding_backfill(backfill))

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

        services = await initialize_runtime_services(
            self.config,
            identity=self._identity,
            llm=self._llm,
            tracer=self._tracer,
            somatic=self._somatic,
            event_bus=event_bus,
            hook_bus=hook_bus,
            typed_hook_registry=typed_hook_registry,
            work_store=work_store,
            stage=self._stage,
            is_first_boot=is_first_boot,
            clean_boot=self.config.clean_boot,
        )
        relational = services.relational
        plugin_store = services.plugin_store
        skill_registry = services.skill_registry
        plugin_lifecycle = services.plugin_lifecycle
        ledger = services.ledger
        readiness = services.readiness
        project_orchestrator = services.project_orchestrator
        daydream_store = services.daydream_store
        conflict_store = services.conflict_store
        curation_store = services.curation_store
        harness = services.harness
        tom_store = services.tom_store
        plan_store = services.plan_store
        schedule_store = services.schedule_store
        schedule_service = services.schedule_service
        mcp_registry = services.mcp_registry
        doctor = services.doctor
        health_monitor = services.health_monitor

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
        workspace_index = await initialize_workspace_index(
            self.config,
            self._embeddings,
            self._llm,
        )

        bctx = build_bootstrap_context(
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
            workspace_index=workspace_index,
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
            background_tasks=(backfill_task,),
        )
        doctor.context = bctx
        return bctx

    def _emit_moral_warning(self) -> None:
        emit_moral_warning(self._stage)

    async def _run_embedding_backfill(self, backfill: EmbeddingBackfill) -> None:
        await run_embedding_backfill(backfill, self._memory, self._stage)

    def _resolve_embedding_model(self) -> str:
        return resolve_embedding_model(self.config, self._llm)

    def _runtime_guard(self) -> None:
        runtime_guard(self.config)

    def _stage(self, name: str, payload: Optional[dict] = None) -> None:
        stage(self._tracer, name, payload)

    @staticmethod
    def _hnsw_runtime_supported() -> bool:
        return hnsw_runtime_supported()
