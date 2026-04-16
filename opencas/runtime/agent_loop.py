"""Main agent runtime loop for OpenCAS."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from opencas.api import LLMClient
from opencas.autonomy import WorkObject, WorkStage
from opencas.autonomy.boredom import BoredomPhysics
from opencas.autonomy.commitment import Commitment
from opencas.autonomy.executive import ExecutiveState
from opencas.autonomy.commitment_extraction import SelfCommitmentCandidate
from opencas.autonomy.creative_ladder import CreativeLadder
from opencas.autonomy.models import ActionRequest, ActionRiskTier
from opencas.autonomy.portfolio import PortfolioStore
from opencas.autonomy.self_approval import SelfApprovalLadder
from opencas.autonomy.spark_router import SparkRouter
from opencas.infra import BaaCompletedEvent
from opencas.bootstrap import BootstrapContext
from opencas.tools import ToolRegistry, ToolUseContext, ToolUseLoop
from opencas.memory import Episode, EpisodeKind, MemoryStore
from opencas.memory.fabric.graph import EpisodeGraph
from opencas.telemetry import Tracer
from opencas.tom import ToMEngine

from opencas.execution import (
    BoundedAssistantAgent,
    BrowserSupervisor,
    ProcessSupervisor,
    PtySupervisor,
    ReliabilityCoordinator,
)
from opencas.context import ContextBuilder, MemoryRetriever, MessageRole
from opencas.compaction import ConversationCompactor
from opencas.consolidation import NightlyConsolidationEngine
from opencas.identity import IdentityRebuilder

from opencas.daydream import (
    ConflictRegistry,
    DaydreamStore,
    ReflectionEvaluator,
    ReflectionResolver,
    SelfCompassionMirror,
)
from opencas.daydream.spark_evaluator import SparkEvaluator
from opencas.runtime.agent_profile import get_agent_profile
from opencas.runtime.readiness import AgentReadiness, ReadinessState
from opencas.runtime.single_instance import SingleInstanceLock
from opencas.somatic import SomaticModulators

from .daydream import DaydreamGenerator
from .conversation_turns import (
    execute_conversation_tool_loop,
    finalize_assistant_turn,
    handle_refusal_turn,
    persist_tool_loop_messages,
    persist_user_turn,
)
from .cycle_phases import (
    drain_executive_cycle_queue,
    enqueue_promoted_cycle_work,
    evaluate_workspace_intervention,
)
from .lifecycle import (
    run_autonomous_runtime,
    run_autonomous_with_server_runtime,
)
from .episodic_runtime import (
    capture_runtime_self_commitments,
    extract_runtime_goal_directives,
    extract_runtime_self_commitments,
    record_runtime_episode,
    run_runtime_continuity_check,
)
from .reflection_runtime import (
    build_runtime_metacognition_status,
    rebuild_runtime_identity,
    run_runtime_daydream,
    run_runtime_daydream_inner,
)
from .maintenance_runtime import (
    close_runtime_stores,
    extract_runtime_response_content,
    handle_runtime_baa_completed,
    maybe_compact_runtime_session,
    maybe_record_runtime_somatic_snapshot,
    run_runtime_consolidation,
    sync_runtime_executive_snapshot,
    trace_runtime_event,
)
from .status_views import (
    build_consolidation_status,
    build_control_plane_status,
    build_workflow_status,
)
from .telegram_runtime import (
    approve_runtime_telegram_pairing,
    build_runtime_telegram_service,
    configure_runtime_telegram,
    get_runtime_telegram_status,
    runtime_telegram_settings,
    start_runtime_telegram,
)
from .tool_registration import register_runtime_default_tools
from .runtime_setup import (
    initialize_runtime_autonomy,
    initialize_runtime_channels,
    initialize_runtime_execution,
    initialize_runtime_memory_surfaces,
)
from .tool_runtime import (
    build_runtime_tool_use_context,
    disable_runtime_plugin,
    discover_and_register_mcp_tools,
    enable_runtime_plugin,
    execute_runtime_tool,
    handle_runtime_action,
    hydrate_runtime_tool_use_context,
    install_runtime_plugin,
    make_mcp_list_servers_adapter,
    make_mcp_register_adapter,
    register_mcp_server_tools,
    submit_runtime_repair,
    uninstall_runtime_plugin,
)
from opencas.telegram_config import TelegramRuntimeConfig
from opencas.telegram_integration import TelegramBotService


class AgentRuntime:
    """Coordinates conversation, memory, creative ladder, and execution."""

    def __init__(self, context: BootstrapContext) -> None:
        self.ctx = context
        self.tracer = context.tracer
        self.readiness: AgentReadiness = context.readiness
        self.memory: MemoryStore = context.memory
        self.llm: LLMClient = context.llm
        self.agent_profile = get_agent_profile(context.config.agent_profile_id)
        self._instance_lock = SingleInstanceLock(context.config.state_dir)

        initialize_runtime_autonomy(self, context)
        initialize_runtime_execution(self, context)
        initialize_runtime_memory_surfaces(self, context)
        initialize_runtime_channels(self, context)

        # Activity tracking — what the runtime is currently doing (operator-visible)
        self._activity: str = "idle"
        self._activity_since: datetime = datetime.now(timezone.utc)
        self._last_consolidation_result: Optional[Dict[str, Any]] = None

    def _set_activity(self, activity: str) -> None:
        """Update the observable runtime activity label."""
        self._activity = activity
        self._activity_since = datetime.now(timezone.utc)

    def _build_telegram_service(self) -> None:
        """Rebuild the Telegram service handle from the persisted runtime config."""
        self._telegram = build_runtime_telegram_service(self)

    async def start_telegram(self) -> None:
        """Start the Telegram polling service if configured."""
        await start_runtime_telegram(self)

    @property
    def telegram_settings(self) -> TelegramRuntimeConfig:
        return runtime_telegram_settings(self)

    async def telegram_status(self) -> Dict[str, Any]:
        return await get_runtime_telegram_status(self)

    async def configure_telegram(self, settings: TelegramRuntimeConfig) -> Dict[str, Any]:
        return await configure_runtime_telegram(self, settings)

    async def approve_telegram_pairing(self, code: str) -> bool:
        return await approve_runtime_telegram_pairing(self, code)

    def _register_default_tools(self) -> None:
        register_runtime_default_tools(self)

    def _register_skills(self) -> None:
        skill_registry = getattr(self.ctx, "skill_registry", None)
        if not skill_registry:
            return
        for skill in skill_registry.list_skills():
            if skill.register_fn is not None:
                try:
                    before = set(self.tools._tools.keys())
                    skill.register_fn(self.tools)
                    after = set(self.tools._tools.keys())
                    for tool_name in after - before:
                        if skill.plugin_id is not None:
                            self.tools._plugin_tools[tool_name] = skill.plugin_id
                except Exception as exc:
                    self._trace(
                        "skill_register_failed",
                        {"skill_id": skill.skill_id, "error": str(exc)},
                    )

    async def converse(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        user_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Process one conversational turn while delegating phase logic to runtime helpers."""
        self._set_activity("conversing")
        sid = session_id or self.ctx.config.session_id or "default"
        persisted_user_meta = dict(user_meta or {})

        # Refusal is handled first so unsafe turns never leak into the tool loop.
        from opencas.refusal.models import ConversationalRequest
        conv_request = ConversationalRequest(text=user_input, session_id=sid)
        refusal = self.refusal_gate.evaluate(conv_request)
        if refusal.refused:
            return await handle_refusal_turn(
                self,
                session_id=sid,
                user_input=user_input,
                user_meta=persisted_user_meta,
                refusal=refusal,
            )

        await persist_user_turn(
            self,
            session_id=sid,
            user_input=user_input,
            user_meta=persisted_user_meta,
        )
        artifacts = await execute_conversation_tool_loop(
            self,
            session_id=sid,
            user_input=user_input,
        )
        await persist_tool_loop_messages(
            self,
            session_id=sid,
            artifacts=artifacts,
        )
        await finalize_assistant_turn(
            self,
            session_id=sid,
            user_input=user_input,
            content=artifacts.content,
            manifest=artifacts.manifest,
        )

        self._trace(
            "converse",
            {
                "session_id": sid,
                "input_len": len(user_input),
                "token_estimate": artifacts.manifest.token_estimate,
            },
        )
        self.boredom.record_activity()
        return artifacts.content

    async def run_daydream(self) -> Dict[str, Any]:
        """Generate daydreams when idle or tense."""
        return await run_runtime_daydream(self)

    async def _run_daydream_inner(self) -> Dict[str, Any]:
        """Inner implementation of run_daydream (wrapped for activity tracking)."""
        return await run_runtime_daydream_inner(self)

    async def run_cycle(self) -> Dict[str, Any]:
        """Run one creative/execution cycle."""
        self._set_activity("cycling")
        await self.executive.restore_queue()
        result = self.creative.run_cycle()

        # Keep run_cycle() focused on phase ordering; the helper module owns the
        # promotion/intervention/drain details and their local invariants.
        promoted_tasks = await enqueue_promoted_cycle_work(self)

        daydream_work_objects: list[WorkObject] = []
        reflections: list[str] = []

        # Relational: record creative collaboration when work is promoted
        if promoted_tasks > 0 and hasattr(self.ctx, "relational") and self.ctx.relational:
            await self.ctx.relational.record_creative_collab(success=True)

        # Somatic update for creative work and daydreaming
        if promoted_tasks > 0:
            self.ctx.somatic.bump_from_work(
                intensity=min(0.25, promoted_tasks * 0.05), success=True
            )
        if reflections:
            await self._maybe_record_somatic_snapshot(
                source="daydream",
                trigger_event_id=str(reflections[-1].reflection_id)
                if reflections
                else None,
            )
        elif promoted_tasks > 0:
            await self._maybe_record_somatic_snapshot(source="creative_cycle")

        # Harness objective cycle
        harness_result: Dict[str, Any] = {}
        if self.harness:
            try:
                harness_result = await self.harness.run_objective_cycle(max_active_loops=3)
            except Exception as exc:
                self._trace("harness_cycle_error", {"error": str(exc)})

        # Build executive workspace and evaluate intervention policy
        workspace_outcome = await evaluate_workspace_intervention(self)
        intervention_decision = workspace_outcome.decision if workspace_outcome else None

        # Drain executive queue: submit ready work to BAA
        drained_count = await drain_executive_cycle_queue(self)

        keepers = sum(1 for r in reflections if getattr(r, "keeper", False))
        self._trace(
            "run_cycle",
            {
                "promoted": result["promoted"],
                "demoted": result["demoted"],
                "enqueued": promoted_tasks,
                "daydreams": len(daydream_work_objects),
                "reflections": len(reflections),
                "keepers": keepers,
                "drained": drained_count,
                "harness": harness_result,
                "intervention": intervention_decision.kind.value if intervention_decision else None,
            },
        )
        return {
            "creative": result,
            "enqueued": promoted_tasks,
            "daydreams": len(daydream_work_objects),
            "reflections": len(reflections),
            "keepers": keepers,
            "drained": drained_count,
            "harness": harness_result,
            "intervention": intervention_decision.model_dump(mode="json") if intervention_decision else None,
        }

    async def maybe_compact_session(self, session_id: str, tail_size: int = 10) -> None:
        """Compact old episodes for a session if there are enough of them."""
        await maybe_compact_runtime_session(self, session_id, tail_size=tail_size)

    async def run_consolidation(self) -> Dict[str, Any]:
        """Run the nightly consolidation engine."""
        return await run_runtime_consolidation(self)

    def check_metacognition(self) -> Dict[str, Any]:
        """Run a metacognitive consistency check via ToM."""
        return build_runtime_metacognition_status(self)

    async def rebuild_identity(self) -> Dict[str, Any]:
        """Rebuild identity from autobiographical memory and apply to self-model."""
        return await rebuild_runtime_identity(self)

    async def _build_tool_use_context(self, session_id: str) -> ToolUseContext:
        """Create a ToolUseContext, restoring active plan state if present."""
        ctx = build_runtime_tool_use_context(self, session_id)
        return await hydrate_runtime_tool_use_context(self, ctx)

    async def _discover_and_register_mcp_tools(self) -> List[str]:
        """Eagerly discover and register all MCP tools."""
        return await discover_and_register_mcp_tools(self)

    async def register_mcp_server_tools(self, server_name: str) -> List[str]:
        """Lazy-register tools from a specific MCP server."""
        return await register_mcp_server_tools(self, server_name)

    def _make_mcp_list_servers_adapter(self):
        return make_mcp_list_servers_adapter(self)

    def _make_mcp_register_adapter(self):
        return make_mcp_register_adapter(self)

    async def execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool through the registry after self-approval."""
        return await execute_runtime_tool(self, name, args)

    async def install_plugin(self, path: Path | str) -> Optional[Any]:
        """Install a plugin from a directory or manifest file."""
        return await install_runtime_plugin(self, path)

    async def uninstall_plugin(self, plugin_id: str) -> None:
        """Uninstall a plugin."""
        await uninstall_runtime_plugin(self, plugin_id)

    async def enable_plugin(self, plugin_id: str) -> None:
        """Enable a plugin."""
        await enable_runtime_plugin(self, plugin_id)

    async def disable_plugin(self, plugin_id: str) -> None:
        """Disable a plugin."""
        await disable_runtime_plugin(self, plugin_id)

    async def submit_repair(self, task) -> Any:
        """Submit a repair task to the bounded assistant agent and return a future."""
        return await submit_runtime_repair(self, task)

    async def handle_action(self, request: ActionRequest) -> Dict[str, Any]:
        """Evaluate an action through the self-approval ladder."""
        return await handle_runtime_action(self, request)

    async def _continuity_check(self) -> None:
        """Phase 9: Continuous Present — run at boot to decay score and generate monologue."""
        await run_runtime_continuity_check(self)

    async def run_autonomous(
        self,
        cycle_interval: int = 300,
        consolidation_interval: int = 86400,
    ) -> None:
        """Start background scheduler and block until interrupted."""
        await run_autonomous_runtime(
            self,
            cycle_interval=cycle_interval,
            consolidation_interval=consolidation_interval,
        )

    async def run_autonomous_with_server(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        cycle_interval: int = 300,
        consolidation_interval: int = 86400,
    ) -> None:
        """Run scheduler + FastAPI server together, shutting down gracefully on signal."""
        await run_autonomous_with_server_runtime(
            self,
            host=host,
            port=port,
            cycle_interval=cycle_interval,
            consolidation_interval=consolidation_interval,
        )

    async def import_bulma(
        self,
        bulma_state_dir: Path,
        checkpoint_path: Optional[Path] = None,
        curated_workspace_dir: Optional[Path] = None,
    ) -> Any:
        """Run a one-way cutover import from an OpenBulma v4 state directory."""
        from opencas.legacy.importer import BulmaImportTask
        task = BulmaImportTask(
            bulma_state_dir,
            runtime=self,
            checkpoint_store=checkpoint_path,
            curated_workspace_dir=curated_workspace_dir,
        )
        return await task.run()

    async def _maybe_record_somatic_snapshot(
        self,
        source: str,
        trigger_event_id: Optional[str] = None,
    ) -> None:
        await maybe_record_runtime_somatic_snapshot(
            self,
            source,
            trigger_event_id=trigger_event_id,
        )

    async def _on_baa_completed(self, event: BaaCompletedEvent) -> None:
        """Handle BAA task completion: resolve goals and persist executive state."""
        await handle_runtime_baa_completed(self, event)

    def _sync_executive_snapshot(self) -> None:
        sync_runtime_executive_snapshot(self)

    @staticmethod
    def _extract_goal_directives(text: str) -> tuple[List[str], Optional[str], List[str]]:
        """Heuristically extract goals, intention, and drop-requests from user text."""
        return extract_runtime_goal_directives(text)

    @staticmethod
    def _extract_self_commitments(text: str) -> List[SelfCommitmentCandidate]:
        """Extract normalized future-action self-commitments from assistant text."""
        return extract_runtime_self_commitments(text)

    async def _capture_self_commitments(
        self,
        content: str,
        session_id: str,
    ) -> List[Commitment]:
        """Persist normalized self-commitments and mirror them into ToM/somatic state."""
        return await capture_runtime_self_commitments(self, content, session_id)

    async def _close_stores(self) -> None:
        """Gracefully close all SQLite stores."""
        await close_runtime_stores(self)

    def control_plane_status(self) -> Dict[str, Any]:
        """Return a monitoring snapshot of workspace, sandbox, and execution state."""
        return build_control_plane_status(self)

    def consolidation_status(self) -> Dict[str, Any]:
        """Return the latest known nightly consolidation summary."""
        return build_consolidation_status(self)

    async def workflow_status(
        self,
        limit: int = 10,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a summarized view of current higher-level workflow state."""
        return await build_workflow_status(self, limit=limit, project_id=project_id)

    async def _record_episode(
        self,
        content: str,
        kind: EpisodeKind,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Episode:
        return await record_runtime_episode(
            self,
            content,
            kind,
            session_id=session_id,
            role=role,
        )

    async def _link_episode_to_previous(self, episode: Episode) -> None:
        """Create graph edges between this episode and the most recent prior episode."""
        from .episodic_runtime import link_runtime_episode_to_previous

        await link_runtime_episode_to_previous(self, episode)

    def _extract_content(self, response: Dict[str, Any]) -> str:
        return extract_runtime_response_content(response)

    def _trace(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        trace_runtime_event(self, event, payload)
