"""Main agent runtime loop for OpenCAS."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from opencas.api import LLMClient
from opencas.autonomy import WorkObject, WorkStage
from opencas.autonomy.boredom import BoredomPhysics
from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.autonomy.creative_ladder import CreativeLadder
from opencas.autonomy.intervention import InterventionKind, InterventionPolicy
from opencas.autonomy.models import ActionRequest, ActionRiskTier, ApprovalLevel
from opencas.autonomy.portfolio import PortfolioCluster, PortfolioStore, build_fascination_key
from opencas.autonomy.self_approval import SelfApprovalLadder
from opencas.autonomy.spark_router import SparkRouter, SparkRung
from opencas.autonomy.workspace import ExecutiveWorkspace, ExecutionMode, WorkspaceAffinity
from opencas.infra import BaaCompletedEvent
from opencas.bootstrap import BootstrapContext
from opencas.tools import (
    FileSystemToolAdapter,
    ShellToolAdapter,
    ToolRegistry,
    ToolUseContext,
    ToolUseLoop,
    UserInputRequired,
)
from opencas.tools.adapters.agent import AgentToolAdapter
from opencas.tools.adapters.browser import BrowserToolAdapter
from opencas.tools.adapters.edit import EditToolAdapter
from opencas.tools.adapters.interactive import InteractiveToolAdapter
from opencas.tools.adapters.lsp import LspToolAdapter
from opencas.tools.adapters.plan import PlanToolAdapter
from opencas.tools.adapters.process import ProcessToolAdapter
from opencas.tools.adapters.pty import PtyToolAdapter
from opencas.tools.adapters.repl import ReplToolAdapter
from opencas.tools.adapters.runtime_state import RuntimeStateToolAdapter
from opencas.tools.adapters.search import SearchToolAdapter
from opencas.tools.adapters.workflow import WorkflowToolAdapter
from opencas.tools.adapters.workflow_state import WorkflowStateToolAdapter
from opencas.tools.adapters.web import WebToolAdapter
from opencas.tools.validation import create_default_tool_validation_pipeline
from opencas.memory import EdgeKind, Episode, EpisodeEdge, EpisodeKind, Memory, MemoryStore
from opencas.memory.fabric.graph import EpisodeGraph
from opencas.telemetry import EventKind, Tracer
from opencas.tom import BeliefSubject, ToMEngine

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
    DaydreamReflection,
    DaydreamStore,
    ReflectionEvaluator,
    ReflectionResolver,
    SelfCompassionMirror,
)
from opencas.daydream.spark_evaluator import SparkEvaluator
from opencas.runtime.agent_profile import get_agent_profile
from opencas.runtime.readiness import AgentReadiness, ReadinessState
from opencas.sandbox import DockerSandbox, SandboxMode
from opencas.somatic import AppraisalEventType, SomaticModulators

from .daydream import DaydreamGenerator
from .scheduler import AgentScheduler
from opencas.telegram_config import (
    TelegramRuntimeConfig,
    load_telegram_runtime_config,
    save_telegram_runtime_config,
)
from opencas.telegram_integration import TelegramBotService

import uvicorn
from opencas.api.server import create_app


class AgentRuntime:
    """Coordinates conversation, memory, creative ladder, and execution."""

    def __init__(self, context: BootstrapContext) -> None:
        self.ctx = context
        self.tracer = context.tracer
        self.readiness: AgentReadiness = context.readiness
        self.memory: MemoryStore = context.memory
        self.llm: LLMClient = context.llm
        self.agent_profile = get_agent_profile(context.config.agent_profile_id)

        self.executive = context.executive
        self.creative = CreativeLadder(
            executive=self.executive,
            embeddings=context.embeddings,
            tracer=self.tracer,
            work_store=context.work_store,
            relational=getattr(context, "relational", None),
            task_store=getattr(context, "tasks", None),
        )
        self.orchestrator = context.project_orchestrator
        from opencas.refusal import ConversationalRefusalGate

        self.approval = SelfApprovalLadder(
            identity=context.identity,
            somatic=context.somatic,
            tracer=self.tracer,
            relational=getattr(context, "relational", None),
            ledger=getattr(context, "ledger", None),
        )
        self.refusal_gate = ConversationalRefusalGate(
            approval=self.approval,
            hook_bus=self.ctx.hook_bus,
        )
        self.spark_evaluator = SparkEvaluator(
            embeddings=context.embeddings,
            work_store=getattr(context, "work_store", None),
            executive=self.executive,
            somatic=context.somatic,
            relational=getattr(context, "relational", None),
            novelty_floor=0.3,
        )
        self.daydream = DaydreamGenerator(
            llm=self.llm,
            memory=self.memory,
            tracer=self.tracer,
            identity=context.identity,
            somatic=context.somatic,
            relational=getattr(context, "relational", None),
            daydream_store=getattr(context, "daydream_store", None),
            spark_evaluator=self.spark_evaluator,
        )
        self.reflection_evaluator = ReflectionEvaluator()
        self.reflection_resolver = ReflectionResolver(mirror=SelfCompassionMirror())
        self.conflict_registry = None
        if getattr(self.ctx, "conflict_store", None):
            self.conflict_registry = ConflictRegistry(self.ctx.conflict_store)
        self._last_daydream_time: Optional[datetime] = None
        self.boredom = BoredomPhysics()
        self.spark_router = SparkRouter()
        self.commitment_store = getattr(context, "commitment_store", None)
        self.portfolio_store = getattr(context, "portfolio_store", None)
        self.schedule_service = getattr(context, "schedule_service", None)
        if self.schedule_service is not None:
            self.schedule_service.runtime = self
        self.tom = ToMEngine(
            identity=context.identity,
            tracer=self.tracer,
            store=getattr(context, "tom_store", None),
        )
        self.process_supervisor = ProcessSupervisor()
        self.pty_supervisor = PtySupervisor()
        self.browser_supervisor = BrowserSupervisor()
        self.plugin_lifecycle = getattr(context, "plugin_lifecycle", None)
        if self.plugin_lifecycle is not None:
            self.tools = self.plugin_lifecycle.tools
        else:
            self.tools = ToolRegistry(tracer=self.tracer, hook_bus=self.ctx.hook_bus)
        self._register_default_tools()
        self._register_skills()
        self.baa = BoundedAssistantAgent(
            tools=self.tools,
            llm=self.llm,
            tracer=self.tracer,
            max_concurrent=2,
            store=context.tasks,
            event_bus=context.event_bus,
            receipt_store=getattr(context, "receipt_store", None),
            runtime=self,
        )
        self.orchestrator.baa = self.baa
        self.tool_loop = ToolUseLoop(
            llm=self.llm,
            tools=self.tools,
            approval=self.approval,
            tracer=self.tracer,
        )
        if context.event_bus:
            context.event_bus.subscribe(BaaCompletedEvent, self._on_baa_completed)
        self.reliability = None
        self.scheduler: Optional["AgentScheduler"] = None  # set by run_live / run_with_server
        if context.event_bus:
            self.reliability = ReliabilityCoordinator(
                event_bus=context.event_bus,
                window_size=10,
                failure_threshold=0.7,
                cooldown_seconds=300,
            )
        self.episode_graph = EpisodeGraph(store=self.memory)
        self.rebuilder = IdentityRebuilder(
            memory=self.memory,
            episode_graph=self.episode_graph,
            llm=self.llm,
        )
        self.retriever = MemoryRetriever(
            memory=self.memory,
            embeddings=context.embeddings,
            episode_graph=self.episode_graph,
            somatic_manager=context.somatic,
            relational_engine=context.relational,
        )
        self.modulators = SomaticModulators(context.somatic.state)
        self.builder = ContextBuilder(
            store=context.context_store,
            retriever=self.retriever,
            identity=context.identity,
            executive=self.executive,
            agent_profile=self.agent_profile,
            modulators=self.modulators,
            relational=getattr(context, "relational", None),
            tom=self.tom,
        )
        self.compactor = ConversationCompactor(
            memory=self.memory,
            llm=self.llm,
            tracer=self.tracer,
            context_store=self.ctx.context_store,
        )
        self.consolidation = NightlyConsolidationEngine(
            memory=self.memory,
            embeddings=context.embeddings,
            llm=self.llm,
            identity=context.identity,
            tracer=self.tracer,
            curation_store=getattr(context, "curation_store", None),
            tom_store=getattr(context, "tom_store", None),
        )
        self.harness = getattr(context, "harness", None)
        if self.harness:
            self.harness.baa = self.baa

        # Telegram integration — load persisted config and build service if enabled
        self._telegram_config: TelegramRuntimeConfig = load_telegram_runtime_config(
            context.config.state_dir
        )
        self._telegram: Optional[TelegramBotService] = None
        self._build_telegram_service()

        # Activity tracking — what the runtime is currently doing (operator-visible)
        self._activity: str = "idle"
        self._activity_since: datetime = datetime.now(timezone.utc)

    def _set_activity(self, activity: str) -> None:
        """Update the observable runtime activity label."""
        self._activity = activity
        self._activity_since = datetime.now(timezone.utc)

    def _build_telegram_service(self) -> None:
        """Instantiate TelegramBotService from current _telegram_config (does not start it)."""
        cfg = self._telegram_config
        if cfg.enabled and cfg.bot_token:
            self._telegram = TelegramBotService(
                runtime=self,
                enabled=True,
                token=cfg.bot_token,
                state_dir=self.ctx.config.state_dir,
                dm_policy=cfg.dm_policy,
                allow_from=cfg.allow_from,
                poll_interval_seconds=cfg.poll_interval_seconds,
                pairing_ttl_seconds=cfg.pairing_ttl_seconds,
                api_base_url=cfg.api_base_url,
                tracer=self.tracer,
            )
        else:
            self._telegram = None

    async def start_telegram(self) -> None:
        """Start the Telegram polling service if configured. Errors are logged, not raised."""
        if self._telegram is None:
            return
        try:
            await self._telegram.start()
            self._trace("telegram_started", {})
        except Exception as exc:
            self._trace("telegram_start_failed", {"error": str(exc)})

    @property
    def telegram_settings(self) -> TelegramRuntimeConfig:
        return self._telegram_config

    async def telegram_status(self) -> Dict[str, Any]:
        if self._telegram is not None:
            return await self._telegram.status()
        return {
            "enabled": self._telegram_config.enabled,
            "configured": bool(self._telegram_config.bot_token),
            "token_configured": bool(self._telegram_config.bot_token),
            "running": False,
            "dm_policy": self._telegram_config.dm_policy,
            "allow_from": self._telegram_config.allow_from,
            "bot": {"id": None, "username": None, "first_name": None, "link": None},
            "last_update_id": None,
            "last_error": None,
            "pairings": {},
        }

    async def configure_telegram(self, settings: TelegramRuntimeConfig) -> Dict[str, Any]:
        # Stop existing service
        if self._telegram is not None:
            try:
                await self._telegram.stop()
            except Exception:
                pass
        self._telegram_config = settings
        save_telegram_runtime_config(self.ctx.config.state_dir, settings)
        self._build_telegram_service()
        await self.start_telegram()
        status = await self.telegram_status()
        status["saved"] = True
        return status

    async def approve_telegram_pairing(self, code: str) -> bool:
        if self._telegram is None:
            return False
        result = await self._telegram.approve_pairing(code)
        return result is not None

    def _register_default_tools(self) -> None:
        roots = [str(r) for r in self.ctx.sandbox.allowed_roots]
        if not roots:
            roots = [str(self.ctx.config.primary_workspace_root())]
        default_cwd = roots[0]
        validation = create_default_tool_validation_pipeline(
            roots=roots,
            max_write_bytes=500_000,
        )
        self.tools.validation_pipeline = validation
        fs = FileSystemToolAdapter(allowed_roots=roots)
        self.tools.register(
            "fs_read_file",
            "Read the contents of a file",
            fs,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to read.",
                    }
                },
                "required": ["file_path"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "fs_list_dir",
            "List the contents of a directory",
            fs,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "Absolute path to the directory to list.",
                    }
                },
                "required": ["dir_path"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "fs_write_file",
            "Write content to a file. Overwrites the file if it exists.",
            fs,
            ActionRiskTier.WORKSPACE_WRITE,
            {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content to write to the file.",
                    },
                },
                "required": ["file_path", "content"],
            },
            plugin_id="core",
        )
        docker_sandbox = None
        if self.ctx.sandbox.mode == SandboxMode.DOCKER:
            docker_sandbox = DockerSandbox(
                allowed_roots=self.ctx.sandbox.allowed_roots or [Path(roots[0])],
                timeout=30.0,
            )
        shell = ShellToolAdapter(
            cwd=default_cwd, timeout=30.0, docker_sandbox=docker_sandbox
        )
        self.tools.register(
            "bash_run_command",
            "Execute a bash shell command in the project repository. Returns stdout and stderr.",
            shell,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute (e.g. pytest tests/)",
                    }
                },
                "required": ["command"],
            },
            plugin_id="core",
        )

        process = ProcessToolAdapter(
            supervisor=self.process_supervisor,
            default_cwd=default_cwd,
        )
        self.tools.register(
            "process_start",
            "Start a long-running background shell process.",
            process,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to start.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory for the process.",
                    },
                    "scope_key": {
                        "type": "string",
                        "description": "Scope key for process isolation.",
                    },
                },
                "required": ["command"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "process_poll",
            "Poll the status and recent output of a managed background process.",
            process,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "process_id": {
                        "type": "string",
                        "description": "The process ID returned by process_start.",
                    },
                    "scope_key": {
                        "type": "string",
                        "description": "Scope key for process isolation.",
                    },
                },
                "required": ["process_id"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "process_write",
            "Write input text to the stdin of a managed background process.",
            process,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "process_id": {
                        "type": "string",
                        "description": "The process ID returned by process_start.",
                    },
                    "input": {
                        "type": "string",
                        "description": "Text to write to the process stdin.",
                    },
                    "scope_key": {
                        "type": "string",
                        "description": "Scope key for process isolation.",
                    },
                },
                "required": ["process_id", "input"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "process_send_signal",
            "Send a POSIX signal to a managed background process (default SIGTERM).",
            process,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "process_id": {
                        "type": "string",
                        "description": "The process ID returned by process_start.",
                    },
                    "signal": {
                        "type": "integer",
                        "description": "Signal number to send (default 15).",
                    },
                    "scope_key": {
                        "type": "string",
                        "description": "Scope key for process isolation.",
                    },
                },
                "required": ["process_id"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "process_kill",
            "Forcefully kill a managed background process.",
            process,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "process_id": {
                        "type": "string",
                        "description": "The process ID returned by process_start.",
                    },
                    "scope_key": {
                        "type": "string",
                        "description": "Scope key for process isolation.",
                    },
                },
                "required": ["process_id"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "process_clear",
            "Kill and remove all managed background processes in a scope.",
            process,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "scope_key": {
                        "type": "string",
                        "description": "Scope key for process isolation.",
                    },
                },
                "required": [],
            },
            plugin_id="core",
        )
        self.tools.register(
            "process_remove",
            "Remove a managed background process from tracking (kills if running).",
            process,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "process_id": {
                        "type": "string",
                        "description": "The process ID returned by process_start.",
                    },
                    "scope_key": {
                        "type": "string",
                        "description": "Scope key for process isolation.",
                    },
                },
                "required": ["process_id"],
            },
            plugin_id="core",
        )

        pty = PtyToolAdapter(
            supervisor=self.pty_supervisor,
            default_cwd=default_cwd,
        )
        self.tools.register(
            "pty_start",
            "Start an interactive PTY-backed terminal session.",
            pty,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to run in the PTY session."},
                    "cwd": {"type": "string", "description": "Working directory for the PTY session."},
                    "rows": {"type": "integer", "description": "Terminal rows."},
                    "cols": {"type": "integer", "description": "Terminal columns."},
                    "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                },
                "required": ["command"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "pty_poll",
            "Poll the current PTY session state and read available terminal output.",
            pty,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "PTY session id from pty_start."},
                    "max_bytes": {"type": "integer", "description": "Maximum bytes of output to read."},
                    "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                },
                "required": ["session_id"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "pty_observe",
            "Observe a PTY session with adaptive backoff until it goes quiet or exits.",
            pty,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "PTY session id from pty_start."},
                    "idle_seconds": {
                        "type": "number",
                        "description": "Return after this much PTY silence once output has started.",
                    },
                    "max_wait_seconds": {
                        "type": "number",
                        "description": "Maximum total time to observe before timing out.",
                    },
                    "max_bytes_per_poll": {
                        "type": "integer",
                        "description": "Maximum bytes to read per internal poll step.",
                    },
                    "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                },
                "required": ["session_id"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "pty_interact",
            "Start or continue a PTY session, optionally send input, then observe until the terminal goes quiet. Prefer this for terminal UIs like claude, codex, vim, shells, and editors.",
            pty,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Existing PTY session id to continue. Omit to start a new session.",
                    },
                    "command": {
                        "type": "string",
                        "description": "Command to start when opening a new PTY session.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory for a new PTY session.",
                    },
                    "rows": {"type": "integer", "description": "Terminal rows for a new PTY session."},
                    "cols": {"type": "integer", "description": "Terminal columns for a new PTY session."},
                    "input": {
                        "type": "string",
                        "description": "Optional text or control sequence to write before observing.",
                    },
                    "idle_seconds": {
                        "type": "number",
                        "description": "Return after this much PTY silence once output has started.",
                    },
                    "max_wait_seconds": {
                        "type": "number",
                        "description": "Maximum total time to observe before timing out.",
                    },
                    "max_bytes_per_poll": {
                        "type": "integer",
                        "description": "Maximum bytes to read per internal poll step.",
                    },
                    "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                },
                "required": [],
            },
            plugin_id="core",
        )
        self.tools.register(
            "pty_write",
            "Write text or control sequences to an interactive PTY session.",
            pty,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "PTY session id from pty_start."},
                    "input": {"type": "string", "description": "Text or control sequence to write."},
                    "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                },
                "required": ["session_id", "input"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "pty_resize",
            "Resize an interactive PTY session.",
            pty,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "PTY session id from pty_start."},
                    "rows": {"type": "integer", "description": "Terminal rows."},
                    "cols": {"type": "integer", "description": "Terminal columns."},
                    "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                },
                "required": ["session_id", "rows", "cols"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "pty_kill",
            "Kill the process attached to a PTY session.",
            pty,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "PTY session id from pty_start."},
                    "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                },
                "required": ["session_id"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "pty_remove",
            "Remove a PTY session from tracking and kill it if still running.",
            pty,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "PTY session id from pty_start."},
                    "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                },
                "required": ["session_id"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "pty_clear",
            "Kill and remove all PTY sessions in a scope.",
            pty,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                },
                "required": [],
            },
            plugin_id="core",
        )

        edit = EditToolAdapter(allowed_roots=roots)
        self.tools.register(
            "edit_file",
            "Precisely edit a file by replacing old_string with new_string. Requires exact match.",
            edit,
            ActionRiskTier.WORKSPACE_WRITE,
            {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to edit.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The new string to insert.",
                    },
                    "occurrence_index": {
                        "type": "integer",
                        "description": "Which occurrence to replace (0-based). Required if multiple matches.",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
            plugin_id="core",
        )

        search = SearchToolAdapter(allowed_roots=roots)
        self.tools.register(
            "grep_search",
            "Search files for a regex pattern.",
            search,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search (default: workspace root).",
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches"],
                        "description": "Return matching lines or just file paths.",
                    },
                },
                "required": ["pattern"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "glob_search",
            "Find files matching a glob pattern.",
            search,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. '**/*.py').",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search (default: workspace root).",
                    },
                },
                "required": ["pattern"],
            },
            plugin_id="core",
        )

        runtime_state = RuntimeStateToolAdapter(runtime=self)
        self.tools.register(
            "runtime_status",
            "Return workspace, sandbox, and execution control-plane state.",
            runtime_state,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
            plugin_id="core",
        )
        workflow_state = WorkflowStateToolAdapter(runtime=self)
        self.tools.register(
            "workflow_status",
            "Return higher-level workflow state including goals, commitments, plans, work objects, and receipts.",
            workflow_state,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum items to include per section.",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Optional project id to focus the workflow summary.",
                    },
                },
                "required": [],
            },
            plugin_id="core",
        )

        workflow = WorkflowToolAdapter(runtime=self)
        self.tools.register(
            "workflow_create_commitment",
            "Create a durable goal or commitment to track ongoing work.",
            workflow,
            ActionRiskTier.WORKSPACE_WRITE,
            {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "What this commitment is about.",
                    },
                    "priority": {
                        "type": "number",
                        "description": "Priority from 1.0 (low) to 10.0 (critical). Default 5.0.",
                    },
                    "deadline": {
                        "type": "string",
                        "description": "Optional ISO-8601 deadline.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags for categorization.",
                    },
                },
                "required": ["content"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "workflow_update_commitment",
            "Update a commitment's status: completed, abandoned, blocked, or active.",
            workflow,
            ActionRiskTier.WORKSPACE_WRITE,
            {
                "type": "object",
                "properties": {
                    "commitment_id": {
                        "type": "string",
                        "description": "The commitment ID to update.",
                    },
                    "status": {
                        "type": "string",
                        "description": "New status: completed, abandoned, blocked, or active.",
                    },
                },
                "required": ["commitment_id", "status"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "workflow_list_commitments",
            "List commitments filtered by status.",
            workflow,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by status: active, completed, abandoned, blocked. Default active.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum items to return. Default 20.",
                    },
                },
                "required": [],
            },
            plugin_id="core",
        )
        self.tools.register(
            "workflow_create_schedule",
            "Create a scheduled task or calendar event. Use ISO-8601 start_at; supports none, interval_hours, daily, weekly, and weekdays recurrence.",
            workflow,
            ActionRiskTier.WORKSPACE_WRITE,
            {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["task", "event"]},
                    "action": {"type": "string", "enum": ["submit_baa", "reminder_only"]},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "objective": {"type": "string"},
                    "start_at": {"type": "string"},
                    "end_at": {"type": "string"},
                    "timezone": {"type": "string"},
                    "recurrence": {"type": "string", "enum": ["none", "interval_hours", "daily", "weekly", "weekdays"]},
                    "interval_hours": {"type": "number"},
                    "weekdays": {"type": "array", "items": {"type": "integer"}},
                    "max_occurrences": {"type": "integer"},
                    "priority": {"type": "number"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "commitment_id": {"type": "string"},
                    "plan_id": {"type": "string"},
                },
                "required": ["title", "start_at"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "workflow_update_schedule",
            "Update a schedule's status, title, description, objective, priority, or tags.",
            workflow,
            ActionRiskTier.WORKSPACE_WRITE,
            {
                "type": "object",
                "properties": {
                    "schedule_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["active", "paused", "completed", "cancelled"]},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "objective": {"type": "string"},
                    "priority": {"type": "number"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["schedule_id"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "workflow_list_schedules",
            "List scheduled tasks and events.",
            workflow,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "kind": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
            plugin_id="core",
        )
        self.tools.register(
            "workflow_create_writing_task",
            "Set up a writing task with commitment tracking, output path, and optional outline scaffold.",
            workflow,
            ActionRiskTier.WORKSPACE_WRITE,
            {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the writing piece.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of the writing task.",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional file path for the output. Auto-generated if omitted.",
                    },
                    "outline": {
                        "description": "Optional outline: a list of section headings or a text outline.",
                    },
                    "priority": {
                        "type": "number",
                        "description": "Priority from 1.0 to 10.0. Default 6.0.",
                    },
                },
                "required": ["title"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "workflow_create_plan",
            "Create a structured plan for a project or task.",
            workflow,
            ActionRiskTier.WORKSPACE_WRITE,
            {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The plan content (markdown or plain text).",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Optional project or commitment ID to link this plan to.",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Optional task ID to link this plan to.",
                    },
                },
                "required": ["content"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "workflow_update_plan",
            "Update a plan's content.",
            workflow,
            ActionRiskTier.WORKSPACE_WRITE,
            {
                "type": "object",
                "properties": {
                    "plan_id": {
                        "type": "string",
                        "description": "The plan ID to update.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Updated plan content.",
                    },
                },
                "required": ["plan_id", "content"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "workflow_repo_triage",
            "Quick repo triage: git status, recent commits, work items, commitments, and plans summary.",
            workflow,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
            plugin_id="core",
        )
        self.tools.register(
            "workflow_supervise_session",
            "Launch or resume a PTY session (claude, kilocode, codex, vim, etc.), send a task with an Enter key, and supervise the cleaned output across multiple observation rounds. Returns a screen-state summary plus a supervision advisory so you can tell whether to keep observing, send follow-up input, or resolve an auth gate. Prefer this over raw PTY choreography for external TUI work.",
            workflow,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to start (e.g. 'claude', 'codex'). Required if no session_id.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Resume an existing PTY session instead of starting new.",
                    },
                    "task": {
                        "type": "string",
                        "description": "Text to send as input to the session.",
                    },
                    "verification_path": {
                        "type": "string",
                        "description": "Optional file path to verify after each supervision round. Useful for bounded artifact-producing tasks.",
                    },
                    "scope_key": {
                        "type": "string",
                        "description": "Scope key for session isolation. Default: workflow-supervision.",
                    },
                    "max_wait_seconds": {
                        "type": "number",
                        "description": "Maximum seconds for the initial submit/observe round. Default 15.",
                    },
                    "startup_wait_seconds": {
                        "type": "number",
                        "description": "When starting a new TUI process, maximum seconds to wait for the UI to reach a stable ready state before task submission. Default min(max_wait_seconds, 8).",
                    },
                    "idle_seconds": {
                        "type": "number",
                        "description": "Seconds of silence before considering output complete. Default 1.0.",
                    },
                    "continue_wait_seconds": {
                        "type": "number",
                        "description": "Maximum seconds for later observation rounds after the initial submit. Defaults to max_wait_seconds.",
                    },
                    "max_rounds": {
                        "type": "integer",
                        "description": "Total supervision rounds including the initial submit round. Default 3.",
                    },
                },
                "required": [],
            },
            plugin_id="core",
        )

        web = WebToolAdapter()
        self.tools.register(
            "web_fetch",
            "Fetch a URL and return extracted text.",
            web,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch.",
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum characters to return.",
                    },
                },
                "required": ["url"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "web_search",
            "Search the live web for real-time information, breaking news, and recent developments. Returns result links and titles.",
            web,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                },
                "required": ["query"],
            },
            plugin_id="core",
        )

        browser = BrowserToolAdapter(supervisor=self.browser_supervisor)
        self.tools.register(
            "browser_start",
            "Start a Playwright-backed browser session.",
            browser,
            ActionRiskTier.NETWORK,
            {
                "type": "object",
                "properties": {
                    "headless": {
                        "type": "boolean",
                        "description": "Run browser headlessly (default true).",
                    },
                    "viewport_width": {
                        "type": "integer",
                        "description": "Browser viewport width.",
                    },
                    "viewport_height": {
                        "type": "integer",
                        "description": "Browser viewport height.",
                    },
                    "scope_key": {
                        "type": "string",
                        "description": "Scope key for browser session isolation.",
                    },
                },
                "required": [],
            },
            plugin_id="core",
        )
        self.tools.register(
            "browser_navigate",
            "Navigate a browser session to a URL.",
            browser,
            ActionRiskTier.NETWORK,
            {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session id from browser_start.",
                    },
                    "url": {"type": "string", "description": "URL to navigate to."},
                    "wait_until": {
                        "type": "string",
                        "description": "Playwright wait state: load, domcontentloaded, networkidle, or commit.",
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Navigation timeout in milliseconds.",
                    },
                    "scope_key": {
                        "type": "string",
                        "description": "Scope key for browser session isolation.",
                    },
                },
                "required": ["session_id", "url"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "browser_click",
            "Click an element inside the active browser page.",
            browser,
            ActionRiskTier.EXTERNAL_WRITE,
            {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session id from browser_start.",
                    },
                    "selector": {
                        "type": "string",
                        "description": "Selector to click.",
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Click timeout in milliseconds.",
                    },
                    "scope_key": {
                        "type": "string",
                        "description": "Scope key for browser session isolation.",
                    },
                },
                "required": ["session_id", "selector"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "browser_type",
            "Type text into an input or editable element.",
            browser,
            ActionRiskTier.EXTERNAL_WRITE,
            {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session id from browser_start.",
                    },
                    "selector": {
                        "type": "string",
                        "description": "Selector to type into.",
                    },
                    "text": {"type": "string", "description": "Text to type."},
                    "clear": {
                        "type": "boolean",
                        "description": "Clear the field before typing.",
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Typing timeout in milliseconds.",
                    },
                    "scope_key": {
                        "type": "string",
                        "description": "Scope key for browser session isolation.",
                    },
                },
                "required": ["session_id", "selector", "text"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "browser_press",
            "Press a keyboard key in the active browser page.",
            browser,
            ActionRiskTier.EXTERNAL_WRITE,
            {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session id from browser_start.",
                    },
                    "key": {
                        "type": "string",
                        "description": "Keyboard key to press, e.g. Enter.",
                    },
                    "scope_key": {
                        "type": "string",
                        "description": "Scope key for browser session isolation.",
                    },
                },
                "required": ["session_id", "key"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "browser_wait",
            "Wait for page readiness or a selector to appear.",
            browser,
            ActionRiskTier.NETWORK,
            {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session id from browser_start.",
                    },
                    "selector": {
                        "type": "string",
                        "description": "Optional selector to wait for.",
                    },
                    "load_state": {
                        "type": "string",
                        "description": "Playwright load state when selector is omitted.",
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Wait timeout in milliseconds.",
                    },
                    "scope_key": {
                        "type": "string",
                        "description": "Scope key for browser session isolation.",
                    },
                },
                "required": ["session_id"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "browser_snapshot",
            "Capture a text-and-link snapshot of the current browser page.",
            browser,
            ActionRiskTier.NETWORK,
            {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session id from browser_start.",
                    },
                    "max_text_length": {
                        "type": "integer",
                        "description": "Maximum text length to return.",
                    },
                    "max_links": {
                        "type": "integer",
                        "description": "Maximum number of links to include.",
                    },
                    "capture_screenshot": {
                        "type": "boolean",
                        "description": "Capture a screenshot to a temp file path.",
                    },
                    "full_page": {
                        "type": "boolean",
                        "description": "Capture the full page when taking a screenshot.",
                    },
                    "scope_key": {
                        "type": "string",
                        "description": "Scope key for browser session isolation.",
                    },
                },
                "required": ["session_id"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "browser_close",
            "Close and remove a browser session.",
            browser,
            ActionRiskTier.NETWORK,
            {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Browser session id from browser_start.",
                    },
                    "scope_key": {
                        "type": "string",
                        "description": "Scope key for browser session isolation.",
                    },
                },
                "required": ["session_id"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "browser_clear",
            "Close and remove all browser sessions in a scope.",
            browser,
            ActionRiskTier.NETWORK,
            {
                "type": "object",
                "properties": {
                    "scope_key": {
                        "type": "string",
                        "description": "Scope key for browser session isolation.",
                    },
                },
                "required": [],
            },
            plugin_id="core",
        )

        repl = ReplToolAdapter()
        self.tools.register(
            "python_repl",
            "Execute Python code in a persistent REPL session.",
            repl,
            ActionRiskTier.SHELL_LOCAL,
            {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    },
                    "research_session_id": {
                        "type": "string",
                        "description": "Session ID for persistent state across calls.",
                    },
                },
                "required": ["code"],
            },
            plugin_id="core",
        )

        lsp = LspToolAdapter()
        self.tools.register(
            "lsp_goto_definition",
            "Go to the definition of a symbol in a file.",
            lsp,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "line": {"type": "integer"},
                    "character": {"type": "integer"},
                },
                "required": ["file_path", "line", "character"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "lsp_find_references",
            "Find all references to a symbol in a file.",
            lsp,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "line": {"type": "integer"},
                    "character": {"type": "integer"},
                },
                "required": ["file_path", "line", "character"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "lsp_hover",
            "Get type/documentation info for a symbol in a file.",
            lsp,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "line": {"type": "integer"},
                    "character": {"type": "integer"},
                },
                "required": ["file_path", "line", "character"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "lsp_document_symbols",
            "List all symbols (functions, classes, variables) in a file.",
            lsp,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
            plugin_id="core",
        )
        self.tools.register(
            "lsp_diagnostics",
            "Get syntax errors and diagnostics for a file.",
            lsp,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
            plugin_id="core",
        )

        interactive = InteractiveToolAdapter()
        self.tools.register(
            "ask_user_question",
            "Ask the user a clarifying question and pause the loop.",
            interactive,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to present to the user.",
                    },
                },
                "required": ["question"],
            },
            plugin_id="core",
        )

        plan_store = getattr(self.ctx, "plan_store", None)
        plan = PlanToolAdapter(store=plan_store)
        self.tools.register(
            "enter_plan_mode",
            "Enter a constrained planning phase where only read tools and plan writes are allowed.",
            plan,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "plan_id": {
                        "type": "string",
                        "description": "Optional identifier for the plan.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Initial plan content.",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Optional project association.",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Optional task association.",
                    },
                },
                "required": [],
            },
            plugin_id="core",
        )
        self.tools.register(
            "exit_plan_mode",
            "Exit planning phase and resume normal tool access.",
            plan,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "plan_id": {
                        "type": "string",
                        "description": "Optional identifier for the plan.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Final plan content to save.",
                    },
                },
                "required": [],
            },
            plugin_id="core",
        )

        # MCP on-demand tools
        mcp_registry = getattr(self.ctx, "mcp_registry", None)
        if mcp_registry is not None and self.ctx.config.mcp_auto_register:
            try:
                import asyncio
                tools = asyncio.run_coroutine_threadsafe(
                    self._discover_and_register_mcp_tools(), asyncio.get_running_loop()
                ).result()
                self._trace("mcp_auto_registered", {"tool_count": len(tools)})
            except Exception as exc:
                self._trace("mcp_auto_register_failed", {"error": str(exc)})

        self.tools.register(
            "mcp_list_servers",
            "List configured MCP servers and their initialization status.",
            self._make_mcp_list_servers_adapter(),
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
            plugin_id="core",
        )
        self.tools.register(
            "mcp_register_server_tools",
            "Initialize a specific MCP server and register its tools for the current session.",
            self._make_mcp_register_adapter(),
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "server_name": {
                        "type": "string",
                        "description": "Name of the MCP server to initialize.",
                    },
                },
                "required": ["server_name"],
            },
            plugin_id="core",
        )

        agent = AgentToolAdapter(runtime=self)
        self.tools.register(
            "agent",
            "Spawn a specialized subagent with a separate tool loop to work on a task.",
            agent,
            ActionRiskTier.READONLY,
            {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Short description of the subagent task.",
                    },
                    "agent_type": {
                        "type": "string",
                        "description": "Type of subagent (e.g. explore, plan, general-purpose).",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Prompt/instructions for the subagent.",
                    },
                },
                "required": ["prompt"],
            },
            plugin_id="core",
        )

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

    async def converse(self, user_input: str, session_id: Optional[str] = None) -> str:
        """Process a user turn, update state, and return a response."""
        self._set_activity("conversing")
        sid = session_id or self.ctx.config.session_id or "default"

        # Conversational refusal check before any state change
        from opencas.refusal.models import ConversationalRequest
        conv_request = ConversationalRequest(text=user_input, session_id=sid)
        refusal = self.refusal_gate.evaluate(conv_request)
        if refusal.refused:
            # Trace escalation and record to ledger if available
            self._trace(
                "converse_refusal",
                {
                    "session_id": sid,
                    "category": refusal.category.value if refusal.category else None,
                    "reasoning": refusal.reasoning,
                },
            )
            if self.approval.ledger is not None:
                try:
                    from opencas.autonomy.models import ActionRequest, ActionRiskTier, ApprovalDecision, ApprovalLevel
                    request = ActionRequest(
                        tier=ActionRiskTier.READONLY,
                        description=user_input,
                        tool_name="conversation",
                    )
                    decision = ApprovalDecision(
                        level=ApprovalLevel.MUST_ESCALATE,
                        action_id=request.action_id,
                        confidence=1.0,
                        reasoning=refusal.reasoning,
                        score=1.0,
                    )
                    await self.approval.ledger.record(decision, request, 1.0, None)
                except Exception:
                    pass
            response_text = refusal.suggested_response or "I'm not able to respond to that."
            await self.ctx.context_store.append(sid, MessageRole.ASSISTANT, response_text)
            return response_text

        await self.ctx.somatic.emit_appraisal_event(
            AppraisalEventType.USER_INPUT_RECEIVED,
            source_text=user_input,
            trigger_event_id=sid,
        )
        await self._record_episode(user_input, EpisodeKind.TURN, session_id=sid)
        await self.ctx.context_store.append(sid, MessageRole.USER, user_input)

        manifest = await self.builder.build(user_input, session_id=sid)
        messages = manifest.to_message_list()
        had_system = len(messages) > 0 and messages[0].get("role") == "system"
        initial_message_count = len(messages)
        loop_result: Optional[Any] = None

        try:
            payload = {"temperature": self.modulators.to_temperature()}
            tool_ctx = await self._build_tool_use_context(session_id=sid)
            loop_result = await self.tool_loop.run(
                objective=user_input,
                messages=messages,
                ctx=tool_ctx,
                payload=payload,
                on_focus_enter=self.scheduler.enter_focus_mode if self.scheduler else None,
                on_focus_exit=self.scheduler.exit_focus_mode if self.scheduler else None,
            )
            content = loop_result.final_output
        except UserInputRequired as exc:
            content = exc.question
        except Exception as exc:
            content = f"[Error generating response: {exc}]"

        # Persist intermediate tool-turn messages for compaction fidelity
        if loop_result is not None and hasattr(loop_result, "messages") and loop_result.messages:
            # If the tool loop added a system prompt, offset by 1
            has_system = len(loop_result.messages) > 0 and loop_result.messages[0].get("role") == "system"
            offset = initial_message_count
            if has_system and not had_system:
                offset += 1
            
            # We only want to persist NEW messages added during this turn
            new_messages = loop_result.messages[offset:]
            for msg in new_messages:
                role = msg.get("role")
                if role == "assistant" and msg.get("tool_calls"):
                    await self.ctx.context_store.append(
                        sid,
                        MessageRole.ASSISTANT,
                        msg.get("content", ""),
                        meta={"tool_calls": msg["tool_calls"]},
                    )
                elif role == "tool":
                    await self.ctx.context_store.append(
                        sid,
                        MessageRole.TOOL,
                        msg.get("content", ""),
                        meta={
                            "tool_call_id": msg.get("tool_call_id", ""),
                            "name": msg.get("name", ""),
                        },
                    )

        await self.ctx.context_store.append(sid, MessageRole.ASSISTANT, content)
        await self._record_episode(content, EpisodeKind.TURN, session_id=sid)

        # Extract goal/intention directives from user input
        goals, intention, drops = self._extract_goal_directives(user_input)
        for g in goals:
            self.executive.add_goal(g)
        if intention:
            self.executive.set_intention(intention)
        for d in drops:
            # Heuristic: if a drop phrase matches a goal substring, remove it
            for goal in list(self.executive.active_goals):
                if any(token in goal.lower() for token in d.split() if len(token) > 3):
                    self.executive.remove_goal(goal)
                    break
        if goals or intention or drops:
            self._sync_executive_snapshot()

        # Trigger compaction if context budget is exceeded
        if manifest.token_estimate and manifest.token_estimate > 4000:
            try:
                await self.maybe_compact_session(sid)
            except Exception as exc:
                self._trace("compaction_error", {"error": str(exc)})

        # Lightweight ToM: record a belief from user input and check consistency
        await self.tom.record_belief(
            BeliefSubject.USER,
            f"said: {user_input[:120]}",
            confidence=0.6,
        )
        metacognition = self.tom.check_consistency()
        if metacognition.contradictions:
            self._trace("metacognitive_alert", {"contradictions": metacognition.contradictions})

        # Relational: record the interaction
        if hasattr(self.ctx, "relational") and self.ctx.relational:
            interaction_ep = Episode(
                kind=EpisodeKind.TURN,
                session_id=sid,
                content=f"User: {user_input[:200]}\nAssistant: {content[:200]}",
                somatic_tag=self.ctx.somatic.state.somatic_tag,
            )
            await self.ctx.relational.record_interaction(
                episode=interaction_ep,
                outcome="neutral",
            )

        self._trace(
            "converse",
            {"session_id": sid, "input_len": len(user_input), "token_estimate": manifest.token_estimate},
        )
        self.boredom.record_activity()
        return content

    async def run_daydream(self) -> Dict[str, Any]:
        """Generate daydreams when idle or tense."""
        self._set_activity("daydreaming")
        try:
            return await self._run_daydream_inner()
        finally:
            self._set_activity("idle")

    async def _run_daydream_inner(self) -> Dict[str, Any]:
        """Inner implementation of run_daydream (wrapped for activity tracking)."""
        daydream_work_objects: List[WorkObject] = []
        reflections: List[DaydreamReflection] = []
        somatic = self.ctx.somatic.state
        now = datetime.now(timezone.utc)
        cooldown_ok = (
            self._last_daydream_time is None
            or (now - self._last_daydream_time).total_seconds() > 300
        )
        somatic_readiness = (somatic.energy + somatic.focus) / 2.0
        if not (self.boredom.should_daydream(somatic_readiness=somatic_readiness) and cooldown_ok):
            return {
                "daydreams": 0,
                "reflections": 0,
                "keepers": 0,
                "daydream_memories_created": 0,
                "daydream_work_objects": daydream_work_objects,
                "reflections_list": reflections,
            }

        try:
            memories_created = 0
            work_objects, reflection_drafts = await self.daydream.generate(
                goals=self.executive.active_goals,
                tension=somatic.tension,
            )
            await self.ctx.somatic.emit_appraisal_event(
                AppraisalEventType.DAYDREAM_GENERATED,
                source_text="daydream generated",
                trigger_event_id=str(now.timestamp()),
                meta={"reflection_count": len(reflection_drafts), "work_count": len(work_objects)},
            )
            recent: List[str] = []
            if getattr(self.ctx, "daydream_store", None):
                recent = [
                    r.spark_content
                    for r in await self.ctx.daydream_store.list_recent(limit=10)
                ]
            for reflection in reflection_drafts:
                self.reflection_evaluator.score_alignment(
                    reflection, self.ctx.identity
                )
                self.reflection_evaluator.score_novelty(reflection, recent)
                self.reflection_evaluator.decide_keeper(reflection)

                # Detect conflicts and register with somatic context
                conflicts = self.reflection_evaluator.detect_conflicts(reflection)
                stored_conflicts: List[Any] = []
                if self.conflict_registry is not None:
                    stored_conflicts = await self.conflict_registry.active(limit=20)
                newly_detected_conflicts: List[Any] = []
                if conflicts and self.conflict_registry is not None:
                    from opencas.daydream.models import ConflictRecord
                    snapshot = self.ctx.somatic.state
                    for kind, description in conflicts:
                        record = ConflictRecord(
                            kind=kind,
                            description=description,
                            source_daydream_id=str(reflection.reflection_id),
                        )
                        stored = await self.conflict_registry.register(
                            record, somatic_context=snapshot
                        )
                        stored_conflicts.append(stored)
                        newly_detected_conflicts.append(stored)

                # Resolve reflection against conflicts and somatic state
                resolution = self.reflection_resolver.resolve(
                    reflection, stored_conflicts, self.ctx.somatic.state
                )

                allow_promotion = reflection.keeper and resolution.strategy in (
                    "accept",
                    "reframe",
                )
                original_spark_content = reflection.spark_content

                if resolution.strategy == "escalate":
                    self._trace(
                        "reflection_escalate",
                        {
                            "reflection_id": str(reflection.reflection_id),
                            "reason": resolution.reason,
                            "conflict_id": resolution.conflict_id,
                        },
                    )
                elif resolution.strategy == "reframe" and resolution.mirror:
                    reflection.spark_content = (
                        f"{resolution.mirror.affirmation}\n\n{reflection.spark_content}"
                    )

                if allow_promotion:
                    for wo in work_objects:
                        if wo.content == original_spark_content:
                            # Set intensity for persistent intent bypass
                            if wo.promotion_score == 0.0:
                                wo.promotion_score = round(reflection.alignment_score, 3)
                            wo.meta.setdefault("intensity", reflection.alignment_score)

                            # Route spark through SparkRouter
                            boredom = self.boredom.compute_boredom(now)
                            rung = self.spark_router.route(wo, None, boredom)

                            if rung == SparkRung.REJECT:
                                self._trace(
                                    "spark_rejected",
                                    {"work_id": str(wo.work_id), "reason": "router rejected"},
                                )
                                break

                            # Portfolio clustering
                            if self.portfolio_store:
                                fkey = build_fascination_key(
                                    wo.content, wo.meta.get("tags")
                                )
                                cluster = await self.portfolio_store.get_by_key(fkey)
                                if cluster is None and wo.promotion_score >= 0.3:
                                    cluster = PortfolioCluster(fascination_key=fkey)
                                    await self.portfolio_store.save(cluster)
                                if cluster is not None:
                                    wo.portfolio_id = str(cluster.cluster_id)
                                    inc = {"sparks": 1}
                                    if rung in (SparkRung.MICRO_TASK, SparkRung.FULL_TASK):
                                        inc["initiatives"] = 1
                                    await self.portfolio_store.increment_counts(fkey, **inc)

                            # Set stage and create commitment for full_task
                            if rung == SparkRung.NOTE:
                                wo.stage = WorkStage.NOTE
                            elif rung == SparkRung.MICRO_TASK:
                                wo.stage = WorkStage.MICRO_TASK
                            elif rung == SparkRung.FULL_TASK:
                                wo.stage = WorkStage.PROJECT
                                if self.commitment_store:
                                    commitment = Commitment(
                                        content=wo.content,
                                        priority=round(5.0 + reflection.alignment_score * 5.0, 1),
                                    )
                                    await self.commitment_store.save(commitment)
                                    wo.commitment_id = str(commitment.commitment_id)

                            self.creative.add(wo)
                            daydream_work_objects.append(wo)
                            break

                if getattr(self.ctx, "daydream_store", None):
                    await self.ctx.daydream_store.save_reflection(reflection)
                    recent.append(reflection.spark_content)
                if reflection.keeper and self.memory:
                    content = reflection.synthesis or reflection.spark_content
                    try:
                        embed_record = await self.ctx.embeddings.embed(
                            content,
                            meta={"origin": "daydream_keeper"},
                            task_type="daydream_memory",
                        )
                        await self.memory.save_memory(
                            Memory(
                                content=content,
                                tags=["daydream", "keeper"],
                                source_episode_ids=[],
                                embedding_id=embed_record.source_hash,
                                salience=round(reflection.alignment_score * 10, 3),
                            )
                        )
                        memories_created += 1
                    except Exception:
                        pass

                # Emit appraisal events
                if newly_detected_conflicts:
                    await self.ctx.somatic.emit_appraisal_event(
                        AppraisalEventType.CONFLICT_DETECTED,
                        source_text=reflection.spark_content,
                        trigger_event_id=str(reflection.reflection_id),
                        meta={"conflict_kinds": [c.kind for c in newly_detected_conflicts]},
                    )
                await self.ctx.somatic.emit_appraisal_event(
                    AppraisalEventType.REFLECTION_RESOLVED,
                    source_text=resolution.reason,
                    trigger_event_id=str(reflection.reflection_id),
                    meta={"strategy": resolution.strategy},
                )
                if resolution.strategy == "reframe" and resolution.mirror:
                    await self.ctx.somatic.emit_appraisal_event(
                        AppraisalEventType.SELF_COMPASSION_OFFERED,
                        source_text=resolution.mirror.affirmation,
                        trigger_event_id=str(reflection.reflection_id),
                        meta={"suggested_strategy": resolution.mirror.suggested_strategy},
                    )

                if reflection.keeper and self.ctx.identity:
                    synth = reflection.synthesis.lower()
                    for prefix in ("i want to", "i should"):
                        idx = synth.find(prefix)
                        if idx != -1:
                            rest = synth[idx + len(prefix) :]
                            end = rest.find(".")
                            if end == -1:
                                end = rest.find("\n")
                            phrase = rest[:end].strip(" ,;:-")
                            if phrase:
                                self.ctx.identity.add_inferred_goal(phrase)
            self._last_daydream_time = now
            self.boredom.record_reset()
            reflections = reflection_drafts
        except Exception as exc:
            self._trace("daydream_error", {"error": str(exc)})

        keepers = sum(1 for r in reflections if r.keeper)
        return {
            "daydreams": len(daydream_work_objects),
            "reflections": len(reflections),
            "keepers": keepers,
            "daydream_memories_created": memories_created,
            "daydream_work_objects": daydream_work_objects,
            "reflections_list": reflections,
        }

    async def run_cycle(self) -> Dict[str, Any]:
        """Run one creative/execution cycle."""
        self._set_activity("cycling")
        await self.executive.restore_queue()
        result = self.creative.run_cycle()

        # Enqueue newly promoted work that reached at least MICRO_TASK
        promoted_tasks = 0
        for stage in [WorkStage.MICRO_TASK, WorkStage.PROJECT_SEED, WorkStage.PROJECT]:
            for work in self.creative.list_by_stage(stage):
                if stage == WorkStage.PROJECT:
                    await self.orchestrator.decompose(work)
                    promoted_tasks += 1
                else:
                    should_enqueue = True
                    if work.commitment_id and self.commitment_store:
                        from opencas.autonomy.commitment import CommitmentStatus
                        commitment = await self.commitment_store.get(work.commitment_id)
                        if commitment and commitment.status in (
                            CommitmentStatus.BLOCKED,
                            CommitmentStatus.ABANDONED,
                        ):
                            should_enqueue = False
                    if should_enqueue and self.executive.enqueue(work):
                        promoted_tasks += 1

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
        intervention_decision = None
        try:
            commitments = []
            if self.commitment_store:
                commitments = await self.commitment_store.list_active(limit=100)
            work_objects = []
            if self.ctx.work_store:
                work_objects = await self.ctx.work_store.list_ready(limit=100)

            portfolio_boosts: Dict[str, Any] = {}
            if self.portfolio_store:
                clusters = await self.portfolio_store.list_all(limit=1000)
                from opencas.autonomy.workspace import PortfolioBoost
                for cluster in clusters:
                    boost = min(0.15, cluster.spark_count * 0.02)
                    portfolio_boosts[str(cluster.cluster_id)] = PortfolioBoost(
                        portfolio_id=str(cluster.cluster_id),
                        spark_count=cluster.spark_count,
                        boost=boost,
                    )

            workspace = ExecutiveWorkspace.rebuild(
                commitments=commitments,
                work_objects=work_objects,
                portfolio_boosts=portfolio_boosts,
            )

            live_orders: List[Dict[str, Any]] = []
            if self.ctx.tasks:
                pending_tasks = await self.ctx.tasks.list_pending(limit=100)
                live_orders = [
                    {
                        "task_id": str(t.task_id),
                        "stage": t.stage.value,
                        "objective": t.objective,
                    }
                    for t in pending_tasks
                ]

            intervention_decision = InterventionPolicy.evaluate(
                workspace=workspace,
                baa_queue_depth=self.baa.queue_size,
                held_count=self.baa.held_size,
                somatic_recommends_pause=self.ctx.somatic.state.fatigue > 0.7,
                live_work_orders=live_orders,
            )

            if intervention_decision.kind == InterventionKind.LAUNCH_BACKGROUND:
                if workspace.focus and workspace.focus.execution_mode == ExecutionMode.BACKGROUND_AGENT:
                    from opencas.execution.models import RepairTask
                    repair_task = RepairTask(
                        objective=workspace.focus.content,
                        project_id=workspace.focus.project_id,
                        commitment_id=workspace.focus.commitment_id,
                        meta={"source": "intervention_launch_background"},
                    )
                    await self.baa.submit(repair_task)
            elif intervention_decision.kind == InterventionKind.RETIRE_OR_DEFER_FOCUS:
                if workspace.focus:
                    focus_id = str(workspace.focus.item_id)
                    removed = self.executive.remove_work(focus_id)
                    if not removed and self.ctx.work_store:
                        await self.ctx.work_store.delete(focus_id)
            elif intervention_decision.kind in (
                InterventionKind.SURFACE_CLARIFICATION,
                InterventionKind.SURFACE_APPROVAL,
            ):
                self._trace(
                    "intervention_surface",
                    {
                        "kind": intervention_decision.kind.value,
                        "target": intervention_decision.target_item_id,
                        "reason": intervention_decision.reason,
                    },
                )
            elif intervention_decision.kind == InterventionKind.VERIFY_COMPLETED_WORK:
                self._trace(
                    "intervention_verify",
                    {
                        "target": intervention_decision.target_item_id,
                        "reason": intervention_decision.reason,
                    },
                )
            elif intervention_decision.kind == InterventionKind.RECLAIM_TO_FOREGROUND:
                self._trace(
                    "intervention_reclaim",
                    {
                        "target": intervention_decision.target_item_id,
                        "reason": intervention_decision.reason,
                    },
                )
        except Exception as exc:
            self._trace("workspace_intervention_error", {"error": str(exc)})

        # Drain executive queue: submit ready work to BAA
        drained_count = 0
        while self.executive.capacity_remaining >= 0 and not self.executive.recommend_pause():
            work = self.executive.dequeue()
            if work is None:
                break
            self.executive.set_intention_from_work(work)
            if work.stage == WorkStage.PROJECT:
                await self.orchestrator.decompose(work)
            else:
                from opencas.execution.models import RepairTask
                meta: Dict[str, Any] = {}
                if work.meta:
                    meta.update(work.meta)
                repair_task = RepairTask(
                    objective=work.content,
                    project_id=work.project_id,
                    commitment_id=work.commitment_id,
                    meta=meta,
                )
                await self.baa.submit(repair_task)
            drained_count += 1

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
        record = await self.compactor.compact_session(session_id, tail_size=tail_size)
        if record:
            self._trace(
                "compaction_triggered",
                {
                    "session_id": session_id,
                    "removed_count": record.removed_count,
                    "compaction_id": str(record.compaction_id),
                },
            )

    async def run_consolidation(self) -> Dict[str, Any]:
        """Run the nightly consolidation engine."""
        self._set_activity("consolidating")
        try:
            result = await self.consolidation.run()
            return result.model_dump(mode="json")
        finally:
            self._set_activity("idle")

    def check_metacognition(self) -> Dict[str, Any]:
        """Run a metacognitive consistency check via ToM."""
        result = self.tom.check_consistency()
        return {
            "contradictions": result.contradictions,
            "warnings": result.warnings,
            "belief_count": result.belief_count,
            "intention_count": result.intention_count,
        }

    async def rebuild_identity(self) -> Dict[str, Any]:
        """Rebuild identity from autobiographical memory and apply to self-model."""
        result = await self.rebuilder.rebuild()
        await self.rebuilder.apply(result, self.ctx.identity)
        self._trace("identity_rebuilt", {
            "source_episode_count": len(result.source_episode_ids),
            "confidence": result.confidence,
            "has_narrative": bool(result.narrative),
        })
        return result.model_dump(mode="json")

    async def _build_tool_use_context(self, session_id: str) -> ToolUseContext:
        """Create a ToolUseContext, restoring active plan state if present."""
        ctx = ToolUseContext(runtime=self, session_id=session_id)
        plan_store = getattr(self.ctx, "plan_store", None)
        if plan_store is not None:
            try:
                active_plans = await plan_store.list_active()
                if active_plans:
                    ctx.plan_mode = True
                    ctx.active_plan_id = active_plans[0].plan_id
            except Exception:
                pass
        return ctx

    async def _discover_and_register_mcp_tools(self) -> List[str]:
        """Eagerly discover and register all MCP tools."""
        registry = getattr(self.ctx, "mcp_registry", None)
        if registry is None:
            return []
        registered: List[str] = []
        from opencas.tools.mcp_adapter import make_mcp_tool_adapter
        for server_name in list(registry._configs.keys()):
            ok = await registry.ensure_initialized(server_name)
            if not ok:
                continue
            for tool_meta in registry._tools.get(server_name, {}).values():
                tool_name = tool_meta["name"]
                adapter = make_mcp_tool_adapter(registry, server_name, tool_name)
                self.tools.register(
                    tool_name,
                    tool_meta.get("description", f"MCP tool {tool_name}"),
                    adapter,
                    ActionRiskTier.READONLY,
                    tool_meta.get("inputSchema", {"type": "object"}),
                    plugin_id=f"mcp:{server_name}",
                )
                registered.append(tool_name)
        return registered

    async def register_mcp_server_tools(self, server_name: str) -> List[str]:
        """Lazy-register tools from a specific MCP server."""
        registry = getattr(self.ctx, "mcp_registry", None)
        if registry is None:
            return []
        ok = await registry.ensure_initialized(server_name)
        if not ok:
            return []
        registered: List[str] = []
        from opencas.tools.mcp_adapter import make_mcp_tool_adapter
        for tool_meta in registry._tools.get(server_name, {}).values():
            tool_name = tool_meta["name"]
            if tool_name in self.tools._tools:
                continue
            adapter = make_mcp_tool_adapter(registry, server_name, tool_name)
            self.tools.register(
                tool_name,
                tool_meta.get("description", f"MCP tool {tool_name}"),
                adapter,
                ActionRiskTier.READONLY,
                tool_meta.get("inputSchema", {"type": "object"}),
                plugin_id=f"mcp:{server_name}",
            )
            registered.append(tool_name)
        return registered

    def _make_mcp_list_servers_adapter(self):
        from opencas.tools.models import ToolResult
        async def adapter(name: str, args: Dict[str, Any]) -> ToolResult:
            registry = getattr(self.ctx, "mcp_registry", None)
            if registry is None:
                return ToolResult(success=True, output="No MCP registry configured.", metadata={})
            configs = getattr(registry, "_configs", {})
            initialized = getattr(registry, "_initialized", set())
            lines = []
            for sname, cfg in configs.items():
                status = "initialized" if sname in initialized else "not_initialized"
                lines.append(f"{sname}: {status} (command: {cfg.command})")
            return ToolResult(
                success=True,
                output="\n".join(lines) or "No MCP servers configured.",
                metadata={"servers": list(configs.keys()), "initialized": list(initialized)},
            )
        return adapter

    def _make_mcp_register_adapter(self):
        from opencas.tools.models import ToolResult
        async def adapter(name: str, args: Dict[str, Any]) -> ToolResult:
            server_name = str(args.get("server_name", ""))
            try:
                registered = await self.register_mcp_server_tools(server_name)
                return ToolResult(
                    success=True,
                    output=f"Registered {len(registered)} tools from server '{server_name}'.",
                    metadata={"registered": registered, "server": server_name},
                )
            except Exception as exc:
                return ToolResult(success=False, output=str(exc), metadata={"server": server_name})
        return adapter

    async def execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool through the registry after self-approval."""
        entry = self.tools.get(name)
        if entry is None:
            return {"success": False, "output": f"Tool not found: {name}"}

        plugin_lifecycle = getattr(self.ctx, "plugin_lifecycle", None)
        if plugin_lifecycle is not None and plugin_lifecycle.is_tool_disabled(name):
            return {
                "success": False,
                "output": f"Tool {name} is disabled because its plugin is disabled.",
            }

        request_payload = dict(args)
        if name in {
            "bash_run_command",
            "pty_start",
            "pty_interact",
            "pty_write",
            "pty_poll",
            "pty_observe",
            "process_start",
            "workflow_supervise_session",
        }:
            from opencas.tools.validation import assess_command

            command = str(args.get("command", "")).strip()
            if command:
                assessment = assess_command(command)
                request_payload.update(
                    {
                        "command_family": assessment.family,
                        "command_permission_class": assessment.permission_class,
                        "command_executable": assessment.executable,
                        "command_subcommand": assessment.subcommand,
                    }
                )
            elif name in {"pty_interact", "pty_write", "workflow_supervise_session"} and args.get("session_id"):
                request_payload.update(
                    {
                        "command_family": "interactive_session",
                        "command_permission_class": "bounded_write",
                    }
                )
            elif name in {"pty_poll", "pty_observe"} and args.get("session_id"):
                request_payload.update(
                    {
                        "command_family": "interactive_session",
                        "command_permission_class": "read_only",
                    }
                )
        elif name in {"pty_kill", "pty_remove"}:
            request_payload.update(
                {
                    "command_family": "interactive_session",
                    "command_permission_class": "bounded_write",
                }
            )

        request = ActionRequest(
            tier=entry.risk_tier,
            description=f"tool {name}: {entry.description}",
            tool_name=name,
            payload=request_payload,
        )
        approval = await self.handle_action(request)
        if not approval["approved"]:
            if entry.risk_tier != ActionRiskTier.READONLY:
                self.ctx.somatic.bump_from_work(intensity=0.05, success=False)
            await self.ctx.somatic.emit_appraisal_event(
                AppraisalEventType.TOOL_REJECTED,
                source_text=f"tool {name} rejected",
                trigger_event_id=str(request.action_id),
                meta={"tool_name": name, "args": args},
            )
            return {
                "success": False,
                "output": f"Tool execution blocked: {approval['decision'].reasoning}",
                "decision": approval["decision"],
            }

        result = await self.tools.execute_async(name, args)
        if entry.risk_tier != ActionRiskTier.READONLY:
            self.ctx.somatic.bump_from_work(intensity=0.1, success=result.success)
        await self.ctx.somatic.emit_appraisal_event(
            AppraisalEventType.TOOL_EXECUTED,
            source_text=f"tool {name} executed",
            trigger_event_id=str(request.action_id),
            meta={"tool_name": name, "success": result.success},
        )
        if result.success:
            resolved_goals = await self.executive.check_goal_resolution(result.output)
            for goal in resolved_goals:
                await self.ctx.somatic.emit_appraisal_event(
                    AppraisalEventType.GOAL_ACHIEVED,
                    source_text=f"Goal achieved: {goal}",
                    trigger_event_id=str(request.action_id),
                )
        self._sync_executive_snapshot()
        return {
            "success": result.success,
            "output": result.output,
            "metadata": result.metadata,
        }

    async def install_plugin(self, path: Path | str) -> Optional[Any]:
        """Install a plugin from a directory or manifest file."""
        lifecycle = getattr(self.ctx, "plugin_lifecycle", None)
        if lifecycle is None:
            return None
        return await lifecycle.install(path)

    async def uninstall_plugin(self, plugin_id: str) -> None:
        """Uninstall a plugin."""
        lifecycle = getattr(self.ctx, "plugin_lifecycle", None)
        if lifecycle is not None:
            await lifecycle.uninstall(plugin_id)

    async def enable_plugin(self, plugin_id: str) -> None:
        """Enable a plugin."""
        lifecycle = getattr(self.ctx, "plugin_lifecycle", None)
        if lifecycle is not None:
            await lifecycle.enable(plugin_id)

    async def disable_plugin(self, plugin_id: str) -> None:
        """Disable a plugin."""
        lifecycle = getattr(self.ctx, "plugin_lifecycle", None)
        if lifecycle is not None:
            await lifecycle.disable(plugin_id)

    async def submit_repair(self, task) -> Any:
        """Submit a repair task to the bounded assistant agent and return a future."""
        await self.baa.start()
        return await self.baa.submit(task)

    async def handle_action(self, request: ActionRequest) -> Dict[str, Any]:
        """Evaluate an action through the self-approval ladder."""
        decision = self.approval.evaluate(request)
        await self.approval.maybe_record(decision, request, decision.score)
        self._trace(
            "handle_action",
            {
                "action_id": str(request.action_id),
                "decision": decision.level.value,
                "confidence": decision.confidence,
            },
        )
        return {
            "approved": decision.level in (
                ApprovalLevel.CAN_DO_NOW,
                ApprovalLevel.CAN_DO_WITH_CAUTION,
            ),
            "decision": decision,
        }

    async def run_autonomous(
        self,
        cycle_interval: int = 300,
        consolidation_interval: int = 86400,
    ) -> None:
        """Start background scheduler and block until interrupted."""
        scheduler = AgentScheduler(
            runtime=self,
            cycle_interval=cycle_interval,
            consolidation_interval=consolidation_interval,
            readiness=self.readiness,
            tracer=self.tracer,
        )
        shutdown_event = asyncio.Event()

        def _on_signal(sig: int) -> None:
            self._trace("signal_received", {"signal": sig})
            shutdown_event.set()

        try:
            import signal
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, _on_signal, signal.SIGINT)
            loop.add_signal_handler(signal.SIGTERM, _on_signal, signal.SIGTERM)
        except (NotImplementedError, ValueError):
            pass  # Windows or tests may not support signal handlers

        self.scheduler = scheduler
        await scheduler.start()
        await self.start_telegram()
        self.readiness.ready("autonomous_mode_active")
        self._trace("autonomous_start", {})
        await shutdown_event.wait()

        self.readiness.shutdown("signal_received")
        await scheduler.stop()
        self.scheduler = None
        await self._close_stores()
        self._trace("autonomous_shutdown", {})

    async def run_autonomous_with_server(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        cycle_interval: int = 300,
        consolidation_interval: int = 86400,
    ) -> None:
        """Run scheduler + FastAPI server together, shutting down gracefully on signal."""
        scheduler = AgentScheduler(
            runtime=self,
            cycle_interval=cycle_interval,
            consolidation_interval=consolidation_interval,
            readiness=self.readiness,
            tracer=self.tracer,
        )

        app = create_app(self)
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)

        shutdown_event = asyncio.Event()

        def _on_signal(sig: int) -> None:
            self._trace("signal_received", {"signal": sig})
            shutdown_event.set()

        try:
            import signal
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, _on_signal, signal.SIGINT)
            loop.add_signal_handler(signal.SIGTERM, _on_signal, signal.SIGTERM)
        except (NotImplementedError, ValueError):
            pass

        self.scheduler = scheduler
        await scheduler.start()
        await self.start_telegram()
        self.readiness.ready("autonomous_mode_with_server")
        self._trace("autonomous_with_server_start", {"host": host, "port": port})

        server_task = asyncio.create_task(server.serve())
        await shutdown_event.wait()

        self.readiness.shutdown("signal_received")
        server.should_exit = True
        await server_task
        await scheduler.stop()
        self.scheduler = None
        await self._close_stores()
        self._trace("autonomous_with_server_shutdown", {})



    async def _maybe_record_somatic_snapshot(
        self,
        source: str,
        trigger_event_id: Optional[str] = None,
    ) -> None:
        if self.ctx.somatic.store is not None:
            await self.ctx.somatic.record_snapshot(
                source=source, trigger_event_id=trigger_event_id
            )

    async def _on_baa_completed(self, event: BaaCompletedEvent) -> None:
        """Handle BAA task completion: resolve goals and persist executive state."""
        if event.success:
            resolved_goals = await self.executive.check_goal_resolution(event.output)
            for goal in resolved_goals:
                await self.ctx.somatic.emit_appraisal_event(
                    AppraisalEventType.GOAL_ACHIEVED,
                    source_text=f"Goal achieved: {goal}",
                    trigger_event_id=event.task_id,
                )
        else:
            await self.ctx.somatic.emit_appraisal_event(
                AppraisalEventType.GOAL_BLOCKED,
                source_text=f"BAA task failed: {event.objective}",
                trigger_event_id=event.task_id,
                meta={"stage": event.stage, "output": event.output},
            )
        self._sync_executive_snapshot()

    def _sync_executive_snapshot(self) -> None:
        snapshot_path = self.ctx.config.state_dir / "executive.json"
        self.executive.save_snapshot(snapshot_path)

    @staticmethod
    def _extract_goal_directives(text: str) -> tuple[List[str], Optional[str], List[str]]:
        """Heuristically extract goals, intention, and drop-requests from user text."""
        text_lower = text.lower()
        goals: List[str] = []
        intention: Optional[str] = None
        drops: List[str] = []

        # Goal patterns
        goal_patterns = [
            r"(?:your goal is|you should|i want you to|focus on|make it your goal to|prioritize)\s+(.*?)(?:[.]|$)",
        ]
        import re
        for pattern in goal_patterns:
            for match in re.finditer(pattern, text_lower):
                clause = match.group(1).strip()
                if clause:
                    goals.append(clause)

        # Intention patterns
        intention_patterns = [
            r"(?:current task is|work on|start on|intention is)\s+(.*?)(?:[.]|$)",
        ]
        for pattern in intention_patterns:
            match = re.search(pattern, text_lower)
            if match:
                intention = match.group(1).strip()
                break

        # Drop patterns
        drop_phrases = ["done with that", "drop the goal", "forget about", "stop working on"]
        for phrase in drop_phrases:
            if phrase in text_lower:
                # Naive: drop the most recently added matching goal or all if vague
                drops.append(phrase)
        return goals, intention, drops

    async def _close_stores(self) -> None:
        """Gracefully close all SQLite stores."""
        if self.reliability:
            self.reliability.stop()
        if hasattr(self, "process_supervisor") and self.process_supervisor:
            self.process_supervisor.shutdown()
        if hasattr(self, "pty_supervisor") and self.pty_supervisor:
            self.pty_supervisor.shutdown()
        if hasattr(self, "browser_supervisor") and self.browser_supervisor:
            await self.browser_supervisor.shutdown()
        if hasattr(self.ctx, "daydream_store") and self.ctx.daydream_store:
            await self.ctx.daydream_store.close()
        if hasattr(self.ctx, "conflict_store") and self.ctx.conflict_store:
            await self.ctx.conflict_store.close()
        if hasattr(self.ctx, "curation_store") and self.ctx.curation_store:
            await self.ctx.curation_store.close()
        await self.ctx.context_store.close()
        await self.memory.close()
        await self.ctx.tasks.close()
        if hasattr(self.ctx, "work_store") and self.ctx.work_store:
            await self.ctx.work_store.close()
        if hasattr(self.ctx, "relational") and self.ctx.relational:
            await self.ctx.relational.close()
        if hasattr(self.ctx, "harness") and self.ctx.harness and self.ctx.harness.store:
            await self.ctx.harness.store.close()
        if self.commitment_store:
            await self.commitment_store.close()
        if self.portfolio_store:
            await self.portfolio_store.close()
        if hasattr(self.ctx, "tom_store") and self.ctx.tom_store:
            await self.ctx.tom_store.close()
        if hasattr(self.ctx, "plugin_store") and self.ctx.plugin_store:
            await self.ctx.plugin_store.close()
        if hasattr(self.ctx, "plan_store") and self.ctx.plan_store:
            await self.ctx.plan_store.close()
        if hasattr(self.ctx, "schedule_store") and self.ctx.schedule_store:
            await self.ctx.schedule_store.close()
        if self._telegram is not None:
            try:
                await self._telegram.stop()
            except Exception:
                pass
        self.ctx.identity.record_shutdown()

    def control_plane_status(self) -> Dict[str, Any]:
        """Return a monitoring snapshot of workspace, sandbox, and execution state."""
        configured_workspace_roots = [
            str(root) for root in self.ctx.config.all_workspace_roots()
        ]
        sandbox = getattr(self.ctx, "sandbox", None)
        sandbox_report = sandbox.report_isolation() if sandbox is not None else {}
        readiness = self.readiness.snapshot() if self.readiness else {"state": "unknown"}
        return {
            "agent_profile": self.agent_profile.model_dump(mode="json"),
            "readiness": readiness,
            "workspace": {
                "session_id": getattr(self.ctx.config, "session_id", None),
                "state_dir": str(self.ctx.config.state_dir),
                "primary_root": str(self.ctx.config.primary_workspace_root()),
                "workspace_roots": configured_workspace_roots,
                "allowed_roots": [
                    str(root) for root in getattr(sandbox, "allowed_roots", [])
                ],
            },
            "sandbox": sandbox_report,
            "execution": {
                "processes": self.process_supervisor.snapshot(sample_limit=10),
                "pty": self.pty_supervisor.snapshot(sample_limit=10),
                "browser": self.browser_supervisor.snapshot(sample_limit=10),
            },
            "activity": {
                "current": self._activity,
                "since": self._activity_since.isoformat(),
            },
            "lanes": self.baa.lane_snapshot() if self.baa else {},
        }

    async def workflow_status(
        self,
        limit: int = 10,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a summarized view of current higher-level workflow state."""
        active_commitments = []
        commitment_count = 0
        if self.commitment_store:
            active_commitments = await self.commitment_store.list_active(limit=limit)
            commitment_count = await self.commitment_store.count_by_status(
                CommitmentStatus.ACTIVE
            )

        work_counts = {"total": 0, "ready": 0, "blocked": 0}
        work_items = []
        blocked_items = []
        if self.ctx.work_store:
            work_counts = await self.ctx.work_store.summary_counts()
            if project_id:
                work_items = await self.ctx.work_store.list_by_project(
                    project_id, limit=limit
                )
                blocked_items = [item for item in work_items if item.blocked_by][:limit]
            else:
                work_items = await self.ctx.work_store.list_all(limit=limit)
                blocked_items = await self.ctx.work_store.list_blocked(limit=limit)

        active_plans = []
        active_plan_count = 0
        if getattr(self.ctx, "plan_store", None) is not None:
            active_plans = await self.ctx.plan_store.list_active(project_id=project_id)
            active_plan_count = await self.ctx.plan_store.count_active(
                project_id=project_id
            )

        plan_entries = []
        for plan in active_plans[:limit]:
            actions = await self.ctx.plan_store.get_actions(plan.plan_id, limit=5)
            plan_entries.append(
                {
                    "plan_id": plan.plan_id,
                    "status": plan.status,
                    "content_preview": plan.content[:240],
                    "project_id": plan.project_id,
                    "task_id": plan.task_id,
                    "updated_at": plan.updated_at.isoformat(),
                    "recent_action_count": len(actions),
                }
            )

        recent_receipts = []
        if getattr(self.ctx, "receipt_store", None) is not None:
            recent_receipts = await self.ctx.receipt_store.list_recent(limit=limit)

        active_projects = []
        seen_projects = set()
        for item in work_items:
            if item.project_id and item.project_id not in seen_projects:
                seen_projects.add(item.project_id)
                active_projects.append(item.project_id)

        return {
            "agent_profile": self.agent_profile.model_dump(mode="json"),
            "executive": {
                "intention": self.executive.intention,
                "active_goals": list(self.executive.active_goals),
                "queued_work_count": len(self.executive.task_queue),
                "capacity_remaining": self.executive.capacity_remaining,
                "recommend_pause": self.executive.recommend_pause(),
            },
            "commitments": {
                "active_count": commitment_count,
                "items": [
                    {
                        "commitment_id": str(item.commitment_id),
                        "content": item.content,
                        "priority": item.priority,
                        "deadline": item.deadline.isoformat()
                        if item.deadline
                        else None,
                        "tags": item.tags,
                    }
                    for item in active_commitments
                ],
            },
            "work": {
                "counts": work_counts,
                "active_projects": active_projects[:limit],
                "items": [
                    {
                        "work_id": str(item.work_id),
                        "content": item.content,
                        "stage": item.stage.value,
                        "project_id": item.project_id,
                        "commitment_id": item.commitment_id,
                        "blocked_by": item.blocked_by,
                        "meta": item.meta,
                    }
                    for item in work_items
                ],
                "blocked_items": [
                    {
                        "work_id": str(item.work_id),
                        "content": item.content,
                        "stage": item.stage.value,
                        "project_id": item.project_id,
                        "blocked_by": item.blocked_by,
                    }
                    for item in blocked_items
                ],
            },
            "plans": {
                "active_count": active_plan_count,
                "items": plan_entries,
            },
            "receipts": {
                "recent_count": len(recent_receipts),
                "items": [
                    {
                        "receipt_id": str(item.receipt_id),
                        "task_id": str(item.task_id),
                        "objective": item.objective,
                        "success": item.success,
                        "created_at": item.created_at.isoformat(),
                        "completed_at": item.completed_at.isoformat()
                        if item.completed_at
                        else None,
                        "checkpoint_commit": item.checkpoint_commit,
                    }
                    for item in recent_receipts
                ],
            },
        }

    async def _record_episode(
        self,
        content: str,
        kind: EpisodeKind,
        session_id: Optional[str] = None,
    ) -> Episode:
        episode = Episode(
            kind=kind,
            session_id=session_id or self.ctx.config.session_id,
            content=content,
            somatic_tag=self.ctx.somatic.state.somatic_tag,
            affect=None,
        )
        # Apply somatic and musubi salience modifiers
        salience = 1.0
        salience *= self.ctx.somatic.state.to_memory_salience_modifier()
        if hasattr(self.ctx, "relational") and self.ctx.relational:
            has_collab_tag = bool(
                episode.affect
                and episode.affect.primary_emotion.value in {"joy", "anticipation", "trust", "excited"}
            )
            salience += self.ctx.relational.to_memory_salience_modifier(
                has_user_collab_tag=has_collab_tag
            )
        episode.salience = round(max(0.0, min(10.0, salience)), 3)
        await self.memory.save_episode(episode)

        # Create temporal / emotional edge to previous episode in session
        await self._link_episode_to_previous(episode)

        # Persist somatic trajectory
        await self._maybe_record_somatic_snapshot(
            source="conversation", trigger_event_id=str(episode.episode_id)
        )
        return episode

    async def _link_episode_to_previous(self, episode: Episode) -> None:
        """Create graph edges between this episode and the most recent prior episode."""
        if not episode.session_id:
            return
        recent = await self.memory.list_recent_episodes(
            session_id=episode.session_id, limit=2
        )
        prev = None
        for ep in recent:
            if str(ep.episode_id) != str(episode.episode_id):
                prev = ep
                break
        if prev is None:
            return

        emotional_weight = 0.0
        structural_weight = 0.0
        if episode.affect and prev.affect:
            if episode.affect.primary_emotion == prev.affect.primary_emotion:
                emotional_weight = 0.8
            # Structural affinity from shared project/task in payload
            ep_project = episode.payload.get("project_id")
            prev_project = prev.payload.get("project_id")
            if ep_project and prev_project and ep_project == prev_project:
                structural_weight = 0.6

        edge = EpisodeEdge(
            source_id=str(prev.episode_id),
            target_id=str(episode.episode_id),
            kind=EdgeKind.TEMPORAL,
            recency_weight=1.0,
            emotional_weight=emotional_weight,
            structural_weight=structural_weight,
            confidence=round(0.5 + (emotional_weight * 0.2) + (structural_weight * 0.1), 3),
        )
        await self.memory.save_edge(edge)

    def _extract_content(self, response: Dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "")
        return ""

    def _trace(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self.tracer:
            self.tracer.log(
                EventKind.TOM_EVAL,
                f"AgentRuntime: {event}",
                payload or {},
            )
