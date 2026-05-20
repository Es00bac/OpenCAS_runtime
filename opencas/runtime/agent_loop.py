"""Main agent runtime loop for OpenCAS."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from opencas.affective_registry.models import ExecutionPhase as AffectiveExecutionPhase
from opencas.api import LLMClient
from opencas.autonomy import WorkObject
from opencas.autonomy.authorization import grant_authorizations_from_user_text
from opencas.autonomy.commitment import Commitment
from opencas.autonomy.commitment_extraction import SelfCommitmentCandidate
from opencas.autonomy.models import ActionRequest
from opencas.bootstrap import BootstrapContext
from opencas.daydream import (
    DaydreamReflection,
)
from opencas.infra import BaaCompletedEvent
from opencas.memory import Episode, EpisodeKind, MemoryStore
from opencas.phone_config import PhoneRuntimeConfig
from opencas.runtime.agent_profile import get_agent_profile
from opencas.runtime.readiness import AgentReadiness
from opencas.runtime.single_instance import SingleInstanceLock
from opencas.telegram_config import TelegramRuntimeConfig
from opencas.tools import ToolUseContext

from .body_double_voice import maybe_speak_body_double_response
from .cognitive_runtime import run_runtime_cognitive_maintenance
from .continuity_breadcrumbs import current_runtime_focus, record_burst_continuity
from .conversation_recovery import (
    complete_conversation_turn_marker,
    start_conversation_turn_marker,
)
from .conversation_directives import should_skip_semantic_values_review_for_direct_turn
from .conversation_turns import (
    execute_conversation_tool_loop,
    finalize_assistant_turn,
    handle_refusal_turn,
    persist_tool_loop_messages,
    persist_user_turn,
)
from .consolidation_worker import cancel_active_consolidation_worker
from .cycle_phases import (
    drain_executive_cycle_queue,
    enqueue_promoted_cycle_work,
    evaluate_workspace_intervention,
)
from .dreaming_runtime import run_runtime_nightly_dream
from .episodic_runtime import (
    capture_runtime_self_commitments,
    extract_runtime_goal_directives,
    extract_runtime_self_commitments,
    record_runtime_episode,
    run_runtime_continuity_check,
)
from .project_return import prepare_project_followthrough_response
from .lifecycle import (
    run_autonomous_runtime,
    run_autonomous_with_server_runtime,
)
from .maintenance_runtime import (
    close_runtime_stores,
    compact_runtime_backlog,
    extract_runtime_response_content,
    handle_runtime_baa_completed,
    maybe_compact_runtime_session,
    maybe_record_runtime_somatic_snapshot,
    run_runtime_consolidation,
    sync_runtime_executive_snapshot,
    trace_runtime_event,
)
from .phone_runtime import (
    autoconfigure_runtime_phone,
    call_owner_via_runtime_phone,
    configure_runtime_phone,
    configure_runtime_phone_menu_config,
    get_runtime_phone_call_detail,
    get_runtime_phone_status,
    get_runtime_recent_phone_calls,
    handle_runtime_phone_gather_webhook,
    handle_runtime_phone_media_stream,
    handle_runtime_phone_poll_webhook,
    handle_runtime_phone_voice_webhook,
    runtime_phone_settings,
)
from .reflection_runtime import (
    build_runtime_metacognition_status,
    rebuild_runtime_identity,
    run_runtime_daydream,
    run_runtime_daydream_inner,
)
from .runtime_setup import (
    initialize_runtime_autonomy,
    initialize_runtime_channels,
    initialize_runtime_execution,
    initialize_runtime_memory_surfaces,
)
from opencas.thread_registry.shadow_bridge import record_shadow_registry_thread_beads
from .shadow_consumer_runtime import consume_shadow_registry
from .status_views import (
    build_consolidation_status,
    build_control_plane_status,
    build_workflow_status,
)
from opencas.platform.telegram_runtime import (
    approve_runtime_telegram_pairing,
    build_runtime_telegram_service,
    configure_runtime_telegram,
    get_runtime_telegram_status,
    runtime_telegram_settings,
    start_runtime_telegram,
)
from opencas.thread_registry.runtime_bridge import record_runtime_daydream_thread_beads
from .tool_registration import register_runtime_default_tools
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
from .audit_mode import is_audit_only_text
from .cognition import CognitionFrame
from .cognition_bus import (
    AffectiveEvent,
    AuditModeTurnObserved,
    AuthorizationGranted,
    CognitionBus,
    LadderHealthSummary,
)
from .cognition_subscribers import classify_user_affective_event, register_runtime_cognition_subscribers
from .wellbeing_runtime import (
    record_runtime_daydream_wellbeing,
    run_runtime_wellbeing_maintenance,
)


async def _capture_affective_registry_turn_end(
    runtime: Any,
    *,
    session_id: str,
    outcome: str,
) -> None:
    writer = getattr(getattr(runtime, "ctx", None), "affective_registry_writer", None)
    if writer is None:
        writer = getattr(runtime, "affective_registry_writer", None)
    somatic = getattr(getattr(runtime, "ctx", None), "somatic", None)
    if writer is None or somatic is None:
        return
    try:
        writer.append_from_somatic_state(
            somatic.state,
            phase=AffectiveExecutionPhase.TURN_END,
            session_id=session_id,
            payload={"outcome": outcome},
        )
    except Exception as exc:
        trace = getattr(runtime, "_trace", None)
        if callable(trace):
            trace(
                "affective_registry_turn_end_error",
                {"session_id": session_id, "error": str(exc)},
            )


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
        self.cognition_bus = CognitionBus(context.config.state_dir / "bus_failures.db")
        self.current_cognition: CognitionFrame | None = None

        initialize_runtime_autonomy(self, context)
        initialize_runtime_execution(self, context)
        initialize_runtime_memory_surfaces(self, context)
        initialize_runtime_channels(self, context)
        register_runtime_cognition_subscribers(self)

        # Activity tracking — what the runtime is currently doing (operator-visible)
        self._activity: str = "idle"
        self._activity_since: datetime = datetime.now(timezone.utc)
        self._last_user_turn_at: Optional[datetime] = None
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

    @property
    def phone_settings(self) -> PhoneRuntimeConfig:
        return runtime_phone_settings(self)

    async def phone_status(self) -> Dict[str, Any]:
        return await get_runtime_phone_status(self)

    async def recent_phone_calls(self, *, limit: int = 10) -> Dict[str, Any]:
        return await get_runtime_recent_phone_calls(self, limit=limit)

    async def phone_call_detail(self, call_sid: str) -> Dict[str, Any]:
        return await get_runtime_phone_call_detail(self, call_sid=call_sid)

    async def configure_phone(self, settings: PhoneRuntimeConfig) -> Dict[str, Any]:
        return await configure_runtime_phone(self, settings)

    async def configure_phone_session_profiles(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from opencas.runtime.phone_runtime import configure_runtime_phone_session_profiles

        return await configure_runtime_phone_session_profiles(self, payload)

    async def configure_phone_menu_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await configure_runtime_phone_menu_config(self, payload)

    async def autoconfigure_phone(
        self,
        *,
        enabled: bool | None = None,
        public_base_url: str | None = None,
        webhook_signature_required: bool | None = None,
        webhook_secret: str | None = None,
        twilio_from_number: str | None = None,
        owner_phone_number: str | None = None,
        owner_display_name: str | None = None,
        owner_workspace_subdir: str | None = None,
    ) -> Dict[str, Any]:
        return await autoconfigure_runtime_phone(
            self,
            enabled=enabled,
            public_base_url=public_base_url,
            webhook_signature_required=webhook_signature_required,
            webhook_secret=webhook_secret,
            twilio_from_number=twilio_from_number,
            owner_phone_number=owner_phone_number,
            owner_display_name=owner_display_name,
            owner_workspace_subdir=owner_workspace_subdir,
        )

    async def call_owner_via_phone(self, *, message: str, reason: str = "") -> Dict[str, Any]:
        return await call_owner_via_runtime_phone(self, message=message, reason=reason)

    async def initiative_contact_owner(
        self,
        *,
        message: str,
        reason: str = "",
        urgency: str = "normal",
        channel: str = "auto",
    ) -> Dict[str, Any]:
        return await self.initiative_contact.request_contact(
            message=message,
            reason=reason,
            urgency=urgency,  # type: ignore[arg-type]
            source="runtime",
            channel=channel,  # type: ignore[arg-type]
        )

    def initiative_contact_status(self) -> Dict[str, Any]:
        return self.initiative_contact.status()

    async def maybe_run_initiative_contact(self) -> Dict[str, Any]:
        return await self.initiative_contact.run_once()

    def desktop_context_status(self) -> Dict[str, Any]:
        return self.desktop_context.status()

    async def maybe_run_desktop_context(self) -> Dict[str, Any]:
        return await self.desktop_context.run_once()

    async def handle_phone_voice_webhook(
        self,
        *,
        request_url: str,
        webhook_base_url: str,
        form_data: Dict[str, Any],
        provided_signature: Optional[str],
        call_token: Optional[str] = None,
        bridge_token: Optional[str] = None,
    ) -> str:
        return await handle_runtime_phone_voice_webhook(
            self,
            request_url=request_url,
            webhook_base_url=webhook_base_url,
            form_data=form_data,
            provided_signature=provided_signature,
            call_token=call_token,
            bridge_token=bridge_token,
        )

    async def handle_phone_gather_webhook(
        self,
        *,
        request_url: str,
        webhook_base_url: str,
        form_data: Dict[str, Any],
        provided_signature: Optional[str],
        call_token: Optional[str] = None,
        bridge_token: Optional[str] = None,
    ) -> str:
        return await handle_runtime_phone_gather_webhook(
            self,
            request_url=request_url,
            webhook_base_url=webhook_base_url,
            form_data=form_data,
            provided_signature=provided_signature,
            call_token=call_token,
            bridge_token=bridge_token,
        )

    async def handle_phone_poll_webhook(
        self,
        *,
        request_url: str,
        webhook_base_url: str,
        form_data: Dict[str, Any],
        provided_signature: Optional[str],
        call_token: Optional[str] = None,
        bridge_token: Optional[str] = None,
        reply_token: Optional[str] = None,
    ) -> str:
        return await handle_runtime_phone_poll_webhook(
            self,
            request_url=request_url,
            webhook_base_url=webhook_base_url,
            form_data=form_data,
            provided_signature=provided_signature,
            call_token=call_token,
            bridge_token=bridge_token,
            reply_token=reply_token,
        )

    async def handle_phone_media_stream(
        self,
        *,
        websocket: Any,
        request_url: str,
        provided_signature: Optional[str],
        stream_secret: str,
    ) -> None:
        await handle_runtime_phone_media_stream(
            self,
            websocket=websocket,
            request_url=request_url,
            provided_signature=provided_signature,
            stream_secret=stream_secret,
        )

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

    async def _interrupt_background_consolidation_for_user_turn(self, session_id: str) -> None:
        """Let foreground conversation preempt retryable consolidation work."""
        config = getattr(self.ctx, "config", None)
        state_dir = getattr(config, "state_dir", None)
        if state_dir is None:
            return
        try:
            payload = await asyncio.to_thread(
                cancel_active_consolidation_worker,
                state_dir,
                reason="foreground_user_turn",
                grace_seconds=2.0,
            )
        except Exception as exc:
            self._trace(
                "foreground_consolidation_interrupt_failed",
                {"session_id": session_id, "error": str(exc)},
            )
            return
        if payload.get("cancelled") is False:
            return
        worker = payload.get("worker") if isinstance(payload, dict) else None
        self._trace(
            "foreground_consolidation_interrupted",
            {
                "session_id": session_id,
                "reason": payload.get("budget_reason") or "foreground_user_turn",
                "worker_status": worker.get("status") if isinstance(worker, dict) else None,
                "worker_pid": worker.get("pid") if isinstance(worker, dict) else None,
                "run_id": worker.get("run_id") if isinstance(worker, dict) else None,
            },
        )

    async def converse(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        user_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Process one conversational turn while delegating phase logic to runtime helpers."""
        turn_started = time.perf_counter()
        self._set_activity("conversing")
        self._last_user_turn_at = datetime.now(timezone.utc)
        sid = session_id or self.ctx.config.session_id or "default"
        await self._interrupt_background_consolidation_for_user_turn(sid)

        def _trace_phase(phase: str, started_at: float, **extra: Any) -> None:
            try:
                self._trace(
                    "conversation_phase_timing",
                    {
                        "session_id": sid,
                        "phase": phase,
                        "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                        "cumulative_ms": int((time.perf_counter() - turn_started) * 1000),
                        "subsystem": "conversation",
                        **extra,
                    },
                )
            except Exception:
                pass

        user_meta_for_turn: Dict[str, Any] = dict(user_meta or {})
        desktop_context = getattr(self, "desktop_context", None)
        observe_for_conversation = getattr(desktop_context, "observe_for_conversation", None)
        if callable(observe_for_conversation):
            phase_started = time.perf_counter()
            try:
                actor_meta = user_meta_for_turn.get("conversation_actor")
                source = (
                    str(actor_meta.get("source") or "conversation")
                    if isinstance(actor_meta, dict)
                    else "conversation"
                )
                desktop_observation = await observe_for_conversation(
                    session_id=sid,
                    user_input=user_input,
                    source=source,
                )
                if isinstance(desktop_observation, dict):
                    existing_desktop_context = user_meta_for_turn.get("desktop_context_turn")
                    if isinstance(existing_desktop_context, dict):
                        merged_desktop_context = dict(desktop_observation)
                        merged_desktop_context["selected_region_context"] = existing_desktop_context
                        merged_desktop_context["whole_desktop_context"] = desktop_observation
                        prompt_notes = [
                            str(existing_desktop_context.get("conversation_prompt_note") or "").strip(),
                            str(desktop_observation.get("conversation_prompt_note") or "").strip(),
                        ]
                        merged_desktop_context["conversation_prompt_note"] = "\n\n".join(
                            note for note in prompt_notes if note
                        )
                        user_meta_for_turn["desktop_context_turn"] = merged_desktop_context
                    else:
                        user_meta_for_turn["desktop_context_turn"] = desktop_observation
                    _trace_phase(
                        "desktop_context_conversation_observe",
                        phase_started,
                        status=desktop_observation.get("status"),
                    )
            except Exception as exc:
                user_meta_for_turn["desktop_context_turn"] = {
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                _trace_phase("desktop_context_conversation_observe", phase_started, status="failed")

        user_meta = user_meta_for_turn or None
        audit_only = is_audit_only_text(user_input, user_meta)
        cognition = CognitionFrame.from_turn(
            runtime=self,
            session_id=sid,
            user_input=user_input,
            user_meta=user_meta,
            audit_only=audit_only,
        )
        self.current_cognition = cognition
        persisted_user_meta = cognition.user_meta
        if should_skip_semantic_values_review_for_direct_turn(
            user_input,
            persisted_user_meta,
        ):
            persisted_user_meta["semantic_values_review"] = "skip_low_risk_direct"
        if cognition.audit_only:
            await self.cognition_bus.publish(
                AuditModeTurnObserved(
                    session_id=sid,
                    payload={"cognition_frame_id": cognition.frame_id},
                )
            )
        else:
            authorization_store = getattr(self, "authorization_store", None)
            if authorization_store is not None:
                grants = grant_authorizations_from_user_text(
                    authorization_store,
                    user_input,
                    session_id=sid,
                    evidence_episode_id=f"cognition_frame:{cognition.frame_id}",
                    now=cognition.created_at,
                        )
                if grants:
                    cognition.authorization_context = {
                        "grants": [grant.to_payload() for grant in grants],
                    }
                    for grant in grants:
                        await self.cognition_bus.publish(
                            AuthorizationGranted(
                                kind="authorization.granted",
                                source="agent_loop.converse",
                                payload={
                                    "session_id": sid,
                                    "cognition_frame_id": cognition.frame_id,
                                    **grant.to_payload(),
                                },
                                evidence_ids=[grant.evidence_episode_id or cognition.frame_id],
                            )
                        )
        affective_classification = classify_user_affective_event(user_input)
        if affective_classification and not cognition.audit_only:
            affective_kind, magnitude = affective_classification
            await self.cognition_bus.publish(
                AffectiveEvent(
                    kind=affective_kind,
                    source="agent_loop.converse",
                    magnitude=magnitude,
                    payload={
                        "session_id": sid,
                        "cognition_frame_id": cognition.frame_id,
                        "user_input_preview": user_input[:240],
                    },
                    evidence_ids=[f"cognition_frame:{cognition.frame_id}"],
                )
            )
        turn_marker_id: Optional[str] = None
        try:
            try:
                phase_started = time.perf_counter()
                marker = start_conversation_turn_marker(
                    self.ctx.config.state_dir,
                    session_id=sid,
                    user_input=user_input,
                    user_meta=persisted_user_meta,
                )
                turn_marker_id = str(marker.get("marker_id") or "") or None
                _trace_phase("turn_marker_start", phase_started, marker_id=turn_marker_id)
            except Exception as exc:
                self._trace(
                    "conversation_turn_marker_start_error",
                    {"session_id": sid, "error": str(exc)},
                )

            try:
                phase_started = time.perf_counter()
                await record_burst_continuity(
                    self,
                    trigger="conversation_burst_started",
                    phase="start",
                    intent=f"Conversation burst for {user_input[:80]}",
                    focus=current_runtime_focus(self, user_input),
                    next_step="process the turn and update executive state",
                    note="conversation turn",
                )
                _trace_phase("continuity_breadcrumb_start", phase_started)
            except Exception as exc:
                self._trace("continuity_breadcrumb_converse_error", {"session_id": sid, "error": str(exc)})

            # Refusal is handled first so unsafe turns never leak into the tool loop.
            from opencas.refusal.models import ConversationalRequest
            from opencas.runtime.capability_context import build_runtime_capability_context
            conv_request = ConversationalRequest(
                text=user_input,
                session_id=sid,
                meta=persisted_user_meta,
            )
            phase_started = time.perf_counter()
            capability_context = build_runtime_capability_context(self)
            _trace_phase("capability_context", phase_started, context_chars=len(capability_context))
            phase_started = time.perf_counter()
            refusal = await self.refusal_gate.evaluate_async(
                conv_request,
                llm=self.llm,
                capability_context=capability_context,
            )
            _trace_phase("refusal_gate", phase_started, refused=bool(getattr(refusal, "refused", False)))
            if refusal.refused:
                response = await handle_refusal_turn(
                    self,
                    session_id=sid,
                    user_input=user_input,
                    user_meta=persisted_user_meta,
                    refusal=refusal,
                )
                if turn_marker_id:
                    try:
                        phase_started = time.perf_counter()
                        complete_conversation_turn_marker(
                            self.ctx.config.state_dir,
                            turn_marker_id,
                            outcome="refusal_response_persisted",
                        )
                        _trace_phase("turn_marker_complete", phase_started, outcome="refusal_response_persisted")
                    except Exception as exc:
                        self._trace(
                            "conversation_turn_marker_complete_error",
                            {"session_id": sid, "marker_id": turn_marker_id, "error": str(exc)},
                        )
                phase_started = time.perf_counter()
                voice_result = await maybe_speak_body_double_response(
                    self,
                    session_id=sid,
                    user_input=user_input,
                    user_meta=persisted_user_meta,
                    response_text=response,
                )
                _trace_phase(
                    "body_double_voice_response",
                    phase_started,
                    status=voice_result.get("status") if isinstance(voice_result, dict) else None,
                )
                return response

            phase_started = time.perf_counter()
            await persist_user_turn(
                self,
                session_id=sid,
                user_input=user_input,
                user_meta=persisted_user_meta,
            )
            _trace_phase("persist_user_turn", phase_started)
            phase_started = time.perf_counter()
            artifacts = await execute_conversation_tool_loop(
                self,
                session_id=sid,
                user_input=user_input,
                user_meta=persisted_user_meta,
            )
            _trace_phase(
                "execute_conversation_tool_loop",
                phase_started,
                token_estimate=getattr(artifacts.manifest, "token_estimate", None),
                retrieved_count=len(getattr(artifacts.manifest, "retrieved", []) or []),
                history_count=len(getattr(artifacts.manifest, "history", []) or []),
            )
            assistant_meta_extra = (
                {"response_integrity": artifacts.integrity_review}
                if artifacts.integrity_review
                else {}
            )
            phase_started = time.perf_counter()
            followthrough = await prepare_project_followthrough_response(
                self,
                session_id=sid,
                user_input=user_input,
                assistant_content=artifacts.content,
                manifest=artifacts.manifest,
            )
            if followthrough is not None:
                artifacts.content = followthrough.content
                assistant_meta_extra["project_followthrough"] = followthrough.meta
            _trace_phase(
                "project_followthrough_prepare",
                phase_started,
                captured=bool(followthrough),
                response_replaced=(
                    bool(followthrough.meta.get("response_replaced"))
                    if followthrough is not None
                    else False
                ),
            )
            phase_started = time.perf_counter()
            await persist_tool_loop_messages(
                self,
                session_id=sid,
                artifacts=artifacts,
            )
            _trace_phase("persist_tool_loop_messages", phase_started)
            phase_started = time.perf_counter()
            await finalize_assistant_turn(
                self,
                session_id=sid,
                user_input=user_input,
                content=artifacts.content,
                manifest=artifacts.manifest,
                assistant_meta_extra=assistant_meta_extra or None,
                tool_use_inspections=artifacts.tool_use_inspections,
                tool_call_transits=artifacts.tool_call_transits,
                tool_chain_summary=artifacts.tool_chain_summary,
            )
            _trace_phase("finalize_assistant_turn", phase_started)
            if turn_marker_id:
                try:
                    phase_started = time.perf_counter()
                    complete_conversation_turn_marker(
                        self.ctx.config.state_dir,
                        turn_marker_id,
                        outcome="assistant_response_persisted",
                    )
                    _trace_phase("turn_marker_complete", phase_started, outcome="assistant_response_persisted")
                except Exception as exc:
                    self._trace(
                        "conversation_turn_marker_complete_error",
                        {"session_id": sid, "marker_id": turn_marker_id, "error": str(exc)},
                    )

            self._trace(
                "converse",
                {
                    "session_id": sid,
                    "input_len": len(user_input),
                    "token_estimate": artifacts.manifest.token_estimate,
                },
            )
            phase_started = time.perf_counter()
            voice_result = await maybe_speak_body_double_response(
                self,
                session_id=sid,
                user_input=user_input,
                user_meta=persisted_user_meta,
                response_text=artifacts.content,
            )
            _trace_phase(
                "body_double_voice_response",
                phase_started,
                status=voice_result.get("status") if isinstance(voice_result, dict) else None,
            )
            _trace_phase("conversation_total", turn_started)
            self.boredom.record_activity()
            if self.ctx.somatic is not None:
                try:
                    self.ctx.somatic.decay()
                except Exception:
                    pass
            return artifacts.content
        finally:
            await _capture_affective_registry_turn_end(
                self,
                session_id=sid,
                outcome="conversation_turn_finished",
            )
            self._set_activity("idle")

    async def run_daydream(
        self,
        *,
        force: bool = False,
        reflective_only: bool = False,
    ) -> Dict[str, Any]:
        """Generate daydreams when idle or tense."""
        return await run_runtime_daydream(
            self,
            force=force,
            reflective_only=reflective_only,
        )

    async def _run_daydream_inner(
        self,
        *,
        force: bool = False,
        reflective_only: bool = False,
    ) -> Dict[str, Any]:
        """Inner implementation of run_daydream (wrapped for activity tracking)."""
        return await run_runtime_daydream_inner(
            self,
            force=force,
            reflective_only=reflective_only,
        )

    async def run_cycle(self) -> Dict[str, Any]:
        """Run one creative/execution cycle."""
        self._set_activity("cycling")
        try:
            current_focus = current_runtime_focus(self, "cycle")
            await record_burst_continuity(
                self,
                trigger="cycle_burst_started",
                phase="start",
                intent=f"Cycle burst for {current_focus}",
                focus=current_focus,
                next_step="promote work, evaluate workspace, and drain the executive queue",
                note="creative cycle",
            )
        except Exception as exc:
            self._trace("continuity_breadcrumb_cycle_error", {"error": str(exc)})
        try:
            await self.executive.restore_queue()
            hydrated = 0
            hydrate_from_store = getattr(self.creative, "hydrate_from_store", None)
            if callable(hydrate_from_store):
                hydrated = await hydrate_from_store()
            result = self.creative.run_cycle()
            drain_creative_tasks = getattr(self.creative, "drain_background_tasks", None)
            if callable(drain_creative_tasks):
                await drain_creative_tasks()
            result["rehydrated"] = hydrated
            await self.cognition_bus.publish(
                LadderHealthSummary(
                    kind="ladder.health_summary",
                    source="agent_loop.run_cycle",
                    payload=dict(getattr(self.creative, "last_cycle_health", {}) or result),
                )
            )

            # Generate daydreams, evaluate reflections, and promote keepers to the
            # creative ladder so they can be enqueued in the same cycle.
            daydream_result = await self._run_daydream_inner()
            daydream_work_objects: list[WorkObject] = daydream_result.get(
                "daydream_work_objects", []
            )
            reflections: list[DaydreamReflection] = daydream_result.get(
                "reflections_list", []
            )

            # Keep run_cycle() focused on phase ordering; the helper module owns the
            # promotion/intervention/drain details and their local invariants.
            promoted_tasks = await enqueue_promoted_cycle_work(self)
            if callable(drain_creative_tasks):
                await drain_creative_tasks()

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
        finally:
            self._set_activity("idle")

    async def maybe_compact_session(
        self,
        session_id: str,
        tail_size: int = 10,
        min_removed_count: int = 1,
    ) -> Any:
        """Compact old episodes for a session if there are enough of them."""
        return await maybe_compact_runtime_session(
            self,
            session_id,
            tail_size=tail_size,
            min_removed_count=min_removed_count,
        )

    async def run_consolidation(self, *, budget: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the nightly consolidation engine."""
        return await run_runtime_consolidation(self, budget=budget)

    async def run_compaction_backlog_maintenance(
        self,
        *,
        max_sessions: int = 8,
        min_session_lag: int = 20,
        tail_size: int = 10,
        max_candidates: int = 1000,
    ) -> Dict[str, Any]:
        """Run bounded context compaction outside the nightly consolidation window."""
        self._set_activity("compacting")
        try:
            return await compact_runtime_backlog(
                self,
                max_sessions=max_sessions,
                min_session_lag=min_session_lag,
                tail_size=tail_size,
                max_candidates=max_candidates,
            )
        finally:
            self._set_activity("idle")

    def check_metacognition(self) -> Dict[str, Any]:
        """Run a metacognitive consistency check via ToM."""
        return build_runtime_metacognition_status(self)

    async def rebuild_identity(
        self,
        *,
        seed_episode_ids: Optional[List[str]] = None,
        min_created_at: Optional[datetime] = None,
        expand_graph: bool = True,
        term_limits: Optional[Dict[str, int]] = None,
        apply: bool = True,
    ) -> Dict[str, Any]:
        """Rebuild identity from autobiographical memory, optionally as a preview."""
        return await rebuild_runtime_identity(
            self,
            seed_episode_ids=seed_episode_ids,
            min_created_at=min_created_at,
            expand_graph=expand_graph,
            term_limits=term_limits,
            apply=apply,
        )

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

    async def execute_tool(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        audit_only: bool = False,
    ) -> Dict[str, Any]:
        """Execute a tool through the registry after self-approval."""
        return await execute_runtime_tool(
            self,
            name,
            args,
            session_id=session_id,
            task_id=task_id,
            audit_only=audit_only,
        )

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

    async def handle_action(
        self,
        request: ActionRequest,
        *,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate an action through the self-approval ladder."""
        return await handle_runtime_action(
            self,
            request,
            session_id=session_id,
            task_id=task_id,
            tool_name=tool_name,
            args=args,
        )

    async def _continuity_check(self) -> None:
        """Phase 9: Continuous Present — run at boot to decay score and generate monologue."""
        await run_runtime_continuity_check(self)

    async def run_autonomous(
        self,
        cycle_interval: int = 600,
        daydream_interval: int = 720,
        baa_heartbeat_interval: int = 120,
        consolidation_interval: int = 86400,
    ) -> None:
        """Start background scheduler and block until interrupted."""
        await run_autonomous_runtime(
            self,
            cycle_interval=cycle_interval,
            daydream_interval=daydream_interval,
            baa_heartbeat_interval=baa_heartbeat_interval,
            consolidation_interval=consolidation_interval,
        )

    async def run_autonomous_with_server(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        cycle_interval: int = 600,
        daydream_interval: int = 720,
        baa_heartbeat_interval: int = 120,
        consolidation_interval: int = 86400,
    ) -> None:
        """Run scheduler + FastAPI server together, shutting down gracefully on signal."""
        await run_autonomous_with_server_runtime(
            self,
            host=host,
            port=port,
            cycle_interval=cycle_interval,
            daydream_interval=daydream_interval,
            baa_heartbeat_interval=baa_heartbeat_interval,
            consolidation_interval=consolidation_interval,
        )

    async def import_bulma(
        self,
        bulma_state_dir: Path,
        checkpoint_path: Optional[Path] = None,
        curated_workspace_dir: Optional[Path] = None,
    ) -> Any:
        """Reject the retired OpenBulma v4 one-time importer."""
        raise RuntimeError(
            "The OpenBulma v4 importer has been decommissioned after the one-time "
            "Bulma cutover. Restore it from git history only for an explicit "
            "forensic migration task."
        )

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
    def _extract_self_commitments(
        text: str,
        *,
        user_input: str | None = None,
    ) -> List[SelfCommitmentCandidate]:
        """Extract normalized future-action self-commitments from assistant text."""
        return extract_runtime_self_commitments(text, user_input=user_input)

    async def _capture_self_commitments(
        self,
        content: str,
        session_id: str,
        *,
        user_input: str | None = None,
    ) -> List[Commitment]:
        """Persist normalized self-commitments and mirror them into ToM/somatic state."""
        return await capture_runtime_self_commitments(
            self,
            content,
            session_id,
            user_input=user_input,
        )

    async def _close_stores(self) -> None:
        """Gracefully close all SQLite stores."""
        await close_runtime_stores(self)

    async def run_wellbeing_maintenance(
        self,
        *,
        force_action: str | None = None,
    ) -> Dict[str, Any]:
        """Run a bounded operational wellbeing maintenance pass."""
        return await run_runtime_wellbeing_maintenance(self, force_action=force_action)

    async def run_cognitive_maintenance(self) -> Dict[str, Any]:
        """Run cognitive feedback-loop closure and learned-skill extraction."""
        return await run_runtime_cognitive_maintenance(self)

    async def run_shadow_registry_consumer(self, *, max_dismissals: int = 20) -> Dict[str, Any]:
        """Consume stale shadow clusters into downstream learning artifacts."""
        return await consume_shadow_registry(self, max_dismissals=max_dismissals)

    async def run_nightly_dream(
        self,
        *,
        mode: Any = "light",
        consolidation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the consolidation-adjacent nightly dreaming pass."""
        return await run_runtime_nightly_dream(
            self,
            mode=mode,
            consolidation_result=consolidation_result,
        )

    async def record_daydream_wellbeing(self, reflection: Any) -> Dict[str, Any]:
        """Record daydream thoughts into wellbeing/fascination state."""
        return await record_runtime_daydream_wellbeing(self, reflection)

    async def record_daydream_thread_beads(self, reflection: Any) -> Dict[str, Any]:
        """Record daydream thoughts into the peripheral thread registry."""
        return await record_runtime_daydream_thread_beads(self, reflection)

    async def backfill_daydream_signal_thread_beads(self, *, limit: int = 50) -> Dict[str, Any]:
        """Backfill already-routed daydream signals into thread-registry beads."""
        promotion = getattr(self, "daydream_promotion", None)
        if promotion is None or not hasattr(promotion, "backfill_signal_thread_beads"):
            return {"available": False, "scanned": 0, "recorded": 0, "skipped": 0}
        return await promotion.backfill_signal_thread_beads(limit=limit)

    async def record_shadow_registry_thread_beads(self, *, limit: int = 10) -> Dict[str, Any]:
        """Record mature ShadowRegistry clusters into peripheral thread beads."""
        return await record_shadow_registry_thread_beads(self, limit=limit)

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
        affect: Optional[Any] = None,
        payload: Optional[dict[str, Any]] = None,
        salience: Optional[float] = None,
    ) -> Episode:
        return await record_runtime_episode(
            self,
            content,
            kind,
            session_id=session_id,
            role=role,
            affect=affect,
            payload=payload,
            salience=salience,
        )

    async def _link_episode_to_previous(self, episode: Episode) -> None:
        """Create graph edges between this episode and the most recent prior episode."""
        from .episodic_runtime import link_runtime_episode_to_previous

        await link_runtime_episode_to_previous(self, episode)

    def _extract_content(self, response: Dict[str, Any]) -> str:
        return extract_runtime_response_content(response)

    def _trace(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        trace_runtime_event(self, event, payload)
