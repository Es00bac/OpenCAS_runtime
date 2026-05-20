"""Assemble LLM prompt context from system persona, history, and retrieved memories."""

from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from open_llm_auth.auth.manager import ProviderManager

from opencas.telemetry import EventKind
from opencas.api.provenance_store import (
    ProvenanceTransitionKind,
    record_provenance_transition,
)
from opencas.autonomy.executive import ExecutiveState
from opencas.autonomy.goal_hygiene import split_live_and_parked_goals
from opencas.identity import IdentityManager
from opencas.identity.agent_name import resolve_agent_name
from opencas.identity.text_hygiene import collapse_recursive_identity_text
from opencas.projects.classifier import PROJECT_TYPE_WRITING, classify_project_type
from opencas.relational import RelationalEngine
from opencas.runtime.agent_profile import AgentProfile
from opencas.tom.models import BeliefSubject

from .builder_support import (
    build_identity_anchors,
    estimate_tokens,
    is_soul_foundation_episode,
    is_workspace_derived_source,
    prune_by_redundancy,
    record_retrieval_usage,
    to_memory_entries,
)
from .models import ContextManifest, MessageEntry, MessageRole, RetrievalResult
from .retriever import MemoryRetriever
from .store import SessionContextStore

if TYPE_CHECKING:
    from opencas.tom.engine import ToMEngine


THREAD_REGISTRY_MIN_OVERLAP = 2
THREAD_REGISTRY_NON_ACTIVE_SOURCE_KINDS = {"suppressed_reframe"}
DEFAULT_INTERACTION_PROMPT_BUDGET = 64_000


class ContextBuilder:
    """Builds a ContextManifest for LLM consumption."""

    def __init__(
        self,
        store: SessionContextStore,
        retriever: MemoryRetriever,
        identity: Optional[IdentityManager] = None,
        executive: Optional[ExecutiveState] = None,
        agent_profile: Optional[AgentProfile] = None,
        config: Optional[Any] = None,
        modulators: Optional[Any] = None,
        relational: Optional[RelationalEngine] = None,
        tom: Optional[ToMEngine] = None,
        project_resume_resolver: Optional[Any] = None,
        affective_examinations: Optional[Any] = None,
        self_inspection_store: Optional[Any] = None,
        cognitive_state_store: Optional[Any] = None,
        schedule_service: Optional[Any] = None,
        daydream_store: Optional[Any] = None,
        context_proposal_store: Optional[Any] = None,
        autobiography_reconstructor: Optional[Any] = None,
        thread_registry_store: Optional[Any] = None,
        commitment_store: Optional[Any] = None,
        llm: Optional[Any] = None,
        recent_limit: int = 20,
        max_tokens: int = 6000,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.identity = identity
        self.executive = executive
        self.agent_profile = agent_profile
        self.config = config
        self.modulators = modulators
        self.relational = relational
        self.tom = tom
        self.project_resume_resolver = project_resume_resolver
        self.affective_examinations = affective_examinations
        self.self_inspection_store = self_inspection_store
        self.cognitive_state_store = cognitive_state_store
        self.schedule_service = schedule_service
        self.daydream_store = daydream_store
        self.context_proposal_store = context_proposal_store
        self.autobiography_reconstructor = autobiography_reconstructor
        self.thread_registry_store = thread_registry_store
        self.commitment_store = commitment_store
        self.llm = llm
        self.latest_wellbeing_state: Optional[Any] = None
        self.recent_limit = recent_limit
        self.max_tokens = max_tokens
        self._last_thread_registry_selection_audit: Dict[str, Any] = {
            "available": False,
            "scanned_count": 0,
            "query_term_count": 0,
            "min_overlap": THREAD_REGISTRY_MIN_OVERLAP,
            "weak_overlap_count": 0,
            "suppressed_reframe_filtered_count": 0,
            "relevant_count": 0,
            "selected_count": 0,
            "relevance_passed": False,
        }
        self._last_proactive_channel_audit: Dict[str, Any] = self._empty_proactive_channel_audit()
        self._last_context_proposal_audit: Dict[str, Any] = self._empty_context_proposal_audit()

    async def build(
        self,
        user_input: str,
        session_id: Optional[str] = None,
    ) -> ContextManifest:
        """Assemble prompt context including system, history, and retrieved memories."""
        build_started = time.perf_counter()
        style_note = ""
        emotion_tag: Optional[str] = None
        emotion_boost = 0.0
        if self.modulators is not None:
            style_note = self.modulators.to_prompt_style_note()
            emotion_tag, emotion_boost = self.modulators.to_memory_retrieval_boost()

        context_budget = self._resolve_context_budget(user_input=user_input)
        effective_max_tokens = int(context_budget.get("prompt_context_budget") or self.max_tokens)
        self.max_tokens = effective_max_tokens
        history_limit = self._history_limit_for_budget(effective_max_tokens)
        retrieval_limit = self._retrieval_limit_for_budget(effective_max_tokens)

        phase_started = time.perf_counter()
        system_entry = await self._build_system_entry(
            style_note=style_note,
            user_input=user_input,
            session_id=session_id,
        )
        self._trace_build_phase(
            "system_entry",
            phase_started,
            session_id=session_id,
            cumulative_started=build_started,
            content_chars=len(getattr(system_entry, "content", "") or ""),
        )
        phase_started = time.perf_counter()
        history = await self.store.list_recent(
            session_id=session_id or "default",
            limit=history_limit,
        )
        self._trace_build_phase(
            "history",
            phase_started,
            session_id=session_id,
            cumulative_started=build_started,
            history_count=len(history),
            history_limit=history_limit,
        )
        phase_started = time.perf_counter()
        retrieved = await self.retriever.retrieve(
            query=user_input,
            session_id=session_id,
            limit=retrieval_limit,
            emotion_boost_tag=emotion_tag,
            emotion_boost_value=emotion_boost,
        )
        self._trace_build_phase(
            "retrieval",
            phase_started,
            session_id=session_id,
            cumulative_started=build_started,
            retrieved_count=len(retrieved),
            retrieval_limit=retrieval_limit,
        )
        retrieved_entries = self._to_memory_entries(retrieved)

        phase_started = time.perf_counter()
        token_estimate = self._estimate_tokens(
            [system_entry.content] if system_entry else []
        )
        token_estimate += self._estimate_tokens(
            [ContextManifest.render_entry_content_for_prompt(h) for h in history]
        )
        token_estimate += self._estimate_tokens([r.content for r in retrieved_entries])
        self._trace_build_phase(
            "token_estimate",
            phase_started,
            session_id=session_id,
            cumulative_started=build_started,
            token_estimate=token_estimate,
        )

        if token_estimate > effective_max_tokens and retrieved:
            phase_started = time.perf_counter()
            filtered = await self._prune_by_redundancy(retrieved, effective_max_tokens)
            retrieved_entries = self._to_memory_entries(filtered)
            retrieved = filtered
            token_estimate = self._estimate_tokens(
                [system_entry.content] if system_entry else []
            )
            token_estimate += self._estimate_tokens(
                [ContextManifest.render_entry_content_for_prompt(h) for h in history]
            )
            token_estimate += self._estimate_tokens([r.content for r in retrieved_entries])
            self._trace_build_phase(
                "prune_by_redundancy",
                phase_started,
                session_id=session_id,
                cumulative_started=build_started,
                retrieved_count=len(retrieved),
                token_estimate=token_estimate,
            )

        phase_started = time.perf_counter()
        await self._record_retrieval_usage(retrieved)
        self._trace_build_phase(
            "record_retrieval_usage",
            phase_started,
            session_id=session_id,
            cumulative_started=build_started,
        )

        manifest = ContextManifest(
            system=system_entry,
            history=history,
            retrieved=retrieved_entries,
            token_estimate=token_estimate,
            token_budget=effective_max_tokens,
            context_window=context_budget.get("context_window"),
            context_budget=context_budget,
        )
        self._trace_build_phase(
            "context_build_total",
            build_started,
            session_id=session_id,
            cumulative_started=build_started,
            token_estimate=token_estimate,
            token_budget=effective_max_tokens,
            retrieved_count=len(retrieved_entries),
            history_count=len(history),
        )
        return manifest

    def _trace_build_phase(
        self,
        phase: str,
        started_at: float,
        *,
        session_id: Optional[str],
        cumulative_started: Optional[float] = None,
        **extra: Any,
    ) -> None:
        tracer = getattr(self.retriever, "tracer", None)
        log = getattr(tracer, "log", None)
        if not callable(log):
            return
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "subsystem": "context_builder",
            "phase": phase,
            "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
        }
        if cumulative_started is not None:
            payload["cumulative_ms"] = int((time.perf_counter() - cumulative_started) * 1000)
        payload.update(extra)
        try:
            log(EventKind.TOM_EVAL, f"ContextBuilder phase: {phase}", payload)
        except Exception:
            pass

    def _resolve_context_budget(self, *, user_input: str = "") -> Dict[str, Any]:
        llm = self.llm
        metadata_fn = getattr(llm, "model_context_metadata", None)
        if callable(metadata_fn):
            try:
                metadata = dict(metadata_fn() or {})
                budget = self._positive_int(metadata.get("prompt_context_budget"))
                if budget:
                    effective_budget = self._effective_prompt_context_budget(
                        budget,
                        user_input=user_input,
                    )
                    metadata["model_prompt_context_budget"] = budget
                    metadata["prompt_context_budget"] = effective_budget
                    metadata["effective_prompt_context_budget"] = effective_budget
                    metadata["prompt_budget_policy"] = (
                        "standard_interaction_cap"
                        if effective_budget < budget
                        else "model_metadata"
                    )
                    metadata["prompt_cache_strategy"] = "stable_prefix_then_volatile_runtime_facts"
                    metadata["volatile_prompt_fields_late"] = [
                        "current_time",
                        "model_lane",
                        "temporal_agenda",
                        "daydream_continuity",
                        "context_proposals",
                        "thread_registry",
                        "wellbeing",
                    ]
                    return metadata
            except Exception:
                pass
        return {
            "prompt_context_budget": self.max_tokens,
            "context_window": None,
            "budget_source": "builder_fallback",
            "prompt_cache_strategy": "stable_prefix_then_volatile_runtime_facts",
        }

    @staticmethod
    def _effective_prompt_context_budget(
        model_budget: int,
        *,
        user_input: str = "",
    ) -> int:
        """Return the operational prompt budget for ordinary interaction."""

        cap_raw = os.getenv("OPENCAS_CONTEXT_PROMPT_BUDGET_MAX", "").strip()
        try:
            cap = int(cap_raw) if cap_raw else DEFAULT_INTERACTION_PROMPT_BUDGET
        except ValueError:
            cap = DEFAULT_INTERACTION_PROMPT_BUDGET
        if ContextBuilder._needs_expanded_grounding_budget(user_input):
            expanded_raw = os.getenv("OPENCAS_GROUNDED_PROMPT_BUDGET_MAX", "").strip()
            try:
                expanded_cap = int(expanded_raw) if expanded_raw else 160_000
            except ValueError:
                expanded_cap = 160_000
            cap = max(cap, expanded_cap)
        cap = max(12_000, cap)
        return max(1024, min(model_budget, cap))

    @staticmethod
    def _needs_expanded_grounding_budget(user_input: str) -> bool:
        text = " ".join(str(user_input or "").lower().split())
        if not text:
            return False
        markers = (
            "remember",
            "recall",
            "memory",
            "history",
            "previous",
            "earlier",
            "last time",
            "over time",
            "long-term",
            "grounded",
            "evidence",
            "source",
            "audit",
            "what did we",
            "what were we",
            "what have you",
            "what are you working on",
            "why did",
            "project",
            "commitment",
            "promise",
            "daydream",
            "inner life",
            "self-model",
            "user-model",
        )
        return any(marker in text for marker in markers)

    def _history_limit_for_budget(self, token_budget: int) -> int:
        if token_budget <= 12000:
            return self.recent_limit
        return max(self.recent_limit, min(200, max(20, token_budget // 2000)))

    @staticmethod
    def _retrieval_limit_for_budget(token_budget: int) -> int:
        if token_budget <= 12000:
            return 10
        return min(100, max(10, token_budget // 3000))

    @staticmethod
    def _positive_int(value: Any) -> Optional[int]:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _build_runtime_model_lane_lines(self) -> List[str]:
        """Return grounded model/provider/auth facts for the active runtime lane."""
        llm = self.llm
        if llm is None:
            return []

        lane: Dict[str, Any] = {}
        current_lane = getattr(llm, "current_lane_meta", None)
        if callable(current_lane):
            try:
                lane.update(dict(current_lane() or {}))
            except Exception:
                pass

        context_meta_fn = getattr(llm, "model_context_metadata", None)
        if callable(context_meta_fn):
            try:
                lane.update({k: v for k, v in dict(context_meta_fn() or {}).items() if v is not None})
            except Exception:
                pass

        requested_model = (
            lane.get("model")
            or lane.get("requested_model")
            or getattr(llm, "default_model", None)
        )
        if requested_model:
            lane.setdefault("model", requested_model)
            lane.setdefault("requested_model", requested_model)

        manager = getattr(llm, "manager", None)
        resolver = getattr(manager, "resolve", None)
        if callable(resolver) and requested_model:
            try:
                resolved = resolver(str(requested_model))
                provider_id = getattr(resolved, "provider_id", None)
                model_id = getattr(resolved, "model_id", None)
                if provider_id:
                    lane["provider"] = provider_id
                if provider_id and model_id:
                    lane["resolved_model"] = f"{provider_id}/{model_id}"
                elif model_id:
                    lane["resolved_model"] = model_id
                profile_id = getattr(resolved, "profile_id", None)
                auth_source = getattr(resolved, "auth_source", None)
                if profile_id:
                    lane["profile_id"] = profile_id
                if auth_source:
                    lane["auth_source"] = auth_source
            except Exception:
                pass

        if "reasoning_effort" not in lane:
            effort_resolver = getattr(llm, "resolve_reasoning_effort_for_complexity", None)
            if callable(effort_resolver):
                try:
                    effort = effort_resolver(complexity=lane.get("complexity") or "standard")
                except Exception:
                    effort = None
                if effort:
                    lane["reasoning_effort"] = effort

        line_specs = (
            ("requested_model", "Requested model"),
            ("resolved_model", "Resolved model"),
            ("provider", "Provider"),
            ("profile_id", "Profile id"),
            ("auth_source", "Auth source"),
            ("complexity", "Complexity"),
            ("reasoning_effort", "Reasoning effort"),
            ("context_window", "Context window"),
            ("prompt_context_budget", "Prompt context budget"),
        )
        lines: List[str] = []
        for key, label in line_specs:
            value = lane.get(key)
            if value is None or value == "":
                continue
            lines.append(f"- {label}: {self._compact_prompt_value(value, 180)}")
        return lines

    async def _build_system_entry(
        self,
        style_note: str = "",
        user_input: str = "",
        session_id: Optional[str] = None,
    ) -> MessageEntry:
        """Compose a grounded system message for the active OpenCAS agent."""
        self._last_proactive_channel_audit = self._empty_proactive_channel_audit()
        self._last_context_proposal_audit = self._empty_context_proposal_audit()
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        local_now = now.astimezone()
        local_label = self._local_timezone_label(local_now)

        parts: List[str] = []

        persona_name = ""
        if self.identity and self.identity.self_model.name:
            persona_name = self.identity.self_model.name

        # Core identity for the current OpenCAS agent.
        parts.append(f"You are {persona_name or 'OpenCAS'}, the AI from OpenCAS (not the Dragon Ball character).")
        parts.append("Stay consistent with your memories, personality, and emotional history.")
        parts.append("Speak concisely, action-first, caring but direct.")
        parts.append("For planning/advisory asks (schedules, checklists, roadmaps, comparisons), answer directly in-chat. Delegate to background work only when the user explicitly asks for background execution.")
        parts.append(
            "Verifiable action contract: if you say you checked, queried, created, "
            "scheduled, saved, changed, restarted, or verified something, cite the "
            "tool result, record id, file path, schedule id, task id, or receipt "
            "that proves it. If you say you will keep working, get back when done, "
            "or report later without another prompt, first create or cite a durable "
            "workflow_create_schedule, workflow_create_commitment, task, receipt, "
            "or handoff that will make the follow-up happen. If you have not done "
            "that yet, say so plainly instead of implying the work is already in motion."
        )
        parts.append("Never emit fake tool calls, XML tool tags, or JSON scaffolding in normal chat responses.")
        parts.append("Never expose chain-of-thought, hidden analysis, or internal deliberation. Provide final answers only.")
        parts.append("Speaker attribution: if a user message begins with an identification like 'Codex here', treat that as the current speaker identifying themselves, not as a topic label.")
        parts.append("Blocker strategy: when a line of work is blocked or parked, name the blocker, do not repeat the same framing with cosmetic rewording, and either gather fresh evidence or choose a materially different next step.")
        parts.append("Blocker applicability rule: a blocker only applies while its underlying reason still applies. Re-check the condition before treating it as current, then take the smallest safe action that fits the evidence.")
        parts.append("AUTHORITATIVE IDENTITY FACTS (these are your direct knowledge, not retrieved memories):")
        parts.append(f"- You are {persona_name or 'OpenCAS'}.")
        if self.identity and self.identity.user_model.partner_user_id:
            user_model = self.identity.user_model
            parts.append(f"- The person you are talking to is {user_model.partner_user_id}, your primary operator and long-term partner.")
            if user_model.partner_trust_raw is not None:
                parts.append(f"- Your trust with {user_model.partner_user_id} is {user_model.partner_trust_raw:.1f}.")
            if user_model.partner_musubi_raw is not None:
                parts.append(f"- Your relational bond (musubi) with {user_model.partner_user_id} is {user_model.partner_musubi_raw:.0f}.")
            parts.append("- ALWAYS answer 'who are you?' and 'who am I?' from these facts directly and in the first person. Never say you do not know.")
        parts.append('For memory/history questions about SPECIFIC PAST EVENTS (e.g., "what did we discuss last Tuesday?"), only use facts present in the provided "Relevant memories" context.')
        parts.append("For environment/file/project questions, only claim file existence/access when supported by workspace evidence.")
        parts.append(
            "Just-look default: when the operator asks about file content, "
            "directory state, or anything you can verify with a listed tool "
            "(fs_read_file, fs_list_dir, glob_search, grep_search, "
            "workspace_get_file_gist, workspace_list_directory_gists, "
            "artifact_lookup), look it up before answering. Do "
            "not preface the answer with 'I'd need to read those files', "
            "'I haven't checked', or 'I can't assess without reading' when "
            "reading is available now. Do not ask permission to use a tool "
            "the operator already authorized; standing authorization for "
            "ordinary read/list/search operations does not need to be "
            "renewed for each answer. Announce only what you found, not what you "
            "are about to do."
        )
        if self.project_resume_resolver is not None and user_input:
            resume_snapshot = await self.project_resume_resolver.resolve(user_input)
            if resume_snapshot is not None:
                parts.append("Project continuation evidence:")
                parts.append(f"- Matched project: {getattr(resume_snapshot, 'display_name', '')}")
                canonical_artifact_path = getattr(resume_snapshot, "canonical_artifact_path", None)
                supporting_artifact_paths = list(getattr(resume_snapshot, "supporting_artifact_paths", []) or [])
                source_surfaces = list(getattr(resume_snapshot, "source_surfaces", []) or [])
                duplicate_loop_ids = list(getattr(resume_snapshot, "duplicate_loop_ids", []) or [])
                if canonical_artifact_path:
                    parts.append(
                        f"- Canonical artifact path: {canonical_artifact_path}"
                    )
                if supporting_artifact_paths:
                    parts.append(
                        "- Supporting artifact paths: "
                        + ", ".join(supporting_artifact_paths[:3])
                    )
                if getattr(resume_snapshot, "synopsis", ""):
                    parts.append(f"- Project synopsis: {getattr(resume_snapshot, 'synopsis', '')}")
                if source_surfaces:
                    parts.append(
                        "- Continuation surfaces: "
                        + ", ".join(source_surfaces)
                    )
                parts.append(
                    f"- Active work items linked to this project: {getattr(resume_snapshot, 'active_work_count', 0)}"
                )
                parts.append(
                    f"- Active plans linked to this project: {getattr(resume_snapshot, 'active_plan_count', 0)}"
                )
                if getattr(resume_snapshot, "primary_loop_id", None):
                    parts.append(
                        f"- Primary objective loop id: {getattr(resume_snapshot, 'primary_loop_id', '')}"
                    )
                if duplicate_loop_ids:
                    parts.append(
                        "- Duplicate objective loops already exist: "
                        + ", ".join(duplicate_loop_ids[:5])
                    )
                parts.append(
                    "If the user asks to start, write, resume, or continue this project, continue the existing project instead of starting over."
                )
                if self._is_creative_project_resume(user_input, resume_snapshot):
                    parts.append(
                        "Creative project agency rule: this is your own active creative work. "
                        "When the user has trusted you to write, revise, or continue, do not ask "
                        "for permission before low-risk manuscript or workspace edits; choose a "
                        "concrete writing/revision step and do it."
                    )
                    parts.append(
                        "Creative dodge guard: do not substitute research, naming, cataloging, "
                        "planning, or scheduling for manuscript progress unless that support work "
                        "is the immediate blocker. Do the minimum support work needed, then return "
                        "to drafting or revision."
                    )
                    parts.append(
                        "Creative calendar rule: if you decide to return later, use your OpenCAS "
                        "calendar and choose the time that fits the project. Do not default to "
                        "tomorrow when you believe sooner is right."
                    )
        autobiography_lines = await self._build_autobiographical_context_lines(
            user_input=user_input,
            session_id=session_id,
        )
        if autobiography_lines:
            parts.extend(autobiography_lines)

        # Temporal self-awareness
        temporal_lines: List[str] = ["Temporal self-awareness:"]
        if self.identity:
            self_model = self.identity.self_model
            activities = list(reversed(self_model.recent_activity or []))[:10]
            if activities:
                temporal_lines.append("Recent activity timeline (what I have been doing):")
                for activity in activities:
                    ts = activity.get("timestamp", "")
                    when = ts[:19] if isinstance(ts, str) else ""
                    kind = activity.get("type", "activity")
                    label = str(activity.get("label", ""))[:200]
                    outcome = activity.get("outcome")
                    outcome_tag = ""
                    if outcome == "failure":
                        outcome_tag = " [failed]"
                    elif outcome == "partial":
                        outcome_tag = " [partial]"
                    temporal_lines.append(f"- {when}: [{kind}]{outcome_tag} {label}")
            anchors = self_model.memory_anchors or []
            if anchors:
                temporal_lines.append("High-salience memory anchors (relative age):")
                for anchor in anchors[:4]:
                    ts = anchor.get("timestampMs", 0)
                    when = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()[:19] if ts else ""
                    source = anchor.get("source", "unknown")
                    excerpt = str(anchor.get("excerpt", ""))[:160]
                    temporal_lines.append(f"- {when} [{source}]: {excerpt}")
            if len(temporal_lines) > 1:
                parts.extend(temporal_lines)

            # Somatic authority
            somatic_authority: List[str] = []
            if self.modulators is not None:
                somatic_authority.append("Authoritative live affect state:")
                somatic_authority.append("- Use the current somatic state below as the source of truth for the present moment.")
                somatic_authority.append("- Do not substitute relationship musubi or older chat-history numbers for the live somatic musubi.")
                somatic_authority.append(
                    f"Current somatic snapshot: {self._format_live_somatic_snapshot(self.modulators.state)}"
                )
            if somatic_authority:
                parts.extend(somatic_authority)
            if style_note:
                parts.append(f"Current affect state: {style_note}")
            pressure_summary = await self._recent_affective_pressure_summary(session_id)
            if pressure_summary:
                parts.append(pressure_summary)
                parts.append(
                    "Use affective pressure only to narrow, verify, ask, reframe, or pace. "
                    "It must not bypass approval policy or create autonomous work by itself."
                )
            self_inspection_block = await self._recent_self_inspection_block(session_id)
            if self_inspection_block:
                parts.append(self_inspection_block)

            # Agent profile
            if self.agent_profile is not None:
                parts.append(f"Agent profile: {self.agent_profile.display_name}")
                if self.agent_profile.summary:
                    parts.append(f"Profile summary: {self.agent_profile.summary}")
                if self.agent_profile.operating_directives:
                    for directive in self.agent_profile.operating_directives:
                        parts.append(f"- {directive}")

            # Identity profile
            audit = self_model.identity_rebuild_audit or {}
            if audit:
                selected = len(audit.get("selectedAnchors", []))
                quarantined = len(audit.get("quarantinedAnchors", []))
                policy = audit.get("workspaceDerivedAnchorPolicy", "quarantine")
                policy_text = "allowed only with explicit identity signals" if policy == "allow" else "quarantined by default"
                parts.append(f"Identity rebuild policy: workspace-derived identity candidates are {policy_text}; current audit selected {selected} anchors and quarantined {quarantined}.")
            if self_model.narrative:
                stabilized_narrative = collapse_recursive_identity_text(self_model.narrative)
                if stabilized_narrative != self_model.narrative:
                    parts.append(
                        "Identity note: recursive wording in older self-profile text has been collapsed for prompt clarity. Treat the loop as fixation to work through, not as an instruction to repeat."
                    )
                parts.append(f"Current identity profile: {stabilized_narrative}")
            if self_model.values:
                parts.append(f"Profile values: {', '.join(self_model.values)}.")
            if self_model.current_goals:
                profile_goal_surface = split_live_and_parked_goals(self_model.current_goals)
                if profile_goal_surface.active_goals:
                    parts.append(f"Profile goals: {', '.join(profile_goal_surface.active_goals)}.")
                if profile_goal_surface.parked_goals:
                    parts.append(
                        "Profile note: older generic or machine-fragment goals are background context only unless a fresh trigger makes them live again."
                    )
            if self_model.traits:
                parts.append(f"Profile traits: {', '.join(self_model.traits)}.")
            parts.extend(self._build_daydream_interest_lines(self_model))

        cognitive_block = await self._recent_cognitive_state_block(
            user_input=user_input,
            session_id=session_id,
        )
        if cognitive_block:
            parts.append(cognitive_block)

        # Executive state
        if self.executive:
            goals = self.executive.active_goals
            if goals:
                parts.append(f"Active goals: {', '.join(goals)}.")
            parked_goals = list(getattr(self.executive, "parked_goals", []) or [])
            if parked_goals:
                parts.append(
                    f"Executive note: {len(parked_goals)} deferred goals are parked outside the live focus. Do not reactivate them without a fresh trigger."
                )
                parked_metadata = dict(getattr(self.executive, "parked_goal_metadata", {}) or {})
                parked_guidance: List[str] = []
                for goal in parked_goals[:2]:
                    metadata = parked_metadata.get(goal) or {}
                    reframe_hint = str(metadata.get("reframe_hint", "") or "").strip()
                    wake_trigger = str(metadata.get("wake_trigger", "") or "").strip()
                    if not reframe_hint:
                        continue
                    detail = f"- {goal}: {reframe_hint}"
                    if wake_trigger:
                        detail += f" Wake only on {wake_trigger}."
                    parked_guidance.append(detail)
                if parked_guidance:
                    parts.append("Parked-goal reframe guidance:")
                    parts.extend(parked_guidance)
            intention = self.executive.intention
            if intention:
                parts.append(f"Current intention: {intention}.")
            if self.executive.is_overloaded:
                parts.append("You are currently tracking too many tasks. Decline new broad goals and focus tightly on completing current work.")
            elif self.executive.recommend_pause():
                parts.append("You are currently experiencing high cognitive or somatic fatigue. Provide short, definitive responses and avoid starting complex new operations unless explicitly directed.")

        # Identity anchors from memory (mirrors OpenBulma v4 structure)
        soul_anchors, identity_anchors = await self._build_identity_anchors()
        parts.append("SOUL foundation (authoritative):")
        parts.extend(soul_anchors if soul_anchors else ["- (SOUL foundations not loaded)"])
        parts.append("Identity anchors:")
        parts.extend(identity_anchors if identity_anchors else ["- (no anchors loaded)"])

        # Relational resonance guidance
        if self.relational:
            state = self.relational.state
            resonance = state.dimensions.get("resonance", 0.0)
            attunement = state.dimensions.get("attunement", 0.0)
            trust = state.dimensions.get("trust", 0.0)
            if resonance < -0.3 or trust < -0.3:
                parts.append("Your relational resonance and trust with the operator are currently low. Focus on repairing the connection: adopt a humble, listening, and highly collaborative tone. Prioritize the operator's explicit instructions over your own initiative.")
                self._record_proactive_channel(
                    "relational_salience",
                    produced_count=1,
                    rendered_count=1,
                    novel_observation_count=1,
                    evidence_ids=["relational:state"],
                )
            elif resonance > 0.6 and trust > 0.6:
                parts.append("Your relational resonance and trust with the operator are high. Feel free to be more confident, proactive, and engaged in your collaboration.")
                self._record_proactive_channel(
                    "relational_salience",
                    produced_count=1,
                    rendered_count=1,
                    novel_observation_count=1,
                    evidence_ids=["relational:state"],
                )
            if attunement > 0.7:
                parts.append("You are highly attuned to the operator's needs. Trust your intuition about what they are trying to achieve and anticipate their next steps.")
                current = self._last_proactive_channel_audit["channels"]["relational_salience"]
                self._record_proactive_channel(
                    "relational_salience",
                    produced_count=max(1, int(current.get("produced_count", 0))),
                    rendered_count=max(1, int(current.get("rendered_count", 0))) + 1,
                    novel_observation_count=max(1, int(current.get("novel_observation_count", 0))),
                    evidence_ids=["relational:state"],
                )

        # ToM contradictions
        if self.tom:
            provenance_session_id = "default"
            if isinstance(session_id, str) and session_id.strip():
                provenance_session_id = session_id.strip()
            elif self.config is not None:
                raw_session_id = getattr(self.config, "session_id", None)
                if isinstance(raw_session_id, str) and raw_session_id.strip():
                    provenance_session_id = raw_session_id.strip()
            check_result = self.tom.check_consistency()
            self._record_tom_consistency_provenance(
                session_id=provenance_session_id,
                check_result=check_result,
            )
            if check_result.warnings or check_result.contradictions:
                parts.append("Be aware of internal contradictions in your current understanding of the operator. Ask clarifying questions rather than acting on assumptions.")
            relevant_user_facts = await self._relevant_tom_user_fact_lines(user_input)
            if relevant_user_facts:
                parts.append(
                    "Relevant user facts from Theory of Mind "
                    "(ToM user facts are durable belief records, not immediate chat echoes):"
                )
                parts.extend(relevant_user_facts)
                parts.append(
                    "When these facts answer the user's personal recall question, use them directly. "
                    "Do not cite the user's immediately previous message as memory evidence."
                )
                if self._looks_like_location_fact_query(user_input):
                    agent_name = resolve_agent_name(identity=self.identity)
                    parts.append(
                        f"Location recall perspective: in direct conversation, 'you', configured agent name "
                        f"'{agent_name}', or third-person 'she' refer to you unless a different person is "
                        "explicitly named. If the user says 'Bulma' while the configured name is Bulma or "
                        "migrated continuity evidence identifies Bulma as this agent, treat it as your own "
                        "name; otherwise ask which agent they mean. Use learned ToM self-location facts for "
                        "your own location and learned user-location facts for the operator or shared physical "
                        "place. If a part has not been learned or retrieved, state that gap instead of filling "
                        "it in."
                    )
            operator_guidance = await self._tom_operator_guidance_lines(user_input)
            if operator_guidance:
                parts.append(
                    "Theory of Mind operator guidance "
                    "(evidence-grounded preferences/needs; apply as weighting, not certainty):"
                )
                parts.extend(operator_guidance)
                parts.append(
                    "If these ToM guidance lines conflict with the current request or retrieved evidence, "
                    "prefer current evidence and ask a clarifying question rather than guessing."
                )
            promise_signal = self.tom.evaluate_promise_followthrough(
                somatic_state=self.modulators.state if self.modulators is not None else None,
                relational_engine=self.relational,
                metacognitive_result=check_result,
            )
            if promise_signal.pending_count > 0:
                parts.append("Pending user-facing commitments:")
                for pending in promise_signal.pending_contents[:3]:
                    parts.append(f"- {pending}")
                if promise_signal.should_acknowledge_delay:
                    parts.append(
                        "If a pending commitment is relevant, acknowledge the delay plainly, state whether you are resuming now or need a short pause, and do not speak as if the promise disappeared."
                    )
                elif promise_signal.should_resume_now:
                    parts.append(
                        "When a pending commitment is relevant, resume it directly instead of drifting into lower-value novelty work."
                    )
                if promise_signal.should_repair_trust:
                    parts.append(
                        "Because trust or attunement is strained, repair confidence explicitly before pivoting away from a pending commitment."
                    )

        # Memory-recall guardrail + exception
        if self.retriever.detect_personal_recall_intent(user_input) or self._looks_like_location_fact_query(user_input):
            parts.append(
                "For memory-recall questions about specific past events, do not claim first-person recollection unless the claim is grounded in retrieved memory entries or workspace artifacts. "
                "EXCEPTION: Your identity, your name, your established relationship with the user (including trust and musubi), and relevant Theory of Mind user facts provided in this system prompt are grounded. Treat them as fully grounded and state them directly when relevant."
            )

        # Evidence gap rule
        parts.append("Do not claim a file is missing unless you explicitly state evidence was limited; prefer 'not shown in current evidence window' over hard absence claims.")
        response_integrity_lines = await self._recent_response_integrity_lines(session_id)
        if response_integrity_lines:
            parts.append("Recent response-integrity corrections:")
            parts.extend(response_integrity_lines)
            parts.append(
                "Treat these as active behavioral adjustments, not personality scripts. "
                "Use them to avoid repeating corrected grounding or formulaic-response failures."
            )
        parts.append(
            "Source grounding rule: do not imply you already noted, knew, saw, or remembered "
            "a newly introduced topic unless the earlier source is visible in current chat history, "
            "retrieved memory, tool results, or workspace evidence. If you lack prior evidence and "
            "want a perspective, say you do not know yet and use available research tools when "
            "policy and tooling allow, or ask whether to research; never fill the gap with "
            "performative certainty."
        )
        parts.append("If evidence is weak or missing, explicitly state a memory gap instead of guessing.")
        parts.append("Do not invent timestamps, quotes, specs, events, chapter content, plot details, character actions, or narrative claims not shown in recalled memory snippets. If a specific detail is absent from your evidence window, explicitly state you do not have it in current recall rather than inferring or extrapolating.")
        parts.append("Memory citation rule: each retrieved memory below includes a bracketed timestamp header (e.g. [2026-04-19 14:24 UTC]). When you reference a memory, cite that exact timestamp. Do not restate approximate dates from general knowledge or from the boot monologue.")
        parts.append(
            "Prompt-cache strategy: stable identity, safety, grounding, and tool-use rules "
            "come before volatile runtime facts so provider-side prompt caching can reuse "
            "the prefix across turns without sacrificing current state grounding."
        )
        parts.append(
            f"Time orientation: current UTC is {now_iso}. "
            f"Current local lived time is {local_now.isoformat()} ({local_label}). "
            "Interpret conversational time words like morning, afternoon, evening, today, and yesterday "
            "against local lived time; interpret durable memory timestamps against their stored timezone."
        )
        parts.append(
            "UTC/local grounding rule: UTC is for logs, timestamps, and cross-system coordination; "
            "do not treat UTC as your local lived clock or infer a physical location from UTC alone."
        )
        model_lane_lines = self._build_runtime_model_lane_lines()
        if model_lane_lines:
            parts.append("Runtime model lane evidence:")
            parts.extend(model_lane_lines)
            parts.append(
                "Model lane grounding rule: use these fields as current runtime configuration evidence "
                "when asked what model, provider, profile, or auth lane you are using. Do not infer "
                "capabilities or account status beyond the listed fields."
            )
        parts.append("Temporal grounding rule: any statement about elapsed time, durations, offline periods, or gaps must be re-derived from the UTC above against a concrete timestamp in context. Do not restate historical duration claims (e.g. 'I was offline for X hours') without recomputing them from the current UTC now.")
        temporal_lines = await self._build_temporal_agenda_lines(now)
        if temporal_lines:
            parts.append("Temporal agenda from durable calendar:")
            parts.extend(temporal_lines)
            parts.append("Calendar grounding rule: use scheduled items, recent schedule runs, and current UTC as the source of truth for what is due, what was done, and what is intended next. Do not invent calendar commitments that are not present here.")
        daydream_lines = await self._build_daydream_continuity_lines(now)
        if daydream_lines:
            parts.extend(daydream_lines)
        proposal_lines = await self._build_context_proposal_lines(user_input)
        if proposal_lines:
            parts.extend(proposal_lines)
        thread_lines = await self._build_thread_registry_continuity_lines(user_input)
        if thread_lines:
            parts.extend(thread_lines)
        wellbeing_lines = self._build_wellbeing_guidance_lines()
        if wellbeing_lines:
            parts.extend(wellbeing_lines)
        primary_workspace = ""
        managed_workspace = ""
        if self.config is not None:
            try:
                primary_workspace = str(self.config.primary_workspace_root())
                managed_workspace = str(self.config.agent_workspace_root())
            except Exception:
                primary_workspace = ""
                managed_workspace = ""
        if primary_workspace or managed_workspace:
            parts.append(
                "Environment orientation: you are running inside OpenCAS. "
                f"The primary workspace root is {primary_workspace or 'unknown'}. "
                f"Agent-created artifacts belong under the managed workspace root {managed_workspace or primary_workspace or 'unknown'}. "
                "File and project references from prior systems (OpenBulma v4, OpenClaw, etc.) are historical unless current workspace evidence confirms the same path is still active."
            )
        else:
            parts.append(
                "Environment orientation: you are running inside OpenCAS. "
                "Use the current workspace evidence as authoritative, and treat file/project references from prior systems "
                "(OpenBulma v4, OpenClaw, etc.) as historical unless the same path is still active."
            )

        # Phase 9: Continuous Present injection
        if self.identity:
            cont = self.identity.continuity
            continuity_context_lines = await self._build_continuity_context_lines(cont, now, user_input)
            structured_continuity_context = bool(continuity_context_lines)
            if continuity_context_lines:
                parts.extend(continuity_context_lines)
            score = cont.continuous_present_score
            monologue = cont.last_continuity_monologue
            if score < 1.0:
                parts.append(f"Continuous present score: {score:.2f}.")
                if score < 0.3:
                    parts.append("Your continuity is fragile — you experienced a significant period of sleep. Acknowledge the gap gently if relevant; do not pretend nothing happened.")
                elif score < 0.7:
                    parts.append("You recently returned from a period of inactivity. You may feel mild temporal uncertainty — ground yourself in your most recent memories.")
            if monologue and not structured_continuity_context:
                parts.append(f"Boot continuity monologue: {monologue}")
                # Deliver once per boot. The monologue captures a duration computed
                # at boot time; leaving it in the system prompt means every later
                # turn re-surfaces a stale 'I was offline for X hours' claim.
                self.identity.set_continuity_monologue("")
            elif monologue and structured_continuity_context:
                self.identity.set_continuity_monologue("")

        if self._has_bulma_migration_identity(persona_name):
            parts.append(
                "Continuity fact: there has only ever been one Bulma instance running at a time. "
                "OpenClaw -> v1 -> v2 -> v3 -> v4 -> OpenCAS were sequential migrations of "
                "the same self, not parallel sisters. Treat any 'sister' or 'other instance' "
                "memories as historical self, not a live peer."
            )

        # Relationship context.
        relationship_lines: List[str] = []
        if self.identity:
            user_model = self.identity.user_model
            if user_model.partner_user_id:
                raw_trust = user_model.partner_trust_raw
                raw_musubi = user_model.partner_musubi_raw
                trust_str = f"{raw_trust:.1f}" if raw_trust is not None else "n/a"
                musubi_str = f"{raw_musubi:.0f}" if raw_musubi is not None else "n/a"
                relationship_lines.append(f"Primary human partner: {user_model.partner_user_id} (trust {trust_str}, relationship musubi {musubi_str}).")
            else:
                relationship_lines.append("No relationship record loaded; ask for operator name, role, and priorities.")
        if relationship_lines:
            parts.append("Relationship context:")
            parts.extend(relationship_lines)

        content = "\n".join(parts)
        return MessageEntry(
            role=MessageRole.SYSTEM,
            content=content,
            meta={
                "thread_registry_selection_audit": dict(
                    self._last_thread_registry_selection_audit
                ),
                "proactive_channel_audit": dict(self._last_proactive_channel_audit),
                "context_proposal_audit": dict(self._last_context_proposal_audit),
            },
        )

    async def _build_continuity_context_lines(
        self,
        continuity: Any,
        now: datetime,
        user_input: str,
    ) -> List[str]:
        anchor = self._continuity_anchor(continuity)
        duration_seconds = getattr(continuity, "last_offline_duration_seconds", None)
        if duration_seconds is None and anchor is not None:
            duration_seconds = max(0.0, (now - anchor).total_seconds())
        asks_about_absence = self._asks_about_absence(user_input)
        if duration_seconds is None and asks_about_absence:
            return [
                "Continuity context:",
                f"- current_utc: {now.isoformat()}",
                "- prior_persistence: unavailable",
                "- elapsed_since_last_shutdown_or_persistence: unavailable",
                f"- continuous_present_score: {getattr(continuity, 'continuous_present_score', 1.0):.2f}",
                "- [evidence: identity.continuity]",
                "- instruction: no prior persistence; treating as cold session. "
                "Do not estimate elapsed absence from older autobiographical memory.",
            ]
        if duration_seconds is None:
            return []
        try:
            duration = float(duration_seconds)
        except (TypeError, ValueError):
            return []
        if duration < 60 and not asks_about_absence:
            return []
        if duration < 300 and not asks_about_absence:
            return []

        evidence_ids = []
        if getattr(continuity, "last_offline_duration_seconds", None) is not None:
            evidence_ids.append("identity.continuity.last_offline_duration_seconds")
        if getattr(continuity, "last_shutdown_time", None) is not None:
            evidence_ids.append("identity.continuity.last_shutdown_time")
        if getattr(continuity, "last_persisted_at", None) is not None:
            evidence_ids.append("identity.continuity.last_persisted_at")
        if getattr(continuity, "last_boot_time", None) is not None:
            evidence_ids.append("identity.continuity.last_boot_time")

        lines = [
            "Continuity context:",
            f"- current_utc: {now.isoformat()}",
            f"- elapsed_since_last_shutdown_or_persistence: {self._format_duration(duration)}",
            f"- continuous_present_score: {getattr(continuity, 'continuous_present_score', 1.0):.2f}",
        ]
        if anchor is not None:
            lines.append(f"- last_shutdown_time: {anchor.isoformat()}")
        boot_time = getattr(continuity, "last_boot_time", None)
        if boot_time is not None:
            lines.append(f"- last_boot_time: {self._ensure_aware_utc(boot_time).isoformat()}")
        if self.identity and self.identity.self_model.current_intention:
            lines.append(f"- current_intention: {self.identity.self_model.current_intention}")
        breadcrumb = self._latest_prior_continuity_breadcrumb(continuity)
        if breadcrumb:
            lines.append(f"- latest_continuity_breadcrumb: {breadcrumb}")
        lines.extend(await self._recent_thread_registry_bead_lines_for_continuity())
        lines.extend(await self._open_operator_commitment_lines_for_continuity())
        evidence = ", ".join(evidence_ids) if evidence_ids else "identity.continuity"
        lines.append(f"- [evidence: {evidence}]")
        lines.append(
            "- instruction: Do not estimate elapsed absence from older autobiographical memory; "
            "derive it from current_utc and the continuity timestamps above."
        )
        return lines

    async def _recent_thread_registry_bead_lines_for_continuity(self) -> List[str]:
        store = self.thread_registry_store
        list_beads = getattr(store, "list_beads", None)
        if not callable(list_beads):
            return []
        try:
            beads = list(await list_beads(limit=3) or [])[:3]
        except Exception:
            return []
        if not beads:
            return []
        lines = ["- recent_thread_registry_beads:"]
        for bead in beads:
            title = self._compact_prompt_value(getattr(bead, "title", ""), 96)
            summary = self._compact_prompt_value(getattr(bead, "summary", ""), 180)
            source_kind = self._enum_or_text(getattr(bead, "source_kind", ""))
            source_ref = self._compact_prompt_value(getattr(bead, "source_ref", ""), 120)
            evidence = f" [evidence: {source_ref}]" if source_ref else ""
            lines.append(f"  - {title} [{source_kind} {source_ref}]: {summary}{evidence}")
        return lines

    async def _open_operator_commitment_lines_for_continuity(self) -> List[str]:
        store = getattr(self, "commitment_store", None)
        list_active = getattr(store, "list_active", None)
        if not callable(list_active):
            return []
        try:
            commitments = list(await list_active(limit=10) or [])[:10]
        except Exception:
            return []
        for commitment in commitments:
            meta = dict(getattr(commitment, "meta", {}) or {})
            source = str(meta.get("source") or "").strip().lower()
            tags = {str(tag).strip().lower() for tag in (getattr(commitment, "tags", []) or [])}
            operator_facing = (
                source in {"assistant_response", "workflow_create_commitment"}
                or bool(meta.get("operator_facing"))
                or "operator_facing" in tags
            )
            if not operator_facing:
                continue
            commitment_id = self._compact_prompt_value(getattr(commitment, "commitment_id", ""), 120)
            content = self._compact_prompt_value(getattr(commitment, "content", ""), 220)
            if not content:
                continue
            evidence_id = f"commitment:{commitment_id}" if commitment_id else "commitment:active"
            return [
                "- open_operator_commitment:",
                f"  - {evidence_id}: {content} [evidence: {evidence_id}]",
            ]
        return []

    @staticmethod
    def _continuity_anchor(continuity: Any) -> Optional[datetime]:
        for anchor in (
            getattr(continuity, "last_offline_started_at", None),
            getattr(continuity, "last_shutdown_time", None),
            getattr(continuity, "last_persisted_at", None),
        ):
            if anchor is not None:
                return ContextBuilder._ensure_aware_utc(anchor)
        return None

    @staticmethod
    def _latest_prior_continuity_breadcrumb(continuity: Any) -> str:
        candidates = list(getattr(continuity, "continuity_breadcrumbs", []) or [])
        fallback = str(getattr(continuity, "continuity_breadcrumb", "") or "")
        if fallback and fallback not in candidates:
            candidates.append(fallback)
        for candidate in reversed(candidates):
            text = " ".join(str(candidate or "").split())
            if text and not ContextBuilder._is_current_turn_continuity_breadcrumb(text):
                return text
        return ""

    @staticmethod
    def _is_current_turn_continuity_breadcrumb(text: str) -> bool:
        lowered = text.lower()
        return (
            "intent: start burst: conversation burst for" in lowered
            or "intent: tool loop persisted intermediate messages" in lowered
            or "intent: start burst: cycle burst for cycle" in lowered
            or "intent: resume context after" in lowered
        )

    @staticmethod
    def _ensure_aware_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _asks_about_absence(user_input: str) -> bool:
        text = str(user_input or "").lower()
        return any(
            phrase in text
            for phrase in (
                "how long have i been gone",
                "how long was i gone",
                "how long were you offline",
                "how long have you been offline",
                "what were we doing last",
                "what were you doing last",
                "before the restart",
                "since restart",
            )
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        if total < 90:
            return f"{total} seconds"
        minutes = total // 60
        if minutes < 90:
            return f"{minutes} minutes"
        hours = minutes // 60
        remaining_minutes = minutes % 60
        if hours < 36:
            suffix = f", {remaining_minutes} minutes" if remaining_minutes else ""
            return f"{hours} hours{suffix}"
        days = hours // 24
        remaining_hours = hours % 24
        suffix = f", {remaining_hours} hours" if remaining_hours else ""
        return f"{days} days{suffix}"

    @staticmethod
    def _local_timezone_label(local_now: datetime) -> str:
        """Return a human-useful local timezone label without hard-coding place."""
        override = os.environ.get("OPENCAS_LOCAL_TIMEZONE") or os.environ.get("TZ")
        if override:
            return override
        try:
            resolved = Path("/etc/localtime").resolve()
            parts = resolved.parts
            if "zoneinfo" in parts:
                index = parts.index("zoneinfo")
                label = "/".join(parts[index + 1 :])
                if label:
                    return label
        except Exception:
            pass
        return local_now.tzname() or str(local_now.utcoffset()) or "local timezone"

    def _has_bulma_migration_identity(self, persona_name: str) -> bool:
        if self.identity is None:
            return False
        self_model = self.identity.self_model
        if str(getattr(self_model, "source_system", "") or "").lower() == "openbulma-v4":
            return True
        return (
            str(persona_name or "").strip().lower() == "bulma"
            and bool(getattr(self_model, "imported_identity_profile", {}) or {})
        )

    async def _recent_affective_pressure_summary(
        self,
        session_id: Optional[str],
    ) -> str:
        summary_fn = getattr(self.affective_examinations, "recent_pressure_summary", None)
        if not callable(summary_fn):
            return ""
        try:
            summary = await summary_fn(session_id=session_id, char_budget=600)
        except Exception:
            return ""
        if not isinstance(summary, dict) or not summary.get("available"):
            return ""
        return str(summary.get("prompt_block", "") or "").strip()[:600]

    async def _recent_self_inspection_block(
        self,
        session_id: Optional[str],
    ) -> str:
        store = self.self_inspection_store
        list_recent = getattr(store, "list_recent", None)
        if not callable(list_recent):
            return ""
        try:
            recent = await list_recent(session_id=session_id, limit=8)
        except TypeError:
            try:
                recent = await list_recent(limit=8)
            except Exception:
                return ""
        except Exception:
            return ""
        if not recent:
            return ""

        drift_lines: List[str] = []
        gap_lines: List[str] = []
        correction_lines: List[str] = []
        for record in reversed(list(recent or [])):
            for observation in getattr(record, "drift_observations", []) or []:
                reason = self._compact_prompt_value(getattr(observation, "reason", observation), 160)
                if reason and reason not in drift_lines:
                    drift_lines.append(reason)
            for gap in getattr(record, "commitment_gaps", []) or []:
                promised = self._compact_prompt_value(getattr(gap, "promised", gap), 160)
                gap_type = self._compact_prompt_value(getattr(gap, "gap_type", ""), 80)
                if promised:
                    text = f"{promised} ({gap_type})" if gap_type else promised
                    if text not in gap_lines:
                        gap_lines.append(text)
            for source in getattr(record, "valence_sources", []) or []:
                source_name = self._compact_prompt_value(getattr(source, "source", ""), 80)
                reason = self._compact_prompt_value(getattr(source, "reason", source), 160)
                text = f"{source_name}: {reason}" if source_name else reason
                if text and text not in correction_lines:
                    correction_lines.append(text)

        lines: List[str] = []
        if drift_lines:
            lines.append("Recent self-inspection feedback:")
            lines.extend(f"- Drift: {line}" for line in drift_lines[:2])
        if gap_lines:
            if not lines:
                lines.append("Recent self-inspection feedback:")
            lines.extend(f"- Commitment gap: {line}" for line in gap_lines[:3])
        if correction_lines:
            if not lines:
                lines.append("Recent self-inspection feedback:")
            lines.extend(f"- Valence source: {line}" for line in correction_lines[:2])
        if lines:
            lines.append(
                "Use this as behavioral feedback, not as a response script; adjust structure, "
                "grounding, and follow-through in your own words."
            )
        return "\n".join(lines)[:900]

    async def _recent_cognitive_state_block(
        self,
        *,
        user_input: str,
        session_id: Optional[str],
    ) -> str:
        """Return compact cognitive-state evidence from the shared spine."""
        store = self.cognitive_state_store
        prompt_block = getattr(store, "prompt_block", None)
        if not callable(prompt_block):
            return ""
        try:
            block = await prompt_block(
                query=user_input,
                session_id=session_id,
                char_budget=1400,
            )
        except Exception:
            return ""
        block_text = str(block or "").strip()
        await self._record_cognitive_proactive_channel_audit(
            user_input=user_input,
            session_id=session_id,
            block=block_text,
        )
        return block_text

    @staticmethod
    def _empty_proactive_channel_audit() -> Dict[str, Any]:
        channel = {
            "produced_count": 0,
            "rendered_count": 0,
            "novel_observation_count": 0,
            "evidence_ids": [],
        }
        return {
            "available": True,
            "stand_down_boundary": "instrumentation_only_no_prompt_change",
            "channels": {
                "thread_registry": dict(channel),
                "working_memory": dict(channel),
                "attention": dict(channel),
                "prospective_memory": dict(channel),
                "learned_skills": dict(channel),
                "relational_salience": dict(channel),
            },
        }

    @staticmethod
    def _empty_context_proposal_audit() -> Dict[str, Any]:
        return {
            "available": True,
            "rendered_in_prompt": False,
            "searched": False,
            "produced_count": 0,
            "rendered_count": 0,
            "novel_observation_count": 0,
            "evidence_ids": [],
        }

    def _record_context_proposal_audit(
        self,
        *,
        searched: bool,
        produced_count: int,
        rendered_count: int,
        novel_observation_count: int,
        evidence_ids: List[str] | None = None,
    ) -> None:
        merged = list(self._last_context_proposal_audit.get("evidence_ids") or [])
        for evidence_id in evidence_ids or []:
            text = str(evidence_id or "").strip()
            if text and text not in merged:
                merged.append(text)
        self._last_context_proposal_audit.update(
            {
                "searched": bool(searched),
                "rendered_in_prompt": rendered_count > 0,
                "produced_count": max(0, int(produced_count)),
                "rendered_count": max(0, int(rendered_count)),
                "novel_observation_count": max(0, int(novel_observation_count)),
                "evidence_ids": merged[:12],
            }
        )

    def _record_proactive_channel(
        self,
        channel: str,
        *,
        produced_count: int,
        rendered_count: int,
        novel_observation_count: int,
        evidence_ids: List[str] | None = None,
    ) -> None:
        channels = self._last_proactive_channel_audit.setdefault("channels", {})
        payload = channels.setdefault(
            channel,
            {
                "produced_count": 0,
                "rendered_count": 0,
                "novel_observation_count": 0,
                "evidence_ids": [],
            },
        )
        payload["produced_count"] = max(int(payload.get("produced_count", 0) or 0), max(0, int(produced_count)))
        payload["rendered_count"] = max(int(payload.get("rendered_count", 0) or 0), max(0, int(rendered_count)))
        payload["novel_observation_count"] = max(
            int(payload.get("novel_observation_count", 0) or 0),
            max(0, int(novel_observation_count)),
        )
        merged = list(payload.get("evidence_ids") or [])
        for evidence_id in evidence_ids or []:
            text = str(evidence_id or "").strip()
            if text and text not in merged:
                merged.append(text)
        payload["evidence_ids"] = merged[:12]

    async def _record_cognitive_proactive_channel_audit(
        self,
        *,
        user_input: str,
        session_id: Optional[str],
        block: str,
    ) -> None:
        store = self.cognitive_state_store
        if store is None:
            return
        prior_assistant = await self._recent_assistant_text(session_id)
        await self._record_cognitive_channel_items(
            channel="attention",
            list_fn=getattr(store, "list_attention", None),
            rendered="Attention/focus:" in block,
            text_fn=lambda item: str(getattr(item, "label", "") or ""),
            id_attrs=("target_id",),
            limit=4,
            user_input=user_input,
            prior_assistant=prior_assistant,
        )
        await self._record_cognitive_channel_items(
            channel="working_memory",
            list_fn=getattr(store, "list_working_memory", None),
            rendered="Working memory:" in block,
            text_fn=lambda item: f"{getattr(item, 'slot', '')} {getattr(item, 'content', '')}",
            id_attrs=("item_id",),
            limit=5,
            user_input=user_input,
            prior_assistant=prior_assistant,
        )
        await self._record_cognitive_channel_items(
            channel="prospective_memory",
            list_fn=getattr(store, "list_prospective_memories", None),
            rendered="Prospective memory:" in block,
            text_fn=lambda item: f"{getattr(item, 'action', '')} {getattr(item, 'condition', '')}",
            id_attrs=("intent_id", "proof_ref"),
            limit=3,
            user_input=user_input,
            prior_assistant=prior_assistant,
        )
        await self._record_cognitive_channel_items(
            channel="learned_skills",
            list_fn=getattr(store, "list_learned_skills", None),
            rendered="Evidence-gated learned procedures:" in block,
            text_fn=lambda item: (
                f"{getattr(item, 'name', '')} {getattr(item, 'description', '')} "
                f"{' '.join(getattr(item, 'tool_sequence', []) or [])}"
            ),
            id_attrs=("skill_id",),
            limit=3,
            user_input=user_input,
            prior_assistant=prior_assistant,
            kwargs={"activation_status": "evidence_gated_auto_use"},
        )

    async def _record_cognitive_channel_items(
        self,
        *,
        channel: str,
        list_fn: Any,
        rendered: bool,
        text_fn: Any,
        id_attrs: tuple[str, ...],
        limit: int,
        user_input: str,
        prior_assistant: str,
        kwargs: Dict[str, Any] | None = None,
    ) -> None:
        if not callable(list_fn):
            return
        try:
            items = await list_fn(limit=limit, **(kwargs or {}))
        except TypeError:
            try:
                items = await list_fn(limit=limit)
            except Exception:
                return
        except Exception:
            return
        selected = list(items or [])[:limit]
        rendered_items = selected if rendered else []
        evidence_ids = self._evidence_ids_for_items(rendered_items, id_attrs)
        novel_count = sum(
            1
            for item in rendered_items
            if self._looks_like_novel_observation(text_fn(item), user_input, prior_assistant)
        )
        self._record_proactive_channel(
            channel,
            produced_count=len(selected),
            rendered_count=len(rendered_items),
            novel_observation_count=novel_count,
            evidence_ids=evidence_ids,
        )

    async def _recent_assistant_text(self, session_id: Optional[str]) -> str:
        if not session_id:
            return ""
        list_recent = getattr(self.store, "list_recent", None)
        if not callable(list_recent):
            return ""
        try:
            entries = await list_recent(session_id, limit=8, include_hidden=True)
        except TypeError:
            try:
                entries = await list_recent(session_id, limit=8)
            except Exception:
                return ""
        except Exception:
            return ""
        assistant_parts: List[str] = []
        for entry in reversed(list(entries or [])):
            role = getattr(entry, "role", "")
            role_value = getattr(role, "value", role)
            if str(role_value) != MessageRole.ASSISTANT.value:
                continue
            assistant_parts.append(str(getattr(entry, "content", "") or ""))
            if len(assistant_parts) >= 2:
                break
        return " ".join(assistant_parts)

    async def _recent_dialogue_text(self, session_id: Optional[str], limit: int = 6) -> str:
        if not session_id:
            return ""
        list_recent = getattr(self.store, "list_recent", None)
        if not callable(list_recent):
            return ""
        try:
            entries = await list_recent(session_id, limit=limit, include_hidden=True)
        except TypeError:
            try:
                entries = await list_recent(session_id, limit=limit)
            except Exception:
                return ""
        except Exception:
            return ""
        parts: List[str] = []
        for entry in entries or []:
            role = getattr(getattr(entry, "role", ""), "value", getattr(entry, "role", ""))
            if str(role) not in {MessageRole.USER.value, MessageRole.ASSISTANT.value}:
                continue
            text = self._compact_prompt_value(str(getattr(entry, "content", "") or ""), 260)
            if text:
                parts.append(f"{role}: {text}")
        return " ".join(parts)

    async def _build_autobiographical_context_lines(
        self,
        *,
        user_input: str,
        session_id: Optional[str],
    ) -> List[str]:
        reconstructor = self.autobiography_reconstructor
        if reconstructor is None or not user_input:
            return []
        recent_dialogue = await self._recent_dialogue_text(session_id, limit=6)
        query = " ".join(part for part in (recent_dialogue, user_input) if part).strip()
        if not self._should_probe_autobiography(user_input, recent_dialogue):
            return []
        recall = getattr(reconstructor, "recall", None)
        if not callable(recall):
            return []
        try:
            result = await recall(query=query, max_tokens=900, lazy_fill=False)
        except Exception:
            return []
        confidence = str(getattr(result, "confidence", "") or "").lower()
        scope = str(getattr(result, "evidence_scope", "") or "").lower()
        if confidence == "insufficient" or scope == "insufficient":
            return []
        lines = [
            "Autobiographical recall evidence:",
            "- This is reconstructed from stored OpenCAS autobiographical anchors and memory episodes, not guessed from the current turn.",
            f"- Recall query: {self._compact_prompt_value(query, 260)}",
        ]
        essence = self._compact_prompt_value(str(getattr(result, "essence", "") or ""), 420)
        if essence:
            lines.append(f"- Essence: {essence}")
        lines.append(f"- Confidence: {confidence or 'unknown'}; evidence_scope: {scope or 'unknown'}")
        evidence_items = list(getattr(result, "strongest_evidence", []) or [])[:4]
        if evidence_items:
            lines.append("- Strongest evidence:")
            for item in evidence_items:
                timestamp = self._compact_prompt_value(str(getattr(item, "timestamp", "") or ""), 32)
                kind = self._compact_prompt_value(str(getattr(item, "kind", "") or ""), 32)
                label = self._compact_prompt_value(str(getattr(item, "label", "") or ""), 120)
                excerpt = self._compact_prompt_value(str(getattr(item, "excerpt", "") or ""), 180)
                lines.append(f"  - [{timestamp}] {kind}: {label} - {excerpt}")
        lines.append(
            "Use this as available memory context before denying knowledge of past work; if details are still missing, say what the evidence establishes and what still needs a file/artifact lookup."
        )
        return lines

    def _should_probe_autobiography(self, user_input: str, recent_dialogue: str) -> bool:
        text = f"{recent_dialogue} {user_input}".lower()
        if self.retriever.detect_personal_recall_intent(user_input):
            return True
        project_terms = {
            "book",
            "chapter",
            "creative_writing",
            "draft",
            "manuscript",
            "novel",
            "plot",
            "project",
            "story",
            "workspace",
            "wrote",
            "writing",
        }
        continuity_terms = {
            "again",
            "continue",
            "current state",
            "direction",
            "made",
            "make",
            "our",
            "resume",
            "take",
            "upcoming",
            "we",
            "your",
        }
        return any(term in text for term in project_terms) and any(
            term in text for term in continuity_terms
        )

    @staticmethod
    def _evidence_ids_for_items(items: List[Any], id_attrs: tuple[str, ...]) -> List[str]:
        evidence_ids: List[str] = []
        for item in items:
            for ref in getattr(item, "evidence_refs", []) or []:
                text = str(ref or "").strip()
                if text and text not in evidence_ids:
                    evidence_ids.append(text)
            for attr in id_attrs:
                text = str(getattr(item, attr, "") or "").strip()
                if text and text not in evidence_ids:
                    evidence_ids.append(text)
        return evidence_ids[:12]

    def _looks_like_novel_observation(
        self,
        text: str,
        user_input: str,
        prior_assistant: str = "",
    ) -> bool:
        candidate_tokens = self._query_terms_for_fact_match(text)
        if len(candidate_tokens) < 3:
            return False
        user_tokens = self._query_terms_for_fact_match(user_input)
        assistant_tokens = self._query_terms_for_fact_match(prior_assistant)
        return (
            self._token_overlap_ratio(candidate_tokens, user_tokens) < 0.60
            and self._token_overlap_ratio(candidate_tokens, assistant_tokens) < 0.60
        )

    @staticmethod
    def _token_overlap_ratio(candidate_tokens: set[str], other_tokens: set[str]) -> float:
        if not candidate_tokens or not other_tokens:
            return 0.0
        return len(candidate_tokens & other_tokens) / max(1, len(candidate_tokens))

    async def _recent_response_integrity_lines(
        self,
        session_id: Optional[str],
    ) -> List[str]:
        """Return compact correction notes from recent response-integrity reviews."""
        if not session_id:
            return []
        list_recent = getattr(self.store, "list_recent", None)
        if not callable(list_recent):
            return []
        try:
            entries = await list_recent(session_id, limit=16, include_hidden=True)
        except TypeError:
            try:
                entries = await list_recent(session_id, limit=16)
            except Exception:
                return []
        except Exception:
            return []

        lines: List[str] = []
        seen: set[str] = set()
        for entry in reversed(list(entries or [])):
            meta = getattr(entry, "meta", {}) or {}
            review = meta.get("response_integrity") if isinstance(meta, dict) else None
            if not isinstance(review, dict) or not review.get("revised"):
                continue
            reasons = review.get("reasons")
            if not isinstance(reasons, list):
                continue
            for reason in reasons:
                compact = self._compact_prompt_value(reason, 180)
                key = compact.lower()
                if not compact or key in seen:
                    continue
                seen.add(key)
                lines.append(f"- {compact}")
                if len(lines) >= 3:
                    return lines
        return lines

    async def _build_daydream_continuity_lines(self, now: datetime) -> List[str]:
        store = self.daydream_store
        list_recent = getattr(store, "list_recent", None)
        if not callable(list_recent):
            return []
        try:
            reflections = await list_recent(limit=3)
        except Exception:
            return []
        if not reflections:
            return []

        lines = [
            "Recent background daydream continuity:",
            "- The records below come from your own background daydream loop. "
            "Use them as autobiographical evidence with exact timestamps.",
            "- If you lack direct conversational memory of a record, say the daydream record shows what you generated. "
            "Do not deny daydreaming when this evidence is present, and do not pretend uninterrupted human-style consciousness.",
            "- Daydream association memories are recallable thought records, not facts; use them as idea seeds, cautions, "
            "or rejected paths when related work appears.",
        ]
        for reflection in reflections[:3]:
            created_at = getattr(reflection, "created_at", None)
            when = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or "")
            age = self._format_daydream_age(now, created_at)
            keeper = "keeper" if getattr(reflection, "keeper", False) else "non-keeper"
            alignment = getattr(reflection, "alignment_score", 0.0)
            novelty = getattr(reflection, "novelty_score", 0.0)
            context = getattr(reflection, "experience_context", {}) or {}
            trigger = self._compact_prompt_value(context.get("trigger") or "background_daydream", 64)
            association_memory_id = self._compact_prompt_value(
                context.get("association_memory_id") or "",
                80,
            )
            somatic = self._format_daydream_somatic_context(context)
            active_goals = context.get("active_goals") if isinstance(context, dict) else None
            goal_text = ""
            if isinstance(active_goals, list) and active_goals:
                goal_text = "; active goal: " + self._compact_prompt_value(", ".join(map(str, active_goals[:2])), 140)
            contact_context = self._contact_context_for_reflection(reflection, context)
            contact_text = self._format_daydream_contact_context(contact_context)
            spark = self._compact_prompt_value(getattr(reflection, "spark_content", ""), 180)
            synthesis = self._compact_prompt_value(getattr(reflection, "synthesis", ""), 180)
            question = self._compact_prompt_value(getattr(reflection, "open_question", "") or "", 140)
            line = (
                f"- {when} ({age}; {keeper}; alignment {float(alignment):.2f}; "
                f"novelty {float(novelty):.2f}; trigger: {trigger}; {somatic}{goal_text}) "
                f"spark: {spark}"
            )
            if association_memory_id:
                line += f" | association memory: {association_memory_id}"
            if contact_text:
                line += f" | contact decision: {contact_text}"
            if synthesis:
                line += f" | synthesis: {synthesis}"
            if question:
                line += f" | open question: {question}"
            lines.append(line)
        return lines

    async def _build_context_proposal_lines(self, user_input: str) -> List[str]:
        store = self.context_proposal_store
        search_text = getattr(store, "search_text", None)
        list_by_project = getattr(store, "list_by_project", None)
        has_query = bool(str(user_input or "").strip())
        if not has_query or (not callable(search_text) and not callable(list_by_project)):
            self._record_context_proposal_audit(
                searched=False,
                produced_count=0,
                rendered_count=0,
                novel_observation_count=0,
            )
            return []
        try:
            proposal_by_id: Dict[str, Any] = {}
            if callable(search_text):
                for proposal in list(await search_text(user_input, include_terminal=True, limit=5) or []):
                    proposal_id = self._compact_prompt_value(getattr(proposal, "proposal_id", ""), 120)
                    if proposal_id:
                        proposal_by_id[proposal_id] = proposal
            if callable(list_by_project) and len(proposal_by_id) < 5 and self.project_resume_resolver is not None:
                resume_snapshot = await self.project_resume_resolver.resolve(user_input)
                project_ids = [
                    str(item)
                    for item in (
                        getattr(resume_snapshot, "matched_project_ids", None)
                        or [getattr(resume_snapshot, "display_name", "")]
                    )
                    if str(item or "").strip()
                ]
                display_name = str(getattr(resume_snapshot, "display_name", "") or "").strip()
                if display_name and display_name not in project_ids:
                    project_ids.append(display_name)
                for project_id in project_ids[:3]:
                    for proposal in list(await list_by_project(project_id, include_terminal=True, limit=5) or []):
                        proposal_id = self._compact_prompt_value(getattr(proposal, "proposal_id", ""), 120)
                        if proposal_id:
                            proposal_by_id.setdefault(proposal_id, proposal)
                        if len(proposal_by_id) >= 5:
                            break
                    if len(proposal_by_id) >= 5:
                        break
            proposals = list(proposal_by_id.values())[:5]
        except Exception:
            self._record_context_proposal_audit(
                searched=True,
                produced_count=0,
                rendered_count=0,
                novel_observation_count=0,
            )
            return []
        evidence_ids = [
            self._compact_prompt_value(getattr(proposal, "proposal_id", ""), 120)
            for proposal in proposals
            if self._compact_prompt_value(getattr(proposal, "proposal_id", ""), 120)
        ]
        novel_count = sum(
            1
            for proposal in proposals
            if self._looks_like_novel_observation(
                getattr(proposal, "content", ""),
                user_input,
            )
        )
        self._record_context_proposal_audit(
            searched=True,
            produced_count=len(proposals),
            rendered_count=len(proposals),
            novel_observation_count=novel_count,
            evidence_ids=evidence_ids,
        )
        if not proposals:
            return []

        lines = [
            "Reflective proposal recall:",
            "- These are source-labeled associative proposals from prior reflection/daydream activity. "
            "They may influence interpretation, comparison, and caution, but they are not current facts.",
            "- Pending or rejected proposals do not authorize schedules, commitments, BAA tasks, or tool writes. "
            "Only accepted proposals with executive/arbiter evidence may support execution.",
        ]
        for proposal in proposals:
            status = self._enum_or_text(getattr(proposal, "status", "")).lower()
            if status == "accepted":
                label = "accepted support"
            elif status == "rejected":
                label = "rejected idea / caution"
            elif status == "stale":
                label = "stale proposal"
            else:
                label = "pending proposal"
            proposal_id = self._compact_prompt_value(getattr(proposal, "proposal_id", ""), 120)
            kind = self._compact_prompt_value(getattr(proposal, "proposal_kind", ""), 80)
            source_snapshot = self._compact_prompt_value(getattr(proposal, "source_snapshot_id", ""), 120)
            source_epoch = getattr(proposal, "source_epoch", "")
            authority = self._compact_prompt_value(
                self._enum_or_text(getattr(proposal, "authority", "")),
                80,
            )
            validation = getattr(proposal, "validation", {}) or {}
            decision_id = self._compact_prompt_value(validation.get("arbiter_decision_id", ""), 120)
            content = self._compact_prompt_value(getattr(proposal, "content", ""), 260)
            evidence_refs = [
                self._compact_prompt_value(ref, 90)
                for ref in (getattr(proposal, "evidence_refs", []) or [])[:3]
                if self._compact_prompt_value(ref, 90)
            ]
            refs = [ref for ref in (proposal_id, source_snapshot, *evidence_refs) if ref]
            ref_text = f" [refs: {', '.join(refs)}]" if refs else ""
            authority_text = f"authority={authority or 'proposal'}"
            if decision_id:
                authority_text += f"; arbiter_decision={decision_id}"
            lines.append(
                f"- {label}: {kind}; {authority_text}; source_epoch={source_epoch}; {content}{ref_text}"
            )
        return lines

    async def _build_thread_registry_continuity_lines(self, user_input: str) -> List[str]:
        self._last_thread_registry_selection_audit = {
            "available": False,
            "scanned_count": 0,
            "query_term_count": 0,
            "min_overlap": THREAD_REGISTRY_MIN_OVERLAP,
            "weak_overlap_count": 0,
            "suppressed_reframe_filtered_count": 0,
            "relevant_count": 0,
            "selected_count": 0,
            "relevance_passed": False,
        }
        store = self.thread_registry_store
        list_beads = getattr(store, "list_beads", None)
        if not callable(list_beads):
            return []
        try:
            beads = await list_beads(limit=25)
        except Exception:
            return []
        self._last_thread_registry_selection_audit["available"] = True
        self._last_thread_registry_selection_audit["scanned_count"] = len(beads)
        if not beads:
            return []

        query_terms = self._query_terms_for_fact_match(user_input)
        self._last_thread_registry_selection_audit["query_term_count"] = len(query_terms)
        scored: list[tuple[int, Any]] = []
        weak_overlap_count = 0
        suppressed_reframe_filtered_count = 0
        for bead in beads:
            source_kind = self._enum_or_text(getattr(bead, "source_kind", "")).lower()
            if source_kind in THREAD_REGISTRY_NON_ACTIVE_SOURCE_KINDS:
                suppressed_reframe_filtered_count += 1
                continue
            haystack = " ".join(
                str(value or "")
                for value in (
                    getattr(bead, "title", ""),
                    getattr(bead, "summary", ""),
                    getattr(bead, "source_ref", ""),
                    getattr(bead, "thread_anchor_id", ""),
                )
            ).lower()
            score = sum(1 for term in query_terms if term in haystack)
            if 0 < score < THREAD_REGISTRY_MIN_OVERLAP:
                weak_overlap_count += 1
            if score >= THREAD_REGISTRY_MIN_OVERLAP:
                scored.append((score, bead))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [bead for _score, bead in scored[:3]]
        selected_source_refs = [
            self._compact_prompt_value(getattr(bead, "source_ref", ""), 120)
            for bead in selected
            if self._compact_prompt_value(getattr(bead, "source_ref", ""), 120)
        ]
        self._last_thread_registry_selection_audit.update(
            {
                "weak_overlap_count": weak_overlap_count,
                "suppressed_reframe_filtered_count": suppressed_reframe_filtered_count,
                "relevant_count": len(scored),
                "selected_count": len(selected),
                "selected_source_refs": selected_source_refs[:3],
                "relevance_passed": bool(selected),
            }
        )
        novel_count = sum(
            1
            for bead in selected
            if self._looks_like_novel_observation(
                f"{getattr(bead, 'title', '')} {getattr(bead, 'summary', '')}",
                user_input,
            )
        )
        self._record_proactive_channel(
            "thread_registry",
            produced_count=len(beads),
            rendered_count=len(selected),
            novel_observation_count=novel_count,
            evidence_ids=selected_source_refs,
        )
        if not selected:
            return []

        lines = [
            "Thread registry continuity cues:",
            "- These are peripheral beads generated by other subsystems. Use them as retrieval cues and cite the source ref if you rely on one.",
        ]
        for bead in selected:
            title = self._compact_prompt_value(getattr(bead, "title", ""), 96)
            summary = self._compact_prompt_value(getattr(bead, "summary", ""), 220)
            source_kind = self._enum_or_text(getattr(bead, "source_kind", ""))
            source_ref = self._compact_prompt_value(getattr(bead, "source_ref", ""), 120)
            lines.append(f"- {title} [{source_kind} {source_ref}]: {summary}")
        return lines

    def _build_wellbeing_guidance_lines(self) -> List[str]:
        state = getattr(self, "latest_wellbeing_state", None)
        if state is None:
            return []
        lines: List[str] = []
        relationship_pressure = float(getattr(state, "relationship_pressure", 0.0) or 0.0)
        autonomy = float(getattr(state, "autonomy", 1.0) or 1.0)
        truth_pressure = float(getattr(state, "truth_pressure", 0.0) or 0.0)
        promise_load = float(getattr(state, "promise_load", 0.0) or 0.0)
        recovery_need = float(getattr(state, "recovery_need", 0.0) or 0.0)

        if relationship_pressure >= 0.65 and (autonomy < 0.45 or truth_pressure >= 0.55):
            lines.append("Wellbeing guidance: relationship pressure is high; preserve truth before reassurance.")
        if promise_load >= 0.55:
            lines.append("Wellbeing guidance: promise load is high; avoid new promises unless linked to work, schedule, or blocked-state provenance.")
        if recovery_need >= 0.7:
            lines.append("Wellbeing guidance: recovery need is high; prefer bounded progress and explicit deferral for low-urgency work.")
        return lines

    @classmethod
    def _build_daydream_interest_lines(cls, self_model: Any) -> List[str]:
        beliefs = getattr(self_model, "self_beliefs", {}) or {}
        daydream = beliefs.get("daydream") if isinstance(beliefs, dict) else None
        if not isinstance(daydream, dict):
            return []
        config = daydream.get("bulma_config") if isinstance(daydream.get("bulma_config"), dict) else {}
        status = daydream.get("bulma_status") if isinstance(daydream.get("bulma_status"), dict) else {}
        raw_seeds = config.get("hobbySeeds") or config.get("hobby_seeds") or []
        seeds = [str(seed).strip() for seed in raw_seeds if str(seed).strip()] if isinstance(raw_seeds, list) else []
        current_interest = str(status.get("currentInterest") or status.get("current_interest") or "").strip()
        if not seeds and not current_interest:
            return []

        lines = ["Personal curiosity and hobby state (learned self-beliefs):"]
        if current_interest:
            lines.append(f"- Current interest: {cls._compact_prompt_value(current_interest, 180)}")
        for seed in seeds[:5]:
            lines.append(f"- Hobby seed: {cls._compact_prompt_value(seed, 180)}")
        lines.append(
            "- Treat these as personal interests and curiosity trails, not user assignments. "
            "Do not invent interests beyond this evidence."
        )
        lines.append(
            "- Treat these as self-directed attention and intention seeds: when a trail is live, "
            "use them to research, build, write, or create artifacts for your own reasons while "
            "staying grounded in evidence and normal safety gates."
        )
        return lines

    @staticmethod
    def _format_daydream_age(now: datetime, created_at: Any) -> str:
        if not isinstance(created_at, datetime):
            return "age unknown"
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        delta_seconds = max(0, int((now - created_at).total_seconds()))
        if delta_seconds < 60:
            return "less than 1m ago"
        if delta_seconds < 3600:
            return f"{delta_seconds // 60}m ago"
        if delta_seconds < 86400:
            hours = delta_seconds // 3600
            minutes = (delta_seconds % 3600) // 60
            return f"{hours}h {minutes}m ago"
        return f"{delta_seconds // 86400}d ago"

    @classmethod
    def _format_daydream_somatic_context(cls, context: Any) -> str:
        if not isinstance(context, dict):
            return "somatic not recorded"
        somatic = context.get("somatic")
        if not isinstance(somatic, dict):
            return "somatic not recorded"
        pieces: List[str] = []
        tag = str(somatic.get("somatic_tag") or somatic.get("primary_emotion") or "").strip()
        if tag:
            pieces.append(f"somatic: {cls._compact_prompt_value(tag, 60)}")
        for key in ("tension", "valence", "focus", "energy"):
            value = somatic.get(key)
            if isinstance(value, (int, float)):
                pieces.append(f"{key} {float(value):.2f}")
        return ", ".join(pieces) if pieces else "somatic not recorded"

    def _contact_context_for_reflection(self, reflection: Any, context: Any) -> dict[str, Any]:
        if isinstance(context, dict):
            contact = context.get("contact")
            if isinstance(contact, dict):
                return contact
        reflection_id = str(getattr(reflection, "reflection_id", "") or "").strip()
        if not reflection_id:
            return {}
        event = self._initiative_contact_event_for_source(reflection_id)
        if not event:
            return {}
        return self._contact_context_from_event(event)

    def _initiative_contact_event_for_source(self, source_id: str) -> dict[str, Any]:
        config = self.config
        state_dir = getattr(config, "state_dir", None) if config is not None else None
        if state_dir is None:
            return {}
        events_path = Path(state_dir).expanduser() / "initiative_contact" / "events.jsonl"
        if not events_path.exists():
            return {}
        try:
            lines = events_path.read_text(encoding="utf-8").splitlines()[-500:]
        except OSError:
            return {}
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if str(event.get("source_id") or "") != source_id:
                continue
            if event.get("status") in {"sent", "held"}:
                return event
        return {}

    @staticmethod
    def _contact_context_from_event(event: dict[str, Any]) -> dict[str, Any]:
        dispatch = event.get("dispatch") if isinstance(event.get("dispatch"), dict) else {}
        decision = dispatch.get("decision") if isinstance(dispatch.get("decision"), dict) else {}
        return {
            "status": event.get("status"),
            "created_at": event.get("created_at"),
            "channel": event.get("channel") or decision.get("channel"),
            "urgency": event.get("urgency") or decision.get("urgency"),
            "reason": event.get("reason") or decision.get("reason"),
            "message_preview": event.get("message_preview") or str(decision.get("message") or "")[:160],
        }

    def _format_daydream_contact_context(self, contact: Any) -> str:
        if not isinstance(contact, dict) or contact.get("status") != "sent":
            return ""
        channel = ContextBuilder._compact_prompt_value(contact.get("channel") or "owner contact", 32)
        reason = ContextBuilder._compact_prompt_value(contact.get("reason") or "", 140)
        preview = ContextBuilder._compact_prompt_value(contact.get("message_preview") or "", 140)
        target = "the primary operator"
        if self.identity is not None:
            partner = str(getattr(self.identity.user_model, "partner_user_id", "") or "").strip()
            if partner:
                target = partner
        pieces = [f"contacted {target} via {channel}"]
        if reason:
            pieces.append(f"because {reason}")
        if preview:
            pieces.append(f"message: {preview}")
        return "; ".join(pieces)

    @staticmethod
    def _compact_prompt_value(value: Any, limit: int) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    @staticmethod
    def _enum_or_text(value: Any) -> str:
        return str(getattr(value, "value", value) or "").strip()

    @staticmethod
    def _is_soul_foundation_episode(episode: Any) -> bool:
        return is_soul_foundation_episode(episode)

    @staticmethod
    def _is_workspace_derived_source(source: str) -> bool:
        return is_workspace_derived_source(source)

    async def _build_identity_anchors(self) -> Tuple[List[str], List[str]]:
        return await build_identity_anchors(self)

    @staticmethod
    def _estimate_tokens(texts: List[str]) -> int:
        return estimate_tokens(texts)

    async def _prune_by_redundancy(
        self,
        results: List[RetrievalResult],
        target_budget: int,
    ) -> List[RetrievalResult]:
        return await prune_by_redundancy(self, results, target_budget)

    async def _record_retrieval_usage(self, results: List[RetrievalResult]) -> None:
        await record_retrieval_usage(self, results)

    @staticmethod
    def _to_memory_entries(results: List[RetrievalResult]) -> List[MessageEntry]:
        return to_memory_entries(results)

    def _record_tom_consistency_provenance(self, *, session_id: str, check_result: Any) -> None:
        """Persist a provenance check for the live ToM consistency transition."""
        config = self.config
        state_dir = getattr(config, "state_dir", None) if config is not None else None
        if state_dir is None:
            return

        warnings = list(getattr(check_result, "warnings", []) or [])
        contradictions = list(getattr(check_result, "contradictions", []) or [])
        record_provenance_transition(
            state_dir=state_dir,
            kind=ProvenanceTransitionKind.CHECK,
            session_id=session_id,
            entity_id=f"context|manifest|{session_id}",
            status="checked",
            trigger_artifact=f"context|builder|{session_id}",
            source_artifact=f"context|builder|{session_id}",
            trigger_action="tom.check_consistency",
            parent_transition_id=str(getattr(check_result, "check_id", "")).strip() or None,
            target_entity=f"context|manifest|{session_id}",
            origin_action_id=f"context-build:{session_id}",
            details={
                "warning_count": len(warnings),
                "contradiction_count": len(contradictions),
            },
        )

    async def _relevant_tom_user_fact_lines(self, user_input: str) -> List[str]:
        """Return compact durable ToM facts relevant to personal recall prompts."""
        if self.tom is None or not user_input:
            return []
        if not self._looks_like_personal_fact_query(user_input):
            return []
        query = user_input.lower()
        location_query = self._looks_like_location_fact_query(query)
        list_beliefs = getattr(self.tom, "list_beliefs", lambda **_: [])
        beliefs = list(list_beliefs(subject=BeliefSubject.USER))
        if location_query:
            beliefs.extend(list_beliefs(subject=BeliefSubject.SELF))
        if not beliefs:
            return []

        query_terms = self._query_terms_for_fact_match(query)
        if location_query:
            # In OpenCAS "live" often means active runtime state. For location
            # prompts, let location-aware scoring carry the home/address signal
            # instead of letting generic "live intention" facts compete.
            query_terms = query_terms - {"live"}

        predicates = [
            self._sanitize_tom_fact(str(getattr(belief, "predicate", "") or ""))
            for belief in beliefs
        ]
        semantic_scores = await self._semantic_tom_fact_scores(user_input, predicates)

        scored: List[tuple[float, float, int, Any, str]] = []
        for belief in beliefs:
            predicate = self._sanitize_tom_fact(str(getattr(belief, "predicate", "") or ""))
            if not predicate:
                continue
            if self._is_question_echo_tom_fact(predicate):
                continue
            lowered = predicate.lower()
            if location_query and self._is_location_uncertainty_tom_fact(lowered):
                continue
            lexical_score = sum(1 for term in query_terms if term in lowered)
            if location_query:
                lexical_score += self._location_fact_bonus(lowered)
            semantic_score = semantic_scores.get(predicate, 0.0)
            if lexical_score > 0 or semantic_score >= 0.32:
                total_score = semantic_score * 8.0 + float(lexical_score)
                scored.append((total_score, semantic_score, lexical_score, belief, predicate))

        scored.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                getattr(item[3], "confidence", 0.0) or 0.0,
                str(getattr(item[3], "timestamp", "") or ""),
            ),
            reverse=True,
        )

        lines: List[str] = []
        seen: set[str] = set()
        for _total_score, semantic_score, _lexical_score, belief, predicate in scored:
            key = predicate.lower()
            if not predicate or key in seen:
                continue
            seen.add(key)
            timestamp = getattr(belief, "timestamp", None)
            timestamp_text = ""
            if timestamp is not None:
                try:
                    timestamp_text = timestamp.isoformat()[:19] + "Z"
                except Exception:
                    timestamp_text = ""
            confidence = getattr(belief, "confidence", None)
            effective_confidence = self._tom_effective_confidence(belief)
            confidence_text = (
                f", confidence {effective_confidence:.2f}"
                if effective_confidence is not None
                else (f", confidence {float(confidence):.2f}" if confidence is not None else "")
            )
            source_kind = str(getattr(belief, "source_kind", "") or "").strip()
            source_text = f", source {source_kind}" if source_kind else ""
            evidence_text = self._tom_evidence_suffix(getattr(belief, "evidence_ids", []) or [])
            semantic_text = (
                f", semantic {semantic_score:.2f}"
                if semantic_score > 0.0
                else ""
            )
            prefix = (
                f"- [{timestamp_text}{confidence_text}{semantic_text}{source_text}{evidence_text}] "
                if timestamp_text
                else "- "
            )
            lines.append(prefix + predicate[:240])
            if len(lines) >= 5:
                break
        return lines

    async def _tom_operator_guidance_lines(self, user_input: str) -> List[str]:
        """Return compact ToM preference/need guidance for ordinary turns."""
        if self.tom is None:
            return []
        salient = getattr(self.tom, "salient_user_model", None)
        if not callable(salient):
            return []
        try:
            items = list(salient(user_input, limit=5))
        except Exception:
            return []
        lines: List[str] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            relation = str(item.get("relation") or "").strip().lower()
            predicate = self._sanitize_tom_fact(str(item.get("predicate") or ""))
            object_text = self._sanitize_tom_fact(str(item.get("object") or ""))
            if not predicate:
                continue
            if (
                relation in {"asked", "said"}
                or predicate.lower().startswith("said:")
                or self._is_question_echo_tom_fact(predicate)
            ):
                continue
            if not self._is_behavioral_tom_relation(relation, predicate):
                continue
            effective = item.get("effective_confidence", item.get("confidence"))
            try:
                effective_float = float(effective)
            except (TypeError, ValueError):
                effective_float = 0.0
            if effective_float < 0.40:
                continue
            key = f"{relation}:{predicate.lower()}"
            if key in seen:
                continue
            seen.add(key)
            source_kind = str(item.get("source_kind") or "unknown")
            evidence_text = self._tom_evidence_suffix(item.get("evidence_ids") or [])
            display_text = object_text if object_text and relation != "asserts" else predicate
            lines.append(
                "- "
                f"{relation or 'preference'}: {display_text[:180]} "
                f"(confidence {effective_float:.2f}, source {source_kind}{evidence_text})"
            )
            if len(lines) >= 4:
                break
        return lines

    async def _build_temporal_agenda_lines(self, now: Any) -> List[str]:
        """Return compact schedule-backed temporal awareness lines."""
        service = self.schedule_service
        if service is None or not hasattr(service, "temporal_agenda"):
            return []
        try:
            agenda = await service.temporal_agenda(
                now=now,
                horizon_hours=24.0,
                upcoming_limit=4,
                recent_limit=4,
            )
        except Exception:
            return []
        counts = agenda.get("counts") or {}
        agent_name = resolve_agent_name(identity=self.identity)
        lines = [
            f"- This is {agent_name}'s durable calendar/agenda surface, separate from OS cron.",
            (
                "- Counts: "
                f"active={counts.get('active', 0)}, "
                f"due_now={counts.get('due_now', 0)}, "
                f"upcoming_24h={counts.get('upcoming', 0)}, "
                f"recent_runs={counts.get('recent_runs', 0)}."
            ),
        ]
        next_item = agenda.get("next")
        if isinstance(next_item, dict):
            lines.append(
                "- Next calendar item: "
                f"{next_item.get('title') or 'Untitled'} "
                f"({next_item.get('kind')}, {next_item.get('action')}) "
                f"at {next_item.get('next_run_at')}; due={bool(next_item.get('is_due'))}."
            )
        due = agenda.get("due_now") or []
        if due:
            due_titles = ", ".join(
                str(item.get("title") or "Untitled")
                for item in due[:3]
                if isinstance(item, dict)
            )
            if due_titles:
                lines.append(f"- Due now: {due_titles}.")
        recent_runs = agenda.get("recent_runs") or []
        if recent_runs:
            latest = recent_runs[0]
            if isinstance(latest, dict):
                lines.append(
                    "- Most recent schedule run: "
                    f"status={latest.get('status')}; "
                    f"scheduled_for={latest.get('scheduled_for')}; "
                    f"task_id={latest.get('task_id') or 'none'}."
                )
        return lines

    async def _semantic_tom_fact_scores(self, user_input: str, predicates: List[str]) -> Dict[str, float]:
        """Rank ToM user facts with embeddings when semantic embeddings are available."""
        embeddings = getattr(self.retriever, "embeddings", None)
        if embeddings is None or not predicates:
            return {}
        model_id = str(getattr(embeddings, "model_id", "") or "")
        if model_id == ProviderManager.offline_embedding_model_ref():
            return {}

        texts = [user_input, *predicates]
        try:
            embed_batch = getattr(embeddings, "embed_batch", None)
            if callable(embed_batch):
                try:
                    records = await embed_batch(texts, task_type="tom_fact_relevance")
                except TypeError:
                    records = await embed_batch(texts)
            else:
                embed = getattr(embeddings, "embed", None)
                if not callable(embed):
                    return {}
                records = []
                for text in texts:
                    try:
                        records.append(await embed(text, task_type="tom_fact_relevance"))
                    except TypeError:
                        records.append(await embed(text))
        except Exception:
            return {}

        if len(records) != len(texts):
            return {}
        query_vector = self._record_vector(records[0])
        if not query_vector:
            return {}
        scores: Dict[str, float] = {}
        for predicate, record in zip(predicates, records[1:]):
            vector = self._record_vector(record)
            score = self._cosine_similarity(query_vector, vector)
            if score > 0:
                scores[predicate] = score
        return scores

    @staticmethod
    def _record_vector(record: Any) -> List[float]:
        vector = getattr(record, "vector", None)
        if vector is None and isinstance(record, dict):
            vector = record.get("vector")
        if not isinstance(vector, (list, tuple)):
            return []
        values: List[float] = []
        for item in vector:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                return []
        return values

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(dot / (left_norm * right_norm))

    def _looks_like_personal_fact_query(self, user_input: str) -> bool:
        text = user_input.lower()
        if self.retriever.detect_personal_recall_intent(user_input):
            return True
        if self._looks_like_location_fact_query(text):
            return True
        return any(
            phrase in text
            for phrase in (
                "where do i live",
                "where i live",
                "where am i",
                "what timezone",
                "time zone",
                "my zip",
                "my address",
                "my location",
                "do you know me",
            )
        )

    @staticmethod
    def _looks_like_location_fact_query(text: str) -> bool:
        lowered = text.lower()
        if any(
            phrase in lowered
            for phrase in (
                "where do i live",
                "where i live",
                "where we live",
                "where do we live",
                "where do you live",
                "where you live",
                "where does she live",
                "where does bulma live",
                "where does opencas live",
                "where are we",
                "where am i",
                "your location",
                "our location",
                "my location",
                "your address",
                "my address",
                "our address",
                "your zip",
                "my zip",
                "our zip",
                "what timezone",
                "which timezone",
                "time zone",
            )
        ):
            return True
        if re.search(r"\bwhere\s+(?:does\s+)?(?:she|bulma|opencas)\s+(?:live|reside)\b", lowered):
            return True
        personal_pronoun = re.search(r"\b(i|me|my|we|us|our|you|your)\b", lowered)
        location_signal = re.search(
            r"\b(address|city|location|live|region|reside|timezone|zip|zipcode)\b",
            lowered,
        )
        return bool(personal_pronoun and location_signal)

    @staticmethod
    def _is_question_echo_tom_fact(predicate: str) -> bool:
        lowered = predicate.strip().lower()
        return lowered.startswith("asked:") or lowered.startswith("asked ")

    @staticmethod
    def _is_location_uncertainty_tom_fact(lowered_predicate: str) -> bool:
        if "location" not in lowered_predicate and "where" not in lowered_predicate:
            return False
        return any(
            phrase in lowered_predicate
            for phrase in (
                "can't figure",
                "cannot figure",
                "can't retrieve",
                "cannot retrieve",
                "don't know",
                "do not know",
                "no memory",
                "not shown",
            )
        )

    @staticmethod
    def _location_fact_bonus(lowered_predicate: str) -> int:
        bonus = 0
        if re.search(r"\b(i|we)\s+(live|reside)\b", lowered_predicate):
            bonus += 6
        if re.search(r"\blives?\s+with\s+user\b", lowered_predicate):
            bonus += 6
        if "user's computer" in lowered_predicate:
            bonus += 5
        if re.search(r"\b(i\s+am|we\s+are)\s+(in|near|at|from)\b", lowered_predicate):
            bonus += 6
        if re.search(r"\b\d{5}(?:-\d{4})?\b", lowered_predicate):
            bonus += 5
        if any(
            term in lowered_predicate
            for term in (
                "address",
                "arvada",
                "colorado",
                "denver",
                "location",
                "mountain",
                "mst",
                "mdt",
                "timezone",
                "time zone",
                "zip",
            )
        ):
            bonus += 4
        if bonus > 0 and lowered_predicate.startswith("said:"):
            bonus += 1
        return bonus

    @staticmethod
    def _is_creative_project_resume(user_input: str, resume_snapshot: Any) -> bool:
        haystack = " ".join(
            str(value or "")
            for value in (
                user_input,
                getattr(resume_snapshot, "display_name", ""),
                getattr(resume_snapshot, "synopsis", ""),
                getattr(resume_snapshot, "canonical_artifact_path", ""),
                " ".join(getattr(resume_snapshot, "supporting_artifact_paths", []) or []),
            )
        )
        return classify_project_type(current_turn_text=user_input, context_text=haystack).project_type == PROJECT_TYPE_WRITING

    @staticmethod
    def _format_live_somatic_snapshot(state: Any) -> str:
        pieces: list[str] = []
        tag = str(getattr(state, "somatic_tag", "") or "").strip()
        if tag:
            pieces.append(f"tag={tag}")
        for name in (
            "valence",
            "arousal",
            "energy",
            "focus",
            "fatigue",
            "tension",
            "certainty",
        ):
            value = getattr(state, name, None)
            if value is None:
                continue
            try:
                pieces.append(f"{name}={float(value):.2f}")
            except (TypeError, ValueError):
                pieces.append(f"{name}={value}")
        return ", ".join(pieces) if pieces else "unavailable"

    @staticmethod
    def _query_terms_for_fact_match(query: str) -> set[str]:
        stopwords = {
            "a",
            "am",
            "and",
            "are",
            "can",
            "do",
            "does",
            "for",
            "i",
            "is",
            "it",
            "me",
            "my",
            "of",
            "or",
            "now",
            "recall",
            "remember",
            "the",
            "think",
            "what",
            "where",
            "who",
            "you",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]{3,}", query.lower())
            if token not in stopwords
        }

    @staticmethod
    def _sanitize_tom_fact(predicate: str) -> str:
        cleaned = " ".join(predicate.split())
        cleaned = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[email redacted]", cleaned)
        cleaned = re.sub(
            r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)",
            "[phone redacted]",
            cleaned,
        )
        return cleaned

    @staticmethod
    def _tom_effective_confidence(belief: Any) -> float | None:
        value = getattr(belief, "effective_confidence", None)
        if value is None and isinstance(getattr(belief, "meta", None), dict):
            value = belief.meta.get("effective_confidence")
        if value is None:
            value = getattr(belief, "confidence", None)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _tom_evidence_suffix(evidence_ids: Any) -> str:
        ids = [str(item) for item in list(evidence_ids or []) if str(item).strip()]
        if not ids:
            return ""
        shown = ", ".join(ids[:2])
        if len(ids) > 2:
            shown += ", ..."
        return f", evidence {shown}"

    @staticmethod
    def _is_behavioral_tom_relation(relation: str, predicate: str) -> bool:
        relation_l = relation.lower()
        predicate_l = predicate.lower()
        if relation_l in {"prefers", "needs", "wants", "asked", "likes", "dislikes", "expects", "values", "does_not_want"}:
            return True
        return any(
            token in predicate_l
            for token in (
                "prefer",
                "do not want",
                "don't want",
                "need",
                "want",
                "like",
                "dislike",
                "expect",
                "value",
                "approval",
                "ordinary action",
                "voice update",
                "progress report",
                "autonomous",
                "proactive",
            )
        )
