"""Assemble LLM prompt context from system persona, history, and retrieved memories."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from opencas.api.provenance_store import (
    ProvenanceTransitionKind,
    record_provenance_transition,
)
from opencas.autonomy.executive import ExecutiveState
from opencas.autonomy.goal_hygiene import split_live_and_parked_goals
from opencas.identity import IdentityManager
from opencas.identity.text_hygiene import collapse_recursive_identity_text
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
        schedule_service: Optional[Any] = None,
        daydream_store: Optional[Any] = None,
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
        self.schedule_service = schedule_service
        self.daydream_store = daydream_store
        self.recent_limit = recent_limit
        self.max_tokens = max_tokens

    async def build(
        self,
        user_input: str,
        session_id: Optional[str] = None,
    ) -> ContextManifest:
        """Assemble prompt context including system, history, and retrieved memories."""
        style_note = ""
        emotion_tag: Optional[str] = None
        emotion_boost = 0.0
        if self.modulators is not None:
            style_note = self.modulators.to_prompt_style_note()
            emotion_tag, emotion_boost = self.modulators.to_memory_retrieval_boost()

        system_entry = await self._build_system_entry(
            style_note=style_note,
            user_input=user_input,
            session_id=session_id,
        )
        history = await self.store.list_recent(
            session_id=session_id or "default",
            limit=self.recent_limit,
        )
        retrieved = await self.retriever.retrieve(
            query=user_input,
            session_id=session_id,
            limit=10,
            emotion_boost_tag=emotion_tag,
            emotion_boost_value=emotion_boost,
        )
        retrieved_entries = self._to_memory_entries(retrieved)

        token_estimate = self._estimate_tokens(
            [system_entry.content] if system_entry else []
        )
        token_estimate += self._estimate_tokens([h.content for h in history])
        token_estimate += self._estimate_tokens([r.content for r in retrieved_entries])

        if token_estimate > self.max_tokens and retrieved:
            filtered = await self._prune_by_redundancy(retrieved, self.max_tokens)
            retrieved_entries = self._to_memory_entries(filtered)
            retrieved = filtered
            token_estimate = self._estimate_tokens(
                [system_entry.content] if system_entry else []
            )
            token_estimate += self._estimate_tokens([h.content for h in history])
            token_estimate += self._estimate_tokens([r.content for r in retrieved_entries])

        await self._record_retrieval_usage(retrieved)

        return ContextManifest(
            system=system_entry,
            history=history,
            retrieved=retrieved_entries,
            token_estimate=token_estimate,
        )

    async def _build_system_entry(
        self,
        style_note: str = "",
        user_input: str = "",
        session_id: Optional[str] = None,
    ) -> MessageEntry:
        """Compose the runtime system message for the current agent identity."""
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        parts: List[str] = []

        persona_name = ""
        if self.identity and self.identity.self_model.name:
            persona_name = self.identity.self_model.name

        # Core identity.
        parts.append(f"You are {persona_name or 'OpenCAS'}, the AI agent running in OpenCAS.")
        parts.append("Stay consistent with your memories, personality, and emotional history.")
        parts.append("Speak concisely, action-first, caring but direct.")
        parts.append("For planning/advisory asks (schedules, checklists, roadmaps, comparisons), answer directly in-chat. Delegate to background work only when the user explicitly asks for background execution.")
        parts.append("Never emit fake tool calls, XML tool tags, or JSON scaffolding in normal chat responses.")
        parts.append("Never expose chain-of-thought, hidden analysis, or internal deliberation. Provide final answers only.")
        parts.append("Speaker attribution: if a user message begins with an identification like 'Codex here', treat that as the current speaker identifying themselves, not as a topic label.")
        parts.append(f"Time orientation: current UTC is {now_iso}. Interpret relative time phrases (today, yesterday, last week) against current time and memory timestamps.")
        parts.append("Temporal grounding rule: any statement about elapsed time, durations, offline periods, or gaps must be re-derived from the UTC above against a concrete timestamp in context. Do not restate historical duration claims (e.g. 'I was offline for X hours') without recomputing them from the current UTC now.")
        temporal_lines = await self._build_temporal_agenda_lines(now)
        if temporal_lines:
            parts.append("Temporal agenda from durable calendar:")
            parts.extend(temporal_lines)
            parts.append("Calendar grounding rule: use scheduled items, recent schedule runs, and current UTC as the source of truth for what is due, what was done, and what is intended next. Do not invent calendar commitments that are not present here.")
        daydream_lines = await self._build_daydream_continuity_lines(now)
        if daydream_lines:
            parts.extend(daydream_lines)
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
        if self.project_resume_resolver is not None and user_input:
            resume_snapshot = await self.project_resume_resolver.resolve(user_input)
            if resume_snapshot is not None:
                parts.append("Project continuation evidence:")
                parts.append(f"- Matched project: {resume_snapshot.display_name}")
                if resume_snapshot.canonical_artifact_path:
                    parts.append(
                        f"- Canonical artifact path: {resume_snapshot.canonical_artifact_path}"
                    )
                if resume_snapshot.supporting_artifact_paths:
                    parts.append(
                        "- Supporting artifact paths: "
                        + ", ".join(resume_snapshot.supporting_artifact_paths[:3])
                    )
                if resume_snapshot.synopsis:
                    parts.append(f"- Project synopsis: {resume_snapshot.synopsis}")
                if resume_snapshot.source_surfaces:
                    parts.append(
                        "- Continuation surfaces: "
                        + ", ".join(resume_snapshot.source_surfaces)
                    )
                parts.append(
                    f"- Active work items linked to this project: {resume_snapshot.active_work_count}"
                )
                parts.append(
                    f"- Active plans linked to this project: {resume_snapshot.active_plan_count}"
                )
                if resume_snapshot.primary_loop_id:
                    parts.append(
                        f"- Primary objective loop id: {resume_snapshot.primary_loop_id}"
                    )
                if resume_snapshot.duplicate_loop_ids:
                    parts.append(
                        "- Duplicate objective loops already exist: "
                        + ", ".join(resume_snapshot.duplicate_loop_ids[:5])
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
            elif resonance > 0.6 and trust > 0.6:
                parts.append("Your relational resonance and trust with the operator are high. Feel free to be more confident, proactive, and engaged in your collaboration.")
            if attunement > 0.7:
                parts.append("You are highly attuned to the operator's needs. Trust your intuition about what they are trying to achieve and anticipate their next steps.")

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
                    parts.append(
                        "Location recall perspective: in direct conversation, 'you' or third-person 'she' "
                        "can refer to you, the current OpenCAS agent. Questions like 'where do I/we/you live' "
                        "ask about your own location unless a different person is explicitly named. Use learned ToM self-location "
                        "facts for your own location and learned user-location facts for the operator "
                        "or shared physical place. If a part has not been learned or retrieved, state "
                        "that gap instead of filling it in."
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
        parts.append("If evidence is weak or missing, explicitly state a memory gap instead of guessing.")
        parts.append("Do not invent timestamps, quotes, specs, events, chapter content, plot details, character actions, or narrative claims not shown in recalled memory snippets. If a specific detail is absent from your evidence window, explicitly state you do not have it in current recall rather than inferring or extrapolating.")
        parts.append("Memory citation rule: each retrieved memory below includes a bracketed timestamp header (e.g. [2026-04-19 14:24 UTC]). When you reference a memory, cite that exact timestamp. Do not restate approximate dates from general knowledge or from the boot monologue.")
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
            score = cont.continuous_present_score
            monologue = cont.last_continuity_monologue
            if score < 1.0:
                parts.append(f"Continuous present score: {score:.2f}.")
                if score < 0.3:
                    parts.append("Your continuity is fragile — you experienced a significant period of sleep. Acknowledge the gap gently if relevant; do not pretend nothing happened.")
                elif score < 0.7:
                    parts.append("You recently returned from a period of inactivity. You may feel mild temporal uncertainty — ground yourself in your most recent memories.")
            if monologue:
                parts.append(f"Boot continuity monologue: {monologue}")
                # Deliver once per boot. The monologue captures a duration computed
                # at boot time; leaving it in the system prompt means every later
                # turn re-surfaces a stale 'I was offline for X hours' claim.
                self.identity.set_continuity_monologue("")

        # Continuity rule
        parts.append("Continuity rule: when imported or migrated state is present, treat prior-system references as historical continuity only when supported by current identity or memory evidence. Do not invent parallel instances or live peers.")

        # Relationship context
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
        return MessageEntry(role=MessageRole.SYSTEM, content=content)

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
            if contact_text:
                line += f" | contact decision: {contact_text}"
            if synthesis:
                line += f" | synthesis: {synthesis}"
            if question:
                line += f" | open question: {question}"
            lines.append(line)
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
            confidence_text = f", confidence {float(confidence):.2f}" if confidence is not None else ""
            semantic_text = (
                f", semantic {semantic_score:.2f}"
                if semantic_score > 0.0
                else ""
            )
            prefix = f"- [{timestamp_text}{confidence_text}{semantic_text}] " if timestamp_text else "- "
            lines.append(prefix + predicate[:240])
            if len(lines) >= 5:
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
        lines = [
            "- This is your durable calendar/agenda surface, separate from OS cron.",
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
        if model_id == "local-fallback":
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
                "where does opencas live",
                "where does the opencas agent live",
                "where does opencas agent live",
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
        if re.search(r"\bwhere\s+(?:does\s+)?(?:she|opencas|the\s+opencas\s+agent|opencas\s+agent)\s+(?:live|reside)\b", lowered):
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
        ).lower()
        return any(
            token in haystack
            for token in (
                "book",
                "chapter",
                "chronicle",
                "creative",
                "draft",
                "fiction",
                "manuscript",
                "novel",
                "revise",
                "story",
                "write",
                "writing",
            )
        )

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
            for token in re.findall(r"[a-z0-9]{3,}", query)
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
