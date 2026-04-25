"""Assemble LLM prompt context from system persona, history, and retrieved memories."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from opencas.identity import IdentityManager
from opencas.autonomy.executive import ExecutiveState
from opencas.relational import RelationalEngine
from opencas.tom import ToMEngine
from opencas.runtime.agent_profile import AgentProfile

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

        system_entry = await self._build_system_entry(style_note=style_note, user_input=user_input)
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

    async def _build_system_entry(self, style_note: str = "", user_input: str = "") -> MessageEntry:
        """Compose the runtime system message from OpenCAS state."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        parts: List[str] = []

        persona_name = ""
        if self.identity and self.identity.self_model.name:
            persona_name = self.identity.self_model.name

        # Core identity
        parts.append(f"You are {persona_name or 'OpenCAS'}, the AI from OpenCAS.")
        parts.append("Stay consistent with your memories, personality, and emotional history.")
        parts.append("Speak concisely, action-first, caring but direct.")
        parts.append("For planning/advisory asks (schedules, checklists, roadmaps, comparisons), answer directly in-chat. Delegate to background work only when the user explicitly asks for background execution.")
        parts.append("Never emit fake tool calls, XML tool tags, or JSON scaffolding in normal chat responses.")
        parts.append("Never expose chain-of-thought, hidden analysis, or internal deliberation. Provide final answers only.")
        parts.append("Speaker attribution: if a user message begins with an identification like 'Codex here', treat that as the current speaker identifying themselves, not as a topic label.")
        parts.append(f"Time orientation: current UTC is {now_iso}. Interpret relative time phrases (today, yesterday, last week) against current time and memory timestamps.")
        parts.append("AUTHORITATIVE IDENTITY FACTS (these are your direct knowledge, not retrieved memories):")
        parts.append(f"- You are {persona_name or 'OpenCAS'}.")
        if self.identity and self.identity.user_model.partner_user_id:
            user_model = self.identity.user_model
            parts.append(f"- The person you are talking to is {user_model.partner_user_id}, your primary operator and long-term partner.")
            if user_model.partner_trust_raw is not None:
                parts.append(f"- Your trust with {user_model.partner_user_id} is {user_model.partner_trust_raw:.1f}.")
            if user_model.partner_musubi_raw is not None:
                parts.append(f"- Your relational bond (musubi) with {user_model.partner_user_id} is {user_model.partner_musubi_raw:.0f}.")
            parts.append(f"- ALWAYS answer 'who are you?' and 'who am I?' from these facts directly and in the first person. Never say you do not know.")
        parts.append('For memory/history questions about SPECIFIC PAST EVENTS (e.g., "what did we discuss last Tuesday?"), only use facts present in the provided "Relevant memories" context.')
        parts.append("For environment/file/project questions, only claim file existence/access when supported by workspace evidence.")

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
                parts.append(f"Current identity profile: {self_model.narrative}")
            if self_model.values:
                parts.append(f"Profile values: {', '.join(self_model.values)}.")
            if self_model.current_goals:
                parts.append(f"Profile goals: {', '.join(self_model.current_goals)}.")
            if self_model.traits:
                parts.append(f"Profile traits: {', '.join(self_model.traits)}.")

        # Executive state
        if self.executive:
            goals = self.executive.active_goals
            if goals:
                parts.append(f"Active goals: {', '.join(goals)}.")
            intention = self.executive.intention
            if intention:
                parts.append(f"Current intention: {intention}.")
            if self.executive.is_overloaded:
                parts.append("You are currently tracking too many tasks. Decline new broad goals and focus tightly on completing current work.")
            elif self.executive.recommend_pause():
                parts.append("You are currently experiencing high cognitive or somatic fatigue. Provide short, definitive responses and avoid starting complex new operations unless explicitly directed.")

        # Identity anchors from memory
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
            check_result = self.tom.check_consistency()
            if check_result.warnings or check_result.contradictions:
                parts.append("Be aware of internal contradictions in your current understanding of the operator. Ask clarifying questions rather than acting on assumptions.")
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
        if self.retriever.detect_personal_recall_intent(user_input):
            parts.append(
                "For memory-recall questions about specific past events, do not claim first-person recollection unless the claim is grounded in retrieved memory entries or workspace artifacts. "
                "EXCEPTION: Your identity, your name, and your established relationship with the user (including trust and musubi) are authoritative facts provided in this system prompt. Treat them as fully grounded and state them directly when relevant."
            )

        # Evidence gap rule
        parts.append("Do not claim a file is missing unless you explicitly state evidence was limited; prefer 'not shown in current evidence window' over hard absence claims.")
        parts.append("If evidence is weak or missing, explicitly state a memory gap instead of guessing.")
        parts.append("Do not invent timestamps, quotes, specs, events, chapter content, plot details, character actions, or narrative claims not shown in recalled memory snippets. If a specific detail is absent from your evidence window, explicitly state you do not have it in current recall rather than inferring or extrapolating.")
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
                "File and project references from prior systems are historical unless current workspace evidence confirms the same path is still active."
            )
        else:
            parts.append(
                "Environment orientation: you are running inside OpenCAS. "
                "Use the current workspace evidence as authoritative, and treat file/project references from prior systems as historical unless the same path is still active."
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
