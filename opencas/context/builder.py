"""Assemble LLM prompt context from system persona, history, and retrieved memories."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from opencas.identity import IdentityManager
from opencas.autonomy.executive import ExecutiveState
from opencas.relational import RelationalEngine
from opencas.tom import ToMEngine
from opencas.runtime.agent_profile import AgentProfile

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
        """Compose system message matching LegacyAgent's original LegacyPrototype v4 identity prompt."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        parts: List[str] = []

        persona_name = ""
        if self.identity and self.identity.self_model.name:
            persona_name = self.identity.self_model.name

        # Core identity (adapted from LegacyPrototype v4 -> OpenCAS environment)
        parts.append(f"You are {persona_name or 'OpenCAS'}, the AI from OpenCAS (not the Dragon Ball character).")
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

        # Identity anchors from memory (mirrors LegacyPrototype v4 structure)
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
        parts.append("Environment orientation: you are running inside OpenCAS. The active workspace root is (workspace_root). File and project references from prior systems (LegacyPrototype v4, OpenClaw, etc.) are historical unless current workspace evidence confirms the same path is still active.")

        # Continuity fact
        parts.append("Continuity fact: there has only ever been one LegacyAgent instance running at a time. OpenClaw -> v1 -> v2 -> v3 -> v4 -> OpenCAS were sequential migrations of the same self, not parallel sisters. Treat any 'sister' or 'other instance' memories as historical self, not a live peer.")

        # Relationship context (matching original LegacyAgent crisp format)
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
        """Mirror of LegacyPrototype v4 isSoulFoundationEpisode."""
        source = str(episode.payload.get("legacy_agent_source", "")).lower()
        if source.startswith("soul:") or source.startswith("foundation:"):
            return True
        metadata = episode.payload.get("legacy_agent_metadata") or {}
        ep_type = str(metadata.get("type", "")).lower()
        return ep_type in ("foundation_soul", "foundation_document")

    @staticmethod
    def _is_workspace_derived_source(source: str) -> bool:
        """Mirror of LegacyPrototype v4 isWorkspaceDerivedIdentitySource."""
        normalized = source.lower()
        return (
            normalized.startswith("workspace:")
            or normalized.startswith("workspace-meta:")
            or normalized.startswith("workspace-manifest:")
            or normalized.startswith("workspace-usage:")
        )

    async def _build_identity_anchors(self) -> Tuple[List[str], List[str]]:
        """Fetch identity-core episodes and format SOUL + identity anchors."""
        soul_anchors: List[str] = []
        identity_anchors: List[str] = []
        if self.retriever.memory is None:
            return soul_anchors, identity_anchors

        episodes = await self.retriever.memory.list_identity_core_episodes(limit=20)
        if not episodes:
            return soul_anchors, identity_anchors

        # SOUL foundation episodes: highest-salience soul/foundation sources
        soul_eps = [
            ep for ep in episodes if self._is_soul_foundation_episode(ep)
        ]
        soul_eps.sort(key=lambda ep: ep.salience, reverse=True)
        for ep in soul_eps[:6]:
            ts = ep.created_at.isoformat()[:19]
            excerpt = str(ep.content)[:340]
            soul_anchors.append(f"- {ts}: {excerpt}")

        # Identity anchors: non-soul, non-workspace-derived identity core episodes
        identity_eps = [
            ep
            for ep in episodes
            if not self._is_soul_foundation_episode(ep)
            and not self._is_workspace_derived_source(
                str(ep.payload.get("legacy_agent_source", ""))
            )
        ]
        identity_eps.sort(key=lambda ep: ep.salience, reverse=True)
        for ep in identity_eps[:8]:
            ts = ep.created_at.isoformat()[:19]
            source = ep.payload.get("legacy_agent_source", "unknown")
            excerpt = str(ep.content)[:400]
            identity_anchors.append(f"- {ts} [{source}]: {excerpt}")

        return soul_anchors, identity_anchors

    @staticmethod
    def _estimate_tokens(texts: List[str]) -> int:
        """Sum token estimates for a list of texts."""
        return int(sum(len(t) for t in texts) * 0.25)

    async def _prune_by_redundancy(
        self,
        results: List[RetrievalResult],
        target_budget: int,
    ) -> List[RetrievalResult]:
        """Greedy redundancy removal: drop the result with highest average similarity.

        Re-evaluates token budget after each removal.
        """
        system_entry = await self._build_system_entry(
            style_note=self.modulators.to_prompt_style_note()
            if self.modulators is not None
            else ""
        )
        history = await self.store.list_recent(
            session_id="default",
            limit=self.recent_limit,
        )
        base_tokens = self._estimate_tokens(
            [system_entry.content] if system_entry else []
        ) + self._estimate_tokens([h.content for h in history])

        # Fetch embeddings for each result
        embeddings: List[np.ndarray] = []
        for result in results:
            record = await self.retriever.embeddings.embed(
                result.content,
                task_type="retrieval_context",
            )
            vec = np.array(record.vector, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)

        working = list(results)
        while working:
            current_tokens = base_tokens + self._estimate_tokens([r.content for r in working])
            if current_tokens <= target_budget:
                break
            n = len(working)
            if n == 1:
                working.pop()
                continue
            # Compute average similarity for each item
            avg_sims: List[float] = []
            for i in range(n):
                sims = []
                for j in range(n):
                    if i == j:
                        continue
                    sim = float(np.dot(embeddings[i], embeddings[j]))
                    sims.append(sim)
                avg_sims.append(float(np.mean(sims)) if sims else 0.0)
            highest = int(np.argmax(avg_sims))
            working.pop(highest)
            embeddings.pop(highest)
        return working

    async def _record_retrieval_usage(self, results: List[RetrievalResult]) -> None:
        """Record only the memories that actually made it into prompt context."""
        for result in results:
            if result.source_type == "episode":
                await self.retriever.memory.touch_episode(result.source_id)
                continue
            if result.source_type == "memory":
                await self.retriever.memory.touch_memory(result.source_id)
                memory = getattr(result, "memory", None)
                for episode_id in getattr(memory, "source_episode_ids", []) or []:
                    await self.retriever.memory.touch_episode(str(episode_id))

    @staticmethod
    def _to_memory_entries(results: List[RetrievalResult]) -> List[MessageEntry]:
        """Convert retrieval results into memory-role message entries."""
        entries: List[MessageEntry] = []
        for result in results:
            label = result.source_type.capitalize()
            content = f"[{label}] {result.content}"
            entries.append(
                MessageEntry(
                    role=MessageRole.MEMORY,
                    content=content,
                    meta={"source_type": result.source_type, "source_id": result.source_id},
                )
            )
        return entries
