"""Compose deterministic session anchors and compact autobiographical gists."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from opencas.identity.agent_name import resolve_agent_name
from opencas.memory.models import Episode, EpisodeKind

from .models import SessionAnchor
from .store import SessionAnchorStore

_HIGH_ACTION_TOOL_PREFIXES = (
    "fs_write_file",
    "edit_file",
    "workflow_create_",
    "workflow_update_",
    "process_start",
    "pty_",
    "browser_",
)
_RECALL_RECOVERY_TOOLS = {"search_memories", "recall_concepts", "recall_autobiography", "artifact_lookup"}
_EPISODE_CITATION_RE = re.compile(r"\[ep:(?P<episode_id>[A-Za-z0-9_.:-]+)\]")
_GIST_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class SessionAutobiographyComposer:
    """Build session autobiography from existing durable records."""

    def __init__(
        self,
        *,
        memory_store: Any,
        anchor_store: SessionAnchorStore,
        context_store: Any = None,
        identity: Any = None,
        commitment_store: Any = None,
        task_store: Any = None,
        schedule_store: Any = None,
        llm: Any = None,
    ) -> None:
        self.memory_store = memory_store
        self.anchor_store = anchor_store
        self.context_store = context_store
        self.identity = identity
        self.commitment_store = commitment_store
        self.task_store = task_store
        self.schedule_store = schedule_store
        self.llm = llm

    async def compose_skeleton(self, session_id: str) -> SessionAnchor:
        episodes = await self.memory_store.list_episodes_for_session(
            session_id,
            include_compacted=True,
            limit=500,
        )
        now = datetime.now(timezone.utc)
        created_at = episodes[0].created_at if episodes else now
        ended_at = episodes[-1].created_at if episodes else None
        duration_s = (
            max(0.0, (ended_at - created_at).total_seconds())
            if ended_at is not None
            else None
        )
        episode_ids = [str(episode.episode_id) for episode in episodes]
        compaction_ids = await self._compaction_ids_for_episode_ids(episode_ids)
        narrative_ids = await self._narrative_bridge_message_ids(session_id)
        continuity_ids = self._continuity_breadcrumb_ids(created_at, ended_at)
        commitments_opened, commitments_closed = await self._commitment_deltas(created_at, ended_at)
        affect_peaks = self._affect_peaks(episodes)
        artifact_paths = self._artifact_paths(episodes)
        identity_mutagen_ids = [
            str(episode.episode_id) for episode in episodes if episode.identity_mutagen
        ]
        recall_failure_ids = [
            str(episode.episode_id)
            for episode in episodes
            if self._is_prior_recall_failure_episode(episode)
        ]
        recall_recovery_ids = [
            str(episode.episode_id)
            for episode in episodes
            if self._is_recall_recovery_episode(episode)
        ]
        anchor = SessionAnchor(
            session_id=session_id,
            created_at=created_at,
            ended_at=ended_at,
            duration_s=duration_s,
            agent_name=self._agent_name(),
            source_system=self._source_system(),
            episode_kind_tally=self._kind_tally(episodes),
            decision_episode_ids=self._decision_episode_ids(episodes),
            affect_peaks=affect_peaks,
            recorded_affect_summary=self._recorded_affect_summary(affect_peaks),
            artifact_paths_touched=artifact_paths,
            commitments_opened=commitments_opened,
            commitments_closed=commitments_closed,
            identity_mutagen_episode_ids=identity_mutagen_ids,
            compaction_record_ids=compaction_ids,
            narrative_bridge_message_ids=narrative_ids,
            continuity_breadcrumb_ids=continuity_ids,
            evidence_episode_ids=episode_ids,
            recall_failure_episode_ids=recall_failure_ids,
            recall_recovery_episode_ids=recall_recovery_ids,
            evidence_strength=self._evidence_strength(episodes, artifact_paths, identity_mutagen_ids),
            gaps_noted=self._gaps_noted(episodes, compaction_ids),
            consolidation_run_id_at_time=None,
        )
        anchor.evidence_hash = self.evidence_hash(anchor)
        return anchor

    async def compose_gist(self, session_id: str) -> str:
        current = await self.compose_skeleton(session_id)
        await self.anchor_store.upsert(current)
        anchor = await self.anchor_store.get(session_id) or current
        if anchor.gist and anchor.evidence_hash == current.evidence_hash and anchor.gist_version == 1:
            return anchor.gist

        if anchor.evidence_strength == "low":
            gist = self._deterministic_gist(anchor)
            await self.anchor_store.update_gist(
                session_id,
                gist,
                "low",
                anchor.evidence_hash or self.evidence_hash(anchor),
                gist_version=0,
            )
            return gist

        gist = await self._llm_gist(anchor)
        confidence = "high" if anchor.evidence_strength == "high" else "medium"
        await self.anchor_store.update_gist(
            session_id,
            gist,
            confidence,
            anchor.evidence_hash or self.evidence_hash(anchor),
            gist_version=1,
        )
        return gist

    async def _llm_gist(self, anchor: SessionAnchor) -> str:
        if self.llm is None or not hasattr(self.llm, "chat_completion"):
            return self._deterministic_gist(anchor)
        episodes = await self.memory_store.get_episodes_by_ids(anchor.evidence_episode_ids[:40])
        bridges = await self._narrative_bridge_texts(anchor.session_id)
        decisions = [
            self._compact(episode.content, 200)
            for episode in episodes
            if str(episode.episode_id) in set(anchor.decision_episode_ids)
        ][:8]
        user_block = {
            "episode_kind_tally": anchor.episode_kind_tally,
            "affect_peaks": anchor.affect_peaks,
            "recorded_affect_summary": anchor.recorded_affect_summary,
            "decision_episodes": decisions,
            "artifacts_touched": anchor.artifact_paths_touched[:12],
            "commitments_opened": anchor.commitments_opened[:8],
            "commitments_closed": anchor.commitments_closed[:8],
            "identity_mutagen_episode_ids": anchor.identity_mutagen_episode_ids[:8],
            "compaction_record_ids": anchor.compaction_record_ids[:8],
            "prior_recall_failures": len(anchor.recall_failure_episode_ids),
            "recall_recovery_actions": len(anchor.recall_recovery_episode_ids),
            "evidence_episodes": [
                {
                    "episode_id": str(episode.episode_id),
                    "kind": episode.kind.value,
                    "excerpt": self._compact(episode.content, 220),
                }
                for episode in episodes[:20]
            ],
        }
        bridge_text = "\n".join(bridges[:2])
        prompt_parts = []
        if bridge_text:
            prompt_parts.append(f"Earlier first-person reflection from this session: {bridge_text}")
        prompt_parts.append(
            "Affect trajectory during session: "
            f"{anchor.recorded_affect_summary or 'no recorded affect peaks'}"
        )
        prompt_parts.append(json.dumps(user_block, sort_keys=True))
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are {anchor.agent_name}, writing a compact autobiographical "
                    "summary of a past session. Write in first person. Be emotionally "
                    "honest but not melodramatic. State what you did, what mattered, "
                    "how recorded affect shaped the session, and what carries forward. "
                    "2-4 sentences max. Target 100-250 tokens. Do not pad or repeat. "
                    "Every factual sentence must include at least one inline citation "
                    "using exactly [ep:<episode_id>] from the provided evidence_episodes. "
                    "Do not invent citations. Sentences without valid episode citations "
                    "will be discarded. Every sentence must carry new information."
                ),
            },
            {"role": "user", "content": "\n".join(prompt_parts)},
        ]
        try:
            response = await self.llm.chat_completion(
                messages,
                complexity="standard",
                source="autobiography_gist",
            )
            choices = response.get("choices", []) if isinstance(response, dict) else []
            if choices:
                content = choices[0].get("message", {}).get("content", "").strip()
                if content:
                    validated = self._validate_cited_gist(content, anchor)
                    if validated:
                        return self._compact(validated, 1200)
        except Exception:
            pass
        return self._deterministic_gist(anchor)

    def _validate_cited_gist(self, content: str, anchor: SessionAnchor) -> str:
        evidence_ids = {str(item) for item in anchor.evidence_episode_ids}
        kept: list[str] = []
        for sentence in self._split_gist_sentences(content):
            citations = self._sentence_episode_citations(sentence)
            if citations and any(citation in evidence_ids for citation in citations):
                kept.append(sentence)
        return " ".join(kept).strip()

    @staticmethod
    def _split_gist_sentences(content: str) -> list[str]:
        pieces = [part.strip() for part in _GIST_SENTENCE_RE.split(str(content or "").strip())]
        return [part for part in pieces if part]

    @staticmethod
    def _sentence_episode_citations(sentence: str) -> list[str]:
        return [
            match.group("episode_id").strip().strip(".,;:)]}")
            for match in _EPISODE_CITATION_RE.finditer(sentence)
            if match.group("episode_id")
        ]

    def _deterministic_gist(self, anchor: SessionAnchor) -> str:
        date = anchor.created_at.date().isoformat()
        duration = (
            f"{int(anchor.duration_s)}s"
            if anchor.duration_s is not None
            else "unknown duration"
        )
        if anchor.evidence_strength == "low":
            return (
                f"Session on {date} ({duration}). Primarily operator mentions. "
                f"Episode tally: {anchor.episode_kind_tally}."
            )
        artifacts = len(anchor.artifact_paths_touched)
        return (
            f"Session {anchor.session_id} on {date} ({duration}). "
            f"Episode tally: {anchor.episode_kind_tally}. {artifacts} artifacts touched."
        )

    def evidence_hash(self, anchor: SessionAnchor) -> str:
        payload = {
            "episode_kind_tally": anchor.episode_kind_tally,
            "decision_episode_ids": anchor.decision_episode_ids,
            "affect_peaks": anchor.affect_peaks,
            "artifact_paths_touched": anchor.artifact_paths_touched,
            "commitments_opened": anchor.commitments_opened,
            "commitments_closed": anchor.commitments_closed,
            "identity_mutagen_episode_ids": anchor.identity_mutagen_episode_ids,
            "compaction_record_ids": anchor.compaction_record_ids,
            "narrative_bridge_message_ids": anchor.narrative_bridge_message_ids,
            "continuity_breadcrumb_ids": anchor.continuity_breadcrumb_ids,
            "evidence_episode_ids": anchor.evidence_episode_ids,
            "recall_failure_episode_ids": anchor.recall_failure_episode_ids,
            "recall_recovery_episode_ids": anchor.recall_recovery_episode_ids,
            "gaps_noted": anchor.gaps_noted,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def _compaction_ids_for_episode_ids(self, episode_ids: list[str]) -> list[str]:
        if not episode_ids or getattr(self.memory_store, "_db", None) is None:
            return []
        wanted = set(episode_ids)
        cursor = await self.memory_store._db.execute("SELECT * FROM compactions")
        rows = await cursor.fetchall()
        found: list[str] = []
        for row in rows:
            raw = row["episode_ids"]
            try:
                parsed = json.loads(raw) if raw else []
            except json.JSONDecodeError:
                parsed = []
            if isinstance(parsed, dict):
                parsed = parsed.get("episode_ids") or []
            if wanted & {str(item) for item in parsed}:
                found.append(str(row["compaction_id"]))
        return found

    async def _narrative_bridge_message_ids(self, session_id: str) -> list[str]:
        rows = await self._narrative_bridge_rows(session_id)
        return [str(row["message_id"]) for row in rows if "message_id" in row.keys()]

    async def _narrative_bridge_texts(self, session_id: str) -> list[str]:
        rows = await self._narrative_bridge_rows(session_id)
        return [str(row["content"]) for row in rows if "content" in row.keys()]

    async def _narrative_bridge_rows(self, session_id: str) -> list[Any]:
        db = getattr(self.context_store, "_db", None)
        if db is None:
            return []
        try:
            cursor = await db.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ? AND meta LIKE '%compaction_narrative_bridge%'
                ORDER BY created_at ASC
                """,
                (session_id,),
            )
            return list(await cursor.fetchall())
        except Exception:
            return []

    def _continuity_breadcrumb_ids(
        self,
        start: datetime,
        end: Optional[datetime],
    ) -> list[str]:
        continuity = getattr(self.identity, "continuity", None)
        breadcrumbs = getattr(continuity, "continuity_breadcrumbs", None)
        if not isinstance(breadcrumbs, list):
            return []
        end = end or start
        buffered_end = end + timedelta(minutes=5)
        ids: list[str] = []
        for idx, breadcrumb in enumerate(breadcrumbs):
            ts = breadcrumb.get("timestamp") if isinstance(breadcrumb, dict) else None
            try:
                when = datetime.fromisoformat(str(ts))
            except Exception:
                continue
            if start <= when <= buffered_end:
                ids.append(str(breadcrumb.get("breadcrumb_id") or idx))
        return ids

    async def _commitment_deltas(
        self,
        start: datetime,
        end: Optional[datetime],
    ) -> tuple[list[str], list[str]]:
        store = self.commitment_store
        if store is None or not callable(getattr(store, "list_all", None)):
            return [], []
        end = end or start
        try:
            commitments = await store.list_all(limit=500)
        except Exception:
            return [], []
        opened: list[str] = []
        closed: list[str] = []
        for commitment in commitments or []:
            cid = str(getattr(commitment, "commitment_id", "") or "")
            created = getattr(commitment, "created_at", None)
            updated = getattr(commitment, "updated_at", None)
            status = str(getattr(commitment, "status", "") or "").lower()
            if isinstance(created, datetime) and start <= created <= end:
                opened.append(cid)
            if isinstance(updated, datetime) and start <= updated <= end and status in {
                "completed",
                "abandoned",
                "cancelled",
                "canceled",
            }:
                closed.append(cid)
        return opened, closed

    def _agent_name(self) -> str:
        try:
            return resolve_agent_name(identity=self.identity)
        except Exception:
            return "OpenCAS"

    def _source_system(self) -> Optional[str]:
        self_model = getattr(self.identity, "self_model", None)
        value = getattr(self_model, "source_system", None)
        return str(value) if value else None

    @staticmethod
    def _kind_tally(episodes: Iterable[Episode]) -> dict[str, int]:
        tally: dict[str, int] = {}
        for episode in episodes:
            key = episode.kind.value
            tally[key] = tally.get(key, 0) + 1
        return tally

    def _decision_episode_ids(self, episodes: list[Episode]) -> list[str]:
        ids: list[str] = []
        for idx, episode in enumerate(episodes):
            if episode.identity_mutagen:
                ids.append(str(episode.episode_id))
                continue
            if episode.kind is EpisodeKind.ACTION:
                ids.append(str(episode.episode_id))
                continue
            if episode.kind is EpisodeKind.TURN and self._turn_role(episode) == "user":
                text = episode.content.lower()
                next_is_action = idx + 1 < len(episodes) and episodes[idx + 1].kind is EpisodeKind.ACTION
                if next_is_action and ("?" in text or re.search(r"\b(please|do|make|create|fix|write|revise)\b", text)):
                    ids.append(str(episode.episode_id))
        return ids[:20]

    @staticmethod
    def _affect_peaks(episodes: list[Episode]) -> list[dict[str, Any]]:
        affected = [episode for episode in episodes if episode.affect is not None]
        affected.sort(key=lambda episode: float(getattr(episode.affect, "intensity", 0.0) or 0.0), reverse=True)
        peaks: list[dict[str, Any]] = []
        for episode in affected[:3]:
            affect = episode.affect
            primary = getattr(getattr(affect, "primary_emotion", None), "value", None)
            peaks.append(
                {
                    "episode_id": str(episode.episode_id),
                    "valence": getattr(affect, "valence", None),
                    "intensity": getattr(affect, "intensity", None),
                    "primary_emotion": primary or str(getattr(affect, "primary_emotion", "")),
                }
            )
        return peaks

    @staticmethod
    def _recorded_affect_summary(affect_peaks: list[dict[str, Any]]) -> Optional[str]:
        if not affect_peaks:
            return None
        strongest = affect_peaks[0]
        emotion = strongest.get("primary_emotion") or "unknown"
        intensity = strongest.get("intensity")
        valence = strongest.get("valence")
        return f"peaked {emotion} with intensity {intensity} and valence {valence}"

    def _artifact_paths(self, episodes: list[Episode]) -> list[str]:
        paths: list[str] = []
        for episode in episodes:
            if episode.kind is not EpisodeKind.ACTION:
                continue
            path = self._path_from_payload(episode) or self._path_from_content(episode.content)
            if path:
                normalized = self._normalize_workspace_path(path)
                if normalized and normalized not in paths:
                    paths.append(normalized)
        return paths

    @staticmethod
    def _path_from_payload(episode: Episode) -> str:
        payload = episode.payload or {}
        args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
        for key in ("file_path", "path", "output_path"):
            value = args.get(key) or payload.get(key)
            if value:
                return str(value)
        return ""

    @staticmethod
    def _path_from_content(content: str) -> str:
        for pattern in (
            r"\bpath=(?P<path>\S+)",
            r"\bfile_path=(?P<path>\S+)",
            r'"file_path"\s*:\s*"(?P<json_path>[^"]+)"',
            r'"path"\s*:\s*"(?P<json_path>[^"]+)"',
        ):
            match = re.search(pattern, content)
            if match:
                return (match.groupdict().get("path") or match.groupdict().get("json_path") or "").strip().strip("`'\",)")
        return ""

    @staticmethod
    def _normalize_workspace_path(path: str) -> str:
        text = path.strip()
        marker = "/workspace/"
        if marker in text:
            return "workspace/" + text.split(marker, 1)[1].strip("/")
        if text.startswith("workspace/"):
            return text
        try:
            resolved = Path(text).expanduser()
            parts = resolved.parts
            if "workspace" in parts:
                idx = parts.index("workspace")
                return "/".join(parts[idx:])
        except Exception:
            pass
        return text

    def _evidence_strength(
        self,
        episodes: list[Episode],
        artifact_paths: list[str],
        identity_mutagen_ids: list[str],
    ) -> str:
        if artifact_paths or identity_mutagen_ids:
            return "high"
        for episode in episodes:
            if episode.kind is EpisodeKind.ACTION:
                tool = self._tool_name(episode)
                if any(tool.startswith(prefix) for prefix in _HIGH_ACTION_TOOL_PREFIXES):
                    return "high"
        for episode in episodes:
            if episode.kind is EpisodeKind.PROCEDURAL:
                return "medium"
            if episode.kind is EpisodeKind.TURN and self._turn_role(episode) == "assistant":
                return "medium"
        return "low"

    def _gaps_noted(self, episodes: list[Episode], compaction_ids: list[str]) -> list[str]:
        if compaction_ids:
            return []
        gaps: list[str] = []
        ordered = sorted(episodes, key=lambda episode: episode.created_at)
        for previous, current in zip(ordered, ordered[1:]):
            delta = current.created_at - previous.created_at
            if delta > timedelta(minutes=30):
                gaps.append(
                    "no episodes between "
                    f"{previous.created_at.isoformat(timespec='minutes')} and "
                    f"{current.created_at.isoformat(timespec='minutes')}"
                )
        return gaps

    def _is_prior_recall_failure_episode(self, episode: Episode) -> bool:
        if episode.kind is not EpisodeKind.TURN or self._turn_role(episode) != "assistant":
            return False
        text = episode.content.lower()
        return bool(
            re.search(
                r"\b(i\s+don'?t\s+(?:remember|recognize)|i\s+only\s+retrieve|"
                r"no\s+evidence\s+in\s+this\s+conversation|"
                r"don'?t\s+have\s+continuous\s+autobiographical\s+memory|"
                r"can'?t\s+say\s+i\s+remember)",
                text,
            )
        )

    def _is_recall_recovery_episode(self, episode: Episode) -> bool:
        if episode.kind is not EpisodeKind.ACTION:
            return False
        return self._tool_name(episode) in _RECALL_RECOVERY_TOOLS

    @staticmethod
    def _tool_name(episode: Episode) -> str:
        payload = episode.payload or {}
        value = str(payload.get("tool_name") or payload.get("name") or "").strip()
        if value:
            return value
        match = re.match(r"^tool\s+([A-Za-z0-9_:-]+)", episode.content or "")
        return match.group(1).rstrip(":") if match else ""

    @staticmethod
    def _turn_role(episode: Episode) -> str:
        payload = episode.payload or {}
        return str(payload.get("role") or "").strip().lower()

    @staticmethod
    def _compact(text: str, limit: int) -> str:
        compact = " ".join(str(text or "").split())
        if len(compact) > limit:
            return compact[: limit - 3].rstrip() + "..."
        return compact
