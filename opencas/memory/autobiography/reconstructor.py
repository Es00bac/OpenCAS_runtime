"""Answer-shaped autobiographical recall across session anchors."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Optional

from opencas.memory.models import Episode, EpisodeKind

from .models import (
    AutobiographyRecallResult,
    EvidenceItem,
    NextBestLookup,
    SessionAnchor,
    SessionRef,
)
from .store import SessionAnchorStore

_EVIDENCE_ORDER = {
    "autobiographical": 0,
    "artifact": 1,
    "procedural": 2,
    "assistant_turn": 3,
    "mention_only": 4,
}


class AutobiographyReconstructor:
    """Reconstruct concise autobiography from session anchors."""

    def __init__(
        self,
        *,
        anchor_store: SessionAnchorStore,
        composer: Any,
        memory_store: Any,
    ) -> None:
        self.anchor_store = anchor_store
        self.composer = composer
        self.memory_store = memory_store

    async def recall(
        self,
        *,
        query: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        max_tokens: int = 1500,
        lazy_fill: bool = True,
    ) -> AutobiographyRecallResult:
        since = since or datetime(1970, 1, 1, tzinfo=timezone.utc)
        anchors = await self.anchor_store.list_recent(since, limit=200)
        if until is not None:
            anchors = [anchor for anchor in anchors if anchor.created_at <= until]
        episode_text_by_anchor = await self._episode_text_by_anchor(anchors)
        matched = self._match_anchors(query, anchors, episode_text_by_anchor)
        if lazy_fill:
            filled = 0
            for anchor in matched:
                if filled >= 3:
                    break
                if anchor.gist_version == 0 and anchor.evidence_strength in {"high", "medium"}:
                    await self.composer.compose_gist(anchor.session_id)
                    refreshed = await self.anchor_store.get(anchor.session_id)
                    if refreshed is not None:
                        index = matched.index(anchor)
                        matched[index] = refreshed
                    filled += 1

        if not matched:
            return AutobiographyRecallResult(
                query=query,
                essence="I do not have anchored autobiographical evidence for that yet.",
                confidence="insufficient",
                evidence_scope="insufficient",
                next_best_lookup=[
                    NextBestLookup(
                        tool="search_memories",
                        args={"query": query, "limit": 10},
                        reason="raw episodic search may find unanchored fragments",
                    )
                ],
            )

        gaps = self._gaps(matched)
        project_arc = self._project_arc(query, matched)
        evidence = self._prioritize_project_evidence(
            await self._strongest_evidence(matched, query=query),
            project_arc=project_arc,
        )
        essence = self._essence(query, matched, evidence, project_arc)
        confidence = self._confidence(matched, evidence)
        scope = self._scope(matched, evidence)
        result = AutobiographyRecallResult(
            query=query,
            essence=essence,
            confidence=confidence,
            evidence_scope=scope,
            strongest_evidence=evidence[:5],
            gaps=gaps,
            next_best_lookup=self._next_best_lookup(query, matched, scope),
            session_refs=[
                SessionRef(
                    session_id=anchor.session_id,
                    date=anchor.created_at.isoformat(timespec="seconds"),
                    duration_s=anchor.duration_s,
                )
                for anchor in matched[:8]
            ],
            project_arc=project_arc,
        )
        return self._enforce_budget(result, max_tokens=max_tokens)

    def _match_anchors(
        self,
        query: str,
        anchors: list[SessionAnchor],
        episode_text_by_anchor: dict[str, str] | None = None,
    ) -> list[SessionAnchor]:
        terms = self._terms(query)
        distinctive_terms = self._distinctive_terms(terms)
        path_candidates = self._query_workspace_paths(query)
        scored: list[tuple[int, SessionAnchor]] = []
        exact_path_matches: list[tuple[int, SessionAnchor]] = []
        episode_text_by_anchor = episode_text_by_anchor or {}
        for anchor in anchors:
            blob = " ".join(
                [
                    anchor.session_id,
                    anchor.gist or "",
                    " ".join(anchor.artifact_paths_touched),
                    " ".join(anchor.gaps_noted),
                    anchor.recorded_affect_summary or "",
                    episode_text_by_anchor.get(anchor.session_id, ""),
                ]
            ).lower()
            normalized_blob = re.sub(r"[^a-z0-9]+", " ", blob)
            if distinctive_terms and not any(term in normalized_blob for term in distinctive_terms):
                continue
            score = sum(2 for term in terms if term in normalized_blob)
            for path in anchor.artifact_paths_touched:
                normalized_path = re.sub(r"[^a-z0-9]+", " ", path.lower())
                score += sum(3 for term in terms if term in normalized_path)
                if self._path_matches_query(path, path_candidates):
                    score += 100
                    exact_path_matches.append((score, anchor))
                elif query.strip().lower() in path.lower():
                    score += 10
            if score > 0:
                scored.append((score, anchor))
        if exact_path_matches:
            exact_path_matches.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
            return [anchor for _score, anchor in exact_path_matches[:10]]
        if not scored:
            return []
        scored.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        return [anchor for _score, anchor in scored[:10]]

    async def _episode_text_by_anchor(
        self,
        anchors: list[SessionAnchor],
    ) -> dict[str, str]:
        id_to_session: dict[str, str] = {}
        ids: list[str] = []
        for anchor in anchors:
            for episode_id in anchor.evidence_episode_ids[:40]:
                if episode_id not in id_to_session:
                    ids.append(episode_id)
                id_to_session[episode_id] = anchor.session_id
        if not ids:
            return {}
        episodes = await self.memory_store.get_episodes_by_ids(ids)
        by_session: dict[str, list[str]] = {}
        for episode in episodes:
            session_id = id_to_session.get(str(episode.episode_id))
            if not session_id:
                continue
            by_session.setdefault(session_id, []).append(self._compact(episode.content, 240))
        return {session_id: " ".join(parts) for session_id, parts in by_session.items()}

    async def _strongest_evidence(
        self,
        anchors: list[SessionAnchor],
        *,
        query: str = "",
    ) -> list[EvidenceItem]:
        ids: list[str] = []
        for anchor in anchors:
            ids.extend(anchor.evidence_episode_ids[:40])
        episodes = await self.memory_store.get_episodes_by_ids(ids)
        items = [self._evidence_item(episode) for episode in episodes]
        terms = self._terms(query)
        distinctive_terms = self._distinctive_terms(terms)
        if distinctive_terms:
            filtered = [
                item
                for item in items
                if self._evidence_item_matches_terms(item, distinctive_terms)
            ]
            items = filtered
        items.sort(key=lambda item: (_EVIDENCE_ORDER.get(item.kind, 99), self._evidence_item_rank(item), item.timestamp))
        return items

    def _prioritize_project_evidence(
        self,
        evidence: list[EvidenceItem],
        *,
        project_arc: dict[str, Any] | None,
    ) -> list[EvidenceItem]:
        if not evidence:
            return []
        project_sessions = {
            str(session_id)
            for session_id in (project_arc or {}).get("session_ids", [])
            if str(session_id or "").strip()
        }
        indexed = list(enumerate(evidence))
        if project_sessions:
            indexed.sort(
                key=lambda item: (
                    0 if item[1].session_id in project_sessions else 1,
                    item[0],
                )
            )
        return self._diversify_evidence([item for _index, item in indexed])

    @staticmethod
    def _diversify_evidence(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        selected: list[EvidenceItem] = []
        seen_sessions: set[str] = set()
        for item in evidence:
            session_id = str(item.session_id or "")
            if session_id and session_id in seen_sessions:
                continue
            if session_id:
                seen_sessions.add(session_id)
            selected.append(item)
            if len(selected) >= 5:
                break
        for item in evidence:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= max(5, len(evidence)):
                break
        return selected

    def _evidence_item(self, episode: Episode) -> EvidenceItem:
        kind = self._evidence_kind(episode)
        label = self._label(episode, kind)
        return EvidenceItem(
            timestamp=episode.created_at.isoformat(timespec="seconds"),
            kind=kind,
            label=label,
            excerpt=self._compact(episode.content, 120),
            session_id=episode.session_id,
        )

    def _evidence_kind(self, episode: Episode) -> str:
        if episode.kind is EpisodeKind.ACTION:
            return "autobiographical"
        if episode.kind is EpisodeKind.ARTIFACT:
            return "artifact"
        if episode.kind in {EpisodeKind.PROCEDURAL, EpisodeKind.CONSOLIDATION, EpisodeKind.COMPACTION}:
            return "procedural"
        if episode.kind is EpisodeKind.TURN and self._turn_role(episode) == "assistant":
            return "assistant_turn"
        return "mention_only"

    def _label(self, episode: Episode, kind: str) -> str:
        if kind == "autobiographical":
            tool = self._tool_name(episode)
            path = self.composer._path_from_payload(episode) or self.composer._path_from_content(episode.content)
            if path:
                return f"{tool or 'tool'} touched {PurePosixPath(path).name}"
            return tool or "self action"
        if kind == "assistant_turn":
            if self.composer._is_prior_recall_failure_episode(episode):
                return "previous recall gap"
            return "assistant remembered or described prior work"
        if kind == "mention_only":
            return "operator mention"
        return kind.replace("_", " ")

    @staticmethod
    def _evidence_item_rank(item: EvidenceItem) -> int:
        label = item.label.lower()
        excerpt = item.excerpt.lower()
        blob = f"{label} {excerpt}"
        if item.kind == "autobiographical":
            if any(tool in blob for tool in ("fs_write_file", "edit_file")):
                return 0
            if any(
                tool in blob
                for tool in (
                    "workflow_create",
                    "workflow_update",
                    "process_start",
                    "pty_",
                    "browser_",
                )
            ):
                return 1
            if any(tool in blob for tool in ("artifact_lookup", "recall_autobiography")):
                return 2
            if any(tool in blob for tool in ("search_memories", "recall_concepts", "fs_read_file", "grep_search", "glob_search")):
                return 3
        return 5

    @staticmethod
    def _evidence_item_matches_terms(
        item: EvidenceItem,
        terms: list[str],
    ) -> bool:
        blob = re.sub(
            r"[^a-z0-9]+",
            " ",
            f"{item.kind} {item.label} {item.excerpt}".lower(),
        )
        return any(term in blob for term in terms)

    def _gaps(self, anchors: list[SessionAnchor]) -> list[str]:
        gaps: list[str] = []
        for anchor in anchors:
            for gap in anchor.gaps_noted:
                if gap not in gaps:
                    gaps.append(gap)
            if anchor.recall_failure_episode_ids:
                note = (
                    f"previous failure to recollect in session {anchor.session_id}; "
                    "treat it as a remembered recall gap when self-evidence is present"
                )
                if note not in gaps:
                    gaps.append(note)
        return gaps[:6]

    def _essence(
        self,
        query: str,
        anchors: list[SessionAnchor],
        evidence: list[EvidenceItem],
        project_arc: dict[str, Any] | None = None,
    ) -> str:
        strongest = next((item for item in evidence if item.kind == "autobiographical"), None)
        if project_arc and project_arc.get("session_count", 0) > 1:
            return (
                f"I can reconstruct work on {project_arc['path']} across {project_arc['session_count']} sessions; "
                f"the strongest evidence is {strongest.label if strongest else 'anchored session continuity'}."
            )
        best_anchor = anchors[0]
        if best_anchor.gist:
            return self._compact(best_anchor.gist, 520)
        if strongest:
            return f"I can reconstruct this from stored autobiographical records; strongest evidence: {strongest.label}."
        if best_anchor.recall_failure_episode_ids:
            return (
                "I can see a prior recall gap about this and the stored records around it; "
                "that gap is context for reconstruction, not a permanent defect."
            )
        return "I found anchored session evidence, but it is weak and needs a raw memory or artifact lookup for detail."

    def _confidence(self, anchors: list[SessionAnchor], evidence: list[EvidenceItem]) -> str:
        if any(anchor.evidence_strength == "high" for anchor in anchors):
            return "high"
        if any(anchor.evidence_strength == "medium" for anchor in anchors):
            return "medium"
        if evidence:
            return "low"
        return "insufficient"

    def _scope(self, anchors: list[SessionAnchor], evidence: list[EvidenceItem]) -> str:
        kinds = {item.kind for item in evidence}
        if "autobiographical" in kinds:
            return "autobiographical"
        if {"assistant_turn", "mention_only"} & kinds:
            return "mixed" if "assistant_turn" in kinds else "mention_only"
        if anchors:
            return "mention_only"
        return "insufficient"

    def _next_best_lookup(
        self,
        query: str,
        anchors: list[SessionAnchor],
        scope: str,
    ) -> list[NextBestLookup]:
        if scope == "insufficient":
            return [
                NextBestLookup(
                    tool="search_memories",
                    args={"query": query, "limit": 10},
                    reason="raw semantic memory may contain unanchored episodes",
                )
            ]
        paths = [path for anchor in anchors for path in anchor.artifact_paths_touched]
        if paths:
            return [
                NextBestLookup(
                    tool="artifact_lookup",
                    args={"path": paths[0]},
                    reason=f"path-specific provenance for {PurePosixPath(paths[0]).name}",
                )
            ]
        return []

    def _project_arc_path(self, query: str, anchors: list[SessionAnchor]) -> str:
        arc = self._project_arc(query, anchors)
        return str(arc.get("path") or "") if arc else ""

    def _project_arc(self, query: str, anchors: list[SessionAnchor]) -> dict[str, Any] | None:
        stats: dict[str, dict[str, Any]] = {}
        for anchor in anchors:
            for path in anchor.artifact_paths_touched:
                project = self._project_root(path)
                bucket = stats.setdefault(
                    project,
                    {"path": project, "session_ids": [], "artifact_paths": []},
                )
                if anchor.session_id not in bucket["session_ids"]:
                    bucket["session_ids"].append(anchor.session_id)
                if path not in bucket["artifact_paths"]:
                    bucket["artifact_paths"].append(path)
        if not stats:
            return None
        terms = self._terms(query)

        def score(item: tuple[str, dict[str, Any]]) -> tuple[int, int, int, str]:
            project, data = item
            blob = re.sub(r"[^a-z0-9]+", " ", project.lower())
            query_bonus = sum(100 for term in terms if term in blob)
            session_count = len(data["session_ids"])
            artifact_count = len(data["artifact_paths"])
            return query_bonus, session_count, artifact_count, project

        project_path, selected = max(stats.items(), key=score)
        query_bonus = score((project_path, selected))[0]
        if query_bonus <= 0 and not self._query_workspace_paths(query):
            return None
        session_ids = list(selected["session_ids"])
        artifact_paths = list(selected["artifact_paths"])
        if len(session_ids) <= 1 and len(artifact_paths) <= 1:
            return None
        return {
            "path": project_path,
            "session_count": len(session_ids),
            "session_ids": session_ids[:20],
            "artifact_paths": artifact_paths[:40],
        }

    @staticmethod
    def _project_root(path: str) -> str:
        parts = PurePosixPath(path).parts
        if "workspace" in parts:
            idx = parts.index("workspace")
            after_workspace = parts[idx + 1 :]
            if not after_workspace:
                return "workspace"
            if (
                after_workspace[0].lower() in {"writing"}
                and len(after_workspace) >= 2
                and "." not in after_workspace[1]
            ):
                root_parts = parts[idx : idx + 3]
            else:
                root_parts = parts[idx : idx + 2]
            return "/".join(root_parts)
        if len(parts) >= 2:
            return "/".join(parts[:-1])
        return path

    @staticmethod
    def _terms(query: str) -> list[str]:
        stop_terms = {"mnt", "xtra", "opencas", "workspace", "the", "and"}
        return [
            term
            for term in re.sub(r"[^a-z0-9]+", " ", query.lower()).split()
            if len(term) >= 2 and term not in stop_terms
        ]

    @staticmethod
    def _distinctive_terms(terms: list[str]) -> list[str]:
        generic = {
            "about",
            "available",
            "broke",
            "changed",
            "codex",
            "concise",
            "context",
            "did",
            "distinguish",
            "evidence",
            "especially",
            "happened",
            "here",
            "issue",
            "jarrod",
            "memory",
            "need",
            "needed",
            "not",
            "past",
            "please",
            "recall",
            "remember",
            "repair",
            "retrieved",
            "runtime",
            "session",
            "thread",
            "uncertainty",
            "use",
            "what",
            "when",
            "where",
            "with",
            "work",
            "worked",
            "you",
        }
        return [term for term in terms if term not in generic]

    @staticmethod
    def _query_workspace_paths(query: str) -> list[str]:
        text = str(query or "").strip().replace("\\", "/")
        candidates: list[str] = []
        marker = "/workspace/"
        if marker in text:
            candidates.append("workspace/" + text.split(marker, 1)[1].strip().strip("`'\".,;:)"))
        for match in re.findall(r"(workspace/[^\s`'\".,;:)]+)", text):
            candidate = match.strip("/")
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _path_matches_query(path: str, candidates: list[str]) -> bool:
        normalized = str(path or "").replace("\\", "/").strip("/")
        for candidate in candidates:
            candidate = candidate.strip("/")
            if normalized == candidate or normalized.endswith(candidate):
                return True
        return False

    @staticmethod
    def _tool_name(episode: Episode) -> str:
        payload = episode.payload or {}
        name = str(payload.get("tool_name") or payload.get("name") or "").strip()
        if name:
            return name
        match = re.match(r"^tool\s+([A-Za-z0-9_:-]+)", episode.content or "")
        return match.group(1).rstrip(":") if match else ""

    @staticmethod
    def _turn_role(episode: Episode) -> str:
        return str((episode.payload or {}).get("role") or "").strip().lower()

    @staticmethod
    def _compact(text: str, limit: int) -> str:
        compact = " ".join(str(text or "").split())
        if len(compact) > limit:
            return compact[: limit - 3].rstrip() + "..."
        return compact

    @staticmethod
    def _estimate_tokens(result: AutobiographyRecallResult) -> int:
        return len(str(result.to_dict())) // 4

    def _enforce_budget(
        self,
        result: AutobiographyRecallResult,
        *,
        max_tokens: int,
    ) -> AutobiographyRecallResult:
        while self._estimate_tokens(result) > max_tokens and result.strongest_evidence:
            result.strongest_evidence.pop()
        if self._estimate_tokens(result) > max_tokens:
            result.gaps = result.gaps[:1]
            result.essence = self._compact(result.essence, max(120, max_tokens * 3))
        return result
