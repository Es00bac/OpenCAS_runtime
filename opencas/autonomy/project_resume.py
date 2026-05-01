"""Resolve project continuity into compact, reusable resume/ledger snapshots."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from opencas.context.retrieval_query import extract_anchor_terms
from opencas.harness.models import ObjectiveLoop, ObjectiveStatus

from .models import WorkObject, WorkStage

_GENERIC_PROJECT_TOKENS = {
    "a",
    "an",
    "and",
    "book",
    "build",
    "chapter",
    "compile",
    "complete",
    "continue",
    "create",
    "critique",
    "draft",
    "final",
    "finish",
    "include",
    "instead",
    "manuscript",
    "novel",
    "novel-length",
    "of",
    "over",
    "project",
    "real",
    "restart",
    "review",
    "starting",
    "third",
    "write",
}
_LIVE_WORK_STAGES = {
    WorkStage.NOTE,
    WorkStage.MICRO_TASK,
    WorkStage.PROJECT_SEED,
    WorkStage.PROJECT,
    WorkStage.DURABLE_WORK_STREAM,
}


@dataclass(frozen=True)
class ProjectResumeArtifact:
    """Artifact evidence relevant to continuing a project."""

    path: str
    title: str
    score: float
    updated_at: Optional[datetime] = None


_BLOCKED_RETRY_MODES: frozenset[str] = frozenset({
    "resume_existing_artifact",
    "deterministic_review",
    "pause_project",
    "complete_partial_and_stop",
})


@dataclass
class ProjectResumeSnapshot:
    """Compact summary of continuation evidence for one project line."""

    signature: str
    display_name: str
    canonical_artifact_path: Optional[str] = None
    supporting_artifact_paths: List[str] = field(default_factory=list)
    synopsis: str = ""
    source_surfaces: List[str] = field(default_factory=list)
    active_work_count: int = 0
    active_plan_count: int = 0
    primary_loop_id: Optional[str] = None
    duplicate_loop_ids: List[str] = field(default_factory=list)
    matched_project_ids: List[str] = field(default_factory=list)
    has_live_workstream: bool = False
    retry_state: str = "healthy"
    best_next_step: str = ""
    latest_salvage_packet_id: Optional[str] = None
    last_salvage_outcome: Optional[str] = None
    latest_salvage_meaningful_progress_signal: Optional[str] = None
    objective_contract: Dict[str, Any] = field(default_factory=dict)

    def to_meta(self) -> Dict[str, Any]:
        """Return a compact metadata payload for runtime bookkeeping."""
        return {
            "signature": self.signature,
            "display_name": self.display_name,
            "canonical_artifact_path": self.canonical_artifact_path,
            "supporting_artifact_paths": list(self.supporting_artifact_paths),
            "synopsis": self.synopsis,
            "source_surfaces": list(self.source_surfaces),
            "active_work_count": self.active_work_count,
            "active_plan_count": self.active_plan_count,
            "primary_loop_id": self.primary_loop_id,
            "duplicate_loop_ids": list(self.duplicate_loop_ids),
            "matched_project_ids": list(self.matched_project_ids),
            "has_live_workstream": self.has_live_workstream,
            "retry_state": self.retry_state,
            "best_next_step": self.best_next_step,
            "latest_salvage_packet_id": self.latest_salvage_packet_id,
            "last_salvage_outcome": self.last_salvage_outcome,
            "latest_salvage_meaningful_progress_signal": (
                self.latest_salvage_meaningful_progress_signal
            ),
            "objective_contract": dict(self.objective_contract),
        }


class ProjectResumeResolver:
    """Unify continuation evidence so the agent can resume instead of restart."""

    def __init__(
        self,
        memory: Any,
        *,
        work_store: Optional[Any] = None,
        plan_store: Optional[Any] = None,
        harness_store: Optional[Any] = None,
        salvage_store: Optional[Any] = None,
    ) -> None:
        self.memory = memory
        self.work_store = work_store
        self.plan_store = plan_store
        self.harness_store = harness_store
        self.salvage_store = salvage_store

    async def resolve(self, query: str) -> Optional[ProjectResumeSnapshot]:
        """Resolve continuation evidence for a query or project title."""
        signature = self.project_signature(query)
        if not signature:
            return None
        return await self._build_snapshot(signature)

    async def list_projects(
        self,
        *,
        limit: int = 20,
        project_id: Optional[str] = None,
    ) -> List[ProjectResumeSnapshot]:
        """Return compact ledger entries for active/resumable projects."""
        signatures = await self._collect_candidate_signatures(
            project_id=project_id,
            limit=max(limit * 6, 30),
        )
        snapshots: List[ProjectResumeSnapshot] = []
        seen = set()
        for signature in signatures:
            if signature in seen:
                continue
            seen.add(signature)
            snapshot = await self._build_snapshot(signature, project_id=project_id)
            if snapshot is None:
                continue
            snapshots.append(snapshot)
        snapshots.sort(key=self._snapshot_sort_key)
        return snapshots[:limit]

    async def find_matching_active_loop(
        self,
        title: str,
        *,
        exclude_loop_id: Optional[str] = None,
    ) -> Optional[ObjectiveLoop]:
        """Return the strongest active loop that appears to match *title*."""
        signature = self.project_signature(title)
        if not signature:
            return None
        loops = await self._find_active_loops(signature)
        if exclude_loop_id is not None:
            loops = [loop for loop in loops if str(loop.loop_id) != exclude_loop_id]
        work_hits = await self._find_work(signature)
        return self._choose_primary_loop(loops, work_hits)

    async def suppress_duplicate_active_objective_loops(self) -> Dict[str, int]:
        """Pause duplicate active loops so one canonical continuation line remains."""
        if self.harness_store is None:
            return {"groups": 0, "paused": 0}

        active_loops = await self.harness_store.list_loops(
            status=ObjectiveStatus.ACTIVE,
            limit=500,
        )
        pending_loops = await self.harness_store.list_loops(
            status=ObjectiveStatus.PENDING,
            limit=500,
        )
        grouped: Dict[str, List[ObjectiveLoop]] = {}
        for loop in [*active_loops, *pending_loops]:
            signature = self.project_signature(f"{loop.title} {loop.description}")
            if not signature:
                continue
            grouped.setdefault(signature, []).append(loop)

        groups = 0
        paused = 0
        now = datetime.now(timezone.utc)
        for signature, loops in grouped.items():
            if len(loops) < 2:
                continue
            groups += 1
            work_hits = await self._find_work(signature)
            primary = self._choose_primary_loop(loops, work_hits)
            if primary is None:
                continue
            for loop in loops:
                if str(loop.loop_id) == str(primary.loop_id):
                    continue
                if loop.status not in {ObjectiveStatus.ACTIVE, ObjectiveStatus.PENDING}:
                    continue
                loop.status = ObjectiveStatus.PAUSED
                loop.updated_at = now
                loop.meta = dict(loop.meta or {})
                loop.meta.update(
                    {
                        "paused_reason": "duplicate_project_resume",
                        "duplicate_of_loop_id": str(primary.loop_id),
                        "duplicate_signature": signature,
                        "reframe_hint": (
                            "Pause this duplicate objective and reframe any distinct work as a narrower "
                            "objective_contract on the primary loop before starting a separate loop."
                        ),
                    }
                )
                await self.harness_store.save_loop(loop)
                paused += 1
        return {"groups": groups, "paused": paused}

    async def _build_snapshot(
        self,
        signature: str,
        *,
        project_id: Optional[str] = None,
    ) -> Optional[ProjectResumeSnapshot]:
        artifacts = await self._find_artifacts(signature)
        work_hits = await self._find_work(signature, project_id=project_id)
        plan_hits = await self._find_active_plans(signature, project_id=project_id)
        loop_hits = await self._find_active_loops(signature, project_id=project_id)

        if not artifacts and not work_hits and not plan_hits and not loop_hits:
            return None

        primary_loop = self._choose_primary_loop(loop_hits, work_hits)
        duplicate_loop_ids = [
            str(loop.loop_id)
            for loop in loop_hits
            if primary_loop is not None and str(loop.loop_id) != str(primary_loop.loop_id)
        ]
        active_work = [item for item in work_hits if item.stage in _LIVE_WORK_STAGES]
        canonical_artifact = self._choose_canonical_artifact(artifacts)
        display_name = self._choose_display_name(
            signature,
            canonical_artifact=canonical_artifact,
            work_hits=work_hits,
            plan_hits=plan_hits,
            loop_hits=loop_hits,
        )
        matched_project_ids = sorted(
            {
                item.project_id
                for item in work_hits
                if item.project_id
            }
            | {
                plan.project_id
                for plan in plan_hits
                if getattr(plan, "project_id", None)
            }
            | {
                str(loop.loop_id)
                for loop in loop_hits
            }
        )
        supporting_artifact_paths = [artifact.path for artifact in artifacts[:3]]
        synopsis = self._build_synopsis(
            signature,
            canonical_artifact=canonical_artifact,
            work_hits=work_hits,
            plan_hits=plan_hits,
            loop_hits=loop_hits,
        )
        source_surfaces = self._build_source_surfaces(
            artifacts=artifacts,
            work_hits=work_hits,
            plan_hits=plan_hits,
            loop_hits=loop_hits,
        )

        latest_packet = await self._latest_salvage_for_signature(signature)
        retry_state = "healthy"
        best_next_step = ""
        latest_salvage_packet_id = None
        last_salvage_outcome = None
        latest_salvage_meaningful_progress_signal = None
        if latest_packet is not None:
            mode_val = getattr(latest_packet.recommended_mode, "value", str(latest_packet.recommended_mode))
            retry_state = "blocked_low_divergence" if mode_val in _BLOCKED_RETRY_MODES else "retrying"
            best_next_step = latest_packet.best_next_step
            latest_salvage_packet_id = str(latest_packet.packet_id)
            last_salvage_outcome = getattr(latest_packet.outcome, "value", str(latest_packet.outcome))
            latest_salvage_meaningful_progress_signal = latest_packet.meaningful_progress_signal

        return ProjectResumeSnapshot(
            signature=signature,
            display_name=display_name,
            canonical_artifact_path=canonical_artifact.path if canonical_artifact else None,
            supporting_artifact_paths=supporting_artifact_paths,
            synopsis=synopsis,
            source_surfaces=source_surfaces,
            active_work_count=len(active_work),
            active_plan_count=len(plan_hits),
            primary_loop_id=str(primary_loop.loop_id) if primary_loop is not None else None,
            duplicate_loop_ids=duplicate_loop_ids,
            matched_project_ids=matched_project_ids,
            has_live_workstream=bool(active_work),
            retry_state=retry_state,
            best_next_step=best_next_step,
            latest_salvage_packet_id=latest_salvage_packet_id,
            last_salvage_outcome=last_salvage_outcome,
            latest_salvage_meaningful_progress_signal=latest_salvage_meaningful_progress_signal,
            objective_contract=self._extract_objective_contract(primary_loop),
        )

    @staticmethod
    def project_signature(text: str) -> Optional[str]:
        """Build a conservative signature for project/title matching."""
        anchors = extract_anchor_terms(text)
        if anchors:
            anchor_tokens = [
                token
                for token in ProjectResumeResolver._tokenize(anchors[0])
                if token not in _GENERIC_PROJECT_TOKENS
            ]
            if len(anchor_tokens) >= 2:
                return " ".join(anchor_tokens[:5])
            return anchors[0].strip().lower()
        tokens = [
            token
            for token in ProjectResumeResolver._tokenize(text)
            if token not in _GENERIC_PROJECT_TOKENS
        ]
        if len(tokens) < 2:
            return None
        return " ".join(tokens[:5])

    async def _collect_candidate_signatures(
        self,
        *,
        project_id: Optional[str] = None,
        limit: int = 60,
    ) -> List[str]:
        signatures: List[str] = []
        seen = set()

        def add(text: str) -> None:
            signature = self.project_signature(text)
            if not signature or signature in seen:
                return
            seen.add(signature)
            signatures.append(signature)

        if self.harness_store is not None:
            if project_id:
                loops = await self.harness_store.list_loops(limit=500)
            else:
                loops = await self.harness_store.list_loops(
                    status=ObjectiveStatus.ACTIVE,
                    limit=500,
                )
                loops.extend(
                    await self.harness_store.list_loops(
                        status=ObjectiveStatus.PENDING,
                        limit=500,
                    )
                )
            for loop in loops:
                if project_id and str(loop.loop_id) != project_id:
                    continue
                add(f"{loop.title} {loop.description}")
                if len(signatures) >= limit:
                    return signatures

        if self.work_store is not None:
            work_items = (
                await self.work_store.list_by_project(project_id, limit=250)
                if project_id
                else await self.work_store.list_all(limit=250)
            )
            for item in work_items:
                if project_id and not self._work_matches_project_id(item, project_id):
                    continue
                add(item.content)
                if len(signatures) >= limit:
                    return signatures

        if self.plan_store is not None:
            plans = await self.plan_store.list_active(project_id=project_id)
            for plan in plans:
                add(plan.content or plan.project_id or "")
                if len(signatures) >= limit:
                    return signatures

        if self.memory is not None and not project_id:
            episodes = await self.memory.list_recent_episodes(limit=250)
            for episode in episodes:
                artifact = self._extract_artifact_payload(episode)
                path = artifact.get("path", "")
                if not path.startswith("workspace/"):
                    continue
                title = artifact.get("title", "") or PurePosixPath(path).stem
                add(title or path)
                if len(signatures) >= limit:
                    return signatures

        return signatures

    async def _find_artifacts(self, signature: str) -> List[ProjectResumeArtifact]:
        terms = [signature]
        artifact_map: Dict[str, ProjectResumeArtifact] = {}

        for term in terms:
            for episode in await self.memory.search_episodes_by_content(term, limit=40):
                payload = getattr(episode, "payload", {}) or {}
                artifact = (payload.get("payload") or {}).get("artifact") or {}
                path = str(artifact.get("path", "")).strip()
                if not path:
                    continue
                title = str(artifact.get("title", "")).strip() or PurePosixPath(path).name
                score = self._artifact_path_priority(path)
                score += self._match_bonus(signature, f"{title} {path} {episode.content}")
                updated_at = getattr(episode, "created_at", None)
                existing = artifact_map.get(path)
                if existing is None or score > existing.score:
                    artifact_map[path] = ProjectResumeArtifact(
                        path=path,
                        title=title,
                        score=score,
                        updated_at=updated_at,
                    )

            for memory in await self.memory.search_memories_by_content(term, limit=20):
                tag_map = self._memory_artifact_tags(getattr(memory, "tags", []))
                path = tag_map.get("artifact_path")
                if not path:
                    continue
                title = tag_map.get("artifact_title") or PurePosixPath(path).name
                score = self._artifact_path_priority(path)
                score += self._match_bonus(signature, f"{title} {path} {memory.content}")
                updated_at = getattr(memory, "updated_at", None)
                existing = artifact_map.get(path)
                if existing is None or score > existing.score:
                    artifact_map[path] = ProjectResumeArtifact(
                        path=path,
                        title=title,
                        score=score,
                        updated_at=updated_at,
                    )

        return sorted(
            artifact_map.values(),
            key=lambda item: (item.score, item.updated_at or datetime.min.replace(tzinfo=timezone.utc)),
            reverse=True,
        )

    async def _find_work(
        self,
        signature: str,
        *,
        project_id: Optional[str] = None,
    ) -> List[WorkObject]:
        if self.work_store is None:
            return []
        items = (
            await self.work_store.list_by_project(project_id, limit=250)
            if project_id
            else await self.work_store.list_all(limit=250)
        )
        matches = []
        for item in items:
            if project_id and not self._work_matches_project_id(item, project_id):
                continue
            haystack = f"{item.content} {json.dumps(item.meta or {}, sort_keys=True)}"
            if self._match_bonus(signature, haystack) <= 0:
                continue
            matches.append(item)
        return matches

    async def _find_active_plans(
        self,
        signature: str,
        *,
        project_id: Optional[str] = None,
    ) -> List[Any]:
        if self.plan_store is None:
            return []
        plans = await self.plan_store.list_active(project_id=project_id)
        matches = []
        for plan in plans:
            haystack = f"{plan.content} {plan.project_id or ''} {plan.task_id or ''}"
            if self._match_bonus(signature, haystack) <= 0:
                continue
            matches.append(plan)
        return matches

    async def _find_active_loops(
        self,
        signature: str,
        *,
        project_id: Optional[str] = None,
    ) -> List[ObjectiveLoop]:
        if self.harness_store is None:
            return []
        loops = await self.harness_store.list_loops(
            status=ObjectiveStatus.ACTIVE,
            limit=500,
        )
        matches = []
        for loop in loops:
            if project_id and str(loop.loop_id) != project_id:
                continue
            haystack = f"{loop.title} {loop.description}"
            if self._match_bonus(signature, haystack) <= 0:
                continue
            matches.append(loop)
        return matches

    def _choose_primary_loop(
        self,
        loops: Iterable[ObjectiveLoop],
        work_hits: Iterable[WorkObject],
    ) -> Optional[ObjectiveLoop]:
        loops = list(loops)
        if not loops:
            return None

        work_hits = list(work_hits)

        def sort_key(loop: ObjectiveLoop) -> tuple[int, int, int, datetime]:
            loop_id = str(loop.loop_id)
            live_work = 0
            for item in work_hits:
                if item.stage not in _LIVE_WORK_STAGES:
                    continue
                meta_loop_id = str((item.meta or {}).get("loop_id", "")).strip()
                if meta_loop_id == loop_id or item.project_id == loop_id:
                    live_work += 1
            return (
                1 if loop.status == ObjectiveStatus.ACTIVE else 0,
                live_work,
                len(loop.generated_task_ids),
                loop.updated_at,
            )

        return max(loops, key=sort_key)

    @staticmethod
    def _extract_objective_contract(loop: Optional[ObjectiveLoop]) -> Dict[str, Any]:
        if loop is None or not isinstance(loop.meta, dict):
            return {}
        contract = loop.meta.get("objective_contract")
        if not isinstance(contract, dict):
            return {}
        return {
            key: value
            for key, value in contract.items()
            if key
            in {
                "goal",
                "expected_output",
                "success_check",
                "stop_condition",
                "max_attempt_budget",
                "reframe_path",
            }
        }

    @staticmethod
    def _choose_canonical_artifact(
        artifacts: Iterable[ProjectResumeArtifact],
    ) -> Optional[ProjectResumeArtifact]:
        artifacts = list(artifacts)
        return artifacts[0] if artifacts else None

    def _choose_display_name(
        self,
        signature: str,
        *,
        canonical_artifact: Optional[ProjectResumeArtifact],
        work_hits: Sequence[WorkObject],
        plan_hits: Sequence[Any],
        loop_hits: Sequence[ObjectiveLoop],
    ) -> str:
        for candidate in (
            canonical_artifact.title if canonical_artifact else "",
            *(loop.title for loop in loop_hits),
            *(item.content for item in work_hits),
            *(getattr(plan, "content", "") for plan in plan_hits),
        ):
            label = self._project_label(candidate)
            if label:
                return label
        return self._display_name(signature)

    def _build_source_surfaces(
        self,
        *,
        artifacts: Sequence[ProjectResumeArtifact],
        work_hits: Sequence[WorkObject],
        plan_hits: Sequence[Any],
        loop_hits: Sequence[ObjectiveLoop],
    ) -> List[str]:
        surfaces: List[str] = []
        if loop_hits:
            surfaces.append("objective_loop")
        if work_hits:
            surfaces.append("work")
        if plan_hits:
            surfaces.append("plan")
        if artifacts:
            surfaces.append("artifact")
        return surfaces

    def _build_synopsis(
        self,
        signature: str,
        *,
        canonical_artifact: Optional[ProjectResumeArtifact],
        work_hits: Sequence[WorkObject],
        plan_hits: Sequence[Any],
        loop_hits: Sequence[ObjectiveLoop],
        max_chars: int = 260,
    ) -> str:
        fragments: List[str] = []

        work_priority = {
            WorkStage.DURABLE_WORK_STREAM: 5,
            WorkStage.PROJECT: 4,
            WorkStage.PROJECT_SEED: 3,
            WorkStage.MICRO_TASK: 2,
            WorkStage.NOTE: 1,
        }
        ordered_work = sorted(
            work_hits,
            key=lambda item: (
                work_priority.get(item.stage, 0),
                item.updated_at,
            ),
            reverse=True,
        )
        for item in ordered_work:
            fragment = self._compact_fragment(signature, item.content)
            if fragment:
                fragments.append(fragment)

        for plan in plan_hits:
            fragment = self._compact_fragment(signature, getattr(plan, "content", ""))
            if fragment:
                fragments.append(fragment)

        for loop in loop_hits:
            fragment = self._compact_fragment(
                signature,
                loop.description or loop.title,
            )
            if fragment:
                fragments.append(fragment)

        if canonical_artifact is not None and not fragments:
            fragments.append(
                f"Canonical manuscript or project artifact is already present at {canonical_artifact.path}."
            )

        selected: List[str] = []
        total = 0
        seen = set()
        for fragment in fragments:
            normalized = fragment.lower()
            if any(normalized == item or normalized in item or item in normalized for item in seen):
                continue
            projected = total + len(fragment) + (2 if selected else 0)
            if projected > max_chars:
                break
            selected.append(fragment)
            seen.add(normalized)
            total = projected

        synopsis = "; ".join(selected).strip()
        if synopsis:
            return synopsis
        if canonical_artifact is not None:
            return f"Canonical artifact already exists at {canonical_artifact.path}."
        return ""

    @classmethod
    def _compact_fragment(cls, signature: str, text: str) -> str:
        clean = " ".join(text.split()).strip()
        if not clean:
            return ""
        signature_pattern = re.escape(signature)
        clean = re.sub(
            rf"^(continue|write|complete|finish|create|build)\s+{signature_pattern}\b[\s,:-]*",
            "",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(r"^[\s:,-]+", "", clean).strip()
        if len(clean) > 160:
            clean = clean[:157].rstrip() + "..."
        return clean

    @staticmethod
    def _extract_artifact_payload(source: Any) -> Dict[str, str]:
        payload = getattr(source, "payload", {}) or {}
        artifact = (payload.get("payload") or {}).get("artifact") or {}
        return {
            "path": str(artifact.get("path", "")).strip(),
            "title": str(artifact.get("title", "")).strip(),
        }

    @staticmethod
    def _work_matches_project_id(item: WorkObject, project_id: str) -> bool:
        meta = item.meta or {}
        return item.project_id == project_id or str(meta.get("loop_id", "")).strip() == project_id

    @classmethod
    def _project_label(cls, text: str) -> str:
        anchors = extract_anchor_terms(text or "")
        if anchors:
            return anchors[0].strip()
        signature = cls.project_signature(text or "")
        if signature:
            return cls._display_name(signature)
        return ""

    async def _latest_salvage_for_signature(self, signature: str) -> Optional[Any]:
        if self.salvage_store is None:
            return None
        get_fn = getattr(self.salvage_store, "get_latest_salvage_packet_for_signature", None)
        if not callable(get_fn):
            return None
        try:
            return await get_fn(signature)
        except Exception:
            return None

    @staticmethod
    def _snapshot_sort_key(item: ProjectResumeSnapshot) -> Tuple[int, int, int, int, int, int, str]:
        return (
            0 if item.has_live_workstream else 1,
            0 if item.primary_loop_id else 1,
            -item.active_work_count,
            -item.active_plan_count,
            -len(item.source_surfaces),
            0 if item.canonical_artifact_path else 1,
            item.display_name.lower(),
        )

    @staticmethod
    def _memory_artifact_tags(tags: Iterable[str]) -> Dict[str, str]:
        parsed: Dict[str, str] = {}
        for tag in tags:
            if ":" not in tag:
                continue
            prefix, value = tag.split(":", 1)
            if prefix in {"artifact_path", "artifact_title"} and value:
                parsed[prefix] = value
        return parsed

    @staticmethod
    def _display_name(signature: str) -> str:
        return " ".join(word.capitalize() for word in signature.split())

    @staticmethod
    def _artifact_path_priority(path: str) -> float:
        lower = path.lower()
        score = 0.0
        if lower.startswith("workspace/chronicles/"):
            score += 3.0
        elif lower.startswith("workspace/review/"):
            score += 1.5
        elif "/archive/" in lower or lower.startswith("workspace/archive/"):
            score -= 1.0

        name = PurePosixPath(lower).name
        if name == "chronicle_4246.md":
            score += 3.0
        if "outline" in name:
            score += 1.0
        if any(term in name for term in ("review", "critique", "source_packet", "status_matrix")):
            score -= 0.5
        return score

    @staticmethod
    def _match_bonus(signature: str, haystack: str) -> float:
        signature = signature.strip().lower()
        haystack = haystack.strip().lower()
        if not signature or not haystack:
            return 0.0
        if signature in haystack:
            return 2.0
        sig_tokens = set(ProjectResumeResolver._tokenize(signature))
        hay_tokens = set(ProjectResumeResolver._tokenize(haystack))
        if not sig_tokens:
            return 0.0
        overlap = len(sig_tokens & hay_tokens)
        if overlap == 0:
            return 0.0
        return overlap / len(sig_tokens)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())
