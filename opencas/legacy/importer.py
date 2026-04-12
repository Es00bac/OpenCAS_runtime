"""Orchestrated, checkpointed import task for OpenBulma v4 → OpenCAS."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from pydantic import BaseModel, Field

from opencas.daydream.models import ConflictRecord, DaydreamReflection
from opencas.autonomy.portfolio import PortfolioCluster
from opencas.embeddings.backfill import EmbeddingBackfill
from opencas.embeddings import EmbeddingRecord
from opencas.execution.models import RepairTask
from opencas.memory.models import Episode, EpisodeEdge, EpisodeKind
from opencas.telemetry import EventKind
from opencas.telemetry.token_telemetry import TokenUsageEvent
from opencas.telegram_config import TelegramRuntimeConfig, load_telegram_runtime_config, save_telegram_runtime_config
from opencas.tom import BeliefSubject

from .cutover import (
    CutoverManifest,
    CutoverManifestEntry,
    copy_curated_legacy_workspace,
    load_json_file,
    preflight_bulma_state,
    record_retired_categories,
    redact_secrets,
)
from .embedding_reconciler import reconcile_embeddings
from .executive_event_index import (
    archive_and_index_executive_events,
    build_executive_event_summary_episodes,
)
from .loader import load_json, stream_jsonl
from opencas.context.models import MessageRole

from .mapper import (
    build_temporal_edge,
    map_bulma_action_approval,
    map_bulma_commitment,
    map_bulma_conflicts,
    map_bulma_consolidation_report,
    map_bulma_emotion_history,
    map_bulma_episode,
    map_bulma_execution_receipt,
    map_bulma_goal,
    map_bulma_goal_thread,
    map_bulma_history,
    map_bulma_identity,
    map_bulma_initiative,
    map_bulma_memory_edge,
    map_bulma_notification,
    map_bulma_objective_loop,
    map_bulma_outcome,
    map_bulma_research_notebook,
    map_bulma_session_message,
    map_bulma_skill,
    map_bulma_somatic_state,
    map_bulma_spark,
    map_bulma_spark_record,
    map_bulma_task_plan,
    bulma_episode_uuid,
)
from .models import (
    BulmaActionApproval,
    BulmaCommitment,
    BulmaConsolidationReport,
    BulmaDaydreamInitiative,
    BulmaDaydreamNotification,
    BulmaDaydreamOutcome,
    BulmaEmotionHistoryEntry,
    BulmaEpisode,
    BulmaExecutionReceipt,
    BulmaGoal,
    BulmaGoalThread,
    BulmaHistoryEntry,
    BulmaIdentityProfile,
    BulmaMemoryEdge,
    BulmaMusubiState,
    BulmaObjectiveLoop,
    BulmaResearchNotebook,
    BulmaSession,
    BulmaSkillEntry,
    BulmaSomaticState,
    BulmaSpark,
    BulmaTaskPlan,
    BulmaWorkspaceSnapshot,
)
from .workspace_importer import import_work_products, import_workspaces


class ImportPhase(str, Enum):
    EPISODES = "episodes"
    EDGES = "edges"
    EMBEDDINGS = "embeddings"
    IDENTITY = "identity"
    DAYDREAMS = "daydreams"
    WORKSPACES = "workspaces"
    SOMATIC = "somatic"
    EXECUTIVE = "executive"
    CONTINUITY = "continuity"
    TASKS = "tasks"
    SKILLS = "skills"
    GOVERNANCE = "governance"
    EXECUTION_RECEIPTS = "execution_receipts"
    HARNESS = "harness"
    RELATIONAL = "relational"
    EXECUTIVE_EVENTS = "executive_events"
    TASK_PLANS = "task_plans"
    SESSIONS = "sessions"
    MEMORY_AUX = "memory_aux"
    CUTOVER = "cutover"
    FINALIZED = "finalized"


class ImportCheckpoint(BaseModel):
    """Resumable checkpoint for the import task."""

    completed_phases: List[str] = Field(default_factory=list)
    last_episode_id: Optional[str] = None
    last_daydream_id: Optional[str] = None
    counts: Dict[str, int] = Field(default_factory=dict)


class ImportReport(BaseModel):
    """Summary of what was imported."""

    success: bool = True
    episodes_imported: int = 0
    duplicate_episodes_skipped: int = 0
    edges_imported: int = 0
    orphan_edges_skipped: int = 0
    embeddings_imported: Dict[str, int] = Field(default_factory=dict)
    daydreams_imported: int = 0
    daydream_sparks_imported: int = 0
    daydream_initiatives_imported: int = 0
    daydream_outcomes_imported: int = 0
    daydream_notifications_imported: int = 0
    conflicts_imported: int = 0
    workspaces_imported: int = 0
    work_products_imported: int = 0
    somatic_snapshots_imported: int = 0
    goals_imported: int = 0
    commitments_imported: int = 0
    intention_imported: bool = False
    tasks_imported: int = 0
    backfill_candidates: int = 0
    skills_imported: int = 0
    governance_entries_imported: int = 0
    execution_receipts_imported: int = 0
    research_notebooks_imported: int = 0
    objective_loops_imported: int = 0
    task_plans_imported: int = 0
    sessions_imported: int = 0
    session_messages_imported: int = 0
    executive_events_imported: int = 0
    executive_events_archived: bool = False
    executive_event_index_path: Optional[str] = None
    executive_event_archive_path: Optional[str] = None
    emotion_history_imported: int = 0
    consolidation_reports_imported: int = 0
    goal_threads_imported: int = 0
    relational_state_imported: bool = False
    preflight: Dict[str, Any] = Field(default_factory=dict)
    cutover_manifest_path: Optional[str] = None
    cutover_manifest_entries: int = 0
    portfolio_clusters_imported: int = 0
    token_telemetry_imported: int = 0
    telegram_state_imported: bool = False
    self_knowledge_imported: int = 0
    curated_workspace_files_imported: int = 0
    errors: List[str] = Field(default_factory=list)


class BulmaImportTask:
    """Checkpointed import runner."""

    def __init__(
        self,
        bulma_state_dir: Path,
        runtime,
        checkpoint_store: Optional[Path] = None,
        max_executive_events: int = 100_000,
        curated_workspace_dir: Optional[Path] = None,
    ) -> None:
        self.bulma_state_dir = Path(bulma_state_dir)
        self.runtime = runtime
        self.checkpoint_path = checkpoint_store
        self.max_executive_events = max_executive_events
        self.curated_workspace_dir = Path(curated_workspace_dir) if curated_workspace_dir else None
        self.checkpoint = self._load_checkpoint()
        self.report = ImportReport()
        self._seen_episode_ids: Set[str] = set()
        self._seen_bulma_episode_ids: Set[str] = set()
        self._episode_id_map: Dict[str, str] = {}

    async def validate(self) -> ImportReport:
        """Validate source availability and count importable records without writing state."""
        report = ImportReport()
        checks = {
            "episodes_imported": self.bulma_state_dir / "memory" / "episodes.jsonl",
            "edges_imported": self.bulma_state_dir / "memory" / "edges.jsonl",
            "daydreams_imported": self.bulma_state_dir / "daydream" / "sparks.jsonl",
            "daydream_initiatives_imported": self.bulma_state_dir / "daydream" / "initiatives.jsonl",
            "daydream_outcomes_imported": self.bulma_state_dir / "daydream" / "spark_outcomes.jsonl",
            "daydream_notifications_imported": self.bulma_state_dir / "daydream" / "notifications.jsonl",
            "somatic_snapshots_imported": self.bulma_state_dir / "somatic" / "history.jsonl",
            "sessions_imported": self.bulma_state_dir / "sessions",
        }
        for attr, path in checks.items():
            if not path.exists():
                continue
            if path.is_dir():
                setattr(report, attr, len(list(path.glob("*.json"))))
            else:
                setattr(report, attr, sum(1 for _ in stream_jsonl(path)))
        events_path = self.bulma_state_dir / "executive" / "events.jsonl"
        if events_path.exists():
            report.executive_events_imported = sum(1 for _ in stream_jsonl(events_path))
        identity_path = self.bulma_state_dir / "identity" / "profile.json"
        if not identity_path.exists():
            report.success = False
            report.errors.append("missing identity/profile.json")
        return report

    async def dry_run(self) -> ImportReport:
        """Backward-compatible dry-run alias for validation."""
        return await self.validate()

    def _load_checkpoint(self) -> ImportCheckpoint:
        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
                return ImportCheckpoint.model_validate_json(
                    self.checkpoint_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass
        return ImportCheckpoint()

    def _save_checkpoint(self) -> None:
        if self.checkpoint_path:
            self.checkpoint_path.write_text(
                self.checkpoint.model_dump_json(indent=2), encoding="utf-8"
            )

    def _trace(self, event: str, payload: Dict[str, Any]) -> None:
        if self.runtime.tracer:
            self.runtime.tracer.log(
                EventKind.BOOTSTRAP_STAGE,
                f"BulmaImport: {event}",
                payload,
            )

    async def run(self) -> ImportReport:
        try:
            self.report.preflight = preflight_bulma_state(self.bulma_state_dir).model_dump(mode="json")
            await self._run_phase(ImportPhase.EPISODES, self._import_episodes)
            await self._run_phase(ImportPhase.EDGES, self._import_edges)
            await self._run_phase(ImportPhase.EMBEDDINGS, self._import_embeddings)
            await self._run_phase(ImportPhase.IDENTITY, self._import_identity)
            await self._run_phase(ImportPhase.DAYDREAMS, self._import_daydreams)
            await self._run_phase(ImportPhase.WORKSPACES, self._import_workspaces)
            await self._run_phase(ImportPhase.SOMATIC, self._import_somatic)
            await self._run_phase(ImportPhase.EXECUTIVE, self._import_executive)
            await self._run_phase(ImportPhase.CONTINUITY, self._import_continuity)
            await self._run_phase(ImportPhase.TASKS, self._import_tasks)
            await self._run_phase(ImportPhase.SKILLS, self._import_skills)
            await self._run_phase(ImportPhase.GOVERNANCE, self._import_governance)
            await self._run_phase(ImportPhase.EXECUTION_RECEIPTS, self._import_execution_receipts)
            await self._run_phase(ImportPhase.HARNESS, self._import_harness)
            await self._run_phase(ImportPhase.RELATIONAL, self._import_relational)
            await self._run_phase(ImportPhase.EXECUTIVE_EVENTS, self._import_executive_events)
            await self._run_phase(ImportPhase.TASK_PLANS, self._import_task_plans)
            await self._run_phase(ImportPhase.SESSIONS, self._import_sessions)
            await self._run_phase(ImportPhase.MEMORY_AUX, self._import_memory_aux)
            await self._run_phase(ImportPhase.CUTOVER, self._import_cutover_operational_state)
            await self._run_phase(ImportPhase.FINALIZED, self._finalize)
        except Exception as exc:
            self.report.success = False
            self.report.errors.append(str(exc))
            self._trace("import_error", {"error": str(exc)})
            raise
        return self.report

    async def _run_phase(self, phase: ImportPhase, handler) -> None:
        if phase.value in self.checkpoint.completed_phases:
            return
        await handler()
        self.checkpoint.completed_phases.append(phase.value)
        self._save_checkpoint()

    async def _import_episodes(self) -> None:
        path = self.bulma_state_dir / "memory" / "episodes.jsonl"
        if not path.exists():
            return

        batch: List[Episode] = []
        count = 0
        duplicate_count = 0
        resume_after = self.checkpoint.last_episode_id
        catching_up = bool(resume_after)
        last_raw_id: Optional[str] = None

        for raw in stream_jsonl(path):
            be = BulmaEpisode.model_validate(raw)
            mapped_id = str(bulma_episode_uuid(be.id))
            self._episode_id_map[be.id] = mapped_id
            if catching_up:
                if be.id == resume_after or mapped_id == resume_after:
                    catching_up = False
                continue
            if be.id in self._seen_bulma_episode_ids:
                duplicate_count += 1
                continue
            self._seen_bulma_episode_ids.add(be.id)
            episode = map_bulma_episode(be)
            self._seen_episode_ids.add(str(episode.episode_id))
            batch.append(episode)
            last_raw_id = be.id
            if len(batch) >= 500:
                await self.runtime.memory.save_episodes_batch(batch)
                count += len(batch)
                self.checkpoint.last_episode_id = last_raw_id
                self._save_checkpoint()
                batch.clear()

        if batch:
            await self.runtime.memory.save_episodes_batch(batch)
            count += len(batch)
            self.checkpoint.last_episode_id = last_raw_id

        self.report.episodes_imported = count
        self.report.duplicate_episodes_skipped = duplicate_count
        self.checkpoint.counts["episodes"] = count
        self.checkpoint.counts["duplicate_episodes_skipped"] = duplicate_count
        self._trace("episodes_imported", {"count": count, "duplicate_episodes_skipped": duplicate_count})

    async def _import_edges(self) -> None:
        total_count = 0
        orphan_count = 0
        if not self._episode_id_map:
            self._episode_id_map = self._build_episode_id_map()

        # Rebuild temporal edges per session by ordering imported episodes.
        if self._seen_episode_ids:
            sessions: Dict[str, List[Episode]] = {}
            cursor = 0
            batch_size = 500
            all_episodes: List[Episode] = []

            while True:
                eps = await self.runtime.memory.list_episodes(limit=batch_size, offset=cursor)
                if not eps:
                    break
                all_episodes.extend(eps)
                cursor += len(eps)

            for ep in all_episodes:
                if ep.session_id:
                    sessions.setdefault(ep.session_id, []).append(ep)

            temporal_batch: List[EpisodeEdge] = []
            for session_id, eps in sessions.items():
                eps.sort(key=lambda e: e.created_at)
                for prev, curr in zip(eps, eps[1:]):
                    edge = build_temporal_edge(prev, curr)
                    if edge is None:
                        continue
                    temporal_batch.append(edge)
                    if len(temporal_batch) >= 500:
                        await self.runtime.memory.save_edges_batch(temporal_batch)
                        total_count += len(temporal_batch)
                        temporal_batch.clear()

            if temporal_batch:
                await self.runtime.memory.save_edges_batch(temporal_batch)
                total_count += len(temporal_batch)

        # Explicit edges from Bulma edges.jsonl overwrite temporal baselines
        edges_path = self.bulma_state_dir / "memory" / "edges.jsonl"
        if edges_path.exists():
            batch: List[EpisodeEdge] = []
            for raw in stream_jsonl(edges_path):
                try:
                    be = BulmaMemoryEdge.model_validate(raw)
                except Exception:
                    continue
                if be.sourceId not in self._episode_id_map or be.targetId not in self._episode_id_map:
                    orphan_count += 1
                    continue
                batch.append(map_bulma_memory_edge(be, self._episode_id_map))
                if len(batch) >= 500:
                    await self.runtime.memory.save_edges_batch(batch)
                    total_count += len(batch)
                    batch.clear()
            if batch:
                await self.runtime.memory.save_edges_batch(batch)
                total_count += len(batch)

        self.report.edges_imported = total_count
        self.report.orphan_edges_skipped = orphan_count
        self.checkpoint.counts["edges"] = total_count
        self.checkpoint.counts["orphan_edges_skipped"] = orphan_count
        self._trace("edges_imported", {"count": total_count, "orphan_edges_skipped": orphan_count})

    def _build_episode_id_map(self) -> Dict[str, str]:
        path = self.bulma_state_dir / "memory" / "episodes.jsonl"
        id_map: Dict[str, str] = {}
        if not path.exists():
            return id_map
        for raw in stream_jsonl(path):
            episode_id = raw.get("id") if isinstance(raw, dict) else None
            if isinstance(episode_id, str) and episode_id:
                id_map[episode_id] = str(bulma_episode_uuid(episode_id))
        return id_map

    async def _import_embeddings(self) -> None:
        # Build a lightweight episode_id → content map for source hashing
        episode_id_to_text: Dict[str, str] = {}
        high_salience_ids: Set[str] = set()
        episodes_for_backfill: List[Episode] = []
        cursor = 0
        while True:
            eps = await self.runtime.memory.list_episodes(limit=500, offset=cursor)
            if not eps:
                break
            for ep in eps:
                episode_id_to_text[str(ep.episode_id)] = ep.content
                episodes_for_backfill.append(ep)
                if ep.salience >= 1.5 or ep.identity_core:
                    high_salience_ids.add(str(ep.episode_id))
            cursor += len(eps)

        summary = await reconcile_embeddings(
            self.bulma_state_dir,
            self.runtime.ctx.embeddings,
            episode_id_to_text=episode_id_to_text,
            high_salience_ids=high_salience_ids,
        )
        self.report.embeddings_imported = summary.get("imported_counts", {})
        self.report.backfill_candidates = summary.get("backfill_candidates", 0)
        backfill = EmbeddingBackfill(self.runtime.ctx.embeddings, self.runtime.memory)
        self.report.backfill_candidates += await backfill.align_episode_embeddings(episodes_for_backfill)
        self._trace("embeddings_reconciled", summary)

    async def _import_identity(self) -> None:
        path = self.bulma_state_dir / "identity" / "profile.json"
        data = load_json(path)
        if not data:
            return

        profile = BulmaIdentityProfile.model_validate(data)
        mapped = map_bulma_identity(profile)
        rebuild_audit = load_json(self.bulma_state_dir / "identity" / "rebuild-audit.json") or {}
        self.runtime.ctx.identity.import_profile(
            narrative=mapped["narrative"],
            values=mapped["values"],
            ongoing_goals=mapped["ongoing_goals"],
            traits=mapped["traits"],
            partner_user_id=mapped.get("partner_user_id"),
            partner_trust=mapped.get("partner_trust"),
            partner_musubi=mapped.get("partner_musubi"),
            source_system="openbulma-v4",
            raw_profile=profile.model_dump(mode="json"),
            recent_themes=list(profile.recentThemes),
            memory_anchors=[anchor.model_dump(mode="json") for anchor in profile.memoryAnchors],
            rebuild_audit=rebuild_audit,
            auto_activate=True,
        )

        # Inject ToM beliefs about the partner
        if mapped.get("partner_user_id"):
            await self.runtime.tom.record_belief(
                BeliefSubject.USER,
                f"partner user is {mapped['partner_user_id']}",
                confidence=0.95,
            )
        if mapped.get("partner_trust") is not None:
            await self.runtime.tom.record_belief(
                BeliefSubject.USER,
                f"partner trust level is {mapped['partner_trust']:.2f}",
                confidence=0.9,
            )

        # Seed relational engine if available
        rel = getattr(self.runtime.ctx, "relational", None)
        if rel and mapped.get("partner_user_id"):
            raw_trust = mapped.get("partner_trust") or 0.0
            raw_musubi = mapped.get("partner_musubi") or 0.0
            await rel.import_partner_state(
                user_id=mapped["partner_user_id"],
                trust=raw_trust / 100.0 if raw_trust > 1.0 else raw_trust,
                musubi=raw_musubi / 100.0 if raw_musubi > 1.0 else raw_musubi,
            )

        self.runtime.executive.restore_goals_from_identity()

        self._trace("identity_imported", {"values_count": len(mapped["values"])})

    async def _import_daydreams(self) -> None:
        sparks_path = self.bulma_state_dir / "daydream" / "sparks.jsonl"
        history_path = self.bulma_state_dir / "daydream" / "history.jsonl"
        initiatives_path = self.bulma_state_dir / "daydream" / "initiatives.jsonl"
        outcomes_path = self.bulma_state_dir / "daydream" / "spark_outcomes.jsonl"
        notifications_path = self.bulma_state_dir / "daydream" / "notifications.jsonl"
        conflicts_path = self.bulma_state_dir / "daydream" / "conflicts.json"

        reflections: List[DaydreamReflection] = []
        count = 0
        spark_count = 0
        resume_after = self.checkpoint.last_daydream_id
        catching_up = bool(resume_after)
        last_raw_daydream_id: Optional[str] = None

        for raw in stream_jsonl(sparks_path):
            bs = BulmaSpark.model_validate(raw)
            if catching_up:
                if bs.id == resume_after or str(map_bulma_spark(bs).reflection_id) == resume_after:
                    catching_up = False
                continue
            if hasattr(self.runtime.ctx.daydream_store, "save_spark"):
                await self.runtime.ctx.daydream_store.save_spark(map_bulma_spark_record(bs))
                spark_count += 1
            reflections.append(map_bulma_spark(bs))
            last_raw_daydream_id = bs.id
            if len(reflections) >= 200:
                await self.runtime.ctx.daydream_store.save_reflections_batch(reflections)
                count += len(reflections)
                self.checkpoint.last_daydream_id = last_raw_daydream_id
                self._save_checkpoint()
                reflections.clear()

        for raw in stream_jsonl(history_path):
            entry = BulmaHistoryEntry.model_validate(raw)
            reflections.append(map_bulma_history(entry))
            if len(reflections) >= 200:
                await self.runtime.ctx.daydream_store.save_reflections_batch(reflections)
                count += len(reflections)
                self.checkpoint.last_daydream_id = str(reflections[-1].reflection_id)
                self._save_checkpoint()
                reflections.clear()

        if reflections:
            await self.runtime.ctx.daydream_store.save_reflections_batch(reflections)
            count += len(reflections)
            self.checkpoint.last_daydream_id = last_raw_daydream_id or str(reflections[-1].reflection_id)

        self.report.daydreams_imported = count
        self.report.daydream_sparks_imported = spark_count
        self.checkpoint.counts["daydreams"] = count
        self.checkpoint.counts["daydream_sparks"] = spark_count

        initiative_count = 0
        for raw in stream_jsonl(initiatives_path):
            try:
                initiative = BulmaDaydreamInitiative.model_validate(raw)
                if hasattr(self.runtime.ctx.daydream_store, "save_initiative"):
                    await self.runtime.ctx.daydream_store.save_initiative(map_bulma_initiative(initiative))
                    initiative_count += 1
            except Exception:
                continue

        outcome_count = 0
        for raw in stream_jsonl(outcomes_path):
            try:
                outcome = BulmaDaydreamOutcome.model_validate(raw)
                if hasattr(self.runtime.ctx.daydream_store, "save_outcome"):
                    await self.runtime.ctx.daydream_store.save_outcome(map_bulma_outcome(outcome))
                    outcome_count += 1
            except Exception:
                continue

        notification_count = 0
        for raw in stream_jsonl(notifications_path):
            try:
                notification = BulmaDaydreamNotification.model_validate(raw)
                if hasattr(self.runtime.ctx.daydream_store, "save_notification"):
                    await self.runtime.ctx.daydream_store.save_notification(map_bulma_notification(notification))
                    notification_count += 1
            except Exception:
                continue

        self.report.daydream_initiatives_imported = initiative_count
        self.report.daydream_outcomes_imported = outcome_count
        self.report.daydream_notifications_imported = notification_count
        self.checkpoint.counts["daydream_initiatives"] = initiative_count
        self.checkpoint.counts["daydream_outcomes"] = outcome_count
        self.checkpoint.counts["daydream_notifications"] = notification_count

        # Conflicts
        conflict_count = 0
        if conflicts_path.exists():
            try:
                raw_conflicts = json.loads(conflicts_path.read_text(encoding="utf-8"))
                if isinstance(raw_conflicts, list):
                    for record in map_bulma_conflicts(raw_conflicts):
                        await self.runtime.ctx.conflict_store.record_conflict(record)
                        conflict_count += 1
                elif isinstance(raw_conflicts, dict) and "conflicts" in raw_conflicts:
                    for record in map_bulma_conflicts(raw_conflicts["conflicts"]):
                        await self.runtime.ctx.conflict_store.record_conflict(record)
                        conflict_count += 1
            except (json.JSONDecodeError, ValueError):
                pass

        self.report.conflicts_imported = conflict_count
        self._trace(
            "daydreams_imported",
            {
                "reflections": count,
                "sparks": spark_count,
                "initiatives": initiative_count,
                "outcomes": outcome_count,
                "notifications": notification_count,
                "conflicts": conflict_count,
            },
        )

    async def _import_workspaces(self) -> None:
        ws_root = self.bulma_state_dir / "workspaces"
        wp_root = self.bulma_state_dir / "work-products"

        if not hasattr(self.runtime.ctx, "work_store") or self.runtime.ctx.work_store is None:
            self._trace("workspaces_skipped", {"reason": "no_work_store"})
            return

        ws_count = await import_workspaces(ws_root, self.runtime.ctx.work_store)
        wp_count = await import_work_products(wp_root, self.runtime.ctx.work_store)

        self.report.workspaces_imported = ws_count
        self.report.work_products_imported = wp_count
        self.checkpoint.counts["workspaces"] = ws_count
        self.checkpoint.counts["work_products"] = wp_count
        self._trace("workspaces_imported", {"workspaces": ws_count, "products": wp_count})

    async def _import_somatic(self) -> None:
        current_path = self.bulma_state_dir / "somatic" / "current.json"
        history_path = self.bulma_state_dir / "somatic" / "history.jsonl"

        somatic_store = getattr(self.runtime.ctx, "somatic_store", None)
        if somatic_store is None:
            self._trace("somatic_skipped", {"reason": "no_somatic_store"})
            return

        count = 0

        # Current snapshot
        if current_path.exists():
            data = load_json(current_path)
            if data:
                bs = BulmaSomaticState.model_validate(data)
                snapshot = map_bulma_somatic_state(bs)
                if self.runtime.ctx.embeddings:
                    record = await self.runtime.ctx.embeddings.embed(
                        snapshot.to_canonical_text(),
                        task_type="somatic_snapshot",
                    )
                    snapshot.embedding_id = record.source_hash
                await somatic_store.save(snapshot)
                count += 1

        # Historical snapshots
        if history_path.exists():
            batch: List[Any] = []
            for raw in stream_jsonl(history_path):
                try:
                    bs = BulmaSomaticState.model_validate(raw)
                except Exception:
                    continue
                snapshot = map_bulma_somatic_state(bs)
                batch.append(snapshot)
                if len(batch) >= 500:
                    await somatic_store.save_batch(batch)
                    count += len(batch)
                    batch.clear()
            if batch:
                await somatic_store.save_batch(batch)
                count += len(batch)

        self.report.somatic_snapshots_imported = count
        self.checkpoint.counts["somatic"] = count
        self._trace("somatic_imported", {"count": count})

    async def _import_continuity(self) -> None:
        # Import temporal continuity state if present
        temporal_path = self.bulma_state_dir / "continuity" / "temporal-state.json"
        integrity_path = self.bulma_state_dir / "continuity" / "integrity-report.json"
        temporal = load_json(temporal_path) or {}
        integrity = load_json(integrity_path) or {}
        if temporal or integrity:
            continuity = self.runtime.ctx.identity.continuity
            continuity.source_system = "openbulma-v4"
            continuity.temporal_bridges = temporal
            continuity.integrity_report = integrity
            self.runtime.ctx.identity.save()
            self._trace(
                "continuity_imported",
                {
                    "has_temporal": bool(temporal),
                    "integrity_status": integrity.get("status") if isinstance(integrity, dict) else None,
                },
            )

        # Record a synthetic boot episode so the agent knows it was imported
        boot_episode = Episode(
            kind=EpisodeKind.OBSERVATION,
            session_id="bulma-import",
            content="State imported from OpenBulma v4.",
            salience=2.0,
            identity_core=True,
            payload={"import_source": "openbulma-v4", "imported_at": datetime.now(timezone.utc).isoformat()},
        )
        await self.runtime.memory.save_episode(boot_episode)
        self._trace("continuity_boot_episode", {})

    async def _import_executive(self) -> None:
        goals_path = self.bulma_state_dir / "executive" / "goals.json"
        commitments_path = self.bulma_state_dir / "executive" / "commitments.json"
        workspace_path = self.bulma_state_dir / "executive" / "workspace.json"

        goals_imported = 0
        if goals_path.exists():
            data = load_json(goals_path)
            if isinstance(data, list):
                for raw in data:
                    try:
                        bg = BulmaGoal.model_validate(raw)
                        if bg.status == "active":
                            self.runtime.executive.add_goal(map_bulma_goal(bg))
                            goals_imported += 1
                    except Exception:
                        continue
            self.report.goals_imported = goals_imported

        commitments_imported = 0
        if commitments_path.exists():
            data = load_json(commitments_path)
            if isinstance(data, list):
                for raw in data:
                    try:
                        bc = BulmaCommitment.model_validate(raw)
                        work = map_bulma_commitment(bc)
                        if hasattr(self.runtime.ctx, "work_store") and self.runtime.ctx.work_store:
                            await self.runtime.ctx.work_store.save(work)
                        if bc.status == "active":
                            self.runtime.executive.enqueue(work)
                        commitments_imported += 1
                    except Exception:
                        continue
            self.report.commitments_imported = commitments_imported

        intention_imported = False
        if workspace_path.exists():
            data = load_json(workspace_path)
            if isinstance(data, dict):
                focus = data.get("focus") or {}
                focus_label = focus.get("label") or focus.get("itemLabel")
                if focus_label:
                    self.runtime.executive.set_intention(str(focus_label))
                    intention_imported = True
                elif focus.get("itemId"):
                    self.runtime.executive.set_intention(f"focus on {focus['itemId']}")
                    intention_imported = True
            self.report.intention_imported = intention_imported

        self.checkpoint.counts["executive_goals"] = goals_imported
        self.checkpoint.counts["executive_commitments"] = commitments_imported
        self._trace(
            "executive_imported",
            {
                "goals": goals_imported,
                "commitments": commitments_imported,
                "intention": intention_imported,
            },
        )

    async def _import_tasks(self) -> None:
        tasks_root = self.bulma_state_dir / "agent_tasks"
        if not tasks_root.exists():
            return

        count = 0
        for path in tasks_root.rglob("*.jsonl"):
            for raw in stream_jsonl(path):
                objective = raw.get("objective") or raw.get("description") or raw.get("label")
                if not objective:
                    continue
                task = RepairTask(objective=objective, meta={"bulma_source": str(path)})
                await self.runtime.ctx.tasks.save(task)
                count += 1

        self.report.tasks_imported = count
        self.checkpoint.counts["tasks"] = count
        self._trace("tasks_imported", {"count": count})

    async def _import_skills(self) -> None:
        registry_path = self.bulma_state_dir / "skills" / "registry.json"
        if not registry_path.exists():
            return

        data = load_json(registry_path)
        if not data or not isinstance(data, dict):
            return

        count = 0
        for raw in data.get("installed", []):
            try:
                entry = BulmaSkillEntry.model_validate(raw)
                skill = map_bulma_skill(entry)
                self.runtime.ctx.skill_registry.register(skill)
                count += 1
            except Exception:
                continue

        self.report.skills_imported = count
        self.checkpoint.counts["skills"] = count
        self._trace("skills_imported", {"count": count})

    async def _import_governance(self) -> None:
        approvals_path = self.bulma_state_dir / "governance" / "action_approvals.jsonl"
        if not approvals_path.exists():
            return

        count = 0
        for raw in stream_jsonl(approvals_path):
            try:
                ba = BulmaActionApproval.model_validate(raw)
                entry = map_bulma_action_approval(ba)
                await self.runtime.ctx.ledger.store.save(entry)
                count += 1
            except Exception:
                continue

        self.report.governance_entries_imported = count
        self.checkpoint.counts["governance"] = count
        self._trace("governance_imported", {"count": count})

    async def _import_execution_receipts(self) -> None:
        receipts_root = self.bulma_state_dir / "execution-receipts"
        if not receipts_root.exists():
            return

        count = 0
        for path in sorted(receipts_root.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                be = BulmaExecutionReceipt.model_validate(data)
                receipt = map_bulma_execution_receipt(be)
                await self.runtime.ctx.receipt_store.save_direct(receipt)
                count += 1
            except Exception:
                continue

        self.report.execution_receipts_imported = count
        self.checkpoint.counts["execution_receipts"] = count
        self._trace("execution_receipts_imported", {"count": count})

    async def _import_harness(self) -> None:
        notebooks_root = self.bulma_state_dir / "research-notebooks" / "notebooks"
        loops_root = self.bulma_state_dir / "objective-loops"

        nb_count = 0
        if notebooks_root.exists():
            for path in sorted(notebooks_root.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    bn = BulmaResearchNotebook.model_validate(data)
                    notebook = map_bulma_research_notebook(bn)
                    await self.runtime.ctx.harness.store.save_notebook(notebook)
                    nb_count += 1
                except Exception:
                    continue

        loop_count = 0
        if loops_root.exists():
            for path in sorted(loops_root.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    bl = BulmaObjectiveLoop.model_validate(data)
                    loop = map_bulma_objective_loop(bl)
                    await self.runtime.ctx.harness.store.save_loop(loop)
                    loop_count += 1
                except Exception:
                    continue

        self.report.research_notebooks_imported = nb_count
        self.report.objective_loops_imported = loop_count
        self.checkpoint.counts["research_notebooks"] = nb_count
        self.checkpoint.counts["objective_loops"] = loop_count
        self._trace("harness_imported", {"notebooks": nb_count, "loops": loop_count})

    async def _import_relational(self) -> None:
        relationship_path = self.bulma_state_dir / "relationship.json"
        if not relationship_path.exists():
            return

        try:
            data = json.loads(relationship_path.read_text(encoding="utf-8"))
        except Exception:
            return

        user_id = data.get("userId")
        trust = data.get("trust")
        musubi = data.get("musubi")
        warmth = data.get("warmth")

        rel = getattr(self.runtime.ctx, "relational", None)
        if rel and user_id is not None and trust is not None:
            await rel.import_partner_state(
                user_id=user_id,
                trust=trust / 100.0 if trust > 1.0 else trust,
                musubi=(musubi / 100.0 if musubi > 1.0 else musubi) if musubi is not None else 0.0,
                warmth=(warmth / 100.0 if warmth > 1.0 else warmth) if warmth is not None else None,
            )
            self.report.relational_state_imported = True
            self._trace("relational_imported", {"user_id": user_id})

    async def _import_executive_events(self) -> None:
        events_path = self.bulma_state_dir / "executive" / "events.jsonl"
        if not events_path.exists():
            return

        summary = archive_and_index_executive_events(
            events_path,
            self.runtime.ctx.config.state_dir,
        )
        summary_episodes = [
            Episode(
                created_at=datetime.now(timezone.utc),
                kind=EpisodeKind.OBSERVATION,
                session_id="bulma-executive-events",
                content=item["content"],
                salience=2.0,
                identity_core=True,
                payload=item["payload"],
            )
            for item in build_executive_event_summary_episodes(summary)
        ]
        if summary_episodes:
            await self.runtime.memory.save_episodes_batch(summary_episodes)

        count = int(summary.get("count", 0) or 0)
        self.report.executive_events_imported = count
        self.report.executive_events_archived = count > 0
        self.report.executive_event_index_path = summary.get("index_path")
        self.report.executive_event_archive_path = summary.get("archive_path")
        self.checkpoint.counts["executive_events"] = count
        self._trace("executive_events_indexed", summary)

    async def _import_task_plans(self) -> None:
        plans_root = self.bulma_state_dir / "task-plans"
        if not plans_root.exists():
            return

        count = 0
        for path in sorted(plans_root.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                bp = BulmaTaskPlan.model_validate(data)
                work = map_bulma_task_plan(bp)
                if self.runtime.ctx.work_store:
                    await self.runtime.ctx.work_store.save(work)
                count += 1
            except Exception:
                continue

        self.report.task_plans_imported = count
        self.checkpoint.counts["task_plans"] = count
        self._trace("task_plans_imported", {"count": count})

    async def _import_sessions(self) -> None:
        sessions_root = self.bulma_state_dir / "sessions"
        if not sessions_root.exists():
            return

        session_count = 0
        message_count = 0
        for path in sorted(sessions_root.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                bs = BulmaSession.model_validate(data)
                session_id = bs.id
                for msg in bs.contextHistory:
                    entry = map_bulma_session_message(msg)
                    await self.runtime.ctx.context_store.import_entry(session_id=session_id, entry=entry)
                    message_count += 1
                session_count += 1
            except Exception:
                continue

        self.report.sessions_imported = session_count
        self.report.session_messages_imported = message_count
        self.checkpoint.counts["sessions"] = session_count
        self.checkpoint.counts["session_messages"] = message_count
        self._trace("sessions_imported", {"sessions": session_count, "messages": message_count})

    async def _import_memory_aux(self) -> None:
        if not self._episode_id_map:
            self._episode_id_map = self._build_episode_id_map()
        emotion_path = self.bulma_state_dir / "memory" / "emotion_history.jsonl"
        consolidation_path = self.bulma_state_dir / "memory" / "consolidation_reports.jsonl"
        goal_threads_path = self.bulma_state_dir / "memory" / "goal_threads.jsonl"

        emotion_count = 0
        if emotion_path.exists():
            batch: List[Episode] = []
            for raw in stream_jsonl(emotion_path):
                try:
                    entry = BulmaEmotionHistoryEntry.model_validate(raw)
                    batch.append(map_bulma_emotion_history(entry))
                    emotion_count += 1
                    if len(batch) >= 500:
                        await self.runtime.memory.save_episodes_batch(batch)
                        batch.clear()
                except Exception:
                    continue
            if batch:
                await self.runtime.memory.save_episodes_batch(batch)

        consolidation_count = 0
        if consolidation_path.exists():
            batch: List[Episode] = []
            for raw in stream_jsonl(consolidation_path):
                try:
                    report = BulmaConsolidationReport.model_validate(raw)
                    batch.append(map_bulma_consolidation_report(report))
                    consolidation_count += 1
                    if len(batch) >= 500:
                        await self.runtime.memory.save_episodes_batch(batch)
                        batch.clear()
                except Exception:
                    continue
            if batch:
                await self.runtime.memory.save_episodes_batch(batch)

        goal_thread_count = 0
        if goal_threads_path.exists():
            for raw in stream_jsonl(goal_threads_path):
                try:
                    thread = BulmaGoalThread.model_validate(raw)
                    work = map_bulma_goal_thread(thread, self._episode_id_map)
                    if self.runtime.ctx.work_store:
                        await self.runtime.ctx.work_store.save(work)
                    goal_thread_count += 1
                except Exception:
                    continue

        self.report.emotion_history_imported = emotion_count
        self.report.consolidation_reports_imported = consolidation_count
        self.report.goal_threads_imported = goal_thread_count
        self.checkpoint.counts["emotion_history"] = emotion_count
        self.checkpoint.counts["consolidation_reports"] = consolidation_count
        self.checkpoint.counts["goal_threads"] = goal_thread_count
        self._trace(
            "memory_aux_imported",
            {
                "emotion_history": emotion_count,
                "consolidation_reports": consolidation_count,
                "goal_threads": goal_thread_count,
            },
        )

    async def _import_cutover_operational_state(self) -> None:
        manifest = CutoverManifest()
        state_dir = self.runtime.ctx.config.state_dir

        self.report.portfolio_clusters_imported = await self._import_portfolio_clusters(manifest)
        self.report.token_telemetry_imported = await self._import_token_telemetry(manifest)
        self.report.telegram_state_imported = await self._import_telegram_state(manifest)
        self.report.self_knowledge_imported = await self._import_self_knowledge(manifest)
        await self._import_somatic_musubi_runtime(manifest)
        await self._import_daydream_runtime_metadata(manifest)

        curated_root = self.curated_workspace_dir or (self.bulma_state_dir / "workspace")
        workspace_target = self.runtime.ctx.config.primary_workspace_root()
        self.report.curated_workspace_files_imported = copy_curated_legacy_workspace(
            curated_root,
            workspace_target,
            manifest,
        )
        if not curated_root.exists():
            manifest.entries.append(
                CutoverManifestEntry(
                    category="workspace",
                    source_path=str(curated_root),
                    disposition="skipped-missing",
                    note="No curated workspace bundle was present.",
                )
            )

        record_retired_categories(
            self.bulma_state_dir,
            manifest,
            (
                "workspaces",
                "work-products",
                "foreground-artifacts",
                "foreground-workbench",
                "tool-result-spill",
                "document-drafts",
                "deliverable-schemas",
                "heartbeat",
                "runtime-hooks",
                "webhooks",
                "logs",
                "reports",
                "backups",
                "migration_runs",
                "root_owned_quarantine",
            ),
        )

        manifest_path = manifest.write(state_dir / "migration" / "bulma" / "cutover_manifest.json")
        self.report.cutover_manifest_path = str(manifest_path)
        self.report.cutover_manifest_entries = len(manifest.entries)
        self.checkpoint.counts["portfolio_clusters"] = self.report.portfolio_clusters_imported
        self.checkpoint.counts["token_telemetry"] = self.report.token_telemetry_imported
        self.checkpoint.counts["self_knowledge"] = self.report.self_knowledge_imported
        self.checkpoint.counts["curated_workspace_files"] = self.report.curated_workspace_files_imported
        self._trace(
            "cutover_operational_state_imported",
            {
                "manifest_path": str(manifest_path),
                "manifest_entries": len(manifest.entries),
                "curated_workspace_files": self.report.curated_workspace_files_imported,
            },
        )

    async def _import_portfolio_clusters(self, manifest: CutoverManifest) -> int:
        path = self.bulma_state_dir / "portfolio" / "clusters.json"
        portfolio_store = getattr(self.runtime.ctx, "portfolio_store", None)
        if not path.exists() or portfolio_store is None:
            return 0
        data = load_json_file(path)
        if not isinstance(data, list):
            return 0
        count = 0
        for raw in data:
            if not isinstance(raw, dict):
                continue
            try:
                cluster_id = str(raw.get("id") or "")
                if cluster_id.startswith("portfolio:"):
                    cluster_id = cluster_id.split("portfolio:", 1)[1]
                cluster_kwargs: Dict[str, Any] = {}
                if cluster_id:
                    cluster_kwargs["cluster_id"] = UUID(cluster_id)
                cluster = PortfolioCluster(
                    **cluster_kwargs,
                    created_at=_dt_from_ms(raw.get("firstSeenAtMs")),
                    updated_at=_dt_from_ms(raw.get("lastTouchedAtMs")),
                    fascination_key=str(raw.get("fascinationKey") or raw.get("title") or cluster_id),
                    spark_count=int(raw.get("sparkCount", 0) or 0),
                    initiative_count=int(raw.get("initiativeCount", 0) or 0),
                    artifact_count=int(raw.get("artifactCount", 0) or 0),
                    last_touched_at=_dt_from_ms(raw.get("lastTouchedAtMs")),
                    tags=[str(tag) for tag in raw.get("tags", []) if str(tag)],
                    meta={"bulma_raw": raw},
                )
                await portfolio_store.save(cluster)
                count += 1
            except Exception:
                continue
        manifest.add_path(path, category="portfolio", disposition="mapped", note=f"{count} clusters imported")
        return count

    async def _import_token_telemetry(self, manifest: CutoverManifest) -> int:
        path = self.bulma_state_dir / "telemetry" / "token-events.jsonl"
        telemetry = getattr(self.runtime.ctx, "token_telemetry", None)
        if not path.exists() or telemetry is None:
            return 0
        await telemetry.flush()
        count = 0
        target = telemetry.events_file
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open("r", encoding="utf-8") as source, target.open("a", encoding="utf-8") as dest:
            for line in source:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    raw = json.loads(stripped)
                    event = TokenUsageEvent.from_dict(raw)
                except Exception:
                    continue
                dest.write(json.dumps(event.to_dict(), ensure_ascii=True) + "\n")
                count += 1
        manifest.add_path(path, category="telemetry", disposition="mapped", imported_path=target, note=f"{count} token events imported")
        return count

    async def _import_telegram_state(self, manifest: CutoverManifest) -> bool:
        telegram_root = self.bulma_state_dir / "telegram"
        config_path = telegram_root / "bot-config.json"
        if not config_path.exists():
            return False
        data = load_json_file(config_path)
        if not isinstance(data, dict):
            return False

        state_dir = self.runtime.ctx.config.state_dir
        existing = load_telegram_runtime_config(state_dir)
        config = TelegramRuntimeConfig(
            enabled=bool(data.get("enabled", existing.enabled)),
            bot_token=data.get("botToken") or existing.bot_token,
            dm_policy=data.get("dmPolicy") or existing.dm_policy,
            allow_from=data.get("allowFrom") or existing.allow_from,
            poll_interval_seconds=existing.poll_interval_seconds,
            pairing_ttl_seconds=existing.pairing_ttl_seconds,
            api_base_url=existing.api_base_url,
        )
        saved_config = save_telegram_runtime_config(state_dir, config)
        manifest.add_path(config_path, category="telegram", disposition="mapped", imported_path=saved_config, note="Telegram config imported with secrets redacted from reports")

        approved_ids = {str(item) for item in config.allow_from if str(item)}
        requests: List[Dict[str, Any]] = []
        pairing_claims = telegram_root / "pairing-claims.jsonl"
        if pairing_claims.exists():
            for raw in stream_jsonl(pairing_claims):
                sender_id = raw.get("senderId") if isinstance(raw, dict) else None
                if sender_id:
                    approved_ids.add(str(sender_id))
                requests.append(
                    {
                        "code": str(raw.get("code") or ""),
                        "user_id": str(sender_id or ""),
                        "approved_at": (float(raw.get("claimedAtMs")) / 1000.0) if raw.get("claimedAtMs") else None,
                    }
                )
            manifest.add_path(pairing_claims, category="telegram", disposition="mapped")

        pairings = telegram_root / "pairings.jsonl"
        if pairings.exists():
            for raw in stream_jsonl(pairings):
                requests.append(
                    {
                        "code": str(raw.get("code") or ""),
                        "user_id": str(raw.get("senderId") or ""),
                        "requested_at": (float(raw.get("createdAtMs")) / 1000.0) if raw.get("createdAtMs") else None,
                    }
                )
            manifest.add_path(pairings, category="telegram", disposition="mapped")

        target = state_dir / "telegram" / "pairings.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "approved_user_ids": sorted(approved_ids),
                    "requests": [item for item in requests if item.get("code")],
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        return True

    async def _import_self_knowledge(self, manifest: CutoverManifest) -> int:
        path = self.bulma_state_dir / "self-knowledge" / "index.json"
        if not path.exists():
            return 0
        data = load_json_file(path)
        if data is None:
            return 0
        self.runtime.ctx.identity.record_self_knowledge(
            "bulma_migration",
            "self_knowledge_index",
            redact_secrets(data),
            confidence=0.9,
            meta={"source_system": "openbulma-v4", "source_path": str(path)},
        )
        manifest.add_path(path, category="self-knowledge", disposition="mapped")
        return 1

    async def _import_somatic_musubi_runtime(self, manifest: CutoverManifest) -> None:
        path = self.bulma_state_dir / "somatic" / "musubi.json"
        data = load_json_file(path)
        if not isinstance(data, dict):
            return
        try:
            state = BulmaMusubiState.model_validate(data)
        except Exception:
            return
        self.runtime.ctx.identity.record_self_knowledge(
            "relational",
            "bulma_musubi_runtime_state",
            state.model_dump(mode="json"),
            confidence=0.85,
            meta={"source_system": "openbulma-v4", "source_path": str(path)},
        )
        manifest.add_path(path, category="somatic", disposition="mapped")

    async def _import_daydream_runtime_metadata(self, manifest: CutoverManifest) -> None:
        for name in ("config.json", "current_focus.json", "status.json"):
            path = self.bulma_state_dir / "daydream" / name
            data = load_json_file(path)
            if data is None:
                continue
            self.runtime.ctx.identity.record_self_knowledge(
                "daydream",
                f"bulma_{name.removesuffix('.json')}",
                redact_secrets(data),
                confidence=0.8,
                meta={"source_system": "openbulma-v4", "source_path": str(path)},
            )
            manifest.add_path(path, category="daydream", disposition="mapped")

    async def _finalize(self) -> None:
        self.runtime.readiness.ready("bulma_import_complete")
        self._trace("import_finalized", self.report.model_dump())


def _dt_from_ms(value: Any) -> datetime:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
    return datetime.now(timezone.utc)
