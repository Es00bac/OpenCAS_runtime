"""Nightly dreaming service layered on top of consolidation evidence."""

from __future__ import annotations

import inspect
from datetime import timezone
from pathlib import Path
from typing import Any

from opencas.cognition import CognitionGrounding, GroundingKind, GroundingSource

from .models import DreamMode, DreamRecord
from .store import DreamStore


class NightlyDreamingService:
    """Create grounded dream syntheses from consolidation-adjacent state."""

    def __init__(
        self,
        *,
        store: DreamStore,
        workspace_root: Path | str,
    ) -> None:
        self.store = store
        self.workspace_root = Path(workspace_root)

    async def run(
        self,
        *,
        mode: DreamMode | str = DreamMode.LIGHT,
        consolidation_result: dict[str, Any] | None = None,
        runtime: Any | None = None,
    ) -> DreamRecord:
        dream_mode = mode if isinstance(mode, DreamMode) else DreamMode(str(mode))
        consolidation_payload = dict(consolidation_result or {})
        signals = await self._collect_signals(runtime)
        record = self._build_record(
            mode=dream_mode,
            consolidation_result=consolidation_payload,
            signals=signals,
        )
        record.artifact_path = str(self._write_artifact(record))
        await self.store.save(record)
        return record

    async def _collect_signals(self, runtime: Any | None) -> dict[str, Any]:
        if runtime is None:
            return {}
        signals: dict[str, Any] = {}
        signals["recent_daydreams"] = await _safe_list_recent(
            getattr(getattr(runtime, "ctx", None), "daydream_store", None),
            limit=5,
        )
        wellbeing_store = getattr(runtime, "wellbeing_store", None)
        latest_state = None
        if wellbeing_store is not None:
            latest = getattr(wellbeing_store, "latest_state", None)
            if callable(latest):
                try:
                    latest_state = latest()
                    if inspect.isawaitable(latest_state):
                        latest_state = await latest_state
                except Exception:
                    latest_state = None
        signals["wellbeing_state"] = latest_state
        signals["maintenance_outcomes"] = await _safe_list_recent(
            wellbeing_store,
            method_name="list_maintenance_outcomes",
            limit=5,
        )
        signals["commitments"] = await _safe_list_recent(
            getattr(runtime, "commitment_store", None),
            method_name="list_active",
            limit=10,
        )
        thread_store = getattr(runtime, "thread_registry_store", None)
        signals["thread_beads"] = await _safe_list_recent(
            thread_store,
            method_name="list_beads",
            limit=10,
        )
        return signals

    def _build_record(
        self,
        *,
        mode: DreamMode,
        consolidation_result: dict[str, Any],
        signals: dict[str, Any],
    ) -> DreamRecord:
        result_id = str(consolidation_result.get("result_id") or "")
        dream_for_date = str(consolidation_result.get("dream_for_date") or "").strip()
        required_nightly_dream = bool(consolidation_result.get("required_nightly_dream"))
        scheduler_trigger = str(consolidation_result.get("scheduler_trigger") or "").strip()
        source_consolidation_result_id = str(
            consolidation_result.get("source_consolidation_result_id") or ""
        ).strip()
        clusters = int(consolidation_result.get("clusters_formed") or consolidation_result.get("clusters") or 0)
        candidates = int(consolidation_result.get("candidate_episodes") or 0)
        memories = int(consolidation_result.get("memories_created") or 0)
        commitments = int(consolidation_result.get("commitments_consolidated") or 0)
        recent_daydreams = list(signals.get("recent_daydreams") or [])
        maintenance_outcomes = list(signals.get("maintenance_outcomes") or [])
        active_commitments = list(signals.get("commitments") or [])
        thread_beads = list(signals.get("thread_beads") or [])

        date_phrase = f" for {dream_for_date}" if dream_for_date else ""
        summary = (
            f"Nightly {mode.value} dream{date_phrase} after consolidation: "
            f"{candidates} candidate episodes, {clusters} clusters, "
            f"{memories} memories, {commitments} commitments."
        )
        insights = [
            f"Consolidation formed {clusters} clusters from {candidates} candidates.",
            f"{len(recent_daydreams)} waking daydream reflections were available as curiosity residue.",
        ]
        if required_nightly_dream:
            insights.append(
                "This record was created by the required nightly continuity sweep."
            )
        if maintenance_outcomes:
            insights.append(
                f"{len(maintenance_outcomes)} maintenance outcomes can inform tomorrow's pacing."
            )
        if active_commitments:
            insights.append(
                f"{len(active_commitments)} active commitments should stay attached to proof and schedules."
            )
        if thread_beads:
            insights.append(
                f"{len(thread_beads)} peripheral thread beads are available for later pickup."
            )

        curiosity_seeds = _curiosity_seeds(
            recent_daydreams=recent_daydreams,
            thread_beads=thread_beads,
            mode=mode,
        )
        action_candidates = _action_candidates(
            active_commitments=active_commitments,
            maintenance_outcomes=maintenance_outcomes,
            mode=mode,
        )
        narrative = _dream_narrative(
            mode=mode,
            summary=summary,
            insights=insights,
            curiosity_seeds=curiosity_seeds,
            action_candidates=action_candidates,
        )
        grounding = [
            CognitionGrounding(
                kind=GroundingKind.DERIVED,
                source=GroundingSource.RUNTIME,
                subject="nightly_dream",
                claim="Dream record derived from consolidation and runtime substrate signals.",
                confidence=0.72,
                evidence_ids=[result_id] if result_id else [],
                allowed_surface="internal",
            )
        ]
        return DreamRecord(
            mode=mode,
            consolidation_result_id=result_id,
            summary=summary,
            narrative=narrative,
            insights=insights,
            curiosity_seeds=curiosity_seeds,
            action_candidates=action_candidates,
            grounding=grounding,
            meta={
                "candidate_episodes": candidates,
                "clusters_formed": clusters,
                "memories_created": memories,
                "commitments_consolidated": commitments,
                "recent_daydream_count": len(recent_daydreams),
                "maintenance_outcome_count": len(maintenance_outcomes),
                "active_commitment_count": len(active_commitments),
                "thread_bead_count": len(thread_beads),
                "dream_for_date": dream_for_date,
                "required_nightly_dream": required_nightly_dream,
                "scheduler_trigger": scheduler_trigger,
                "source_consolidation_result_id": source_consolidation_result_id,
            },
        )

    def _write_artifact(self, record: DreamRecord) -> Path:
        day = record.created_at.astimezone(timezone.utc).strftime("%Y-%m-%d")
        artifact_dir = self.workspace_root / "dreams" / day
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{record.dream_id}-{record.mode.value}.md"
        lines = [
            "# Nightly Dream",
            "",
            f"Dream ID: {record.dream_id}",
            f"Mode: {record.mode.value}",
            f"Source: {record.source}",
            f"Consolidation result: {record.consolidation_result_id or 'unknown'}",
            f"Dream for date: {record.meta.get('dream_for_date') or 'unspecified'}",
            f"Required nightly dream: {record.meta.get('required_nightly_dream')}",
            "",
            "## Summary",
            "",
            record.summary,
            "",
            "## Narrative",
            "",
            record.narrative,
            "",
            "## Insights",
            "",
            *[f"- {item}" for item in record.insights],
            "",
            "## Curiosity Seeds",
            "",
            *[f"- {item}" for item in record.curiosity_seeds],
            "",
            "## Action Candidates",
            "",
            *[f"- {item}" for item in record.action_candidates],
            "",
        ]
        artifact_path.write_text("\n".join(lines), encoding="utf-8")
        return artifact_path


async def _safe_list_recent(
    store: Any,
    *,
    method_name: str = "list_recent",
    limit: int = 5,
) -> list[Any]:
    if store is None:
        return []
    method = getattr(store, method_name, None)
    if not callable(method):
        return []
    try:
        result = method(limit=limit)
        if inspect.isawaitable(result):
            result = await result
        return list(result or [])
    except Exception:
        return []


def _curiosity_seeds(
    *,
    recent_daydreams: list[Any],
    thread_beads: list[Any],
    mode: DreamMode,
) -> list[str]:
    seeds: list[str] = []
    for reflection in recent_daydreams[:3]:
        open_question = str(getattr(reflection, "open_question", "") or "").strip()
        spark = str(getattr(reflection, "spark_content", "") or "").strip()
        if open_question:
            seeds.append(open_question)
        elif spark:
            seeds.append(spark[:160])
    if mode in {DreamMode.MEDIUM, DreamMode.HEAVY}:
        for bead in thread_beads[:3]:
            title = str(getattr(bead, "title", "") or "").strip()
            summary = str(getattr(bead, "summary", "") or "").strip()
            if title or summary:
                seeds.append((title or summary)[:160])
    if not seeds:
        seeds.append("Look for one small unresolved thread that can become useful tomorrow.")
    return seeds[:6]


def _action_candidates(
    *,
    active_commitments: list[Any],
    maintenance_outcomes: list[Any],
    mode: DreamMode,
) -> list[str]:
    candidates: list[str] = []
    for commitment in active_commitments[:3]:
        content = str(getattr(commitment, "content", "") or "").strip()
        if content:
            candidates.append(f"Re-check commitment: {content[:140]}")
    if mode in {DreamMode.MEDIUM, DreamMode.HEAVY}:
        for outcome in maintenance_outcomes[:3]:
            action_type = str(getattr(outcome, "action_type", "") or "").strip()
            result = str(getattr(outcome, "outcome", "") or "").strip()
            if action_type or result:
                candidates.append(f"Review maintenance outcome: {action_type} {result}".strip())
    if mode is DreamMode.HEAVY:
        candidates.append("Extract one proposal candidate only if evidence points to a recurring system gap.")
    return candidates[:6]


def _dream_narrative(
    *,
    mode: DreamMode,
    summary: str,
    insights: list[str],
    curiosity_seeds: list[str],
    action_candidates: list[str],
) -> str:
    if mode is DreamMode.LIGHT:
        return " ".join([summary, "The dream keeps only the clearest signal for tomorrow."])
    if mode is DreamMode.MEDIUM:
        return " ".join(
            [
                summary,
                "The dream connects consolidation, curiosity residue, and maintenance receipts into a small set of inspectable questions.",
                "The strongest seed is:",
                curiosity_seeds[0],
            ]
        )
    return " ".join(
        [
            summary,
            "The dream treats memory, promises, wellbeing, and peripheral threads as one night-side workspace.",
            "It does not act directly; it distills insight and hands the waking system reviewable candidates.",
            "First insight:",
            insights[0] if insights else "No strong insight surfaced.",
            "First action candidate:",
            action_candidates[0] if action_candidates else "No action candidate surfaced.",
        ]
    )
