"""Domain mappers: transform OpenBulma v4 structures into OpenCAS models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import NAMESPACE_OID, UUID, uuid5

from opencas.autonomy.models import ActionRiskTier, WorkObject, WorkStage
from opencas.context.models import MessageEntry, MessageRole
from opencas.daydream.models import (
    ConflictRecord,
    DaydreamInitiative,
    DaydreamNotification,
    DaydreamOutcome,
    DaydreamReflection,
    DaydreamSpark,
)
from opencas.execution.models import ExecutionPhase, ExecutionReceipt, PhaseRecord, RepairTask
from opencas.governance.models import ApprovalLedgerEntry
from opencas.harness.models import NotebookEntry, NotebookEntryKind, ObjectiveLoop, ObjectiveStatus, ResearchNotebook
from opencas.memory.models import Episode, EpisodeEdge, EpisodeKind
from opencas.plugins.models import SkillEntry
from opencas.somatic.models import AffectState, PrimaryEmotion, SomaticSnapshot, SocialTarget

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
    BulmaGoalThread,
    BulmaHistoryEntry,
    BulmaIdentityProfile,
    BulmaMemoryEdge,
    BulmaObjectiveLoop,
    BulmaResearchNotebook,
    BulmaSession,
    BulmaSessionMessage,
    BulmaSkillEntry,
    BulmaSomaticState,
    BulmaSpark,
    BulmaTaskPlan,
    BulmaWorkspaceManifest,
)


_BULMA_EMOTION_MAP: Dict[str, PrimaryEmotion] = {
    "joy": PrimaryEmotion.JOY,
    "trust": PrimaryEmotion.TRUST,
    "anticipation": PrimaryEmotion.ANTICIPATION,
    "surprise": PrimaryEmotion.SURPRISE,
    "sadness": PrimaryEmotion.SADNESS,
    "fear": PrimaryEmotion.FEAR,
    "anger": PrimaryEmotion.ANGER,
    "disgust": PrimaryEmotion.DISGUST,
    "neutral": PrimaryEmotion.NEUTRAL,
    "excited": PrimaryEmotion.EXCITED,
    "playful": PrimaryEmotion.PLAYFUL,
    "curious": PrimaryEmotion.CURIOUS,
    "focused": PrimaryEmotion.FOCUSED,
    "thoughtful": PrimaryEmotion.THOUGHTFUL,
    "concerned": PrimaryEmotion.CONCERNED,
    "caring": PrimaryEmotion.CARING,
    "apologetic": PrimaryEmotion.APOLOGETIC,
    "annoyed": PrimaryEmotion.ANNOYED,
    "proud": PrimaryEmotion.PROUD,
    "tired": PrimaryEmotion.TIRED,
    "determined": PrimaryEmotion.DETERMINED,
}

_BULMA_SOCIAL_TARGET_MAP: Dict[str, SocialTarget] = {
    "self": SocialTarget.SELF,
    "user": SocialTarget.USER,
    "other": SocialTarget.OTHER,
    "project": SocialTarget.PROJECT,
    "system": SocialTarget.SYSTEM,
}


def _parse_dt(ms: float) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _guess_kind(source: str) -> EpisodeKind:
    if "chat" in source.lower():
        return EpisodeKind.TURN
    if "v3" in source.lower():
        return EpisodeKind.OBSERVATION
    return EpisodeKind.OBSERVATION


def _guess_session_id(bulma_id: str) -> str:
    if bulma_id.startswith("2026-"):
        return bulma_id[:10]
    return "bulma-import"


def map_bulma_emotion(bulma: BulmaEpisode) -> Optional[AffectState]:
    if bulma.emotion is None:
        return None
    be = bulma.emotion
    return AffectState(
        primary_emotion=_BULMA_EMOTION_MAP.get(be.primaryEmotion.lower(), PrimaryEmotion.NEUTRAL),
        valence=be.valence,
        arousal=be.arousal,
        certainty=be.certainty,
        intensity=be.emotionalIntensity,
        social_target=_BULMA_SOCIAL_TARGET_MAP.get(be.socialTarget.lower(), SocialTarget.OTHER),
        emotion_tags=list(be.emotionTags),
    )


_BULMA_EPISODE_NAMESPACE = uuid5(NAMESPACE_OID, "openbulma-v4:episode")
_BULMA_REFLECTION_NAMESPACE = uuid5(NAMESPACE_OID, "openbulma-v4:reflection")


def bulma_episode_uuid(bulma_id: str) -> UUID:
    """Return the stable OpenCAS episode UUID for a Bulma episode ID."""
    if _is_uuid(bulma_id):
        return UUID(bulma_id)
    return uuid5(_BULMA_EPISODE_NAMESPACE, bulma_id)


def bulma_reflection_uuid(bulma_id: str) -> UUID:
    """Return the stable OpenCAS reflection UUID for a Bulma daydream ID."""
    if _is_uuid(bulma_id):
        return UUID(bulma_id)
    return uuid5(_BULMA_REFLECTION_NAMESPACE, bulma_id)


def map_bulma_episode(bulma: BulmaEpisode) -> Episode:
    payload: Dict[str, Any] = {
        "bulma_id": bulma.id,
        "bulma_source": bulma.source,
        "bulma_timestamp_ms": bulma.timestampMs,
    }
    if bulma.metadata:
        payload["bulma_metadata"] = bulma.metadata.model_dump(mode="json")
    if bulma.metadata and bulma.metadata.v3:
        payload["bulma_v3"] = bulma.metadata.v3.model_dump(mode="json")

    return Episode(
        episode_id=bulma_episode_uuid(bulma.id),
        created_at=_parse_dt(bulma.timestampMs),
        kind=_guess_kind(bulma.source),
        session_id=_guess_session_id(bulma.id),
        content=bulma.textContent,
        somatic_tag=bulma.emotion.primaryEmotion if bulma.emotion else None,
        affect=map_bulma_emotion(bulma),
        salience=round(max(0.0, min(10.0, bulma.salience)), 3),
        compacted=False,
        identity_core=bool(bulma.identityCore),
        payload=payload,
    )


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
        return True
    except ValueError:
        return False


def map_bulma_identity(profile: BulmaIdentityProfile) -> Dict[str, Any]:
    """Return a plain dict suitable for IdentityManager.import_profile()."""
    partner = profile.partner
    return {
        "narrative": profile.coreNarrative,
        "values": profile.values,
        "ongoing_goals": profile.ongoingGoals,
        "traits": profile.traits,
        "partner_user_id": partner.userId if partner else None,
        "partner_trust": partner.trust if partner else None,
        "partner_musubi": partner.musubi if partner else None,
    }


def map_bulma_spark(bulma: BulmaSpark) -> DaydreamReflection:
    return DaydreamReflection(
        reflection_id=bulma_reflection_uuid(bulma.id),
        created_at=_parse_dt(bulma.timestampMs),
        spark_content=f"{bulma.interest}\n{bulma.summary}",
        recollection=bulma.summary,
        interpretation=bulma.label,
        synthesis=bulma.objective,
        open_question=None,
        changed_self_view="",
        tension_hints=list(bulma.tags),
        alignment_score=0.0,
        novelty_score=0.0,
        keeper=False,
    )


def map_bulma_spark_record(bulma: BulmaSpark) -> DaydreamSpark:
    """Map a Bulma spark into a first-class OpenCAS daydream spark."""
    return DaydreamSpark(
        spark_id=bulma.id,
        created_at=_parse_dt(bulma.timestampMs),
        mode=bulma.mode,
        trigger=bulma.trigger,
        interest=bulma.interest,
        summary=bulma.summary,
        label=bulma.label,
        kind=bulma.kind,
        intensity=bulma.intensity,
        objective=bulma.objective,
        tags=list(bulma.tags),
        task_id=bulma.taskId,
        raw=bulma.model_dump(mode="json"),
    )


def map_bulma_initiative(bulma: BulmaDaydreamInitiative) -> DaydreamInitiative:
    """Map a Bulma routed daydream initiative into OpenCAS lifecycle storage."""
    return DaydreamInitiative(
        initiative_id=bulma.id,
        spark_id=bulma.sparkId,
        created_at=_parse_dt(bulma.timestampMs),
        mode=bulma.mode,
        trigger=bulma.trigger,
        interest=bulma.interest,
        summary=bulma.summary,
        label=bulma.label,
        kind=bulma.kind,
        intensity=bulma.intensity,
        rung=bulma.rung,
        desired_rung=bulma.desiredRung,
        objective=bulma.objective,
        focus=bulma.focus,
        source_kind=bulma.sourceKind,
        source_label=bulma.sourceLabel,
        artifact_paths=list(bulma.artifactPaths),
        task_id=bulma.taskId,
        route_debug=dict(bulma.routeDebug),
        tags=list(bulma.tags),
        raw=bulma.model_dump(mode="json"),
    )


def map_bulma_outcome(bulma: BulmaDaydreamOutcome) -> DaydreamOutcome:
    """Map a Bulma daydream task outcome into OpenCAS lifecycle storage."""
    return DaydreamOutcome(
        task_id=bulma.taskId,
        recorded_at=_parse_dt(bulma.recordedAtMs),
        outcome=bulma.outcome,
        value_delivered=bulma.valueDelivered,
        raw=bulma.model_dump(mode="json"),
    )


def map_bulma_notification(bulma: BulmaDaydreamNotification) -> DaydreamNotification:
    """Map a Bulma daydream notification into OpenCAS lifecycle storage."""
    basis = f"{bulma.sparkId}:{bulma.chatId}:{bulma.sentAtMs}"
    return DaydreamNotification(
        notification_id=str(uuid5(NAMESPACE_OID, basis)),
        spark_id=bulma.sparkId,
        chat_id=bulma.chatId,
        sent_at=_parse_dt(bulma.sentAtMs),
        label=bulma.label,
        intensity=bulma.intensity,
        kind=bulma.kind,
        raw=bulma.model_dump(mode="json"),
    )


def map_bulma_history(entry: BulmaHistoryEntry) -> DaydreamReflection:
    return DaydreamReflection(
        reflection_id=UUID(int=int(entry.timestampMs)),
        created_at=_parse_dt(entry.timestampMs),
        spark_content=f"{entry.interest}\n{entry.summary}",
        recollection=entry.summary,
        interpretation=entry.mode,
        synthesis=entry.interest,
        open_question=None,
        changed_self_view="",
        tension_hints=list(entry.tags),
        alignment_score=0.0,
        novelty_score=0.0,
        keeper=False,
    )


def map_bulma_conflicts(raw_conflicts: List[Dict[str, Any]]) -> List[ConflictRecord]:
    records: List[ConflictRecord] = []
    for item in raw_conflicts:
        records.append(
            ConflictRecord(
                kind=item.get("kind", "unknown"),
                description=item.get("description", ""),
                source_daydream_id=item.get("source_daydream_id"),
            )
        )
    return records


def map_bulma_goal(bulma: Any) -> str:
    """Map a Bulma goal into a plain goal string."""
    label = getattr(bulma, "label", "")
    if not label and isinstance(bulma, dict):
        label = bulma.get("label", "")
    return str(label)


def map_bulma_commitment(bulma: BulmaCommitment) -> WorkObject:
    """Map a Bulma commitment into an OpenCAS WorkObject."""
    stage = WorkStage.MICRO_TASK
    if bulma.status in {"released", "completed"}:
        stage = WorkStage.ARTIFACT
    blocked_by: List[str] = []
    if bulma.blockedReason:
        blocked_by.append("bulma-import-blocked")
    return WorkObject(
        content=bulma.label,
        stage=stage,
        blocked_by=blocked_by,
        meta={
            "bulma_goal_id": bulma.goalId,
            "bulma_commitment_id": bulma.id,
            "bulma_execution_state": bulma.executionState,
            "bulma_verification_state": bulma.verificationState,
            "bulma_closure_state": bulma.closureState,
            "bulma_blocked_reason": bulma.blockedReason,
        },
    )


def map_bulma_workspace(manifest: BulmaWorkspaceManifest) -> WorkObject:
    stage = WorkStage.ARTIFACT
    if manifest.meta.get("has_subdirectories") or len(manifest.files) > 3:
        stage = WorkStage.PROJECT
    elif manifest.source_dir and "task" in manifest.source_dir.lower():
        stage = WorkStage.MICRO_TASK

    created_at = (
        datetime.fromtimestamp(manifest.created_at / 1000.0, tz=timezone.utc)
        if manifest.created_at
        else datetime.now(timezone.utc)
    )

    return WorkObject(
        created_at=created_at,
        updated_at=created_at,
        stage=stage,
        content=f"Project: {manifest.project_name}\nFiles: {', '.join(manifest.files)}",
        meta={
            "bulma_source_dir": manifest.source_dir,
            "bulma_status": manifest.status,
            **manifest.meta,
        },
    )


def build_temporal_edge(prev: Episode, curr: Episode) -> Optional[EpisodeEdge]:
    """Create a simple temporal edge between two chronologically adjacent episodes."""
    if not prev.session_id or not curr.session_id:
        return None
    if prev.session_id != curr.session_id:
        return None
    emotional_weight = 0.0
    structural_weight = 0.0
    if prev.affect and curr.affect:
        if prev.affect.primary_emotion == curr.affect.primary_emotion:
            emotional_weight = 0.8
        ep_project = prev.payload.get("project_id")
        curr_project = curr.payload.get("project_id")
        if ep_project and curr_project and ep_project == curr_project:
            structural_weight = 0.6
    return EpisodeEdge(
        source_id=str(prev.episode_id),
        target_id=str(curr.episode_id),
        recency_weight=1.0,
        emotional_weight=emotional_weight,
        structural_weight=structural_weight,
        confidence=round(0.5 + (emotional_weight * 0.2) + (structural_weight * 0.1), 3),
    )


def map_bulma_memory_edge(
    bulma: BulmaMemoryEdge,
    id_map: Optional[Dict[str, str]] = None,
) -> EpisodeEdge:
    """Map a Bulma memory graph edge into an OpenCAS EpisodeEdge."""
    id_map = id_map or {}
    return EpisodeEdge(
        source_id=id_map.get(bulma.sourceId, str(bulma_episode_uuid(bulma.sourceId))),
        target_id=id_map.get(bulma.targetId, str(bulma_episode_uuid(bulma.targetId))),
        semantic_weight=round(max(0.0, min(1.0, bulma.semanticWeight)), 6),
        emotional_weight=round(max(0.0, min(1.0, bulma.emotionalResonanceWeight)), 6),
        recency_weight=round(max(0.0, min(1.0, bulma.recencyWeight)), 6),
        structural_weight=round(max(0.0, min(1.0, bulma.salienceWeight)), 6),
        confidence=round(max(0.0, min(1.0, bulma.confidence)), 6),
        created_at=_parse_dt(bulma.lastUpdatedMs) if bulma.lastUpdatedMs is not None else datetime.now(timezone.utc),
    )


def map_bulma_somatic_state(bulma: BulmaSomaticState) -> SomaticSnapshot:
    """Map Bulma's current somatic snapshot into OpenCAS SomaticSnapshot."""
    emotion_lower = bulma.primaryEmotion.lower()
    primary_emotion = _BULMA_EMOTION_MAP.get(emotion_lower, PrimaryEmotion.NEUTRAL)

    recorded_at = datetime.now(timezone.utc)
    if bulma.updatedAtMs is not None:
        recorded_at = datetime.fromtimestamp(bulma.updatedAtMs / 1000.0, tz=timezone.utc)

    return SomaticSnapshot(
        recorded_at=recorded_at,
        arousal=round(max(0.0, min(1.0, bulma.arousal)), 3),
        fatigue=round(max(0.0, min(1.0, bulma.fatigue)), 3),
        tension=round(max(0.0, min(1.0, bulma.stress)), 3),
        valence=round(max(-1.0, min(1.0, bulma.valence)), 3),
        focus=round(max(0.0, min(1.0, bulma.focus)), 3),
        energy=round(max(0.0, min(1.0, bulma.energy or 0.5)), 3),
        musubi=round(max(-1.0, min(1.0, bulma.musubi or 0.0)), 3) if bulma.musubi is not None else None,
        primary_emotion=primary_emotion,
        somatic_tag=bulma.source if bulma.source else None,
        certainty=round(max(0.0, min(1.0, bulma.certainty)), 3),
        source="bulma-import",
    )


def map_bulma_skill(bulma: BulmaSkillEntry) -> SkillEntry:
    """Map a Bulma skill registry entry into an OpenCAS SkillEntry."""
    return SkillEntry(
        skill_id=bulma.id,
        name=bulma.name or bulma.id,
        description=bulma.description,
        entrypoint=bulma.skillFile,
        capabilities=list(bulma.tags),
        parameters={"version": bulma.version, "source": bulma.source, "enabled": bulma.enabled},
        meta={"bulma_install_path": bulma.installPath, "installed_at_ms": bulma.installedAtMs},
    )


def _map_risk_class(risk_class: str) -> ActionRiskTier:
    """Map Bulma risk class strings to OpenCAS ActionRiskTier."""
    mapping = {
        "safe": ActionRiskTier.READONLY,
        "read_only": ActionRiskTier.READONLY,
        "destructive_fs_ops": ActionRiskTier.DESTRUCTIVE,
        "network": ActionRiskTier.NETWORK,
        "privilege_escalation": ActionRiskTier.SHELL_LOCAL,
        "unsafe_shell": ActionRiskTier.SHELL_LOCAL,
        "high_risk": ActionRiskTier.DESTRUCTIVE,
        "unknown": ActionRiskTier.READONLY,
    }
    return mapping.get(risk_class.lower(), ActionRiskTier.READONLY)


def map_bulma_action_approval(bulma: BulmaActionApproval) -> ApprovalLedgerEntry:
    """Map a Bulma action approval into an OpenCAS ApprovalLedgerEntry."""
    from uuid import uuid4

    ts = datetime.fromisoformat(bulma.ts.replace("Z", "+00:00"))
    return ApprovalLedgerEntry(
        entry_id=uuid4(),
        decision_id=UUID(bulma.id),
        action_id=UUID(bulma.id),
        level=bulma.status,
        score=1.0 if bulma.status == "approved" else 0.0,
        reasoning=bulma.reason,
        tool_name=bulma.metadata.get("taskId"),
        tier=_map_risk_class(bulma.riskClass),
        somatic_state=None,
        created_at=ts,
    )


def map_bulma_execution_receipt(bulma: BulmaExecutionReceipt) -> ExecutionReceipt:
    """Map a Bulma execution receipt into an OpenCAS ExecutionReceipt."""
    created_at = datetime.now(timezone.utc)
    if bulma.createdAtMs is not None:
        created_at = datetime.fromtimestamp(bulma.createdAtMs / 1000.0, tz=timezone.utc)

    phases: List[PhaseRecord] = []
    if bulma.capabilityIntent:
        plan_notes = bulma.capabilityIntent.get("planningNotes", [])
        if plan_notes:
            phases.append(
                PhaseRecord(
                    phase=ExecutionPhase.PLAN,
                    output="\n".join(plan_notes),
                    success=True,
                )
            )
        verify_notes = bulma.capabilityIntent.get("verificationChecklist", [])
        if verify_notes:
            phases.append(
                PhaseRecord(
                    phase=ExecutionPhase.VERIFY,
                    output="\n".join(verify_notes),
                    success=bool(bulma.verificationPassed),
                )
            )

    task_id = UUID(bulma.taskId) if bulma.taskId and _is_uuid(bulma.taskId) else uuid5(NAMESPACE_OID, bulma.taskId or bulma.id or str(created_at.timestamp()))
    return ExecutionReceipt(
        task_id=task_id,
        objective=bulma.objective,
        plan=bulma.summary or None,
        phases=phases,
        verification_result=bulma.verificationPassed,
        created_at=created_at,
        completed_at=created_at,
        success=bulma.status == "completed",
        output=bulma.summary,
    )


def map_bulma_research_notebook(bulma: BulmaResearchNotebook) -> ResearchNotebook:
    """Map a Bulma research notebook into an OpenCAS ResearchNotebook."""
    created_at = datetime.now(timezone.utc)
    updated_at = created_at
    if bulma.createdAtMs is not None:
        created_at = datetime.fromtimestamp(bulma.createdAtMs / 1000.0, tz=timezone.utc)
    if bulma.updatedAtMs is not None:
        updated_at = datetime.fromtimestamp(bulma.updatedAtMs / 1000.0, tz=timezone.utc)

    return ResearchNotebook(
        notebook_id=UUID(bulma.id) if _is_uuid(bulma.id) else uuid5(NAMESPACE_OID, bulma.id),
        created_at=created_at,
        updated_at=updated_at,
        title=bulma.objective[:120] if bulma.objective else "Imported notebook",
        description=bulma.objective,
        meta={
            "bulma_work_product_id": bulma.workProductId,
            "bulma_repo_path": bulma.repoPath,
            "bulma_channel": bulma.channel,
            "bulma_status": bulma.status,
            **bulma.metadata,
        },
    )


def map_bulma_objective_loop(bulma: BulmaObjectiveLoop) -> ObjectiveLoop:
    """Map a Bulma objective loop into an OpenCAS ObjectiveLoop."""
    created_at = datetime.now(timezone.utc)
    updated_at = created_at
    if bulma.createdAtMs is not None:
        created_at = datetime.fromtimestamp(bulma.createdAtMs / 1000.0, tz=timezone.utc)
    if bulma.updatedAtMs is not None:
        updated_at = datetime.fromtimestamp(bulma.updatedAtMs / 1000.0, tz=timezone.utc)

    status = ObjectiveStatus.PENDING
    if bulma.status == "active":
        status = ObjectiveStatus.ACTIVE
    elif bulma.status == "completed":
        status = ObjectiveStatus.COMPLETED
    elif bulma.status == "paused":
        status = ObjectiveStatus.PAUSED
    elif bulma.status == "failed":
        status = ObjectiveStatus.FAILED

    return ObjectiveLoop(
        loop_id=UUID(bulma.id) if _is_uuid(bulma.id) else uuid5(NAMESPACE_OID, bulma.id),
        created_at=created_at,
        updated_at=updated_at,
        status=status,
        title=bulma.objective[:120] if bulma.objective else "Imported loop",
        description=bulma.objective,
        meta={
            "bulma_plan_id": bulma.planId,
            "bulma_work_product_id": bulma.workProductId,
            "bulma_dispatch_id": bulma.dispatchId,
            "bulma_repo_path": bulma.repoPath,
            "max_iterations": bulma.maxIterations,
            "iteration_count": bulma.iterationCount,
            **bulma.metadata,
        },
    )


def map_bulma_task_plan(bulma: BulmaTaskPlan) -> WorkObject:
    """Map a Bulma task plan into an OpenCAS WorkObject."""
    created_at = datetime.now(timezone.utc)
    updated_at = created_at
    if bulma.createdAtMs is not None:
        created_at = datetime.fromtimestamp(bulma.createdAtMs / 1000.0, tz=timezone.utc)
    if bulma.updatedAtMs is not None:
        updated_at = datetime.fromtimestamp(bulma.updatedAtMs / 1000.0, tz=timezone.utc)

    dependency_ids = [item.id for item in bulma.items if item.status != "pending"]
    blocked_by = [item.id for item in bulma.items if item.status == "blocked"]

    return WorkObject(
        work_id=UUID(bulma.id) if _is_uuid(bulma.id) else uuid5(NAMESPACE_OID, bulma.id),
        created_at=created_at,
        updated_at=updated_at,
        stage=WorkStage.MICRO_TASK,
        content=bulma.objective,
        dependency_ids=dependency_ids,
        blocked_by=blocked_by,
        meta={
            "bulma_work_product_id": bulma.workProductId,
            "bulma_dispatch_id": bulma.dispatchId,
            "bulma_repo_path": bulma.repoPath,
            "bulma_channel": bulma.channel,
            "bulma_items": [item.model_dump(mode="json") for item in bulma.items],
            "bulma_checkpoints": bulma.checkpoints,
            **bulma.metadata,
        },
    )


def map_bulma_session_message(msg: BulmaSessionMessage) -> MessageEntry:
    """Map a Bulma session message into an OpenCAS MessageEntry."""
    role = MessageRole.USER
    if msg.role.lower() == "assistant":
        role = MessageRole.ASSISTANT
    elif msg.role.lower() == "system":
        role = MessageRole.SYSTEM

    created_at = datetime.now(timezone.utc)
    if msg.timestampMs is not None:
        created_at = datetime.fromtimestamp(msg.timestampMs / 1000.0, tz=timezone.utc)

    return MessageEntry(
        role=role,
        content=msg.content,
        created_at=created_at,
        meta={"bulma_emotion": msg.emotion} if msg.emotion else {},
    )


def map_bulma_executive_event(bulma: Any) -> Episode:
    """Map a Bulma executive event into a synthetic OpenCAS Episode."""
    if isinstance(bulma, dict):
        ts = bulma.get("ts", datetime.now(timezone.utc).isoformat())
        event_type = bulma.get("type", "unknown")
        details = bulma.get("details", {})
    else:
        ts = bulma.ts
        event_type = bulma.type
        details = bulma.details

    created_at = datetime.now(timezone.utc)
    if isinstance(ts, (int, float)):
        created_at = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
    elif isinstance(ts, str):
        try:
            created_at = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            pass

    label = details.get("label", "") if isinstance(details, dict) else ""
    entity = details.get("entity", "") if isinstance(details, dict) else ""
    content = f"Executive event: {event_type}"
    if label:
        content += f" | {label}"
    if entity:
        content += f" ({entity})"

    return Episode(
        created_at=created_at,
        kind=EpisodeKind.OBSERVATION,
        session_id="bulma-executive-events",
        content=content,
        salience=0.5,
        payload={"bulma_event_type": event_type, "bulma_details": details},
    )


def map_bulma_emotion_history(entry: BulmaEmotionHistoryEntry) -> Episode:
    """Map a Bulma emotion history entry into a synthetic Episode."""
    created_at = datetime.now(timezone.utc)
    if entry.timestampMs is not None:
        created_at = datetime.fromtimestamp(entry.timestampMs / 1000.0, tz=timezone.utc)

    return Episode(
        created_at=created_at,
        kind=EpisodeKind.OBSERVATION,
        session_id="bulma-emotion-history",
        content=f"Emotion update for {entry.episodeId or 'unknown'}: {entry.reason}",
        salience=0.3,
        payload=entry.model_dump(mode="json"),
    )


def map_bulma_consolidation_report(report: BulmaConsolidationReport) -> Episode:
    """Map a Bulma consolidation report into a synthetic Episode."""
    created_at = datetime.now(timezone.utc)
    if report.timestampMs is not None:
        created_at = datetime.fromtimestamp(report.timestampMs / 1000.0, tz=timezone.utc)

    return Episode(
        created_at=created_at,
        kind=EpisodeKind.OBSERVATION,
        session_id="bulma-consolidation-reports",
        content=f"Consolidation run {report.runId}: merged={report.clustersMerged}, rejected={report.clustersRejected}. {report.summary}",
        salience=0.3,
        payload=report.model_dump(mode="json"),
    )


def map_bulma_goal_thread(
    thread: BulmaGoalThread,
    id_map: Optional[Dict[str, str]] = None,
) -> WorkObject:
    """Map a Bulma goal thread into an OpenCAS WorkObject."""
    created_at = datetime.now(timezone.utc)
    if thread.timestampMs is not None:
        created_at = datetime.fromtimestamp(thread.timestampMs / 1000.0, tz=timezone.utc)

    stage = WorkStage.MICRO_TASK
    if thread.status in {"completed", "done"}:
        stage = WorkStage.ARTIFACT
    elif thread.status in {"archived"}:
        stage = WorkStage.SPARK

    goal_id = thread.goalId or thread.threadId or str(thread.timestampMs or 0)
    id_map = id_map or {}
    source_memory_ids = [
        id_map.get(episode_id, str(bulma_episode_uuid(episode_id)))
        for episode_id in thread.episodeIds
    ]

    return WorkObject(
        work_id=UUID(goal_id) if _is_uuid(goal_id) else uuid5(NAMESPACE_OID, goal_id),
        created_at=created_at,
        updated_at=created_at,
        stage=stage,
        content=thread.goalLabel,
        source_memory_ids=source_memory_ids,
        meta={
            "bulma_thread_id": thread.threadId,
            "bulma_status": thread.status,
            "bulma_source_episode_ids": list(thread.episodeIds),
        },
    )
