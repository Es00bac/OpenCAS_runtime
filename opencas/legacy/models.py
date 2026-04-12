"""Pydantic models mirroring OpenBulma v4 JSONL/JSON state structures."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class BulmaEmotion(BaseModel):
    """Emotion object inside a Bulma episode."""

    primaryEmotion: str = "neutral"
    valence: float = 0.0
    arousal: float = 0.5
    certainty: float = 0.5
    emotionalIntensity: float = 0.5
    socialTarget: str = "other"
    emotionTags: List[str] = Field(default_factory=list)


class BulmaEpisodeMetadataV3(BaseModel):
    """v3 migration metadata nested inside Bulma episode metadata."""

    model_config = ConfigDict(extra="allow")

    source: str = ""
    file: str = ""
    sensitivity: str = "public"
    hash: str = ""
    entities: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    emotionalValence: float = 0.0
    encodedMetadata: Any = None


class BulmaEpisodeMetadata(BaseModel):
    """Top-level metadata object on a Bulma episode."""

    model_config = ConfigDict(extra="allow")

    v3: Optional[BulmaEpisodeMetadataV3] = None


class BulmaEpisode(BaseModel):
    """Single record from Bulma state/memory/episodes.jsonl."""

    model_config = ConfigDict(extra="allow")

    id: str
    timestampMs: float
    source: str
    textContent: str
    emotion: Optional[BulmaEmotion] = None
    salience: float = 1.0
    identityCore: bool = False
    metadata: Optional[BulmaEpisodeMetadata] = None


class BulmaMemoryAnchor(BaseModel):
    """Anchor excerpt used in identity rebuild."""

    source: str
    timestampMs: float
    excerpt: str
    classification: str
    reason: str


class BulmaRecentActivity(BaseModel):
    """Activity entry in the Bulma identity profile."""

    timestampMs: float
    type: str
    label: str
    outcome: str = ""


class BulmaPartner(BaseModel):
    """Partner block inside Bulma identity."""

    userId: str
    trust: float
    musubi: float


class BulmaIdentityProfile(BaseModel):
    """Contents of Bulma state/identity/profile.json."""

    updatedAtMs: float
    coreNarrative: str = ""
    values: List[str] = Field(default_factory=list)
    ongoingGoals: List[str] = Field(default_factory=list)
    traits: List[str] = Field(default_factory=list)
    partner: Optional[BulmaPartner] = None
    recentThemes: List[Dict[str, Any]] = Field(default_factory=list)
    memoryAnchors: List[BulmaMemoryAnchor] = Field(default_factory=list)
    recentActivities: List[BulmaRecentActivity] = Field(default_factory=list)


class BulmaSpark(BaseModel):
    """Single record from Bulma state/daydream/sparks.jsonl."""

    id: str
    timestampMs: float
    mode: str
    trigger: str
    interest: str
    summary: str
    label: str
    kind: str
    intensity: float
    objective: str
    tags: List[str] = Field(default_factory=list)
    taskId: Optional[str] = None


class BulmaDaydreamInitiative(BaseModel):
    """Single record from Bulma state/daydream/initiatives.jsonl."""

    id: str
    sparkId: Optional[str] = None
    timestampMs: float
    mode: str = ""
    trigger: str = ""
    interest: str = ""
    summary: str = ""
    label: str = ""
    kind: str = ""
    intensity: float = 0.0
    rung: str = ""
    desiredRung: str = ""
    objective: str = ""
    focus: str = ""
    sourceKind: str = ""
    sourceLabel: str = ""
    artifactPaths: List[str] = Field(default_factory=list)
    taskId: Optional[str] = None
    routeDebug: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class BulmaDaydreamOutcome(BaseModel):
    """Single record from Bulma state/daydream/spark_outcomes.jsonl."""

    taskId: str
    outcome: str = ""
    valueDelivered: bool = False
    recordedAtMs: float


class BulmaDaydreamNotification(BaseModel):
    """Single record from Bulma state/daydream/notifications.jsonl."""

    sparkId: Optional[str] = None
    chatId: Optional[str] = None
    sentAtMs: float
    label: str = ""
    intensity: float = 0.0
    kind: str = ""


class BulmaHistoryEntry(BaseModel):
    """Single record from Bulma state/daydream/history.jsonl."""

    timestampMs: float
    mode: str
    trigger: str
    boredom: float = 0.0
    motivation: float = 0.0
    interest: str
    summary: str
    tags: List[str] = Field(default_factory=list)


class BulmaRelationship(BaseModel):
    """Contents of Bulma state/relationship.json."""

    userId: Optional[str] = None
    trust: Optional[float] = None
    musubi: Optional[float] = None
    warmth: Optional[float] = None


class BulmaSomaticState(BaseModel):
    """Contents of Bulma state/somatic/current.json."""

    primaryEmotion: str = "neutral"
    valence: float = 0.0
    arousal: float = 0.0
    certainty: float = 0.5
    intensity: float = 0.0
    stress: float = 0.0
    fatigue: float = 0.0
    focus: float = 0.0
    updatedAtMs: Optional[float] = None
    source: str = "unknown"
    musubi: Optional[float] = None
    energy: Optional[float] = None


class BulmaMemoryEdge(BaseModel):
    """Single record from Bulma state/memory/edges.jsonl."""

    sourceId: str
    targetId: str
    semanticWeight: float = 0.0
    emotionalResonanceWeight: float = 0.0
    recencyWeight: float = 0.0
    salienceWeight: float = 0.0
    confidence: float = 0.0
    lastUpdatedMs: Optional[float] = None


class BulmaGoal(BaseModel):
    """Single record from Bulma state/executive/goals.json."""

    id: str
    label: str
    status: str = "active"
    createdAtMs: float
    updatedAtMs: float
    importanceScore: float = 1.0
    source: str = "unknown"
    tags: List[str] = Field(default_factory=list)


class BulmaCommitment(BaseModel):
    """Single record from Bulma state/executive/commitments.json."""

    id: str
    goalId: Optional[str] = None
    label: str
    status: str = "active"
    executionState: Optional[str] = None
    verificationState: Optional[str] = None
    closureState: Optional[str] = None
    blockedReason: Optional[str] = None
    createdAtMs: float
    updatedAtMs: float
    source: str = "unknown"


class BulmaWorkspaceSnapshot(BaseModel):
    """Contents of Bulma state/executive/workspace.json snapshot block."""

    updatedAtMs: float
    focus: Optional[Dict[str, Any]] = None


class BulmaWorkspaceManifest(BaseModel):
    """Synthesized or loaded workspace manifest from Bulma workspaces/."""

    project_name: str
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    status: str = "imported"
    source_dir: str = ""
    files: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class BulmaSkillEntry(BaseModel):
    """Single skill entry from Bulma state/skills/registry.json."""

    id: str
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    tags: List[str] = Field(default_factory=list)
    source: str = "local"
    installPath: Optional[str] = None
    skillFile: Optional[str] = None
    enabled: bool = True
    installedAtMs: Optional[float] = None


class BulmaActionApproval(BaseModel):
    """Single record from Bulma state/governance/action_approvals.jsonl."""

    id: str
    ts: str
    riskClass: str = "unknown"
    operation: str = ""
    reason: str = ""
    actor: str = "bulma"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    approvals: Dict[str, str] = Field(default_factory=dict)
    status: str = "approved"


class BulmaExecutionReceipt(BaseModel):
    """Single receipt from Bulma state/execution-receipts/*.json."""

    id: str
    createdAtMs: Optional[float] = None
    source: str = "background_task"
    kind: str = "task_completion"
    taskId: Optional[str] = None
    dispatchId: Optional[str] = None
    workProductIds: List[str] = Field(default_factory=list)
    parentWorkProductId: Optional[str] = None
    repoPath: Optional[str] = None
    objective: str = ""
    summary: str = ""
    status: str = "completed"
    verificationPassed: Optional[bool] = None
    capabilityIntent: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BulmaResearchNotebook(BaseModel):
    """Single notebook from Bulma state/research-notebooks/*.json."""

    id: str
    workProductId: Optional[str] = None
    objective: str = ""
    repoPath: Optional[str] = None
    channel: Optional[str] = None
    capabilityIntent: Dict[str, Any] = Field(default_factory=dict)
    status: str = "planned"
    sourceIds: List[str] = Field(default_factory=list)
    createdAtMs: Optional[float] = None
    updatedAtMs: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BulmaObjectiveLoop(BaseModel):
    """Single loop from Bulma state/objective-loops/*.json."""

    id: str
    planId: Optional[str] = None
    workProductId: Optional[str] = None
    dispatchId: Optional[str] = None
    objective: str = ""
    repoPath: Optional[str] = None
    capabilityIntent: Dict[str, Any] = Field(default_factory=dict)
    status: str = "planned"
    maxIterations: int = 8
    iterationCount: int = 0
    checkpoints: List[Any] = Field(default_factory=list)
    createdAtMs: Optional[float] = None
    updatedAtMs: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BulmaTaskPlanItem(BaseModel):
    """Single item inside a Bulma task plan."""

    id: str
    label: str = ""
    status: str = "pending"
    updatedAtMs: Optional[float] = None


class BulmaTaskPlan(BaseModel):
    """Single task plan from Bulma state/task-plans/*.json."""

    id: str
    workProductId: Optional[str] = None
    dispatchId: Optional[str] = None
    objective: str = ""
    repoPath: Optional[str] = None
    channel: Optional[str] = None
    status: str = "planned"
    capabilityIntent: Dict[str, Any] = Field(default_factory=dict)
    items: List[BulmaTaskPlanItem] = Field(default_factory=list)
    checkpoints: List[Any] = Field(default_factory=list)
    createdAtMs: Optional[float] = None
    updatedAtMs: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BulmaSessionMessage(BaseModel):
    """Single message inside a Bulma session."""

    role: str = "user"
    content: str = ""
    timestampMs: Optional[float] = None
    emotion: Optional[str] = None


class BulmaSession(BaseModel):
    """Single session from Bulma state/sessions/*.json."""

    id: str
    type: str = "main"
    channelType: Optional[str] = None
    peerId: Optional[str] = None
    createdAtMs: Optional[float] = None
    lastActiveMs: Optional[float] = None
    contextHistory: List[BulmaSessionMessage] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BulmaExecutiveEvent(BaseModel):
    """Single event from Bulma state/executive/events.jsonl."""

    ts: str
    type: str = "unknown"
    details: Dict[str, Any] = Field(default_factory=dict)


class BulmaMusubiState(BaseModel):
    """Contents of Bulma state/somatic/musubi.json."""

    lastRelationalContactAtMs: Optional[float] = None
    lastMicroGainAtMs: Optional[float] = None
    microGainWindowAccumulated: float = 0.0
    lastCollaborativeSuccessAtMs: Optional[float] = None
    lastAbsenceDecayAtMs: Optional[float] = None


class BulmaEmotionHistoryEntry(BaseModel):
    """Single record from Bulma state/memory/emotion_history.jsonl."""

    episodeId: Optional[str] = None
    timestampMs: Optional[float] = None
    previous: Optional[Dict[str, Any]] = None
    next: Optional[Dict[str, Any]] = None
    reason: str = ""
    consolidationRunId: Optional[str] = None


class BulmaConsolidationReport(BaseModel):
    """Single record from Bulma state/memory/consolidation_reports.jsonl."""

    runId: Optional[str] = None
    timestampMs: Optional[float] = None
    clustersMerged: int = 0
    clustersRejected: int = 0
    summary: str = ""


class BulmaGoalThread(BaseModel):
    """Single record from Bulma state/memory/goal_threads.jsonl."""

    threadId: Optional[str] = None
    timestampMs: Optional[float] = None
    goalId: Optional[str] = None
    goalLabel: str = ""
    status: str = "active"
    episodeIds: List[str] = Field(default_factory=list)
