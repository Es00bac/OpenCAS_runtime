from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from opencas.execution.models import AttemptOutcome, AttemptSalvagePacket, ExecutionStage, RepairTask, RetryMode
from opencas.recovery.classifier import RecoveryClassifier
from opencas.recovery.continuity import ContinuityPacketBuilder
from opencas.recovery.coordinator import AutonomousRecoveryCoordinator
from opencas.recovery.executor import RecoveryExecutorAdapter
from opencas.recovery.planner import RecoveryPlanner
from opencas.recovery.scanner import RecoveryScanner
from opencas.recovery.models import (
    ContinuityPacket,
    RecoveryCandidate,
    RecoveryCandidateKind,
    RecoveryClassification,
    RecoveryDecision,
    RecoveryPlan,
    RecoveryStrategy,
)


def make_candidate(candidate_id: str) -> RecoveryCandidate:
    return RecoveryCandidate(
        candidate_id=candidate_id,
        kind=RecoveryCandidateKind.FAILED_TASK,
        title="Failed task",
        status="failed",
        updated_at=datetime.now(timezone.utc),
        evidence_refs=[candidate_id],
        payload={},
    )


@dataclass
class FakeScanner:
    candidates: list[RecoveryCandidate]

    async def scan(self, *, limit_per_store: int):
        return self.candidates


@dataclass
class FakeClassifier:
    classifications: dict[str, RecoveryClassification] = field(default_factory=dict)
    classified: list[str] = field(default_factory=list)

    def classify(self, candidate):
        self.classified.append(candidate.candidate_id)
        return self.classifications.get(
            candidate.candidate_id,
            RecoveryClassification(candidate.candidate_id, "resume_now", "resume", evidence_refs=candidate.evidence_refs),
        )


@dataclass
class FakePlanner:
    planned: list[tuple[str, ContinuityPacket | None]] = field(default_factory=list)

    def plan(self, candidate, classification, *, continuity_packet=None):
        self.planned.append((candidate.candidate_id, continuity_packet))
        return RecoveryPlan(
            candidate.candidate_id,
            RecoveryStrategy.DETERMINISTIC_REVIEW,
            "Review artifact",
            continuity_packet=continuity_packet,
            evidence_refs=candidate.evidence_refs,
        )


@dataclass
class FakeExecutor:
    results: list[str] = field(default_factory=list)
    decisions: list[RecoveryDecision] = field(default_factory=list)

    async def execute(self, plan):
        result = self.results[len(self.decisions)] if len(self.decisions) < len(self.results) else "submitted"
        decision = RecoveryDecision(
            decision_id=f"decision-{len(self.decisions) + 1}",
            candidate_id=plan.candidate_id,
            classification="resume_now",
            strategy=plan.strategy.value,
            result=result,
            continuity_packet_id=plan.continuity_packet.packet_id if plan.continuity_packet is not None else None,
            evidence_refs=plan.evidence_refs,
        )
        self.decisions.append(decision)
        return decision


@dataclass
class FakeLedger:
    skip_candidate_ids: set[str] = field(default_factory=set)
    checked: list[str] = field(default_factory=list)
    recorded: list[RecoveryDecision] = field(default_factory=list)

    async def should_skip_candidate(self, candidate_id):
        self.checked.append(candidate_id)
        return candidate_id in self.skip_candidate_ids

    async def record_decision(self, decision):
        self.recorded.append(decision)


@dataclass
class FakeContinuityBuilder:
    packet: ContinuityPacket
    built: list[str] = field(default_factory=list)

    async def build(self, candidate):
        self.built.append(candidate.candidate_id)
        return self.packet


@dataclass
class FakeTaskStore:
    tasks: list[RepairTask] = field(default_factory=list)
    packets: dict[str, AttemptSalvagePacket] = field(default_factory=dict)

    async def list_all(self, limit: int = 100):
        return self.tasks[:limit]

    async def get_latest_salvage_packet(self, task_id):
        return self.packets.get(str(task_id))


@dataclass
class FakeBAA:
    submitted: list[RepairTask] = field(default_factory=list)

    async def submit(self, task):
        self.submitted.append(task)
        return "future"


@dataclass
class FakeHarness:
    loops: list[dict] = field(default_factory=list)

    async def create_objective_loop(self, **kwargs):
        self.loops.append(kwargs)
        return SimpleNamespace(loop_id="loop-new")


@pytest.mark.asyncio
async def test_coordinator_submits_stale_failed_task_with_salvage_alternate_route() -> None:
    now = datetime.now(timezone.utc)
    failed_task = RepairTask(
        task_id=uuid4(),
        objective="Finish app",
        stage=ExecutionStage.FAILED,
        meta={"recovery_strategy": RetryMode.CONTINUE_RETRY.value},
    )
    packet = AttemptSalvagePacket(
        packet_id=uuid4(),
        task_id=failed_task.task_id,
        attempt=2,
        objective="Finish app",
        divergence_signature="same_tool_loop",
        outcome=AttemptOutcome.FAILED,
        best_next_step="Review existing artifact before another repair attempt.",
        recommended_mode=RetryMode.DETERMINISTIC_REVIEW,
        created_at=now,
    )
    baa = FakeBAA()
    task_store = FakeTaskStore(tasks=[failed_task], packets={str(failed_task.task_id): packet})
    coordinator = AutonomousRecoveryCoordinator(
        scanner=RecoveryScanner(task_store=task_store),
        classifier=RecoveryClassifier(),
        continuity_builder=None,
        planner=RecoveryPlanner(),
        executor=RecoveryExecutorAdapter(baa=baa),
        ledger=FakeLedger(),
    )

    summary = await coordinator.run_once(limit_per_store=10, max_actions=1)

    assert summary.submitted == 1
    submitted_task = baa.submitted[0]
    assert submitted_task.metadata["recovery_strategy"] == "deterministic_review"
    assert submitted_task.metadata["recovery_candidate_id"].startswith("task:")
    assert submitted_task.metadata["recovery_strategy"] != failed_task.meta["recovery_strategy"]


@pytest.mark.asyncio
async def test_coordinator_scans_classifies_executes_and_records_summary() -> None:
    candidate = make_candidate("task:abc")
    ledger = FakeLedger()
    executor = FakeExecutor()
    coordinator = AutonomousRecoveryCoordinator(
        scanner=FakeScanner([candidate]),
        classifier=FakeClassifier(),
        continuity_builder=None,
        planner=FakePlanner(),
        executor=executor,
        ledger=ledger,
    )

    summary = await coordinator.run_once(limit_per_store=10, max_actions=3)

    assert summary.scanned == 1
    assert summary.resume_now == 1
    assert summary.submitted == 1
    assert ledger.recorded[0].candidate_id == "task:abc"


@pytest.mark.asyncio
async def test_coordinator_honors_max_actions() -> None:
    candidates = [make_candidate(f"task:{index}") for index in range(5)]
    executor = FakeExecutor()
    coordinator = AutonomousRecoveryCoordinator(
        scanner=FakeScanner(candidates),
        classifier=FakeClassifier(),
        continuity_builder=None,
        planner=FakePlanner(),
        executor=executor,
        ledger=FakeLedger(),
    )

    summary = await coordinator.run_once(limit_per_store=10, max_actions=2)

    assert summary.scanned == 5
    assert summary.submitted == 2
    assert len(executor.decisions) == 2


@pytest.mark.asyncio
async def test_coordinator_skips_ledger_candidates_without_orchestration() -> None:
    skipped = make_candidate("task:skip")
    active = make_candidate("task:active")
    classifier = FakeClassifier()
    planner = FakePlanner()
    executor = FakeExecutor()
    ledger = FakeLedger(skip_candidate_ids={"task:skip"})
    coordinator = AutonomousRecoveryCoordinator(
        scanner=FakeScanner([skipped, active]),
        classifier=classifier,
        continuity_builder=None,
        planner=planner,
        executor=executor,
        ledger=ledger,
    )

    summary = await coordinator.run_once(limit_per_store=10, max_actions=3)

    assert summary.skipped_by_ledger == 1
    assert classifier.classified == ["task:active"]
    assert planner.planned == [("task:active", None)]
    assert [decision.candidate_id for decision in executor.decisions] == ["task:active"]
    assert [decision.candidate_id for decision in ledger.recorded] == ["task:active"]


@pytest.mark.asyncio
async def test_coordinator_counts_blocked_decisions_toward_max_actions() -> None:
    candidates = [make_candidate(f"task:{index}") for index in range(5)]
    executor = FakeExecutor(results=["blocked", "blocked", "blocked"])
    classifier = FakeClassifier()
    planner = FakePlanner()
    ledger = FakeLedger()
    coordinator = AutonomousRecoveryCoordinator(
        scanner=FakeScanner(candidates),
        classifier=classifier,
        continuity_builder=None,
        planner=planner,
        executor=executor,
        ledger=ledger,
    )

    summary = await coordinator.run_once(limit_per_store=10, max_actions=2)

    assert summary.scanned == 5
    assert summary.resume_now == 2
    assert summary.submitted == 0
    assert len(classifier.classified) == 2
    assert len(planner.planned) == 2
    assert [decision.result for decision in executor.decisions] == ["blocked", "blocked"]
    assert len(ledger.recorded) == 2


@pytest.mark.asyncio
async def test_coordinator_counts_scheduled_decisions_toward_max_actions_without_submitting() -> None:
    candidates = [make_candidate(f"schedule:{index}") for index in range(5)]
    classifications = {
        candidate.candidate_id: RecoveryClassification(
            candidate.candidate_id,
            "scheduled_or_recurring",
            "healthy recurring schedule",
            evidence_refs=candidate.evidence_refs,
        )
        for candidate in candidates
    }
    executor = FakeExecutor(results=["scheduled", "scheduled", "scheduled"])
    classifier = FakeClassifier(classifications=classifications)
    planner = FakePlanner()
    ledger = FakeLedger()
    coordinator = AutonomousRecoveryCoordinator(
        scanner=FakeScanner(candidates),
        classifier=classifier,
        continuity_builder=None,
        planner=planner,
        executor=executor,
        ledger=ledger,
    )

    summary = await coordinator.run_once(limit_per_store=10, max_actions=2)

    assert summary.scanned == 5
    assert summary.scheduled_or_recurring == 2
    assert summary.submitted == 0
    assert len(classifier.classified) == 2
    assert len(planner.planned) == 2
    assert [decision.result for decision in executor.decisions] == ["scheduled", "scheduled"]
    assert len(ledger.recorded) == 2


@pytest.mark.asyncio
async def test_coordinator_creates_creative_writing_continuity_then_resumes_loop() -> None:
    candidate = RecoveryCandidate(
        candidate_id="loop:story",
        kind=RecoveryCandidateKind.PAUSED_OBJECTIVE_LOOP,
        title="Continue creative writing project",
        status="paused",
        updated_at=datetime.now(timezone.utc),
        evidence_refs=["loop:story"],
        payload={
            "project_key": "story-project",
            "project_type": "creative_writing",
            "canonical_artifact_paths": ["workspace/writing/story.md"],
            "latest_completed_unit": "chapter 2",
            "current_status": "chapter 3 needs drafting",
            "continuity_facts": ["Character A left town in chapter 2."],
            "unresolved_threads": ["Character B has not learned why A left."],
            "next_concrete_action": "Draft chapter 3 opening scene.",
            "completion_criteria": ["chapter 3 draft exists"],
        },
    )
    harness = FakeHarness()
    coordinator = AutonomousRecoveryCoordinator(
        scanner=FakeScanner([candidate]),
        classifier=RecoveryClassifier(),
        continuity_builder=ContinuityPacketBuilder(),
        planner=RecoveryPlanner(),
        executor=RecoveryExecutorAdapter(harness=harness),
        ledger=FakeLedger(),
    )

    summary = await coordinator.run_once(limit_per_store=10, max_actions=1)

    assert summary.needs_artifact_continuity == 1
    assert summary.submitted == 1
    created_loop = harness.loops[0]
    assert created_loop["meta"]["recovery_strategy"] == "create_continuity_then_resume"
    assert created_loop["meta"]["recovery_candidate_id"] == "loop:story"
    assert "continuity:" in "|".join(created_loop["meta"]["evidence_refs"])


@pytest.mark.asyncio
async def test_coordinator_builds_continuity_packet_only_for_continuity_classification() -> None:
    continuity_candidate = make_candidate("task:continuity")
    resume_candidate = make_candidate("task:resume")
    packet = ContinuityPacket(
        packet_id="packet-1",
        project_key="project-alpha",
        title="Project Alpha",
        project_type="artifact",
        canonical_artifact_paths=["workspace/project-alpha.md"],
        latest_completed_unit="Collected evidence",
        current_status="needs handoff",
        continuity_facts=["The artifact exists."],
        unresolved_threads=["Write next section."],
        next_concrete_action="Resume the artifact.",
        completion_criteria=["Artifact updated."],
        evidence_refs=["task:continuity"],
    )
    classifier = FakeClassifier(
        classifications={
            "task:continuity": RecoveryClassification(
                "task:continuity",
                "needs_artifact_continuity",
                "needs continuity packet",
                evidence_refs=continuity_candidate.evidence_refs,
            ),
            "task:resume": RecoveryClassification(
                "task:resume",
                "resume_now",
                "resume directly",
                evidence_refs=resume_candidate.evidence_refs,
            ),
        }
    )
    continuity_builder = FakeContinuityBuilder(packet)
    planner = FakePlanner()
    coordinator = AutonomousRecoveryCoordinator(
        scanner=FakeScanner([continuity_candidate, resume_candidate]),
        classifier=classifier,
        continuity_builder=continuity_builder,
        planner=planner,
        executor=FakeExecutor(),
        ledger=FakeLedger(),
    )

    summary = await coordinator.run_once(limit_per_store=10, max_actions=3)

    assert summary.needs_artifact_continuity == 1
    assert summary.resume_now == 1
    assert continuity_builder.built == ["task:continuity"]
    assert planner.planned == [("task:continuity", packet), ("task:resume", None)]


@dataclass
class FakeActivitySink:
    events: list[dict] = field(default_factory=list)

    async def post_activity(self, *, agent: str, summary: str, status: str, files_changed: list[str], details: dict):
        self.events.append(
            {
                "agent": agent,
                "summary": summary,
                "status": status,
                "files_changed": files_changed,
                "details": details,
            }
        )


@pytest.mark.asyncio
async def test_coordinator_posts_structured_activity_summary() -> None:
    candidate = RecoveryCandidate(
        candidate_id="task:abc",
        kind=RecoveryCandidateKind.FAILED_TASK,
        title="Failed task",
        status="failed",
        updated_at=datetime.now(timezone.utc),
        evidence_refs=["task:abc"],
        payload={},
    )
    sink = FakeActivitySink()
    coordinator = AutonomousRecoveryCoordinator(
        scanner=FakeScanner([candidate]),
        classifier=FakeClassifier(),
        continuity_builder=None,
        planner=FakePlanner(),
        executor=FakeExecutor(),
        ledger=FakeLedger(),
        activity_sink=sink,
    )

    await coordinator.run_once(limit_per_store=10, max_actions=1)

    assert sink.events[0]["agent"] == "AutonomousRecoveryCoordinator"
    assert sink.events[0]["status"] == "completed"
    assert sink.events[0]["details"]["submitted"] == 1
