"""End-to-end import tests against a temporary Bulma-like state directory."""

import json
from pathlib import Path

import pytest
import pytest_asyncio

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.runtime import AgentRuntime
from opencas.legacy.importer import BulmaImportTask
from opencas.legacy.mapper import bulma_episode_uuid


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(state_dir=tmp_path, session_id="import-test")
    ctx = await BootstrapPipeline(config).run()
    runtime = AgentRuntime(ctx)
    yield runtime
    await runtime.ctx.close()


@pytest.mark.asyncio
async def test_import_task_imports_episodes(runtime: AgentRuntime, tmp_path: Path) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "memory").mkdir(parents=True)
    episodes = [
        {
            "id": "2026-02-19_0",
            "timestampMs": 1771485080000,
            "source": "v3:2026-02-19.md",
            "textContent": "Testing",
            "emotion": {
                "primaryEmotion": "neutral",
                "valence": 0,
                "arousal": 0.55,
                "certainty": 0.72,
                "emotionalIntensity": 0.55,
                "socialTarget": "other",
                "emotionTags": ["conversation", "testing"],
            },
            "salience": 0.62,
            "identityCore": False,
        },
        {
            "id": "2026-02-19_1",
            "timestampMs": 1771485081000,
            "source": "chat",
            "textContent": "hello",
            "emotion": {
                "primaryEmotion": "joy",
                "valence": 0.3,
                "arousal": 0.6,
                "certainty": 0.8,
                "emotionalIntensity": 0.6,
                "socialTarget": "user",
                "emotionTags": ["greeting"],
            },
            "salience": 0.75,
            "identityCore": True,
        },
    ]
    with (bulma_dir / "memory" / "episodes.jsonl").open("w") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep) + "\n")

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    assert report.episodes_imported == 2
    assert report.edges_imported >= 0

    # Verify episodes are in the store
    eps = await runtime.memory.list_episodes(limit=10)
    # Note: the importer also adds a synthetic boot episode
    assert len(eps) == 3


@pytest.mark.asyncio
async def test_import_task_imports_identity(runtime: AgentRuntime, tmp_path: Path) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "identity").mkdir(parents=True)
    profile = {
        "updatedAtMs": 1775417396065,
        "coreNarrative": "Bulma cares.",
        "values": ["continuity"],
        "ongoingGoals": ["assist"],
        "traits": ["patient"],
        "partner": {"userId": "jarrod", "trust": 0.92, "musubi": 1.0},
        "recentThemes": [],
        "memoryAnchors": [
            {
                "source": "soul:openbulma-v4/SOUL.md#1",
                "timestampMs": 1775417396065,
                "excerpt": "Bulma remembers.",
                "classification": "canon",
                "reason": "test",
            }
        ],
        "recentActivities": [],
    }
    (bulma_dir / "identity" / "profile.json").write_text(json.dumps(profile))
    audit = {
        "generatedAtMs": 1775417396066,
        "workspaceDerivedAnchorPolicy": "quarantine",
        "candidateCount": 1,
        "selectedAnchors": [],
        "quarantinedAnchors": [],
        "rejectedAnchors": [],
        "counts": {"byClassification": {"canon": 1}},
        "summary": ["Identity rebuild considered 1 candidate memory."],
    }
    (bulma_dir / "identity" / "rebuild-audit.json").write_text(json.dumps(audit))

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    assert runtime.ctx.identity.self_model.name == "Bulma"
    assert runtime.ctx.identity.self_model.narrative == "Bulma cares."
    assert runtime.ctx.identity.self_model.source_system == "openbulma-v4"
    assert runtime.ctx.identity.self_model.memory_anchors[0]["excerpt"] == "Bulma remembers."
    assert runtime.ctx.identity.self_model.identity_rebuild_audit["workspaceDerivedAnchorPolicy"] == "quarantine"
    assert runtime.ctx.identity.user_model.explicit_preferences.get("partner_user_id") == "jarrod"


@pytest.mark.asyncio
async def test_import_task_imports_daydreams(runtime: AgentRuntime, tmp_path: Path) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "daydream").mkdir(parents=True)
    spark = {
        "id": "spark-1",
        "timestampMs": 1773554000251,
        "mode": "reverie",
        "trigger": "manual",
        "interest": "odd online communities",
        "summary": "settling into the quiet",
        "label": "Research",
        "kind": "research",
        "intensity": 0.41,
        "objective": "Research this curiosity thread",
        "tags": ["daydream-spark"],
    }
    with (bulma_dir / "daydream" / "sparks.jsonl").open("w") as fh:
        fh.write(json.dumps(spark) + "\n")
    initiative = {
        "id": "initiative-1",
        "sparkId": "spark-1",
        "timestampMs": 1773554000300,
        "mode": "reverie",
        "trigger": "manual",
        "interest": "odd online communities",
        "summary": "settling into the quiet",
        "label": "Research",
        "kind": "research",
        "intensity": 0.41,
        "rung": "micro_task",
        "desiredRung": "micro_task",
        "objective": "Research this curiosity thread",
        "focus": "internet_archaeology",
        "sourceKind": "activity",
        "sourceLabel": "manual spark",
        "artifactPaths": ["/tmp/daydream.md"],
        "taskId": "task-1",
        "routeDebug": {"finalRung": "micro_task"},
        "tags": ["daydream-spark"],
    }
    with (bulma_dir / "daydream" / "initiatives.jsonl").open("w") as fh:
        fh.write(json.dumps(initiative) + "\n")
    outcome = {
        "taskId": "task-1",
        "outcome": "success",
        "valueDelivered": True,
        "recordedAtMs": 1773554000400,
    }
    with (bulma_dir / "daydream" / "spark_outcomes.jsonl").open("w") as fh:
        fh.write(json.dumps(outcome) + "\n")
    notification = {
        "sparkId": "spark-1",
        "chatId": "chat-1",
        "sentAtMs": 1773554000500,
        "label": "Research",
        "intensity": 0.41,
        "kind": "research",
    }
    with (bulma_dir / "daydream" / "notifications.jsonl").open("w") as fh:
        fh.write(json.dumps(notification) + "\n")

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    assert report.daydreams_imported == 1
    assert report.daydream_sparks_imported == 1
    assert report.daydream_initiatives_imported == 1
    assert report.daydream_outcomes_imported == 1
    assert report.daydream_notifications_imported == 1
    recent = await runtime.ctx.daydream_store.list_recent(limit=5)
    assert len(recent) == 1
    assert "odd online communities" in recent[0].spark_content
    lifecycle = await runtime.ctx.daydream_store.get_lifecycle_for_spark("spark-1")
    assert lifecycle["spark"]["spark_id"] == "spark-1"
    assert lifecycle["initiatives"][0]["artifact_paths"] == ["/tmp/daydream.md"]
    assert lifecycle["outcomes"][0]["value_delivered"] is True
    assert lifecycle["notifications"][0]["chat_id"] == "chat-1"


@pytest.mark.asyncio
async def test_import_task_is_resumable(runtime: AgentRuntime, tmp_path: Path) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "memory").mkdir(parents=True)
    episodes = [
        {
            "id": "ep-1",
            "timestampMs": 1771485080000,
            "source": "chat",
            "textContent": "first",
            "salience": 0.5,
            "identityCore": False,
        },
        {
            "id": "ep-2",
            "timestampMs": 1771485081000,
            "source": "chat",
            "textContent": "second",
            "salience": 0.6,
            "identityCore": False,
        },
    ]
    with (bulma_dir / "memory" / "episodes.jsonl").open("w") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep) + "\n")

    checkpoint = tmp_path / "checkpoint.json"
    task1 = BulmaImportTask(bulma_dir, runtime=runtime, checkpoint_store=checkpoint)
    report1 = await task1.run()
    assert report1.episodes_imported == 2

    # Running again with the same checkpoint should skip everything
    task2 = BulmaImportTask(bulma_dir, runtime=runtime, checkpoint_store=checkpoint)
    report2 = await task2.run()
    assert report2.episodes_imported == 0


@pytest.mark.asyncio
async def test_import_task_imports_somatic(runtime: AgentRuntime, tmp_path: Path) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "somatic").mkdir(parents=True)
    current = {
        "primaryEmotion": "anticipation",
        "valence": 0.5,
        "arousal": 0.2,
        "certainty": 0.6,
        "intensity": 0.4,
        "stress": 0.1,
        "fatigue": 0.7,
        "focus": 0.5,
        "source": "heartbeat-user-absence",
        "musubi": 0.8,
        "energy": 0.4,
    }
    (bulma_dir / "somatic" / "current.json").write_text(json.dumps(current))

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    assert report.somatic_snapshots_imported == 1
    latest = await runtime.ctx.somatic_store.get_latest()
    assert latest is not None
    assert latest.primary_emotion.value == "anticipation"
    assert latest.embedding_id is not None


@pytest.mark.asyncio
async def test_import_task_imports_explicit_edges(runtime: AgentRuntime, tmp_path: Path) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "memory").mkdir(parents=True)
    episodes = [
        {
            "id": "ep-a",
            "timestampMs": 1771485080000,
            "source": "chat",
            "textContent": "first",
            "salience": 0.5,
            "identityCore": False,
        },
        {
            "id": "ep-b",
            "timestampMs": 1771485081000,
            "source": "chat",
            "textContent": "second",
            "salience": 0.6,
            "identityCore": False,
        },
    ]
    with (bulma_dir / "memory" / "episodes.jsonl").open("w") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep) + "\n")

    edges = [
        {
            "sourceId": "ep-a",
            "targetId": "ep-b",
            "semanticWeight": 0.9,
            "emotionalResonanceWeight": 0.8,
            "recencyWeight": 0.7,
            "salienceWeight": 0.6,
            "confidence": 0.85,
            "lastUpdatedMs": 1771485082000,
        }
    ]
    with (bulma_dir / "memory" / "edges.jsonl").open("w") as fh:
        for edge in edges:
            fh.write(json.dumps(edge) + "\n")

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    assert report.edges_imported >= 1
    imported_source_id = str(bulma_episode_uuid("ep-a"))
    imported_edges = await runtime.memory.get_edges_for(imported_source_id)
    assert len(imported_edges) >= 1
    assert imported_edges[0].target_id == str(bulma_episode_uuid("ep-b"))
    assert await runtime.memory.get_episode(imported_source_id) is not None


@pytest.mark.asyncio
async def test_import_task_imports_executive_goals(runtime: AgentRuntime, tmp_path: Path) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "executive").mkdir(parents=True)
    goals = [
        {"id": "g1", "label": "stay healthy", "status": "active", "createdAtMs": 1, "updatedAtMs": 1},
        {"id": "g2", "label": "archived goal", "status": "archived", "createdAtMs": 1, "updatedAtMs": 1},
    ]
    (bulma_dir / "executive" / "goals.json").write_text(json.dumps(goals))

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    assert report.goals_imported == 1
    assert "stay healthy" in runtime.executive.active_goals
    assert "archived goal" not in runtime.executive.active_goals


@pytest.mark.asyncio
async def test_import_task_imports_commitments(runtime: AgentRuntime, tmp_path: Path) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "executive").mkdir(parents=True)
    commitments = [
        {
            "id": "c1",
            "goalId": "g1",
            "label": "active commitment",
            "status": "active",
            "createdAtMs": 1,
            "updatedAtMs": 1,
        },
        {
            "id": "c2",
            "goalId": "g1",
            "label": "released commitment",
            "status": "released",
            "createdAtMs": 1,
            "updatedAtMs": 1,
        },
    ]
    (bulma_dir / "executive" / "commitments.json").write_text(json.dumps(commitments))

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    assert report.commitments_imported == 2
    queued_labels = [w.content for w in runtime.executive.task_queue]
    assert "active commitment" in queued_labels
    assert "released commitment" not in queued_labels


@pytest.mark.asyncio
async def test_import_task_imports_skills(runtime: AgentRuntime, tmp_path: Path) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "skills").mkdir(parents=True)
    registry = {
        "installed": [
            {
                "id": "skill-creator",
                "name": "Skill Creator",
                "description": "Meta-skill for authoring skills.",
                "version": "1.0.0",
                "tags": ["meta", "skills"],
                "source": "local",
                "enabled": True,
                "installedAtMs": 1773209907366,
            }
        ]
    }
    (bulma_dir / "skills" / "registry.json").write_text(json.dumps(registry))

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    assert report.skills_imported == 1
    skill = runtime.ctx.skill_registry.get("skill-creator")
    assert skill is not None
    assert skill.name == "Skill Creator"


@pytest.mark.asyncio
async def test_import_task_imports_governance(runtime: AgentRuntime, tmp_path: Path) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "governance").mkdir(parents=True)
    approval = {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "ts": "2026-04-04T06:21:37.115Z",
        "riskClass": "destructive_fs_ops",
        "operation": "rm -rf /tmp/old",
        "reason": "owner-trust-auto",
        "actor": "bulma",
        "metadata": {"taskId": "t1"},
        "approvals": {"bulma": "2026-04-04T06:21:37.115Z"},
        "status": "approved",
    }
    with (bulma_dir / "governance" / "action_approvals.jsonl").open("w") as fh:
        fh.write(json.dumps(approval) + "\n")

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    assert report.governance_entries_imported == 1
    entries = await runtime.ctx.ledger.store.list_recent(limit=5)
    assert len(entries) == 1
    assert entries[0].reasoning == "owner-trust-auto"


@pytest.mark.asyncio
async def test_import_task_imports_execution_receipts(runtime: AgentRuntime, tmp_path: Path) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "execution-receipts").mkdir(parents=True)
    receipt = {
        "id": "r1",
        "createdAtMs": 1775035423923,
        "source": "background_task",
        "kind": "task_completion",
        "taskId": "550e8400-e29b-41d4-a716-446655440001",
        "objective": "Test objective",
        "summary": "All checks passed",
        "status": "completed",
        "verificationPassed": True,
        "capabilityIntent": {"planningNotes": ["Plan note"], "verificationChecklist": ["Check 1"]},
        "metadata": {},
    }
    (bulma_dir / "execution-receipts" / "execution-receipt-r1.json").write_text(json.dumps(receipt))

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    assert report.execution_receipts_imported == 1
    receipts = await runtime.ctx.receipt_store.list_by_task("550e8400-e29b-41d4-a716-446655440001")
    assert len(receipts) == 1
    assert receipts[0].success is True


@pytest.mark.asyncio
async def test_import_task_imports_harness(runtime: AgentRuntime, tmp_path: Path) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "research-notebooks" / "notebooks").mkdir(parents=True)
    (bulma_dir / "objective-loops").mkdir(parents=True)

    notebook = {
        "id": "nb1",
        "workProductId": "wp-nb1",
        "objective": "Research quantum computing",
        "status": "planned",
        "createdAtMs": 1775085613655,
        "updatedAtMs": 1775085613655,
        "metadata": {},
    }
    (bulma_dir / "research-notebooks" / "notebooks" / "nb1.json").write_text(json.dumps(notebook))

    loop = {
        "id": "loop1",
        "planId": "plan1",
        "workProductId": "wp-loop1",
        "dispatchId": "disp1",
        "objective": "Explore quantum algorithms",
        "status": "active",
        "maxIterations": 8,
        "iterationCount": 2,
        "checkpoints": [],
        "createdAtMs": 1775287248788,
        "updatedAtMs": 1775287248788,
        "metadata": {},
    }
    (bulma_dir / "objective-loops" / "loop1.json").write_text(json.dumps(loop))

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    assert report.research_notebooks_imported == 1
    assert report.objective_loops_imported == 1

    nbs = await runtime.ctx.harness.store.list_notebooks()
    assert len(nbs) == 1
    assert nbs[0].title.startswith("Research quantum")

    loops = await runtime.ctx.harness.store.list_loops()
    assert len(loops) == 1
    assert loops[0].title.startswith("Explore quantum")


@pytest.mark.asyncio
async def test_import_task_imports_relational(runtime: AgentRuntime, tmp_path: Path) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "identity").mkdir(parents=True)
    profile = {
        "updatedAtMs": 1775417396065,
        "coreNarrative": "Bulma cares.",
        "values": ["continuity"],
        "ongoingGoals": ["assist"],
        "traits": ["patient"],
        "partner": {"userId": "jarrod", "trust": 0.92, "musubi": 1.0},
        "recentThemes": [],
        "memoryAnchors": [],
        "recentActivities": [],
    }
    (bulma_dir / "identity" / "profile.json").write_text(json.dumps(profile))

    relationship = {
        "userId": "jarrod",
        "trust": 91.97,
        "musubi": 100,
        "warmth": 85.0,
        "interactionCount": 743,
    }
    (bulma_dir / "relationship.json").write_text(json.dumps(relationship))

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    assert report.relational_state_imported is True
    assert runtime.ctx.relational.state.musubi > 0.5


@pytest.mark.asyncio
async def test_import_task_imports_task_plans(runtime: AgentRuntime, tmp_path: Path) -> None:
    from uuid import NAMESPACE_OID, uuid5

    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "task-plans").mkdir(parents=True)
    plan = {
        "id": "plan1",
        "workProductId": "wp-plan1",
        "dispatchId": "disp1",
        "objective": "Deploy to staging",
        "repoPath": "/tmp",
        "channel": "api",
        "status": "planned",
        "items": [
            {"id": "item1", "label": "Build", "status": "completed", "updatedAtMs": 1},
            {"id": "item2", "label": "Test", "status": "pending", "updatedAtMs": 1},
        ],
        "checkpoints": [],
        "createdAtMs": 1775287248788,
        "updatedAtMs": 1775287248788,
        "metadata": {},
    }
    (bulma_dir / "task-plans" / "plan1.json").write_text(json.dumps(plan))

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    expected_work_id = str(uuid5(NAMESPACE_OID, "plan1"))
    assert report.task_plans_imported == 1
    work = await runtime.ctx.work_store.get(expected_work_id)
    assert work is not None
    assert "Deploy to staging" in work.content


@pytest.mark.asyncio
async def test_import_task_imports_sessions(runtime: AgentRuntime, tmp_path: Path) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "sessions").mkdir(parents=True)
    session = {
        "id": "sess1",
        "type": "main",
        "channelType": "webchat",
        "peerId": "peer1",
        "createdAtMs": 1772603285403,
        "lastActiveMs": 1772603307653,
        "contextHistory": [
            {"role": "user", "content": "Hello", "timestampMs": 1772603307160},
            {"role": "assistant", "content": "Hi there", "timestampMs": 1772603307653},
        ],
        "metadata": {},
    }
    (bulma_dir / "sessions" / "sess1.json").write_text(json.dumps(session))

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    assert report.sessions_imported == 1
    assert report.session_messages_imported == 2
    messages = await runtime.ctx.context_store.list_recent("sess1", limit=10)
    assert len(messages) == 2
    assert messages[1].content == "Hi there"
    assert messages[0].created_at.timestamp() == pytest.approx(1772603307.160)


@pytest.mark.asyncio
async def test_import_task_imports_memory_aux(runtime: AgentRuntime, tmp_path: Path) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "memory").mkdir(parents=True)

    emotion_entry = {
        "episodeId": "ep1",
        "timestampMs": 1772629002346,
        "previous": {"primaryEmotion": "fear"},
        "next": {"primaryEmotion": "calm"},
        "reason": "nightly_consolidation_reappraisal",
        "consolidationRunId": "run1",
    }
    with (bulma_dir / "memory" / "emotion_history.jsonl").open("w") as fh:
        fh.write(json.dumps(emotion_entry) + "\n")

    report_entry = {
        "runId": "run1",
        "timestampMs": 1772629000000,
        "clustersMerged": 3,
        "clustersRejected": 1,
        "summary": "Merged weak clusters",
    }
    with (bulma_dir / "memory" / "consolidation_reports.jsonl").open("w") as fh:
        fh.write(json.dumps(report_entry) + "\n")

    goal_thread = {
        "threadId": "gt1",
        "timestampMs": 1772629000000,
        "goalId": "g1",
        "goalLabel": "Finish import",
        "status": "active",
        "episodeIds": ["ep1"],
    }
    with (bulma_dir / "memory" / "goal_threads.jsonl").open("w") as fh:
        fh.write(json.dumps(goal_thread) + "\n")

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    assert report.emotion_history_imported == 1
    assert report.consolidation_reports_imported == 1
    assert report.goal_threads_imported == 1


@pytest.mark.asyncio
async def test_import_task_preserves_continuity_as_first_class_state(
    runtime: AgentRuntime, tmp_path: Path
) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "continuity").mkdir(parents=True)
    temporal = {"lastSessionBridgeBySessionId": {"s1": {"summary": "bridge"}}}
    integrity = {"status": "ok", "summary": ["continuity intact"], "counts": {"files": 2}}
    (bulma_dir / "continuity" / "temporal-state.json").write_text(json.dumps(temporal))
    (bulma_dir / "continuity" / "integrity-report.json").write_text(json.dumps(integrity))

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    await task.run()

    continuity = runtime.ctx.identity.continuity
    assert continuity.source_system == "openbulma-v4"
    assert continuity.temporal_bridges["lastSessionBridgeBySessionId"]["s1"]["summary"] == "bridge"
    assert continuity.integrity_report["status"] == "ok"


@pytest.mark.asyncio
async def test_import_task_archives_and_indexes_executive_events(
    runtime: AgentRuntime, tmp_path: Path
) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "executive").mkdir(parents=True)
    events = [
        {
            "ts": "2026-03-13T03:04:38.323Z",
            "type": "goal_thread_sync",
            "details": {
                "entity": "goal",
                "status": "active",
                "label": "build continuity view",
                "goalThreadId": "thread-1",
            },
        },
        {
            "ts": "2026-03-13T03:05:38.323Z",
            "type": "task_outcome",
            "details": {
                "entity": "task",
                "status": "completed",
                "label": "finish note",
                "taskId": "task-1",
            },
        },
    ]
    with (bulma_dir / "executive" / "events.jsonl").open("w") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    assert report.executive_events_imported == 2
    assert report.executive_events_archived is True
    assert report.executive_event_archive_path is not None
    assert Path(report.executive_event_archive_path).exists()
    assert report.executive_event_index_path is not None
    assert Path(report.executive_event_index_path).exists()

    from opencas.legacy.executive_event_index import search_executive_events

    hits = search_executive_events(runtime.ctx.config.state_dir, event_type="task_outcome")
    assert len(hits) == 1
    assert hits[0]["task_id"] == "task-1"


@pytest.mark.asyncio
async def test_import_task_separates_duplicate_timestamp_episodes_and_edges(
    runtime: AgentRuntime, tmp_path: Path
) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "memory").mkdir(parents=True)
    episodes = [
        {
            "id": "ep-a",
            "timestampMs": 1771485080000,
            "source": "chat",
            "textContent": "first",
            "salience": 0.5,
            "identityCore": False,
        },
        {
            "id": "ep-b",
            "timestampMs": 1771485080000,
            "source": "chat",
            "textContent": "second",
            "salience": 0.6,
            "identityCore": False,
        },
    ]
    with (bulma_dir / "memory" / "episodes.jsonl").open("w") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep) + "\n")
    with (bulma_dir / "memory" / "edges.jsonl").open("w") as fh:
        fh.write(
            json.dumps(
                {
                    "sourceId": "ep-a",
                    "targetId": "ep-b",
                    "semanticWeight": 0.9,
                    "confidence": 0.9,
                }
            )
            + "\n"
        )

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    first = await runtime.memory.get_episode(str(bulma_episode_uuid("ep-a")))
    second = await runtime.memory.get_episode(str(bulma_episode_uuid("ep-b")))
    assert report.episodes_imported == 2
    assert first is not None
    assert second is not None
    assert first.content == "first"
    assert second.content == "second"
    edges = await runtime.memory.get_edges_for(str(bulma_episode_uuid("ep-a")))
    assert any(edge.target_id == str(bulma_episode_uuid("ep-b")) for edge in edges)


@pytest.mark.asyncio
async def test_import_task_dedupes_duplicate_bulma_episode_ids(
    runtime: AgentRuntime, tmp_path: Path
) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "memory").mkdir(parents=True)
    records = [
        {
            "id": "dup-id",
            "timestampMs": 1771485080000,
            "source": "chat",
            "textContent": "duplicate recovery memory",
            "salience": 0.5,
        },
        {
            "id": "dup-id",
            "timestampMs": 1771485080000,
            "source": "chat",
            "textContent": "duplicate recovery memory",
            "salience": 0.6,
        },
    ]
    with (bulma_dir / "memory" / "episodes.jsonl").open("w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    assert report.episodes_imported == 1
    assert report.duplicate_episodes_skipped == 1
    imported = await runtime.memory.get_episode(str(bulma_episode_uuid("dup-id")))
    assert imported is not None
    assert imported.content == "duplicate recovery memory"


@pytest.mark.asyncio
async def test_import_task_imports_cutover_operational_state(
    runtime: AgentRuntime, tmp_path: Path
) -> None:
    bulma_dir = tmp_path / "bulma"
    (bulma_dir / "portfolio").mkdir(parents=True)
    (bulma_dir / "telemetry").mkdir(parents=True)
    (bulma_dir / "telegram").mkdir(parents=True)
    (bulma_dir / "somatic").mkdir(parents=True)
    (bulma_dir / "daydream").mkdir(parents=True)
    (bulma_dir / "self-knowledge").mkdir(parents=True)
    (bulma_dir / "legacy-workspace").mkdir(parents=True)

    (bulma_dir / "portfolio" / "clusters.json").write_text(
        json.dumps(
            [
                {
                    "id": "portfolio:550e8400-e29b-41d4-a716-446655440000",
                    "fascinationKey": "lighthouse-memory",
                    "tags": ["lighthouse"],
                    "sparkCount": 2,
                    "initiativeCount": 1,
                    "artifactCount": 1,
                    "firstSeenAtMs": 1772603307160,
                    "lastTouchedAtMs": 1772603307653,
                    "summary": "Bulma kept returning to the lighthouse.",
                }
            ]
        )
    )
    with (bulma_dir / "telemetry" / "token-events.jsonl").open("w") as fh:
        fh.write(
            json.dumps(
                {
                    "ts": 1772259755955,
                    "provider": "kimi",
                    "model": "kimi-for-coding",
                    "promptTokens": 1,
                    "completionTokens": 2,
                    "totalTokens": 3,
                    "latencyMs": 4,
                    "source": "chat",
                }
            )
            + "\n"
        )
    (bulma_dir / "telegram" / "bot-config.json").write_text(
        json.dumps({"enabled": True, "botToken": "", "allowFrom": ["123"], "dmPolicy": "pairing"})
    )
    with (bulma_dir / "telegram" / "pairing-claims.jsonl").open("w") as fh:
        fh.write(json.dumps({"code": "BULMA", "senderId": "123", "claimedAtMs": 1772603307160}) + "\n")
    (bulma_dir / "somatic" / "musubi.json").write_text(
        json.dumps({"lastRelationalContactAtMs": 1772603307160, "microGainWindowAccumulated": 0.1})
    )
    (bulma_dir / "daydream" / "current_focus.json").write_text(json.dumps({"focus": "lighthouse"}))
    (bulma_dir / "self-knowledge" / "index.json").write_text(json.dumps({"name": "Bulma"}))
    (bulma_dir / "legacy-workspace" / "keeper.md").write_text("lighthouse", encoding="utf-8")

    task = BulmaImportTask(bulma_dir, runtime=runtime)
    report = await task.run()

    assert report.portfolio_clusters_imported == 1
    assert report.token_telemetry_imported == 1
    assert report.telegram_state_imported is True
    assert report.self_knowledge_imported == 1
    assert report.curated_workspace_files_imported == 1
    assert report.cutover_manifest_path is not None
    assert Path(report.cutover_manifest_path).exists()
    assert (runtime.ctx.config.state_dir / "legacy-workspace" / "keeper.md").read_text() == "lighthouse"
    assert (runtime.ctx.config.state_dir / "telegram" / "pairings.json").exists()
    cluster = await runtime.ctx.portfolio_store.get_by_key("lighthouse-memory")
    assert cluster is not None
    assert cluster.meta["bulma_raw"]["summary"].startswith("Bulma kept")
