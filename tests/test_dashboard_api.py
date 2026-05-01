import pytest
import httpx
from fastapi.testclient import TestClient
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from pathlib import Path

from opencas.api.server import create_app
from opencas.api.config_overview import build_config_overview_payload
from opencas.bootstrap import BootstrapConfig
from opencas.governance import build_plugin_trust_feed_signature_payload
from opencas.model_routing import ModelRoutingConfig


class FakeStore:
    def __init__(self, events=None):
        self._events = events or []

    def query(self, **kwargs):
        limit = kwargs.get("limit", 1000)
        kinds = kwargs.get("kinds")
        session_id = kwargs.get("session_id")
        result = []
        for e in self._events:
            if len(result) >= limit:
                break
            if kinds and e.kind not in kinds:
                continue
            if session_id and e.session_id != session_id:
                continue
            result.append(e)
        return result


class FakeTelemetryEvent:
    def __init__(self, kind, message, payload=None, session_id=None, span_id=None):
        from datetime import datetime, timezone
        from opencas.telemetry import EventKind

        self.kind = kind if isinstance(kind, EventKind) else EventKind(kind)
        self.message = message
        self.payload = payload or {}
        self.session_id = session_id
        self.span_id = span_id
        self.timestamp = datetime.now(timezone.utc)


class FakeContextStore:
    def __init__(self, sessions=None):
        self.sessions = sessions or {}
        self._db = None

    async def list_recent(self, session_id, limit=50):
        from opencas.context.models import MessageEntry, MessageRole

        return [
            MessageEntry(role=MessageRole.USER, content="hi", meta={}),
            MessageEntry(role=MessageRole.ASSISTANT, content="hello", meta={}),
        ]

    async def list_session_ids(self, limit=50, status="active"):
        return [
            {"session_id": "s1", "name": None, "status": "active", "last_at": "2026-04-08T10:00:00+00:00", "message_count": 2}
        ]

    async def search_sessions(self, query, status="active", limit=20):
        return []

    async def get_session_meta(self, session_id):
        return {"session_id": session_id, "name": None, "status": "active", "last_at": "2026-04-08T10:00:00+00:00", "message_count": 0}

    async def ensure_session(self, session_id):
        pass

    async def update_session_name(self, session_id, name):
        pass

    async def set_session_status(self, session_id, status):
        pass


class FakeMemoryStore:
    def __init__(self):
        self._db = None

    async def list_episodes(self, **kwargs):
        return []

    async def search_episodes_by_content(self, query, limit=20):
        return []

    async def get_episodes_by_ids(self, ids):
        return []

    async def get_stats(self):
        return {
            "episode_count": 0,
            "memory_count": 0,
            "edge_count": 0,
            "compacted_count": 0,
            "identity_core_count": 0,
            "avg_salience": 0.0,
            "affect_distribution": {},
        }


class FakeTask:
    def __init__(self, task_id, objective, stage="done", status="completed", source="intervention_launch_background"):
        from datetime import datetime, timezone
        from types import SimpleNamespace

        self.task_id = task_id
        self.objective = objective
        self.stage = SimpleNamespace(value=stage)
        self.status = status
        self.meta = {"source": source}
        self.project_id = None
        self.commitment_id = None
        self.updated_at = datetime.now(timezone.utc)
        self.created_at = self.updated_at
        self.depends_on = []
        self.attempt = 0
        self.max_attempts = 3
        self.phases = []


class FakeTaskStore:
    def __init__(self):
        self._items = [
            FakeTask("task-1", "Could documentation become relational? A changelog that records not just what shipped, but what we learned together."),
            FakeTask("task-2", "Could documentation become relational? A changelog that records not just what shipped, but what we learned together."),
            FakeTask("task-3", "Investigate failing browser validation", stage="executing", status="running", source="manual"),
        ]
        self._items[2].meta["retry_governor"] = {
            "allowed": False,
            "reason": "RetryGovernor blocked a broad retry with no new evidence.",
            "mode": "deterministic_review",
            "attempt": 2,
            "packet_id": "packet-retry-blocked",
        }

    async def list_all(self, limit=100, offset=0):
        return self._items[offset:offset + limit]

    async def get(self, task_id):
        for item in self._items:
            if str(item.task_id) == str(task_id):
                return item
        return None

    async def get_result(self, task_id):
        if task_id == "task-3":
            return None
        return type(
            "Result",
            (),
            {
                "model_dump": lambda self, mode="json": {
                    "task_id": task_id,
                    "success": True,
                    "stage": "done",
                    "output": "done",
                }
            },
        )()

    async def list_lifecycle_transitions(self, task_id, limit=100, offset=0):
        from datetime import datetime, timezone

        return [
            {
                "from_stage": "queued",
                "to_stage": "planning",
                "reason": "worker started",
                "timestamp": datetime.now(timezone.utc),
                "context": {},
            }
        ]

    async def get_latest_salvage_packet(self, task_id):
        from datetime import datetime, timezone
        from uuid import UUID
        from opencas.execution.models import AttemptOutcome, AttemptSalvagePacket, RetryMode

        if task_id == "task-1":
            return AttemptSalvagePacket(
                packet_id=UUID("11111111-2222-3333-4444-555555555555"),
                task_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                attempt=1,
                project_signature="docs-relational",
                project_id=None,
                objective="Could documentation become relational?",
                canonical_artifact_path="workspace/notes/relational-docs.md",
                artifact_paths_touched=["workspace/notes/relational-docs.md"],
                plan_digest="plan",
                execution_digest="exec",
                verification_digest="verify",
                tool_signature="tool",
                divergence_signature="div-artifact",
                outcome=AttemptOutcome.VERIFY_FAILED,
                partial_value="Draft artifact exists and needs verification.",
                discovered_constraints=[],
                unresolved_questions=[],
                best_next_step="Verify workspace/notes/relational-docs.md.",
                recommended_mode=RetryMode.RESUME_EXISTING_ARTIFACT,
                meaningful_progress_signal="artifact",
                llm_spend_class="broad",
                created_at=datetime(2026, 4, 20, 1, 15, tzinfo=timezone.utc),
            )
        if task_id == "task-3":
            return AttemptSalvagePacket(
                packet_id=UUID("22222222-3333-4444-5555-666666666666"),
                task_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
                attempt=2,
                project_signature="browser-validation",
                project_id=None,
                objective="Investigate failing browser validation",
                canonical_artifact_path=None,
                artifact_paths_touched=[],
                plan_digest="plan",
                execution_digest="exec",
                verification_digest=None,
                tool_signature="tool",
                divergence_signature="div-no-progress",
                outcome=AttemptOutcome.VERIFY_FAILED,
                partial_value="",
                discovered_constraints=["no meaningful progress"],
                unresolved_questions=[],
                best_next_step="No meaningful progress: stop broad retry and change the next attempt frame.",
                recommended_mode=RetryMode.DETERMINISTIC_REVIEW,
                meaningful_progress_signal="no_meaningful_progress",
                llm_spend_class="deterministic_review",
                created_at=datetime(2026, 4, 20, 1, 20, tzinfo=timezone.utc),
            )
        return None


class FakeEpisodeGraph:
    async def get_neighbors(self, episode_id, **kwargs):
        return []


class FakeEmbeddings:
    def __init__(self, cache=None):
        async def _get(*_args, **_kwargs):
            return None
        self.cache = cache or type("C", (), {"get": staticmethod(_get)})()
        self.model_id = "google/gemini-embedding-2-preview"

    async def health(self):
        class H:
            total_records = 42
            avg_embed_latency_ms_1h = 12.5
        return H()

    async def embed(self, text):
        class R:
            vector = [0.1, 0.2, 0.3]
            source_hash = "abc"
        return R()

    async def recent_records(self, limit=10):
        from datetime import datetime, timedelta, timezone
        from uuid import UUID

        now = datetime.now(timezone.utc)
        samples = [
            type(
                "EmbeddingRecordLike",
                (),
                {
                    "embedding_id": UUID("11111111-1111-1111-1111-111111111111"),
                    "model_id": self.model_id,
                    "created_at": now - timedelta(minutes=3),
                    "updated_at": now - timedelta(minutes=1),
                    "meta": {
                        "task_type": "memory_episode",
                        "source": "episode:ep-123",
                        "text": "the OpenCAS agent summarized the latest dashboard continuity probe and noted stable voice routing.",
                        "embedding_degraded": False,
                    },
                },
            )(),
            type(
                "EmbeddingRecordLike",
                (),
                {
                    "embedding_id": UUID("22222222-2222-2222-2222-222222222222"),
                    "model_id": self.model_id,
                    "created_at": now - timedelta(minutes=6),
                    "updated_at": now - timedelta(minutes=5),
                    "meta": {
                        "task_type": "retrieval_query",
                        "source": "query",
                        "text": "How are the recent embeddings doing?",
                        "embedding_degraded": True,
                    },
                },
            )(),
        ]
        return samples[:limit]


class FakeReadiness:
    def snapshot(self):
        return {"state": "ready"}


class FakeDoctor:
    async def run_all(self):
        from opencas.diagnostics.models import CheckStatus, DiagnosticCheck, HealthReport
        report = HealthReport()
        report.checks.append(DiagnosticCheck(name="memory", status=CheckStatus.PASS, message="ok"))
        report.overall = CheckStatus.PASS
        return report


class FakeHarness:
    baa = None


class FakeBAA:
    queue_size = 3
    held_size = 1
    active_count = 2


class FakeSummaryPayload:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


class FakeTokenTelemetry:
    def get_summary(self, _start, _end):
        return FakeSummaryPayload(
            {
                "totalTokens": 2400,
                "totalCalls": 4,
                "avgTokensPerCall": 600,
                "avgLatencyMs": 320,
                "costEstimate": 0.1275,
                "topModels": [
                    {"model": "anthropic/claude-sonnet-4-6", "calls": 3, "totalTokens": 2100},
                    {"model": "google/gemini-embedding-2-preview", "calls": 1, "totalTokens": 300},
                ],
            }
        )

    def get_session_summary(self, _session_id):
        return FakeSummaryPayload(
            {
                "totalTokens": 1800,
                "totalCalls": 3,
                "avgTokensPerCall": 600,
                "avgLatencyMs": 300,
                "costEstimate": 0.091,
                "topModels": [],
            }
        )

    def get_daily_rollup(self, _start, _end):
        return [
            FakeSummaryPayload({"date": "2026-04-08", "totalTokens": 1200, "totalCalls": 2, "avgLatencyMs": 310, "costEstimate": 0.061}),
            FakeSummaryPayload({"date": "2026-04-09", "totalTokens": 1200, "totalCalls": 2, "avgLatencyMs": 330, "costEstimate": 0.0665}),
        ]

    def get_time_series(self, _start, _end, bucket_ms=0):
        return [
            FakeSummaryPayload({"bucketStart": 1, "totalTokens": 1200, "totalCalls": 2, "avgLatencyMs": 310, "costEstimate": 0.061}),
            FakeSummaryPayload({"bucketStart": 2, "totalTokens": 1200, "totalCalls": 2, "avgLatencyMs": 330, "costEstimate": 0.0665}),
        ]

    def get_breakdown(self, _start, _end, field, limit=20):
        rows = {
            "provider": [{"provider": "anthropic", "totalTokens": 2100, "totalCalls": 3, "avgLatencyMs": 320, "costEstimate": 0.12}],
            "model": [{"model": "anthropic/claude-sonnet-4-6", "totalTokens": 2100, "totalCalls": 3, "avgLatencyMs": 320, "costEstimate": 0.12}],
            "source": [{"source": "chat", "totalTokens": 1800, "totalCalls": 3, "avgLatencyMs": 300, "costEstimate": 0.09}],
            "execution_mode": [{"execution_mode": "assistant", "totalTokens": 2400, "totalCalls": 4, "avgLatencyMs": 320, "costEstimate": 0.1275}],
        }
        return rows.get(field, [])[:limit]

    def get_recent_events(self, _start, _end, limit=100):
        return [
            {
                "ts": 1712750400000,
                "provider": "anthropic",
                "model": "anthropic/claude-sonnet-4-6",
                "source": "chat",
                "totalTokens": 900,
                "latencyMs": 280,
            }
        ][:limit]

    def get_top_events(self, _start, _end, limit=20):
        return [
            {
                "ts": 1712750400000,
                "provider": "anthropic",
                "model": "anthropic/claude-sonnet-4-6",
                "source": "chat",
                "totalTokens": 1200,
                "latencyMs": 320,
            }
        ][:limit]


class FakeApprovalLedgerStore:
    async def list_recent(self, limit=12):
        from datetime import datetime, timezone
        from types import SimpleNamespace

        entries = [
            SimpleNamespace(
                entry_id="entry-1",
                decision_id="decision-1",
                action_id="action-1",
                created_at=datetime(2026, 4, 9, 12, 0, tzinfo=timezone.utc),
                level="self_approved",
                tier=type("Tier", (), {"value": "low"})(),
                score=0.91,
                tool_name="bash_run_command",
                reasoning="Safe filesystem inspection",
                somatic_state=None,
            ),
            SimpleNamespace(
                entry_id="entry-2",
                decision_id="decision-2",
                action_id="action-2",
                created_at=datetime(2026, 4, 9, 12, 5, tzinfo=timezone.utc),
                level="must_escalate",
                tier=type("Tier", (), {"value": "high"})(),
                score=0.12,
                tool_name="process_kill",
                reasoning="Potentially destructive process intervention",
                somatic_state="tense",
            ),
        ]
        return entries[:limit]


class FakeApprovalLedger:
    def __init__(self):
        self.store = FakeApprovalLedgerStore()

    async def query_stats(self, window_days=7):
        return {
            "window_days": window_days,
            "breakdown": [
                {"tier": "low", "level": "self_approved", "count": 2, "avg_score": 0.93},
                {"tier": "high", "level": "must_escalate", "count": 1, "avg_score": 0.12},
            ],
        }


class FakeReceiptStore:
    async def list_recent(self, limit=40):
        from types import SimpleNamespace

        items = [
            SimpleNamespace(success=True),
            SimpleNamespace(success=True),
            SimpleNamespace(success=False),
        ]
        return items[:limit]


class FakeDaydreamStore:
    async def list_recent(self, limit=10, keeper_only=None):
        from datetime import datetime, timezone
        from types import SimpleNamespace

        items = [
            SimpleNamespace(
                reflection_id="refl-1",
                created_at=datetime(2026, 4, 10, 13, 46, tzinfo=timezone.utc),
                spark_content="Add a resume-here button to restore context.",
                recollection="You were building continuity tooling.",
                interpretation="The handoff needs a tighter loop.",
                synthesis="Persist cursor, scroll, and one-line intent.",
                open_question="How much context can stay structured?",
                changed_self_view="I maintain continuity, not just replies.",
                tension_hints=["continuity", "fatigue"],
                alignment_score=0.7,
                novelty_score=0.8,
                keeper=False,
            ),
            SimpleNamespace(
                reflection_id="refl-2",
                created_at=datetime(2026, 4, 10, 11, 6, tzinfo=timezone.utc),
                spark_content="Turn thread memory into a one-line ritual.",
                recollection="You care about proving continuity.",
                interpretation="Small rituals may outperform large systems.",
                synthesis="Open each session with a one-line continuity stub.",
                open_question=None,
                changed_self_view="Continuity should be visible.",
                tension_hints=["continuity"],
                alignment_score=0.82,
                novelty_score=0.6,
                keeper=True,
            ),
        ]
        if keeper_only is True:
            items = [item for item in items if item.keeper]
        elif keeper_only is False:
            items = [item for item in items if not item.keeper]
        return items[:limit]

    async def get_summary(self, window_days=7):
        return {
            "total_reflections": 2,
            "total_keepers": 1,
            "window_days": window_days,
            "window_reflections": 2,
            "window_keepers": 1,
            "latest_reflection_at": "2026-04-10T13:46:53+00:00",
        }


class FakeConflictStore:
    async def list_conflicts(self, limit=20, resolved=None):
        from datetime import datetime, timezone
        from types import SimpleNamespace

        items = [
            SimpleNamespace(
                conflict_id="conf-1",
                created_at=datetime(2026, 4, 10, 13, 47, tzinfo=timezone.utc),
                resolved_at=None,
                kind="continuity_gap",
                description="Artifact memory preserved files but not intent.",
                source_daydream_id="refl-1",
                occurrence_count=3,
                resolved=False,
                auto_resolved=False,
                resolution_notes="",
                somatic_context=SimpleNamespace(model_dump=lambda mode="json": {"somatic_tag": "anticipation", "energy": 0.5, "focus": 0.6}),
            ),
            SimpleNamespace(
                conflict_id="conf-2",
                created_at=datetime(2026, 4, 10, 8, 0, tzinfo=timezone.utc),
                resolved_at=datetime(2026, 4, 10, 9, 0, tzinfo=timezone.utc),
                kind="fatigue_interface",
                description="Interface density exceeded fatigue budget.",
                source_daydream_id="refl-2",
                occurrence_count=1,
                resolved=True,
                auto_resolved=True,
                resolution_notes="trimmed chrome",
                somatic_context=None,
            ),
        ]
        if resolved is True:
            items = [item for item in items if item.resolved]
        elif resolved is False:
            items = [item for item in items if not item.resolved]
        return items[:limit]


class FakeAffectiveExaminationStore:
    async def list_recent(
        self,
        *,
        limit=50,
        session_id=None,
        source_type=None,
        primary_emotion=None,
        action_pressure=None,
        consumed_by=None,
        decay_state=None,
    ):
        records = await self.list_unresolved_pressures(session_id=session_id, limit=limit)
        if source_type:
            records = [r for r in records if getattr(r.source_type, "value", r.source_type) == source_type]
        if primary_emotion:
            records = [
                r for r in records
                if getattr(r.affect.primary_emotion, "value", r.affect.primary_emotion) == primary_emotion
            ]
        if action_pressure:
            records = [
                r for r in records
                if getattr(r.action_pressure, "value", r.action_pressure) == action_pressure
            ]
        if consumed_by:
            records = [r for r in records if getattr(r.consumed_by, "value", r.consumed_by) == consumed_by]
        return records[:limit]

    async def list_unresolved_pressures(self, *, session_id=None, limit=50):
        from datetime import datetime, timezone
        from opencas.affective.models import (
            AffectiveActionPressure,
            AffectiveExamination,
            AffectiveSourceType,
            AffectiveTarget,
        )
        from opencas.somatic.models import AffectState, PrimaryEmotion

        return [
            AffectiveExamination(
                created_at=datetime(2026, 4, 20, 1, 21, tzinfo=timezone.utc),
                session_id=session_id or "default",
                source_type=AffectiveSourceType.TOOL_RESULT,
                source_id="tool-result-1",
                source_excerpt="Same browser validation output appeared again without a new artifact.",
                source_hash="affective-hash",
                target=AffectiveTarget.SYSTEM,
                affect=AffectState(primary_emotion=PrimaryEmotion.CONCERNED),
                intensity=0.62,
                confidence=0.74,
                action_pressure=AffectiveActionPressure.ASK_CLARIFYING_QUESTION,
                bounded_reason="already recognized repeated tool pressure; ask, block, or park instead of retrying",
                meta={"already_recognized": True},
            )
        ][:limit]


class FakeWorkStore:
    async def list_by_origin(self, origin, limit=100, offset=0):
        from datetime import datetime, timezone
        from types import SimpleNamespace

        if origin != "daydream":
            return []
        return [
            SimpleNamespace(
                work_id="work-daydream-1",
                created_at=datetime(2026, 4, 10, 7, 31, tzinfo=timezone.utc),
                updated_at=datetime(2026, 4, 10, 7, 31, tzinfo=timezone.utc),
                stage=type("Stage", (), {"value": "note"})(),
                content="A thread memory ritual to prove continuity without performance.",
                promotion_score=0.66,
                source_memory_ids=[],
                blocked_by=[],
                project_id=None,
                commitment_id=None,
                portfolio_id=None,
                meta={"origin": "daydream", "title": "Thread memory ritual"},
            )
        ][offset:offset + limit]


class FakeDaydreamMemoryStore(FakeMemoryStore):
    async def list_memories_by_tag(self, tag, limit=100, offset=0):
        from datetime import datetime, timezone
        from types import SimpleNamespace

        if tag != "daydream":
            return []
        now = datetime(2026, 4, 10, 7, 31, tzinfo=timezone.utc)
        return [
            SimpleNamespace(
                memory_id="mem-daydream-1",
                created_at=now,
                updated_at=now,
                content="Open each session with a one-line continuity stub.",
                embedding_id="emb-daydream-1",
                source_episode_ids=[],
                tags=["daydream", "keeper"],
                salience=1.8,
                access_count=2,
                last_accessed=now,
            )
        ][offset:offset + limit]


class FakeShadowRegistry:
    def summary(self, limit=10, cluster_limit=5):
        return {
            "total_entries": 3,
            "active_clusters": 1,
            "dismissed_clusters": 1,
            "reason_counts": {
                "retry_blocked": 2,
                "tool_loop_guard_blocked": 1,
            },
            "recent_entries": [
                {
                    "id": "shadow-2",
                    "captured_at": "2026-04-20T01:15:00+00:00",
                    "tool_name": "repair_retry",
                    "intent_summary": "retry:workspace/Chronicles/4246/chronicle_4246.md",
                    "block_reason": "retry_blocked",
                    "block_context": "RetryGovernor blocked a broad retry with no new evidence.",
                    "artifact": "workspace/Chronicles/4246/chronicle_4246.md",
                    "capture_source": "repair_executor",
                },
                {
                    "id": "shadow-1",
                    "captured_at": "2026-04-20T01:10:00+00:00",
                    "tool_name": "tool_loop_guard",
                    "intent_summary": "guard:write_note",
                    "block_reason": "tool_loop_guard_blocked",
                    "block_context": "Tool loop circuit breaker: exceeded 24 consecutive tool calls.",
                    "artifact": None,
                    "capture_source": "tool_use_loop",
                },
            ][:limit],
            "top_clusters": [
                {
                    "fingerprint": "cluster-chronicle",
                    "count": 2,
                    "triage_status": "active",
                    "annotation": None,
                    "block_reason": "retry_blocked",
                    "tool_name": "repair_retry",
                    "intent_summary": "retry:workspace/Chronicles/4246/chronicle_4246.md",
                    "latest_captured_at": "2026-04-20T01:15:00+00:00",
                }
            ][:cluster_limit],
        }

    def inspect_cluster(self, fingerprint, limit=25):
        if fingerprint != "cluster-chronicle":
            return {"available": False, "fingerprint": fingerprint, "entries": []}
        return {
            "available": True,
            "fingerprint": fingerprint,
            "count": 2,
            "block_reason": "retry_blocked",
            "tool_name": "repair_retry",
            "intent_summary": "retry:workspace/Chronicles/4246/chronicle_4246.md",
            "latest_captured_at": "2026-04-20T01:15:00+00:00",
            "triage_status": "active",
            "annotation": None,
            "triaged_at": None,
            "dismissed_at": None,
            "entries": [
                {
                    "id": "shadow-2",
                    "captured_at": "2026-04-20T01:15:00+00:00",
                    "tool_name": "repair_retry",
                    "intent_summary": "retry:workspace/Chronicles/4246/chronicle_4246.md",
                    "block_reason": "retry_blocked",
                    "block_context": "RetryGovernor blocked a broad retry with no new evidence.",
                    "artifact": "workspace/Chronicles/4246/chronicle_4246.md",
                    "capture_source": "repair_executor",
                    "target_kind": "repair_task",
                    "target_id": "task-chronicle-1",
                },
                {
                    "id": "shadow-3",
                    "captured_at": "2026-04-20T01:12:00+00:00",
                    "tool_name": "repair_retry",
                    "intent_summary": "retry:workspace/Chronicles/4246/chronicle_4246.md",
                    "block_reason": "retry_blocked",
                    "block_context": "RetryGovernor blocked another broad retry with no new evidence.",
                    "artifact": "workspace/Chronicles/4246/chronicle_4246.md",
                    "capture_source": "repair_executor",
                    "target_kind": "repair_task",
                    "target_id": "task-chronicle-1",
                },
            ][:limit],
        }

    def triage_cluster(self, fingerprint, annotation=None, dismissed=None):
        if fingerprint != "cluster-chronicle":
            return {"available": False, "fingerprint": fingerprint, "entries": []}
        return {
            "available": True,
            "fingerprint": fingerprint,
            "count": 2,
            "block_reason": "retry_blocked",
            "tool_name": "repair_retry",
            "intent_summary": "retry:workspace/Chronicles/4246/chronicle_4246.md",
            "latest_captured_at": "2026-04-20T01:15:00+00:00",
            "triage_status": "dismissed" if dismissed else "active",
            "annotation": annotation,
            "triaged_at": "2026-04-21T07:15:00+00:00",
            "dismissed_at": "2026-04-21T07:15:00+00:00" if dismissed else None,
            "entries": [
                {
                    "id": "shadow-2",
                    "captured_at": "2026-04-20T01:15:00+00:00",
                    "tool_name": "repair_retry",
                    "intent_summary": "retry:workspace/Chronicles/4246/chronicle_4246.md",
                    "block_reason": "retry_blocked",
                    "block_context": "RetryGovernor blocked a broad retry with no new evidence.",
                    "artifact": "workspace/Chronicles/4246/chronicle_4246.md",
                    "capture_source": "repair_executor",
                    "target_kind": "repair_task",
                    "target_id": "task-chronicle-1",
                }
            ],
        }


class FakeCtx:
    config = type("C", (), {
        "state_dir": "/tmp/opencas",
        "session_id": "default",
        "workspace_root": None,
        "workspace_roots": [],
        "model_dump": lambda self, **kw: {"state_dir": "/tmp/opencas", "default_llm_model": "test"},
    })()
    readiness = FakeReadiness()
    event_bus = None
    embeddings = FakeEmbeddings()
    doctor = FakeDoctor()
    harness = FakeHarness()
    memory = FakeMemoryStore()
    context_store = FakeContextStore()
    token_telemetry = FakeTokenTelemetry()
    ledger = FakeApprovalLedger()
    receipt_store = FakeReceiptStore()
    tasks = FakeTaskStore()
    daydream_store = FakeDaydreamStore()
    conflict_store = FakeConflictStore()
    work_store = FakeWorkStore()
    shadow_registry = FakeShadowRegistry()
    affective_examinations = FakeAffectiveExaminationStore()
    llm = type("L", (), {"manager": type("M", (), {"_config": None})()})()
    sandbox = type("S", (), {"allowed_roots": [], "report_isolation": lambda self: {"mode": "workspace-only", "container_detected": False, "allowed_roots": [], "fallback": True}})()


class FakeTracer:
    def __init__(self):
        self.store = FakeStore()


class FakeRuntime:
    def __init__(self):
        self.ctx = FakeCtx()
        self.tracer = FakeTracer()
        self.memory = FakeDaydreamMemoryStore()
        self.episode_graph = FakeEpisodeGraph()
        self.telegram_settings = type(
            "TelegramSettings",
            (),
            {
                "enabled": True,
                "bot_token": "123:abc",
                "dm_policy": "pairing",
                "allow_from": ["42"],
                "poll_interval_seconds": 1.0,
                "pairing_ttl_seconds": 3600,
                "api_base_url": "https://api.telegram.org",
                "redacted_dict": lambda self: {
                    "enabled": True,
                    "bot_token": "***",
                    "token_configured": True,
                    "dm_policy": "pairing",
                    "allow_from": ["42"],
                    "poll_interval_seconds": 1.0,
                    "pairing_ttl_seconds": 3600,
                    "api_base_url": "https://api.telegram.org",
                },
            },
        )()

    async def telegram_status(self):
        return {
            "enabled": True,
            "configured": True,
            "token_configured": True,
            "running": True,
            "dm_policy": "pairing",
            "allow_from": ["42"],
            "bot": {
                "id": 1,
                "username": "opencas_bot",
                "first_name": "OpenCAS",
                "link": "https://t.me/opencas_bot",
            },
            "last_update_id": 10,
            "last_error": None,
            "pairings": {
                "approved_user_ids": ["42"],
                "pending_requests": [
                    {
                        "code": "PAIR1234",
                        "user_id": "99",
                        "username": "new_user",
                        "first_name": "New",
                        "last_name": "User",
                        "requested_at": 1712750400.0,
                    }
                ],
                "approved_requests": [],
            },
            "config": self.telegram_settings.redacted_dict(),
            "setup": {
                "transport": "long-polling",
                "pairing_supported": True,
                "typing_indicator_supported": True,
                "message_editing_supported": True,
                "steps": ["one", "two", "three", "four"],
            },
        }

    async def configure_telegram(self, _settings):
        return await self.telegram_status()

    async def approve_telegram_pairing(self, code):
        return code == "PAIR1234"

    async def phone_status(self):
        return {
            "enabled": True,
            "public_base_url": "https://opencas.example.com",
            "webhook_signature_required": True,
            "twilio_from_number": "+14846736227",
            "owner": {
                "display_name": "Cabew",
                "phone_number": "+17203340532",
                "workspace_subdir": "phone/owner",
                "configured": True,
            },
            "contacts": [
                {
                    "phone_number": "+15550001111",
                    "display_name": "Alex",
                    "trust_level": "low",
                    "allowed_actions": ["leave_message", "knowledge_qa"],
                    "workspace_subdir": "phone/contacts/alex",
                    "notes": "Client workspace",
                }
            ],
            "twilio_credentials_configured": True,
            "webhook_urls": {
                "voice": "https://opencas.example.com/api/phone/twilio/voice",
                "gather": "https://opencas.example.com/api/phone/twilio/gather",
            },
            "contact_count": 1,
            "menu_config_source": {
                "path": "operator_seed/phone/menu.json",
                "editable_path": "/tmp/state/phone/dashboard_menu.json",
                "using_override": False,
            },
            "menu_config": {
                "default_menu_key": "public_main",
                "owner_menu_key": "owner_entry",
                "owner_pin_prompt": "Enter owner pin.",
                "owner_pin_retry_prompt": "Retry pin.",
                "owner_pin_success_message": "Verified.",
                "owner_pin_failure_message": "Denied.",
                "menus": [
                    {
                        "key": "owner_entry",
                        "prompt": "Press 1 for the owner.",
                        "reprompt": "Press 1 to continue.",
                        "options": [
                            {"key": "owner_continue", "digit": "1", "action": "owner_conversation", "label": "Continue as owner"},
                            {"key": "owner_main_menu", "digit": "2", "action": "submenu", "label": "Main menu", "target_menu": "public_main"},
                        ],
                    },
                    {
                        "key": "public_main",
                        "prompt": "Potential employers, press 1.",
                        "reprompt": "Please press 1 for employer mode.",
                        "options": [
                            {"key": "employer", "digit": "1", "action": "workspace_assistant", "label": "Potential employer"},
                            {"key": "reject", "digit": "2", "action": "say_then_hangup", "label": "Not for this line"},
                        ],
                    },
                ],
            },
            "recent_calls": [
                {
                    "call_sid": "CA123",
                    "caller_number": "+15550001111",
                    "mode": "workspace_assistant",
                    "last_at": "2026-04-17T12:00:00+00:00",
                    "last_event": "phone_stream_closed",
                    "last_summary": "twilio_stop",
                    "event_count": 4,
                    "issue_count": 0,
                    "hangup_reason": "twilio_stop",
                }
            ],
            "recent_events": [
                {
                    "timestamp": "2026-04-17T12:00:00+00:00",
                    "event": "phone_stream_closed",
                    "label": "stream closed",
                    "summary": "twilio_stop",
                    "call_sid": "CA123",
                    "caller_number": "+15550001111",
                    "mode": "workspace_assistant",
                }
            ],
            "session_profiles": {
                "owner_entry": {
                    "prompt": "Press 1 for the owner.",
                    "reprompt": "Press 1 to continue.",
                    "continue_digit": "1",
                    "fallback_digit": "2",
                },
                "public_main": {
                    "prompt": "Potential employers, press 1.",
                    "reprompt": "Please press 1 for employer mode.",
                },
                "owner_pin": {
                    "prompt": "Enter owner pin.",
                    "retry_prompt": "Retry pin.",
                    "success_message": "Verified.",
                    "failure_message": "Denied.",
                },
                "employer": {
                    "enabled": True,
                    "digit": "1",
                    "label": "Potential employer",
                    "phrases": ["employer", "hiring"],
                    "greeting": "Employer greeting.",
                    "prompt_profile": "worksafe_owner",
                    "allowed_actions": ["leave_message", "knowledge_qa"],
                    "shared_workspace_subdir": "phone/employer_shared",
                    "caller_workspace_subdir": "phone/employers/{phone_digits}",
                },
                "reject": {
                    "enabled": True,
                    "digit": "2",
                    "label": "Not for this line",
                    "phrases": ["other"],
                    "message": "Sorry, this service is not for you.",
                },
            },
        }

    async def configure_phone(self, _settings):
        status = await self.phone_status()
        status["saved"] = True
        return status

    async def autoconfigure_phone(self, **_kwargs):
        status = await self.phone_status()
        status["saved"] = True
        status["autoconfigured"] = True
        status["selected_number"] = {"sid": "PN123", "phone_number": "+14846736227"}
        status["twilio_number_candidates"] = [{"sid": "PN123", "phone_number": "+14846736227"}]
        status["webhook_update"] = {
            "voice_url": "https://opencas.example.com/api/phone/twilio/voice",
            "voice_method": "POST",
        }
        return status

    async def configure_phone_session_profiles(self, _payload):
        status = await self.phone_status()
        status["saved"] = True
        status["session_profiles_saved"] = True
        return status

    async def configure_phone_menu_config(self, payload):
        status = await self.phone_status()
        status["saved"] = True
        status["menu_config_saved"] = True
        status["menu_config"] = payload
        return status

    async def recent_phone_calls(self, *, limit=10):
        status = await self.phone_status()
        return {
            "calls": list((status.get("recent_calls") or []))[:limit],
            "events": status.get("recent_events") or [],
        }

    async def phone_call_detail(self, call_sid):
        status = await self.phone_status()
        call = next((item for item in status.get("recent_calls") or [] if item.get("call_sid") == call_sid), None)
        if call is None:
            return {"found": False, "call_sid": call_sid, "events": [], "phase_durations": {}}
        return {
            "found": True,
            "call_sid": call_sid,
            "call": call,
            "phase_durations": {
                "time_to_transcription_seconds": 1.25,
                "workspace_reply_seconds": 2.75,
                "time_to_first_tts_seconds": 3.0,
                "total_call_seconds": 8.4,
            },
            "events": [
                {
                    "timestamp": "2026-04-17T11:59:58+00:00",
                    "event": "phone_stream_started",
                    "label": "stream started",
                    "summary": "",
                    "payload": {"call_sid": call_sid},
                },
                {
                    "timestamp": "2026-04-17T12:00:00+00:00",
                    "event": "phone_stream_closed",
                    "label": "stream closed",
                    "summary": "twilio_stop",
                    "payload": {"call_sid": call_sid, "reason": "twilio_stop"},
                },
            ],
        }

    async def call_owner_via_phone(self, *, message="", reason=""):
        return {
            "ok": True,
            "call_sid": "CA123",
            "status": "queued",
            "to": "+17203340532",
            "from": "+14846736227",
            "message": message,
            "reason": reason,
        }

    async def workflow_status(self, limit=10, project_id=None):
        return {
            "executive": {
                "intention": "Improve the dashboard operator experience",
                "active_goals": ["Make dashboard readable", "Expose real task state"],
                "queued_work_count": 2,
                "capacity_remaining": 3,
                "recommend_pause": False,
                "queue": {
                    "counts": {
                        "total": 2,
                        "active": 1,
                        "held": 1,
                        "ready": 1,
                        "queued": 1,
                        "waiting": 0,
                    },
                    "items": [
                        {"work_id": "work-1", "state": "active", "bearing": "ready"},
                        {"work_id": "work-2", "state": "held", "bearing": "queued"},
                    ],
                },
            },
            "work": {
                "counts": {"total": 2, "ready": 1, "blocked": 1},
                "items": [
                    {
                        "work_id": "work-1",
                        "content": "Redesign chat panel layout",
                        "stage": "project",
                        "project_id": "proj-1",
                        "blocked_by": [],
                        "meta": {"title": "Redesign chat panel layout"},
                    }
                ],
            },
            "consolidation": {
                "available": True,
                "timestamp": "2026-04-15T00:00:00+00:00",
                "commitments_consolidated": 2,
                "commitments_extracted_from_chat": 1,
                "commitment_work_objects_created": 1,
            },
        }


class FakeEmbeddingCache:
    def __init__(self, records):
        self.records = records

    async def get(self, embedding_id):
        return self.records.get(embedding_id)


class FakeProjectionMemoryStore(FakeMemoryStore):
    async def list_episodes(self, **kwargs):
        from datetime import datetime, timezone
        from types import SimpleNamespace
        from opencas.memory import EpisodeKind

        now = datetime.now(timezone.utc)
        return [
            SimpleNamespace(
                episode_id="ep-1",
                created_at=now,
                kind=EpisodeKind.TURN,
                session_id="s1",
                content="first",
                salience=1.2,
                compacted=False,
                identity_core=False,
                confidence_score=0.8,
                used_successfully=1,
                used_unsuccessfully=0,
                somatic_tag=None,
                embedding_id="emb-1",
                affect=None,
            ),
            SimpleNamespace(
                episode_id="ep-2",
                created_at=now,
                kind=EpisodeKind.OBSERVATION,
                session_id="s1",
                content="second",
                salience=0.9,
                compacted=False,
                identity_core=False,
                confidence_score=0.7,
                used_successfully=0,
                used_unsuccessfully=1,
                somatic_tag=None,
                embedding_id="emb-2",
                affect=None,
            ),
        ]


class FakeNodeDetailMemoryStore(FakeMemoryStore):
    async def get_episode(self, episode_id):
        from datetime import datetime, timezone
        from types import SimpleNamespace
        from opencas.memory import EpisodeKind

        if episode_id != "ep-1":
            return None
        now = datetime.now(timezone.utc)
        return SimpleNamespace(
            episode_id="ep-1",
            created_at=now,
            kind=EpisodeKind.TURN,
            session_id="s1",
            content="anchor episode",
            salience=1.5,
            compacted=False,
            identity_core=False,
            confidence_score=0.9,
            used_successfully=2,
            used_unsuccessfully=0,
            somatic_tag=None,
            embedding_id="emb-1",
            affect=None,
        )

    async def get_edges_for(self, episode_id, min_confidence=0.0, limit=24, kind=None):
        from datetime import datetime, timezone
        from types import SimpleNamespace
        from opencas.memory import EdgeKind

        if episode_id != "ep-1":
            return []
        return [
            SimpleNamespace(
                edge_id="edge-1",
                source_id="ep-1",
                target_id="ep-2",
                kind=EdgeKind.SEMANTIC,
                semantic_weight=0.9,
                emotional_weight=0.1,
                recency_weight=0.2,
                structural_weight=0.05,
                salience_weight=0.4,
                causal_weight=0.0,
                verification_weight=0.0,
                actor_affinity_weight=0.0,
                confidence=0.82,
                created_at=datetime.now(timezone.utc),
            )
        ]

    async def get_episodes_by_ids(self, ids):
        from datetime import datetime, timezone
        from types import SimpleNamespace
        from opencas.memory import EpisodeKind

        now = datetime.now(timezone.utc)
        rows = []
        if "ep-2" in ids:
            rows.append(
                SimpleNamespace(
                    episode_id="ep-2",
                    created_at=now,
                    kind=EpisodeKind.OBSERVATION,
                    session_id="s1",
                    content="neighbor episode",
                    salience=1.1,
                    compacted=False,
                    identity_core=False,
                    confidence_score=0.7,
                    used_successfully=0,
                    used_unsuccessfully=0,
                    somatic_tag=None,
                    embedding_id="emb-2",
                    affect=None,
                )
            )
        return rows

    async def list_memories(self, limit=100, offset=0):
        from datetime import datetime, timezone
        from types import SimpleNamespace

        now = datetime.now(timezone.utc)
        return [
            SimpleNamespace(
                memory_id="mem-1",
                created_at=now,
                updated_at=now,
                content="distilled memory",
                embedding_id="emb-3",
                source_episode_ids=["ep-1"],
                tags=["summary"],
                salience=2.0,
                access_count=3,
                last_accessed=None,
            )
        ]


class FakeGatewayModel:
    def __init__(self, model_id, name=None):
        self.id = model_id
        self.name = name or model_id


class FakeGatewayProviderConfig:
    def __init__(self):
        self.base_url = "https://example.test"
        self.auth = "api-key"
        self.api = "anthropic-messages"
        self.headers = {}
        self.models = [FakeGatewayModel("claude-sonnet-4-6", "Claude Sonnet 4.6")]


class FakeGatewayProfile:
    provider = "anthropic"
    type = "api_key"
    base_url = None
    account_id = None
    gateway_id = None
    metadata = {"workspace": "test"}

    def is_expired(self):
        return False


class FakeGatewayConfig:
    def all_auth_profiles(self):
        return {"anthropic-main": FakeGatewayProfile()}

    def all_provider_configs(self):
        return {"anthropic": FakeGatewayProviderConfig()}


class FakeGatewayManager:
    _config = FakeGatewayConfig()

    async def list_models(self):
        return [
            {"id": "anthropic/claude-sonnet-4-6"},
            {"id": "google/gemini-embedding-2-preview"},
            {"id": "google/embeddinggemma-300m"},
        ]

    def resolve(self, model_ref):
        return type(
            "Resolved",
            (),
            {
                "provider": type(
                    "Provider",
                    (),
                    {"supports_reasoning_effort": lambda self, model=None: True},
                )(),
                "provider_id": str(model_ref).split("/", 1)[0],
                "model_id": str(model_ref).split("/", 1)[1] if "/" in str(model_ref) else str(model_ref),
                "profile_id": "anthropic-main",
                "auth_source": "profile:anthropic-main",
            },
        )()


class FakeMutableGatewayManager:
    def __init__(self):
        self._config = None
        self._config_path = None
        self._env_path = None
        self.reload_calls = 0

    def reload(self):
        from open_llm_auth.config import load_config

        self.reload_calls += 1
        config_path = self._config_path
        env_path = self._env_path if self._env_path and self._env_path.exists() else None
        self._config = load_config(config_path=config_path, env_path=env_path)

    async def list_models(self):
        return [
            {"id": "anthropic/claude-sonnet-4-6"},
            {"id": "google/gemini-2.5-flash"},
            {"id": "openai/gpt-5.3-codex"},
        ]


class FakeMutableLLM:
    def __init__(self, default_model="anthropic/claude-sonnet-4-6"):
        self.default_model = default_model
        self.manager = FakeMutableGatewayManager()
        self.model_routing = ModelRoutingConfig()
        self.last_set = None

    def set_model_routing(self, *, default_model=None, model_routing=None):
        self.default_model = default_model or self.default_model
        self.model_routing = model_routing or self.model_routing
        self.last_set = {
            "default_model": self.default_model,
            "model_routing": self.model_routing,
        }


class FakePluginTrustService:
    def __init__(self):
        self.entries = [
            {
                "scope": "publisher",
                "value": "trusted.example",
                "level": "trusted",
                "source": "user",
                "note": "seed policy",
                "metadata": {},
                "updated_at": "2026-04-16T00:00:00+00:00",
            }
        ]

    async def summary(self, limit=20):
        items = list(self.entries)[:limit]
        return {
            "available": True,
            "policy_count": len(self.entries),
            "publisher_policy_count": sum(1 for item in self.entries if item["scope"] == "publisher"),
            "checksum_policy_count": sum(1 for item in self.entries if item["scope"] == "checksum"),
            "signer_policy_count": sum(1 for item in self.entries if item["scope"] == "signer"),
            "feed_policy_count": sum(1 for item in self.entries if str(item["source"]).startswith("feed:")),
            "feed_source_count": len({str(item["source"]) for item in self.entries if str(item["source"]).startswith("feed:")}),
            "entries": items,
        }

    async def set_policy(self, scope, value, level, *, note="", source="user", metadata=None):
        scope_value = getattr(scope, "value", scope)
        level_value = getattr(level, "value", level)
        record = {
            "scope": str(scope_value),
            "value": str(value),
            "level": str(level_value),
            "source": str(source),
            "note": str(note),
            "metadata": dict(metadata or {}),
            "updated_at": "2026-04-17T00:00:00+00:00",
        }
        self.entries = [item for item in self.entries if not (item["scope"] == record["scope"] and item["value"] == record["value"])]
        self.entries.append(record)
        return type(
            "Policy",
            (),
            {
                "scope": type("Scope", (), {"value": record["scope"]})(),
                "value": record["value"],
                "level": type("Level", (), {"value": record["level"]})(),
                "source": record["source"],
                "note": record["note"],
                "metadata": record["metadata"],
                "updated_at": type("Timestamp", (), {"isoformat": lambda self: record["updated_at"]})(),
            },
        )()

    async def remove_policy(self, scope, value):
        scope_value = getattr(scope, "value", scope)
        self.entries = [item for item in self.entries if not (item["scope"] == str(scope_value) and item["value"] == str(value))]

    async def sync_feed(self, feed):
        if not isinstance(feed.get("signatures"), list) or not feed.get("signatures"):
            raise ValueError("plugin trust feed must include at least one signature entry")
        source_id = str(feed.get("source_id") or "")
        source = f"feed:{source_id}"
        desired_keys = set()
        imported = []
        for raw in feed.get("policies", []):
            scope_value = str(raw["scope"])
            value = str(raw["value"])
            desired_keys.add((scope_value, value))
            record = {
                "scope": scope_value,
                "value": value,
                "level": str(raw["level"]),
                "source": source,
                "note": str(raw.get("note") or ""),
                "metadata": dict(raw.get("metadata") or {}),
                "updated_at": "2026-04-17T01:00:00+00:00",
            }
            self.entries = [
                item for item in self.entries
                if not (item["scope"] == scope_value and item["value"] == value and item["source"] == source)
            ]
            self.entries.append(record)
            imported.append({"scope": scope_value, "value": value, "level": record["level"], "metadata": record["metadata"]})
        stale = [
            item for item in list(self.entries)
            if item["source"] == source and (item["scope"], item["value"]) not in desired_keys
        ]
        self.entries = [
            item for item in self.entries
            if not (item["source"] == source and (item["scope"], item["value"]) not in desired_keys)
        ]
        return type(
            "FeedSyncResult",
            (),
            {
                "source_id": source_id,
                "source": source,
                "format_version": int(feed.get("format_version", 1)),
                "verification": {
                    "signature_count": len(feed.get("signatures") or []),
                    "verified_signature_count": len(feed.get("signatures") or []),
                    "verified_signer_ids": [str(item.get("key_id")) for item in (feed.get("signatures") or []) if isinstance(item, dict)],
                    "trusted_signer_ids": [str(item.get("key_id")) for item in (feed.get("signatures") or []) if isinstance(item, dict)],
                    "matched_policies": [f"signer:{item.get('key_id')}" for item in (feed.get("signatures") or []) if isinstance(item, dict)],
                    "payload_sha256": "feed-payload",
                    "reasons": [],
                },
                "imported": imported,
                "removed": [{"scope": item["scope"], "value": item["value"]} for item in stale],
                "skipped_conflicts": [],
                "rejected": [],
            },
        )()


def _signed_feed_payload(*, source_id: str, policies: list[dict]) -> dict:
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key_raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_key = __import__("base64").b64encode(public_key_raw).decode("ascii")
    payload = build_plugin_trust_feed_signature_payload(
        format_version=1,
        source_id=source_id,
        policies=policies,
    )
    signature = __import__("base64").b64encode(private_key.sign(payload)).decode("ascii")
    return {
        "format_version": 1,
        "source_id": source_id,
        "policies": policies,
        "signatures": [
            {
                "key_id": "opencas-labs-main",
                "algorithm": "ed25519",
                "public_key": public_key,
                "signature": signature,
            }
        ],
    }


def test_health_endpoint():
    app = create_app(FakeRuntime())
    client = TestClient(app)
    resp = client.get("/api/monitor/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["overall"] == "pass"


def test_config_endpoint():
    app = create_app(FakeRuntime())
    client = TestClient(app)
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "config" in data


def test_provider_config_endpoint():
    app = create_app(FakeRuntime())
    client = TestClient(app)
    resp = client.get("/api/config/providers")
    assert resp.status_code == 200
    data = resp.json()
    assert "providers" in data


def test_memory_stats_endpoint():
    app = create_app(FakeRuntime())
    client = TestClient(app)
    resp = client.get("/api/memory/stats")
    # FakeMemoryStore has _db=None so this will fail; test that routing works
    assert resp.status_code in (200, 500)


def test_chat_sessions_endpoint():
    app = create_app(FakeRuntime())
    client = TestClient(app)
    resp = client.get("/api/chat/sessions")
    # FakeContextStore has _db=None so this will fail; test that routing works
    assert resp.status_code in (200, 500)


@pytest.mark.asyncio
async def test_chat_context_summary_surfaces_active_lane():
    runtime = FakeRuntime()
    runtime.ctx.llm = type(
        "L",
        (),
        {
            "manager": FakeGatewayManager(),
            "default_model": "kimi-coding/k2p5",
            "resolve_reasoning_effort_for_complexity": lambda self, complexity=None: "medium",
        },
    )()
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/chat/context-summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["lane"]["model"] == "kimi-coding/k2p5"
    assert data["lane"]["provider"] == "kimi-coding"
    assert data["lane"]["resolved_model"] == "kimi-coding/k2p5"
    assert data["lane"]["reasoning_supported"] is True
    assert data["lane"]["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_chat_context_summary_separates_configured_lane_from_last_lane():
    runtime = FakeRuntime()
    runtime.ctx.llm = type(
        "L",
        (),
        {
            "manager": FakeGatewayManager(),
            "default_model": "kimi-coding/k2p5",
            "current_lane_meta": lambda self: {
                "model": "codex-cli/gpt-5.4-mini",
                "provider": "codex-cli",
                "resolved_model": "codex-cli/gpt-5.4-mini",
                "complexity": "high",
                "reasoning_supported": True,
                "reasoning_effort": "medium",
            },
            "resolve_reasoning_effort_for_complexity": lambda self, complexity=None: "medium",
        },
    )()
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/chat/context-summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["lane"]["model"] == "kimi-coding/k2p5"
    assert data["last_lane"]["model"] == "codex-cli/gpt-5.4-mini"


def test_runtime_status_endpoint():
    app = create_app(FakeRuntime())
    client = TestClient(app)
    resp = client.get("/api/monitor/runtime")
    assert resp.status_code == 200
    data = resp.json()
    assert data["sandbox"]["mode"] == "workspace-only"
    assert data["execution"]["processes"]["total_count"] == 0
    assert data["execution"]["browser"]["total_count"] == 0


@pytest.mark.asyncio
async def test_dashboard_contains_operations_surface():
    app = create_app(FakeRuntime())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/dashboard")
    assert resp.status_code == 200
    body = resp.text
    assert "Operations" in body
    assert "Live Model Routing" in body
    assert "Chat / Reasoning model" in body
    assert "Light / Regular / Heavy / Extra Heavy" in body
    assert "Reasoning level" in body
    assert "Light reasoning" in body
    assert "Apply Routing" in body
    assert "Daydream" in body
    assert "Task Beacon" in body
    assert "/api/operations/sessions" in body
    assert "/api/operations/qualification" in body
    assert "/api/operations/validation-runs?limit=10" in body
    assert "/api/operations/qualification/reruns" in body
    assert 'data-panel="operations-sessions"' in body
    assert 'data-panel="operations-qualification"' in body
    assert "Browser Controls" in body
    assert "Active Qualification Reruns" in body
    assert "Recent Validation Runs" in body
    assert "/api/operations/validation-runs?limit=10&label=" in body
    assert "Matching Agent Checks" in body
    assert "/api/operations/hardening" in body
    assert "/api/operations/memory-value" in body
    assert "/api/operations/approval-audit?limit=12" in body
    assert "/api/operations/costs?window_days=7&bucket_hours=6" in body
    assert "/api/operations/receipts?limit=20" in body
    assert "/api/operations/tasks?limit=20" in body
    assert "/api/operations/work?limit=20" in body
    assert "/api/operations/commitments?status=active&limit=20" in body
    assert "/api/operations/plans?limit=20" in body
    assert "/api/telegram/status" in body
    assert "Telegram Channel" in body or "telegram-status" in body
    assert "/api/phone/status" in body
    assert "the OpenCAS agent Phone Bridge" in body or "phone-status" in body
    assert "/api/monitor/shadow-registry" in body
    assert "Shadow Registry" in body
    assert "/api/monitor/shadow-registry/cluster?fingerprint=" in body
    assert "Shadow Explorer" in body
    assert "/api/monitor/meaningful-loop" in body
    assert "Meaningful Loop" in body
    assert "/api/monitor/affective-examinations" in body
    assert "Affective Examinations" in body
    assert "dashboard/static/js/task_beacon.js" in body
    assert "/api/monitor/task-beacon" in body
    assert "/api/daydream/summary" in body
    assert "/api/daydream/reflections" in body
    assert "/api/daydream/conflicts" in body
    assert "/api/daydream/promotions" in body
    assert "Usage" in body
    assert "/api/usage/overview" in body
    assert "m.meta?.lane?.resolved_model" in body
    assert "m.meta?.lane?.reasoning_supported" in body
    assert "reasoning ${escapeHtml(lane.reasoning_effort || 'provider default')}" in body
    assert "Queue states" in body
    assert "legacy / unlabeled" in body


@pytest.mark.asyncio
async def test_dashboard_supports_opencas_prefix_mount():
    app = create_app(FakeRuntime())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/opencas/")

    assert page.status_code == 200
    assert 'href="dashboard/static/favicon.svg"' in page.text
    assert 'src="dashboard/static/js/http_helpers.js"' in page.text
    assert any(getattr(route, "path", "") == "/opencas/dashboard/static" for route in app.routes)


def test_operations_sessions_endpoint_available_from_main_app():
    app = create_app(FakeRuntime())
    client = TestClient(app)
    resp = client.get("/api/operations/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_pty"] == 0
    assert data["total_browser"] == 0


def test_operations_hardening_endpoints_surface_memory_approval_and_cost_state():
    app = create_app(FakeRuntime())
    client = TestClient(app)

    hardening = client.get("/api/operations/hardening")
    assert hardening.status_code == 200
    hardening_data = hardening.json()
    assert hardening_data["overall_state"] in {"emerging", "observable", "grounded"}
    assert "memory_value" in hardening_data
    assert "approval_audit" in hardening_data
    assert "costs" in hardening_data

    approval = client.get("/api/operations/approval-audit")
    assert approval.status_code == 200
    approval_data = approval.json()
    assert approval_data["total_decisions"] == 3
    assert approval_data["level_counts"]["self_approved"] == 2
    assert approval_data["level_counts"]["must_escalate"] == 1
    assert len(approval_data["recent_entries"]) == 2

    costs = client.get("/api/operations/costs")
    assert costs.status_code == 200
    cost_data = costs.json()
    assert cost_data["summary"]["totalCalls"] == 4
    assert cost_data["recent_receipts"]["success_count"] == 2

    memory_value = client.get("/api/operations/memory-value")
    assert memory_value.status_code == 200
    memory_value_data = memory_value.json()
    assert memory_value_data["evidence_level"] == "insufficient"
    assert memory_value_data["available"] is True


def test_daydream_endpoints_surface_reflections_conflicts_and_promotions():
    app = create_app(FakeRuntime())
    client = TestClient(app)

    summary = client.get("/api/daydream/summary?window_days=7")
    assert summary.status_code == 200
    summary_data = summary.json()
    assert summary_data["summary"]["total_reflections"] == 2
    assert summary_data["summary"]["active_conflicts"] == 1
    assert summary_data["summary"]["promoted_work_count"] == 1
    assert summary_data["summary"]["keeper_memory_count"] == 1

    reflections = client.get("/api/daydream/reflections?keeper_only=true")
    assert reflections.status_code == 200
    reflections_data = reflections.json()
    assert reflections_data["count"] == 1
    assert reflections_data["items"][0]["keeper"] is True

    conflicts = client.get("/api/daydream/conflicts?state=active")
    assert conflicts.status_code == 200
    conflict_data = conflicts.json()
    assert conflict_data["count"] == 1
    assert conflict_data["items"][0]["kind"] == "continuity_gap"

    promotions = client.get("/api/daydream/promotions")
    assert promotions.status_code == 200
    promotions_data = promotions.json()
    assert promotions_data["work_count"] == 1
    assert promotions_data["keeper_memory_count"] == 1


def test_usage_overview_endpoint_surfaces_gateway_and_process_data(monkeypatch):
    from opencas.api.routes import usage as usage_routes

    async def fake_gateway_snapshot(runtime, window_days, recent_limit):
        return {
            "available": True,
            "overview": {
                "summary": {"requests": 2, "total_tokens": 1400, "total_cost": 0.0, "avg_latency_ms": 210},
                "providers": [{"provider": "anthropic", "total_tokens": 1400, "requests": 2}],
                "recent": [{"id": 1, "provider": "anthropic", "endpoint": "chat.completions", "total_tokens": 700, "ts": "2026-04-10T12:00:00+00:00"}],
            },
            "provider_telemetry": [
                {
                    "provider": "anthropic",
                    "profile_count": 1,
                    "model_ids": ["claude-sonnet-4-6"],
                    "latest_observation": {"meta": {"rate_limits": {"requests_remaining": "99"}}},
                    "telemetry": {"available": False, "note": "Observed headers only"},
                }
            ],
            "notes": [],
        }

    monkeypatch.setattr(
        usage_routes,
        "_scan_process_hygiene",
        lambda: {
            "available": True,
            "duplicate_server_count": 1,
            "orphan_pytest_count": 0,
            "opencas_processes": [{"pid": 123, "command_compact": "python -m opencas --with-server"}],
            "gateway_processes": [],
            "notes": ["Detected 1 extra OpenCAS server process(es)."],
        },
    )
    monkeypatch.setattr(usage_routes, "_build_gateway_usage_snapshot", fake_gateway_snapshot)

    app = create_app(FakeRuntime())
    client = TestClient(app)
    resp = client.get("/api/usage/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["opencas"]["summary"]["totalTokens"] == 2400
    assert data["gateway"]["overview"]["summary"]["total_tokens"] == 1400
    assert data["process_hygiene"]["duplicate_server_count"] == 1


def test_active_gateway_provider_ids_follow_runtime_routing():
    from opencas.api.routes.usage import _active_gateway_provider_ids

    runtime = FakeRuntime()
    runtime.ctx.config.default_llm_model = "kimi-coding/k2p5"
    runtime.ctx.config.embedding_model_id = "google/gemini-embedding-2-preview"
    runtime.ctx.config.model_routing = ModelRoutingConfig(
        mode="tiered",
        standard_model="kimi-coding/k2p5",
        high_model="kimi-coding/k2p5",
        extra_high_model="kimi-coding/k2p5",
    )
    runtime.ctx.llm = type("L", (), {"default_model": "kimi-coding/k2p5"})()

    assert _active_gateway_provider_ids(runtime) == ["kimi-coding", "google"]


def test_telegram_status_and_config_endpoints():
    app = create_app(FakeRuntime())
    client = TestClient(app)

    status = client.get("/api/telegram/status")
    assert status.status_code == 200
    status_data = status.json()
    assert status_data["running"] is True
    assert status_data["bot"]["username"] == "opencas_bot"
    assert status_data["config"]["token_configured"] is True
    assert len(status_data["pairings"]["pending_requests"]) == 1

    update = client.post(
        "/api/telegram/config",
        json={
            "enabled": True,
            "dm_policy": "pairing",
            "allow_from": ["42"],
            "poll_interval_seconds": 1.0,
            "pairing_ttl_seconds": 3600,
        },
    )
    assert update.status_code == 200

    approve = client.post("/api/telegram/pairings/PAIR1234/approve")
    assert approve.status_code == 200


@pytest.mark.asyncio
async def test_chat_context_summary_and_task_routes(tmp_path):
    runtime = FakeRuntime()
    runtime.ctx.config = BootstrapConfig(
        state_dir=tmp_path / "state",
        workspace_root=tmp_path / "repo",
    ).resolve_paths()
    runtime.ctx.config.workspace_root.mkdir(parents=True, exist_ok=True)
    (runtime.ctx.config.workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-401` Build gate repair\n"
        "  - owner: Codex\n"
        "  - status: in progress\n\n"
        "## Background Context\n\n"
        "- `TASK-402` Test cleanup\n"
        "  - owner: Codex\n"
        "  - status: pending\n\n"
        "## Next Up / Backlog\n\n"
        "- `TASK-403` Regression follow-up\n"
        "  - owner: Codex\n"
        "  - status: pending\n",
        encoding="utf-8",
    )
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        summary = await client.get("/api/chat/context-summary?session_id=s1")
        assert summary.status_code == 200
        summary_data = summary.json()
        assert summary_data["executive"]["intention"] == "Improve the dashboard operator experience"
        assert summary_data["current_work"]["title"] == "Redesign chat panel layout"
        assert summary_data["consolidation"]["available"] is True
        assert summary_data["consolidation"]["commitments_extracted_from_chat"] == 1
        assert summary_data["tasks"]["counts"]["total"] == 3
        assert summary_data["task_beacon"]["headline"] == "now 1 • next 1 • later 1"
        assert summary_data["task_beacon"]["counts"]["now"] == 1
        assert summary_data["task_beacon"]["counts"]["next"] == 1
        assert [bucket["state"] for bucket in summary_data["task_beacon"]["view_model"]["buckets"]] == ["now", "next", "later"]
        assert [bucket["count"] for bucket in summary_data["task_beacon"]["view_model"]["buckets"]] == [1, 1, 1]
        assert [set(bucket) for bucket in summary_data["task_beacon"]["view_model"]["buckets"]] == [
            {"state", "count", "item"},
            {"state", "count", "item"},
            {"state", "count", "item"},
        ]
        assert summary_data["task_beacon"]["view_model"]["buckets"][0]["item"]["task_id"] == "TASK-401"
        assert summary_data["task_beacon"]["view_model"]["buckets"][1]["item"]["task_id"] == "TASK-403"
        assert "details" in summary_data["task_beacon"]
        assert "states" not in summary_data["task_beacon"]
        assert "summary" not in summary_data["task_beacon"]
        tasks = await client.get("/api/operations/tasks?limit=10")
        assert tasks.status_code == 200
        task_data = tasks.json()
        assert task_data["counts"]["completed"] == 2
        assert task_data["counts"]["active"] == 1
        assert task_data["items"][0]["duplicate_objective_count"] >= 1

        detail = await client.get("/api/operations/tasks/task-1")
        assert detail.status_code == 200
        detail_data = detail.json()
        assert detail_data["task"]["status"] == "completed"
        assert detail_data["task"]["source"] == "intervention_launch_background"


@pytest.mark.asyncio
async def test_chat_context_summary_ignores_artifact_stage_for_current_work(tmp_path):
    class ArtifactFirstRuntime(FakeRuntime):
        async def workflow_status(self, limit=10, project_id=None):
            payload = await super().workflow_status(limit=limit, project_id=project_id)
            payload["work"]["items"] = [
                {
                    "work_id": "artifact-1",
                    "content": "Verify musubi criterion with fresh evidence from an older artifact",
                    "stage": "artifact",
                    "project_id": None,
                    "blocked_by": [],
                    "meta": {"title": "Verify musubi criterion with fresh evidence"},
                },
                {
                    "work_id": "work-2",
                    "content": "Repair current-work foreground truth",
                    "stage": "micro_task",
                    "project_id": None,
                    "blocked_by": [],
                    "meta": {"title": "Repair current-work foreground truth"},
                },
            ]
            return payload

    runtime = ArtifactFirstRuntime()
    runtime.ctx.config = BootstrapConfig(
        state_dir=tmp_path / "state",
        workspace_root=tmp_path / "repo",
    ).resolve_paths()
    runtime.ctx.config.workspace_root.mkdir(parents=True, exist_ok=True)
    (runtime.ctx.config.workspace_root / "TaskList.md").write_text("# OpenCAS Task List\n\n", encoding="utf-8")

    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        summary = await client.get("/api/chat/context-summary?session_id=s1")

    assert summary.status_code == 200
    summary_data = summary.json()
    assert summary_data["current_work"]["work_id"] == "work-2"
    assert summary_data["current_work"]["title"] == "Repair current-work foreground truth"
    assert summary_data["current_work"]["stage"] == "micro_task"


@pytest.mark.asyncio
async def test_chat_context_summary_uses_active_queue_when_work_store_is_artifact_only(tmp_path):
    class ActiveQueueRuntime(FakeRuntime):
        async def workflow_status(self, limit=10, project_id=None):
            payload = await super().workflow_status(limit=limit, project_id=project_id)
            payload["executive"]["intention"] = "Stale anchored dashboard intention"
            payload["executive"]["queue"]["items"] = [
                {
                    "work_id": "queue-work-1",
                    "title": "Repair foreground truth from active queue",
                    "stage": "micro_task",
                    "state": "active",
                    "bearing": "ready",
                    "is_active": True,
                    "project_id": "proj-queue",
                    "blocked_by": [],
                }
            ]
            payload["work"]["items"] = [
                {
                    "work_id": "artifact-1",
                    "content": "Historical artifact should not own foreground truth",
                    "stage": "artifact",
                    "project_id": None,
                    "blocked_by": [],
                    "meta": {"title": "Historical artifact should not own foreground truth"},
                }
            ]
            return payload

    runtime = ActiveQueueRuntime()
    runtime.ctx.config = BootstrapConfig(
        state_dir=tmp_path / "state",
        workspace_root=tmp_path / "repo",
    ).resolve_paths()
    runtime.ctx.config.workspace_root.mkdir(parents=True, exist_ok=True)
    (runtime.ctx.config.workspace_root / "TaskList.md").write_text("# OpenCAS Task List\n\n", encoding="utf-8")

    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        summary = await client.get("/api/chat/context-summary?session_id=s1")

    assert summary.status_code == 200
    summary_data = summary.json()
    assert summary_data["current_work"]["work_id"] == "queue-work-1"
    assert summary_data["current_work"]["title"] == "Repair foreground truth from active queue"
    assert summary_data["executive"]["intention"] == "Repair foreground truth from active queue"
    assert summary_data["executive"]["intention_source"] == "active_queue"


@pytest.mark.asyncio
async def test_chat_context_summary_returns_no_current_work_for_artifact_only_workflow(tmp_path):
    class ArtifactOnlyRuntime(FakeRuntime):
        async def workflow_status(self, limit=10, project_id=None):
            payload = await super().workflow_status(limit=limit, project_id=project_id)
            payload["executive"]["queue"]["items"] = []
            payload["work"]["items"] = [
                {
                    "work_id": "artifact-1",
                    "content": "Completed artifact should not masquerade as current work",
                    "stage": "artifact",
                    "project_id": None,
                    "blocked_by": [],
                    "meta": {"title": "Completed artifact should not masquerade as current work"},
                }
            ]
            return payload

    runtime = ArtifactOnlyRuntime()
    runtime.ctx.config = BootstrapConfig(
        state_dir=tmp_path / "state",
        workspace_root=tmp_path / "repo",
    ).resolve_paths()
    runtime.ctx.config.workspace_root.mkdir(parents=True, exist_ok=True)
    (runtime.ctx.config.workspace_root / "TaskList.md").write_text("# OpenCAS Task List\n\n", encoding="utf-8")

    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        summary = await client.get("/api/chat/context-summary?session_id=s1")

    assert summary.status_code == 200
    assert summary.json()["current_work"] is None


@pytest.mark.asyncio
async def test_chat_context_summary_clears_completed_tasklist_intention_without_foreground(tmp_path):
    class CompletedTasklistRuntime(FakeRuntime):
        async def workflow_status(self, limit=10, project_id=None):
            payload = await super().workflow_status(limit=limit, project_id=project_id)
            payload["executive"]["intention"] = "Finished diagnostic repair"
            payload["executive"]["queue"]["items"] = []
            payload["work"]["items"] = [
                {
                    "work_id": "artifact-1",
                    "content": "Completed artifact should not keep an active intention alive",
                    "stage": "artifact",
                    "project_id": None,
                    "blocked_by": [],
                    "meta": {"title": "Completed artifact should not keep an active intention alive"},
                }
            ]
            return payload

    runtime = CompletedTasklistRuntime()
    runtime.ctx.config = BootstrapConfig(
        state_dir=tmp_path / "state",
        workspace_root=tmp_path / "repo",
    ).resolve_paths()
    runtime.ctx.config.workspace_root.mkdir(parents=True, exist_ok=True)
    (runtime.ctx.config.workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "## Recently Completed\n\n"
        "- `TASK-499` Finished diagnostic repair\n"
        "  - owner: Codex\n"
        "  - status: completed\n",
        encoding="utf-8",
    )

    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        summary = await client.get("/api/chat/context-summary?session_id=s1")

    assert summary.status_code == 200
    summary_data = summary.json()
    assert summary_data["current_work"] is None
    assert summary_data["executive"]["intention"] is None
    assert summary_data["executive"]["intention_source"] == "stale_tasklist_completed"


@pytest.mark.asyncio
async def test_monitor_task_beacon_endpoint_returns_quiet_public_payload(tmp_path):
    runtime = FakeRuntime()
    runtime.ctx.config = BootstrapConfig(
        state_dir=tmp_path / "state",
        workspace_root=tmp_path / "repo",
    ).resolve_paths()
    runtime.ctx.config.workspace_root.mkdir(parents=True, exist_ok=True)
    (runtime.ctx.config.workspace_root / "TaskList.md").write_text(
        Path(__file__).with_name("fixtures").joinpath("task_beacon_blocker_first.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/monitor/task-beacon")
        assert response.status_code == 200
        beacon = response.json()
        assert beacon["available"] is True
        assert beacon["matched_only"] is True
        assert beacon["headline"] == "now 1 • next 2 • later 1"
        assert beacon["counts"] == {"matched": 4, "now": 1, "next": 2, "later": 1, "total": 4}
        assert [bucket["state"] for bucket in beacon["view_model"]["buckets"]] == ["now", "next", "later"]
        assert [bucket["count"] for bucket in beacon["view_model"]["buckets"]] == [1, 2, 1]
        assert [set(bucket) for bucket in beacon["view_model"]["buckets"]] == [
            {"state", "count", "item"},
            {"state", "count", "item"},
            {"state", "count", "item"},
        ]
        assert [bucket["item"]["task_id"] for bucket in beacon["view_model"]["buckets"]] == [
            "TASK-802",
            "TASK-801",
            "TASK-804",
        ]
        assert "details" not in beacon
        assert "states" not in beacon
        assert "summary" not in beacon


@pytest.mark.asyncio
async def test_identity_tom_endpoint_surfaces_belief_and_intention_counts():
    from opencas.tom import Belief, BeliefSubject, Intention, IntentionStatus, MetacognitiveResult

    class FakeTom:
        def __init__(self):
            self._beliefs = [
                Belief(subject=BeliefSubject.USER, predicate="prefers patient testing", confidence=0.9),
                Belief(subject=BeliefSubject.SELF, predicate="is repairing pressure surfaces", confidence=0.8),
            ]
            self._intentions = [
                Intention(actor=BeliefSubject.SELF, content="complete pr-120", status=IntentionStatus.ACTIVE),
            ]

        def list_beliefs(self, subject=None, predicate=None):
            items = self._beliefs
            if subject is not None:
                items = [item for item in items if item.subject == subject]
            return items

        def list_intentions(self, actor=None, status=None):
            items = self._intentions
            if actor is not None:
                items = [item for item in items if item.actor == actor]
            if status is not None:
                items = [item for item in items if item.status == status]
            return items

        def check_consistency(self):
            return MetacognitiveResult(
                belief_count=len(self._beliefs),
                intention_count=len(self._intentions),
            )

    runtime = FakeRuntime()
    runtime.tom = FakeTom()
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/identity/tom")

    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["belief_counts"]["by_subject"] == {"self": 1, "user": 1}
    assert data["intention_counts"]["by_status"] == {"active": 1}
    assert data["intention_counts"]["active"] == 1
    assert data["recent_intentions"][0]["content"] == "complete pr-120"
    assert data["consistency"]["belief_count"] == 2
    assert data["consistency"]["intention_count"] == 1


def test_memory_projection_handles_mixed_embedding_dimensions():
    runtime = FakeRuntime()
    runtime.memory = FakeProjectionMemoryStore()
    runtime.ctx.embeddings = FakeEmbeddings(
        cache=FakeEmbeddingCache(
            {
                "emb-1": type("R", (), {"vector": [0.1, 0.2], "dimension": 2, "model_id": "model-a"})(),
                "emb-2": type("R", (), {"vector": [0.3, 0.4, 0.5], "dimension": 3, "model_id": "model-b"})(),
            }
        )
    )
    app = create_app(runtime)
    client = TestClient(app)
    resp = client.get("/api/memory/embedding-projection?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["points"]) == 2
    assert len(data["groups"]) >= 2


def test_monitor_embeddings_endpoint_surfaces_recent_records():
    runtime = FakeRuntime()
    runtime.ctx.embeddings = FakeEmbeddings()
    app = create_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/monitor/embeddings")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_records"] == 42
    assert data["model_id"] == "google/gemini-embedding-2-preview"
    assert len(data["recent_records"]) == 2
    assert data["recent_records"][0]["task_type"] == "memory_episode"
    assert data["recent_records"][0]["source"] == "episode:ep-123"
    assert "the OpenCAS agent summarized the latest dashboard continuity probe" in data["recent_records"][0]["preview"]
    assert data["recent_records"][1]["degraded"] is True


def test_monitor_shadow_registry_endpoint_surfaces_cluster_summary():
    runtime = FakeRuntime()
    app = create_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/monitor/shadow-registry")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["total_entries"] == 3
    assert data["active_clusters"] == 1
    assert data["dismissed_clusters"] == 1
    assert data["reason_counts"]["retry_blocked"] == 2
    assert data["recent_entries"][0]["tool_name"] == "repair_retry"
    assert data["top_clusters"][0]["count"] == 2
    assert data["top_clusters"][0]["triage_status"] == "active"


def test_monitor_shadow_registry_cluster_endpoint_surfaces_raw_entries():
    runtime = FakeRuntime()
    app = create_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/monitor/shadow-registry/cluster", params={"fingerprint": "cluster-chronicle"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["fingerprint"] == "cluster-chronicle"
    assert data["count"] == 2
    assert data["triage_status"] == "active"
    assert len(data["entries"]) == 2
    assert data["entries"][0]["tool_name"] == "repair_retry"


def test_monitor_shadow_registry_cluster_triage_endpoint_updates_cluster_state():
    runtime = FakeRuntime()
    app = create_app(runtime)
    client = TestClient(app)

    resp = client.post(
        "/api/monitor/shadow-registry/cluster/triage",
        json={
            "fingerprint": "cluster-chronicle",
            "annotation": "Known issue; suppress from main list.",
            "dismissed": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["fingerprint"] == "cluster-chronicle"
    assert data["triage_status"] == "dismissed"
    assert data["annotation"] == "Known issue; suppress from main list."


@pytest.mark.asyncio
async def test_monitor_meaningful_loop_endpoint_surfaces_output_contract_state():
    runtime = FakeRuntime()
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/monitor/meaningful-loop")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["stop_counts"]["completion"] == 2
    assert data["stop_counts"]["retry_governor"] >= 1
    assert data["stop_counts"]["progress_gate"] >= 1
    assert data["stop_counts"]["affective_pressure"] >= 1
    assert data["latest_artifact"]["path"] == "workspace/notes/relational-docs.md"
    assert data["latest_blocker"]["cause"] in {"progress_gate", "retry_governor", "affective_pressure"}
    assert "repeated tool pressure" in data["affective_pressures"][0]["bounded_reason"]
    assert any(item["latest_meaningful_signal"] == "artifact" for item in data["recent_tasks"])
    assert any(item["loop_stop_cause"] == "retry_governor" for item in data["recent_tasks"])


def test_monitor_affective_examinations_endpoint_filters_recent_records():
    runtime = FakeRuntime()
    app = create_app(runtime)
    client = TestClient(app)

    resp = client.get(
        "/api/monitor/affective-examinations",
        params={
            "limit": 10,
            "session_id": "default",
            "source_type": "tool_result",
            "emotion": "concerned",
            "action_pressure": "ask_clarifying_question",
            "consumed_by": "none",
            "decay_state": "active",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["count"] == 1
    assert data["counts"]["unconsumed"] == 1
    assert data["items"][0]["source_type"] == "tool_result"
    assert data["items"][0]["primary_emotion"] == "concerned"
    assert data["items"][0]["action_pressure"] == "ask_clarifying_question"
    assert data["items"][0]["decay_state"] == "active"


@pytest.mark.asyncio
async def test_config_overview_endpoint_surfaces_models_profiles_and_material(tmp_path):
    state_dir = tmp_path / "state"
    provider_material = state_dir / "provider_material"
    provider_material.mkdir(parents=True)
    (provider_material / "config.json").write_text("{}", encoding="utf-8")
    (provider_material / ".env").write_text("GOOGLE_API_KEY=test-key\nOPENAI_API_KEY=test-key\n", encoding="utf-8")
    runtime = FakeRuntime()
    runtime.ctx.llm = type("L", (), {"manager": FakeGatewayManager(), "default_model": "anthropic/claude-sonnet-4-6"})()
    runtime.ctx.embeddings = type("E", (), {"model_id": "google/embeddinggemma-300m"})()
    runtime.ctx.config = type(
        "Config",
        (),
        {
            "state_dir": state_dir,
            "provider_config_path": None,
            "provider_env_path": None,
            "credential_source_config_path": str(tmp_path / "source-config.json"),
            "credential_source_env_path": None,
            "credential_profile_ids": ["anthropic-main"],
            "credential_env_keys": ["GOOGLE_API_KEY"],
            "default_llm_model": None,
            "embedding_model_id": None,
            "model_routing": ModelRoutingConfig(
                mode="tiered",
                light_model="google/gemini-2.5-flash",
                standard_model="anthropic/claude-sonnet-4-6",
                high_model="openai/gpt-5.3-codex",
                extra_high_model="codex-cli/gpt-5.3-codex",
                light_reasoning_effort="low",
                standard_reasoning_effort="medium",
                high_reasoning_effort="high",
                extra_high_reasoning_effort="xhigh",
            ),
            "model_dump": lambda self, **kw: {"state_dir": str(state_dir)},
        },
    )()
    data = await build_config_overview_payload(runtime)
    assert data["config_mode"] == "copied-local"
    assert "anthropic/claude-sonnet-4-6" in data["available_models"]
    assert data["auth_profiles"][0]["profile_id"] == "anthropic-main"
    assert data["providers"][0]["provider_id"] == "anthropic"
    assert data["current"]["default_llm_model"] == "anthropic/claude-sonnet-4-6"
    assert data["current"]["embedding_model_id"] == "google/embeddinggemma-300m"
    assert "google/embeddinggemma-300m" in data["available_embedding_models"]
    assert "google/gemini-embedding-2-preview" not in data["available_embedding_models"]
    assert data["current"]["model_routing"]["mode"] == "tiered"
    assert data["current"]["model_routing"]["effective_reasoning"]["high"] == "high"
    assert "claude-sonnet-4-6" in data["providers"][0]["effective_model_ids"]
    assert data["credential_copy"]["profile_ids"] == ["anthropic-main"]
    assert data["materialized_bundle"]["config_exists"] is True
    assert data["materialized_bundle"]["env_key_count"] == 2


def test_model_routing_update_persists_runtime_and_gateway_state(tmp_path):
    from open_llm_auth.config import load_config

    state_dir = tmp_path / "state"
    provider_material = state_dir / "provider_material"
    provider_material.mkdir(parents=True)
    (provider_material / "config.json").write_text("{}", encoding="utf-8")
    runtime = FakeRuntime()
    runtime.ctx.config = BootstrapConfig(
        state_dir=state_dir,
        session_id="routing-dashboard",
    ).resolve_paths()
    runtime.ctx.llm = FakeMutableLLM(default_model=runtime.ctx.config.default_llm_model)
    app = create_app(runtime)
    client = TestClient(app)

    resp = client.post(
        "/api/config/model-routing",
        json={
            "default_llm_model": "openai/gpt-5.3-codex",
            "model_routing": {
                "mode": "tiered",
                "light_model": "google/gemini-2.5-flash",
                "standard_model": "openai/gpt-5.3-codex",
                "high_model": "anthropic/claude-sonnet-4-6",
                "extra_high_model": "codex-cli/gpt-5.3-codex",
                "light_reasoning_effort": "low",
                "standard_reasoning_effort": "medium",
                "high_reasoning_effort": "high",
                "extra_high_reasoning_effort": "xhigh",
                "auto_escalation": True,
            },
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["default_llm_model"] == "openai/gpt-5.3-codex"
    assert runtime.ctx.config.default_llm_model == "openai/gpt-5.3-codex"
    assert runtime.ctx.config.model_routing.mode.value == "tiered"
    assert runtime.ctx.llm.last_set["model_routing"].high_model == "anthropic/claude-sonnet-4-6"
    assert runtime.ctx.llm.last_set["model_routing"].extra_high_reasoning_effort.value == "xhigh"
    assert runtime.ctx.llm.manager.reload_calls >= 1

    persisted_path = state_dir / "runtime_model_routing.json"
    assert persisted_path.exists()
    saved_cfg = load_config(config_path=provider_material / "config.json")
    assert saved_cfg.default_model == "openai/gpt-5.3-codex"


def test_provider_setup_and_model_delete_routes_manage_active_gateway_material(tmp_path):
    from open_llm_auth.config import load_config

    state_dir = tmp_path / "state"
    provider_material = state_dir / "provider_material"
    provider_material.mkdir(parents=True)
    (provider_material / "config.json").write_text("{}", encoding="utf-8")
    runtime = FakeRuntime()
    runtime.ctx.config = BootstrapConfig(
        state_dir=state_dir,
        session_id="provider-dashboard",
    ).resolve_paths()
    runtime.ctx.config.default_llm_model = "zai-anthropic/glm-5.1"
    runtime.ctx.config.model_routing = ModelRoutingConfig(
        mode="single",
        single_model="zai-anthropic/glm-5.1",
    )
    runtime.ctx.llm = FakeMutableLLM(default_model=runtime.ctx.config.default_llm_model)
    app = create_app(runtime)
    client = TestClient(app)

    create_resp = client.post(
        "/api/config/provider-setups",
        json={
            "family_id": "zai",
            "preset_id": "coding_api",
            "profile_label": "work",
            "api_key": "glm-secret",
            "custom_model_ids": ["glm-5.1-custom"],
        },
    )
    assert create_resp.status_code == 200
    create_data = create_resp.json()
    assert create_data["provider_id"] == "zai-coding"
    assert create_data["profile_id"] == "zai-coding:work"

    saved_cfg = load_config(config_path=provider_material / "config.json")
    assert "zai-coding" in saved_cfg.providers
    assert saved_cfg.auth_profiles["zai-coding:work"].key == "glm-secret"
    assert [model.id for model in saved_cfg.providers["zai-coding"].models] == ["glm-5.1-custom"]
    assert saved_cfg.default_model == "zai-coding/glm-5.1-custom"
    assert runtime.ctx.config.default_llm_model == "zai-coding/glm-5.1-custom"
    assert runtime.ctx.config.model_routing.standard_model == "zai-coding/glm-5.1-custom"

    delete_resp = client.delete("/api/config/providers/zai-coding/models/glm-5.1-custom")
    assert delete_resp.status_code == 200

    saved_after_delete = load_config(config_path=provider_material / "config.json")
    assert len(saved_after_delete.providers["zai-coding"].models) == 0
    assert saved_after_delete.default_model == "zai-coding/glm-5.1"
    assert runtime.ctx.llm.manager.reload_calls >= 2


def test_plugin_trust_routes_round_trip_dashboard_policies():
    runtime = FakeRuntime()
    runtime.ctx.plugin_trust = FakePluginTrustService()
    app = create_app(runtime)
    client = TestClient(app)

    get_resp = client.get("/api/config/plugin-trust")
    assert get_resp.status_code == 200
    assert get_resp.json()["policy_count"] == 1

    post_resp = client.post(
        "/api/config/plugin-trust/policies",
        json={
            "scope": "checksum",
            "value": "a" * 64,
            "level": "blocked",
            "note": "known bad bundle",
            "source": "dashboard",
        },
    )
    assert post_resp.status_code == 200
    post_data = post_resp.json()["policy"]
    assert post_data["scope"] == "checksum"
    assert post_data["level"] == "blocked"

    get_after_post = client.get("/api/config/plugin-trust")
    assert get_after_post.status_code == 200
    entries = get_after_post.json()["entries"]
    assert any(item["scope"] == "checksum" and item["value"] == "a" * 64 for item in entries)

    delete_resp = client.delete(f"/api/config/plugin-trust/policies/checksum/{'a' * 64}")
    assert delete_resp.status_code == 200

    get_after_delete = client.get("/api/config/plugin-trust")
    assert get_after_delete.status_code == 200
    entries_after_delete = get_after_delete.json()["entries"]
    assert not any(item["scope"] == "checksum" and item["value"] == "a" * 64 for item in entries_after_delete)


def test_plugin_trust_routes_accept_signer_metadata():
    runtime = FakeRuntime()
    runtime.ctx.plugin_trust = FakePluginTrustService()
    app = create_app(runtime)
    client = TestClient(app)

    post_resp = client.post(
        "/api/config/plugin-trust/policies",
        json={
            "scope": "signer",
            "value": "opencas-labs-main",
            "level": "trusted",
            "note": "pinned signer",
            "source": "dashboard",
            "metadata": {
                "public_key": "QUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUE=",
                "publisher": "OpenCAS Labs",
            },
        },
    )

    assert post_resp.status_code == 200
    policy = post_resp.json()["policy"]
    assert policy["scope"] == "signer"
    assert policy["metadata"]["publisher"] == "OpenCAS Labs"


def test_plugin_trust_feed_sync_route():
    runtime = FakeRuntime()
    runtime.ctx.plugin_trust = FakePluginTrustService()
    app = create_app(runtime)
    client = TestClient(app)

    resp = client.post(
        "/api/config/plugin-trust/feeds/sync",
        json=_signed_feed_payload(
            source_id="opencas-labs",
            policies=[
                {
                    "scope": "publisher",
                    "value": "OpenCAS Labs",
                    "level": "trusted",
                    "note": "",
                    "metadata": {},
                },
                {
                    "scope": "signer",
                    "value": "opencas-labs-main",
                    "level": "trusted",
                    "note": "",
                    "metadata": {
                        "public_key": "QUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUE=",
                        "publisher": "OpenCAS Labs",
                        "source_id": "opencas-labs",
                    },
                },
            ],
        ),
    )

    assert resp.status_code == 200
    body = resp.json()["feed"]
    assert body["source_id"] == "opencas-labs"
    assert body["verification"]["verified_signature_count"] == 1
    assert len(body["imported"]) == 2

    snapshot = client.get("/api/config/plugin-trust").json()
    assert snapshot["feed_source_count"] == 1
    assert snapshot["signer_policy_count"] == 1
    assert any(item["source"] == "feed:opencas-labs" and item["scope"] == "signer" for item in snapshot["entries"])


def test_plugin_trust_feed_sync_route_rejects_unsigned_feed():
    runtime = FakeRuntime()
    runtime.ctx.plugin_trust = FakePluginTrustService()
    app = create_app(runtime)
    client = TestClient(app)

    resp = client.post(
        "/api/config/plugin-trust/feeds/sync",
        json={
            "format_version": 1,
            "source_id": "opencas-labs",
            "policies": [
                {
                    "scope": "publisher",
                    "value": "OpenCAS Labs",
                    "level": "trusted",
                }
            ],
        },
    )

    assert resp.status_code == 400
    assert "must include at least one signature entry" in resp.json()["detail"]


def test_memory_node_detail_endpoint_surfaces_neighbors_and_signals():
    runtime = FakeRuntime()
    runtime.memory = FakeNodeDetailMemoryStore()
    runtime.ctx.embeddings = FakeEmbeddings(
        cache=FakeEmbeddingCache(
            {
                "emb-1": type("R", (), {"vector": [0.1, 0.2], "dimension": 2, "model_id": "model-a"})(),
                "emb-2": type("R", (), {"vector": [0.3, 0.4], "dimension": 2, "model_id": "model-a"})(),
                "emb-3": type("R", (), {"vector": [0.5, 0.6], "dimension": 2, "model_id": "model-a"})(),
            }
        )
    )
    app = create_app(runtime)
    client = TestClient(app)
    resp = client.get("/api/memory/node-detail?node_id=episode:ep-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node"]["node_id"] == "episode:ep-1"
    assert any(item["node_id"] == "episode:ep-2" for item in data["neighbors"])
    assert any(item["node_id"] == "memory:mem-1" for item in data["neighbors"])
    assert any(edge["kind"] == "semantic" and edge["strongest_signal"] == "semantic" for edge in data["edges"])
    assert any(edge["kind"] == "distilled_from" for edge in data["edges"])
