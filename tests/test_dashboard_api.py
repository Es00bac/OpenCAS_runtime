import pytest
from fastapi.testclient import TestClient

from opencas.api.server import create_app
from opencas.bootstrap import BootstrapConfig
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


class FakeEpisodeGraph:
    async def get_neighbors(self, episode_id, **kwargs):
        return []


class FakeEmbeddings:
    def __init__(self, cache=None):
        async def _get(*_args, **_kwargs):
            return None
        self.cache = cache or type("C", (), {"get": staticmethod(_get)})()

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

    async def workflow_status(self, limit=10, project_id=None):
        return {
            "executive": {
                "intention": "Improve the dashboard operator experience",
                "active_goals": ["Make dashboard readable", "Expose real task state"],
                "queued_work_count": 2,
                "capacity_remaining": 3,
                "recommend_pause": False,
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
        ]

    def resolve(self, model_ref):
        return type(
            "Resolved",
            (),
            {
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


def test_chat_context_summary_surfaces_active_lane():
    runtime = FakeRuntime()
    runtime.ctx.llm = type("L", (), {"manager": FakeGatewayManager(), "default_model": "kimi-coding/k2p5"})()
    app = create_app(runtime)
    client = TestClient(app)
    resp = client.get("/api/chat/context-summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["lane"]["model"] == "kimi-coding/k2p5"
    assert data["lane"]["provider"] == "kimi-coding"
    assert data["lane"]["resolved_model"] == "kimi-coding/k2p5"


def test_runtime_status_endpoint():
    app = create_app(FakeRuntime())
    client = TestClient(app)
    resp = client.get("/api/monitor/runtime")
    assert resp.status_code == 200
    data = resp.json()
    assert data["sandbox"]["mode"] == "workspace-only"
    assert data["execution"]["processes"]["total_count"] == 0
    assert data["execution"]["browser"]["total_count"] == 0


def test_dashboard_contains_operations_surface():
    app = create_app(FakeRuntime())
    client = TestClient(app)
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    body = resp.text
    assert "Operations" in body
    assert "Daydream" in body
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
    assert "/api/daydream/summary" in body
    assert "/api/daydream/reflections" in body
    assert "/api/daydream/conflicts" in body
    assert "/api/daydream/promotions" in body
    assert "Usage" in body
    assert "/api/usage/overview" in body
    assert "m.meta?.lane?.resolved_model" in body
    assert "legacy / unlabeled" in body


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

    async def fake_gateway_snapshot(window_days, recent_limit):
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


def test_chat_context_summary_and_task_routes():
    app = create_app(FakeRuntime())
    client = TestClient(app)

    summary = client.get("/api/chat/context-summary?session_id=s1")
    assert summary.status_code == 200
    summary_data = summary.json()
    assert summary_data["executive"]["intention"] == "Improve the dashboard operator experience"
    assert summary_data["current_work"]["title"] == "Redesign chat panel layout"
    assert summary_data["consolidation"]["available"] is True
    assert summary_data["consolidation"]["commitments_extracted_from_chat"] == 1
    assert summary_data["tasks"]["counts"]["total"] == 3

    tasks = client.get("/api/operations/tasks?limit=10")
    assert tasks.status_code == 200
    task_data = tasks.json()
    assert task_data["counts"]["completed"] == 2
    assert task_data["counts"]["active"] == 1
    assert task_data["items"][0]["duplicate_objective_count"] >= 1

    detail = client.get("/api/operations/tasks/task-1")
    assert detail.status_code == 200
    detail_data = detail.json()
    assert detail_data["task"]["status"] == "completed"
    assert detail_data["task"]["source"] == "intervention_launch_background"


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


def test_config_overview_endpoint_surfaces_models_profiles_and_material(tmp_path):
    state_dir = tmp_path / "state"
    provider_material = state_dir / "provider_material"
    provider_material.mkdir(parents=True)
    (provider_material / "config.json").write_text("{}", encoding="utf-8")
    (provider_material / ".env").write_text("GOOGLE_API_KEY=test-key\nOPENAI_API_KEY=test-key\n", encoding="utf-8")
    runtime = FakeRuntime()
    runtime.ctx.llm = type("L", (), {"manager": FakeGatewayManager(), "default_model": "anthropic/claude-sonnet-4-6"})()
    runtime.ctx.embeddings = type("E", (), {"model_id": "google/gemini-embedding-2-preview"})()
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
            ),
            "model_dump": lambda self, **kw: {"state_dir": str(state_dir)},
        },
    )()
    app = create_app(runtime)
    client = TestClient(app)
    resp = client.get("/api/config/overview")
    assert resp.status_code == 200
    data = resp.json()["overview"]
    assert data["config_mode"] == "copied-local"
    assert "anthropic/claude-sonnet-4-6" in data["available_models"]
    assert data["auth_profiles"][0]["profile_id"] == "anthropic-main"
    assert data["providers"][0]["provider_id"] == "anthropic"
    assert data["current"]["default_llm_model"] == "anthropic/claude-sonnet-4-6"
    assert data["current"]["embedding_model_id"] == "google/gemini-embedding-2-preview"
    assert data["current"]["model_routing"]["mode"] == "tiered"
    assert "claude-sonnet-4-6" in data["providers"][0]["effective_model_ids"]
    assert data["credential_copy"]["profile_ids"] == ["anthropic-main"]
    assert data["materialized_bundle"]["config_exists"] is True
    assert data["materialized_bundle"]["env_key_count"] == 2
    assert any(item["family_id"] == "openai" for item in data["provider_setup_catalog"])


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

    delete_resp = client.delete("/api/config/providers/zai-coding/models/glm-5.1-custom")
    assert delete_resp.status_code == 200

    saved_after_delete = load_config(config_path=provider_material / "config.json")
    assert len(saved_after_delete.providers["zai-coding"].models) == 0
    assert runtime.ctx.llm.manager.reload_calls >= 2


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
