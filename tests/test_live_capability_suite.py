import argparse
import asyncio
import sqlite3

from scripts.opencas_live_capability_suite import (
    CheckResult,
    _chat_sessions_validator,
    _cognitive_flow_validator,
    _daydream_validator,
    _partner_readiness_score,
    _persistence_check,
    _result_from_http,
    _score,
    _tom_summary_validator,
    run_suite,
)


def _check(name: str, *, ok: bool = True, required: bool = True, details: dict | None = None) -> CheckResult:
    return CheckResult(name=name, ok=ok, required=required, details=details or {})


def test_result_from_http_retries_valid_latency_outlier_before_failure(monkeypatch) -> None:
    calls = []

    def fake_json_get(url, *, timeout, max_bytes):
        calls.append(url)
        elapsed_ms = 4500.0 if len(calls) == 1 else 42.0
        return {"ok": True}, {"status": 200, "elapsed_ms": elapsed_ms, "bytes": 12}

    monkeypatch.setattr(
        "scripts.opencas_live_capability_suite._json_get",
        fake_json_get,
    )

    result = _result_from_http(
        "memory_stats",
        "http://127.0.0.1:32147",
        "/api/memory/stats",
        timeout=1,
        validator=lambda payload: (bool(payload.get("ok")), "valid payload", {}),
        max_latency_ms=3000.0,
    )

    assert result.ok is True
    assert len(calls) == 2
    assert result.elapsed_ms == 42.0
    assert result.details["latency_retry_count"] == 1
    assert result.details["first_elapsed_ms"] == 4500.0


def test_partner_readiness_is_separate_from_endpoint_health() -> None:
    results = [
        _check("health"),
        _check(
            "context_summary",
            details={"provider": "openai-codex", "model": "openai/gpt-5.5", "auth_source": "profile:openai-codex:chatgpt-plus"},
        ),
        _check("schedule_agenda", details={"counts": {"active": 2}, "auth_failures": [], "recent_run_count": 3, "latest_run_status": "recorded"}),
        _check("memory_stats", details={"memory_count": 100, "edge_count": 200}),
        _check("memory_activity", required=False, details={"event_count": 0}),
        _check("cognitive_flow", details={"event_count": 80, "edge_count": 120, "temporal_missing_node_count": 0}),
        _check("daydream_summary", details={"active_conflicts": 8, "recent_associations": 0, "recent_run_count": 200, "consecutive_skip_count": 99}),
        _check("runtime_truth", details={"tool_count": 130, "capability_count": 109}),
        _check("cognitive_health", details={"unhealthy_checks": []}),
        _check("google_workspace_auth", details={}),
        _check(
            "executive_assistant",
            details={
                "pa_score": 0.777,
                "duplicate_tool_call_rate": 0.3726,
                "daydream_consecutive_skip_count": 99,
                "item_status_counts": {"pass": 3, "pending": 2},
            },
        ),
        _check("qualification_summary", required=False, details={"agent_success_rate": 0.667, "total_agent_checks": 6}),
    ]

    endpoint_score = _score(results)
    readiness = _partner_readiness_score(results)

    assert endpoint_score["percent_required_ok"] == 100.0
    assert readiness["percent"] < endpoint_score["percent_required_ok"]
    assert readiness["dimension_scores"]["reflective_processing"]["status"] == "warn"
    assert readiness["dimension_scores"]["behavioral_qualification"]["status"] == "warn"


def test_partner_readiness_reaches_full_when_behavioral_evidence_is_strong() -> None:
    results = [
        _check("health"),
        _check(
            "context_summary",
            details={"provider": "openai-codex", "model": "openai/gpt-5.5", "auth_source": "profile:openai-codex:chatgpt-plus"},
        ),
        _check("schedule_agenda", details={"counts": {"active": 2}, "auth_failures": [], "recent_run_count": 3, "latest_run_status": "recorded"}),
        _check("memory_stats", details={"memory_count": 100, "edge_count": 200}),
        _check("memory_activity", required=False, details={"event_count": 20}),
        _check("cognitive_flow", details={"event_count": 80, "edge_count": 120, "temporal_missing_node_count": 0}),
        _check("daydream_summary", details={"active_conflicts": 8, "recent_associations": 8, "recent_run_count": 200, "consecutive_skip_count": 3}),
        _check("runtime_truth", details={"tool_count": 130, "capability_count": 109}),
        _check("cognitive_health", details={"unhealthy_checks": []}),
        _check("google_workspace_auth", details={}),
        _check(
            "executive_assistant",
            details={
                "pa_score": 0.94,
                "duplicate_tool_call_rate": 0.05,
                "daydream_consecutive_skip_count": 3,
                "item_status_counts": {"pass": 5, "pending": 0},
            },
        ),
        _check("qualification_summary", required=False, details={"agent_success_rate": 0.95, "total_agent_checks": 20}),
    ]

    readiness = _partner_readiness_score(results)

    assert readiness["percent"] == 100.0
    assert all(dimension["status"] == "pass" for dimension in readiness["dimension_scores"].values())


def test_partner_readiness_prefers_current_model_qualification_over_stale_history() -> None:
    results = [
        _check("health"),
        _check(
            "context_summary",
            details={"provider": "openai-codex", "model": "openai/gpt-5.5", "auth_source": "profile:openai-codex:chatgpt-plus"},
        ),
        _check("schedule_agenda", details={"counts": {"active": 2}, "auth_failures": [], "recent_run_count": 3, "latest_run_status": "recorded"}),
        _check("memory_stats", details={"memory_count": 100, "edge_count": 200}),
        _check("memory_activity", required=False, details={"event_count": 20}),
        _check("cognitive_flow", details={"event_count": 80, "edge_count": 120, "temporal_missing_node_count": 0}),
        _check("daydream_summary", details={"active_conflicts": 8, "recent_associations": 8, "recent_run_count": 200, "consecutive_skip_count": 3}),
        _check("runtime_truth", details={"tool_count": 130, "capability_count": 109}),
        _check("cognitive_health", details={"unhealthy_checks": []}),
        _check("google_workspace_auth", details={}),
        _check(
            "executive_assistant",
            details={
                "pa_score": 0.94,
                "duplicate_tool_call_rate": 0.05,
                "daydream_consecutive_skip_count": 3,
                "item_status_counts": {"pass": 5, "pending": 0},
            },
        ),
        _check(
            "qualification_summary",
            required=False,
            details={
                "agent_success_rate": 0.5,
                "total_agent_checks": 30,
                "current_model_agent_success_rate": 1.0,
                "current_model_total_agent_checks": 12,
            },
        ),
    ]

    readiness = _partner_readiness_score(results)

    assert readiness["dimension_scores"]["behavioral_qualification"]["status"] == "pass"
    assert (
        readiness["dimension_scores"]["behavioral_qualification"]["details"]["qualification_agent_success_rate"]
        == 1.0
    )


def test_partner_readiness_accepts_flow_when_short_memory_activity_window_is_empty() -> None:
    results = [
        _check("health"),
        _check("context_summary", details={"provider": "openai-codex", "model": "openai/gpt-5.5", "auth_source": "profile:openai-codex:chatgpt-plus"}),
        _check("schedule_agenda", details={"counts": {"active": 2}, "auth_failures": [], "recent_run_count": 3, "latest_run_status": "recorded"}),
        _check("memory_stats", details={"memory_count": 100, "edge_count": 200}),
        _check("memory_activity", required=False, details={"event_count": 0}),
        _check("cognitive_flow", details={"event_count": 80, "edge_count": 120, "temporal_missing_node_count": 0}),
        _check("daydream_summary", details={"recent_associations": 8, "recent_run_count": 20, "consecutive_skip_count": 3}),
        _check("runtime_truth", details={"tool_count": 130, "capability_count": 109}),
        _check("cognitive_health", details={"unhealthy_checks": []}),
        _check("google_workspace_auth", details={}),
        _check("executive_assistant", details={"pa_score": 0.94, "duplicate_tool_call_rate": 0.05, "daydream_consecutive_skip_count": 3, "item_status_counts": {"pass": 5, "pending": 0}}),
        _check("qualification_summary", required=False, details={"agent_success_rate": 0.95, "total_agent_checks": 20}),
    ]

    readiness = _partner_readiness_score(results)

    assert readiness["dimension_scores"]["memory_observability"]["status"] == "pass"


def test_partner_readiness_accepts_daydream_healthy_pause_skip_diagnostic() -> None:
    results = [
        _check("health"),
        _check("context_summary", details={"provider": "openai-codex", "model": "openai/gpt-5.5", "auth_source": "profile:openai-codex:chatgpt-plus"}),
        _check("schedule_agenda", details={"counts": {"active": 2}, "auth_failures": [], "recent_run_count": 3, "latest_run_status": "recorded"}),
        _check("memory_stats", details={"memory_count": 100, "edge_count": 200}),
        _check("memory_activity", required=False, details={"event_count": 20}),
        _check("cognitive_flow", details={"event_count": 80, "edge_count": 120, "temporal_missing_node_count": 0}),
        _check(
            "daydream_summary",
            details={
                "recent_associations": 8,
                "recent_run_count": 20,
                "consecutive_skip_count": 15,
                "skip_diagnostic_status_for_operator": "healthy_pause",
            },
        ),
        _check("runtime_truth", details={"tool_count": 130, "capability_count": 109}),
        _check("cognitive_health", details={"unhealthy_checks": []}),
        _check("google_workspace_auth", details={}),
        _check("executive_assistant", details={"pa_score": 0.94, "duplicate_tool_call_rate": 0.05, "daydream_consecutive_skip_count": 14, "item_status_counts": {"pass": 5, "pending": 0}}),
        _check("qualification_summary", required=False, details={"agent_success_rate": 0.95, "total_agent_checks": 20}),
    ]

    readiness = _partner_readiness_score(results)

    assert readiness["dimension_scores"]["reflective_processing"]["status"] == "pass"


def test_cognitive_flow_validator_requires_temporal_metadata_and_reports_readable_counts() -> None:
    ok, summary, details = _cognitive_flow_validator(
        {
            "event_count": 2,
            "nodes": [{"temporal_has_metadata": True}, {"temporal_has_metadata": True}],
            "edges": [{"temporal_has_metadata": True}],
            "stats": {
                "available": True,
                "limit": 10,
                "temporal_missing_node_count": 0,
                "temporal_missing_edge_count": 0,
                "readable_thought_count": 1,
                "recent_readable_thoughts": [{"thought_text": "working"}],
            },
        }
    )

    assert ok is True
    assert "2 flow events" in summary
    assert details["temporal_missing_node_count"] == 0
    assert details["temporal_missing_edge_count"] == 0
    assert details["readable_thought_count"] == 1
    assert details["recent_readable_thought_count"] == 1

    bad_ok, _, bad_details = _cognitive_flow_validator(
        {
            "event_count": 2,
            "nodes": [{}, {}],
            "edges": [],
            "stats": {"available": True, "temporal_missing_node_count": 1},
        }
    )

    assert bad_ok is False
    assert bad_details["temporal_missing_node_count"] == 1


def test_daydream_validator_uses_compacted_association_counts_and_useful_runs() -> None:
    ok, summary, details = _daydream_validator(
        {
            "active_conflicts": [],
            "recent_association_memories": [],
            "recent_runs": [
                {"quality_status": "skipped", "had_activity": False},
                {"quality_status": "useful", "had_activity": True},
            ],
            "summary": {
                "details_compacted": True,
                "association_memory_count": 42,
                "recent_run_count": 2,
                "consecutive_skip_count": 0,
                "latest_active_run": {"quality_status": "useful"},
                "last_successful_run_at": "2026-05-13T00:00:00Z",
            },
        }
    )

    assert ok is True
    assert "42 associations" in summary
    assert details["recent_associations"] == 42
    assert details["useful_recent_run_count"] == 1
    assert details["latest_quality_status"] == "useful"


def test_tom_summary_validator_surfaces_question_echo_gap_without_endpoint_failure() -> None:
    ok, summary, details = _tom_summary_validator(
        {
            "available": True,
            "belief_counts": {"total": 10, "by_subject": {"user": 8}},
            "intention_counts": {"total": 3, "active": 1},
            "recent_beliefs": [
                {
                    "relation": "asked",
                    "predicate": "asked: can you do this",
                    "effective_confidence": 0.7,
                    "evidence_ids": ["conversation_turn:abc"],
                },
                {
                    "relation": "prefers",
                    "predicate": "prefers concise updates",
                    "effective_confidence": 0.8,
                    "evidence_ids": ["conversation_turn:def"],
                },
            ],
            "consistency": {"contradictions": [], "warnings": []},
        }
    )

    assert ok is True
    assert "10 beliefs" in summary
    assert details["recent_question_echo_count"] == 1
    assert details["recent_behavioral_relation_count"] == 1
    assert details["recent_evidence_id_count"] == 2


def test_chat_sessions_validator_requires_continuity_surface() -> None:
    ok, _, details = _chat_sessions_validator(
        {"sessions": [{"session_id": "s1", "status": "active", "message_count": 3}]}
    )

    assert ok is True
    assert details["session_count"] == 1
    assert details["sample_message_total"] == 3

    empty_ok, _, empty_details = _chat_sessions_validator({"sessions": []})

    assert empty_ok is False
    assert empty_details["session_count"] == 0


def test_persistence_check_reads_required_store_counts(tmp_path) -> None:
    store_tables = {
        "memory.db": ("episodes", "memories"),
        "tom.db": ("beliefs", "intentions"),
        "daydream.db": ("daydream_reflections",),
        "schedules.db": ("schedules", "schedule_runs"),
        "context.db": ("sessions", "messages"),
        "cognitive_state.db": ("cognitive_events", "working_memory_items"),
        "daydream_signals.db": ("daydream_signals", "daydream_signal_routes"),
    }
    for db_name, tables in store_tables.items():
        conn = sqlite3.connect(tmp_path / db_name)
        try:
            for table in tables:
                conn.execute(f'create table "{table}" (id integer primary key)')
                conn.execute(f'insert into "{table}" default values')
            conn.commit()
        finally:
            conn.close()

    result = _persistence_check(tmp_path, tmp_path, timeout=1.0)

    assert result.ok is True
    assert result.details["counts"]["memory"]["episodes"] == 1
    assert result.details["missing_files"] == []


def test_partner_readiness_does_not_fail_duplicate_rate_on_tiny_qualified_sample() -> None:
    results = [
        _check("health"),
        _check("context_summary", details={"provider": "openai-codex", "model": "openai/gpt-5.5", "auth_source": "profile:openai-codex:chatgpt-plus"}),
        _check("schedule_agenda", details={"counts": {"active": 2}, "auth_failures": [], "recent_run_count": 3, "latest_run_status": "recorded"}),
        _check("memory_stats", details={"memory_count": 100, "edge_count": 200}),
        _check("memory_activity", required=False, details={"event_count": 20}),
        _check("cognitive_flow", details={"event_count": 80, "edge_count": 120, "temporal_missing_node_count": 0}),
        _check("daydream_summary", details={"recent_associations": 8, "recent_run_count": 20, "consecutive_skip_count": 3}),
        _check("runtime_truth", details={"tool_count": 130, "capability_count": 109}),
        _check("cognitive_health", details={"unhealthy_checks": []}),
        _check("google_workspace_auth", details={}),
        _check(
            "executive_assistant",
            details={
                "pa_score": 0.976,
                "duplicate_tool_call_rate": 0.3333,
                "duplicate_tool_call_total": 3,
                "daydream_consecutive_skip_count": 3,
                "item_status_counts": {"pass": 5, "pending": 0},
            },
        ),
        _check("qualification_summary", required=False, details={"agent_success_rate": 0.95, "total_agent_checks": 20}),
    ]

    readiness = _partner_readiness_score(results)

    details = readiness["dimension_scores"]["behavioral_qualification"]["details"]
    assert details["duplicate_tool_call_sample_sufficient"] is False
    assert readiness["dimension_scores"]["behavioral_qualification"]["status"] == "pass"


def test_partner_readiness_penalizes_tom_question_echo_when_tom_is_present() -> None:
    results = [
        _check("health"),
        _check("context_summary", details={"provider": "openai-codex", "model": "openai/gpt-5.5", "auth_source": "profile:openai-codex:chatgpt-plus"}),
        _check("chat_sessions", details={"session_count": 2}),
        _check("schedule_agenda", details={"counts": {"active": 2}, "auth_failures": [], "recent_run_count": 3, "latest_run_status": "recorded"}),
        _check("memory_stats", details={"memory_count": 100, "edge_count": 200}),
        _check("memory_activity", required=False, details={"event_count": 20}),
        _check("cognitive_flow", details={"event_count": 80, "edge_count": 120, "temporal_missing_node_count": 0, "temporal_missing_edge_count": 0, "readable_thought_count": 12}),
        _check("daydream_summary", details={"recent_associations": 8, "recent_run_count": 20, "useful_recent_run_count": 5, "consecutive_skip_count": 3}),
        _check("tom_summary", details={"belief_total": 20, "contradiction_count": 0, "recent_belief_count": 4, "recent_question_echo_count": 4}),
        _check("runtime_truth", details={"tool_count": 130, "capability_count": 109}),
        _check("cognitive_health", details={"unhealthy_checks": []}),
        _check("persistence", details={"missing_files": [], "missing_tables": [], "empty_required_tables": []}),
        _check("google_workspace_auth", details={}),
        _check("executive_assistant", details={"pa_score": 0.94, "duplicate_tool_call_rate": 0.05, "daydream_consecutive_skip_count": 3, "item_status_counts": {"pass": 5, "pending": 0}}),
        _check("qualification_summary", required=False, details={"agent_success_rate": 0.95, "total_agent_checks": 20}),
    ]

    readiness = _partner_readiness_score(results)

    tom_dimension = readiness["dimension_scores"]["theory_of_mind_and_persistence"]
    assert tom_dimension["status"] == "warn"
    assert tom_dimension["details"]["recent_question_echo_ratio"] == 1.0


def test_live_suite_records_daydream_latency_without_required_failure(monkeypatch, tmp_path) -> None:
    captured_latency: dict[str, float | None] = {}

    def _fake_result_from_http(
        name,
        base_url,
        path,
        *,
        timeout,
        required=True,
        validator=None,
        max_bytes=0,
        max_latency_ms=0,
    ):
        captured_latency[name] = max_latency_ms
        return CheckResult(name=name, ok=True, required=required, details={})

    async def _fake_gws_auth_check(required: bool) -> CheckResult:
        return CheckResult(name="google_workspace_auth", ok=True, required=required, details={})

    monkeypatch.setattr(
        "scripts.opencas_live_capability_suite._result_from_http",
        _fake_result_from_http,
    )
    monkeypatch.setattr(
        "scripts.opencas_live_capability_suite._cognitive_health_check",
        lambda repo, state_dir, timeout: CheckResult(name="cognitive_health", ok=True, details={"unhealthy_checks": []}),
    )
    monkeypatch.setattr("scripts.opencas_live_capability_suite._gws_auth_check", _fake_gws_auth_check)

    asyncio.run(
        run_suite(
            argparse.Namespace(
                repo=str(tmp_path),
                state_dir=str(tmp_path),
                base_url="http://127.0.0.1:32147",
                timeout=1.0,
            )
        )
    )

    assert captured_latency["daydream_summary"] is None
