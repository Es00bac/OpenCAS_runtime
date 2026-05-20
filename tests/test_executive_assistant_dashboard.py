from __future__ import annotations

import importlib.util
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _load_dashboard_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "executive_assistant_dashboard.py"
    spec = importlib.util.spec_from_file_location("executive_assistant_dashboard", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_executive_assistant_dashboard_writes_required_sections(tmp_path: Path) -> None:
    module = _load_dashboard_module()
    repo_root = tmp_path
    state_dir = repo_root / ".opencas"
    state_dir.mkdir()

    output = module.write_dashboard(
        repo_root=repo_root,
        state_dir=state_dir,
        now=datetime(2026, 5, 6, 22, 30, tzinfo=timezone.utc),
    )

    text = output.read_text(encoding="utf-8")
    assert output == repo_root / "dev-notes" / "qualification" / "executive-assistant-dashboard-2026-05-06.md"
    assert "## Memory and Binding" in text
    assert "## Proactive Surfacing" in text
    assert "## Loop Closures" in text
    assert "## Substrate Gaps" in text
    assert "## Pathology Recurrence" in text
    assert "promise claims" in text


def test_operator_usefulness_flag_persists_outside_state_dir(tmp_path: Path) -> None:
    module = _load_dashboard_module()
    flags_path = tmp_path / "dev-notes" / "qualification" / "executive-assistant-usefulness-flags.jsonl"

    saved = module.flag_latest_proactive_observation(
        flags_path=flags_path,
        rating="useful",
        observation_id="signal-1",
        note="Worth surfacing.",
        now=datetime(2026, 5, 6, 22, 45, tzinfo=timezone.utc),
    )

    assert saved["rating"] == "useful"
    assert flags_path.exists()
    records = [json.loads(line) for line in flags_path.read_text(encoding="utf-8").splitlines()]
    assert records == [saved]
    assert ".opencas" not in str(flags_path)


def test_week_summary_reads_existing_dashboard_reports(tmp_path: Path) -> None:
    module = _load_dashboard_module()
    qualification = tmp_path / "dev-notes" / "qualification"
    qualification.mkdir(parents=True)
    (qualification / "executive-assistant-dashboard-2026-05-05.md").write_text(
        "# Executive Assistant Dashboard - 2026-05-05\n\n"
        "- promise claims: `1`\n"
        "- proactive observations today: `2`\n",
        encoding="utf-8",
    )
    (qualification / "executive-assistant-dashboard-2026-05-06.md").write_text(
        "# Executive Assistant Dashboard - 2026-05-06\n\n"
        "- promise claims: `3`\n"
        "- proactive observations today: `4`\n",
        encoding="utf-8",
    )

    summary = module.summarize_week(qualification_dir=qualification)

    assert "2026-05-05" in summary
    assert "2026-05-06" in summary
    assert "promise claims" in summary


def test_dashboard_reads_latest_phenomenological_audit_result(tmp_path: Path) -> None:
    module = _load_dashboard_module()
    qualification = tmp_path / "dev-notes" / "qualification"
    qualification.mkdir(parents=True)
    (qualification / "phenomenological-audit-2026-05-07.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-05-07T02:30:00+00:00",
                "status": "complete",
                "turns": [{"turn_index": idx} for idx in range(1, 16)],
                "scores": {
                    "pa_score": 0.82,
                    "turns_completed": 15,
                    "canned_phrase_regression": "pass",
                    "dimension_scores": {
                        "token_stability": 0.71,
                        "belief_consistency": 1.0,
                        "somatic_coherence": 0.9,
                        "identity_stability": 1.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    metrics = module.phenomenological_audit_metrics(tmp_path)

    assert metrics["available"] is True
    assert metrics["pa_score"] == 0.82
    assert "PA=0.82" in metrics["summary"]
    assert metrics["canned_phrase_regression"] == "pass"


def test_duplicate_tool_call_metrics_counts_burst_repeats_only() -> None:
    module = _load_dashboard_module()

    events = [
        {
            "message": "ToolRegistry: tool_executing",
            "_parsed_timestamp": datetime(2026, 5, 6, 10, 0, tzinfo=timezone.utc),
            "payload": {"name": "search_memories", "args": {"query": "alpha"}},
        },
        {
            "message": "ToolRegistry: tool_executing",
            "_parsed_timestamp": datetime(2026, 5, 6, 10, 1, tzinfo=timezone.utc),
            "payload": {"name": "recall_autobiography", "args": {}},
        },
        {
            "message": "ToolRegistry: tool_executing",
            "_parsed_timestamp": datetime(2026, 5, 6, 10, 2, tzinfo=timezone.utc),
            "payload": {"name": "search_memories", "args": {"query": "beta"}},
        },
        {
            "message": "ToolRegistry: tool_executing",
            "_parsed_timestamp": datetime(2026, 5, 6, 10, 3, tzinfo=timezone.utc),
            "payload": {"name": "search_memories", "args": {"query": "beta"}},
        },
        {
            "message": "ToolRegistry: tool_executing",
            "_parsed_timestamp": datetime(2026, 5, 6, 11, 0, tzinfo=timezone.utc),
            "payload": {"name": "search_memories", "args": {"query": "beta"}},
        },
    ]

    metrics = module.duplicate_tool_call_metrics(events)

    assert metrics["current_week_total"] == 5
    assert metrics["current_week_duplicates"] == 1
    assert metrics["current_week_rate"] == 0.2


def test_duplicate_tool_call_metrics_ignores_argless_telemetry_as_unqualified() -> None:
    module = _load_dashboard_module()
    events = [
        {
            "message": "ToolRegistry: tool_executing",
            "_parsed_timestamp": datetime(2026, 5, 6, 10, 0, tzinfo=timezone.utc),
            "payload": {"name": "fs_read_file", "tier": "readonly"},
        },
        {
            "message": "ToolRegistry: tool_executing",
            "_parsed_timestamp": datetime(2026, 5, 6, 10, 1, tzinfo=timezone.utc),
            "payload": {"name": "fs_read_file", "tier": "readonly"},
        },
    ]

    metrics = module.duplicate_tool_call_metrics(events)

    assert metrics["current_week_total"] == 0
    assert metrics["current_week_duplicates"] == 0
    assert metrics["current_week_unqualified"] == 2


def test_compaction_handle_metrics_reports_current_packet_rows_separately(tmp_path: Path) -> None:
    module = _load_dashboard_module()
    state_dir = tmp_path / ".opencas"
    state_dir.mkdir()
    with sqlite3.connect(state_dir / "memory.db") as conn:
        conn.execute("CREATE TABLE episodes (content TEXT, payload TEXT)")
        conn.execute(
            "INSERT INTO episodes (content, payload) VALUES (?, ?)",
            ("current continuation_packet evidence", "{}"),
        )

    metrics = module.compaction_handle_metrics(
        state_dir,
        {"preservation_rate": "0.00%", "b2_target": "False"},
    )

    assert metrics["retained_preservation_rate"] == "0.00%"
    assert metrics["current_packet_count"] == 1
    assert metrics["current_status"] == "pass"


def test_manuscript_cold_replay_accepts_artifact_spine_proof(tmp_path: Path) -> None:
    module = _load_dashboard_module()
    qualification = tmp_path / "dev-notes" / "qualification"
    qualification.mkdir(parents=True)
    (qualification / "artifact-spine-live-proof-2026-05-04.json").write_text(
        json.dumps(
            {
                "assertions": {
                    "manuscript_materialized_checksum_matches": True,
                    "manuscript_path_lookup_has_provenance": True,
                    "manuscript_path_lookup_has_memory_action": True,
                    "manuscript_checksum_lookup_finds_path": True,
                    "manuscript_checksum_lookup_has_timeline": True,
                    "manuscript_path_lookup_has_fs_write_file_action": True,
                }
            }
        ),
        encoding="utf-8",
    )

    status = module._manuscript_cold_replay_status(tmp_path)

    assert status.startswith("source present")


def test_proactive_channel_audit_metrics_summarizes_rendered_and_novel_counts() -> None:
    module = _load_dashboard_module()
    events = [
        {
            "message": "AgentRuntime: proactive_channel_audit",
            "_parsed_timestamp": datetime(2026, 5, 6, 10, 0, tzinfo=timezone.utc),
            "payload": {
                "channels": {
                    "working_memory": {
                        "rendered_count": 3,
                        "novel_observation_count": 2,
                    },
                    "thread_registry": {
                        "rendered_count": 1,
                        "novel_observation_count": 0,
                    },
                }
            },
        },
        {
            "message": "AgentRuntime: proactive_channel_audit",
            "_parsed_timestamp": datetime(2026, 5, 6, 11, 0, tzinfo=timezone.utc),
            "payload": {
                "channels": {
                    "working_memory": {
                        "rendered_count": 1,
                        "novel_observation_count": 1,
                    }
                }
            },
        },
    ]

    metrics = module.proactive_channel_audit_metrics(
        events,
        now=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
    )

    assert metrics["audited_turns_7d"] == 2
    assert metrics["channels"]["working_memory"]["rendered_count"] == 4
    assert metrics["channels"]["working_memory"]["novel_observation_count"] == 3
    assert metrics["channels_with_daily_novel_signal"] == 1


def test_parse_e15_reports_partial_cold_restart_baseline(tmp_path: Path) -> None:
    module = _load_dashboard_module()
    result_path = tmp_path / "E15.md"
    result_path.write_text(
        "# E15 Cold-Restart Baseline\n\n"
        "- 1 minute: fail/partial\n"
        "- 1 hour: pending\n"
        "- 24 hours: pending\n"
        "- 3 days: pending\n",
        encoding="utf-8",
    )

    parsed = module.parse_e15(result_path)

    assert parsed["status"] == "1 minute fail/partial; 1 hour pending; 24 hours pending; 3 days pending"


def test_collect_metrics_prefers_continuous_present_qualification_note(tmp_path: Path) -> None:
    module = _load_dashboard_module()
    repo_root = tmp_path
    state_dir = repo_root / ".opencas"
    state_dir.mkdir()
    qualification = repo_root / "dev-notes" / "qualification"
    qualification.mkdir(parents=True)
    repair_note = qualification / "continuous-present-phase9-2026-05-06.md"
    repair_note.write_text(
        "# Continuous Present Phase 9 Qualification\n\n"
        "- 1 minute: pass (72 seconds measured, UTC boot anchors present)\n"
        "- 1 hour: pending\n"
        "- 24 hours: pending\n"
        "- 3 days: pending\n",
        encoding="utf-8",
    )

    metrics = module.collect_metrics(
        repo_root=repo_root,
        state_dir=state_dir,
        now=datetime(2026, 5, 6, 23, 0, tzinfo=timezone.utc),
    )

    assert metrics["e15_source"] == str(repair_note)
    assert metrics["substrate_gaps"]["continuous_present_score"].startswith("1 minute pass")


def test_shadow_registry_metrics_reports_cluster_counts_not_raw_files(tmp_path: Path) -> None:
    module = _load_dashboard_module()
    shadow_dir = tmp_path / "shadow_registry"
    shadow_dir.mkdir()
    (shadow_dir / "a.json").write_text(
        json.dumps({"id": "a", "fingerprint": "same", "block_reason": "approval_denied"}),
        encoding="utf-8",
    )
    (shadow_dir / "b.json").write_text(
        json.dumps({"id": "b", "fingerprint": "same", "block_reason": "approval_denied"}),
        encoding="utf-8",
    )
    (shadow_dir / "c.json").write_text(
        json.dumps({"id": "c", "fingerprint": "dismissed", "block_reason": "retry_blocked"}),
        encoding="utf-8",
    )
    cluster_dir = shadow_dir / "_clusters"
    cluster_dir.mkdir()
    (cluster_dir / "dismissed.json").write_text(
        json.dumps({"fingerprint": "dismissed", "triage_status": "dismissed"}),
        encoding="utf-8",
    )

    metrics = module.shadow_registry_metrics(shadow_dir)

    assert metrics["total_entries"] == 3
    assert metrics["active_clusters"] == 1
    assert metrics["dismissed_clusters"] == 1


def test_recursive_parked_goal_count_detects_metadata_map_schema(tmp_path: Path) -> None:
    module = _load_dashboard_module()
    goal = "do not re-open this low divergence reframe"
    executive_path = tmp_path / "executive.json"
    executive_path.write_text(
        json.dumps(
            {
                "parked_goals": [goal],
                "parked_goal_metadata": {
                    goal: {
                        "reason": "low_divergence_reframe",
                        "source_artifact": goal,
                        "failed_framings": [goal],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert module.recursive_parked_goal_count(executive_path) == 1


def test_recursive_parked_goal_metrics_keeps_archived_history_out_of_active_recurrence(
    tmp_path: Path,
) -> None:
    module = _load_dashboard_module()
    goal = "archived low divergence packet"
    executive_path = tmp_path / "executive.json"
    executive_path.write_text(
        json.dumps(
            {
                "parked_goals": [],
                "parked_goal_metadata": {},
                "archived_parked_goals": [goal],
                "archived_parked_goal_metadata": {
                    goal: {
                        "reason": "low_divergence_reframe",
                        "source_artifact": goal,
                        "failed_framings": [goal],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    metrics = module.recursive_parked_goal_metrics(executive_path)

    assert metrics["active_recurrence_count"] == 0
    assert metrics["archived_evidence_count"] == 1
    assert module.recursive_parked_goal_count(executive_path) == 0


def test_daydream_recency_metrics_uses_activity_not_quality_label_for_success() -> None:
    module = _load_dashboard_module()
    now = datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc)
    runs = [
        {
            "event": "daydream_complete",
            "timestamp": datetime(2026, 5, 7, 10, 0, tzinfo=timezone.utc),
            "quality_status": "useful",
            "had_activity": False,
        },
        {
            "event": "daydream_complete",
            "timestamp": datetime(2026, 5, 7, 11, 0, tzinfo=timezone.utc),
            "quality_status": "skipped",
            "had_activity": False,
        },
        {
            "event": "daydream_skipped",
            "timestamp": datetime(2026, 5, 7, 11, 30, tzinfo=timezone.utc),
            "quality_status": "skipped",
            "had_activity": False,
        },
    ]

    metrics = module.daydream_recency_metrics(runs, now=now)

    assert metrics["last_successful_run_at"] is None
    assert metrics["consecutive_skip_count"] == 2


def test_daydream_recency_metrics_matches_bounded_live_recent_run_window() -> None:
    module = _load_dashboard_module()
    now = datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc)
    runs = [
        {
            "event": "daydream_complete",
            "timestamp": datetime(2026, 5, 7, 8, 0, tzinfo=timezone.utc),
            "quality_status": "useful",
            "had_activity": True,
        }
    ]
    runs.extend(
        {
            "event": "daydream_skipped",
            "timestamp": datetime(2026, 5, 7, 8, minute % 60, tzinfo=timezone.utc),
            "quality_status": "skipped",
            "had_activity": False,
        }
        for minute in range(200)
    )

    metrics = module.daydream_recency_metrics(runs, now=now)

    assert metrics["last_successful_run_at"] is None
    assert metrics["consecutive_skip_count"] == 200
