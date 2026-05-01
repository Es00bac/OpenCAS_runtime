from pathlib import Path

import pytest

from opencas.api import operator_actions as oa
from opencas.api.operator_action_store import OperatorActionRegistryStore
from opencas.api import provenance_entry as pe


def test_load_recent_operator_actions_reads_only_canonical_registry_lines(tmp_path: Path) -> None:
    path = tmp_path / "operator_action_history.jsonl"
    path.write_text(
        "session:default:123 | browser\\|default\\|007 | TOOL_CALL | navigation action | MEDIUM\n",
        encoding="utf-8",
    )

    runtime = type("Runtime", (), {"ctx": type("Ctx", (), {})()})()

    records = oa.load_recent_operator_actions(
        runtime,
        target_kind="browser",
        target_id="007",
        default_path=path,
        scope_key="default",
        limit=10,
    )

    assert len(records) == 1
    assert records[0]["session_id"] == "session:default:123"
    assert records[0]["artifact"] == "browser|default|007"
    assert records[0]["target_kind"] == "browser"
    assert records[0]["target_id"] == "007"
    assert records[0]["scope_key"] == "default"


def test_load_recent_operator_actions_skips_malformed_registry_line_and_keeps_valid_entries(tmp_path: Path) -> None:
    path = tmp_path / "operator_action_history.jsonl"
    path.write_text(
        "session:default:123 | browser\\|default\\|007 | TOOL_CALL | navigation action | MEDIUM\n"
        "session:default:123 | browser:default:007 | TOOL_CALL | why\\x | MEDIUM\n",
        encoding="utf-8",
    )

    runtime = type("Runtime", (), {"ctx": type("Ctx", (), {})()})()

    records = oa.load_recent_operator_actions(
        runtime,
        target_kind="browser",
        target_id="007",
        default_path=path,
        scope_key="default",
        limit=10,
    )

    assert records == [
        {
            "session_id": "session:default:123",
            "artifact": "browser|default|007",
            "action": "TOOL_CALL",
            "why": "navigation action",
            "risk": "MEDIUM",
            "target_kind": "browser",
            "target_id": "007",
            "scope_key": "default",
        }
    ]


def test_append_operator_action_persists_canonical_record_without_noisy_metadata(tmp_path: Path) -> None:
    path = tmp_path / "operator_action_history.jsonl"
    runtime = type(
        "Runtime",
        (),
        {
            "ctx": type(
                "Ctx",
                (),
                {
                    "operator_action_sink": path,
                },
            )()
        },
    )()

    projected = oa.append_operator_action(
        runtime,
        {
            "action": "browser_navigate",
            "target_kind": "browser",
            "target_id": "browser-001",
            "scope_key": "test-browser",
            "why": "https://example.com/next",
            "ts": "2026-04-18T09:30:00+00:00",
            "event_id": "evt-123",
            "ok": True,
            "selector": "#nav",
        },
        default_path=path,
    )

    assert path.read_text(encoding="utf-8") == (
        "browser\\|test-browser\\|browser-001 | browser\\|test-browser\\|browser-001 | TOOL_CALL | "
        "https://example.com/next | MEDIUM\n"
    )
    assert projected == {
        "session_id": "browser|test-browser|browser-001",
        "artifact": "browser|test-browser|browser-001",
        "action": "TOOL_CALL",
        "why": "https://example.com/next",
        "risk": "MEDIUM",
        "target_kind": "browser",
        "target_id": "browser-001",
        "scope_key": "test-browser",
    }
    assert "event_id" not in projected
    assert "ok" not in projected
    assert "selector" not in projected

    loaded = oa.load_recent_operator_actions(
        runtime,
        target_kind="browser",
        target_id="browser-001",
        default_path=path,
        scope_key="test-browser",
        limit=10,
    )
    assert loaded == [
        {
            "session_id": "browser|test-browser|browser-001",
            "artifact": "browser|test-browser|browser-001",
            "action": "TOOL_CALL",
            "why": "https://example.com/next",
            "risk": "MEDIUM",
            "target_kind": "browser",
            "target_id": "browser-001",
            "scope_key": "test-browser",
        }
    ]


class _RegistryStore:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def append(self, line: str) -> None:
        self.lines.append(line)

    def list_recent(self, limit: int = 10, offset: int = 0) -> list[pe.ProvenanceRecordV1]:
        window = self.lines[offset : offset + limit]
        return [pe.parse_registry_entry(line) for line in window]


def test_operator_actions_registry_store_recovers_noisy_legacy_log_and_preserves_restart_order(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "operator_action_history.db"
    legacy_path = db_path.with_suffix(".jsonl")

    first = pe.create_registry_entry(
        session_id="session:default:111",
        artifact="browser|review|001",
        action=pe.Action.TOOL_CALL,
        why="first durable entry",
        risk=pe.Risk.MEDIUM,
    )
    second = pe.create_registry_entry(
        session_id="session:default:222",
        artifact="browser|review|002",
        action=pe.Action.UPDATE,
        why="second durable entry",
        risk=pe.Risk.LOW,
    )

    legacy_path.write_text(
        "\n".join(
            [
                "pytest session output:",
                pe.format_provenance_entry(first),
                "build noise: one failed test and a warning",
                pe.format_provenance_entry(second),
                "1 passed, 1 failed in 0.01s",
            ]
        ),
        encoding="utf-8",
    )

    store = OperatorActionRegistryStore(db_path)
    assert store.list_recent(limit=10, offset=0) == [first, second]

    legacy_path.write_text(
        legacy_path.read_text(encoding="utf-8") + "\n" + "\n".join(
            [
                "compiler noise after restart",
                "session:default:999 | browser\\|review\\|noise | UPDATE | should be ignored | LOW",
            ]
        ),
        encoding="utf-8",
    )

    third = pe.create_registry_entry(
        session_id="session:default:333",
        artifact="browser|review|003",
        action=pe.Action.READ,
        why="third durable entry",
        risk=pe.Risk.MEDIUM,
    )
    store.append(pe.format_provenance_entry(third))

    reopened_store = OperatorActionRegistryStore(db_path)
    assert reopened_store.list_recent(limit=10, offset=0) == [first, second, third]


def test_operator_actions_round_trip_through_registry_store_without_optional_noise() -> None:
    store = _RegistryStore()
    runtime = type(
        "Runtime",
        (),
        {
            "ctx": type("Ctx", (), {"registry_store": store})(),
        },
    )()

    projected = oa.append_operator_action(
        runtime,
        {
            "action": "browser_navigate",
            "target_kind": "browser",
            "target_id": "browser-002",
            "scope_key": "review",
            "why": "https://example.com/next",
            "ts": "2026-04-18T09:30:00+00:00",
            "actor": "runtime",
            "source_trace": {"event_id": "evt-123", "origin": "test"},
            "event_id": "evt-123",
            "ok": True,
        },
        default_path=Path("/tmp/unused-operator-action-history.jsonl"),
    )

    assert store.lines == [
        "browser\\|review\\|browser-002 | browser\\|review\\|browser-002 | TOOL_CALL | "
        "https://example.com/next | MEDIUM"
    ]
    assert projected == {
        "session_id": "browser|review|browser-002",
        "artifact": "browser|review|browser-002",
        "action": "TOOL_CALL",
        "why": "https://example.com/next",
        "risk": "MEDIUM",
        "target_kind": "browser",
        "target_id": "browser-002",
        "scope_key": "review",
    }

    loaded = oa.load_recent_operator_actions(
        runtime,
        target_kind="browser",
        target_id="browser-002",
        default_path=Path("/tmp/unused-operator-action-history.jsonl"),
        scope_key="review",
        limit=10,
    )

    assert loaded == [projected]
    assert "ts" not in loaded[0]
    assert "actor" not in loaded[0]
    assert "source_trace" not in loaded[0]
    assert "event_id" not in loaded[0]
