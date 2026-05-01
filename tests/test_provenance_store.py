from __future__ import annotations

from pathlib import Path

import pytest

from opencas.api import provenance_store as ps


def _record() -> ps.ProvenanceRecord:
    source = ps.SourceReference(source_id="src-1", kind=ps.SourceKind.EVENT, label="runtime event")
    return ps.ProvenanceRecord(
        checked_items=(
            ps.CheckedItem(
                item_id="workspace/notes.md",
                status=ps.CheckedItemStatus.PASS,
                source_ids=("src-1",),
                checked_at="2026-04-18T12:00:00+00:00",
                label="notes.md",
                notes="checked for trust-critical consistency",
            ),
        ),
        changes=(
            ps.ChangeRecord(
                change_id="chg-1",
                kind=ps.ChangeKind.UPDATE,
                target="workspace/notes.md",
                source_ids=("src-1",),
                changed_at="2026-04-18T12:00:00+00:00",
                summary="normalized trust-critical record",
                after="verified",
            ),
        ),
        pending_work=(),
        actor_identity=ps.ActorIdentity(
            actor_id="runtime",
            kind=ps.ActorKind.SYSTEM,
            display_name="OpenCAS runtime",
            session_id="session:default:123",
        ),
        timestamps=ps.TimestampBundle(
            recorded_at="2026-04-18T12:00:00+00:00",
            checked_at="2026-04-18T12:00:00+00:00",
            verified_at="2026-04-18T12:00:01+00:00",
            updated_at="2026-04-18T12:00:02+00:00",
        ),
        sources=(source,),
        verification_status=ps.VerificationStatus.VERIFIED,
    )


def _transition(
    *,
    session_id: str,
    entity_id: str,
    kind: ps.ProvenanceTransitionKind,
    status: str,
    recorded_at: str,
    details: dict[str, object] | None = None,
) -> ps.ProvenanceTransitionRecord:
    return ps.ProvenanceTransitionRecord(
        transition_id=f"{session_id}:{entity_id}:{kind.value}:{recorded_at}",
        session_id=session_id,
        entity_id=entity_id,
        kind=kind,
        status=status,
        recorded_at=recorded_at,
        details=details or {},
    )


def test_format_and_parse_provenance_entry_round_trip() -> None:
    record = _record()

    line = ps.format_provenance_entry(record)
    parsed = ps.parse_provenance_entry(line)

    assert parsed == record
    assert ps.format_provenance_entry(parsed) == line
    assert ps.provenance_record_to_dict(parsed)["verification_status"] == "VERIFIED"


def test_parse_provenance_entry_rejects_malformed_input() -> None:
    with pytest.raises(ps.ProvenanceParseError):
        ps.parse_provenance_entry("not-json")

    with pytest.raises(ps.ProvenanceParseError):
        ps.parse_provenance_entry('{"v":"9","checked_items":[],"changes":[],"pending_work":[]}')


def test_format_provenance_entry_rejects_non_mapping_input() -> None:
    with pytest.raises(ps.ProvenanceValidationError, match="mapping or ProvenanceRecord"):
        ps.format_provenance_entry("not-a-mapping")


def test_provenance_entry_store_appends_and_reads_in_order(tmp_path: Path) -> None:
    path = tmp_path / "provenance.jsonl"
    store = ps.ProvenanceEntryStore(path)

    first = _record()
    second = ps.ProvenanceRecord(
        checked_items=(),
        changes=(),
        pending_work=(
            ps.PendingWorkItem(
                work_id="work-1",
                status=ps.PendingWorkStatus.OPEN,
                source_ids=("src-2",),
                summary="follow up on the reviewed note",
                owner="runtime",
            ),
        ),
        actor_identity=ps.ActorIdentity(
            actor_id="reviewer",
            kind=ps.ActorKind.REVIEWER,
            display_name="Human reviewer",
            session_id="session:default:456",
        ),
        timestamps=ps.TimestampBundle(
            recorded_at="2026-04-18T12:01:00+00:00",
            checked_at="2026-04-18T12:01:00+00:00",
        ),
        sources=(ps.SourceReference(source_id="src-2", kind=ps.SourceKind.FILE, uri="file:///tmp/review.md"),),
        verification_status=ps.VerificationStatus.CHECKED,
    )

    store.append(first)
    store.append(second)

    assert store.list_recent() == [first, second]
    assert store.list_recent(limit=1) == [first]
    assert path.read_text(encoding="utf-8").splitlines() == [
        ps.format_provenance_entry(first),
        ps.format_provenance_entry(second),
    ]


def test_provenance_entry_store_example_workflow_writes_then_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "provenance.jsonl"
    store = ps.ProvenanceEntryStore(path)
    record = _record()

    written = store.append(record)
    reopened_store = ps.ProvenanceEntryStore(path)
    reviewed = reopened_store.list_recent(limit=1)[0]

    assert written == record
    assert reviewed == record
    assert reopened_store.list_recent(limit=1) == [record]
    assert path.read_text(encoding="utf-8").splitlines() == [ps.format_provenance_entry(record)]


def test_provenance_entry_store_ignores_other_streams(tmp_path: Path) -> None:
    store = ps.ProvenanceEntryStore(tmp_path / "provenance.jsonl")
    store.append(_record())

    unrelated = tmp_path / "build-and-test.log"
    unrelated.write_text(
        "\n".join(
            [
                "pytest session output:",
                "======================",
                '{"v":"2","checked_items":[],"changes":[],"pending_work":[],"actor_identity":{"actor_id":"x","kind":"SYSTEM"},"timestamps":{"recorded_at":"2026-04-18T12:00:00+00:00"},"sources":[],"verification_status":"PENDING"}',
                "1 passed in 0.01s",
            ]
        ),
        encoding="utf-8",
    )

    assert store.list_recent() == [_record()]
    assert unrelated.read_text(encoding="utf-8").count('"verification_status":"PENDING"') == 1


def test_transition_history_appends_in_order_and_replays_after_restart(tmp_path: Path) -> None:
    store = ps.ProvenanceEntryStore(tmp_path / "provenance.jsonl")

    first = _transition(
        session_id="session-1",
        entity_id="entity-a",
        kind=ps.ProvenanceTransitionKind.CHECK,
        status="checked",
        recorded_at="2026-04-18T12:00:00+00:00",
        details={"result": "pass"},
    )
    second = _transition(
        session_id="session-1",
        entity_id="entity-a",
        kind=ps.ProvenanceTransitionKind.MUTATION,
        status="mutated",
        recorded_at="2026-04-18T12:00:01+00:00",
        details={"field": "summary"},
    )
    third = _transition(
        session_id="session-1",
        entity_id="entity-b",
        kind=ps.ProvenanceTransitionKind.WAITING,
        status="waiting",
        recorded_at="2026-04-18T12:00:02+00:00",
        details={"reason": "approval"},
    )

    store.record_transition(first)
    store.record_transition(second)
    store.record_transition(third)

    reopened_store = ps.ProvenanceEntryStore(tmp_path / "provenance.jsonl")

    assert reopened_store.list_transition_history() == [first, second, third]
    assert reopened_store.list_transition_history(limit=2) == [first, second]
    assert reopened_store.list_transition_history(limit=1, offset=1) == [second]


def test_transition_history_is_immutable_across_appends_and_queries(tmp_path: Path) -> None:
    store = ps.ProvenanceEntryStore(tmp_path / "provenance.jsonl")
    first = _transition(
        session_id="session-2",
        entity_id="entity-c",
        kind=ps.ProvenanceTransitionKind.CHECK,
        status="checked",
        recorded_at="2026-04-18T12:10:00+00:00",
    )
    second = _transition(
        session_id="session-2",
        entity_id="entity-c",
        kind=ps.ProvenanceTransitionKind.WAITING,
        status="waiting",
        recorded_at="2026-04-18T12:11:00+00:00",
        details={"reason": "queued"},
    )

    store.record_transition(first)
    before = (tmp_path / "provenance.transitions.jsonl").read_text(encoding="utf-8")

    store.record_transition(second)
    store.list_current_status()
    after = (tmp_path / "provenance.transitions.jsonl").read_text(encoding="utf-8")

    assert before.splitlines() == [ps.format_provenance_transition(first)]
    assert after.splitlines() == [
        ps.format_provenance_transition(first),
        ps.format_provenance_transition(second),
    ]


def test_current_status_uses_latest_effective_state_after_mixed_histories(tmp_path: Path) -> None:
    store = ps.ProvenanceEntryStore(tmp_path / "provenance.jsonl")
    history = [
        _transition(
            session_id="session-3",
            entity_id="entity-a",
            kind=ps.ProvenanceTransitionKind.CHECK,
            status="checked",
            recorded_at="2026-04-18T12:20:00+00:00",
            details={"result": "pass"},
        ),
        _transition(
            session_id="session-3",
            entity_id="entity-b",
            kind=ps.ProvenanceTransitionKind.MUTATION,
            status="mutated",
            recorded_at="2026-04-18T12:20:01+00:00",
            details={"field": "title"},
        ),
        _transition(
            session_id="session-3",
            entity_id="entity-a",
            kind=ps.ProvenanceTransitionKind.WAITING,
            status="waiting",
            recorded_at="2026-04-18T12:20:02+00:00",
            details={"reason": "human review"},
        ),
        _transition(
            session_id="session-3",
            entity_id="entity-b",
            kind=ps.ProvenanceTransitionKind.CHECK,
            status="checked",
            recorded_at="2026-04-18T12:20:03+00:00",
            details={"result": "pass"},
        ),
    ]

    for item in history:
        store.record_transition(item)

    current = store.list_current_status()

    assert current == [history[3], history[2]]
    assert store.get_current_status(session_id="session-3", entity_id="entity-a") == history[2]
    assert store.get_current_status(session_id="session-3", entity_id="entity-b") == history[3]
    assert store.get_current_status(session_id="session-3", entity_id="missing") is None


@pytest.mark.parametrize(
    ("method_name", "kind", "status", "origin_action_id"),
    [
        ("record_check", ps.ProvenanceTransitionKind.CHECK, "checked", "check-001"),
        ("record_mutation", ps.ProvenanceTransitionKind.MUTATION, "mutated", "mutation-001"),
        ("record_waiting", ps.ProvenanceTransitionKind.WAITING, "blocked", "waiting-001"),
    ],
)
def test_transition_helpers_include_linkage_fields(
    tmp_path: Path,
    method_name: str,
    kind: ps.ProvenanceTransitionKind,
    status: str,
    origin_action_id: str,
) -> None:
    store = ps.ProvenanceEntryStore(tmp_path / "provenance.jsonl")
    method = getattr(store, method_name)

    record = method(
        session_id="session-1",
        entity_id="entity-1",
        status=status,
        details={"note": "linked"},
        recorded_at="2026-04-18T12:20:00+00:00",
        source_artifact="source|artifact|1",
        trigger_action="workflow.check",
        target_entity="entity-1",
        origin_action_id=origin_action_id,
    )

    assert record.kind == kind
    assert record.status == status
    assert record.entity_id == "entity-1"
    assert record.details["source_artifact"] == "source|artifact|1"
    assert record.details["trigger_artifact"] == "source|artifact|1"
    assert record.details["trigger_action"] == "workflow.check"
    assert record.details["target_entity"] == "entity-1"
    assert record.details["parent_transition_id"] == origin_action_id
    assert record.details["linked_transition_ids"] == [origin_action_id, "entity-1"]
    assert record.details["origin_action_id"] == origin_action_id
    assert record.transition_id == origin_action_id
