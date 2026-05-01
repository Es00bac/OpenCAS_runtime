from __future__ import annotations

import pytest

from opencas.api import provenance_store as ps


def _base_record(*, status: ps.VerificationStatus = ps.VerificationStatus.CHECKED) -> ps.ProvenanceRecord:
    source = ps.SourceReference(source_id="src-1", kind=ps.SourceKind.FILE, uri="file:///tmp/check.md")
    return ps.ProvenanceRecord(
        checked_items=(
            ps.CheckedItem(
                item_id="check-1",
                status=ps.CheckedItemStatus.PASS,
                source_ids=("src-1",),
                checked_at="2026-04-18T12:00:00+00:00",
            ),
        ),
        changes=(),
        pending_work=(
            ps.PendingWorkItem(
                work_id="work-1",
                status=ps.PendingWorkStatus.OPEN,
                source_ids=("src-1",),
                summary="follow up later",
            ),
        ),
        actor_identity=ps.ActorIdentity(
            actor_id="reviewer",
            kind=ps.ActorKind.REVIEWER,
            session_id="session:default:123",
        ),
        timestamps=ps.TimestampBundle(
            recorded_at="2026-04-18T12:00:00+00:00",
            checked_at="2026-04-18T12:00:01+00:00",
            verified_at="2026-04-18T12:00:02+00:00" if status is ps.VerificationStatus.VERIFIED else None,
        ),
        sources=(source,),
        verification_status=status,
    )


def test_provenance_record_requires_source_traceability() -> None:
    record = _base_record()
    broken = ps.ProvenanceRecord(
        checked_items=(
            ps.CheckedItem(
                item_id="check-1",
                status=ps.CheckedItemStatus.PASS,
                source_ids=("missing-source",),
                checked_at="2026-04-18T12:00:00+00:00",
            ),
        ),
        changes=(),
        pending_work=(),
        actor_identity=record.actor_identity,
        timestamps=record.timestamps,
        sources=record.sources,
        verification_status=record.verification_status,
    )

    with pytest.raises(ps.ProvenanceValidationError, match="unknown source"):
        ps.validate_provenance_record(broken)


def test_verification_status_transitions_are_explicit() -> None:
    assert ps.can_transition_verification_status(ps.VerificationStatus.PENDING, ps.VerificationStatus.CHECKED)
    assert ps.can_transition_verification_status(ps.VerificationStatus.CHECKED, ps.VerificationStatus.VERIFIED)
    assert not ps.can_transition_verification_status(ps.VerificationStatus.VERIFIED, ps.VerificationStatus.PENDING)

    with pytest.raises(ps.ProvenanceValidationError, match="illegal verification status transition"):
        ps.transition_verification_status(ps.VerificationStatus.VERIFIED, ps.VerificationStatus.PENDING)


def test_timestamp_and_actor_consistency() -> None:
    record = _base_record(status=ps.VerificationStatus.VERIFIED)
    assert ps.validate_provenance_record(record) == record

    missing_actor = ps.ProvenanceRecord(
        checked_items=record.checked_items,
        changes=record.changes,
        pending_work=record.pending_work,
        actor_identity=ps.ActorIdentity(actor_id="", kind=ps.ActorKind.SYSTEM),
        timestamps=record.timestamps,
        sources=record.sources,
        verification_status=record.verification_status,
    )
    with pytest.raises(ps.ProvenanceValidationError, match="actor identity"):
        ps.validate_provenance_record(missing_actor)

    missing_checked_at = ps.ProvenanceRecord(
        checked_items=record.checked_items,
        changes=record.changes,
        pending_work=record.pending_work,
        actor_identity=record.actor_identity,
        timestamps=ps.TimestampBundle(
            recorded_at="2026-04-18T12:00:00+00:00",
            checked_at=None,
            verified_at="2026-04-18T12:00:02+00:00",
        ),
        sources=record.sources,
        verification_status=ps.VerificationStatus.VERIFIED,
    )
    with pytest.raises(ps.ProvenanceValidationError, match="checked_at is required"):
        ps.validate_provenance_record(missing_checked_at)


def test_provenance_record_from_mapping_rejects_unknown_fields() -> None:
    payload = {
        "v": "2",
        "checked_items": [],
        "changes": [],
        "pending_work": [],
        "actor_identity": {"actor_id": "runtime", "kind": "SYSTEM"},
        "timestamps": {"recorded_at": "2026-04-18T12:00:00+00:00"},
        "sources": [],
        "verification_status": "PENDING",
        "extra_field": "should not survive parsing",
    }

    with pytest.raises(ps.ProvenanceValidationError, match="unknown provenance fields"):
        ps.ProvenanceRecord.from_mapping(payload)


def test_provenance_record_from_mapping_rejects_null_required_sections() -> None:
    payload = {
        "v": "2",
        "checked_items": None,
        "changes": [],
        "pending_work": [],
        "actor_identity": {"actor_id": "runtime", "kind": "SYSTEM"},
        "timestamps": {"recorded_at": "2026-04-18T12:00:00+00:00"},
        "sources": [],
        "verification_status": "PENDING",
    }

    with pytest.raises(ps.ProvenanceValidationError, match="checked_items must be a list"):
        ps.ProvenanceRecord.from_mapping(payload)


def test_provenance_record_rejects_checked_at_while_pending() -> None:
    record = _base_record(status=ps.VerificationStatus.PENDING)
    pending_with_check_time = ps.ProvenanceRecord(
        checked_items=record.checked_items,
        changes=record.changes,
        pending_work=record.pending_work,
        actor_identity=record.actor_identity,
        timestamps=ps.TimestampBundle(
            recorded_at="2026-04-18T12:00:00+00:00",
            checked_at="2026-04-18T12:00:01+00:00",
        ),
        sources=record.sources,
        verification_status=ps.VerificationStatus.PENDING,
    )

    with pytest.raises(ps.ProvenanceValidationError, match="checked_at cannot be set while verification_status is PENDING"):
        ps.validate_provenance_record(pending_with_check_time)
