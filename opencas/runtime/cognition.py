"""Per-turn cognition frame for OpenCAS runtime phases.

Fields and intended consumers:
- `audit_only`: populated at turn entry from audit markers; extractors and
  persistence helpers use it to suppress or route audit-only state.
- `somatic_state`: populated from the current somatic manager when available;
  affective workstreams can compare pre/post turn movement.
- `relational_snapshot`: reserved for musubi/trust readouts consumed by approval
  and affective routing once their reliability is restored.
- `wellbeing_readout`: reserved for wellbeing assessments consumed by the turn
  loop and creative ladder health checks.
- `authorization_context`: reserved for standing/session authorization evidence
  consumed by self-approval and refusal gates.

This is a frame, not a runtime god object: it carries already-owned state through
the turn so producers and consumers share the same context boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

from .audit_mode import is_audit_only_text, with_audit_only_meta


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _default_conversation_actor_meta() -> dict[str, Any]:
    return {
        "type": "operator",
        "label": "operator",
        "source": "runtime_default",
        "is_operator": True,
        "memory_scope": "operator_turn",
        "remember_as": "owner/operator",
    }


@dataclass(slots=True)
class CognitionFrame:
    frame_id: str
    session_id: str
    created_at: datetime
    audit_only: bool
    user_input_preview: str
    user_meta: dict[str, Any]
    somatic_state: Any = None
    relational_snapshot: Mapping[str, Any] | None = None
    wellbeing_readout: Mapping[str, Any] | None = None
    authorization_context: Mapping[str, Any] | None = None
    working_memory: Mapping[str, Any] = field(default_factory=dict)
    tom_context: Mapping[str, Any] = field(default_factory=dict)
    identity_context: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_turn(
        cls,
        *,
        session_id: str,
        user_input: str,
        user_meta: Mapping[str, Any] | None = None,
        runtime: Any = None,
        audit_only: bool | None = None,
    ) -> "CognitionFrame":
        resolved_audit_only = (
            is_audit_only_text(user_input, user_meta)
            if audit_only is None
            else bool(audit_only)
        )
        frame_id = str(uuid4())
        meta = with_audit_only_meta(user_meta, audit_only=resolved_audit_only)
        meta["cognition_frame_id"] = frame_id
        if not isinstance(meta.get("conversation_actor"), dict):
            meta["conversation_actor"] = _default_conversation_actor_meta()

        ctx = getattr(runtime, "ctx", None)
        somatic = getattr(ctx, "somatic", None)
        relational = getattr(ctx, "relational", None)
        wellbeing = getattr(ctx, "wellbeing", None)

        return cls(
            frame_id=frame_id,
            session_id=session_id,
            created_at=_utc_now(),
            audit_only=resolved_audit_only,
            user_input_preview=user_input[:240],
            user_meta=meta,
            somatic_state=getattr(somatic, "state", None),
            relational_snapshot=_relational_snapshot(relational),
            wellbeing_readout=_wellbeing_readout(wellbeing),
        )


def _relational_snapshot(relational: Any) -> Mapping[str, Any] | None:
    if relational is None:
        return None
    snapshot: dict[str, Any] = {}
    for attr in ("musubi", "trust", "resonance", "presence", "attunement"):
        if hasattr(relational, attr):
            snapshot[attr] = getattr(relational, attr)
    return snapshot or None


def _wellbeing_readout(wellbeing: Any) -> Mapping[str, Any] | None:
    if wellbeing is None:
        return None
    readout: dict[str, Any] = {}
    for attr in ("drift_load", "coherence", "recovery_need", "overall_risk"):
        if hasattr(wellbeing, attr):
            readout[attr] = getattr(wellbeing, attr)
    return readout or None
