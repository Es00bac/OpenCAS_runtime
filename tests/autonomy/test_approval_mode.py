from __future__ import annotations

from opencas.autonomy.mode_utils import normalize_approval_mode
from opencas.autonomy.models import ApprovalMode


def test_approval_mode_includes_autonomous_modes() -> None:
    assert ApprovalMode.DEFAULT.value == "default"
    assert ApprovalMode.AUTO_REVIEW.value == "auto_review"
    assert ApprovalMode.FULLY_AUTONOMOUS.value == "fully_autonomous"
    assert ApprovalMode.TRUST_BASED.value == "trust_based"


def test_normalize_approval_mode_accepts_aliases() -> None:
    assert normalize_approval_mode(None) is ApprovalMode.DEFAULT
    assert normalize_approval_mode("auto-review") is ApprovalMode.AUTO_REVIEW
    assert normalize_approval_mode("trust-based") is ApprovalMode.TRUST_BASED
    assert normalize_approval_mode("YOLO") is ApprovalMode.FULLY_AUTONOMOUS
