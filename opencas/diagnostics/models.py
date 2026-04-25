"""Diagnostic data models for OpenCAS."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class CheckStatus(str, Enum):
    """Status of a single diagnostic check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class DiagnosticCheck(BaseModel):
    """Result of one diagnostic check."""

    check_id: UUID = Field(default_factory=uuid4)
    name: str
    status: CheckStatus
    message: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthReport(BaseModel):
    """Aggregate health report from the doctor."""

    report_id: UUID = Field(default_factory=uuid4)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    overall: CheckStatus = CheckStatus.PASS
    checks: List[DiagnosticCheck] = Field(default_factory=list)

    def add(self, check: DiagnosticCheck) -> None:
        self.checks.append(check)
        if check.status == CheckStatus.FAIL:
            self.overall = CheckStatus.FAIL
        elif check.status == CheckStatus.WARN and self.overall == CheckStatus.PASS:
            self.overall = CheckStatus.WARN
