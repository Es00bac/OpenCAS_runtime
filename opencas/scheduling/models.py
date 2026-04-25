"""Scheduling data models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class ScheduleKind(str, Enum):
    TASK = "task"
    EVENT = "event"


class ScheduleAction(str, Enum):
    SUBMIT_BAA = "submit_baa"
    REMINDER_ONLY = "reminder_only"


class ScheduleStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ScheduleRecurrence(str, Enum):
    NONE = "none"
    INTERVAL_HOURS = "interval_hours"
    DAILY = "daily"
    WEEKLY = "weekly"
    WEEKDAYS = "weekdays"


class ScheduleRunStatus(str, Enum):
    SUBMITTED = "submitted"
    RECORDED = "recorded"
    SKIPPED = "skipped"
    FAILED = "failed"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class ScheduleItem(BaseModel):
    schedule_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    kind: ScheduleKind
    action: ScheduleAction
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    title: str
    description: str = ""
    objective: Optional[str] = None
    start_at: datetime
    end_at: Optional[datetime] = None
    timezone: str = "America/Denver"
    next_run_at: Optional[datetime] = None
    last_run_at: Optional[datetime] = None
    recurrence: ScheduleRecurrence = ScheduleRecurrence.NONE
    interval_hours: Optional[float] = None
    weekdays: List[int] = Field(default_factory=list)
    max_occurrences: Optional[int] = None
    occurrence_count: int = 0
    priority: float = Field(default=5.0, ge=1.0, le=10.0)
    tags: List[str] = Field(default_factory=list)
    commitment_id: Optional[str] = None
    plan_id: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at", "updated_at", "start_at", "end_at", "next_run_at", "last_run_at")
    @classmethod
    def _normalize_dt(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        return ensure_aware_utc(value)

    @field_validator("weekdays")
    @classmethod
    def _normalize_weekdays(cls, value: List[int]) -> List[int]:
        return sorted({int(day) for day in value if 0 <= int(day) <= 6})

    @model_validator(mode="after")
    def _validate_schedule(self) -> "ScheduleItem":
        if not self.title.strip():
            raise ValueError("title is required")
        if self.action == ScheduleAction.SUBMIT_BAA and not (self.objective or "").strip():
            raise ValueError("objective is required for submit_baa schedules")
        if self.recurrence == ScheduleRecurrence.INTERVAL_HOURS:
            if self.interval_hours is None or self.interval_hours <= 0:
                raise ValueError("interval_hours must be > 0 for interval recurrence")
        if self.recurrence == ScheduleRecurrence.WEEKLY and not self.weekdays:
            raise ValueError("weekly recurrence requires at least one weekday")
        if self.end_at is not None and self.end_at < self.start_at:
            raise ValueError("end_at cannot be before start_at")
        if self.next_run_at is None and self.status == ScheduleStatus.ACTIVE:
            self.next_run_at = self.start_at
        return self


class ScheduleRun(BaseModel):
    run_id: UUID = Field(default_factory=uuid4)
    schedule_id: UUID
    scheduled_for: datetime
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: Optional[datetime] = None
    status: ScheduleRunStatus
    task_id: Optional[str] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("scheduled_for", "started_at", "finished_at")
    @classmethod
    def _normalize_dt(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        return ensure_aware_utc(value)
