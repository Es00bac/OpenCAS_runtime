"""Data models for the affective registry writer."""

from __future__ import annotations

import os
import platform as platform_module
import sys
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ExecutionPhase(str, Enum):
    """Lifecycle phases during which an affective snapshot may be captured."""

    BOOT = "boot"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    TOOL_CALL = "tool_call"
    MEMORY_COMPACT = "memory_compact"
    CONSOLIDATION = "consolidation"
    DAYDREAM = "daydream"
    ERROR_RECOVERY = "error_recovery"
    SHUTDOWN = "shutdown"
    MANUAL = "manual"
    TEST = "test"


class ExecutionContext(BaseModel):
    """Runtime context captured at the moment of registry write."""

    process_id: int = Field(default_factory=os.getpid)
    thread_id: int = Field(default_factory=threading.get_ident)
    thread_name: str = Field(default_factory=lambda: threading.current_thread().name)
    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    platform: str = Field(default_factory=platform_module.platform)
    hostname: str = Field(default_factory=platform_module.node)
    cwd: str = Field(default_factory=os.getcwd)
    exec_path: str = Field(default_factory=lambda: sys.executable)
    # Optional caller-provided tags
    session_id: Optional[str] = None
    span_id: Optional[str] = None
    trace_id: Optional[str] = None
    user_id: Optional[str] = None


class SystemMetrics(BaseModel):
    """Lightweight system metrics captured at write time.

    All fields are best-effort; missing values are represented as None so the
    entry remains valid even when /proc or psutil are unavailable.
    """

    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    memory_rss_mb: Optional[float] = None
    memory_vms_mb: Optional[float] = None
    open_file_descriptors: Optional[int] = None
    thread_count: Optional[int] = None
    uptime_seconds: Optional[float] = None

    @classmethod
    def capture(cls) -> SystemMetrics:
        """Attempt to capture live system metrics; never raise."""
        metrics = cls()
        try:
            import psutil

            proc = psutil.Process()
            with proc.oneshot():
                metrics.cpu_percent = round(proc.cpu_percent(interval=None), 2)
                mem_info = proc.memory_info()
                metrics.memory_rss_mb = round(mem_info.rss / (1024 * 1024), 2)
                metrics.memory_vms_mb = round(mem_info.vms / (1024 * 1024), 2)
                metrics.memory_percent = round(proc.memory_percent(), 2)
                metrics.open_file_descriptors = proc.num_fds()
                metrics.thread_count = proc.num_threads()
                metrics.uptime_seconds = round(
                    time.time() - proc.create_time(), 2
                )
        except Exception:
            # psutil unavailable or permission denied — leave fields as None
            pass
        return metrics


class AffectiveState(BaseModel):
    """Affective dimensions captured at execution moment.

    Mirrors the canonical somatic dimensions used across OpenCAS so that
    downstream affect analysis can consume registry entries directly.
    """

    primary_emotion: str = "neutral"
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.5, ge=0.0, le=1.0)
    fatigue: float = Field(default=0.0, ge=0.0, le=1.0)
    tension: float = Field(default=0.0, ge=0.0, le=1.0)
    focus: float = Field(default=0.5, ge=0.0, le=1.0)
    energy: float = Field(default=0.5, ge=0.0, le=1.0)
    certainty: float = Field(default=0.5, ge=0.0, le=1.0)
    musubi: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    somatic_tag: Optional[str] = None


class AffectiveRegistryEntry(BaseModel):
    """A single structured entry in the affective registry.

    Each entry is an atomic, append-only record that captures the system's
    affective state and execution context at a specific moment in time.
    """

    entry_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"

    phase: ExecutionPhase = ExecutionPhase.MANUAL
    affective_state: AffectiveState = Field(default_factory=AffectiveState)
    execution_context: ExecutionContext = Field(default_factory=ExecutionContext)
    system_metrics: SystemMetrics = Field(default_factory=SystemMetrics.capture)

    # Free-form payload for caller-specific extensions
    payload: Dict[str, Any] = Field(default_factory=dict)

    # Provenance / lineage
    source_module: Optional[str] = None
    source_function: Optional[str] = None
    source_line: Optional[int] = None

    def to_jsonl(self) -> str:
        """Serialize to a single JSON Lines record."""
        return self.model_dump_json() + "\n"

    @classmethod
    def from_jsonl(cls, line: str) -> AffectiveRegistryEntry:
        """Deserialize from a JSON Lines record."""
        return cls.model_validate_json(line.strip())
