"""Append-only affective state registry writer for OpenCAS.

Captures system affective state at execution moments and writes structured
entries to a durable JSONL log without overwriting historical runs.
"""

from .writer import AffectiveRegistryWriter
from .models import (
    AffectiveRegistryEntry,
    AffectiveState,
    ExecutionContext,
    ExecutionPhase,
    SystemMetrics,
)

__all__ = [
    "AffectiveRegistryWriter",
    "AffectiveRegistryEntry",
    "AffectiveState",
    "ExecutionContext",
    "ExecutionPhase",
    "SystemMetrics",
]
