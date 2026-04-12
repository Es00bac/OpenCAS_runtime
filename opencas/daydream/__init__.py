"""Daydream and reflective inner-life subsystem for OpenCAS."""

from .models import (
    ConflictRecord,
    DaydreamInitiative,
    DaydreamNotification,
    DaydreamOutcome,
    DaydreamReflection,
    DaydreamSpark,
)
from .spark_evaluator import SparkEvaluator
from .store import ConflictStore, DaydreamStore
from .evaluator import ReflectionEvaluator
from .mirror import CompassionResponse, SelfCompassionMirror
from .registry import ConflictRegistry
from .resolver import ReflectionResolution, ReflectionResolver

__all__ = [
    "CompassionResponse",
    "ConflictRecord",
    "ConflictRegistry",
    "ConflictStore",
    "DaydreamInitiative",
    "DaydreamNotification",
    "DaydreamOutcome",
    "DaydreamReflection",
    "DaydreamSpark",
    "DaydreamStore",
    "ReflectionEvaluator",
    "ReflectionResolution",
    "ReflectionResolver",
    "SelfCompassionMirror",
    "SparkEvaluator",
]
