"""Daydream and reflective inner-life subsystem for OpenCAS."""

from .conflict_store import ConflictStore
from .daydream_store import DaydreamStore
from .evaluator import ReflectionEvaluator
from .mirror import CompassionResponse, SelfCompassionMirror
from .models import (
    ConflictRecord,
    DaydreamDialogueTurn,
    DaydreamInitiative,
    DaydreamNotification,
    DaydreamOutcome,
    DaydreamReflection,
    DaydreamSpark,
    DaydreamThought,
    DaydreamThoughtKind,
    DaydreamThoughtRoute,
)
from .registry import ConflictRegistry
from .resolver import ReflectionResolution, ReflectionResolver
from .promotion import DaydreamPromotionService
from .signal_builder import DaydreamSignalBuilder
from .signal_store import DaydreamSignalStore
from .self_workspace import SelfWorkspaceService
from .signals import (
    ContactPosture,
    PossibilitySignal,
    PossibilitySignalRoute,
    SelfWorkKind,
    SelfWorkReceipt,
)
from .spark_evaluator import SparkEvaluator
from .association_memory import (
    DAYDREAM_ASSOCIATION_TAG,
    DAYDREAM_RETRIEVABLE_TAG,
    DaydreamAssociationPersistResult,
    attach_daydream_association_context,
    build_daydream_association_memory,
    daydream_association_memory_id,
    persist_daydream_association_memory,
    should_persist_daydream_association,
)

__all__ = [
    "CompassionResponse",
    "ConflictRecord",
    "ConflictRegistry",
    "ConflictStore",
    "DaydreamInitiative",
    "DaydreamDialogueTurn",
    "DaydreamNotification",
    "DaydreamPromotionService",
    "DaydreamOutcome",
    "DaydreamReflection",
    "DaydreamSignalBuilder",
    "DaydreamSignalStore",
    "DaydreamSpark",
    "DaydreamStore",
    "DaydreamThought",
    "DaydreamThoughtKind",
    "DaydreamThoughtRoute",
    "ContactPosture",
    "PossibilitySignal",
    "PossibilitySignalRoute",
    "ReflectionEvaluator",
    "ReflectionResolution",
    "ReflectionResolver",
    "SelfCompassionMirror",
    "SelfWorkspaceService",
    "SelfWorkKind",
    "SelfWorkReceipt",
    "SparkEvaluator",
    "DAYDREAM_ASSOCIATION_TAG",
    "DAYDREAM_RETRIEVABLE_TAG",
    "DaydreamAssociationPersistResult",
    "attach_daydream_association_context",
    "build_daydream_association_memory",
    "daydream_association_memory_id",
    "persist_daydream_association_memory",
    "should_persist_daydream_association",
]
