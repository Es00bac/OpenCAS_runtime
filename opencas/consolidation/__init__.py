"""Consolidation module for OpenCAS."""

from .curation_store import ConsolidationCurationStore
from .engine import NightlyConsolidationEngine
from .models import ConsolidationResult, RejectedMerge, SalienceUpdate
from .signal_ranker import EpisodeSignalRanker, SignalScore

__all__ = [
    "ConsolidationCurationStore",
    "ConsolidationResult",
    "EpisodeSignalRanker",
    "NightlyConsolidationEngine",
    "RejectedMerge",
    "SalienceUpdate",
    "SignalScore",
]
