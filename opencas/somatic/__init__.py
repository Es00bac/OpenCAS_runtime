"""Somatic module for OpenCAS: physiological state and affective signals."""

from .appraisal import AppraisalEventType, SomaticAppraisalEvent
from .manager import SomaticManager
from .models import SomaticSnapshot, SomaticState
from .modulators import SomaticModulators
from .store import SomaticStore

__all__ = [
    "AppraisalEventType",
    "SomaticAppraisalEvent",
    "SomaticManager",
    "SomaticModulators",
    "SomaticSnapshot",
    "SomaticState",
    "SomaticStore",
]
