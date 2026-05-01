"""Theory of Mind module for OpenCAS: beliefs, intentions, and metacognition."""

from .models import Belief, BeliefSubject, Intention, IntentionStatus, MetacognitiveResult
from .store import TomStore

__all__ = ["Belief", "BeliefSubject", "Intention", "IntentionStatus", "MetacognitiveResult", "ToMEngine", "TomStore"]


def __getattr__(name: str):
    if name == "ToMEngine":
        from .engine import ToMEngine

        return ToMEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
