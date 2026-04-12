"""Identity module for OpenCAS: self-model, user-model, and continuity."""

from .manager import IdentityManager
from .models import ContinuityState, SelfModel, UserModel
from .rebuilder import IdentityRebuilder, IdentityRebuildResult
from .registry import KnowledgeEntry, SelfKnowledgeRegistry
from .store import IdentityStore

__all__ = [
    "ContinuityState",
    "IdentityManager",
    "IdentityRebuilder",
    "IdentityRebuildResult",
    "IdentityStore",
    "KnowledgeEntry",
    "SelfKnowledgeRegistry",
    "SelfModel",
    "UserModel",
]
