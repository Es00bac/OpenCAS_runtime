"""Context management for OpenCAS: session messages, retrieval, and prompt building."""

from .builder import ContextBuilder
from .hemispheres import ContextAuthority, ContextLane
from .models import (
    ContextManifest,
    ContinuationHandle,
    ContinuationPacket,
    MessageEntry,
    MessageRole,
    RetrievalResult,
    repair_tool_message_sequence,
)
from .packet_builder import ContextPacket, ContextPacketBuilder, ContextSection
from .proposals import ContextProposal, ContextProposalStore, ProposalStatus
from .retriever import MemoryRetriever
from .store import SessionContextStore
from .truth_arbiter import TruthArbiter, TruthSnapshot

__all__ = [
    "ContextBuilder",
    "ContextAuthority",
    "ContextManifest",
    "ContextPacket",
    "ContextPacketBuilder",
    "ContextProposal",
    "ContextProposalStore",
    "ContextSection",
    "ContextLane",
    "ContinuationHandle",
    "ContinuationPacket",
    "MessageEntry",
    "MessageRole",
    "MemoryRetriever",
    "ProposalStatus",
    "RetrievalResult",
    "SessionContextStore",
    "TruthArbiter",
    "TruthSnapshot",
    "repair_tool_message_sequence",
]
