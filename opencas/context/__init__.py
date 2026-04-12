"""Context management for OpenCAS: session messages, retrieval, and prompt building."""

from .builder import ContextBuilder
from .models import ContextManifest, MessageEntry, MessageRole, RetrievalResult
from .retriever import MemoryRetriever
from .store import SessionContextStore

__all__ = [
    "ContextBuilder",
    "ContextManifest",
    "MessageEntry",
    "MessageRole",
    "MemoryRetriever",
    "RetrievalResult",
    "SessionContextStore",
]
