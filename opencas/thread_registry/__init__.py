"""Peripheral thread registry for autonomous artifacts and loose insights."""

from .models import (
    BeadEntry,
    BeadSourceKind,
    BeadStatus,
    BeadValidationResult,
    ThreadAnchor,
    ThreadStatus,
)
from .service import (
    ThreadRegistryService,
    compute_content_hashes,
    make_anchor_id,
    normalize_content_for_hash,
)
from .store import ThreadRegistryStore

__all__ = [
    "BeadEntry",
    "BeadSourceKind",
    "BeadStatus",
    "BeadValidationResult",
    "ThreadAnchor",
    "ThreadRegistryService",
    "ThreadRegistryStore",
    "ThreadStatus",
    "compute_content_hashes",
    "make_anchor_id",
    "normalize_content_for_hash",
]
