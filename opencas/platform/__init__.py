"""Canonical platform capability models and registry."""

from __future__ import annotations

from .models import (
    CapabilityDescriptor,
    CapabilitySource,
    CapabilityStatus,
    ExtensionDescriptor,
)
from .registry import CapabilityRegistry

__all__ = [
    "CapabilityDescriptor",
    "CapabilityRegistry",
    "CapabilitySource",
    "CapabilityStatus",
    "ExtensionDescriptor",
]
