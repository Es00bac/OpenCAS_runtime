"""Lazy adapter for canonical provenance event helpers."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any, Dict, MutableMapping


@lru_cache(maxsize=1)
def _event_module() -> Any:
    return import_module("opencas.api.provenance_events")


def __getattr__(name: str) -> Any:
    if name in {"ProvenanceEvent", "ProvenanceEventType"}:
        return getattr(_event_module(), name)
    raise AttributeError(name)


def build_provenance_event(*args: Any, **kwargs: Any) -> Any:
    return _event_module().build_provenance_event(*args, **kwargs)


def append_provenance_event(*args: Any, **kwargs: Any) -> Any:
    return _event_module().append_provenance_event(*args, **kwargs)


def emit_provenance_event(*args: Any, **kwargs: Any) -> Any:
    return _event_module().emit_provenance_event(*args, **kwargs)


def parse_provenance_event(*args: Any, **kwargs: Any) -> Any:
    return _event_module().parse_provenance_event(*args, **kwargs)


def provenance_event_to_dict(event: Any) -> Dict[str, Any]:
    return _event_module().provenance_event_to_dict(event)


def serialize_provenance_event(*args: Any, **kwargs: Any) -> Any:
    return _event_module().serialize_provenance_event(*args, **kwargs)
