"""Lazy provenance adapter for core subsystems.

The canonical provenance contract lives in ``opencas.api.provenance_entry``.
Core packages import this module instead so they do not pull the ``opencas.api``
package during module initialization and re-enter identity/context imports.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any, Dict, MutableMapping


@lru_cache(maxsize=1)
def _provenance_module() -> Any:
    return import_module("opencas.api.provenance_entry")


def __getattr__(name: str) -> Any:
    if name in {"Action", "Risk"}:
        return getattr(_provenance_module(), name)
    raise AttributeError(name)


def build_entry(*args: Any, **kwargs: Any) -> Any:
    return _provenance_module().build_entry(*args, **kwargs)


def create_registry_entry(*args: Any, **kwargs: Any) -> Any:
    return _provenance_module().create_registry_entry(*args, **kwargs)


def validate_registry_entry(*args: Any, **kwargs: Any) -> Any:
    return _provenance_module().validate_registry_entry(*args, **kwargs)


def encode_provenance_entry(*args: Any, **kwargs: Any) -> Any:
    return _provenance_module().encode_provenance_entry(*args, **kwargs)


def decode_provenance_entry(*args: Any, **kwargs: Any) -> Any:
    return _provenance_module().decode_provenance_entry(*args, **kwargs)


def serialize_registry_entry(*args: Any, **kwargs: Any) -> Any:
    return _provenance_module().serialize_registry_entry(*args, **kwargs)


def serialize_registry_line(*args: Any, **kwargs: Any) -> Any:
    return _provenance_module().serialize_registry_line(*args, **kwargs)


def format_provenance_entry(*args: Any, **kwargs: Any) -> Any:
    return _provenance_module().format_provenance_entry(*args, **kwargs)


def parse_registry_entry(*args: Any, **kwargs: Any) -> Any:
    return _provenance_module().parse_registry_entry(*args, **kwargs)


def parse_registry_line(*args: Any, **kwargs: Any) -> Any:
    return _provenance_module().parse_registry_line(*args, **kwargs)


def parse_provenance_entry(*args: Any, **kwargs: Any) -> Any:
    return _provenance_module().parse_provenance_entry(*args, **kwargs)


def read_registry_entry(*args: Any, **kwargs: Any) -> Any:
    return _provenance_module().read_registry_entry(*args, **kwargs)


def project_provenance_entry(entry: Any) -> Dict[str, Any]:
    return _provenance_module().project_provenance_entry(entry)


def attach_provenance(
    record: MutableMapping[str, Any] | None,
    entry: Any,
    *,
    field: str = "provenance",
) -> Dict[str, Any]:
    return _provenance_module().attach_provenance(record, entry, field=field)


def append_provenance_event(
    record: MutableMapping[str, Any] | None,
    entry: Any,
    *,
    field: str = "provenance_events",
) -> Dict[str, Any]:
    return _provenance_module().append_provenance_event(record, entry, field=field)


def append_provenance_record(
    record: MutableMapping[str, Any] | None,
    *,
    session_id: str,
    artifact: str,
    action: Any,
    why: str,
    risk: Any,
    field: str = "provenance",
    ts: str | None = None,
    actor: str | None = None,
    source_trace: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return _provenance_module().append_provenance_record(
        record,
        session_id=session_id,
        artifact=artifact,
        action=action,
        why=why,
        risk=risk,
        field=field,
        ts=ts,
        actor=actor,
        source_trace=source_trace,
    )
