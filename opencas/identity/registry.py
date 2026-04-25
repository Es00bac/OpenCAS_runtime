"""Self-knowledge registry for structured, versioned self-beliefs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class KnowledgeEntry(BaseModel):
    """A structured entry in the self-knowledge registry."""

    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    domain: str
    key: str
    value: Any
    confidence: float = 1.0
    evidence_ids: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class SelfKnowledgeRegistry:
    """File-backed (JSONL) registry for structured self-beliefs."""

    def __init__(self, store_path: Path | str) -> None:
        self.store_path = Path(store_path)
        self._entries: List[KnowledgeEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            with self.store_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    self._entries.append(KnowledgeEntry.model_validate(raw))
        except (json.JSONDecodeError, OSError):
            pass

    def _flush(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.store_path.open("w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(entry.model_dump_json() + "\n")

    def record(
        self,
        domain: str,
        key: str,
        value: Any,
        confidence: float = 1.0,
        evidence_ids: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeEntry:
        entry = KnowledgeEntry(
            domain=domain,
            key=key,
            value=value,
            confidence=confidence,
            evidence_ids=evidence_ids or [],
            meta=meta or {},
        )
        self._entries.append(entry)
        self._flush()
        return entry

    def get(self, domain: str, key: str) -> Optional[KnowledgeEntry]:
        for entry in reversed(self._entries):
            if entry.domain == domain and entry.key == key:
                return entry
        return None

    def list_by_domain(self, domain: str) -> List[KnowledgeEntry]:
        return [e for e in self._entries if e.domain == domain]

    def search(self, key_substring: str) -> List[KnowledgeEntry]:
        sub = key_substring.lower()
        return [e for e in self._entries if sub in e.key.lower()]

    def to_self_beliefs(self) -> Dict[str, Any]:
        """Flatten latest entry per (domain, key) into a nested dict."""
        latest: Dict[str, Any] = {}
        for entry in self._entries:
            latest[(entry.domain, entry.key)] = entry.value
        nested: Dict[str, Any] = {}
        for (domain, key), value in latest.items():
            nested.setdefault(domain, {})[key] = value
        return nested
