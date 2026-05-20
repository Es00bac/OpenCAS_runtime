"""Contained workspace writer for agent-initiated daydream artifacts."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

from .signals import (
    PossibilitySignal,
    PossibilitySignalRoute,
    SelfWorkKind,
    SelfWorkReceipt,
)

if TYPE_CHECKING:
    from opencas.memory import ArtifactMemoryBridge

_LOG = logging.getLogger(__name__)


class SelfWorkspaceService:
    """Write self-initiated artifacts under the managed ``workspace/self`` root."""

    def __init__(
        self,
        *,
        workspace_root: Path | str,
        artifact_bridge: Optional["ArtifactMemoryBridge"] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.self_root = (self.workspace_root / "self").resolve()
        self.artifact_bridge = artifact_bridge

    async def _ingest_into_memory(self, paths: Iterable[Path | str]) -> None:
        """Best-effort ingest just-written reflective artifacts into searchable memory.

        Without this, ``workspace/self`` notes/prototypes/research never land in the
        episodic store and stay invisible to retrieval — the exact gap that prevented
        Bulma from surfacing her own reflections during creative_writing work.
        """
        bridge = self.artifact_bridge
        if bridge is None:
            return
        for raw_path in paths:
            try:
                await bridge.sync_directory(Path(raw_path))
            except Exception as exc:
                _LOG.warning("self workspace ingest failed for %s: %s", raw_path, exc)

    def resolve_self_path(self, relative_path: str | Path) -> Path:
        """Resolve a path inside ``workspace/self`` and reject escapes."""
        candidate = (self.self_root / relative_path).resolve()
        if candidate != self.self_root and self.self_root not in candidate.parents:
            raise ValueError("resolved path is outside self workspace")
        return candidate

    async def write_note(
        self,
        signal: PossibilitySignal,
        *,
        reason: str = "",
    ) -> SelfWorkReceipt:
        path = self.resolve_self_path(
            Path("notes") / f"{self._stamp()}-{_slug(signal.summary)}.md"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._markdown(signal, title="Self Note", reason=reason), encoding="utf-8")
        await self._ingest_into_memory([path])
        return self.write_receipt(
            signal,
            route=PossibilitySignalRoute.SELF_NOTE,
            kind=SelfWorkKind.NOTE,
            outcome="self_note_written",
            summary=reason or signal.summary,
            artifact_paths=[str(path)],
        )

    async def write_experiment(
        self,
        signal: PossibilitySignal,
        *,
        reason: str = "",
    ) -> SelfWorkReceipt:
        root = self.resolve_self_path(
            Path("experiments") / f"{self._stamp()}-{_slug(signal.summary)}"
        )
        root.mkdir(parents=True, exist_ok=True)
        readme = root / "README.md"
        manifest = root / "EXPERIMENT_MANIFEST.json"
        validation = root / "VALIDATION.md"
        readme.write_text(
            self._markdown(signal, title="Self Experiment", reason=reason),
            encoding="utf-8",
        )
        manifest.write_text(
            json.dumps(
                self._manifest(signal, artifact_kind="experiment", validation_status="scaffold_only"),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        validation.write_text(self._validation_note("experiment"), encoding="utf-8")
        await self._ingest_into_memory([readme, manifest, validation])
        return self.write_receipt(
            signal,
            route=PossibilitySignalRoute.SELF_EXPERIMENT,
            kind=SelfWorkKind.EXPERIMENT,
            outcome="self_experiment_scaffolded",
            summary=reason or signal.summary,
            artifact_paths=[str(readme), str(manifest), str(validation)],
            raw_extra={"validation_status": "scaffold_only"},
        )

    async def write_prototype(
        self,
        signal: PossibilitySignal,
        *,
        reason: str = "",
    ) -> SelfWorkReceipt:
        root = self.resolve_self_path(
            Path("prototypes") / f"{self._stamp()}-{_slug(signal.summary)}"
        )
        root.mkdir(parents=True, exist_ok=True)
        readme = root / "README.md"
        manifest = root / "PROTOTYPE_MANIFEST.json"
        validation = root / "VALIDATION.md"
        readme.write_text(
            self._markdown(signal, title="Self Prototype", reason=reason),
            encoding="utf-8",
        )
        manifest.write_text(
            json.dumps(
                self._manifest(signal, artifact_kind="prototype", validation_status="scaffold_only"),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        validation.write_text(self._validation_note("prototype"), encoding="utf-8")
        await self._ingest_into_memory([readme, manifest, validation])
        return self.write_receipt(
            signal,
            route=PossibilitySignalRoute.SELF_PROTOTYPE,
            kind=SelfWorkKind.PROTOTYPE,
            outcome="self_prototype_scaffolded",
            summary=reason or signal.summary,
            artifact_paths=[str(readme), str(manifest), str(validation)],
            raw_extra={"validation_status": "scaffold_only"},
        )

    async def write_research(
        self,
        signal: PossibilitySignal,
        *,
        reason: str = "",
    ) -> SelfWorkReceipt:
        root = self.resolve_self_path(
            Path("research") / f"{self._stamp()}-{_slug(signal.summary)}"
        )
        root.mkdir(parents=True, exist_ok=True)
        note = root / "RESEARCH.md"
        manifest = root / "RESEARCH_MANIFEST.json"
        note.write_text(
            self._markdown(signal, title="Self Research Note", reason=reason),
            encoding="utf-8",
        )
        manifest.write_text(
            json.dumps(
                self._manifest(signal, artifact_kind="research", validation_status="research_question_recorded"),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        await self._ingest_into_memory([note, manifest])
        return self.write_receipt(
            signal,
            route=PossibilitySignalRoute.RESEARCH,
            kind=SelfWorkKind.RESEARCH,
            outcome="self_research_note_written",
            summary=reason or signal.summary,
            artifact_paths=[str(note), str(manifest)],
            raw_extra={"validation_status": "research_question_recorded"},
        )

    async def compost(
        self,
        signal: PossibilitySignal,
        *,
        reason: str = "",
    ) -> SelfWorkReceipt:
        root = self.resolve_self_path(
            Path("compost") / f"{self._stamp()}-{_slug(signal.summary)}"
        )
        root.mkdir(parents=True, exist_ok=True)
        note = root / "COMPOST.md"
        salvage = root / "SALVAGE_INDEX.json"
        note.write_text(
            self._markdown(signal, title="Self Work Compost", reason=reason),
            encoding="utf-8",
        )
        salvage.write_text(
            json.dumps(
                self._compost_index(signal, reason=reason),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        await self._ingest_into_memory([note, salvage])
        return self.write_receipt(
            signal,
            route=PossibilitySignalRoute.COMPOST,
            kind=SelfWorkKind.COMPOST,
            outcome="self_work_composted",
            summary=reason or signal.summary,
            artifact_paths=[str(note), str(salvage)],
            raw_extra={"compost_status": "composted"},
        )

    def write_receipt(
        self,
        signal: PossibilitySignal,
        *,
        route: PossibilitySignalRoute,
        kind: SelfWorkKind,
        outcome: str,
        summary: str,
        artifact_paths: list[str],
        raw_extra: dict[str, object] | None = None,
    ) -> SelfWorkReceipt:
        raw = {
            "route_schema_version": 2,
            "source_reflection_id": signal.source_reflection_id,
            "source_thought_index": signal.source_thought_index,
            "suggested_route": signal.suggested_route.value,
        }
        raw.update(dict(raw_extra or {}))
        receipt = SelfWorkReceipt(
            signal_id=signal.signal_id,
            route=route,
            kind=kind,
            outcome=outcome,
            summary=summary,
            artifact_paths=list(artifact_paths),
            raw=raw,
        )
        path = self.resolve_self_path(
            Path("receipts") / f"{self._stamp()}-{receipt.receipt_id}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        receipt.artifact_paths.append(str(path))
        path.write_text(
            json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return receipt

    @staticmethod
    def _stamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    @staticmethod
    def _markdown(
        signal: PossibilitySignal,
        *,
        title: str,
        reason: str,
    ) -> str:
        lines = [
            f"# {title}",
            "",
            f"Signal ID: {signal.signal_id}",
            f"Source reflection: {signal.source_reflection_id}",
            f"Route: {signal.suggested_route.value}",
            "",
            "## Summary",
            "",
            signal.summary,
            "",
            "## Imaginative Branch",
            "",
            signal.imaginative_branch or "(empty)",
            "",
            "## Practical Branch",
            "",
            signal.practical_branch or "(empty)",
            "",
            "## Bridge",
            "",
            signal.bridge or "(empty)",
            "",
            "## Reason",
            "",
            reason or signal.route_reason or "(not specified)",
            "",
            "## Evidence IDs",
            "",
            *[f"- {item}" for item in signal.evidence_ids],
            "" if signal.evidence_ids else "(empty)",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _manifest(
        signal: PossibilitySignal,
        *,
        artifact_kind: str,
        validation_status: str,
    ) -> dict[str, object]:
        return {
            "artifact_kind": artifact_kind,
            "validation_status": validation_status,
            "signal_id": signal.signal_id,
            "source_reflection_id": signal.source_reflection_id,
            "summary": signal.summary,
            "suggested_route": signal.suggested_route.value,
            "evidence_ids": list(signal.evidence_ids),
            "created_from_daydream": True,
        }

    @staticmethod
    def _validation_note(kind: str) -> str:
        return "\n".join(
            [
                "# Validation",
                "",
                f"This {kind} is a scaffold until a later worker or agent validates it.",
                "",
                "Current status: scaffold_only",
                "",
            ]
        )

    @staticmethod
    def _compost_index(signal: PossibilitySignal, *, reason: str) -> dict[str, object]:
        salvage_candidates = [
            item
            for item in (
                signal.bridge,
                signal.practical_branch,
                signal.imaginative_branch,
            )
            if item.strip()
        ]
        return {
            "compost_status": "composted",
            "signal_id": signal.signal_id,
            "source_reflection_id": signal.source_reflection_id,
            "reason": reason or signal.route_reason or "",
            "discarded_active_work": True,
            "salvage_candidates": salvage_candidates,
            "evidence_ids": list(signal.evidence_ids),
        }


def _slug(value: str, *, limit: int = 60) -> str:
    chars: list[str] = []
    last_dash = False
    for char in str(value or "").lower():
        if char.isalnum():
            chars.append(char)
            last_dash = False
        elif not last_dash:
            chars.append("-")
            last_dash = True
        if len(chars) >= limit:
            break
    return "".join(chars).strip("-") or "signal"
