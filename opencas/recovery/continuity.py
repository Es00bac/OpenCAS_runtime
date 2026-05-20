from __future__ import annotations

from hashlib import sha256
from typing import Any

from opencas.recovery.models import ContinuityPacket, RecoveryCandidate


class ContinuityPacketBuilder:
    async def build(self, candidate: RecoveryCandidate) -> ContinuityPacket:
        payload = candidate.payload
        project_type = _first(payload, "project_type", ("meta", "project_type"), default="project")
        artifact_paths = _artifact_paths(payload)
        packet_id = _packet_id(candidate.candidate_id, artifact_paths)
        completion = _completion_criteria(payload, project_type)
        return ContinuityPacket(
            packet_id=packet_id,
            project_key=str(_first(payload, "project_key", "project_id", default=candidate.candidate_id)),
            title=candidate.title,
            project_type=str(project_type),
            canonical_artifact_paths=artifact_paths,
            latest_completed_unit=str(_first(payload, "latest_completed_unit", default="unknown")),
            current_status=str(_first(payload, "current_status", "status", default=candidate.status)),
            continuity_facts=list(_first(payload, "continuity_facts", default=[])),
            unresolved_threads=_unresolved_threads(payload),
            next_concrete_action=str(_first(payload, "next_concrete_action", "best_next_step", default="Resume from the latest artifact.")),
            completion_criteria=completion,
            evidence_refs=[*candidate.evidence_refs, *[f"artifact:{path}" for path in artifact_paths]],
        )


def _artifact_paths(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("canonical_artifact_paths") or payload.get("artifact_paths_touched") or []
    if isinstance(raw, str):
        return [raw]
    return [str(path) for path in raw]


def _unresolved_threads(payload: dict[str, Any]) -> list[str]:
    threads = list(payload.get("unresolved_threads") or [])
    threads.extend(str(defect) for defect in payload.get("known_defects") or [])
    return threads


def _completion_criteria(payload: dict[str, Any], project_type: Any) -> list[str]:
    criteria = list(payload.get("completion_criteria") or [])
    if str(project_type) == "software":
        criteria.extend(str(command) for command in payload.get("verification_commands") or [])
    return criteria or ["Recovery objective reaches evidence-backed completion."]


def _first(payload: dict[str, Any], *keys: Any, default: Any = None) -> Any:
    for key in keys:
        if isinstance(key, tuple):
            current: Any = payload
            for part in key:
                if not isinstance(current, dict) or part not in current:
                    current = None
                    break
                current = current[part]
            if current is not None:
                return current
        elif key in payload and payload[key] is not None:
            return payload[key]
    return default


def _packet_id(candidate_id: str, artifact_paths: list[str]) -> str:
    digest = sha256("|".join([candidate_id, *artifact_paths]).encode("utf-8")).hexdigest()[:16]
    return f"continuity:{digest}"
