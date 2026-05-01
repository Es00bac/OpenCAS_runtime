"""Meaningful-progress guard for tool-use loops."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ProgressAssessment:
    """Classified progress signal for one tool result."""

    meaningful: bool
    signal: str
    reason: str
    terminal: bool = False


class MeaningfulProgressGuard:
    """Stop tool loops that are spending actions without useful progress."""

    _ARTIFACT_KEYS = {
        "artifact",
        "artifact_path",
        "artifact_paths",
        "changed",
        "changed_files",
        "commitment_id",
        "created",
        "file_path",
        "files_changed",
        "modified",
        "path",
        "plan_id",
        "provenance_events",
        "updated",
        "work_id",
    }
    _BLOCKER_MARKERS = (
        "approval required",
        "blocked",
        "forbidden",
        "needs approval",
        "permission denied",
        "requires approval",
        "tool execution blocked",
        "unauthorized",
    )
    _RECOVERABLE_WEB_FAILURE_MARKERS = (
        "client error",
        "server error",
        "http status",
        "for url",
        "timed out",
        "timeout",
        "connection error",
        "too many redirects",
    )

    def __init__(
        self,
        *,
        max_consecutive_no_progress: int = 4,
        repeated_evidence_limit: int = 3,
    ) -> None:
        self.max_consecutive_no_progress = max(1, max_consecutive_no_progress)
        self.repeated_evidence_limit = max(2, repeated_evidence_limit)
        self.consecutive_no_progress = 0
        self._evidence_counts: Dict[str, int] = {}
        self.last_assessment: Optional[ProgressAssessment] = None

    def record_result(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Optional[str]:
        """Record a tool result and return a stop reason when the loop should halt."""
        assessment = self.assess_result(tool_name, args, result)
        self.last_assessment = assessment

        if assessment.terminal:
            return f"Meaningful progress contract blocker: {assessment.reason}"

        if assessment.meaningful:
            self.consecutive_no_progress = 0
            if assessment.signal == "evidence":
                signature = self._evidence_signature(tool_name, result)
                count = self._evidence_counts.get(signature, 0) + 1
                self._evidence_counts[signature] = count
                if count >= self.repeated_evidence_limit:
                    return (
                        "Meaningful progress contract: repeated stale evidence "
                        f"from {tool_name} without a new artifact or decision."
                    )
            return None

        self.consecutive_no_progress += 1
        if self.consecutive_no_progress >= self.max_consecutive_no_progress:
            return (
                "Meaningful progress contract: no meaningful progress after "
                f"{self.consecutive_no_progress} tool results."
            )
        return None

    def assess_result(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Dict[str, Any],
    ) -> ProgressAssessment:
        """Classify whether one tool result advanced the loop."""
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        explicit = metadata.get("meaningful_progress")
        if explicit is True:
            return ProgressAssessment(True, "explicit", "tool metadata marked progress")
        if explicit is False:
            return ProgressAssessment(False, "explicit", "tool metadata marked no progress")

        output = str(result.get("output", "") or "").strip()
        output_lower = output.lower()
        success = bool(result.get("success", False))
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        is_approval_block = not success and "decision" in result

        if not success and self._recoverable_web_failure(tool_name, output_lower):
            return ProgressAssessment(
                False,
                "recoverable_web_failure",
                f"{tool_name} hit recoverable web friction",
            )

        if not success and not is_approval_block and any(marker in output_lower for marker in self._BLOCKER_MARKERS):
            return ProgressAssessment(
                False,
                "blocker",
                f"{tool_name} reported a blocker",
                terminal=True,
            )

        if self._metadata_has_artifact(metadata):
            return ProgressAssessment(True, "artifact", f"{tool_name} produced artifact metadata")

        if not success:
            return ProgressAssessment(False, "failure", f"{tool_name} failed without new evidence")

        if output and not self._generic_success_output(output_lower):
            if self._observational_tool(tool_name):
                return ProgressAssessment(True, "evidence", f"{tool_name} produced evidence")
            return ProgressAssessment(True, "action", f"{tool_name} reported action progress")

        if output:
            return ProgressAssessment(False, "generic_success", f"{tool_name} returned only generic success")

        return ProgressAssessment(False, "empty", f"{tool_name} returned no output")

    def _recoverable_web_failure(self, tool_name: str, output_lower: str) -> bool:
        name = tool_name.lower()
        if name not in {"web_fetch", "web_search"}:
            return False
        return any(marker in output_lower for marker in self._RECOVERABLE_WEB_FAILURE_MARKERS)

    def _metadata_has_artifact(self, metadata: Dict[str, Any]) -> bool:
        for key in self._ARTIFACT_KEYS:
            value = metadata.get(key)
            if value:
                return True
        return False

    @staticmethod
    def _generic_success_output(output_lower: str) -> bool:
        normalized = " ".join(output_lower.split())
        return normalized in {"ok", "done", "success", "successful", "completed"}

    @staticmethod
    def _observational_tool(tool_name: str) -> bool:
        name = tool_name.lower()
        markers = (
            "fetch",
            "grep",
            "list",
            "observe",
            "poll",
            "read",
            "search",
            "snapshot",
            "status",
        )
        return any(marker in name for marker in markers)

    @staticmethod
    def _evidence_signature(tool_name: str, result: Dict[str, Any]) -> str:
        output = str(result.get("output", "") or "")
        normalized_output = " ".join(output.lower().split())[:500]
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        safe_metadata = {
            key: metadata[key]
            for key in sorted(metadata)
            if isinstance(metadata.get(key), (str, int, float, bool, type(None)))
        }
        raw = json.dumps(
            {
                "tool": tool_name,
                "output": normalized_output,
                "metadata": safe_metadata,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
