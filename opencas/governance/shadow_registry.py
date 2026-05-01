"""Production ShadowRegistry for blocked-intention capture."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import re
from typing import Any, Dict, Iterable, List, Optional

from opencas.telemetry import EventKind, Tracer

from .shadow_models import (
    BlockReason,
    BlockedIntention,
    ClusterTriageStatus,
    ShadowClusterTriageState,
)
from .shadow_store import ShadowRegistryStore


class ShadowRegistry:
    """Capture blocked intentions without obscuring the original failure."""

    def __init__(
        self,
        store: ShadowRegistryStore,
        tracer: Optional[Tracer] = None,
    ) -> None:
        self.store = store
        self.tracer = tracer

    def capture(
        self,
        *,
        tool_name: str,
        parameters: Dict[str, Any],
        reason: BlockReason,
        context: str,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        action_id: Optional[str] = None,
        artifact: Optional[str] = None,
        target_kind: Optional[str] = None,
        target_id: Optional[str] = None,
        risk_tier: Optional[str] = None,
        decision_level: Optional[str] = None,
        capture_source: Optional[str] = None,
        agent_state: Optional[Dict[str, Any]] = None,
    ) -> BlockedIntention:
        """Persist one blocked intention."""

        intention = BlockedIntention(
            tool_name=tool_name or "unknown_tool",
            intent_summary=_summarize_intent(tool_name, parameters),
            raw_parameters=dict(parameters or {}),
            block_reason=reason,
            block_context=str(context or "").strip() or reason.value,
            session_id=_clean(session_id),
            task_id=_clean(task_id),
            action_id=_clean(action_id),
            artifact=_clean(artifact),
            target_kind=_clean(target_kind),
            target_id=_clean(target_id),
            risk_tier=_clean(risk_tier),
            decision_level=_clean(decision_level),
            capture_source=_clean(capture_source),
            agent_state=dict(agent_state or {}),
        )
        saved = self.store.save(intention)
        if self.tracer is not None:
            self.tracer.log(
                EventKind.WARNING,
                "ShadowRegistry: blocked intention captured",
                {
                    "intention_id": saved.id,
                    "tool_name": saved.tool_name,
                    "block_reason": saved.block_reason.value,
                    "fingerprint": saved.fingerprint,
                    "session_id": saved.session_id,
                },
            )
        return saved

    def capture_action_decision(self, ctx: Dict[str, Any]) -> Optional[BlockedIntention]:
        """Capture denied runtime approvals."""

        if bool(ctx.get("approved", False)):
            return None
        tool_name = str(ctx.get("tool_name", "") or "").strip()
        if not tool_name:
            return None
        return self.capture(
            tool_name=tool_name,
            parameters=_coerce_mapping(ctx.get("args")),
            reason=BlockReason.APPROVAL_DENIED,
            context=str(ctx.get("reasoning", "") or "action denied").strip() or "action denied",
            session_id=ctx.get("session_id"),
            task_id=ctx.get("task_id"),
            artifact=ctx.get("artifact"),
            target_kind=ctx.get("target_kind"),
            target_id=ctx.get("target_id"),
            risk_tier=ctx.get("risk_tier"),
            decision_level=ctx.get("decision_level"),
            capture_source="action_decision",
            agent_state=_coerce_mapping(ctx.get("source_trace")),
        )

    def capture_tool_block(self, ctx: Dict[str, Any]) -> Optional[BlockedIntention]:
        """Capture tool-level blocked executions without logging generic failures."""

        if bool(ctx.get("result_success", False)):
            return None
        tool_name = str(ctx.get("tool_name", "") or "").strip()
        if not tool_name:
            return None

        metadata = _coerce_mapping(ctx.get("result_metadata"))
        output = str(ctx.get("result_output", "") or "").strip()
        reason, block_context = _classify_tool_block(metadata, output)
        if reason is None:
            return None

        return self.capture(
            tool_name=tool_name,
            parameters=_coerce_mapping(ctx.get("args")),
            reason=reason,
            context=block_context,
            session_id=ctx.get("session_id"),
            task_id=ctx.get("task_id"),
            artifact=ctx.get("artifact"),
            target_kind=ctx.get("target_kind"),
            target_id=ctx.get("target_id"),
            risk_tier=ctx.get("risk_tier"),
            capture_source="tool_execution",
            agent_state={
                "metadata": metadata,
                "source_trace": _coerce_mapping(ctx.get("source_trace")),
            },
        )

    def list_recent(self, limit: int = 10, offset: int = 0) -> List[BlockedIntention]:
        return self.store.list_recent(limit=limit, offset=offset)

    def get_cluster(self, intention_id: str) -> List[BlockedIntention]:
        target = self.store.get(intention_id)
        if target is None:
            return []
        return [
            item
            for item in self.store.list_all()
            if item.fingerprint == target.fingerprint
        ]

    def capture_tool_loop_guard(self, payload: Dict[str, Any]) -> Optional[BlockedIntention]:
        """Persist a tool-loop guard stop with enough context for later reuse."""

        if not isinstance(payload, dict):
            return None
        dominant_tool = _clean(payload.get("dominant_tool")) or "tool_loop_guard"
        executed_counts = _coerce_mapping(payload.get("executed_tool_counts"))
        pending_tools = payload.get("pending_tools") if isinstance(payload.get("pending_tools"), list) else []
        return self.capture(
            tool_name="tool_loop_guard",
            parameters={
                "objective": str(payload.get("objective", "") or "").strip(),
                "dominant_tool": dominant_tool,
                "executed_tool_counts": executed_counts,
                "pending_tools": pending_tools,
            },
            reason=BlockReason.TOOL_LOOP_GUARD_BLOCKED,
            context=str(payload.get("guard_reason", "") or "tool loop guard fired").strip() or "tool loop guard fired",
            session_id=payload.get("session_id"),
            task_id=payload.get("task_id"),
            artifact=payload.get("artifact"),
            target_kind="tool_loop",
            target_id=_clean(payload.get("task_id")) or _clean(payload.get("session_id")),
            capture_source="tool_use_loop",
            agent_state={
                "executed_steps": payload.get("executed_steps", []),
                "pending_calls": payload.get("pending_calls", []),
            },
        )

    def capture_retry_blocked(self, payload: Dict[str, Any]) -> Optional[BlockedIntention]:
        """Persist a retry-governor refusal for later safer-alternative planning."""

        if not isinstance(payload, dict):
            return None
        artifact = _clean(payload.get("artifact")) or _clean(payload.get("canonical_artifact_path"))
        return self.capture(
            tool_name="repair_retry",
            parameters={
                "objective": str(payload.get("objective", "") or "").strip(),
                "attempt": payload.get("attempt"),
                "canonical_artifact_path": artifact,
                "retry_mode": _clean(payload.get("retry_mode")),
                "governor_mode": _clean(payload.get("governor_mode")),
                "best_next_step": _clean(payload.get("best_next_step")),
                "failed_framings": _coerce_string_list(payload.get("failed_framings")),
                "suppression_reason": _clean(payload.get("suppression_reason")),
            },
            reason=BlockReason.RETRY_BLOCKED,
            context=str(payload.get("reason", "") or "retry blocked").strip() or "retry blocked",
            session_id=payload.get("session_id"),
            task_id=payload.get("task_id"),
            artifact=artifact,
            target_kind=_clean(payload.get("target_kind")) or "repair_task",
            target_id=_clean(payload.get("target_id")) or _clean(payload.get("task_id")),
            capture_source=_clean(payload.get("capture_source")) or "repair_executor",
            agent_state={
                "retry_governor": _coerce_mapping(payload.get("retry_governor")),
                "resume_project": _coerce_mapping(payload.get("resume_project")),
                "reframe_hint": _clean(payload.get("reframe_hint")),
                "duplicate_context": _coerce_mapping(payload.get("duplicate_context")),
            },
        )

    def summary(self, limit: int = 10, cluster_limit: int = 5) -> Dict[str, Any]:
        """Return a compact dashboard-safe summary of blocked intentions."""

        items = self.store.list_all()
        reason_counts = Counter(item.block_reason.value for item in items)
        grouped = self._group_clusters(items)
        dismissed_clusters = sum(
            1
            for fingerprint in grouped
            if self._cluster_state(fingerprint).triage_status == ClusterTriageStatus.DISMISSED
        )
        clusters = self._top_clusters(items, limit=cluster_limit, include_dismissed=False)
        return {
            "total_entries": len(items),
            "active_clusters": max(len(grouped) - dismissed_clusters, 0),
            "dismissed_clusters": dismissed_clusters,
            "reason_counts": dict(reason_counts),
            "recent_entries": [self._serialize_item(item) for item in items[:limit]],
            "top_clusters": clusters,
        }

    def build_planning_context(
        self,
        *,
        objective: str,
        artifact: Optional[str] = None,
        limit: int = 3,
    ) -> Dict[str, Any]:
        """Build planner-facing guidance from related blocked-intention clusters."""

        related = self._find_related_items(objective=objective, artifact=artifact, limit=max(limit * 3, limit))
        if not related:
            return {
                "available": False,
                "clusters": [],
                "prompt_block": "",
            }

        clusters = self._top_clusters(related, limit=limit, include_dismissed=False)
        if not clusters:
            return {
                "available": False,
                "clusters": [],
                "prompt_block": "",
            }
        suggestions = self._safer_alternative_suggestions(cluster["block_reason"] for cluster in clusters)
        failed_framings = self._blocked_framings(related, limit=limit)
        reframe_hints = self._reframe_hints(related, limit=limit)
        cluster_lines = [
            f"- {cluster['count']}x {cluster['block_reason']} around {cluster['intent_summary']}"
            for cluster in clusters
        ]
        suggestion_lines = [f"- {line}" for line in suggestions]
        prompt_lines = [
            "Related blocked-intention clusters:",
            *cluster_lines,
            "Safer alternatives:",
            *suggestion_lines,
        ]
        if failed_framings:
            prompt_lines.extend(
                [
                    "Previously blocked framings to avoid repeating:",
                    *[f"- {line}" for line in failed_framings],
                ]
            )
        if reframe_hints:
            prompt_lines.extend(
                [
                    "Distinct next-step candidates:",
                    *[f"- {line}" for line in reframe_hints],
                ]
            )
        prompt_lines.extend(
            [
                "Blocker handling rule:",
                "- Do not reuse a blocked framing with cosmetic rewording.",
                "- Before accepting the blocker as current, verify that the underlying reason still applies.",
                "- If the blocker no longer applies, take the smallest safe next action and collect fresh evidence.",
            ]
        )
        prompt_block = "\n".join(prompt_lines)
        return {
            "available": True,
            "clusters": clusters,
            "failed_framings": failed_framings,
            "reframe_hints": reframe_hints,
            "prompt_block": prompt_block,
        }

    def inspect_cluster(self, fingerprint: str, limit: int = 25) -> Dict[str, Any]:
        """Return raw entries for one recurring blocked-intention cluster."""

        fingerprint_text = _clean(fingerprint)
        if not fingerprint_text:
            return {"available": False, "fingerprint": None, "entries": []}

        matching = [
            item for item in self.store.list_all() if item.fingerprint == fingerprint_text
        ]
        if not matching:
            return {"available": False, "fingerprint": fingerprint_text, "entries": []}

        matching.sort(key=lambda item: item.captured_at, reverse=True)
        latest = matching[0]
        triage = self._cluster_state(fingerprint_text)
        return {
            "available": True,
            "fingerprint": fingerprint_text,
            "count": len(matching),
            "block_reason": latest.block_reason.value,
            "tool_name": latest.tool_name,
            "intent_summary": latest.intent_summary,
            "latest_captured_at": latest.captured_at.isoformat(),
            **self._serialize_cluster_state(triage),
            "entries": [self._serialize_item(item) for item in matching[:limit]],
        }

    def triage_cluster(
        self,
        fingerprint: str,
        *,
        annotation: Optional[str] = None,
        dismissed: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Persist operator triage state for one cluster and return refreshed detail."""

        fingerprint_text = _clean(fingerprint)
        if not fingerprint_text:
            return {"available": False, "fingerprint": None, "entries": []}

        matching = [
            item for item in self.store.list_all() if item.fingerprint == fingerprint_text
        ]
        if not matching:
            return {"available": False, "fingerprint": fingerprint_text, "entries": []}

        triage = self._cluster_state(fingerprint_text)
        changed = False
        now = datetime.now(timezone.utc)

        if annotation is not None:
            cleaned_annotation = _clean(annotation)
            if triage.annotation != cleaned_annotation:
                triage.annotation = cleaned_annotation
                changed = True
        if dismissed is not None:
            next_status = ClusterTriageStatus.DISMISSED if dismissed else ClusterTriageStatus.ACTIVE
            if triage.triage_status != next_status:
                triage.triage_status = next_status
                triage.dismissed_at = now if next_status == ClusterTriageStatus.DISMISSED else None
                changed = True
        if changed:
            triage.triaged_at = now
            self.store.save_cluster_state(triage)
        return self.inspect_cluster(fingerprint_text)

    def _find_related_items(
        self,
        *,
        objective: str,
        artifact: Optional[str],
        limit: int,
    ) -> List[BlockedIntention]:
        artifact_text = _clean(artifact)
        objective_tokens = _tokenize(objective)
        scored: list[tuple[int, BlockedIntention]] = []
        for item in self.store.list_all():
            score = 0
            if artifact_text and item.artifact == artifact_text:
                score += 10
            content_tokens = _tokenize(" ".join([item.intent_summary, item.block_context]))
            score += len(objective_tokens & content_tokens)
            if item.target_kind == "repair_task":
                score += 1
            if item.block_reason in {BlockReason.RETRY_BLOCKED, BlockReason.TOOL_LOOP_GUARD_BLOCKED}:
                score += 2
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: (pair[0], pair[1].captured_at), reverse=True)
        return [item for _, item in scored[:limit]]

    def _top_clusters(
        self,
        items: List[BlockedIntention],
        limit: int,
        *,
        include_dismissed: bool,
    ) -> List[Dict[str, Any]]:
        grouped = self._group_clusters(items)
        ranked = sorted(
            grouped.values(),
            key=lambda cluster: (len(cluster), max(entry.captured_at for entry in cluster)),
            reverse=True,
        )
        clusters: list[Dict[str, Any]] = []
        for cluster in ranked:
            latest = max(cluster, key=lambda item: item.captured_at)
            triage = self._cluster_state(latest.fingerprint or latest.id)
            if not include_dismissed and triage.triage_status == ClusterTriageStatus.DISMISSED:
                continue
            clusters.append(
                {
                    "fingerprint": latest.fingerprint,
                    "count": len(cluster),
                    "block_reason": latest.block_reason.value,
                    "tool_name": latest.tool_name,
                    "intent_summary": latest.intent_summary,
                    "latest_captured_at": latest.captured_at.isoformat(),
                    **self._serialize_cluster_state(triage),
                }
            )
            if len(clusters) >= limit:
                break
        return clusters

    @staticmethod
    def _group_clusters(items: List[BlockedIntention]) -> dict[str, list[BlockedIntention]]:
        grouped: dict[str, list[BlockedIntention]] = defaultdict(list)
        for item in items:
            grouped[item.fingerprint or item.id].append(item)
        return grouped

    @staticmethod
    def _serialize_item(item: BlockedIntention) -> Dict[str, Any]:
        return {
            "id": item.id,
            "captured_at": item.captured_at.isoformat(),
            "tool_name": item.tool_name,
            "intent_summary": item.intent_summary,
            "block_reason": item.block_reason.value,
            "block_context": item.block_context,
            "artifact": item.artifact,
            "capture_source": item.capture_source,
            "target_kind": item.target_kind,
            "target_id": item.target_id,
        }

    def _cluster_state(self, fingerprint: str) -> ShadowClusterTriageState:
        state = self.store.get_cluster_state(fingerprint)
        if state is not None:
            return state
        return ShadowClusterTriageState(fingerprint=fingerprint)

    @staticmethod
    def _serialize_cluster_state(state: ShadowClusterTriageState) -> Dict[str, Any]:
        return {
            "triage_status": state.triage_status.value,
            "annotation": state.annotation,
            "triaged_at": state.triaged_at.isoformat() if state.triaged_at else None,
            "dismissed_at": state.dismissed_at.isoformat() if state.dismissed_at else None,
        }

    @staticmethod
    def _blocked_framings(items: List[BlockedIntention], limit: int) -> List[str]:
        framings: list[str] = []
        for item in items:
            objective = str(item.raw_parameters.get("objective", "") or "").strip()
            if objective:
                framings.append(objective)
            for framing in _coerce_string_list(item.raw_parameters.get("failed_framings")):
                framings.append(framing)
        return _dedupe_strings(framings)[:limit]

    @staticmethod
    def _reframe_hints(items: List[BlockedIntention], limit: int) -> List[str]:
        hints: list[str] = []
        for item in items:
            raw_next = str(item.raw_parameters.get("best_next_step", "") or "").strip()
            if raw_next:
                hints.append(raw_next)
            raw_hint = str(item.agent_state.get("reframe_hint", "") or "").strip()
            if raw_hint:
                hints.append(raw_hint)
            resume_project = _coerce_mapping(item.agent_state.get("resume_project"))
            project_next = str(resume_project.get("best_next_step", "") or "").strip()
            if project_next:
                hints.append(project_next)
        return _dedupe_strings(hints)[:limit]

    @staticmethod
    def _safer_alternative_suggestions(reasons: Iterable[str]) -> List[str]:
        suggestions: list[str] = []
        reason_set = list(dict.fromkeys(reasons))
        for reason in reason_set:
            if reason == BlockReason.RETRY_BLOCKED.value:
                suggestions.append("Prefer deterministic review of the existing artifact rather than broad replanning.")
                suggestions.append("Prefer one narrow edit tied to the canonical artifact, then rerun verification.")
            elif reason == BlockReason.TOOL_LOOP_GUARD_BLOCKED.value:
                suggestions.append("Cap the next pass to one targeted read/write sequence and stop after the first verification result.")
            elif reason == BlockReason.VALIDATION_BLOCKED.value:
                suggestions.append("Validate tool arguments and workspace paths before the next tool call.")
            elif reason == BlockReason.SAFETY_BLOCKED.value:
                suggestions.append("Prefer workspace-contained, non-destructive commands before escalating shell actions.")
        if not suggestions:
            suggestions.append("Prefer the smallest safe next step that preserves artifact continuity.")
        return list(dict.fromkeys(suggestions))


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _coerce_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    items: List[str] = []
    for entry in value:
        text = str(entry or "").strip()
        if text:
            items.append(text)
    return items


def _clean(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _summarize_intent(tool_name: str, parameters: Dict[str, Any]) -> str:
    if tool_name == "tool_loop_guard":
        dominant_tool = str(parameters.get("dominant_tool", "") or "").strip()
        return f"guard:{dominant_tool or 'unknown'}"
    if tool_name == "repair_retry":
        canonical_artifact = str(parameters.get("canonical_artifact_path", "") or "").strip()
        return f"retry:{canonical_artifact}" if canonical_artifact else "retry:unknown"
    if tool_name == "bash_run_command":
        command = str(parameters.get("command", "") or "").strip()
        return f"shell:{command[:80]}" if command else "shell:unknown"
    file_path = str(parameters.get("file_path", "") or parameters.get("path", "")).strip()
    if file_path:
        return f"file:{file_path}"
    url = str(parameters.get("url", "") or parameters.get("web_url", "")).strip()
    if url:
        return f"url:{url[:80]}"
    keys = ",".join(sorted(parameters.keys()))
    return f"{tool_name}:[{keys}]"


def _classify_tool_block(
    metadata: Dict[str, Any],
    output: str,
) -> tuple[Optional[BlockReason], str]:
    if metadata.get("hook_blocked"):
        return BlockReason.HOOK_BLOCKED, str(metadata.get("reason", "") or output).strip() or "hook blocked"
    validation_error = str(metadata.get("validation_error", "") or "").strip()
    if validation_error:
        return BlockReason.VALIDATION_BLOCKED, f"tool validation failed: {validation_error}"
    lowered = output.lower()
    if "command blocked by safety policy:" in lowered:
        return BlockReason.SAFETY_BLOCKED, output
    return None, ""


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_./-]+", (text or "").lower()) if len(token) >= 4}


def _dedupe_strings(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered
