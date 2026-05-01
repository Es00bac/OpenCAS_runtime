"""Deterministic salvage packet builder for execution retries."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Iterable, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from .models import AttemptOutcome, AttemptSalvagePacket, RepairTask, RetryMode


def build_salvage_packet(
    task: RepairTask,
    *,
    outcome: str | AttemptOutcome,
    canonical_artifact_path: str | None,
    artifact_paths_touched: list[str],
    tool_calls: list[dict[str, Any]],
) -> AttemptSalvagePacket:
    """Build a deterministic salvage packet from the latest execution attempt."""

    plan_output = _phase_output(task, "plan")
    exec_output = _phase_output(task, "execute")
    verify_output = _phase_output(task, "verify")
    project_signature = _extract_project_signature(task)
    tool_signature = _tool_signature(tool_calls)
    normalized_artifact_paths = _unique_sorted_paths(artifact_paths_touched)
    normalized_outcome = _coerce_outcome(outcome)
    meaningful_progress_signal = _meaningful_progress_signal(
        exec_output=exec_output,
        verify_output=verify_output,
        canonical_artifact_path=canonical_artifact_path,
        artifact_paths_touched=normalized_artifact_paths,
        outcome=normalized_outcome,
    )
    discovered_constraints = _extract_constraints(exec_output, verify_output)
    if meaningful_progress_signal == "no_meaningful_progress":
        discovered_constraints = _dedupe_preserve_order(
            [*discovered_constraints, "no meaningful progress"]
        )
    divergence_signature = build_divergence_signature(
        objective=task.objective,
        project_signature=project_signature,
        canonical_artifact_path=canonical_artifact_path,
        artifact_paths_touched=normalized_artifact_paths,
        plan_output=plan_output,
        exec_output=exec_output,
        tool_signature=tool_signature,
    )

    return AttemptSalvagePacket(
        packet_id=_packet_id(
            task,
            divergence_signature=divergence_signature,
            outcome=normalized_outcome,
        ),
        task_id=task.task_id,
        attempt=task.attempt,
        project_signature=project_signature,
        project_id=task.project_id,
        objective=task.objective,
        canonical_artifact_path=canonical_artifact_path,
        artifact_paths_touched=normalized_artifact_paths,
        plan_digest=_digest(plan_output),
        execution_digest=_digest(exec_output),
        verification_digest=_digest(verify_output) if verify_output else None,
        tool_signature=tool_signature,
        divergence_signature=divergence_signature,
        outcome=normalized_outcome,
        partial_value=_partial_value(exec_output, verify_output),
        discovered_constraints=discovered_constraints,
        unresolved_questions=_extract_questions(verify_output),
        best_next_step=_best_next_step(
            exec_output,
            verify_output,
            canonical_artifact_path,
            meaningful_progress_signal=meaningful_progress_signal,
        ),
        recommended_mode=(
            RetryMode.DETERMINISTIC_REVIEW
            if meaningful_progress_signal == "no_meaningful_progress"
            else
            RetryMode.RESUME_EXISTING_ARTIFACT
            if canonical_artifact_path
            else RetryMode.DETERMINISTIC_REVIEW
        ),
        meaningful_progress_signal=meaningful_progress_signal,
        llm_spend_class=(
            "deterministic_review"
            if meaningful_progress_signal == "no_meaningful_progress"
            else "broad"
        ),
        created_at=_stable_created_at(task),
    )


def _phase_output(task: RepairTask, phase_name: str) -> str:
    for phase_record in reversed(task.phases):
        if _phase_name(phase_record.phase) == phase_name:
            return (phase_record.output or "").strip()
    return ""


def _phase_name(phase: Any) -> str:
    value = getattr(phase, "value", phase)
    return str(value).lower()


def _extract_project_signature(task: RepairTask) -> str | None:
    meta = task.meta or {}
    resume_project = meta.get("resume_project")
    if isinstance(resume_project, dict):
        signature = resume_project.get("signature")
        if isinstance(signature, str) and signature.strip():
            return signature.strip()
    signature = meta.get("project_signature")
    if isinstance(signature, str) and signature.strip():
        return signature.strip()
    return None


def _tool_signature(tool_calls: Sequence[dict[str, Any]]) -> str:
    normalized = [_normalize_jsonish(tool_call) for tool_call in tool_calls]
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _digest(payload)


def _normalize_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_jsonish(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize_jsonish(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_jsonish(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_jsonish(item) for item in value)
    return value


def _meaningful_progress_signal(
    *,
    exec_output: str,
    verify_output: str,
    canonical_artifact_path: str | None,
    artifact_paths_touched: Sequence[str],
    outcome: AttemptOutcome,
) -> str:
    if canonical_artifact_path or artifact_paths_touched:
        return "artifact"
    context = "\n".join(part for part in (exec_output, verify_output) if part).strip()
    if _extract_constraints(exec_output, verify_output):
        return "constraint"
    if _extract_questions(verify_output):
        return "question"
    if outcome == AttemptOutcome.GUARD_STOPPED:
        return "blocker"
    if outcome == AttemptOutcome.DONE:
        return "completed"
    if context and not _generic_no_progress_output(context):
        return "evidence"
    return "no_meaningful_progress"


def _generic_no_progress_output(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return normalized in {
        "",
        "failed",
        "failure",
        "no output",
        "retry failed",
        "try again failed",
        "unknown error",
    }


def _best_next_step(
    exec_output: str,
    verify_output: str,
    canonical_artifact_path: str | None,
    *,
    meaningful_progress_signal: str = "",
) -> str:
    if meaningful_progress_signal == "no_meaningful_progress":
        return "No meaningful progress: stop broad retry, perform deterministic review, and change the next attempt frame before retrying."
    context = verify_output or exec_output
    artifact_hint = canonical_artifact_path or "the current artifact"
    if context:
        if _has_gap_language(context):
            return f"Repair the remaining gap in {artifact_hint} and rerun verification."
        if _has_question(context):
            return f"Answer the unresolved question and rerun verification for {artifact_hint}."
    if canonical_artifact_path:
        return f"Resume from {canonical_artifact_path} with the narrowest useful edit."
    return "Inspect the failed attempt, narrow the next edit, and rerun verification."


def _has_gap_language(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "missing",
            "still missing",
            "failed",
            "verification failed",
            "not found",
            "incomplete",
            "gap",
        )
    )


def _has_question(text: str) -> bool:
    return "?" in text


def _partial_value(exec_output: str, verify_output: str) -> str:
    source = exec_output.strip() or verify_output.strip()
    if not source:
        return ""
    fragments = [fragment.strip() for fragment in re.split(r"[.;\n]+", source) if fragment.strip()]
    return fragments[0] if fragments else source[:240].strip()


def _extract_constraints(exec_output: str, verify_output: str) -> list[str]:
    text = "\n".join(part for part in (exec_output, verify_output) if part).strip()
    if not text:
        return []
    candidates = _split_clauses(text)
    constraints: list[str] = []
    for clause in candidates:
        lowered = clause.lower()
        if any(token in lowered for token in ("must ", "cannot", "can't", "missing", "required", "needs ", "need ", "blocked by")):
            constraints.append(clause)
    return _dedupe_preserve_order(constraints)


def _extract_questions(verify_output: str) -> list[str]:
    if not verify_output.strip():
        return []
    clauses = _split_clauses(verify_output)
    questions = [clause for clause in clauses if clause.endswith("?")]
    return _dedupe_preserve_order(questions)


def _split_clauses(text: str) -> list[str]:
    clauses = [piece.strip() for piece in re.split(r"[.\n;]+", text) if piece.strip()]
    return clauses


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = value.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def build_divergence_signature(
    *,
    objective: str,
    project_signature: str | None,
    canonical_artifact_path: str | None,
    artifact_paths_touched: Sequence[str],
    plan_output: str,
    exec_output: str,
    tool_signature: str,
) -> str:
    payload = {
        "objective": objective.strip(),
        "project_signature": project_signature,
        "canonical_artifact_path": canonical_artifact_path,
        "artifact_paths_touched": _unique_sorted_paths(list(artifact_paths_touched)),
        "plan_digest": _digest(plan_output),
        "execution_digest": _digest(exec_output),
        "tool_signature": tool_signature,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _digest(serialized)


def _divergence_signature(
    *,
    objective: str,
    project_signature: str | None,
    canonical_artifact_path: str | None,
    artifact_paths_touched: Sequence[str],
    plan_output: str,
    exec_output: str,
    tool_signature: str,
) -> str:
    """Backward-compatible alias for the public divergence signature helper."""

    return build_divergence_signature(
        objective=objective,
        project_signature=project_signature,
        canonical_artifact_path=canonical_artifact_path,
        artifact_paths_touched=artifact_paths_touched,
        plan_output=plan_output,
        exec_output=exec_output,
        tool_signature=tool_signature,
    )


def _unique_sorted_paths(paths: Sequence[str]) -> list[str]:
    normalized = sorted({path.strip() for path in paths if isinstance(path, str) and path.strip()})
    return normalized


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _coerce_outcome(outcome: str | AttemptOutcome) -> AttemptOutcome:
    if isinstance(outcome, AttemptOutcome):
        return outcome
    return AttemptOutcome(str(outcome))


def _packet_id(
    task: RepairTask,
    *,
    divergence_signature: str,
    outcome: AttemptOutcome,
) -> UUID:
    seed = f"{task.task_id}:{task.attempt}:{outcome.value}:{divergence_signature}"
    return uuid5(NAMESPACE_URL, seed)


def _stable_created_at(task: RepairTask) -> datetime:
    for phase_record in reversed(task.phases):
        if phase_record.ended_at is not None:
            return phase_record.ended_at
        if phase_record.started_at is not None:
            return phase_record.started_at
    if task.updated_at is not None:
        return task.updated_at
    return task.created_at
