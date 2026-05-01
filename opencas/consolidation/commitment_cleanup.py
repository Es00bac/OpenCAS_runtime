"""Commitment deduplication and chat-recovery helpers for nightly consolidation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.autonomy.commitment_extraction import extract_self_commitments
from opencas.autonomy.models import WorkObject, WorkStage


@dataclass(frozen=True)
class CommitmentRecoveryCandidate:
    """A normalized candidate recovered from recent conversational episodes."""

    candidate_id: str
    content: str
    source_sentence: str
    session_id: str
    episode_id: str
    created_at: datetime
    previous_user_turn: str
    role_source: str


async def consolidate_commitments(engine: Any, similarity_threshold: float = 0.75) -> Dict[str, Any]:
    """Deduplicate active and blocked commitments via embedding clustering plus merge logic."""
    if not engine.commitment_store:
        return {"clusters_formed": 0, "commitments_merged": 0, "work_objects_created": 0}

    active = await engine.commitment_store.list_by_status(CommitmentStatus.ACTIVE)
    blocked = await engine.commitment_store.list_by_status(CommitmentStatus.BLOCKED)
    commitments = active + blocked

    if not commitments:
        return {"clusters_formed": 0, "commitments_merged": 0, "work_objects_created": 0}

    vectors: List[Optional[np.ndarray]] = []
    for commitment in commitments:
        record = await engine.embeddings.embed(commitment.content, task_type="retrieval_query")
        vectors.append(np.array(record.vector, dtype=np.float32))

    clusters: List[List[int]] = []
    used: set[int] = set()
    for i, vi in enumerate(vectors):
        if i in used or vi is None:
            continue
        cluster = [i]
        used.add(i)
        norm_i = float(np.linalg.norm(vi)) or 1.0
        for j in range(i + 1, len(vectors)):
            if j in used or vectors[j] is None:
                continue
            vj = vectors[j]
            norm_j = float(np.linalg.norm(vj)) or 1.0
            sim = float(np.dot(vi, vj) / (norm_i * norm_j))
            if sim >= similarity_threshold:
                cluster.append(j)
                used.add(j)
        clusters.append(cluster)

    commitments_merged = 0
    for cluster_indices in clusters:
        if len(cluster_indices) < 2:
            continue
        cluster_commits = [commitments[idx] for idx in cluster_indices]
        refined_clusters = refine_commitment_cluster(cluster_commits)
        for refined_cluster in refined_clusters:
            if len(refined_cluster) < 2:
                continue

            survivor_idx, rationale = await pick_commitment_survivor(engine, refined_cluster)
            survivor = refined_cluster[survivor_idx]

            merged_work_ids: set[str] = set(survivor.linked_work_ids)
            merged_task_ids: set[str] = set(survivor.linked_task_ids)
            for idx, commitment in enumerate(refined_cluster):
                if idx == survivor_idx:
                    continue
                merged_work_ids.update(commitment.linked_work_ids)
                merged_task_ids.update(commitment.linked_task_ids)
                await engine.commitment_store.update_status(
                    str(commitment.commitment_id), CommitmentStatus.ABANDONED
                )
                commitments_merged += 1

            survivor.linked_work_ids = list(merged_work_ids)
            survivor.linked_task_ids = list(merged_task_ids)
            survivor.status = merged_commitment_status(refined_cluster)
            survivor.meta = dict(survivor.meta or {})
            survivor.meta["consolidation_merged_ids"] = [
                str(commitment.commitment_id)
                for commitment in refined_cluster
                if commitment.commitment_id != survivor.commitment_id
            ]
            survivor.meta["consolidation_status_basis"] = survivor.status.value
            survivor.meta["consolidation_merge_rationale"] = rationale
            await engine.commitment_store.save(survivor)

    work_objects_created = 0
    if engine.work_store:
        active_after = await engine.commitment_store.list_by_status(CommitmentStatus.ACTIVE)
        for commitment in active_after:
            if commitment.linked_work_ids:
                continue
            work = WorkObject(
                content=commitment.content,
                stage=WorkStage.MICRO_TASK,
                commitment_id=str(commitment.commitment_id),
            )
            await engine.work_store.save(work)
            await engine.commitment_store.link_work(str(commitment.commitment_id), str(work.work_id))
            work_objects_created += 1

    return {
        "clusters_formed": len([cluster for cluster in clusters if len(cluster) >= 2]),
        "commitments_merged": commitments_merged,
        "work_objects_created": work_objects_created,
    }


async def llm_pick_commitment_survivor(engine: Any, cluster: List[Commitment]) -> Optional[int]:
    """Ask the LLM which commitment in a cluster should survive."""
    budget_consumer = getattr(engine, "_consume_consolidation_llm_budget", None)
    if callable(budget_consumer) and not budget_consumer():
        return None
    numbered = "\n".join(
        f"[{i}] {c.content} (status={c.status.value}, priority={c.priority}, linked_work={len(c.linked_work_ids)})"
        for i, c in enumerate(cluster)
    )
    prompt = (
        "Several self-commitments have been found to be duplicates or near-duplicates. "
        "Pick the single best one to keep as the survivor. Reply with ONLY the index number "
        "(e.g. \"0\"). If any are already completed or subsumed by another, prefer the most "
        "comprehensive one.\n\n"
        + numbered
    )
    messages = [
        {"role": "system", "content": "You are a consolidation assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        try:
            response = await engine.llm.chat_completion(
                messages,
                complexity="high",
                source="consolidation",
            )
        except TypeError:
            response = await engine.llm.chat_completion(
                messages,
                source="consolidation",
            )
        text = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        idx = int(text.strip("[]").split()[0])
        if 0 <= idx < len(cluster):
            return idx
    except Exception:
        pass
    return None


async def extract_commitments_from_chat_logs(engine: Any) -> int:
    """Scan recent conversational episodes for assistant commitments missed earlier."""
    if not engine.commitment_store:
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(hours=72)
    try:
        episodes = await engine.memory.list_non_compacted_episodes(limit=200)
    except Exception:
        return 0

    candidates = collect_commitment_recovery_candidates(episodes, cutoff=cutoff)
    if not candidates:
        return 0
    budget_consumer = getattr(engine, "_consume_consolidation_llm_budget", None)
    if callable(budget_consumer) and not budget_consumer():
        return 0

    numbered_candidates = "\n\n".join(
        (
            f"[{candidate.candidate_id}] session={candidate.session_id} "
            f"time={candidate.created_at.isoformat()} role_source={candidate.role_source}\n"
            f"normalized_commitment: {candidate.content}\n"
            f"source_sentence: {candidate.source_sentence}\n"
            f"previous_user_turn: {candidate.previous_user_turn or '[none]'}"
        )
        for candidate in candidates[:50]
    )
    prompt = (
        "Review these normalized candidate self-commitments recovered from recent assistant turns. "
        "Each candidate already comes from a turn-level extraction pass and includes local session context. "
        "Keep only candidates that represent real assistant promises or durable follow-up commitments.\n\n"
        'Return a JSON array of objects with fields: candidate_id, content, inferred_status ("active", '
        '"blocked", or "completed"), reason.\n'
        "If no commitments are found, return an empty array: []\n\n"
        + numbered_candidates
    )
    prompt_limit = None
    prompt_limiter = getattr(engine, "_budget_prompt_limit", None)
    if callable(prompt_limiter):
        prompt_limit = prompt_limiter()
    if prompt_limit is not None and len(prompt) > prompt_limit:
        prompt = prompt[:prompt_limit].rstrip()
    messages = [
        {"role": "system", "content": "You are a consolidation assistant. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    extracted_raw: List[Dict[str, Any]] = []
    try:
        try:
            response = await engine.llm.chat_completion(
                messages,
                complexity="high",
                source="consolidation",
            )
        except TypeError:
            response = await engine.llm.chat_completion(
                messages,
                source="consolidation",
            )
        text = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        extracted_raw = json.loads(text)
        if not isinstance(extracted_raw, list):
            extracted_raw = []
    except Exception:
        return 0

    if not extracted_raw:
        return 0

    existing_active = await engine.commitment_store.list_by_status(CommitmentStatus.ACTIVE)
    existing_blocked = await engine.commitment_store.list_by_status(CommitmentStatus.BLOCKED)
    existing = existing_active + existing_blocked
    existing_keys = {commitment_key(c.content) for c in existing}
    candidate_index = {candidate.candidate_id: candidate for candidate in candidates}

    created = 0
    for item in extracted_raw:
        content = item.get("content", "").strip()
        if not content:
            continue
        candidate = candidate_index.get(str(item.get("candidate_id", "")).strip())
        if candidate is not None and not are_conservative_duplicate_contents(content, candidate.content):
            candidate = None
        if candidate is None:
            matching_candidates = [
                existing_candidate
                for existing_candidate in candidates
                if are_conservative_duplicate_contents(content, existing_candidate.content)
            ]
            if len(matching_candidates) == 1:
                candidate = matching_candidates[0]
        if candidate is None:
            continue

        content_key = commitment_key(content)
        is_dup = content_key in existing_keys or any(
            are_conservative_duplicate_contents(content, existing_commitment.content)
            for existing_commitment in existing
        )
        if is_dup:
            continue

        raw_status = item.get("inferred_status", "active").lower()
        try:
            status = CommitmentStatus(raw_status)
        except ValueError:
            status = CommitmentStatus.ACTIVE

        commitment = Commitment(
            content=content,
            status=status,
            tags=["chat_extraction"],
            meta={
                "source": "nightly_consolidation",
                "reason": item.get("reason", ""),
                "source_session_id": candidate.session_id,
                "source_episode_id": candidate.episode_id,
                "source_sentence": candidate.source_sentence,
                "previous_user_turn": candidate.previous_user_turn,
                "role_source": candidate.role_source,
                "recovery_method": "session_window_review",
            },
        )
        await engine.commitment_store.save(commitment)
        existing_keys.add(content_key)
        existing.append(commitment)
        created += 1

        if engine.work_store and status == CommitmentStatus.ACTIVE:
            work = WorkObject(
                content=content,
                stage=WorkStage.MICRO_TASK,
                commitment_id=str(commitment.commitment_id),
            )
            await engine.work_store.save(work)
            await engine.commitment_store.link_work(
                str(commitment.commitment_id), str(work.work_id)
            )

    return created


def merged_commitment_status(cluster: List[Commitment]) -> CommitmentStatus:
    """Preserve blocked commitments unless the merged cluster still has active work."""
    if any(commitment.status == CommitmentStatus.ACTIVE for commitment in cluster):
        return CommitmentStatus.ACTIVE
    return CommitmentStatus.BLOCKED


def collect_commitment_recovery_candidates(
    episodes: List[Any],
    cutoff: datetime,
) -> List[CommitmentRecoveryCandidate]:
    """Collect normalized candidates from recent assistant turns with session context."""
    session_groups: Dict[str, List[Any]] = {}
    for episode in episodes:
        if episode.created_at < cutoff:
            continue
        if str(episode.kind.value) != "turn" or not episode.content:
            continue
        session_key = episode.session_id or "__no_session__"
        session_groups.setdefault(session_key, []).append(episode)

    candidates: List[CommitmentRecoveryCandidate] = []
    seen_keys: set[Tuple[str, str]] = set()
    candidate_counter = 1
    for session_id, group in session_groups.items():
        ordered = sorted(group, key=lambda episode: episode.created_at)
        last_user_turn = ""
        for episode in ordered:
            payload = getattr(episode, "payload", {}) or {}
            role = str(payload.get("role", "")).lower()
            role_source = "payload"
            if role == "user":
                last_user_turn = episode.content
                continue
            if role != "assistant":
                captures = extract_self_commitments(episode.content)
                if not captures:
                    continue
                role_source = "roleless_fallback"
            else:
                captures = extract_self_commitments(episode.content)
                if not captures:
                    continue

            for capture in captures:
                content_key = commitment_key(capture.content)
                dedup_key = (session_id, content_key)
                if not content_key or dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                candidates.append(
                    CommitmentRecoveryCandidate(
                        candidate_id=str(candidate_counter),
                        content=capture.content,
                        source_sentence=capture.source_sentence,
                        session_id=session_id,
                        episode_id=str(episode.episode_id),
                        created_at=episode.created_at,
                        previous_user_turn=last_user_turn[:240],
                        role_source=role_source,
                    )
                )
                candidate_counter += 1

    candidates.sort(key=lambda candidate: candidate.created_at, reverse=True)
    return candidates


async def pick_commitment_survivor(
    engine: Any,
    cluster: List[Commitment],
) -> Tuple[int, str]:
    """Prefer heuristic survivor selection for obvious duplicates, otherwise ask the LLM."""
    if cluster_has_obvious_duplicate(cluster):
        return heuristic_survivor_index(cluster), "heuristic_exact_duplicate"

    survivor_idx = await llm_pick_commitment_survivor(engine, cluster)
    if survivor_idx is not None:
        return survivor_idx, "llm_disambiguation"
    return heuristic_survivor_index(cluster), "heuristic_fallback"


def refine_commitment_cluster(cluster: List[Commitment]) -> List[List[Commitment]]:
    """Split embedding clusters into conservative duplicate subclusters before merging."""
    refined: List[List[Commitment]] = []
    used: set[int] = set()
    for i, commitment in enumerate(cluster):
        if i in used:
            continue
        subcluster = [commitment]
        used.add(i)
        for j in range(i + 1, len(cluster)):
            if j in used:
                continue
            if are_conservative_duplicates(commitment, cluster[j]):
                subcluster.append(cluster[j])
                used.add(j)
        refined.append(subcluster)
    return refined


def cluster_has_obvious_duplicate(cluster: List[Commitment]) -> bool:
    if len(cluster) < 2:
        return False
    keys = {commitment_key(commitment.content) for commitment in cluster}
    return len(keys) == 1


def heuristic_survivor_index(cluster: List[Commitment]) -> int:
    def sort_key(commitment: Commitment) -> tuple:
        active_bonus = 1 if commitment.status == CommitmentStatus.ACTIVE else 0
        return (
            active_bonus,
            len(commitment.linked_work_ids),
            len(commitment.linked_task_ids),
            commitment.priority,
            commitment.updated_at.timestamp(),
        )

    return max(range(len(cluster)), key=lambda i: sort_key(cluster[i]))


def are_conservative_duplicates(left: Commitment, right: Commitment) -> bool:
    return are_conservative_duplicate_contents(left.content, right.content)


def are_conservative_duplicate_contents(left_content: str, right_content: str) -> bool:
    left_key = commitment_key(left_content)
    right_key = commitment_key(right_content)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    if left_key in right_key or right_key in left_key:
        return True

    left_tokens = set(left_key.split())
    right_tokens = set(right_key.split())
    if not left_tokens or not right_tokens:
        return False
    overlap = left_tokens & right_tokens
    jaccard = len(overlap) / len(left_tokens | right_tokens)
    return len(overlap) >= 3 and jaccard >= 0.8


def commitment_key(content: str) -> str:
    normalized = content.strip().lower()
    normalized = normalized.removeprefix("self-commitment:")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()
