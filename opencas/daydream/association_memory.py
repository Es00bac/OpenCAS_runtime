"""Semantic memory records for retrievable daydream associations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
from uuid import NAMESPACE_URL, UUID, uuid5

from opencas.memory import Memory

from .models import DaydreamReflection, DaydreamThought, DaydreamThoughtRoute


DAYDREAM_ASSOCIATION_TAG = "daydream_association"
DAYDREAM_RETRIEVABLE_TAG = "retrievable_thought"


@dataclass(frozen=True)
class DaydreamAssociationPersistResult:
    """Result of persisting a reflection as a semantic association."""

    memory: Memory
    created: bool


def daydream_association_memory_id(reflection: DaydreamReflection) -> UUID:
    """Return the deterministic semantic-memory id for a reflection."""
    return uuid5(
        NAMESPACE_URL,
        f"opencas:daydream-association:{reflection.reflection_id}",
    )


def should_persist_daydream_association(reflection: DaydreamReflection) -> bool:
    """True when a daydream has enough content to be recallable later."""
    return any(
        str(part or "").strip()
        for part in (
            reflection.spark_content,
            reflection.recollection,
            reflection.interpretation,
            reflection.synthesis,
            reflection.open_question,
            *(thought.summary for thought in reflection.thoughts),
        )
    )


def build_daydream_association_memory(
    reflection: DaydreamReflection,
    *,
    embedding_id: Optional[str] = None,
) -> Memory:
    """Build the semantic memory used for later associative recall."""
    return Memory(
        memory_id=daydream_association_memory_id(reflection),
        created_at=reflection.created_at,
        updated_at=datetime.now(timezone.utc),
        content=build_daydream_association_content(reflection),
        tags=daydream_association_tags(reflection),
        source_episode_ids=[],
        embedding_id=embedding_id,
        salience=daydream_association_salience(reflection),
        confidence_score=daydream_association_confidence(reflection),
    )


def build_daydream_association_content(reflection: DaydreamReflection) -> str:
    """Render a reflection as a compact downstream recall packet."""
    lines = [
        "Background daydream association (not a factual claim): "
        "this is a thought Bulma generated while not directly working on the task.",
        f"Reflection id: {reflection.reflection_id}",
        f"Spark: {reflection.spark_content}",
    ]
    if reflection.synthesis:
        lines.append(f"Synthesis: {reflection.synthesis}")
    if reflection.interpretation:
        lines.append(f"Interpretation: {reflection.interpretation}")
    if reflection.recollection:
        lines.append(f"Recollection: {reflection.recollection}")
    if reflection.open_question:
        lines.append(f"Open question: {reflection.open_question}")
    if reflection.changed_self_view:
        lines.append(f"Changed self-view: {reflection.changed_self_view}")
    if reflection.fascination_thread:
        lines.append(f"Fascination thread: {reflection.fascination_thread}")

    thoughts = list(_non_empty_thought_lines(reflection.thoughts))
    if thoughts:
        lines.append("Thoughts:")
        lines.extend(f"- {item}" for item in thoughts)

    lines.append(
        "Possible downstream use: recall this as an idea seed, caution, rejected path, "
        "or hypothesis when related future work appears."
    )
    lines.append(
        "Scores: "
        f"keeper={reflection.keeper}; "
        f"alignment={reflection.alignment_score:.2f}; "
        f"novelty={reflection.novelty_score:.2f}; "
        f"max_risk={_max_thought_score(reflection.thoughts, 'risk'):.2f}."
    )
    return "\n".join(line for line in lines if line.strip())


def daydream_association_tags(reflection: DaydreamReflection) -> list[str]:
    """Return tags that keep association memories separate from keeper memories."""
    tags = [
        "daydream",
        DAYDREAM_ASSOCIATION_TAG,
        DAYDREAM_RETRIEVABLE_TAG,
        "background_daydream",
        "daydream_keeper_association" if reflection.keeper else "daydream_non_keeper",
    ]
    if reflection.open_question:
        tags.append("daydream_open_question")
    if _is_caution(reflection):
        tags.append("daydream_caution")
    if _has_inner_dialogue(reflection.thoughts):
        tags.append("daydream_self_dialogue")
    if reflection.fascination_thread:
        tags.append(_tag_value("daydream_thread", reflection.fascination_thread))

    for thought in reflection.thoughts[:6]:
        tags.append(_tag_value("daydream_kind", getattr(thought.kind, "value", thought.kind)))
        tags.append(_tag_value("daydream_route", getattr(thought.route, "value", thought.route)))

    deduped: list[str] = []
    for tag in tags:
        if tag and tag not in deduped:
            deduped.append(tag)
    return deduped


def daydream_association_salience(reflection: DaydreamReflection) -> float:
    """Compute bounded salience so associations are retrievable without dominating."""
    novelty = max(float(reflection.novelty_score or 0.0), _max_thought_score(reflection.thoughts, "novelty"))
    usefulness = _max_thought_score(reflection.thoughts, "usefulness")
    risk = _max_thought_score(reflection.thoughts, "risk")
    salience = 0.45 + (novelty * 1.4) + (usefulness * 0.9) + (float(reflection.alignment_score or 0.0) * 0.6)
    if reflection.keeper:
        salience += 0.5
    if risk >= 0.5 or _has_discard_route(reflection.thoughts):
        salience += 0.25
    return round(max(0.35, min(4.5, salience)), 3)


def daydream_association_confidence(reflection: DaydreamReflection) -> float:
    """Confidence reflects recall reliability, not truth of the thought."""
    confidences = [
        float(getattr(thought, "confidence", 0.0) or 0.0)
        for thought in reflection.thoughts
    ]
    if confidences:
        confidence = sum(confidences) / len(confidences)
    else:
        confidence = 0.55
    if reflection.keeper:
        confidence += 0.1
    if _is_caution(reflection):
        confidence = min(confidence, 0.72)
    return round(max(0.25, min(0.9, confidence)), 3)


async def persist_daydream_association_memory(
    *,
    memory_store: Any,
    embeddings: Any,
    reflection: DaydreamReflection,
) -> Optional[DaydreamAssociationPersistResult]:
    """Persist *reflection* as a deterministic semantic association memory."""
    if memory_store is None or not should_persist_daydream_association(reflection):
        return None

    memory_id = daydream_association_memory_id(reflection)
    created = True
    get_memory = getattr(memory_store, "get_memory", None)
    if callable(get_memory):
        existing = await get_memory(str(memory_id))
        created = existing is None

    content = build_daydream_association_content(reflection)
    embedding_id: Optional[str] = None
    if embeddings is not None and hasattr(embeddings, "embed"):
        try:
            embed_record = await embeddings.embed(
                content,
                meta={
                    "origin": "daydream_association",
                    "reflection_id": str(reflection.reflection_id),
                    "keeper": reflection.keeper,
                },
                task_type="daydream_association_memory",
            )
            embedding_id = getattr(embed_record, "source_hash", None)
        except Exception:
            embedding_id = None

    memory = build_daydream_association_memory(reflection, embedding_id=embedding_id)
    await memory_store.save_memory(memory)
    return DaydreamAssociationPersistResult(memory=memory, created=created)


def attach_daydream_association_context(
    reflection: DaydreamReflection,
    memory: Memory,
) -> None:
    """Record semantic association lineage on the reflection itself."""
    reflection.experience_context = {
        **(reflection.experience_context or {}),
        "association_memory_id": str(memory.memory_id),
        "association_memory_tags": list(memory.tags),
        "association_memory_salience": memory.salience,
    }


def _non_empty_thought_lines(thoughts: Iterable[DaydreamThought]) -> Iterable[str]:
    for thought in thoughts:
        parts = []
        route = getattr(thought.route, "value", thought.route)
        kind = getattr(thought.kind, "value", thought.kind)
        if kind:
            parts.append(f"kind={kind}")
        if route:
            parts.append(f"route={route}")
        for label, value in (
            ("summary", thought.summary),
            ("question", thought.question),
            ("hypothesis", thought.hypothesis),
            ("possible_experiment", thought.possible_experiment),
            ("imaginative_branch", thought.imaginative_branch),
            ("practical_branch", thought.practical_branch),
            ("bridge", thought.bridge),
            ("self_work_intent", thought.self_work_intent),
        ):
            text = str(value or "").strip()
            if text:
                parts.append(f"{label}={text}")
        dialogue_lines = list(_dialogue_lines(thought))
        if dialogue_lines:
            parts.append("Self-dialogue: " + " | ".join(dialogue_lines))
        if parts:
            parts.append(
                "scores="
                f"usefulness:{thought.usefulness:.2f},"
                f"novelty:{thought.novelty:.2f},"
                f"confidence:{thought.confidence:.2f},"
                f"risk:{thought.risk:.2f}"
            )
            yield "; ".join(parts)


def _max_thought_score(thoughts: Iterable[DaydreamThought], field: str) -> float:
    values = [float(getattr(thought, field, 0.0) or 0.0) for thought in thoughts]
    return max(values) if values else 0.0


def _has_discard_route(thoughts: Iterable[DaydreamThought]) -> bool:
    return any(getattr(thought, "route", None) == DaydreamThoughtRoute.DISCARD for thought in thoughts)


def _has_inner_dialogue(thoughts: Iterable[DaydreamThought]) -> bool:
    return any(bool(getattr(thought, "inner_dialogue", []) or []) for thought in thoughts)


def _dialogue_lines(thought: DaydreamThought) -> Iterable[str]:
    for turn in list(getattr(thought, "inner_dialogue", []) or [])[:5]:
        text = str(getattr(turn, "text", "") or "").strip()
        if not text:
            continue
        voice = str(getattr(turn, "voice", "") or "").strip()
        stance = str(getattr(turn, "stance", "") or "").strip()
        label = voice or stance
        if voice and stance:
            label = f"{voice}/{stance}"
        yield f"{label}: {text}" if label else text


def _is_caution(reflection: DaydreamReflection) -> bool:
    return _max_thought_score(reflection.thoughts, "risk") >= 0.5 or _has_discard_route(reflection.thoughts)


def _tag_value(prefix: str, raw: Any) -> str:
    value = str(raw or "").strip().lower()
    value = "".join(char if char.isalnum() else "_" for char in value)
    value = "_".join(part for part in value.split("_") if part)
    return f"{prefix}:{value[:48]}" if value else prefix
