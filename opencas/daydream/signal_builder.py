"""Build possibility signals from daydream reflections."""

from __future__ import annotations

from typing import Iterable, List

from .models import DaydreamReflection, DaydreamThought, DaydreamThoughtRoute
from .signals import (
    ContactPosture,
    PossibilitySignal,
    PossibilitySignalRoute,
    SelfWorkKind,
)


class DaydreamSignalBuilder:
    """Normalize daydream thoughts into inspectable possibility signals."""

    def build_from_reflection(
        self,
        reflection: DaydreamReflection,
        *,
        source_mode: str = "waking_daydream",
    ) -> List[PossibilitySignal]:
        signals: list[PossibilitySignal] = []
        for index, thought in enumerate(list(reflection.thoughts or [])):
            summary = _first_non_empty(
                thought.summary,
                thought.question,
                thought.hypothesis,
                thought.possible_experiment,
                reflection.synthesis,
                reflection.spark_content,
            )
            if not summary:
                continue
            original_route = self._mapped_route_for_thought(thought)
            route = self._route_for_thought(thought)
            evidence_ids = _evidence_ids(thought)
            meta = {
                "route_schema_version": 2,
                "thought_kind": thought.kind.value,
                "thought_route": thought.route.value,
                "keeper": reflection.keeper,
                "fascination_thread": reflection.fascination_thread,
                "inner_dialogue_turn_count": len(thought.inner_dialogue or []),
                "inner_dialogue_voices": _dialogue_voices(thought),
            }
            if original_route != route:
                meta["original_route"] = original_route.value
            signals.append(
                PossibilitySignal(
                    source_reflection_id=str(reflection.reflection_id),
                    source_thought_index=index,
                    source_mode=source_mode,
                    summary=summary,
                    imaginative_branch=thought.imaginative_branch,
                    practical_branch=thought.practical_branch,
                    bridge=thought.bridge,
                    novelty=thought.novelty,
                    usefulness=thought.usefulness,
                    confidence=thought.confidence,
                    risk=thought.risk,
                    grounding=list(thought.grounding or []),
                    evidence_ids=evidence_ids,
                    suggested_route=route,
                    suggested_handler=thought.suggested_handler,
                    contact_posture=_contact_posture(thought.contact_posture, route),
                    self_work_kind=_self_work_kind(route),
                    self_work_intent=thought.self_work_intent,
                    route_reason=self._route_reason(thought, route),
                    meta=meta,
                )
            )
        return signals

    def _mapped_route_for_thought(self, thought: DaydreamThought) -> PossibilitySignalRoute:
        mapped = _route_from_handler(thought.suggested_handler)
        if mapped is None:
            mapped = _route_from_thought_route(thought.route)
        return mapped

    def _route_for_thought(self, thought: DaydreamThought) -> PossibilitySignalRoute:
        mapped = self._mapped_route_for_thought(thought)
        if thought.confidence < 0.25:
            return PossibilitySignalRoute.INCUBATE
        if not thought.practical_branch.strip() and thought.usefulness < 0.4:
            return PossibilitySignalRoute.INCUBATE
        if thought.risk >= 0.75 and mapped != PossibilitySignalRoute.ASK_USER:
            return PossibilitySignalRoute.INCUBATE
        if _requires_grounding(mapped, thought):
            return PossibilitySignalRoute.INCUBATE
        return mapped

    @staticmethod
    def _route_reason(
        thought: DaydreamThought,
        route: PossibilitySignalRoute,
    ) -> str:
        if route == PossibilitySignalRoute.INCUBATE and thought.confidence < 0.25:
            return "confidence below promotion floor"
        if (
            route == PossibilitySignalRoute.INCUBATE
            and not thought.practical_branch.strip()
            and thought.usefulness < 0.4
        ):
            return "no practical branch and usefulness below promotion floor"
        if route == PossibilitySignalRoute.INCUBATE and thought.risk >= 0.75:
            return "risk requires incubation before action"
        mapped = _route_from_handler(thought.suggested_handler) or _route_from_thought_route(thought.route)
        if route == PossibilitySignalRoute.INCUBATE and _requires_grounding(mapped, thought):
            return "high-action route requires grounding or evidence ids"
        if thought.suggested_handler:
            return f"handler suggested {thought.suggested_handler}"
        return f"mapped from thought route {thought.route.value}"


def _route_from_handler(value: str) -> PossibilitySignalRoute | None:
    normalized = str(value or "").strip().lower()
    mapping = {
        "thread": PossibilitySignalRoute.THREAD,
        "self_note": PossibilitySignalRoute.SELF_NOTE,
        "self_experiment": PossibilitySignalRoute.SELF_EXPERIMENT,
        "self_prototype": PossibilitySignalRoute.SELF_PROTOTYPE,
        "research": PossibilitySignalRoute.RESEARCH,
        "work_candidate": PossibilitySignalRoute.WORK_CANDIDATE,
        "ask_user": PossibilitySignalRoute.ASK_USER,
        "compost": PossibilitySignalRoute.COMPOST,
        "discard": PossibilitySignalRoute.DISCARD,
        "incubate": PossibilitySignalRoute.INCUBATE,
    }
    return mapping.get(normalized)


def _route_from_thought_route(route: DaydreamThoughtRoute) -> PossibilitySignalRoute:
    mapping = {
        DaydreamThoughtRoute.ACT_NOW: PossibilitySignalRoute.WORK_CANDIDATE,
        DaydreamThoughtRoute.DEEP_THINK: PossibilitySignalRoute.SELF_NOTE,
        DaydreamThoughtRoute.ASK_USER: PossibilitySignalRoute.ASK_USER,
        DaydreamThoughtRoute.RESEARCH: PossibilitySignalRoute.RESEARCH,
        DaydreamThoughtRoute.INCUBATE: PossibilitySignalRoute.INCUBATE,
        DaydreamThoughtRoute.DISCARD: PossibilitySignalRoute.DISCARD,
    }
    return mapping.get(route, PossibilitySignalRoute.INCUBATE)


def _contact_posture(value: str, route: PossibilitySignalRoute) -> ContactPosture:
    normalized = str(value or "").strip().lower()
    try:
        return ContactPosture(normalized)
    except ValueError:
        if route == PossibilitySignalRoute.ASK_USER:
            return ContactPosture.ASK_FIRST
        return ContactPosture.SILENT


def _self_work_kind(route: PossibilitySignalRoute) -> SelfWorkKind:
    if route == PossibilitySignalRoute.SELF_NOTE:
        return SelfWorkKind.NOTE
    if route == PossibilitySignalRoute.RESEARCH:
        return SelfWorkKind.RESEARCH
    if route == PossibilitySignalRoute.SELF_EXPERIMENT:
        return SelfWorkKind.EXPERIMENT
    if route == PossibilitySignalRoute.SELF_PROTOTYPE:
        return SelfWorkKind.PROTOTYPE
    if route == PossibilitySignalRoute.COMPOST:
        return SelfWorkKind.COMPOST
    return SelfWorkKind.NONE


def _first_non_empty(*values: str) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _evidence_ids(thought: DaydreamThought) -> list[str]:
    ids: list[str] = []
    for item in thought.grounding or []:
        ids.extend(str(eid) for eid in getattr(item, "evidence_ids", []) or [])
    return _dedupe(ids)


def _dialogue_voices(thought: DaydreamThought) -> list[str]:
    return _dedupe(
        str(turn.voice).strip()
        for turn in list(thought.inner_dialogue or [])
        if str(turn.voice).strip()
    )


def _requires_grounding(route: PossibilitySignalRoute, thought: DaydreamThought) -> bool:
    if route not in {
        PossibilitySignalRoute.SELF_EXPERIMENT,
        PossibilitySignalRoute.SELF_PROTOTYPE,
        PossibilitySignalRoute.RESEARCH,
        PossibilitySignalRoute.WORK_CANDIDATE,
        PossibilitySignalRoute.ASK_USER,
    }:
        return False
    if thought.confidence < 0.75:
        return False
    return not (thought.grounding or _evidence_ids(thought))


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
