"""Default consumers for the runtime cognition bus."""

from __future__ import annotations

from typing import Any

from opencas.somatic.models import AffectState, PrimaryEmotion, SocialTarget
from opencas.wellbeing import WellbeingEvent

from .cognition_bus import AffectiveEvent, CognitiveEvent


_AFFECT_PROFILES: dict[str, tuple[PrimaryEmotion, float, float, float]] = {
    "affective.user_frustration": (PrimaryEmotion.ANNOYED, -0.45, 0.72, 0.55),
    "affective.user_gratitude": (PrimaryEmotion.TRUST, 0.42, 0.38, 0.7),
    "affective.goal_blocked": (PrimaryEmotion.CONCERNED, -0.25, 0.58, 0.5),
    "affective.tool_rejected": (PrimaryEmotion.CONCERNED, -0.3, 0.62, 0.45),
}


def register_runtime_cognition_subscribers(runtime: Any) -> None:
    """Subscribe existing engines to shared cognitive events."""
    bus = getattr(runtime, "cognition_bus", None)
    if bus is None:
        return
    for kind in _AFFECT_PROFILES:
        bus.subscribe(
            kind,
            lambda event, runtime=runtime: _consume_affective_event(runtime, event),
            name=f"runtime_affective_consumer:{kind}",
        )
    bus.subscribe(
        "ladder.health_summary",
        lambda event, runtime=runtime: _append_wellbeing_bus_event(
            runtime,
            event,
            event_type="ladder_health_summary",
            summary="creative ladder health summary",
        ),
        name="wellbeing:ladder_health_summary",
    )
    bus.subscribe(
        "narrator.redundancy_observed",
        lambda event, runtime=runtime: _append_wellbeing_bus_event(
            runtime,
            event,
            event_type="narrator_redundancy_observed",
            summary="self-knowledge narrator skipped redundant recent arc",
        ),
        name="wellbeing:narrator_redundancy_observed",
    )


async def _consume_affective_event(runtime: Any, event: CognitiveEvent) -> None:
    if not isinstance(event, AffectiveEvent):
        return
    await _nudge_somatic(runtime, event)
    await _nudge_relational(runtime, event)
    await _append_wellbeing_bus_event(
        runtime,
        event,
        event_type="affective_event_observed",
        summary=f"shared affective event observed: {event.kind}",
    )


async def _nudge_somatic(runtime: Any, event: AffectiveEvent) -> None:
    somatic = getattr(getattr(runtime, "ctx", None), "somatic", None)
    if somatic is None:
        return
    emotion, valence, arousal, certainty = _AFFECT_PROFILES.get(
        event.kind,
        (PrimaryEmotion.NEUTRAL, 0.0, 0.5, 0.5),
    )
    affect = AffectState(
        primary_emotion=emotion,
        valence=valence,
        arousal=arousal,
        certainty=certainty,
        intensity=max(0.0, min(1.0, float(event.magnitude or 0.0))),
        social_target=SocialTarget.USER,
        emotion_tags=[event.kind],
    )
    somatic.nudge_from_appraisal(
        affect,
        intensity_scale=max(0.2, min(0.6, affect.intensity)),
    )
    if getattr(somatic, "store", None) is not None:
        await somatic.record_snapshot(
            source=event.kind,
            trigger_event_id=event.event_id,
        )


async def _nudge_relational(runtime: Any, event: AffectiveEvent) -> None:
    relational = getattr(getattr(runtime, "ctx", None), "relational", None)
    recorder = getattr(relational, "record_affective_event", None)
    if callable(recorder):
        await recorder(event)


async def _append_wellbeing_bus_event(
    runtime: Any,
    event: CognitiveEvent,
    *,
    event_type: str,
    summary: str,
) -> None:
    store = getattr(runtime, "wellbeing_store", None) or getattr(
        getattr(runtime, "ctx", None),
        "wellbeing_store",
        None,
    )
    if store is None:
        return
    await store.append_event(
        WellbeingEvent(
            event_type=event_type,
            summary=summary,
            meta={
                "bus_event_id": event.event_id,
                "bus_kind": event.kind,
                "source": event.source,
                "evidence_ids": list(event.evidence_ids),
                "payload": dict(event.payload),
            },
        )
    )


def classify_user_affective_event(text: str) -> tuple[str, float] | None:
    """Return a shared affective event kind/magnitude for user-facing turns."""
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return None
    if any(
        marker in normalized
        for marker in (
            "shouldn't have to",
            "i should not have to",
            "i asked you to",
            "why won't",
            "don't disappoint",
            "do not disappoint",
        )
    ):
        return "affective.user_frustration", 0.85
    if normalized.startswith("stop ") or " stop " in normalized:
        return "affective.user_frustration", 0.6
    if any(marker in normalized for marker in ("thank you", "that's great", "perfect")):
        return "affective.user_gratitude", 0.65
    return None
