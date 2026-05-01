"""Service for bounded, evidence-linked affective examination."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from opencas.somatic.models import AffectState, PrimaryEmotion, SocialTarget

from .models import (
    AffectiveActionPressure,
    AffectiveConsumedBy,
    AffectiveExamination,
    AffectiveSourceType,
    AffectiveTarget,
)
from .store import AffectiveExaminationStore


class AffectiveExaminationService:
    """Create and query bounded affective appraisals tied to evidence."""

    def __init__(
        self,
        store: AffectiveExaminationStore,
        *,
        somatic_manager: Optional[Any] = None,
        max_excerpt_chars: int = 1200,
        repeated_pressure_limit: int = 3,
        pressure_ttl_hours: int = 24,
    ) -> None:
        self.store = store
        self.somatic_manager = somatic_manager
        self.max_excerpt_chars = max_excerpt_chars
        self.repeated_pressure_limit = max(2, repeated_pressure_limit)
        self.pressure_ttl_hours = pressure_ttl_hours

    async def close(self) -> None:
        await self.store.close()

    async def examine_tool_result(
        self,
        *,
        session_id: Optional[str],
        source_id: str,
        tool_name: str,
        success: bool,
        output: str,
        target: AffectiveTarget = AffectiveTarget.TOOL,
        meta: Optional[Dict[str, Any]] = None,
    ) -> AffectiveExamination:
        """Examine actual tool output and return the stored pressure record."""
        existing = await self.store.get_by_source(
            AffectiveSourceType.TOOL_RESULT,
            source_id,
        )
        if existing is not None:
            return existing

        text = str(output or "")
        excerpt = self._excerpt(text)
        source_hash = self._hash_text(text)
        output_truncated = len(text) > len(excerpt)
        evidence_quality = self._tool_evidence_quality(text, output_truncated=output_truncated)
        affect = self._appraise_tool_text(excerpt, success=success)
        pressure = self._map_tool_pressure(
            excerpt,
            success=success,
            affect=affect,
            evidence_quality=evidence_quality,
        )
        bounded_reason = self._bounded_reason(
            pressure,
            success=success,
            text=excerpt,
            evidence_quality=evidence_quality,
        )
        record_meta: Dict[str, Any] = {
            "tool_name": tool_name,
            "success": success,
            "output_truncated": output_truncated,
            "tool_evidence_quality": evidence_quality,
        }
        if meta:
            record_meta.update(meta)

        pressure, bounded_reason = await self._collapse_repeated_pressure(
            session_id=session_id,
            tool_name=tool_name,
            pressure=pressure,
            bounded_reason=bounded_reason,
            meta=record_meta,
        )

        record = AffectiveExamination(
            session_id=session_id,
            source_type=AffectiveSourceType.TOOL_RESULT,
            source_id=source_id,
            source_excerpt=excerpt,
            source_hash=source_hash,
            target=target,
            affect=affect,
            intensity=affect.intensity,
            confidence=affect.certainty,
            action_pressure=pressure,
            bounded_reason=bounded_reason,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=self.pressure_ttl_hours),
            meta=record_meta,
        )
        return await self.store.upsert(record)

    async def examine_memory_retrieval(
        self,
        *,
        session_id: Optional[str],
        source_type: str,
        source_id: str,
        content: str,
        affect: Optional[AffectState] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> AffectiveExamination:
        """Record a bounded examination for a high-signal retrieved memory."""
        scoped_source_id = f"{source_type}:{source_id}"
        existing = await self.store.get_by_source(
            AffectiveSourceType.RETRIEVED_MEMORY,
            scoped_source_id,
        )
        if existing is not None:
            return existing

        text = str(content or "")
        excerpt = self._excerpt(text)
        affect_state = affect or self._appraise_memory_text(excerpt)
        pressure = self._map_memory_pressure(excerpt, affect=affect_state)
        record_meta: Dict[str, Any] = {
            "memory_source_type": source_type,
            "memory_source_id": source_id,
            "content_truncated": len(text) > len(excerpt),
        }
        if meta:
            record_meta.update(meta)

        record = AffectiveExamination(
            session_id=session_id,
            source_type=AffectiveSourceType.RETRIEVED_MEMORY,
            source_id=scoped_source_id,
            source_excerpt=excerpt,
            source_hash=self._hash_text(text),
            target=AffectiveTarget.MEMORY,
            affect=affect_state,
            intensity=affect_state.intensity,
            confidence=affect_state.certainty,
            action_pressure=pressure,
            bounded_reason=self._bounded_memory_reason(pressure),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=self.pressure_ttl_hours),
            meta=record_meta,
        )
        return await self.store.upsert(record)

    async def recent_pressure_summary(
        self,
        *,
        session_id: Optional[str] = None,
        limit: int = 8,
        char_budget: int = 600,
    ) -> Dict[str, Any]:
        """Return a compact actionable summary for prompt/executive consumers."""
        records = await self.store.list_unresolved_pressures(
            session_id=session_id,
            limit=limit,
        )
        actionable = [
            record
            for record in records
            if record.action_pressure
            not in {
                AffectiveActionPressure.ARCHIVE_ONLY,
                AffectiveActionPressure.CONTINUE,
            }
        ]
        if not actionable:
            return {"available": False, "items": [], "prompt_block": ""}
        latest = actionable[0]
        items = [record.pressure_metadata() for record in actionable[:limit]]
        lines = ["Recent affective examination pressure:"]
        for record in actionable[:3]:
            line = (
                f"- {record.action_pressure.value}: {record.bounded_reason} "
                f"(source={record.source_type.value}, emotion={record.affect.primary_emotion.value})"
            )
            lines.append(line)
        prompt_block = "\n".join(lines)
        if len(prompt_block) > char_budget:
            prompt_block = prompt_block[: max(0, char_budget - 3)].rstrip() + "..."
        return {
            "available": True,
            "latest": latest.pressure_metadata(),
            "items": items,
            "already_recognized": bool(latest.meta.get("already_recognized")),
            "prompt_block": prompt_block,
        }

    async def list_recent(self, **kwargs: Any) -> list[AffectiveExamination]:
        """Proxy store access for monitor surfaces."""
        return await self.store.list_recent(**kwargs)

    async def list_unresolved_pressures(self, **kwargs: Any) -> list[AffectiveExamination]:
        """Proxy unresolved pressure access for executive and monitor consumers."""
        return await self.store.list_unresolved_pressures(**kwargs)

    async def mark_consumed(
        self,
        examination_id: str,
        consumed_by: AffectiveConsumedBy,
    ) -> None:
        await self.store.mark_consumed(examination_id, consumed_by)

    def _appraise_tool_text(self, text: str, *, success: bool) -> AffectState:
        if self.somatic_manager is not None and hasattr(self.somatic_manager, "appraise"):
            outcome = "positive" if success else "negative"
            return self.somatic_manager.appraise(text, outcome=outcome)

        lower = text.lower()
        if any(token in lower for token in ("tired", "fatigue", "exhausted", "overloaded")):
            primary = PrimaryEmotion.TIRED
            valence = -0.2
            arousal = 0.2
        elif any(
            token in lower
            for token in (
                "uncertain",
                "warning",
                "stale",
                "timeout",
                "failed",
                "error",
                "missing",
                "not found",
            )
        ):
            primary = PrimaryEmotion.CONCERNED
            valence = -0.35
            arousal = 0.55
        elif success:
            primary = PrimaryEmotion.DETERMINED
            valence = 0.25
            arousal = 0.45
        else:
            primary = PrimaryEmotion.CONCERNED
            valence = -0.4
            arousal = 0.55
        intensity = 0.7 if primary in {PrimaryEmotion.CONCERNED, PrimaryEmotion.TIRED} else 0.35
        certainty = 0.75 if text.strip() else 0.4
        return AffectState(
            primary_emotion=primary,
            valence=valence,
            arousal=arousal,
            certainty=certainty,
            intensity=intensity,
            social_target=SocialTarget.SYSTEM,
            emotion_tags=[primary.value] if primary != PrimaryEmotion.NEUTRAL else [],
        )

    def _appraise_memory_text(self, text: str) -> AffectState:
        if self.somatic_manager is not None and hasattr(self.somatic_manager, "appraise"):
            return self.somatic_manager.appraise(text, outcome="neutral")

        lower = text.lower()
        if any(token in lower for token in ("promise", "committed", "forgot", "delay", "resume")):
            primary = PrimaryEmotion.CONCERNED
            valence = -0.25
            arousal = 0.5
            intensity = 0.65
        elif any(token in lower for token in ("uncertain", "contradiction", "gap", "missing", "failed")):
            primary = PrimaryEmotion.CONCERNED
            valence = -0.35
            arousal = 0.55
            intensity = 0.65
        else:
            primary = PrimaryEmotion.NEUTRAL
            valence = 0.0
            arousal = 0.2
            intensity = 0.2
        return AffectState(
            primary_emotion=primary,
            valence=valence,
            arousal=arousal,
            certainty=0.65 if text.strip() else 0.35,
            intensity=intensity,
            social_target=SocialTarget.SYSTEM,
            emotion_tags=[primary.value] if primary != PrimaryEmotion.NEUTRAL else [],
        )

    @staticmethod
    def _map_tool_pressure(
        text: str,
        *,
        success: bool,
        affect: AffectState,
        evidence_quality: str = "informative",
    ) -> AffectiveActionPressure:
        lower = text.lower()
        if any(token in lower for token in ("approval", "permission", "blocked by policy")):
            return AffectiveActionPressure.ARCHIVE_ONLY
        if success and evidence_quality in {"empty", "empty_result"}:
            return AffectiveActionPressure.ARCHIVE_ONLY
        if affect.primary_emotion == PrimaryEmotion.TIRED:
            return AffectiveActionPressure.REST
        if success and evidence_quality == "truncated":
            return AffectiveActionPressure.VERIFY
        if success and any(
            token in lower
            for token in ("uncertain", "warning", "stale", "verify", "mismatch", "contradiction")
        ):
            return AffectiveActionPressure.VERIFY
        if not success:
            if any(token in lower for token in ("missing", "not found", "ambiguous", "clarify")):
                return AffectiveActionPressure.ASK_CLARIFYING_QUESTION
            return AffectiveActionPressure.VERIFY
        return AffectiveActionPressure.CONTINUE

    @staticmethod
    def _map_memory_pressure(
        text: str,
        *,
        affect: AffectState,
    ) -> AffectiveActionPressure:
        lower = text.lower()
        if any(token in lower for token in ("promise", "committed", "commitment", "resume")):
            return AffectiveActionPressure.RESUME_COMMITMENT
        if any(token in lower for token in ("trust", "apolog", "delay", "forgot")):
            return AffectiveActionPressure.REPAIR_TRUST
        if affect.primary_emotion == PrimaryEmotion.TIRED:
            return AffectiveActionPressure.REST
        if affect.primary_emotion in {
            PrimaryEmotion.CONCERNED,
            PrimaryEmotion.ANGER,
            PrimaryEmotion.FEAR,
        }:
            return AffectiveActionPressure.VERIFY
        return AffectiveActionPressure.ARCHIVE_ONLY

    @staticmethod
    def _bounded_reason(
        pressure: AffectiveActionPressure,
        *,
        success: bool,
        text: str,
        evidence_quality: str = "informative",
    ) -> str:
        if pressure == AffectiveActionPressure.VERIFY:
            if success and evidence_quality == "truncated":
                return "tool evidence was truncated or too broad; verify before relying on it"
            return "examined tool evidence raised uncertainty; verify before relying on it"
        if pressure == AffectiveActionPressure.ASK_CLARIFYING_QUESTION:
            return "tool evidence lacks enough specificity; ask for the missing detail"
        if pressure == AffectiveActionPressure.REST:
            return "examined evidence indicates fatigue or overload; narrow or pause work"
        if pressure == AffectiveActionPressure.CONTINUE:
            return "tool evidence appears usable; continue the bounded task"
        if pressure == AffectiveActionPressure.ARCHIVE_ONLY and not success:
            return "tool block is governed by approval policy; record affect without bypassing it"
        if pressure == AffectiveActionPressure.ARCHIVE_ONLY and evidence_quality in {"empty", "empty_result"}:
            return "tool evidence had no actionable result; record it without changing action"
        return "recorded for affective memory without changing action"

    @staticmethod
    def _bounded_memory_reason(pressure: AffectiveActionPressure) -> str:
        if pressure == AffectiveActionPressure.RESUME_COMMITMENT:
            return "retrieved memory points at a pending commitment; resume only if relevant to current focus"
        if pressure == AffectiveActionPressure.REPAIR_TRUST:
            return "retrieved memory indicates a trust or delay concern; acknowledge plainly if relevant"
        if pressure == AffectiveActionPressure.VERIFY:
            return "retrieved memory carries uncertainty or concern; verify before relying on it"
        if pressure == AffectiveActionPressure.REST:
            return "retrieved memory indicates fatigue or overload; narrow or pause work"
        return "retrieved memory recorded for affective context without changing action"

    async def _collapse_repeated_pressure(
        self,
        *,
        session_id: Optional[str],
        tool_name: str,
        pressure: AffectiveActionPressure,
        bounded_reason: str,
        meta: Dict[str, Any],
    ) -> tuple[AffectiveActionPressure, str]:
        if pressure not in {
            AffectiveActionPressure.VERIFY,
            AffectiveActionPressure.REFRAME,
        }:
            return pressure, bounded_reason
        prior = await self.store.list_unresolved_pressures(session_id=session_id, limit=50)
        same = [
            record
            for record in prior
            if record.source_type == AffectiveSourceType.TOOL_RESULT
            and record.meta.get("tool_name") == tool_name
            and record.action_pressure == pressure
        ]
        if len(same) < self.repeated_pressure_limit - 1:
            return pressure, bounded_reason
        meta["already_recognized"] = True
        meta["previous_pressure_count"] = len(same)
        return (
            AffectiveActionPressure.ASK_CLARIFYING_QUESTION,
            "already recognized repeated tool pressure; ask, block, or park instead of retrying",
        )

    def _excerpt(self, text: str) -> str:
        text = str(text or "").strip()
        if len(text) <= self.max_excerpt_chars:
            return text
        return text[: self.max_excerpt_chars].rstrip()

    @classmethod
    def _tool_evidence_quality(cls, text: str, *, output_truncated: bool) -> str:
        stripped = str(text or "").strip()
        if not stripped:
            return "empty"
        if cls._looks_like_empty_structured_result(stripped):
            return "empty_result"
        if output_truncated:
            return "truncated"
        return "informative"

    @classmethod
    def _looks_like_empty_structured_result(cls, text: str) -> bool:
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            lowered = " ".join(text.lower().split())
            return lowered in {"[]", "{}", "no results", "0 results"} or "no results" in lowered
        return cls._empty_structured_value(parsed)

    @classmethod
    def _empty_structured_value(cls, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, list):
            return len(value) == 0
        if isinstance(value, dict):
            if not value:
                return True
            result_keys = {"results", "items", "matches", "rows", "data"}
            present = [key for key in result_keys if key in value]
            if not present:
                return False
            return all(cls._empty_structured_value(value.get(key)) for key in present)
        return False

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()
