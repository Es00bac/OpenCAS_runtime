"""Runtime service for desktop screenshot context and spoken body-double nudges."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field, field_validator

from opencas.cognition import CognitionGrounding, GroundingKind, GroundingSource
from opencas.context import ContextLane, ContextProposal, MessageRole
from opencas.daydream.models import DaydreamThought, DaydreamThoughtKind, DaydreamThoughtRoute
from opencas.identity.agent_name import resolve_agent_name
from opencas.memory import EpisodeKind, Memory

from .capture import DesktopCapture, capture_desktop_image, run_tesseract_ocr
from .media import MprisMediaController


class DesktopContextConfig(BaseModel):
    """Operator-controlled settings for desktop observation."""

    enabled: bool = True
    capture_interval_seconds: int = Field(default=60, ge=0)
    min_speech_interval_seconds: int = Field(default=60, ge=0)
    tts_enabled: bool = True
    play_audio: bool = True
    vision_enabled: bool = True
    ocr_enabled: bool = True
    capture_backend: str = "auto"
    vision_model: Optional[str] = None
    session_id: Optional[str] = None
    declared_task: Optional[str] = None
    declared_task_source: Optional[str] = None
    declared_task_updated_at: Optional[str] = None
    speech_relevance_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    youtube_transcripts_enabled: bool = True
    youtube_transcript_max_chars: int = Field(default=12000, ge=0, le=60000)
    yt_dlp_path: Optional[str] = None
    self_interest_followup_enabled: bool = True
    self_interest_match_threshold: float = Field(default=0.62, ge=0.0, le=1.0)
    self_interest_interrupt_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    observed_context_relevance_enabled: bool = True
    observed_context_relevance_match_threshold: float = Field(default=0.62, ge=0.0, le=1.0)
    observed_context_relevance_interrupt_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    project_context_relevance_enabled: bool = True
    proactive_video_commentary_enabled: bool = True
    youtube_transcript_excerpt_chars: int = Field(default=4000, ge=400, le=12000)
    youtube_transcript_commentary_lag_seconds: float = Field(default=6.0, ge=0.0, le=60.0)
    live_transcription_enabled: bool = False
    live_transcription_capture_seconds: float = Field(default=6.0, ge=1.0, le=30.0)
    live_transcription_min_interval_seconds: float = Field(default=12.0, ge=0.0, le=120.0)
    live_transcription_max_chars: int = Field(default=1600, ge=200, le=6000)
    live_transcription_whisper_model: Optional[str] = "base"
    live_transcription_audio_input: Optional[str] = None
    live_transcription_ffmpeg_path: Optional[str] = None
    live_transcription_whisper_path: Optional[str] = None
    live_transcription_timeout_seconds: int = Field(default=45, ge=5, le=300)
    media_state_poll_seconds: int = Field(default=2, ge=1, le=30)
    media_commentary_mode_enabled: bool = False
    media_commentary_requested_at: Optional[str] = None
    media_commentary_source: Optional[str] = None
    media_commentary_request: Optional[str] = None
    media_commentary_request_source: Optional[str] = None
    media_commentary_request_text: Optional[str] = None
    livestream_resume_catchup_enabled: bool = True
    livestream_resume_catchup_rate: float = Field(default=1.5, ge=1.0, le=2.0)
    livestream_resume_catchup_max_seconds: float = Field(default=90.0, ge=0.0, le=600.0)
    max_spoken_chars: int = Field(default=360, ge=80, le=1000)
    max_ocr_chars: int = Field(default=4000, ge=0, le=24000)
    max_image_bytes: int = Field(default=5_000_000, ge=1)
    vision_max_dimension: int = Field(default=1600, ge=320, le=4096)
    vision_jpeg_quality: int = Field(default=82, ge=30, le=95)

    @field_validator("capture_backend", mode="before")
    @classmethod
    def _normalize_capture_backend(cls, value: Any) -> str:
        cleaned = str(value or "auto").strip()
        return cleaned or "auto"

    @field_validator(
        "vision_model",
        "session_id",
        "declared_task",
        "declared_task_source",
        "declared_task_updated_at",
        "yt_dlp_path",
        "live_transcription_whisper_model",
        "live_transcription_audio_input",
        "live_transcription_ffmpeg_path",
        "live_transcription_whisper_path",
        "media_commentary_requested_at",
        "media_commentary_source",
        "media_commentary_request",
        "media_commentary_request_source",
        "media_commentary_request_text",
        mode="before",
    )
    @classmethod
    def _blank_string_to_none(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None


class DesktopContextService:
    """Capture the active desktop and turn it into durable collaboration context."""

    def __init__(
        self,
        *,
        runtime: Any,
        state_dir: Path | str,
        config: Optional[DesktopContextConfig] = None,
        capture_provider: Optional[Callable[[Path], DesktopCapture]] = None,
        ocr_provider: Optional[Callable[[Path], str]] = None,
        speech_synthesizer: Optional[Callable[[str], Any]] = None,
        audio_player: Optional[Callable[[Path], Any]] = None,
        media_controller: Optional[Any] = None,
        youtube_transcript_provider: Optional[Callable[..., dict[str, Any]]] = None,
        live_transcript_provider: Optional[Callable[..., dict[str, Any]]] = None,
        time_source: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.runtime = runtime
        self.root = Path(state_dir).expanduser() / "desktop_context"
        self.root.mkdir(parents=True, exist_ok=True)
        self._capture_provider = capture_provider
        self._ocr_provider = ocr_provider
        self._speech_synthesizer = speech_synthesizer
        self._audio_player = audio_player or play_audio_file
        self._media_controller = media_controller or MprisMediaController()
        self._youtube_transcript_provider = youtube_transcript_provider
        self._live_transcript_provider = live_transcript_provider
        self._time_source = time_source or (lambda: datetime.now(timezone.utc))
        self._project_relevance_cache: Optional[list[dict[str, str]]] = None
        self._last_live_transcript_at_by_media: dict[str, datetime] = {}
        self._last_live_transcript_payload_by_media: dict[str, dict[str, Any]] = {}
        self._livestream_rate_restore_tasks: set[asyncio.Task[Any]] = set()
        self.config = config or self._load_config()

    def status(self) -> dict[str, Any]:
        """Return a dashboard/tool-friendly status snapshot."""

        events = self._list_events(limit=50)
        return {
            "config": self.config.model_dump(),
            "paths": {
                "root": str(self.root),
                "screenshots": str(self._screenshots_dir()),
                "notes": str(self._notes_dir()),
                "audio": str(self._audio_dir()),
            },
            "capture_backend_available": self._capture_backend_available(),
            "local_tts_available": shutil.which("edge-tts") is not None,
            "local_whisper_available": shutil.which(self.config.live_transcription_whisper_path or "whisper") is not None,
            "live_transcription_audio_input": self.config.live_transcription_audio_input
            or self._default_pulse_monitor_source(),
            "event_count": len(events),
            "last_event": events[-1] if events else None,
            "last_observed_at": self._last_event_at("observed"),
            "last_spoken_at": self._last_event_at("spoken"),
        }

    def configure(self, **updates: Any) -> dict[str, Any]:
        """Update persisted desktop-context settings."""

        allowed = set(DesktopContextConfig.model_fields)
        clean_updates: dict[str, Any] = {}
        nullable_fields = {
            "declared_task",
            "declared_task_source",
            "declared_task_updated_at",
            "live_transcription_audio_input",
            "live_transcription_ffmpeg_path",
            "live_transcription_whisper_path",
        }
        for key, value in updates.items():
            if key not in allowed:
                continue
            if value is None and key not in nullable_fields:
                continue
            clean_updates[key] = value
        if "declared_task" in clean_updates:
            task_text = str(clean_updates.get("declared_task") or "").strip()
            if task_text:
                clean_updates["declared_task"] = task_text
                clean_updates.setdefault(
                    "declared_task_source",
                    str(updates.get("declared_task_source") or "operator").strip() or "operator",
                )
                clean_updates["declared_task_updated_at"] = self._now().isoformat()
            else:
                clean_updates["declared_task"] = None
                clean_updates["declared_task_source"] = None
                clean_updates["declared_task_updated_at"] = None
        payload = {**self.config.model_dump(), **clean_updates}
        self.config = DesktopContextConfig(**payload)
        self._save_config()
        event = self._event("configured", {"updates": clean_updates})
        return {"config": self.config.model_dump(), "event": event}

    async def capture_once(self, *, force: bool = False) -> dict[str, Any]:
        """Capture one screenshot and optional OCR payload."""

        if not self.config.enabled and not force:
            return {"status": "skipped", "reason": "disabled"}

        target = self._screenshots_dir() / f"desktop_{self._timestamp_slug()}.png"
        capture = await self._capture(target)
        if not capture.success:
            event = self._event(
                "capture_failed",
                {
                    "path": str(capture.path),
                    "backend": capture.backend,
                    "error": capture.error,
                },
            )
            return {
                "status": "failed",
                "reason": "capture_failed",
                "capture": self._capture_to_dict(capture),
                "event": event,
            }

        ocr_text = ""
        if self.config.ocr_enabled:
            try:
                ocr_text = await self._call_maybe_async(self._ocr_provider or run_tesseract_ocr, capture.path)
            except Exception:
                ocr_text = ""
            if self.config.max_ocr_chars and len(ocr_text) > self.config.max_ocr_chars:
                ocr_text = ocr_text[: self.config.max_ocr_chars].rstrip()

        payload = {
            "status": "captured",
            "capture": self._capture_to_dict(capture),
            "ocr_text": ocr_text,
        }
        payload["event"] = self._event(
            "captured",
            {
                "path": str(capture.path),
                "backend": capture.backend,
                "ocr_chars": len(ocr_text),
            },
        )
        return payload

    async def observe_once(
        self,
        *,
        force: bool = False,
        reason: str = "manual",
        speak: Optional[bool] = None,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Capture, analyze, persist context, and optionally speak a short nudge."""

        if not self.config.enabled and not force:
            return {"status": "skipped", "reason": "disabled"}
        if not force and not self._observation_due():
            return {"status": "skipped", "reason": "not_due"}

        capture_result = await self.capture_once(force=True)
        if capture_result.get("status") != "captured":
            return capture_result
        await self._enrich_capture_context(capture_result)

        analysis = await self._analyze_capture(capture_result, reason=reason)
        ambiguity = self._media_ambiguity(capture_result)
        if ambiguity is not None and self.config.media_commentary_mode_enabled:
            analysis = self._media_ambiguity_analysis(ambiguity, reason=reason)
            followup = None
        else:
            followup = await self._maybe_self_interest_followup(
                capture_result,
                analysis,
                reason=reason,
                session_id=session_id,
            )
        if followup:
            analysis["self_interest_followup"] = followup
            if followup.get("should_speak"):
                analysis["should_speak"] = True
                analysis["speech_intent"] = "screen_relevant"
                analysis["speech_relevance_score"] = max(
                    self._coerce_float(analysis.get("speech_relevance_score"), default=0.0),
                    self._coerce_float(followup.get("salience"), default=0.0),
                )
                if followup.get("spoken_text"):
                    analysis["spoken_text"] = str(followup.get("spoken_text") or "")
                analysis["reason"] = "observed media intersects with durable self-interest evidence"
                if followup.get("connection_summary"):
                    analysis["note"] = str(followup.get("connection_summary") or "")
        if self._analysis_speech_is_media_summary(capture_result, analysis):
            self._suppress_media_summary_speech(analysis)
        analysis = self._apply_speech_policy(analysis, reason=reason)
        analysis = self._apply_media_playback_speech_policy(analysis, capture_result, reason=reason)
        context_text = self._build_context_text(capture_result, analysis, reason=reason)
        await self._persist_context(
            context_text,
            capture_result,
            analysis,
            reason=reason,
            session_id=session_id,
        )

        speech: Optional[dict[str, Any]] = None
        should_speak = bool(analysis.get("should_speak"))
        speech_requested = self.config.tts_enabled if speak is None else bool(speak)
        if should_speak and speech_requested:
            speech = await self._speak_analysis(analysis, capture_result, reason=reason)

        payload = {
            "status": "observed",
            "reason": reason,
            "capture": capture_result.get("capture"),
            "media_context": capture_result.get("media_context") or [],
            "media_state_changes": capture_result.get("media_state_changes") or [],
            "youtube_transcript": capture_result.get("youtube_transcript"),
            "live_transcript": capture_result.get("live_transcript"),
            "ocr_chars": len(str(capture_result.get("ocr_text") or "")),
            "analysis": analysis,
            "speech": speech,
        }
        payload["event"] = self._event(
            "observed",
            {
                "reason": reason,
                "capture_path": (capture_result.get("capture") or {}).get("path"),
                "should_speak": should_speak,
                "speech_status": speech.get("status") if isinstance(speech, dict) else None,
                "activity_summary": str(analysis.get("activity_summary") or "")[:240],
            },
        )
        return payload

    async def observe_for_conversation(
        self,
        *,
        session_id: str,
        user_input: str,
        source: str = "conversation",
    ) -> Optional[dict[str, Any]]:
        """Force one silent observation to ground an active conversation turn."""

        if not self.config.enabled:
            return None
        if self._looks_like_media_commentary_request(user_input):
            self._activate_media_commentary_mode(request=user_input, source=source)
        result = await self.observe_once(
            force=True,
            reason=f"conversation_context:{source}",
            speak=False,
            session_id=session_id,
        )
        if result.get("status") != "observed":
            return result
        result["conversation_prompt_note"] = self._conversation_prompt_note(result, user_input=user_input)
        return result

    async def analyze_region_for_conversation(
        self,
        *,
        image_path: Path | str,
        prompt: str,
        session_id: Optional[str] = None,
        selection: Optional[dict[str, Any]] = None,
        source: str = "desktop_region_prompt",
    ) -> dict[str, Any]:
        """Analyze a user-selected desktop region and build turn-scoped prompt context."""

        path = Path(image_path).expanduser()
        if not path.exists():
            return {
                "status": "failed",
                "reason": "region_image_missing",
                "path": str(path),
            }
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        width, height = self._image_dimensions(path)
        capture = DesktopCapture(
            success=True,
            path=path,
            backend=source,
            media_type=media_type,
            width=width,
            height=height,
        )
        ocr_text = ""
        if self.config.ocr_enabled:
            try:
                ocr_text = await self._call_maybe_async(self._ocr_provider or run_tesseract_ocr, path)
            except Exception:
                ocr_text = ""
            if self.config.max_ocr_chars and len(ocr_text) > self.config.max_ocr_chars:
                ocr_text = ocr_text[: self.config.max_ocr_chars].rstrip()

        capture_result: dict[str, Any] = {
            "status": "captured",
            "capture": self._capture_to_dict(capture),
            "ocr_text": ocr_text,
            "region_selection": dict(selection or {}),
        }
        await self._enrich_capture_context(capture_result)
        reason = f"region_prompt:{source}"
        analysis = await self._analyze_region_prompt(capture_result, prompt=prompt, source=source)
        context_text = self._build_region_context_text(capture_result, analysis, prompt=prompt, reason=reason)
        await self._persist_context(
            context_text,
            capture_result,
            analysis,
            reason=reason,
            session_id=session_id,
        )
        payload = {
            "status": "observed",
            "reason": reason,
            "source": source,
            "prompt": prompt,
            "capture": capture_result.get("capture"),
            "region_selection": capture_result.get("region_selection") or {},
            "media_context": capture_result.get("media_context") or [],
            "youtube_transcript": capture_result.get("youtube_transcript"),
            "live_transcript": capture_result.get("live_transcript"),
            "ocr_chars": len(str(capture_result.get("ocr_text") or "")),
            "analysis": analysis,
            "capture_result": capture_result,
        }
        payload["conversation_prompt_note"] = self._region_conversation_prompt_note(payload, prompt=prompt)
        payload["event"] = self._event(
            "region_prompt_observed",
            {
                "source": source,
                "capture_path": str(path),
                "prompt_excerpt": self._shorten_for_speech(prompt, 220),
                "activity_summary": str(analysis.get("activity_summary") or "")[:240],
            },
        )
        return payload

    async def run_once(self) -> dict[str, Any]:
        """Scheduler entrypoint for enabled body-double observations."""

        return await self.observe_once(force=False, reason="scheduled_body_double")

    async def poll_media_state_once(self) -> dict[str, Any]:
        """Poll MPRIS media state without taking a screenshot or calling an LLM."""

        if not self.config.enabled:
            return {"status": "skipped", "reason": "disabled", "commentary_observation_requested": False}
        media_context = await self._current_media_context()
        changes = self._record_media_state_changes(media_context)
        commentary_reason = self._media_commentary_observation_reason(changes)
        return {
            "status": "changed" if changes else "unchanged",
            "media_context": media_context,
            "media_state_changes": changes,
            "commentary_observation_requested": commentary_reason is not None,
            "commentary_observation_reason": commentary_reason,
        }

    async def speak_text(
        self,
        text: str,
        *,
        reason: str = "manual",
        force: bool = False,
        max_chars: Optional[int] = None,
        allow_note_redirect: bool = True,
    ) -> dict[str, Any]:
        """Speak a direct natural-language message through the configured TTS path."""

        analysis = {
            "should_speak": True,
            "activity_summary": "Direct desktop-context speech request.",
            "spoken_text": text,
            "reason": reason,
            "max_spoken_chars": max_chars,
            "allow_note_redirect": allow_note_redirect,
        }
        return await self._speak_analysis(analysis, {"capture": {}}, reason=reason, force=force)

    async def _maybe_self_interest_followup(
        self,
        capture_result: dict[str, Any],
        analysis: dict[str, Any],
        *,
        reason: str,
        session_id: Optional[str],
    ) -> Optional[dict[str, Any]]:
        if not (self.config.self_interest_followup_enabled or self.config.observed_context_relevance_enabled):
            return None
        if not self._has_media_interest_material(capture_result):
            return None
        if self._has_playing_video_context(capture_result) and not self.config.proactive_video_commentary_enabled:
            return None
        interests = await self._observed_context_relevance_evidence(limit=12)
        if not interests:
            return None
        followup = await self._analyze_self_interest_overlap(
            capture_result,
            analysis,
            interests=interests,
            reason=reason,
        )
        if not (followup.get("matches_observed_context") or followup.get("matches_self_interest")):
            if analysis.get("should_speak") and not self._analysis_speech_is_media_summary(capture_result, analysis):
                return None
            fallback_followup = self._heuristic_observed_context_followup(
                capture_result,
                analysis,
                interests=interests,
            )
            if fallback_followup is None:
                return None
            followup = fallback_followup
        confidence = self._coerce_float(followup.get("confidence"), default=0.0)
        salience = self._coerce_float(followup.get("salience"), default=0.0)
        match_threshold = self.config.observed_context_relevance_match_threshold
        interrupt_threshold = self.config.observed_context_relevance_interrupt_threshold
        if bool(followup.get("matches_self_interest")) and not bool(followup.get("matches_shared_work")):
            match_threshold = self.config.self_interest_match_threshold
            interrupt_threshold = self.config.self_interest_interrupt_threshold
        if max(confidence, salience) < match_threshold:
            return None
        followup["interests_considered"] = interests
        followup["relevance_targets_considered"] = interests
        followup["evidence_refs"] = self._media_interest_evidence_refs(capture_result)
        followup["source"] = "desktop_context.observed_context_relevance"
        followup["status"] = "accepted_for_observed_context_relevance"
        if followup.get("heuristic_fallback"):
            followup["operator_interruption_warranted"] = False
            followup_spoken_text = str(followup.get("spoken_text") or "").strip()
        else:
            followup_spoken_text = str(followup.get("spoken_text") or followup.get("agent_viewpoint") or "").strip()
        if followup_spoken_text:
            followup["spoken_text"] = followup_spoken_text
        media_playback_allows_speech = self._media_playback_speech_allowed(capture_result)
        followup["should_speak"] = bool(
            followup.get("operator_interruption_warranted")
            and max(confidence, salience) >= interrupt_threshold
            and followup_spoken_text
            and media_playback_allows_speech
        )
        if not media_playback_allows_speech and followup_spoken_text:
            followup["speech_suppressed_reason"] = "media_not_playing"
        await self._persist_self_interest_followup(followup, capture_result, session_id=session_id)
        return followup

    async def _capture(self, target: Path) -> DesktopCapture:
        provider = self._capture_provider
        if provider is not None:
            return await self._call_maybe_async(provider, target)
        return await self._call_maybe_async(
            capture_desktop_image,
            target,
            backend=self.config.capture_backend,
        )

    async def _enrich_capture_context(self, capture_result: dict[str, Any]) -> None:
        media_context = await self._current_media_context()
        if media_context:
            capture_result["media_context"] = media_context
        media_state_changes = self._record_media_state_changes(media_context)
        if media_state_changes:
            capture_result["media_state_changes"] = media_state_changes
        ambiguity = self._media_ambiguity({"media_context": media_context})
        if ambiguity is not None:
            capture_result["media_ambiguity"] = ambiguity
        transcript: Optional[dict[str, Any]] = None
        if self.config.youtube_transcripts_enabled and ambiguity is None:
            youtube_url = self._extract_youtube_url(capture_result, media_context)
            if youtube_url:
                transcript = await self._retrieve_youtube_transcript(
                    youtube_url,
                    media_context=media_context,
                )
                capture_result["youtube_transcript"] = transcript
                self._apply_youtube_transcript_timing_to_media_context(capture_result, transcript)
        if self.config.live_transcription_enabled:
            live_transcript = await self._retrieve_live_media_transcript(
                media_context=media_context,
                youtube_transcript=transcript or capture_result.get("youtube_transcript"),
                ambiguity=ambiguity,
            )
            capture_result["live_transcript"] = live_transcript

    async def _current_media_context(self) -> list[dict[str, Any]]:
        current = getattr(self._media_controller, "current_media", None)
        if not callable(current):
            return []
        try:
            value = await self._call_maybe_async(current)
        except Exception:
            return []
        if not isinstance(value, list):
            return []
        cleaned: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            position_us = self._coerce_int(item.get("position_us"))
            length_us = self._coerce_int(item.get("length_us"))
            cleaned.append(
                {
                    "player": str(item.get("player") or ""),
                    "status": str(item.get("status") or ""),
                    "title": str(item.get("title") or ""),
                    "artist": str(item.get("artist") or ""),
                    "album": str(item.get("album") or ""),
                    "url": str(item.get("url") or ""),
                    "length_us": length_us,
                    "position_us": position_us,
                    "position_label": self._format_media_position(position_us, length_us),
                    "progress_percent": self._media_progress_percent(position_us, length_us),
                    "is_live": item.get("is_live"),
                    "live_status": str(item.get("live_status") or ""),
                }
            )
        return cleaned

    def _extract_youtube_url(
        self,
        capture_result: dict[str, Any],
        media_context: list[dict[str, Any]],
    ) -> Optional[str]:
        media_candidates: list[tuple[int, int, str]] = []
        for index, item in enumerate(media_context or []):
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "").strip().lower()
            status_score = 100 if status == "playing" else 40 if status in {"paused", "stopped"} else 60
            for field_score, field in ((20, "url"), (5, "title")):
                url = self._first_youtube_url(str(item.get(field) or ""))
                if url:
                    media_candidates.append((status_score + field_score, -index, url))
        if media_candidates:
            media_candidates.sort(reverse=True)
            return media_candidates[0][2]
        return self._first_youtube_url(str(capture_result.get("ocr_text") or ""))

    def _first_youtube_url(self, text: str) -> Optional[str]:
        for match in re.finditer(
            r"https?://(?:www\.)?(?:youtube\.com/(?:watch\?[^\s]+|shorts/[^\s]+)|youtu\.be/[^\s]+)",
            str(text or ""),
            re.IGNORECASE,
        ):
            url = match.group(0).rstrip(").,;\"'")
            if self._youtube_video_id(url):
                return url
        return None

    async def _retrieve_live_media_transcript(
        self,
        *,
        media_context: list[dict[str, Any]],
        youtube_transcript: Any = None,
        ambiguity: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if ambiguity is not None:
            return self._live_transcript_status_payload(
                "skipped",
                reason="media_ambiguity",
                youtube_transcript=youtube_transcript,
            )
        media_item = self._select_live_transcript_media_item(media_context)
        if media_item is None:
            return self._live_transcript_status_payload(
                "skipped",
                reason="media_not_playing",
                youtube_transcript=youtube_transcript,
            )
        media_key = self._media_item_key(media_item) or self._media_identity_for_item(media_item)
        if media_key and not self._live_transcription_due(media_key):
            cached = self._last_live_transcript_payload_by_media.get(media_key)
            if cached:
                reused = dict(cached)
                reused["cached"] = True
                reused["reason"] = "recent_live_transcript_reused"
                return reused

        provider = self._live_transcript_provider
        try:
            if provider is not None:
                raw = await self._call_maybe_async(
                    provider,
                    media_item,
                    media_context=media_context,
                    youtube_transcript=youtube_transcript,
                )
            else:
                raw = await self._capture_and_transcribe_live_audio(media_item)
        except Exception as exc:
            raw = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "source": "whisper",
                "mode": "local",
            }
        payload = self._live_transcript_payload(
            raw,
            media_item,
            youtube_transcript=youtube_transcript,
        )
        if media_key and payload.get("status") == "available":
            self._last_live_transcript_at_by_media[media_key] = self._now()
            self._last_live_transcript_payload_by_media[media_key] = dict(payload)
        self._event(
            "live_transcript_observed",
            {
                "status": payload.get("status"),
                "reason": payload.get("reason"),
                "source": payload.get("source"),
                "mode": payload.get("mode"),
                "media_title": payload.get("media_title"),
                "media_url": payload.get("media_url"),
                "transcript_chars": payload.get("transcript_chars"),
                "cached": payload.get("cached", False),
            },
        )
        return payload

    def _select_live_transcript_media_item(self, media_context: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        for item in self._ordered_media_context(media_context):
            if not isinstance(item, dict):
                continue
            if not self._media_item_is_playing(item):
                continue
            if str(item.get("title") or item.get("url") or item.get("player") or "").strip():
                return item
        return None

    def _live_transcription_due(self, media_key: str) -> bool:
        if not media_key:
            return True
        interval = float(self.config.live_transcription_min_interval_seconds)
        if interval <= 0:
            return True
        last = self._last_live_transcript_at_by_media.get(media_key)
        if last is None:
            return True
        return (self._now() - last).total_seconds() >= interval

    def _live_transcript_status_payload(
        self,
        status: str,
        *,
        reason: str,
        youtube_transcript: Any = None,
        media_item: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        payload = {
            "status": status,
            "reason": reason,
            "source": "whisper",
            "mode": "local",
            "observed_at": self._now().isoformat(),
            "prefetched_transcript_status": self._prefetched_transcript_status(youtube_transcript),
        }
        if media_item:
            payload.update(self._live_transcript_media_fields(media_item))
        if extra:
            payload.update(extra)
        return payload

    def _live_transcript_payload(
        self,
        raw: Any,
        media_item: dict[str, Any],
        *,
        youtube_transcript: Any = None,
    ) -> dict[str, Any]:
        raw_payload = raw if isinstance(raw, dict) else {"transcript_text": str(raw or "")}
        text = str(
            raw_payload.get("transcript_text")
            or raw_payload.get("text")
            or raw_payload.get("transcript_excerpt")
            or ""
        ).strip()
        status = str(raw_payload.get("status") or ("available" if text else "unavailable")).strip().lower()
        reason = str(raw_payload.get("reason") or "").strip()
        if status == "available" and not text:
            status = "unavailable"
            reason = reason or "empty_transcript"
        max_chars = int(self.config.live_transcription_max_chars)
        excerpt = text[:max_chars].rstrip()
        source = str(raw_payload.get("source") or "whisper").strip() or "whisper"
        mode = str(raw_payload.get("mode") or "local").strip() or "local"
        model = str(raw_payload.get("model") or self.config.live_transcription_whisper_model or "").strip()
        payload = {
            "status": status,
            "reason": reason,
            "source": source,
            "mode": mode,
            "model": model,
            "observed_at": self._now().isoformat(),
            "capture_seconds": self._coerce_float(
                raw_payload.get("capture_seconds"),
                default=float(self.config.live_transcription_capture_seconds),
            ),
            "transcript_text": excerpt if status == "available" else "",
            "transcript_excerpt": excerpt if status == "available" else "",
            "transcript_chars": len(text) if status == "available" else 0,
            "transcript_hash": self._short_hash(excerpt) if excerpt else "",
            "segments": raw_payload.get("segments") if isinstance(raw_payload.get("segments"), list) else [],
            "prefetched_transcript_status": self._prefetched_transcript_status(youtube_transcript),
            "prefetched_transcript_video_id": self._prefetched_transcript_video_id(youtube_transcript),
            "cached": bool(raw_payload.get("cached")),
        }
        payload.update(self._live_transcript_media_fields(media_item))
        if status != "available" and not payload["reason"]:
            payload["reason"] = "transcript_unavailable"
        return payload

    def _live_transcript_media_fields(self, media_item: dict[str, Any]) -> dict[str, Any]:
        return {
            "media_key": self._media_item_key(media_item),
            "media_identity": self._media_identity_for_item(media_item),
            "media_player": str(media_item.get("player") or ""),
            "media_status": str(media_item.get("status") or ""),
            "media_title": str(media_item.get("title") or ""),
            "media_artist": str(media_item.get("artist") or ""),
            "media_url": str(media_item.get("url") or ""),
            "media_position_label": str(media_item.get("position_label") or ""),
            "media_position_us": self._coerce_int(media_item.get("position_us")),
            "media_length_us": self._coerce_int(media_item.get("length_us")),
        }

    def _prefetched_transcript_status(self, transcript: Any) -> str:
        if not isinstance(transcript, dict):
            return "none"
        return str(transcript.get("status") or "unknown").strip() or "unknown"

    def _prefetched_transcript_video_id(self, transcript: Any) -> str:
        if not isinstance(transcript, dict):
            return ""
        video_id = str(transcript.get("video_id") or "").strip()
        url = str(transcript.get("url") or "").strip()
        return video_id or self._youtube_video_id(url) or ""

    async def _capture_and_transcribe_live_audio(self, media_item: dict[str, Any]) -> dict[str, Any]:
        ffmpeg = self.config.live_transcription_ffmpeg_path or shutil.which("ffmpeg")
        if not ffmpeg:
            return self._live_transcript_status_payload(
                "unavailable",
                reason="ffmpeg_not_available",
                media_item=media_item,
            )
        whisper = self.config.live_transcription_whisper_path or shutil.which("whisper")
        if not whisper:
            return self._live_transcript_status_payload(
                "unavailable",
                reason="whisper_cli_not_available",
                media_item=media_item,
            )
        audio_input = self.config.live_transcription_audio_input or self._default_pulse_monitor_source()
        if not audio_input:
            return self._live_transcript_status_payload(
                "unavailable",
                reason="audio_monitor_source_unavailable",
                media_item=media_item,
            )
        capture_seconds = float(self.config.live_transcription_capture_seconds)
        with tempfile.TemporaryDirectory(prefix="opencas_live_whisper_") as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "live_media.wav"
            record_cmd = self._live_audio_record_command(
                ffmpeg,
                audio_input=audio_input,
                audio_path=audio_path,
                capture_seconds=capture_seconds,
            )
            try:
                record = await asyncio.to_thread(
                    subprocess.run,
                    record_cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=capture_seconds + 8.0,
                )
            except subprocess.TimeoutExpired:
                return self._live_transcript_status_payload(
                    "failed",
                    reason="audio_capture_timeout",
                    media_item=media_item,
                )
            if record.returncode != 0 or not audio_path.exists():
                return self._live_transcript_status_payload(
                    "failed",
                    reason="audio_capture_failed",
                    media_item=media_item,
                    extra={"stderr": str(record.stderr or "").strip()[:500]},
                )

            whisper_cmd = [
                whisper,
                str(audio_path),
                "--model",
                str(self.config.live_transcription_whisper_model or "base"),
                "--task",
                "transcribe",
                "--output_format",
                "json",
                "--output_dir",
                str(temp_path),
                "--fp16",
                "False",
            ]
            try:
                transcript = await asyncio.to_thread(
                    subprocess.run,
                    whisper_cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=float(self.config.live_transcription_timeout_seconds),
                )
            except subprocess.TimeoutExpired:
                return self._live_transcript_status_payload(
                    "failed",
                    reason="whisper_transcription_timeout",
                    media_item=media_item,
                )
            if transcript.returncode != 0:
                return self._live_transcript_status_payload(
                    "failed",
                    reason="whisper_transcription_failed",
                    media_item=media_item,
                    extra={"stderr": str(transcript.stderr or "").strip()[:500]},
                )
            data = self._read_whisper_json_result(temp_path)
            text = str(data.get("text") or "").strip()
            return {
                "status": "available" if text else "unavailable",
                "reason": "" if text else "empty_transcript",
                "source": "whisper",
                "mode": "local",
                "model": f"whisper-{self.config.live_transcription_whisper_model or 'base'}",
                "transcript_text": text,
                "segments": data.get("segments") if isinstance(data.get("segments"), list) else [],
                "capture_seconds": capture_seconds,
            }

    def _live_audio_record_command(
        self,
        ffmpeg: str,
        *,
        audio_input: str,
        audio_path: Path,
        capture_seconds: float,
    ) -> list[str]:
        return [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "pulse",
            "-i",
            audio_input,
            "-t",
            f"{capture_seconds:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ]

    def _default_pulse_monitor_source(self) -> Optional[str]:
        pactl = shutil.which("pactl")
        if not pactl:
            return None
        try:
            sink_result = subprocess.run(
                [pactl, "get-default-sink"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return None
        sink = str(sink_result.stdout or "").strip()
        source_names: list[str] = []
        try:
            sources_result = subprocess.run(
                [pactl, "list", "short", "sources"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            for line in str(sources_result.stdout or "").splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    source_names.append(parts[1])
        except Exception:
            source_names = []
        if sink:
            monitor = f"{sink}.monitor"
            if not source_names or monitor in source_names:
                return monitor
        for source in source_names:
            if source.endswith(".monitor"):
                return source
        return None

    def _read_whisper_json_result(self, directory: Path) -> dict[str, Any]:
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict):
                return data
        return {}

    def _short_hash(self, text: str) -> str:
        return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]

    async def _retrieve_youtube_transcript(
        self,
        url: str,
        *,
        media_context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        video_id = self._youtube_video_id(url)
        if not video_id:
            return {"status": "skipped", "reason": "no_video_id", "url": url}
        transcript_media_context = self._matching_youtube_media_context(video_id, url, media_context)
        cached = self._cached_transcript(video_id, url=url, media_context=transcript_media_context)
        if cached is not None:
            return cached
        provider = self._youtube_transcript_provider
        if callable(provider):
            try:
                provided = await self._call_maybe_async(provider, url, transcript_media_context)
                if isinstance(provided, dict):
                    return self._store_transcript_result(video_id, url, provided, media_context=transcript_media_context)
            except Exception as exc:
                return {
                    "status": "failed",
                    "reason": f"provider_failed:{type(exc).__name__}",
                    "url": url,
                    "video_id": video_id,
                    "error": str(exc),
                }
        yt_dlp = self.config.yt_dlp_path or shutil.which("yt-dlp")
        if not yt_dlp:
            return {"status": "unavailable", "reason": "yt_dlp_missing", "url": url, "video_id": video_id}
        with tempfile.TemporaryDirectory(prefix=f"yt_{video_id}_", dir=str(self._transcripts_dir())) as tmp_raw:
            tmp_dir = Path(tmp_raw)
            output = tmp_dir / "%(id)s.%(ext)s"
            command = [
                yt_dlp,
                "--skip-download",
                "--write-subs",
                "--write-auto-subs",
                "--sub-langs",
                "en.*,en",
                "--sub-format",
                "json3/vtt/best",
                "--output",
                str(output),
                url,
            ]
            try:
                completed = await asyncio.to_thread(
                    subprocess.run,
                    command,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except Exception as exc:
                return {
                    "status": "failed",
                    "reason": f"yt_dlp_failed:{type(exc).__name__}",
                    "url": url,
                    "video_id": video_id,
                    "error": str(exc),
                }
            transcript_data = self._read_downloaded_transcript_data(tmp_dir)
            transcript_text = str(transcript_data.get("transcript_text") or "")
            if not transcript_text:
                return {
                    "status": "unavailable",
                    "reason": "no_transcript_downloaded",
                    "url": url,
                    "video_id": video_id,
                    "stderr": str(completed.stderr or "")[-800:],
                }
            return self._store_transcript_result(
                video_id,
                url,
                {
                    "status": "available",
                    "transcript_text": transcript_text,
                    "segments": transcript_data.get("segments") or [],
                    "source": "yt-dlp",
                },
                media_context=transcript_media_context,
            )

    def _youtube_video_id(self, url: str) -> Optional[str]:
        try:
            parsed = urlparse(str(url or ""))
        except Exception:
            return None
        host = parsed.netloc.lower()
        if host.endswith("youtu.be"):
            video_id = parsed.path.strip("/").split("/", 1)[0]
            return video_id or None
        if "youtube.com" not in host:
            return None
        if parsed.path.startswith("/watch"):
            video_id = parse_qs(parsed.query).get("v", [""])[0]
            return video_id or None
        if parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/shorts/", 1)[1].strip("/").split("/", 1)[0]
            return video_id or None
        return None

    def _matching_youtube_media_context(
        self,
        video_id: str,
        url: str,
        media_context: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        matching: list[dict[str, Any]] = []
        for item in media_context or []:
            if not isinstance(item, dict):
                continue
            if self._media_item_matches_youtube_video(item, video_id=video_id, url=url):
                matching.append(item)
        return matching

    def _media_item_matches_youtube_video(
        self,
        item: dict[str, Any],
        *,
        video_id: str,
        url: str,
    ) -> bool:
        expected_id = str(video_id or self._youtube_video_id(url) or "").strip()
        expected_url = str(url or "").strip()
        candidates = [
            str(item.get("url") or "").strip(),
            str(item.get("title") or "").strip(),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            candidate_id = self._youtube_video_id(candidate)
            if expected_id and candidate_id == expected_id:
                return True
            if expected_url and candidate == expected_url:
                return True
        return False

    def _cached_transcript(
        self,
        video_id: str,
        *,
        url: str,
        media_context: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        text_path = self._transcripts_dir() / f"{video_id}.txt"
        meta_path = self._transcripts_dir() / f"{video_id}.json"
        segments_path = self._transcripts_dir() / f"{video_id}.segments.json"
        if not text_path.exists():
            return None
        try:
            transcript_text = text_path.read_text(encoding="utf-8")
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            segments = (
                json.loads(segments_path.read_text(encoding="utf-8"))
                if segments_path.exists()
                else []
            )
        except Exception:
            return None
        if not segments_path.exists() and "timed_segment_count" not in meta:
            return None
        return self._transcript_payload(
            video_id,
            url,
            transcript_text,
            source=str(meta.get("source") or "cache"),
            media_context=media_context,
            transcript_path=text_path,
            cached=True,
            segments=segments,
        )

    def _store_transcript_result(
        self,
        video_id: str,
        url: str,
        result: dict[str, Any],
        *,
        media_context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        transcript_text = str(result.get("transcript_text") or result.get("text") or "").strip()
        if not transcript_text:
            return {"status": "unavailable", "reason": "empty_transcript", "url": url, "video_id": video_id}
        segments = self._normalize_transcript_segments(
            result.get("segments") or result.get("transcript_segments") or []
        )
        text_path = self._transcripts_dir() / f"{video_id}.txt"
        meta_path = self._transcripts_dir() / f"{video_id}.json"
        segments_path = self._transcripts_dir() / f"{video_id}.segments.json"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(transcript_text, encoding="utf-8")
        if segments:
            segments_path.write_text(
                json.dumps(segments, indent=2, sort_keys=True, ensure_ascii=True),
                encoding="utf-8",
            )
        meta = {
            "url": url,
            "video_id": video_id,
            "source": str(result.get("source") or "provider"),
            "created_at": self._now().isoformat(),
            "media_context": media_context,
            "timed_segment_count": len(segments),
        }
        if segments:
            meta["segments_path"] = str(segments_path)
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
        return self._transcript_payload(
            video_id,
            url,
            transcript_text,
            source=meta["source"],
            media_context=media_context,
            transcript_path=text_path,
            cached=False,
            segments=segments,
        )

    def _transcript_payload(
        self,
        video_id: str,
        url: str,
        transcript_text: str,
        *,
        source: str,
        media_context: list[dict[str, Any]],
        transcript_path: Path,
        cached: bool,
        segments: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        timed_segments = self._normalize_transcript_segments(segments or [])
        excerpt_limit = min(
            len(transcript_text),
            self.config.youtube_transcript_max_chars or len(transcript_text),
            self.config.youtube_transcript_excerpt_chars,
        )
        transcript_media_context = self._matching_youtube_media_context(video_id, url, media_context)
        excerpt, excerpt_meta = self._transcript_excerpt_for_media_position(
            transcript_text,
            transcript_media_context,
            limit=excerpt_limit,
            segments=timed_segments,
        )
        title = ""
        artist = ""
        for item in transcript_media_context:
            if not title:
                title = str(item.get("title") or "")
            if not artist:
                artist = str(item.get("artist") or "")
        return {
            "status": "available",
            "url": url,
            "video_id": video_id,
            "title": title,
            "artist": artist,
            "source": source,
            "cached": cached,
            "transcript_path": str(transcript_path),
            "transcript_chars": len(transcript_text),
            "transcript_excerpt": excerpt,
            "transcript_full_available": True,
            "transcript_timed_segment_count": len(timed_segments),
            **excerpt_meta,
        }

    def _transcript_excerpt_for_media_position(
        self,
        transcript_text: str,
        media_context: list[dict[str, Any]],
        *,
        limit: int,
        segments: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, dict[str, Any]]:
        text = str(transcript_text or "").strip()
        if not text or limit <= 0:
            return "", {
                "transcript_excerpt_basis": "empty",
                "transcript_excerpt_start_char": 0,
                "transcript_excerpt_end_char": 0,
                "transcript_progress_percent": None,
                "transcript_position_label": "",
            }
        bounded_limit = max(1, min(len(text), int(limit)))
        position_us = self._primary_media_position_us(media_context)
        target_seconds = (float(position_us) / 1_000_000.0) if position_us is not None else None
        timed_segments = self._normalize_transcript_segments(segments or [])
        duration_meta = self._transcript_duration_meta(timed_segments, media_context)
        progress = self._effective_media_progress_percent(media_context, duration_meta)
        position_label = self._effective_media_position_label(media_context, duration_meta)
        if target_seconds is not None and timed_segments:
            position_warning = self._caption_position_warning(
                timed_segments,
                target_seconds,
                media_context=media_context,
            )
            if position_warning:
                excerpt = self._caption_start_excerpt(timed_segments, limit=bounded_limit)
                return excerpt, {
                    "transcript_excerpt_basis": "position_out_of_range",
                    "transcript_excerpt_start_char": self._safe_find(
                        text,
                        self._strip_timecode_prefix(excerpt),
                    ),
                    "transcript_excerpt_end_char": None,
                    "transcript_progress_percent": None,
                    "transcript_position_label": self._timing_uncertain_position_label(media_context),
                    "transcript_target_seconds": round(target_seconds, 3),
                    "transcript_position_valid": False,
                    "transcript_position_warning": position_warning,
                    "transcript_aligned_start_seconds": None,
                    "transcript_aligned_end_seconds": None,
                    "transcript_aligned_start_label": "",
                    "transcript_aligned_end_label": "",
                    "transcript_alignment_confidence": 0.0,
                    "transcript_aligned_segments": [],
                    **duration_meta,
                }
            timestamp_excerpt = self._caption_timestamp_excerpt(
                timed_segments,
                target_seconds=target_seconds,
                limit=bounded_limit,
            )
            if timestamp_excerpt is not None:
                excerpt, timestamp_meta = timestamp_excerpt
                return excerpt, {
                    "transcript_excerpt_basis": "caption_timestamp_reached",
                    "transcript_excerpt_start_char": self._safe_find(
                        text,
                        self._strip_timecode_prefix(excerpt),
                    ),
                    "transcript_excerpt_end_char": None,
                    "transcript_progress_percent": round(progress, 3) if progress is not None else None,
                    "transcript_position_label": position_label,
                    "transcript_target_seconds": round(target_seconds, 3),
                    "transcript_position_valid": True,
                    **duration_meta,
                    **timestamp_meta,
                }
            cutoff_seconds = max(0.0, target_seconds - float(self.config.youtube_transcript_commentary_lag_seconds))
            return "", {
                "transcript_excerpt_basis": "caption_timestamp_waiting",
                "transcript_excerpt_start_char": 0,
                "transcript_excerpt_end_char": 0,
                "transcript_progress_percent": round(progress, 3) if progress is not None else None,
                "transcript_position_label": position_label,
                "transcript_target_seconds": round(target_seconds, 3),
                "transcript_position_valid": True,
                "transcript_aligned_start_seconds": None,
                "transcript_aligned_end_seconds": None,
                "transcript_aligned_start_label": "",
                "transcript_aligned_end_label": "",
                "transcript_alignment_confidence": 0.0,
                "transcript_commentary_cutoff_seconds": round(cutoff_seconds, 3),
                "transcript_commentary_cutoff_label": self._format_seconds_label(cutoff_seconds),
                "transcript_aligned_segments": [],
                **duration_meta,
            }
        if progress is None:
            start = 0
            basis = "start"
        else:
            effective_progress = progress
            length_seconds = self._primary_media_length_seconds(media_context)
            if target_seconds is not None and length_seconds and length_seconds > 0:
                cutoff_seconds = max(
                    0.0,
                    target_seconds - float(self.config.youtube_transcript_commentary_lag_seconds),
                )
                effective_progress = max(0.0, min(100.0, (cutoff_seconds / length_seconds) * 100.0))
            center = int(round((max(0.0, min(100.0, effective_progress)) / 100.0) * len(text)))
            start = max(0, center - (bounded_limit // 2))
            basis = "playback_position"
        end = min(len(text), start + bounded_limit)
        start = max(0, end - bounded_limit)
        if start > 0:
            snapped_start = text.find(" ", start)
            if snapped_start != -1 and snapped_start < end:
                start = snapped_start + 1
        if end < len(text):
            snapped_end = text.rfind(" ", start, end)
            if snapped_end > start:
                end = snapped_end
        excerpt = text[start:end].strip()
        return excerpt, {
            "transcript_excerpt_basis": basis,
            "transcript_excerpt_start_char": start,
            "transcript_excerpt_end_char": end,
            "transcript_progress_percent": round(progress, 3) if progress is not None else None,
            "transcript_position_label": position_label,
            **duration_meta,
        }

    def _transcript_duration_meta(
        self,
        segments: list[dict[str, Any]],
        media_context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        caption_duration = self._caption_duration_seconds(segments)
        reported_length = self._primary_media_length_seconds(media_context)
        if caption_duration is not None:
            meta["transcript_caption_duration_seconds"] = round(caption_duration, 3)
            meta["transcript_caption_duration_label"] = self._format_seconds_label(caption_duration)
        if reported_length is not None:
            meta["media_reported_length_seconds"] = round(reported_length, 3)
            meta["media_reported_length_label"] = self._format_seconds_label(reported_length)
        if caption_duration is None or reported_length is None:
            return meta
        if caption_duration > reported_length + 120.0 and reported_length / max(caption_duration, 1.0) < 0.95:
            meta["media_duration_reliable"] = False
            meta["media_duration_warning"] = (
                f"media backend reported duration {self._format_seconds_label(reported_length)}, "
                f"but timestamped captions extend to {self._format_seconds_label(caption_duration)}; "
                "do not treat the reported duration or 100% progress as the video ending"
            )
        else:
            meta["media_duration_reliable"] = True
        return meta

    def _caption_duration_seconds(self, segments: list[dict[str, Any]]) -> Optional[float]:
        last_end: Optional[float] = None
        for segment in segments or []:
            if not isinstance(segment, dict):
                continue
            end_seconds = self._coerce_float(segment.get("end_seconds"), default=-1.0)
            if end_seconds < 0:
                continue
            if last_end is None or end_seconds > last_end:
                last_end = end_seconds
        return last_end

    def _effective_media_progress_percent(
        self,
        media_context: list[dict[str, Any]],
        duration_meta: dict[str, Any],
    ) -> Optional[float]:
        if duration_meta.get("media_duration_reliable") is False:
            position_us = self._primary_media_position_us(media_context)
            caption_duration = self._coerce_float(
                duration_meta.get("transcript_caption_duration_seconds"),
                default=0.0,
            )
            if position_us is not None and caption_duration > 0:
                return self._media_progress_percent(position_us, int(round(caption_duration * 1_000_000)))
        return self._primary_media_progress_percent(media_context)

    def _effective_media_position_label(
        self,
        media_context: list[dict[str, Any]],
        duration_meta: dict[str, Any],
    ) -> str:
        if duration_meta.get("media_duration_reliable") is False:
            position_us = self._primary_media_position_us(media_context)
            caption_duration = self._coerce_float(
                duration_meta.get("transcript_caption_duration_seconds"),
                default=0.0,
            )
            if position_us is not None and caption_duration > 0:
                return self._format_media_position(position_us, int(round(caption_duration * 1_000_000)))
        return self._primary_media_position_label(media_context)

    def _timing_uncertain_position_label(self, media_context: list[dict[str, Any]]) -> str:
        position_us = self._primary_media_position_us(media_context)
        if position_us is None:
            return "timing uncertain"
        return f"position {self._format_microseconds(position_us)} (timing uncertain)"

    def _apply_youtube_transcript_timing_to_media_context(
        self,
        capture_result: dict[str, Any],
        transcript: Any,
    ) -> None:
        if not isinstance(capture_result, dict) or not isinstance(transcript, dict):
            return
        if transcript.get("status") != "available":
            return
        media_context = capture_result.get("media_context")
        if not isinstance(media_context, list):
            return
        video_id = str(transcript.get("video_id") or "").strip()
        url = str(transcript.get("url") or "").strip()
        caption_duration = self._coerce_float(
            transcript.get("transcript_caption_duration_seconds"),
            default=0.0,
        )
        duration_warning = str(transcript.get("media_duration_warning") or "").strip()
        position_warning = str(transcript.get("transcript_position_warning") or "").strip()
        for item in media_context:
            if not isinstance(item, dict):
                continue
            if not self._media_item_matches_youtube_video(item, video_id=video_id, url=url):
                continue
            if transcript.get("media_duration_reliable") is False and caption_duration > 0:
                reported_length_us = self._coerce_int(item.get("length_us"))
                if reported_length_us is not None:
                    item["reported_length_us"] = reported_length_us
                    item["reported_position_label"] = str(item.get("position_label") or "")
                caption_length_us = int(round(caption_duration * 1_000_000))
                position_us = self._coerce_int(item.get("position_us"))
                item["length_us"] = caption_length_us
                item["position_label"] = self._format_media_position(position_us, caption_length_us)
                item["progress_percent"] = self._media_progress_percent(position_us, caption_length_us)
                item["duration_source"] = "caption_timestamps"
                if duration_warning:
                    item["media_timing_warning"] = duration_warning
                continue
            if transcript.get("transcript_position_valid") is False:
                item["reported_position_label"] = str(item.get("position_label") or "")
                item["position_label"] = self._timing_uncertain_position_label([item])
                item["progress_percent"] = None
                item["duration_source"] = "unreliable_media_backend"
                item["media_timing_warning"] = position_warning or (
                    "media backend position is outside the timestamped transcript range"
                )

    def _caption_position_warning(
        self,
        segments: list[dict[str, Any]],
        target_seconds: float,
        *,
        media_context: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        if not segments:
            return ""
        first_start = self._coerce_float(segments[0].get("start_seconds"), default=0.0)
        last_end = self._coerce_float(segments[-1].get("end_seconds"), default=first_start)
        tolerance_seconds = 30.0
        if target_seconds < first_start - tolerance_seconds or target_seconds > last_end + tolerance_seconds:
            return (
                f"media position {self._format_seconds_label(target_seconds)} is outside caption range "
                f"{self._format_seconds_label(first_start)}-{self._format_seconds_label(last_end)}"
            )
        return ""

    def _caption_start_excerpt(self, segments: list[dict[str, Any]], *, limit: int) -> str:
        selected: list[dict[str, Any]] = []
        current_len = 0
        effective_limit = max(int(limit), 400)
        for segment in segments:
            text = str(segment.get("text") or "")
            projected = current_len + len(text) + 1
            if selected and projected > effective_limit:
                break
            selected.append(segment)
            current_len = projected
        lines = []
        for segment in selected:
            start_label = self._format_seconds_label(
                self._coerce_float(segment.get("start_seconds"), default=0.0)
            )
            end_seconds = self._coerce_float(segment.get("end_seconds"), default=-1.0)
            end_label = self._format_seconds_label(end_seconds) if end_seconds >= 0 else ""
            label = f"[{start_label}-{end_label}]" if end_label else f"[{start_label}]"
            lines.append(f"{label} {segment.get('text')}")
        return " ".join(lines).strip()

    def _caption_opening_excerpt(
        self,
        segments: list[dict[str, Any]],
        *,
        limit: int,
        max_seconds: float = 120.0,
    ) -> str:
        opening = [
            segment
            for segment in segments
            if self._coerce_float(segment.get("start_seconds"), default=0.0) <= max_seconds
        ]
        return self._caption_start_excerpt(opening or segments[:1], limit=limit)

    def _caption_timestamp_excerpt(
        self,
        segments: list[dict[str, Any]],
        *,
        target_seconds: float,
        limit: int,
    ) -> Optional[tuple[str, dict[str, Any]]]:
        if not segments:
            return None
        cutoff_seconds = max(0.0, target_seconds - float(self.config.youtube_transcript_commentary_lag_seconds))
        current_index = self._nearest_transcript_segment_index(segments, cutoff_seconds)
        if current_index is None:
            return None
        selected = [
            segment
            for segment in segments
            if self._segment_completed_by_cutoff(segment, cutoff_seconds)
            and self._segment_overlaps_window(
                segment,
                start_seconds=cutoff_seconds - 30.0,
                end_seconds=cutoff_seconds,
            )
        ]
        if not selected:
            completed_before_cutoff = [
                segment for segment in segments if self._segment_completed_by_cutoff(segment, cutoff_seconds)
            ]
            if not completed_before_cutoff:
                return None
            selected = [completed_before_cutoff[-1]]
        if sum(len(str(item.get("text") or "")) + 16 for item in selected) > limit:
            fallback_selected = selected[:1]
            bounded = self._bounded_segments_around_index(segments, current_index, limit)
            selected = [
                segment for segment in bounded if self._segment_completed_by_cutoff(segment, cutoff_seconds)
            ] or fallback_selected
        lines = []
        for segment in selected:
            start_label = self._format_seconds_label(
                self._coerce_float(segment.get("start_seconds"), default=0.0)
            )
            end_seconds = self._coerce_float(segment.get("end_seconds"), default=-1.0)
            end_label = self._format_seconds_label(end_seconds) if end_seconds >= 0 else ""
            label = f"[{start_label}-{end_label}]" if end_label else f"[{start_label}]"
            lines.append(f"{label} {segment.get('text')}")
        excerpt = " ".join(lines).strip()
        first = selected[0]
        last = selected[-1]
        start_seconds = self._coerce_float(first.get("start_seconds"), default=target_seconds)
        end_seconds = self._coerce_float(last.get("end_seconds"), default=start_seconds)
        return excerpt, {
            "transcript_aligned_start_seconds": round(start_seconds, 3),
            "transcript_aligned_end_seconds": round(end_seconds, 3),
            "transcript_aligned_start_label": self._format_seconds_label(start_seconds),
            "transcript_aligned_end_label": self._format_seconds_label(end_seconds),
            "transcript_alignment_confidence": 1.0,
            "transcript_commentary_cutoff_seconds": round(cutoff_seconds, 3),
            "transcript_commentary_cutoff_label": self._format_seconds_label(cutoff_seconds),
            "transcript_aligned_segments": selected[:12],
        }

    def _nearest_transcript_segment_index(
        self,
        segments: list[dict[str, Any]],
        target_seconds: float,
    ) -> Optional[int]:
        best_index: Optional[int] = None
        best_distance: Optional[float] = None
        for index, segment in enumerate(segments):
            start = self._coerce_float(segment.get("start_seconds"), default=0.0)
            end = self._coerce_float(segment.get("end_seconds"), default=start)
            if start <= target_seconds <= max(start, end):
                return index
            midpoint = start + ((max(start, end) - start) / 2.0)
            distance = abs(midpoint - target_seconds)
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index

    def _segment_overlaps_window(
        self,
        segment: dict[str, Any],
        *,
        start_seconds: float,
        end_seconds: float,
    ) -> bool:
        segment_start = self._coerce_float(segment.get("start_seconds"), default=0.0)
        segment_end = self._coerce_float(segment.get("end_seconds"), default=segment_start)
        return segment_start <= end_seconds and max(segment_start, segment_end) >= start_seconds

    def _segment_completed_by_cutoff(self, segment: dict[str, Any], cutoff_seconds: float) -> bool:
        segment_start = self._coerce_float(segment.get("start_seconds"), default=0.0)
        segment_end = self._coerce_float(segment.get("end_seconds"), default=segment_start)
        return max(segment_start, segment_end) <= cutoff_seconds

    def _bounded_segments_around_index(
        self,
        segments: list[dict[str, Any]],
        center_index: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        selected = [segments[center_index]]
        left = center_index - 1
        right = center_index + 1
        while left >= 0 or right < len(segments):
            current_len = sum(len(str(item.get("text") or "")) + 16 for item in selected)
            if current_len >= limit:
                break
            candidates: list[tuple[int, dict[str, Any]]] = []
            if left >= 0:
                candidates.append((left, segments[left]))
            if right < len(segments):
                candidates.append((right, segments[right]))
            if not candidates:
                break
            next_index, next_segment = min(candidates, key=lambda item: abs(item[0] - center_index))
            projected = current_len + len(str(next_segment.get("text") or "")) + 16
            if projected > limit:
                break
            selected.append(next_segment)
            selected.sort(key=lambda item: self._coerce_float(item.get("start_seconds"), default=0.0))
            if next_index == left:
                left -= 1
            else:
                right += 1
        return selected

    def _safe_find(self, haystack: str, needle: str) -> int:
        cleaned = str(needle or "").strip()
        if not cleaned:
            return -1
        position = str(haystack or "").find(cleaned[:120])
        return position if position >= 0 else -1

    def _strip_timecode_prefix(self, excerpt: str) -> str:
        return re.sub(r"(?:^|\s)\[[0-9:.]+(?:-[0-9:.]+)?\]\s*", " ", str(excerpt or "")).strip()

    def _primary_media_progress_percent(self, media_context: list[dict[str, Any]]) -> Optional[float]:
        for item in media_context or []:
            if not isinstance(item, dict):
                continue
            progress = item.get("progress_percent")
            if isinstance(progress, (int, float)):
                return float(progress)
            position_us = self._coerce_int(item.get("position_us"))
            length_us = self._coerce_int(item.get("length_us"))
            computed = self._media_progress_percent(position_us, length_us)
            if computed is not None:
                return computed
        return None

    def _primary_media_position_us(self, media_context: list[dict[str, Any]]) -> Optional[int]:
        for item in media_context or []:
            if not isinstance(item, dict):
                continue
            position_us = self._coerce_int(item.get("position_us"))
            if position_us is not None:
                return position_us
        return None

    def _primary_media_length_seconds(self, media_context: list[dict[str, Any]]) -> Optional[float]:
        for item in media_context or []:
            if not isinstance(item, dict):
                continue
            length_us = self._coerce_int(item.get("length_us"))
            if length_us is not None and length_us > 0:
                return float(length_us) / 1_000_000.0
        return None

    def _primary_media_position_label(self, media_context: list[dict[str, Any]]) -> str:
        for item in media_context or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("position_label") or "").strip()
            if label:
                return label
            position_us = self._coerce_int(item.get("position_us"))
            length_us = self._coerce_int(item.get("length_us"))
            label = self._format_media_position(position_us, length_us)
            if label:
                return label
        return ""

    def _read_downloaded_transcript(self, directory: Path) -> str:
        return str(self._read_downloaded_transcript_data(directory).get("transcript_text") or "")

    def _read_downloaded_transcript_data(self, directory: Path) -> dict[str, Any]:
        candidates = sorted(
            [
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in {".json3", ".vtt", ".srv1", ".srv2", ".srv3", ".ttml"}
            ],
            key=lambda item: item.stat().st_size if item.exists() else 0,
            reverse=True,
        )
        for path in candidates:
            try:
                raw = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if path.suffix.lower() == ".json3":
                data = self._json3_transcript_data(raw)
            else:
                data = self._subtitle_transcript_data(raw)
            if data.get("transcript_text"):
                return data
        return {"transcript_text": "", "segments": []}

    def _json3_transcript_text(self, raw: str) -> str:
        return str(self._json3_transcript_data(raw).get("transcript_text") or "")

    def _json3_transcript_data(self, raw: str) -> dict[str, Any]:
        try:
            data = json.loads(raw)
        except Exception:
            return {"transcript_text": "", "segments": []}
        segments: list[dict[str, Any]] = []
        untimed_chunks: list[str] = []
        for event in data.get("events", []):
            if not isinstance(event, dict):
                continue
            text = self._clean_transcript_line(
                "".join(
                    str(seg.get("utf8") or "")
                    for seg in event.get("segs", []) or []
                    if isinstance(seg, dict)
                )
            )
            if not text:
                continue
            start_ms = self._coerce_float(event.get("tStartMs"), default=-1.0)
            duration_ms = self._coerce_float(event.get("dDurationMs"), default=0.0)
            if start_ms >= 0:
                segments.append(
                    {
                        "start_seconds": round(start_ms / 1000.0, 3),
                        "end_seconds": round((start_ms + max(0.0, duration_ms)) / 1000.0, 3),
                        "text": text,
                    }
                )
            else:
                untimed_chunks.append(text)
        transcript_text = self._dedupe_transcript_lines(
            [str(item.get("text") or "") for item in segments] + untimed_chunks
        )
        return {"transcript_text": transcript_text, "segments": self._normalize_transcript_segments(segments)}

    def _subtitle_text(self, raw: str) -> str:
        return str(self._subtitle_transcript_data(raw).get("transcript_text") or "")

    def _subtitle_transcript_data(self, raw: str) -> dict[str, Any]:
        lines = str(raw or "").splitlines()
        segments: list[dict[str, Any]] = []
        untimed_lines: list[str] = []
        index = 0
        while index < len(lines):
            cleaned = lines[index].strip()
            if not cleaned or cleaned.upper().startswith("WEBVTT") or cleaned.startswith("NOTE"):
                index += 1
                continue
            if "-->" not in cleaned:
                if not cleaned.isdigit() and not cleaned.startswith(("<", "{")):
                    untimed_lines.append(re.sub(r"<[^>]+>", "", cleaned))
                index += 1
                continue
            start_seconds, end_seconds = self._parse_subtitle_time_range(cleaned)
            index += 1
            cue_lines: list[str] = []
            while index < len(lines):
                cue = lines[index].strip()
                if not cue:
                    index += 1
                    break
                if "-->" in cue:
                    break
                if not cue.startswith(("<", "{")):
                    cue_lines.append(re.sub(r"<[^>]+>", "", cue))
                index += 1
            text = self._clean_transcript_line(" ".join(cue_lines))
            if text and start_seconds is not None:
                segments.append(
                    {
                        "start_seconds": round(start_seconds, 3),
                        "end_seconds": round(
                            end_seconds if end_seconds is not None else start_seconds,
                            3,
                        ),
                        "text": text,
                    }
                )
            elif text:
                untimed_lines.append(text)
        transcript_text = self._dedupe_transcript_lines(
            [str(item.get("text") or "") for item in segments] + untimed_lines
        )
        return {"transcript_text": transcript_text, "segments": self._normalize_transcript_segments(segments)}

    def _parse_subtitle_time_range(self, line: str) -> tuple[Optional[float], Optional[float]]:
        parts = str(line or "").split("-->", 1)
        if len(parts) != 2:
            return None, None
        start = self._parse_subtitle_timestamp(parts[0].strip())
        end = self._parse_subtitle_timestamp(parts[1].split()[0].strip())
        return start, end

    def _parse_subtitle_timestamp(self, value: str) -> Optional[float]:
        match = re.search(r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?", str(value or ""))
        if not match:
            return None
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        millis = int((match.group(4) or "0").ljust(3, "0")[:3])
        return float((hours * 3600) + (minutes * 60) + seconds) + (millis / 1000.0)

    def _dedupe_transcript_lines(self, lines: list[str]) -> str:
        chunks: list[str] = []
        previous = ""
        for line in lines:
            cleaned = self._clean_transcript_line(line)
            if not cleaned or cleaned == previous:
                continue
            chunks.append(cleaned)
            previous = cleaned
        return " ".join(chunks).strip()

    def _clean_transcript_line(self, value: Any) -> str:
        return " ".join(re.sub(r"<[^>]+>", "", str(value or "")).split())

    def _normalize_transcript_segments(self, segments: Any) -> list[dict[str, Any]]:
        if not isinstance(segments, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in segments:
            if not isinstance(item, dict):
                continue
            text = self._clean_transcript_line(item.get("text"))
            if not text:
                continue
            start = self._coerce_float(item.get("start_seconds"), default=-1.0)
            if start < 0:
                start = self._coerce_float(item.get("start"), default=-1.0)
            if start < 0:
                continue
            end = self._coerce_float(item.get("end_seconds"), default=start)
            if end < start:
                end = start
            normalized.append(
                {
                    "start_seconds": round(start, 3),
                    "end_seconds": round(end, 3),
                    "text": text,
                }
            )
        normalized.sort(key=lambda item: item["start_seconds"])
        return normalized

    async def _analyze_capture(self, capture_result: dict[str, Any], *, reason: str) -> dict[str, Any]:
        llm = getattr(self.runtime, "llm", None)
        if llm is None or not hasattr(llm, "chat_completion"):
            return self._fallback_analysis(capture_result, reason=reason, fallback_reason="llm_unavailable")

        content_text = self._analysis_prompt(capture_result, reason=reason)
        content: Any = content_text
        image_uri = self._image_data_uri((capture_result.get("capture") or {}).get("path"))
        if self.config.vision_enabled and image_uri:
            content = [
                {"type": "text", "text": content_text},
                {"type": "image_url", "image_url": {"url": image_uri}},
            ]
        agent_name = resolve_agent_name(runtime=self.runtime)
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are {agent_name} reviewing a private screenshot of the operator's active desktop "
                    "for an explicitly enabled body-double collaboration skill. Decide whether a "
                    "short spoken interruption is useful. Do not speak just because an observation "
                    "interval happened. Do not read code, logs, stack traces, or "
                    "long technical text aloud. For those, summarize briefly and refer to a file. "
                    "Activity_summary is internal evidence only. spoken_text must not summarize what "
                    "the operator can already see or what a video is about. If speaking about media, "
                    "say a concise grounded viewpoint, critique, implication, or useful synthesis. "
                    "Return strict JSON with keys: should_speak boolean, activity_summary string, "
                    "reason string, spoken_text string, note string, speech_intent string, "
                    "speech_relevance_score number. Use speech_intent one of none, task_coaching, "
                    "screen_relevant, system_issue, safety_privacy. Only use task_coaching when the "
                    "prompt includes an explicit declared operator task."
                ),
            },
            {"role": "user", "content": content},
        ]
        try:
            response = await llm.chat_completion(
                messages=messages,
                model=self.config.vision_model,
                complexity="light",
                payload={"temperature": 0.2, "max_tokens": 600},
                source="desktop_context_observation",
                session_id=self._session_id(),
            )
            parsed = self._parse_response_json(response)
            if isinstance(parsed, dict):
                return self._normalize_analysis(parsed)
        except Exception as exc:
            if content is not content_text:
                try:
                    response = await llm.chat_completion(
                        messages=[
                            messages[0],
                            {
                                "role": "user",
                                "content": (
                                    content_text
                                    + f"\n\nVision input failed with {type(exc).__name__}; use OCR and metadata only."
                                ),
                            },
                        ],
                        model=self.config.vision_model,
                        complexity="light",
                        payload={"temperature": 0.2, "max_tokens": 600},
                        source="desktop_context_observation_fallback",
                        session_id=self._session_id(),
                    )
                    parsed = self._parse_response_json(response)
                    if isinstance(parsed, dict):
                        return self._normalize_analysis(parsed)
                except Exception:
                    pass
            return self._fallback_analysis(capture_result, reason=reason, fallback_reason=str(exc))
        return self._fallback_analysis(capture_result, reason=reason, fallback_reason="unparseable_llm_response")

    async def _analyze_self_interest_overlap(
        self,
        capture_result: dict[str, Any],
        analysis: dict[str, Any],
        *,
        interests: list[dict[str, str]],
        reason: str,
    ) -> dict[str, Any]:
        llm = getattr(self.runtime, "llm", None)
        if llm is None or not hasattr(llm, "chat_completion"):
            return {
                "matches_self_interest": False,
                "confidence": 0.0,
                "salience": 0.0,
                "reason": "llm_unavailable",
            }
        prompt = self._self_interest_overlap_prompt(capture_result, analysis, interests=interests, reason=reason)
        agent_name = resolve_agent_name(runtime=self.runtime)
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are {agent_name} comparing a live desktop media observation against "
                    "the agent's durable self-interest evidence, active shared-work evidence, and "
                    "OpenCAS project-context evidence. "
                    "Do not invent interests, goals, opinions, or memories. A match is valid only "
                    "when the screen/media/transcript evidence and one supplied relevance item overlap. "
                    "Your job is not to summarize the media. Generate a compact, grounded thought "
                    "artifact from the supplied evidence: viewpoint, relevance, implications, "
                    "questions, and whether it is useful to say now. "
                    "Do not copy wording from these instructions into any output field. If the "
                    "only available wording is a general product principle, leave spoken_text empty. "
                    "Return strict JSON with keys: matches_observed_context boolean, matches_self_interest "
                    "boolean, matches_shared_work boolean, match_category string, matched_targets list, "
                    "matched_interests list, connection_summary string, why_it_matters string, "
                    "agent_viewpoint string, implications list, open_questions list, "
                    "self_directed_next_step string, operator_interruption_warranted boolean, spoken_text "
                    "string, confidence number, salience number, novelty number, timecode_or_position string. "
                    "spoken_text should be the concise verbal version of agent_viewpoint, not a recap."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            response = await llm.chat_completion(
                messages=messages,
                model=self.config.vision_model,
                complexity="light",
                payload={"temperature": 0.25, "max_tokens": 900},
                source="desktop_context_self_interest_overlap",
                session_id=self._session_id(),
            )
            parsed = self._parse_response_json(response)
            if isinstance(parsed, dict):
                return self._normalize_self_interest_followup(parsed)
        except Exception as exc:
            return {
                "matches_self_interest": False,
                "confidence": 0.0,
                "salience": 0.0,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        return {
            "matches_self_interest": False,
            "confidence": 0.0,
            "salience": 0.0,
            "reason": "unparseable_llm_response",
        }

    def _self_interest_overlap_prompt(
        self,
        capture_result: dict[str, Any],
        analysis: dict[str, Any],
        *,
        interests: list[dict[str, str]],
        reason: str,
    ) -> str:
        media_lines = self._media_prompt_lines(capture_result.get("media_context"))
        media_change_lines = self._media_state_change_prompt_lines(capture_result.get("media_state_changes"))
        commentary_lines = self._media_commentary_mode_prompt_lines(capture_result.get("media_state_changes"))
        transcript_lines = self._youtube_transcript_prompt_lines(capture_result.get("youtube_transcript"))
        live_transcript_lines = self._live_transcript_prompt_lines(
            capture_result.get("live_transcript"),
            capture_result.get("youtube_transcript"),
        )
        interest_lines = [
            (
                f"- {item.get('label') or 'relevance item'} "
                f"[{item.get('category') or 'unknown'}; {item.get('source') or 'unknown'}]: "
                f"{item.get('text') or ''}"
            )
            for item in interests
        ]
        capture = capture_result.get("capture") or {}
        return "\n".join(
            [
                "Decide whether this observed desktop/media/website context is relevant enough to preserve as follow-up.",
                f"Observation reason: {reason}",
                f"Screenshot evidence: {capture.get('path') or ''}",
                f"Observed activity summary: {analysis.get('activity_summary') or ''}",
                "Relevance evidence from Bulma's self-interests and shared active work:",
                *interest_lines,
                *media_lines,
                *self._media_ambiguity_prompt_lines(capture_result),
                *media_change_lines,
                *commentary_lines,
                *transcript_lines,
                *live_transcript_lines,
                "OCR excerpt:",
                str(capture_result.get("ocr_text") or "")[:1800] or "(no OCR text)",
                "Body-double stance: while this skill is enabled, treat the visible desktop and currently "
                "playing media as intentionally shared context for the agent to think about. Do not wait "
                "for the operator to separately declare that a video, page, or screen is relevant.",
                "If there is a match, propose what should happen next. "
                "For self-interest matches this may be self-directed exploration; for shared-work matches "
                "it should help the operator and agent use the observed context for active work. "
                "Create a thought artifact, not a media summary: include only claims grounded in "
                "the observed segment, supplied relevance evidence, or active work context. "
                "The transcript excerpt is centered near the current playback position when possible; "
                "do not speak about unreached future transcript content as if the operator has already "
                "seen it. If a useful comment belongs later in the video, set interruption false and "
                "include the intended timecode_or_position for later reconsideration. "
                "Only set operator_interruption_warranted when the comment is concrete, timely, concise, "
                "and anchored in this observed moment. Leave spoken_text empty for generic principles.",
            ]
        )

    def _normalize_self_interest_followup(self, raw: dict[str, Any]) -> dict[str, Any]:
        matched = raw.get("matched_interests")
        if isinstance(matched, str):
            matched_interests = [matched]
        elif isinstance(matched, list):
            matched_interests = [str(item) for item in matched if str(item or "").strip()]
        else:
            matched_interests = []
        matched_targets_raw = raw.get("matched_targets") or raw.get("matched_work") or matched_interests
        if isinstance(matched_targets_raw, str):
            matched_targets = [matched_targets_raw]
        elif isinstance(matched_targets_raw, list):
            matched_targets = [str(item) for item in matched_targets_raw if str(item or "").strip()]
        else:
            matched_targets = []
        confidence = self._coerce_float(raw.get("confidence"), default=0.0)
        salience = self._coerce_float(raw.get("salience"), default=confidence)
        novelty = self._coerce_float(raw.get("novelty"), default=0.5)
        matches_self_interest = bool(raw.get("matches_self_interest"))
        matches_shared_work = bool(raw.get("matches_shared_work") or raw.get("matches_active_work"))
        matches_observed_context = bool(raw.get("matches_observed_context")) or matches_self_interest or matches_shared_work
        match_category = str(raw.get("match_category") or "").strip().lower()
        if not match_category:
            match_category = "self_interest" if matches_self_interest else "shared_work" if matches_shared_work else "observed_context"
        connection_summary = self._scrub_generated_followup_text(raw.get("connection_summary"))
        why_it_matters = self._scrub_generated_followup_text(raw.get("why_it_matters"))
        agent_viewpoint = self._scrub_generated_followup_text(
            raw.get("agent_viewpoint")
            or raw.get("viewpoint")
            or raw.get("opinion")
            or raw.get("takeaway")
            or ""
        )
        implications = [
            item
            for item in self._normalize_text_list(
                raw.get("implications") or raw.get("design_implications") or raw.get("useful_implications")
            )
            if self._scrub_generated_followup_text(item)
        ][:6]
        open_questions = [
            item
            for item in self._normalize_text_list(
                raw.get("open_questions") or raw.get("questions") or raw.get("research_questions")
            )
            if self._scrub_generated_followup_text(item)
        ][:6]
        self_directed_next_step = self._scrub_generated_followup_text(raw.get("self_directed_next_step"))
        spoken_text = self._scrub_generated_followup_text(raw.get("spoken_text"))
        return {
            "matches_observed_context": matches_observed_context,
            "matches_self_interest": matches_self_interest,
            "matches_shared_work": matches_shared_work,
            "match_category": match_category,
            "matched_targets": matched_targets[:8],
            "matched_interests": matched_interests[:6],
            "connection_summary": connection_summary,
            "why_it_matters": why_it_matters,
            "agent_viewpoint": agent_viewpoint,
            "implications": implications,
            "open_questions": open_questions,
            "artifact_title": str(raw.get("artifact_title") or raw.get("title") or "").strip(),
            "artifact_kind": str(raw.get("artifact_kind") or "observed_media_thought").strip(),
            "self_directed_next_step": self_directed_next_step,
            "operator_interruption_warranted": bool(raw.get("operator_interruption_warranted")),
            "spoken_text": spoken_text,
            "confidence": max(0.0, min(1.0, confidence)),
            "salience": max(0.0, min(1.0, salience)),
            "novelty": max(0.0, min(1.0, novelty)),
            "timecode_or_position": str(raw.get("timecode_or_position") or "").strip(),
            "raw": raw,
        }

    def _normalize_text_list(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item or "").strip()]
        return []

    def _heuristic_observed_context_followup(
        self,
        capture_result: dict[str, Any],
        analysis: dict[str, Any],
        *,
        interests: list[dict[str, str]],
    ) -> Optional[dict[str, Any]]:
        text = self._observed_context_text_for_matching(capture_result, analysis)
        tokens = self._relevance_tokens(text)
        if len(tokens) < 4:
            return None
        lower_text = text.lower()
        high_signal_terms = {
            "agent",
            "agents",
            "assistant",
            "assistants",
            "autonomous",
            "autonomy",
            "chatbot",
            "chatbots",
            "context",
            "daydream",
            "embeddings",
            "guardrails",
            "interrupt",
            "learning",
            "memory",
            "openclaw",
            "opencas",
            "persistent",
            "proactive",
            "proactivity",
            "reasoning",
            "schedules",
            "tool",
            "tools",
            "workspace",
        }
        strong_phrases = [
            "another chatbot",
            "another agent",
            "another thing to manage",
            "management overhead",
            "proactive assistant",
            "proactive agents",
            "know when to interrupt",
            "understand context",
            "waiting for me to assign",
        ]
        high_overlap = tokens & high_signal_terms
        phrase_bonus = 0.18 if any(phrase in lower_text for phrase in strong_phrases) else 0.0
        best: Optional[dict[str, Any]] = None
        for item in interests:
            target_text = " ".join(
                str(item.get(key) or "")
                for key in ("label", "text", "category", "source")
            )
            target_tokens = self._relevance_tokens(target_text)
            overlap = tokens & target_tokens
            category = str(item.get("category") or "observed_context").strip().lower()
            is_work_context = category in {"project_context", "shared_work", "active_work", "recent_context"}
            category_bonus = 0.12 if is_work_context else 0.04
            if not overlap and not (is_work_context and high_overlap and phrase_bonus):
                continue
            if is_work_context and len(high_overlap) < 2 and len(overlap) < 3:
                continue
            score = min(
                1.0,
                0.45
                + min(0.22, len(overlap) * 0.035)
                + min(0.25, len(high_overlap) * 0.035)
                + category_bonus
                + phrase_bonus,
            )
            if score < self.config.observed_context_relevance_match_threshold:
                continue
            candidate = {
                "item": item,
                "category": category,
                "score": score,
                "overlap": sorted(overlap)[:12],
                "high_overlap": sorted(high_overlap)[:12],
                "is_work_context": is_work_context,
            }
            if best is None or score > best["score"]:
                best = candidate
        if best is None:
            return None
        item = best["item"]
        category = "project_context" if best["category"] == "project_context" else best["category"]
        matches_self_interest = best["category"] == "self_interest"
        matches_shared_work = bool(best["is_work_context"])
        media_title = self._primary_media_title(capture_result)
        position = self._current_media_position_label(capture_result)
        target_label = str(item.get("label") or item.get("source") or category).strip()
        connection = self._heuristic_connection_summary(media_title, category, best["high_overlap"])
        confidence = float(best["score"])
        salience = min(1.0, confidence + 0.03)
        return {
            "matches_observed_context": True,
            "matches_self_interest": matches_self_interest,
            "matches_shared_work": matches_shared_work,
            "match_category": category,
            "matched_targets": [target_label],
            "matched_interests": [target_label] if matches_self_interest else [],
            "connection_summary": connection,
            "agent_viewpoint": "",
            "heuristic_evidence_summary": self._heuristic_agent_viewpoint(media_title, category, best["high_overlap"]),
            "why_it_matters": "",
            "implications": [],
            "open_questions": [],
            "artifact_title": media_title or "Observed media thought",
            "artifact_kind": "observed_media_thought",
            "heuristic_fallback": True,
            "self_directed_next_step": "",
            "operator_interruption_warranted": False,
            "spoken_text": "",
            "confidence": confidence,
            "salience": salience,
            "novelty": 0.45,
            "timecode_or_position": position,
            "raw": {
                "source": "desktop_context.heuristic_observed_context_relevance",
                "token_overlap": best["overlap"],
                "high_signal_overlap": best["high_overlap"],
            },
        }

    def _observed_context_text_for_matching(self, capture_result: dict[str, Any], analysis: dict[str, Any]) -> str:
        parts: list[str] = []
        parts.append(str(analysis.get("activity_summary") or ""))
        media_context = capture_result.get("media_context")
        if isinstance(media_context, list):
            for item in media_context[:3]:
                if isinstance(item, dict):
                    parts.extend(
                        str(item.get(key) or "")
                        for key in ("title", "artist", "url", "position_label")
                    )
        transcript = capture_result.get("youtube_transcript")
        if isinstance(transcript, dict):
            parts.extend(
                str(transcript.get(key) or "")
                for key in (
                    "title",
                    "artist",
                    "url",
                    "transcript_position_label",
                    "transcript_excerpt",
                )
            )
        live_transcript = capture_result.get("live_transcript")
        if isinstance(live_transcript, dict):
            parts.extend(
                str(live_transcript.get(key) or "")
                for key in (
                    "media_title",
                    "media_artist",
                    "media_url",
                    "media_position_label",
                    "transcript_excerpt",
                    "transcript_text",
                )
            )
        parts.append(str(capture_result.get("ocr_text") or ""))
        return "\n".join(part for part in parts if part)

    def _relevance_tokens(self, text: str) -> set[str]:
        stopwords = {
            "about",
            "after",
            "again",
            "also",
            "another",
            "because",
            "before",
            "being",
            "could",
            "current",
            "enough",
            "from",
            "have",
            "into",
            "more",
            "need",
            "that",
            "their",
            "there",
            "thing",
            "this",
            "through",
            "user",
            "video",
            "when",
            "while",
            "with",
            "work",
            "would",
        }
        tokens: set[str] = set()
        for raw in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", str(text or "").lower()):
            token = raw.strip("_-")
            if token and token not in stopwords:
                tokens.add(token)
            if token.endswith("s") and len(token) > 4:
                singular = token[:-1]
                if singular not in stopwords:
                    tokens.add(singular)
        return tokens

    def _heuristic_connection_summary(
        self,
        media_title: str,
        category: str,
        high_overlap: list[str],
    ) -> str:
        title = media_title or "The current media"
        terms = ", ".join(high_overlap[:5])
        target = "OpenCAS project context" if category == "project_context" else "active relevance context"
        if terms:
            return f"{title} overlaps {target} through {terms}."
        return f"{title} overlaps {target}."

    def _heuristic_spoken_comment(
        self,
        media_title: str,
        category: str,
        high_overlap: list[str],
    ) -> str:
        if category == "project_context":
            return ""
        if high_overlap:
            return ""
        return ""

    def _heuristic_agent_viewpoint(
        self,
        media_title: str,
        category: str,
        high_overlap: list[str],
    ) -> str:
        if category == "project_context":
            return (
                f"Heuristic relevance match: {media_title or 'current media'} overlaps OpenCAS project context. "
                "This is evidence for storage and later review, not generated agent viewpoint."
            )
        terms = ", ".join(high_overlap[:4])
        if terms:
            return f"Heuristic relevance match: current media overlaps active OpenCAS work terms: {terms}."
        return f"Heuristic relevance match: {media_title or 'current media'} overlaps active OpenCAS work context."

    def _analysis_speech_is_media_summary(self, capture_result: dict[str, Any], analysis: dict[str, Any]) -> bool:
        if not analysis.get("should_speak") or not self._has_playing_video_context(capture_result):
            return False
        text = str(analysis.get("spoken_text") or "").strip()
        if not text or self._spoken_text_has_viewpoint(text):
            return False
        lower = text.lower()
        summary_phrases = [
            "the video is about",
            "this video is about",
            "the video discusses",
            "this video discusses",
            "this part is directly about",
            "the current transcript segment is",
            "you are watching",
            "the desktop shows",
            "the screen shows",
            "a youtube video about",
        ]
        if any(phrase in lower for phrase in summary_phrases):
            return True
        return bool(
            re.search(
                r"\b(video|segment|screen|desktop|transcript)\b.{0,80}\b(about|discuss|shows|playing|visible)\b",
                lower,
            )
        )

    def _spoken_text_has_viewpoint(self, text: str) -> bool:
        lower = str(text or "").lower()
        markers = [
            "i think",
            "my take",
            "my read",
            "my view",
            "i would",
            "i disagree",
            "i agree",
            "what matters",
            "the implication",
            "this suggests",
            "this changes",
            "we should",
            "open question",
            "the useful",
            "the risk",
            "the failure mode",
            "earned the interruption",
            "grounded synthesis",
        ]
        return any(marker in lower for marker in markers)

    def _suppress_media_summary_speech(self, analysis: dict[str, Any]) -> None:
        analysis["should_speak"] = False
        analysis["spoken_text"] = ""
        analysis["speech_intent"] = "none"
        analysis["speech_relevance_score"] = 0.0
        analysis["speech_policy"] = "suppressed_media_summary_without_viewpoint"
        analysis["reason"] = (
            "Media narration was suppressed because body-double speech should express a grounded "
            "viewpoint or useful synthesis, not summarize what the operator is already watching."
        )

    async def _analyze_region_prompt(
        self,
        capture_result: dict[str, Any],
        *,
        prompt: str,
        source: str,
    ) -> dict[str, Any]:
        llm = getattr(self.runtime, "llm", None)
        if llm is None or not hasattr(llm, "chat_completion"):
            return self._region_prompt_fallback_analysis(
                capture_result,
                prompt=prompt,
                fallback_reason="llm_unavailable",
            )

        content_text = self._region_prompt_analysis_prompt(capture_result, prompt=prompt, source=source)
        content: Any = content_text
        image_uri = self._image_data_uri((capture_result.get("capture") or {}).get("path"))
        if self.config.vision_enabled and image_uri:
            content = [
                {"type": "text", "text": content_text},
                {"type": "image_url", "image_url": {"url": image_uri}},
            ]
        agent_name = resolve_agent_name(runtime=self.runtime)
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are {agent_name} reviewing a user-selected desktop rectangle as direct "
                    "visual evidence for the current operator question. This is not a timed "
                    "body-double check-in. Extract only what the selected pixels and OCR support. "
                    "Return strict JSON with keys: visual_summary string, visible_text string, "
                    "relevant_details string, answer_context string, uncertainty string, confidence number."
                ),
            },
            {"role": "user", "content": content},
        ]
        try:
            response = await llm.chat_completion(
                messages=messages,
                model=self.config.vision_model,
                complexity="light",
                payload={"temperature": 0.1, "max_tokens": 900},
                source="desktop_region_prompt",
                session_id=self._session_id(),
            )
            parsed = self._parse_response_json(response)
            if isinstance(parsed, dict):
                return self._normalize_region_prompt_analysis(parsed, capture_result, prompt=prompt)
        except Exception as exc:
            if content is not content_text:
                try:
                    response = await llm.chat_completion(
                        messages=[
                            messages[0],
                            {
                                "role": "user",
                                "content": (
                                    content_text
                                    + f"\n\nVision input failed with {type(exc).__name__}; use OCR and metadata only."
                                ),
                            },
                        ],
                        model=self.config.vision_model,
                        complexity="light",
                        payload={"temperature": 0.1, "max_tokens": 700},
                        source="desktop_region_prompt_fallback",
                        session_id=self._session_id(),
                    )
                    parsed = self._parse_response_json(response)
                    if isinstance(parsed, dict):
                        return self._normalize_region_prompt_analysis(parsed, capture_result, prompt=prompt)
                except Exception:
                    pass
            return self._region_prompt_fallback_analysis(
                capture_result,
                prompt=prompt,
                fallback_reason=str(exc),
            )
        return self._region_prompt_fallback_analysis(
            capture_result,
            prompt=prompt,
            fallback_reason="unparseable_llm_response",
        )

    def _analysis_prompt(self, capture_result: dict[str, Any], *, reason: str) -> str:
        capture = capture_result.get("capture") or {}
        ocr_text = str(capture_result.get("ocr_text") or "").strip()
        media_lines = self._media_prompt_lines(capture_result.get("media_context"))
        media_change_lines = self._media_state_change_prompt_lines(capture_result.get("media_state_changes"))
        commentary_lines = self._media_commentary_mode_prompt_lines(capture_result.get("media_state_changes"))
        transcript_lines = self._youtube_transcript_prompt_lines(capture_result.get("youtube_transcript"))
        live_transcript_lines = self._live_transcript_prompt_lines(
            capture_result.get("live_transcript"),
            capture_result.get("youtube_transcript"),
        )
        declared_task = self._declared_task()
        task_lines = [
            f"Declared operator task: {declared_task}",
            "Task coaching is allowed only relative to this declared task.",
        ] if declared_task else [
            "Declared operator task: (none)",
            "No task has been declared. Do not give stay-on-task, focus, drift, productivity, or next-step coaching.",
            "A directly relevant video, webpage, transcript segment, system state, or visible work context is a concrete screen-relevant reason to speak.",
            "If there is no concrete screen-relevant reason to speak, return should_speak false.",
        ]
        return "\n".join(
            [
                "Review this desktop observation.",
                f"Reason: {reason}",
                f"Screenshot path: {capture.get('path') or ''}",
                f"Capture backend: {capture.get('backend') or ''}",
                "Operator goal: body-double support for productive ADHD-friendly work.",
                "Speak only if a timely, concise collaborator comment would help right now.",
                *task_lines,
                (
                    "Classify speech_intent as task_coaching only for declared-task focus support; "
                    "use screen_relevant, system_issue, or safety_privacy when the visible desktop, "
                    "current media, or current transcript segment itself gives a concrete reason to interrupt."
                ),
                *media_lines,
                *self._media_ambiguity_prompt_lines(capture_result),
                *media_change_lines,
                *commentary_lines,
                *transcript_lines,
                *live_transcript_lines,
                "OCR text excerpt:",
                ocr_text or "(no OCR text)",
            ]
        )

    def _region_prompt_analysis_prompt(
        self,
        capture_result: dict[str, Any],
        *,
        prompt: str,
        source: str,
    ) -> str:
        capture = capture_result.get("capture") or {}
        selection = capture_result.get("region_selection") or {}
        ocr_text = str(capture_result.get("ocr_text") or "").strip()
        return "\n".join(
            [
                "Analyze the selected desktop region for the operator's question.",
                f"Source: {source}",
                f"Operator question/instruction: {prompt}",
                f"Screenshot path: {capture.get('path') or ''}",
                f"Selection geometry: {json.dumps(selection, sort_keys=True)}",
                f"Image size: {capture.get('width') or '?'}x{capture.get('height') or '?'}",
                *self._media_prompt_lines(capture_result.get("media_context")),
                *self._media_ambiguity_prompt_lines(capture_result),
                *self._media_state_change_prompt_lines(capture_result.get("media_state_changes")),
                *self._youtube_transcript_prompt_lines(capture_result.get("youtube_transcript")),
                *self._live_transcript_prompt_lines(
                    capture_result.get("live_transcript"),
                    capture_result.get("youtube_transcript"),
                ),
                "OCR text excerpt from selected region:",
                ocr_text or "(no OCR text)",
                "Answer context should be concise evidence for a later full OpenCAS conversation turn.",
            ]
        )

    def _normalize_region_prompt_analysis(
        self,
        raw: dict[str, Any],
        capture_result: dict[str, Any],
        *,
        prompt: str,
    ) -> dict[str, Any]:
        ocr_text = str(capture_result.get("ocr_text") or "").strip()
        visual_summary = str(raw.get("visual_summary") or raw.get("activity_summary") or "").strip()
        visible_text = str(raw.get("visible_text") or raw.get("text") or ocr_text).strip()
        relevant_details = str(raw.get("relevant_details") or raw.get("details") or "").strip()
        answer_context = str(raw.get("answer_context") or raw.get("note") or relevant_details).strip()
        uncertainty = str(raw.get("uncertainty") or raw.get("limitations") or "").strip()
        confidence = self._coerce_float(raw.get("confidence"), default=0.68)
        summary = visual_summary or relevant_details or answer_context or (visible_text[:240] if visible_text else "")
        return {
            "should_speak": False,
            "activity_summary": summary,
            "reason": "selected_region_prompt",
            "spoken_text": "",
            "note": answer_context or summary,
            "speech_intent": "none",
            "speech_relevance_score": 0.0,
            "region_prompt": {
                "prompt": prompt,
                "visual_summary": visual_summary,
                "visible_text": visible_text,
                "relevant_details": relevant_details,
                "answer_context": answer_context,
                "uncertainty": uncertainty,
                "confidence": confidence,
            },
            "raw": raw,
        }

    def _region_prompt_fallback_analysis(
        self,
        capture_result: dict[str, Any],
        *,
        prompt: str,
        fallback_reason: str,
    ) -> dict[str, Any]:
        ocr_text = str(capture_result.get("ocr_text") or "").strip()
        summary = ocr_text.splitlines()[0][:240] if ocr_text else "Selected desktop region captured."
        return {
            "should_speak": False,
            "activity_summary": summary,
            "reason": f"region_prompt_fallback:{fallback_reason}",
            "spoken_text": "",
            "note": summary,
            "speech_intent": "none",
            "speech_relevance_score": 0.0,
            "fallback": True,
            "region_prompt": {
                "prompt": prompt,
                "visual_summary": summary,
                "visible_text": ocr_text,
                "relevant_details": "",
                "answer_context": summary,
                "uncertainty": f"Vision analysis unavailable: {fallback_reason}",
                "confidence": 0.35 if ocr_text else 0.2,
            },
        }

    def _media_prompt_lines(self, media_context: Any) -> list[str]:
        if not isinstance(media_context, list) or not media_context:
            return ["Playing media context: (none detected)"]
        lines = ["Playing media context:"]
        for item in self._ordered_media_context(media_context)[:3]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            artist = str(item.get("artist") or "").strip()
            url = str(item.get("url") or "").strip()
            player = str(item.get("player") or "").strip()
            status = str(item.get("status") or "").strip()
            position = str(item.get("position_label") or "").strip()
            progress = item.get("progress_percent")
            progress_label = f"{progress:.1f}% elapsed" if isinstance(progress, (int, float)) else ""
            timing_warning = str(item.get("media_timing_warning") or "").strip()
            warning_label = f"timing warning: {timing_warning[:260]}" if timing_warning else ""
            parts = [
                part
                for part in [title, artist, url, player, status, position, progress_label, warning_label]
                if part
            ]
            if parts:
                lines.append(f"- {' | '.join(parts)}")
        return lines

    def _media_ambiguity_prompt_lines(self, capture_result: dict[str, Any]) -> list[str]:
        ambiguity = self._media_ambiguity(capture_result)
        if ambiguity is None:
            return []
        labels = [
            str(item.get("label") or "").strip()
            for item in ambiguity.get("playing_media") or []
            if isinstance(item, dict) and str(item.get("label") or "").strip()
        ]
        return [
            "Media ambiguity: multiple media sources are playing.",
            f"- playing sources: {'; '.join(labels[:4]) or '(unknown)'}",
            "- Do not guess which video the operator wants commentary on. Say that the current media target is ambiguous.",
        ]

    def _media_ambiguity(self, capture_result: dict[str, Any]) -> Optional[dict[str, Any]]:
        media_context = capture_result.get("media_context") if isinstance(capture_result, dict) else None
        if not isinstance(media_context, list):
            return None
        playing = [
            item
            for item in self._ordered_media_context(media_context)
            if isinstance(item, dict)
            and self._media_item_is_playing(item)
            and str(item.get("title") or item.get("url") or "").strip()
        ]
        identities = {
            self._media_identity_for_item(item)
            for item in playing
            if self._media_identity_for_item(item)
        }
        if len(identities) <= 1:
            return None
        playing_media = []
        for item in playing[:5]:
            label = " | ".join(
                part
                for part in [
                    str(item.get("title") or "").strip(),
                    str(item.get("artist") or "").strip(),
                    str(item.get("url") or "").strip(),
                    str(item.get("player") or "").strip(),
                    str(item.get("position_label") or "").strip(),
                ]
                if part
            )
            playing_media.append(
                {
                    "identity": self._media_identity_for_item(item),
                    "label": label,
                    "player": str(item.get("player") or ""),
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                }
            )
        return {
            "reason": "multiple_playing_media",
            "playing_media": playing_media,
            "playing_count": len(identities),
        }

    def _media_ambiguity_analysis(self, ambiguity: dict[str, Any], *, reason: str) -> dict[str, Any]:
        labels = [
            str(item.get("label") or "").strip()
            for item in ambiguity.get("playing_media") or []
            if isinstance(item, dict) and str(item.get("label") or "").strip()
        ]
        joined = "; ".join(labels[:3])
        spoken = (
            "I see more than one media source playing, so I am not sure which video to comment on."
            if not joined
            else self._shorten_for_speech(
                f"I see more than one media source playing, so I am not sure which video to comment on: {joined}.",
                self.config.max_spoken_chars,
            )
        )
        return {
            "should_speak": True,
            "activity_summary": "Multiple playing media sources make the current video target ambiguous.",
            "reason": f"media_ambiguity:{reason}",
            "spoken_text": spoken,
            "note": "Commentary was withheld from specific video content because more than one media source was playing.",
            "speech_intent": "screen_relevant",
            "speech_relevance_score": 1.0,
            "speech_policy": "allowed_media_ambiguity_notice",
            "media_ambiguity": ambiguity,
        }

    def _ordered_media_context(self, media_context: Any) -> list[dict[str, Any]]:
        if not isinstance(media_context, list):
            return []
        items = [item for item in media_context if isinstance(item, dict)]

        def sort_key(indexed: tuple[int, dict[str, Any]]) -> tuple[int, int, int, int]:
            index, item = indexed
            status = str(item.get("status") or "").strip().lower()
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            status_score = 100 if status == "playing" else 40 if status in {"paused", "stopped"} else 60
            youtube_score = 30 if self._youtube_video_id(url) else 0
            content_score = 10 if url else 5 if title else 0
            return (status_score, youtube_score, content_score, -index)

        return [
            item
            for _index, item in sorted(
                enumerate(items),
                key=sort_key,
                reverse=True,
            )
        ]

    def _media_state_change_prompt_lines(self, changes: Any) -> list[str]:
        if not isinstance(changes, list) or not changes:
            return ["Media state changes since last observation: (none detected)"]
        lines = ["Media state changes since last observation:"]
        for change in changes[:8]:
            if not isinstance(change, dict):
                continue
            parts = [
                str(change.get("event") or "").strip(),
                str(change.get("title") or "").strip(),
                str(change.get("url") or "").strip(),
                f"{change.get('from_status') or '?'}->{change.get('to_status') or '?'}",
                " -> ".join(
                    part
                    for part in [
                        str(change.get("from_position_label") or "").strip(),
                        str(change.get("to_position_label") or "").strip(),
                    ]
                    if part
                ),
            ]
            if change.get("position_delta_seconds") is not None:
                parts.append(f"delta {change.get('position_delta_seconds')}s")
            lines.append(f"- {' | '.join(part for part in parts if part)}")
        return lines

    def _youtube_transcript_prompt_lines(self, transcript: Any) -> list[str]:
        if not isinstance(transcript, dict):
            return ["YouTube transcript context: (none detected)"]
        status = str(transcript.get("status") or "")
        if status != "available":
            return [f"YouTube transcript context: unavailable ({transcript.get('reason') or status})"]
        excerpt = str(transcript.get("transcript_excerpt") or "").strip()
        title = str(transcript.get("title") or "").strip()
        url = str(transcript.get("url") or "").strip()
        path = str(transcript.get("transcript_path") or "").strip()
        chars = transcript.get("transcript_chars")
        basis = str(transcript.get("transcript_excerpt_basis") or "").strip()
        position = str(transcript.get("transcript_position_label") or "").strip()
        progress = transcript.get("transcript_progress_percent")
        progress_line = f"{progress:.1f}% elapsed" if isinstance(progress, (int, float)) else ""
        duration_warning = str(transcript.get("media_duration_warning") or "").strip()
        target_seconds = transcript.get("transcript_target_seconds")
        target_line = (
            f"{self._format_seconds_label(float(target_seconds))} target"
            if isinstance(target_seconds, (int, float))
            else ""
        )
        cutoff_seconds = transcript.get("transcript_commentary_cutoff_seconds")
        cutoff_line = (
            f"{self._format_seconds_label(float(cutoff_seconds))} reached-commentary cutoff"
            if isinstance(cutoff_seconds, (int, float))
            else ""
        )
        aligned_start = str(transcript.get("transcript_aligned_start_label") or "").strip()
        aligned_end = str(transcript.get("transcript_aligned_end_label") or "").strip()
        aligned_line = ""
        if aligned_start or aligned_end:
            aligned_line = f"{aligned_start or '?'}-{aligned_end or '?'} caption cue window"
        segment_count = transcript.get("transcript_timed_segment_count")
        segment_line = (
            f"{segment_count} timestamped cues available"
            if isinstance(segment_count, int) and segment_count > 0
            else ""
        )
        safe_excerpt = self._transcript_excerpt_is_timing_safe(transcript)
        excerpt_header = (
            "- current-position transcript excerpt:"
            if safe_excerpt
            else "- current-position transcript excerpt withheld:"
        )
        excerpt_body = (
            excerpt[:4000]
            if safe_excerpt
            else self._transcript_excerpt_withheld_reason(transcript)
        )
        usage_rule = (
            "Treat this excerpt as the timestamp-confirmed reached part of the video. "
            "Do not interrupt about unreached later content unless playback reaches that point."
            if safe_excerpt
            else (
                "Do not use the cached full transcript for spoken commentary yet. Comment only from "
                "visible screen evidence, media title, or timestamp-confirmed captions."
            )
        )
        lines = [
            "YouTube transcript context:",
            f"- title: {title or '(unknown)'}",
            f"- url: {url or '(unknown)'}",
            f"- full transcript path: {path or '(unavailable)'}",
            f"- full transcript characters: {chars if chars is not None else '(unknown)'}",
            f"- current media position: {' | '.join(part for part in [position, progress_line] if part) or '(unknown)'}",
            *([f"- media timing warning: {duration_warning}"] if duration_warning else []),
            f"- transcript excerpt basis: {basis or '(unknown)'}",
            f"- timestamp alignment: {' | '.join(part for part in [target_line, cutoff_line, aligned_line, segment_line] if part) or '(unavailable)'}",
            excerpt_header,
            excerpt_body,
            usage_rule,
        ]
        return lines

    def _live_transcript_prompt_lines(self, live_transcript: Any, youtube_transcript: Any = None) -> list[str]:
        if not isinstance(live_transcript, dict):
            return []
        status = str(live_transcript.get("status") or "").strip()
        if status != "available":
            reason = str(live_transcript.get("reason") or status or "unknown").strip()
            if reason in {"media_not_playing", "media_ambiguity"}:
                return [f"Live transcript context: skipped ({reason})"]
            return [f"Live transcript context: unavailable ({reason})"]
        excerpt = str(live_transcript.get("transcript_excerpt") or live_transcript.get("transcript_text") or "").strip()
        if not excerpt:
            return ["Live transcript context: unavailable (empty_transcript)"]
        source = str(live_transcript.get("source") or "whisper").strip() or "whisper"
        mode = str(live_transcript.get("mode") or "local").strip() or "local"
        model = str(live_transcript.get("model") or "").strip()
        header = (
            "Live transcript context (local Whisper):"
            if source == "whisper" and mode == "local"
            else "Live transcript context:"
        )
        media_parts = [
            str(live_transcript.get("media_title") or "").strip(),
            str(live_transcript.get("media_artist") or "").strip(),
            str(live_transcript.get("media_url") or "").strip(),
            str(live_transcript.get("media_status") or "").strip(),
            str(live_transcript.get("media_position_label") or "").strip(),
        ]
        seconds = self._coerce_float(live_transcript.get("capture_seconds"), default=0.0)
        source_parts = [
            f"source {source}",
            f"mode {mode}",
            f"model {model}" if model else "",
            f"{seconds:g}s captured" if seconds > 0 else "",
        ]
        prefetched_status = self._prefetched_transcript_status(youtube_transcript)
        if prefetched_status == "none":
            prefetched_status = str(live_transcript.get("prefetched_transcript_status") or "none")
        if prefetched_status == "available":
            fusion_rule = (
                "- Fusion rule: use the retrieved transcript as the timestamped map and durable context; "
                "Trust the live transcript for what is being heard now if it conflicts with cached transcript timing."
            )
        else:
            fusion_rule = (
                "- Live-stream rule: no usable retrieved transcript is available; this live Whisper excerpt is "
                "the current heard segment for following along."
            )
        return [
            header,
            f"- capture: {' | '.join(part for part in source_parts if part) or '(unknown)'}",
            f"- media target: {' | '.join(part for part in media_parts if part) or '(unknown)'}",
            f"- prefetched transcript status: {prefetched_status or 'none'}",
            "- current audio transcript excerpt:",
            excerpt[: self.config.live_transcription_max_chars],
            fusion_rule,
        ]

    def _transcript_excerpt_is_timing_safe(self, transcript: dict[str, Any]) -> bool:
        if not isinstance(transcript, dict) or transcript.get("status") != "available":
            return False
        if not str(transcript.get("transcript_excerpt") or "").strip():
            return False
        basis = str(transcript.get("transcript_excerpt_basis") or "").strip()
        return basis == "caption_timestamp_reached" and bool(transcript.get("transcript_position_valid", True))

    def _transcript_excerpt_withheld_reason(self, transcript: dict[str, Any]) -> str:
        basis = str(transcript.get("transcript_excerpt_basis") or "").strip() or "unknown"
        if basis == "caption_timestamp_waiting":
            return "No completed timestamped caption segment is far enough behind the current playback position yet."
        if basis == "playback_position":
            return "Only approximate transcript-position alignment is available; withholding it from speech to avoid future or stale commentary."
        if basis in {"position_out_of_range", "position_inconsistent_with_transcript_duration"}:
            warning = str(transcript.get("transcript_position_warning") or "").strip()
            return warning or "The media backend position is inconsistent with the timestamped transcript."
        if basis == "start":
            return "Playback position is unavailable, so the transcript cannot be timed to what has actually been watched."
        if basis == "empty":
            return "Transcript excerpt is empty."
        return f"Transcript timing basis is {basis}; timestamp-confirmed reached commentary is not available."

    def _conversation_prompt_note(self, result: dict[str, Any], *, user_input: str) -> str:
        analysis = result.get("analysis") if isinstance(result.get("analysis"), dict) else {}
        capture = result.get("capture") if isinstance(result.get("capture"), dict) else {}
        transcript = result.get("youtube_transcript") or (
            result.get("capture_result", {}).get("youtube_transcript")
            if isinstance(result.get("capture_result"), dict)
            else None
        )
        live_transcript = result.get("live_transcript") or (
            result.get("capture_result", {}).get("live_transcript")
            if isinstance(result.get("capture_result"), dict)
            else None
        )
        media_context = result.get("media_context") or (
            result.get("capture_result", {}).get("media_context")
            if isinstance(result.get("capture_result"), dict)
            else None
        )
        media_state_changes = result.get("media_state_changes") or (
            result.get("capture_result", {}).get("media_state_changes")
            if isinstance(result.get("capture_result"), dict)
            else None
        )
        lines = [
            "Body-double environment is active for this user turn.",
            "Treat the user message as conversational input from someone in the room whose desktop you can currently observe.",
            "Respond like a person in the room: address what the user asked directly, out loud, and conversationally.",
            "Use ordinary spoken English. Do not format the answer as markdown, bullets, headings, changelogs, reports, task logs, file manifests, or timestamped status notes when speaking to the user.",
            "Do not default to writing a report, naming a saved file, reciting timestamps, or narrating artifact metadata unless the user explicitly asks for an artifact.",
            f"- user turn excerpt: {self._shorten_for_speech(user_input, 220)}",
            f"- observed activity: {analysis.get('activity_summary') or '(not summarized)'}",
            f"- screenshot evidence: {capture.get('path') or '(unavailable)'}",
        ]
        if isinstance(media_context, list) and media_context:
            item = media_context[0]
            lines.append(
                "- playing media: "
                + " | ".join(
                    part
                    for part in [
                        str(item.get("title") or ""),
                        str(item.get("artist") or ""),
                        str(item.get("url") or ""),
                        str(item.get("status") or ""),
                        str(item.get("position_label") or ""),
                    ]
                    if part
                )
            )
        if self.config.media_commentary_mode_enabled:
            lines.append(
                "- Media commentary mode: active; the operator asked OpenCAS to keep reacting to current "
                "and autoplayed media as shared room context without another prompt."
            )
        if isinstance(media_state_changes, list) and media_state_changes:
            lines.append("- media state changes:")
            for change in media_state_changes[:4]:
                if isinstance(change, dict):
                    lines.append(
                        f"  - {change.get('event')}: {change.get('title') or change.get('url') or ''} "
                        f"{change.get('from_status') or '?'}->{change.get('to_status') or '?'} "
                        f"{change.get('from_position_label') or ''}->{change.get('to_position_label') or ''}".strip()
                    )
        if isinstance(transcript, dict) and transcript.get("status") == "available":
            lines.append(f"- YouTube transcript evidence: {transcript.get('transcript_path') or ''}")
            excerpt = str(transcript.get("transcript_excerpt") or "").strip()
            if excerpt:
                lines.append("- transcript excerpt:")
                lines.append(excerpt[:2200])
        if isinstance(live_transcript, dict) and live_transcript.get("status") == "available":
            excerpt = str(live_transcript.get("transcript_excerpt") or "").strip()
            if excerpt:
                lines.append("- live Whisper transcript evidence:")
                lines.append(excerpt[:2200])
                if isinstance(transcript, dict) and transcript.get("status") == "available":
                    lines.append(
                        "- transcript fusion: use the retrieved transcript for the timestamped map and "
                        "the live Whisper transcript for the current heard moment."
                    )
        followup = analysis.get("self_interest_followup") if isinstance(analysis.get("self_interest_followup"), dict) else {}
        if followup:
            lines.append(f"- observed-context relevance match: {followup.get('connection_summary') or '(not summarized)'}")
            if followup.get("self_directed_next_step"):
                lines.append(f"- next relevance step: {followup.get('self_directed_next_step')}")
        lines.append(
            "Use this as live room context for the current reply. Do not pretend certainty beyond the observation evidence."
        )
        return "\n".join(lines)

    def _region_conversation_prompt_note(self, result: dict[str, Any], *, prompt: str) -> str:
        analysis = result.get("analysis") if isinstance(result.get("analysis"), dict) else {}
        region = analysis.get("region_prompt") if isinstance(analysis.get("region_prompt"), dict) else {}
        capture = result.get("capture") if isinstance(result.get("capture"), dict) else {}
        selection = result.get("region_selection") if isinstance(result.get("region_selection"), dict) else {}
        capture_result = result.get("capture_result") if isinstance(result.get("capture_result"), dict) else {}
        ocr_text = str(capture_result.get("ocr_text") or "").strip()
        lines = [
            "Selected desktop region context is active for this user turn.",
            "Treat the selected rectangle as direct visual evidence for the current question.",
            f"- user question/instruction: {self._shorten_for_speech(prompt, 500)}",
            f"- screenshot evidence: {capture.get('path') or '(unavailable)'}",
            f"- selection geometry: {json.dumps(selection, sort_keys=True)}",
            f"- visual summary: {region.get('visual_summary') or analysis.get('activity_summary') or '(not summarized)'}",
        ]
        if region.get("visible_text"):
            lines.append("- visible/OCR text:")
            lines.append(str(region.get("visible_text"))[:1800])
        elif ocr_text:
            lines.append("- OCR text:")
            lines.append(ocr_text[:1800])
        if region.get("relevant_details"):
            lines.append(f"- relevant details: {region.get('relevant_details')}")
        if region.get("answer_context"):
            lines.append(f"- answer context from visual analysis: {region.get('answer_context')}")
        if region.get("uncertainty"):
            lines.append(f"- uncertainty: {region.get('uncertainty')}")
        if region.get("confidence") is not None:
            lines.append(f"- confidence: {region.get('confidence')}")
        lines.append(
            "Use this selected-region evidence when answering. If the evidence is insufficient, say what is missing."
        )
        return "\n".join(lines)

    def _fallback_analysis(
        self,
        capture_result: dict[str, Any],
        *,
        reason: str,
        fallback_reason: str,
    ) -> dict[str, Any]:
        ocr_text = str(capture_result.get("ocr_text") or "").strip()
        summary = ocr_text.splitlines()[0][:240] if ocr_text else "Desktop screenshot captured."
        return {
            "should_speak": False,
            "activity_summary": summary,
            "reason": f"fallback:{fallback_reason}",
            "spoken_text": "",
            "note": f"Observation created from screenshot; reason={reason}.",
            "speech_intent": "none",
            "speech_relevance_score": 0.0,
            "fallback": True,
        }

    def _normalize_analysis(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "should_speak": bool(raw.get("should_speak")),
            "activity_summary": str(raw.get("activity_summary") or raw.get("summary") or "").strip(),
            "reason": str(raw.get("reason") or "").strip(),
            "spoken_text": str(raw.get("spoken_text") or raw.get("message") or "").strip(),
            "note": str(raw.get("note") or "").strip(),
            "speech_intent": self._normalize_speech_intent(raw.get("speech_intent") or raw.get("intent")),
            "speech_relevance_score": self._coerce_float(
                raw.get("speech_relevance_score") or raw.get("relevance_score"),
                default=0.0,
            ),
            "raw": raw,
        }

    def _has_media_interest_material(self, capture_result: dict[str, Any]) -> bool:
        transcript = capture_result.get("youtube_transcript")
        if isinstance(transcript, dict) and transcript.get("status") == "available":
            return True
        live_transcript = capture_result.get("live_transcript")
        if isinstance(live_transcript, dict) and live_transcript.get("status") == "available":
            return True
        media_context = capture_result.get("media_context")
        if isinstance(media_context, list):
            for item in media_context:
                if not isinstance(item, dict):
                    continue
                if str(item.get("title") or item.get("url") or "").strip():
                    return True
        if len(str(capture_result.get("ocr_text") or "").strip()) >= 80:
            return True
        return False

    def _has_playing_video_context(self, capture_result: dict[str, Any]) -> bool:
        transcript = capture_result.get("youtube_transcript")
        if isinstance(transcript, dict) and transcript.get("status") == "available":
            return True
        live_transcript = capture_result.get("live_transcript")
        if isinstance(live_transcript, dict) and live_transcript.get("status") == "available":
            return True
        media_context = capture_result.get("media_context")
        if not isinstance(media_context, list):
            return False
        for item in media_context:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").lower()
            title = str(item.get("title") or "").lower()
            if "youtube.com" in url or "youtu.be" in url or "video" in title:
                return True
        return False

    async def _observed_context_relevance_evidence(self, *, limit: int = 12) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        items.extend(self._project_context_relevance_evidence(limit=min(3, limit)))
        for item in self._self_interest_evidence(limit=max(0, min(4, limit - len(items)))):
            enriched = dict(item)
            enriched.setdefault("category", "self_interest")
            items.append(enriched)
        items.extend(self._active_work_relevance_evidence(limit=max(0, min(4, limit - len(items)))))
        if len(items) < limit:
            items.extend(await self._proposal_relevance_evidence(limit=max(0, limit - len(items))))
        deduped: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in items:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            key = f"{item.get('category') or ''}:{text.lower()}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def _self_interest_evidence(self, *, limit: int = 8) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        identity = getattr(getattr(self.runtime, "ctx", None), "identity", None) or getattr(self.runtime, "identity", None)
        self_model = getattr(identity, "self_model", None)
        beliefs = getattr(self_model, "self_beliefs", {}) or {}
        daydream = beliefs.get("daydream") if isinstance(beliefs, dict) else None
        if isinstance(daydream, dict):
            config = daydream.get("bulma_config") if isinstance(daydream.get("bulma_config"), dict) else {}
            status = daydream.get("bulma_status") if isinstance(daydream.get("bulma_status"), dict) else {}
            current_interest = status.get("currentInterest") or status.get("current_interest")
            if current_interest:
                        items.append(
                            {
                                "label": "Current self-directed interest",
                                "text": self._shorten_for_speech(current_interest, 360),
                                "source": "identity.self_beliefs.daydream.status",
                                "category": "self_interest",
                            }
                        )
            raw_seeds = config.get("hobbySeeds") or config.get("hobby_seeds") or []
            if isinstance(raw_seeds, list):
                for seed in raw_seeds:
                    if len(items) >= limit:
                        break
                    text = self._shorten_for_speech(seed, 360)
                    if text:
                        items.append(
                            {
                                "label": "Hobby or curiosity seed",
                                "text": text,
                                "source": "identity.self_beliefs.daydream.config",
                                "category": "self_interest",
                            }
                        )
        graph = getattr(self.runtime, "fascination_graph", None)
        active = getattr(graph, "active", None)
        if callable(active) and len(items) < limit:
            try:
                nodes = active(limit=limit)
            except Exception:
                nodes = []
            for node in nodes:
                if len(items) >= limit:
                    break
                summary = str(getattr(node, "summary", "") or "").strip()
                if summary:
                    items.append(
                        {
                            "label": "Active fascination trail",
                            "text": self._shorten_for_speech(summary, 360),
                            "source": "fascination_graph",
                            "category": "self_interest",
                        }
                    )
        deduped: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in items:
            key = str(item.get("text") or "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def _active_work_relevance_evidence(self, *, limit: int) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        if limit <= 0:
            return items
        declared_task = self._declared_task()
        if declared_task:
            items.append(
                {
                    "label": "Declared body-double task",
                    "text": self._shorten_for_speech(declared_task, 420),
                    "source": "desktop_context.declared_task",
                    "category": "shared_work",
                }
            )
        runtime_activity = str(getattr(self.runtime, "_activity", "") or "").strip()
        if runtime_activity and runtime_activity != "idle":
            items.append(
                {
                    "label": "Current runtime activity",
                    "text": self._shorten_for_speech(runtime_activity, 260),
                    "source": "runtime.activity",
                    "category": "shared_work",
                }
            )
        executive = getattr(self.runtime, "executive", None)
        intention = str(getattr(executive, "intention", "") or "").strip()
        if intention:
            items.append(
                {
                    "label": "Executive intention",
                    "text": self._shorten_for_speech(intention, 420),
                    "source": "executive.intention",
                    "category": "shared_work",
                }
            )
        for goal in list(getattr(executive, "active_goals", []) or [])[:4]:
            if len(items) >= limit:
                break
            text = self._shorten_for_speech(goal, 420)
            if text:
                items.append(
                    {
                        "label": "Active shared goal",
                        "text": text,
                        "source": "executive.active_goals",
                        "category": "shared_work",
                    }
                )
        for owner_name, owner in [("runtime", self.runtime), ("context", getattr(self.runtime, "ctx", None))]:
            if len(items) >= limit:
                break
            for attr in ("current_task", "active_task", "active_goal", "active_project", "current_project"):
                value = str(getattr(owner, attr, "") or "").strip()
                if not value:
                    continue
                items.append(
                    {
                        "label": attr.replace("_", " ").title(),
                        "text": self._shorten_for_speech(value, 420),
                        "source": f"{owner_name}.{attr}",
                        "category": "shared_work",
                    }
                )
                if len(items) >= limit:
                    break
        return items[:limit]

    def _project_context_relevance_evidence(self, *, limit: int) -> list[dict[str, str]]:
        if limit <= 0 or not self.config.project_context_relevance_enabled:
            return []
        if self._project_relevance_cache is not None:
            return self._project_relevance_cache[:limit]
        root = Path(__file__).resolve().parents[2]
        candidates = [
            ("AGENTS.md", "OpenCAS project contract"),
            ("OPENCAS_PRODUCT_SPEC.md", "OpenCAS product spec"),
            ("TaskList.md", "OpenCAS active task list"),
        ]
        items: list[dict[str, str]] = []
        for relative, label in candidates:
            path = root / relative
            if not path.exists() or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            snippet = self._project_context_snippet(text)
            if not snippet:
                continue
            items.append(
                {
                    "label": label,
                    "text": snippet,
                    "source": f"project_context.{relative}",
                    "category": "project_context",
                }
            )
        self._project_relevance_cache = items
        return items[:limit]

    def _project_context_snippet(self, text: str) -> str:
        lines: list[str] = []
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            if (
                "opencas" in lower
                or "autonomous" in lower
                or "persistent" in lower
                or "memory" in lower
                or "daydream" in lower
                or "body-double" in lower
                or "desktop context" in lower
                or "current active stance" in lower
                or line.startswith("- `TASK-")
            ):
                lines.append(line)
            if len(lines) >= 12:
                break
        if not lines:
            lines = [line.strip() for line in str(text or "").splitlines() if line.strip()][:8]
        return self._shorten_for_speech(" ".join(lines), 1200)

    async def _proposal_relevance_evidence(self, *, limit: int) -> list[dict[str, str]]:
        if limit <= 0:
            return []
        store = getattr(self.runtime, "context_proposals", None) or getattr(
            getattr(self.runtime, "ctx", None),
            "context_proposal_store",
            None,
        )
        list_recent = getattr(store, "list_recent", None)
        if not callable(list_recent):
            return []
        try:
            proposals = await self._call_maybe_async(list_recent, limit=limit)
        except TypeError:
            try:
                proposals = await self._call_maybe_async(list_recent, limit)
            except Exception:
                return []
        except Exception:
            return []
        items: list[dict[str, str]] = []
        for proposal in list(proposals or [])[:limit]:
            content = str(getattr(proposal, "content", "") or "").strip()
            if not content:
                continue
            kind = str(getattr(proposal, "proposal_kind", "") or "context_proposal")
            items.append(
                {
                    "label": f"Recent reflective proposal: {kind}",
                    "text": self._shorten_for_speech(content, 520),
                    "source": f"context_proposal.{kind}",
                    "category": "recent_context",
                }
            )
        return items

    def _media_interest_evidence_refs(self, capture_result: dict[str, Any]) -> list[str]:
        refs: list[str] = []
        capture = capture_result.get("capture") or {}
        if capture.get("path"):
            refs.append(f"screenshot:{capture.get('path')}")
        transcript = capture_result.get("youtube_transcript")
        if isinstance(transcript, dict):
            if transcript.get("transcript_path"):
                refs.append(f"youtube_transcript:{transcript.get('transcript_path')}")
            if transcript.get("url"):
                refs.append(f"media_url:{transcript.get('url')}")
        live_transcript = capture_result.get("live_transcript")
        if isinstance(live_transcript, dict):
            if live_transcript.get("transcript_hash"):
                refs.append(f"live_whisper_transcript:{live_transcript.get('transcript_hash')}")
            if live_transcript.get("media_url"):
                refs.append(f"media_url:{live_transcript.get('media_url')}")
        media_context = capture_result.get("media_context")
        if isinstance(media_context, list):
            for item in media_context[:2]:
                if isinstance(item, dict) and item.get("url"):
                    refs.append(f"media_url:{item.get('url')}")
        return list(dict.fromkeys(refs))

    async def _persist_self_interest_followup(
        self,
        followup: dict[str, Any],
        capture_result: dict[str, Any],
        *,
        session_id: Optional[str],
    ) -> None:
        await self._persist_self_interest_context_proposal(followup, capture_result, session_id=session_id)
        await self._record_self_interest_cognitive_event(followup, session_id=session_id)
        self._record_self_interest_fascination(followup)
        trace = getattr(self.runtime, "_trace", None)
        if callable(trace):
            trace(
                "desktop_context_self_interest_followup",
                {
                    "matches_self_interest": followup.get("matches_self_interest"),
                    "matched_interests": followup.get("matched_interests") or [],
                    "confidence": followup.get("confidence"),
                    "salience": followup.get("salience"),
                    "should_speak": followup.get("should_speak"),
                    "proposal_id": followup.get("context_proposal_id"),
                    "fascination_key": followup.get("fascination_key"),
                },
            )

    async def _persist_self_interest_context_proposal(
        self,
        followup: dict[str, Any],
        capture_result: dict[str, Any],
        *,
        session_id: Optional[str],
    ) -> None:
        store = getattr(self.runtime, "context_proposals", None) or getattr(
            getattr(self.runtime, "ctx", None),
            "context_proposal_store",
            None,
        )
        save = getattr(store, "save", None)
        if not callable(save):
            return
        snapshot_id = "truth:unknown"
        epoch = 0
        arbiter = getattr(self.runtime, "truth_arbiter", None) or getattr(
            getattr(self.runtime, "ctx", None),
            "truth_arbiter",
            None,
        )
        issue_snapshot = getattr(arbiter, "issue_snapshot", None)
        if callable(issue_snapshot):
            try:
                snapshot = issue_snapshot(reason="desktop_context_self_interest_followup")
                if inspect.isawaitable(snapshot):
                    snapshot = await snapshot
                snapshot_id = str(getattr(snapshot, "snapshot_id", snapshot_id) or snapshot_id)
                epoch = int(getattr(snapshot, "epoch", epoch) or epoch)
            except Exception:
                pass
        media_title = self._primary_media_title(capture_result)
        match_category = str(followup.get("match_category") or "observed_context")
        matched_targets = list(followup.get("matched_targets") or followup.get("matched_interests") or [])
        self_directed = bool(followup.get("matches_self_interest")) and not bool(followup.get("matches_shared_work"))
        content = "\n".join(
            part
            for part in [
                "Observed context relevance follow-up.",
                f"Media: {media_title}" if media_title else "",
                f"Match category: {match_category}",
                f"Matched targets: {', '.join(matched_targets)}",
                f"Connection: {followup.get('connection_summary') or ''}",
                f"Agent viewpoint: {followup.get('agent_viewpoint') or ''}",
                f"Why it matters: {followup.get('why_it_matters') or ''}",
                self._format_followup_list("Implications", followup.get("implications")),
                self._format_followup_list("Open questions", followup.get("open_questions")),
                f"Next step: {followup.get('self_directed_next_step') or ''}",
            ]
            if part
        )
        if not content.strip():
            return
        proposal = ContextProposal(
            source_lane=ContextLane.REFLECTIVE,
            source_snapshot_id=snapshot_id,
            source_epoch=epoch,
            proposal_kind="self_directed_media_interest_followup"
            if self_directed
            else "observed_context_relevance_followup",
            content=content,
            evidence_refs=list(followup.get("evidence_refs") or []),
            confidence=self._coerce_float(followup.get("confidence"), default=0.6),
            validation={
                "source": "desktop_context",
                "session_id": session_id or self._session_id(),
                "authority_note": "Reflective proposal only; it does not execute until downstream policy promotes it.",
                "self_directed": self_directed,
                "work_relevant": bool(followup.get("matches_shared_work")) or match_category in {"shared_work", "active_work", "recent_context"},
                "match_category": match_category,
                "media_context": capture_result.get("media_context") or [],
                "media_state_changes": capture_result.get("media_state_changes") or [],
                "live_transcript": capture_result.get("live_transcript"),
                "matched_targets": matched_targets,
                "matched_interests": followup.get("matched_interests") or [],
                "agent_viewpoint": followup.get("agent_viewpoint"),
                "implications": followup.get("implications") or [],
                "open_questions": followup.get("open_questions") or [],
                "artifact_title": followup.get("artifact_title"),
                "artifact_kind": followup.get("artifact_kind"),
                "salience": followup.get("salience"),
                "novelty": followup.get("novelty"),
                "timecode_or_position": followup.get("timecode_or_position"),
            },
        )
        try:
            await save(proposal)
            followup["context_proposal_id"] = proposal.proposal_id
        except Exception:
            return

    def _format_followup_list(self, label: str, value: Any) -> str:
        items = self._normalize_text_list(value)
        if not items:
            return ""
        return f"{label}: {'; '.join(items)}"

    async def _record_self_interest_cognitive_event(
        self,
        followup: dict[str, Any],
        *,
        session_id: Optional[str],
    ) -> None:
        store = getattr(self.runtime, "cognitive_state_store", None) or getattr(
            getattr(self.runtime, "ctx", None),
            "cognitive_state_store",
            None,
        )
        record_event = getattr(store, "record_event", None)
        if not callable(record_event):
            return
        from opencas.cognition import CognitiveEventKind

        content_parts = [
            str(followup.get("agent_viewpoint") or "").strip(),
            str(followup.get("why_it_matters") or "").strip(),
            self._format_followup_list("Implications", followup.get("implications")),
            self._format_followup_list("Open questions", followup.get("open_questions")),
            str(followup.get("self_directed_next_step") or "").strip(),
        ]
        event_content = "\n".join(part for part in content_parts if part)
        summary = self._shorten_for_speech(
            followup.get("agent_viewpoint")
            or followup.get("connection_summary")
            or "Desktop context overlapped with relevant self/work evidence.",
            240,
        )
        try:
            event = await record_event(
                CognitiveEventKind.CURIOSITY,
                summary,
                content=event_content,
                source="desktop_context.observed_context_relevance",
                session_id=session_id or self._session_id(),
                confidence=self._coerce_float(followup.get("confidence"), default=0.6),
                salience=1.0 + self._coerce_float(followup.get("salience"), default=0.5),
                evidence_refs=list(followup.get("evidence_refs") or []),
                payload={key: value for key, value in followup.items() if key != "raw"},
            )
            followup["cognitive_event_id"] = str(getattr(event, "event_id", "") or "")
        except Exception:
            return

    def _record_self_interest_fascination(self, followup: dict[str, Any]) -> None:
        graph = getattr(self.runtime, "fascination_graph", None)
        observe = getattr(graph, "observe_thought", None)
        if not callable(observe):
            return
        summary = str(
            followup.get("agent_viewpoint")
            or followup.get("self_directed_next_step")
            or followup.get("connection_summary")
            or followup.get("why_it_matters")
            or ""
        ).strip()
        if not summary:
            return
        confidence = self._coerce_float(followup.get("confidence"), default=0.6)
        evidence_refs = list(followup.get("evidence_refs") or [])
        thought = DaydreamThought(
            kind=DaydreamThoughtKind.QUESTION,
            route=DaydreamThoughtRoute.RESEARCH,
            summary=summary,
            question=str(followup.get("self_directed_next_step") or ""),
            hypothesis=str(followup.get("agent_viewpoint") or followup.get("connection_summary") or ""),
            possible_experiment=str(followup.get("self_directed_next_step") or ""),
            usefulness=self._coerce_float(followup.get("salience"), default=0.65),
            novelty=self._coerce_float(followup.get("novelty"), default=0.55),
            confidence=confidence,
            grounding=[
                CognitionGrounding(
                    kind=GroundingKind.OBSERVED,
            source=GroundingSource.RUNTIME,
            subject="desktop_observed_context_relevance",
                    claim=str(followup.get("connection_summary") or summary),
                    confidence=confidence,
                    evidence_ids=evidence_refs,
                    allowed_surface="internal",
                    meta={"source": "desktop_context"},
                )
            ],
        )
        try:
            node = observe(thought, source="desktop_context.observed_context_relevance")
            followup["fascination_key"] = str(getattr(node, "key", "") or "")
        except Exception:
            return

    def _primary_media_title(self, capture_result: dict[str, Any]) -> str:
        transcript = capture_result.get("youtube_transcript")
        if isinstance(transcript, dict):
            title = str(transcript.get("title") or "").strip()
            artist = str(transcript.get("artist") or "").strip()
            if title and artist:
                return f"{title} by {artist}"
            if title:
                return title
            url = str(transcript.get("url") or "").strip()
            if url:
                return url
        live_transcript = capture_result.get("live_transcript")
        if isinstance(live_transcript, dict) and live_transcript.get("status") == "available":
            title = str(live_transcript.get("media_title") or "").strip()
            artist = str(live_transcript.get("media_artist") or "").strip()
            if title and artist:
                return f"{title} by {artist}"
            if title:
                return title
            url = str(live_transcript.get("media_url") or "").strip()
            if url:
                return url
        media_context = capture_result.get("media_context")
        for item in self._ordered_media_context(media_context):
            title = str(item.get("title") or "").strip()
            artist = str(item.get("artist") or "").strip()
            if title and artist:
                return f"{title} by {artist}"
            if title:
                return title
        return ""

    def _current_media_position_label(self, capture_result: dict[str, Any]) -> str:
        transcript = capture_result.get("youtube_transcript")
        if isinstance(transcript, dict):
            label = str(transcript.get("transcript_position_label") or "").strip()
            if label:
                return label
        live_transcript = capture_result.get("live_transcript")
        if isinstance(live_transcript, dict) and live_transcript.get("status") == "available":
            label = str(live_transcript.get("media_position_label") or "").strip()
            if label:
                return label
        media_context = capture_result.get("media_context")
        for item in self._ordered_media_context(media_context):
            label = str(item.get("position_label") or "").strip()
            if label:
                return label
        return ""

    def _looks_like_media_commentary_request(self, text: str) -> bool:
        lowered = str(text or "").lower()
        if not lowered.strip():
            return False
        media_terms = (
            "video",
            "youtube",
            "media",
            "watching",
            "playback",
            "transcript",
        )
        commentary_terms = (
            "commentary",
            "comment on",
            "comment about",
            "react to",
            "reaction",
            "key point",
            "key points",
            "as i'm watching",
            "as i am watching",
            "while i'm watching",
            "while i am watching",
        )
        return any(term in lowered for term in media_terms) and any(
            term in lowered for term in commentary_terms
        )

    def _activate_media_commentary_mode(self, *, request: str, source: str) -> None:
        updates = self.config.model_dump()
        updates.update(
            {
                "media_commentary_mode_enabled": True,
                "media_commentary_requested_at": self._now().isoformat(),
                "media_commentary_source": source,
                "media_commentary_request": str(request or "").strip()[:1000],
                "media_commentary_request_source": source,
                "media_commentary_request_text": str(request or "").strip()[:1000],
                "live_transcription_enabled": True,
            }
        )
        self.config = DesktopContextConfig(**updates)
        self._save_config()
        self._event(
            "media_commentary_mode_enabled",
            {
                "source": source,
                "request": str(request or "").strip()[:240],
            },
        )

    def _media_commentary_observation_reason(self, changes: Any) -> Optional[str]:
        if not self.config.media_commentary_mode_enabled:
            return None
        if not isinstance(changes, list):
            return None
        for change in changes:
            if not isinstance(change, dict):
                continue
            event = str(change.get("event") or "")
            to_status = str(change.get("to_status") or "").strip().lower()
            if event in {"media_started", "media_resumed", "media_seeked"} and to_status == "playing":
                return f"media_commentary_mode:{event}"
        return None

    def _media_commentary_mode_prompt_lines(self, changes: Any = None) -> list[str]:
        if not self.config.media_commentary_mode_enabled:
            return []
        request = str(
            self.config.media_commentary_request
            or self.config.media_commentary_request_text
            or ""
        ).strip()
        source = str(
            self.config.media_commentary_source
            or self.config.media_commentary_request_source
            or ""
        ).strip()
        change_reason = self._media_commentary_observation_reason(changes)
        lines = [
            "Media commentary mode: active.",
            "- Operator intent: keep reacting to current and autoplayed media as shared room context without another prompt.",
            "- New media, resumes, seeks, and the current transcript segment are concrete reasons to consider speaking even when the subject is not OpenCAS.",
            "- Do not summarize what the operator can already see or hear; give a viewpoint, critique, implication, connection, or question.",
            "- If this observation was triggered by a new video and transcript context is available, speak with a concise first reaction.",
        ]
        if request:
            lines.append(f"- Original media-commentary request: {request[:500]}")
        if source:
            lines.append(f"- Request source: {source}")
        if change_reason:
            lines.append(f"- Current commentary trigger: {change_reason}")
        return lines

    def _record_media_state_changes(self, media_context: list[dict[str, Any]]) -> list[dict[str, Any]]:
        current_items = self._media_snapshot_items(media_context)
        previous = self._load_media_state()
        previous_items = previous.get("items") if isinstance(previous.get("items"), dict) else {}
        previous_observed_at = str(previous.get("observed_at") or "")
        observed_at = self._now().isoformat()
        changes = self._detect_media_state_changes(
            previous_items,
            current_items,
            observed_at=observed_at,
            previous_observed_at=previous_observed_at,
        )
        self._save_media_state({"observed_at": observed_at, "items": current_items})
        if changes:
            event = self._event("media_state_changed", {"changes": changes})
            tracer = getattr(self.runtime, "_trace", None)
            if callable(tracer):
                try:
                    tracer("desktop_context_media_state_changed", {"changes": changes, "event": event})
                except Exception:
                    pass
        return changes

    def _media_snapshot_items(self, media_context: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        items: dict[str, dict[str, Any]] = {}
        for item in media_context or []:
            if not isinstance(item, dict):
                continue
            key = self._media_item_key(item)
            if not key:
                continue
            items[key] = {
                "key": key,
                "player": str(item.get("player") or ""),
                "status": str(item.get("status") or ""),
                "title": str(item.get("title") or ""),
                "artist": str(item.get("artist") or ""),
                "url": str(item.get("url") or ""),
                "length_us": self._coerce_int(item.get("length_us")),
                "position_us": self._coerce_int(item.get("position_us")),
                "position_label": str(item.get("position_label") or ""),
                "progress_percent": item.get("progress_percent"),
            }
        return items

    def _media_item_key(self, item: dict[str, Any]) -> str:
        player = str(item.get("player") or "").strip()
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        if url:
            return f"{player}|{url}"
        if title:
            return f"{player}|title:{title}"
        return player

    def _detect_media_state_changes(
        self,
        previous_items: dict[str, Any],
        current_items: dict[str, dict[str, Any]],
        *,
        observed_at: str,
        previous_observed_at: str = "",
    ) -> list[dict[str, Any]]:
        changes: list[dict[str, Any]] = []
        elapsed_seconds = self._elapsed_seconds_between(previous_observed_at, observed_at)
        previous = {
            str(key): value
            for key, value in (previous_items or {}).items()
            if isinstance(value, dict)
        }
        for key, old in previous.items():
            if key not in current_items:
                changes.append(self._media_change("media_stopped", old, None, observed_at=observed_at))
        for key, current in current_items.items():
            old = previous.get(key)
            if old is None:
                new_status = str(current.get("status") or "")
                if new_status == "Playing":
                    event = "media_started"
                elif new_status == "Paused":
                    event = "media_loaded_paused"
                else:
                    event = "media_loaded"
                changes.append(self._media_change(event, None, current, observed_at=observed_at))
                continue
            old_status = str(old.get("status") or "")
            new_status = str(current.get("status") or "")
            if old_status == "Playing" and new_status == "Paused":
                changes.append(self._media_change("media_paused", old, current, observed_at=observed_at))
            elif old_status == "Paused" and new_status == "Playing":
                changes.append(self._media_change("media_resumed", old, current, observed_at=observed_at))
            elif new_status == "Stopped" and old_status != "Stopped":
                changes.append(self._media_change("media_stopped", old, current, observed_at=observed_at))
            seek_change = self._media_seek_change(
                old,
                current,
                observed_at=observed_at,
                elapsed_seconds=elapsed_seconds,
            )
            if seek_change is not None:
                changes.append(seek_change)
        return changes

    def _media_seek_change(
        self,
        old: dict[str, Any],
        current: dict[str, Any],
        *,
        observed_at: str,
        elapsed_seconds: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        old_position = self._coerce_int(old.get("position_us"))
        new_position = self._coerce_int(current.get("position_us"))
        if old_position is None or new_position is None:
            return None
        delta_us = new_position - old_position
        if abs(delta_us) < 15_000_000:
            return None
        old_status = str(old.get("status") or "").strip().lower()
        new_status = str(current.get("status") or "").strip().lower()
        if elapsed_seconds is not None and old_status == "playing" and new_status == "playing":
            expected_delta_us = max(0.0, elapsed_seconds) * 1_000_000.0
            tolerance_us = max(20_000_000.0, min(120_000_000.0, expected_delta_us * 0.35))
            if abs(float(delta_us) - expected_delta_us) <= tolerance_us:
                return None
        if elapsed_seconds is not None and old_status == "playing" and new_status == "paused" and delta_us > 0:
            expected_max_us = (max(0.0, elapsed_seconds) + 20.0) * 1_000_000.0
            if float(delta_us) <= expected_max_us:
                return None
        if elapsed_seconds is not None and new_status == "playing" and delta_us > 0:
            expected_max_us = (max(0.0, elapsed_seconds) + 20.0) * 1_000_000.0
            if float(delta_us) <= expected_max_us:
                return None
        return self._media_change(
            "media_seeked",
            old,
            current,
            observed_at=observed_at,
            extra={"position_delta_seconds": round(delta_us / 1_000_000.0, 3)},
        )

    def _elapsed_seconds_between(self, previous_observed_at: str, observed_at: str) -> Optional[float]:
        if not previous_observed_at or not observed_at:
            return None
        try:
            previous = datetime.fromisoformat(str(previous_observed_at))
            current = datetime.fromisoformat(str(observed_at))
        except ValueError:
            return None
        if previous.tzinfo is None:
            previous = previous.replace(tzinfo=timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        delta = (current - previous).total_seconds()
        if delta < 0:
            return None
        return delta

    def _media_change(
        self,
        event: str,
        previous: Optional[dict[str, Any]],
        current: Optional[dict[str, Any]],
        *,
        observed_at: str,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        item = current or previous or {}
        payload = {
            "event": event,
            "observed_at": observed_at,
            "player": str(item.get("player") or ""),
            "title": str(item.get("title") or ""),
            "artist": str(item.get("artist") or ""),
            "url": str(item.get("url") or ""),
            "from_status": str((previous or {}).get("status") or ""),
            "to_status": str((current or {}).get("status") or ""),
            "from_position_label": str((previous or {}).get("position_label") or ""),
            "to_position_label": str((current or {}).get("position_label") or ""),
            "from_position_us": self._coerce_int((previous or {}).get("position_us")),
            "to_position_us": self._coerce_int((current or {}).get("position_us")),
        }
        if extra:
            payload.update(extra)
        return payload

    def _load_media_state(self) -> dict[str, Any]:
        path = self._media_state_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _save_media_state(self, payload: dict[str, Any]) -> None:
        path = self._media_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")

    def _declared_task(self) -> Optional[str]:
        task = str(self.config.declared_task or "").strip()
        return task or None

    def _apply_speech_policy(self, analysis: dict[str, Any], *, reason: str) -> dict[str, Any]:
        normalized = dict(analysis)
        declared_task = self._declared_task()
        normalized["declared_task"] = declared_task
        normalized["task_coaching_allowed"] = bool(declared_task)
        normalized["speech_intent"] = self._normalize_speech_intent(normalized.get("speech_intent"))
        normalized["speech_relevance_score"] = self._coerce_float(
            normalized.get("speech_relevance_score"),
            default=0.0,
        )
        if not normalized.get("should_speak"):
            normalized["speech_policy"] = str(normalized.get("speech_policy") or "silent_by_model")
            return normalized

        speech_text = str(normalized.get("spoken_text") or "")
        reason_text = str(normalized.get("reason") or reason or "")
        is_task_coaching = (
            normalized["speech_intent"] == "task_coaching"
            or self._looks_like_task_coaching(speech_text)
            or self._looks_like_task_coaching(reason_text)
        )
        if is_task_coaching and not declared_task:
            normalized["should_speak"] = False
            normalized["speech_policy"] = "suppressed_no_declared_task_for_coaching"
            return normalized

        if declared_task:
            normalized["speech_policy"] = "allowed_declared_task"
            return normalized

        relevant_intents = {"screen_relevant", "system_issue", "safety_privacy"}
        if (
            normalized["speech_intent"] in relevant_intents
            and normalized["speech_relevance_score"] >= self.config.speech_relevance_threshold
        ):
            normalized["speech_policy"] = "allowed_relevant_observation"
            return normalized

        normalized["should_speak"] = False
        normalized["speech_policy"] = "suppressed_no_relevant_reason"
        return normalized

    def _apply_media_playback_speech_policy(
        self,
        analysis: dict[str, Any],
        capture_result: dict[str, Any],
        *,
        reason: str,
    ) -> dict[str, Any]:
        normalized = dict(analysis)
        if not normalized.get("should_speak"):
            return normalized
        if self._media_playback_speech_allowed(capture_result):
            return normalized
        speech_intent = str(normalized.get("speech_intent") or "none")
        if (
            speech_intent not in {"screen_relevant", "none"}
            and not str(reason or "").startswith("media_commentary_mode:")
        ):
            return normalized
        normalized["should_speak"] = False
        normalized["speech_policy"] = "suppressed_media_not_playing"
        normalized["speech_suppressed_reason"] = "media_not_playing"
        return normalized

    def _media_playback_speech_allowed(self, capture_result: dict[str, Any]) -> bool:
        if not self._has_playback_sensitive_media_context(capture_result):
            return True
        media_context = capture_result.get("media_context")
        if not isinstance(media_context, list) or not media_context:
            return True
        observed_identity = self._primary_media_identity(capture_result)
        if observed_identity:
            for item in self._ordered_media_context(media_context):
                if not isinstance(item, dict):
                    continue
                if self._media_identity_for_item(item) == observed_identity:
                    return self._media_item_is_playing(item)
            return False
        return any(
            self._media_item_is_playing(item)
            for item in self._ordered_media_context(media_context)
            if isinstance(item, dict)
        )

    def _has_playback_sensitive_media_context(self, capture_result: dict[str, Any]) -> bool:
        transcript = capture_result.get("youtube_transcript")
        if isinstance(transcript, dict) and transcript.get("status") == "available":
            return True
        live_transcript = capture_result.get("live_transcript")
        if isinstance(live_transcript, dict) and live_transcript.get("status") == "available":
            return True
        media_context = capture_result.get("media_context")
        if isinstance(media_context, list):
            for item in media_context:
                if not isinstance(item, dict):
                    continue
                if str(item.get("title") or item.get("url") or "").strip():
                    return True
        return False

    def _media_item_is_playing(self, item: dict[str, Any]) -> bool:
        return str(item.get("status") or "").strip().lower() == "playing"

    async def _maybe_apply_livestream_resume_catchup(
        self,
        *,
        paused_players: list[str],
        media_context: list[dict[str, Any]],
        pause_duration_seconds: float,
    ) -> dict[str, Any]:
        if not self.config.livestream_resume_catchup_enabled:
            return {"status": "skipped", "reason": "disabled"}
        players = [str(player or "").strip() for player in paused_players if str(player or "").strip()]
        if not players:
            return {"status": "skipped", "reason": "no_paused_players"}
        livestream_items = [
            item
            for item in media_context or []
            if isinstance(item, dict)
            and str(item.get("player") or "").strip() in players
            and self._media_item_is_probable_livestream(item)
        ]
        if not livestream_items:
            return {"status": "skipped", "reason": "no_livestream_players"}
        livestream_players = list(
            dict.fromkeys(
                str(item.get("player") or "").strip()
                for item in livestream_items
                if str(item.get("player") or "").strip()
            )
        )
        rate = float(self.config.livestream_resume_catchup_rate)
        if rate <= 1.0:
            return {"status": "skipped", "reason": "rate_not_above_normal", "rate": rate}

        rate_result: dict[str, Any] = {"rate_set_players": [], "errors": []}
        set_rate = getattr(self._media_controller, "set_players_rate", None)
        if callable(set_rate):
            try:
                raw = await self._call_maybe_async(set_rate, livestream_players, rate)
                if isinstance(raw, dict):
                    rate_result.update(raw)
            except Exception as exc:
                rate_result.setdefault("errors", []).append(
                    {"action": "set_rate", "error": f"{type(exc).__name__}: {exc}"}
                )

        rate_set_players = list(rate_result.get("rate_set_players") or [])
        youtube_result: Optional[dict[str, Any]] = None
        failed_reason = "rate_set_failed"
        method = "mpris_rate"
        if not rate_set_players and any(self._media_item_is_youtube(item) for item in livestream_items):
            if len(livestream_players) > 1:
                youtube_result = {
                    "ok": False,
                    "method": "youtube_browser_fallback",
                    "error": "ambiguous_multiple_livestream_players",
                }
                failed_reason = "ambiguous_multiple_livestream_players"
            else:
                fallback = getattr(self._media_controller, "set_youtube_browser_rate", None)
                if callable(fallback):
                    try:
                        raw_fallback = await self._call_youtube_browser_rate(
                            fallback,
                            rate,
                            media_item=livestream_items[0],
                        )
                        if isinstance(raw_fallback, dict):
                            youtube_result = raw_fallback
                            if raw_fallback.get("ok"):
                                method = "youtube_browser_fallback"
                                rate_set_players = list(livestream_players)
                    except Exception as exc:
                        youtube_result = {
                            "ok": False,
                            "method": "youtube_browser_fallback",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                else:
                    youtube_result = {
                        "ok": False,
                        "method": "youtube_browser_fallback",
                        "error": "youtube_browser_fallback_unavailable",
                    }
                    failed_reason = "youtube_browser_fallback_unavailable"

        status = "applied" if rate_set_players else "failed"
        payload: dict[str, Any] = {
            "status": status,
            "method": method,
            "rate": rate,
            "players": livestream_players,
            "rate_set_players": rate_set_players,
            "media": [
                {
                    "player": item.get("player"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "length_us": item.get("length_us"),
                    "position_us": item.get("position_us"),
                }
                for item in livestream_items
            ],
            "mpris": rate_result,
        }
        if youtube_result is not None:
            payload["youtube_browser"] = youtube_result
        if status == "applied":
            restore_after = self._livestream_catchup_restore_after_seconds(
                pause_duration_seconds=pause_duration_seconds,
                rate=rate,
            )
            if restore_after is not None:
                payload["restore_after_seconds"] = restore_after
                self._schedule_livestream_rate_restore(
                    players=livestream_players,
                    method=method,
                    delay_seconds=restore_after,
                    media_snapshot=payload["media"],
                )
        else:
            payload["reason"] = failed_reason
        return payload

    def _media_item_is_probable_livestream(self, item: dict[str, Any]) -> bool:
        length_us = self._coerce_int(item.get("length_us"))
        if length_us is not None and length_us > 0:
            return False
        if bool(item.get("is_live")):
            return True
        live_status = str(item.get("live_status") or "").strip().lower()
        if live_status in {"is_live", "live", "live_stream", "livestream", "currently_live"}:
            return True
        url = str(item.get("url") or "").strip().lower()
        title = str(item.get("title") or "").strip().lower()
        album = str(item.get("album") or "").strip().lower()
        haystack = f" {title} {album} {url} "
        return any(term in haystack for term in (" livestream ", " live-stream ", " live stream ", " live now "))

    async def _call_youtube_browser_rate(
        self,
        fallback: Callable[..., Any],
        rate: float,
        *,
        media_item: dict[str, Any],
    ) -> Any:
        if self._callable_accepts_keyword(fallback, "media_item"):
            return await self._call_maybe_async(fallback, rate, media_item=media_item)
        return await self._call_maybe_async(fallback, rate)

    def _callable_accepts_keyword(self, fn: Callable[..., Any], keyword: str) -> bool:
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                return True
            if parameter.name == keyword:
                return True
        return False

    def _media_item_is_youtube(self, item: dict[str, Any]) -> bool:
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        return bool(self._youtube_video_id(url) or self._first_youtube_url(title))

    def _livestream_catchup_restore_after_seconds(
        self,
        *,
        pause_duration_seconds: float,
        rate: float,
    ) -> Optional[float]:
        max_seconds = float(self.config.livestream_resume_catchup_max_seconds)
        if max_seconds <= 0:
            return None
        if rate <= 1.0:
            return None
        required = max(1.0, float(pause_duration_seconds or 0.0) / (rate - 1.0))
        return round(min(max_seconds, required), 2)

    def _schedule_livestream_rate_restore(
        self,
        *,
        players: list[str],
        method: str,
        delay_seconds: float,
        media_snapshot: list[dict[str, Any]],
    ) -> None:
        if delay_seconds <= 0:
            return
        task = asyncio.create_task(
            self._restore_livestream_rate_after_delay(
                players=list(players),
                method=method,
                delay_seconds=delay_seconds,
                media_snapshot=list(media_snapshot),
            )
        )
        self._livestream_rate_restore_tasks.add(task)
        task.add_done_callback(self._livestream_rate_restore_tasks.discard)

    async def _restore_livestream_rate_after_delay(
        self,
        *,
        players: list[str],
        method: str,
        delay_seconds: float,
        media_snapshot: list[dict[str, Any]],
    ) -> None:
        await asyncio.sleep(delay_seconds)
        result: dict[str, Any] = {}
        try:
            if method == "youtube_browser_fallback":
                current_media = await self._current_media_context()
                if not self._livestream_catchup_media_still_current(
                    current_media=current_media,
                    media_snapshot=media_snapshot,
                ):
                    result = {"ok": False, "skipped": True, "reason": "media_changed_or_missing"}
                else:
                    fallback = getattr(self._media_controller, "set_youtube_browser_rate", None)
                    if callable(fallback):
                        raw = await self._call_youtube_browser_rate(
                            fallback,
                            1.0,
                            media_item=media_snapshot[0] if media_snapshot else {},
                        )
                        result = raw if isinstance(raw, dict) else {"ok": bool(raw)}
                    else:
                        result = {"ok": False, "error": "youtube_browser_fallback_unavailable"}
            else:
                set_rate = getattr(self._media_controller, "set_players_rate", None)
                if callable(set_rate):
                    raw = await self._call_maybe_async(set_rate, players, 1.0)
                    result = raw if isinstance(raw, dict) else {"ok": bool(raw)}
                else:
                    result = {"ok": False, "error": "mpris_rate_unavailable"}
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self._event(
            "livestream_catchup_rate_restored",
            {
                "method": method,
                "players": players,
                "rate": 1.0,
                "delay_seconds": delay_seconds,
                "result": result,
            },
        )

    def _livestream_catchup_media_still_current(
        self,
        *,
        current_media: list[dict[str, Any]],
        media_snapshot: list[dict[str, Any]],
    ) -> bool:
        expected: list[tuple[str, str]] = []
        for item in media_snapshot or []:
            if not isinstance(item, dict):
                continue
            player = str(item.get("player") or "").strip()
            url = str(item.get("url") or "").strip()
            if player and url:
                expected.append((player, url))
        if not expected:
            return False
        for item in current_media or []:
            if not isinstance(item, dict):
                continue
            player = str(item.get("player") or "").strip()
            url = str(item.get("url") or "").strip()
            if (player, url) in expected and self._media_item_is_probable_livestream(item):
                return True
        return False

    def _looks_like_task_coaching(self, text: str) -> bool:
        lowered = str(text or "").lower()
        patterns = [
            "stay on task",
            "get back to",
            "back to the task",
            "focus on",
            "keep focusing",
            "keep working",
            "drifting",
            "drifted",
            "next small step",
            "next step",
            "productive",
            "productivity",
        ]
        return any(pattern in lowered for pattern in patterns)

    def _normalize_speech_intent(self, value: Any) -> str:
        raw = str(value or "none").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "task": "task_coaching",
            "focus": "task_coaching",
            "focus_checkin": "task_coaching",
            "screen": "screen_relevant",
            "observation": "screen_relevant",
            "system": "system_issue",
            "error": "system_issue",
            "safety": "safety_privacy",
            "privacy": "safety_privacy",
        }
        normalized = aliases.get(raw, raw)
        allowed = {"none", "task_coaching", "screen_relevant", "system_issue", "safety_privacy"}
        return normalized if normalized in allowed else "none"

    def _coerce_float(self, value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _coerce_int(self, value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _media_progress_percent(self, position_us: Optional[int], length_us: Optional[int]) -> Optional[float]:
        if position_us is None or length_us is None or length_us <= 0:
            return None
        return max(0.0, min(100.0, (float(position_us) / float(length_us)) * 100.0))

    def _format_media_position(self, position_us: Optional[int], length_us: Optional[int]) -> str:
        if position_us is None:
            return ""
        position = self._format_microseconds(position_us)
        if length_us is None or length_us <= 0:
            return f"position {position}"
        return f"position {position} / {self._format_microseconds(length_us)}"

    def _format_microseconds(self, value: int) -> str:
        total_seconds = max(0, int(round(value / 1_000_000)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    def _format_seconds_label(self, value: float) -> str:
        total_seconds = max(0, int(round(float(value))))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    def _parse_response_json(self, response: dict[str, Any]) -> Optional[dict[str, Any]]:
        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict)
            )
        text = str(content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None

    async def _persist_context(
        self,
        content: str,
        capture_result: dict[str, Any],
        analysis: dict[str, Any],
        *,
        reason: str,
        session_id: Optional[str] = None,
    ) -> None:
        store = getattr(getattr(self.runtime, "ctx", None), "context_store", None)
        session_id = session_id or self._session_id()
        observed_at = self._now().isoformat()
        observation_payload = self._observation_payload(
            capture_result,
            analysis,
            reason=reason,
            observed_at=observed_at,
        )
        meta = {
            "source": "desktop_context",
            "capture": capture_result.get("capture"),
            "analysis": analysis,
            "observation": observation_payload,
        }
        if store is not None and hasattr(store, "append"):
            await store.append(session_id, MessageRole.SYSTEM, content, meta=meta)
        record_episode = getattr(self.runtime, "_record_episode", None)
        episode = None
        if callable(record_episode):
            try:
                episode = await record_episode(
                    content,
                    EpisodeKind.OBSERVATION,
                    session_id=session_id,
                    role="desktop_context",
                    payload=observation_payload,
                )
            except Exception:
                pass
        await self._persist_observation_memory(
            episode,
            capture_result,
            analysis,
            payload=observation_payload,
        )

    async def _persist_observation_memory(
        self,
        episode: Any,
        capture_result: dict[str, Any],
        analysis: dict[str, Any],
        *,
        payload: dict[str, Any],
    ) -> None:
        episode_id = getattr(episode, "episode_id", None)
        if not episode_id:
            return
        memory_store = getattr(self.runtime, "memory", None) or getattr(
            getattr(self.runtime, "ctx", None),
            "memory",
            None,
        )
        save_memory = getattr(memory_store, "save_memory", None)
        if not callable(save_memory):
            return
        memory = Memory(
            content=self._build_observation_memory_text(capture_result, analysis, payload=payload),
            source_episode_ids=[str(episode_id)],
            tags=[
                "desktop_context",
                "body_double",
                "observed_user_activity",
                "learning_from_observation",
                "live_observation",
            ],
            salience=2.4,
            confidence_score=float(payload.get("confidence_score") or 0.72),
        )
        try:
            await self._call_maybe_async(save_memory, memory)
        except Exception:
            pass

    def _observation_payload(
        self,
        capture_result: dict[str, Any],
        analysis: dict[str, Any],
        *,
        reason: str,
        observed_at: str,
    ) -> dict[str, Any]:
        capture = capture_result.get("capture") or {}
        ocr_text = str(capture_result.get("ocr_text") or "").strip()
        activity_summary = str(analysis.get("activity_summary") or "").strip()
        note = str(analysis.get("note") or "").strip()
        declared_task = self._declared_task()
        return {
            "source": "desktop_context",
            "source_lane": "executive",
            "context_authority": "live_observation",
            "context_material": "desktop_observation",
            "origin": "desktop_context_observation",
            "observation_kind": "user_activity",
            "reason": reason,
            "declared_task": declared_task,
            "declared_task_source": self.config.declared_task_source,
            "declared_task_updated_at": self.config.declared_task_updated_at,
            "task_coaching_allowed": bool(declared_task),
            "observed_user_activity": activity_summary,
            "operator_activity_summary": activity_summary,
            "note": note,
            "should_speak": bool(analysis.get("should_speak")),
            "speech_intent": str(analysis.get("speech_intent") or "none"),
            "speech_relevance_score": float(analysis.get("speech_relevance_score") or 0.0),
            "speech_policy": str(analysis.get("speech_policy") or ""),
            "confidence_score": 0.78 if activity_summary else 0.62,
            "media_context": capture_result.get("media_context") or [],
            "media_state_changes": capture_result.get("media_state_changes") or [],
            "youtube_transcript": capture_result.get("youtube_transcript"),
            "live_transcript": capture_result.get("live_transcript"),
            "self_interest_followup": {
                key: value
                for key, value in (analysis.get("self_interest_followup") or {}).items()
                if key not in {"raw", "interests_considered"}
            }
            if isinstance(analysis.get("self_interest_followup"), dict)
            else None,
            "temporal": {
                "observed_at": observed_at,
                "captured_at": observed_at,
                "timezone": "UTC",
            },
            "evidence": {
                "screenshot_path": str(capture.get("path") or ""),
                "screenshot_backend": str(capture.get("backend") or ""),
                "media_type": str(capture.get("media_type") or ""),
                "width": capture.get("width"),
                "height": capture.get("height"),
                "ocr_chars": len(ocr_text),
                "ocr_excerpt": ocr_text[: min(len(ocr_text), self.config.max_ocr_chars, 1200)],
            },
        }

    def _build_observation_memory_text(
        self,
        capture_result: dict[str, Any],
        analysis: dict[str, Any],
        *,
        payload: dict[str, Any],
    ) -> str:
        evidence = payload.get("evidence") or {}
        temporal = payload.get("temporal") or {}
        activity = str(payload.get("observed_user_activity") or "(not summarized)").strip()
        note = str(payload.get("note") or "").strip()
        ocr_excerpt = str(evidence.get("ocr_excerpt") or "").strip()
        transcript = payload.get("youtube_transcript") if isinstance(payload.get("youtube_transcript"), dict) else {}
        live_transcript = payload.get("live_transcript") if isinstance(payload.get("live_transcript"), dict) else {}
        media_context = payload.get("media_context") if isinstance(payload.get("media_context"), list) else []
        media_state_changes = (
            payload.get("media_state_changes")
            if isinstance(payload.get("media_state_changes"), list)
            else []
        )
        followup = payload.get("self_interest_followup") if isinstance(payload.get("self_interest_followup"), dict) else {}
        lines = [
            "Observed user activity from desktop context.",
            f"Observed at: {temporal.get('observed_at') or ''}",
            f"Observed user activity: {activity}",
            f"Screenshot evidence: {evidence.get('screenshot_path') or ''}",
            f"Reason: {payload.get('reason') or ''}",
        ]
        if media_context:
            item = media_context[0] if isinstance(media_context[0], dict) else {}
            media_parts = [
                str(item.get("title") or "").strip(),
                str(item.get("artist") or "").strip(),
                str(item.get("url") or "").strip(),
                str(item.get("status") or "").strip(),
                str(item.get("position_label") or "").strip(),
            ]
            lines.append(f"Media observed: {' | '.join(part for part in media_parts if part)}")
        if media_state_changes:
            lines.append("Media state changes observed:")
            for change in media_state_changes[:6]:
                if isinstance(change, dict):
                    lines.append(
                        f"- {change.get('event')}: {change.get('title') or change.get('url') or ''} "
                        f"{change.get('from_position_label') or ''} -> {change.get('to_position_label') or ''}".strip()
                    )
        if transcript and transcript.get("status") == "available":
            lines.append(f"YouTube transcript evidence: {transcript.get('transcript_path') or ''}")
            excerpt = str(transcript.get("transcript_excerpt") or "").strip()
            if excerpt:
                lines.append("YouTube transcript excerpt:")
                lines.append(excerpt[:2500])
        if live_transcript and live_transcript.get("status") == "available":
            excerpt = str(live_transcript.get("transcript_excerpt") or "").strip()
            lines.append("Live Whisper transcript evidence:")
            if excerpt:
                lines.append(excerpt[:2500])
            if transcript and transcript.get("status") == "available":
                lines.append("Transcript fusion: prefetched transcript is the timestamped map; live Whisper is current audio.")
        if followup:
            if followup.get("matches_self_interest") and not followup.get("matches_shared_work"):
                lines.append("Self-directed media curiosity follow-up:")
            else:
                lines.append("Observed context relevance follow-up:")
            if followup.get("agent_viewpoint") or followup.get("implications") or followup.get("open_questions"):
                lines.append("Observed media thought artifact:")
            if followup.get("agent_viewpoint"):
                lines.append(f"Agent viewpoint: {followup.get('agent_viewpoint')}")
            if followup.get("connection_summary"):
                lines.append(str(followup.get("connection_summary")))
            implications = self._format_followup_list("Implications", followup.get("implications"))
            if implications:
                lines.append(implications)
            questions = self._format_followup_list("Open questions", followup.get("open_questions"))
            if questions:
                lines.append(questions)
            matched_targets = followup.get("matched_targets") or followup.get("matched_interests")
            if matched_targets:
                lines.append(f"Matched relevance targets: {', '.join(matched_targets or [])}")
            elif followup.get("matched_interests"):
                lines.append(f"Matched self-interests: {', '.join(followup.get('matched_interests') or [])}")
            if followup.get("self_directed_next_step"):
                lines.append(f"Next self-directed step: {followup.get('self_directed_next_step')}")
            if followup.get("timecode_or_position"):
                lines.append(f"Relevant media position: {followup.get('timecode_or_position')}")
        if note:
            lines.append(f"Observation note: {note}")
        if ocr_excerpt:
            lines.append("OCR evidence excerpt:")
            lines.append(ocr_excerpt)
        return "\n".join(lines).strip()

    def _build_context_text(
        self,
        capture_result: dict[str, Any],
        analysis: dict[str, Any],
        *,
        reason: str,
    ) -> str:
        capture = capture_result.get("capture") or {}
        ocr_text = str(capture_result.get("ocr_text") or "").strip()
        lines = [
            f"Recent desktop context ({self._now().isoformat()}):",
            f"- reason: {reason}",
            f"- screenshot: {capture.get('path') or '(unavailable)'}",
            f"- Observed user activity: {analysis.get('activity_summary') or '(not summarized)'}",
            f"- activity: {analysis.get('activity_summary') or '(not summarized)'}",
            f"- comment decision: {'speak' if analysis.get('should_speak') else 'hold'}",
        ]
        media_context = capture_result.get("media_context")
        if isinstance(media_context, list) and media_context:
            item = media_context[0]
            if isinstance(item, dict):
                media_parts = [
                    str(item.get("title") or "").strip(),
                    str(item.get("artist") or "").strip(),
                    str(item.get("url") or "").strip(),
                    str(item.get("status") or "").strip(),
                    str(item.get("position_label") or "").strip(),
                ]
                lines.append(f"- media: {' | '.join(part for part in media_parts if part)}")
        media_state_changes = capture_result.get("media_state_changes")
        if isinstance(media_state_changes, list) and media_state_changes:
            lines.append("- media state changes:")
            for change in media_state_changes[:6]:
                if isinstance(change, dict):
                    lines.append(
                        f"  - {change.get('event')}: {change.get('title') or change.get('url') or ''} "
                        f"{change.get('from_position_label') or ''} -> {change.get('to_position_label') or ''}".strip()
                    )
        transcript = capture_result.get("youtube_transcript")
        if isinstance(transcript, dict) and transcript.get("status") == "available":
            lines.append(f"- YouTube transcript: {transcript.get('transcript_path') or ''}")
            excerpt = str(transcript.get("transcript_excerpt") or "").strip()
            if excerpt:
                lines.append("- YouTube transcript excerpt:")
                lines.append(excerpt[:2500])
        live_transcript = capture_result.get("live_transcript")
        if isinstance(live_transcript, dict) and live_transcript.get("status") == "available":
            lines.append("- live Whisper transcript: local audio")
            excerpt = str(live_transcript.get("transcript_excerpt") or "").strip()
            if excerpt:
                lines.append("- live Whisper transcript excerpt:")
                lines.append(excerpt[:2500])
        note = str(analysis.get("note") or "").strip()
        followup = analysis.get("self_interest_followup") if isinstance(analysis.get("self_interest_followup"), dict) else {}
        if followup:
            lines.append(f"- observed-context relevance match: {followup.get('connection_summary') or '(not summarized)'}")
            if followup.get("agent_viewpoint"):
                lines.append(f"- agent viewpoint: {followup.get('agent_viewpoint')}")
            implications = self._format_followup_list("implications", followup.get("implications"))
            if implications:
                lines.append(f"- {implications}")
            questions = self._format_followup_list("open questions", followup.get("open_questions"))
            if questions:
                lines.append(f"- {questions}")
            matched_targets = followup.get("matched_targets") or followup.get("matched_interests")
            if matched_targets:
                lines.append(f"- matched relevance targets: {', '.join(matched_targets or [])}")
            if followup.get("self_directed_next_step"):
                lines.append(f"- next relevance step: {followup.get('self_directed_next_step')}")
            if followup.get("timecode_or_position"):
                lines.append(f"- relevant media position: {followup.get('timecode_or_position')}")
        if note:
            lines.append(f"- note: {note}")
        if ocr_text:
            lines.append("- OCR excerpt:")
            lines.append(ocr_text[: self.config.max_ocr_chars])
        return "\n".join(lines)

    def _build_region_context_text(
        self,
        capture_result: dict[str, Any],
        analysis: dict[str, Any],
        *,
        prompt: str,
        reason: str,
    ) -> str:
        capture = capture_result.get("capture") or {}
        region = analysis.get("region_prompt") if isinstance(analysis.get("region_prompt"), dict) else {}
        ocr_text = str(capture_result.get("ocr_text") or "").strip()
        lines = [
            f"Selected desktop region context ({self._now().isoformat()}):",
            f"- reason: {reason}",
            f"- operator question/instruction: {prompt}",
            f"- screenshot: {capture.get('path') or '(unavailable)'}",
            f"- selection: {json.dumps(capture_result.get('region_selection') or {}, sort_keys=True)}",
            f"- visual summary: {region.get('visual_summary') or analysis.get('activity_summary') or '(not summarized)'}",
        ]
        if region.get("visible_text"):
            lines.append("- visible/OCR text:")
            lines.append(str(region.get("visible_text"))[: self.config.max_ocr_chars])
        if region.get("answer_context"):
            lines.append(f"- answer context: {region.get('answer_context')}")
        if region.get("uncertainty"):
            lines.append(f"- uncertainty: {region.get('uncertainty')}")
        if ocr_text and not region.get("visible_text"):
            lines.append("- OCR excerpt:")
            lines.append(ocr_text[: self.config.max_ocr_chars])
        return "\n".join(lines)

    async def _speak_analysis(
        self,
        analysis: dict[str, Any],
        capture_result: dict[str, Any],
        *,
        reason: str,
        force: bool = False,
    ) -> dict[str, Any]:
        if not self.config.tts_enabled:
            return {"status": "skipped", "reason": "tts_disabled"}
        if not force and not self._speech_due():
            return {"status": "skipped", "reason": "speech_not_due"}
        prepared = self._prepare_spoken_text(analysis, capture_result, reason=reason)
        spoken_text = prepared["spoken_text"]
        if not spoken_text:
            if prepared.get("redirected_to_note"):
                return {
                    "status": "skipped",
                    "reason": "redirected_to_note_without_canned_speech",
                    **{key: value for key, value in prepared.items() if key != "spoken_text"},
                }
            return {"status": "skipped", "reason": "empty_spoken_text"}
        if self._is_policy_boilerplate_spoken_text(spoken_text):
            return {"status": "skipped", "reason": "policy_boilerplate_spoken_text"}
        spoken_hash = self._spoken_text_hash(spoken_text)
        if not force and self._recent_spoken_text_has_hash(spoken_hash):
            return {"status": "skipped", "reason": "repeated_spoken_text"}
        context_signature = self._spoken_context_signature(capture_result)
        if not force and context_signature and self._recent_spoken_context_has_signature(context_signature):
            return {"status": "skipped", "reason": "repeated_spoken_context"}
        if not force:
            stale_reason = await self._media_context_stale_for_speech(analysis, capture_result, reason=reason)
            if stale_reason:
                return {"status": "skipped", "reason": stale_reason}

        try:
            synth = self._speech_synthesizer or self._default_speech_synthesizer
            voice_meta = await self._call_maybe_async(synth, spoken_text)
        except Exception as exc:
            return {"status": "failed", "reason": f"tts_failed:{type(exc).__name__}", "error": str(exc)}

        playback: Optional[dict[str, Any]] = None
        audio_path = self._voice_path(voice_meta)
        if self.config.play_audio and audio_path is not None:
            media: dict[str, Any] = {"paused_players": [], "errors": []}
            pre_pause_media_context: list[dict[str, Any]] = []
            pause_started_at: Optional[float] = None
            try:
                pause = getattr(self._media_controller, "pause_playing", None)
                if callable(pause):
                    pre_pause_media_context = await self._current_media_context()
                    pause_started_at = asyncio.get_running_loop().time()
                    paused = await self._call_maybe_async(pause)
                    if isinstance(paused, dict):
                        media.update(paused)
                played = await self._call_maybe_async(self._audio_player, audio_path)
                playback = played if isinstance(played, dict) else {"played": bool(played), "path": str(audio_path)}
            except Exception as exc:
                playback = {"played": False, "error": str(exc), "path": str(audio_path)}
            finally:
                resume = getattr(self._media_controller, "resume_players", None)
                paused_players = list(media.get("paused_players") or [])
                if callable(resume) and paused_players:
                    try:
                        resumed = await self._call_maybe_async(resume, paused_players)
                        if isinstance(resumed, dict):
                            media.update(resumed)
                    except Exception as exc:
                        media.setdefault("errors", []).append(
                            {"action": "resume", "error": f"{type(exc).__name__}: {exc}"}
                        )
                pause_duration_seconds = 0.0
                if pause_started_at is not None:
                    pause_duration_seconds = max(
                        0.0,
                        asyncio.get_running_loop().time() - pause_started_at,
                    )
                media["livestream_catchup"] = await self._maybe_apply_livestream_resume_catchup(
                    paused_players=paused_players,
                    media_context=pre_pause_media_context,
                    pause_duration_seconds=pause_duration_seconds,
                )
                if playback is not None:
                    playback["media"] = media

        event = self._event(
            "spoken",
            {
                "reason": reason,
                "chars": len(spoken_text),
                "spoken_text_excerpt": spoken_text[:500],
                "spoken_text_hash": spoken_hash,
                "spoken_context_signature": context_signature,
                "voice": voice_meta,
                "playback": playback,
                "redirected_to_note": prepared.get("redirected_to_note", False),
            },
        )
        return {
            "status": "spoken",
            "spoken_text": spoken_text,
            "spoken_text_hash": spoken_hash,
            "spoken_context_signature": context_signature,
            "voice": voice_meta,
            "playback": playback,
            "event": event,
            **{key: value for key, value in prepared.items() if key != "spoken_text"},
        }

    def _spoken_text_hash(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "").strip()).lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _recent_spoken_text_has_hash(self, spoken_hash: str, *, limit: int = 80) -> bool:
        if not spoken_hash:
            return False
        for event in reversed(self._list_events(limit=limit)):
            if event.get("type") != "spoken":
                continue
            if str(event.get("spoken_text_hash") or "") == spoken_hash:
                return True
        return False

    def _recent_spoken_context_has_signature(self, signature: str, *, limit: int = 80) -> bool:
        if not signature:
            return False
        for event in reversed(self._list_events(limit=limit)):
            if event.get("type") != "spoken":
                continue
            if str(event.get("spoken_context_signature") or "") == signature:
                return True
        return False

    async def _media_context_stale_for_speech(
        self,
        analysis: dict[str, Any],
        capture_result: dict[str, Any],
        *,
        reason: str,
    ) -> str:
        if not self._speech_depends_on_media(analysis, capture_result, reason=reason):
            return ""
        observed_identity = self._primary_media_identity(capture_result)
        if not observed_identity:
            return ""
        current_context = await self._current_media_context()
        if not current_context:
            return "media_context_stale"
        for item in self._ordered_media_context(current_context):
            if not isinstance(item, dict):
                continue
            if self._media_identity_for_item(item) != observed_identity:
                continue
            if self._media_item_is_playing(item):
                if self._media_position_is_stale_for_speech(capture_result, item):
                    return "media_position_stale"
                return ""
            return "media_not_playing"
        return "media_context_stale"

    def _media_position_is_stale_for_speech(
        self,
        capture_result: dict[str, Any],
        current_item: dict[str, Any],
    ) -> bool:
        observed_position = self._primary_media_position_us(capture_result.get("media_context") or [])
        current_position = self._coerce_int(current_item.get("position_us"))
        if observed_position is None or current_position is None:
            return False
        delta_seconds = abs(float(current_position - observed_position) / 1_000_000.0)
        return delta_seconds > self._max_commentary_position_staleness_seconds(capture_result)

    def _max_commentary_position_staleness_seconds(self, capture_result: dict[str, Any]) -> float:
        transcript = capture_result.get("youtube_transcript")
        if isinstance(transcript, dict) and transcript.get("status") == "available":
            if self._transcript_excerpt_is_timing_safe(transcript):
                return max(8.0, float(self.config.youtube_transcript_commentary_lag_seconds) + 6.0)
            return 8.0
        return 20.0

    def _speech_depends_on_media(
        self,
        analysis: dict[str, Any],
        capture_result: dict[str, Any],
        *,
        reason: str,
    ) -> bool:
        if not self._has_playback_sensitive_media_context(capture_result):
            return False
        if str(reason or "").startswith("media_commentary_mode:"):
            return True
        if (
            self.config.media_commentary_mode_enabled
            and str(analysis.get("speech_intent") or "") == "screen_relevant"
        ):
            return True
        if isinstance(analysis.get("self_interest_followup"), dict):
            return True
        return False

    def _primary_media_identity(self, capture_result: dict[str, Any]) -> str:
        transcript = capture_result.get("youtube_transcript")
        if isinstance(transcript, dict) and transcript.get("status") == "available":
            video_id = str(transcript.get("video_id") or "").strip()
            url = str(transcript.get("url") or "").strip()
            return video_id or self._youtube_video_id(url) or url
        live_transcript = capture_result.get("live_transcript")
        if isinstance(live_transcript, dict) and live_transcript.get("status") == "available":
            media_identity = str(live_transcript.get("media_identity") or "").strip()
            media_url = str(live_transcript.get("media_url") or "").strip()
            media_title = str(live_transcript.get("media_title") or "").strip()
            return media_identity or self._youtube_video_id(media_url) or media_url or media_title
        media_context = capture_result.get("media_context")
        for item in self._ordered_media_context(media_context):
            identity = self._media_identity_for_item(item)
            if identity:
                return identity
        return ""

    def _media_identity_for_item(self, item: dict[str, Any]) -> str:
        if not isinstance(item, dict):
            return ""
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        return self._youtube_video_id(url) or url or title

    def _spoken_context_signature(self, capture_result: dict[str, Any]) -> str:
        if not isinstance(capture_result, dict):
            return ""
        live_transcript = capture_result.get("live_transcript")
        if isinstance(live_transcript, dict) and live_transcript.get("status") == "available":
            media_key = str(live_transcript.get("media_identity") or live_transcript.get("media_url") or "").strip()
            if not media_key:
                media_key = str(live_transcript.get("media_title") or "").strip()
            text_hash = str(live_transcript.get("transcript_hash") or "").strip()
            if not text_hash:
                text_hash = self._short_hash(
                    str(live_transcript.get("transcript_excerpt") or live_transcript.get("transcript_text") or "")
                )
            if media_key and text_hash:
                return hashlib.sha256(f"{media_key}|live_whisper|{text_hash}".encode("utf-8")).hexdigest()
        transcript = capture_result.get("youtube_transcript")
        media_context = capture_result.get("media_context")
        media_key = ""
        basis = "media"
        bucket = "unknown"
        if isinstance(transcript, dict) and transcript.get("status") == "available":
            video_id = str(transcript.get("video_id") or "").strip()
            url = str(transcript.get("url") or "").strip()
            media_key = video_id or self._youtube_video_id(url) or url
            basis = str(transcript.get("transcript_excerpt_basis") or "transcript").strip() or "transcript"
            start_seconds = self._coerce_float(transcript.get("transcript_aligned_start_seconds"), default=None)
            target_seconds = self._coerce_float(transcript.get("transcript_target_seconds"), default=None)
            seconds = start_seconds if start_seconds is not None else target_seconds
            if seconds is not None:
                bucket = str(int(max(0.0, seconds) // 60.0))
            else:
                position_label = str(transcript.get("transcript_position_label") or "").strip()
                if position_label:
                    bucket = position_label
        if not media_key and isinstance(media_context, list):
            for item in media_context:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                title = str(item.get("title") or "").strip()
                media_key = self._youtube_video_id(url) or url or title
                position_us = self._coerce_int(item.get("position_us"))
                if position_us is not None:
                    bucket = str(int(max(0, position_us) / 1_000_000 // 60))
                elif item.get("position_label"):
                    bucket = str(item.get("position_label"))
                if media_key:
                    break
        if not media_key:
            return ""
        raw = f"{media_key}|{basis}|{bucket}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _is_policy_boilerplate_spoken_text(self, text: str) -> bool:
        lower = re.sub(r"\s+", " ", str(text or "").strip().lower())
        if not lower:
            return False
        policy_patterns = [
            ("earns", "interruptions", "synthesis", "follow-through"),
            ("useful", "synthesis", "follow-through"),
            ("assistant", "useful", "management overhead"),
            ("opencas memory", "specific claim", "transcript segment", "design choice"),
            ("open cas memory", "specific claim", "transcript segment", "design choice"),
        ]
        return any(all(fragment in lower for fragment in pattern) for pattern in policy_patterns)

    def _scrub_generated_followup_text(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if self._is_policy_boilerplate_spoken_text(text):
            return ""
        return text

    async def _default_speech_synthesizer(self, text: str) -> dict[str, Any]:
        from opencas.api.voice_service import synthesize_speech

        result = await synthesize_speech(
            self._audio_dir(),
            text=text,
            prefer_local=True,
            expressive=False,
        )
        return result.to_meta()

    def _prepare_spoken_text(
        self,
        analysis: dict[str, Any],
        capture_result: dict[str, Any],
        *,
        reason: str,
    ) -> dict[str, Any]:
        raw = str(analysis.get("spoken_text") or "").strip()
        if not raw:
            return {"spoken_text": ""}
        if analysis.get("allow_note_redirect", True) and self._speech_should_be_note(raw):
            note_path = self._write_note_file(analysis, capture_result, reason=reason)
            return {
                "spoken_text": "",
                "redirected_to_note": True,
                "note_path": str(note_path),
            }
        max_chars = self._coerce_int(analysis.get("max_spoken_chars"))
        if max_chars is None or max_chars <= 0:
            max_chars = self.config.max_spoken_chars
        return {
            "spoken_text": self._shorten_for_speech(raw, max_chars),
            "redirected_to_note": False,
        }

    def _speech_should_be_note(self, text: str) -> bool:
        stripped = text.strip()
        if "```" in stripped:
            return True
        lines = [line for line in stripped.splitlines() if line.strip()]
        if len(lines) > 5:
            return True
        code_or_log = re.search(
            r"\b(traceback|stack trace|runtimeerror|syntaxerror|debug|warning|exception|"
            r"def |class |function |const |select |insert |update )\b",
            stripped,
            re.IGNORECASE,
        )
        return bool(code_or_log and len(stripped) > 160)

    def _write_note_file(
        self,
        analysis: dict[str, Any],
        capture_result: dict[str, Any],
        *,
        reason: str,
    ) -> Path:
        note_path = self._notes_dir() / f"desktop_observation_{self._timestamp_slug()}.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        capture = capture_result.get("capture") or {}
        content = "\n".join(
            [
                "# Desktop Observation",
                "",
                f"- created_at: {self._now().isoformat()}",
                f"- reason: {reason}",
                f"- screenshot: {capture.get('path') or ''}",
                "",
                "## Activity Summary",
                str(analysis.get("activity_summary") or ""),
                "",
                "## Spoken Text Requested",
                str(analysis.get("spoken_text") or ""),
                "",
                "## Note",
                str(analysis.get("note") or ""),
                "",
                "## Raw Analysis",
                "```json",
                json.dumps(analysis.get("raw") or analysis, indent=2, ensure_ascii=True),
                "```",
            ]
        )
        note_path.write_text(content, encoding="utf-8")
        return note_path

    def _shorten_for_speech(self, text: str, limit: int) -> str:
        normalized = " ".join(str(text or "").split())
        if len(normalized) <= limit:
            return normalized
        trimmed = normalized[: max(20, limit - 3)].rstrip()
        sentence_break = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"))
        if sentence_break > limit // 2:
            return trimmed[: sentence_break + 1]
        return trimmed.rstrip(",;:") + "..."

    def _image_data_uri(self, path_value: Any) -> Optional[str]:
        if not self.config.vision_enabled:
            return None
        path = self._vision_image_path(Path(str(path_value or "")))
        if not path.exists() or not path.is_file():
            return None
        try:
            if path.stat().st_size > self.config.max_image_bytes:
                return None
            payload = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            return None
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        return f"data:{media_type};base64,{payload}"

    def _vision_image_path(self, path: Path) -> Path:
        if not path.exists() or not path.is_file():
            return path
        try:
            if path.stat().st_size <= self.config.max_image_bytes:
                return path
        except OSError:
            return path
        try:
            from PIL import Image
        except Exception:
            return path
        target = self._vision_dir() / f"{path.stem}_vision.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(path) as image:
                image.thumbnail(
                    (self.config.vision_max_dimension, self.config.vision_max_dimension),
                    Image.Resampling.LANCZOS,
                )
                image.convert("RGB").save(
                    target,
                    format="JPEG",
                    quality=self.config.vision_jpeg_quality,
                    optimize=True,
                )
        except Exception:
            return path
        return target

    def _voice_path(self, voice_meta: Any) -> Optional[Path]:
        if isinstance(voice_meta, dict):
            raw = voice_meta.get("path")
            if raw:
                return Path(str(raw))
            audio = voice_meta.get("audio")
            if isinstance(audio, dict) and audio.get("path"):
                return Path(str(audio["path"]))
        return None

    def _observation_due(self) -> bool:
        last = self._last_event_datetime("observed")
        if last is None:
            return True
        return (self._now() - last).total_seconds() >= self.config.capture_interval_seconds

    def _speech_due(self) -> bool:
        last = self._last_event_datetime("spoken")
        if last is None:
            return True
        return (self._now() - last).total_seconds() >= self.config.min_speech_interval_seconds

    def _capture_backend_available(self) -> bool:
        if self._capture_provider is not None:
            return True
        from .capture import choose_screenshot_backend

        return choose_screenshot_backend(self.config.capture_backend) is not None

    def _session_id(self) -> str:
        if self.config.session_id:
            return self.config.session_id
        cfg = getattr(getattr(self.runtime, "ctx", None), "config", None)
        return str(getattr(cfg, "session_id", None) or "default")

    def _load_config(self) -> DesktopContextConfig:
        path = self._config_path()
        if not path.exists():
            return DesktopContextConfig()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                migrated = False
                if (
                    "proactive_video_commentary_enabled" not in payload
                    and int(payload.get("capture_interval_seconds") or 300) == 300
                ):
                    payload["capture_interval_seconds"] = 60
                    migrated = True
                declared_task = str(payload.get("declared_task") or "")
                if (
                    not payload.get("media_commentary_mode_enabled")
                    and self._looks_like_media_commentary_request(declared_task)
                ):
                    payload["media_commentary_mode_enabled"] = True
                    payload["media_commentary_requested_at"] = payload.get("declared_task_updated_at") or self._now().isoformat()
                    payload["media_commentary_source"] = "declared_task_migration"
                    payload["media_commentary_request"] = declared_task[:1000]
                    payload["media_commentary_request_source"] = "declared_task_migration"
                    payload["media_commentary_request_text"] = declared_task[:1000]
                    payload["live_transcription_enabled"] = True
                    migrated = True
                config = DesktopContextConfig(**payload)
                if migrated:
                    path.write_text(
                        json.dumps(config.model_dump(), indent=2, sort_keys=True, ensure_ascii=True),
                        encoding="utf-8",
                    )
                return config
        except Exception:
            pass
        return DesktopContextConfig()

    def _save_config(self) -> None:
        path = self._config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.config.model_dump(), indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )

    def _event(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = {
            "type": event_type,
            "created_at": self._now().isoformat(),
            **payload,
        }
        self._events_path().parent.mkdir(parents=True, exist_ok=True)
        with self._events_path().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True))
            handle.write("\n")
        return event

    def _list_events(self, *, limit: int = 200) -> list[dict[str, Any]]:
        path = self._events_path()
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                events.append(parsed)
        return events[-max(1, int(limit)) :]

    def _last_event_at(self, event_type: str) -> Optional[str]:
        dt = self._last_event_datetime(event_type)
        return dt.isoformat() if dt is not None else None

    def _last_event_datetime(self, event_type: str) -> Optional[datetime]:
        for event in reversed(self._list_events(limit=500)):
            if event.get("type") != event_type:
                continue
            raw = str(event.get("created_at") or "")
            try:
                parsed = datetime.fromisoformat(raw)
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        return None

    async def _call_maybe_async(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def _capture_to_dict(self, capture: DesktopCapture) -> dict[str, Any]:
        return {
            "success": capture.success,
            "path": str(capture.path),
            "backend": capture.backend,
            "media_type": capture.media_type,
            "width": capture.width,
            "height": capture.height,
            "error": capture.error,
        }

    def _image_dimensions(self, path: Path) -> tuple[Optional[int], Optional[int]]:
        try:
            from PIL import Image

            with Image.open(path) as image:
                return int(image.size[0]), int(image.size[1])
        except Exception:
            return None, None

    def _timestamp_slug(self) -> str:
        return self._now().strftime("%Y%m%dT%H%M%S%fZ")

    def _now(self) -> datetime:
        value = self._time_source()
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _config_path(self) -> Path:
        return self.root / "config.json"

    def _events_path(self) -> Path:
        return self.root / "events.jsonl"

    def _media_state_path(self) -> Path:
        return self.root / "media_state.json"

    def _screenshots_dir(self) -> Path:
        path = self.root / "screenshots"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _notes_dir(self) -> Path:
        path = self.root / "notes"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _audio_dir(self) -> Path:
        path = self.root / "audio"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _vision_dir(self) -> Path:
        path = self.root / "vision"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _transcripts_dir(self) -> Path:
        path = self.root / "transcripts"
        path.mkdir(parents=True, exist_ok=True)
        return path


def _current_default_audio_sink() -> Optional[str]:
    """Return the current PulseAudio/PipeWire default sink name when available."""

    pactl = shutil.which("pactl")
    if pactl:
        try:
            completed = subprocess.run(
                [pactl, "get-default-sink"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if completed.returncode == 0:
                sink = completed.stdout.strip()
                if sink:
                    return sink
        except Exception:
            pass
    wpctl = shutil.which("wpctl")
    if wpctl:
        try:
            completed = subprocess.run(
                [wpctl, "inspect", "@DEFAULT_AUDIO_SINK@"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if completed.returncode == 0:
                for line in completed.stdout.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("node.name ="):
                        sink = stripped.split("=", 1)[1].strip().strip('"')
                        if sink:
                            return sink
        except Exception:
            pass
    return None


def play_audio_file(path: Path) -> dict[str, Any]:
    """Play a generated TTS file and return after playback finishes."""

    audio_path = Path(path)
    audio_sink = _current_default_audio_sink()
    mpv_args = ["--no-terminal", "--really-quiet"]
    if audio_sink:
        mpv_args.extend(["--ao=pulse", f"--audio-device=pulse/{audio_sink}"])
    mpv_args.append(str(audio_path))
    ffplay_env = {"PULSE_SINK": audio_sink} if audio_sink else {}
    candidates = [
        ("mpv", mpv_args, {}),
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)], ffplay_env),
    ]
    for name, args, env_update in candidates:
        executable = shutil.which(name)
        if not executable:
            continue
        env = None
        if env_update:
            env = os.environ.copy()
            env.update(env_update)
        try:
            completed = subprocess.run(
                [executable, *args],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                timeout=300,
                check=False,
            )
        except Exception as exc:
            return {
                "played": False,
                "player": name,
                "path": str(audio_path),
                "audio_sink": audio_sink,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "played": completed.returncode == 0,
            "player": name,
            "path": str(audio_path),
            "audio_sink": audio_sink,
            "returncode": completed.returncode,
        }
    return {"played": False, "reason": "no_audio_player", "path": str(audio_path), "audio_sink": audio_sink}
