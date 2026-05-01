"""Runtime service for desktop screenshot context and spoken body-double nudges."""

from __future__ import annotations

import base64
import inspect
import json
import mimetypes
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, field_validator

from opencas.context import MessageRole
from opencas.memory import EpisodeKind

from .capture import DesktopCapture, capture_desktop_image, run_tesseract_ocr


class DesktopContextConfig(BaseModel):
    """Operator-controlled settings for desktop observation."""

    enabled: bool = False
    capture_interval_seconds: int = Field(default=300, ge=0)
    min_speech_interval_seconds: int = Field(default=60, ge=0)
    tts_enabled: bool = True
    play_audio: bool = True
    vision_enabled: bool = True
    ocr_enabled: bool = True
    capture_backend: str = "auto"
    vision_model: Optional[str] = None
    session_id: Optional[str] = None
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

    @field_validator("vision_model", "session_id", mode="before")
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
        time_source: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.runtime = runtime
        self.root = Path(state_dir).expanduser() / "desktop_context"
        self.root.mkdir(parents=True, exist_ok=True)
        self._capture_provider = capture_provider
        self._ocr_provider = ocr_provider
        self._speech_synthesizer = speech_synthesizer
        self._audio_player = audio_player or play_audio_file
        self._time_source = time_source or (lambda: datetime.now(timezone.utc))
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
            "event_count": len(events),
            "last_event": events[-1] if events else None,
            "last_observed_at": self._last_event_at("observed"),
            "last_spoken_at": self._last_event_at("spoken"),
        }

    def configure(self, **updates: Any) -> dict[str, Any]:
        """Update persisted desktop-context settings."""

        allowed = set(DesktopContextConfig.model_fields)
        clean_updates = {
            key: value
            for key, value in updates.items()
            if key in allowed and value is not None
        }
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
    ) -> dict[str, Any]:
        """Capture, analyze, persist context, and optionally speak a short nudge."""

        if not self.config.enabled and not force:
            return {"status": "skipped", "reason": "disabled"}
        if not force and not self._observation_due():
            return {"status": "skipped", "reason": "not_due"}

        capture_result = await self.capture_once(force=True)
        if capture_result.get("status") != "captured":
            return capture_result

        analysis = await self._analyze_capture(capture_result, reason=reason)
        context_text = self._build_context_text(capture_result, analysis, reason=reason)
        await self._persist_context(context_text, capture_result, analysis)

        speech: Optional[dict[str, Any]] = None
        should_speak = bool(analysis.get("should_speak"))
        speech_requested = self.config.tts_enabled if speak is None else bool(speak)
        if should_speak and speech_requested:
            speech = await self._speak_analysis(analysis, capture_result, reason=reason)

        payload = {
            "status": "observed",
            "reason": reason,
            "capture": capture_result.get("capture"),
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

    async def run_once(self) -> dict[str, Any]:
        """Scheduler entrypoint for enabled body-double observations."""

        return await self.observe_once(force=False, reason="scheduled_body_double")

    async def speak_text(self, text: str, *, reason: str = "manual") -> dict[str, Any]:
        """Speak a direct natural-language message through the configured TTS path."""

        analysis = {
            "should_speak": True,
            "activity_summary": "Direct desktop-context speech request.",
            "spoken_text": text,
            "reason": reason,
        }
        return await self._speak_analysis(analysis, {"capture": {}}, reason=reason)

    async def _capture(self, target: Path) -> DesktopCapture:
        provider = self._capture_provider
        if provider is not None:
            return await self._call_maybe_async(provider, target)
        return await self._call_maybe_async(
            capture_desktop_image,
            target,
            backend=self.config.capture_backend,
        )

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
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the OpenCAS agent reviewing a private screenshot of the operator's active desktop "
                    "for an explicitly enabled body-double collaboration skill. Decide whether a "
                    "short spoken interruption is useful. Do not read code, logs, stack traces, or "
                    "long technical text aloud. For those, summarize briefly and refer to a file. "
                    "Return strict JSON with keys: should_speak boolean, activity_summary string, "
                    "reason string, spoken_text string, note string."
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

    def _analysis_prompt(self, capture_result: dict[str, Any], *, reason: str) -> str:
        capture = capture_result.get("capture") or {}
        ocr_text = str(capture_result.get("ocr_text") or "").strip()
        return "\n".join(
            [
                "Review this desktop observation.",
                f"Reason: {reason}",
                f"Screenshot path: {capture.get('path') or ''}",
                f"Capture backend: {capture.get('backend') or ''}",
                "Operator goal: body-double support for productive ADHD-friendly work.",
                "Speak only if a timely, concise collaborator comment would help.",
                "OCR text excerpt:",
                ocr_text or "(no OCR text)",
            ]
        )

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
            "fallback": True,
        }

    def _normalize_analysis(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "should_speak": bool(raw.get("should_speak")),
            "activity_summary": str(raw.get("activity_summary") or raw.get("summary") or "").strip(),
            "reason": str(raw.get("reason") or "").strip(),
            "spoken_text": str(raw.get("spoken_text") or raw.get("message") or "").strip(),
            "note": str(raw.get("note") or "").strip(),
            "raw": raw,
        }

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
    ) -> None:
        store = getattr(getattr(self.runtime, "ctx", None), "context_store", None)
        session_id = self._session_id()
        meta = {
            "source": "desktop_context",
            "capture": capture_result.get("capture"),
            "analysis": analysis,
        }
        if store is not None and hasattr(store, "append"):
            await store.append(session_id, MessageRole.SYSTEM, content, meta=meta)
        record_episode = getattr(self.runtime, "_record_episode", None)
        if callable(record_episode):
            try:
                await record_episode(
                    content,
                    EpisodeKind.OBSERVATION,
                    session_id=session_id,
                    role="desktop_context",
                )
            except Exception:
                pass

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
            f"- activity: {analysis.get('activity_summary') or '(not summarized)'}",
            f"- comment decision: {'speak' if analysis.get('should_speak') else 'hold'}",
        ]
        note = str(analysis.get("note") or "").strip()
        if note:
            lines.append(f"- note: {note}")
        if ocr_text:
            lines.append("- OCR excerpt:")
            lines.append(ocr_text[: self.config.max_ocr_chars])
        return "\n".join(lines)

    async def _speak_analysis(
        self,
        analysis: dict[str, Any],
        capture_result: dict[str, Any],
        *,
        reason: str,
    ) -> dict[str, Any]:
        if not self.config.tts_enabled:
            return {"status": "skipped", "reason": "tts_disabled"}
        if not self._speech_due():
            return {"status": "skipped", "reason": "speech_not_due"}
        prepared = self._prepare_spoken_text(analysis, capture_result, reason=reason)
        spoken_text = prepared["spoken_text"]
        if not spoken_text:
            return {"status": "skipped", "reason": "empty_spoken_text"}

        try:
            synth = self._speech_synthesizer or self._default_speech_synthesizer
            voice_meta = await self._call_maybe_async(synth, spoken_text)
        except Exception as exc:
            return {"status": "failed", "reason": f"tts_failed:{type(exc).__name__}", "error": str(exc)}

        playback: Optional[dict[str, Any]] = None
        audio_path = self._voice_path(voice_meta)
        if self.config.play_audio and audio_path is not None:
            try:
                played = await self._call_maybe_async(self._audio_player, audio_path)
                playback = played if isinstance(played, dict) else {"played": bool(played), "path": str(audio_path)}
            except Exception as exc:
                playback = {"played": False, "error": str(exc), "path": str(audio_path)}

        event = self._event(
            "spoken",
            {
                "reason": reason,
                "chars": len(spoken_text),
                "voice": voice_meta,
                "playback": playback,
                "redirected_to_note": prepared.get("redirected_to_note", False),
            },
        )
        return {
            "status": "spoken",
            "spoken_text": spoken_text,
            "voice": voice_meta,
            "playback": playback,
            "event": event,
            **{key: value for key, value in prepared.items() if key != "spoken_text"},
        }

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
        if self._speech_should_be_note(raw):
            note_path = self._write_note_file(analysis, capture_result, reason=reason)
            summary = str(analysis.get("activity_summary") or "I found something worth reading.").strip()
            summary = self._shorten_for_speech(summary, 180)
            return {
                "spoken_text": f"I wrote the desktop observation details to {note_path}. Short version: {summary}",
                "redirected_to_note": True,
                "note_path": str(note_path),
            }
        return {
            "spoken_text": self._shorten_for_speech(raw, self.config.max_spoken_chars),
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
                return DesktopContextConfig(**payload)
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


def play_audio_file(path: Path) -> dict[str, Any]:
    """Start local playback for a generated TTS file without blocking the runtime."""

    audio_path = Path(path)
    candidates = [
        ("mpv", ["--no-terminal", "--really-quiet", str(audio_path)]),
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)]),
    ]
    for name, args in candidates:
        executable = shutil.which(name)
        if not executable:
            continue
        subprocess.Popen(
            [executable, *args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {"played": True, "player": name, "path": str(audio_path)}
    return {"played": False, "reason": "no_audio_player", "path": str(audio_path)}
