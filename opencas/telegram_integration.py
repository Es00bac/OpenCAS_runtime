"""Telegram long-poll integration for OpenCAS."""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from opencas.telemetry import EventKind, Tracer


TELEGRAM_TEXT_LIMIT = 4096


class TelegramPairingRequest(BaseModel):
    """Pending or approved Telegram pairing request."""

    code: str
    user_id: str
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    requested_at: float = Field(default_factory=time.time)
    approved_at: Optional[float] = None

    @property
    def approved(self) -> bool:
        return self.approved_at is not None


class TelegramPairingStore:
    """Small JSON-backed store for Telegram pairing approvals."""

    def __init__(self, path: Path | str, ttl_seconds: int = 3600) -> None:
        self.path = Path(path)
        self.ttl_seconds = max(60, int(ttl_seconds))
        self._lock = asyncio.Lock()

    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            state = self._load_state()
            self._prune_expired(state)
            self._save_state(state)
            return self._serialize_state(state)

    async def is_authorized(self, user_id: str, allow_from: List[str]) -> bool:
        async with self._lock:
            state = self._load_state()
            self._prune_expired(state)
            if user_id in set(state.get("approved_user_ids", [])):
                self._save_state(state)
                return True
            return user_id in {str(item).strip() for item in allow_from if str(item).strip()}

    async def create_or_get_request(
        self,
        user_id: str,
        *,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> TelegramPairingRequest:
        async with self._lock:
            state = self._load_state()
            self._prune_expired(state)
            if user_id in set(state.get("approved_user_ids", [])):
                request = TelegramPairingRequest(
                    code=self._generate_code(),
                    user_id=user_id,
                    username=username,
                    first_name=first_name,
                    last_name=last_name,
                    approved_at=time.time(),
                )
                return request
            for item in state.get("requests", []):
                if item.get("user_id") == user_id and item.get("approved_at") is None:
                    request = TelegramPairingRequest.model_validate(item)
                    self._save_state(state)
                    return request
            request = TelegramPairingRequest(
                code=self._generate_code(),
                user_id=user_id,
                username=username,
                first_name=first_name,
                last_name=last_name,
            )
            state.setdefault("requests", []).append(request.model_dump(mode="json"))
            self._save_state(state)
            return request

    async def approve(self, code: str) -> Optional[TelegramPairingRequest]:
        normalized = code.strip().upper()
        if not normalized:
            return None
        async with self._lock:
            state = self._load_state()
            self._prune_expired(state)
            for item in state.get("requests", []):
                if str(item.get("code", "")).upper() != normalized:
                    continue
                item["approved_at"] = time.time()
                approved_ids = set(state.get("approved_user_ids", []))
                approved_ids.add(str(item.get("user_id", "")))
                state["approved_user_ids"] = sorted(approved_ids)
                self._save_state(state)
                return TelegramPairingRequest.model_validate(item)
            self._save_state(state)
            return None

    def _load_state(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"approved_user_ids": [], "requests": []}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                raw.setdefault("approved_user_ids", [])
                raw.setdefault("requests", [])
                return raw
        except Exception:
            pass
        return {"approved_user_ids": [], "requests": []}

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")

    def _prune_expired(self, state: Dict[str, Any]) -> None:
        cutoff = time.time() - self.ttl_seconds
        kept: List[Dict[str, Any]] = []
        for item in state.get("requests", []):
            approved_at = item.get("approved_at")
            if approved_at is not None:
                kept.append(item)
                continue
            requested_at = float(item.get("requested_at", 0.0) or 0.0)
            if requested_at >= cutoff:
                kept.append(item)
        state["requests"] = kept

    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        requests = [TelegramPairingRequest.model_validate(item) for item in state.get("requests", [])]
        pending = [item.model_dump(mode="json") for item in requests if not item.approved]
        approved_requests = [item.model_dump(mode="json") for item in requests if item.approved]
        return {
            "approved_user_ids": sorted({str(item) for item in state.get("approved_user_ids", []) if str(item)}),
            "pending_requests": pending,
            "approved_requests": approved_requests,
        }

    @staticmethod
    def _generate_code() -> str:
        return secrets.token_hex(4).upper()


class TelegramApiClient:
    """Very small Telegram Bot API client backed by httpx."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://api.telegram.org",
        timeout: float = 40.0,
        transport: Any = None,
    ) -> None:
        self.token = token
        self.base_url = f"{base_url.rstrip('/')}/bot{token}"
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def close(self) -> None:
        await self._client.aclose()

    async def get_me(self) -> Dict[str, Any]:
        return await self._request("getMe")

    async def get_updates(self, *, offset: int, timeout: int = 30) -> List[Dict[str, Any]]:
        result = await self._request("getUpdates", json_payload={"offset": offset, "timeout": timeout})
        return result if isinstance(result, list) else []

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id
        return await self._request("sendMessage", json_payload=payload)

    async def edit_message_text(self, chat_id: int, message_id: int, text: str) -> Dict[str, Any]:
        return await self._request(
            "editMessageText",
            json_payload={"chat_id": chat_id, "message_id": message_id, "text": text},
        )

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> Dict[str, Any]:
        return await self._request(
            "sendChatAction",
            json_payload={"chat_id": chat_id, "action": action},
        )

    async def _request(self, method: str, *, json_payload: Optional[Dict[str, Any]] = None) -> Any:
        response = await self._client.post(f"{self.base_url}/{method}", json=json_payload or {})
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok", False):
            raise RuntimeError(str(payload.get("description", f"Telegram API error in {method}")))
        return payload.get("result")


class TelegramBotService:
    """Long-poll Telegram bot service that routes DMs into OpenCAS conversations."""

    def __init__(
        self,
        runtime: Any,
        *,
        enabled: bool,
        token: str,
        state_dir: Path | str,
        dm_policy: str = "pairing",
        allow_from: Optional[List[str]] = None,
        poll_interval_seconds: float = 1.0,
        pairing_ttl_seconds: int = 3600,
        api_base_url: str = "https://api.telegram.org",
        tracer: Optional[Tracer] = None,
        client: Optional[TelegramApiClient] = None,
    ) -> None:
        self.runtime = runtime
        self.enabled = bool(enabled)
        self.token = token.strip()
        self.dm_policy = dm_policy
        self.allow_from = [str(item).strip() for item in (allow_from or []) if str(item).strip()]
        self.poll_interval_seconds = max(0.2, float(poll_interval_seconds))
        self.tracer = tracer
        telegram_state_dir = Path(state_dir).expanduser() / "telegram"
        self._offset_path = telegram_state_dir / "update_offset.json"
        self.pairing_store = TelegramPairingStore(
            telegram_state_dir / "pairings.json",
            ttl_seconds=pairing_ttl_seconds,
        )
        self.client = client or TelegramApiClient(self.token, base_url=api_base_url)
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._last_update_id = self._load_offset()
        self._last_error: Optional[str] = None
        self._bot_info: Optional[Dict[str, Any]] = None
        self._chat_locks: Dict[int, asyncio.Lock] = {}

    @property
    def configured(self) -> bool:
        return bool(self.token)

    async def start(self) -> None:
        if not self.enabled or not self.configured or self._task is not None:
            return
        self._stop_event.clear()
        try:
            self._bot_info = await self.client.get_me()
        except Exception as exc:
            self._last_error = str(exc)
            self._trace("telegram_start_failed", {"error": str(exc)}, kind=EventKind.ERROR)
            raise
        self._task = asyncio.create_task(self._poll_loop())
        self._trace("telegram_started", {"bot_username": self._bot_info.get("username")})

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self.client.close()
        self._trace("telegram_stopped", {})

    async def status(self) -> Dict[str, Any]:
        pairings = await self.pairing_store.snapshot()
        return {
            "enabled": self.enabled,
            "configured": self.configured,
            "token_configured": self.configured,
            "running": self._task is not None and not self._task.done(),
            "dm_policy": self.dm_policy,
            "allow_from": sorted(set(self.allow_from)),
            "bot": {
                "id": self._bot_info.get("id") if self._bot_info else None,
                "username": self._bot_info.get("username") if self._bot_info else None,
                "first_name": self._bot_info.get("first_name") if self._bot_info else None,
                "link": (
                    f"https://t.me/{self._bot_info.get('username')}"
                    if self._bot_info and self._bot_info.get("username")
                    else None
                ),
            },
            "last_update_id": self._last_update_id,
            "last_error": self._last_error,
            "pairings": pairings,
        }

    async def approve_pairing(self, code: str) -> Optional[TelegramPairingRequest]:
        request = await self.pairing_store.approve(code)
        if request is None:
            return None
        try:
            await self.client.send_message(
                int(request.user_id),
                "Pairing approved. You can chat with OpenCAS now.",
            )
        except Exception as exc:
            self._last_error = str(exc)
            self._trace("telegram_pairing_notify_failed", {"error": str(exc), "user_id": request.user_id}, kind=EventKind.ERROR)
        self._trace("telegram_pairing_approved", {"user_id": request.user_id, "code": request.code})
        return request

    async def handle_update(self, update: Dict[str, Any]) -> None:
        message = update.get("message") or update.get("edited_message")
        if not isinstance(message, dict):
            return
        chat = message.get("chat") or {}
        sender = message.get("from") or {}
        chat_id = chat.get("id")
        user_id = sender.get("id")
        if not isinstance(chat_id, int) or not isinstance(user_id, int):
            return
        if sender.get("is_bot"):
            return

        text = message.get("text") or message.get("caption")
        if not isinstance(text, str) or not text.strip():
            if chat.get("type") == "private":
                await self.client.send_message(
                    chat_id,
                    "Telegram media and non-text messages are not supported yet in OpenCAS.",
                    reply_to_message_id=message.get("message_id"),
                )
            return

        authorized = await self._is_authorized(chat.get("type"), str(user_id))
        if not authorized:
            if chat.get("type") == "private" and self.dm_policy == "pairing":
                request = await self.pairing_store.create_or_get_request(
                    str(user_id),
                    username=sender.get("username"),
                    first_name=sender.get("first_name"),
                    last_name=sender.get("last_name"),
                )
                reply = (
                    "This Telegram account is not paired with OpenCAS yet.\n"
                    f"Your Telegram user id: {user_id}\n"
                    f"Pairing code: {request.code}\n"
                    "Approve it in the OpenCAS dashboard Telegram panel."
                )
                await self.client.send_message(chat_id, reply, reply_to_message_id=message.get("message_id"))
                self._trace("telegram_pairing_requested", {"user_id": user_id, "code": request.code})
            return

        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            reply_to_message_id = message.get("message_id")
            session_id = f"telegram:{chat.get('type', 'chat')}:{chat_id}"
            stop_typing = asyncio.Event()
            try:
                await self.client.send_chat_action(chat_id, "typing")
            except Exception as exc:
                self._last_error = str(exc)
                self._trace(
                    "telegram_typing_failed",
                    {"chat_id": chat_id, "error": str(exc)},
                    kind=EventKind.ERROR,
                )
            typing_task = asyncio.create_task(self._typing_keepalive(chat_id, stop_typing))
            placeholder_task = asyncio.create_task(
                self._delayed_placeholder(chat_id, reply_to_message_id, stop_typing)
            )
            try:
                self._trace(
                    "telegram_message_received",
                    {"chat_id": chat_id, "user_id": user_id, "session_id": session_id},
                )
                response = await self.runtime.converse(text.strip(), session_id=session_id)
            except Exception as exc:
                response = f"[Error: {exc}]"
                self._last_error = str(exc)
                self._trace(
                    "telegram_message_failed",
                    {"chat_id": chat_id, "user_id": user_id, "error": str(exc)},
                    kind=EventKind.ERROR,
                )
            finally:
                stop_typing.set()
                await typing_task

            placeholder_message_id: Optional[int] = None
            if placeholder_task.done():
                try:
                    placeholder_message_id = placeholder_task.result()
                except Exception:
                    placeholder_message_id = None
            else:
                placeholder_task.cancel()
                try:
                    await placeholder_task
                except asyncio.CancelledError:
                    pass

            await self._deliver_response(
                chat_id,
                response,
                reply_to_message_id=reply_to_message_id,
                placeholder_message_id=placeholder_message_id,
            )

    async def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                updates = await self.client.get_updates(offset=self._last_update_id, timeout=30)
                for update in updates:
                    update_id = int(update.get("update_id", 0) or 0)
                    await self.handle_update(update)
                    self._last_update_id = max(self._last_update_id, update_id + 1)
                    self._save_offset(self._last_update_id)
                self._last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = str(exc)
                self._trace("telegram_poll_error", {"error": str(exc)}, kind=EventKind.ERROR)
                await asyncio.sleep(self.poll_interval_seconds)

    async def _is_authorized(self, chat_type: Optional[str], user_id: str) -> bool:
        if self.dm_policy == "disabled":
            return False
        if self.dm_policy == "open":
            return True
        if self.dm_policy == "allowlist":
            return user_id in set(self.allow_from)
        if chat_type != "private":
            return await self.pairing_store.is_authorized(user_id, self.allow_from)
        return await self.pairing_store.is_authorized(user_id, self.allow_from)

    async def _typing_keepalive(self, chat_id: int, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await self.client.send_chat_action(chat_id, "typing")
            except Exception as exc:
                self._last_error = str(exc)
                self._trace(
                    "telegram_typing_failed",
                    {"chat_id": chat_id, "error": str(exc)},
                    kind=EventKind.ERROR,
                )
                return
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=3.5)
            except asyncio.TimeoutError:
                continue

    async def _delayed_placeholder(
        self,
        chat_id: int,
        reply_to_message_id: Optional[int],
        stop_event: asyncio.Event,
    ) -> Optional[int]:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
            return None
        except asyncio.TimeoutError:
            sent = await self.client.send_message(
                chat_id,
                "Thinking…",
                reply_to_message_id=reply_to_message_id,
            )
            message_id = sent.get("message_id")
            return int(message_id) if isinstance(message_id, int) else None

    async def _deliver_response(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: Optional[int],
        placeholder_message_id: Optional[int],
    ) -> None:
        chunks = _chunk_telegram_text(text)
        if not chunks:
            return
        if placeholder_message_id is not None:
            await self.client.edit_message_text(chat_id, placeholder_message_id, chunks[0])
            remaining = chunks[1:]
        else:
            await self.client.send_message(
                chat_id,
                chunks[0],
                reply_to_message_id=reply_to_message_id,
            )
            remaining = chunks[1:]
        for chunk in remaining:
            await self.client.send_message(chat_id, chunk)

    def _trace(self, message: str, payload: Dict[str, Any], *, kind: EventKind = EventKind.TOOL_CALL) -> None:
        if self.tracer is None:
            return
        self.tracer.log(kind, message, payload)

    def _load_offset(self) -> int:
        if not self._offset_path.exists():
            return 0
        try:
            payload = json.loads(self._offset_path.read_text(encoding="utf-8"))
            return int(payload.get("offset", 0) or 0)
        except Exception:
            return 0

    def _save_offset(self, offset: int) -> None:
        self._offset_path.parent.mkdir(parents=True, exist_ok=True)
        self._offset_path.write_text(json.dumps({"offset": offset}, ensure_ascii=True), encoding="utf-8")


def _chunk_telegram_text(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> List[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return []
    if len(normalized) <= limit:
        return [normalized]

    chunks: List[str] = []
    remaining = normalized
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunk = remaining[:split_at].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    return chunks
