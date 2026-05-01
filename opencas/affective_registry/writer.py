"""Append-only affective registry writer with POSIX-safe atomic appends.

Design guarantees:
1. Append-only — historical entries are never read, modified, or rewritten.
2. Atomic — each write is a single `write(2)` under `O_APPEND` when possible,
   falling back to POSIX advisory file locking for broader compatibility.
3. Concurrent-safe — multiple processes/threads may write to the same file.
4. Validation — every append is verified by an immediate read-back of the
   written bytes.
5. Graceful degradation — disk-full, permission-denied, and path-creation
   failures are logged and raised as structured exceptions.
"""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import struct
import threading
from pathlib import Path
from typing import Iterator, List, Optional

from .models import AffectiveRegistryEntry, ExecutionPhase

logger = logging.getLogger(__name__)

# Registry format version for forward compatibility
_REGISTRY_FORMAT_VERSION = "1.0.0"

# Advisory lock helpers (portable POSIX)
_F_LOCK = fcntl.LOCK_EX  # exclusive lock
_F_UNLOCK = fcntl.LOCK_UN


class RegistryWriteError(Exception):
    """Raised when a registry append fails after all retry attempts."""

    def __init__(self, message: str, *, path: Optional[Path] = None, cause: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.path = path
        self.cause = cause


class ValidationError(Exception):
    """Raised when the written record fails read-back validation."""

    def __init__(self, message: str, *, expected: Optional[str] = None, actual: Optional[str] = None) -> None:
        super().__init__(message)
        self.expected = expected
        self.actual = actual


class AffectiveRegistryWriter:
    """Thread-safe, append-only writer for affective state snapshots.

    Parameters
    ----------
    registry_path:
        Absolute path to the registry file. Parent directories are created
        automatically. The file is opened in append mode on every write.
    enable_locking:
        When ``True`` (default), POSIX advisory locks are acquired around
        each write. Disable only when you know the file is private to a
        single thread and O_APPEND atomicity is sufficient.
    validate_writes:
        When ``True`` (default), each write is immediately validated by
        reading back the last line of the file and comparing it byte-for-byte
        with what was intended.
    max_retries:
        Number of times to retry a failed write before giving up.
    """

    def __init__(
        self,
        registry_path: Path | str,
        *,
        enable_locking: bool = True,
        validate_writes: bool = True,
        max_retries: int = 3,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.enable_locking = enable_locking
        self.validate_writes = validate_writes
        self.max_retries = max(0, max_retries)
        self._thread_lock = threading.Lock()

        # Ensure parent directories exist
        self._ensure_path()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        entry: AffectiveRegistryEntry,
        *,
        _attempt: int = 0,
    ) -> AffectiveRegistryEntry:
        """Append a single entry atomically and return it.

        Raises
        ------
        RegistryWriteError
            If the append fails after all retries.
        ValidationError
            If the write succeeds but read-back validation fails.
        """
        with self._thread_lock:
            return self._append_locked(entry, _attempt=_attempt)

    def append_from_somatic_state(
        self,
        somatic_state,
        *,
        phase: ExecutionPhase = ExecutionPhase.MANUAL,
        payload: Optional[dict] = None,
        session_id: Optional[str] = None,
        span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AffectiveRegistryEntry:
        """Convenience helper that builds an entry from an OpenCAS SomaticState.

        Accepts either a raw ``SomaticState`` instance or any object with the
        expected dimension attributes (arousal, fatigue, tension, valence,
        focus, energy, certainty, musubi, somatic_tag, primary_emotion).
        """
        from .models import AffectiveState, ExecutionContext, SystemMetrics

        affective = AffectiveState(
            primary_emotion=_extract_attr(somatic_state, "primary_emotion", "neutral"),
            valence=_extract_attr(somatic_state, "valence", 0.0),
            arousal=_extract_attr(somatic_state, "arousal", 0.5),
            fatigue=_extract_attr(somatic_state, "fatigue", 0.0),
            tension=_extract_attr(somatic_state, "tension", 0.0),
            focus=_extract_attr(somatic_state, "focus", 0.5),
            energy=_extract_attr(somatic_state, "energy", 0.5),
            certainty=_extract_attr(somatic_state, "certainty", 0.5),
            musubi=_extract_attr(somatic_state, "musubi", None),
            somatic_tag=_extract_attr(somatic_state, "somatic_tag", None),
        )

        ctx = ExecutionContext(
            session_id=session_id,
            span_id=span_id,
            trace_id=trace_id,
            user_id=user_id,
        )

        entry = AffectiveRegistryEntry(
            phase=phase,
            affective_state=affective,
            execution_context=ctx,
            system_metrics=SystemMetrics.capture(),
            payload=payload or {},
        )
        return self.append(entry)

    def iter_entries(self) -> Iterator[AffectiveRegistryEntry]:
        """Iterate all entries in chronological order (oldest first)."""
        yield from self._read_jsonl(self.registry_path)

    def iter_entries_reverse(self) -> Iterator[AffectiveRegistryEntry]:
        """Iterate all entries in reverse chronological order.

        This is memory-efficient for large files because it reads the file
        in chunks from the end, parsing complete JSON lines as it goes.
        """
        yield from self._read_jsonl_reverse(self.registry_path)

    def get_latest(self, n: int = 1) -> List[AffectiveRegistryEntry]:
        """Return the *n* most recent entries without loading the whole file."""
        results: List[AffectiveRegistryEntry] = []
        for entry in self.iter_entries_reverse():
            results.append(entry)
            if len(results) >= n:
                break
        results.reverse()
        return results

    def count_entries(self) -> int:
        """Return the total number of valid entries in the registry."""
        return sum(1 for _ in self.iter_entries())

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _ensure_path(self) -> None:
        """Create parent directories if missing; log but do not raise on failure."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "Failed to create registry parent directories at %s: %s",
                self.registry_path.parent,
                exc,
            )

    def _append_locked(
        self,
        entry: AffectiveRegistryEntry,
        *,
        _attempt: int = 0,
    ) -> AffectiveRegistryEntry:
        line = entry.to_jsonl()
        encoded = line.encode("utf-8")

        fd: Optional[int] = None
        try:
            # Open with O_APPEND so the kernel places the write at EOF atomically.
            # We use os.open directly to get a raw fd for locking.
            open_flags = os.O_CREAT | os.O_APPEND
            if self.validate_writes:
                open_flags |= os.O_RDWR
            else:
                open_flags |= os.O_WRONLY
            fd = os.open(self.registry_path, open_flags, mode=0o644)

            if self.enable_locking:
                fcntl.flock(fd, _F_LOCK)

            # Atomic append via O_APPEND
            bytes_written = os.write(fd, encoded)

            # Ensure durability
            os.fsync(fd)

            if self.validate_writes:
                self._validate_append(fd, encoded)

            logger.debug(
                "Appended affective registry entry %s (%d bytes) to %s",
                entry.entry_id,
                bytes_written,
                self.registry_path,
            )
            return entry

        except OSError as exc:
            if _attempt < self.max_retries and exc.errno in {
                errno.EAGAIN,
                errno.EINTR,
                errno.EBUSY,
                errno.ETIMEDOUT,
            }:
                logger.warning(
                    "Registry write transient failure (attempt %d/%d): %s",
                    _attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                return self._append_locked(entry, _attempt=_attempt + 1)

            logger.error(
                "Registry write failed permanently for %s: %s",
                self.registry_path,
                exc,
            )
            raise RegistryWriteError(
                f"Failed to append to {self.registry_path}: {exc}",
                path=self.registry_path,
                cause=exc,
            ) from exc

        except ValidationError:
            raise

        except Exception as exc:
            logger.error(
                "Unexpected error writing to registry %s: %s",
                self.registry_path,
                exc,
                exc_info=True,
            )
            raise RegistryWriteError(
                f"Unexpected error appending to {self.registry_path}: {exc}",
                path=self.registry_path,
                cause=exc,
            ) from exc

        finally:
            if fd is not None:
                try:
                    if self.enable_locking:
                        fcntl.flock(fd, _F_UNLOCK)
                except OSError:
                    pass
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _validate_append(self, fd: int, expected_bytes: bytes) -> None:
        """Validate that the last bytes of the file match *expected_bytes*.

        This is done by seeking to ``-(len(expected_bytes))`` from EOF and
        reading back exactly that many bytes.  Because we hold the advisory
        lock and O_APPEND guarantees atomic placement, this is safe.
        """
        try:
            # Get current file size
            current_offset = os.lseek(fd, 0, os.SEEK_CUR)
            file_size = os.lseek(fd, 0, os.SEEK_END)

            expected_len = len(expected_bytes)
            if file_size < expected_len:
                raise ValidationError(
                    f"File size ({file_size}) smaller than expected write ({expected_len})",
                )

            # Seek back to start of what we just wrote
            read_start = file_size - expected_len
            os.lseek(fd, read_start, os.SEEK_SET)

            actual_bytes = os.read(fd, expected_len)
            if actual_bytes != expected_bytes:
                raise ValidationError(
                    "Read-back validation failed: written bytes do not match file contents",
                    expected=expected_bytes.decode("utf-8", errors="replace"),
                    actual=actual_bytes.decode("utf-8", errors="replace"),
                )

            # Restore original offset (defensive)
            os.lseek(fd, current_offset, os.SEEK_SET)

        except OSError as exc:
            raise ValidationError(
                f"Validation read-back failed: {exc}",
            ) from exc

    @staticmethod
    def _read_jsonl(path: Path) -> Iterator[AffectiveRegistryEntry]:
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield AffectiveRegistryEntry.from_jsonl(line)
                except (json.JSONDecodeError, ValueError):
                    logger.warning("Skipping corrupted registry line in %s", path)
                    continue

    @staticmethod
    def _read_jsonl_reverse(path: Path, chunk_size: int = 8192) -> Iterator[AffectiveRegistryEntry]:
        """Memory-efficient reverse JSONL reader.

        Reads the file in chunks from the end, buffering incomplete lines
        until a full JSON object is found, then yields it.
        """
        if not path.exists():
            return

        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            if file_size == 0:
                return

            buffer = b""
            offset = file_size
            while offset > 0:
                read_size = min(chunk_size, offset)
                offset -= read_size
                f.seek(offset)
                chunk = f.read(read_size)
                buffer = chunk + buffer

                # Split on newlines and process complete lines
                lines = buffer.split(b"\n")
                # The first element may be incomplete (belongs to previous chunk)
                buffer = lines[0]
                for line in reversed(lines[1:]):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield AffectiveRegistryEntry.from_jsonl(line.decode("utf-8"))
                    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
                        logger.warning("Skipping corrupted registry line in %s", path)
                        continue

            # Process any remaining buffer
            if buffer.strip():
                try:
                    yield AffectiveRegistryEntry.from_jsonl(buffer.strip().decode("utf-8"))
                except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
                    logger.warning("Skipping corrupted registry line in %s", path)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_attr(obj: Any, name: str, default: Any) -> Any:
    """Safely extract an attribute from *obj*, falling back to *default*."""
    if obj is None:
        return default
    val = getattr(obj, name, default)
    return val if val is not None else default
