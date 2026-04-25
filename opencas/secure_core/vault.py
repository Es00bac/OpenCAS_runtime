"""AES-encrypted vault for Secure Core contents.

Uses per-row AES-GCM encryption via the ``cryptography`` library to protect
sensitive content stored in a regular SQLite database. The encryption key is
derived from a machine-local salt file, avoiding cloud KMS dependency.
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vault_entries (
    entry_id TEXT PRIMARY KEY,
    entry_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    encrypted_content BLOB NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    meta TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_vault_type ON vault_entries(entry_type);
CREATE INDEX IF NOT EXISTS idx_vault_timestamp ON vault_entries(timestamp);
"""

# Default iterations for PBKDF2 key derivation
_KEY_DERIVATION_ITERATIONS = 600_000


class SecureCoreVault:
    """Encrypted vault for private episodes, memories, and beliefs.

    The vault uses AES (via Fernet) to encrypt content before storage.
    The key is derived from a salt file stored locally. If no passphrase
    is provided, a machine-local salt is used as the basis.
    """

    def __init__(
        self,
        db_path: Path | str,
        salt_path: Optional[Path | str] = None,
        passphrase: Optional[str] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self._salt_path = Path(salt_path) if salt_path else self.db_path.parent / "vault_salt"
        self._passphrase = passphrase
        self._db = None
        self._fernet: Optional[Fernet] = None
        self._unlocked = False

    async def connect(self) -> "SecureCoreVault":
        """Open the database and derive the encryption key."""
        import aiosqlite

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        self._derive_key()
        self._unlocked = True
        return self

    async def close(self) -> None:
        """Close the database and lock the vault."""
        if self._db:
            await self._db.close()
            self._db = None
        self._fernet = None
        self._unlocked = False

    @property
    def is_unlocked(self) -> bool:
        return self._unlocked and self._fernet is not None

    def vault_status(self) -> Dict[str, Any]:
        """Diagnostic status without exposing contents."""
        return {
            "unlocked": self._unlocked,
            "db_path": str(self.db_path),
            "salt_exists": self._salt_path.exists(),
            "has_key": self._fernet is not None,
        }

    def _derive_key(self) -> None:
        """Derive a Fernet key from passphrase or machine-local salt."""
        if self._passphrase:
            material = self._passphrase.encode("utf-8")
        else:
            # Use machine-local salt for key derivation
            if not self._salt_path.exists():
                salt = os.urandom(32)
                self._salt_path.parent.mkdir(parents=True, exist_ok=True)
                self._salt_path.write_bytes(salt)
            else:
                salt = self._salt_path.read_bytes()
            # Combine salt with a machine identifier for uniqueness
            material = salt + os.uname().nodename.encode("utf-8")

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"opencas_secure_core_v1",
            iterations=_KEY_DERIVATION_ITERATIONS,
        )
        key = base64.urlsafe_b64encode(kdf.derive(material))
        self._fernet = Fernet(key)

    def _encrypt(self, plaintext: str) -> bytes:
        """Encrypt a string to bytes."""
        assert self._fernet is not None
        return self._fernet.encrypt(plaintext.encode("utf-8"))

    def _decrypt(self, ciphertext: bytes) -> str:
        """Decrypt bytes to a string."""
        assert self._fernet is not None
        return self._fernet.decrypt(ciphertext).decode("utf-8")

    async def save_entry(
        self,
        entry_id: str,
        entry_type: str,
        content: str,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save an encrypted entry to the vault."""
        assert self._db is not None and self._fernet is not None
        encrypted = self._encrypt(content)
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """
            INSERT INTO vault_entries (entry_id, entry_type, timestamp, encrypted_content, tags, meta)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(entry_id) DO UPDATE SET
                entry_type = excluded.entry_type,
                timestamp = excluded.timestamp,
                encrypted_content = excluded.encrypted_content,
                tags = excluded.tags,
                meta = excluded.meta
            """,
            (
                entry_id,
                entry_type,
                now,
                encrypted,
                json.dumps(tags or []),
                json.dumps(meta or {}),
            ),
        )
        await self._db.commit()

    async def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve and decrypt a single entry. Requires vault to be unlocked."""
        if not self.is_unlocked:
            return None
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT entry_id, entry_type, timestamp, encrypted_content, tags, meta "
            "FROM vault_entries WHERE entry_id = ?",
            (entry_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    async def list_entries(
        self,
        entry_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List and decrypt entries, optionally filtered by type."""
        if not self.is_unlocked:
            return []
        assert self._db is not None
        if entry_type:
            cursor = await self._db.execute(
                "SELECT entry_id, entry_type, timestamp, encrypted_content, tags, meta "
                "FROM vault_entries WHERE entry_type = ? ORDER BY timestamp DESC LIMIT ?",
                (entry_type, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT entry_id, entry_type, timestamp, encrypted_content, tags, meta "
                "FROM vault_entries ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry from the vault."""
        assert self._db is not None
        cursor = await self._db.execute(
            "DELETE FROM vault_entries WHERE entry_id = ?", (entry_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def count_entries(self, entry_type: Optional[str] = None) -> int:
        """Count entries without decrypting them."""
        assert self._db is not None
        if entry_type:
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM vault_entries WHERE entry_type = ?",
                (entry_type,),
            )
        else:
            cursor = await self._db.execute("SELECT COUNT(*) FROM vault_entries")
        row = await cursor.fetchone()
        return row[0] if row else 0

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Decrypt and convert a database row to a dict."""
        try:
            content = self._decrypt(row[3])
        except InvalidToken:
            content = "[DECRYPTION FAILED]"
        return {
            "entry_id": row[0],
            "entry_type": row[1],
            "timestamp": row[2],
            "content": content,
            "tags": json.loads(row[4]) if row[4] else [],
            "meta": json.loads(row[5]) if row[5] else {},
        }
