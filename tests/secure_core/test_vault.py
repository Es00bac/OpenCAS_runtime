"""Tests for the Secure Core vault."""

import pytest
import pytest_asyncio
from pathlib import Path

from opencas.secure_core.vault import SecureCoreVault


@pytest_asyncio.fixture
async def vault(tmp_path: Path):
    v = SecureCoreVault(
        db_path=tmp_path / "secure_core.db",
        salt_path=tmp_path / "vault_salt",
    )
    await v.connect()
    yield v
    await v.close()


@pytest_asyncio.fixture
async def vault_with_passphrase(tmp_path: Path):
    v = SecureCoreVault(
        db_path=tmp_path / "secure_core_pass.db",
        salt_path=tmp_path / "vault_salt_pass",
        passphrase="test-secret-passphrase",
    )
    await v.connect()
    yield v
    await v.close()


@pytest.mark.asyncio
async def test_vault_connect_and_status(vault: SecureCoreVault) -> None:
    status = vault.vault_status()
    assert status["unlocked"] is True
    assert status["has_key"] is True


@pytest.mark.asyncio
async def test_save_and_retrieve_entry(vault: SecureCoreVault) -> None:
    await vault.save_entry(
        entry_id="ep-001",
        entry_type="episode",
        content="This is a private memory about my deepest fear.",
        tags=["secure_core", "private"],
    )
    entry = await vault.get_entry("ep-001")
    assert entry is not None
    assert entry["content"] == "This is a private memory about my deepest fear."
    assert entry["entry_type"] == "episode"
    assert "secure_core" in entry["tags"]


@pytest.mark.asyncio
async def test_list_entries_filtered(vault: SecureCoreVault) -> None:
    await vault.save_entry("ep-1", "episode", "Private episode content")
    await vault.save_entry("mem-1", "memory", "Private memory content")
    await vault.save_entry("blf-1", "belief", "Private belief content")

    episodes = await vault.list_entries(entry_type="episode")
    assert len(episodes) == 1
    assert episodes[0]["entry_type"] == "episode"

    all_entries = await vault.list_entries()
    assert len(all_entries) == 3


@pytest.mark.asyncio
async def test_encryption_is_actual(vault: SecureCoreVault) -> None:
    """Verify that data stored in SQLite is actually encrypted, not plaintext."""
    import aiosqlite

    await vault.save_entry("ep-secret", "episode", "This should be encrypted")

    # Read raw bytes from SQLite - content should NOT be plaintext
    db = await aiosqlite.connect(str(vault.db_path))
    cursor = await db.execute(
        "SELECT encrypted_content FROM vault_entries WHERE entry_id = ?", ("ep-secret",)
    )
    row = await cursor.fetchone()
    await db.close()

    assert row is not None
    raw = row[0]
    assert isinstance(raw, bytes)
    # The raw bytes should not contain the plaintext
    assert b"This should be encrypted" not in raw


@pytest.mark.asyncio
async def test_delete_entry(vault: SecureCoreVault) -> None:
    await vault.save_entry("ep-del", "episode", "To be deleted")
    assert await vault.delete_entry("ep-del") is True
    assert await vault.get_entry("ep-del") is None
    assert await vault.delete_entry("nonexistent") is False


@pytest.mark.asyncio
async def test_count_entries(vault: SecureCoreVault) -> None:
    await vault.save_entry("c-1", "episode", "Content 1")
    await vault.save_entry("c-2", "episode", "Content 2")
    await vault.save_entry("c-3", "memory", "Content 3")

    assert await vault.count_entries() == 3
    assert await vault.count_entries("episode") == 2
    assert await vault.count_entries("memory") == 1


@pytest.mark.asyncio
async def test_locked_vault_returns_empty(vault: SecureCoreVault) -> None:
    """When vault is locked (closed), reads should return nothing."""
    await vault.save_entry("ep-lock", "episode", "Secret content")
    await vault.close()

    # Re-open without deriving key (simulating locked state)
    vault._unlocked = False
    vault._fernet = None

    result = await vault.get_entry("ep-lock")
    assert result is None
    entries = await vault.list_entries()
    assert entries == []


@pytest.mark.asyncio
async def test_passphrase_derived_key(vault_with_passphrase: SecureCoreVault) -> None:
    """Vault with passphrase should encrypt and decrypt correctly."""
    await vault_with_passphrase.save_entry(
        "ep-pass", "episode", "Encrypted with passphrase"
    )
    entry = await vault_with_passphrase.get_entry("ep-pass")
    assert entry is not None
    assert entry["content"] == "Encrypted with passphrase"


@pytest.mark.asyncio
async def test_different_passphrase_cannot_decrypt(tmp_path: Path) -> None:
    """Data encrypted with one passphrase should not be readable with another."""
    v1 = SecureCoreVault(
        db_path=tmp_path / "multi.db",
        salt_path=tmp_path / "salt1",
        passphrase="passphrase-A",
    )
    await v1.connect()
    await v1.save_entry("ep-cross", "episode", "Secret for A")
    await v1.close()

    # Open same DB with different passphrase
    v2 = SecureCoreVault(
        db_path=tmp_path / "multi.db",
        salt_path=tmp_path / "salt2",
        passphrase="passphrase-B",
    )
    await v2.connect()
    entry = await v2.get_entry("ep-cross")
    assert entry is not None
    # Decryption should fail — content shows error marker
    assert entry["content"] == "[DECRYPTION FAILED]"
    await v2.close()


@pytest.mark.asyncio
async def test_update_existing_entry(vault: SecureCoreVault) -> None:
    """Saving with the same entry_id should update, not duplicate."""
    await vault.save_entry("ep-up", "episode", "Original content")
    await vault.save_entry("ep-up", "episode", "Updated content")

    entry = await vault.get_entry("ep-up")
    assert entry is not None
    assert entry["content"] == "Updated content"

    count = await vault.count_entries()
    assert count == 1
