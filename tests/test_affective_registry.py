"""Tests for the append-only affective registry writer."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import List

import pytest

from opencas.affective_registry import (
    AffectiveRegistryEntry,
    AffectiveRegistryWriter,
    ExecutionContext,
    ExecutionPhase,
    SystemMetrics,
)
from opencas.affective_registry.models import AffectiveState
from opencas.affective_registry.writer import RegistryWriteError, ValidationError


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    return tmp_path / "test_registry.jsonl"


@pytest.fixture
def writer(registry_path: Path) -> AffectiveRegistryWriter:
    return AffectiveRegistryWriter(
        registry_path=registry_path,
        enable_locking=True,
        validate_writes=True,
    )


class TestAffectiveRegistryWriter:
    """Unit tests for AffectiveRegistryWriter."""

    def test_append_creates_file(self, writer: AffectiveRegistryWriter, registry_path: Path):
        """Writing the first entry should create the registry file."""
        assert not registry_path.exists()
        entry = AffectiveRegistryEntry()
        writer.append(entry)
        assert registry_path.exists()

    def test_append_preserves_history(self, writer: AffectiveRegistryWriter):
        """Multiple appends must not overwrite earlier entries."""
        entries: List[AffectiveRegistryEntry] = []
        for i in range(5):
            entry = AffectiveRegistryEntry(
                phase=ExecutionPhase.BOOT,
                affective_state=AffectiveState(valence=i * 0.1),
            )
            writer.append(entry)
            entries.append(entry)

        read_back = list(writer.iter_entries())
        assert len(read_back) == 5
        for original, recovered in zip(entries, read_back):
            assert recovered.entry_id == original.entry_id
            assert recovered.affective_state.valence == pytest.approx(original.affective_state.valence)

    def test_append_is_atomic_under_concurrency(self, registry_path: Path):
        """Concurrent writers must not corrupt or lose entries."""
        writer = AffectiveRegistryWriter(
            registry_path=registry_path,
            enable_locking=True,
            validate_writes=True,
        )
        errors: List[Exception] = []
        threads: List[threading.Thread] = []

        def worker(idx: int):
            try:
                for _ in range(20):
                    entry = AffectiveRegistryEntry(
                        phase=ExecutionPhase.TEST,
                        payload={"worker": idx},
                    )
                    writer.append(entry)
            except Exception as exc:
                errors.append(exc)

        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert not errors, f"Concurrent writes raised exceptions: {errors}"
        assert writer.count_entries() == 100

    def test_validation_failure_raises(self, registry_path: Path, monkeypatch):
        """If validation detects a mismatch, ValidationError must be raised."""
        writer = AffectiveRegistryWriter(
            registry_path=registry_path,
            enable_locking=True,
            validate_writes=True,
        )

        original_validate = writer._validate_append

        def bad_validate(fd: int, expected: bytes):
            raise ValidationError("injected mismatch")

        monkeypatch.setattr(writer, "_validate_append", bad_validate)

        with pytest.raises(ValidationError):
            writer.append(AffectiveRegistryEntry())

    def test_read_back_reverse(self, writer: AffectiveRegistryWriter):
        """Reverse iteration must yield entries newest-first."""
        for i in range(3):
            writer.append(
                AffectiveRegistryEntry(
                    phase=ExecutionPhase.TEST,
                    payload={"seq": i},
                )
            )

        reverse = list(writer.iter_entries_reverse())
        assert len(reverse) == 3
        assert reverse[0].payload["seq"] == 2
        assert reverse[1].payload["seq"] == 1
        assert reverse[2].payload["seq"] == 0

    def test_get_latest(self, writer: AffectiveRegistryWriter):
        """get_latest must return the most recent entries."""
        for i in range(10):
            writer.append(
                AffectiveRegistryEntry(
                    phase=ExecutionPhase.TEST,
                    payload={"seq": i},
                )
            )

        latest = writer.get_latest(3)
        assert len(latest) == 3
        assert latest[0].payload["seq"] == 7
        assert latest[1].payload["seq"] == 8
        assert latest[2].payload["seq"] == 9

    def test_corrupted_lines_skipped(self, writer: AffectiveRegistryWriter, registry_path: Path):
        """Corrupted JSON lines must be skipped, not crash iteration."""
        writer.append(AffectiveRegistryEntry(payload={"valid": True}))
        with open(registry_path, "a") as f:
            f.write("this is not json\n")
        writer.append(AffectiveRegistryEntry(payload={"valid": True}))

        entries = list(writer.iter_entries())
        assert len(entries) == 2
        assert all(e.payload.get("valid") for e in entries)

    def test_append_from_somatic_state(self, writer: AffectiveRegistryWriter):
        """The convenience helper must map somatic attributes correctly."""

        class FakeSomaticState:
            primary_emotion = "joy"
            valence = 0.8
            arousal = 0.7
            fatigue = 0.1
            tension = 0.0
            focus = 0.9
            energy = 0.8
            certainty = 0.95
            musubi = 0.75
            somatic_tag = "flow"

        entry = writer.append_from_somatic_state(
            FakeSomaticState(),
            phase=ExecutionPhase.TURN_START,
            session_id="sess-42",
            payload={"note": "hello"},
        )

        assert entry.affective_state.primary_emotion == "joy"
        assert entry.affective_state.valence == pytest.approx(0.8)
        assert entry.affective_state.musubi == pytest.approx(0.75)
        assert entry.affective_state.somatic_tag == "flow"
        assert entry.phase == ExecutionPhase.TURN_START
        assert entry.execution_context.session_id == "sess-42"
        assert entry.payload["note"] == "hello"

    def test_system_metrics_capture(self):
        """SystemMetrics.capture must return a model without raising."""
        metrics = SystemMetrics.capture()
        assert metrics is not None
        # At minimum, the call itself must not raise

    def test_path_creation_on_init(self, tmp_path: Path):
        """Writer init must create parent directories if missing."""
        deep_path = tmp_path / "a" / "b" / "c" / "registry.jsonl"
        assert not deep_path.parent.exists()
        AffectiveRegistryWriter(deep_path)
        assert deep_path.parent.exists()

    def test_entry_jsonl_roundtrip(self):
        """to_jsonl / from_jsonl must be lossless."""
        original = AffectiveRegistryEntry(
            phase=ExecutionPhase.DAYDREAM,
            affective_state=AffectiveState(valence=-0.3, arousal=0.9),
            execution_context=ExecutionContext(session_id="roundtrip-test"),
            payload={"nested": {"key": [1, 2, 3]}},
        )
        line = original.to_jsonl()
        recovered = AffectiveRegistryEntry.from_jsonl(line)
        assert recovered.entry_id == original.entry_id
        assert recovered.affective_state.valence == pytest.approx(original.affective_state.valence)
        assert recovered.execution_context.session_id == "roundtrip-test"
        assert recovered.payload["nested"]["key"] == [1, 2, 3]


class TestAffectiveRegistrySmokeTest:
    """Smoke-test against the actual target path."""

    def test_smoke_test_target_path(self, tmp_path: Path):
        """Simulate the real target path and verify append-only semantics."""
        # We can't write to /tmp/opencas-public-fixture/openbulma/tmp in tests,
        # so we verify the writer works with a path that has the same
        # structural characteristics (deep nested directory).
        target = tmp_path / "mnt" / "xtra" / "openbulma-v4" / "tmp" / "baa-neural-link-smoke-test"
        writer = AffectiveRegistryWriter(target)

        # Simulate multiple "historical runs"
        for run in range(3):
            entry = AffectiveRegistryEntry(
                phase=ExecutionPhase.TEST,
                payload={"run": run, "simulated": True},
            )
            writer.append(entry)

        entries = list(writer.iter_entries())
        assert len(entries) == 3
        assert entries[0].payload["run"] == 0
        assert entries[1].payload["run"] == 1
        assert entries[2].payload["run"] == 2

        # Verify file is JSONL (one complete JSON object per line)
        with open(target, "r") as f:
            lines = f.readlines()
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert "entry_id" in obj
            assert "timestamp" in obj
            assert "affective_state" in obj
