"""Tests for persisted phone bridge configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from opencas.phone_config import (
    PhoneContactPolicy,
    PhoneRuntimeConfig,
    load_phone_runtime_config,
    normalize_phone_number,
    save_phone_runtime_config,
)


def test_normalize_phone_number_handles_common_inputs() -> None:
    assert normalize_phone_number("+1 (555) 123-4567") == "+15551234567"
    assert normalize_phone_number("5551234567") == "+15551234567"
    assert normalize_phone_number("1-555-123-4567") == "+15551234567"
    assert normalize_phone_number("442071838750") == "+442071838750"
    assert normalize_phone_number("") is None
    assert normalize_phone_number(None) is None


def test_phone_contact_policy_rejects_owner_trust_and_workspace_escape() -> None:
    with pytest.raises(ValueError, match="contacts must use low trust"):
        PhoneContactPolicy(phone_number="+15551234567", trust_level="owner")

    with pytest.raises(ValueError, match="managed workspace"):
        PhoneContactPolicy(phone_number="+15551234567", workspace_subdir="../escape")


def test_phone_contact_policy_preserves_explicit_empty_actions() -> None:
    contact = PhoneContactPolicy(phone_number="+15551234567", allowed_actions=[])

    assert contact.allowed_actions == []


def test_phone_runtime_config_round_trips_and_dedupes_contacts(tmp_path: Path) -> None:
    config = PhoneRuntimeConfig(
        enabled=True,
        public_base_url="https://opencas.example.com/",
        twilio_from_number="5551112222",
        owner_phone_number="+1 (555) 123-4567",
        owner_workspace_subdir="phone/owner",
        contacts=[
            {
                "phone_number": "5550001111",
                "display_name": "Alex",
                "allowed_actions": ["knowledge_qa"],
                "workspace_subdir": "phone/contacts/alex",
            },
            {
                "phone_number": "+1 555 000 1111",
                "display_name": "Duplicate Alex",
                "allowed_actions": ["leave_message"],
            },
        ],
    )

    saved_path = save_phone_runtime_config(tmp_path, config)
    loaded = load_phone_runtime_config(tmp_path)

    assert saved_path == Path(tmp_path) / "phone" / "config.json"
    assert loaded.enabled is True
    assert loaded.public_base_url == "https://opencas.example.com"
    assert loaded.twilio_from_number == "+15551112222"
    assert loaded.owner_phone_number == "+15551234567"
    assert len(loaded.contacts) == 1
    assert loaded.contacts[0].phone_number == "+15550001111"
    assert loaded.contacts[0].display_name == "Alex"
    assert loaded.contacts[0].allowed_actions == ["knowledge_qa"]
    assert loaded.contacts[0].workspace_subdir == "phone/contacts/alex"


def test_phone_runtime_config_rejects_owner_in_contacts() -> None:
    with pytest.raises(ValueError, match="must not also appear in contacts"):
        PhoneRuntimeConfig(
            owner_phone_number="+15551234567",
            contacts=[{"phone_number": "+15551234567"}],
        )
