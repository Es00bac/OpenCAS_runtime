"""In-memory canonical capability registry."""

from __future__ import annotations

from .models import CapabilityDescriptor, CapabilitySource, CapabilityStatus


class CapabilityRegistry:
    """Store and query canonical capability descriptors."""

    def __init__(self) -> None:
        self._capabilities: dict[str, CapabilityDescriptor] = {}

    def register(self, descriptor: CapabilityDescriptor) -> None:
        """Register or replace a capability descriptor by id."""

        self._capabilities[descriptor.capability_id] = descriptor

    def get(self, capability_id: str) -> CapabilityDescriptor | None:
        """Return the stored descriptor for *capability_id* if present."""

        return self._capabilities.get(capability_id)

    def list_capabilities(
        self,
        source: CapabilitySource | str | None = None,
        owner_id: str | None = None,
        status: CapabilityStatus | str | None = None,
    ) -> list[CapabilityDescriptor]:
        """Return sorted descriptors, optionally filtered by source, owner, and status."""

        normalized_source = source.value if isinstance(source, CapabilitySource) else source
        normalized_status = status.value if isinstance(status, CapabilityStatus) else status

        capabilities = [
            descriptor
            for descriptor in self._capabilities.values()
            if (normalized_source is None or descriptor.source.value == normalized_source)
            and (owner_id is None or descriptor.owner_id == owner_id)
            and (normalized_status is None or descriptor.status.value == normalized_status)
        ]
        return sorted(capabilities, key=lambda descriptor: descriptor.capability_id)

    def update_status(
        self,
        capability_id: str,
        status: CapabilityStatus,
        *,
        errors: list[str] | None = None,
    ) -> None:
        """Update status and validation errors for a registered capability."""

        descriptor = self._capabilities.get(capability_id)
        if descriptor is None:
            raise KeyError(capability_id)
        self._capabilities[capability_id] = descriptor.with_status(status, errors=errors)

    def unregister_owner(self, owner_id: str) -> None:
        """Remove every capability descriptor owned by *owner_id*."""

        for capability_id, descriptor in list(self._capabilities.items()):
            if descriptor.owner_id == owner_id:
                self._capabilities.pop(capability_id, None)
