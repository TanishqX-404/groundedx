"""Alarm emission helpers are kept separate for domain-specific extensions."""

from __future__ import annotations

from groundedx.config import DomainConfig


def alarm_codes(domain: DomainConfig) -> list[str]:
    """Return all alarm codes declared by the active domain."""

    return sorted({code for fault in domain.taxonomy for code in fault.alarms})
