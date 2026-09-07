"""Domain-neutral alarm-overlap baseline."""

from __future__ import annotations

from groundedx.config import DomainConfig


def rule_based_predict(sample: dict, domain: DomainConfig) -> str:
    """Predict the class with the most overlapping active alarms."""

    active = {str(item.get("code", "")) for item in sample.get("alarms", [])}
    scores = [
        (len(active.intersection(set(fault.alarms))), fault.name) for fault in domain.taxonomy
    ]
    return min(scores, key=lambda row: (-row[0], row[1]))[1]
