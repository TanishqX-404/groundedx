"""Build retrieval documents without reading scoring labels."""

from __future__ import annotations

from typing import Any

from groundedx.config import DomainConfig
from groundedx.encoding.context_encoder import encode_context

KB_VARIANTS = ("full", "no_remediation", "generic_remediation")
# Control condition only: adds class-named reference cards to the index, as a
# label-exposed configuration did. Never used for the leakage-controlled results.
LABEL_EXPOSED = "label_exposed"
GENERIC_REMEDIATION = (
    "The incident was escalated to the operations team, who applied the standard "
    "runbook and verified service recovery."
)


def case_text(
    sample: dict[str, Any], domain: DomainConfig, variant: str = "full", ranking: str = "raw"
) -> str:
    """Render one historical case. Never includes the evaluator label."""

    if variant not in (*KB_VARIANTS, LABEL_EXPOSED):
        raise ValueError(f"Unknown KB variant {variant!r}; expected one of {KB_VARIANTS}")
    text = encode_context(sample, domain, ranking=ranking)
    if variant in ("full", LABEL_EXPOSED):
        text += f" Remediation applied by operations team: {sample.get('remediation', '')}"
    elif variant == "generic_remediation":
        text += f" Remediation applied by operations team: {GENERIC_REMEDIATION}"
    return text


def build_documents(
    samples: list[dict[str, Any]],
    domain: DomainConfig,
    variant: str = "full",
    ranking: str = "raw",
) -> list[dict[str, str]]:
    """Create label-free historical-case documents with opaque sequential IDs.

    This function deliberately does not read evaluator labels. Callers choose
    which windows become cases (see ``groundedx.kb.scoring_labels``).
    """

    return [
        {
            "chunk_id": f"case_{index:04d}",
            "text": case_text(sample, domain, variant, ranking),
            "doc_type": "historical_case",
        }
        for index, sample in enumerate(samples, start=1)
    ]


def label_cards(domain: DomainConfig) -> list[dict[str, str]]:
    """Class-named reference cards for the label-exposed CONTROL condition only.

    Mirrors a label-exposed configuration: each card states the diagnosis label,
    its alarm codes, KPI signature, temporal pattern and remediation.
    """

    cards = []
    for index, fault in enumerate(domain.taxonomy, start=1):
        effects = ", ".join(f"{k} changes by {v:+}" for k, v in fault.kpi_effects.items())
        alarms = ", ".join(fault.alarms)
        text = (
            f"Fault signature card. Diagnosis label: {fault.name}. Domain: {fault.domain}. "
            f"Signature: this fault is indicated by alarms {alarms}. "
            f"The expected KPI signature is: {effects}. "
            f"The temporal pattern is usually {fault.pattern} shaped over a diagnosis window. "
            f"Remediation: {fault.remediation}"
        )
        cards.append({"chunk_id": f"card_{index:03d}", "text": text, "doc_type": "fault_card"})
    return cards
