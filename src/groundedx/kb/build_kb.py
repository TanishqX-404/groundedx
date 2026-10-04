"""Build retrieval documents without reading scoring labels."""

from __future__ import annotations

from typing import Any

from groundedx.config import DomainConfig
from groundedx.encoding.context_encoder import encode_context

KB_VARIANTS = ("full", "no_remediation", "generic_remediation")
GENERIC_REMEDIATION = (
    "The incident was escalated to the operations team, who applied the standard "
    "runbook and verified service recovery."
)


def case_text(sample: dict[str, Any], domain: DomainConfig, variant: str = "full") -> str:
    """Render one historical case. Never includes the evaluator label."""

    if variant not in KB_VARIANTS:
        raise ValueError(f"Unknown KB variant {variant!r}; expected one of {KB_VARIANTS}")
    text = encode_context(sample, domain)
    if variant == "full":
        text += f" Remediation applied by operations team: {sample.get('remediation', '')}"
    elif variant == "generic_remediation":
        text += f" Remediation applied by operations team: {GENERIC_REMEDIATION}"
    return text


def build_documents(
    samples: list[dict[str, Any]], domain: DomainConfig, variant: str = "full"
) -> list[dict[str, str]]:
    """Create label-free historical-case documents with opaque sequential IDs.

    This function deliberately does not read evaluator labels. Callers choose
    which windows become cases (see ``groundedx.kb.scoring_labels``).
    """

    return [
        {
            "chunk_id": f"case_{index:04d}",
            "text": case_text(sample, domain, variant),
            "doc_type": "historical_case",
        }
        for index, sample in enumerate(samples, start=1)
    ]
