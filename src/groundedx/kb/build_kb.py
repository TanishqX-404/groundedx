"""Build retrieval documents without reading scoring labels."""

from __future__ import annotations

from typing import Any

from groundedx.config import DomainConfig
from groundedx.encoding.context_encoder import encode_context


def build_documents(
    samples: list[dict[str, Any]], domain: DomainConfig, limit: int | None = None
) -> list[dict[str, str]]:
    """Create label-free historical-case documents from telemetry windows.

    This function deliberately does not import or read evaluator labels. The
    returned text contains observations and remediation, but no evaluator class.
    """

    documents: list[dict[str, str]] = []
    for sample in samples[:limit]:
        chunk_id = f"case_{len(documents) + 1:04d}"
        text = (
            f"Historical telemetry observation {sample.get('id', chunk_id)}. "
            f"{encode_context(sample, domain)} "
            f"Remediation applied by operations team: {sample.get('remediation', '')}"
        )
        documents.append({"chunk_id": chunk_id, "text": text, "doc_type": "historical_case"})
    return documents
