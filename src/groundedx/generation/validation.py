"""Strict validation for model outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Diagnosis(BaseModel):
    """The public diagnosis schema."""

    root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    remediation: str
    explanation: str
    citations: list[str]


def validate_diagnosis(
    payload: dict[str, Any] | Diagnosis, taxonomy: set[str], retrieved_ids: set[str]
) -> Diagnosis:
    """Validate schema, closed taxonomy, and citation membership."""

    result = payload if isinstance(payload, Diagnosis) else Diagnosis.model_validate(payload)
    if result.root_cause not in taxonomy:
        raise ValueError(
            f"root_cause must be one of the active taxonomy classes: {sorted(taxonomy)}"
        )
    invalid = sorted(set(result.citations) - retrieved_ids)
    if invalid:
        raise ValueError(f"citations reference IDs not present in retrieved evidence: {invalid}")
    return result
