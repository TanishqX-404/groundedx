"""Strict validation for model outputs. There is no fallback path: an output
that fails any check is invalid and is scored as wrong by the evaluator."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

INVALID_REASONS = ("truncated", "parse_error", "schema_error", "out_of_taxonomy", "bad_citation")


class Diagnosis(BaseModel):
    """The public diagnosis schema."""

    root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    remediation: str
    explanation: str
    citations: list[str]


class InvalidOutput(ValueError):
    """Raised for a rejected generation; ``reason`` is one of INVALID_REASONS."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def validate_diagnosis(
    payload: dict[str, Any] | Diagnosis, taxonomy: set[str], retrieved_ids: set[str]
) -> Diagnosis:
    """Validate schema, closed taxonomy, and citation membership."""

    try:
        result = payload if isinstance(payload, Diagnosis) else Diagnosis.model_validate(payload)
    except ValidationError as exc:
        raise InvalidOutput("schema_error", str(exc)) from exc
    if result.root_cause not in taxonomy:
        raise InvalidOutput(
            "out_of_taxonomy",
            f"root_cause must be one of the active taxonomy classes: {sorted(taxonomy)}",
        )
    invalid = sorted(set(result.citations) - retrieved_ids)
    if invalid:
        raise InvalidOutput(
            "bad_citation", f"citations reference IDs not present in retrieved evidence: {invalid}"
        )
    return result


def parse_and_validate(
    raw: str, taxonomy: set[str], retrieved_ids: set[str], finish_reason: str | None = None
) -> Diagnosis:
    """Parse raw model text and validate it, raising ``InvalidOutput`` on failure."""

    if finish_reason == "length":
        raise InvalidOutput("truncated", "generation hit max_tokens")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidOutput("parse_error", str(exc)) from exc
    if not isinstance(payload, dict):
        raise InvalidOutput("parse_error", "output is not a JSON object")
    return validate_diagnosis(payload, taxonomy, retrieved_ids)
