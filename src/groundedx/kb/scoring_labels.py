"""Evaluator-only label helpers.

Core retrieval and generation modules must never import this module.
"""

from __future__ import annotations

from typing import Any


def make_scoring_labels(samples: list[dict[str, Any]], chunk_ids: list[str]) -> dict[str, str]:
    """Create an evaluator-owned mapping after retrieval documents are built."""

    return {chunk_id: str(sample["fault_class"]) for chunk_id, sample in zip(chunk_ids, samples)}
