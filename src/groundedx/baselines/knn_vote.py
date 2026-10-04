"""Retrieval-only baseline: similarity-weighted vote over retrieved case labels.

This uses the same TF-IDF index and top-k as RAG, but replaces the SLM with a
vote over the evaluator-side labels of the retrieved cases. It measures how
much of the RAG accuracy comes from retrieval alone.
"""

from __future__ import annotations

from typing import Any


def knn_vote(retrieved: list[dict[str, Any]], scoring_labels: dict[str, str]) -> str:
    """Return the class with the largest summed similarity; ties go to the top-ranked case."""

    weights: dict[str, float] = {}
    first_rank: dict[str, int] = {}
    for rank, row in enumerate(retrieved):
        label = scoring_labels[row["chunk_id"]]
        weights[label] = weights.get(label, 0.0) + float(row["score"])
        first_rank.setdefault(label, rank)
    return min(weights, key=lambda label: (-weights[label], first_rank[label]))
