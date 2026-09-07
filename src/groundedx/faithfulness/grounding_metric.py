"""A minimal reusable citation faithfulness check."""

from __future__ import annotations

import re
from collections.abc import Iterable

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
    "without",
}


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_/-]+", text.lower())
        if token not in STOPWORDS and len(token) > 2
    }


def grounding_check(
    explanation: str,
    citations: Iterable[str],
    retrieved: dict[str, str] | Iterable[dict[str, str]],
) -> bool:
    """Return true iff citations satisfy the three-condition grounding rule.

    The explanation must cite at least one ID, every cited ID must be in the
    retrieved set, and the explanation must share at least two content terms
    with the text of the cited evidence.
    """

    cited = list(citations)
    if not cited:
        return False
    evidence = (
        retrieved
        if isinstance(retrieved, dict)
        else {str(row["chunk_id"]): str(row["text"]) for row in retrieved}
    )
    if any(chunk_id not in evidence for chunk_id in cited):
        return False
    cited_text = " ".join(evidence[chunk_id] for chunk_id in cited)
    return len(_terms(explanation) & _terms(cited_text)) >= 2
