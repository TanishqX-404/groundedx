"""Evaluator-only label helpers.

Core retrieval and generation modules must never import this module. The
offline builder uses it to pick which training windows become KB cases
(stratified, seeded) and to write the evaluator-only chunk-to-class map.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def select_kb_cases(train: list[dict[str, Any]], per_class: int, seed: int) -> list[dict[str, Any]]:
    """Sample up to ``per_class`` training windows per class, in shuffled order."""

    rng = np.random.default_rng(seed)
    by_class: dict[str, list[dict[str, Any]]] = {}
    for sample in train:
        by_class.setdefault(str(sample["fault_class"]), []).append(sample)
    chosen: list[dict[str, Any]] = []
    for label in sorted(by_class):
        pool = by_class[label]
        picks = rng.choice(len(pool), size=min(per_class, len(pool)), replace=False)
        chosen.extend(pool[int(i)] for i in picks)
    order = rng.permutation(len(chosen))
    return [chosen[int(i)] for i in order]


def make_scoring_labels(samples: list[dict[str, Any]], chunk_ids: list[str]) -> dict[str, str]:
    """Create an evaluator-owned mapping after retrieval documents are built."""

    return {chunk_id: str(sample["fault_class"]) for chunk_id, sample in zip(chunk_ids, samples)}
