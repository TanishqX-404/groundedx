"""Reproducible CPU evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable
from statistics import mean, stdev

from groundedx.config import DomainConfig
from groundedx.evaluation.metrics import accuracy, macro_f1


def evaluate_seeds(
    samples: list[dict],
    domain: DomainConfig,
    predictor: Callable[[dict, DomainConfig, int], str],
    seeds: list[int],
) -> dict[str, float | str]:
    """Evaluate a predictor for each seed and report mean and standard deviation."""

    scores = []
    f1_scores = []
    for seed in seeds:
        predictions = [predictor(sample, domain, seed) for sample in samples]
        truth = [str(sample["fault_class"]) for sample in samples]
        scores.append(accuracy(truth, predictions))
        f1_scores.append(macro_f1(truth, predictions))
    return {
        "accuracy_mean": mean(scores),
        "accuracy_std": stdev(scores) if len(scores) > 1 else 0.0,
        "macro_f1_mean": mean(f1_scores),
        "macro_f1_std": stdev(f1_scores) if len(f1_scores) > 1 else 0.0,
    }
