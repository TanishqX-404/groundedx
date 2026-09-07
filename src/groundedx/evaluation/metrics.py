"""Metrics that do not depend on a particular domain."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, f1_score

from groundedx.faithfulness.grounding_metric import grounding_check


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Return top-1 accuracy."""

    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Return macro-F1 with zero division handled."""

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


__all__ = ["accuracy", "grounding_check", "macro_f1"]
