"""Classical and retrieval-only baselines."""

from .knn_vote import knn_vote
from .rule_based import rule_based_predict

__all__ = ["knn_vote", "rule_based_predict"]
