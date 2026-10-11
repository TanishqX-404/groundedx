"""Small CPU-friendly TF-IDF retriever."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    """Retrieve label-free documents by cosine similarity."""

    def __init__(
        self,
        documents: list[dict[str, Any]],
        *,
        sublinear_tf: bool = True,
        doc_type_bonus: dict[str, float] | None = None,
    ) -> None:
        if not documents:
            raise ValueError("At least one document is required")
        self.documents = [dict(document) for document in documents]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=sublinear_tf)
        self.matrix = self.vectorizer.fit_transform([doc["text"] for doc in self.documents])
        bonus = doc_type_bonus or {}
        self._bonus = np.array([bonus.get(doc.get("doc_type", ""), 0.0) for doc in self.documents])

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return up to ``k`` documents, never exposing private label fields."""

        if k < 1:
            raise ValueError("k must be positive")
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        scores = scores + self._bonus
        order = np.argsort(-scores)[: min(k, len(self.documents))]
        results = []
        for index in order:
            row = {
                key: value
                for key, value in self.documents[int(index)].items()
                if not key.startswith("_")
            }
            row["score"] = float(scores[int(index)])
            results.append(row)
        return results
