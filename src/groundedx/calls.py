"""Build one SLM call (prompt, allowed classes, citable IDs) for each mode.

Modes
- ``zero-shot`` / ``rag``: the original setup; optional ``class_docs`` adds a
  fault reference (taxonomy documentation, the operator's runbook) to the prompt.
- ``hybrid``: a feature-based GBDT proposes its top-N classes with
  probabilities; the SLM must choose among them (grammar-enforced) and explain
  the choice with citations to retrieved cases. N=1 means the classifier
  decides and the SLM only explains and grounds.

The GBDT is trained on the seed's training split with the hyper-parameters
selected on the validation split (``results/v2/gbdt/seed_<s>/manifest.json``).
No evaluator label of the window being diagnosed is ever used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from groundedx.config import DomainConfig, FaultSpec
from groundedx.encoding.context_encoder import encode_context
from groundedx.experiment import RESULTS_ROOT, ROOT, SeedSetup
from groundedx.generation.prompt_templates import render_prompt


@dataclass(frozen=True)
class CallSpec:
    mode: str  # "zero-shot" | "rag" | "hybrid"
    k: int = 5
    class_docs: bool = False
    candidates: int = 3

    @property
    def uses_retrieval(self) -> bool:
        return self.mode in ("rag", "hybrid")


def describe_fault(fault: FaultSpec) -> str:
    """One-line fault reference derived from the taxonomy (no window data)."""

    signature = ", ".join(
        f"{name} {'up' if float(effect) > 0 else 'down'}"
        for name, effect in fault.kpi_effects.items()
    )
    return (
        f"[{fault.domain}] alarms {', '.join(fault.alarms)}; KPI signature: {signature}; "
        f"temporal pattern: {fault.pattern}; runbook: {fault.remediation}"
    )


def experiment_name(model: str, spec: CallSpec, kb_variant: str, split: str) -> str:
    if spec.mode == "zero-shot":
        name = f"llm__{model}__zs"
    elif spec.mode == "rag":
        name = f"llm__{model}__rag__{kb_variant}__k{spec.k}"
    else:
        name = f"llm__{model}__hybrid{spec.candidates}__{kb_variant}__k{spec.k}"
    if spec.class_docs and spec.mode != "hybrid":
        name += "__docs"
    if split != "test":
        name += f"__{split}"
    return name


def _gbdt_params(seed: int) -> dict[str, Any] | None:
    path = RESULTS_ROOT / "gbdt" / f"seed_{seed}" / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("selected")
    return None


class CallBuilder:
    def __init__(self, setup: SeedSetup, spec: CallSpec) -> None:
        self.setup = setup
        self.spec = spec
        domain_dir = ROOT / "domains" / setup.domain.name
        self.domain: DomainConfig = setup.domain
        if spec.mode == "hybrid":
            self.template = domain_dir / "prompt_template_hybrid.jinja"
        elif spec.class_docs:
            self.template = domain_dir / "prompt_template_docs.jinja"
        else:
            self.template = domain_dir / "prompt_template.jinja"
        self.docs = {fault.name: describe_fault(fault) for fault in self.domain.taxonomy}
        self._proba: dict[str, dict[str, float]] = {}
        self.gbdt_params: dict[str, Any] | None = None
        if spec.mode == "hybrid":
            self._fit_classifier()

    def _fit_classifier(self) -> None:
        from xgboost import XGBClassifier

        from groundedx.baselines.random_forest import extract_features, train_gbdt

        seed = self.setup.seed
        train = self.setup.splits["train"]
        params = _gbdt_params(seed)
        if params is None:
            _, params, _ = train_gbdt(train, self.setup.splits["val"], self.domain, seed)
        self.gbdt_params = params
        x_train, y_train = extract_features(train, self.domain)
        self._labels = sorted(self.domain.classes)
        index = {label: i for i, label in enumerate(self._labels)}
        self._columns = list(x_train.columns)
        self._clf = XGBClassifier(random_state=seed, n_jobs=-1, tree_method="hist", **params)
        self._clf.fit(x_train, np.array([index[y] for y in y_train]))

    def precompute(self, samples: list[dict[str, Any]]) -> None:
        """Batch classifier inference for the windows that will be diagnosed."""

        if self.spec.mode != "hybrid" or not samples:
            return
        from groundedx.baselines.random_forest import extract_features

        features, _ = extract_features(samples, self.domain)
        proba = self._clf.predict_proba(features.reindex(columns=self._columns, fill_value=0.0))
        for sample, row in zip(samples, proba):
            self._proba[sample["id"]] = dict(zip(self._labels, map(float, row)))

    def build(self, sample: dict[str, Any]) -> dict[str, Any]:
        query = encode_context(sample, self.domain)
        retrieved = (
            self.setup.retriever.retrieve(query, k=self.spec.k) if self.spec.uses_retrieval else []
        )
        classes = self.domain.classes
        candidates: list[tuple[str, float]] = []
        reference = ""
        if self.spec.mode == "hybrid":
            proba = self._proba[sample["id"]]
            candidates = sorted(proba.items(), key=lambda item: -item[1])[: self.spec.candidates]
            classes = [name for name, _ in candidates]
            reference = "\n".join(
                f"- {name} (p={p:.2f}): {self.docs[name]}" for name, p in candidates
            )
        elif self.spec.class_docs:
            reference = "\n".join(f"- {name}: {self.docs[name]}" for name in classes)
        prompt = render_prompt(
            query, retrieved, self.domain, self.template, taxonomy=classes, reference=reference
        )
        return {
            "prompt": prompt,
            "classes": classes,
            "citation_ids": [row["chunk_id"] for row in retrieved],
            "retrieved": retrieved,
            "candidates": [[name, round(p, 4)] for name, p in candidates],
        }
