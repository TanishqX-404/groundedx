import re
from pathlib import Path

import pytest

from groundedx.config import load_domain_bundle
from groundedx.encoding.context_encoder import encode_context
from groundedx.generation.grammar import build_gbnf_grammar
from groundedx.generation.validation import validate_diagnosis
from groundedx.kb.build_kb import build_documents
from groundedx.retrieval.tfidf_retriever import TfidfRetriever
from groundedx.simulator.kpi_generator import generate_dataset

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("domain_name", ["oran", "sre-k8s"])
def test_context_encoding_is_domain_agnostic(domain_name):
    domain = load_domain_bundle(ROOT / "domains" / domain_name)
    sample = generate_dataset(domain, 1, seed=11)[0]
    query = encode_context(sample, domain)
    assert "Alarms:" in query
    assert any(
        metric.name in query for metric in domain.metrics if metric.name in sample["kpis"][0]
    )
    assert sample["fault_class"] not in query


def test_retriever_returns_k_without_private_labels():
    domain = load_domain_bundle(ROOT / "domains" / "oran")
    samples = generate_dataset(domain, 8, seed=3)
    docs = build_documents(samples, domain)
    retriever = TfidfRetriever(docs)
    results = retriever.retrieve(encode_context(samples[0], domain), k=3)
    assert len(results) == 3
    assert all(not key.startswith("_") for row in results for key in row)
    assert all("fault_class" not in row["text"] for row in results)


@pytest.mark.parametrize("domain_name", ["oran", "sre-k8s"])
def test_validation_and_grammar_use_active_domain(domain_name):
    domain = load_domain_bundle(ROOT / "domains" / domain_name)
    valid = {
        "root_cause": domain.classes[0],
        "confidence": 0.8,
        "remediation": "do the thing",
        "explanation": "metric alarm evidence supports the diagnosis",
        "citations": ["case_1"],
    }
    assert (
        validate_diagnosis(valid, set(domain.classes), {"case_1"}).root_cause == domain.classes[0]
    )
    with pytest.raises(ValueError, match="active taxonomy"):
        validate_diagnosis({**valid, "root_cause": "NOT_A_CLASS"}, set(domain.classes), {"case_1"})
    with pytest.raises(ValueError, match="not present"):
        validate_diagnosis({**valid, "citations": ["missing"]}, set(domain.classes), {"case_1"})
    grammar = build_gbnf_grammar(domain.classes, ["case_1"])
    assert domain.classes[0] in grammar


def test_core_has_no_domain_or_evaluator_imports():
    """Modules on the model's input path never import evaluator labels or domain data."""

    source_root = ROOT / "src" / "groundedx"
    core = [
        *(source_root / "encoding").rglob("*.py"),
        *(source_root / "retrieval").rglob("*.py"),
        *(source_root / "generation").rglob("*.py"),
        source_root / "kb" / "build_kb.py",
    ]
    offenders = []
    for path in core:
        text = path.read_text(encoding="utf-8")
        if re.search(
            r"import .*scoring_labels|scoring_labels import|fault_class|domains[./]", text
        ):
            offenders.append(str(path))
    assert offenders == []


def test_simulator_is_balanced_for_both_domains():
    for domain_name in ("oran", "sre-k8s"):
        domain = load_domain_bundle(ROOT / "domains" / domain_name)
        samples = generate_dataset(domain, 600, seed=17)
        counts = {name: 0 for name in domain.classes}
        for sample in samples:
            counts[sample["fault_class"]] += 1
        assert sum(bool(sample["alarms"]) for sample in samples) > len(samples) // 2
        assert max(counts.values()) - min(counts.values()) <= 1
