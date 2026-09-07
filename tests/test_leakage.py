from pathlib import Path

from groundedx.config import load_domain_bundle
from groundedx.kb.build_kb import build_documents
from groundedx.simulator.kpi_generator import generate_dataset


def test_kb_builder_output_does_not_contain_labels():
    root = Path(__file__).parents[1]
    builder_source = (root / "src" / "groundedx" / "kb" / "build_kb.py").read_text(encoding="utf-8")
    assert "fault_class" not in builder_source
    domain = load_domain_bundle(root / "domains" / "oran")
    samples = generate_dataset(domain, 16, seed=9)
    documents = build_documents(samples, domain, limit=16)
    for document in documents:
        assert all(label not in document["text"] for label in domain.classes)
