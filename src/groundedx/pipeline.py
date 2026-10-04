"""Small public pipeline entry points and a CPU smoke CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from groundedx.config import load_domain_bundle
from groundedx.encoding.context_encoder import encode_context
from groundedx.kb.build_kb import build_documents
from groundedx.retrieval.tfidf_retriever import TfidfRetriever
from groundedx.simulator.kpi_generator import generate_dataset


def diagnose(sample: dict, domain_path: str | Path, documents: list[dict], k: int = 5) -> dict:
    """Return a retrieval result for one sample; generation is optional."""

    domain = load_domain_bundle(domain_path)
    query = encode_context(sample, domain)
    retrieved = TfidfRetriever(documents).retrieve(query, k=k)
    return {"query": query, "retrieved": retrieved}


def main() -> None:
    parser = argparse.ArgumentParser(description="GroundedX domain smoke run")
    parser.add_argument("--domain", default="domains/oran")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()
    domain = load_domain_bundle(args.domain)
    rows = generate_dataset(domain, args.n, args.seed)
    documents = build_documents(rows[:32], domain)
    result = diagnose(rows[0], args.domain, documents, args.k)
    print(
        json.dumps(
            {"domain": domain.name, "classes": len(domain.classes), "result": result}, indent=2
        )
    )


if __name__ == "__main__":
    main()
