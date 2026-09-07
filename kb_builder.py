from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import KB_DIR, ROOT, load_split, load_taxonomy, summarize_window


INDEX_PATH = KB_DIR / "kb_index.pkl"
CARDS_DIR = KB_DIR / "cards"
CASES_DIR = KB_DIR / "cases"


def card_text(fault: dict[str, Any]) -> str:
    effects = ", ".join(f"{k} changes by {v:+}" for k, v in fault["kpi_effects"].items())
    alarms = ", ".join(fault["alarms"])
    return (
        f"---\n"
        f"doc_type: fault_card\n"
        f"alarm_codes: {alarms}\n"
        f"---\n\n"
        f"# Fault signature card\n\n"
        f"Diagnosis label: {fault['class']}.\n\n"
        f"Domain: {fault['domain']}.\n\n"
        f"Signature: this fault is indicated by alarms {alarms}. "
        f"The expected KPI signature is: {effects}. "
        f"The temporal pattern is usually {fault['pattern']} shaped over a diagnosis window. "
        f"Root cause evidence is strongest when the listed alarms co-occur with the KPI deviations.\n\n"
        f"Remediation: {fault['remediation']}\n"
    )


def write_cards() -> None:
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    for old in CARDS_DIR.glob("*.md"):
        old.unlink()
    for idx, fault in enumerate(load_taxonomy(), start=1):
        (CARDS_DIR / f"card_{idx:03d}.md").write_text(card_text(fault), encoding="utf-8")


def write_cases(cases_per_class: int) -> None:
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    for old in CASES_DIR.glob("*.md"):
        old.unlink()
    train = load_split("train")
    counts: dict[str, int] = {}
    global_case_idx = 0
    for sample in train:
        fault_class = sample["fault_class"]
        if counts.get(fault_class, 0) >= cases_per_class:
            continue
        counts[fault_class] = counts.get(fault_class, 0) + 1
        global_case_idx += 1
        text = (
            f"---\n"
            f"doc_type: historical_case\n"
            f"case_id: {sample['id']}\n"
            f"---\n\n"
            f"# Historical case {sample['id']}\n\n"
            f"{summarize_window(sample)}\n\n"
            f"Remediation applied by operations team: {sample['remediation']}\n"
        )
        (CASES_DIR / f"case_{global_case_idx:04d}.md").write_text(text, encoding="utf-8")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    _, fm, body = text.split("---", 2)
    meta: dict[str, str] = {}
    for line in fm.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, body.strip()


def load_documents() -> list[dict[str, Any]]:
    docs = []
    taxonomy_by_alarm = {}
    for fault in load_taxonomy():
        for alarm in fault["alarms"]:
            taxonomy_by_alarm.setdefault(alarm, []).append(fault["class"])
    card_faults = load_taxonomy()
    card_paths = sorted(CARDS_DIR.glob("*.md"))
    case_label_by_id = {s["id"]: s["fault_class"] for s in load_split("train")} if (ROOT / "data" / "train.pkl").exists() else {}

    for path in sorted([*card_paths, *CASES_DIR.glob("*.md")]):
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        doc_type = meta.get("doc_type", "unknown")
        doc_id = path.stem
        if doc_type == "fault_card":
            match = re.search(r"Diagnosis label:\s*([A-Z0-9_]+)", body)
            hidden_fault_class = match.group(1) if match else card_faults[min(len(docs), len(card_faults) - 1)]["class"]
        else:
            hidden_fault_class = case_label_by_id.get(meta.get("case_id", ""), "UNKNOWN")
        docs.append(
            {
                "chunk_id": doc_id,
                "text": body,
                "doc_type": doc_type,
                "_hidden_fault_class": hidden_fault_class,
            }
        )
    return docs


class TfidfRetriever:
    def __init__(self, docs: list[dict[str, Any]], vectorizer: TfidfVectorizer, matrix: Any):
        self.docs = docs
        self.vectorizer = vectorizer
        self.matrix = matrix

    def retrieve(self, query: str, k: int = 5, include_hidden: bool = False) -> list[dict[str, Any]]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        for idx, doc in enumerate(self.docs):
            if doc["doc_type"] == "fault_card":
                sims[idx] += 0.20
        order = np.argsort(-sims)[:k]
        results = []
        for idx in order:
            row = dict(self.docs[int(idx)])
            row["score"] = float(sims[int(idx)])
            if not include_hidden:
                row.pop("_hidden_fault_class", None)
            results.append(row)
        return results


def build_index() -> TfidfRetriever:
    docs = load_documents()
    if not docs:
        raise RuntimeError("No KB documents found. Run python kb_builder.py --build first.")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=40000)
    matrix = vectorizer.fit_transform([d["text"] for d in docs])
    retriever = TfidfRetriever(docs, vectorizer, matrix)
    with INDEX_PATH.open("wb") as f:
        pickle.dump({"docs": docs, "vectorizer": vectorizer, "matrix": matrix}, f)
    with (KB_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in d.items() if not k.startswith("_hidden")} for d in docs], f, indent=2)
    with (KB_DIR / "scoring_labels.json").open("w", encoding="utf-8") as f:
        json.dump({d["chunk_id"]: d["_hidden_fault_class"] for d in docs}, f, indent=2)
    return retriever


def load_retriever() -> TfidfRetriever:
    with INDEX_PATH.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, TfidfRetriever):
        return payload
    return TfidfRetriever(payload["docs"], payload["vectorizer"], payload["matrix"])


def retrieval_metrics(samples: list[dict[str, Any]], ks: list[int]) -> list[dict[str, float]]:
    retriever = load_retriever()
    rows = []
    for k in ks:
        hits = 0
        reciprocal = []
        for sample in samples:
            results = retriever.retrieve(summarize_window(sample), k=max(ks), include_hidden=True)
            ranks = [
                i + 1
                for i, doc in enumerate(results)
                if doc["_hidden_fault_class"] == sample["fault_class"]
            ]
            if ranks and ranks[0] <= k:
                hits += 1
                reciprocal.append(1.0 / ranks[0])
            else:
                reciprocal.append(0.0)
        rows.append({"k": k, "recall_at_k": hits / len(samples), "mrr": float(np.mean(reciprocal))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--cases-per-class", type=int, default=8)
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    if args.build:
        write_cards()
        write_cases(args.cases_per_class)
        retriever = build_index()
        print(f"Built KB with {len(retriever.docs)} chunks at {INDEX_PATH}")
    if args.query:
        retriever = load_retriever()
        for row in retriever.retrieve(args.query, args.k):
            print(f"{row['chunk_id']} {row['doc_type']} score={row['score']:.3f}")


if __name__ == "__main__":
    main()
