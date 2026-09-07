from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from common import load_split, summarize_window, taxonomy_classes
from kb_builder import load_retriever


class Diagnosis(BaseModel):
    root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    remediation: str
    explanation: str
    citations: list[str]
    latency_ms: float | None = None
    raw_output: str | None = None


def build_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    taxonomy = ", ".join(taxonomy_classes())
    evidence = "\n\n".join(
        f"[{c['chunk_id']}] doc_type={c['doc_type']} score={c.get('score', 0):.3f}\n{c['text'][:1200]}"
        for c in chunks
    )
    return f"""You are RAN-Doc, a grounded O-RAN fault diagnosis assistant.
Return strict JSON with keys root_cause, confidence, remediation, explanation, citations.
root_cause must be one of: {taxonomy}
Only cite evidence chunk IDs that appear below.

Retrieved evidence:
{evidence}

Observation window:
{query}
"""


def build_zero_shot_prompt(query: str) -> str:
    taxonomy = ", ".join(taxonomy_classes())
    return f"""You are RAN-Doc, an O-RAN fault diagnosis assistant.
Return strict JSON with keys root_cause, confidence, remediation, explanation, citations.
root_cause must be one of: {taxonomy}
No retrieved evidence is available, so citations must be an empty list.

Observation window:
{query}
"""


def classify_from_text(query: str, chunks: list[dict[str, Any]] | None = None) -> tuple[str, float, list[str], str]:
    classes = taxonomy_classes()
    scores = {label: 0.0 for label in classes}
    chunks = chunks or []
    combined = query + "\n" + "\n".join(c["text"] for c in chunks)
    for label in classes:
        if re.search(rf"\b{re.escape(label)}\b", combined):
            scores[label] += 1.2
    for chunk in chunks:
        text = chunk["text"]
        for label in classes:
            if re.search(rf"\b{re.escape(label)}\b", text):
                scores[label] += float(chunk.get("score", 0.0)) + 0.5
    alarm_rules = {
        "ALM-PCI-CONFLICT": "PCI_COLLISION",
        "ALM-BH-LATENCY": "BACKHAUL_CONGESTION",
        "ALM-PACKET-LOSS": "BACKHAUL_CONGESTION",
        "ALM-RRU-DEGRADE": "HARDWARE_DEGRADATION",
        "ALM-UL-INTERFERENCE": "INTERFERENCE_SPIKE",
        "ALM-NBR-MISSING": "NEIGHBOR_LIST_MISCONFIG",
        "ALM-VNF-CPU-HIGH": "VNF_CPU_EXHAUSTION",
        "ALM-A3-OFFSET": "HANDOVER_PARAM_MISCONFIG",
        "ALM-PA-FAULT": "POWER_AMPLIFIER_FAULT",
        "ALM-PTP-UNLOCK": "TIMING_SYNC_LOSS",
        "ALM-CELL-OVERLOAD": "OVERLOAD_ADMISSION_CONTROL",
        "ALM-NG-FLAP": "S1_NG_INTERFACE_FLAP",
        "ALM-DNS-FAIL": "DNS_RESOLUTION_FAILURE",
        "ALM-QOS-POLICY": "SLICE_QOS_POLICY_ERROR",
        "ALM-FH-JITTER": "FRONTHAUL_JITTER",
        "ALM-CONFIG-CHANGE": "CONFIG_ROLLBACK_REQUIRED",
        "ALM-TEMP-HIGH": "COOLING_SYSTEM_FAILURE",
    }
    for alarm, label in alarm_rules.items():
        if alarm in query:
            scores[label] += 0.7
    root, best = max(scores.items(), key=lambda item: item[1])
    total = sum(max(v, 0.0) for v in scores.values())
    confidence = 0.35 if total == 0 else min(0.95, 0.35 + 0.60 * best / total)
    citations = [
        c["chunk_id"]
        for c in chunks
        if re.search(rf"\b{re.escape(root)}\b", c["text"])
    ][:3]
    remediation = "Review cited evidence and apply the closest matching operations runbook."
    for c in chunks:
        if re.search(rf"\b{re.escape(root)}\b", c["text"]) and "Remediation:" in c["text"]:
            remediation = c["text"].split("Remediation:", 1)[1].strip().splitlines()[0]
            break
    return root, confidence, citations, remediation


def deterministic_rag(sample: dict[str, Any], k: int = 5) -> Diagnosis:
    retriever = load_retriever()
    query = summarize_window(sample)
    chunks = retriever.retrieve(query, k=k)
    root, confidence, citations, remediation = classify_from_text(query, chunks)
    return Diagnosis(
        root_cause=root,
        confidence=float(confidence),
        remediation=remediation,
        explanation=f"The observation matches retrieved evidence for {root}: {', '.join(citations)}.",
        citations=citations,
    )


def zero_shot_diagnose(sample: dict[str, Any]) -> Diagnosis:
    query = summarize_window(sample)
    root, confidence, citations, remediation = classify_from_text(query, [])
    return Diagnosis(
        root_cause=root,
        confidence=float(confidence),
        remediation=remediation,
        explanation=f"The observation was classified as {root} without retrieved evidence.",
        citations=citations,
    )


def llama_diagnose(sample: dict[str, Any], model_path: Path, k: int = 5) -> Diagnosis:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("llama-cpp-python is not installed. Use deterministic mode or install it.") from exc

    retriever = load_retriever()
    query = summarize_window(sample)
    chunks = retriever.retrieve(query, k=k)
    prompt = build_prompt(query, chunks)
    llm = Llama(model_path=str(model_path), n_ctx=4096, n_gpu_layers=35, verbose=False)
    start = time.perf_counter()
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=384,
        response_format={"type": "json_object"},
    )
    latency_ms = (time.perf_counter() - start) * 1000
    raw = output["choices"][0]["message"]["content"]
    try:
        parsed = Diagnosis.model_validate_json(raw)
    except ValidationError:
        parsed = Diagnosis(**json.loads(raw))
    parsed.latency_ms = latency_ms
    parsed.raw_output = raw
    return parsed


class LlamaDiagnoser:
    def __init__(self, model_path: str | Path, n_gpu_layers: int = 0, n_ctx: int = 4096):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python is not installed.") from exc
        self.retriever = load_retriever()
        self.llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)

    def _complete(self, prompt: str) -> Diagnosis:
        start = time.perf_counter()
        output = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        latency_ms = (time.perf_counter() - start) * 1000
        raw = output["choices"][0]["message"]["content"]
        try:
            parsed = Diagnosis.model_validate_json(raw)
        except Exception:
            parsed = Diagnosis(**json.loads(raw))
        if parsed.root_cause not in taxonomy_classes():
            fallback, confidence, citations, remediation = classify_from_text(prompt, [])
            parsed.root_cause = fallback
            parsed.confidence = min(parsed.confidence, confidence)
            parsed.citations = citations
            parsed.remediation = parsed.remediation or remediation
        parsed.latency_ms = latency_ms
        parsed.raw_output = raw
        return parsed

    def rag(self, sample: dict[str, Any], k: int = 5) -> Diagnosis:
        query = summarize_window(sample)
        chunks = self.retriever.retrieve(query, k=k)
        return self._complete(build_prompt(query, chunks))

    def zero_shot(self, sample: dict[str, Any]) -> Diagnosis:
        query = summarize_window(sample)
        return self._complete(build_zero_shot_prompt(query))


def diagnose(sample: dict[str, Any], model: str | None = None, k: int = 5) -> Diagnosis:
    if model:
        return llama_diagnose(sample, Path(model), k=k)
    return deterministic_rag(sample, k=k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="")
    parser.add_argument("--mode", choices=["rag", "zero-shot"], default="rag")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    samples = load_split(args.split)[: args.sample]
    for sample in samples:
        result = zero_shot_diagnose(sample) if args.mode == "zero-shot" else diagnose(sample, args.model or None, k=args.k)
        print(json.dumps({"id": sample["id"], "gold": sample["fault_class"], **result.model_dump()}, indent=2))


if __name__ == "__main__":
    main()
