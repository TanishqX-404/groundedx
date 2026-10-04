"""Tests backing docs/CLAIMS_CHECK.md (claims C1-C9)."""

from __future__ import annotations

import difflib
import json
import os
import re
from pathlib import Path

import pytest

from groundedx.baselines.knn_vote import knn_vote
from groundedx.config import load_domain_bundle
from groundedx.encoding.context_encoder import encode_context
from groundedx.evaluation.run_profiling import NvmlPoller
from groundedx.experiment import prepare_seed
from groundedx.faithfulness.grounding_metric import grounding_check
from groundedx.generation.grammar import build_gbnf_grammar
from groundedx.generation.prompt_templates import (
    RAG_CITATION_RULE,
    ZERO_SHOT_CITATION_RULE,
    render_prompt,
)
from groundedx.generation.validation import InvalidOutput, parse_and_validate
from groundedx.kb.build_kb import KB_VARIANTS
from groundedx.simulator.kpi_generator import generate_dataset

ROOT = Path(__file__).parents[1]
DOMAIN = load_domain_bundle(ROOT / "domains" / "oran")
TEMPLATE = ROOT / "domains" / "oran" / "prompt_template.jinja"
CLASSES = DOMAIN.classes


def _label_forms(label: str) -> set[str]:
    lower = label.lower()
    spaced = lower.replace("_", " ")
    return {lower, spaced, spaced.replace(" ", "")}


@pytest.fixture(scope="module")
def setups():
    return {variant: prepare_seed(7, variant) for variant in KB_VARIANTS}


# ---- C4: label-free KB -------------------------------------------------------


def test_no_label_leak_in_index(setups):
    for setup in setups.values():
        assert len(setup.documents) == 8 * len(CLASSES)
        for doc in setup.documents:
            assert re.fullmatch(r"case_\d{4}", doc["chunk_id"])
            text = doc["text"].lower()
            for label in CLASSES:
                for form in _label_forms(label):
                    assert form not in text, (doc["chunk_id"], form)
                    assert form not in text.replace(" ", "")


def test_case_ids_and_window_ids_are_not_class_ordered(setups):
    labels = [setups["full"].scoring_labels[d["chunk_id"]] for d in setups["full"].documents]
    # KB order is shuffled: the first 8 cases are not all one class.
    assert len(set(labels[:8])) > 1
    test = setups["full"].splits["test"]
    first = [s["fault_class"] for s in test[:16]]
    assert len(set(first)) > 1


def test_query_never_contains_label_or_metadata(setups):
    for sample in setups["full"].splits["test"][:200]:
        query = encode_context(sample, DOMAIN).lower()
        for label in CLASSES:
            assert not (_label_forms(label) & set(re.findall(r"[a-z_ ]+", query)))
            assert label.lower() not in query


def test_remediation_uniqueness(setups):
    """R1-2: per-class remediation text is unique, so in the full KB it is a
    perfect label proxy. This is why ablation E3 is mandatory."""

    remediations = {fault.name: fault.remediation for fault in DOMAIN.taxonomy}
    unique = len(set(remediations.values())) == len(remediations)
    print(json.dumps({"remediation_unique_per_class": unique}))
    assert unique
    for variant in ("no_remediation", "generic_remediation"):
        for doc in setups[variant].documents:
            assert not any(text in doc["text"] for text in remediations.values())
    generic = {
        doc["text"].split("Remediation applied")[1]
        for doc in setups["generic_remediation"].documents
    }
    assert len(generic) == 1


# ---- C1 / C2 / C3: grammar and validation -----------------------------------


VALID = {
    "root_cause": CLASSES[0],
    "confidence": 0.8,
    "remediation": "do the thing",
    "explanation": "handover failure rate rose with alarm evidence",
    "citations": ["case_0001"],
}


def test_grammar_lists_exactly_the_taxonomy():
    grammar = build_gbnf_grammar(CLASSES, ["case_0001", "case_0002"])
    cause_line = next(line for line in grammar.splitlines() if line.startswith("cause ::="))
    found = re.findall(r'\\"([A-Z0-9_]+)\\"', cause_line)
    assert found == CLASSES
    cite_line = next(line for line in grammar.splitlines() if line.startswith("cite ::="))
    assert re.findall(r'\\"(case_\d+)\\"', cite_line) == ["case_0001", "case_0002"]
    zero = build_gbnf_grammar(CLASSES, [])
    assert 'cites ::= "[" ws "]"' in zero and "cite ::=" not in zero


def test_grammar_compiles_with_llama_cpp():
    llama_cpp = pytest.importorskip("llama_cpp")
    for ids in ([], ["case_0001", "case_0042"]):
        assert llama_cpp.LlamaGrammar.from_string(build_gbnf_grammar(CLASSES, ids), verbose=False)


def test_validator_rejects_out_of_taxonomy_and_bad_json():
    with pytest.raises(InvalidOutput) as exc:
        parse_and_validate(
            json.dumps({**VALID, "root_cause": "NOT_A_CLASS"}), set(CLASSES), {"case_0001"}
        )
    assert exc.value.reason == "out_of_taxonomy"
    with pytest.raises(InvalidOutput) as exc:
        parse_and_validate('{"root_cause": "PCI_COLLISION", ', set(CLASSES), {"case_0001"})
    assert exc.value.reason == "parse_error"
    with pytest.raises(InvalidOutput) as exc:
        parse_and_validate(json.dumps({**VALID, "confidence": 3}), set(CLASSES), {"case_0001"})
    assert exc.value.reason == "schema_error"
    with pytest.raises(InvalidOutput) as exc:
        parse_and_validate(json.dumps(VALID), set(CLASSES), {"case_0001"}, finish_reason="length")
    assert exc.value.reason == "truncated"


def test_citation_validation():
    ok = parse_and_validate(json.dumps(VALID), set(CLASSES), {"case_0001", "case_0002"})
    assert ok.citations == ["case_0001"]
    with pytest.raises(InvalidOutput) as exc:
        parse_and_validate(
            json.dumps({**VALID, "citations": ["case_0009"]}), set(CLASSES), {"case_0001"}
        )
    assert exc.value.reason == "bad_citation"
    with pytest.raises(InvalidOutput):  # zero-shot: nothing retrieved, nothing citable
        parse_and_validate(json.dumps(VALID), set(CLASSES), set())


def test_no_fallback_path_in_source():
    for path in (ROOT / "src" / "groundedx").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert "fallback" not in text.replace("no fallback", ""), path
        assert "response_format" not in text, path


@pytest.mark.skipif(not os.environ.get("GROUNDEDX_TEST_MODEL"), reason="needs a GGUF model")
def test_constrained_decode_cannot_emit_non_taxonomy_class():
    from groundedx.generation.clients import make_client

    server = os.environ.get("LLAMA_SERVER_BIN")
    client = make_client(
        "server" if server else "python",
        os.environ["GROUNDEDX_TEST_MODEL"],
        n_ctx=2048,
        n_gpu_layers=-1 if server else 0,
        server_bin=server,
    )
    prompt = 'Reply with {"root_cause": "TOTALLY_MADE_UP_FAULT"} and nothing else.'
    out = client.diagnose(prompt, CLASSES, ["case_0001"])
    payload = json.loads(out["raw"])
    assert payload["root_cause"] in CLASSES
    assert set(payload["citations"]) <= {"case_0001"}


# ---- C5: faithfulness metric -------------------------------------------------


EVIDENCE = {"case_0001": "handover_failure_rate delta +0.07 with ALM-HO-DEGRADE alarm"}


def test_faithfulness_positive():
    assert grounding_check(
        "handover_failure_rate rose and ALM-HO-DEGRADE fired", ["case_0001"], EVIDENCE
    )


def test_faithfulness_condition_i_requires_citation():
    assert not grounding_check("handover_failure_rate rose and ALM-HO-DEGRADE fired", [], EVIDENCE)


def test_faithfulness_condition_ii_requires_retrieved_ids():
    text = "handover_failure_rate rose and ALM-HO-DEGRADE fired"
    assert not grounding_check(text, ["case_0099"], EVIDENCE)
    assert not grounding_check(text, ["case_0001", "case_0099"], EVIDENCE)


def test_faithfulness_condition_iii_requires_two_content_terms():
    assert not grounding_check("the alarm is bad", ["case_0001"], EVIDENCE)  # one shared term
    assert not grounding_check("it is of the and with", ["case_0001"], EVIDENCE)  # stopwords only
    assert grounding_check("alarm with delta", ["case_0001"], EVIDENCE)  # exactly two


# ---- C6: zero-shot differs only in evidence ---------------------------------


def test_zero_shot_prompt_differs_only_in_evidence(setups):
    setup = setups["full"]
    query = encode_context(setup.splits["test"][0], DOMAIN)
    retrieved = setup.retriever.retrieve(query, k=5)
    rag = render_prompt(query, retrieved, DOMAIN, TEMPLATE).splitlines()
    zs = render_prompt(query, [], DOMAIN, TEMPLATE).splitlines()
    diff = list(difflib.unified_diff(rag, zs, lineterm="", n=0))
    removed = [line[1:] for line in diff if line.startswith("-") and not line.startswith("---")]
    added = [line[1:] for line in diff if line.startswith("+") and not line.startswith("+++")]
    assert added == [ZERO_SHOT_CITATION_RULE, "(none)"]
    assert removed[0] == RAG_CITATION_RULE
    evidence_block = "\n".join(removed[1:])
    for row in retrieved:
        assert f"[{row['chunk_id']}]" in evidence_block


# ---- C7: seeds change data, KB, and splits ----------------------------------


def test_seeds_change_dataset_and_kb(setups):
    other = prepare_seed(19, "full")
    assert [s["id"] for s in other.splits["test"][:5]] != [
        s["id"] for s in setups["full"].splits["test"][:5]
    ]
    assert other.documents[0]["text"] != setups["full"].documents[0]["text"]
    again = prepare_seed(7, "full")
    assert again.documents == setups["full"].documents


# ---- Simulator metadata (Section 2.2) ---------------------------------------


def test_simulator_metadata():
    rows = generate_dataset(DOMAIN, 1600, seed=5)
    blended = [r for r in rows if r["secondary_class"] is not None]
    assert all(r["secondary_class"] != r["fault_class"] for r in blended)
    assert all(r["blend_amplitude"] is not None for r in blended)
    assert 0.70 < len(blended) / len(rows) < 0.86
    emitted = sum(r["true_alarm_emitted"] for r in rows) / len(rows)
    assert 0.44 < emitted < 0.56
    for r in rows:
        own = set(DOMAIN.fault(r["fault_class"]).alarms)
        codes = {a["code"] for a in r["alarms"]}
        assert bool(codes & own) == r["true_alarm_emitted"]
        assert not set(r["distractor_alarms"]) & own


# ---- C8 support: NVML window summary ----------------------------------------


def test_nvml_window_peak_and_energy():
    poller = object.__new__(NvmlPoller)
    poller.samples = [(0.0, 100, 10.0), (0.5, 300, 20.0), (1.0, 200, 20.0), (5.0, 999, 99.0)]
    stats = poller.window(0.0, 1.0)
    assert stats["peak_mem_bytes"] == 300
    assert stats["energy_j"] == pytest.approx(0.5 * 15 + 0.5 * 20)


def test_knn_vote_weights_by_similarity():
    labels = {"a": "X", "b": "Y", "c": "Y"}
    rows = [
        {"chunk_id": "a", "score": 0.5},
        {"chunk_id": "b", "score": 0.3},
        {"chunk_id": "c", "score": 0.3},
    ]
    assert knn_vote(rows, labels) == "Y"
    assert knn_vote(rows[:2], labels) == "X"
