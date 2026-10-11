"""Evaluator-side checks of SLM explanations (supervisor review, item C).

For every test-split SLM run that produced citations, per window:
- grounded: the paper's 3-condition lexical metric (as stored by run_llm.py);
- ctrl_generic: the same metric for one fixed generic sentence citing the same IDs;
- ctrl_shuffled: the metric for another window's explanation with this window's
  citations and evidence (negative controls: a useful metric should drop sharply);
- cite_any/all_consistent, cite_frac_consistent: share of cited cases whose
  evaluator-side class equals the predicted class;
- remed_matches_pred / remed_matches_gold: generated remediation's closest runbook
  (term Jaccard over the 16 runbooks) is the predicted / gold class's runbook;
- entity_mentions, entity_unsupported: KPI names / alarm codes mentioned in the
  explanation, and whether any of them is absent from the window's own observation
  text (a cheap hallucination check);
- grounded_and_wrong: grounded explanation for a wrong diagnosis.

Writes results/v2/tables/explanation_checks_per_seed.csv (one row per run x seed).
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundedx.encoding.context_encoder import encode_context  # noqa: E402
from groundedx.experiment import RESULTS_ROOT, prepare_seed, read_jsonl  # noqa: E402
from groundedx.faithfulness.grounding_metric import _terms, grounding_check  # noqa: E402

GENERIC = "The alarms and KPI anomalies observed in this window indicate the diagnosed fault."
_SETUPS: dict[tuple, object] = {}


def setup_for(seed: int, manifest: dict):
    key = (
        seed,
        manifest.get("kb_variant") or "full",
        manifest.get("data_variant") or "default",
        manifest.get("kb_size"),
        manifest.get("ranking") or "raw",
    )
    if key not in _SETUPS:
        _SETUPS[key] = prepare_seed(
            seed, key[1], data_variant=key[2], kb_size=key[3], ranking=key[4]
        )
    return _SETUPS[key]


def entities(text: str, metric_names: list[str]) -> set[str]:
    low = text.lower()
    found = {m for m in metric_names if m in low or m.replace("_", " ") in low}
    found |= set(re.findall(r"ALM-[A-Z0-9-]+", text))
    return found


def main() -> None:
    rows = []
    rng = random.Random(0)
    for exp_dir in sorted(RESULTS_ROOT.glob("llm__*")):
        if exp_dir.name.endswith("__val") or "__zs" in exp_dir.name:
            continue
        for seed_dir in sorted(exp_dir.glob("seed_*")):
            preds = read_jsonl(seed_dir / "predictions.jsonl")
            if not preds:
                continue
            seed = int(seed_dir.name.split("_")[1])
            manifest = json.loads((seed_dir / "manifest.json").read_text())
            setup = setup_for(seed, manifest)
            domain = setup.domain
            texts = {d["chunk_id"]: d["text"] for d in setup.documents}
            runbook = {f.name: f.remediation for f in domain.taxonomy}
            window = {s["id"]: s for s in setup.splits[manifest.get("split", "test")]}
            shuffled = preds[:]
            rng.shuffle(shuffled)
            acc: dict[str, list] = {}

            def add(name, value):
                acc.setdefault(name, []).append(float(value))

            for r, other in zip(preds, shuffled):
                if not r["valid"]:
                    continue
                pred, cits = r["prediction"], r["citations"] or []
                evidence = {x["id"]: texts[x["id"]] for x in r["retrieved"]}
                add("grounded", r["grounded"])
                add("grounded_and_wrong", r["grounded"] and pred != r["gold"])
                add("ctrl_generic", bool(cits) and grounding_check(GENERIC, cits, evidence))
                add(
                    "ctrl_shuffled",
                    bool(cits) and grounding_check(other["explanation"] or "", cits, evidence),
                )
                if cits:
                    consistent = [setup.scoring_labels[c] == pred for c in cits]
                    add("cite_any_consistent", any(consistent))
                    add("cite_all_consistent", all(consistent))
                    add("cite_frac_consistent", sum(consistent) / len(consistent))
                rem = _terms(r["remediation"] or "")
                best = max(
                    runbook,
                    key=lambda c: (
                        len(rem & _terms(runbook[c])) / max(1, len(rem | _terms(runbook[c])))
                    ),
                )
                add("remed_matches_pred", best == pred)
                add("remed_matches_gold", best == r["gold"])
                observation = encode_context(window[r["id"]], domain, ranking=setup.ranking)
                mentioned = entities(r["explanation"] or "", domain.metric_names)
                add("entity_mentions", len(mentioned) > 0)
                if mentioned:
                    present = entities(observation, domain.metric_names)
                    add("entity_unsupported", bool(mentioned - present))
            rows.append(
                {"experiment": exp_dir.name, "seed": seed, "n": len(preds)}
                | {k: sum(v) / len(v) for k, v in acc.items()}
            )
            print(f"{exp_dir.name} seed {seed}: done", flush=True)
    out = RESULTS_ROOT / "tables" / "explanation_checks_per_seed.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
