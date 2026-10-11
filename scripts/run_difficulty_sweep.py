"""Classical-baseline difficulty sweep (supervisor review, item E).

Varies the window-level probabilities that set benchmark difficulty and records
rule / RandomForest / GBDT accuracy and TF-IDF retrieval Recall@5 / kNN-vote
accuracy, 3 seeds per grid point. GBDT hyper-parameters are re-selected on each
point's validation split. Writes results/v2/tables/difficulty_sweep_per_seed.csv.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from groundedx.baselines.knn_vote import knn_vote  # noqa: E402
from groundedx.baselines.random_forest import (  # noqa: E402
    extract_features,
    predict_gbdt,
    train_gbdt,
    train_random_forest,
)
from groundedx.baselines.rule_based import rule_based_predict  # noqa: E402
from groundedx.config import load_domain_bundle  # noqa: E402
from groundedx.encoding.context_encoder import encode_context  # noqa: E402
from groundedx.kb.build_kb import build_documents  # noqa: E402
from groundedx.kb.scoring_labels import make_scoring_labels, select_kb_cases  # noqa: E402
from groundedx.retrieval.tfidf_retriever import TfidfRetriever  # noqa: E402
from groundedx.simulator.kpi_generator import generate_dataset, split_dataset  # noqa: E402

TRUE_ALARM = [0.3, 0.5, 0.7, 0.9]
BLEND = [0.3, 0.55, 0.78]
EXTRA_ALARM = 0.82
SEEDS = [7, 19, 41]


def acc(pred, gold) -> float:
    return float(np.mean([p == g for p, g in zip(pred, gold)]))


def main() -> None:
    base = load_domain_bundle(ROOT / "domains" / "oran")
    out = ROOT / "results" / "v2" / "tables" / "difficulty_sweep_per_seed.csv"
    rows = []
    for true_alarm, blend in itertools.product(TRUE_ALARM, BLEND):
        dom = base.model_copy(
            update={
                "true_alarm_probability": true_alarm,
                "distractor_probability": blend,
                "extra_alarm_probability": EXTRA_ALARM,
            }
        )
        for seed in SEEDS:
            splits = split_dataset(generate_dataset(dom, 4000, seed))
            train, val, test = splits["train"], splits["val"], splits["test"]
            gold = [s["fault_class"] for s in test]
            rf = train_random_forest(train, dom, seed)
            x_tr, _ = extract_features(train, dom)
            x_te, _ = extract_features(test, dom)
            rf_pred = rf.predict(x_te.reindex(columns=x_tr.columns, fill_value=0.0))
            gbdt, best, _ = train_gbdt(train, val, dom, seed)
            kb = select_kb_cases(train, 8, seed)
            docs = build_documents(kb, dom)
            labels = make_scoring_labels(kb, [d["chunk_id"] for d in docs])
            retriever = TfidfRetriever(docs)
            retrieved = [retriever.retrieve(encode_context(s, dom), k=5) for s in test]
            recall5 = np.mean(
                [g in [labels[r["chunk_id"]] for r in rs] for g, rs in zip(gold, retrieved)]
            )
            rows.append(
                {
                    "true_alarm_probability": true_alarm,
                    "distractor_probability": blend,
                    "extra_alarm_probability": EXTRA_ALARM,
                    "seed": seed,
                    "rule": acc([rule_based_predict(s, dom) for s in test], gold),
                    "random_forest": acc(rf_pred, gold),
                    "gbdt": acc(predict_gbdt(gbdt, test, dom), gold),
                    "knn_vote": acc([knn_vote(rs, labels) for rs in retrieved], gold),
                    "recall_at_5": float(recall5),
                    "gbdt_params": str(best),
                }
            )
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"alarm={true_alarm} blend={blend} seed={seed}: gbdt={rows[-1]['gbdt']:.3f}",
                  flush=True)  # fmt: skip
    print("wrote", out)


if __name__ == "__main__":
    main()
