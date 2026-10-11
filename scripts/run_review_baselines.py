"""CPU baselines for the review controls (matching the Kaggle 'review' SLM runs).

For each data / KB / encoding variant: retrieval ranks (Recall@k, MRR@k) and the
retrieval-only kNN vote; for data variants also rule, RandomForest and GBDT
(hyper-parameters re-selected on that variant's validation split).
Experiment names follow groundedx.calls.experiment_name suffixes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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
from groundedx.encoding.context_encoder import encode_context  # noqa: E402
from groundedx.experiment import (  # noqa: E402
    load_experiment_config,
    prepare_seed,
    run_dir,
    write_manifest,
)

_spec = importlib.util.spec_from_file_location("rb", ROOT / "scripts" / "run_baselines.py")
_rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rb)
write_predictions = _rb.write_predictions

# (kb_variant, data_variant, kb_size, ranking)
RETRIEVAL_CONFIGS = [
    ("label_exposed", "default", None, "raw"),
    ("label_exposed", "leaky", None, "raw"),
    ("full", "leaky", None, "raw"),
    ("full", "default", "all", "raw"),
    ("full", "default", None, "z"),
    ("full", "default", "all", "z"),
    ("full", "easy", None, "raw"),
    ("full", "mid", None, "raw"),
]
CLASSIFIER_DATA = ["leaky", "easy", "mid"]


def suffix(data_variant: str, kb_size: str | None, ranking: str) -> str:
    out = ""
    if data_variant != "default":
        out += f"__data-{data_variant}"
    if kb_size:
        out += f"__kb{kb_size}"
    if ranking != "raw":
        out += f"__enc-{ranking}"
    return out


def main() -> None:
    cfg = load_experiment_config()
    k = int(cfg["retrieval"]["k"])
    ks = [int(v) for v in cfg["retrieval"]["ablation_k"]]
    for seed in cfg["seeds"]:
        for data_variant in CLASSIFIER_DATA:
            setup = prepare_seed(seed, "full", cfg, data_variant=data_variant)
            dom, train, val, test = (
                setup.domain,
                *(setup.splits[s] for s in ("train", "val", "test")),
            )
            sfx = suffix(data_variant, None, "raw")
            write_predictions(f"rule{sfx}", seed, test, [rule_based_predict(s, dom) for s in test],
                              data_variant=data_variant)  # fmt: skip
            rf = train_random_forest(train, dom, seed)
            x_tr, _ = extract_features(train, dom)
            x_te, _ = extract_features(test, dom)
            write_predictions(f"random_forest{sfx}", seed, test,
                              list(rf.predict(x_te.reindex(columns=x_tr.columns, fill_value=0.0))),
                              data_variant=data_variant)  # fmt: skip
            gbdt, best, grid = train_gbdt(train, val, dom, seed)
            write_predictions(f"gbdt{sfx}", seed, test, predict_gbdt(gbdt, test, dom),
                              data_variant=data_variant, selected=best, val_grid=grid)  # fmt: skip
            print(f"seed {seed} {data_variant}: classifiers done", flush=True)
        for kb_variant, data_variant, kb_size, ranking in RETRIEVAL_CONFIGS:
            setup = prepare_seed(seed, kb_variant, cfg, data_variant=data_variant, kb_size=kb_size,
                                 ranking=ranking)  # fmt: skip
            test = setup.splits["test"]
            sfx = suffix(data_variant, kb_size, ranking)
            retrieved = [
                setup.retriever.retrieve(
                    encode_context(s, setup.domain, ranking=ranking), k=max(ks)
                )
                for s in test
            ]
            preds = [knn_vote(rows[:k], setup.scoring_labels) for rows in retrieved]
            meta = dict(kb_variant=kb_variant, data_variant=data_variant, kb_size=kb_size,
                        ranking=ranking, k=k, n_kb_docs=len(setup.documents))  # fmt: skip
            write_predictions(f"knn_vote__{kb_variant}__k{k}{sfx}", seed, test, preds, **meta)
            out_dir = run_dir(f"retrieval__{kb_variant}{sfx}", seed)
            with (out_dir / "ranks.jsonl").open("w", encoding="utf-8") as handle:
                for sample, rows in zip(test, retrieved):
                    labels = [setup.scoring_labels[r["chunk_id"]] for r in rows]
                    gold = sample["fault_class"]
                    rank = labels.index(gold) + 1 if gold in labels else None
                    handle.write(json.dumps({"id": sample["id"], "same_class_rank": rank}) + "\n")
            write_manifest(out_dir, seed, max_k=max(ks), **meta)
            print(f"seed {seed} retrieval {kb_variant}{sfx}: done", flush=True)


if __name__ == "__main__":
    main()
