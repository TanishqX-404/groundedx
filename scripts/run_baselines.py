"""CPU baselines and retrieval metrics (E1 baselines, E7 retrieval half).

Experiments written to results/v2/<name>/seed_<s>/:
  rule, random_forest, gbdt, knn_vote__<kb_variant>__k5, retrieval__<kb_variant>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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


def write_predictions(exp: str, seed: int, samples: list[dict], preds: list[str], **meta) -> None:
    out_dir = run_dir(exp, seed)
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for sample, pred in zip(samples, preds):
            handle.write(
                json.dumps(
                    {
                        "id": sample["id"],
                        "gold": sample["fault_class"],
                        "secondary_class": sample["secondary_class"],
                        "true_alarm_emitted": sample["true_alarm_emitted"],
                        "prediction": pred,
                        "valid": True,
                    }
                )
                + "\n"
            )
    write_manifest(out_dir, seed, n_windows=len(samples), **meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    cfg = load_experiment_config()
    k = int(cfg["retrieval"]["k"])
    ks = [int(v) for v in cfg["retrieval"]["ablation_k"]]

    for seed in args.seeds or cfg["seeds"]:
        base = prepare_seed(seed, "full", cfg)
        domain, train, val, test = base.domain, *(base.splits[s] for s in ("train", "val", "test"))
        write_predictions("rule", seed, test, [rule_based_predict(s, domain) for s in test])

        rf = train_random_forest(train, domain, seed)
        x_train, _ = extract_features(train, domain)
        x_test, _ = extract_features(test, domain)
        rf_pred = list(rf.predict(x_test.reindex(columns=x_train.columns, fill_value=0.0)))
        write_predictions("random_forest", seed, test, rf_pred, n_estimators=300)

        gbdt, best, grid = train_gbdt(train, val, domain, seed)
        write_predictions(
            "gbdt", seed, test, predict_gbdt(gbdt, test, domain), selected=best, val_grid=grid
        )

        for variant in cfg["kb"]["variants"]:
            setup = base if variant == "full" else prepare_seed(seed, variant, cfg)
            assert [d["chunk_id"] for d in setup.documents] == [
                d["chunk_id"] for d in base.documents
            ]
            retrieved = [
                setup.retriever.retrieve(encode_context(s, domain), k=max(ks)) for s in test
            ]
            preds = [knn_vote(rows[:k], setup.scoring_labels) for rows in retrieved]
            write_predictions(
                f"knn_vote__{variant}__k{k}", seed, test, preds, kb_variant=variant, k=k
            )

            out_dir = run_dir(f"retrieval__{variant}", seed)
            with (out_dir / "ranks.jsonl").open("w", encoding="utf-8") as handle:
                for sample, rows in zip(test, retrieved):
                    labels = [setup.scoring_labels[r["chunk_id"]] for r in rows]
                    rank = (
                        labels.index(sample["fault_class"]) + 1
                        if sample["fault_class"] in labels
                        else None
                    )
                    handle.write(json.dumps({"id": sample["id"], "same_class_rank": rank}) + "\n")
            write_manifest(
                out_dir, seed, kb_variant=variant, max_k=max(ks), n_kb_docs=len(setup.documents)
            )

        stats_dir = run_dir("dataset", seed)
        stats = {
            split: {
                "n": len(rows),
                "blended": sum(r["secondary_class"] is not None for r in rows),
                "true_alarm_emitted": sum(r["true_alarm_emitted"] for r in rows),
                "with_distractor_alarm": sum(bool(r["distractor_alarms"]) for r in rows),
                "per_class": {c: sum(r["fault_class"] == c for r in rows) for c in domain.classes},
            }
            for split, rows in base.splits.items()
        }
        stats["kb_docs"] = len(base.documents)
        (stats_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        with (stats_dir / "kb_full.jsonl").open("w", encoding="utf-8") as handle:
            for doc in base.documents:
                handle.write(json.dumps(doc) + "\n")
        (stats_dir / "scoring_labels.json").write_text(
            json.dumps(base.scoring_labels, indent=2), encoding="utf-8"
        )
        write_manifest(stats_dir, seed)
        print(f"seed {seed}: baselines done", flush=True)


if __name__ == "__main__":
    main()
