from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from common import RESULTS_DIR, load_split
from evaluate import load_scoring_labels, write_main_table
from rag_diagnose import LlamaDiagnoser


def run_mode(runner: LlamaDiagnoser, samples: list[dict], mode: str, k: int, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoring_labels = load_scoring_labels()
    rows = []
    for i, sample in enumerate(samples, start=1):
        diag = runner.rag(sample, k=k) if mode == "rag" else runner.zero_shot(sample)
        cited_classes = [scoring_labels.get(c) for c in diag.citations]
        citation_fault_match = bool(diag.citations) and all(c == diag.root_cause for c in cited_classes)
        rows.append(
            {
                "id": sample["id"],
                "gold": sample["fault_class"],
                "prediction": diag.root_cause,
                "confidence": diag.confidence,
                "citations": ";".join(diag.citations),
                "citation_fault_match": citation_fault_match,
                "latency_ms": diag.latency_ms,
                "raw_output": diag.raw_output,
            }
        )
        if i % 25 == 0:
            print(f"{mode}: completed {i}/{len(samples)}", flush=True)
    pred_df = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "method": f"{mode.upper()} SLM {model_name}",
                "top1_acc": accuracy_score(pred_df["gold"], pred_df["prediction"]),
                "top3_acc": float("nan"),
                "macro_f1": f1_score(pred_df["gold"], pred_df["prediction"], average="macro", zero_division=0),
                "explanation_faithfulness": float(pred_df["citation_fault_match"].mean()) if mode == "rag" else 0.0,
                "latency_ms": float(pred_df["latency_ms"].mean()),
                "vram_gb": float("nan"),
            }
        ]
    )
    return summary, pred_df


def main() -> None:
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mode", choices=["rag", "zero-shot", "both"], default="both")
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    samples = load_split(args.split)
    if args.limit:
        samples = samples[: args.limit]
    RESULTS_DIR.mkdir(exist_ok=True)
    runner = LlamaDiagnoser(args.model, n_gpu_layers=args.n_gpu_layers)
    summaries = []
    if args.mode in {"zero-shot", "both"}:
        summary, preds = run_mode(runner, samples, "zero-shot", args.k, Path(args.model).name)
        preds.to_csv(RESULTS_DIR / "zero_shot_predictions.csv", index=False)
        summaries.append(summary)
    if args.mode in {"rag", "both"}:
        summary, preds = run_mode(runner, samples, "rag", args.k, Path(args.model).name)
        preds.to_csv(RESULTS_DIR / "rag_predictions.csv", index=False)
        summaries.append(summary)
    llm_summary = pd.concat(summaries, ignore_index=True)
    llm_summary.to_csv(RESULTS_DIR / "llm_results.csv", index=False)
    write_main_table(llm_summary)
    print(json.dumps({"wrote": str(RESULTS_DIR), "rows": len(samples)}, indent=2))


if __name__ == "__main__":
    main()
