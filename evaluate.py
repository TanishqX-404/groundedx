from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score

from common import FIG_DIR, RESULTS_DIR, load_split, taxonomy_classes
from kb_builder import retrieval_metrics
from rag_diagnose import deterministic_rag, zero_shot_diagnose


def load_scoring_labels() -> dict[str, str]:
    path = Path("kb") / "scoring_labels.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_method(name: str, samples: list[dict], k: int, use_rag: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoring_labels = load_scoring_labels()
    rows = []
    preds = []
    y_true = []
    for sample in samples:
        diag = deterministic_rag(sample, k=k) if use_rag else zero_shot_diagnose(sample)
        preds.append(diag.root_cause)
        y_true.append(sample["fault_class"])
        cited_classes = [scoring_labels.get(c) for c in diag.citations]
        citation_fault_match = bool(diag.citations) and all(c == diag.root_cause for c in cited_classes)
        rows.append(
            {
                "id": sample["id"],
                "gold": sample["fault_class"],
                "prediction": diag.root_cause,
                "confidence": diag.confidence,
                "citations": ";".join(diag.citations),
                "citation_fault_match": bool(citation_fault_match),
                "method": name,
            }
        )
    pred_df = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "method": name,
                "top1_acc": accuracy_score(y_true, preds),
                "top3_acc": np.nan,
                "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
                "explanation_faithfulness": float(pred_df["citation_fault_match"].mean()) if use_rag else 0.0,
                "latency_ms": np.nan,
                "vram_gb": np.nan,
            }
        ]
    )
    return summary, pred_df


def evaluate_generation(limit: int | None, k: int) -> pd.DataFrame:
    samples = load_split("test")
    if limit:
        samples = samples[:limit]
    zero_summary, zero_df = evaluate_method("Zero-shot heuristic, no retrieval", samples, k, use_rag=False)
    rag_summary, rag_df = evaluate_method(f"RAN-Doc retrieval-grounded k={k}", samples, k, use_rag=True)
    zero_df.to_csv(RESULTS_DIR / "zero_shot_predictions.csv", index=False)
    rag_df.to_csv(RESULTS_DIR / "rag_predictions.csv", index=False)
    return pd.concat([zero_summary, rag_summary], ignore_index=True)


def write_main_table(rag_summary: pd.DataFrame) -> None:
    baseline_path = RESULTS_DIR / "baseline_results.csv"
    frames = []
    if baseline_path.exists():
        base = pd.read_csv(baseline_path)
        base["explanation_faithfulness"] = np.nan
        base["latency_ms"] = np.nan
        base["vram_gb"] = np.nan
        frames.append(base)
    frames.append(rag_summary)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(RESULTS_DIR / "table1_main_results.csv", index=False)
    print(table.to_string(index=False))


def retrieval_ablation() -> None:
    samples = load_split("test")
    rows = retrieval_metrics(samples, [1, 3, 5, 10])
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "table3_retrieval_ablation.csv", index=False)


def confusion_figure() -> None:
    pred_path = RESULTS_DIR / "rag_predictions.csv"
    if not pred_path.exists():
        return
    df = pd.read_csv(pred_path)
    labels = taxonomy_classes()
    cm = confusion_matrix(df["gold"], df["prediction"], labels=labels)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title("RAN-Doc retrieval-only confusion matrix")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix_rag.png", dpi=300)
    plt.close(fig)


def faithfulness_figure() -> None:
    table_path = RESULTS_DIR / "table1_main_results.csv"
    if not table_path.exists():
        return
    df = pd.read_csv(table_path)
    plot_df = df.dropna(subset=["explanation_faithfulness"])
    if plot_df.empty:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(plot_df["method"], plot_df["explanation_faithfulness"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Citation precision proxy")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "faithfulness_bar.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    generation_summary = evaluate_generation(args.limit or None, args.k)
    write_main_table(generation_summary)
    retrieval_ablation()
    confusion_figure()
    faithfulness_figure()
    print(json.dumps({"wrote": str(RESULTS_DIR)}, indent=2))


if __name__ == "__main__":
    main()
