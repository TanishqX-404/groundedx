from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from common import DATA_DIR, RESULTS_DIR, extract_features, load_split, load_taxonomy


MODEL_DIR = Path(__file__).resolve().parent / "models"


def rule_based_predict(sample: dict) -> str:
    alarm_set = {a["code"] for a in sample["alarms"]}
    faults = load_taxonomy()
    scores = []
    for fault in faults:
        overlap = len(alarm_set.intersection(fault["alarms"]))
        scores.append((overlap, fault["class"]))
    best = sorted(scores, key=lambda item: (-item[0], item[1]))[0]
    return best[1]


def train_random_forest(train: list[dict]) -> RandomForestClassifier:
    x_train, y_train = extract_features(train)
    model = RandomForestClassifier(n_estimators=300, random_state=7, class_weight="balanced", n_jobs=-1)
    model.fit(x_train, y_train)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "columns": list(x_train.columns)}, MODEL_DIR / "random_forest.joblib")
    return model


def align_features(samples: list[dict], columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    x, y = extract_features(samples)
    if columns is not None:
        for col in columns:
            if col not in x.columns:
                x[col] = 0.0
        x = x[columns]
    return x, y


def evaluate_classifier(name: str, y_true: list[str], y_pred: list[str]) -> dict[str, float | str]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"method": name, "top1_acc": acc, "top3_acc": np.nan, "macro_f1": f1}


def per_class_report(y_true: list[str], y_pred: list[str], method: str) -> pd.DataFrame:
    labels = sorted(set(y_true))
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame(
        {
            "method": method,
            "fault_class": labels,
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": support,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()

    train = load_split("train")
    test = load_split("test")
    RESULTS_DIR.mkdir(exist_ok=True)

    rf = train_random_forest(train)
    x_train, _ = extract_features(train)
    if args.train_only:
        print(f"Trained RandomForest on {len(train)} samples")
        return

    x_test, y_test = align_features(test, list(x_train.columns))
    rf_pred = list(rf.predict(x_test))
    rule_pred = [rule_based_predict(s) for s in test]
    y_true = [s["fault_class"] for s in test]

    rows = [
        evaluate_classifier("Rule-based expert system", y_true, rule_pred),
        evaluate_classifier("RandomForest structured features", y_true, rf_pred),
    ]
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "baseline_results.csv", index=False)
    pd.concat(
        [
            per_class_report(y_true, rule_pred, "Rule-based expert system"),
            per_class_report(y_true, rf_pred, "RandomForest structured features"),
        ],
        ignore_index=True,
    ).to_csv(RESULTS_DIR / "baseline_per_class.csv", index=False)
    with (RESULTS_DIR / "baseline_predictions.jsonl").open("w", encoding="utf-8") as f:
        for sample, rule, rf_label in zip(test, rule_pred, rf_pred):
            f.write(json.dumps({"id": sample["id"], "gold": sample["fault_class"], "rule": rule, "rf": rf_label}) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
