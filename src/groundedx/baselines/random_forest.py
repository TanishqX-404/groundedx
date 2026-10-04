"""Tree-ensemble baselines over hand-engineered telemetry features."""

from __future__ import annotations

import itertools
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from groundedx.config import DomainConfig


def extract_features(
    samples: list[dict[str, Any]], domain: DomainConfig
) -> tuple[pd.DataFrame, pd.Series]:
    """Per-KPI mean/std/min/max/window-end delta, one-hot alarms, log keyword counts."""

    alarm_codes = sorted({code for fault in domain.taxonomy for code in fault.alarms})
    rows = []
    labels = []
    for sample in samples:
        frame = pd.DataFrame(sample.get("kpis", []))
        row: dict[str, float] = {}
        for name in domain.metric_names:
            if name not in frame:
                continue
            values = frame[name].astype(float)
            row[f"{name}_mean"] = float(values.mean())
            row[f"{name}_std"] = float(values.std())
            row[f"{name}_min"] = float(values.min())
            row[f"{name}_max"] = float(values.max())
            row[f"{name}_delta"] = float(values.tail(5).mean() - values.head(3).mean())
        active = {str(item.get("code", "")) for item in sample.get("alarms", [])}
        row.update({f"alarm_{code}": float(code in active) for code in alarm_codes})
        text = " ".join(sample.get("logs", [])).lower()
        for token in domain.log_keywords:
            row[f"log_kw_{token}"] = float(len(re.findall(re.escape(token), text)))
        rows.append(row)
        labels.append(str(sample["fault_class"]))
    return pd.DataFrame(rows).fillna(0.0), pd.Series(labels, name="fault_class")


def train_random_forest(
    samples: list[dict[str, Any]], domain: DomainConfig, seed: int = 7
) -> RandomForestClassifier:
    """Train the class-balanced 300-tree baseline with a caller-provided seed."""

    features, labels = extract_features(samples, domain)
    model = RandomForestClassifier(
        n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1
    )
    model.fit(features, labels)
    return model


GBDT_GRID = {"max_depth": [3, 6], "n_estimators": [200, 400], "learning_rate": [0.05, 0.1]}


def train_gbdt(
    train: list[dict[str, Any]], val: list[dict[str, Any]], domain: DomainConfig, seed: int = 7
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """XGBoost on the RandomForest features; grid selected on validation accuracy only."""

    from xgboost import XGBClassifier

    x_train, y_train = extract_features(train, domain)
    x_val, y_val = extract_features(val, domain)
    classes = sorted(domain.classes)
    to_idx = {label: i for i, label in enumerate(classes)}
    yt = np.array([to_idx[y] for y in y_train])
    yv = np.array([to_idx[y] for y in y_val])
    grid_log = []
    best: tuple[float, dict[str, Any]] | None = None
    keys = list(GBDT_GRID)
    for values in itertools.product(*(GBDT_GRID[k] for k in keys)):
        params = dict(zip(keys, values))
        model = XGBClassifier(random_state=seed, n_jobs=-1, tree_method="hist", **params)
        model.fit(x_train, yt)
        acc = float(accuracy_score(yv, model.predict(x_val[x_train.columns])))
        grid_log.append({**params, "val_accuracy": acc})
        if best is None or acc > best[0]:
            best = (acc, params)
    assert best is not None
    model = XGBClassifier(random_state=seed, n_jobs=-1, tree_method="hist", **best[1])
    model.fit(x_train, yt)
    model.label_names_ = classes
    model.feature_columns_ = list(x_train.columns)
    return model, best[1], grid_log


def predict_gbdt(model: Any, samples: list[dict[str, Any]], domain: DomainConfig) -> list[str]:
    features, _ = extract_features(samples, domain)
    features = features.reindex(columns=model.feature_columns_, fill_value=0.0)
    return [model.label_names_[int(i)] for i in model.predict(features)]
