"""RandomForest baseline over generic telemetry features."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from groundedx.config import DomainConfig


def extract_features(
    samples: list[dict[str, Any]], domain: DomainConfig
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract schema-driven numeric and alarm features."""

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
            row[f"{name}_delta"] = float(values.tail(5).mean() - values.head(3).mean())
        active = {str(item.get("code", "")) for item in sample.get("alarms", [])}
        row.update({f"alarm_{code}": float(code in active) for code in alarm_codes})
        rows.append(row)
        labels.append(str(sample["fault_class"]))
    return pd.DataFrame(rows).fillna(0.0), pd.Series(labels, name="fault_class")


def train_random_forest(
    samples: list[dict[str, Any]], domain: DomainConfig, seed: int = 7
) -> RandomForestClassifier:
    """Train the class-balanced baseline with a caller-provided seed."""

    features, labels = extract_features(samples, domain)
    model = RandomForestClassifier(
        n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1
    )
    model.fit(features, labels)
    return model
