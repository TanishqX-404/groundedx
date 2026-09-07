from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
KB_DIR = ROOT / "kb"
TAXONOMY_PATH = ROOT / "fault_taxonomy.yaml"


def load_taxonomy() -> list[dict[str, Any]]:
    with TAXONOMY_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["faults"]


def taxonomy_classes() -> list[str]:
    return [f["class"] for f in load_taxonomy()]


def load_split(name: str) -> list[dict[str, Any]]:
    return pd.read_pickle(DATA_DIR / f"{name}.pkl")


def summarize_window(sample: dict[str, Any]) -> str:
    df = pd.DataFrame(sample["kpis"])
    numeric = [c for c in df.columns if c not in {"timestamp", "window_minute"}]
    deltas = []
    for col in numeric:
        start = float(df[col].head(3).mean())
        end = float(df[col].tail(5).mean())
        delta = end - start
        if abs(delta) > max(abs(start) * 0.04, 0.03):
            deltas.append((col, delta, end))
    deltas = sorted(deltas, key=lambda row: abs(row[1]), reverse=True)[:6]
    alarms = ", ".join(a["code"] for a in sample["alarms"])
    logs = " | ".join(sample["logs"][:4])
    kpi_text = "; ".join(f"{k}: delta {d:+.2f}, recent {v:.2f}" for k, d, v in deltas)
    return f"Alarms: [{alarms}]. KPI anomalies: {kpi_text}. Logs: {logs}."


def extract_features(samples: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.Series]:
    alarm_codes = sorted({a["code"] for s in samples for a in s["alarms"]})
    rows = []
    labels = []
    for sample in samples:
        df = pd.DataFrame(sample["kpis"])
        row: dict[str, float] = {}
        numeric = [c for c in df.columns if c not in {"timestamp", "window_minute"}]
        for col in numeric:
            values = df[col].astype(float)
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std())
            row[f"{col}_min"] = float(values.min())
            row[f"{col}_max"] = float(values.max())
            row[f"{col}_delta"] = float(values.tail(5).mean() - values.head(3).mean())
        active = {a["code"] for a in sample["alarms"]}
        for code in alarm_codes:
            row[f"alarm_{code}"] = 1.0 if code in active else 0.0
        text = " ".join(sample["logs"]).lower()
        for token in ["latency", "throughput", "handover", "sync", "cpu", "power", "jitter", "policy"]:
            row[f"log_kw_{token}"] = float(len(re.findall(token, text)))
        rows.append(row)
        labels.append(sample["fault_class"])
    return pd.DataFrame(rows).fillna(0.0), pd.Series(labels, name="fault_class")
