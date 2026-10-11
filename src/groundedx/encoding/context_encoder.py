"""Encode a telemetry window without exposing its scoring label."""

from __future__ import annotations

from typing import Any

import pandas as pd

from groundedx.config import DomainConfig


def encode_context(
    sample: dict[str, Any],
    domain: DomainConfig,
    top_k_metrics: int = 6,
    max_logs: int | None = None,
    ranking: str = "raw",
) -> str:
    """Turn metrics, alarms, and logs into a compact label-free query.

    ``ranking="raw"`` (default) keeps KPIs whose change exceeds
    max(4% of the start value, 0.03) and ranks them by raw magnitude.
    ``ranking="z"`` standardizes each change by the KPI's schema noise scale
    (baseline x noise_fraction, from the domain config - never from labels),
    keeps changes above one noise unit and ranks by that standardized size.
    """

    if ranking not in ("raw", "z"):
        raise ValueError(f"unknown ranking {ranking!r}")
    scale = {m.name: max(abs(m.baseline) * m.noise_fraction, 0.04) for m in domain.metrics}
    frame = pd.DataFrame(sample.get("kpis", []))
    deltas: list[tuple[str, float, float, float]] = []
    for metric in domain.metric_names:
        if metric not in frame:
            continue
        values = frame[metric].astype(float)
        if values.empty:
            continue
        start = float(values.head(3).mean())
        end = float(values.tail(5).mean())
        delta = end - start
        if ranking == "z":
            size = abs(delta) / scale[metric]
            keep = size > 1.0
        else:
            size = abs(delta)
            keep = abs(delta) > max(abs(start) * 0.04, 0.03)
        if keep:
            deltas.append((metric, delta, end, size))
    deltas.sort(key=lambda row: row[3], reverse=True)
    alarm_codes = ", ".join(str(item.get("code", "")) for item in sample.get("alarms", []))
    logs = sample.get("logs", [])[: max_logs if max_logs is not None else domain.max_log_lines]
    metric_text = "; ".join(
        f"{name}: delta {delta:+.2f}, recent {recent:.2f}"
        for name, delta, recent, _ in deltas[:top_k_metrics]
    )
    return f"Alarms: [{alarm_codes}]. KPI anomalies: {metric_text}. Logs: {' | '.join(logs)}."
