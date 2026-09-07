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
) -> str:
    """Turn metrics, alarms, and logs into a compact label-free query."""

    frame = pd.DataFrame(sample.get("kpis", []))
    deltas: list[tuple[str, float, float]] = []
    for metric in domain.metric_names:
        if metric not in frame:
            continue
        values = frame[metric].astype(float)
        if values.empty:
            continue
        start = float(values.head(3).mean())
        end = float(values.tail(5).mean())
        delta = end - start
        if abs(delta) > max(abs(start) * 0.04, 0.03):
            deltas.append((metric, delta, end))
    deltas.sort(key=lambda row: abs(row[1]), reverse=True)
    alarm_codes = ", ".join(str(item.get("code", "")) for item in sample.get("alarms", []))
    logs = sample.get("logs", [])[: max_logs if max_logs is not None else domain.max_log_lines]
    metric_text = "; ".join(
        f"{name}: delta {delta:+.2f}, recent {recent:.2f}"
        for name, delta, recent in deltas[:top_k_metrics]
    )
    return f"Alarms: [{alarm_codes}]. KPI anomalies: {metric_text}. Logs: {' | '.join(logs)}."
