"""Generic step/ramp/spike KPI perturbation simulator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from groundedx.config import DomainConfig, FaultSpec


def _shape(pattern: str, steps: int, rng: np.random.Generator) -> np.ndarray:
    onset = int(rng.integers(max(2, steps // 5), max(3, steps // 2)))
    curve = np.zeros(steps)
    if pattern == "ramp":
        curve[onset:] = np.linspace(0.2, 1.0, steps - onset)
    elif pattern == "spike":
        center = int(rng.integers(onset, steps))
        width = float(rng.uniform(1.4, 3.2))
        curve = np.exp(-0.5 * ((np.arange(steps) - center) / width) ** 2)
        curve[:onset] *= 0.15
    else:
        curve[onset:] = 1.0
    return curve


def simulate_window(
    domain: DomainConfig, fault: FaultSpec, rng: np.random.Generator, steps: int = 15
) -> dict[str, Any]:
    """Generate one label-bearing synthetic window for evaluation."""

    timestamps = pd.date_range("2026-01-01 00:00:00", periods=steps, freq="min")
    data: dict[str, np.ndarray] = {}
    for metric in domain.metrics:
        scale = max(abs(metric.baseline) * metric.noise_fraction, 0.04)
        data[metric.name] = rng.normal(metric.baseline, scale, steps)
    curve = _shape(fault.pattern, steps, rng)
    severity = float(rng.uniform(0.25, 0.78))
    for name, effect in fault.kpi_effects.items():
        if name in data:
            data[name] += float(effect) * rng.uniform(0.45, 0.95) * curve * severity
    if rng.random() < domain.distractor_probability:
        distractor = domain.taxonomy[int(rng.integers(0, len(domain.taxonomy)))]
        for name, effect in list(distractor.kpi_effects.items())[:3]:
            if name in data:
                data[name] += float(effect) * curve * rng.uniform(0.18, 0.42)

    frame = pd.DataFrame(data, index=timestamps).reset_index(names="timestamp")
    frame["window_minute"] = np.arange(steps)
    all_alarms = sorted({code for item in domain.taxonomy for code in item.alarms})
    emitted = [code for code in fault.alarms if rng.random() < domain.true_alarm_probability]
    if not emitted:
        emitted = [fault.alarms[int(rng.integers(0, len(fault.alarms)))]]
    if rng.random() < domain.extra_alarm_probability:
        emitted.append(all_alarms[int(rng.integers(0, len(all_alarms)))])
    emitted = list(dict.fromkeys(emitted))
    alarm_start = int(np.argmax(curve > 0.25)) if np.any(curve > 0.25) else steps // 2
    severity_names = list(domain.alarm_severity_distribution)
    severity_probs = np.array(list(domain.alarm_severity_distribution.values()), dtype=float)
    severity_probs /= severity_probs.sum()
    alarms = [
        {
            "timestamp": str(timestamps[min(steps - 1, alarm_start + int(rng.integers(0, 3)))]),
            "code": code,
            "severity": str(rng.choice(severity_names, p=severity_probs)),
        }
        for code in emitted
    ]
    effects = sorted(fault.kpi_effects.items(), key=lambda item: abs(float(item[1])), reverse=True)[
        :3
    ]
    logs = rng.choice(
        domain.log_templates, size=min(2, len(domain.log_templates)), replace=False
    ).tolist()
    logs.extend(
        f"{name} deviation observed with approximate delta {effect * severity:+.2f}"
        for name, effect in effects
    )
    rng.shuffle(logs)
    frame["timestamp"] = frame["timestamp"].astype(str)
    return {
        "id": "",
        "fault_class": fault.name,
        "domain": fault.domain,
        "kpis": frame.to_dict(orient="records"),
        "alarms": alarms,
        "logs": logs[: int(rng.integers(1, domain.max_log_lines + 1))],
        "remediation": fault.remediation,
    }


def generate_dataset(
    domain: DomainConfig, n: int, seed: int, steps: int = 15
) -> list[dict[str, Any]]:
    """Generate a reproducibly shuffled, approximately balanced dataset."""

    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n):
        fault = domain.taxonomy[index % len(domain.taxonomy)]
        sample = simulate_window(domain, fault, rng, steps)
        sample["id"] = f"win-{index:05d}"
        rows.append(sample)
    rng.shuffle(rows)
    return rows
