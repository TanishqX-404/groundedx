"""Generic step/ramp/spike KPI perturbation simulator.

Each window records evaluator-side metadata used by the revision analyses:
``secondary_class`` / ``blend_amplitude`` (cross-class KPI blending),
``true_alarm_emitted`` (whether any of the class's own alarm codes fired) and
``distractor_alarms`` (injected codes that belong to other classes). None of
these fields is ever passed to the context encoder, retriever, or prompt.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from groundedx.config import DomainConfig, FaultSpec

# Per-code emission probability once a window is chosen to emit its own alarms.
OWN_ALARM_CODE_PROBABILITY = 0.5


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


def _noise_scale(baseline: float, noise_fraction: float) -> float:
    return max(abs(baseline) * noise_fraction, 0.04)


def simulate_window(
    domain: DomainConfig, fault: FaultSpec, rng: np.random.Generator, steps: int = 15
) -> dict[str, Any]:
    """Generate one label-bearing synthetic window for evaluation."""

    timestamps = pd.date_range("2026-01-01 00:00:00", periods=steps, freq="min")
    data: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    for metric in domain.metrics:
        scales[metric.name] = _noise_scale(metric.baseline, metric.noise_fraction)
        data[metric.name] = rng.normal(metric.baseline, scales[metric.name], steps)
    curve = _shape(fault.pattern, steps, rng)
    severity = float(rng.uniform(0.25, 0.78))
    for name, effect in fault.kpi_effects.items():
        if name in data:
            data[name] += float(effect) * rng.uniform(0.45, 0.95) * curve * severity

    secondary_class: str | None = None
    blend_amplitude: float | None = None
    if rng.random() < domain.distractor_probability:
        others = [item for item in domain.taxonomy if item.name != fault.name]
        secondary = others[int(rng.integers(0, len(others)))]
        secondary_class = secondary.name
        blend_amplitude = float(rng.uniform(0.18, 0.42))
        for name, effect in list(secondary.kpi_effects.items())[:3]:
            if name in data:
                data[name] += float(effect) * curve * blend_amplitude

    true_alarm_emitted = bool(rng.random() < domain.true_alarm_probability)
    emitted: list[str] = []
    if true_alarm_emitted:
        emitted = [code for code in fault.alarms if rng.random() < OWN_ALARM_CODE_PROBABILITY]
        if not emitted:
            emitted = [fault.alarms[int(rng.integers(0, len(fault.alarms)))]]
    distractor_alarms: list[str] = []
    if rng.random() < domain.extra_alarm_probability:
        pool = sorted({c for item in domain.taxonomy for c in item.alarms} - set(fault.alarms))
        distractor_alarms = [pool[int(rng.integers(0, len(pool)))]]
    codes = list(dict.fromkeys(emitted + distractor_alarms))
    rng.shuffle(codes)

    alarm_start = int(np.argmax(curve > 0.25)) if np.any(curve > 0.25) else steps // 2
    severity_names = list(domain.alarm_severity_distribution)
    severity_probs = np.array(list(domain.alarm_severity_distribution.values()), dtype=float)
    severity_probs /= severity_probs.sum()
    alarms = [
        {
            "timestamp": str(timestamps[min(steps - 1, alarm_start + int(rng.integers(0, 3)))]),
            "code": code,
            "severity": str(rng.choice(severity_names, p=severity_probs)),
            "affected_ne": f"site-{int(rng.integers(100, 999))}/cell-{int(rng.integers(1, 4))}",
        }
        for code in codes
    ]

    # Log lines describe the largest *observed* deviations (after blending and
    # noise), ranked by standardized magnitude, plus generic operations lines.
    observed = []
    for name, values in data.items():
        delta = float(values[-5:].mean() - values[:3].mean())
        observed.append((abs(delta) / scales[name], name, delta))
    observed.sort(reverse=True)
    logs = rng.choice(
        domain.log_templates, size=min(2, len(domain.log_templates)), replace=False
    ).tolist()
    logs.extend(
        f"{name} deviation observed with approximate delta {delta:+.2f}"
        for _, name, delta in observed[:3]
    )
    rng.shuffle(logs)
    n_logs = int(rng.integers(2, len(logs) + 1))

    frame = pd.DataFrame(data, index=timestamps).reset_index(names="timestamp")
    frame["window_minute"] = np.arange(steps)
    frame["timestamp"] = frame["timestamp"].astype(str)
    return {
        "id": "",
        "fault_class": fault.name,
        "domain": fault.domain,
        "kpis": frame.to_dict(orient="records"),
        "alarms": alarms,
        "logs": logs[:n_logs],
        "remediation": fault.remediation,
        "secondary_class": secondary_class,
        "blend_amplitude": blend_amplitude,
        "true_alarm_emitted": true_alarm_emitted,
        "distractor_alarms": distractor_alarms,
        "severity": severity,
    }


def generate_dataset(
    domain: DomainConfig, n: int, seed: int, steps: int = 15
) -> list[dict[str, Any]]:
    """Generate a reproducibly shuffled, class-balanced dataset.

    IDs are assigned after shuffling so that a window ID carries no
    information about its class.
    """

    rng = np.random.default_rng(seed)
    rows = [
        simulate_window(domain, domain.taxonomy[index % len(domain.taxonomy)], rng, steps)
        for index in range(n)
    ]
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["id"] = f"w{seed}-{index:05d}"
    return rows


def split_dataset(
    rows: list[dict[str, Any]], train: float = 0.70, val: float = 0.15
) -> dict[str, list[dict[str, Any]]]:
    """Split an already-shuffled dataset into train/val/test."""

    n = len(rows)
    train_end = int(train * n)
    val_end = int((train + val) * n)
    return {"train": rows[:train_end], "val": rows[train_end:val_end], "test": rows[val_end:]}
