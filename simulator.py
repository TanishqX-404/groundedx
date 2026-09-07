from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "results" / "figures"
TAXONOMY_PATH = ROOT / "fault_taxonomy.yaml"

BASELINES = {
    "prb_util_pct": 55.0,
    "rsrp_dbm": -88.0,
    "rsrq_db": -10.0,
    "sinr_db": 18.0,
    "throughput_mbps": 95.0,
    "latency_ms": 24.0,
    "packet_loss_pct": 0.2,
    "handover_failure_rate": 0.025,
    "drop_call_rate": 0.01,
    "cpu_util_pct": 45.0,
    "setup_success_rate": 0.975,
    "ping_pong_rate": 0.015,
    "availability_pct": 99.8,
}

ALARM_SEVERITIES = ["minor", "major", "critical"]

OVERLAP_EFFECTS = {
    "PCI_COLLISION": {"handover_failure_rate": 0.07, "sinr_db": -4.8, "rsrq_db": -2.5, "packet_loss_pct": 0.8},
    "INTERFERENCE_SPIKE": {"handover_failure_rate": 0.055, "sinr_db": -5.4, "rsrq_db": -2.0, "packet_loss_pct": 1.0},
    "HANDOVER_PARAM_MISCONFIG": {"handover_failure_rate": 0.065, "ping_pong_rate": 0.045, "drop_call_rate": 0.035},
    "NEIGHBOR_LIST_MISCONFIG": {"handover_failure_rate": 0.06, "ping_pong_rate": 0.035, "drop_call_rate": 0.04},
    "BACKHAUL_CONGESTION": {"latency_ms": 24, "packet_loss_pct": 1.45, "throughput_mbps": -18},
    "FRONTHAUL_JITTER": {"latency_ms": 22, "packet_loss_pct": 1.25, "throughput_mbps": -14, "sinr_db": -1.8},
    "SLICE_QOS_POLICY_ERROR": {"latency_ms": 20, "packet_loss_pct": 1.15, "throughput_mbps": -16},
    "HARDWARE_DEGRADATION": {"rsrp_dbm": -6.5, "throughput_mbps": -14, "availability_pct": -1.4},
    "POWER_AMPLIFIER_FAULT": {"rsrp_dbm": -7.0, "throughput_mbps": -15, "availability_pct": -1.5},
    "COOLING_SYSTEM_FAILURE": {"rsrp_dbm": -4.5, "throughput_mbps": -13, "availability_pct": -1.7, "cpu_util_pct": 10},
}

GENERIC_LOG_TEMPLATES = [
    "mobility counters changed after recent window onset",
    "transport delay variation observed on monitored path",
    "radio quality indicators deviated from rolling baseline",
    "service setup counters below expected range",
    "resource utilization shifted from normal operating band",
]


def load_taxonomy(path: Path = TAXONOMY_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["faults"]


def _shape(pattern: str, steps: int, rng: np.random.Generator) -> np.ndarray:
    onset = rng.integers(max(2, steps // 5), max(3, steps // 2))
    x = np.zeros(steps)
    if pattern == "ramp":
        x[onset:] = np.linspace(0.2, 1.0, steps - onset)
    elif pattern == "spike":
        center = rng.integers(onset, steps)
        width = rng.uniform(1.4, 3.2)
        t = np.arange(steps)
        x = np.exp(-0.5 * ((t - center) / width) ** 2)
        x[:onset] *= 0.15
    else:
        x[onset:] = 1.0
    return x


def simulate_window(fault: dict[str, Any], rng: np.random.Generator, steps: int = 15) -> dict[str, Any]:
    minutes = pd.date_range("2026-01-01 00:00:00", periods=steps, freq="min")
    data: dict[str, np.ndarray] = {}
    for kpi, base in BASELINES.items():
        noise_scale = max(abs(base) * 0.13, 0.04)
        data[kpi] = rng.normal(base, noise_scale, steps)

    curve = _shape(fault["pattern"], steps, rng)
    severity_scale = rng.uniform(0.25, 0.78)
    effects = dict(fault["kpi_effects"])
    effects.update(OVERLAP_EFFECTS.get(fault["class"], {}))
    for kpi, effect in effects.items():
        noisy_effect = float(effect) * rng.uniform(0.45, 0.95)
        data[kpi] = data[kpi] + noisy_effect * curve * severity_scale

    if rng.random() < 0.70:
        taxonomy = load_taxonomy()
        distractor = taxonomy[int(rng.integers(0, len(taxonomy)))]
        for kpi, effect in list(distractor["kpi_effects"].items())[:3]:
            if kpi in data:
                data[kpi] = data[kpi] + float(effect) * curve * rng.uniform(0.18, 0.42)

    df = pd.DataFrame(data, index=minutes).reset_index(names="timestamp")
    df["window_minute"] = np.arange(steps)

    alarm_start = int(np.argmax(curve > 0.25)) if np.any(curve > 0.25) else steps // 2
    taxonomy = load_taxonomy()
    all_alarm_codes = sorted({code for item in taxonomy for code in item["alarms"]})
    emitted_codes = [code for code in fault["alarms"] if rng.random() > 0.58]
    if not emitted_codes:
        emitted_codes = [rng.choice(fault["alarms"]).item()]
    if rng.random() < 0.80:
        emitted_codes.append(rng.choice(all_alarm_codes).item())
    if rng.random() < 0.35:
        emitted_codes.append(rng.choice(all_alarm_codes).item())
    emitted_codes = list(dict.fromkeys(emitted_codes))

    alarms = []
    for code in emitted_codes:
        alarms.append(
            {
                "timestamp": str(minutes[min(steps - 1, alarm_start + int(rng.integers(0, 3)))]),
                "code": code,
                "severity": rng.choice(ALARM_SEVERITIES, p=[0.15, 0.55, 0.30]).item(),
                "affected_ne": f"site-{int(rng.integers(100, 999))}/cell-{int(rng.integers(1, 4))}",
            }
        )

    top_effects = sorted(effects.items(), key=lambda item: abs(float(item[1])), reverse=True)[:3]
    logs = rng.choice(GENERIC_LOG_TEMPLATES, size=2, replace=False).tolist()
    logs.extend(
        f"{kpi} deviation observed with approximate delta {float(effect) * severity_scale:+.2f}"
        for kpi, effect in top_effects
    )
    rng.shuffle(logs)

    records = df.copy()
    records["timestamp"] = records["timestamp"].astype(str)

    return {
        "id": "",
        "fault_class": fault["class"],
        "domain": fault["domain"],
        "kpis": records.to_dict(orient="records"),
        "alarms": alarms,
        "logs": logs[: int(rng.integers(2, min(5, len(logs)) + 1))],
        "remediation": fault["remediation"],
    }


def generate_dataset(n: int, seed: int, steps: int = 15) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    faults = load_taxonomy()
    samples = []
    for i in range(n):
        fault = faults[i % len(faults)] if i < len(faults) else faults[int(rng.integers(0, len(faults)))]
        sample = simulate_window(fault, rng, steps=steps)
        sample["id"] = f"win-{i:05d}"
        samples.append(sample)
    rng.shuffle(samples)
    return samples


def split_dataset(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    n = len(samples)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    return {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }


def save_splits(splits: dict[str, list[dict[str, Any]]]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name, rows in splits.items():
        pd.to_pickle(rows, DATA_DIR / f"{name}.pkl")
        with (DATA_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def plot_examples(samples: list[dict[str, Any]], limit: int = 8) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    selected = []
    seen = set()
    for sample in samples:
        if sample["fault_class"] not in seen:
            selected.append(sample)
            seen.add(sample["fault_class"])
        if len(selected) >= limit:
            break

    fig, axes = plt.subplots(len(selected), 1, figsize=(8, max(3, 1.6 * len(selected))), sharex=True)
    if len(selected) == 1:
        axes = [axes]
    for ax, sample in zip(axes, selected):
        df = pd.DataFrame(sample["kpis"])
        ax.plot(df["window_minute"], df["latency_ms"], label="latency_ms")
        ax.plot(df["window_minute"], df["throughput_mbps"], label="throughput_mbps")
        ax.set_title(sample["fault_class"], fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "simulator_kpi_examples.png", dpi=180)
    plt.close(fig)


def dataset_stats(splits: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for split, samples in splits.items():
        counts = pd.Series([s["fault_class"] for s in samples]).value_counts()
        for fault_class, count in counts.items():
            rows.append({"split": split, "fault_class": fault_class, "count": int(count)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=15)
    args = parser.parse_args()

    samples = generate_dataset(args.n, args.seed, args.steps)
    splits = split_dataset(samples)
    save_splits(splits)
    plot_examples(samples)
    stats = dataset_stats(splits)
    stats.to_csv(DATA_DIR / "dataset_stats.csv", index=False)
    print(stats.groupby("split")["count"].sum().to_string())
    print(f"Wrote {sum(len(v) for v in splits.values())} windows to {DATA_DIR}")


if __name__ == "__main__":
    main()
