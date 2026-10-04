"""NVML polling for peak VRAM and power during generation calls."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


def require_model(path: str | Path) -> Path:
    """Validate a model path before a profiling sweep starts."""

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return model_path


class NvmlPoller:
    """Background thread sampling device memory (bytes) and power (W) at ``hz``.

    Samples are kept in memory as (perf_counter time, mem_used_bytes, power_w).
    ``window(t0, t1)`` summarizes the samples recorded during one call.
    """

    def __init__(self, hz: float = 50.0, device_index: int = 0) -> None:
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.period = 1.0 / hz
        self.samples: list[tuple[float, int, float | None]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.power_supported = self._read_power() is not None

    def _read_power(self) -> float | None:
        try:
            return self._nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except Exception:
            return None

    def read_memory(self) -> int:
        return int(self._nvml.nvmlDeviceGetMemoryInfo(self.handle).used)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(), self.read_memory(), self._read_power()))
            time.sleep(self.period)

    def __enter__(self) -> NvmlPoller:
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def window(self, t0: float, t1: float) -> dict[str, float | int | None]:
        rows = [s for s in list(self.samples) if t0 <= s[0] <= t1]
        if not rows:
            return {"n_samples": 0, "peak_mem_bytes": None, "mean_power_w": None, "energy_j": None}
        times = np.array([r[0] for r in rows])
        powers = [r[2] for r in rows]
        result: dict[str, float | int | None] = {
            "n_samples": len(rows),
            "peak_mem_bytes": max(r[1] for r in rows),
            "mean_power_w": None,
            "energy_j": None,
        }
        if all(p is not None for p in powers):
            power = np.array(powers, dtype=float)
            result["mean_power_w"] = float(power.mean())
            # Trapezoidal integral over the sampled interval, extended to the call
            # boundaries with the nearest sample value.
            t = np.concatenate([[t0], times, [t1]])
            p = np.concatenate([[power[0]], power, [power[-1]]])
            result["energy_j"] = float(np.sum((t[1:] - t[:-1]) * (p[1:] + p[:-1]) / 2.0))
        return result
