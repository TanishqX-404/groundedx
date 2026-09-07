from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from common import RESULTS_DIR, load_split
from rag_diagnose import LlamaDiagnoser, diagnose


def gpu_snapshot() -> tuple[float | None, float | None]:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        return mem.used / (1024**3), power_w
    except Exception:
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--mode", choices=["rag", "zero-shot"], default="rag")
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    if args.model and not Path(args.model).exists():
        raise FileNotFoundError(args.model)

    samples = load_split("test")[: args.limit]
    runner = LlamaDiagnoser(args.model, n_gpu_layers=args.n_gpu_layers) if args.model else None
    rows = []
    for sample in samples:
        mem_before, power_before = gpu_snapshot()
        start = time.perf_counter()
        if runner and args.mode == "rag":
            result = runner.rag(sample, k=args.k)
        elif runner:
            result = runner.zero_shot(sample)
        else:
            result = diagnose(sample, None, k=args.k)
        elapsed = time.perf_counter() - start
        mem_after, power_after = gpu_snapshot()
        rows.append(
            {
                "id": sample["id"],
                "gold": sample["fault_class"],
                "prediction": result.root_cause,
                "mode": f"llama_cpp_{args.mode}" if args.model else "retrieval_only",
                "latency_ms": elapsed * 1000,
                "vram_gb_before": mem_before,
                "vram_gb_after": mem_after,
                "power_w_before": power_before,
                "power_w_after": power_after,
                "joules_proxy": ((power_before or 0) + (power_after or 0)) * 0.5 * elapsed if power_after else None,
            }
        )
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "profile_results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
