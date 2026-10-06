"""Latency / peak-VRAM / power profiling harness (E2 profiling half).

Per seed: 5 warm-up calls (discarded), then 100 measured calls on a fixed
class-stratified sample of that seed's test split. An NVML thread polls device
memory and power at 50 Hz. Accuracy is NOT taken from this harness.

Writes results/v2/profile__<model>__<mode>/seed_<s>/{calls.jsonl, manifest.json}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundedx.calls import CallBuilder, CallSpec  # noqa: E402
from groundedx.evaluation.run_profiling import NvmlPoller, require_model  # noqa: E402
from groundedx.experiment import (  # noqa: E402
    ROOT,
    load_experiment_config,
    prepare_seed,
    run_dir,
    write_manifest,
)


def stratified_sample(samples: list[dict], n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed + 100_000)
    by_class: dict[str, list[dict]] = {}
    for sample in samples:
        by_class.setdefault(sample["fault_class"], []).append(sample)
    for pool in by_class.values():
        rng.shuffle(pool)
    chosen: list[dict] = []
    depth = 0
    while len(chosen) < n:
        for label in sorted(by_class):
            if depth < len(by_class[label]) and len(chosen) < n:
                chosen.append(by_class[label][depth])
        depth += 1
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", choices=["rag", "zero-shot", "hybrid"], required=True)
    parser.add_argument("--class-docs", action="store_true")
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--backend", choices=["server", "python"], default="server")
    parser.add_argument("--server-bin", default=os.environ.get("LLAMA_SERVER_BIN"))
    parser.add_argument("--tag", default="", help="hardware tag appended to the experiment name")
    parser.add_argument("--nvml-index", type=int, default=0, help="physical GPU index for NVML")
    args = parser.parse_args()

    cfg = load_experiment_config()
    prof = cfg["profiling"]
    seeds = args.seeds or cfg["seeds"]
    require_model(args.model_path)
    spec = CallSpec(args.mode, args.k, args.class_docs, args.candidates)
    mode_tag = {"rag": "rag", "zero-shot": "zs", "hybrid": f"hybrid{args.candidates}"}[args.mode]
    if args.class_docs and args.mode != "hybrid":
        mode_tag += "__docs"
    exp = f"profile__{args.model}__{mode_tag}" + (f"__{args.tag}" if args.tag else "")

    with NvmlPoller(hz=float(prof["nvml_poll_hz"]), device_index=args.nvml_index) as poller:
        mem_before_load = poller.read_memory()
        from groundedx.generation.clients import backend_info, make_client

        log_dir = ROOT / "results" / "v2" / "_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        client = make_client(
            args.backend,
            args.model_path,
            n_ctx=int(cfg["generation"]["n_ctx"]),
            n_gpu_layers=args.n_gpu_layers,
            server_bin=args.server_bin,
            log_path=log_dir / f"server_{exp}.log",
        )
        time.sleep(0.5)
        mem_after_load = poller.read_memory()
        for seed in seeds:
            setup = prepare_seed(seed, "full", cfg)
            test = setup.splits["test"]
            measured = stratified_sample(test, int(prof["measured_calls"]), seed)
            measured_ids = {s["id"] for s in measured}
            warmup = [s for s in test if s["id"] not in measured_ids][: int(prof["warmup_calls"])]
            builder = CallBuilder(setup, spec)
            out_dir = run_dir(exp, seed)
            write_manifest(
                out_dir,
                seed,
                model_path=args.model_path,
                model=args.model,
                mode=args.mode,
                k=args.k if spec.uses_retrieval else 0,
                class_docs=args.class_docs,
                candidates=args.candidates if args.mode == "hybrid" else None,
                warmup_calls=len(warmup),
                measured_calls=len(measured),
                nvml_poll_hz=prof["nvml_poll_hz"],
                power_supported=poller.power_supported,
                mem_before_load_bytes=mem_before_load,
                mem_after_load_bytes=mem_after_load,
                n_gpu_layers=args.n_gpu_layers,
                **backend_info(client),
            )
            with (out_dir / "calls.jsonl").open("w", encoding="utf-8") as handle:
                for phase, batch in (("warmup", warmup), ("measured", measured)):
                    for sample in batch:
                        # Hybrid: time single-window classifier inference (features +
                        # GBDT) separately; end-to-end latency = classifier_ms + latency_ms.
                        c0 = time.perf_counter()
                        builder.precompute([sample])
                        classifier_ms = (time.perf_counter() - c0) * 1000.0
                        call = builder.build(sample)
                        t0 = time.perf_counter()
                        out = client.diagnose(call["prompt"], call["classes"], call["citation_ids"])
                        t1 = time.perf_counter()
                        time.sleep(0.05)  # let the poller record the tail of the call
                        stats = poller.window(t0, t1)
                        handle.write(
                            json.dumps(
                                {
                                    "id": sample["id"],
                                    "phase": phase,
                                    "latency_ms": (t1 - t0) * 1000.0,
                                    "classifier_ms": classifier_ms
                                    if spec.mode == "hybrid"
                                    else 0.0,
                                    "valid": out["valid"],
                                    "prompt_tokens": out["prompt_tokens"],
                                    "completion_tokens": out["completion_tokens"],
                                    **stats,
                                }
                            )
                            + "\n"
                        )
            print(f"[{exp} seed={seed}] done", flush=True)


if __name__ == "__main__":
    main()
