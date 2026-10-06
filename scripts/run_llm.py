"""Full-split SLM accuracy runs (E1/E2/E3/E7 and the hybrid).

Writes results/v2/<experiment>/seed_<s>/{predictions.jsonl, manifest.json}.
Resumable: windows already present in predictions.jsonl are skipped, so a
Kaggle session that times out can be restarted with the same command.

Example:
  python scripts/run_llm.py --model qwen2.5-3b-q4 \
      --model-path models/qwen2.5-3b-instruct-q4_k_m.gguf \
      --mode rag --kb-variant full -k 5 --seeds 7 19 41
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundedx.calls import CallBuilder, CallSpec, experiment_name  # noqa: E402
from groundedx.experiment import (  # noqa: E402
    ROOT,
    load_experiment_config,
    prepare_seed,
    read_jsonl,
    run_dir,
    write_manifest,
)
from groundedx.faithfulness.grounding_metric import grounding_check  # noqa: E402
from groundedx.generation.clients import backend_info, make_client  # noqa: E402
from groundedx.generation.llama_cpp_client import MAX_TOKENS, TEMPERATURE  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="config key, e.g. qwen2.5-3b-q4")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", choices=["rag", "zero-shot", "hybrid"], required=True)
    parser.add_argument("--class-docs", action="store_true", help="add the fault reference")
    parser.add_argument("--candidates", type=int, default=3, help="hybrid: GBDT top-N")
    parser.add_argument("--kb-variant", default="full")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0, help="smoke tests only")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--backend", choices=["server", "python"], default="server")
    parser.add_argument("--server-bin", default=os.environ.get("LLAMA_SERVER_BIN"))
    parser.add_argument("--experiment", default=None, help="override output directory name")
    args = parser.parse_args()

    cfg = load_experiment_config()
    seeds = args.seeds or cfg["seeds"]
    spec = CallSpec(args.mode, args.k, args.class_docs, args.candidates)
    exp = args.experiment or experiment_name(args.model, spec, args.kb_variant, args.split)
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

    for seed in seeds:
        setup = prepare_seed(seed, args.kb_variant, cfg)
        samples = setup.splits[args.split]
        if args.limit:
            samples = samples[: args.limit]
        builder = CallBuilder(setup, spec)
        out_dir = run_dir(exp, seed)
        pred_path = out_dir / "predictions.jsonl"
        done = {row["id"] for row in read_jsonl(pred_path)}
        write_manifest(
            out_dir,
            seed,
            model_path=args.model_path,
            model=args.model,
            mode=args.mode,
            class_docs=args.class_docs,
            candidates=args.candidates if args.mode == "hybrid" else None,
            gbdt_params=builder.gbdt_params,
            kb_variant=args.kb_variant if spec.uses_retrieval else None,
            k=args.k if spec.uses_retrieval else 0,
            split=args.split,
            n_windows=len(samples),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            n_gpu_layers=args.n_gpu_layers,
            **backend_info(client),
        )
        builder.precompute([s for s in samples if s["id"] not in done])
        started = time.time()
        with pred_path.open("a", encoding="utf-8") as handle:
            for index, sample in enumerate(samples, start=1):
                if sample["id"] in done:
                    continue
                call = builder.build(sample)
                retrieved, ids = call["retrieved"], call["citation_ids"]
                out = client.diagnose(call["prompt"], call["classes"], ids)
                diag = out["diagnosis"] or {}
                # Evaluator-side bookkeeping (after generation; never shown to the model).
                labels = [setup.scoring_labels[i] for i in ids]
                gold = sample["fault_class"]
                record = {
                    "id": sample["id"],
                    "gold": gold,
                    "secondary_class": sample["secondary_class"],
                    "true_alarm_emitted": sample["true_alarm_emitted"],
                    "prediction": diag.get("root_cause") if out["valid"] else None,
                    "valid": out["valid"],
                    "invalid_reason": out["invalid_reason"],
                    "confidence": diag.get("confidence"),
                    "citations": diag.get("citations", []),
                    "explanation": diag.get("explanation"),
                    "remediation": diag.get("remediation"),
                    "grounded": bool(
                        out["valid"]
                        and grounding_check(
                            diag["explanation"],
                            diag["citations"],
                            {row["chunk_id"]: row["text"] for row in retrieved},
                        )
                    ),
                    "retrieved": [
                        {"id": row["chunk_id"], "score": round(row["score"], 6)}
                        for row in retrieved
                    ],
                    "retrieved_labels": labels,
                    "candidates": call["candidates"],
                    "classifier_top1": call["candidates"][0][0] if call["candidates"] else None,
                    "same_class_rank": labels.index(gold) + 1 if gold in labels else None,
                    "latency_ms": out["latency_ms"],
                    "prompt_tokens": out["prompt_tokens"],
                    "completion_tokens": out["completion_tokens"],
                    "finish_reason": out["finish_reason"],
                    "raw": out["raw"],
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                if index % 50 == 0:
                    rate = (time.time() - started) / max(1, index - len(done))
                    print(
                        f"[{exp} seed={seed}] {index}/{len(samples)} ({rate:.2f}s/call)", flush=True
                    )
        print(f"[{exp} seed={seed}] done -> {pred_path}", flush=True)


if __name__ == "__main__":
    main()
