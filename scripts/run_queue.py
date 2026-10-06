"""Run the revision's GPU jobs as a priority queue over one or more GPUs.

Each GPU gets a worker that pulls the next unfinished job and runs it with
CUDA_VISIBLE_DEVICES set. Jobs are resumable, so re-running this after a
session timeout continues where it stopped.

  python scripts/run_queue.py --phase accuracy --gpus 0 1 --models-dir models
  python scripts/run_queue.py --phase profile  --gpus 0   --models-dir models --tag t4
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from groundedx.calls import CallSpec, experiment_name  # noqa: E402
from groundedx.experiment import RESULTS_ROOT, load_experiment_config  # noqa: E402

CFG = load_experiment_config()
SEEDS = [str(s) for s in CFG["seeds"]]
MAIN = "qwen2.5-3b-q4"
SMALL = "qwen2.5-1.5b-q4"

# Priority order: E1 + E3 first, then the rest of the E2 sweep, then E7 accuracy.
ACCURACY_JOBS: list[dict] = [
    {"model": MAIN, "mode": "zero-shot"},
    {"model": MAIN, "mode": "rag", "kb": "full", "k": 5},
    {"model": MAIN, "mode": "rag", "kb": "no_remediation", "k": 5},
    {"model": MAIN, "mode": "rag", "kb": "generic_remediation", "k": 5},
    {"model": SMALL, "mode": "zero-shot"},
    {"model": SMALL, "mode": "rag", "kb": "full", "k": 5},
    {"model": "qwen2.5-3b-q8", "mode": "zero-shot"},
    {"model": "qwen2.5-3b-q8", "mode": "rag", "kb": "full", "k": 5},
    {"model": "qwen2.5-1.5b-q8", "mode": "zero-shot"},
    {"model": "qwen2.5-1.5b-q8", "mode": "rag", "kb": "full", "k": 5},
    {"model": MAIN, "mode": "rag", "kb": "full", "k": 1},
    {"model": MAIN, "mode": "rag", "kb": "full", "k": 3},
]
# Validation-split pilot for the hybrid (decision rule only; seed 7).
PILOT_JOBS: list[dict] = [
    {"model": MAIN, "mode": "hybrid", "candidates": 3, "split": "val", "seeds": ["7"]},
    {"model": SMALL, "mode": "hybrid", "candidates": 3, "split": "val", "seeds": ["7"]},
    {"model": MAIN, "mode": "rag", "docs": True, "split": "val", "seeds": ["7"]},
    {"model": MAIN, "mode": "zero-shot", "docs": True, "split": "val", "seeds": ["7"]},
]
# Full test runs for the hybrid system and the fault-reference ablation.
HYBRID_JOBS: list[dict] = [
    {"model": MAIN, "mode": "hybrid", "candidates": 3},
    {"model": MAIN, "mode": "hybrid", "candidates": 1},
    {"model": SMALL, "mode": "hybrid", "candidates": 3},
    {"model": SMALL, "mode": "hybrid", "candidates": 1},
    {"model": MAIN, "mode": "rag", "docs": True},
    {"model": MAIN, "mode": "zero-shot", "docs": True},
]
PROFILE_JOBS: list[dict] = [
    {"model": m, "mode": mode} for m in CFG["generation"]["models"] for mode in ("zero-shot", "rag")
] + [{"model": m, "mode": "hybrid", "candidates": n} for m in (MAIN, SMALL) for n in (3, 1)]
PHASE_JOBS = {"accuracy": ACCURACY_JOBS, "pilot": PILOT_JOBS, "hybrid": HYBRID_JOBS}


def spec_of(job: dict) -> CallSpec:
    return CallSpec(job["mode"], job.get("k", 5), job.get("docs", False), job.get("candidates", 3))


def model_path(models_dir: Path, model: str) -> Path:
    return models_dir / CFG["generation"]["models"][model]["file"]


def _lines(path: Path) -> int:
    return sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0


def accuracy_done(job: dict, n_expected: int = 600) -> bool:
    exp = experiment_name(
        job["model"], spec_of(job), job.get("kb", "full"), job.get("split", "test")
    )
    return all(
        _lines(RESULTS_ROOT / exp / f"seed_{s}" / "predictions.jsonl") >= n_expected
        for s in job.get("seeds", SEEDS)
    )


def profile_done(job: dict, tag: str) -> bool:
    spec = spec_of(job)
    mode_tag = {"rag": "rag", "zero-shot": "zs", "hybrid": f"hybrid{spec.candidates}"}[spec.mode]
    exp = f"profile__{job['model']}__{mode_tag}" + (f"__{tag}" if tag else "")
    n = int(CFG["profiling"]["warmup_calls"]) + int(CFG["profiling"]["measured_calls"])
    return all(_lines(RESULTS_ROOT / exp / f"seed_{s}" / "calls.jsonl") >= n for s in SEEDS)


def command(job: dict, phase: str, models_dir: Path, gpu: str, tag: str) -> list[str]:
    script = "run_profile.py" if phase == "profile" else "run_llm.py"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / script),
        "--model",
        job["model"],
        "--model-path",
        str(model_path(models_dir, job["model"])),
        "--mode",
        job["mode"],
        "--seeds",
        *job.get("seeds", SEEDS),
    ]
    if job["mode"] in ("rag", "hybrid"):
        cmd += ["-k", str(job.get("k", 5))]
        if phase != "profile":
            cmd += ["--kb-variant", job.get("kb", "full")]
    if job["mode"] == "hybrid":
        cmd += ["--candidates", str(job["candidates"])]
    if job.get("docs"):
        cmd += ["--class-docs"]
    if phase != "profile":
        cmd += ["--split", job.get("split", "test")]
    if phase == "profile":
        cmd += ["--nvml-index", gpu] + (["--tag", tag] if tag else [])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=[*PHASE_JOBS, "profile"], required=True)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--tag", default="")
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these model keys")
    parser.add_argument(
        "--deadline-hours",
        type=float,
        default=0.0,
        help="stop (and kill running jobs) after this many hours; jobs resume next session",
    )
    args = parser.parse_args()
    deadline = time.time() + args.deadline_hours * 3600 if args.deadline_hours else None

    jobs = PROFILE_JOBS if args.phase == "profile" else PHASE_JOBS[args.phase]
    if args.only:
        jobs = [j for j in jobs if j["model"] in args.only]
    if args.phase == "profile":
        jobs = [j for j in jobs if not profile_done(j, args.tag)]
    else:
        jobs = [j for j in jobs if not accuracy_done(j)]
    pending: queue.Queue = queue.Queue()
    for job in jobs:
        pending.put(job)
    log_dir = RESULTS_ROOT / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} {args.phase} jobs pending on GPUs {args.gpus}", flush=True)

    def worker(gpu: str) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            name = "_".join(f"{v}" for v in job.values()).replace("/", "-")
            cmd = command(job, args.phase, Path(args.models_dir), gpu, args.tag)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
            start = time.time()
            print(f"[gpu{gpu}] start {name}", flush=True)
            if deadline and time.time() > deadline:
                print(f"[gpu{gpu}] deadline reached, not starting {name}", flush=True)
                return
            log_path = log_dir / f"{args.phase}_{name}.log"
            with log_path.open("a", encoding="utf-8") as log:
                proc = subprocess.Popen(
                    cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT
                )
                while proc.poll() is None:
                    if deadline and time.time() > deadline:
                        proc.terminate()
                        proc.wait(timeout=60)
                        print(f"[gpu{gpu}] deadline: stopped {name} (resumable)", flush=True)
                        return
                    time.sleep(5)
                code = proc.returncode
            print(
                f"[gpu{gpu}] end {name} code={code} {(time.time() - start) / 60:.1f} min",
                flush=True,
            )
            if code != 0:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                print(f"[gpu{gpu}] FAILED {name}; last log lines:\n" + "\n".join(tail), flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
