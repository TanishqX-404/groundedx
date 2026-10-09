"""One Kaggle session, end to end. The notebook only clones the branch and runs this.

Keeping all logic here (not in notebook cells) means the notebook never goes
stale: whatever is on the branch is what runs.

Steps
  1. restore partial results from an attached earlier output (never overwriting
     files that are already in the repo);
  2. reuse or build llama-server (CUDA, pinned build) and reuse or download GGUFs;
  3. print the job PLAN (what is pending per phase) - check this first;
  4. correctness gate (pytest, incl. a real constrained decode);
  5. run the requested phases on all GPUs, then any pending profiling;
  6. regenerate results/v2/RESULTS.md and zip results/v2 to /kaggle/working/results_v2.zip.

  python scripts/kaggle_session.py --phases pilot --deadline-hours 11
  python scripts/kaggle_session.py --phases pilot --dry-run      # plan only
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
LLAMA_TAG = "b11398"
T0 = time.time()


def run(args: list, check: bool = True, env: dict | None = None) -> int:
    print("+", " ".join(map(str, args)), flush=True)
    proc = subprocess.Popen(
        [str(a) for a in args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=ROOT,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    code = proc.wait()
    if check and code != 0:
        raise RuntimeError(f"command failed with exit code {code}: {args[:3]}")
    return code


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}  (t+{(time.time() - T0) / 60:.1f} min)\n{'=' * 70}", flush=True)


def queue_module():
    spec = importlib.util.spec_from_file_location("rq", ROOT / "scripts" / "run_queue.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def restore_results() -> None:
    results = ROOT / "results"
    for archive in glob.glob("/kaggle/input/**/results_v2.zip", recursive=True):
        added = 0
        with zipfile.ZipFile(archive) as handle:
            for name in handle.namelist():
                target = results / name
                if name.endswith("/") or target.exists():
                    continue
                handle.extract(name, results)
                added += 1
        print(f"restored {added} files from {archive} (existing files kept)")


def llama_server() -> Path:
    binary = WORK / "llama-bin" / "llama-server"
    if binary.exists():
        return binary
    for found in glob.glob("/kaggle/input/**/llama-bin/llama-server", recursive=True):
        binary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(found, binary)
        binary.chmod(0o755)
        print("reusing llama-server from", found)
        return binary
    nvcc = shutil.which("nvcc") or next(iter(glob.glob("/usr/local/cuda*/bin/nvcc")), None)
    if not nvcc:
        raise RuntimeError("nvcc not found - is the GPU accelerator enabled?")
    cuda_root = Path(os.path.realpath(nvcc)).parents[1]
    libcuda = next(
        iter(
            glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so*")
            + glob.glob(f"{cuda_root}/lib64/stubs/libcuda.so")
        ),
        None,
    )
    src = Path("/tmp/llama.cpp")
    shutil.rmtree(src, ignore_errors=True)
    run(["git", "-c", "advice.detachedHead=false", "clone", "-q", "--depth", "1", "-b",
         LLAMA_TAG, "https://github.com/ggml-org/llama.cpp", src])  # fmt: skip
    base = ["cmake", "-S", src, "-B", src / "build", "-DGGML_CUDA=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=75", f"-DCMAKE_CUDA_COMPILER={nvcc}",
            f"-DCUDAToolkit_ROOT={cuda_root}", "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_CURL=OFF",
            "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF"]  # fmt: skip
    attempts = [("driver-lib", [f"-DCUDA_cuda_driver_LIBRARY={libcuda}"])] if libcuda else []
    attempts.append(("no-vmm", ["-DGGML_CUDA_NO_VMM=ON"]))
    for name, extra in attempts:
        shutil.rmtree(src / "build", ignore_errors=True)
        if (
            run(base + extra, check=False) == 0
            and run(
                [
                    "cmake",
                    "--build",
                    src / "build",
                    "--config",
                    "Release",
                    "-j4",
                    "--target",
                    "llama-server",
                ],
                check=False,  # fmt: skip
            )
            == 0
        ):
            (WORK / "llama_build_variant.txt").write_text(name)
            print("llama-server build OK with", name)
            break
    else:
        raise RuntimeError("llama-server build failed with all configurations")
    binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "build" / "bin" / "llama-server", binary)
    return binary


def models(cfg: dict) -> None:
    from huggingface_hub import hf_hub_download

    target = ROOT / "models"
    target.mkdir(exist_ok=True)
    for key, spec in cfg["generation"]["models"].items():
        dst = target / spec["file"]
        if dst.exists():
            continue
        found = glob.glob(f"/kaggle/input/**/models/{spec['file']}", recursive=True)
        if found:
            os.symlink(found[0], dst)
            print("reusing", found[0])
        else:
            print(key, hf_hub_download(spec["repo"], spec["file"], local_dir=target), flush=True)


def plan(rq, phases: list[str], profile_tag: str) -> dict[str, list[dict]]:
    pending = {ph: [j for j in rq.PHASE_JOBS[ph] if not rq.accuracy_done(j)] for ph in phases}
    pending["profile"] = [j for j in rq.PROFILE_JOBS if not rq.profile_done(j, profile_tag)]
    banner("PLAN")
    for phase in ["accuracy"] + [p for p in phases if p != "accuracy"] + ["profile"]:
        jobs = pending.get(phase, [])
        total = len(rq.PROFILE_JOBS if phase == "profile" else rq.PHASE_JOBS[phase])
        print(f"{phase:9s}: {len(jobs)} pending of {total}")
        for job in jobs:
            print("           ", job)
    return pending


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", nargs="+", default=["pilot"], help="pilot and/or hybrid")
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--deadline-hours", type=float, default=11.0)
    parser.add_argument("--profile-tag", default="t4")
    parser.add_argument("--no-profiling", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="restore + print the plan, then stop"
    )
    args = parser.parse_args()
    phases = ["accuracy"] + [p for p in args.phases if p != "accuracy"]

    sys.path.insert(0, str(ROOT / "src"))
    from groundedx.experiment import load_experiment_config

    cfg = load_experiment_config()
    rq = queue_module()
    unknown = [p for p in phases if p not in rq.PHASE_JOBS]
    if unknown:
        raise SystemExit(f"unknown phase(s) {unknown}; choose from {list(rq.PHASE_JOBS)}")

    banner("RESTORE")
    restore_results()
    pending = plan(rq, phases, args.profile_tag)
    if args.dry_run:
        return

    banner("SETUP")
    os.environ["LLAMA_SERVER_BIN"] = str(llama_server())
    run([os.environ["LLAMA_SERVER_BIN"], "--version"], check=False)
    models(cfg)

    banner("CORRECTNESS GATE")
    env = {**os.environ, "GROUNDEDX_TEST_MODEL": "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"}
    run([sys.executable, "-m", "pytest", "-q", "-x", "tests"], env=env)

    deadline = T0 + args.deadline_hours * 3600
    for phase in phases:
        if not pending[phase]:
            continue
        banner(f"PHASE {phase}")
        left = (deadline - time.time()) / 3600
        run([sys.executable, "scripts/run_queue.py", "--phase", phase, "--gpus", *args.gpus,
             "--models-dir", "models", "--deadline-hours", f"{left:.2f}"], check=False)  # fmt: skip

    unfinished = [(p, j) for p in phases for j in rq.PHASE_JOBS[p] if not rq.accuracy_done(j)]
    left = (deadline - time.time()) / 3600
    if not args.no_profiling and not unfinished and left > 0.3:
        banner("PROFILING (pending configs only, one GPU)")
        run([sys.executable, "scripts/run_queue.py", "--phase", "profile", "--gpus", "0",
             "--models-dir", "models", "--tag", args.profile_tag,
             "--deadline-hours", f"{left:.2f}"], check=False)  # fmt: skip

    banner("RESULTS")
    run([sys.executable, "scripts/make_results.py"], check=False)
    archive = WORK / "results_v2.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in (ROOT / "results" / "v2").rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(ROOT / "results"))
    print(f"zipped -> {archive} ({archive.stat().st_size // 1024} KB)")
    plan(rq, phases, args.profile_tag)
    if unfinished:
        print("UNFINISHED jobs (attach this output and run again to resume):")
        for phase, job in unfinished:
            print("   ", phase, job)
    text = (ROOT / "results" / "v2" / "RESULTS.md").read_text(encoding="utf-8")
    for header in ("## Hybrid", "## Validation-split pilots"):
        if header in text:
            print(text[text.index(header) :].split("\n## ")[0])


if __name__ == "__main__":
    main()
