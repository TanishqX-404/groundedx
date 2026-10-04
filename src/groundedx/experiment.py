"""Seeded experiment setup and run manifests shared by the revision scripts.

A seed controls: (1) dataset generation and the 70/15/15 split, (2) the
stratified choice of KB cases from that seed's training split, and (3) the
random_state of the tree baselines. LLM decoding is greedy and deterministic.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from groundedx.config import DomainConfig, load_domain_bundle
from groundedx.kb.build_kb import build_documents
from groundedx.kb.scoring_labels import make_scoring_labels, select_kb_cases
from groundedx.retrieval.tfidf_retriever import TfidfRetriever
from groundedx.simulator.kpi_generator import generate_dataset, split_dataset

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
RESULTS_ROOT = ROOT / "results" / "v2"


def load_experiment_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@dataclass
class SeedSetup:
    seed: int
    domain: DomainConfig
    splits: dict[str, list[dict[str, Any]]]
    kb_samples: list[dict[str, Any]]
    documents: list[dict[str, str]]
    scoring_labels: dict[str, str]
    retriever: TfidfRetriever


def prepare_seed(
    seed: int, kb_variant: str = "full", config: dict[str, Any] | None = None
) -> SeedSetup:
    cfg = config or load_experiment_config()
    domain = load_domain_bundle(ROOT / cfg["domain"])
    rows = generate_dataset(domain, int(cfg["dataset"]["n_windows"]), seed)
    splits = split_dataset(rows)
    kb_samples = select_kb_cases(splits["train"], int(cfg["kb"]["cases_per_class"]), seed)
    documents = build_documents(kb_samples, domain, variant=kb_variant)
    scoring_labels = make_scoring_labels(kb_samples, [d["chunk_id"] for d in documents])
    retriever = TfidfRetriever(documents, sublinear_tf=bool(cfg["retrieval"]["sublinear_tf"]))
    return SeedSetup(seed, domain, splits, kb_samples, documents, scoring_labels, retriever)


def run_dir(experiment: str, seed: int) -> Path:
    path = RESULTS_ROOT / experiment / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _git_commit() -> str | None:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", ".", ":!results"], cwd=ROOT, text=True)
        return sha + ("-dirty" if dirty.strip() else "")
    except Exception:
        # Code shipped as a `git archive` zip carries its commit in GIT_COMMIT.
        stamp = ROOT / "GIT_COMMIT"
        return stamp.read_text(encoding="utf-8").strip() if stamp.exists() else None


def _gpu_info() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
        cuda = subprocess.check_output(["nvidia-smi"], text=True)
        cuda_version = next(
            (
                line.split("CUDA Version:")[1].split("|")[0].strip()
                for line in cuda.splitlines()
                if "CUDA Version:" in line
            ),
            None,
        )
        return {"gpus": out.splitlines(), "cuda_version": cuda_version}
    except Exception:
        return {"gpus": [], "cuda_version": None}


_SHA_CACHE: dict[str, str] = {}


def file_sha256(path: str | Path) -> str:
    key = str(Path(path).resolve())
    if key not in _SHA_CACHE:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        _SHA_CACHE[key] = digest.hexdigest()
    return _SHA_CACHE[key]


def write_manifest(directory: Path, seed: int, **extra: Any) -> None:
    versions = {}
    for name in ("numpy", "pandas", "sklearn", "xgboost", "llama_cpp", "pynvml"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    model_path = extra.pop("model_path", None)
    manifest = {
        "git_commit": _git_commit(),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": versions,
        **_gpu_info(),
        **extra,
    }
    if model_path:
        manifest["model_file"] = Path(model_path).name
        manifest["model_sha256"] = file_sha256(model_path)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
