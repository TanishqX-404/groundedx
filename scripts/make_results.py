"""Aggregate every run under results/v2/ into one place.

Outputs (all regenerated from raw per-window files; nothing is typed by hand):
  results/v2/RESULTS.md     human-readable tables, mean +/- std over n seeds
  results/v2/numbers.json   flat {key: value} map of every reported number
  results/v2/tables/*.csv   per-seed metric tables

Accuracy always comes from full-split predictions.jsonl, never from profiling.
Invalid generations count as wrong. Std is the sample std (ddof=1) over the
seeds actually present; n is always reported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundedx.config import load_domain_bundle  # noqa: E402
from groundedx.experiment import (  # noqa: E402
    RESULTS_ROOT,
    ROOT,
    load_experiment_config,
    read_jsonl,
)
from groundedx.generation.validation import INVALID_REASONS  # noqa: E402

CFG = load_experiment_config()
CLASSES = load_domain_bundle(ROOT / CFG["domain"]).classes
NUMBERS: dict[str, float | int | str | None] = {}


def seed_dirs(exp: str) -> list[Path]:
    return sorted((RESULTS_ROOT / exp).glob("seed_*"), key=lambda p: int(p.name.split("_")[1]))


def experiments(prefix: str = "", filename: str = "predictions.jsonl") -> list[str]:
    if not RESULTS_ROOT.exists():
        return []
    return sorted(
        p.name
        for p in RESULTS_ROOT.iterdir()
        if p.is_dir() and p.name.startswith(prefix) and any(p.glob(f"seed_*/{filename}"))
    )


def acc(rows: list[dict]) -> float | None:
    return float(np.mean([r["prediction"] == r["gold"] for r in rows])) if rows else None


def summarize(values: list[float | None]) -> dict:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return {
        "mean": mean(vals) if vals else None,
        "std": stdev(vals) if len(vals) > 1 else None,
        "n": len(vals),
    }


def fmt(stat: dict, digits: int = 3, scale: float = 1.0) -> str:
    if stat["mean"] is None:
        return "-"
    text = f"{stat['mean'] * scale:.{digits}f}"
    if stat["std"] is not None:
        text += f" ± {stat['std'] * scale:.{digits}f}"
    return text + f" (n={stat['n']})"


def record(key: str, stat: dict) -> None:
    NUMBERS[f"{key}.mean"] = stat["mean"]
    NUMBERS[f"{key}.std"] = stat["std"]
    NUMBERS[f"{key}.n_seeds"] = stat["n"]


# ---- reference subsets: same-class case in full-KB top-5 (E6) ----------------


def retrieval_hits(seed: int, k: int = 5) -> dict[str, bool]:
    path = RESULTS_ROOT / "retrieval__full" / f"seed_{seed}" / "ranks.jsonl"
    return {
        r["id"]: r["same_class_rank"] is not None and r["same_class_rank"] <= k
        for r in read_jsonl(path)
    }


def seed_metrics(exp: str, seed_dir: Path) -> dict:
    rows = read_jsonl(seed_dir / "predictions.jsonl")
    seed = int(seed_dir.name.split("_")[1])
    for r in rows:
        if not r.get("valid", True):
            r["prediction"] = None  # invalid -> scored wrong
    gold = [r["gold"] for r in rows]
    pred = [r["prediction"] or "__INVALID__" for r in rows]
    clean = [r for r in rows if r["secondary_class"] is None]
    blended = [r for r in rows if r["secondary_class"] is not None]
    wrong_blended = [r for r in blended if r["prediction"] != r["gold"]]
    hits = retrieval_hits(seed)
    m = {
        "experiment": exp,
        "seed": seed,
        "n": len(rows),
        "accuracy": acc(rows),
        "macro_f1": float(f1_score(gold, pred, labels=CLASSES, average="macro", zero_division=0)),
        "invalid_rate": float(np.mean([not r.get("valid", True) for r in rows])),
        "acc_clean": acc(clean),
        "acc_blended": acc(blended),
        "acc_blended_primary_or_secondary": (
            float(np.mean([r["prediction"] in (r["gold"], r["secondary_class"]) for r in blended]))
            if blended
            else None
        ),
        "wrong_blended_equals_secondary": (
            float(np.mean([r["prediction"] == r["secondary_class"] for r in wrong_blended]))
            if wrong_blended
            else None
        ),
        "acc_alarm_absent": acc([r for r in rows if not r["true_alarm_emitted"]]),
        "acc_alarm_present": acc([r for r in rows if r["true_alarm_emitted"]]),
        "acc_retrieval_hit": acc([r for r in rows if hits.get(r["id"]) is True]),
        "acc_retrieval_miss": acc([r for r in rows if hits.get(r["id"]) is False]),
    }
    for reason in INVALID_REASONS:
        m[f"invalid_{reason}"] = float(np.mean([r.get("invalid_reason") == reason for r in rows]))
    if exp.startswith("llm__"):
        m["faithfulness"] = float(np.mean([bool(r.get("grounded")) for r in rows]))
        m["mean_citations"] = float(np.mean([len(r.get("citations") or []) for r in rows]))
        lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        m["eval_median_latency_ms"] = median(lat) if lat else None
    if any(r.get("classifier_top1") for r in rows):
        # Hybrid: how the SLM's choice relates to the classifier it reranks.
        top1 = [r["classifier_top1"] for r in rows]
        n = len(rows)
        overrides = [r for r, t in zip(rows, top1) if r["prediction"] != t]
        m["classifier_top1_acc"] = float(np.mean([t == r["gold"] for r, t in zip(rows, top1)]))
        m["candidate_recall"] = float(
            np.mean([r["gold"] in [c for c, _ in r["candidates"]] for r in rows])
        )
        m["override_rate"] = len(overrides) / n
        m["override_fixes"] = sum(r["prediction"] == r["gold"] != t for r, t in zip(rows, top1)) / n
        m["override_breaks"] = (
            sum(t == r["gold"] != r["prediction"] for r, t in zip(rows, top1)) / n
        )
    per_class = f1_score(gold, pred, labels=CLASSES, average=None, zero_division=0)
    m["per_class_f1"] = dict(zip(CLASSES, map(float, per_class)))
    return m


METRIC_COLUMNS = [
    "accuracy",
    "macro_f1",
    "invalid_rate",
    "faithfulness",
    "acc_clean",
    "acc_blended",
    "acc_blended_primary_or_secondary",
    "wrong_blended_equals_secondary",
    "acc_alarm_absent",
    "acc_alarm_present",
    "acc_retrieval_hit",
    "acc_retrieval_miss",
    *[f"invalid_{r}" for r in INVALID_REASONS],
    "mean_citations",
    "eval_median_latency_ms",
    "classifier_top1_acc",
    "candidate_recall",
    "override_rate",
    "override_fixes",
    "override_breaks",
]


M3 = "llm__qwen2.5-3b-q4"
M7 = "llm__qwen2.5-7b-q4"
KEY_RUNS = [
    "rule",
    "random_forest",
    "gbdt",
    "gbdt_frozen",
    "knn_vote__full__k5",
    f"{M3}__zs",
    f"{M3}__rag__full__k5",
    f"{M3}__zs__docs",
    f"{M3}__rag__full__k5__docs",
    f"{M3}__hybrid3__full__k5",
    f"{M3}__hybrid1__full__k5",
]
PAIRED_TESTS = [
    ("3B RAG vs zero-shot", f"{M3}__rag__full__k5", f"{M3}__zs"),
    ("3B zero-shot+docs vs zero-shot", f"{M3}__zs__docs", f"{M3}__zs"),
    ("3B zero-shot+docs vs RAG+docs", f"{M3}__zs__docs", f"{M3}__rag__full__k5__docs"),
    ("3B RAG k=1 vs k=5", f"{M3}__rag__full__k1", f"{M3}__rag__full__k5"),
    ("3B RAG k=3 vs k=5", f"{M3}__rag__full__k3", f"{M3}__rag__full__k5"),
    ("3B RAG full vs no-remediation KB", f"{M3}__rag__full__k5", f"{M3}__rag__no_remediation__k5"),
    (
        "3B RAG full vs generic-remediation KB",
        f"{M3}__rag__full__k5",
        f"{M3}__rag__generic_remediation__k5",
    ),
    ("kNN full vs no-remediation KB", "knn_vote__full__k5", "knn_vote__no_remediation__k5"),
    ("3B RAG vs kNN-vote", f"{M3}__rag__full__k5", "knn_vote__full__k5"),
    ("GBDT (frozen) vs RandomForest", "gbdt_frozen", "random_forest"),
    ("GBDT (frozen) vs 3B hybrid top-3", "gbdt_frozen", f"{M3}__hybrid3__full__k5"),
    (
        "GBDT (frozen) vs 1.5B hybrid top-3",
        "gbdt_frozen",
        "llm__qwen2.5-1.5b-q4__hybrid3__full__k5",
    ),
    (
        "3B hybrid top-3 blind vs top-3",
        f"{M3}__hybrid3__full__k5__blind",
        f"{M3}__hybrid3__full__k5",
    ),
    ("GBDT (frozen) vs 3B hybrid top-3 blind", "gbdt_frozen", f"{M3}__hybrid3__full__k5__blind"),
    (
        "3B RAG label-exposed KB vs controlled",
        f"{M3}__rag__label_exposed__k5",
        f"{M3}__rag__full__k5",
    ),
    (
        "3B RAG label-exposed KB + leaky logs vs controlled",
        f"{M3}__rag__label_exposed__k5__data-leaky",
        f"{M3}__rag__full__k5",
    ),
    ("3B zero-shot leaky logs vs controlled", f"{M3}__zs__data-leaky", f"{M3}__zs"),
    (
        "3B RAG+docs KB-all vs KB-128",
        f"{M3}__rag__full__k5__docs__kball",
        f"{M3}__rag__full__k5__docs",
    ),
    ("3B zero-shot+docs z-ranking vs raw", f"{M3}__zs__docs__enc-z", f"{M3}__zs__docs"),
    (
        "3B RAG+docs KB-all+z vs KB-128 raw",
        f"{M3}__rag__full__k5__docs__kball__enc-z",
        f"{M3}__rag__full__k5__docs",
    ),
    ("7B zero-shot+docs vs 3B", f"{M7}__zs__docs", f"{M3}__zs__docs"),
    ("7B RAG+docs vs 3B", f"{M7}__rag__full__k5__docs", f"{M3}__rag__full__k5__docs"),
    ("7B zero-shot+docs vs RAG+docs", f"{M7}__zs__docs", f"{M7}__rag__full__k5__docs"),
    (
        "GBDT (frozen) vs 7B RAG+docs KB-all+z",
        "gbdt_frozen",
        f"{M7}__rag__full__k5__docs__kball__enc-z",
    ),
]
EXPLANATION_COLUMNS = [
    "grounded",
    "ctrl_generic",
    "ctrl_shuffled",
    "cite_any_consistent",
    "cite_all_consistent",
    "remed_matches_pred",
    "remed_matches_gold",
    "entity_mentions",
    "entity_unsupported",
    "grounded_and_wrong",
]


def correct_by_window(exp: str) -> dict[str, bool] | None:
    out: dict[str, bool] = {}
    for d in seed_dirs(exp):
        for r in read_jsonl(d / "predictions.jsonl"):
            out[r["id"]] = bool(r.get("valid", True)) and r["prediction"] == r["gold"]
    return out or None


def mcnemar(a: str, b: str) -> dict | None:
    """Exact McNemar test on windows both runs scored (only when all seeds complete)."""

    from scipy.stats import binomtest

    if not (RESULTS_ROOT / a).exists() or not (RESULTS_ROOT / b).exists():
        return None
    ca, cb = correct_by_window(a), correct_by_window(b)
    if not ca or not cb:
        return None
    ids = sorted(ca.keys() & cb.keys())
    if len(ids) < 1800:
        return None
    n10 = sum(ca[i] and not cb[i] for i in ids)
    n01 = sum(cb[i] and not ca[i] for i in ids)
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return {
        "n": len(ids),
        "acc_a": sum(ca[i] for i in ids) / len(ids),
        "acc_b": sum(cb[i] for i in ids) / len(ids),
        "n10": n10,
        "n01": n01,
        "p_value": float(p),
    }


def main() -> None:
    out_tables = RESULTS_ROOT / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    md: list[str] = [
        "# Revision results (auto-generated by scripts/make_results.py — do not edit)",
        "",
        "Values are mean ± sample std over n seeds (seeds control dataset generation, "
        "split, KB case sampling and tree-model random_state; decoding is greedy). "
        "Invalid generations are scored as wrong. Full numbers: `numbers.json`; per-seed: `tables/`.",
        "",
    ]

    # ---------------- dataset ----------------
    ds = {
        d.name: json.loads((d / "stats.json").read_text())
        for d in seed_dirs("dataset")
        if (d / "stats.json").exists()
    }
    if ds:
        md += [
            "## Dataset (per seed)",
            "",
            "| seed | train/val/test | test blended | test true-alarm emitted | test with distractor alarm | KB docs | test per-class min–max |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, s in ds.items():
            t = s["test"]
            pc = list(t["per_class"].values())
            md.append(
                f"| {name} | {s['train']['n']}/{s['val']['n']}/{t['n']} | {t['blended']} ({t['blended'] / t['n']:.1%}) | {t['true_alarm_emitted']} ({t['true_alarm_emitted'] / t['n']:.1%}) | {t['with_distractor_alarm']} ({t['with_distractor_alarm'] / t['n']:.1%}) | {s['kb_docs']} | {min(pc)}–{max(pc)} |"
            )
            NUMBERS[f"dataset.{name}.test_blended_frac"] = t["blended"] / t["n"]
            NUMBERS[f"dataset.{name}.test_true_alarm_frac"] = t["true_alarm_emitted"] / t["n"]
            NUMBERS[f"dataset.{name}.test_n"] = t["n"]
        md.append("")

    # ---------------- per-experiment metrics ----------------
    all_rows = []
    for exp in experiments():
        for d in seed_dirs(exp):
            if (d / "predictions.jsonl").exists():
                all_rows.append(seed_metrics(exp, d))
    stats: dict[str, dict[str, dict]] = {}
    if all_rows:
        frame = pd.DataFrame(
            [{k: v for k, v in r.items() if k != "per_class_f1"} for r in all_rows]
        )
        frame.to_csv(out_tables / "per_seed_metrics.csv", index=False)
        for exp, group in frame.groupby("experiment"):
            stats[exp] = {}
            for col in METRIC_COLUMNS:
                if col in group:
                    stat = summarize(list(group[col]))
                    stats[exp][col] = stat
                    record(f"{exp}.{col}", stat)
            NUMBERS[f"{exp}.windows_per_seed"] = ",".join(str(int(v)) for v in group["n"])

        pilots = sorted(e for e in stats if e.endswith("__val"))
        main_exps = sorted(e for e in stats if not e.endswith("__val"))
        md += [
            "## All accuracy runs (full split)",
            "",
            "| experiment | windows/seed | accuracy | macro-F1 | invalid rate | faithfulness |",
            "|---|---|---|---|---|---|",
        ]
        for exp in main_exps:
            s = stats[exp]
            md.append(
                f"| {exp} | {NUMBERS[f'{exp}.windows_per_seed']} | {fmt(s['accuracy'])} | {fmt(s['macro_f1'])} | {fmt(s['invalid_rate'])} | {fmt(s.get('faithfulness', summarize([])))} |"
            )
        md.append("")

        md += [
            "## E4 multi-fault / E8 alarm-absent / E6 retrieval-conditioned accuracy",
            "",
            "Retrieval hit = a same-class case is in the full-KB TF-IDF top-5 for that window (same subsets for every method).",
            "",
            "| experiment | clean | blended | blended: primary OR secondary | wrong(blended)=secondary | alarm absent | alarm present | retrieval hit | retrieval miss |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for exp in main_exps:
            s = stats[exp]
            md.append(
                "| "
                + " | ".join(
                    [exp]
                    + [
                        fmt(s[c])
                        for c in [
                            "acc_clean",
                            "acc_blended",
                            "acc_blended_primary_or_secondary",
                            "wrong_blended_equals_secondary",
                            "acc_alarm_absent",
                            "acc_alarm_present",
                            "acc_retrieval_hit",
                            "acc_retrieval_miss",
                        ]
                    ]
                )
                + " |"
            )
        md.append("")

        hybrid = [e for e in sorted(stats) if "__hybrid" in e]
        if hybrid:
            md += [
                "## Hybrid (GBDT top-N candidates -> SLM choice + grounded explanation)",
                "",
                "classifier top-1 = GBDT alone on the same windows; candidate recall = gold in the "
                "N candidates (hybrid ceiling); fixes/breaks = fraction of windows where the SLM's "
                "override of GBDT top-1 turned wrong->right / right->wrong. `__val` = validation pilot.",
                "",
                "| experiment | windows/seed | hybrid acc | classifier top-1 | candidate recall | override rate | fixes | breaks | faithfulness |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for exp in hybrid:
                s = stats[exp]
                cells = [
                    fmt(s[c])
                    for c in [
                        "accuracy",
                        "classifier_top1_acc",
                        "candidate_recall",
                        "override_rate",
                        "override_fixes",
                        "override_breaks",
                        "faithfulness",
                    ]
                ]
                md.append(
                    f"| {exp} | {NUMBERS[f'{exp}.windows_per_seed']} | " + " | ".join(cells) + " |"
                )
            md.append("")
        if pilots:
            md += [
                "## Validation-split pilots (not test results)",
                "",
                "| experiment | windows/seed | accuracy | macro-F1 | invalid rate | faithfulness |",
                "|---|---|---|---|---|---|",
            ]
            for exp in pilots:
                s = stats[exp]
                md.append(
                    f"| {exp} | {NUMBERS[f'{exp}.windows_per_seed']} | {fmt(s['accuracy'])} | {fmt(s['macro_f1'])} | {fmt(s['invalid_rate'])} | {fmt(s.get('faithfulness', summarize([])))} |"
                )
            md.append("")

        llm = [e for e in main_exps if e.startswith("llm__")]
        if llm:
            md += [
                "## E5 invalid outputs by reason (fraction of windows)",
                "",
                "| experiment | " + " | ".join(INVALID_REASONS) + " | mean #citations |",
                "|---|" + "---|" * (len(INVALID_REASONS) + 1),
            ]
            for exp in llm:
                md.append(
                    "| "
                    + " | ".join(
                        [exp]
                        + [fmt(stats[exp][f"invalid_{r}"]) for r in INVALID_REASONS]
                        + [fmt(stats[exp]["mean_citations"], 2)]
                    )
                    + " |"
                )
            md.append("")

        # per-class F1 (Table III style), every pair zs vs rag-full-k5 of the same model
        pcs = []
        for r in all_rows:
            for label, value in r["per_class_f1"].items():
                pcs.append(
                    {"experiment": r["experiment"], "seed": r["seed"], "class": label, "f1": value}
                )
        pc = pd.DataFrame(pcs)
        pc.to_csv(out_tables / "per_class_f1_per_seed.csv", index=False)
        for exp in llm:
            if not exp.endswith("__zs"):
                continue
            model = exp.split("__")[1]
            rag = f"llm__{model}__rag__full__k5"
            if rag not in stats:
                continue
            md += [
                f"## Per-class F1: {model} zero-shot vs RAG (full KB, k=5), sorted by Δ",
                "",
                "| class | zero-shot | RAG | Δ |",
                "|---|---|---|---|",
            ]
            rows = []
            for label in CLASSES:
                z = summarize(list(pc[(pc.experiment == exp) & (pc["class"] == label)].f1))
                g = summarize(list(pc[(pc.experiment == rag) & (pc["class"] == label)].f1))
                rows.append((g["mean"] - z["mean"], label, z, g))
                record(f"per_class_f1.{model}.zs.{label}", z)
                record(f"per_class_f1.{model}.rag.{label}", g)
            for delta, label, z, g in sorted(rows, reverse=True):
                md.append(f"| {label} | {fmt(z, 2)} | {fmt(g, 2)} | {delta:+.2f} |")
            md.append("")

    # ---------------- retrieval ----------------
    rret = []
    for exp in experiments("retrieval__", "ranks.jsonl"):
        for d in seed_dirs(exp):
            ranks = [r["same_class_rank"] for r in read_jsonl(d / "ranks.jsonl")]
            for k in CFG["retrieval"]["ablation_k"]:
                rret.append(
                    {
                        "experiment": exp,
                        "seed": int(d.name.split("_")[1]),
                        "k": k,
                        "recall": float(np.mean([r is not None and r <= k for r in ranks])),
                        "mrr": float(
                            np.mean([1.0 / r if r is not None and r <= k else 0.0 for r in ranks])
                        ),
                    }
                )
    if rret:
        rf = pd.DataFrame(rret)
        rf.to_csv(out_tables / "retrieval_per_seed.csv", index=False)
        md += [
            "## E7 retrieval quality (same-class case in top-k)",
            "",
            "| KB variant | k | Recall@k | MRR@k |",
            "|---|---|---|---|",
        ]
        for (exp, k), g in rf.groupby(["experiment", "k"]):
            r, m = summarize(list(g.recall)), summarize(list(g.mrr))
            record(f"{exp}.recall@{k}", r)
            record(f"{exp}.mrr@{k}", m)
            md.append(f"| {exp[len('retrieval__') :]} | {k} | {fmt(r)} | {fmt(m)} |")
        md.append("")

    # ---------------- profiling ----------------
    prof = []
    for exp in experiments("profile__", "calls.jsonl"):
        for d in seed_dirs(exp):
            calls = [c for c in read_jsonl(d / "calls.jsonl") if c["phase"] == "measured"]
            if not calls:
                continue
            man = json.loads((d / "manifest.json").read_text())
            # End-to-end per diagnosis: classifier (hybrid only) + SLM call.
            clf = np.array([c.get("classifier_ms", 0.0) for c in calls])
            lat = np.array([c["latency_ms"] for c in calls]) + clf
            peaks = [c["peak_mem_bytes"] for c in calls if c["peak_mem_bytes"] is not None]
            power = [c["mean_power_w"] for c in calls if c["mean_power_w"] is not None]
            energy = [c["energy_j"] for c in calls if c["energy_j"] is not None]
            prof.append(
                {
                    "experiment": exp,
                    "seed": int(d.name.split("_")[1]),
                    "calls": len(calls),
                    "gpu": (man.get("gpus") or ["?"])[0],
                    "median_latency_ms": float(np.median(lat)),
                    "p95_latency_ms": float(np.percentile(lat, 95)),
                    "median_classifier_ms": float(np.median(clf)),
                    "peak_vram_gb": max(peaks) / 1e9 if peaks else None,
                    "peak_vram_over_idle_gb": (max(peaks) - man["mem_before_load_bytes"]) / 1e9
                    if peaks
                    else None,
                    "mean_power_w": float(np.mean(power)) if power else None,
                    "energy_per_call_j": float(np.mean(energy)) if energy else None,
                }
            )
    if prof:
        pf = pd.DataFrame(prof)
        pf.to_csv(out_tables / "profiling_per_seed.csv", index=False)
        cols = [
            "median_latency_ms",
            "p95_latency_ms",
            "median_classifier_ms",
            "peak_vram_gb",
            "peak_vram_over_idle_gb",
            "mean_power_w",
            "energy_per_call_j",
        ]
        md += [
            "## E2 profiling (100 measured calls/seed after 5 warm-up; NVML @ 50 Hz; latency is end-to-end incl. classifier)",
            "",
            "| experiment | GPU | calls/seed | " + " | ".join(cols) + " |",
            "|---|---|---|" + "---|" * len(cols),
        ]
        for exp, g in pf.groupby("experiment"):
            cells = []
            for c in cols:
                s = summarize(list(g[c]))
                record(f"{exp}.{c}", s)
                cells.append(fmt(s, 0 if "ms" in c else 2))
            md.append(
                f"| {exp} | {g.gpu.iloc[0]} | {int(g.calls.iloc[0])} | " + " | ".join(cells) + " |"
            )
        md.append("")

    # ---------------- per-seed accuracy (key runs) ----------------
    if all_rows:
        md += [
            "## Per-seed test accuracy (key runs)",
            "",
            "| experiment | seed 7 | seed 19 | seed 41 |",
            "|---|---|---|---|",
        ]
        for exp in [e for e in KEY_RUNS if e in stats]:
            per = {r["seed"]: r["accuracy"] for r in all_rows if r["experiment"] == exp}
            cells = " | ".join(f"{per.get(s, float('nan')):.4f}" for s in (7, 19, 41))
            md.append(f"| {exp} | {cells} |")
        md.append("")

    # ---------------- paired significance tests ----------------
    paired = []
    for name, a, b in PAIRED_TESTS:
        res = mcnemar(a, b)
        if res is None:
            continue
        paired.append({"comparison": name, "a": a, "b": b, **res})
        for key in ("acc_a", "acc_b", "n10", "n01", "p_value"):
            NUMBERS[f"paired.{a}.vs.{b}.{key}"] = res[key]
    if paired:
        pd.DataFrame(paired).to_csv(out_tables / "paired_tests.csv", index=False)
        md += [
            "## Paired tests (exact McNemar over the pooled test windows of all seeds)",
            "",
            "n10 = windows only A gets right; n01 = windows only B gets right.",
            "",
            "| comparison | A | B | n10 / n01 | p |",
            "|---|---|---|---|---|",
        ]
        for r in paired:
            md.append(
                f"| {r['comparison']} | {r['acc_a']:.3f} | {r['acc_b']:.3f} | "
                f"{r['n10']} / {r['n01']} | {r['p_value']:.2g} |"
            )
        md.append("")

    # ---------------- explanation checks ----------------
    exp_csv = out_tables / "explanation_checks_per_seed.csv"
    if exp_csv.exists():
        ec = pd.read_csv(exp_csv)
        cols = [c for c in EXPLANATION_COLUMNS if c in ec.columns]
        md += [
            "## Explanation checks (scripts/explanation_checks.py; evaluator-side)",
            "",
            "grounded = 3-condition lexical metric; ctrl_generic / ctrl_shuffled = the same metric "
            "for a fixed generic sentence / another window's explanation (negative controls); "
            "cite_* = cited cases whose class equals the prediction; remed_* = the generated "
            "remediation's closest runbook is the predicted / gold class's; entity_unsupported = "
            "among explanations naming a KPI or alarm, the share naming one absent from the "
            "window's own observation.",
            "",
            "| experiment | " + " | ".join(cols) + " |",
            "|---|" + "---|" * len(cols),
        ]
        for exp, g in ec.groupby("experiment"):
            cells = []
            for c in cols:
                st = summarize(list(g[c]))
                record(f"explain.{exp}.{c}", st)
                cells.append(fmt(st))
            md.append(f"| {exp} | " + " | ".join(cells) + " |")
        md.append("")

    # ---------------- difficulty sweep ----------------
    sweep_csv = out_tables / "difficulty_sweep_per_seed.csv"
    if sweep_csv.exists():
        sw = pd.read_csv(sweep_csv)
        metrics = ["rule", "random_forest", "gbdt", "knn_vote", "recall_at_5"]
        md += [
            "## Difficulty sweep (classical baselines + retrieval; extra-alarm probability 0.82)",
            "",
            "| P(own alarm) | P(blend) | seeds | " + " | ".join(metrics) + " |",
            "|---|---|---|" + "---|" * len(metrics),
        ]
        for (ta, bl), g in sw.groupby(["true_alarm_probability", "distractor_probability"]):
            cells = []
            for c in metrics:
                st = summarize(list(g[c]))
                record(f"sweep.alarm{ta}.blend{bl}.{c}", st)
                cells.append(fmt(st))
            md.append(f"| {ta} | {bl} | {len(g)} | " + " | ".join(cells) + " |")
        md.append("")

    # ---------------- provenance (from manifests) ----------------
    prov = []
    for exp_dir in sorted(
        p for p in RESULTS_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")
    ):
        mans = [json.loads(m.read_text()) for m in sorted(exp_dir.glob("seed_*/manifest.json"))]
        if not mans:
            continue

        def distinct(key, sub=None):
            vals = {str((m.get(sub) or {}).get(key) if sub else m.get(key)) for m in mans}
            return ", ".join(sorted(vals))

        gpus = {(m.get("gpus") or ["CPU"])[0].split(",")[0] for m in mans}
        prov.append(
            f"| {exp_dir.name} | {', '.join(sorted(gpus))} | {distinct('platform').split('-')[0]} | "
            f"{distinct('xgboost', 'package_versions')} | {distinct('llama_server_build')} | "
            f"{', '.join(sorted({str(m.get('git_commit'))[:7] for m in mans}))} |"
        )
    md += [
        "## Provenance (from each run's manifest.json)",
        "",
        "Hybrid runs train their own GBDT (same validation-selected hyper-parameters) on the machine that runs them, "
        "so 'classifier top-1' in the hybrid table can differ slightly from the `gbdt` baseline row when the XGBoost build differs.",
        "",
        "| experiment | GPU | OS | xgboost | llama-server | commit |",
        "|---|---|---|---|---|---|",
        *prov,
        "",
    ]

    (RESULTS_ROOT / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RESULTS_ROOT / "numbers.json").write_text(
        json.dumps(NUMBERS, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {RESULTS_ROOT / 'RESULTS.md'} ({len(NUMBERS)} numbers)")


if __name__ == "__main__":
    main()
