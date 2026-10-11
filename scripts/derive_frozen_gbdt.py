"""Single GBDT for all reporting: the classifier used inside the hybrid runs.

Every hybrid run stores the per-window top-1 of the GBDT it was given
(``classifier_top1``). That GBDT is trained once per seed with the
validation-selected hyper-parameters on the inference host. This script
checks that all hybrid runs agree on it window by window, then writes it as
the experiment ``gbdt_frozen`` so baselines and hybrids are compared against
one and the same classifier.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundedx.experiment import RESULTS_ROOT, read_jsonl, run_dir, write_manifest  # noqa: E402


def main() -> None:
    sources = sorted(
        p.name for p in RESULTS_ROOT.glob("llm__*__hybrid*") if not p.name.endswith("__val")
    )
    if not sources:
        raise SystemExit("no hybrid runs found")
    for seed in (7, 19, 41):
        reference: dict[str, dict] | None = None
        for exp in sources:
            path = RESULTS_ROOT / exp / f"seed_{seed}" / "predictions.jsonl"
            if not path.exists():
                continue
            rows = {r["id"]: r for r in read_jsonl(path)}
            if reference is None:
                reference, ref_exp = rows, exp
                continue
            diff = sum(rows[i]["classifier_top1"] != reference[i]["classifier_top1"] for i in rows)
            if diff:
                raise SystemExit(f"seed {seed}: {exp} disagrees with {ref_exp} on {diff} windows")
        assert reference is not None
        out = run_dir("gbdt_frozen", seed)
        with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for r in reference.values():
                handle.write(
                    json.dumps(
                        {
                            "id": r["id"],
                            "gold": r["gold"],
                            "secondary_class": r["secondary_class"],
                            "true_alarm_emitted": r["true_alarm_emitted"],
                            "prediction": r["classifier_top1"],
                            "valid": True,
                        }
                    )
                    + "\n"
                )
        src_manifest = json.loads(
            (RESULTS_ROOT / ref_exp / f"seed_{seed}" / "manifest.json").read_text()
        )
        write_manifest(
            out,
            seed,
            derived_from=sources,
            gbdt_params=src_manifest.get("gbdt_params"),
            source_xgboost=src_manifest.get("package_versions", {}).get("xgboost"),
            source_platform=src_manifest.get("platform"),
        )
        print(f"seed {seed}: gbdt_frozen from {len(sources)} hybrid runs (all agree)")


if __name__ == "__main__":
    main()
