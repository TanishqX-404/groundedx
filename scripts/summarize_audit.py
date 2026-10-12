"""Summarize explanation-audit ratings (results/v2/audit/ratings_*.csv).

Each ratings file holds one rater's Y/P/N answers for q1..q4 (see
AUDIT_INSTRUCTIONS.md). Ratings are joined with audit_key.csv only here, after
rating, and summarized per rater overall and by classifier correctness. Writes
results/v2/tables/audit_summary.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundedx.experiment import RESULTS_ROOT  # noqa: E402

QUESTIONS = ["q1_consistent", "q2_evidence_supports", "q3_remediation_fits", "q4_useful"]


def main() -> None:
    audit = RESULTS_ROOT / "audit"
    key = pd.read_csv(audit / "audit_key.csv", encoding="utf-8-sig")
    sheet = pd.read_csv(audit / "audit_sheet.csv", encoding="utf-8-sig")
    rows = []
    for path in sorted(audit.glob("ratings_*.csv")):
        rater = path.stem[len("ratings_") :]
        df = pd.read_csv(path, encoding="utf-8-sig").merge(key, on="audit_id")
        if len(df) != len(key):
            raise SystemExit(f"{path.name}: {len(df)} rated items, expected {len(key)}")
        groups = {
            "all": df,
            "correct": df[df.prediction_correct],
            "wrong": df[~df.prediction_correct],
        }
        for group, g in groups.items():
            for q in QUESTIONS:
                answers = g[q].str.strip().str.upper()
                rows.append(
                    {
                        "rater": rater,
                        "subset": group,
                        "question": q,
                        "n": len(g),
                        "Y": float((answers == "Y").mean()),
                        "P": float((answers == "P").mean()),
                        "N": float((answers == "N").mean()),
                    }
                )
    if not rows:
        raise SystemExit("no ratings_*.csv files in " + str(audit))
    out = RESULTS_ROOT / "tables" / "audit_summary.csv"
    summary = pd.DataFrame(rows)
    # explanations cut off by the 240-character field bound (not ending a sentence)
    expl = sheet["explanation"].astype(str)
    truncated = float(((expl.str.len() >= 230) & ~expl.str.rstrip().str.endswith(".")).mean())
    summary["explanations_truncated"] = truncated
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
