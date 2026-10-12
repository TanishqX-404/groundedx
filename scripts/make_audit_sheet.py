"""Human audit sheet for SLM explanations (supervisor review, item C).

Samples 50 test windows from the 3B-Q4 top-1 hybrid (25 with a correct and 25
with a wrong classifier decision, balanced over seeds) and writes:
- results/v2/audit/audit_sheet.csv: what a rater sees (no gold label);
- results/v2/audit/audit_key.csv: gold label, correctness, cited cases' classes;
- results/v2/audit/AUDIT_INSTRUCTIONS.md: the rating questions.
Raters fill the q1..q4 columns; scripts/make_results.py can then summarize them.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundedx.calls import describe_fault  # noqa: E402
from groundedx.encoding.context_encoder import encode_context  # noqa: E402
from groundedx.experiment import RESULTS_ROOT, prepare_seed, read_jsonl  # noqa: E402

EXP = "llm__qwen2.5-3b-q4__hybrid1__full__k5"
PER_GROUP = 25
INSTRUCTIONS = """# Explanation audit (50 items)

Open `audit_sheet.csv`. For each row read the observation, the predicted class
and its reference description, the explanation, the remediation and the cited
cases. Do NOT open `audit_key.csv` until all ratings are done.

Fill each question column with Y, N or P (partly):

- q1_consistent: Is every factual statement in the explanation (KPIs, directions,
  alarms) consistent with the observation window?
- q2_evidence_supports: Do the cited cases actually support the predicted class
  (similar alarms/KPI pattern), rather than merely sharing words?
- q3_remediation_fits: Is the remediation an appropriate action for the
  predicted class?
- q4_useful: Would this explanation help an operator decide whether to trust
  the predicted class?

Rate independently (two raters if possible: copy the sheet, one per rater,
and save it as `ratings_<name>.csv` in this folder; then run
`python scripts/summarize_audit.py` and `python scripts/make_results.py`).
Write any remarks in `notes`.
"""


def main() -> None:
    rng = random.Random(2027)
    rows = []
    for seed in (7, 19, 41):
        for r in read_jsonl(RESULTS_ROOT / EXP / f"seed_{seed}" / "predictions.jsonl"):
            rows.append((seed, r))
    correct = [x for x in rows if x[1]["prediction"] == x[1]["gold"]]
    wrong = [x for x in rows if x[1]["prediction"] != x[1]["gold"]]
    chosen = rng.sample(correct, PER_GROUP) + rng.sample(wrong, PER_GROUP)
    rng.shuffle(chosen)
    setups = {}
    sheet, key = [], []
    for n, (seed, r) in enumerate(chosen, start=1):
        if seed not in setups:
            setups[seed] = prepare_seed(seed)
        setup = setups[seed]
        window = next(s for s in setup.splits["test"] if s["id"] == r["id"])
        texts = {d["chunk_id"]: d["text"] for d in setup.documents}
        fault = setup.domain.fault(r["prediction"])
        sheet.append(
            {
                "audit_id": f"A{n:02d}",
                "observation": encode_context(window, setup.domain),
                "predicted_class": r["prediction"],
                "class_reference": describe_fault(fault),
                "explanation": r["explanation"],
                "remediation": r["remediation"],
                "cited_cases": "\n\n".join(f"[{c}] {texts[c]}" for c in r["citations"]),
                "q1_consistent": "",
                "q2_evidence_supports": "",
                "q3_remediation_fits": "",
                "q4_useful": "",
                "notes": "",
            }
        )
        key.append(
            {
                "audit_id": f"A{n:02d}",
                "seed": seed,
                "window_id": r["id"],
                "gold_class": r["gold"],
                "prediction_correct": r["prediction"] == r["gold"],
                "cited_case_classes": json.dumps(
                    {c: setup.scoring_labels[c] for c in r["citations"]}
                ),
                "lexical_grounded": r["grounded"],
            }
        )
    out = RESULTS_ROOT / "audit"
    out.mkdir(exist_ok=True)
    pd.DataFrame(sheet).to_csv(out / "audit_sheet.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv(out / "audit_key.csv", index=False, encoding="utf-8-sig")
    (out / "AUDIT_INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8")
    print(f"wrote {len(sheet)} audit items to {out}")


if __name__ == "__main__":
    main()
