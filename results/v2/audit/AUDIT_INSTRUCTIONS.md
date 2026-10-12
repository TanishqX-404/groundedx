# Explanation audit (50 items)

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

Rate independently (two raters if possible: copy the sheet, one per rater).
Write any remarks in `notes`.
