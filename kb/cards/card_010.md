---
doc_type: fault_card
alarm_codes: ALM-CELL-OVERLOAD, ALM-ADMISSION-REJECT
---

# Fault signature card

Diagnosis label: OVERLOAD_ADMISSION_CONTROL.

Domain: RAN.

Signature: this fault is indicated by alarms ALM-CELL-OVERLOAD, ALM-ADMISSION-REJECT. The expected KPI signature is: prb_util_pct changes by +30, setup_success_rate changes by -0.1, latency_ms changes by +20. The temporal pattern is usually ramp shaped over a diagnosis window. Root cause evidence is strongest when the listed alarms co-occur with the KPI deviations.

Remediation: Tune admission thresholds, add capacity, or offload users to neighbors.
