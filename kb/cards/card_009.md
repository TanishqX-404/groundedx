---
doc_type: fault_card
alarm_codes: ALM-PTP-UNLOCK, ALM-SYNC-LOSS
---

# Fault signature card

Diagnosis label: TIMING_SYNC_LOSS.

Domain: RAN.

Signature: this fault is indicated by alarms ALM-PTP-UNLOCK, ALM-SYNC-LOSS. The expected KPI signature is: handover_failure_rate changes by +0.08, latency_ms changes by +18, setup_success_rate changes by -0.06. The temporal pattern is usually spike shaped over a diagnosis window. Root cause evidence is strongest when the listed alarms co-occur with the KPI deviations.

Remediation: Restore PTP/GNSS timing source and verify synchronization lock.
