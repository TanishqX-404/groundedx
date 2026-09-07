---
doc_type: fault_card
alarm_codes: ALM-HO-DEGRADE, ALM-A3-OFFSET
---

# Fault signature card

Diagnosis label: HANDOVER_PARAM_MISCONFIG.

Domain: RAN.

Signature: this fault is indicated by alarms ALM-HO-DEGRADE, ALM-A3-OFFSET. The expected KPI signature is: handover_failure_rate changes by +0.09, ping_pong_rate changes by +0.07. The temporal pattern is usually step shaped over a diagnosis window. Root cause evidence is strongest when the listed alarms co-occur with the KPI deviations.

Remediation: Restore handover thresholds, hysteresis, and time-to-trigger values.
