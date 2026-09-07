---
doc_type: fault_card
alarm_codes: ALM-TEMP-HIGH, ALM-RRU-DEGRADE
---

# Fault signature card

Diagnosis label: COOLING_SYSTEM_FAILURE.

Domain: Site.

Signature: this fault is indicated by alarms ALM-TEMP-HIGH, ALM-RRU-DEGRADE. The expected KPI signature is: availability_pct changes by -2.5, throughput_mbps changes by -16, cpu_util_pct changes by +15. The temporal pattern is usually ramp shaped over a diagnosis window. Root cause evidence is strongest when the listed alarms co-occur with the KPI deviations.

Remediation: Restore site cooling and verify radio/baseband thermal status.
