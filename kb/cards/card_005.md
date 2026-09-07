---
doc_type: fault_card
alarm_codes: ALM-NBR-MISSING, ALM-HO-DEGRADE
---

# Fault signature card

Diagnosis label: NEIGHBOR_LIST_MISCONFIG.

Domain: RAN.

Signature: this fault is indicated by alarms ALM-NBR-MISSING, ALM-HO-DEGRADE. The expected KPI signature is: handover_failure_rate changes by +0.1, drop_call_rate changes by +0.05. The temporal pattern is usually step shaped over a diagnosis window. Root cause evidence is strongest when the listed alarms co-occur with the KPI deviations.

Remediation: Audit and repair neighbor relations, then verify mobility counters.
