---
doc_type: fault_card
alarm_codes: ALM-CONFIG-CHANGE, ALM-KPI-REGRESSION
---

# Fault signature card

Diagnosis label: CONFIG_ROLLBACK_REQUIRED.

Domain: RAN.

Signature: this fault is indicated by alarms ALM-CONFIG-CHANGE, ALM-KPI-REGRESSION. The expected KPI signature is: throughput_mbps changes by -18, setup_success_rate changes by -0.05, drop_call_rate changes by +0.03. The temporal pattern is usually step shaped over a diagnosis window. Root cause evidence is strongest when the listed alarms co-occur with the KPI deviations.

Remediation: Roll back recent configuration change and perform controlled revalidation.
