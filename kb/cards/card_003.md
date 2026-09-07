---
doc_type: fault_card
alarm_codes: ALM-RRU-DEGRADE, ALM-TX-POWER-LOW
---

# Fault signature card

Diagnosis label: HARDWARE_DEGRADATION.

Domain: RAN.

Signature: this fault is indicated by alarms ALM-RRU-DEGRADE, ALM-TX-POWER-LOW. The expected KPI signature is: rsrp_dbm changes by -8.0, throughput_mbps changes by -18, availability_pct changes by -1.5. The temporal pattern is usually ramp shaped over a diagnosis window. Root cause evidence is strongest when the listed alarms co-occur with the KPI deviations.

Remediation: Inspect RRU health, cabling, and replace degraded radio hardware if confirmed.
