---
doc_type: fault_card
alarm_codes: ALM-PA-FAULT, ALM-TX-POWER-LOW
---

# Fault signature card

Diagnosis label: POWER_AMPLIFIER_FAULT.

Domain: RAN.

Signature: this fault is indicated by alarms ALM-PA-FAULT, ALM-TX-POWER-LOW. The expected KPI signature is: rsrp_dbm changes by -10.0, throughput_mbps changes by -20, availability_pct changes by -2.0. The temporal pattern is usually step shaped over a diagnosis window. Root cause evidence is strongest when the listed alarms co-occur with the KPI deviations.

Remediation: Inspect PA module and replace or recalibrate the transmitter chain.
