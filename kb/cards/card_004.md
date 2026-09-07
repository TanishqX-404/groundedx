---
doc_type: fault_card
alarm_codes: ALM-UL-INTERFERENCE, ALM-SINR-LOW
---

# Fault signature card

Diagnosis label: INTERFERENCE_SPIKE.

Domain: RAN.

Signature: this fault is indicated by alarms ALM-UL-INTERFERENCE, ALM-SINR-LOW. The expected KPI signature is: sinr_db changes by -7.0, packet_loss_pct changes by +1.8, throughput_mbps changes by -22. The temporal pattern is usually spike shaped over a diagnosis window. Root cause evidence is strongest when the listed alarms co-occur with the KPI deviations.

Remediation: Locate interference source and adjust power, tilt, or frequency planning.
