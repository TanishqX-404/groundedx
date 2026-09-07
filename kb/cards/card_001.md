---
doc_type: fault_card
alarm_codes: ALM-PCI-CONFLICT, ALM-HO-DEGRADE
---

# Fault signature card

Diagnosis label: PCI_COLLISION.

Domain: RAN.

Signature: this fault is indicated by alarms ALM-PCI-CONFLICT, ALM-HO-DEGRADE. The expected KPI signature is: handover_failure_rate changes by +0.12, rsrq_db changes by -3.0, sinr_db changes by -4.0. The temporal pattern is usually step shaped over a diagnosis window. Root cause evidence is strongest when the listed alarms co-occur with the KPI deviations.

Remediation: Reassign physical cell IDs and validate neighbor PCI uniqueness.
