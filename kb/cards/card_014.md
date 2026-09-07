---
doc_type: fault_card
alarm_codes: ALM-FH-JITTER, ALM-ECPRI-DELAY
---

# Fault signature card

Diagnosis label: FRONTHAUL_JITTER.

Domain: Transport.

Signature: this fault is indicated by alarms ALM-FH-JITTER, ALM-ECPRI-DELAY. The expected KPI signature is: latency_ms changes by +24, sinr_db changes by -2.5, packet_loss_pct changes by +1.5. The temporal pattern is usually spike shaped over a diagnosis window. Root cause evidence is strongest when the listed alarms co-occur with the KPI deviations.

Remediation: Inspect fronthaul timing, transport queues, and eCPRI delay variation.
