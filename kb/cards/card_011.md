---
doc_type: fault_card
alarm_codes: ALM-NG-FLAP, ALM-SIGNALING-LOSS
---

# Fault signature card

Diagnosis label: S1_NG_INTERFACE_FLAP.

Domain: Core.

Signature: this fault is indicated by alarms ALM-NG-FLAP, ALM-SIGNALING-LOSS. The expected KPI signature is: setup_success_rate changes by -0.08, latency_ms changes by +22, drop_call_rate changes by +0.04. The temporal pattern is usually spike shaped over a diagnosis window. Root cause evidence is strongest when the listed alarms co-occur with the KPI deviations.

Remediation: Check NG/S1 link stability, transport errors, and endpoint keepalive settings.
