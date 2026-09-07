---
doc_type: fault_card
alarm_codes: ALM-QOS-POLICY, ALM-SLICE-SLA
---

# Fault signature card

Diagnosis label: SLICE_QOS_POLICY_ERROR.

Domain: Core.

Signature: this fault is indicated by alarms ALM-QOS-POLICY, ALM-SLICE-SLA. The expected KPI signature is: latency_ms changes by +28, throughput_mbps changes by -24, packet_loss_pct changes by +1.2. The temporal pattern is usually step shaped over a diagnosis window. Root cause evidence is strongest when the listed alarms co-occur with the KPI deviations.

Remediation: Correct slice QoS policy mapping and reapply SLA profiles.
