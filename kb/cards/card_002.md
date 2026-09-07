---
doc_type: fault_card
alarm_codes: ALM-BH-LATENCY, ALM-PACKET-LOSS
---

# Fault signature card

Diagnosis label: BACKHAUL_CONGESTION.

Domain: Transport.

Signature: this fault is indicated by alarms ALM-BH-LATENCY, ALM-PACKET-LOSS. The expected KPI signature is: latency_ms changes by +35, packet_loss_pct changes by +2.5, throughput_mbps changes by -28. The temporal pattern is usually ramp shaped over a diagnosis window. Root cause evidence is strongest when the listed alarms co-occur with the KPI deviations.

Remediation: Check transport queue utilization, reroute traffic, or increase backhaul capacity.
