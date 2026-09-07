---
doc_type: fault_card
alarm_codes: ALM-VNF-CPU-HIGH, ALM-SERVICE-DELAY
---

# Fault signature card

Diagnosis label: VNF_CPU_EXHAUSTION.

Domain: Core.

Signature: this fault is indicated by alarms ALM-VNF-CPU-HIGH, ALM-SERVICE-DELAY. The expected KPI signature is: cpu_util_pct changes by +35, latency_ms changes by +25, setup_success_rate changes by -0.07. The temporal pattern is usually ramp shaped over a diagnosis window. Root cause evidence is strongest when the listed alarms co-occur with the KPI deviations.

Remediation: Scale the VNF, rebalance workload, and check noisy neighbor processes.
