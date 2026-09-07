---
doc_type: fault_card
alarm_codes: ALM-DNS-FAIL, ALM-SERVICE-DELAY
---

# Fault signature card

Diagnosis label: DNS_RESOLUTION_FAILURE.

Domain: Core.

Signature: this fault is indicated by alarms ALM-DNS-FAIL, ALM-SERVICE-DELAY. The expected KPI signature is: setup_success_rate changes by -0.06, latency_ms changes by +30. The temporal pattern is usually step shaped over a diagnosis window. Root cause evidence is strongest when the listed alarms co-occur with the KPI deviations.

Remediation: Restore DNS reachability and validate service discovery records.
