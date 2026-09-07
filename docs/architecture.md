# Architecture

The active domain is loaded from `domains/<name>/taxonomy.yaml` and
`kpi_schema.yaml`. The generic code receives a `DomainConfig`; it does not
import domain files.

The KB builder produces observation documents containing metrics, alarms, logs,
and remediation text. It does not include `fault_class` in document text and
does not import evaluator labels. The evaluator owns the mapping from chunk ID
to scoring label.

At diagnosis time, the context encoder creates a compact query. TF-IDF returns
retrieved chunks. A Jinja prompt and domain-derived grammar constrain optional
local generation. The validator checks the output schema, active taxonomy, and
retrieved citation IDs. The standalone `groundedx.faithfulness.grounding_check`
utility then tests whether the explanation shares at least two content terms
with cited retrieved evidence.

