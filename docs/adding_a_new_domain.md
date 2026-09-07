# Adding A New Domain

1. Create `domains/<name>/taxonomy.yaml` with `name` and a `taxonomy` list.
   Each class needs a unique `class`, optional `domain`, `alarms`, numeric
   `kpi_effects`, a `pattern` (`step`, `ramp`, or `spike`), and remediation.
2. Create `kpi_schema.yaml` with a `metrics` list. Each metric has `name`,
   `baseline`, and optionally `noise_fraction` and `unit`.
3. Add simulator settings such as `distractor_probability`, alarm rates, and
   domain-specific `log_templates`.
4. Add `prompt_template.jinja`. Use `{{ taxonomy }}`, `{{ evidence }}`,
   `{{ query }}`, and `{{ domain.name }}` rather than hardcoded class names.
5. Generate a smoke dataset and run the same pipeline:

   ```powershell
   python -m groundedx.pipeline --domain domains/<name> --n 64 --seed 7
   ```

6. Add a parameterized test for context encoding, simulator balance, and
   validation. Do not add imports from `domains/` to `src/groundedx/`.

The included `domains/sre-k8s/` configuration is the smallest worked example:
it covers crash loops, memory leaks, DNS failure, CPU contention, disk
pressure, and network partitions with Kubernetes-flavored metrics and alarms.

