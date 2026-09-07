# v0.1.0 Release Notes

## Included

- Generic `groundedx` package for telemetry, alert, and log diagnosis.
- Runtime domain configuration with O-RAN as the flagship reference domain.
- Lightweight `sre-k8s` worked example.
- Label-free KB construction and TF-IDF retrieval.
- Domain-derived prompt and GBNF grammar generation.
- Strict taxonomy and retrieved-citation validation.
- Standalone citation grounding metric.
- Synthetic simulator, classical baseline helpers, evaluation utilities, tests,
  CPU CI, and documentation for adding domains and reproducing the benchmark.

## Explicitly not included yet

- Embedding-model retriever.
- Real-world testbed validation.
- Multi-seed GPU profiling sweep.
- Additional shipped domains beyond O-RAN and SRE/Kubernetes.
- Redistribution of GGUF model weights or a Zenodo-assigned dataset DOI.

The source tree is prepared for a `v0.1.0` tag once it is connected to the
intended GitHub repository. This workspace currently has no `.git` metadata,
so the tag and GitHub/Zenodo enablement must be performed after repository
initialization or remote checkout.

