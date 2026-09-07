# Contributing

Install the development extras with `python -m pip install -e ".[dev]"`.
Run `ruff check src tests` and `pytest` before opening a pull request.

The CI job is intentionally CPU-only. Tests that require `llama-cpp-python`, a
GGUF model, CUDA, NVML, or a particular GPU are local/integration checks and
must not be made mandatory for the pull-request suite.

The core package must remain domain-neutral. Domain data belongs under
`domains/`; core code must receive a `DomainConfig` at runtime. Retrieval and
generation code must never import evaluator-only scoring labels. The boundary
tests are part of the contract.

## Dataset releases

Generated benchmark data is kept out of Git. Once the repository is enabled in
Zenodo, tagging a GitHub release such as `data-v1.0.0` should archive the
generated benchmark as a Zenodo release asset with a citable DOI. Keep the
generator and configuration in Git, and attach the generated dataset only to
the release.

