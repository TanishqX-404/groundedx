# RAN-Doc Repository Audit

Audit date: 2026-09-07

This is a first-pass inventory of the prototype in this workspace. It describes
what is present rather than what the paper or requested repository structure
would ideally contain.

## Proposed package name

**Proposed name: `groundedx`**

The name is intentionally domain-neutral and leaves room for O-RAN, SRE,
industrial, and security telemetry domains. The Python import name would also
be `groundedx`; the repository/project display name could be
`groundedx` or `GroundedX`. This name is a proposal only and should be
confirmed before it is used in paths, package metadata, or public-facing docs.

## What exists today

| Area | Current implementation | Location |
| --- | --- | --- |
| Synthetic benchmark generator | Generates 4,000 labeled windows by default, with KPI noise, step/ramp/spike effects, distractor effects, alarms, logs, train/validation/test splits, and a seed argument. | `simulator.py` |
| O-RAN taxonomy | One YAML file with 16 classes, alarm codes, KPI effects, temporal patterns, and remediation text. | `fault_taxonomy.yaml` |
| KB cards and cases | Writes taxonomy cards and label-blind-looking historical case Markdown files from the training split. | `kb_builder.py`, `kb/cards/`, `kb/cases/` |
| Retrieval | TF-IDF with unigram/bigram features and cosine similarity; cards receive a fixed score bonus. | `kb_builder.py` |
| Deterministic diagnosis | Heuristic retrieval-based classifier and a zero-shot heuristic ablation. | `rag_diagnose.py` |
| Local LLM path | Optional `llama-cpp-python` wrapper for GGUF chat completion with JSON-object response mode. | `rag_diagnose.py`, `run_llm_eval.py` |
| Baselines | Alarm-overlap rule baseline and class-balanced RandomForest over engineered KPI/alarm/log features. | `baselines.py`, `common.py` |
| Evaluation | Accuracy, macro-F1, retrieval recall/MRR, CSV prediction/table output, confusion and faithfulness plots. | `evaluate.py`, `kb_builder.py` |
| Profiling | Per-sample wall-clock latency, before/after NVML memory and power snapshots, and a joules proxy. | `profile_runs.py` |
| Existing artifacts | Generated splits, 128 historical cases, 16 cards, serialized TF-IDF index, scoring labels, result CSVs, and figures are present. | `data/`, `kb/`, `results/` |

## Missing or incomplete relative to the requested pipeline

These are gaps, not planned behavior that should be represented as already
implemented:

- No installable Python package, `pyproject.toml`, typed package layout, CLI
  entry points, license, citation metadata, contribution policy, or test suite.
- No domain-config object or runtime domain loading. All behavior assumes the
  single root-level O-RAN taxonomy file.
- No second worked domain. The requested lightweight SRE/Kubernetes example
  does not exist.
- The simulator is not generic: KPI baselines, overlap effects, plotting, log
  keywords, alarms, and sample fields are O-RAN-specific.
- The KB builder has no explicit leakage boundary. It writes labels into
  `scoring_labels.json`, reads labels from `train.pkl` while building document
  metadata, and fault-card text itself contains `Diagnosis label: ...`, which
  is visible to TF-IDF retrieval.
- The retriever is not fully config-driven: `sublinear_tf=True`, retrieval
  depth, feature limits, score bonus, and paths are not configuration fields.
- There is no generated GBNF grammar. LLM calls request a generic JSON object;
  taxonomy and citation constraints are prompt text rather than decoder-level
  constraints.
- Generation validation does not reject an out-of-taxonomy root cause or a
  citation outside the retrieved set. The LLM wrapper instead applies a
  fallback for an invalid root cause, and does not validate citation IDs.
- Prompt templates are hardcoded Python strings and identify the assistant as
  an O-RAN/RAN-Doc system. There is no Jinja/domain prompt file.
- The context encoder is a single O-RAN-specific `summarize_window` function;
  KPI names, top-delta behavior, log limit, and alarm formatting are not
  domain-configurable.
- The alarm-to-class heuristic in `rag_diagnose.py` is a hardcoded O-RAN map.
- Evaluation is single-run rather than a reproducible three-seed protocol.
  The evaluator uses sample labels directly and also reads `kb/scoring_labels.json`.
  There is no standalone evidence-groundedness metric with the requested three
  conditions: at least one citation, every citation retrieved, and at least
  two shared non-stopword content terms with cited text.
- The reported `citation_fault_match` is a class-match proxy, not the requested
  citation faithfulness/groundedness metric.
- Profiling is a script, but not a model-size/quantization/domain/mode/retrieval
  config sweep. It records before/after memory rather than an explicit peak
  VRAM measurement and uses a rough two-snapshot power estimate.
- `configs/model_configs.yaml` contains generator names and paths only; it does
  not configure the full experiment dimensions.
- `xgboost`, `faiss-cpu`, and `sentence-transformers` appear in requirements,
  but the current source does not use them for the shipped baseline/retriever
  path. There is no embedding-model retriever implementation.
- No CI, lint configuration, pre-commit configuration, boundary check, or
  documented GPU-test exclusion exists.
- No dataset card, Zenodo DOI placeholder, release automation documentation,
  benchmark reproduction guide, or domain-authoring guide exists.
- This directory is not currently a Git repository (`git status` reports
  "not a git repository"), so a `v0.1.0` tag or GitHub/Zenodo integration
  cannot be created from the current workspace until repository metadata is
  initialized or made available.

## O-RAN assumptions to remove from core code

The following assumptions must move behind a runtime domain configuration:

- `common.py`: root-level `fault_taxonomy.yaml`, fixed data paths, fixed
  O-RAN-ish log keyword list, and direct use of `fault_class`/`alarms` fields.
- `simulator.py`: `BASELINES`, `OVERLAP_EFFECTS`, alarm severities, generic log
  templates, O-RAN KPI names, O-RAN plotting fields, root taxonomy path, and
  the assumption that every fault has `domain`, `alarms`, `kpi_effects`,
  `pattern`, and `remediation` in this exact shape.
- `kb_builder.py`: taxonomy-card rendering, class-label parsing, taxonomy
  lookup by alarm, fixed KB paths, and the hidden label map construction.
- `rag_diagnose.py`: O-RAN prompt wording, taxonomy lookup, the hardcoded
  `ALM-*` to class map, fixed output fields, and the fixed retriever/index.
- `evaluate.py`: RAN-branded method names, root-level scoring-label path,
  taxonomy lookup, and fixed result/figure paths.
- `baselines.py`: taxonomy loading from the root file and feature extraction
  that assumes the current sample field names and log vocabulary.
- `profile_runs.py` and `run_llm_eval.py`: fixed dataset/index locations and
  model/runtime values that should be supplied by experiment configuration.
- Checked-in generated cards, cases, data, and figures encode the current
  O-RAN benchmark and should be treated as reference artifacts, not generic
  library inputs.

## Paper/code correspondence

Implemented in some form: synthetic labeled windows; an authored KB; TF-IDF
retrieval; deterministic RAG and zero-shot heuristic paths; rule and
RandomForest baselines; an optional local GGUF wrapper; accuracy/macro-F1 and
retrieval metrics; and basic profiling hooks.

Present but materially different from the described method: cards expose
diagnosis labels to the retriever, the LLM path uses JSON-object mode rather
than GBNF, and the faithfulness result is a citation/class-match proxy.

Not implemented in the current code: configurable multi-domain execution,
decoder-level grammar constraints, strict citation validation, the requested
standalone grounding utility, three-seed evaluation protocol, config-driven
profiling sweep, embedding-model retriever, and real testbed validation.

The existing result CSVs are preserved as prototype outputs. This audit does
not treat them as independently reproducible paper results without checking
the generation commands, model files, runtime versions, and seed protocol.

## Refactor implications and deviations from the requested tree

The requested structure is a sound target, but the current code is small
enough that a staged extraction is safer than a blind file move. The likely
path is:

1. Add a domain/config model and generic core modules while keeping temporary
   compatibility wrappers for the current root scripts.
2. Move O-RAN YAML and rendering details into `domains/oran/` and add the
   lightweight `domains/sre-k8s/` example.
3. Separate public, label-free KB documents from evaluator-only scoring labels.
4. Add validation, the standalone grounding utility, focused tests, packaging,
   docs, and CI.
5. Remove or clearly mark compatibility wrappers and generated artifacts once
   the new commands reproduce the current CPU benchmark.

No large-scale move or package rename has been made in this audit stage.
