# Claims check (Phase 0)

Checked against branch `revision-minds` on 2026-10-05. The old root-level
prototype scripts (`rag_diagnose.py`, `kb_builder.py`, `evaluate.py`, ...) that
produced the old numbers are deleted. All revision runs use `src/groundedx` plus
`scripts/run_*.py`. Every claim below is covered by a test in `tests/test_claims.py`
(run `pytest`).

| # | Paper claim | Paper § | Code location | Status before | Status now | Test |
|---|---|---|---|---|---|---|
| C1 | Decoding constrained by a GBNF grammar generated from the 16-class taxonomy | IV-D | `generation/grammar.py`, `generation/llama_cpp_client.py` | **Missing.** The LLM path used `response_format=json_object`. The package grammar existed but was never used, and it emitted class names *without* JSON quotes, so it could not produce valid JSON. | Implemented. A real `LlamaGrammar` is passed on every call. The grammar fixes the key order, restricts `root_cause` to the taxonomy and `citations` to this query's retrieved IDs (zero-shot: only `[]`), and bounds string length. | `test_grammar_lists_exactly_the_taxonomy`, `test_grammar_compiles_with_llama_cpp`, `test_constrained_decode_cannot_emit_non_taxonomy_class` (needs `GROUNDEDX_TEST_MODEL`) |
| C2 | Out-of-taxonomy root cause is rejected as invalid, with no fallback | IV-D | `generation/validation.py` | **Violated.** `LlamaDiagnoser._complete` replaced an invalid class with a heuristic `classify_from_text` prediction. | Implemented. Nothing in `src/` falls back. Invalid outputs are recorded with a reason (`truncated` / `parse_error` / `schema_error` / `out_of_taxonomy` / `bad_citation`) and **scored as wrong**. | `test_validator_rejects_out_of_taxonomy_and_bad_json`, `test_no_fallback_path_in_source` |
| C3 | Citations outside the retrieved set are rejected | IV-D | `generation/validation.py` | **Missing** in the LLM path. | Implemented twice: the grammar makes it impossible, and the validator checks cited IDs ⊆ retrieved IDs. | `test_citation_validation` |
| C4 | Label-free KB: the retriever and prompt never see class labels | IV-A | `kb/build_kb.py`, `kb/scoring_labels.py` | **Violated.** 16 taxonomy cards containing `Diagnosis label: <CLASS>` were in the TF-IDF index, with a +0.20 score bonus. | Fixed. Cards are deleted and the index holds only 128 `case_XXXX` documents. Two further leaks were also fixed: (a) window IDs were `index % 16 → class`, and IDs are now assigned after shuffling and never enter the KB text; (b) simulator log lines named the **true class's** top-3 KPI effects, and they now describe the top-3 *observed* standardized deltas. | `test_no_label_leak_in_index`, `test_case_ids_and_window_ids_are_not_class_ordered`, `test_query_never_contains_label_or_metadata` |
| C4b | (R1-2) Remediation text may reveal the class | IV-A | `kb/build_kb.py` (`--kb-variant`) | Not addressed | Confirmed: remediation text is **unique per class**, so in the `full` KB it is a perfect label proxy. Variants `no_remediation` and `generic_remediation` are implemented, which makes **E3 mandatory**. | `test_remediation_uniqueness` |
| C5 | Faithfulness = 3 conditions (≥1 citation; all cited ∈ retrieved; ≥2 shared non-stopword content terms) | VI-C | `faithfulness/grounding_metric.py` | **Missing** in the reported numbers. The old figure was `citation_fault_match`, a class-match proxy built from scoring labels. | Implemented as a standalone function and used in `scripts/run_llm.py`. It never reads scoring labels. Invalid outputs count as not grounded. | `test_faithfulness_*` (one positive and one negative case per condition) |
| C6 | Zero-shot uses the identical prompt, grammar and decoding, with no evidence and forced-empty citations | IV-E | `generation/prompt_templates.py` | Partial: a separate hand-written prompt string | Implemented: one template. The diff is exactly the evidence block plus a one-line citation rule. The grammar admits only `[]`. | `test_zero_shot_prompt_differs_only_in_evidence` |
| C7 | 3 seeds `[7, 19, 41]` | VI-D | `experiment.py`, `configs/experiment.yaml` | **Missing.** Everything was single-run. | Implemented. A seed controls dataset generation, the 70/15/15 split, the stratified KB case sample (8/class), and RF/GBDT `random_state`. Decoding is greedy (temperature 0), so it is deterministic. | `test_seeds_change_dataset_and_kb` |
| C8 | Peak VRAM measured via NVML during generation | VI-C | `evaluation/run_profiling.py`, `scripts/run_profile.py` | **Missing.** Only before/after snapshots were taken. | Implemented: a 50 Hz polling thread records the device-memory max and power per call. Energy is the trapezoidal integral over the call. 5 warm-up calls + 100 stratified measured calls per seed. | `test_nvml_window_peak_and_energy` |
| C9 | Context encoding: first-3 vs last-5 minute delta, top-6 KPIs, ≤4 log lines | IV-B | `encoding/context_encoder.py` | Matches, with one undocumented detail | Matches. **The paper text must add** that a KPI is listed only if \|delta\| > max(4% of its start value, 0.03), and that ranking uses the **raw** delta magnitude, so large-unit KPIs (throughput, RSRP, availability) dominate the top-6. | `test_context_encoding_is_domain_agnostic` |

## Simulator facts the paper text must match (§V-A)

- 13 KPIs, 15-minute windows, class-specific step/ramp/spike shape × severity ~ U(0.25, 0.78).
- `true_alarm_probability = 0.50` is now **window-level**. Half of the windows contain *none* of their own class's alarm codes. The old code always forced at least one true alarm, which contradicted the paper.
- `extra_alarm_probability = 0.82`: one alarm code from another class is injected.
- `distractor_probability = 0.78`: a different class's first 3 KPI effects are blended at amplitude U(0.18, 0.42). The window records `secondary_class` and `blend_amplitude`.
- The old prototype's `OVERLAP_EFFECTS` table (extra hand-tuned class confusions, not described in the paper) is gone.
- Logs: 2 generic templates + 3 "`<kpi>` deviation observed with approximate delta ±x" lines for the largest *observed* standardized deltas. 2–5 lines are kept, and the encoder uses ≤4.

## Not fixable / out of scope

- Real testbed or operator data (R1-1, R2-1, R3-2): limitation only.
- TeleLogs-style fine-tuned reasoning baselines (R3-4): discussion only.
