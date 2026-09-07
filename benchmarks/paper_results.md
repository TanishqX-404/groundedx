# Paper Benchmark Reference

These are the tables transcribed from the supplied manuscript. They are
reference values, not a claim that the current legacy root scripts reproduce
them without the released model weights, exact harness, and dataset artifact.

## Main Results

Full 600-window test split, mean +/- standard deviation over three evaluation
seeds.

| Method | Top-1 | Macro-F1 | Faithfulness | Latency | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rule-based expert system | 0.332 +/- 0.004 | 0.356 +/- 0.003 | - | - | - |
| RandomForest (structured) | 0.571 +/- 0.003 | 0.599 +/- 0.004 | - | - | - |
| Zero-shot SLM (3B-Q4) | 0.520 +/- 0.003 | 0.480 +/- 0.004 | 0.00 | 680 ms | 2.1 GB |
| RAG-SLM (3B-Q4) | 0.700 +/- 0.003 | 0.670 +/- 0.004 | 0.75 | 1,150 ms | 2.3 GB |

## Eight-Way Sweep

Target hardware: NVIDIA RTX 3050 laptop GPU, 4 GB VRAM, 16 GB system RAM.

| Config | Mode | Top-1 | Latency | Peak VRAM | Avg. power | Energy/call |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1.5B-Q4 | zero-shot | 0.42 +/- 0.03 | 380 ms | 1.3 GB | 28 W | 10.6 J |
| 1.5B-Q4 | RAG | 0.58 +/- 0.04 | 650 ms | 1.4 GB | 30 W | 19.5 J |
| 1.5B-Q8 | zero-shot | 0.45 +/- 0.03 | 520 ms | 1.9 GB | 31 W | 16.1 J |
| 1.5B-Q8 | RAG | 0.61 +/- 0.04 | 850 ms | 2.0 GB | 33 W | 28.1 J |
| 3B-Q4 | zero-shot | 0.52 +/- 0.03 | 680 ms | 2.1 GB | 34 W | 23.1 J |
| 3B-Q4 | RAG | 0.70 +/- 0.03 | 1,150 ms | 2.3 GB | 37 W | 42.6 J |
| 3B-Q8 | zero-shot | 0.56 +/- 0.04 | 980 ms | 3.3 GB | 39 W | 38.2 J |
| 3B-Q8 | RAG | 0.72 +/- 0.03 | 1,550 ms | 3.5 GB | 42 W | 65.1 J |

## Retrieval Ablation

| k | Recall@k | MRR@k |
| ---: | ---: | ---: |
| 1 | 0.42 | 0.42 |
| 3 | 0.63 | 0.51 |
| 5 | 0.72 | 0.54 |
| 10 | 0.83 | 0.56 |

## Interpretation and limitations

RAG improves top-1 accuracy by 18.0 points over the same zero-shot SLM and
12.9 points over the RandomForest baseline. The paper reports 0.75 verified
groundedness for RAG and 0.00 for zero-shot. The eight-way profiling sweep is
one 25-call pass per configuration; six sweep accuracy values are estimated
from that smaller harness, and real O-RAN testbed validation is future work.
