# GenIntAI Research

Open research from **GenIntAI** — *General Intrinsic Artificial Intelligence* · https://genintai.in

We work on AI whose memory, learning, reasoning and self-correction are built into the architecture. This repository holds our reports and experiment logs, including the experiments that failed.

## Programmes

| Programme | Status | Where |
|---|---|---|
| **Tree-Memory architecture** — log-time certified memory, no context window | Validated on a chess memory world; language-model conversion in research | [`tree-memory/`](tree-memory/) |
| **Ternary-SLM** — small language model (~47M) with hybrid ternary weights: bucket {−1, 0, +1} plus exact outliers | Prototype stage; design and data plan documented | [`ternary-slm/`](ternary-slm/) |
| **SHVQ-LR** — codebook + low-rank-residual weight compression | Prototype | coming |
| **Graph-Compressed Embeddings (GCE)** | Coming soon | coming |

## Tree-Memory

- [`tree_memory_v1_and_v2.md`](tree-memory/tree_memory_v1_and_v2.md) — architectures, prototype results, conversion and from-scratch routes, tested fixes
- [`experiments/`](tree-memory/experiments/) — experiment logs, oldest to newest:
  1. [exp01–05](tree-memory/experiments/exp01-05_real_model_conversion_root_cause.md) — why a tree of real-model keys did not prune
  2. [exp06](tree-memory/experiments/exp06_routing_keys_design_and_status.md) — trained routing keys, run 1 (missed the target)
  3. [exp07](tree-memory/experiments/exp07_chess_tree_memory_results.md) — chess testbed
  4. [exp08](tree-memory/experiments/exp08_failure_fixes_results.md) — fixes for exp07 failures
  5. [exp09](tree-memory/experiments/exp09_final_chess_validation_results.md) — final chess validation (passed): ≈9.4 × log₂ n reads up to 1M addresses

## How we report

Every result is measured against full attention (recall, KL divergence, top-1 agreement) across memory sizes up to 1M+, with the cost growth exponent fitted. Negative results are kept.

## Contact

contact@genintai.in · [LinkedIn](https://in.linkedin.com/company/genintai)

> Reports describe work in progress. Numbers come from the testbeds stated in each document.
