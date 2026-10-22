# Ternary-SLM — Hybrid Quantized Semantic Transformer

A small decoder-only language model (~47M parameters, GPT-2 tokenizer, English only, trained from scratch) whose stored weights use a **hybrid ternary format** instead of pure `{-1, 0, +1}`.

**Status:** prototype stage. The design is documented; the model is not yet a fully trained or validated result.

## The idea

Every stored weight (and embedding) is in one of two regimes:

| Regime | Values | Role |
|---|---|---|
| **Bucket** | `{-1, 0, +1}` | Cheap, sparse, ternary-like |
| **Exact** | `[-9.99, -1.00] ∪ [1.00, 9.99]`, two decimals | Outliers that carry strong semantic match |

Values are read on a **semantic match scale**: `-9.99` = strong anti-match, `0` = neutral, `+1.00` = balance point, `+9.99` = strongest match. Raw scores are squashed with `M(x) = 9.99 · tanh(x / κ)`.

Key design rules:

- **Quantize storage, not every temporary number.** Attention scores, logits and accumulators stay in FP32 or exact form; bucketing them destroys ranking.
- **Regime-switch optimizer.** Bucket weights use the ternary optimizer; large momentum promotes a weight to exact, `|V| < 1.00` demotes it, with a 100-step refractory period and a per-layer exact budget (`f_max = 10%`).
- **Honest size claim.** Hybrid does not beat pure ternary on size: about 15–18 MB at 5–10% exact weights, versus about 11.7 MB at 0%. The point is expressiveness, not compression.
- **One change at a time.** Phase 1 keeps softmax attention, ternary boundary activations and a standard GLU FFN; semantic attention (Balance-Gate) and MatchGLU come later.

## Contents

| File | What it is |
|---|---|
| [`docs/Final_Research_Ternary.md`](docs/Final_Research_Ternary.md) | Full research plan: format, attention, FFN, optimizer, export size, training phases, ablations, risks |
| [`docs/DataSet_Collection_Technique.md`](docs/DataSet_Collection_Technique.md) | Dataset plan: token math, category mix, open sources, quality auditing, cleaning pipeline, tokenization |
| [`data/Dataset_Count_Summary.xlsx`](data/Dataset_Count_Summary.xlsx) | Per-category cleaned/raw token targets and document counts for a 3B-token corpus |

Note: the final recipe block at the end of the dataset document describes the earlier **pure-ternary** baseline. The hybrid design in the research plan supersedes it.

## Data plan in one line

About 4.9B raw tokens collected, about 3B cleaned: 30% Wikipedia, 30% educational web, 10% books, 10% StackExchange, 5% textbooks, 6.7% code, 5% math, 3.3% news.

## Planned phases

1. **Tiny prototype** — 1M–5M params, 2 layers, 10M–50M tokens
2. **Scaling test** — 10M–20M params, 4–6 layers, 100M–300M tokens
3. **Main model** — ~47M params, ~8 layers, 2.5B–4B cleaned tokens

Experiment logs and results will be added here as they are run, including the ones that fail.
