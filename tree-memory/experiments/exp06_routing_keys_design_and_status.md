# exp06: trained routing keys for certified tree search — run 1 (r16) FAILED the target

Goal: frozen Qwen2.5-0.5B + r=16 routing key/query per head (LoRA-style head on q/k, 8 rotary planes, 5.5M params) so that certified
B&B over a routing-key tree finds the true top-32 far keys near the exact-node-bound curve (260/296/347/398 reads at 2K/4K/8K/16K).

## Design (unchanged from the code)
Loss = listwise distillation on true top-32 + recall@M hinge + bound-aware hierarchical clustering (w=0.3, ramp steps 150-300) + variance term.
Read = certified best-first B&B on routing keys (box bound) -> exact top-M routing -> exact rescoring on 64-d keys -> top-32 + 1 background.

## Run 1 result (GTX 1070 Ti, 1500 steps x 2 samples, 83.6 min, ~3.3 s/step; held-out essays gap/gba/gh/goodtaste/hubs)
Held-out validation, n_far = 2048, M = 64:

| step | recall@64 of true top-32 | all 32 found | reads_box (certified) | reads_orc (exact routing node max) | reads_true (exact true-logit node max, same tree) |
|---|---|---|---|---|---|
| 0 (init = 8 frozen planes) | 0.594 | 4.9% | 1451 | 385 | 279 |
| 250 | 0.484 | 0.4% | 1257 | 403 | 263 |
| 500 | 0.299 | 0% | 982 | 417 | 268 |
| 1500 (final) | 0.279 | 0% | 989 | 423 | 271 |

Target at 2K: 260 reads with all 32 found. End-to-end pass-key evaluation not run yet.

Training-set averages: ranking-only phase (steps 0-160) recall 0.52 -> 0.59, ce 5.65 -> 5.26. After the clustering loss switched on:
cluster 6.5 -> 2.9 and reads 1800 -> 1100, but recall 0.59 -> 0.40 and ce 5.3 -> 10.7 (above uniform, log 2048 = 7.6: routing now
ranks the true top keys BELOW average).

## Diagnosis
1. Clustering loss is self-reinforcing: "protected" nodes are defined by the routing's OWN top-M, so true top-32 keys the routing
   currently misses count as negatives and get pushed further down. It destroys recall instead of tightening bounds around the right keys.
2. Ranking alone is weak: 16 dims reach only ~0.59 recall@64 (exp05 PCA gave similar), and ce barely moves -> capacity/optimisation
   limit of a 16-d linear head on h, not just the loss mix.
3. Tree structure is fine: reads_true ~265-270 at 2K, i.e. the routing-key tree already sits on the 260 oracle curve for the TRUE ranking.
4. Box bound in 16-d is 2.3-3.8x looser than the exact routing node max (989 vs 423) -> even a perfect ranker would need a tighter bound.

## Next (run 2)
* Protected set = TRUE top-32 (or true top-32 union routing top-M), never routing-only.
* Ranking first (clustering off until held-out recall@64 >= 0.95), then clustering with small weight + early stop on held-out recall.
* More capacity: r=32, or a 2-layer MLP routing head; train on more text (current train haystack only 32K tokens, 13 essays).
* Report recall at M = 128/256 too; consider per-node learned bound offsets instead of a pure box bound.
