# exp08 — Fixes for exp07 failures: results (Colab T4, 29.0 min, 2026-10-16)

Same chess testbed as exp07 (4,096 games, 262,144 memory addresses).

## S — Sleep consolidation: FIXED
All rows are measured after the episodic store is cleared. 427 surprise transitions were replayed (1,708 with symmetry).

| Method | En passant | Castle | Promotion | Quiet/capture |
|---|---|---|---|---|
| exp07 adapter on frozen features | 33.7% | 100% | 100% | 99.9% |
| Fine-tune last layer + replay + KL distillation | 96.7% | 100% | 99.9% | 100% |
| **Fine-tune last layer + chess-symmetry replay** (winner) | **97.9%** | 100% | 99.9% | 100% |
| Fine-tune all layers + symmetry replay | 92.8% | 100% | 100% | 100% |

- In the full replay of unseen games, the sleep model (store cleared) scored **99.66% board-exact per ply, 96.9% of games exact**, and 100% on as-of and past-board queries. The episodic memory alone scored 92.3% / 79.7%, and full rules 100% / 100%.
- Sleep model + the old episodic store did worse: 98.7% / 93.8%. The store is keyed on pre-sleep features, which go stale after sleep. Rule: clear or re-key the episodic store after every sleep.
- Fine-tuning only the last layer beat fine-tuning everything: less forgetting and less overfitting to 427 examples.

## F — Fast weights (zero backprop): partly fixed
En passant, at full feedback:

| Method | Final | Worst in 2nd half | Online |
|---|---|---|---|
| Delta rule (exp07) | 21% | 21% | — |
| Sparse random-feature delta | 16% | 16% | — |
| **Kernel fast weights** (dual form, τ = 120, α = 40) | 82% | 64% | 82% |
| Episodic kNN (θ = 0.95) | 94% | 92.5% | 90.7% |

- Old rules stayed at ≥ 97% for every method.
- Kernel fast weights are much better than the delta rule, but still noisier than hard nearest-neighbour episodic memory.
- In the full replay: episodic 92.3% per ply, kernel fast weights 88.2%.
- Conclusion: in-session fast memory = episodic kNN; permanent knowledge = sleep.

## M — Memory scaling (exact everywhere except one arm)
Agreement with full attention was 1.000 in every row. Recall was 1.000 everywhere except routed_cert_detach (0.998).

| Arm / tree | Reads at 262k | Exponent (all sizes) | Exponent (last 3) |
|---|---|---|---|
| **routed_hier:16 / page64** (winner) | **640 (0.24%)** | 0.48 | 0.69 |
| routed_hier:16 / subpage8 | 1,202 | 0.55 | — |
| routed_hier:16 / taxonomy-built tree (oracle control) | **463** | **0.33** | **0.11** |
| routed_hier:32 / page64 | 740 | 0.53 | — |
| cert, β detached | 4,242 | 0.95 | — |
| cert, β fixed at 2,000 | 3,758 | 0.91 | — |

- The cert loss itself is harmful, not just β collapse. With β protected, it pushed geometry wrong: about 2,500–2,950 reads at shared-prefix depth 3, and NCE no longer converged (1.37). **Drop the cert loss.**
- Sub-pages add node overhead. Sibling-page reads fell from 185 to 133, but upper-level reads rose.
- 32-dim keys give no gain over 16-dim.
- **Key finding: the k-means tree build is a large share of the remaining cost.** The taxonomy-built tree almost stops growing at large n (exponent 0.11 over the last three sizes). Even it still pays about 133 reads on sibling pages plus about 170 at depths 1–2.

## R — Replay of 64 unseen games
- Tree reads averaged 752 of 262,144 (349× fewer); p90 was 1,971. Address reads were 100% correct.

## Next (exp09)
1. Build the tree to match the learned hierarchy without labels: fanout 4 to match the training fanout, k-means restarts, or agglomerative/balanced clustering on route keys. The target is the oracle curve.
2. Factorized sibling read: slot keys are shared across games, so an opened sibling page should score only the matching slot (1 read instead of 64).
3. Keep sleep = last-layer fine-tune + symmetry replay + KL distillation; clear the episodic store after each sleep.
4. Drop the cert loss and the sparse/delta fast weights.
