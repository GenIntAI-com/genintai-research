# exp01–exp05: zero-shot tree-memory conversion of Qwen2.5-0.5B, root cause and fix

Task: pass-key retrieval in Paul Graham essays, n = 2K–16K. Run on a cloud sandbox (Xeon 2.8 GHz x2, 7 GB, no GPU).
Sinks (4) + local window (64) + question tokens always exact; "far" = the rest of the context.

## 1. exp01 result
Certified eps-rule tree read, ball/cone bound: answer identical everywhere, KL ~1e-10, but reads = 183–196 % of
dense at every n (opens every node). Pre-RoPE keys: same. Cost is O(n) with a worse constant than dense.

## 2. Root cause (exp02, exp04)
| n | box-bound B&B (certified top-32) | exact node max (oracle bound) | ball-bound tree (exp01) |
|---|---|---|---|
| 2K | 2,418 reads | 260 | 3,705 |
| 4K | 4,356 | 296 | 7,294 |
| 8K | 7,981 | 347 | 14,259 |
| 16K | 14,599 | 398 | 27,746 |

* The tree STRUCTURE is fine: with exact per-node maxima, certified top-32 search costs about k·log n
  (+~46 reads per doubling).
* The BOUND is the failure: computable bounds are 4–17x wider than the true score spread inside a node
  (median half-width, ball: 10–35 logits vs actual 0.6–8 logits), so nothing can be pruned.
* Training-free tightening failed: rotated boxes (key-PCA, query-PCA) stay O(n); low-dim routing
  (PCA up to 32 dims, best RoPE planes up to 16 planes) loses recall as n grows (e.g. 32-dim PCA, 512 candidates:
  96% recall at 2K, 84% at 8K). Pretrained keys use all 64 dims / all 32 RoPE planes to rank far tokens.
* The eps target was also wrong: keeping all but 1% of attention mass needs ~18–21% of far keys at EVERY n
  (2K: 21%, 4K: 19%, 8K: 20%, 16K: 18%) -> mass certification is Theta(n) for a pretrained model, by any index.

## 3. What the answer actually needs (exp03, end to end)
| n | top-32 exact + 1 background summary | top-128 + bg | HNSW(ef=64) top-32 + bg |
|---|---|---|---|
| 2K | identical, KL 0.0017, 32 reads | KL 0.0003 | identical, 604 reads (30%) |
| 4K | identical, KL 0.0049 | KL 0.0006 | – |
| 8K | identical, KL 0.016 | KL 0.0069 | identical, 839 reads (10%) |
| 16K | identical, KL 0.0065 | KL 0.0028 | identical, 928 reads (5.7%) |

Budgeted tree reads with the loose bound (exp03 tree64/256/1024) give WRONG answers: the bound steers search to
the wrong nodes.

## 4. Fix
1. Read objective: certified exact top-k + background summary (not eps of total mass).
2. Make bounds tight by TRAINING the keys (the step the conversion plan already requires): dedicated low-dim
   routing sub-key per head (Matryoshka key), ranking-distillation loss from the frozen model's top-k +
   hierarchical clustering loss, LoRA on q/k. Target curve = the oracle row above (k·log n). Needs a GPU.
3. Works today (not certified): HNSW + background, identical answers to 16K, reads ~n^0.2.
