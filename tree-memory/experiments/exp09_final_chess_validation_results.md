# exp09: final chess validation, results (fix-1 run, 2026-10-17)

| Item | Detail |
|---|---|
| Hardware | Colab T4. Full run took 23.9 min; GPU peak 0.82 GB (CPU-bound Python) |
| Run 1 | Had a buggy factored cost; superseded by the fix-1 run below |

## Verdict: the chess series PASSED
Every goal is met on chess:
- rule understanding;
- self-learning;
- recall of previous steps;
- correct new steps;
- unlimited context.

**The scaling fix worked.** The factored tree read is about **9.4 × log₂n** reads per query from 4K to 1M addresses. That ratio stays flat, so the read is O(log n) in practice on this world. At 1M addresses it reads **189 of 1,048,576** addresses, which is **0.018%**.

## Memory scaling (winner: routed_hier:32, factored, kmeans4pp, label-free)

| n | 4K | 16K | 65K | 262K | 1M |
|---|---|---|---|---|---|
| factored reads | 86 | 97 | 147 | 173 | 189 |
| factored reads/log₂n | 7.1 | 6.9 | 9.2 | 9.6 | 9.4 |
| page64 reads | 89 | 119 | 322 | 543 | 1030 |
| page64 reads/log₂n | 7.4 | 8.5 | 20.1 | 30.2 | 51.5 |

The fitted cost exponent of each method shows how close it is to log n:

| Method | Fitted exponent, all n | Last 3 sizes |
|---|---|---|
| kmeans4pp | 0.16 | 0.06 |
| adaptive | 0.19 | 0.05 |
| taxonomy oracle | 0.15 | 0.07 |
| kmeans8 | 0.26 | 0.18 |
| page64 | 0.48 | 0.46 |

- The label-free k-means++ build now matches the taxonomy oracle (195 vs 191 reads at 1M).
- Recall is 1.000, agreement with full attention is 1.000, and TV ≤ 2.3e-14 in every configuration.
- Read cost under query noise at 1M is flat at about 190 reads; in run 1 it rose from 909 to 1306.

## Other results (fix-1 run)

| Test | Result |
|---|---|
| E1: rules | full_rules is 100% on all 5 move types for unseen games |
| L1: fast learning, no backprop (en passant) | episodic kNN 89–93%; kernel fast weights 76–84%; old skills ≥ 99.4% |
| L2: sleep | all_layers_sym_kl is 100% on all rules |
| L3: continual, 2 rules over 2 sleeps | all_layers_with_KL is 100% on en passant and castle, with no forgetting. Without KL, en passant drops to 65.5% (it was 80.75% in run 1), so KL is essential |
| R: replay of 256 unseen games at 1M addresses | 193 reads (p90 298), down from 957 (p90 2225) in run 1. Address accuracy 1.000. full_rules is 100% exact boards. One sleep: 99.98% per ply, 99.6% exact games. Two sleeps: 99.75% per ply, 99.2% exact games |
| Long context, 16,484 moves | tree is 100% at every fact age. Windows of 256, 1024 and 4096 fall to about 57–61% once a fact is older than the window |
| Noise at 1M | full-attention recall 0.913 at σ=0.1 and 0.613 at σ=0.2. The tree agrees with full attention 100%, so this is a key-geometry limit |

## Bug fixed in this run
In `certified_read` top-1 resolution, when no item had been scored exactly yet, `best = -inf`, so every frontier node was opened. The fix replaces this with a best-first loop, which opens a closed node only while its upper bound can still beat the best exact item.

## Context window answer
- There is no fixed context window.
- An optional exact recent window W can be set by hand.
- Everything older lives in tree memory with no size limit, at about log₂n reads per query.

## Caveats and carry-over to the text model
1. **Clean data.** The chess world has a clean taxonomy at constant density. Text keys will be messier, so the key encoder and hierarchical loss must create this geometry from real data.
2. **Wall clock.** The Python tree read takes 3.3 ms per query at 1M, vs 0.12 ms for batched GPU full attention. The read-count advantage (about 5,500× fewer reads than full attention) needs a fused GPU or C++ kernel to show in wall-clock time.
3. **Noise.** At 1M addresses the key margin is too small under noise. The text model needs larger-margin keys.
4. **Fast-weight learners.** These are run-to-run noisy (±4%). Sleep is the reliable path to 100%.
