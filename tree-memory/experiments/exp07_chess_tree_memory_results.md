# exp07 — Chess testbed for tree memory: results (run 2026-10-16, GTX 1070 Ti, 31.7 min)

Setup: real legal chess games (python-chess). 4,096 "master" games are the leaves of a complete opening tree with 6 levels × fanout 4, giving 262,144 memory addresses (game, square). A further 8,800 games train, stream and test the rule model. The model's only board state is tree memory: each move reads 7 squares (certified reads), a 241k-parameter rule transformer predicts the new contents, and changed squares are written as new versions.

## E1 — Does it understand the rules?
- Full-rules model, tested on unseen games: **100.0%** of moves exactly right, for every move type (quiet, capture, castle, en passant, promotion), in both test pools (76,884 and 65,655 transitions).
- The naive "piece jumps from→to" baseline scores 0% on castle, en passant and promotion.
- The model trained without en passant gets 0% on en passant and 100% on everything else.
- Caveat: the 7 squares to read are given by design. This measures learned transition rules, not legality checking or strategy.

## E2 — Does it learn by itself? (no-en-passant model meets en passant in a stream)
Final en-passant accuracy, and whether other rules held, at feedback 1.0 and 0.25:

| Learner | En passant (fb 1.0) | En passant (fb 0.25) | Other rules |
|---|---|---|---|
| Frozen | 0% | 0% | 100% |
| Episodic surprise memory (no backprop) | 92% after 77 examples; 93% final; online 90.9% | 90.7% | Quiet/capture ≥99.1% and castle 100% at every checkpoint |
| Fast-weight delta rule (no backprop) | Unstable, 14–98% between checkpoints; final 45% | 29% | Intact |
| Backprop SGD | Best final, 98.7% | 96.5% | Castle temporarily fell to 64% (fb 1.0) and 74% (fb 0.25) — forgetting |
| Sleep consolidation (episodic → adapter, store cleared) | **Failed**: 13.7% | — | Kept, 99.95% |

## E3 — Scaling at constant taxonomy density, n = 4k → 262k (eps = 0.01)
All arms: recall 1.0 on seen and unseen games, top-1 agreement with full attention 1.0, max TV ≤ 1e-4, KL ≤ 1e-3.

Paged tree at n = 262,144:

| Arm | Reads | Share of full | Exponent (all sizes) | Exponent (last 3 sizes) |
|---|---|---|---|---|
| flat:16 | 216,962 | 83% | 0.96 | — |
| hier:16 | 1,353 | 0.52% | 0.65 | — |
| **routed_hier:16** | **671** | **0.26% (391× fewer)** | **0.50** | 0.69 |
| routed_cert:16 | 4,412 | — | 0.48 | — |
| routed_cert_linear:16 | 3,914 | — | 0.51 | — |
| routed_cert_linear:8 | 20,675 | — | 0.71 | — |

- For routed_hier, cost/log2(n) went from 7.4 to 37.3 while n grew 64×. Not yet log n.
- Where the reads go (routed_hier, 262k): about 192 reads go to the 3 sibling game pages (games differing only in the last opening digit, 64 slots each), 97 to the target path and page, and about 390 to depths 1–4. p90 = 1,759.
- **Cert loss backfired**: when it ramped in, β fell from about 2,700 to about 150, because lowering β is the easiest way to shrink an overlap violation β(q·μ+R−1). Fix: detach or fix β in the cert term.
- 8-dim keys were 5× worse than 16-dim.
- Address-level trees (leaf = 8 addresses) were much worse than game pages for every arm.

## E4 — Previous steps + new steps (64 unseen games, 4,404 plies, 262k addresses in memory)
- Full rules + tree reads: **100% board exact after every ply, 100% of games exact.** This is identical to oracle reads and full attention.
- Tree reads averaged 756 versus 262,144 (347× fewer). Address reads were 100% correct.
- Previous steps: as-of square 100%, last-change ply 100%, full-board reconstruction at a past ply 100% (640 boards).
- Without supersession (equal-weight blend of versions) as-of accuracy is 82.9%. A soft recency bias (λ ≥ 0.1) gets 99.8–100%.
- No-en-passant model in the full loop (board exact per ply / games exact):

| Variant | Per ply | Games |
|---|---|---|
| Frozen | 61.8% | 21.9% |
| + fast weights | 69.5% | 29.7% |
| + episodic memory | **93.4%** | **82.8%** |
| + sleep adapter | 64.6% | 26.6% |

  One wrong write corrupts the rest of the game, so per-ply exactness compounds.

## Next fixes
1. Stop the cert loss from lowering β: detach β, or fix it after warm-up.
2. Cut the sibling-page term: smaller pages, or order slots inside the page by route key.
3. Fix sleep consolidation: the adapter on frozen h features does not transfer the episodic correction.
4. Stabilise fast weights: per-pattern gating instead of one shared output matrix.
