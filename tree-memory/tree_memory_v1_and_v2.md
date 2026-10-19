# Tree-Memory Architectures for Context-Free, $O(\log n)$ Language Models (v1)
**10 Oct 2026**

## Abstract
Four tree-memory architectures were built and run; the strongest, T4, answered every question in a 7,504-fact timeline correctly while reading about 300 entries per query instead of up to 7,504, but the $O(\log n)$ goal is not yet met.

The candidates share an append-only tree of exact facts. T1 reads it with a bounded search, T2 replaces it with fixed-size recurrent states, T3 adds a prediction-error loop and fast weights, and T4 adds a working window and sleep-time consolidation into slow weights. Single-head NumPy prototypes ($d=64$) showed three results.

* **Accuracy at lower cost, with conditions:** with clustered keys, certified tree reads matched full attention to $2\times 10^{-7}$ at 14 times fewer lookups for $n=16,384$, but cost grew as $n^{0.67}$ and random keys saved nothing.
* **Memory behaviour:** surprise-gated writes stored 300 leaves for a 6,000-token stream of 300 distinct facts, changed facts returned the latest value 100% of the time against 0% for full attention, and replay from the tree removed catastrophic forgetting in a toy network (task A accuracy 0% to 100%).
* **Where it falls short:** a fixed-size state (T2) lost exact facts after about 16 tokens, and the cheaper statistical routing reached 95 times fewer lookups at 5.6% mean error.

The next experiments target key geometry and tighter bounds, then a PyTorch port distilled from a pretrained model.

## Problem statement
Eleven limitations of current language models share one root: memory is either frozen weights or a transient cache that grows with the sequence length $n$.

| # | Limitation | Why it happens today | Target |
| :--- | :--- | :--- | :--- |
| 1 | Quadratic or linear cost | $QK^T$ is $n\times n$; decoding reads all $n$ cached keys | $O(d\cdot(\log n+B))$ per token |
| 2 | Bounded context window | Cache grows with $n$; windows cap near 1M tokens | Unbounded memory, $O(\log n)$ hot |
| 3 | One token at a time | One forward pass per generated token | $k$ tokens per memory read |
| 4 | Inference and cache cost | KV cache rebuilt and billed per session | Memory persists, written once |
| 5 | Frozen weights, knowledge cutoff | Weights are static after training | New facts usable at once, then consolidated |
| 6 | Ephemeral in-context learning | What is learned vanishes with the context | Persistent episodic memory |
| 7 | Catastrophic forgetting | New gradient steps overwrite old weights | Append-only memory, replay-based consolidation |
| 8 | No self-correction | No error signal at inference time | Prediction error (surprise) drives writes |
| 9 | Data drift | Static assumptions about the data | Timestamps, supersession, drift monitor |
| 10 | Costly updating | Backprop over large batches | Local $O(\log n)$ writes, small amortized updates |
| 11 | Not brain-like | One undifferentiated mechanism | Working, episodic and semantic stores |

## Shared machinery
All four candidates use one memory: an append-only ball tree whose leaves hold exact `(key, value)` facts and whose internal nodes summarise everything below them.

* **Node summary:** mean key $\mu$, radius $r$ (an upper bound on the distance from $\mu$ to any key below), value sum, count, and the sum of squared key norms, which gives the node's spread.
* **Write:** a new fact descends to the nearer child and updates the summaries on its path. A scapegoat-style rebuild (a node is rebuilt when one child holds over 80% of its leaves) kept mean depth at 16.2 and maximum depth at 24 for $n=16,384$, against $\log_2 n=14$.
* **Read:** best-first refinement. Start from the root summary and repeatedly open the node whose mass is most uncertain, until total uncertainty falls below $\epsilon$ of total mass or the budget $B$ of openings is spent.
* **Supersede:** when a new fact has the same key as a stored one but a different value, the old leaf is retired and removed from every ancestor, so the newest fact wins.
* **Consolidate:** replay stored leaves into slow weights offline (T4 only).

The bound on a node's attention mass, for a query $q$ in dimension $d$:

$$w_{\min} \le \sum_{i \in \text{node}} \exp\left(\frac{q \cdot k_i}{\sqrt{d}}\right) \le w_{\max}, \quad w_{\min, \max} = c \cdot \exp\left(\frac{q \cdot \mu \mp \|q\| r}{\sqrt{d}}\right)$$

Leaves have $r=0$, so they are exact. Two bound modes are implemented. *Certified* uses the radius above. *Statistical* replaces the radius with three standard deviations of the node's key spread: cheaper, but with no guarantee.

## Candidate architectures
The four candidates differ in what the memory holds and how it learns; T1 is the base the others add to.

| Candidate | Memory | Learns by | Read cost | Built to solve |
| :--- | :--- | :--- | :--- | :--- |
| **T1 Tree-Attention** | Exact leaves and summaries, append-only | Writes only; weights frozen | $O(d\cdot(\text{depth} + B))$ lookups | Cost, context length, cache cost |
| **T2 Multi-timescale recurrent** | $L$ fixed $d\times d$ matrices, one per timescale | Delta rule at each level | $O(L\cdot d^2)$ | Smallest state |
| **T3 Tree + fast weights** | T1 plus one $d\times d$ fast matrix | Surprise-gated writes, delta rule | T1 plus $O(d^2)$ | Self-correction, updates, rule learning |
| **T4 Brain hybrid** | 256-fact window, T3 as episodic memory, slow MLP | Writes, then replay during sleep | T3 plus the window | All eleven limitations |

T4 writes each fact to the window first; a fact that is still surprising when it leaves the window enters the tree, and sleep replays tree leaves into the slow weights.

## Worked example
Only T1 with supersession, T3 and T4 answered all three questions correctly in a 7,504-fact timeline; full attention blended the old and new language.

The timeline: the project language is Python and the port 8080; 3,000 unrelated facts pass; the language changes to Go; 3,000 more pass; the model is asked for language and port; the user corrects the route to `/healthz`; 1,500 more facts pass; the model is asked for the route. An answer counts as correct only if it decodes to the expected token with cosine above 0.9, so a blend of two stored values is a failure.

| Architecture | Language (Go) | Port (8080) | Route (`/healthz`) | Lookups per read |
| :--- | :--- | :--- | :--- | :--- |
| Standard, 2,048-token window | wrong | wrong | correct | 2,048 |
| Standard, full attention | blend of Python and Go | correct | correct | 6,003 to 7,504 |
| T1, append-only | blend of Python and Go | correct | correct | 59 to 153 |
| T1 with supersession | correct | correct | correct | 511 to 542 |
| T2 multi-timescale | wrong | wrong | wrong | 14 matrix reads |
| T3 tree + fast weights | correct | correct | correct | 33 to 60 |
| T4 brain hybrid | correct | correct | correct | 289 to 316, of which 256 are the window |

T3 stored 104 leaves for the 7,504 facts, because the 7,400 repeats were predictable and skipped. T1 with supersession paid about ten times more per read than T3 because its 7,400 retirements loosened node radii. After a sleep step, the slow weights alone, with the tree switched off, answered language, port and route correctly and recalled 100 of 100 stored facts.

## Prototype results
Certified routing matched full attention to $2\times 10^{-7}$ at 14x fewer lookups for $n=16,384$, but its cost still grows roughly as $n^{0.67}$, not $\log n$.

Certified cost grew as $n^{0.67}$ on hierarchical keys and $n^{0.87}$ on flat clusters. Statistical routing grew as $n^{0.08}$ (95x fewer lookups at 16,384), but its mean error was 5.6%, with the worst single query 3.4 times the size of the true output, because it can miss one aligned key inside an unopened node.

| Experiment | Setup | Result |
| :--- | :--- | :--- |
| **Recall by age** | $n=4,096$ stream, queries 1 to 4,000 facts back | Full attention and T1: 100% at every age (T1 about 340 of 4,096 lookups). T2: 62% at age 1, 38% at age 4, 0% from age 16 |
| **Surprise-gated writes** | 6,000 tokens drawn from 300 distinct facts | T1 stores 6,000 leaves; T3 stores 300. Recall 100% for both |
| **Rule learning** | 400 samples of $v=A\cdot k$, tested on unseen keys | Relative error falls from 0.86 (25 samples) to 0.20 (400) for T3; tree lookup stays at 0.97 |
| **Knowledge update** | 100 facts changed after 2,000 others | Latest value returned: full attention 0%, T1 append-only 0%, T1 with supersession 100%, T3 100% |
| **Catastrophic forgetting** | Slow MLP, 40 facts per task, $d=32$ | After task B, task A accuracy: naive fine-tuning 0%, replay from the tree 100%; task B 100% for both |
| **Consolidation** | 100 facts from the timeline | Slow weights alone recalled 100 of 100, but only after whitening the keys and 10,000 epochs |
| **Worst case** | Keys with no cluster structure, $n=4,096$ | 5,179 to 8,185 lookups, more than full attention's 4,096 |

Two findings change the plan. First, the savings depend on key geometry: with clustered, hierarchical keys the tree saves 7 to 14 times at exact accuracy, and with random keys it saves nothing. Second, a fixed budget of openings does not scale: on hierarchical sharp keys, 256 openings gave 0% error at $n=1,024$ but 12.5% at $n=16,384$.

## Limitations and how each is addressed
Six of the eleven limitations are addressed in the prototype, four partly, and one (multi-token generation) is not yet built.

| # | Limitation | Mechanism | Prototype evidence | Status |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Quadratic or linear cost | Tree read with $\epsilon$ and budget $B$ | 7 to 14 times fewer lookups at exact accuracy; 95 times with the statistical bound at 5.6% error | Partly: cost still grows as $n^{0.67}$ |
| 2 | Bounded context window | Unbounded tree, only top levels hot | Top 12 levels held about 2 MiB at $n=16,384$ against 8 MiB of full KV cache | Partly: cold-tier fetch not built |
| 3 | One token at a time | Draft heads sharing retrieved nodes | None | Not built |
| 4 | Inference and cache cost | Memory persists; only surprising facts are written | T3 stored 104 of 7,504 facts | Addressed in prototype |
| 5 | Frozen weights, knowledge cutoff | Immediate leaf writes, then sleep consolidation | Slow weights alone recalled 100 of 100 facts | Addressed in prototype |
| 6 | Ephemeral in-context learning | Persistent tree plus consolidation | Facts survive far beyond a 2,048-token window | Partly: persistence across sessions not tested |
| 7 | Catastrophic forgetting | Exact replay from the tree | Task A accuracy 0% naive, 100% with replay | Addressed in prototype, toy MLP |
| 8 | No self-correction | Surprise triggers supersession and fast-weight update | 100 of 100 changed facts returned the latest value | Addressed in prototype |
| 9 | Data drift | Supersession by surprise; timestamps | Same test as 8 | Partly: no drift monitor yet |
| 10 | Costly updating | Local writes, no backprop | Rule learning from 400 samples with a $d\times d$ matrix | Addressed in prototype, not timed |
| 11 | Not brain-like | Window, episodic tree, slow weights, sleep | Structure only | Addressed by design |

## Evaluation plan
The winner is the candidate with the best accuracy-and-cost curve across six measurements, scored as a weighted total and plotted as a Pareto front of accuracy against cost.

| Measurement | Metric | Scale |
| :--- | :--- | :--- |
| **Accuracy parity** | Relative error and KL divergence against full attention; perplexity; needle-in-haystack and multi-query associative recall | 1K to 1M tokens |
| **Cost** | Lookups per read, hot bytes, tokens per second with draft heads | Same sizes |
| **Knowledge update** | Share of changed facts returning the latest value, with unrelated traffic between | Python-to-Go test at scale |
| **Continual learning** | Backward transfer (old-task accuracy after new training) | Sequence of 10 tasks |
| **Drift** | Time to adapt when facts in a stream change | Synthetic and real streams |
| **Update cost** | Wall-clock and operations per write, against one backprop step | Per fact |

The prototype already reports the first four on small synthetic data; drift and update cost are not yet measured; the next step is to repeat them on a distilled language model. Pass thresholds are not set yet and should be agreed before the first large run.

## Open problems and roadmap
The central open problem is that tree cost still grows with $n$ unless keys are clustered and bounds are tight; the first three rows below attack it.

| Problem | Evidence | First experiment |
| :--- | :--- | :--- |
| **Tighter certified bounds** | Certified cost grows as $n^{0.67}$ because a node spanning several topics always has a vacuous radius bound | Bounding volumes aligned with a node's principal directions; re-measure the exponent |
| **Key geometry** | Random keys save nothing; hierarchical keys save 7 to 14 times | Train a key encoder with a clustering loss; distil attention from a pretrained small model with KL loss |
| **Statistical-bound misses** | 5.6% mean error, worst query 3.4 times the true output | Add a verification pass over the highest-scoring unopened nodes; learn the bound |
| **Multi-token generation** | Not built | Draft heads that share the nodes retrieved for the first token |
| **Consolidation at scale** | 100 facts needed whitened keys and 10,000 epochs | Low-rank adapters per domain, replay from the tree, 100K facts |
| **Supersession cost** | T1 with supersession used 511 to 542 lookups against 33 to 60 for T3, because retirements loosen radii | Periodic subtree rebuilds; keep surprise gating |
| **Online poisoning** | A model that writes at inference can be fed false facts | Trust score per write, checked before consolidation |
| **Hardware** | NumPy reference only | PyTorch port with a level-packed layout so a read is a few contiguous fetches |

Proposed order: port T1 to PyTorch and distil a small pretrained model into it; add T3's surprise loop; add T4's window and sleep stage. Runnable code and benchmarks will be added to this repository.

---

# Tree-Memory v2: Conversion, Training and Tested Fixes for the Open Problems (v2)
**11 Oct 2026**

## Summary
Once keys are trained to cluster, a certified tree read touches 9 to 22 times fewer slots than full attention at 32,768 facts with recall identical to full attention on the same model, but cost still grows about $n^{0.6}$, so $O(\log n)$ is not reached yet.

**Pretrained model:** it converts by keeping all weights, moving the KV cache into the tree, and fine-tuning the key projection with a clustering loss. Reads fell 5.6 times at identical accuracy when real cluster labels existed, and only 1.5 times when the labels came from k-means on the model's own keys.

**From scratch:** training with the same loss reached 9 times fewer reads at 64-dimensional keys and 22 times fewer at 8-dimensional keys (98% recall). Both routes, the dataset format and every fix for the v1 open problems are tested below.

Two changes beat everything else. First, v1's $n^{0.67}$ growth was mostly stale radii: recomputing them from live keys cut sharp-attention reads at 32,768 keys from 2,364 to 343 nodes. Second, the tree is cheap only when keys have low intrinsic dimension and tight topic clusters, which is a training outcome, not a data-structure property.

**Scope of the evidence:** everything runs on a synthetic knowledge-memory task (32,768 slots) in PyTorch and NumPy on 2 CPU cores. Model hubs were blocked in that workspace, so the "pretrained" model is a small teacher trained there. No real LLM and no GPU were tested in v2. (Later experiments, exp01–exp09, did use a real 0.5B model and chess worlds up to 1M addresses; see `experiments/`.)

## The v1 open problems, one by one
Four of the eight are solved, three are partly solved, and hardware is not solved.

| Problem | What was tested | Measured result | Status |
| :--- | :--- | :--- | :--- |
| **1. Tighter certified bounds** | Recompute radii from live keys (refresh); per-node subspace bound (rank 2 to 16); coordinate-block bound | Refresh cut sharp-attention reads at 32,768 keys from 2,364 to 343 nodes and the growth exponent from 0.82 to 0.21. The subspace bound reads 1.5 to 4.7 times fewer nodes, but each costs $1+r$ units, so work rises ($r = 4$: 1,002 units against 343). The block bound saved 0 to 26% of nodes | Solved by refresh; the other two bounds do not pay |
| **2. Key geometry** | Prototype loss, conversion and scratch training, key dimension 8, 16 and 64, k-means pseudo-labels | At 32,768 facts the teacher needs 32,744 node reads; conversion 5,848; scratch 3,656; 16-d 2,914; 8-d 1,506; pseudo-labels 22,424 | Partly: 9 to 22 times fewer, still about $n^{0.6}$ |
| **3. Statistical-bound misses** | Calibrate $\kappa$ from 2 to 8 on 150 queries, test on 150 others | $\kappa = 8$ was safe (99th-percentile error 0.000, worst 0.109) but cost 189 nodes against 196 for the certified read. Smaller $\kappa$ erred by up to 151% | Solved by removal: use certified reads on fresh radii |
| **4. Multi-token generation** | One grouped read certifies $m$ queries over a shared cover | Per token, nodes read fell from 1,857 to 269 for 8 queries in one topic, and from 1,861 to 425 for 8 random queries. No draft heads were built | Partly: needs queries that stay local |
| **5. Consolidation at scale** | 16 domain experts routed by tree centroids against one MLP of equal size (529K parameters), 16,384 facts | Recall 100% against 18% after 150 epochs. Experts for old domains were untouched while new ones trained, so nothing was forgotten. Routing was 100% right on learned clusters and 72.5% right on the unclustered teacher | Solved at 16K facts; 100K untested |
| **6. Supersession cost** | After 50% of 4,096 slots were rewritten: no maintenance, refresh each round, full rebuild | 478, 182 and 161 nodes per read. Most of the inflation came from incremental inserts, not from retirements | Solved: refresh after write bursts |
| **7. Online poisoning** | Simulation: 2,000 facts, four attacker types, three write policies | Newest-wins let 74 to 81% of attacked facts be poisoned (32% against the slow drip). Reputation plus corroboration stopped new accounts but lost to sleeper accounts (77%). Adding an overwrite budget and a quorum that scales with existing support stopped all four, at the price of 16 to 23% of community updates not showing within 30,000 steps | Partly: simulation only, attacker model is ours |
| **8. Hardware** | Byte-traffic model from measured node counts | Per head per token at 32,768 slots: full attention reads 4.7 MB (8-d keys, 64-d values, fp16), the tree 0.22 MB, about 21 times less. No GPU kernel or wall-clock was measured. | Not solved |

Problem 1 deserves one warning. Refresh is exact only on a static snapshot, so a live tree needs it after every write burst; problem 6 measures that policy.

## Route A: convert a pretrained model
A pretrained model converts by keeping every weight, moving old keys and values into the tree, and fine-tuning the key projection so keys cluster; in the test this cut node reads 5.6 times at identical accuracy, but only with real cluster labels.

1. **Copy and freeze a teacher.** Keep embeddings, MLPs and the output head. Only attention changes: for each layer and head, tokens that leave an exact local window (the last 2,048) are written to a tree as `(key, value)` leaves.
2. **Measure how tree-friendly the keys already are.** On the test teacher a certified read touched 32,744 of 32,768 slots. Key cosine was 0.70 within an entity, 0.43 within a topic and 0.02 across domains. If a model already prunes well, stop here.
3. **Get cluster labels.** Use real metadata: topic or domain tags, document and section IDs, entity types. Hierarchical k-means on the teacher's own keys had only 0.17 purity against the true domains and saved just 1.5 times.
4. **Fine-tune.** Train the student (a copy of the teacher) for 800 steps with AdamW at `3e-3` on the loss below, with $\lambda = 1$.
5. **Build and gate.** Bulk-build a balanced 2-means tree over the leaves, refresh radii, set $\epsilon$, and accept the model only if tree answers match full-attention answers.

```python
# L = KL(student || teacher) + lambda * L_proto
def proto_loss(net, K, labels, temp=16.0):
    Kn = F.normalize(K, dim=-1)
    l_domain = F.cross_entropy(temp * Kn @ F.normalize(net.P_domain, dim=-1).T, labels.domain)
    l_topic  = F.cross_entropy(temp * Kn @ F.normalize(net.P_topic,  dim=-1).T, labels.topic)
    return l_domain + l_topic  # pulls keys to their prototype AND pushes other prototypes away
```

A plain compactness loss (pull each key to its group centroid) failed: it collapsed every key into one cone, with cosine 0.84 even across domains, and the tree could not prune. The prototype loss's push-apart term is what works.

**Result.** The converted student read 1,076, 2,624 and 5,848 nodes at 2,048, 8,192 and 32,768 facts, with 100% recall matching full attention. Its key cosine rose to 0.94 within an entity, 0.79 within a topic and 0.35 within a domain.

**Not tested, and likely to bite on a real LLM:**
* **RoPE and other position-dependent scores.** The tree bounds assume a score that depends only on content, so the tree should serve pre-rotation keys or content-only heads, with a local window handling position.
* **Grouped-query attention and many layers:** one tree per key-value head and layer multiplies memory.
* **Labels at inference.** Here the encoder receives the taxonomy IDs; a real model must infer the topic from text, which will blur clusters.

## Route B: train from scratch, with the dataset format
Training from scratch with the same prototype loss reached 9 times fewer reads at 64-dimensional keys and 22 times fewer at 8-dimensional keys, with recall equal to full attention (100% and 98%) at 32,768 facts.

**Dataset.** Two files. `schema.json` holds the taxonomy that supplies cluster labels (16 domains, each with 16 topics, each with 16 entities, plus 8 attributes and 64 value tokens). `episodes.jsonl` holds one record per operation. Facts are written with fresh random values in every episode, so the model must read its memory and cannot memorise answers. A tiny sample ships with the code.

```json
{"episode": 0, "t": 0, "op": "write", "domain": "health", "topic": "health/t05", "entity": "health/t05/e07", "attr": "definition", "value": "val_41"}
{"episode": 0, "t": 1, "op": "write", "domain": "sports", "topic": "sports/t11", "entity": "sports/t11/e02", "attr": "owner", "value": "val_16"}
{"episode": 0, "t": 29, "op": "query", "entity": "chemistry/t03/e04", "attr": "definition", "expected": "val_29"}
```

```json
{"taxonomy": {"biology": {"t00": ["e00", "e01", "..."], "t01": ["..."]}, "chemistry": {"...": []}},
 "attributes": ["definition", "origin", "owner", "status", "size", "date", "rule", "exception"],
 "values": ["val_00", "...", "val_63"]}
```

The labels (`domain`, `topic`) feed the clustering loss during training only; inference needs none. For a real corpus the same fields come from document metadata: source, section, entity type, or the cluster of a frozen embedding model.

**Recipe.**
1. Each step samples an episode of $n$ facts, $n$ drawn from 512, 2,048, 8,192 or 32,768 with probabilities 0.5, 0.3, 0.15 and 0.05, plus 128 queries.
2. Reads use full softmax attention, so the model is trained exactly as a standard transformer layer is; the tree is attached afterwards.
3. $\text{Loss} = \text{cross-entropy on the answer token} + \lambda \text{ times the prototype loss from Route A}$, with $\lambda = 1$.
4. 1,200 steps of AdamW at `1e-2` with cosine decay, under two minutes on 2 CPU cores at this toy scale.
5. Initialise the query projection equal to the key projection. With separate random matrices the model stayed at chance for the 300 steps tried.
6. Build the tree, refresh radii, evaluate at 2,048, 8,192 and 32,768 facts.

**Key dimension is the main dial.** The model scored 100% with 16- and 64-dimensional keys and 98% with 8-dimensional keys at 32,768 facts, while reads fell as dimension fell. Stronger clustering ($\lambda = 3$, temperature 32) changed little: 1,368 against 1,506 nodes at 8 dimensions.

**Caveat.** The encoder is handed the taxonomy IDs, so learning to cluster by them is easy. A language model must infer the topic from text, so expect weaker clusters than shown here.

## Experiment results
At 32,768 facts, trained keys cut certified reads 5.6 to 22 times, while a teacher never trained to cluster reads every slot.

### Reads against memory size
Cluster labels matter more than the training route: conversion with k-means labels saved only 1.5 times, because the teacher's keys had almost no structure to find. Accuracy of the tree read equalled full attention in every run, since reads are certified.

### Is the growth caused by $n$ or by density?
The earlier curve conflates two things, because adding facts to a fixed taxonomy also packs every topic more densely. Adding whole domains keeps each topic at 128 facts and isolates the effect of $n$: reads grew far slower than $n$ up to 8 domains, then roughly doubled at 16. Whether that last jump is key-space crowding or a real limit of the method was the first open problem (see exp07–exp09 for the follow-up).

### Where v1's $n^{0.67}$ came from
Most of v1's growth was stale radii: a node's radius only ever grows as keys are inserted or retired, so reads kept opening nodes whose true bounds were small. Refresh recomputes radii from the live keys. The subspace bound does read fewer nodes (231 against 343 at 32,768 keys), but each costs 5 units, so total work is higher. In diffuse attention, where mass spreads over many topics, every bound scaled about linearly with $n$, so the certified approach does not help there.

### Multi-token reads
One grouped read shares the upper tree among its queries, so traffic per token falls most when queries stay in one topic. Random queries still gain 4.4 times from the shared top levels; the eight attributes of one entity gain only 1.5 times because each needs its own close neighbourhood certified. Every grouped answer was correct. The 8-d model showed the same pattern: 703 separate against 126 grouped nodes for one topic. These are certified verification reads; no draft heads were trained.

### Consolidation into slow weights
Domain experts consolidated 16,384 random facts with 100% recall in 33 seconds; one MLP of the same size reached 18% after 150 epochs. The tree's top-level clusters route queries to experts, so routing costs nothing extra, and an expert is never touched while other domains train.

| Slow-weight design (529K parameters, 16,384 facts) | Learned clusters | Unclustered teacher keys |
| :--- | :--- | :--- |
| One MLP, 150 epochs | 18% | 21% |
| 16 domain experts, 150 epochs | 100% | 73% (routing was 72.5% right) |
| Old-domain recall after training new domains: experts / replay into one MLP / naive fine-tuning | 100% / 33% / 3% | not run |

The single MLP was still under-trained at 150 epochs, so its forgetting numbers mostly show slow learning; the experts' advantage is that each one is a small problem that converges quickly.

### Write trust against poisoning
Share of attacked facts showing a false value after 30,000 simulated steps (2,000 facts, 400 attacked, honest sources confirming and correcting continuously):

| Attacker | Newest wins (v1) | Reputation plus two-source quorum | Plus overwrite budget and support-scaled quorum |
| :--- | :--- | :--- | :--- |
| One account | 74% | 0% | 0% |
| Five new accounts | 78% | 0% | 0% |
| Five sleeper accounts with built-up reputation | 81% | 77% | 0% |
| Sleepers, slow drip (2 overwrites per 1,000 steps each) | 32% | 12% | 0% |

The last policy lets a source contradict visible facts only 3 times per 1,000 steps, and requires half as many independent reputable sources as already support the current value. Its cost: 16 to 23% of legitimate community updates were still hidden at the end, with a median delay of about 560 steps. The owner's own writes apply immediately. These results hold only for the attackers simulated here; a patient attacker with many identities is not covered, which is why identity cost is on the open list.

## Which route to take
Try conversion first because it is cheap and tests whether your labels work; if reads do not fall at least 5 times, train from scratch with 16-dimensional keys, which gave full recall and 11 times fewer reads at 32,768 facts.

| Route | Needs | Cost at this toy scale | Main risk | Pick it when |
| :--- | :--- | :--- | :--- | :--- |
| **Convert, real labels** | A trained model plus topic or domain metadata | 800 fine-tuning steps, under a minute | Labels may not match what the keys encode, so gate on accuracy | A good checkpoint exists and metadata is available |
| **Convert, k-means labels** | A trained model only | Same | Keys with little structure give noise clusters (purity 0.17) | As a quick probe only |
| **Scratch, 64-d keys** | The dataset format above | 1,200 steps, under two minutes | Cost of a real pretraining run is unknown | The memory behaviour is the product |
| **Scratch, 16-d keys** | Same | Same | Less capacity per key | Best balance of recall and reads in this test |
| **Scratch, 8-d keys** | Same | Same | Recall fell from 100% to 98% at 32,768 facts | Traffic matters more than the last 2% of recall |

All costs are for a model with 64-dimensional embeddings. Real-scale training cost was not measured.

## What was still open after v2
Three gaps blocked the $O(\log n)$ claim at the time: cost still grew about $n^{0.6}$ on learned keys, no real LLM was tested, and no hardware timing existed.

| Open problem | Evidence so far | Next experiment |
| :--- | :--- | :--- |
| **1. Growth is about $n^{0.6}$, not $\log n$** | Learned keys: 855 to 3,656 nodes over a 16-fold larger memory. Three quarters of node openings land on nodes holding fewer than 64 slots, so cost is certifying the local neighbourhood, not walking the tree. With whole domains added at constant topic density, 8-d keys went from 417 to 744 nodes over 8 times more facts (about $n^{0.28}$), then doubled at 16 domains | Build a world with 4,096 or more topics at constant topic size; test whether cost is a local term plus $\log n$, and whether 16 domains simply crowd 8 dimensions |
| **2. No real LLM** | Model hubs were blocked in that workspace, so the teacher is a toy | Convert a small open checkpoint (100M to 1B parameters) using document metadata as labels |
| **3. Position-dependent scores** | Tree bounds assume the score depends on content only | Tree on pre-rotation keys for retrieval heads, plus a local window for position |
| **4. Labels at inference** | The toy encoder receives taxonomy IDs directly | Corrupt 5 to 30% of labels at input and measure reads and recall |
| **5. Hardware** | Bytes were modelled, not timed | Level-packed layout and a GPU kernel; measure milliseconds per token against full attention |
| **6. Poisoning** | Simulation with our own attacker model | Test adaptive attackers and an identity cost for new sources |
| **7. Consolidation beyond 16K facts** | Experts reached 100% at 16,384 facts; recall of unseen paraphrases untested | 100K facts and held-out paraphrases |
| **8. Draft heads** | Grouped reads share traffic only when queries stay local | Train draft heads and measure acceptance against traffic saved |

Proposed order: items 2 and 3 first, since they decide whether the toy results carry over, then item 1, which decides whether the $\log n$ claim survives.
