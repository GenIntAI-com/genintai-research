Approved A, B, C are now locked into the documentation baseline:

1. **A — Add true zero bucket code:** bucket regime becomes `{−1, 0, +1}`.
2. **B — Do not bucket transient scores:** attention scores/logits/intermediate accumulators may use FP32 or exact 2-decimal match values during computation.
3. **C — Update export expectation:** hybrid model should honestly target about **15–18 MB** at 5–10% exact/outlier weights, not 10–12 MB.

Below is the integrated documentation draft.

---

# Hybrid Quantized Semantic Transformer Research Plan  
## Final Documentation Draft v1

## 1. Purpose

This research plan extends the original fully ternary small language model design into a **hybrid-quantized semantic model**.

The original model stored weights only as:

```text
{-1, 0, +1}
```

The new model keeps the compression benefits of ternary-style storage, but adds an **exact/outlier regime** for weights that need stronger semantic meaning.

The goal is not just smaller storage. The goal is to test whether model weights and selected scores can carry a more interpretable **semantic match scale**:

```text
-9.99  ...  0  ...  +9.99
```

where:

- strong positive values mean strong semantic match,
- strong negative values mean strong semantic anti-match,
- zero means neutral / no strong evidence,
- bucket values retain ternary efficiency.

---

# 2. Baseline Model Specification

| Item | Decision |
|---|---|
| Model type | Small decoder-only Transformer |
| Target size | 20M–50M parameters |
| Primary target | ~47M parameters |
| Tokenizer | GPT-2 tokenizer, fixed token IDs |
| Language | English only |
| Training | From scratch |
| FP16/BF16 pretraining | Not used |
| Dataset target | 4B–6B raw tokens → 2.5B–4B cleaned tokens |
| Stored weights | Hybrid quantized |
| Stored embeddings | Hybrid quantized |
| Stored boundary activations | Ternary in baseline |
| Temporary accumulators | Allowed in FP32/integer/2-decimal exact form |
| Attention baseline | Softmax attention over match-scaled scores |
| Research attention variant | Raw signed match attention / gated match attention |
| Optimizer | Discrete regime-switch optimizer |

---

# 3. Core Hybrid Quantization Format

## 3.1 Stored weight regimes

Every stored weight belongs to one of two regimes:

```text
1. Bucket regime
2. Exact regime
```

## 3.2 Bucket regime

The bucket regime now has a true ternary state:

```text
{-1, 0, +1}
```

This fixes the earlier issue where all small values decoded to either `-1` or `+1`, eliminating true sparsity.

### Bucket meaning

| Bucket code | Decode value | Meaning |
|---|---:|---|
| `- bucket` | -1 | weak negative / simple negative state |
| `0 bucket` | 0 | neutral / inactive / sparse |
| `+ bucket` | +1 | weak positive / simple positive state |

This makes the bucket regime compatible with the original ternary model.

---

## 3.3 Exact regime

The exact regime stores stronger values:

```text
[-9.99, -1.00] ∪ [1.00, 9.99]
```

with two-decimal precision.

Examples:

| Stored value | Regime |
|---:|---|
| -9.99 | exact |
| -4.25 | exact |
| -1.00 | exact |
| 0 | bucket |
| +1 | bucket or exact +1.00 depending on regime flag |
| +3.72 | exact |
| +9.99 | exact |

Important note:

`bucket +1` and `exact +1.00` decode to the same numeric value, but they are **not the same training state**.

- `bucket +1` is cheap and ternary-like.
- `exact +1.00` can move to `1.01`, `1.02`, etc.

---

# 4. Semantic Match Scale

The semantic interpretation uses the following scale:

| Value range | Meaning |
|---|---|
| -9.99 to -1.00 | strong negative / anti-match |
| -1.00 to 0 | weak negative |
| 0 | neutral |
| 0 to +1.00 | weak positive |
| +1.00 to +9.99 | positive match |
| +9.99 | strongest match |

The balance point is:

```text
+1.00
```

Meaning:

```text
score >= 1.00 → meaningful positive match
score < 1.00 → weak, neutral, or negative
```

---

# 5. Important Rule: Bucketing Applies Only to Stored Tensors

This is now a locked decision.

## Correct rule

```text
Use bucket/exact quantization for stored tensors.
Do not forcibly bucket transient scores during computation.
```

Stored tensors include:

- model weights,
- embedding table,
- possibly activation checkpoints if later tested.

Transient tensors include:

- attention scores,
- raw dot products,
- logits before loss,
- temporary accumulators,
- intermediate FFN sums.

These may be computed in:

```text
FP32, integer accumulators, or exact 2-decimal match values
```

before final storage quantization.

## Reason

Bucketing temporary attention scores destroys useful ordering.

Example:

```text
raw scores = [0.8, 0.6, -0.3]
```

If kept exact:

```text
softmax ≈ [0.47, 0.38, 0.15]
```

If bucketed too early:

```text
[+1, +1, -1] → softmax ≈ [0.47, 0.47, 0.06]
```

The model loses the difference between `0.8` and `0.6`.

Therefore:

```text
Quantize storage, not every temporary number.
```

---

# 6. Squash Function for Match Scores

Raw dot products can become too large. If directly clipped to `±9.99`, many strong matches collapse to the same value.

Therefore use a squash function:

```text
M(x) = 9.99 × tanh(x / κ)
```

where:

```text
κ = scale measured during warmup
```

Possible choices for `κ`:

- standard deviation of raw attention scores,
- moving average of raw dot-product magnitude,
- per-layer calibration constant.

This gives a smooth match scale:

| Raw value | Match score |
|---:|---:|
| very negative | near -9.99 |
| small negative | weak negative |
| 0 | 0 |
| small positive | weak positive |
| very positive | near +9.99 |

---

# 7. Embedding Layer

## 7.1 Storage

Token embeddings are stored in the hybrid format:

```text
bucket {-1, 0, +1}
or exact value in [-9.99, -1.00] ∪ [1.00, 9.99]
```

## 7.2 Initialization

Use the original ternary-style sparse initialization:

```text
25% +1
50% 0
25% -1
```

or test:

```text
1/3 -1
1/3 0
1/3 +1
```

Recommended baseline:

```text
25% +1 / 50% 0 / 25% -1
```

because it preserves sparsity and is closer to the original ternary plan.

## 7.3 Semantic role

Embedding dot products can be interpreted as token similarity after squashing:

```text
similarity(token_a, token_b) = M(e_a · e_b)
```

Example target behavior:

| Pair | Expected score |
|---|---:|
| dog / puppy | high positive |
| Paris / London | medium positive |
| dog / airplane | negative or weak |
| true antonyms | strong negative if learned |

---

# 8. Attention Design

## 8.1 Baseline attention: softmax over match scores

This is the baseline and should be built first.

```text
q = x Wq
k = x Wk
v = x Wv

raw_score = q · k / sqrt(d_head)
match_score = M(raw_score)

attention_prob = softmax(match_score + causal_mask)
attention_out = attention_prob @ v
```

Future tokens receive a causal mask:

```text
future score = -∞ before softmax
```

or, on the semantic meter:

```text
future score ≈ -9.99 or masked before softmax
```

Recommended implementation:

```text
use standard causal softmax mask
```

because this is numerically safer.

---

## 8.2 Why softmax first?

Softmax is the lower-risk baseline.

It allows:

- stable comparison against normal Transformer attention,
- easier training,
- isolation of the hybrid-weight change,
- clear ablation later.

This follows the original research principle:

```text
change one major variable at a time
```

---

## 8.3 Research variant: Balance-Gate Match Attention

After baseline trains, test the stronger semantic attention design.

```text
raw = q · k
m = M(raw)

m[i][future] = masked
m[i][i] += 1.00   # optional self-match bonus

gate = m if m >= 1.00 else 0
attention = gate / sum(gate)
out = attention @ v
```

Meaning:

```text
Only tokens with match score >= balance point are attended.
```

This is more semantically interpretable but riskier than softmax.

Recommended status:

```text
Phase 2 experiment, not Phase 1 baseline.
```

---

# 9. FFN / MLP Design

## 9.1 Baseline FFN

Keep the original GLU-style FFN first:

```text
x = norm(h)

gate = x W_gate
up = x W_up

gate = activation_or_quantized_gate(gate)
up = activation_or_quantized_up(up)

glu = gate * up
down = glu W_down

h_next = h + residual_scale * down
```

Stored weights are hybrid quantized.

Boundary activations remain ternary in the first baseline.

---

## 9.2 Research variant: MatchGLU

Once the baseline is stable, test semantic FFN behavior:

```text
g = M(W_gate · x)
u = M(W_up · x)

p = (g × u) / 9.99
h_update = M(W_down · p)
```

Interpretation:

- `g` asks: does the input match this concept?
- `u` asks: what evidence is carried?
- `p` combines match and evidence.
- `h_update` writes signed evidence back into the residual stream.

Recommended status:

```text
Phase 2 or Phase 3 experiment.
```

---

# 10. Residual Stream and Activations

## 10.1 Baseline

To isolate the effect of hybrid weights, keep layer-boundary activations ternary:

```text
h = ternary_quantize(h + residual_scale * block_output)
```

This carries over from the original ternary plan.

Recommended residual scale:

```text
alpha = 1 / sqrt(2 × n_layers)
```

For 8 layers:

```text
alpha ≈ 0.25
```

## 10.2 Later experiment

After the hybrid-weight baseline is stable, test hybrid/match-scale activations:

```text
h = Q(M(h + alpha × block_output))
```

But do not start here.

Reason:

```text
Hybrid weights + hybrid activations + semantic attention all at once makes debugging too difficult.
```

---

# 11. Output Head

Use tied embeddings if possible:

```text
logits = h @ embedding_table.T
```

Then squash or scale:

```text
match_logits = M(logits / κ_out)
```

The output values can be interpreted as match scores between the current hidden state and vocabulary tokens.

---

# 12. Loss Function

## 12.1 Baseline loss

Use normal cross-entropy first:

```text
L_ce = cross_entropy(logits, target_token)
```

Add match calibration for the correct token:

```text
L_match = (9.99 - m_correct)^2
```

Final baseline:

```text
L = L_ce + λ L_match + z_loss
```

Recommended starting values:

```text
λ = 0.1
z_loss = 1e-4
```

Run ablation:

```text
λ ∈ {0, 0.1, 0.3}
```

---

## 12.2 Semantic soft-target variant

Later research variant:

```text
t_w = M(e_correct · e_w)
```

where:

- semantically similar wrong tokens get mild positive targets,
- unrelated tokens get negative or low targets.

Example:

If correct token is `"Paris"`:

| Token | Target behavior |
|---|---|
| Paris | near +9.99 |
| London | medium positive |
| France | medium/high positive |
| banana | negative or near zero |

Loss variant:

```text
L_soft_semantic = mean_topK((m_w - t_w)^2)
```

Recommended status:

```text
Phase 2 experiment.
```

---

# 13. Optimizer: Regime-Switch Hybrid Optimizer

The optimizer combines:

1. original ternary state optimizer for bucket weights,
2. promotion/demotion logic for exact weights,
3. small exact-regime movement.

---

## 13.1 Bucket regime update

For bucket weights:

```text
S ∈ {-1, 0, +1}
```

Use the original ternary optimizer:

```text
direction = -sign(momentum)
```

Allowed moves:

| Current state | Allowed moves |
|---|---|
| -1 | 0 |
| 0 | -1 or +1 |
| +1 | 0 |

Only top-k momentum weights are updated, controlled by:

```text
update_ratio
```

Recommended starting value:

```text
update_ratio = 0.001
```

---

## 13.2 Promotion: bucket → exact

A bucket weight may promote into the exact regime when:

```text
|momentum| > entry_threshold
```

and the layer has not exceeded its exact/outlier budget.

Promotion examples:

| Bucket state | Momentum direction | New exact state |
|---|---|---|
| +1 | wants stronger positive | +1.00 exact |
| -1 | wants stronger negative | -1.00 exact |
| 0 | strong positive signal | +1.00 exact |
| 0 | strong negative signal | -1.00 exact |

Recommended entry threshold:

```text
entry_threshold = p95(|momentum|) per layer
```

---

## 13.3 Exact weight update

For exact weights:

```text
V_new = clamp(V - lr_exact × sign(momentum), -9.99, +9.99)
```

Then round to two decimals.

Recommended sweep:

```text
lr_exact ∈ {0.01, 0.05, 0.1}
```

Do not use a step smaller than `0.01`, because two-decimal storage would erase it.

---

## 13.4 Demotion: exact → bucket

Exact weights demote automatically when:

```text
|V| < 1.00
```

Demotion rule:

```text
if V > 0 → bucket +1
if V < 0 → bucket -1
```

Then the normal ternary optimizer can later move:

```text
+1 → 0
-1 → 0
```

This avoids abrupt exact-to-zero collapse.

---

## 13.5 Hysteresis and refractory period

To avoid oscillation:

```text
promotion is hard
demotion is automatic but followed by cooldown
```

Recommended:

```text
refractory period = 100 optimizer steps
```

After a weight changes regime, it cannot change regime again for 100 steps.

---

## 13.6 Exact/outlier budget

Set a hard per-layer budget:

```text
f_max = 10%
```

Optional sweep:

```text
f_max ∈ {5%, 10%, 15%}
```

If a layer already has too many exact weights:

```text
block further promotion
```

This keeps export size controlled.

---

# 14. Memory and Export Size

## 14.1 Important conclusion

Hybrid quantization does **not** beat pure ternary storage size.

At best, when exact fraction is zero:

```text
hybrid ≈ ternary size
```

The reason to use hybrid quantization is not smaller size.

The reason is:

```text
more expressive weights + semantic strong-match values
```

---

## 14.2 Practical bit estimate

Bucket regime needs approximately:

```text
2 bits
```

to encode:

```text
{-1, 0, +1}
```

Exact regime needs extra payload for two-decimal signed values.

A practical estimate:

```text
bits_per_weight ≈ 2 + 10f to 2 + 11f
```

where:

```text
f = fraction of exact weights
```

For safety, use the conservative estimate:

```text
bits_per_weight ≈ 2 + 11f
```

---

## 14.3 Export table for ~46.7M parameters

| Exact fraction `f` | Bits/weight approx | Export size approx |
|---:|---:|---:|
| 0% | 2.00 | 11.7 MB |
| 5% | 2.55 | 14.9 MB |
| 10% | 3.10 | 18.1 MB |
| 15% | 3.65 | 21.3 MB |

Recommended documentation claim:

```text
Expected export size: about 15–18 MB for f = 5–10%.
```

Do not claim 10–12 MB unless `f ≈ 0`.

---

# 15. Dataset Plan

Use the same dataset plan as the original ternary documentation.

## 15.1 Target amount

```text
Raw tokens: 4B–6B
Cleaned tokens: 2.5B–4B
```

## 15.2 Dataset categories

Use mixed English data:

- web text,
- books,
- Wikipedia-style reference text,
- StackExchange-style Q&A,
- math text,
- code-adjacent reasoning text where appropriate,
- news,
- factual documents,
- high-quality curated subsets.

## 15.3 Validation split

Recommended:

```text
train: 98%
validation: 2%
```

For 3B cleaned tokens:

```text
validation ≈ 60M tokens
```

If too large:

```text
validation = 10M–30M tokens
```

But each major domain should have at least:

```text
1M validation tokens
```

---

# 16. Data Filtering

Carry over the filtering strategy from the dataset document.

Use:

- language detection,
- deduplication,
- quality scoring,
- contamination checks,
- category balancing,
- removal of low-quality boilerplate,
- removal/downweighting of poor documents.

For quality scoring:

```text
overall_score >= 4 → gold/high-quality
overall_score >= 3 → acceptable general training
overall_score < 3 → discard or downweight
```

Gold evaluation subset:

```text
overall_score >= 4
manual verification of 500–1000 examples
```

---

# 17. Training Phases

Do not start directly with the 47M model.

## Phase 1: Tiny prototype

| Item | Setting |
|---|---|
| Parameters | 1M–5M |
| Layers | 2 |
| d_model | 128 or 256 |
| Heads | 4 |
| Sequence length | 256 or 512 |
| Data | 10M–50M tokens |

Success criteria:

- validation loss decreases,
- bucket/exact ratio remains healthy,
- no collapse to all exact,
- no collapse to all zero,
- activations do not die,
- training does not oscillate.

---

## Phase 2: Small scaling test

| Item | Setting |
|---|---|
| Parameters | 10M–20M |
| Layers | 4–6 |
| d_model | 384 or 512 |
| Sequence length | 512 or 1024 |
| Data | 100M–300M tokens |

Success criteria:

- stable loss curve,
- domain validation improves,
- exact fraction remains within budget,
- regime-switch rate stabilizes,
- generated samples become coherent.

---

## Phase 3: Main model

| Item | Setting |
|---|---|
| Parameters | ~47M |
| Layers | around 8 |
| Sequence length | 1024 main, 2048 later |
| Data | 2.5B–4B cleaned tokens |
| Tokenizer | GPT-2 fixed tokenizer |

Success criteria:

- validation perplexity competitive with original ternary baseline,
- stable bucket/exact distribution,
- no layer exact-budget explosion,
- semantic match calibration improves,
- generation is fluent enough for human evaluation.

---

# 18. Validation Metrics

Use all original validation metrics plus hybrid-specific metrics.

## 18.1 Standard metrics

- validation loss,
- perplexity,
- zero-shot benchmarks,
- factual QA,
- human evaluation,
- repetition rate,
- safety checks.

## 18.2 Statistical validation

Use:

- Wilson confidence interval for accuracy,
- bootstrap confidence interval for perplexity difference,
- McNemar test for paired QA comparisons.

---

## 18.3 Hybrid-specific metrics

Track:

```text
exact fraction f per layer
bucket distribution {-1, 0, +1}
regime-switch rate
promotion rate
demotion rate
refractory-blocked switches
mean exact magnitude
percentage at ±9.99 saturation
match-score calibration
```

Important warning metric:

```text
too many weights at ±9.99
```

This may mean the squash scale `κ` is wrong or training is saturating.

---

# 19. Match-Score Calibration Evaluation

Create a small evaluation set of token pairs.

Examples:

| Pair type | Expected score |
|---|---|
| synonym / near synonym | high positive |
| related words | medium positive |
| unrelated words | near zero or negative |
| antonyms | negative |
| same token | very high positive |

Example pairs:

```text
dog / puppy
cat / kitten
Paris / France
Paris / London
dog / airplane
hot / cold
true / false
```

Measure:

```text
M(e_a · e_b)
```

Check whether the semantic scale is actually learned.

This is important because the semantic meaning is not guaranteed by storage alone.

The loss and training dynamics must create it.

---

# 20. Inference

## 20.1 Stored model

Inference model stores:

```text
hybrid quantized weights
hybrid quantized embeddings
ternary boundary activations in baseline
```

## 20.2 Computation

Inference may use:

- integer accumulation for bucket weights,
- lookup/decode for exact weights,
- FP32 or fixed-point temporary accumulators,
- normal softmax for output sampling.

## 20.3 Tie-breaking

Because many bucket values can be identical, inference needs deterministic tie-breaking.

Recommended order:

1. use raw accumulator value before final quantization if available,
2. otherwise use exact value if one candidate is exact,
3. otherwise use stable token ID order or deterministic index order.

---

# 21. High-Level Forward Pass

Baseline forward pass:

```python
def forward(token_ids):
    h = hybrid_embedding_lookup(token_ids)

    for layer in layers:
        # Attention block
        x = fixed_rmsnorm(h)
        x = ternary_quantize(x)

        q = hybrid_linear(x, W_q[layer])
        k = hybrid_linear(x, W_k[layer])
        v = hybrid_linear(x, W_v[layer])

        raw_scores = q @ k.transpose(-1, -2)
        raw_scores = raw_scores / sqrt(d_head)

        match_scores = squash(raw_scores)  # M(x)

        match_scores = apply_causal_mask(match_scores)

        probs = softmax(match_scores)
        attn = probs @ v

        attn = ternary_quantize(attn)
        h = ternary_quantize(h + residual_scale * attn)

        # FFN block
        x = fixed_rmsnorm(h)
        x = ternary_quantize(x)

        gate = hybrid_linear(x, W_gate[layer])
        up = hybrid_linear(x, W_up[layer])

        gate = ternary_quantize(gate)
        up = ternary_quantize(up)

        glu = gate * up
        glu = ternary_quantize(glu)

        down = hybrid_linear(glu, W_down[layer])
        down = ternary_quantize(down)

        h = ternary_quantize(h + residual_scale * down)

    h = fixed_rmsnorm(h)
    h = ternary_quantize(h)

    logits = h @ embedding_table.T
    return logits
```

Research variant later replaces attention and FFN with match-gated versions.

---

# 22. Main Ablation Experiments

Carry over original ternary ablations and add hybrid-specific ones.

## 22.1 Original ablations to keep

1. Optimizer comparison  
2. Quantizer comparison  
3. Embedding initialization  
4. Activation ternarization  
5. Update ratio  
6. Residual scale  

---

## 22.2 New hybrid ablations

### Ablation 7: Attention rule

Compare:

```text
A: softmax over match scores
B: balance-gated match attention
C: raw signed match attention
```

Recommended order:

```text
A first, B second, C last
```

---

### Ablation 8: Bucket initialization

Compare:

```text
25% +1 / 50% 0 / 25% -1
1/3 +1 / 1/3 0 / 1/3 -1
mostly zero sparse init
```

---

### Ablation 9: Exact fraction budget

Compare:

```text
f_max ∈ {5%, 10%, 15%}
```

---

### Ablation 10: Match calibration loss

Compare:

```text
λ = 0
λ = 0.1
λ = 0.3
```

---

### Ablation 11: Exact learning rate

Compare:

```text
lr_exact ∈ {0.01, 0.05, 0.1}
```

---

### Ablation 12: Squash scale

Compare:

```text
κ = global warmup std
κ = per-layer warmup std
κ = moving average
```

---

### Ablation 13: Transient score quantization

Recommended baseline:

```text
no bucketing of transient scores
```

But for proof, compare:

```text
exact/FP32 transient scores
vs.
bucketed transient scores
```

Expected result:

```text
bucketed transient scores perform worse
```

---

# 23. Risks and Mitigations

## Risk 1: Exact weights grow too much

Problem:

```text
model becomes too large
```

Mitigation:

```text
f_max per layer
promotion blocking
track exact fraction
```

---

## Risk 2: Regime oscillation

Problem:

```text
weights bounce bucket ↔ exact repeatedly
```

Mitigation:

```text
hysteresis
p95 promotion threshold
automatic demotion
100-step refractory period
```

---

## Risk 3: Saturation at ±9.99

Problem:

```text
many scores lose ranking
```

Mitigation:

```text
squash function M(x)
κ warmup calibration
track saturation percentage
```

---

## Risk 4: Semantic scale not learned

Problem:

```text
9.99 does not actually mean top match
```

Mitigation:

```text
match calibration loss
embedding-pair evaluation
semantic soft-target experiment
```

---

## Risk 5: Hybrid system too complex

Problem:

```text
too many changes at once
```

Mitigation:

```text
Phase 1 uses softmax attention
ternary boundary activations
standard FFN
only stored weights become hybrid
```

Then add semantic variants later.

---

# 24. Recommended Locked Baseline

This is the recommended baseline for first implementation.

```text
Model:
  decoder-only Transformer, ~1M–5M first

Tokenizer:
  GPT-2 fixed tokenizer

Weights:
  hybrid quantized

Bucket:
  {-1, 0, +1}

Exact:
  [-9.99, -1.00] ∪ [1.00, 9.99]
  two-decimal precision

Activations:
  ternary at layer boundaries

Attention:
  softmax over squashed match scores

FFN:
  standard GLU/MLP with hybrid stored weights

Optimizer:
  bucket weights use original ternary optimizer
  exact weights use sign-momentum step
  promotion threshold = p95(|momentum|)
  demotion when |V| < 1.00
  refractory = 100 steps

Exact budget:
  f_max = 10%

Loss:
  cross-entropy + λ match calibration
  λ = 0.1

Export expectation:
  about 15–18 MB for 47M parameters at 5–10% exact weights
```

---

# 25. Final Recommendation

The new hybrid plan has stronger logic than the earlier version because it now has:

1. true zero state,
2. honest memory math,
3. no destructive bucketing of temporary scores,
4. preserved ternary optimizer behavior for bucket weights,
5. controlled exact/outlier promotion,
6. clear semantic calibration path,
7. staged research progression.

The best implementation path is:

```text
Step 1: Reproduce original ternary tiny model.
Step 2: Replace stored weights with hybrid format only.
Step 3: Keep softmax attention and ternary boundary activations.
Step 4: Add match-score squash and calibration metrics.
Step 5: Train tiny model until stable.
Step 6: Scale to 10M–20M model.
Step 7: Only then test Balance-Gate Attention / MatchGLU / semantic soft targets.
```

This keeps the project scientifically clean and prevents too many experimental changes from being mixed together at once.
