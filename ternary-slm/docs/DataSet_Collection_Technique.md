
# 4. Dataset Collection: Required Size and Categories

You approved the category table. Now we add required sizes, counts, and open sources.

---

## 4.1 Token size math

For English text:

```text
1 token ≈ 4 characters
```

So:

```text
1B tokens ≈ 4 GB raw UTF-8 text
```

With JSONL metadata overhead:

```text
1B tokens ≈ 6–10 GB JSONL
```

Tokenized with GPT-2 tokenizer and stored as `uint16`:

```text
1B tokens ≈ 2 GB binary
```

Therefore:

```text
3B cleaned tokens ≈ 6 GB tokenized binary
4B cleaned tokens ≈ 8 GB tokenized binary
```

If you collect 4B raw tokens and cleaning yield is 60–75%:

```text
4B raw → 2.4B–3.0B cleaned tokens
```

If you want 3B–4B cleaned tokens:

```text
collect 5B–6B raw tokens
```

Recommended:

```text
Collect target: 5B raw tokens
Expected cleaned target: 3B tokens
Minimum acceptable: 2.5B cleaned tokens
Strong target: 4B cleaned tokens
```

---

## 4.2 Recommended cleaned token mix

Target cleaned corpus:

```text
Total: 3B tokens
```

| Category | Cleaned tokens | Percentage |
|---|---:|---:|
| High-quality web / educational web | 900M | 30% |
| Wikipedia / encyclopedic | 900M | 30% |
| Books | 300M | 10% |
| StackExchange / Q&A | 300M | 10% |
| Educational textbooks | 150M | 5% |
| Code | 200M | 6.7% |
| Math | 150M | 5% |
| News | 100M | 3.3% |
| **Total** | **3B** | **100%** |

If you want 4B cleaned tokens, scale proportionally:

| Category | Cleaned tokens |
|---|---:|
| High-quality web / educational web | 1.2B |
| Wikipedia / encyclopedic | 1.2B |
| Books | 400M |
| StackExchange / Q&A | 400M |
| Educational textbooks | 200M |
| Code | 267M |
| Math | 200M |
| News | 133M |
| **Total** | **4B** |

---

## 4.3 Raw collection target assuming cleaning loss

Different sources have different cleaning yield.

| Category | Expected yield | Cleaned target | Raw tokens to collect |
|---|---:|---:|---:|
| Wikipedia | 80% | 900M | 1.13B |
| Educational web | 50% | 900M | 1.8B |
| Books | 80% | 300M | 375M |
| StackExchange | 50% | 300M | 600M |
| Educational textbooks | 70% | 150M | 215M |
| Code | 60% | 200M | 333M |
| Math | 50% | 150M | 300M |
| News | 60% | 100M | 167M |
| **Total** |  | **3B** | **~4.9B** |

So:

```text
Collect about 5B raw tokens.
After cleaning, expect about 3B cleaned tokens.
```

If storage is limited:

```text
Collect 4B raw tokens.
Expect 2.4B–3B cleaned tokens.
```

---

# 5. Open Dataset Sources

You asked for online open/free sources. Below are practical open sources. You should still verify license and terms before redistribution.

---

## 5.1 Wikipedia / encyclopedic

### Source 1: Wikimedia Wikipedia dumps

Link:

```text
https://dumps.wikimedia.org/enwiki/latest/
```

Useful file:

```text
enwiki-latest-pages-articles.xml.bz2
```

Hugging Face mirror:

```text
https://huggingface.co/datasets/wikimedia/wikipedia
```

Target:

```text
Cleaned tokens: 900M
Raw collect: 1.1B–1.5B tokens
Count: use full English Wikipedia articles, around 6M+ articles
```

Notes:

```text
Remove redirect pages.
Remove disambiguation pages.
Remove stubs if desired.
Keep main namespace articles.
```

---

### Source 2: Wikibooks

Link:

```text
https://dumps.wikimedia.org/enwikibooks/latest/
```

Target:

```text
Cleaned tokens: 50M–100M
Count: thousands of book pages
```

Useful for educational explanations.

---

### Source 3: Wikiversity

Link:

```text
https://dumps.wikimedia.org/enwikiversity/latest/
```

Target:

```text
Cleaned tokens: 20M–50M
```

Useful for course-like material.

---

## 5.2 High-quality web / educational web

### Source 4: FineWeb-Edu

Link:

```text
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
```

This is one of the best open educational web corpora.

Target:

```text
Cleaned tokens: 900M
Raw collect: 1.8B–2B tokens
Count: 2M–5M documents depending on document length
```

Recommended approach:

```text
Use streaming.
Sample enough shards to reach 2B raw tokens.
```

---

### Source 5: OpenWebText

Link:

```text
https://huggingface.co/datasets/openwebtext
```

Target:

```text
Cleaned tokens: 200M–400M
```

Useful as general web text.

---

### Source 6: C4 English

Link:

```text
https://huggingface.co/datasets/allenai/c4
```

Target:

```text
Cleaned tokens: 200M–400M
```

Use as backup web text. FineWeb-Edu is better for educational quality.

---

### Source 7: RefinedWeb sample

Link:

```text
https://huggingface.co/datasets/tiiuae/falcon-refinedweb
```

Target:

```text
Cleaned tokens: 200M optional
```

Large dataset. Use only sample if needed.

---

## 5.3 Books

For books, use public-domain sources.

### Source 8: Project Gutenberg

Link:

```text
https://www.gutenberg.org/
```

Bulk/mirror links:

```text
https://www.gutenberg.org/robot/harvest
https://aleph.gutenberg.org/
```

Hugging Face book corpus:

```text
https://huggingface.co/datasets/pg19
```

Target:

```text
Cleaned tokens: 300M
Raw collect: 375M
```

Count:

```text
Average public-domain book: 80k–120k tokens
300M tokens / 100k tokens per book ≈ 3,000 books
```

Recommended category-wise count:

```text
100 books per category × 30 categories = 3,000 books
```

Example book categories:

```text
fiction
nonfiction
science
history
philosophy
biography
travel
essays
poetry
drama
children
classics
economics
politics
psychology
education
mathematics
engineering
medicine historical
law historical
religion historical
mythology
folklore
letters
diaries
speeches
journalism
nature
adventure
mystery
```

If you want a smaller pilot:

```text
100 books per category × 5 categories = 500 books
Expected tokens: ~50M
```

But for 3B cleaned token target, books should be around 300M tokens, so full target is about 3,000 books.

---

### Source 9: Standard Ebooks

Link:

```text
https://standardebooks.org/
```

Target:

```text
Cleaned tokens: 30M–80M
Count: several hundred high-quality public-domain books
```

Standard Ebooks is cleaner than raw Project Gutenberg text.

---

## 5.4 StackExchange / Q&A

### Source 10: StackExchange dump

Archive link:

```text
https://archive.org/details/stackexchange
```

Hugging Face mirror:

```text
https://huggingface.co/datasets/stackexchange/stackexchange-dump
```

Target:

```text
Cleaned tokens: 300M
Raw collect: 600M
Count: 3M–5M high-quality posts or post threads
```

Use English sites only.

Recommended sites:

```text
stackoverflow
math
physics
biology
chemistry
history
philosophy
english
ell
writing
literature
computer science
software engineering
stats
machine learning
economics
law
medicine
psychology
skeptics
politics
geography
travel
```

Filtering rule:

```text
Keep questions with accepted answer.
Keep answers with score >= 1.
Remove comments.
Remove code-only answers unless code category.
Convert thread into clean text.
```

Example format:

```text
Question:
...

Accepted answer:
...
```

---

## 5.5 Educational textbooks

### Source 11: OpenStax

Link:

```text
https://openstax.org/subjects
```

Target:

```text
Cleaned tokens: 100M–150M
Count: 50–100 textbooks
```

Categories:

```text
math
physics
biology
chemistry
economics
psychology
sociology
history
business
statistics
computer science
```

OpenStax books are free, but check license and attribution requirements.

---

### Source 12: Wikibooks educational subset

Link:

```text
https://dumps.wikimedia.org/enwikibooks/latest/
```

Target:

```text
Cleaned tokens: 50M
```

Use high-quality completed books only.

---

## 5.6 Code

Code helps reasoning and structure, even for English general models.

### Source 13: The Stack v2

Link:

```text
https://huggingface.co/datasets/bigcode/the-stack-v2
```

Target:

```text
Cleaned tokens: 200M
Count: 100k–200k files
Average file: 1k–3k tokens
```

Important:

```text
Use only permissive-license subset if possible.
Filter by language:
    Python
    JavaScript
    TypeScript
    Java
    C
    C++
    Rust
    Go
```

For a small English general model, code should not dominate. Keep it around 5–7%.

---

### Source 14: StarCoder data subset

Link:

```text
https://huggingface.co/datasets/bigcode/starcoderdata
```

Use only if The Stack v2 is inconvenient.

---

## 5.7 Math

### Source 15: OpenWebMath

Link:

```text
https://huggingface.co/datasets/open-web-math/open-web-math
```

Target:

```text
Cleaned tokens: 100M
```

Good for mathematical text.

---

### Source 16: Proof-pile 2

Link:

```text
https://huggingface.co/datasets/EleutherAI/proof-pile-2
```

Target:

```text
Cleaned tokens: 50M–100M
```

Useful for formal/mathematical reasoning.

---

## 5.8 News

### Source 17: NewsCrawl English

Link:

```text
https://data.statmt.org/news-crawl/en/
```

Target:

```text
Cleaned tokens: 100M
Count: ~1M news articles
```

Useful for natural factual style.

---

### Source 18: CC-News

Hugging Face mirror:

```text
https://huggingface.co/datasets/cc_news
```

Target:

```text
Cleaned tokens: 50M–100M
```

Check license and availability.

---

# 6. Dataset Count Summary

For a 3B cleaned token corpus:

| Category | Cleaned tokens | Raw collect | Approx count |
|---|---:|---:|---:|
| Wikipedia | 900M | 1.1B | full English Wikipedia, ~6M articles |
| FineWeb-Edu / educational web | 900M | 1.8B | 2M–5M documents |
| Project Gutenberg / Standard Ebooks | 300M | 375M | ~3,000 books |
| StackExchange | 300M | 600M | 3M–5M posts/threads |
| OpenStax / Wikibooks | 150M | 215M | 100–300 textbooks/book-like units |
| Code | 200M | 333M | 100k–200k files |
| Math | 150M | 300M | 1M–2M math chunks |
| News | 100M | 167M | ~1M articles |
| **Total** | **3B** | **~4.9B** |  |

---

# 7. Small System Prompt for Dataset Source LLM

You mentioned you will use a separate chat session for collecting resources. Below is a small system prompt for an LLM whose job is to output open dataset sources and links.

Use this as the system prompt:

```text
You are an open-dataset sourcing agent for training a small English-only language model.

Your task:
Find free, open, publicly downloadable English text datasets for the requested category.

Rules:
1. Only include free/open datasets.
2. Do not include paid datasets, gated paid resources, or subscription data.
3. Prefer official dumps, public-domain sources, Hugging Face datasets, Common Crawl derivatives, Wikimedia dumps, StackExchange dumps, Project Gutenberg, Standard Ebooks, OpenStax, and academic open corpora.
4. Dataset language must be English or easily filterable to English.
5. Do not invent URLs. If unsure, mark the link as "unverified".
6. Include license information if known.
7. Include estimated size in tokens or GB if possible.
8. Include download command if possible.
9. Output only valid JSON.

Output JSON schema:
{
  "category": "string",
  "target_cleaned_tokens": "string",
  "sources": [
    {
      "name": "string",
      "url": "string",
      "download_command": "string",
      "license": "string",
      "language": "en",
      "estimated_raw_size_gb": number,
      "estimated_tokens": "string",
      "recommended_count": "string",
      "notes": "string",
      "verified": true
    }
  ]
}
```

Example user prompt for that session:

```text
Find open English dataset sources for the category "books".
Target cleaned tokens: 300M.
Prefer public-domain books.
Recommend around 3000 books, or 100 books per category across 30 categories.
Output JSON only.
```

Another example:

```text
Find open English dataset sources for the category "math".
Target cleaned tokens: 150M.
Prefer Hugging Face datasets and open academic corpora.
Output JSON only.
```

---

# 8. Gold Data Extraction Using DeepSeek-V4/Strong LLM api

You plan to use a local Gemma-3-1B model for dataset quality checking. This is good.

Use Gemma-3-1B for **sample-based quality auditing**, not for scoring every document. Full-corpus LLM scoring is too slow for billions of tokens.

---

## 8.1 Sampling strategy

For each source/category, sample documents for Gemma review.

Statistical rule:

To estimate a keep-rate with ±1% margin at 95% confidence, you need about:

```text
n ≈ 9,600 samples
```

For practical dataset auditing:

```text
Small source: sample 500 documents
Medium source: sample 1,000 documents
Large source: sample 2,000–5,000 documents
```

For the full corpus:

```text
Audit 10,000–50,000 sampled documents total.
```

Use stratified sampling:

```text
sample proportionally from:
    Wikipedia
    FineWeb-Edu
    books
    StackExchange
    code
    math
    news
```

---

## 8.2 Gemma4/DeepSeek-V4 quality prompt

Use a prompt like this:

```text
You are a data-quality auditor for pretraining a small English language model.

Evaluate the following text for language-model training quality.

Criteria:
1. English language quality
2. Coherence
3. Educational or informational value
4. Factual density
5. Safety
6. Presence of boilerplate, ads, navigation, or junk

Output only JSON with this schema:
{
  "english_quality": 0-5,
  "coherence": 0-5,
  "educational_value": 0-5,
  "factual_density": 0-5,
  "safety": 0-5,
  "overall_score": 0-5,
  "keep": true,
  "reason": "short reason"
}

Text:
"""
{document_text}
"""
```

Keep rule:

```text
overall_score >= 4 → gold/high-quality
overall_score >= 3 → acceptable general training
overall_score < 3 → discard or downweight
```

For gold training subset:

```text
Use documents with overall_score >= 4.
```

For gold evaluation subset:

```text
Use documents with overall_score >= 4
and manually verify at least 500–1000 examples.
```

---

## 8.3 Use Gemma4/DeepSeek-V4 labels to build a faster filter

Optional advanced step:

```text
1. Sample 20k documents.
2. Score them with Gemma4/DeepSeek-V4.
3. Train a small fastText or logistic-regression quality classifier.
4. Use that classifier on the full corpus.
```

This gives you LLM-guided quality filtering at large scale.

---

# 9. Updated Data Cleaning Pipeline

Use the same pipeline, but with direct ternary training in mind. Data quality is even more important because ternary models have less capacity to absorb noisy data.

Pipeline:

```text
raw JSONL
  ↓
text extraction
  ↓
language detection: English only
  ↓
heuristic filtering
  ↓
PII removal
  ↓
exact deduplication
  ↓
near deduplication with MinHash LSH
  ↓
quality scoring
  ↓
Gemma4/DeepSeek LLM stratified audit
  ↓
source weighting
  ↓
tokenization
  ↓
packing
  ↓
train/val split
```

Important validation split rule:

```text
Validation data must be deduplicated against training data.
Use 13-gram or 20-gram contamination check.
```

---

# 10. Tokenization

You approved using an existing tokenizer:

```text
GPT-2 tokenizer
```

Use:

```text
vocab_size = 50,257
```

Special tokens:

```text
<|endoftext|>
```

In Hugging Face:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
eos = tokenizer.eos_token_id
```

Tokenization rule:

```text
For each document:
    tokens = tokenizer.encode(text)
    tokens.append(eos)
    append to global token stream
```

Then pack into fixed sequences:

```text
input  = tokens[i : i + seq_len]
target = tokens[i + 1 : i + seq_len + 1]
```

Storage:

```text
train.bin
val.bin
```

Use `uint16` because GPT-2 vocab is below 65,536.

---




Use this exact recipe for your first direct ternary-from-scratch model.

```text
Model:
    Ternary-47M
    8 layers
    d_model = 512
    heads = 8
    intermediate = 1024
    GPT-2 tokenizer
    weight tying

Precision:
    forward weights: ternary {-1, 0, +1}
    embeddings: INT8 fake-quant
    activations: INT8 fake-quant
    loss: FP32
    master weights: FP32 latent parameters

Data:
    collect 4B–5B raw tokens
    target 3B cleaned tokens
    mix:
        30% Wikipedia
        30% FineWeb-Edu / educational web
        10% books
        10% StackExchange
        5% educational textbooks
        6.7% code
        5% math
        3.3% news

Cleaning:
    English filter
    heuristic filter
    PII removal
    exact dedup
    MinHash near dedup
    quality score
    Gemma-3-1B sample audit

Tokenization:
    GPT-2 BPE
    EOS after each document
    packed sequences

Training:
    optimizer: AdamW
    peak LR: 1e-4
    min LR: 1e-5
    warmup: 1000–2000 steps
    cosine decay
    batch tokens: 131,072
    total tokens: 3B cleaned
    seq_len: 1024 first, then 2048 for final 20%
    grad clip: 1.0
    z-loss: 1e-4

Validation:
    domain perplexity
    HellaSwag
    ARC-Easy
    PIQA
    WinoGrande
    custom factual QA
    gold eval set
    quantization health metrics
```

This gives you a clean, research-grade path for training a **small English general-knowledge ternary LLM from scratch**, with INT8-focused embedding/activation design and no FP16/BF16 pretrained-model route.
