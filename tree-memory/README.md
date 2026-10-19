# Tree-Memory

A hierarchical memory for language models that reads the past in logarithmic time, keeps every fact, and learns while it runs.

- **Report:** [tree_memory_v1_and_v2.md](tree_memory_v1_and_v2.md)
- **Experiment logs:** [experiments/](experiments/) (exp01–exp09)

**Headline result (exp09, chess memory world):** reads ≈ 9.4 × log₂ n from 4K to 1M addresses — 189 of 1,048,576 — with recall 1.000 and answers identical to full attention.

**Open:** conversion of pretrained language models (needs trained routing keys), a fused GPU kernel for wall-clock gains, and noise-robust keys for text.
