# Adaptive Tokenization

Post-training models to handle variable token granularity via span compression.

## Concept

Standard tokenization is fixed — every token gets its own embedding. **Adaptive tokenization** lets the model merge nearby tokens into "spans" at runtime, reducing sequence length while preserving information.

We insert a lightweight **SpanEncoder** between the embedding layer and transformer. It groups token embeddings into spans and merges them into single representations. The model then processes fewer positions, trading granularity for context capacity.

```
Standard:  [E0] [E1] [E2] [E3] [E4] [E5] ... → Transformer → LM Head
Adaptive:  [E0,E1] [E2] [E3,E4,E5] ... → SpanEncoder → [S0] [S1] [S2] ... → Transformer → LM Head
```

- **SpanEncoder**: Mean pooling + zero-initialized MLP residual (~7M params). Starts near-identity, learns to merge.
- **Compression curriculum**: 0% compression for first 10% of training, cosine ramp to 70% over 10-90%, constant at 70% for final 10%.
- **Loss**: Standard autoregressive cross-entropy on answer tokens only (prompt compression = preprocessing).
- **Dataset**: [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) — instruction/response pairs, scalable to 1B+ tokens.

## Phase 1: SpanEncoder Training

**Setup**: GPT-2 small (124M + 7M SpanEncoder), 100M tokens, 6000 steps, A100-80GB, bf16 mixed precision.

**Naive**: Standard GPT-2 fine-tuning on OpenOrca. Loss on answer tokens only.
**Adaptive**: Same but with SpanEncoder compressing the prompt via curriculum schedule.

### Evaluation

Both models evaluated on held-out OpenOrca examples with no-merge and random merge (cr=0.5).

| Model | No Merge (loss/ppl/bpb) | Random Merge (loss/ppl/bpb) | CORE ↓ |
|-------|------------------------|-----------------------------|--------|
| **Naive** | 1.46 / 4.29 / 2.10 | 1.81 / 6.12 / 2.61 | 1.245 |
| **Adaptive** | 1.53 / 4.60 / 2.20 | 1.63 / 5.10 / 2.35 | **1.067** |

CORE = merged_loss / no_merge_loss. Lower = less degradation.

### Oracle Analysis

Brute-force sampled configurations across compression ratios to find the loss-compression Pareto frontier.

![Oracle Plot](oracle_dual.png)

| Metric | 96-token prompt | 157-token prompt |
|--------|----------------|------------------|
| Naive no-merge | 6.32 | 2.05 |
| Adaptive no-merge | 6.22 | 2.22 |
| **Adaptive best (with compression)** | **6.16 (cr=61%)** | **2.10 (cr=31%)** |
| Gain vs Naive | **+2.5% better** | -2.6% worse |

Key finding: Smart compression can *beat* no compression — the SpanEncoder acts as a noise filter, improving prediction signal.

---

## Phase 2: Learned Boundary Prediction (Hybrid RL)

**Goal**: Train a BoundaryPredictor to decide which tokens to merge, maximizing reward (low loss + high compression).

**Architecture**: Shared GPT-2 embedding (frozen), 1-layer transformer encoder (dim=256, 4 heads), MLP head → sigmoid per position. ~2M trainable params.

### 3-Stage Training

1. **Oracle Dataset (Stage 1)**: Sample K=128 random boundary configs per prompt. Score all via frozen Adaptive model. Keep best (reward = -loss + 0.3 × cr). Store 1000 (prompt, best_boundaries) pairs.

2. **Supervised Imitation (Stage 2)**: BCE loss — train predictor to predict oracle boundaries. 5 epochs, batch=32. Gives the predictor a strong initialization.

3. **GRPO Fine-tuning (Stage 3)**: Online RL starting from imitation checkpoint. 2000 steps, group size 16, 4 samples per prompt. Reward = -loss + 0.3 × cr.

### Results

| Method | Compression | Loss | CORE |
|--------|------------|------|------|
| No-merge baseline | 0% | 1.363 | 1.000 |
| Random merge | 56% | 1.493 | 1.096 |
| **Learned (hybrid)** | **65%** | **1.355** | **0.994** |

**CORE < 1.0** — the learned predictor achieves *better* quality than no compression while using 65% fewer prompt tokens. Compression improves predictions.

### Why Hybrid Works

- Pure GRPO failed because Bernoulli sampling from a sigmoid head collapses to deterministic output → zero advantage → no gradient
- Stage 1 guarantees diversity (random sampling) and creates a strong supervised signal
- Stage 2 teaches the predictor what "good" boundaries look like
- Stage 3 fine-tunes from good layouts, maintaining enough variance for RL to work

## Metrics

| Metric | Meaning |
|--------|---------|
| **loss** | Cross-entropy on answer tokens (lower = better) |
| **ppl** | Perplexity = e^loss. Effective "branching factor" |
| **bpb** | Bits per byte = loss / ln(2). Encoding efficiency |
| **CORE** | Compression ratio error = merged_loss / no_merge_loss. < 1.0 = compression improves quality |

## Project Structure

```
adaptive_tokenization/
├── src/
│   ├── span_encoder.py       # SpanEncoder: merges token groups
│   ├── merging.py            # Random span boundary generation
│   ├── data.py               # OpenOrca dataset (prompt/answer splits)
│   ├── model.py              # AdaptiveGPT2Model with forward_chat
│   ├── train.py              # V1 (naive) and V2 (adaptive) training loops
│   ├── evaluate.py           # Dual-scenario evaluation
│   ├── boundary_predictor.py # Phase 2: learned boundary predictor
│   ├── boundary_sampling.py  # Phase 2: sampling with logit biases
│   └── hybrid_train.py       # Phase 2: 3-stage training (oracle → BCE → GRPO)
├── modal_app.py              # Phase 1 Modal orchestration
├── phase2_app.py             # Phase 2 Modal orchestration
├── oracle_dual.png           # Oracle analysis plot
├── plot_oracle.py            # Plot generation script
├── README.md
└── EXPERIMENTS.md            # Detailed experiment log
```

## Running

```bash
# Phase 1
modal run --detach modal_app.py::train_naive_fn
modal run --detach modal_app.py::train_adaptive_fn
modal run --detach modal_app.py::evaluate_fn
modal run --detach modal_app.py::oracle_fn

# Phase 2
modal run --detach phase2_app.py::train_hybrid_fn
modal run --detach phase2_app.py::evaluate_hybrid_fn
```

Runs on Modal cloud with A100-80GB GPU.

## Next Steps

1. **Scale tokens** — 100M → 1B tokens to close the Adaptive/Naive gap
2. **Ablate curriculum** — compare constant compression vs ramping schedule
3. **Multi-compression eval** — measure at cr=0.1–0.9 to plot full degradation curve
4. **End-to-end generation** — use predictor at inference for actual text generation, measure speed + quality
5. **Downstream benchmarks** — test on QA tasks where context compression matters (RULER, LongBench)
