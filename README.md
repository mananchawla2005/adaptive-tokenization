# Adaptive Tokenization

Post-training models to handle variable token granularity via span compression.

## Concept

Standard tokenization is fixed — every token gets its own embedding. **Adaptive tokenization** lets the model merge nearby tokens into "spans" at runtime, reducing sequence length while preserving information.

We insert a lightweight **SpanEncoder** between the embedding layer and transformer. It groups token embeddings into spans and merges them into single representations. The model then processes fewer positions, trading granularity for context capacity.

## Approach

```
Standard:  [E0] [E1] [E2] [E3] [E4] [E5] ... → Transformer → LM Head
Adaptive:  [E0,E1] [E2] [E3,E4,E5] ... → SpanEncoder → [S0] [S1] [S2] ... → Transformer → LM Head
```

- **SpanEncoder**: Mean pooling + zero-initialized MLP residual (~7M params). Starts near-identity, learns to merge.
- **Compression curriculum**: 0% compression for first 10% of training, cosine ramp to 70% over 10-90%, constant at 70% for final 10%.
- **Loss**: Standard autoregressive cross-entropy on answer tokens only (prompt compression = preprocessing).

## Results (100M tokens, GPT-2 small, 6000 steps)

Evaluated on OpenOrca validation set. Both models tested with and without prompt compression.

| Model | No Merge (loss/ppl/bpb) | Random Merge (loss/ppl/bpb) | Compression degradation ↓ |
|-------|------------------------|-----------------------------|--------|
| **Naive** | 1.46 / 4.29 / 2.10 | 1.81 / 6.12 / 2.61 | 1.245 |
| **Adaptive** | 1.53 / 4.60 / 2.20 | 1.63 / 5.10 / 2.35 | **1.067** |

**Compression degradation** = merged_loss / no_merge_loss. 1.0 = no degradation from compression.

### Findings

- **Adaptive is compression-robust**: Only 7% degradation when compressing vs 24% for Naive
- **Minimal quality tradeoff**: Without compression, Adaptive is only ~4% worse than Naive
- **Curriculum works**: Starting at 0% compression lets the SpanEncoder warm up before handling aggressive merges
- **Speed**: V2 (Adaptive) trains ~3x slower than V1 due to SpanEncoder overhead — optimization needed for scale

## Metrics

| Metric | Meaning |
|--------|---------|
| **loss** | Cross-entropy on answer tokens (lower = better) |
| **ppl** | Perplexity = e^loss. Effective "branching factor" |
| **bpb** | Bits per byte = loss / ln(2). Encoding efficiency |
| **Compression degradation** | Compression ratio error = merged_loss / no_merge_loss |

## Project Structure

```
adaptive_tokenization/
├── src/
│   ├── span_encoder.py    # SpanEncoder: merges token groups
│   ├── merging.py         # Random span boundary generation
│   ├── data.py            # OpenOrca dataset (prompt/answer splits)
│   ├── model.py           # AdaptiveGPT2Model with forward_chat
│   ├── train.py           # V1 (naive) and V2 (adaptive) training loops
│   └── evaluate.py        # Dual-scenario evaluation
├── modal_app.py           # Modal cloud orchestration
└── README.md
```

## Running

```bash
# Set MAX_STEPS in modal_app.py, then:
modal run --detach modal_app.py::train_naive_fn    # V1 training
modal run --detach modal_app.py::train_adaptive_fn  # V2 training
modal run --detach modal_app.py::evaluate_fn        # Evaluation
```

Runs on Modal cloud with A100-80GB GPU. Checkpoints saved to Modal Volume for resumption.

## Next Steps

1. Scale to 1B tokens
2. Ablate curriculum schedule (constant vs ramping)
3. Multi-compression evaluation (cr=0.1–0.9)
4. Learned boundary prediction instead of random spans
5. Test on downstream QA/instruction benchmarks
