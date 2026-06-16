# Adaptive Tokenization — Experiment Log

## Goal

Make LLMs robust to variable token granularity. Insert a **SpanEncoder** that merges nearby token embeddings into span representations, reducing sequence length. Train a **BoundaryPredictor** via RL to decide which tokens to merge.

## Phase 1: Training the SpanEncoder

### Architecture

```
Standard:  [E₀] [E₁] [E₂] [E₃] ... → Transformer → LM Head
Adaptive:  [E₀,E₁] [E₂] [E₃,E₄,E₅] ... → SpanEncoder → [S₀] [S₁] [S₂] ... → Transformer → LM Head
```

**SpanEncoder** (~7M params):
- Mean pooling over tokens in each span
- MLP: `Linear(4D → 2D) → GELU → Linear(2D → D)` with zero-initialized residual
- Local position embedding per span position (zero-initialized)
- Starts near-identity for single-token spans, learns to merge

**Dataset**: Open-Orca/OpenOrca (instruction/response pairs). Prompt compressed via SpanEncoder, answer embedded normally. Loss only on answer tokens.

**Training**: GPT-2 small (124M) as base. Curriculum schedule: 0% compression for first 10% of steps, cosine ramp to 70% over 10-90%, constant at 70% for final 10%.

### Experiment 0: Initial Approach (Failed)

Started with C4 pretraining data, full-sequence compression, and loss on all positions as span-level prediction. Multiple failures:

1. **Wrong dataset**: C4 is pretraining data, not instruction data. Switched to OpenOrca.
2. **Wrong labels**: Loss should only be on answer tokens, not prompt tokens. During fine-tuning, you compute loss only on targets.
3. **Training on Modal**: `modal run --detach` with `@app.local_entrypoint()` chains functions sequentially via `.remote()`. If the local process disconnects, remote functions get cancelled. Fix: run functions individually with `modal run --detach modal_app.py::function_name`.

### Experiment 1: 100M C4 tokens (Failed)

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Loss spike (3.5→8.3) at compression onset | SpanEncoder's random weights injected noise when merging first activated | Zero-initialize `local_pos` and use small-init for MLP |
| Training slowed 40x at step ~1000 | `tokenizer.decode()` called per token position in Python loop; nested Python loops for span grouping | Remove tokenizer calls; vectorize with `scatter_reduce_`, `bincount` |
| Loss=8.8 even with zero-init | Double-shift bug: `target_token_ids` already stored next-span token, but `compute_loss` shifted labels again with `labels[:, 1:]` | Use `labels[:, :-1]` — no double shift |

### Experiment 2: 100M OpenOrca tokens (Successful)

| Config | Value |
|--------|-------|
| Model | GPT-2 small (124M) |
| Dataset | Open-Orca/OpenOrca |
| Tokens | 100M |
| Steps | 6,000 per version |
| Prompt max | 768 tokens |
| Answer max | 256 tokens |
| GPU | A100-80GB |
| Precision | bf16 mixed |

**V1 (Naive)**: Standard fine-tuning, loss on answer tokens only. 6000 steps, ~5 it/s.
**V2 (Adaptive)**: SpanEncoder with curriculum. 6000 steps, ~1.8 it/s (3x slower due to SpanEncoder overhead).

### Results

Evaluated on held-out OpenOrca examples. Two scenarios: no merge (standard) and random merge (cr=0.5).

| Model | No Merge (loss/ppl/bpb) | Random Merge (loss/ppl/bpb) | CORE ↓ |
|-------|------------------------|-----------------------------|--------|
| **Naive** | 1.46 / 4.29 / 2.10 | 1.81 / 6.12 / 2.61 | 1.245 |
| **Adaptive** | 1.53 / 4.60 / 2.20 | 1.63 / 5.10 / 2.35 | **1.067** |

CORE = merged_loss / no_merge_loss. Lower = less hurt by compression.

### Findings

- **Adaptive is compression-robust**: 6.7% degradation vs 24.5% for Naive
- **Minimal quality tradeoff**: Without compression, Adaptive is ~4% worse than Naive (1.53 vs 1.46)
- **Curriculum works**: Starting at 0% compression lets SpanEncoder warm up before merging
- **Speed**: V2 is ~3x slower — SpanEncoder overhead + extra tensor allocations per step

### Oracle Analysis

Brute-force sampled configurations for single prompts to find the loss-compression Pareto frontier.

**96-token prompt**: Best configuration at cr=61% achieves loss=6.16 — beating naive no-merge (6.32) by 2.5%.

**157-token prompt**: Best configuration at cr=31% achieves loss=2.10 — nearly matching naive no-merge (2.05) within 2.6%. Green zone (compression improves quality) extends to cr≈55%.

**Conclusion**: Smart boundary selection can *improve* prediction quality while using fewer tokens. The SpanEncoder acts as a noise filter — merging filler tokens improves signal for predicting answers. This motivates Phase 2: learning boundaries via RL.

---

## Phase 2: RL Boundary Predictor

### Architecture Evolution

**V1**: 2-layer transformer encoder, own 50K embedding (15M params). Too heavy.

**V2 (Final)**: Shared GPT-2 embedding (frozen), 1-layer transformer encoder (dim=256, 4 heads), MLP head → sigmoid per position. ~2M trainable params.

### Sampling Strategy Evolution

**Attempt 1: Nucleus sampling + temperature**: Predictor output → sigmoid → p_boundary → sample Bernoulli. All samples collapsed to identical output because the predictor quickly converged to high-confidence predictions.

**Attempt 2: Logit biases for forced diversity**: Added biases [2.0, 0, -1.5, -3.0] to logits. This worked for diversity (cr=0.1-0.7 range) but the predictor was learning from biased samples, not its natural output.

**Attempt 3: Unbiased + epsilon-greedy**: Removed biases, used epsilon=0.05 randomness. All 4 samples still nearly identical — 5% perturbation not enough for meaningful advantage variance.

**Attempt 4: Mild biases [-0.5, -0.15, 0.15, 0.5]**: Small perturbations meant to preserve natural output while adding diversity. Still insufficient — predictor too confident.

### GRPO Setup

| Parameter | Value |
|-----------|-------|
| Prompts per step | 16 |
| Samples per prompt | 4 |
| Configs evaluated per step | 64 |
| Reward | -loss + 0.3 × compression_ratio |
| Advantage | (reward - group_mean) / group_std |
| Loss | -(log_prob × advantage).mean() |

### Current Challenge

GRPO with Bernoulli sampling can't produce enough variance for a useful advantage signal. The predictor converges to confident outputs → all 4 samples identical → advantage=0 → no gradient → no learning.

**Proposed fix**: Switch to **Best-of-K supervised** — sample 4 random configs, pick best, supervised training to imitate it. Guaranteed diversity.

### Phase 2 Results

| Method | Compression | Loss | CORE |
|--------|------------|------|------|
| No-merge baseline | 0% | 1.363 | 1.000 |
| Learned predictor (biased GRPO, 1K steps) | 75% | 1.528 | 1.121 |
| Random merge | 86% | 1.551 | 1.138 |

(The GRPO-only predictor was trained with biased sampling. Unbiased training did not converge due to the diversity problem described above.)

### Solution: Hybrid 3-Stage Training

The fundamental issue with pure GRPO is that Bernoulli sampling from a sigmoid head collapses to deterministic output. When the predictor becomes confident, all 4 samples per prompt are identical → advantage = 0 → no gradient.

**Stage 1: Oracle Dataset**
- For each of 1000 prompts, sample K=128 random boundary configurations
- Score all 128 via the frozen Adaptive model (reward = -loss + 0.3 × cr)
- Keep the best configuration per prompt
- Store (prompt, best_boundaries) pairs
- Guaranteed diversity — random sampling never collapses

**Stage 2: Supervised Imitation**
- BCE loss per position: predictor learns to predict oracle boundaries
- 5 epochs, batch size 32
- Gives the predictor a strong initialization — it already knows what "good" looks like

**Stage 3: GRPO Fine-tuning**
- Start from imitation checkpoint (not random)
- 2000 steps, group size 16, 4 samples per prompt
- Mild logit biases [-0.5, -0.15, 0.15, 0.5] for diversity
- The predictor starts from good layouts → maintains enough variance for RL

### Hybrid Phase 2 Results

| Method | Compression | Loss | CORE |
|--------|------------|------|------|
| No-merge baseline | 0% | 1.363 | 1.000 |
| Random merge | 56% | 1.493 | 1.096 |
| **Learned (hybrid)** | **65%** | **1.355** | **0.994** |

**CORE < 1.0**: The learned predictor achieves *better* quality than no compression while using 65% fewer prompt tokens. Compression improves predictions — the SpanEncoder + BoundaryPredictor together act as a learned noise filter.

### Why Hybrid Works

- **Stage 1 guarantees diversity**: Random sampling can't collapse, and the oracle dataset provides a strong supervised signal
- **Stage 2 provides warm start**: BCE gives the predictor a good initialization — it learns "what kinds of structures survive compression" from the oracle
- **Stage 3 fine-tunes from strength**: Starting from good layouts, the predictor maintains enough variance across samples for the RL advantage signal to be meaningful
- Pure GRPO from scratch fails because the predictor has no idea what "good" looks like and collapses to the safe default (all boundaries)

---

## Key Technical Lessons

### Modal Platform
- `modal run --detach function_name` works for individual functions
- `@app.local_entrypoint()` chains fail if local process disconnects — remote calls get cancelled
- Checkpoints saved to Modal Volumes survive between runs
- Always use `--detach` for long-running jobs

### Training
- In fine-tuning/post-training, **loss is only on target/answer tokens**, never on prompt
- Curriculum schedules are critical for stability when introducing new modules
- Zero-initializing new sub-modules prevents "random noise shock" at activation

### SpanEncoder Design
- Mean pooling + small learnable residual is sufficient
- Working in logit space (not probability space) for logit adjustments
- Vectorized span grouping is critical for performance — Python loops kill GPU throughput

### RL for Boundary Prediction
- Bernoulli sampling from a sigmoid head collapses to deterministic output
- GRPO needs meaningful variance within each group — if all samples identical, advantage = 0
- Pure GRPO from scratch fails for this problem; the predictor has no idea what "good" looks like
- **Hybrid 3-stage approach works**: oracle dataset (random → best) → supervised imitation → GRPO fine-tune
- Shared embeddings dramatically reduce parameter count (15M → 2M) without quality loss
- **CORE < 1.0 achieved**: learned compression can improve prediction quality over no compression
