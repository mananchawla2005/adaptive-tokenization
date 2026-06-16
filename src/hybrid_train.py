"""
Phase 2 Hybrid: 3-stage boundary predictor training
  Stage 1: Create oracle dataset (random sampling → best config per prompt)
  Stage 2: Supervised imitation (BCE loss on best boundaries)
  Stage 3: Online GRPO fine-tuning
"""
import os
import random
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from .data import create_dataloader, TOTAL_TOKENS
from .model import AdaptiveGPT2Model
from .boundary_predictor import BoundaryPredictor
from .boundary_sampling import sample_boundaries_grpo, boundaries_to_log_probs


BETA = 0.3
K_SAMPLES = 128
NUM_PROMPTS_ORACLE = 2000
BCE_EPOCHS = 5
BCE_BATCH = 32
GRPO_STEPS = 2000
GROUP_SIZE = 16
NUM_SAMPLES_PER_PROMPT = 4


def evaluate_boundaries_batch(adaptive_model, prompt_ids, answer_ids, all_boundaries, max_span_len=4):
    """Same as before — returns losses as (B, S) tensor"""
    S_total, B, L = all_boundaries.shape
    device = prompt_ids.device
    K = max_span_len
    D = adaptive_model.config.n_embd
    positions = torch.arange(L, device=device)
    all_losses = []

    for b_idx in range(B):
        prompt_emb_b = adaptive_model.transformer.wte(prompt_ids[b_idx].unsqueeze(0))
        answer_id_b = answer_ids[b_idx].unsqueeze(0)
        answer_emb_b = adaptive_model.transformer.wte(answer_id_b)
        Al = answer_ids.shape[1]
        config_losses = []

        for s in range(S_total):
            bnd = all_boundaries[s, b_idx]
            n_spans = int(bnd.sum().item())
            if n_spans == 0:
                config_losses.append(0.0)
                continue

            assigns = torch.zeros(L, dtype=torch.long, device=device)
            sid = 0; pos = 0
            while pos < L:
                end = pos + 1
                while end < L and not bnd[end]:
                    end += 1
                assigns[pos:end] = sid
                sid += 1
                pos = end

            first_pos = positions[bnd]
            local_k = positions - first_pos[assigns]
            valid = local_k < K
            flat_idx = assigns * K + local_k

            se = torch.zeros(n_spans, K, D, dtype=prompt_emb_b.dtype, device=device)
            sm = torch.zeros(n_spans, K, dtype=torch.bool, device=device)
            se_flat = se.view(-1, D)
            se_flat[flat_idx[valid]] = prompt_emb_b[0, positions[valid]]
            sm.view(-1)[flat_idx[valid]] = True

            merged = adaptive_model.span_encoder(se.unsqueeze(0), sm.unsqueeze(0))
            combined = torch.cat([merged, answer_emb_b], dim=1)
            combined_mask = torch.ones(1, combined.shape[1], dtype=torch.long, device=device)
            out = adaptive_model.transformer(inputs_embeds=combined, attention_mask=combined_mask)
            logits = adaptive_model.lm_head(out.last_hidden_state)

            labels = torch.cat([
                torch.full((1, n_spans), -100, dtype=torch.long, device=device),
                answer_id_b.clone(),
            ], dim=1)
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1), ignore_index=-100,
            )
            config_losses.append(loss.item())

        all_losses.append(config_losses)

    return torch.tensor(all_losses, device=device)


def sample_random_boundaries(prompt_ids, num_samples, max_span_len=4):
    B, L = prompt_ids.shape
    device = prompt_ids.device
    boundaries = torch.zeros(num_samples, B, L, dtype=torch.bool, device=device)

    for k in range(num_samples):
        target_cr = random.random() * 0.75
        for b in range(B):
            pos = 0; sid = 0
            while pos < L:
                if sid > 0:
                    cur_cr = 1.0 - (sid + (L - pos)) / L
                    if cur_cr >= target_cr:
                        boundaries[k, b, pos] = True
                        pos += 1; sid += 1
                        continue
                remaining = L - pos
                slen = random.randint(1, min(max_span_len, remaining) + 1)
                slen = min(slen, remaining)
                boundaries[k, b, pos] = True
                pos += slen; sid += 1

    return boundaries


# ═══════════════════════════════════════════════════════════════
# Stage 1: Oracle Dataset
# ═══════════════════════════════════════════════════════════════

def stage1_create_oracle(
    adaptive_model, tokenizer, output_dir="/checkpoints/phase2",
    volume=None, num_prompts=NUM_PROMPTS_ORACLE, k_samples=K_SAMPLES,
):
    device = next(adaptive_model.parameters()).device
    dtype = adaptive_model.transformer.wte.weight.dtype

    dataloader = create_dataloader(tokenizer, batch_size=2, max_tokens=min(TOTAL_TOKENS, 100_000_000))

    all_prompt_ids = []
    all_answer_ids = []
    all_best_boundaries = []

    pbar = tqdm(total=num_prompts, desc="Stage 1: Oracle dataset")
    collected = 0

    for batch in dataloader:
        if collected >= num_prompts:
            break

        prompt_ids = batch["prompt_ids"].to(device)
        answer_ids = batch["answer_ids"].to(device)
        B, L = prompt_ids.shape

        boundaries = sample_random_boundaries(prompt_ids, k_samples, max_span_len=4)

        with torch.no_grad():
            losses_raw = evaluate_boundaries_batch(
                adaptive_model, prompt_ids, answer_ids, boundaries, max_span_len=4,
            )

        losses_t = losses_raw.t().contiguous()
        n_spans = boundaries.sum(dim=-1).float()
        cr = 1.0 - n_spans / L
        rewards = -losses_t + BETA * cr

        best_idx = rewards.argmax(dim=0)

        for b in range(B):
            best_bnd = boundaries[best_idx[b], b].cpu().clone()
            all_prompt_ids.append(prompt_ids[b].cpu().clone())
            all_answer_ids.append(answer_ids[b].cpu().clone())
            all_best_boundaries.append(best_bnd)
            collected += 1
            pbar.update(1)

            if collected >= num_prompts:
                break

    pbar.close()

    max_p = max(p.shape[0] for p in all_prompt_ids)
    max_a = max(a.shape[0] for a in all_answer_ids)

    prompts_t = torch.full((len(all_prompt_ids), max_p), 0, dtype=torch.long)
    answers_t = torch.full((len(all_answer_ids), max_a), 0, dtype=torch.long)
    boundaries_t = torch.zeros(len(all_best_boundaries), max_p, dtype=torch.bool)

    for i in range(len(all_prompt_ids)):
        pl = all_prompt_ids[i].shape[0]
        al = all_answer_ids[i].shape[0]
        prompts_t[i, :pl] = all_prompt_ids[i]
        answers_t[i, :al] = all_answer_ids[i]
        boundaries_t[i, :all_best_boundaries[i].shape[0]] = all_best_boundaries[i]

    if volume is not None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save({
            "prompts": prompts_t,
            "answers": answers_t,
            "boundaries": boundaries_t,
        }, os.path.join(output_dir, "oracle_dataset.pt"))
        volume.commit()
        print(f"Saved oracle dataset: {len(all_prompt_ids)} examples")

    return prompts_t, answers_t, boundaries_t


# ═══════════════════════════════════════════════════════════════
# Stage 2: Supervised Imitation (BCE)
# ═══════════════════════════════════════════════════════════════

def stage2_train_imitation(
    adaptive_model, tokenizer, output_dir="/checkpoints/phase2",
    volume=None, epochs=BCE_EPOCHS, batch_size=BCE_BATCH,
):
    device = next(adaptive_model.parameters()).device
    embed_weight = adaptive_model.transformer.wte.weight.detach()

    oracle_path = os.path.join(output_dir, "oracle_dataset.pt")
    if volume is not None:
        volume.reload()

    if not os.path.exists(oracle_path):
        print("No oracle dataset found. Run Stage 1 first.")
        return None

    data = torch.load(oracle_path, map_location="cpu")
    prompts_t = data["prompts"]
    boundaries_t = data["boundaries"].float()

    predictor = BoundaryPredictor(embed_weight=embed_weight, hidden_dim=256, num_layers=1, num_heads=4)
    predictor.to(device)

    dataset = TensorDataset(prompts_t, boundaries_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(predictor.parameters(), lr=3e-4, weight_decay=0.01)

    for epoch in range(epochs):
        predictor.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Stage 2: BCE epoch {epoch+1}/{epochs}")

        for prompt_batch, bnd_batch in pbar:
            prompt_batch = prompt_batch.to(device)
            bnd_batch = bnd_batch.to(device)

            logits = predictor.forward(prompt_batch)
            bce_loss = F.binary_cross_entropy_with_logits(logits, bnd_batch)

            optimizer.zero_grad()
            bce_loss.backward()
            optimizer.step()

            total_loss += bce_loss.item()
            pbar.set_postfix({"bce": f"{bce_loss.item():.4f}"})

        print(f"Epoch {epoch+1}: avg BCE = {total_loss / len(loader):.4f}")

    if volume is not None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save({
            "predictor_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, os.path.join(output_dir, "stage2_imitation.pt"))
        volume.commit()

    return predictor


# ═══════════════════════════════════════════════════════════════
# Stage 3: Online GRPO Fine-tuning
# ═══════════════════════════════════════════════════════════════

def stage3_train_grpo(
    adaptive_model, tokenizer, predictor, output_dir="/checkpoints/phase2",
    volume=None, max_steps=GRPO_STEPS,
):
    device = next(adaptive_model.parameters()).device

    dataloader = create_dataloader(tokenizer, batch_size=GROUP_SIZE, max_tokens=min(TOTAL_TOKENS, 50_000_000))

    optimizer = AdamW(predictor.parameters(), lr=1e-4, weight_decay=0.01)

    predictor.train()
    step = 0
    pbar = tqdm(total=max_steps, desc="Stage 3: GRPO fine-tune")
    accumulated_reward = 0.0

    for batch in dataloader:
        if step >= max_steps:
            break

        prompt_ids = batch["prompt_ids"].to(device)
        answer_ids = batch["answer_ids"].to(device)
        B, Pl = prompt_ids.shape

        boundaries = sample_boundaries_grpo(
            predictor, prompt_ids, num_samples=NUM_SAMPLES_PER_PROMPT, max_span_len=4,
        )

        with torch.no_grad():
            losses_raw = evaluate_boundaries_batch(
                adaptive_model, prompt_ids, answer_ids, boundaries, max_span_len=4,
            )
        losses_t = losses_raw.t().contiguous().to(device)

        n_spans = boundaries.sum(dim=-1).float()
        compression_ratios = 1.0 - n_spans / Pl
        rewards = -losses_t + BETA * compression_ratios

        mean_r = rewards.mean(dim=0, keepdim=True)
        std_r = rewards.std(dim=0, keepdim=True) + 1e-8
        advantages = (rewards - mean_r) / std_r

        log_probs_per = boundaries_to_log_probs(predictor, prompt_ids, boundaries)

        policy_loss = -(log_probs_per.flatten() * advantages.flatten()).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        step += 1
        accumulated_reward += rewards.mean().item()

        pbar.set_postfix({
            "reward": f"{accumulated_reward / max(1, step):.3f}",
            "loss": f"{losses_t.mean().item():.3f}",
            "cr": f"{compression_ratios.mean().item():.2f}",
        })
        pbar.update(1)

        if step % 500 == 0 and volume is not None:
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                "step": step,
                "predictor_state_dict": predictor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(output_dir, "stage3_grpo.pt"))
            volume.commit()

    if volume is not None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save({
            "step": step,
            "predictor_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, os.path.join(output_dir, "final_predictor.pt"))
        volume.commit()

    pbar.close()
    return predictor
