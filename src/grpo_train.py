import os
import random
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer
from tqdm import tqdm

from .data import create_dataloader, TOTAL_TOKENS, MAX_PROMPT, MAX_ANSWER
from .model import AdaptiveGPT2Model
from .boundary_predictor import BoundaryPredictor
from .boundary_sampling import sample_boundaries_grpo, boundaries_to_log_probs


BETA = 0.3
ENTROPY_COEF = 0.01
NUM_SAMPLES_PER_PROMPT = 4
GROUP_SIZE = 16
TOTAL_SAMPLES = GROUP_SIZE * NUM_SAMPLES_PER_PROMPT


def evaluate_boundaries_batch(adaptive_model, prompt_ids, answer_ids, all_boundaries, max_span_len=4):
    S_total, B, L = all_boundaries.shape
    device = prompt_ids.device

    total_spans = all_boundaries.sum(dim=-1)
    max_spans = int(total_spans.max().item())
    K = max_span_len
    D = adaptive_model.config.n_embd

    prompt_emb = adaptive_model.transformer.wte(prompt_ids)

    span_emb = torch.zeros(S_total, B, max_spans, K, D, dtype=prompt_emb.dtype, device=device)
    span_msk = torch.zeros(S_total, B, max_spans, K, dtype=torch.bool, device=device)

    positions = torch.arange(L, device=device)
    all_losses = []

    for b_idx in range(B):
        prompt_emb_b = prompt_emb[b_idx]
        prompt_id_b = prompt_ids[b_idx]
        answer_id_b = answer_ids[b_idx]
        answer_emb_b = adaptive_model.transformer.wte(answer_id_b.unsqueeze(0))
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
            se_flat[flat_idx[valid]] = prompt_emb_b[positions[valid]]
            sm.view(-1)[flat_idx[valid]] = True

            merged = adaptive_model.span_encoder(se.unsqueeze(0), sm.unsqueeze(0))
            combined = torch.cat([merged, answer_emb_b], dim=1)
            combined_mask = torch.ones(1, combined.shape[1], dtype=torch.long, device=device)
            out = adaptive_model.transformer(inputs_embeds=combined, attention_mask=combined_mask)
            logits = adaptive_model.lm_head(out.last_hidden_state)

            labels = torch.cat([
                torch.full((1, n_spans), -100, dtype=torch.long, device=device),
                answer_id_b.unsqueeze(0).clone(),
            ], dim=1)
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1), ignore_index=-100,
            )
            config_losses.append(loss.item())

        all_losses.append(config_losses)

    losses_t = torch.tensor(all_losses, device=device)
    return losses_t


def train_grpo(
    phase2_checkpoint_dir="/checkpoints/phase2",
    volume=None,
    max_steps=5000,
    learning_rate=3e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading frozen Adaptive model...")
    adaptive_model = AdaptiveGPT2Model(base_model_name="gpt2", max_span_len=4)
    adaptive_model.to(device=device, dtype=amp_dtype)
    ckpt = torch.load("/checkpoints/adaptive/final.pt", map_location=device)
    adaptive_model.load_state_dict(ckpt["model_state_dict"])
    adaptive_model.eval()
    for p in adaptive_model.parameters():
        p.requires_grad = False

    print("Creating BoundaryPredictor...")
    embed_weight = adaptive_model.transformer.wte.weight.detach()
    predictor = BoundaryPredictor(embed_weight=embed_weight, hidden_dim=256, num_layers=1, num_heads=4)
    predictor.to(device=device)
    predictor.train()

    optimizer = AdamW(predictor.parameters(), lr=learning_rate, weight_decay=0.01)

    dataloader = create_dataloader(tokenizer, batch_size=GROUP_SIZE, max_tokens=TOTAL_TOKENS)

    start_step = 0
    if volume is not None:
        ckpt_path = os.path.join(phase2_checkpoint_dir, "latest.pt")
        if os.path.exists(ckpt_path):
            print("Resuming phase 2...")
            checkpoint = torch.load(ckpt_path, map_location=device)
            predictor.load_state_dict(checkpoint["predictor_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = checkpoint["step"]
            volume.reload()

    step = start_step
    pbar = tqdm(total=max_steps, initial=start_step, desc="GRPO training")
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

        log_probs_per = boundaries_to_log_probs(
            predictor, prompt_ids, boundaries, attention_mask=None,
        )

        advantages_flat = advantages.flatten()
        log_probs_flat = log_probs_per.flatten()
        policy_loss = -(log_probs_flat * advantages_flat).mean()
        total_loss = policy_loss

        optimizer.zero_grad()
        total_loss.backward()
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

        if step % 1000 == 0 and volume is not None:
            os.makedirs(phase2_checkpoint_dir, exist_ok=True)
            torch.save({
                "step": step,
                "predictor_state_dict": predictor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(phase2_checkpoint_dir, "latest.pt"))
            volume.commit()

    if volume is not None:
        os.makedirs(phase2_checkpoint_dir, exist_ok=True)
        torch.save({
            "step": step,
            "predictor_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, os.path.join(phase2_checkpoint_dir, "final.pt"))
        volume.commit()

    pbar.close()
    return predictor
