import os
import math
import random
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer
from tqdm import tqdm

from .data import create_dataloader, TOTAL_TOKENS, MAX_PROMPT, MAX_ANSWER
from .model import AdaptiveGPT2Model


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _compression_schedule(progress, max_compression=0.7):
    if progress < 0.1:
        return 0.0
    elif progress > 0.9:
        return max_compression
    else:
        frac = (progress - 0.1) / 0.8
        return max_compression * 0.5 * (1.0 - math.cos(math.pi * frac))


def _setup_model_and_optimizer(
    model_name,
    device,
    max_span_len=4,
    learning_rate=1e-4,
    warmup_steps=500,
    total_steps=6103,
):
    model = AdaptiveGPT2Model(base_model_name=model_name, max_span_len=max_span_len)
    model = model.to(dtype=torch.bfloat16, device=device)

    all_params = list(model.parameters())
    optimizer = AdamW(all_params, lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    return model, optimizer, scheduler


def _save_checkpoint(model, optimizer, step, checkpoint_dir, volume, final=False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_dict = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state_dict, os.path.join(checkpoint_dir, "latest.pt"))
    if final:
        torch.save(state_dict, os.path.join(checkpoint_dir, "final.pt"))
    else:
        torch.save(state_dict, os.path.join(checkpoint_dir, f"step_{step}.pt"))
    volume.commit()
    print(f"Checkpoint saved at step {step} (latest + {'final' if final else f'step_{step}'})")


def train_naive(
    model_name="gpt2",
    batch_size=4,
    grad_accum_steps=4,
    learning_rate=1e-4,
    warmup_steps=500,
    max_steps=None,
    checkpoint_dir="/checkpoints/naive",
    volume=None,
    total_tokens=TOTAL_TOKENS,
    tracker=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    print(f"[V1 Naive] Training on: {device}, AMP: {use_amp}, dtype: {amp_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    tokens_per_step = batch_size * (MAX_PROMPT + MAX_ANSWER)
    total_steps = total_tokens // tokens_per_step
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    dataloader = create_dataloader(tokenizer, batch_size=batch_size, max_tokens=total_tokens)

    print(f"Total training steps: {total_steps}")
    print(f"Batch size: {batch_size}, Prompt max: {MAX_PROMPT}, Answer max: {MAX_ANSWER}")

    base_model, optimizer, scheduler = _setup_model_and_optimizer(
        model_name, device,
        learning_rate=learning_rate, warmup_steps=warmup_steps, total_steps=total_steps,
    )

    start_step = 0
    if volume is not None:
        ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
        if os.path.exists(ckpt_path):
            print("Resuming from checkpoint...")
            checkpoint = torch.load(ckpt_path, map_location=device)
            base_model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = checkpoint["step"]
            volume.reload()

    base_model.train()
    accumulated_loss = 0.0
    step = start_step
    pbar = tqdm(total=total_steps, initial=start_step, desc="Naive training")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        if step >= total_steps:
            break

        prompt_ids = batch["prompt_ids"].to(device)
        answer_ids = batch["answer_ids"].to(device)

        with torch.amp.autocast("cuda" if use_amp else "cpu", dtype=amp_dtype):
            loss = base_model.forward_chat_no_compress(prompt_ids, answer_ids)

        loss = loss / grad_accum_steps
        loss.backward()
        accumulated_loss += loss.item()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": f"{accumulated_loss:.4f}",
                "lr": f"{lr:.2e}",
            })
            pbar.update(1)

            if tracker is not None:
                tracker.log({
                    "train/loss": accumulated_loss,
                    "train/lr": lr,
                    "train/tokens": step * batch_size * grad_accum_steps * (MAX_PROMPT + MAX_ANSWER),
                }, step=step)
            accumulated_loss = 0.0

            if step % 1000 == 0 and volume is not None:
                _save_checkpoint(base_model, optimizer, step, checkpoint_dir, volume)

    if volume is not None:
        _save_checkpoint(base_model, optimizer, step, checkpoint_dir, volume, final=True)

    pbar.close()

    if tracker is not None:
        tracker.summary({
            "final_step": step,
            "total_tokens_processed": step * batch_size * grad_accum_steps * (MAX_PROMPT + MAX_ANSWER),
        })

    return base_model, step


def train_adaptive(
    model_name="gpt2",
    batch_size=4,
    grad_accum_steps=4,
    learning_rate=1e-4,
    warmup_steps=500,
    max_steps=None,
    checkpoint_dir="/checkpoints/adaptive",
    volume=None,
    max_compression_ratio=0.7,
    max_span_len=4,
    total_tokens=TOTAL_TOKENS,
    tracker=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    print(f"[V2 Adaptive] Training on: {device}, AMP: {use_amp}, dtype: {amp_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    tokens_per_step = batch_size * (MAX_PROMPT + MAX_ANSWER)
    total_steps = total_tokens // tokens_per_step
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    dataloader = create_dataloader(tokenizer, batch_size=batch_size, max_tokens=total_tokens)

    print(f"Total training steps: {total_steps}")
    print(f"Batch size: {batch_size}, Prompt max: {MAX_PROMPT}, Answer max: {MAX_ANSWER}")
    print(f"Max compression ratio: {max_compression_ratio}")
    print(f"Schedule: 0% for first 10%, cosine ramp to {max_compression_ratio} over 10-90%, then constant")

    base_model, optimizer, scheduler = _setup_model_and_optimizer(
        model_name, device, max_span_len=max_span_len,
        learning_rate=learning_rate, warmup_steps=warmup_steps, total_steps=total_steps,
    )

    start_step = 0
    if volume is not None:
        ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
        if os.path.exists(ckpt_path):
            print("Resuming from checkpoint...")
            checkpoint = torch.load(ckpt_path, map_location=device)
            base_model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = checkpoint["step"]
            volume.reload()

    base_model.train()
    accumulated_loss = 0.0
    step = start_step
    pbar = tqdm(total=total_steps, initial=start_step, desc="Adaptive training")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        if step >= total_steps:
            break

        progress = step / max(total_steps, 1)
        current_cr = _compression_schedule(progress, max_compression_ratio)

        prompt_ids = batch["prompt_ids"].to(device)
        answer_ids = batch["answer_ids"].to(device)

        rng = random.Random(step * batch_idx)
        with torch.amp.autocast("cuda" if use_amp else "cpu", dtype=amp_dtype):
            loss = base_model.forward_chat(
                prompt_ids, answer_ids,
                tokenizer=tokenizer,
                compression_ratio=max(current_cr, 0.001),
                rng=rng,
            )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf loss at step {step}, skipping batch")
            optimizer.zero_grad()
            continue

        loss = loss / grad_accum_steps
        loss.backward()
        accumulated_loss += loss.item()

        if (batch_idx + 1) % grad_accum_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"WARNING: NaN/Inf gradient at step {step}, skipping update")
                optimizer.zero_grad()
                step += 1
                pbar.update(1)
                accumulated_loss = 0.0
                continue

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            lr = scheduler.get_last_lr()[0]
            num_spans = getattr(base_model, '_last_num_spans', MAX_PROMPT)
            actual_cr = 1.0 - num_spans / MAX_PROMPT
            pbar.set_postfix({
                "loss": f"{accumulated_loss:.4f}",
                "cr": f"{current_cr:.2f}",
                "act_cr": f"{actual_cr:.2f}",
                "lr": f"{lr:.2e}",
            })
            pbar.update(1)

            if tracker is not None:
                tracker.log({
                    "train/loss": accumulated_loss,
                    "train/lr": lr,
                    "train/cr_schedule": current_cr,
                    "train/cr_actual": actual_cr,
                    "train/compression": MAX_PROMPT / max(num_spans, 1),
                    "train/tokens": step * batch_size * grad_accum_steps * (MAX_PROMPT + MAX_ANSWER),
                }, step=step)
            accumulated_loss = 0.0

            if step % 1000 == 0 and volume is not None:
                _save_checkpoint(base_model, optimizer, step, checkpoint_dir, volume)

    if volume is not None:
        _save_checkpoint(base_model, optimizer, step, checkpoint_dir, volume, final=True)

    pbar.close()

    if tracker is not None:
        tracker.summary({
            "final_step": step,
            "total_tokens_processed": step * batch_size * grad_accum_steps * (MAX_PROMPT + MAX_ANSWER),
        })

    return base_model, step
