import os
import sys
import time
import modal
from pathlib import Path


GPU_CONFIG = "A100-80GB"
GPU_COUNT = 1
TRAINING_TIMEOUT = 23 * 60 * 60
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_PROMPT = 768
MAX_ANSWER = 256
WARMUP_STEPS = 200
LEARNING_RATE = 1e-4
TOTAL_TOKENS = 100_000_000
MAX_COMPRESSION_RATIO = 0.7
MAX_SPAN_LEN = 4
MAX_STEPS = 6000
EXPERIMENT_SEED = 42
VOLUME_NAME = "adaptive-tok-v2"

SOURCE_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch>=2.5.0",
        "transformers>=4.46.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        "tqdm>=4.66.0",
        "wandb>=0.18.0",
    )
    .add_local_dir(SOURCE_DIR / "src", remote_path="/root/src", copy=True)
)

app = modal.App("adaptive-tokenization", image=image)

checkpoints_volume = modal.Volume.from_name(
    VOLUME_NAME, create_if_missing=True
)

wandb_secret = modal.Secret.from_dict({
    "WANDB_API_KEY": "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"
})


@app.function(
    gpu=f"{GPU_CONFIG}:{GPU_COUNT}",
    timeout=TRAINING_TIMEOUT,
    volumes={"/checkpoints": checkpoints_volume},
    secrets=[wandb_secret],
    retries=modal.Retries(
        initial_delay=0.0,
        max_retries=3,
        backoff_coefficient=1.0,
    ),
)
def train_naive_fn():
    from src.tracker import Tracker, generate_experiment_id, set_seed
    set_seed(EXPERIMENT_SEED)
    exp_id = generate_experiment_id()
    tracker = Tracker(
        name=f"{exp_id}-phase1-naive",
        project="adaptive-tokenization",
        group=exp_id,
        job_type="phase1-naive",
        config={
            "seed": EXPERIMENT_SEED,
            "model": "gpt2",
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "warmup_steps": WARMUP_STEPS,
            "max_prompt": MAX_PROMPT,
            "max_answer": MAX_ANSWER,
            "total_tokens": TOTAL_TOKENS,
            "experiment_id": exp_id,
            "volume": VOLUME_NAME,
        },
    )
    tracker.init()

    sys.path.insert(0, "/root")
    from src.train import train_naive

    checkpoints_volume.reload()

    model, steps = train_naive(
        model_name="gpt2",
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        checkpoint_dir="/checkpoints/naive",
        volume=checkpoints_volume,
        tracker=tracker,
    )
    print(f"Naive training completed: {steps} steps")
    tracker.finish()
    return steps


@app.function(
    gpu=f"{GPU_CONFIG}:{GPU_COUNT}",
    timeout=TRAINING_TIMEOUT,
    volumes={"/checkpoints": checkpoints_volume},
    secrets=[wandb_secret],
    retries=modal.Retries(
        initial_delay=0.0,
        max_retries=3,
        backoff_coefficient=1.0,
    ),
)
def train_adaptive_fn():
    from src.tracker import Tracker, generate_experiment_id, set_seed
    set_seed(EXPERIMENT_SEED)
    exp_id = generate_experiment_id()
    tracker = Tracker(
        name=f"{exp_id}-phase1-adaptive",
        project="adaptive-tokenization",
        group=exp_id,
        job_type="phase1-adaptive",
        config={
            "seed": EXPERIMENT_SEED,
            "model": "gpt2",
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "warmup_steps": WARMUP_STEPS,
            "max_compression_ratio": MAX_COMPRESSION_RATIO,
            "max_span_len": MAX_SPAN_LEN,
            "max_prompt": MAX_PROMPT,
            "max_answer": MAX_ANSWER,
            "total_tokens": TOTAL_TOKENS,
            "experiment_id": exp_id,
            "volume": VOLUME_NAME,
        },
    )
    tracker.init()

    sys.path.insert(0, "/root")
    from src.train import train_adaptive

    checkpoints_volume.reload()

    model, steps = train_adaptive(
        model_name="gpt2",
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        checkpoint_dir="/checkpoints/adaptive",
        volume=checkpoints_volume,
        max_compression_ratio=MAX_COMPRESSION_RATIO,
        max_span_len=MAX_SPAN_LEN,
        tracker=tracker,
    )
    print(f"Adaptive training completed: {steps} steps")
    tracker.finish()
    return steps


@app.function(
    gpu=f"{GPU_CONFIG}:{GPU_COUNT}",
    timeout=3 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume},
    secrets=[wandb_secret],
)
def evaluate_fn():
    from src.tracker import Tracker, generate_experiment_id, set_seed
    set_seed(EXPERIMENT_SEED)
    exp_id = generate_experiment_id()
    tracker = Tracker(
        name=f"{exp_id}-phase1-eval",
        project="adaptive-tokenization",
        group=exp_id,
        job_type="phase1-eval",
        config={
            "seed": EXPERIMENT_SEED,
            "max_eval_tokens": 5_000_000,
            "eval_batch_size": 2,
            "experiment_id": exp_id,
            "volume": VOLUME_NAME,
        },
    )
    tracker.init()

    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, "/root")
    from src.model import AdaptiveGPT2Model
    from src.evaluate import evaluate_both_models

    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    checkpoints_volume.reload()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    naive_model = AdaptiveGPT2Model(base_model_name="gpt2")
    naive_model.to(device=device, dtype=dtype)

    adaptive_model = AdaptiveGPT2Model(base_model_name="gpt2", max_span_len=MAX_SPAN_LEN)
    adaptive_model.to(device=device, dtype=dtype)

    naive_ckpt = "/checkpoints/naive/final.pt"
    adaptive_ckpt = "/checkpoints/adaptive/final.pt"

    if os.path.exists(naive_ckpt):
        print(f"Loading naive checkpoint from {naive_ckpt}")
        checkpoint = torch.load(naive_ckpt, map_location=device)
        naive_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"WARNING: Naive checkpoint not found at {naive_ckpt}")
        return None

    if os.path.exists(adaptive_ckpt):
        print(f"Loading adaptive checkpoint from {adaptive_ckpt}")
        checkpoint = torch.load(adaptive_ckpt, map_location=device)
        adaptive_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"WARNING: Adaptive checkpoint not found at {adaptive_ckpt}")
        return None

    results = evaluate_both_models(
        naive_model,
        adaptive_model,
        tokenizer,
        max_eval_tokens=5_000_000,
        eval_batch_size=2,
        tracker=tracker,
    )

    if tracker is not None:
        tracker.summary({
            "naive_CORE": results[0].get("CORE", 0),
            "adaptive_CORE": results[1].get("CORE", 0),
        })
    tracker.finish()
    return results


@app.function(
    gpu=f"{GPU_CONFIG}:{GPU_COUNT}",
    timeout=2 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume},
    secrets=[wandb_secret],
)
def oracle_fn():
    from src.tracker import Tracker, generate_experiment_id, set_seed
    set_seed(EXPERIMENT_SEED)
    exp_id = generate_experiment_id()
    tracker = Tracker(
        name=f"{exp_id}-oracle",
        project="adaptive-tokenization",
        group=exp_id,
        job_type="oracle",
        config={
            "seed": EXPERIMENT_SEED,
            "experiment_id": exp_id,
            "volume": VOLUME_NAME,
        },
    )
    tracker.init()

    import random
    import torch
    import numpy as np
    from transformers import AutoTokenizer
    from datasets import load_dataset
    sys.path.insert(0, "/root")
    from src.model import AdaptiveGPT2Model

    checkpoints_volume.reload()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    adaptive_model = AdaptiveGPT2Model(base_model_name="gpt2", max_span_len=MAX_SPAN_LEN)
    adaptive_model.to(device=device, dtype=dtype)
    naive_model = AdaptiveGPT2Model(base_model_name="gpt2")
    naive_model.to(device=device, dtype=dtype)

    adapt_ckpt = "/checkpoints/adaptive/final.pt"
    naive_ckpt = "/checkpoints/naive/final.pt"
    if not os.path.exists(adapt_ckpt) or not os.path.exists(naive_ckpt):
        print("ERROR: Checkpoints not found")
        return None

    adaptive_model.load_state_dict(torch.load(adapt_ckpt, map_location=device)["model_state_dict"])
    naive_model.load_state_dict(torch.load(naive_ckpt, map_location=device)["model_state_dict"])
    adaptive_model.eval()
    naive_model.eval()

    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    ds = ds.shuffle(seed=EXPERIMENT_SEED)
    for ex in ds:
        q = ex.get("question", "")
        r = ex.get("response", "")
        if q and r and len(q) > 50:
            break

    all_results = {}
    for max_prompt, label in [(96, "96tok"), (384, "384tok")]:
        prompt = tokenizer(q, add_special_tokens=True, truncation=True, max_length=max_prompt)["input_ids"]
        answer = tokenizer(r, add_special_tokens=True, truncation=True, max_length=32)["input_ids"]
        Pl, Al = len(prompt), len(answer)
        print(f"\n=== {label}: Prompt={Pl}, Answer={Al} ===")

        prompt_t = torch.tensor([prompt], dtype=torch.long).to(device)
        answer_t = torch.tensor([answer], dtype=torch.long).to(device)

        with torch.no_grad():
            naive_loss = naive_model.forward_chat_no_compress(prompt_t, answer_t).item()
        print(f"Naive no-merge baseline: {naive_loss:.4f}")

        @torch.no_grad()
        def eval_boundaries(boundaries):
            span_bnd = torch.zeros(1, Pl, dtype=torch.bool, device=device)
            span_assign = torch.zeros(1, Pl, dtype=torch.long, device=device)
            pos = 0; sid = 0
            for slen in boundaries:
                if pos >= Pl: break
                slen = min(slen, Pl - pos)
                span_bnd[0, pos] = True
                span_assign[0, pos:pos + slen] = sid
                pos += slen; sid += 1
            n_spans = int(span_bnd.sum().item())
            K = MAX_SPAN_LEN; D = adaptive_model.config.n_embd
            prompt_emb = adaptive_model.transformer.wte(prompt_t)
            span_emb = torch.zeros(1, n_spans, K, D, dtype=dtype, device=device)
            span_msk = torch.zeros(1, n_spans, K, dtype=torch.bool, device=device)
            positions = torch.arange(Pl, device=device)
            assigns = span_assign[0]; bnd_bool = span_bnd[0]
            first_pos = positions[bnd_bool]
            local_k = positions - first_pos[assigns]
            valid = local_k < K
            flat_idx = assigns * K + local_k
            span_flat = span_emb[0].view(-1, D)
            span_flat[flat_idx[valid]] = prompt_emb[0, positions[valid]]
            span_msk[0].view(-1)[flat_idx[valid]] = True
            merged = adaptive_model.span_encoder(span_emb, span_msk)
            answer_emb = adaptive_model.transformer.wte(answer_t)
            combined = torch.cat([merged, answer_emb], dim=1)
            combined_mask = torch.ones(1, combined.shape[1], dtype=torch.long, device=device)
            out = adaptive_model.transformer(inputs_embeds=combined, attention_mask=combined_mask)
            logits = adaptive_model.lm_head(out.last_hidden_state)
            labels = torch.cat([torch.full((1, n_spans), -100, dtype=torch.long, device=device), answer_t.clone()], dim=1)
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1), ignore_index=-100)
            return loss.item(), n_spans

        adapt_no_merge = eval_boundaries([1] * Pl)[0]
        print(f"Adapt no-merge baseline: {adapt_no_merge:.4f}")

        results = [(0.0, adapt_no_merge)]
        SAMPLES = 2000
        for i in range(SAMPLES):
            cr_target = random.random() * 0.9
            boundaries = []; pos = 0
            while pos < Pl:
                remaining = Pl - pos; n_so_far = len(boundaries)
                cur_cr = 1.0 - (n_so_far + remaining) / Pl if Pl > 0 else 0
                if n_so_far > 0 and cur_cr >= cr_target:
                    boundaries.append(1); pos += 1; continue
                slen = random.randint(1, min(4, remaining) + 1)
                boundaries.append(min(slen, remaining)); pos += min(slen, remaining)
            loss_val, n_sc = eval_boundaries(boundaries)
            cr_val = 1.0 - n_sc / Pl
            results.append((cr_val, loss_val))
            if (i + 1) % 500 == 0:
                print(f"  sampled {i+1}/{SAMPLES}...")

        crs = np.array([r[0] for r in results])
        losses_arr = np.array([r[1] for r in results])

        bins = np.linspace(0, 0.9, 19)
        best_crs, best_losses_list = [], []
        for j in range(len(bins) - 1):
            mask = (crs >= bins[j]) & (crs < bins[j + 1])
            if mask.sum() > 0:
                idx = mask.nonzero()[0][losses_arr[mask].argmin()]
                best_crs.append(float(crs[idx]))
                best_losses_list.append(float(losses_arr[idx]))

        print(f"Min loss: {losses_arr.min():.4f}, Max: {losses_arr.max():.4f}")
        print(f"Frontier: {list(zip([f'{c:.2f}' for c in best_crs], [f'{l:.3f}' for l in best_losses_list]))}")

        all_results[label] = {
            "crs": crs.tolist(), "losses": losses_arr.tolist(),
            "best_crs": best_crs, "best_losses": best_losses_list,
            "adapt_no_merge": adapt_no_merge, "naive_no_merge": naive_loss,
            "prompt_len": Pl, "answer_len": Al,
        }

        if tracker is not None:
            for j, (c, l) in enumerate(zip(best_crs, best_losses_list)):
                tracker.log({f"oracle/{label}/frontier_cr": c, f"oracle/{label}/frontier_loss": l}, step=j)
            tracker.summary({
                f"oracle/{label}/adapt_no_merge": adapt_no_merge,
                f"oracle/{label}/naive_no_merge": naive_loss,
                f"oracle/{label}/min_loss": float(losses_arr.min()),
                f"oracle/{label}/max_loss": float(losses_arr.max()),
            })

    tracker.finish()
    return all_results


@app.local_entrypoint()
def main():
    use_h100 = os.environ.get("USE_H100", "0") == "1"
    gpu_str = "H100" if use_h100 else GPU_CONFIG

    print("=" * 60)
    print("ADAPTIVE TOKENIZATION -- POST-TRAINING PIPELINE")
    print("=" * 60)
    print(f"GPU: {gpu_str}:{GPU_COUNT}")
    print(f"Dataset: Open-Orca/OpenOrca (streaming)")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"Prompt max: {MAX_PROMPT}, Answer max: {MAX_ANSWER}")
    print(f"Total tokens: {TOTAL_TOKENS:,}")
    print(f"Max steps: {MAX_STEPS:,}")
    print(f"Max compression: {MAX_COMPRESSION_RATIO} (curriculum)")
    print(f"Seed: {EXPERIMENT_SEED}")
    print(f"Volume: {VOLUME_NAME}")
    print("=" * 60)

    print("\n[1/3] Starting NAIVE training (standard GPT-2 fine-tuning)...")
    naive_steps = train_naive_fn.remote()
    print(f"Naive training done: {naive_steps} steps\n")

    print("\n[2/3] Starting ADAPTIVE training (span encoder + merging)...")
    adaptive_steps = train_adaptive_fn.remote()
    print(f"Adaptive training done: {adaptive_steps} steps\n")

    print("\n[3/3] Running evaluation on both models...")
    results = evaluate_fn.remote()
    print("Evaluation complete!\n")
    print("=" * 60)
    if results:
        print("RESULTS SUMMARY:")
        print(results)
    else:
        print("Evaluation returned no results (checkpoints not found).")
    print("=" * 60)
