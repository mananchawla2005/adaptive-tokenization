import os
import sys
import modal
from pathlib import Path


GPU_CONFIG = "A100-80GB"
GPU_COUNT = 1
TIMEOUT = 6 * 60 * 60

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
    )
    .add_local_dir(SOURCE_DIR / "src", remote_path="/root/src", copy=True)
)

app = modal.App("adaptive-tokenization-phase2", image=image)

checkpoints_volume = modal.Volume.from_name(
    "adaptive-tok-checkpoints", create_if_missing=True
)


@app.function(
    gpu=f"{GPU_CONFIG}:{GPU_COUNT}",
    timeout=TIMEOUT,
    volumes={"/checkpoints": checkpoints_volume},
    retries=modal.Retries(initial_delay=0.0, max_retries=1, backoff_coefficient=1.0),
)
def train_hybrid_fn():
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, "/root")
    from src.model import AdaptiveGPT2Model
    from src.hybrid_train import (
        stage1_create_oracle, stage2_train_imitation, stage3_train_grpo,
    )

    checkpoints_volume.reload()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading frozen Adaptive model...")
    adaptive_model = AdaptiveGPT2Model(base_model_name="gpt2", max_span_len=4)
    adaptive_model.to(device=device, dtype=dtype)
    adaptive_model.load_state_dict(
        torch.load("/checkpoints/adaptive/final.pt", map_location=device)["model_state_dict"]
    )
    adaptive_model.eval()
    for p in adaptive_model.parameters():
        p.requires_grad = False

    # Stage 1: Create oracle dataset
    print("\n" + "=" * 60)
    print("STAGE 1: Creating oracle dataset (K=128 per prompt)")
    print("=" * 60)
    stage1_create_oracle(
        adaptive_model, tokenizer, output_dir="/checkpoints/phase2",
        volume=checkpoints_volume, num_prompts=1000, k_samples=128,
    )

    # Stage 2: Supervised imitation
    print("\n" + "=" * 60)
    print("STAGE 2: Supervised imitation (BCE)")
    print("=" * 60)
    predictor = stage2_train_imitation(
        adaptive_model, tokenizer, output_dir="/checkpoints/phase2",
        volume=checkpoints_volume, epochs=5, batch_size=32,
    )

    if predictor is None:
        return {"error": "Stage 2 failed"}

    # Stage 3: GRPO fine-tuning
    print("\n" + "=" * 60)
    print("STAGE 3: GRPO fine-tuning")
    print("=" * 60)
    predictor = stage3_train_grpo(
        adaptive_model, tokenizer, predictor, output_dir="/checkpoints/phase2",
        volume=checkpoints_volume, max_steps=2000,
    )

    return {"status": "complete"}


@app.function(
    gpu=f"{GPU_CONFIG}:{GPU_COUNT}",
    timeout=2 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume},
)
def evaluate_hybrid_fn():
    import random
    import torch
    from transformers import AutoTokenizer
    from datasets import load_dataset
    sys.path.insert(0, "/root")
    from src.model import AdaptiveGPT2Model
    from src.boundary_predictor import BoundaryPredictor
    from src.boundary_sampling import sample_boundaries_grpo
    from src.hybrid_train import evaluate_boundaries_batch, sample_random_boundaries

    checkpoints_volume.reload()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    adaptive_model = AdaptiveGPT2Model(base_model_name="gpt2", max_span_len=4)
    adaptive_model.to(device=device, dtype=dtype)
    adaptive_model.load_state_dict(
        torch.load("/checkpoints/adaptive/final.pt", map_location=device)["model_state_dict"]
    )
    adaptive_model.eval()
    for p in adaptive_model.parameters():
        p.requires_grad = False

    embed_weight = adaptive_model.transformer.wte.weight.detach()
    predictor = BoundaryPredictor(embed_weight=embed_weight, hidden_dim=256, num_layers=1, num_heads=4)
    predictor.to(device=device)

    ckpt_path = "/checkpoints/phase2/final_predictor.pt"
    if not os.path.exists(ckpt_path):
        print("ERROR: Predictor checkpoint not found")
        return None
    predictor.load_state_dict(torch.load(ckpt_path, map_location=device)["predictor_state_dict"])
    predictor.eval()

    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    ds = ds.shuffle(seed=42)
    prompts, answers = [], []
    for ex in ds:
        q, r = ex.get("question", ""), ex.get("response", "")
        if q and r and len(q) > 50:
            p = tokenizer(q, add_special_tokens=True, truncation=True, max_length=384)["input_ids"]
            a = tokenizer(r, add_special_tokens=True, truncation=True, max_length=128)["input_ids"]
            prompts.append(p); answers.append(a)
            if len(prompts) >= 64:
                break

    max_p = max(len(p) for p in prompts)
    max_a = max(len(a) for a in answers)
    prompt_t = torch.full((64, max_p), 0, dtype=torch.long).to(device)
    answer_t = torch.full((64, max_a), 0, dtype=torch.long).to(device)
    for i in range(64):
        prompt_t[i, :len(prompts[i])] = torch.tensor(prompts[i])
        answer_t[i, :len(answers[i])] = torch.tensor(answers[i])

    with torch.no_grad():
        no_merge_loss = adaptive_model.forward_chat_no_compress(prompt_t, answer_t).item()

    with torch.no_grad():
        random_bnd = sample_random_boundaries(prompt_t, 1, max_span_len=4)
        random_loss = evaluate_boundaries_batch(adaptive_model, prompt_t, answer_t, random_bnd, 4)
        random_loss = random_loss.mean().item()
        random_cr = 1.0 - random_bnd.float().sum(dim=-1).mean().item() / max_p

    with torch.no_grad():
        learned_bnd = sample_boundaries_grpo(predictor, prompt_t, num_samples=1, max_span_len=4)
        learned_loss = evaluate_boundaries_batch(adaptive_model, prompt_t, answer_t, learned_bnd, 4)
        learned_loss = learned_loss.mean().item()
        learned_cr = 1.0 - learned_bnd.float().sum(dim=-1).mean().item() / max_p

    print("\n=== Hybrid Phase 2 Evaluation ===")
    print(f"No-merge baseline:       loss={no_merge_loss:.4f}")
    print(f"Random merge (cr={random_cr:.2f}):  loss={random_loss:.4f}")
    print(f"Learned merge (cr={learned_cr:.2f}): loss={learned_loss:.4f}")

    return {
        "no_merge_loss": no_merge_loss,
        "random_loss": random_loss, "random_cr": random_cr,
        "learned_loss": learned_loss, "learned_cr": learned_cr,
    }
