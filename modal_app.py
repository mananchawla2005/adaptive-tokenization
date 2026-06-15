import os
import sys
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

app = modal.App("adaptive-tokenization", image=image)

checkpoints_volume = modal.Volume.from_name(
    "adaptive-tok-checkpoints", create_if_missing=True
)


@app.function(
    gpu=f"{GPU_CONFIG}:{GPU_COUNT}",
    timeout=TRAINING_TIMEOUT,
    volumes={"/checkpoints": checkpoints_volume},
    retries=modal.Retries(
        initial_delay=0.0,
        max_retries=3,
        backoff_coefficient=1.0,
    ),
)
def train_naive_fn():
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
    )
    print(f"Naive training completed: {steps} steps")
    return steps


@app.function(
    gpu=f"{GPU_CONFIG}:{GPU_COUNT}",
    timeout=TRAINING_TIMEOUT,
    volumes={"/checkpoints": checkpoints_volume},
    retries=modal.Retries(
        initial_delay=0.0,
        max_retries=3,
        backoff_coefficient=1.0,
    ),
)
def train_adaptive_fn():
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
    )
    print(f"Adaptive training completed: {steps} steps")
    return steps


@app.function(
    gpu=f"{GPU_CONFIG}:{GPU_COUNT}",
    timeout=3 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume},
)
def evaluate_fn():
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
    )

    return results


@app.local_entrypoint()
def main():
    use_h100 = os.environ.get("USE_H100", "0") == "1"
    gpu_str = "H100" if use_h100 else GPU_CONFIG

    print("=" * 60)
    print("ADAPTIVE TOKENIZATION — POST-TRAINING PIPELINE")
    print("=" * 60)
    print(f"GPU: {gpu_str}:{GPU_COUNT}")
    print(f"Dataset: Open-Orca/OpenOrca (streaming)")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"Prompt max: {MAX_PROMPT}, Answer max: {MAX_ANSWER}")
    print(f"Total tokens: {TOTAL_TOKENS:,}")
    print(f"Max steps: {MAX_STEPS:,}")
    print(f"Max compression: {MAX_COMPRESSION_RATIO} (curriculum)")
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
