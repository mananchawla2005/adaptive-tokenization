import os
import sys
import modal
from pathlib import Path


GPU_CONFIG = "A100-80GB"
GPU_COUNT = 1
TIMEOUT = 6 * 60 * 60
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

app = modal.App("adaptive-tokenization-phase2", image=image)

checkpoints_volume = modal.Volume.from_name(
    VOLUME_NAME, create_if_missing=True
)

wandb_secret = modal.Secret.from_dict({
    "WANDB_API_KEY": ""
})


@app.function(
    gpu=f"{GPU_CONFIG}:{GPU_COUNT}",
    timeout=TIMEOUT,
    volumes={"/checkpoints": checkpoints_volume},
    secrets=[wandb_secret],
    retries=modal.Retries(initial_delay=0.0, max_retries=1, backoff_coefficient=1.0),
)
def train_hybrid_fn():
    from src.tracker import Tracker, generate_experiment_id, set_seed
    set_seed(EXPERIMENT_SEED)
    exp_id = generate_experiment_id()

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

    # Stage 1: Create oracle dataset (skip if already exists)
    print("\n" + "=" * 60)
    print("STAGE 1: Creating oracle dataset (K=128 per prompt)")
    print("=" * 60)
    oracle_path = "/checkpoints/phase2/oracle_dataset.pt"
    if os.path.exists(oracle_path):
        print("Oracle dataset already exists, skipping Stage 1.")
    else:
        tracker1 = Tracker(
            name=f"{exp_id}-phase2-stage1-oracle",
            project="adaptive-tokenization", group=exp_id,
            job_type="phase2-stage1", config={"stage": 1, "experiment_id": exp_id},
        )
        tracker1.init()
        stage1_create_oracle(
            adaptive_model, tokenizer, output_dir="/checkpoints/phase2",
            volume=checkpoints_volume, num_prompts=1000, k_samples=128,
            tracker=tracker1,
        )
        tracker1.finish()

    # Stage 2: Supervised imitation
    print("\n" + "=" * 60)
    print("STAGE 2: Supervised imitation (BCE)")
    print("=" * 60)
    tracker2 = Tracker(
        name=f"{exp_id}-phase2-stage2-bce",
        project="adaptive-tokenization", group=exp_id,
        job_type="phase2-stage2", config={"stage": 2, "experiment_id": exp_id},
    )
    tracker2.init()
    predictor = stage2_train_imitation(
        adaptive_model, tokenizer, output_dir="/checkpoints/phase2",
        volume=checkpoints_volume, epochs=5, batch_size=32,
        tracker=tracker2,
    )
    tracker2.finish()

    if predictor is None:
        return {"error": "Stage 2 failed"}

    # Stage 3: GRPO fine-tuning
    print("\n" + "=" * 60)
    print("STAGE 3: GRPO fine-tuning")
    print("=" * 60)
    tracker3 = Tracker(
        name=f"{exp_id}-phase2-stage3-grpo",
        project="adaptive-tokenization", group=exp_id,
        job_type="phase2-stage3", config={"stage": 3, "experiment_id": exp_id},
    )
    tracker3.init()
    predictor = stage3_train_grpo(
        adaptive_model, tokenizer, predictor, output_dir="/checkpoints/phase2",
        volume=checkpoints_volume, max_steps=2000,
        tracker=tracker3,
    )
    tracker3.finish()

    return {"status": "complete"}


@app.function(
    gpu=f"{GPU_CONFIG}:{GPU_COUNT}",
    timeout=2 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume},
    secrets=[wandb_secret],
)
def evaluate_hybrid_fn():
    from src.tracker import Tracker, generate_experiment_id, set_seed
    set_seed(EXPERIMENT_SEED)
    exp_id = generate_experiment_id()
    tracker = Tracker(
        name=f"{exp_id}-phase2-eval",
        project="adaptive-tokenization",
        group=exp_id,
        job_type="phase2-eval",
        config={
            "seed": EXPERIMENT_SEED,
            "experiment_id": exp_id,
            "volume": VOLUME_NAME,
        },
    )
    tracker.init()

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

    # Load evaluation prompts from held-out shard (skip first 1M training examples)
    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    ds = ds.skip(1_000_000)  # fix #3: val split via deterministic skip
    ds = ds.shuffle(seed=EXPERIMENT_SEED)
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
    prompt_m = torch.zeros(64, max_p, dtype=torch.long, device=device)
    answer_m = torch.zeros(64, max_a, dtype=torch.long, device=device)
    for i in range(64):
        pl = len(prompts[i]); al = len(answers[i])
        prompt_t[i, :pl] = torch.tensor(prompts[i])
        answer_t[i, :al] = torch.tensor(answers[i])
        prompt_m[i, :pl] = 1
        answer_m[i, :al] = 1

    def _decode_merge(predictor, label):
        """Decode how the predictor merges the first 10 prompts."""
        import wandb
        table = wandb.Table(columns=["idx", "prompt", f"{label}_spans", f"{label}_cr"])
        for i in range(min(10, len(prompts))):
            pl = len(prompts[i])
            single = prompt_t[i:i+1]  # full padded length (same as training)
            single_mask = prompt_m[i:i+1]
            with torch.no_grad():
                bnd = sample_boundaries_grpo(predictor, single, attention_mask=single_mask,
                                             num_samples=1, max_span_len=4)
            boundaries = bnd[0, 0, :pl].cpu()  # only real token positions
            # Build spans: group tokens between boundary=True positions
            token_ids = prompts[i]
            spans = []
            start = 0
            for pos in range(1, len(token_ids)):
                if boundaries[pos]:
                    span_text = tokenizer.decode(token_ids[start:pos])
                    spans.append(span_text)
                    start = pos
            span_text = tokenizer.decode(token_ids[start:])
            spans.append(span_text)
            n_spans = int(boundaries.sum().item())
            cr = 1.0 - n_spans / pl
            cr_str = f"{cr*100:.0f}% ({n_spans} spans)"
            table.add_data(
                i, tokenizer.decode(token_ids),
                " | ".join(spans), cr_str,
            )
        return table

    def _eval_predictor(ckpt, label):
        p = BoundaryPredictor(embed_weight=embed_weight, hidden_dim=512, num_layers=2, num_heads=8)
        p.to(device=device)
        ckpt_path = f"/checkpoints/phase2/{ckpt}"
        if not os.path.exists(ckpt_path):
            print(f"WARNING: {ckpt} not found, skipping {label}")
            return None
        p.load_state_dict(torch.load(ckpt_path, map_location=device)["predictor_state_dict"])
        p.eval()
        with torch.no_grad():
            bnd = sample_boundaries_grpo(p, prompt_t, attention_mask=prompt_m,
                                         num_samples=1, max_span_len=4)
            loss = evaluate_boundaries_batch(adaptive_model, prompt_t, answer_t, bnd, 4, answer_mask=answer_m)
            loss = loss.mean().item()
            real_spans = (bnd & prompt_m.unsqueeze(0)).float().sum(dim=-1).mean().item()
            real_tokens = prompt_m.float().sum(dim=-1).mean().item()
            cr = 1.0 - real_spans / real_tokens
        core_val = loss / max(no_merge_loss, 1e-8)
        print(f"[{label}] loss={loss:.4f}, cr={cr:.2f}, CORE={core_val:.4f}")
        return {"loss": loss, "cr": cr, "CORE": core_val}

    with torch.no_grad():
        no_merge_loss = adaptive_model.forward_chat_no_compress(
            prompt_t, answer_t, prompt_mask=prompt_m, answer_mask=answer_m).item()

    with torch.no_grad():
        random_bnd = sample_random_boundaries(prompt_t, 1, max_span_len=4, prompt_mask=prompt_m)
        random_loss = evaluate_boundaries_batch(adaptive_model, prompt_t, answer_t, random_bnd, 4, answer_mask=answer_m)
        random_loss = random_loss.mean().item()
        real_spans = (random_bnd & prompt_m.unsqueeze(0)).float().sum(dim=-1).mean().item()
        real_tokens = prompt_m.float().sum(dim=-1).mean().item()
        random_cr = 1.0 - real_spans / real_tokens
    random_core = random_loss / max(no_merge_loss, 1e-8)

    bce_results = _eval_predictor("stage2_imitation.pt", "BCE-only")
    grpo_results = _eval_predictor("final_predictor.pt", "BCE+GRPO")

    print("\n=== Hybrid Phase 2 Evaluation ===")
    print(f"No-merge baseline:       loss={no_merge_loss:.4f}")
    print(f"Random merge (cr={random_cr:.2f}):  loss={random_loss:.4f}, CORE={random_core:.4f}")
    if bce_results:
        print(f"BCE-only  (cr={bce_results['cr']:.2f}): loss={bce_results['loss']:.4f}, CORE={bce_results['CORE']:.4f}")
    if grpo_results:
        print(f"BCE+GRPO  (cr={grpo_results['cr']:.2f}): loss={grpo_results['loss']:.4f}, CORE={grpo_results['CORE']:.4f}")

    if tracker is not None:
        log_data = {
            "eval/no_merge_loss": no_merge_loss,
            "eval/random_loss": random_loss, "eval/random_cr": random_cr, "eval/random_CORE": random_core,
        }
        if bce_results:
            log_data.update({
                "eval/bce_loss": bce_results["loss"], "eval/bce_cr": bce_results["cr"], "eval/bce_CORE": bce_results["CORE"],
            })
        if grpo_results:
            log_data.update({
                "eval/grpo_loss": grpo_results["loss"], "eval/grpo_cr": grpo_results["cr"], "eval/grpo_CORE": grpo_results["CORE"],
            })
        tracker.log(log_data)
        tracker.summary({
            "no_merge_loss": no_merge_loss,
            "bce_CORE": bce_results["CORE"] if bce_results else None,
            "grpo_CORE": grpo_results["CORE"] if grpo_results else None,
            "random_CORE": random_core,
        })
        # Log merge visualization tables for first 10 prompts
        import wandb as wb_mod
        bce_p = BoundaryPredictor(embed_weight=embed_weight, hidden_dim=512, num_layers=2, num_heads=8)
        bce_p.to(device=device)
        bce_p.load_state_dict(torch.load("/checkpoints/phase2/stage2_imitation.pt", map_location=device)["predictor_state_dict"])
        bce_p.eval()
        grpo_p = BoundaryPredictor(embed_weight=embed_weight, hidden_dim=512, num_layers=2, num_heads=8)
        grpo_p.to(device=device)
        grpo_p.load_state_dict(torch.load("/checkpoints/phase2/final_predictor.pt", map_location=device)["predictor_state_dict"])
        grpo_p.eval()
        bce_table = _decode_merge(bce_p, "BCE-only")
        grpo_table = _decode_merge(grpo_p, "BCE+GRPO")
        tracker.log({"merge_bce_samples": bce_table, "merge_grpo_samples": grpo_table})

    tracker.finish()
    return {
        "no_merge_loss": no_merge_loss,
        "random_loss": random_loss, "random_cr": random_cr, "random_CORE": random_core,
        "bce": bce_results, "grpo": grpo_results,
    }
