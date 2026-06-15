import math
import random
import torch
import torch.nn as nn
from tqdm import tqdm

from .data import create_eval_dataloader
from .model import AdaptiveGPT2Model


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    model_name="Naive",
    max_eval_tokens=5_000_000,
    eval_batch_size=2,
    eval_scenarios=None,
):
    if eval_scenarios is None:
        eval_scenarios = ["no_merge", "random_merge"]

    device = next(model.parameters()).device
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    model.eval()

    results = {}

    for scenario in eval_scenarios:
        dataloader = create_eval_dataloader(
            tokenizer, batch_size=eval_batch_size, max_tokens=max_eval_tokens
        )

        total_loss = 0.0
        total_tokens = 0
        pbar = tqdm(dataloader, desc=f"Eval [{model_name}] {scenario}")

        for batch_idx, batch in enumerate(pbar):
            prompt_ids = batch["prompt_ids"].to(device)
            answer_ids = batch["answer_ids"].to(device)

            with torch.amp.autocast("cuda" if use_amp else "cpu", dtype=amp_dtype):
                if scenario == "no_merge":
                    loss = model.forward_chat_no_compress(prompt_ids, answer_ids)
                elif scenario == "random_merge":
                    rng = random.Random(batch_idx)
                    loss = model.forward_chat(
                        prompt_ids, answer_ids,
                        tokenizer=tokenizer,
                        compression_ratio=0.5,
                        rng=rng,
                    )
                else:
                    raise ValueError(f"Unknown scenario: {scenario}")

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            num_tokens = (answer_ids != 0).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            if total_tokens >= max_eval_tokens:
                break

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))
        bpb = avg_loss / math.log(2)

        results[scenario] = {
            "loss": avg_loss,
            "ppl": ppl,
            "bpb": bpb,
        }

        print(f"\n[{model_name}] {scenario}: loss={avg_loss:.4f}, ppl={ppl:.2f}, bpb={bpb:.4f}")

    if "no_merge" in results and "random_merge" in results:
        nm_loss = results["no_merge"]["loss"]
        rm_loss = results["random_merge"]["loss"]
        core_score = rm_loss / max(nm_loss, 1e-8)
        results["CORE"] = core_score
        print(f"[{model_name}] CORE (merged_loss / no_merge_loss): {core_score:.4f}")

    return results


@torch.no_grad()
def evaluate_both_models(
    naive_model,
    adaptive_model,
    tokenizer,
    max_eval_tokens=5_000_000,
    eval_batch_size=2,
):
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    naive_results = evaluate_model(
        naive_model, tokenizer, model_name="Naive",
        max_eval_tokens=max_eval_tokens, eval_batch_size=eval_batch_size,
    )

    adaptive_results = evaluate_model(
        adaptive_model, tokenizer, model_name="Adaptive",
        max_eval_tokens=max_eval_tokens, eval_batch_size=eval_batch_size,
    )

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<15} {'Naive':<15} {'Adaptive':<15}")
    print("-" * 60)

    for scenario in ["no_merge", "random_merge"]:
        for metric in ["loss", "ppl", "bpb"]:
            naive_val = naive_results[scenario][metric]
            adapt_val = adaptive_results[scenario][metric]
            print(f"{scenario[:6]}_{metric:<8} {naive_val:<15.4f} {adapt_val:<15.4f}")

    print(f"\n{'CORE':<15} {naive_results['CORE']:<15.4f} {adaptive_results['CORE']:<15.4f}")

    return naive_results, adaptive_results
