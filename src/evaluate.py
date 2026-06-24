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
    tracker=None,
    prefix="eval",
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

        total_nats = 0.0
        total_tokens = 0
        total_bytes_val = 0
        pbar = tqdm(dataloader, desc=f"Eval [{model_name}] {scenario}")

        for batch_idx, batch in enumerate(pbar):
            prompt_ids = batch["prompt_ids"].to(device)
            answer_ids = batch["answer_ids"].to(device)
            pm = batch.get("prompt_mask")
            am = batch.get("answer_mask")

            with torch.amp.autocast("cuda" if use_amp else "cpu", dtype=amp_dtype):
                if scenario == "no_merge":
                    loss = model.forward_chat_no_compress(
                        prompt_ids, answer_ids,
                        prompt_mask=pm.to(device) if pm is not None else None,
                        answer_mask=am.to(device) if am is not None else None,
                    )
                elif scenario == "random_merge":
                    rng = random.Random(batch_idx)
                    loss = model.forward_chat(
                        prompt_ids, answer_ids,
                        tokenizer=tokenizer,
                        compression_ratio=0.5,
                        rng=rng,
                        prompt_mask=pm.to(device) if pm is not None else None,
                        answer_mask=am.to(device) if am is not None else None,
                    )
                else:
                    raise ValueError(f"Unknown scenario: {scenario}")

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            if am is not None:
                num_tokens = am.sum().item()
            else:
                num_tokens = (answer_ids != 0).sum().item()

            total_nats += loss.item() * num_tokens  # fix #6: nats = per-token-loss * tokens
            total_tokens += num_tokens

            # fix #4: count decoded bytes for true bits-per-byte
            for b in range(answer_ids.shape[0]):
                real_len = int(am[b].sum().item()) if am is not None else (answer_ids[b] != 0).sum().item()
                if real_len > 0:
                    decoded = tokenizer.decode(answer_ids[b, :real_len].tolist(), skip_special_tokens=True)
                    total_bytes_val += len(decoded.encode("utf-8"))

            if total_tokens >= max_eval_tokens:
                break

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_nats / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))
        bpb = total_nats / (max(total_bytes_val, 1) * math.log(2))

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

    if tracker is not None:
        log_data = {}
        for scenario, metrics in results.items():
            if scenario == "CORE":
                log_data[f"{prefix}/{model_name}/CORE"] = metrics
            else:
                for metric, value in metrics.items():
                    log_data[f"{prefix}/{model_name}/{scenario}/{metric}"] = value
        tracker.log(log_data)

    return results


@torch.no_grad()
def evaluate_both_models(
    naive_model,
    adaptive_model,
    tokenizer,
    max_eval_tokens=5_000_000,
    eval_batch_size=2,
    tracker=None,
):
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    naive_results = evaluate_model(
        naive_model, tokenizer, model_name="Naive",
        max_eval_tokens=max_eval_tokens, eval_batch_size=eval_batch_size,
        tracker=tracker,
    )

    adaptive_results = evaluate_model(
        adaptive_model, tokenizer, model_name="Adaptive",
        max_eval_tokens=max_eval_tokens, eval_batch_size=eval_batch_size,
        tracker=tracker,
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
