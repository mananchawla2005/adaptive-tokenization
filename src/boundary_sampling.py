import torch
import torch.nn.functional as F


def get_predictor_logits(predictor, input_ids, attention_mask=None):
    return predictor.forward(input_ids, attention_mask)


@torch.no_grad()
def sample_boundaries_grpo(predictor, input_ids, attention_mask=None, num_samples=4, max_span_len=4, epsilon=0.05):
    B, L = input_ids.shape
    device = input_ids.device

    logits = predictor.forward(input_ids, attention_mask)

    if num_samples == 1:
        biases = torch.tensor([0.0], device=device)
    else:
        biases = torch.tensor([-0.5, -0.15, 0.15, 0.5], device=device)

    boundaries = torch.zeros(num_samples, B, L, dtype=torch.bool, device=device)

    for k in range(min(num_samples, len(biases))):
        bias = biases[k]
        biased_logits = logits + bias
        p_boundary = torch.sigmoid(biased_logits)

        if epsilon > 0:
            noise_mask = torch.rand(B, L, device=device) < epsilon
            p_boundary = torch.where(noise_mask, torch.full_like(p_boundary, 0.5), p_boundary)

        if attention_mask is not None:
            p_boundary = p_boundary * attention_mask.float()  # padding → 0 → always merge
        p_boundary[:, 0] = 1.0

        merged_count = torch.zeros(B, L, dtype=torch.long, device=device)

        for pos in range(L):
            if pos == 0:
                boundaries[k, :, 0] = True
                continue

            p_m = 1.0 - p_boundary[:, pos]

            is_forced = merged_count[:, pos - 1] >= max_span_len - 1

            rand = torch.rand(B, device=device)
            should_merge = rand < p_m
            new_span = (~should_merge) | is_forced

            boundaries[k, :, pos] = new_span
            merged_count[:, pos] = torch.where(
                new_span,
                torch.zeros(B, dtype=torch.long, device=device),
                merged_count[:, pos - 1] + 1,
            )

    return boundaries


def boundaries_to_log_probs(predictor, input_ids, boundaries, attention_mask=None):
    S, B, L = boundaries.shape
    device = input_ids.device

    p_boundary = predictor.get_boundary_probs(input_ids, attention_mask)
    p_boundary = p_boundary.unsqueeze(0).expand(S, -1, -1)

    target_boundary = boundaries.float()

    log_probs = target_boundary * torch.log(p_boundary.clamp_min(1e-8)) + \
                (1.0 - target_boundary) * torch.log((1.0 - p_boundary).clamp_min(1e-8))

    if attention_mask is not None:
        mask = attention_mask.unsqueeze(0).float()
        mask[:, :, 0] = 0.0  # exclude forced first boundary from log-prob
        log_probs = log_probs * mask
        norm = mask.sum(dim=-1).clamp_min(1)
    else:
        norm = L

    return log_probs.sum(dim=-1) / norm
