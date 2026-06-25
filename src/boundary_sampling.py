import torch
import torch.nn.functional as F

# Biases used during GRPO sampling to create diverse boundary sets
GRPO_BIASES = [-0.5, -0.15, 0.15, 0.5]


def get_predictor_logits(predictor, input_ids, attention_mask=None):
    return predictor.forward(input_ids, attention_mask)


@torch.no_grad()
def sample_boundaries_grpo(predictor, input_ids, attention_mask=None, num_samples=4, max_span_len=4, epsilon=0.0):
    """Sample boundary configurations from predictor. @no_grad is required because
    the sequential sampling loop is non-differentiable.

    Returns boundaries tensor (S, B, L) with True = start of new span.
    """
    B, L = input_ids.shape
    device = input_ids.device

    logits = predictor.forward(input_ids, attention_mask)

    if num_samples == 1:
        biases = torch.tensor([0.0], device=device)
    else:
        biases = torch.tensor(GRPO_BIASES, device=device)

    boundaries = torch.zeros(num_samples, B, L, dtype=torch.bool, device=device)

    for k in range(min(num_samples, len(biases))):
        bias = biases[k]
        biased_logits = logits + bias
        p_boundary = torch.sigmoid(biased_logits)

        if epsilon > 0:
            noise_mask = torch.rand(B, L, device=device) < epsilon
            p_boundary = torch.where(noise_mask, torch.full_like(p_boundary, 0.5), p_boundary)

        if attention_mask is not None:
            p_boundary = p_boundary * attention_mask.float()  # padding -> 0 -> always merge
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


def compute_grpo_log_probs(predictor, input_ids, boundaries, attention_mask=None, max_span_len=4):
    """Compute per-position log-probs and valid-action mask for GRPO.
    Must be called OUTSIDE torch.no_grad() so predictor gradients flow.

    Returns:
      log_probs: (S, B, L) per-position log-prob of sampled action
      valid_mask: (S, B, L) True for positions that contribute to policy gradient.
      logit_reg: scalar L1 penalty on logits to prevent saturation (logits → ±∞)
    """
    S, B, L = boundaries.shape
    device = input_ids.device

    logits = predictor.forward(input_ids, attention_mask)
    logit_reg = logits.abs().mean()  # prevent saturation to ±∞
    if S == 1:
        biases = torch.tensor([0.0], device=device)
    else:
        biases = torch.tensor(GRPO_BIASES[:S], device=device)

    log_probs = torch.zeros(S, B, L, device=device)
    valid_mask = torch.zeros(S, B, L, dtype=torch.bool, device=device)

    for k in range(S):
        bias = biases[k]
        biased_logits = logits + bias
        p_boundary = torch.sigmoid(biased_logits)

        if attention_mask is not None:
            p_boundary = p_boundary * attention_mask.float()
        p_boundary[:, 0] = 1.0

        bnd_k = boundaries[k].float()
        merge_k = 1.0 - bnd_k

        log_p_bound = torch.log(p_boundary.clamp_min(1e-8))
        log_p_merge = torch.log((1.0 - p_boundary).clamp_min(1e-8))

        per_pos = bnd_k * log_p_bound + merge_k * log_p_merge
        log_probs[k] = per_pos

        # Reconstruct which positions were deterministic (forced by max_span_len)
        merged_count = torch.zeros(B, L, dtype=torch.long, device=device)
        valid = torch.ones(B, L, dtype=torch.bool, device=device)
        if attention_mask is not None:
            valid = attention_mask.bool()
        valid[:, 0] = False  # position 0 is always forced

        for pos in range(1, L):
            is_forced = merged_count[:, pos - 1] >= max_span_len - 1
            valid[:, pos] = valid[:, pos] & (~is_forced)
            merged_count[:, pos] = torch.where(
                boundaries[k, :, pos],
                torch.zeros(B, dtype=torch.long, device=device),
                merged_count[:, pos - 1] + 1,
            )

        valid_mask[k] = valid

    return log_probs, valid_mask, logit_reg


def boundaries_to_log_probs(predictor, input_ids, boundaries, attention_mask=None):
    """Compute log-probs of given boundaries under the UNBIASED predictor.
    Used for inference/eval, not for GRPO training (use compute_grpo_log_probs).
    """
    S, B, L = boundaries.shape
    device = input_ids.device

    p_boundary = predictor.get_boundary_probs(input_ids, attention_mask)
    p_boundary = p_boundary.unsqueeze(0).expand(S, -1, -1)

    target_boundary = boundaries.float()

    log_probs = target_boundary * torch.log(p_boundary.clamp_min(1e-8)) + \
                (1.0 - target_boundary) * torch.log((1.0 - p_boundary).clamp_min(1e-8))

    if attention_mask is not None:
        mask = attention_mask.unsqueeze(0).float()
        mask[:, :, 0] = 0.0
        log_probs = log_probs * mask
        norm = mask.sum(dim=-1).clamp_min(1)
    else:
        norm = L

    return log_probs.sum(dim=-1) / norm
