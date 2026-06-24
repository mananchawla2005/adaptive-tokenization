import torch
import torch.nn.functional as F


def get_predictor_logits(predictor, input_ids, attention_mask=None):
    return predictor.forward(input_ids, attention_mask)


@torch.no_grad()
def sample_boundaries_grpo(predictor, input_ids, attention_mask=None, num_samples=4, max_span_len=4, epsilon=0.0, return_log_probs=False):
    """Sample boundary configurations from predictor.

    Fix #7: When return_log_probs=True, returns log_probs computed from the SAME
    biased/perturbed distribution used for sampling, ensuring on-policy GRPO.
    """
    B, L = input_ids.shape
    device = input_ids.device

    logits = predictor.forward(input_ids, attention_mask)

    if num_samples == 1:
        biases = torch.tensor([0.0], device=device)
    else:
        biases = torch.tensor([-0.5, -0.15, 0.15, 0.5], device=device)

    boundaries = torch.zeros(num_samples, B, L, dtype=torch.bool, device=device)
    sample_log_probs = torch.zeros(num_samples, B, L, device=device) if return_log_probs else None

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

        # Track log-probs of sampled actions (on-policy with biased/noised probs)
        if return_log_probs:
            log_p_bound = torch.log(p_boundary.clamp_min(1e-8))
            log_p_merge = torch.log((1.0 - p_boundary).clamp_min(1e-8))

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

            if return_log_probs:
                # Forced boundaries (is_forced=True) are deterministic — exclude from log-prob
                forced_mask = ~is_forced
                sample_log_probs[k, :, pos] = torch.where(
                    new_span & forced_mask, log_p_bound[:, pos],
                    torch.where((~new_span) & forced_mask, log_p_merge[:, pos],
                                torch.zeros_like(log_p_bound[:, pos]))
                )

    if return_log_probs:
        # Mask and normalize log-probs (same as boundaries_to_log_probs logic)
        if attention_mask is not None:
            mask = attention_mask.float()
            mask[:, 0] = 0.0  # exclude forced first boundary
            sample_log_probs = sample_log_probs * mask.unsqueeze(0)
            norm = mask.sum(dim=-1).clamp_min(1).unsqueeze(0)
        else:
            norm = L
        return boundaries, sample_log_probs.sum(dim=-1) / norm

    return boundaries


def boundaries_to_log_probs(predictor, input_ids, boundaries, attention_mask=None):
    """Compute log-probs of given boundaries under the UNBIASED predictor.
    This is used for inference/eval, not for GRPO training.
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
        mask[:, :, 0] = 0.0  # exclude forced first boundary from log-prob
        log_probs = log_probs * mask
        norm = mask.sum(dim=-1).clamp_min(1)
    else:
        norm = L

    return log_probs.sum(dim=-1) / norm
