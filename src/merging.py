import torch
import random


def build_span_masks_random(
    input_ids,
    max_span_len=4,
    max_compression_ratio=0.5,
    rng=None,
):
    B, L = input_ids.shape
    device = input_ids.device

    if rng is None:
        rng = random.Random()

    span_boundaries = torch.zeros(B, L, dtype=torch.bool, device=device)
    span_assignments = torch.zeros(B, L, dtype=torch.long, device=device)

    for b in range(B):
        pos = 0
        sid = 0
        while pos < L:
            if sid > 0:
                current_cr = 1.0 - (sid + (L - pos)) / L
                if current_cr >= max_compression_ratio:
                    span_boundaries[b, pos] = True
                    span_assignments[b, pos] = sid
                    pos += 1
                    sid += 1
                    continue

            span_len = rng.randint(1, min(max_span_len, L - pos))
            span_len = min(span_len, L - pos)

            span_boundaries[b, pos] = True
            span_assignments[b, pos : pos + span_len] = sid
            pos += span_len
            sid += 1

    return span_boundaries, span_assignments


def build_span_masks_with_heuristics(
    input_ids,
    tokenizer=None,
    max_span_len=4,
    max_compression_ratio=0.5,
    rng=None,
):
    return build_span_masks_random(
        input_ids,
        max_span_len=max_span_len,
        max_compression_ratio=max_compression_ratio,
        rng=rng,
    )


def group_embeddings_by_spans(
    hidden_states,
    input_ids,
    span_boundaries,
    span_assignments,
    max_span_len=4,
):
    B, L, D = hidden_states.shape
    device = hidden_states.device

    num_spans = span_boundaries.sum(dim=1)
    max_spans = int(num_spans.max().item())
    if max_spans == 0:
        return (
            hidden_states.new_zeros(B, 1, max_span_len, D),
            torch.zeros(B, 1, max_span_len, dtype=torch.bool, device=device),
            torch.zeros(B, 1, dtype=torch.long, device=device),
            torch.zeros(B, 1, dtype=torch.long, device=device),
        )

    span_embeds = torch.zeros(B, max_spans, max_span_len, D, dtype=hidden_states.dtype, device=device)
    span_mask = torch.zeros(B, max_spans, max_span_len, dtype=torch.bool, device=device)
    target_token_ids = torch.zeros(B, max_spans, dtype=torch.long, device=device)
    span_sizes = torch.zeros(B, max_spans, dtype=torch.long, device=device)

    positions = torch.arange(L, device=device)

    for b in range(B):
        assigns = span_assignments[b]
        boundaries = span_boundaries[b]
        n_spans = int(num_spans[b].item())
        if n_spans == 0:
            continue

        first_pos = positions[boundaries]

        local_k = positions - first_pos[assigns]
        valid = local_k < max_span_len
        flat_idx = assigns * max_span_len + local_k

        span_flat = span_embeds[b].view(-1, D)
        span_flat[flat_idx[valid]] = hidden_states[b, positions[valid]]
        span_mask[b].view(-1)[flat_idx[valid]] = True

        sizes = torch.zeros(n_spans, dtype=torch.long, device=device)
        sizes[:-1] = first_pos[1:] - first_pos[:-1]
        sizes[-1] = L - first_pos[-1]
        span_sizes[b, :n_spans] = torch.clamp(sizes, max=max_span_len)

        for s in range(n_spans - 1):
            target_token_ids[b, s] = input_ids[b, first_pos[s + 1]]

    return span_embeds, span_mask, target_token_ids, span_sizes


def merge_token_embeddings(
    hidden_states,
    input_ids,
    span_encoder,
    span_boundaries,
    span_assignments,
    max_span_len=4,
):
    span_embeds, span_mask, target_token_ids, span_sizes = group_embeddings_by_spans(
        hidden_states, input_ids, span_boundaries, span_assignments, max_span_len
    )

    merged = span_encoder(span_embeds, span_mask)
    return merged, span_mask, target_token_ids, span_sizes
