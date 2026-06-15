import torch
import torch.nn as nn
import torch.nn.functional as F


class SpanEncoder(nn.Module):
    def __init__(self, hidden_dim, max_span_len=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_span_len = max_span_len

        self.local_pos = nn.Embedding(max_span_len, hidden_dim)
        nn.init.zeros_(self.local_pos.weight)

        self.mlp = nn.Sequential(
            nn.Linear(max_span_len * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        for module in self.mlp:
            if hasattr(module, 'weight'):
                nn.init.normal_(module.weight, std=0.02 / (2 * hidden_dim) ** 0.5)
                nn.init.zeros_(module.bias)

        self.zero_residual = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.zero_residual.weight)
        nn.init.zeros_(self.zero_residual.bias)

    def forward(self, token_embeds, span_mask):
        B, S, K, D = token_embeds.shape

        pos_ids = torch.arange(K, device=token_embeds.device)
        pos_emb = self.local_pos(pos_ids).view(1, 1, K, D).to(token_embeds.dtype)

        h = token_embeds + pos_emb

        mask = span_mask.unsqueeze(-1).to(h.dtype)

        safe_mean = (h * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)

        local_flat = (h * mask).reshape(B, S, K * D)
        local_proj = self.mlp(local_flat)

        return safe_mean + self.zero_residual(local_proj)
