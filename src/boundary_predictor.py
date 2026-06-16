import torch
import torch.nn as nn


class BoundaryPredictor(nn.Module):
    def __init__(self, embed_weight, hidden_dim=256, num_layers=1, num_heads=4, max_span_len=4):
        super().__init__()
        vocab_size, embed_dim = embed_weight.shape

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        with torch.no_grad():
            self.embedding.weight.data.copy_(embed_weight)
        for p in self.embedding.parameters():
            p.requires_grad = False

        self.proj_in = nn.Linear(embed_dim, hidden_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, dim_feedforward=hidden_dim * 4,
            batch_first=True, dropout=0.1, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.max_span_len = max_span_len

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            x = self.embedding(input_ids)
        x = self.proj_in(x)
        mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.transformer(x, src_key_padding_mask=mask)
        logits = self.head(x).squeeze(-1)
        return logits

    def get_boundary_probs(self, input_ids, attention_mask=None):
        logits = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logits)
