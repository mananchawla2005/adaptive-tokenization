import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

from .span_encoder import SpanEncoder
from .merging import build_span_masks_with_heuristics, merge_token_embeddings


class AdaptiveGPT2Model(nn.Module):
    def __init__(self, base_model_name="gpt2", max_span_len=4):
        super().__init__()
        self.config = GPT2Config.from_pretrained(base_model_name)
        self.transformer = GPT2Model.from_pretrained(base_model_name)

        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

        self.span_encoder = SpanEncoder(self.config.n_embd, max_span_len=max_span_len)
        self.max_span_len = max_span_len

    def get_input_embeddings(self):
        return self.transformer.wte

    def forward_with_merging(self, input_ids, tokenizer=None, compression_ratio=0.5, rng=None):
        B, L = input_ids.shape

        hidden_states = self.transformer.wte(input_ids)
        attention_mask = torch.ones(B, L, dtype=torch.long, device=input_ids.device)

        span_boundaries, span_assignments = build_span_masks_with_heuristics(
            input_ids,
            tokenizer=tokenizer,
            max_span_len=self.max_span_len,
            max_compression_ratio=compression_ratio,
            rng=rng,
        )

        merged_embeds, span_mask, target_token_ids, span_sizes = merge_token_embeddings(
            hidden_states,
            input_ids,
            self.span_encoder,
            span_boundaries,
            span_assignments,
            max_span_len=self.max_span_len,
        )

        B_s, S = merged_embeds.shape[:2]
        span_attn_mask = torch.ones(B_s, S, dtype=torch.long, device=input_ids.device)

        outputs = self.transformer(
            inputs_embeds=merged_embeds,
            attention_mask=span_attn_mask,
            output_hidden_states=False,
        )

        hidden = outputs.last_hidden_state
        logits = self.lm_head(hidden)

        return logits, target_token_ids, span_sizes, {
            "span_boundaries": span_boundaries,
            "span_assignments": span_assignments,
        }

    def forward(self, input_ids, tokenizer=None, apply_merging=False, compression_ratio=0.5, rng=None):
        if apply_merging:
            return self.forward_with_merging(
                input_ids, tokenizer=tokenizer, compression_ratio=compression_ratio, rng=rng
            )

        hidden_states = self.transformer.wte(input_ids)
        attention_mask = torch.ones(input_ids.shape[0], input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        outputs = self.transformer(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        logits = self.lm_head(outputs.last_hidden_state)

        return logits

    def forward_context_target(self, input_ids, tokenizer=None, compression_ratio=0.5, rng=None):
        B, L = input_ids.shape
        device = input_ids.device

        split = max(1, int(L * 0.75))
        ctx_ids = input_ids[:, :split]
        tgt_ids = input_ids[:, split:]

        ctx_embeds = self.transformer.wte(ctx_ids)

        span_boundaries, span_assignments = build_span_masks_with_heuristics(
            ctx_ids,
            tokenizer=tokenizer,
            max_span_len=self.max_span_len,
            max_compression_ratio=compression_ratio,
            rng=rng,
        )

        merged_ctx, _, _, _ = merge_token_embeddings(
            ctx_embeds,
            ctx_ids,
            self.span_encoder,
            span_boundaries,
            span_assignments,
            max_span_len=self.max_span_len,
        )

        tgt_embeds = self.transformer.wte(tgt_ids)

        combined_embeds = torch.cat([merged_ctx, tgt_embeds], dim=1)
        B_s, total_len = combined_embeds.shape[:2]
        combined_mask = torch.ones(B_s, total_len, dtype=torch.long, device=device)

        outputs = self.transformer(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=False,
        )

        logits = self.lm_head(outputs.last_hidden_state)

        ctx_len = merged_ctx.shape[1]
        ctx_labels = torch.full((B, ctx_len), -100, dtype=torch.long, device=device)
        tgt_labels = tgt_ids.clone()
        labels = torch.cat([ctx_labels, tgt_labels], dim=1)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        self._last_num_spans = merged_ctx.shape[1]
        return loss

    def forward_loss_on_target(self, input_ids):
        B, L = input_ids.shape
        device = input_ids.device

        split = max(1, int(L * 0.75))

        hidden_states = self.transformer.wte(input_ids)
        attn_mask = torch.ones(B, L, dtype=torch.long, device=device)

        outputs = self.transformer(
            inputs_embeds=hidden_states,
            attention_mask=attn_mask,
        )
        logits = self.lm_head(outputs.last_hidden_state)

        shift_logits = logits[:, split - 1 : -1, :].contiguous()
        shift_labels = input_ids[:, split:].contiguous()

        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
        )
        return loss

    def forward_chat(self, prompt_ids, answer_ids, tokenizer=None, compression_ratio=0.5, rng=None, prompt_mask=None, answer_mask=None):
        B, Pl = prompt_ids.shape
        Al = answer_ids.shape[1]
        device = prompt_ids.device

        prompt_embeds = self.transformer.wte(prompt_ids)

        span_boundaries, span_assignments = build_span_masks_with_heuristics(
            prompt_ids,
            tokenizer=tokenizer,
            max_span_len=self.max_span_len,
            max_compression_ratio=compression_ratio,
            rng=rng,
        )

        merged_prompt, _, _, _ = merge_token_embeddings(
            prompt_embeds,
            prompt_ids,
            self.span_encoder,
            span_boundaries,
            span_assignments,
            max_span_len=self.max_span_len,
        )

        answer_embeds = self.transformer.wte(answer_ids)

        combined_embeds = torch.cat([merged_prompt, answer_embeds], dim=1)
        B_s, total_len = combined_embeds.shape[:2]
        combined_mask = torch.ones(B_s, total_len, dtype=torch.long, device=device)

        outputs = self.transformer(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=False,
        )

        logits = self.lm_head(outputs.last_hidden_state)

        prompt_len = merged_prompt.shape[1]
        prompt_labels = torch.full((B, prompt_len), -100, dtype=torch.long, device=device)
        answer_labels = answer_ids.clone()
        if answer_mask is not None:
            answer_labels[answer_mask == 0] = -100
        labels = torch.cat([prompt_labels, answer_labels], dim=1)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        self._last_num_spans = merged_prompt.shape[1]
        return loss

    def forward_chat_no_compress(self, prompt_ids, answer_ids, prompt_mask=None, answer_mask=None):
        B, Pl = prompt_ids.shape
        device = prompt_ids.device

        prompt_embeds = self.transformer.wte(prompt_ids)
        answer_embeds = self.transformer.wte(answer_ids)

        combined_embeds = torch.cat([prompt_embeds, answer_embeds], dim=1)
        B_s, total_len = combined_embeds.shape[:2]

        if prompt_mask is not None and answer_mask is not None:
            combined_mask = torch.cat([prompt_mask, answer_mask], dim=1)
        else:
            combined_mask = torch.ones(B_s, total_len, dtype=torch.long, device=device)

        outputs = self.transformer(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=False,
        )

        logits = self.lm_head(outputs.last_hidden_state)

        prompt_labels = torch.full((B, Pl), -100, dtype=torch.long, device=device)
        answer_labels = answer_ids.clone()
        if answer_mask is not None:
            answer_labels[answer_mask == 0] = -100
        labels = torch.cat([prompt_labels, answer_labels], dim=1)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    def compute_loss(self, logits, labels, span_sizes=None):
        if span_sizes is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, :-1].contiguous()

            valid_mask = (span_sizes[:, :-1] > 0) & (span_sizes[:, 1:] > 0)

            flat_logits = shift_logits[valid_mask]
            flat_labels = shift_labels[valid_mask]

            if flat_labels.numel() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            loss = nn.functional.cross_entropy(flat_logits, flat_labels)
        else:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return loss
