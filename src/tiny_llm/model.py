"""TinyLLM Transformer model definition."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import TinyLLMConfig
from .layers import RMSNorm, TransformerBlock
from .rope import build_rope_cache


class TinyLLM(nn.Module):
    def __init__(self, config: TinyLLMConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.rope_cos: torch.Tensor | None = None
        self.rope_sin: torch.Tensor | None = None

    def _ensure_rope(self, device: torch.device, dtype: torch.dtype) -> None:
        head_dim = self.config.d_model // self.config.n_heads
        if head_dim % 2 != 0:
            raise ValueError("Head dimension must be even for RoPE")
        if self.rope_cos is None or self.rope_cos.size(0) < self.config.max_seq_len:
            cos, sin = build_rope_cache(
                seq_len=self.config.max_seq_len,
                head_dim=head_dim,
                base=self.config.rope_base,
                device=device,
                dtype=dtype,
            )
            self.rope_cos = cos
            self.rope_sin = sin

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        device = input_ids.device
        b, s = input_ids.shape

        self._ensure_rope(device, torch.float32)

        x = self.token_embedding(input_ids)

        mask = torch.full((1, 1, s, s), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)

        for block in self.blocks:
            x = block(x, attn_mask=mask, rope_cos=self.rope_cos, rope_sin=self.rope_sin)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))

        return logits, loss
