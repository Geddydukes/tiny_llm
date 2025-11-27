"""Rotary position embedding helpers."""
from __future__ import annotations

import torch


def build_rope_cache(seq_len: int, head_dim: int, base: float, device: torch.device, dtype: torch.dtype):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    angles = positions * theta

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, head_dim)
    sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, head_dim)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to queries or keys."""

    b, s, h, d = x.shape
    # Ensure cos and sin are on the same device and dtype as x
    cos = cos[:s].to(x.device).to(x.dtype)
    sin = sin[:s].to(x.device).to(x.dtype)
    # Reshape to (1, s, 1, d) for broadcasting with (b, s, h, d)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)
    return x * cos + x_rot * sin
