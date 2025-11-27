"""Model configuration utilities."""
from dataclasses import dataclass


@dataclass
class TinyLLMConfig:
    """Small yet modern Transformer configuration."""

    vocab_size: int = 32_000
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512

    rope_base: float = 10_000.0
    dropout: float = 0.0

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
