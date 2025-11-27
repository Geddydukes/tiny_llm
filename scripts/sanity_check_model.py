#!/usr/bin/env python3
"""Quick smoke test for TinyLLM forward pass."""
from __future__ import annotations

import torch

from tiny_llm.config import TinyLLMConfig
from tiny_llm.model import TinyLLM


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    cfg = TinyLLMConfig()
    model = TinyLLM(cfg).to(device)
    x = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
    logits, loss = model(x, labels=x)
    print("Logits shape:", logits.shape, "Loss:", float(loss))


if __name__ == "__main__":
    main()
