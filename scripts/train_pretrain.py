#!/usr/bin/env python3
"""Tiny LLM pretraining entrypoint."""
from __future__ import annotations

import argparse
import math
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tiny_llm.config import TinyLLMConfig
from tiny_llm.data import NumpyShardDataset
from tiny_llm.model import TinyLLM
from tiny_llm.tokenizer import TinyTokenizer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyLLM on numpy shards")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=50_000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = get_device()
    print("Using device:", device)

    tokenizer = TinyTokenizer(args.tokenizer)
    cfg = TinyLLMConfig()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.bos_token_id = tokenizer.bos_token_id
    cfg.eos_token_id = tokenizer.eos_token_id

    model = TinyLLM(cfg).to(device)

    dataset = NumpyShardDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    global_step = 0
    total_tokens = 0

    def lr_schedule(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        progress = min(progress, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    model.train()
    while global_step < args.max_steps:
        for batch in dataloader:
            global_step += 1
            if global_step > args.max_steps:
                break

            batch = batch.to(device)
            labels = batch.clone()

            lr_factor = lr_schedule(global_step)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_factor

            optimizer.zero_grad()
            _, loss = model(batch, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_tokens += batch.numel()

            if global_step % args.log_every == 0:
                print(
                    f"step {global_step} | loss {loss.item():.4f} | "
                    f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                    f"tokens_seen {total_tokens}"
                )

            if global_step % args.save_every == 0:
                ckpt = {
                    "config": cfg.__dict__,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": global_step,
                }
                path = os.path.join(args.out_dir, f"pretrain_step_{global_step:06d}.pt")
                torch.save(ckpt, path)
                print(f"Saved checkpoint to {path}")

    print("Pretraining complete at step", global_step)


if __name__ == "__main__":
    main()
