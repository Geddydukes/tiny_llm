#!/usr/bin/env python3
"""Instruction fine-tuning script for CLI command generation."""
import argparse
import math
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tiny_llm.config import TinyLLMConfig
from tiny_llm.data import InstructionCommandDataset, cli_sft_collate_fn
from tiny_llm.model import TinyLLM
from tiny_llm.tokenizer import TinyTokenizer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=str, required=True)
    p.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="JSONL with {instruction, command}",
    )
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument(
        "--from_ckpt",
        type=str,
        default=None,
        help="Optional: start from a pretrained checkpoint",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print("Using device:", device)

    tok = TinyTokenizer(args.tokenizer)

    cfg = TinyLLMConfig()
    cfg.vocab_size = tok.vocab_size
    cfg.pad_token_id = tok.pad_token_id
    cfg.bos_token_id = tok.bos_token_id
    cfg.eos_token_id = tok.eos_token_id
    cfg.max_seq_len = args.max_seq_len

    model = TinyLLM(cfg).to(device)

    if args.from_ckpt is not None:
        ckpt = torch.load(args.from_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded pretrained weights from {args.from_ckpt}")

    dataset = InstructionCommandDataset(args.jsonl, tok, max_seq_len=cfg.max_seq_len)
    collate = lambda b: cli_sft_collate_fn(b, pad_token_id=tok.pad_token_id)
    dl = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    global_step = 0

    def lr_schedule(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    model.train()
    while global_step < args.max_steps:
        for batch in dl:
            global_step += 1
            if global_step > args.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            factor = lr_schedule(global_step)
            for g in optimizer.param_groups:
                g["lr"] = args.lr * factor

            optimizer.zero_grad()
            _, loss = model(input_ids, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if global_step % args.log_every == 0:
                print(
                    f"step {global_step} | loss {loss.item():.4f} | "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}"
                )

            if global_step % args.save_every == 0:
                ckpt = {
                    "config": cfg.__dict__,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": global_step,
                }
                path = os.path.join(args.out_dir, f"cli_sft_step_{global_step:06d}.pt")
                torch.save(ckpt, path)
                print(f"Saved checkpoint to {path}")

    print("CLI SFT training complete at step", global_step)


if __name__ == "__main__":
    main()
