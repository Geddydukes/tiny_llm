#!/usr/bin/env python3
"""Benchmark generation speed (tokens per second)."""
import argparse
import time

import torch

from tiny_llm.config import TinyLLMConfig
from tiny_llm.model import TinyLLM
from tiny_llm.tokenizer import TinyTokenizer


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--runs", type=int, default=20, help="Number of benchmark runs")
    parser.add_argument("--max_tokens", type=int, default=64, help="Max tokens per generation")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load model
    tok = TinyTokenizer(args.tokenizer)
    ckpt = torch.load(args.ckpt, map_location="cpu")

    cfg = TinyLLMConfig()
    for k, v in ckpt.get("config", {}).items():
        setattr(cfg, k, v)

    cfg.vocab_size = tok.vocab_size
    cfg.pad_token_id = tok.pad_token_id
    cfg.bos_token_id = tok.bos_token_id
    cfg.eos_token_id = tok.eos_token_id

    model = TinyLLM(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Test instruction
    instruction = "List all files in the current directory"
    prompt = f"<instruction>\n{instruction}\n\n<command>\n"
    prompt_ids = tok.sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids)

    # Benchmark
    print(f"\nBenchmarking {args.runs} runs with max {args.max_tokens} tokens each...")
    start_time = time.time()
    total_tokens = 0

    with torch.no_grad():
        for run in range(args.runs):
            current_input = input_ids.clone()
            for i in range(args.max_tokens):
                logits, _ = model(current_input)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                current_input = torch.cat([current_input, next_token], dim=1)
                total_tokens += 1

    end_time = time.time()

    total_time = end_time - start_time
    tokens_per_second = total_tokens / total_time
    time_per_token = 1000 * total_time / total_tokens

    print(f"\n=== Generation Speed Benchmark ===")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Time per token: {time_per_token:.2f} ms")
    print(f"Throughput: {tokens_per_second:.1f} tok/s")


if __name__ == "__main__":
    main()

