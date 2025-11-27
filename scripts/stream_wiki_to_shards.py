#!/usr/bin/env python3
"""Stream Wikipedia and tokenize into numpy shards for pretraining."""
import argparse
import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from tiny_llm.tokenizer import TinyTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument(
        "--shard_tokens",
        type=int,
        default=1_000_000,
        help="Approx tokens per shard before reshaping",
    )
    p.add_argument(
        "--max_articles",
        type=int,
        default=None,
        help="Optional cap on number of wiki articles to stream",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = TinyTokenizer(args.tokenizer)

    # Streaming English Wikipedia
    wiki = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    if args.max_articles is not None:
        wiki = wiki.take(args.max_articles)

    buffer = []
    shard_idx = 0
    total_tokens = 0

    for sample in tqdm(wiki, desc="Streaming Wikipedia"):
        text = sample.get("text") or ""
        if not text:
            continue

        ids = tok.encode(text, add_special_tokens=True)
        buffer.extend(ids)
        total_tokens += len(ids)

        while len(buffer) >= args.shard_tokens:
            arr = np.array(buffer, dtype=np.int32)
            usable = (len(arr) // args.seq_len) * args.seq_len
            if usable == 0:
                break
            arr = arr[:usable]
            buffer = buffer[usable:]

            arr = arr.reshape(-1, args.seq_len)
            out_path = os.path.join(args.out_dir, f"shard_{shard_idx:05d}.npy")
            np.save(out_path, arr)
            print(f"Saved shard {shard_idx} with shape {arr.shape}")
            shard_idx += 1

    # Final shard if enough tokens to make at least one sequence
    if buffer:
        arr = np.array(buffer, dtype=np.int32)
        usable = (len(arr) // args.seq_len) * args.seq_len
        if usable > 0:
            arr = arr[:usable].reshape(-1, args.seq_len)
            out_path = os.path.join(args.out_dir, f"shard_{shard_idx:05d}.npy")
            np.save(out_path, arr)
            print(f"Saved final shard {shard_idx} with shape {arr.shape}")

    print(f"Done. Total tokens processed: {total_tokens}")


if __name__ == "__main__":
    main()

