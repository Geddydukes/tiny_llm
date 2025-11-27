#!/usr/bin/env python3
"""Tokenize text files into fixed-length numpy shards."""
from __future__ import annotations

import argparse
import os
from typing import Iterable, List

import numpy as np
from tqdm import tqdm

from tiny_llm.tokenizer import TinyTokenizer


def iter_lines(files: List[str]) -> Iterable[str]:
    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shard tokenized data into numpy files")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Comma separated text files")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--shard_tokens", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tok = TinyTokenizer(args.tokenizer)
    files = args.input.split(",")

    buf: List[int] = []
    total_tokens = 0
    shard_idx = 0

    for line in tqdm(iter_lines(files), desc="Tokenizing"):
        ids = tok.encode(line, add_special_tokens=True)
        buf.extend(ids)
        total_tokens += len(ids)

        if len(buf) >= args.shard_tokens:
            arr = np.array(buf, dtype=np.int32)
            usable = (len(arr) // args.seq_len) * args.seq_len
            arr, buf = arr[:usable], arr[usable:]
            arr = arr.reshape(-1, args.seq_len)

            out_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.npy")
            np.save(out_path, arr)
            shard_idx += 1

    if buf:
        arr = np.array(buf, dtype=np.int32)
        usable = (len(arr) // args.seq_len) * args.seq_len
        if usable > 0:
            arr = arr[:usable].reshape(-1, args.seq_len)
            out_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.npy")
            np.save(out_path, arr)

    print(f"Total tokens: {total_tokens}, shards: {shard_idx + 1}")


if __name__ == "__main__":
    main()
