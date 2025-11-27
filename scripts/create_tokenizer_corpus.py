#!/usr/bin/env python3
"""Create tokenizer training corpus from streaming Wikipedia."""
import argparse

from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True)
    p.add_argument(
        "--articles",
        type=int,
        default=50000,
        help="Number of wiki articles to stream into tokenizer corpus",
    )
    args = p.parse_args()

    print(f"Streaming {args.articles} Wikipedia articles...")

    wiki = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    ).take(args.articles)

    with open(args.out, "w", encoding="utf-8") as f:
        for sample in wiki:
            text = sample.get("text") or ""
            text = text.strip()
            if text:
                f.write(text + "\n\n")

    print(f"Done. Tokenizer corpus saved to: {args.out}")


if __name__ == "__main__":
    main()

