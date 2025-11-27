#!/usr/bin/env python3
"""Train SentencePiece tokenizer with chat special tokens."""
import argparse

import sentencepiece as spm

SPECIAL_TOKENS = [
    "<system>",
    "<user>",
    "<assistant>",
    "<instruction>",
    "<command>",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer")
    parser.add_argument("--input", type=str, required=True, help="Text file or comma separated list")
    parser.add_argument("--vocab_size", type=int, default=32_000)
    parser.add_argument("--model_prefix", type=str, default="tokenizer")
    parser.add_argument("--model_type", type=str, default="bpe", choices=["bpe", "unigram"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    user_symbols = ",".join(SPECIAL_TOKENS)

    spm.SentencePieceTrainer.Train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=0.9995,
        shuffle_input_sentence=True,
        input_sentence_size=10_000_000,
        pad_id=0,
        unk_id=3,
        bos_id=1,
        eos_id=2,
        user_defined_symbols=user_symbols,
    )
    print(f"Saved {args.model_prefix}.model and {args.model_prefix}.vocab")


if __name__ == "__main__":
    main()
