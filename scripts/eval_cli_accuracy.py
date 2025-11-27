#!/usr/bin/env python3
"""Evaluate NL→CLI model accuracy on held-out instructions."""
import argparse
import json
from typing import List, Tuple

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


def build_model_from_ckpt(ckpt_path: str, tokenizer_path: str, device):
    tok = TinyTokenizer(tokenizer_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("config", {})

    cfg = TinyLLMConfig()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    # Sync vocab / special IDs with tokenizer
    cfg.vocab_size = tok.vocab_size
    cfg.pad_token_id = tok.pad_token_id
    cfg.bos_token_id = tok.bos_token_id
    cfg.eos_token_id = tok.eos_token_id

    model = TinyLLM(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, tok


def generate_command(model, tok, instruction: str, max_new_tokens: int, device):
    """
    Generate a command from a single natural-language instruction using
    the <instruction> / <command> format.
    """
    prompt = "<instruction>\n" + instruction.strip() + "\n\n<command>\n"
    prompt_ids = tok.sp.encode(prompt, out_type=int)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(input_ids)
            next_logits = logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            next_id = next_token.item()

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_id == tok.eos_token_id:
                break

    full_ids = input_ids[0].tolist()
    gen_ids = full_ids[len(prompt_ids) :]

    if tok.eos_token_id in gen_ids:
        eos_pos = gen_ids.index(tok.eos_token_id)
        gen_ids = gen_ids[:eos_pos]

    command = tok.decode(gen_ids).strip()
    return command


def _strip_instruction_prefix(line: str) -> str:
    """
    Handles lines like:
      'Instruction: List all files...'
    or just:
      'List all files...'
    """
    line = line.strip()
    if not line:
        return ""
    # Ignore comment-style lines if you add any later
    if line.lstrip().startswith("#"):
        return ""
    prefix = "Instruction:"
    if line.startswith(prefix):
        line = line[len(prefix) :].strip()
    return line


def load_instruction_gold_pairs(instr_path: str, gold_path: str, limit: int = 100) -> List[Tuple[str, str]]:
    with open(instr_path, "r", encoding="utf-8") as fi:
        raw_instr_lines = [l.rstrip("\n") for l in fi]

    with open(gold_path, "r", encoding="utf-8") as fg:
        gold_lines = [l.rstrip("\n") for l in fg]

    if len(raw_instr_lines) != len(gold_lines):
        raise ValueError(
            f"Instruction file has {len(raw_instr_lines)} lines but gold file has {len(gold_lines)}"
        )

    pairs: List[Tuple[str, str]] = []
    for i in range(len(raw_instr_lines)):
        instr = _strip_instruction_prefix(raw_instr_lines[i])
        gold = gold_lines[i].strip()
        if not instr:
            # skip empty / comment lines in instructions; also skip same index in gold
            continue
        pairs.append((instr, gold))
        if len(pairs) >= limit:
            break

    return pairs


def normalize_command(cmd: str) -> str:
    """
    Simple normalization for comparison:
    - strip leading/trailing whitespace
    - collapse internal whitespace to single spaces
    """
    parts = cmd.strip().split()
    return " ".join(parts)


def main():
    p = argparse.ArgumentParser(description="Evaluate NL→CLI model accuracy on held-out instructions.")
    p.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer.model")
    p.add_argument("--ckpt", type=str, required=True, help="Path to fine-tuned checkpoint (.pt)")
    p.add_argument(
        "--instructions",
        type=str,
        required=True,
        help="Text file with NL instructions (lines like 'Instruction: ...').",
    )
    p.add_argument("--gold", type=str, required=True, help="Text file with gold commands (one per line, aligned).")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--limit", type=int, default=100, help="Max number of pairs to evaluate")
    p.add_argument("--output_jsonl", type=str, default=None, help="Optional path to write detailed results as JSONL")

    args = p.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model, tok = build_model_from_ckpt(args.ckpt, args.tokenizer, device)

    pairs = load_instruction_gold_pairs(args.instructions, args.gold, limit=args.limit)
    print(f"Loaded {len(pairs)} instruction/gold pairs")

    n_total = len(pairs)
    n_exact = 0

    detailed_results = []

    for idx, (instr, gold) in enumerate(pairs, start=1):
        pred = generate_command(model, tok, instr, args.max_new_tokens, device=device)

        gold_norm = normalize_command(gold)
        pred_norm = normalize_command(pred)

        exact = gold_norm == pred_norm
        if exact:
            n_exact += 1

        detailed_results.append(
            {
                "index": idx,
                "instruction": instr,
                "gold": gold,
                "gold_norm": gold_norm,
                "pred": pred,
                "pred_norm": pred_norm,
                "exact_match": exact,
            }
        )

        print("=" * 80)
        print(f"[{idx}] INSTRUCTION:")
        print(instr)
        print("\nGOLD COMMAND:")
        print(gold)
        print("\nMODEL COMMAND:")
        print(pred)
        print(f"\nEXACT MATCH: {exact}")
        print("=" * 80)

    accuracy = (n_exact / n_total) * 100 if n_total > 0 else 0.0
    print("\n" + "#" * 80)
    print(f"Total examples: {n_total}")
    print(f"Exact matches:  {n_exact}")
    print(f"Accuracy:       {accuracy:.2f}%")
    print("#" * 80)

    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl) if os.path.dirname(args.output_jsonl) else ".", exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for r in detailed_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nDetailed results written to {args.output_jsonl}")


if __name__ == "__main__":
    import os

    main()

