#!/usr/bin/env python3
"""Generate CLI commands from natural language instructions."""
import argparse

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

    # Rebuild config safely
    cfg = TinyLLMConfig()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    # Make sure vocab / special IDs match tokenizer
    cfg.vocab_size = tok.vocab_size
    cfg.pad_token_id = tok.pad_token_id
    cfg.bos_token_id = tok.bos_token_id
    cfg.eos_token_id = tok.eos_token_id

    model = TinyLLM(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, tok


def generate_command(model, tok, instruction: str, max_new_tokens: int = 64, device=None):
    """
    Greedy decoding for:

        <instruction>
        {instruction}

        <command>
        {generated...}
    """
    prompt = "<instruction>\n" + instruction.strip() + "\n\n<command>\n"
    prompt_ids = tok.sp.encode(prompt, out_type=int)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(input_ids)
            next_logits = logits[:, -1, :]  # last position
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            next_id = next_token.item()

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_id == tok.eos_token_id:
                break

    full_ids = input_ids[0].tolist()
    gen_ids = full_ids[len(prompt_ids) :]

    # Stop at EOS if present
    if tok.eos_token_id in gen_ids:
        eos_pos = gen_ids.index(tok.eos_token_id)
        gen_ids = gen_ids[:eos_pos]

    command = tok.decode(gen_ids).strip()
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Optional single instruction; if omitted, enters REPL mode.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    print("Using device:", device)

    model, tok = build_model_from_ckpt(args.ckpt, args.tokenizer, device)

    if args.instruction is not None:
        cmd = generate_command(
            model, tok, args.instruction, max_new_tokens=args.max_new_tokens, device=device
        )
        print("Command:")
        print(cmd)
    else:
        # REPL mode
        print("Enter instructions (Ctrl+C to quit):")
        while True:
            try:
                instr = input("> ").strip()
                if not instr:
                    continue
                cmd = generate_command(
                    model, tok, instr, max_new_tokens=args.max_new_tokens, device=device
                )
                print(f"[command] {cmd}\n")
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break


if __name__ == "__main__":
    main()

