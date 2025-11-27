#!/usr/bin/env python3
"""Convert raw instruction/command text file to JSONL format."""
import argparse
import json


def parse_blocks(lines):
    """
    Parse alternating Instruction/Command blocks from messy text.

    Handles:
    - Any number of blank lines
    - Multi-line instruction text
    - Multi-line command text
    - Extra blank lines between blocks

    Expected markers (case-insensitive):

        Instruction
        <instruction text...>

        Command
        <command text...>
    """

    mode = None  # None, "instruction", "command"
    instruction_lines = []
    command_lines = []

    def flush():
        """Yield (instruction, command) when both exist."""
        if instruction_lines and command_lines:
            instr = " ".join(l.strip() for l in instruction_lines).strip()
            cmd = " ".join(l.strip() for l in command_lines).strip()
            yield instr, cmd

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        # Skip completely empty lines
        if stripped == "":
            continue

        # Detect markers
        lower = stripped.lower()
        if lower == "instruction":
            # Flush previous complete block (if any)
            for pair in flush():
                yield pair
            mode = "instruction"
            instruction_lines = []
            command_lines = []
            continue

        if lower == "command":
            mode = "command"
            continue

        # Accumulate content depending on mode
        if mode == "instruction":
            instruction_lines.append(line)
        elif mode == "command":
            command_lines.append(line)

    # Final block at EOF
    for pair in flush():
        yield pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw txt file")
    parser.add_argument("--output", required=True, help="JSONL output file")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    count = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for instr, cmd in parse_blocks(lines):
            out.write(json.dumps({"instruction": instr, "command": cmd}) + "\n")
            count += 1

    print(f"Saved {count} pairs to {args.output}")


if __name__ == "__main__":
    main()

