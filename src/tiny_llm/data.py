"""Dataset utilities for training."""
from __future__ import annotations

import glob
import json
import os
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from .tokenizer import TinyTokenizer


class NumpyShardDataset(Dataset):
    """Reads many [N, seq_len] .npy shards as one logical dataset."""

    def __init__(self, data_dir: str) -> None:
        self.paths: List[str] = sorted(glob.glob(os.path.join(data_dir, "shard_*.npy")))
        if not self.paths:
            raise FileNotFoundError(f"No shards found in {data_dir}")

        self.shard_sizes: List[int] = []
        self.cum_sizes: List[int] = []
        total = 0
        for path in self.paths:
            shard = np.load(path, mmap_mode="r")
            if shard.ndim != 2:
                raise ValueError(f"Shard {path} must be 2D [N, seq_len]")
            n = shard.shape[0]
            self.shard_sizes.append(n)
            total += n
            self.cum_sizes.append(total)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def _locate(self, idx: int) -> Tuple[int, int]:
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        lo, hi = 0, len(self.cum_sizes) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if idx < self.cum_sizes[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        shard_idx = lo
        shard_start = 0 if shard_idx == 0 else self.cum_sizes[shard_idx - 1]
        pos = idx - shard_start
        return shard_idx, pos

    def __getitem__(self, idx: int) -> torch.Tensor:
        shard_idx, pos = self._locate(idx)
        arr = np.load(self.paths[shard_idx], mmap_mode="r")
        x = arr[pos].astype(np.int64)
        return torch.from_numpy(x)


class InstructionCommandDataset(Dataset):
    """
    Supervised fine-tuning dataset for:
    <instruction> ... <command> ...
    Only the command part contributes to the loss.
    """

    def __init__(self, jsonl_path: str, tokenizer: "TinyTokenizer", max_seq_len: int = 512) -> None:
        self.examples: List[Tuple[str, str]] = []
        self.tok = tokenizer
        self.max_seq_len = max_seq_len

        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                instr = obj["instruction"]
                cmd = obj["command"]
                self.examples.append((instr, cmd))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        instruction, command = self.examples[idx]

        # Build plain text with our special tokens (tokenizer must include these)
        text = (
            "<instruction>\n"
            + instruction.strip()
            + "\n\n"
            + "<command>\n"
            + command.strip()
        )

        # Tokenize via SentencePiece directly, no automatic BOS/EOS
        ids = self.tok.sp.encode(text, out_type=int)
        # Append EOS explicitly
        ids.append(self.tok.eos_token_id)

        # Truncate
        ids = ids[: self.max_seq_len]

        # Build labels: ignore everything before (and including) <command>
        labels = [-100] * len(ids)
        cmd_token_id = self.tok.sp.PieceToId("<command>")
        try:
            cmd_pos = ids.index(cmd_token_id)
        except ValueError:
            cmd_pos = 0  # no <command> token, supervise all

        # Start supervising AFTER the <command> token
        start = cmd_pos + 1
        for i in range(start, len(ids)):
            labels[i] = ids[i]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def cli_sft_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Collate function for variable-length SFT batches."""
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids_batch: List[torch.Tensor] = []
    labels_batch: List[torch.Tensor] = []

    for item in batch:
        ids = item["input_ids"]
        labs = item["labels"]

        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            labs = torch.cat([labs, torch.full((pad_len,), -100, dtype=torch.long)])

        input_ids_batch.append(ids)
        labels_batch.append(labs)

    return {
        "input_ids": torch.stack(input_ids_batch, dim=0),
        "labels": torch.stack(labels_batch, dim=0),
    }
