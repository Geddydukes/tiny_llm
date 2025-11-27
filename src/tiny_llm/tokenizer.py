"""SentencePiece wrapper for TinyLLM."""
from __future__ import annotations

import sentencepiece as sp


class TinyTokenizer:
    """Thin wrapper that enforces TinyLLM special tokens."""

    def __init__(self, model_path: str) -> None:
        self.sp = sp.SentencePieceProcessor()
        if not self.sp.load(model_path):
            raise ValueError(f"Failed to load SentencePiece model from {model_path}")

        self.pad_token_id = self.sp.PieceToId("<pad>")
        self.bos_token_id = self.sp.PieceToId("<bos>")
        self.eos_token_id = self.sp.PieceToId("<eos>")
        self.unk_token_id = self.sp.unk_id()

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_special_tokens:
            return [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        if ids and ids[0] == self.bos_token_id:
            ids = ids[1:]
        if ids and ids[-1] == self.eos_token_id:
            ids = ids[:-1]
        return self.sp.decode(ids)
