from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


@dataclass
class SimpleTokenizerConfig:
    lowercase: bool = True


class SimpleTokenizer:
    """A tiny whitespace tokenizer to keep the scaffold runnable without external deps.

    Replace with a real tokenizer (SentencePiece/Tiktoken) for experiments.
    """

    def __init__(self, cfg: SimpleTokenizerConfig | None = None):
        self.cfg = cfg or SimpleTokenizerConfig()
        self.token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.id_to_token: Dict[int, str] = {i: tok for tok, i in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    def build_from_corpus(self, texts: List[str], min_freq: int = 1, max_size: int = 50000) -> None:
        from collections import Counter

        cnt: Counter = Counter()
        for t in texts:
            if self.cfg.lowercase:
                t = t.lower()
            cnt.update(t.split())
        # Reserve special token ids
        next_id = len(self.token_to_id)
        for tok, freq in cnt.most_common():
            if freq < min_freq:
                continue
            if tok in self.token_to_id:
                continue
            self.token_to_id[tok] = next_id
            self.id_to_token[next_id] = tok
            next_id += 1
            if next_id >= max_size:
                break

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        if self.cfg.lowercase:
            text = text.lower()
        toks = text.split()
        ids = [self.token_to_id.get(tok, self.token_to_id["<unk>"]) for tok in toks]
        if add_special:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        toks: List[str] = []
        for i in ids:
            tok = self.id_to_token.get(i, "<unk>")
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            toks.append(tok)
        return " ".join(toks)

