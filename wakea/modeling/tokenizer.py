from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any


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


class HFTokenizerWrapper:
    """Thin wrapper over Hugging Face tokenizers if available.

    Use with `pretrained` model name. Falls back to basic tokenization when unavailable.
    """

    def __init__(self, pretrained: str):
        from transformers import AutoTokenizer  # type: ignore

        self.tok = AutoTokenizer.from_pretrained(pretrained)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or "<pad>"

    @property
    def pad_id(self) -> int:
        return int(self.tok.pad_token_id)

    @property
    def bos_id(self) -> int:
        return int(self.tok.bos_token_id or self.tok.cls_token_id or self.pad_id)

    @property
    def eos_id(self) -> int:
        return int(self.tok.eos_token_id or self.tok.sep_token_id or self.pad_id)

    def build_from_corpus(self, *_args: Any, **_kwargs: Any) -> None:  # no-op
        return None

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        return list(self.tok.encode(text, add_special_tokens=add_special))

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        return str(self.tok.decode(ids, skip_special_tokens=skip_special))


def get_text_tokenizer(pretrained: Optional[str] = None):
    """Return a tokenizer.

    - If `pretrained` is provided and transformers is available → HF tokenizer
    - Else → SimpleTokenizer
    """
    if pretrained:
        try:
            # Delay import to avoid hard dependency
            import transformers  # type: ignore  # noqa: F401

            return HFTokenizerWrapper(pretrained)
        except Exception:
            pass
    return SimpleTokenizer()
