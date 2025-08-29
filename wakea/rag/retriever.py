from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .embed import hash_embed
from .index_faiss import InMemoryIndex, Doc


@dataclass
class RetrievedDoc:
    text: str
    score: float


class InMemoryRetriever:
    _GLOBAL: Optional["InMemoryRetriever"] = None

    def __init__(self, dim: int = 256):
        self.index = InMemoryIndex(dim=dim)

    @classmethod
    def get_global(cls) -> Optional["InMemoryRetriever"]:
        return cls._GLOBAL

    @classmethod
    def set_global(cls, r: "InMemoryRetriever") -> None:
        cls._GLOBAL = r

    def ingest(self, texts: List[str]):
        vecs = np.stack([hash_embed(t, self.index.dim) for t in texts], axis=0)
        self.index.add(texts, vecs)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedDoc]:
        q = hash_embed(query, self.index.dim)
        docs: List[Doc] = self.index.search(q, top_k=top_k)
        return [RetrievedDoc(text=d.text, score=float((d.vec @ q))) for d in docs]

