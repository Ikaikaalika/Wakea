from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

from .embed import EmbeddingModel, hash_embed
from .index_faiss import InMemoryIndex, FAISSIndex, Doc


@dataclass
class RetrievedDoc:
    text: str
    score: float


class Retriever:
    _GLOBAL: Optional["Retriever"] = None

    def __init__(self, backend: str = "in_memory", dim: int = 256, embed_model: Optional[str] = None):
        self.backend = backend
        self.embed = EmbeddingModel(embed_model, dim)
        self.dim = self.embed.dim
        if backend == "faiss":
            try:
                self.index: Union[InMemoryIndex, FAISSIndex] = FAISSIndex(self.dim)
            except Exception:
                self.index = InMemoryIndex(self.dim)
                self.backend = "in_memory"
        else:
            self.index = InMemoryIndex(self.dim)

    @classmethod
    def get_global(cls) -> Optional["Retriever"]:
        return cls._GLOBAL

    @classmethod
    def set_global(cls, r: "Retriever") -> None:
        cls._GLOBAL = r

    def ingest(self, texts: List[str]):
        vecs = self.embed.encode(texts)
        self.index.add(texts, vecs)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedDoc]:
        if self.embed.model is not None:
            q = self.embed.encode([query])[0]
        else:
            q = hash_embed(query, self.dim)
        # FAISS path with direct scores
        if isinstance(self.index, FAISSIndex):
            texts, scores = self.index.search_with_scores(q, top_k=top_k)
            return [RetrievedDoc(text=t, score=float(s)) for t, s in zip(texts, scores)]
        docs: List[Doc] = self.index.search(q, top_k=top_k)
        return [RetrievedDoc(text=d.text, score=float((d.vec @ q))) for d in docs]

    def save(self, path: str) -> int:
        if isinstance(self.index, FAISSIndex):
            self.index.save(path)
            return len(self.index.texts)
        # In-memory fallback
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.index, f)
        return len(self.index.docs)

    @classmethod
    def load(cls, path: str, backend: str = "in_memory", dim: int = 256, embed_model: Optional[str] = None) -> "Retriever":
        if backend == "faiss":
            try:
                idx = FAISSIndex.load(path)
                r = cls(backend="faiss", dim=idx.dim, embed_model=embed_model)
                r.index = idx
                return r
            except Exception:
                pass
        import pickle
        with open(path, "rb") as f:
            index = pickle.load(f)
        r = cls(backend="in_memory", dim=index.dim, embed_model=embed_model)
        r.index = index
        return r
