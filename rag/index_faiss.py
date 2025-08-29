from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class Doc:
    text: str
    vec: np.ndarray


class InMemoryIndex:
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.docs: List[Doc] = []

    def add(self, texts: List[str], vecs: np.ndarray):
        for t, v in zip(texts, vecs):
            self.docs.append(Doc(t, v.astype(np.float32)))

    def search(self, q: np.ndarray, top_k: int = 5) -> List[Doc]:
        if not self.docs:
            return []
        mat = np.stack([d.vec for d in self.docs], axis=0)
        sims = (mat @ q)  # cosine if q and vecs normalized
        idx = np.argsort(-sims)[:top_k]
        return [self.docs[i] for i in idx]


class FAISSIndex:
    """Thin wrapper around FAISS with text storage.

    Uses IndexFlatIP with normalized embeddings for cosine similarity.
    """

    def __init__(self, dim: int):
        try:
            import faiss  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("faiss is required for FAISSIndex")
        self.faiss = faiss
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.texts: List[str] = []

    def add(self, texts: List[str], vecs: np.ndarray):
        # Ensure float32 and normalized
        v = vecs.astype("float32")
        norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        v = v / norms
        self.index.add(v)
        self.texts.extend(texts)

    def search(self, q: np.ndarray, top_k: int = 5) -> List[Doc]:
        if len(self.texts) == 0:
            return []
        q = q.astype("float32")
        q = q / (np.linalg.norm(q) + 1e-8)
        D, I = self.index.search(q[None, :], top_k)
        idxs = I[0]
        # We do not reconstruct vectors; return dummy zero vecs to satisfy type
        return [Doc(self.texts[i], np.zeros(self.dim, dtype=np.float32)) for i in idxs if i >= 0]

    def search_with_scores(self, q: np.ndarray, top_k: int = 5):
        """Return (texts, scores) for cosine similarity queries."""
        if len(self.texts) == 0:
            return [], np.array([], dtype=np.float32)
        q = q.astype("float32")
        q = q / (np.linalg.norm(q) + 1e-8)
        D, I = self.index.search(q[None, :], top_k)
        idxs = I[0]
        texts = [self.texts[i] for i in idxs if i >= 0]
        scores = D[0][: len(texts)].astype(np.float32)
        return texts, scores

    def save(self, path: str) -> None:
        import pickle

        faiss = self.faiss
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".meta", "wb") as f:
            pickle.dump({"texts": self.texts, "dim": self.dim}, f)

    @classmethod
    def load(cls, path: str) -> "FAISSIndex":
        import pickle
        import os
        import faiss  # type: ignore

        index = faiss.read_index(path + ".faiss")
        with open(path + ".meta", "rb") as f:
            meta = pickle.load(f)
        dim = int(meta.get("dim"))
        inst = cls(dim)
        inst.index = index
        inst.texts = list(meta.get("texts", []))
        return inst
