from __future__ import annotations

# Placeholder FAISS-like interface for consistency; actual index is in-memory.

from dataclasses import dataclass
from typing import List

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

