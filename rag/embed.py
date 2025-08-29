from __future__ import annotations

import hashlib
from typing import List, Optional

import numpy as np


def hash_embed(text: str, dim: int = 256) -> np.ndarray:
    """Cheap, deterministic embedding via hashing tokens. Use only as fallback."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()
    for t in tokens:
        h = int(hashlib.md5(t.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    n = np.linalg.norm(vec) + 1e-8
    return vec / n


class EmbeddingModel:
    """Wraps sentence-transformers if available; falls back to hash embeddings."""

    def __init__(self, model_name: Optional[str] = None, dim: int = 256):
        self.model = None
        self.dim = dim
        self.name = model_name
        if model_name:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self.model = SentenceTransformer(model_name)
                # Infer dimension from model output
                test = self.model.encode(["test"], normalize_embeddings=True)
                self.dim = int(test.shape[1])
            except Exception:
                self.model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is not None:
            arr = self.model.encode(texts, normalize_embeddings=True)
            return np.asarray(arr, dtype=np.float32)
        # Fallback to hash embeddings
        vecs = [hash_embed(t, dim=self.dim) for t in texts]
        return np.stack(vecs, axis=0)

