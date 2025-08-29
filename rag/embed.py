from __future__ import annotations

import hashlib
import numpy as np


def hash_embed(text: str, dim: int = 256) -> np.ndarray:
    """Cheap, deterministic embedding via hashing n-grams.

    Replace with sentence-transformers or custom encoders for real use.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()
    for t in tokens:
        h = int(hashlib.md5(t.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    # L2 normalize
    n = np.linalg.norm(vec) + 1e-8
    return vec / n

