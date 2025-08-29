from __future__ import annotations

import math
from typing import Tuple

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover - optional runtime dep
    torch = None  # type: ignore


def _build_sin_cos_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    theta = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    pos = np.arange(seq_len, dtype=np.float32)[:, None]
    freqs = pos * theta[None, :]
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply RoPE to query/key projections.

    x: (..., seq, heads, head_dim)
    cos/sin: (seq, head_dim/2)
    """
    if torch is None:
        raise ImportError("Torch not available: RoPE requires torch tensors.")
    bsz = x.shape[:-3]
    seq, n_heads, d = x.shape[-3:]
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]
    # Expand cos/sin
    cos_t = torch.tensor(cos, device=x.device, dtype=x.dtype).unsqueeze(1)  # (seq,1,half)
    sin_t = torch.tensor(sin, device=x.device, dtype=x.dtype).unsqueeze(1)  # (seq,1,half)
    x_ro = torch.cat([x1 * cos_t - x2 * sin_t, x1 * sin_t + x2 * cos_t], dim=-1)
    return x_ro


def rope_cache(seq_len: int, head_dim: int, base: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    cos, sin = _build_sin_cos_cache(seq_len, head_dim, base)
    return cos.astype(np.float32), sin.astype(np.float32)

