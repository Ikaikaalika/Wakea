from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore


class AudioProjector(nn.Module):  # type: ignore[misc]
    """Project audio encoder features to LM hidden size.

    Expects: (B, T, d_in) -> (B, N, d_model)
    """

    def __init__(self, d_in: int, d_model: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(d_in, d_model)

    def forward(self, feats):
        x = feats[:, :: self.stride, :]
        return self.proj(x)

