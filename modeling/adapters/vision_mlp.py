from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore


class VisionMLPAdapter(nn.Module):  # type: ignore[misc]
    """Project frozen CLIP ViT features to LM hidden size.

    Expects input: (B, N, d_in) from a frozen encoder.
    Returns projected tokens: (B, N, d_model)
    """

    def __init__(self, d_in: int, d_model: int, n_tokens: int = 32):
        super().__init__()
        self.n_tokens = n_tokens
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, feats):
        # Optionally pool or select top-n tokens
        x = feats[:, : self.n_tokens, :]
        return self.mlp(x)

