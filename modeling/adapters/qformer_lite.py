from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore


class QFormerLite(nn.Module):  # type: ignore[misc]
    """Toy Q-Former-like adapter stub.

    For the scaffold, this just linearly projects features and prepends learnable queries.
    """

    def __init__(self, d_in: int, d_model: int, n_queries: int = 16):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.proj = nn.Linear(d_in, d_model)

    def forward(self, feats):
        B = feats.size(0)
        q = self.queries.expand(B, -1, -1)
        k = self.proj(feats)
        # In a real Q-Former, we'd attend queries over k/v. Here we just concat.
        return torch.cat([q, k], dim=1)

