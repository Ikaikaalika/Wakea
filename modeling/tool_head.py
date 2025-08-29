from __future__ import annotations

from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore
    F = None  # type: ignore


class ToolHead(nn.Module):  # type: ignore[misc]
    """A minimal classifier over a small tool set.

    For richer serialization (function calling), project to a JSON token stream instead.
    """

    def __init__(self, d_model: int, tool_names: List[str]):
        super().__init__()
        self.tool_names = tool_names
        self.out = nn.Linear(d_model, len(tool_names))

    def forward(self, h):
        logits = self.out(h[:, -1, :])  # pool at last token
        return logits

    def decode(self, logits, temperature: float = 0.0) -> Optional[str]:
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            idx = torch.multinomial(probs, 1)
        else:
            idx = torch.argmax(logits, dim=-1, keepdim=True)
        tool = self.tool_names[idx.item()]
        return tool

