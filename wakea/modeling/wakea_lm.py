from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore
    F = None  # type: ignore

from .transformer import WakeaTransformerLM, TransformerConfig
from .lm_head import LMHead


class WakeaForCausalLM(nn.Module):  # type: ignore[misc]
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.backbone = WakeaTransformerLM(cfg)
        self.lm_head = LMHead(cfg.d_model, cfg.vocab_size)
        self.pad_id = 0

    def forward(
        self,
        input_ids: "torch.Tensor",
        fused_mm: Optional["torch.Tensor"] = None,
        labels: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        h = self.backbone(input_ids=input_ids, fused_mm=fused_mm)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            vocab = logits.size(-1)
            loss = F.cross_entropy(
                logits.view(-1, vocab), labels.view(-1), ignore_index=self.pad_id
            )
        return logits, loss

    @torch.no_grad()  # type: ignore[misc]
    def generate(self, input_ids: "torch.Tensor", max_new_tokens: int = 32, temperature: float = 0.0) -> "torch.Tensor":
        self.eval()
        x = input_ids
        for _ in range(max_new_tokens):
            h = self.backbone(x)
            logits = self.lm_head(h)
            next_token = logits[:, -1, :]
            if temperature and temperature > 0:
                probs = F.softmax(next_token / temperature, dim=-1)
                tok = torch.multinomial(probs, 1)
            else:
                tok = torch.argmax(next_token, dim=-1, keepdim=True)
            x = torch.cat([x, tok], dim=1)
        return x


def build_model_from_cfg(cfg_dict: dict) -> WakeaForCausalLM:
    cfg = TransformerConfig(**cfg_dict)
    if torch is None:
        raise ImportError("PyTorch is required to build the model.")
    return WakeaForCausalLM(cfg)

