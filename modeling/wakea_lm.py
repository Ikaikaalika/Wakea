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
from .tool_head import ToolHead


class WakeaForCausalLM(nn.Module):  # type: ignore[misc]
    def __init__(self, cfg: TransformerConfig, tool_names: Optional[list[str]] = None):
        super().__init__()
        self.backbone = WakeaTransformerLM(cfg)
        self.lm_head = LMHead(cfg.d_model, cfg.vocab_size)
        self.pad_id = 0
        self.tool_head: Optional[ToolHead] = None
        if tool_names:
            self.tool_head = ToolHead(cfg.d_model, tool_names)

    def forward(
        self,
        input_ids: "torch.Tensor",
        fused_mm: Optional["torch.Tensor"] = None,
        labels: Optional["torch.Tensor"] = None,
        tool_labels: Optional["torch.Tensor"] = None,
        tool_loss_weight: float = 0.0,
    ):
        h = self.backbone(input_ids=input_ids, fused_mm=fused_mm)
        logits = self.lm_head(h)
        loss = None
        losses = {}
        if labels is not None:
            # Shift for next-token prediction
            vocab = logits.size(-1)
            lm_loss = F.cross_entropy(
                logits.view(-1, vocab), labels.view(-1), ignore_index=self.pad_id
            )
            losses["lm_loss"] = lm_loss
            loss = lm_loss
        # Optional tool classification at last hidden state
        if self.tool_head is not None and tool_labels is not None:
            tool_logits = self.tool_head(h)
            tool_loss = F.cross_entropy(tool_logits, tool_labels)
            losses["tool_loss"] = tool_loss
            if loss is None:
                loss = tool_loss * tool_loss_weight
            else:
                loss = loss + tool_loss_weight * tool_loss
            losses["loss"] = loss
            return {"logits": logits, "tool_logits": tool_logits, **losses}
        if loss is not None:
            losses["loss"] = loss
            return {"logits": logits, **losses}
        return {"logits": logits}

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


def build_model_from_cfg(cfg_dict: dict, tool_names: Optional[list[str]] = None) -> WakeaForCausalLM:
    cfg = TransformerConfig(**cfg_dict)
    if torch is None:
        raise ImportError("PyTorch is required to build the model.")
    return WakeaForCausalLM(cfg, tool_names=tool_names)
