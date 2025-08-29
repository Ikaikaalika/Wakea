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

from .rope import rope_cache, apply_rope


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    max_seq_len: int = 2048
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):  # type: ignore[misc]
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)

    def _rope(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cos is None or self.cos.shape[0] < q.shape[-3]:
            cos, sin = rope_cache(self.cfg.max_seq_len, self.cfg.d_model // self.cfg.n_heads)
            self.cos = torch.tensor(cos, device=q.device, dtype=q.dtype)
            self.sin = torch.tensor(sin, device=q.device, dtype=q.dtype)
        q = apply_rope(q, self.cos[: q.shape[-3]], self.sin[: q.shape[-3]])
        k = apply_rope(k, self.cos[: k.shape[-3]], self.sin[: k.shape[-3]])
        return q, k

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        H = self.cfg.n_heads
        qkv = self.qkv(x).view(B, T, 3, H, C // H).transpose(1, 2)  # (B,3,T,H,hd)
        q, k, v = qkv[:, 0].transpose(1, 2), qkv[:, 1].transpose(1, 2), qkv[:, 2].transpose(1, 2)
        # q,k,v: (B,T,H,hd)
        q, k = self._rope(q, k)
        att = (q @ k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        # causal mask
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal, float("-inf"))
        if attn_mask is not None:
            att = att + attn_mask
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y


class TransformerBlock(nn.Module):  # type: ignore[misc]
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class WakeaTransformerLM(nn.Module):  # type: ignore[misc]
    """Tiny decoder-only LM with hooks for multimodal fused tokens.

    This is a teaching/reference scaffold; not optimized for performance.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        fused_mm: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = input_ids.size()
        assert T <= self.cfg.max_seq_len, "Sequence length exceeds model max_seq_len"
        x = self.tok_emb(input_ids) + self.pos_emb[:, :T, :]
        if fused_mm is not None:
            # naive fusion: prepend fused tokens then crop to max length
            x = torch.cat([fused_mm, x], dim=1)
            x = x[:, -self.cfg.max_seq_len :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.ln_f(x)
        return x

