"""Multi-head causal self-attention with Rotary Positional Embeddings."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from microformer.config import ModelConfig
from microformer.rope import apply_rope


class CausalSelfAttention(nn.Module):
    """Scaled dot-product multi-head attention with a causal mask.

    Q, K, V are computed from a single fused linear projection and then
    split into heads — this is more efficient than three separate
    ``nn.Linear`` layers because it requires only one matmul + one bias
    add instead of three.

    RoPE is applied to Q and K *after* splitting into heads and *before*
    computing attention scores.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.dropout_p = config.dropout

        # Fused QKV projection: one big Linear → split into Q, K, V.
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Pre-compute and register the causal mask as a non-persistent buffer
        # so it travels with .to(device) but is not part of state_dict.
        causal_mask = torch.triu(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: Tensor, rope_freqs: Tensor) -> Tensor:
        """
        Args:
            x: ``(batch, seq_len, d_model)``
            rope_freqs: Pre-computed RoPE frequencies from
                :func:`~microformer.rope.precompute_rope_frequencies`.

        Returns:
            ``(batch, seq_len, d_model)``
        """
        B, T, C = x.shape

        # Fused QKV → split into three (B, T, d_model) tensors.
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)

        # Reshape into (B, n_heads, T, d_head) for multi-head attention.
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K (not V — positional info only in dot product).
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # Scaled dot-product attention.
        # scores: (B, n_heads, T, T)
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask: set future positions to -inf before softmax.
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        # Numerically stable softmax: subtract max before exp.
        # PyTorch's F.softmax already does this internally, but we call it
        # explicitly so the stability guarantee is visible in the code.
        attn = torch.softmax(scores, dim=-1)
        attn = torch.dropout(attn, self.dropout_p, self.training)

        # Weighted sum of values, then concatenate heads.
        out = torch.matmul(attn, v)  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(out))
