"""Transformer block: the repeating unit of MicroFormer."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from microformer.attention import CausalSelfAttention
from microformer.config import ModelConfig
from microformer.feedforward import FeedForward


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Unlike LayerNorm, RMSNorm does not re-center activations (no mean
    subtraction).  This makes it ~10-15 % faster while achieving
    comparable training stability.  The formulation is:

        RMSNorm(x) = x / RMS(x) * γ

    where RMS(x) = sqrt(mean(x²) + ε).
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        # Compute in float32 for numerical stability, then cast back.
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return (self.weight * x).to(dtype)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm residual connections.

    Architecture:
        x → RMSNorm → CausalSelfAttention → + residual
          → RMSNorm → FeedForward           → + residual

    Pre-norm (normalise *before* the sub-layer) is more stable during
    training than post-norm and is the default in modern LLMs.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: Tensor, rope_freqs: Tensor) -> Tensor:
        """``(batch, seq_len, d_model) → (batch, seq_len, d_model)``."""
        x = x + self.attn(self.attn_norm(x), rope_freqs)
        x = x + self.ffn(self.ffn_norm(x))
        return x
