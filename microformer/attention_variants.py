"""Attention variants: MHA, MQA, and GQA.

All three are drop-in replacements for ``CausalSelfAttention``:
same ``__init__(config)`` / ``forward(x, rope_freqs) -> Tensor`` contract.

Grouped-Query Attention (GQA) is the general form:
  - ``n_kv_heads == n_heads`` → standard Multi-Head Attention (MHA)
  - ``n_kv_heads == 1``       → Multi-Query Attention (MQA)
  - ``1 < n_kv_heads < n_heads`` → GQA

Reference: Ainslie et al., "GQA: Training Generalized Multi-Query
Transformer Models from Multi-Head Checkpoints", 2023.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from microformer.config import ModelConfig
from microformer.rope import apply_rope


class GroupedQueryAttention(nn.Module):
    """Causal self-attention with configurable KV head sharing.

    Query heads are split into ``n_heads // n_kv_heads`` groups.
    All query heads within a group share the same K and V projections,
    reducing KV-cache memory by a factor of ``n_heads / n_kv_heads``
    with minimal quality degradation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_head
        self.dropout_p = config.dropout

        # Number of query heads per KV group.
        self.n_groups = self.n_heads // self.n_kv_heads

        # Separate Q and KV projections so KV has fewer heads.
        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(
            config.d_model, self.n_kv_heads * self.d_head, bias=False
        )
        self.v_proj = nn.Linear(
            config.d_model, self.n_kv_heads * self.d_head, bias=False
        )
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

        causal_mask = torch.triu(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: Tensor, rope_freqs: Tensor) -> Tensor:
        """``(B, T, d_model) → (B, T, d_model)``."""
        B, T, C = x.shape

        # --- Projections ---
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # --- RoPE ---
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # --- Expand KV heads to match query heads ---
        # (B, n_kv_heads, T, d) → (B, n_heads, T, d)
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # --- Scaled dot-product attention ---
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.dropout(attn, self.dropout_p, self.training)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class MultiHeadAttention(GroupedQueryAttention):
    """Standard Multi-Head Attention (all heads have independent KV).

    Equivalent to ``GroupedQueryAttention`` with
    ``n_kv_heads == n_heads``.
    """

    def __init__(self, config: ModelConfig) -> None:
        # Override n_kv_heads to match n_heads regardless of config value.
        patched = ModelConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
            n_kv_heads=config.n_heads,
        )
        super().__init__(patched)


class MultiQueryAttention(GroupedQueryAttention):
    """Multi-Query Attention (single KV head shared across all Q heads).

    Equivalent to ``GroupedQueryAttention`` with ``n_kv_heads == 1``.
    Drastically reduces KV-cache memory at the cost of some quality.

    Reference: Shazeer, "Fast Transformer Decoding: One Write-Head is
    All You Need", 2019.
    """

    def __init__(self, config: ModelConfig) -> None:
        patched = ModelConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
            n_kv_heads=1,
        )
        super().__init__(patched)
