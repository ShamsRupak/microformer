"""KV-cache for efficient auto-regressive generation.

During auto-regressive decoding the model generates one token at a time.
Without a cache, every step re-computes the Key and Value projections for
the *entire* sequence so far — O(n²) in sequence length.  With a KV-cache
we store the K/V tensors from previous steps and only compute the
projections for the single new token — reducing each step to O(n).

This module provides an inference-only code-path that is intentionally
separate from the training ``forward()`` to keep the two concerns
decoupled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from microformer.attention import CausalSelfAttention
from microformer.block import TransformerBlock
from microformer.model import MicroFormer
from microformer.rope import apply_rope

# ---------------------------------------------------------------------------
# Cache data structure
# ---------------------------------------------------------------------------


@dataclass
class KVCache:
    """Per-layer key/value cache.

    ``keys[i]`` and ``values[i]`` hold the cached tensors for layer *i*,
    each with shape ``(batch, n_heads, seq_len_so_far, d_head)``.
    """

    keys: list[Tensor | None] = field(default_factory=list)
    values: list[Tensor | None] = field(default_factory=list)

    @classmethod
    def empty(cls, n_layers: int) -> KVCache:
        return cls(
            keys=[None] * n_layers,
            values=[None] * n_layers,
        )

    @property
    def seq_len(self) -> int:
        """Number of tokens currently cached."""
        if not self.keys or self.keys[0] is None:
            return 0
        return self.keys[0].shape[2]


# ---------------------------------------------------------------------------
# Cached forward helpers (inference only — no dropout)
# ---------------------------------------------------------------------------


def _cached_attn(
    attn: CausalSelfAttention,
    x: Tensor,
    rope_freqs: Tensor,
    cache_k: Tensor | None,
    cache_v: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run one attention layer with KV-cache.

    Args:
        attn: The ``CausalSelfAttention`` module.
        x: ``(B, T_new, d_model)`` — new token hidden states.
        rope_freqs: Pre-sliced to the correct absolute positions.
        cache_k, cache_v: Previous K/V or ``None`` on first call.

    Returns:
        ``(output, updated_k, updated_v)``
    """
    B, T, C = x.shape

    qkv = attn.qkv_proj(x)
    q, k, v = qkv.split(C, dim=-1)

    q = q.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
    k = k.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
    v = v.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)

    # RoPE — rope_freqs is already sliced to the right absolute positions,
    # so apply_rope's internal [:seq_len] gives the correct window.
    q = apply_rope(q, rope_freqs)
    k = apply_rope(k, rope_freqs)

    # Concatenate with cache.
    if cache_k is not None:
        k = torch.cat([cache_k, k], dim=2)
        v = torch.cat([cache_v, v], dim=2)

    # Scaled dot-product attention.
    T_total = k.shape[2]
    scale = 1.0 / math.sqrt(attn.d_head)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Causal mask — only needed during prefill (T > 1).
    # For single-token decode the new token can attend to everything.
    if T > 1:
        mask = torch.triu(
            torch.ones(T, T_total, dtype=torch.bool, device=x.device),
            diagonal=T_total - T + 1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v)
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    out = attn.out_proj(out)

    return out, k, v


def _cached_block(
    block: TransformerBlock,
    x: Tensor,
    rope_freqs: Tensor,
    cache_k: Tensor | None,
    cache_v: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run one transformer block with KV-cache."""
    attn_out, new_k, new_v = _cached_attn(
        block.attn, block.attn_norm(x), rope_freqs, cache_k, cache_v
    )
    x = x + attn_out
    x = x + block.ffn(block.ffn_norm(x))
    return x, new_k, new_v


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------


def _sample(
    logits: Tensor,
    temperature: float,
    top_k: int | None,
) -> Tensor:
    """Sample a single token from logits.

    When *temperature* is 0 the result is deterministic (greedy argmax).
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        top_vals, _ = torch.topk(logits, k, dim=-1)
        logits[logits < top_vals[:, -1:]] = float("-inf")

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_cached(
    model: MicroFormer,
    prompt_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> Tensor:
    """Auto-regressive generation with KV-cache.

    Args:
        model: A ``MicroFormer`` model (will be set to eval mode).
        prompt_ids: ``(batch, prompt_len)`` integer token ids.
        max_new_tokens: Tokens to generate.
        temperature: Softmax temperature.  0 → greedy.
        top_k: If set, restrict sampling to top-k tokens.

    Returns:
        ``(batch, prompt_len + generated_len)``
    """
    model.eval()
    rope_freqs = model.rope_freqs
    cache = KVCache.empty(model.config.n_layers)

    # ---- Prefill: process entire prompt, populate cache ------------------
    x = model.token_emb(prompt_ids)

    for i, block in enumerate(model.blocks):
        x, new_k, new_v = _cached_block(
            block, x, rope_freqs, cache.keys[i], cache.values[i]
        )
        cache.keys[i] = new_k
        cache.values[i] = new_v

    x = model.final_norm(x)
    logits = model.lm_head(x)[:, -1, :]

    # ---- Decode: one token at a time ------------------------------------
    generated: list[Tensor] = []

    for _ in range(max_new_tokens):
        next_token = _sample(logits, temperature, top_k)
        generated.append(next_token)

        pos = cache.seq_len
        x = model.token_emb(next_token)

        for i, block in enumerate(model.blocks):
            x, new_k, new_v = _cached_block(
                block, x, rope_freqs[pos:], cache.keys[i], cache.values[i]
            )
            cache.keys[i] = new_k
            cache.values[i] = new_v

        x = model.final_norm(x)
        logits = model.lm_head(x)[:, -1, :]

    if generated:
        return torch.cat([prompt_ids] + generated, dim=1)
    return prompt_ids
