"""Rotary Positional Embeddings (RoPE).

Encodes absolute position information into Q and K vectors by rotating
pairs of dimensions at frequencies determined by their index.  This gives
the dot-product between Q_m and K_n a dependency on the *relative*
distance (m − n), without any learned positional parameters.

Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
Position Embedding", 2021.  https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

import torch
from torch import Tensor


def precompute_rope_frequencies(
    d_head: int,
    max_seq_len: int,
    theta: float = 10_000.0,
    device: torch.device | None = None,
) -> Tensor:
    """Pre-compute the complex-valued rotation frequencies for RoPE.

    Returns a tensor of shape ``(max_seq_len, d_head // 2)`` whose dtype is
    ``torch.complex64``.  Each entry ``freqs[t, i]`` equals
    ``exp(j · t · θ_i)`` where ``θ_i = theta^{-2i/d}``.

    We store these as complex exponentials so that applying the rotation
    later is a single complex multiply — no sin/cos bookkeeping required.
    """
    assert d_head % 2 == 0, "d_head must be even for RoPE"

    # θ_i = theta^{-2i / d_head} for i in [0, d_head/2)
    dim_pairs = torch.arange(0, d_head, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (dim_pairs / d_head))  # (d_head // 2,)

    # Outer product: positions × frequencies → angles
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)  # (max_seq_len, d_head // 2)

    # Convert to complex exponential: e^{jθ} = cos θ + j sin θ
    return torch.polar(torch.ones_like(angles), angles)  # (max_seq_len, d_head // 2)


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply rotary embeddings to a Q or K tensor.

    Args:
        x: Shape ``(batch, n_heads, seq_len, d_head)`` — the tensor to
           rotate.
        freqs: Shape ``(max_seq_len, d_head // 2)`` — pre-computed complex
               rotation factors from :func:`precompute_rope_frequencies`.

    Returns:
        Tensor of same shape as *x* with rotary embeddings applied.

    The trick: view consecutive pairs of the head dimension as real and
    imaginary parts of a complex number, multiply by the unit-magnitude
    rotation, then view back as real.  This is mathematically equivalent
    to the 2 × 2 rotation-matrix formulation but avoids allocating
    separate sin/cos buffers.
    """
    seq_len = x.shape[2]
    # Slice to actual sequence length and reshape for broadcasting:
    # (seq_len, d_head//2) → (1, 1, seq_len, d_head//2)
    freqs = freqs[:seq_len].unsqueeze(0).unsqueeze(0)

    # Pair up consecutive dims: (..., d_head) → (..., d_head//2, 2) → complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # Rotate and convert back to real
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(start_dim=-2)

    return x_rotated.to(x.dtype)
