"""Tests for Rotary Positional Embeddings."""

from __future__ import annotations

import torch

from microformer.rope import apply_rope, precompute_rope_frequencies


class TestPrecomputeFrequencies:
    def test_shape(self) -> None:
        freqs = precompute_rope_frequencies(d_head=64, max_seq_len=128)
        assert freqs.shape == (128, 32)  # (max_seq_len, d_head // 2)

    def test_dtype_is_complex(self) -> None:
        freqs = precompute_rope_frequencies(d_head=64, max_seq_len=128)
        assert freqs.is_complex()

    def test_unit_magnitude(self) -> None:
        """All rotation factors should have magnitude 1 (unit circle)."""
        freqs = precompute_rope_frequencies(d_head=64, max_seq_len=128)
        magnitudes = freqs.abs()
        torch.testing.assert_close(
            magnitudes, torch.ones_like(magnitudes), atol=1e-6, rtol=0
        )

    def test_position_zero_is_identity(self) -> None:
        """At position 0, all angles are 0, so freqs = 1+0j (identity rotation)."""
        freqs = precompute_rope_frequencies(d_head=64, max_seq_len=128)
        expected = torch.ones(32, dtype=torch.complex64)
        torch.testing.assert_close(freqs[0], expected, atol=1e-6, rtol=0)


class TestApplyRope:
    def test_output_shape(self) -> None:
        B, H, T, D = 2, 4, 16, 64
        x = torch.randn(B, H, T, D)
        freqs = precompute_rope_frequencies(d_head=D, max_seq_len=32)
        out = apply_rope(x, freqs)
        assert out.shape == x.shape

    def test_preserves_norm(self) -> None:
        """RoPE is a rotation — it should preserve vector norms."""
        B, H, T, D = 2, 4, 16, 64
        x = torch.randn(B, H, T, D)
        freqs = precompute_rope_frequencies(d_head=D, max_seq_len=32)
        out = apply_rope(x, freqs)

        x_norms = x.float().norm(dim=-1)
        out_norms = out.float().norm(dim=-1)
        torch.testing.assert_close(x_norms, out_norms, atol=1e-4, rtol=1e-4)

    def test_different_positions_give_different_outputs(self) -> None:
        """Vectors at different positions should be rotated differently."""
        D = 64
        x = torch.ones(1, 1, 2, D)  # Same vector at positions 0 and 1
        freqs = precompute_rope_frequencies(d_head=D, max_seq_len=4)
        out = apply_rope(x, freqs)
        # Position 0 and position 1 should differ
        assert not torch.allclose(out[0, 0, 0], out[0, 0, 1], atol=1e-6)
