"""Tests for CausalSelfAttention."""

from __future__ import annotations

import torch

from microformer.attention import CausalSelfAttention
from microformer.config import ModelConfig


class TestCausalSelfAttention:
    def test_output_shape(
        self,
        config: ModelConfig,
        dummy_hidden: torch.Tensor,
        rope_freqs: torch.Tensor,
    ) -> None:
        attn = CausalSelfAttention(config)
        out = attn(dummy_hidden, rope_freqs)
        assert out.shape == dummy_hidden.shape

    def test_causal_mask_blocks_future(self, config: ModelConfig) -> None:
        """Perturbing a future token should not affect earlier positions."""
        attn = CausalSelfAttention(config)
        attn.eval()

        from microformer.rope import precompute_rope_frequencies

        freqs = precompute_rope_frequencies(config.d_head, config.max_seq_len)

        torch.manual_seed(42)
        x = torch.randn(1, 8, config.d_model)

        out_original = attn(x, freqs)

        # Perturb position 7 (last token).
        x_perturbed = x.clone()
        x_perturbed[:, 7, :] += 100.0
        out_perturbed = attn(x_perturbed, freqs)

        # Positions 0-6 should be unchanged.
        torch.testing.assert_close(
            out_original[:, :7, :],
            out_perturbed[:, :7, :],
            atol=1e-5,
            rtol=1e-5,
        )

        # Position 7 should differ.
        assert not torch.allclose(
            out_original[:, 7, :], out_perturbed[:, 7, :], atol=1e-3
        )
