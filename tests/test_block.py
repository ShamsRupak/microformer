"""Tests for RMSNorm and TransformerBlock."""

from __future__ import annotations

import torch

from microformer.block import RMSNorm, TransformerBlock
from microformer.config import ModelConfig


class TestRMSNorm:
    def test_output_shape(self) -> None:
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        assert norm(x).shape == x.shape

    def test_normalises_rms_to_one(self) -> None:
        """After RMSNorm (with unit weight), RMS of output ≈ 1."""
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64) * 10  # Large input
        out = norm(x).float()
        rms = out.pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=0.1, rtol=0.1)

    def test_scale_invariance(self) -> None:
        """RMSNorm(α·x) ≈ RMSNorm(x) since it divides by RMS."""
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        out1 = norm(x)
        out2 = norm(x * 5.0)
        torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)


class TestTransformerBlock:
    def test_output_shape(
        self,
        config: ModelConfig,
        dummy_hidden: torch.Tensor,
        rope_freqs: torch.Tensor,
    ) -> None:
        block = TransformerBlock(config)
        out = block(dummy_hidden, rope_freqs)
        assert out.shape == dummy_hidden.shape

    def test_residual_connection(
        self,
        config: ModelConfig,
        rope_freqs: torch.Tensor,
    ) -> None:
        """Output should differ from input (residual + sub-layer contribution)."""
        block = TransformerBlock(config)
        x = torch.randn(1, 4, config.d_model)
        out = block(x, rope_freqs)
        assert not torch.allclose(x, out, atol=1e-6)
