"""Tests for the FeedForward module."""

from __future__ import annotations

import torch

from microformer.config import ModelConfig
from microformer.feedforward import FeedForward


class TestFeedForward:
    def test_output_shape(
        self, config: ModelConfig, dummy_hidden: torch.Tensor
    ) -> None:
        ffn = FeedForward(config)
        out = ffn(dummy_hidden)
        assert out.shape == dummy_hidden.shape

    def test_position_independence(self, config: ModelConfig) -> None:
        """FFN processes each position identically (no cross-position leak)."""
        ffn = FeedForward(config)
        ffn.eval()

        x = torch.randn(1, 4, config.d_model)
        full_out = ffn(x)

        # Process each position individually — should match.
        for t in range(4):
            single_out = ffn(x[:, t : t + 1, :])
            torch.testing.assert_close(
                full_out[:, t : t + 1, :], single_out, atol=1e-6, rtol=1e-6
            )
