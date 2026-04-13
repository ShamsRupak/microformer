"""Tests for the full MicroFormer model."""

from __future__ import annotations

import torch

from microformer.config import ModelConfig
from microformer.model import MicroFormer


class TestMicroFormer:
    def test_logits_shape(
        self, config: ModelConfig, dummy_token_ids: torch.Tensor
    ) -> None:
        model = MicroFormer(config)
        logits, loss = model(dummy_token_ids)
        B, T = dummy_token_ids.shape
        assert logits.shape == (B, T, config.vocab_size)
        assert loss is None

    def test_loss_computation(
        self, config: ModelConfig, dummy_token_ids: torch.Tensor
    ) -> None:
        model = MicroFormer(config)
        targets = torch.randint(0, config.vocab_size, dummy_token_ids.shape)
        logits, loss = model(dummy_token_ids, targets=targets)
        assert loss is not None
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Untrained model → high loss

    def test_generate_produces_tokens(self, config: ModelConfig) -> None:
        model = MicroFormer(config)
        prompt = torch.randint(0, config.vocab_size, (1, 4))
        generated = model.generate(prompt, max_new_tokens=8)
        assert generated.shape == (1, 12)  # 4 prompt + 8 new

    def test_generate_with_top_k(self, config: ModelConfig) -> None:
        model = MicroFormer(config)
        prompt = torch.randint(0, config.vocab_size, (1, 4))
        generated = model.generate(prompt, max_new_tokens=4, top_k=10)
        assert generated.shape == (1, 8)

    def test_weight_tying(self, config: ModelConfig) -> None:
        """LM head and token embedding should share the same weight tensor."""
        model = MicroFormer(config)
        assert model.lm_head.weight is model.token_emb.weight

    def test_parameter_count_gpt2_small(self) -> None:
        """GPT-2 small should have roughly 117M-124M parameters."""
        config = ModelConfig.from_name("gpt2-small")
        model = MicroFormer(config)
        n_params = model.count_parameters()
        # Weight tying means embedding params are counted once.
        # Expected: ~85M (excluding embedding) + 38M (embedding) ≈ 124M
        # Exact count varies due to RMSNorm vs LayerNorm, no bias, etc.
        assert 80_000_000 < n_params < 160_000_000, f"Got {n_params:,}"
