"""Shared fixtures for MicroFormer tests."""

from __future__ import annotations

import pytest
import torch

from microformer.config import ModelConfig
from microformer.rope import precompute_rope_frequencies

# Use a tiny config so tests run in seconds, not minutes.
_TINY_CONFIG = ModelConfig(
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=128,
    vocab_size=256,
    max_seq_len=32,
    dropout=0.0,  # Deterministic during testing.
)

BATCH = 2
SEQ_LEN = 16


@pytest.fixture
def config() -> ModelConfig:
    return _TINY_CONFIG


@pytest.fixture
def rope_freqs(config: ModelConfig) -> torch.Tensor:
    return precompute_rope_frequencies(config.d_head, config.max_seq_len)


@pytest.fixture
def dummy_hidden(config: ModelConfig) -> torch.Tensor:
    """Random hidden states: (batch, seq_len, d_model)."""
    return torch.randn(BATCH, SEQ_LEN, config.d_model)


@pytest.fixture
def dummy_token_ids(config: ModelConfig) -> torch.Tensor:
    """Random token ids: (batch, seq_len)."""
    return torch.randint(0, config.vocab_size, (BATCH, SEQ_LEN))
