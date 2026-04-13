"""Tests for MHA, MQA, and GQA attention variants."""

from __future__ import annotations

import torch

from microformer.attention_variants import (
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
)
from microformer.config import ModelConfig
from microformer.rope import precompute_rope_frequencies

# ---- Shared config ---------------------------------------------------------

_CFG = ModelConfig(
    d_model=64,
    n_heads=8,
    n_layers=2,
    d_ff=128,
    vocab_size=256,
    max_seq_len=32,
    dropout=0.0,
    n_kv_heads=4,  # GQA: 4 KV groups, 2 Q heads per group.
)

B, T = 2, 16


def _freqs() -> torch.Tensor:
    return precompute_rope_frequencies(_CFG.d_head, _CFG.max_seq_len)


# ===========================================================================
# Output shape correctness
# ===========================================================================


class TestOutputShapes:
    def test_mha_shape(self) -> None:
        attn = MultiHeadAttention(_CFG)
        x = torch.randn(B, T, _CFG.d_model)
        out = attn(x, _freqs())
        assert out.shape == (B, T, _CFG.d_model)

    def test_mqa_shape(self) -> None:
        attn = MultiQueryAttention(_CFG)
        x = torch.randn(B, T, _CFG.d_model)
        out = attn(x, _freqs())
        assert out.shape == (B, T, _CFG.d_model)

    def test_gqa_shape(self) -> None:
        attn = GroupedQueryAttention(_CFG)
        x = torch.randn(B, T, _CFG.d_model)
        out = attn(x, _freqs())
        assert out.shape == (B, T, _CFG.d_model)

    def test_all_variants_same_output_shape(self) -> None:
        """All three variants produce identically shaped output."""
        x = torch.randn(B, T, _CFG.d_model)
        freqs = _freqs()
        for cls in (MultiHeadAttention, MultiQueryAttention, GroupedQueryAttention):
            out = cls(_CFG)(x, freqs)
            assert out.shape == (B, T, _CFG.d_model)


# ===========================================================================
# MQA KV sharing
# ===========================================================================


class TestMQAKVSharing:
    def test_mqa_has_single_kv_head(self) -> None:
        """MQA should use exactly 1 KV head regardless of n_heads."""
        attn = MultiQueryAttention(_CFG)
        assert attn.n_kv_heads == 1

    def test_mqa_k_proj_is_one_head(self) -> None:
        """k_proj output dim should be d_head (1 head), not n_heads * d_head."""
        attn = MultiQueryAttention(_CFG)
        expected_kv_dim = 1 * _CFG.d_head  # Single KV head.
        assert attn.k_proj.out_features == expected_kv_dim
        assert attn.v_proj.out_features == expected_kv_dim

    def test_mqa_q_proj_unchanged(self) -> None:
        """Q projection should still have full n_heads * d_head output dim."""
        attn = MultiQueryAttention(_CFG)
        assert attn.q_proj.out_features == _CFG.n_heads * _CFG.d_head

    def test_mqa_fewer_kv_params_than_mha(self) -> None:
        """MQA should have fewer parameters than MHA due to shared KV."""
        mha = MultiHeadAttention(_CFG)
        mqa = MultiQueryAttention(_CFG)
        mha_params = sum(p.numel() for p in mha.parameters())
        mqa_params = sum(p.numel() for p in mqa.parameters())
        assert mqa_params < mha_params


# ===========================================================================
# GQA group assignment
# ===========================================================================


class TestGQAGroupAssignment:
    def test_gqa_n_kv_heads_from_config(self) -> None:
        """GQA should use n_kv_heads from config."""
        attn = GroupedQueryAttention(_CFG)
        assert attn.n_kv_heads == _CFG.n_kv_heads  # 4

    def test_gqa_n_groups(self) -> None:
        """n_groups = n_heads // n_kv_heads."""
        attn = GroupedQueryAttention(_CFG)
        assert attn.n_groups == _CFG.n_heads // _CFG.n_kv_heads  # 8//4 = 2

    def test_gqa_kv_proj_dims(self) -> None:
        """KV projections should have n_kv_heads * d_head output dim."""
        attn = GroupedQueryAttention(_CFG)
        expected = _CFG.n_kv_heads * _CFG.d_head
        assert attn.k_proj.out_features == expected
        assert attn.v_proj.out_features == expected

    def test_gqa_params_between_mha_and_mqa(self) -> None:
        """GQA parameter count should be between MQA and MHA."""
        mha = MultiHeadAttention(_CFG)
        gqa = GroupedQueryAttention(_CFG)
        mqa = MultiQueryAttention(_CFG)
        mha_p = sum(p.numel() for p in mha.parameters())
        gqa_p = sum(p.numel() for p in gqa.parameters())
        mqa_p = sum(p.numel() for p in mqa.parameters())
        assert mqa_p < gqa_p < mha_p

    def test_gqa_with_n_kv_heads_equals_n_heads_is_mha(self) -> None:
        """GQA with n_kv_heads == n_heads should behave like MHA."""
        cfg_mha = ModelConfig(
            d_model=64,
            n_heads=8,
            n_layers=2,
            d_ff=128,
            vocab_size=256,
            max_seq_len=32,
            dropout=0.0,
            n_kv_heads=8,
        )
        attn = GroupedQueryAttention(cfg_mha)
        assert attn.n_kv_heads == attn.n_heads
        assert attn.n_groups == 1
        # Q and K should have same output dim.
        assert attn.q_proj.out_features == attn.k_proj.out_features

    def test_invalid_n_kv_heads_raises(self) -> None:
        """n_heads must be divisible by n_kv_heads."""
        try:
            ModelConfig(
                d_model=64,
                n_heads=8,
                n_layers=2,
                d_ff=128,
                vocab_size=256,
                max_seq_len=32,
                n_kv_heads=3,  # 8 % 3 != 0
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ===========================================================================
# Causal masking still works
# ===========================================================================


class TestCausalMask:
    def test_gqa_causal_mask(self) -> None:
        """Perturbing a future token should not affect earlier positions."""
        attn = GroupedQueryAttention(_CFG)
        attn.eval()
        freqs = _freqs()

        x = torch.randn(1, 8, _CFG.d_model)
        out_orig = attn(x, freqs)

        x_pert = x.clone()
        x_pert[:, 7, :] += 100.0
        out_pert = attn(x_pert, freqs)

        # Positions 0-6 should be unchanged.
        torch.testing.assert_close(
            out_orig[:, :7, :], out_pert[:, :7, :], atol=1e-5, rtol=1e-5
        )

    def test_mqa_causal_mask(self) -> None:
        attn = MultiQueryAttention(_CFG)
        attn.eval()
        freqs = _freqs()

        x = torch.randn(1, 8, _CFG.d_model)
        out_orig = attn(x, freqs)

        x_pert = x.clone()
        x_pert[:, 7, :] += 100.0
        out_pert = attn(x_pert, freqs)

        torch.testing.assert_close(
            out_orig[:, :7, :], out_pert[:, :7, :], atol=1e-5, rtol=1e-5
        )


# ===========================================================================
# Drop-in compatibility with TransformerBlock
# ===========================================================================


class TestDropIn:
    def test_gqa_in_block(self) -> None:
        """GQA must work as a drop-in inside TransformerBlock."""
        from microformer.block import TransformerBlock

        block = TransformerBlock(_CFG)
        block.attn = GroupedQueryAttention(_CFG)
        x = torch.randn(B, T, _CFG.d_model)
        out = block(x, _freqs())
        assert out.shape == x.shape

    def test_mqa_in_block(self) -> None:
        from microformer.block import TransformerBlock

        block = TransformerBlock(_CFG)
        block.attn = MultiQueryAttention(_CFG)
        x = torch.randn(B, T, _CFG.d_model)
        out = block(x, _freqs())
        assert out.shape == x.shape
