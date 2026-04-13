"""Tests for the MicroFormer evaluation module."""

from __future__ import annotations

import math

import torch

from microformer.config import ModelConfig
from microformer.eval import EvalMetrics, compute_bleu, eval_loop
from microformer.eval_config import EvalConfig
from microformer.model import MicroFormer

# ---- Tiny config shared across tests --------------------------------------

_TINY_MODEL = ModelConfig(
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=128,
    vocab_size=256,
    max_seq_len=32,
    dropout=0.0,
)

_EVAL_CONFIG = EvalConfig(
    eval_batch_size=2,
    max_seq_len=32,
    max_eval_batches=0,
)

_BATCH_SIZE = 2
_SEQ_LEN = 16


def _make_batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.randint(0, _TINY_MODEL.vocab_size, (_BATCH_SIZE, _SEQ_LEN))
    targets = torch.randint(0, _TINY_MODEL.vocab_size, (_BATCH_SIZE, _SEQ_LEN))
    return ids, targets


def _batch_list(n: int = 4) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [_make_batch() for _ in range(n)]


# ===========================================================================
# Perplexity matches manual calculation
# ===========================================================================


class TestPerplexity:
    def test_perplexity_matches_manual(self) -> None:
        """eval_loop perplexity should match exp(avg cross-entropy loss)."""
        torch.manual_seed(42)
        model = MicroFormer(_TINY_MODEL)
        model.eval()

        batches = _batch_list(n=3)

        # Manual calculation: weighted average of per-batch losses.
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for ids, targets in batches:
                logits, loss = model(ids, targets=targets)
                batch_tokens = targets.numel()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

        manual_avg_loss = total_loss / total_tokens
        manual_ppl = math.exp(manual_avg_loss)

        # eval_loop calculation.
        metrics = eval_loop(model, iter(batches), _EVAL_CONFIG)

        assert (
            abs(metrics.loss - manual_avg_loss) < 1e-5
        ), f"Loss mismatch: {metrics.loss} vs {manual_avg_loss}"
        assert (
            abs(metrics.perplexity - manual_ppl) < 0.1
        ), f"PPL mismatch: {metrics.perplexity} vs {manual_ppl}"

    def test_perplexity_is_exp_of_loss(self) -> None:
        torch.manual_seed(0)
        model = MicroFormer(_TINY_MODEL)
        metrics = eval_loop(model, iter(_batch_list(2)), _EVAL_CONFIG)
        expected_ppl = math.exp(metrics.loss)
        assert abs(metrics.perplexity - expected_ppl) < 0.1


# ===========================================================================
# Metrics dataclass fields
# ===========================================================================


class TestEvalMetricsFields:
    def test_all_fields_present(self) -> None:
        torch.manual_seed(0)
        model = MicroFormer(_TINY_MODEL)
        metrics = eval_loop(model, iter(_batch_list(2)), _EVAL_CONFIG)

        assert isinstance(metrics, EvalMetrics)
        assert isinstance(metrics.perplexity, float)
        assert isinstance(metrics.loss, float)
        assert isinstance(metrics.bleu, float)
        assert isinstance(metrics.tokens_evaluated, int)
        assert isinstance(metrics.eval_time_sec, float)

    def test_tokens_evaluated_count(self) -> None:
        torch.manual_seed(0)
        model = MicroFormer(_TINY_MODEL)
        n_batches = 3
        batches = _batch_list(n_batches)
        metrics = eval_loop(model, iter(batches), _EVAL_CONFIG)

        expected_tokens = n_batches * _BATCH_SIZE * _SEQ_LEN
        assert metrics.tokens_evaluated == expected_tokens

    def test_eval_time_is_positive(self) -> None:
        torch.manual_seed(0)
        model = MicroFormer(_TINY_MODEL)
        metrics = eval_loop(model, iter(_batch_list(1)), _EVAL_CONFIG)
        assert metrics.eval_time_sec > 0

    def test_max_eval_batches_limits_evaluation(self) -> None:
        torch.manual_seed(0)
        model = MicroFormer(_TINY_MODEL)
        cfg = EvalConfig(eval_batch_size=2, max_seq_len=32, max_eval_batches=2)
        batches = _batch_list(n=5)
        metrics = eval_loop(model, iter(batches), cfg)

        expected_tokens = 2 * _BATCH_SIZE * _SEQ_LEN
        assert metrics.tokens_evaluated == expected_tokens


# ===========================================================================
# no_grad is active during eval
# ===========================================================================


class TestNoGrad:
    def test_no_grad_active(self) -> None:
        """Model params should not accumulate gradients during eval."""
        torch.manual_seed(0)
        model = MicroFormer(_TINY_MODEL)

        # Ensure no existing grads.
        for p in model.parameters():
            p.grad = None

        eval_loop(model, iter(_batch_list(2)), _EVAL_CONFIG)

        # After eval_loop, no parameter should have a gradient.
        for name, p in model.named_parameters():
            assert p.grad is None, f"{name} has gradient after eval"

    def test_model_in_eval_mode_after(self) -> None:
        model = MicroFormer(_TINY_MODEL)
        model.train()  # Start in train mode.
        eval_loop(model, iter(_batch_list(1)), _EVAL_CONFIG)
        assert not model.training


# ===========================================================================
# Empty dataset
# ===========================================================================


class TestEmptyDataset:
    def test_empty_iterator(self) -> None:
        model = MicroFormer(_TINY_MODEL)
        metrics = eval_loop(model, iter([]), _EVAL_CONFIG)

        assert metrics.tokens_evaluated == 0
        assert metrics.loss == float("inf")
        assert metrics.perplexity == float("inf")
        assert metrics.bleu == 0.0

    def test_zero_max_eval_batches(self) -> None:
        """max_eval_batches=0 means unlimited, not zero batches."""
        torch.manual_seed(0)
        model = MicroFormer(_TINY_MODEL)
        cfg = EvalConfig(eval_batch_size=2, max_seq_len=32, max_eval_batches=0)
        batches = _batch_list(n=3)
        metrics = eval_loop(model, iter(batches), cfg)
        assert metrics.tokens_evaluated > 0


# ===========================================================================
# BLEU scoring
# ===========================================================================


class TestBLEU:
    def test_perfect_bleu(self) -> None:
        """Identical hypothesis and reference should give BLEU ≈ 100."""
        refs = ["the cat sat on the mat"]
        hyps = ["the cat sat on the mat"]
        score = compute_bleu(hyps, refs)
        assert score > 99.0

    def test_zero_bleu_on_mismatch(self) -> None:
        refs = ["the cat sat on the mat"]
        hyps = ["completely different sentence here"]
        score = compute_bleu(hyps, refs)
        assert score < 20.0

    def test_empty_inputs(self) -> None:
        assert compute_bleu([], []) == 0.0
        assert compute_bleu([], ["ref"]) == 0.0
        assert compute_bleu(["hyp"], []) == 0.0
