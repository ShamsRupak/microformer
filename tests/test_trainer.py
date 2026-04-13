"""Tests for the MicroFormer training loop."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from microformer.config import ModelConfig
from microformer.model import MicroFormer
from microformer.train_config import TrainConfig
from microformer.trainer import (
    Trainer,
    clip_grad_norm,
    configure_optimizer,
    cosine_lr,
    load_checkpoint,
    save_checkpoint,
)

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

_TINY_TRAIN = TrainConfig(
    lr=6e-4,
    min_lr=6e-5,
    weight_decay=0.1,
    warmup_steps=100,
    max_steps=1000,
    batch_size=2,
    accumulation_steps=2,
    grad_clip=1.0,
    log_every=1,
    save_every=500,
)


@pytest.fixture
def tiny_model() -> MicroFormer:
    torch.manual_seed(0)
    return MicroFormer(_TINY_MODEL)


@pytest.fixture
def tiny_train_config() -> TrainConfig:
    return _TINY_TRAIN


def _make_batch(
    config: ModelConfig = _TINY_MODEL, batch_size: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a random (token_ids, targets) pair."""
    ids = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_len))
    return ids, targets


def _infinite_batches():
    """Yield random micro-batches forever."""
    while True:
        yield _make_batch()


# ===========================================================================
# Optimizer param groups
# ===========================================================================


class TestOptimizerParamGroups:
    def test_two_groups(
        self, tiny_model: MicroFormer, tiny_train_config: TrainConfig
    ) -> None:
        opt = configure_optimizer(tiny_model, tiny_train_config)
        assert len(opt.param_groups) == 2

    def test_decay_group_has_weight_decay(
        self, tiny_model: MicroFormer, tiny_train_config: TrainConfig
    ) -> None:
        opt = configure_optimizer(tiny_model, tiny_train_config)
        decay_group = opt.param_groups[0]
        assert decay_group["weight_decay"] == tiny_train_config.weight_decay

    def test_no_decay_group_has_zero_weight_decay(
        self, tiny_model: MicroFormer, tiny_train_config: TrainConfig
    ) -> None:
        opt = configure_optimizer(tiny_model, tiny_train_config)
        no_decay_group = opt.param_groups[1]
        assert no_decay_group["weight_decay"] == 0.0

    def test_all_params_assigned(
        self, tiny_model: MicroFormer, tiny_train_config: TrainConfig
    ) -> None:
        """Every trainable param should be in exactly one group."""
        opt = configure_optimizer(tiny_model, tiny_train_config)
        opt_param_ids = set()
        for group in opt.param_groups:
            for p in group["params"]:
                opt_param_ids.add(id(p))

        model_param_ids = {id(p) for p in tiny_model.parameters() if p.requires_grad}
        assert opt_param_ids == model_param_ids

    def test_norm_params_not_decayed(
        self, tiny_model: MicroFormer, tiny_train_config: TrainConfig
    ) -> None:
        """1-D params (norms, biases) should all be in the no-decay group."""
        opt = configure_optimizer(tiny_model, tiny_train_config)
        no_decay_params = opt.param_groups[1]["params"]
        for p in no_decay_params:
            assert p.dim() < 2, f"Expected 1-D param, got dim={p.dim()}"


# ===========================================================================
# LR schedule shape
# ===========================================================================


class TestCosineSchedule:
    def test_starts_at_zero(self, tiny_train_config: TrainConfig) -> None:
        assert cosine_lr(0, tiny_train_config) == 0.0

    def test_peak_at_warmup_end(self, tiny_train_config: TrainConfig) -> None:
        mult = cosine_lr(tiny_train_config.warmup_steps, tiny_train_config)
        # At the end of warmup the multiplier should be 1.0 (= peak LR).
        assert abs(mult - 1.0) < 1e-6

    def test_monotonic_warmup(self, tiny_train_config: TrainConfig) -> None:
        """LR should strictly increase during warmup."""
        prev = -1.0
        for step in range(tiny_train_config.warmup_steps + 1):
            lr = cosine_lr(step, tiny_train_config)
            assert lr >= prev, f"step {step}: {lr} < {prev}"
            prev = lr

    def test_monotonic_decay(self, tiny_train_config: TrainConfig) -> None:
        """LR should monotonically decrease after warmup."""
        prev = float("inf")
        for step in range(
            tiny_train_config.warmup_steps, tiny_train_config.max_steps + 1
        ):
            lr = cosine_lr(step, tiny_train_config)
            assert lr <= prev + 1e-9, f"step {step}: {lr} > {prev}"
            prev = lr

    def test_ends_at_min_lr_ratio(self, tiny_train_config: TrainConfig) -> None:
        mult = cosine_lr(tiny_train_config.max_steps, tiny_train_config)
        expected = tiny_train_config.min_lr / tiny_train_config.lr
        assert abs(mult - expected) < 1e-6

    def test_after_max_steps_stays_at_min(self, tiny_train_config: TrainConfig) -> None:
        mult = cosine_lr(tiny_train_config.max_steps + 5000, tiny_train_config)
        expected = tiny_train_config.min_lr / tiny_train_config.lr
        assert abs(mult - expected) < 1e-6


# ===========================================================================
# Gradient clipping
# ===========================================================================


class TestGradClipping:
    def test_clips_large_gradients(self, tiny_model: MicroFormer) -> None:
        """After clipping at 1.0, global grad norm should be ≤ 1.0."""
        ids, targets = _make_batch()
        _, loss = tiny_model(ids, targets=targets)
        loss.backward()

        # Artificially inflate gradients.
        for p in tiny_model.parameters():
            if p.grad is not None:
                p.grad.mul_(1000.0)

        clip_grad_norm(tiny_model, max_norm=1.0)

        # Post-clip norm should be ≤ 1.0.
        post_clip_norm = torch.nn.utils.clip_grad_norm_(
            tiny_model.parameters(), float("inf")
        ).item()
        assert post_clip_norm <= 1.0 + 1e-5

    def test_returns_preclip_norm(self, tiny_model: MicroFormer) -> None:
        ids, targets = _make_batch()
        _, loss = tiny_model(ids, targets=targets)
        loss.backward()

        raw_norm = clip_grad_norm(tiny_model, max_norm=float("inf"))
        assert raw_norm > 0.0


# ===========================================================================
# Checkpoint save / load round-trip
# ===========================================================================


class TestCheckpointing:
    def test_save_load_roundtrip(
        self,
        tiny_model: MicroFormer,
        tiny_train_config: TrainConfig,
        tmp_path: Path,
    ) -> None:
        """Model weights and optimizer state survive a save→load cycle."""
        opt = configure_optimizer(tiny_model, tiny_train_config)

        # Take one optimiser step so state dicts are non-trivial.
        ids, targets = _make_batch()
        _, loss = tiny_model(ids, targets=targets)
        loss.backward()
        opt.step()

        save_checkpoint(
            tmp_path,
            tiny_model,
            opt,
            _TINY_MODEL,
            tiny_train_config,
            step=42,
            best_loss=3.14,
        )

        # Create a fresh model + optimizer and load the checkpoint.
        model2 = MicroFormer(_TINY_MODEL)
        opt2 = configure_optimizer(model2, tiny_train_config)
        ckpt = load_checkpoint(tmp_path, model2, opt2)

        assert ckpt["step"] == 42
        assert abs(ckpt["best_loss"] - 3.14) < 1e-6

        # Model weights should match.
        for (n1, p1), (n2, p2) in zip(
            tiny_model.named_parameters(), model2.named_parameters()
        ):
            torch.testing.assert_close(p1, p2, msg=f"Mismatch in {n1}")

    def test_checkpoint_contains_configs(
        self,
        tiny_model: MicroFormer,
        tiny_train_config: TrainConfig,
        tmp_path: Path,
    ) -> None:
        opt = configure_optimizer(tiny_model, tiny_train_config)
        save_checkpoint(
            tmp_path, tiny_model, opt, _TINY_MODEL, tiny_train_config, 0, 9.9
        )
        ckpt = torch.load(tmp_path / "latest.pt", weights_only=False)
        assert "model_config" in ckpt
        assert "train_config" in ckpt
        assert ckpt["model_config"]["d_model"] == 64


# ===========================================================================
# Trainer integration
# ===========================================================================


class TestTrainerIntegration:
    def test_train_step_returns_metrics(
        self, tiny_model: MicroFormer, tiny_train_config: TrainConfig
    ) -> None:
        trainer = Trainer(tiny_model, _TINY_MODEL, tiny_train_config)
        it = _infinite_batches()
        metrics = trainer.train_step(it)

        assert metrics.step == 1
        assert metrics.loss > 0
        assert metrics.perplexity > 1.0
        assert metrics.lr > 0
        assert metrics.grad_norm >= 0
        assert metrics.tokens_per_sec > 0

    def test_grad_accumulation_consumes_n_batches(
        self, tiny_model: MicroFormer, tiny_train_config: TrainConfig
    ) -> None:
        """One train_step should consume accumulation_steps micro-batches."""
        batches = [_make_batch() for _ in range(10)]
        consumed = iter(batches)
        trainer = Trainer(tiny_model, _TINY_MODEL, tiny_train_config)
        trainer.train_step(consumed)
        # consumed should have advanced by accumulation_steps (2).
        remaining = list(consumed)
        assert len(remaining) == 10 - tiny_train_config.accumulation_steps

    def test_resume_from_checkpoint(
        self, tiny_train_config: TrainConfig, tmp_path: Path
    ) -> None:
        tc = TrainConfig(
            lr=tiny_train_config.lr,
            min_lr=tiny_train_config.min_lr,
            weight_decay=tiny_train_config.weight_decay,
            warmup_steps=tiny_train_config.warmup_steps,
            max_steps=tiny_train_config.max_steps,
            batch_size=tiny_train_config.batch_size,
            accumulation_steps=tiny_train_config.accumulation_steps,
            grad_clip=tiny_train_config.grad_clip,
            log_every=tiny_train_config.log_every,
            save_every=tiny_train_config.save_every,
            checkpoint_dir=str(tmp_path),
        )

        model1 = MicroFormer(_TINY_MODEL)
        trainer1 = Trainer(model1, _TINY_MODEL, tc)
        it = _infinite_batches()

        # Run 3 steps.
        for _ in range(3):
            trainer1.train_step(it)

        save_checkpoint(
            tc.checkpoint_path,
            model1,
            trainer1.optimizer,
            _TINY_MODEL,
            tc,
            step=trainer1.global_step,
            best_loss=trainer1.best_loss,
        )

        # Resume into a new Trainer.
        model2 = MicroFormer(_TINY_MODEL)
        trainer2 = Trainer(model2, _TINY_MODEL, tc, resume=True)
        assert trainer2.global_step == 3
