"""Training loop for MicroFormer."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from microformer.config import ModelConfig
from microformer.model import MicroFormer
from microformer.train_config import TrainConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass for a single training-step log record
# ---------------------------------------------------------------------------


@dataclass
class StepMetrics:
    """Metrics emitted after each optimiser step."""

    step: int
    loss: float
    perplexity: float
    lr: float
    grad_norm: float
    tokens_per_sec: float


# ---------------------------------------------------------------------------
# Optimiser helpers
# ---------------------------------------------------------------------------


def configure_optimizer(
    model: MicroFormer,
    train_config: TrainConfig,
) -> torch.optim.AdamW:
    """Build AdamW with separate param groups for decayed / non-decayed params.

    Weight decay is applied to all 2-D+ parameters (linear weights,
    embeddings) but *not* to biases or normalisation parameters (1-D),
    following the GPT-2 / GPT-3 convention.
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2:
            # 1-D tensors: bias, RMSNorm weight → no decay.
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        param_groups,
        lr=train_config.lr,
        betas=train_config.betas,
        eps=train_config.eps,
    )


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def cosine_lr(
    step: int,
    train_config: TrainConfig,
) -> float:
    """Cosine learning-rate schedule with linear warmup.

    Returns a *multiplier* to be applied to the base LR:
      - Linear ramp from 0 → 1 over ``warmup_steps``.
      - Cosine decay from ``lr`` → ``min_lr`` over the remaining steps.
      - Clamp at ``min_lr / lr`` after ``max_steps``.
    """
    # 1) Linear warmup.
    if step < train_config.warmup_steps:
        return step / train_config.warmup_steps

    # 2) After max_steps → constant min_lr.
    if step >= train_config.max_steps:
        return train_config.min_lr / train_config.lr

    # 3) Cosine decay between warmup_steps and max_steps.
    decay_steps = train_config.max_steps - train_config.warmup_steps
    progress = (step - train_config.warmup_steps) / decay_steps
    # Decays from 1.0 → min_lr/lr following a half-cosine.
    min_ratio = train_config.min_lr / train_config.lr
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------


def clip_grad_norm(model: MicroFormer, max_norm: float) -> float:
    """Clip gradients by global norm and return the *unclipped* norm.

    We compute the norm ourselves rather than relying solely on
    ``torch.nn.utils.clip_grad_norm_`` so the raw (pre-clip) grad norm
    is available for logging.
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_LATEST_NAME = "latest.pt"
_BEST_NAME = "best.pt"


def save_checkpoint(
    path: Path,
    model: MicroFormer,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    step: int,
    best_loss: float,
) -> None:
    """Persist model + optimiser state so training can resume."""
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
    }
    torch.save(payload, path / _LATEST_NAME)
    logger.info("Saved checkpoint at step %d → %s", step, path / _LATEST_NAME)


def save_best_checkpoint(
    path: Path,
    model: MicroFormer,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    step: int,
    best_loss: float,
) -> None:
    """Save as the best checkpoint (called when validation loss improves)."""
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
    }
    torch.save(payload, path / _BEST_NAME)
    logger.info("New best checkpoint at step %d (loss=%.4f)", step, best_loss)


def load_checkpoint(
    path: Path,
    model: MicroFormer,
    optimizer: torch.optim.Optimizer | None = None,
    filename: str = _LATEST_NAME,
) -> dict:
    """Load a checkpoint and restore model (and optionally optimiser) state.

    Returns the full checkpoint dict so callers can read ``step`` and
    ``best_loss`` for resumption.
    """
    ckpt = torch.load(path / filename, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info("Resumed from step %d (%s)", ckpt["step"], path / filename)
    return ckpt


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Encapsulates the full training loop with gradient accumulation.

    Usage::

        trainer = Trainer(model, model_config, train_config)
        trainer.fit(train_iterator)
    """

    def __init__(
        self,
        model: MicroFormer,
        model_config: ModelConfig,
        train_config: TrainConfig,
        *,
        resume: bool = False,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.train_config = train_config

        self.optimizer = configure_optimizer(model, train_config)
        self.global_step = 0
        self.best_loss = float("inf")

        if resume:
            ckpt_path = train_config.checkpoint_path
            if (ckpt_path / _LATEST_NAME).exists():
                ckpt = load_checkpoint(ckpt_path, model, self.optimizer)
                self.global_step = ckpt["step"]
                self.best_loss = ckpt["best_loss"]

    # ---- LR helpers -------------------------------------------------------

    def _get_lr(self) -> float:
        """Current learning rate from the cosine schedule."""
        multiplier = cosine_lr(self.global_step, self.train_config)
        return self.train_config.lr * multiplier

    def _set_lr(self) -> None:
        """Apply the scheduled LR to all param groups."""
        lr = self._get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    # ---- Core training step -----------------------------------------------

    def train_step(self, batch_iter: Iterator[tuple[Tensor, Tensor]]) -> StepMetrics:
        """Execute one optimiser step with gradient accumulation.

        Consumes ``accumulation_steps`` micro-batches from *batch_iter*,
        accumulates gradients, clips, steps the optimiser, and returns
        metrics.
        """
        self.model.train()
        self._set_lr()

        accum = self.train_config.accumulation_steps
        total_loss = 0.0
        total_tokens = 0
        t0 = time.perf_counter()

        self.optimizer.zero_grad(set_to_none=True)

        for micro_step in range(accum):
            token_ids, targets = next(batch_iter)
            _, loss = self.model(token_ids, targets=targets)
            # Scale loss by accumulation steps so the effective gradient
            # is the mean over all micro-batches.
            scaled_loss = loss / accum
            scaled_loss.backward()
            total_loss += loss.item()
            total_tokens += token_ids.numel()

        # Gradient clipping (returns pre-clip norm).
        grad_norm = clip_grad_norm(self.model, self.train_config.grad_clip)

        self.optimizer.step()
        self.global_step += 1

        dt = time.perf_counter() - t0
        avg_loss = total_loss / accum

        return StepMetrics(
            step=self.global_step,
            loss=avg_loss,
            perplexity=math.exp(min(avg_loss, 20.0)),  # Cap to avoid overflow.
            lr=self._get_lr(),
            grad_norm=grad_norm,
            tokens_per_sec=total_tokens / dt if dt > 0 else 0.0,
        )

    # ---- Full training loop -----------------------------------------------

    def fit(
        self,
        batch_iter: Iterator[tuple[Tensor, Tensor]],
    ) -> None:
        """Run training from ``global_step`` to ``max_steps``.

        Args:
            batch_iter: Infinite iterator yielding ``(token_ids, targets)``
                micro-batches.
        """
        tc = self.train_config
        logger.info(
            "Starting training from step %d → %d", self.global_step, tc.max_steps
        )

        while self.global_step < tc.max_steps:
            metrics = self.train_step(batch_iter)

            # --- Logging ---
            if metrics.step % tc.log_every == 0:
                logger.info(
                    "step=%d | loss=%.4f | ppl=%.2f | lr=%.2e | "
                    "grad_norm=%.4f | tok/s=%.0f",
                    metrics.step,
                    metrics.loss,
                    metrics.perplexity,
                    metrics.lr,
                    metrics.grad_norm,
                    metrics.tokens_per_sec,
                )

            # --- Checkpointing ---
            if metrics.step % tc.save_every == 0:
                save_checkpoint(
                    tc.checkpoint_path,
                    self.model,
                    self.optimizer,
                    self.model_config,
                    tc,
                    metrics.step,
                    self.best_loss,
                )

            if metrics.loss < self.best_loss:
                self.best_loss = metrics.loss
                save_best_checkpoint(
                    tc.checkpoint_path,
                    self.model,
                    self.optimizer,
                    self.model_config,
                    tc,
                    metrics.step,
                    self.best_loss,
                )

        logger.info("Training complete at step %d.", self.global_step)
