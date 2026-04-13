"""Training configuration for MicroFormer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    """Immutable training hyper-parameters.

    Separating training config from model config keeps checkpoints clean:
    model architecture is decoupled from the optimiser settings used to
    produce a particular set of weights.
    """

    # Optimiser.
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

    # Schedule.
    warmup_steps: int = 2000
    max_steps: int = 600_000

    # Batching.
    batch_size: int = 8
    max_seq_len: int = 1024
    accumulation_steps: int = 4

    # Regularisation.
    grad_clip: float = 1.0

    # Checkpointing.
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1000

    # Logging.
    log_every: int = 10

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir)
