"""Evaluation configuration for MicroFormer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalConfig:
    """Immutable evaluation hyper-parameters."""

    # Path to validation data (plain text file).
    val_data_path: str = "data/val.txt"

    # Batching.
    eval_batch_size: int = 8
    max_seq_len: int = 1024

    # Limit the number of evaluation batches (0 = unlimited).
    max_eval_batches: int = 0
