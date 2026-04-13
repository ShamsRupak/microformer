#!/usr/bin/env python3
"""Ablation study comparing MHA vs MQA vs GQA attention variants.

Trains a tiny MicroFormer (4 layers, 256 dim) for 500 steps with each
attention variant and records perplexity, throughput, and peak memory.

Usage::

    python -m ablation.run_ablation          # run all three
    python -m ablation.run_ablation --variant mqa   # run one
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from microformer.attention_variants import (
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
)
from microformer.config import ModelConfig
from microformer.model import MicroFormer
from microformer.train_config import TrainConfig
from microformer.trainer import Trainer

# ---------------------------------------------------------------------------
# Ablation config
# ---------------------------------------------------------------------------

ABLATION_MODEL = ModelConfig(
    d_model=256,
    n_heads=8,
    n_layers=4,
    d_ff=512,
    vocab_size=512,
    max_seq_len=128,
    dropout=0.0,
)

ABLATION_TRAIN = TrainConfig(
    lr=3e-4,
    min_lr=3e-5,
    weight_decay=0.1,
    warmup_steps=50,
    max_steps=500,
    batch_size=4,
    max_seq_len=128,
    accumulation_steps=1,
    grad_clip=1.0,
    log_every=100,
    save_every=99999,  # No checkpointing needed for ablation.
)

ATTN_CLASSES: dict[str, type[nn.Module]] = {
    "mha": MultiHeadAttention,
    "mqa": MultiQueryAttention,
    "gqa": GroupedQueryAttention,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AblationResult:
    variant: str
    final_loss: float
    final_perplexity: float
    avg_tokens_per_sec: float
    peak_memory_mb: float
    total_params: int


# ---------------------------------------------------------------------------
# Model builder with swapped attention
# ---------------------------------------------------------------------------


def _build_model(config: ModelConfig, attn_cls: type[nn.Module]) -> MicroFormer:
    """Build a MicroFormer and replace each block's attention with *attn_cls*."""
    torch.manual_seed(42)
    model = MicroFormer(config)
    for block in model.blocks:
        block.attn = attn_cls(config)
    return model


# ---------------------------------------------------------------------------
# Random data iterator
# ---------------------------------------------------------------------------


def _random_batches(
    config: ModelConfig, train_config: TrainConfig
) -> list[tuple[Tensor, Tensor]]:
    """Pre-generate enough random batches to cover all training steps."""
    total = train_config.max_steps * train_config.accumulation_steps + 10
    batches: list[tuple[Tensor, Tensor]] = []
    for _ in range(total):
        ids = torch.randint(
            0, config.vocab_size, (train_config.batch_size, config.max_seq_len)
        )
        targets = torch.randint(
            0, config.vocab_size, (train_config.batch_size, config.max_seq_len)
        )
        batches.append((ids, targets))
    return batches


# ---------------------------------------------------------------------------
# Run a single variant
# ---------------------------------------------------------------------------


def run_variant(
    name: str,
    attn_cls: type[nn.Module],
    config: ModelConfig = ABLATION_MODEL,
    train_config: TrainConfig = ABLATION_TRAIN,
) -> AblationResult:
    """Train one attention variant and return metrics."""
    model = _build_model(config, attn_cls)

    # Reset peak memory tracking.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    batches = _random_batches(config, train_config)
    batch_iter = iter(batches)

    trainer = Trainer(model, config, train_config)

    tok_per_sec_samples: list[float] = []
    final_loss = float("inf")

    while trainer.global_step < train_config.max_steps:
        metrics = trainer.train_step(batch_iter)
        tok_per_sec_samples.append(metrics.tokens_per_sec)
        final_loss = metrics.loss

    final_ppl = math.exp(min(final_loss, 20.0))
    avg_tps = sum(tok_per_sec_samples) / len(tok_per_sec_samples)

    peak_mem = 0.0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return AblationResult(
        variant=name,
        final_loss=final_loss,
        final_perplexity=final_ppl,
        avg_tokens_per_sec=avg_tps,
        peak_memory_mb=peak_mem,
        total_params=model.count_parameters(),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> list[AblationResult]:
    parser = argparse.ArgumentParser(description="MHA/MQA/GQA ablation")
    parser.add_argument(
        "--variant",
        choices=["mha", "mqa", "gqa", "all"],
        default="all",
        help="Which variant(s) to run.",
    )
    args = parser.parse_args()

    variants = (
        list(ATTN_CLASSES.items())
        if args.variant == "all"
        else [(args.variant, ATTN_CLASSES[args.variant])]
    )

    results: list[AblationResult] = []
    for name, cls in variants:
        print(f"\n{'='*60}")
        print(f"  Running ablation: {name.upper()}")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        result = run_variant(name, cls)
        elapsed = time.perf_counter() - t0
        print(
            f"  Done in {elapsed:.1f}s — loss={result.final_loss:.4f} "
            f"ppl={result.final_perplexity:.2f} "
            f"tok/s={result.avg_tokens_per_sec:.0f}"
        )
        results.append(result)

    return results


if __name__ == "__main__":
    main()
