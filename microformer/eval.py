"""Evaluation module for MicroFormer.

Provides:
- Perplexity calculation over a validation dataset.
- BLEU scoring via sacrebleu for generation quality.
- A structured ``EvalMetrics`` result dataclass.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass

import sacrebleu
import torch
from torch import Tensor

from microformer.eval_config import EvalConfig
from microformer.model import MicroFormer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvalMetrics:
    """Results from a single evaluation pass."""

    perplexity: float
    loss: float
    bleu: float
    tokens_evaluated: int
    eval_time_sec: float


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_loop(
    model: MicroFormer,
    batch_iter: Iterator[tuple[Tensor, Tensor]],
    eval_config: EvalConfig,
) -> EvalMetrics:
    """Run the model over validation batches and compute metrics.

    The model is set to eval mode and all computation happens under
    ``torch.no_grad()`` to save memory and avoid polluting gradient
    state.

    Args:
        model: The MicroFormer model to evaluate.
        batch_iter: Iterator yielding ``(token_ids, targets)`` batches.
        eval_config: Evaluation configuration.

    Returns:
        An :class:`EvalMetrics` instance.
    """
    model.eval()
    t0 = time.perf_counter()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for token_ids, targets in batch_iter:
        if (
            eval_config.max_eval_batches > 0
            and num_batches >= eval_config.max_eval_batches
        ):
            break

        logits, loss = model(token_ids, targets=targets)
        assert loss is not None

        batch_tokens = targets.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        num_batches += 1

    dt = time.perf_counter() - t0

    if total_tokens == 0:
        return EvalMetrics(
            perplexity=float("inf"),
            loss=float("inf"),
            bleu=0.0,
            tokens_evaluated=0,
            eval_time_sec=dt,
        )

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))

    logger.info(
        "eval | loss=%.4f | ppl=%.2f | tokens=%d | time=%.2fs",
        avg_loss,
        ppl,
        total_tokens,
        dt,
    )

    return EvalMetrics(
        perplexity=ppl,
        loss=avg_loss,
        bleu=0.0,  # BLEU computed separately via compute_bleu().
        tokens_evaluated=total_tokens,
        eval_time_sec=dt,
    )


# ---------------------------------------------------------------------------
# BLEU scoring
# ---------------------------------------------------------------------------


def compute_bleu(
    hypotheses: list[str],
    references: list[str],
) -> float:
    """Compute corpus-level BLEU score using sacrebleu.

    Args:
        hypotheses: Model-generated text strings.
        references: Ground-truth reference strings.

    Returns:
        BLEU score as a float (0–100 scale, per sacrebleu convention).
    """
    if not hypotheses or not references:
        return 0.0

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score
