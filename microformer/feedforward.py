"""Position-wise feed-forward network."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from microformer.config import ModelConfig


class FeedForward(nn.Module):
    """Two-layer MLP applied independently to each position.

    Architecture: Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model) → Dropout.

    Uses the tanh-approximation of GELU to match GPT-2's original
    implementation (Hendrycks & Gimpel, 2016).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """``(batch, seq_len, d_model) → (batch, seq_len, d_model)``."""
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return self.dropout(x)
