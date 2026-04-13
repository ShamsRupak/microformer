"""MicroFormer: a GPT-style language model built from scratch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from microformer.block import RMSNorm, TransformerBlock
from microformer.config import ModelConfig
from microformer.rope import precompute_rope_frequencies


class MicroFormer(nn.Module):
    """Decoder-only transformer language model.

    Architecture:
        token_ids → Embedding → [TransformerBlock × n_layers] → RMSNorm → LM head

    No learned positional embedding — position is injected via RoPE inside
    each attention layer.  The LM head shares weights with the token
    embedding (weight tying) to reduce parameter count and improve
    generalisation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding (no positional embedding — RoPE handles position).
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        # Transformer blocks.
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final norm before the output projection.
        self.final_norm = RMSNorm(config.d_model)

        # LM head — weight-tied with token embedding.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying.

        # Pre-compute RoPE frequencies (registered as buffer for device transfer).
        rope_freqs = precompute_rope_frequencies(
            config.d_head, config.max_seq_len, theta=config.rope_theta
        )
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        # Initialise weights.
        self._init_weights()

    def _init_weights(self) -> None:
        """GPT-2-style weight initialisation.

        - Embeddings and most linear layers: N(0, 0.02)
        - Residual output projections (attn.out_proj, ffn.fc2): scaled by
          1/sqrt(2 * n_layers) to keep the residual stream variance stable
          as depth increases.
        """
        std = 0.02
        residual_scale = 1.0 / math.sqrt(2 * self.config.n_layers)

        for name, param in self.named_parameters():
            if param.dim() < 2:
                # Bias / norm parameters — leave at default (zeros / ones).
                continue
            nn.init.normal_(param, mean=0.0, std=std)
            # Scale residual-path projections.
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                param.data.mul_(residual_scale)

    def forward(
        self, token_ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            token_ids: ``(batch, seq_len)`` — integer token indices.
            targets: ``(batch, seq_len)`` — optional target token ids for
                computing cross-entropy loss.

        Returns:
            Tuple of ``(logits, loss)`` where *logits* has shape
            ``(batch, seq_len, vocab_size)`` and *loss* is a scalar tensor
            (or ``None`` if *targets* is not provided).
        """
        x = self.embed_dropout(self.token_emb(token_ids))

        for block in self.blocks:
            x = block(x, self.rope_freqs)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (B*T, vocab) vs (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Auto-regressive token generation.

        Args:
            prompt_ids: ``(1, prompt_len)`` — tokenised prompt.
            max_new_tokens: Number of tokens to generate.
            temperature: Softmax temperature (< 1 = sharper, > 1 = flatter).
            top_k: If set, only sample from the *top_k* most probable tokens.

        Returns:
            ``(1, prompt_len + max_new_tokens)`` — full sequence including
            the prompt.
        """
        self.eval()
        seq = prompt_ids

        for _ in range(max_new_tokens):
            # Crop to max_seq_len if the sequence has grown beyond it.
            context = seq[:, -self.config.max_seq_len :]

            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature  # Last position only.

            if top_k is not None:
                # Zero out everything below the top-k threshold.
                top_vals, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < top_vals[:, -1:]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, next_token], dim=1)

        return seq

    def count_parameters(self, exclude_embeddings: bool = False) -> int:
        """Return total number of trainable parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if exclude_embeddings:
            total -= self.token_emb.weight.numel()
        return total
